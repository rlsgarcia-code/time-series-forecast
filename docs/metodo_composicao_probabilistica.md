# Método de composição probabilística

## 1. Finalidade e status

Este documento é a especificação estatística da composição probabilística implementada pelo
`ForecastEngine`. Ele explica **o que é estimado, por que cada transformação existe, quais
hipóteses são feitas e como validar o resultado**.

Documentos complementares:

- [`composition_mechanics.md`](composition_mechanics.md): correspondência entre estas etapas e a
  implementação;
- [`engine_api.md`](engine_api.md): contratos e chamadas públicas;
- [Notebook 15](../notebooks/15_composicao_probabilistica_hierarquica.ipynb): derivação executável
  para uma saída genérica;
- [Notebook 16](../notebooks/16_ciclo_hierarquico_dict.ipynb): aplicação a folhas, grupos e total.

A versão atual estima distribuições marginais separadamente, recupera dependência contemporânea
dos erros probabilísticos e propaga trajetórias conjuntas por um grafo. Não é um modelo estrutural
multivariado e não reconcilia estatisticamente agregados com folhas.

## 2. Problema estatístico

Considere entradas aleatórias previstas

$$X_{1,t+h},\ldots,X_{p,t+h},$$

entradas determinísticas ou de cenário $K_{t+h}$ e uma função, possivelmente não linear,

$$Y_{t+h}=g(X_{1,t+h},\ldots,X_{p,t+h},K_{t+h}).$$

O objetivo não é somente estimar $E[Y_{t+h}]$. O motor procura uma aproximação da distribuição
condicional completa

$$P(Y_{t+h}\le y\mid\mathcal I_t),$$

incluindo quantis, intervalos, probabilidade de limiar e risco de trajetória. Isso exige conhecer
as distribuições marginais dos $X_i$ e sua dependência conjunta. Em geral,

$$E[g(X_1,\ldots,X_p)]\ne g(E[X_1],\ldots,E[X_p]).$$

Para um produto,

$$E[X_1X_2]=E[X_1]E[X_2]+\operatorname{Cov}(X_1,X_2).$$

Combinar apenas previsões pontuais descartaria esse termo e toda a assimetria introduzida por
$g$.

## 3. Visão completa do método

```text
históricos e vintages
        │
        ▼
caracterização de cada série
        │ CandidatePolicy
        ▼
candidatos marginais ──► backtest rolling-origin ──► pré-seleção marginal
                                                       │ CompositionPolicy
                                                       ▼
                                              combinações limitadas
                                                       │
                                                       ▼
CDFs preditivas ──► PITs sincronizados ──► dependência residual
        │                                      │
        ▼                                      ▼
caminhos marginais ───────────────► caminhos conjuntos
                                            │
                                            ▼
                                  avaliação caminho a caminho de g
                                            │
                                            ▼
                                  distribuição empírica de Y
                                            │
                                            ▼
                                  CRPS, cobertura e seleção
```

O treinamento termina em um `CompositionRun`. O forecast operacional posterior reconstrói os
caminhos a partir desse manifesto e devolve um `ComposedForecast`.

## 4. Entradas e grafo

Um único tipo imutável, `ForecastVariable`, representa folhas e nós derivados. As folhas possuem
quatro modos:

| Modo | Interpretação | Incerteza no motor |
|---|---|---|
| `forecast` | série histórica modelada | distribuição preditiva marginal |
| `known` | escalar ou série futura conhecida | zero, condicional ao plano conhecido |
| `constant` | escalar repetido no horizonte | zero |
| `scenario` | valor determinado por uma chave de cenário | zero dentro de cada cenário |

Operadores `+`, `-`, `*`, `/`, negação e potência escalar criam novos nós. Uma função privada pode
ser registrada por `transform(name, function, *inputs, version=...)`. O grafo é avaliado em ordem
topológica e nós compartilhados são calculados uma vez.

O método condiciona nos valores `known` e `scenario`; ele não atribui incerteza a eles. Se um plano
futuro também for incerto, deve ser modelado como `forecast` ou expandido em cenários explícitos.

## 5. Modelos marginais e CDF preditiva

Em uma origem $t$, cada modelo recebe somente $\mathcal I_t$ e produz

$$\widehat F_{i,t,h}(x)
=P(X_{i,t+h}\le x\mid\mathcal I_t,\widehat\theta_{i,t}).$$

O chapéu registra parâmetros estimados e, quando aplicável, aproximação de Monte Carlo.

### 5.1 Relação entre família e distribuição

| Família | Formulação e hipótese | Distribuição entregue | Construção de $\widehat F$ |
|---|---|---|---|
| AR(1) | $X_t=c+\phi X_{t-1}+\varepsilon_t$, $\varepsilon_t\sim N(0,\sigma^2)$ | `GaussianDistribution` | CDF normal analítica com média e covariância recursivas |
| ARIMA | dinâmica ARIMA em estado-espaço; perturbações do modelo ajustado | `EmpiricalDistribution` | simulação do `ARIMAResults` e CDF empírica |
| nível local/Kalman | ruídos gaussianos de estado e observação | `GaussianDistribution` | recursões de Kalman e CDF normal analítica |
| GARCH(1,1) | variância condicional e choque padronizado normal | `EmpiricalDistribution` | simulação recursiva e CDF empírica |
| Markov switching | normal condicional ao regime oculto | `GaussianMixtureDistribution` | mistura das CDFs ponderadas pelas probabilidades dos regimes |

Para as distribuições gaussianas,

$$\widehat F_h(x)=\Phi\left(\frac{x-\widehat\mu_h}{\widehat\sigma_h}\right).$$

Para $K$ regimes,

$$\widehat F_h(x)=\sum_{k=1}^{K}\widehat p_{k,h}
\Phi\left(\frac{x-\widehat\mu_k}{\widehat\sigma_k}\right).$$

Com $M$ trajetórias simuladas,

$$\widehat F_h(x)=\frac1M\sum_{m=1}^{M}
\mathbf 1\{X_h^{(m)}\le x\}.$$

O motor não converte todas as marginais em normais. Ele chama `distribution.cdf(actual)` no
objeto produzido pelo adaptador selecionado.

### 5.2 Papel de `CandidatePolicy`

`CandidatePolicy` recebe a caracterização de **uma série** e propõe uma lista limitada e
explicável de `ModelSpec`. Suas decisões usam evidências como estacionariedade, ACF/PACF, memória
serial, heterocedasticidade e indícios de mudança.

Ela responde:

> Quais hipóteses marginais merecem competir para esta entrada?

Ela não ajusta os modelos, não conhece $g$ e não escolhe o vencedor. A seleção é empírica, por
backtest fora da amostra.

## 6. Backtest causal e vintages

O backtest é rolling-origin. Para cada origem $o$:

1. o modelo é ajustado com observações até o cutoff $c_o$;
2. emite uma distribuição para timestamps posteriores;
3. os realizados são comparados à distribuição que existia antes de observá-los;
4. a origem avança e o procedimento se repete.

Entradas previstas precisam ter frequência, timezone e cutoff comuns. Históricos podem começar em
datas diferentes, mas somente a interseção temporal participa do backtest conjunto.

Uma entrada `known` temporal precisa responder: **qual valor era conhecido naquela origem?** O
formato longo de vintages registra `as_of`, `target`, `value`. Para um cutoff $c_o$, usa-se a
última versão com `as_of <= c_o`. Sem vintages, a análise histórica seria contaminada por
informação futura. O motor bloqueia promoção causal, salvo quando a análise não causal é
explicitamente autorizada e marcada.

## 7. PIT: erro probabilístico padronizado

Quando o realizado $x_{i,t}$ se torna conhecido, calcula-se

$$u_{i,t}=\widehat F_{i,t}(x_{i,t}).$$

$u$ é a posição do realizado na distribuição emitida anteriormente. Um valor perto de $0{,}5$
indica realização central; valores próximos de 0 ou 1 indicam surpresa de cauda.

Para uma CDF preditiva contínua corretamente especificada, o teorema do Probability Integral
Transform implica

$$U_{i,t}\sim U(0,1).$$

Isso é um diagnóstico verificável de calibração, não uma suposição automaticamente satisfeita.
Concentração nas bordas sugere distribuição estreita demais; concentração central sugere
distribuição larga demais; assimetria sugere viés.

O motor converte o PIT para uma escala comum:

$$z_{i,t}=\Phi^{-1}(\operatorname{clip}(u_{i,t},10^{-6},1-10^{-6})).$$

Essa normalização não afirma que $X_i$ seja normal. A hipótese marginal continua sendo a do
adaptador. Para CDF empírica, os PITs ficam numa grade de passo $1/M$ e carregam erro de Monte
Carlo.

## 8. Dependência residual sincronizada

### 8.1 Matriz de erros

Os escores são alinhados pelo timestamp-alvo:

$$Z=
\begin{bmatrix}
z_{1,t_1}&\cdots&z_{p,t_1}\\
\vdots&&\vdots\\
z_{1,t_n}&\cdots&z_{p,t_n}
\end{bmatrix}.$$

Cada linha representa o mesmo evento temporal e cada coluna uma entrada. A correlação mede
surpresas contemporâneas depois que cada dinâmica marginal foi removida. Não é a correlação dos
níveis, que poderia refletir somente tendência ou escala comum.

A versão atual usa erros um passo à frente para estimar essa associação contemporânea. A
dependência temporal entre horizontes permanece dentro da trajetória produzida por cada modelo
marginal.

### 8.2 Shrinkage e reparo

Com $n$ linhas completas e $p$ variáveis, seja $S$ a correlação amostral. No modo automático,

$$\lambda=\min\left(1,\frac{p+1}{n-1}\right),$$

e

$$R=(1-\lambda)S+\lambda I.$$

Autovalores menores que $10^{-8}$ são truncados; em seguida a matriz é renormalizada para diagonal
unitária. Isso reduz instabilidade em amostras curtas e garante uma matriz utilizável na simulação.

Com menos de `min_paired_errors`, usa-se $R=I$. O diagnóstico registra `fallback=True` e a causa.
Isso significa **evidência insuficiente**, não prova de independência.

### 8.3 Hipótese de cópula

O motor sorteia vetores latentes normais com correlação $R$ e usa seus ranks para reordenar os
caminhos marginais. Para cada variável, a severidade de um caminho é a média, entre horizontes,
dos valores centralizados e padronizados daquela trajetória.

Somente as linhas das matrizes são permutadas. Portanto, permanecem inalterados:

- os valores e quantis marginais em cada horizonte;
- a trajetória temporal completa de cada caminho;
- a distribuição produzida por cada modelo marginal.

O que muda é qual caminho de uma variável acompanha qual caminho das demais. Essa construção é
uma aproximação por ranks inspirada em cópula gaussiana; ela não garante dependência assimétrica ou
dependência de cauda correta.

## 9. Propagação pelo grafo

Depois do alinhamento, todas as entradas possuem forma `(M, H)`. Entradas determinísticas têm seu
vetor repetido nas $M$ linhas. Para cada caminho $m$ e horizonte $h$:

$$Y_h^{(m)}=g(X_{1,h}^{(m)},\ldots,X_{p,h}^{(m)},K_h).$$

A distribuição da saída é a medida empírica

$$\widehat P(Y_h\le y)=\frac1M\sum_{m=1}^{M}
\mathbf1\{Y_h^{(m)}\le y\}.$$

Média, variância, CDF e quantis são calculados diretamente dessa matriz. Divisão por zero e saídas
não finitas são rejeitadas. Como folhas, grupos e total são avaliados nas mesmas linhas, identidades
algébricas como `total = group_a + group_b` valem caminho a caminho, não apenas em média.

## 10. Seleção do conjunto

### 10.1 Papel de `CompositionPolicy`

`CompositionPolicy` governa o experimento conjunto:

- `preselect_top_k`: quantos candidatos marginais reter por entrada;
- `max_combinations`: orçamento para o produto cartesiano;
- `dependency`: cópula residual ou independência;
- `min_paired_errors` e `shrinkage`: regularização da dependência;
- `n_backtest_paths`: resolução Monte Carlo do backtest;
- `coverage_range`: guardrail de cobertura;
- `output_weights`: importância relativa das saídas;
- `allow_noncausal_known`: autorização explícita de análise não causal.

Ela responde:

> Como limitar, combinar e julgar os modelos marginais pela qualidade das saídas de interesse?

Ela não é um modelo estatístico e não cria candidatos.

### 10.2 O que `backtest_expression()` executa

1. normaliza as saídas e coleta as folhas do grafo;
2. valida alinhamento temporal e causalidade;
3. caracteriza cada entrada e aplica `CandidatePolicy`;
4. executa backtest marginal e retém os melhores candidatos por entrada;
5. prioriza combinações pela soma dos CRPS marginais normalizados;
6. limita o conjunto a `max_combinations`;
7. em cada origem, estima dependência apenas com erros já observáveis;
8. gera caminhos, materializa entradas determinísticas e avalia $g$;
9. calcula métricas por saída e horizonte;
10. escolhe a combinação conjunta, ajusta os vencedores no histórico completo e cria um
    `CompositionRun`.

O método não produz o forecast futuro.

### 10.3 Critério de decisão

Para uma saída $j$, o CRPS amostral compara toda a distribuição simulada ao realizado:

$$\operatorname{CRPS}(F,y)=E|X-y|-\frac12E|X-X'|.$$

O score é normalizado pela escala da saída, preferencialmente a média das diferenças absolutas dos
realizados e, como fallback, seu desvio-padrão. Com pesos $w_j$:

$$L=\frac{\sum_j w_j\operatorname{nCRPS}_j}{\sum_jw_j}.$$

A combinação elegível com menor $L$ vence. O guardrail de cobertura impede que uma distribuição
excessivamente estreita vença apenas por concentração. MAE, RMSE, largura e cobertura permanecem
no placar para auditoria.

## 11. Artefatos e ciclo operacional

### 11.1 `CompositionRun`

É o manifesto congelado do treinamento. Contém:

- hashes das expressões;
- IDs e specs dos modelos marginais escolhidos;
- revisões das entradas;
- placar das combinações e métricas vencedoras;
- `DependencyDiagnostics`;
- indicador causal.

Não contém caminhos futuros, distribuição final, código de função privada, árvore serializada da
fórmula ou futuros de entradas `known`.

### 11.2 `compose()` e `ComposedForecast`

`compose()` resolve um `CompositionRun` por objeto, ID ou alias, verifica hashes e revisões,
reconstrói as distribuições marginais, amostra caminhos, aplica a dependência, avalia o grafo e
devolve um `ComposedForecast`.

O `ComposedForecast` contém targets, trajetórias e `EmpiricalDistribution` por saída, modelos
usados, revisões, cenário e diagnóstico de dependência. Ele pode ser passado diretamente a
`assess_risk()`.

## 12. Validação recomendada

Uma composição não deve ser validada somente pelo score agregado. Recomenda-se examinar:

| Camada | Evidência |
|---|---|
| Marginal | CRPS, cobertura, largura, MAE/RMSE, resíduos, PIT por entrada |
| Calibração | histograma e uniformidade do PIT, viés e autocorrelação dos PITs |
| Dependência | pares disponíveis, shrinkage, fallback, correlação estimada e realizada |
| Composição | CRPS e cobertura por saída e horizonte |
| Monte Carlo | estabilidade ao aumentar `n_paths` e variar seeds |
| Coerência | igualdade caminho a caminho entre folhas, grupos e total |
| Estabilidade | frequência de troca da combinação vencedora entre janelas |
| Causalidade | vintages, cutoffs, revisões e ausência de informação futura |

Para amostras curtas, a incerteza da estimativa de dependência deve ser tratada como limitação
central. Shrinkage reduz variância, mas não cria informação.

## 13. Hipóteses e limitações

1. **Especificação marginal:** uma CDF errada gera PITs mal calibrados e contamina a dependência.
2. **Estabilidade:** parâmetros e associação residual são tratados como suficientemente estáveis
   na janela usada.
3. **Cópula aproximada:** ranks gaussianos não capturam necessariamente assimetria ou caudas.
4. **Dependência contemporânea:** a versão atual estima associação com erros um passo à frente;
   não ajusta uma cópula dinâmica multihorizonte.
5. **Parâmetros plug-in:** a incerteza dos parâmetros não é integrada explicitamente, salvo o que
   a simulação do adaptador já propaga.
6. **Determinísticos condicionais:** `known` e `scenario` têm variância zero dentro da execução.
7. **Sem feedback agregado:** totais não corrigem médias das folhas.
8. **Sem reconciliação estatística:** a coerência decorre do mesmo grafo e caminhos, não de MinT.
9. **Sem modelo causal:** correlação de PITs não identifica relações causais.
10. **Risco de seleção:** muitas combinações em pouco backtest podem sobreajustar; por isso existem
    os orçamentos das políticas.

## 14. Extensões previstas

- cópulas t, empíricas ou com dependência de cauda;
- dependência dinâmica e específica por horizonte;
- modelos globais ou multivariados para compartilhar informação;
- reconciliação estatística com informação dos níveis agregados;
- propagação de incerteza de parâmetros e de entradas hoje determinísticas;
- monitoramento sequencial da distribuição composta;
- reestimação acionada por drift marginal ou de dependência.

## 15. Referências

- Diebold, F. X., Gunther, T. A. e Tay, A. S. (1998), [*Evaluating Density Forecasts with
  Applications to Financial Risk Management*](https://doi.org/10.2307/2527342).
- Gneiting, T., Balabdaoui, F. e Raftery, A. E. (2007), [*Probabilistic Forecasts, Calibration
  and Sharpness*](https://doi.org/10.1111/j.1467-9868.2007.00587.x).
- Gneiting, T. e Raftery, A. E. (2007), [*Strictly Proper Scoring Rules, Prediction, and
  Estimation*](https://doi.org/10.1198/016214506000001437).
- Statsmodels, [`ARIMAResults.simulate`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMAResults.simulate.html).
- Código normativo da versão atual: [`composition.py`](../src/time_series_lab/composition.py),
  [`contracts.py`](../src/time_series_lab/contracts.py) e
  [`adapters.py`](../src/time_series_lab/adapters.py).
