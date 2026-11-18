# Referência técnica — composição probabilística

Esta referência descreve a implementação usada pelo Notebook 15. A API pública está em
[`engine_api.md`](engine_api.md); o código principal está em
[`composition.py`](../src/time_series_lab/composition.py) e a coordenação pública em
[`engine.py`](../src/time_series_lab/engine.py).

## 1. Objetos e responsabilidades

| Objeto | Entra com | Produz / conserva |
|---|---|---|
| `CandidatePolicy` | `SeriesProfile` de uma entrada | lista explicável e limitada de `ModelSpec`; não ajusta nem escolhe vencedor |
| `CompositionPolicy` | limites e regras do experimento conjunto | `top_k`, orçamento de combinações, caminhos, dependência, shrinkage, cobertura, pesos e regra causal |
| `ModelRun` | série, cutoff e `ModelSpec` já ajustado | estado estimado e identidade reproduzível do modelo marginal |
| `PredictiveDistribution` | estado marginal e horizonte | média, variância, CDF, quantis e amostras `(M,H)` |
| `DependencyDiagnostics` | erros PIT sincronizados | correlação, shrinkage, quantidade de pares e eventual fallback independente |
| `CompositionRun` | seleção conjunta concluída | hashes das expressões, IDs/specs dos modelos, revisões, scores, dependência e flag causal |
| `EmpiricalDistribution` | matriz final de caminhos | distribuição da saída derivada |

`CandidatePolicy` responde **quais modelos merecem competir em cada entrada**.
`CompositionPolicy` responde **como limitar, combinar e avaliar essa competição no grafo**.
Nenhuma das duas é um modelo estatístico.

## 2. O que `backtest_expression()` executa

Para saídas como `{"result_y": g(x1, x2, x3)}`:

1. **Normaliza o grafo:** identifica as saídas e coleta folhas `forecast`, `known`, `constant` e
   `scenario`.
2. **Valida causalidade e tempo:** exige frequência, timezone e cutoff comuns nas entradas
   previstas; calcula a interseção dos históricos. Uma `known` temporal exige vintages para uma
   execução causal.
3. **Gera candidatos marginais:** caracteriza cada folha `forecast` e chama
   `CandidatePolicy.propose()`. Cada especificação é avaliada nas mesmas origens comuns.
4. **Pré-seleciona por entrada:** ordena candidatos por CRPS marginal, respeitando cobertura quando
   configurada, e conserva `preselect_top_k`.
5. **Monta combinações:** forma o produto cartesiano dos pré-selecionados, ordena pela soma dos
   CRPS marginais normalizados e limita a `max_combinations`.
6. **Avalia cada combinação:** em cada origem, amostra caminhos marginais, estima dependência usando
   somente erros observáveis até o cutoff, materializa entradas determinísticas, avalia o grafo e
   calcula MAE, RMSE, cobertura, largura e CRPS por saída.
7. **Seleciona:** minimiza a média ponderada do CRPS normalizado entre combinações que passam o
   guardrail de cobertura.
8. **Congela:** ajusta os specs vencedores no histórico completo, estima a dependência final e cria
   um `CompositionRun`. `promote_as` apenas associa um alias a esse manifesto.

O método não produz o forecast futuro; isso pertence a `compose()`.

<a id="cdf-preditiva"></a>

## 2.1 Como o motor constrói a CDF preditiva

O PIT não impõe uma família à série. Em cada origem rolling-origin, o adaptador ajusta seus
parâmetros apenas com o histórico anterior ao cutoff e devolve um `PredictiveDistribution`. Para
o alvo $t+h$, esse objeto representa

$$\widehat F_{t,h}(x)=P(X_{t+h}\le x\mid\mathcal I_t,\widehat\theta_t).$$

O chapéu registra que os parâmetros $\widehat\theta_t$ foram estimados e, em distribuições
empíricas, que a própria CDF possui aproximação de Monte Carlo.

| Adaptador | Hipótese distributiva desta implementação | CDF entregue ao PIT |
|---|---|---|
| `AR1Adapter` | inovações gaussianas homocedásticas | `GaussianDistribution`: $\Phi((x-\widehat\mu_h)/\widehat\sigma_h)$ |
| `LocalLevelAdapter` | ruídos gaussianos de estado e observação | `GaussianDistribution` obtida pelas recursões de Kalman |
| `TwoRegimeGaussianAdapter` | observação gaussiana condicional ao regime | `GaussianMixtureDistribution`: $\sum_k p_{k,h}\Phi((x-\mu_k)/\sigma_k)$ |
| `ARIMAAdapter` | perturbações do modelo estado-espaço do `statsmodels` | CDF empírica das trajetórias simuladas |
| `GARCH11Adapter` | choques padronizados $N(0,1)$ | CDF empírica das trajetórias simuladas |

Para caminhos $X_h^{(1)},\ldots,X_h^{(M)}$, a CDF empírica é

$$\widehat F_h(x)=\frac{1}{M}\sum_{m=1}^{M}\mathbf 1\{X_h^{(m)}\le x\}.$$

O motor chama `distribution.cdf(actual)`. Portanto, não substitui uma marginal empírica ou uma
mistura por uma normal. A normal padrão em $z=\Phi^{-1}(u)$ aparece somente depois do PIT, para
padronizar marginais diferentes e parametrizar a dependência residual.

Se a CDF preditiva contínua for correta, o teorema do PIT implica $U=F(X)\sim U(0,1)$. Isso é uma
consequência verificável — não uma garantia do modelo. Para uma CDF empírica com $M$ caminhos, os
PITs pertencem a uma grade de passo $1/M$ e carregam erro de Monte Carlo. A implementação limita
os valores a $[10^{-6},1-10^{-6}]$ antes de `norm.ppf`, evitando escores infinitos.

### Referências

- Diebold, Gunther e Tay (1998), [*Evaluating Density Forecasts with Applications to Financial
  Risk Management*](https://doi.org/10.2307/2527342): PIT para avaliação de forecasts de densidade.
- Gneiting, Balabdaoui e Raftery (2007), [*Probabilistic Forecasts, Calibration and
  Sharpness*](https://doi.org/10.1111/j.1467-9868.2007.00587.x): calibração, histogramas PIT e
  sharpness.
- [`ARIMAResults.simulate`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMAResults.simulate.html):
  simulação estado-espaço usada pelo adaptador ARIMA.
- Implementação local: [`contracts.py`](../src/time_series_lab/contracts.py) define as três
  distribuições e [`adapters.py`](../src/time_series_lab/adapters.py) associa cada modelo a elas.

## 3. Dependência: contexto do “Passo 2”

Em uma origem $o$, cada modelo foi ajustado somente em dados até o cutoff. Para o primeiro alvo
posterior, o realizado $x_{i,t}$ é transformado pela CDF que existia antes de observá-lo:

$$u_{i,t}=\widehat F_{i,t}(x_{i,t}),\qquad
z_{i,t}=\Phi^{-1}(\operatorname{clip}(u_{i,t},10^{-6},1-10^{-6})).$$

As linhas da matriz $Z$ são timestamps-alvo e as colunas são entradas previstas. A implementação
usa erros **um passo à frente** para estimar a associação contemporânea; a dependência entre
horizontes já permanece dentro dos caminhos produzidos por cada modelo marginal.

Durante o backtest de uma combinação em cutoff $c$, só entram linhas com alvo $t\le c$. Depois de
selecionar a combinação, o `CompositionRun` recebe a estimativa final usando todos os erros
rolling-origin disponíveis na janela de desenvolvimento.

Se houver $n$ linhas completas e $p$ variáveis:

1. calcula-se a correlação amostral $S$;
2. no modo automático, $\lambda=\min(1,(p+1)/(n-1))$;
3. aplica-se $R=(1-\lambda)S+\lambda I$;
4. autovalores negativos são truncados e a diagonal é renormalizada para 1.

Com menos de `min_paired_errors`, $R=I$ e `fallback=True`. Isso significa “amostra insuficiente”,
não evidência de independência.

## 4. Como a correlação vira caminhos conjuntos

`impose_rank_dependence()`:

1. sorteia uma matriz latente normal com correlação $R$;
2. calcula um score de severidade para cada caminho marginal completo;
3. ordena os caminhos de cada variável pelo score;
4. associa essa ordem aos ranks da coluna latente correspondente.

Somente as linhas são permutadas. Logo, a distribuição marginal e a trajetória temporal completa
de cada caminho permanecem inalteradas. O método aproxima dependência entre severidades; não é um
modelo estrutural multivariado.

## 5. O que `CompositionRun` contém — e o que não contém

Contém:

- `expression_ids` por saída;
- `selected_model_runs` e `selected_specs` por entrada prevista;
- `input_revisions`;
- métricas do vencedor e placar das combinações;
- `DependencyDiagnostics`;
- `causal` e timestamp de criação.

Não contém:

- caminhos futuros;
- distribuição final da saída;
- código da função privada;
- árvore serializada da fórmula;
- valores futuros de entradas `known`.

A aplicação reconstrói a expressão em memória. `compose()` confere seus hashes e rejeita revisões
de entrada diferentes das registradas no manifesto.

## 6. O que `compose()` executa

1. resolve `CompositionRun` por objeto, ID ou alias;
2. confere hashes das expressões e revisões;
3. reconstrói cada `PredictiveDistribution` marginal pelos `ModelRun` congelados;
4. amostra caminhos com seeds controladas;
5. aplica a dependência do `DependencyDiagnostics`;
6. materializa `known`, `constant` e `scenario` em matrizes `(M,H)`;
7. avalia o grafo caminho a caminho;
8. envolve cada matriz de saída em `EmpiricalDistribution`;
9. persiste e devolve `ComposedForecast`.

O Notebook 15 reproduz esses passos antes de chamar a fachada e testa igualdade elemento a
elemento entre as duas execuções.
