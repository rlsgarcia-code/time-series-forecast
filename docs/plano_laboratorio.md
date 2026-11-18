# Plano mestre — Time Series Forecast Lab

## Objetivo

Construir uma trilha executável, em português, que parte dos processos estocásticos fundamentais
e termina em uma proposta reproduzível de sistema de previsão probabilística e monitoramento
sequencial. Cada conceito deve
ser apresentado como equação, implementação explícita, visualização, exercício e teste.

O laboratório não será apenas uma coleção de notebooks. Ao final, o repositório deverá conter:

- notebooks didáticos que podem ser executados em ordem;
- biblioteca Python reutilizável com a lógica numérica validada;
- testes automatizados e validação de execução dos notebooks;
- ambiente bloqueado por `uv.lock`;
- documentação de decisões, protocolos de avaliação e arquitetura;
- uma proposta de sistema end-to-end de forecasting.

## Princípios de construção

1. **Do mecanismo à inferência:** primeiro simular com parâmetros conhecidos; depois estimá-los.
2. **Fórmula, código e figura juntos:** funções matemáticas centrais ficam explícitas no notebook.
3. **Uma mudança por vez:** seeds e dados são controlados em comparações de parâmetros.
4. **Incerteza é parte da previsão:** forecast significa distribuição, não apenas ponto.
5. **Validação temporal:** nenhuma avaliação usa embaralhamento que permita vazamento do futuro.
6. **Resíduos antes de confiança:** todo modelo deve justificar o que deixou sem explicar.
7. **Baseline obrigatório:** complexidade só é aceita quando supera regras simples sob o mesmo
   protocolo.
8. **Reprodutibilidade verificável:** cada notebook deve executar em kernel limpo e seus cálculos
   críticos devem possuir testes automatizados.
9. **Profundidade antes de velocidade:** conceitos novos seguem o padrão problema → intuição →
   símbolos → hipóteses → derivação → código → interpretação → armadilhas → checagem, documentado
   em `docs/padrao_didatico.md`.

## Trilha de notebooks

| Nº | Notebook | Pergunta central | Resultado esperado | Estado |
|---:|---|---|---|---|
| 01 | Ruído, memória e processos estocásticos | Como ruído branco, passeio aleatório e AR(1) diferem? | Simular, visualizar e produzir forecast probabilístico de AR(1) | concluído |
| 02 | Estimação e diagnóstico de AR(1) | Como aprender os parâmetros e verificar o modelo? | OLS, incerteza amostral, resíduos, Ljung–Box e backtest temporal | concluído |
| 03 | AR(p), MA(q) e ARMA | Como representar memórias e choques mais ricos? | ACF/PACF, seleção de ordens e comparação de modelos | concluído |
| 04 | Não estacionariedade e ARIMA | Quando e como diferenciar uma série? | raiz unitária, tendência, diferenciação e forecast em nível | concluído |
| 05 | Movimento Browniano, GBM e Itô | Como passar do tempo discreto ao contínuo? | caminhos Brownianos, transformação lognormal e lema de Itô | concluído |
| 06 | Volatilidade ARCH/GARCH | Como modelar variância que muda no tempo? | clustering, estimação GARCH e intervalos condicionais | concluído |
| 07 | Backtesting e seleção probabilística | Como comparar modelos sem vazamento? | rolling origin, baselines, cobertura, sharpness e scores | concluído |
| 08 | Primeiro sistema integrado | Como integrar o primeiro ciclo? | pipeline reproduzível, contratos, seleção, artefatos e monitoramento | concluído |
| 09 | Estado-espaço e Kalman | Como inferir estados latentes sob medições ruidosas? | filtro, smoother, MLE, gaps e forecast probabilístico | concluído |
| 10 | Markov switching | Como representar regimes discretos persistentes? | cadeia de Markov, probabilidades filtradas/suavizadas e forecast | concluído |
| 11 | Change-point detection | Como localizar quebras estruturais? | custos segmentados, penalização, calibração e avaliação offline | concluído |
| 12 | CUSUM e EWMA | Como detectar drift sequencialmente? | estatísticas online, ARL, limiares e compromisso atraso/falso alarme | concluído |
| 13 | Sistema integrado final | Como combinar modelos, detectores, risco e alertas? | pipeline persistente com políticas e auditoria end-to-end | concluído |
| 14 | Ciclo completo do usuário | Como levar uma `pandas.Series` por todo o motor? | caracterização, candidatos, seleção, risco e monitoramento guiados | concluído |
| 15 | Composição probabilística de variáveis | Como obter a distribuição de $Y=g(X_1,X_2,X_3)$? | objetos de baixo nível, marginais, cópula, broadcast e distribuição empírica | concluído |
| 16 | Hierarquia de expressões do usuário | Como agregar folhas $g(X_1,\ldots,X_p)$ em grupos e total? | entradas primitivas, fórmulas privadas, dependência, somas e risco | concluído |

O plano de cada notebook é refinado e documentado em `docs/plano_notebook_XX.md` antes de sua
implementação. A ordem pode ser ajustada com base no aprendizado, mas mudanças devem ser
registradas neste documento.

## Arquitetura-alvo do sistema

```text
dados brutos
    ↓
validação e contrato temporal
    ↓
transformações reproduzíveis
    ↓
baselines + modelos candidatos
    ↓
backtesting rolling-origin
    ↓
métricas pontuais e probabilísticas
    ↓
seleção com critérios explícitos
    ↓
forecast versionado + intervalos
    ↓
monitoramento de erro, cobertura e drift
```

### Componentes previstos

- **Data contract:** frequência, timezone, chaves, valores ausentes, horizonte e instante de corte.
- **Feature/transform pipeline:** diferenciação, escala e inversões sem vazamento.
- **Model interface:** `fit`, `predict_distribution`, serialização e metadados.
- **Backtester:** janelas expansivas ou móveis, múltiplos horizontes e baselines.
- **Evaluator:** MAE/RMSE, cobertura, largura, pinball loss e score probabilístico.
- **Selector:** comparação por horizonte e regra de decisão documentada.
- **Artifact store local:** configuração, parâmetros, previsões, métricas e versões.
- **Monitor:** erro realizado, calibração dos intervalos e mudança de distribuição.

## Critérios de conclusão por notebook

Um notebook só é considerado concluído quando:

1. possui plano didático/técnico documentado;
2. define objetivos, pré-requisitos e interpretação dos resultados;
3. mantém funções matemáticas importantes visíveis;
4. executa integralmente do topo ao fim em kernel limpo;
5. não depende de estado oculto;
6. inclui exercícios e gabarito executável;
7. possui testes automatizados proporcionais ao risco;
8. atualiza README, plano mestre e comandos de validação.

## Critério de conclusão do laboratório

O **primeiro ciclo educacional (01–08)** está concluído. A visão completa do projeto estará
concluída quando um usuário em macOS puder clonar o repositório, executar
`uv sync`, rodar todos os notebooks e testes, reproduzir os experimentos, alterar parâmetros e
executar um fluxo end-to-end que gere forecasts probabilísticos avaliados e versionados sem
vazamento temporal.

## Evidência de conclusão

- notebooks e planos específicos versionados para toda a trilha 01–16;
- funções matemáticas centrais explícitas em cada notebook;
- geradores reproduzíveis em `scripts/build_notebook_XX.py`;
- biblioteca reutilizável para processos, estimação, volatilidade, backtesting e artefatos;
- validação automatizada de cada notebook em kernel limpo;
- suíte de testes cobrindo invariantes numéricas, contratos e estrutura;
- Notebook 14 demonstrando o fluxo univariado guiado e Notebook 15 demonstrando composição
  probabilística, dependência residual e coerência hierárquica.

O Notebook 08 permanece como integração do primeiro ciclo; ele não substitui os capítulos de
Kalman, regimes, change-point ou detecção sequencial.

## Decisão sobre os Notebooks 03 e 04

A avaliação de densidade e coerência concluiu que ambos devem permanecer como capítulos únicos na
trilha principal. O racional e os possíveis complementos 03A/04A estão documentados em
`docs/avaliacao_desdobramento_03_04.md`.

