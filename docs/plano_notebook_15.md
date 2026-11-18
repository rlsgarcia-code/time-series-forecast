# Plano — Notebook 15: distribuição de $Y=g(X_1,X_2,X_3)$

## Objetivo

Explicar detalhadamente como forecasts marginais e entradas conhecidas são convertidos em caminhos
conjuntos e, depois, na distribuição empírica de uma única saída $Y=g(X_1,X_2,X_3)$. A hierarquia
aparece somente como extensão e é operacionalizada no Notebook 16.

## Decisões didáticas e técnicas

- O exemplo possui três entradas e permite alternar $X_3$ entre `known` e `forecast`.
- A função genérica $g(x_1,x_2,x_3)=x_1x_2+x_3$ torna explícito o termo de covariância.
- Duas etapas de seleção avaliam até quatro combinações pelo CRPS da saída derivada.
- `CandidatePolicy` e `CompositionPolicy` são distinguidas por unidade de decisão: a primeira
  propõe modelos por entrada; a segunda limita e avalia combinações no grafo.
- O ciclo interno de `backtest_expression()` é enumerado até a criação do `CompositionRun`,
  deixando explícito que o método treina e seleciona, mas não produz o forecast futuro.
- A mecânica é construída primeiro com objetos de baixo nível: `ModelRun`, `ModelAdapter`,
  `PredictiveDistribution`, `DependencyDiagnostics`, folhas materializadas, avaliação do grafo e
  `EmpiricalDistribution`.
- A implementação explícita reconstrói as mesmas matrizes de `compose()` com seeds idênticas.
- Média, variância, quantis e CDF são calculados diretamente da matriz de caminhos.
- Uma seção intermediária liga cada família de modelo à hipótese de inovação, ao objeto
  `PredictiveDistribution` e à construção analítica ou empírica de $\widehat F$.
- O texto separa a hipótese marginal do modelo, a implicação de calibração do PIT e a aproximação
  de cópula usada apenas na dependência entre entradas.
- Dependência residual e independência são comparadas sem trocar os modelos marginais.
- O passo de dependência explicita a linha do tempo causal, o significado das linhas e colunas da
  matriz PIT e a separação entre dependência contemporânea e temporal.
- A fachada `ForecastEngine.compose()` aparece somente depois da reconstrução de baixo nível.
- Fórmulas, algoritmos e contratos completos ficam na referência concisa
  [`composition_mechanics.md`](composition_mechanics.md), ligada diretamente pelo notebook.
- A formulação estatística, hipóteses, validação e limitações são mantidas no documento normativo
  [`metodo_composicao_probabilistica.md`](metodo_composicao_probabilistica.md).

## Critérios de conclusão

- Executar integralmente em kernel limpo e sem dados externos.
- Produzir ao menos duas combinações candidatas para `result_y`.
- Reproduzir exatamente os caminhos do motor usando objetos e funções de baixo nível.
- Demonstrar o caso de entrada conhecida com variância zero e indicar como torná-la forecast.
- Verificar numericamente o papel da covariância e construir a CDF empírica da saída.
- Mostrar, para os modelos escolhidos, qual objeto distributivo fornece `cdf()` ao PIT.
- Explicar o conteúdo e os limites de `CompositionRun` sem confundi-lo com `ComposedForecast`.
- Integrar o conteúdo ao gerador, README e testes estruturais.
