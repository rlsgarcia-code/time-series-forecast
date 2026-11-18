# Plano — Notebook 15: distribuição de $Y=g(X_1,X_2,X_3)$

## Objetivo

Explicar detalhadamente como forecasts marginais e entradas conhecidas são convertidos em caminhos
conjuntos e, depois, na distribuição empírica de uma única saída $Y=g(X_1,X_2,X_3)$. A hierarquia
aparece somente como extensão e é operacionalizada no Notebook 16.

## Decisões didáticas e técnicas

- O exemplo possui três entradas e permite alternar $X_3$ entre `known` e `forecast`.
- A função genérica $g(x_1,x_2,x_3)=x_1x_2+x_3$ torna explícito o termo de covariância.
- Duas etapas de seleção avaliam até quatro combinações pelo CRPS da saída derivada.
- A mecânica é construída primeiro com objetos de baixo nível: `ModelRun`, `ModelAdapter`,
  `PredictiveDistribution`, `DependencyDiagnostics`, folhas materializadas, avaliação do grafo e
  `EmpiricalDistribution`.
- A implementação explícita reconstrói as mesmas matrizes de `compose()` com seeds idênticas.
- Média, variância, quantis e CDF são calculados diretamente da matriz de caminhos.
- Dependência residual e independência são comparadas sem trocar os modelos marginais.
- A fachada `ForecastEngine.compose()` aparece somente depois da reconstrução de baixo nível.

## Critérios de conclusão

- Executar integralmente em kernel limpo e sem dados externos.
- Produzir ao menos duas combinações candidatas para `result_y`.
- Reproduzir exatamente os caminhos do motor usando objetos e funções de baixo nível.
- Demonstrar o caso de entrada conhecida com variância zero e indicar como torná-la forecast.
- Verificar numericamente o papel da covariância e construir a CDF empírica da saída.
- Integrar o conteúdo ao gerador, README e testes estruturais.
