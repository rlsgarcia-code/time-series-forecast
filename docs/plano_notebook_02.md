# Plano didático e técnico — Notebook 02

## Propósito

Transformar o AR(1) de mecanismo conhecido em modelo estimado. O notebook conecta quatro etapas
que frequentemente aparecem separadas: estimação, incerteza dos parâmetros, diagnóstico residual
e avaliação fora da amostra.

## Sequência didática

1. gerar uma série com parâmetros conhecidos e depois ocultá-los do estimador;
2. reescrever AR(1) como regressão entre pares consecutivos;
3. derivar e implementar mínimos quadrados explicitamente;
4. comparar estimativas com a verdade apenas para fins pedagógicos;
5. repetir a amostragem para visualizar dispersão e viés;
6. diagnosticar resíduos de um modelo correto e de um modelo deliberadamente errado;
7. separar treino/teste cronologicamente;
8. comparar forecast multi-step de origem fixa, rolling one-step e baseline ingênuo.

## Decisões técnicas

- As funções centrais aparecem integralmente no notebook e também em
  `src/time_series_lab/estimation.py` para reutilização e testes.
- OLS usa `numpy.linalg.lstsq`; a inversa aparece apenas na covariância estimada.
- ACF residual é implementada no notebook; Ljung–Box usa `statsmodels`.
- Simulações repetidas usam sementes derivadas de um gerador local.
- Forecasts são avaliados apenas em divisões temporais, nunca com embaralhamento.
- O notebook explicita que intervalos plug-in ignoram incerteza dos parâmetros.

## Critérios de conclusão

- O notebook executa do início ao fim em kernel limpo.
- O estimador recupera aproximadamente parâmetros conhecidos em amostra longa.
- As condições de ortogonalidade OLS são verificadas numericamente.
- O exemplo AR(2) mal especificado deixa autocorrelação residual detectável.
- Os protocolos origem fixa e rolling são explicados e avaliados separadamente.
- A suíte automatizada cobre estimação, intervalos, rolling e métricas.

