# Plano — Notebook 14: ciclo completo com uma `pandas.Series`

## Objetivo

Oferecer um tutorial executável no qual o usuário altera uma única célula para carregar uma
`pandas.Series` e percorre o ciclo completo do motor: contrato temporal, caracterização, definição
de candidatos, backtest, seleção, diagnóstico residual, forecast, simulação, risco, change-point e
monitoramento sequencial.

## Decisões didáticas e técnicas

- O notebook possui uma série sintética padrão para permanecer executável sem dados externos.
- A célula de entrada é marcada com `EDITE SOMENTE ESTA CÉLULA` e deve produzir `USER_SERIES`.
- Séries curtas são aceitas com backtest adaptativo; o caso de referência possui 57 observações.
- `NaN`/infinitos são auditados: bordas ausentes são removidas e valores internos seguem uma
  política explícita (`error`, `ffill` causal ou interpolação temporal não causal).
- Uma célula separada reúne parâmetros operacionais e flags para GARCH e Markov switching.
- A cauda da série é reservada antes de qualquer ajuste e revelada somente durante o monitoramento.
- A caracterização alimenta `CandidatePolicy`; inclusões e exclusões exibem regra, motivo e
  evidência, sem confundir geração de hipóteses com escolha do vencedor.
- O backtest usa rolling-origin; cobertura é guardrail e CRPS é a métrica primária.
- Se o guardrail eliminar todos os modelos, um fallback só ocorre quando autorizado pela
  configuração e fica registrado como tal.
- Diagnóstico residual não é confundido com a regra automática de seleção.
- Figuras retornadas pelo motor são exibidas sem depender de estado oculto do notebook.

## Sequência

1. Objetivos, pré-requisitos e mapa do ciclo.
2. Imports e configuração visual.
3. Célula editável de carregamento.
4. Configuração e validação do contrato temporal.
5. Separação desenvolvimento/monitoramento.
6. Caracterização, evidências e visualização.
7. Construção explicável e limitada dos candidatos por `CandidatePolicy`.
8. Backtest rolling-origin e placar.
9. Seleção, ajuste completo e promoção.
10. Resíduos e testes de adequação.
11. Forecast probabilístico.
12. Simulação, risco marginal e risco de trajetória.
13. Change-point retrospectivo.
14. Monitoramento prequential sobre a cauda reservada.
15. Auditoria, interpretação, limites, exercícios e gabarito orientativo.

## Critérios de conclusão

- Executar integralmente em kernel limpo com a série sintética padrão.
- Conter ao menos oito células de código e explicações antes de cada decisão.
- Não usar a cauda reservada na caracterização, no backtest ou na seleção.
- Executar com 57 observações e valores ausentes internos usando a política causal padrão.
- Exibir candidatos, métricas, política de seleção e eventual fallback.
- Expor resíduos, intervalos, caminhos, risco e eventos de monitoramento.
- Integrar validação ao `Makefile` e teste estrutural à suíte.
