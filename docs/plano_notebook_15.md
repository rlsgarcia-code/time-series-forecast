# Plano — Notebook 15: composição probabilística e hierárquica

## Objetivo

Ensinar como combinar forecasts univariados, séries conhecidas e constantes em um grafo pequeno,
selecionar modelos pelo resultado derivado e produzir folhas, grupos e total coerentes caminho a
caminho, sem persistir fórmulas privadas.

## Decisões didáticas e técnicas

- O exemplo sintético usa dois drivers correlacionados e uma série conhecida com vintages.
- A formulação explicita Monte Carlo para $E[g(X)]$ e o termo de covariância de $E[XY]$.
- O backtest mostra pré-seleção marginal e seleção final por CRPS normalizado das saídas.
- A dependência é explicada por PIT, shrinkage, fallback e reordenação por ranks.
- Três folhas formam grupo e total sobre as mesmas trajetórias, com teste de igualdade exata.
- A função privada demonstra persistência por nome/versão/hash, sem revelar código.
- Risco, limitações, exercícios e gabarito completam o ciclo tutorial.

## Critérios de conclusão

- Executar integralmente em kernel limpo e sem dados externos.
- Demonstrar vintages sem vazamento, dependência e efeito de covariância.
- Exibir métricas da composição, modelos selecionados e diagnóstico da cópula.
- Provar coerência da árvore caminho a caminho.
- Integrar o notebook ao gerador, `Makefile` e testes estruturais.
