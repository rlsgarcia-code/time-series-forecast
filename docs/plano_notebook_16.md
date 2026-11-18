# Plano — Notebook 16: ciclo hierárquico a partir de um dicionário

## Objetivo

Permitir que o usuário edite uma única célula para fornecer `dict[str, list[pd.Series]]` e execute
o ciclo completo de forecasting hierárquico: validação, caracterização marginal, seleção composta,
dependência, forecast de folhas/grupos/total, coerência e risco.

## Decisões didáticas e técnicas

- Cada chave é um grupo e cada lista contém suas folhas; o total global é opcional.
- Séries recebem IDs estáveis `grupo/nome` e precisam terminar no mesmo cutoff/frequência.
- `NaN` interno pode usar `ffill` causal; ausência no último ponto é rejeitada.
- A construção do grafo é genérica e não depende da quantidade de grupos ou folhas.
- `top_k=1` é o padrão escalável; o custo combinatório é explicado antes da execução.
- Dependência residual, distribuições dos nós, risco e limites aparecem separadamente.
- Coerência é testada sobre as matrizes completas de trajetórias, não apenas sobre médias.

## Critérios de conclusão

- Executar integralmente com o dicionário sintético padrão e `NaN` internos.
- Exigir apenas a edição de `load_grouped_series()` para conectar dados reais.
- Exibir auditoria da entrada, candidatos, modelos escolhidos e matriz de dependência.
- Construir qualquer quantidade de grupos/folhas sem alteração das células seguintes.
- Validar igualdade exata entre folhas, grupos e total em todos os caminhos.
- Integrar gerador, plano, `Makefile`, README e testes estruturais.
