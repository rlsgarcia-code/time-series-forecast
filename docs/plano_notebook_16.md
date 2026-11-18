# Plano — Notebook 16: hierarquia de expressões probabilísticas

## Objetivo

Permitir que o usuário construa folhas de negócio derivadas
$Y_j=g_j(X_{j1},\ldots,X_{jp})$, organize-as em grupos e execute o ciclo completo de seleção,
composição, agregação, risco e coerência. Os modelos marginais pertencem às séries primitivas $X$;
as folhas $Y$ são distribuições obtidas pela avaliação caminho a caminho das funções.

## Contrato de entrada

Uma única célula editável retorna:

```python
GROUPED_EXPRESSIONS: dict[str, dict[str, ForecastVariable]]
```

- chave externa: nome do grupo;
- chave interna: nome estável da folha derivada;
- valor: expressão construída com operadores ou `engine.transform()`.

Entradas compartilhadas devem reutilizar o mesmo `ForecastVariable`, preservando exatamente o
mesmo caminho em todas as expressões consumidoras.

## Decisões didáticas e técnicas

- O exemplo usa quatro folhas derivadas e sete séries primitivas com menos de 80 observações.
- A função genérica $g(a,b,c)=ab+c$ demonstra multiplicação de aleatórias seguida de soma.
- `NaN` interno pode usar `ffill` causal; ausência no cutoff é rejeitada.
- A caracterização e `CandidatePolicy` atuam nas entradas primitivas.
- O backtest conjunto mede CRPS e cobertura nas folhas derivadas, grupos e total.
- `CompositionPolicy` limita a explosão combinatória pelo número de entradas primitivas.
- A dependência PIT também pertence às entradas; folhas derivadas não recebem uma segunda cópula.
- Funções privadas permanecem em memória; somente nome, versão e hash são persistidos.
- Coerência é testada sobre matrizes completas de trajetórias, não apenas médias.
- A documentação metodológica é ligada diretamente nos limites do tutorial.

## Critérios de conclusão

- Executar integralmente em kernel limpo, sem dados externos.
- Exigir alteração apenas em `build_grouped_expressions()` para conectar dados e fórmulas reais.
- Exibir auditoria das séries primitivas, candidatos e modelos escolhidos.
- Construir qualquer quantidade de grupos e folhas derivadas sem mudar as células posteriores.
- Demonstrar compartilhamento de uma entrada aleatória entre várias expressões.
- Produzir distribuições para folhas, grupos e total numa única chamada de `compose()`.
- Validar igualdade caminho a caminho entre folhas derivadas, grupos e total.
- Integrar gerador, README, plano do laboratório e testes estruturais.
