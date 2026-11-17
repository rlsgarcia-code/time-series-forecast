# Plano didático e técnico — Notebook 01

## Propósito

Construir a intuição mínima para previsão probabilística antes de introduzir estimação,
seleção de modelos ou dados reais. O notebook acompanha uma progressão deliberada:

1. ruído branco como inovação sem memória;
2. passeio aleatório como soma de inovações com choques permanentes;
3. AR(1) como memória geométrica controlada por `phi`;
4. ACF e resposta a choques como duas leituras da mesma persistência;
5. forecast como distribuição condicional, validado por Monte Carlo.

## Decisões didáticas

- **Uma mudança por vez:** comparações de `phi` reutilizam a mesma semente aleatória.
- **Uma trajetória versus um conjunto:** o aluno vê primeiro uma realização e depois 100
  trajetórias, reduzindo o risco de generalizar a partir de um único caminho.
- **Fórmula + figura + checagem:** cada conceito central aparece nas três formas.
- **Interatividade com propósito:** o slider é restrito a `|phi| < 1` e atualiza o caminho e
  a ACF simultaneamente.
- **Forecast calibrado:** a distribuição analítica é comparada a 5.000 simulações futuras.
- **Prática com feedback:** exercícios conceituais e numéricos precedem um gabarito recolhível
  e executável.

## Decisões técnicas

- Python 3.12 gerenciado por `uv`, com versões exatas registradas em `uv.lock`.
- As funções numéricas são escritas explicitamente no notebook, junto às respectivas equações,
  para que recorrências, somas e fórmulas preditivas possam ser inspecionadas passo a passo.
- A implementação equivalente em `src/time_series_lab/simulations.py` permanece como versão
  reutilizável e testada para os próximos notebooks.
- `numpy.random.default_rng` e sementes explícitas em todos os experimentos.
- `matplotlib` para figuras estáticas e `ipywidgets` para o controle interativo.
- Testes unitários cobrem reprodutibilidade, momentos, ACF, resposta a choques e equivalência
  entre forecast analítico e Monte Carlo.
- O validador executa todas as células em um kernel limpo e falha no primeiro erro.

## Critérios de conclusão

- O notebook executa do início ao fim sem estado oculto.
- A suíte de testes passa.
- As 100 trajetórias deixam visível a diferença entre variância crescente e estacionariedade.
- O slider de `phi`, a ACF e a resposta a choques concordam entre si.
- A média e o desvio-padrão de Monte Carlo ficam próximos da solução analítica.
- Uma instalação nova no macOS pode ser feita apenas com as instruções do `README.md`.

## Sequência planejada do laboratório

O Notebook 01 entrega a base. Os próximos notebooks podem avançar para estimação de AR,
diagnóstico de resíduos, ARMA/ARIMA, movimento Browniano e Itô, e volatilidade ARCH/GARCH.
A arquitetura atual permite reutilizar as simulações sem copiar células entre notebooks.

