# API do motor de forecasting

## 1. Objetivo

`ForecastEngine` é a fachada de aplicação do Time Series Lab. Ela coordena os modelos,
backtesting, seleção, forecasts probabilísticos, simulação, risco e monitoramento sem substituir
as funções matemáticas dos capítulos.

A biblioteca não escolhe banco, serviço HTTP, fila ou agendador. A aplicação fornece uma
implementação de `EngineRepository`, enquanto o motor preserva as regras estatísticas e causais.

## 2. Visão dos contratos

```text
aplicação
   │
   ├── TimeSeries / Observation / políticas
   ▼
ForecastEngine ─── ModelRegistry ─── ModelAdapter
   │                                  │
   │                                  └── distribuição probabilística
   ├── DetectorRegistry
   └── EngineRepository (injetado pela aplicação)
```

As principais propriedades são:

- uma instância gerencia várias séries univariadas isoladas por `series_id`;
- todo forecast referencia a revisão, o cutoff e o modelo que o produziram;
- forecasts usados no monitoramento são preservados antes da chegada do realizado;
- promoção de modelo é explícita, salvo quando uma `PromotionPolicy` autoriza automação;
- efeitos externos, como notificações, ficam a cargo da aplicação consumidora.

## 3. Início rápido

O repositório didático abaixo existe apenas em `examples/`; ele não é um backend de produção.

```python
import pandas as pd

from examples.memory_repository import MemoryRepository
from time_series_lab import (
    AR1Spec,
    BacktestConfig,
    ForecastEngine,
    LocalLevelSpec,
    PromotionPolicy,
    SelectionPolicy,
    ThresholdRisk,
    TimeSeries,
)

repository = MemoryRepository()
engine = ForecastEngine(repository=repository)

history = pd.Series(
    values,
    index=pd.date_range("2025-01-01", periods=len(values), freq="D", tz="UTC"),
)
engine.register_series(TimeSeries.from_pandas("demanda", history))

training = engine.run_training_cycle(
    "demanda",
    [AR1Spec(), LocalLevelSpec()],
    BacktestConfig(initial_train_size=100, horizon=6, step=5),
    selection=SelectionPolicy(primary_metric="crps"),
    promotion=PromotionPolicy(automatic=True),
)

forecast = engine.forecast("demanda", horizon=12)
simulation = engine.simulate("demanda", horizon=12, n_paths=10_000, seed=7)
risk = engine.assess_risk(simulation, ThresholdRisk(limit=5.0))
```

## 4. Métodos de `ForecastEngine`

| Grupo | Métodos | Resultado |
|---|---|---|
| Dados | `register_series`, `append_history`, `get_series`, `list_series` | série validada e versionada |
| Diagnóstico | `characterize`, `propose_candidates`, `diagnose`, `plot_characterization`, `plot_residuals`, `detect_changes` | perfil, plano explicável, resíduos, figuras e change-points |
| Modelagem | `fit`, `backtest`, `select`, `promote`, `run_training_cycle` | modelo, relatório, decisão e promoção auditáveis |
| Forecast | `forecast`, `get_forecast` | distribuição preditiva preservada |
| Composição | `variable`, `transform`, `backtest_expression`, `compose`, `get_composition_run`, `get_composed_forecast` | grafo, seleção derivada e árvore probabilística coerente |
| Simulação e risco | `simulate`, `assess_risk` | trajetórias conjuntas e probabilidades de evento |
| Monitoramento | `start_monitoring`, `process_observation` | surpresa, CUSUM, EWMA, alerta e próximo forecast |
| Consulta | `get_champion`, `get_model_run`, `get_status`, `list_events`, `list_alerts` | estado operacional sem mutação |

Os atributos `config`, `repository`, `models`, `detectors` e `capabilities` são públicos para
inspeção. Registries podem receber adaptadores adicionais antes de construir o motor.

## 5. Dados e tempo

`TimeSeries` exige:

- timestamps crescentes, únicos, regulares e com timezone;
- valores finitos;
- uma frequência pandas explícita ou inferível;
- `series_id` não vazio.

`Observation` acrescenta `observation_id`, usado para retries idempotentes. Repetir o mesmo ID e
conteúdo devolve o `CycleResult` anterior; reutilizar o ID com outro valor gera
`RepositoryConflictError`.

Observações atrasadas, duplicadas ou fora da frequência são rejeitadas. Uma correção histórica
deve gerar nova revisão da série e novo ajuste, em vez de reescrever silenciosamente a evidência.

## 6. Modelos e distribuições

### 6.1 Caracterização antes da escolha

`characterize()` devolve os números e também os vetores usados nas visualizações:

```python
profile = engine.characterize("demanda", max_lag=24)

profile.summary                 # média, escala, assimetria, curtose e tendência
profile.values                  # série original
profile.trend                   # tendência linear descritiva
profile.detrended               # série menos a tendência
profile.differences             # primeira diferença
profile.acf, profile.pacf       # nível
profile.differenced_acf         # primeira diferença
profile.squared_acf             # memória na magnitude/volatilidade
profile.adf_pvalue              # H0: raiz unitária
profile.kpss_pvalue             # H0: estacionariedade em torno de tendência
profile.arch_lm_pvalue          # heterocedasticidade condicional
profile.evidence                # leituras textuais com as famílias favorecidas
profile.suggested_models        # candidatos, não uma seleção automática

dataframe = profile.to_frame()
figure = engine.plot_characterization("demanda")
figure.savefig("caracterizacao.png", dpi=150)
```

ADF e KPSS são usados em conjunto porque possuem hipóteses nulas diferentes. ACF/PACF sugerem
ordens candidatas, mas não substituem backtest. O sinal de quebra estrutural é uma evidência para
investigação, não uma prova de regime oculto.

### 6.2 Política explicável de candidatos

```python
from time_series_lab import CandidatePolicy

policy = CandidatePolicy(
    differencing="auto",
    garch="auto",
    markov_switching="auto",
    max_ar_order=2,
    max_ma_order=2,
    max_total_order=3,
    max_candidates=12,
)
plan = engine.propose_candidates(
    "demanda",
    policy=policy,
    profile=profile,       # evita caracterizar novamente
    series_kind="level",  # ou "returns"
)
```

`CandidatePlan.candidates` alimenta diretamente `backtest()`. `plan_id` identifica de forma
determinística a política, evidências e decisões. `decisions` e `to_frame()` registram
também as exclusões, com a regra, o motivo e os valores observados. `explain()` oferece a mesma
trilha em texto. A política:

- mantém baselines simples para verificar se complexidade realmente ajuda;
- calcula a faixa aproximada de significância da ACF/PACF e limita `(p,q)`;
- classifica ADF/KPSS como `stationary`, `nonstationary` ou `ambiguous`;
- em `differencing="auto"`, abre `d=1` nos dois últimos casos;
- condiciona GARCH a retornos, ARCH-LM e amostra mínima no modo automático;
- condiciona Markov switching ao sinal de quebra e a pelo menos 36 observações — seis por cada
  parâmetro livre do modelo de dois regimes; abaixo de 80, marca a hipótese como exploratória;
- aplica `max_candidates` para controlar sobreajuste da seleção.

Os modos `always` e `compare` forçam hipóteses para comparação, mas não as promovem. A escolha
continua em `select()` depois do backtest.

### 6.3 Resíduos depois do ajuste

Após ajustar ou promover um modelo:

```python
run = engine.fit("demanda", ARIMASpec(order=(1, 1, 1)))
diagnostics = engine.diagnose("demanda", model_ref=run.model_run_id)

diagnostics.observed
diagnostics.fitted_values
diagnostics.residuals
diagnostics.standardized_residuals
diagnostics.conditional_variance
diagnostics.acf
diagnostics.pacf
diagnostics.squared_acf
diagnostics.qq_theoretical, diagnostics.qq_sample
diagnostics.tests
diagnostics.interpretation

residual_frame = diagnostics.to_frame()
figure = engine.plot_residuals("demanda", model_ref=run.model_run_id)
```

Os resíduos respeitam cada família:

- AR/ARIMA: erro observado menos valor ajustado;
- Kalman: inovação um passo à frente e sua variância preditiva;
- GARCH: retorno centrado padronizado pela volatilidade condicional;
- Markov switching: erro frente à média preditiva da mistura, incluindo incerteza entre regimes.

`tests` inclui Ljung–Box dos resíduos, Ljung–Box dos quadrados, Jarque–Bera e ARCH-LM. A escolha
deve buscar resíduos sem autocorrelação remanescente e avaliar separadamente distribuição e
volatilidade. Um p-valor baixo não determina sozinho qual modelo vencerá; essa decisão continua no
backtest fora da amostra.

### 6.4 Famílias disponíveis

Especificações concretas incluídas:

- `AR1Spec` — formulação analítica transparente;
- `ARIMASpec(order=(p,d,q))` — inclui AR, MA, ARMA e ARIMA;
- `LocalLevelSpec` — nível local com filtro de Kalman;
- `GARCH11Spec` — volatilidade condicional para retornos;
- `TwoRegimeGaussianSpec` — mistura gaussiana e cadeia de Markov com dois regimes.

Todo adaptador produz um `PredictiveDistribution` com:

```python
distribution.mean
distribution.variance
distribution.quantile([0.025, 0.5, 0.975])
distribution.cdf(values)
distribution.sample(n_paths=10_000, seed=7)
```

As implementações são `GaussianDistribution`, `GaussianMixtureDistribution` e
`EmpiricalDistribution`. `sample()` devolve matriz `(n_paths, horizon)` e conserva a dependência
temporal usada no risco de trajetória.

## 7. Backtest, seleção e promoção

`backtest()` usa rolling-origin. Cada candidato recebe exatamente o histórico anterior à origem;
as métricas são calculadas por horizonte e de forma agregada:

- MAE e RMSE;
- cobertura e largura de intervalo;
- CRPS amostral, que funciona também para misturas e distribuições empíricas.

`select()` aplica `SelectionPolicy`, ajusta o vencedor no histórico atual e devolve uma
`SelectionDecision`. Esse resultado ainda não altera o alias operacional. `promote()` é a única
operação que troca `champion` explicitamente.

`run_training_cycle()` reúne essas etapas. `PromotionPolicy(automatic=False)` é o padrão.

## 8. Composição probabilística e hierárquica

`variable()` cria folhas com modos `forecast`, `known`, `constant` ou `scenario`. Operadores
aritméticos produzem novos `ForecastVariable`; `transform()` registra uma função privada por nome e
versão sem serializar seu código.

`backtest_expression()` executa duas etapas: reduz candidatos por entrada e avalia combinações
limitadas pelo CRPS normalizado das saídas. Séries `known` variáveis no tempo exigem vintages com
colunas `as_of`, `target`, `value`. Sem vintages, somente uma análise explicitamente não causal é
permitida, e ela não pode ser promovida. Entradas `scenario` seguem a mesma regra: usam a chave
`observed` no backtest não causal e uma chave escolhida em `compose(scenario=...)`.

`compose()` exige um `CompositionRun` congelado, confere hashes e revisões, coordena os caminhos
por uma cópula de erros PIT sincronizados e devolve `ComposedForecast`. Cada saída é uma
`EmpiricalDistribution`, portanto pode ser passada diretamente a `assess_risk()`.

Papéis das políticas, fases exatas de `backtest_expression()`, estimação causal da dependência e
conteúdo de `CompositionRun` estão detalhados em
[`composition_mechanics.md`](composition_mechanics.md).
A fundamentação estatística, suas hipóteses e limitações estão em
[`metodo_composicao_probabilistica.md`](metodo_composicao_probabilistica.md).

```python
run = engine.backtest_expression(
    {"leaf": leaf, "total": total},
    backtest=BacktestConfig(initial_train_size=40, horizon=1, step=4),
    policy=CompositionPolicy(preselect_top_k=2, max_combinations=32),
    promote_as="composition-champion",
)
result = engine.compose(
    {"leaf": leaf, "total": total},
    model_ref="composition-champion",
    horizon=12,
    n_paths=5_000,
)
```

O hash identifica a estrutura em memória, mas o payload persistido contém somente hashes,
linhagem, revisões, modelos e métricas. Alterar uma função privada exige alterar sua versão.
O Notebook 16 fornece o ciclo didático para
`dict[str, dict[str, ForecastVariable]]`: as séries primitivas recebem modelos, expressões privadas
formam as folhas e as somas constroem grupos e total com coerência caminho a caminho.

## 9. Monitoramento causal

`start_monitoring()` preserva o forecast inicial. Depois, cada `process_observation()` executa:

1. localiza o forecast emitido antes do timestamp observado;
2. calcula o PIT e o transforma para a escala normal;
3. atualiza anomalia pontual, CUSUM e EWMA;
4. registra evento e anexa a observação à série;
5. refaz o ajuste do mesmo modelo, se a política assim determinar;
6. emite o forecast do próximo ciclo.

Para uma distribuição Normal, a transformação PIT equivale ao resíduo padronizado. Para Markov
switching ela usa a CDF da mistura completa. Se não existir forecast pendente, a observação é
registrada como `observation_unscored` e não alimenta os detectores.

Alertas e recomendações de retreino são eventos. O motor não envia e-mail, webhook ou comando de
intervenção.

## 10. Contrato de persistência

`EngineRepository` define operações para:

- séries e controle otimista de `revision`;
- snapshots de modelos e aliases;
- forecasts e marcação dos alvos avaliados;
- sessões/checkpoints de monitoramento;
- eventos e resultados idempotentes de ciclo.
- execuções, aliases e forecasts de composição sem a árvore da fórmula privada.

O repositório deve garantir atomicidade por série. Snapshots contêm estruturas versionáveis e
arrays, sem exigir `pickle`. A implementação pode usar PostgreSQL, object storage, SQLite ou os
serviços da aplicação, desde que preserve o contrato.

## 11. Extensão

Um novo modelo implementa `ModelAdapter` e declara `ModelCapabilities`. O adaptador deve ajustar
uma `TimeSeries`, produzir estado seguro, reconstruir uma distribuição e simular trajetórias.
Depois ele é registrado em `ModelRegistry`.

As APIs funcionais dos notebooks permanecem disponíveis. A fachada é uma camada de coordenação,
não uma substituição das formulações matemáticas usadas para estudo e validação.
