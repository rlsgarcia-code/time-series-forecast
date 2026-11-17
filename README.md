# Time Series Forecast

Laboratório e motor Python, em português, para caracterização de séries temporais, comparação de
modelos, forecasting probabilístico, simulação, avaliação de risco e monitoramento sequencial.

O projeto combina duas entregas:

- uma biblioteca importável, centrada em `ForecastEngine`;
- 14 notebooks que apresentam a formulação matemática e validam cada componente passo a passo.

## Fluxo completo

```text
série temporal
    ↓
validação do contrato temporal
    ↓
caracterização e geração de evidências
    ↓
definição explícita dos modelos candidatos
    ↓
backtest rolling-origin sem vazamento
    ↓
guardrail de cobertura + seleção por CRPS
    ↓
ajuste no histórico completo e promoção
    ↓
diagnóstico residual
    ↓
forecast probabilístico e simulação de trajetórias
    ↓
risco de limite / tempo até evento
    ↓
monitoramento prequential com anomalia, CUSUM e EWMA
```

Não existe uma escolha automática baseada apenas na aparência da série. A caracterização orienta
quais candidatos devem ser comparados, mas a decisão operacional é baseada em desempenho fora da
amostra.

## 1. Contrato de entrada

O motor trabalha atualmente com séries univariadas. `TimeSeries` exige:

- `series_id` não vazio;
- `DatetimeIndex` crescente, único e com timezone;
- frequência regular e explícita ou inferível pelo pandas;
- valores numéricos finitos;
- ausência de gaps e duplicações.

```python
from time_series_lab import TimeSeries

series = TimeSeries.from_pandas(
    "demanda",
    pandas_series,  # pd.Series com DatetimeIndex
)
```

O contrato interno permanece estrito e não aceita `NaN`. Para entrada exploratória, o
[Notebook 14](notebooks/14_ciclo_completo_usuario.ipynb) faz uma etapa anterior e auditável de
limpeza: aceita séries curtas, remove bordas ausentes e permite escolher entre erro, preenchimento
causal pelo último valor (`ffill`) ou interpolação temporal não causal. O notebook registra quantos
valores foram tratados antes de criar `TimeSeries`.

Observações atrasadas ou revisões históricas não são sobrescritas silenciosamente. Uma alteração da
série incrementa sua `revision`, e um modelo ajustado em revisão anterior é considerado obsoleto.

Critério implementado em [`TimeSeries`](src/time_series_lab/contracts.py) e aplicado por
[`ForecastEngine.register_series()`](src/time_series_lab/engine.py).

## 2. Como a série é caracterizada

```python
profile = engine.characterize("demanda", max_lag=24)
figure = engine.plot_characterization("demanda", max_lag=24)
```

`SeriesProfile` expõe os dados e não apenas um parecer final:

- valores, tendência linear, série detrendida e primeira diferença;
- ACF e PACF do nível;
- ACF e PACF da primeira diferença;
- ACF dos valores detrendidos ao quadrado;
- média, desvio, mediana, assimetria e excesso de curtose;
- ADF e KPSS para estacionariedade;
- Ljung–Box para dependência serial;
- ARCH-LM para agrupamento de volatilidade;
- Jarque–Bera para forma da distribuição;
- redução de custo produzida por uma possível quebra de média.

### Regras que geram as sugestões

| Evidência | Regra atual | Famílias favorecidas |
|---|---|---|
| Estacionariedade | ADF `p < 0,05` e KPSS `p > 0,05` | AR(1), ARIMA com `d=0` |
| Possível não estacionariedade | combinação acima não satisfeita | ARIMA com diferenciação, nível local |
| Memória serial | Ljung–Box da série detrendida com `p < 0,05` | AR(1), ARIMA |
| Tendência relevante | variação da tendência linear ≥ 1 desvio-padrão da série | ARIMA, nível local |
| Volatilidade condicional | ARCH-LM com `p < 0,05` | GARCH(1,1) |
| Possível mudança estrutural | quebra única reduz ≥ 15% do custo sem quebra | Markov switching, nível local |

Essas regras produzem `profile.evidence` e `profile.suggested_models`. Elas não promovem modelos e
não escolhem ordens `(p,d,q)`. O código está em
[`diagnostics.characterize_series()`](src/time_series_lab/diagnostics.py); a fundamentação aparece
principalmente nos Notebooks [03](notebooks/03_arp_maq_arma.ipynb),
[04](notebooks/04_nao_estacionariedade_arima.ipynb),
[06](notebooks/06_arch_garch.ipynb) e [11](notebooks/11_change_point.ipynb).

## 3. Modelos disponíveis

O motor avalia somente os objetos fornecidos em `candidates`. Por exemplo, três ordens ARIMA
diferentes são três candidatos distintos.

| Especificação | Processo representado | Forecast produzido |
|---|---|---|
| `AR1Spec()` | AR(1), com ou sem intercepto | Normal multivariada analítica |
| `ARIMASpec(order=(p,d,q))` | AR, MA, ARMA e ARIMA | distribuição empírica obtida por simulação |
| `LocalLevelSpec()` | nível local em estado-espaço/Kalman | Normal multivariada |
| `GARCH11Spec()` | GARCH(1,1), com média zero ou constante | distribuição empírica com variância condicional |
| `TwoRegimeGaussianSpec()` | Markov switching gaussiano com dois regimes | mistura gaussiana com transição de regimes |

Exemplos de equivalência pela ordem ARIMA:

```python
ARIMASpec(order=(2, 0, 0))  # AR(2)
ARIMASpec(order=(0, 0, 2))  # MA(2)
ARIMASpec(order=(1, 0, 1))  # ARMA(1,1)
ARIMASpec(order=(1, 1, 1))  # ARIMA(1,1,1)
```

As especificações públicas estão em [`contracts.py`](src/time_series_lab/contracts.py) e suas
implementações em [`adapters.py`](src/time_series_lab/adapters.py).

## 4. Como os modelos são avaliados

`backtest()` usa rolling-origin:

1. começa com `initial_train_size` observações;
2. ajusta cada candidato somente com dados anteriores à origem;
3. prevê `horizon` passos;
4. avança `step` observações;
5. repete ajuste e forecast até o fim da janela;
6. agrega métricas globalmente e separadamente por horizonte.

```python
from time_series_lab import BacktestConfig

config = BacktestConfig(
    initial_train_size=180,
    horizon=6,
    step=5,
    confidence=0.95,
    seed=42,
)

report = engine.backtest("demanda", candidates, config)
```

Métricas registradas:

- `mae`: erro absoluto médio do forecast pontual;
- `rmse`: raiz do erro quadrático médio;
- `coverage`: proporção de realizados dentro do intervalo configurado;
- `mean_width`: largura média do intervalo;
- `crps`: qualidade da distribuição preditiva, combinando calibração e concentração.

O CRPS é calculado por amostragem para funcionar de forma uniforme com Normais, misturas e
distribuições empíricas. Os registros completos ficam em `report.rows`; agregados por modelo e
horizonte ficam em `report.scores`.

Configuração e algoritmo:

- [`BacktestConfig`](src/time_series_lab/contracts.py);
- [`ForecastEngine.backtest()`](src/time_series_lab/engine.py);
- Notebook [07 — Backtesting e seleção](notebooks/07_backtesting_selecao.ipynb).

## 5. Critério de escolha do modelo

A política padrão é:

1. eliminar candidatos cuja cobertura agregada esteja fora de `[0,80; 1,00]`;
2. entre os elegíveis, escolher o menor `crps` agregado;
3. reajustar o vencedor usando todo o histórico disponível;
4. não promovê-lo automaticamente, salvo quando isso for solicitado explicitamente.

```python
from time_series_lab import SelectionPolicy, PromotionPolicy

decision = engine.select(
    report,
    SelectionPolicy(
        primary_metric="crps",
        coverage_range=(0.80, 1.00),
    ),
)

promotion = engine.promote(
    "demanda",
    decision.model_run_id,
    decision_id=decision.decision_id,
)
```

Ou execute backtest, seleção, ajuste final e promoção em um ciclo:

```python
training = engine.run_training_cycle(
    "demanda",
    candidates=candidates,
    backtest=config,
    selection=SelectionPolicy(primary_metric="crps"),
    promotion=PromotionPolicy(automatic=True),
)
```

Se nenhum candidato satisfizer o guardrail, a seleção falha em vez de promover o “menos ruim”. É
possível remover o filtro usando `coverage_range=None` ou trocar `primary_metric`, desde que a
métrica exista no relatório.

O critério está em [`SelectionPolicy`](src/time_series_lab/contracts.py) e
[`ForecastEngine.select()`](src/time_series_lab/engine.py).

## 6. Diagnóstico depois do ajuste

O diagnóstico residual é uma etapa separada da seleção automática atual. Hoje os testes residuais
geram evidências e visualizações; eles não eliminam candidatos no `select()`.

```python
diagnostics = engine.diagnose(
    "demanda",
    model_ref=training.model_run.model_run_id,
    max_lag=24,
)

diagnostics.residuals
diagnostics.standardized_residuals
diagnostics.fitted_values
diagnostics.conditional_variance
diagnostics.acf
diagnostics.pacf
diagnostics.squared_acf
diagnostics.tests
diagnostics.interpretation

figure = engine.plot_residuals(
    "demanda",
    model_ref=training.model_run.model_run_id,
)
```

Testes disponíveis:

- Ljung–Box dos resíduos: dinâmica da média ainda não explicada;
- Ljung–Box dos resíduos ao quadrado: memória na magnitude;
- ARCH-LM: heterocedasticidade condicional;
- Jarque–Bera e Q–Q: incompatibilidade com a hipótese gaussiana.

Os resíduos respeitam a família: erro ajustado em AR/ARIMA, inovação em Kalman, retorno dividido
pela volatilidade em GARCH e erro frente à mistura preditiva em Markov switching.

Implementação em [`diagnose_residuals()`](src/time_series_lab/diagnostics.py).

## 7. Forecast, simulação e risco

Todo forecast registra série, revisão, cutoff, modelo, timestamps-alvo e distribuição. O contrato
comum oferece:

```python
forecast = engine.forecast("demanda", horizon=12)

forecast.distribution.mean
forecast.distribution.variance
forecast.distribution.quantile([0.025, 0.50, 0.975])
forecast.distribution.cdf(values)
paths = forecast.distribution.sample(10_000, seed=42)
```

Para cenários conjuntos:

```python
simulation = engine.simulate(
    "demanda",
    horizon=12,
    n_paths=10_000,
    seed=42,
)
```

O risco pode ser calculado acima ou abaixo de um limite:

```python
from time_series_lab import ThresholdRisk

risk = engine.assess_risk(
    simulation,
    ThresholdRisk(limit=100.0, direction="above"),
)

risk.marginal_probability       # por horizonte
risk.any_horizon_probability    # algum horizonte cruza o limite
risk.first_passage_probability  # primeira passagem por horizonte
```

O risco de trajetória usa caminhos conjuntos; probabilidades marginais não são somadas.

## 8. Monitoramento sequencial

`start_monitoring()` emite e preserva um forecast antes de qualquer realizado. Quando a nova
observação chega, `process_observation()`:

1. recupera o forecast anteriormente emitido;
2. calcula o PIT e o transforma para escala Normal;
3. atualiza os detectores;
4. aplica cooldown e política de reset;
5. registra a observação e os eventos;
6. refaz o ajuste, se configurado;
7. emite o forecast do próximo ciclo.

Parâmetros padrão de `MonitoringPolicy`:

| Parâmetro | Padrão | Uso |
|---|---:|---|
| `point_threshold` | `3.0` | anomalia se `|surpresa| ≥ 3` |
| `cusum_reference` | `0.4` | mudança mínima acumulada pelo CUSUM |
| `cusum_threshold` | `6.0` | limite de alarme do CUSUM |
| `ewma_smoothing` | `0.15` | peso da observação atual |
| `ewma_limit_multiplier` | `3.6` | largura do limite EWMA |
| `cooldown` | `12` | ciclos mínimos entre alertas emitidos |
| `reset_on_alert` | `True` | reinicia detectores após alerta |
| `refit_each_observation` | `True` | reajusta o modelo a cada realizado |
| `forecast_horizon` | `1` | horizonte emitido a cada ciclo |

```python
from time_series_lab import MonitoringPolicy, Observation

session = engine.start_monitoring("demanda", MonitoringPolicy())

cycle = engine.process_observation(
    session.session_id,
    Observation(
        timestamp=proximo_timestamp,
        value=novo_valor,
        observation_id="observacao-001",
    ),
)
```

Uma observação sem forecast preservado é marcada como `observation_unscored` e não alimenta os
detectores. O change-point offline é retrospectivo e permanece separado de CUSUM/EWMA.

Critérios em [`MonitoringPolicy`](src/time_series_lab/contracts.py), detectores em
[`detectors.py`](src/time_series_lab/detectors.py) e ciclo em
[`ForecastEngine.process_observation()`](src/time_series_lab/engine.py).

## 9. Exemplo end-to-end

O repositório em memória abaixo é didático. Uma aplicação real deve implementar
`EngineRepository` com persistência transacional.

```python
from examples.memory_repository import MemoryRepository
from time_series_lab import (
    AR1Spec,
    ARIMASpec,
    BacktestConfig,
    ForecastEngine,
    LocalLevelSpec,
    PromotionPolicy,
    SelectionPolicy,
    TimeSeries,
)

engine = ForecastEngine(repository=MemoryRepository())
engine.register_series(TimeSeries.from_pandas("demanda", pandas_series))

profile = engine.characterize("demanda")

candidates = [
    AR1Spec(),
    ARIMASpec(order=(1, 0, 1)),
    ARIMASpec(order=(1, 1, 1)),
    LocalLevelSpec(),
]

# Acrescente GARCH apenas para retornos/erros com volatilidade condicional.
# Acrescente Markov switching quando houver hipótese substantiva de regimes.

training = engine.run_training_cycle(
    "demanda",
    candidates=candidates,
    backtest=BacktestConfig(initial_train_size=180, horizon=6, step=5),
    selection=SelectionPolicy(primary_metric="crps", coverage_range=(0.80, 1.00)),
    promotion=PromotionPolicy(automatic=True),
)

diagnostics = engine.diagnose("demanda")
forecast = engine.forecast("demanda", horizon=12)
simulation = engine.simulate("demanda", horizon=12, n_paths=10_000, seed=42)
```

## 10. Onde cada regra está

| Responsabilidade | Código | Explicação didática |
|---|---|---|
| Contrato temporal e tipos | [`contracts.py`](src/time_series_lab/contracts.py) | [Notebook 08](notebooks/08_sistema_integrado.ipynb) |
| Caracterização e resíduos | [`diagnostics.py`](src/time_series_lab/diagnostics.py) | Notebooks [02](notebooks/02_estimacao_diagnostico_ar1.ipynb), [03](notebooks/03_arp_maq_arma.ipynb), [04](notebooks/04_nao_estacionariedade_arima.ipynb) e [06](notebooks/06_arch_garch.ipynb) |
| Modelos concretos | [`adapters.py`](src/time_series_lab/adapters.py) | Notebooks 01–06, 09 e 10 |
| Backtest e seleção | [`engine.py`](src/time_series_lab/engine.py) | [Notebook 07](notebooks/07_backtesting_selecao.ipynb) |
| Distribuições e artefatos | [`contracts.py`](src/time_series_lab/contracts.py) | [Notebook 13](notebooks/13_sistema_final.ipynb) |
| Change-point offline | [`change_points.py`](src/time_series_lab/change_points.py) | [Notebook 11](notebooks/11_change_point.ipynb) |
| CUSUM/EWMA online | [`detectors.py`](src/time_series_lab/detectors.py) | [Notebook 12](notebooks/12_cusum_ewma.ipynb) |
| Orquestração completa | [`ForecastEngine`](src/time_series_lab/engine.py) | Notebooks [13](notebooks/13_sistema_final.ipynb) e [14](notebooks/14_ciclo_completo_usuario.ipynb) |

A referência detalhada da API está em [`docs/engine_api.md`](docs/engine_api.md). A visão do
sistema e seus limites estão em [`docs/system_vision.md`](docs/system_vision.md).

## 11. Trilha dos notebooks

| Nº | Tema | Entrega principal |
|---:|---|---|
| [01](notebooks/01_processos_estocasticos.ipynb) | Processos estocásticos | ruído, passeio aleatório, AR(1), choque e forecast probabilístico |
| [02](notebooks/02_estimacao_diagnostico_ar1.ipynb) | Estimação AR(1) | OLS, incerteza, resíduos e avaliação temporal |
| [03](notebooks/03_arp_maq_arma.ipynb) | AR(p), MA(q) e ARMA | ACF/PACF, AIC e seleção de ordens |
| [04](notebooks/04_nao_estacionariedade_arima.ipynb) | ARIMA | ADF, diferenciação e inversão |
| [05](notebooks/05_browniano_gbm_ito.ipynb) | Browniano, GBM e Itô | variação quadrática, Euler e solução exata |
| [06](notebooks/06_arch_garch.ipynb) | ARCH/GARCH | MLE, volatilidade, resíduos e VaR |
| [07](notebooks/07_backtesting_selecao.ipynb) | Backtesting | rolling-origin, CRPS, cobertura e auditoria |
| [08](notebooks/08_sistema_integrado.ipynb) | Primeiro sistema integrado | contratos, registry, artefatos e risco |
| [09](notebooks/09_estado_espaco_kalman.ipynb) | Estado-espaço | filtro, smoother, gaps e forecast |
| [10](notebooks/10_markov_switching.ipynb) | Regimes | Hamilton/Kim, MLE e mistura preditiva |
| [11](notebooks/11_change_point.ipynb) | Mudanças estruturais | CUSUM retrospectivo e segmentação penalizada |
| [12](notebooks/12_cusum_ewma.ipynb) | Detecção sequencial | ARL, falso alarme, atraso, CUSUM e EWMA |
| [13](notebooks/13_sistema_final.ipynb) | Sistema final | seleção, persistência, monitoramento, risco e API |
| [14](notebooks/14_ciclo_completo_usuario.ipynb) | Ciclo guiado | uma `pandas.Series` do usuário percorre caracterização, seleção, forecast, risco e monitoramento |

Cada notebook possui plano em `docs/plano_notebook_XX.md`, funções matemáticas explícitas,
exercícios, gabarito e testes rápidos.

## 12. Instalação local no macOS

Instale as ferramentas de linha de comando da Apple e o `uv`:

```bash
xcode-select --install
brew install uv
```

Clone e prepare o ambiente:

```bash
git clone https://github.com/rlsgarcia-code/time-series-forecast.git
cd time-series-forecast
uv sync
```

Abra os notebooks:

```bash
uv run jupyter lab
```

O ambiente é reproduzido pelas versões registradas em `uv.lock`.

## 13. Verificação

Execute testes e todos os notebooks em kernels limpos:

```bash
make check
```

Ou apenas a suíte automatizada:

```bash
uv run pytest
```

Para validar um notebook específico:

```bash
uv run python scripts/validate_notebook.py notebooks/14_ciclo_completo_usuario.ipynb
```

## 14. Estrutura

```text
time-series-forecast/
├── src/time_series_lab/        # motor, contratos, modelos e diagnósticos
├── examples/                   # adaptadores didáticos, não produtivos
├── notebooks/                  # formulação, experimentos e exercícios
├── tests/                      # testes unitários e de integração
├── scripts/                    # geradores e validação dos notebooks
├── docs/                       # planos, referência e visão do sistema
├── pyproject.toml              # pacote e dependências
└── uv.lock                     # ambiente reproduzível
```

## 15. Limites atuais

- séries univariadas, regulares e sem valores ausentes no contrato global;
- candidatos e ordens ARIMA definidos explicitamente pela aplicação;
- diagnóstico residual ainda não é guardrail automático de seleção;
- `MemoryRepository` é apenas didático;
- paralelismo, banco transacional, autenticação e agendamento pertencem à aplicação hospedeira;
- detecção de mudança não implica antecipação causal antes de surgirem sinais observáveis.

O padrão didático do projeto está em [`docs/padrao_didatico.md`](docs/padrao_didatico.md), e os
critérios de conclusão da trilha estão em [`docs/plano_laboratorio.md`](docs/plano_laboratorio.md).
