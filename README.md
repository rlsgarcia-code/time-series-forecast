# Time Series Lab

Laboratório didático e reproduzível, em português, para aprender séries temporais por meio de
simulação, visualização e testes.

## Motor Python

A versão 0.2 adiciona `ForecastEngine`, uma fachada importável para aplicações gerenciarem várias
séries univariadas, modelos, backtests, forecasts probabilísticos, simulações, risco e
monitoramento causal. A persistência é injetada por meio de `EngineRepository`; a biblioteca não
impõe banco ou serviço externo.

```python
from time_series_lab import ForecastEngine, TimeSeries, AR1Spec

engine = ForecastEngine(repository=my_repository)
engine.register_series(TimeSeries.from_pandas("demanda", pandas_series))
run = engine.fit("demanda", AR1Spec())
engine.promote("demanda", run.model_run_id)
forecast = engine.forecast("demanda", horizon=12)
```

Para embasar a escolha e inspecionar o modelo:

```python
profile = engine.characterize("demanda")
diagnostics = engine.diagnose("demanda", model_ref=run.model_run_id)
characterization_figure = engine.plot_characterization("demanda")
residual_figure = engine.plot_residuals("demanda", model_ref=run.model_run_id)
```

Os resultados expõem valores, diferenças, ACF/PACF, testes de estacionariedade, resíduos,
resíduos padronizados, Q–Q, testes de autocorrelação e volatilidade condicional.

A referência completa, ciclo de monitoramento e interfaces de extensão estão em
[`docs/engine_api.md`](docs/engine_api.md). Um repositório exclusivamente didático está em
[`examples/memory_repository.py`](examples/memory_repository.py).

O roteiro completo, a arquitetura-alvo e os critérios de conclusão estão em
[`docs/plano_laboratorio.md`](docs/plano_laboratorio.md).
O padrão usado para introduzir e aprofundar conceitos está em
[`docs/padrao_didatico.md`](docs/padrao_didatico.md).

## Notebook 01

O arquivo [`notebooks/01_processos_estocasticos.ipynb`](notebooks/01_processos_estocasticos.ipynb)
cobre:

- ruído branco e momentos amostrais;
- passeio aleatório, diferenças e variância crescente;
- visualização conjunta de 100 trajetórias;
- AR(1), estacionariedade e um slider interativo de `phi`;
- ACF amostral versus ACF teórica;
- resposta a choques, persistência e meia-vida;
- forecast probabilístico analítico validado por Monte Carlo;
- exercícios, gabarito executável e testes rápidos.

O plano didático e as escolhas técnicas estão documentados em
[`docs/plano_notebook_01.md`](docs/plano_notebook_01.md).

## Notebook 02

O arquivo [`notebooks/02_estimacao_diagnostico_ar1.ipynb`](notebooks/02_estimacao_diagnostico_ar1.ipynb)
cobre:

- AR(1) como regressão entre pares consecutivos;
- estimação explícita de `c`, `phi` e `sigma` por mínimos quadrados;
- erros-padrão, intervalos de confiança e distribuição amostral;
- efeito do tamanho da amostra e persistência próxima à raiz unitária;
- diagnóstico residual com ACF, Q–Q plot e Ljung–Box;
- contraste entre modelo correto e AR(1) mal especificado para dados AR(2);
- avaliação temporal com origem fixa, rolling one-step e baseline ingênuo.

O plano específico está em [`docs/plano_notebook_02.md`](docs/plano_notebook_02.md).

## Notebooks 03–13

| Nº | Tema | Entrega principal |
|---:|---|---|
| [03](notebooks/03_arp_maq_arma.ipynb) | AR(p), MA(q) e ARMA | ACF/PACF, AIC, resíduos e seleção de ordens |
| [04](notebooks/04_nao_estacionariedade_arima.ipynb) | Não estacionariedade e ARIMA | ADF, diferenciação, inversão e regressão espúria |
| [05](notebooks/05_browniano_gbm_ito.ipynb) | Browniano, GBM e Itô | variação quadrática, Euler e solução exata |
| [06](notebooks/06_arch_garch.ipynb) | ARCH/GARCH | MLE, resíduos padronizados, volatilidade e VaR |
| [07](notebooks/07_backtesting_selecao.ipynb) | Backtesting probabilístico | rolling-origin, CRPS, PIT, calibração e auditoria |
| [08](notebooks/08_sistema_integrado.ipynb) | Sistema integrado | contratos, registry, artefatos, monitoramento e risco |
| [09](notebooks/09_estado_espaco_kalman.ipynb) | Estado-espaço e Kalman | estado latente, filtro, smoother, MLE e forecast |
| [10](notebooks/10_markov_switching.ipynb) | Markov switching | regimes ocultos, Hamilton/Kim, MLE e misturas |
| [11](notebooks/11_change_point.ipynb) | Change-point detection | CUSUM, custos segmentados, penalização e programação dinâmica |
| [12](notebooks/12_cusum_ewma.ipynb) | CUSUM e EWMA sequenciais | ARL, falso alarme, atraso, calibração e política de alerta |
| [13](notebooks/13_sistema_final.ipynb) | Sistema integrado final | artefatos imutáveis, monitor, risco, alertas e auditoria |

Cada notebook possui um plano correspondente em `docs/plano_notebook_XX.md`, funções centrais
explícitas, exercícios, gabarito e testes rápidos.

### Continuação da trilha

O primeiro ciclo (01–08) produziu o protótipo educacional. A segunda fase amplia os componentes
necessários para a visão completa do sistema:

| Nº | Tema | Estado |
|---:|---|---|
| 09 | Estado-espaço e filtro de Kalman | concluído |
| 10 | Regimes e Markov switching | concluído |
| 11 | Quebras estruturais e change-point | concluído |
| 12 | Detecção sequencial com CUSUM e EWMA | concluído |
| 13 | Sistema integrado com detectores e políticas de alerta | concluído |

A decisão de manter 03 e 04 como capítulos coesos, com opções de complementos avançados, está em
[`docs/avaliacao_desdobramento_03_04.md`](docs/avaliacao_desdobramento_03_04.md).

## Instalação local no macOS

### 1. Instale as ferramentas

No Terminal, confirme que o Git e as ferramentas de linha de comando da Apple estão disponíveis:

```bash
xcode-select --install
```

Instale o `uv` pelo Homebrew:

```bash
brew install uv
```

Se você não usa Homebrew, consulte a instalação oficial do `uv` e depois volte a estas instruções.

### 2. Prepare o ambiente

Entre na raiz do repositório e execute:

```bash
uv sync
```

O comando cria `.venv`, instala o projeto e usa as versões registradas em `uv.lock`. Não é
necessário ativar o ambiente manualmente.

### 3. Abra o laboratório

```bash
uv run jupyter lab
```

No navegador, abra os notebooks em ordem e use **Run → Run All Cells**.
O kernel esperado aparece como **Python 3 (ipykernel)**. O slider funciona no JupyterLab sem
extensão adicional.

Para encerrar o servidor, volte ao Terminal e pressione `Control+C`.

## Verificação

Execute a suíte automatizada:

```bash
uv run pytest
```

Execute o notebook inteiro em um kernel limpo, sem alterar o arquivo original:

```bash
uv run python scripts/validate_notebook.py notebooks/01_processos_estocasticos.ipynb
uv run python scripts/validate_notebook.py notebooks/02_estimacao_diagnostico_ar1.ipynb
uv run python scripts/validate_notebook.py notebooks/03_arp_maq_arma.ipynb
uv run python scripts/validate_notebook.py notebooks/04_nao_estacionariedade_arima.ipynb
uv run python scripts/validate_notebook.py notebooks/05_browniano_gbm_ito.ipynb
uv run python scripts/validate_notebook.py notebooks/06_arch_garch.ipynb
uv run python scripts/validate_notebook.py notebooks/07_backtesting_selecao.ipynb
uv run python scripts/validate_notebook.py notebooks/08_sistema_integrado.ipynb
uv run python scripts/validate_notebook.py notebooks/09_estado_espaco_kalman.ipynb
uv run python scripts/validate_notebook.py notebooks/10_markov_switching.ipynb
uv run python scripts/validate_notebook.py notebooks/11_change_point.ipynb
uv run python scripts/validate_notebook.py notebooks/12_cusum_ewma.ipynb
uv run python scripts/validate_notebook.py notebooks/13_sistema_final.ipynb
```

Ou faça as duas verificações:

```bash
make check
```

## Estrutura

```text
time-series-lab/
├── notebooks/                  # narrativa, experimentos e exercícios
├── src/time_series_lab/        # simulações e fórmulas testáveis
├── tests/                      # testes unitários e de estrutura
├── scripts/                    # construção e validação do notebook
├── docs/                       # plano didático/técnico
├── pyproject.toml              # dependências e configuração
└── uv.lock                     # resolução exata do ambiente
```

## Desenvolvimento

Cada notebook possui um gerador `scripts/build_notebook_XX.py`, preservando fontes revisáveis para
suas células. Após alterar um gerador, reconstrua e valide:

```bash
make rebuild-notebooks
make check
```

As simulações seguem a convenção de incluir `t=0` em processos recursivos: um passeio de
`n_steps=100` retorna 101 observações. Ruído branco retorna exatamente `n_steps` inovações.

