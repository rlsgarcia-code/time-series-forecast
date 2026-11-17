"""Constrói o Notebook 14: ciclo completo a partir de uma pandas Series."""

import json
from pathlib import Path

from notebook_builder import code, md, setup_cell, write_notebook


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "14_ciclo_completo_usuario.ipynb"


CELLS = [
    md(r"""
    # Notebook 14 — Ciclo completo com uma `pandas.Series`

    **Público:** quem já conhece o vocabulário básico de séries temporais e quer operar o motor
    completo sem esconder os critérios estatísticos.

    **Pré-requisitos:** uma `pandas.Series` univariada, numérica, ordenada por um
    `DatetimeIndex`. O notebook adiciona UTC quando o índice não possui timezone, mas não inventa
    observações para gaps.

    **Ao final você será capaz de:**

    1. validar o contrato temporal;
    2. caracterizar a série e ler as evidências;
    3. declarar exatamente quais modelos serão avaliados;
    4. selecionar por rolling-origin, cobertura e CRPS;
    5. examinar resíduos, forecast e risco;
    6. revelar observações futuras uma a uma e monitorar drift sem vazamento.

    > A única célula obrigatoriamente editável está marcada com **EDITE SOMENTE ESTA CÉLULA**.
    > As demais podem ser executadas de cima para baixo com **Run → Run All Cells**.
    """),
    md(r"""
    ## 1. Mapa causal

    O histórico será dividido antes do ajuste:

    $$\text{desenvolvimento}\;|\;\text{cauda reservada para monitoramento}.$$

    A parte de desenvolvimento passa por caracterização, backtest e seleção. A cauda não participa
    dessas decisões: ela é revelada somente depois que o modelo e a política estão congelados.

    ```text
    carregar → validar → separar → caracterizar → candidatos → backtest → selecionar
                                                             ↓
    monitorar ← revelar cauda ← emitir forecast ← risco/simulação ← diagnosticar
    ```
    """),
    setup_cell(
        "from dataclasses import asdict\n"
        "from examples.memory_repository import MemoryRepository\n"
        "from time_series_lab import (\n"
        "    AR1Spec, ARIMASpec, BacktestConfig, CandidatePolicy, ForecastEngine, GARCH11Spec,\n"
        "    LocalLevelSpec, MonitoringPolicy, Observation, SelectionPolicy,\n"
        "    ThresholdRisk, TimeSeries, TwoRegimeGaussianSpec, ValidationError,\n"
        ")"
    ),
    md(r"""
    ## 2. Carregue sua série

    Substitua o corpo de `load_user_series()` pelo código que lê seu CSV, banco ou API. O retorno
    deve ser uma `pd.Series`; seu índice representa o tempo e seus valores representam uma única
    variável.

    O exemplo padrão simula somente 57 observações de um AR(1), inclui três valores ausentes e
    coloca um pequeno deslocamento apenas na cauda. Assim o próprio exemplo valida o caso curto
    que motivou esta revisão antes de você conectar seus dados.
    """),
    code(r"""
    # ==================== EDITE SOMENTE ESTA CÉLULA ====================
    def load_user_series() -> pd.Series:
        # Exemplo para CSV:
        # frame = pd.read_csv("meus_dados.csv", parse_dates=["timestamp"])
        # return frame.set_index("timestamp")["valor"].sort_index()

        rng = np.random.default_rng(1401)
        index = pd.date_range("2024-01-01", periods=57, freq="D", tz="UTC")
        values = np.empty(len(index)); values[0] = 0.0
        for t in range(1, len(index)):
            late_shift = 0.75 if t >= 47 else 0.0
            values[t] = 0.15 + late_shift + 0.72 * values[t-1] + rng.normal(0, 0.65)
        values[[8, 21, 35]] = np.nan
        return pd.Series(values, index=index, name="valor")

    USER_SERIES = load_user_series()
    if not isinstance(USER_SERIES, pd.Series):
        raise TypeError("load_user_series() deve retornar uma pandas.Series")
    USER_SERIES.head()
    """),
    md(r"""
    ## 3. Configuração do experimento

    - `MONITORING_HOLDOUT` define quantas observações finais ficam invisíveis durante a escolha.
    - `MISSING_VALUE_POLICY="ffill"` usa apenas o passado e é compatível com avaliação causal.
      `"interpolate"` usa os dois lados do ponto ausente e deve ser uma decisão consciente.
    - `CANDIDATE_DIFFERENCING` controla se $d=1$ depende dos testes (`auto`), sempre concorre
      (`compare`) ou é proibido (`never`).
    - GARCH e Markov switching usam gates explicáveis; `auto` exige evidência e amostra mínima.
      `always` força a hipótese, mas ainda deixa o backtest decidir.
    - O fallback de cobertura é explícito: se nenhum candidato passar o guardrail, podemos parar ou
      selecionar apenas por CRPS, registrando que o guardrail foi relaxado.
    """),
    code(r"""
    SERIES_ID = "serie-usuario"
    FREQUENCY = None             # exemplo: "D"; None tenta inferir
    TIMEZONE = "UTC"            # usado apenas se o índice for ingênuo
    SERIES_KIND = "level"       # "level" ou "returns"
    MISSING_VALUE_POLICY = "ffill"  # "error", "ffill" ou "interpolate"
    MONITORING_HOLDOUT = 10
    FORECAST_HORIZON = 12
    N_SIMULATION_PATHS = 5_000
    RISK_LIMIT = None            # None usa média + 2 desvios do desenvolvimento
    CANDIDATE_DIFFERENCING = "auto"  # "auto", "compare" ou "never"
    CANDIDATE_GARCH = "auto"         # "auto", "always" ou "never"
    CANDIDATE_MARKOV = "auto"        # "auto", "always" ou "never"
    MAX_CANDIDATES = 12
    ALLOW_COVERAGE_FALLBACK = True
    RANDOM_SEED = 1401

    if SERIES_KIND not in {"level", "returns"}:
        raise ValueError("SERIES_KIND deve ser 'level' ou 'returns'")
    if MISSING_VALUE_POLICY not in {"error", "ffill", "interpolate"}:
        raise ValueError("MISSING_VALUE_POLICY deve ser 'error', 'ffill' ou 'interpolate'")
    """),
    md(r"""
    ## 4. Contrato temporal e separação

    A validação ocorre antes de qualquer teste estatístico. Duplicações, gaps e frequência
    incompatível interrompem o fluxo, pois ACF, horizonte e forecast dependem de uma noção temporal
    coerente. `NaN` e infinitos são contabilizados e tratados segundo a política escolhida.

    Valores ausentes no começo ou no fim são removidos: não existe observação passada para preencher
    o início, e preencher o fim inventaria os dados mais recentes. Valores internos usam `ffill` por
    padrão, isto é, $\tilde y_t=y_{t-1}$ quando $y_t$ está ausente. Isso não olha o futuro, embora
    ainda seja uma hipótese de imputação que deve constar na auditoria. A interpolação temporal é
    oferecida, mas usa informação posterior e por isso não é a escolha padrão.

    O mínimo agora é adaptativo: 24 valores de desenvolvimento para modelos de nível e 34 quando
    GARCH entra na comparação. Amostras curtas produzem menos origens, intervalos mais incertos e
    testes com menor poder; elas não são rejeitadas apenas por terem menos de 80 pontos.
    """),
    code(r"""
    source = USER_SERIES.copy()
    if not isinstance(source.index, pd.DatetimeIndex):
        raise TypeError("USER_SERIES.index deve ser um DatetimeIndex")
    source = source.sort_index()
    if not source.index.is_unique:
        raise ValueError("O índice possui timestamps duplicados; agregue-os antes da modelagem")
    if source.index.tz is None:
        source.index = source.index.tz_localize(TIMEZONE)

    source_frequency = FREQUENCY or pd.infer_freq(source.index)
    if source_frequency is None:
        raise ValueError("Não foi possível inferir a frequência; preencha FREQUENCY, por exemplo 'D'")
    expected_index = pd.date_range(source.index[0], periods=len(source), freq=source_frequency)
    if not source.index.equals(expected_index):
        raise ValueError("O índice possui gaps; regularize/resample a série antes da modelagem")

    numeric_source = pd.to_numeric(source, errors="coerce").astype(float)
    numeric_source = numeric_source.replace([np.inf, -np.inf], np.nan)
    valid_positions = np.flatnonzero(numeric_source.notna().to_numpy())
    if valid_positions.size == 0:
        raise ValueError("A série não possui nenhum valor numérico finito")

    first_valid, last_valid = int(valid_positions[0]), int(valid_positions[-1])
    trimmed = numeric_source.iloc[first_valid:last_valid + 1].copy()
    missing_before = trimmed.isna()
    if missing_before.any() and MISSING_VALUE_POLICY == "error":
        examples = [str(value) for value in trimmed.index[missing_before][:5]]
        raise ValueError(f"Há valores ausentes; exemplos: {examples}")
    if MISSING_VALUE_POLICY == "ffill":
        raw = trimmed.ffill()
    elif MISSING_VALUE_POLICY == "interpolate":
        print("ATENÇÃO: interpolate usa observações futuras ao ponto ausente.")
        raw = trimmed.interpolate(method="time", limit_area="inside")
    else:
        raw = trimmed
    if raw.isna().any():
        raise ValueError("A política escolhida deixou valores ausentes sem tratamento")

    imputed_mask = missing_before.reindex(raw.index, fill_value=False)
    cleaning_summary = pd.Series({
        "observações recebidas": len(source),
        "ausentes/não finitos recebidos": int(numeric_source.isna().sum()),
        "removidos nas bordas": len(source) - len(trimmed),
        "imputados internamente": int(imputed_mask.sum()),
        "política": MISSING_VALUE_POLICY,
        "frequência": source_frequency,
    }, name="limpeza")
    cleaning_detail = pd.DataFrame({
        "original": numeric_source.reindex(raw.index),
        "tratado": raw,
        "foi_imputado": imputed_mask,
    })
    display(cleaning_summary.to_frame(), cleaning_detail[cleaning_detail.foi_imputado])

    minimum_development = 34 if SERIES_KIND == "returns" else 24
    if len(raw) < minimum_development + 1:
        raise ValueError(
            f"Após a limpeza são necessários ao menos {minimum_development + 1} valores; "
            f"foram obtidos {len(raw)}"
        )
    if not 1 <= MONITORING_HOLDOUT <= len(raw) - minimum_development:
        raise ValueError(
            f"MONITORING_HOLDOUT deve preservar ao menos {minimum_development} "
            "observações de desenvolvimento"
        )

    development_pd = raw.iloc[:-MONITORING_HOLDOUT]
    monitoring_pd = raw.iloc[-MONITORING_HOLDOUT:]
    development = TimeSeries.from_pandas(
        SERIES_ID, development_pd, frequency=source_frequency, timezone_name=TIMEZONE,
        metadata={
            "series_kind": SERIES_KIND, "source": "notebook-14",
            "missing_value_policy": MISSING_VALUE_POLICY,
            "imputed_count": int(imputed_mask.sum()),
        },
    )

    repository = MemoryRepository()
    engine = ForecastEngine(repository=repository)
    engine.register_series(development)

    pd.DataFrame({
        "parte": ["desenvolvimento", "monitoramento reservado"],
        "observações": [len(development_pd), len(monitoring_pd)],
        "início": [development_pd.index[0], monitoring_pd.index[0]],
        "fim": [development_pd.index[-1], monitoring_pd.index[-1]],
    })
    """),
    code(r"""
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(raw.index, raw.values, color="black", lw=1, label="série carregada")
    ax.axvspan(monitoring_pd.index[0], monitoring_pd.index[-1], color=COLORS["orange"],
               alpha=.18, label="cauda ainda não revelada")
    ax.axvline(development_pd.index[-1], color=COLORS["red"], ls="--", label="cutoff")
    ax.set_title("Separação definida antes da modelagem"); ax.legend(); plt.show()
    """),
    md(r"""
    ## 5. Caracterização: evidências, não decisão final

    A caracterização combina nível, diferenças, ACF/PACF, distribuição, tendência, ADF, KPSS,
    Ljung–Box, ARCH-LM e um sinal descritivo de quebra. `suggested_models` indica famílias que
    merecem entrar na comparação; não escolhe ordens nem promove modelos.

    Como ler os principais testes:

    - **ADF:** $H_0$ afirma raiz unitária; $p$ pequeno é evidência contra não estacionariedade;
    - **KPSS:** $H_0$ afirma estacionariedade; $p$ pequeno é evidência contra estacionariedade;
    - **Ljung–Box:** $H_0$ afirma ausência conjunta de autocorrelação até a defasagem testada;
    - **ARCH-LM:** $H_0$ afirma ausência de dependência na variância condicional.

    ACF e PACF são estimativas amostrais sujeitas a incerteza. Elas descrevem memória linear, mas
    não provam sozinhas qual processo gerou os dados. ADF e KPSS têm hipóteses nulas opostas;
    interpretá-los juntos é mais informativo do que tratar um único $p$-valor como diagnóstico.
    """),
    code(r"""
    profile = engine.characterize(SERIES_ID, max_lag=24)
    summary = pd.Series(profile.summary, name="valor").to_frame()
    evidence = pd.DataFrame([asdict(item) for item in profile.evidence])
    display(summary.round(4), evidence, pd.Series(profile.suggested_models, name="famílias sugeridas"))
    """),
    code(r"""
    engine.plot_characterization(SERIES_ID, max_lag=24)
    plt.show()
    """),
    md(r"""
    ## 6. Candidate Policy: da evidência às hipóteses

    A lista abaixo continua sendo a fonte de verdade do backtest, mas agora é produzida por uma
    política pública e auditável. A caracterização **não escolhe o vencedor**: ela decide quais
    hipóteses merecem competir e registra também por que outras foram excluídas.

    A política sempre mantém benchmarks simples, usa ACF/PACF para limitar ordens, usa ADF/KPSS
    para abrir ou podar $d=1$, e aplica gates amostrais/conceituais a GARCH e Markov switching.
    `MAX_CANDIDATES` limita a busca porque testar uma grade enorme na mesma amostra também produz
    sobreajuste de seleção.

    As famílias correspondem às formulações:

    $$\text{AR(1):}\quad Y_t=c+\phi Y_{t-1}+\varepsilon_t,$$

    $$\text{ARIMA}(p,d,q):\quad
      \phi(B)(1-B)^dY_t=c+\theta(B)\varepsilon_t,$$

    $$\text{nível local:}\quad
      Y_t=\mu_t+\varepsilon_t,\qquad \mu_t=\mu_{t-1}+\eta_t,$$

    $$\text{GARCH(1,1):}\quad
      \sigma_t^2=\omega+\alpha\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2.$$

    Em Markov switching, os parâmetros dependem de um estado latente $S_t$ cuja evolução obedece
    a $P(S_t=j\mid S_{t-1}=i)=p_{ij}$. Cada equação introduz uma hipótese diferente sobre memória,
    tendência, variância ou regimes; por isso todas precisam ser julgadas fora da amostra.
    """),
    code(r"""
    candidate_policy = CandidatePolicy(
        differencing=CANDIDATE_DIFFERENCING,
        garch=CANDIDATE_GARCH,
        markov_switching=CANDIDATE_MARKOV,
        max_candidates=MAX_CANDIDATES,
        max_ar_order=2,
        max_ma_order=2,
        max_total_order=3,
        simulation_paths=300,
    )
    candidate_plan = engine.propose_candidates(
        SERIES_ID,
        policy=candidate_policy,
        profile=profile,
        series_kind=SERIES_KIND,
    )
    candidates = list(candidate_plan.candidates)

    display(
        pd.Series(candidate_plan.summary, name="valor").to_frame(),
        candidate_plan.to_frame(include_excluded=True),
    )
    print(candidate_plan.explain())
    """),
    md(r"""
    ## 7. Backtest rolling-origin

    Em cada origem, todos os candidatos recebem o mesmo histórico anterior ao cutoff e são
    avaliados nos mesmos realizados. O horizonte e o passo são escolhidos em função do tamanho do
    desenvolvimento para que o notebook permaneça executável em séries de escalas diferentes.

    Métricas:

    - MAE/RMSE avaliam o centro da distribuição;
    - cobertura e largura avaliam calibração e concentração do intervalo;
    - CRPS avalia a distribuição inteira e é a métrica primária.

    Para erros $e_i=y_i-\hat y_i$,

    $$\operatorname{MAE}=\frac1n\sum_i|e_i|,\qquad
      \operatorname{RMSE}=\sqrt{\frac1n\sum_i e_i^2}.$$

    A cobertura observada é a fração de realizados dentro de $[L_i,U_i]$. Já o CRPS compara toda
    a CDF preditiva $F_i$ com o realizado:

    $$\operatorname{CRPS}(F_i,y_i)=
      \int_{-\infty}^{\infty}\left(F_i(z)-\mathbf{1}\{y_i\le z\}\right)^2\,dz.$$

    Menor CRPS é melhor, mas somente entre candidatos que respeitem os guardrails definidos.
    """),
    code(r"""
    n_development = len(development_pd)
    minimum_train_size = 30 if SERIES_KIND == "returns" else 20
    backtest_horizon = min(6, max(1, n_development // 15))
    initial_train_size = min(
        n_development - backtest_horizon,
        max(minimum_train_size, int(0.65 * n_development)),
    )
    backtest_step = max(1, backtest_horizon)
    backtest_config = BacktestConfig(
        initial_train_size=initial_train_size,
        horizon=backtest_horizon,
        step=backtest_step,
        confidence=0.95,
        seed=RANDOM_SEED,
    )
    report = engine.backtest(SERIES_ID, candidates, backtest_config)

    label_by_id = {
        score.model_run_family: repr(spec)
        for score, spec in zip(report.scores, report.candidates)
    }
    scoreboard = pd.DataFrame({
        label_by_id[score.model_run_family]: score.metrics for score in report.scores
    }).T.sort_values("crps")
    scoreboard.round(4)
    """),
    code(r"""
    rows = pd.DataFrame(report.rows)
    rows["candidate"] = rows["candidate_id"].map(label_by_id)
    # Cobertura por horizonte é calculada explicitamente para não esconder o intervalo usado.
    coverage_by_horizon = rows.assign(
        covered=lambda frame: (frame.actual >= frame.lower) & (frame.actual <= frame.upper)
    ).pivot_table(index="candidate", columns="horizon", values="covered", aggfunc="mean")
    coverage_by_horizon.round(3)
    """),
    md(r"""
    ## 8. Seleção e promoção

    A política padrão exige cobertura agregada entre 80% e 100% e escolhe o menor CRPS entre os
    elegíveis. Se ninguém passar, o notebook só relaxa o guardrail quando
    `ALLOW_COVERAGE_FALLBACK=True`; o resultado registra essa decisão para não confundir fallback
    com aprovação estatística.
    """),
    code(r"""
    selection_policy = SelectionPolicy(primary_metric="crps", coverage_range=(0.80, 1.00))
    coverage_fallback_used = False
    try:
        decision = engine.select(report, selection_policy)
    except ValidationError as error:
        if not ALLOW_COVERAGE_FALLBACK:
            raise
        coverage_fallback_used = True
        print("ATENÇÃO:", error)
        print("Fallback autorizado: seleção por CRPS sem guardrail de cobertura.")
        selection_policy = SelectionPolicy(primary_metric="crps", coverage_range=None)
        decision = engine.select(report, selection_policy)

    promotion = engine.promote(
        SERIES_ID, decision.model_run_id, decision_id=decision.decision_id
    )
    selected_run = engine.get_champion(SERIES_ID)
    pd.Series({
        "modelo selecionado": repr(decision.selected_spec),
        "métrica primária": decision.primary_metric,
        "CRPS": decision.scores["crps"],
        "cobertura": decision.scores["coverage"],
        "fallback de cobertura": coverage_fallback_used,
        "model_run_id": selected_run.model_run_id,
    })
    """),
    md(r"""
    ## 9. Diagnóstico residual

    Seleção fora da amostra e diagnóstico residual respondem a perguntas diferentes. O backtest
    pergunta “quem prevê melhor?”. Os resíduos perguntam “que estrutura o vencedor ainda deixou
    para trás?”. Ljung–Box, memória nos quadrados, ARCH-LM e Q–Q não são guardrails automáticos
    nesta versão; eles fundamentam revisão do conjunto de candidatos.
    """),
    code(r"""
    diagnostics = engine.diagnose(SERIES_ID, model_ref=selected_run.model_run_id, max_lag=24)
    display(pd.Series(diagnostics.tests, name="valor").round(4).to_frame())
    for interpretation in diagnostics.interpretation:
        print("•", interpretation)
    diagnostics.to_frame().tail()
    """),
    code(r"""
    engine.plot_residuals(SERIES_ID, model_ref=selected_run.model_run_id, max_lag=24)
    plt.show()
    """),
    md(r"""
    ## 10. Forecast probabilístico

    O resultado não é apenas uma linha. `PredictiveDistribution` oferece média, variância,
    quantis, CDF e trajetórias. A faixa abaixo usa quantis de 2,5% e 97,5%; portanto ela não precisa
    assumir que toda distribuição é Normal.

    No caso AR(1), a previsão de um passo é aplicada recursivamente. Para
    $Y_t=c+\phi Y_{t-1}+\varepsilon_t$, com
    $\varepsilon_t\sim\mathcal N(0,\sigma^2)$,

    $$\mathbb E_T[Y_{T+h}]
      =\phi^hY_T+c\frac{1-\phi^h}{1-\phi},$$

    $$\operatorname{Var}_T(Y_{T+h})
      =\sigma^2\sum_{j=0}^{h-1}\phi^{2j}.$$

    O primeiro termo da média é a memória do último valor; o segundo converge para a média de
    longo prazo $c/(1-\phi)$ quando $|\phi|<1$. A variância acumula os choques futuros, ponderados
    pela persistência. Para $\phi=1$, usa-se o limite: média $Y_T+ch$ e variância $h\sigma^2$.
    Outros modelos implementam a mesma interface, embora sua distribuição seja calculada por
    recorrências de estado, mistura ou simulação.
    """),
    code(r"""
    forecast = engine.forecast(
        SERIES_ID, horizon=FORECAST_HORIZON, model_ref=selected_run.model_run_id,
        seed=RANDOM_SEED,
    )
    quantiles = forecast.distribution.quantile([0.025, 0.50, 0.975])
    forecast_frame = pd.DataFrame({
        "mean": forecast.distribution.mean,
        "q025": quantiles[0],
        "median": quantiles[1],
        "q975": quantiles[2],
    }, index=forecast.targets)
    forecast_frame
    """),
    code(r"""
    fig, ax = plt.subplots(figsize=(13, 4.5))
    tail = development_pd.iloc[-min(80, len(development_pd)):]
    ax.plot(tail.index, tail.values, color="black", label="histórico")
    ax.plot(forecast_frame.index, forecast_frame["median"], color=COLORS["blue"], label="mediana")
    ax.fill_between(forecast_frame.index, forecast_frame.q025, forecast_frame.q975,
                    color=COLORS["blue"], alpha=.18, label="intervalo 95%")
    ax.axvline(development_pd.index[-1], color=COLORS["red"], ls="--", label="cutoff")
    ax.set_title("Forecast emitido antes de revelar a cauda"); ax.legend(); plt.show()
    """),
    md(r"""
    ## 11. Simulação e risco de trajetória

    O risco marginal $P(Y_{T+h}>L)$ é diferente do evento conjunto
    $P(\max_hY_{T+h}>L)$. Somar probabilidades marginais contaria a mesma trajetória várias vezes.
    Por isso o segundo evento é estimado sobre caminhos conjuntos.
    """),
    code(r"""
    simulation = engine.simulate(
        SERIES_ID, horizon=FORECAST_HORIZON, n_paths=N_SIMULATION_PATHS,
        model_ref=selected_run.model_run_id, seed=RANDOM_SEED,
    )
    risk_limit = (
        float(RISK_LIMIT) if RISK_LIMIT is not None
        else profile.summary["mean"] + 2 * profile.summary["std"]
    )
    risk = engine.assess_risk(
        simulation, ThresholdRisk(limit=risk_limit, direction="above", seed=RANDOM_SEED)
    )
    risk_frame = pd.DataFrame({
        "marginal": risk.marginal_probability,
        "primeira_passagem": risk.first_passage_probability,
    }, index=simulation.targets)
    display(pd.Series({
        "limite": risk_limit,
        "probabilidade em qualquer horizonte": risk.any_horizon_probability,
    }).round(4), risk_frame.round(4))
    """),
    code(r"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for path in simulation.paths[:80]:
        axes[0].plot(simulation.targets, path, color=COLORS["blue"], alpha=.06)
    axes[0].axhline(risk_limit, color=COLORS["red"], ls="--", label="limite")
    axes[0].set_title("Amostra de trajetórias futuras"); axes[0].legend()
    axes[1].plot(risk_frame.index, risk_frame.marginal, marker="o", label="marginal")
    axes[1].plot(risk_frame.index, risk_frame.primeira_passagem, marker="o", label="1ª passagem")
    axes[1].set_ylim(0, 1); axes[1].set_title("Probabilidades por horizonte"); axes[1].legend()
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ## 12. Change-point retrospectivo

    A segmentação abaixo usa todo o desenvolvimento e, portanto, é retrospectiva. Ela ajuda a
    investigar quebras, mas não pode ser apresentada como um alerta que existia antes dos dados.
    A penalidade escala com a variância e com $\log(n)$ para evitar comparar SSEs em unidades
    incompatíveis.
    """),
    code(r"""
    change_penalty = 2 * np.var(development.values, ddof=1) * np.log(len(development.values))
    change_report = engine.detect_changes(
        SERIES_ID, penalty=float(change_penalty), min_size=max(10, len(development.values)//20)
    )
    print("Quebras retrospectivas:", [str(value) for value in change_report.breakpoints])

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(development.timestamps, development.values, color="black", lw=1)
    for point in change_report.breakpoints:
        ax.axvline(point, color=COLORS["red"], ls="--", alpha=.8)
    ax.set_title("Change-points estimados usando todo o desenvolvimento"); plt.show()
    """),
    md(r"""
    ## 13. Monitoramento prequential na cauda reservada

    Agora congelamos a política e iniciamos a sessão. O motor já emite o forecast do próximo
    instante. Para cada ponto reservado ele primeiro recupera essa distribuição, calcula a surpresa
    e atualiza os detectores; só depois anexa o realizado, reajusta o mesmo modelo e emite o próximo
    forecast.

    A surpresa usa $\Phi^{-1}(F_t(y_t))$. Para Normal isso coincide com o resíduo padronizado; para
    mistura de regimes usa a CDF completa.
    """),
    code(r"""
    monitoring_policy = MonitoringPolicy(
        point_threshold=3.0,
        cusum_reference=0.4,
        cusum_threshold=6.0,
        ewma_smoothing=0.15,
        ewma_limit_multiplier=3.6,
        cooldown=3,
        reset_on_alert=True,
        refit_each_observation=True,
        forecast_horizon=1,
    )
    session = engine.start_monitoring(SERIES_ID, monitoring_policy)
    monitoring_rows = []
    last_cycle = None
    for number, (timestamp, value) in enumerate(monitoring_pd.items(), start=1):
        last_cycle = engine.process_observation(
            session.session_id,
            Observation(timestamp=timestamp, value=float(value),
                        observation_id=f"notebook-14-{number:03d}"),
        )
        event = last_cycle.monitoring_event
        monitoring_rows.append({
            "timestamp": timestamp,
            "actual": event.actual,
            "observation_was_imputed": bool(imputed_mask.loc[timestamp]),
            "surprise": event.surprise,
            "cusum_positive": event.cusum_positive,
            "cusum_negative": event.cusum_negative,
            "ewma": event.ewma,
            "ewma_limit": event.ewma_limit,
            "reasons": event.reasons,
            "alert_emitted": event.emitted,
            "forecast_id": event.forecast_id,
        })
    monitoring = pd.DataFrame(monitoring_rows).set_index("timestamp")
    monitoring
    """),
    code(r"""
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(monitoring.index, monitoring.actual, color="black")
    axes[0].set_title("Realizados revelados sequencialmente")
    axes[1].plot(monitoring.index, monitoring.surprise, color=COLORS["blue"], marker="o")
    axes[1].axhline(0, color="black", lw=.8); axes[1].axhline(3, color=COLORS["red"], ls="--")
    axes[1].axhline(-3, color=COLORS["red"], ls="--"); axes[1].set_title("Surpresa prequential")
    axes[2].plot(monitoring.index, monitoring.cusum_positive, label="CUSUM+")
    axes[2].plot(monitoring.index, monitoring.cusum_negative, label="CUSUM−")
    axes[2].plot(monitoring.index, monitoring.ewma, label="EWMA")
    alerts = monitoring[monitoring.alert_emitted]
    axes[2].scatter(alerts.index, alerts.ewma, color=COLORS["red"], label="alerta", zorder=5)
    axes[2].set_title("Estatísticas sequenciais"); axes[2].legend()
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ## 14. Auditoria e resultado consolidado

    Uma execução é confiável quando preserva relações causais, não apenas quando produz figuras.
    Verificamos que cada realizado reservado encontrou um forecast anterior, que a série terminou
    com o tamanho original e que o próximo forecast usa como cutoff o último realizado revelado.
    """),
    code(r"""
    final_series = engine.get_series(SERIES_ID)
    audit = {
        "observações carregadas": len(raw),
        "observações finais no motor": len(final_series.values),
        "eventos monitorados": len(monitoring),
        "valores imputados no total": int(imputed_mask.sum()),
        "valores imputados no monitoramento": int(monitoring.observation_was_imputed.sum()),
        "eventos com forecast anterior": int(monitoring.forecast_id.notna().sum()),
        "alertas emitidos": int(monitoring.alert_emitted.sum()),
        "modelo selecionado": repr(decision.selected_spec),
        "fallback de cobertura": coverage_fallback_used,
        "CRPS selecionado": decision.scores["crps"],
        "risco conjunto": risk.any_horizon_probability,
        "change-points retrospectivos": len(change_report.breakpoints),
    }
    display(pd.Series(audit))

    assert len(final_series.values) == len(raw)
    assert monitoring.forecast_id.notna().all()
    assert last_cycle is not None
    assert last_cycle.next_forecast.cutoff == monitoring_pd.index[-1]
    assert forecast.cutoff == development_pd.index[-1]
    print("Testes rápidos do Notebook 14 passaram.")
    """),
    md(r"""
    ## 15. Como interpretar sem automatizar demais

    - Uma família sugerida pela caracterização é apenas candidata.
    - O menor CRPS não compensa cobertura inaceitável quando o guardrail está ativo.
    - Resíduos autocorrelacionados indicam dinâmica faltante; resíduos quadráticos correlacionados
      indicam variância faltante.
    - Change-point retrospectivo não antecipa uma quebra.
    - Um alerta é evidência para investigação, não autorização automática para intervenção.
    - Se os dados forem retornos, configure `SERIES_KIND="returns"`; não aplique GARCH a níveis
      apenas porque o adaptador existe.

    **Extensões:** testar outras ordens, comparar métricas por horizonte, calibrar limiares por ARL,
    implementar repositório transacional e executar várias séries em processos separados.
    """),
    md(r"""
    ## 16. Exercícios

    1. Adicione `ARIMASpec(order=(2,1,2))` e compare CRPS, cobertura e resíduos.
    2. Desative `ALLOW_COVERAGE_FALLBACK` e observe a falha quando nenhum modelo é elegível.
    3. Use uma série de retornos, configure `SERIES_KIND="returns"` e avalie GARCH.
    4. Troque `CANDIDATE_MARKOV` para `"always"` e examine a justificativa e o custo computacional.
    5. Altere `RISK_LIMIT` e compare risco marginal, qualquer horizonte e primeira passagem.
    6. Aumente a cauda reservada e compare falso alarme, atraso e custo computacional do refit.

    <details><summary><strong>Gabarito orientativo</strong></summary>

    1. Acrescente o spec à lista; mais parâmetros não garantem melhor previsão fora da amostra.
    2. O erro “nenhum candidato passou os guardrails” é o comportamento seguro esperado.
    3. Verifique ACF dos quadrados e ARCH-LM antes de interpretar o GARCH.
    4. Compare probabilidades de regime, não apenas a classificação mais provável.
    5. A probabilidade de qualquer horizonte não é a soma das marginais.
    6. Uma cauda maior mede melhor o monitor, mas deixa menos dados para seleção.

    </details>
    """),
    code(r"""
    # Scaffold executável para o exercício 1:
    exercise_candidates = [*candidates, ARIMASpec(order=(2, 1, 2), simulation_paths=300)]
    print("Candidatos atuais:", len(candidates))
    print("Candidatos após a extensão:", len(exercise_candidates))
    print("Hipótese manual fora da policy:", exercise_candidates[-1])
    # Para executar a comparação, troque `candidates` por `exercise_candidates` na célula
    # de backtest. Em produção, registre essa exceção junto com sua justificativa de domínio.
    """),
    md(r"""
    ## Conclusão

    Uma única `pandas.Series` percorreu o sistema inteiro com separação causal entre pesquisa e
    monitoramento. O notebook deixou visíveis os candidatos, os critérios, os resíduos, a
    distribuição futura, o risco e os eventos sequenciais. Para produção, substitua o repositório
    em memória e mantenha os mesmos contratos estatísticos e de auditoria.
    """),
]


def preserve_user_loader(path: Path, cells: list[dict]) -> None:
    """Preserva a célula que o usuário pode ter personalizado no notebook existente."""
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    marker = "EDITE SOMENTE ESTA CÉLULA"
    existing = next(
        (
            cell
            for cell in payload.get("cells", [])
            if cell.get("cell_type") == "code" and marker in cell.get("source", "")
        ),
        None,
    )
    generated = next(
        (
            cell
            for cell in cells
            if cell.get("cell_type") == "code" and marker in cell.get("source", "")
        ),
        None,
    )
    if existing is not None and generated is not None:
        generated["source"] = existing["source"]


if __name__ == "__main__":
    preserve_user_loader(PATH, CELLS)
    write_notebook(PATH, CELLS, 14)
