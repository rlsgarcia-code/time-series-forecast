"""Constrói o Notebook 15: distribuição de Y = g(X, Y, Z)."""

from pathlib import Path

from notebook_builder import code, md, setup_cell, write_notebook


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "15_composicao_probabilistica_hierarquica.ipynb"


CELLS = [
    md(r"""
    # Notebook 15 — De variáveis previstas à distribuição de $Y=g(X_1,X_2,X_3)$

    **Pergunta central:** se $X_1$, $X_2$ e $X_3$ são forecasts probabilísticos ou valores futuros
    já conhecidos, como obtemos a distribuição de uma função

    $$Y=g(X_1,X_2,X_3)?$$

    Este capítulo para deliberadamente antes da árvore hierárquica. Primeiro construiremos uma
    única saída `result_y` e abriremos cada etapa executada por `compose()`: marginais, dependência,
    caminhos conjuntos, aplicação de $g$ e distribuição empírica resultante. Folhas e grupos são
    apenas uma repetição desse mesmo mecanismo e ficam para o Notebook 16.

    **Ao final você será capaz de:**

    1. distinguir distribuição marginal de distribuição conjunta;
    2. combinar variáveis previstas, séries conhecidas e escalares;
    3. entender onde a covariância entra em uma função não linear;
    4. reconstruir manualmente os caminhos produzidos pelo motor;
    5. obter média, variância, quantis, CDF e risco da saída derivada.
    """),
    md(r"""
    ## 1. A mecânica em cinco passos

    Para horizonte $H$ e $M$ simulações, cada entrada será representada por uma matriz
    `(M, H)`:

    ```text
    1. modelos marginais  → X₁, X₂, X₃ com M caminhos cada
    2. erros de backtest  → dependência residual entre as entradas previstas
    3. cópula por ranks   → caminhos conjuntos alinhados
    4. função caminho a caminho → Y[m,h] = g(X₁[m,h], X₂[m,h], X₃[m,h])
    5. resumo das linhas → distribuição empírica de Y em cada horizonte
    ```

    Uma entrada conhecida não precisa ser simulada: seu vetor de horizonte é repetido nas $M$
    linhas. Isso equivale a uma distribuição degenerada com variância zero.
    """),
    setup_cell(
        "from examples.memory_repository import MemoryRepository\n"
        "from time_series_lab import (\n"
        "    BacktestConfig, CandidatePolicy, CompositionPolicy, CompositionRun,\n"
        "    DependencyDiagnostics, EmpiricalDistribution, ForecastEngine,\n"
        "    ModelAdapter, ModelRun, PredictiveDistribution, ThresholdRisk,\n"
        ")\n"
        "from time_series_lab.composition import (\n"
        "    collect_leaves, evaluate_outputs, impose_rank_dependence, materialize_leaves,\n"
        ")"
    ),
    md(r"""
    ### Objetos de baixo nível que aparecerão

    | Objeto | Responsabilidade |
    |---|---|
    | `CandidatePolicy` | transforma a caracterização de **uma entrada** em `ModelSpec` candidatos; não ajusta nem escolhe |
    | `CompositionPolicy` | controla pré-seleção, combinações, caminhos, dependência, cobertura e causalidade do **conjunto** |
    | `ModelRun` | modelo ajustado, cutoff, revisão, especificação e estado estimado |
    | `ModelAdapter` | reconstrói o forecast do modelo sem conhecer a aplicação |
    | `PredictiveDistribution` | contrato marginal: média, variância, CDF, quantis e `sample()` |
    | `DependencyDiagnostics` | correlação residual, shrinkage, quantidade de pares e fallback |
    | `ForecastVariable` | nó imutável do grafo e seus operandos |
    | `EmpiricalDistribution` | distribuição construída diretamente de uma matriz de caminhos |
    | `CompositionRun` | manifesto que congela modelos, revisões, hashes e dependência |

    Usaremos esses objetos diretamente. `ForecastEngine.compose()` aparecerá somente depois de a
    mecânica de baixo nível produzir a distribuição completa de `result_y`.

    **Referências:** [método estatístico](../docs/metodo_composicao_probabilistica.md),
    [mecânica completa da composição](../docs/composition_mechanics.md) e
    [contratos públicos do motor](../docs/engine_api.md).
    """),
    md(r"""
    ## 2. Três entradas: duas previstas e uma configurável

    Simularemos três processos correlacionados. Por padrão, $X_1$ e $X_2$ serão previstos e $X_3$
    será uma série futura conhecida. Trocar `Z_MODE` para `"forecast"` faz as três entradas serem
    previstas sem alterar a função $g$ nem as células posteriores.

    A série conhecida possui **vintages**: para cada alvo registramos quando seu valor ficou
    disponível. Isso impede que o plano conhecido hoje seja usado retrospectivamente em uma origem
    na qual ele ainda não existia.
    """),
    code(r"""
    rng = np.random.default_rng(1501)
    n_history, horizon = 58, 6
    full_index = pd.date_range("2024-01-01", periods=n_history + horizon, freq="D", tz="UTC")
    innovation_covariance = np.array([
        [1.00,  0.65,  0.35],
        [0.65,  1.00, -0.25],
        [0.35, -0.25,  1.00],
    ])
    innovations = rng.multivariate_normal(np.zeros(3), innovation_covariance, size=n_history)
    drivers = np.zeros((n_history, 3))
    intercepts = np.array([0.12, -0.05, 0.08])
    persistence = np.array([0.62, 0.48, 0.55])
    for t in range(1, n_history):
        drivers[t] = intercepts + persistence * drivers[t - 1] + innovations[t]

    driver_x = pd.Series(drivers[:, 0], index=full_index[:n_history], name="driver_x")
    driver_y = pd.Series(drivers[:, 1], index=full_index[:n_history], name="driver_y")
    driver_z = pd.Series(drivers[:, 2], index=full_index[:n_history], name="input_z")

    known_z = pd.Series(
        1.20 + 0.003 * np.arange(len(full_index))
        + 0.05 * np.sin(np.arange(len(full_index)) / 5),
        index=full_index,
        name="input_z",
    )
    known_z_vintages = pd.DataFrame({
        "as_of": full_index - pd.Timedelta(days=1),
        "target": full_index,
        "value": known_z.to_numpy(),
    })

    Z_MODE = "known"  # altere para "forecast" para modelar também X3
    if Z_MODE not in {"known", "forecast"}:
        raise ValueError("Z_MODE deve ser 'known' ou 'forecast'")
    pd.concat([driver_x, driver_y, driver_z], axis=1).head()
    """),
    code(r"""
    ax = pd.concat([driver_x, driver_y, driver_z], axis=1).plot(figsize=(12, 4), lw=1.1)
    ax.set(title="Históricos disponíveis para as três entradas", xlabel="tempo")
    plt.show()
    """),
    md(r"""
    ## 3. Distribuições marginais versus distribuição conjunta

    Um forecast marginal responde, por exemplo,

    $$F_{1,h}(x)=P(X_{1,t+h}\le x\mid\mathcal F_t).$$

    Conhecer separadamente $F_1$, $F_2$ e $F_3$ não determina como as entradas variam juntas. Duas
    combinações podem ter as mesmas marginais e correlações muito diferentes. Para calcular uma
    função não linear precisamos de uma aproximação da distribuição conjunta

    $$F_{1,2,3}(x_1,x_2,x_3).$$

    O motor preserva cada forecast marginal e aprende a associação entre eles com erros
    rolling-origin sincronizados. Uma entrada conhecida entra como caminho repetido e não participa
    da matriz de dependência residual.
    """),
    md(r"""
    ## 4. Declare as variáveis e a função $g$

    A função abaixo é intencionalmente genérica:

    $$g(x_1,x_2,x_3)=x_1x_2+x_3.$$

    Ela contém um produto para tornar o efeito da covariância visível. Na aplicação real,
    `generic-g` pode ser substituído por um nome opaco e a função pode permanecer no código privado.
    O motor persiste somente nome, versão e hash da expressão.
    """),
    code(r"""
    repository = MemoryRepository()
    engine = ForecastEngine(repository=repository)

    x1 = engine.variable("driver_x", driver_x, mode="forecast")
    x2 = engine.variable("driver_y", driver_y, mode="forecast")
    if Z_MODE == "known":
        x3 = engine.variable(
            "input_z", known_z, mode="known", vintages=known_z_vintages
        )
    else:
        x3 = engine.variable("input_z", driver_z, mode="forecast")

    def g_values(x1_values: np.ndarray, x2_values: np.ndarray,
                 x3_values: np.ndarray) -> np.ndarray:
        '''Aplica g caminho a caminho; todas as entradas têm forma (M, H).'''
        return x1_values * x2_values + x3_values

    result_y = engine.transform("generic-g", g_values, x1, x2, x3, version="v1")
    outputs = {"result_y": result_y}
    result_y.to_payload()
    """),
    md(r"""
    ## 5. Por que $g(E[X_1],E[X_2],E[X_3])$ não basta

    Para esta função,

    $$E[Y]=E[X_1X_2]+E[X_3]
          =E[X_1]E[X_2]+\operatorname{Cov}(X_1,X_2)+E[X_3].$$

    Aplicar $g$ somente às médias elimina o termo de covariância. Para funções mais gerais — razão,
    máximo, mínimo, `clip`, exponencial ou condicionais — nem sequer existe uma correção simples.
    A solução uniforme é avaliar $g$ nas amostras conjuntas.
    """),
    md(r"""
    ## 6. Selecione os modelos pela distribuição de `result_y`

    A seleção tem duas etapas reais neste exemplo:

    1. cada entrada prevista retém seus dois melhores candidatos marginais;
    2. até quatro combinações são avaliadas pelo CRPS de `result_y`.

    Assim podemos observar se uma combinação que não contém todos os campeões marginais produz uma
    distribuição derivada melhor. Como há apenas uma saída, não existe ambiguidade sobre pesos de
    folhas, grupos ou total.

    As duas políticas têm papéis diferentes:

    | Política | Unidade de decisão | Neste exemplo |
    |---|---|---|
    | `CandidatePolicy` | uma série `forecast` | propõe até 4 specs por entrada usando caracterização |
    | `CompositionPolicy` | o grafo completo | retém 2 por entrada, avalia até 4 combinações e usa dependência residual |

    `CandidatePolicy` não vê $g$ e não declara vencedor. `CompositionPolicy` não cria modelos; ela
    define como os candidatos marginais serão combinados e julgados pela distribuição de
    `result_y`. Veja a [referência técnica](../docs/composition_mechanics.md#2-o-que-backtest_expression-executa).
    """),
    code(r"""
    candidate_policy = CandidatePolicy(
        max_candidates=4,
        differencing="never",
        garch="never",
        markov_switching="never",
        simulation_paths=120,
    )
    composition_policy = CompositionPolicy(
        preselect_top_k=2,
        max_combinations=4,
        dependency="residual_copula",
        min_paired_errors=3,  # didático; use o padrão mais conservador em produção
        n_backtest_paths=160,
        coverage_range=None,
    )
    training = engine.backtest_expression(
        outputs,
        backtest=BacktestConfig(initial_train_size=34, horizon=1, step=6, seed=1501),
        candidate_policy=candidate_policy,
        policy=composition_policy,
        promote_as="generic-result-champion",
    )
    score_table = pd.DataFrame([
        {
            "combinação": score["combination_id"],
            "CRPS normalizado de Y": score["aggregate_normalized_crps"],
            "cobertura de Y": score["aggregate_coverage"],
            **{f"modelo {name}": spec for name, spec in score["specifications"].items()},
        }
        for score in training.scores
    ]).sort_values("CRPS normalizado de Y")
    score_table
    """),
    md(r"""
    ### O que ocorreu dentro de `backtest_expression()`

    1. coletou do grafo as folhas `forecast` e `known`;
    2. validou cutoff/frequência comuns e os vintages de `input_z`;
    3. caracterizou cada entrada prevista e aplicou `CandidatePolicy`;
    4. executou backtest marginal e reteve os dois menores CRPS por entrada;
    5. formou até quatro combinações desses modelos;
    6. em cada origem, gerou caminhos, estimou dependência apenas com erros já observáveis,
       materializou $X_3$, aplicou $g$ e calculou o CRPS de `result_y`;
    7. selecionou a combinação com menor CRPS normalizado;
    8. reajustou os modelos escolhidos no histórico completo e criou um `CompositionRun`.

    Portanto, `backtest_expression()` faz **treinamento e seleção do conjunto**. Ele não emite o
    forecast futuro e não retorna caminhos. A referência detalhada está em
    [`docs/composition_mechanics.md`](../docs/composition_mechanics.md).
    """),
    code(r"""
    pd.DataFrame([
        {"entrada": name, "modelo selecionado": repr(spec),
         "model_run_id": training.selected_model_runs[name]}
        for name, spec in training.selected_specs.items()
    ])
    """),
    md(r"""
    ## 7. `CompositionRun`: o manifesto congelado

    `backtest_expression()` devolve um `CompositionRun`, não o forecast. Ele congela quais
    `ModelRun` serão usados por entrada, suas revisões, os hashes do grafo e o
    `DependencyDiagnostics`. A partir daqui não haverá novo treinamento implícito.

    Ele contém o placar e a decisão reproduzível, mas **não** contém caminhos futuros, a
    distribuição de `result_y`, o código privado de $g$ ou os valores futuros de uma entrada
    `known`. Esses elementos só são reunidos quando a aplicação chama `compose()`.
    """),
    code(r"""
    N_PATHS = 2_000
    COMPOSE_SEED = 1502
    assert isinstance(training, CompositionRun)
    base_series = engine.get_series("driver_x")
    offset = pd.tseries.frequencies.to_offset(base_series.frequency)
    targets = pd.date_range(
        base_series.cutoff + offset, periods=horizon, freq=base_series.frequency
    )
    pd.Series({
        "composition_run_id": training.composition_run_id,
        "causal": training.causal,
        "expressão result_y": training.expression_ids["result_y"],
        "entradas previstas": tuple(training.selected_model_runs),
        "targets": tuple(targets),
    }, name="CompositionRun").to_frame()
    """),
    md(r"""
    ## 8. Passo 1 — Gere os caminhos marginais

    Para cada variável `forecast`, o adaptador selecionado produz uma distribuição com dependência
    temporal própria. `sample(M)` gera uma matriz

    $$\mathbf X_i\in\mathbb R^{M\times H}.$$

    Neste ponto as matrizes foram amostradas separadamente: cada uma possui a marginal correta, mas
    os números de linha ainda não têm significado conjunto entre variáveis.
    """),
    code(r"""
    forecast_names = training.dependency.variable_names
    raw_paths = {}
    marginal_rows = []
    for variable_number, name in enumerate(forecast_names):
        model_run: ModelRun = repository.get_model_run(training.selected_model_runs[name])
        adapter: ModelAdapter = engine.models.get(model_run.spec.family)
        variable_seed = COMPOSE_SEED + variable_number * 10_000
        distribution: PredictiveDistribution = adapter.forecast(
            model_run, horizon, seed=variable_seed
        )
        paths = distribution.sample(N_PATHS, seed=variable_seed + 1)
        raw_paths[name] = paths
        quantiles = np.quantile(paths, [0.05, 0.50, 0.95], axis=0)
        for step in range(horizon):
            marginal_rows.append({
                "entrada": name,
                "horizonte": step + 1,
                "média": paths[:, step].mean(),
                "p05": quantiles[0, step],
                "mediana": quantiles[1, step],
                "p95": quantiles[2, step],
            })
    pd.DataFrame(marginal_rows).head(12).round(3)
    """),
    md(r"""
    ### 8.1 De onde vem a CDF preditiva $\widehat F$

    O PIT **não escolhe uma distribuição**. Cada modelo marginal entrega um objeto
    `PredictiveDistribution`; é esse objeto que define $\widehat F_{i,t,h}$ condicional à
    informação disponível no cutoff. A hipótese distributiva pertence ao adaptador:

    | Família | Hipótese das inovações | `PredictiveDistribution` | Como calcula $\widehat F_h(x)$ |
    |---|---|---|---|
    | AR(1) | $\varepsilon_t\sim N(0,\sigma^2)$ | `GaussianDistribution` | normal analítica com média e variância recursivas |
    | nível local/Kalman | ruídos de estado e observação gaussianos | `GaussianDistribution` | normal analítica obtida pelo filtro |
    | Markov switching | normal condicional a cada regime | `GaussianMixtureDistribution` | soma ponderada das CDFs dos regimes |
    | ARIMA | perturbações do estado-espaço ajustado | `EmpiricalDistribution` | proporção de trajetórias simuladas abaixo de $x$ |
    | GARCH(1,1) | choque padronizado normal nesta implementação | `EmpiricalDistribution` | proporção de trajetórias simuladas abaixo de $x$ |

    Nos casos gaussianos,

    $$\widehat F_h(x)=\Phi\!\left(\frac{x-\widehat\mu_h}{\widehat\sigma_h}\right).$$

    Para mistura de regimes,

    $$\widehat F_h(x)=\sum_k \widehat p_{k,h}\,
      \Phi\!\left(\frac{x-\widehat\mu_k}{\widehat\sigma_k}\right).$$

    Para uma distribuição empírica com $M$ caminhos,

    $$\widehat F_h(x)=\frac{1}{M}\sum_{m=1}^{M}
      \mathbf 1\{X_h^{(m)}\le x\}.$$

    Portanto, o chapéu reúne incerteza estimada pelo modelo e, no caso empírico, aproximação de
    Monte Carlo. O motor avalia `distribution.cdf(realized)`; não ajusta uma normal por cima de
    todas as famílias. A normal aparece novamente **depois** do PIT, em
    $z=\Phi^{-1}(u)$, para colocar as surpresas numa escala comum e construir a cópula residual.

    A derivação, hipóteses e limitações estão no
    [método de composição probabilística](../docs/metodo_composicao_probabilistica.md#5-modelos-marginais-e-cdf-preditiva);
    a correspondência com o código está na
    [referência técnica — CDF preditiva](../docs/composition_mechanics.md#cdf-preditiva).
    """),
    code(r"""
    cdf_rows = []
    for variable_number, name in enumerate(forecast_names):
        model_run = repository.get_model_run(training.selected_model_runs[name])
        adapter = engine.models.get(model_run.spec.family)
        distribution = adapter.forecast(
            model_run, 1, seed=COMPOSE_SEED + variable_number * 10_000
        )
        reference_value = float(distribution.mean[0])
        cdf_rows.append({
            "entrada": name,
            "modelo marginal": model_run.spec.family,
            "objeto distributivo": type(distribution).__name__,
            "x de referência": reference_value,
            "F_hat(x)": float(distribution.cdf([reference_value])[0]),
            "caminhos na CDF": (
                distribution.paths.shape[0]
                if isinstance(distribution, EmpiricalDistribution)
                else "analítica"
            ),
        })
    pd.DataFrame(cdf_rows).round(4)
    """),
    md(r"""
    `F_hat(x)` não precisa ser $0{,}5$ quando `x` é a média de uma distribuição empírica ou
    assimétrica. Isso é esperado: média e mediana são conceitos distintos.

    A implicação $u=\widehat F(X)\sim U(0,1)$ vale quando a CDF preditiva contínua está
    corretamente especificada. Logo, uniformidade do PIT é um **diagnóstico de calibração**, não
    uma premissa garantida. Com CDF empírica finita, os PITs ficam numa grade; o motor os limita a
    $[10^{-6},1-10^{-6}]$ antes de aplicar $\Phi^{-1}$.
    """),
    md(r"""
    ## 9. Passo 2 — Estime a dependência com erros de forecast

    **Contexto:** os candidatos já foram avaliados marginalmente. Em cada origem rolling-origin,
    cada modelo foi ajustado somente até um cutoff e emitiu uma CDF para o próximo timestamp. Quando
    esse realizado fica disponível, o motor mede onde ele caiu na CDF emitida anteriormente:

    $$u_{i,t}=\widehat F_{i,t}(x_{i,t}),\qquad
      z_{i,t}=\Phi^{-1}(u_{i,t}).$$

    Cada linha da matriz $Z$ é um **mesmo timestamp-alvo**; cada coluna é uma variável prevista.
    Assim, a correlação dos $z$ mede surpresas que ocorreram juntas depois que a dinâmica marginal
    já foi removida. Não usamos a correlação dos níveis, que pode refletir tendência comum ou escala.

    Durante a avaliação de uma combinação no cutoff $c$, entram somente erros com alvo $t\le c$.
    Isso evita usar um erro futuro para coordenar caminhos de uma previsão passada. O
    `CompositionRun` final usa todos os erros um passo à frente disponíveis na janela de
    desenvolvimento. A dependência temporal entre horizontes continua preservada dentro dos
    caminhos de cada `PredictiveDistribution`.

    A correlação amostral recebe shrinkage para a identidade e reparo para permanecer
    semidefinida positiva. A fórmula e o algoritmo exatos estão na
    [seção de dependência da referência técnica](../docs/composition_mechanics.md#3-dependência-contexto-do-passo-2).

    Com poucos pares, o motor usa independência e registra o fallback; não interpreta ausência de
    evidência como prova de independência.
    """),
    code(r"""
    dependency: DependencyDiagnostics = training.dependency
    display(pd.Series({
        "variáveis previstas": dependency.variable_names,
        "erros PIT emparelhados": dependency.paired_errors,
        "shrinkage": dependency.shrinkage,
        "fallback": dependency.fallback,
        "motivo": dependency.reason,
    }, name="diagnóstico").to_frame())
    pd.DataFrame(
        dependency.correlation,
        index=dependency.variable_names,
        columns=dependency.variable_names,
    ).round(3)
    """),
    md(r"""
    ## 10. Passo 3 — Associe linhas com uma cópula por ranks

    O motor não altera os valores marginais. Ele sorteia vetores gaussianos correlacionados,
    resume a severidade de cada caminho completo, ordena os caminhos marginais e associa seus ranks
    aos ranks gaussianos. Reordenar linhas preserva cada caminho e sua dependência temporal; muda
    apenas qual caminho de $X_1$ fica na mesma linha de qual caminho de $X_2$.
    """),
    code(r"""
    def align_paths_by_rank(
        paths_by_name: dict[str, np.ndarray],
        names: tuple[str, ...],
        correlation: np.ndarray,
        *,
        seed: int,
    ) -> dict[str, np.ndarray]:
        '''Versão explícita da reordenação usada pelo motor.'''
        n_paths = paths_by_name[names[0]].shape[0]
        latent = np.random.default_rng(seed).multivariate_normal(
            np.zeros(len(names)), correlation, size=n_paths
        )
        aligned = dict(paths_by_name)
        for column, name in enumerate(names):
            original = paths_by_name[name]
            scale = np.std(original, axis=0, ddof=1)
            scale = np.where(scale > 1e-12, scale, 1.0)
            severity = np.mean(
                (original - np.mean(original, axis=0)) / scale, axis=1
            )
            sorted_paths = original[np.argsort(severity)]
            latent_order = np.argsort(latent[:, column])
            reordered = np.empty_like(original)
            reordered[latent_order] = sorted_paths
            aligned[name] = reordered
        return aligned

    explicit_joint_paths = align_paths_by_rank(
        raw_paths,
        forecast_names,
        dependency.correlation,
        seed=COMPOSE_SEED + 99,
    )
    joint_paths, realized_dependency = impose_rank_dependence(
        raw_paths, dependency, seed=COMPOSE_SEED + 99
    )
    for name in forecast_names:
        np.testing.assert_array_equal(explicit_joint_paths[name], joint_paths[name])
    pd.DataFrame({
        "correlação antes": [
            np.corrcoef(raw_paths[forecast_names[0]][:, 0],
                        raw_paths[forecast_names[1]][:, 0])[0, 1]
        ],
        "correlação depois": [
            np.corrcoef(joint_paths[forecast_names[0]][:, 0],
                        joint_paths[forecast_names[1]][:, 0])[0, 1]
        ],
    }).round(3)
    """),
    md(r"""
    ## 11. Passo 4 — Transforme valores dados em caminhos

    Quando $X_3$ é conhecido, seu vetor futuro $[x_{3,1},\ldots,x_{3,H}]$ é difundido pelas
    $M$ linhas. A variância entre caminhos é zero. Se `Z_MODE="forecast"`, $X_3$ já estará em
    `joint_paths` e participará da cópula exatamente como as demais variáveis previstas.
    """),
    code(r"""
    graph_leaves = collect_leaves(outputs)
    leaf_values = materialize_leaves(
        graph_leaves,
        joint_paths,
        targets,
        n_paths=N_PATHS,
        scenario=None,
    )
    x3_paths = leaf_values[graph_leaves["input_z"].expression_id]

    assert x3_paths.shape == (N_PATHS, horizon)
    pd.DataFrame({
        "target": targets,
        "E[X3]": x3_paths.mean(axis=0),
        "Var[X3]": x3_paths.var(axis=0, ddof=1),
    }).round(6)
    """),
    md(r"""
    ## 12. Passo 5 — Aplique $g$ linha a linha

    Agora os índices de linha representam o mesmo cenário conjunto. Para cada caminho $m$ e
    horizonte $h$:

    $$Y_h^{(m)}=g\!\left(X_{1,h}^{(m)},X_{2,h}^{(m)},X_{3,h}^{(m)}\right).$$

    As $M$ linhas resultantes são amostras da distribuição futura de $Y$. Nenhuma hipótese
    gaussiana é necessária para a saída derivada.
    """),
    code(r"""
    manual_y_paths = g_values(
        joint_paths["driver_x"],
        joint_paths["driver_y"],
        x3_paths,
    )
    low_level_outputs = evaluate_outputs(outputs, leaf_values)
    low_level_y_paths = low_level_outputs["result_y"]
    graph_evaluation_error = float(np.max(np.abs(manual_y_paths - low_level_y_paths)))
    assert graph_evaluation_error == 0.0
    print("Erro máximo entre g explícita e evaluate_outputs():", graph_evaluation_error)
    """),
    md(r"""
    ## 13. Da matriz de caminhos à distribuição empírica

    Para cada horizonte $h$:

    $$\widehat E[Y_h]=\frac1M\sum_mY_h^{(m)},$$

    $$\widehat F_h(y)=\frac1M\sum_m\mathbf1\{Y_h^{(m)}\le y\}.$$

    Variância e quantis são calculados pelas colunas. `EmpiricalDistribution` apenas organiza essas
    operações sobre a mesma matriz que acabamos de construir.
    """),
    code(r"""
    manual_distribution = EmpiricalDistribution(manual_y_paths)
    manual_quantiles = np.quantile(manual_y_paths, [0.05, 0.50, 0.95], axis=0)
    cdf_threshold = float(np.median(manual_y_paths[:, -1]))
    manual_cdf = np.mean(manual_y_paths <= cdf_threshold, axis=0)

    distribution_summary = pd.DataFrame({
        "target": targets,
        "média": manual_y_paths.mean(axis=0),
        "variância": manual_y_paths.var(axis=0, ddof=1),
        "p05": manual_quantiles[0],
        "mediana": manual_quantiles[1],
        "p95": manual_quantiles[2],
        f"P(Y <= {cdf_threshold:.2f})": manual_cdf,
    })
    np.testing.assert_allclose(manual_distribution.mean, distribution_summary["média"])
    np.testing.assert_allclose(
        manual_distribution.quantile([0.05, 0.50, 0.95]), manual_quantiles
    )
    distribution_summary.round(3)
    """),
    md(r"""
    ## 14. Agora use a fachada `compose()`

    Já construímos a distribuição usando `ModelRun`, `ModelAdapter`, `PredictiveDistribution`,
    `DependencyDiagnostics`, matrizes de caminhos, materialização de folhas, avaliação do grafo e
    `EmpiricalDistribution`. A fachada deve apenas coordenar esses mesmos passos e persistir o
    `ComposedForecast`.
    """),
    code(r"""
    forecast = engine.compose(
        outputs,
        model_ref="generic-result-champion",
        horizon=horizon,
        n_paths=N_PATHS,
        seed=COMPOSE_SEED,
    )
    engine_y_paths = forecast.paths["result_y"]
    reconstruction_error = float(np.max(np.abs(manual_y_paths - engine_y_paths)))
    assert reconstruction_error == 0.0
    distribution_y = forecast.distributions["result_y"]
    assert isinstance(distribution_y, EmpiricalDistribution)
    print("Erro máximo entre baixo nível e compose():", reconstruction_error)
    """),
    code(r"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].scatter(
        joint_paths["driver_x"][:, -1], joint_paths["driver_y"][:, -1],
        c=manual_y_paths[:, -1], cmap="viridis", s=10, alpha=.45,
    )
    axes[0].set(title="Cenários conjuntos no último horizonte",
                xlabel="X1", ylabel="X2")
    axes[1].hist(manual_y_paths[:, -1], bins=35, density=True,
                 color=COLORS["purple"], alpha=.75)
    axes[1].axvline(manual_y_paths[:, -1].mean(), color="black", ls="--", label="média")
    axes[1].set(title="Distribuição resultante de Y", xlabel="Y")
    axes[1].legend(); plt.tight_layout(); plt.show()
    """),
    md(r"""
    ## 15. Onde aparece a covariância

    Podemos conferir numericamente:

    $$E[X_1X_2]-E[X_1]E[X_2]=\operatorname{Cov}(X_1,X_2).$$
    """),
    code(r"""
    x1_paths = joint_paths["driver_x"]
    x2_paths = joint_paths["driver_y"]
    covariance = np.array([
        np.cov(x1_paths[:, step], x2_paths[:, step], ddof=1)[0, 1]
        for step in range(horizon)
    ])
    product_correction = (
        np.mean(x1_paths * x2_paths, axis=0)
        - np.mean(x1_paths, axis=0) * np.mean(x2_paths, axis=0)
    )
    pd.DataFrame({
        "target": targets,
        "Cov(X1,X2)": covariance,
        "E[X1X2]-E[X1]E[X2]": product_correction,
        "diferença Monte Carlo": product_correction - covariance,
    }).round(5)
    """),
    md(r"""
    ## 16. O que mudaria se assumíssemos independência?

    `raw_paths` preserva as marginais, mas suas linhas foram sorteadas separadamente. Aplicar $g$
    antes da associação equivale aproximadamente a assumir independência. A comparação isola o
    efeito da dependência sem trocar nenhum modelo marginal.
    """),
    code(r"""
    independent_x3 = x3_paths if Z_MODE == "known" else raw_paths["input_z"]
    independent_y_paths = g_values(
        raw_paths["driver_x"], raw_paths["driver_y"], independent_x3
    )
    pd.DataFrame({
        "target": targets,
        "média conjunta": manual_y_paths.mean(axis=0),
        "média independente": independent_y_paths.mean(axis=0),
        "desvio conjunto": manual_y_paths.std(axis=0, ddof=1),
        "desvio independente": independent_y_paths.std(axis=0, ddof=1),
    }).round(3)
    """),
    md(r"""
    ## 17. Risco é calculado sobre a distribuição de $Y$

    Depois da composição, `result_y` possui o mesmo contrato de qualquer distribuição preditiva.
    Podemos calcular probabilidades marginais, ocorrência em qualquer horizonte e primeira passagem
    sem voltar às variáveis de entrada.
    """),
    code(r"""
    limit = float(np.quantile(manual_y_paths, 0.82))
    risk = engine.assess_risk(
        distribution_y,
        ThresholdRisk(limit=limit, n_paths=2_000, seed=1503),
    )
    pd.DataFrame({
        "target": targets,
        "P(Y > limite)": risk.marginal_probability,
        "primeira passagem": risk.first_passage_probability,
    }).assign(
        limite=limit,
        probabilidade_em_algum_horizonte=risk.any_horizon_probability,
    ).round(3)
    """),
    md(r"""
    ## 18. De uma saída para folhas e grupos

    Uma árvore não introduz outra mecânica probabilística. Ela apenas avalia várias funções sobre as
    mesmas entradas e caminhos:

    ```python
    folha_a = g_a(x1, x2, x3)
    folha_b = g_b(x1, x2, x3)
    grupo = folha_a + folha_b
    total = grupo_a + grupo_b
    ```

    Como as somas são aplicadas caminho a caminho, a coerência é automática. O Notebook 16 recebe
    `dict[str, dict[str, ForecastVariable]]`: cada valor interno já é uma expressão derivada.
    """),
    md(r"""
    ## 19. Limitações e armadilhas

    - O exemplo usa poucos erros emparelhados; a correlação é didática, não produtiva.
    - A cópula por ranks preserva marginais e caminhos, mas não é um modelo estrutural multivariado.
    - Correlação dos níveis não deve substituir dependência dos erros rolling-origin.
    - Uma variável `known` sem vintages torna o backtest retrospectivo não causal.
    - Alterar uma função privada exige alterar sua versão; o motor não persiste seu código.
    - Mais caminhos reduzem erro Monte Carlo, mas não corrigem modelos marginais ruins.
    - Validar as entradas não valida automaticamente `result_y`.
    """),
    md(r"""
    ## 20. Exercícios

    1. Troque `Z_MODE` para `"forecast"` e observe a matriz passar de 2×2 para 3×3.
    2. Substitua $g$ por `x1 / np.maximum(np.abs(x2), 0.2) + x3`.
    3. Troque `dependency` para `"independent"` e compare média, desvio e risco.
    4. Reduza `N_PATHS` para 100 e repita com seeds diferentes.
    5. Compare o vencedor composto com os campeões marginais.
    6. Transforme $X_3$ em escalar com `mode="constant"`.

    <details><summary><strong>Gabarito orientativo</strong></summary>

    1. $X_3$ passa a ter incerteza própria e participa do alinhamento conjunto.
    2. A razão pode produzir caudas longas; quantis e CRPS são essenciais.
    3. As marginais permanecem iguais; somente a associação entre linhas muda.
    4. As estimativas oscilam mais com uma amostra menor.
    5. O menor CRPS de `result_y` pode usar um modelo marginal em segundo lugar.
    6. O escalar é repetido em todas as linhas e horizontes, com variância zero.

    </details>
    """),
    code(r"""
    # Testes rápidos: o tutorial reproduziu exatamente a mecânica do motor.
    assert training.causal
    assert len(training.scores) >= 2
    assert reconstruction_error == 0.0
    assert manual_y_paths.shape == (N_PATHS, horizon)
    assert np.all(np.isfinite(manual_y_paths))
    assert repository.get_composition_alias("generic-result-champion") == training.composition_run_id
    if Z_MODE == "known":
        assert np.allclose(x3_paths.var(axis=0), 0.0)
    print("Testes rápidos do Notebook 15 passaram.")
    """),
    md(r"""
    ## Conclusão

    A distribuição de $Y=g(X_1,X_2,X_3)$ surgiu de uma sequência concreta: previsões marginais,
    dependência estimada fora da amostra, associação de caminhos, broadcast de valores conhecidos,
    aplicação vetorizada de $g$ e resumo empírico. A hierarquia do Notebook 16 reutiliza exatamente
    essa mecânica em várias folhas.

    **Próximo passo:** o [Notebook 16](16_ciclo_hierarquico_dict.ipynb) recebe expressões
    $Y_j=g_j(X_{j1},\ldots,X_{jp})$ organizadas por grupo e constrói automaticamente grupos e total.
    """),
]


if __name__ == "__main__":
    write_notebook(PATH, CELLS, 15)
