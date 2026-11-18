"""Constrói o Notebook 16: hierarquia de expressões probabilísticas derivadas."""

from pathlib import Path

from notebook_builder import code, md, setup_cell, write_notebook


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "16_ciclo_hierarquico_dict.ipynb"


CELLS = [
    md(r"""
    # Notebook 16 — Hierarquia de expressões probabilísticas

    Aqui uma folha de negócio **não é uma série prevista diretamente**. Ela é uma saída derivada:

    $$Y_j=g_j(X_{j1},\ldots,X_{jp}),$$

    em que as entradas $X$ podem ser previstas, conhecidas, constantes ou de cenário. As folhas
    derivadas são organizadas em grupos e somadas caminho a caminho:

    $$G_k=\sum_{j\in k}Y_j,\qquad T=\sum_kG_k.$$

    A entrada principal será `GROUPED_EXPRESSIONS`, um
    `dict[str, dict[str, ForecastVariable]]`. A chave externa identifica o grupo; a interna dá um
    nome estável à folha derivada.

    **Ao final você será capaz de:** construir fórmulas privadas, identificar as séries primitivas
    realmente modeladas, selecionar seus modelos pela qualidade das saídas derivadas, gerar a
    distribuição de cada folha/grupo/total e provar coerência em todas as trajetórias.

    **Pré-requisito:** o [Notebook 15](15_composicao_probabilistica_hierarquica.ipynb) explica CDF,
    PIT, dependência e avaliação de uma única função $g$.
    """),
    md(r"""
    ## 1. A árvore correta

    ```text
    séries primitivas X                        ← recebem modelos marginais
       ├── X_global ───────────────┐
       ├── X_grupo_a ───────┐      │
       └── X_aditivo_a1 ────┴──────┴─► Y_a1 = g(X...)  ← folha derivada
                                            │
    outras entradas ────────────────────────┴─► Y_a2
                                                    │
                                  G_a = Y_a1 + Y_a2│
                                                    ├─► T = G_a + G_b
                                  G_b = Y_b1 + Y_b2│
    ```

    `CandidatePolicy` atua nas séries $X$. `CompositionPolicy` compara combinações desses modelos
    usando CRPS e cobertura das saídas $Y$, $G$ e $T$. Uma entrada compartilhada é o **mesmo
    objeto** em várias expressões e, por isso, usa exatamente o mesmo caminho em todas elas.
    """),
    setup_cell(
        "from functools import reduce\n"
        "from operator import add\n"
        "from examples.memory_repository import MemoryRepository\n"
        "from time_series_lab import (\n"
        "    BacktestConfig, CandidatePolicy, CompositionPolicy, ForecastEngine,\n"
        "    ForecastVariable, ThresholdRisk,\n"
        ")\n"
        "from time_series_lab.composition import collect_leaves"
    ),
    md(r"""
    ## 2. Configuração e validação das séries primitivas

    O exemplo usa 72 observações, deliberadamente menos de 80. `ffill` só preenche ausências
    internas com informação passada; um `NaN` no cutoff é rejeitado. Frequência, timezone e cutoff
    comuns continuam obrigatórios para as entradas `forecast` de uma composição.
    """),
    code(r"""
    TIMEZONE = "UTC"
    MISSING_VALUE_POLICY = "ffill"  # "ffill" ou "error"
    MINIMUM_OBSERVATIONS = 36
    BUILD_GLOBAL_TOTAL = True
    FORECAST_HORIZON = 6
    N_FORECAST_PATHS = 2_000
    RANDOM_SEED = 1601

    PRESELECT_TOP_K = 1
    MAX_COMBINATIONS = 1
    CANDIDATE_BUDGET = 4
    BACKTEST_PATHS = 120
    OUTPUT_WEIGHTS = {"total": 2.0}

    if MISSING_VALUE_POLICY not in {"ffill", "error"}:
        raise ValueError("MISSING_VALUE_POLICY deve ser 'ffill' ou 'error'")
    """),
    code(r"""
    def prepare_forecast_series(
        name: str,
        source: pd.Series,
        *,
        timezone_name: str,
        missing_policy: str,
        minimum_observations: int,
    ) -> tuple[pd.Series, dict]:
        '''Valida e limpa uma série primitiva sem usar informação futura.'''
        if not isinstance(source, pd.Series):
            raise TypeError(f"{name}: source deve ser pd.Series")
        if not isinstance(source.index, pd.DatetimeIndex):
            raise TypeError(f"{name}: índice deve ser DatetimeIndex")

        series = source.sort_index().copy()
        if not series.index.is_unique:
            raise ValueError(f"{name}: timestamps duplicados")
        if series.index.tz is None:
            series.index = series.index.tz_localize(timezone_name)

        frequency = pd.infer_freq(series.index)
        if frequency is None:
            raise ValueError(f"{name}: frequência não pôde ser inferida")
        expected = pd.date_range(series.index[0], periods=len(series), freq=frequency)
        if not series.index.equals(expected):
            raise ValueError(f"{name}: existem gaps; regularize a grade temporal")

        numeric = pd.to_numeric(series, errors="coerce").astype(float)
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        if numeric.notna().sum() == 0:
            raise ValueError(f"{name}: nenhum valor finito")
        if pd.isna(numeric.iloc[-1]):
            raise ValueError(f"{name}: o valor no cutoff não pode ser ausente")

        first_valid = int(np.flatnonzero(numeric.notna().to_numpy())[0])
        cleaned = numeric.iloc[first_valid:].copy()
        missing_count = int(cleaned.isna().sum())
        if missing_count and missing_policy == "error":
            raise ValueError(f"{name}: {missing_count} valores ausentes")
        if missing_policy == "ffill":
            cleaned = cleaned.ffill()
        if cleaned.isna().any() or len(cleaned) < minimum_observations:
            raise ValueError(
                f"{name}: limpeza deixou menos de {minimum_observations} observações"
            )
        cleaned.name = name
        audit = {
            "entrada": name,
            "recebidas": len(source),
            "removidas_no_início": first_valid,
            "imputadas": missing_count,
            "observações_finais": len(cleaned),
            "início": cleaned.index[0],
            "cutoff": cleaned.index[-1],
            "frequência": frequency,
        }
        return cleaned, audit
    """),
    code(r"""
    repository = MemoryRepository()
    engine = ForecastEngine(repository=repository)
    input_audit_rows: list[dict] = []

    def forecast_input(name: str, source: pd.Series) -> ForecastVariable:
        '''Ponto único de validação e registro das entradas aleatórias.'''
        cleaned, audit = prepare_forecast_series(
            name,
            source,
            timezone_name=TIMEZONE,
            missing_policy=MISSING_VALUE_POLICY,
            minimum_observations=MINIMUM_OBSERVATIONS,
        )
        input_audit_rows.append(audit)
        return engine.variable(name, cleaned, mode="forecast")
    """),
    md(r"""
    ## 3. Construa suas folhas derivadas

    **Edite somente a célula abaixo.** Em uso real, carregue suas `pd.Series`, crie uma única vez
    cada entrada compartilhada e componha as folhas. A função de exemplo
    $g(a,b,c)=ab+c$ representa genericamente “variáveis aleatórias multiplicadas e depois
    somadas”; substitua-a pela função privada da aplicação.

    `engine.transform()` mantém a função apenas em memória. Persistem nome, versão e hash, nunca o
    código ou a árvore privada.
    """),
    code(r"""
    # ==================== EDITE SOMENTE ESTA CÉLULA ====================
    def build_grouped_expressions(
        engine: ForecastEngine,
    ) -> dict[str, dict[str, ForecastVariable]]:
        rng = np.random.default_rng(RANDOM_SEED)
        index = pd.date_range("2024-01-01", periods=72, freq="D", tz="UTC")
        common_shock = rng.normal(0.0, 0.08, len(index))

        def ar_series(name: str, level: float, phi: float,
                      shock: np.ndarray) -> pd.Series:
            values = np.empty(len(index))
            values[0] = level
            for t in range(1, len(index)):
                values[t] = level + phi * (values[t - 1] - level) + shock[t]
            return pd.Series(values, index=index, name=name)

        # Um fator global compartilhado por todas as folhas.
        global_series = ar_series(
            "global_factor", 1.15, 0.62,
            common_shock + rng.normal(0.0, 0.05, len(index)),
        )
        global_factor = forecast_input("global_factor", global_series)

        group_factors: dict[str, ForecastVariable] = {}
        additive_terms: dict[tuple[str, str], ForecastVariable] = {}
        for group_number, group in enumerate(["group_north", "group_south"]):
            group_shock = rng.normal(0.0, 0.07, len(index))
            group_series = ar_series(
                f"{group}/factor", 0.90 + 0.18 * group_number, 0.52,
                0.35 * common_shock + group_shock,
            )
            if group == "group_north":
                group_series.iloc[12] = np.nan  # demonstra limpeza causal
            group_factors[group] = forecast_input(f"{group}/factor", group_series)

            for leaf_number in range(2):
                leaf = f"item_{leaf_number + 1}"
                additive_series = ar_series(
                    f"{group}/{leaf}/additive",
                    0.35 + 0.12 * leaf_number + 0.08 * group_number,
                    0.45,
                    0.20 * common_shock + 0.35 * group_shock
                    + rng.normal(0.0, 0.06, len(index)),
                )
                if group == "group_south" and leaf == "item_2":
                    additive_series.iloc[27] = np.nan
                additive_terms[(group, leaf)] = forecast_input(
                    f"{group}/{leaf}/additive", additive_series
                )

        def private_g(a_paths: np.ndarray, b_paths: np.ndarray,
                      c_paths: np.ndarray) -> np.ndarray:
            # Exemplo genérico. O código real pode permanecer no pacote da aplicação.
            return a_paths * b_paths + c_paths

        grouped: dict[str, dict[str, ForecastVariable]] = {}
        for group in ["group_north", "group_south"]:
            grouped[group] = {}
            for leaf_number in range(2):
                leaf = f"item_{leaf_number + 1}"
                grouped[group][leaf] = engine.transform(
                    "private-derived-leaf",
                    private_g,
                    global_factor,
                    group_factors[group],
                    additive_terms[(group, leaf)],
                    version="v1",
                )
        return grouped

    GROUPED_EXPRESSIONS = build_grouped_expressions(engine)
    GROUPED_EXPRESSIONS
    """),
    md(r"""
    ## 4. Valide o contrato do dicionário

    O dicionário interno usa nomes de folhas em vez de uma lista para manter identidade estável em
    auditoria, persistência e comparação de resultados. A quantidade de grupos e folhas pode mudar
    sem alterar as células posteriores.
    """),
    code(r"""
    def validate_grouped_expressions(
        grouped: dict[str, dict[str, ForecastVariable]],
    ) -> dict[str, dict[str, ForecastVariable]]:
        if not isinstance(grouped, dict) or not grouped:
            raise TypeError("GROUPED_EXPRESSIONS deve ser um dict não vazio")
        validated: dict[str, dict[str, ForecastVariable]] = {}
        seen_outputs: set[str] = set()
        for raw_group, raw_leaves in grouped.items():
            group = str(raw_group).strip()
            if not group or not isinstance(raw_leaves, dict) or not raw_leaves:
                raise TypeError("cada grupo deve ter nome e dict de expressões não vazio")
            validated[group] = {}
            for raw_leaf, expression in raw_leaves.items():
                leaf = str(raw_leaf).strip()
                output_name = f"leaf/{group}/{leaf}"
                if not leaf or output_name in seen_outputs:
                    raise ValueError(f"nome de folha vazio ou duplicado: {output_name}")
                if not isinstance(expression, ForecastVariable):
                    raise TypeError(f"{output_name}: valor deve ser ForecastVariable")
                seen_outputs.add(output_name)
                validated[group][leaf] = expression
        return validated

    grouped_expressions = validate_grouped_expressions(GROUPED_EXPRESSIONS)
    derived_leaf_outputs = {
        f"leaf/{group}/{leaf}": expression
        for group, leaves in grouped_expressions.items()
        for leaf, expression in leaves.items()
    }
    primitive_leaves = collect_leaves(derived_leaf_outputs)
    forecast_leaves = {
        name: leaf for name, leaf in primitive_leaves.items() if leaf.mode == "forecast"
    }

    structure_rows = [
        {
            "grupo": group,
            "folha derivada": leaf,
            "expression_id": expression.expression_id,
        }
        for group, leaves in grouped_expressions.items()
        for leaf, expression in leaves.items()
    ]
    display(pd.DataFrame(structure_rows))
    display(pd.DataFrame(input_audit_rows))
    print(
        f"{len(grouped_expressions)} grupos; {len(derived_leaf_outputs)} folhas derivadas; "
        f"{len(forecast_leaves)} entradas aleatórias primitivas."
    )
    """),
    code(r"""
    histories = {}
    for name, leaf in forecast_leaves.items():
        registered = engine.get_series(str(leaf.source))
        histories[name] = pd.Series(
            registered.values, index=registered.timestamps, name=name
        )
    ax = pd.concat(histories, axis=1).plot(figsize=(13, 5), lw=1, alpha=.85)
    ax.set(title="Séries primitivas modeladas — não confundir com folhas derivadas",
           xlabel="tempo", ylabel="valor")
    ax.legend(ncol=3, fontsize=8); plt.show()
    """),
    md(r"""
    ## 5. Caracterize as entradas primitivas

    Não existe uma caracterização univariada separada de $Y_j$ nesta etapa: $Y_j$ é uma expressão.
    ACF, estacionariedade, resíduos e candidatos pertencem a cada entrada aleatória $X$. O
    desempenho de $Y_j$ será medido depois, quando as trajetórias conjuntas atravessarem $g$.
    """),
    code(r"""
    candidate_policy = CandidatePolicy(
        max_ar_order=1,
        max_ma_order=1,
        max_total_order=2,
        max_candidates=CANDIDATE_BUDGET,
        differencing="never",
        garch="never",
        markov_switching="never",
        simulation_paths=100,
    )
    characterization_rows = []
    candidate_plans = {}
    for input_name in forecast_leaves:
        profile = engine.characterize(input_name, max_lag=10)
        plan = engine.propose_candidates(
            input_name, policy=candidate_policy, profile=profile, series_kind="level"
        )
        candidate_plans[input_name] = plan
        characterization_rows.append({
            "entrada primitiva": input_name,
            "nobs": profile.nobs,
            "ADF p": profile.adf_pvalue,
            "KPSS p": profile.kpss_pvalue,
            "Ljung-Box p": profile.ljung_box_pvalue,
            "candidatos": len(plan.candidates),
        })
    pd.DataFrame(characterization_rows).round(4)
    """),
    code(r"""
    example_input = next(iter(candidate_plans))
    print(candidate_plans[example_input].explain())
    display(candidate_plans[example_input].to_frame(include_excluded=True))
    """),
    md(r"""
    ## 6. Construa grupos e total a partir das expressões

    Para cada caminho $m$ e horizonte $h$:

    $$Y_{j,h}^{(m)}=g_j(X_{j1,h}^{(m)},\ldots,X_{jp,h}^{(m)}),$$

    $$G_{k,h}^{(m)}=\sum_{j\in k}Y_{j,h}^{(m)},\qquad
      T_h^{(m)}=\sum_kG_{k,h}^{(m)}.$$

    A multiplicação ocorre antes da soma na própria trajetória. Fatores compartilhados e
    dependência estimada entre entradas afetam automaticamente a distribuição dos agregados.
    """),
    code(r"""
    def sum_nodes(nodes: list[ForecastVariable]) -> ForecastVariable:
        if not nodes:
            raise ValueError("não é possível somar uma lista vazia")
        return reduce(add, nodes)

    group_nodes = {
        group: sum_nodes(list(leaves.values()))
        for group, leaves in grouped_expressions.items()
    }
    total_node = sum_nodes(list(group_nodes.values())) if BUILD_GLOBAL_TOTAL else None

    outputs = dict(derived_leaf_outputs)
    outputs.update({f"group/{group}": node for group, node in group_nodes.items()})
    if total_node is not None:
        outputs["total"] = total_node

    hierarchy_rows = []
    for group, leaves in grouped_expressions.items():
        for leaf in leaves:
            hierarchy_rows.append({
                "nível": "folha derivada",
                "nó": f"leaf/{group}/{leaf}",
                "pai": f"group/{group}",
            })
        hierarchy_rows.append({
            "nível": "grupo",
            "nó": f"group/{group}",
            "pai": "total" if BUILD_GLOBAL_TOTAL else "—",
        })
    if BUILD_GLOBAL_TOTAL:
        hierarchy_rows.append({"nível": "total", "nó": "total", "pai": "—"})
    pd.DataFrame(hierarchy_rows)
    """),
    md(r"""
    ## 7. Backtest das expressões

    O ciclo avalia modelos em duas escalas diferentes:

    1. pré-seleção pelo CRPS marginal de cada entrada primitiva;
    2. escolha final pelo CRPS normalizado das folhas derivadas, grupos e total.

    Com `top_k=2` e $p$ entradas, existiriam até $2^p$ combinações. O tutorial usa `top_k=1` para
    ser rápido; em produção, aumente-o somente com `max_combinations` e orçamento explícitos.
    `OUTPUT_WEIGHTS` dá peso adicional ao total sem retirar as folhas e grupos da avaliação.
    """),
    code(r"""
    primitive_series = [engine.get_series(name) for name in forecast_leaves]
    common_index = primitive_series[0].timestamps
    for series in primitive_series[1:]:
        common_index = common_index.intersection(series.timestamps)
    common_n = len(common_index)
    initial_train_size = max(30, int(0.50 * common_n))
    backtest_step = max(1, (common_n - initial_train_size) // 10)

    backtest_config = BacktestConfig(
        initial_train_size=initial_train_size,
        horizon=1,
        step=backtest_step,
        confidence=0.95,
        seed=RANDOM_SEED,
    )
    composition_policy = CompositionPolicy(
        preselect_top_k=PRESELECT_TOP_K,
        max_combinations=MAX_COMBINATIONS,
        dependency="residual_copula",
        min_paired_errors=3,
        n_backtest_paths=BACKTEST_PATHS,
        coverage_range=None,
        output_weights=OUTPUT_WEIGHTS,
    )
    training = engine.backtest_expression(
        outputs,
        backtest=backtest_config,
        candidate_policy=candidate_policy,
        policy=composition_policy,
        promote_as="derived-hierarchy-champion",
    )
    pd.Series({
        "histórico comum": common_n,
        "treino inicial": initial_train_size,
        "passo": backtest_step,
        "entradas modeladas": len(forecast_leaves),
        "saídas avaliadas": len(outputs),
        "combinações avaliadas": len(training.scores),
        **training.metrics,
    }, name="treinamento das expressões").to_frame()
    """),
    code(r"""
    selected_models = pd.DataFrame([
        {
            "entrada primitiva": name,
            "modelo": repr(spec),
            "model_run_id": training.selected_model_runs[name],
        }
        for name, spec in training.selected_specs.items()
    ])
    selected_models
    """),
    md(r"""
    ## 8. Dependência entre entradas, não entre folhas derivadas

    A matriz abaixo pertence aos erros PIT das entradas primitivas. As folhas $Y_j$ não recebem
    uma cópula adicional: sua dependência é consequência dos caminhos conjuntos dos $X$, dos nós
    compartilhados e das funções $g_j$.

    Shrinkage alto é esperado com poucas observações e muitas entradas. Isso deve aparecer no
    diagnóstico, não ser escondido como se a dependência tivesse sido estimada com precisão.
    """),
    code(r"""
    dependency = training.dependency
    correlation = pd.DataFrame(
        dependency.correlation,
        index=dependency.variable_names,
        columns=dependency.variable_names,
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(correlation)), correlation.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(correlation)), correlation.index)
    for row in range(len(correlation)):
        for column in range(len(correlation)):
            ax.text(column, row, f"{correlation.iloc[row, column]:.2f}",
                    ha="center", va="center", fontsize=8)
    ax.set_title(
        f"PIT das entradas — pares={dependency.paired_errors}, "
        f"shrinkage={dependency.shrinkage:.2f}"
    )
    fig.colorbar(image, ax=ax, shrink=.8); plt.tight_layout(); plt.show()
    print({"fallback": dependency.fallback, "motivo": dependency.reason})
    """),
    md(r"""
    ## 9. Forecast de folhas derivadas, grupos e total

    `compose()` reconstrói as distribuições marginais congeladas no `CompositionRun`, coordena seus
    caminhos, avalia todas as funções privadas e agrega os resultados. Nenhum $Y_j$ é previsto por
    um modelo univariado independente.
    """),
    code(r"""
    forecast = engine.compose(
        outputs,
        model_ref="derived-hierarchy-champion",
        horizon=FORECAST_HORIZON,
        n_paths=N_FORECAST_PATHS,
        seed=RANDOM_SEED + 1,
    )

    summary_rows = []
    for node_name, distribution in forecast.distributions.items():
        quantiles = distribution.quantile([0.05, 0.50, 0.95])
        for step, target in enumerate(forecast.targets):
            summary_rows.append({
                "nó": node_name,
                "target": target,
                "média": distribution.mean[step],
                "p05": quantiles[0, step],
                "mediana": quantiles[1, step],
                "p95": quantiles[2, step],
            })
    forecast_summary = pd.DataFrame(summary_rows)
    forecast_summary.head(14).round(4)
    """),
    code(r"""
    nodes_to_plot = list(derived_leaf_outputs) + [
        name for name in forecast.distributions if name.startswith("group/")
    ]
    if BUILD_GLOBAL_TOTAL:
        nodes_to_plot.append("total")
    fig, ax = plt.subplots(figsize=(13, 6))
    for node_name in nodes_to_plot:
        distribution = forecast.distributions[node_name]
        line, = ax.plot(forecast.targets, distribution.mean, marker="o", label=node_name)
        if node_name == "total":
            bounds = distribution.quantile([0.05, 0.95])
            ax.fill_between(
                forecast.targets, bounds[0], bounds[1], alpha=.14, color=line.get_color()
            )
    ax.set(title="Folhas derivadas, grupos e total", xlabel="target", ylabel="valor")
    ax.legend(ncol=3, fontsize=8); plt.show()
    """),
    md(r"""
    ## 10. Prova de coerência caminho a caminho

    A igualdade relevante não é apenas entre médias. Para cada linha $m$ e horizonte $h$, a soma
    das matrizes das folhas deve reproduzir o grupo, e a soma dos grupos deve reproduzir o total.
    """),
    code(r"""
    coherence_rows = []
    for group, leaves in grouped_expressions.items():
        leaf_sum = sum(
            forecast.paths[f"leaf/{group}/{leaf}"] for leaf in leaves
        )
        error = np.max(np.abs(forecast.paths[f"group/{group}"] - leaf_sum))
        coherence_rows.append({"nó": f"group/{group}", "erro_máximo": error})

    if BUILD_GLOBAL_TOTAL:
        group_sum = sum(forecast.paths[f"group/{group}"] for group in group_nodes)
        total_error = np.max(np.abs(forecast.paths["total"] - group_sum))
        coherence_rows.append({"nó": "total", "erro_máximo": total_error})

    coherence = pd.DataFrame(coherence_rows)
    assert (coherence["erro_máximo"] <= 1e-12).all()
    coherence
    """),
    md(r"""
    ## 11. Risco sobre qualquer saída derivada

    Como toda saída é uma `EmpiricalDistribution`, a mesma API de risco funciona para uma folha,
    um grupo ou o total sem refazer o forecast.
    """),
    code(r"""
    risk_node = "total" if BUILD_GLOBAL_TOTAL else f"group/{next(iter(group_nodes))}"
    risk_distribution = forecast.distributions[risk_node]
    risk_limit = float(np.quantile(risk_distribution.paths, 0.80))
    risk = engine.assess_risk(
        risk_distribution,
        ThresholdRisk(limit=risk_limit, n_paths=2_000, seed=RANDOM_SEED + 2),
    )
    pd.DataFrame({
        "target": forecast.targets,
        "probabilidade marginal": risk.marginal_probability,
        "primeira passagem": risk.first_passage_probability,
    }).assign(
        nó=risk_node,
        limite=risk_limit,
        probabilidade_em_algum_horizonte=risk.any_horizon_probability,
    ).round(3)
    """),
    md(r"""
    ## 12. Limites operacionais

    - O número relevante para a explosão combinatória é o de **entradas primitivas**, não o de
      folhas derivadas. Com $p$ entradas e `top_k=k`, a grade ingênua tem $k^p$ combinações.
    - Muitos $X$ e poucos erros emparelhados levam a shrinkage forte ou fallback independente.
    - A mesma série econômica deve reutilizar o mesmo `ForecastVariable`; criar cópias semânticas
      diferentes quebra o compartilhamento exato de caminhos.
    - `known` e `scenario` são condicionais e têm variância zero dentro de uma execução.
    - Grupos e total não corrigem as médias dos $X$; não há feedback agregado nem MinT nesta versão.
    - O hash protege a identidade da função privada, mas sua aplicação deve reconstruí-la antes de
      chamar `compose()` e incrementar a versão quando mudar o comportamento.

    Consulte o [método de composição](../docs/metodo_composicao_probabilistica.md) para hipóteses,
    validação e limitações completas.
    """),
    md(r"""
    ## 13. Exercícios

    1. Adicione `item_3` a `group_north` usando as mesmas entradas compartilhadas.
    2. Crie uma nova entrada aleatória usada simultaneamente por folhas dos dois grupos.
    3. Troque `private_g` por $g(a,b,c)=a(b+c)$ e incremente `version` para `v2`.
    4. Use `engine.variable(..., mode="known")` em um dos operandos e forneça vintages.
    5. Troque `dependency` para `"independent"` e compare a dispersão do total.
    6. Aumente `PRESELECT_TOP_K` para 2 e explique por que `MAX_COMBINATIONS` precisa permanecer
       limitado.

    <details><summary><strong>Gabarito orientativo</strong></summary>

    1. Apenas `GROUPED_EXPRESSIONS` muda; grupo, total e verificações incorporam a folha.
    2. Reutilize o mesmo objeto em ambas as expressões para compartilhar exatamente os caminhos.
    3. O novo hash impede que um `CompositionRun` antigo seja usado silenciosamente.
    4. Vintages impedem que o backtest enxergue planos que ainda não existiam na origem.
    5. A coerência algébrica permanece; quantis e risco mudam porque a distribuição conjunta muda.
    6. Com $p$ entradas, até $2^p$ combinações podem existir antes do limite.

    </details>
    """),
    code(r"""
    # Testes rápidos do tutorial.
    assert len(training.selected_model_runs) == len(forecast_leaves)
    assert set(derived_leaf_outputs).issubset(forecast.distributions)
    assert forecast.targets.tz is not None
    assert forecast.paths[risk_node].shape == (N_FORECAST_PATHS, FORECAST_HORIZON)
    assert (coherence["erro_máximo"] <= 1e-12).all()
    assert repository.get_composition_alias(
        "derived-hierarchy-champion"
    ) == training.composition_run_id
    assert all("function" not in str(node.to_payload()) for node in derived_leaf_outputs.values())
    print("Testes rápidos do Notebook 16 passaram.")
    """),
    md(r"""
    ## Conclusão

    O ciclo agora começa nas séries primitivas, passa por distribuições conjuntas e funções $g$ e
    somente então forma folhas, grupos e total. Esse é o caso em que multiplicações e somas devem
    acontecer sobre caminhos compartilhados: a hierarquia agrega **expressões probabilísticas**,
    não forecasts univariados independentes das folhas de negócio.
    """),
]


if __name__ == "__main__":
    write_notebook(PATH, CELLS, 16)
