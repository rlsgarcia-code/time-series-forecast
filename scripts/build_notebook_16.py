"""Constrói o Notebook 16: ciclo hierárquico a partir de dict de pandas Series."""

from pathlib import Path

from notebook_builder import code, md, setup_cell, write_notebook


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "16_ciclo_hierarquico_dict.ipynb"


CELLS = [
    md(r"""
    # Notebook 16 — Ciclo hierárquico a partir de um dicionário de séries

    **Entrada:** um `dict[str, list[pd.Series]]`. Cada chave identifica um grupo no nível
    agregado; a lista contém as séries-folha pertencentes ao grupo.

    ```python
    GROUPED_SERIES = {
        "grupo_a": [serie_a1, serie_a2],
        "grupo_b": [serie_b1, serie_b2, serie_b3],
    }
    ```

    O notebook constrói automaticamente forecasts para as folhas, soma as trajetórias dentro de
    cada grupo e, opcionalmente, cria `TOTAL = soma(grupos)`.

    **Ao final você será capaz de:** validar o dicionário, auditar valores ausentes, caracterizar
    as folhas, selecionar modelos, estimar dependência, produzir a árvore probabilística e provar
    sua coerência caminho a caminho.
    """),
    md(r"""
    ## 1. Convenção da hierarquia

    Neste notebook usamos os nomes:

    ```text
    TOTAL opcional
      ├── chave do dicionário / grupo 1
      │     ├── pandas Series 1
      │     └── pandas Series 2
      └── chave do dicionário / grupo 2
            ├── pandas Series 3
            └── pandas Series 4
    ```

    Se, no seu domínio, as chaves já representam o nível mais alto desejado, configure
    `BUILD_GLOBAL_TOTAL=False`. Nenhuma reconciliação posterior é necessária para as somas: todos
    os níveis são calculados sobre exatamente as mesmas trajetórias das folhas.
    """),
    setup_cell(
        "from functools import reduce\n"
        "from operator import add\n"
        "from examples.memory_repository import MemoryRepository\n"
        "from time_series_lab import (\n"
        "    BacktestConfig, CandidatePolicy, CompositionPolicy, ForecastEngine,\n"
        "    ThresholdRisk, ValidationError,\n"
        ")"
    ),
    md(r"""
    ## 2. Carregue suas séries

    Edite somente a função abaixo. Cada item precisa ser uma `pd.Series` numérica com
    `DatetimeIndex`. Dê nomes únicos às séries dentro de cada grupo: o identificador interno será
    `grupo/nome`.

    O exemplo padrão cria dois grupos e quatro folhas com choques globais e de grupo. Alguns `NaN`
    internos foram colocados de propósito para exercitar a limpeza causal.
    """),
    code(r"""
    # ==================== EDITE SOMENTE ESTA CÉLULA ====================
    def load_grouped_series() -> dict[str, list[pd.Series]]:
        rng = np.random.default_rng(1601)
        index = pd.date_range("2023-01-01", periods=62, freq="D", tz="UTC")
        global_shock = rng.normal(0, 0.45, len(index))
        result: dict[str, list[pd.Series]] = {}

        for group_number, group_name in enumerate(["grupo_norte", "grupo_sul"]):
            group_shock = rng.normal(0, 0.55, len(index))
            leaves = []
            for leaf_number in range(2):
                noise = rng.normal(0, 0.45, len(index))
                values = np.zeros(len(index))
                for t in range(1, len(index)):
                    values[t] = (
                        0.12 * (group_number + 1)
                        + 0.58 * values[t - 1]
                        + global_shock[t]
                        + 0.65 * group_shock[t]
                        + noise[t]
                    )
                leaves.append(pd.Series(values, index=index, name=f"item_{leaf_number + 1}"))
            result[group_name] = leaves

        result["grupo_norte"][0].iloc[11] = np.nan
        result["grupo_sul"][1].iloc[27] = np.nan
        return result

    GROUPED_SERIES = load_grouped_series()
    GROUPED_SERIES
    """),
    md(r"""
    ## 3. Configuração

    `ffill` substitui um valor interno ausente pelo último valor disponível e não olha o futuro.
    Valores ausentes no último timestamp são rejeitados, pois preenchê-los inventaria a observação
    mais recente. O backtest usa poucos candidatos no exemplo para permanecer rápido; aumente os
    limites somente quando houver dados e orçamento computacional suficientes.
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

    if MISSING_VALUE_POLICY not in {"ffill", "error"}:
        raise ValueError("MISSING_VALUE_POLICY deve ser 'ffill' ou 'error'")
    """),
    md(r"""
    ## 4. Validação e limpeza explícitas

    A função abaixo não está escondida no pacote para que as decisões de entrada permaneçam
    visíveis. Ela verifica estrutura, nomes, tempo, frequência e finitude; remove somente `NaN`
    iniciais, aplica a política aos ausentes internos e produz uma tabela de auditoria.

    Históricos podem começar em datas diferentes, mas todas as folhas precisam terminar no mesmo
    cutoff, ter a mesma frequência e possuir uma interseção suficiente para o backtest conjunto.
    """),
    code(r"""
    def prepare_grouped_series(
        grouped: dict[str, list[pd.Series]],
        *,
        timezone_name: str,
        missing_policy: str,
        minimum_observations: int,
    ) -> tuple[dict[str, dict[str, pd.Series]], pd.DataFrame]:
        if not isinstance(grouped, dict) or not grouped:
            raise TypeError("GROUPED_SERIES deve ser um dict não vazio")

        cleaned: dict[str, dict[str, pd.Series]] = {}
        audit_rows: list[dict] = []
        seen_ids: set[str] = set()
        frequencies: set[str] = set()
        cutoffs: set[pd.Timestamp] = set()

        for raw_group, series_list in grouped.items():
            group = str(raw_group).strip()
            if not group:
                raise ValueError("as chaves dos grupos não podem ser vazias")
            if not isinstance(series_list, (list, tuple)) or not series_list:
                raise TypeError(f"{group}: o valor deve ser uma lista não vazia de pd.Series")
            cleaned[group] = {}

            for position, source in enumerate(series_list, start=1):
                if not isinstance(source, pd.Series):
                    raise TypeError(f"{group}[{position}] não é uma pd.Series")
                leaf_name = str(source.name or f"serie_{position}").strip()
                leaf_id = f"{group}/{leaf_name}"
                if leaf_id in seen_ids:
                    raise ValueError(f"identificador duplicado: {leaf_id}")
                seen_ids.add(leaf_id)

                if not isinstance(source.index, pd.DatetimeIndex):
                    raise TypeError(f"{leaf_id}: índice deve ser DatetimeIndex")
                series = source.sort_index().copy()
                if not series.index.is_unique:
                    raise ValueError(f"{leaf_id}: timestamps duplicados")
                if series.index.tz is None:
                    series.index = series.index.tz_localize(timezone_name)

                frequency = pd.infer_freq(series.index)
                if frequency is None:
                    raise ValueError(f"{leaf_id}: não foi possível inferir frequência")
                expected = pd.date_range(series.index[0], periods=len(series), freq=frequency)
                if not series.index.equals(expected):
                    raise ValueError(f"{leaf_id}: existem gaps; regularize antes da modelagem")

                numeric = pd.to_numeric(series, errors="coerce").astype(float)
                numeric = numeric.replace([np.inf, -np.inf], np.nan)
                if numeric.notna().sum() == 0:
                    raise ValueError(f"{leaf_id}: nenhum valor finito")
                if pd.isna(numeric.iloc[-1]):
                    raise ValueError(f"{leaf_id}: o último valor não pode ser ausente")

                first_valid = int(np.flatnonzero(numeric.notna().to_numpy())[0])
                trimmed = numeric.iloc[first_valid:].copy()
                missing_count = int(trimmed.isna().sum())
                if missing_count and missing_policy == "error":
                    raise ValueError(f"{leaf_id}: {missing_count} valores ausentes")
                if missing_policy == "ffill":
                    trimmed = trimmed.ffill()
                if trimmed.isna().any() or len(trimmed) < minimum_observations:
                    raise ValueError(
                        f"{leaf_id}: limpeza deixou menos de {minimum_observations} observações"
                    )

                trimmed.name = leaf_name
                cleaned[group][leaf_id] = trimmed
                frequencies.add(str(frequency))
                cutoffs.add(trimmed.index[-1])
                audit_rows.append({
                    "grupo": group,
                    "folha": leaf_id,
                    "recebidas": len(source),
                    "removidas_no_início": first_valid,
                    "imputadas": missing_count,
                    "observações_finais": len(trimmed),
                    "início": trimmed.index[0],
                    "cutoff": trimmed.index[-1],
                    "frequência": frequency,
                })

        if len(frequencies) != 1:
            raise ValueError(f"todas as folhas devem ter a mesma frequência: {frequencies}")
        if len(cutoffs) != 1:
            raise ValueError("todas as folhas devem possuir o mesmo cutoff")
        return cleaned, pd.DataFrame(audit_rows)
    """),
    code(r"""
    cleaned_groups, input_audit = prepare_grouped_series(
        GROUPED_SERIES,
        timezone_name=TIMEZONE,
        missing_policy=MISSING_VALUE_POLICY,
        minimum_observations=MINIMUM_OBSERVATIONS,
    )
    display(input_audit)
    print(
        f"{len(cleaned_groups)} grupos; "
        f"{sum(len(items) for items in cleaned_groups.values())} folhas validadas."
    )
    """),
    code(r"""
    fig, axes = plt.subplots(len(cleaned_groups), 1, figsize=(12, 3.5 * len(cleaned_groups)),
                             squeeze=False)
    for ax, (group, leaves) in zip(axes[:, 0], cleaned_groups.items()):
        for leaf_id, series in leaves.items():
            ax.plot(series.index, series.values, lw=1, label=leaf_id.split("/", 1)[1])
        ax.set_title(group); ax.legend(ncol=3)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ## 5. Registro e caracterização das folhas

    Cada folha continua sendo uma série univariada: ela recebe caracterização e candidatos próprios.
    A hierarquia não substitui ACF, estacionariedade ou diagnóstico residual. Ela entra depois,
    quando as distribuições marginais precisam ser avaliadas em conjunto.
    """),
    code(r"""
    repository = MemoryRepository()
    engine = ForecastEngine(repository=repository)
    leaf_nodes: dict[str, dict[str, object]] = {}

    for group, leaves in cleaned_groups.items():
        leaf_nodes[group] = {
            leaf_id: engine.variable(leaf_id, series, mode="forecast")
            for leaf_id, series in leaves.items()
        }

    candidate_policy = CandidatePolicy(
        max_candidates=CANDIDATE_BUDGET,
        differencing="auto",
        garch="never",
        markov_switching="never",
        simulation_paths=100,
    )
    characterization_rows = []
    candidate_plans = {}
    for group, leaves in leaf_nodes.items():
        for leaf_id in leaves:
            profile = engine.characterize(leaf_id, max_lag=12)
            plan = engine.propose_candidates(
                leaf_id, policy=candidate_policy, profile=profile, series_kind="level"
            )
            candidate_plans[leaf_id] = plan
            characterization_rows.append({
                "grupo": group,
                "folha": leaf_id,
                "nobs": profile.nobs,
                "ADF p": profile.adf_pvalue,
                "KPSS p": profile.kpss_pvalue,
                "Ljung-Box p": profile.ljung_box_pvalue,
                "candidatos": len(plan.candidates),
            })
    pd.DataFrame(characterization_rows).round(4)
    """),
    code(r"""
    # Exemplo da trilha explicável de uma folha; as demais estão em candidate_plans.
    example_leaf = next(iter(candidate_plans))
    print(candidate_plans[example_leaf].explain())
    display(candidate_plans[example_leaf].to_frame(include_excluded=True))
    """),
    md(r"""
    ## 6. Construção automática do grafo

    Para folhas $Y_{g,1},\ldots,Y_{g,n_g}$ do grupo $g$,

    $$G_g=\sum_{i=1}^{n_g}Y_{g,i},\qquad
      T=\sum_g G_g.$$

    Essas igualdades não são impostas às médias depois do forecast. Elas são avaliadas para cada
    caminho $m$ e horizonte $h$, garantindo
    $T_h^{(m)}=\sum_g\sum_iY_{g,i,h}^{(m)}$ exatamente.
    """),
    code(r"""
    def sum_nodes(nodes: list):
        if not nodes:
            raise ValueError("não é possível somar lista vazia")
        return reduce(add, nodes)

    group_nodes = {
        group: sum_nodes(list(leaves.values()))
        for group, leaves in leaf_nodes.items()
    }
    total_node = sum_nodes(list(group_nodes.values())) if BUILD_GLOBAL_TOTAL else None

    outputs = {}
    for group, leaves in leaf_nodes.items():
        outputs.update({f"leaf/{leaf_id}": node for leaf_id, node in leaves.items()})
        outputs[f"group/{group}"] = group_nodes[group]
    if total_node is not None:
        outputs["total"] = total_node

    hierarchy_table = []
    for group, leaves in leaf_nodes.items():
        for leaf_id in leaves:
            hierarchy_table.append({
                "nível": "folha", "nó": f"leaf/{leaf_id}", "pai": f"group/{group}"
            })
        hierarchy_table.append({
            "nível": "grupo", "nó": f"group/{group}",
            "pai": "total" if BUILD_GLOBAL_TOTAL else "—",
        })
    if BUILD_GLOBAL_TOTAL:
        hierarchy_table.append({"nível": "total", "nó": "total", "pai": "—"})
    pd.DataFrame(hierarchy_table)
    """),
    md(r"""
    ## 7. Backtest composto

    A seleção ocorre em duas etapas:

    1. cada folha retém `PRESELECT_TOP_K` candidatos pelo desempenho marginal;
    2. combinações limitadas por `MAX_COMBINATIONS` são avaliadas pelas saídas da árvore.

    Com $k$ candidatos e $n$ folhas, a grade ingênua teria $k^n$ combinações. O padrão `k=1`
    mantém o exemplo escalável e ainda seleciona cada folha fora da amostra. Use `k=2` somente com
    orçamento explícito e mantenha `MAX_COMBINATIONS` limitado.
    """),
    code(r"""
    common_start = max(series.index[0] for group in cleaned_groups.values() for series in group.values())
    common_end = min(series.index[-1] for group in cleaned_groups.values() for series in group.values())
    common_n = min(
        int(((series.loc[common_start:common_end]).shape[0]))
        for group in cleaned_groups.values() for series in group.values()
    )
    initial_train_size = max(24, int(0.58 * common_n))
    available_origins = common_n - initial_train_size
    backtest_step = max(1, available_origins // 4)

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
        coverage_range=None,  # poucas origens no exemplo; cobertura continua sendo exibida
    )
    training = engine.backtest_expression(
        outputs,
        backtest=backtest_config,
        candidate_policy=candidate_policy,
        policy=composition_policy,
        promote_as="hierarchy-champion",
    )
    pd.Series({
        "histórico comum": common_n,
        "treino inicial": initial_train_size,
        "passo": backtest_step,
        "combinações avaliadas": len(training.scores),
        **training.metrics,
    }, name="treinamento hierárquico").to_frame()
    """),
    code(r"""
    selected_models = pd.DataFrame([
        {"folha": name, "modelo": repr(spec), "model_run_id": training.selected_model_runs[name]}
        for name, spec in training.selected_specs.items()
    ])
    selected_models
    """),
    md(r"""
    ## 8. Dependência entre folhas

    O motor sincroniza erros PIT de forecast, aplica shrinkage e usa uma cópula de ranks. Uma
    correlação positiva significa que caminhos relativamente altos tendem a ser associados; não
    significa causalidade nem que uma folha foi usada para prever a média da outra.
    """),
    code(r"""
    dependency = training.dependency
    correlation = pd.DataFrame(
        dependency.correlation,
        index=dependency.variable_names,
        columns=dependency.variable_names,
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(correlation)), correlation.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(correlation)), correlation.index)
    for row in range(len(correlation)):
        for column in range(len(correlation)):
            ax.text(column, row, f"{correlation.iloc[row, column]:.2f}", ha="center", va="center")
    ax.set_title(
        f"Dependência residual — pares={dependency.paired_errors}, "
        f"shrinkage={dependency.shrinkage:.2f}"
    )
    fig.colorbar(image, ax=ax, shrink=.8); plt.tight_layout(); plt.show()
    print(dependency.reason)
    """),
    md(r"""
    ## 9. Forecast de toda a árvore em uma chamada

    `compose()` usa os modelos congelados pelo alias, verifica revisões e calcula todas as saídas
    com os mesmos índices de caminhos. As distribuições das folhas permanecem disponíveis; os
    grupos e o total são `EmpiricalDistribution` como qualquer outra saída.
    """),
    code(r"""
    forecast = engine.compose(
        outputs,
        model_ref="hierarchy-champion",
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
    forecast_summary.head(12).round(3)
    """),
    code(r"""
    nodes_to_plot = [name for name in forecast.distributions if name.startswith("group/")]
    if BUILD_GLOBAL_TOTAL:
        nodes_to_plot.append("total")
    fig, ax = plt.subplots(figsize=(12, 5))
    for node_name in nodes_to_plot:
        distribution = forecast.distributions[node_name]
        bounds = distribution.quantile([0.05, 0.95])
        line, = ax.plot(forecast.targets, distribution.mean, marker="o", label=node_name)
        ax.fill_between(forecast.targets, bounds[0], bounds[1], alpha=.12, color=line.get_color())
    ax.set(title="Forecast dos grupos e do total", xlabel="target", ylabel="valor")
    ax.legend(); plt.show()
    """),
    md(r"""
    ## 10. Prova de coerência caminho a caminho

    Comparar apenas médias não é suficiente. A checagem correta usa a matriz completa
    `(n_paths, horizon)`. O erro abaixo deve ser numericamente zero para cada grupo e para o total.
    """),
    code(r"""
    coherence_rows = []
    for group, leaves in leaf_nodes.items():
        leaf_sum = sum(forecast.paths[f"leaf/{leaf_id}"] for leaf_id in leaves)
        error = np.max(np.abs(forecast.paths[f"group/{group}"] - leaf_sum))
        coherence_rows.append({"nó": f"group/{group}", "erro_máximo": error})

    if BUILD_GLOBAL_TOTAL:
        group_sum = sum(forecast.paths[f"group/{group}"] for group in group_nodes)
        total_error = np.max(np.abs(forecast.paths["total"] - group_sum))
        coherence_rows.append({"nó": "total", "erro_máximo": total_error})

    coherence = pd.DataFrame(coherence_rows)
    assert (coherence["erro_máximo"] == 0.0).all()
    coherence
    """),
    md(r"""
    ## 11. Risco no total ou em qualquer grupo

    Escolher um nó não exige refazer o forecast. Aqui o limite é apenas didático; em uso real ele
    deve vir de uma regra de domínio versionada.
    """),
    code(r"""
    risk_node = "total" if BUILD_GLOBAL_TOTAL else next(
        name for name in forecast.distributions if name.startswith("group/")
    )
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
    ## 12. Escala, limites e interpretação

    - O custo cresce com folhas × candidatos × origens; `top_k=1` evita explosão combinatória.
    - A implementação atual executa o treinamento composto localmente e de forma sequencial. Para
      centenas de folhas, distribua os backtests marginais na aplicação e preserve os mesmos seeds.
    - Grupos compartilham incerteza por meio da cópula, mas não corrigem as médias uns dos outros.
    - Uma chave com muitas folhas recebe o mesmo peso de saída que as demais somente se a política
      de `output_weights` assim for configurada; revise pesos conforme a decisão de negócio.
    - Séries com frequências diferentes exigem uma etapa anterior de agregação/alinhamento.
    - Monitoramento composto e reconciliação com forecasts independentes dos agregados continuam
      fora desta versão.
    """),
    md(r"""
    ## 13. Exercícios

    1. Adicione uma terceira série a `grupo_norte` e confirme que nenhuma outra célula muda.
    2. Configure `BUILD_GLOBAL_TOTAL=False` e observe quais saídas permanecem.
    3. Use séries com inícios diferentes e confirme que o backtest usa a interseção comum.
    4. Troque `dependency` para `"independent"` e compare a largura do total.
    5. Aumente `PRESELECT_TOP_K` para 2, limite combinações em 8 e compare o CRPS.
    6. Forneça uma série com `NaN` no último ponto e explique por que a validação deve falhar.

    <details><summary><strong>Gabarito orientativo</strong></summary>

    1. A nova folha aparece em `leaf_nodes`, no grupo e no total automaticamente.
    2. As folhas e grupos continuam; somente `total` deixa de ser construído.
    3. `common_n` diminui para o intervalo disponível em todas as folhas.
    4. A soma continua coerente, mas sua dispersão muda quando a dependência é removida.
    5. Com quatro folhas, existem até $2^4=16$ combinações; somente oito serão avaliadas.
    6. `ffill` no último ponto inventaria o estado mais recente e esconderia ausência operacional.

    </details>
    """),
    code(r"""
    # Testes rápidos do tutorial.
    expected_leaf_count = sum(len(leaves) for leaves in cleaned_groups.values())
    assert len(training.selected_model_runs) == expected_leaf_count
    assert forecast.targets.tz is not None
    assert forecast.paths[risk_node].shape == (N_FORECAST_PATHS, FORECAST_HORIZON)
    assert (coherence["erro_máximo"] == 0.0).all()
    assert repository.get_composition_alias("hierarchy-champion") == training.composition_run_id
    print("Testes rápidos do Notebook 16 passaram.")
    """),
    md(r"""
    ## Conclusão

    Um único dicionário de séries percorreu validação, caracterização, seleção, dependência,
    forecast probabilístico, agregação hierárquica e risco. A estrutura do dicionário virou um
    grafo de trajetórias coerentes sem exigir que a aplicação exponha uma fórmula específica.
    """),
]


if __name__ == "__main__":
    write_notebook(PATH, CELLS, 16)

