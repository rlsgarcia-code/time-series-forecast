"""Constrói o Notebook 15: composição probabilística e hierárquica."""

from pathlib import Path

from notebook_builder import code, md, setup_cell, write_notebook


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "15_composicao_probabilistica_hierarquica.ipynb"


CELLS = [
    md(r"""
    # Notebook 15 — Composição probabilística e hierárquica

    **Público:** usuários do `ForecastEngine` que precisam transformar várias previsões em
    resultados derivados e agregá-los em uma árvore.

    **Pré-requisitos:** distribuição preditiva, rolling-origin, CRPS e leitura do Notebook 14.

    **Ao final você será capaz de:**

    1. representar séries previstas, conhecidas e constantes no mesmo grafo;
    2. entender por que multiplicação exige trajetórias conjuntas e covariância;
    3. selecionar modelos pelo resultado derivado, não apenas pelas entradas;
    4. construir folhas, grupos e total coerentes caminho a caminho;
    5. auditar dependência, causalidade, risco e limitações da primeira versão.

    O exemplo é totalmente sintético e genérico. Nenhuma regra de negócio privada é armazenada
    neste notebook ou nos artefatos persistidos pelo motor.
    """),
    md(r"""
    ## 1. O fluxo

    ```text
    séries históricas ─→ modelos marginais ─→ trajetórias independentes
                                                  │
    erros rolling-origin ─→ cópula residual ──────┤
                                                  ▼
    séries conhecidas ─────────────────────→ trajetórias conjuntas
                                                  │
                                                  ▼
                                  grafo → folhas → grupos → total → risco
    ```

    Os agregados são calculados sobre os mesmos caminhos das folhas. Nesta versão eles não
    corrigem a média das folhas: há coerência contábil, mas ainda não há reconciliação MinT nem
    modelo global multivariado.
    """),
    setup_cell(
        "from examples.memory_repository import MemoryRepository\n"
        "from time_series_lab import (\n"
        "    BacktestConfig, CandidatePolicy, CompositionPolicy, ForecastEngine,\n"
        "    ThresholdRisk,\n"
        ")"
    ),
    md(r"""
    ## 2. Dados sintéticos com dependência conhecida

    Simularemos dois drivers AR(1) cujas inovações contemporâneas têm correlação positiva. A
    dependência relevante não é a correlação entre níveis observados, mas a correlação entre os
    erros de previsão sincronizados: é essa informação ainda não explicada pelos modelos que deve
    coordenar os caminhos futuros.
    """),
    code(r"""
    rng = np.random.default_rng(1501)
    n_history, horizon = 54, 6
    full_index = pd.date_range("2024-01-01", periods=n_history + horizon, freq="D", tz="UTC")
    innovations = rng.multivariate_normal(
        mean=[0.0, 0.0], cov=[[1.0, 0.70], [0.70, 1.0]], size=n_history
    )
    drivers = np.zeros((n_history, 2))
    for t in range(1, n_history):
        drivers[t, 0] = 0.15 + 0.62 * drivers[t - 1, 0] + innovations[t, 0]
        drivers[t, 1] = -0.05 + 0.48 * drivers[t - 1, 1] + innovations[t, 1]

    driver_a = pd.Series(drivers[:, 0], index=full_index[:n_history], name="driver_a")
    driver_b = pd.Series(drivers[:, 1], index=full_index[:n_history], name="driver_b")
    known_scale = pd.Series(
        1.0 + 0.002 * np.arange(len(full_index)), index=full_index, name="known_scale"
    )

    # Cada alvo já era conhecido um período antes. Esse é o contrato causal de vintages.
    known_vintages = pd.DataFrame({
        "as_of": full_index - pd.Timedelta(days=1),
        "target": full_index,
        "value": known_scale.to_numpy(),
    })
    pd.concat([driver_a, driver_b], axis=1).head()
    """),
    code(r"""
    ax = pd.concat([driver_a, driver_b], axis=1).plot(figsize=(12, 4), lw=1.2)
    ax.set(title="Drivers correlacionados", xlabel="tempo", ylabel="valor")
    plt.show()
    """),
    md(r"""
    ## 3. Um único objeto para folhas e expressões

    `mode="forecast"` registra a série e informa que o motor deve modelá-la. `known` representa
    uma trajetória futura determinística, enquanto `constant` difunde um escalar pelo horizonte.
    Os operadores devolvem outro `ForecastVariable`; não é necessário criar classes distintas para
    soma, produto ou nó hierárquico.
    """),
    code(r"""
    repository = MemoryRepository()
    engine = ForecastEngine(repository=repository)

    a = engine.variable("driver_a", driver_a, mode="forecast")
    b = engine.variable("driver_b", driver_b, mode="forecast")
    scale = engine.variable(
        "known_scale", known_scale, mode="known", vintages=known_vintages
    )
    baseline = engine.variable("baseline", 1.0, mode="constant")

    leaf_1 = scale * a
    leaf_2 = b + baseline
    leaf_3 = 0.50 * a - 0.25 * b
    group = leaf_1 + leaf_2
    total = group + leaf_3
    interaction = a * b

    outputs = {
        "leaf/1": leaf_1,
        "leaf/2": leaf_2,
        "leaf/3": leaf_3,
        "group": group,
        "total": total,
        "interaction": interaction,
    }
    pd.Series({name: node.expression_id for name, node in outputs.items()}, name="expression_id")
    """),
    md(r"""
    ## 4. Formulação matemática

    Para uma função $g$ de entradas aleatórias $mathbf X$,

    $$Y=g(\mathbf X),\qquad
      E[Y]=\int g(\mathbf x)p(\mathbf x)\,d\mathbf x
      \approx \frac1M\sum_{m=1}^M g(\mathbf X^{(m)}).$$

    Em geral, não podemos aplicar a função apenas às médias. Para um produto,

    $$E[XY]=E[X]E[Y]+\operatorname{Cov}(X,Y).$$

    Portanto, multiplicar os forecasts pontuais descartaria justamente o termo que mede como os
    drivers variam juntos. O grafo resolve isso aplicando a operação a cada trajetória conjunta e
    resumindo a distribuição somente depois.
    """),
    md(r"""
    ## 5. Seleção em duas etapas

    Primeiro, a `CandidatePolicy` mantém poucos modelos plausíveis por driver. Depois, o motor
    avalia combinações desses modelos pelo CRPS normalizado das **saídas**. Isso permite que um
    modelo marginalmente segundo colocado vença quando produz uma distribuição derivada melhor.

    O backtest usa somente origens comuns. A série conhecida é recuperada pelo último vintage com
    `as_of <= cutoff`; usar o plano atual retrospectivamente seria vazamento de informação.
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
        preselect_top_k=1,
        max_combinations=1,
        dependency="residual_copula",
        min_paired_errors=3,
        n_backtest_paths=150,
        coverage_range=None,  # amostra didática curta; ainda exibiremos a cobertura
    )
    training = engine.backtest_expression(
        outputs,
        backtest=BacktestConfig(initial_train_size=32, horizon=1, step=6, seed=1501),
        candidate_policy=candidate_policy,
        policy=composition_policy,
        promote_as="composition-champion",
    )
    pd.Series(training.metrics, name="seleção composta").to_frame()
    """),
    code(r"""
    score_table = pd.DataFrame([
        {
            "combination_id": score["combination_id"],
            "CRPS normalizado": score["aggregate_normalized_crps"],
            "cobertura": score["aggregate_coverage"],
            **{f"modelo {name}": spec for name, spec in score["specifications"].items()},
        }
        for score in training.scores
    ])
    score_table
    """),
    md(r"""
    ## 6. Dependência explicável

    Para cada driver e origem calculamos o PIT do realizado na distribuição prevista e o levamos
    à escala normal:

    $$z_{i,t}=\Phi^{-1}\!\left(\widehat F_{i,t}(y_{i,t})\right).$$

    A correlação desses erros sincronizados recebe shrinkage para a identidade. Se houver poucos
    pares, o fallback é independência e o motivo fica no artefato. A reordenação por ranks preserva
    cada conjunto de caminhos marginais e induz a dependência entre seus graus de severidade.
    """),
    code(r"""
    dependency = training.dependency
    display(pd.Series({
        "erros emparelhados": dependency.paired_errors,
        "shrinkage": dependency.shrinkage,
        "fallback": dependency.fallback,
        "motivo": dependency.reason,
    }, name="dependência").to_frame())
    pd.DataFrame(
        dependency.correlation,
        index=dependency.variable_names,
        columns=dependency.variable_names,
    ).round(3)
    """),
    md(r"""
    ## 7. Forecast conjunto e coerência da árvore

    `compose()` não treina nada implicitamente. Ele exige o `CompositionRun` ou alias congelado,
    confere os hashes das expressões e as revisões das entradas, gera caminhos marginais, aplica a
    dependência e só então percorre o grafo.
    """),
    code(r"""
    forecast = engine.compose(
        outputs,
        model_ref="composition-champion",
        horizon=horizon,
        n_paths=2_000,
        seed=1502,
    )
    summary = pd.DataFrame({
        (name, "média"): distribution.mean
        for name, distribution in forecast.distributions.items()
    }, index=forecast.targets)
    summary.round(3)
    """),
    code(r"""
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, color in [("leaf/1", COLORS["blue"]), ("group", COLORS["orange"]),
                        ("total", COLORS["purple"])]:
        distribution = forecast.distributions[name]
        bounds = distribution.quantile([0.05, 0.95])
        ax.plot(forecast.targets, distribution.mean, marker="o", label=name, color=color)
        ax.fill_between(forecast.targets, bounds[0], bounds[1], color=color, alpha=.12)
    ax.set(title="Distribuições derivadas sobre os mesmos caminhos", xlabel="alvo")
    ax.legend(); plt.show()
    """),
    code(r"""
    group_error = np.max(np.abs(
        forecast.paths["group"] - (forecast.paths["leaf/1"] + forecast.paths["leaf/2"])
    ))
    total_error = np.max(np.abs(
        forecast.paths["total"] - (forecast.paths["group"] + forecast.paths["leaf/3"])
    ))
    assert group_error == 0.0 and total_error == 0.0
    pd.Series({"erro máximo do grupo": group_error, "erro máximo do total": total_error})
    """),
    md(r"""
    ## 8. O termo de covariância aparece nos caminhos

    A saída `interaction` foi definida como produto dos dois drivers. A diferença abaixo não é um
    ajuste colocado manualmente: ela surge da dependência das amostras conjuntas.
    """),
    code(r"""
    mean_product = forecast.distributions["interaction"].mean
    product_of_means = (
        np.mean(forecast.paths["leaf/3"] * 0 + forecast.paths["leaf/1"] / known_scale.loc[forecast.targets].to_numpy(), axis=0)
        * np.mean(forecast.paths["leaf/2"] - 1.0, axis=0)
    )
    covariance_effect = mean_product - product_of_means
    pd.DataFrame({
        "E[A B]": mean_product,
        "E[A]E[B]": product_of_means,
        "efeito da covariância": covariance_effect,
    }, index=forecast.targets).round(3)
    """),
    md(r"""
    ## 9. Funções privadas

    Quando os operadores não bastarem, `transform()` recebe uma função vetorizada. O código fica
    na aplicação privada; o motor persiste somente nome opaco, versão e hash. Alterar a função sem
    alterar a versão quebraria a reprodutibilidade e é responsabilidade explícita da aplicação.
    """),
    code(r"""
    def tutorial_transform(values: np.ndarray) -> np.ndarray:
        return np.maximum(values, 0.0)

    nonnegative_total = engine.transform(
        "opaque-example", tutorial_transform, total, version="v1"
    )
    safe_payload = nonnegative_total.to_payload()
    assert "function" not in safe_payload and "operator" not in safe_payload
    safe_payload
    """),
    md(r"""
    ## 10. Risco sobre qualquer nó

    Como cada saída é uma `EmpiricalDistribution`, a mesma API calcula risco marginal, risco em
    qualquer horizonte e primeira passagem. A probabilidade de exceder em algum passo não é a soma
    das probabilidades marginais porque os horizontes pertencem à mesma trajetória.
    """),
    code(r"""
    limit = float(np.quantile(forecast.distributions["total"].paths, 0.80))
    risk = engine.assess_risk(
        forecast.distributions["total"],
        ThresholdRisk(limit=limit, n_paths=2_000, seed=1503),
    )
    pd.DataFrame({
        "probabilidade marginal": risk.marginal_probability,
        "primeira passagem": risk.first_passage_probability,
    }, index=forecast.targets).assign(
        probabilidade_em_algum_horizonte=risk.any_horizon_probability
    ).round(3)
    """),
    md(r"""
    ## 11. Limitações e armadilhas

    - Correlação de níveis não substitui dependência dos erros de forecast.
    - Sem vintages, uma série futura conhecida pode vazar o plano atual para o passado; a promoção
      é bloqueada por padrão.
    - O fallback independente não significa que independência foi demonstrada: significa que não
      houve evidência emparelhada suficiente para estimá-la.
    - A cópula coordena ranks de caminhos; não cria um modelo estrutural multivariado.
    - Os agregados são coerentes por construção, mas ainda não influenciam as médias das folhas.
    - Monitoramento composto e reconciliação estatística pertencem à próxima fase.
    """),
    md(r"""
    ## 12. Exercícios

    1. Troque `dependency` para `"independent"` e compare o intervalo do total.
    2. Remova `known_vintages` e confirme que o backtest causal falha antes da seleção.
    3. Aumente `preselect_top_k` para 2 e observe quantas combinações entram no placar.
    4. Altere a correlação das inovações para $-0{,}70$ e examine `interaction` e o total.
    5. Crie uma quarta folha e prove caminho a caminho que o novo total fecha exatamente.

    <details><summary><strong>Gabarito orientativo</strong></summary>

    1. A média pode mudar pouco, mas a dispersão do total deve mudar.
    2. O erro seguro contém “exige vintages”; `allow_noncausal_known=True` serve apenas para análise
       marcada como não causal e não pode ser promovida.
    3. Com dois drivers, o limite superior é $2^2=4$, respeitando `max_combinations`.
    4. $E[AB]-E[A]E[B]$ deve mudar de sinal em média.
    5. Some as matrizes das folhas e use `np.testing.assert_allclose` contra o total.

    </details>
    """),
    code(r"""
    # Testes rápidos: invariantes essenciais do capítulo.
    assert training.causal
    assert repository.get_composition_alias("composition-champion") == training.composition_run_id
    assert forecast.paths["total"].shape == (2_000, horizon)
    assert set(forecast.distributions) == set(outputs)
    assert np.all(np.isfinite(forecast.paths["total"]))
    print("Testes rápidos do Notebook 15 passaram.")
    """),
    md(r"""
    ## Conclusão

    O motor continuou ajustando modelos univariados transparentes, mas deixou de tratar suas
    saídas como isoladas. Dependência foi estimada com evidência causal, fórmulas foram avaliadas
    sobre caminhos e a hierarquia fechou exatamente sem expor regras privadas nos artefatos.

    **Próximo passo:** o [Notebook 16](16_ciclo_hierarquico_dict.ipynb) recebe um
    `dict[str, list[pd.Series]]` e constrói automaticamente folhas, grupos e total.
    """),
]


if __name__ == "__main__":
    write_notebook(PATH, CELLS, 15)

