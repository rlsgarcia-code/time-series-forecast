from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from examples.memory_repository import MemoryRepository
from time_series_lab import (
    BacktestConfig,
    CandidatePolicy,
    CompositionPolicy,
    ForecastEngine,
    Observation,
    ThresholdRisk,
    ValidationError,
)
from time_series_lab.composition import (
    collect_leaves,
    estimate_dependency,
    evaluate_outputs,
    impose_rank_dependence,
    materialize_leaves,
)


def _series_pair(n: int = 46) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(91)
    index = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    innovations = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.75], [0.75, 1.0]], size=n)
    values = np.zeros((n, 2))
    for step in range(1, n):
        values[step] = 0.55 * values[step - 1] + innovations[step]
    return pd.Series(values[:, 0], index=index), pd.Series(values[:, 1], index=index)


def _small_candidate_policy() -> CandidatePolicy:
    return CandidatePolicy(
        max_candidates=4,
        differencing="never",
        garch="never",
        markov_switching="never",
        simulation_paths=100,
    )


def test_expression_hash_is_deterministic_and_private_function_is_not_serialized() -> None:
    engine = ForecastEngine(repository=MemoryRepository())
    left = engine.variable("left", 2.0, mode="constant")
    right = engine.variable("right", 3.0, mode="constant")
    first = (left * right + 4) ** 2
    second = (left * right + 4) ** 2
    assert first.expression_id == second.expression_id

    secret = lambda values: values + 7  # noqa: E731 - função privada proposital no teste
    transformed = engine.transform("opaque-rule", secret, first, version="v1")
    payload = transformed.to_payload()
    assert payload == {"expression_id": transformed.expression_id, "name": "opaque-rule"}
    assert "function" not in repr(payload)
    assert "lambda" not in repr(payload)


def test_graph_evaluates_nonlinearity_and_rejects_nonfinite_results() -> None:
    engine = ForecastEngine(repository=MemoryRepository())
    x = engine.variable("x", 1.0, mode="constant")
    y = engine.variable("y", 1.0, mode="constant")
    expression = x * y
    leaves = collect_leaves({"xy": expression})
    x_paths = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_paths = np.array([[2.0], [4.0], [6.0], [8.0]])
    values = {
        leaves["x"].expression_id: x_paths,
        leaves["y"].expression_id: y_paths,
    }
    result = evaluate_outputs({"xy": expression}, values)["xy"]
    assert result.mean() == pytest.approx(np.mean(x_paths * y_paths))
    assert result.mean() != pytest.approx(x_paths.mean() * y_paths.mean())

    invalid = x / (y - y)
    invalid_leaves = collect_leaves({"invalid": invalid})
    invalid_values = {
        leaf.expression_id: (
            x_paths
            if leaf.name == "x"
            else y_paths
            if leaf.name == "y"
            else np.ones_like(x_paths)
        )
        for leaf in invalid_leaves.values()
    }
    with pytest.raises(ValidationError, match="forma/valores inválidos"):
        evaluate_outputs({"invalid": invalid}, invalid_values)


def test_residual_copula_recovers_dependence_and_has_auditable_fallback() -> None:
    rng = np.random.default_rng(7)
    errors = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.8], [0.8, 1.0]], size=200)
    frame = pd.DataFrame(errors, columns=["a", "b"])
    diagnostics = estimate_dependency(
        frame,
        ["a", "b"],
        mode="residual_copula",
        min_paired_errors=12,
        shrinkage=0.05,
    )
    assert not diagnostics.fallback
    assert diagnostics.correlation[0, 1] > 0.65

    paths = {
        "a": rng.normal(size=(2_000, 3)),
        "b": rng.normal(size=(2_000, 3)),
    }
    _, realized = impose_rank_dependence(paths, diagnostics, seed=11)
    assert realized.realized_rank_correlation[0, 1] > 0.60

    fallback = estimate_dependency(
        frame.iloc[:4],
        ["a", "b"],
        mode="residual_copula",
        min_paired_errors=12,
        shrinkage="auto",
    )
    assert fallback.fallback
    np.testing.assert_allclose(fallback.correlation, np.eye(2))
    assert "mínimo=12" in fallback.reason


def test_scenario_leaf_selects_named_future_without_changing_the_graph() -> None:
    engine = ForecastEngine(repository=MemoryRepository())
    targets = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    scenario = engine.variable(
        "scenario-input",
        {
            "observed": pd.Series([1.0, 1.0], index=targets),
            "up": pd.Series([1.2, 1.3], index=targets),
        },
        mode="scenario",
    )
    values = materialize_leaves(
        {"scenario-input": scenario},
        {},
        targets,
        n_paths=3,
        scenario="up",
    )
    np.testing.assert_allclose(
        values[scenario.expression_id], np.array([[1.2, 1.3]] * 3)
    )


def test_end_to_end_composition_is_causal_persisted_and_pathwise_coherent() -> None:
    first, second = _series_pair()
    repository = MemoryRepository()
    engine = ForecastEngine(repository=repository)
    a = engine.variable("a", first, mode="forecast")
    b = engine.variable("b", second, mode="forecast")

    known_index = pd.date_range(first.index[0], periods=len(first) + 3, freq="D", tz="UTC")
    known_series = pd.Series(np.linspace(1.0, 1.4, len(known_index)), index=known_index)
    vintage_rows = [
        {"as_of": target - pd.Timedelta(days=1), "target": target, "value": value}
        for target, value in known_series.items()
    ]
    scale = engine.variable(
        "scale",
        known_series,
        mode="known",
        vintages=pd.DataFrame(vintage_rows),
    )
    leaf_a = scale * a
    leaf_b = b + 1.0
    outputs = {"leaf/a": leaf_a, "leaf/b": leaf_b, "total": leaf_a + leaf_b}

    run = engine.backtest_expression(
        outputs,
        backtest=BacktestConfig(initial_train_size=28, horizon=1, step=8),
        candidate_policy=_small_candidate_policy(),
        policy=CompositionPolicy(
            preselect_top_k=1,
            max_combinations=1,
            min_paired_errors=3,
            n_backtest_paths=100,
            coverage_range=None,
        ),
        promote_as="composition-champion",
    )
    assert run.causal
    assert repository.get_composition_alias("composition-champion") == run.composition_run_id
    assert "operator" not in repr(run.to_payload())

    forecast = engine.compose(
        outputs,
        model_ref="composition-champion",
        horizon=3,
        n_paths=500,
        seed=19,
    )
    np.testing.assert_allclose(
        forecast.paths["total"], forecast.paths["leaf/a"] + forecast.paths["leaf/b"]
    )
    assert repository.get_composed_forecast(forecast.forecast_id) == forecast
    risk = engine.assess_risk(
        forecast.distributions["total"], ThresholdRisk(limit=3.0, n_paths=500, seed=2)
    )
    assert 0 <= risk.any_horizon_probability <= 1

    next_timestamp = first.index[-1] + pd.Timedelta(days=1)
    engine.append_history("a", [Observation(next_timestamp, 0.0, "a-next")])
    engine.append_history("b", [Observation(next_timestamp, 0.0, "b-next")])
    with pytest.raises(ValidationError, match="mudou de revisão"):
        engine.compose(outputs, model_ref=run, horizon=1, n_paths=20)


def test_known_series_without_vintages_requires_explicit_noncausal_mode() -> None:
    first, _ = _series_pair()
    engine = ForecastEngine(repository=MemoryRepository())
    forecast_variable = engine.variable("a", first, mode="forecast")
    known = engine.variable("known", pd.Series(1.0, index=first.index), mode="known")
    with pytest.raises(ValidationError, match="exige vintages"):
        engine.backtest_expression(
            forecast_variable + known,
            backtest=BacktestConfig(initial_train_size=28, horizon=1, step=8),
            candidate_policy=_small_candidate_policy(),
            policy=CompositionPolicy(
                preselect_top_k=1,
                max_combinations=1,
                n_backtest_paths=100,
                coverage_range=None,
            ),
        )

