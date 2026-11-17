from pathlib import Path

import numpy as np
import pytest

from time_series_lab.backtesting import (
    evaluate_gaussian,
    gaussian_crps,
    rolling_origins,
    select_model,
)
from time_series_lab.classical import difference, integrate_differences, simulate_arma
from time_series_lab.continuous import brownian_paths, gbm_exact_paths, quadratic_variation
from time_series_lab.pipeline import ForecastArtifact, SeriesContract, standardized_surprise
from time_series_lab.volatility import fit_garch11, garch_variance, simulate_garch11


def test_arma_simulation_is_reproducible_and_stationary_in_scale() -> None:
    first = simulate_arma(20_000, ar=[0.6], ma=[0.3], seed=30)
    second = simulate_arma(20_000, ar=[0.6], ma=[0.3], seed=30)
    np.testing.assert_array_equal(first, second)
    assert abs(first.mean()) < 0.05
    assert 0.5 < np.var(first) < 5.0


def test_difference_and_integration_are_inverse_for_one_step() -> None:
    levels = np.array([10.0, 11.5, 11.0, 14.0])
    diffs = difference(levels)
    np.testing.assert_allclose(integrate_differences(levels[0], diffs), levels[1:])


def test_brownian_moments_and_quadratic_variation() -> None:
    _, paths = brownian_paths(1_000, n_paths=5_000, horizon=2.0, seed=31)
    assert paths[:, -1].mean() == pytest.approx(0.0, abs=0.04)
    assert paths[:, -1].var() == pytest.approx(2.0, rel=0.07)
    assert quadratic_variation(paths[0]) == pytest.approx(2.0, rel=0.15)


def test_gbm_is_positive_and_matches_expected_mean() -> None:
    _, paths = gbm_exact_paths(100.0, 0.08, 0.2, 252, n_paths=30_000, seed=32)
    assert np.all(paths > 0)
    assert paths[:, -1].mean() == pytest.approx(100 * np.exp(0.08), rel=0.015)


def test_garch_variance_and_fit_are_valid() -> None:
    returns, true_variance = simulate_garch11(
        4_000, omega=0.05, alpha=0.08, beta=0.88, seed=33
    )
    filtered = garch_variance(returns, 0.05, 0.08, 0.88)
    np.testing.assert_allclose(filtered[1:], true_variance[1:])
    fit = fit_garch11(returns)
    assert fit.success
    assert fit.omega > 0 and fit.alpha >= 0 and fit.beta >= 0
    assert fit.alpha + fit.beta < 1


def test_probabilistic_metrics_and_selection() -> None:
    y = np.array([0.0, 1.0, -1.0])
    mean = np.zeros(3)
    std = np.ones(3)
    assert np.all(gaussian_crps(y, mean, std) >= 0)
    metrics = evaluate_gaussian(y, mean, std)
    assert set(metrics) == {"mae", "rmse", "coverage", "mean_width", "crps"}
    assert rolling_origins(20, 10, 3, step=2) == [10, 12, 14, 16]
    winner = select_model({"a": {"crps": 1.0}, "b": {"crps": 0.8}})
    assert winner == "b"


def test_contract_artifact_and_monitoring_roundtrip(tmp_path: Path) -> None:
    contract = SeriesContract("D", "UTC", horizon=2, minimum_history=4)
    np.testing.assert_array_equal(contract.validate([1, 2, 3, 4]), [1, 2, 3, 4])
    artifact = ForecastArtifact(
        model_name="ar1",
        cutoff=10,
        horizon=[1, 2],
        mean=[1.0, 1.1],
        lower=[0.0, 0.1],
        upper=[2.0, 2.1],
        metadata={"version": 1},
    )
    path = artifact.save(tmp_path / "forecast.json")
    assert ForecastArtifact.load(path) == artifact
    assert standardized_surprise(3.0, 1.0, 2.0) == pytest.approx(1.0)

