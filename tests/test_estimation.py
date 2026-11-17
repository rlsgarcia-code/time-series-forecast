import numpy as np
import pytest

from time_series_lab.estimation import (
    ar1_parameter_intervals,
    fit_ar1_ols,
    forecast_metrics,
    rolling_one_step_ar1,
)
from time_series_lab.simulations import simulate_ar1


def test_fit_matches_direct_least_squares() -> None:
    values = np.array([1.0, 1.8, 1.2, 2.5, 2.1, 3.0])
    response = values[1:]
    design = np.column_stack([np.ones(values.size - 1), values[:-1]])
    expected = np.linalg.lstsq(design, response, rcond=None)[0]
    fit = fit_ar1_ols(values)
    np.testing.assert_allclose(fit.parameters, expected)


def test_fit_recovers_parameters_in_large_sample() -> None:
    values = simulate_ar1(
        30_000, phi=0.72, intercept=0.4, sigma=0.8, x0=0.4 / (1 - 0.72), seed=20
    )[0]
    fit = fit_ar1_ols(values)
    assert fit.intercept == pytest.approx(0.4, abs=0.025)
    assert fit.phi == pytest.approx(0.72, abs=0.012)
    assert fit.sigma == pytest.approx(0.8, abs=0.012)


def test_ols_residuals_are_orthogonal_to_regressors() -> None:
    values = simulate_ar1(500, phi=0.6, intercept=0.2, sigma=1.0, seed=21)[0]
    fit = fit_ar1_ols(values)
    assert fit.residuals.sum() == pytest.approx(0.0, abs=1e-10)
    assert np.dot(fit.residuals, values[:-1]) == pytest.approx(0.0, abs=1e-10)


def test_parameter_intervals_have_expected_shape_and_order() -> None:
    values = simulate_ar1(1_000, phi=0.5, intercept=0.3, sigma=1.0, seed=22)[0]
    fit = fit_ar1_ols(values)
    lower, upper = ar1_parameter_intervals(fit)
    assert lower.shape == upper.shape == (2,)
    assert np.all(lower < fit.parameters)
    assert np.all(fit.parameters < upper)


def test_rolling_forecast_has_no_future_leakage() -> None:
    values = simulate_ar1(40, phi=0.7, intercept=0.1, sigma=0.5, seed=23)[0]
    predictions = rolling_one_step_ar1(values, initial_train_size=25)
    first_fit = fit_ar1_ols(values[:25])
    expected_first = first_fit.intercept + first_fit.phi * values[24]
    assert predictions.shape == (values.size - 25,)
    assert predictions[0] == pytest.approx(expected_first)


def test_forecast_metrics() -> None:
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 1.0, 5.0])
    metrics = forecast_metrics(
        actual,
        predicted,
        lower=np.array([0.0, 0.0, 0.0]),
        upper=np.array([2.0, 2.0, 2.0]),
    )
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["mean_interval_width"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: fit_ar1_ols([1.0, 1.0, 1.0, 1.0]),
        lambda: fit_ar1_ols([1.0, 2.0, np.nan, 3.0]),
        lambda: rolling_one_step_ar1(np.arange(10.0), 3),
        lambda: forecast_metrics([1, 2], [1]),
    ],
)
def test_invalid_inputs_fail(call) -> None:
    with pytest.raises(ValueError):
        call()

