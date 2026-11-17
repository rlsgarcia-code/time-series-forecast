import numpy as np

from time_series_lab.state_space import (
    forecast_local_level,
    kalman_filter_local_level,
    kalman_smoother_local_level,
    simulate_local_level,
    standardized_innovations,
)


def test_local_level_simulation_is_reproducible() -> None:
    first = simulate_local_level(
        100, state_variance=0.08, observation_variance=0.5, seed=901
    )
    second = simulate_local_level(
        100, state_variance=0.08, observation_variance=0.5, seed=901
    )
    np.testing.assert_array_equal(first.state, second.state)
    np.testing.assert_array_equal(first.observation, second.observation)


def test_filter_reduces_observation_noise_for_local_level() -> None:
    simulation = simulate_local_level(
        500, state_variance=0.03, observation_variance=1.0, seed=902
    )
    result = kalman_filter_local_level(
        simulation.observation,
        initial_mean=0.0,
        initial_variance=10.0,
        state_variance=0.03,
        observation_variance=1.0,
    )
    raw_rmse = np.sqrt(np.mean((simulation.observation - simulation.state) ** 2))
    filtered_rmse = np.sqrt(np.mean((result.filtered_mean - simulation.state) ** 2))
    assert filtered_rmse < raw_rmse
    assert np.all((result.gain >= 0) & (result.gain <= 1))


def test_missing_observation_skips_measurement_update() -> None:
    observations = np.array([0.2, 0.4, np.nan, 0.7])
    result = kalman_filter_local_level(
        observations,
        initial_mean=0.0,
        initial_variance=1.0,
        state_variance=0.1,
        observation_variance=0.5,
    )
    assert np.isnan(result.gain[2])
    assert result.filtered_mean[2] == result.predicted_mean[2]
    assert result.filtered_variance[2] == result.predicted_variance[2]


def test_smoothing_uses_future_information_without_changing_last_state() -> None:
    simulation = simulate_local_level(
        250, state_variance=0.05, observation_variance=0.8, seed=903
    )
    result = kalman_filter_local_level(
        simulation.observation,
        initial_mean=0.0,
        initial_variance=10.0,
        state_variance=0.05,
        observation_variance=0.8,
    )
    smoothed_mean, smoothed_variance = kalman_smoother_local_level(result)
    assert smoothed_mean[-1] == result.filtered_mean[-1]
    assert smoothed_variance[-1] == result.filtered_variance[-1]
    assert np.mean((smoothed_mean - simulation.state) ** 2) <= np.mean(
        (result.filtered_mean - simulation.state) ** 2
    )


def test_forecast_uncertainty_increases_with_horizon() -> None:
    result = kalman_filter_local_level(
        np.arange(20, dtype=float),
        initial_mean=0.0,
        initial_variance=1.0,
        state_variance=0.2,
        observation_variance=0.5,
    )
    forecast = forecast_local_level(
        result, horizon=8, state_variance=0.2, observation_variance=0.5
    )
    assert np.all(np.diff(forecast.state_variance) > 0)
    np.testing.assert_allclose(
        forecast.observation_variance - forecast.state_variance, 0.5
    )


def test_standardized_innovations_have_unit_scale_after_burnin() -> None:
    simulation = simulate_local_level(
        5_000, state_variance=0.06, observation_variance=0.7, seed=904
    )
    result = kalman_filter_local_level(
        simulation.observation,
        initial_mean=0.0,
        initial_variance=10.0,
        state_variance=0.06,
        observation_variance=0.7,
    )
    standardized = standardized_innovations(result)[200:]
    assert abs(np.mean(standardized)) < 0.06
    assert abs(np.std(standardized) - 1.0) < 0.06
