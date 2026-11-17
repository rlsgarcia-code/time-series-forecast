import numpy as np

from time_series_lab.regime_switching import (
    forecast_regime_probabilities,
    gaussian_mixture_moments,
    hamilton_filter_gaussian,
    kim_smoother,
    simulate_markov_switching,
    stationary_distribution,
)


P = np.array([[0.96, 0.04], [0.10, 0.90]])
MEANS = np.array([-1.0, 2.5])
STDS = np.array([0.7, 1.1])


def test_stationary_distribution_is_invariant() -> None:
    stationary = stationary_distribution(P)
    np.testing.assert_allclose(stationary @ P, stationary, atol=1e-12)
    np.testing.assert_allclose(stationary.sum(), 1.0)


def test_simulation_is_reproducible_and_has_valid_states() -> None:
    first = simulate_markov_switching(
        300, transition=P, means=MEANS, standard_deviations=STDS, seed=1001
    )
    second = simulate_markov_switching(
        300, transition=P, means=MEANS, standard_deviations=STDS, seed=1001
    )
    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_array_equal(first.observations, second.observations)
    assert set(np.unique(first.states)) <= {0, 1}


def test_hamilton_filter_probabilities_are_normalized_and_informative() -> None:
    simulation = simulate_markov_switching(
        2_000, transition=P, means=MEANS, standard_deviations=STDS, seed=1002
    )
    result = hamilton_filter_gaussian(
        simulation.observations,
        transition=P,
        means=MEANS,
        standard_deviations=STDS,
    )
    np.testing.assert_allclose(result.filtered_probabilities.sum(axis=1), 1.0)
    decoded = result.filtered_probabilities.argmax(axis=1)
    assert np.mean(decoded == simulation.states) > 0.90
    assert np.isfinite(result.loglikelihood)


def test_kim_smoother_is_normalized_and_does_not_change_last_probability() -> None:
    simulation = simulate_markov_switching(
        500, transition=P, means=MEANS, standard_deviations=STDS, seed=1003
    )
    filtered = hamilton_filter_gaussian(
        simulation.observations,
        transition=P,
        means=MEANS,
        standard_deviations=STDS,
    )
    smoothed = kim_smoother(filtered, P)
    np.testing.assert_allclose(smoothed.sum(axis=1), 1.0)
    np.testing.assert_allclose(smoothed[-1], filtered.filtered_probabilities[-1])
    assert np.mean(smoothed.argmax(axis=1) == simulation.states) >= np.mean(
        filtered.filtered_probabilities.argmax(axis=1) == simulation.states
    )


def test_forecast_probabilities_converge_toward_stationary_distribution() -> None:
    forecasts = forecast_regime_probabilities(np.array([1.0, 0.0]), P, 100)
    np.testing.assert_allclose(forecasts.sum(axis=1), 1.0)
    np.testing.assert_allclose(forecasts[-1], stationary_distribution(P), atol=1e-6)


def test_gaussian_mixture_moments_include_between_regime_uncertainty() -> None:
    probabilities = np.array([[0.5, 0.5]])
    mean, variance = gaussian_mixture_moments(probabilities, MEANS, STDS)
    expected_mean = np.mean(MEANS)
    expected_variance = np.mean(STDS**2 + MEANS**2) - expected_mean**2
    np.testing.assert_allclose(mean, [expected_mean])
    np.testing.assert_allclose(variance, [expected_variance])
