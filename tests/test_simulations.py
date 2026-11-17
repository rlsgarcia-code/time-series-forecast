import numpy as np
import pytest

from time_series_lab import (
    ar1_stationary_moments,
    forecast_ar1,
    sample_acf,
    shock_response,
    simulate_ar1,
    simulate_ar1_future_paths,
    simulate_random_walk,
    simulate_white_noise,
    theoretical_ar1_acf,
)


def test_white_noise_is_reproducible_and_has_expected_shape() -> None:
    first = simulate_white_noise(30, n_paths=4, seed=7)
    second = simulate_white_noise(30, n_paths=4, seed=7)
    assert first.shape == (4, 30)
    np.testing.assert_array_equal(first, second)


def test_white_noise_empirical_moments_are_close() -> None:
    values = simulate_white_noise(200_000, mu=2.0, sigma=3.0, seed=42).ravel()
    assert values.mean() == pytest.approx(2.0, abs=0.03)
    assert values.var() == pytest.approx(9.0, rel=0.015)


def test_random_walk_includes_x0_and_variance_grows_with_time() -> None:
    paths = simulate_random_walk(200, n_paths=20_000, x0=5.0, sigma=1.2, seed=1)
    assert paths.shape == (20_000, 201)
    np.testing.assert_array_equal(paths[:, 0], np.full(20_000, 5.0))
    assert paths[:, 200].var() == pytest.approx(200 * 1.2**2, rel=0.035)


def test_stationary_ar1_matches_theoretical_moments() -> None:
    phi, intercept, sigma = 0.7, 0.6, 0.8
    expected_mean, expected_variance = ar1_stationary_moments(
        phi=phi, intercept=intercept, sigma=sigma
    )
    paths = simulate_ar1(
        1_500,
        phi=phi,
        n_paths=4_000,
        intercept=intercept,
        sigma=sigma,
        seed=2,
    )
    terminal = paths[:, -1]
    assert terminal.mean() == pytest.approx(expected_mean, abs=0.04)
    assert terminal.var() == pytest.approx(expected_variance, rel=0.06)


def test_theoretical_acf_and_sample_acf() -> None:
    expected = np.array([1.0, 0.6, 0.36, 0.216])
    np.testing.assert_allclose(theoretical_ar1_acf(0.6, 3), expected)
    series = simulate_ar1(30_000, phi=0.6, sigma=1.0, seed=3)[0, 1:]
    np.testing.assert_allclose(sample_acf(series, 3), expected, atol=0.025)


def test_shock_decay_changes_sign_when_phi_is_negative() -> None:
    np.testing.assert_allclose(shock_response(0.5, 4.0, 3), [4.0, 2.0, 1.0, 0.5])
    np.testing.assert_allclose(shock_response(-0.5, 4.0, 3), [4.0, -2.0, 1.0, -0.5])


def test_ar1_forecast_matches_closed_form() -> None:
    result = forecast_ar1(2.0, 3, phi=0.5, intercept=1.0, sigma=2.0)
    np.testing.assert_allclose(result.mean, [2.0, 2.0, 2.0])
    np.testing.assert_allclose(result.std**2, [4.0, 5.0, 5.25])
    assert np.all(result.lower < result.mean)
    assert np.all(result.mean < result.upper)


def test_monte_carlo_forecast_agrees_with_analytic_distribution() -> None:
    kwargs = dict(phi=0.75, intercept=0.2, sigma=0.9)
    analytic = forecast_ar1(1.5, 12, **kwargs)
    paths = simulate_ar1_future_paths(1.5, 12, n_paths=80_000, seed=4, **kwargs)
    np.testing.assert_allclose(paths.mean(axis=0), analytic.mean, atol=0.015)
    np.testing.assert_allclose(paths.std(axis=0), analytic.std, atol=0.02)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: simulate_white_noise(0), "n_steps"),
        (lambda: simulate_ar1(5, phi=np.inf), "phi"),
        (lambda: ar1_stationary_moments(phi=1.0), "\\|phi\\|"),
        (lambda: theoretical_ar1_acf(-1.0, 5), "\\|phi\\|"),
        (lambda: forecast_ar1(0.0, 0, phi=0.5), "horizon"),
        (lambda: sample_acf([1.0, 1.0, 1.0], 1), "constante"),
    ],
)
def test_invalid_inputs_fail_with_clear_messages(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()

