import numpy as np

from time_series_lab.sequential_detection import (
    detection_summary,
    estimate_cusum_run_lengths,
    estimate_ewma_run_lengths,
    ewma_chart,
    simulate_standardized_shift,
    two_sided_cusum,
)


def test_shift_simulation_is_reproducible() -> None:
    first = simulate_standardized_shift(100, 100, shift=1.0, seed=1201)
    second = simulate_standardized_shift(100, 100, shift=1.0, seed=1201)
    np.testing.assert_array_equal(first, second)


def test_cusum_recursions_match_hand_calculation() -> None:
    result = two_sided_cusum([0.2, 1.2, 1.0, -2.0], reference=0.5, threshold=5.0)
    np.testing.assert_allclose(result.positive, [0.0, 0.7, 1.2, 0.0])
    np.testing.assert_allclose(result.negative, [0.0, 0.0, 0.0, -1.5])
    assert result.first_alarm is None


def test_cusum_detects_sustained_positive_shift() -> None:
    values = np.r_[np.zeros(40), np.ones(40)]
    result = two_sided_cusum(values, reference=0.25, threshold=4.0)
    assert result.first_alarm is not None
    assert 40 <= result.first_alarm < 50


def test_ewma_statistic_and_transient_variance_formula() -> None:
    result = ewma_chart([1.0, 1.0, 1.0], smoothing=0.2, limit_multiplier=3.0)
    np.testing.assert_allclose(result.statistic, [0.2, 0.36, 0.488])
    expected_variance = 0.2 / 1.8 * (1 - 0.8 ** (2 * np.arange(1, 4)))
    np.testing.assert_allclose(result.upper_limit, 3 * np.sqrt(expected_variance))
    np.testing.assert_allclose(result.lower_limit, -result.upper_limit)


def test_higher_threshold_reduces_cusum_false_alarms() -> None:
    low = estimate_cusum_run_lengths(
        400, 200, reference=0.5, threshold=3.0, seed=1202
    )
    high = estimate_cusum_run_lengths(
        400, 200, reference=0.5, threshold=6.0, seed=1202
    )
    assert np.mean(high <= 200) < np.mean(low <= 200)


def test_shift_reduces_run_length_for_both_detectors() -> None:
    cusum_null = estimate_cusum_run_lengths(
        300, 250, reference=0.5, threshold=5.0, seed=1203
    )
    cusum_shift = estimate_cusum_run_lengths(
        300, 250, reference=0.5, threshold=5.0, shift=1.0, seed=1203
    )
    ewma_null = estimate_ewma_run_lengths(
        300, 250, smoothing=0.15, limit_multiplier=3.0, seed=1204
    )
    ewma_shift = estimate_ewma_run_lengths(
        300, 250, smoothing=0.15, limit_multiplier=3.0, shift=1.0, seed=1204
    )
    assert np.mean(cusum_shift) < np.mean(cusum_null)
    assert np.mean(ewma_shift) < np.mean(ewma_null)


def test_detection_summary_separates_false_alarm_delay_and_censoring() -> None:
    summary = detection_summary(
        np.array([50, 101, 110, 201]), change_at=100, max_length=200
    )
    assert summary["false_alarm_rate"] == 0.25
    assert summary["detection_probability"] == 0.5
    assert summary["mean_delay_given_detection"] == 5.5
    assert summary["censoring_rate"] == 0.25
