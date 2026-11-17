import numpy as np

from time_series_lab.change_points import (
    best_single_mean_change,
    cusum_mean_statistic,
    match_breakpoints,
    optimal_mean_segmentation,
    segment_sse,
    simulate_piecewise_constant,
)


def test_piecewise_simulation_is_reproducible() -> None:
    first = simulate_piecewise_constant([80, 100, 70], [0.0, 2.0, -1.0], seed=1101)
    second = simulate_piecewise_constant([80, 100, 70], [0.0, 2.0, -1.0], seed=1101)
    np.testing.assert_array_equal(first.observations, second.observations)
    assert first.breakpoints == (80, 180)


def test_segment_sse_matches_direct_computation() -> None:
    values = np.array([1.0, 4.0, 2.0, 8.0, 3.0])
    cumulative = np.r_[0.0, np.cumsum(values)]
    squares = np.r_[0.0, np.cumsum(values**2)]
    expected = np.sum((values[1:4] - np.mean(values[1:4])) ** 2)
    np.testing.assert_allclose(segment_sse(cumulative, squares, 1, 4), expected)


def test_single_change_search_recovers_large_mean_shift() -> None:
    simulation = simulate_piecewise_constant([180, 220], [0.0, 2.5], sigma=0.8, seed=1102)
    result = best_single_mean_change(simulation.observations, min_size=30)
    assert abs(result.breakpoint - 180) <= 3
    assert result.improvement > 0


def test_penalized_segmentation_recovers_multiple_changes() -> None:
    simulation = simulate_piecewise_constant(
        [120, 140, 130, 110], [0.0, 2.2, -1.3, 1.4], sigma=0.7, seed=1103
    )
    result = optimal_mean_segmentation(
        simulation.observations, penalty=12.0, min_size=30
    )
    metrics = match_breakpoints(simulation.breakpoints, result.breakpoints, tolerance=5)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert np.mean((result.fitted_values - simulation.means) ** 2) < 0.05


def test_larger_penalty_cannot_create_more_breakpoints() -> None:
    simulation = simulate_piecewise_constant([100, 100, 100], [0.0, 1.5, -1.0], seed=1104)
    low = optimal_mean_segmentation(simulation.observations, penalty=5.0, min_size=20)
    high = optimal_mean_segmentation(simulation.observations, penalty=40.0, min_size=20)
    assert len(high.breakpoints) <= len(low.breakpoints)


def test_cusum_peaks_near_a_single_change() -> None:
    simulation = simulate_piecewise_constant([150, 150], [0.0, 2.0], sigma=0.6, seed=1105)
    statistic = cusum_mean_statistic(simulation.observations)
    assert abs(int(np.argmax(np.abs(statistic))) + 1 - 150) <= 5
    assert abs(statistic[-1]) < 1e-12


def test_breakpoint_matching_is_one_to_one() -> None:
    metrics = match_breakpoints([100, 200], [98, 103, 205], tolerance=5)
    assert metrics["matches"] == 2
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0
