"""Detecção offline de mudanças de média por custos segmentados."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PiecewiseSimulation:
    observations: FloatArray
    means: FloatArray
    breakpoints: tuple[int, ...]


@dataclass(frozen=True)
class SingleChangeResult:
    breakpoint: int
    no_change_cost: float
    split_cost: float
    improvement: float
    cost_profile: FloatArray


@dataclass(frozen=True)
class SegmentationResult:
    breakpoints: tuple[int, ...]
    objective: float
    fitted_values: FloatArray


def simulate_piecewise_constant(
    lengths: list[int] | tuple[int, ...],
    means: list[float] | tuple[float, ...],
    *,
    sigma: float = 1.0,
    seed: int | None = None,
) -> PiecewiseSimulation:
    """Simula segmentos gaussianos com média constante e variância comum."""

    segment_lengths = np.asarray(lengths, dtype=int)
    segment_means = np.asarray(means, dtype=float)
    if segment_lengths.ndim != 1 or segment_lengths.size != segment_means.size:
        raise ValueError("lengths e means devem ter o mesmo tamanho")
    if np.any(segment_lengths < 1) or sigma <= 0:
        raise ValueError("comprimentos e sigma devem ser positivos")
    latent_mean = np.repeat(segment_means, segment_lengths)
    rng = np.random.default_rng(seed)
    observations = latent_mean + rng.normal(0.0, sigma, latent_mean.size)
    breakpoints = tuple(np.cumsum(segment_lengths)[:-1].tolist())
    return PiecewiseSimulation(observations, latent_mean, breakpoints)


def _prefix_sums(values: FloatArray) -> tuple[FloatArray, FloatArray]:
    return np.r_[0.0, np.cumsum(values)], np.r_[0.0, np.cumsum(values**2)]


def segment_sse(
    cumulative_sum: ArrayLike,
    cumulative_sum_of_squares: ArrayLike,
    start: int,
    end: int,
) -> float:
    """Calcula SSE de ``values[start:end]`` em O(1) usando somas prefixadas."""

    total = np.asarray(cumulative_sum, dtype=float)
    squares = np.asarray(cumulative_sum_of_squares, dtype=float)
    if not 0 <= start < end < total.size or squares.shape != total.shape:
        raise ValueError("segmento ou somas prefixadas inválidos")
    length = end - start
    segment_sum = total[end] - total[start]
    segment_square_sum = squares[end] - squares[start]
    return float(max(segment_square_sum - segment_sum**2 / length, 0.0))


def best_single_mean_change(values: ArrayLike, min_size: int = 10) -> SingleChangeResult:
    """Busca exata da divisão que mais reduz a soma de quadrados."""

    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or y.size < 2 * min_size or min_size < 1 or np.any(~np.isfinite(y)):
        raise ValueError("série finita e espaço para dois segmentos são necessários")
    total, squares = _prefix_sums(y)
    profile = np.full(y.size + 1, np.nan)
    for split in range(min_size, y.size - min_size + 1):
        profile[split] = segment_sse(total, squares, 0, split) + segment_sse(
            total, squares, split, y.size
        )
    breakpoint = int(np.nanargmin(profile))
    no_change = segment_sse(total, squares, 0, y.size)
    split_cost = float(profile[breakpoint])
    return SingleChangeResult(
        breakpoint=breakpoint,
        no_change_cost=no_change,
        split_cost=split_cost,
        improvement=no_change - split_cost,
        cost_profile=profile,
    )


def cusum_mean_statistic(values: ArrayLike) -> FloatArray:
    """Retorna C_t=sum_{i<=t}(y_i-y_bar)/(s sqrt(n))."""

    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or y.size < 2 or np.any(~np.isfinite(y)):
        raise ValueError("série finita com ao menos duas observações é necessária")
    scale = np.std(y, ddof=1)
    if scale == 0:
        return np.zeros(y.size)
    return np.cumsum(y - np.mean(y)) / (scale * np.sqrt(y.size))


def optimal_mean_segmentation(
    values: ArrayLike,
    *,
    penalty: float,
    min_size: int = 10,
) -> SegmentationResult:
    """Minimiza soma de SSE + penalty por mudança via programação dinâmica exata."""

    y = np.asarray(values, dtype=float)
    n = y.size
    if y.ndim != 1 or n < min_size or min_size < 1 or penalty < 0 or np.any(~np.isfinite(y)):
        raise ValueError("dados ou configuração inválidos")
    total, squares = _prefix_sums(y)
    objective = np.full(n + 1, np.inf)
    previous = np.full(n + 1, -1, dtype=int)
    objective[0] = -penalty

    for end in range(min_size, n + 1):
        starts = [0] + list(range(min_size, end - min_size + 1))
        valid_starts = [start for start in starts if np.isfinite(objective[start])]
        candidates = np.array(
            [
                objective[start]
                + segment_sse(total, squares, start, end)
                + penalty
                for start in valid_starts
            ]
        )
        winner = int(np.argmin(candidates))
        objective[end] = candidates[winner]
        previous[end] = valid_starts[winner]

    endpoints: list[int] = []
    end = n
    while previous[end] > 0:
        endpoints.append(previous[end])
        end = previous[end]
    breakpoints = tuple(sorted(endpoints))

    fitted = np.empty(n)
    boundaries = (0, *breakpoints, n)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        fitted[start:end] = np.mean(y[start:end])
    return SegmentationResult(breakpoints, float(objective[n]), fitted)


def match_breakpoints(
    true_breakpoints: list[int] | tuple[int, ...],
    estimated_breakpoints: list[int] | tuple[int, ...],
    *,
    tolerance: int,
) -> dict[str, float]:
    """Faz matching guloso um-para-um e calcula precision, recall e F1."""

    if tolerance < 0:
        raise ValueError("tolerance deve ser não negativa")
    true_values = list(true_breakpoints)
    estimated_values = list(estimated_breakpoints)
    available = set(range(len(true_values)))
    matches = 0
    for estimate in estimated_values:
        candidates = [
            index
            for index in available
            if abs(true_values[index] - estimate) <= tolerance
        ]
        if candidates:
            winner = min(candidates, key=lambda index: abs(true_values[index] - estimate))
            available.remove(winner)
            matches += 1
    precision = matches / len(estimated_values) if estimated_values else float(not true_values)
    recall = matches / len(true_values) if true_values else float(not estimated_values)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "matches": float(matches)}
