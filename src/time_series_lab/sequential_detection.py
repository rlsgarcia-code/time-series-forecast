"""Detectores sequenciais CUSUM e EWMA para resíduos padronizados."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class CusumResult:
    positive: FloatArray
    negative: FloatArray
    alarm: BoolArray
    first_alarm: int | None


@dataclass(frozen=True)
class EwmaResult:
    statistic: FloatArray
    lower_limit: FloatArray
    upper_limit: FloatArray
    alarm: BoolArray
    first_alarm: int | None


def simulate_standardized_shift(
    n_pre: int,
    n_post: int,
    *,
    shift: float,
    seed: int | None = None,
) -> FloatArray:
    """Simula N(0,1) antes da mudança e N(shift,1) depois."""

    if n_pre < 0 or n_post < 0 or n_pre + n_post < 1:
        raise ValueError("comprimentos inválidos")
    rng = np.random.default_rng(seed)
    return np.r_[rng.normal(size=n_pre), rng.normal(loc=shift, size=n_post)]


def two_sided_cusum(values: ArrayLike, *, reference: float, threshold: float) -> CusumResult:
    """CUSUM bilateral: C+ e C- para mudanças positivas e negativas."""

    z = np.asarray(values, dtype=float)
    if z.ndim != 1 or z.size < 1 or np.any(~np.isfinite(z)):
        raise ValueError("values deve ser um vetor finito e não vazio")
    if reference < 0 or threshold <= 0:
        raise ValueError("reference não negativa e threshold positivo são necessários")
    positive = np.empty(z.size)
    negative = np.empty(z.size)
    previous_positive = 0.0
    previous_negative = 0.0
    for t, value in enumerate(z):
        previous_positive = max(0.0, previous_positive + value - reference)
        previous_negative = min(0.0, previous_negative + value + reference)
        positive[t] = previous_positive
        negative[t] = previous_negative
    alarm = (positive >= threshold) | (negative <= -threshold)
    indices = np.flatnonzero(alarm)
    first_alarm = int(indices[0]) if indices.size else None
    return CusumResult(positive, negative, alarm, first_alarm)


def ewma_chart(
    values: ArrayLike,
    *,
    smoothing: float,
    limit_multiplier: float,
    initial: float = 0.0,
) -> EwmaResult:
    """EWMA bilateral com limites transitórios sob N(0,1)."""

    z = np.asarray(values, dtype=float)
    if z.ndim != 1 or z.size < 1 or np.any(~np.isfinite(z)):
        raise ValueError("values deve ser um vetor finito e não vazio")
    if not 0 < smoothing <= 1 or limit_multiplier <= 0:
        raise ValueError("smoothing em (0,1] e multiplicador positivo são necessários")
    statistic = np.empty(z.size)
    previous = float(initial)
    for t, value in enumerate(z):
        previous = smoothing * value + (1.0 - smoothing) * previous
        statistic[t] = previous
    steps = np.arange(1, z.size + 1)
    variance = smoothing / (2.0 - smoothing) * (
        1.0 - (1.0 - smoothing) ** (2 * steps)
    )
    upper = limit_multiplier * np.sqrt(variance)
    lower = -upper
    alarm = (statistic >= upper) | (statistic <= lower)
    indices = np.flatnonzero(alarm)
    first_alarm = int(indices[0]) if indices.size else None
    return EwmaResult(statistic, lower, upper, alarm, first_alarm)


def first_run_length(first_alarm: int | None, max_length: int) -> int:
    """Converte índice zero-based em run length; usa max_length+1 para censura."""

    if max_length < 1:
        raise ValueError("max_length deve ser positivo")
    return max_length + 1 if first_alarm is None else first_alarm + 1


def estimate_cusum_run_lengths(
    repetitions: int,
    max_length: int,
    *,
    reference: float,
    threshold: float,
    shift: float = 0.0,
    change_at: int = 0,
    seed: int | None = None,
) -> NDArray[np.int_]:
    """Monte Carlo de run lengths com possível mudança em ``change_at``."""

    if repetitions < 1 or not 0 <= change_at <= max_length:
        raise ValueError("configuração inválida")
    rng = np.random.default_rng(seed)
    run_lengths = np.empty(repetitions, dtype=int)
    for repetition in range(repetitions):
        values = np.r_[
            rng.normal(size=change_at),
            rng.normal(loc=shift, size=max_length - change_at),
        ]
        result = two_sided_cusum(values, reference=reference, threshold=threshold)
        run_lengths[repetition] = first_run_length(result.first_alarm, max_length)
    return run_lengths


def estimate_ewma_run_lengths(
    repetitions: int,
    max_length: int,
    *,
    smoothing: float,
    limit_multiplier: float,
    shift: float = 0.0,
    change_at: int = 0,
    seed: int | None = None,
) -> NDArray[np.int_]:
    """Monte Carlo de run lengths para o gráfico EWMA."""

    if repetitions < 1 or not 0 <= change_at <= max_length:
        raise ValueError("configuração inválida")
    rng = np.random.default_rng(seed)
    run_lengths = np.empty(repetitions, dtype=int)
    for repetition in range(repetitions):
        values = np.r_[
            rng.normal(size=change_at),
            rng.normal(loc=shift, size=max_length - change_at),
        ]
        result = ewma_chart(
            values, smoothing=smoothing, limit_multiplier=limit_multiplier
        )
        run_lengths[repetition] = first_run_length(result.first_alarm, max_length)
    return run_lengths


def detection_summary(
    run_lengths: ArrayLike, *, change_at: int, max_length: int
) -> dict[str, float]:
    """Resume falso alarme, detecção após mudança e atraso condicional."""

    runs = np.asarray(run_lengths, dtype=int)
    if runs.ndim != 1 or runs.size < 1 or not 0 <= change_at <= max_length:
        raise ValueError("entrada inválida")
    false_alarm = runs <= change_at
    detected_after = (runs > change_at) & (runs <= max_length)
    delays = runs[detected_after] - change_at
    return {
        "false_alarm_rate": float(np.mean(false_alarm)),
        "detection_probability": float(np.mean(detected_after)),
        "mean_delay_given_detection": float(np.mean(delays)) if delays.size else np.nan,
        "censoring_rate": float(np.mean(runs > max_length)),
    }
