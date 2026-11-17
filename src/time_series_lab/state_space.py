"""Modelos lineares gaussianos de estado-espaço para uso didático."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LocalLevelSimulation:
    state: FloatArray
    observation: FloatArray


@dataclass(frozen=True)
class KalmanFilterResult:
    predicted_mean: FloatArray
    predicted_variance: FloatArray
    innovation: FloatArray
    innovation_variance: FloatArray
    gain: FloatArray
    filtered_mean: FloatArray
    filtered_variance: FloatArray
    loglikelihood: float


@dataclass(frozen=True)
class KalmanForecast:
    horizon: FloatArray
    mean: FloatArray
    state_variance: FloatArray
    observation_variance: FloatArray


def simulate_local_level(
    n: int,
    *,
    initial_state: float = 0.0,
    state_variance: float,
    observation_variance: float,
    seed: int | None = None,
) -> LocalLevelSimulation:
    """Simula alpha_t=alpha_(t-1)+eta_t e y_t=alpha_t+epsilon_t."""

    if n < 1 or state_variance < 0 or observation_variance < 0:
        raise ValueError("n deve ser positivo e as variâncias não negativas")
    rng = np.random.default_rng(seed)
    state = np.empty(n, dtype=float)
    state[0] = initial_state + rng.normal(0.0, np.sqrt(state_variance))
    for t in range(1, n):
        state[t] = state[t - 1] + rng.normal(0.0, np.sqrt(state_variance))
    observation = state + rng.normal(0.0, np.sqrt(observation_variance), n)
    return LocalLevelSimulation(state=state, observation=observation)


def kalman_filter_local_level(
    observations: ArrayLike,
    *,
    initial_mean: float,
    initial_variance: float,
    state_variance: float,
    observation_variance: float,
) -> KalmanFilterResult:
    """Filtra o modelo de nível local e aceita observações ausentes como ``nan``."""

    y = np.asarray(observations, dtype=float)
    if y.ndim != 1 or y.size < 1:
        raise ValueError("observations deve ser um vetor não vazio")
    if initial_variance < 0 or state_variance < 0 or observation_variance <= 0:
        raise ValueError("variâncias inválidas")

    n = y.size
    predicted_mean = np.empty(n)
    predicted_variance = np.empty(n)
    innovation = np.full(n, np.nan)
    innovation_variance = np.full(n, np.nan)
    gain = np.full(n, np.nan)
    filtered_mean = np.empty(n)
    filtered_variance = np.empty(n)

    previous_mean = float(initial_mean)
    previous_variance = float(initial_variance)
    loglikelihood = 0.0

    for t in range(n):
        predicted_mean[t] = previous_mean
        predicted_variance[t] = previous_variance + state_variance

        if np.isnan(y[t]):
            filtered_mean[t] = predicted_mean[t]
            filtered_variance[t] = predicted_variance[t]
        else:
            innovation[t] = y[t] - predicted_mean[t]
            innovation_variance[t] = predicted_variance[t] + observation_variance
            gain[t] = predicted_variance[t] / innovation_variance[t]
            filtered_mean[t] = predicted_mean[t] + gain[t] * innovation[t]
            filtered_variance[t] = (1.0 - gain[t]) * predicted_variance[t]
            loglikelihood += -0.5 * (
                np.log(2.0 * np.pi)
                + np.log(innovation_variance[t])
                + innovation[t] ** 2 / innovation_variance[t]
            )

        previous_mean = filtered_mean[t]
        previous_variance = filtered_variance[t]

    return KalmanFilterResult(
        predicted_mean=predicted_mean,
        predicted_variance=predicted_variance,
        innovation=innovation,
        innovation_variance=innovation_variance,
        gain=gain,
        filtered_mean=filtered_mean,
        filtered_variance=filtered_variance,
        loglikelihood=float(loglikelihood),
    )


def kalman_smoother_local_level(result: KalmanFilterResult) -> tuple[FloatArray, FloatArray]:
    """Aplica o smoother de Rauch--Tung--Striebel ao resultado filtrado."""

    smoothed_mean = result.filtered_mean.copy()
    smoothed_variance = result.filtered_variance.copy()
    for t in range(smoothed_mean.size - 2, -1, -1):
        next_prediction_variance = result.predicted_variance[t + 1]
        smoothing_gain = result.filtered_variance[t] / next_prediction_variance
        smoothed_mean[t] = result.filtered_mean[t] + smoothing_gain * (
            smoothed_mean[t + 1] - result.predicted_mean[t + 1]
        )
        smoothed_variance[t] = result.filtered_variance[t] + smoothing_gain**2 * (
            smoothed_variance[t + 1] - next_prediction_variance
        )
    return smoothed_mean, smoothed_variance


def forecast_local_level(
    result: KalmanFilterResult,
    *,
    horizon: int,
    state_variance: float,
    observation_variance: float,
) -> KalmanForecast:
    """Produz a distribuição preditiva h passos à frente do nível local."""

    if horizon < 1 or state_variance < 0 or observation_variance < 0:
        raise ValueError("horizon e variâncias inválidos")
    h = np.arange(1, horizon + 1, dtype=float)
    state_var = result.filtered_variance[-1] + h * state_variance
    return KalmanForecast(
        horizon=h,
        mean=np.repeat(result.filtered_mean[-1], horizon),
        state_variance=state_var,
        observation_variance=state_var + observation_variance,
    )


def standardized_innovations(result: KalmanFilterResult) -> FloatArray:
    """Retorna v_t/sqrt(F_t), preservando ``nan`` em períodos sem observação."""

    return result.innovation / np.sqrt(result.innovation_variance)
