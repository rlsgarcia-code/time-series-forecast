"""Cadeias de Markov e modelos gaussianos de mudança de regime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True)
class RegimeSimulation:
    states: IntArray
    observations: FloatArray


@dataclass(frozen=True)
class HamiltonFilterResult:
    predicted_probabilities: FloatArray
    filtered_probabilities: FloatArray
    loglikelihood: float


def _validate_parameters(
    transition: ArrayLike, means: ArrayLike, standard_deviations: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    matrix = np.asarray(transition, dtype=float)
    location = np.asarray(means, dtype=float)
    scale = np.asarray(standard_deviations, dtype=float)
    k = location.size
    if matrix.shape != (k, k) or scale.shape != (k,) or k < 2:
        raise ValueError("dimensões incompatíveis")
    if np.any(matrix < 0) or not np.allclose(matrix.sum(axis=1), 1.0):
        raise ValueError("cada linha da transição deve ser uma distribuição")
    if np.any(scale <= 0):
        raise ValueError("desvios-padrão devem ser positivos")
    return matrix, location, scale


def stationary_distribution(transition: ArrayLike) -> FloatArray:
    """Resolve pi=pi P com soma unitária."""

    matrix = np.asarray(transition, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition deve ser quadrada")
    if np.any(matrix < 0) or not np.allclose(matrix.sum(axis=1), 1.0):
        raise ValueError("transition deve ser estocástica por linha")
    k = matrix.shape[0]
    system = np.vstack([matrix.T - np.eye(k), np.ones(k)])
    target = np.r_[np.zeros(k), 1.0]
    solution = np.linalg.lstsq(system, target, rcond=None)[0]
    return solution / solution.sum()


def simulate_markov_switching(
    n: int,
    *,
    transition: ArrayLike,
    means: ArrayLike,
    standard_deviations: ArrayLike,
    initial_probabilities: ArrayLike | None = None,
    seed: int | None = None,
) -> RegimeSimulation:
    """Simula S_t~P(S_t|S_t-1) e Y_t|S_t=k~N(mu_k,sigma_k²)."""

    matrix, location, scale = _validate_parameters(
        transition, means, standard_deviations
    )
    if n < 1:
        raise ValueError("n deve ser positivo")
    initial = (
        stationary_distribution(matrix)
        if initial_probabilities is None
        else np.asarray(initial_probabilities, dtype=float)
    )
    if initial.shape != location.shape or np.any(initial < 0) or not np.isclose(initial.sum(), 1):
        raise ValueError("initial_probabilities inválida")
    rng = np.random.default_rng(seed)
    states = np.empty(n, dtype=int)
    states[0] = rng.choice(location.size, p=initial)
    for t in range(1, n):
        states[t] = rng.choice(location.size, p=matrix[states[t - 1]])
    observations = rng.normal(location[states], scale[states])
    return RegimeSimulation(states=states, observations=observations)


def hamilton_filter_gaussian(
    observations: ArrayLike,
    *,
    transition: ArrayLike,
    means: ArrayLike,
    standard_deviations: ArrayLike,
    initial_probabilities: ArrayLike | None = None,
) -> HamiltonFilterResult:
    """Executa o algoritmo forward/Hamilton com normalização a cada instante."""

    matrix, location, scale = _validate_parameters(
        transition, means, standard_deviations
    )
    y = np.asarray(observations, dtype=float)
    if y.ndim != 1 or y.size < 1 or np.any(~np.isfinite(y)):
        raise ValueError("observations deve ser um vetor finito e não vazio")
    initial = (
        stationary_distribution(matrix)
        if initial_probabilities is None
        else np.asarray(initial_probabilities, dtype=float)
    )
    if initial.shape != location.shape or np.any(initial < 0) or not np.isclose(initial.sum(), 1):
        raise ValueError("initial_probabilities inválida")

    predicted = np.empty((y.size, location.size))
    filtered = np.empty_like(predicted)
    loglikelihood = 0.0
    previous = initial
    for t, value in enumerate(y):
        predicted[t] = previous if t == 0 else previous @ matrix
        emission = norm.pdf(value, loc=location, scale=scale)
        joint = predicted[t] * emission
        predictive_density = joint.sum()
        if predictive_density <= 0 or not np.isfinite(predictive_density):
            raise FloatingPointError("densidade preditiva numericamente inválida")
        filtered[t] = joint / predictive_density
        loglikelihood += np.log(predictive_density)
        previous = filtered[t]
    return HamiltonFilterResult(predicted, filtered, float(loglikelihood))


def kim_smoother(
    filtered: HamiltonFilterResult, transition: ArrayLike
) -> FloatArray:
    """Calcula P(S_t=k|y_1:T) pela recursão backward de Kim."""

    matrix = np.asarray(transition, dtype=float)
    smoothed = filtered.filtered_probabilities.copy()
    for t in range(smoothed.shape[0] - 2, -1, -1):
        ratio = np.divide(
            smoothed[t + 1],
            filtered.predicted_probabilities[t + 1],
            out=np.zeros_like(smoothed[t + 1]),
            where=filtered.predicted_probabilities[t + 1] > 0,
        )
        smoothed[t] = filtered.filtered_probabilities[t] * (matrix @ ratio)
        smoothed[t] /= smoothed[t].sum()
    return smoothed


def forecast_regime_probabilities(
    last_filtered_probabilities: ArrayLike,
    transition: ArrayLike,
    horizon: int,
) -> FloatArray:
    """Propaga probabilidades de regime sem novas observações."""

    probabilities = np.asarray(last_filtered_probabilities, dtype=float)
    matrix = np.asarray(transition, dtype=float)
    if horizon < 1 or probabilities.ndim != 1 or matrix.shape != (probabilities.size, probabilities.size):
        raise ValueError("configuração inválida")
    forecasts = np.empty((horizon, probabilities.size))
    current = probabilities
    for h in range(horizon):
        current = current @ matrix
        forecasts[h] = current
    return forecasts


def gaussian_mixture_moments(
    probabilities: ArrayLike, means: ArrayLike, standard_deviations: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Retorna média e variância de misturas gaussianas por linha."""

    weights = np.asarray(probabilities, dtype=float)
    location = np.asarray(means, dtype=float)
    scale = np.asarray(standard_deviations, dtype=float)
    if weights.ndim == 1:
        weights = weights[None, :]
    if weights.shape[1] != location.size or scale.shape != location.shape:
        raise ValueError("dimensões incompatíveis")
    mean = weights @ location
    second_moment = weights @ (scale**2 + location**2)
    return mean, second_moment - mean**2
