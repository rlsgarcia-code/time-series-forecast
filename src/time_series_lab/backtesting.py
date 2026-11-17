"""Utilidades de backtesting probabilístico sem vazamento temporal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GaussianForecast:
    mean: FloatArray
    std: FloatArray


def rolling_origins(n: int, initial: int, horizon: int, step: int = 1) -> list[int]:
    """Retorna cortes ``origin`` tais que ``origin+horizon <= n``."""

    if not 1 <= initial < n or horizon < 1 or step < 1:
        raise ValueError("configuração de backtest inválida")
    return list(range(initial, n - horizon + 1, step))


def gaussian_crps(actual: ArrayLike, mean: ArrayLike, std: ArrayLike) -> FloatArray:
    """CRPS fechado para uma distribuição normal."""

    y, mu, sigma = map(lambda x: np.asarray(x, dtype=float), (actual, mean, std))
    if y.shape != mu.shape or y.shape != sigma.shape or np.any(sigma <= 0):
        raise ValueError("formas devem coincidir e std deve ser positivo")
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def evaluate_gaussian(
    actual: ArrayLike,
    mean: ArrayLike,
    std: ArrayLike,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Calcula acurácia, calibração, sharpness e CRPS."""

    y, mu, sigma = map(lambda x: np.asarray(x, dtype=float), (actual, mean, std))
    z = norm.ppf((1 + confidence) / 2)
    lower, upper = mu - z * sigma, mu + z * sigma
    errors = y - mu
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "coverage": float(np.mean((y >= lower) & (y <= upper))),
        "mean_width": float(np.mean(upper - lower)),
        "crps": float(np.mean(gaussian_crps(y, mu, sigma))),
    }


def select_model(scoreboard: dict[str, dict[str, float]], metric: str = "crps") -> str:
    """Seleciona o menor score, exigindo a métrica para todos os modelos."""

    if not scoreboard or any(metric not in scores for scores in scoreboard.values()):
        raise ValueError("scoreboard vazio ou métrica ausente")
    return min(scoreboard, key=lambda name: scoreboard[name][metric])

