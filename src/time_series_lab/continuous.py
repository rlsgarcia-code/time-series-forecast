"""Simulações em tempo contínuo discretizadas para ensino."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def brownian_paths(
    n_steps: int,
    *,
    horizon: float = 1.0,
    n_paths: int = 1,
    seed: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Gera movimento Browniano padrão e sua grade temporal."""

    if n_steps < 1 or horizon <= 0 or n_paths < 1:
        raise ValueError("n_steps, horizon e n_paths devem ser positivos")
    dt = horizon / n_steps
    rng = np.random.default_rng(seed)
    increments = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
    paths = np.column_stack([np.zeros(n_paths), np.cumsum(increments, axis=1)])
    return np.linspace(0.0, horizon, n_steps + 1), paths


def gbm_exact_paths(
    s0: float,
    mu: float,
    sigma: float,
    n_steps: int,
    *,
    horizon: float = 1.0,
    n_paths: int = 1,
    seed: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Simula GBM pela solução exata ``S_t=S0 exp((mu-sigma²/2)t+sigma W_t)``."""

    if s0 <= 0 or sigma < 0:
        raise ValueError("s0 deve ser positivo e sigma não negativo")
    time, brownian = brownian_paths(
        n_steps, horizon=horizon, n_paths=n_paths, seed=seed
    )
    exponent = (mu - 0.5 * sigma**2) * time + sigma * brownian
    return time, s0 * np.exp(exponent)


def quadratic_variation(path: FloatArray) -> float:
    """Soma dos quadrados dos incrementos de um caminho."""

    x = np.asarray(path, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("path deve ser unidimensional")
    return float(np.sum(np.diff(x) ** 2))

