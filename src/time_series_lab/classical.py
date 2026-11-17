"""Processos ARMA e transformações de integração."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def simulate_arma(
    n: int,
    *,
    ar: ArrayLike = (),
    ma: ArrayLike = (),
    sigma: float = 1.0,
    burnin: int = 300,
    seed: int | None = None,
) -> FloatArray:
    """Simula ``X_t = sum(phi_i X_t-i) + eps_t + sum(theta_j eps_t-j)``."""

    if n < 1 or burnin < 0 or sigma <= 0:
        raise ValueError("n e sigma devem ser positivos; burnin não negativo")
    phi = np.asarray(ar, dtype=float)
    theta = np.asarray(ma, dtype=float)
    p, q = phi.size, theta.size
    pad = max(p, q)
    total = n + burnin + pad
    rng = np.random.default_rng(seed)
    epsilon = rng.normal(0.0, sigma, total)
    values = np.zeros(total)
    for t in range(pad, total):
        ar_part = sum(phi[i] * values[t - i - 1] for i in range(p))
        ma_part = sum(theta[j] * epsilon[t - j - 1] for j in range(q))
        values[t] = ar_part + epsilon[t] + ma_part
    return values[burnin + pad:]


def difference(values: ArrayLike, order: int = 1) -> FloatArray:
    """Aplica a diferença ``(1-B)^order``."""

    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or order < 0 or order >= x.size:
        raise ValueError("série unidimensional e ordem válida são necessárias")
    return np.diff(x, n=order)


def integrate_differences(last_level: float, differences: ArrayLike) -> FloatArray:
    """Reconstrói níveis futuros acumulando diferenças previstas."""

    diffs = np.asarray(differences, dtype=float)
    if diffs.ndim != 1:
        raise ValueError("differences deve ser unidimensional")
    return last_level + np.cumsum(diffs)

