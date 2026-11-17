"""Simulação, filtragem e estimação simples de GARCH(1,1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]


def garch_variance(
    returns: ArrayLike, omega: float, alpha: float, beta: float
) -> FloatArray:
    """Filtra ``h_t = omega + alpha r_(t-1)^2 + beta h_(t-1)``."""

    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size < 2 or omega <= 0 or min(alpha, beta) < 0 or alpha + beta >= 1:
        raise ValueError("parâmetros GARCH devem satisfazer omega>0, alpha,beta>=0 e alpha+beta<1")
    variance = np.empty(r.size)
    variance[0] = omega / (1 - alpha - beta)
    for t in range(1, r.size):
        variance[t] = omega + alpha * r[t - 1] ** 2 + beta * variance[t - 1]
    return variance


def simulate_garch11(
    n: int,
    *,
    omega: float,
    alpha: float,
    beta: float,
    seed: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Simula retornos e variâncias condicionais GARCH(1,1)."""

    if n < 2:
        raise ValueError("n deve ser ao menos 2")
    # Validação reaproveitando a mesma restrição.
    if omega <= 0 or min(alpha, beta) < 0 or alpha + beta >= 1:
        raise ValueError("parâmetros GARCH inválidos")
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    variance = np.empty(n)
    returns = np.empty(n)
    variance[0] = omega / (1 - alpha - beta)
    returns[0] = np.sqrt(variance[0]) * z[0]
    for t in range(1, n):
        variance[t] = omega + alpha * returns[t - 1] ** 2 + beta * variance[t - 1]
        returns[t] = np.sqrt(variance[t]) * z[t]
    return returns, variance


@dataclass(frozen=True)
class GARCHFit:
    omega: float
    alpha: float
    beta: float
    variance: FloatArray
    success: bool


def fit_garch11(returns: ArrayLike) -> GARCHFit:
    """Estima GARCH(1,1) por máxima verossimilhança gaussiana."""

    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size < 30 or not np.all(np.isfinite(r)):
        raise ValueError("returns deve ser um vetor finito com ao menos 30 valores")

    def objective(params: FloatArray) -> float:
        omega, alpha, beta = params
        try:
            h = garch_variance(r, omega, alpha, beta)
        except ValueError:
            return 1e100
        return float(0.5 * np.sum(np.log(h) + r**2 / h))

    initial_variance = float(np.var(r))
    result = minimize(
        objective,
        x0=np.array([0.05 * initial_variance, 0.08, 0.88]),
        method="SLSQP",
        bounds=[(1e-10, None), (0.0, 0.999), (0.0, 0.999)],
        constraints=[{"type": "ineq", "fun": lambda p: 0.999 - p[1] - p[2]}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    omega, alpha, beta = result.x
    variance = garch_variance(r, float(omega), float(alpha), float(beta))
    return GARCHFit(float(omega), float(alpha), float(beta), variance, bool(result.success))

