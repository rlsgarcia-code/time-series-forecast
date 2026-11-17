"""Simulações e fórmulas analíticas usadas no Notebook 01.

Convenção: processos recursivos retornam ``n_steps + 1`` observações, incluindo
o valor inicial em t=0. Ruído branco retorna exatamente ``n_steps`` inovações.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def _validate_counts(n_steps: int, n_paths: int = 1) -> None:
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError("n_steps deve ser um inteiro positivo")
    if not isinstance(n_paths, (int, np.integer)) or n_paths < 1:
        raise ValueError("n_paths deve ser um inteiro positivo")


def _validate_sigma(sigma: float) -> None:
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma deve ser finito e estritamente positivo")


def simulate_white_noise(
    n_steps: int,
    *,
    n_paths: int = 1,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed: int | None = None,
) -> FloatArray:
    """Gera caminhos independentes de ruído branco gaussiano."""

    _validate_counts(n_steps, n_paths)
    _validate_sigma(sigma)
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, size=(n_paths, n_steps))


def simulate_random_walk(
    n_steps: int,
    *,
    n_paths: int = 1,
    x0: float = 0.0,
    drift: float = 0.0,
    sigma: float = 1.0,
    seed: int | None = None,
) -> FloatArray:
    """Simula ``X_t = X_(t-1) + drift + epsilon_t``."""

    _validate_counts(n_steps, n_paths)
    _validate_sigma(sigma)
    rng = np.random.default_rng(seed)
    increments = drift + rng.normal(0.0, sigma, size=(n_paths, n_steps))
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = x0
    paths[:, 1:] = x0 + np.cumsum(increments, axis=1)
    return paths


def simulate_ar1(
    n_steps: int,
    *,
    phi: float,
    n_paths: int = 1,
    x0: float = 0.0,
    intercept: float = 0.0,
    sigma: float = 1.0,
    seed: int | None = None,
) -> FloatArray:
    """Simula ``X_t = intercept + phi X_(t-1) + epsilon_t``.

    Valores de ``phi`` fora do intervalo estacionário são permitidos para fins
    didáticos, desde que finitos.
    """

    _validate_counts(n_steps, n_paths)
    _validate_sigma(sigma)
    if not np.isfinite(phi):
        raise ValueError("phi deve ser finito")
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, sigma, size=(n_paths, n_steps))
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = x0
    for t in range(1, n_steps + 1):
        paths[:, t] = intercept + phi * paths[:, t - 1] + innovations[:, t - 1]
    return paths


def ar1_stationary_moments(
    *, phi: float, intercept: float = 0.0, sigma: float = 1.0
) -> tuple[float, float]:
    """Retorna média e variância incondicionais de um AR(1) estacionário."""

    _validate_sigma(sigma)
    if not np.isfinite(phi) or abs(phi) >= 1:
        raise ValueError("os momentos estacionários exigem |phi| < 1")
    mean = intercept / (1.0 - phi)
    variance = sigma**2 / (1.0 - phi**2)
    return float(mean), float(variance)


def theoretical_ar1_acf(phi: float, nlags: int) -> FloatArray:
    """Calcula a ACF teórica ``rho(k) = phi**k`` para k=0,...,nlags."""

    if not np.isfinite(phi) or abs(phi) >= 1:
        raise ValueError("a ACF estacionária exige |phi| < 1")
    if not isinstance(nlags, (int, np.integer)) or nlags < 0:
        raise ValueError("nlags deve ser um inteiro não negativo")
    return np.power(phi, np.arange(nlags + 1, dtype=float))


def sample_acf(values: ArrayLike, nlags: int) -> FloatArray:
    """Estima a autocorrelação amostral com normalização pelo lag zero."""

    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("values deve ser um vetor finito com ao menos 2 valores")
    if not isinstance(nlags, (int, np.integer)) or not 0 <= nlags < x.size:
        raise ValueError("nlags deve estar entre 0 e len(values)-1")
    centered = x - x.mean()
    denominator = np.dot(centered, centered)
    if denominator == 0:
        raise ValueError("a ACF não é definida para uma série constante")
    result = np.empty(nlags + 1, dtype=float)
    result[0] = 1.0
    for lag in range(1, nlags + 1):
        result[lag] = np.dot(centered[lag:], centered[:-lag]) / denominator
    return result


def shock_response(phi: float, shock: float, horizon: int) -> FloatArray:
    """Retorna o efeito de um choque isolado: ``shock * phi**h``."""

    if not np.isfinite(phi) or not np.isfinite(shock):
        raise ValueError("phi e shock devem ser finitos")
    if not isinstance(horizon, (int, np.integer)) or horizon < 0:
        raise ValueError("horizon deve ser um inteiro não negativo")
    return shock * np.power(phi, np.arange(horizon + 1, dtype=float))


@dataclass(frozen=True)
class ForecastResult:
    """Distribuição preditiva analítica de um AR(1) gaussiano."""

    horizon: NDArray[np.int64]
    mean: FloatArray
    std: FloatArray
    lower: FloatArray
    upper: FloatArray
    confidence: float


def forecast_ar1(
    last_value: float,
    horizon: int,
    *,
    phi: float,
    intercept: float = 0.0,
    sigma: float = 1.0,
    confidence: float = 0.95,
) -> ForecastResult:
    """Calcula média e intervalo preditivo h-passos-à-frente de um AR(1)."""

    _validate_sigma(sigma)
    if not np.isfinite(last_value) or not np.isfinite(phi) or not np.isfinite(intercept):
        raise ValueError("last_value, phi e intercept devem ser finitos")
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon deve ser um inteiro positivo")
    if not 0 < confidence < 1:
        raise ValueError("confidence deve estar entre 0 e 1")

    h = np.arange(1, horizon + 1, dtype=np.int64)
    if np.isclose(phi, 1.0):
        mean = last_value + intercept * h
    else:
        mean = (phi**h) * last_value + intercept * (1.0 - phi**h) / (1.0 - phi)

    # Soma geométrica das variâncias das inovações futuras.
    powers = phi ** (2 * np.arange(horizon, dtype=float))
    variance = sigma**2 * np.cumsum(powers)
    std = np.sqrt(variance)
    z = norm.ppf((1.0 + confidence) / 2.0)
    return ForecastResult(
        horizon=h,
        mean=np.asarray(mean, dtype=float),
        std=std,
        lower=np.asarray(mean - z * std, dtype=float),
        upper=np.asarray(mean + z * std, dtype=float),
        confidence=confidence,
    )


def simulate_ar1_future_paths(
    last_value: float,
    horizon: int,
    *,
    phi: float,
    n_paths: int = 1_000,
    intercept: float = 0.0,
    sigma: float = 1.0,
    seed: int | None = None,
) -> FloatArray:
    """Amostra caminhos futuros condicionais, sem repetir o valor observado."""

    _validate_counts(horizon, n_paths)
    paths = simulate_ar1(
        horizon,
        phi=phi,
        n_paths=n_paths,
        x0=last_value,
        intercept=intercept,
        sigma=sigma,
        seed=seed,
    )
    return paths[:, 1:]

