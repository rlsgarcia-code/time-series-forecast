"""Estimação e avaliação de modelos AR(1) usados no Notebook 02."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def _as_series(values: ArrayLike, *, minimum: int = 4) -> FloatArray:
    series = np.asarray(values, dtype=float)
    if series.ndim != 1 or series.size < minimum or not np.all(np.isfinite(series)):
        raise ValueError(f"values deve ser um vetor finito com ao menos {minimum} observações")
    return series


@dataclass(frozen=True)
class AR1Fit:
    """Resultado de mínimos quadrados para ``X_t = c + phi X_(t-1) + erro_t``."""

    intercept: float
    phi: float
    sigma: float
    standard_errors: FloatArray
    covariance: FloatArray
    fitted_values: FloatArray
    residuals: FloatArray
    nobs: int
    degrees_of_freedom: int

    @property
    def parameters(self) -> FloatArray:
        return np.array([self.intercept, self.phi], dtype=float)


def fit_ar1_ols(values: ArrayLike) -> AR1Fit:
    """Estima intercepto e ``phi`` de um AR(1) por mínimos quadrados ordinários."""

    series = _as_series(values)
    response = series[1:]
    lagged = series[:-1]
    design = np.column_stack([np.ones_like(lagged), lagged])

    coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    if rank < 2:
        raise ValueError("a regressão AR(1) exige variação nos valores defasados")

    fitted = design @ coefficients
    residuals = response - fitted
    degrees_of_freedom = response.size - design.shape[1]
    residual_variance = float(residuals @ residuals / degrees_of_freedom)
    covariance = residual_variance * np.linalg.inv(design.T @ design)

    return AR1Fit(
        intercept=float(coefficients[0]),
        phi=float(coefficients[1]),
        sigma=float(np.sqrt(residual_variance)),
        standard_errors=np.sqrt(np.diag(covariance)),
        covariance=covariance,
        fitted_values=fitted,
        residuals=residuals,
        nobs=int(response.size),
        degrees_of_freedom=int(degrees_of_freedom),
    )


def ar1_parameter_intervals(
    fit: AR1Fit, confidence: float = 0.95
) -> tuple[FloatArray, FloatArray]:
    """Intervalos normais aproximados para ``c`` e ``phi``."""

    if not 0 < confidence < 1:
        raise ValueError("confidence deve estar entre 0 e 1")
    z = norm.ppf((1 + confidence) / 2)
    margin = z * fit.standard_errors
    return fit.parameters - margin, fit.parameters + margin


def rolling_one_step_ar1(values: ArrayLike, initial_train_size: int) -> FloatArray:
    """Prevê um passo por vez, reestimando o AR(1) após cada nova observação."""

    series = _as_series(values, minimum=6)
    if not 4 <= initial_train_size < series.size:
        raise ValueError("initial_train_size deve estar entre 4 e len(values)-1")

    predictions = np.empty(series.size - initial_train_size, dtype=float)
    for index, target in enumerate(range(initial_train_size, series.size)):
        fit = fit_ar1_ols(series[:target])
        predictions[index] = fit.intercept + fit.phi * series[target - 1]
    return predictions


def forecast_metrics(
    actual: ArrayLike,
    predicted: ArrayLike,
    *,
    lower: ArrayLike | None = None,
    upper: ArrayLike | None = None,
) -> dict[str, float]:
    """Calcula MAE, RMSE e, opcionalmente, cobertura de intervalo."""

    observed = np.asarray(actual, dtype=float)
    mean = np.asarray(predicted, dtype=float)
    if observed.shape != mean.shape or observed.ndim != 1:
        raise ValueError("actual e predicted devem ser vetores com o mesmo formato")
    errors = observed - mean
    metrics = {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }
    if (lower is None) != (upper is None):
        raise ValueError("lower e upper devem ser informados juntos")
    if lower is not None and upper is not None:
        low = np.asarray(lower, dtype=float)
        high = np.asarray(upper, dtype=float)
        if low.shape != observed.shape or high.shape != observed.shape:
            raise ValueError("os limites devem ter o mesmo formato de actual")
        metrics["coverage"] = float(np.mean((observed >= low) & (observed <= high)))
        metrics["mean_interval_width"] = float(np.mean(high - low))
    return metrics

