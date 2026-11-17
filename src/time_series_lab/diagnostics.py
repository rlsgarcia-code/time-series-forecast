"""Caracterização de séries e diagnóstico residual orientados à escolha de modelo."""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from .change_points import best_single_mean_change
from .contracts import (
    ModelChoiceEvidence,
    ModelRun,
    ResidualDiagnostics,
    SeriesProfile,
    TimeSeries,
    ValidationError,
)
from .regime_switching import hamilton_filter_gaussian
from .state_space import kalman_filter_local_level
from .volatility import garch_variance


def _safe_lag(size: int, requested: int) -> int:
    return max(1, min(requested, size // 3 - 1))


def _acf_tuple(values: np.ndarray, lag: int) -> tuple[float, ...]:
    return tuple(float(value) for value in acf(values, nlags=lag, fft=True))


def _pacf_tuple(values: np.ndarray, lag: int) -> tuple[float, ...]:
    return tuple(float(value) for value in pacf(values, nlags=lag, method="ywm"))


def characterize_series(series: TimeSeries, *, max_lag: int = 24) -> SeriesProfile:
    """Extrai evidências descritivas; não seleciona modelo automaticamente."""

    values = np.asarray(series.values, dtype=float)
    if values.size < 12:
        raise ValidationError("caracterização exige ao menos 12 observações")
    lag = _safe_lag(values.size, max_lag)
    time = np.arange(values.size, dtype=float)
    slope, intercept = np.polyfit(time, values, 1)
    trend = intercept + slope * time
    detrended = values - trend
    differences = np.diff(values)
    difference_lag = _safe_lag(differences.size, max_lag)

    try:
        adf_result = adfuller(values, autolag="AIC", result_object=False)
        adf_statistic, adf_pvalue = float(adf_result[0]), float(adf_result[1])
    except ValueError:
        adf_statistic, adf_pvalue = np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        try:
            kpss_result = kpss(
                values, regression="ct", nlags="auto", result_object=False
            )
            kpss_statistic, kpss_pvalue = float(kpss_result[0]), float(kpss_result[1])
        except ValueError:
            kpss_statistic, kpss_pvalue = np.nan, np.nan

    ljung = acorr_ljungbox(detrended, lags=[lag], return_df=True)
    jarque = stats.jarque_bera(detrended)
    arch_lags = min(10, max(1, values.size // 5))
    try:
        arch_pvalue = float(
            het_arch(detrended, nlags=arch_lags, result_object=False)[1]
        )
    except (ValueError, np.linalg.LinAlgError):
        arch_pvalue = np.nan

    min_size = max(6, values.size // 10)
    try:
        change = best_single_mean_change(values, min_size=min_size)
        change_ratio = float(change.improvement / max(change.no_change_cost, 1e-12))
    except ValueError:
        change_ratio = 0.0

    stationary = bool(
        np.isfinite(adf_pvalue)
        and np.isfinite(kpss_pvalue)
        and adf_pvalue < 0.05
        and kpss_pvalue > 0.05
    )
    serial_memory = float(ljung["lb_pvalue"].iloc[-1]) < 0.05
    volatility_clustering = np.isfinite(arch_pvalue) and arch_pvalue < 0.05
    scale = max(float(np.std(values, ddof=1)), 1e-12)
    trend_strength = float(abs(trend[-1] - trend[0]) / scale)
    structural_signal = change_ratio >= 0.15

    evidence = [
        ModelChoiceEvidence(
            "stationarity",
            f"ADF p={adf_pvalue:.4g}; KPSS p={kpss_pvalue:.4g}",
            "evidência compatível com estacionariedade"
            if stationary
            else "nível possivelmente não estacionário; testar diferenciação ou estado latente",
            ("ar1", "arima") if stationary else ("arima", "local_level"),
        ),
        ModelChoiceEvidence(
            "serial_memory",
            float(ljung["lb_pvalue"].iloc[-1]),
            "há dependência serial a explicar" if serial_memory else "não há memória linear forte no nível detrendido",
            ("ar1", "arima") if serial_memory else (),
        ),
        ModelChoiceEvidence(
            "trend_strength",
            trend_strength,
            "tendência relevante em relação à escala" if trend_strength >= 1 else "tendência linear fraca",
            ("arima", "local_level") if trend_strength >= 1 else (),
        ),
        ModelChoiceEvidence(
            "conditional_variance",
            arch_pvalue,
            "há agrupamento de volatilidade" if volatility_clustering else "sem evidência forte de ARCH",
            ("garch11",) if volatility_clustering else (),
        ),
        ModelChoiceEvidence(
            "structural_change",
            change_ratio,
            "quebra de nível merece investigação" if structural_signal else "uma quebra única não explica grande fração da variação",
            ("markov_switching", "local_level") if structural_signal else (),
        ),
    ]
    suggestions: list[str] = []
    for item in evidence:
        suggestions.extend(item.favors)
    suggestions = list(dict.fromkeys(suggestions))

    return SeriesProfile(
        series_id=series.series_id,
        nobs=int(values.size),
        summary={
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "skewness": float(stats.skew(values, bias=False)),
            "excess_kurtosis": float(stats.kurtosis(values, bias=False)),
            "trend_slope": float(slope),
            "trend_strength": trend_strength,
        },
        acf=_acf_tuple(values, lag),
        pacf=_pacf_tuple(values, lag),
        adf_statistic=adf_statistic,
        adf_pvalue=adf_pvalue,
        ljung_box_pvalue=float(ljung["lb_pvalue"].iloc[-1]),
        timestamps=series.timestamps,
        values=values,
        trend=trend,
        detrended=detrended,
        differences=differences,
        differenced_acf=_acf_tuple(differences, difference_lag),
        differenced_pacf=_pacf_tuple(differences, difference_lag),
        squared_acf=_acf_tuple(detrended**2, lag),
        kpss_statistic=kpss_statistic,
        kpss_pvalue=kpss_pvalue,
        jarque_bera_pvalue=float(jarque.pvalue),
        arch_lm_pvalue=arch_pvalue,
        change_improvement_ratio=change_ratio,
        evidence=tuple(evidence),
        suggested_models=tuple(suggestions),
    )


def _series_until_run(series: TimeSeries, run: ModelRun) -> TimeSeries:
    positions = np.flatnonzero(series.timestamps <= run.cutoff)
    if positions.size < 1 or series.timestamps[positions[-1]] != run.cutoff:
        raise ValidationError("cutoff do modelo não existe na série")
    end = int(positions[-1]) + 1
    return TimeSeries(
        series.series_id,
        series.timestamps[:end],
        series.values[:end],
        series.frequency,
        series.metadata,
        run.series_revision,
    )


def _residual_components(
    series: TimeSeries, run: ModelRun
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    series = _series_until_run(series, run)
    values = np.asarray(series.values, dtype=float)
    family = run.spec.family
    variance: np.ndarray | None = None

    if family == "ar1":
        fitted = float(run.state["intercept"]) + float(run.state["phi"]) * values[:-1]
        observed = values[1:]
        timestamps = series.timestamps[1:]
        residuals = observed - fitted
        variance = np.repeat(float(run.state["sigma"]) ** 2, residuals.size)
    elif family == "arima":
        order = tuple(int(value) for value in run.state["order"])
        result = ARIMA(values, order=order, trend=run.state["trend"]).filter(
            np.asarray(run.state["parameters"], dtype=float)
        )
        burn = max(1, sum(order))
        observed = values[burn:]
        fitted = np.asarray(result.fittedvalues, dtype=float)[burn:]
        timestamps = series.timestamps[burn:]
        residuals = observed - fitted
        variance = np.repeat(float(np.var(residuals, ddof=1)), residuals.size)
    elif family == "local_level":
        filtered = kalman_filter_local_level(
            values,
            initial_mean=float(values[0]),
            initial_variance=float(run.spec.initial_variance),
            state_variance=float(run.state["state_variance"]),
            observation_variance=float(run.state["observation_variance"]),
        )
        valid = np.isfinite(filtered.innovation)
        observed = values[valid]
        fitted = filtered.predicted_mean[valid]
        timestamps = series.timestamps[valid]
        residuals = filtered.innovation[valid]
        variance = filtered.innovation_variance[valid]
    elif family == "garch11":
        mean = float(run.state["mean"])
        observed = values
        fitted = np.repeat(mean, values.size)
        timestamps = series.timestamps
        residuals = values - mean
        variance = garch_variance(
            residuals,
            float(run.state["omega"]),
            float(run.state["alpha"]),
            float(run.state["beta"]),
        )
    elif family == "markov_switching":
        transition = np.asarray(run.state["transition"], dtype=float)
        means = np.asarray(run.state["means"], dtype=float)
        scales = np.asarray(run.state["scales"], dtype=float)
        filtered = hamilton_filter_gaussian(
            values,
            transition=transition,
            means=means,
            standard_deviations=scales,
        )
        weights = filtered.predicted_probabilities
        fitted = weights @ means
        variance = weights @ (scales**2 + means**2) - fitted**2
        observed = values
        timestamps = series.timestamps
        residuals = observed - fitted
    else:
        raise ValidationError(f"diagnóstico residual não implementado para {family}")
    return timestamps, observed, fitted, residuals, variance


def diagnose_residuals(
    series: TimeSeries, run: ModelRun, *, max_lag: int = 24
) -> ResidualDiagnostics:
    timestamps, observed, fitted, residuals, variance = _residual_components(series, run)
    mask = np.isfinite(observed) & np.isfinite(fitted) & np.isfinite(residuals)
    if variance is not None:
        mask &= np.isfinite(variance) & (variance > 0)
    timestamps = timestamps[mask]
    observed, fitted, residuals = observed[mask], fitted[mask], residuals[mask]
    variance = None if variance is None else np.asarray(variance)[mask]
    if residuals.size < 12:
        raise ValidationError("diagnóstico residual exige ao menos 12 resíduos válidos")
    scale = np.sqrt(variance) if variance is not None else np.repeat(np.std(residuals, ddof=1), residuals.size)
    standardized = residuals / np.maximum(scale, 1e-12)
    lag = _safe_lag(residuals.size, max_lag)
    ljung = acorr_ljungbox(standardized, lags=[lag], return_df=True)
    squared_ljung = acorr_ljungbox(standardized**2, lags=[lag], return_df=True)
    jarque = stats.jarque_bera(standardized)
    try:
        arch_pvalue = float(
            het_arch(standardized, nlags=min(10, lag), result_object=False)[1]
        )
    except (ValueError, np.linalg.LinAlgError):
        arch_pvalue = np.nan
    qq_theoretical, qq_sample = stats.probplot(standardized, dist="norm", fit=False)
    tests = {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals, ddof=1)),
        "ljung_box_pvalue": float(ljung["lb_pvalue"].iloc[-1]),
        "squared_ljung_box_pvalue": float(squared_ljung["lb_pvalue"].iloc[-1]),
        "jarque_bera_pvalue": float(jarque.pvalue),
        "arch_lm_pvalue": arch_pvalue,
    }
    interpretation: list[str] = []
    if tests["ljung_box_pvalue"] < 0.05:
        interpretation.append("resíduos ainda possuem autocorrelação; a dinâmica da média está incompleta")
    if tests["squared_ljung_box_pvalue"] < 0.05 or (
        np.isfinite(arch_pvalue) and arch_pvalue < 0.05
    ):
        interpretation.append("quadrados dos resíduos possuem memória; considerar volatilidade condicional")
    if tests["jarque_bera_pvalue"] < 0.05:
        interpretation.append("resíduos não parecem gaussianos; revisar caudas, outliers ou distribuição")
    if not interpretation:
        interpretation.append("não foram detectadas violações fortes nos testes residuais selecionados")
    return ResidualDiagnostics(
        series.series_id,
        run.model_run_id,
        run.spec.family,
        timestamps,
        observed,
        fitted,
        residuals,
        standardized,
        variance,
        _acf_tuple(standardized, lag),
        _pacf_tuple(standardized, lag),
        _acf_tuple(standardized**2, lag),
        qq_theoretical,
        qq_sample,
        tests,
        tuple(interpretation),
    )


def _correlation_bars(ax, values: tuple[float, ...], nobs: int, title: str) -> None:
    lags = np.arange(len(values))
    ax.bar(lags, values, color="#2563EB", width=0.65)
    limit = 1.96 / np.sqrt(nobs)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(limit, color="#DC2626", linestyle="--", linewidth=1)
    ax.axhline(-limit, color="#DC2626", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("lag")


def plot_series_characterization(
    profile: SeriesProfile, *, figsize: tuple[float, float] = (14, 11)
):
    """Retorna figura com nível, tendência, diferenças, ACF/PACF e distribuição."""

    figure, axes = plt.subplots(3, 2, figsize=figsize)
    axes[0, 0].plot(profile.timestamps, profile.values, color="black", linewidth=1, label="série")
    axes[0, 0].plot(profile.timestamps, profile.trend, color="#DC2626", label="tendência linear")
    axes[0, 0].set_title(f"{profile.series_id}: nível e tendência")
    axes[0, 0].legend()
    axes[0, 1].plot(profile.timestamps[1:], profile.differences, color="#059669")
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Primeira diferença")
    _correlation_bars(axes[1, 0], profile.acf, profile.nobs, "ACF do nível")
    _correlation_bars(axes[1, 1], profile.pacf, profile.nobs, "PACF do nível")
    _correlation_bars(
        axes[2, 0], profile.differenced_acf, profile.differences.size, "ACF da diferença"
    )
    axes[2, 1].hist(profile.detrended, bins="auto", density=True, color="#7C3AED", alpha=0.7)
    axes[2, 1].set_title("Distribuição após retirar tendência linear")
    figure.tight_layout()
    return figure


def plot_residual_diagnostics(
    diagnostics: ResidualDiagnostics, *, figsize: tuple[float, float] = (14, 11)
):
    """Retorna figura residual; não chama ``show`` nem salva arquivos."""

    figure, axes = plt.subplots(3, 2, figsize=figsize)
    axes[0, 0].plot(diagnostics.timestamps, diagnostics.observed, color="black", label="observado")
    axes[0, 0].plot(diagnostics.timestamps, diagnostics.fitted_values, color="#EA580C", label="ajustado")
    axes[0, 0].set_title(f"{diagnostics.model_family}: observado × ajustado")
    axes[0, 0].legend()
    axes[0, 1].plot(diagnostics.timestamps, diagnostics.standardized_residuals, color="#2563EB")
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Resíduos padronizados")
    _correlation_bars(axes[1, 0], diagnostics.acf, diagnostics.residuals.size, "ACF dos resíduos")
    _correlation_bars(axes[1, 1], diagnostics.pacf, diagnostics.residuals.size, "PACF dos resíduos")
    axes[2, 0].scatter(diagnostics.qq_theoretical, diagnostics.qq_sample, s=14, alpha=0.65)
    low = min(diagnostics.qq_theoretical.min(), diagnostics.qq_sample.min())
    high = max(diagnostics.qq_theoretical.max(), diagnostics.qq_sample.max())
    axes[2, 0].plot([low, high], [low, high], color="#DC2626", linestyle="--")
    axes[2, 0].set_title("Q–Q Normal")
    axes[2, 0].set_xlabel("quantil teórico")
    axes[2, 0].set_ylabel("quantil residual")
    if diagnostics.conditional_variance is None:
        axes[2, 1].hist(diagnostics.standardized_residuals, bins="auto", density=True, color="#7C3AED")
        axes[2, 1].set_title("Distribuição dos resíduos padronizados")
    else:
        axes[2, 1].plot(diagnostics.timestamps, diagnostics.conditional_variance, color="#7C3AED")
        axes[2, 1].set_title("Variância condicional/preditiva")
    figure.tight_layout()
    return figure


__all__ = [
    "characterize_series",
    "diagnose_residuals",
    "plot_residual_diagnostics",
    "plot_series_characterization",
]
