"""Adaptadores concretos das famílias estudadas nos notebooks."""

from __future__ import annotations

from dataclasses import asdict
import warnings
from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

from .contracts import (
    AR1Spec,
    ARIMASpec,
    AdditiveShockScenario,
    BaselineScenario,
    EmpiricalDistribution,
    ForcedRegimeScenario,
    GARCH11Spec,
    GaussianDistribution,
    GaussianMixtureDistribution,
    LocalLevelSpec,
    ModelAdapter,
    ModelCapabilities,
    ModelRun,
    ModelSpec,
    ParameterShiftScenario,
    PredictiveDistribution,
    SimulationScenario,
    TimeSeries,
    TwoRegimeGaussianSpec,
    UnsupportedCapabilityError,
    ValidationError,
)
from .estimation import fit_ar1_ols
from .regime_switching import (
    forecast_regime_probabilities,
    hamilton_filter_gaussian,
)
from .state_space import kalman_filter_local_level
from .volatility import fit_garch11


def _validate_horizon_paths(horizon: int, n_paths: int) -> None:
    if horizon < 1 or n_paths < 1:
        raise ValidationError("horizon e n_paths devem ser positivos")


def _post_process_scenario(
    paths: np.ndarray, scenario: SimulationScenario
) -> np.ndarray:
    result = np.asarray(paths, dtype=float).copy()
    if isinstance(scenario, BaselineScenario):
        return result
    if isinstance(scenario, AdditiveShockScenario):
        start = scenario.step - 1
        stop = start + scenario.duration
        if start < 0 or stop > result.shape[1] or scenario.duration < 1:
            raise ValidationError("choque fora do horizonte simulado")
        result[:, start:stop] += scenario.magnitude
        return result
    if isinstance(scenario, (ParameterShiftScenario, ForcedRegimeScenario)):
        raise UnsupportedCapabilityError(
            f"cenário {scenario.name} não é suportado por este adaptador"
        )
    raise UnsupportedCapabilityError("cenário desconhecido")


def _ar1_covariance(phi: float, sigma: float, horizon: int) -> np.ndarray:
    covariance = np.empty((horizon, horizon), dtype=float)
    for i in range(horizon):
        accumulated = sigma**2 * sum(phi ** (2 * k) for k in range(i + 1))
        for j in range(i, horizon):
            covariance[i, j] = covariance[j, i] = phi ** (j - i) * accumulated
    return covariance


class AR1Adapter:
    family = "ar1"
    version = "1.0"
    capabilities = ModelCapabilities(
        scenarios=("baseline", "additive_shock", "parameter_shift")
    )

    def fit(
        self, series: TimeSeries, spec: ModelSpec
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        if not isinstance(spec, AR1Spec):
            raise ValidationError("AR1Adapter exige AR1Spec")
        fit = fit_ar1_ols(series.values)
        intercept = fit.intercept if spec.include_intercept else 0.0
        if not spec.include_intercept:
            lagged, response = series.values[:-1], series.values[1:]
            phi = float(lagged @ response / (lagged @ lagged))
            residuals = response - phi * lagged
            sigma = float(np.sqrt(residuals @ residuals / (response.size - 1)))
        else:
            phi, sigma, residuals = fit.phi, fit.sigma, fit.residuals
        state = {
            "intercept": float(intercept),
            "phi": float(phi),
            "sigma": float(sigma),
            "last_value": float(series.values[-1]),
        }
        diagnostics = {
            "nobs": int(fit.nobs),
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals, ddof=1)),
            "stationary": bool(abs(phi) < 1),
        }
        return state, diagnostics

    def forecast(
        self, run: ModelRun, horizon: int, *, seed: int
    ) -> PredictiveDistribution:
        if horizon < 1:
            raise ValidationError("horizon deve ser positivo")
        c = float(run.state["intercept"])
        phi = float(run.state["phi"])
        sigma = float(run.state["sigma"])
        last = float(run.state["last_value"])
        h = np.arange(1, horizon + 1)
        if abs(phi - 1.0) < 1e-10:
            mean = last + c * h
        else:
            mean = phi**h * last + c * (1 - phi**h) / (1 - phi)
        return GaussianDistribution(mean, _ar1_covariance(phi, sigma, horizon))

    def simulate(
        self,
        run: ModelRun,
        horizon: int,
        n_paths: int,
        *,
        scenario: SimulationScenario,
        seed: int | None,
    ) -> np.ndarray:
        _validate_horizon_paths(horizon, n_paths)
        rng = np.random.default_rng(seed)
        state = dict(run.state)
        c, phi, sigma = state["intercept"], state["phi"], state["sigma"]
        current = np.repeat(float(state["last_value"]), n_paths)
        paths = np.empty((n_paths, horizon))
        for step in range(horizon):
            if isinstance(scenario, ParameterShiftScenario) and step + 1 >= scenario.start_step:
                c = scenario.overrides.get("intercept", c)
                phi = scenario.overrides.get("phi", phi)
                sigma = scenario.overrides.get("sigma", sigma)
            current = c + phi * current + rng.normal(0.0, sigma, n_paths)
            if (
                isinstance(scenario, AdditiveShockScenario)
                and scenario.step <= step + 1 < scenario.step + scenario.duration
            ):
                current = current + scenario.magnitude
            paths[:, step] = current
        if isinstance(scenario, ForcedRegimeScenario):
            raise UnsupportedCapabilityError("AR1 não possui regimes discretos")
        return paths


class ARIMAAdapter:
    family = "arima"
    version = "1.0"
    capabilities = ModelCapabilities(scenarios=("baseline", "additive_shock"))

    @staticmethod
    def _result(state: Mapping[str, Any]):
        history = np.asarray(state["history"], dtype=float)
        order = tuple(int(value) for value in state["order"])
        trend = state["trend"]
        model = ARIMA(history, order=order, trend=trend)
        return model.filter(np.asarray(state["parameters"], dtype=float))

    def fit(
        self, series: TimeSeries, spec: ModelSpec
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        if not isinstance(spec, ARIMASpec):
            raise ValidationError("ARIMAAdapter exige ARIMASpec")
        trend = spec.trend if spec.trend is not None else ("c" if spec.order[1] == 0 else "n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = ARIMA(series.values, order=spec.order, trend=trend).fit()
        state = {
            "history": tuple(float(value) for value in series.values),
            "order": tuple(spec.order),
            "trend": trend,
            "parameters": tuple(float(value) for value in result.params),
            "simulation_paths": spec.simulation_paths,
        }
        diagnostics = {
            "aic": float(result.aic),
            "bic": float(result.bic),
            "loglikelihood": float(result.llf),
            "converged": bool(result.mle_retvals.get("converged", True)),
        }
        return state, diagnostics

    def forecast(
        self, run: ModelRun, horizon: int, *, seed: int
    ) -> PredictiveDistribution:
        result = self._result(run.state)
        paths = self._simulate_result(
            result, horizon, int(run.state["simulation_paths"]), seed=seed
        )
        return EmpiricalDistribution(paths)

    @staticmethod
    def _simulate_result(result, horizon: int, n_paths: int, *, seed: int | None) -> np.ndarray:
        _validate_horizon_paths(horizon, n_paths)
        values = np.asarray(
            result.simulate(
                nsimulations=horizon,
                repetitions=n_paths,
                anchor="end",
                rng=np.random.default_rng(seed),
            )
        )
        if values.ndim == 3:
            values = values[:, 0, :]
        if values.ndim == 1:
            values = values[:, None]
        return values.T

    def simulate(
        self,
        run: ModelRun,
        horizon: int,
        n_paths: int,
        *,
        scenario: SimulationScenario,
        seed: int | None,
    ) -> np.ndarray:
        paths = self._simulate_result(self._result(run.state), horizon, n_paths, seed=seed)
        return _post_process_scenario(paths, scenario)


class LocalLevelAdapter:
    family = "local_level"
    version = "1.0"
    capabilities = ModelCapabilities(
        missing_values=True,
        scenarios=("baseline", "additive_shock", "parameter_shift"),
    )

    def fit(
        self, series: TimeSeries, spec: ModelSpec
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        if not isinstance(spec, LocalLevelSpec):
            raise ValidationError("LocalLevelAdapter exige LocalLevelSpec")
        differences = np.diff(series.values)
        if differences.size < 3:
            raise ValidationError("nível local exige ao menos quatro observações")
        gamma0 = float(np.var(differences, ddof=1))
        gamma1 = float(
            np.mean(
                (differences[1:] - differences[1:].mean())
                * (differences[:-1] - differences[:-1].mean())
            )
        )
        observation_variance = max(-gamma1, 1e-8)
        state_variance = max(gamma0 - 2 * observation_variance, 1e-8)
        filtered = kalman_filter_local_level(
            series.values,
            initial_mean=float(series.values[0]),
            initial_variance=spec.initial_variance,
            state_variance=state_variance,
            observation_variance=observation_variance,
        )
        state = {
            "filtered_mean": float(filtered.filtered_mean[-1]),
            "filtered_variance": float(filtered.filtered_variance[-1]),
            "state_variance": state_variance,
            "observation_variance": observation_variance,
        }
        diagnostics = {
            "loglikelihood": filtered.loglikelihood,
            "nobs": int(series.values.size),
        }
        return state, diagnostics

    def forecast(
        self, run: ModelRun, horizon: int, *, seed: int
    ) -> PredictiveDistribution:
        q = float(run.state["state_variance"])
        r = float(run.state["observation_variance"])
        p = float(run.state["filtered_variance"])
        mean = np.repeat(float(run.state["filtered_mean"]), horizon)
        steps = np.arange(1, horizon + 1)
        covariance = p + q * np.minimum.outer(steps, steps)
        covariance += np.eye(horizon) * r
        return GaussianDistribution(mean, covariance)

    def simulate(
        self,
        run: ModelRun,
        horizon: int,
        n_paths: int,
        *,
        scenario: SimulationScenario,
        seed: int | None,
    ) -> np.ndarray:
        _validate_horizon_paths(horizon, n_paths)
        rng = np.random.default_rng(seed)
        q = float(run.state["state_variance"])
        r = float(run.state["observation_variance"])
        latent = rng.normal(
            float(run.state["filtered_mean"]),
            np.sqrt(float(run.state["filtered_variance"])),
            n_paths,
        )
        paths = np.empty((n_paths, horizon))
        for step in range(horizon):
            if isinstance(scenario, ParameterShiftScenario) and step + 1 >= scenario.start_step:
                q = scenario.overrides.get("state_variance", q)
                r = scenario.overrides.get("observation_variance", r)
            latent = latent + rng.normal(0.0, np.sqrt(q), n_paths)
            paths[:, step] = latent + rng.normal(0.0, np.sqrt(r), n_paths)
        return _post_process_scenario(paths, scenario) if not isinstance(
            scenario, ParameterShiftScenario
        ) else paths


class GARCH11Adapter:
    family = "garch11"
    version = "1.0"
    capabilities = ModelCapabilities(
        scenarios=("baseline", "additive_shock", "parameter_shift")
    )

    def fit(
        self, series: TimeSeries, spec: ModelSpec
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        if not isinstance(spec, GARCH11Spec):
            raise ValidationError("GARCH11Adapter exige GARCH11Spec")
        mean = float(np.mean(series.values)) if spec.mean == "constant" else 0.0
        centered = series.values - mean
        fit = fit_garch11(centered)
        state = {
            "omega": fit.omega,
            "alpha": fit.alpha,
            "beta": fit.beta,
            "mean": mean,
            "last_return": float(centered[-1]),
            "last_variance": float(fit.variance[-1]),
            "simulation_paths": spec.simulation_paths,
        }
        diagnostics = {
            "converged": fit.success,
            "persistence": fit.alpha + fit.beta,
            "unconditional_variance": fit.omega / (1 - fit.alpha - fit.beta),
        }
        return state, diagnostics

    def _paths(
        self,
        run: ModelRun,
        horizon: int,
        n_paths: int,
        scenario: SimulationScenario,
        seed: int | None,
    ) -> np.ndarray:
        _validate_horizon_paths(horizon, n_paths)
        rng = np.random.default_rng(seed)
        omega, alpha, beta = (
            float(run.state["omega"]),
            float(run.state["alpha"]),
            float(run.state["beta"]),
        )
        mean = float(run.state["mean"])
        previous_return = np.repeat(float(run.state["last_return"]), n_paths)
        previous_variance = np.repeat(float(run.state["last_variance"]), n_paths)
        paths = np.empty((n_paths, horizon))
        for step in range(horizon):
            if isinstance(scenario, ParameterShiftScenario) and step + 1 >= scenario.start_step:
                omega = scenario.overrides.get("omega", omega)
                alpha = scenario.overrides.get("alpha", alpha)
                beta = scenario.overrides.get("beta", beta)
            if omega <= 0 or min(alpha, beta) < 0 or alpha + beta >= 1:
                raise ValidationError("mudança de parâmetro torna GARCH inválido")
            variance = omega + alpha * previous_return**2 + beta * previous_variance
            previous_return = np.sqrt(variance) * rng.normal(size=n_paths)
            paths[:, step] = mean + previous_return
            previous_variance = variance
        return _post_process_scenario(paths, scenario) if not isinstance(
            scenario, ParameterShiftScenario
        ) else paths

    def forecast(
        self, run: ModelRun, horizon: int, *, seed: int
    ) -> PredictiveDistribution:
        paths = self._paths(
            run,
            horizon,
            int(run.state["simulation_paths"]),
            BaselineScenario(),
            seed,
        )
        return EmpiricalDistribution(paths)

    def simulate(
        self,
        run: ModelRun,
        horizon: int,
        n_paths: int,
        *,
        scenario: SimulationScenario,
        seed: int | None,
    ) -> np.ndarray:
        return self._paths(run, horizon, n_paths, scenario, seed)


def _unpack_markov(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean0 = raw[0]
    means = np.array([mean0, mean0 + np.exp(raw[1])])
    scales = np.exp(raw[2:4])
    p00, p11 = expit(raw[4]), expit(raw[5])
    transition = np.array([[p00, 1 - p00], [1 - p11, p11]])
    return transition, means, scales


class TwoRegimeGaussianAdapter:
    family = "markov_switching"
    version = "1.0"
    capabilities = ModelCapabilities(
        scenarios=("baseline", "additive_shock", "parameter_shift", "forced_regime")
    )

    def fit(
        self, series: TimeSeries, spec: ModelSpec
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        if not isinstance(spec, TwoRegimeGaussianSpec):
            raise ValidationError("TwoRegimeGaussianAdapter exige TwoRegimeGaussianSpec")
        values = series.values
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = max(float(np.std(values)), 1e-3)
        initial = np.array(
            [q25, np.log(max(q75 - q25, 0.2)), np.log(scale / 2), np.log(scale / 2), logit(0.9), logit(0.9)]
        )

        def objective(raw: np.ndarray) -> float:
            transition, means, scales = _unpack_markov(raw)
            return -hamilton_filter_gaussian(
                values,
                transition=transition,
                means=means,
                standard_deviations=scales,
            ).loglikelihood

        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            options={"maxiter": spec.max_iterations},
        )
        transition, means, scales = _unpack_markov(result.x)
        filtered = hamilton_filter_gaussian(
            values,
            transition=transition,
            means=means,
            standard_deviations=scales,
        )
        state = {
            "transition": transition.tolist(),
            "means": means.tolist(),
            "scales": scales.tolist(),
            "last_probabilities": filtered.filtered_probabilities[-1].tolist(),
        }
        diagnostics = {
            "converged": bool(result.success),
            "loglikelihood": filtered.loglikelihood,
            "message": str(result.message),
        }
        return state, diagnostics

    def forecast(
        self, run: ModelRun, horizon: int, *, seed: int
    ) -> PredictiveDistribution:
        transition = np.asarray(run.state["transition"], dtype=float)
        weights = forecast_regime_probabilities(
            run.state["last_probabilities"], transition, horizon
        )
        return GaussianMixtureDistribution(
            weights,
            np.asarray(run.state["means"], dtype=float),
            np.asarray(run.state["scales"], dtype=float),
            transition,
        )

    def simulate(
        self,
        run: ModelRun,
        horizon: int,
        n_paths: int,
        *,
        scenario: SimulationScenario,
        seed: int | None,
    ) -> np.ndarray:
        _validate_horizon_paths(horizon, n_paths)
        transition = np.asarray(run.state["transition"], dtype=float)
        means = np.asarray(run.state["means"], dtype=float)
        scales = np.asarray(run.state["scales"], dtype=float)
        if isinstance(scenario, ParameterShiftScenario):
            if scenario.start_step != 1:
                raise UnsupportedCapabilityError(
                    "Markov aceita mudança de parâmetro apenas desde o primeiro passo"
                )
            means = np.array([scenario.overrides.get(f"mean_{i}", value) for i, value in enumerate(means)])
            scales = np.array([scenario.overrides.get(f"scale_{i}", value) for i, value in enumerate(scales)])
        rng = np.random.default_rng(seed)
        states = np.empty((n_paths, horizon), dtype=int)
        initial = np.asarray(run.state["last_probabilities"], dtype=float) @ transition
        states[:, 0] = [rng.choice(2, p=initial) for _ in range(n_paths)]
        for step in range(1, horizon):
            if isinstance(scenario, ForcedRegimeScenario) and step + 1 >= scenario.start_step:
                if scenario.regime not in (0, 1):
                    raise ValidationError("regime deve ser 0 ou 1")
                states[:, step] = scenario.regime
            else:
                states[:, step] = [rng.choice(2, p=transition[state]) for state in states[:, step - 1]]
        paths = rng.normal(means[states], scales[states])
        return _post_process_scenario(paths, scenario) if isinstance(
            scenario, (BaselineScenario, AdditiveShockScenario)
        ) else paths


def built_in_adapters() -> tuple[ModelAdapter, ...]:
    return (
        AR1Adapter(),
        ARIMAAdapter(),
        LocalLevelAdapter(),
        GARCH11Adapter(),
        TwoRegimeGaussianAdapter(),
    )


__all__ = [
    "AR1Adapter",
    "ARIMAAdapter",
    "GARCH11Adapter",
    "LocalLevelAdapter",
    "TwoRegimeGaussianAdapter",
    "built_in_adapters",
]
