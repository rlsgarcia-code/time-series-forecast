"""Contratos públicos do motor de forecasting.

Este módulo contém apenas tipos de domínio e protocolos. Ele não escolhe banco,
fila ou framework web: a aplicação hospedeira injeta essas decisões.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Any, Callable, ClassVar, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class EngineError(RuntimeError):
    """Erro-base da API de alto nível."""


class ValidationError(EngineError, ValueError):
    """Entrada incompatível com um contrato do motor."""


class NotFittedError(EngineError):
    """Não existe modelo ajustado/promovido para a operação."""


class RepositoryConflictError(EngineError):
    """Uma escrita concorrente ou idempotente entrou em conflito."""


class UnsupportedCapabilityError(EngineError):
    """O adaptador selecionado não implementa a capacidade pedida."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _readonly_vector(values: ArrayLike, *, name: str, finite: bool = True) -> FloatArray:
    result = np.asarray(values, dtype=float).copy()
    if result.ndim != 1 or (finite and np.any(~np.isfinite(result))):
        raise ValidationError(f"{name} deve ser um vetor finito")
    result.setflags(write=False)
    return result


def _readonly_matrix(values: ArrayLike, *, name: str) -> FloatArray:
    result = np.asarray(values, dtype=float).copy()
    if result.ndim != 2 or np.any(~np.isfinite(result)):
        raise ValidationError(f"{name} deve ser uma matriz finita")
    result.setflags(write=False)
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.DatetimeIndex):
        return [timestamp.isoformat() for timestamp in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_payload"):
        return _jsonable(value.to_payload())
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable({
            key: _jsonable(getattr(value, key))
            for key in value.__dataclass_fields__
            if not key.startswith("_")
        })
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def content_id(prefix: str, payload: Any) -> str:
    canonical = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


@dataclass(frozen=True)
class TimeSeries:
    """Série univariada regular e identificada."""

    series_id: str
    timestamps: pd.DatetimeIndex
    values: FloatArray
    frequency: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    revision: int = 0

    def __post_init__(self) -> None:
        index = pd.DatetimeIndex(self.timestamps).copy()
        values = _readonly_vector(self.values, name="values")
        if not self.series_id.strip():
            raise ValidationError("series_id não pode ser vazio")
        if len(index) != len(values) or len(index) < 1:
            raise ValidationError("timestamps e values devem ter o mesmo tamanho não vazio")
        if index.tz is None:
            raise ValidationError("timestamps devem ter timezone explícito")
        if not index.is_monotonic_increasing or not index.is_unique:
            raise ValidationError("timestamps devem ser crescentes e únicos")
        if not self.frequency:
            raise ValidationError("frequency deve ser informada")
        expected = pd.date_range(index[0], periods=len(index), freq=self.frequency)
        if not index.equals(expected):
            raise ValidationError("timestamps possuem gaps ou frequência incompatível")
        if self.revision < 0:
            raise ValidationError("revision não pode ser negativa")
        object.__setattr__(self, "timestamps", index)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_pandas(
        cls,
        series_id: str,
        series: pd.Series,
        *,
        frequency: str | None = None,
        timezone_name: str = "UTC",
        metadata: Mapping[str, Any] | None = None,
    ) -> "TimeSeries":
        index = pd.DatetimeIndex(series.index)
        if index.tz is None:
            index = index.tz_localize(timezone_name)
        inferred = frequency or pd.infer_freq(index)
        if inferred is None:
            raise ValidationError("não foi possível inferir a frequência")
        return cls(series_id, index, series.to_numpy(dtype=float), inferred, metadata or {})

    @property
    def cutoff(self) -> pd.Timestamp:
        return self.timestamps[-1]

    def append(self, observation: "Observation") -> "TimeSeries":
        timestamp = pd.Timestamp(observation.timestamp)
        if timestamp.tzinfo is None:
            raise ValidationError("observation.timestamp deve ter timezone")
        expected = self.cutoff + pd.tseries.frequencies.to_offset(self.frequency)
        if timestamp != expected:
            raise ValidationError(
                f"observação deve ocorrer em {expected.isoformat()}, não {timestamp.isoformat()}"
            )
        return replace(
            self,
            timestamps=self.timestamps.append(pd.DatetimeIndex([timestamp])),
            values=np.r_[self.values, float(observation.value)],
            revision=self.revision + 1,
        )


@dataclass(frozen=True)
class Observation:
    timestamp: pd.Timestamp | datetime | str
    value: float
    observation_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        timestamp = pd.Timestamp(self.timestamp)
        if timestamp.tzinfo is None or not self.observation_id or not np.isfinite(self.value):
            raise ValidationError("observação exige id, timestamp com timezone e valor finito")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "metadata", dict(self.metadata))


@runtime_checkable
class PredictiveDistribution(Protocol):
    family: str
    horizon: int

    @property
    def mean(self) -> FloatArray: ...

    @property
    def variance(self) -> FloatArray: ...

    def quantile(self, probabilities: ArrayLike) -> FloatArray: ...

    def cdf(self, values: ArrayLike) -> FloatArray: ...

    def sample(self, n_paths: int, *, seed: int | None = None) -> FloatArray: ...

    def to_payload(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class GaussianDistribution:
    """Normal multivariada; métodos de CDF/quantil são marginais."""

    location: FloatArray
    covariance: FloatArray
    family: ClassVar[str] = "gaussian"

    def __post_init__(self) -> None:
        location = _readonly_vector(self.location, name="location")
        covariance = _readonly_matrix(self.covariance, name="covariance")
        if covariance.shape != (location.size, location.size):
            raise ValidationError("covariance deve ter forma (horizon, horizon)")
        if np.any(np.diag(covariance) <= 0) or not np.allclose(covariance, covariance.T):
            raise ValidationError("covariance deve ser simétrica com diagonal positiva")
        if np.linalg.eigvalsh(covariance).min() < -1e-8:
            raise ValidationError("covariance deve ser semidefinida positiva")
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "covariance", covariance)

    @property
    def horizon(self) -> int:
        return int(self.location.size)

    @property
    def mean(self) -> FloatArray:
        return self.location

    @property
    def variance(self) -> FloatArray:
        result = np.diag(self.covariance).copy()
        result.setflags(write=False)
        return result

    def quantile(self, probabilities: ArrayLike) -> FloatArray:
        probabilities = np.asarray(probabilities, dtype=float)
        if np.any((probabilities <= 0) | (probabilities >= 1)):
            raise ValidationError("probabilidades devem estar em (0,1)")
        return self.mean[None, :] + norm.ppf(probabilities)[..., None] * np.sqrt(
            self.variance
        )[None, :]

    def cdf(self, values: ArrayLike) -> FloatArray:
        values = np.asarray(values, dtype=float)
        return norm.cdf(values, loc=self.mean, scale=np.sqrt(self.variance))

    def sample(self, n_paths: int, *, seed: int | None = None) -> FloatArray:
        if n_paths < 1:
            raise ValidationError("n_paths deve ser positivo")
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(self.mean, self.covariance, size=n_paths)

    def to_payload(self) -> Mapping[str, Any]:
        return {"family": self.family, "location": self.location, "covariance": self.covariance}


@dataclass(frozen=True)
class GaussianMixtureDistribution:
    """Mistura gaussiana marginal com cadeia de regimes opcional."""

    weights: FloatArray
    locations: FloatArray
    scales: FloatArray
    transition: FloatArray | None = None
    family: ClassVar[str] = "gaussian_mixture"

    def __post_init__(self) -> None:
        weights = _readonly_matrix(self.weights, name="weights")
        locations = _readonly_vector(self.locations, name="locations")
        scales = _readonly_vector(self.scales, name="scales")
        if weights.shape[1] != locations.size or scales.shape != locations.shape:
            raise ValidationError("dimensões incompatíveis na mistura")
        if np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1.0):
            raise ValidationError("cada linha de weights deve somar 1")
        if np.any(scales <= 0):
            raise ValidationError("scales deve ser positivo")
        transition = None
        if self.transition is not None:
            transition = _readonly_matrix(self.transition, name="transition")
            if transition.shape != (locations.size, locations.size):
                raise ValidationError("transition possui dimensão inválida")
            if np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1.0):
                raise ValidationError("transition deve ser estocástica por linha")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "locations", locations)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "transition", transition)

    @property
    def horizon(self) -> int:
        return int(self.weights.shape[0])

    @property
    def mean(self) -> FloatArray:
        return self.weights @ self.locations

    @property
    def variance(self) -> FloatArray:
        return self.weights @ (self.scales**2 + self.locations**2) - self.mean**2

    def cdf(self, values: ArrayLike) -> FloatArray:
        values = np.asarray(values, dtype=float)
        if values.ndim == 0:
            values = np.repeat(values, self.horizon)
        if values.shape != (self.horizon,):
            raise ValidationError("values deve ter um valor por horizonte")
        component_cdf = norm.cdf(
            values[:, None], loc=self.locations[None, :], scale=self.scales[None, :]
        )
        return np.sum(self.weights * component_cdf, axis=1)

    def quantile(self, probabilities: ArrayLike) -> FloatArray:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim == 0:
            probabilities = probabilities[None]
        if np.any((probabilities <= 0) | (probabilities >= 1)):
            raise ValidationError("probabilidades devem estar em (0,1)")
        lower = float(np.min(self.locations - 10 * self.scales))
        upper = float(np.max(self.locations + 10 * self.scales))
        result = np.empty((probabilities.size, self.horizon))
        for row, probability in enumerate(probabilities):
            lo = np.full(self.horizon, lower)
            hi = np.full(self.horizon, upper)
            for _ in range(70):
                middle = (lo + hi) / 2
                below = self.cdf(middle) < probability
                lo = np.where(below, middle, lo)
                hi = np.where(below, hi, middle)
            result[row] = (lo + hi) / 2
        return result

    def sample(self, n_paths: int, *, seed: int | None = None) -> FloatArray:
        if n_paths < 1:
            raise ValidationError("n_paths deve ser positivo")
        rng = np.random.default_rng(seed)
        states = np.empty((n_paths, self.horizon), dtype=int)
        states[:, 0] = [rng.choice(self.locations.size, p=self.weights[0]) for _ in range(n_paths)]
        for step in range(1, self.horizon):
            if self.transition is None:
                states[:, step] = [
                    rng.choice(self.locations.size, p=self.weights[step]) for _ in range(n_paths)
                ]
            else:
                states[:, step] = [
                    rng.choice(self.locations.size, p=self.transition[state])
                    for state in states[:, step - 1]
                ]
        return rng.normal(self.locations[states], self.scales[states])

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "family": self.family,
            "weights": self.weights,
            "locations": self.locations,
            "scales": self.scales,
            "transition": self.transition,
        }


@dataclass(frozen=True)
class EmpiricalDistribution:
    paths: FloatArray
    family: ClassVar[str] = "empirical"

    def __post_init__(self) -> None:
        paths = _readonly_matrix(self.paths, name="paths")
        if paths.shape[0] < 2 or paths.shape[1] < 1:
            raise ValidationError("paths deve conter ao menos duas trajetórias")
        object.__setattr__(self, "paths", paths)

    @property
    def horizon(self) -> int:
        return int(self.paths.shape[1])

    @property
    def mean(self) -> FloatArray:
        return np.mean(self.paths, axis=0)

    @property
    def variance(self) -> FloatArray:
        return np.var(self.paths, axis=0, ddof=1)

    def quantile(self, probabilities: ArrayLike) -> FloatArray:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim == 0:
            probabilities = probabilities[None]
        return np.quantile(self.paths, probabilities, axis=0)

    def cdf(self, values: ArrayLike) -> FloatArray:
        values = np.asarray(values, dtype=float)
        if values.ndim == 0:
            values = np.repeat(values, self.horizon)
        return np.mean(self.paths <= values[None, :], axis=0)

    def sample(self, n_paths: int, *, seed: int | None = None) -> FloatArray:
        if n_paths < 1:
            raise ValidationError("n_paths deve ser positivo")
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, self.paths.shape[0], size=n_paths)
        return self.paths[indices].copy()

    def to_payload(self) -> Mapping[str, Any]:
        return {"family": self.family, "paths": self.paths}


class ModelSpec(Protocol):
    family: str


@dataclass(frozen=True)
class AR1Spec:
    include_intercept: bool = True
    family: str = field(default="ar1", init=False)


@dataclass(frozen=True)
class ARIMASpec:
    order: tuple[int, int, int] = (1, 0, 0)
    trend: str | None = None
    simulation_paths: int = 1_000
    family: str = field(default="arima", init=False)

    def __post_init__(self) -> None:
        if len(self.order) != 3 or min(self.order) < 0 or self.simulation_paths < 100:
            raise ValidationError("ARIMA exige ordem (p,d,q) não negativa e >=100 simulações")


@dataclass(frozen=True)
class LocalLevelSpec:
    initial_variance: float = 10.0
    family: str = field(default="local_level", init=False)


@dataclass(frozen=True)
class GARCH11Spec:
    mean: str = "zero"
    simulation_paths: int = 2_000
    family: str = field(default="garch11", init=False)

    def __post_init__(self) -> None:
        if self.mean not in {"zero", "constant"} or self.simulation_paths < 100:
            raise ValidationError("GARCH mean deve ser zero/constant e simulation_paths >=100")


@dataclass(frozen=True)
class TwoRegimeGaussianSpec:
    max_iterations: int = 300
    family: str = field(default="markov_switching", init=False)


@dataclass(frozen=True)
class ModelCapabilities:
    probabilistic: bool = True
    simulation: bool = True
    cdf: bool = True
    missing_values: bool = False
    online_update: bool = False
    scenarios: tuple[str, ...] = ("baseline", "additive_shock")


@dataclass(frozen=True)
class ModelRun:
    model_run_id: str
    series_id: str
    series_revision: int
    cutoff: pd.Timestamp
    spec: ModelSpec
    adapter_version: str
    state: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class Forecast:
    forecast_id: str
    series_id: str
    model_run_id: str
    cutoff: pd.Timestamp
    targets: pd.DatetimeIndex
    distribution: PredictiveDistribution
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    evaluated_targets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        targets = pd.DatetimeIndex(self.targets)
        if targets.tz is None or len(targets) != self.distribution.horizon:
            raise ValidationError("targets com timezone devem coincidir com o horizonte")
        if not targets.is_monotonic_increasing or not targets.is_unique:
            raise ValidationError("targets devem ser crescentes e únicos")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class BacktestConfig:
    initial_train_size: int
    horizon: int = 1
    step: int = 1
    confidence: float = 0.95
    seed: int = 0

    def __post_init__(self) -> None:
        if self.initial_train_size < 4 or self.horizon < 1 or self.step < 1:
            raise ValidationError("configuração de backtest inválida")
        if not 0 < self.confidence < 1:
            raise ValidationError("confidence deve estar em (0,1)")


@dataclass(frozen=True)
class ModelScore:
    model_run_family: str
    metrics: Mapping[str, float]
    by_horizon: Mapping[int, Mapping[str, float]]


@dataclass(frozen=True)
class BacktestReport:
    report_id: str
    series_id: str
    config: BacktestConfig
    candidates: tuple[ModelSpec, ...]
    scores: tuple[ModelScore, ...]
    rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class SelectionPolicy:
    primary_metric: str = "crps"
    coverage_range: tuple[float, float] | None = (0.80, 1.0)


@dataclass(frozen=True)
class SelectionDecision:
    decision_id: str
    report_id: str
    series_id: str
    selected_spec: ModelSpec
    model_run_id: str
    primary_metric: str
    scores: Mapping[str, float]
    rationale: str


@dataclass(frozen=True)
class PromotionRecord:
    promotion_id: str
    series_id: str
    alias: str
    model_run_id: str
    decision_id: str | None
    promoted_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class PromotionPolicy:
    automatic: bool = False
    alias: str = "champion"


@dataclass(frozen=True)
class TrainingResult:
    report: BacktestReport
    decision: SelectionDecision
    model_run: ModelRun
    promotion: PromotionRecord | None


@dataclass(frozen=True)
class BaselineScenario:
    name: str = field(default="baseline", init=False)


@dataclass(frozen=True)
class AdditiveShockScenario:
    step: int
    magnitude: float
    duration: int = 1
    name: str = field(default="additive_shock", init=False)

    def __post_init__(self) -> None:
        if self.step < 1 or self.duration < 1 or not np.isfinite(self.magnitude):
            raise ValidationError("choque exige step/duration positivos e magnitude finita")


@dataclass(frozen=True)
class ParameterShiftScenario:
    start_step: int
    overrides: Mapping[str, float]
    name: str = field(default="parameter_shift", init=False)

    def __post_init__(self) -> None:
        if self.start_step < 1 or not self.overrides or any(
            not np.isfinite(value) for value in self.overrides.values()
        ):
            raise ValidationError("mudança de parâmetros inválida")


@dataclass(frozen=True)
class ForcedRegimeScenario:
    start_step: int
    regime: int
    name: str = field(default="forced_regime", init=False)

    def __post_init__(self) -> None:
        if self.start_step < 1 or self.regime < 0:
            raise ValidationError("regime forçado inválido")


SimulationScenario = BaselineScenario | AdditiveShockScenario | ParameterShiftScenario | ForcedRegimeScenario


@dataclass(frozen=True)
class SimulationResult:
    simulation_id: str
    series_id: str
    model_run_id: str
    targets: pd.DatetimeIndex
    paths: FloatArray
    scenario: SimulationScenario
    seed: int | None

    def __post_init__(self) -> None:
        paths = _readonly_matrix(self.paths, name="paths")
        if paths.shape[1] != len(self.targets):
            raise ValidationError("paths e targets incompatíveis")
        object.__setattr__(self, "paths", paths)


@dataclass(frozen=True)
class ThresholdRisk:
    limit: float
    direction: str = "above"
    n_paths: int = 10_000
    seed: int = 0

    def __post_init__(self) -> None:
        if self.direction not in {"above", "below"} or self.n_paths < 1:
            raise ValidationError("risco exige direction above/below e n_paths positivo")


@dataclass(frozen=True)
class RiskReport:
    marginal_probability: FloatArray
    any_horizon_probability: float
    first_passage_probability: FloatArray
    threshold: ThresholdRisk


@dataclass(frozen=True)
class MonitoringPolicy:
    point_threshold: float = 3.0
    cusum_reference: float = 0.4
    cusum_threshold: float = 6.0
    ewma_smoothing: float = 0.15
    ewma_limit_multiplier: float = 3.6
    cooldown: int = 12
    reset_on_alert: bool = True
    refit_each_observation: bool = True
    forecast_horizon: int = 1
    detectors: tuple[str, ...] = ("point_anomaly", "cusum", "ewma")

    def __post_init__(self) -> None:
        if (
            self.point_threshold <= 0
            or self.cusum_reference < 0
            or self.cusum_threshold <= 0
            or not 0 < self.ewma_smoothing <= 1
            or self.ewma_limit_multiplier <= 0
            or self.cooldown < 0
            or self.forecast_horizon < 1
            or not self.detectors
            or len(set(self.detectors)) != len(self.detectors)
        ):
            raise ValidationError("política de monitoramento inválida")


@dataclass(frozen=True)
class MonitoringSession:
    session_id: str
    series_id: str
    model_alias: str
    model_run_id: str
    policy: MonitoringPolicy
    sequence: int = 0
    cusum_positive: float = 0.0
    cusum_negative: float = 0.0
    ewma: float = 0.0
    last_emitted: int = -(10**12)
    detector_states: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    revision: int = 0


@dataclass(frozen=True)
class MonitoringEvent:
    event_id: str
    session_id: str
    series_id: str
    forecast_id: str | None
    observation_id: str
    sequence: int
    actual: float
    surprise: float | None
    cusum_positive: float
    cusum_negative: float
    ewma: float
    ewma_limit: float | None
    reasons: tuple[str, ...]
    emitted: bool
    kind: str
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class EngineEvent:
    event_id: str
    kind: str
    series_id: str
    payload: Mapping[str, Any]
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class CycleResult:
    observation: Observation
    monitoring_event: MonitoringEvent
    next_forecast: Forecast
    events: tuple[EngineEvent, ...]
    idempotent_replay: bool = False


@dataclass(frozen=True)
class ModelChoiceEvidence:
    signal: str
    value: float | str
    interpretation: str
    favors: tuple[str, ...] = ()


@dataclass(frozen=True)
class SeriesProfile:
    series_id: str
    nobs: int
    summary: Mapping[str, float]
    acf: tuple[float, ...]
    pacf: tuple[float, ...]
    adf_statistic: float
    adf_pvalue: float
    ljung_box_pvalue: float
    timestamps: pd.DatetimeIndex
    values: FloatArray
    trend: FloatArray
    detrended: FloatArray
    differences: FloatArray
    differenced_acf: tuple[float, ...]
    differenced_pacf: tuple[float, ...]
    squared_acf: tuple[float, ...]
    kpss_statistic: float
    kpss_pvalue: float
    jarque_bera_pvalue: float
    arch_lm_pvalue: float
    change_improvement_ratio: float
    evidence: tuple[ModelChoiceEvidence, ...]
    suggested_models: tuple[str, ...]

    def __post_init__(self) -> None:
        timestamps = pd.DatetimeIndex(self.timestamps)
        values = _readonly_vector(self.values, name="values")
        trend = _readonly_vector(self.trend, name="trend")
        detrended = _readonly_vector(self.detrended, name="detrended")
        differences = _readonly_vector(self.differences, name="differences")
        if not (len(timestamps) == values.size == trend.size == detrended.size):
            raise ValidationError("componentes da caracterização possuem tamanhos incompatíveis")
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "trend", trend)
        object.__setattr__(self, "detrended", detrended)
        object.__setattr__(self, "differences", differences)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "value": self.values,
                "trend": self.trend,
                "detrended": self.detrended,
            },
            index=self.timestamps,
        )


@dataclass(frozen=True)
class ResidualDiagnostics:
    series_id: str
    model_run_id: str
    model_family: str
    timestamps: pd.DatetimeIndex
    observed: FloatArray
    fitted_values: FloatArray
    residuals: FloatArray
    standardized_residuals: FloatArray
    conditional_variance: FloatArray | None
    acf: tuple[float, ...]
    pacf: tuple[float, ...]
    squared_acf: tuple[float, ...]
    qq_theoretical: FloatArray
    qq_sample: FloatArray
    tests: Mapping[str, float]
    interpretation: tuple[str, ...]

    def __post_init__(self) -> None:
        timestamps = pd.DatetimeIndex(self.timestamps)
        observed = _readonly_vector(self.observed, name="observed")
        fitted = _readonly_vector(self.fitted_values, name="fitted_values")
        residuals = _readonly_vector(self.residuals, name="residuals")
        standardized = _readonly_vector(
            self.standardized_residuals, name="standardized_residuals"
        )
        size = observed.size
        if not (len(timestamps) == size == fitted.size == residuals.size == standardized.size):
            raise ValidationError("diagnóstico residual possui tamanhos incompatíveis")
        variance = None
        if self.conditional_variance is not None:
            variance = _readonly_vector(
                self.conditional_variance, name="conditional_variance"
            )
            if variance.size != size or np.any(variance <= 0):
                raise ValidationError("variância condicional incompatível")
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "fitted_values", fitted)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "standardized_residuals", standardized)
        object.__setattr__(self, "conditional_variance", variance)
        object.__setattr__(
            self, "qq_theoretical", _readonly_vector(self.qq_theoretical, name="qq_theoretical")
        )
        object.__setattr__(self, "qq_sample", _readonly_vector(self.qq_sample, name="qq_sample"))

    def to_frame(self) -> pd.DataFrame:
        payload: dict[str, Any] = {
            "observed": self.observed,
            "fitted": self.fitted_values,
            "residual": self.residuals,
            "standardized_residual": self.standardized_residuals,
        }
        if self.conditional_variance is not None:
            payload["conditional_variance"] = self.conditional_variance
        return pd.DataFrame(payload, index=self.timestamps)


@dataclass(frozen=True)
class ChangePointReport:
    series_id: str
    breakpoints: tuple[pd.Timestamp, ...]
    objective: float
    fitted_values: FloatArray


@dataclass(frozen=True)
class EngineStatus:
    series_id: str
    series_revision: int
    cutoff: pd.Timestamp
    aliases: Mapping[str, str]
    monitoring_sessions: tuple[str, ...]


@dataclass(frozen=True)
class EngineConfig:
    default_alias: str = "champion"
    default_seed: int = 0
    pit_clip: float = 1e-6

    def __post_init__(self) -> None:
        if not self.default_alias or not 0 < self.pit_clip < 0.5:
            raise ValidationError("configuração do motor inválida")


@dataclass(frozen=True)
class EngineCapabilities:
    model_families: tuple[str, ...]
    detector_families: tuple[str, ...] = ("point_anomaly", "cusum", "ewma", "change_point")
    multivariate: bool = False
    asynchronous: bool = False


@runtime_checkable
class ModelAdapter(Protocol):
    family: str
    version: str
    capabilities: ModelCapabilities

    def fit(self, series: TimeSeries, spec: ModelSpec) -> tuple[Mapping[str, Any], Mapping[str, Any]]: ...

    def forecast(self, run: ModelRun, horizon: int, *, seed: int) -> PredictiveDistribution: ...

    def simulate(
        self,
        run: ModelRun,
        horizon: int,
        n_paths: int,
        *,
        scenario: SimulationScenario,
        seed: int | None,
    ) -> FloatArray: ...


@runtime_checkable
class Detector(Protocol):
    name: str

    def update(self, value: float) -> tuple[Mapping[str, float], tuple[str, ...]]: ...

    def snapshot(self) -> Mapping[str, Any]: ...

    def restore(self, state: Mapping[str, Any]) -> None: ...

    def reset(self) -> None: ...


@runtime_checkable
class EngineRepository(Protocol):
    """Porta de persistência. A aplicação deve garantir atomicidade por série."""

    def add_series(self, series: TimeSeries) -> None: ...

    def get_series(self, series_id: str) -> TimeSeries: ...

    def list_series(self) -> Sequence[TimeSeries]: ...

    def update_series(self, series: TimeSeries, *, expected_revision: int) -> None: ...

    def save_model_run(self, run: ModelRun) -> None: ...

    def get_model_run(self, model_run_id: str) -> ModelRun: ...

    def set_alias(self, series_id: str, alias: str, model_run_id: str) -> None: ...

    def get_alias(self, series_id: str, alias: str) -> str | None: ...

    def list_aliases(self, series_id: str) -> Mapping[str, str]: ...

    def save_forecast(self, forecast: Forecast) -> None: ...

    def get_forecast(self, forecast_id: str) -> Forecast: ...

    def find_pending_forecast(
        self, series_id: str, model_run_id: str, target: pd.Timestamp
    ) -> Forecast | None: ...

    def mark_forecast_evaluated(self, forecast_id: str, target: pd.Timestamp) -> None: ...

    def save_monitoring_session(
        self, session: MonitoringSession, *, expected_revision: int | None = None
    ) -> None: ...

    def get_monitoring_session(self, session_id: str) -> MonitoringSession: ...

    def list_monitoring_sessions(self, series_id: str) -> Sequence[MonitoringSession]: ...

    def save_monitoring_event(self, event: MonitoringEvent) -> None: ...

    def find_cycle_result(self, observation_id: str) -> CycleResult | None: ...

    def save_cycle_result(self, result: CycleResult) -> None: ...

    def save_event(self, event: EngineEvent) -> None: ...

    def list_events(self, series_id: str) -> Sequence[EngineEvent]: ...


class ModelRegistry:
    def __init__(self, adapters: Sequence[ModelAdapter] = ()) -> None:
        self._adapters: dict[str, ModelAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: ModelAdapter) -> None:
        if adapter.family in self._adapters:
            raise ValidationError(f"adaptador já registrado: {adapter.family}")
        self._adapters[adapter.family] = adapter

    def get(self, family: str) -> ModelAdapter:
        try:
            return self._adapters[family]
        except KeyError as error:
            raise UnsupportedCapabilityError(f"modelo não registrado: {family}") from error

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


class DetectorRegistry:
    def __init__(
        self,
        factories: Mapping[str, Callable[[MonitoringPolicy], Detector]] | None = None,
    ) -> None:
        self._factories = dict(factories or {})

    def register(
        self, name: str, factory: Callable[[MonitoringPolicy], Detector]
    ) -> None:
        if name in self._factories:
            raise ValidationError(f"detector já registrado: {name}")
        self._factories[name] = factory

    def create(self, name: str, policy: MonitoringPolicy) -> Detector:
        try:
            return self._factories[name](policy)
        except KeyError as error:
            raise UnsupportedCapabilityError(f"detector não registrado: {name}") from error

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


__all__ = [name for name in globals() if not name.startswith("_")]
