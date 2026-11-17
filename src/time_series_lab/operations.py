"""Artefatos imutáveis, monitoramento prequential e risco operacional."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class ForecastEnvelope:
    model_name: str
    cutoff: int
    horizons: tuple[int, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    metadata: dict[str, Any]
    forecast_id: str = field(compare=True)

    @classmethod
    def create(
        cls,
        *,
        model_name: str,
        cutoff: int,
        horizons: ArrayLike,
        mean: ArrayLike,
        std: ArrayLike,
        metadata: dict[str, Any] | None = None,
    ) -> "ForecastEnvelope":
        horizon_values = tuple(int(value) for value in np.asarray(horizons).tolist())
        mean_values = tuple(float(value) for value in np.asarray(mean).tolist())
        std_values = tuple(float(value) for value in np.asarray(std).tolist())
        if not model_name or cutoff < 0 or not horizon_values:
            raise ValueError("modelo, cutoff e horizontes válidos são necessários")
        if len(horizon_values) != len(mean_values) or len(mean_values) != len(std_values):
            raise ValueError("horizons, mean e std devem ter o mesmo tamanho")
        if tuple(sorted(horizon_values)) != horizon_values or len(set(horizon_values)) != len(horizon_values):
            raise ValueError("horizontes devem ser crescentes e únicos")
        if any(value <= 0 for value in horizon_values) or any(value <= 0 for value in std_values):
            raise ValueError("horizontes e desvios devem ser positivos")
        payload = {
            "model_name": model_name,
            "cutoff": cutoff,
            "horizons": horizon_values,
            "mean": mean_values,
            "std": std_values,
            "metadata": metadata or {},
        }
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:20]
        return cls(**payload, forecast_id=digest)

    def verify(self) -> bool:
        recreated = ForecastEnvelope.create(
            model_name=self.model_name,
            cutoff=self.cutoff,
            horizons=self.horizons,
            mean=self.mean,
            std=self.std,
            metadata=self.metadata,
        )
        return recreated.forecast_id == self.forecast_id


class LocalArtifactStore:
    """Armazena cada forecast uma única vez, identificado por hash do conteúdo."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, envelope: ForecastEnvelope) -> Path:
        if not envelope.verify():
            raise ValueError("forecast_id não corresponde ao conteúdo")
        destination = self.directory / f"{envelope.forecast_id}.json"
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(asdict(envelope), handle, ensure_ascii=False, sort_keys=True, indent=2)
        return destination

    def load(self, forecast_id: str) -> ForecastEnvelope:
        payload = json.loads((self.directory / f"{forecast_id}.json").read_text(encoding="utf-8"))
        payload["horizons"] = tuple(payload["horizons"])
        payload["mean"] = tuple(payload["mean"])
        payload["std"] = tuple(payload["std"])
        envelope = ForecastEnvelope(**payload)
        if not envelope.verify():
            raise ValueError("artefato corrompido ou alterado")
        return envelope


@dataclass(frozen=True)
class AlertPolicy:
    point_threshold: float = 3.0
    cusum_reference: float = 0.4
    cusum_threshold: float = 6.0
    ewma_smoothing: float = 0.15
    ewma_limit_multiplier: float = 3.6
    cooldown: int = 12
    reset_on_alert: bool = True

    def __post_init__(self) -> None:
        if (
            self.point_threshold <= 0
            or self.cusum_reference < 0
            or self.cusum_threshold <= 0
            or not 0 < self.ewma_smoothing <= 1
            or self.ewma_limit_multiplier <= 0
            or self.cooldown < 0
        ):
            raise ValueError("política inválida")


@dataclass(frozen=True)
class MonitoringEvent:
    sequence: int
    forecast_id: str
    actual: float
    surprise: float
    cusum_positive: float
    cusum_negative: float
    ewma: float
    ewma_limit: float
    reasons: tuple[str, ...]
    emitted: bool


class SequentialMonitor:
    """Atualiza estatísticas somente depois de receber forecast preservado e realizado."""

    def __init__(self, policy: AlertPolicy):
        self.policy = policy
        self.sequence = 0
        self.cusum_positive = 0.0
        self.cusum_negative = 0.0
        self.ewma = 0.0
        self.last_emitted = -10**12

    def update(
        self,
        *,
        forecast_id: str,
        actual: float,
        mean: float,
        std: float,
    ) -> MonitoringEvent:
        if not forecast_id or std <= 0 or not np.all(np.isfinite([actual, mean, std])):
            raise ValueError("forecast e realizado inválidos")
        self.sequence += 1
        surprise = float((actual - mean) / std)
        self.cusum_positive = max(
            0.0, self.cusum_positive + surprise - self.policy.cusum_reference
        )
        self.cusum_negative = min(
            0.0, self.cusum_negative + surprise + self.policy.cusum_reference
        )
        self.ewma = (
            self.policy.ewma_smoothing * surprise
            + (1.0 - self.policy.ewma_smoothing) * self.ewma
        )
        variance = self.policy.ewma_smoothing / (2.0 - self.policy.ewma_smoothing) * (
            1.0 - (1.0 - self.policy.ewma_smoothing) ** (2 * self.sequence)
        )
        ewma_limit = self.policy.ewma_limit_multiplier * np.sqrt(variance)

        reasons: list[str] = []
        if abs(surprise) >= self.policy.point_threshold:
            reasons.append("point_anomaly")
        if self.cusum_positive >= self.policy.cusum_threshold:
            reasons.append("cusum_positive")
        if self.cusum_negative <= -self.policy.cusum_threshold:
            reasons.append("cusum_negative")
        if abs(self.ewma) >= ewma_limit:
            reasons.append("ewma_drift")

        emitted = bool(reasons) and self.sequence - self.last_emitted > self.policy.cooldown
        event = MonitoringEvent(
            sequence=self.sequence,
            forecast_id=forecast_id,
            actual=float(actual),
            surprise=surprise,
            cusum_positive=float(self.cusum_positive),
            cusum_negative=float(self.cusum_negative),
            ewma=float(self.ewma),
            ewma_limit=float(ewma_limit),
            reasons=tuple(reasons),
            emitted=emitted,
        )
        if emitted:
            self.last_emitted = self.sequence
            if self.policy.reset_on_alert:
                self.cusum_positive = 0.0
                self.cusum_negative = 0.0
                self.ewma = 0.0
        return event


def gaussian_exceedance_probability(mean: ArrayLike, std: ArrayLike, threshold: float) -> np.ndarray:
    """Calcula P(Y_h>threshold) para marginais gaussianas."""

    location = np.asarray(mean, dtype=float)
    scale = np.asarray(std, dtype=float)
    if location.shape != scale.shape or np.any(scale <= 0):
        raise ValueError("mean e std compatíveis e std positivo são necessários")
    return 1.0 - norm.cdf((threshold - location) / scale)


def path_exceedance_probability(paths: ArrayLike, threshold: float) -> float:
    """Estima P(max_h Y_h>threshold) a partir de trajetórias conjuntas."""

    values = np.asarray(paths, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("paths deve ter forma (n_paths, horizon)")
    return float(np.mean(np.max(values, axis=1) > threshold))
