"""Detectores sequenciais restauráveis usados pelo motor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .contracts import MonitoringPolicy, ValidationError


@dataclass
class PointAnomalyDetector:
    threshold: float
    name: str = "point_anomaly"

    def update(self, value: float) -> tuple[Mapping[str, float], tuple[str, ...]]:
        reasons = (self.name,) if abs(value) >= self.threshold else ()
        return {"value": float(value)}, reasons

    def snapshot(self) -> Mapping[str, Any]:
        return {}

    def restore(self, state: Mapping[str, Any]) -> None:
        return None

    def reset(self) -> None:
        return None


@dataclass
class CusumDetector:
    reference: float
    threshold: float
    positive: float = 0.0
    negative: float = 0.0
    name: str = "cusum"

    def update(self, value: float) -> tuple[Mapping[str, float], tuple[str, ...]]:
        self.positive = max(0.0, self.positive + value - self.reference)
        self.negative = min(0.0, self.negative + value + self.reference)
        reasons: list[str] = []
        if self.positive >= self.threshold:
            reasons.append("cusum_positive")
        if self.negative <= -self.threshold:
            reasons.append("cusum_negative")
        return {
            "positive": float(self.positive),
            "negative": float(self.negative),
        }, tuple(reasons)

    def snapshot(self) -> Mapping[str, Any]:
        return {"positive": float(self.positive), "negative": float(self.negative)}

    def restore(self, state: Mapping[str, Any]) -> None:
        self.positive = float(state.get("positive", 0.0))
        self.negative = float(state.get("negative", 0.0))

    def reset(self) -> None:
        self.positive = 0.0
        self.negative = 0.0


@dataclass
class EwmaDetector:
    smoothing: float
    limit_multiplier: float
    statistic: float = 0.0
    sequence: int = 0
    name: str = "ewma"

    def update(self, value: float) -> tuple[Mapping[str, float], tuple[str, ...]]:
        self.sequence += 1
        self.statistic = (
            self.smoothing * value + (1.0 - self.smoothing) * self.statistic
        )
        variance = self.smoothing / (2.0 - self.smoothing) * (
            1.0 - (1.0 - self.smoothing) ** (2 * self.sequence)
        )
        limit = self.limit_multiplier * np.sqrt(variance)
        reasons = ("ewma_drift",) if abs(self.statistic) >= limit else ()
        return {
            "statistic": float(self.statistic),
            "limit": float(limit),
            "sequence": float(self.sequence),
        }, reasons

    def snapshot(self) -> Mapping[str, Any]:
        return {"statistic": float(self.statistic), "sequence": self.sequence}

    def restore(self, state: Mapping[str, Any]) -> None:
        self.statistic = float(state.get("statistic", 0.0))
        self.sequence = int(state.get("sequence", 0))

    def reset(self) -> None:
        self.statistic = 0.0
        self.sequence = 0


def point_anomaly_factory(policy: MonitoringPolicy) -> PointAnomalyDetector:
    return PointAnomalyDetector(policy.point_threshold)


def cusum_factory(policy: MonitoringPolicy) -> CusumDetector:
    return CusumDetector(policy.cusum_reference, policy.cusum_threshold)


def ewma_factory(policy: MonitoringPolicy) -> EwmaDetector:
    return EwmaDetector(policy.ewma_smoothing, policy.ewma_limit_multiplier)


def built_in_detector_factories():
    return {
        "point_anomaly": point_anomaly_factory,
        "cusum": cusum_factory,
        "ewma": ewma_factory,
    }


__all__ = [
    "CusumDetector",
    "EwmaDetector",
    "PointAnomalyDetector",
    "built_in_detector_factories",
]
