"""Contratos e artefatos mínimos do sistema integrado."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SeriesContract:
    frequency: str
    timezone: str
    horizon: int
    minimum_history: int

    def validate(self, values: ArrayLike) -> FloatArray:
        series = np.asarray(values, dtype=float)
        if series.ndim != 1 or series.size < self.minimum_history:
            raise ValueError("histórico insuficiente ou formato inválido")
        if not np.all(np.isfinite(series)) or self.horizon < 1:
            raise ValueError("série deve ser finita e horizonte positivo")
        return series


@dataclass(frozen=True)
class ForecastArtifact:
    model_name: str
    cutoff: int
    horizon: list[int]
    mean: list[float]
    lower: list[float]
    upper: list[float]
    metadata: dict[str, Any]

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "ForecastArtifact":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def standardized_surprise(actual: float, mean: float, std: float) -> float:
    """Erro prequential padronizado usado em monitoramento."""

    if std <= 0:
        raise ValueError("std deve ser positivo")
    return float((actual - mean) / std)

