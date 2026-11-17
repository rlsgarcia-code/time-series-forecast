"""Repositório didático em memória para exemplos e notebooks.

Uma aplicação real deve implementar ``EngineRepository`` com as garantias de
durabilidade e transação apropriadas ao seu ambiente.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

import pandas as pd

from time_series_lab.contracts import (
    CycleResult,
    EngineEvent,
    Forecast,
    ModelRun,
    MonitoringEvent,
    MonitoringSession,
    RepositoryConflictError,
    TimeSeries,
)


class MemoryRepository:
    def __init__(self) -> None:
        self.series: dict[str, TimeSeries] = {}
        self.runs: dict[str, ModelRun] = {}
        self.aliases: dict[str, dict[str, str]] = {}
        self.forecasts: dict[str, Forecast] = {}
        self.sessions: dict[str, MonitoringSession] = {}
        self.monitoring_events: list[MonitoringEvent] = []
        self.cycles: dict[str, CycleResult] = {}
        self.events: list[EngineEvent] = []

    def add_series(self, series: TimeSeries) -> None:
        if series.series_id in self.series:
            raise RepositoryConflictError("série já existe")
        self.series[series.series_id] = series

    def get_series(self, series_id: str) -> TimeSeries:
        return self.series[series_id]

    def list_series(self) -> Sequence[TimeSeries]:
        return tuple(self.series.values())

    def update_series(self, series: TimeSeries, *, expected_revision: int) -> None:
        if self.series[series.series_id].revision != expected_revision:
            raise RepositoryConflictError("revision conflitante")
        self.series[series.series_id] = series

    def save_model_run(self, run: ModelRun) -> None:
        self.runs.setdefault(run.model_run_id, run)

    def get_model_run(self, model_run_id: str) -> ModelRun:
        return self.runs[model_run_id]

    def set_alias(self, series_id: str, alias: str, model_run_id: str) -> None:
        self.aliases.setdefault(series_id, {})[alias] = model_run_id

    def get_alias(self, series_id: str, alias: str) -> str | None:
        return self.aliases.get(series_id, {}).get(alias)

    def list_aliases(self, series_id: str) -> Mapping[str, str]:
        return dict(self.aliases.get(series_id, {}))

    def save_forecast(self, forecast: Forecast) -> None:
        self.forecasts.setdefault(forecast.forecast_id, forecast)

    def get_forecast(self, forecast_id: str) -> Forecast:
        return self.forecasts[forecast_id]

    def find_pending_forecast(
        self, series_id: str, model_run_id: str, target: pd.Timestamp
    ) -> Forecast | None:
        target = pd.Timestamp(target)
        target_key = target.isoformat()
        matches = [
            forecast
            for forecast in self.forecasts.values()
            if forecast.series_id == series_id
            and forecast.model_run_id == model_run_id
            and target in forecast.targets
            and target_key not in forecast.evaluated_targets
        ]
        return matches[-1] if matches else None

    def mark_forecast_evaluated(self, forecast_id: str, target: pd.Timestamp) -> None:
        forecast = self.forecasts[forecast_id]
        target_key = pd.Timestamp(target).isoformat()
        self.forecasts[forecast_id] = replace(
            forecast, evaluated_targets=forecast.evaluated_targets + (target_key,)
        )

    def save_monitoring_session(
        self, session: MonitoringSession, *, expected_revision: int | None = None
    ) -> None:
        previous = self.sessions.get(session.session_id)
        if expected_revision is not None and (
            previous is None or previous.revision != expected_revision
        ):
            raise RepositoryConflictError("sessão conflitante")
        self.sessions[session.session_id] = session

    def get_monitoring_session(self, session_id: str) -> MonitoringSession:
        return self.sessions[session_id]

    def list_monitoring_sessions(self, series_id: str) -> Sequence[MonitoringSession]:
        return tuple(
            session for session in self.sessions.values() if session.series_id == series_id
        )

    def save_monitoring_event(self, event: MonitoringEvent) -> None:
        self.monitoring_events.append(event)

    def find_cycle_result(self, observation_id: str) -> CycleResult | None:
        return self.cycles.get(observation_id)

    def save_cycle_result(self, result: CycleResult) -> None:
        self.cycles[result.observation.observation_id] = result

    def save_event(self, event: EngineEvent) -> None:
        self.events.append(event)

    def list_events(self, series_id: str) -> Sequence[EngineEvent]:
        return tuple(event for event in self.events if event.series_id == series_id)


__all__ = ["MemoryRepository"]
