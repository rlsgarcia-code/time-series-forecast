from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from time_series_lab import (
    AR1Spec,
    ARIMASpec,
    BacktestConfig,
    ForecastEngine,
    GARCH11Spec,
    GaussianDistribution,
    GaussianMixtureDistribution,
    LocalLevelSpec,
    MonitoringPolicy,
    NotFittedError,
    Observation,
    PromotionPolicy,
    RepositoryConflictError,
    SelectionPolicy,
    ThresholdRisk,
    TimeSeries,
    TwoRegimeGaussianSpec,
    ValidationError,
)
from time_series_lab.detectors import CusumDetector, EwmaDetector
from time_series_lab.contracts import (
    CycleResult,
    EngineEvent,
    Forecast,
    ModelRun,
    MonitoringEvent,
    MonitoringSession,
)


class MemoryRepository:
    """Implementação de teste do protocolo; não faz parte da biblioteca pública."""

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
        target_key = pd.Timestamp(target).isoformat()
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
        self.forecasts[forecast_id] = replace(
            forecast,
            evaluated_targets=forecast.evaluated_targets + (pd.Timestamp(target).isoformat(),),
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


def make_series(series_id: str = "x", n: int = 160, seed: int = 5) -> TimeSeries:
    rng = np.random.default_rng(seed)
    values = np.empty(n)
    values[0] = 0.0
    for index in range(1, n):
        values[index] = 0.2 + 0.7 * values[index - 1] + rng.normal(0, 0.7)
    return TimeSeries(
        series_id,
        pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC"),
        values,
        "D",
    )


def configured_engine() -> tuple[ForecastEngine, MemoryRepository]:
    repository = MemoryRepository()
    engine = ForecastEngine(repository=repository)
    engine.register_series(make_series())
    return engine, repository


def test_time_series_requires_regular_timezone_aware_index() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        TimeSeries("x", pd.date_range("2025-01-01", periods=4), np.arange(4), "D")
    with pytest.raises(ValidationError, match="gaps"):
        TimeSeries(
            "x",
            pd.DatetimeIndex(["2025-01-01", "2025-01-03"]).tz_localize("UTC"),
            [1, 2],
            "D",
        )


def test_generic_distributions_expose_moments_cdf_quantiles_and_joint_samples() -> None:
    gaussian = GaussianDistribution([0.0, 1.0], [[1.0, 0.7], [0.7, 2.0]])
    assert gaussian.horizon == 2
    np.testing.assert_allclose(gaussian.cdf([0.0, 1.0]), [0.5, 0.5])
    assert gaussian.quantile([0.025, 0.975]).shape == (2, 2)
    assert np.corrcoef(gaussian.sample(20_000, seed=10).T)[0, 1] > 0.4

    mixture = GaussianMixtureDistribution(
        [[0.8, 0.2], [0.6, 0.4]],
        [-2.0, 3.0],
        [0.5, 1.0],
        [[0.9, 0.1], [0.2, 0.8]],
    )
    assert mixture.quantile([0.5]).shape == (1, 2)
    assert mixture.sample(100, seed=3).shape == (100, 2)
    assert np.all((mixture.cdf([0.0, 0.0]) > 0) & (mixture.cdf([0.0, 0.0]) < 1))


def test_detectors_snapshot_restore_and_reset_state() -> None:
    cusum = CusumDetector(reference=0.2, threshold=2.0)
    for _ in range(3):
        cusum.update(1.0)
    restored = CusumDetector(reference=0.2, threshold=2.0)
    restored.restore(cusum.snapshot())
    assert restored.snapshot() == cusum.snapshot()
    restored.reset()
    assert restored.snapshot() == {"positive": 0.0, "negative": 0.0}

    ewma = EwmaDetector(smoothing=0.2, limit_multiplier=3.0)
    ewma.update(1.0)
    restored_ewma = EwmaDetector(smoothing=0.2, limit_multiplier=3.0)
    restored_ewma.restore(ewma.snapshot())
    assert restored_ewma.snapshot() == ewma.snapshot()


@pytest.mark.parametrize(
    "spec,min_history",
    [
        (AR1Spec(), 80),
        (ARIMASpec(order=(1, 0, 1), simulation_paths=200), 80),
        (LocalLevelSpec(), 80),
        (GARCH11Spec(simulation_paths=200), 120),
        (TwoRegimeGaussianSpec(max_iterations=100), 120),
    ],
)
def test_every_builtin_adapter_fits_forecasts_and_simulates(spec, min_history) -> None:
    repository = MemoryRepository()
    engine = ForecastEngine(repository=repository)
    engine.register_series(make_series(n=min_history))
    run = engine.fit("x", spec)
    engine.promote("x", run.model_run_id)
    forecast = engine.forecast("x", 3, seed=4)
    simulation = engine.simulate("x", 3, 50, seed=4)
    diagnostics = engine.diagnose("x")
    assert forecast.distribution.horizon == 3
    assert simulation.paths.shape == (50, 3)
    assert np.all(np.isfinite(forecast.distribution.mean))
    assert diagnostics.residuals.size >= 12
    assert diagnostics.to_frame().index.is_monotonic_increasing
    assert set(diagnostics.tests) >= {
        "ljung_box_pvalue",
        "squared_ljung_box_pvalue",
        "jarque_bera_pvalue",
        "arch_lm_pvalue",
    }


def test_training_selection_promotion_forecast_simulation_and_risk() -> None:
    engine, repository = configured_engine()
    result = engine.run_training_cycle(
        "x",
        [AR1Spec(), LocalLevelSpec()],
        BacktestConfig(initial_train_size=100, horizon=2, step=20),
        selection=SelectionPolicy(coverage_range=None),
        promotion=PromotionPolicy(automatic=True),
    )
    assert result.promotion is not None
    assert engine.get_champion("x").model_run_id == result.model_run.model_run_id
    forecast = engine.forecast("x", 5)
    simulation = engine.simulate("x", 5, 500, seed=11)
    forecast_risk = engine.assess_risk(
        forecast, ThresholdRisk(limit=2.0, n_paths=1_000, seed=12)
    )
    simulation_risk = engine.assess_risk(simulation, ThresholdRisk(limit=2.0))
    assert forecast.cutoff < forecast.targets[0]
    assert 0 <= forecast_risk.any_horizon_probability <= 1
    assert 0 <= simulation_risk.any_horizon_probability <= 1
    assert forecast.forecast_id in repository.forecasts


def test_one_engine_keeps_multiple_series_isolated() -> None:
    repository = MemoryRepository()
    engine = ForecastEngine(repository=repository)
    engine.register_series(make_series("a", seed=1))
    engine.register_series(make_series("b", seed=2))
    for series_id in ("a", "b"):
        run = engine.fit(series_id, AR1Spec())
        engine.promote(series_id, run.model_run_id)
    assert engine.get_champion("a").series_id == "a"
    assert engine.get_champion("b").series_id == "b"
    assert engine.forecast("a", 1).series_id == "a"
    assert engine.get_status("b").series_revision == 0


def test_forecast_rejects_model_fitted_before_latest_series_revision() -> None:
    engine, _ = configured_engine()
    run = engine.fit("x", AR1Spec())
    engine.promote("x", run.model_run_id)
    series = engine.get_series("x")
    engine.append_history(
        "x",
        [Observation(series.cutoff + pd.Timedelta(days=1), 0.0, "history-1")],
    )
    with pytest.raises(NotFittedError, match="obsoleto"):
        engine.forecast("x", 1)


def test_monitoring_scores_preserved_forecast_before_append_and_is_idempotent() -> None:
    engine, repository = configured_engine()
    run = engine.fit("x", AR1Spec())
    engine.promote("x", run.model_run_id)
    session = engine.start_monitoring(
        "x",
        MonitoringPolicy(
            point_threshold=1.0,
            cooldown=0,
            reset_on_alert=False,
            refit_each_observation=True,
        ),
    )
    before = repository.get_series("x")
    observation = Observation(
        before.cutoff + pd.Timedelta(days=1),
        6.0,
        "observation-1",
    )
    result = engine.process_observation(session.session_id, observation)
    assert result.monitoring_event.forecast_id is not None
    assert result.monitoring_event.surprise is not None
    assert repository.get_series("x").revision == before.revision + 1
    assert result.next_forecast.cutoff == observation.timestamp
    replay = engine.process_observation(session.session_id, observation)
    assert replay.idempotent_replay
    with pytest.raises(RepositoryConflictError):
        engine.process_observation(
            session.session_id,
            Observation(observation.timestamp, 7.0, observation.observation_id),
        )


def test_characterization_change_points_and_status_queries() -> None:
    engine, _ = configured_engine()
    profile = engine.characterize("x", max_lag=8)
    change = engine.detect_changes("x", penalty=12.0, min_size=20)
    assert profile.nobs == 160
    assert len(profile.acf) == 9
    assert len(profile.pacf) == 9
    assert profile.differences.shape == (159,)
    assert profile.to_frame().shape == (160, 3)
    assert profile.evidence
    assert profile.suggested_models
    assert change.fitted_values.shape == (160,)
    assert engine.get_status("x").aliases == {}


def test_characterization_and_residual_plots_are_returned_without_showing() -> None:
    engine, _ = configured_engine()
    run = engine.fit("x", AR1Spec())
    engine.promote("x", run.model_run_id)
    characterization_figure = engine.plot_characterization("x", max_lag=8)
    residual_figure = engine.plot_residuals("x", max_lag=8)
    assert len(characterization_figure.axes) == 6
    assert len(residual_figure.axes) == 6
    plt.close(characterization_figure)
    plt.close(residual_figure)
