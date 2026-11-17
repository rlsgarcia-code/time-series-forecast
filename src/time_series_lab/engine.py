"""Fachada de aplicação para forecasting, simulação e monitoramento."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .adapters import built_in_adapters
from .change_points import optimal_mean_segmentation
from .detectors import built_in_detector_factories
from .diagnostics import (
    characterize_series,
    diagnose_residuals,
    plot_residual_diagnostics,
    plot_series_characterization,
)
from .contracts import (
    BacktestConfig,
    BacktestReport,
    BaselineScenario,
    ChangePointReport,
    CycleResult,
    DetectorRegistry,
    EngineCapabilities,
    EngineConfig,
    EngineEvent,
    EngineRepository,
    EngineStatus,
    Forecast,
    ModelRegistry,
    ModelRun,
    ModelScore,
    ModelSpec,
    MonitoringEvent,
    MonitoringPolicy,
    MonitoringSession,
    NotFittedError,
    Observation,
    PromotionPolicy,
    PromotionRecord,
    RepositoryConflictError,
    ResidualDiagnostics,
    RiskReport,
    SelectionDecision,
    SelectionPolicy,
    SeriesProfile,
    SimulationResult,
    SimulationScenario,
    ThresholdRisk,
    TimeSeries,
    TrainingResult,
    ValidationError,
    content_id,
    utc_now,
)


def _candidate_id(spec: ModelSpec) -> str:
    return content_id("candidate", spec)


def _sample_crps(distribution, actual: np.ndarray, *, seed: int) -> np.ndarray:
    first = distribution.sample(384, seed=seed)
    second = distribution.sample(384, seed=seed + 1)
    return np.mean(np.abs(first - actual[None, :]), axis=0) - 0.5 * np.mean(
        np.abs(first - second), axis=0
    )


class ForecastEngine:
    """Coordena componentes mantendo persistência e efeitos sob controle da aplicação."""

    def __init__(
        self,
        *,
        repository: EngineRepository,
        config: EngineConfig | None = None,
        models: ModelRegistry | None = None,
        detectors: DetectorRegistry | None = None,
    ) -> None:
        self.repository = repository
        self.config = config or EngineConfig()
        self.models = models or ModelRegistry(built_in_adapters())
        self.detectors = detectors or DetectorRegistry(built_in_detector_factories())
        self.capabilities = EngineCapabilities(
            model_families=self.models.families,
            detector_families=self.detectors.families + ("change_point",),
        )

    # ------------------------------------------------------------------ dados
    def register_series(self, series: TimeSeries) -> TimeSeries:
        self.repository.add_series(series)
        self._event("series_registered", series.series_id, {"revision": series.revision})
        return series

    def append_history(
        self, series_id: str, observations: Sequence[Observation]
    ) -> TimeSeries:
        if self.repository.list_monitoring_sessions(series_id):
            raise ValidationError(
                "append_history não pode contornar uma sessão de monitoramento ativa"
            )
        series = self.repository.get_series(series_id)
        expected_revision = series.revision
        for observation in observations:
            series = series.append(observation)
        self.repository.update_series(series, expected_revision=expected_revision)
        self._event(
            "history_appended",
            series_id,
            {"count": len(observations), "revision": series.revision},
        )
        return series

    def get_series(self, series_id: str) -> TimeSeries:
        return self.repository.get_series(series_id)

    def list_series(self) -> tuple[TimeSeries, ...]:
        return tuple(self.repository.list_series())

    def characterize(self, series_id: str, *, max_lag: int = 20) -> SeriesProfile:
        return characterize_series(self.repository.get_series(series_id), max_lag=max_lag)

    def diagnose(
        self,
        series_id: str,
        *,
        model_ref: str | None = None,
        max_lag: int = 20,
    ) -> ResidualDiagnostics:
        """Retorna resíduos e testes do modelo sem exigir que ele seja o champion atual."""

        series = self.repository.get_series(series_id)
        run = self._resolve_run(series_id, model_ref)
        return diagnose_residuals(series, run, max_lag=max_lag)

    def plot_characterization(
        self, series_id: str, *, max_lag: int = 20, figsize=(14, 11)
    ):
        return plot_series_characterization(
            self.characterize(series_id, max_lag=max_lag), figsize=figsize
        )

    def plot_residuals(
        self,
        series_id: str,
        *,
        model_ref: str | None = None,
        max_lag: int = 20,
        figsize=(14, 11),
    ):
        return plot_residual_diagnostics(
            self.diagnose(series_id, model_ref=model_ref, max_lag=max_lag),
            figsize=figsize,
        )

    # ------------------------------------------------------------- modelagem
    def _make_run(self, series: TimeSeries, spec: ModelSpec) -> ModelRun:
        adapter = self.models.get(spec.family)
        state, diagnostics = adapter.fit(series, spec)
        payload = {
            "series_id": series.series_id,
            "series_revision": series.revision,
            "cutoff": series.cutoff,
            "spec": spec,
            "adapter_version": adapter.version,
            "state": state,
        }
        return ModelRun(
            model_run_id=content_id("model", payload),
            series_id=series.series_id,
            series_revision=series.revision,
            cutoff=series.cutoff,
            spec=spec,
            adapter_version=adapter.version,
            state=state,
            diagnostics=diagnostics,
        )

    def fit(
        self, series_id: str, spec: ModelSpec, *, cutoff: pd.Timestamp | None = None
    ) -> ModelRun:
        series = self.repository.get_series(series_id)
        if cutoff is not None:
            cutoff = pd.Timestamp(cutoff)
            positions = np.flatnonzero(series.timestamps <= cutoff)
            if positions.size < 1 or series.timestamps[positions[-1]] != cutoff:
                raise ValidationError("cutoff deve existir na série")
            end = int(positions[-1]) + 1
            series = TimeSeries(
                series.series_id,
                series.timestamps[:end],
                series.values[:end],
                series.frequency,
                series.metadata,
                series.revision,
            )
        run = self._make_run(series, spec)
        self.repository.save_model_run(run)
        self._event(
            "model_fitted",
            series_id,
            {"model_run_id": run.model_run_id, "family": spec.family},
        )
        return run

    def backtest(
        self,
        series_id: str,
        candidates: Sequence[ModelSpec],
        config: BacktestConfig,
    ) -> BacktestReport:
        series = self.repository.get_series(series_id)
        if not candidates:
            raise ValidationError("ao menos um candidato é necessário")
        origins = range(
            config.initial_train_size,
            len(series.values) - config.horizon + 1,
            config.step,
        )
        rows: list[dict[str, Any]] = []
        for origin in origins:
            train = TimeSeries(
                series_id,
                series.timestamps[:origin],
                series.values[:origin],
                series.frequency,
                series.metadata,
                revision=origin - config.initial_train_size,
            )
            actual = series.values[origin : origin + config.horizon]
            for candidate_number, spec in enumerate(candidates):
                adapter = self.models.get(spec.family)
                run = self._make_run(train, spec)
                seed = config.seed + 10_000 * origin + candidate_number
                distribution = adapter.forecast(run, config.horizon, seed=seed)
                standard_deviation = np.sqrt(distribution.variance)
                alpha = 1 - config.confidence
                bounds = distribution.quantile([alpha / 2, 1 - alpha / 2])
                crps = _sample_crps(distribution, actual, seed=seed + 100)
                candidate_id = _candidate_id(spec)
                for step in range(config.horizon):
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "family": spec.family,
                            "origin": origin,
                            "horizon": step + 1,
                            "actual": float(actual[step]),
                            "mean": float(distribution.mean[step]),
                            "std": float(standard_deviation[step]),
                            "lower": float(bounds[0, step]),
                            "upper": float(bounds[1, step]),
                            "crps": float(crps[step]),
                        }
                    )
        if not rows:
            raise ValidationError("configuração não produz origens de backtest")

        scores: list[ModelScore] = []
        for spec in candidates:
            candidate_id = _candidate_id(spec)
            candidate_rows = [row for row in rows if row["candidate_id"] == candidate_id]
            by_horizon: dict[int, Mapping[str, float]] = {}
            for horizon in range(1, config.horizon + 1):
                selected = [row for row in candidate_rows if row["horizon"] == horizon]
                by_horizon[horizon] = self._metrics(selected)
            scores.append(
                ModelScore(candidate_id, self._metrics(candidate_rows), by_horizon)
            )
        report_payload = {
            "series_id": series_id,
            "revision": series.revision,
            "config": config,
            "candidates": tuple(candidates),
            "scores": scores,
        }
        return BacktestReport(
            content_id("backtest", report_payload),
            series_id,
            config,
            tuple(candidates),
            tuple(scores),
            tuple(rows),
        )

    @staticmethod
    def _metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, float]:
        actual = np.array([row["actual"] for row in rows])
        mean = np.array([row["mean"] for row in rows])
        lower = np.array([row["lower"] for row in rows])
        upper = np.array([row["upper"] for row in rows])
        errors = actual - mean
        return {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "coverage": float(np.mean((actual >= lower) & (actual <= upper))),
            "mean_width": float(np.mean(upper - lower)),
            "crps": float(np.mean([row["crps"] for row in rows])),
        }

    def select(
        self, report: BacktestReport, policy: SelectionPolicy | None = None
    ) -> SelectionDecision:
        policy = policy or SelectionPolicy()
        eligible = list(report.scores)
        if policy.coverage_range is not None:
            low, high = policy.coverage_range
            eligible = [
                score
                for score in eligible
                if low <= score.metrics["coverage"] <= high
            ]
        if not eligible:
            raise ValidationError("nenhum candidato passou os guardrails")
        try:
            winner = min(eligible, key=lambda score: score.metrics[policy.primary_metric])
        except KeyError as error:
            raise ValidationError(
                f"métrica de seleção desconhecida: {policy.primary_metric}"
            ) from error
        spec_by_id = {_candidate_id(spec): spec for spec in report.candidates}
        spec = spec_by_id[winner.model_run_family]
        model_run = self.fit(report.series_id, spec)
        payload = {
            "report_id": report.report_id,
            "selected": winner.model_run_family,
            "metric": policy.primary_metric,
            "model_run_id": model_run.model_run_id,
        }
        return SelectionDecision(
            content_id("decision", payload),
            report.report_id,
            report.series_id,
            spec,
            model_run.model_run_id,
            policy.primary_metric,
            winner.metrics,
            f"menor {policy.primary_metric} entre candidatos elegíveis",
        )

    def promote(
        self,
        series_id: str,
        model_run_id: str,
        *,
        alias: str | None = None,
        decision_id: str | None = None,
    ) -> PromotionRecord:
        alias = alias or self.config.default_alias
        run = self.repository.get_model_run(model_run_id)
        if run.series_id != series_id:
            raise ValidationError("modelo pertence a outra série")
        self.repository.set_alias(series_id, alias, model_run_id)
        promotion = PromotionRecord(
            content_id(
                "promotion",
                {
                    "series_id": series_id,
                    "alias": alias,
                    "model_run_id": model_run_id,
                    "decision_id": decision_id,
                },
            ),
            series_id,
            alias,
            model_run_id,
            decision_id,
        )
        self._event(
            "model_promoted",
            series_id,
            {"alias": alias, "model_run_id": model_run_id, "decision_id": decision_id},
        )
        return promotion

    def get_champion(self, series_id: str, *, alias: str | None = None) -> ModelRun:
        alias = alias or self.config.default_alias
        model_run_id = self.repository.get_alias(series_id, alias)
        if model_run_id is None:
            raise NotFittedError(f"alias não promovido: {series_id}/{alias}")
        return self.repository.get_model_run(model_run_id)

    def run_training_cycle(
        self,
        series_id: str,
        candidates: Sequence[ModelSpec],
        backtest: BacktestConfig,
        *,
        selection: SelectionPolicy | None = None,
        promotion: PromotionPolicy | None = None,
    ) -> TrainingResult:
        report = self.backtest(series_id, candidates, backtest)
        decision = self.select(report, selection)
        run = self.repository.get_model_run(decision.model_run_id)
        promotion = promotion or PromotionPolicy()
        record = (
            self.promote(
                series_id,
                run.model_run_id,
                alias=promotion.alias,
                decision_id=decision.decision_id,
            )
            if promotion.automatic
            else None
        )
        return TrainingResult(report, decision, run, record)

    # -------------------------------------------------------- forecast e risco
    def _resolve_run(self, series_id: str, model_ref: str | None) -> ModelRun:
        reference = model_ref or self.config.default_alias
        try:
            run = self.repository.get_model_run(reference)
        except (KeyError, LookupError):
            model_run_id = self.repository.get_alias(series_id, reference)
            if model_run_id is None:
                raise NotFittedError(f"modelo/alias não encontrado: {reference}")
            run = self.repository.get_model_run(model_run_id)
        if run.series_id != series_id:
            raise ValidationError("modelo pertence a outra série")
        return run

    def forecast(
        self,
        series_id: str,
        horizon: int,
        *,
        model_ref: str | None = None,
        seed: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Forecast:
        if horizon < 1:
            raise ValidationError("horizon deve ser positivo")
        series = self.repository.get_series(series_id)
        run = self._resolve_run(series_id, model_ref)
        if run.series_revision != series.revision or run.cutoff != series.cutoff:
            raise NotFittedError(
                "modelo obsoleto: ajuste novamente após alterar a série"
            )
        adapter = self.models.get(run.spec.family)
        distribution = adapter.forecast(
            run, horizon, seed=self.config.default_seed if seed is None else seed
        )
        offset = pd.tseries.frequencies.to_offset(series.frequency)
        targets = pd.date_range(series.cutoff + offset, periods=horizon, freq=series.frequency)
        payload = {
            "series_id": series_id,
            "model_run_id": run.model_run_id,
            "cutoff": series.cutoff,
            "targets": targets,
            "distribution": distribution,
            "metadata": metadata or {},
        }
        result = Forecast(
            content_id("forecast", payload),
            series_id,
            run.model_run_id,
            series.cutoff,
            targets,
            distribution,
            metadata or {},
        )
        self.repository.save_forecast(result)
        self._event(
            "forecast_issued",
            series_id,
            {"forecast_id": result.forecast_id, "model_run_id": run.model_run_id},
        )
        return result

    def simulate(
        self,
        series_id: str,
        horizon: int,
        n_paths: int,
        *,
        model_ref: str | None = None,
        scenario: SimulationScenario | None = None,
        seed: int | None = None,
    ) -> SimulationResult:
        series = self.repository.get_series(series_id)
        run = self._resolve_run(series_id, model_ref)
        if run.series_revision != series.revision or run.cutoff != series.cutoff:
            raise NotFittedError(
                "modelo obsoleto: ajuste novamente após alterar a série"
            )
        scenario = scenario or BaselineScenario()
        adapter = self.models.get(run.spec.family)
        paths = adapter.simulate(
            run,
            horizon,
            n_paths,
            scenario=scenario,
            seed=self.config.default_seed if seed is None else seed,
        )
        offset = pd.tseries.frequencies.to_offset(series.frequency)
        targets = pd.date_range(series.cutoff + offset, periods=horizon, freq=series.frequency)
        payload = {
            "series_id": series_id,
            "model_run_id": run.model_run_id,
            "targets": targets,
            "scenario": scenario,
            "seed": seed,
            "paths_digest": content_id("paths", paths),
        }
        return SimulationResult(
            content_id("simulation", payload),
            series_id,
            run.model_run_id,
            targets,
            paths,
            scenario,
            seed,
        )

    def assess_risk(
        self, source: Forecast | SimulationResult, threshold: ThresholdRisk
    ) -> RiskReport:
        if isinstance(source, Forecast):
            paths = source.distribution.sample(threshold.n_paths, seed=threshold.seed)
            values = np.repeat(threshold.limit, source.distribution.horizon)
            if threshold.direction == "above":
                marginal = 1.0 - source.distribution.cdf(values)
            elif threshold.direction == "below":
                marginal = source.distribution.cdf(values)
            else:
                raise ValidationError("direction deve ser above ou below")
        else:
            paths = source.paths
            if threshold.direction == "above":
                marginal = np.mean(paths > threshold.limit, axis=0)
            elif threshold.direction == "below":
                marginal = np.mean(paths < threshold.limit, axis=0)
            else:
                raise ValidationError("direction deve ser above ou below")
        hits = paths > threshold.limit if threshold.direction == "above" else paths < threshold.limit
        any_probability = float(np.mean(np.any(hits, axis=1)))
        first = np.where(np.any(hits, axis=1), np.argmax(hits, axis=1), -1)
        first_passage = np.array(
            [np.mean(first == step) for step in range(paths.shape[1])], dtype=float
        )
        return RiskReport(np.asarray(marginal), any_probability, first_passage, threshold)

    # -------------------------------------------------------------- operação
    def start_monitoring(
        self,
        series_id: str,
        policy: MonitoringPolicy | None = None,
        *,
        model_alias: str | None = None,
    ) -> MonitoringSession:
        alias = model_alias or self.config.default_alias
        run = self.get_champion(series_id, alias=alias)
        policy = policy or MonitoringPolicy()
        payload = {
            "series_id": series_id,
            "model_alias": alias,
            "model_run_id": run.model_run_id,
            "policy": policy,
        }
        detector_states: dict[str, Mapping[str, Any]] = {}
        for name in policy.detectors:
            detector_states[name] = self.detectors.create(name, policy).snapshot()
        session = MonitoringSession(
            content_id("monitor", payload),
            series_id,
            alias,
            run.model_run_id,
            policy,
            detector_states=detector_states,
        )
        self.repository.save_monitoring_session(session)
        self.forecast(
            series_id,
            policy.forecast_horizon,
            model_ref=run.model_run_id,
            metadata={"monitoring_session_id": session.session_id},
        )
        self._event("monitoring_started", series_id, {"session_id": session.session_id})
        return session

    def process_observation(
        self, session_id: str, observation: Observation
    ) -> CycleResult:
        previous = self.repository.find_cycle_result(observation.observation_id)
        if previous is not None:
            if previous.observation != observation:
                raise RepositoryConflictError(
                    "observation_id já existe com conteúdo diferente"
                )
            return replace(previous, idempotent_replay=True)

        session = self.repository.get_monitoring_session(session_id)
        series = self.repository.get_series(session.series_id)
        updated_series = series.append(observation)  # valida ordem antes de efeitos
        forecast = self.repository.find_pending_forecast(
            session.series_id, session.model_run_id, pd.Timestamp(observation.timestamp)
        )

        sequence = session.sequence + 1
        surprise: float | None = None
        reasons: list[str] = []
        positive, negative, ewma = session.cusum_positive, session.cusum_negative, session.ewma
        ewma_limit: float | None = None
        emitted = False
        kind = "observation_unscored"
        detector_states = {
            name: dict(state) for name, state in session.detector_states.items()
        }
        active_detectors = []
        if forecast is not None:
            position = int(forecast.targets.get_loc(pd.Timestamp(observation.timestamp)))
            cdf = float(
                forecast.distribution.cdf(
                    np.repeat(observation.value, forecast.distribution.horizon)
                )[position]
            )
            probability = np.clip(cdf, self.config.pit_clip, 1 - self.config.pit_clip)
            surprise = float(norm.ppf(probability))
            statistics: dict[str, Mapping[str, float]] = {}
            for name in session.policy.detectors:
                detector = self.detectors.create(name, session.policy)
                detector.restore(detector_states.get(name, {}))
                stats, detector_reasons = detector.update(surprise)
                statistics[name] = stats
                reasons.extend(detector_reasons)
                active_detectors.append((name, detector))
            cusum_stats = statistics.get("cusum", {})
            ewma_stats = statistics.get("ewma", {})
            positive = float(cusum_stats.get("positive", positive))
            negative = float(cusum_stats.get("negative", negative))
            ewma = float(ewma_stats.get("statistic", ewma))
            if "limit" in ewma_stats:
                ewma_limit = float(ewma_stats["limit"])
            emitted = bool(reasons) and sequence - session.last_emitted > session.policy.cooldown
            kind = "alert" if emitted else "observation_scored"
            self.repository.mark_forecast_evaluated(
                forecast.forecast_id, pd.Timestamp(observation.timestamp)
            )

        monitoring_event = MonitoringEvent(
            content_id(
                "monitoring_event",
                {
                    "session_id": session_id,
                    "observation_id": observation.observation_id,
                    "sequence": sequence,
                    "surprise": surprise,
                    "reasons": reasons,
                },
            ),
            session_id,
            session.series_id,
            forecast.forecast_id if forecast else None,
            observation.observation_id,
            sequence,
            float(observation.value),
            surprise,
            float(positive),
            float(negative),
            float(ewma),
            None if ewma_limit is None else float(ewma_limit),
            tuple(reasons),
            emitted,
            kind,
        )
        self.repository.update_series(updated_series, expected_revision=series.revision)
        self.repository.save_monitoring_event(monitoring_event)

        last_emitted = sequence if emitted else session.last_emitted
        if emitted and session.policy.reset_on_alert:
            for _, detector in active_detectors:
                detector.reset()
        for name, detector in active_detectors:
            detector_states[name] = detector.snapshot()
        positive = float(detector_states.get("cusum", {}).get("positive", positive))
        negative = float(detector_states.get("cusum", {}).get("negative", negative))
        ewma = float(detector_states.get("ewma", {}).get("statistic", ewma))

        model_run_id = session.model_run_id
        if session.policy.refit_each_observation:
            old_run = self.repository.get_model_run(session.model_run_id)
            new_run = self.fit(session.series_id, old_run.spec)
            self.repository.set_alias(session.series_id, session.model_alias, new_run.model_run_id)
            model_run_id = new_run.model_run_id
        updated_session = replace(
            session,
            model_run_id=model_run_id,
            sequence=sequence,
            cusum_positive=float(positive),
            cusum_negative=float(negative),
            ewma=float(ewma),
            last_emitted=last_emitted,
            detector_states=detector_states,
            revision=session.revision + 1,
        )
        self.repository.save_monitoring_session(
            updated_session, expected_revision=session.revision
        )
        next_forecast = self.forecast(
            session.series_id,
            session.policy.forecast_horizon,
            model_ref=model_run_id,
            metadata={"monitoring_session_id": session_id},
        )
        events = [
            self._event(
                kind,
                session.series_id,
                {
                    "monitoring_event_id": monitoring_event.event_id,
                    "observation_id": observation.observation_id,
                },
            )
        ]
        if any(reason in {"cusum_positive", "cusum_negative", "ewma_drift"} for reason in reasons):
            events.append(
                self._event(
                    "retraining_recommended",
                    session.series_id,
                    {"session_id": session_id, "reasons": tuple(reasons)},
                )
            )
        result = CycleResult(
            observation, monitoring_event, next_forecast, tuple(events)
        )
        self.repository.save_cycle_result(result)
        return result

    def detect_changes(
        self,
        series_id: str,
        *,
        penalty: float,
        min_size: int = 10,
        start: pd.Timestamp | None = None,
    ) -> ChangePointReport:
        series = self.repository.get_series(series_id)
        first = 0
        if start is not None:
            first = int(series.timestamps.searchsorted(pd.Timestamp(start)))
        result = optimal_mean_segmentation(
            series.values[first:], penalty=penalty, min_size=min_size
        )
        breakpoints = tuple(
            series.timestamps[first + position] for position in result.breakpoints
        )
        return ChangePointReport(
            series_id, breakpoints, result.objective, result.fitted_values
        )

    # --------------------------------------------------------------- consulta
    def get_forecast(self, forecast_id: str) -> Forecast:
        return self.repository.get_forecast(forecast_id)

    def get_model_run(self, model_run_id: str) -> ModelRun:
        return self.repository.get_model_run(model_run_id)

    def list_events(self, series_id: str) -> tuple[EngineEvent, ...]:
        return tuple(self.repository.list_events(series_id))

    def list_alerts(self, series_id: str) -> tuple[EngineEvent, ...]:
        return tuple(event for event in self.list_events(series_id) if event.kind == "alert")

    def get_status(self, series_id: str) -> EngineStatus:
        series = self.repository.get_series(series_id)
        sessions = self.repository.list_monitoring_sessions(series_id)
        return EngineStatus(
            series_id,
            series.revision,
            series.cutoff,
            dict(self.repository.list_aliases(series_id)),
            tuple(session.session_id for session in sessions),
        )

    def _event(
        self, kind: str, series_id: str, payload: Mapping[str, Any]
    ) -> EngineEvent:
        event_payload = {
            "kind": kind,
            "series_id": series_id,
            "payload": payload,
            "created_at": utc_now(),
        }
        event = EngineEvent(
            content_id("event", event_payload), kind, series_id, dict(payload)
        )
        self.repository.save_event(event)
        return event


__all__ = ["ForecastEngine"]
