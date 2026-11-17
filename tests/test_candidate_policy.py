from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from time_series_lab import (
    AR1Spec,
    ARIMASpec,
    CandidatePolicy,
    ForecastEngine,
    GARCH11Spec,
    LocalLevelSpec,
    TimeSeries,
    ValidationError,
)


class CandidateRepository:
    """Backend mínimo necessário para caracterizar e propor candidatos."""

    def __init__(self) -> None:
        self.series: dict[str, TimeSeries] = {}
        self.events = []

    def add_series(self, series: TimeSeries) -> None:
        self.series[series.series_id] = series

    def get_series(self, series_id: str) -> TimeSeries:
        return self.series[series_id]

    def list_series(self):
        return tuple(self.series.values())

    def save_event(self, event) -> None:
        self.events.append(event)


def make_engine(values: np.ndarray, *, kind: str = "level") -> ForecastEngine:
    repository = CandidateRepository()
    engine = ForecastEngine(repository=repository)
    engine.register_series(
        TimeSeries(
            "candidate-series",
            pd.date_range("2025-01-01", periods=len(values), freq="D", tz="UTC"),
            values,
            "D",
            metadata={"series_kind": kind},
        )
    )
    return engine


def ar1_values(n: int = 120, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.empty(n)
    values[0] = 0.0
    for index in range(1, n):
        values[index] = 0.2 + 0.65 * values[index - 1] + rng.normal(0, 0.5)
    return values


def test_candidate_plan_is_bounded_and_explains_every_decision() -> None:
    engine = make_engine(ar1_values())
    profile = engine.characterize("candidate-series", max_lag=16)
    plan = engine.propose_candidates("candidate-series", profile=profile)

    assert 4 <= len(plan.candidates) <= CandidatePolicy().max_candidates
    assert any(isinstance(spec, AR1Spec) for spec in plan.candidates)
    assert any(isinstance(spec, LocalLevelSpec) for spec in plan.candidates)
    assert ARIMASpec(order=(0, 0, 0), simulation_paths=300) in plan.candidates
    assert ARIMASpec(order=(0, 0, 1), simulation_paths=300) in plan.candidates
    assert all(item.reason and item.rule for item in plan.decisions)
    assert set(plan.to_frame().status) >= {"included", "excluded"}
    assert "Estacionariedade:" in plan.explain()
    assert plan.plan_id.startswith("candidate_plan_")
    assert engine.propose_candidates(
        "candidate-series", profile=profile
    ).plan_id == plan.plan_id


def test_compare_mode_includes_level_and_differenced_arima_hypotheses() -> None:
    engine = make_engine(ar1_values(n=80))
    policy = CandidatePolicy(
        differencing="compare",
        garch="never",
        markov_switching="never",
        max_candidates=14,
    )
    plan = engine.propose_candidates("candidate-series", policy=policy)
    orders = {
        spec.order for spec in plan.candidates if isinstance(spec, ARIMASpec)
    }

    assert {(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)} <= orders
    assert {(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)} <= orders


def test_specialized_families_have_explicit_gates() -> None:
    engine = make_engine(ar1_values(n=57), kind="level")
    plan = engine.propose_candidates(
        "candidate-series",
        policy=CandidatePolicy(minimum_markov_observations=60),
    )
    excluded = plan.to_frame().query("status == 'excluded'")

    assert not any(isinstance(spec, GARCH11Spec) for spec in plan.candidates)
    assert "garch_gate" in set(excluded.rule)
    assert "markov_gate" in set(excluded.rule)
    assert any("returns" in reason for reason in excluded.reason)
    assert any("60" in reason for reason in excluded.reason)


def test_markov_can_be_proposed_with_less_than_80_observations() -> None:
    rng = np.random.default_rng(19)
    values = np.r_[rng.normal(0.0, 0.25, 29), rng.normal(2.0, 0.25, 28)]
    engine = make_engine(values, kind="level")
    plan = engine.propose_candidates(
        "candidate-series",
        policy=CandidatePolicy(max_candidates=14),
    )

    markov_decision = next(
        item for item in plan.decisions if item.family == "markov_switching"
    )
    assert markov_decision.status == "included"
    assert markov_decision.evidence["free_parameters"] == 6
    assert "exploratória" in markov_decision.reason


def test_policy_validates_configuration_and_profile_identity() -> None:
    with pytest.raises(ValidationError):
        CandidatePolicy(max_candidates=3)

    engine = make_engine(ar1_values(), kind="level")
    profile = engine.characterize("candidate-series")
    with pytest.raises(ValidationError, match="outra série"):
        engine.propose_candidates(
            "candidate-series", profile=replace(profile, series_id="other-series")
        )

