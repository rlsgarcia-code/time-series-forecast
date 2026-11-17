import json

import numpy as np
import pytest

from time_series_lab.operations import (
    AlertPolicy,
    ForecastEnvelope,
    LocalArtifactStore,
    SequentialMonitor,
    gaussian_exceedance_probability,
    path_exceedance_probability,
)


def make_envelope() -> ForecastEnvelope:
    return ForecastEnvelope.create(
        model_name="ar1",
        cutoff=120,
        horizons=[1, 2, 3],
        mean=[0.1, 0.2, 0.3],
        std=[1.0, 1.1, 1.2],
        metadata={"version": "test"},
    )


def test_forecast_envelope_has_deterministic_content_hash() -> None:
    first = make_envelope()
    second = make_envelope()
    assert first.forecast_id == second.forecast_id
    assert first.verify()


def test_artifact_store_roundtrip_and_refuses_overwrite(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    envelope = make_envelope()
    path = store.save(envelope)
    assert store.load(envelope.forecast_id) == envelope
    with pytest.raises(FileExistsError):
        store.save(envelope)
    assert path.exists()


def test_artifact_store_detects_tampering(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    envelope = make_envelope()
    path = store.save(envelope)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["mean"][0] = 999.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="corrompido"):
        store.load(envelope.forecast_id)


def test_monitor_detects_sustained_shift_and_respects_cooldown() -> None:
    policy = AlertPolicy(
        point_threshold=10.0,
        cusum_reference=0.25,
        cusum_threshold=3.0,
        ewma_smoothing=0.2,
        ewma_limit_multiplier=10.0,
        cooldown=5,
        reset_on_alert=True,
    )
    monitor = SequentialMonitor(policy)
    events = [
        monitor.update(forecast_id=f"f{index}", actual=1.0, mean=0.0, std=1.0)
        for index in range(20)
    ]
    emitted = [event.sequence for event in events if event.emitted]
    assert emitted
    assert all(right - left > policy.cooldown for left, right in zip(emitted, emitted[1:]))
    assert any("cusum_positive" in event.reasons for event in events)


def test_monitor_flags_point_anomaly_with_forecast_provenance() -> None:
    monitor = SequentialMonitor(AlertPolicy(point_threshold=3.0, cooldown=0))
    event = monitor.update(forecast_id="abc", actual=4.0, mean=0.0, std=1.0)
    assert event.forecast_id == "abc"
    assert "point_anomaly" in event.reasons
    assert event.emitted


def test_gaussian_exceedance_probability_is_monotone_in_mean() -> None:
    probability = gaussian_exceedance_probability(
        mean=np.array([-1.0, 0.0, 1.0]), std=np.ones(3), threshold=0.5
    )
    assert np.all(np.diff(probability) > 0)
    assert np.all((probability >= 0) & (probability <= 1))


def test_path_exceedance_uses_joint_paths() -> None:
    paths = np.array([[0.0, 0.1], [0.0, 2.0], [3.0, -1.0], [-2.0, -1.0]])
    assert path_exceedance_probability(paths, threshold=1.0) == 0.5
