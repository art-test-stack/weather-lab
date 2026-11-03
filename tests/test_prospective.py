import copy

import numpy as np
import pandas as pd
import pytest

from weather_lab import models, prospective as p
from weather_lab.forecast import ForecastBatch
from weather_lab.forecast_config import ForecastConfig
from weather_lab.prospective_collect import rows_json
from weather_lab.prospective_store import Archive, digest, encoded

ORIGIN = pd.Timestamp("2026-11-03T04:00Z")


def fixture(archive):
    cfg = ForecastConfig.make("Timestral-28M", targets=("t", "pmer"), step_hours=1, calendar=False)
    ck = models.CHECKPOINTS[cfg.model]
    rec = {"label": "Saved leader", "config": cfg.as_dict(),
           "checkpoint": {k: getattr(ck, k) for k in ("repo", "revision", "sha256")},
           "adapter_version": 5, "preprocessing_version": 1}
    ident = digest(encoded(rec))
    archive.put("catalog", ident, rec)
    times = pd.date_range(end=ORIGIN, periods=24 * 31, freq="h")
    obs = pd.DataFrame({"station": "07149", "time": times, "t": 10., "pmer": 1000., "u": 0., "v": 0.})
    # Future observation revisions must not replace the version actually available at decision.
    for available, temperature in [("2026-11-03T03:00Z", 10.), ("2026-11-03T05:00Z", 50.)]:
        rec_obs = {"identity": "07149/history", "collected_at": available,
                   "rows": rows_json(obs.assign(t=temperature))}
        archive.put("observations", available, rec_obs)
    issued = pd.DataFrame([{"station": "07149", "run": "2026-11-03T00:00Z",
        "valid_time": valid, "available_at": "2026-11-03T03:30Z", "variable": variable,
        "value": value, "product": p.PRODUCT, "availability_basis": "actual_first_collection", "source_kind": "issued_run"}
        for valid in pd.date_range("2026-11-03T01:00Z", periods=51, freq="h")
        for variable, value in [("t", 12.), ("pmer", 1002.)]])
    archive.put("forecasts", "issue", {"run": "2026-11-03T00:00Z", "collected_at": "2026-11-03T03:30Z",
                                         "rows": rows_json(issued)})
    return ident


@pytest.mark.parametrize("finished,missing", [("2026-11-03T04:30:00Z", 0),
                                             ("2026-11-03T05:00:00Z", 1),
                                             ("2026-11-03T06:30:00Z", 2)])
def test_inference_point_in_time_immutable_and_provenance(tmp_path, monkeypatch, finished, missing):
    monkeypatch.setattr(p, "stations", lambda: [{"id": "07149", "lat": 48., "lon": 2.}])
    archive = Archive(tmp_path)
    seen = []
    def predict(cfg, record, batch, model, **kwargs):
        assert batch.history[0, 0, -1] == 10  # later corrected 50 is unavailable
        assert pd.Timestamp(batch.history_times()[0, -1], tz="UTC") == ORIGIN - pd.Timedelta(hours=3)
        assert batch.history.shape == (1, 2, 720)
        seen.append(batch)
        point = np.broadcast_to(np.array([10., 1000.])[None, :, None], (1, 2, batch.horizon)).copy()
        return ForecastBatch(model, "frozen", "point", batch.variables, point).to_frame(batch, "test"), {}
    with archive.writer():
        fixture(archive)
        count = p.infer(archive, ORIGIN, ORIGIN, predict=predict, clock=lambda: finished)
        assert count == 2
        assert p.infer(archive, ORIGIN, ORIGIN, predict=lambda *a: pytest.fail("regenerated prediction"),
                       clock=lambda: finished) == 0
    assert len(seen) == 1
    record = next(r["value"] for r in archive.records("predictions") if r["value"]["model"] != "arome-raw")
    assert record["provenance"]["configuration"]["checkpoint"]["revision"] == models.CHECKPOINTS["Timestral-28M"].revision
    assert record["provenance"]["input_sha256"]
    assert record["provenance"]["code_revision"]
    assert all(r["lead_h"] > 0 for r in record["rows"])
    assert record["cohort"] == "prospective-2026-11"
    assert sum(row["point"] is None for row in record["rows"]) == 2 * missing


def test_late_freeze_and_missing_inputs_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "stations", lambda: [{"id": "07630", "lat": 43., "lon": 1.}])
    archive = Archive(tmp_path)
    with archive.writer():
        fixture(archive)
        p.infer(archive, ORIGIN, ORIGIN, predict=lambda *a: pytest.fail("missing data reached inference"),
                clock=lambda: "2026-11-06T04:00:00Z")
    records = [r["value"] for r in archive.records("predictions")]
    assert all(r["status"] == "unavailable" for r in records)
    assert all(row["point"] is None for r in records for row in r["rows"])
    assert len(archive.records("availability")) == 2


@pytest.mark.parametrize("point,gain", [(11., 2.), (15., -2.)])
def test_common_support_metrics_revisions_and_no_cross_cohort_pooling(tmp_path, point, gain):
    archive = Archive(tmp_path)
    base = {"cohort": "prospective-2026-11", "station": "07149", "case_id": "case", "variable": "t",
            "valid_time": "2026-11-03T05:00Z", "lead_h": 1, "issue": "2026-11-03T00:00Z", "cycle": 0}
    with archive.writer():
        for model, value in [("arome-raw", 13.), ("first", point), ("missing", None)]:
            archive.put("predictions", model, {"rows": [{**base, "model": model, "point": value}]})
        archive.put("observations", "truth", {"identity": "07149/2026-11-03", "collected_at": "2026-11-04T04:00Z",
            "rows": [{"station": "07149", "time": "2026-11-03T05:00Z", "t": 10., "pmer": None}]})
    metrics, coverage = p.evaluate(archive)
    assert metrics.empty and not coverage.paired.any()  # every selected model must have the same pair
    metrics, coverage = p.evaluate(archive, ["first"])
    model = metrics[metrics.model.eq("first")].iloc[0]
    assert model.mae == model.rmse == point - 10
    assert model.gain == gain and model.absolute_gain == abs(gain)
    assert metrics.cohort.unique().tolist() == ["prospective-2026-11"]
    assert coverage.observation_version.notna().all()
    with archive.writer():
        score = p.score(archive)
        assert p.score(archive) == score
        assert len(archive.records("evaluations")) == 1
    assert score["prediction_records"] and score["observation_records"] and score["scorer_sha256"]


def test_configuration_version_and_checkpoint_changes_refused():
    cfg = ForecastConfig.make("Timestral-28M", features=["solar"], targets=("t", "pmer"), step_hours=1)
    saved = cfg.as_dict()
    assert p.configuration(saved) == cfg
    changed = copy.deepcopy(saved)
    changed["features"][0]["version"] += 1
    with pytest.raises(ValueError, match="version"):
        p.configuration(changed)
    with pytest.raises(ValueError, match="checkpoint"):
        p.predict_config(cfg, {"checkpoint": {"revision": "newer"}}, None, "saved")


def test_retrospective_official_availability_is_distinct_from_live_collection(tmp_path):
    archive = Archive(tmp_path)
    with archive.writer():
        archive.put("forecasts", "official", {"run": "2026-10-20T00:00Z", "collected_at": "2026-11-03T12:00Z",
            "published_at": "2026-10-20T03:20Z", "rows": [{"station": "07149", "product": p.PRODUCT,
                "run": "2026-10-20T00:00Z", "available_at": "2026-11-03T12:00Z", "valid_time": "2026-10-20T05:00Z",
                "variable": "t", "value": 10., "availability_basis": "actual_first_collection", "source_kind": "issued_run"}]})
    cutoff = pd.Timestamp("2026-10-20T04:00Z")
    assert p.forecast_frame(archive, cutoff).empty
    replay = p.forecast_frame(archive, cutoff, replay=True)
    assert len(replay) == 1
    assert replay.iloc[0].available_at == pd.Timestamp("2026-10-20T03:20Z")
    assert "retrospective" in replay.iloc[0].availability_basis
    assert list(p.decisions(archive, cutoff, cutoff, True))[0][1] == cutoff


@pytest.mark.real_model
@pytest.mark.parametrize("model", ["Timestral-28M", "Timestral-209M", "t0-gamma"])
def test_real_pinned_timestral_adapter(model):
    """Explicit opt-in; real model outputs are temporary test fixtures, never scores."""
    cfg = ForecastConfig.make(model, targets=("t", "pmer"), step_hours=1, calendar=False)
    ck = models.CHECKPOINTS[cfg.model]
    models.prefetch(ck, allow_download=False)
    from weather_lab.forecast import hourly_inputs
    from weather_lab.forecast_config import build_adapter

    cases = pd.DataFrame([{"case_id": "synthetic-test-only", "station": "07149", "origin": ORIGIN}])
    obs = pd.DataFrame({"station": "07149", "time": pd.date_range(end=ORIGIN, periods=24 * 31, freq="h"),
                        "t": 10., "pmer": 1000.})
    batch = hourly_inputs(obs, cases, list(cfg.targets), 720, 48, 3)
    adapter = build_adapter(cfg, "synthetic-test-only", device="cpu", arome_fields=[])
    record = {"checkpoint": {k: getattr(ck, k) for k in ("repo", "revision", "sha256")},
              "adapter_version": adapter.ADAPTER_VERSION, "preprocessing_version": adapter.PREPROCESSING_VERSION}
    frame, info = p.predict_config(cfg, record, batch, "synthetic-test-only")
    assert len(frame) == 96 and np.isfinite(frame.point).all()
    assert frame.lead_h.min() == 1 and frame.lead_h.max() == 48
    assert info["adapter"]
