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
    archive.put("status", "catalog-complete", {"task": "catalog", "status": "ok",
        "collected_at": "2026-11-01T00:00Z", "configurations": [ident]})
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
    early, early_coverage = p.evaluate(archive, ["first"], asof="2026-11-03T06:00Z")
    assert early.empty and not early_coverage.paired.any()
    metrics, coverage = p.evaluate(archive, ["first"], asof="2026-11-06T00:00Z")
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
        archive.put("forecasts", "official", {"run": "2026-10-20T00:00Z", "collected_at": "2026-11-03T12:00:00.000001Z",
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
    assert p.forecast_frame(archive, pd.Timestamp("2026-11-03T12:00Z")).empty
    live = list(p.decisions(archive, pd.Timestamp("2026-11-03T12:00Z"), pd.Timestamp("2026-11-03T13:00Z")))
    assert live[0][1] == pd.Timestamp("2026-11-03T13:00Z")


def test_end_to_end_freeze_then_delayed_verification(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "stations", lambda: [{"id": "07149", "lat": 48., "lon": 2.}])
    archive = Archive(tmp_path)
    def predict(cfg, record, batch, model, **kwargs):
        point = np.broadcast_to(np.array([10., 1000.])[None, :, None], (1, 2, batch.horizon)).copy()
        return ForecastBatch(model, "frozen", "point", batch.variables, point).to_frame(batch, "test"), {}
    with archive.writer():
        model = fixture(archive)
        p.infer(archive, ORIGIN, ORIGIN, predict=predict, clock=lambda: "2026-11-03T04:30Z")
        assert p.evaluate(archive, [model], asof="2026-11-03T04:30Z")[0].empty
        truth = pd.DataFrame({"station": "07149", "time": pd.date_range(ORIGIN + pd.Timedelta(hours=1), periods=48, freq="h"),
                              "t": 10., "pmer": 1000.})
        archive.put("observations", "verification", {"identity": "07149/verification", "collected_at": "2026-11-06T00:00Z",
                                                       "rows": rows_json(truth)})
        monkeypatch.setattr(p, "now_utc", lambda: "2026-11-06T01:00:00Z")
        result = p.score(archive)
    table = pd.DataFrame(result["metrics"])
    scores = table[table.model.eq(model)]
    assert len(scores) == 94  # two variables, 47 common leads; raw AROME's last hour is missing
    assert scores.mae.eq(0).all() and scores.gain.eq(2).all()
    assert set(scores.cohort) == {"prospective-2026-11"}
    assert len(archive.records("predictions")) == 2


def test_catalog_partial_upload_is_invisible_and_retryable(tmp_path, monkeypatch):
    import json

    cfg = ForecastConfig.make("Timestral-28M", targets=("t", "pmer"), step_hours=1)
    source = {"forecast_config": cfg.as_dict(), "checkpoint": {}, "adapter_version": 1, "preprocessing_version": 1}
    saved = {}
    for name in ("first", "second"):
        path = tmp_path / name
        path.mkdir()
        (path / "run.json").write_text(json.dumps(source))
        saved[str(path)] = {"label": name, "config": cfg.as_dict(), "summary": {}}
    monkeypatch.setattr(p, "saved_configurations", lambda suite: saved)
    archive = Archive(tmp_path / "archive")
    original = archive.put
    def fail_second(kind, key, value):
        if kind == "catalog" and len(archive.records("catalog")) == 1:
            raise OSError("interrupted")
        return original(kind, key, value)
    with archive.writer():
        monkeypatch.setattr(archive, "put", fail_second)
        with pytest.raises(OSError):
            p.freeze_catalog(archive, "saved")
        assert p.catalog_records(archive) == []
        monkeypatch.setattr(archive, "put", original)
        ids = p.freeze_catalog(archive, "saved")
        assert len(p.catalog_records(archive)) == 2
        assert p.freeze_catalog(archive, "saved") == ids


def test_absent_historical_issues_are_coverage_gaps_not_forecasts(tmp_path, monkeypatch):
    from weather_lab import prospective_collect as collection

    monkeypatch.setattr(collection, "stations", lambda: [{"id": "07149"}])
    archive = Archive(tmp_path)
    with archive.writer():
        model = fixture(archive)
        collection.record_backfill_gaps(archive, pd.Timestamp("2026-10-01T00:00Z"), pd.Timestamp("2026-10-01T03:00Z"))
    assert len(archive.records("availability")) == 4
    assert archive.records("predictions") == []
    metrics, coverage = p.evaluate(archive, [model], asof="2026-11-06T00:00Z")
    assert metrics.empty and len(coverage) == 384
    assert coverage.point.isna().all()
    assert set(coverage.cohort) == {"retrospective-2026-10"}


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
