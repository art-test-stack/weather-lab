"""Full-suite development execution with synthetic data and no model weights."""
import numpy as np
import pandas as pd
import pytest

from weather_lab import development
from weather_lab.common import read_json, sha256_file, write_json
from weather_lab.forecast import ForecastBatch, QUANTILE_LEVELS, Timestral as NativeTimestral
from weather_lab.forecast_config import ForecastConfig


@pytest.fixture
def source(tmp_path):
    p = tmp_path / "source"
    p.mkdir()
    origins = pd.date_range("2026-10-10T06:00Z", periods=3, freq="24h")
    cases = [{"case_id": f"c{i}", "station": "07149", "origin": str(o), "split": "validation",
              "season": "SON", "eligible": True} for i, o in enumerate(origins)]
    write_json(p / "manifest.json", {"suite": "fixture", "cases": cases,
        "dataset": {"stations": [{"id": "07149", "lat": 48.7, "lon": 2.3}]},
        "protocol": {"stations": ["07149"], "variables": ["t"],
                     "forecast": {"step_hours": 3, "horizon_steps": 2, "context_days": 7}}})
    obs = pd.DataFrame({"station": "07149", "time": pd.date_range(origins[0] - pd.Timedelta(days=7),
                       origins[-1] + pd.Timedelta(hours=6), freq="3h"), "t": 10.})
    obs.to_parquet(p / "observation_windows.parquet", index=False)
    targets, predictions = [], []
    for i, o in enumerate(origins):
        for lead in (3, 6):
            row = {"case_id": f"c{i}", "variable": "t", "lead_h": lead,
                   "valid_time": o + pd.Timedelta(hours=lead)}
            targets.append(row | {"observed": 10.})
            predictions.append(row | {"suite": "fixture", "model": "persistence", "origin": o,
                                      "station": "07149", "point": 12., "output_type": "point"})
    pd.DataFrame(targets).to_parquet(p / "targets.parquet", index=False)
    pd.DataFrame(predictions).to_parquet(p / "predictions.parquet", index=False)
    write_json(p / "run.json", {"suite": "fixture", "run_id": "fixture-run", "data": {},
        "systems": [{"id": "persistence", "kind": "persistence", "per_case_ms": 0, "peak_rss_mb_after": 0}]})
    return p


def render_development(out, monkeypatch):
    from weather_lab_app import benchmark as bench, main as app
    from weather_lab_app.results import load_run

    run = load_run(out)
    monkeypatch.setitem(app.RUNS, "fixture", run)
    monkeypatch.setitem(app.EXPERIMENTS, "fixture", {"headline": run.models, "default_systems": run.models,
                        "band": run.models[-1], "reference": "persistence", "sample": "synthetic test"})
    d = {"suite": "fixture", "variable": "t", "metric": "rmse", "lead": "all", "split": "validation",
         "station": "ALL", "season": "ALL", "statuses": app.STATUSES, "size": "28M", "conditioning": "none",
         "lead_mode": "at"}
    cfg = ForecastConfig.make("Timestral-28M", context_days=7, horizon_h=6, targets=("t",))
    outputs = bench.render({"draft": d, "config": cfg, "path": str(out), "kind": "development"})
    assert len(outputs) == 19 and "Development benchmark" in outputs[0]
    assert outputs[15] == str(out / "predictions.parquet")
    assert outputs[17] == str(out / "run.json")
    assert outputs[18] == str(out / "case_availability.csv")
    return outputs


def test_full_suite_saves_matched_scores_failures_and_provenance_without_modifying_source(source, tmp_path,
                                                                                      monkeypatch):
    before = {p.name: sha256_file(p) for p in source.iterdir()}
    calls = []

    def predict(self, batch):
        calls.append(batch.cases.iloc[0].case_id)
        assert batch.history.shape == (1, 1, 56)
        assert batch.history.max() == 10  # no targets or future observations in adapter input
        if calls[-1] == "c1":
            raise ValueError("fixture prediction failure")
        point = np.full((1, 1, 2), 11.)
        q = point[..., None] + np.array(QUANTILE_LEVELS) - .5
        return ForecastBatch(self.id, "fixture-revision", "quantiles", ["t"], point, q, QUANTILE_LEVELS)

    monkeypatch.setattr(NativeTimestral, "predict", predict)
    monkeypatch.setattr(development.models, "prefetch", lambda ck, allow_download: {"downloaded": False})
    cfg = ForecastConfig.make("Timestral-28M", context_days=7, horizon_h=6, targets=("t",))
    out = development.run(source, cfg, root=tmp_path / "results")
    assert calls == ["c0", "c1", "c2"]
    rec = read_json(out / "run.json")
    assert rec["development"] and rec["source_suite"] == "fixture" and rec["n_eligible"] == 3
    assert rec["forecast_config"] == cfg.as_dict()
    assert rec["checkpoint"]["revision"] == development.models.CHECKPOINTS[cfg.model].revision
    assert set(rec["input_sha256"]) == {"c0", "c1", "c2"}
    assert all(sha256_file(out / name) == digest for name, digest in rec["files_sha256"].items())
    derived = read_json(out / "manifest.json")
    assert derived["suite"] != "fixture" and derived["leads_h"] == [3, 6]
    available = pd.read_csv(out / "case_availability.csv").set_index("case_id")
    assert available.status.to_dict() == {"c0": "available", "c1": "failed", "c2": "available"}
    assert "fixture prediction failure" in available.loc["c1", "reason"]
    metrics = pd.read_csv(out / "metrics_point.csv")
    metrics = metrics[metrics.station.eq("ALL") & metrics.season.eq("ALL") & metrics.lead_h.isna()]
    assert set(metrics.n) == {4}
    assert metrics.set_index("model").loc["persistence", "mae"] == 2
    primary = metrics[metrics.model.str.startswith("timestral-dev")].iloc[0]
    assert primary.mae == primary.rmse == 1
    assert {p.name: sha256_file(p) for p in source.iterdir()} == before
    monkeypatch.setattr(development, "location", lambda: tmp_path / "results")
    assert development.saved_runs("fixture")[0][1] == str(out)
    outputs = render_development(out, monkeypatch)
    assert list(outputs[6].columns[:3]) == ["System", "RMSE (°C)", "MAE (°C)"]


def test_missing_selected_inputs_are_unavailable_without_loading_a_model(source, tmp_path, monkeypatch):
    monkeypatch.setattr(NativeTimestral, "predict", lambda *args: pytest.fail("missing inputs ran a model"))
    cfg = ForecastConfig.make("Timestral-28M", features=["atmos"], context_days=7, horizon_h=6, targets=("t",))
    out = development.run(source, cfg, root=tmp_path / "results")
    availability = pd.read_csv(out / "case_availability.csv")
    assert set(availability.status) == {"unavailable"}
    assert availability.reason.str.contains("auxiliary").all()
    assert pd.read_csv(out / "metrics_point.csv").empty
    assert pd.read_csv(out / "availability.csv").query('model.str.startswith("timestral-dev")').failed_predictions.iloc[0] == 6
    outputs = render_development(out, monkeypatch)
    assert outputs[6].empty and "No matched" in outputs[5]


def test_error_correction_saves_physical_predictions_and_missing_archives_without_fallback(source, tmp_path,
                                                                                        monkeypatch):
    manifest, rows = read_json(source / "manifest.json"), []
    for case in manifest["cases"]:
        origin = pd.Timestamp(case["origin"])
        run = origin - pd.Timedelta(hours=3)
        case["arome_run"] = str(run)
        for lead in (3, 6):
            rows.append({"station": case["station"], "run": run, "available_at": origin - pd.Timedelta(minutes=15),
                         "valid_time": origin + pd.Timedelta(hours=lead), "variable": "t", "value": 12.})
    write_json(source / "manifest.json", manifest)
    ar = source / "arome_station_extract.parquet"
    pd.DataFrame(rows).to_parquet(ar, index=False)
    rec = read_json(source / "run.json")
    write_json(source / "run.json", rec | {"data": {"arome": {"sha256": sha256_file(ar)}}})
    calls = []

    def predict(self, batch):
        calls.append(batch.cases.iloc[0].case_id)
        errors, baseline, _ = self.error_inputs(batch)
        assert np.nanmax(errors) == -2
        point = baseline - 1.
        return ForecastBatch(self.id, "fixture-revision", "quantiles", ["t"], point,
                             point[..., None] + np.array(QUANTILE_LEVELS) - .5, QUANTILE_LEVELS)

    monkeypatch.setattr(NativeTimestral, "predict", predict)
    monkeypatch.setattr(development.models, "prefetch", lambda ck, allow_download: None)
    cfg = ForecastConfig.make("Timestral-28M", prediction_target="arome_error", context_days=7,
                              horizon_h=6, targets=("t",))
    out = development.run(source, cfg, root=tmp_path / "results")
    assert calls == ["c1", "c2"]
    availability = pd.read_csv(out / "case_availability.csv").set_index("case_id")
    assert availability.loc["c0", "status"] == "unavailable"
    assert "no observed t errors" in availability.loc["c0", "reason"]
    predictions = pd.read_parquet(out / "predictions.parquet")
    corrected = predictions[predictions.model.str.startswith("timestral-dev") & predictions.point.notna()]
    assert len(corrected) == 4
    assert corrected.arome_baseline.eq(12).all() and corrected.predicted_error.eq(-1).all()
    assert corrected.point.eq(11).all()
