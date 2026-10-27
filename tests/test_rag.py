"""Scientific boundaries and persistence for the new, independent RAG experiment."""
from __future__ import annotations

import copy
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from weather_lab.common import read_json
from weather_lab.forecast import CaseBatch
from weather_lab.rag import config
from weather_lab.rag_adapters import Analog, foundation
from weather_lab.rag_archive import ingest, issued_forecasts
from weather_lab.rag_evaluate import choose_arome, metrics, run
from weather_lab.rag_index import Index, build
from weather_lab.rag_store import Store


@pytest.fixture
def setup(tmp_path):
    cfg = copy.deepcopy(config())
    cfg.update(context_hours=24, foundation_models=[], stations=[cfg["stations"][0]])
    cfg["retrieval"].update(query_hours=12, k=1, seasonal_days=None)
    times = pd.date_range("2025-11-01", "2026-01-04T23:00", freq="h", tz="UTC")
    x = np.arange(len(times))
    frame = pd.DataFrame({"station": "07110", "time": times,
                          "t": 10 + np.sin(x / 24), "pmer": 1010 + np.cos(x / 72)})
    obs = tmp_path / "source.parquet"
    frame.to_parquet(obs, index=False)
    store = Store(tmp_path / "store")
    archive = ingest(obs, cfg, store=store)
    index = Index(build(archive.name, cfg, store))
    return cfg, frame, store, archive, index


def query(frame, index):
    origin = pd.Timestamp("2026-01-01T06:00Z")
    end = origin - pd.Timedelta(hours=3)
    history = frame.set_index("time").reindex(pd.date_range(end=end, periods=24, freq="h"))[["t", "pmer"]]
    times = pd.date_range(end + pd.Timedelta(hours=1), periods=27, freq="h")
    return CaseBatch(pd.DataFrame([{"case_id": "case", "station": "07110", "origin": origin}]),
                     ["t", "pmer"], history.to_numpy(np.float32, copy=True).T[None],
                     np.array([end.tz_localize(None)], dtype="datetime64[ns]"),
                     times.tz_localize(None).to_numpy()[None], 1)


def test_cutoff_covers_entire_payload_and_scale_ignores_2026(setup, tmp_path):
    cfg, frame, store, archive, index = setup
    assert (index.windows.available_at < pd.Timestamp("2026-01-01T00:00Z")).all()
    assert (index.windows.payload_end + pd.Timedelta(hours=3) == index.windows.available_at).all()
    # Changing held-out labels cannot change the embeddings, scale, payloads or retrieval choices.
    changed = frame.copy()
    changed.loc[changed.time.dt.year == 2026, ["t", "pmer"]] += 100000
    source = tmp_path / "changed.parquet"
    changed.to_parquet(source, index=False)
    other_archive = ingest(source, cfg, store=store)
    other = Index(build(other_archive.name, cfg, store))
    np.testing.assert_array_equal(index.scale, other.scale)
    np.testing.assert_array_equal(index.payload, other.payload)
    for method in ("level", "shape"):
        np.testing.assert_array_equal(index.vectors[method], other.vectors[method])
        a, audit = index.retrieve(query(frame, index), method, 1)
        b, other_audit = other.retrieve(query(frame, index), method, 1)
        np.testing.assert_array_equal(a, b)
        assert audit[0]["neighbors"] == other_audit[0]["neighbors"]
    assert index.windows.origin.dt.hour.eq(3).all()  # index stride matches the delayed query anchor


def test_retrieval_enters_both_native_generation_inputs(setup):
    cfg, frame, store, archive, index = setup
    batch = query(frame, index)
    rows, audit = index.retrieve(batch, "shape", 1)
    np.testing.assert_allclose(rows[:, :, 23], batch.history[:, :, -1], atol=1e-4)
    for name in ("Timestral-209M", "t0-gamma"):
        plain = foundation(name, index)
        rag = foundation(name, index, "shape", 1)
        native, augmented = plain.build_inputs(batch), rag.build_inputs(batch)
        np.testing.assert_array_equal(native[0], augmented[0])
        np.testing.assert_allclose(augmented[1][:, -2:], rows)
        assert augmented[1].shape[1] == native[1].shape[1] + 2
        assert rag.params["index_id"] == index.ident
    result = Analog("analogue", index, "shape", 1).predict(batch)
    result.validate(batch)
    assert result.output == "point" and result.quantiles is None
    batch.history[0, 0, -1] = np.nan
    with pytest.raises(ValueError, match="complete window"):
        index.retrieve(batch, "shape", 1)


def test_gpu_worker_serialization_reopens_index_instead_of_copying_vectors(setup):
    cfg, frame, store, archive, index = setup
    serialized = pickle.dumps(index)
    assert len(serialized) < 1024
    restored = pickle.loads(serialized)
    a, _ = index.retrieve(query(frame, index), "shape", 1)
    b, _ = restored.retrieve(query(frame, index), "shape", 1)
    np.testing.assert_array_equal(a, b)


def test_issued_runs_cannot_be_stitched_or_mixed():
    origin = pd.Timestamp("2026-06-01T06:00Z")
    times = pd.date_range(origin + pd.Timedelta(hours=1), periods=2, freq="h")
    rows = []
    for var in ("t", "pmer"):
        for i, valid in enumerate(times):
            rows.append({"station": "07110", "run": origin - pd.Timedelta(hours=i + 3),
                         "available_at": origin, "valid_time": valid, "variable": var, "value": 1.,
                         "product": "meteofrance_arome_france", "availability_basis": "test_evidence",
                         "source_kind": "issued_run"})
    issued = issued_forecasts(pd.DataFrame(rows))
    with pytest.raises(ValueError, match="single available issued run"):
        choose_arome(issued, "07110", origin, times, ["t", "pmer"], "meteofrance_arome_france")
    issued["source_kind"] = "stitched_historical_forecast"
    with pytest.raises(ValueError, match="stitched"):
        issued_forecasts(issued)
    issued["source_kind"], issued["run"] = "issued_run", origin - pd.Timedelta(hours=3)
    issued["available_at"] = origin + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="single available issued run"):
        choose_arome(issued, "07110", origin, times, ["t", "pmer"], "meteofrance_arome_france")


def test_replay_keeps_whole_year_and_explicit_forecast_gaps(setup):
    cfg, frame, store, archive, index = setup
    path = run(index.ident, cfg, 24, store, log=lambda _: None)
    Store.verify(path)
    rec = read_json(path / "run.json")
    assert rec["requested_cases"] == 365 and rec["development"]
    cases = pd.read_parquet(path / "cases.parquet")
    assert cases.origin.min().month == 1 and cases.origin.max().month == 12
    avail = pd.read_csv(path / "availability.csv")
    assert len(avail) == 365 * 4
    assert not avail.duplicated(["case_id", "model"]).any()
    assert avail[avail.model == "arome-raw"].status.eq("unavailable").all()
    preds = pd.read_parquet(path / "predictions.parquet")
    assert preds.lead_h.min() == 1 and preds.lead_h.max() == 24  # bridge excluded
    assert not preds.duplicated(["case_id", "variable", "lead_h", "model"]).any()
    scores = pd.read_csv(path / "metrics_point.csv")
    assert scores.n.eq(0).all() and scores.mae.isna().all()  # no manufactured AROME efficacy claim
    assert read_json(store.get("runs", path.name) / "run.json") == rec


def test_metrics_match_support_and_keep_units_separate():
    rows = []
    for variable in ("t", "pmer"):
        for case in ("a", "b"):
            for mid, value in [("arome-raw", 2.), ("x", 1.), ("y", 0. if case == "a" else np.nan)]:
                rows.append({"variable": variable, "case_id": case, "lead_h": 1,
                             "model": mid, "point": value, "observed": 0.})
    scores = metrics(pd.DataFrame(rows))
    common = scores[(scores.scope == "all_systems_common") & scores.lead_h.eq(0)]
    assert common.n.eq(1).all() and common.groupby("variable").support_sha256.nunique().eq(1).all()
    pair = scores[(scores.scope == "paired_vs_arome") & scores.lead_h.eq(0) & scores.variable.eq("t")]
    assert pair.set_index("model").loc["x", "n"] == 2 and pair.set_index("model").loc["y", "n"] == 1
    assert len(scores.variable.unique()) == 2


def test_replay_scores_real_matched_rows_and_records_run_availability(setup, tmp_path):
    cfg, frame, store, archive, index = setup
    origin = pd.Timestamp("2026-01-01T06:00Z")
    truth = frame[frame.time.between(origin + pd.Timedelta(hours=1), origin + pd.Timedelta(hours=24))]
    issued = truth.melt(id_vars=["station", "time"], value_vars=["t", "pmer"],
                        var_name="variable", value_name="value").rename(columns={"time": "valid_time"})
    issued["run"], issued["available_at"] = origin - pd.Timedelta(hours=3), origin
    issued["product"], issued["source_kind"] = cfg["product"], "issued_run"
    issued["availability_basis"] = "synthetic_fixture_only"
    p, o = tmp_path / "issued.parquet", tmp_path / "obs.parquet"
    issued.to_parquet(p, index=False)
    frame.to_parquet(o, index=False)
    archive = ingest(o, cfg, issued=p, store=store)
    index_path = build(archive.name, cfg, store)
    result = run(index_path.name, cfg, 24, store, log=lambda _: None)
    scores = pd.read_csv(result / "metrics_point.csv")
    ar = scores[(scores.model == "arome-raw") & scores.lead_h.eq(0)]
    assert ar.n.eq(24).all() and ar.mae.eq(0).all()
    common = scores[(scores.scope == "all_systems_common") & scores.lead_h.eq(0)]
    assert common.n.eq(24).all()
    avail = pd.read_csv(result / "availability.csv")
    ar = avail[(avail.model == "arome-raw") & avail.status.eq("available")]
    assert len(ar) == 1 and "synthetic_fixture_only" in ar.availability_basis.iloc[0]


def test_corruption_and_unsafe_ids_fail_before_use(setup):
    cfg, frame, store, archive, index = setup
    with pytest.raises(ValueError, match="invalid"):
        store.get("archives", "../../escape")
    (index.path / "vectors-shape.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        store.get("indexes", index.ident)


def test_bucket_upload_is_verified_before_remote_commit_marker(tmp_path):
    class Bucket:
        def __init__(self):
            self.files, self.adds = {}, []

        def batch_bucket_files(self, bucket, add):
            for local, remote in add:
                self.files[remote] = Path(local).read_bytes()
                self.adds.append(remote)

        def download_bucket_files(self, bucket, items, **kwargs):
            for remote, local in items:
                Path(local).write_bytes(self.files[remote])

    api = Bucket()
    store = Store(tmp_path / "cache", api)
    store.local_only = False
    source = tmp_path / "source"
    source.mkdir()
    (source / "evidence.txt").write_text("weather evidence")
    path = store.put("archives", source)
    assert api.adds[-1].endswith("/artifact.json")
    ident = path.name
    shutil.rmtree(path)
    restored = store.get("archives", ident)
    assert (restored / "evidence.txt").read_text() == "weather evidence"
    other = tmp_path / "other"
    other.mkdir()
    (other / "new.txt").write_text("must not be committed after corruption")
    original_download = api.download_bucket_files
    def corrupt(bucket, items, **kwargs):
        original_download(bucket, items, **kwargs)
        Path(items[0][1]).write_bytes(b"corrupt")
    api.download_bucket_files = corrupt
    commits_before = len([n for n in api.adds if n.endswith("artifact.json")])
    with pytest.raises(RuntimeError, match="bucket verification failed"):
        store.put("archives", other)
    assert len([n for n in api.adds if n.endswith("artifact.json")]) == commits_before
    assert len(store.list("archives")) == 1  # incomplete uploads are never selectable


def test_research_panel_builds_without_loading_or_running_models(setup, monkeypatch):
    import gradio as gr
    from weather_lab import models
    from weather_lab_app import rag

    cfg, frame, store, archive, index = setup
    def boom(*args, **kwargs):
        raise AssertionError("model operation while browsing RAG controls")
    monkeypatch.setattr(rag, "Store", lambda: store)
    monkeypatch.setattr(models, "prefetch", boom)
    monkeypatch.setattr(models.MODELS, "get", boom)
    with gr.Blocks() as panel:
        with gr.Tabs():
            rag.build()
    assert len(panel.config["dependencies"]) == 6
    assert all(event == "click" for d in panel.config["dependencies"] for _, event in d["targets"])
    assert rag.show(None)[0] == "Select a saved RAG run."
