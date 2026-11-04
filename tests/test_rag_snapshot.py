import numpy as np
import pandas as pd
import pytest

from weather_lab.common import read_json, write_json
from weather_lab.prospective_collect import PRODUCT, rows_json
from weather_lab.prospective_store import Archive
from weather_lab.rag_index import METHODS, embedding, load_index
from weather_lab.rag_snapshot import build
from weather_lab.rag_store import Store

ORIGIN = pd.Timestamp("2026-01-03T12:00Z")


def parent_index(tmp_path, kind):
    source = tmp_path / "parent-files"
    source.mkdir()
    origin = pd.Timestamp("2025-12-30T06:00Z")
    meta = {"version": 1, "kind": kind, "variables": ["t", "pmer"], "context_hours": 24,
            "query_hours": 6, "horizon_hours": 5, "observation_delay_hours": 3,
            "start": "2025-01-01T00:00Z", "end_exclusive": "2026-01-01T00:00Z",
            "scale": [2., 5.], "product": PRODUCT, "error_summary_horizon_hours": 2, "forecast_query_hours": 2,
            "retrieval": {"query_hours": 6, "forecast_query_hours": 2, "stride_hours": 1, "k": 1,
                          "scale_end_exclusive": "2026-01-01T00:00Z"},
            "configuration": {"retrieval": {"scale_end_exclusive": "2026-01-01T00:00Z"}}}
    row = {"station": "07149", "origin": origin, "payload_end": origin + pd.Timedelta(hours=2),
           "available_at": origin + pd.Timedelta(hours=5)}
    if kind == "weather":
        row["origin"] = origin - pd.Timedelta(hours=3)
    else:
        row.update(run=origin - pd.Timedelta(hours=3), forecast_available_at=origin - pd.Timedelta(hours=2),
                   mae_t=1., mae_pmer=1., bias_t=0., bias_pmer=0.)
    block = np.stack([10 + np.arange(29) / 10, 1000 + np.arange(29)]).astype(np.float32)
    np.save(source / "payload.npy", block[None], allow_pickle=False)
    for method in METHODS:
        vector = embedding(block[:, 18:24], np.array(meta["scale"], np.float32), method) if kind == "weather" else np.zeros(28, np.float32)
        np.save(source / f"vectors-{method}.npy", vector[None], allow_pickle=False)
    pd.DataFrame([row]).to_parquet(source / "windows.parquet", index=False)
    write_json(source / "index.json", meta)
    return Store(root=tmp_path / "active").put("indexes", source)


def live_archive(tmp_path, *, retrospective=False, late_freeze=False):
    archive = Archive(tmp_path / "archive")
    past_times = pd.date_range(ORIGIN - pd.Timedelta(hours=30), ORIGIN - pd.Timedelta(hours=3), freq="h")
    past = pd.DataFrame({"station": "07149", "time": past_times,
                         "t": 10 + np.sin(np.arange(len(past_times))), "pmer": 1000 + np.cos(np.arange(len(past_times)))})
    future = pd.DataFrame({"station": "07149", "time": pd.date_range(ORIGIN - pd.Timedelta(hours=2), periods=5, freq="h"),
                           "t": 11., "pmer": 1001.})
    with archive.writer():
        archive.put("observations", "history", {"identity": "history", "collected_at": str(ORIGIN - pd.Timedelta(hours=2)),
                                                "rows": rows_json(past)})
        archive.put("observations", "later-history-revision", {"identity": "history", "collected_at": str(ORIGIN + pd.Timedelta(hours=2)),
                                                               "rows": rows_json(past.assign(t=100., pmer=500.))})
        archive.put("observations", "verification", {"identity": "verification", "collected_at": str(ORIGIN + pd.Timedelta(hours=3)),
                                                     "rows": rows_json(future)})
        for run in pd.date_range(ORIGIN - pd.Timedelta(hours=48), ORIGIN - pd.Timedelta(hours=3), freq="3h"):
            available = run + pd.Timedelta(minutes=30)
            rows = [{"station": "07149", "run": run, "available_at": available, "valid_time": run + pd.Timedelta(hours=lead),
                     "variable": variable, "value": value, "product": PRODUCT,
                     "source_kind": "issued_run", "availability_basis": "actual_first_collection"}
                    for lead in range(1, 49) for variable, value in [("t", 10.), ("pmer", 1000.)]]
            archive.put("forecasts", str(run), {"run": str(run), "collected_at": str(available), "rows": rows_json(pd.DataFrame(rows))})
        for model in ("arome-raw", "another-existing-model"):
            record = {"case_id": "live-case", "station": "07149", "origin": str(ORIGIN),
                      "issue": str(ORIGIN - pd.Timedelta(hours=3)), "cohort": "retrospective-2026-01" if retrospective else "prospective-2026-01",
                      "model": model, "status": "available", "provenance": {"decision_cutoff": str(ORIGIN),
                          "finished_at": str(ORIGIN + pd.Timedelta(minutes=60 if late_freeze else 10))},
                      "rows": [{"point": 10., "variable": "t", "lead_h": 1}]}
            archive.put("predictions", model, record)
    return archive


@pytest.mark.parametrize("kind", ["weather", "arome_error"])
def test_snapshot_preserves_parent_scale_and_real_availability(tmp_path, kind):
    parent_path = parent_index(tmp_path, kind)
    parent = load_index(parent_path)  # real v1 index parser, arrays, integrity and schema checks
    before = Store.verify(parent_path)
    archive = live_archive(tmp_path)
    staged = Store(root=tmp_path / "staged")
    clock = lambda: "2026-11-03T00:00:00Z"
    early = build(archive, parent_path, ORIGIN + pd.Timedelta(hours=2), destination=staged, clock=clock)
    assert early["snapshot"] is None and early["added_cases"] == 0
    assert "not fully verified" in early["excluded"][0]["reason"]
    cutoff = ORIGIN + pd.Timedelta(hours=4)
    result = build(archive, parent_path, cutoff, destination=staged, clock=clock)
    assert result["added_cases"] == 1 and result["total_cases"] == 2  # duplicate model results produce one memory case
    path = staged.get("indexes", result["snapshot_id"])
    meta = read_json(path / "index.json")
    assert meta["version"] == 2 and meta["activation"] == "staged_only"
    assert meta["scale"] == parent.meta["scale"]
    assert meta["retrieval"] == parent.meta["retrieval"]
    assert meta["parent_index"] == parent.ident
    windows = pd.read_parquet(path / "windows.parquet")
    assert windows.iloc[-1].available_at == ORIGIN + pd.Timedelta(hours=3)
    assert windows.iloc[-1].payload_end == ORIGIN + pd.Timedelta(hours=2)
    payload = np.load(path / "payload.npy")
    np.testing.assert_array_equal(payload[:1], parent.payload)
    assert np.nanmax(payload[-1, 0, :24]) < 20  # later history revision did not enter the original query
    for method in METHODS:
        vectors = np.load(path / f"vectors-{method}.npy")
        np.testing.assert_array_equal(vectors[:1], parent.vectors[method])
        assert len(vectors) == 2
    assert Store.verify(parent_path) == before
    again = build(archive, parent_path, cutoff, destination=staged, clock=lambda: "2026-11-03T01:00:00Z")
    assert again["snapshot_id"] == result["snapshot_id"]
    assert again["available_at"] == result["available_at"]  # first commit receipt is immutable
    with pytest.raises(ValueError, match="version"):
        load_index(path)  # staged format cannot accidentally enter the frozen v1 adapter
    assert len(Store(root=tmp_path / "active").list("indexes")) == 1


@pytest.mark.parametrize("mode", ["retrospective", "late_freeze"])
def test_snapshot_refuses_cases_without_prospective_freeze(tmp_path, mode):
    parent = parent_index(tmp_path, "weather")
    archive = live_archive(tmp_path, **{mode: True})
    staged = Store(root=tmp_path / "staged")
    result = build(archive, parent, ORIGIN + pd.Timedelta(hours=4), destination=staged)
    assert result["snapshot"] is None and result["added_cases"] == 0
    assert result["excluded"]
    assert not staged.root.exists()


def test_snapshot_cutoff_and_local_only_boundary(tmp_path):
    parent = parent_index(tmp_path, "weather")
    archive = live_archive(tmp_path)
    with pytest.raises(ValueError, match="at or before now"):
        build(archive, parent, "2099-01-01")
    with pytest.raises(ValueError, match="local store"):
        build(archive, parent, ORIGIN + pd.Timedelta(hours=4), destination=Store())
