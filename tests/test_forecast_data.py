import gzip
import json

import pandas as pd
import pytest
import yaml

from weather_lab import forecast_data as data
from weather_lab.common import sha256_file
from weather_lab.hourly_observations import SOURCE_COLUMNS


def payload(times):
    rows = [["29075001", "BREST", 48.4, -4.4, 90, time, 12.5, "0", 1008., "1", 2., "0", 180., "1"] for time in times]
    return gzip.compress(pd.DataFrame(rows, columns=SOURCE_COLUMNS).to_csv(sep=";", index=False).encode())


@pytest.fixture
def sources(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    history = tmp_path / "data/raw/history.csv.gz"
    history.parent.mkdir(parents=True)
    history.write_bytes(payload(["2022010100", "2024123122"]))
    digest = sha256_file(history)
    inventory = tmp_path / "data/evidence/hourly-archive-v1/raw-downloads.json"
    inventory.parent.mkdir(parents=True)
    inventory.write_text(json.dumps([{"url": f"{data.OBS_BASE}/H_29_previous-2020-2024.csv.gz",
                                     "path": str(history.relative_to(tmp_path)), "sha256": digest, "bytes": history.stat().st_size}]))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"start": "2022-01-01T00:00:00Z", "history_start": "2021-10-01T00:00:00Z",
        "download_cap_mib": 1, "stations": [{"id": "07110", "national_id": "29075001", "department": "29", "name": "BREST"}]}))
    current = payload(["2025010100", "2026010100"])
    calls = []
    class Response:
        headers = {"Content-Length": str(len(current)), "Last-Modified": "today"}
        def raise_for_status(self): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_content(self, size): yield current
    class Session:
        def head(self, url, **kw):
            calls.append(("HEAD", url))
            return Response()
        def get(self, url, **kw):
            calls.append(("GET", url))
            return Response()
    monkeypatch.setattr(data.requests, "Session", Session)
    return config, history, digest, calls


def test_hourly_prepare_fills_missing_year_without_rewriting_history(sources):
    config, history, digest, calls = sources
    path = data.prepare(config, log=lambda _: None)
    frame = pd.read_parquet(path).set_index("time")
    assert frame.loc[pd.Timestamp("2025-01-01T00:00Z"), "t"] == 12.5
    assert frame.index.max() == pd.Timestamp("2026-01-01T00:00Z")
    assert frame.loc[pd.Timestamp("2025-06-01T00:00Z"), ["t", "pmer", "u", "v"]].isna().all()
    assert frame.loc[pd.Timestamp("2024-12-31T23:00Z"), ["t", "pmer", "u", "v"]].isna().all()
    assert frame.loc[pd.Timestamp("2025-01-01T00:00Z"), "QT"] == "0"
    assert sha256_file(history) == digest
    assert [kind for kind, _ in calls] == ["HEAD", "GET"]
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["exploration_only"] and meta["cadence_hours"] == 1
    assert meta["sha256"] == sha256_file(path)
    assert meta["stations"][0]["last"].startswith("2026-01-01")
    assert any(row["year"] == 2025 and row["t"] == 1 for row in meta["coverage"])


def test_hourly_prepare_refuses_changed_history_and_over_budget_downloads(sources):
    config, history, _, calls = sources
    history.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="checksum differs"):
        data.prepare(config)
    assert not calls


def test_hourly_download_preflight_stops_before_get(sources):
    config, _, _, calls = sources
    cfg = yaml.safe_load(config.read_text())
    cfg["download_cap_mib"] = 0
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="exceeding"):
        data.prepare(config)
    assert [kind for kind, _ in calls] == ["HEAD"]
