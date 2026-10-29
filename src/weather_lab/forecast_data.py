"""Explicit hourly exploration data preparation; frozen suite inputs stay untouched."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import requests
import yaml

from weather_lab.common import ROOT, now_utc, read_json, sha256_file, write_json
from weather_lab.hourly_observations import _utc_hour, read_hourly
from weather_lab.rag_archive import OBS_BASE

CONFIG = ROOT / "configs/forecast_hourly.yaml"
OBSERVATIONS = "forecast_hourly_v1.parquet"
ISSUED = "forecast_hourly_arome_v1.parquet"
SUITE = "hourly-explorer-v1"


def make_case(meta, station, origin, issued_times=None):
    """One hourly exploratory origin, with only issued runs available at that time."""
    origin = _utc_hour(origin, "Forecast origin")
    record = next((s for s in meta["stations"] if s["id"] == station), None)
    if record is None:
        raise ValueError("Select a station in the hourly archive")
    first = max(pd.Timestamp(meta["start"]), pd.Timestamp(record["first"]))
    last = pd.Timestamp(record["last"])
    if not first <= origin <= last:
        raise ValueError(f"Hourly observations for this station cover {first} through {last}")
    ident = f"{SUITE}/{station}/{origin:%Y%m%dT%H}"
    case = pd.Series({"case_id": ident, "station": station, "origin": origin,
                      "eligible": True, "split": "exploratory", "arome_run": None,
                      "arome_available_at": None}, name=ident)
    if issued_times is not None and not issued_times.empty:
        available = issued_times[issued_times.station.eq(station) & issued_times.available_at.le(origin)]
        if not available.empty:
            run = available.sort_values("run").iloc[-1]
            case["arome_run"], case["arome_available_at"] = run.run.isoformat(), run.available_at.isoformat()
    return case


def prepare(config=CONFIG, log=print):
    """Refresh explicitly, reusing verified history; never infer, score or publish.

    Missing/QC-rejected values remain missing. Each station ends at its last
    actual observation, not a padded future timestamp.
    """
    cfg = yaml.safe_load(Path(config).read_text())
    now = pd.Timestamp.now(tz="UTC").floor("h")
    if now.year != 2026:
        raise ValueError("The named current-period files cover 2025–2026; update the collection periods deliberately")
    previous = ROOT / "data/evidence/hourly-archive-v1/raw-downloads.json"
    cache = read_json(previous) if previous.exists() else []
    session, plans = requests.Session(), []
    for department in sorted({s["department"] for s in cfg["stations"]}):
        for period in ("previous-2020-2024", "latest-2025-2026"):
            name = f"H_{department}_{period}.csv.gz"
            url = f"{OBS_BASE}/{name}"
            prior = next((r for r in cache if r["url"] == url), None)
            cached = ROOT / prior["path"] if prior else None
            if period.startswith("previous") and cached and cached.is_file():
                if sha256_file(cached) != prior["sha256"]:
                    raise ValueError(f"Historical source checksum differs: {cached}")
                plans.append((department, period, cached, prior))
            else:
                response = session.head(url, timeout=60)
                response.raise_for_status()
                plans.append((department, period, None, {"url": url, "bytes": int(response.headers["Content-Length"])}))
    cap = cfg["download_cap_mib"] * 2**20
    planned = sum(rec["bytes"] for _, _, path, rec in plans if path is None)
    if planned > cap:
        raise ValueError(f"Hourly download needs {planned / 2**20:.1f} MiB, exceeding the {cfg['download_cap_mib']} MiB cap")
    log(f"Hourly exploration: {planned / 2**20:.1f} MiB to download; verified historical sources are reused")
    raw = ROOT / "data/raw/forecast-hourly-v1" / now_utc().replace(":", "")
    raw.mkdir(parents=True, exist_ok=False)
    temp_root = ROOT / ".tmp/codex/forecast-hourly"
    temp_root.mkdir(parents=True, exist_ok=True)
    frames, sources, consumed = [], [], 0
    with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
        stage = Path(tmp)
        for department, period, path, rec in plans:
            if path is None:
                name = rec["url"].rsplit("/", 1)[-1]
                downloading = stage / name
                with session.get(rec["url"], stream=True, timeout=120) as response:
                    response.raise_for_status()
                    with downloading.open("wb") as stream:
                        for block in response.iter_content(1 << 20):
                            consumed += len(block)
                            if consumed > cap:
                                raise ValueError("Hourly download cap exceeded; no exploration dataset was installed")
                            stream.write(block)
                    rec = {"url": rec["url"], "bytes": downloading.stat().st_size,
                           "sha256": sha256_file(downloading), "retrieved_at": now_utc(),
                           "http": {k: response.headers.get(k) for k in ("ETag", "Last-Modified", "Content-Length")}}
                path = raw / name
                downloading.replace(path)
                rec["path"] = str(path.relative_to(ROOT))
            sources.append(rec)
            write_json(raw / "sources.json", sources)
            lo = pd.Timestamp(cfg["history_start"]) if period.startswith("previous") else pd.Timestamp("2025-01-01T00:00Z")
            hi = pd.Timestamp("2024-12-31T23:00Z") if period.startswith("previous") else now
            for station in (s for s in cfg["stations"] if s["department"] == department):
                frame, _ = read_hourly(path, station["id"], station["national_id"], lo, hi)
                if period.startswith("latest"):
                    last = frame.loc[frame.NUM_POSTE.notna(), "time"].max()
                    frame = frame[frame.time.le(last)]
                frames.append(frame)
            log(f"Read hourly department {department} · {period}")
        observations = pd.concat(frames, ignore_index=True).sort_values(["station", "time"])
        if observations.duplicated(["station", "time"]).any():
            raise ValueError("Duplicate station/time in hourly exploration observations")
        stations = []
        for station in cfg["stations"]:
            frame = observations[observations.station.eq(station["id"])]
            valid = frame[frame[["t", "pmer", "u", "v"]].notna().any(axis=1)]
            if valid.empty:
                raise ValueError(f"No usable hourly observations for {station['id']}")
            position = frame.dropna(subset=["LAT", "LON", "ALTI"]).iloc[-1]
            stations.append({**station, "lat": float(position.LAT), "lon": float(position.LON),
                             "altitude_m": float(position.ALTI), "first": str(valid.time.min()), "last": str(valid.time.max())})
        observations.to_parquet(stage / OBSERVATIONS, index=False)
        meta = {"version": 1, "exploration_only": True, "created_at": now_utc(), "start": cfg["start"],
                "sha256": sha256_file(stage / OBSERVATIONS), "sources": sources, "stations": stations,
                "cadence_hours": 1, "quality_rule": "accept QC 0 and 1; retain other samples as missing, never interpolate",
                "observation_availability": "Historical replay assumes observations known at their valid time; original publication/revision timestamps are unavailable.",
                "coverage": observations[observations.time.ge(pd.Timestamp(cfg["start"]))].assign(year=lambda x: x.time.dt.year)
                    .groupby(["station", "year"])[["t", "pmer", "u", "v"]].count().reset_index().to_dict("records")}
        from weather_lab.hourly_benchmark import configuration
        from weather_lab.rag_store import Store

        # Copy verified named issues only; stitched historical series cannot substitute.
        source = ROOT / "data/cache/rag/weather-v1/suites" / configuration()["artifact_id"]
        if (source / "artifact.json").is_file():
            Store.verify(source)
            shutil.copy2(source / "arome_station_extract.parquet", stage / ISSUED)
            meta["arome"] = {"file": ISSUED, "sha256": sha256_file(stage / ISSUED),
                             "source_artifact": configuration()["artifact_id"],
                             "availability": "Named issued runs, with the source archive's assumed 3-hour publication delay"}
        destination = ROOT / "data/processed"
        destination.mkdir(parents=True, exist_ok=True)
        write_json(stage / "forecast_hourly_v1.meta.json", meta)
        for path in stage.iterdir():
            path.replace(destination / path.name)
    log(f"Installed hourly exploration data for {len(stations)} stations; latest record {max(s['last'] for s in stations)}")
    return destination / OBSERVATIONS
