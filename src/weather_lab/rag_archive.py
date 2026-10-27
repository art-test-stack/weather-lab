"""Collect/ingest weather evidence without turning stitched series into issued runs."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pandas as pd
import requests

from weather_lab.common import ROOT, now_utc, sha256_file, ts, write_json
from weather_lab.hourly_observations import read_hourly
from weather_lab.rag_store import Store

PRODUCTS = {"meteofrance_arome_france", "meteofrance_arome_france_hd"}
FIELDS = {"temperature_2m": ("t", "°C"), "pressure_msl": ("pmer", "hPa")}
OBS_BASE = "https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/HOR"


def observations(frame: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    if not {"station", "time", *variables} <= set(frame):
        raise ValueError("observations need station, time and configured variables in benchmark units")
    frame["time"] = pd.to_datetime(frame.time, utc=True)
    if not frame.station.map(lambda s: isinstance(s, str) and len(s) == 5 and s.isdigit()).all():
        raise ValueError("observation station ids must be five-character strings")
    if frame.time.isna().any() or (frame.time != frame.time.dt.floor("h")).any():
        raise ValueError("observations must have exact UTC hourly timestamps")
    if frame.duplicated(["station", "time"]).any():
        raise ValueError("duplicate observation station/time")
    return frame.sort_values(["station", "time"]).reset_index(drop=True)


def issued_forecasts(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    required = {"station", "run", "available_at", "valid_time", "variable", "value", "product",
                "availability_basis", "source_kind"}
    if not required <= set(frame):
        raise ValueError("issued forecasts require " + ", ".join(sorted(required)))
    if not frame["product"].isin(PRODUCTS).all() or not frame.source_kind.eq("issued_run").all():
        raise ValueError("only explicitly named AROME issued runs qualify; stitched series cannot be substituted")
    for col in ("run", "available_at", "valid_time"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
        if frame[col].isna().any():
            raise ValueError(f"issued forecast missing {col}")
    if (frame.available_at < frame.run).any() or (frame.valid_time < frame.run).any():
        raise ValueError("issued forecast availability/validity precedes its run")
    if frame.availability_basis.isna().any() or frame.availability_basis.eq("").any():
        raise ValueError("issued forecast availability needs evidence or an explicit replay assumption")
    if frame.duplicated(["station", "product", "run", "valid_time", "variable"]).any():
        raise ValueError("duplicate issued forecast values")
    return frame


def ingest(obs: Path, cfg: dict, issued: Path | None = None, historical: Path | None = None,
           store: Store | None = None) -> Path:
    store = store or Store()
    frame = observations(pd.read_parquet(obs), cfg["variables"])
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        frame.to_parquet(out / "observations.parquet", index=False)
        sources = {str(obs): sha256_file(obs)}
        if issued:
            issued_forecasts(pd.read_parquet(issued)).to_parquet(out / "issued.parquet", index=False)
            sources[str(issued)] = sha256_file(issued)
        if historical:
            history = pd.read_parquet(historical)
            if not history.source_kind.eq("stitched_historical_forecast").all():
                raise ValueError("historical series must be labelled stitched_historical_forecast")
            history.to_parquet(out / "historical.parquet", index=False)
            sources[str(historical)] = sha256_file(historical)
        write_json(out / "archive.json", {"version": 1, "created_at": now_utc(), "sources_sha256": sources,
                   "units": {"t": "°C", "pmer": "hPa", "u": "m/s", "v": "m/s"},
                   "observation_availability": cfg["observation_availability"],
                   "observation_delay_hours": cfg["observation_delay_hours"],
                   "source_role": "pre-2026 observations supply measured analogue continuations; issued AROME is separate",
                   "coverage": frame.assign(year=frame.time.dt.year).groupby(["station", "year"])[cfg["variables"]]
                       .count().reset_index().to_dict("records")})
        return store.put("archives", out)


def _parse(data: dict, station: dict, product: str, source: dict, run=None, available=None, basis=None):
    if data.get("utc_offset_seconds") != 0:
        raise ValueError("forecast API must return UTC")
    times = pd.to_datetime(data["hourly"]["time"], utc=True)
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("forecast time axis must be unique and increasing")
    frames = []
    for name, (var, unit) in FIELDS.items():
        if data["hourly_units"].get(name) != unit:
            raise ValueError(f"unexpected forecast unit for {name}")
        frame = pd.DataFrame({"station": station["id"], "product": product, "variable": var,
                              "valid_time" if run is not None else "time": times,
                              "value": data["hourly"][name], "source_sha256": source["sha256"],
                              "grid_lat": data["latitude"], "grid_lon": data["longitude"],
                              "source_kind": "issued_run" if run is not None else "stitched_historical_forecast"})
        if run is not None:
            frame["run"], frame["available_at"], frame["availability_basis"] = run, available, basis
            frame = frame[frame.valid_time >= run]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def collect(cfg: dict, store: Store | None = None, log=print) -> Path:
    """Explicit collection action. Bound bytes; preserve failed forecast requests as evidence."""
    store = store or Store()
    start = ts(cfg["retrieval"]["start"])
    end = min(ts(cfg["evaluation"]["end_exclusive"]) - pd.Timedelta(hours=1), pd.Timestamp.now(tz="UTC").floor("h"))
    raw = ROOT / "data/raw/rag-weather-v1" / now_utc().replace(":", "")
    raw.mkdir(parents=True, exist_ok=False)
    session, consumed, evidence = requests.Session(), 0, []
    cap = cfg["collection"]["download_cap_mib"] * 2**20

    def download(url, name, params=None):
        nonlocal consumed
        response = session.get(url, params=params, timeout=120, stream=True)
        with response:
            response.raise_for_status()
            dest = raw / name
            with open(dest, "wb") as stream:
                for block in response.iter_content(1 << 20):
                    consumed += len(block)
                    if consumed > cap:
                        raise RuntimeError("RAG collection download cap exceeded; sources retained locally")
                    stream.write(block)
            rec = {"url": response.url, "sha256": sha256_file(dest), "path": str(dest),
                   "retrieved_at": now_utc(), "bytes": dest.stat().st_size}
            evidence.append(rec)
            write_json(raw / "sources.json", evidence)
            return dest, rec

    frames = []
    for department in sorted({s["department"] for s in cfg["stations"]}):
        for period, lo, hi in [("previous-2020-2024", start, ts("2024-12-31T23:00Z")),
                               ("latest-2025-2026", ts("2025-01-01"), end)]:
            name = f"H_{department}_{period}.csv.gz"
            path, rec = download(f"{OBS_BASE}/{name}", name)
            for station in [s for s in cfg["stations"] if s["department"] == department]:
                frame, _ = read_hourly(path, station["id"], station["national_id"], lo, hi)
                frames.append(frame)
            log(f"downloaded {name}")
    obs = raw / "observations.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(obs, index=False)
    historical, issued, failures = [], [], []
    for station in cfg["stations"]:
        for product in cfg["collection"]["products"]:
            if product not in PRODUCTS:
                raise ValueError("collection needs an explicit supported AROME product")
            params = {"latitude": station["lat"], "longitude": station["lon"],
                      "hourly": ",".join(FIELDS), "models": product, "cell_selection": "nearest", "elevation": "nan"}
            # Keep each year separate; unavailable early forecast years remain explicit.
            for year in range(start.year, end.year + 1):
                time.sleep(cfg["collection"]["historical_spacing_seconds"])
                try:
                    path, rec = download("https://historical-forecast-api.open-meteo.com/v1/forecast",
                                         f"historical-{station['id']}-{product}-{year}.json",
                                         params | {"start_date": f"{year}-01-01",
                                                   "end_date": str(min(end, ts(f"{year}-12-31T23:00Z")).date())})
                    historical.append(_parse(json.loads(path.read_text()), station, product, rec))
                except requests.HTTPError as exc:
                    if exc.response.status_code not in (400, 404):
                        raise
                    failures.append({"station": station["id"], "product": product, "year": year,
                                     "kind": "historical", "reason": str(exc)})
            if not cfg["collection"].get("issued_runs", False):
                continue
            first = max(ts(cfg["evaluation"]["start"]), ts("2026-04-02"))
            for day in pd.date_range(first.floor("D"), end.floor("D"), freq="D"):
                run = day + pd.Timedelta(hours=cfg["collection"]["run_hour_utc"])
                available = run + pd.Timedelta(hours=cfg["collection"]["run_availability_lag_hours"])
                if available > end:
                    continue
                time.sleep(cfg["collection"]["request_spacing_seconds"])
                try:
                    path, rec = download("https://single-runs-api.open-meteo.com/v1/forecast",
                                         f"issued-{station['id']}-{product}-{run:%Y%m%dT%H}.json",
                                         params | {"run": run.strftime("%Y-%m-%dT%H:%M"), "forecast_days": 3})
                    issued.append(_parse(json.loads(path.read_text()), station, product, rec, run, available,
                                         f"assumed_run_plus_{cfg['collection']['run_availability_lag_hours']}h"))
                except requests.HTTPError as exc:
                    if exc.response.status_code not in (400, 404):
                        raise
                    failures.append({"station": station["id"], "product": product, "run": str(run),
                                     "kind": "issued", "reason": str(exc)})
    # Include download evidence and raw source bytes in the persisted archive.
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        import shutil

        shutil.copytree(raw, out / "raw")
        observations(pd.read_parquet(obs), cfg["variables"]).to_parquet(out / "observations.parquet", index=False)
        if historical:
            pd.concat(historical, ignore_index=True).to_parquet(out / "historical.parquet", index=False)
        if issued:
            issued_forecasts(pd.concat(issued, ignore_index=True)).to_parquet(out / "issued.parquet", index=False)
        write_json(out / "archive.json", {"version": 1, "created_at": now_utc(), "configuration": cfg,
                   "sources": evidence, "forecast_gaps": failures, "observation_availability": cfg["observation_availability"],
                   "observation_delay_hours": cfg["observation_delay_hours"],
                   "issued_start_limitation": "Open-Meteo named runs start 2026-04-02; January–March remain unavailable",
                   "issued_availability": "Assumed run lag; original publication/revision times are not established"})
        return store.put("archives", out)
