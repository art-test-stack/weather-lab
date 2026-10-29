"""Manually collect the named hourly sources; verified local downloads are reused."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests
import yaml

from weather_lab.common import ROOT, read_json, sha256_file, write_json
from weather_lab.data import _download

HOURLY_PRODUCTS = {"meteofrance_arome_france", "meteofrance_arome_france_hd"}
OBS_BASE = "https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/HOR"
PERIODS = ("previous-2020-2024", "latest-2025-2026")


def collect_file(url: str, dest: Path, records: list[dict], session: requests.Session) -> dict:
    prior = next((r for r in records if (ROOT / r["path"]).resolve() == dest.resolve()), None)
    if dest.exists():
        if prior is None or prior["url"] != url or sha256_file(dest) != prior["sha256"]:
            raise ValueError(f"Unverified or changed cached source: {dest}")
        return prior
    if prior is not None:
        raise FileNotFoundError(f"Recorded source is missing: {dest}; restore it rather than replacing its snapshot")
    staging = ROOT / ".tmp/codex/hourly-archive-v1/downloads" / dest.name
    rec = _download(url, staging, session)
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(dest)
    return {**rec, "path": str(dest)}


def fetch(protocol: Path) -> None:
    cfg = yaml.safe_load(protocol.read_text())
    if not set(cfg["products"]) <= HOURLY_PRODUCTS:
        raise ValueError("This collector supports only the two explicit hourly AROME products")
    start, end = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    if start.year != 2022 or end.year != 2026 or start > end:
        raise ValueError("This snapshot collector requires the declared 2022–2026 archive period")
    raw, evidence = ROOT / cfg["raw_dir"], ROOT / cfg["evidence_dir"]
    paths = {kind: evidence / f"{kind}-downloads.json" for kind in ("raw", "historical")}
    records = {kind: read_json(path) if path.exists() else [] for kind, path in paths.items()}
    session = requests.Session()
    for department in sorted({s["department"] for s in cfg["stations"]}):
        for period in PERIODS:
            name = f"H_{department}_{period}.csv.gz"
            dest = raw / name
            record = collect_file(f"{OBS_BASE}/{name}", dest, records["raw"], session)
            if record not in records["raw"]:
                records["raw"].append(record)
                write_json(paths["raw"], records["raw"])
            print(f"verified {dest}", flush=True)
    # Long date ranges count as multiple API calls; stay below 500 weighted calls/minute.
    spacing = math.ceil(((end - start).days + 1) / 14) * 60 / 500
    last_request = None
    for station in cfg["stations"]:
        for product in cfg["products"]:
            params = {"latitude": station["lat"], "longitude": station["lon"],
                      "start_date": str(start.date()), "end_date": str(end.date()),
                      "hourly": "temperature_2m,pressure_msl", "models": product,
                      "cell_selection": "nearest", "elevation": "nan"}
            url = "https://historical-forecast-api.open-meteo.com/v1/forecast?" + urlencode(params)
            dest = raw / "open-meteo" / f"{station['id']}-{product}-{start:%Y%m%d}-{end:%Y%m%d}.json"
            if not dest.exists():
                if last_request is not None:
                    time.sleep(max(0, spacing - (time.monotonic() - last_request)))
                last_request = time.monotonic()
            record = collect_file(url, dest, records["historical"], session)
            if record not in records["historical"]:
                record |= {"station": station["id"], "national_id": station["national_id"],
                           "product": product, "requested_lat": station["lat"], "requested_lon": station["lon"],
                           "source_kind": "stitched_historical_forecast"}
                records["historical"].append(record)
                write_json(paths["historical"], records["historical"])
            print(f"verified {dest}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmarks/protocols/hourly-archive-v1.yaml")
    fetch(parser.parse_args().protocol)
