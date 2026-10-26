"""Collect and validate exact issued runs for the bounded October feature screen."""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

from weather_lab.common import ROOT, read_json, sha256_file, write_json

ENDPOINT = "https://single-runs-api.open-meteo.com/v1/forecast"
FIELDS = {"relative_humidity_2m": "%", "cloud_cover": "%", "shortwave_radiation": "W/m²"}


def validate(response, run, stations, anchors):
    if not isinstance(response, list) or len(response) != len(stations):
        raise ValueError("Expected one response per requested station")
    records, audits, seen = [], [], set()
    for item in response:
        station = str(item["location_id"]).zfill(5)
        if station in seen or station not in stations:
            raise ValueError("Unknown or duplicate station response")
        seen.add(station)
        source = anchors[anchors.station.eq(station) & anchors.run.eq(run)]
        if source.empty:
            raise ValueError("No frozen anchors for the requested station/run")
        grid = source[["grid_lat", "grid_lon"]].drop_duplicates()
        if len(grid) != 1 or not np.allclose(grid.iloc[0], [item["latitude"], item["longitude"]], atol=1e-5, rtol=0):
            raise ValueError("Issued weather grid differs from the frozen baseline cell")
        times = pd.to_datetime(item["hourly"]["time"], utc=True)
        expected = pd.date_range(run, periods=52, freq="h")
        if not times.equals(expected) or item.get("utc_offset_seconds") != 0:
            raise ValueError("Issued run/time axis does not match the request")
        audit = {"station": station, "run": str(run), "grid_lat": item["latitude"], "grid_lon": item["longitude"]}
        for variable, field, unit in (("t", "temperature_2m", "°C"), ("pmer", "pressure_msl", "hPa")):
            if item["hourly_units"][field] != unit:
                raise ValueError("Wrong source anchor units")
            values = pd.Series(item["hourly"][field], index=times, dtype=float)
            old = source[source.variable.eq(variable)].set_index("valid_time").value
            delta = abs(values.reindex(old.index).to_numpy() - old.to_numpy())
            if not np.isfinite(delta).all() or delta.max() > .15:
                raise ValueError("Source anchor differs beyond the frozen decoder-audit tolerance")
            audit[variable + "_max_absolute_decoder_difference"] = float(delta.max())
        for field, unit in FIELDS.items():
            if item["hourly_units"][field] != unit or len(item["hourly"][field]) != 52:
                raise ValueError("Wrong issued-field units or array length")
            values = np.array(item["hourly"][field], dtype=float)
            finite = values[np.isfinite(values)]
            upper = 1500 if field == "shortwave_radiation" else 100
            if np.isinf(values).any() or ((finite < 0) | (finite > upper)).any():
                raise ValueError("Issued field outside physical source-unit bounds")
            for valid_time, value in zip(times, values):
                records.append({"station": station, "run": run, "available_at": run + pd.Timedelta(hours=3),
                    "valid_time": valid_time, "variable": field, "value": value, "unit": unit,
                    "product": "meteofrance_arome_france", "grid_lat": item["latitude"], "grid_lon": item["longitude"]})
        audits.append(audit)
    return records, audits


def main():
    out = ROOT / "data/evidence/error-features-v1"
    folder = out / "sources"
    folder.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/examples/arome-issued-v2"
    manifest = read_json(source / "manifest.json")
    stations = {s["id"]: s for s in manifest["dataset"]["stations"]}
    anchors = pd.read_parquet(source / "arome_station_extract.parquet")
    ledger_path = out / "downloads.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else {}
    records, audits = [], []
    for day in pd.date_range("2026-10-11", "2026-10-21", freq="D", tz="UTC"):
        run = day + pd.Timedelta(hours=3)
        name = run.strftime("%Y%m%dT%H")
        path = folder / f"{name}.json"
        params = {"latitude": ",".join(str(s["lat"]) for s in stations.values()),
                  "longitude": ",".join(str(s["lon"]) for s in stations.values()),
                  "location_id": ",".join(str(int(s)) for s in stations),
                  "run": run.strftime("%Y-%m-%dT%H:%M"), "models": "meteofrance_arome_france",
                  "hourly": "temperature_2m,pressure_msl," + ",".join(FIELDS), "forecast_hours": 52,
                  "elevation": ",".join(["nan"] * len(stations)), "cell_selection": "nearest", "timezone": "UTC"}
        if path.exists():
            if name in ledger:
                assert ledger[name]["params"] == params and ledger[name]["sha256"] == sha256_file(path)
            elif name != "20261011T03":
                raise ValueError("Refusing an unverified pre-existing response")
        else:
            response = requests.get(ENDPOINT, params=params, timeout=45)
            response.raise_for_status()
            if sum(p.stat().st_size for p in folder.glob("*.json")) + len(response.content) > 2 * 1024**2:
                raise RuntimeError("Download cap reached")
            staging = ROOT / ".tmp/codex/error-features-v1/response.json"
            staging.write_bytes(response.content)
            validate(response.json(), run, stations, anchors)
            staging.rename(path)
            time.sleep(1)  # five locations count separately; this small batch remains below free limits
        rows, audit = validate(json.loads(path.read_text()), run, stations, anchors)
        ledger[name] = {"endpoint": ENDPOINT, "params": params, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        write_json(ledger_path, ledger)
        records.extend({**r, "source_sha256": ledger[name]["sha256"]} for r in rows)
        audits.extend(audit)
        print(f"Validated {name}: five stations", flush=True)
    frame = pd.DataFrame(records)
    assert not frame.duplicated(["station", "run", "valid_time", "variable"]).any()
    destination = out / "issued_atmosphere.parquet"
    if destination.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(destination), frame)
    else:
        frame.to_parquet(destination, index=False)
    write_json(out / "source_audit.json", {"stations_runs": audits, "download_bytes": sum(r["bytes"] for r in ledger.values()),
        "fields": FIELDS, "radiation": "backward one-hour mean, W/m²", "availability": "assumed init+3h",
        "issued_sha256": sha256_file(destination), "frozen_anchor_sha256": sha256_file(source / "arome_station_extract.parquet")})


if __name__ == "__main__":
    main()
