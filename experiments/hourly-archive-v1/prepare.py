"""Prepare the local hourly archive and stitched historical products as raw evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from weather_lab.common import ROOT, sha256_file, write_json
from weather_lab.hourly_observations import read_hourly

VARIABLES = (("temperature_2m", "t", "°C"), ("pressure_msl", "pmer", "hPa"))
PRODUCTS = {"meteofrance_arome_france", "meteofrance_arome_france_hd"}
RAW_PERIODS = (
    ("previous-2020-2024", None, pd.Timestamp("2024-12-31 23:00", tz="UTC")),
    ("latest-2025-2026", pd.Timestamp("2025-01-01 00:00", tz="UTC"), None),
)


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _utc_hour(value: Any, label: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    result = result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")
    if result.minute or result.second or result.microsecond or result.nanosecond:
        raise ValueError(f"protocol {label} must be on an exact UTC hour")
    return result


def _manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError(f"{path} must contain a list of records")
    return records


def _verified_path(record: dict, directory: Path, label: str) -> tuple[Path, str]:
    if not isinstance(record.get("path"), str) or not isinstance(record.get("sha256"), str):
        raise ValueError(f"{label} manifest record requires path and sha256")
    path = _path(record["path"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"{label} source path is outside {directory}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} source is missing: {path}")
    actual = sha256_file(path)
    if actual != record["sha256"].lower():
        raise ValueError(f"{label} SHA-256 mismatch for {path}: expected {record['sha256']}, got {actual}")
    return path, actual


def _raw_files(records: list[dict], raw_dir: Path, stations: list[dict], start: pd.Timestamp,
               end: pd.Timestamp) -> dict[tuple[str, str], tuple[Path, str, str]]:
    departments = sorted({str(station["department"]) for station in stations})
    expected = {}
    for department in departments:
        for period, period_start, period_end in RAW_PERIODS:
            lo = start if period_start is None else max(start, period_start)
            hi = end if period_end is None else min(end, period_end)
            if lo <= hi:
                name = f"H_{department}_{period}.csv.gz"
                expected[(department, period)] = (raw_dir / name).resolve()

    seen: set[Path] = set()
    found = {}
    for record in records:
        if not isinstance(record.get("url"), str):
            raise ValueError("raw archive manifest record requires url provenance")
        path, digest = _verified_path(record, raw_dir, "raw archive")
        if path in seen:
            raise ValueError(f"duplicate raw archive manifest path: {path}")
        seen.add(path)
        match = next((key for key, expected_path in expected.items() if path == expected_path), None)
        if match is None:
            raise ValueError(f"unknown raw archive manifest path: {path}")
        found[match] = (path, digest, record["url"])
    missing = set(expected) - set(found)
    if missing:
        raise ValueError(f"raw archive manifest is missing expected files: {sorted(missing)}")
    return found


def _historical_files(records: list[dict], raw_dir: Path, stations: list[dict],
                      products: list[str]) -> dict[tuple[str, str], tuple[dict, Path, str]]:
    station_ids = {station["id"] for station in stations}
    coordinates = {station["id"]: (station["lat"], station["lon"]) for station in stations}
    expected = {(sid, product) for sid in station_ids for product in products}
    found = {}
    for record in records:
        sid, product = record.get("station"), record.get("product")
        key = (sid, product)
        if key not in expected:
            raise ValueError(f"unknown historical station/product record: {key}")
        if key in found:
            raise ValueError(f"duplicate historical station/product record: {key}")
        if (record.get("requested_lat"), record.get("requested_lon")) != coordinates[sid]:
            raise ValueError(f"historical requested coordinates differ from protocol for {sid}")
        if not isinstance(record.get("url"), str):
            raise ValueError(f"historical record {key} requires url provenance")
        path, digest = _verified_path(record, raw_dir, f"historical {key}")
        found[key] = (record, path, digest)
    missing = expected - set(found)
    if missing:
        raise ValueError(f"historical manifest is missing station/product records: {sorted(missing)}")
    return found


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _parse_historical(path: Path, source_sha256: str, station: str, product: str,
                      record: dict, expected_axis: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or isinstance(data.get("utc_offset_seconds"), bool) \
            or data.get("utc_offset_seconds") != 0:
        raise ValueError(f"{path}: utc_offset_seconds must be 0")
    grid_lat = _finite_number(data.get("latitude"), f"{path}: latitude")
    grid_lon = _finite_number(data.get("longitude"), f"{path}: longitude")
    hourly, units = data.get("hourly"), data.get("hourly_units")
    if not isinstance(hourly, dict) or not isinstance(units, dict):
        raise ValueError(f"{path}: hourly and hourly_units objects are required")
    raw_times = hourly.get("time")
    if not isinstance(raw_times, list) or len(raw_times) != len(expected_axis):
        raise ValueError(f"{path}: time array length must match protocol hourly axis ({len(expected_axis)})")
    try:
        times = pd.DatetimeIndex(pd.to_datetime(raw_times, utc=True, errors="coerce"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid hourly time axis") from exc
    if times.isna().any() or not times.is_unique or not times.is_monotonic_increasing \
            or not times.equals(expected_axis):
        raise ValueError(f"{path}: time axis must be unique, ascending, hourly, and match the protocol")

    blocks = []
    for source_variable, variable, expected_unit in VARIABLES:
        if units.get(source_variable) != expected_unit:
            raise ValueError(f"{path}: {source_variable} unit must be {expected_unit!r}")
        values = hourly.get(source_variable)
        if not isinstance(values, list) or len(values) != len(times):
            raise ValueError(f"{path}: {source_variable} values must match the hourly time axis length")
        parsed = [np.nan if value is None else _finite_number(value, f"{path}: {source_variable} value")
                  for value in values]
        blocks.append(pd.DataFrame({
            "station": station,
            "time": times,
            "product": product,
            "variable": variable,
            "value": parsed,
            "source_kind": "stitched_historical_forecast",
            "grid_lat": grid_lat,
            "grid_lon": grid_lon,
            "source_sha256": source_sha256,
        }))
    requested_lat = _finite_number(record.get("requested_lat"), f"{path}: requested_lat")
    requested_lon = _finite_number(record.get("requested_lon"), f"{path}: requested_lon")
    request = {"station": station, "product": product, "requested_lat": requested_lat,
               "requested_lon": requested_lon, "grid_lat": grid_lat, "grid_lon": grid_lon,
               "url": record["url"], "path": _display_path(path), "sha256": source_sha256}
    return pd.concat(blocks, ignore_index=True), request


def _finite_counts(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    rows = frame.assign(year=frame["time"].dt.year)
    keys = ["station", "year", "product", "variable"]
    counts = rows.assign(finite=np.isfinite(rows["value"].to_numpy(dtype=float)))
    return [{**{key: record[key] for key in keys}, "count": int(record["finite"])}
            for record in counts.groupby(keys, dropna=False)["finite"].sum().reset_index().to_dict("records")]


def _observation_counts(frame: pd.DataFrame) -> list[dict]:
    values = ["t", "pmer", "u", "v", "ff", "dd"]
    rows = frame.assign(year=frame["time"].dt.year)
    records = []
    for (station, year), group in rows.groupby(["station", "year"], sort=True):
        for variable in values:
            records.append({"station": station, "year": int(year), "variable": variable,
                            "count": int(np.isfinite(group[variable].to_numpy(dtype=float)).sum())})
    return records


def prepare(protocol_path: Path, output_dir: Path | None = None) -> Path:
    protocol_path = Path(protocol_path)
    if not protocol_path.is_absolute():
        protocol_path = ROOT / protocol_path
    with open(protocol_path, encoding="utf-8") as stream:
        protocol = yaml.safe_load(stream)
    raw_dir, evidence_dir = _path(protocol["raw_dir"]), _path(protocol["evidence_dir"])
    out_dir = _path(output_dir if output_dir is not None else protocol["output_dir"])
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {out_dir}")

    start, end = _utc_hour(protocol["start"], "start"), _utc_hour(protocol["end"], "end")
    if start > end:
        raise ValueError("protocol start must be at or before end")
    expected_axis = pd.date_range(start, end, freq="h")
    stations = protocol["stations"]
    products = protocol["products"]
    if not isinstance(stations, list) or not stations or any(not isinstance(station, dict) for station in stations):
        raise ValueError("protocol stations must be a nonempty list of station records")
    if not isinstance(products, list) or not products or any(not isinstance(product, str) for product in products):
        raise ValueError("protocol products must be a nonempty list of product names")
    if not set(products) <= PRODUCTS:
        raise ValueError("unsupported product: this preparation accepts only named hourly AROME France/HD")
    station_ids = [station["id"] for station in stations]
    if len(station_ids) != len(set(station_ids)):
        raise ValueError("protocol contains duplicate station ids")
    if len(products) != len(set(products)):
        raise ValueError("protocol contains duplicate products")

    raw_manifest = _manifest(evidence_dir / "raw-downloads.json")
    historical_manifest = _manifest(evidence_dir / "historical-downloads.json")
    raw_files = _raw_files(raw_manifest, raw_dir, stations, start, end)
    historical_files = _historical_files(historical_manifest, raw_dir, stations, products)

    observations, observation_reports = [], []
    source_hashes = {}
    for station in stations:
        for period, period_start, period_end in RAW_PERIODS:
            lo = start if period_start is None else max(start, period_start)
            hi = end if period_end is None else min(end, period_end)
            if lo > hi:
                continue
            path, digest, url = raw_files[(str(station["department"]), period)]
            frame, report = read_hourly(path, station["id"], station["national_id"], lo, hi)
            observations.append(frame)
            observation_reports.append({"station": station["id"], "national_id": station["national_id"],
                                         "department": station["department"], "period": period,
                                         "path": _display_path(path), "sha256": digest, "url": url,
                                         "parser": report})
            source_hashes[str(path)] = digest
    observation_frame = pd.concat(observations, ignore_index=True)
    if observation_frame.duplicated(["station", "time"]).any():
        raise ValueError("hourly archive periods contain duplicate station/time rows")

    historical_frames, requests = [], []
    for station in stations:
        for product in products:
            record, path, digest = historical_files[(station["id"], product)]
            frame, request = _parse_historical(path, digest, station["id"], product, record, expected_axis)
            historical_frames.append(frame)
            requests.append(request)
            source_hashes[str(path)] = digest
    historical_frame = pd.concat(historical_frames, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    obs_path, hist_path = out_dir / "observations.parquet", out_dir / "historical.parquet"
    observation_frame.to_parquet(obs_path, index=False)
    historical_frame.to_parquet(hist_path, index=False)
    output_hashes = {obs_path.name: sha256_file(obs_path), hist_path.name: sha256_file(hist_path)}
    provenance = {"protocol_sha256": sha256_file(protocol_path),
                  "preparer_sha256": sha256_file(Path(__file__)), "pandas": pd.__version__}
    write_json(out_dir / "observations.meta.json", {
        **provenance,
        "time_axis": {"start": start.isoformat(), "end": end.isoformat(), "step": "1h UTC"},
        "stations": stations,
        "source_reports": observation_reports,
        "source_hashes": {report["path"]: report["sha256"] for report in observation_reports},
        "output_sha256": output_hashes[obs_path.name],
    })
    write_json(out_dir / "historical.meta.json", {
        **provenance,
        "time_axis": {"start": start.isoformat(), "end": end.isoformat(), "step": "1h UTC"},
        "requests": requests,
        "units": {"t": "degC", "pmer": "hPa"},
        "source_kind": "stitched_historical_forecast",
        "provenance_limitation": "These stitched historical forecasts have no issued-run or publication timestamps and must not be treated as future covariates available at the observation origin.",
        "output_sha256": output_hashes[hist_path.name],
    })
    scope_start = max(start, pd.Timestamp("2026-01-01", tz="UTC"))
    scope_end = min(end, pd.Timestamp("2026-12-31T23:00:00Z"))
    write_json(out_dir / "availability.json", {
        **provenance,
        "observation_finite_counts": _observation_counts(observation_frame),
        "historical_finite_counts": _finite_counts(historical_frame),
        "source_hashes": {_display_path(Path(path)): digest for path, digest in source_hashes.items()},
        "output_hashes": output_hashes,
        "requested_2026_scope": {
            "included": scope_start <= scope_end,
            "start": scope_start.isoformat() if scope_start <= scope_end else None,
            "end": scope_end.isoformat() if scope_start <= scope_end else None,
            "claim": "requested collection scope only; no validation/test split or scoring claim",
        },
    })
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("benchmarks/protocols/hourly-archive-v1.yaml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    print(prepare(args.protocol, args.output_dir))


if __name__ == "__main__":
    main()
