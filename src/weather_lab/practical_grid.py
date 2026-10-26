"""Predeclared station extraction sensitivities from installed GRIB fields, with no downloads."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from weather_lab.arome import FIELDS, _grid, _messages, haversine_km, nearest_index
from weather_lab.common import ROOT, read_json, sha256_file, write_json
from weather_lab.practical import PROTOCOL, period_cases, save_outputs


def bilinear_cells(grid: dict, lat: float, lon: float) -> list[tuple[int, int, float]]:
    dj = grid["jDirectionIncrementInDegrees"] * (1 if grid["jScansPositively"] else -1)
    lon0 = (grid["longitudeOfFirstGridPointInDegrees"] + 180) % 360 - 180
    x = (lon - lon0) / grid["iDirectionIncrementInDegrees"]
    y = (lat - grid["latitudeOfFirstGridPointInDegrees"]) / dj
    i, j = int(np.floor(x)), int(np.floor(y))
    if not (0 <= i < grid["Ni"] - 1 and 0 <= j < grid["Nj"] - 1):
        return []
    dx, dy = x - i, y - j
    return [(j, i, (1 - dx) * (1 - dy)), (j, i + 1, dx * (1 - dy)),
            (j + 1, i, (1 - dx) * dy), (j + 1, i + 1, dx * dy)]


def land_cell(grid: dict, mask: np.ndarray, lat: float, lon: float) -> list[tuple[int, int, float]]:
    j0, i0, _, _ = nearest_index(grid, lat, lon)
    dj = grid["jDirectionIncrementInDegrees"] * (1 if grid["jScansPositively"] else -1)
    di = grid["iDirectionIncrementInDegrees"]
    radius = int(np.ceil(5 / (111 * min(abs(dj), abs(di) * np.cos(np.radians(lat)))))) + 1
    lon0 = (grid["longitudeOfFirstGridPointInDegrees"] + 180) % 360 - 180
    candidates = []
    for j in range(max(0, j0 - radius), min(grid["Nj"], j0 + radius + 1)):
        for i in range(max(0, i0 - radius), min(grid["Ni"], i0 + radius + 1)):
            distance = haversine_km(lat, lon, grid["latitudeOfFirstGridPointInDegrees"] + j * dj, lon0 + i * di)
            if mask[j, i] >= .5 and distance <= 5:
                candidates.append((distance, j, i))
    if not candidates:
        return []
    _, j, i = min(candidates)
    return [(j, i, 1.)]


def weighted(values: np.ndarray, cells: list[tuple[int, int, float]]) -> float:
    if not cells or any(not np.isfinite(values[j, i]) for j, i, w in cells if w > 0):
        return np.nan
    return float(sum(values[j, i] * w for j, i, w in cells if w > 0))


def run(out: Path):
    import eccodes

    started, cpu = time.perf_counter(), time.process_time()
    if out.exists():
        raise FileExistsError(out)
    protocol = yaml.safe_load(PROTOCOL.read_text())
    stage_protocol = ROOT / "docs/experiments/practical_improvements_c4_v1.yaml"
    source = ROOT / protocol["source"]["run"]
    manifest = read_json(source / "manifest.json")
    original = pd.read_parquet(ROOT / protocol["source"]["arome"]["file"])
    cases = period_cases(manifest)
    stations = manifest["dataset"]["stations"]
    const = ROOT / "data/raw/arome/constant-eurw1s40.grib2"
    constants, hashes = {}, {str(const): sha256_file(const)}
    for handle in _messages(const):
        grid = _grid(handle)
        constants[eccodes.codes_get(handle, "shortName")] = eccodes.codes_get_values(handle).reshape(grid["Nj"], grid["Ni"])
    choices, metadata = {}, []
    for station in stations:
        lat, lon = station["lat"], station["lon"]
        j, i, _, _ = nearest_index(grid, lat, lon)
        for method, cells in (("nearest", [(j, i, 1.)]), ("bilinear", bilinear_cells(grid, lat, lon)),
                              ("land5km", land_cell(grid, constants["lsm"], lat, lon))):
            choices[(station["id"], method)] = cells
            for jj, ii, w in cells:
                dj = grid["jDirectionIncrementInDegrees"] * (1 if grid["jScansPositively"] else -1)
                glat = grid["latitudeOfFirstGridPointInDegrees"] + jj * dj
                glon = (grid["longitudeOfFirstGridPointInDegrees"] + 180) % 360 - 180 + ii * grid["iDirectionIncrementInDegrees"]
                metadata.append({"station": station["id"], "method": method, "j": jj, "i": ii, "weight": w,
                                 "grid_lat": glat, "grid_lon": glon, "distance_km": haversine_km(lat, lon, glat, glon),
                                 "land_mask": constants["lsm"][jj, ii],
                                 "terrain_m": next(constants[n][jj, ii] for n in ("h", "orog") if n in constants),
                                 "station_altitude_m": station["altitude_m"]})
    records = []
    benchmark = None
    for number, folder in enumerate(sorted((ROOT / "data/raw/arome").glob("*/run.json"))):
        rec = read_json(folder)
        init = pd.Timestamp(rec["run"])
        if init not in set(original.run):
            continue
        c0 = time.process_time()
        hashes[str(folder)] = sha256_file(folder)
        for file in rec["files"]:
            path = folder.parent / file["subset"]
            hashes[str(path)] = sha256_file(path)
            assert hashes[str(path)] == file["subset_sha256"]
            for handle in _messages(path):
                key = tuple(eccodes.codes_get(handle, k, ktype=int) for k in (
                    "discipline", "parameterCategory", "parameterNumber", "typeOfFirstFixedSurface", "scaledValueOfFirstFixedSurface"))
                field = next(n for n, definition in FIELDS.items() if definition["key"] == key)
                assert _grid(handle) == grid, "Constant and forecast grids differ"
                stamp = f"{eccodes.codes_get(handle, 'dataDate'):08d}{eccodes.codes_get(handle, 'dataTime'):04d}"
                assert pd.Timestamp(stamp, tz="UTC") == init
                lead = eccodes.codes_get(handle, "forecastTime", ktype=int)
                values = eccodes.codes_get_values(handle).reshape(grid["Nj"], grid["Ni"])
                for station in stations:
                    for method in ("nearest", "bilinear", "land5km"):
                        val = weighted(values, choices[(station["id"], method)])
                        records.append({"run": init, "station": station["id"], "variable": field,
                                        "lead_h": lead, "valid_time": init + pd.Timedelta(hours=lead),
                                        "available_at": pd.Timestamp(rec["available_at"]), "method": method,
                                        "value": FIELDS[field]["convert"](val)})
        if benchmark is None:
            benchmark = {"runs": 1, "cpu_s": time.process_time() - c0}
            benchmark["projected_cpu_s"] = benchmark["cpu_s"] * original.run.nunique()
            assert benchmark["projected_cpu_s"] < 10 * 60
            print("Grid small-batch estimate:", benchmark, flush=True)
    extracts = pd.DataFrame(records)
    keys = ["station", "run", "variable", "valid_time"]
    check = original.merge(extracts[extracts.method.eq("nearest")], on=keys, suffixes=("", "_rebuilt"), validate="one_to_one")
    assert len(check) == len(original) and np.array_equal(check.value, check.value_rebuilt)
    saved = pd.read_parquet(source / "predictions.parquet")
    raw = saved[saved.model.eq("arome-raw")].copy()
    raw["arome_run"] = pd.to_datetime(raw.arome_run, utc=True)
    frames = [saved[saved.model.isin(["arome-raw", "climatology"])].assign(policy="verified_saved_prediction")]
    for method in ("bilinear", "land5km"):
        a = extracts[extracts.method.eq(method)].rename(columns={"run": "arome_run", "value": "extracted_value"})
        pred = raw.merge(a[["station", "arome_run", "valid_time", "variable", "extracted_value"]],
                         on=["station", "arome_run", "valid_time", "variable"], how="left", validate="one_to_one")
        pred["point"] = pred.extracted_value
        pred["model"] = f"c4-{method}-raw"
        pred["policy"] = np.where(pred.point.notna(), method, "unavailable_no_bounded_extraction")
        frames.append(pred)
    provenance = {"stage": "extraction", "stage_protocol_sha256": sha256_file(stage_protocol),
                  "script_sha256": sha256_file(Path(__file__)), "source_hashes": hashes,
                  "nearest_reproduced_exactly": True, "small_batch": benchmark,
                  "comparison": "raw extraction only; conditioned forecasts from different extractions are not mixed",
                  "cpu_s_before_outputs": time.process_time() - cpu}
    predictions = pd.concat(frames, ignore_index=True)
    predictions["arome_run"] = pd.to_datetime(predictions.arome_run, utc=True)
    save_outputs(out, predictions, pd.read_parquet(source / "targets.parquet"), cases, provenance)
    extracts.to_parquet(out / "extracts.parquet", index=False)
    pd.DataFrame(metadata).to_csv(out / "grid_metadata.csv", index=False)
    assert all(sha256_file(Path(p)) == h for p, h in hashes.items())
    provenance.update(cpu_s=time.process_time() - cpu, wall_s=time.perf_counter() - started)
    write_json(out / "run.json", provenance)
    print("Saved", out, provenance["cpu_s"], "CPU seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
