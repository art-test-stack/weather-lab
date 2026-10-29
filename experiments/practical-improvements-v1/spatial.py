"""Explicit bounded spatial screen. Run with --prepare, then --infer after a compute slot grant."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import resource
import time

os.environ["WEATHER_LAB_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import yaml

from weather_lab import common
from weather_lab.common import environment, load_config, read_json, sha256_file, write_json
from weather_lab.practical_spatial import DEFINITION, ROWS, SpatialTimestral, cell_coords, patch_cells, patch_statistics


def prepare(root: Path, out: Path):
    import eccodes
    from weather_lab.arome import FIELDS, _grid, _messages, haversine_km

    cpu, wall = time.process_time(), time.perf_counter()
    protocol_path = root / "docs/experiments/practical_improvements_v1.yaml"
    parent = yaml.safe_load(protocol_path.read_text())
    source = root / parent["source"]["run"]
    original_path = root / parent["source"]["arome"]["file"]
    original = pd.read_parquet(original_path)
    manifest = read_json(source / "manifest.json")
    hashes = {str(source / n): sha256_file(source / n) for n in parent["source"]["hashes"]}
    assert all(hashes[str(source / n)] == h for n, h in parent["source"]["hashes"].items())
    assert sha256_file(original_path) == parent["source"]["arome"]["sha256"]
    out.mkdir(parents=True, exist_ok=False)
    stage = {"stage": "spatial-information-v1", "parent_protocol_sha256": sha256_file(protocol_path),
             "definition": DEFINITION, "context_days": 30, "calendar": True, "target": "raw weather",
             "cases": "same60 frozen October origins; warmup10-13,purge14-15,selection16-21",
             "comparators": ["original-nearest raw/corrected/conditioned209M", "direct patchmean raw/corrected"],
             "candidate": "original-nearest point+four patch statistics conditioned209M",
             "checkpoint": "mistralai/Timestral-209M@9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
             "budget_cpu_s": 480, "confirmation": "none", "source_hashes": hashes,
             "runner_sha256": sha256_file(Path(__file__)),
             "module_sha256": sha256_file(Path(__file__).parents[2] / "src/weather_lab/practical_spatial.py")}
    (out / "protocol.yaml").write_text(yaml.safe_dump(stage, sort_keys=False))
    const = root / "data/raw/arome/constant-eurw1s40.grib2"
    constants = {}
    hashes[str(const)] = sha256_file(const)
    for handle in _messages(const):
        grid = _grid(handle)
        constants[eccodes.codes_get(handle, "shortName")] = eccodes.codes_get_values(handle).reshape(grid["Nj"], grid["Ni"])
    choices, metadata = {}, []
    for s in manifest["dataset"]["stations"]:
        cells = patch_cells(grid, s["lat"], s["lon"])
        choices[s["id"]] = cells
        for j, i in cells:
            lat, lon = cell_coords(grid, j, i)
            metadata.append(dict(station=s["id"], j=j, i=i, lat=lat, lon=lon,
                                 distance_km=haversine_km(s["lat"], s["lon"], lat, lon),
                                 land_fraction=constants["lsm"][j, i],
                                 terrain_m=next(constants[n][j, i] for n in ("h", "orog") if n in constants)))
    records, pilot = [], None
    for folder in sorted((root / "data/raw/arome").glob("*/run.json")):
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
            for h in _messages(path):
                key = tuple(eccodes.codes_get(h, k, ktype=int) for k in (
                    "discipline", "parameterCategory", "parameterNumber", "typeOfFirstFixedSurface", "scaledValueOfFirstFixedSurface"))
                if key != FIELDS["t"]["key"]:
                    continue
                assert _grid(h) == grid
                stamp = f"{eccodes.codes_get(h, 'dataDate'):08d}{eccodes.codes_get(h, 'dataTime'):04d}"
                assert pd.Timestamp(stamp, tz="UTC") == init
                lead = eccodes.codes_get(h, "forecastTime", ktype=int)
                values = FIELDS["t"]["convert"](eccodes.codes_get_values(h).reshape(grid["Nj"], grid["Ni"]))
                for station, cells in choices.items():
                    for name, value in zip(ROWS, patch_statistics(values, grid, cells)):
                        records.append(dict(station=station, run=init, variable=name, value=value,
                                            valid_time=init + pd.Timedelta(hours=lead), lead_h=lead,
                                            available_at=pd.Timestamp(rec["available_at"])))
        if pilot is None:
            pilot = time.process_time() - c0
            assert pilot * original.run.nunique() < 60
    frame = pd.DataFrame(records)
    assert len(frame) == len(original[original.variable.eq("t")]) * 4
    frame.to_parquet(out / "spatial_inputs.parquet", index=False)
    pd.DataFrame(metadata).to_csv(out / "cell_metadata.csv", index=False)
    write_json(out / "preparation.json", {"source_hashes": hashes, "cpu_s": time.process_time() - cpu,
                                          "wall_s": time.perf_counter() - wall, "pilot_one_run_cpu_s": pilot,
                                          "unavailable_stations": [s for s, c in choices.items() if not c]})
    assert all(sha256_file(Path(p)) == h for p, h in hashes.items())
    print("Prepared", out, flush=True)


def infer(root: Path, out: Path):
    import torch
    from weather_lab import models
    from weather_lab.evaluate import make_batch
    from weather_lab.forecast import AromeBiasCorrected, Timestral
    from weather_lab.practical import period_cases, save_outputs
    from weather_lab.suite import cases_from_manifest

    cpu, wall = time.process_time(), time.perf_counter()
    torch.set_num_threads(4)
    stage = yaml.safe_load((out / "protocol.yaml").read_text())
    assert stage["runner_sha256"] == sha256_file(Path(__file__))
    assert stage["module_sha256"] == sha256_file(Path(__file__).parents[2] / "src/weather_lab/practical_spatial.py")
    parent = yaml.safe_load((root / "docs/experiments/practical_improvements_v1.yaml").read_text())
    source = root / parent["source"]["run"]
    assert all(sha256_file(Path(p)) == h for p, h in read_json(out / "preparation.json")["source_hashes"].items())
    manifest = read_json(source / "manifest.json")
    cases, cfg = cases_from_manifest(manifest), load_config(source / "config.yaml")
    obs_path = root / parent["source"]["observations"]["file"]
    assert sha256_file(obs_path) == parent["source"]["observations"]["sha256"]
    obs = pd.read_parquet(obs_path)
    original = pd.read_parquet(root / parent["source"]["arome"]["file"])
    patch = pd.read_parquet(out / "spatial_inputs.parquet")
    saved = pd.read_parquet(source / "predictions.parquet")
    frames = [saved[saved.model.isin(["arome-raw", "arome-biascorr", "climatology", "timestral-209m-mv-cal",
                                    "timestral-209m-mv-cal-arome-archived"])].assign(policy="verified_saved_identical_inputs")]
    raw_t = saved[saved.model.eq("arome-raw") & saved.variable.eq("t")].copy()
    raw_t["arome_run"] = pd.to_datetime(raw_t.arome_run, utc=True)
    patch_mean = patch[patch.variable.eq(ROWS[0])].assign(variable="t")
    key = ["station", "variable", "valid_time"]
    direct = raw_t.drop(columns="point").merge(
        patch_mean.rename(columns={"run": "arome_run", "value": "point"})[key + ["arome_run", "point"]],
        on=key + ["arome_run"], how="left", validate="one_to_one")
    frames.append(direct.assign(model="spatial-patchmean-arome", policy="direct_patch_mean"))
    mean_archive = original[original.variable.ne("t")].copy()
    mean_archive = pd.concat([mean_archive, patch_mean], ignore_index=True)
    mean_batch, _ = make_batch(cfg, obs, cases, 240, mean_archive)
    corrector = AromeBiasCorrected("spatial-patchmean-biascorr", cfg["variables"], ["t"])
    corrector.attach_arome(mean_archive)
    frames.append(corrector.predict(mean_batch).to_frame(mean_batch, "development").assign(policy="patchmean_existing_causal_corrector"))
    batch, _ = make_batch(cfg, obs, cases, 240, original)
    ck = models.CHECKPOINTS["Timestral-209M"]
    models.prefetch(ck, allow_download=False)
    kwargs = dict(variables=cfg["variables"], targets=cfg["variables"], checkpoint=ck.repo, revision=ck.revision,
                  context_days=30, calendar=True, arome="archived", arome_fields=["t", "pmer"])
    baseline = Timestral("timestral-209m-mv-cal-arome-archived", **kwargs)
    c0 = time.process_time()
    reproduced = baseline.predict(batch)
    reference = saved[saved.model.eq(baseline.id)].sort_values(["case_id", "variable", "lead_h"])
    rebuilt = reproduced.to_frame(batch, "development").sort_values(["case_id", "variable", "lead_h"])
    qcols = [c for c in reference if c.startswith("q") and c[1:2].isdigit()]
    assert qcols
    baseline_delta = float(np.max(abs(reference[qcols].to_numpy() - rebuilt[qcols].to_numpy())))
    assert baseline_delta == 0
    baseline_cpu = time.process_time() - c0
    candidate = SpatialTimestral("spatial-patch-209m-v1", spatial_frame=patch, **kwargs)
    c0 = time.process_time()
    pilot = candidate.predict(batch.subset(slice(0, 3)))
    pilot.validate(batch.subset(slice(0, 3)))
    pilot_cpu = time.process_time() - c0
    assert time.process_time() - cpu + 20 * pilot_cpu < 480
    inputs, known = candidate.build_inputs(batch)
    np.savez(out / "input_arrays.npz", context=inputs, known=known, context_mask=np.isfinite(inputs), known_mask=np.isfinite(known))
    result = candidate.predict(batch)
    result.validate(batch)
    frames.append(result.to_frame(batch, "development").assign(policy="original_point_plus_bounded_patch_stats"))
    provenance = {"stage": stage, "protocol_sha256": sha256_file(out / "protocol.yaml"),
                  "spatial_inputs_sha256": sha256_file(out / "spatial_inputs.parquet"),
                  "input_arrays_sha256": sha256_file(out / "input_arrays.npz"),
                  "observation_sha256": sha256_file(obs_path), "environment": environment(),
                  "baseline_rerun_max_quantile_difference": baseline_delta, "baseline_rerun_cpu_s": baseline_cpu,
                  "pilot_three_cases_cpu_s": pilot_cpu, "inference": result.info,
                  "cpu_s_before_scoring": time.process_time() - cpu}
    predictions = pd.concat(frames, ignore_index=True)
    predictions["arome_run"] = pd.to_datetime(predictions.arome_run, utc=True)
    save_outputs(out / "forecasts", predictions, pd.read_parquet(source / "targets.parquet"), period_cases(manifest), provenance)
    write_json(out / "runtime.json", {"cpu_s": time.process_time() - cpu, "wall_s": time.perf_counter() - wall,
                                      "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
                                      "model_cache": dict(models.STATS)})
    print("Completed", out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--infer", action="store_true")
    args = parser.parse_args()
    common.ROOT = args.root.resolve()
    (prepare if args.prepare else infer)(common.ROOT, args.output.resolve())
