"""Run the frozen M1 development comparison using shared, read-only local data/weights."""
from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import resource
import time

os.environ["WEATHER_LAB_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch
import yaml

from weather_lab import models
from weather_lab.baselines import Climatology
from weather_lab.common import environment, load_config, read_json, sha256_file, write_json
from weather_lab.evaluate import make_batch
from weather_lab.forecast import Timestral
from weather_lab.practical import period_cases, save_outputs
from weather_lab.practical_climatology import HarmonicClimatology
from weather_lab.suite import cases_from_manifest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output or root / "data/evidence/practical-improvements-continuation-v1/m1"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    cpu, wall = time.process_time(), time.perf_counter()
    torch.set_num_threads(4)
    parent_path = root / "docs/experiments/practical_improvements_v1.yaml"
    parent = yaml.safe_load(parent_path.read_text())
    source = root / parent["source"]["run"]
    for name, expected in parent["source"]["hashes"].items():
        assert sha256_file(source / name) == expected, name
    obs_path = root / parent["source"]["observations"]["file"]
    assert sha256_file(obs_path) == parent["source"]["observations"]["sha256"]
    training_path = root / "data/processed/synop_07630-07149-07110-07190-07690_2018-2023.parquet"
    stage = {"stage": "m1-pooled-harmonic-v1", "parent_protocol_sha256": sha256_file(parent_path),
             "source_hashes": parent["source"]["hashes"], "training_source": str(training_path.relative_to(root)),
             "training_source_sha256": sha256_file(training_path),
             "training_start": "2018-01-01T00:00:00Z", "training_end": "2022-12-31T21:00:00Z",
             "transform": {"station_intercepts": "unpenalized", "pooled_annual_harmonics": 2,
                           "pooled_UTC_daily_harmonics": 2, "ridge_penalty": 1,
                           "minimum_support": "1000 finite rows and all12 months per station/variable"},
             "model": models.CHECKPOINTS["Timestral-209M"].ident,
             "context_days": 30, "calendar": True, "features": [], "conditioning": None,
             "inference": {"symmetric_averaging": True, "clip_nonnegative": False, "snap_integers": False},
             "device": "cpu", "precision": "float32", "threads": 4,
             "cpu_budget_seconds": 480, "pilot_cases": 3, "full_cases": 60,
             "candidates": ["m1-pooled-harmonic-climatology-v1", "m1-empirical-climatology-fit2018-2022-v1",
                            "timestral-209m-mv-cal", "m1-anomaly-209m-v1"],
             "comparators": ["climatology (saved empirical fit2022-2024)", "arome-raw", "arome-biascorr"],
             "periods": parent["periods"], "selection": parent["selection"],
             "uncertainty": parent["uncertainty"], "confirmation": "none",
             "script_sha256": sha256_file(Path(__file__)),
             "implementation_sha256": sha256_file(Path(__file__).parents[2] / "src/weather_lab/practical_climatology.py")}
    out.mkdir(parents=True)
    (out / "protocol.yaml").write_text(yaml.safe_dump(stage, sort_keys=False))
    training = pd.read_parquet(training_path)
    training = training[training.time.between(pd.Timestamp(stage["training_start"]),
                                              pd.Timestamp(stage["training_end"]))]
    cfg = load_config(source / "config.yaml")
    fit = HarmonicClimatology.fit(training, cfg["variables"], source_fingerprint=stage["training_source_sha256"])
    write_json(out / "transform_fit.json", {"identity": fit.identity, **fit.provenance})
    empirical = Climatology("m1-empirical-climatology-fit2018-2022-v1", cfg["variables"])
    empirical.fit(training)
    manifest = read_json(source / "manifest.json")
    cases = cases_from_manifest(manifest)
    assert len(cases) == 60
    batch, _ = make_batch(cfg, pd.read_parquet(obs_path), cases, 240)
    transformed = fit.anomaly_batch(batch)
    assert np.array_equal(batch.future_times, transformed.future_times)
    ck = models.CHECKPOINTS["Timestral-209M"]
    models.prefetch(ck, allow_download=False)
    adapter = Timestral("m1-anomaly-209m-v1", cfg["variables"], cfg["variables"],
                        ck.repo, ck.revision, calendar=True)
    pilot_cpu, pilot_wall = time.process_time(), time.perf_counter()
    pilot = adapter.predict(transformed.subset(slice(0, 3)))
    pilot_cost = {"cpu_s": time.process_time() - pilot_cpu, "wall_s": time.perf_counter() - pilot_wall}
    pilot_cost["projected_full_cpu_s"] = pilot_cost["cpu_s"] * 20
    print("M1 pilot:", pilot_cost, flush=True)
    if time.process_time() - cpu + pilot_cost["projected_full_cpu_s"] + 30 >= 480:
        raise RuntimeError("M1 pilot exceeds allocated CPU budget")
    rest = adapter.predict(transformed.subset(slice(3, None)))
    result = fit.restore(replace(pilot, point=np.concatenate([pilot.point, rest.point]),
                                 quantiles=np.concatenate([pilot.quantiles, rest.quantiles]),
                                 info={**rest.info, "n_cases": len(cases), "pilot_cases_reused": 3}), batch)
    result.validate(batch)
    saved = pd.read_parquet(source / "predictions.parquet")
    reused = saved[saved.model.isin(["climatology", "persistence", "arome-raw", "arome-biascorr",
                                    "timestral-209m-mv-cal", "timestral-209m-mv-cal-arome-archived"])].copy()
    reused["policy"] = "verified_frozen_saved_prediction"
    frames = [reused, result.to_frame(batch, "development").assign(policy="anomaly_target_restored"),
              fit.predict(batch).to_frame(batch, "development").assign(policy="earlier_training_climatology"),
              empirical.predict(batch).to_frame(batch, "development").assign(policy="earlier_training_empirical_climatology")]
    provenance = {"stage": stage["stage"], "protocol_sha256": sha256_file(out / "protocol.yaml"),
                  "transform_fit_identity": fit.identity, "inference": result.info,
                  "source_hashes": parent["source"]["hashes"], "environment": environment(),
                  "target_units": fit.provenance["units"], "observation_cutoff": "time<=origin, zero-latency replay",
                  "context_days": 30, "features": [], "postprocessor": "none", "pilot": pilot_cost}
    save_outputs(out / "scored", pd.concat(frames, ignore_index=True), pd.read_parquet(source / "targets.parquet"),
                 period_cases(manifest), provenance)
    for name, expected in parent["source"]["hashes"].items():
        assert sha256_file(source / name) == expected, name
    assert models.STATS["downloads"] == 0 and models.STATS["loads"] == 1
    runtime = {"cpu_s": time.process_time() - cpu, "wall_s": time.perf_counter() - wall,
               "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
               "model_cache": dict(models.STATS), "pilot": pilot_cost}
    write_json(out / "runtime.json", runtime)
    print(runtime, flush=True)
    summary = pd.read_csv(out / "scored/summary.csv")
    print(summary[summary.period.eq("development_selection") & summary.variable.eq("t")][
        ["model", "station_balanced_mae", "valid_pairs", "delta_mae_vs_raw"]].to_string(index=False))


if __name__ == "__main__":
    main()
