"""Verify and independently rescore the exact saved platform error configuration."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import resource
import time

os.environ["WEATHER_LAB_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch
import yaml

from weather_lab import features, models
from weather_lab.common import ROOT, load_config, read_json, sha256_file, write_json
from weather_lab.evaluate import make_batch
from weather_lab.forecast import Timestral, qcol
from weather_lab.predcache import make_key
from weather_lab.suite import cases_from_manifest

MODEL = "timestral-209m-error-ctx90-all-features"
KEY = ["case_id", "variable", "lead_h"]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    cpu, wall = time.process_time(), time.perf_counter()
    torch.set_num_threads(4)
    previous = ROOT / "experiments/arome-error-ctx90-20261025T214801Z"
    source = ROOT / "data/examples/arome-issued-v2"
    saved = read_json(previous / "protocol.json")
    assert saved["context_days"] == 90 and saved["inference_horizon_h"] == 48
    assert saved["conditioning"] == "futureonly" and not saved["calendar"]
    cfg = load_config(source / "config.yaml")
    manifest = read_json(source / "manifest.json")
    cases = cases_from_manifest(manifest)
    paths = {name: ROOT / "data/processed" / saved["data"][name]["file"]
             for name in ("observations", "aux_observations")}
    paths["arome"] = source / "arome_station_extract.parquet"
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    assert all(hashes[name] == saved["data"][name]["sha256"] for name in paths)
    coords = {s["id"]: (s["lat"], s["lon"]) for s in manifest["dataset"]["stations"]}
    batch, _ = make_batch(cfg, pd.read_parquet(paths["observations"]), cases, 720,
                          pd.read_parquet(paths["arome"]), pd.read_parquet(paths["aux_observations"]), coords)
    ck = models.CHECKPOINTS["Timestral-209M"]
    assert ck.revision == saved["revision"]
    spec = features.normalize(["resid", "atmos", "solar", "tend"])
    assert features.definition(spec) == saved["features"]
    adapter = Timestral(MODEL, batch.variables, targets=["t", "pmer"], checkpoint=ck.repo,
                        revision=ck.revision, context_days=90, calendar=False, arome="missing",
                        arome_fields=["t", "pmer"], features=spec, batch_size=5, prediction_target="arome_error")
    assert adapter.describe() == next(s for s in saved["systems"] if s["id"] == MODEL)
    protocol = {"stage": "platform-error-check-v1", "configuration": adapter.describe(),
                "context_days": 90, "native_horizon_hours": 48, "step_hours": 3,
                "observation_delay_hours": 0, "purpose": "exact platform developmental reproduction",
                "eligibility": "reuse matching cached cases; not the new 80-percent residual-history benchmark",
                "new_confirmation": False, "source_sha256": hashes,
                "cached_predictions_sha256": sha256_file(previous / "predictions.parquet"),
                "script_sha256": sha256_file(Path(__file__)), "cpu_cap_seconds": 180,
                "pilot_cases": 5, "reuse": "all input fingerprints must match before cached predictions are used"}
    (out / "protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    fingerprints = pd.read_csv(previous / "input_fingerprints.csv")
    fingerprints = fingerprints[fingerprints.model.eq(MODEL)].set_index("case_id")
    valid, support = [], []
    for i, case in cases.iterrows():
        if case.case_id not in fingerprints.index:
            continue
        one = batch.subset([i])
        ctx, known = adapter.build_inputs(one)
        inputs = features.build_inputs(spec, one, 720)
        assert all(v > 0 for v in inputs.report["observed_fraction"].values())
        identity = make_key({"config": adapter.describe(), "case": case.case_id,
                             "features": features.definition(spec)},
                            {"context": ctx, "known": known, "past": inputs.past, "baseline": one.arome[:, :2]})
        assert identity == fingerprints.loc[case.case_id, "input_sha256"], case.case_id
        valid.append(i)
        support.append({"case_id": case.case_id, "t_steps": int(np.isfinite(ctx[0, 0]).sum()),
                        "pmer_steps": int(np.isfinite(ctx[0, 1]).sum()), "total_steps": 720})
    assert len(valid) == 55
    pd.DataFrame(support).to_csv(out / "residual_support.csv", index=False)
    predictions = pd.read_parquet(previous / "predictions.parquet")
    corrected = predictions[predictions.model.eq(MODEL)]
    models.prefetch(ck, allow_download=False)
    pilot = batch.subset(valid[:5])
    pilot_cpu = time.process_time()
    fresh = adapter.predict(pilot)
    fresh.validate(pilot)
    generated = fresh.to_frame(pilot, cfg.suite)
    compared = generated.merge(corrected, on=KEY, suffixes=("_new", "_cached"), validate="one_to_one")
    columns = ["point"] + [qcol(q) for q in fresh.quantile_levels]
    differences = [float(np.max(np.abs(compared[c + "_new"] - compared[c + "_cached"]))) for c in columns]
    assert max(differences) <= 1e-6, differences
    generated.to_parquet(out / "fresh_parity_predictions.parquet", index=False)
    print(f"Fresh 5-case parity: max difference {max(differences):.3g}; CPU {time.process_time()-pilot_cpu:.2f}s", flush=True)
    raw = predictions[predictions.model.eq("arome-raw")][KEY + ["point"]].rename(columns={"point": "raw"})
    targets = pd.read_parquet(source / "targets.parquet")
    rows = corrected.merge(raw, on=KEY, validate="one_to_one").merge(
        targets[KEY + ["observed"]], on=KEY, validate="one_to_one")
    assert len(rows) == 1760 and rows.case_id.nunique() == 55
    np.testing.assert_array_equal(rows.raw, rows.arome_baseline)
    np.testing.assert_allclose(rows.point, rows.raw + rows.predicted_error, atol=1e-10, rtol=0)
    rows.to_parquet(out / "matched_predictions_targets.parquet", index=False)
    old_scores = pd.read_csv(previous / "summary.csv")
    records, score_differences = [], []
    for variable, group in rows.groupby("variable"):
        for horizon in (24, 48):
            window = group[group.lead_h <= horizon]
            for model, column in (("arome-raw", "raw"), (MODEL, "point")):
                err = window[column].to_numpy(dtype=float) - window.observed.to_numpy(dtype=float)
                record = {"variable": variable, "window_hours": horizon, "model": model,
                          "pairs": len(window), "mae": float(np.abs(err).mean()),
                          "rmse": float(np.sqrt(np.mean(err**2))), "bias": float(err.mean())}
                reference = old_scores[old_scores.model.eq(model) & old_scores.variable.eq(variable)
                                       & old_scores.window_h.eq(horizon) & old_scores.station.eq("ALL")].iloc[0]
                score_differences.extend(abs(record[m] - reference[m]) for m in ("mae", "rmse", "bias"))
                records.append(record)
    assert max(score_differences) < 1e-10
    pd.DataFrame(records).to_csv(out / "results.csv", index=False)
    assert models.STATS["downloads"] == 0
    runtime = {"cpu_seconds": time.process_time() - cpu, "wall_seconds": time.perf_counter() - wall,
               "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}
    assert runtime["cpu_seconds"] < 180
    write_json(out / "verification.json", {"matched_input_fingerprints": len(valid), "fresh_cases": 5,
               "quantile_parity_max_difference": max(differences), "metric_tolerance": 1e-10,
               "independent_metric_max_difference": max(score_differences), "runtime": runtime,
               "model_cache": dict(models.STATS), "source_hashes_unchanged":
                   all(sha256_file(paths[n]) == hashes[n] for n in paths)})
    print(pd.DataFrame(records).to_string(index=False), flush=True)
    print(runtime, flush=True)


if __name__ == "__main__":
    main()
