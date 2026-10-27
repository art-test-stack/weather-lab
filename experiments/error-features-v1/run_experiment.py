"""Run the three approved information arms, one model family at a time."""
import argparse
import importlib.util
import os
from pathlib import Path
import resource
import sys
import time

os.environ["WEATHER_LAB_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch
import yaml

from adapters import ExtraInputs, ExtraT0, ExtraTimestral
from weather_lab import models
from weather_lab.common import ROOT, environment, load_config, read_json, sha256_file, write_json
from weather_lab.evaluate import make_batch
from weather_lab.predcache import make_key
from weather_lab.suite import cases_from_manifest

# Reuse the previous isolated reporting helper without extending the production scorer.
sys.path.insert(0, str(ROOT / "experiments/t0-error-wrapper-v1"))
spec = importlib.util.spec_from_file_location("previous_error_experiment",
    ROOT / "experiments/t0-error-wrapper-v1/run_experiment.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
sys.path.pop(0)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--family", choices=["timestral", "t0"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", nargs="+", choices=["all", "existing", "past-wind", "issued-atmosphere", "both"], default=["all"])
    parser.add_argument("--horizon-hours", type=int, choices=[24, 48], default=48)
    parser.add_argument("--cpu-budget-seconds", type=int, default=240)
    args = parser.parse_args()
    maximum = (480 if args.family == "timestral" else 180) if args.horizon_hours == 24 else 240
    if not 1 <= args.cpu_budget_seconds <= maximum:
        raise ValueError(f"Execution allocation must be between 1 and {maximum} CPU-seconds")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    cpu, wall = time.process_time(), time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(20261026)
    source = ROOT / "data/examples/arome-issued-v2"
    existing = ROOT / "data/evidence/t0-error-wrapper-v1-verified/scored"
    evidence = ROOT / "data/evidence/error-features-v1"
    frozen = read_json(existing / "run.json")
    old_protocol = read_json(ROOT / "experiments/arome-error-ctx90-20261025T214801Z/protocol.json")
    paths = {name: ROOT / "data/processed" / old_protocol["data"][name]["file"]
             for name in ("observations", "aux_observations")}
    paths["arome"] = source / "arome_station_extract.parquet"
    assert all(sha256_file(path) == frozen["sources"][name] for name, path in paths.items())
    plan = yaml.safe_load((Path(__file__).with_name("protocol.yaml")).read_text())
    if args.horizon_hours == 24:
        native_plan = yaml.safe_load(Path(__file__).with_name("protocol-native24.yaml").read_text())
        plan = {**plan, **native_plan}
    ck = models.CHECKPOINTS["Timestral-209M" if args.family == "timestral" else "t0-gamma"]
    models.prefetch(ck, allow_download=False)
    identity = {"plan_sha256": sha256_file(Path(__file__).with_name("protocol.yaml")),
                "issued_sha256": sha256_file(evidence / "issued_atmosphere.parquet"),
                "baseline_predictions_sha256": sha256_file(existing / "predictions.parquet"),
                "checkpoint": ck.ident, "weight_sha256": ck.sha256,
                "execution_allocation": {"arm": args.arm, "cpu_cap_seconds": args.cpu_budget_seconds},
                **({"native24_plan_sha256": sha256_file(Path(__file__).with_name("protocol-native24.yaml"))}
                   if args.horizon_hours == 24 else {}),
                "environment": {**environment(), "torch": str(torch.__version__)},
                "sources": frozen["sources"], "plan": plan,
                "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in (
                    Path(__file__).resolve(), Path(__file__).with_name("adapters.py").resolve(),
                    ROOT / "experiments/t0-error-wrapper-v1/error_wrapper.py",
                    ROOT / "experiments/t0-error-wrapper-v1/run_experiment.py",
                    ROOT / "src/weather_lab/forecast.py", ROOT / "src/weather_lab/features.py",
                    ROOT / "src/weather_lab/t0_adapter.py", ROOT / "src/weather_lab/evaluate.py")}}
    (out / "protocol.yaml").write_text(yaml.safe_dump(identity, sort_keys=False))
    predictions = pd.read_parquet(existing / "predictions.parquet")
    targets = pd.read_parquet(existing / "targets.parquet")
    predictions = predictions[predictions.lead_h.le(args.horizon_hours)]
    targets = targets[targets.lead_h.le(args.horizon_hours)]
    cases = pd.read_parquet(existing / "cases.parquet")
    assert len(cases) == 55
    cfg = load_config(source / "config.yaml")
    cfg.raw["forecast"] = {**cfg["forecast"], "horizon_steps": args.horizon_hours // 3}
    manifest = read_json(source / "manifest.json")
    source_cases = cases_from_manifest(manifest).set_index("case_id").loc[cases.case_id].reset_index()
    coords = {s["id"]: (s["lat"], s["lon"]) for s in manifest["dataset"]["stations"]}
    batch, _ = make_batch(cfg, pd.read_parquet(paths["observations"]), source_cases, 720,
        pd.read_parquet(paths["arome"]), pd.read_parquet(paths["aux_observations"]), coords)
    baseline_id = ("timestral-209m-error-ctx90-all-features" if args.family == "timestral"
                   else "t0-gamma-error-ctx90-all-features-v1")
    references = ["arome-raw", "arome-biascorr"] + ([baseline_id] if args.horizon_hours == 48 else [])
    frames = [predictions[predictions.model.isin(references)].copy()]
    issued = pd.read_parquet(evidence / "issued_atmosphere.parquet")
    runtimes = []
    arms = [("past-wind", True, False), ("issued-atmosphere", False, True), ("both", True, True)]
    if args.horizon_hours == 24 or "existing" in args.arm:
        arms.insert(0, ("existing", False, False))
    for arm, wind, weather in arms:
        if "all" not in args.arm and arm not in args.arm:
            continue
        adapter_class = ExtraTimestral if args.family == "timestral" else ExtraT0
        kwargs = {"prediction_target": "arome_error"} if args.family == "timestral" else {
            "trim_leading_missing": False, "symmetric_averaging": False}
        extra_identity = {"past_wind_v1": wind, "issued_atmosphere_v1": weather,
                          "issued_source_sha256": identity["issued_sha256"] if weather else None}
        suffix = "-h24" if args.horizon_hours == 24 else ""
        adapter = adapter_class(f"{args.family}-error-ctx90-{arm}{suffix}-v1", cfg["variables"], ["t", "pmer"],
            ck.repo, ck.revision, context_days=90, calendar=False, arome="missing", arome_fields=["t", "pmer"],
            features=["resid", "atmos", "solar", "tend"], batch_size=5,
            extras=ExtraInputs(wind, issued if weather else None), extra_identity=extra_identity, **kwargs)
        built = adapter.build_inputs(batch)
        context, known = built[:2]
        past = adapter.feature_rows(batch)[0] if args.family == "timestral" else None
        key = make_key({"configuration": adapter.describe(), "provenance": identity,
                        "origins": batch.decision_origins.astype(str).tolist()},
                       {"context": context, "known": known, "past": past})
        np.savez_compressed(out / f"{arm}-inputs.npz", context=context, known=known,
                            past=np.empty(0) if past is None else past,
                            context_mask=np.isfinite(context), known_mask=np.isfinite(known))
        started = time.process_time()
        pilot = adapter.predict(batch.subset(slice(0, 5)))
        cost = time.process_time() - started
        print(f"{args.family}/{arm}: 5-case pilot {cost:.2f} CPU-s; projected {cost*11:.2f}s", flush=True)
        if time.process_time() - cpu + cost * 10 + 10 > args.cpu_budget_seconds:
            raise RuntimeError("Family CPU allocation would be exceeded; remaining arms deferred")
        parts = [pilot]
        for start in range(5, len(cases), 5):
            if time.process_time() - cpu > args.cpu_budget_seconds - 10:
                raise RuntimeError("Family CPU reserve reached before next batch")
            parts.append(adapter.predict(batch.subset(slice(start, start + 5))))
        from dataclasses import replace
        forecast = replace(pilot, point=np.concatenate([p.point for p in parts]),
            quantiles=np.concatenate([p.quantiles for p in parts]), info={**pilot.info, "input_identity": key, "n_cases": 55})
        forecast.validate(batch)
        assert np.isfinite(forecast.quantiles).all()
        frame = forecast.to_frame(batch, cfg.suite)
        frame["arome_baseline"] = batch.arome[:, :2].ravel()
        frame["predicted_error"] = frame.point - frame.arome_baseline
        frame["policy"] = arm
        frame.to_parquet(out / f"{arm}-predictions.parquet", index=False)
        frames.append(frame)
        runtimes.append({"arm": arm, "pilot_cpu_seconds": cost, "cpu_seconds": time.process_time() - started,
                         "input_identity": key, "configuration": adapter.describe()})
        write_json(out / "inference_runtime.json", {"arms": runtimes, "cpu_seconds": time.process_time() - cpu})
        print(f"Completed {args.family}/{arm}: {len(cases)} cases", flush=True)
    previous.score_outputs(out / "scored", pd.concat(frames, ignore_index=True), targets, cases, identity)
    runtime = {"cpu_seconds": time.process_time() - cpu, "wall_seconds": time.perf_counter() - wall,
               "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
               "model_cache": dict(models.STATS), "arms": runtimes}
    assert models.STATS["downloads"] == 0 and models.STATS["loads"] == 1
    assert runtime["cpu_seconds"] < args.cpu_budget_seconds
    write_json(out / "runtime.json", runtime)
    table = pd.read_csv(out / "scored/summary.csv")
    print(table[table.period.eq("all_development_descriptive")][
        ["model", "variable", "station_balanced_mae", "rmse"]].to_string(index=False), flush=True)
    print(runtime, flush=True)


if __name__ == "__main__":
    main()
