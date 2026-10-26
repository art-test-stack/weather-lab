"""Matched t0-gamma development baseline and three-option native inference ablation."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from importlib.metadata import version
import inspect
import os
from pathlib import Path
import resource
import sys
import time

os.environ["WEATHER_LAB_OFFLINE"] = "1"
# The runner's filename must not shadow the installed native t0 package.
sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]

import numpy as np
import pandas as pd
import torch
import yaml

from weather_lab import models
from weather_lab.common import environment, load_config, read_json, sha256_file, write_json
from weather_lab.evaluate import make_batch
from weather_lab.forecast import QUANTILE_LEVELS
from weather_lab.practical import period_cases, save_outputs
from weather_lab.suite import cases_from_manifest
from weather_lab.t0_adapter import T0


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output or root / "data/evidence/practical-improvements-continuation-v1/t0"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    cpu, wall = time.process_time(), time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(20261026)
    parent_path = root / "docs/experiments/practical_improvements_v1.yaml"
    parent = yaml.safe_load(parent_path.read_text())
    source = root / parent["source"]["run"]
    for name, expected in parent["source"]["hashes"].items():
        assert sha256_file(source / name) == expected, name
    obs_path = root / parent["source"]["observations"]["file"]
    arome_path = root / parent["source"]["arome"]["file"]
    assert sha256_file(obs_path) == parent["source"]["observations"]["sha256"]
    assert sha256_file(arome_path) == parent["source"]["arome"]["sha256"]
    ck = models.CHECKPOINTS["t0-gamma"]
    metadata = models.prefetch(ck, allow_download=False)
    folder = Path(metadata["path"])
    assert sha256_file(folder / "model.safetensors") == ck.sha256
    runtime = {package: version(package) for package in ("tfc-t0", "jaxtyping", "einops", "rotary-embedding-torch",
                                                          "wadler-lindig", "torch", "numpy", "huggingface-hub")}
    assert runtime["tfc-t0"] == "0.6.0"
    options = [("t0-gamma-mv-cal-baseline-v1", False, False, None),
               ("t0-gamma-mv-cal-trim-v1", True, False, None),
               ("t0-gamma-mv-cal-trim-symmetric-v1", True, True, None),
               ("t0-gamma-mv-cal-arome-archived-baseline-v1", False, False, "archived")]
    stage = {"stage": "m2-t0-gamma-v1", "parent_protocol_sha256": sha256_file(parent_path),
             "checkpoint": ck.ident, "checkpoint_sha256": ck.sha256,
             "config_sha256": sha256_file(folder / "config.json"), "runtime": runtime,
             "runtime_api_source": "https://github.com/theforecastingcompany/tfc-t0",
             "model_card": "https://huggingface.co/theforecastingcompany/t0-gamma",
             "source_hashes": parent["source"]["hashes"], "observations": parent["source"]["observations"],
             "arome": parent["source"]["arome"], "new_download_authorization": "user-approved isolated setup; no bootstrap/Space preload",
             "context_days": 30, "calendar": True, "features": [], "target": "raw weather", "postprocessor": "none",
             "quantile_levels": list(QUANTILE_LEVELS), "device": "cpu", "precision": "float32", "threads": 4,
             "clip_nonnegative": False, "snap_integers": False, "batch_size": 8,
             "candidates": [{"id": name, "trim_leading_missing": trim, "symmetric_averaging": symmetric,
                             "arome": arome} for name, trim, symmetric, arome in options],
             "periods": parent["periods"], "selection": parent["selection"], "uncertainty": parent["uncertainty"],
             "confirmation": "none", "cpu_budget_seconds": 1500, "pilot_cases_per_candidate": 3,
             "script_sha256": sha256_file(Path(__file__)),
             "adapter_sha256": sha256_file(Path(__file__).parents[2] / "src/weather_lab/t0_adapter.py")}
    out.mkdir(parents=True)
    (out / "protocol.yaml").write_text(yaml.safe_dump(stage, sort_keys=False))
    (out / "requirements-isolated.txt").write_text("".join(f"{name}=={value}\n" for name, value in runtime.items()))
    cfg = load_config(source / "config.yaml")
    manifest = read_json(source / "manifest.json")
    cases = cases_from_manifest(manifest)
    obs = pd.read_parquet(obs_path)
    batch, _ = make_batch(cfg, obs, cases, 240)
    conditioned, _ = make_batch(cfg, obs, cases, 240, pd.read_parquet(arome_path))
    adapters = [(T0(name, cfg["variables"], cfg["variables"], ck.repo, ck.revision,
                    calendar=True, arome=arome, arome_fields=["t", "pmer"] if arome else [],
                    trim_leading_missing=trim, symmetric_averaging=symmetric),
                 conditioned if arome else batch) for name, trim, symmetric, arome in options]
    pilots, costs = [], []
    for adapter, inputs in adapters:
        pcpu, pwall = time.process_time(), time.perf_counter()
        pilot = adapter.predict(inputs.subset(slice(0, 3)))
        pilot.validate(inputs.subset(slice(0, 3)))
        pilots.append(pilot)
        costs.append({"model": adapter.id, "pilot_cpu_s": time.process_time() - pcpu,
                      "pilot_wall_s": time.perf_counter() - pwall})
        print("t0 pilot:", costs[-1], flush=True)
    projected = sum(c["pilot_cpu_s"] for c in costs) * 20
    write_json(out / "pilot.json", {"candidates": costs, "projected_all_full_cpu_s": projected})
    if time.process_time() - cpu + projected + 120 >= 1500:
        raise RuntimeError("t0 pilot predicts the four-arm batch exceeds its CPU budget")
    frames, inference = [], {}
    (out / "candidate_predictions").mkdir()
    for (adapter, inputs), pilot, cost in zip(adapters, pilots, costs, strict=True):
        pcpu, pwall = time.process_time(), time.perf_counter()
        rest = adapter.predict(inputs.subset(slice(3, None)))
        result = replace(rest, point=np.concatenate([pilot.point, rest.point]),
                         quantiles=np.concatenate([pilot.quantiles, rest.quantiles]),
                         info={**rest.info, "n_cases": len(cases), "pilot_cases_reused": 3,
                               "input_identity": adapter.input_identity(inputs)})
        result.validate(inputs)
        cost.update(cpu_s=time.process_time() - pcpu + cost["pilot_cpu_s"],
                    wall_s=time.perf_counter() - pwall + cost["pilot_wall_s"])
        frame = result.to_frame(inputs, "development").assign(policy="native_t0_inference")
        frame.to_parquet(out / "candidate_predictions" / f"{adapter.id}.parquet", index=False)
        frames.append(frame)
        inference[adapter.id] = result.info
        print("t0 completed:", cost, flush=True)
    saved = pd.read_parquet(source / "predictions.parquet")
    frames.append(saved[saved.model.isin(["climatology", "persistence", "arome-raw", "arome-biascorr",
                                         "timestral-209m-mv-cal", "timestral-209m-mv-cal-arome-archived"])].assign(
                                             policy="verified_frozen_saved_prediction"))
    provenance = {"stage": stage["stage"], "protocol_sha256": sha256_file(out / "protocol.yaml"),
                  "checkpoint": ck.ident, "checkpoint_sha256": ck.sha256, "runtime": runtime,
                  "environment": environment(), "inference": inference, "candidates_runtime": costs,
                  "source_hashes": parent["source"]["hashes"], "observation_cutoff": "time<=origin; zero-latency historical replay",
                  "native_semantics": "past covariates forecast jointly; future_covariates conditioned, not forecast",
                  "matched_information": "same raw history, calendar, 30days, 3hour grid and21 quantiles; native normalization differs"}
    save_outputs(out / "scored", pd.concat(frames, ignore_index=True), pd.read_parquet(source / "targets.parquet"),
                 period_cases(manifest), provenance)

    # Native one-case validity probes are excluded from candidate selection and headline efficacy.
    item = models.MODELS.get(ck)
    fc = item.forecaster
    single = batch.subset(slice(0, 1))
    context, known, _ = adapters[0][0].build_inputs(single)

    def native(x, cov=known, **flags):
        with item.lock, torch.inference_mode():
            return fc.predict(x, 16, QUANTILE_LEVELS, future_covariates=cov,
                              trim_leading_missing=flags.get("trim_leading_missing", False),
                              symmetric_averaging=False, clip_nonnegative=False, snap_integers=False).quantiles.cpu().numpy()[:, :4]

    baseline = native(context)
    empty = native(np.concatenate([context, np.full((1, 1, 240), np.nan, np.float32)], axis=1))
    constant17 = native(np.concatenate([context, np.full((1, 1, 240), 17, np.float32)], axis=1))
    constant29 = native(np.concatenate([context, np.full((1, 1, 240), 29, np.float32)], axis=1))
    leading = context.copy()
    leading[:, :, :16] = np.nan
    untrimmed, trimmed = native(leading), native(leading, trim_leading_missing=True)
    delayed = context.copy()
    delayed[:, :, -1] = np.nan
    delay_prediction = native(delayed)
    aux = pd.read_parquet(root / "data/processed/synop_aux_07630-07149-07110-07190-07690_2022-2026.parquet")
    cloud_batch, _ = make_batch(cfg, obs, single.cases, 240, aux_df=aux)
    cloud = cloud_batch.aux_history[:, 1:2]
    changed_cloud = cloud.copy()
    changed_cloud[:, :, -8:] = 100 - changed_cloud[:, :, -8:]
    cloud_prediction = native(np.concatenate([context, cloud], axis=1))
    cloud_changed_prediction = native(np.concatenate([context, changed_cloud], axis=1))
    series, _, _, _ = fc._series_from_array(context, 16, known, None, None)
    scaled, loc_scale = fc.scaler.scale_input(series)
    np.savez_compressed(out / "tensor_trace.npz", history_times=single.history_times(), future_times=single.future_times,
                        context=context, known_covariates=known, native_values=series.variates.cpu().numpy(),
                        mask=series.mask.cpu().numpy(), roles=series.variate_type.cpu().numpy(),
                        groups=series.group_ids.cpu().numpy(), scaled=scaled.variates.cpu().numpy(),
                        loc=loc_scale.loc.cpu().numpy(), scale=loc_scale.scale.cpu().numpy())
    perturbed = obs.copy()
    perturbed.loc[perturbed.time.gt(single.cases.origin.iloc[0]), cfg["variables"]] = 9999
    causal, _ = make_batch(cfg, perturbed, single.cases, 240)
    assert np.array_equal(causal.history, single.history, equal_nan=True)
    differences = {}
    for j, variable in enumerate(cfg["variables"]):
        differences[variable] = {"empty_joint_row_max_quantile_abs_change": float(abs(empty[:, j] - baseline[:, j]).max()),
                                 "constant17_vs29_max_quantile_abs_change": float(abs(constant17[:, j] - constant29[:, j]).max()),
                                 "leading_missing_trim_max_quantile_abs_change": float(abs(trimmed[:, j] - untrimmed[:, j]).max()),
                                 "delayed3h_max_quantile_abs_change": float(abs(delay_prediction[:, j] - baseline[:, j]).max()),
                                 "changed_partial_cloud_max_quantile_abs_change": float(abs(cloud_changed_prediction[:, j] - cloud_prediction[:, j]).max())}
    write_json(out / "diagnostics.json", {"label": "one fixed development case; diagnostics excluded from selection",
                                         "per_variable_physical_units": differences, "cloud_observed_fraction": float(np.isfinite(cloud).mean()),
                                         "future_target_perturbation_changes_history": False,
                                         "native_config": asdict(fc.config),
                                         "predict_signature_defaults": {name: str(p.default) for name, p in inspect.signature(fc.predict).parameters.items()},
                                         "scaling": "targets/joint past rows causal Welford; known rows global; zero_no_value=False fromgamma config"})
    for name, expected in parent["source"]["hashes"].items():
        assert sha256_file(source / name) == expected, name
    assert models.STATS["downloads"] == 0 and models.STATS["loads"] == 1
    write_json(out / "runtime.json", {"cpu_s": time.process_time() - cpu, "wall_s": time.perf_counter() - wall,
                                      "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
                                      "model_cache": dict(models.STATS), "candidates": costs})
    summary = pd.read_csv(out / "scored/summary.csv")
    print(summary[summary.period.eq("development_selection") & summary.variable.eq("t")][
        ["model", "station_balanced_mae", "valid_pairs", "delta_mae_vs_raw"]].to_string(index=False))
    print((out / "runtime.json").read_text())


if __name__ == "__main__":
    main()
