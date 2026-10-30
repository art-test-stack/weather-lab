"""Run one frozen t0 residual wrapper on the platform's matched October cases."""
from __future__ import annotations

import argparse
from dataclasses import replace
from importlib.metadata import version
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

from error_wrapper import ResidualT0
from weather_lab import features, models
from weather_lab.common import ROOT, environment, load_config, read_json, sha256_file, write_json
from weather_lab.evaluate import make_batch
from weather_lab.scoring import score
from weather_lab.forecast import Timestral
from weather_lab.practical import matched_rows, period_cases, summaries
from weather_lab.suite import cases_from_manifest


def score_outputs(out, predictions, targets, cases, provenance):
    out.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    targets.to_parquet(out / "targets.parquet", index=False)
    cases.to_parquet(out / "cases.parquet", index=False)
    rows = matched_rows(predictions, targets, cases)
    assert rows.target_observed.all() and rows.prediction_finite.all()
    rows.to_parquet(out / "verification_pairs.parquet", index=False)
    summary, leads = summaries(rows)
    summary.to_csv(out / "summary.csv", index=False)
    leads.to_csv(out / "leads.csv", index=False)
    # Score all systems together: the legacy scorer needs quantile rows for its probability cube.
    standard = score(predictions, targets, cases, {}, [6, 24, 48])
    for name, table in standard.items():
        table.to_csv(out / f"{name}.csv", index=False)
    count, differences = 0, []
    for row in summary[summary.period.ne("all_development_descriptive")].itertuples():
        point = standard["metrics_point"]
        point = point[point.split.eq(row.period) & point.model.eq(row.model) & point.variable.eq(row.variable)
                      & point.station.eq("ALL") & point.season.eq("ALL") & point.lead_h.isna()].iloc[0]
        for name, value in (("mae", row.micro_mae), ("rmse", row.rmse), ("bias", row.bias)):
            differences.append(abs(float(point[name]) - value))
            count += 1
        if row.model not in ("arome-raw", "arome-biascorr"):
            prob = standard["metrics_probabilistic"]
            prob = prob[prob.split.eq(row.period) & prob.model.eq(row.model) & prob.variable.eq(row.variable)
                        & prob.station.eq("ALL") & prob.season.eq("ALL") & prob.lead_h.isna()].iloc[0]
            for name in ("pinball", "coverage80", "coverage90", "width80", "width90"):
                differences.append(abs(float(prob[name]) - getattr(row, name)))
                count += 1
    assert max(differences) < 1e-10
    write_json(out / "verification.json", {"checked_values": count, "tolerance": 1e-10,
               "max_absolute_difference": max(differences), "common_cases": cases.case_id.nunique()})
    write_json(out / "run.json", provenance)
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    fig = make_subplots(rows=3, cols=2, subplot_titles=("MAE", "RMSE", "Bias", "Paired delta MAE", "Pinball", "80% coverage"))
    for model, group in leads[leads.period.eq("all_development_descriptive") & leads.variable.eq("t")].groupby("model"):
        for i, metric in enumerate(("micro_mae", "rmse", "bias", "delta_mae_vs_raw", "pinball", "coverage80")):
            fig.add_trace(go.Scatter(x=group.lead_h, y=group.get(metric), name=model,
                legendgroup=model, showlegend=i == 0), row=i // 2 + 1, col=i % 2 + 1)
    fig.update_layout(title="Matched October t0 residual diagnostic; no confirmation", height=1000)
    fig.write_html(out / "lead_curves.html", include_plotlyjs=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rescore-from", type=Path, help="Verify saved predictions without loading weights")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    if args.rescore_from:
        source = args.rescore_from
        identity = read_json(source.parent / "input_provenance.json")
        provenance = yaml.safe_load((source.parent / "protocol.yaml").read_text())
        provenance = {**provenance, "full_batch_input_identity": identity["identity"],
                      "rescored_prediction_sha256": sha256_file(source / "predictions.parquet"),
                      "report_recovery": "original inference completed; per-model deterministic rescoring failed"}
        score_outputs(out / "scored", pd.read_parquet(source / "predictions.parquet"),
                      pd.read_parquet(source / "targets.parquet"), pd.read_parquet(source / "cases.parquet"), provenance)
        print(pd.read_csv(out / "scored/summary.csv").query("period == 'all_development_descriptive'")[
            ["model", "variable", "valid_pairs", "station_balanced_mae", "rmse"]].to_string(index=False))
        return
    cpu, wall = time.process_time(), time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(20261026)
    source = ROOT / "data/examples/arome-issued-v2"
    previous = ROOT / "experiments/arome-error-ctx90-20261025T214801Z"
    old_protocol = read_json(previous / "protocol.json")
    cfg = load_config(source / "config.yaml")
    manifest = read_json(source / "manifest.json")
    cached = pd.read_parquet(previous / "predictions.parquet")
    timestral_id = "timestral-209m-error-ctx90-all-features"
    ids = set(cached.loc[cached.model.eq(timestral_id), "case_id"])
    all_cases = cases_from_manifest(manifest)
    cases = all_cases[all_cases.case_id.isin(ids)].reset_index(drop=True)
    assert len(cases) == 55 and cases.station.nunique() == 5
    paths = {name: ROOT / "data/processed" / old_protocol["data"][name]["file"]
             for name in ("observations", "aux_observations")}
    paths["arome"] = source / "arome_station_extract.parquet"
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    assert all(hashes[n] == old_protocol["data"][n]["sha256"] for n in paths)
    ck = models.CHECKPOINTS["t0-gamma"]
    assert version("tfc-t0") == "0.6.0"
    metadata = models.prefetch(ck, allow_download=False)
    assert sha256_file(Path(metadata["path"]) / "model.safetensors") == ck.sha256
    spec = features.normalize(["resid", "atmos", "solar", "tend"])
    adapter = ResidualT0("t0-gamma-error-ctx90-all-features-v1", cfg["variables"], ["t", "pmer"],
                         ck.repo, ck.revision, context_days=90, calendar=False, arome="missing",
                         arome_fields=["t", "pmer"], features=spec, batch_size=5,
                         trim_leading_missing=False, symmetric_averaging=False,
                         clip_nonnegative=False, snap_integers=False)
    protocol = {"stage": "t0-error-wrapper-v1", "purpose": "user-requested matched platform diagnostic",
                "configuration": adapter.describe(), "native_horizon_hours": 48, "step_hours": 3,
                "context_steps": 720, "observation_delay_hours": 0,
                "residual_history": "latest qualifying issued forecast; not fixed-lead histories",
                "case_rule": "same 55 cases as saved Timestral; Oct 10 excluded for absent error history",
                "residual_support_policy": "sparse developmental diagnostic; not the new 80-percent eligibility suite",
                "confirmation": "none; no model selection or promotion", "sources": hashes,
                "cached_timestral_sha256": sha256_file(previous / "predictions.parquet"),
                "model": ck.ident, "checkpoint_sha256": ck.sha256,
                "runtime": {p: version(p) for p in ("tfc-t0", "torch", "numpy")},
                "environment": {**environment(), "torch": str(torch.__version__)},
                "device": "cpu", "precision": "float32", "seed": 20261026,
                "cpu_cap_seconds": 180, "pilot_cases": 5,
                "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in (
                    Path(__file__).resolve(), Path(__file__).with_name("error_wrapper.py").resolve(),
                    ROOT / "src/weather_lab/t0_adapter.py", ROOT / "src/weather_lab/features.py",
                    ROOT / "src/weather_lab/evaluate.py", ROOT / "src/weather_lab/arome.py",
                    ROOT / "src/weather_lab/practical.py")}}
    (out / "protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    coords = {s["id"]: (s["lat"], s["lon"]) for s in manifest["dataset"]["stations"]}
    batch, _ = make_batch(cfg, pd.read_parquet(paths["observations"]), cases, 720,
                          pd.read_parquet(paths["arome"]), pd.read_parquet(paths["aux_observations"]), coords)
    # Match residual targets and lead inputs exactly; each family translates covariates natively.
    reference = Timestral("input-parity", cfg["variables"], ["t", "pmer"],
                          models.CHECKPOINTS["Timestral-209M"].repo, models.CHECKPOINTS["Timestral-209M"].revision,
                          context_days=90, prediction_target="arome_error")
    errors, baseline, leads = adapter.residual_inputs(batch)
    for native, expected in zip((errors, baseline, leads), reference.error_inputs(batch)):
        np.testing.assert_array_equal(native, expected)
    context, known, report = adapter.build_inputs(batch)
    write_json(out / "input_provenance.json", {"identity": adapter.input_identity(batch), "report": report,
                                               "matched_timestral_residual_targets_and_run_leads": True})
    np.savez_compressed(out / "input_tensors.npz", context=context, known=known,
                        context_mask=np.isfinite(context), known_mask=np.isfinite(known),
                        origins=batch.origins, valid_times=batch.future_times)
    pilot_cpu = time.process_time()
    pilot = adapter.predict(batch.subset(slice(0, 5)))
    pilot_cost = time.process_time() - pilot_cpu
    projected = pilot_cost * len(cases) / 5
    write_json(out / "pilot.json", {"cpu_seconds": pilot_cost, "projected_full_cpu_seconds": projected})
    print(f"Pilot: {pilot_cost:.2f} CPU seconds; projected full inference {projected:.2f}s", flush=True)
    if time.process_time() - cpu + projected - pilot_cost + 20 > 180:
        raise RuntimeError("Pilot exceeds the 180 CPU-second allocation; main batch deferred")
    results = [pilot]
    for start in range(5, len(cases), 5):
        if time.process_time() - cpu > 160:
            raise RuntimeError("CPU reserve reached; incomplete arm, no aggregate efficacy claim")
        results.append(adapter.predict(batch.subset(slice(start, start + 5))))
        print(f"Predicted {min(start+5, len(cases))}/{len(cases)}", flush=True)
    forecast = replace(pilot, point=np.concatenate([r.point for r in results]),
                       quantiles=np.concatenate([r.quantiles for r in results]),
                       info={**pilot.info, "input_identity": adapter.input_identity(batch), "n_cases": len(cases)})
    assert forecast.info["input_identity"] == adapter.input_identity(batch)
    assert forecast.info["input_identity"] != pilot.info["input_identity"]
    forecast.validate(batch)
    assert np.isfinite(forecast.quantiles).all()
    frame = forecast.to_frame(batch, cfg.suite)
    frame["arome_baseline"] = baseline.ravel()
    frame["predicted_error"] = frame.point - frame.arome_baseline
    frame["policy"] = "experimental_native_t0_residual_wrapper"
    refs = cached[cached.model.isin(["arome-raw", timestral_id]) & cached.case_id.isin(ids)].copy()
    refs["policy"] = "verified_saved_prediction"
    corrected_arome = pd.read_parquet(source / "predictions.parquet")
    corrected_arome = corrected_arome[corrected_arome.model.eq("arome-biascorr")
        & corrected_arome.case_id.isin(ids) & corrected_arome.variable.isin(["t", "pmer"])].copy()
    corrected_arome["policy"] = "verified_saved_prediction"
    assert set(corrected_arome.case_id) == ids
    predictions = pd.concat([refs, corrected_arome, frame], ignore_index=True)
    targets = pd.read_parquet(source / "targets.parquet")
    targets = targets[targets.case_id.isin(ids) & targets.variable.isin(["t", "pmer"])]
    periods = period_cases(manifest)
    periods = periods[periods.case_id.isin(ids)].reset_index(drop=True)
    write_json(out / "inference_runtime.json", {"cpu_seconds": time.process_time() - cpu,
               "wall_seconds": time.perf_counter() - wall, "forecast_info": forecast.info})
    score_outputs(out / "scored", predictions, targets, periods, protocol | {"forecast_info": forecast.info})
    assert models.STATS["downloads"] == 0 and models.STATS["loads"] == 1
    assert all(sha256_file(paths[n]) == hashes[n] for n in paths)
    runtime = {"cpu_seconds": time.process_time() - cpu, "wall_seconds": time.perf_counter() - wall,
               "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
               "model_cache": dict(models.STATS)}
    write_json(out / "runtime.json", runtime)
    table = pd.read_csv(out / "scored/summary.csv")
    print(table[table.period.eq("all_development_descriptive")][
        ["model", "variable", "valid_pairs", "station_balanced_mae", "rmse"]].to_string(index=False), flush=True)
    print(runtime, flush=True)


if __name__ == "__main__":
    main()
