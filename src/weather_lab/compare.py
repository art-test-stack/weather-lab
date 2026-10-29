"""Compare candidate runs on one frozen suite and apply a pre-declared selection rule."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from weather_lab.common import ROOT, load_config, now_utc, read_json, write_json
from weather_lab.scoring import score
from weather_lab.suite import cases_from_manifest

OUT = ROOT / "experiments" / "context_selection"


def latest_run(cfg) -> Path:
    runs = sorted(p.parent for p in (cfg.dir("runs") / cfg.suite).glob(f"*-{cfg['experiment']}-*/run.json"))
    if not runs:
        raise FileNotFoundError(f"no run for {cfg.path.name}; run `weather-lab evaluate --config {cfg.path}`")
    return runs[-1]


def compare(rule_path: Path, log=print) -> dict:
    rule = yaml.safe_load(open(rule_path))
    cands = rule["candidates"]
    runs = {k: latest_run(load_config(ROOT / c["config"])) for k, c in cands.items()}
    manifests = {k: read_json(r / "manifest.json") for k, r in runs.items()}
    protos = {m["protocol_sha256"] for m in manifests.values()}
    case_sets = {tuple(c["case_id"] for c in m["cases"] if c["eligible"]) for m in manifests.values()}
    if len(protos) != 1 or len(case_sets) != 1:
        raise RuntimeError("candidate runs do not share one frozen manifest / eligibility set")
    preds = {k: pd.read_parquet(r / "predictions.parquet") for k, r in runs.items()}
    first = next(iter(preds))
    base = preds[first][~preds[first]["model"].str.startswith("timestral")]
    for k, p in preds.items():   # baselines must be identical (same fit, same cases)
        b = p[~p["model"].str.startswith("timestral")].reset_index(drop=True)
        pd.testing.assert_frame_equal(b.drop(columns="run_id"), base.reset_index(drop=True).drop(columns="run_id"))
    combined = pd.concat([base] + [p[p["model"] == cands[k]["model"]] for k, p in preds.items()], ignore_index=True)
    targets = pd.read_parquet(runs[first] / "targets.parquet")
    cases = cases_from_manifest(manifests[first])
    thresholds = read_json(runs[first] / "tail_thresholds.json")
    tables = score(combined, targets, cases, thresholds, [6, 24, 48])

    point, prob = tables["metrics_point"], tables["metrics_probabilistic"]
    allsel = (point["station"] == "ALL") & (point["season"] == "ALL")
    psel = (prob["station"] == "ALL") & (prob["season"] == "ALL")
    model_to_cand = {c["model"]: k for k, c in cands.items()}
    p_all = prob[psel & prob["lead_h"].isna() & prob["model"].isin(model_to_cand)]
    prim = p_all[p_all["variable"] == rule["primary"]["variable"]].set_index("model")[rule["primary"]["metric"]]
    best = prim.min()
    tied = [m for m in prim.index if prim[m] - best <= rule["tie"]["tolerance_abs"]]
    chosen = min(tied, key=lambda m: cands[model_to_cand[m]]["context_days"])

    systems = {k: next(s for s in read_json(r / "run.json")["systems"] if s["id"] == cands[k]["model"])
               for k, r in runs.items()}
    resources = pd.DataFrame([{"candidate": k, "context_steps": cands[k]["steps"], "batch_wall_s": s["wall_s"],
                               "per_case_ms": s["per_case_ms"], "warm_single_case_ms": s["warm_single_case_ms_median"],
                               "peak_rss_mb_after": s["peak_rss_mb_after"], "n_cases": s["n_cases"]}
                              for k, s in systems.items()])
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
    resources.to_csv(OUT / "resources.csv", index=False)
    result = {
        "rule": str(rule_path.resolve().relative_to(ROOT)), "decided_at": now_utc(),
        "runs": {k: r.name for k, r in runs.items()}, "protocol_sha256": protos.pop(),
        "n_eligible_cases": len(next(iter(case_sets))),
        "primary": {model_to_cand[m]: round(float(v), 5) for m, v in prim.items()},
        "n_common_pairs_t": int(p_all[p_all["variable"] == "t"]["n"].iloc[0]),
        "tied_within_tolerance": [model_to_cand[m] for m in tied],
        "selected": model_to_cand[chosen], "selected_context_days": cands[model_to_cand[chosen]]["context_days"],
    }
    write_json(OUT / "selection.json", result)
    log(f"selected {result['selected']} (primary {result['primary']}, tied {result['tied_within_tolerance']})")
    return result


def freeze_selection(result: dict, rule_path: Path) -> tuple[Path, Path]:
    """Commit the choice as YAML and derive the (not yet run) frozen test configuration."""
    rule = yaml.safe_load(open(rule_path))
    days = result["selected_context_days"]
    frozen = ROOT / "configs" / "frozen" / "context-selection-2023.yaml"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    rec = {"selected": result["selected"], "context_days": days, "context_steps": days * 8,
           "checkpoint": "mistralai/Timestral-28M@49b26409ac87b5935539a5d8ffe7f397b720db47",
           "variant": "multivariate (t, pmer, u, v) + calendar covariates", "rule": result["rule"],
           "primary_validation_pinball_t": result["primary"], "tied": result["tied_within_tolerance"],
           "decided_at": result["decided_at"], "validation_runs": result["runs"],
           "suite": rule["suite"], "protocol_sha256": result["protocol_sha256"],
           "note": "Chosen on 2023 validation only. 2024 smoke cases were inspected earlier and are exploratory, "
                   "not a pristine unseen test; the full 2024 test run has not been executed."}
    with open(frozen, "w") as f:
        f.write("# Frozen by `weather-lab compare --freeze`. Change only by a new, documented selection.\n")
        yaml.safe_dump(rec, f, sort_keys=False, allow_unicode=True)
    base = yaml.safe_load(open(ROOT / "configs" / "benchmark.yaml"))
    base.update(suite="synop-historical-v2", variant="test-frozen", label="benchmark test (frozen choices)",
                manifest="benchmarks/manifests/synop-historical-v2.test.json")
    base["origins"] = dict(base["origins"], splits=["test"])
    # same history window and eligibility rule as the validation suite the choice was made on
    val = load_config(ROOT / rule["candidates"][result["selected"]]["config"])
    base["forecast"] = dict(base["forecast"], context_days=val["forecast"]["context_days"])
    base["eligibility"] = dict(val["eligibility"])
    base["models"] = [m for m in base["models"] if not m["kind"] == "timestral"] + [
        {"id": f"timestral-28m-mv-cal-ctx{days:02d}", "kind": "timestral", "targets": ["t", "pmer", "u", "v"],
         "calendar": True, "context_days": days}]
    test_cfg = ROOT / "configs" / "benchmark_test_frozen.yaml"
    with open(test_cfg, "w") as f:
        f.write(f"# Full 2024 TEST evaluation with the context frozen in {frozen.relative_to(ROOT)}.\n"
                "# NOT run yet. Run once; do not tune on its results.\n")
        yaml.safe_dump(base, f, sort_keys=False, allow_unicode=True)
    return frozen, test_cfg


def export_combined(rule_path: Path, out_dir: Path, log=print) -> Path:
    """One run directory holding the shared baselines and every candidate (for the app and the snapshot).

    The candidates were run separately on one frozen manifest; here their predictions are
    concatenated (baselines once, verified identical) and re-scored with the standard scorer.
    """
    import shutil

    from weather_lab.evaluate import score_run

    rule = yaml.safe_load(open(rule_path))
    cands = rule["candidates"]
    runs = {k: latest_run(load_config(ROOT / c["config"])) for k, c in cands.items()}
    longest = max(cands, key=lambda k: cands[k]["context_days"])
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    preds = {k: pd.read_parquet(r / "predictions.parquet") for k, r in runs.items()}
    base = preds[longest][~preds[longest]["model"].str.startswith("timestral")]
    for p in preds.values():
        b = p[~p["model"].str.startswith("timestral")].reset_index(drop=True)
        pd.testing.assert_frame_equal(b.drop(columns="run_id"), base.reset_index(drop=True).drop(columns="run_id"))
    combined = pd.concat([base] + [p[p["model"] == cands[k]["model"]] for k, p in preds.items()], ignore_index=True)
    combined["run_id"] = "combined:" + "+".join(r.name for r in runs.values())
    combined.to_parquet(out_dir / "predictions.parquet", index=False)
    for f in ("targets.parquet", "manifest.json", "tail_thresholds.json", "observation_windows.parquet"):
        shutil.copy(runs[longest] / f, out_dir / f)
    cfg = yaml.safe_load(open(runs[longest] / "config.yaml"))
    cfg["experiment"] = "context-length-combined"
    cfg["models"] = cfg["models"][:3] + [{"id": cands[k]["model"], "kind": "timestral", "context_days":
                                          cands[k]["context_days"]} for k in cands]
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    run = read_json(runs[longest] / "run.json")
    systems = [s for s in run["systems"] if not s["id"].startswith("timestral")]
    for k, r in runs.items():
        systems += [s for s in read_json(r / "run.json")["systems"] if s["id"] == cands[k]["model"]]
    run.update(run_id=combined["run_id"].iloc[0], label="validation 2023 (history-length selection)",
               systems=systems, combined_from={k: r.name for k, r in runs.items()},
               selection=read_json(OUT / "selection.json") if (OUT / "selection.json").exists() else None)
    write_json(out_dir / "run.json", run)
    for name, df in score_run(out_dir).items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    log(f"combined run {out_dir} ({len(combined)} prediction rows)")
    return out_dir
