"""Mechanical application of the pre-declared feature-selection rule (configs/features/selection_rule.yaml).

Reads only saved validation tables (paired differences with block-bootstrap intervals, all-lead scores);
never looks at test data. The result is written as a frozen selection before the test is run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from weather_lab.common import now_utc, read_json


def _all(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    return df[(df["variable"] == variable) & (df["station"] == "ALL") & (df["season"] == "ALL")
              & df["lead_h"].isna()].set_index("model")


def evaluate_rule(run_dirs: list[Path], rule: dict) -> dict:
    point = pd.concat([pd.read_csv(d / "metrics_point.csv", dtype={"station": str}) for d in run_dirs])
    prob = pd.concat([pd.read_csv(d / "metrics_probabilistic.csv", dtype={"station": str}) for d in run_dirs])
    paired = pd.concat([pd.read_csv(d / "paired_matched.csv") for d in run_dirs])
    base, keep = rule["base"], rule["keep_if"]
    var = rule["primary"]["variable"]
    t_mae, p_mae, t_pin = _all(point, var)["mae"], _all(point, "pmer")["mae"], _all(prob, var)["pinball"]
    rows = []
    for group, mid in rule["candidates"].items():
        pr = paired[(paired["variable"] == var) & (paired["model"] == mid) & (paired["reference"] == base)
                    & (paired["loss"] == rule["primary"]["loss"]) & paired["lead_h"].isna()]
        if pr.empty or mid not in t_mae:
            rows.append({"group": group, "model": mid, "status": "not run"})
            continue
        r = pr.iloc[0]
        pmer_rel = float(p_mae[mid] / p_mae[base] - 1)
        pin_diff = float(t_pin[mid] - t_pin[base])
        kept = (pd.notna(r["ci95_high"]) and r["ci95_high"] < keep["ci95_high_below"]
                and pmer_rel <= keep["max_pmer_mae_increase"] and pin_diff <= keep["max_t_pinball_increase"])
        rows.append({"group": group, "model": mid, "t_mae": float(t_mae[mid]), "t_mae_base": float(t_mae[base]),
                     "t_mae_diff": float(r["mean_diff"]), "ci95": [float(r["ci95_low"]), float(r["ci95_high"])],
                     "n_blocks": int(r["n_blocks"]), "n_pairs": int(r["n_pairs"]), "pmer_mae_rel_change": pmer_rel,
                     "t_pinball_diff": pin_diff, "status": "kept" if kept else "rejected"})
    kept = [r for r in rows if r["status"] == "kept"]
    if not kept:
        selection, why = [], "no group met the rule: the selection is 'no features'"
    elif len(kept) == 1:
        selection, why = [kept[0]["group"]], "exactly one group kept"
    else:
        union = sorted(r["group"] for r in kept)
        combo = rule.get("combination_model", base + "-f-" + "-".join(union))
        best = min(kept, key=lambda r: r["t_mae"])
        if combo in t_mae:
            selection = union if t_mae[combo] <= best["t_mae"] else [best["group"]]
            why = f"combination {combo} {'≤' if selection == union else '>'} best single ({best['group']})"
        else:
            return {"status": "needs_combination_run", "combination_model": combo, "groups": union, "rows": rows}
    return {"status": "selected", "features": selection, "reason": why, "rows": rows}


def freeze(run_dirs: list[Path], rule_path: Path, out: Path) -> dict:
    rule = yaml.safe_load(rule_path.read_text())
    res = evaluate_rule(run_dirs, rule)
    if res["status"] != "selected":
        return res
    runs = [read_json(d / "run.json")["run_id"] for d in run_dirs]
    doc = {"frozen_at": now_utc(), "rule": str(rule_path), "validation_runs": runs, "base": rule["base"],
           "selected_features": res["features"], "reason": res["reason"], "evidence": res["rows"],
           "note": "Frozen before the 2025 test; the test is run once with the base and this selection."}
    out.write_text("# Written by `weather-lab features select`; do not edit by hand.\n" + yaml.safe_dump(doc, sort_keys=False))
    return {**res, "frozen": str(out)}
