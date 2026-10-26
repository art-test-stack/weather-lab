"""Consolidate completed fixed arms and independently verify their aggregate metrics."""
import argparse
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from weather_lab.common import ROOT, read_json, sha256_file, write_json

sys.path.insert(0, str(ROOT / "experiments/t0-error-wrapper-v1"))
spec = importlib.util.spec_from_file_location("matched_error_reporting",
    ROOT / "experiments/t0-error-wrapper-v1/run_experiment.py")
reporting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporting)
sys.path.pop(0)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpu = time.process_time()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    root = ROOT / "data/evidence/error-features-v1"
    baseline = ROOT / "data/evidence/t0-error-wrapper-v1-verified/scored"
    paths = [root / "timestral" / f"{arm}-predictions.parquet" for arm in ("past-wind", "issued-atmosphere")]
    paths += [root / "timestral-combined/both-predictions.parquet"]
    paths += [root / "t0" / f"{arm}-predictions.parquet" for arm in ("past-wind", "issued-atmosphere", "both")]
    assert all(path.exists() for path in paths), "Missing arm: do not silently omit an approved experiment"
    frames = [pd.read_parquet(baseline / "predictions.parquet")] + [pd.read_parquet(p) for p in paths]
    targets, cases = (pd.read_parquet(baseline / f"{name}.parquet") for name in ("targets", "cases"))
    predictions = pd.concat(frames, ignore_index=True)
    key = ["case_id", "variable", "lead_h"]
    assert not predictions.duplicated(["model"] + key).any()
    assert predictions.model.nunique() == 10
    expected = pd.MultiIndex.from_frame(targets[key])
    for model, group in predictions.groupby("model"):
        assert len(group) == 1760
        assert pd.MultiIndex.from_frame(group[key]).sort_values().equals(expected.sort_values()), model
    provenance = {"stage": "error-features-v1", "kind": "development_descriptive_no_selection",
        "plan_sha256": sha256_file(ROOT / "experiments/error-features-v1/protocol.yaml"),
        "prediction_sources": {str(p.relative_to(ROOT)): sha256_file(p) for p in [baseline / "predictions.parquet"] + paths},
        "source_audit": read_json(root / "source_audit.json"), "models": sorted(predictions.model.unique())}
    reporting.score_outputs(out / "scored", predictions, targets, cases, provenance)
    summary = pd.read_csv(out / "scored/summary.csv")
    rows = predictions.merge(targets[key + ["observed"]], on=key, validate="many_to_one")
    differences = []
    for row in summary[summary.period.eq("all_development_descriptive")].itertuples():
        group = rows[rows.model.eq(row.model) & rows.variable.eq(row.variable)]
        err = group.point.to_numpy(dtype=float) - group.observed.to_numpy(dtype=float)
        assert len(group) == 880 and np.isfinite(err).all()
        for expected_value, actual in ((row.micro_mae, np.abs(err).mean()),
                (row.rmse, np.sqrt(np.mean(err**2))), (row.bias, err.mean())):
            differences.append(abs(expected_value - actual))
        if "error" in row.model:
            np.testing.assert_array_equal(group.point, group["q0.50"])
            np.testing.assert_allclose(group.point, group.arome_baseline + group.predicted_error, atol=1e-10, rtol=0)
    assert max(differences) < 1e-10
    write_json(out / "independent_overall_verification.json", {"checked_values": len(differences),
        "max_absolute_difference": max(differences), "tolerance": 1e-10, "same_cases": 55,
        "pairs_per_variable_per_model": 880})
    primary = summary[summary.period.eq("all_development_descriptive")].copy()
    primary.to_csv(out / "results.csv", index=False)
    candidates = [{"model": str(g.model.iloc[0]), "cases": 55, "pairs_per_variable": 880,
                   "status": "completed_development_diagnostic", "fallbacks": 0} for g in frames[1:]]
    write_json(out / "candidate_manifest.json", {"candidates": candidates, "confirmation": False,
        "selection": "none; all prescribed arms reported",
        "budget_addendum": str((root / "budget-addendum.yaml").relative_to(ROOT)),
        "budget_addendum_sha256": sha256_file(root / "budget-addendum.yaml")})
    print(primary[primary.variable.eq("t")][["model", "station_balanced_mae", "rmse"]].to_string(index=False))
    write_json(out / "runtime.json", {"cpu_seconds": time.process_time() - cpu})


if __name__ == "__main__":
    main()
