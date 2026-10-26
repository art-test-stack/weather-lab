"""Fit one approved warm-up AROME distribution and score only chronological selection cases."""
from __future__ import annotations

import argparse
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import yaml

from weather_lab.common import read_json, sha256_file, write_json
from weather_lab.forecast import QUANTILE_LEVELS
from weather_lab.practical import period_cases, save_outputs
from weather_lab.practical_distribution import MODEL, apply_distribution, fit_distribution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output or root / "data/evidence/practical-improvements-continuation-v1/distribution"
    if out.exists():
        raise FileExistsError(out)
    cpu, wall = time.process_time(), time.perf_counter()
    parent_path = root / "docs/experiments/practical_improvements_v1.yaml"
    parent = yaml.safe_load(parent_path.read_text())
    source = root / parent["source"]["run"]
    fingerprints = {str((source / name).relative_to(root)): sha256_file(source / name)
                    for name in ("predictions.parquet", "targets.parquet", "manifest.json")}
    for name in ("predictions.parquet", "targets.parquet", "manifest.json"):
        assert fingerprints[str((source/name).relative_to(root))] == parent["source"]["hashes"][name]
    out.parent.mkdir(parents=True, exist_ok=True)
    # The stage protocol is published first, separately from the scorer's output-directory creation.
    protocol_path = out.parent / (out.name + "_protocol.yaml")
    if protocol_path.exists():
        raise FileExistsError(protocol_path)
    warmup_start = pd.Timestamp("2026-10-10T06:00Z")
    warmup_end = pd.Timestamp("2026-10-13T06:00Z")
    cutoff = pd.Timestamp("2026-10-16T06:00Z")
    protocol = {"id": MODEL, "parent_protocol_sha256": sha256_file(parent_path), "source_fingerprints": fingerprints,
                "fit_origins": [warmup_start.isoformat(), warmup_end.isoformat()],
                "all_fit_labels_strictly_before": cutoff.isoformat(), "score_period": "development_selection_only",
                "pool": "all five stations, exact actual AROME run lead and UTC run cycle, separately per variable",
                "min_support": 20, "quantile_method": "numpy.quantile(method=linear)", "quantiles": list(QUANTILE_LEVELS),
                "observation_latency_h": 0, "source_forecast_anchor": "saved original-nearest AROME point float32",
                "fit_once": True, "local_station_fits": False, "independent_calibration": False,
                "coverage_guarantee": False, "confirmation": None, "budget_cpu_s": 120}
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False))
    saved = pd.read_parquet(source / "predictions.parquet")
    targets = pd.read_parquet(source / "targets.parquet")
    all_cases = period_cases(read_json(source / "manifest.json"))
    selected = all_cases[all_cases.split.eq("development_selection")].copy()
    warmup_raw = saved[saved.model.eq("arome-raw") & saved.origin.between(warmup_start, warmup_end)]
    training = warmup_raw.merge(targets[["case_id", "variable", "lead_h", "observed"]],
                               on=["case_id", "variable", "lead_h"], validate="one_to_one")
    training["observation_available_at"] = training.valid_time
    fit = fit_distribution(training, warmup_start=warmup_start, warmup_end=warmup_end, cutoff=cutoff,
                           source_identity=fingerprints)
    assert fit.groups.support.eq(20).all() and fit.groups.quantiles_available.all()
    comparators = saved[saved.case_id.isin(selected.case_id)].copy()
    comparators["policy"] = "verified_saved_original_nearest_selection_prediction"
    distribution = apply_distribution(comparators[comparators.model.eq("arome-raw")], fit)
    assert distribution.point.notna().all()
    predictions = pd.concat([comparators, distribution], ignore_index=True)
    selected_targets = targets[targets.case_id.isin(selected.case_id)].copy()
    provenance = {"stage": MODEL, "protocol_sha256": sha256_file(protocol_path), "fit_identity": fit.identity,
                  "source_fingerprints": fingerprints, "fit_label_max": fit.training.valid_time.max().isoformat(),
                  "fit_residuals": len(fit.training), "groups": len(fit.groups),
                  "distribution_evaluation": "October development selection only, no independent calibration",
                  "cpu_s_before_scoring": time.process_time() - cpu, "script_sha256": sha256_file(Path(__file__))}
    save_outputs(out, predictions, selected_targets, selected, provenance)
    fit.groups.to_parquet(out / "fit_quantiles.parquet", index=False)
    fit.training.to_parquet(out / "training_residuals.parquet", index=False)
    (out / "protocol.yaml").write_text(protocol_path.read_text())
    for path, fingerprint in fingerprints.items():
        assert sha256_file(root / path) == fingerprint
    summary = pd.read_csv(out / "summary.csv")
    print(summary[summary.period.eq("development_selection") & summary.variable.eq("t")][
        ["model", "valid_pairs", "station_balanced_mae", "pinball", "coverage80", "width80", "coverage90", "width90"]].to_string(index=False))
    write_json(out / "runtime.json", {"cpu_s": time.process_time()-cpu, "wall_s": time.perf_counter()-wall,
                                      "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6,
                                      "sources_unchanged": True, "inference": False})


if __name__ == "__main__":
    main()
