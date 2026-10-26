"""Rescore persisted continuation artifacts and build a common-pair development comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml

from weather_lab.common import sha256_file, write_json
from weather_lab.forecast import QUANTILE_LEVELS, qcol
from weather_lab.practical import PAIR, independent_rescore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    cpu, wall = time.process_time(), time.perf_counter()
    base = root / "data/evidence/practical-improvements-continuation-v1"
    directories = {"m1": base / "m1/scored", "spatial": base / "spatial/forecasts",
                   "t0": base / "t0/scored", "distribution": base / "distribution"}
    reports = {name: independent_rescore(path) for name, path in directories.items()}
    parent = yaml.safe_load((root / "docs/experiments/practical_improvements_v1.yaml").read_text())
    source = root / parent["source"]["run"]
    assert all(sha256_file(source / name) == expected for name, expected in parent["source"]["hashes"].items())
    targets = pd.read_parquet(source / "targets.parquet")
    chosen = {
        "m1": ["arome-raw", "arome-biascorr", "timestral-209m-mv-cal",
               "timestral-209m-mv-cal-arome-archived", "m1-anomaly-209m-v1",
               "m1-pooled-harmonic-climatology-v1", "m1-empirical-climatology-fit2018-2022-v1"],
        "spatial": ["spatial-patchmean-arome", "spatial-patchmean-biascorr", "spatial-patch-209m-v1"],
        "t0": ["t0-gamma-mv-cal-baseline-v1", "t0-gamma-mv-cal-trim-v1",
               "t0-gamma-mv-cal-trim-symmetric-v1", "t0-gamma-mv-cal-arome-archived-baseline-v1"],
        "distribution": ["arome-residual-quantiles-warmup-v1"],
    }
    frames, hashes = [], {}
    for name, path in directories.items():
        cases = pd.read_parquet(path / "cases.parquet")
        selection = cases.loc[cases.split.eq("development_selection") & cases.eligible, "case_id"]
        frame = pd.read_parquet(path / "predictions.parquet")
        frame = frame[frame.case_id.isin(selection) & frame.variable.eq("t") & frame.model.isin(chosen[name])].copy()
        assert set(frame.model) == set(chosen[name])
        frame["stage"] = name
        frames.append(frame)
        hashes[name] = {file: sha256_file(path / file) for file in
                        ("predictions.parquet", "targets.parquet", "cases.parquet", "summary.csv")}
    predictions = pd.concat(frames, ignore_index=True)
    assert not predictions.duplicated(["model"] + PAIR).any()
    rows = predictions.merge(targets[PAIR + ["observed"]], on=PAIR, validate="many_to_one")
    rows = rows[np.isfinite(rows.observed) & np.isfinite(rows.point)].copy()
    counts = rows.groupby(PAIR).model.nunique()
    common = counts[counts.eq(rows.model.nunique())].reset_index()[PAIR]
    rows = rows.merge(common, on=PAIR, validate="many_to_one")
    assert len(common) == 480 and rows.groupby("model").size().eq(480).all()
    reference = rows[rows.model.eq("spatial-patchmean-biascorr")][PAIR + ["point"]].rename(columns={"point": "reference"})
    rows = rows.merge(reference, on=PAIR, validate="many_to_one")
    rows["error"] = rows.point - rows.observed
    rows["absolute_error"] = abs(rows.error)
    rows["paired_delta"] = rows.absolute_error - abs(rows.reference - rows.observed)
    for name in ("arome_run", "arome_available_at"):
        if name in rows:
            rows[name] = pd.to_datetime(rows[name], utc=True)
    results = []
    for model, group in rows.groupby("model"):
        macro = group.groupby("station").absolute_error.mean().mean()
        stage = group.stage.iloc[0]
        stored = pd.read_csv(directories[stage] / "summary.csv")
        stored = stored[stored.period.eq("development_selection") & stored.variable.eq("t") & stored.model.eq(model)]
        assert len(stored) == 1 and abs(macro - stored.station_balanced_mae.iloc[0]) <= 1e-10
        record = {"model": model, "stage": stage, "period": "development_selection",
                  "variable": "t", "unit": "degC", "common_pairs": len(group),
                  "station_balanced_mae": macro, "micro_mae": group.absolute_error.mean(),
                  "rmse": np.sqrt(np.mean(group.error ** 2)), "bias": group.error.mean(),
                  "delta_vs_patchmean_corrector": group.groupby("station").paired_delta.mean().mean(),
                  "blocks_2d": group.origin.dt.floor("2D").nunique(),
                  "blocks_7d": group.origin.dt.floor("7D").nunique(),
                  "uncertainty": "withheld: fewer than ten time blocks", "confirmation": False}
        if group.output_type.eq("quantiles").all():
            error = group.observed.to_numpy()[:, None] - group[[qcol(q) for q in QUANTILE_LEVELS]].to_numpy()
            levels = np.asarray(QUANTILE_LEVELS)
            record["pinball"] = np.maximum(levels * error, (levels - 1) * error).mean()
            for nominal, lo, hi in ((80, .1, .9), (90, .05, .95)):
                record[f"coverage{nominal}"] = group.observed.between(group[qcol(lo)], group[qcol(hi)]).mean()
                record[f"width{nominal}"] = (group[qcol(hi)] - group[qcol(lo)]).mean()
        results.append(record)
    table = pd.DataFrame(results)
    reference_mae = table.loc[table.model.eq("spatial-patchmean-biascorr"), "station_balanced_mae"].iloc[0]
    table["relative_delta_vs_patchmean_corrector"] = table.station_balanced_mae / reference_mae - 1
    foundation = table[table.model.str.contains("209m|t0-gamma")]
    assert not foundation.station_balanced_mae.lt(.98 * reference_mae).any()
    out.mkdir(parents=True)
    table.to_csv(out / "temperature_selection.csv", index=False)
    rows.to_parquet(out / "common_temperature_pairs.parquet", index=False)
    rows.groupby(["model", "station"]).agg(n=("error", "size"), mae=("absolute_error", "mean"),
                                             bias=("error", "mean")).to_csv(out / "stations.csv")
    costs = {name: json.loads((base / file).read_text())["cpu_s"] for name, file in {
        "m1": "m1/runtime.json", "spatial_inference": "spatial/runtime.json",
        "spatial_preparation": "spatial/preparation.json", "t0": "t0/runtime.json",
        "distribution": "distribution/runtime.json"}.items()}
    write_json(out / "verification.json", {"stages": reports, "artifact_hashes": hashes,
               "tolerance": 1e-10, "common_temperature_pairs": len(common),
               "primary_station_balanced_metrics_recomputed": True, "source_hashes_unchanged": True})
    write_json(out / "selection.json", {"decision": "no_foundation_improvement", "engineering_target": .02,
               "reference": "spatial-patchmean-biascorr", "reference_mae": reference_mae,
               "best_foundation_candidate": foundation.loc[foundation.station_balanced_mae.idxmin(), "model"],
               "confirmation": "none; prospective frozen comparison required", "promotion": False})
    write_json(out / "runtime.json", {"completed_stage_cpu_s": costs,
               "consolidation_cpu_s": time.process_time() - cpu, "wall_s": time.perf_counter() - wall,
               "note": "Residual inventories, prior124.62CPU-s, setup and tests accounted separately; not an exact session total"})
    print(table[["model", "station_balanced_mae", "delta_vs_patchmean_corrector"]].to_string(index=False))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
