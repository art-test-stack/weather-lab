"""Inventory fixed-lead residual support and C2 recovery metadata, without efficacy inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import requests
import yaml

from weather_lab.common import read_json, sha256_file, write_json
from weather_lab.practical_residual import fixed_lead_history, issuance_lattice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe-recovery-head", action="store_true", help="HEAD only; never download GRIB payloads")
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output or root / "data/evidence/practical-improvements-continuation-v1/residual"
    cpu0, wall0 = time.process_time(), time.perf_counter()
    protocol_file = root / "docs/experiments/practical_improvements_v1.yaml"
    protocol = yaml.safe_load(protocol_file.read_text())
    source = root / protocol["source"]["run"]
    arome_path = root / protocol["source"]["arome"]["file"]
    obs_path = root / protocol["source"]["observations"]["file"]
    fingerprints = {str(p.relative_to(root)): sha256_file(p) for p in
                    [arome_path, obs_path, protocol_file, source / "predictions.parquet", source / "targets.parquet"]}
    assert fingerprints[str(arome_path.relative_to(root))] == protocol["source"]["arome"]["sha256"]
    assert fingerprints[str(obs_path.relative_to(root))] == protocol["source"]["observations"]["sha256"]
    out.mkdir(parents=True, exist_ok=False)
    stage = {"id": "fixed-lead-residual-builder-v1", "parent_protocol_sha256": sha256_file(protocol_file),
             "context_days": 30, "min_observed": 24, "support_sensitivity": [24, 30],
             "source_fingerprints": fingerprints, "model_inference": False,
             "rule": "observation-minus-exact-init=valid-L issued forecast, available by origin and valid time",
             "fallback": "unavailable R2; separately labelled raw AROME fallback if selected-run anchor exists",
             "no_interpolation": True, "observation_latency_h": 0, "budget_cpu_s": 240,
             "head_probe_only": args.probe_recovery_head, "confirmation": None}
    (out / "protocol.yaml").write_text(yaml.safe_dump(stage, sort_keys=False))
    arome, obs = pd.read_parquet(arome_path), pd.read_parquet(obs_path)
    raw = pd.read_parquet(source / "predictions.parquet")
    raw = raw[raw.model.eq("arome-raw")].copy()
    summaries, histories = [], []
    pilot_cpu = None
    for index, row in enumerate(raw.itertuples()):
        got = fixed_lead_history(arome, obs, station=row.station, variable=row.variable,
                                 origin=row.origin, selected_run=pd.Timestamp(row.arome_run), target_valid=row.valid_time)
        summary = {"case_id": row.case_id, "requested_lead_h": row.lead_h, **got.info,
                   "ready_min30": got.info["observed_slots"] >= 30 and np.isfinite(got.info["anchor"])}
        summaries.append(summary)
        histories.append(got.rows.assign(case_id=row.case_id, station=row.station, variable=row.variable,
                                         origin=row.origin, fixed_run_lead_h=got.info["fixed_run_lead_h"]))
        if index == 31:
            pilot_cpu = time.process_time() - cpu0
            estimate = pilot_cpu * len(raw) / 32
            if estimate > 230:
                raise RuntimeError(f"Projected CPU cost {estimate:.1f}s exceeds residual budget")
    summary = pd.DataFrame(summaries)
    summary.to_parquet(out / "support.parquet", index=False)
    pd.concat(histories, ignore_index=True).to_parquet(out / "histories.parquet", index=False)
    summary.groupby(["origin", "variable", "fixed_run_lead_h"]).agg(
        observed_min=("observed_slots", "min"), observed_max=("observed_slots", "max"),
        observed_fraction=("observed_fraction", "mean"), ready_min24=("ready", "sum"),
        ready_min30=("ready_min30", "sum")).reset_index().to_csv(out / "support_by_origin_lead.csv", index=False)
    assert not summary.ready.any(), "New sufficient archive support requires a separately frozen inference protocol"
    # These are explicit integration fallbacks, never labelled completed residual-model predictions.
    fallbacks = raw.merge(summary[["case_id", "variable", "requested_lead_h", "policy", "fallback_model"]],
                          left_on=["case_id", "variable", "lead_h"],
                          right_on=["case_id", "variable", "requested_lead_h"], validate="one_to_one")
    fallbacks["requested_model"] = "r2-fixed-lead-v1"
    fallbacks["model"] = "r2-unavailable-fallback-arome-raw"
    fallbacks["efficacy_prediction"] = False
    fallbacks.to_parquet(out / "fallback_predictions.parquet", index=False)
    pd.read_parquet(source / "targets.parquet").to_parquet(out / "targets.parquet", index=False)
    recovery = []
    for record_path in sorted((root / "data/raw/arome").glob("*/run.json")):
        record = read_json(record_path)
        first = next(f for f in record["files"] if "00H06H" in f["file"])
        subset = record_path.parent / first["subset"]
        assert sha256_file(subset) == first["subset_sha256"]
        item = {"run": record["run"], "url": first["url"], "installed_leads": sorted({m["lead_h"] for m in first["messages"]}),
                "source_message_count": first["n_messages_in_file"], "installed_message_count": len(first["messages"]),
                "requested_missing_leads": [0, 3], "payload_recovered": False}
        if args.probe_recovery_head:
            try:
                response = requests.head(first["url"], timeout=20)
                item.update(http_status=response.status_code,
                            headers={k: response.headers.get(k) for k in ("Content-Length", "ETag", "Last-Modified", "Accept-Ranges")})
                item["same_vintage_headers"] = all(response.headers.get(k) == first["http"].get(k)
                                                   for k in ("Content-Length", "ETag", "Last-Modified"))
            except requests.RequestException as exc:
                item["probe_error"] = str(exc)
        recovery.append(item)
    write_json(out / "c2_recovery_inventory.json", recovery)
    for path, fingerprint in fingerprints.items():
        assert sha256_file(root / path) == fingerprint
    report = {"builder_implemented": True, "efficacy_status": "archive_blocked_no_model_inference",
              "lattice": issuance_lattice(arome), "history_count": len(summary),
              "max_observed_slots": int(summary.observed_slots.max()), "required_slots": 30,
              "ready_min24": int(summary.ready.sum()), "ready_min30": int(summary.ready_min30.sum()),
              "fallback_pairs": len(fallbacks), "all_fallback_values_equal_saved_raw": bool(np.array_equal(fallbacks.point, raw.point)),
              "collection_days_for_30day_context_through_longest_lead": 32,
              "r1_status": "not_fitted;12 daily runs lack independent fit/selection/calibration support",
              "c2_status": "missing installed leads0/3;HEAD metadata is not proof of field availability",
              "c2_next": "new frozen extraction ID plus bounded header scan and field download required; do not overwrite original",
              "sources_unchanged": True, "pilot_32histories_cpu_s": pilot_cpu,
              "cpu_s": time.process_time() - cpu0, "wall_s": time.perf_counter() - wall0,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
              "script_sha256": sha256_file(Path(__file__))}
    write_json(out / "run.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
