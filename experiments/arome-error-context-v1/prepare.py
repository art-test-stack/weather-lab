"""Prepare the fixed prospective case plan and inventory inputs; never run or score a model."""
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
import sys
import time

import pandas as pd
import yaml

from weather_lab.common import now_utc, read_json, sha256_file, write_json
from weather_lab.context_inventory import inventory_cases
from weather_lab.context_protocol import candidate_identity, candidates, origins, schedule
from weather_lab.models import CHECKPOINTS
from weather_lab.practical_residual import issuance_lattice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New local evidence directory; never overwritten")
    parser.add_argument("--observations", type=Path, help="Explicit updated collection, fingerprinted separately")
    parser.add_argument("--forecasts", type=Path, help="Explicit updated issued archive, fingerprinted separately")
    args = parser.parse_args()
    cpu0, wall0 = time.process_time(), time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    protocol_path = root / "benchmarks/protocols/arome-error-context-v1.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    # This is a versioned design, not a configurable search that can silently change its cases.
    assert [protocol["schedule"][key] for key in ("history_days", "validation_origin_days",
            "purge_origin_days", "confirmation_origin_days")] == [90, 30, 2, 60]
    eligibility = protocol["case_plan"]["eligibility"]
    assert eligibility["history_windows_days"] == [30, 90]
    assert eligibility["minimum_observation_fraction_per_variable"] == .8
    assert eligibility["minimum_residual_fraction_per_variable"] == .8
    assert eligibility["observation_within_hours_per_variable"] == 24
    assert eligibility["observation_latency_hours"] == 0
    assert protocol["variables"] == ["t", "pmer"]
    assert protocol["case_plan"]["maximum_native_horizon_hours"] == 48
    assert protocol["case_plan"]["step_hours"] == 3
    grid = protocol["candidate_grid"]
    assert (grid["contexts_days"], grid["native_horizons_hours"], grid["optional_feature_groups"]) == ([30, 90], [24, 48], [])
    assert grid["prediction_target"] == "arome_error" and grid["conditioning"] == "futureonly" and not grid["calendar"]
    sources = {}
    for name, declared in protocol["inventory_inputs"].items():
        override = getattr(args, name, None)
        path = (override or root / declared["path"]).resolve()
        digest = sha256_file(path)
        if override is None and digest != declared["sha256"]:
            raise ValueError(f"Frozen inventory source changed: {path}; pass an updated collection explicitly")
        sources[name] = {"path": str(path), "sha256": digest, "updated_collection": override is not None}
    ck = CHECKPOINTS[protocol["model"]["name"]]
    if (ck.revision != protocol["model"]["checkpoint_revision"]
            or ck.sha256 != protocol["model"]["checkpoint_sha256"]):
        raise ValueError("Model registry differs from the frozen protocol")
    plan = schedule(protocol["collection_start"])
    assert plan["validation_window"]["start"] == protocol["schedule"]["validation_first_origin"]
    assert plan["confirmation_window"]["start"] == protocol["schedule"]["confirmation_first_origin"]
    assert plan["last_confirmation_label_time"] == protocol["schedule"]["last_confirmation_label"]
    planned = origins(protocol["collection_start"], protocol["stations"])
    assert planned.groupby("split").size().to_dict() == {
        "validation": protocol["case_plan"]["validation_cases_before_input_eligibility"],
        "confirmation": protocol["case_plan"]["confirmation_cases_before_input_eligibility"]}
    code_paths = [Path(__file__), root / "src/weather_lab/context_protocol.py",
                  root / "src/weather_lab/context_inventory.py", root / "src/weather_lab/arome.py",
                  root / "src/weather_lab/practical_residual.py", root / "src/weather_lab/forecast_config.py"]
    provenance = {"protocol_sha256": sha256_file(protocol_path), "sources": sources,
                  "model": protocol["model"], "residual_definition": protocol["candidate_grid"]["residual_definition"],
                  "code_sha256": {str(p.relative_to(root)): sha256_file(p) for p in code_paths},
                  "runtime": {"python": sys.version.split()[0], **{p: version(p) for p in ("numpy", "pandas", "torch")}}}
    candidate_records = {name: {"configuration": cfg.as_dict(), "identity": candidate_identity(cfg, provenance)}
                         for name, cfg in candidates().items()}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Write design and identities before any availability computation. No targets/predictions are loaded.
    (out / "protocol.yaml").write_text(protocol_path.read_text())
    write_json(out / "design.json", {"schedule": plan, "candidates": candidate_records,
                                    "provenance": provenance, "created_at": now_utc(),
                                    "artifact_type": "case_plan_and_availability_snapshot_not_final_scored_manifest"})
    planned.to_parquet(out / "planned_origins.parquet", index=False)
    obs = pd.read_parquet(sources["observations"]["path"])
    forecasts = pd.read_parquet(sources["forecasts"]["path"])
    development_manifest = read_json(Path(sources["development_manifest"]["path"]))
    development = pd.DataFrame(development_manifest["cases"])
    development["origin"] = pd.to_datetime(development.origin, utc=True)
    development["split"] = "development_already_observed"
    development = development[["case_id", "station", "origin", "split", "arome_run"]]
    pilot = inventory_cases(obs, forecasts, development.iloc[:1])
    pilot_cpu = time.process_time() - cpu0
    projected = pilot_cpu * len(development)
    if projected > protocol["budget"]["preparation_allocation_cpu_seconds"]:
        raise RuntimeError(f"Conservative inventory projection {projected:.1f}s exceeds the preparation allocation")
    actual = pd.concat([pilot, inventory_cases(obs, forecasts, development.iloc[1:])], ignore_index=True)
    prospective = inventory_cases(obs, forecasts, planned)
    actual.to_parquet(out / "development_input_availability.parquet", index=False)
    actual.to_csv(out / "development_input_availability.csv", index=False)
    prospective.to_parquet(out / "planned_input_availability.parquet", index=False)
    prospective.to_csv(out / "planned_input_availability.csv", index=False)
    fields = forecasts.groupby(["station", "variable"]).agg(
        first_run=("run", "min"), last_run=("run", "max"), runs=("run", "nunique"),
        first_valid=("valid_time", "min"), last_valid=("valid_time", "max"), rows=("value", "size"))
    fields.to_csv(out / "archive_by_station_field.csv")
    counts = actual.get("residual_t_90d_steps", pd.Series(dtype=float)).dropna()
    result = {"status": "archive_blocked" if not prospective.eligible.any() else "inputs_available_not_evaluated",
              "planned_cases": planned.groupby("split").size().to_dict(),
              "currently_eligible_planned_cases": int(prospective.eligible.sum()),
              "development_cases": len(actual), "development_cases_passing_shared_rule": int(actual.eligible.sum()),
              "development_residual_90d_steps_range": [int(counts.min()), int(counts.max())] if len(counts) else None,
              "required_residual_steps": {"30d": 192, "90d": 576},
              "issuance_lattice": issuance_lattice(forecasts),
              "pilot_cpu_seconds_including_setup": pilot_cpu, "projected_inventory_cpu_seconds": projected,
              "cpu_seconds": time.process_time() - cpu0, "wall_seconds": time.perf_counter() - wall0,
              "model_inference": False, "targets_read": False, "confirmation_scored": False,
              "next_step": "Collect missing vintage-checked daily runs and causal observation metadata; keep fixed dates. "
                           "Rerun with explicit updated source paths. Freeze the common eligible mask before computing errors."}
    write_json(out / "availability.json", result)
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
