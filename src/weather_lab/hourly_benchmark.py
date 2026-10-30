"""Versioned hourly 2026 replay from a verified archive; no model inference during preparation."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from weather_lab.common import ROOT, environment, git_revision, now_utc, read_json, sha256_file, sha256_text, ts, write_json
from weather_lab.forecast import hourly_inputs
from weather_lab.rag_archive import issued_forecasts, observations
from weather_lab.rag_store import Store
from weather_lab.suite import SEASONS

CONFIG = ROOT / "configs/hourly_2026_v3.yaml"


def configuration():
    return yaml.safe_load(CONFIG.read_text())


def inputs(obs, issued, case, cfg):
    """Causal inputs only; the verifying observation window is never read here."""
    batch = hourly_inputs(obs, pd.DataFrame([case]), list(cfg.targets), cfg.context_days * 24, cfg.horizon_h, 3)
    if cfg.conditioning != "none" or cfg.prediction_target == "arome_error" or "resid" in cfg.feature_names:
        from weather_lab.rag_arome_error import make_batch

        batch, _ = make_batch(obs, issued, case, list(cfg.targets), cfg.context_days * 24,
                              cfg.horizon_h, 3, "meteofrance_arome_france")
    return batch


def prepare(cfg, store=None):
    """Freeze every hourly origin through each station's last genuine 2026 measurement."""
    from weather_lab.scoring import score

    store = store or Store()
    if cfg.get("artifact_id"):
        frozen = store.get("suites", cfg["artifact_id"])
        manifest = read_json(frozen / "manifest.json")
        protocol = manifest["protocol"]
        if any((manifest["dataset"]["stations"] if k == "stations" else protocol.get(k)) != v
               for k, v in cfg.items() if k != "artifact_id"):
            raise ValueError("The pinned hourly suite is frozen; change the suite version for a new protocol or archive")
        return frozen
    archive = store.get("archives", cfg["archive_id"])
    variables = cfg["variables"]
    obs = observations(pd.read_parquet(archive / "observations.parquet"), variables)
    obs = obs[obs.station.isin([s["id"] for s in cfg["stations"]]) & (obs.time >= ts("2025-10-01"))].copy()
    issued = issued_forecasts(pd.read_parquet(archive / "issued.parquet"))
    issued = issued[issued["product"].eq(cfg["product"]) & issued.station.isin(obs.station.unique())].copy()
    cases, targets, predictions, references, availability = [], [], [], [], []
    for station, frame in obs.groupby("station", sort=True):
        series = frame.set_index("time")[variables]
        measured = frame[frame[variables].notna().any(axis=1) & frame.time.lt(ts("2027-01-01"))]
        last = measured.time.max()
        if pd.isna(last) or last < ts("2026-01-01"):
            continue
        # Pre-index actual issued values; selecting a run never combines different issues.
        runs = []
        for run, rows in issued[issued.station.eq(station)].groupby("run", sort=True):
            runs.append((run, rows.available_at.max(), rows.pivot(index="valid_time", columns="variable", values="value")))
        for origin in pd.date_range(ts("2026-01-01"), last, freq="h"):
            cid = f"{cfg['suite']}/{station}/{origin.strftime('%Y%m%dT%H')}"
            end = origin - pd.Timedelta(hours=3)
            endpoint = series.reindex([end]).to_numpy(dtype=np.float32, na_value=np.nan)[0]
            eligible = bool(np.isfinite(endpoint).all())
            case = {"case_id": cid, "station": station, "origin": str(origin), "split": "validation",
                    "season": SEASONS[origin.month], "eligible": eligible,
                    "reason": "" if eligible else "latest delayed observation missing"}
            cases.append(case)
            times = pd.date_range(origin + pd.Timedelta(hours=1), periods=48, freq="h")
            truth = series.reindex(times).to_numpy(dtype=np.float32, na_value=np.nan).T
            baseline, provenance = np.full((len(variables), 48), np.nan), {}
            for run, available, table in reversed(runs):
                if available > origin:
                    continue
                if run + pd.Timedelta(hours=71) < times[-1]:
                    break
                values = table.reindex(index=times, columns=variables).to_numpy(np.float32).T
                if np.isfinite(values).all():
                    baseline = values
                    provenance = {"arome_run": str(run), "arome_available_at": str(available)}
                    break
            for j, variable in enumerate(variables):
                for i, valid in enumerate(times):
                    row = {"case_id": cid, "variable": variable, "lead_h": i + 1, "valid_time": valid}
                    targets.append(row | {"observed": truth[j, i]})
                    common = row | {"suite": cfg["suite"], "run_id": cfg["suite"], "station": station,
                                    "origin": origin, "output_type": "point"}
                    predictions.append(common | {"model": "persistence", "point": endpoint[j] if eligible else np.nan})
                    references.append(common | {"model": "arome-raw", "point": baseline[j, i], **provenance})
            availability.append(case | {"status": "available" if eligible else "unavailable",
                                       "verified_targets": int(np.isfinite(truth).sum()),
                                       "expected_targets": truth.size,
                                       "arome_48h_available": bool(provenance)})
    if not cases:
        raise ValueError("archive has no measured 2026 observations for the selected stations")
    cases = pd.DataFrame(cases)
    pred, truth = pd.DataFrame(predictions), pd.DataFrame(targets)
    protocol = {k: v for k, v in cfg.items() if k != "artifact_id"}
    protocol["stations"] = [s["id"] for s in cfg["stations"]]
    protocol["forecast"] = {"step_hours": 1, "horizon_steps": 48, "context_days": 30, "observation_delay_hours": 3}
    temp_root = ROOT / ".tmp/codex/hourly-benchmark"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
        out = Path(tmp)
        pred.to_parquet(out / "predictions.parquet", index=False)
        truth.to_parquet(out / "targets.parquet", index=False)
        pd.DataFrame(references).to_parquet(out / "arome_reference.parquet", index=False)
        obs.to_parquet(out / "observation_windows.parquet", index=False)
        issued.to_parquet(out / "arome_station_extract.parquet", index=False)
        pd.DataFrame(availability).to_csv(out / "case_availability.csv", index=False)
        for name, table in score(pred, truth, cases.assign(origin=pd.to_datetime(cases.origin, utc=True)), {}, [6, 24, 48]).items():
            if name == "paired_differences":
                table = table.reindex(columns=["split", "variable", "model", "reference", "loss", "lead_h", "n_pairs",
                                               "n_blocks", "mean_diff", "ci95_low", "ci95_high"])
            table.to_csv(out / f"{name}.csv", index=False)
        manifest = {"suite": cfg["suite"], "created_at": now_utc(), "protocol": protocol,
                    "protocol_sha256": sha256_text(json.dumps(protocol, sort_keys=True)),
                    "dataset": {"stations": cfg["stations"]}, "cases": cases.to_dict("records"),
                    "n_cases": len(cases), "n_eligible": int(cases.eligible.sum()),
                    "leads_h": list(range(1, 49)), "label": cfg["label"]}
        write_json(out / "manifest.json", manifest)
        write_json(out / "run.json", {"suite": cfg["suite"], "run_id": cfg["suite"], "label": cfg["label"],
                   "created_at": now_utc(), "code_revision": git_revision(), "hourly_benchmark": True, "reference_only": True,
                   "archive_id": cfg["archive_id"], "n_cases": len(cases), "n_eligible": int(cases.eligible.sum()),
                   "systems": [{"id": "persistence", "kind": "persistence", "per_case_ms": None, "peak_rss_mb_after": None}],
                   "environment": environment(), "data": {"arome": {"sha256": sha256_file(out / "arome_station_extract.parquet")}},
                   "files_sha256": {p.name: sha256_file(p) for p in out.iterdir()}})
        return store.put("suites", out)


def command(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    print(prepare(cfg), flush=True)
