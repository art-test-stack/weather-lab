"""Rolling 2026 replay; missing cases stay visible and comparisons share exact supports."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from weather_lab.common import git_revision, now_utc, sha256_file, sha256_text, ts, write_json
from weather_lab.forecast import CaseBatch, ForecastBatch
from weather_lab.rag_adapters import Analog, foundation
from weather_lab.rag_archive import issued_forecasts, observations
from weather_lab.rag_index import Index
from weather_lab.rag_store import Store


def metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Paired AROME comparisons, plus an all-systems common set; no cross-support ranking."""
    rows = []
    for variable, frame in predictions.groupby("variable", sort=True):
        pivot = frame.pivot(index=["case_id", "lead_h"], columns="model", values="point")
        truth = frame.drop_duplicates(["case_id", "lead_h"]).set_index(["case_id", "lead_h"])["observed"]
        for scope in ("paired_vs_arome", "all_systems_common"):
            for model in pivot.columns:
                mask = np.isfinite(truth) & np.isfinite(pivot[model]) & np.isfinite(pivot["arome-raw"])
                if scope == "all_systems_common":
                    mask &= np.isfinite(pivot).all(axis=1)
                for lead in [0] + sorted(pivot.index.get_level_values("lead_h").unique()):
                    selected = mask if lead == 0 else mask & (pivot.index.get_level_values("lead_h") == lead)
                    keys = pivot.index[selected]
                    y, p, a = truth.loc[keys], pivot.loc[keys, model], pivot.loc[keys, "arome-raw"]
                    mae = float(np.mean(np.abs(p - y))) if len(y) else np.nan
                    baseline = float(np.mean(np.abs(a - y))) if len(y) else np.nan
                    rows.append({"scope": scope, "variable": variable, "model": model, "lead_h": int(lead),
                                 "n": len(y), "mae": mae,
                                 "rmse": float(np.sqrt(np.mean((p - y) ** 2))) if len(y) else np.nan,
                                 "arome_mae": baseline, "skill_mae_vs_arome": 1 - mae / baseline if baseline > 0 else np.nan,
                                 "support_sha256": sha256_text(json.dumps(list(keys), default=str)),
                                 "lead_scope": "all positive leads" if lead == 0 else "single lead"})
    return pd.DataFrame(rows)


def choose_arome(frame, station, origin, times, variables, product):
    if frame is None:
        raise ValueError("no issued-run archive; stitched historical series are not a forecast substitute")
    valid = frame[(frame.station == station) & (frame["product"] == product)
                  & (frame.available_at <= origin) & (frame.run <= origin)]
    # Never stitch several runs to construct a complete forecast horizon.
    for run in sorted(valid.run.unique(), reverse=True):
        selected = valid[valid.run == run]
        table = selected.pivot(index="variable", columns="valid_time", values="value").reindex(
            index=variables, columns=times)
        if np.isfinite(table.to_numpy(dtype=float)).all():
            return table.to_numpy(np.float32), {"run": str(run),
                    "available_at": str(selected.available_at.max()),
                    "availability_basis": sorted(selected.availability_basis.unique())}
    raise ValueError("no single available issued run covers every requested variable and lead")


def predict_cpu(adapter, batches):
    results = []
    for batch in batches:
        try:
            result = adapter.predict(batch)
            result.validate(batch)
            results.append((result, ""))
        except Exception as exc:
            results.append((None, f"{type(exc).__name__}: {exc}"))
    return results


def run(index_id: str, cfg: dict, horizon_h: int, store: Store | None = None,
        predict=predict_cpu, progress=None, log=print) -> Path:
    store = store or Store()
    if horizon_h not in cfg["horizons_hours"]:
        raise ValueError("horizon must be predeclared in the RAG protocol")
    index = Index(store.get("indexes", index_id))
    archive = store.get("archives", index.meta["archive_id"])
    obs = observations(pd.read_parquet(archive / "observations.parquet"), cfg["variables"])
    issued = issued_forecasts(pd.read_parquet(archive / "issued.parquet")) if (archive / "issued.parquet").is_file() else None
    if (cfg["variables"] != index.meta["variables"] or cfg["context_hours"] != index.meta["context_hours"]
            or cfg["observation_delay_hours"] != index.meta["observation_delay_hours"]
            or cfg["retrieval"] != index.meta["retrieval"]):
        raise ValueError("configuration differs from the frozen index; build a new index")
    start, end = ts(cfg["evaluation"]["start"]), ts(cfg["evaluation"]["end_exclusive"])
    if start != ts("2026-01-01") or end != ts("2027-01-01"):
        raise ValueError("this experiment retains the entire requested 2026 window")
    delay, t = int(cfg["observation_delay_hours"]), cfg["context_hours"]
    if delay < 0 or not cfg.get("observation_availability"):
        raise ValueError("declare the historical observation availability assumption")
    h = horizon_h + delay
    methods, k = cfg["retrieval"]["methods"], cfg["retrieval"]["k"]
    adapters = [Analog(f"analogue-{m}", index, m, k) for m in methods]
    for name in cfg["foundation_models"]:
        adapters.extend([foundation(name, index)] + [foundation(name, index, m, k) for m in methods])
    model_ids = ["persistence", "arome-raw"] + [a.id for a in adapters]
    origins = pd.date_range(start + pd.Timedelta(hours=cfg["origin_hour_utc"]), end, freq="D", inclusive="left")
    per_station = {s: f.set_index("time") for s, f in obs.groupby("station")}
    cases, availability, predictions, batches, result_frames = [], [], [], [], []
    created, started = now_utc(), time.perf_counter()
    # One run always has a single product and horizon. Missing/future dates remain in the case manifest.
    for station in [s["id"] for s in cfg["stations"]]:
        for origin in origins:
            case_id = f"{station}/{origin.isoformat()}"
            future = pd.date_range(origin + pd.Timedelta(hours=1), periods=horizon_h, freq="h")
            source = per_station.get(station, pd.DataFrame(columns=cfg["variables"], index=pd.DatetimeIndex([], tz="UTC")))
            truth = source.reindex(future)[cfg["variables"]].to_numpy(np.float32, copy=True).T
            truth[:, future > pd.Timestamp.now(tz="UTC")] = np.nan
            cases.append({"case_id": case_id, "station": station, "origin": origin,
                          "observed_targets": int(np.isfinite(truth).sum()), "expected_targets": truth.size})
            for j, variable in enumerate(cfg["variables"]):
                for lead in range(1, horizon_h + 1):
                    predictions.append({"case_id": case_id, "station": station, "origin": origin,
                        "valid_time": future[lead - 1], "variable": variable, "lead_h": lead,
                        "observed": truth[j, lead - 1]})
            def status(mid, state, reason="", **extra):
                availability.append({"case_id": case_id, "model": mid, "status": state, "reason": reason, **extra})
            if not np.isfinite(truth).any():
                for mid in model_ids:
                    status(mid, "unavailable", "no verifying observations in this forecast window")
                continue
            hist_end = origin - pd.Timedelta(hours=delay)
            history_times = pd.date_range(end=hist_end, periods=t, freq="h")
            hist = source.reindex(history_times)[cfg["variables"]].to_numpy(np.float32).T
            if ((np.isfinite(hist).mean(axis=-1) < cfg["min_context_observed_fraction"]).any()
                    or not np.isfinite(hist[:, -1]).all()):
                for mid in model_ids:
                    if mid != "arome-raw":
                        status(mid, "unavailable", "insufficient causal observation context")
                batch = None
            else:
                model_times = pd.date_range(hist_end + pd.Timedelta(hours=1), periods=h, freq="h")
                case = pd.DataFrame([{"case_id": case_id, "station": station, "origin": origin}])
                batch = CaseBatch(case, cfg["variables"], hist[None],
                                   np.array([hist_end.tz_localize(None)], dtype="datetime64[ns]"),
                                   model_times.tz_localize(None).to_numpy(dtype="datetime64[ns]")[None], 1)
                batches.append(batch)
                fb = ForecastBatch("persistence", "v1", "point", cfg["variables"],
                                   np.broadcast_to(hist[None, :, -1:], (1, len(cfg["variables"]), h)).copy())
                status("persistence", "available")
                result_frames.append(fb.to_frame(batch, cfg["experiment"]))
            try:
                values, provenance = choose_arome(issued, station, origin, future, cfg["variables"], cfg["product"])
                status("arome-raw", "available", **provenance)
                # Store AROME directly at decision-relative leads; the model bridge is never scored.
                result_frames.append(pd.DataFrame([
                    {"case_id": case_id, "variable": v, "lead_h": lead, "model": "arome-raw",
                     "point": values[j, lead - 1], "model_revision": "named-issued-run",
                     "output_type": "point"}
                    for j, v in enumerate(cfg["variables"]) for lead in range(1, horizon_h + 1)]))
            except ValueError as exc:
                status("arome-raw", "unavailable", str(exc))
    audits, system_records = {}, []
    for number, adapter in enumerate(adapters, 1):
        log(f"{adapter.id}: {len(batches)} cases")
        # Refuse implicit model downloads; model setup is a separate explicit action.
        setup_error = ""
        if hasattr(adapter, "checkpoint"):
            try:
                from weather_lab.models import prefetch, resolve
                prefetch(resolve(adapter.checkpoint, adapter.revision), allow_download=False)
                if adapter.family == "t0":
                    from importlib.metadata import version
                    if version("tfc-t0") != "0.6.0":
                        raise ValueError("t0 requires the pinned tfc-t0==0.6.0 runtime")
            except Exception as exc:
                setup_error = f"{type(exc).__name__}: {exc}"
        for offset in range(0, len(batches), 8):
            chunk = batches[offset:offset + 8]
            results = [(None, setup_error)] * len(chunk) if setup_error else predict(adapter, chunk)
            if len(results) != len(chunk):
                raise ValueError("prediction worker returned an incomplete chunk")
            for batch, (result, reason) in zip(chunk, results):
                case_id = batch.cases.case_id.iloc[0]
                if result is not None:
                    result.validate(batch)
                    result_frames.append(result.to_frame(batch, cfg["experiment"]))
                    audits[f"{case_id}/{adapter.id}"] = result.info
                availability.append({"case_id": case_id, "model": adapter.id,
                                     "status": "available" if result is not None else "failed", "reason": reason})
            if progress:
                progress(((number - 1) + (offset + len(chunk)) / max(len(batches), 1)) / len(adapters),
                         desc=f"{adapter.id}: {offset + len(chunk)}/{len(batches)}")
        system_records.append(adapter.describe())
    keys = ["case_id", "variable", "lead_h", "model"]
    base = pd.DataFrame(predictions).merge(pd.DataFrame({"model": model_ids}), how="cross")
    if result_frames:
        results = pd.concat(result_frames, ignore_index=True)
        keep = keys + [c for c in results if c in ("point", "output_type", "model_revision") or c.startswith("q0.")]
        preds = base.merge(results[keep], on=keys, how="left", validate="one_to_one")
    else:
        preds = base.assign(point=np.nan)
    scores = metrics(preds)
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for package in ("numpy", "pandas", "pyarrow", "torch", "timestral", "tfc-t0"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        preds.to_parquet(out / "predictions.parquet", index=False)
        pd.DataFrame(cases).to_parquet(out / "cases.parquet", index=False)
        pd.DataFrame(availability).to_csv(out / "availability.csv", index=False)
        scores.to_csv(out / "metrics_point.csv", index=False)
        write_json(out / "case_info.json", audits)
        write_json(out / "config.json", cfg)
        write_json(out / "run.json", {"version": 1, "run_id": uuid4().hex, "created_at": created,
                   "experiment": cfg["experiment"], "development": True, "code_revision": git_revision(),
                   "index_id": index_id, "archive_id": index.meta["archive_id"], "horizon_hours": horizon_h,
                   "product": cfg["product"], "systems": system_records, "configuration": cfg,
                   "status": "completed_with_gaps" if any(r["status"] != "available" for r in availability) else "completed",
                   "wall_s": time.perf_counter() - started, "requested_cases": len(cases),
                   "cases_with_context": len(batches), "observation_availability": cfg["observation_availability"],
                   "runtime_versions": versions,
                   "code_sha256": {p.name: sha256_file(p) for p in Path(__file__).parent.glob("rag*.py")},
                   "observation_delay_hours": delay, "selection": "predeclared pilot; no tuning on 2026 outcomes",
                   "comparison": "rank only all_systems_common; paired_vs_arome supports can differ",
                   "data_sha256": {p.name: sha256_file(p) for p in archive.glob("*.parquet")}})
        return store.put("runs", out)
