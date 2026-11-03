"""Point-in-time inference and paired scoring of frozen Benchmark configurations."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from weather_lab import models
from weather_lab.common import ROOT, environment, git_revision, now_utc, read_json, sha256_file
from weather_lab.forecast_config import ForecastConfig, build_adapter, check
from weather_lab.hourly_benchmark import inputs
from weather_lab.prospective_collect import PRODUCT, event, rows_json, stations
from weather_lab.prospective_store import digest, encoded
from weather_lab.saved_catalog import saved_configurations

PAIR = ["cohort", "case_id", "station", "variable", "valid_time", "lead_h", "issue", "cycle"]


def configuration(record):
    """Reconstruct only an exact, version-compatible saved identity."""
    cfg = ForecastConfig.make(record["model"], record["conditioning"],
        {f["name"]: f["params"] for f in record["features"]},
        calendar=record["calendar"], context_days=record["context_days"], horizon_h=record["horizon_h"],
        targets=tuple(record["targets"]), step_hours=record["step_hours"],
        prediction_target=record["prediction_target"], retrieval_method=record.get("retrieval_method", "none"),
        retrieval_index=record.get("retrieval_index", ""), precision=record.get("precision", {}).get("policy", "native"))
    if cfg.as_dict() != record:
        raise ValueError("saved configuration/feature version no longer reproduces exactly")
    issues = check(cfg, max_horizon_h=48)
    if issues:
        raise ValueError("; ".join(i.text() for i in issues))
    return cfg


def freeze_catalog(archive, suite):
    existing = archive.records("catalog")
    if existing:
        return [r["key"] for r in existing]
    records = []
    for path, item in saved_configurations(suite).items():
        source = read_json(ROOT / path / "run.json")
        source_hash = sha256_file(ROOT / path / "run.json")
        expected = item["summary"].get("run_json_sha256")
        if expected and expected != source_hash:
            raise ValueError("saved Benchmark catalog checksum mismatch")
        if source["forecast_config"] != item["config"]:
            raise ValueError("saved Benchmark summary/configuration mismatch")
        record = {"label": item["label"], "config": item["config"], "checkpoint": source["checkpoint"],
                  "source_run_sha256": source_hash, "source_suite": suite,
                  "adapter_version": source["adapter_version"], "preprocessing_version": source["preprocessing_version"]}
        if item["config"].get("retrieval_index"):
            from weather_lab.rag_store import Store

            store = Store()
            index = store.root / "indexes" / item["config"]["retrieval_index"]
            if (index / "artifact.json").is_file():
                Store.verify(index)
                record["retrieval_available_at"] = now_utc()
        configuration(record["config"])
        ident = digest(encoded(record))
        archive.put("catalog", ident, record)
        records.append(ident)
    if not records:
        raise ValueError("no saved Benchmark configurations installed; nothing substituted")
    event(archive, "catalog", "ok", configurations=records, suite=suite)
    return records


def observation_frame(archive, cutoff=None, *, replay=False):
    """Latest *then available* revision per station/day; targets never enter inputs."""
    versions = [r["value"] for r in archive.records("observations")]
    if cutoff is not None and not replay:
        versions = [r for r in versions if pd.Timestamp(r["collected_at"]) <= cutoff]
    selected = {}
    for rec in sorted(versions, key=lambda r: r["collected_at"]):
        selected[rec["identity"]] = rec
    rows = [row | {"observation_version": digest(encoded(rec)), "available_at": rec["collected_at"]}
            for rec in selected.values() for row in rec["rows"]]
    frame = pd.DataFrame(rows, columns=None if rows else ["station", "time", "t", "pmer", "u", "v",
                                                         "available_at", "observation_version"])
    frame["time"] = pd.to_datetime(frame.time, utc=True)
    frame["available_at"] = pd.to_datetime(frame.available_at, utc=True)
    if cutoff is not None:
        frame = frame[frame.time.le(cutoff)]
    # Qualified archive revisions can supersede an earlier real-time station sample.
    return frame.sort_values("available_at", kind="stable").drop_duplicates(["station", "time"], keep="last")


def forecast_frame(archive, cutoff=None, *, replay=False):
    rows = []
    for record in archive.records("forecasts"):
        rec = record["value"]
        if cutoff is not None and not replay and pd.Timestamp(rec["collected_at"]) > cutoff:
            continue
        if not replay and record["key"].startswith("historical/"):
            continue
        for row in rec["rows"]:
            row = dict(row)
            if replay and rec.get("published_at"):
                row["available_at"] = rec["published_at"]
                row["availability_basis"] = "official_archive_last_modified_retrospective"
            elif not replay:
                row["available_at"] = rec["collected_at"]
            rows.append(row | {"forecast_version": digest(encoded(rec))})
    cols = ["station", "run", "valid_time", "available_at", "variable", "value", "product",
            "availability_basis", "source_kind", "forecast_version"]
    frame = pd.DataFrame(rows, columns=None if rows else cols)
    for col in ("run", "available_at", "valid_time"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    if cutoff is not None:
        frame = frame[frame.available_at.le(cutoff) & frame.run.le(cutoff)]
    # First actually collected value wins, including overlapping partial files.
    return frame.sort_values("available_at", kind="stable").drop_duplicates(
        ["station", "product", "run", "valid_time", "variable"], keep="first")


def decisions(archive, start, end, replay=False):
    """One hourly-grid origin per discovered issue; late execution is recorded, never backdated."""
    frame = forecast_frame(archive, replay=replay)
    for run, rows in frame.groupby("run"):
        origin = rows.available_at.min().ceil("h")
        if start <= origin <= end:
            yield run, origin


def predict_config(cfg, record, batch, model_id, *, replay=False):
    ck = models.CHECKPOINTS[cfg.model]
    if {k: getattr(ck, k) for k in ("repo", "revision", "sha256")} != record["checkpoint"]:
        raise ValueError("checkpoint differs from frozen Benchmark checkpoint; prediction refused")
    models.prefetch(ck, allow_download=False)
    index = None
    if cfg.retrieval_method != "none":
        from weather_lab.rag_index import load_index
        from weather_lab.rag_store import Store

        available = record.get("retrieval_available_at")
        if not replay and (not available or pd.Timestamp(available) > pd.Timestamp(batch.cases.iloc[0].origin)):
            raise ValueError("frozen retrieval index was not recorded available at this decision")
        index = load_index(Store().get("indexes", cfg.retrieval_index))
        if index.meta.get("kind") != "arome_error" and pd.Timestamp(index.meta["end_exclusive"]) > pd.Timestamp("2026-01-01", tz="UTC"):
            raise ValueError("weather retrieval corpus must end before 2026")
    adapter = build_adapter(cfg, model_id, index=index, arome_fields=list(cfg.targets) if cfg.arome_mode else [])
    if (adapter.ADAPTER_VERSION != record["adapter_version"] or
            adapter.PREPROCESSING_VERSION != record["preprocessing_version"]):
        raise ValueError("adapter/preprocessing version differs from saved Benchmark")
    native = adapter.build_inputs(batch)
    if models.CHECKPOINTS[cfg.model].family != "t0":
        report = adapter.feature_rows(batch)[-1]
    else:
        report = native[-1]
    empty = [n for n, fraction in report.get("observed_fraction", {}).items() if fraction == 0]
    if empty:
        raise ValueError("selected feature rows entirely missing: " + ", ".join(empty))
    result = adapter.predict(batch)
    result.validate(batch)
    baseline = adapter.error_inputs(batch)[1] if cfg.prediction_target == "arome_error" else None
    return result.to_frame(batch, "prospective-v1", arome_baseline=baseline), {
        "adapter": adapter.describe(), "feature_report": report}


def infer(archive, start, end, *, replay=False, predict=predict_config, clock=now_utc):
    start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
    catalog = archive.records("catalog")
    if not catalog:
        raise ValueError("freeze the existing saved Benchmark catalog before inference")
    coords = {s["id"]: (s["lat"], s["lon"]) for s in stations()}
    code = {p.name: sha256_file(p) for p in sorted((ROOT / "src/weather_lab").glob("*.py"))}
    runtime = environment()
    count = 0
    for run, origin in decisions(archive, start, end, replay):
        if not replay and origin > pd.Timestamp(clock()):
            continue
        cohort = ("retrospective-" if replay else "prospective-") + origin.strftime("%Y-%m")
        obs = observation_frame(archive, origin, replay=replay)
        issued = forecast_frame(archive, origin, replay=replay)
        # Current issue only for future conditioning; all qualifying older issues remain for history.
        issued = issued[issued.run.le(run)]
        for station in coords:
            case_id = f"{cohort}/{station}/{run.isoformat()}/{origin.isoformat()}"
            common = {"case_id": case_id, "station": station, "origin": origin.isoformat(),
                      "issue": run.isoformat(), "cycle": run.hour, "cohort": cohort}
            times = pd.date_range(origin + pd.Timedelta(hours=1), periods=48, freq="h")
            own = issued[issued.run.eq(run) & issued.station.eq(station)]
            for entry in [{"key": "arome-raw", "value": None}, *catalog]:
                model, record = entry["key"], entry["value"]
                key = f"{case_id}/{model}"
                previous = archive.get("predictions", key)
                if previous is not None:
                    # Repair a crash between the primary record and its availability index.
                    archive.put("availability", key, {**common, "model": model, "status": previous["status"],
                        "reason": previous["reason"], "expected": len(previous["rows"]),
                        "predicted": sum(row["point"] is not None for row in previous["rows"])})
                    continue
                started = clock()
                if not replay and pd.Timestamp(started) < origin:
                    continue
                variables = ["t", "pmer"] if record is None else record["config"]["targets"]
                expected = [{**common, "variable": variable, "valid_time": valid.isoformat(),
                             "lead_h": i + 1, "issue_lead_h": int((valid - run).total_seconds() / 3600),
                             "model": model, "point": None}
                            for variable in variables for i, valid in enumerate(times)]
                info, reason, status = {}, "", "available"
                try:
                    if record is None:
                        table = own.pivot(index="valid_time", columns="variable", values="value")
                        frame = pd.DataFrame(expected)
                        frame["point"] = [table.get(v, pd.Series(dtype=float)).get(pd.Timestamp(t), np.nan)
                                          for v, t in zip(frame.variable, frame.valid_time)]
                    else:
                        cfg = configuration(record["config"])
                        if cfg.conditioning == "substobs":
                            raise ValueError("observation substitution is prohibited in prospective evaluation")
                        # Restrict even unconditioned models to this station's actually available past.
                        station_obs = obs[obs.station.eq(station)]
                        case = pd.Series({**common, "origin": origin}, name=case_id)
                        batch = inputs(station_obs, issued, case, cfg)
                        if batch.arome_inputs is not None and not np.all(
                                batch.arome_inputs.future_run[np.isfinite(batch.arome)] == run.to_datetime64()):
                            raise ValueError("conditioning selected another issue")
                        batch.station_coords = coords
                        frame, info = predict(cfg, record, batch, model, replay=replay)
                        # Adapter output is aligned by explicit keys, never by row order.
                        actual = frame[["variable", "lead_h", "point"]]
                        if actual.duplicated(["variable", "lead_h"]).any():
                            raise ValueError("duplicate adapter prediction keys")
                        frame = pd.DataFrame(expected).drop(columns="point").merge(actual,
                            on=["variable", "lead_h"], how="left", validate="one_to_one")
                    finished = clock()
                    frame["point"] = pd.to_numeric(frame.point, errors="coerce")
                    late = pd.to_datetime(frame.valid_time, utc=True).le(pd.Timestamp(finished))
                    if not replay:
                        frame.loc[late, "point"] = np.nan
                    missing = ~np.isfinite(frame.point)
                    frame["exclusion"] = np.where(missing, "late_freeze_or_missing_prediction", "")
                    if missing.any():
                        status = "partial" if (~missing).any() else "unavailable"
                except Exception as exc:
                    finished, reason, status = clock(), f"{type(exc).__name__}: {exc}", "unavailable"
                    frame = pd.DataFrame(expected).assign(exclusion=reason)
                provenance = {"configuration": record, "started_at": started, "finished_at": finished,
                    "decision_cutoff": origin.isoformat(), "code_revision": git_revision(),
                    "code_sha256": code, "environment": runtime,
                    "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
                    "source_records": sorted(set(obs.observation_version) | set(issued.forecast_version)),
                    "input_sha256": digest(encoded({"observations": rows_json(obs), "issued": rows_json(issued)})),
                    "availability_policy": "retrospective: source forecast availability evidence/assumption; observation bridge 3h" if replay
                                           else "actual collection cutoff; freeze strictly before valid time",
                    "inference": json.loads(json.dumps(info, default=str))}
                rec = {**common, "model": model, "status": status, "reason": reason,
                       "rows": rows_json(frame), "provenance": provenance}
                archive.put("predictions", key, rec)
                archive.put("availability", key, {**common, "model": model, "status": status, "reason": reason,
                                                   "expected": len(expected), "predicted": int(frame.point.notna().sum())})
                count += 1
    event(archive, "inference", "ok", frozen_records=count, replay=replay)
    return count


def evaluate(archive, selected=None):
    """Score identical support across selected configurations and raw AROME, per cohort/variable."""
    records = archive.records("predictions")
    rows = [row | {"prediction_record": digest(encoded(rec["value"]))}
            for rec in records for row in rec["value"]["rows"]]
    # Missing historical issues have an availability ledger, never fabricated forecasts.
    present = {(row["case_id"], row["model"]) for row in rows}
    for record in archive.records("availability"):
        rec = record["value"]
        if not record["key"].startswith("missing-issue/") or (rec["case_id"], rec["model"]) in present:
            continue
        for variable in ("t", "pmer"):
            for lead in range(1, 49):
                rows.append({k: rec[k] for k in ("cohort", "case_id", "station", "model", "issue", "cycle")} |
                    {"variable": variable, "lead_h": lead, "point": None, "prediction_record": None,
                     "valid_time": (pd.Timestamp(rec["origin"]) + pd.Timedelta(hours=lead)).isoformat()})
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    frame = pd.DataFrame(rows)
    if selected is None:
        selected = sorted(set(frame.model) - {"arome-raw"})
    required = set(selected) | {"arome-raw"}
    frame = frame[frame.model.isin(required)].copy()
    obs = observation_frame(archive)
    truth = obs.melt(id_vars=["station", "time", "observation_version"], value_vars=["t", "pmer"],
                     var_name="variable", value_name="observed")
    frame["valid_time"] = pd.to_datetime(frame.valid_time, utc=True)
    frame["point"] = pd.to_numeric(frame.point, errors="coerce")
    frame = frame.merge(truth, left_on=["station", "valid_time", "variable"],
                        right_on=["station", "time", "variable"], how="left", validate="many_to_one")
    frame["observed"] = pd.to_numeric(frame.observed, errors="coerce")
    if frame.duplicated(PAIR + ["model"]).any():
        raise ValueError("duplicate prediction pairs")
    finite = frame[np.isfinite(frame.point) & np.isfinite(frame.observed)]
    counts = finite.groupby(PAIR).model.nunique()
    support = counts[counts.eq(len(required))].index
    frame["paired"] = pd.MultiIndex.from_frame(frame[PAIR]).isin(support)
    frame["exclusion"] = np.select([~np.isfinite(frame.point), ~np.isfinite(frame.observed), ~frame.paired],
        ["missing_or_ineligible_prediction", "missing_observation", "not_on_common_support"], default="")
    paired = frame[frame.paired].copy()
    if paired.empty:
        return pd.DataFrame(), frame
    raw = paired[paired.model.eq("arome-raw")][PAIR + ["point"]].rename(columns={"point": "arome"})
    paired = paired.merge(raw, on=PAIR, validate="many_to_one")
    paired["abs_err"] = abs(paired.point - paired.observed)
    paired["sq_err"] = (paired.point - paired.observed) ** 2
    paired["arome_abs_err"] = abs(paired.arome - paired.observed)
    paired["date"] = paired.valid_time.dt.strftime("%Y-%m-%d")
    group = ["cohort", "variable", "model", "station", "date", "cycle", "lead_h"]
    metrics = paired.groupby(group, as_index=False).agg(mae=("abs_err", "mean"),
        mse=("sq_err", "mean"), mae_arome=("arome_abs_err", "mean"), n=("point", "size"))
    metrics["rmse"] = np.sqrt(metrics.pop("mse"))
    metrics["gain"] = metrics.mae_arome - metrics.mae
    metrics["absolute_gain"] = abs(metrics.gain)
    metrics["support_sha256"] = digest(encoded(rows_json(support.to_frame(index=False))))
    return metrics, frame


def score(archive):
    metrics, coverage = evaluate(archive)
    value = {"metrics": rows_json(metrics), "coverage": rows_json(coverage),
             "prediction_records": [digest(encoded(r["value"])) for r in archive.records("predictions")],
             "observation_records": [digest(encoded(r["value"])) for r in archive.records("observations")],
             "scorer_sha256": sha256_file(ROOT / "src/weather_lab/prospective.py")}
    key = digest(encoded(value))
    previous = archive.get("evaluations", key)
    if previous is not None:
        return previous
    value["created_at"] = now_utc()
    archive.put("evaluations", key, value)
    event(archive, "scoring", "ok", paired_rows=int(coverage.paired.sum()) if not coverage.empty else 0)
    return value
