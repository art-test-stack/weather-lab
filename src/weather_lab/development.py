"""Explicit full-case development benchmarks, separate from frozen runs and exploratory caches."""
from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import time
from dataclasses import fields, replace
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from weather_lab import features, models
from weather_lab.common import ROOT, environment, git_revision, now_utc, read_json, sha256_file, sha256_text, write_json
from weather_lab.scoring import score
from weather_lab.forecast import QUANTILE_LEVELS, exploratory_inputs, qcol
from weather_lab.forecast_config import ForecastConfig, build_adapter, check
from weather_lab.predcache import make_key
from weather_lab.suite import cases_from_manifest


def location() -> Path:
    from weather_lab.bootstrap import load_bootstrap_config

    bootstrap = load_bootstrap_config()["bootstrap"]
    bucket = (bootstrap.get("caches", {}).get("predictions") or {}).get("bucket")
    mount = (bootstrap.get("snapshot", {}).get("mounts") or {}).get(bucket)
    if mount and Path(mount).is_dir() and os.access(mount, os.W_OK):
        return Path(mount) / "development-runs"
    return ROOT / "data/evidence/development-runs"


def verify_run(path):
    rec = read_json(path / "run.json")
    for name, digest in rec["files_sha256"].items():
        if Path(name).name != name or sha256_file(path / name) != digest:
            raise ValueError(f"saved run failed verification: {path / name}")
    return rec


def saved_runs(suite: str) -> list[tuple[str, str]]:
    runs = []
    # Local matrix outputs and explicitly published runs on the existing bucket mount.
    paths = set(location().glob("*/run.json")) | set((ROOT / "runs").glob("*/runs/*/run.json"))
    for p in sorted(paths, reverse=True):
        summary = p.parent / "matrix_setup.json"
        rec = read_json(summary if summary.exists() else p)
        if rec.get("source_suite") == suite:
            c = rec["forecast_config"]
            label = (f"{rec['created_at']} · {rec['configuration_id']}" if rec.get("configuration_id") else
                     f"{rec['created_at']} · {c['model']} · {c['prediction_target']} · {c['context_days']} d")
            if c.get("precision"):
                label += f" · {c['precision']['policy']}"
            runs.append((label, str(p.parent)))
    return runs


def _combine_cases(batches):
    """Join compatible inputs without losing issued-forecast provenance."""
    first = batches[0]
    values = {}
    for field in fields(first):
        name = field.name
        items = [getattr(batch, name) for batch in batches]
        if name == "cases":
            values[name] = pd.concat(items, ignore_index=True)
        elif name == "arome_inputs" and items[0] is not None:
            if any(item is None or item.fields != items[0].fields for item in items):
                raise ValueError("incompatible named AROME inputs")
            values[name] = type(items[0])(**{
                f.name: items[0].fields if f.name == "fields" else
                np.concatenate([getattr(item, f.name) for item in items]) for f in fields(items[0])})
        elif isinstance(items[0], np.ndarray):
            if any(item is None for item in items):
                raise ValueError(f"incompatible {name} presence")
            values[name] = np.concatenate(items)
        else:
            if any(item != items[0] for item in items):
                raise ValueError(f"incompatible {name}")
            values[name] = items[0]
    return type(first)(**values)


def predict_cases(adapter, batches, batch_size=1):
    """Isolate prediction failures while sharing one model load within a device allocation."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if batch_size > 1:
        results = []
        for start in range(0, len(batches), batch_size):
            group = batches[start:start + batch_size]
            try:
                combined = _combine_cases(group)
                result = adapter.predict(combined)
                result.validate(combined)
                offset = 0
                split = []
                for batch in group:
                    sl = slice(offset, offset + len(batch.cases))
                    info = {**result.info, "inference_batch_case_ids": combined.cases.case_id.tolist()}
                    if "retrieval" in info:
                        info["retrieval"] = info["retrieval"][sl]
                    if "residual_observed_steps" in info:
                        info["residual_observed_steps"] = info["residual_observed_steps"][sl]
                    single = replace(result, point=result.point[sl],
                        quantiles=None if result.quantiles is None else result.quantiles[sl],
                        members=None if result.members is None else result.members[sl], info=info)
                    single.validate(batch)
                    split.append((single, ""))
                    offset = sl.stop
                results.extend(split)
            except Exception as exc:
                # Retry the same model individually so one invalid case cannot discard its neighbours.
                for result, reason in predict_cases(adapter, group):
                    if result is not None:
                        result = replace(result, info={**result.info,
                            "batch_retry_reason": f"{type(exc).__name__}: {exc}"})
                    results.append((result, reason))
        return results
    results = []
    for batch in batches:
        try:
            result = adapter.predict(batch)
            result.validate(batch)
            results.append((result, ""))
        except Exception as exc:
            results.append((None, f"{type(exc).__name__}: {exc}"))
    return results


def run(source: Path, cfg: ForecastConfig, predict=predict_cases, progress=None, root: Path | None = None) -> Path:
    """Run every eligible source case; targets never enter adapters. Publish a new immutable directory."""
    original, manifest = read_json(source / "run.json"), read_json(source / "manifest.json")
    source_h = manifest["protocol"]["forecast"]["step_hours"] * manifest["protocol"]["forecast"]["horizon_steps"]
    issues = check(cfg, max_horizon_h=source_h, context_range=(7, 90))
    if cfg.step_hours != manifest["protocol"]["forecast"]["step_hours"]:
        raise ValueError("The benchmark step must match the selected suite")
    if issues:
        raise ValueError("; ".join(i.text() for i in issues))
    if not cfg.targets or set(cfg.targets) - set(manifest["protocol"].get("variables", cfg.targets)):
        raise ValueError("Selected targets are not available in this suite")
    cases = cases_from_manifest(manifest).reset_index(drop=True)
    eligible = cases[cases.eligible]
    if eligible.empty:
        raise ValueError("This suite has no eligible cases")
    source_hashes = {p.name: sha256_file(p) for p in (source / "run.json", source / "manifest.json",
                                                     source / "predictions.parquet", source / "targets.parquet")}
    data = copy.deepcopy(original.get("data") or {})

    def observations(name, fallback=None):
        meta = data.get(name) or {}
        p = ROOT / "data/processed" / meta.get("file", "")
        if meta.get("file") and p.is_file():
            if sha256_file(p) != meta.get("sha256"):
                raise ValueError(f"Installed {name} differs from the source run")
            source_hashes[str(p)] = sha256_file(p)
            return pd.read_parquet(p)
        if fallback is not None:
            source_hashes[fallback.name] = sha256_file(fallback)
            return pd.read_parquet(fallback)
        return None

    obs = observations("observations", source / "observation_windows.parquet")
    aux = observations("aux_observations") if "atmos" in cfg.feature_names else None
    arome_path = source / "arome_station_extract.parquet"
    if not arome_path.exists():
        arome_path = ROOT / f"data/processed/arome_{original['suite']}.parquet"
    arome = None
    if arome_path.exists():
        if sha256_file(arome_path) != (data.get("arome") or {}).get("sha256"):
            raise ValueError("Installed AROME differs from the source run")
        source_hashes[str(arome_path)] = sha256_file(arome_path)
        arome = pd.read_parquet(arome_path)
    coords = {s["id"]: (s["lat"], s["lon"]) for s in manifest["dataset"]["stations"]}
    ck = models.CHECKPOINTS[cfg.model]
    code_files = [Path(__file__), Path(__file__).with_name("forecast.py"),
                  Path(__file__).with_name("features.py"), Path(__file__).with_name("evaluate.py"),
                  Path(__file__).with_name("forecast_config.py"), Path(__file__).with_name("arome_inputs.py"),
                  Path(__file__).with_name("scoring.py")]
    if cfg.precision != "native":
        code_files.append(Path(__file__).with_name("precision.py"))
    if ck.family == "t0":
        code_files.append(Path(__file__).with_name("t0_adapter.py"))
    model_id = f"{ck.family}-dev {cfg.model} {cfg.prediction_target} {cfg.conditioning} " + sha256_text(
        json.dumps(cfg.as_dict(), sort_keys=True))[:8]
    if cfg.retrieval_method != "none":
        from weather_lab.rag_index import load_index
        from weather_lab.rag_store import Store

        index = load_index(Store().get("indexes", cfg.retrieval_index))
        if (index.meta["variables"] != list(cfg.targets) or index.meta["context_hours"] != cfg.context_days * 24
                or index.meta["observation_delay_hours"] != 3):
            raise ValueError("The retrieval index variables, context or observation delay differ from the benchmark")
        if index.meta["retrieval"].get("stride_hours") != 1:
            raise ValueError("Select an hourly sliding retrieval index for this benchmark")
        if index.meta["archive_id"] != original.get("archive_id"):
            raise ValueError("The retrieval index must belong to the benchmark's pinned archive")
        if index.meta.get("kind") == "arome_error" and cfg.conditioning != "archived":
            raise ValueError("The error index requires archived AROME conditioning")
        adapter = build_adapter(cfg, model_id, index=index, batch_size=8,
                                arome_fields=list(cfg.targets) if cfg.arome_mode else [])
        code_files.extend(Path(__file__).with_name(n) for n in ("rag_adapters.py", "rag_index.py", "rag_arome_error.py"))
    else:
        adapter = build_adapter(cfg, model_id, arome_fields=[v for v in ("t", "pmer") if v in cfg.targets])
    if original.get("hourly_benchmark"):
        code_files.append(Path(__file__).with_name("hourly_benchmark.py"))
    run_id = f"development-{now_utc().replace(':', '').replace('-', '')}-{uuid4().hex[:10]}"
    suite = f"development-{original['suite']}-{run_id.rsplit('-', 1)[-1]}"
    targets = pd.read_parquet(source / "targets.parquet")
    targets = targets[targets.variable.isin(cfg.targets) & targets.lead_h.le(cfg.horizon_h)].copy()
    assert not targets.duplicated(["case_id", "variable", "lead_h"]).any()
    expected = pd.MultiIndex.from_product([eligible.case_id, cfg.targets,
                                          range(cfg.step_hours, cfg.horizon_h + 1, cfg.step_hours)])
    assert expected.isin(pd.MultiIndex.from_frame(targets[["case_id", "variable", "lead_h"]])).all()
    saved = pd.read_parquet(source / "predictions.parquet")
    if original.get("hourly_benchmark") and (cfg.conditioning != "none" or cfg.prediction_target == "arome_error"):
        baseline_path = source / "arome_reference.parquet"
        source_hashes[baseline_path.name] = sha256_file(baseline_path)
        saved = pd.concat([saved, pd.read_parquet(baseline_path)], ignore_index=True)
    refs = [m for m in ("persistence", "seasonal-naive-24h", "climatology", "arome-raw", "arome-biascorr")
            if m in set(saved.model)]
    reference = saved[saved.model.isin(refs) & saved.variable.isin(cfg.targets)
                      & saved.lead_h.le(cfg.horizon_h) & saved.case_id.isin(eligible.case_id)].copy()
    reference["suite"], reference["run_id"] = suite, run_id
    rows, availability, pending, fingerprints, case_info = [], [], [], {}, {}
    need_history = any("arome_history" in features.GROUPS[n].needs for n in cfg.feature_names)
    started = time.perf_counter()

    def finish(batch, result, reason, status):
        case = batch.cases.iloc[0]
        availability.append({"case_id": case.case_id, "station": case.station, "origin": case.origin,
                             "model": model_id, "status": status, "reason": reason})
        if result is not None:
            case_info[case.case_id] = result.info
            baseline = None
            if cfg.prediction_target == "arome_error":
                _, baseline, _ = adapter.error_inputs(batch)
            rows.append(result.to_frame(batch, suite, arome_baseline=baseline))

    def flush():
        if not pending:
            return
        models.prefetch(ck, allow_download=False)
        results = predict(adapter, list(pending))
        if len(results) != len(pending):
            raise ValueError("The prediction worker returned an incomplete batch")
        for batch, (result, reason) in zip(pending, results):
            finish(batch, result, reason, "available" if result is not None else "failed")
        pending.clear()

    for number, (_, case) in enumerate(cases.set_index("case_id", drop=False).iterrows(), 1):
        if not case.eligible:
            availability.append({"case_id": case.name, "station": case.station, "origin": case.origin,
                                 "model": model_id, "status": "ineligible", "reason": case.get("reason", "source suite eligibility")})
            continue
        try:
            if original.get("hourly_benchmark"):
                from weather_lab.hourly_benchmark import inputs

                station_obs = obs[obs.station.eq(case.station)]
                relevant = None if arome is None else arome[arome.station.eq(case.station)
                    & (arome.available_at <= case.origin)
                    & (arome.valid_time >= case.origin - pd.Timedelta(days=cfg.context_days, hours=3))]
                batch = inputs(station_obs, relevant, case, cfg)
                batch.station_coords = coords
            else:
                batch = exploratory_inputs(obs[obs.station.eq(case.station)], case, list(cfg.targets), cfg.context_days,
                                           cfg.arome_mode, arome, horizon_steps=cfg.horizon_h // cfg.step_hours,
                                           step_hours=cfg.step_hours, aux_df=aux, station_coords=coords,
                                           arome_history=need_history, arome_baseline=cfg.prediction_target == "arome_error")
            native = adapter.build_inputs(batch)
            ctx, known = native[:2]
            if ck.family == "t0":
                past, report = None, native[2]
            else:
                past, _, report = adapter.feature_rows(batch)
            empty = [n for n, share in report.get("observed_fraction", {}).items() if share == 0]
            if empty:
                raise ValueError("Selected feature rows have no observed history: " + ", ".join(empty))
            fingerprints[case.name] = make_key({"config": cfg.as_dict(), "case_id": case.name,
                                               "source_sha256": source_hashes},
                                              {"context": ctx, "known": known, "past": past,
                                               "baseline": batch.arome if cfg.prediction_target == "arome_error" else None})
            pending.append(batch)
        except (ValueError, KeyError) as exc:
            availability.append({"case_id": case.name, "station": case.station, "origin": case.origin,
                                 "model": model_id, "status": "unavailable", "reason": str(exc)})
        if len(pending) == 8:
            flush()
        if progress:
            progress(number / len(cases), desc=f"Benchmark: {number}/{len(cases)} cases")
    flush()
    # Explicit NaN rows keep unavailable cases in the scorer's availability accounting.
    skeleton = targets[targets.case_id.isin(eligible.case_id)].merge(
        cases[["case_id", "station", "origin"]], on="case_id", validate="many_to_one").drop(columns="observed")
    skeleton["suite"], skeleton["model"], skeleton["model_revision"] = suite, model_id, f"{ck.repo}@{ck.revision[:12]}"
    skeleton["output_type"], skeleton["point"] = "quantiles", np.nan
    for q in QUANTILE_LEVELS:
        skeleton[qcol(q)] = np.nan
    primary = pd.concat(rows, ignore_index=True) if rows else skeleton.iloc[:0]
    missing = ~pd.MultiIndex.from_frame(skeleton[["case_id", "variable", "lead_h"]]).isin(
        pd.MultiIndex.from_frame(primary[["case_id", "variable", "lead_h"]]))
    predictions = pd.concat([reference, primary, skeleton[missing]], ignore_index=True)
    predictions["run_id"] = run_id
    for frame in (predictions, targets):
        for col in ("origin", "valid_time"):
            if col in frame:
                frame[col] = pd.to_datetime(frame[col], utc=True).astype("datetime64[us, UTC]")
    if cfg.precision != "native":
        float_columns = ["point", "arome_baseline", "predicted_error", *[qcol(q) for q in QUANTILE_LEVELS]]
        for name in float_columns:
            if name in predictions:
                predictions[name] = predictions[name].astype(np.float32)
    tables = score(predictions, targets, cases, {}, [6, 24, 48])
    tables["paired_differences"] = tables["paired_differences"].reindex(columns=[
        "split", "variable", "model", "reference", "loss", "lead_h", "n_pairs", "n_blocks", "mean_diff",
        "ci95_low", "ci95_high"])
    destination = (root if root is not None else location()) / run_id
    temp_root = ROOT / ".tmp/codex/development-benchmark"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
        out = Path(tmp)
        predictions.to_parquet(out / "predictions.parquet", index=False)
        targets.to_parquet(out / "targets.parquet", index=False)
        obs.to_parquet(out / "observation_windows.parquet", index=False)
        if arome is not None:
            shutil.copy2(arome_path, out / "arome_station_extract.parquet")
        for name, table in tables.items():
            table.to_csv(out / f"{name}.csv", index=False)
        pd.DataFrame(availability).to_csv(out / "case_availability.csv", index=False)
        derived = copy.deepcopy(manifest)
        derived["suite"] = suite
        derived["created_at"] = now_utc()
        derived["label"] = "Interactive development benchmark derived from " + original["suite"]
        derived["leads_h"] = list(range(cfg.step_hours, cfg.horizon_h + 1, cfg.step_hours))
        derived["source_protocol_sha256"] = derived.get("protocol_sha256")
        derived["source_observation_availability"] = derived.pop("observation_availability", None)
        derived["protocol"]["forecast"].update(context_days=cfg.context_days,
                                                horizon_steps=cfg.horizon_h // cfg.step_hours)
        derived["protocol"].update(variables=list(cfg.targets), development_configuration=cfg.as_dict())
        derived["source_case_data_sha256"] = derived.get("case_data_sha256")
        derived["case_data_sha256"] = sha256_text(json.dumps(fingerprints, sort_keys=True))
        derived["source_manifest_sha256"] = source_hashes["manifest.json"]
        derived["protocol_sha256"] = sha256_text(json.dumps(derived["protocol"], sort_keys=True))
        write_json(out / "manifest.json", derived)
        write_json(out / "forecast_config.json", cfg.as_dict())
        systems = [s | {"reused_from": original["run_id"]} for s in original["systems"] if s["id"] in refs]
        if "arome-raw" in refs and not any(s["id"] == "arome-raw" for s in systems):
            systems.append({"id": "arome-raw", "kind": "arome", "per_case_ms": None, "peak_rss_mb_after": None})
        systems.append(adapter.describe() | {"revision": f"{ck.repo}@{ck.revision[:12]}",
                                             "per_case_ms": None, "peak_rss_mb_after": None})
        write_json(out / "run.json", {"run_id": run_id, "suite": suite, "development": True,
                   "label": "Interactive development benchmark; no local fine-tuning or held-out claim",
                   "source_suite": original["suite"], "source_run_id": original["run_id"],
                   "source_sha256": source_hashes, "forecast_config": cfg.as_dict(),
                   "checkpoint": {"repo": ck.repo, "revision": ck.revision, "sha256": ck.sha256},
                   "hourly_benchmark": bool(original.get("hourly_benchmark")), "archive_id": original.get("archive_id"),
                   "adapter_version": adapter.ADAPTER_VERSION, "preprocessing_version": adapter.PREPROCESSING_VERSION,
                   "input_sha256": fingerprints, "case_info": case_info,
                   "code_sha256": {p.name: sha256_file(p) for p in code_files},
                   "files_sha256": {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()},
                   "created_at": now_utc(), "code_revision": git_revision(), "systems": systems,
                   "data": data, "n_cases": len(cases), "n_eligible": len(eligible),
                   "environment": environment(), "elapsed_s": round(time.perf_counter() - started, 3),
                   "comparison": "matched observed case/lead pairs, separately per split and variable"})
        for name, digest in source_hashes.items():
            p = Path(name) if Path(name).is_absolute() else source / name
            if sha256_file(p) != digest:
                raise ValueError("A source artifact changed during the development benchmark")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # run.json appears last, so only complete runs enter replay choices, including on mounted storage.
        destination.mkdir()
        try:
            for p in out.iterdir():
                if p.name != "run.json":
                    shutil.copy2(p, destination / p.name)
            shutil.copy2(out / "run.json", destination / ".run.json")
            (destination / ".run.json").replace(destination / "run.json")
        except Exception:
            shutil.rmtree(destination)
            raise
    return destination
