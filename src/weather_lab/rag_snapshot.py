"""Stage immutable RAG updates from fully verified prospective cases, without activation."""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from weather_lab import prospective
from weather_lab.common import ROOT, git_revision, read_json, sha256_file, write_json
from weather_lab.forecast import hourly_inputs
from weather_lab.prospective_store import digest, encoded, now_utc
from weather_lab.rag_index import METHODS, embedding, load_index
from weather_lab.rag_store import Store

ROOT_DIR = ROOT / "data/evidence/prospective-rag-snapshots"


def candidates(archive, kind, cutoff):
    """One candidate per station/decision (and issue for AROME-error memory)."""
    grouped = {}
    for entry in archive.records("predictions"):
        rec = entry["value"]
        grouped.setdefault(rec["case_id"], []).append(rec)
    selected, excluded, seen = [], [], set()
    for case_id, records in sorted(grouped.items()):
        rec = records[0]
        if not rec["cohort"].startswith("prospective-"):
            excluded.append({"case_id": case_id, "reason": "retrospective case; original live availability is not established"})
            continue
        origin = pd.Timestamp(rec["origin"])
        valid = []
        for item in records:
            proof = item.get("provenance", {})
            if kind == "arome_error" and item["model"] != "arome-raw":
                continue
            if item["status"] not in ("available", "partial") or not proof.get("finished_at"):
                continue
            finished = pd.Timestamp(proof["finished_at"])
            if (pd.Timestamp(proof.get("decision_cutoff")) != origin or finished < origin
                    or finished >= origin + pd.Timedelta(hours=1) or finished > cutoff):
                continue
            if any(row.get("point") is not None for row in item["rows"]):
                valid.append(item)
        if not valid:
            excluded.append({"case_id": case_id, "reason": "no forecast frozen before the first verifying hour"})
            continue
        rec = min(valid, key=lambda item: (pd.Timestamp(item["provenance"]["finished_at"]), item["model"]))
        key = (rec["station"], origin, pd.Timestamp(rec["issue"]) if kind == "arome_error" else None)
        if key in seen:
            continue
        seen.add(key)
        selected.append(rec)
    return selected, excluded


def build(archive, parent_path: Path, cutoff, *, destination=None, clock=now_utc):
    """Append candidates to a pinned parent; keep every parent byte and configuration intact.

    Version 2 is staged outside the active index store. Frozen v1 adapters refuse
    it until a separately reviewed loader/configuration update is implemented.
    """
    cutoff = pd.to_datetime(cutoff, utc=True)
    if pd.isna(cutoff) or cutoff > pd.Timestamp(clock()):
        raise ValueError("RAG snapshot cutoff must be a real UTC time at or before now")
    destination = destination or Store(root=ROOT_DIR)
    if not destination.local_only:
        raise ValueError("RAG snapshots must be staged in an explicit local store")
    parent = load_index(Path(parent_path))
    kind = parent.meta.get("kind", "weather")
    if kind not in ("weather", "arome_error"):
        raise ValueError("unsupported parent RAG index kind")
    t, h = parent.meta["context_hours"], parent.meta["horizon_hours"]
    delay, variables = parent.meta["observation_delay_hours"], parent.meta["variables"]
    if parent.payload.shape[1:] != (len(variables), t + h) or h <= delay:
        raise ValueError("parent payload dimensions do not match its recorded configuration")
    records, excluded = candidates(archive, kind, cutoff)
    report = {"parent_index": parent.ident, "selection_cutoff": cutoff.isoformat(), "snapshot": None,
              "added_cases": 0, "excluded": excluded, "activation": "staged_only"}
    if not records:
        return report
    verified = prospective.observation_frame(archive, cutoff)
    payload, windows, vectors, evidence = [], [], {method: [] for method in METHODS}, []
    indexed = set()
    for row in parent.windows.itertuples():
        decision = pd.Timestamp(row.origin) + (pd.Timedelta(hours=delay) if kind == "weather" else pd.Timedelta(0))
        indexed.add((row.station, decision, pd.Timestamp(row.run) if kind == "arome_error" else None))
    for rec in records:
        origin, issue = pd.Timestamp(rec["origin"]), pd.Timestamp(rec["issue"])
        key = (rec["station"], origin, issue if kind == "arome_error" else None)
        if key in indexed:
            report["excluded"].append({"case_id": rec["case_id"], "reason": "candidate already present in parent index"})
            continue
        try:
            past = prospective.observation_frame(archive, origin)
            past = past[past.station.eq(rec["station"])]
            case = pd.Series({"case_id": rec["case_id"], "station": rec["station"], "origin": origin}, name=rec["case_id"])
            issued = None
            if kind == "arome_error":
                from weather_lab.rag_arome_error import make_batch, vector

                issued = prospective.forecast_frame(archive, origin)
                batch, vintage = make_batch(past, issued, case, variables, t, h - delay, delay, parent.meta["product"])
                if pd.Timestamp(vintage["run"]) != issue:
                    raise ValueError("another issued AROME run was selected for this case")
            else:
                batch = hourly_inputs(past, pd.DataFrame([case]), variables, t, h - delay, delay)
            future = pd.to_datetime(batch.future_times[0], utc=True)
            history = pd.to_datetime(batch.history_times()[0], utc=True)
            truth = verified[verified.station.eq(rec["station"])].set_index("time").reindex(future)
            values = truth[variables].to_numpy(dtype=np.float32, na_value=np.nan).T
            if not np.isfinite(values).all() or truth.available_at.isna().any():
                raise ValueError("forecast continuation is not fully verified at the snapshot cutoff")
            if kind == "weather":
                block = np.concatenate([batch.history[0], values], axis=1)
                if not np.isfinite(block).all():
                    raise ValueError("weather memory requires complete measured context and continuation")
                encoded_query = {method: embedding(batch.history[0, :, -parent.meta["query_hours"]:], parent.scale, method)
                                 for method in METHODS}
                row = {"station": rec["station"], "origin": pd.Timestamp(batch.origins[0], tz="UTC")}
            else:
                past_errors = batch.history[0] - batch.arome_inputs.history[0]
                future_errors = values - batch.arome[0]
                block = np.concatenate([past_errors, future_errors], axis=1)
                encoded_query = {method: vector(batch, parent.scale, parent.meta["query_hours"],
                    parent.meta["retrieval"]["forecast_query_hours"], delay, method) for method in METHODS}
                row = {"station": rec["station"], "origin": origin, "run": issue,
                       "forecast_available_at": pd.Timestamp(vintage["available_at"])}
                for j, variable in enumerate(variables):
                    errors = future_errors[j, delay:]
                    row[f"mae_{variable}"], row[f"bias_{variable}"] = float(abs(errors).mean()), float(errors.mean())
            context_rows = past[past.time.isin(history)]
            availability = max(pd.Timestamp(rec["provenance"]["finished_at"]), context_rows.available_at.max(), truth.available_at.max())
            sources = set(context_rows.observation_version) | set(truth.observation_version)
            if issued is not None:
                relevant = issued[issued.station.eq(rec["station"]) & issued.valid_time.isin(history.append(future))]
                sources.update(relevant.forecast_version)
                availability = max(availability, relevant.available_at.max())
            if pd.isna(availability) or availability > cutoff:
                raise ValueError("candidate contains inputs unavailable at the snapshot cutoff")
            proof = {"case_id": rec["case_id"], "prediction_record": digest(encoded(rec)),
                     "source_records": sorted(sources), "available_at": availability.isoformat()}
            for method in METHODS:
                if encoded_query[method].shape != parent.vectors[method].shape[1:]:
                    raise ValueError("candidate embedding differs from the parent retrieval settings")
            row.update(case_id=rec["case_id"], decision_origin=origin, available_at=availability,
                       payload_end=future[-1], evidence_sha256=digest(encoded(proof)))
            windows.append(row)
            payload.append(block)
            evidence.append(proof)
            for method in METHODS:
                vectors[method].append(encoded_query[method])
            indexed.add(key)
        except (ValueError, KeyError, TypeError) as exc:
            report["excluded"].append({"case_id": rec["case_id"], "reason": str(exc)})
    if not windows:
        return report
    temp_root = ROOT / ".tmp/codex/rag-snapshot"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
        out = Path(tmp)
        pd.concat([parent.windows, pd.DataFrame(windows)], ignore_index=True).to_parquet(out / "windows.parquet", index=False)
        np.save(out / "payload.npy", np.concatenate([np.asarray(parent.payload), np.stack(payload)]), allow_pickle=False)
        for method in METHODS:
            np.save(out / f"vectors-{method}.npy", np.concatenate([np.asarray(parent.vectors[method]), np.stack(vectors[method])]), allow_pickle=False)
        meta = {**parent.meta, "version": 2, "kind": "prospective_" + kind, "parent_index": parent.ident,
                "selection_cutoff": cutoff.isoformat(), "windows": len(parent.windows) + len(windows),
                "new_windows": len(windows), "activation": "staged_only", "scale_fit": "unchanged pinned parent scaling",
                "new_candidate_availability": "actual input/verification collection times; prospective freeze before first target",
                "code_revision": git_revision(), "builder_sha256": sha256_file(Path(__file__))}
        for field in ("archive_id", "configuration", "experiment", "start", "end_exclusive"):
            if field in meta:
                meta["parent_" + field] = meta.pop(field)
        write_json(out / "index.json", meta)
        write_json(out / "prospective_evidence.json", {"candidates": evidence, "exclusions": report["excluded"]})
        result = destination.put("indexes", out)
    # Receipt is outside the content-addressed payload and appears only after verification.
    receipt = destination.root / "availability" / (result.name + ".json")
    if not receipt.exists():
        pending = receipt.with_suffix(".pending")
        write_json(pending, {"index_id": result.name, "available_at": clock(), "activation": "staged_only"})
        pending.replace(receipt)
    return {**report, "snapshot": str(result), "snapshot_id": result.name, "added_cases": len(windows),
            "total_cases": meta["windows"], "available_at": read_json(receipt)["available_at"]}
