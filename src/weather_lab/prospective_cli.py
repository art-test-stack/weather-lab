"""Explicit, serialized acquisition / inference commands for GitHub Actions."""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from weather_lab.common import ROOT
from weather_lab.prospective_store import Archive, now_utc
from weather_lab.publish import DEFAULT_BUCKET


def install_catalog(api, suite, allow_large):
    """Read existing published discovery records; never synthesize a matrix."""
    from weather_lab.development import location
    from weather_lab.saved_catalog import saved_configurations
    from weather_lab.rag_store import Store

    if api is not None:
        files = []
        for entry in api.list_bucket_tree(DEFAULT_BUCKET, prefix="development-runs", recursive=True):
            if re.fullmatch(r"development-runs/[A-Za-z0-9_-]+/(run|matrix_setup)\.json", entry.path):
                target = location() / Path(entry.path).relative_to("development-runs")
                target.parent.mkdir(parents=True, exist_ok=True)
                files.append((entry.path, str(target)))
        if files:
            api.download_bucket_files(DEFAULT_BUCKET, files, raise_on_missing_files=True)
    catalog = saved_configurations(suite)
    store = Store()
    for ident in {rec["config"].get("retrieval_index") for rec in catalog.values()} - {None, ""}:
        path = store.root / "indexes" / ident
        if not (path / "artifact.json").is_file() and not allow_large:
            raise ValueError("saved retrieval indexes need explicit setup; rerun catalog with --allow-large-downloads")
        store.get("indexes", ident)


def prepare_models(archive, allow_large):
    from weather_lab import models
    from weather_lab.prospective_collect import event
    from weather_lab.prospective import catalog_records

    names = sorted({r["value"]["config"]["model"] for r in catalog_records(archive)})
    for record in catalog_records(archive):
        rec = record["value"]
        ck = models.CHECKPOINTS[rec["config"]["model"]]
        if {k: getattr(ck, k) for k in ("repo", "revision", "sha256")} != rec["checkpoint"]:
            raise ValueError("model registry differs from frozen Benchmark catalog; no checkpoint download permitted")
    missing = [models.CHECKPOINTS[name] for name in names if models.local_dir(models.CHECKPOINTS[name]) is None]
    mb = sum(ck.mb for ck in missing)
    if mb > 500 and not allow_large:
        event(archive, "models", "unavailable", reason=f"{mb} MB checkpoints exceed 500 MiB initialization budget; explicit --allow-large-downloads required")
        print(f"Checkpoint setup needs approximately {mb} MB; inference will record missing weights as unavailable.")
        return
    for name in names:
        models.prefetch(models.CHECKPOINTS[name], allow_download=True)
    event(archive, "models", "ok", checkpoints=names, downloaded_mb=mb)


def command(args):
    from weather_lab import prospective as engine
    from weather_lab.prospective_collect import (collect_forecasts, collect_observations, collect_realtime,
                                                backfill_saved, record_backfill_gaps)
    from weather_lab.rag_store import Store

    api = None
    if args.remote:
        from huggingface_hub import HfApi

        if args.action != "sync" and not args.serialized_writer:
            raise ValueError("remote writes require --serialized-writer and the shared prospective-bucket-writer concurrency group")
        api = HfApi()
    archive = Archive(Path(args.archive_root), api=api)
    now = pd.Timestamp(now_utc())
    start = pd.to_datetime(args.start, utc=True) if args.start else now.floor("D") - pd.Timedelta(days=2)
    end = pd.to_datetime(args.end, utc=True) if args.end else now
    if start > end or end > now:
        raise ValueError("require start <= end <= current UTC time")
    with archive.writer():
        archive.sync()
        if args.action == "sync":
            return
        if args.action == "catalog":
            if not engine.catalog_records(archive):
                install_catalog(api, args.suite, args.allow_large_downloads)
                engine.freeze_catalog(archive, args.suite)
        elif args.action == "models":
            prepare_models(archive, args.allow_large_downloads)
        elif args.action == "forecasts":
            collect_forecasts(archive, start, end)
        elif args.action == "observations":
            # Retry seven preceding days; explicit dates can seed genuine historical context.
            lo = start.floor("D") if args.start else now.floor("D") - pd.Timedelta(days=7)
            hi = end.floor("h") if args.end else now.floor("D") - pd.Timedelta(hours=1)
            collect_observations(archive, lo, hi)
        elif args.action == "realtime":
            collect_realtime(archive)
        elif args.action == "infer":
            engine.infer(archive, start, end, replay=args.retrospective)
        elif args.action == "score":
            engine.score(archive)
        elif args.action == "backfill":
            if not args.start or not args.end:
                raise ValueError("backfill requires explicit --start and --end")
            if not engine.catalog_records(archive):
                raise ValueError("run prospective catalog/setup before backfill to inventory every saved configuration")
            store = Store()
            if not (store.root / "suites" / args.source_artifact / "artifact.json").exists() and not args.allow_large_downloads:
                raise ValueError("historical source artifact size may exceed 500 MiB; explicit --allow-large-downloads required")
            source = store.get("suites", args.source_artifact)
            backfill_saved(archive, start, end, source)
            collect_forecasts(archive, start, end)
            verification_end = min(now.floor("h"), end.floor("h") + pd.Timedelta(hours=51))
            collect_observations(archive, start.floor("D") - pd.Timedelta(days=90), verification_end)
            record_backfill_gaps(archive, start, end)
    print(f"Prospective {args.action}: saved archive {archive.root}")


def parser(sub):
    root = sub.add_parser("prospective", help="explicit saved Benchmark collection, frozen inference and delayed scoring")
    root.add_argument("action", choices=["sync", "catalog", "models", "forecasts", "observations", "realtime", "infer", "score", "backfill"])
    root.add_argument("--archive-root", default=str(ROOT / "data/prospective/v1"))
    root.add_argument("--remote", action="store_true", help="sync/persist through the configured Hugging Face bucket")
    root.add_argument("--serialized-writer", action="store_true", help="caller holds the shared workflow concurrency slot")
    root.add_argument("--start", help="inclusive UTC origin/date")
    root.add_argument("--end", help="inclusive UTC origin/date; use 23:00Z for a complete final day")
    root.add_argument("--retrospective", action="store_true", help="explicit historical replay, never prospective")
    root.add_argument("--suite", default="hourly-2026-v3")
    root.add_argument("--source-artifact", default="0aac87538da2016cf7d4a8265088de1da1f6b0feb8c63dbeb74a49b45322399b")
    root.add_argument("--allow-large-downloads", action="store_true", help="explicit checkpoint/index setup exceeding 500 MiB")
    root.set_defaults(func=command)
