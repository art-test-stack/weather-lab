"""RAG research entry points, independent of the frozen benchmark suites."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from weather_lab.common import ROOT
from weather_lab.rag_store import Store

CONFIG = ROOT / "configs/rag_weather_v1.yaml"


def config(path: Path = CONFIG) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg["experiment"] not in ("rag-weather-2026-v1", "rag-arome-error-2026-v1"):
        raise ValueError("unsupported RAG experiment version")
    return cfg


def command(args):
    store = Store(Path(args.local_only) if args.local_only else None)
    cfg = config(Path(args.config))
    if cfg["experiment"] == "rag-arome-error-2026-v1" and args.rag_command not in ("index", "restore", "status"):
        raise ValueError("This protocol builds exploratory AROME-error indexes from the existing archive; no scoring or collection")
    if args.rag_command == "collect":
        from weather_lab.rag_archive import collect
        path = collect(cfg, store)
    elif args.rag_command == "ingest":
        from weather_lab.rag_archive import ingest
        path = ingest(Path(args.observations), cfg, Path(args.issued) if args.issued else None,
                      Path(args.historical) if args.historical else None, store)
    elif args.rag_command == "index":
        if cfg["experiment"] == "rag-arome-error-2026-v1":
            from weather_lab.rag_arome_error import build
        else:
            from weather_lab.rag_index import build
        archive = args.archive or cfg.get("archive_id")
        if not archive:
            raise ValueError("Select a source archive")
        path = build(archive, cfg, store)
    elif args.rag_command == "evaluate":
        from weather_lab.rag_evaluate import run
        path = run(args.index, cfg, args.horizon, store)
    elif args.rag_command == "restore":
        path = store.get(args.kind, args.id)
    elif args.rag_command == "status":
        print(json.dumps({"root": str(store.root), "mounted": store.mounted,
                          **{k: store.list(k) for k in ("archives", "indexes", "runs")}}, indent=2))
        return
    elif args.rag_command == "prefetch":
        from weather_lab.models import CHECKPOINTS, prefetch
        names = cfg["foundation_models"]
        print(f"Explicit model setup: up to {sum(CHECKPOINTS[n].mb for n in names)} MB; no inference.")
        for name in names:
            print(prefetch(CHECKPOINTS[name]))
        return
    else:
        raise ValueError(args.rag_command)
    print(f"saved {path}\nartifact id: {path.name}")


def parser(sub):
    root = sub.add_parser("rag", help="2026 weather RAG: collect, persist, index and evaluate")
    children = root.add_subparsers(dest="rag_command", required=True)
    for name in ("collect", "ingest", "index", "evaluate", "restore", "status", "prefetch"):
        p = children.add_parser(name)
        p.add_argument("--config", default=str(CONFIG))
        p.add_argument("--local-only", metavar="DIRECTORY", help="explicit offline/local artifact store; skips bucket writes")
        p.set_defaults(func=command)
        if name == "ingest":
            p.add_argument("--observations", required=True)
            p.add_argument("--issued")
            p.add_argument("--historical")
        elif name == "index":
            p.add_argument("--archive", help="source archive id; defaults to the protocol's pinned archive when present")
        elif name == "evaluate":
            p.add_argument("--index", required=True)
            p.add_argument("--horizon", type=int, choices=(24, 48), required=True)
        elif name == "restore":
            p.add_argument("--kind", choices=("archives", "indexes", "runs"), required=True)
            p.add_argument("--id", required=True)
