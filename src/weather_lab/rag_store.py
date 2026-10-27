"""Immutable RAG artifacts on the Space bucket, with verified local caches elsewhere."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from weather_lab.common import ROOT, read_json, sha256_file, sha256_text, write_json
from weather_lab.publish import DEFAULT_BUCKET

KINDS = {"archives", "indexes", "runs"}
PREFIX = "rag/weather-v1"


class Store:
    def __init__(self, root: Path | None = None, api=None):
        self.api = api
        self.local_only = root is not None
        self.mounted = False
        if root is None:
            from weather_lab.bootstrap import load_bootstrap_config

            mounts = load_bootstrap_config()["bootstrap"]["snapshot"].get("mounts", {})
            mount = Path(mounts.get(DEFAULT_BUCKET, "/nonexistent-weather-lab-mount"))
            self.mounted = mount.is_dir() and os.access(mount, os.W_OK)
            root = mount / PREFIX if self.mounted else ROOT / "data/cache" / PREFIX
        self.root = Path(root)

    def _api(self):
        if self.api is None:
            from huggingface_hub import HfApi
            self.api = HfApi()
        return self.api

    def _path(self, kind: str, ident: str) -> Path:
        if kind not in KINDS or not re.fullmatch(r"[0-9a-f]{64}", ident):
            raise ValueError("invalid RAG artifact kind or SHA-256 id")
        return self.root / kind / ident

    @staticmethod
    def verify(path: Path) -> dict:
        rec = read_json(path / "artifact.json")
        if sha256_text(json.dumps(rec["files_sha256"], sort_keys=True)) != rec["id"]:
            raise ValueError("RAG artifact inventory identity mismatch")
        for name, digest in rec["files_sha256"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("unsafe RAG artifact path")
            if sha256_file(path / name) != digest:
                raise ValueError(f"RAG artifact checksum mismatch: {name}")
        return rec

    def get(self, kind: str, ident: str) -> Path:
        path = self._path(kind, ident)
        if not (path / "artifact.json").is_file() and not (self.mounted or self.local_only):
            api = self._api()
            prefix = f"{PREFIX}/{kind}/{ident}"
            with tempfile.TemporaryDirectory() as tmp:
                stage = Path(tmp)
                api.download_bucket_files(DEFAULT_BUCKET, [(f"{prefix}/artifact.json", str(stage / "artifact.json"))],
                                          raise_on_missing_files=True)
                rec = read_json(stage / "artifact.json")
                if rec["id"] != ident or rec["kind"] != kind:
                    raise ValueError("remote RAG artifact identity differs from the requested object")
                for name in rec["files_sha256"]:
                    if Path(name).is_absolute() or ".." in Path(name).parts:
                        raise ValueError("unsafe RAG artifact path")
                    (stage / name).parent.mkdir(parents=True, exist_ok=True)
                api.download_bucket_files(DEFAULT_BUCKET,
                                          [(f"{prefix}/{n}", str(stage / n)) for n in rec["files_sha256"]],
                                          raise_on_missing_files=True)
                self.verify(stage)
                path.mkdir(parents=True, exist_ok=True)
                for name in rec["files_sha256"]:
                    (path / name).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(stage / name, path / name)
                shutil.copy2(stage / "artifact.json", path / "artifact.json")
        rec = self.verify(path)
        if rec["id"] != ident or rec["kind"] != kind:
            raise ValueError("RAG artifact identity differs from requested object")
        return path

    def put(self, kind: str, source: Path) -> Path:
        files = {p.relative_to(source).as_posix(): sha256_file(p) for p in sorted(source.rglob("*"))
                 if p.is_file() and p.name not in ("artifact.json", "artifact.pending")}
        ident = sha256_text(json.dumps(files, sort_keys=True))
        destination = self._path(kind, ident)
        # A commit marker appears last; incomplete copies can be safely retried.
        if (destination / "artifact.json").exists():
            self.verify(destination)
        else:
            destination.mkdir(parents=True, exist_ok=True)
            for name in files:
                (destination / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, destination / name)
            write_json(destination / "artifact.pending", {"version": 1, "kind": kind, "id": ident,
                                                          "files_sha256": files})
            if any(sha256_file(destination / n) != digest for n, digest in files.items()):
                raise RuntimeError("local RAG artifact copy failed verification")
        if not (self.mounted or self.local_only):
            api, prefix = self._api(), f"{PREFIX}/{kind}/{ident}"
            # Verify uploaded bytes before publishing the remote commit marker.
            api.batch_bucket_files(DEFAULT_BUCKET, add=[(str(destination / n), f"{prefix}/{n}") for n in files])
            with tempfile.TemporaryDirectory() as tmp:
                for name in files:
                    (Path(tmp) / name).parent.mkdir(parents=True, exist_ok=True)
                api.download_bucket_files(DEFAULT_BUCKET, [(f"{prefix}/{n}", str(Path(tmp) / n)) for n in files],
                                          raise_on_missing_files=True)
                if any(sha256_file(Path(tmp) / n) != digest for n, digest in files.items()):
                    raise RuntimeError(f"bucket verification failed; local artifact retained at {destination}")
            marker = destination / "artifact.pending" if (destination / "artifact.pending").exists() else destination / "artifact.json"
            api.batch_bucket_files(DEFAULT_BUCKET, add=[(str(marker),
                                                         f"{prefix}/artifact.json")])
        if (destination / "artifact.pending").exists():
            (destination / "artifact.pending").replace(destination / "artifact.json")
        self.verify(destination)
        return destination

    def list(self, kind: str) -> list[tuple[str, str]]:
        if kind not in KINDS:
            raise ValueError(kind)
        found = []
        for p in sorted((self.root / kind).glob("*/artifact.json"), reverse=True):
            try:
                rec = self.verify(p.parent)
                found.append((rec["id"][:12], rec["id"]))
            except (ValueError, OSError, KeyError):
                continue
        return found
