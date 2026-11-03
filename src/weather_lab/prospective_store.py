"""Append-only prospective evidence on the existing bucket.

Bucket writes require the workflow's shared concurrency group: the bucket has no
compare-and-swap operation. A local lock also serializes writers on one host.
Readers see only checksum-verified records whose commit marker exists.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import re
import tempfile

from weather_lab.common import ROOT
from weather_lab.publish import DEFAULT_BUCKET

PREFIX = "prospective/v1"
KINDS = {"forecasts", "observations", "predictions", "availability", "catalog", "evaluations", "status"}


def encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class Archive:
    def __init__(self, root: Path | None = None, *, api=None):
        self.root = Path(root) if root is not None else ROOT / "data" / PREFIX
        self.api = api

    @contextmanager
    def writer(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".writer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield self
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _marker(self, kind, key):
        if kind not in KINDS or not isinstance(key, str) or not key:
            raise ValueError("invalid prospective record kind or key")
        return self.root / "records" / kind / (digest(key.encode()) + ".json")

    def get(self, kind, key):
        path = self._marker(kind, key)
        if not path.exists():
            return None
        record = self._read(path)
        if record["kind"] != kind or record["key"] != key:
            raise ValueError("prospective record identity mismatch")
        return record["value"]

    def _read(self, path):
        marker = json.loads(path.read_bytes())
        ident = marker["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", ident):
            raise ValueError("invalid prospective object digest")
        payload = (self.root / "objects" / (ident + ".json")).read_bytes()
        if digest(payload) != ident:
            raise ValueError("prospective object checksum mismatch")
        record = json.loads(payload)
        if self._marker(record["kind"], record["key"]) != path:
            raise ValueError("prospective commit marker identity mismatch")
        return record

    def records(self, kind):
        self._marker(kind, "validate")
        return [self._read(p) for p in sorted((self.root / "records" / kind).glob("*.json"))]

    def put(self, kind, key, value):
        """Call under writer(); identical retries succeed, conflicting values fail."""
        marker = self._marker(kind, key)
        payload = encoded({"version": 1, "kind": kind, "key": key, "value": value})
        ident = digest(payload)
        if marker.exists():
            if encoded(self.get(kind, key)) != encoded(value):
                raise ValueError(f"immutable prospective record conflict: {kind}/{key}")
            return ident
        obj = self.root / "objects" / (ident + ".json")
        obj.parent.mkdir(parents=True, exist_ok=True)
        marker.parent.mkdir(parents=True, exist_ok=True)
        obj.write_bytes(payload)
        commit = encoded({"sha256": ident})
        if self.api is not None:
            self.api.batch_bucket_files(DEFAULT_BUCKET, add=[(str(obj), f"{PREFIX}/objects/{obj.name}")])
            # Read back payload bytes before exposing the completion marker.
            temp_root = ROOT / ".tmp/codex/prospective-store"
            temp_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
                check = Path(tmp) / "object.json"
                self.api.download_bucket_files(DEFAULT_BUCKET, [(f"{PREFIX}/objects/{obj.name}", str(check))],
                                               raise_on_missing_files=True)
                if digest(check.read_bytes()) != ident:
                    raise ValueError("prospective bucket upload checksum mismatch")
                pending = Path(tmp) / "marker.json"
                pending.write_bytes(commit)
                self.api.batch_bucket_files(DEFAULT_BUCKET, add=[(str(pending),
                    f"{PREFIX}/records/{kind}/{marker.name}")])
        pending = marker.with_suffix(".pending")
        pending.write_bytes(commit)
        pending.replace(marker)
        return ident

    def sync(self):
        """Explicit read-only download; never invoked by the Gradio view."""
        if self.api is None:
            return
        entries = list(self.api.list_bucket_tree(DEFAULT_BUCKET, prefix=PREFIX, recursive=True))
        objects, markers = [], []
        for entry in entries:
            relative = entry.path.removeprefix(PREFIX + "/")
            if not re.fullmatch(r"(?:objects/[0-9a-f]{64}|records/(?:" +
                                "|".join(sorted(KINDS)) + r")/[0-9a-f]{64})\.json", relative):
                continue
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if relative.startswith("objects/"):
                if not destination.exists() or digest(destination.read_bytes()) != destination.stem:
                    objects.append((entry.path, str(destination)))
            elif not destination.exists():
                markers.append((entry.path, destination))
        if objects:
            self.api.download_bucket_files(DEFAULT_BUCKET, objects, raise_on_missing_files=True)
        for _, path in objects:
            if digest(Path(path).read_bytes()) != Path(path).stem:
                raise ValueError("prospective object checksum mismatch")
        # Download markers separately; local readers must never see a partial download.
        temp_root = ROOT / ".tmp/codex/prospective-store"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            staged = [(remote, str(Path(tmp) / str(i))) for i, (remote, _) in enumerate(markers)]
            if staged:
                self.api.download_bucket_files(DEFAULT_BUCKET, staged, raise_on_missing_files=True)
            for (_, source), (_, destination) in zip(staged, markers):
                marker = json.loads(Path(source).read_bytes())
                ident = marker["sha256"]
                if not re.fullmatch(r"[0-9a-f]{64}", ident):
                    raise ValueError("invalid prospective object digest")
                payload = (self.root / "objects" / (ident + ".json")).read_bytes()
                record = json.loads(payload)
                if digest(payload) != ident or self._marker(record["kind"], record["key"]) != destination:
                    raise ValueError("prospective remote commit identity/checksum mismatch")
                pending = destination.with_suffix(".pending")
                pending.write_bytes(Path(source).read_bytes())
                pending.replace(destination)
        for kind in KINDS:
            self.records(kind)
