"""Content-addressed cache of exploratory forecasts (never benchmark scores).

The key hashes everything that can change the output: checkpoint repo + full revision, adapter
and preprocessing versions, the exact input arrays (values and NaN masks), origin, horizon,
history length, covariate construction, quantile levels, device, dtype and torch version.
A different revision or conditioning variant therefore never reuses an entry.

Location (first match): `WEATHER_LAB_PREDICTION_CACHE` (a path, or `off`), else the persistent
volume configured under `caches.predictions` in configs/bootstrap.yaml when it is mounted and
writable (on the Space: the Storage Bucket), else `data/cache/predictions` (ephemeral on a Space).
Writes are atomic (temporary file + rename); identical concurrent requests compute once
(in-process lock + a file lock on local disk).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from weather_lab.common import ROOT

CACHE_VERSION = 1
STATS = {"hits": 0, "misses": 0}
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def location() -> tuple[Path | None, str]:
    """(directory or None if disabled, description of its persistence)."""
    env = os.environ.get("WEATHER_LAB_PREDICTION_CACHE")
    if env:
        return (None, "disabled (WEATHER_LAB_PREDICTION_CACHE=off)") if env.lower() == "off" else \
            (Path(env), "WEATHER_LAB_PREDICTION_CACHE (persistence depends on that path)")
    try:
        from weather_lab.bootstrap import load_bootstrap_config

        conf = (load_bootstrap_config()["bootstrap"].get("caches") or {}).get("predictions") or {}
        mounts = (load_bootstrap_config()["bootstrap"].get("snapshot") or {}).get("mounts") or {}
    except Exception:  # noqa: BLE001 - no config: local default
        conf, mounts = {}, {}
    mount = mounts.get(conf.get("bucket", ""))
    if mount and Path(mount).is_dir() and os.access(mount, os.W_OK):
        return Path(mount) / conf.get("subdir", "cache/predictions"), f"Storage Bucket {conf['bucket']} at {mount}"
    return ROOT / "data" / "cache" / "predictions", "local disk (lost when a Space container restarts)"


def _digest(a: np.ndarray | None) -> str:
    if a is None:
        return "none"
    a = np.ascontiguousarray(a, dtype=np.float32)
    a = np.where(np.isnan(a), np.float32(np.nan), a)          # one NaN bit pattern: mask, not payload
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def make_key(payload: dict, arrays: dict[str, np.ndarray | None]) -> str:
    full = {"cache_version": CACHE_VERSION, **payload, "arrays": {k: _digest(v) for k, v in sorted(arrays.items())}}
    return hashlib.sha256(json.dumps(full, sort_keys=True, default=str).encode()).hexdigest()


def get_or_compute(key: str, payload: dict, compute: Callable[[], pd.DataFrame],
                   root: Path | None = None) -> tuple[pd.DataFrame, bool]:
    """(frame, hit). `compute` runs at most once per key across concurrent identical requests."""
    from filelock import FileLock

    root = root if root is not None else location()[0]
    if root is None:
        STATS["misses"] += 1
        return compute(), False
    path = root / key[:2] / f"{key}.parquet"
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, threading.Lock())
    lock_dir = Path(tempfile.gettempdir()) / "weather-lab-locks"   # local disk: locks on FUSE mounts are unreliable
    lock_dir.mkdir(exist_ok=True)
    with lock, FileLock(str(lock_dir / f"{key}.lock"), timeout=900):
        if path.exists():
            STATS["hits"] += 1
            return pd.read_parquet(path), True
        STATS["misses"] += 1
        frame = compute()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        os.close(fd)
        try:
            frame.to_parquet(tmp, index=False)
            Path(tmp).with_suffix(".json").write_text(json.dumps(payload, sort_keys=True, default=str))
            os.replace(Path(tmp).with_suffix(".json"), path.with_suffix(".json"))
            os.replace(tmp, path)                       # the parquet appears last: present means complete
        finally:
            for t in (Path(tmp), Path(tmp).with_suffix(".json")):
                t.unlink(missing_ok=True)
        return frame, False


def lookup(key: str, root: Path | None = None) -> pd.DataFrame | None:
    root = root if root is not None else location()[0]
    path = None if root is None else root / key[:2] / f"{key}.parquet"
    if path is None or not path.exists():
        return None
    STATS["hits"] += 1
    return pd.read_parquet(path)


def status(root: Path | None = None) -> dict:
    root, where = (root, "given") if root is not None else location()
    files = list(root.glob("*/*.parquet")) if root is not None and root.exists() else []
    return {"path": str(root) if root else None, "persistence": where, "entries": len(files),
            "mb": round(sum(f.stat().st_size for f in files) / 1e6, 2), **STATS}
