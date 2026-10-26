"""Model registry, pinned checkpoint files and the in-process cache of loaded models.

Three layers, cached separately:

1. **Checkpoint files** live in the Hugging Face Hub cache (`HF_HOME` / `HF_HUB_CACHE`, honoured by
   `huggingface_hub`). `prefetch` downloads only the inference files of a pinned full revision and
   verifies their SHA-256; every conditioning variant of a checkpoint shares those files.
   With `WEATHER_LAB_OFFLINE=1` (or `HF_HUB_OFFLINE=1`) nothing contacts the Hub: models load from the
   local snapshot directory or fail with the command that fixes it.
2. **Loaded models** are kept per (repo, revision, device, dtype) in a small LRU (`ModelCache`), bounded
   by a count and an approximate RAM budget. Request data never lives in the cached object: adapters
   pass inputs to each forecast call, and a per-model lock serialises inference.
3. **Predictions** of benchmark runs are saved files (`runs/`, `data/examples/`); exploratory
   predictions go to `weather_lab.predcache`.

`FAMILIES` records what each model family can consume, so a missing system can be reported as
"unsupported input" (the family cannot take it) or "not run" (it could, but no result is saved).
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

INFERENCE_FILES = ("config.json", "model.safetensors")


class CheckpointError(RuntimeError):
    """A checkpoint cannot be used as pinned; the message says what to run."""


@dataclass(frozen=True)
class Checkpoint:
    name: str                 # display name, e.g. "Timestral-209M"
    family: str               # key into FAMILIES
    repo: str
    revision: str             # full 40-character commit sha, never a branch or tag
    mb: int
    sha256: str | None = None  # of model.safetensors; verified after download
    files: tuple[str, ...] = INFERENCE_FILES

    @property
    def size(self) -> str:
        return self.name.split("-")[-1]

    @property
    def ident(self) -> str:
        return f"{self.repo}@{self.revision}"


@dataclass(frozen=True)
class Family:
    """Capabilities of a model family. Adapters must not be asked for inputs they cannot take."""

    name: str
    adapter: str | None                 # adapter kind in configs, None while not integrated
    past_covariates: bool
    known_covariates: bool              # future-known covariates (calendar, issued AROME)
    missing_values: str                 # how NaN inputs are handled
    native_quantiles: tuple[float, ...] | None
    max_context: int | None             # steps
    max_horizon: int | None             # steps
    runtime: str
    status: str                         # "integrated" | "planned"
    notes: str = ""
    checkpoints: tuple[str, ...] = field(default_factory=tuple)


TIMESTRAL_QUANTILES = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                       0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)

FAMILIES: dict[str, Family] = {
    "timestral": Family(
        "Timestral", adapter="timestral", past_covariates=True, known_covariates=True,
        missing_values="NaN masked; a covariate row needs observed history to be attended (key validity)",
        native_quantiles=TIMESTRAL_QUANTILES, max_context=None, max_horizon=None,
        runtime="PyTorch (timestral @ a534bdb)", status="integrated",
        checkpoints=("Timestral-5M", "Timestral-28M", "Timestral-209M")),
    "t0": Family(
        "TFC t0", adapter="t0", past_covariates=True, known_covariates=True,
        missing_values="NaN means missing; past rows are jointly forecast native variates",
        native_quantiles=TIMESTRAL_QUANTILES, max_context=None, max_horizon=1024,
        runtime="tfc-t0==0.6.0 (optional isolated PyTorch environment)", status="integrated",
        notes="Research adapter; runtime and weights are not included in bootstrap or Space preload",
        checkpoints=("t0-gamma",)),
}

CHECKPOINTS: dict[str, Checkpoint] = {c.name: c for c in (  # pinned Hub revisions inspected on 2026-10-24
    Checkpoint("Timestral-5M", "timestral", "mistralai/Timestral-5M", "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f", 21,
               "7ace108c4987e7f2a99a75a445240019d1d31c2117139d1bd05ffdfd25aca738"),
    Checkpoint("Timestral-28M", "timestral", "mistralai/Timestral-28M", "49b26409ac87b5935539a5d8ffe7f397b720db47",
               111, "dfc83c6220937ce5fc6687b35a4df94cd25e28a2cfe6fb92a23dad6301b2015d"),
    Checkpoint("Timestral-209M", "timestral", "mistralai/Timestral-209M",
               "9874d252ab0aae16f7f7c999f98f98d6ad7d8365", 837,
               "1456effb9927e3daa8c7234f10497e3424233869245a9c72b8ecebaf7ad2b7fa"),
    Checkpoint("t0-gamma", "t0", "theforecastingcompany/t0-gamma",
               "e08aee083778f1f32da4974c0a34276776a35c52", 1023,
               "1603e255629be22ea45518b502f16c55d06821c7090a1a851673e779e9c245a6"),
)}


def resolve(name_or_repo: str, revision: str | None = None) -> Checkpoint:
    """A configured name ("Timestral-209M" / "209M") or repo id (+ revision) -> pinned checkpoint."""
    for c in CHECKPOINTS.values():
        if name_or_repo in (c.name, c.size, c.repo):
            if revision not in (None, c.revision):
                if len(revision) != 40:
                    raise CheckpointError(f"{c.repo}: revision {revision!r} is not a full commit sha")
                return Checkpoint(c.name, c.family, c.repo, revision, c.mb, None, c.files)
            return c
    if revision is None or len(revision) != 40:
        raise CheckpointError(f"unknown checkpoint {name_or_repo!r}: give a registered name or a repo with a full "
                              "40-character revision (no moving references in benchmark runs)")
    return Checkpoint(name_or_repo, "unknown", name_or_repo, revision, 0)


def offline() -> bool:
    on = ("1", "true", "yes", "on")
    return (os.environ.get("WEATHER_LAB_OFFLINE", "").lower() in on
            or os.environ.get("HF_HUB_OFFLINE", "").lower() in on)


def hub_cache() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def local_dir(ck: Checkpoint) -> Path | None:
    """The snapshot directory holding every inference file of this revision, or None (no network)."""
    from huggingface_hub import try_to_load_from_cache

    paths = [try_to_load_from_cache(ck.repo, f, revision=ck.revision) for f in ck.files]
    if not all(isinstance(p, str) for p in paths):
        return None
    return Path(paths[0]).parent


def _verify(ck: Checkpoint, folder: Path) -> None:
    from weather_lab.common import sha256_file

    for f in ck.files:
        p = folder / f
        if not p.exists() or p.resolve().stat().st_size == 0:
            raise CheckpointError(f"{ck.ident}: {f} missing or empty in {folder}; delete that snapshot folder and "
                                  f"run `weather-lab models prefetch {ck.name}`")
    if ck.sha256:
        digest = sha256_file((folder / "model.safetensors").resolve())
        if digest != ck.sha256:
            raise CheckpointError(f"{ck.ident}: model.safetensors sha256 {digest[:12]} != pinned {ck.sha256[:12]} "
                                  f"(incomplete or corrupted file). Delete {folder.parent.parent} and run "
                                  f"`weather-lab models prefetch {ck.name}`")


_VERIFIED: set[str] = set()
STATS = {"downloads": 0, "download_bytes": 0, "loads": 0, "load_hits": 0, "evictions": 0}
_STATS_LOCK = threading.Lock()


def prefetch(ck: Checkpoint, allow_download: bool = True, verify: bool = True) -> dict:
    """Make the pinned inference files local. Concurrent callers are safe (Hub cache file locks)."""
    folder = local_dir(ck)
    downloaded = False
    if folder is None:
        if offline():
            raise CheckpointError(f"{ck.name} ({ck.ident}) is not in the local cache {hub_cache()} and offline mode "
                                  f"is on; run `weather-lab models prefetch {ck.name}` with network access first")
        if not allow_download:
            raise PermissionError(f"{ck.name} weights (~{ck.mb} MB) are not cached; allow the download to fetch them")
        from huggingface_hub import snapshot_download

        try:
            folder = Path(snapshot_download(ck.repo, revision=ck.revision, allow_patterns=list(ck.files)))
        except Exception as e:  # noqa: BLE001 - surface hub/network errors with the remedy
            raise CheckpointError(f"{ck.ident}: download failed ({type(e).__name__}: {e}); retry "
                                  f"`weather-lab models prefetch {ck.name}`") from e
        downloaded = True
        with _STATS_LOCK:
            STATS["downloads"] += 1
            STATS["download_bytes"] += sum((folder / f).resolve().stat().st_size for f in ck.files)
    if verify and ck.ident not in _VERIFIED:
        _verify(ck, folder)
        _VERIFIED.add(ck.ident)
    return {"checkpoint": ck.name, "repo": ck.repo, "revision": ck.revision, "path": str(folder),
            "downloaded": downloaded, "bytes": sum((folder / f).resolve().stat().st_size for f in ck.files)}


@dataclass
class Loaded:
    forecaster: object
    info: dict
    lock: threading.Lock
    nbytes: int


class ModelCache:
    """LRU of loaded models keyed by (repo, revision, device, dtype); lazy loading, bounded retention."""

    def __init__(self, max_models: int | None = None, max_mb: int | None = None) -> None:
        self.max_models = max_models or int(os.environ.get("WEATHER_LAB_MAX_LOADED_MODELS", 2))
        self.max_mb = max_mb or int(os.environ.get("WEATHER_LAB_MODEL_RAM_MB", 4096))
        self._items: OrderedDict[tuple, Loaded] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, ck: Checkpoint, device: str = "cpu", dtype: str = "float32") -> Loaded:
        key = (ck.repo, ck.revision, device, dtype)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                with _STATS_LOCK:
                    STATS["load_hits"] += 1
                return self._items[key]
            item = self._load(ck, device, dtype)
            self._items[key] = item
            self._evict(keep=key)
            return item

    def _load(self, ck: Checkpoint, device: str, dtype: str) -> Loaded:
        import torch

        if dtype != "float32":
            raise ValueError("only float32 inference is supported here")
        if ck.family not in ("timestral", "t0"):
            raise CheckpointError(f"No model loader for family {ck.family!r}")
        if ck.family == "t0":
            from importlib.metadata import PackageNotFoundError, version

            try:
                runtime = version("tfc-t0")
            except PackageNotFoundError as e:
                raise CheckpointError("t0 requires the isolated tfc-t0==0.6.0 environment") from e
            if runtime != "0.6.0":
                raise CheckpointError(f"t0 runtime {runtime} differs from pinned tfc-t0==0.6.0")
        meta = prefetch(ck)                                 # local files (or an actionable error)
        t0 = time.perf_counter()
        # Load from the local snapshot directory: no Hub request, whatever the online state.
        if ck.family == "timestral":
            from timestral import TimestralForecaster

            fc = TimestralForecaster.from_pretrained(meta["path"], device=device)
            module = fc.model
            known, levels = bool(fc.supports_known_covariates), list(fc.quantile_levels)
        else:
            from t0 import T0Forecaster

            fc = T0Forecaster.from_pretrained(meta["path"], local_files_only=True).to(device)
            module = fc
            if fc.config.xpos_scale_layout != "halves" or fc.config.scaler_zero_no_value is not False:
                raise CheckpointError("t0-gamma config conventions were not loaded correctly")
            known, levels = True, list(fc.config.quantile_levels)
        module.eval()
        n = int(sum(p.numel() for p in module.parameters()))
        with _STATS_LOCK:
            STATS["loads"] += 1
        info = {"first_load_s": round(time.perf_counter() - t0, 3), "weights_were_cached": not meta["downloaded"],
                "weights_path": meta["path"], "device": device, "dtype": dtype,
                "torch_threads": torch.get_num_threads(), "n_parameters": n,
                "supports_known_covariates": known, "quantile_levels": levels,
                **({"tfc_t0": runtime} if ck.family == "t0" else {})}
        return Loaded(fc, info, threading.Lock(), 4 * n)

    def _evict(self, keep: tuple) -> None:
        def over() -> bool:
            total = sum(i.nbytes for i in self._items.values()) / 1e6
            return len(self._items) > self.max_models or (total > self.max_mb and len(self._items) > 1)

        while over():
            key = next(k for k in self._items if k != keep)
            del self._items[key]
            with _STATS_LOCK:
                STATS["evictions"] += 1

    def release(self, device: str | None = None) -> int:
        """Drop loaded models (on one device). ZeroGPU: GPU state never outlives one GPU call."""
        with self._lock:
            keys = [k for k in self._items if device is None or k[2] == device]
            for k in keys:
                del self._items[k]
            return len(keys)

    def status(self) -> list[dict]:
        with self._lock:
            return [{"repo": k[0], "revision": k[1], "device": k[2], "dtype": k[3], "approx_mb": round(v.nbytes / 1e6)}
                    for k, v in self._items.items()]


MODELS = ModelCache()


def checkpoint_status() -> list[dict]:
    """Files on disk per registered checkpoint (no network)."""
    rows = []
    for c in CHECKPOINTS.values():
        folder = local_dir(c)
        size = sum((folder / f).resolve().stat().st_size for f in c.files) if folder else 0
        rows.append({"checkpoint": c.name, "repo": c.repo, "revision": c.revision[:12], "on_disk": folder is not None,
                     "mb": round(size / 1e6), "path": str(folder) if folder else ""})
    return rows
