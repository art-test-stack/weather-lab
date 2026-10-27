"""Exact multivariate window retrieval; embeddings and payloads use pre-2026 data only."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from weather_lab.common import ROOT, read_json, ts, write_json
from weather_lab.rag_store import Store

VERSION = 1
METHODS = ("level", "shape")


def load_index(path):
    if read_json(Path(path) / "index.json").get("kind") == "arome_error":
        from weather_lab.rag_arome_error import ErrorIndex

        return ErrorIndex(path)
    return Index(path)


def embedding(values: np.ndarray, scale: np.ndarray, method: str) -> np.ndarray:
    if method not in METHODS:
        raise ValueError(f"unknown retrieval method {method}")
    a = np.asarray(values, np.float32)
    if not np.isfinite(a).all():
        raise ValueError("retrieval query needs a complete window; missing observations are not imputed")
    if method == "shape":
        a = a - a[..., -1:]
    return (a / scale[:, None]).reshape(-1)


def build(archive_id: str, cfg: dict, store: Store | None = None) -> Path:
    store = store or Store()
    archive = store.get("archives", archive_id)
    obs = pd.read_parquet(archive / "observations.parquet")
    variables = cfg["variables"]
    cutoff, start = ts(cfg["retrieval"]["end_exclusive"]), ts(cfg["retrieval"]["start"])
    if cutoff > ts("2026-01-01"):
        raise ValueError("the retrieval archive must be frozen before 2026")
    delay = pd.Timedelta(hours=cfg["observation_delay_hours"])
    train = obs[(obs.time >= start) & (obs.time + delay < cutoff)].copy()
    scale = train[variables].std().to_numpy(np.float32)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("retrieval variables need nonzero pre-2026 variance")
    t, q = int(cfg["context_hours"]), int(cfg["retrieval"]["query_hours"])
    h = max(cfg["horizons_hours"]) + int(cfg["observation_delay_hours"])
    stride = int(cfg["retrieval"]["stride_hours"])
    if not 0 < q <= t or stride <= 0 or stride > 24 or 24 % stride or t % 24:
        raise ValueError("positive query/stride and whole-day context required")
    payload, rows, vectors = [], [], {m: [] for m in METHODS}
    for station, frame in train.groupby("station", sort=True):
        frame = frame.set_index("time").sort_index()
        axis = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
        frame = frame.reindex(axis)
        values = frame[variables].to_numpy(np.float32).T
        anchor = (cfg["origin_hour_utc"] - cfg["observation_delay_hours"]) % 24
        for end in range(t - 1, len(axis) - h):
            if (axis[end].hour - anchor) % stride:
                continue
            block = values[:, end - t + 1:end + h + 1]
            if not np.isfinite(block).all():
                continue
            available = axis[end + h] + delay
            if available >= cutoff:
                continue
            payload.append(block)
            rows.append({"station": station, "origin": axis[end], "available_at": available,
                         "payload_end": axis[end + h]})
            for m in METHODS:
                vectors[m].append(embedding(block[:, t - q:t], scale, m))
    if not payload:
        raise ValueError("no complete pre-2026 context+horizon windows; inspect archive coverage")
    temp_root = ROOT / ".tmp/codex/rag-index"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
        out = Path(tmp)
        np.save(out / "payload.npy", np.stack(payload), allow_pickle=False)
        for m in METHODS:
            np.save(out / f"vectors-{m}.npy", np.stack(vectors[m]), allow_pickle=False)
        pd.DataFrame(rows).to_parquet(out / "windows.parquet", index=False)
        write_json(out / "index.json", {"version": VERSION, "archive_id": archive_id,
                   "variables": variables, "context_hours": t, "query_hours": q, "horizon_hours": h,
                   "start": start.isoformat(), "end_exclusive": cutoff.isoformat(), "scale": scale.tolist(),
                   "scale_fit": "pre-cutoff observations only; one standard deviation per physical variable",
                   "retrieval": cfg["retrieval"], "observation_delay_hours": cfg["observation_delay_hours"],
                   "windows": len(rows), "embedding": "flattened scaled hourly window; exact squared L2",
                   "payload": "historical measured continuations, not historical issued AROME forecasts"})
        return store.put("indexes", out)


class Index:
    def __init__(self, path: Path):
        self.path = path
        self.ident = Store.verify(path)["id"]
        self.meta = read_json(path / "index.json")
        if self.meta.get("kind") == "arome_error":
            raise ValueError("AROME-error indexes use the new exploratory protocol; select them in Forecast")
        if self.meta["version"] != VERSION:
            raise ValueError("unsupported retrieval index version")
        self.windows = pd.read_parquet(path / "windows.parquet")
        self.payload = np.load(path / "payload.npy", mmap_mode="r", allow_pickle=False)
        self.vectors = {m: np.load(path / f"vectors-{m}.npy", mmap_mode="r", allow_pickle=False) for m in METHODS}
        self.scale = np.asarray(self.meta["scale"], np.float32)
        cutoff = ts(self.meta["end_exclusive"])
        if cutoff > ts("2026-01-01") or (self.windows.available_at >= cutoff).any():
            raise ValueError("retrieval index contains information available in 2026")

    def __getstate__(self):
        # ZeroGPU workers reopen verified memory maps; do not serialize the entire database per chunk.
        return {"path": self.path}

    def __setstate__(self, state):
        self.__init__(state["path"])

    def retrieve(self, batch, method: str = "shape", k: int = 5) -> tuple[np.ndarray, list[dict]]:
        """Weighted analogue past+future rows, suitable for native known covariates."""
        t, q, h = self.meta["context_hours"], self.meta["query_hours"], batch.horizon
        if batch.step_hours != 1 or batch.history.shape[-1] < t or h > self.meta["horizon_hours"] or k < 1:
            raise ValueError("retrieval context, hourly cadence, horizon or k differs from the index")
        idx = [batch.variables.index(v) for v in self.meta["variables"]]
        trajectories, audit = [], []
        for i, c in enumerate(batch.cases.itertuples()):
            origin = pd.Timestamp(batch.origins[i], tz="UTC")
            query = batch.history[i, idx, -q:]
            vector = embedding(query, self.scale, method)
            # Same station and UTC hour; every stored continuation must predate the query.
            mask = (self.windows.station.eq(c.station) & self.windows.origin.dt.hour.eq(origin.hour)
                    & self.windows.available_at.lt(origin))
            seasonal = self.meta["retrieval"].get("seasonal_days")
            if seasonal is not None:
                distance = np.abs(self.windows.origin.dt.dayofyear - origin.dayofyear)
                mask &= np.minimum(distance, 366 - distance) <= seasonal
            candidates = np.flatnonzero(mask.to_numpy())
            if len(candidates) < k:
                raise ValueError(f"only {len(candidates)} causal analogues for station {c.station}; need {k}")
            distances = np.mean((self.vectors[method][candidates] - vector) ** 2, axis=1)
            order = np.argsort(distances, kind="stable")[:k]
            chosen, distances = candidates[order], distances[order]
            weight = 1 / np.maximum(np.sqrt(distances), 1e-6)
            weight /= weight.sum()
            blocks = np.asarray(self.payload[chosen, :, :t + h]).copy()
            if method == "shape":
                blocks += query[:, -1][None, :, None] - blocks[:, :, t - 1:t]
            trajectories.append(np.sum(blocks * weight[:, None, None], axis=0))
            audit.append({"case_id": c.case_id, "index_id": self.ident, "method": method,
                          "neighbors": [{"row": int(n), "origin": str(self.windows.iloc[n].origin),
                                         "available_at": str(self.windows.iloc[n].available_at),
                                         "distance": float(d), "weight": float(w)}
                                        for n, d, w in zip(chosen, distances, weight)]})
        return np.stack(trajectories).astype(np.float32), audit
