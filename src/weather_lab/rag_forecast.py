"""One hourly, delayed RAG forecast, cached outside benchmark runs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from weather_lab import predcache
from weather_lab.forecast import CaseBatch, QUANTILE_LEVELS
from weather_lab.forecast_config import check
from weather_lab.models import prefetch, resolve
from weather_lab.rag_adapters import foundation
from weather_lab.rag_index import Index
from weather_lab.rag_store import Store


def forecast(case, cfg, device="cpu", lookup_only=False, store=None):
    issues = check(cfg)
    if issues or cfg.retrieval_method == "none":
        raise ValueError("; ".join(i.text() for i in issues) or "Select a RAG retrieval method")
    store = store or Store()
    index = Index(store.get("indexes", cfg.retrieval_index))
    if (index.meta["variables"] != list(cfg.targets) or index.meta["context_hours"] != 720
            or index.meta["observation_delay_hours"] != 3):
        raise ValueError("The selected index does not match the 30-day hourly temperature/pressure protocol")
    archive = store.get("archives", index.meta["archive_id"])
    source = pd.read_parquet(archive / "observations.parquet")
    source = source[source.station.eq(case["station"])].set_index("time").sort_index()
    origin = pd.Timestamp(case["origin"])
    end = origin - pd.Timedelta(hours=3)
    times = pd.date_range(end=end, periods=720, freq="h")
    history = source.reindex(times)[list(cfg.targets)].to_numpy(np.float32).T
    if (np.isfinite(history).mean(axis=-1) < 0.8).any() or not np.isfinite(history[:, -1]).all():
        raise ValueError("Hourly archive unavailable: insufficient causal observation context for this station/time")
    horizon = cfg.horizon_h + 3
    future = pd.date_range(end + pd.Timedelta(hours=1), periods=horizon, freq="h")
    cases = pd.DataFrame([{"case_id": case.name, "station": case["station"], "origin": origin}])
    batch = CaseBatch(cases, list(cfg.targets), history[None],
                     np.array([end.tz_localize(None)], dtype="datetime64[ns]"),
                     future.tz_localize(None).to_numpy(dtype="datetime64[ns]")[None], 1)
    model = foundation(cfg.model, index, cfg.retrieval_method, k=int(index.meta["retrieval"]["k"]),
                       device=device, calendar=cfg.calendar)
    model.id = (f"exploratory {cfg.model} ctx30d h{cfg.horizon_h} {'+cal' if cfg.calendar else ''} "
                f"+RAG {cfg.retrieval_method} {index.ident} [{device}]")
    # The artifact id distinguishes different archives even when their query windows match.
    native = model.build_inputs(batch)
    ck = resolve(cfg.model)
    import torch

    payload = {"forecast_config": cfg.as_dict(), "archive_id": index.meta["archive_id"],
               "index_id": index.ident, "adapter": model.describe(), "adapter_version": model.ADAPTER_VERSION,
               "preprocessing": model.PREPROCESSING_VERSION, "rag_version": 1,
               "station": case["station"], "origin": str(origin), "case_id": case.name,
               "revision": ck.revision, "device": device, "dtype": "float32", "torch": torch.__version__,
               "quantiles": QUANTILE_LEVELS, "retrieval": model.audit}
    if ck.family == "t0":
        payload["runtime"] = "tfc-t0==0.6.0"
    key = predcache.make_key(payload, {"context": native[0], "known": native[1]})

    def compute():
        prefetch(ck, allow_download=False)
        result = model.predict(batch)
        result.validate(batch)
        return result.to_frame(batch, "exploratory RAG (outside frozen benchmark)")

    if lookup_only:
        frame = predcache.lookup(key)
        hit = frame is not None
    else:
        frame, hit = predcache.get_or_compute(key, payload, compute)
    settings = {"checkpoint": cfg.model, "revision": ck.revision, "context_days": 30,
                "context_steps": 720, "horizon_steps": horizon, "step_hours": 1,
                "calendar": cfg.calendar, "features": [], "arome_mode": None,
                "retrieval_method": cfg.retrieval_method, "index_id": index.ident,
                "archive_id": index.meta["archive_id"], "observation_delay_hours": 3,
                "retrieval": model.audit}
    return frame, {"cache_hit": hit, "key": key[:16], "model": model.id, "settings": settings}
