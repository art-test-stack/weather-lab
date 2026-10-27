"""One hourly, delayed RAG forecast, cached outside benchmark runs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from weather_lab import predcache
from weather_lab.forecast import CaseBatch, QUANTILE_LEVELS
from weather_lab.forecast_config import check
from weather_lab.models import prefetch, resolve
from weather_lab.rag_adapters import foundation
from weather_lab.rag_index import load_index
from weather_lab.rag_store import Store


def forecast(case, cfg, device="cpu", lookup_only=False, store=None):
    issues = check(cfg)
    if issues or cfg.retrieval_method == "none":
        raise ValueError("; ".join(i.text() for i in issues) or "Select a RAG retrieval method")
    store = store or Store()
    index = load_index(store.get("indexes", cfg.retrieval_index))
    kind = index.meta.get("kind", "weather")
    if kind == "arome_error" and cfg.conditioning != "archived":
        raise ValueError("The AROME-error index requires archived AROME conditioning")
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
    provenance, product = {}, None
    correction = cfg.prediction_target == "arome_error"
    if cfg.conditioning != "none" or correction:
        from weather_lab.rag_arome_error import make_batch
        from weather_lab.rag_archive import issued_forecasts

        if not (archive / "issued.parquet").is_file():
            raise ValueError("Selected archive has no issued AROME runs; stitched history cannot substitute")
        product = index.meta.get("product") or "meteofrance_arome_france"
        issued = issued_forecasts(pd.read_parquet(archive / "issued.parquet"))
        obs = pd.read_parquet(archive / "observations.parquet")
        batch, provenance = make_batch(obs, issued, case, list(cfg.targets), 720, cfg.horizon_h, 3, product)
        if cfg.conditioning == "archived" and not np.isfinite(batch.arome_inputs.history).any(axis=-1).all():
            raise ValueError("Archived AROME conditioning has no qualifying history for a target variable")
    model = foundation(cfg.model, index, cfg.retrieval_method, k=int(index.meta["retrieval"]["k"]),
                       device=device, calendar=cfg.calendar, arome=cfg.arome_mode,
                       prediction_target=cfg.prediction_target)
    condition = f" +AROME {cfg.conditioning} run{provenance['run']}" if provenance else ""
    error_label = " +error-cases" if kind == "arome_error" else ""
    target_label = " +AROME error correction" if correction else ""
    model.id = (f"exploratory {cfg.model} ctx30d h{cfg.horizon_h} {'+cal' if cfg.calendar else ''} "
                f"+RAG {cfg.retrieval_method} {index.ident}{error_label}{condition}{target_label} [{device}]")
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
    if provenance:
        payload.update(arome_product=product, arome_provenance=provenance,
                       arome_history_runs=batch.arome_inputs.history_run.astype(str).tolist(),
                       arome_history_available_at=batch.arome_inputs.history_available_at.astype(str).tolist())
    arrays = {"context": native[0], "known": native[1]}
    error_history = {}
    if correction:
        residuals, baseline, _ = model.error_inputs(batch)
        arrays["arome_baseline"] = baseline
        payload.update(prediction_target="arome_error", residual_version=1, forecast_lead_input=True)
        for j, name in enumerate(cfg.targets):
            seen = batch.history_times()[0, np.isfinite(residuals[0, j])]
            error_history[name] = {"observed_steps": len(seen), "total_steps": 720,
                                   "first": str(seen[0]), "last": str(seen[-1])}
    key = predcache.make_key(payload, arrays)

    def compute():
        prefetch(ck, allow_download=False)
        result = model.predict(batch)
        result.validate(batch)
        frame = result.to_frame(batch, "exploratory RAG (outside frozen benchmark)")
        if correction:
            positive = batch.lead_hours[0] > 0
            frame["arome_baseline"] = baseline[0][:, positive].ravel()
            frame["predicted_error"] = frame.point - frame.arome_baseline
        return frame

    if lookup_only:
        frame = predcache.lookup(key)
        hit = frame is not None
    else:
        frame, hit = predcache.get_or_compute(key, payload, compute)
    settings = {"checkpoint": cfg.model, "revision": ck.revision, "context_days": 30,
                "context_steps": 720, "horizon_steps": horizon, "step_hours": 1,
                "calendar": cfg.calendar, "features": [], "arome_mode": cfg.arome_mode,
                "retrieval_method": cfg.retrieval_method, "index_id": index.ident,
                "archive_id": index.meta["archive_id"], "observation_delay_hours": 3,
                "retrieval": model.audit, "retrieval_kind": kind,
                "arome_product": product, "arome_run": provenance.get("run"), "arome_provenance": provenance}
    if correction:
        settings.update(prediction_target="arome_error", residual_version=1,
                        forecast_lead_input=True, error_history=error_history)
    return frame, {"cache_hit": hit, "key": key[:16], "model": model.id, "settings": settings}
