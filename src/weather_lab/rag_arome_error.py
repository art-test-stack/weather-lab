"""Causal retrieval of issued AROME cases and their verified errors."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from weather_lab.arome_inputs import build_arome_inputs
from weather_lab.common import read_json, sha256_file, ts, write_json
from weather_lab.forecast import CaseBatch
from weather_lab.rag_archive import issued_forecasts, observations
from weather_lab.rag_evaluate import choose_arome
from weather_lab.rag_index import Index, METHODS
from weather_lab.rag_store import Store

KIND = "arome_error"
VERSION = 1


def make_batch(obs, issued, case, variables, context_hours, horizon_hours, delay, product):
    """Inputs only: observations end at decision minus delay; one issued run covers the horizon."""
    origin = ts(case["origin"])
    end = origin - pd.Timedelta(hours=delay)
    past = pd.date_range(end=end, periods=context_hours, freq="h")
    future = pd.date_range(end + pd.Timedelta(hours=1), periods=horizon_hours + delay, freq="h")
    source = obs[obs.station.eq(case["station"])].set_index("time")
    history = source.reindex(past)[variables].to_numpy(np.float32).T
    if (np.isfinite(history).mean(axis=-1) < 0.8).any() or not np.isfinite(history[:, -1]).all():
        raise ValueError("insufficient causal hourly observation context")
    values, provenance = choose_arome(issued, case["station"], origin, future, variables, product)
    cases = pd.DataFrame([{"case_id": case.name, "station": case["station"], "origin": origin,
                           "arome_run": provenance["run"], "arome_product": product}])
    anchors = np.array([end.tz_localize(None)], dtype="datetime64[ns]")
    past_axis = past.tz_localize(None).to_numpy(dtype="datetime64[ns]")[None]
    future_axis = future.tz_localize(None).to_numpy(dtype="datetime64[ns]")[None]
    named = build_arome_inputs(issued[issued["product"].eq(product)], cases, anchors,
                               past_axis, future_axis, variables)
    batch = CaseBatch(cases, variables, history[None], anchors, future_axis, 1,
                      arome=values[None], arome_inputs=named)
    return batch, provenance


def vector(batch, scale, query_hours, forecast_hours, delay, method):
    if method not in METHODS or batch.arome_inputs is None or batch.arome is None:
        raise ValueError("error retrieval requires issued AROME and vintage-checked history")
    observed = batch.history[0, :, -query_hours:]
    errors = observed - batch.arome_inputs.history[0, :, -query_hours:]
    issued = batch.arome[0, :, delay:delay + forecast_hours]
    if observed.shape[-1] != query_hours or issued.shape[-1] != forecast_hours:
        raise ValueError("error retrieval query window is too short")
    if not all(np.isfinite(a).all() for a in (observed, errors, issued)):
        raise ValueError("error retrieval needs complete recent observations, verified errors and issued forecasts")
    if method == "shape":
        issued = issued - observed[:, -1:]
        observed = observed - observed[:, -1:]
    # Error levels retain their sign and bias, including in shape matching.
    return np.concatenate([a / scale[:, None] for a in (observed, errors, issued)], axis=1).ravel()


def build(archive_id, cfg, store=None):
    store = store or Store()
    archive = store.get("archives", archive_id)
    variables, product = cfg["variables"], cfg["product"]
    obs = observations(pd.read_parquet(archive / "observations.parquet"), variables)
    issued = issued_forecasts(pd.read_parquet(archive / "issued.parquet"))
    issued = issued[issued.station.isin(cfg["stations"]) & issued["product"].eq(product)]
    rule = cfg["retrieval"]
    cutoff = ts(rule["scale_end_exclusive"])
    if cutoff > ts("2026-01-01") or rule["kind"] != KIND:
        raise ValueError("error-index scaling must use pre-2026 observations only")
    training = obs[obs.station.isin(cfg["stations"]) & (obs.time >= ts(rule["scale_start"])) & (obs.time < cutoff)]
    scale = training[variables].std().to_numpy(np.float32)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("pre-2026 observations need nonzero variance for every retrieval variable")
    context, horizon, delay = cfg["context_hours"], max(cfg["horizons_hours"]), cfg["observation_delay_hours"]
    if (context < rule["query_hours"] or rule["forecast_query_hours"] > min(cfg["horizons_hours"])
            or delay < 0 or context % 24 or rule["k"] < 1):
        raise ValueError("invalid error-index context, forecast-query window, delay or neighbor count")
    rows, payloads, gaps, vectors = [], [], [], {m: [] for m in METHODS}
    for (station, run), group in issued.groupby(["station", "run"], sort=True):
        origin = group.available_at.max()
        if origin < ts(rule["start"]):
            continue
        case = pd.Series({"station": station, "origin": origin}, name=f"{station}/{origin.isoformat()}")
        try:
            batch, provenance = make_batch(obs, issued, case, variables, context, horizon, delay, product)
            if ts(provenance["run"]) != run:
                raise ValueError("another issued run was selected for this decision")
            encoded = {m: vector(batch, scale, rule["query_hours"], rule["forecast_query_hours"], delay, m)
                       for m in METHODS}
            times = pd.to_datetime(batch.future_times[0], utc=True)
            truth = obs[obs.station.eq(station)].set_index("time").reindex(times)[variables].to_numpy(np.float32).T
            available = times[-1] + pd.Timedelta(hours=delay)
            if not np.isfinite(truth).all() or available > pd.Timestamp.now(tz="UTC"):
                raise ValueError("forecast continuation is not completely verified")
            past_errors = batch.history[0] - batch.arome_inputs.history[0]
            future_errors = truth - batch.arome[0]
            payloads.append(np.concatenate([past_errors, future_errors], axis=1))
            scored = future_errors[:, delay:]
            rows.append({"station": station, "origin": origin, "run": run, "available_at": available,
                         "payload_end": times[-1], "forecast_available_at": ts(provenance["available_at"]),
                         **{f"mae_{v}": float(np.mean(np.abs(scored[j]))) for j, v in enumerate(variables)},
                         **{f"bias_{v}": float(np.mean(scored[j])) for j, v in enumerate(variables)}})
            for m in METHODS:
                vectors[m].append(encoded[m])
        except ValueError as exc:
            gaps.append({"station": station, "run": str(run), "reason": str(exc)})
    if not rows:
        raise ValueError("no complete issued AROME/error cases; inspect archive and recent-error coverage")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        np.save(out / "payload.npy", np.stack(payloads).astype(np.float32), allow_pickle=False)
        for m in METHODS:
            np.save(out / f"vectors-{m}.npy", np.stack(vectors[m]).astype(np.float32), allow_pickle=False)
        pd.DataFrame(rows).to_parquet(out / "windows.parquet", index=False)
        write_json(out / "index.json", {"version": VERSION, "kind": KIND, "experiment": cfg["experiment"],
                   "archive_id": archive_id, "product": product, "variables": variables, "context_hours": context,
                   "horizon_hours": horizon + delay, "observation_delay_hours": delay,
                   "observation_availability": cfg["observation_availability"],
                   "error_summary_horizon_hours": horizon,
                   "query_hours": rule["query_hours"], "forecast_query_hours": rule["forecast_query_hours"],
                   "retrieval": rule, "scale": scale.tolist(), "windows": len(rows), "configuration": cfg,
                   "scale_fit": "pre-2026 station observation standard deviations; no 2026 outcomes in scaling",
                   "payload": "verified observation-minus-issued-AROME errors as native known covariates",
                   "builder_sha256": sha256_file(Path(__file__)),
                   "missing": "weighted finite historical errors; steps absent in every neighbor remain NaN",
                   "eligibility": "same station/product; full verification plus delay strictly before query decision",
                   "gaps": gaps, "local_fine_tuning": "none"})
        return store.put("indexes", out)


class ErrorIndex(Index):
    def __init__(self, path):
        self.path = Path(path)
        self.ident = Store.verify(self.path)["id"]
        self.meta = read_json(self.path / "index.json")
        if self.meta.get("kind") != KIND or self.meta["version"] != VERSION:
            raise ValueError("unsupported AROME-error index")
        self.windows = pd.read_parquet(self.path / "windows.parquet")
        self.payload = np.load(self.path / "payload.npy", mmap_mode="r", allow_pickle=False)
        self.vectors = {m: np.load(self.path / f"vectors-{m}.npy", mmap_mode="r", allow_pickle=False) for m in METHODS}
        self.scale = np.asarray(self.meta["scale"], np.float32)
        n, v, t, h = len(self.windows), len(self.meta["variables"]), self.meta["context_hours"], self.meta["horizon_hours"]
        width = v * (2 * self.meta["query_hours"] + self.meta["forecast_query_hours"])
        if (self.payload.shape != (n, v, t + h) or self.scale.shape != (v,) or not np.isfinite(self.scale).all()
                or (self.scale <= 0).any() or any(a.shape != (n, width) or not np.isfinite(a).all() for a in self.vectors.values())):
            raise ValueError("invalid error-index array shapes or scaling")
        if ts(self.meta["configuration"]["retrieval"]["scale_end_exclusive"]) > ts("2026-01-01"):
            raise ValueError("error-index scaling includes 2026 outcomes")
        expected = self.windows.payload_end + pd.Timedelta(hours=self.meta["observation_delay_hours"])
        ends = self.windows.origin + pd.Timedelta(hours=self.meta["error_summary_horizon_hours"])
        if (not expected.eq(self.windows.available_at).all() or not (self.windows.run < self.windows.origin).all()
                or not ends.eq(self.windows.payload_end).all()
                or not (self.windows.forecast_available_at <= self.windows.origin).all()):
            raise ValueError("invalid error-case publication/verification provenance")

    def retrieve(self, batch, method="shape", k=5):
        t, h = self.meta["context_hours"], batch.horizon
        if (batch.step_hours != 1 or batch.history.shape[-1] != t or h > self.meta["horizon_hours"]
                or batch.variables != self.meta["variables"] or k < 1):
            raise ValueError("error retrieval cadence, context, variables or horizon differs from the index")
        if "arome_product" not in batch.cases or not batch.cases.arome_product.eq(self.meta["product"]).all():
            raise ValueError("error retrieval requires the index's exact AROME product")
        trajectories, audit = [], []
        for i, case in enumerate(batch.cases.itertuples()):
            single = batch.subset(slice(i, i + 1))
            query = vector(single, self.scale, self.meta["query_hours"], self.meta["forecast_query_hours"],
                           self.meta["observation_delay_hours"], method)
            origin = pd.Timestamp(batch.decision_origins[i], tz="UTC")
            mask = self.windows.station.eq(case.station) & self.windows.available_at.lt(origin)
            candidates = np.flatnonzero(mask.to_numpy())
            if len(candidates) < k:
                raise ValueError(f"only {len(candidates)} fully verified earlier AROME cases; need {k}")
            distances = np.mean((self.vectors[method][candidates] - query) ** 2, axis=1)
            order = np.argsort(distances, kind="stable")[:k]
            chosen, distances = candidates[order], distances[order]
            weights = 1 / np.maximum(np.sqrt(distances), 1e-6)
            weights /= weights.sum()
            blocks = np.asarray(self.payload[chosen, :, :t + h])
            finite = np.isfinite(blocks)
            total = np.sum(finite * weights[:, None, None], axis=0)
            summed = np.sum(np.where(finite, blocks, 0) * weights[:, None, None], axis=0)
            result = np.full(summed.shape, np.nan, np.float32)
            np.divide(summed, total, out=result, where=total > 0)
            trajectories.append(result)
            audit.append({"case_id": case.case_id, "index_id": self.ident, "kind": KIND, "method": method,
                          "product": self.meta["product"], "neighbors": [
                              {"row": int(n), "origin": str(self.windows.iloc[n].origin),
                               "run": str(self.windows.iloc[n].run), "available_at": str(self.windows.iloc[n].available_at),
                               "distance": float(d), "weight": float(w),
                               "summary_horizon_hours": self.meta["error_summary_horizon_hours"],
                               "mae_by_variable": {v: float(self.windows.iloc[n][f"mae_{v}"]) for v in batch.variables},
                               "bias_by_variable": {v: float(self.windows.iloc[n][f"bias_{v}"]) for v in batch.variables}}
                              for n, d, w in zip(chosen, distances, weights)]})
        return np.stack(trajectories), audit
