"""Adapter contract, forecast-result structure and the Timestral / raw AROME adapters.

An adapter forecasts a batch of benchmark cases. It declares what it outputs (point,
marginal quantiles or ensemble members) and which inputs it needs. Inputs are split
explicitly into targets' past (`history`), known-future covariates (calendar, issued
AROME) and nothing else: future measured weather is never part of a `CaseBatch`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from weather_lab.common import unit_of

OutputType = Literal["point", "quantiles", "ensemble"]
# Timestral's 21 levels; empirical-quantile climatology uses the same grid so pinball losses compare.
QUANTILE_LEVELS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)


@dataclass
class CaseBatch:
    """Inputs available at each case's origin.

    history:      (N, V, T) observations at times <= origin (NaN = missing), V = `variables`.
    history_times:(T,) offsets in steps relative to origin (-(T-1) ... 0) -> absolute via `origins`.
    future_times: (N, H) valid times.
    arome:        optional (N, V, H) issued AROME forecasts at the valid times (NaN where the
                  variable is not provided), from runs available at the origin.
    arome_history:optional (N, V, T) ARCHIVED AROME forecasts at the history times (NaN where no
                  qualifying archived forecast exists; never filled with observations), with
                  provenance `arome_history_run` (N, T) init times (NaT if none) and
                  `arome_history_lead_h` (N, T) leads. Each value comes from the most recent run that
                  was in the archive both before the origin and before the value's own valid time.
    """

    cases: pd.DataFrame
    variables: list[str]
    history: np.ndarray
    origins: np.ndarray
    future_times: np.ndarray
    step_hours: int
    arome: np.ndarray | None = None
    arome_history: np.ndarray | None = None
    arome_history_run: np.ndarray | None = None
    arome_history_lead_h: np.ndarray | None = None

    def __post_init__(self) -> None:
        n, v, t = self.history.shape
        assert v == len(self.variables), "history variable axis mismatch"
        assert len(self.cases) == n == len(self.origins) == len(self.future_times)
        step = np.timedelta64(self.step_hours, "h")
        # Future targets strictly after the origin; history ends at the origin.
        assert (self.future_times[:, 0] == self.origins + step).all(), "first lead must be origin + 1 step"
        if self.arome is not None:
            assert self.arome.shape == (n, v, self.future_times.shape[1])
        if self.arome_history is not None:
            assert self.arome_history.shape == (n, v, t)
            # claimed vintage: run init + lead must equal the history valid time
            times = self.history_times()
            has = ~np.isnat(self.arome_history_run)
            implied = self.arome_history_run + (np.nan_to_num(self.arome_history_lead_h) * 3600).astype(
                "timedelta64[s]")
            assert (implied[has] == times[has]).all(), "archived AROME provenance does not match valid times"
            no_run = np.broadcast_to(~has[:, None, :], self.arome_history.shape)
            assert np.isnan(self.arome_history[no_run]).all(), "historical AROME value without a source run"

    def subset(self, sl) -> "CaseBatch":
        pick = (lambda a: None if a is None else a[sl])
        return CaseBatch(self.cases.iloc[sl], self.variables, self.history[sl], self.origins[sl],
                         self.future_times[sl], self.step_hours, pick(self.arome), pick(self.arome_history),
                         pick(self.arome_history_run), pick(self.arome_history_lead_h))

    @property
    def horizon(self) -> int:
        return self.future_times.shape[1]

    def history_times(self, length: int | None = None) -> np.ndarray:
        """(N, T) absolute times of the last `length` history steps."""
        t = self.history.shape[-1] if length is None else length
        offsets = np.arange(-(t - 1), 1) * np.timedelta64(self.step_hours, "h")
        return self.origins[:, None] + offsets[None, :]


@dataclass
class ForecastBatch:
    model: str
    revision: str
    output: OutputType
    variables: list[str]
    point: np.ndarray                                   # (N, V, H)
    quantiles: np.ndarray | None = None                 # (N, V, H, Q)
    quantile_levels: tuple[float, ...] | None = None
    members: np.ndarray | None = None                   # (N, V, H, M)
    info: dict = field(default_factory=dict)

    def validate(self, batch: CaseBatch) -> None:
        n, h = len(batch.cases), batch.horizon
        expect = (n, len(self.variables), h)
        if self.point.shape != expect:
            raise ValueError(f"{self.model}: point shape {self.point.shape} != {expect}")
        if not set(self.variables) <= set(batch.variables):
            raise ValueError(f"{self.model}: unknown variables {self.variables}")
        if self.output == "quantiles":
            if self.quantiles is None or self.quantiles.shape != expect + (len(self.quantile_levels),):
                raise ValueError(f"{self.model}: quantiles shape mismatch")
            q = self.quantiles[np.isfinite(self.quantiles).all(-1)]
            if len(q) and (np.diff(q, axis=-1) < -1e-6).any():
                raise ValueError(f"{self.model}: crossing quantiles")
        if self.output == "ensemble" and (self.members is None or self.members.shape[:3] != expect):
            raise ValueError(f"{self.model}: members shape mismatch")
        # Physical sanity in benchmark units (°C, hPa, m/s): catches unit slips at the boundary.
        bounds = {"t": (-60, 60), "pmer": (850, 1100), "u": (-80, 80), "v": (-80, 80)}
        for i, var in enumerate(self.variables):
            vals = self.point[:, i]
            lo, hi = bounds[var]
            bad = np.isfinite(vals) & ((vals < lo) | (vals > hi))
            if bad.any():
                raise ValueError(f"{self.model}: {var} outside [{lo}, {hi}] {unit_of(var)}: {vals[bad][:3]}")

    def to_frame(self, batch: CaseBatch, suite: str) -> pd.DataFrame:
        """Long format: one row per case x variable x lead."""
        n, v, h = self.point.shape
        idx_c, idx_v, idx_h = np.meshgrid(np.arange(n), np.arange(v), np.arange(h), indexing="ij")
        idx_c, idx_v, idx_h = idx_c.ravel(), idx_v.ravel(), idx_h.ravel()
        cases = batch.cases.reset_index(drop=True)
        df = pd.DataFrame({
            "suite": suite,
            "case_id": cases["case_id"].to_numpy()[idx_c],
            "station": cases["station"].to_numpy()[idx_c],
            "origin": pd.to_datetime(batch.origins[idx_c], utc=True),
            "valid_time": pd.to_datetime(batch.future_times[idx_c, idx_h], utc=True),
            "lead_h": (idx_h + 1) * batch.step_hours,
            "variable": np.array(self.variables)[idx_v],
            "model": self.model,
            "model_revision": self.revision,
            "output_type": self.output,
            "point": self.point[idx_c, idx_v, idx_h],
        })
        df["unit"] = df["variable"].map(unit_of)
        if self.quantiles is not None:
            q = self.quantiles[idx_c, idx_v, idx_h]
            for k, level in enumerate(self.quantile_levels):
                df[qcol(level)] = q[:, k]
        if self.members is not None:
            df["members"] = list(self.members[idx_c, idx_v, idx_h])
        for col in ("arome_run", "arome_available_at"):
            if col in cases:
                df[col] = cases[col].to_numpy()[idx_c]
        return df


def qcol(level: float) -> str:
    return f"q{level:.2f}"


class Adapter:
    """Base class: subclasses set the class attributes and implement `predict`."""

    kind: str = ""
    output: OutputType = "point"
    inputs: frozenset[str] = frozenset({"history"})
    revision: str = "n/a"

    def __init__(self, id: str, variables: list[str], **params) -> None:
        self.id = id
        self.variables = list(variables)
        self.params = params

    def fit(self, fit_obs: pd.DataFrame) -> None:  # noqa: B027 - optional hook
        """Fit on fitting-period observations only (no-op for most adapters)."""

    def predict(self, batch: CaseBatch) -> ForecastBatch:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"id": self.id, "kind": self.kind, "output": self.output, "inputs": sorted(self.inputs),
                "variables": self.variables, "revision": self.revision, "params": self.params}


# --------------------------------------------------------------------------- raw AROME

class AromeRaw(Adapter):
    """Issued AROME point forecasts at the nearest grid point. Deterministic: no quantiles."""

    kind = "arome"
    output = "point"
    inputs = frozenset({"arome"})

    def __init__(self, id: str, variables: list[str], fields: list[str], revision: str = "AROME 0.025 SP1", **p):
        super().__init__(id, [v for v in variables if v in fields], **p)
        self.revision = revision

    def predict(self, batch: CaseBatch) -> ForecastBatch:
        if batch.arome is None:
            raise ValueError("arome adapter needs issued AROME forecasts in the batch")
        idx = [batch.variables.index(v) for v in self.variables]
        return ForecastBatch(self.id, self.revision, "point", self.variables, batch.arome[:, idx].copy())


# --------------------------------------------------------------------------- Timestral

_MODEL_CACHE: dict[tuple[str, str], tuple[object, dict]] = {}
_MODEL_LOCK = threading.Lock()


def load_timestral(checkpoint: str, revision: str, local_files_only: bool = False) -> tuple[object, dict]:
    """Cached `TimestralForecaster.from_pretrained`; returns (forecaster, load-timing info)."""
    key = (checkpoint, revision)
    with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            import torch
            from huggingface_hub import try_to_load_from_cache
            from timestral import TimestralForecaster

            cached = isinstance(try_to_load_from_cache(checkpoint, "model.safetensors", revision=revision), str)
            t0 = time.perf_counter()
            fc = TimestralForecaster.from_pretrained(checkpoint, device="cpu", revision=revision,
                                                     local_files_only=local_files_only)
            fc.model.eval()
            info = {"first_load_s": round(time.perf_counter() - t0, 3), "weights_were_cached": cached,
                    "torch_threads": torch.get_num_threads(),
                    "n_parameters": int(sum(p.numel() for p in fc.model.parameters())),
                    "supports_known_covariates": bool(fc.supports_known_covariates),
                    "quantile_levels": list(fc.quantile_levels)}
            _MODEL_CACHE[key] = (fc, info)
        return _MODEL_CACHE[key]


def weights_cached(checkpoint: str, revision: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(checkpoint, "model.safetensors", revision=revision), str)


def calendar_features(times: np.ndarray) -> np.ndarray:
    """(N, T) datetime64 -> (N, 4, T): sin/cos hour-of-day, sin/cos day-of-year. Known in advance."""
    t = pd.DatetimeIndex(times.ravel())
    hour = (t.hour + t.minute / 60).to_numpy().reshape(times.shape)
    doy = (t.dayofyear - 1).to_numpy().reshape(times.shape)
    a, b = 2 * np.pi * hour / 24, 2 * np.pi * doy / 365.25
    return np.stack([np.sin(a), np.cos(a), np.sin(b), np.cos(b)], axis=1).astype(np.float32)


class Timestral(Adapter):
    """Timestral zero-shot inference: no local fine-tuning, no calibration.

    targets  : variables forecast jointly per station (stations are independent batch items).
    calendar : add known-future cyclic hour-of-day / day-of-year covariates.
    arome    : adds one known-future covariate row per AROME field (t, pmer). Over the horizon the
               row holds the issued forecast of the run available at the origin. Over the context
               (history) it holds, depending on the mode:
               - "archived": archived AROME forecasts from earlier runs (available before the origin
                 and before each value's valid time); NaN where none exists. No substitution.
               - "missing": NaN everywhere (future-only covariate).
               - "substituted_observations" (alias "observations", the v1 behaviour): the station's
                 own observations stand in for AROME. This is NOT an AROME value.
    """

    AROME_MODES = ("archived", "missing", "substituted_observations", "observations")

    kind = "timestral"
    output = "quantiles"

    def __init__(self, id: str, variables: list[str], targets: list[str], checkpoint: str, revision: str,
                 context_days: int = 30, calendar: bool = False, arome: str | None = None,
                 arome_fields: list[str] | None = None, batch_size: int = 32, **p):
        super().__init__(id, targets, calendar=calendar, arome=arome, context_days=context_days,
                         checkpoint=checkpoint, **p)
        self.checkpoint, self.revision = checkpoint, revision
        self.context_days, self.calendar, self.arome_mode = context_days, calendar, arome
        self.arome_fields = [f for f in (arome_fields or []) if f in targets]
        self.batch_size = batch_size
        inputs = {"history"} | ({"calendar"} if calendar else set()) | ({"arome"} if arome else set())
        self.inputs = frozenset(inputs)
        if arome is not None and arome not in self.AROME_MODES:
            raise ValueError(f"arome mode {arome!r}")

    def context_steps(self, step_hours: int) -> int:
        return self.context_days * 24 // step_hours

    def arome_history_rows(self, batch: CaseBatch, j: int, t_ctx: int) -> np.ndarray:
        """(N, t_ctx) historical segment of the AROME covariate row for batch variable `j`."""
        n = len(batch.cases)
        if self.arome_mode == "archived":
            if batch.arome_history is None:
                raise ValueError(f"{self.id} needs archived historical AROME in the batch")
            return batch.arome_history[:, j, -t_ctx:]
        if self.arome_mode in ("substituted_observations", "observations"):
            return batch.history[:, j, -t_ctx:]
        return np.full((n, t_ctx), np.nan)

    def build_inputs(self, batch: CaseBatch) -> tuple[np.ndarray, np.ndarray | None]:
        """(context (N, V, T), known covariates (N, C, T + H) or None) exactly as passed to Timestral."""
        t_ctx = self.context_steps(batch.step_hours)
        if batch.history.shape[-1] < t_ctx:
            raise ValueError(f"batch history {batch.history.shape[-1]} < context {t_ctx}")
        idx = [batch.variables.index(v) for v in self.variables]
        ctx = batch.history[:, idx, -t_ctx:].astype(np.float32)
        known_rows = []
        if self.calendar:
            times = np.concatenate([batch.history_times(t_ctx), batch.future_times], axis=1)
            known_rows.append(calendar_features(times))
        if self.arome_mode:
            if batch.arome is None:
                raise ValueError(f"{self.id} needs issued AROME forecasts")
            for f in self.arome_fields:
                j = batch.variables.index(f)
                past = self.arome_history_rows(batch, j, t_ctx)
                known_rows.append(np.concatenate([past, batch.arome[:, j]], axis=1)[:, None].astype(np.float32))
        return ctx, (np.concatenate(known_rows, axis=1) if known_rows else None)

    def predict(self, batch: CaseBatch) -> ForecastBatch:
        import torch

        fc, load_info = load_timestral(self.checkpoint, self.revision)
        t_ctx = self.context_steps(batch.step_hours)
        idx = [batch.variables.index(v) for v in self.variables]
        ctx, known = self.build_inputs(batch)
        h = batch.horizon
        out = np.full((len(ctx), len(idx), h, len(fc.quantile_levels)), np.nan, dtype=np.float32)
        t0 = time.perf_counter()
        with torch.inference_mode():
            for s in range(0, len(ctx), self.batch_size):
                sl = slice(s, s + self.batch_size)
                res = fc.forecast(ctx[sl], prediction_length=h,
                                  known_covariates=None if known is None else known[sl],
                                  clip_nonnegative=False)  # temperatures and wind components can be negative
                out[sl] = res.quantiles.cpu().numpy()
        elapsed = time.perf_counter() - t0
        levels = tuple(float(x) for x in fc.quantile_levels)
        point = out[..., levels.index(0.5)]
        return ForecastBatch(self.id, f"{self.checkpoint}@{self.revision[:12]}", "quantiles", self.variables,
                             point, out, levels,
                             info={**load_info, "predict_s": round(elapsed, 3), "n_cases": len(ctx),
                                   "batch_size": self.batch_size, "context_steps": t_ctx,
                                   "known_covariate_rows": 0 if known is None else known.shape[1]})

    def latency_probe(self, batch: CaseBatch, repeats: int = 5) -> dict:
        """Warm single-case latency (median of `repeats`) on the first case of `batch`."""
        one = batch.subset(slice(0, 1))
        self.predict(one)  # warm-up
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            self.predict(one)
            times.append(time.perf_counter() - t0)
        return {"warm_single_case_ms_median": round(1000 * float(np.median(times)), 2), "repeats": repeats}


# --------------------------------------------------------------------------- exploratory (UI)

CHECKPOINTS = {  # pinned Hub revisions inspected on 2026-10-24
    "Timestral-5M": ("mistralai/Timestral-5M", "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f", 21),
    "Timestral-28M": ("mistralai/Timestral-28M", "49b26409ac87b5935539a5d8ffe7f397b720db47", 111),
    "Timestral-209M": ("mistralai/Timestral-209M", "9874d252ab0aae16f7f7c999f98f98d6ad7d8365", 837),
}
_EXPLORE_SEMAPHORE = threading.Semaphore(1)  # cap concurrent CPU inference requests


def exploratory_forecast(obs: pd.DataFrame, case: pd.Series, variables: list[str], size: str, context_days: int,
                         calendar: bool, horizon_steps: int = 16, step_hours: int = 3,
                         allow_download: bool = False) -> pd.DataFrame:
    """One-off Timestral run on a single case, OUTSIDE the frozen benchmark (never saved to runs/).

    `obs` holds the station's observation window; only rows at or before the origin are used.
    """
    checkpoint, revision, mb = CHECKPOINTS[size]
    if not weights_cached(checkpoint, revision) and not allow_download:
        raise PermissionError(f"{size} weights (~{mb} MB) are not cached; tick 'allow download' to fetch them")
    origin = pd.Timestamp(case["origin"])
    step = pd.Timedelta(hours=step_hours)
    t_ctx = context_days * 24 // step_hours
    axis = pd.date_range(origin - (t_ctx - 1) * step, origin, freq=step)
    hist = obs.set_index("time").reindex(axis)[variables]          # strictly <= origin
    if hist.notna().mean().min() < 0.5:
        raise ValueError("observation window too short or sparse for this context length")
    batch = CaseBatch(pd.DataFrame([{"case_id": case.name, "station": case["station"]}]), variables,
                      hist.to_numpy(dtype=np.float32).T[None], np.array([origin.to_datetime64()]),
                      np.array([[(origin + (k + 1) * step).to_datetime64() for k in range(horizon_steps)]]),
                      step_hours)
    model = Timestral(f"exploratory {size} ctx{context_days}d{' +cal' if calendar else ''}", variables,
                      targets=variables, checkpoint=checkpoint, revision=revision, context_days=context_days,
                      calendar=calendar, batch_size=1)
    with _EXPLORE_SEMAPHORE:
        fb = model.predict(batch)
    fb.validate(batch)
    return fb.to_frame(batch, "exploratory (outside frozen benchmark)")
