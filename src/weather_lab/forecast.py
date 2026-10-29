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

from weather_lab.common import ts, unit_of
from weather_lab.arome_inputs import AromeInputs, build_frozen_arome_inputs

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
    aux_history:  optional (N, F, T) auxiliary observations (`aux_variables`, e.g. dew point, cloud
                  cover) at the same history times (<= origin); context-enrichment inputs only.
    station_coords: optional {station: (lat, lon)} for astronomical features.
    arome_inputs: optional named fields, independent of targets; source runs and availability per field.
    origins: model-axis anchors (history ends here). cases.origin retains the decision time;
             delayed replay generates bridge steps which are excluded from exported leads.
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
    aux_history: np.ndarray | None = None
    aux_variables: list[str] | None = None
    station_coords: dict | None = None
    arome_inputs: AromeInputs | None = None

    def __post_init__(self) -> None:
        n, v, t = self.history.shape
        assert v == len(self.variables), "history variable axis mismatch"
        assert len(self.cases) == n == len(self.origins) == len(self.future_times)
        step = np.timedelta64(self.step_hours, "h")
        # Future targets strictly after the origin; history ends at the origin.
        assert (self.future_times[:, 0] == self.origins + step).all(), "first lead must be origin + 1 step"
        assert (self.origins <= self.decision_origins).all(), "history ends after the decision time"
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
        if self.aux_history is not None:
            assert self.aux_history.shape == (n, len(self.aux_variables or []), t), "aux history shape"
        if self.arome_inputs is not None:
            self.arome_inputs.validate(self.origins, self.history_times(), self.future_times,
                                       self.decision_origins)

    def subset(self, sl) -> "CaseBatch":
        pick = (lambda a: None if a is None else a[sl])
        return CaseBatch(self.cases.iloc[sl], self.variables, self.history[sl], self.origins[sl],
                         self.future_times[sl], self.step_hours, pick(self.arome), pick(self.arome_history),
                         pick(self.arome_history_run), pick(self.arome_history_lead_h), pick(self.aux_history),
                         self.aux_variables, self.station_coords,
                         None if self.arome_inputs is None else self.arome_inputs.subset(sl))

    @property
    def horizon(self) -> int:
        return self.future_times.shape[1]

    @property
    def decision_origins(self) -> np.ndarray:
        return (self.cases["origin"].to_numpy(dtype="datetime64[ns]")
                if "origin" in self.cases else self.origins)

    @property
    def lead_hours(self) -> np.ndarray:
        return ((self.future_times - self.decision_origins[:, None]) / np.timedelta64(1, "h")).astype(int)

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
            if tuple(round(q, 6) for q in self.quantile_levels) != QUANTILE_LEVELS:
                # Pinball and coverage are compared on one grid. Never relabel or invent levels: an adapter
                # whose native levels differ must map them explicitly and record how (info["quantile_mapping"]).
                raise ValueError(f"{self.model}: quantile levels {self.quantile_levels} differ from the benchmark "
                                 "grid; map them explicitly in the adapter")
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
            "origin": pd.to_datetime(batch.decision_origins[idx_c], utc=True),
            "valid_time": pd.to_datetime(batch.future_times[idx_c, idx_h], utc=True),
            "lead_h": batch.lead_hours[idx_c, idx_h],
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
        return df.loc[df["lead_h"] > 0].reset_index(drop=True)


def qcol(level: float) -> str:
    return f"q{level:.2f}"


class Adapter:
    """Base class: subclasses set the class attributes and implement `predict`.

    Contract (also docs/adding_a_model.md): `predict(CaseBatch) -> ForecastBatch` with point forecasts
    (N, V, H) in benchmark units and, for probabilistic systems, quantiles on `QUANTILE_LEVELS`. An
    adapter declares the inputs it consumes in `inputs`; `family` names its entry in
    `weather_lab.models.FAMILIES` (capabilities), so an impossible input setup is reported as
    unsupported rather than run. Preprocessing is the adapter's own business.
    """

    kind: str = ""
    family: str | None = None
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


class AromeBiasCorrected(Adapter):
    """Issued AROME + a shrunk mean of its recent errors at the same run lead (deterministic, point only).

    For a target valid time with run lead l (valid − run init), the past residuals are
    observation(r + l) − AROME(r, r + l) for earlier runs r available at the origin, with r + l at or
    before the origin (so the verifying observation was available) and r within `window_days`.
    Correction = sum(residuals) / (n + shrink_k): 0 with no residuals (then exactly raw AROME), and
    the plain mean as n grows. `window_days` and `shrink_k` are fixed in the config, not tuned.
    """

    kind = "arome_biascorr"
    output = "point"
    inputs = frozenset({"arome", "history"})

    def __init__(self, id: str, variables: list[str], fields: list[str], window_days: int = 7, shrink_k: float = 5.0,
                 revision: str = "AROME 0.025 SP1 + lead bias correction", **p):
        super().__init__(id, [v for v in variables if v in fields], window_days=window_days, shrink_k=shrink_k, **p)
        self.window_days, self.shrink_k, self.revision = window_days, float(shrink_k), revision
        self.table: pd.DataFrame | None = None

    def attach_arome(self, arome_df: pd.DataFrame) -> None:
        self.table = arome_df

    def predict(self, batch: CaseBatch) -> ForecastBatch:
        if batch.arome is None or self.table is None:
            raise ValueError("arome_biascorr needs issued AROME forecasts and the AROME table")
        a = self.table
        avail = a.groupby("run")["available_at"].max()
        vals = a.set_index(["station", "run", "variable", "valid_time"])["value"]
        idx = [batch.variables.index(v) for v in self.variables]
        out = batch.arome[:, idx].copy()
        step = np.timedelta64(batch.step_hours, "h")
        n_used = np.zeros(out.shape, dtype=np.int16)
        for i, c in enumerate(batch.cases.itertuples()):
            origin = pd.Timestamp(batch.decision_origins[i], tz="UTC")
            run0 = pd.Timestamp(c.arome_run)
            hist_t = pd.DatetimeIndex(batch.history_times()[i], tz="UTC")
            hist_pos = {t: k for k, t in enumerate(hist_t)}
            past_runs = [r for r in avail.index if avail[r] <= origin and r < run0
                         and r >= origin - pd.Timedelta(days=self.window_days)]
            for jj, var in enumerate(self.variables):
                j = batch.variables.index(var)
                for k in range(out.shape[-1]):
                    lead = pd.Timestamp(batch.future_times[i, k], tz="UTC") - run0
                    res = []
                    for r in past_runs:
                        v = r + lead
                        if v > origin or v not in hist_pos:
                            continue
                        f = vals.get((c.station, r, var, v))
                        o = batch.history[i, j, hist_pos[v]]
                        if f is not None and np.isfinite(o):
                            res.append(o - f)
                    out[i, jj, k] += sum(res) / (len(res) + self.shrink_k) if res else 0.0
                    n_used[i, jj, k] = len(res)
        return ForecastBatch(self.id, self.revision, "point", self.variables, out,
                             info={"residuals_per_value_median": float(np.median(n_used)),
                                   "share_without_residuals": round(float((n_used == 0).mean()), 4)})


# --------------------------------------------------------------------------- Timestral

def load_timestral(checkpoint: str, revision: str, device: str = "cpu") -> tuple[object, dict]:
    """Shared loaded model for this checkpoint revision/device (see `weather_lab.models.ModelCache`)."""
    from weather_lab.models import MODELS, resolve

    item = MODELS.get(resolve(checkpoint, revision), device)
    return item.forecaster, item.info


def ensure_checkpoint(size: str, allow_download: bool) -> bool:
    """Make the pinned checkpoint available locally (no GPU needed). Returns True if it was downloaded."""
    from weather_lab.models import CHECKPOINTS as REGISTRY, prefetch

    return prefetch(REGISTRY[size], allow_download=allow_download)["downloaded"]


def calendar_features(times: np.ndarray) -> np.ndarray:
    """(N, T) datetime64 -> (N, 4, T): sin/cos hour-of-day, sin/cos day-of-year. Known in advance."""
    t = pd.DatetimeIndex(times.ravel())
    hour = (t.hour + t.minute / 60).to_numpy().reshape(times.shape)
    doy = (t.dayofyear - 1).to_numpy().reshape(times.shape)
    a, b = 2 * np.pi * hour / 24, 2 * np.pi * doy / 365.25
    return np.stack([np.sin(a), np.cos(a), np.sin(b), np.cos(b)], axis=1).astype(np.float32)


def error_inputs(batch: CaseBatch, variables: list[str], t_ctx: int):
    """Verified residual history, issued baseline and source-run leads for native adapters."""
    if batch.arome is None or (batch.arome_history is None and batch.arome_inputs is None):
        raise ValueError("AROME error prediction needs issued and qualifying archived AROME forecasts")
    idx = [batch.variables.index(v) for v in variables]
    baseline = batch.arome[:, idx]
    for j, name in enumerate(variables):
        if not np.isfinite(baseline[:, j]).all():
            raise ValueError(f"AROME error prediction needs an issued {name} forecast at every requested lead")
    if batch.arome_inputs is not None:
        fields = [batch.arome_inputs.fields.index(v) for v in variables]
        archived = batch.arome_inputs.history[:, fields, -t_ctx:]
    else:
        archived = batch.arome_history[:, idx, -t_ctx:]
    history = batch.history[:, idx, -t_ctx:] - archived
    for j, name in enumerate(variables):
        if not np.isfinite(history[:, j]).any(axis=-1).all():
            raise ValueError(f"no observed {name} errors from qualifying archived AROME in this context")
    runs = pd.to_datetime(batch.cases["arome_run"], utc=True).to_numpy(dtype="datetime64[ns]")
    if np.isnat(runs).any() or (runs > batch.decision_origins).any():
        raise ValueError("AROME error prediction needs a source run at or before the forecast origin")
    future_leads = (batch.future_times - runs[:, None]) / np.timedelta64(1, "h")
    if not np.isfinite(future_leads).all() or (future_leads <= 0).any():
        raise ValueError("AROME error prediction needs a source run earlier than every future valid time")
    if batch.arome_inputs is not None:
        if not np.array_equal(baseline, batch.arome_inputs.future[:, fields], equal_nan=True):
            raise ValueError("AROME baseline differs from its named provenance")
        past_leads = (batch.history_times(t_ctx)[:, None] - batch.arome_inputs.history_run[:, fields, -t_ctx:]) / np.timedelta64(1, "h")
        past_leads = np.where(np.isfinite(history), past_leads, np.nan)
        future = np.broadcast_to(future_leads[:, None], (*history.shape[:2], batch.horizon))
        leads = np.concatenate([past_leads, future], axis=-1)
    else:
        past_leads = np.where(np.isfinite(history).any(axis=1), batch.arome_history_lead_h[:, -t_ctx:], np.nan)
        leads = np.concatenate([past_leads, future_leads], axis=1)[:, None]
    return history.astype(np.float32), baseline, leads.astype(np.float32)


class Timestral(Adapter):
    """Timestral zero-shot inference: no local fine-tuning, no calibration.

    targets  : variables forecast jointly per station (stations are independent batch items).
    prediction_target: "weather" forecasts observations; "arome_error" forecasts obs − archived
               AROME, then adds the issued AROME baseline to every predicted quantile.
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
    ADAPTER_VERSION = 1          # bump when predict()/build_inputs() change outputs
    PREPROCESSING_VERSION = 1    # bump when the input construction (context, covariates) changes

    kind = "timestral"
    family = "timestral"
    output = "quantiles"

    def __init__(self, id: str, variables: list[str], targets: list[str], checkpoint: str, revision: str,
                 context_days: int = 30, calendar: bool = False, arome: str | None = None,
                 arome_fields: list[str] | None = None, batch_size: int = 32, device: str = "cpu",
                 features: list[str] | dict | None = None, prediction_target: str = "weather", precision: str = "native", **p):
        from weather_lab import features as feats

        if prediction_target not in ("weather", "arome_error"):
            raise ValueError(f"unknown prediction target {prediction_target!r}")
        from weather_lab import precision as precision_module
        self.precision = precision
        if precision != "native":
            p["precision"] = precision_module.describe(precision)
        self.prediction_target = prediction_target
        self.feature_spec = feats.normalize(features or [])     # {group: complete parameters}
        self.features = list(self.feature_spec)
        changed = feats.non_default(self.feature_spec)
        extra = {"features": self.features, "feature_versions": {f: feats.GROUPS[f].version for f in self.features},
                 **({"feature_params": changed} if changed else {})} if self.features else {}
        if prediction_target == "arome_error":
            extra.update(prediction_target=prediction_target, residual_version=1,
                         residual_history="latest qualifying archived forecast", forecast_lead_input=True)
        super().__init__(id, targets, calendar=calendar, arome=arome, context_days=context_days,
                         checkpoint=checkpoint, **extra, **p)
        self.checkpoint, self.revision = checkpoint, revision
        self.context_days, self.calendar, self.arome_mode = context_days, calendar, arome
        self.arome_fields = [f for f in (arome_fields or []) if f in targets]
        self.batch_size, self.device = batch_size, device
        needs = {dep for name in self.features for dep in feats.GROUPS[name].needs}
        inputs = {"history"} | ({"calendar"} if calendar else set()) | ({"arome"} if arome else set()) \
            | ({"features"} if self.features else set()) | ({"arome"} if "arome_history" in needs else set())
        if prediction_target == "arome_error":
            inputs.add("arome")
        self.inputs = frozenset(inputs)
        if arome is not None and arome not in self.AROME_MODES:
            raise ValueError(f"arome mode {arome!r}")

    def context_steps(self, step_hours: int) -> int:
        return self.context_days * 24 // step_hours

    def arome_history_rows(self, batch: CaseBatch, j: int, t_ctx: int) -> np.ndarray:
        """(N, t_ctx) historical segment of the AROME covariate row for batch variable `j`."""
        n = len(batch.cases)
        if self.arome_mode == "archived":
            if batch.arome_inputs is not None:
                field = batch.arome_inputs.fields.index(batch.variables[j])
                return batch.arome_inputs.history[:, field, -t_ctx:]
            if batch.arome_history is None:
                raise ValueError(f"{self.id} needs archived historical AROME in the batch")
            return batch.arome_history[:, j, -t_ctx:]
        if self.arome_mode in ("substituted_observations", "observations"):
            return batch.history[:, j, -t_ctx:]
        return np.full((n, t_ctx), np.nan)

    def error_inputs(self, batch: CaseBatch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Residual history, issued baseline and lead row; original observation inputs stay intact."""
        return error_inputs(batch, self.variables, self.context_steps(batch.step_hours))

    def build_inputs(self, batch: CaseBatch) -> tuple[np.ndarray, np.ndarray | None]:
        """(context (N, V, T), known covariates (N, C, T + H) or None) exactly as passed to Timestral."""
        t_ctx = self.context_steps(batch.step_hours)
        if batch.history.shape[-1] < t_ctx:
            raise ValueError(f"batch history {batch.history.shape[-1]} < context {t_ctx}")
        idx = [batch.variables.index(v) for v in self.variables]
        ctx = batch.history[:, idx, -t_ctx:].astype(np.float32)
        if self.prediction_target == "arome_error":
            ctx, _, leads = self.error_inputs(batch)
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
        if self.features:
            _, known_feat, _ = self.feature_rows(batch)
            if known_feat is not None:
                known_rows.append(known_feat)
        if self.prediction_target == "arome_error":
            known_rows.append(leads)
        return ctx, (np.concatenate(known_rows, axis=1) if known_rows else None)

    def feature_rows(self, batch: CaseBatch) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
        """(past covariates (N, Cp, T) | None, known feature rows (N, Ck, T + H) | None, report)."""
        from weather_lab import features as feats

        if not self.features:
            return None, None, {}
        return feats.build(self.feature_spec, batch, self.context_steps(batch.step_hours))

    def predict(self, batch: CaseBatch) -> ForecastBatch:
        import torch

        from weather_lab.models import MODELS, resolve

        ctx, known = self.build_inputs(batch)
        item = MODELS.get(resolve(self.checkpoint, self.revision), self.device)
        fc, load_info = item.forecaster, item.info
        t_ctx = self.context_steps(batch.step_hours)
        idx = [batch.variables.index(v) for v in self.variables]
        past, _, feat_report = self.feature_rows(batch)
        h = batch.horizon
        out = np.full((len(ctx), len(idx), h, len(fc.quantile_levels)), np.nan, dtype=np.float32)
        t0 = time.perf_counter()
        from weather_lab import precision as precision_module
        with item.lock, torch.inference_mode(), precision_module.inference(self.precision, self.device) as precision_guard:
            # Inputs are per call; precision contexts never mutate the cached model.
            for s in range(0, len(ctx), self.batch_size):
                sl = slice(s, s + self.batch_size)
                res = fc.forecast(ctx[sl], prediction_length=h,
                                  known_covariates=None if known is None else known[sl],
                                  past_covariates=None if past is None else past[sl],
                                  clip_nonnegative=False)  # temperatures and wind components can be negative
                out[sl] = res.quantiles.float().cpu().numpy()
        residual_info = {}
        if self.prediction_target == "arome_error":
            _, baseline, _ = self.error_inputs(batch)
            out += baseline[..., None]
            residual_info = {"prediction_target": "arome_error", "residual_version": 1,
                             "residual_observed_steps": np.isfinite(ctx).sum(axis=-1).tolist(),
                             "forecast_lead_input": True}
        elapsed = time.perf_counter() - t0
        levels = tuple(float(x) for x in fc.quantile_levels)
        point = out[..., levels.index(0.5)]
        return ForecastBatch(self.id, f"{self.checkpoint}@{self.revision[:12]}", "quantiles", self.variables,
                             point, out, levels,
                             info={**load_info, "predict_s": round(elapsed, 3), "n_cases": len(ctx),
                                   "batch_size": self.batch_size, "context_steps": t_ctx,
                                   "known_covariate_rows": 0 if known is None else known.shape[1],
                                   "past_covariate_rows": 0 if past is None else past.shape[1],
                                   **residual_info,
                                   **({"precision": precision_module.describe(self.precision),
                                       "narrowed_fp64_ops": precision_guard.narrowed_fp64_ops} if precision_guard else {}),
                                   **({"feature_inputs": feat_report} if feat_report else {})})

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

def _registry() -> dict:
    from weather_lab.models import CHECKPOINTS as REGISTRY

    return {name: ck for name, ck in REGISTRY.items() if ck.family == "timestral"}


CHECKPOINTS = {k: (c.repo, c.revision, c.mb) for k, c in _registry().items()}   # (repo, revision, MB)
_EXPLORE_SEMAPHORE = threading.Semaphore(1)  # cap concurrent inference requests
AROME_LABELS = {None: "", "archived": " +AROME archived history", "substituted_observations": " +AROME observed history",
                "missing": " +AROME forecast only"}


AROME_RUN_HOURS = 51          # SP1 0.025 deg runs used by the AROME suites (manifest `arome.max_lead_h`)


def arome_coverage_steps(rows: np.ndarray) -> int:
    """Leading forecast steps for which every AROME field (rows (F, H)) has a value."""
    ok = np.isfinite(rows).all(axis=0)
    return int(ok.argmin()) if not ok.all() else int(ok.size)


STATS = {"inputs_built": 0, "inference": 0}   # exploratory input constructions / model forward runs (process)


def hourly_inputs(obs: pd.DataFrame, cases: pd.DataFrame, variables: list[str], context_hours: int,
                  horizon_hours: int, delay: int, *,
                  missing_error: str = "insufficient causal hourly observation context") -> CaseBatch:
    """One station's delayed hourly inputs; preserve the caller's case metadata and never read targets."""
    end = ts(cases.iloc[0].origin) - pd.Timedelta(hours=delay)
    past = pd.date_range(end=end, periods=context_hours, freq="h")
    future = pd.date_range(end + pd.Timedelta(hours=1), periods=horizon_hours + delay, freq="h")
    history = obs.set_index("time").reindex(past)[variables].to_numpy(np.float32).T
    if (np.isfinite(history).mean(axis=-1) < 0.8).any() or not np.isfinite(history[:, -1]).all():
        raise ValueError(missing_error)
    return CaseBatch(cases, variables, history[None],
                     np.array([end.tz_localize(None)], dtype="datetime64[ns]"),
                     future.tz_localize(None).to_numpy(dtype="datetime64[ns]")[None], 1)


def exploratory_inputs(obs: pd.DataFrame, case: pd.Series, variables: list[str], context_days: int,
                       arome_mode: str | None = None, arome_df: pd.DataFrame | None = None,
                       arome_fields: tuple[str, ...] = ("t", "pmer"), horizon_steps: int = 16,
                       step_hours: int = 3, aux_df: pd.DataFrame | None = None,
                       station_coords: dict | None = None, arome_history: bool = False,
                       arome_baseline: bool = False) -> CaseBatch:
    """A one-case batch from saved files: observations <= origin, and (optionally) issued + archived AROME
    from the station extract with the same rules as the benchmark (`arome_inputs.build_frozen_arome_inputs`).

    `arome_history=True` adds the archived AROME history without the issued forecast (for features that
    need past AROME errors when AROME is not a conditioning input); it never falls back to observations.
    """
    STATS["inputs_built"] += 1
    origin = pd.Timestamp(case["origin"])
    step = pd.Timedelta(hours=step_hours)
    t_ctx = context_days * 24 // step_hours
    axis = pd.date_range(origin - (t_ctx - 1) * step, origin, freq=step)
    hist = obs.set_index("time").reindex(axis)[variables]          # strictly <= origin
    if hist.notna().mean().min() < 0.5:
        raise ValueError(f"the installed observations cover too little of the {context_days}-day context before "
                         "this forecast time (more than half missing); choose a shorter context")
    cases = pd.DataFrame([{"case_id": case.name, "station": case["station"], "origin": origin,
                           "arome_run": case.get("arome_run")}])
    origins = np.array([origin.to_datetime64()]).astype("datetime64[ns]")
    future = origins[:, None] + (np.arange(1, horizon_steps + 1) * np.timedelta64(step_hours, "h"))[None]
    arome_arr, hist_args = None, {}
    if arome_mode or arome_baseline:
        if arome_df is None or not isinstance(case.get("arome_run"), str):
            raise ValueError("this forecast has no saved AROME run, so AROME conditioning is unavailable")
        arome_arr, hist_args = build_frozen_arome_inputs(arome_df, cases, origins, variables, list(arome_fields), horizon_steps,
                                            step_hours, t_ctx)
        rows = arome_arr[0, [variables.index(f) for f in arome_fields if f in variables]]
        covered = arome_coverage_steps(rows)
        if covered < horizon_steps:   # never pad: the issued run simply ends there
            raise ValueError(f"AROME run {str(case['arome_run'])[:16].replace('T', ' ')} UTC covers only "
                             f"{covered * step_hours} h after this forecast time (run length "
                             f"{AROME_RUN_HOURS} h); request at most {covered * step_hours} h, or choose "
                             "'No AROME input' for a longer Timestral-only forecast")
    elif arome_history:
        if arome_df is None:
            raise ValueError("no archived AROME forecasts are installed for this experiment")
        from weather_lab import arome as arome_mod

        hist_times = origins[:, None] + (np.arange(-(t_ctx - 1), 1) * np.timedelta64(step_hours, "h"))[None]
        vals, runs, leads = arome_mod.archived_history(arome_df, cases["station"].to_numpy(), origins, hist_times,
                                                       variables)
        hist_args = {"arome_history": vals, "arome_history_run": runs, "arome_history_lead_h": leads}
    aux_args = {}
    if aux_df is not None:
        from weather_lab.features import AUX_VARIABLES

        a = aux_df[aux_df["station"] == case["station"]].set_index("time").reindex(axis)[AUX_VARIABLES]
        aux_args = {"aux_history": a.to_numpy(dtype=np.float32).T[None], "aux_variables": list(AUX_VARIABLES)}
    return CaseBatch(cases, variables, hist.to_numpy(dtype=np.float32).T[None], origins, future, step_hours,
                     arome_arr, **hist_args, **aux_args, station_coords=station_coords)


def exploratory_forecast(obs: pd.DataFrame, case: pd.Series, variables: list[str], size: str, context_days: int,
                         calendar: bool, arome_mode: str | None = None, arome_df: pd.DataFrame | None = None,
                         horizon_steps: int = 16, step_hours: int = 3, allow_download: bool = False,
                         device: str = "cpu", use_cache: bool = True,
                         lookup_only: bool = False, features: list[str] | dict | None = None,
                         aux_df: pd.DataFrame | None = None,
                         station_coords: dict | None = None,
                         prediction_target: str = "weather") -> tuple[pd.DataFrame | None, dict]:
    """One-off Timestral run on a single case, OUTSIDE the frozen benchmark (never saved to runs/).

    Returns (prediction rows, info) where info says whether the prediction cache was hit. The model is
    loaded only on a cache miss. `lookup_only` returns (None, info) on a miss without computing (used to
    avoid requesting a ZeroGPU slot for a cached result). `features`: group names (default parameters)
    or {group: parameters}; built by the shared pipeline (`weather_lab.features`), missing data raise.
    """
    import torch

    from weather_lab import features as feats
    from weather_lab import predcache
    from weather_lab.models import offline, prefetch
    from weather_lab.forecast_config import ForecastConfig, _FROM_MODE, build_adapter

    from weather_lab.models import CHECKPOINTS as REGISTRY

    ck = REGISTRY[size]
    if prediction_target not in ("weather", "arome_error"):
        raise ValueError(f"unknown prediction target {prediction_target!r}")
    correction = prediction_target == "arome_error"
    spec = feats.normalize(features or [])
    needs = {dep for name in spec for dep in feats.GROUPS[name].needs}
    batch = exploratory_inputs(obs, case, variables, context_days, arome_mode, arome_df,
                               horizon_steps=horizon_steps, step_hours=step_hours,
                               aux_df=aux_df if "aux_history" in needs else None,
                               station_coords=station_coords, arome_history="arome_history" in needs,
                               arome_baseline=correction)
    horizon_h = horizon_steps * step_hours
    flabel = "".join(" +" + n + ("*" if feats.non_default({n: p}) else "") for n, p in spec.items())
    method_label = " +AROME error correction" if correction else ""
    cfg = ForecastConfig.make(ck.name, _FROM_MODE[arome_mode], spec, context_days=context_days,
                              horizon_h=horizon_h, calendar=calendar, targets=tuple(variables),
                              step_hours=step_hours, prediction_target=prediction_target)
    model = build_adapter(cfg, f"exploratory {size} ctx{context_days}d h{horizon_h}{' +cal' if calendar else ''}"
                          f"{AROME_LABELS[arome_mode]}{flabel}{method_label} [{device}]",
                          arome=arome_mode, arome_fields=["t", "pmer"], batch_size=1, device=device)
    native = model.build_inputs(batch)
    ctx, known = native[:2]
    if ck.family == "t0":
        past, feat_report = None, native[2]
    else:
        past, _, feat_report = model.feature_rows(batch)
    for row, share in (feat_report.get("observed_fraction") or {}).items():
        if share == 0:      # an empty covariate row is masked but not neutral: never run it silently
            raise ValueError(f"feature row '{row}' has no observed value in the {context_days}-day context before "
                             "this forecast time")
    payload = {"adapter": model.kind, "adapter_version": model.ADAPTER_VERSION,
               "preprocessing": model.PREPROCESSING_VERSION, "checkpoint": ck.repo, "revision": ck.revision,
               "variables": variables, "station": case["station"], "origin": str(case["origin"]),
               "horizon_steps": horizon_steps, "step_hours": step_hours, "context_steps": ctx.shape[-1],
               "calendar": calendar, "arome_mode": arome_mode, "arome_run": case.get("arome_run") if arome_mode else None,
               "quantile_levels": list(QUANTILE_LEVELS), "device": device, "dtype": "float32",
               "torch": torch.__version__, "stochastic": "none (deterministic forward pass)"}
    if ck.family == "t0":
        payload.update(inference_options=model.options, runtime="tfc-t0==0.6.0")
    if spec:     # feature definitions (name, version, role, rows, parameters); actual values/masks hashed below
        payload["features"] = feats.definition(spec)
        payload["arome_history_runs"] = sorted({str(r) for r in np.ravel(batch.arome_history_run)
                                                if not np.isnat(r)}) if "arome_history" in needs else None
    arrays = {"context": ctx, "known_covariates": known, **({"past_covariates": past} if spec else {})}
    error_history = {}
    if correction:
        payload.update(prediction_target=prediction_target, residual_version=1,
                       residual_history="latest qualifying archived forecast", forecast_lead_input=True,
                       arome_run=case.get("arome_run"),
                       residual_source_runs=batch.arome_history_run.astype(str).tolist())
        arrays["arome_baseline"] = batch.arome
        for j, name in enumerate(variables):
            times = batch.history_times()[0, np.isfinite(ctx[0, j])]
            error_history[name] = {"observed_steps": len(times), "total_steps": ctx.shape[-1],
                                   "first": str(times[0]), "last": str(times[-1])}
    key = predcache.make_key(payload, arrays)

    def compute() -> pd.DataFrame:
        if not offline():
            prefetch(ck, allow_download=allow_download)  # PermissionError if not cached and not allowed
        with _EXPLORE_SEMAPHORE:
            STATS["inference"] += 1
            fb = model.predict(batch)
        fb.validate(batch)
        frame = fb.to_frame(batch, "exploratory (outside frozen benchmark)")
        if correction:
            frame["arome_baseline"] = batch.arome.ravel()
            frame["predicted_error"] = frame["point"] - frame["arome_baseline"]
        return frame

    settings = {"checkpoint": ck.name, "revision": ck.revision, "context_days": context_days,
                "context_steps": int(ctx.shape[-1]), "horizon_steps": horizon_steps, "step_hours": step_hours,
                "calendar": calendar, "arome_mode": arome_mode, "device": device, "features": list(spec),
                "feature_definitions": feats.definition(spec) if spec else [], "feature_inputs": feat_report}
    if correction:
        settings.update(prediction_target=prediction_target, residual_version=1,
                        forecast_lead_input=True, error_history=error_history)
    if lookup_only:
        frame = predcache.lookup(key)
    elif use_cache:
        frame, hit = predcache.get_or_compute(key, payload, compute)
    else:
        frame, hit = compute(), False
    if lookup_only:
        hit = frame is not None
    if frame is not None:
        frame = frame.assign(model=model.id)   # the key fixes the content; the display id follows this request
    return frame, {"cache_hit": hit, "key": key[:16], "model": model.id, "settings": settings}
