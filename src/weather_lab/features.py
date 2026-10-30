"""Context-enrichment feature groups: a shared pipeline, independent of any model adapter.

A feature group turns data available at a forecast's origin into extra input rows. Each group declares
its version, role, data dependencies, availability rule, missing-value handling and parameters
(defaults + allowed values). `build` produces model-independent arrays (`FeatureInputs`); an adapter
translates them into its native interface (Timestral: past rows -> `past_covariates`, known rows ->
`known_covariates`). Whether a model family can take a role is in `weather_lab.models.FAMILIES`; whether
the data exist for a case is checked by `weather_lab.forecast_config.check`.

How the pinned Timestral (`a534bdb`) takes extra inputs, checked in its source:

- `past_covariates` (B, C, T): known over the context only. They get the same causal per-patch
  normalisation as targets (`TimestralModel.scale`). NaN is masked, and a row with no observed patch
  is never a valid attention key.
- `known_covariates` (B, C, T + H): known over context and horizon, normalised with whole-row
  instance statistics (`scaler.instance_norm_stats`). Rows with values only in {0, 1} are left
  unscaled.
- A constant row normalises to zero (minimum scale), so static station metadata supplied as a
  constant time series carries no information. No group does that.
- A fully missing row is masked but not exactly neutral: any covariate row switches the model to its
  covariate code path (measured on Timestral-5M: quantiles move by up to ~0.008 on a series around 10,
  `tests/test_features.py`). So "feature present but empty" is not identical to "no feature", and
  `forecast_config.check` refuses a selected group whose data are entirely absent for the case.

Groups (version 1 each; bump the version when the construction changes, add a parameter when a
variant should be selectable; default parameters reproduce version 1 exactly):

- `resid`  (A, past): AROME error history, observation − archived AROME, at each past valid time
  v ≤ origin. The forecast comes from the latest run available before both v and the origin
  (`arome.archived_history`), so the residual's observation was available at the origin. A further
  row gives that forecast's lead in hours. Needs archived AROME forecasts even when the issued AROME
  forecast is not used as an input.
- `atmos`  (B, past): observed dew point (°C) and total cloud cover (%) from SYNOP, at history times
  ≤ origin. Future values are never used (no forecast of these fields was collected).
- `solar`  (C, known): sine of the clear-sky solar elevation at the station, from its coordinates and
  each valid time (deterministic astronomy, not an observation).
- `tend`   (C, past): tendencies (lagged differences) of observed t and pmer.

Not implemented: wind components (u and v are already targets), day length (almost constant over a
30-day context), radiation (not in SYNOP) and spatial context (group D: needs a versioned
neighbour/grid extraction).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    version: int
    role: str                       # "past" (context only) | "known" (context + horizon, known in advance)
    label: str                      # readable name
    needs: tuple[str, ...]          # data dependencies: "arome_history", "aux_history", "station_coords",
    #                                 "history:<var>"
    availability: str               # rule that keeps the inputs available at the origin
    missing: str                    # missing-value handling
    description: str
    defaults: dict = field(default_factory=dict)        # parameter -> default (tuples)
    choices: dict = field(default_factory=dict)         # parameter -> allowed values

    def rows(self, params: dict | None = None) -> tuple[str, ...]:
        p = {**self.defaults, **(params or {})}
        if self.name == "resid":
            return tuple(f"resid_{v}" for v in p["variables"]) + (("resid_lead_h",) if p["lead_row"] else ())
        if self.name == "atmos":
            return tuple(p["fields"])
        if self.name == "tend":
            return tuple(f"d{v}_{lag}h" for v in p["variables"] for lag in p["lags_h"])
        return ("sin_solar_elevation",)


GROUPS: dict[str, FeatureGroup] = {g.name: g for g in (
    FeatureGroup("resid", 1, "past", "Recent AROME errors", ("arome_history", "history:t", "history:pmer"),
                 "observation − archived AROME at past valid times ≤ origin; the forecast comes from the latest run "
                 "in the archive before both that valid time and the origin",
                 "NaN where no qualifying archived forecast or observation exists (masked); refused when the "
                 "whole context has none",
                 "AROME error history (obs − archived AROME), with the archived forecast's lead",
                 {"variables": ("t", "pmer"), "lead_row": True},
                 {"variables": ("t", "pmer"), "lead_row": (True, False)}),
    FeatureGroup("atmos", 1, "past", "Dew point and cloud cover", ("aux_history",),
                 "SYNOP observations at history times ≤ origin; no future values",
                 "NaN where not reported (masked; cloud cover 101 = obscured -> missing); refused when absent",
                 "observed dew point and total cloud cover (SYNOP), history only",
                 {"fields": ("td", "cloud")}, {"fields": ("td", "cloud")}),
    FeatureGroup("solar", 1, "known", "Solar elevation", ("station_coords",),
                 "astronomical: known in advance for every valid time",
                 "never missing (computed)",
                 "sine of clear-sky solar elevation at the station (astronomical)"),
    FeatureGroup("tend", 1, "past", "Tendencies", ("history:t", "history:pmer"),
                 "differences of observed history ≤ origin",
                 "NaN where either observation is missing or the lag reaches before the window (masked)",
                 "lagged differences (tendencies) of observed t and pmer",
                 {"variables": ("t", "pmer"), "lags_h": (3, 24)},
                 {"variables": ("t", "pmer"), "lags_h": (3, 6, 12, 24)}),
)}
AUX_VARIABLES = ["td", "cloud"]


@dataclass
class FeatureInputs:
    """Model-independent feature arrays for a batch (adapters translate them)."""

    past: np.ndarray | None             # (N, Cp, T)       context only
    known: np.ndarray | None            # (N, Ck, T + H)   context + horizon
    past_rows: list[str]
    known_rows: list[str]
    report: dict                        # observed share per row over the context, n_cases


def validate(names) -> list[str]:
    unknown = [n for n in names if n not in GROUPS]
    if unknown:
        raise ValueError(f"unknown feature group(s) {unknown}; known: {list(GROUPS)}")
    return [n for n in GROUPS if n in names]


def normalize(spec) -> dict[str, dict]:
    """Feature selection -> {group: complete parameters}, in registry order.

    `spec` is a list of names (default parameters) or {name: partial parameters}. Parameters are
    validated against the group's allowed values; list values become sorted tuples in the allowed order.
    """
    spec = dict.fromkeys(spec, {}) if not isinstance(spec, dict) else spec
    out = {}
    for name in validate(list(spec)):
        g, given = GROUPS[name], dict(spec[name] or {})
        extra = set(given) - set(g.defaults)
        if extra:
            raise ValueError(f"feature '{name}' has no parameter(s) {sorted(extra)}")
        p = {}
        for k, default in g.defaults.items():
            v = given.get(k, default)
            if isinstance(default, tuple):
                v = tuple(x for x in g.choices[k] if x in tuple(v))
                bad = set(given.get(k, ())) - set(g.choices[k]) if k in given else set()
                if bad or not v:
                    what = k.replace("_h", " (h)")
                    raise ValueError(f"feature '{name}': {what} must be a non-empty subset of {list(g.choices[k])}")
            elif v not in g.choices.get(k, (v,)):
                raise ValueError(f"feature '{name}': {k} must be one of {list(g.choices[k])}")
            p[k] = v
        out[name] = p
    return out


def non_default(spec: dict[str, dict]) -> dict[str, dict]:
    """Only the parameters that differ from the defaults (benchmark provenance stays v1-compatible)."""
    return {n: {k: list(v) if isinstance(v, tuple) else v for k, v in p.items() if v != GROUPS[n].defaults[k]}
            for n, p in spec.items() if any(v != GROUPS[n].defaults[k] for k, v in p.items())}


def definition(spec) -> list[dict]:
    """Provenance / cache-key description: name, version, role, rows and complete parameters."""
    spec = normalize(spec)
    return [{"name": n, "version": GROUPS[n].version, "role": GROUPS[n].role, "rows": list(GROUPS[n].rows(p)),
             "params": {k: list(v) if isinstance(v, tuple) else v for k, v in p.items()}} for n, p in spec.items()]


def solar_elevation_sin(times: np.ndarray, lat: float, lon: float) -> np.ndarray:
    """sin(solar elevation) for UTC datetime64 `times` (any shape), NOAA low-precision formulae."""
    t = pd.DatetimeIndex(np.asarray(times).ravel())
    doy = t.dayofyear.to_numpy()
    hour = (t.hour + t.minute / 60).to_numpy()
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g)
                       - 0.040849 * np.sin(2 * g))
    ha = np.deg2rad((hour * 60 + eqtime + 4 * lon) / 4 - 180)
    phi = np.deg2rad(lat)
    out = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    return out.reshape(np.asarray(times).shape).astype(np.float32)


def _diff(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    out[..., k:] = x[..., k:] - x[..., :-k]
    return out


def _block(name: str, p: dict, batch, t_ctx: int) -> np.ndarray:
    step = batch.step_hours
    if name == "resid":
        if batch.arome_inputs is not None:
            fields = [batch.arome_inputs.fields.index(v) for v in p["variables"]]
            archived = batch.arome_inputs.history[:, fields, -t_ctx:]
            runs = batch.arome_inputs.history_run[:, fields, -t_ctx:]
            leads = (batch.history_times(t_ctx)[:, None] - runs) / np.timedelta64(1, "h")
            leads = np.where(np.isfinite(archived), leads, np.nan).astype(np.float32)
            shared_lead = np.fmax.reduce(leads, axis=1)
            if p["lead_row"] and (np.isfinite(leads) & (leads != shared_lead[:, None])).any():
                raise ValueError("resid v1 requires a common archived forecast lead across selected fields")
            rows = [batch.history[:, batch.variables.index(v), -t_ctx:] - archived[:, i]
                    for i, v in enumerate(p["variables"])]
            if p["lead_row"]:
                rows.append(shared_lead)
        else:
            if batch.arome_history is None:
                raise ValueError("feature 'resid' needs archived AROME history in the batch")
            rows = [batch.history[:, batch.variables.index(v), -t_ctx:]
                    - batch.arome_history[:, batch.variables.index(v), -t_ctx:] for v in p["variables"]]
            if p["lead_row"]:
                rows.append(batch.arome_history_lead_h[:, -t_ctx:].astype(np.float32))
        return np.stack(rows, axis=1)
    if name == "atmos":
        if batch.aux_history is None:
            raise ValueError("feature 'atmos' needs auxiliary SYNOP observations in the batch")
        return batch.aux_history[:, [batch.aux_variables.index(v) for v in p["fields"]], -t_ctx:]
    if name == "tend":
        bad = [lag for lag in p["lags_h"] if lag % step]
        if bad:
            raise ValueError(f"feature 'tend': lags {bad} h are not multiples of the {step} h step")
        h = batch.history                               # diffs inside the full history window, then trim
        return np.stack([_diff(h[:, batch.variables.index(v)], lag // step) for v in p["variables"]
                         for lag in p["lags_h"]], axis=1)[..., -t_ctx:]
    if name == "solar":
        if batch.station_coords is None:
            raise ValueError("feature 'solar' needs station coordinates in the batch")
        times = np.concatenate([batch.history_times(t_ctx), batch.future_times], axis=1)
        return np.stack([solar_elevation_sin(times[i], *batch.station_coords[s])
                         for i, s in enumerate(batch.cases["station"])])[:, None]
    raise AssertionError(name)  # pragma: no cover - normalize() guards this


def build_inputs(spec, batch, t_ctx: int) -> FeatureInputs:
    """Feature arrays for `batch` over a context of `t_ctx` steps (any model)."""
    spec = normalize(spec)
    past, known, prow, krow, report = [], [], [], [], {}
    for name, p in spec.items():
        g = GROUPS[name]
        block = _block(name, p, batch, t_ctx).astype(np.float32)
        rows = g.rows(p)
        report.update({r: round(float(np.isfinite(block[..., :t_ctx][:, k]).mean()), 4) for k, r in enumerate(rows)})
        (past if g.role == "past" else known).append(block)
        (prow if g.role == "past" else krow).extend(rows)
    return FeatureInputs(np.concatenate(past, axis=1) if past else None,
                         np.concatenate(known, axis=1) if known else None, prow, krow,
                         {"observed_fraction": report, "n_cases": len(batch.cases)})


def build(spec, batch, t_ctx: int) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """(past rows (N, Cp, t_ctx) | None, known rows (N, Ck, t_ctx + H) | None, report)."""
    f = build_inputs(spec, batch, t_ctx)
    return f.past, f.known, f.report
