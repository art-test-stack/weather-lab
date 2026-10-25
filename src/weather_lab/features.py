"""Context-enrichment feature groups for Timestral (inference only; no weights change).

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
  `tests/test_features.py`). So "feature present but empty" is not identical to "no feature".

Groups (each with a version; bump it when its construction changes):

- `resid`  (A, past): AROME error history, observation − archived AROME for t and pmer, at each past
  valid time v ≤ origin. The forecast comes from the latest run available before both v and the
  origin (`arome.archived_history`), so the residual's observation was available at the origin. A
  third row gives that forecast's lead in hours, so differences between leads are explicit.
  Requires archived AROME in the batch.
- `atmos`  (B, past): observed dew point (°C) and total cloud cover (%) from SYNOP, at history times
  ≤ origin. Future values are never used (no forecast of these fields was collected).
- `solar`  (C, known): sine of the clear-sky solar elevation at the station, from its coordinates and
  each valid time (deterministic astronomy, not an observation). It varies within the day and across
  the year, unlike the existing calendar sin/cos.
- `tend`   (C, past): 3 h and 24 h tendencies of observed t and pmer (differences of the target
  history).

Not implemented: wind components (u and v are already targets), day length (almost constant over a
30-day context, so it normalises to a near-linear trend that the time-of-year calendar already
spans) and spatial context (group D: needs a versioned neighbour/grid extraction).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    version: int
    role: str                 # "past" | "known"
    rows: tuple[str, ...]
    needs: tuple[str, ...]    # batch fields required
    description: str


GROUPS: dict[str, FeatureGroup] = {g.name: g for g in (
    FeatureGroup("resid", 1, "past", ("resid_t", "resid_pmer", "resid_lead_h"), ("arome_history",),
                 "AROME error history (obs − archived AROME, latest run available before each valid time)"),
    FeatureGroup("atmos", 1, "past", ("td", "cloud"), ("aux_history",),
                 "observed dew point and total cloud cover (SYNOP), history only"),
    FeatureGroup("solar", 1, "known", ("sin_solar_elevation",), ("station_coords",),
                 "sine of clear-sky solar elevation at the station (astronomical)"),
    FeatureGroup("tend", 1, "past", ("dt_3h", "dt_24h", "dpmer_3h", "dpmer_24h"), (),
                 "3 h and 24 h tendencies of observed t and pmer"),
)}
AUX_VARIABLES = ["td", "cloud"]


def validate(names: list[str] | tuple[str, ...]) -> list[str]:
    unknown = [n for n in names if n not in GROUPS]
    if unknown:
        raise ValueError(f"unknown feature group(s) {unknown}; known: {list(GROUPS)}")
    return list(names)


def definition(names: list[str]) -> list[dict]:
    """Provenance / cache-key description of the groups (name, version, role, rows)."""
    return [{"name": n, "version": GROUPS[n].version, "role": GROUPS[n].role, "rows": list(GROUPS[n].rows)}
            for n in names]


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


def build(names: list[str], batch, t_ctx: int) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """(past rows (N, Cp, t_ctx) | None, known rows (N, Ck, t_ctx + H) | None, report).

    The report gives, per row, the share of observed (non-NaN) values in the context, so missing
    inputs are visible in provenance.
    """
    past, known, report = [], [], {}
    n, step = len(batch.cases), batch.step_hours
    for name in validate(names):
        g = GROUPS[name]
        if name == "resid":
            if batch.arome_history is None:
                raise ValueError("feature 'resid' needs archived AROME history in the batch")
            rows = []
            for var in ("t", "pmer"):
                j = batch.variables.index(var)
                rows.append(batch.history[:, j, -t_ctx:] - batch.arome_history[:, j, -t_ctx:])
            rows.append(batch.arome_history_lead_h[:, -t_ctx:].astype(np.float32))
            block = np.stack(rows, axis=1)
        elif name == "atmos":
            if batch.aux_history is None:
                raise ValueError("feature 'atmos' needs auxiliary SYNOP observations in the batch")
            block = batch.aux_history[:, [batch.aux_variables.index(v) for v in AUX_VARIABLES], -t_ctx:]
        elif name == "tend":
            j_t, j_p = batch.variables.index("t"), batch.variables.index("pmer")
            h = batch.history                               # diffs inside the full history window, then trim
            per_day = 24 // step
            block = np.stack([_diff(h[:, j_t], 1), _diff(h[:, j_t], per_day), _diff(h[:, j_p], 1),
                              _diff(h[:, j_p], per_day)], axis=1)[..., -t_ctx:]
        elif name == "solar":
            if batch.station_coords is None:
                raise ValueError("feature 'solar' needs station coordinates in the batch")
            times = np.concatenate([batch.history_times(t_ctx), batch.future_times], axis=1)
            block = np.stack([solar_elevation_sin(times[i], *batch.station_coords[s])
                              for i, s in enumerate(batch.cases["station"])])[:, None]
        else:  # pragma: no cover - validate() guards this
            raise AssertionError(name)
        block = block.astype(np.float32)
        ctx_part = block[..., :t_ctx]
        report.update({r: round(float(np.isfinite(ctx_part[:, k]).mean()), 4) for k, r in enumerate(g.rows)})
        (past if g.role == "past" else known).append(block)
    return (np.concatenate(past, axis=1) if past else None, np.concatenate(known, axis=1) if known else None,
            {"observed_fraction": report, "n_cases": n})
