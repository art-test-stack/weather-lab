"""Frozen warm-up residual quantiles for a deterministic issued AROME forecast."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from weather_lab.forecast import QUANTILE_LEVELS, qcol

MODEL = "arome-residual-quantiles-warmup-v1"
GROUP = ["variable", "run_cycle_utc", "actual_run_lead_h"]


def _keys(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = forecasts.copy()
    if rows.duplicated(["case_id", "variable", "lead_h"]).any():
        raise ValueError("Duplicate forecast pairs")
    run = pd.to_datetime(rows.arome_run, utc=True)
    available = pd.to_datetime(rows.arome_available_at, utc=True)
    if (run.gt(rows.origin) | available.gt(rows.origin) | available.lt(run)
            | rows.valid_time.le(rows.origin)).any():
        raise ValueError("AROME forecast violates causal issue-time availability")
    lead = (rows.valid_time - run).dt.total_seconds() / 3600
    if not np.equal(lead, np.floor(lead)).all():
        raise ValueError("Noninteger actual AROME run lead")
    requested = (rows.valid_time - rows.origin).dt.total_seconds() / 3600
    if not np.array_equal(requested, rows.lead_h):
        raise ValueError("Requested lead differs from valid time minus origin")
    rows["run_cycle_utc"] = run.dt.hour
    rows["actual_run_lead_h"] = lead.astype(int)
    return rows


@dataclass
class ResidualDistribution:
    groups: pd.DataFrame
    identity: str
    cutoff: pd.Timestamp
    training: pd.DataFrame


def fit_distribution(training: pd.DataFrame, *, warmup_start: pd.Timestamp, warmup_end: pd.Timestamp,
                     cutoff: pd.Timestamp, source_identity: dict, min_support: int = 20) -> ResidualDistribution:
    """Pool station residuals by exact run cycle/lead, fitting once before selection."""
    rows = _keys(training)
    if min_support < 1 or not warmup_start <= warmup_end < cutoff:
        raise ValueError("Invalid chronological fit bounds or support")
    if (~rows.origin.between(warmup_start, warmup_end) | rows.valid_time.ge(cutoff)).any():
        raise ValueError("Training origins or labels cross the purged selection boundary")
    available = pd.to_datetime(rows.observation_available_at, utc=True)
    if available.lt(rows.valid_time).any():
        raise ValueError("Observation availability precedes valid time")
    rows = rows[available.lt(cutoff) & np.isfinite(rows.observed) & np.isfinite(rows.point)].copy()
    rows["residual"] = rows.observed - rows.point
    rows = rows.sort_values(GROUP + ["origin", "station", "case_id"]).reset_index(drop=True)
    records = []
    for key, group in rows.groupby(GROUP, sort=True):
        record = dict(zip(GROUP, key))
        record["support"] = len(group)
        record["station_count"] = group.station.nunique()
        record["quantiles_available"] = len(group) >= min_support
        values = (np.quantile(group.residual, QUANTILE_LEVELS, method="linear")
                  if len(group) >= min_support else np.full(len(QUANTILE_LEVELS), np.nan))
        record.update({qcol(q): float(value) for q, value in zip(QUANTILE_LEVELS, values)})
        records.append(record)
    groups = pd.DataFrame(records, columns=GROUP + ["support", "station_count", "quantiles_available"]
                          + [qcol(q) for q in QUANTILE_LEVELS])
    payload = {"schema": "pooled-warmup-residual-quantiles-v1", "source_identity": source_identity,
               "warmup_start": warmup_start.isoformat(), "warmup_end": warmup_end.isoformat(),
               "cutoff": cutoff.isoformat(), "min_support": min_support, "group": GROUP,
               "quantiles": QUANTILE_LEVELS, "method": "linear",
               "training": rows[["case_id", "station", "origin", "valid_time", "variable", "lead_h",
                                  "run_cycle_utc", "actual_run_lead_h", "point", "observed", "residual"]].to_json(
                                      orient="records", date_format="iso", double_precision=15)}
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return ResidualDistribution(groups, identity, cutoff, rows)


def apply_distribution(forecasts: pd.DataFrame, fit: ResidualDistribution) -> pd.DataFrame:
    rows = _keys(forecasts)
    if rows.origin.lt(fit.cutoff).any():
        raise ValueError("Distribution cannot be scored on warm-up or purged origins")
    qcols = [qcol(q) for q in QUANTILE_LEVELS]
    rows = rows.drop(columns=qcols, errors="ignore").merge(fit.groups, on=GROUP, how="left", validate="many_to_one")
    anchor = rows.point.to_numpy(copy=True)
    rows[qcols] = rows[qcols].to_numpy(dtype=float) + anchor[:, None]
    rows["point"] = rows[qcol(.5)]
    rows["model"] = MODEL
    rows["model_revision"] = fit.identity
    rows["postprocessing_fit_id"] = fit.identity
    rows["output_type"] = "quantiles"
    rows["policy"] = np.where(rows[qcols].notna().all(axis=1), "frozen_pooled_warmup_residual_quantiles",
                               "unavailable_distribution_insufficient_support_or_anchor")
    return rows
