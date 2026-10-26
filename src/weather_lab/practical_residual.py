"""Causal fixed-run-lead residual histories; no interpolation across issuance slots."""
from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import gcd

import numpy as np
import pandas as pd


def validate_archive(arome: pd.DataFrame) -> None:
    keys = ["station", "run", "valid_time", "variable"]
    if arome.duplicated(keys).any():
        raise ValueError("Duplicate issued forecast keys")
    for name in ("run", "valid_time", "available_at"):
        if not isinstance(arome[name].dtype, pd.DatetimeTZDtype) or arome[name].isna().any():
            raise ValueError(f"{name} must contain timezone-aware timestamps")
    implied = (arome.valid_time - arome.run).dt.total_seconds() / 3600
    if not np.array_equal(implied, arome.lead_h) or implied.lt(0).any():
        raise ValueError("Fixed run lead does not match init and valid time")
    if arome.available_at.lt(arome.run).any():
        raise ValueError("Forecast available before initialization")


def issuance_lattice(arome: pd.DataFrame) -> dict:
    """Infer cadence from actual run differences, retaining gaps rather than filling values."""
    validate_archive(arome)
    runs = pd.DatetimeIndex(arome.run.unique()).sort_values()
    if len(runs) < 2:
        return {"cadence_hours": None, "runs": len(runs), "status": "insufficient_runs"}
    differences = np.diff(runs.as_unit("s").asi8)
    seconds = reduce(gcd, map(int, differences))
    if seconds % 3600:
        raise ValueError("Unsupported non-hourly issuance lattice")
    expected = int((runs[-1] - runs[0]).total_seconds() / seconds) + 1
    return {"cadence_hours": seconds // 3600, "runs": len(runs),
            "missing_run_slots": expected - len(runs), "status": "observed_lattice",
            "first_run": runs[0].isoformat(), "last_run": runs[-1].isoformat(),
            "run_hours_utc": sorted(set(runs.tz_convert("UTC").hour))}


@dataclass
class FixedLeadHistory:
    rows: pd.DataFrame
    info: dict


def fixed_lead_history(arome: pd.DataFrame, observations: pd.DataFrame, *, station: str,
                       variable: str, origin: pd.Timestamp, selected_run: pd.Timestamp,
                       target_valid: pd.Timestamp, context_days: int = 30,
                       observation_latency_h: float = 0, min_observed: int = 24) -> FixedLeadHistory:
    """Build r^L(v)=obs(v)-A(init=v-L,valid=v), on its actual issuance lattice.

    Observations may include later outcomes: availability is filtered before residual arithmetic.
    `available_at` on observations, when supplied, replaces the replay latency assumption.
    """
    lattice = issuance_lattice(arome)
    origin, selected_run, target_valid = map(pd.Timestamp, (origin, selected_run, target_valid))
    if any(x.tzinfo is None for x in (origin, selected_run, target_valid)):
        raise ValueError("Origin, selected run and valid time must be timezone-aware")
    if selected_run > origin or target_valid <= origin or context_days <= 0 or observation_latency_h < 0:
        raise ValueError("Invalid causal origin, target, context or observation latency")
    lead = (target_valid - selected_run).total_seconds() / 3600
    if lead != int(lead) or lead <= 0:
        raise ValueError("Requested run lead must be a positive integer hour")
    source = arome[arome.station.eq(station) & arome.variable.eq(variable)]
    anchor = source[source.run.eq(selected_run) & source.valid_time.eq(target_valid)
                    & source.available_at.le(origin)]
    anchor_value = float(anchor.value.iloc[0]) if len(anchor) == 1 else np.nan
    info = {**lattice, "station": station, "variable": variable, "origin": origin.isoformat(),
            "selected_run": selected_run.isoformat(), "target_valid": target_valid.isoformat(),
            "fixed_run_lead_h": int(lead), "context_days": context_days,
            "observation_latency_h": observation_latency_h, "anchor": anchor_value,
            "min_observed": min_observed}
    cadence = lattice["cadence_hours"]
    if cadence is None:
        return FixedLeadHistory(pd.DataFrame(), {**info, "observed_slots": 0, "expected_slots": 0,
                                                 "observed_fraction": 0., "ready": False,
                                                 "policy": "unavailable_insufficient_issuance_history",
                                                 "fallback_model": "arome-raw" if np.isfinite(anchor_value) else None})
    step = pd.Timedelta(hours=cadence)
    # Anchor the history phase to the requested valid time, not to arbitrary observation timestamps.
    end = target_valid - int(np.ceil((target_valid - origin) / step)) * step
    times = pd.date_range(end=end, periods=int(np.ceil(context_days * 24 / cadence)), freq=step)
    rows = pd.DataFrame({"valid_time": times})
    rows["run"] = rows.valid_time - pd.Timedelta(hours=lead)
    rows = rows.merge(source[source.lead_h.eq(lead)][["run", "valid_time", "value", "available_at"]],
                      on=["run", "valid_time"], how="left", validate="one_to_one")
    obs = observations[observations.station.eq(station)].copy()
    if obs.duplicated(["station", "time"]).any():
        raise ValueError("Duplicate station observations")
    obs["observation_available_at"] = (obs.available_at if "available_at" in obs else
                                         obs.time + pd.Timedelta(hours=observation_latency_h))
    if obs.observation_available_at.lt(obs.time).any():
        raise ValueError("Observation available before its valid time")
    obs = obs[obs.time.le(origin) & obs.observation_available_at.le(origin)]
    obs = obs[["time", variable, "observation_available_at"]].rename(columns={"time": "valid_time", variable: "observation"})
    rows = rows.merge(obs, on="valid_time", how="left", validate="one_to_one")
    rows["forecast_available"] = (rows.available_at.le(origin) & rows.available_at.le(rows.valid_time)
                                   & np.isfinite(rows.value))
    rows["observation_available"] = np.isfinite(rows.observation)
    rows["observed"] = rows.forecast_available & rows.observation_available
    rows["residual"] = (rows.observation - rows.value).where(rows.observed)
    rows["missing_reason"] = np.select([rows.value.isna(), ~rows.forecast_available,
                                         ~rows.observation_available],
                                        ["missing_issued_forecast", "forecast_not_causally_available",
                                         "missing_or_delayed_observation"], default="observed")
    count, expected = int(rows.observed.sum()), len(rows)
    if not 1 <= min_observed <= expected:
        raise ValueError("Minimum observed support must lie within context slots")
    ready = count >= min_observed and np.isfinite(anchor_value)
    policy = ("fixed_lead_ready" if ready else "unavailable_selected_run_anchor" if not np.isfinite(anchor_value)
              else "unavailable_insufficient_fixed_lead_support")
    forecast_steps = int((target_valid - end) / step)
    return FixedLeadHistory(rows, {**info, "observed_slots": count, "expected_slots": expected,
                                  "observed_fraction": count / expected, "ready": ready,
                                  "forecast_steps": forecast_steps, "policy": policy,
                                  "fallback_model": None if ready or not np.isfinite(anchor_value) else "arome-raw"})


def restore_residual_quantiles(residual_quantiles: np.ndarray, anchor: float) -> np.ndarray:
    """Restore the same issued anchor to every quantile; unavailable anchors remain unavailable."""
    quantiles = np.asarray(residual_quantiles, dtype=float)
    if quantiles.ndim == 0 or (np.diff(quantiles, axis=-1) < 0).any():
        raise ValueError("Residual quantiles must be ordered on the last dimension")
    return quantiles + anchor
