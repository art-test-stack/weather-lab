"""Input-only eligibility for the frozen 30/90-day AROME-error comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd

from weather_lab.arome import archived_history
from weather_lab.practical_residual import validate_archive


def inventory_cases(observations: pd.DataFrame, forecasts: pd.DataFrame,
                    cases: pd.DataFrame) -> pd.DataFrame:
    """Use one 90-day/48-hour eligibility rule for every native context/horizon.

    This matches the app's latest-qualifying-run residual, not the separate fixed-lead
    R2 definition. Missing future *observations* never affect eligibility. No values
    from after an origin are read from observations; issued future forecasts are inputs.
    """
    validate_archive(forecasts)
    if observations.duplicated(["station", "time"]).any():
        raise ValueError("Duplicate station observations")
    if cases.case_id.duplicated().any():
        raise ValueError("Duplicate case ids")
    for field in ("time", "available_at"):
        if field in observations and (not isinstance(observations[field].dtype, pd.DatetimeTZDtype)
                                      or observations[field].isna().any()):
            raise ValueError(f"Observation {field} must be timezone-aware and nonmissing")
    if "available_at" in observations and observations.available_at.lt(observations.time).any():
        raise ValueError("Observation available before validity")
    rows = []
    for case in cases.itertuples():
        origin = pd.Timestamp(case.origin)
        if origin.tzinfo is None:
            raise ValueError("Origin must be timezone-aware")
        origin = origin.tz_convert("UTC")
        if origin != origin.normalize() + pd.Timedelta(hours=6):
            raise ValueError("Origins must be at 06 UTC")
        run = origin - pd.Timedelta(hours=3)
        if hasattr(case, "arome_run") and pd.Timestamp(case.arome_run) != run:
            raise ValueError("Selected run must be the frozen same-day 03 UTC run")
        selected = forecasts[forecasts.station.eq(case.station) & forecasts.run.eq(run)]
        # The vintage rule applies to the complete run, as in archived_history.
        available = forecasts.loc[forecasts.run.eq(run), "available_at"].max()
        future = pd.date_range(origin + pd.Timedelta(hours=3), periods=16, freq="3h")
        anchor_counts = {}
        for var in ("t", "pmer"):
            values = selected[selected.variable.eq(var)].set_index("valid_time").value.reindex(future)
            anchor_counts[var] = int(np.isfinite(values).sum()) if available <= origin else 0
        row = {"case_id": case.case_id, "station": case.station, "origin": origin,
               "split": case.split, "arome_run": run,
               "anchor_t_steps": anchor_counts["t"], "anchor_pmer_steps": anchor_counts["pmer"],
               "history_assessed": not selected.empty}
        reasons = [f"issued_{var}_anchors_{count}/16" for var, count in anchor_counts.items() if count < 16]
        if not selected.empty:
            times = pd.date_range(end=origin, periods=720, freq="3h")
            past = observations[observations.station.eq(case.station) & observations.time.le(origin)]
            if "available_at" in past:
                past = past[past.available_at.le(origin)]
            obs = past.set_index("time").reindex(times)[["t", "pmer"]].to_numpy(dtype=float).T
            values, _, _ = archived_history(
                forecasts, np.array([case.station]), np.array([origin.to_datetime64()], dtype="datetime64[ns]"),
                times.to_numpy(dtype="datetime64[ns]")[None], ["t", "pmer"])
            residual_mask = np.isfinite(obs) & np.isfinite(values[0])
            for j, var in enumerate(("t", "pmer")):
                for days in (30, 90):
                    steps = days * 8
                    for kind, mask in (("observation", np.isfinite(obs)), ("residual", residual_mask)):
                        count = int(mask[j, -steps:].sum())
                        row[f"{kind}_{var}_{days}d_steps"] = count
                        row[f"{kind}_{var}_{days}d_fraction"] = count / steps
                        if count < .8 * steps:
                            reasons.append(f"{kind}_{var}_{days}d_below_80pct")
                if not np.isfinite(obs[j, -8:]).any():
                    reasons.append(f"no_{var}_observation_within_24h")
        else:
            reasons.append("selected_run_missing_history_not_assessed")
        row.update(eligible=not reasons, exclusion_reason=";".join(reasons))
        rows.append(row)
    return pd.DataFrame(rows)
