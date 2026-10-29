"""Descriptive comparisons of saved point forecasts on identical observed pairs.

Each (case, lead) has equal weight. Windows overlap; spread is descriptive,
never a confidence interval or evidence of independent observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from weather_lab.horizon import by_lead, scored_rows

KEYS = ["case_id", "variable", "lead_h"]
POINT_COLUMNS = (*KEYS, "station", "origin", "valid_time", "model", "point")


def annual_temperature(predictions, targets, cases, station="ALL", split=None, year=2026):
    """Pair-weighted temperature MAE for UTC forecast origins in one year, on common support."""
    origins = pd.to_datetime(cases.origin, utc=True)
    rows, support = matched(predictions, targets, cases[origins.dt.year.eq(year)], "t", station, split)
    table = rows.groupby("model").agg(mae=("abs_err", "mean"), n=("abs_err", "size"),
                                       n_cases=("case_id", "nunique"))
    table = table.reindex(list(predictions))
    table[["n", "n_cases"]] = table[["n", "n_cases"]].fillna(0).astype(int)
    return table.sort_values("mae", kind="stable", na_position="last").reset_index(), support


def matched(predictions: dict[str, pd.DataFrame], targets: pd.DataFrame, cases: pd.DataFrame,
            variable: str, station: str = "ALL", split: str | None = None):
    """Return matched row losses and explicit support/exclusions for one variable.

    `predictions` must include every selected system, even an entirely empty one.
    Include AROME here to restrict a comparison to AROME-supported coverage.
    Missing rows and nonfinite values remain exclusions, never substitutions.
    """
    if not predictions or variable not in ("t", "pmer"):
        raise ValueError("Select saved systems and either temperature or pressure")
    cases = cases.copy()
    cases["origin"] = pd.to_datetime(cases.origin, utc=True)
    if station != "ALL":
        cases = cases[cases.station.eq(station)]
    if split is not None:
        cases = cases[cases.split.eq(split)]
    scheduled = len(cases)
    cases = cases[cases.eligible].copy()
    grid = cases[["case_id", "station", "origin", "split"]].merge(
        pd.DataFrame({"lead_h": range(1, 49)}), how="cross")
    grid["variable"] = variable
    grid["valid_time"] = grid.origin + pd.to_timedelta(grid.lead_h, unit="h")
    truth = targets[targets.variable.eq(variable) & targets.lead_h.between(1, 48)]
    grid = grid.merge(truth[KEYS + ["observed", "valid_time"]], on=KEYS, how="left",
                      suffixes=("", "_target"), validate="one_to_one")
    if (grid.valid_time_target.notna() & grid.valid_time.ne(grid.valid_time_target)).any():
        raise ValueError("Saved target valid time differs from origin + lead")
    grid.loc[~np.isfinite(grid.observed), "observed"] = np.nan
    frames, exclusions = [], []
    for label, frame in predictions.items():
        p = frame[frame.variable.eq(variable) & frame.lead_h.between(1, 48)]
        joined = grid.merge(p[KEYS + ["point", "valid_time"]], on=KEYS, how="left",
                            suffixes=("", "_prediction"), validate="one_to_one")
        if (joined.valid_time_prediction.notna() & joined.valid_time.ne(joined.valid_time_prediction)).any():
            raise ValueError("Saved prediction valid time differs from origin + lead")
        joined.loc[~np.isfinite(joined.point), "point"] = np.nan
        exclusions.append({"model": label, "unavailable_prediction_pairs": int(joined.point.isna().sum()),
                           "available_observed_pairs": int((joined.point.notna() & joined.observed.notna()).sum())})
        joined["model"], joined["output_type"] = label, "point"
        frames.append(joined[list(POINT_COLUMNS) + ["output_type"]])
    points = pd.concat(frames, ignore_index=True)
    rows = scored_rows(points, grid[KEYS + ["observed"]], cases, variable, list(predictions), "mae",
                       station=station, split=split)
    pairs = rows[KEYS].drop_duplicates()
    for record in exclusions:
        record["excluded_by_matching"] = record["available_observed_pairs"] - len(pairs)
    used = cases[cases.case_id.isin(rows.case_id)]
    lead_support = grid.groupby("lead_h").agg(candidate_pairs=("case_id", "size"), observed_target_pairs=("observed", "count"))
    lead_support["matched_pairs"] = pairs.groupby("lead_h").size()
    lead_support = lead_support.reindex(range(1, 49)).fillna(0).astype(int).reset_index()
    support = {"variable": variable, "scheduled_windows": scheduled, "eligible_windows": len(cases),
               "windows": len(used), "observed_pairs": len(pairs), "candidate_pairs": len(grid),
               "missing_target_pairs": int(grid.observed.isna().sum()),
               "excluded_pairs": len(grid) - len(pairs), "stations": sorted(used.station.unique().tolist()),
               "origin_start": used.origin.min(), "origin_end": used.origin.max(),
               "candidate_origin_start": cases.origin.min(), "candidate_origin_end": cases.origin.max(),
               "valid_start": rows.valid_time.min(), "valid_end": rows.valid_time.max(),
               "exclusions": exclusions, "lead_support": lead_support.to_dict("records")}
    return rows, support


def horizon(rows: pd.DataFrame, cumulative: bool = False) -> pd.DataFrame:
    """MAE at h, or pair-weighted MAE over leads 1…h; population SD only at h.

    At a fixed lead each case contributes one absolute error. SD uses ddof=0.
    Cumulative means use all available matched pairs, not means of window means.
    """
    columns = ["model", "lead_h", "mae", "n", "n_cases", "ci95_low", "ci95_high", "n_blocks", "spread"]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    table = by_lead(rows, "mae", cumulative=cumulative, intervals=False)
    index = pd.MultiIndex.from_product([rows.model.unique(), range(1, 49)], names=["model", "lead_h"])
    table = table.set_index(["model", "lead_h"]).reindex(index)
    if cumulative:
        table = table.groupby(level="model", sort=False).ffill()
    table[["n", "n_cases", "n_blocks"]] = table[["n", "n_cases", "n_blocks"]].fillna(0).astype(int)
    table = table.reset_index()
    table["spread"] = np.nan
    if not cumulative:
        spread = rows.groupby(["model", "lead_h"]).abs_err.std(ddof=0).rename("spread")
        table = table.drop(columns="spread").join(spread, on=["model", "lead_h"])
    return table


def arome_window_mae(predictions, targets, cases, variable, station="ALL", split=None):
    """One full 48-lead MAE per saved station/origin, using AROME's own observed support.

    Partial windows retain their pair count but never receive a full-window score.
    """
    rows, support = matched({"arome-raw": predictions}, targets, cases, variable, station, split)
    windows = cases[cases.eligible].copy()
    if station != "ALL":
        windows = windows[windows.station.eq(station)]
    if split is not None:
        windows = windows[windows.split.eq(split)]
    windows["origin"] = pd.to_datetime(windows.origin, utc=True)
    scores = rows.groupby("case_id").agg(mae=("abs_err", "mean"), n=("lead_h", "nunique"))
    table = windows[["case_id", "station", "origin"]].join(scores, on="case_id")
    table["n"] = table.n.fillna(0).astype(int)
    table["complete"] = table.n.eq(48)
    table.loc[~table.complete, "mae"] = np.nan
    support["complete_windows"] = int(table.complete.sum())
    support["incomplete_windows"] = int((~table.complete).sum())
    return table.sort_values(["origin", "station"]), support


def one_forecast(predictions, targets, cases, case_id, variable, reference=None):
    """Full saved curves with gaps, but errors only on identical observed pairs.

    An optional AROME reference outside `predictions` is drawing context only.
    Its issued-run metadata must identify one run available at the origin.
    """
    case = cases[cases.case_id.eq(case_id)]
    if len(case) != 1:
        raise ValueError("Select one saved forecast origin")
    case = case.iloc[0]
    origin = pd.Timestamp(case.origin)
    selected = {m: p[p.case_id.eq(case_id)] for m, p in predictions.items()}
    errors, support = matched(selected, targets, cases[cases.case_id.eq(case_id)], variable)
    grid = pd.DataFrame({"case_id": case_id, "variable": variable, "lead_h": range(1, 49)})
    grid["valid_time"] = origin + pd.to_timedelta(grid.lead_h, unit="h")
    truth = targets[targets.case_id.eq(case_id) & targets.variable.eq(variable)]
    obs = grid.merge(truth[KEYS + ["observed"]], on=KEYS, how="left", validate="one_to_one")
    obs = obs.rename(columns={"valid_time": "time"})[["time", "observed"]]
    obs.loc[~np.isfinite(obs.observed), "observed"] = np.nan
    frames = []
    if reference is not None:
        ref = reference[reference.case_id.eq(case_id) & reference.variable.eq(variable)].copy()
        has = np.isfinite(ref.point)
        for column in ("arome_run", "arome_available_at"):
            ref[column] = pd.to_datetime(ref[column], utc=True)
            if has.any() and (ref.loc[has, column].isna().any() or ref.loc[has, column].nunique() != 1):
                raise ValueError("Saved AROME reference lacks a single issued-run provenance")
        if has.any() and ((ref.loc[has, "arome_available_at"] > origin).any()
                          or (ref.loc[has, "arome_run"] > ref.loc[has, "arome_available_at"]).any()):
            raise ValueError("Saved AROME run was unavailable at the forecast origin")
        ref["arome_lead_h"] = (ref.valid_time - ref.arome_run) / pd.Timedelta(hours=1)
        if has.any() and (ref.loc[has, "arome_lead_h"] <= 0).any():
            raise ValueError("Saved AROME value is not a forecast after initialization")
        selected["arome-raw"] = ref
    for model, frame in selected.items():
        p = frame[frame.variable.eq(variable) & frame.lead_h.between(1, 48)]
        extra = [c for c in ("arome_run", "arome_available_at", "arome_lead_h") if c in p]
        curve = grid.merge(p[KEYS + ["point", *extra]], on=KEYS, how="left", validate="one_to_one")
        curve.loc[~np.isfinite(curve.point), "point"] = np.nan
        curve["model"], curve["output_type"] = model, "point"
        curve["station"], curve["origin"] = case.station, origin
        frames.append(curve)
    return pd.concat(frames, ignore_index=True), obs, case, errors, support


def calendar(rows, clock="valid_time", lead=None, start=None, end=None):
    """Daily/monthly/hourly pair-weighted MAE and population SD of absolute errors.

    Each matched (case, lead) pair has equal weight, including repeated verifying
    observations from overlapping windows. Monthly means use pairs, not daily
    means. SD (ddof=0) describes these absolute errors, never uncertainty of a mean.
    Buckets are UTC; empty dates/hours retain zero counts and missing statistics.
    """
    if clock not in ("origin", "valid_time") or (lead is not None and lead not in range(1, 49)):
        raise ValueError("Choose origin or valid_time, and one lead in 1–48 or all leads")
    if rows.variable.nunique() > 1:
        raise ValueError("Calendar errors must use one variable")
    models = rows.model.unique()
    selected = rows if lead is None else rows[rows.lead_h.eq(lead)]
    dates = pd.to_datetime(selected[clock], utc=True)
    start = pd.Timestamp(start) if start is not None else dates.min()
    end = pd.Timestamp(end) if end is not None else dates.max()
    out = {}
    for period in ("day", "month", "hour"):
        if period == "day":
            bucket = dates.dt.floor("D")
        elif period == "month":
            bucket = dates.dt.strftime("%Y-%m")
        else:
            bucket = dates.dt.hour
        grouped = selected.assign(bucket=bucket).groupby(["model", "bucket"])
        table = grouped.agg(mae=("abs_err", "mean"), n=("abs_err", "size"), n_cases=("case_id", "nunique"))
        table["spread"] = grouped.abs_err.std(ddof=0)
        if pd.notna(start) and pd.notna(end):
            if period == "day":
                bins = pd.date_range(start.floor("D"), end.floor("D"), freq="D")
            elif period == "month":
                bins = pd.date_range(start.floor("D").replace(day=1), end.floor("D").replace(day=1), freq="MS").strftime("%Y-%m")
            else:
                bins = range(24)
            table = table.reindex(pd.MultiIndex.from_product([models, bins], names=["model", "bucket"]))
        table[["n", "n_cases"]] = table[["n", "n_cases"]].fillna(0).astype(int)
        out[period] = table.reset_index()
    return out
