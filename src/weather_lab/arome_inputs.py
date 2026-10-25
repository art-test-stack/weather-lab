"""Named AROME inputs, independent of the variables an adapter predicts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AromeInputs:
    fields: tuple[str, ...]
    history: np.ndarray                 # (N, F, T), NaN when no qualifying forecast exists
    future: np.ndarray                  # (N, F, H), from the case's issued run only
    history_run: np.ndarray             # per-field source runs, NaT for missing values
    history_available_at: np.ndarray
    future_run: np.ndarray
    future_available_at: np.ndarray

    def validate(self, origins, history_times, future_times) -> None:
        assert len(set(self.fields)) == len(self.fields), "duplicate AROME input fields"
        assert (history_times <= origins[:, None]).all(), "AROME history extends past the origin"
        assert (future_times > origins[:, None]).all(), "AROME future times must follow the origin"
        for values, runs, available, times, past in (
            (self.history, self.history_run, self.history_available_at, history_times, True),
            (self.future, self.future_run, self.future_available_at, future_times, False),
        ):
            shape = (len(origins), len(self.fields), times.shape[1])
            assert values.shape == runs.shape == available.shape == shape, "named AROME input shape"
            assert not np.isinf(values).any(), "infinite AROME input"
            has = np.isfinite(values)
            valid = np.broadcast_to(times[:, None, :], shape)
            origin = np.broadcast_to(origins[:, None, None], shape)
            assert (~np.isnat(runs[has]) & ~np.isnat(available[has])).all(), "AROME input without provenance"
            assert (runs[has] <= available[has]).all(), "AROME input published before its run"
            assert (available[has] <= origin[has]).all(), "AROME input unavailable at the origin"
            assert (runs[has] < valid[has]).all(), "AROME input is not a forecast ahead of its valid time"
            if past:
                assert (available[has] <= valid[has]).all(), "AROME history published after its valid time"

    def subset(self, sl) -> "AromeInputs":
        return AromeInputs(self.fields, self.history[sl], self.future[sl], self.history_run[sl],
                           self.history_available_at[sl], self.future_run[sl], self.future_available_at[sl])


def build_arome_inputs(frame: pd.DataFrame, cases: pd.DataFrame, origins: np.ndarray,
                       history_times: np.ndarray, future_times: np.ndarray, fields) -> AromeInputs:
    """Align selected station fields without fetching or substituting observations.

    History selects the latest qualifying run separately for each field. Future inputs use only
    the run recorded for the case. Missing or late inputs remain NaN, with no source run.
    """
    fields = tuple(fields)
    if not fields:
        raise ValueError("select at least one AROME input field")
    if len(set(fields)) != len(fields):
        raise ValueError("duplicate AROME input fields")
    values, runs, available = [], [], []
    for times in (history_times, future_times):
        shape = (len(cases), len(fields), times.shape[1])
        values.append(np.full(shape, np.nan, dtype=np.float32))
        runs.append(np.full(shape, np.datetime64("NaT", "ns"), dtype="datetime64[ns]"))
        available.append(np.full(shape, np.datetime64("NaT", "ns"), dtype="datetime64[ns]"))
    data = frame[frame["variable"].isin(fields)].copy()
    for name in ("run", "available_at", "valid_time"):
        data[name] = pd.to_datetime(data[name], utc=True)
    if data.duplicated(["station", "run", "valid_time", "variable"]).any():
        raise ValueError("duplicate AROME station/run/valid-time/field records")
    if np.isinf(data["value"]).any():
        raise ValueError("infinite AROME input")
    data = data.dropna(subset=["value"])
    for i, case in enumerate(cases.itertuples()):
        origin = pd.Timestamp(origins[i], tz="UTC")
        sub = data[(data["station"] == case.station) & (data["available_at"] <= origin)]
        past = sub[(sub["available_at"] <= sub["valid_time"]) & (sub["run"] < sub["valid_time"])]
        past = past.sort_values(["run", "available_at"]).drop_duplicates(["variable", "valid_time"], keep="last")
        issued = getattr(case, "arome_run", None)
        future = sub[sub["run"] == pd.Timestamp(issued)] if pd.notna(issued) else sub.iloc[:0]
        for k, (rows, times) in enumerate(((past, history_times), (future, future_times))):
            index = pd.MultiIndex.from_product([pd.to_datetime(times[i], utc=True), fields],
                                               names=["valid_time", "variable"])
            aligned = rows.set_index(["valid_time", "variable"]).reindex(index)
            values[k][i] = aligned["value"].to_numpy(dtype=np.float32).reshape(-1, len(fields)).T
            runs[k][i] = aligned["run"].to_numpy(dtype="datetime64[ns]").reshape(-1, len(fields)).T
            available[k][i] = aligned["available_at"].to_numpy(dtype="datetime64[ns]").reshape(-1, len(fields)).T
    inputs = AromeInputs(fields, values[0], values[1], runs[0], available[0], runs[1], available[1])
    inputs.validate(origins, history_times, future_times)
    return inputs
