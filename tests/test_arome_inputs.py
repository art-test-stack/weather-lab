"""Named AROME input selection and vintage checks; no network or model weights."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from weather_lab.arome_inputs import build_arome_inputs, choose_arome
from weather_lab.forecast import CaseBatch
from weather_lab.rag_archive import issued_forecasts


def _inputs():
    origin = np.array([np.datetime64("2026-10-12T06:00", "ns")])
    history = origin[:, None] + np.array([-3, 0]) * np.timedelta64(1, "h")
    future = origin[:, None] + np.array([3, 6]) * np.timedelta64(1, "h")
    cases = pd.DataFrame([{"case_id": "c", "station": "07149", "arome_run": "2026-10-12T03:00Z"}])
    rows = [
        ("10", "12T03", "t", 8., "10T05:45"),
        ("10", "12T03", "rh", 40., "10T05:45"),
        ("11", "12T03", "t", 9., "11T05:45"),
        ("12", "12T06", "t", 11., "12T05:45"),
        ("12", "12T09", "t", 12., "12T05:45"),
        ("12", "12T09", "rh", 99., "12T07:00"),  # published after the origin
    ]
    frame = pd.DataFrame([{"station": "07149", "run": f"2026-10-{day}T03:00Z",
                           "valid_time": f"2026-10-{valid}Z", "variable": field, "value": value,
                           "available_at": f"2026-10-{available}Z"}
                          for day, valid, field, value, available in rows])
    inputs = build_arome_inputs(frame, cases, origin, history, future, ["rh", "t", "cloud"])
    return inputs, cases, origin, history, future


def test_selection_is_independent_of_targets_and_preserves_legacy_arrays():
    inputs, cases, origin, _, future = _inputs()
    legacy_future = np.full((1, 1, 2), 20., dtype=np.float32)
    original = legacy_future.tobytes()
    batch = CaseBatch(cases, ["t"], np.full((1, 1, 2), 10., dtype=np.float32), origin, future, 3,
                      arome=legacy_future, arome_inputs=None)
    with_inputs = replace(batch, arome_inputs=inputs)
    assert with_inputs.arome_inputs.fields == ("rh", "t", "cloud")
    assert with_inputs.arome is legacy_future and with_inputs.arome.tobytes() == original
    subset = with_inputs.subset(slice(0, 1))
    np.testing.assert_array_equal(subset.arome_inputs.future, inputs.future)
    assert subset.variables == ["t"]


def test_history_uses_per_field_vintages_and_late_or_missing_fields_remain_missing():
    inputs, _, origin, history, future = _inputs()
    assert inputs.history[0, 0, 0] == 40. and inputs.history[0, 1, 0] == 9.
    assert inputs.history_run[0, 0, 0] == np.datetime64("2026-10-10T03:00")
    assert inputs.history_run[0, 1, 0] == np.datetime64("2026-10-11T03:00")
    assert np.isnan(inputs.history[:, 2]).all() and np.isnan(inputs.future[:, 0]).all()
    assert np.isnat(inputs.future_run[:, 0]).all() and inputs.future[0, 1, 0] == 12.
    late = inputs.future_available_at.copy()
    late[0, 1, 0] = np.datetime64("2026-10-12T07:00")
    with pytest.raises(AssertionError, match="unavailable at the origin"):
        replace(inputs, future_available_at=late).validate(origin, history, future)


@pytest.mark.parametrize("fields", [[], ["t", "t"]])
def test_empty_or_duplicate_named_selections_are_rejected(fields):
    _, cases, origin, history, future = _inputs()
    with pytest.raises(ValueError):
        build_arome_inputs(pd.DataFrame(), cases, origin, history, future, fields)


def test_issued_runs_cannot_be_stitched_or_mixed():
    origin = pd.Timestamp("2026-06-01T06:00Z")
    times = pd.date_range(origin + pd.Timedelta(hours=1), periods=2, freq="h")
    rows = []
    for var in ("t", "pmer"):
        for i, valid in enumerate(times):
            rows.append({"station": "07110", "run": origin - pd.Timedelta(hours=i + 3),
                         "available_at": origin, "valid_time": valid, "variable": var, "value": 1.,
                         "product": "meteofrance_arome_france", "availability_basis": "test_evidence",
                         "source_kind": "issued_run"})
    issued = issued_forecasts(pd.DataFrame(rows))
    with pytest.raises(ValueError, match="single available issued run"):
        choose_arome(issued, "07110", origin, times, ["t", "pmer"], "meteofrance_arome_france")
    issued["source_kind"] = "stitched_historical_forecast"
    with pytest.raises(ValueError, match="stitched"):
        issued_forecasts(issued)
    issued["source_kind"], issued["run"] = "issued_run", origin - pd.Timedelta(hours=3)
    issued["available_at"] = origin + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="single available issued run"):
        choose_arome(issued, "07110", origin, times, ["t", "pmer"], "meteofrance_arome_france")
