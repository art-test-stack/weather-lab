import numpy as np
import pandas as pd
import pytest

from weather_lab.practical import causal_errors, median_adjustment, nudge, point_blend
from weather_lab.forecast import qcol


def archive():
    origin = pd.Timestamp("2026-10-16T06:00Z")
    rows, observations = [], []
    for day in (12, 13, 14):
        run = pd.Timestamp(f"2026-10-{day}T03:00Z")
        valid = run + pd.Timedelta(hours=6)
        for station in ("07110", "07149", "07190", "07630", "07690"):
            rows.append(dict(station=station, run=run, valid_time=valid, lead_h=6,
                             available_at=run + pd.Timedelta(hours=2), variable="t", value=10.))
            observations.append(dict(station=station, time=valid, t=12. if station != "07149" else 14.))
    return origin, pd.DataFrame(rows), pd.DataFrame(observations)


def test_median_pooling_causal_support_and_run_lead_alignment():
    origin, arome, past = archive()
    errors = causal_errors(arome, past, origin)
    run = origin.normalize() + pd.Timedelta(hours=3)
    got = median_adjustment(errors, "07149", run, run + pd.Timedelta(hours=6), "t", origin)
    assert got["local_n"] == 3 and got["pooled_n"] == 15
    assert got["correction"] == (3 * 4 + 5 * 2) / 8
    # A different run lead must not borrow these errors.
    assert median_adjustment(errors, "07149", run, run + pd.Timedelta(hours=9), "t", origin)["correction"] == 0
    pooled = median_adjustment(errors[errors.station.ne("07149")], "07149", run,
                               run + pd.Timedelta(hours=6), "t", origin)
    assert pooled["policy"] == "pooled_insufficient_local" and pooled["correction"] == 2
    assert median_adjustment(errors.iloc[:9], "07149", run, run + pd.Timedelta(hours=6),
                             "t", origin)["policy"] == "zero_insufficient_pool"
    late = arome.assign(available_at=origin + pd.Timedelta(hours=1))
    assert causal_errors(late, past, origin).empty
    with pytest.raises(ValueError, match="cutoff"):
        causal_errors(arome, past.assign(time=origin + pd.Timedelta(hours=1)), origin)
    with pytest.raises(ValueError, match="run lead"):
        causal_errors(arome.assign(lead_h=9), past, origin)


def test_nudging_uses_same_run_and_elapsed_age_or_explicit_raw_fallback():
    origin, arome, past = archive()
    run = origin.normalize() + pd.Timedelta(hours=3)
    valid = origin + pd.Timedelta(hours=3)
    past = pd.DataFrame([dict(station="07149", time=origin - pd.Timedelta(hours=3), t=14.)])
    source = pd.DataFrame([dict(station="07149", run=run, valid_time=past.time.iloc[0],
                                available_at=origin, variable="t", value=10.)])
    got = nudge(source, past, "07149", run, valid, "t", origin, 6)
    assert got["observation_age_h"] == 3
    assert got["correction"] == pytest.approx(4 * np.exp(-1))
    got = nudge(arome, past, "07149", run, valid, "t", origin, 6)
    assert got["correction"] == 0 and got["policy"] == "raw_no_selected_run_value_at_observation"


def test_blend_is_point_only_and_preserves_unavailable_components():
    raw = pd.DataFrame([dict(case_id="x", variable="t", lead_h=3, point=10., model="arome-raw",
                              output_type="point", **{qcol(.5): 1.})])
    model = raw.assign(point=14.)
    out = point_blend(raw, model, .25)
    assert out.point.iloc[0] == 11. and out.output_type.iloc[0] == "point"
    assert out[qcol(.5)].isna().all()
    assert point_blend(raw, model.assign(point=np.nan), .25).point.isna().all()
