"""Historical AROME covariate: vintage rule, no observation substitution, mode contents (synthetic)."""

import numpy as np
import pandas as pd

from weather_lab.arome import archived_history
from weather_lab.forecast import CaseBatch, Timestral


def _arome(runs):
    rows = []
    for init, avail in runs:
        for lead in range(6, 52, 3):
            vt = init + pd.Timedelta(hours=lead)
            rows.append({"run": init, "available_at": avail, "lead_h": lead, "valid_time": vt, "station": "07149",
                         "variable": "t", "value": float(init.day * 100 + lead)})
    return pd.DataFrame(rows)


def test_archived_history_vintage_and_no_substitution():
    r1 = (pd.Timestamp("2026-10-10T03:00Z"), pd.Timestamp("2026-10-10T05:45Z"))
    r2 = (pd.Timestamp("2026-10-11T03:00Z"), pd.Timestamp("2026-10-11T05:45Z"))
    late = (pd.Timestamp("2026-10-11T15:00Z"), pd.Timestamp("2026-10-12T09:00Z"))   # published after origin
    df = _arome([r1, r2, late])
    origin = np.datetime64("2026-10-12T06:00")
    times = origin + np.arange(-23, 1) * np.timedelta64(3, "h")                    # 2026-10-09T09 .. 10-12T06
    vals, run, lead = archived_history(df, np.array(["07149"]), np.array([origin]), times[None], ["t", "pmer"])
    t = pd.DatetimeIndex(times)
    # before the first run's availability nothing qualifies: missing, never an observation
    before = t < pd.Timestamp("2026-10-10T09:00")
    assert np.isnan(vals[0, 0, before]).all() and np.isnat(run[0, before]).all()
    # a value at 10-11T06 must come from a run available BEFORE 06:00 -> r2 (avail 05:45) has no lead 3,
    # so the 10-10 run at lead 27 is used
    k = int(np.flatnonzero(t == pd.Timestamp("2026-10-11T06:00"))[0])
    assert run[0, k] == np.datetime64("2026-10-10T03:00") and lead[0, k] == 27
    # after r2 becomes available the most recent run wins (smallest lead)
    k = int(np.flatnonzero(t == pd.Timestamp("2026-10-11T09:00"))[0])
    assert run[0, k] == np.datetime64("2026-10-11T03:00") and lead[0, k] == 6
    # the run published after the origin never appears
    assert not (run[0] == np.datetime64("2026-10-11T15:00")).any()
    assert np.isnan(vals[0, 1]).all()                                               # pmer absent -> missing


def test_modes_build_documented_rows():
    t_ctx, h = 16, 4
    origin = np.datetime64("2026-10-12T06:00")
    hist = np.full((1, 2, t_ctx), 10.0, dtype=np.float32)
    hist_times = origin + np.arange(-(t_ctx - 1), 1) * np.timedelta64(3, "h")
    ah = np.full((1, 2, t_ctx), np.nan, dtype=np.float32)
    runs = np.full((1, t_ctx), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    leads = np.full((1, t_ctx), np.nan)
    ah[0, 0, -2:] = [11.0, 12.0]
    runs[0, -2:] = np.datetime64("2026-10-11T03:00")
    leads[0, -2:] = (hist_times[-2:] - np.datetime64("2026-10-11T03:00")) / np.timedelta64(1, "h")
    batch = CaseBatch(pd.DataFrame([{"case_id": "c", "station": "07149"}]), ["t", "pmer"], hist, np.array([origin]),
                      (origin + np.arange(1, h + 1) * np.timedelta64(3, "h"))[None], 3,
                      arome=np.full((1, 2, h), 20.0, dtype=np.float32), arome_history=ah, arome_history_run=runs,
                      arome_history_lead_h=leads)

    def rows(mode):
        m = Timestral("x", ["t", "pmer"], targets=["t", "pmer"], checkpoint="c", revision="r", context_days=2,
                      arome=mode, arome_fields=["t"])
        return m.build_inputs(batch)[1][0, 0]

    archived, missing, subst = rows("archived"), rows("missing"), rows("substituted_observations")
    assert (archived[t_ctx:] == 20).all() and (missing[t_ctx:] == 20).all() and (subst[t_ctx:] == 20).all()
    assert np.isnan(archived[: t_ctx - 2]).all() and list(archived[t_ctx - 2: t_ctx]) == [11, 12]
    assert np.isnan(missing[:t_ctx]).all()
    assert (subst[:t_ctx] == 10).all()          # the documented v1 substitution, only when explicitly requested
