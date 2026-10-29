"""Shared eligibility uses issued inputs, never realised future errors."""
import numpy as np
import pandas as pd

from weather_lab.context_inventory import inventory_cases


def fixture():
    origin = pd.Timestamp("2027-01-08T06:00Z")
    runs = pd.date_range(end=origin - pd.Timedelta(hours=3), periods=93, freq="1D")
    records = [{"station": "07149", "run": run, "available_at": run + pd.Timedelta(hours=2),
                "valid_time": run + pd.Timedelta(hours=lead), "lead_h": lead,
                "variable": var, "value": value}
               for run in runs for lead in range(6, 52, 3) for var, value in (("t", 10.), ("pmer", 1010.))]
    obs = pd.DataFrame({"station": "07149", "time": pd.date_range(
        end=origin + pd.Timedelta(hours=48), periods=752, freq="3h"), "t": 12., "pmer": 1012.})
    cases = pd.DataFrame([{"case_id": "c", "station": "07149", "origin": origin, "split": "validation"}])
    return obs, pd.DataFrame(records), cases


def test_shared_support_and_future_observations_do_not_change_eligibility():
    obs, forecasts, cases = fixture()
    got = inventory_cases(obs, forecasts, cases)
    assert got.eligible.all()
    assert got.observation_t_90d_steps.item() == got.residual_t_90d_steps.item() == 720
    assert got.anchor_t_steps.item() == got.anchor_pmer_steps.item() == 16
    future = obs.time.gt(cases.origin.item())
    obs.loc[future, ["t", "pmer"]] = np.nan
    pd.testing.assert_frame_equal(got, inventory_cases(obs, forecasts, cases))
    # A fully observed short window cannot rescue the same case for the 30-day candidate.
    obs.loc[obs.time.lt(cases.origin.item() - pd.Timedelta(days=30)), "t"] = np.nan
    bad = inventory_cases(obs, forecasts, cases)
    assert bad.observation_t_30d_fraction.item() == 1.
    assert not bad.eligible.item()
    assert "residual_t_90d_below_80pct" in bad.exclusion_reason.item()


def test_delayed_observations_and_unissued_or_missing_anchors_are_unavailable():
    obs, forecasts, cases = fixture()
    delayed = obs.assign(available_at=obs.time + pd.Timedelta(days=100))
    got = inventory_cases(delayed, forecasts, cases)
    assert not got.eligible.item() and got.residual_t_90d_steps.item() == 0
    selected = forecasts.run.eq(cases.origin.item() - pd.Timedelta(hours=3))
    # A late field makes the whole selected run unavailable under the vintage rule.
    late = forecasts.copy()
    late.loc[selected & late.variable.eq("pmer"), "available_at"] = cases.origin.item() + pd.Timedelta(hours=1)
    got = inventory_cases(obs, late, cases)
    assert got.anchor_t_steps.item() == got.anchor_pmer_steps.item() == 0
    missing = forecasts[~(selected & forecasts.lead_h.eq(51))]
    got = inventory_cases(obs, missing, cases)
    assert not got.eligible.item() and got.anchor_t_steps.item() == 15
    # Planned future cases are retained even when none of their inputs exists yet.
    future = cases.assign(origin=cases.origin + pd.Timedelta(days=200))
    got = inventory_cases(obs, forecasts, future)
    assert len(got) == 1 and not got.eligible.item() and not got.history_assessed.item()
