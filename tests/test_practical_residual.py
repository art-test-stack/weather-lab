import numpy as np
import pandas as pd
import pytest

from weather_lab.practical_residual import fixed_lead_history, issuance_lattice, restore_residual_quantiles


def fixture():
    runs = pd.date_range("2026-01-01T03:00Z", periods=36, freq="1D")
    arome = pd.DataFrame({"station": "07149", "run": runs, "valid_time": runs + pd.Timedelta(hours=6),
                          "lead_h": 6, "available_at": runs + pd.Timedelta(hours=2),
                          "variable": "t", "value": 10.})
    obs = pd.DataFrame({"station": "07149", "time": arome.valid_time, "t": 12.})
    return arome, obs, dict(station="07149", variable="t", origin=pd.Timestamp("2026-02-05T06:00Z"),
                            selected_run=pd.Timestamp("2026-02-05T03:00Z"),
                            target_valid=pd.Timestamp("2026-02-05T09:00Z"))


def test_fixed_lead_uses_daily_issuance_and_masks_missing_runs_without_interpolation():
    arome, obs, args = fixture()
    history = fixed_lead_history(arome, obs, **args)
    assert history.info["cadence_hours"] == 24 and history.info["expected_slots"] == 30
    assert history.info["ready"] and history.info["forecast_steps"] == 1
    assert np.array_equal(history.rows.residual, np.full(30, 2.))
    assert (history.rows.valid_time - history.rows.run).eq(pd.Timedelta(hours=6)).all()
    missing = arome[arome.run.ne(pd.Timestamp("2026-01-20T03:00Z"))]
    history = fixed_lead_history(missing, obs, min_observed=30, **args)
    assert history.info["observed_slots"] == 29 and not history.info["ready"]
    assert history.info["fallback_model"] == "arome-raw"
    assert history.rows.residual.isna().sum() == 1
    assert issuance_lattice(missing)["missing_run_slots"] == 1


def test_causal_observation_and_forecast_availability_and_target_changes():
    arome, obs, args = fixture()
    baseline = fixed_lead_history(arome, obs, **args)
    changed = obs.copy()
    changed.loc[changed.time.gt(args["origin"]), "t"] = -999.
    pd.testing.assert_frame_equal(baseline.rows, fixed_lead_history(arome, changed, **args).rows)
    delayed = fixed_lead_history(arome, obs, observation_latency_h=24, **args)
    assert delayed.info["observed_slots"] == 29
    late = obs.assign(available_at=obs.time + pd.Timedelta(days=100))
    assert fixed_lead_history(arome, late, **args).info["observed_slots"] == 0
    late_forecasts = arome.assign(available_at=arome.valid_time + pd.Timedelta(hours=1))
    got = fixed_lead_history(late_forecasts, obs, **args)
    assert got.info["observed_slots"] == 0 and not got.info["ready"]


def test_actual_run_lead_is_not_requested_origin_lead_and_anchor_is_required():
    arome, obs, args = fixture()
    # origin lead 3h needs actual run lead 6h, so a 9h archive cannot be substituted.
    wrong = arome.assign(lead_h=9, valid_time=arome.run + pd.Timedelta(hours=9))
    got = fixed_lead_history(wrong, obs, **args)
    assert got.info["fixed_run_lead_h"] == 6 and not got.info["ready"]
    assert got.info["policy"] == "unavailable_selected_run_anchor" and got.info["fallback_model"] is None
    with pytest.raises(ValueError, match="run lead"):
        fixed_lead_history(arome.assign(lead_h=9), obs, **args)
    with pytest.raises(ValueError, match="Duplicate"):
        fixed_lead_history(pd.concat([arome, arome.iloc[:1]]), obs, **args)


def test_quantile_restore_preserves_celsius_and_signed_wind_without_clipping():
    residual = np.array([[-4., -2., 1.], [-3., 0., 2.]])
    np.testing.assert_array_equal(restore_residual_quantiles(residual, -1.), residual - 1.)
    assert np.isnan(restore_residual_quantiles(residual, np.nan)).all()
    with pytest.raises(ValueError, match="ordered"):
        restore_residual_quantiles(np.array([1., 0.]), 10.)
