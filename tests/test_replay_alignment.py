"""Delayed replay keeps the decision clock and model input clock distinct."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from weather_lab.arome_inputs import build_arome_inputs
from weather_lab.common import Config
from weather_lab.evaluate import make_batch, observation_windows, targets_frame
from weather_lab.features import AUX_VARIABLES
from weather_lab.forecast import AromeRaw, ForecastBatch, QUANTILE_LEVELS
from weather_lab.forecast_config import from_system
from weather_lab.models import CHECKPOINTS
from weather_lab.suite import build_cases, case_data_checksum, observation_delay
from weather_lab.t0_adapter import T0


def fixture(tmp_path, horizon=48, delay=1):
    origin = pd.Timestamp("2026-07-02T06:00Z")
    time = pd.date_range(origin - pd.Timedelta(days=3), origin + pd.Timedelta(hours=48), freq="h")
    obs = pd.DataFrame({"station": "07149", "time": time, "t": np.arange(len(time), dtype=float) / 10})
    cfg = Config({
        "suite": "replay-fixture", "data": {"stations": ["07149"]}, "variables": ["t"],
        "forecast": {"step_hours": 1, "horizon_steps": horizon, "context_days": 1,
                     "observation_delay_hours": delay},
        "origins": {"rule": "explicit", "dates": ["2026-07-02"], "hour_utc": 6, "splits": ["validation"]},
        "splits": {"validation": [str(origin), str(origin + pd.Timedelta(hours=48))]},
        "eligibility": {"variable": "t", "min_context_observed_fraction": .8,
                        "require_observation_within_hours": 2, "windows_days": [1]},
        "arome": {"fields": ["t"]},
    }, tmp_path / "config.yaml")
    cases = build_cases(cfg, obs)
    cases["arome_run"] = origin - pd.Timedelta(hours=3)
    aux = obs[["station", "time"]].copy()
    for name in AUX_VARIABLES:
        aux[name] = obs["t"] + 10
    run = origin - pd.Timedelta(hours=3)
    arome = pd.DataFrame({"station": "07149", "run": run, "available_at": origin,
                          "valid_time": time[time >= origin], "variable": "t", "value": 12.})
    return cfg, obs, cases, aux, arome


@pytest.mark.parametrize("horizon", [24, 48])
def test_shift_generate_bridge_and_realign_every_output(tmp_path, horizon):
    cfg, obs, cases, aux, arome = fixture(tmp_path, horizon)
    batch, targets = make_batch(cfg, obs, cases, 24, arome, aux)
    origin = cases.origin.iloc[0]
    assert batch.horizon == horizon + 1
    assert pd.Timestamp(batch.origins[0], tz="UTC") == origin - pd.Timedelta(hours=1)
    assert pd.Timestamp(batch.future_times[0, 0], tz="UTC") == origin
    assert batch.history.shape == (1, 1, 24)
    latest = obs.loc[obs.time == origin - pd.Timedelta(hours=1), "t"].iloc[0]
    assert batch.history[0, 0, -1] == pytest.approx(latest)
    assert batch.aux_history[0, 0, -1] == pytest.approx(latest + 10)
    # The selected run becomes available at the decision, after the shifted model anchor.
    np.testing.assert_array_equal(batch.arome, 12.)
    named = build_arome_inputs(arome, cases, batch.origins, batch.history_times(), batch.future_times, ["t"])
    batch.arome_inputs = named
    batch.__post_init__()
    with pytest.raises(AssertionError, match="unavailable"):
        named.validate(batch.origins, batch.history_times(), batch.future_times)
    actual = targets_frame(batch, targets)
    point = np.broadcast_to(np.arange(horizon + 1), (1, 1, horizon + 1)).astype(float)
    quantiles = point[..., None] + np.array(QUANTILE_LEVELS) - .5
    forecast = ForecastBatch("fixture", "v1", "quantiles", ["t"], point, quantiles, QUANTILE_LEVELS)
    forecast.validate(batch)
    exported = forecast.to_frame(batch, cfg.suite)
    assert exported.lead_h.tolist() == actual.lead_h.tolist() == list(range(1, horizon + 1))
    assert exported.point.tolist() == list(range(1, horizon + 1))
    assert exported.origin.eq(origin).all()
    assert exported.valid_time.iloc[-1] == origin + pd.Timedelta(hours=horizon)
    np.testing.assert_array_equal(exported["q0.50"], exported.point)
    truth = obs.set_index("time").loc[actual.valid_time, "t"].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(actual.observed, truth)
    assert AromeRaw("a", ["t"], ["t"]).predict(batch).to_frame(batch, cfg.suite).lead_h.min() == 1
    np.testing.assert_array_equal(batch.subset(slice(0, 1)).lead_hours, batch.lead_hours)


def test_unavailable_observations_cannot_change_inputs_or_eligibility(tmp_path):
    cfg, obs, cases, aux, _ = fixture(tmp_path)
    origin = cases.origin.iloc[0]
    batch, _ = make_batch(cfg, obs, cases, 24, aux_df=aux)
    changed, changed_aux = obs.copy(), aux.copy()
    changed.loc[changed.time >= origin, "t"] = np.nan
    changed_aux.loc[changed_aux.time >= origin, AUX_VARIABLES] = 999
    again, _ = make_batch(cfg, changed, cases, 24, aux_df=changed_aux)
    np.testing.assert_array_equal(batch.history, again.history)
    np.testing.assert_array_equal(batch.aux_history, again.aux_history)
    pd.testing.assert_frame_equal(build_cases(cfg, obs), build_cases(cfg, changed))
    # A missing latest usable observation stays missing; it must not move the axis again.
    changed.loc[changed.time == origin - pd.Timedelta(hours=1), "t"] = np.nan
    missing, _ = make_batch(cfg, changed, cases, 24)
    np.testing.assert_array_equal(batch.origins, missing.origins)
    assert np.isnan(missing.history[0, 0, -1])


def test_delayed_context_enters_checksum_saved_window_and_eligibility(tmp_path):
    cfg, obs, cases, _, _ = fixture(tmp_path)
    first = cases.origin.iloc[0] - pd.Timedelta(hours=24)
    windows = observation_windows(obs, cases, 24, cfg)
    assert windows.time.min() == first
    checksum = case_data_checksum(obs, cases, cfg)
    changed = obs.copy()
    changed.loc[changed.time == first, "t"] += 1
    assert case_data_checksum(changed, cases, cfg) != checksum
    changed.loc[(changed.time >= first) & (changed.time < first + pd.Timedelta(hours=5)), "t"] = np.nan
    assert not build_cases(cfg, changed).eligible.iloc[0]
    assert cfg.hash() != Config({**cfg.raw, "forecast": {**cfg["forecast"], "observation_delay_hours": 0}},
                                cfg.path).hash()


@pytest.mark.parametrize("age,eligible", [(2, True), (3, False)])
def test_recent_observation_boundary_uses_actual_age(tmp_path, age, eligible):
    cfg, obs, cases, _, _ = fixture(tmp_path)
    cfg["eligibility"]["min_context_observed_fraction"] = 0
    origin = cases.origin.iloc[0]
    obs.loc[obs.time != origin - pd.Timedelta(hours=age), "t"] = np.nan
    assert bool(build_cases(cfg, obs).eligible.iloc[0]) is eligible


def test_diagnostic_exports_and_errors_exclude_bridge(tmp_path, monkeypatch):
    from weather_lab import diagnostics

    cfg, obs, cases, _, arome = fixture(tmp_path)
    origin = cases.origin.iloc[0]
    cases["arome_available_at"] = origin.strftime("%Y-%m-%dT%H:%M:%SZ")
    past = arome.iloc[:24].copy()
    past["run"] = origin - pd.Timedelta(hours=48)
    past["available_at"] = origin - pd.Timedelta(hours=45)
    past["valid_time"] = pd.date_range(origin - pd.Timedelta(hours=24), periods=24, freq="h")
    arome.loc[arome.valid_time == origin, "value"] = 999.
    arome = pd.concat([past, arome], ignore_index=True)
    batch, targets = make_batch(cfg, obs, cases, 24, arome)
    meta = {"runs": [{"run": r.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "available_at": g.available_at.iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ")}
                     for r, g in arome.groupby("run")]}
    monkeypatch.setattr(diagnostics, "load_suite_batch", lambda *_: (batch, targets, meta))
    audit = diagnostics.audit_table(cfg, cases.case_id.iloc[0])
    future = audit[audit.segment == "future"]
    assert len(future) == 48
    assert future.valid_time.min() == origin + pd.Timedelta(hours=1)
    summary = diagnostics.lead_and_coverage_summary(cfg)
    assert summary["future_lead_h"]["min"] == 4  # actual issued-run lead, after discarded run lead 3
    assert sum(row["size"] for row in summary["t_abs_err_future_by_lead"]) == 48
    assert max(row["mean"] for row in summary["t_abs_err_future_by_lead"]) < 20


def test_zero_delay_matches_legacy_default(tmp_path):
    cfg, obs, cases, aux, arome = fixture(tmp_path, delay=0)
    explicit, targets = make_batch(cfg, obs, cases, 24, arome, aux)
    del cfg.raw["forecast"]["observation_delay_hours"]
    legacy, old_targets = make_batch(cfg, obs, cases, 24, arome, aux)
    for name in ("history", "origins", "future_times", "arome", "arome_history", "aux_history"):
        np.testing.assert_array_equal(getattr(explicit, name), getattr(legacy, name))
    pd.testing.assert_frame_equal(targets_frame(explicit, targets), targets_frame(legacy, old_targets))
    assert explicit.horizon == cfg["forecast"]["horizon_steps"]


def test_issued_cutoff_and_prediction_identity_use_decision_time(tmp_path):
    cfg, obs, cases, _, arome = fixture(tmp_path)
    batch, _ = make_batch(cfg, obs, cases, 24)
    late = arome.assign(available_at=arome.available_at + pd.Timedelta(hours=1))
    with pytest.raises(ValueError, match="not available at origin"):
        make_batch(cfg, obs, cases, 24, late)
    named = build_arome_inputs(late, cases, batch.origins, batch.history_times(), batch.future_times, ["t"])
    assert np.isnan(named.future).all()
    adapter = T0("fixture", ["t"], ["t"], "unused", "unused", context_days=1)
    different_decision = replace(batch, cases=cases.assign(origin=cases.origin + pd.Timedelta(hours=1)))
    assert adapter.input_identity(batch) != adapter.input_identity(different_decision)


def test_delayed_saved_system_cannot_match_zero_delay_explorer():
    checkpoint = CHECKPOINTS["Timestral-209M"]
    record = {"kind": "timestral", "revision": f"{checkpoint.repo}@{checkpoint.revision[:12]}",
              "params": {"checkpoint": checkpoint.repo, "context_days": 30}, "variables": ["t"]}
    assert from_system(record, 48) is not None
    assert from_system(record | {"input_alignment": {"observation_delay_hours": 1}}, 48) is None


def test_benchmark_history_display_uses_recorded_cutoff(tmp_path):
    from weather_lab_app.results import _compute_input_history

    cfg, obs, cases, _, _ = fixture(tmp_path)
    obs.loc[obs.time == cases.origin.iloc[0], "t"] = np.nan
    record = {"id": "timestral-fixture", "params": {"context_days": 1}, "variables": ["t"],
              "input_alignment": {"observation_delay_hours": 1}}
    run = SimpleNamespace(step_hours=1, run={"systems": [record]}, cases=lambda: cases,
                          observations=lambda: obs, arome_extract=lambda: None)
    display = _compute_input_history(run)
    assert display.loc["timestral-fixture", "effective_min"] == 24
    assert display.loc["timestral-fixture", "cases_short"] == 0


@pytest.mark.parametrize("delay", [-1, .5, True, "1", 1])
def test_delay_must_be_nonnegative_and_on_the_data_grid(tmp_path, delay):
    cfg, *_ = fixture(tmp_path)
    cfg["forecast"].update(observation_delay_hours=delay, step_hours=3)
    with pytest.raises(ValueError, match="observation_delay_hours"):
        observation_delay(cfg)
