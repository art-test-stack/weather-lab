"""UTC calendar routing and explicit hourly exploration; no model inference."""
import json
import gradio as gr
import numpy as np
import pandas as pd
import pytest

from weather_lab.common import ROOT, sha256_file
from weather_lab.forecast import QUANTILE_LEVELS, STATS, exploratory_inputs, qcol
from weather_lab.forecast_data import OBSERVATIONS, SUITE, make_case
from weather_lab_app import main as app, results


@pytest.fixture
def hourly(tmp_path, monkeypatch):
    path = tmp_path / "data/processed"
    path.mkdir(parents=True)
    times = pd.date_range("2021-12-01", "2022-01-04", freq="h", tz="UTC").append(
        pd.date_range("2024-12-01", "2025-01-04", freq="h", tz="UTC")).append(
        pd.date_range("2025-12-01", "2026-01-04", freq="h", tz="UTC"))
    obs = pd.DataFrame({"station": "07149", "time": times, "t": 12., "pmer": 1000., "u": 1., "v": 2.,
                        "LAT": 48.7, "LON": 2.4, "ALTI": 90.})
    obs.to_parquet(path / OBSERVATIONS)
    meta = {"version": 1, "exploration_only": True, "start": "2022-01-01T00:00:00Z",
            "sha256": sha256_file(path / OBSERVATIONS), "observation_availability": "Observations assumed known at validity.",
            "stations": [{"id": "07149", "national_id": "91027002", "name": "ORLY", "lat": 48.7, "lon": 2.4, "altitude_m": 90.,
                          "first": str(times.min()), "last": str(times.max())}]}
    issued = []
    for run in pd.to_datetime(["2025-01-01T00:00Z", "2025-01-01T03:00Z"]):
        for variable in ("t", "pmer"):
            for lead in range(1, 52):
                issued.append({"station": "07149", "run": run, "available_at": run + pd.Timedelta(hours=3),
                               "valid_time": run + pd.Timedelta(hours=lead), "variable": variable,
                               "value": 12. if variable == "t" else 1000., "product": "meteofrance_arome_france",
                               "source_kind": "issued_run", "availability_basis": "assumed 3-hour delay"})
    pd.DataFrame(issued).to_parquet(path / "issued.parquet")
    meta["arome"] = {"file": "issued.parquet", "sha256": sha256_file(path / "issued.parquet")}
    (path / OBSERVATIONS).with_suffix(".meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr(results, "ROOT", tmp_path)
    monkeypatch.setattr(app, "ROOT", tmp_path)
    run = results.hourly_exploration()
    monkeypatch.setattr(app, "RUNS", {SUITE: run})
    monkeypatch.setattr(app, "DEFAULT_SUITE", SUITE)
    yield run
    results._parquet.cache_clear()
    results._sha256.cache_clear()
    app._availability.cache_clear()
    app._aux_path.cache_clear()


def values(d):
    return [d[k] for k in app.DRAFT_KEYS]


def test_calendar_utc_hour_bounds_and_frozen_routing(hourly):
    run = hourly
    key = results.calendar_case(run, "07149", "2025-01-01T14:00:00+01:00")
    assert key == f"{SUITE}/07149/20250101T13"
    assert results.calendar_value(run, key) == "2025-01-01 13:00:00"
    for time in ("2021-12-31T23:00Z", "2026-01-04T01:00Z"):
        with pytest.raises(ValueError, match="cover"):
            make_case(run.run["hourly_data"], "07149", time)
    with pytest.raises(ValueError, match="exact UTC hour"):
        make_case(run.run["hourly_data"], "07149", "2025-01-01T13:30Z")
    for time in (None, "not-a-date", "2025-01-01 13:30:00", "2021-01-01 00:00:00"):
        d = {**app.default_draft(SUITE), "case": time}
        cfg, issues = app.draft_config(app.make_draft(*values(d)))
        assert issues and issues[0].kind == "invalid"
    assert app.saved_match(app.default_draft(SUITE), app.draft_config(app.default_draft(SUITE))[0]) is None


def test_hourly_arome_is_fixed_and_missing_early_coverage_is_explicit(hourly):
    case = hourly.case(f"{SUITE}/07149/20250101T03")
    assert case.arome_run == "2025-01-01T00:00:00+00:00"
    d = {**app.default_draft(SUITE), "case": f"{SUITE}/07149/20220101T00", "conditioning": "archived"}
    _, issues = app.draft_config(d)
    assert issues and any(i.kind == "unavailable" and "AROME" in i.message for i in issues)


def test_hourly_calendar_edits_wait_for_run_and_inputs_use_hourly_history(hourly, monkeypatch):
    calls = []
    def predict(obs, case, variables, model, context_days, calendar, **kw):
        batch = exploratory_inputs(obs, case, variables, context_days, None,
                                   horizon_steps=kw["horizon_steps"], step_hours=kw["step_hours"])
        assert batch.history.shape == (1, 4, 720)
        assert batch.history_times().max() <= np.datetime64(case.origin.tz_localize(None))
        calls.append(case.name)
        rows = []
        for variable in variables:
            for lead in range(1, 49):
                rows.append({"case_id": case.name, "station": case.station, "origin": case.origin,
                    "valid_time": case.origin + pd.Timedelta(hours=lead), "lead_h": lead, "variable": variable,
                    "model": "exploratory Timestral-209M ctx30d h48", "model_revision": "fixture", "point": 10.,
                    "output_type": "quantiles", **{qcol(q): 10. + q for q in QUANTILE_LEVELS}})
        return pd.DataFrame(rows), {"cache_hit": True, "key": "fixture", "settings": {
            "checkpoint": model, "context_days": context_days, "context_steps": 720, "step_hours": 1,
            "horizon_steps": 48, "calendar": calendar}}
    monkeypatch.setattr("weather_lab.forecast.exploratory_forecast", predict)
    d = {**app.default_draft(SUITE), "case": "2025-01-01 13:00:00"}
    assert app.initial_applied(SUITE) is None
    assert len(app.render(None)) == 9
    before = dict(STATS)
    with monkeypatch.context() as edits:
        edits.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("Calendar edits must use preloaded metadata"))
        feedback = app.draft_feedback(*values(d), None, {"busy": False})
    assert "No forecast has run" in feedback[0] and feedback[4]["interactive"] and not calls
    assert STATS == before
    result, message = app.run_forecast(*values(d), None)
    assert result is not None, message
    applied, _, changed = app.apply_if_current(result, *values(d), None, message)
    assert changed and calls == [f"{SUITE}/07149/20250101T13"]
    assert applied["source"] == "exploratory" and applied["config"].step_hours == 1
    output = app.render(applied)
    assert "720 steps" in output[3] and "outside the frozen benchmark" in output[3]
    new = {**d, "case": "2025-01-02 13:00:00"}
    feedback = app.draft_feedback(*values(new), applied, {"busy": True})
    assert "Changes pending" in feedback[0] and not feedback[4]["interactive"] and len(calls) == 1
    stale, _, changed = app.apply_if_current(result, *values(new), applied, message)
    assert stale is applied and not changed
    missing = {**d, "case": "2023-06-01 00:00:00"}
    failed, message = app.run_forecast(*values(missing), applied)
    assert failed is None and "too little" in message and len(calls) == 1


@pytest.mark.skipif(not (ROOT / "data/examples/arome-issued-v2/run.json").exists(), reason="saved example not installed")
def test_calendar_widget_change_event_never_applies_or_runs(hourly, monkeypatch):
    frozen = results.load_run(ROOT / "data/examples/arome-issued-v2")
    monkeypatch.setattr(app, "RUNS", {frozen.suite: frozen, SUITE: hourly})
    demo = app.build()
    h = demo.wl
    controls = dict(zip(app.DRAFT_KEYS, h["draft"]))
    date = controls["case"]
    assert isinstance(date, gr.DateTime) and date.timezone == "UTC"
    assert h["applied"].value is None
    assert dict(zip(app.bench.KEYS, h["benchmark"]["draft"]))["suite"].value == frozen.suite
    protected = {h["applied"]._id} | {c._id for c in h["render"]}
    events = [f for f in demo.fns.values() if (date._id, "change") in f.targets]
    assert len(events) == 1 and events[0].fn is app.draft_feedback
    assert not protected & {o._id for o in events[0].outputs}
    key = results.calendar_case(frozen, "07149", results.calendar_value(frozen, app.default_case(frozen.suite, "07149")))
    assert key == app.default_case(frozen.suite, "07149")
