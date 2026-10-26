"""Composable forecast configuration and strictly explicit execution (Run forecast is the only way a forecast
is resolved and applied). Fast tests use the installed October example and a fake exploratory forecaster;
the real pipeline is exercised in the `real_model` test at the end."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from weather_lab import features as feats
from weather_lab.common import ROOT
from weather_lab.forecast import QUANTILE_LEVELS, qcol
from weather_lab.forecast_config import Availability, ForecastConfig, check, from_system

EXAMPLE = ROOT / "data" / "examples" / "arome-issued-v2"
needs_example = pytest.mark.skipif(not (EXAMPLE / "run.json").exists(),
                                   reason="saved example not installed (weather-lab bootstrap)")
ALL_DATA = Availability(arome_issued_h=48, arome_run="2026-10-21T03:00:00Z", arome_history=True, aux=True,
                        station_coords=True)


# --------------------------------------------------------------------------- configuration and capabilities

def test_both_timestral_sizes_accept_the_same_feature_selections():
    groups = list(feats.GROUPS)
    for cond in ("none", "archived", "substobs", "futureonly"):
        for k in range(len(groups) + 1):
            for combo in itertools.combinations(groups, k):            # includes "no additional features"
                small = ForecastConfig.make("Timestral-28M", cond, list(combo), horizon_h=48)
                large = ForecastConfig.make("Timestral-209M", cond, list(combo), horizon_h=48)
                assert check(small, ALL_DATA) == check(large, ALL_DATA) == []
                assert small.features == large.features and small.model != large.model


def test_unsupported_capabilities_are_distinguished_from_missing_data(monkeypatch):
    from weather_lab import models

    cfg = ForecastConfig.make("Timestral-28M", "none", ["resid", "atmos", "solar"], horizon_h=48)
    issues = check(cfg, Availability(arome_issued_h=48, arome_history=False, aux=False, station_coords=True))
    assert {(i.subject, i.kind) for i in issues} == {("resid", "unavailable"), ("atmos", "unavailable")}
    assert "even without AROME conditioning" in next(i.message for i in issues if i.subject == "resid")
    # a family that takes known-future but no past-only covariates (e.g. a future model): explicit, per group
    fam = models.FAMILIES["timestral"]
    monkeypatch.setitem(models.FAMILIES, "timestral",
                        models.Family(**{**fam.__dict__, "past_covariates": False, "name": "NoPast"}))
    issues = check(cfg, ALL_DATA)
    assert {(i.subject, i.kind) for i in issues} == {("resid", "unsupported"), ("atmos", "unsupported")}
    assert all("NoPast takes no past-only covariates" in i.message for i in issues)
    planned = ForecastConfig.make("Timestral-28M", "none", [], horizon_h=48)
    monkeypatch.setitem(models.FAMILIES, "timestral", models.Family(**{**fam.__dict__, "status": "planned"}))
    assert [i.kind for i in check(planned, ALL_DATA)] == ["unsupported"]


def test_no_silent_changes_for_invalid_parameters_and_horizons():
    with pytest.raises(ValueError):
        ForecastConfig.make("Timestral-28M", "none", {"tend": {"lags_h": []}})
    long = ForecastConfig.make("Timestral-28M", "archived", [], horizon_h=72)
    assert [(i.subject, i.kind) for i in check(long, ALL_DATA)] == [("horizon", "unavailable")]
    assert check(ForecastConfig.make("Timestral-28M", "none", [], horizon_h=72), ALL_DATA) == []
    no_run = Availability(arome_issued_h=0, notes={"arome": "no AROME run"})
    assert [(i.subject, i.kind) for i in check(long, no_run)][0] == ("conditioning", "unavailable")


def test_feature_parameters_change_rows_versions_and_keys_but_defaults_match_version_1():
    default = feats.definition(["tend"])[0]
    custom = feats.definition({"tend": {"lags_h": [6, 12]}})[0]
    assert default["rows"] == ["dt_3h", "dt_24h", "dpmer_3h", "dpmer_24h"]           # version 1, unchanged
    assert custom["rows"] == ["dt_6h", "dt_12h", "dpmer_6h", "dpmer_12h"] and custom["version"] == 1
    from weather_lab.forecast import Timestral

    t = Timestral("x", ["t", "pmer"], ["t", "pmer"], "mistralai/Timestral-28M", "r" * 40, features=["tend"])
    assert "feature_params" not in t.params                     # benchmark provenance of defaults unchanged
    t2 = Timestral("x", ["t", "pmer"], ["t", "pmer"], "mistralai/Timestral-28M", "r" * 40,
                   features={"tend": {"lags_h": [6]}})
    assert t2.params["feature_params"] == {"tend": {"lags_h": [6]}}


@needs_example
def test_every_saved_timestral_system_maps_into_the_configuration():
    from weather_lab.common import read_json

    for ex in sorted((ROOT / "data" / "examples").glob("*/run.json")):
        run = read_json(ex)
        for s in run["systems"]:
            cfg = from_system(s, 48)
            if s["kind"] != "timestral":
                assert cfg is None
                continue
            p = s["params"]
            assert cfg is not None, s["id"]
            assert cfg.model.lower().replace("timestral-", "") in s["id"]
            assert cfg.context_days == p["context_days"] and cfg.calendar == p["calendar"]
            assert cfg.feature_names == p.get("features", [])
            assert cfg.targets == tuple(s["variables"])
    v2 = read_json(EXAMPLE / "run.json")
    f = next(s for s in v2["systems"] if s["id"] == "timestral-209m-mv-cal-arome-archived-f-resid")
    assert from_system(f, 48) == ForecastConfig.make("Timestral-209M", "archived", ["resid"], context_days=30,
                                                     horizon_h=48)
    obs_v1 = {**f, "params": {**f["params"], "arome": "observations", "features": []}}
    assert from_system(obs_v1, 48).conditioning == "substobs"          # v1 alias, never "archived"
    other_rev = {**f, "revision": "mistralai/Timestral-209M@000000000000"}
    assert from_system(other_rev, 48) is None                          # another revision is another model


# --------------------------------------------------------------------------- explicit execution (app)

def _frame(model: str, case, horizon_h: int, base: float = 12.0) -> pd.DataFrame:
    leads = list(range(3, horizon_h + 1, 3))
    rows = []
    for var in ("t", "pmer", "u", "v"):
        b = {"t": base, "pmer": 1012.0, "u": 1.0, "v": 1.0}[var]
        for h in leads:
            r = {"case_id": case.name, "station": case["station"], "origin": pd.Timestamp(case["origin"]),
                 "valid_time": pd.Timestamp(case["origin"]) + pd.Timedelta(hours=h), "lead_h": h, "variable": var,
                 "model": model, "model_revision": "x", "output_type": "quantiles", "point": b + 0.01 * h,
                 "unit": "°C"}
            r.update({qcol(q): b + 0.01 * h + (q - 0.5) * 3 for q in QUANTILE_LEVELS})
            rows.append(r)
    df = pd.DataFrame(rows)
    for col in ("origin", "valid_time"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


class FakeForecaster:
    """Stands in for `weather_lab.forecast.exploratory_forecast`: a content-addressed cache + inference count."""

    def __init__(self):
        self.cache, self.calls, self.inference = {}, 0, 0

    def __call__(self, obs, case, variables, size, context_days, calendar, arome_mode=None, arome_df=None,
                 horizon_steps=16, step_hours=3, allow_download=False, device="cpu", use_cache=True,
                 lookup_only=False, features=None, aux_df=None, station_coords=None, prediction_target="weather"):
        self.calls += 1
        key = (case.name, size, context_days, calendar, arome_mode, horizon_steps, repr(features), prediction_target)
        mid = f"exploratory {size} ctx{context_days}d h{horizon_steps * step_hours} {sorted(features or {})}"
        if prediction_target == "arome_error":
            mid += " +AROME error correction"
        hit = key in self.cache
        if not hit:
            if lookup_only:
                return None, {"cache_hit": False, "key": "k", "model": mid, "settings": {}}
            self.inference += 1
            frame = _frame(mid, case, horizon_steps * step_hours)
            if prediction_target == "arome_error":
                frame["arome_baseline"] = frame["point"] - .5
                frame["predicted_error"] = .5
            self.cache[key] = frame[frame["variable"].isin(variables)].copy()
        st = {"checkpoint": size, "context_days": context_days, "context_steps": context_days * 8,
              "horizon_steps": horizon_steps, "step_hours": step_hours, "calendar": calendar,
              "arome_mode": arome_mode, "features": list(features or {}), "prediction_target": prediction_target}
        return self.cache[key].assign(model=mid), {"cache_hit": hit, "key": "k", "model": mid, "settings": st}


@pytest.fixture
def app_env(monkeypatch):
    from weather_lab import forecast, models, predcache
    from weather_lab_app import main as app
    from weather_lab_app.results import load_run

    monkeypatch.setattr(app, "RUNS", {"arome-issued-v2": load_run(EXAMPLE)})
    fake = FakeForecaster()
    monkeypatch.setattr(forecast, "exploratory_forecast", fake)
    monkeypatch.setattr(forecast, "ensure_checkpoint", lambda *a, **k: False)
    monkeypatch.setattr(app, "_exploratory_on_device", lambda obs, case, cfg, kw: fake(
        obs, case, list(cfg.targets), cfg.model, cfg.context_days, cfg.calendar, **kw))

    def counters():
        return (dict(forecast.STATS), dict(models.STATS), dict(predcache.STATS), dict(app.COUNTERS), fake.calls,
                fake.inference)

    a0 = app.initial_applied("arome-issued-v2")
    return app, fake, counters, a0


def _vals(app, d):
    return [d[k] for k in app.DRAFT_KEYS]


def _run(app, d, applied):
    """Run forecast followed by the apply step with the same draft (the click chain)."""
    res, msg = app.run_forecast(*_vals(app, d), applied)
    return app.apply_if_current(res, *_vals(app, d), applied, msg)


EDITS = {"station": "07110", "variable": "pmer", "size": "28M", "conditioning": "none",
         "features": ["resid", "atmos", "solar", "tend"], "calendar": False, "ctx": 45, "horizon": 24,
         "resid_vars": ["t"], "atmos_fields": ["td"], "tend_lags": [6, 12], "overlays": ["persistence"],
         "band": "", "error_mode": "signed", "prediction_target": "arome_error", "metric": "rmse"}


@needs_example
def test_editing_every_control_changes_nothing_but_the_draft_feedback(app_env):
    app, fake, counters, a0 = app_env
    d0 = a0["draft"]
    before, snapshot = counters(), repr(a0)
    d = dict(d0)
    for k, v in EDITS.items():
        d[k] = v
        if k == "station":
            d["case"] = app.default_case("arome-issued-v2", v)
        out = app.draft_feedback(*_vals(app, d), a0, {"busy": False})
        assert len(out) == 6 and "Changes pending" in out[0]
        assert "Timestral 209M + AROME" in out[0]                      # the result summary is still the applied one
    assert counters() == before and repr(a0) == snapshot                # applied state never touched
    # a configuration whose prediction is already cached is still not applied by editing to it
    cached = {**d0, "conditioning": "none", "ctx": 45}
    res, _ = app.run_forecast(*_vals(app, cached), a0)                  # computed + cached, result discarded
    assert res and fake.inference == 1
    mid = counters()
    out = app.draft_feedback(*_vals(app, cached), a0, {"busy": False})
    assert counters() == mid and "Saved benchmark forecast" in out[0] and "Changes pending" in out[0]


@needs_example
def test_edits_never_reach_resolution_inference_loading_or_rendering(app_env):
    app, fake, counters, a0 = app_env
    before = counters()
    d = dict(a0["draft"])
    for k, v in EDITS.items():
        d[k] = v
        app.draft_feedback(*_vals(app, d), a0, {"busy": False})
    assert counters() == before                                         # no lookup, inference, load, download


@needs_example
@pytest.mark.parametrize("computed", [False, True])
def test_main_metric_is_applied_only_on_run_without_new_predictions(app_env, tmp_path, monkeypatch, computed):
    from pathlib import Path

    app, fake, counters, a0 = app_env
    if computed:
        a0, _, changed = _run(app, {**a0["draft"], "conditioning": "none", "ctx": 45}, a0)
        assert changed and fake.inference == 1
    monkeypatch.setattr(app, "ROOT", tmp_path)
    original = app.render(a0)
    original_csv = Path(original[6]).read_bytes()
    before = counters()
    d = {**a0["draft"], "metric": "rmse"}
    feedback = app.draft_feedback(*_vals(app, d), a0, {"busy": False})
    assert "main metric → RMSE" in feedback[0] and "main metric MAE" in feedback[0]
    assert app.render(a0)[4] == original[4] and counters() == before
    result, msg = app.run_forecast(*_vals(app, d), a0)
    stale, _, changed = app.apply_if_current(result, *_vals(app, a0["draft"]), a0, msg)
    assert stale is a0 and not changed
    applied, _, changed = app.apply_if_current(result, *_vals(app, d), a0, msg)
    assert changed and counters()[:3] == before[:3] and fake.calls == before[-2]
    assert applied["config"] == a0["config"] and applied["view"]["metric"] == "rmse"
    output = app.render(applied)
    assert "<th>RMSE</th><th>MAE</th>" in output[4] and "Main metric: RMSE" in output[3]
    score_label = "RMSE · this forecast" if computed else "root mean squared error"
    assert score_label in output[8]
    exported = pd.read_csv(output[6])
    assert set(exported.main_metric) == {"rmse"}
    assert exported.applied_configuration.str.contains("main metric RMSE").all()
    assert output[6] != original[6] and Path(original[6]).read_bytes() == original_csv


@needs_example
def test_only_run_forecast_and_page_load_write_the_applied_state_or_the_display():
    """Event graph: every function writing `applied` or a displayed output descends from the Run button's click
    or from the page load; draft controls trigger only draft feedback."""
    import os

    os.environ["WEATHER_LAB_BOOTSTRAP"] = "off"
    from weather_lab_app import main as app

    demo = app.init(run_bootstrap=False)
    h = demo.wl
    main_metric = dict(zip(app.DRAFT_KEYS, h["draft"]))["metric"]
    assert main_metric.label == h["benchmark_metric"].label == "Main metric"
    assert main_metric.value == h["benchmark_metric"].value == "mae"
    assert {v for _, v in main_metric.choices} == {"mae", "rmse"}
    assert {"mae", "rmse"} <= {v for _, v in h["benchmark_metric"].choices}
    assert any(c.get("props", {}).get("label") == "Forecast" for c in demo.config["components"])
    overview = h["render"]
    protected = {h["applied"]._id} | {c._id for c in overview if c not in h["draft"]}
    fns = demo.fns

    def root(fn):
        while fn.trigger_after is not None:
            fn = fns[fn.trigger_after]
        return fn

    draft_ids = {c._id for c in h["draft"]}
    edit_ids = draft_ids | {c._id for c in h["advanced"]}
    for fn in fns.values():
        r = root(fn)
        writes = {o._id for o in fn.outputs} & protected
        roots = {(t[0], t[1]) for t in r.targets}
        if any(t[0] in edit_ids and t[1] == "input" for t in roots):
            assert fn.show_progress == "hidden", fn.name
        if writes:
            assert roots <= {(h["run"]._id, "click"), (None, "load")} or any(t[1] == "load" for t in roots), fn.name
        if any(t[0] in draft_ids for t in roots) and fn.trigger_after is None:
            assert fn.name in ("draft_feedback", "on_experiment", "on_station", "<lambda>"), fn.name
    run_click = [f for f in fns.values() if (h["run"]._id, "click") in f.targets]
    assert len(run_click) == 1 and run_click[0].trigger_mode == "once"
    assert run_click[0].fn is app.begin_forecast
    assert run_click[0].inputs == h["draft"] + [h["applied"], h["submission"]]
    worker = next(f for f in fns.values() if f.fn is app.run_captured)
    assert worker.trigger_only_on_success and worker.inputs == [h["captured_request"]]
    assert any(f.fn is app.finish_forecast for f in fns.values())
    from weather_lab_app import benchmark as bench

    b = h["benchmark"]
    assert not draft_ids & {c._id for c in b["draft"]}
    protected_b = {b["applied"]._id} | {c._id for c in b["render"]}
    for fn in fns.values():
        roots = {(t[0], t[1]) for t in root(fn).targets}
        if {o._id for o in fn.outputs} & protected_b:
            assert roots <= {(b["run"]._id, "click"), (None, "load")} or any(t[1] == "load" for t in roots), fn.name
        if roots & {(c._id, "input") for c in b["draft"]}:
            assert not ({o._id for o in fn.outputs} & protected_b)
            assert fn.fn in (bench.feedback, bench.on_suite, bench.on_saved)
            assert fn.show_progress == "hidden"
    click = [f for f in fns.values() if (b["run"]._id, "click") in f.targets]
    assert len(click) == 1 and click[0].fn is bench.begin and click[0].trigger_mode == "once"
    worker = next(f for f in fns.values() if f.fn is bench.execute)
    assert worker.trigger_only_on_success and worker.inputs == [b["captured_request"]]


@needs_example
def test_benchmark_owns_its_draft_capture_guard_and_applied_result(app_env, tmp_path, monkeypatch):
    from weather_lab_app import benchmark as bench

    app, _, counters, forecast = app_env
    d = bench.defaults(forecast["suite"])
    previous = {"draft": d, "config": bench.configuration(d), "path": str(app.RUNS[d["suite"]].path), "kind": "frozen"}
    guard = {"busy": False}
    vals = lambda v: [v[k] for k in bench.KEYS]
    before = counters()
    monkeypatch.setattr(bench.development, "saved_runs", lambda suite: [])
    calls = []

    def run(source, cfg, **kw):
        calls.append((source, cfg))
        return tmp_path / "development-run"

    monkeypatch.setattr(bench.development, "run", run)
    for key, value in (("size", "28M"), ("prediction_target", "arome_error"), ("conditioning", "none"),
                       ("features", ["tend"]), ("calendar", False), ("ctx", 90), ("horizon", 24),
                       ("resid_vars", ["t"]), ("atmos_fields", ["td"]), ("tend_lags", [3]),
                       ("variable", "pmer"), ("metric", "rmse"), ("lead", "6"), ("station", "07110"),
                       ("season", "SON"), ("statuses", [app.STATUSES[0]]), ("lead_mode", "cumulative")):
        text, button = bench.feedback(*vals({**d, key: value}), previous, guard)
        assert "pending" in text and button["interactive"]
    assert not calls and counters() == before
    request, _, _ = bench.begin(*vals(d), previous, guard)
    edited = {**d, "metric": "rmse", "features": ["tend"]}
    assert not bench.feedback(*vals(edited), previous, guard)[1]["interactive"]
    with pytest.raises(Exception, match="already running"):
        bench.begin(*vals(edited), previous, guard)
    result, message = bench.execute(request)
    assert len(calls) == 1 and calls[0][1] == previous["config"]
    assert bench.apply(result, *vals(edited), previous, message)[0] is previous
    applied, _, changed = bench.apply(result, *vals(d), previous, message)
    assert changed and applied["kind"] == "development"
    bench.finish(*vals(d), applied, guard)
    assert not guard["busy"] and counters() == before
    display_only = {**d, "metric": "rmse"}
    reused, _ = bench.execute({"draft": display_only, "applied": applied})
    assert reused["path"] == applied["path"] and len(calls) == 1
    bench.execute({"draft": display_only, "applied": reused})
    assert len(calls) == 2  # unchanged settings + Rerun deliberately executes a new full-suite run
    monkeypatch.setattr(bench.development, "run", lambda *a, **kw: (_ for _ in ()).throw(ValueError("fixture failure")))
    failed, message = bench.execute({"draft": d, "applied": previous})
    assert failed is None and "displayed result is unchanged" in message
    assert bench.apply(failed, *vals(d), previous, message) == (previous, message, False)

    # Saved replay fills only the draft, verifies artifacts and never calls the runner.
    from weather_lab.common import sha256_file, write_json

    saved = tmp_path / "saved"
    saved.mkdir()
    artifact = saved / "predictions.parquet"
    artifact.write_bytes(b"fixture artifact")
    write_json(saved / "run.json", {"forecast_config": previous["config"].as_dict(),
                                   "files_sha256": {artifact.name: sha256_file(artifact)}})
    monkeypatch.setattr(bench.development, "saved_runs", lambda suite: [("fixture", str(saved))])
    assert bench.on_saved(str(saved), d["suite"])[:7] == (
        d["size"], d["prediction_target"], d["conditioning"], d["features"], d["calendar"], d["ctx"], d["horizon"])
    replay = {**d, "saved": str(saved)}
    result, message = bench.execute({"draft": replay, "applied": previous})
    assert result["path"] == str(saved) and "Replayed" in message and len(calls) == 2
    failed, message = bench.execute({"draft": {**replay, "ctx": 45}, "applied": previous})
    assert failed is None and "Settings differ" in message
    artifact.write_bytes(b"changed")
    failed, message = bench.execute({"draft": replay, "applied": previous})
    assert failed is None and "checksum" in message


@needs_example
def test_submission_guard_captures_the_click_and_stays_busy_through_edits_and_failures(app_env, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    import gradio as gr

    app, fake, counters, a0 = app_env
    d = {**a0["draft"], "conditioning": "none", "features": ["solar", "tend"]}
    submission, ready = {"busy": False}, Barrier(2)

    def submit():
        ready.wait()
        try:
            return app.begin_forecast(*_vals(app, d), a0, submission)[0]
        except gr.Error:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        requests = list(pool.map(lambda _: submit(), range(2)))
    assert sum(r is not None for r in requests) == 1
    request = next(r for r in requests if r is not None)
    d["features"].append("atmos")
    d["horizon"] = 24
    feedback = app.draft_feedback(*_vals(app, d), a0, submission)
    assert feedback[4]["interactive"] is False and feedback[4]["value"] == "Running…"
    result, msg = app.run_captured(request)
    assert result["draft"]["features"] == ["solar", "tend"] and result["draft"]["horizon"] == 48
    got, _, changed = app.apply_if_current(result, *_vals(app, d), a0, msg)
    assert got is a0 and not changed and submission["busy"]
    feedback = app.finish_forecast(*_vals(app, d), a0, submission)
    assert not submission["busy"] and feedback[4]["interactive"] is True

    other_session = {"busy": False}
    request, _, _ = app.begin_forecast(*_vals(app, d), a0, submission)
    app.begin_forecast(*_vals(app, d), a0, other_session)  # sessions do not block each other

    def broken(*vals):
        raise ValueError("invalid request")

    monkeypatch.setattr(app, "run_forecast", broken)
    result, msg = app.run_captured(request)
    assert result is None and "invalid request" in msg
    app.finish_forecast(*_vals(app, d), a0, submission)
    assert not submission["busy"]


@needs_example
def test_cache_hit_is_applied_only_by_run_and_display_only_runs_infer_nothing(app_env):
    app, fake, counters, a0 = app_env
    d = {**a0["draft"], "conditioning": "none", "features": ["solar", "tend"], "horizon": 72}
    a1, msg, changed = _run(app, d, a0)                                   # cache miss: computes once
    assert changed and a1["source"] == "exploratory" and fake.inference == 1 and "Computed" in msg
    a2, msg, changed = _run(app, {**d, "variable": "pmer", "error_mode": "signed"}, a1)
    assert changed and fake.inference == 1 and fake.calls == 2 and "same forecast" in msg
    assert a2["view"]["variable"] == "pmer" and a2["view"]["error_mode"] == "signed"
    assert a2["exploratory"] is a1["exploratory"] or a2["exploratory"] == a1["exploratory"]
    back, _, _ = _run(app, a0["draft"], a2)                               # back to the saved forecast
    assert back["source"] == "saved" and fake.inference == 1
    again, msg, _ = _run(app, d, back)                                    # same configuration: reused
    assert fake.inference == 1 and "cached" in msg and again["primary"] == a1["primary"]


@needs_example
@pytest.mark.parametrize("previous_source", ["saved", "exploratory"])
def test_error_method_reuses_previous_direct_result_and_exports_each_configuration(
        app_env, tmp_path, monkeypatch, previous_source):
    app, fake, counters, previous = app_env
    if previous_source == "exploratory":
        previous, _, _ = _run(app, {**previous["draft"], "conditioning": "none", "features": ["tend"]}, previous)
    direct = previous["primary"]
    snapshot = repr(previous)
    d = {**previous["draft"], "prediction_target": "arome_error", "horizon": 24, "features": ["solar", "tend"]}
    before, n = counters(), fake.inference
    assert "Changes pending" in app.draft_feedback(*_vals(app, d), previous, {"busy": False})[0]
    assert counters() == before and repr(previous) == snapshot
    assert app.saved_match(d, app.draft_config(d)[0]) is None
    result, msg = app.run_forecast(*_vals(app, d), previous)
    old, _, changed = app.apply_if_current(result, *_vals(app, {**d, "prediction_target": "weather"}), previous, msg)
    assert old is previous and not changed  # switching methods during execution invalidates the response
    a, _, changed = app.apply_if_current(result, *_vals(app, d), previous, msg)
    assert changed and fake.inference == n + 1 and a["comparison"] == direct
    assert a["config"].targets == ("t", "pmer") and a["config"].prediction_target == "arome_error"
    assert {"arome-raw", direct, a["primary"]} <= set(a["view"]["shown"])
    assert a["view"]["band"] == a["primary"]
    assert a["configs_by"][direct] == previous["config"] and repr(previous) == snapshot
    if previous_source == "exploratory":
        pd.testing.assert_frame_equal(a["exploratory"][direct], previous["exploratory"][direct])
    monkeypatch.setattr(app, "ROOT", tmp_path)
    _, _, fig, caption, _, _, csv, _, cards = app.render(a)
    assert "AROME + predicted error" in app.applied_text(a) and "AROME + predicted error" in cards
    assert "Previous direct prediction retained" in caption and "previously applied forecast" in cards
    assert {"arome-raw", direct, a["primary"]} <= {t.legendgroup for t in fig.data}
    assert app.model_style(a["primary"]) != app.model_style(direct)
    corrected_traces = [t for t in fig.data if t.legendgroup == a["primary"]]
    assert corrected_traces and all(t.line.color == "#6250d6" for t in corrected_traces)
    assert all(t.name == "Timestral 209M · AROME error correction" for t in corrected_traces if t.meta != "bridge")
    assert all(t.fillcolor.startswith("rgba(98,80,214,") for t in fig.data if t.legendgroup == "band")
    out = pd.read_csv(csv)
    methods = out.groupby("model")["prediction_method"].first()
    assert methods[a["primary"]] == "arome_error" and methods[direct] == "weather" and methods["arome-raw"] == "raw_arome"
    assert out[out.model == direct]["forecast_configuration"].str.contains("AROME + predicted error", regex=False).sum() == 0
    assert out[out.model == a["primary"]]["forecast_configuration"].str.contains("AROME + predicted error", regex=False).all()
    corrected = out[out.model == a["primary"]]
    np.testing.assert_allclose(corrected["point"], corrected["arome_baseline"] + corrected["predicted_error"])
    assert (corrected["inference_horizon_h"] == 24).all()
    assert (out[out.model == direct]["inference_horizon_h"] == 48).all()
    # Presentation-only Run keeps the comparison and uses no new predictions.
    a2, msg, _ = _run(app, {**d, "variable": "pmer", "error_mode": "signed"}, a)
    assert "same forecast" in msg and fake.inference == n + 1 and a2["comparison"] == direct
    back, _, _ = _run(app, previous["draft"], a2)
    again, msg, _ = _run(app, d, back)
    assert "cached" in msg and fake.inference == n + 1 and again["comparison"] == direct


@needs_example
def test_error_method_reports_baseline_history_and_horizon_limits(app_env):
    app, _, counters, a = app_env
    d = {**a["draft"], "prediction_target": "arome_error", "conditioning": "none"}
    assert app.horizon_limit(d["suite"], d["case"], "none", "arome_error")[0] == 48
    cfg, issues = app.draft_config(d)
    assert issues == [] and cfg.prediction_target == "arome_error"
    assert any(i.subject == "horizon" for i in app.draft_config({**d, "horizon": 72})[1])
    assert any("wind baseline" in i.message for i in app.draft_config({**d, "variable": "u"})[1])
    no_history = Availability(arome_issued_h=48, arome_run=ALL_DATA.arome_run)
    assert any("archived error history" in i.message for i in check(cfg, no_history))
    direct = ForecastConfig.make(cfg.model, targets=cfg.targets, horizon_h=cfg.horizon_h)
    assert cfg.forecast_part() != direct.forecast_part()


@needs_example
def test_failed_and_outdated_runs_never_overwrite_the_active_result(app_env, monkeypatch):
    app, fake, counters, a0 = app_env
    d = {**a0["draft"], "conditioning": "none", "ctx": 60}
    res, msg = app.run_forecast(*_vals(app, d), a0)
    edited = {**d, "ctx": 61}                                             # the user edits while it runs
    got, msg2, changed = app.apply_if_current(res, *_vals(app, edited), a0, msg)
    assert got is a0 and not changed and "not applied" in msg2
    got, _, changed = app.apply_if_current(res, *_vals(app, d), a0, msg)
    assert got is res["applied"] and changed

    def broken(*a, **k):
        raise RuntimeError("GPU quota exceeded")

    monkeypatch.setattr(app, "_exploratory_on_device", broken)
    res, msg = app.run_forecast(*_vals(app, {**d, "ctx": 62}), a0)
    assert res is None and "failed" in msg and "unchanged" in msg
    got, _, changed = app.apply_if_current(res, *_vals(app, {**d, "ctx": 62}), a0, msg)
    assert got is a0 and not changed
    res, msg = app.run_forecast(*_vals(app, {**a0["draft"], "horizon": 72}), a0)       # unsupported: explained
    assert res is None and "unavailable" in msg


@needs_example
def test_chart_cards_intervals_captions_and_export_share_one_applied_configuration(app_env):
    app, fake, counters, a0 = app_env
    d = {**a0["draft"], "conditioning": "none", "features": ["tend"], "horizon": 72, "error_mode": "signed"}
    a, _, _ = _run(app, d, a0)
    systems, band, fig, caption, this, desc, csv, smap, cards = app.render(a)
    mid = a["primary"]
    assert band["value"] == mid and mid in systems["value"]
    shaded = [t for t in fig.data if t.legendgroup == "band" and t.fill == "toself"]
    assert shaded and all(t.name.endswith(app.model_name(mid)) for t in shaded if t.name)   # interval: applied system
    plotted = max(pd.Timestamp(x, tz="UTC") if pd.Timestamp(x).tzinfo is None else pd.Timestamp(x) for t in fig.data if t.legendgroup == mid and t.yaxis == "y" and t.meta != "bridge"
                  for x in t.x)
    origin = pd.Timestamp(app.RUNS["arome-issued-v2"].cases().set_index("case_id").loc[a["case"], "origin"])
    assert plotted - origin == pd.Timedelta(hours=72) and "leads 3–72 h" in caption
    assert "inference horizon 72 h" in cards and "Tendencies" in cards
    out = pd.read_csv(csv)
    assert out["lead_h"].max() == 72 and set(out["model"]) == set(systems["value"])
    assert (out["applied_configuration"] == app.applied_text(a)).all()
    assert set(out.loc[out["model"] == mid, "source"]) == {"exploratory (outside the frozen benchmark, not scored)"}
    assert "signed error" in caption and fig.layout.yaxis2.title.text.startswith("Forecast −")
    import gradio as gr

    assert all(x == gr.skip() for x in app.render(a, False))           # nothing applied: nothing redrawn


# --------------------------------------------------------------------------- applied exports

@needs_example
def test_new_registered_group_uses_its_defaults_and_declared_data_dependencies(tmp_path, monkeypatch):
    from weather_lab import forecast, models
    from weather_lab_app import main as app
    from weather_lab_app.results import load_run

    monkeypatch.setattr(app, "RUNS", {"arome-issued-v2": load_run(EXAMPLE)})
    a0 = app.initial_applied("arome-issued-v2")
    # The test group combines two existing inputs; its construction remains in the shared pipeline.
    class RegisteredGroup(feats.FeatureGroup):
        def rows(self, params=None):
            return ("extra_history",)

    group = RegisteredGroup("extra", 1, "past", "Extra history", ("arome_history", "aux_history"),
                            "history at or before origin", "mask NaN", "test-only combined history")
    monkeypatch.setitem(feats.GROUPS, "extra", group)
    original = feats._block

    def block(name, params, batch, t_ctx):
        if name == "extra":
            return (batch.history[:, :1, -t_ctx:] - batch.arome_history[:, :1, -t_ctx:]
                    + batch.aux_history[:, :1, -t_ctx:])
        return original(name, params, batch, t_ctx)

    monkeypatch.setattr(feats, "_block", block)
    monkeypatch.setenv("WEATHER_LAB_PREDICTION_CACHE", str(tmp_path))
    d = {**a0["draft"], "conditioning": "none", "features": ["extra"]}
    cfg, issues = app.draft_config(d)
    assert not issues and cfg.feature_spec == {"extra": {}}
    obs, case, ar, aux, coords = app._explore_inputs(d, cfg)
    assert ar is not None and aux is not None
    adapter = forecast.Timestral("extra", list(cfg.targets), list(cfg.targets), "mistralai/Timestral-5M", "r" * 40,
                                 features=cfg.feature_spec)
    assert "arome" in adapter.inputs
    # A lookup miss builds the real inputs without loading weights or running inference.
    before = dict(models.STATS), forecast.STATS["inference"]
    frame, info = forecast.exploratory_forecast(
        obs, case, list(cfg.targets), "Timestral-5M", cfg.context_days, cfg.calendar,
        features=cfg.feature_spec, arome_df=ar, aux_df=aux, station_coords=coords, lookup_only=True)
    assert frame is None and info["settings"]["feature_inputs"]["observed_fraction"]["extra_history"] > 0
    assert (dict(models.STATS), forecast.STATS["inference"]) == before

@needs_example
def test_exports_keep_the_applied_overlays_without_overwriting_earlier_downloads(app_env, tmp_path, monkeypatch):
    from pathlib import Path

    app, _, _, a0 = app_env
    monkeypatch.setattr(app, "ROOT", tmp_path)
    first = Path(app.render(a0)[6])
    original = first.read_bytes()
    a1 = {**a0, "view": {**a0["view"], "shown": [a0["primary"]]}}
    second = Path(app.render(a1)[6])
    assert second != first and first.read_bytes() == original
    assert set(pd.read_csv(second)["model"]) == {a0["primary"]}
    assert Path(app.render(a0)[6]) == first
    assert first.parent == tmp_path / ".tmp" / "codex" / "forecast-exports"


# --------------------------------------------------------------------------- real pipeline

@pytest.mark.real_model
@needs_example
def test_multi_group_combination_runs_through_the_real_pipeline(tmp_path, monkeypatch):
    """Recent AROME errors + dew point/cloud + solar elevation + tendencies, on a saved October case, through the
    shared feature pipeline and the Timestral adapter (Timestral-5M, ~21 MB). Cache keys follow the features."""
    from weather_lab.forecast import STATS, exploratory_forecast
    from weather_lab.models import STATS as MS
    from weather_lab_app.results import load_run

    monkeypatch.setenv("WEATHER_LAB_PREDICTION_CACHE", str(tmp_path))
    run = load_run(EXAMPLE)
    if run.aux_observations() is None:
        pytest.skip("auxiliary observations not installed")
    case = run.cases().set_index("case_id").loc["arome-issued-v2/07149/20261021T06"]
    obs = run.full_observations()
    obs = obs[obs["station"] == case["station"]]
    kw = {"arome_df": run.arome_extract(), "aux_df": run.aux_observations(), "station_coords": run.station_coords(),
          "allow_download": True}
    spec = {"resid": {}, "atmos": {}, "solar": {}, "tend": {"lags_h": [3, 12]}}
    f1, i1 = exploratory_forecast(obs, case, ["t", "pmer", "u", "v"], "Timestral-5M", 14, True, features=spec, **kw)
    loads = MS["loads"]
    f2, i2 = exploratory_forecast(obs, case, ["t", "pmer", "u", "v"], "Timestral-5M", 14, True,
                                  features=["solar"], **kw)
    assert not i1["cache_hit"] and not i2["cache_hit"] and i1["key"] != i2["key"]
    assert MS["loads"] == loads                                  # one loaded checkpoint for every combination
    rows = i1["settings"]["feature_inputs"]["observed_fraction"]
    assert set(rows) == {"resid_t", "resid_pmer", "resid_lead_h", "td", "cloud", "sin_solar_elevation", "dt_3h",
                         "dt_12h", "dpmer_3h", "dpmer_12h"} and all(v > 0 for v in rows.values())
    assert not np.allclose(f1["point"].to_numpy(), f2["point"].to_numpy())
    n = STATS["inference"]
    f3, i3 = exploratory_forecast(obs, case, ["t", "pmer", "u", "v"], "Timestral-5M", 14, True, features=spec, **kw)
    assert i3["cache_hit"] and STATS["inference"] == n
    pd.testing.assert_frame_equal(f1.reset_index(drop=True), f3.reset_index(drop=True))
