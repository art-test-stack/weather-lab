"""Forecast configuration, horizon slicing, the observation bridge and exploratory cards (no model calls)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from weather_lab.common import ROOT
from weather_lab.forecast import QUANTILE_LEVELS, arome_coverage_steps, qcol

ORIGIN = pd.Timestamp("2026-10-21T06:00Z")


def _rows(model: str, leads=range(3, 25, 3), base: float = 10.0, quantiles: bool = True) -> pd.DataFrame:
    leads = list(leads)
    df = pd.DataFrame({"model": model, "variable": "t", "lead_h": leads, "case_id": "c1",
                       "valid_time": [ORIGIN + pd.Timedelta(hours=h) for h in leads],
                       "point": [base + 0.1 * h for h in leads],
                       "output_type": "quantiles" if quantiles else "point"})
    for q in QUANTILE_LEVELS:
        df[qcol(q)] = df["point"] + (q - 0.5) * 4 if quantiles else np.nan
    return df


def _obs(last_hours_before_origin: int = 0, future: bool = True) -> pd.DataFrame:
    hist = [ORIGIN - pd.Timedelta(hours=h) for h in range(last_hours_before_origin, 72, 3)]
    fut = [ORIGIN + pd.Timedelta(hours=h) for h in range(3, 25, 3)] if future else []
    times = sorted(hist + fut)
    return pd.DataFrame({"time": times, "observed": [9.0 + 0.05 * i for i in range(len(times))]})


def test_bridge_is_drawing_only_and_error_panel_matches_scoring():
    from weather_lab_app import plots

    rows = pd.concat([_rows("timestral-209m-mv-cal"), _rows("arome-raw", base=11, quantiles=False)])
    obs, case = _obs(), pd.Series({"origin": ORIGIN})
    systems = ["arome-raw", "timestral-209m-mv-cal"]
    for mode in ("absolute", "signed"):
        fig = plots.forecast_figure(rows, obs, case, "t", systems, "timestral-209m-mv-cal", context_hours=72,
                                    error_mode=mode)
        bridges = [t for t in fig.data if t.meta == "bridge"]
        assert len(bridges) == 2 and all(t.yaxis == "y" for t in bridges)          # never in the error panel
        assert all(t.x[0] == ORIGIN and t.y[0] == obs.loc[obs.time == ORIGIN, "observed"].iloc[0] for t in bridges)
        panel = [t for t in fig.data if t.yaxis == "y2"]
        verif = obs.set_index("time")["observed"]
        for t in panel:
            m = next(m for m in systems if t.legendgroup == m)
            g = rows[rows.model == m]
            err = g["point"].to_numpy() - verif.reindex(g["valid_time"]).to_numpy()
            np.testing.assert_allclose(t.y, np.abs(err) if mode == "absolute" else err)
        assert fig.layout.yaxis2.title.text.startswith("|Forecast" if mode == "absolute" else "Forecast −")
    # the data passed in is untouched: no artificial first point, nothing added to the predictions
    assert len(rows) == 16 and rows["lead_h"].min() == 3


def test_bridge_starts_at_the_real_last_observation_when_the_origin_is_missing():
    from weather_lab_app import plots

    obs = _obs(last_hours_before_origin=9)
    assert plots.last_observation(obs, ORIGIN)[0] == ORIGIN - pd.Timedelta(hours=9)
    fig = plots.forecast_figure(_rows("timestral-209m-mv-cal"), obs, pd.Series({"origin": ORIGIN}), "t",
                                ["timestral-209m-mv-cal"], None, context_hours=72)
    bridge = next(t for t in fig.data if t.meta == "bridge")
    assert bridge.x[0] == ORIGIN - pd.Timedelta(hours=9)


def test_forecast_scores_report_not_yet_observable_without_truth():
    from weather_lab_app import main as app

    rows = _rows("exploratory x", leads=range(3, 73, 3))
    sc, n = app.forecast_scores(rows, _obs(future=False), ["exploratory x"])
    assert n == 0 and "Not yet observable" in app.this_forecast_html(sc, n, "t")
    sc, n = app.forecast_scores(pd.concat([rows, _rows("arome-raw", quantiles=False)]), _obs(),
                                ["exploratory x", "arome-raw"])
    assert n == 8 and int(sc.loc["exploratory x", "covers_h"]) == 72      # shared window only: 8 observed leads


def test_mae_and_rmse_share_observed_leads_and_selected_metric_is_first():
    from weather_lab_app import main as app

    first = _rows("first", leads=[3, 6, 9]).assign(point=[0.0, 4.0, 1000.0])
    second = _rows("second", leads=[3, 6]).assign(point=[2.0, 2.0])
    obs = pd.DataFrame({"time": first.valid_time, "observed": [0.0, 0.0, 0.0]})
    scores, n = app.forecast_scores(pd.concat([first, second]), obs, ["first", "second"])
    assert n == 2 and scores["n"].tolist() == [2, 2]
    assert scores["mae"].tolist() == [2.0, 2.0]
    assert scores.loc["first", "rmse"] == pytest.approx(np.sqrt(8))
    assert scores.loc["second", "rmse"] == 2.0
    for metric, headers in (("mae", "<th>MAE</th><th>RMSE</th>"), ("rmse", "<th>RMSE</th><th>MAE</th>")):
        assert headers in app.this_forecast_html(scores, n, "t", metric)


def test_arome_coverage_never_pads():
    rows = np.ones((2, 24), dtype=np.float32)
    rows[:, 16:] = np.nan
    assert arome_coverage_steps(rows) == 16
    rows[1, 5] = np.nan
    assert arome_coverage_steps(rows) == 5
    assert arome_coverage_steps(np.ones((2, 8))) == 8


class _Run:
    """Minimal saved run with provenance (run.json systems + manifest protocol)."""

    suite = "arome-issued-v2"
    splits = ["test"]
    source = "test fixture"
    step_hours = 3
    horizon_hours = 48
    manifest = {"protocol": {"stations": ["07630", "07149", "07110"], "forecast": {"context_days": 30}}}

    def __init__(self, ids):
        self.models = list(ids)
        self.run = {"run_id": "fake-run", "systems": [
            {"id": m, "params": {"context_days": 30, "calendar": True} if m.startswith("timestral") else {}}
            for m in ids]}

    def params(self, model):
        return next((s["params"] for s in self.run["systems"] if s["id"] == model), {})


def test_defaults_are_paris_orly_and_the_largest_checkpoint(monkeypatch):
    from weather_lab_app import main as app

    assert app.DEFAULT_SIZE == "209M" and app.DEFAULT_STATION == "07149"
    monkeypatch.setattr(app, "RUNS", {"arome-issued-v2": _Run(["arome-raw", "timestral-209m-mv-cal"])})
    assert app.default_station("arome-issued-v2") == "07149"
    assert app.default_context("arome-issued-v2", "209M", "archived") == 30   # from the run, not a slider default


EXAMPLE = ROOT / "data" / "examples" / "arome-issued-v2"


@pytest.mark.skipif(not (EXAMPLE / "run.json").exists(), reason="saved example not installed (weather-lab bootstrap)")
def test_horizon_slice_equals_saved_scores_at_full_horizon_and_saved_predictions_below():
    from weather_lab_app.results import headline, headline_to, load_run

    run = load_run(EXAMPLE)
    ms = [m for m in ("arome-raw", "timestral-209m-mv-cal", "timestral-209m-mv-cal-arome-archived") if m in run.models]
    a, b = headline(run, "t", models=ms).set_index("model"), headline_to(run, "t", ms, 48).set_index("model")
    pd.testing.assert_frame_equal(a[["mae", "rmse", "n"]], b[["mae", "rmse", "n"]], check_dtype=False)
    p, t = run.predictions(), run.targets()
    x = p[(p.variable == "t") & (p.model == ms[-1]) & (p.lead_h <= 24)].merge(t, on=["case_id", "variable", "lead_h"])
    assert headline_to(run, "t", ms, 24).set_index("model").loc[ms[-1], "mae"] == pytest.approx(
        (x.point - x.observed).abs().mean())
    assert headline_to(run, "t", ms, 24).set_index("model").loc[ms[-1], "rmse"] == pytest.approx(
        np.sqrt((x.point - x.observed).pow(2).mean()))


@pytest.mark.skipif(not (EXAMPLE / "run.json").exists(), reason="saved example not installed (weather-lab bootstrap)")
def test_benchmark_rmse_uses_saved_scores_in_findings_plots_and_tables(monkeypatch):
    from weather_lab import models
    from weather_lab_app import main as app
    from weather_lab_app.results import load_run

    run = load_run(EXAMPLE)
    monkeypatch.setattr(app, "RUNS", {run.suite: run})

    def no_model(*args, **kwargs):
        raise AssertionError("Selecting a metric must not load a model")

    monkeypatch.setattr(models.ModelCache, "get", no_model)
    finding, _, bars, lines, _, point, _ = app.benchmark(
        run.suite, "t", "rmse", "all", run.splits[-1], "ALL", "ALL", app.STATUSES,
        "209M", "archived", "cumulative")
    saved = app._select(run.tables["metrics_point"], run.splits[-1], "t", "ALL", "ALL", "all")
    assert "root mean squared error" in finding
    assert "Root-mean-square error" in bars.layout.xaxis.title.text
    assert "Root-mean-square error" in lines.layout.yaxis.title.text
    np.testing.assert_allclose(bars.data[0].x, saved.rmse.sort_values(ascending=False))
    assert list(point.columns[:3]) == ["System", "RMSE (°C)", "MAE (°C)"]
    np.testing.assert_allclose(point["RMSE (°C)"], saved.rmse.sort_values().round(2))


@pytest.mark.skipif(not (EXAMPLE / "run.json").exists(), reason="saved example not installed (weather-lab bootstrap)")
def test_horizon_limit_and_saved_resolution_on_the_real_example(monkeypatch):
    from weather_lab_app import main as app
    from weather_lab_app.results import load_run

    run = load_run(EXAMPLE)
    monkeypatch.setattr(app, "RUNS", {"arome-issued-v2": run})
    case = app.default_case("arome-issued-v2", "07149")
    assert app.horizon_limit("arome-issued-v2", case, "archived")[0] == 48       # AROME run ends there
    assert app.horizon_limit("arome-issued-v2", case, "none")[0] == app.EXPLORE_MAX_HOURS
    upd, info = app.horizon_control("arome-issued-v2", case, "archived", 96)
    assert upd["value"] == 48 and upd["maximum"] == 48 and "48 h" in info and "Reduced from 96 h" in info
    d = app.default_draft("arome-issued-v2")
    cfg, issues = app.draft_config({**d, "horizon": 24})
    assert not issues and app.saved_match(d, cfg) == "timestral-209m-mv-cal-arome-archived"
    cfg45, _ = app.draft_config({**d, "ctx": 45})
    assert app.saved_match(d, cfg45) is None                                     # different context: inference
    cfg72, issues = app.draft_config({**d, "horizon": 72})
    assert [i.kind for i in issues] == ["unavailable"] and "AROME" in issues[0].message
