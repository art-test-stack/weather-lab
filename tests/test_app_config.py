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


def test_config_status_separates_applied_settings_from_pending_drafts(monkeypatch):
    from weather_lab_app import main as app

    ids = ["arome-raw", "timestral-209m-mv-cal", "timestral-209m-mv-cal-arome-archived"]
    monkeypatch.setattr(app, "RUNS", {"arome-issued-v2": _Run(ids)})
    s = app.config_status("arome-issued-v2", "c1", "209M", "archived", 30, 48, True, app.EMPTY_EXTRA)
    assert "saved benchmark run" in s and "Pending" not in s
    s = app.config_status("arome-issued-v2", "c1", "209M", "archived", 30, 24, True, app.EMPTY_EXTRA)
    assert "leads ≤ 24 h (slice)" in s and "Pending" not in s                  # shortening needs no inference
    s = app.config_status("arome-issued-v2", "c1", "209M", "archived", 60, 72, True, app.EMPTY_EXTRA)
    assert "Pending" in s and "context 60 d" in s and "horizon 72 h" in s
    s = app.config_status("arome-issued-v2", "c1", "209M", "substobs", 30, 48, True, app.EMPTY_EXTRA)
    assert "no saved result" in s                                               # honest not-run, no substitute
    st = {"checkpoint": "Timestral-209M", "context_days": 30, "context_steps": 240, "horizon_steps": 24,
          "step_hours": 3, "calendar": True, "arome_mode": None, "size": "209M", "setup": "none", "cache_hit": True}
    extra = {"case": "c1", "runs": {"exploratory x": _rows("exploratory x")}, "settings": {"exploratory x": st},
             "active": "exploratory x"}
    s = app.config_status("arome-issued-v2", "c1", "209M", "none", 30, 72, True, extra)
    assert "exploratory forecast" in s and "inference horizon 72 h" in s and "Pending" not in s
    s = app.config_status("arome-issued-v2", "c1", "209M", "none", 90, 72, True, extra)
    assert "Pending" in s and "context 90 d" in s


EXAMPLE = ROOT / "data" / "examples" / "arome-issued-v2"


@pytest.mark.skipif(not (EXAMPLE / "run.json").exists(), reason="saved example not installed (weather-lab bootstrap)")
def test_horizon_slice_equals_saved_scores_at_full_horizon_and_saved_predictions_below():
    from weather_lab_app.results import headline, headline_to, load_run

    run = load_run(EXAMPLE)
    ms = [m for m in ("arome-raw", "timestral-209m-mv-cal", "timestral-209m-mv-cal-arome-archived") if m in run.models]
    a, b = headline(run, "t", models=ms).set_index("model"), headline_to(run, "t", ms, 48).set_index("model")
    pd.testing.assert_frame_equal(a[["mae", "n"]], b[["mae", "n"]], check_dtype=False)
    p, t = run.predictions(), run.targets()
    x = p[(p.variable == "t") & (p.model == ms[-1]) & (p.lead_h <= 24)].merge(t, on=["case_id", "variable", "lead_h"])
    assert headline_to(run, "t", ms, 24).set_index("model").loc[ms[-1], "mae"] == pytest.approx(
        (x.point - x.observed).abs().mean())
