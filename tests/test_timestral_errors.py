"""AROME-error targets, lead/time inputs and reconstruction (synthetic fixtures, no weights)."""

import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from weather_lab import models
from weather_lab.forecast import QUANTILE_LEVELS, CaseBatch, Timestral, calendar_features, exploratory_inputs


def _batch():
    origin = np.array([np.datetime64("2026-10-12T06:00", "ns")])
    times = origin[:, None] + np.arange(-15, 1)[None] * np.timedelta64(3, "h")
    history = np.stack([10. + np.arange(16), np.full(16, 1015.)])[None].astype(np.float32)
    archived = history - np.array([np.linspace(-2., 2., 16), np.full(16, .5)])[None]
    archived[..., :4] = np.nan
    runs = times - np.timedelta64(9, "h")
    runs[:, :4] = np.datetime64("NaT", "ns")
    leads = np.full((1, 16), 9.)
    leads[:, :4] = np.nan
    cases = pd.DataFrame([{"case_id": "c", "station": "07149", "arome_run": "2026-10-12T03:00Z"}])
    return CaseBatch(cases, ["t", "pmer"], history, origin,
                     origin[:, None] + np.arange(1, 5)[None] * np.timedelta64(3, "h"), 3,
                     arome=np.array([[[20., 21., 22., 23.], [1010., 1011., 1012., 1013.]]], dtype=np.float32),
                     arome_history=archived.astype(np.float32), arome_history_run=runs,
                     arome_history_lead_h=leads, station_coords={"07149": (48.7168, 2.3843)})


def _model(size="Timestral-28M", **kwargs):
    ck = models.CHECKPOINTS[size]
    return Timestral("AROME + Timestral errors", ["t", "pmer"], targets=["t", "pmer"], checkpoint=ck.repo,
                      revision=ck.revision, context_days=2, calendar=True, arome="archived",
                      arome_fields=["t", "pmer"], prediction_target="arome_error", **kwargs)


@pytest.mark.parametrize("size", ["Timestral-28M", "Timestral-209M"])
def test_errors_keep_calendar_lead_and_selected_features_in_the_original_units(size):
    batch = _batch()
    snapshot = batch.history.copy()
    model = _model(size, features={"solar": {}, "tend": {"variables": ["t"], "lags_h": [3]}})
    ctx, known = model.build_inputs(batch)
    np.testing.assert_allclose(ctx, batch.history - batch.arome_history, equal_nan=True)
    assert np.isnan(ctx[..., :4]).all()
    times = np.concatenate([batch.history_times(), batch.future_times], axis=1)
    np.testing.assert_allclose(known[:, :4], calendar_features(times))
    np.testing.assert_allclose(known[:, -1, -4:], [[6., 9., 12., 15.]])
    np.testing.assert_allclose(known[:, -1, 4:16], 9.)
    np.testing.assert_allclose(known[:, 4:6, :16], batch.arome_history, equal_nan=True)
    past, solar, _ = model.feature_rows(batch)
    np.testing.assert_allclose(past[0, 0, 1:], 1.)  # differences of observations, not differences of errors
    np.testing.assert_array_equal(known[:, 6:7], solar)
    np.testing.assert_array_equal(batch.history, snapshot)
    assert model.describe()["params"]["residual_version"] == 1


def test_every_error_quantile_is_added_to_the_issued_baseline(monkeypatch):
    import torch

    batch = _batch()
    seen = []

    def forecast(context, **kwargs):
        seen.append((context.copy(), kwargs))
        errors = np.broadcast_to(-2. + np.array(QUANTILE_LEVELS) - .5, (1, 2, 4, 21)).copy()
        return SimpleNamespace(quantiles=torch.tensor(errors, dtype=torch.float32))

    item = SimpleNamespace(forecaster=SimpleNamespace(forecast=forecast, quantile_levels=QUANTILE_LEVELS),
                           info={}, lock=threading.Lock())
    monkeypatch.setattr(models.MODELS, "get", lambda *args: item)
    model = _model(features=["solar", "tend"])
    result = model.predict(batch)
    result.validate(batch)
    expected = batch.arome[..., None] - 2. + np.array(QUANTILE_LEVELS) - .5
    np.testing.assert_allclose(result.quantiles, expected, atol=1e-4)
    np.testing.assert_allclose(result.point, batch.arome - 2.)
    assert seen[0][1]["clip_nonnegative"] is False
    np.testing.assert_allclose(seen[0][0], batch.history - batch.arome_history, equal_nan=True)
    assert result.info["residual_observed_steps"] == [[12, 12]]


def test_unavailable_baseline_or_error_history_fails_before_loading_a_model(monkeypatch):
    monkeypatch.setattr(models.MODELS, "get", lambda *args: pytest.fail("invalid inputs loaded a model"))
    batch = _batch()
    for missing in (replace(batch, arome=None), replace(batch, arome_history=None)):
        with pytest.raises(ValueError, match="issued and qualifying archived"):
            _model().predict(missing)
    baseline = batch.arome.copy()
    baseline[0, 1, -1] = np.nan
    with pytest.raises(ValueError, match="issued pmer forecast at every requested lead"):
        _model().predict(replace(batch, arome=baseline))
    archived = batch.arome_history.copy()
    archived[:, 0] = np.nan
    with pytest.raises(ValueError, match="no observed t errors"):
        _model().predict(replace(batch, arome_history=archived))
    cases = batch.cases.assign(arome_run="2026-10-12T07:00Z")
    with pytest.raises(ValueError, match="at or before the forecast origin"):
        _model().predict(replace(batch, cases=cases))


def test_future_observations_and_late_arome_runs_cannot_change_error_context():
    origin = pd.Timestamp("2026-10-12T06:00Z")
    times = pd.date_range(origin - pd.Timedelta(hours=45), origin + pd.Timedelta(hours=12), freq="3h")
    obs = pd.DataFrame({"time": times, "t": 12., "pmer": 1015.})
    rows = []
    for run, published, value in (("2026-10-10T03:00Z", "2026-10-10T05:45Z", 10.),
                                   ("2026-10-11T03:00Z", "2026-10-11T05:45Z", 11.),
                                   ("2026-10-12T03:00Z", "2026-10-12T05:45Z", 12.),
                                   ("2026-10-11T15:00Z", "2026-10-12T09:00Z", 999.)):
        for lead in range(6, 52, 3):
            for var, delta in (("t", 0.), ("pmer", 1000.)):
                rows.append({"run": pd.Timestamp(run), "available_at": pd.Timestamp(published), "lead_h": lead,
                             "station": "07149", "variable": var, "value": value + delta,
                             "valid_time": pd.Timestamp(run) + pd.Timedelta(hours=lead)})
    frame = pd.DataFrame(rows)
    case = pd.Series({"station": "07149", "origin": origin, "arome_run": "2026-10-12T03:00Z"}, name="c")

    def inputs(observations, forecasts):
        batch = exploratory_inputs(observations, case, ["t", "pmer"], 2, "archived", forecasts, horizon_steps=4)
        return _model().build_inputs(batch)

    expected = inputs(obs, frame)
    obs.loc[obs["time"] > origin, ["t", "pmer"]] = 9999.
    frame.loc[frame["available_at"] > origin, "value"] = -9999.
    actual = inputs(obs, frame)
    for first, second in zip(expected, actual):
        np.testing.assert_array_equal(first, second)


def test_direct_weather_inputs_and_recorded_parameters_keep_the_existing_behavior():
    ck = models.CHECKPOINTS["Timestral-28M"]
    model = Timestral("direct", ["t", "pmer"], targets=["t", "pmer"], checkpoint=ck.repo,
                      revision=ck.revision, context_days=2, calendar=True)
    batch = _batch()
    ctx, known = model.build_inputs(batch)
    np.testing.assert_array_equal(ctx, batch.history)
    np.testing.assert_array_equal(known, calendar_features(np.concatenate([batch.history_times(), batch.future_times], 1)))
    assert "prediction_target" not in model.describe()["params"]
    with pytest.raises(ValueError, match="unknown prediction target"):
        Timestral("invalid", [], targets=[], checkpoint=ck.repo, revision=ck.revision, prediction_target="other")


def test_real_cache_pipeline_separates_methods_and_hashes_the_reconstruction_baseline(tmp_path, monkeypatch):
    import json

    import torch

    from weather_lab.forecast import exploratory_forecast

    batch = _batch()
    origin = pd.Timestamp(batch.origins[0], tz="UTC")
    obs = pd.DataFrame(batch.history[0].T, columns=batch.variables)
    obs["time"] = pd.to_datetime(batch.history_times()[0], utc=True)
    case = pd.Series({"station": "07149", "origin": origin, "arome_run": "2026-10-12T03:00Z"}, name="c")
    rows = []
    for j, name in enumerate(batch.variables):
        for k in range(4, 16):
            rows.append({"station": "07149", "variable": name,
                         "run": pd.Timestamp(batch.arome_history_run[0, k], tz="UTC"),
                         "available_at": pd.Timestamp(batch.arome_history_run[0, k], tz="UTC") + pd.Timedelta(hours=1),
                         "valid_time": obs["time"].iloc[k], "value": batch.arome_history[0, j, k]})
        for k in range(4):
            rows.append({"station": "07149", "variable": name, "run": pd.Timestamp(case.arome_run),
                         "available_at": origin - pd.Timedelta(minutes=15),
                         "valid_time": pd.Timestamp(batch.future_times[0, k], tz="UTC"), "value": batch.arome[0, j, k]})
    frame, calls = pd.DataFrame(rows), []

    def forecast(context, prediction_length, **kwargs):
        calls.append(context.copy())
        values = context[..., -1:, None] + np.array(QUANTILE_LEVELS) - .5
        values = np.broadcast_to(values, context.shape[:2] + (prediction_length, 21)).copy()
        return SimpleNamespace(quantiles=torch.tensor(values, dtype=torch.float32))

    item = SimpleNamespace(forecaster=SimpleNamespace(forecast=forecast, quantile_levels=QUANTILE_LEVELS),
                           info={}, lock=threading.Lock())
    monkeypatch.setattr(models.MODELS, "get", lambda *args: item)
    monkeypatch.setenv("WEATHER_LAB_OFFLINE", "1")
    monkeypatch.setenv("WEATHER_LAB_PREDICTION_CACHE", str(tmp_path))

    def run(method, forecasts, **kwargs):
        return exploratory_forecast(obs, case, batch.variables, "Timestral-28M", 2, True,
                                    arome_df=forecasts, horizon_steps=4, prediction_target=method, **kwargs)

    missing, _ = run("arome_error", frame, lookup_only=True)
    assert missing is None and not calls
    corrected, ci = run("arome_error", frame)
    direct, di = run("weather", frame)
    again, ai = run("arome_error", frame)
    assert len(calls) == 2 and not ci["cache_hit"] and ai["cache_hit"]
    assert ci["key"] != di["key"] and ci["model"] != di["model"]
    pd.testing.assert_frame_equal(corrected, again)
    np.testing.assert_allclose(corrected["point"], corrected["arome_baseline"] + corrected["predicted_error"])
    assert ci["settings"]["error_history"]["t"]["observed_steps"] == 12
    # With raw AROME conditioning off, only the reconstruction baseline changes.
    changed = frame.copy()
    changed.loc[(changed["valid_time"] > origin) & (changed["variable"] == "t"), "value"] += .25
    updated, ui = run("arome_error", changed)
    assert len(calls) == 3 and ui["key"] != ci["key"] and not ui["cache_hit"]
    np.testing.assert_allclose(updated[updated.variable == "t"]["point"],
                               corrected[corrected.variable == "t"]["point"] + .25)
    metadata = [json.loads(p.read_text()) for p in tmp_path.glob("*/*.json")]
    corrections = [m for m in metadata if m.get("prediction_target") == "arome_error"]
    assert len(corrections) == 2 and all(m["residual_version"] == 1 and m["residual_source_runs"] for m in corrections)
