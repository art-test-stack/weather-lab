"""Opt-in real-model smoke check: downloads Timestral-5M (~21 MB) and runs inference.

    uv run pytest -m real_model
"""

import numpy as np
import pandas as pd
import pytest

from weather_lab.forecast import CHECKPOINTS, CaseBatch, Timestral

pytestmark = pytest.mark.real_model


def test_timestral_5m_forecast_shapes_and_sanity():
    checkpoint, revision, _ = CHECKPOINTS["Timestral-5M"]
    origin = pd.Timestamp("2024-07-01T00:00Z")
    t_ctx = 7 * 8
    hours = np.arange(-(t_ctx - 1), 1) * 3
    t = 20 + 6 * np.sin(2 * np.pi * (hours % 24 - 9) / 24)      # synthetic diurnal cycle, not evidence
    hist = np.stack([t, 1015 + 0 * t, 0 * t + 1, 0 * t - 1]).astype(np.float32)[None]
    batch = CaseBatch(pd.DataFrame([{"case_id": "x", "station": "07149"}]), ["t", "pmer", "u", "v"], hist,
                      np.array([origin.to_datetime64()]),
                      np.array([[(origin + pd.Timedelta(hours=3 * (k + 1))).to_datetime64() for k in range(16)]]), 3)
    model = Timestral("t5m", batch.variables, targets=["t", "pmer", "u", "v"], checkpoint=checkpoint,
                      revision=revision, context_days=7, calendar=True)
    fb = model.predict(batch)
    fb.validate(batch)
    assert fb.quantiles.shape == (1, 4, 16, 21)
    assert np.isfinite(fb.quantiles).all()
    assert 10 < fb.point[0, 0].mean() < 30 and abs(fb.point[0, 1].mean() - 1015) < 5


def test_one_load_per_checkpoint_and_no_request_state_leak(monkeypatch, tmp_path):
    """Real loader: covariate setups share one loaded model; outputs do not depend on earlier requests;
    an offline process with the files cached needs no Hub access; exploratory predictions are cached."""
    from weather_lab import models
    from weather_lab.forecast import exploratory_forecast

    monkeypatch.setattr(models, "MODELS", models.ModelCache(max_models=2))
    checkpoint, revision, _ = CHECKPOINTS["Timestral-5M"]
    origin = pd.Timestamp("2024-07-01T00:00Z")
    t_ctx, h = 7 * 8, 16
    hours = np.arange(-(t_ctx - 1), 1) * 3
    t = 20 + 6 * np.sin(2 * np.pi * (hours % 24 - 9) / 24)      # synthetic, not evidence
    hist = np.stack([t, 1015 + 0 * t, 0 * t + 1, 0 * t - 1]).astype(np.float32)[None]
    future = np.array([[(origin + pd.Timedelta(hours=3 * (k + 1))).to_datetime64() for k in range(h)]])
    arome = np.full((1, 4, h), np.nan, np.float32)
    arome[0, 0] = 22.0
    arome[0, 1] = 1012.0
    batch = CaseBatch(pd.DataFrame([{"case_id": "x", "station": "07149"}]), ["t", "pmer", "u", "v"], hist,
                      np.array([origin.to_datetime64()]), future, 3, arome)
    mk = lambda mode: Timestral(f"t5m-{mode}", batch.variables, targets=batch.variables, checkpoint=checkpoint,  # noqa: E731
                                revision=revision, context_days=7, calendar=True, arome=mode,
                                arome_fields=["t", "pmer"])
    before = dict(models.STATS)
    q_sub = mk("substituted_observations").predict(batch).quantiles
    mk("missing").predict(batch)
    assert np.array_equal(mk("substituted_observations").predict(batch).quantiles, q_sub)
    assert models.STATS["loads"] - before["loads"] == 1 and models.STATS["load_hits"] - before["load_hits"] == 2
    monkeypatch.setenv("WEATHER_LAB_OFFLINE", "1")                       # files cached now: no Hub needed
    monkeypatch.setenv("WEATHER_LAB_PREDICTION_CACHE", str(tmp_path))
    obs = pd.DataFrame({"time": pd.date_range(origin - pd.Timedelta(hours=3 * (t_ctx - 1)), origin, freq="3h"),
                        "t": t, "pmer": 1015.0, "u": 1.0, "v": -1.0})
    case = pd.Series({"station": "07149", "origin": origin}, name="x")
    f1, i1 = exploratory_forecast(obs, case, batch.variables, "Timestral-5M", 7, True)
    f2, i2 = exploratory_forecast(obs, case, batch.variables, "Timestral-5M", 7, True)
    f3, i3 = exploratory_forecast(obs, case, batch.variables, "Timestral-5M", 7, False)   # other covariates
    assert (i1["cache_hit"], i2["cache_hit"], i3["cache_hit"]) == (False, True, False) and f1.equals(f2)
