from dataclasses import replace
from types import SimpleNamespace
import threading

import numpy as np
import pandas as pd
import pytest
import torch

from weather_lab import models
from weather_lab.forecast import CaseBatch, QUANTILE_LEVELS
from weather_lab.t0_adapter import T0


def batch():
    origins = np.array(["2026-10-16T06:00"], dtype="datetime64[ns]")
    times = origins[:, None] + np.arange(1, 3)[None] * np.timedelta64(3, "h")
    hist = np.full((1, 2, 8), 10, np.float32)
    hist[:, 1] = 1013
    hist[:, :, :2] = np.nan
    return CaseBatch(pd.DataFrame({"station": ["07149"]}), ["t", "pmer"], hist, origins, times, 3)


def adapter(**kwargs):
    return T0("t0-test", ["t", "pmer"], ["t", "pmer"], "theforecastingcompany/t0-gamma",
              "e08aee083778f1f32da4974c0a34276776a35c52", context_days=1, **kwargs)


def test_native_roles_masks_options_and_output_shape(monkeypatch):
    b = replace(batch(), aux_history=np.array([[[2] * 8, [0, np.nan, 50, 50, 50, 50, 50, 75]]], np.float32),
                aux_variables=["td", "cloud"])
    a = adapter(calendar=True, features=["atmos"], trim_leading_missing=True, symmetric_averaging=True)
    context, known, report = a.build_inputs(b)
    assert context.shape == (1, 4, 8) and known.shape == (1, 4, 10)
    np.testing.assert_equal(context[:, :2], b.history)
    assert np.isnan(context[0, 3, 1])
    assert report["native_past_semantics"].startswith("extra jointly")
    calls = []

    def predict(x, **kwargs):
        calls.append(kwargs)
        q = np.broadcast_to(x[..., -1, None, None], (*x.shape[:2], 2, 21)).copy()
        q += np.linspace(-1, 1, 21)
        return SimpleNamespace(quantiles=torch.from_numpy(q), quantile_levels=QUANTILE_LEVELS)

    forecaster = SimpleNamespace(config=SimpleNamespace(quantile_levels=QUANTILE_LEVELS), predict=predict)
    monkeypatch.setattr(models.MODELS, "get", lambda *args: SimpleNamespace(forecaster=forecaster, lock=threading.Lock(), info={}))
    result = a.predict(b)
    result.validate(b)
    assert result.quantiles.shape == (1, 2, 2, 21)
    assert result.point[0, 1, 0] == 1013
    assert calls[0]["trim_leading_missing"] and calls[0]["symmetric_averaging"]
    assert calls[0]["clip_nonnegative"] is False and calls[0]["snap_integers"] is False
    assert calls[0]["quantile_levels"] == QUANTILE_LEVELS
    forecaster.config.quantile_levels = (.1, .5, .9)
    with pytest.raises(ValueError, match="native quantile grid"):
        a.predict(b)


def test_cache_identity_and_capability_rejections():
    b = batch()
    a = adapter(calendar=True)
    key = a.input_identity(b)
    assert key != adapter(calendar=True, trim_leading_missing=True).input_identity(b)
    assert key != adapter(calendar=True, symmetric_averaging=True).input_identity(b)
    assert key != adapter(calendar=False).input_identity(b)
    changed = b.history.copy()
    changed[0, 0, -1] += 1
    assert key != a.input_identity(replace(b, history=changed))
    changed[0, 0, -1] = np.nan
    assert key != a.input_identity(replace(b, history=changed))
    with pytest.raises(ValueError, match="clipping"):
        adapter(clip_nonnegative=True)
    with pytest.raises(ValueError, match="snapping"):
        adapter(snap_integers=True)
    with pytest.raises(ValueError, match="residual"):
        adapter(prediction_target="arome_error")
    with pytest.raises(ValueError, match="substitution"):
        adapter(arome="substituted_observations")
    with pytest.raises(ValueError, match="entirely missing"):
        a.build_inputs(replace(b, history=np.full_like(b.history, np.nan)))


def test_issued_arome_values_and_selected_feature_missing_policy():
    b = replace(batch(), arome=np.array([[[11, 12], [1012, 1011]]], np.float32))
    a = adapter(calendar=True, arome="missing", arome_fields=["t", "pmer"])
    context, known, _ = a.build_inputs(b)
    assert np.isnan(known[:, -2:, :8]).all()
    np.testing.assert_equal(known[:, -2:, -2:], b.arome)
    changed = b.arome.copy()
    changed[0, 0, 0] += 2
    assert a.input_identity(b) != a.input_identity(replace(b, arome=changed))
    changed[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="native NaN-to-zero"):
        a.build_inputs(replace(b, arome=changed))
    with pytest.raises(ValueError, match="historical feature row is entirely missing"):
        adapter(features=["atmos"]).build_inputs(replace(b, aux_history=np.full((1, 2, 8), np.nan, np.float32),
                                                       aux_variables=["td", "cloud"]))
