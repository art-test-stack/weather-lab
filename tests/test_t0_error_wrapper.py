"""Isolated wrapper: causal residuals, native feature roles and inverse transform."""
from dataclasses import replace
import importlib.util
from types import SimpleNamespace
import threading

import numpy as np
import pandas as pd
import pytest
import torch

from weather_lab import models
from weather_lab.common import ROOT
from weather_lab.forecast import CaseBatch, QUANTILE_LEVELS
from weather_lab.t0_adapter import T0

spec = importlib.util.spec_from_file_location("isolated_t0_error_wrapper",
    ROOT / "experiments/t0-error-wrapper-v1/error_wrapper.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ResidualT0 = module.ResidualT0


def inputs():
    origin = np.array(["2026-10-21T06"], dtype="datetime64[ns]")
    times = origin[:, None] + np.arange(-15, 1) * np.timedelta64(3, "h")
    history = np.stack([np.arange(16.) + 10, np.arange(16.) + 1000])[None].astype(np.float32)
    archived = history + 2
    archived[:, :, :2] = np.nan
    runs = times - np.timedelta64(6, "h")
    runs[:, :2] = np.datetime64("NaT", "ns")
    leads = np.full((1, 16), 6.)
    leads[:, :2] = np.nan
    cases = pd.DataFrame({"case_id": ["fixture"], "station": ["07149"],
                          "arome_run": ["2026-10-21T03:00Z"], "arome_available_at": ["2026-10-21T05:00Z"]})
    return CaseBatch(cases, ["t", "pmer"], history, origin,
        origin[:, None] + np.arange(1, 3) * np.timedelta64(3, "h"), 3,
        arome=np.array([[[10., 11.], [1000., 1001.]]], np.float32), arome_history=archived,
        arome_history_run=runs, arome_history_lead_h=leads,
        aux_history=np.array([[[1.] * 16, [50.] * 16]], np.float32), aux_variables=["td", "cloud"],
        station_coords={"07149": (48.7, 2.4)})


def adapter():
    ck = models.CHECKPOINTS["t0-gamma"]
    return ResidualT0("fixture", ["t", "pmer"], ["t", "pmer"], ck.repo, ck.revision,
                      context_days=2, arome_fields=["t", "pmer"], features=["resid", "atmos", "solar", "tend"])


def test_native_inputs_keep_original_features_and_restore_every_quantile(monkeypatch):
    batch, wrapper = inputs(), adapter()
    original = batch.history.copy()
    context, known, _ = wrapper.build_inputs(batch)
    native = T0("raw", batch.variables, batch.variables, wrapper.checkpoint, wrapper.revision,
                context_days=2, arome="missing", arome_fields=["t", "pmer"], features=wrapper.feature_spec)
    raw_context, raw_known, _ = native.build_inputs(batch)
    np.testing.assert_array_equal(context[:, :2], batch.history - batch.arome_history)
    np.testing.assert_array_equal(context[:, 2:], raw_context[:, 2:])
    np.testing.assert_array_equal(known[:, :-1], raw_known)
    np.testing.assert_array_equal(known[:, -1, -2:], [[6., 9.]])
    np.testing.assert_array_equal(batch.history, original)
    seen = []

    def predict(x, **kwargs):
        seen.append(kwargs)
        q = np.broadcast_to(np.array(QUANTILE_LEVELS) - 2.5, (*x.shape[:2], 2, 21)).copy()
        return SimpleNamespace(quantiles=torch.tensor(q, dtype=torch.float32), quantile_levels=QUANTILE_LEVELS)

    forecaster = SimpleNamespace(config=SimpleNamespace(quantile_levels=QUANTILE_LEVELS), predict=predict)
    monkeypatch.setattr(models.MODELS, "get", lambda *_: SimpleNamespace(forecaster=forecaster, lock=threading.Lock(), info={}))
    result = wrapper.predict(batch)
    result.validate(batch)
    np.testing.assert_allclose(result.quantiles, batch.arome[..., None] + np.array(QUANTILE_LEVELS) - 2.5, atol=1e-4)
    np.testing.assert_allclose(result.point, batch.arome - 2.)
    assert not seen[0]["clip_nonnegative"] and not seen[0]["snap_integers"]
    changed = batch.arome.copy()
    changed[0, 0, 0] += 1
    assert wrapper.input_identity(batch) != wrapper.input_identity(replace(batch, arome=changed))
    assert wrapper.input_identity(batch) != native.input_identity(batch)


def test_unavailable_residuals_and_runs_fail_before_loading(monkeypatch):
    batch, wrapper = inputs(), adapter()
    monkeypatch.setattr(models.MODELS, "get", lambda *_: pytest.fail("invalid inputs loaded weights"))
    for changed in (replace(batch, arome=None), replace(batch, arome_history=None),
                    replace(batch, arome_history=np.full_like(batch.history, np.nan)),
                    replace(batch, arome=np.full_like(batch.arome, np.nan)),
                    replace(batch, cases=batch.cases.assign(arome_available_at="2026-10-21T07:00Z"))):
        with pytest.raises(ValueError):
            wrapper.predict(changed)
