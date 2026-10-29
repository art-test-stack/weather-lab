"""Native 24-hour calls generate eight steps from a full ninety-day context."""
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

spec = importlib.util.spec_from_file_location("native24_extra_adapters", ROOT / "experiments/error-features-v1/adapters.py")
extra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extra)


@pytest.mark.parametrize("family", ["timestral", "t0"])
def test_ninety_day_native24_inputs_and_generation(monkeypatch, family):
    origins = np.array(["2026-10-21T06"], dtype="datetime64[ns]")
    future = origins[:, None] + np.arange(1, 9) * np.timedelta64(3, "h")
    history_times = origins[:, None] + np.arange(-719, 1) * np.timedelta64(3, "h")
    history = np.stack([10 + np.sin(np.arange(720)), 1000 + np.sin(np.arange(720)),
                        np.sin(np.arange(720)), np.cos(np.arange(720))])[None].astype(np.float32)
    baseline = np.array([[[12.] * 8, [1000.] * 8, [0.] * 8, [0.] * 8]], np.float32)
    cases = pd.DataFrame({"case_id": ["c"], "station": ["07149"], "arome_run": ["2026-10-21T03:00Z"]})
    batch = CaseBatch(cases, ["t", "pmer", "u", "v"], history, origins, future, 3,
        arome=baseline, arome_history=history + .5, arome_history_run=history_times - np.timedelta64(6, "h"),
        arome_history_lead_h=np.full((1, 720), 6.), aux_history=np.array([[[5.] * 720, [50.] * 720]], np.float32),
        aux_variables=["td", "cloud"], station_coords={"07149": (48.7, 2.4)})
    issued = pd.DataFrame([{"station": "07149", "run": pd.Timestamp("2026-10-21T03:00Z"),
        "available_at": pd.Timestamp("2026-10-21T06:00Z"), "valid_time": pd.Timestamp(t, tz="UTC"),
        "variable": field, "value": value, "product": "meteofrance_arome_france"}
        for t in future[0] for field, value in zip(extra.FIELDS, (60., 75., 100.))])
    ck = models.CHECKPOINTS["Timestral-209M" if family == "timestral" else "t0-gamma"]
    cls = extra.ExtraTimestral if family == "timestral" else extra.ExtraT0
    kwargs = {"prediction_target": "arome_error"} if family == "timestral" else {}
    adapter = cls(f"{family}-error-both-h24-v1", batch.variables, ["t", "pmer"], ck.repo, ck.revision,
        context_days=90, calendar=False, arome="missing", arome_fields=["t", "pmer"],
        features=["resid", "atmos", "solar", "tend"], extras=extra.ExtraInputs(True, issued),
        extra_identity={"horizon_hours": 24}, **kwargs)
    calls = []

    def forecast(context, prediction_length, **options):
        calls.append((prediction_length, context.shape, options["known_covariates"].shape))
        q = np.broadcast_to(np.array(QUANTILE_LEVELS) - 2.5, (len(context), 2, prediction_length, 21)).copy()
        return SimpleNamespace(quantiles=torch.tensor(q, dtype=torch.float32))

    def predict(context, horizon, **options):
        calls.append((horizon, context.shape, options["future_covariates"].shape))
        q = np.broadcast_to(np.array(QUANTILE_LEVELS) - 2.5, (*context.shape[:2], horizon, 21)).copy()
        return SimpleNamespace(quantiles=torch.tensor(q, dtype=torch.float32), quantile_levels=QUANTILE_LEVELS)

    fc = SimpleNamespace(forecast=forecast, predict=predict, quantile_levels=QUANTILE_LEVELS,
                         config=SimpleNamespace(quantile_levels=QUANTILE_LEVELS))
    monkeypatch.setattr(models.MODELS, "get", lambda *_: SimpleNamespace(forecaster=fc, lock=threading.Lock(), info={}))
    result = adapter.predict(batch)
    result.validate(batch)
    assert calls[0][0] == 8 and calls[0][1][-1] == 720 and calls[0][2][-1] == 728
    frame = result.to_frame(batch, "native24")
    assert len(frame) == 16 and frame.lead_h.max() == 24
    np.testing.assert_allclose(result.point, baseline[:, :2] - 2.)
