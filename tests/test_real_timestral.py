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
