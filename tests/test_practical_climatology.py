from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from weather_lab.forecast import CaseBatch, ForecastBatch, QUANTILE_LEVELS
from weather_lab.practical_climatology import HarmonicClimatology


def training():
    times = pd.date_range("2018-01-01", "2018-12-31T21:00", freq="3h", tz="UTC")
    return pd.DataFrame({"station": "07149", "time": times,
                         "t": 10 + 5 * np.sin(2 * np.pi * times.day_of_year / 365.25),
                         "pmer": 1013., "u": -2., "v": 3.})


def test_fit_cutoff_identity_support_and_units():
    obs = training()
    fit = HarmonicClimatology.fit(obs, ["t", "pmer", "u", "v"], source_fingerprint="source1")
    assert fit.identity == HarmonicClimatology.fit(obs.iloc[::-1], fit.variables, source_fingerprint="source1").identity
    assert fit.identity != HarmonicClimatology.fit(obs, fit.variables, source_fingerprint="source2").identity
    changed = obs.copy()
    changed.loc[0, "t"] += 1
    assert fit.identity != HarmonicClimatology.fit(changed, fit.variables, source_fingerprint="source1").identity
    with pytest.raises(ValueError, match="training interval"):
        HarmonicClimatology.fit(obs.assign(time=obs.time + pd.DateOffset(years=5)), ["t"], source_fingerprint="x")
    with pytest.raises(ValueError, match="annual climatology support"):
        HarmonicClimatology.fit(obs.iloc[:100], ["t"], source_fingerprint="x")
    with pytest.raises(ValueError, match="units"):
        HarmonicClimatology.fit(obs, ["kelvin"], source_fingerprint="x")
    values = fit.values(["07149"], np.array([["2026-10-16T09:00", "2026-10-16T12:00"]], dtype="datetime64[ns]"))
    np.testing.assert_allclose(values[:, 1], 1013, atol=1e-10)
    np.testing.assert_allclose(values[:, 2], -2, atol=1e-10)


def test_anomaly_masks_inverse_every_quantile_and_chronology():
    fit = HarmonicClimatology.fit(training(), ["t", "pmer", "u", "v"], source_fingerprint="x")
    origin = np.array(["2026-10-16T06:00"], dtype="datetime64[ns]")
    future = origin[:, None] + np.arange(1, 3)[None, :] * np.timedelta64(3, "h")
    batch = CaseBatch(pd.DataFrame({"station": ["07149"]}), fit.variables,
                      np.array([[[12, np.nan], [1012, 1013], [-1, -2], [3, 4]]], dtype=np.float32),
                      origin, future, 3)
    transformed = fit.anomaly_batch(batch)
    np.testing.assert_array_equal(np.isnan(transformed.history), np.isnan(batch.history))
    np.testing.assert_allclose(transformed.history + fit.values(["07149"], batch.history_times()), batch.history, atol=1e-6)
    q = np.broadcast_to(np.linspace(-3, 3, len(QUANTILE_LEVELS)), (1, 4, 2, len(QUANTILE_LEVELS))).copy()
    result = fit.restore(ForecastBatch("anomaly", "x", "quantiles", fit.variables, q[..., 10], q, QUANTILE_LEVELS), batch)
    result.validate(batch)
    baseline = fit.values(["07149"], future)
    np.testing.assert_allclose(result.quantiles - q, np.broadcast_to(baseline[..., None], q.shape))
    assert (np.diff(result.quantiles, axis=-1) >= 0).all()
    assert result.info["transform_fit_identity"] == fit.identity
    with pytest.raises(ValueError, match="overlap"):
        old = np.array(["2022-01-01"], dtype="datetime64[ns]")
        fit.anomaly_batch(replace(batch, origins=old, future_times=old[:, None] + np.arange(1, 3)[None, :] * np.timedelta64(3, "h")))
