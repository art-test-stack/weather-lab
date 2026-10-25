"""Context-enrichment features: availability, no future leakage, provenance and cache identity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from weather_lab import features as feats
from weather_lab.forecast import CaseBatch

ORIGIN = pd.Timestamp("2026-10-21T06:00Z")
STEP = 3


def _batch(t_hist=16, h=8, with_arome=True, aux=None):
    hist_times = pd.date_range(end=ORIGIN, periods=t_hist, freq="3h")
    history = np.stack([np.arange(t_hist, dtype=np.float32) + 10, np.full(t_hist, 1015, np.float32),
                        np.zeros(t_hist, np.float32), np.zeros(t_hist, np.float32)])[None]
    cases = pd.DataFrame({"case_id": ["c1"], "station": ["07149"], "origin": [ORIGIN], "arome_run": ["x"]})
    origins = np.array([ORIGIN.tz_convert(None).to_datetime64()]).astype("datetime64[ns]")
    future = origins[:, None] + (np.arange(1, h + 1) * np.timedelta64(STEP, "h"))[None]
    kw = {}
    if with_arome:
        ah = np.full((1, 4, t_hist), np.nan, np.float32)
        runs = np.full((1, t_hist), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        leads = np.full((1, t_hist), np.nan)
        for k in range(t_hist - 6, t_hist):              # only the last 6 history steps have an archived run
            ah[0, 0, k], ah[0, 1, k] = history[0, 0, k] - 0.5, 1014.0
            runs[0, k] = (hist_times[k] - pd.Timedelta(hours=9)).tz_convert(None).to_datetime64()
            leads[0, k] = 9.0
        kw = {"arome": np.full((1, 4, h), 1.0, np.float32), "arome_history": ah, "arome_history_run": runs,
              "arome_history_lead_h": leads}
    if aux is not None:
        kw.update(aux_history=aux, aux_variables=feats.AUX_VARIABLES)
    return CaseBatch(cases, ["t", "pmer", "u", "v"], history, origins, future, STEP,
                     station_coords={"07149": (48.7167, 2.3844)}, **kw)


def test_resid_rows_exist_only_where_a_qualifying_run_and_past_observation_exist():
    past, known, rep = feats.build(["resid"], _batch(), 16)
    assert known is None and past.shape == (1, 3, 16)
    np.testing.assert_allclose(past[0, 0, -6:], 0.5)               # obs − archived AROME
    np.testing.assert_allclose(past[0, 1, -6:], 1.0)
    np.testing.assert_allclose(past[0, 2, -6:], 9.0)               # lead made explicit
    assert np.isnan(past[0, :, :-6]).all()                         # nothing invented where no run qualifies
    assert rep["observed_fraction"]["resid_t"] == pytest.approx(6 / 16, abs=1e-4)
    with pytest.raises(ValueError):
        feats.build(["resid"], _batch(with_arome=False), 16)


def test_aux_history_never_contains_values_after_the_origin():
    from weather_lab.evaluate import make_batch

    times = pd.date_range(ORIGIN - pd.Timedelta(days=3), ORIGIN + pd.Timedelta(days=3), freq="3h")
    obs = pd.DataFrame({"station": "07149", "time": times, "t": 10.0, "pmer": 1015.0, "u": 0.0, "v": 0.0})
    aux = pd.DataFrame({"station": "07149", "time": times, "td": np.where(times > ORIGIN, 999.0, 5.0),
                        "cloud": np.where(times > ORIGIN, 999.0, 50.0)})

    class Cfg(dict):
        pass

    cfg = Cfg(forecast={"step_hours": 3, "horizon_steps": 8}, variables=["t", "pmer", "u", "v"])
    cases = pd.DataFrame({"case_id": ["c1"], "station": ["07149"], "origin": [ORIGIN]})
    batch, targets = make_batch(cfg, obs, cases, 16, None, aux, {"07149": (48.7, 2.4)})
    assert batch.aux_history.shape == (1, 2, 16) and np.nanmax(batch.aux_history) < 999
    past, _, _ = feats.build(["atmos"], batch, 16)
    assert np.nanmax(past) < 999


def test_solar_elevation_is_astronomical_and_known_in_advance():
    lat, lon = 48.7167, 2.3844                                      # Paris-Orly
    noon = np.array([np.datetime64("2026-06-21T12:00")])
    night = np.array([np.datetime64("2026-06-21T00:00")])
    assert feats.solar_elevation_sin(noon, lat, lon)[0] == pytest.approx(np.sin(np.deg2rad(64.7)), abs=0.02)
    assert feats.solar_elevation_sin(night, lat, lon)[0] < 0
    _, known, rep = feats.build(["solar"], _batch(with_arome=False), 16)
    assert known.shape == (1, 1, 16 + 8) and rep["observed_fraction"]["sin_solar_elevation"] == 1.0


def test_tendencies_are_differences_of_the_target_history():
    past, _, _ = feats.build(["tend"], _batch(with_arome=False), 16)
    np.testing.assert_allclose(past[0, 0, 1:], 1.0)                 # dt_3h of a +1 per step series
    np.testing.assert_allclose(past[0, 1, 8:], 8.0)                 # dt_24h = 8 steps
    assert np.isnan(past[0, 1, :8]).all()


def test_feature_definitions_enter_provenance_and_unknown_groups_fail():
    assert feats.definition(["solar"]) == [{"name": "solar", "version": 1, "role": "known",
                                             "rows": ["sin_solar_elevation"]}]
    with pytest.raises(ValueError):
        feats.validate(["radiation"])


def test_exploratory_cache_key_changes_with_features():
    from weather_lab import predcache

    base = {"checkpoint": "x", "revision": "r"}
    k0 = predcache.make_key(base, {"context": np.zeros(3)})
    k1 = predcache.make_key({**base, "features": feats.definition(["tend"])},
                            {"context": np.zeros(3), "past_covariates": np.ones(3)})
    k2 = predcache.make_key({**base, "features": feats.definition(["tend"])},
                            {"context": np.zeros(3), "past_covariates": np.array([1, np.nan, 1])})
    assert len({k0, k1, k2}) == 3                                   # definition and actual values/masks matter


@pytest.mark.real_model
def test_past_covariates_reach_the_pinned_model():
    """A non-constant past covariate changes the forecast. A fully missing row is masked but NOT exactly
    neutral: any covariate row switches Timestral to its covariate code path (measured: ≤ 0.01 on a series
    of amplitude ~1 around 10), which matters for early cases whose AROME-error history is empty."""
    from weather_lab.forecast import Timestral
    from weather_lab.models import CHECKPOINTS, MODELS

    ck = CHECKPOINTS["Timestral-5M"]
    fc = MODELS.get(ck, "cpu").forecaster
    rng = np.random.default_rng(0)
    ctx = (10 + np.sin(np.arange(240) / 8) + 0.1 * rng.standard_normal(240)).astype(np.float32)[None, None]
    base = fc.forecast(ctx, prediction_length=16, clip_nonnegative=False).quantiles.numpy()
    pc = rng.standard_normal((1, 1, 240)).astype(np.float32)
    with_pc = fc.forecast(ctx, prediction_length=16, past_covariates=pc, clip_nonnegative=False).quantiles.numpy()
    empty = fc.forecast(ctx, prediction_length=16, past_covariates=np.full((1, 1, 240), np.nan, np.float32),
                        clip_nonnegative=False).quantiles.numpy()
    d_pc, d_empty = np.abs(with_pc - base).max(), np.abs(empty - base).max()
    assert d_empty < 0.02                       # masked: only the code-path effect remains
    assert d_pc > d_empty                       # real values change the forecast more than an empty row
