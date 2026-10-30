"""AROME error retrieval: native conditioning, time eligibility and future-label isolation."""
from dataclasses import replace
import copy
import pickle

import numpy as np
import pandas as pd
import pytest

from weather_lab.common import ROOT
from weather_lab.forecast_config import ForecastConfig, build_adapter, check
from weather_lab.rag import config
from weather_lab.rag_archive import ingest
from weather_lab.rag_arome_error import ErrorIndex, build, make_batch
from weather_lab.rag_index import load_index
from weather_lab.rag_store import Store


@pytest.fixture
def error_cases(tmp_path):
    cfg = copy.deepcopy(config(ROOT / "configs/rag_arome_error_v1.yaml"))
    cfg.update(context_hours=24, stations=["07149"])
    cfg["retrieval"].update(query_hours=12, forecast_query_hours=6, k=2)
    times = pd.date_range("2025-11-01T00:00Z", "2026-05-01T23:00Z", freq="h")
    x = np.arange(len(times))
    obs = pd.DataFrame({"station": "07149", "time": times,
                        "t": 12 + 2 * np.sin(x / 40), "pmer": 1010 + 3 * np.cos(x / 80)})
    rows = []
    for run in pd.date_range("2026-04-02T03:00Z", "2026-04-26T03:00Z", freq="D"):
        valid = pd.date_range(run + pd.Timedelta(hours=1), periods=51, freq="h")
        truth = obs.set_index("time").reindex(valid)
        for variable, bias in (("t", .5), ("pmer", -2.)):
            rows.extend({"station": "07149", "run": run, "available_at": run + pd.Timedelta(hours=3),
                         "valid_time": t, "variable": variable, "value": value - bias,
                         "product": cfg["product"], "source_kind": "issued_run",
                         "availability_basis": "synthetic_fixture_only"}
                        for t, value in truth[variable].items())
    issued = pd.DataFrame(rows)
    source, forecasts = tmp_path / "obs.parquet", tmp_path / "issued.parquet"
    obs.to_parquet(source, index=False)
    issued.to_parquet(forecasts, index=False)
    store = Store(tmp_path / "store")
    archive = ingest(source, cfg, issued=forecasts, store=store)
    index = ErrorIndex(build(archive.name, cfg, store))
    return cfg, obs, issued, store, archive, index


def query(cfg, obs, issued, origin="2026-04-24T06:00Z"):
    case = pd.Series({"station": "07149", "origin": pd.Timestamp(origin)}, name="query")
    return make_batch(obs, issued, case, cfg["variables"], cfg["context_hours"], 24, 3, cfg["product"])[0]


def test_error_payload_and_native_arome_conditioning_for_both_families(error_cases):
    cfg, obs, issued, store, archive, index = error_cases
    batch = query(cfg, obs, issued)
    assert batch.arome is batch.arome_inputs.future
    rows, audit = index.retrieve(batch, "shape", 2)
    np.testing.assert_allclose(rows[0, :, -27:], np.repeat([[.5], [-2.]], 27, axis=1), atol=1e-4)
    origin = pd.Timestamp(batch.decision_origins[0], tz="UTC")
    assert all(pd.Timestamp(n["available_at"]) < origin for n in audit[0]["neighbors"])
    assert audit[0]["product"] == cfg["product"]
    assert set(audit[0]["neighbors"][0]["mae_by_variable"]) == {"t", "pmer"}
    assert isinstance(load_index(index.path), ErrorIndex)
    assert store.list("indexes") == [(f"AROME errors · {index.ident[:12]}", index.ident)]
    for name in ("Timestral-209M", "t0-gamma"):
        request = ForecastConfig.make(name, 'archived', context_days=1, targets=('t', 'pmer'), step_hours=1,
                                      retrieval_method='shape', retrieval_index=index.ident)
        model = build_adapter(request, f'{name.lower()}-rag-shape', index=index, batch_size=8,
                              arome_fields=['t', 'pmer'])
        native = model.build_inputs(batch)
        np.testing.assert_array_equal(native[0][:, :2], batch.history)
        np.testing.assert_allclose(native[1][:, 4:6, -27:], batch.arome)
        np.testing.assert_allclose(native[1][:, -2:], rows, equal_nan=True)
        assert model.params["retrieval_kind"] == "arome_error"
    restored = pickle.loads(pickle.dumps(index))
    np.testing.assert_allclose(restored.retrieve(batch, "shape", 2)[0], rows, equal_nan=True)


def test_future_outcomes_and_late_runs_cannot_change_earlier_retrieval(error_cases, tmp_path):
    cfg, obs, issued, store, archive, index = error_cases
    batch = query(cfg, obs, issued)
    expected, audit = index.retrieve(batch, "shape", 2)
    changed = obs.copy()
    changed.loc[changed.time > pd.Timestamp(batch.origins[0], tz="UTC"), cfg["variables"]] += 100
    late = issued.copy()
    late.loc[late.available_at > pd.Timestamp(batch.decision_origins[0], tz="UTC"), "value"] += 50
    source, forecasts = tmp_path / "changed.parquet", tmp_path / "late.parquet"
    changed.to_parquet(source, index=False)
    late.to_parquet(forecasts, index=False)
    other_archive = ingest(source, cfg, issued=forecasts, store=store)
    other = ErrorIndex(build(other_archive.name, cfg, store))
    altered_batch = query(cfg, changed, late)
    np.testing.assert_array_equal(batch.history, altered_batch.history)
    np.testing.assert_array_equal(batch.arome, altered_batch.arome)
    np.testing.assert_array_equal(index.scale, other.scale)
    actual, altered_audit = other.retrieve(altered_batch, "shape", 2)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert audit[0]["neighbors"] == altered_audit[0]["neighbors"]


def test_unverified_case_missing_arome_and_wrong_station_are_refused(error_cases):
    cfg, obs, issued, store, archive, index = error_cases
    batch = query(cfg, obs, issued)
    index.windows.loc[:, "available_at"] = pd.Timestamp(batch.decision_origins[0], tz="UTC")
    with pytest.raises(ValueError, match="fully verified earlier"):
        index.retrieve(batch, "shape", 2)
    with pytest.raises(ValueError, match="issued AROME"):
        index.retrieve(replace(batch, arome=None), "shape", 2)
    with pytest.raises(ValueError, match="single available issued run"):
        query(cfg, obs, issued[issued.available_at > pd.Timestamp("2026-04-24T06:00Z")])
    wrong = replace(batch, cases=batch.cases.assign(station="07630"))
    with pytest.raises(ValueError, match="fully verified earlier"):
        index.retrieve(wrong, "shape", 2)
    wrong_product = replace(batch, cases=batch.cases.assign(arome_product="meteofrance_arome_france_hd"))
    with pytest.raises(ValueError, match="exact AROME product"):
        index.retrieve(wrong_product, "shape", 2)


def test_t0_residual_identity_includes_baseline_without_conditioning(error_cases):
    cfg, obs, issued, store, archive, index = error_cases
    batch = query(cfg, obs, issued)
    request = ForecastConfig.make('t0-gamma', context_days=1, targets=('t', 'pmer'), step_hours=1,
                                  prediction_target='arome_error')
    model = build_adapter(request, 't0-gamma', batch_size=8)
    changed = batch.arome.copy() + 1
    other = replace(batch, arome=changed, arome_inputs=replace(batch.arome_inputs, future=changed))
    original_native, other_native = model.build_inputs(batch), model.build_inputs(other)
    np.testing.assert_allclose(original_native[0], other_native[0], equal_nan=True)
    np.testing.assert_allclose(original_native[1], other_native[1], equal_nan=True)
    assert model.input_identity(batch) != model.input_identity(other)


def test_conditioned_forecast_lookup_has_distinct_cache_identity(error_cases, monkeypatch):
    from weather_lab import predcache, rag_forecast

    cfg, obs, issued, store, archive, _ = error_cases
    cfg["context_hours"] = 720
    index = ErrorIndex(build(archive.name, cfg, store))
    case = pd.Series({"station": "07149", "origin": pd.Timestamp("2026-04-24T06:00Z")}, name="query")
    selected = ForecastConfig.make("t0-gamma", "archived", context_days=30, horizon_h=24,
                                   targets=("t", "pmer"), step_hours=1,
                                   retrieval_method="shape", retrieval_index=index.ident)
    assert check(selected) == []
    monkeypatch.setattr(predcache, "lookup", lambda *a: None)
    monkeypatch.setattr(rag_forecast, "prefetch", lambda *a, **k: pytest.fail("model operation on a cache lookup"))
    result, info = rag_forecast.forecast(case, selected, lookup_only=True, store=store)
    assert result is None and info["settings"]["retrieval_kind"] == "arome_error"
    assert info["settings"]["arome_provenance"]["run"] == "2026-04-24 03:00:00+00:00"
    _, other = rag_forecast.forecast(case, replace(selected, retrieval_method="level"), lookup_only=True, store=store)
    assert other["key"] != info["key"]
    with pytest.raises(ValueError, match="requires archived AROME"):
        rag_forecast.forecast(case, replace(selected, conditioning="none"), lookup_only=True, store=store)
    assert check(replace(selected, conditioning="substobs"))


@pytest.mark.parametrize("model_name", ["Timestral-209M", "t0-gamma"])
def test_residual_rag_shifts_quantiles_once_and_excludes_bridge(error_cases, tmp_path, monkeypatch, model_name):
    import threading
    from types import SimpleNamespace
    import torch
    from weather_lab import models, predcache, rag_forecast
    from weather_lab.forecast import QUANTILE_LEVELS

    cfg, obs, issued, store, archive, _ = error_cases
    cfg["context_hours"] = 720
    index = ErrorIndex(build(archive.name, cfg, store))
    batch = query(cfg, obs, issued)
    errors = batch.history - batch.arome_inputs.history
    calls = []
    def generate(context, **kwargs):
        calls.append(kwargs)
        np.testing.assert_allclose(context[:, :2], errors, equal_nan=True)
        h = kwargs.get("horizon", kwargs.get("prediction_length"))
        quantiles = np.broadcast_to(.2 + np.array(QUANTILE_LEVELS) - .5,
                                   (*context.shape[:2], h, len(QUANTILE_LEVELS))).copy().astype(np.float32)
        return SimpleNamespace(quantiles=torch.from_numpy(quantiles), quantile_levels=QUANTILE_LEVELS)
    native = SimpleNamespace(config=SimpleNamespace(quantile_levels=QUANTILE_LEVELS),
                             quantile_levels=QUANTILE_LEVELS, predict=generate, forecast=generate)
    monkeypatch.setattr(models.MODELS, "get", lambda *a: SimpleNamespace(forecaster=native, lock=threading.Lock(), info={}))
    monkeypatch.setattr(rag_forecast, "prefetch", lambda *a, **k: None)
    monkeypatch.setattr(predcache, "location", lambda: (tmp_path / "predictions", "test"))
    case = pd.Series({"station": "07149", "origin": pd.Timestamp("2026-04-24T06:00Z")}, name="residual-case")
    selected = ForecastConfig.make(model_name, "archived", context_days=30, horizon_h=24,
                                   targets=("t", "pmer"), step_hours=1, prediction_target="arome_error",
                                   retrieval_method="shape", retrieval_index=index.ident)
    assert not check(selected)
    frame, info = rag_forecast.forecast(case, selected, store=store)
    assert len(calls) == 1 and len(frame) == 48
    assert frame.lead_h.min() == 1 and frame.lead_h.max() == 24
    np.testing.assert_allclose(frame.predicted_error, .2, atol=1e-4)
    np.testing.assert_allclose(frame["q0.90"] - frame["q0.10"], .8, atol=1e-4)
    expected = batch.arome[0][:, batch.lead_hours[0] > 0].ravel()
    np.testing.assert_allclose(frame.arome_baseline, expected)
    assert info["settings"]["prediction_target"] == "arome_error" and info["settings"]["error_history"]
    cached, hit = rag_forecast.forecast(case, selected, lookup_only=True, store=store)
    assert hit["cache_hit"] and len(calls) == 1
    pd.testing.assert_frame_equal(frame, cached)


def test_error_index_uses_hourly_query_windows_and_strict_verification(error_cases):
    cfg, obs, issued, store, archive, _ = error_cases
    hourly = copy.deepcopy(cfg)
    hourly['retrieval']['stride_hours'] = 1
    hourly['horizons_hours'] = [24]
    # A short fixture interval is enough to cover all hourly decisions between daily issues.
    hourly['retrieval']['start'] = '2026-04-25T00:00Z'
    index = ErrorIndex(build(archive.name, hourly, store))
    assert 7 in set(index.windows.origin.dt.hour)
    assert not index.windows.duplicated(['station', 'origin']).any()
    assert (index.windows.payload_end + pd.Timedelta(hours=3)).eq(index.windows.available_at).all()
    batch = query(cfg, obs, issued, '2026-04-27T06:00Z')
    _, audit = index.retrieve(batch, 'shape', 2)
    assert all(pd.Timestamp(n['available_at']) < pd.Timestamp('2026-04-27T06:00Z')
               for n in audit[0]['neighbors'])
