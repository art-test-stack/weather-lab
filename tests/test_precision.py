"""Scoped precision, provenance identity and named residual-history compatibility."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from weather_lab.forecast_config import ForecastConfig, build_adapter
from weather_lab.precision import inference


def test_fp32_math_caps_positional_and_keyword_casts_without_global_changes():
    x = torch.tensor([1000., 1000.25], dtype=torch.float32)
    with torch.inference_mode(), inference('fp32', 'cpu') as guard:
        y = x.to(torch.float64)
        z = torch.ones(2, dtype=torch.float64)
        assert y.dtype == z.dtype == torch.float32
        assert guard.narrowed_fp64_ops >= 2
        np.testing.assert_array_equal(y.numpy(), x.numpy())
    assert x.double().dtype == torch.float64
    with inference('native', 'cpu') as guard:
        assert guard is None and x.double().dtype == torch.float64


def test_bf16_compute_preserves_fp32_outputs_and_precision_identity():
    x = torch.ones((4, 4), dtype=torch.float32)
    with inference('bf16', 'cpu'):
        out = x @ x
        assert out.dtype == torch.bfloat16
        assert out.float().dtype == torch.float32
    native = ForecastConfig.make('Timestral-28M', targets=('t','pmer'))
    mixed = replace(native, precision='bf16')
    assert 'precision' not in native.as_dict()
    assert mixed.as_dict()['precision']['compute'] == 'bfloat16 mixed'
    assert mixed.forecast_part() != native.forecast_part()
    model = build_adapter(mixed, 'test')
    assert model.params['precision'] == mixed.as_dict()['precision']
    from weather_lab.forecast_config import from_system
    record = model.describe() | {'revision': model.checkpoint + '@' + model.revision[:12]}
    assert from_system(record, mixed.horizon_h, mixed.step_hours) == mixed
    with pytest.raises(ValueError, match='precision'):
        ForecastConfig.make('Timestral-28M', precision='fp16')


def test_resid_v1_named_history_matches_legacy_and_refuses_ambiguous_lead():
    from test_t0_adapter import batch
    from weather_lab.arome_inputs import AromeInputs
    from weather_lab.features import build_inputs

    b = replace(batch(), arome=np.array([[[11.,11.],[1014.,1014.]]],np.float32))
    n, v, t = b.history.shape
    h = b.horizon
    history = np.broadcast_to(np.array([9.,1012.],np.float32)[None,:,None],b.history.shape).copy()
    times = b.history_times()
    runs = np.broadcast_to(times[:, None] - np.timedelta64(3,'h'), (n,v,t)).copy()
    available = runs + np.timedelta64(1,'h')
    future_runs = np.broadcast_to(b.origins[:,None,None] - np.timedelta64(3,'h'), (n,v,h)).copy()
    future_available = future_runs + np.timedelta64(1,'h')
    named = AromeInputs(tuple(b.variables), history, b.arome, runs, available, future_runs, future_available)
    old = replace(b, arome_history=history, arome_history_lead_h=np.full((n,t),3.), arome_history_run=runs[:,0])
    new = replace(b, arome_inputs=named, arome_history=None, arome_history_lead_h=None)
    a = build_inputs(['resid'], old, t)
    actual = build_inputs(['resid'], new, t)
    np.testing.assert_array_equal(a.past, actual.past)
    other_runs = runs.copy();other_runs[:,1] -= np.timedelta64(1,'h')
    ambiguous = replace(new, arome_inputs=replace(named, history_run=other_runs))
    with pytest.raises(ValueError, match='common archived forecast lead'):
        build_inputs(['resid'], ambiguous, t)


@pytest.mark.real_model
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires Apple MPS')
def test_cached_timestral_mps_precision_forecasts_are_finite_fp32():
    import pandas as pd
    from weather_lab import models
    from weather_lab.forecast import CaseBatch

    checkpoint = models.CHECKPOINTS['Timestral-28M']
    if models.local_dir(checkpoint) is None:
        pytest.skip('prefetch Timestral-28M explicitly before this device integration check')
    axis = np.arange(7 * 24, dtype=np.float32)
    history = np.stack([10 + np.sin(axis / 24), 1010 + np.cos(axis / 48)])[None]
    origin = np.array(['2026-01-15T00:00'], dtype='datetime64[ns]')
    future = origin[:, None] + np.arange(1, 49)[None] * np.timedelta64(1, 'h')
    batch = CaseBatch(pd.DataFrame({'station':['07149']}), ['t','pmer'], history, origin, future, 1)
    try:
        for policy in ('fp32', 'bf16'):
            cfg = ForecastConfig.make('Timestral-28M', context_days=7, horizon_h=48, step_hours=1,
                                       targets=('t','pmer'), precision=policy)
            result = build_adapter(cfg, 'device-test', device='mps').predict(batch)
            result.validate(batch)
            assert np.isfinite(result.point).all() and np.isfinite(result.quantiles).all()
            assert result.point.dtype == result.quantiles.dtype == np.float32
            assert result.info['narrowed_fp64_ops'] > 0
    finally:
        models.MODELS.release('mps')
