"""Hourly origins, causal year-boundary inputs and explicit unverified tails."""
import numpy as np
import pandas as pd
import pytest

from weather_lab.common import read_json
from weather_lab.forecast_config import ForecastConfig
from weather_lab.hourly_benchmark import configuration, inputs, prepare
from weather_lab.rag_store import Store


def test_hourly_suite_preserves_january_context_and_unverified_tail(tmp_path, monkeypatch):
    cfg = configuration()
    cfg['artifact_id'] = None
    cfg['stations'] = cfg['stations'][:1]
    store = Store(tmp_path / 'store')
    raw = tmp_path / 'raw'
    raw.mkdir()
    times = pd.date_range('2025-10-01', '2026-01-03', freq='h', tz='UTC')
    obs = pd.DataFrame({'station': '07149', 'time': times, 't': np.arange(len(times), dtype=float), 'pmer': 1010.})
    obs.to_parquet(raw / 'observations.parquet', index=False)
    run = pd.Timestamp('2025-12-31T03:00Z')
    rows = [{'station': '07149', 'run': run, 'available_at': run + pd.Timedelta(hours=3),
             'valid_time': time, 'variable': var, 'value': 10. if var == 't' else 1000., 'product': cfg['product'],
             'availability_basis': 'synthetic fixture only', 'source_kind': 'issued_run'}
            for time in pd.date_range(run, periods=120, freq='h') for var in cfg['variables']]
    issued = pd.DataFrame(rows)
    issued.to_parquet(raw / 'issued.parquet', index=False)
    cfg['archive_id'] = store.put('archives', raw).name
    out = prepare(cfg, store)
    manifest = read_json(out / 'manifest.json')
    cases = pd.DataFrame(manifest['cases'])
    cases['origin'] = pd.to_datetime(cases.origin, utc=True)
    assert len(cases) == 49
    assert cases.origin.diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert cases.origin.iloc[0] == pd.Timestamp('2026-01-01T00:00Z')
    assert cases.origin.iloc[-1] == times[-1]
    selected = ForecastConfig.make('Timestral-28M', context_days=30, horizon_h=48, step_hours=1, targets=('t', 'pmer'))
    case = cases.iloc[0]
    batch = inputs(obs, issued, case, selected)
    assert batch.history.shape == (1, 2, 720)
    assert pd.Timestamp(batch.history_times()[0, -1], tz='UTC') == case.origin - pd.Timedelta(hours=3)
    assert (pd.to_datetime(batch.history_times()[0], utc=True).year == 2025).all()
    changed = obs.copy()
    changed.loc[changed.time > case.origin - pd.Timedelta(hours=3), 't'] += 10000
    np.testing.assert_array_equal(inputs(changed, issued, case, selected).history, batch.history)
    truth = pd.read_parquet(out / 'targets.parquet')
    assert truth[truth.case_id.eq(cases.case_id.iloc[-1])].observed.isna().all()
    assert set(truth.lead_h) == set(range(1, 49))
    availability = pd.read_csv(out / 'case_availability.csv')
    assert availability.iloc[-1].verified_targets == 0
    assert read_json(out / 'run.json')['reference_only']
    assert store.get('suites', out.name) == out
    assert prepare(cfg | {"artifact_id": out.name}, store) == out
    with pytest.raises(ValueError, match="frozen"):
        prepare(cfg | {"artifact_id": out.name, "label": "changed protocol"}, store)

    from weather_lab import development
    from weather_lab.forecast import ForecastBatch, QUANTILE_LEVELS, Timestral

    seen = []
    def predict(self, batch):
        seen.append(batch.cases.iloc[0].origin)
        assert batch.horizon == 51
        assert (batch.history_times() <= batch.origins[:, None]).all()
        self.build_inputs(batch)
        point = np.zeros((1, 2, 51))
        point[:, 1] = 1010.
        return ForecastBatch(self.id, "synthetic fixture only", "quantiles", ["t", "pmer"], point,
                             point[..., None] + np.array(QUANTILE_LEVELS) - .5, QUANTILE_LEVELS)

    monkeypatch.setattr(Timestral, "predict", predict)
    monkeypatch.setattr(development.models, "prefetch", lambda *a, **k: None)
    saved = development.run(out, selected, root=tmp_path / "results")
    prediction = pd.read_parquet(saved / "predictions.parquet")
    native = prediction[prediction.model.str.startswith("timestral-dev")]
    assert len(seen) == 49 and len(native) == 49 * 2 * 48
    assert native.point.notna().all()
    assert native.lead_h.min() == 1 and native.lead_h.max() == 48
    assert read_json(saved / "run.json")["hourly_benchmark"]

    from dataclasses import replace
    from weather_lab import rag_index

    class FixtureIndex:
        ident = "f" * 64
        meta = {"archive_id": cfg["archive_id"], "context_hours": 720, "variables": ["t", "pmer"], "observation_delay_hours": 3,
                "retrieval": {"stride_hours": 1, "k": 1}}

        def retrieve(self, batch, method, k):
            return np.zeros((1, 2, 720 + batch.horizon), dtype=np.float32), [{"case_id": batch.cases.iloc[0].case_id}]

    monkeypatch.setattr(Store, "get", lambda *a: tmp_path / "synthetic-index")
    monkeypatch.setattr(rag_index, "load_index", lambda path: FixtureIndex())
    rag_cfg = replace(selected, retrieval_method="shape", retrieval_index="f" * 64)
    rag_saved = development.run(out, rag_cfg, root=tmp_path / "results")
    assert read_json(rag_saved / "run.json")["forecast_config"]["retrieval_index"] == "f" * 64
    assert len(pd.read_parquet(rag_saved / "predictions.parquet").query('model.str.startswith("timestral-dev")')) == 49 * 2 * 48

    residual_cfg = replace(selected, prediction_target="arome_error", conditioning="archived")
    residual_saved = development.run(out, residual_cfg, root=tmp_path / "results")
    residual = pd.read_parquet(residual_saved / "predictions.parquet")
    residual = residual[residual.model.str.startswith("timestral-dev") & residual.point.notna()]
    assert len(residual) == 49 * 2 * 48
    assert residual[residual.variable.eq("t")].arome_baseline.eq(10.).all()
    assert residual[residual.variable.eq("pmer")].arome_baseline.eq(1000.).all()
    assert residual.predicted_error.eq(residual.point - residual.arome_baseline).all()
    assert residual.lead_h.min() == 1 and residual.lead_h.max() == 48
