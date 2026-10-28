"""Local matrix orchestration, exact common supports and explicit later publication."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
import yaml

from weather_lab import local_benchmark as local
from weather_lab.common import read_json, sha256_file, write_json


def test_declared_matrix_has_39_distinct_supported_configurations():
    config = yaml.safe_load(local.CONFIG.read_text())
    configs = local.configurations(config)
    assert len(configs) == len({c.as_dict().__str__() for _, c in configs}) == 39
    assert {c.model for _, c in configs} == {'Timestral-28M', 'Timestral-209M', 't0-gamma'}
    assert all(c.context_days == 30 and c.horizon_h == 48 and c.step_hours == 1 for _, c in configs)
    assert {'raw', 'calendar', 'solar', 'tendencies'} <= {n.split('/')[1] for n, _ in configs}


def test_local_resume_common_support_and_publication_order(tmp_path, monkeypatch):
    config = yaml.safe_load(local.CONFIG.read_text())
    config.update(suite='fixture', checkpoints=['Timestral-28M'], setups=config['setups'][:2])
    source = tmp_path / 'source'
    source.mkdir()
    cases = [{'case_id': f'c{i}', 'station': '07149', 'origin': f'2026-01-0{i+1}T00:00:00Z',
              'split': 'validation', 'season': 'DJF', 'eligible': True} for i in range(2)]
    write_json(source / 'manifest.json', {'cases': cases})
    write_json(source / 'run.json', {'suite': 'fixture', 'archive_id': 'fixture-archive'})
    write_json(source / 'index.json', {'archive_id': 'fixture-archive', 'variables': ['t','pmer'],
               'context_hours': 720, 'horizon_hours': 51, 'observation_delay_hours': 3, 'retrieval': {'stride_hours': 1}})
    rows, targets = [], []
    for i, case in enumerate(cases):
        for variable, value in [('t', 10. + i), ('pmer', 1000. + i)]:
            row = {'case_id': case['case_id'], 'station': case['station'], 'origin': pd.Timestamp(case['origin']),
                   'valid_time': pd.Timestamp(case['origin']) + pd.Timedelta(hours=1), 'variable': variable, 'lead_h': 1}
            targets.append({k: row[k] for k in local.KEYS} | {'observed': value})
            rows.append(row | {'point': value + 1, 'model': 'arome-raw'})
    ar = pd.DataFrame(rows)
    ar.to_parquet(source / 'arome_reference.parquet', index=False)
    persistence = ar.assign(model='persistence', point=ar.point - 2)
    persistence.to_parquet(source / 'predictions.parquet', index=False)
    pd.DataFrame(targets).to_parquet(source / 'targets.parquet', index=False)
    monkeypatch.setattr(local, 'Store', lambda: type('Store', (), {'get': lambda *args: source})())
    monkeypatch.setattr(local.models, 'prefetch', lambda *a, **k: None)
    monkeypatch.setattr(local.models.MODELS, 'release', lambda: 0)
    calls = []
    def fake_run(source, cfg, predict, progress, root):
        calls.append(cfg)
        p = Path(root) / ('calendar' if cfg.calendar else 'raw')
        p.mkdir(parents=True)
        frame = ar.assign(model='native', point=ar.point - 1 if cfg.calendar else ar.point + 1)
        if cfg.calendar:
            frame.loc[frame.case_id.eq('c1'), 'point'] = np.nan
        frame.to_parquet(p / 'predictions.parquet', index=False)
        for name in ('targets.parquet', 'manifest.json'):
            shutil.copy2(source / name, p / name)
        write_json(p / 'run.json', {'run_id': p.name, 'source_suite': 'fixture', 'forecast_config': cfg.as_dict(),
                    'created_at': 'synthetic fixture only', 'systems': [{'id': 'native'}],
                    'files_sha256': {f.name: sha256_file(f) for f in p.iterdir()}})
        return p
    monkeypatch.setattr(local.development, 'run', fake_run)
    out = tmp_path / 'local-results'
    found = local.run(config, out, log=lambda *a, **k: None)
    assert len(calls) == len(found) == 2
    monkeypatch.setattr(local.development, 'location', lambda: out / 'runs')
    assert all('Timestral-28M/' in label for label, _ in local.development.saved_runs('fixture'))
    support = pd.read_parquet(out / 'comparisons/common_support.parquet')
    assert set(support.case_id) == {'c0'}
    scores = pd.read_csv(out / 'comparisons/shared_arome_metrics.csv')
    common = scores[scores.station.eq('ALL') & scores.season.eq('ALL') & scores.lead_h.isna()]
    assert set(common.n) == {1} and common.support_sha256.nunique() == 1
    assert common[common.model.eq('Timestral-28M/raw')].mae.eq(2).all()
    assert common[common.model.eq('Timestral-28M/calendar')].mae.eq(0).all()
    assert common[common.model.eq('arome-raw')].mae.eq(1).all()
    local.run(config, out, log=lambda *a, **k: None)
    assert len(calls) == 2  # resume uses verified saved outputs; never reruns completed setups
    with pytest.raises(ValueError, match='frozen'):
        local.run(config | {'context_days': 31}, out, log=lambda *a, **k: None)
    uploads = []
    monkeypatch.setattr(subprocess, 'run', lambda argv, check: uploads.append(argv))
    scores_path = out / 'comparisons/shared_arome_metrics.csv'
    saved_scores = scores_path.read_bytes()
    scores_path.write_bytes(b'partial fixture report')
    with pytest.raises(ValueError, match='checksum'):
        local.publish(config, out)
    assert uploads == []
    scores_path.write_bytes(saved_scores)
    local.publish(config, out)
    assert len(uploads) == 3
    assert uploads[0][-2:] == ['--exclude', '*/run.json']
    assert uploads[1][-2:] == ['--include', '*/run.json']
    assert '/benchmark-matrices/' in uploads[2][-1]
    p = next(iter(found.values())) / 'predictions.parquet'
    p.write_bytes(b'corrupt fixture')
    with pytest.raises(ValueError, match='verification'):
        local.completed(out, config, local.configurations(config))
