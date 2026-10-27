"""Resumable issue collection reuses observations and preserves unavailable requests."""
import json
from types import SimpleNamespace

import pandas as pd

from weather_lab import rag_archive
from weather_lab.common import ROOT, read_json, sha256_file
from weather_lab.rag import config
from weather_lab.rag_store import Store


def test_extension_reuses_saved_responses_and_records_real_gaps(tmp_path, monkeypatch):
    cfg = config(ROOT / 'configs/rag_weather_hourly_v2.yaml')
    cfg['collection']['download_cap_mib'] = 1
    store = Store(tmp_path / 'store')
    raw = tmp_path / 'base'
    raw.mkdir()
    times = pd.date_range('2026-04-01T00:00Z', '2026-04-02T12:00Z', freq='h', tz='UTC')
    obs = pd.concat([pd.DataFrame({'station': s['id'], 'time': times, 't': 10., 'pmer': 1010.}) for s in cfg['stations']])
    obs.to_parquet(raw / 'observations.parquet', index=False)
    product = cfg['product']
    run = pd.Timestamp('2026-04-02T03:00Z')
    issued = pd.DataFrame([{'station': s['id'], 'product': product, 'run': run,
                           'available_at': run + pd.Timedelta(hours=3), 'valid_time': run,
                           'variable': v, 'value': 10., 'source_kind': 'issued_run', 'availability_basis': 'fixture'}
                          for s in cfg['stations'] for v in cfg['variables']])
    issued.to_parquet(raw / 'issued.parquet', index=False)
    (raw / 'archive.json').write_text('{}')
    base = store.put('archives', raw)
    original = {p.name: sha256_file(p) for p in base.iterdir()}
    calls = []
    def get(self, endpoint, params, timeout):
        calls.append(params['run'])
        origin = pd.Timestamp(params['run'], tz='UTC')
        missing = origin.hour == 9
        data = {'error': True, 'reason': 'synthetic unavailable issue'} if missing else [
            {'utc_offset_seconds': 0, 'latitude': s['lat'], 'longitude': s['lon'],
             'hourly_units': {'temperature_2m': '°C', 'pressure_msl': 'hPa'},
             'hourly': {'time': [str(t) for t in pd.date_range(origin, periods=52, freq='h')],
                        'temperature_2m': [10.] * 52, 'pressure_msl': [1010.] * 52}}
            for s in cfg['stations']]
        content = json.dumps(data).encode()
        return SimpleNamespace(status_code=400 if missing else 200, content=content, url=endpoint,
                               json=lambda: data)
    monkeypatch.setattr(rag_archive.requests.Session, 'get', get)
    monkeypatch.setattr(rag_archive.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(rag_archive, 'ROOT', tmp_path)
    out = rag_archive.extend_issued(base.name, cfg, store, log=lambda *a, **k: None)
    assert len(calls) == 3 and not any(s.endswith('03:00') for s in calls)
    actual = pd.read_parquet(out / 'issued.parquet')
    assert set(actual.run.dt.hour) == {0, 3, 6}
    assert not actual.duplicated(['station', 'product', 'run', 'valid_time', 'variable']).any()
    assert sha256_file(out / 'observations.parquet') == sha256_file(base / 'observations.parquet')
    record = read_json(out / 'archive.json')['extension']
    assert record['requests'] == 3 and len(record['unavailable_requests']) == 1
    rag_archive.extend_issued(base.name, cfg, store, log=lambda *a, **k: None)
    assert len(calls) == 3  # restarting never redownloads verified cached responses
    assert original == {p.name: sha256_file(p) for p in base.iterdir()}
