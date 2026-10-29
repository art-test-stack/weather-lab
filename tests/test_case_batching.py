"""Local grouped inference keeps each case's data and failure status."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from weather_lab import development
from weather_lab.arome_inputs import AromeInputs
from weather_lab.forecast import CaseBatch, ForecastBatch


def cases(n=9):
    result = []
    for i in range(n):
        origin = np.array([np.datetime64('2026-10-01T00', 'ns') + np.timedelta64(i, 'h')])
        history = np.full((1, 1, 4), 10 + i, np.float32)
        future = origin[:, None] + np.arange(1, 3) * np.timedelta64(1, 'h')
        batch = CaseBatch(pd.DataFrame({'case_id': [f'c{i}'], 'station': ['07149'],
                          'origin': pd.to_datetime(origin, utc=True)}), ['t'], history, origin, future, 1)
        ht = batch.history_times()[:, None]
        ft = future[:, None]
        runs = ht - np.timedelta64(2, 'h')
        fr = np.broadcast_to(origin[:, None, None] - np.timedelta64(2, 'h'), ft.shape).copy()
        named = AromeInputs(('t',), history.copy(), np.full((1, 1, 2), 11 + i, np.float32),
                            runs, runs + np.timedelta64(1, 'h'), fr, fr + np.timedelta64(1, 'h'))
        result.append(replace(batch, arome_inputs=named, arome=named.future))
    return result


class Echo:
    def __init__(self, bad=None):
        self.calls = []
        self.bad = bad

    def predict(self, batch):
        ids = batch.cases.case_id.tolist()
        self.calls.append(ids)
        if self.bad in ids:
            raise ValueError('synthetic bad input')
        # Both named source runs and values must stay attached to the correct window.
        for i, case in enumerate(batch.cases.itertuples()):
            k = int(case.case_id[1:])
            assert (batch.arome_inputs.history[i] == 10 + k).all()
            assert (batch.arome_inputs.future_run[i] == batch.origins[i] - np.timedelta64(2, 'h')).all()
        point = np.broadcast_to(batch.history[:, :, -1:], (len(ids), 1, batch.horizon)).copy()
        return ForecastBatch('echo', 'test', 'point', ['t'], point, info={
            'n_cases': len(ids), 'retrieval': [{'case_id': case} for case in ids],
            'residual_observed_steps': [[4] for _ in ids]})


def test_eight_cases_share_one_call_and_keep_order_and_provenance():
    batches = cases()
    adapter = Echo()
    grouped = development.predict_cases(adapter, batches, batch_size=8)
    assert [len(ids) for ids in adapter.calls] == [8, 1]
    singles = development.predict_cases(Echo(), batches)
    for i, ((result, reason), (single, _)) in enumerate(zip(grouped, singles)):
        assert reason == ''
        np.testing.assert_array_equal(result.point, single.point)
        assert result.info['retrieval'] == [{'case_id': f'c{i}'}]
        assert result.info['residual_observed_steps'] == [[4]]
        result.validate(batches[i])


def test_group_failure_retries_cases_without_losing_or_duplicating_neighbours():
    adapter = Echo(bad='c3')
    results = development.predict_cases(adapter, cases(8), batch_size=8)
    assert len(results) == 8 and len(adapter.calls) == 9
    assert results[3] == (None, 'ValueError: synthetic bad input')
    assert all(result is not None and 'batch_retry_reason' in result.info
               for i, (result, _) in enumerate(results) if i != 3)


def test_incompatible_inputs_are_not_combined_and_batch_size_is_validated():
    batches = cases(2)
    batches[1] = replace(batches[1], arome=None, arome_inputs=None)
    adapter = Echo()
    # Check compatibility directly: the individual adapter intentionally requires named inputs.
    with pytest.raises(ValueError, match='incompatible'):
        development._combine_cases(batches)
    with pytest.raises(ValueError, match='positive'):
        development.predict_cases(adapter, batches, batch_size=0)
