from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from weather_lab import evaluate
from weather_lab.forecast import CaseBatch, ForecastBatch, QUANTILE_LEVELS


def example(delay=0):
    cases = pd.DataFrame({"case_id": ["first", "second"], "station": ["07149", "07630"],
                          "origin": pd.to_datetime(["2026-04-03T06:00Z", "2026-04-04T06:00Z"])})
    origins = cases.origin.to_numpy(dtype="datetime64[ns]") - np.timedelta64(delay, "h")
    future = origins[:, None] + np.arange(1, delay + 3) * np.timedelta64(1, "h")
    baseline = (np.array([[10., 1000.], [20., 1010.]])[:, :, None]
                + np.arange(delay + 2)[None, None, :])
    point = baseline + np.array([[1., -2.], [3., -4.]])[:, :, None]
    quantiles = point[..., None] + np.array(QUANTILE_LEVELS) - .5
    batch = CaseBatch(cases, ["t", "pmer"], np.zeros((2, 2, 1)), origins, future, 1)
    result = ForecastBatch("example", "v1", "quantiles", batch.variables, point,
                           quantiles, QUANTILE_LEVELS)
    return batch, result, baseline


def test_direct_export_preserves_weather_and_quantiles():
    batch, result, _ = example()
    frame = result.to_frame(batch, "example-suite")

    assert frame.case_id.tolist() == ["first"] * 4 + ["second"] * 4
    assert frame.variable.tolist() == ["t", "t", "pmer", "pmer"] * 2
    assert frame.lead_h.tolist() == [1, 2] * 4
    np.testing.assert_array_equal(frame.point, [11, 12, 998, 999, 23, 24, 1006, 1007])
    np.testing.assert_array_equal(frame["q0.50"], frame.point)
    assert "arome_baseline" not in frame and "predicted_error" not in frame


@pytest.mark.parametrize("delay", [0, 3])
def test_residual_export_aligns_baseline_after_bridge_filtering(delay):
    batch, result, baseline = example(delay)
    frame = result.to_frame(batch, "example-suite", arome_baseline=baseline)

    assert frame.lead_h.tolist() == [1, 2] * 4
    np.testing.assert_array_equal(frame.arome_baseline,
                                  np.array([10, 11, 1000, 1001, 20, 21, 1010, 1011]) + delay)
    np.testing.assert_array_equal(frame.predicted_error, [1, 1, -2, -2, 3, 3, -4, -4])
    direct = result.to_frame(batch, "example-suite")
    pd.testing.assert_frame_equal(frame.drop(columns=["arome_baseline", "predicted_error"]), direct)


def test_export_rejects_mismatched_baseline():
    batch, result, baseline = example()
    with pytest.raises(ValueError, match="baseline must match the forecast shape"):
        result.to_frame(batch, "example-suite", arome_baseline=baseline[:1])


def test_arome_evaluation_reaches_extract_loader(monkeypatch):
    cfg = SimpleNamespace(manifest_path=Mock())
    monkeypatch.setattr(evaluate, "read_json", lambda _: {"case_data_sha256": "frozen"})
    monkeypatch.setattr(evaluate, "load_observations", lambda _: (pd.DataFrame(), {}))
    monkeypatch.setattr(evaluate, "cases_from_manifest", lambda _: pd.DataFrame({"eligible": [True]}))
    monkeypatch.setattr(evaluate, "case_data_checksum", lambda *_: "frozen")
    monkeypatch.setattr(evaluate, "build_adapters", lambda _: [SimpleNamespace(inputs=["arome"])])

    class ExtractRequested(Exception):
        pass

    from weather_lab import arome

    loader = Mock(side_effect=ExtractRequested)
    monkeypatch.setattr(arome, "load_arome", loader)
    with pytest.raises(ExtractRequested):
        evaluate.run_evaluation(cfg)
    loader.assert_called_once_with(cfg)
