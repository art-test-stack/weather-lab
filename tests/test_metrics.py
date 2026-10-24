"""Metric values on hand-computable arrays, common-case comparison, deterministic N/A."""

import numpy as np
import pandas as pd
import pytest

from weather_lab.evaluate import pinball, score
from weather_lab.forecast import QUANTILE_LEVELS, qcol


def test_pinball_hand_computed():
    levels = np.array([0.1, 0.5, 0.9])
    y = np.array([10.0])
    q = np.array([[8.0, 10.0, 13.0]])
    # 0.1*(10-8)=0.2 ; 0 ; (0.9-1)*(10-13)=0.3 -> mean 0.5/3
    assert pinball(y, q, levels)[0] == pytest.approx(0.5 / 3)
    # median-only pinball is half the absolute error
    assert pinball(np.array([3.0]), np.array([[1.0]]), np.array([0.5]))[0] == pytest.approx(1.0)


def _rows(model, output, points, quantiles=None):
    n = len(points)
    df = pd.DataFrame({"suite": "s", "run_id": "r", "case_id": [f"c{i // 2}" for i in range(n)],
                       "station": "07149", "origin": pd.Timestamp("2024-01-01T00:00Z"),
                       "valid_time": pd.Timestamp("2024-01-01T03:00Z"), "lead_h": [3 * (1 + i % 2) for i in range(n)],
                       "variable": "t", "model": model, "model_revision": "x", "output_type": output,
                       "point": points, "unit": "°C"})
    if quantiles is not None:
        for k, level in enumerate(QUANTILE_LEVELS):
            df[qcol(level)] = [q[k] for q in quantiles]
    return df


def _setup():
    obs = [10.0, 12.0, 14.0, np.nan]          # last target missing -> excluded everywhere
    targets = pd.DataFrame({"case_id": ["c0", "c0", "c1", "c1"], "variable": "t", "lead_h": [3, 6, 3, 6],
                            "valid_time": pd.Timestamp("2024-01-01T03:00Z"), "observed": obs})
    cases = pd.DataFrame({"case_id": ["c0", "c1"], "station": "07149", "split": "test", "season": "DJF",
                          "origin": [pd.Timestamp("2024-01-01T00:00Z"), pd.Timestamp("2024-01-02T00:00Z")],
                          "eligible": True})
    spread = np.array(QUANTILE_LEVELS) - 0.5   # quantiles = median + (level - 0.5)
    persistence = _rows("persistence", "point", [11.0, 11.0, 11.0, 11.0])
    clim = _rows("climatology", "quantiles", [10.0, 10.0, 10.0, 10.0], [10.0 + spread * 4] * 4)
    # model fails one prediction that persistence made -> that pair leaves the common set
    model = _rows("model", "quantiles", [10.0, 12.0, np.nan, 13.0], [m + spread for m in (10.0, 12.0, 0.0, 13.0)])
    model.loc[2, [qcol(level) for level in QUANTILE_LEVELS]] = np.nan
    preds = pd.concat([persistence, clim, model], ignore_index=True)
    return preds, targets, cases


def test_common_case_scores_and_availability():
    preds, targets, cases = _setup()
    t = score(preds, targets, cases, thresholds={}, highlight=[3])
    pt = t["metrics_point"]
    allrow = pt[(pt["station"] == "ALL") & (pt["season"] == "ALL") & pt["lead_h"].isna()].set_index("model")
    # common set = c0 lead 3, c0 lead 6 (c1 lead 3: model failed; c1 lead 6: target missing)
    assert (allrow["n"] == 2).all()
    assert allrow.loc["persistence", "mae"] == pytest.approx((1 + 1) / 2)
    assert allrow.loc["model", "mae"] == pytest.approx(0.0)
    assert allrow.loc["climatology", "rmse"] == pytest.approx(np.sqrt((0 + 4) / 2))
    assert allrow.loc["model", "skill_mae_vs_persistence"] == pytest.approx(1.0)
    av = t["availability"].set_index("model")
    assert av.loc["model", "failed_predictions"] == 1 and av.loc["model", "valid_pairs"] == 2
    assert av.loc["persistence", "valid_pairs"] == 3       # own coverage is reported separately


def test_probabilistic_metrics_only_for_probabilistic_outputs():
    preds, targets, cases = _setup()
    t = score(preds, targets, cases, thresholds={}, highlight=[3])
    prob = t["metrics_probabilistic"]
    assert "persistence" not in set(prob["model"])
    assert set(t["probabilistic_not_applicable"]["model"]) == {"persistence"}
    row = prob[(prob["model"] == "model") & (prob["station"] == "ALL") & (prob["season"] == "ALL")
               & prob["lead_h"].isna()].iloc[0]
    # perfect median, symmetric +/-0.4 at q0.1/q0.9 -> both observations inside the 80 % interval
    assert row["coverage80"] == 1.0 and row["width80"] == pytest.approx(0.8)
