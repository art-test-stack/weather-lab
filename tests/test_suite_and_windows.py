"""Split boundaries, history/target windows and future-target exclusion (synthetic observations)."""

import numpy as np
import pandas as pd
import pytest

from weather_lab.baselines import Persistence, SeasonalNaive
from weather_lab.common import Config
from weather_lab.evaluate import make_batch
from weather_lab.suite import build_cases, build_origins


def _cfg(tmp_path, **over):
    raw = {
        "suite": "test-suite", "variant": "unit", "manifest": str(tmp_path / "m.json"),
        "paths": {"raw": str(tmp_path), "processed": str(tmp_path), "runs": str(tmp_path)},
        "data": {"years": [2023, 2024], "stations": ["07149"]},
        "splits": {"fit": ["2023-01-01T00:00Z", "2023-06-30T21:00Z"],
                   "validation": ["2023-07-01T00:00Z", "2023-12-31T21:00Z"],
                   "test": ["2024-01-01T00:00Z", "2024-01-31T21:00Z"]},
        "origins": {"rule": "daily", "hour_utc": 0, "splits": ["validation", "test"]},
        "forecast": {"step_hours": 3, "horizon_steps": 16, "context_days": 2, "highlight_leads_h": [6, 24, 48]},
        "eligibility": {"variable": "t", "min_context_observed_fraction": 0.8, "require_observation_within_hours": 24},
        "variables": ["t", "pmer", "u", "v"],
    }
    raw.update(over)
    return Config(raw=raw, path=tmp_path / "cfg.yaml")


def _obs(start="2023-06-01", end="2024-02-10"):
    time = pd.date_range(f"{start}T00:00Z", f"{end}T21:00Z", freq="3h")
    n = len(time)
    return pd.DataFrame({"station": "07149", "time": time, "t": np.arange(n, dtype=float),
                         "pmer": 1000.0 + np.zeros(n), "u": np.zeros(n), "v": np.zeros(n)})


def test_target_windows_lie_inside_split(tmp_path):
    cfg = _cfg(tmp_path)
    origins = build_origins(cfg)
    horizon = pd.Timedelta(hours=48)
    for origin, split in origins:
        start, end = (pd.Timestamp(x) for x in cfg["splits"][split])
        assert origin >= start and origin + horizon <= end
    # the last two days of each split cannot host a 48 h window
    assert max(o for o, s in origins if s == "test") == pd.Timestamp("2024-01-29T00:00Z")
    assert max(o for o, s in origins if s == "validation") == pd.Timestamp("2023-12-29T00:00Z")


def test_history_ends_at_origin_and_targets_are_not_inputs(tmp_path):
    cfg = _cfg(tmp_path)
    obs = _obs()
    cases = build_cases(cfg, obs)
    cases = cases[cases["eligible"]].head(5).reset_index(drop=True)
    batch, targets = make_batch(cfg, obs, cases, history_steps=16)
    t_idx = cfg["variables"].index("t")
    # t equals the step index, so the last history value is the origin's index and targets follow it
    for i, c in enumerate(cases.itertuples()):
        origin_idx = int((c.origin - obs["time"].iloc[0]) / pd.Timedelta(hours=3))
        assert batch.history[i, t_idx, -1] == origin_idx
        assert targets[i, t_idx, 0] == origin_idx + 1
        assert batch.history[i, t_idx].max() < targets[i, t_idx].min()
    # changing observations after the origins must not change any input
    obs2 = obs.copy()
    after = obs2["time"] > cases["origin"].max()
    obs2.loc[after, ["t", "pmer", "u", "v"]] = -999.0
    batch2, _ = make_batch(cfg, obs2, cases, history_steps=16)
    np.testing.assert_array_equal(batch.history[-1], batch2.history[-1])


def test_eligibility_uses_context_missingness(tmp_path):
    cfg = _cfg(tmp_path)
    obs = _obs()
    gap = (obs["time"] > "2024-01-08") & (obs["time"] <= "2024-01-10")
    obs.loc[gap, "t"] = np.nan
    cases = build_cases(cfg, obs).set_index("case_id")
    assert not cases.loc["test-suite/07149/20240110T00", "eligible"]
    assert "observed <" in cases.loc["test-suite/07149/20240110T00", "exclusion_reason"]
    assert cases.loc["test-suite/07149/20240105T00", "eligible"]


def test_persistence_and_seasonal_naive_only_use_past(tmp_path):
    cfg = _cfg(tmp_path)
    obs = _obs()
    cases = build_cases(cfg, obs)
    cases = cases[cases["eligible"]].head(2).reset_index(drop=True)
    batch, targets = make_batch(cfg, obs, cases, history_steps=16)
    p = Persistence("p", ["t"]).predict(batch).point
    assert (p[:, 0, :] == batch.history[:, 0, -1:]).all()
    s = SeasonalNaive("s", ["t"]).predict(batch).point
    # lead 3 h repeats the value 21 h before the origin; lead 24 h is the origin value; lead 27 h -> 21 h before
    assert s[0, 0, 0] == batch.history[0, 0, -8]
    assert s[0, 0, 7] == batch.history[0, 0, -1]
    assert s[0, 0, 8] == batch.history[0, 0, -8]
    assert (s < targets[:, :1]).all()
