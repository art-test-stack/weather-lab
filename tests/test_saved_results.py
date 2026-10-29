"""Small saved files exercise the same matching/reader path as the Benchmark views."""
import numpy as np
import pandas as pd
import pytest

from weather_lab.saved_results import annual_temperature, calendar, horizon, matched, one_forecast
from weather_lab_app.results import Run, _parquet


@pytest.fixture
def saved(tmp_path):
    cases = pd.DataFrame({"case_id": ["a", "b", "c"], "station": ["07149"] * 3,
                          "origin": pd.to_datetime(["2026-01-01T23:00Z", "2026-01-02T00:00Z", "2026-01-03T00:00Z"]),
                          "eligible": [True, True, False], "split": ["test"] * 3})
    truth = cases.iloc[:2][["case_id", "origin"]].merge(pd.DataFrame({"lead_h": [1, 2]}), how="cross")
    truth["valid_time"] = truth.origin + pd.to_timedelta(truth.lead_h, unit="h")
    truth["variable"], truth["observed"] = "t", [0., 0., 0., np.nan]
    truth = truth.drop(columns="origin")
    truth.to_parquet(tmp_path / "targets.parquet")
    points = truth.drop(columns="observed").merge(cases[["case_id", "origin", "station"]], on="case_id")
    points["model"], points["point"] = "first", [1., 9., 3., 50.]
    other = points.assign(model="second", point=[2., np.nan, 4., 60.])
    pd.concat([points, other.assign(variable="pmer"), other]).to_parquet(tmp_path / "predictions.parquet")
    run = Run("fixture", tmp_path, "fixture", {}, {})
    return run, truth, cases


def test_projected_reader_matched_support_and_missing_data(saved, monkeypatch):
    run, truth, cases = saved
    calls = []
    read = pd.read_parquet
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **kw: (calls.append(kw), read(*a, **kw))[1])
    first = run.point_predictions("first", "t")
    assert run.point_predictions("first", "t") is first
    assert len(calls) == 1 and "q0.50" not in calls[0]["columns"]
    second = run.point_predictions("second", "t")
    assert len(second) == 4 and set(second.variable) == {"t"}
    rows, support = matched({"one": first, "two": second}, truth, cases, "t")
    assert set(zip(rows.case_id, rows.lead_h)) == {("a", 1), ("b", 1)}
    assert rows.groupby("model").size().to_dict() == {"one": 2, "two": 2}
    assert support["windows"] == 2 and support["scheduled_windows"] == 3
    assert support["candidate_pairs"] == 96 and support["observed_pairs"] == 2
    assert support["missing_target_pairs"] == 93 and support["excluded_pairs"] == 94
    assert support["exclusions"][0]["excluded_by_matching"] == 1
    empty, audit = matched({"one": first, "absent": second.iloc[:0]}, truth, cases, "t")
    assert empty.empty and audit["observed_pairs"] == 0
    _parquet.cache_clear()


def test_lead_alignment_pair_weighted_cumulative_and_population_spread(saved):
    run, truth, cases = saved
    p = run.point_predictions("first", "t")
    rows, _ = matched({"one": p}, truth, cases, "t")
    at = horizon(rows).set_index("lead_h")
    assert at.index.tolist() == list(range(1, 49))
    assert at.loc[1, "mae"] == 2 and at.loc[1, "spread"] == 1
    assert at.loc[2, "mae"] == 9 and at.loc[2, "spread"] == 0
    assert pd.isna(at.loc[3, "mae"]) and at.loc[3, "n"] == 0
    through = horizon(rows, cumulative=True).set_index("lead_h")
    assert through.loc[2, "mae"] == pytest.approx(13 / 3)
    assert through.loc[2, "n"] == 3 and through.loc[2, "n_cases"] == 2
    assert through.loc[48, "mae"] == pytest.approx(13 / 3)
    assert through.spread.isna().all() and through.ci95_low.isna().all()
    wrong = p.copy()
    wrong.loc[0, "valid_time"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="prediction valid time"):
        matched({"one": wrong}, truth, cases, "t")
    with pytest.raises(ValueError, match="target valid time"):
        matched({"one": p}, truth.assign(valid_time=truth.valid_time + pd.Timedelta(hours=1)), cases, "t")


def test_arome_support_is_explicit_and_variables_are_separate(saved):
    run, truth, cases = saved
    p = run.point_predictions("first", "t")
    ar = p.assign(point=[np.nan, 2., 1., np.nan])
    broad, _ = matched({"one": p}, truth, cases, "t")
    narrow, _ = matched({"one": p, "arome-raw": ar}, truth, cases, "t")
    assert len(broad) == 3 and len(narrow) == 4
    assert set(zip(narrow.case_id, narrow.lead_h)) == {("a", 2), ("b", 1)}
    pressure, support = matched({"one": p}, truth, cases, "pmer")
    assert pressure.empty and support["observed_pairs"] == 0
    duplicate = pd.concat([p, p.iloc[:1]])
    with pytest.raises(pd.errors.MergeError):
        matched({"one": duplicate}, truth, cases, "t")


def test_annual_temperature_ranks_identical_pairs_by_origin_year_and_retains_unavailable(saved):
    run, truth, cases = saved
    first, second = (run.point_predictions(model, "t") for model in ("first", "second"))
    old = cases.iloc[:1].assign(case_id="old", origin=pd.Timestamp("2025-12-31T23:00Z"))
    old_truth = truth.iloc[:1].assign(case_id="old", valid_time=pd.Timestamp("2026-01-01T00:00Z"))
    old_point = first.iloc[:1].assign(case_id="old", origin=old.origin.iloc[0],
                                     valid_time=old_truth.valid_time.iloc[0], point=1000.)
    frames = {"one": pd.concat([first, old_point]), "two": pd.concat([second, old_point])}
    table, support = annual_temperature(frames, pd.concat([truth, old_truth]), pd.concat([cases, old]), split="test")
    assert table.model.tolist() == ["one", "two"] and table.mae.tolist() == [2., 3.]
    assert table.n.tolist() == [2, 2] and table.n_cases.tolist() == [2, 2]
    assert support["scheduled_windows"] == 3 and support["observed_pairs"] == 2
    broad, _ = annual_temperature({"one": first}, truth, cases)
    assert broad.mae.iloc[0] == pytest.approx(13 / 3)  # pairs, not the mean of per-window means
    empty, audit = annual_temperature({"one": first, "absent": second.iloc[:0]}, truth, cases)
    assert empty.model.tolist() == ["one", "absent"] and empty.mae.isna().all()
    assert empty.n.tolist() == [0, 0] and audit["observed_pairs"] == 0


def test_one_forecast_preserves_gaps_and_fixed_issue_relative_leads(saved):
    run, truth, cases = saved
    p = run.point_predictions("first", "t")
    reference = p[p.case_id.eq("a")].assign(point=[2., np.nan], arome_run=pd.Timestamp("2026-01-01T18:00Z"),
                                           arome_available_at=pd.Timestamp("2026-01-01T21:00Z"))
    curves, obs, case, errors, support = one_forecast({"one": p, "arome-raw": reference}, truth, cases, "a", "t", reference)
    assert len(curves) == 96 and curves[curves.model.eq("one")].point.notna().sum() == 2
    assert support["observed_pairs"] == 1 and set(errors.lead_h) == {1}
    ar = curves[curves.model.eq("arome-raw")]
    assert ar.arome_lead_h.iloc[0] == 6 and ar.arome_run.nunique() == 1
    assert errors.set_index("model").loc["one", "err"] == 1
    missing = truth.assign(observed=np.nan)
    curves, _, _, errors, support = one_forecast({"one": p}, missing, cases, "a", "t", reference)
    assert errors.empty and support["observed_pairs"] == 0
    assert curves.point.notna().sum() == 3  # forecasts remain visible without verifying targets
    with pytest.raises(ValueError, match="unavailable at"):
        one_forecast({"one": p}, truth, cases, "a", "t", reference.assign(arome_available_at=pd.Timestamp("2026-01-02T00:00Z")))
    with pytest.raises(ValueError, match="single issued-run"):
        one_forecast({"one": p}, truth, cases, "a", "t", reference.assign(point=[1., 2.], arome_run=pd.to_datetime(["2026-01-01T18:00Z", "2026-01-01T21:00Z"])))


def test_calendar_timestamps_pair_weights_population_spread_and_gaps(saved):
    run, truth, cases = saved
    cases = cases.copy()
    cases["origin"] += pd.Timedelta(days=30)
    truth = truth.copy()
    truth["valid_time"] += pd.Timedelta(days=30)
    p = run.point_predictions("first", "t").copy()
    p["origin"] += pd.Timedelta(days=30)
    p["valid_time"] += pd.Timedelta(days=30)
    rows, support = matched({"one": p}, truth, cases, "t")
    origin = calendar(rows, "origin", start=pd.Timestamp("2026-01-31T00:00Z"), end=pd.Timestamp("2026-02-02T00:00Z"))
    valid = calendar(rows, "valid_time", start=pd.Timestamp("2026-01-31T00:00Z"), end=pd.Timestamp("2026-02-02T00:00Z"))
    monthly = origin["month"].set_index("bucket")
    assert monthly.loc["2026-01", "mae"] == 5 and monthly.loc["2026-01", "spread"] == 4
    assert monthly.loc["2026-02", "mae"] == 3
    monthly_valid = valid["month"].set_index("bucket")
    assert monthly_valid.loc["2026-02", "mae"] == pytest.approx(13 / 3)
    assert monthly_valid.loc["2026-02", "spread"] == pytest.approx(np.std([1., 9., 3.]))
    assert monthly_valid.loc["2026-02", "n"] == 3 and monthly_valid.loc["2026-02", "n_cases"] == 2
    assert monthly_valid.loc["2026-01", "n"] == 0 and pd.isna(monthly_valid.loc["2026-01", "mae"])
    hourly = valid["hour"].set_index("bucket")
    assert hourly.loc[0, "mae"] == 1 and hourly.loc[1, "mae"] == 6 and hourly.loc[1, "spread"] == 3
    assert len(hourly) == 24 and hourly.loc[23, "n"] == 0
    lead_one = calendar(rows, lead=1)["month"].iloc[0]
    assert lead_one.mae == 2 and lead_one.spread == 1 and lead_one.n == 2
    daily = valid["day"].set_index("bucket")
    assert daily.loc[pd.Timestamp("2026-02-02T00:00Z"), "n"] == 0
    assert sum(r["matched_pairs"] for r in support["lead_support"]) == 3
    assert support["lead_support"][1]["observed_target_pairs"] == 1
    with pytest.raises(ValueError, match="Choose origin"):
        calendar(rows, "station")
    with pytest.raises(ValueError, match="one variable"):
        calendar(pd.concat([rows, rows.assign(variable="pmer")]))
