import pandas as pd
import pytest

from weather_lab.context_protocol import candidate_identity, candidates, origins, schedule
from weather_lab.forecast_config import ForecastConfig


START = "2026-10-10T06:00:00Z"


def test_schedule_windows_and_label_purge_are_frozen():
    plan = schedule(START)
    assert plan["history_window"] == {"start": START, "end": "2027-01-08T06:00:00Z"}
    assert plan["validation_window"] == {"start": "2027-01-08T06:00:00Z", "end": "2027-02-07T06:00:00Z"}
    assert plan["purge_window"] == {"start": "2027-02-07T06:00:00Z", "end": "2027-02-09T06:00:00Z"}
    assert plan["confirmation_window"] == {"start": "2027-02-09T06:00:00Z", "end": "2027-04-10T06:00:00Z"}
    assert plan["last_validation_label_time"] == "2027-02-08T06:00:00Z"
    assert plan["last_confirmation_label_time"] == "2027-04-11T06:00:00Z"
    assert pd.Timestamp(plan["last_validation_label_time"]) < pd.Timestamp(plan["confirmation_window"]["start"])
    assert plan["calibration"] is None


def test_origins_have_all_daily_cases_without_candidate_trimming():
    cases = origins(START, ["07110", "07149", "07190", "07630", "07690"])
    assert list(cases.columns) == ["case_id", "station", "origin", "split"]
    assert len(cases) == 450
    assert cases.groupby("split").size().to_dict() == {"confirmation": 300, "validation": 150}
    assert cases[cases.split.eq("validation")].origin.nunique() == 30
    assert cases[cases.split.eq("confirmation")].origin.nunique() == 60
    assert cases.case_id.is_unique and cases.origin.dt.tz is not None
    assert cases.groupby("split").origin.apply(lambda x: x.drop_duplicates().sort_values().diff().dropna().eq(pd.Timedelta(days=1)).all()).all()
    assert set((cfg.context_days, cfg.horizon_h) for cfg in candidates().values()) == {(30, 24), (30, 48), (90, 24), (90, 48)}


@pytest.mark.parametrize("value", ["2026-10-10T06:00:00", "2026-10-10T07:00:00Z", "not-a-time"])
def test_schedule_rejects_naive_or_non_06_utc_anchors(value):
    with pytest.raises((ValueError, TypeError)):
        schedule(value)


@pytest.mark.parametrize("stations", [[], ["7110"], ["0711A"], ["07110", "07110"]])
def test_origins_reject_invalid_station_lists(stations):
    with pytest.raises(ValueError):
        origins(START, stations)


def test_candidate_identity_includes_full_config_and_provenance():
    config = candidates()["timestral209m-error-ctx30-h24-v1"]
    provenance = {"revision": "rev-a", "runtime": "r1", "source": "src-a", "masks": "m1", "fit": "f1"}
    identity = candidate_identity(config, provenance)
    assert identity == candidate_identity(config, dict(reversed(list(provenance.items()))))
    assert len({identity, candidate_identity(candidates()["timestral209m-error-ctx30-h48-v1"], provenance),
                candidate_identity(candidates()["timestral209m-error-ctx90-h24-v1"], provenance),
                candidate_identity(ForecastConfig.make("Timestral-209M", "futureonly", ["resid"],
                                                        calendar=False, context_days=30, horizon_h=24,
                                                        targets=("t", "pmer"), step_hours=3,
                                                        prediction_target="arome_error"), provenance),
                candidate_identity(config, {**provenance, "revision": "rev-b"}),
                candidate_identity(config, {**provenance, "runtime": "r2"}),
                candidate_identity(config, {**provenance, "source": "src-b"}),
                candidate_identity(config, {**provenance, "masks": "m2"}),
                candidate_identity(config, {**provenance, "fit": "f2"})}) == 9
    assert all(cfg.prediction_target == "arome_error" and cfg.conditioning == "futureonly"
               and not cfg.features and not cfg.calendar and cfg.targets == ("t", "pmer")
               for cfg in candidates().values())
