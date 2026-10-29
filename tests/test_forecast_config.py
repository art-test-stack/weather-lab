"""Configuration, capability and saved-system mapping contracts without downloaded artifacts."""
import itertools

import pytest

from weather_lab import features as feats, models
from weather_lab.forecast_config import Availability, ForecastConfig, check, from_system

ALL_DATA = Availability(arome_issued_h=48, arome_run="2026-10-21T03:00:00Z", arome_history=True, aux=True,
                        station_coords=True)


def test_both_timestral_sizes_accept_the_same_feature_selections():
    groups = list(feats.GROUPS)
    for cond in ("none", "archived", "substobs", "futureonly"):
        for k in range(len(groups) + 1):
            for combo in itertools.combinations(groups, k):            # includes "no additional features"
                small = ForecastConfig.make("Timestral-28M", cond, list(combo), horizon_h=48)
                large = ForecastConfig.make("Timestral-209M", cond, list(combo), horizon_h=48)
                assert check(small, ALL_DATA) == check(large, ALL_DATA) == []
                assert small.features == large.features and small.model != large.model


def test_unsupported_capabilities_are_distinguished_from_missing_data(monkeypatch):
    from weather_lab import models

    cfg = ForecastConfig.make("Timestral-28M", "none", ["resid", "atmos", "solar"], horizon_h=48)
    issues = check(cfg, Availability(arome_issued_h=48, arome_history=False, aux=False, station_coords=True))
    assert {(i.subject, i.kind) for i in issues} == {("resid", "unavailable"), ("atmos", "unavailable")}
    assert "even without AROME conditioning" in next(i.message for i in issues if i.subject == "resid")
    # a family that takes known-future but no past-only covariates (e.g. a future model): explicit, per group
    fam = models.FAMILIES["timestral"]
    monkeypatch.setitem(models.FAMILIES, "timestral",
                        models.Family(**{**fam.__dict__, "past_covariates": False, "name": "NoPast"}))
    issues = check(cfg, ALL_DATA)
    assert {(i.subject, i.kind) for i in issues} == {("resid", "unsupported"), ("atmos", "unsupported")}
    assert all("NoPast takes no past-only covariates" in i.message for i in issues)
    planned = ForecastConfig.make("Timestral-28M", "none", [], horizon_h=48)
    monkeypatch.setitem(models.FAMILIES, "timestral", models.Family(**{**fam.__dict__, "status": "planned"}))
    assert [i.kind for i in check(planned, ALL_DATA)] == ["unsupported"]


def test_no_silent_changes_for_invalid_parameters_and_horizons():
    with pytest.raises(ValueError):
        ForecastConfig.make("Timestral-28M", "none", {"tend": {"lags_h": []}})
    long = ForecastConfig.make("Timestral-28M", "archived", [], horizon_h=72)
    assert [(i.subject, i.kind) for i in check(long, ALL_DATA)] == [("horizon", "unavailable")]
    assert check(ForecastConfig.make("Timestral-28M", "none", [], horizon_h=72), ALL_DATA) == []
    no_run = Availability(arome_issued_h=0, notes={"arome": "no AROME run"})
    assert [(i.subject, i.kind) for i in check(long, no_run)][0] == ("conditioning", "unavailable")


def test_feature_parameters_change_rows_versions_and_keys_but_defaults_match_version_1():
    default = feats.definition(["tend"])[0]
    custom = feats.definition({"tend": {"lags_h": [6, 12]}})[0]
    assert default["rows"] == ["dt_3h", "dt_24h", "dpmer_3h", "dpmer_24h"]           # version 1, unchanged
    assert custom["rows"] == ["dt_6h", "dt_12h", "dpmer_6h", "dpmer_12h"] and custom["version"] == 1
    from weather_lab.forecast import Timestral

    t = Timestral("x", ["t", "pmer"], ["t", "pmer"], "mistralai/Timestral-28M", "r" * 40, features=["tend"])
    assert "feature_params" not in t.params                     # benchmark provenance of defaults unchanged
    t2 = Timestral("x", ["t", "pmer"], ["t", "pmer"], "mistralai/Timestral-28M", "r" * 40,
                   features={"tend": {"lags_h": [6]}})
    assert t2.params["feature_params"] == {"tend": {"lags_h": [6]}}

@pytest.fixture(params=["Timestral-28M", "Timestral-209M"])
def recorded_system(request):
    checkpoint = models.CHECKPOINTS[request.param]
    return {"id": request.param.lower() + "-synthetic-record", "kind": "timestral",
            "revision": checkpoint.repo + "@" + checkpoint.revision[:12], "variables": ["t", "pmer"],
            "params": {"checkpoint": checkpoint.repo, "context_days": 30, "calendar": True,
                       "arome": "archived", "features": ["resid"],
                       "feature_versions": {"resid": feats.GROUPS["resid"].version}}}


def test_saved_configuration_mapping_uses_recorded_parameters(recorded_system):
    expected = ForecastConfig.make(recorded_system["params"]["checkpoint"].split("/")[-1], "archived", ["resid"],
                                   context_days=30, horizon_h=48, targets=("t", "pmer"))
    assert from_system(recorded_system, 48) == expected


def test_legacy_observations_spelling_never_becomes_archived_arome(recorded_system):
    recorded_system["params"].update(arome="observations", features=[], feature_versions={})
    cfg = from_system(recorded_system, 48)
    assert cfg.conditioning == "substobs" and not cfg.feature_names


@pytest.mark.parametrize("mismatch", ["revision", "feature_version", "alignment", "missing_context"])
def test_saved_configuration_mapping_refuses_incompatible_or_missing_provenance(recorded_system, mismatch):
    if mismatch == "revision":
        recorded_system["revision"] = recorded_system["params"]["checkpoint"] + "@000000000000"
    elif mismatch == "feature_version":
        recorded_system["params"]["feature_versions"]["resid"] += 1
    elif mismatch == "alignment":
        recorded_system["input_alignment"] = {"observation_delay_hours": 3}
    else:
        del recorded_system["params"]["context_days"]
    assert from_system(recorded_system, 48) is None


def test_nondefault_feature_parameters_survive_saved_configuration_mapping(recorded_system):
    recorded_system["params"].update(features=["tend"], feature_versions={"tend": feats.GROUPS["tend"].version},
                                     feature_params={"tend": {"lags_h": [6]}})
    assert from_system(recorded_system, 48).feature_spec == feats.normalize({"tend": {"lags_h": [6]}})
