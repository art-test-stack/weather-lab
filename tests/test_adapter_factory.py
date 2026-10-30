"""Parity checks for the shared ForecastConfig adapter factory (no model loading)."""
import pytest

from weather_lab import models
from weather_lab.forecast import Timestral
from weather_lab.forecast_config import ForecastConfig, build_adapter
from weather_lab.t0_adapter import T0


@pytest.fixture(autouse=True)
def no_weight_loading(monkeypatch):
    monkeypatch.setattr(models.MODELS, "get", lambda *args, **kwargs: pytest.fail("factory loaded model weights"))


def _assert_same_native_configuration(actual, expected):
    assert type(actual) is type(expected)
    assert actual.describe() == expected.describe()
    assert actual.checkpoint == expected.checkpoint
    assert actual.revision == expected.revision
    assert actual.variables == expected.variables
    assert actual.batch_size == expected.batch_size
    assert actual.device == expected.device
    assert actual.arome_fields == expected.arome_fields


@pytest.mark.parametrize(
    ("prediction_target", "conditioning", "features", "calendar", "context_days", "batch_size", "device"),
    [
        ("weather", "none", (), False, 30, 7, "cpu"),
        ("weather", "archived", ("solar", "resid"), True, 7, 13, "cuda"),
        ("arome_error", "futureonly", ("tend",), True, 90, 4, "cpu"),
    ],
)
def test_timestral_factory_matches_native_configuration(
    prediction_target, conditioning, features, calendar, context_days, batch_size, device
):
    targets = ("t", "pmer") if prediction_target == "arome_error" else ("t", "pmer", "u", "v")
    cfg = ForecastConfig.make(
        "Timestral-209M", conditioning, features, calendar=calendar, context_days=context_days,
        horizon_h=48, targets=targets, prediction_target=prediction_target,
    )
    model_id = "factory-parity"
    variables = ["t", "pmer", "u", "v"]
    runtime = {"arome_fields": ["t", "pmer", "u"], "batch_size": batch_size, "device": device}
    checkpoint = models.resolve(cfg.model)

    expected = Timestral(
        model_id, variables, targets=list(cfg.targets), checkpoint=checkpoint.repo, revision=checkpoint.revision,
        context_days=cfg.context_days, calendar=cfg.calendar, arome=cfg.arome_mode,
        features=cfg.feature_spec, prediction_target=cfg.prediction_target, **runtime,
    )
    actual = build_adapter(cfg, model_id, variables=variables, **runtime)

    _assert_same_native_configuration(actual, expected)


def test_t0_factory_matches_native_configuration_and_forwards_native_flags():
    cfg = ForecastConfig.make(
        "t0-gamma", "archived", ["solar"], calendar=True, context_days=14, horizon_h=48, targets=("t", "pmer")
    )
    model_id = "factory-t0"
    variables = ["t", "pmer", "u", "v"]
    runtime = {
        "arome_fields": ["t", "pmer"], "batch_size": 11, "device": "cpu",
        "trim_leading_missing": True, "symmetric_averaging": True,
    }
    checkpoint = models.resolve(cfg.model)

    expected = T0(
        model_id, variables, targets=list(cfg.targets), checkpoint=checkpoint.repo, revision=checkpoint.revision,
        context_days=cfg.context_days, calendar=cfg.calendar, arome=cfg.arome_mode,
        features=cfg.feature_spec, **runtime,
    )
    actual = build_adapter(cfg, model_id, variables=variables, **runtime)

    _assert_same_native_configuration(actual, expected)
    assert actual.options["trim_leading_missing"] is True
    assert actual.options["symmetric_averaging"] is True


@pytest.mark.parametrize("model", ["Timestral-209M", "t0-gamma"])
def test_factory_uses_pinned_checkpoint_defaults(model):
    cfg = ForecastConfig.make(model, targets=("t",))
    adapter = build_adapter(cfg, "factory-default")
    checkpoint = models.resolve(model)

    assert adapter.checkpoint == checkpoint.repo
    assert adapter.revision == checkpoint.revision
    assert adapter.device == "cpu"
    assert adapter.batch_size == (8 if checkpoint.family == "t0" else 32)
    assert "horizon_h" not in adapter.describe()["params"]
    if checkpoint.family == "t0":
        assert adapter.options == {
            "trim_leading_missing": False, "symmetric_averaging": False,
            "clip_nonnegative": False, "snap_integers": False,
        }


def test_factory_accepts_explicit_full_checkpoint_revision_override():
    cfg = ForecastConfig.make("Timestral-209M", targets=("t",))
    revision = "a" * 40
    adapter = build_adapter(cfg, "factory-revision", revision=revision)

    assert adapter.checkpoint == models.resolve(cfg.model).repo
    assert adapter.revision == revision


def test_factory_rejects_unknown_model():
    cfg = ForecastConfig.make("unknown-model", targets=("t",))

    with pytest.raises(models.CheckpointError):
        build_adapter(cfg, "factory-unknown")


def test_factory_rejects_t0_substitution():
    cfg = ForecastConfig.make("t0-gamma", "substobs", targets=("t",))
    with pytest.raises(ValueError, match="substitution is unsupported"):
        build_adapter(cfg, "factory-unsupported")


@pytest.mark.parametrize("model", ["Timestral-209M", "t0-gamma"])
def test_factory_supports_residual_prediction_without_additional_arome_covariates(model):
    cfg = ForecastConfig.make(model, calendar=False, targets=("t",), prediction_target="arome_error")
    adapter = build_adapter(cfg, "residual-without-conditioning")
    assert adapter.prediction_target == "arome_error" and adapter.arome_mode is None
    assert "arome" in adapter.inputs
    assert adapter.params["residual_version"] == 1 and adapter.params["forecast_lead_input"]


def test_factory_preserves_legacy_observations_arome_mode_override():
    cfg = ForecastConfig.make("Timestral-209M", "substobs", targets=("t",))
    checkpoint = models.resolve(cfg.model)
    expected = Timestral(
        "factory-legacy-mode", ["t"], targets=["t"], checkpoint=checkpoint.repo, revision=checkpoint.revision,
        context_days=cfg.context_days, calendar=cfg.calendar, arome="observations",
        features=cfg.feature_spec, prediction_target=cfg.prediction_target,
    )

    actual = build_adapter(cfg, "factory-legacy-mode", arome="observations")

    _assert_same_native_configuration(actual, expected)


def test_factory_rejects_conflicting_runtime_arome_mode():
    cfg = ForecastConfig.make("Timestral-209M", "none", targets=("t",))

    with pytest.raises(ValueError):
        build_adapter(cfg, "factory-conflicting-mode", arome="archived")


@pytest.mark.parametrize("model", ["Timestral-209M", "t0-gamma"])
@pytest.mark.parametrize("method", ["level", "shape"])
@pytest.mark.parametrize("kind", ["weather", "arome_error"])
def test_retrieved_factory_preserves_native_options_and_retrieval_provenance(model, method, kind):
    from types import SimpleNamespace
    from weather_lab.rag_adapters import RetrievedTimestral, RetrievedT0

    index = SimpleNamespace(ident="a" * 64, meta={"variables": ["t", "pmer"], "context_hours": 720,
        "retrieval": {"k": 2}, "kind": kind, "payload": "synthetic errors", "product": "synthetic product"})
    cfg = ForecastConfig.make(model, "archived", calendar=False, targets=("t", "pmer"), step_hours=1,
        prediction_target="arome_error", precision="bf16", retrieval_method=method, retrieval_index=index.ident)
    runtime = {"arome_fields": ["t", "pmer"], "batch_size": 8, "device": "cpu"}
    ck = models.resolve(model)
    constructor = RetrievedTimestral if ck.family == "timestral" else RetrievedT0
    expected = constructor("retrieved", variables=list(cfg.targets), targets=list(cfg.targets),
        checkpoint=ck.repo, revision=ck.revision, context_days=30, calendar=False, arome="archived",
        prediction_target="arome_error", precision="bf16", index=index, method=method, k=2, **runtime)
    actual = build_adapter(cfg, "retrieved", index=index, **runtime)
    _assert_same_native_configuration(actual, expected)
    assert actual.index is index and actual.method == method and actual.k == 2


@pytest.mark.parametrize("mismatch", ["identity", "variables", "context"])
def test_factory_rejects_an_index_from_another_forecast_configuration(mismatch):
    from types import SimpleNamespace

    index = SimpleNamespace(ident="a" * 64, meta={"variables": ["t", "pmer"], "context_hours": 720,
                                                "retrieval": {"k": 2}})
    cfg = ForecastConfig.make("Timestral-209M", targets=("t", "pmer"), retrieval_method="shape",
                              retrieval_index=index.ident)
    if mismatch == "identity":
        index.ident = "b" * 64
    elif mismatch == "variables":
        index.meta["variables"] = ["pmer", "t"]
    else:
        index.meta["context_hours"] = 24
    with pytest.raises(ValueError, match="index identity, variables or context"):
        build_adapter(cfg, "wrong-index", index=index)
