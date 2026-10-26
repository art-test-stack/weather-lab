"""Focused CLI execution/replay contracts for native foundation adapters; no inference or downloads."""
from pathlib import Path

import pandas as pd
import pytest

from weather_lab import evaluate, models
from weather_lab.common import Config


@pytest.fixture
def no_weight_loading(monkeypatch):
    monkeypatch.setattr(models.MODELS, "get", lambda *args, **kwargs: pytest.fail("test loaded model weights"))


def _config(tmp_path: Path | None = None, context_days: int = 30, model_overrides: bool = True) -> Config:
    timestral = models.CHECKPOINTS["Timestral-28M"]
    t0 = models.CHECKPOINTS["t0-gamma"]
    models_config = [
        {"id": "timestral-test", "kind": "timestral", "targets": ["t"]},
        {"id": "t0-test", "kind": "t0", "targets": ["t"],
         **({"context_days": 90} if model_overrides else {})},
    ]
    raw = {
        "suite": "adapter-execution-test",
        "manifest": str(tmp_path / "manifest.json") if tmp_path else "manifest.json",
        "paths": {"runs": str(tmp_path / "runs") if tmp_path else "runs"},
        "variables": ["t", "pmer", "u", "v"],
        "forecast": {"step_hours": 3, "horizon_steps": 16, "context_days": context_days,
                     "highlight_leads_h": [6, 24, 48]},
        "timestral": {"checkpoint": timestral.repo, "revision": timestral.revision},
        "t0": {"checkpoint": t0.repo, "revision": t0.revision},
        "models": models_config,
    }
    return Config(raw, (tmp_path / "test.yaml") if tmp_path else Path("test.yaml"))


def _record_with_native_prediction_revision(adapter) -> dict:
    # Native formats: Timestral.predict returns repo@12-char revision (forecast.py:490);
    # T0.predict returns repo@full revision (t0_adapter.py:145). No prediction is executed here.
    if adapter.kind == "timestral":
        revision = f"{adapter.checkpoint}@{adapter.revision[:12]}"
    elif adapter.kind == "t0":
        revision = f"{adapter.checkpoint}@{adapter.revision}"
    else:  # pragma: no cover - the fixture builds only these two native adapter families
        raise AssertionError(adapter.kind)
    return adapter.describe() | {"revision": revision}


def test_saved_system_match_uses_each_native_revision_format_and_rejects_changes(no_weight_loading):
    cfg = _config()
    adapters = evaluate.build_adapters(cfg)
    assert {adapter.kind for adapter in adapters} == {"timestral", "t0"}

    for adapter in adapters:
        record = _record_with_native_prediction_revision(adapter)
        assert evaluate._same_system(adapter, record)
        changed = record | {"revision": record["revision"] + "-changed"}
        assert not evaluate._same_system(adapter, changed)


def test_run_evaluation_sizes_history_for_t0_context_override_before_inference(
    tmp_path, monkeypatch, no_weight_loading
):
    cfg = _config(tmp_path, context_days=30)
    cfg.manifest_path.touch()
    manifest = {"case_data_sha256": "fixture", "dataset": {"stations": [
        {"id": "07149", "lat": 48.7, "lon": 2.3},
    ]}}
    origin = pd.Timestamp("2026-10-10T06:00:00Z")
    cases = pd.DataFrame([{"case_id": "fixture/07149/20261010T06", "station": "07149", "origin": origin,
                           "split": "validation", "season": "SON", "eligible": True}])
    obs = pd.DataFrame({"station": ["07149"], "time": [origin], "t": [12.0]})
    monkeypatch.setattr(evaluate, "read_json", lambda path: manifest)
    monkeypatch.setattr(evaluate, "load_observations", lambda _cfg: (obs, {"sha256": "observations"}))
    monkeypatch.setattr(evaluate, "cases_from_manifest", lambda _manifest: cases)
    monkeypatch.setattr(evaluate, "case_data_checksum", lambda *_args: "fixture")
    monkeypatch.setattr(evaluate, "split_bounds", lambda _cfg: {"fit": (origin, origin)})

    class StopBeforeInference(Exception):
        pass

    captured = {}

    def capture_batch(_cfg, _obs, _cases, history_steps, *_args, **_kwargs):
        captured["history_steps"] = history_steps
        raise StopBeforeInference

    monkeypatch.setattr(evaluate, "make_batch", capture_batch)
    with pytest.raises(StopBeforeInference):
        evaluate.run_evaluation(cfg, log=lambda *_args, **_kwargs: None)

    assert captured["history_steps"] == 90 * 24 // cfg["forecast"]["step_hours"]
