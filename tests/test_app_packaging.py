"""Package boundary and Space build output (no network)."""

import ast
from pathlib import Path

import yaml

from weather_lab.common import ROOT
from weather_lab_app.space import build_space

SCIENCE = ROOT / "src" / "weather_lab"


def test_science_package_never_imports_the_app():
    for path in SCIENCE.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(n.startswith(("weather_lab_app", "gradio")) for n in names), f"{path.name} imports {names}"


def test_space_build_has_metadata_launcher_and_no_data(tmp_path):
    out = build_space(tmp_path / "space", log=lambda *_: None)
    readme = (out / "README.md").read_text()
    front = yaml.safe_load(readme.split("---\n")[1])
    assert front["sdk"] == "gradio" and front["app_file"] == "app.py" and front["title"] == "Weather-lab"
    assert len(front["short_description"]) <= 60
    assert "weather_lab_app.main import main" in (out / "app.py").read_text()
    assert not (ROOT / "README.md").read_text().startswith("---")       # GitHub README stays plain
    files = [p for p in out.rglob("*") if p.is_file()]
    assert (out / "src" / "weather_lab_app" / "main.py").exists() and (out / "configs" / "bootstrap.yaml").exists()
    assert not [p for p in files if p.suffix in {".json", ".parquet", ".csv", ".gz", ".grib2", ".safetensors"}]
    assert not any("tests" in p.relative_to(out).parts or ".github" in p.relative_to(out).parts for p in files)


def test_zerogpu_decorator_is_transparent_off_zerogpu(monkeypatch):
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    from weather_lab_app import zerogpu

    @zerogpu.gpu(duration=60)
    def f(x, y=1):
        return x + y

    assert f(2, y=3) == 5 and not zerogpu.on_zerogpu()
    assert zerogpu.inference_device() in ("cpu", "cuda")


def test_score_references_are_keyed_by_environment(tmp_path):
    from weather_lab.specs import _score_references, environment_key

    legacy = {"expected_scores": {"metrics_point.csv": "abc"}}
    assert _score_references(legacy) == {"torch-2.14.1/darwin-arm64": {"metrics_point.csv": "abc"}}
    env_old = {"torch": "2.14.1", "machine": "arm64", "platform": "macOS-26.5-arm64-arm-64bit"}
    env_new = {"torch": "2.13.0", "machine": "x86_64", "platform": "Linux-6", "sys_platform": "linux"}
    assert environment_key(env_old) == "torch-2.14.1/darwin-arm64"
    assert environment_key(env_new) == "torch-2.13.0/linux-x86_64"


def test_lockfile_matches_what_the_spaces_builder_installs():
    """The builder adds `gradio[oauth,mcp]==<sdk_version>` (and torch<=2.13.0 on ZeroGPU) on top of
    requirements.txt, so those must already be part of the locked resolution."""
    import tomllib

    meta = yaml.safe_load((ROOT / "src" / "weather_lab_app" / "space.yaml").read_text())
    reqs = {l.split("==")[0]: l.split("==")[1].split()[0] for l in (ROOT / "requirements.txt").read_text().splitlines()
            if "==" in l and not l.startswith(("#", " "))}
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    assert reqs["gradio"] == meta["sdk_version"]
    assert any(d.startswith("gradio[oauth,mcp]") for d in deps)
    assert "mcp" in reqs and "authlib" in reqs                       # the extras are in the lock
    if meta.get("suggested_hardware", "").startswith("zero"):
        major, minor = (int(x) for x in reqs["torch"].split(".")[:2])
        assert (major, minor) <= (2, 13)                               # ZeroGPU supports torch <= 2.13.0


def test_checkpoint_is_fetched_before_any_gpu_work(monkeypatch):
    import pytest

    from weather_lab import forecast

    calls = []
    monkeypatch.setattr(forecast, "weights_cached", lambda c, r: False)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo, name, revision: calls.append((repo, name)))
    with pytest.raises(PermissionError):
        forecast.ensure_checkpoint("Timestral-209M", allow_download=False)
    assert forecast.ensure_checkpoint("Timestral-209M", allow_download=True) is True
    assert calls == [("mistralai/Timestral-209M", "config.json"), ("mistralai/Timestral-209M", "model.safetensors")]
    monkeypatch.setattr(forecast, "weights_cached", lambda c, r: True)
    assert forecast.ensure_checkpoint("Timestral-28M", allow_download=False) is False


def test_explorer_systems_persist_across_variables_and_exploratory_runs(monkeypatch):
    import pandas as pd

    from weather_lab_app import main as app

    preds = pd.DataFrame({"model": ["persistence", "arome-raw", "timestral-28m-mv-cal", "timestral-209m-mv-cal",
                                    "persistence", "timestral-28m-mv-cal", "timestral-209m-mv-cal"],
                          "variable": ["t", "t", "t", "t", "u", "u", "u"],
                          "output_type": ["point", "point", "quantiles", "quantiles", "point", "quantiles", "quantiles"]})

    class FakeRun:
        models = ["persistence", "arome-raw", "timestral-28m-mv-cal", "timestral-209m-mv-cal"]

        def predictions(self):
            return preds

    s_id = "arome-issued-v2"                       # defaults: AROME, Timestral 28M (+ archived, absent here)
    monkeypatch.setattr(app, "RUNS", {s_id: FakeRun()})
    extra = app.keep_extra_for_case("c1", app.EMPTY_EXTRA)
    s, b = app.systems_update(s_id, "c1", "t", {}, None, extra)
    assert s["value"] == ["arome-raw", "timestral-28m-mv-cal"] and b["value"] == "timestral-28m-mv-cal"
    assert ("AROME", "arome-raw") in s["choices"]                      # readable names, exact ids as values
    pref = app.update_pref(s_id, "c1", "t", ["arome-raw", "timestral-209m-mv-cal"], {}, extra)
    s, _ = app.systems_update(s_id, "c1", "u", pref, None, extra)       # no AROME wind
    assert "arome-raw" not in [v for _, v in s["choices"]] and s["value"] == ["timestral-209m-mv-cal"]
    s, _ = app.systems_update(s_id, "c1", "t", pref, None, extra)       # preference restored
    assert s["value"] == ["arome-raw", "timestral-209m-mv-cal"]
    mid = "exploratory Timestral-209M ctx14d [cpu]"
    extra = {"case": "c1", "runs": {mid: pd.DataFrame()}}
    pref = {s_id: pref[s_id] + [mid]}
    pref = app.update_pref(s_id, "c1", "t", [mid], pref, extra)         # untick the saved systems
    s, b = app.systems_update(s_id, "c1", "t", pref, "timestral-209m-mv-cal", extra)
    assert s["value"] == [mid] and b["value"] == mid                    # band follows the selected systems
    assert app.keep_extra_for_case("c2", extra)["runs"] == {}           # another case starts fresh


def test_labels_cover_every_configured_system():
    import glob

    from weather_lab_app.labels import MODELS, STYLE

    ids = {m["id"] for p in glob.glob(str(ROOT / "configs" / "**" / "*.yaml"), recursive=True)
           for m in (yaml.safe_load(open(p)) or {}).get("models", []) or []}
    missing = sorted(i for i in ids if i not in MODELS or i not in STYLE)
    assert not missing, f"systems without a readable name or colour: {missing}"


def _fake_v2_run(mae_by_model: dict[str, float], variable: str = "t", extra_models: tuple = ()):
    import pandas as pd

    rows = {"split": "test", "variable": variable, "station": "ALL", "season": "ALL", "lead_h": float("nan"),
            "model": list(mae_by_model), "n": 960}

    class FakeRun:
        suite = "arome-issued-v2"
        splits = ["test"]
        models = list(mae_by_model) + list(extra_models)      # e.g. AROME saved, but not for this variable
        tables = {"metrics_point": pd.DataFrame({**rows, "mae": list(mae_by_model.values())}),
                  "metrics_probabilistic": pd.DataFrame({**rows, "coverage80": 0.8})}

        def predictions(self):
            return pd.DataFrame({"model": self.models, "variable": variable,
                                 "output_type": ["point" if m == "arome-raw" else "quantiles" for m in self.models]})

    return FakeRun()


V2_T = {"arome-raw": 1.09, "timestral-28m-mv-cal": 2.52, "timestral-209m-mv-cal": 2.42,
        "timestral-28m-mv-cal-arome-archived": 2.19, "timestral-209m-mv-cal-arome-archived": 1.90,
        "timestral-28m-mv-cal-arome-substobs": 1.0}          # hatched (different input): never "most accurate"


def test_checkpoint_selector_switches_headline_defaults_and_preferences(monkeypatch):
    from weather_lab_app import main as app
    from weather_lab_app.labels import with_checkpoint

    assert with_checkpoint("timestral-28m-mv-cal-arome-archived", "209M") == "timestral-209m-mv-cal-arome-archived"
    assert with_checkpoint("arome-raw", "209M") == "arome-raw"
    monkeypatch.setattr(app, "RUNS", {"arome-issued-v2": _fake_v2_run(V2_T)})
    v = app.experiment_view("arome-issued-v2", "209M")
    assert v["headline"] == ["arome-raw", "timestral-209m-mv-cal", "timestral-209m-mv-cal-arome-archived"]
    assert v["compare"] == {"timestral-209m-mv-cal-arome-archived": "timestral-209m-mv-cal"}
    assert v["band"] == "timestral-209m-mv-cal-arome-archived" and app.has_checkpoints("arome-issued-v2")
    assert app.experiment_view("arome-issued-v2", "28M")["headline"][1] == "timestral-28m-mv-cal"
    s, b = app.systems_update("arome-issued-v2", "c1", "t", {}, None, app.EMPTY_EXTRA, "209M")
    assert s["value"] == v["default_systems"] and b["value"] == "timestral-209m-mv-cal-arome-archived"
    pref = {"arome-issued-v2": ["timestral-28m-mv-cal", "timestral-28m-mv-cal-arome-substobs", "exploratory x"]}
    out = app.switch_checkpoint("arome-issued-v2", "209M", pref)["arome-issued-v2"]
    assert out == ["timestral-209m-mv-cal", "timestral-28m-mv-cal-arome-substobs", "exploratory x"]  # no 209M twin
    assert "209M" in app.tiles_html("arome-issued-v2", "t", "209M") and "1.90" in app.tiles_html(
        "arome-issued-v2", "t", "209M")


def test_benchmark_finding_only_states_what_the_scores_support(monkeypatch):
    from weather_lab_app import main as app

    monkeypatch.setattr(app, "RUNS", {"arome-issued-v2": _fake_v2_run(V2_T)})
    text = app.finding("arome-issued-v2", "t")
    assert ("AROME remains most accurate; conditioning on AROME improves Timestral, with the larger checkpoint "
            "performing better.") in text and "within what this small sample" not in text
    wind = {k: v for k, v in V2_T.items() if k != "arome-raw"} | {"timestral-28m-mv-cal": 1.883,
                                                                   "timestral-28m-mv-cal-arome-archived": 1.887}
    monkeypatch.setattr(app, "RUNS", {"arome-issued-v2": _fake_v2_run(wind, "u", extra_models=("arome-raw",))})
    text = app.finding("arome-issued-v2", "u")
    assert "AROME is not scored for this variable" in text and "improves Timestral 209M only" in text
    assert "within what this small sample" in text
    monkeypatch.setattr(app, "RUNS", {"arome-issued-v2": _fake_v2_run({"timestral-28m-mv-cal": 2.0})})
    assert app.finding("arome-issued-v2", "t") == ""


def test_checkpoint_selector_hidden_where_both_checkpoints_are_already_shown(monkeypatch):
    from weather_lab_app import main as app

    run = _fake_v2_run({"persistence": 3.0, "timestral-28m-mv-cal": 2.5, "timestral-209m-mv-cal": 2.4})
    run.suite = "synop-historical-v1"
    monkeypatch.setattr(app, "RUNS", {"synop-historical-v1": run})
    assert not app.has_checkpoints("synop-historical-v1")
    assert app.experiment_view("synop-historical-v1", "209M") is app.EXPERIMENTS["synop-historical-v1"]
