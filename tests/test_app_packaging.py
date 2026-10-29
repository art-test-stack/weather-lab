"""Package boundary and Space build output (no network)."""

from weather_lab_app import benchmark as bench
from weather_lab_app import forecast_view as view
from weather_lab_app import labels
from weather_lab_app import main as application
from weather_lab_app import methods
from weather_lab_app import results

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


def test_callback_owners_do_not_import_application_composition():
    for name in ("forecast_state", "forecast_view", "benchmark", "saved_benchmark", "results", "methods"):
        path = ROOT / "src/weather_lab_app" / f"{name}.py"
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "weather_lab_app.main", path.name
                assert not (node.module == "weather_lab_app" and any(a.name == "main" for a in node.names)), path.name
            elif isinstance(node, ast.Import):
                assert not any(a.name == "weather_lab_app.main" for a in node.names), path.name


def test_latin_modern_assets_are_served_locally_and_shared_with_charts(monkeypatch):
    from fastapi.testclient import TestClient
    from gradio.routes import App
    from weather_lab_app import forecast_state as main, plots

    monkeypatch.setattr(application, "DEFAULT_SUITE", None)
    demo = application.build()
    client = TestClient(App.create_app(demo))
    assert plots.FONT == "Latin Modern Roman, Georgia, serif"
    for face in ("regular", "bold", "italic", "bolditalic"):
        path = application.FONT_DIR / f"lmroman10-{face}.otf"
        response = client.get(f"/gradio_api/file={path}")
        assert response.status_code == 200 and response.content == path.read_bytes()
        assert str(path) in application.CSS
    assert "font-display: swap" in application.CSS


def test_space_build_has_metadata_launcher_and_no_data(tmp_path):
    out = build_space(tmp_path / "space", log=lambda *_: None)
    readme = (out / "README.md").read_text()
    front = yaml.safe_load(readme.split("---\n")[1])
    assert front["sdk"] == "gradio" and front["app_file"] == "app.py" and front["title"] == "Weather-lab"
    assert len(front["short_description"]) <= 60
    from weather_lab.models import CHECKPOINTS
    for name in ("Timestral-28M", "Timestral-209M"):           # pinned, full revisions, inference files only
        c = CHECKPOINTS[name]
        assert f"{c.repo} config.json,model.safetensors {c.revision}" in front["preload_from_hub"]
    assert "weather_lab_app.main import main" in (out / "app.py").read_text()
    assert not (ROOT / "README.md").read_text().startswith("---")       # GitHub README stays plain
    files = [p for p in out.rglob("*") if p.is_file()]
    assert (out / "src" / "weather_lab_app" / "main.py").exists() and (out / "configs" / "bootstrap.yaml").exists()
    assert (out / "src/weather_lab_app/assets/fonts/lmroman10-regular.otf").exists()
    assert (out / "src/weather_lab_app/assets/fonts/GUST-FONT-LICENSE.txt").exists()
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


def _applied(size="28M", setup="archived", source="saved", primary=None, exploratory=None, case="c1"):
    from weather_lab.forecast_config import ForecastConfig

    return {"suite": "arome-issued-v2", "case": case, "size": size, "setup": setup, "ctx": 30, "horizon": 48,
            "source": source, "primary": primary or f"timestral-{size.lower()}-mv-cal-arome-{setup}",
            "settings": {}, "exploratory": exploratory or {}, "shown_h": 48,
            "config": ForecastConfig.make(f"Timestral-{size}", setup)}


def test_explorer_overlays_persist_across_variables_views_and_exploratory_runs(monkeypatch):
    import pandas as pd

    from weather_lab_app import forecast_state as app

    preds = pd.DataFrame({"model": ["persistence", "arome-raw", "timestral-28m-mv-cal", "timestral-209m-mv-cal",
                                    "persistence", "timestral-28m-mv-cal", "timestral-209m-mv-cal"],
                          "variable": ["t", "t", "t", "t", "u", "u", "u"],
                          "output_type": ["point", "point", "quantiles", "quantiles", "point", "quantiles", "quantiles"]})

    class FakeRun:
        models = ["persistence", "arome-raw", "timestral-28m-mv-cal", "timestral-209m-mv-cal"]

        def predictions(self):
            return preds

    monkeypatch.setattr(results, "RUNS", {"arome-issued-v2": FakeRun()})
    a = _applied("28M", "none", primary="timestral-28m-mv-cal")
    choices, value, _, band = app.select_systems(a, "t", {}, None)
    assert value == ["arome-raw", "timestral-28m-mv-cal"] and band == "timestral-28m-mv-cal"
    pref = app.update_pref(a, "t", ["arome-raw", "timestral-28m-mv-cal", "persistence"], {})   # add an overlay
    _, value, _, _ = app.select_systems(a, "u", pref, None)                  # no AROME wind
    assert "arome-raw" not in value and "persistence" in value
    b = _applied("209M", "none", primary="timestral-209m-mv-cal")            # another applied view keeps the overlay
    _, value, _, band = app.select_systems(b, "t", pref, "timestral-28m-mv-cal")
    assert value == ["persistence", "arome-raw", "timestral-209m-mv-cal"] and band == "timestral-209m-mv-cal"
    mid = "exploratory Timestral-209M ctx14d h72 [cpu]"
    e = _applied("209M", "none", source="exploratory", primary=mid,
                 exploratory={mid: pd.DataFrame({"variable": ["t"]})})
    _, value, _, band = app.select_systems(e, "t", pref, "timestral-209m-mv-cal")
    assert mid in value and band == mid                                      # interval source = applied result


def test_labels_cover_every_configured_system():
    import glob

    from weather_lab_app.labels import MODELS, STYLE

    ids = {m["id"] for p in glob.glob(str(ROOT / "configs" / "**" / "*.yaml"), recursive=True)
           for m in (yaml.safe_load(open(p)) or {}).get("models", []) or []}
    missing = sorted(i for i in ids if i not in MODELS or i not in STYLE)
    assert not missing, f"systems without a readable name or colour: {missing}"


def test_custom_forecast_names_and_colours_distinguish_prediction_methods():
    from weather_lab_app.labels import model_description, model_name, model_style

    for size in ("28M", "209M"):
        direct = f"exploratory Timestral-{size} ctx90d h48 +resid +atmos +solar +tend [cuda]"
        assert model_name(direct) == f"Timestral {size} · direct"
        assert model_style(direct)[0] == "#184f95"
        for conditioning in ("archived history", "observed history", "forecast only"):
            conditioned = direct.replace(" +resid", f" +AROME {conditioning} +resid")
            correction = conditioned.replace(" [cuda]", " +AROME error correction [cuda]")
            assert model_name(conditioned) == f"Timestral {size} · direct + AROME"
            assert model_name(correction) == f"Timestral {size} · AROME error correction"
            assert model_name(correction.replace("ctx90d", "ctx30d").replace("[cuda]", "[cpu]")) == model_name(correction)
            assert model_style(conditioned)[0] == "#0f7a54"
            assert model_style(correction) == ("#6250d6", "solid", 2.8)
            assert "outside the frozen benchmark" in model_description(correction)
        unconditioned_correction = direct.replace(" [cuda]", " +AROME error correction [cuda]")
        assert model_name(unconditioned_correction) == f"Timestral {size} · AROME error correction"
        assert model_style(unconditioned_correction) == model_style(correction)
    assert model_name("timestral-209m-mv-cal-arome-archived") == "Timestral 209M + AROME"
    assert model_name("future-model") == "future-model"


def _fake_v2_run(mae_by_model: dict[str, float], variable: str = "t", extra_models: tuple = ()):
    import pandas as pd

    rows = {"split": "test", "variable": variable, "station": "ALL", "season": "ALL", "lead_h": float("nan"),
            "model": list(mae_by_model), "n": 960}

    class FakeRun:
        suite = "arome-issued-v2"
        splits = ["test"]
        source = "test fixture"
        run = {"run_id": "fake"}
        models = list(mae_by_model) + list(extra_models)      # e.g. AROME saved, but not for this variable
        tables = {"metrics_point": pd.DataFrame({**rows, "mae": list(mae_by_model.values()),
                                                "rmse": [v * 1.2 for v in mae_by_model.values()]}),
                  "metrics_probabilistic": pd.DataFrame({**rows, "coverage80": 0.8, "pinball": 0.5})}

        def predictions(self):
            return pd.DataFrame({"model": self.models, "variable": variable,
                                 "output_type": ["point" if m == "arome-raw" else "quantiles" for m in self.models]})

    return FakeRun()


def test_main_metric_changes_saved_cards_finding_and_matrix(monkeypatch):
    from weather_lab_app import forecast_state as app

    mid = "timestral-209m-mv-cal-arome-archived"
    run = _fake_v2_run({"arome-raw": 1.0, mid: 1.2})
    point = run.tables["metrics_point"]
    point.loc[point.model == "arome-raw", "rmse"] = 2.0
    point.loc[point.model == mid, "rmse"] = 1.3
    monkeypatch.setattr(results, "RUNS", {run.suite: run})
    assert "AROME remains most accurate" in bench.finding(run.suite, "t")
    text = bench.finding(run.suite, "t", metric="rmse")
    assert "root mean squared error" in text and "is more accurate than AROME" in text
    assert "AROME remains most accurate" not in text
    cards = view.tiles_html(run.suite, "t", "209M", metric="rmse")
    assert "1.30 °C</div>" in cards and "RMSE · MAE 1.20 °C" in cards
    matrix = bench.matrix_html(run.suite, "t", metric="rmse")
    assert "Root-mean-square error: 1.30 °C" in matrix


V2_T = {"arome-raw": 1.09, "timestral-28m-mv-cal": 2.52, "timestral-209m-mv-cal": 2.42,
        "timestral-28m-mv-cal-arome-archived": 2.19, "timestral-209m-mv-cal-arome-archived": 1.90,
        "timestral-28m-mv-cal-arome-substobs": 1.0}          # hatched (different input): never "most accurate"


def test_checkpoint_selector_switches_headline_defaults_and_preferences(monkeypatch):
    from weather_lab_app import forecast_state as app
    from weather_lab_app.labels import with_checkpoint

    assert with_checkpoint("timestral-28m-mv-cal-arome-archived", "209M") == "timestral-209m-mv-cal-arome-archived"
    assert with_checkpoint("arome-raw", "209M") == "arome-raw"
    monkeypatch.setattr(results, "RUNS", {"arome-issued-v2": _fake_v2_run(V2_T)})
    v = app.experiment_view("arome-issued-v2", "209M")
    assert v["headline"] == ["arome-raw", "timestral-209m-mv-cal", "timestral-209m-mv-cal-arome-archived"]
    assert v["compare"] == {"timestral-209m-mv-cal-arome-archived": "timestral-209m-mv-cal"}
    assert v["band"] == "timestral-209m-mv-cal-arome-archived" and app.has_checkpoints("arome-issued-v2")
    assert app.experiment_view("arome-issued-v2", "28M")["headline"][1] == "timestral-28m-mv-cal"
    _, value, _, band = app.select_systems(_applied("209M", "archived"), "t", {}, None)
    assert value == v["default_systems"] and band == "timestral-209m-mv-cal-arome-archived"
    assert "209M" in view.tiles_html("arome-issued-v2", "t", "209M") and "1.90" in view.tiles_html(
        "arome-issued-v2", "t", "209M")


def test_benchmark_finding_only_states_what_the_scores_support(monkeypatch):
    from weather_lab_app import forecast_state as app

    monkeypatch.setattr(results, "RUNS", {"arome-issued-v2": _fake_v2_run(V2_T)})
    text = bench.finding("arome-issued-v2", "t")
    assert ("AROME remains most accurate; adding the AROME forecast + archived AROME history improves Timestral, "
            "with the larger checkpoint performing better.") in text
    assert "within what this small sample" not in text
    wind = {k: v for k, v in V2_T.items() if k != "arome-raw"} | {"timestral-28m-mv-cal": 1.883,
                                                                   "timestral-28m-mv-cal-arome-archived": 1.887}
    monkeypatch.setattr(results, "RUNS", {"arome-issued-v2": _fake_v2_run(wind, "u", extra_models=("arome-raw",))})
    text = bench.finding("arome-issued-v2", "u")
    assert "AROME is not scored for this variable" in text and "improves Timestral 209M only" in text
    assert "within what this small sample" in text
    tiny = V2_T | {"timestral-28m-mv-cal-arome-futureonly": 2.513, "timestral-209m-mv-cal-arome-futureonly": 2.426}
    monkeypatch.setattr(results, "RUNS", {"arome-issued-v2": _fake_v2_run(tiny)})
    text = bench.finding("arome-issued-v2", "t", setup="futureonly")       # < 1 % either way: no direction claimed
    assert "leaves Timestral's error essentially unchanged (within 1%)" in text and "improves" not in text
    text = bench.finding("arome-issued-v2", "t", setup="none")
    assert "without AROME input, the larger Timestral checkpoint performs better" in text and "adding" not in text
    monkeypatch.setattr(results, "RUNS", {"arome-issued-v2": _fake_v2_run({"timestral-28m-mv-cal": 2.0})})
    assert bench.finding("arome-issued-v2", "t") == ""


def test_checkpoint_selector_hidden_where_both_checkpoints_are_already_shown(monkeypatch):
    from weather_lab_app import forecast_state as app

    run = _fake_v2_run({"persistence": 3.0, "timestral-28m-mv-cal": 2.5, "timestral-209m-mv-cal": 2.4})
    run.suite = "synop-historical-v1"
    monkeypatch.setattr(results, "RUNS", {"synop-historical-v1": run})
    assert not app.has_checkpoints("synop-historical-v1")
    view = app.experiment_view("synop-historical-v1", "209M")
    assert view["headline"] == labels.EXPERIMENTS["synop-historical-v1"]["headline"] and view["missing"] == []


def test_input_setup_switches_conditioned_system_and_reports_not_run(monkeypatch):
    from dataclasses import replace
    from weather_lab import models
    from weather_lab_app import forecast_state as app

    monkeypatch.setitem(models.FAMILIES, "t0", replace(models.FAMILIES["t0"], status="planned"))
    monkeypatch.setattr(results, "RUNS", {"arome-issued-v2": _fake_v2_run(V2_T)})   # no 209M observed-history run
    v = app.experiment_view("arome-issued-v2", "28M", "substobs")
    assert v["headline"] == ["arome-raw", "timestral-28m-mv-cal", "timestral-28m-mv-cal-arome-substobs"]
    assert v["band"] == "timestral-28m-mv-cal-arome-substobs" and v["missing"] == []
    v = app.experiment_view("arome-issued-v2", "209M", "substobs")
    assert v["missing"] == ["timestral-209m-mv-cal-arome-substobs"] and v["band"] == ""   # never substituted
    assert "timestral-209m-mv-cal-arome-substobs" not in v["default_systems"]
    assert "not run" in view.tiles_html("arome-issued-v2", "t", "209M", "substobs")
    text = bench.finding("arome-issued-v2", "t", setup="substobs")
    assert "Not run: Timestral 209M + AROME (observed history)" in text
    assert "different input construction" in text
    _, value, _, band = app.select_systems(_applied("28M", "substobs"), "t", {}, None)
    assert "timestral-28m-mv-cal-arome-substobs" in value and "timestral-28m-mv-cal-arome-archived" not in value
    html = bench.matrix_html("arome-issued-v2", "t")
    assert html.count("not run") >= 2 and "not integrated yet" in html


def test_benchmark_views_never_load_a_model(monkeypatch):
    """Browsing saved results reads files only: loading a model here is a bug."""
    from weather_lab import models
    from weather_lab_app import forecast_state as app

    def boom(*a, **k):
        raise AssertionError("a model was loaded while browsing saved results")

    monkeypatch.setattr(models.ModelCache, "get", boom)
    monkeypatch.setattr(models, "prefetch", boom)
    monkeypatch.setattr(results, "RUNS", {"arome-issued-v2": _fake_v2_run(V2_T)})
    view.tiles_html("arome-issued-v2", "t", "209M", "archived")
    bench.finding("arome-issued-v2", "t")
    bench.matrix_html("arome-issued-v2", "t")
    methods.cache_md()
