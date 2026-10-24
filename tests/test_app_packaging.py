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
