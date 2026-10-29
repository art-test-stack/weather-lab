"""Source cache checks for the manual hourly collector; no network."""
import importlib.util
from pathlib import Path

import pytest

from weather_lab.common import ROOT, sha256_file


@pytest.fixture
def collector():
    path = ROOT / "experiments/hourly-archive-v1/fetch.py"
    spec = importlib.util.spec_from_file_location("hourly_archive_fetch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_source_cache_reused_and_changed_source_refused(tmp_path, monkeypatch, collector):
    path = tmp_path / "source.json"
    path.write_text("original")
    record = {"path": str(path), "url": "https://example.invalid/source", "sha256": sha256_file(path)}
    monkeypatch.setattr(collector, "_download", lambda *_: pytest.fail("cache hit downloaded"))
    assert collector.collect_file(record["url"], path, [record], None) == record
    with pytest.raises(ValueError, match="Unverified or changed"):
        collector.collect_file(record["url"] + "?different", path, [record], None)
    path.write_text("changed")
    with pytest.raises(ValueError, match="Unverified or changed"):
        collector.collect_file(record["url"], path, [record], None)


def test_new_source_is_staged_locally_and_recorded_snapshot_not_replaced(tmp_path, monkeypatch, collector):
    monkeypatch.setattr(collector, "ROOT", tmp_path)
    dest = tmp_path / "raw/source.json"

    def download(url, staging, session):
        assert staging.is_relative_to(tmp_path / ".tmp/codex")
        staging.parent.mkdir(parents=True)
        staging.write_text("new source")
        return {"url": url, "sha256": sha256_file(staging)}

    monkeypatch.setattr(collector, "_download", download)
    record = collector.collect_file("https://example.invalid/source", dest, [], None)
    assert dest.read_text() == "new source" and record["path"] == str(dest)
    dest.unlink()
    with pytest.raises(FileNotFoundError, match="restore"):
        collector.collect_file(record["url"], dest, [record], None)
