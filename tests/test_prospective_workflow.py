import subprocess
from pathlib import Path

import pytest
import yaml

from weather_lab.cli import main


def test_manual_workflow_dependencies_and_inactive_schedules():
    entry = yaml.load(Path(".github/workflows/prospective.yml").read_text(), Loader=yaml.BaseLoader)
    stage = yaml.load(Path(".github/workflows/prospective-stage.yml").read_text(), Loader=yaml.BaseLoader)
    assert set(entry["on"]) == {"workflow_dispatch"}
    assert set(stage["on"]) == {"workflow_call"}
    assert entry["concurrency"] == {"group": "prospective-bucket-writer", "cancel-in-progress": "false"}
    assert entry["jobs"]["inference"]["needs"] == "collect"
    assert entry["jobs"]["scoring"]["needs"] == ["collect", "inference"]
    scripts = [s["run"] for s in stage["jobs"]["stage"]["steps"] if "run" in s]
    for script in scripts:
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "--serialized-writer" in scripts[-1] and "--retrospective" in scripts[-1]
    assert "for attempt in 1 2 3" in scripts[-1]


def test_cli_saved_only_and_serialized_writer_guard(tmp_path):
    main(["prospective", "sync", "--archive-root", str(tmp_path)])
    main(["prospective", "score", "--archive-root", str(tmp_path)])
    with pytest.raises(ValueError, match="serialized"):
        main(["prospective", "score", "--archive-root", str(tmp_path), "--remote"])
    with pytest.raises(ValueError, match="current UTC"):
        main(["prospective", "sync", "--archive-root", str(tmp_path), "--end", "2099-01-01"])


def test_backfill_fetches_context_and_verifying_window(tmp_path, monkeypatch):
    from weather_lab import prospective_collect as collection
    from weather_lab.prospective_store import Archive
    from weather_lab.rag_store import Store

    archive = Archive(tmp_path / "archive")
    with archive.writer():
        archive.put("catalog", "existing", {"label": "existing"})
        archive.put("status", "catalog-complete", {"configurations": ["existing"]})
    calls = []
    monkeypatch.setattr(Store, "get", lambda *a: tmp_path / "source")
    monkeypatch.setattr(collection, "backfill_saved", lambda *a: calls.append("saved"))
    monkeypatch.setattr(collection, "collect_forecasts", lambda *a: calls.append("forecasts"))
    monkeypatch.setattr(collection, "collect_observations", lambda archive, start, end: calls.append((start, end)))
    monkeypatch.setattr(collection, "record_backfill_gaps", lambda *a: calls.append("gaps"))
    main(["prospective", "backfill", "--archive-root", str(archive.root), "--start", "2026-10-01",
          "--end", "2026-10-02", "--allow-large-downloads"])
    assert calls[0:2] == ["saved", "forecasts"] and calls[-1] == "gaps"
    assert calls[2][0].isoformat() == "2026-07-03T00:00:00+00:00"
    assert calls[2][1].isoformat() == "2026-10-04T03:00:00+00:00"
