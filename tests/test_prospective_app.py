import gradio as gr
import pytest

from weather_lab.prospective_store import Archive
from weather_lab_app import prospective as view


def populate(archive):
    row = {"cohort": "prospective-2026-11", "station": "07149", "case_id": "one", "variable": "t",
           "valid_time": "2026-11-03T05:00Z", "lead_h": 1, "issue": "2026-11-03T00:00Z", "cycle": 0}
    with archive.writer():
        archive.put("catalog", "model", {"label": "Existing configuration", "config": {}, "checkpoint": {"revision": "pinned"}})
        for model, point in [("arome-raw", 13.), ("model", 11.)]:
            archive.put("predictions", model, {"model": model, "case_id": "one", "cohort": row["cohort"],
                "rows": [row | {"model": model, "point": point}], "provenance": {"code_revision": "frozen"}})
        archive.put("observations", "truth", {"identity": "07149/day", "collected_at": "2026-11-04T01:00Z",
            "rows": [{"station": "07149", "time": row["valid_time"], "t": 10., "pmer": None}]})


@pytest.mark.parametrize("populated", [False, True])
def test_gradio_empty_populated_and_read_only_callbacks(tmp_path, monkeypatch, populated):
    archive = Archive(tmp_path)
    if populated:
        populate(archive)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    monkeypatch.setattr(archive, "put", lambda *a, **k: pytest.fail("UI wrote an archive"))
    monkeypatch.setattr(archive, "sync", lambda: pytest.fail("UI contacted storage"))
    with gr.Blocks() as demo:
        handles = view.build("hourly-2026-v3", archive=archive)
    demo.get_config_file()
    assert len(demo.fns) == 9  # refresh plus eight saved-only filters
    output = handles["render"]("ALL", "", "", "t", ["model"], "prospective-2026-11", "ALL")
    if populated:
        assert len(output[1].data) == 4  # two metrics for model and reference
        model = output[5].set_index("model").loc["Existing configuration"]
        assert model.mae == model.rmse == 1
        assert model.gain == model.absolute_gain == 2
        assert "100.0%" in output[0]
        filtered = handles["render"]("07630", "", "", "t", ["model"], "prospective-2026-11", "ALL")
        assert "No saved records" in filtered[0]
        invalid = handles["render"]("ALL", "invalid", "", "t", ["model"], "prospective-2026-11", "ALL")
        assert "Invalid date" in invalid[0]
    else:
        assert "No frozen predictions" in output[0]
        assert not output[1].data
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before
