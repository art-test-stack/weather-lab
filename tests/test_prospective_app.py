from html.parser import HTMLParser
import json

import gradio as gr
import pandas as pd
import pytest

from weather_lab.prospective_store import Archive
from weather_lab_app import date_picker
from weather_lab_app import prospective as view


def populate(archive, model_count=1):
    models = ["model", *[f"model-{i}" for i in range(1, model_count)]]
    origin = pd.Timestamp("2026-11-03T04:00Z")
    times = pd.date_range(origin + pd.Timedelta(hours=1), periods=48, freq="h")
    with archive.writer():
        for i, model in enumerate(models):
            archive.put("catalog", model, {"label": "Existing configuration" if i == 0 else
                f"Existing configuration {i} · long saved configuration label", "config": {
                    "model": "Timestral-209M", "conditioning": "none", "prediction_target": "weather", "features": []},
                "checkpoint": {"revision": "pinned"}})
        archive.put("status", "catalog-complete", {"task": "catalog", "status": "ok",
            "collected_at": "2026-11-01T00:00Z", "configurations": models})
        for model in ["arome-raw", *models]:
            rows = [{"cohort": "prospective-2026-11", "station": "07149", "case_id": "one", "variable": "t",
                     "valid_time": str(valid), "lead_h": lead + 1, "issue": "2026-11-03T00:00Z", "cycle": 0,
                     "model": model, "point": 13. if model == "arome-raw" else 11.}
                    for lead, valid in enumerate(times)]
            archive.put("predictions", model, {"model": model, "case_id": "one", "cohort": "prospective-2026-11",
                                                "rows": rows, "provenance": {"code_revision": "frozen"}})
        archive.put("observations", "truth", {"identity": "07149/verification", "collected_at": "2026-11-06T00:00Z",
            "rows": [{"station": "07149", "time": str(valid), "t": 10., "pmer": None} for valid in times]})
    return models


@pytest.mark.parametrize("populated", [False, True])
def test_only_refresh_can_read_and_update_results(tmp_path, monkeypatch, populated):
    monkeypatch.setattr("weather_lab.prospective.now_utc", lambda: "2026-11-06T01:00:00Z")
    archive = Archive(tmp_path)
    if populated:
        populate(archive)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    monkeypatch.setattr(archive, "put", lambda *a, **k: pytest.fail("UI wrote an archive"))
    monkeypatch.setattr(archive, "sync", lambda: pytest.fail("UI contacted remote storage"))
    with monkeypatch.context() as edits:
        edits.setattr(view.engine, "evaluate", lambda *a, **k: pytest.fail("Opening/editing triggered evaluation"))
        with gr.Blocks() as demo:
            handles = view.build("hourly-2026-v3", archive=archive)
        values = [control.value for control in handles["controls"]]
        handles["edit"](*values, handles["available"].value, None)
        if populated:
            values[1] = {"date": "2026-11-05"}
            _, last, _ = handles["edit"](*values, handles["available"].value, None)
            assert CalendarButtons(last["html"]).enabled == ["2026-11-05"]
    config = demo.get_config_file()
    result_ids = {component._id for component in handles["outputs"]}
    writers = [dep for dep in config["dependencies"] if result_ids.intersection(dep["outputs"])]
    assert len(writers) == 1
    assert writers[0]["targets"] == [(handles["refresh"]._id, "click")]
    assert handles["controls"][0].choices == view.STATIONS
    assert set(value for _, value in view.STATIONS) == {"ALL", "07149", "07630"}
    assert isinstance(handles["controls"][4], gr.Dropdown)  # compact multi-select, like Benchmark
    for component in handles["outputs"][1:8]:
        assert not json.loads(component.value["plot"])["data"]  # no automatic initial result
    result = handles["render"]("ALL", "", "", "t", ["model"], "prospective-2026-11", "ALL")
    if populated:
        model = result[8].set_index("model").loc["Existing configuration"]
        assert model.mae == model.rmse == 1
        assert model.gain == model.absolute_gain == 2
        assert "100.0%" in result[0]
        assert result[2].layout.xaxis.title.text == "hours ahead (H)"
        assert result[2].layout.yaxis.title.text == "Mean absolute error (°C)"
        assert result[3].layout.xaxis.title.text == "Hour of day by target valid time (UTC)"
        assert result[4].layout.xaxis.title.text == "Date by target valid time (UTC)"
        assert result[5].layout.xaxis.title.text == "Forecast origin (UTC)"
        assert result[5].layout.yaxis.title.text == "48-hour MAE (°C)"
        assert result[5].data[0].mode == "lines+markers"  # one real issue remains visible
        assert result[6].layout.yaxis.title.text == "Signed gain (°C)"
        assert result[7].layout.yaxis.title.text == "Absolute gain (°C)"
        filtered = handles["render"]("07630", "", "", "t", ["model"], "prospective-2026-11", "ALL")
        assert "No saved records" in filtered[0]
        invalid = handles["render"]("ALL", "invalid", "", "t", ["model"], "prospective-2026-11", "ALL")
        assert "Invalid date" in invalid[0]
    else:
        assert "No frozen predictions" in result[0]
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_large_configuration_selection_keeps_legends_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr("weather_lab.prospective.now_utc", lambda: "2026-11-06T01:00:00Z")
    archive = Archive(tmp_path)
    models = populate(archive, 12)
    result = view.read_view(archive, selected=models, cohort="prospective-2026-11")
    assert len(result[2].data) == 13
    for figure in result[2:8]:
        assert figure.layout.legend.orientation == "h"
        assert figure.layout.legend.maxheight == 110
        assert figure.layout.margin.b >= 150
    assert result[1].data[0].orientation == "h"
    assert result[1].layout.yaxis.automargin is True


class CalendarButtons(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.enabled = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "button" and "data-date" in attrs and "disabled" not in attrs:
            self.enabled.append(attrs["data-date"])


def test_calendars_offer_only_verified_dates_and_keep_holes_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr("weather_lab.prospective.now_utc", lambda: "2026-11-06T01:00:00Z")
    archive = Archive(tmp_path)
    populate(archive)
    index = view.date_inventory(archive)
    index = index[~index.date.eq("2026-11-04")]
    days = view.available_dates(index, "07149", "t", ["model"], "prospective-2026-11")
    assert days == ["2026-11-03", "2026-11-05"]
    last = date_picker.value(days, "2026-11-05")
    assert CalendarButtons(last["html"]).enabled == days
    assert last["date"] == "2026-11-05"
    assert 'data-date="2026-11-04"' in last["html"]  # visible, disabled day in calendar grid
    assert view.available_dates(index, "07630", "t", ["model"], "prospective-2026-11") == []
    assert view.available_dates(index, "07149", "pmer", ["model"], "prospective-2026-11") == []
    assert view.period_label("retrospective-2026-10") == "Historical replay · October 2026"


def test_corrupt_archive_does_not_prevent_other_tabs_from_building(tmp_path):
    archive = Archive(tmp_path)
    populate(archive)
    marker = next((tmp_path / "records/catalog").glob("*.json"))
    ident = json.loads(marker.read_text())["sha256"]
    (tmp_path / "objects" / (ident + ".json")).write_bytes(b"corrupt")
    with gr.Blocks() as demo:
        assert view.build("hourly-2026-v3", archive=archive) == {}
    assert "archive unavailable" in str(demo.get_config_file())
