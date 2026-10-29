"""Saved configuration discovery, routing and the explicit read/apply event graph."""
import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import pytest

from weather_lab import development
from weather_lab.common import sha256_file
from weather_lab.forecast_config import ForecastConfig
from weather_lab_app import results, saved_benchmark as panel


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(development, "ROOT", tmp_path)
    monkeypatch.setattr(development, "location", lambda: tmp_path / "mounted/development-runs")
    cases = [{"case_id": "a", "station": "07149", "origin": "2026-01-01T23:00:00Z", "eligible": True, "split": "test"}]
    truth = pd.DataFrame({"case_id": ["a", "a"], "variable": ["t", "t"], "lead_h": [1, 2],
                          "valid_time": pd.to_datetime(["2026-01-02T00:00Z", "2026-01-02T01:00Z"]), "observed": [0., 0.]})
    for number, (setup, cond) in enumerate((("raw", "none"), ("archived-arome", "archived"))):
        path = tmp_path / "runs/matrix/runs" / str(number)
        path.mkdir(parents=True)
        cfg = ForecastConfig.make("Timestral-209M", cond, calendar=False, step_hours=1, targets=("t", "pmer")).as_dict()
        truth.to_parquet(path / "targets.parquet")
        primary = truth.drop(columns="observed").assign(station="07149", origin=pd.Timestamp(cases[0]["origin"]),
                    model=f"timestral-dev Timestral-209M weather {cond} {number}", point=[1., 3.])
        frames = [primary]
        if cond != "none":
            frames.append(primary.assign(model="arome-raw", point=[2., float("nan")],
                arome_run="2026-01-01T20:00:00Z", arome_available_at="2026-01-01T23:00:00Z"))
            pd.DataFrame({"run": ["2026-01-02T00:00:00Z"], "value": [999.]}).to_parquet(path / "arome_station_extract.parquet")
        pd.concat(frames).to_parquet(path / "predictions.parquet")
        (path / "manifest.json").write_text(json.dumps({"cases": cases, "dataset": {"stations": [{"id": "07149"}]},
                "protocol": {"forecast": {"step_hours": 1, "horizon_steps": 48}}}))
        pd.DataFrame({"split": ["test"]}).to_csv(path / "metrics_point.csv", index=False)
        rec = {"run_id": str(number), "suite": "development", "source_suite": "hourly-2026-v3", "created_at": "2026-01-01",
               "forecast_config": cfg, "systems": [{"id": f.model.iloc[0]} for f in frames[::-1]],
               "source_sha256": {"manifest.json": "source-manifest", "targets.parquet": "source-targets"},
               "checkpoint": {"revision": "a" * 40}, "case_info": {"a": {"n_cases": 1 if number == 0 else 8}},
               "files_sha256": {p.name: sha256_file(p) for p in path.iterdir()}}
        (path / "run.json").write_text(json.dumps(rec))
        summary = {"configuration_id": f"Timestral-209M/{setup}", "source_suite": rec["source_suite"],
                   "created_at": rec["created_at"], "forecast_config": cfg, "run_json_sha256": sha256_file(path / "run.json")}
        (path / "matrix_setup.json").write_text(json.dumps(summary))
    cached_verifier = results.verified_run
    yield results.saved_configurations("hourly-2026-v3")
    cached_verifier.cache_clear()
    results.load_run.cache_clear()
    results._parquet.cache_clear()


def test_discovery_filters_and_saved_only_routing(catalog, monkeypatch):
    assert len(catalog) == 2
    d = panel.defaults(catalog, "test")
    paths = d["selected"]
    assert len(paths) == 2
    assert panel.matching(catalog, {**d, "cohort": "weather"}) == [(catalog[paths[0]]["label"], paths[0])]
    assert panel.matching(catalog, {**d, "rag": "shape"}) == []
    applied, message = panel.read_saved({"draft": d, "previous": None}, catalog)
    assert applied is not None, message
    assert applied["support"]["observed_pairs"] == 1
    assert set(applied["rows"].model) == {*paths, "arome-raw"}
    assert applied["provenance"]["Windows per model call (recorded)"].tolist() == ["[1]", "[8]"]
    figure = panel.render(applied)[1]
    assert len(figure.data) == 3 and "AROME" in figure.data[-1].name
    table = panel.render(applied)[2]
    for trace in figure.data:
        expected = table[table.model.eq(trace.name)].spread
        np.testing.assert_allclose(trace.error_y.array, expected, equal_nan=True)
        assert trace.error_y.symmetric and "population SD" in trace.hovertemplate
    assert applied["leaderboard"].mae.tolist() == [1., 1., 2.]
    assert applied["leaderboard"].n.tolist() == [1, 1, 1]
    monkeypatch.setattr(results, "verified_run", lambda *_: pytest.fail("Presentation reused points must not reload runs"))
    changed = {**d, "lead_mode": "through", "spread": True, "calendar_clock": "origin", "calendar_lead": "1",
               "calendar_spread": True}
    reused, message = panel.read_saved({"draft": changed, "previous": applied}, catalog)
    assert reused["rows"] is applied["rows"] and "presentation" in message
    assert len(panel.render(reused)[1].data) == 3  # no cumulative spread band
    note, daily, monthly, hourly, detail = panel.render_calendar(reused)
    assert "forecast origin in UTC" in note and "lead 1 h only" in note
    for trace in hourly.data:
        expected = detail[(detail.model.eq(trace.name)) & detail.period.eq("hour")].spread
        np.testing.assert_allclose(trace.error_y.array, expected, equal_nan=True)
    assert all(trace.error_y.array is None for trace in panel.render(reused)[1].data)
    assert monthly["Observed pairs"].tolist() == [1, 1, 1]
    assert "not confidence intervals" in note and len(hourly.data) == 3
    invalid, message = panel.read_saved({"draft": {**d, "selected": ["not-saved"]}, "previous": None}, catalog)
    assert invalid is None and "Select configurations" in message


def test_mounted_discovery_needs_no_download_and_checksum_failure_is_explicit(catalog, tmp_path):
    import shutil

    original = Path(next(p for p, r in catalog.items() if panel.weather_only(r["config"])))
    mounted = tmp_path / "mounted/development-runs/copy"
    shutil.copytree(original, mounted)
    found = results.saved_configurations("hourly-2026-v3")
    assert str(mounted) in found
    (mounted / "predictions.parquet").write_bytes(b"changed saved file")
    d = {**panel.defaults(found, "test"), "selected": [str(mounted)], "cohort": "weather"}
    applied, message = panel.read_saved({"draft": d, "previous": None}, found)
    assert applied is None and "verification" in message


def test_saved_event_graph_edits_never_read_or_apply(catalog, monkeypatch):
    monkeypatch.setattr(results, "saved_configurations", lambda _: catalog)
    source = results.load_run(Path(next(iter(catalog))))
    with gr.Blocks() as demo:
        handles = panel.build("hourly-2026-v3", source, demo)
    controls = dict(zip(panel.KEYS, handles["draft"]))
    assert controls["calendar_clock"].value == "valid_time" and controls["calendar_lead"].value == "all"
    assert handles["submission"].value["busy"] and not handles["run"].interactive
    assert controls["spread"].value and not controls["calendar_spread"].value
    headings = [c["props"].get("value") for c in demo.config["components"] if c["type"] == "markdown"]
    assert [v for v in headings if v and v.startswith("### ")] == [
        "### 2026 temperature leaderboard", "### Error versus forecast horizon", "### Hour-of-day MAE",
        "### Error over the calendar", "### Inspect one saved forecast"]
    columns = {c["props"].get("elem_id"): c["id"] for c in demo.config["components"] if c["type"] == "column"}
    def subtree(node):
        return {node["id"]} | set().union(*(subtree(c) for c in node.get("children", [])))
    def find(node, id):
        if node["id"] == id:
            return node
        return next((found for c in node.get("children", []) if (found := find(c, id))), None)
    sidebar = subtree(find(demo.config["layout"], columns["benchmark-config"]))
    results_ids = subtree(find(demo.config["layout"], columns["benchmark-results"]))
    assert {c._id for c in handles["draft"]} <= sidebar
    assert {c._id for c in handles["render"]} <= results_ids
    protected = {handles["applied"]._id} | {o._id for o in handles["render"]}
    def root(fn):
        while fn.trigger_after is not None:
            fn = demo.fns[fn.trigger_after]
        return fn
    for fn in demo.fns.values():
        roots = root(fn).targets
        if protected & {o._id for o in fn.outputs}:
            assert roots == [(handles["run"]._id, "click")] or any(t[1] == "load" for t in roots)
        if any(t[0] in {c._id for c in handles["draft"]} and t[1] == "input" for t in roots):
            assert fn.name == "edit" and fn.show_progress == "hidden"
            assert not protected & {o._id for o in fn.outputs}
    monkeypatch.setattr(panel, "read_saved", lambda *_: pytest.fail("Editing must never read predictions"))
    edit = next(f.fn for f in demo.fns.values() if f.name == "edit")
    d = panel.defaults(catalog, "test")
    _, _, message, button = edit(*[d[k] for k in panel.KEYS], None, {"busy": True})
    assert "pending" in message and not button["interactive"]
    guard = {"busy": False}
    vals = [d[k] for k in panel.KEYS]
    request, _, _ = panel.begin(*vals, None, guard)
    assert request["draft"] == d and guard["busy"]
    with pytest.raises(gr.Error, match="already"):
        panel.begin(*vals, None, guard)
    newer = {**d, "variable": "pmer"}
    previous = {"draft": d}
    applied, _, changed = panel.apply({"draft": d}, *[newer[k] for k in panel.KEYS], previous, "")
    assert applied is previous and not changed


def test_inspector_reads_saved_issue_never_later_archive_forecasts(catalog, monkeypatch):
    monkeypatch.setattr(results.Run, "arome_extract", lambda *_: pytest.fail("Never choose a later AROME issue"))
    d = {**panel.defaults(catalog, "test"), "origin": "a"}
    applied, message = panel.read_saved({"draft": d, "previous": None}, catalog)
    assert applied and applied["inspection"] is not None, message
    note, fig, table = panel.render_inspection(applied)
    assert "20:00:00" in note and "4–4 h" in note
    assert set(table.loc[table.point.notna(), "point"]) == {1., 2., 3.}
    error_lines = [t for t in fig.data if t.yaxis == "y2"]
    assert len(error_lines) == 3 and all(np.isfinite(t.y).sum() == 1 for t in error_lines)
    ar = next(t for t in fig.data if t.name == "AROME · fixed issued run" and t.yaxis == "y")
    assert ar.customdata[0][1] == 4
    assert "initialization" in ar.hovertemplate


def test_missing_targets_stay_empty_across_all_views(catalog):
    d = {**panel.defaults(catalog, "test"), "variable": "pmer", "origin": "a"}
    applied, message = panel.read_saved({"draft": d, "previous": None}, catalog)
    assert applied is not None, message
    assert applied["support"]["observed_pairs"] == 0
    output = panel.render(applied)
    assert len(output) == 16
    assert "0 matched observed pairs" in output[8]
    assert output[10].empty
    assert output[15]["Observed pairs"].tolist() == [1, 1, 1]  # leaderboard stays temperature


def test_saved_comparison_rejects_different_source_cases_and_stale_discovery(catalog, monkeypatch):
    import copy

    d = panel.defaults(catalog, "test")
    runs = {path: copy.deepcopy(results.verified_run(Path(path))) for path in d["selected"]}
    runs[d["selected"][1]].run["source_sha256"]["manifest.json"] = "another-case-set"
    monkeypatch.setattr(results, "verified_run", lambda path: runs[str(path)])
    result, message = panel.read_saved({"draft": d, "previous": None}, catalog)
    assert result is None and "different suites or source case/target sets" in message
    stale = copy.deepcopy(catalog)
    stale[d["selected"][0]]["summary"]["source_suite"] = "another-suite"
    result, message = panel.read_saved({"draft": d, "previous": None}, stale)
    assert result is None and "discovery suite differs" in message
