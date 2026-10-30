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
from weather_lab.saved_results import matrix_overview
from weather_lab_app import plots, results, benchmark as panel
from weather_lab_app.labels import model_style


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


@pytest.fixture
def matrix(catalog):
    directory = Path(next(iter(catalog))).parents[1] / "comparisons"
    directory.mkdir()
    support = pd.DataFrame({"case_id": ["a"] * 3, "variable": ["t", "t", "pmer"], "lead_h": [1, 2, 1]})
    support.to_parquet(directory / "common_support.parquet")
    digest = sha256_file(directory / "common_support.parquet")
    configurations = {r["summary"]["configuration_id"]: {"run_id": Path(p).name,
        "run_json_sha256": r["summary"]["run_json_sha256"]} for p, r in catalog.items()}
    models = [*configurations, "arome-raw", "persistence"]
    scores = pd.DataFrame([{"model": model, "variable": variable, "split": "test", "station": "ALL", "season": "ALL",
        "lead_h": np.nan, "mae": (i + 1) * scale, "n": pairs, "support_sha256": digest}
        for i, model in enumerate(models) for variable, scale, pairs in (("t", 1., 2), ("pmer", 10., 1))])
    scores.to_csv(directory / "shared_arome_metrics.csv", index=False)
    record = {"suite": "hourly-2026-v3", "matrix": "synthetic", "configurations": configurations,
        "support_sha256": digest, "scores_sha256": sha256_file(directory / "shared_arome_metrics.csv")}
    (directory / "provenance.json").write_text(json.dumps(record))
    yield catalog, directory
    results._matrix_files.cache_clear()


def test_discovery_filters_and_saved_only_routing(catalog, monkeypatch):
    from weather_lab import models

    def no_model(*args, **kwargs):
        pytest.fail("Saved Benchmark performed a model operation")
    monkeypatch.setattr(models, "prefetch", no_model)
    monkeypatch.setattr(models.MODELS, "get", no_model)
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
    assert reused["window_errors"] is applied["window_errors"]
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


@pytest.mark.parametrize("selected_complete", [False, True])
def test_window_plot_includes_selected_models_on_their_own_support(catalog, monkeypatch, selected_complete):
    truth = pd.DataFrame({"case_id": "a", "variable": "t", "lead_h": range(1, 49), "observed": 0.})
    origin = pd.Timestamp("2026-01-01T23:00Z")
    truth["valid_time"] = origin + pd.to_timedelta(truth.lead_h, unit="h")
    expected_mae = {path: float(number + 3) for number, path in enumerate(catalog)}
    expected_mae["arome-raw"] = 2.
    for number, (path, record) in enumerate(catalog.items()):
        path = Path(path)
        run = json.loads((path / "run.json").read_text())
        primary = truth.drop(columns="observed").assign(station="07149", origin=origin,
                    model=run["systems"][-1]["id"], point=float(number + 3) if selected_complete else np.nan)
        points = [primary]
        if not panel.weather_only(record["config"]):
            points.append(primary.assign(model="arome-raw", point=2.))
        pd.concat(points).to_parquet(path / "predictions.parquet")
        truth.to_parquet(path / "targets.parquet")
        for name in ("predictions.parquet", "targets.parquet"):
            run["files_sha256"][name] = sha256_file(path / name)
        (path / "run.json").write_text(json.dumps(run))
        record["summary"]["run_json_sha256"] = sha256_file(path / "run.json")
    monkeypatch.setattr(results.Run, "arome_extract", lambda *_: pytest.fail("Use saved points, never a later issue"))
    d = panel.defaults(catalog, "test")
    weather = {p: r for p, r in catalog.items() if panel.weather_only(r["config"])}
    for selection in (d, {**d, "cohort": "weather", "selected": list(weather)}):
        applied, message = panel.read_saved({"draft": selection, "previous": None}, catalog)
        assert applied is not None, message
        assert applied["support"]["observed_pairs"] == (48 if selected_complete else 0)
        note, fig, table = panel.render_windows(applied)
        assert "1 complete windows" in note and "48 observed pairs" in note
        assert "own observed support" in note and "not a ranking" in note
        expected_models = {*selection["selected"], "arome-raw"} if selected_complete else {"arome-raw"}
        assert {trace.legendgroup for trace in fig.data} == expected_models
        assert sum(len(trace.x) for trace in fig.data) == len(expected_models)
        assert len(table) == len(selection["selected"]) + 1
        for trace in fig.data:
            name = {**applied["names"], "arome-raw": "AROME · fixed issued run"}[trace.legendgroup]
            expected = table[table.Configuration.eq(name)]
            assert expected["Complete"].all() and trace.name == name
            assert list(trace.customdata[0]) == [1, 48]
            assert list(trace.y) == expected["48-hour MAE"].tolist() == [expected_mae[trace.legendgroup]]
            assert trace.line.color == applied["styles"].get(trace.legendgroup, model_style("arome-raw"))[0]
            assert trace.mode == "lines" and not trace.connectgaps and trace.error_y.array is None
        ar = table[table.Configuration.str.startswith("AROME")]
        assert ar["48-hour MAE"].tolist() == [2.] and ar["Complete"].tolist() == [True]
        if not selected_complete:
            missing = table[~table.Configuration.str.startswith("AROME")]
            assert missing["48-hour MAE"].isna().all() and not missing["Complete"].any()
            assert missing["Available pairs / 48"].eq(0).all()
    applied, _ = panel.read_saved({"draft": selection, "previous": None}, weather)
    note, fig, table = panel.render_windows(applied)
    assert "No saved AROME reference" in note and len(table) == 1
    assert len(fig.data) == (1 if selected_complete else 0)


def test_window_lines_average_available_locations_at_each_origin():
    table = pd.DataFrame({"model": "arome-raw", "station": ["07149"] * 3 + ["07630"],
        "case_id": ["c", "a", "b", "d"], "origin": pd.to_datetime([
            "2026-01-01T02:00Z", "2026-01-01T00:00Z", "2026-01-01T01:00Z", "2026-01-01T00:00Z"]),
        "mae": [3., 1., np.nan, 8.], "n": [48, 48, 47, 48], "complete": [True, True, False, True]})
    fig = plots.forecast_windows(table, "t", styles={"arome-raw": ("#123456", "dash", 2)})
    assert len(fig.data) == 1
    first = fig.data[0]
    assert pd.to_datetime(first.x, utc=True).tolist() == sorted(table.origin.iloc[[0, 1]])
    assert list(first.y) == [4.5, 3.]
    assert list(map(list, first.customdata)) == [[2, 96], [1, 48]]
    assert all(t.mode == "lines" and not t.connectgaps and t.line.dash == "dash" for t in fig.data)


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
        "### All saved matrix configurations · temperature and pressure",
        "### 2026 temperature leaderboard", "### Error versus forecast horizon", "### Hour-of-day MAE",
        "### Error over the calendar", "### Error per 48-hour forecast window", "### Inspect one saved forecast"]
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
    _, _, message, button, _ = edit(*[d[k] for k in panel.KEYS], None, {"busy": True})
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


def test_frozen_matrix_reads_both_variables_on_separate_common_support_and_caches(matrix, monkeypatch):
    catalog, directory = matrix
    source = results.load_run(Path(next(iter(catalog))))
    monkeypatch.setattr(results.Run, "point_predictions", lambda *_: pytest.fail("The matrix reads frozen aggregates only"))
    reads, read = [], pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda *a, **kw: (reads.append(a[0]), read(*a, **kw))[1])
    first = panel.overview("hourly-2026-v3", catalog, source)
    second = panel.overview("hourly-2026-v3", catalog, source)
    assert len(reads) == 1
    pd.testing.assert_frame_equal(first["table"], second["table"])
    table = first["table"]
    assert len(table) == 4 and table["Temperature MAE (°C)"].tolist() == [1., 2., 3., 4.]
    assert table["Pressure MAE (hPa)"].tolist() == [10., 20., 30., 40.]
    assert table["Temperature pairs"].tolist() == [2] * 4 and table["Pressure pairs"].tolist() == [1] * 4
    assert "Temperature: **2 pairs · 1 windows**" in first["note"] and "Pressure: **1 pairs · 1 windows**" in first["note"]
    saved = results.saved_matrix("hourly-2026-v3", catalog)
    wrong = saved["scores"].copy()
    wrong.loc[0, "n"] = 3
    with pytest.raises(ValueError, match="pair counts differ"):
        matrix_overview(wrong, saved["support"], source.cases(), wrong.model.unique(), "test")
    results._matrix_files.cache_clear()
    (directory / "shared_arome_metrics.csv").write_text("changed frozen scores")
    with pytest.raises(ValueError, match="checksum differs"):
        results.saved_matrix("hourly-2026-v3", catalog)


def test_matrix_row_selection_routes_by_label_and_only_changes_pending_controls(matrix, monkeypatch):
    catalog, _ = matrix
    source = results.load_run(Path(next(iter(catalog))))
    with gr.Blocks() as demo:
        h = panel.build("hourly-2026-v3", source, demo)
    assert h["matrix_data"] is not None
    monkeypatch.setattr(panel, "read_saved", lambda *_: pytest.fail("Table clicks cannot read saved predictions"))
    monkeypatch.setattr(results, "verified_run", lambda *_: pytest.fail("Table clicks cannot load runs"))
    chooser = next(f for f in demo.fns.values() if (h["matrix"]._id, "select") in f.targets)
    assert chooser.name == "choose" and chooser.show_progress == "hidden" and not chooser.queue
    protected = {h["applied"]._id} | {o._id for o in h["render"]}
    assert not protected & {o._id for o in chooser.outputs}
    d = panel.defaults(catalog, "test")
    rows = panel.overview_selection(h["matrix_data"], d).data.values.tolist()
    path = h["matrix_data"]["routes"][rows[0][1]]
    event = gr.SelectData(None, {"index": [99, 0], "value": rows[0][1], "row_value": rows[0], "selected": True})
    outputs = chooser.fn(event, *[d[k] for k in panel.KEYS], None, {"busy": True})
    assert outputs[0]["value"] == [p for p in d["selected"] if p != path]
    assert "pending" in outputs[2] and not outputs[3]["interactive"]
    updated = {**d, "selected": outputs[0]["value"]}
    added = chooser.fn(event, *[updated[k] for k in panel.KEYS], None, {"busy": False})
    assert set(added[0]["value"]) == set(d["selected"])
    conditioned = next(row for row in rows if "archived-arome" in row[1])
    blocked = gr.SelectData(None, {"index": [0, 1], "value": conditioned[1], "row_value": conditioned})
    out = chooser.fn(blocked, *[{**d, "cohort": "weather"}[k] for k in panel.KEYS], None, {"busy": False})
    assert "excluded by the current cohort" in out[2]
    reference = gr.SelectData(None, {"index": [2, 1], "value": rows[2][1], "row_value": rows[2]})
    out = chooser.fn(reference, *[d[k] for k in panel.KEYS], None, {"busy": False})
    assert "reference" in out[2] and h["applied"].value is None


def test_frozen_matrix_is_discovered_after_explicit_bucket_publication(matrix):
    import shutil

    catalog, directory = matrix
    mounted = development.location().parent / "benchmark-matrices/synthetic-published"
    mounted.parent.mkdir(parents=True)
    shutil.copytree(directory, mounted)
    directory.rename(directory.with_name("unmounted-comparison"))
    saved = results.saved_matrix("hourly-2026-v3", catalog)
    assert saved["provenance"]["matrix"] == "synthetic"
    assert set(saved["paths"].values()) == set(catalog)


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
    assert len(output) == 19
    assert "0 matched observed pairs" in output[8]
    assert output[10].empty
    assert output[15]["Observed pairs"].tolist() == [1, 1, 1]  # leaderboard stays temperature
    assert "0 complete windows" in output[16] and output[18]["Available pairs / 48"].tolist() == [0, 0, 0]


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
