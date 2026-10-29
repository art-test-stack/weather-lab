"""Saved-only Benchmark controls; edits never read predictions or apply a result."""
from __future__ import annotations

import copy
import json
import threading
from pathlib import Path

import gradio as gr
import pandas as pd

from weather_lab.common import sha256_file
from weather_lab.saved_results import annual_temperature, calendar, horizon, matched, one_forecast
from weather_lab_app import plots, results
from weather_lab_app.labels import model_style
from weather_lab_app.results import station_label

KEYS = ("cohort", "model", "target", "conditioning", "features", "rag", "selected",
        "variable", "station", "split", "lead_mode", "spread", "inspect_station", "origin",
        "calendar_clock", "calendar_lead", "calendar_spread")
_LOCK = threading.Lock()


def draft(*values):
    return copy.deepcopy(dict(zip(KEYS, values)))


def weather_only(c):
    return (c["conditioning"] == "none" and c["prediction_target"] == "weather"
            and not any(f["name"] == "resid" for f in c["features"]))


def matching(catalog, d):
    """Filter recorded configurations, never manufacture combinations from controls."""
    out = []
    for path, record in catalog.items():
        c = record["config"]
        feature_keys = [f["name"] for f in c["features"]] + (["calendar"] if c["calendar"] else [])
        if d["cohort"] == "weather" and not weather_only(c):
            continue
        if any(d[key] != "ALL" and d[key] != value for key, value in (
            ("model", c["model"]), ("target", c["prediction_target"]), ("conditioning", c["conditioning"]),
            ("rag", c.get("retrieval_method", "none")))):
            continue
        if d["features"] == "none" and feature_keys:
            continue
        if d["features"] not in ("ALL", "none") and d["features"] not in feature_keys:
            continue
        out.append((record["label"], path))
    return out


def defaults(catalog, split):
    wanted = ["Timestral-209M/raw", "Timestral-209M/archived-arome",
              "Timestral-209M/residual-prediction", "t0-gamma/residual-prediction"]
    selected = [next(p for p, r in catalog.items() if r["summary"].get("configuration_id") == name)
                for name in wanted if any(r["summary"].get("configuration_id") == name for r in catalog.values())]
    return dict(cohort="arome", model="ALL", target="ALL", conditioning="ALL", features="ALL", rag="ALL",
                selected=selected or list(catalog)[:4], variable="t", station="ALL", split=split,
                lead_mode="at", spread=True, inspect_station="07149", origin="",
                calendar_clock="valid_time", calendar_lead="all", calendar_spread=False)


def feedback(d, applied, guard):
    pending = applied is None or d != applied["draft"]
    text = "Changes pending — click Run saved comparison. Displayed results are unchanged." if pending else "Showing the applied saved comparison."
    if not d["selected"]:
        text += " Select at least one saved configuration."
    return text, gr.update(interactive=bool(d["selected"]) and not guard["busy"],
                           value="Reading saved results…" if guard["busy"] else "Run saved comparison")


def begin(*values):
    *vals, applied, guard = values
    with _LOCK:
        if guard["busy"]:
            raise gr.Error("A saved comparison is already being read in this session")
        guard["busy"] = True
    return {"draft": draft(*vals), "previous": applied}, gr.update(interactive=False), "Reading saved results…"


def data_key(d):
    return tuple(d["selected"]), d["cohort"], d["variable"], d["station"], d["split"]


def _source(run):
    hashes = run.run["source_sha256"]
    return run.run["source_suite"], hashes["manifest.json"], hashes["targets.parquet"]


def _read_horizon(request, catalog):
    """Explicit saved-only worker. Presentation-only Runs reuse the applied row losses."""
    d, previous = request["draft"], request["previous"]
    try:
        allowed = {p for _, p in matching(catalog, d)}
        if not d["selected"] or not set(d["selected"]) <= allowed:
            raise ValueError("Select configurations present in this saved cohort and filter selection")
        if previous and data_key(d) == data_key(previous["draft"]):
            return {**previous, "draft": d}, "Applied presentation settings to the same saved predictions."
        runs, frames, names, styles, provenance = [], {}, {}, {}, []
        for number, path in enumerate(d["selected"]):
            record = catalog[path]
            run = results.verified_run(Path(path))
            if run.run["forecast_config"] != record["config"]:
                raise ValueError("Saved discovery configuration differs from run provenance")
            if run.run["source_suite"] != record["summary"]["source_suite"]:
                raise ValueError("Saved discovery suite differs from run provenance")
            digest = record["summary"].get("run_json_sha256")
            if digest and sha256_file(Path(path) / "run.json") != digest:
                raise ValueError("Saved discovery run checksum differs")
            if runs and _source(run) != _source(runs[0]):
                raise ValueError("Saved configurations belong to different suites or source case/target sets")
            runs.append(run)
            frames[path] = run.point_predictions(run.models[-1], d["variable"])
            names[path] = record["label"]
            base = model_style(run.models[-1])
            if record["config"]["model"].startswith("Timestral-"):
                base = model_style("timestral-209m-mv-cal" + ("-arome-archived" if not weather_only(record["config"]) else ""))
            styles[path] = (base[0], ["solid", "dash", "dot", "dashdot"][number % 4], base[2])
            counts = sorted({info["n_cases"] for info in run.run.get("case_info", {}).values() if "n_cases" in info})
            provenance.append({"Configuration": record["label"], "Run": run.run["run_id"],
                               "Checkpoint revision": run.run["checkpoint"]["revision"],
                               "Windows per model call (recorded)": str(counts) if counts else "not recorded",
                               "Saved configuration": json.dumps(record["config"], sort_keys=True),
                               "Run SHA-256": sha256_file(Path(path) / "run.json")})
        source, reference = runs[0], None
        if d["cohort"] == "arome":
            reference = next((r for r in runs if "arome-raw" in r.models), None)
            if reference is None:
                refpath = next((p for p, r in catalog.items() if not weather_only(r["config"])), None)
                if refpath is None:
                    raise ValueError("Saved AROME reference is unavailable; select broader weather-only coverage")
                reference = results.verified_run(Path(refpath))
            if _source(reference) != _source(source):
                raise ValueError("AROME reference belongs to a different source case/target set")
            frames["arome-raw"] = reference.point_predictions("arome-raw", d["variable"])
            names["arome-raw"] = "AROME · fixed issued run"
            styles["arome-raw"] = model_style("arome-raw")
        targets, cases = source.targets(), source.cases()
        rows, support = matched(frames, targets, cases, d["variable"], d["station"], d["split"])
        temperature = frames if d["variable"] == "t" else {
            path: run.point_predictions(run.models[-1], "t") for path, run in zip(d["selected"], runs)}
        if d["variable"] != "t" and reference is not None:
            temperature["arome-raw"] = reference.point_predictions("arome-raw", "t")
        leaderboard, annual_support = annual_temperature(temperature, targets, cases, d["station"], d["split"])
        return {"draft": d, "rows": rows, "support": support, "names": names, "styles": styles,
                "leaderboard": leaderboard, "annual_support": annual_support,
                "frames": frames, "targets": targets, "cases": cases,
                "reference_path": reference.path if reference else None,
                "provenance": pd.DataFrame(provenance)}, "Read saved predictions and targets; no inference, downloads or frozen-score writes."
    except Exception as exc:
        return None, f"Saved comparison unavailable: {type(exc).__name__}: {exc}. Displayed results are unchanged."


def read_saved(request, catalog):
    result, message = _read_horizon(request, catalog)
    if result is None or not result["draft"]["origin"]:
        return result, message
    d, previous = result["draft"], request["previous"]
    if (previous and data_key(d) == data_key(previous["draft"])
            and d["origin"] == previous["draft"]["origin"]
            and d["inspect_station"] == previous["draft"]["inspect_station"]):
        return result, message
    try:
        case = result["cases"].set_index("case_id").loc[d["origin"]]
        if case.station != d["inspect_station"]:
            raise ValueError("Forecast origin does not belong to the selected inspector station")
        refpath = result["reference_path"] or next((p for p, r in catalog.items() if not weather_only(r["config"])), None)
        reference = None
        if refpath:
            ref = results.verified_run(Path(refpath))
            if _source(ref) != _source(results.load_run(Path(d["selected"][0]))):
                raise ValueError("Inspector AROME reference belongs to different source cases/targets")
            reference = ref.issued_reference(d["origin"], d["variable"])
        inspection = one_forecast(result["frames"], result["targets"], result["cases"],
                                  d["origin"], d["variable"], reference)
        availability = []
        for path in d["selected"]:
            saved = results.load_run(Path(path)).case_availability()
            saved = saved[saved.case_id.eq(d["origin"])]
            availability.append({"Configuration": result["names"][path],
                                 "Status": saved.status.iloc[0] if len(saved) else "not recorded",
                                 "Reason": saved.reason.iloc[0] if len(saved) else "No saved case availability record"})
        result["inspection"] = inspection
        result["case_availability"] = pd.DataFrame(availability)
        result["inspection_error"] = ""
    except (ValueError, KeyError) as exc:
        result["inspection"] = None
        result["inspection_error"] = f"Forecast inspector unavailable: {exc}"
    return result, message


def apply(result, *values):
    *vals, previous, message = values
    if result is None:
        return previous, message, False
    if result["draft"] != draft(*vals):
        return previous, "Settings changed while reading. Click Run saved comparison to apply the current selection.", False
    return result, message, True


def support_text(applied):
    s, d = applied["support"], applied["draft"]
    cohort = "AROME-supported coverage (AROME required on every matched pair)" if d["cohort"] == "arome" else "Broader weather-only coverage (separate comparison)"
    return (f"**{cohort} · {'Temperature (°C)' if d['variable'] == 't' else 'Pressure (hPa)'}**\n\n"
            f"Eligible origins: {s['candidate_origin_start']} – {s['candidate_origin_end']}. "
            f"Matched origins: {s['origin_start']} – {s['origin_end']}; target valid times: {s['valid_start']} – {s['valid_end']} (UTC). "
            f"Stations: {', '.join(station_label(st) for st in s['stations']) or 'none'}. "
            f"**{s['windows']:,} matched windows · {s['observed_pairs']:,} observed pairs** over leads 1–48 h. "
            f"Scheduled windows: {s['scheduled_windows']:,}; eligible: {s['eligible_windows']:,}. "
            f"Excluded pairs: {s['excluded_pairs']:,} / {s['candidate_pairs']:,}, including {s['missing_target_pairs']:,} missing targets. "
            "Per-system exclusions below overlap and must not be added together.\n\n"
            "Each matched (window, lead) pair has equal weight. At lead h uses h only; Through lead h uses leads 1…h. "
            "Spread is ±1 population SD of absolute errors across windows at each lead (ddof=0), shown only for At lead. "
            "It is not a confidence interval. Overlapping hourly windows share observations and are dependent. "
            "Development replay, inspected configurations; no held-out validation or calibrated uncertainty claim.")


def render(applied, changed=True):
    if not changed or applied is None:
        return tuple(gr.skip() for _ in range(16))
    d = applied["draft"]
    table = horizon(applied["rows"], cumulative=d["lead_mode"] == "through")
    fig = plots.metric_by_lead(table, "mae", d["variable"], list(applied["names"]),
                               cumulative=d["lead_mode"] == "through", spread=d["spread"],
                               names=applied["names"], styles=applied["styles"])
    fig.update_xaxes(range=[1, 48], tickvals=[1, 6, 12, 18, 24, 30, 36, 42, 48])
    exclusions = pd.DataFrame(applied["support"]["exclusions"])
    exclusions["model"] = exclusions.model.map(applied["names"])
    table["model"] = table.model.map(applied["names"])
    return (support_text(applied), fig, table.drop(columns=["ci95_low", "ci95_high", "n_blocks"]), exclusions,
            applied["provenance"], *render_inspection(applied), *render_calendar(applied), *render_leaderboard(applied))


def render_leaderboard(applied):
    table, s = applied["leaderboard"].copy(), applied["annual_support"]
    cohort = "AROME-supported" if applied["draft"]["cohort"] == "arome" else "broader weather-only"
    note = (f"**Temperature MAE · 2026 forecast origins (UTC) · {cohort}.** "
            "Selected configurations, equal weight per matched (window, lead) pair over leads 1–48 h. "
            f"Matched origins: {s['origin_start']} – {s['origin_end']}; valid times: {s['valid_start']} – {s['valid_end']}. "
            f"Stations: {', '.join(station_label(st) for st in s['stations']) or 'none'}. "
            f"**{s['windows']:,} windows · {s['observed_pairs']:,} identical observed pairs** per system. "
            f"{s['excluded_pairs']:,} excluded / {s['candidate_pairs']:,} candidate pairs "
            f"({s['missing_target_pairs']:,} missing targets). "
            "Available saved coverage only; this is not a complete-year or held-out validation score.")
    fig = plots.comparison_bars(table, "mae", "t", "arome-raw", names=applied["names"], styles=applied["styles"])
    table["model"] = table.model.map(applied["names"])
    table = table.rename(columns={"model": "Configuration", "mae": "Temperature MAE (°C)",
                                  "n": "Observed pairs", "n_cases": "Windows"})
    return note, fig, table.round(4)


def render_inspection(applied):
    inspection = applied.get("inspection")
    if inspection is None:
        return applied.get("inspection_error", "Select a station and origin, then Run saved comparison."), plots._empty("No saved forecast selected"), pd.DataFrame()
    curves, obs, case, errors, support = inspection
    names = {**applied["names"], "arome-raw": "AROME · fixed issued run"}
    fig = plots.forecast_figure(curves, obs, case, applied["draft"]["variable"], list(names), None,
                                context_hours=0, error_mode="signed", error_rows=errors,
                                names=names, styles=applied["styles"])
    fig.update_xaxes(range=[case.origin, case.origin + pd.Timedelta(hours=48)])
    ar = curves[curves.model.eq("arome-raw") & curves.point.notna()]
    issue = (f"AROME initialization: {ar.arome_run.iloc[0]}; assumed available: {ar.arome_available_at.iloc[0]}. "
             f"Available AROME issue-relative leads: {ar.arome_lead_h.min():g}–{ar.arome_lead_h.max():g} h. "
             if len(ar) and "arome_run" in ar else "AROME issued-run forecast unavailable at this origin. ")
    note = (f"**{station_label(case.station)} · origin {case.origin} UTC · 48-hour saved forecast.** " + issue
            + f"Signed errors (forecast − observation) use {support['observed_pairs']} identical observed pairs. "
            "Upper curves retain available forecasts and gaps; no confidence or forecast bands are drawn. "
            "AROME is a single saved issue, never updated after the origin. Archive release/revision timestamps are unavailable; availability assumes a 3-hour delay.")
    if applied["draft"]["cohort"] == "weather":
        note += " AROME is context only in this broader weather-only inspector; its errors are not included."
    table = curves.merge(obs.rename(columns={"time": "valid_time"}), on="valid_time", how="left")
    table = table.merge(errors[["model", "lead_h", "err"]], on=["model", "lead_h"], how="left")
    table["model"] = table.model.map(names)
    table = table.merge(applied["case_availability"], left_on="model", right_on="Configuration", how="left")
    columns = ["model", "lead_h", "valid_time", "point", "observed", "err", "Status", "Reason"]
    columns += [c for c in ("arome_run", "arome_available_at", "arome_lead_h") if c in table]
    return note, fig, table[columns]


def origin_choices(cases, station):
    return [(f"{c.origin:%Y-%m-%d %H:%M} UTC", c.case_id)
            for c in cases[cases.station.eq(station)].sort_values("origin").itertuples()]


def render_calendar(applied):
    d, support = applied["draft"], applied["support"]
    lead = None if d["calendar_lead"] == "all" else int(d["calendar_lead"])
    start, end = support["candidate_origin_start"], support["candidate_origin_end"]
    if d["calendar_clock"] == "valid_time":
        start += pd.Timedelta(hours=lead or 1)
        end += pd.Timedelta(hours=lead or 48)
    tables = calendar(applied["rows"], d["calendar_clock"], lead, start, end)
    audit = pd.DataFrame(support["lead_support"])
    if lead is not None:
        audit = audit[audit.lead_h.eq(lead)]
    expected, observed, pairs = (int(audit[c].sum()) for c in ("candidate_pairs", "observed_target_pairs", "matched_pairs"))
    sel = applied["rows"] if lead is None else applied["rows"][applied["rows"].lead_h.eq(lead)]
    clock = "target valid time" if d["calendar_clock"] == "valid_time" else "forecast origin"
    note = (f"**Calendar grouped by {clock} in UTC · {'all leads 1–48 h' if lead is None else f'lead {lead} h only'}.** "
            f"{sel.case_id.nunique():,} windows · {pairs:,} matched observed pairs; {expected - pairs:,} excluded pairs, "
            f"including {expected - observed:,} missing targets and {observed - pairs:,} observed pairs without every selected prediction. "
            f"Matched {clock}: {sel[d['calendar_clock']].min()} – {sel[d['calendar_clock']].max()}. "
            "Stations and coverage cohort are those shown above.\n\n"
            "Daily MAE, monthly MAE and the hour-of-day profile each give every matched (window, lead) pair equal weight within the bucket. "
            "Monthly MAE is computed from pairs, not from daily means. With all 48 leads, windows with more observed pairs receive more weight. "
            "Spread in each bucket is ±1 population SD of those same absolute errors (ddof=0); monthly SD appears in the table. "
            "These are descriptive spreads, not confidence intervals; overlapping windows are dependent. Empty buckets remain gaps.")
    daily = plots.calendar_errors(tables["day"], d["variable"], "day", d["calendar_clock"],
                                   d["calendar_spread"], applied["names"], applied["styles"])
    hourly = plots.calendar_errors(tables["hour"], d["variable"], "hour", d["calendar_clock"],
                                    True, applied["names"], applied["styles"])
    for table in tables.values():
        table["model"] = table.model.map(applied["names"])
    detail = pd.concat([table.assign(period=period) for period, table in tables.items()], ignore_index=True)
    monthly = tables["month"].rename(columns={"bucket": "Month (UTC)", "mae": "MAE", "spread": "Population SD of absolute errors",
                                               "n": "Observed pairs", "n_cases": "Windows"})
    return note, daily, monthly.round(4), hourly, detail


def build(suite, run, demo):
    catalog = results.saved_configurations(suite)
    gr.Markdown("## Compare saved configurations\nSaved predictions and targets only. Change settings, then click **Run saved comparison**.")
    if not catalog:
        gr.Markdown("No saved hourly 48-hour configurations found locally or on the mounted bucket. "
                    "After explicitly publishing or installing results, restart the app to discover them. This panel never downloads results.")
        return None
    d0 = defaults(catalog, run.splits[-1])
    cases = run.cases()
    d0["inspect_station"] = run.stations()[0]["id"]
    origins = origin_choices(cases, d0["inspect_station"])
    d0["origin"] = origins[max(0, len(origins) - 49)][1] if origins else ""
    availability = run.case_availability()
    if {"arome_48h_available", "verified_targets"} <= set(availability):
        available = set(availability.loc[availability.arome_48h_available & availability.verified_targets.ge(96), "case_id"])
        supported = [value for _, value in origins if value in available]
        if supported:
            d0["origin"] = supported[-1]
    applied, guard = gr.State(None), gr.State({"busy": True})
    request, pending, changed = gr.State(None), gr.State(None), gr.State(True)
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=290, elem_id="benchmark-config"):
            cohort = gr.Radio([("AROME-supported", "arome"), ("Broader weather-only", "weather")], value=d0["cohort"], label="Coverage cohort")
            variable = gr.Radio([("Temperature · °C", "t"), ("Pressure · hPa", "pmer")], value="t", label="Prediction variable")
            with gr.Accordion("Filter saved configurations", open=False):
                model = gr.Dropdown(["ALL"] + sorted({r["config"]["model"] for r in catalog.values()}), value="ALL", label="Model")
                target = gr.Dropdown([("All", "ALL"), ("Direct weather", "weather"), ("AROME residual", "arome_error")], value="ALL", label="Prediction target")
                conditioning = gr.Dropdown(["ALL"] + sorted({r["config"]["conditioning"] for r in catalog.values()}), value="ALL", label="AROME conditioning")
                features = gr.Dropdown(["ALL", "none", "calendar"] + sorted({f["name"] for r in catalog.values() for f in r["config"]["features"]}), value="ALL", label="Feature / calendar inputs")
                rag = gr.Dropdown(["ALL"] + sorted({r["config"].get("retrieval_method", "none") for r in catalog.values()}), value="ALL", label="RAG")
            selected = gr.Dropdown(matching(catalog, d0), value=d0["selected"], multiselect=True, label="Saved configurations to compare")
            station = gr.Dropdown([("All stations", "ALL")] + [(station_label(s["id"]), s["id"]) for s in run.stations()], value="ALL", label="Station")
            split = gr.Dropdown(run.splits, value=d0["split"], label="Saved split", visible=len(run.splits) > 1)
            lead_mode = gr.Radio([("At lead", "at"), ("Through lead", "through")], value="at", label="Horizon aggregation")
            spread = gr.Checkbox(True, label="±1 SD bars (At lead only)")
            inspect_station = gr.Dropdown([(station_label(s["id"]), s["id"]) for s in run.stations()], value=d0["inspect_station"], label="Inspect station")
            origin = gr.Dropdown(origins, value=d0["origin"], label="Forecast origin to inspect (UTC)")
            calendar_clock = gr.Radio([("Target valid time", "valid_time"), ("Forecast origin", "origin")], value="valid_time", label="Calendar grouping (UTC)")
            calendar_lead = gr.Dropdown([("All 48 leads", "all")] + [(f"Lead {h} h", str(h)) for h in range(1, 49)], value="all", label="Calendar lead selection")
            calendar_spread = gr.Checkbox(False, label="Daily ±1 SD bars")
            status = gr.Markdown()
            button = gr.Button("Run saved comparison", variant="primary", interactive=False)
        with gr.Column(scale=3, min_width=500, elem_id="benchmark-results"):
            message = gr.Markdown()
            gr.Markdown("### 2026 temperature leaderboard")
            leaderboard_note, leaderboard_plot = gr.Markdown(), gr.Plot(show_label=False)
            leaderboard_table = gr.Dataframe(interactive=False, label="Selected configurations · matched temperature pairs")
            gr.Markdown("### Error versus forecast horizon")
            scope, figure = gr.Markdown(), gr.Plot(show_label=False)
            with gr.Accordion("Matched support, saved configuration and provenance", open=False):
                table = gr.Dataframe(interactive=False, label="Horizon MAE and descriptive spread")
                exclusions = gr.Dataframe(interactive=False, label="Exclusions (overlapping counts)")
                provenance = gr.Dataframe(interactive=False, label="Original saved provenance")
            gr.Markdown("### Hour-of-day MAE")
            calendar_note, hourly_plot = gr.Markdown(), gr.Plot(show_label=False)
            gr.Markdown("### Error over the calendar")
            daily_plot = gr.Plot(show_label=False)
            monthly_table = gr.Dataframe(interactive=False, label="Monthly MAE and population SD (same units as the selected variable)")
            with gr.Accordion("Calendar pairs, window counts and spread", open=False):
                calendar_table = gr.Dataframe(interactive=False)
            gr.Markdown("### Inspect one saved forecast")
            inspection_note, inspection_plot = gr.Markdown(), gr.Plot(show_label=False)
            with gr.Accordion("Saved forecast values, signed errors and availability", open=False):
                inspection_table = gr.Dataframe(interactive=False)
    controls = [cohort, model, target, conditioning, features, rag, selected, variable, station, split, lead_mode, spread,
                inspect_station, origin, calendar_clock, calendar_lead, calendar_spread]
    outputs = [scope, figure, table, exclusions, provenance, inspection_note, inspection_plot, inspection_table,
               calendar_note, daily_plot, monthly_table, hourly_plot, calendar_table,
               leaderboard_note, leaderboard_plot, leaderboard_table]

    def edit(*values):
        *vals, current, busy = values
        d = draft(*vals)
        choices = matching(catalog, d)
        d["selected"] = [p for p in d["selected"] if p in {p for _, p in choices}]
        origins = origin_choices(cases, d["inspect_station"])
        if d["origin"] not in {v for _, v in origins}:
            d["origin"] = origins[max(0, len(origins) - 49)][1] if origins else ""
        return gr.update(choices=choices, value=d["selected"]), gr.update(choices=origins, value=d["origin"]), *feedback(d, current, busy)

    def worker(captured):
        return read_saved(captured, catalog)

    def finish(*values):
        *vals, current, busy = values
        with _LOCK:
            busy["busy"] = False
        return feedback(draft(*vals), current, busy)

    for control in controls:
        control.input(edit, controls + [applied, guard], [selected, origin, status, button], queue=False, show_progress="hidden")
    (button.click(begin, controls + [applied, guard], [request, button, message], trigger_mode="once", queue=False)
     .success(worker, request, [pending, message], concurrency_limit=1)
     .then(apply, [pending] + controls + [applied, message], [applied, message, changed])
     .then(render, [applied, changed], outputs)
     .then(finish, controls + [applied, guard], [status, button], queue=False))
    (demo.load(lambda: worker({"draft": d0, "previous": None}), None, [pending, message])
     .then(apply, [pending] + controls + [applied, message], [applied, message, changed])
     .then(render, [applied, changed], outputs)
     .then(finish, controls + [applied, guard], [status, button], queue=False))
    return {"draft": controls, "applied": applied, "run": button, "render": outputs, "submission": guard,
            "captured_request": request, "catalog": catalog}
