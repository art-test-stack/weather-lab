"""Collected evaluation source for Benchmark; only Refresh reads and applies results."""
from __future__ import annotations

import os
from pathlib import Path

from weather_lab_app import zerogpu

import gradio as gr
import numpy as np
import pandas as pd

from weather_lab import prospective as engine
from weather_lab.prospective_store import Archive, PREFIX
from weather_lab.publish import DEFAULT_BUCKET
from weather_lab.saved_results import calendar, horizon, window_mae
from weather_lab_app import date_picker, plots
from weather_lab_app.labels import model_style

STATIONS = [("Both stations", "ALL"), ("Paris-Orly", "07149"), ("Toulouse-Blagnac", "07630")]
PLOTS = ["Configuration comparison", "Error versus forecast horizon", "Hour-of-day MAE", "Daily MAE",
         "Error per 48-hour forecast window", "Signed AROME gain", "Absolute AROME gain"]


def archive_location():
    if path := os.environ.get("WEATHER_LAB_PROSPECTIVE_ARCHIVE"):
        return Path(path)
    from weather_lab.bootstrap import load_bootstrap_config

    mounts = load_bootstrap_config()["bootstrap"]["snapshot"].get("mounts", {})
    mounted = Path(mounts.get(DEFAULT_BUCKET, "/nonexistent")) / PREFIX
    return mounted if mounted.is_dir() else Archive().root


def period_label(cohort):
    month = pd.Timestamp(cohort[-7:] + "-01").strftime("%B %Y")
    return ("Prospective forecasts" if cohort.startswith("prospective-") else "Historical replay") + " · " + month


def defaults(catalog):
    wanted = ["Timestral-209M · raw", "Timestral-209M · archived-arome",
              "Timestral-209M · residual prediction", "t0-gamma · residual prediction"]
    found = [key for label in wanted for key, record in catalog.items() if record["label"] == label]
    return found or list(catalog)[:4]


def date_inventory(archive):
    """Verified date choices only; this discovery does not compute or apply plotted metrics."""
    columns = ["cohort", "station", "variable", "model", "date", "cycle", "lead_h"]
    records = [row for rec in archive.records("predictions") for row in rec["value"]["rows"]]
    if not records:
        return pd.DataFrame(columns=columns)
    points = pd.DataFrame(records)
    points["valid_time"] = pd.to_datetime(points.valid_time, utc=True)
    points = points[np.isfinite(pd.to_numeric(points.point, errors="coerce"))]
    obs = engine.observation_frame(archive, pd.Timestamp(engine.now_utc()))
    truth = obs.melt(id_vars=["station", "time"], value_vars=["t", "pmer"], var_name="variable", value_name="observed")
    truth = truth[np.isfinite(pd.to_numeric(truth.observed, errors="coerce"))]
    points = points.merge(truth, left_on=["station", "valid_time", "variable"],
                          right_on=["station", "time", "variable"], validate="many_to_one")
    points["date"] = points.valid_time.dt.strftime("%Y-%m-%d")
    return points[columns].drop_duplicates()


def available_dates(inventory, station, variable, selected, cohort, cycle="ALL", lead=48):
    if inventory.empty or not selected:
        return []
    required = set(selected) | {"arome-raw"}
    rows = inventory[inventory.cohort.eq(cohort) & inventory.variable.eq(variable)
                     & inventory.model.isin(required) & inventory.lead_h.le(lead)]
    if station != "ALL":
        rows = rows[rows.station.eq(station)]
    if cycle != "ALL":
        rows = rows[rows.cycle.eq(int(cycle))]
    counts = rows.groupby("date").model.nunique()
    return sorted(counts[counts.eq(len(required))].index)


def fit_dates(days, first, last):
    first, last = date_picker.selected(first), date_picker.selected(last)
    if not days:
        return None, None
    first = first if first in days else days[0]
    last = last if last in days and last >= first else days[-1]
    return first, last


def read_view(archive, station="ALL", start="", end="", variable="t", selected=None,
              cohort="prospective-2026-11", cycle="ALL", lead=48, lead_mode="at", spread=False):
    catalog = {r["key"]: r["value"] for r in engine.catalog_records(archive)}
    selected = list(selected) if selected is not None else defaults(catalog)
    events = sorted([r["value"] for r in archive.records("status")], key=lambda r: r.get("collected_at", ""))
    provenance = pd.DataFrame([{"Configuration": r["value"]["model"], "Saved provenance": r["value"].get("provenance")}
                               for r in archive.records("predictions") if r["value"]["model"] in set(selected) | {"arome-raw"}])
    def empty(message, coverage=None):
        return (message, *[plots._empty(message) for _ in PLOTS], pd.DataFrame(),
                pd.DataFrame() if coverage is None else coverage, provenance, pd.DataFrame(events))
    if not archive.records("predictions"):
        return empty("No frozen predictions collected yet.", pd.DataFrame([r["value"] for r in archive.records("availability")]))
    if not selected:
        return empty("Select at least one saved configuration, then Refresh saved evaluation.")
    if not set(selected) <= catalog.keys() or station not in dict((value, label) for label, value in STATIONS):
        return empty("Select saved configurations and one of the available stations.")
    try:
        first, last = date_picker.selected(start), date_picker.selected(end)
        lo = pd.to_datetime(first, utc=True) if first else None
        hi = pd.to_datetime(last, utc=True) + pd.Timedelta(days=1) if last else None
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError("first date follows last date")
    except (ValueError, TypeError) as exc:
        return empty(f"Invalid date range: {exc}")
    _, coverage = engine.evaluate(archive, selected)
    if coverage.empty:
        return empty("No saved records match these filters.")
    keep = coverage.cohort.eq(cohort) & coverage.variable.eq(variable) & coverage.lead_h.le(lead)
    if station != "ALL":
        keep &= coverage.station.eq(station)
    if cycle != "ALL":
        keep &= coverage.cycle.eq(int(cycle))
    if lo is not None:
        keep &= coverage.valid_time.ge(lo)
    if hi is not None:
        keep &= coverage.valid_time.lt(hi)
    coverage = coverage[keep].copy()
    if coverage.empty:
        return empty("No saved records match these filters.")
    diagnostics = coverage.groupby(["model", "exclusion"], dropna=False).size().rename("pairs").reset_index()
    rows = coverage[coverage.paired].copy()
    note = (f"**{period_label(cohort)} · {'Temperature (°C)' if variable == 't' else 'Pressure (hPa)'}**\n\n"
            f"Applied valid dates: {first or 'first available'} – {last or 'last available'} UTC. "
            f"Leads 1–{lead} h; {'through-lead' if lead_mode == 'through' else 'at-lead'} horizon aggregation. "
            f"Common-support coverage: **{len(rows)}/{len(coverage)} configuration-pairs** ({len(rows) / len(coverage):.1%}). "
            "Every selected configuration and raw AROME use identical observed pairs. "
            "Complete-window diagnostics use each model's complete 48-hour cases and matched AROME reference. "
            "Overlapping forecast windows are dependent; a performance decrease alone is not evidence of leakage.")
    if rows.empty:
        return empty(note + "\n\nNo verified common support. Inspect coverage and exclusions.", diagnostics)
    rows["origin"] = rows.valid_time - pd.to_timedelta(rows.lead_h, unit="h")
    rows["err"] = rows.point - rows.observed
    rows["abs_err"], rows["sq_err"] = abs(rows.err), rows.err ** 2
    names = {key: record["label"] for key, record in catalog.items()} | {"arome-raw": "AROME · fixed issued run"}
    styles = {"arome-raw": model_style("arome-raw")}
    for number, key in enumerate(selected):
        cfg = catalog[key]["config"]
        family = "t0" if cfg.get("model", "").startswith("t0") else "timestral"
        color, _, width = model_style(f"{family}-dev saved {cfg.get('prediction_target', 'weather')} {cfg.get('conditioning', 'none')} saved")
        if family == "timestral":
            conditioned = cfg.get("conditioning", "none") != "none" or cfg.get("prediction_target") == "arome_error" or any(
                f["name"] == "resid" for f in cfg.get("features", []))
            color, _, width = model_style("timestral-209m-mv-cal" + ("-arome-archived" if conditioned else ""))
        styles[key] = (color, ["solid", "dash", "dot", "dashdot"][number % 4], width)
    summary = rows.groupby("model", as_index=False).agg(mae=("abs_err", "mean"), mse=("sq_err", "mean"), n=("abs_err", "size"))
    summary["rmse"] = np.sqrt(summary.pop("mse"))
    summary["gain"] = summary.loc[summary.model.eq("arome-raw"), "mae"].iloc[0] - summary.mae
    summary["absolute_gain"] = abs(summary.gain)
    comparison = plots.comparison_bars(summary, "mae", variable, "arome-raw", names=names, styles=styles)
    comparison.update_yaxes(automargin=True)
    by_lead = horizon(rows, cumulative=lead_mode == "through")
    by_lead = by_lead[by_lead.lead_h.le(lead)]
    curve = plots.metric_by_lead(by_lead, "mae", variable, list(rows.model.unique()),
                                cumulative=lead_mode == "through", spread=spread, names=names, styles=styles)
    curve.update_xaxes(range=[1, 48], tickvals=[1, 6, 12, 18, 24, 30, 36, 42, 48])
    dates = calendar(rows, "valid_time", start=lo, end=hi - pd.Timedelta(days=1) if hi is not None else None)
    hourly = plots.calendar_errors(dates["hour"], variable, "hour", "valid_time", names=names, styles=styles)
    daily = plots.calendar_errors(dates["day"], variable, "day", "valid_time", names=names, styles=styles)
    coverage["origin"] = coverage.valid_time - pd.to_timedelta(coverage.lead_h, unit="h")
    cases = coverage[["case_id", "station", "origin"]].drop_duplicates().assign(eligible=True, split=cohort)
    targets = coverage[["case_id", "variable", "lead_h", "valid_time", "observed"]].drop_duplicates()
    windows = []
    for model, frame in coverage.groupby("model", sort=False):
        table, _ = window_mae(frame, targets, cases, variable)
        windows.append(table.assign(model=model))
    windows = pd.concat(windows, ignore_index=True)
    window_plot = plots.forecast_windows(windows, variable, names, styles)
    signed, absolute = plots.correction_gains(windows, selected, variable, names, styles)
    signed.update_layout(title_text="Signed gain versus AROME per 48-hour forecast window")
    absolute.update_layout(title_text="Absolute gain versus AROME per 48-hour forecast window")
    figures = [comparison, curve, hourly, daily, window_plot, signed, absolute]
    for figure in figures[1:]:
        figure.update_layout(legend={"maxheight": 110}, margin={"b": 150})
    summary["model"] = summary.model.map(names)
    return note, *figures, summary, diagnostics, provenance, pd.DataFrame(events)


def build(suite: str, *, archive=None):
    archive = archive or Archive(archive_location())
    try:
        catalog = {r["key"]: r["value"] for r in engine.catalog_records(archive)}
        inventory = date_inventory(archive)
        periods = sorted({r["value"]["cohort"] for r in archive.records("predictions")}, reverse=True)
    except (OSError, ValueError, KeyError) as exc:
        gr.Markdown(f"Collected archive unavailable: {exc}. Restore verified saved artifacts and reload.")
        return {}
    periods = periods or ["prospective-2026-11"]
    initial_selected = defaults(catalog)
    days = available_dates(inventory, "ALL", "t", initial_selected, periods[0])
    first, last = fit_dates(days, None, None)
    available = gr.State(inventory)
    applied = gr.State(None)
    gr.Markdown("## Collected evaluation\nChoose settings, then **Refresh saved evaluation**. "
                "Historical replay uses archived data; prospective forecasts were frozen before verification. "
                "Each evaluation period is compared separately.")
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=290):
            station = gr.Dropdown(STATIONS, value="ALL", label="Station")
            start = date_picker.build(days, first, "First valid date (UTC)")
            end = date_picker.build(days, last, "Last valid date (UTC, inclusive)")
            variable = gr.Radio([("Temperature · °C", "t"), ("Pressure · hPa", "pmer")], value="t", label="Prediction variable")
            period = gr.Dropdown([(period_label(p), p) for p in periods], value=periods[0], label="Evaluation period")
            selected = gr.Dropdown([(r["label"], key) for key, r in catalog.items()], value=initial_selected,
                                   multiselect=True, label="Saved configurations to compare")
            cycle = gr.Dropdown(["ALL", *[str(h) for h in range(24)]], value="ALL", label="Issuance hour (UTC)")
            lead = gr.Slider(1, 48, value=48, step=1, label="Evaluate through lead (hours)")
            lead_mode = gr.Radio([("At lead", "at"), ("Through lead", "through")], value="at", label="Horizon aggregation")
            spread = gr.Checkbox(False, label="±1 SD bars (At lead only)")
            refresh = gr.Button("Refresh saved evaluation", variant="primary")
            pending = gr.Markdown("Choose settings and refresh to display saved results.")
        with gr.Column(scale=3, min_width=500):
            status = gr.Markdown()
            figures = []
            for name in PLOTS:
                gr.Markdown("### " + name)
                figures.append(gr.Plot(value=plots._empty("Click Refresh saved evaluation"), show_label=False))
            with gr.Accordion("Metrics, coverage and saved provenance", open=False):
                table = gr.Dataframe(interactive=False, label="Paired MAE, RMSE and AROME gains")
                coverage = gr.Dataframe(interactive=False, label="Coverage and exclusions")
                provenance = gr.Dataframe(interactive=False, label="Saved provenance")
                events = gr.Dataframe(interactive=False, label="Collection status and attempts")
    controls = [station, start, end, variable, selected, period, cycle, lead, lead_mode, spread]
    outputs = [status, *figures, table, coverage, provenance, events]
    def normalized(values):
        values = list(values)
        values[1], values[2] = date_picker.selected(values[1]), date_picker.selected(values[2])
        return values
    def feedback(*values):
        *values, current = values
        return ("Showing the applied saved evaluation." if normalized(values) == current else
                "Changes pending — click Refresh saved evaluation. Displayed results are unchanged.")
    def edit(*values):
        *values, index, current = values
        station, first, last, variable, selected, period, cycle, lead, *_ = normalized(values)
        days = available_dates(index, station, variable, selected, period, cycle, lead)
        first, last = fit_dates(days, first, last)
        values[1], values[2] = first, last
        end_days = [day for day in days if first is None or day >= first]
        return date_picker.value(days, first), date_picker.value(end_days, last), feedback(*values, current)
    def choose_date(which):
        def choose(evt: gr.EventData, *values):
            *values, index, current = values
            days = available_dates(index, values[0], values[3], values[4], values[5], values[6], values[7])
            if which == 2 and date_picker.selected(values[1]):
                days = [day for day in days if day >= date_picker.selected(values[1])]
            chosen, month = evt._data.get("date"), evt._data.get("month")
            if month:
                return (date_picker.value(days, values[which], month, True), gr.skip(), gr.skip()) if which == 1 else (
                    gr.skip(), date_picker.value(days, values[which], month, True), gr.skip())
            if chosen not in days:
                return gr.skip(), gr.skip(), "Choose a date with saved verified data."
            values[which] = chosen
            if which == 1 and date_picker.selected(values[2]) and chosen > date_picker.selected(values[2]):
                values[2] = chosen
            return edit(*values, index, current)
        return choose
    def reload(*values):
        current = {r["key"]: r["value"] for r in engine.catalog_records(archive)}
        index = date_inventory(archive)
        values = normalized(values)
        periods = sorted({r["value"]["cohort"] for r in archive.records("predictions")}, reverse=True) or [values[5]]
        if values[5] not in periods:
            values[5] = periods[0]
        days = available_dates(index, values[0], values[3], values[4], values[5], values[6], values[7])
        values[1], values[2] = fit_dates(days, values[1], values[2])
        result = read_view(archive, *values)
        return (gr.update(choices=[(r["label"], key) for key, r in current.items()]),
                gr.update(choices=[(period_label(p), p) for p in periods], value=values[5]),
                date_picker.value(days, values[1]),
                date_picker.value([day for day in days if values[1] is None or day >= values[1]], values[2]), index, values, *result)
    refresh.click(reload, controls, [selected, period, start, end, available, applied, *outputs], trigger_mode="once", concurrency_limit=1).then(
        feedback, controls + [applied], pending, queue=False, show_progress="hidden")
    for control in [station, variable, selected, period, cycle, lead, lead_mode, spread]:
        control.input(edit, controls + [available, applied], [start, end, pending], queue=False, show_progress="hidden")
    for which, control in [(1, start), (2, end)]:
        control.input(choose_date(which), controls + [available, applied], [start, end, pending], queue=False, show_progress="hidden")
    return {"controls": controls, "outputs": outputs, "refresh": refresh, "render": lambda *v: read_view(archive, *v),
            "edit": edit, "available": available, "applied": applied, "pending": pending}
