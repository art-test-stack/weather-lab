"""Saved-only prospective evaluation. No collection, inference or archive writes."""
from __future__ import annotations

import os
from pathlib import Path

from weather_lab_app import zerogpu

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from weather_lab.common import STATION_NAMES
from weather_lab.prospective import evaluate
from weather_lab.prospective_store import Archive, PREFIX
from weather_lab.publish import DEFAULT_BUCKET
from weather_lab_app.labels import model_style


def archive_location():
    if path := os.environ.get("WEATHER_LAB_PROSPECTIVE_ARCHIVE"):
        return Path(path)
    from weather_lab.bootstrap import load_bootstrap_config

    mounts = load_bootstrap_config()["bootstrap"]["snapshot"].get("mounts", {})
    mounted = Path(mounts.get(DEFAULT_BUCKET, "/nonexistent")) / PREFIX
    return mounted if mounted.is_dir() else Archive().root


def summarize(metrics, group):
    """Weight by pair counts, and take |G| after aggregation."""
    values = metrics.assign(mae_sum=metrics.mae * metrics.n, mse_sum=metrics.rmse ** 2 * metrics.n,
                            raw_sum=metrics.mae_arome * metrics.n)
    out = values.groupby(group, as_index=False).agg(n=("n", "sum"), mae_sum=("mae_sum", "sum"),
                                                   mse_sum=("mse_sum", "sum"), raw_sum=("raw_sum", "sum"))
    out["mae"], out["rmse"] = out.mae_sum / out.n, np.sqrt(out.mse_sum / out.n)
    out["gain"] = out.raw_sum / out.n - out.mae
    out["absolute_gain"] = abs(out.gain)
    return out.drop(columns=["mae_sum", "mse_sum", "raw_sum"])


def read_view(archive, station="ALL", start="", end="", variable="t", selected=None,
              cohort="prospective-2026-11", cycle="ALL", horizon=48):
    catalog = {r["key"]: r["value"] for r in archive.records("catalog")}
    status = []
    for kind in ("forecasts", "observations"):
        records = archive.records(kind)
        latest = max((r["value"]["collected_at"] for r in records), default="not collected")
        status.append(f"Latest successful {kind} ingestion: **{latest}**")
    events = sorted([r["value"] for r in archive.records("status")], key=lambda r: r["collected_at"])
    status.extend(f"{task}: **{rec['status']}** ({rec['collected_at']})"
                  for task, rec in {r["task"]: r for r in events}.items())
    note = "\n\n".join(status)
    provenance = pd.DataFrame([{"configuration": key, "label": rec["label"], "checkpoint": rec["checkpoint"],
        "source_sha256": rec.get("source_run_sha256"), "configuration_and_features": rec["config"]}
        for key, rec in catalog.items()])
    def empty(message, coverage=None):
        return (note + "\n\n" + message, *[go.Figure() for _ in range(4)], pd.DataFrame(),
                pd.DataFrame() if coverage is None else coverage, provenance, pd.DataFrame(events))
    if not archive.records("predictions"):
        missing = pd.DataFrame([r["value"] for r in archive.records("availability")])
        return empty("No frozen predictions collected yet. Nothing is substituted.", missing)
    selected = list(selected) if selected is not None else list(catalog)
    if not selected:
        return empty("Select at least one saved configuration.")
    metrics, coverage = evaluate(archive, selected)
    try:
        lo = pd.to_datetime(start, utc=True) if start else None
        hi = pd.to_datetime(end, utc=True) + pd.Timedelta(days=1) if end else None
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError("start date must not follow end date")
    except (ValueError, TypeError) as exc:
        return empty(f"Invalid date range: {exc}")
    def filtered(frame):
        if frame.empty:
            return frame
        keep = frame.variable.eq(variable) & frame.cohort.eq(cohort) & frame.lead_h.le(horizon)
        if station != "ALL":
            keep &= frame.station.eq(station)
        if cycle != "ALL":
            keep &= frame.cycle.eq(int(cycle))
        axis = pd.to_datetime(frame["date"] if "date" in frame else frame.valid_time, utc=True)
        if lo is not None:
            keep &= axis.ge(lo)
        if hi is not None:
            keep &= axis.lt(hi)
        return frame[keep].copy()
    metrics, coverage = filtered(metrics), filtered(coverage)
    if coverage.empty:
        return empty("No saved records match these filters.")
    diagnostics = coverage.groupby(["model", "exclusion"], dropna=False).size().rename("pairs").reset_index()
    n, total = int(coverage.paired.sum()), len(coverage)
    note += (f"\n\nCommon-support coverage: **{n}/{total} configuration-pairs** ({n / total:.1%}). "
             "Every selected configuration and raw AROME use identical station/valid-time/lead pairs. "
             "Overlapping windows are dependent. Cohorts and physical variables stay separate. "
             "A performance decrease alone is not evidence of leakage.")
    provenance = pd.DataFrame([{"model": r["value"]["model"], "case_id": r["value"].get("case_id"),
        "provenance": r["value"].get("provenance")} for r in archive.records("predictions")
        if r["value"]["model"] in set(selected) | {"arome-raw"}])
    if metrics.empty:
        return empty("No verified common support. Inspect missing inputs, observations and exclusions below.", diagnostics)
    labels = {key: rec["label"] for key, rec in catalog.items()} | {"arome-raw": "Raw AROME"}
    colors = {"Raw AROME": model_style("arome-raw")[0]}
    for rec in catalog.values():
        c = rec["config"]
        family = "t0" if c.get("model", "").startswith("t0") else "timestral"
        style_id = f"{family}-dev {c.get('model', 'saved')} {c.get('prediction_target', 'weather')} {c.get('conditioning', 'none')} saved"
        colors[rec["label"]] = model_style(style_id)[0]
    metrics["model"] = metrics.model.map(lambda m: labels.get(m, m))
    daily, lead, comparison = [summarize(metrics, ["model", dim] if dim else ["model"])
                               for dim in ("date", "lead_h", None)]
    day_plot = px.line(daily.melt(id_vars=["model", "date"], value_vars=["mae", "rmse"],
                                 var_name="metric", value_name="error"), x="date", y="error", color="model",
                       line_dash="metric", color_discrete_map=colors, title="Daily MAE / RMSE · valid date (UTC)")
    lead_plot = px.line(lead, x="lead_h", y="mae", color="model", color_discrete_map=colors,
                        title="MAE by hours after decision")
    gain_plot = px.line(lead.melt(id_vars=["model", "lead_h"], value_vars=["gain", "absolute_gain"],
                                  var_name="gain_type", value_name="gain_value"),
                        x="lead_h", y="gain_value", color="model", line_dash="gain_type",
                        color_discrete_map=colors, title="Signed G = MAE(AROME) − MAE(model), and |G|")
    gain_plot.add_hline(y=0, line_dash="dot")
    comparison_plot = px.bar(comparison, x="model", y="mae", color="model", color_discrete_map=colors,
                             title="Configurations on identical support")
    for plot in (day_plot, lead_plot, gain_plot, comparison_plot):
        plot.update_yaxes(title="°C" if variable == "t" else "hPa")
    return (note, day_plot, lead_plot, gain_plot, comparison_plot, comparison, diagnostics,
            provenance, pd.DataFrame(events))


def build(suite: str, *, archive=None):
    archive = archive or Archive(archive_location())
    catalog = {r["key"]: r["value"] for r in archive.records("catalog")}
    cohorts = sorted({r["value"]["cohort"] for r in archive.records("predictions")}, reverse=True)
    cohorts = cohorts or ["prospective-2026-11", "retrospective-2026-10"]
    initial = read_view(archive, selected=list(catalog)[:4], cohort=cohorts[0])
    gr.Markdown("## Prospective Evaluation\nSaved artifacts only. Refresh never collects data or runs models. "
                "Historical replay remains separate from genuine prospective predictions.")
    with gr.Row():
        station = gr.Dropdown([("All stations", "ALL"), *[(name, sid) for sid, name in STATION_NAMES.items()]],
                              value="ALL", label="Station")
        start = gr.Textbox(label="First valid date (UTC)", placeholder="2026-11-01")
        end = gr.Textbox(label="Last valid date (UTC, inclusive)", placeholder="2026-11-30")
        variable = gr.Dropdown([("Temperature (°C)", "t"), ("Pressure (hPa)", "pmer")], value="t", label="Variable")
    with gr.Row():
        cohort = gr.Dropdown(cohorts, value=cohorts[0], label="Evaluation cohort")
        cycle = gr.Dropdown(["ALL", *[str(h) for h in range(24)]], value="ALL", label="Issuance hour (UTC)")
        horizon = gr.Slider(1, 48, value=48, step=1, label="Evaluate through lead (hours)")
    selected = gr.CheckboxGroup([(r["label"], k) for k, r in catalog.items()], value=list(catalog)[:4],
                               label="Saved Benchmark configurations (shared support)")
    refresh = gr.Button("Refresh saved evaluation")
    status = gr.Markdown(initial[0])
    plots = [gr.Plot(value=initial[i + 1], label=name) for i, name in enumerate(
        ("Daily performance", "Lead curve", "Signed / absolute gain", "Configurations"))]
    table = gr.Dataframe(value=initial[5], interactive=False, label="Paired metrics")
    coverage = gr.Dataframe(value=initial[6], interactive=False, label="Coverage and exclusions")
    with gr.Accordion("Saved provenance and collection status", open=False):
        provenance = gr.Dataframe(value=initial[7], interactive=False, label="Provenance")
        events = gr.Dataframe(value=initial[8], interactive=False, label="Collection attempts")
    controls = [station, start, end, variable, selected, cohort, cycle, horizon]
    outputs = [status, *plots, table, coverage, provenance, events]
    def render(*values):
        return read_view(archive, *values)
    def reload(*values):
        current = {r["key"]: r["value"] for r in archive.records("catalog")}
        available = sorted({r["value"]["cohort"] for r in archive.records("predictions")}, reverse=True) or cohorts
        values = list(values)
        values[4] = [m for m in values[4] if m in current] or list(current)[:4]
        values[5] = values[5] if values[5] in available else available[0]
        return (gr.update(choices=[(r["label"], k) for k, r in current.items()], value=values[4]),
                gr.update(choices=available, value=values[5]), *render(*values))
    refresh.click(reload, controls, [selected, cohort, *outputs])
    for control in controls:
        control.input(render, controls, outputs)
    return {"controls": controls, "outputs": outputs, "refresh": refresh, "render": render}
