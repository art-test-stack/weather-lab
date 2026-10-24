"""Plotly figures for the explorer (pure functions of saved results)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from weather_lab.forecast import qcol
from weather_lab_app.labels import OBS_COLOR, model_name, model_style, unit, variable_name
from weather_lab_app.results import station_label

INK, INK2, GRID, SURFACE, FORECAST_BG = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb", "#eef4fb"
FONT = "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif"


def _layout(fig: go.Figure, title: str = "", height: int = 380, legend_top: bool = True, **kw) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 15, "color": INK}} if title else None, height=height,
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font={"family": FONT, "color": INK2, "size": 12},
        margin={"l": 52, "r": 12, "t": 44 if title else 16, "b": 40}, hovermode="x unified", autosize=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "font": {"size": 12}} if legend_top
        else {}, **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font={"size": 13, "color": INK2})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, height=260)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------------------------- forecast

def forecast_figure(rows: pd.DataFrame, obs: pd.DataFrame, case: pd.Series, variable: str, systems: list[str],
                    band_model: str | None, history_hours: int = 48, height: int = 340) -> go.Figure:
    """The forecast window first: short history, shaded forecast period, readable names, ids in hover."""
    origin = pd.Timestamp(case["origin"])
    u = unit(variable)
    end = pd.Timestamp(rows["valid_time"].max()) if len(rows) else origin
    start = origin - pd.Timedelta(hours=history_hours)
    fig = go.Figure()
    fig.add_vrect(x0=origin, x1=end + pd.Timedelta(hours=1.5), fillcolor=FORECAST_BG, line_width=0, layer="below")
    fig.add_annotation(x=origin, y=1, yref="paper", xanchor="left", yanchor="top", showarrow=False,
                       text="  forecast →", font={"size": 12, "color": "#184f95"})
    fig.add_annotation(x=origin, y=1, yref="paper", xanchor="right", yanchor="top", showarrow=False,
                       text="← observed  ", font={"size": 12, "color": INK2})
    sel = rows[rows["model"].isin(systems)]
    b = sel[sel["model"] == band_model] if band_model else sel.iloc[:0]
    if len(b) and b["output_type"].iloc[0] == "quantiles":       # deterministic systems never get bands
        color = model_style(band_model)[0]
        for lo, hi, alpha, label in ((0.05, 0.95, 0.12, "90%"), (0.1, 0.9, 0.24, "80%")):
            fig.add_trace(go.Scatter(
                x=list(b["valid_time"]) + list(b["valid_time"][::-1]),
                y=list(b[qcol(hi)]) + list(b[qcol(lo)][::-1]), fill="toself", line={"width": 0},
                fillcolor=_rgba(color, alpha), name=f"80% / 90% range · {model_name(band_model)}",
                legendgroup="band", showlegend=label == "80%", hoverinfo="skip"))
    hist = obs[(obs["time"] >= start) & (obs["time"] <= origin)]
    fut = obs[(obs["time"] > origin) & (obs["time"] <= end)]
    fig.add_trace(go.Scatter(x=hist["time"], y=hist["observed"], name="Observed", mode="lines+markers",
                             line={"color": OBS_COLOR, "width": 2}, marker={"size": 4}, legendgroup="obs",
                             hovertemplate="%{y:.1f} " + u + "<extra>Observed</extra>"))
    fig.add_trace(go.Scatter(x=fut["time"], y=fut["observed"], name="Observed (verification)", mode="markers",
                             legendgroup="obs", showlegend=False,
                             marker={"color": OBS_COLOR, "size": 9, "symbol": "circle-open", "line": {"width": 2}},
                             hovertemplate="%{y:.1f} " + u + "<extra>Observed (not an input)</extra>"))
    for model in [m for m in systems if m in set(sel["model"])]:
        g = sel[sel["model"] == model]
        color, dash, width = model_style(model)
        name = model_name(model)
        fig.add_trace(go.Scatter(x=g["valid_time"], y=g["point"], name=name, mode="lines+markers",
                                 line={"color": color, "dash": dash, "width": width}, marker={"size": 5},
                                 hovertemplate="%{y:.1f} " + u + f"<extra>{name}<br><sub>{model}</sub></extra>"))
    fig.add_vline(x=origin, line={"color": INK2, "width": 1, "dash": "dash"})
    fig.update_yaxes(title=variable_name(variable))
    fig.update_xaxes(range=[start, end + pd.Timedelta(hours=1.5)], tickformat="%a %d %b\n%H:%M")
    return _layout(fig, height=height)


# --------------------------------------------------------------------------- benchmark

METRICS = {
    "mae": ("Mean absolute error", True, "point"),
    "rmse": ("Root-mean-square error", True, "point"),
    "pinball": ("Pinball loss (probabilistic)", True, "prob"),
    "coverage80": ("80% range coverage", False, "prob"),
}


def comparison_bars(table: pd.DataFrame, metric: str, variable: str, reference: str | None,
                    hatched: frozenset[str] = frozenset()) -> go.Figure:
    """Horizontal bars sorted best first; reference system highlighted; values labelled with units."""
    if table.empty:
        return _empty("No scores for this selection")
    label, lower_better, _ = METRICS[metric]
    u = "" if metric == "coverage80" else f" {unit(variable)}"
    t = table.dropna(subset=[metric]).copy()
    if metric == "coverage80":
        t["key"] = (t[metric] - 0.8).abs()                      # closest to the nominal 80% first
    else:
        t["key"] = t[metric] if lower_better else -t[metric]
    t = t.sort_values("key", ascending=False)                   # plotly draws bottom-up: best ends on top
    names = [model_name(m) for m in t["model"]]
    colors = [model_style(m)[0] for m in t["model"]]
    text = [f"{v:.0%}" if metric == "coverage80" else f"{v:.2f}{u}" for v in t[metric]]
    fig = go.Figure(go.Bar(
        x=t[metric], y=names, orientation="h", marker={"color": colors, "pattern": {
            "shape": ["/" if m in hatched else "" for m in t["model"]], "fgcolor": SURFACE, "size": 7}, "line": {
            "color": [INK if m == reference else SURFACE for m in t["model"]],
            "width": [2.5 if m == reference else 0 for m in t["model"]]}},
        text=text, textposition="outside", cliponaxis=False,
        customdata=list(zip(t["model"], t["n"])),
        hovertemplate="%{y}: %{text}<br>n = %{customdata[1]} forecast–observation pairs"
                      "<br><sub>%{customdata[0]}</sub><extra></extra>"))
    if metric == "coverage80":
        fig.add_vline(x=0.8, line={"color": INK2, "dash": "dot"}, annotation_text="nominal 80%",
                      annotation_position="top")
        fig.update_xaxes(tickformat=".0%", range=[0, 1.05])
    else:
        fig.update_xaxes(title=f"{label} ({unit(variable)}) · lower is better", rangemode="tozero")
    fig.update_layout(hovermode="closest", bargap=0.35)
    return _layout(fig, height=max(240, 46 * len(t) + 70), legend_top=False)


def metric_by_lead(table: pd.DataFrame, metric: str, variable: str, models: list[str],
                   primary: list[str] | None = None) -> go.Figure:
    """Score by lead; systems outside `primary` start hidden (one click on the legend shows them)."""
    df = table[table["lead_h"].notna() & table["model"].isin(models)]
    if df.empty:
        return _empty("No lead-time scores for this selection")
    label = METRICS[metric][0]
    fig = go.Figure()
    for m in models:
        g = df[df["model"] == m].sort_values("lead_h")
        if g.empty:
            continue
        color, dash, width = model_style(m)
        fig.add_trace(go.Scatter(x=g["lead_h"], y=g[metric], name=model_name(m), mode="lines+markers",
                                 line={"color": color, "dash": dash, "width": width}, marker={"size": 6},
                                 visible=True if primary is None or m in primary else "legendonly",
                                 hovertemplate="%{y:.2f}<extra>" + model_name(m) + "</extra>"))
    if metric == "coverage80":
        fig.add_hline(y=0.8, line={"color": INK2, "dash": "dot"})
        fig.update_yaxes(tickformat=".0%")
    else:
        fig.update_yaxes(title=f"{label} ({unit(variable)})", rangemode="tozero")
    fig.update_xaxes(title="hours ahead", tickvals=list(range(0, 49, 6)))
    return _layout(fig, height=360)


def station_map(stations: list[dict], selected: str | None) -> go.Figure:
    fig = go.Figure()
    for s in stations:
        on = s["id"] == selected
        fig.add_trace(go.Scattergeo(
            lon=[s["lon"]], lat=[s["lat"]], mode="markers+text", text=[station_label(s["id"]).split(" ", 1)[-1]],
            textposition="top center", textfont={"size": 10, "color": INK if on else INK2},
            marker={"size": 13 if on else 8, "color": "#2a78d6" if on else "#9ec5f4",
                    "line": {"color": SURFACE, "width": 2}},
            hovertemplate=f"{s['name']} ({s['id']})<br>{s['lat']:.2f}N {s['lon']:.2f}E, {s['altitude_m']:.0f} m"
                          "<extra></extra>", showlegend=False))
    fig.update_geos(scope="europe", lataxis_range=[41.5, 51.5], lonaxis_range=[-5.5, 9.5], showcountries=True,
                    countrycolor=GRID, showland=True, landcolor="#f3f2ee", showocean=True, oceancolor="#eef4fb",
                    coastlinecolor="#c9c8c2", bgcolor=SURFACE)
    fig.update_layout(height=230, margin={"l": 0, "r": 0, "t": 0, "b": 0}, paper_bgcolor=SURFACE,
                      font={"family": FONT})
    return fig
