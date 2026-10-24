"""Plotly figures for the explorer (pure functions of saved results)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from weather_lab.common import VARIABLES
from weather_lab.forecast import qcol
from weather_lab_app.results import station_label, style

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
FONT = "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif"


def _layout(fig: go.Figure, title: str, height: int = 380, **kw) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 15, "color": INK}}, height=height,
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font={"family": FONT, "color": INK2, "size": 12},
        margin={"l": 64, "r": 16, "t": 52, "b": 110}, hovermode="x unified",
        legend={"orientation": "h", "yanchor": "top", "y": -0.24, "x": 0, "font": {"size": 11}}, **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font={"size": 13, "color": INK2})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, "", height=260)


def skill_vs_lead(point: pd.DataFrame, variable: str) -> go.Figure:
    """MAE skill vs persistence by lead, on the common valid set."""
    df = point[point["lead_h"].notna()]
    if df.empty:
        return _empty("No scores for this selection")
    fig = go.Figure()
    fig.add_hline(y=0, line={"color": INK2, "width": 1})
    for model, g in df.groupby("model", sort=False):
        if model == "persistence":
            continue
        color, dash = style(model)
        g = g.sort_values("lead_h")
        fig.add_trace(go.Scatter(
            x=g["lead_h"], y=g["skill_mae_vs_persistence"], name=model, mode="lines+markers",
            line={"color": color, "dash": dash, "width": 2}, marker={"size": 7},
            customdata=g[["mae", "mae_ref", "n"]].to_numpy(),
            hovertemplate="%{y:.2f} (MAE %{customdata[0]:.2f} vs %{customdata[1]:.2f}, n=%{customdata[2]})"))
    unit = VARIABLES[variable][1]
    fig.update_xaxes(title="lead time (h)", tickvals=list(range(0, 49, 6)))
    fig.update_yaxes(title="MAE skill vs persistence")
    return _layout(fig, f"Point skill by lead (1 − MAE/MAE persistence) · {VARIABLES[variable][0]} ({unit})", height=430)


def coverage(prob: pd.DataFrame) -> go.Figure:
    """Empirical vs nominal central-interval coverage for probabilistic systems."""
    if prob.empty:
        return _empty("No probabilistic systems for this selection")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[75, 95], y=[0.75, 0.95], mode="lines", name="perfect calibration",
                             line={"color": INK2, "width": 1, "dash": "dot"}, hoverinfo="skip"))
    for _, r in prob.iterrows():
        color, dash = style(r["model"])
        fig.add_trace(go.Scatter(
            x=[80, 90], y=[r["coverage80"], r["coverage90"]], name=r["model"], mode="lines+markers",
            line={"color": color, "dash": dash, "width": 2}, marker={"size": 9, "line": {"color": SURFACE, "width": 2}},
            hovertemplate="%{y:.1%} observed in nominal %{x}% interval (n=" + str(r["n"]) + ")<extra>"
                          + r["model"] + "</extra>"))
    fig.update_xaxes(title="nominal central interval (%)", tickvals=[80, 90], range=[76, 94])
    fig.update_yaxes(title="empirical coverage", tickformat=".0%")
    fig.update_layout(hovermode="closest")
    return _layout(fig, "Interval coverage vs nominal (all leads)", height=430)


def forecast_figure(rows: pd.DataFrame, obs: pd.DataFrame, case: pd.Series, variable: str, systems: list[str],
                    band_model: str | None, history_days: int = 5, extra: pd.DataFrame | None = None) -> go.Figure:
    name, unit = VARIABLES[variable]
    origin = pd.Timestamp(case["origin"])
    fig = go.Figure()
    sel = rows[rows["model"].isin(systems)]
    band_sources = [(band_model, sel[sel["model"] == band_model])] if band_model else []
    if extra is not None and not extra.empty:
        band_sources.append((extra["model"].iloc[0], extra))
    for model, b in band_sources:
        if b.empty or b["output_type"].iloc[0] != "quantiles":
            continue  # deterministic systems never get bands
        color, _ = style(model)
        for lo, hi, alpha, label in ((0.05, 0.95, 0.12, "90%"), (0.1, 0.9, 0.22, "80%")):
            fig.add_trace(go.Scatter(
                x=list(b["valid_time"]) + list(b["valid_time"][::-1]),
                y=list(b[qcol(hi)]) + list(b[qcol(lo)][::-1]), fill="toself", line={"width": 0},
                fillcolor=_rgba(color, alpha), name=f"{model} {label} interval", hoverinfo="skip"))
    hist = obs[(obs["time"] > origin - pd.Timedelta(days=history_days)) & (obs["time"] <= origin)]
    fut = obs[(obs["time"] > origin) & (obs["time"] <= rows["valid_time"].max())] if len(rows) else obs.iloc[:0]
    fig.add_trace(go.Scatter(x=hist["time"], y=hist["observed"], name="observed (input history)", mode="lines",
                             line={"color": INK, "width": 2}))
    fig.add_trace(go.Scatter(x=fut["time"], y=fut["observed"], name="observed (verification, not an input)",
                             mode="markers", marker={"color": INK, "size": 8, "symbol": "circle-open",
                                                     "line": {"width": 2}}))
    for model, g in sel.groupby("model", sort=False):
        color, dash = style(model)
        fig.add_trace(go.Scatter(x=g["valid_time"], y=g["point"], name=model, mode="lines+markers",
                                 line={"color": color, "dash": dash, "width": 2}, marker={"size": 6}))
    if extra is not None and not extra.empty:
        color, dash = style("exploratory")
        fig.add_trace(go.Scatter(x=extra["valid_time"], y=extra["point"], name=extra["model"].iloc[0],
                                 mode="lines+markers", line={"color": color, "dash": dash, "width": 2}))
    fig.add_vline(x=origin, line={"color": INK2, "width": 1, "dash": "dash"})
    note = f"origin {origin:%Y-%m-%d %H:%M} UTC · historical replay"
    if isinstance(case.get("arome_run"), str):
        note += f" · AROME run {case['arome_run'][:16]}Z, in archive {case['arome_available_at'][11:16]}Z"
    fig.add_annotation(x=origin, y=1.06, yref="paper", text=note, showarrow=False, xanchor="left",
                       font={"size": 11, "color": INK2})
    fig.update_yaxes(title=f"{name} ({unit})")
    fig.update_xaxes(title="valid time (UTC)")
    return _layout(fig, f"{station_label(case['station'])} · {name}", height=520)


def station_map(stations: list[dict], selected: str | None) -> go.Figure:
    fig = go.Figure()
    for s in stations:
        on = s["id"] == selected
        fig.add_trace(go.Scattergeo(
            lon=[s["lon"]], lat=[s["lat"]], mode="markers+text", text=[s["name"].title()],
            textposition="top center", textfont={"size": 10, "color": INK if on else INK2},
            marker={"size": 13 if on else 9, "color": "#2a78d6" if on else "#9ec5f4",
                    "line": {"color": SURFACE, "width": 2}},
            hovertemplate=f"{s['id']} {s['name']}<br>{s['lat']:.3f}N {s['lon']:.3f}E, {s['altitude_m']:.0f} m"
                          "<extra></extra>", showlegend=False))
    fig.update_geos(scope="europe", lataxis_range=[41.5, 51.5], lonaxis_range=[-5.5, 9.5], showcountries=True,
                    countrycolor=GRID, showland=True, landcolor="#f3f2ee", showocean=True, oceancolor="#eef4fb",
                    coastlinecolor="#c9c8c2", bgcolor=SURFACE)
    fig.update_layout(height=300, margin={"l": 0, "r": 0, "t": 8, "b": 0}, paper_bgcolor=SURFACE,
                      font={"family": FONT})
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
