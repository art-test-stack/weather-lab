"""Plotly figures for the explorer (pure functions of saved results)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from weather_lab.forecast import qcol
from weather_lab_app.labels import OBS_COLOR, model_name, model_style, unit, variable_name
from weather_lab_app.results import station_label

INK, INK2, GRID, SURFACE, FORECAST_BG = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb", "#eef4fb"
FONT = "Latin Modern Roman, Georgia, serif"


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

def last_observation(obs: pd.DataFrame, origin: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    """Latest observed (non-missing) target at or before the forecast time, with its real timestamp."""
    o = obs[(obs["time"] <= origin) & obs["observed"].notna()]
    if o.empty:
        return None
    r = o.loc[o["time"].idxmax()]
    return pd.Timestamp(r["time"]), float(r["observed"])


ERROR_LABELS = {"absolute": ("|Forecast − obs.|", "Absolute error |forecast − observation|"),
                "signed": ("Forecast − obs.", "Signed error (forecast − observation)")}


def forecast_figure(rows: pd.DataFrame, obs: pd.DataFrame, case: pd.Series, variable: str, systems: list[str],
                    band_model: str | None, context_hours: int = 720, view_hours: int = 48, height: int = 300,
                    error_mode: str = "absolute", error_rows: pd.DataFrame | None = None,
                    names: dict | None = None, styles: dict | None = None) -> go.Figure:
    """The forecast window first; the whole input context is plotted but the view opens on the last
    `view_hours` (zoom out to see more: display only, no model input changes).

    - A dotted bridge joins the last real observation at or before the forecast time to each point forecast's
      first value. It is a drawing aid only: not a prediction, not scored, not exported, absent from the
      error panel.
    - error_mode "absolute" (default) / "signed" adds an aligned panel (shared time axis) with each system's
      point forecast minus the observation, the same point forecast the benchmark scores; "off" hides it.
    """
    origin = pd.Timestamp(case["origin"])
    u = unit(variable)
    end = pd.Timestamp(rows["valid_time"].max()) if len(rows) else origin
    start = origin - pd.Timedelta(hours=context_hours)
    view_start = origin - pd.Timedelta(hours=min(view_hours, context_hours))
    panel = error_mode in ERROR_LABELS
    fig = make_subplots(rows=2 if panel else 1, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.73, 0.27] if panel else None)
    fig.add_vrect(x0=origin, x1=end + pd.Timedelta(hours=1.5), fillcolor=FORECAST_BG, line_width=0, layer="below",
                  row="all", col=1)
    fig.add_annotation(x=origin, y=1, yref="paper", xanchor="left", yanchor="top", showarrow=False,
                       text="  forecast →", font={"size": 12, "color": "#184f95"})
    fig.add_annotation(x=origin, y=1, yref="paper", xanchor="right", yanchor="top", showarrow=False,
                       text="← observed  ", font={"size": 12, "color": INK2})
    sel = rows[rows["model"].isin(systems)]
    b = sel[sel["model"] == band_model] if band_model else sel.iloc[:0]
    if len(b) and b["output_type"].iloc[0] == "quantiles":       # bands only where quantiles were predicted
        color = model_style(band_model)[0]
        for lo, hi, alpha, label in ((0.05, 0.95, 0.12, "90%"), (0.1, 0.9, 0.24, "80%")):
            fig.add_trace(go.Scatter(
                x=list(b["valid_time"]) + list(b["valid_time"][::-1]),
                y=list(b[qcol(hi)]) + list(b[qcol(lo)][::-1]), fill="toself", line={"width": 0},
                fillcolor=_rgba(color, alpha), name=f"80% / 90% range · {model_name(band_model)}",
                legendgroup="band", showlegend=label == "80%", hoverinfo="skip"), row=1, col=1)
    hist = obs[(obs["time"] > start) & (obs["time"] <= origin)]
    fut = obs[(obs["time"] > origin) & (obs["time"] <= end)]
    fig.add_trace(go.Scatter(x=hist["time"], y=hist["observed"], name="Observed", mode="lines+markers",
                             line={"color": OBS_COLOR, "width": 2}, marker={"size": 4}, legendgroup="obs",
                             hovertemplate="%{y:.1f} " + u + "<extra>Observed</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=fut["time"], y=fut["observed"], name="Observed (verification)", mode="markers",
                             legendgroup="obs", showlegend=False,
                             marker={"color": OBS_COLOR, "size": 9, "symbol": "circle-open", "line": {"width": 2}},
                             hovertemplate="%{y:.1f} " + u + "<extra>Observed (not an input)</extra>"), row=1, col=1)
    verif = fut.set_index("time")["observed"]
    last = last_observation(obs, origin)
    for model in [m for m in systems if m in set(sel["model"])]:
        g = sel[sel["model"] == model].sort_values("valid_time")
        color, dash, width = (styles or {}).get(model, model_style(model))
        name = (names or {}).get(model, model_name(model))
        lead = ((pd.to_datetime(g["valid_time"]) - origin) / pd.Timedelta(hours=1)).round().astype(int)
        first = g.dropna(subset=["point"]).head(1)
        if last is not None and len(first):                   # rendering only: never part of the data
            fig.add_trace(go.Scatter(x=[last[0], first["valid_time"].iloc[0]], y=[last[1], first["point"].iloc[0]],
                                     mode="lines", line={"color": color, "dash": "dot", "width": max(width - 0.8, 1)},
                                     legendgroup=model, showlegend=False, hoverinfo="skip",
                                     name=f"bridge · {name}", meta="bridge"), row=1, col=1)
        issued = "arome_lead_h" in g and g.arome_lead_h.notna().any()
        custom = list(zip(lead, g.arome_lead_h, g.arome_run.astype(str))) if issued else lead
        hover = ("%{customdata[0]} h after origin · %{customdata[1]} h after AROME initialization"
                 "<br>AROME initialized %{customdata[2]}" if issued else "%{customdata} h ahead")
        identifier = "" if names and model in names else f"<br><sub>{model}</sub>"
        fig.add_trace(go.Scatter(x=g["valid_time"], y=g["point"], name=name, mode="lines+markers",
                                 line={"color": color, "dash": dash, "width": width}, marker={"size": 5},
                                 legendgroup=model, customdata=custom,
                                 hovertemplate="%{y:.1f} " + u + " · " + hover +
                                               f"<extra>{name}{identifier}</extra>"), row=1, col=1)
        if panel:
            err = g["point"].to_numpy() - verif.reindex(pd.to_datetime(g["valid_time"])).to_numpy()
            if error_rows is not None:
                saved_errors = error_rows[error_rows.model.eq(model)].set_index("valid_time").err
                err = g.valid_time.map(saved_errors).to_numpy()
            err = abs(err) if error_mode == "absolute" else err
            fig.add_trace(go.Scatter(x=g["valid_time"], y=err, name=name, mode="lines+markers", showlegend=False,
                                     line={"color": color, "dash": dash, "width": max(width - 0.6, 1.2)},
                                     marker={"size": 4}, legendgroup=model, customdata=lead,
                                     hovertemplate=ERROR_LABELS[error_mode][0] + " "
                                                   + ("%{y:+.2f} " if error_mode == "signed" else "%{y:.2f} ") + u
                                                   + " · %{customdata} h ahead<br>valid %{x|%a %d %b %H:%M} UTC"
                                                   f"<extra>{name}</extra>"), row=2, col=1)
    fig.add_vline(x=origin, line={"color": INK2, "width": 1, "dash": "dash"})
    fig.update_yaxes(title=variable_name(variable), row=1, col=1)
    fig.update_xaxes(range=[view_start, end + pd.Timedelta(hours=1.5)], tickformat="%a %d %b\n%H:%M")
    if panel:
        if error_mode == "signed":
            fig.add_hline(y=0, line={"color": INK2, "width": 1}, row=2, col=1)
        fig.update_yaxes(title=f"{ERROR_LABELS[error_mode][0]} ({u})", row=2, col=1, title_font={"size": 11},
                         zeroline=False)
        if error_mode == "absolute":
            fig.update_yaxes(rangemode="tozero", row=2, col=1)
    return _layout(fig, height=height + (100 if panel else 40))


# --------------------------------------------------------------------------- benchmark

METRICS = {
    "mae": ("Mean absolute error", True, "point"),
    "rmse": ("Root-mean-square error", True, "point"),
    "pinball": ("Pinball loss (probabilistic)", True, "prob"),
    "coverage80": ("80% range coverage", False, "prob"),
    "coverage90": ("90% range coverage", False, "prob"),
    "width80": ("80% range width", None, "prob"),
    "width90": ("90% range width", None, "prob"),
}
NOMINAL = {"coverage80": 0.8, "coverage90": 0.9}


def comparison_bars(table: pd.DataFrame, metric: str, variable: str, reference: str | None,
                    hatched: frozenset[str] = frozenset(), names: dict | None = None,
                    styles: dict | None = None) -> go.Figure:
    """Horizontal bars sorted best first; reference system highlighted; values labelled with units."""
    if table.empty:
        return _empty("No scores for this selection")
    label, lower_better, _ = METRICS[metric]
    cov = metric in NOMINAL
    u = "" if cov else f" {unit(variable)}"
    t = table.dropna(subset=[metric]).copy()
    if cov:
        t["key"] = (t[metric] - NOMINAL[metric]).abs()          # closest to nominal first
    else:
        t["key"] = t[metric] if lower_better is not False else -t[metric]
    t = t.sort_values("key", ascending=False)                   # plotly draws bottom-up: best ends on top
    names = [(names or {}).get(m, model_name(m)) for m in t["model"]]
    colors = [(styles or {}).get(m, model_style(m))[0] for m in t["model"]]
    text = [f"{v:.0%}" if cov else f"{v:.2f}{u}" for v in t[metric]]
    fig = go.Figure(go.Bar(
        x=t[metric], y=names, orientation="h", marker={"color": colors, "pattern": {
            "shape": ["/" if m in hatched else "" for m in t["model"]], "fgcolor": SURFACE, "size": 7}, "line": {
            "color": [INK if m == reference else SURFACE for m in t["model"]],
            "width": [2.5 if m == reference else 0 for m in t["model"]]}},
        text=text, textposition="outside", cliponaxis=False,
        customdata=list(zip(t["model"], t["n"])),
        hovertemplate="%{y}: %{text}<br>n = %{customdata[1]} forecast–observation pairs"
                      "<br><sub>%{customdata[0]}</sub><extra></extra>"))
    if cov:
        fig.add_vline(x=NOMINAL[metric], line={"color": INK2, "dash": "dot"},
                      annotation_text=f"nominal {NOMINAL[metric]:.0%}", annotation_position="top")
        fig.update_xaxes(tickformat=".0%", range=[0, 1.05])
    else:
        hint = " · lower is better" if lower_better else " · narrower only helps if coverage holds"
        fig.update_xaxes(title=f"{label} ({unit(variable)}){hint}", rangemode="tozero")
    fig.update_layout(hovermode="closest", bargap=0.35)
    return _layout(fig, height=max(240, 46 * len(t) + 70), legend_top=False)


def metric_by_lead(table: pd.DataFrame, metric: str, variable: str, models: list[str],
                   primary: list[str] | None = None, cumulative: bool = False, spread: bool = False,
                   names: dict | None = None, styles: dict | None = None) -> go.Figure:
    """Score against lead from `weather_lab.horizon.by_lead`; systems outside `primary` start hidden.

    cumulative=False: value at each lead. cumulative=True: value over all leads up to H (a different
    quantity). Spread requests use population SD bars; otherwise saved bootstrap intervals are retained.
    """
    df = table[table["model"].isin(models)]
    if df.empty:
        return _empty("No lead-time scores for this selection")
    label = METRICS[metric][0]
    cov = metric in NOMINAL
    fig = go.Figure()
    for m in models:
        g = df[df["model"] == m].sort_values("lead_h")
        if g.empty:
            continue
        color, dash, width = (styles or {}).get(m, model_style(m))
        name = (names or {}).get(m, model_name(m))
        has_spread = spread and not cumulative and "spread" in g
        has_ci = not has_spread and g["ci95_low"].notna().any()
        err = ({"type": "data", "symmetric": True, "array": g.spread,
                "color": _rgba(color, 0.5), "thickness": 1.2, "width": 3} if has_spread else
               {"type": "data", "symmetric": False, "array": g["ci95_high"] - g[metric],
                "arrayminus": g[metric] - g["ci95_low"], "color": _rgba(color, 0.5), "thickness": 1.2,
                "width": 3} if has_ci else None)
        fmt = ".0%" if cov else ".2f"
        where = "all leads ≤ %{x} h" if cumulative else "%{x} h ahead"
        fig.add_trace(go.Scatter(
            x=g["lead_h"], y=g[metric], name=name, mode="lines+markers", error_y=err, legendgroup=m,
            line={"color": color, "dash": dash, "width": width}, marker={"size": 6},
            visible=True if primary is None or m in primary else "legendonly",
            customdata=list(zip(g["n"], g["n_cases"], g["ci95_low"], g["ci95_high"], g["spread"] if "spread" in g else [None] * len(g))),
            hovertemplate=f"%{{y:{fmt}}} · {where}<br>%{{customdata[0]}} pairs from %{{customdata[1]}} forecasts"
                          + ("<br>95% interval of the mean %{customdata[2]:.2f}–%{customdata[3]:.2f}" if has_ci
                             else "<br>±1 population SD of absolute errors %{customdata[4]:.3f}" if has_spread else "") + "<extra>" + name + "</extra>"))
    if cov:
        fig.add_hline(y=NOMINAL[metric], line={"color": INK2, "dash": "dot"},
                      annotation_text=f"nominal {NOMINAL[metric]:.0%}", annotation_position="bottom right")
        fig.update_yaxes(tickformat=".0%")
    else:
        fig.update_yaxes(title=f"{label} ({unit(variable)})", rangemode="tozero")
    fig.update_xaxes(title="all leads up to H (hours ahead)" if cumulative else "hours ahead (H)",
                     tickvals=list(range(0, 49, 6)))
    title = (f"{label}, cumulative: mean over every lead up to H" if cumulative else f"{label} at each lead H")
    fig = _layout(fig, title=title, height=470, legend_top=False)
    fig.update_layout(legend={"orientation": "h", "yanchor": "top", "y": -0.2, "x": 0, "font": {"size": 11}},
                      margin={"b": 130})
    return fig


def calendar_errors(table: pd.DataFrame, variable: str, period: str, clock: str,
                    spread: bool = False, names: dict | None = None, styles: dict | None = None) -> go.Figure:
    """Descriptive calendar MAE, retaining missing buckets as gaps."""
    if table.empty or table.mae.isna().all():
        return _empty("No matched observed pairs for this calendar selection")
    fig = go.Figure()
    for model, g in table.groupby("model", sort=False):
        color, dash, width = (styles or {}).get(model, model_style(model))
        name = (names or {}).get(model, model_name(model))
        fig.add_trace(go.Scatter(x=g.bucket, y=g.mae, name=name, mode="lines+markers", marker={"size": 5}, connectgaps=False,
            error_y={"type": "data", "symmetric": True, "array": g.spread, "color": _rgba(color, 0.5),
                     "thickness": 1, "width": 3} if spread else None,
            line={"color": color, "dash": dash, "width": width}, legendgroup=model,
            customdata=list(zip(g.n, g.n_cases, g.spread)),
            hovertemplate="%{x}<br>MAE %{y:.3f} " + unit(variable)
                + "<br>%{customdata[0]} observed pairs · %{customdata[1]} overlapping windows"
                + "<br>Population SD of absolute errors %{customdata[2]:.3f}<extra>" + name + "</extra>"))
    clock_label = "target valid time" if clock == "valid_time" else "forecast origin"
    fig.update_xaxes(title=f"{'Hour of day' if period == 'hour' else 'Date'} by {clock_label} (UTC)")
    if period == "hour":
        fig.update_xaxes(tickvals=list(range(0, 24, 3)), range=[0, 23])
    fig.update_yaxes(title=f"MAE ({unit(variable)})", rangemode="tozero")
    fig = _layout(fig, title=f"{'Hour-of-day profile' if period == 'hour' else 'Daily MAE'} · {clock_label} (UTC)",
                   height=440, legend_top=False)
    fig.update_layout(legend={"orientation": "h", "yanchor": "top", "y": -0.25, "x": 0}, margin={"b": 120})
    return fig


def forecast_windows(table: pd.DataFrame, variable: str, names: dict | None = None,
                     styles: dict | None = None) -> go.Figure:
    """One line per system: equal-location means of complete windows at each origin."""
    complete = table[table.complete]
    if complete.empty:
        return _empty("No complete saved 48-hour windows with all 48 observed pairs")
    fig = go.Figure()
    for model, model_rows in complete.groupby("model", sort=False):
        color, dash, width = (styles or {}).get(model, model_style(model))
        name = (names or {}).get(model, model_name(model))
        group = model_rows.groupby("origin", sort=True).agg(
            mae=("mae", "mean"), locations=("station", "nunique"), pairs=("n", "sum")).reset_index()
        fig.add_trace(go.Scatter(x=group.origin, y=group.mae, mode="lines", name=name, legendgroup=model,
            line={"color": color, "dash": dash, "width": width}, connectgaps=False,
            customdata=list(zip(group.locations, group.pairs)),
            hovertemplate="Origin %{x}<br>Mean location 48-hour MAE %{y:.3f} " + unit(variable)
                + "<br>%{customdata[0]} available locations · %{customdata[1]} hourly pairs · leads 1–48"
                + "<extra>" + name + "</extra>"))
    fig.update_xaxes(title="Forecast origin (UTC)")
    fig.update_yaxes(title=f"48-hour MAE ({unit(variable)})", rangemode="tozero")
    fig = _layout(fig, title="Mean location MAE per 48-hour forecast window", height=470, legend_top=False)
    fig.update_layout(legend={"orientation": "h", "yanchor": "top", "y": -0.2, "x": 0}, margin={"b": 100})
    return fig


def correction_gains(table: pd.DataFrame, corrected: list[str], variable: str,
                     names: dict | None = None, styles: dict | None = None) -> tuple[go.Figure, go.Figure]:
    """Pair complete location windows before averaging; take |gain| after averaging."""
    complete = table[table.complete]
    keys = ["case_id", "station", "origin"]
    reference = complete[complete.model.eq("arome-raw")][keys + ["mae"]].rename(columns={"mae": "arome_mae"})
    paired = complete[complete.model.isin(corrected)].merge(reference, on=keys, how="inner", validate="many_to_one")
    if paired.empty:
        message = "No selected residual-prediction models with complete windows shared with AROME"
        return _empty(message), _empty(message)
    paired["gain"] = paired.arome_mae - paired.mae
    signed, absolute = go.Figure(), go.Figure()
    for model, rows in paired.groupby("model", sort=False):
        group = rows.groupby("origin", sort=True).agg(gain=("gain", "mean"), locations=("station", "nunique")).reset_index()
        color, dash, width = (styles or {}).get(model, model_style(model))
        name = (names or {}).get(model, model_name(model))
        for fig, magnitude in ((signed, False), (absolute, True)):
            label = "|Gain|" if magnitude else "Gain"
            fig.add_trace(go.Scatter(x=group.origin, y=group.gain.abs() if magnitude else group.gain,
                mode="lines", name=name, legendgroup=model, connectgaps=False,
                line={"color": color, "dash": dash, "width": width}, customdata=group.locations,
                hovertemplate="Origin %{x}<br>" + label + " %{y:.3f} " + unit(variable)
                    + "<br>%{customdata} matched locations · 48 identical hourly pairs per location"
                    + "<extra>" + name + "</extra>"))
    signed.add_hline(y=0, line_color=INK2, line_dash="dash", line_width=1)
    for fig, label in ((signed, "Signed gain"), (absolute, "Absolute gain")):
        fig.update_xaxes(title="Forecast origin (UTC)")
        fig.update_yaxes(title=f"{label} ({unit(variable)})", rangemode="tozero")
        _layout(fig, title=f"{label} from AROME correction per 48-hour forecast window", height=470, legend_top=False)
        fig.update_layout(legend={"orientation": "h", "yanchor": "top", "y": -0.2, "x": 0}, margin={"b": 100})
    return signed, absolute


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
