"""Forecast presentation and exports derived from the applied result."""

from __future__ import annotations

from weather_lab_app import zerogpu, plots, forecast_state as forecast, results

import tempfile
from pathlib import Path
import gradio as gr
import numpy as np
import pandas as pd
from weather_lab.common import ROOT, sha256_text
from weather_lab_app.labels import SPLIT_NAMES, model_description, model_name, model_style, unit
from weather_lab.forecast import qcol
from weather_lab_app.results import case_forecast, forecast_csv, headline, headline_to, station_label
from weather_lab.forecast_config import describe


def tiles_html(suite: str, variable: str, size: str | None = None, setup: str | None = None,
               horizon: int | None = None, extra: str | None = None, metric: str = "mae") -> str:
    """Benchmark aggregate cards for the selected experiment, checkpoint and input setup (saved score tables).

    With `horizon` below the run's inference horizon they are recomputed from the saved predictions on leads
    <= horizon and labelled as an interactive slice; the frozen full-horizon scores stay in the saved tables.
    """
    run, e = results.RUNS[suite], forecast.experiment_view(suite, size, setup)
    if extra and extra in run.models and extra not in e["headline"]:      # the applied saved system is always shown
        e = {**e, "headline": e["headline"] + [extra]}
    sliced = horizon is not None and int(horizon) < run.horizon_hours
    h = headline_to(run, variable, e["headline"], int(horizon)) if sliced else headline(run, variable,
                                                                                         models=e["headline"])
    not_run = "".join(f"<div class='tile missing' title='{m}'><div class='v'>not run</div><div class='n'>"
                      f"{model_name(m)}</div><div class='d'>no saved result in this experiment; nothing is "
                      "substituted</div></div>" for m in e["missing"])
    if h.empty:
        return "<div class='note'>No headline systems for this variable.</div>"
    u = unit(variable)
    split = run.splits[-1]
    ref = e["reference"]
    errors = h.set_index("model")[metric]
    secondary = "rmse" if metric == "mae" else "mae"

    def versus(value: float, other: str) -> str:
        if errors[other] == 0:
            return f"{model_name(other)} has zero {metric.upper()}"
        ratio = value / errors[other]
        if ratio >= 1.5:
            return f"{ratio:.1f}× the error of {model_name(other)}"
        return f"{abs(ratio - 1):.0%} {'higher' if ratio > 1 else 'lower'} error than {model_name(other)}"

    out = []
    for r in h.itertuples():
        color = model_style(r.model)[0]
        parts = []
        value = getattr(r, metric)
        other = e.get("compare", {}).get(r.model)
        if other in errors.index:
            parts.append(versus(value, other))
        if r.model != ref and ref in errors.index:
            parts.append(versus(value, ref))
        detail = "reference" if r.model == ref else " · ".join(parts)
        out.append(f"<div class='tile{' ref' if r.model == ref else ''}' title='{r.model}'>"
                   f"<div class='v'>{value:.2f} {u}</div>"
                   f"<div class='n'><span class='sw' style='background:{color}'></span>{r.name}</div>"
                   f"<div class='d'>{metric.upper()} · {secondary.upper()} {getattr(r, secondary):.2f} {u}"
                   f" · {detail}</div></div>")
    band = e.get("band")
    cov = h.set_index("model")["coverage80"].get(band) if band in set(h["model"]) else None
    if cov is not None and pd.notna(cov):
        out.append(f"<div class='tile' title='{band}'><div class='v'>{cov:.0%}</div>"
                   f"<div class='n'>of observations inside the 80% range</div>"
                   f"<div class='d'>{model_name(band)} · nominal 80%</div></div>")
    n = int(h["n"].max())
    split_txt = f" · {SPLIT_NAMES.get(split, split)}" if len(run.splits) > 1 else ""
    score_name = "mean absolute error" if metric == "mae" else "root mean squared error"
    what = (f"<b>Benchmark aggregate, interactive slice</b>: {score_name} over {n:,} forecast–observation "
            f"pairs with lead ≤ {int(horizon)} h of the saved {run.horizon_hours} h forecasts, all stations"
            f"{split_txt}, recomputed from saved predictions. Frozen full-horizon scores: Benchmark tab."
            if sliced else
            f"<b>Benchmark aggregate</b> (frozen): {score_name} over {n:,} forecast–observation pairs, all "
            f"stations and lead times{split_txt}; lower is better.")
    return f"<div class='tiles'>{''.join(out)}{not_run}</div><div class='note'>{what}</div>"


def render(applied, changed=True):
    """Every Forecast output from the applied result, in one step. Skipped when nothing was applied."""
    if not changed:
        return tuple(gr.skip() for _ in range(9))
    if applied is None:
        return (gr.update(choices=[], value=[]), gr.update(choices=[("none", "")], value=""),
                plots._empty("Select an hourly UTC origin, then Run forecast"),
                "No forecast has run. Date edits only change the draft.", "", "", None,
                plots._empty("No forecast selected"), "")
    a = applied
    v = a["view"]
    fig, caption, this, desc, csv, smap = overview(a, v["variable"], v["shown"], v["band"], v["error_mode"])
    return (gr.update(choices=[(model_name(m), m) for m in v["choices"]], value=v["shown"]),
            gr.update(choices=[("none", "")] + [(model_name(m), m) for m in v["prob"]], value=v["band"]),
            fig, caption, this, desc, csv, smap, cards_html(a, v["variable"]))


def _utc(df: pd.DataFrame) -> pd.DataFrame:
    """Saved (ns) and fresh timestamps must share one dtype before they are combined."""
    df = df.copy()
    for col in ("origin", "valid_time"):
        if col in df:
            df[col] = pd.to_datetime(df[col], utc=True).astype("datetime64[us, UTC]")
    return df


def _window(rows: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return rows[rows["lead_h"] <= int(horizon)]


def forecast_scores(rows: pd.DataFrame, obs: pd.DataFrame, systems: list[str]) -> tuple[pd.DataFrame, int]:
    """Per system on THIS forecast: leads covered, and on the leads where every listed system has a point
    forecast and the observation exists (shared window): MAE, RMSE, 80% coverage and width."""
    r = rows[rows["model"].isin(systems)].merge(obs.rename(columns={"time": "valid_time"}), on="valid_time",
                                                how="left")
    covers = r.groupby("model")["lead_h"].max()
    ok = r.dropna(subset=["point", "observed"])
    shared = set.intersection(*(set(g["lead_h"]) for _, g in ok.groupby("model"))) if len(ok) else set()
    have = set(ok["model"])
    shared = shared if have == set(covers.index) else set()
    s = ok[ok["lead_h"].isin(shared)].copy()
    s["abs_err"] = (s["point"] - s["observed"]).abs()
    s["sq_err"] = (s["point"] - s["observed"]).pow(2)
    if qcol(0.1) in s:
        s["in80"] = ((s["observed"] >= s[qcol(0.1)]) & (s["observed"] <= s[qcol(0.9)])).where(s[qcol(0.1)].notna())
        s["width80"] = s[qcol(0.9)] - s[qcol(0.1)]
    out = pd.DataFrame({"covers_h": covers})
    if len(s):
        agg = s.groupby("model").agg(mae=("abs_err", "mean"), n=("abs_err", "size"),
                                     rmse=("sq_err", "mean"),
                                     **({"cov80": ("in80", "mean"), "width80": ("width80", "mean")}
                                        if "in80" in s else {}))
        agg["rmse"] = agg["rmse"].pow(0.5)
        out = out.join(agg)
    return out.reindex([m for m in systems if m in out.index]), len(shared)


def this_forecast_html(scores: pd.DataFrame, n_shared: int, variable: str, metric: str = "mae") -> str:
    u = unit(variable)
    metrics = (metric, "rmse" if metric == "mae" else "mae")
    observable = n_shared > 0
    cell = lambda x, f: f(x) if observable and pd.notna(x) else "Not yet observable" if not observable else "–"  # noqa: E731
    rows = "".join(
        f"<tr><td><span class='sw' style='background:{model_style(m)[0]}'></span>{model_name(m)}</td>"
        f"<td>{int(r.covers_h)} h</td>"
        + "".join(f"<td>{cell(r.get(key), lambda v: f'{v:.2f} {u}')}</td>" for key in metrics)
        + f"<td>{cell(r.get('cov80'), lambda v: f'{v:.0%}')}</td></tr>" for m, r in scores.iterrows())
    return ("<table class='mini'><thead><tr><th>System</th><th>Shown to</th>"
            + "".join(f"<th>{key.upper()}</th>" for key in metrics) +
            f"<th>80% coverage</th></tr></thead><tbody>{rows}</tbody></table><div class='note'>This forecast only, "
            f"on the {n_shared} lead(s) where every listed system has a forecast and the observation exists"
            + ("" if observable else ": none observed yet") + ".</div>")


def cards_html(applied, variable) -> str:
    """Primary cards: the applied exploratory forecast (labelled), or the benchmark aggregate of the applied
    saved view over its displayed slice."""
    a = applied
    metric = a["view"].get("metric", "mae")
    secondary = "rmse" if metric == "mae" else "mae"
    if a["source"] == "saved":
        return tiles_html(a["suite"], variable, a["size"], a["setup"] or forecast.e_main(a["suite"]), a["shown_h"],
                          a["primary"], metric)
    run, mid, st = results.RUNS[a["suite"]], a["primary"], a["settings"]
    frame = a["exploratory"][mid]
    rows, _, case = case_forecast(run, a["case"], variable)
    obs = _station_obs(run, case["station"], variable, a)
    f = _window(frame[frame["variable"] == variable], a["shown_h"])
    u = unit(variable)
    v = f.merge(obs.rename(columns={"time": "valid_time"}), on="valid_time", how="left")
    seen = v.dropna(subset=["point", "observed"])
    n_obs, n_leads = len(seen), len(f)
    head = (f"<div class='note'><b>Current forecast · Exploratory</b> — {forecast.settings_text(st)}; shown to "
            f"{a['shown_h']} h; {n_obs} of {n_leads} leads observed so far.</div>")
    color = model_style(mid)[0]
    nyo = "<div class='v'>Not yet observable</div>"
    err = seen["point"] - seen["observed"]
    values = {"mae": err.abs().mean(), "rmse": np.sqrt(err.pow(2).mean())}
    cards = []
    for key in (metric, secondary):
        value = f"<div class='v'>{values[key]:.2f} {u}</div>" if n_obs else nyo
        cards.append(f"<div class='tile{' ref' if key == metric else ''}' title='{mid}'>{value}<div class='n'>"
                     f"<span class='sw' style='background:{color}'></span>{key.upper()} · this forecast</div>"
                     f"<div class='d'>{n_obs} observed lead(s)</div></div>")
    if qcol(0.1) in f and f[qcol(0.1)].notna().any():
        width = (f[qcol(0.9)] - f[qcol(0.1)]).mean()
        inside = ((seen["observed"] >= seen[qcol(0.1)]) & (seen["observed"] <= seen[qcol(0.9)])).mean()
        cards.append(f"<div class='tile'>{f'<div class=v>{inside:.0%}</div>' if n_obs else nyo}<div class='n'>"
                     f"inside its 80% range</div><div class='d'>nominal 80%; {n_obs} observation(s) — too few to "
                     "judge calibration</div></div>")
        cards.append(f"<div class='tile'><div class='v'>{width:.1f} {u}</div><div class='n'>mean 80% range width"
                     f"</div><div class='d'>over its {n_leads} lead(s)</div></div>")
    rows = rows[~rows.model.isin(a.get("exploratory", {}))]
    all_rows = _utc(pd.concat([_utc(rows), *[_utc(df[df["variable"] == variable])
                                           for df in a["exploratory"].values()]], ignore_index=True))
    comparison = a.get("comparison") if a["config"].prediction_target == "arome_error" else \
        forecast.primary_saved(a["suite"], a["size"], a["setup"])
    others = [m for m in ("arome-raw", "arome-biascorr", comparison) if m and m != mid and m in set(all_rows["model"])]
    sc, n_shared = forecast_scores(_window(all_rows, a["shown_h"]), obs, [mid] + others)
    for m in others:
        val = sc.loc[m, metric] if metric in sc and m in sc.index else None
        other_score = (f"{secondary.upper()} {sc.loc[m, secondary]:.2f} {u} · "
                       if secondary in sc and pd.notna(sc.loc[m, secondary]) else "")
        cards.append(f"<div class='tile' title='{m}'><div class='v'>"
                     + (f"{val:.2f} {u}" if val is not None and pd.notna(val) else "Not yet observable")
                     + f"</div><div class='n'><span class='sw' style='background:{model_style(m)[0]}'></span>"
                     f"{model_name(m)}</div><div class='d'>{metric.upper()} · {other_score}"
                     + ("previously applied forecast" if m == a.get("comparison") else "issued AROME comparison" if m in a["exploratory"] else "saved forecast")
                     + f", on the {n_shared} lead(s) both cover and "
                     "observed" + (f"; this forecast scores {sc.loc[mid, metric]:.2f} {u} {metric.upper()} there"
                                   if metric in sc and pd.notna(sc.loc[mid, metric]) else "") + "</div></div>")
    return (head + f"<div class='tiles'>{''.join(cards)}</div><div class='note'>One forecast, outside the frozen "
            "benchmark: not saved to results and never scored there. The benchmark aggregates over all cases are in "
            "the Benchmark tab.</div>")


def _station_obs(run, station: str, variable: str, applied=None) -> pd.DataFrame:
    if applied and applied["config"].retrieval_method != "none":
        from weather_lab.rag_store import Store

        path = Store().get("archives", applied["settings"]["archive_id"])
        obs = pd.read_parquet(path / "observations.parquet")
    else:
        obs = run.full_observations()
    return obs[obs["station"] == station][["time", variable]].rename(columns={variable: "observed"})


def overview(applied, variable, systems, band_model, error_mode="absolute"):
    a = applied
    run = results.RUNS[a["suite"]]
    rows, _, case = case_forecast(run, a["case"], variable)
    obs = _station_obs(run, case["station"], variable, a)
    rows = rows[~rows.model.isin(a.get("exploratory", {}))]
    exploratory = [df[df["variable"] == variable] for df in a.get("exploratory", {}).values()]
    if exploratory:
        rows = _utc(pd.concat([_utc(rows), *map(_utc, exploratory)], ignore_index=True))
    rows = _window(rows, a["shown_h"])
    systems = [m for m in (systems or []) if m in set(rows["model"])]
    ctx_days = a["config"].context_days
    fig = plots.forecast_figure(rows, obs, case, variable, systems, band_model or None,
                                context_hours=int(ctx_days) * 24, error_mode=error_mode or "off")
    scores, n_shared = forecast_scores(rows, obs, systems)
    metric = a["view"].get("metric", "mae")
    this = this_forecast_html(scores, n_shared, variable, metric)
    desc = "\n".join(f"- **{model_name(m)}**: {model_description(m)}" for m in systems)
    origin = pd.Timestamp(case["origin"])
    plotted = int(rows.loc[rows["model"].isin(systems), "lead_h"].max()) if systems else 0
    arome = (f" AROME run used: {case['arome_run'][:16].replace('T', ' ')} UTC (in the archive at "
             f"{case['arome_available_at'][11:16]} UTC)." if isinstance(case.get("arome_run"), str) else "")
    assumed = (" AROME availability is inferred retrospectively from archive timestamps."
               if isinstance(case.get("arome_run"), str) else "")
    if a["config"].retrieval_method != "none" and a["settings"].get("arome_run"):
        st = a["settings"]
        arome = f" Conditioning AROME run: {st['arome_run']} · product {st['arome_product']}."
        assumed = " AROME availability uses the archived run's explicit replay assumption."
    last = plots.last_observation(obs, origin)
    gap = ("" if last is None or last[0] == origin else
           f" The last observation before the forecast time is at {last[0]:%a %d %b %H:%M} UTC, "
           f"{(origin - last[0]) / pd.Timedelta(hours=1):.0f} h earlier; the dotted lines start there.")
    err_txt = {"absolute": " Lower panel: absolute error |forecast − observation|",
               "signed": " Lower panel: signed error, forecast − observation (above zero: too high)"}
    context_label = "Model input window: up to" if a["config"].prediction_target == "arome_error" else "Model input:"
    caption = (f"**{station_label(case['station']).split(' ', 1)[-1]}**, forecast issued "
               f"{origin:%a %d %b %Y, %H:%M} UTC (historical replay: inputs restricted to data assumed available "
               f"at the forecast time).{assumed}{arome} {context_label} **{ctx_days} days** of observations "
               f"({int(ctx_days) * 24 // a['config'].step_hours} steps); the chart opens on the last 48 h — zoom out to see "
               f"the rest. Plotted: leads {a['config'].step_hours}–{plotted} h. Shaded ranges show where the probabilistic system expects "
               "the outcome 80% / 90% of the time; AROME and the simple references give single values. Dotted lines "
               "join the last observation to each forecast's first value: a drawing aid, not a prediction, never "
               "scored or exported." + gap
               + (err_txt[error_mode] + ", from the same point forecast the benchmark scores." if error_mode in
                  err_txt else ""))
    caption += f" Main metric: {metric.upper()}; MAE and RMSE use the same observed leads."
    if run.run.get("exploration") and a["config"].retrieval_method == "none":
        caption += (" Hourly exploration uses QC-accepted station records; rejected or absent samples remain missing. "
                    + run.run["hourly_data"]["observation_availability"] + " Forecasts are outside the frozen benchmark.")
    if a["config"].retrieval_method != "none":
        caption += (" RAG inputs end 3 h before the decision; generated bridge steps are excluded. "
                    "Observations on this display come from the selected index's hourly archive. "
                    "Saved benchmark overlays retain their original inputs and cadence.")
    if a["config"].prediction_target == "arome_error":
        coverage = "; ".join(f"{name}: {r['observed_steps']} of {r['total_steps']} steps, "
                              f"{r['first'][:16]} to {r['last'][:16]} UTC"
                              for name, r in a["settings"].get("error_history", {}).items())
        caption += (" **AROME + predicted error:** the model context is observation minus qualifying archived "
                    "AROME forecasts; missing archived errors stay missing. "
                    + (f"Available error history: {coverage}. " if coverage else "")
                    + "Forecast lead is supplied for the history and horizon. Calendar inputs are "
                    + ("enabled (UTC hour of day and day of year). " if a["config"].calendar else "disabled. ")
                    + "Selected engineered features use the original observation history. "
                    + (f"Previous direct prediction retained: {model_name(a['comparison'])}. "
                       if a.get("comparison") else "No previous direct prediction exists for this case. ")
                    + "Intervals are model error quantiles shifted by the issued AROME baseline; their "
                    "coverage has not been established by this single forecast.")
    csv_rows = rows[rows["model"].isin(systems)]            # what is plotted, over the applied horizon
    out = forecast_csv(csv_rows, obs)
    out.insert(0, "source", np.where(out["model"].isin(a.get("exploratory", {})),
                                     "exploratory (outside the frozen benchmark, not scored)", "saved benchmark run"))
    out["applied_configuration"] = forecast.applied_text(a)
    out["main_metric"] = metric
    configs = {**forecast._systems_as_configs(a["suite"]), **a.get("configs_by", {})}
    out["forecast_configuration"] = out["model"].map(
        lambda m: describe(configs[m]) if m in configs else model_description(m))
    out["prediction_method"] = out["model"].map(
        lambda m: configs[m].prediction_target if m in configs else "raw_arome" if m == "arome-raw" else "reference")
    out["inference_horizon_h"] = out["model"].map(
        lambda m: run.horizon_hours if m in run.models else configs[m].horizon_h if m in configs else None)
    for column in ("retrieval_kind", "arome_product", "arome_run"):
        out[column if column != "arome_run" else "conditioning_arome_run"] = out["model"].map(
            lambda m: a.get("settings_by", {}).get(m, {}).get(column))
    out["retrieval_index"] = out["model"].map(lambda m: configs[m].retrieval_index if m in configs else None)
    csv = out.to_csv(index=False)
    export_dir = ROOT / ".tmp" / "codex" / "forecast-exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / f"forecast_{a['case'].replace('/', '_')}_{variable}_{sha256_text(csv)}.csv"
    if not path.exists():
        with tempfile.TemporaryDirectory(dir=export_dir) as tmp:
            staging = Path(tmp) / "forecast.csv"
            staging.write_text(csv)
            staging.replace(path)
    return fig, caption, this, desc, str(path), plots.station_map(run.stations(), case["station"])
