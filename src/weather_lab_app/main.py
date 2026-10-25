"""Weather-lab: Gradio app presenting saved benchmark results.

Overview (landing): what is compared, a large forecast chart and headline numbers for the selected
experiment. Benchmark: a compact comparison, details below. Methods: scope, definitions, exact
identifiers, previous versions and integrity details. Nothing here runs the benchmark; the only
model call is the optional exploratory forecast, labelled as such.
"""

from __future__ import annotations

from weather_lab_app import zerogpu  # noqa: I001 - must precede torch/gradio on ZeroGPU Spaces

import sys
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from weather_lab.common import ROOT, VARIABLES
from weather_lab_app import plots
from weather_lab_app.labels import (CHECKPOINTS, DEFAULT_EXPERIMENT, DIFFERENT_INPUT, EXPERIMENTS, MODELS,
                                    SEASON_NAMES, SETUP_NAMES, SPLIT_NAMES, model_description, model_name,
                                    model_style, unit, variable_name, with_checkpoint)
from weather_lab_app.results import (available_runs, case_forecast, forecast_csv, headline, station_label,
                                     training_status)


def startup_bootstrap() -> dict | None:
    """Fetch/verify required artifacts before building the UI (idempotent; reuses the cache)."""
    import os

    from weather_lab.bootstrap import BootstrapError, bootstrap

    if os.environ.get("WEATHER_LAB_BOOTSTRAP", "on") == "off":
        return None
    try:
        return bootstrap()
    except BootstrapError as e:
        print(f"bootstrap failed: {e}", flush=True)
        from weather_lab.bootstrap import _state

        return _state()


BOOT: dict | None = None
RUNS: dict = {}
DEFAULT_SUITE: str | None = None
INTRO = ("Météo-France's **AROME**, the pretrained **Timestral** model (inference only) and Timestral given "
         "AROME's forecast, at French stations 3–48 h ahead.")
QUESTION = ("How do Timestral forecasts compare with AROME's at French stations, 3–48 h ahead, and does giving "
            "Timestral AROME's forecast change its error?")
STATUSES = ["pretrained, no local fine-tuning", "fitted on fitting period", "external NWP (issued run)", "no fitting"]
EMPTY_EXTRA = {"case": None, "runs": {}}

CSS = """
.gradio-container {max-width: 1240px !important; margin: auto;}
#hero h1 {font-size: 1.45rem; margin: 0 0 .1rem 0;}
#hero p {color: #3d3c39; margin: 0; font-size: .98rem;}
.finding {font-size: 1rem; color: #0b0b0b; margin: .2rem 0;}
.scope {font-size: .9rem; color: #52514e; line-height: 1.45;}
.scope .status {display:inline-block; padding:1px 9px; border-radius:10px; background:#fff4e0; color:#7a4b00;
                font-weight:600; margin-right:6px; font-size:.82rem;}
.scope .status.validation {background:#e6f4ea; color:#0f5a2e;}
#scope {margin: -6px 0;}
.tiles {display:flex; flex-wrap:wrap; gap:10px; margin:.2rem 0 0 0;}
.tile {flex:1 1 170px; border:1px solid #e6e5e1; border-radius:10px; padding:7px 12px; background:#fcfcfb;}
.tile .v {font-size:1.4rem; font-weight:650; color:#0b0b0b; line-height:1.1;}
.tile .n {font-size:.86rem; color:#3d3c39; margin-top:2px;}
.tile .d {font-size:.78rem; color:#6b6a65; margin-top:4px;}
.tile.ref {border:2px solid #0b0b0b;}
.tile.missing {border-style:dashed; background:transparent;}
.tile.missing .v {font-size:1.1rem; color:#6b6a65;}
table.matrix td, table.matrix th {text-align:left !important; white-space:normal !important;}
.tile .sw {display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px;}
.note {font-size:.85rem; color:#52514e;}
table.mini {border-collapse:collapse; width:100%; font-size:.9rem;}
table.mini th {text-align:left; font-weight:600; color:#52514e; border-bottom:1px solid #e6e5e1; padding:6px 4px;}
table.mini td {border-bottom:1px solid #f0efec; padding:6px 4px;}
table.mini td:last-child, table.mini th:last-child {text-align:right; white-space:nowrap;}
table.mini .sw {display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px;}
/* data tables scroll inside their own box instead of widening the page */
.gradio-container .table-wrap, .gradio-container .table-container {max-width:100%; overflow-x:auto;}
.gradio-container {overflow-x:hidden;}
@media (max-width: 640px) { #hero h1 {font-size:1.3rem;} .tile {flex:1 1 140px;} .tile .v {font-size:1.3rem;} }
"""


# --------------------------------------------------------------------------- shared helpers

def experiment_choices(show_superseded: bool = False) -> list[tuple[str, str]]:
    return [(EXPERIMENTS[s]["name"], s) for s in RUNS if show_superseded or not EXPERIMENTS[s].get("superseded")]


def variable_choices(suite: str) -> list[tuple[str, str]]:
    have = set(RUNS[suite].predictions()["variable"])
    return [(variable_name(v, unit=False), v) for v in VARIABLES if v in have]


def scope_html(suite: str) -> str:
    e, run = EXPERIMENTS[suite], RUNS[suite]
    cls = "status validation" if e["status"] == "Validation" else "status"
    return (f"<div class='scope'><span class='{cls}'>{e['status']}</span>{e.get('short', e['scope'])} · "
            "scope, caveats and availability rules in <b>Methods</b></div>")


def has_checkpoints(suite: str) -> bool:
    """True when a headline 28M system has a saved 209M twin not already in the headline (selector shown)."""
    if has_setups(suite):
        return True
    models, head = set(RUNS[suite].models), EXPERIMENTS[suite]["headline"]
    twins = [with_checkpoint(m, "209M") for m in head if m.startswith("timestral-28m-")]
    return any(t in models and t not in head for t in twins)


def matrix_id(suite: str, size: str, setup: str) -> str | None:
    """System id of an input setup at a checkpoint size, from the experiment's matrix (None if no matrix)."""
    mx = EXPERIMENTS[suite].get("matrix")
    if not mx or setup not in mx["setups"]:
        return None
    return mx["base"].format(size=size.lower()) + mx["setups"][setup]


def has_setups(suite: str) -> bool:
    return bool(EXPERIMENTS[suite].get("matrix"))


def setup_choices(suite: str) -> list[tuple[str, str]]:
    mx = EXPERIMENTS[suite].get("matrix") or {}
    return [(SETUP_NAMES[k][0], k) for k in mx.get("setups", {}) if k != "none"]


def experiment_view(suite: str, size: str | None, setup: str | None = None) -> dict:
    """The experiment's presentation for a checkpoint size and AROME input setup.

    Ids that were not run stay in the view (shown as "not run"); nothing is substituted. `missing` lists them.
    """
    e, models = EXPERIMENTS[suite], set(RUNS[suite].models)
    mx = e.get("matrix")
    if mx:
        size = size if size in mx["sizes"] else mx["sizes"][0]
        setup = setup if setup in mx["setups"] else mx["main"]
        main_ids = {matrix_id(suite, s, mx["main"]) for s in mx["sizes"]}
        base_ids = {matrix_id(suite, s, "none") for s in mx["sizes"]}

        def swap(m: str) -> str:
            if m in main_ids:
                return matrix_id(suite, size, setup)
            if m in base_ids:
                return matrix_id(suite, size, "none")
            return m
    elif size in CHECKPOINTS and has_checkpoints(suite):
        swap = lambda m: with_checkpoint(m, size) if with_checkpoint(m, size) in models else m  # noqa: E731
    else:                                                    # hidden selectors: the declared systems
        return {**e, "missing": []}
    uniq = lambda ms: list(dict.fromkeys(swap(m) for m in ms))  # noqa: E731
    compare = {swap(k): swap(v) for k, v in e.get("compare", {}).items() if swap(k) != swap(v)}
    headline = uniq(e["headline"])
    band = swap(e["band"])
    return {**e, "headline": headline, "default_systems": [m for m in uniq(e["default_systems"]) if m in models],
            "band": band if band in models else "", "compare": compare,
            "missing": [m for m in headline if m not in models]}


def _saved_models(run, variable: str) -> list[str]:
    p = run.predictions()
    have = set(p.loc[p["variable"] == variable, "model"])
    return [m for m in run.models if m in have]


# --------------------------------------------------------------------------- overview

def station_choices(suite: str) -> list[tuple[str, str]]:
    return [(station_label(s).split(" ", 1)[-1], s) for s in RUNS[suite].manifest["protocol"]["stations"]]


def date_choices(suite: str, station: str) -> list[tuple[str, str]]:
    run = RUNS[suite]
    cases = run.cases()
    cases = cases[cases["eligible"] & (cases["station"] == station)].sort_values("origin")
    multi = cases["split"].nunique() > 1
    return [(f"{c.origin:%a %d %b %Y · %H:%M} UTC" + (f" · {SPLIT_NAMES.get(c.split, c.split)}" if multi else "")
             , c.case_id) for c in cases.itertuples()]


def default_case(suite: str, station: str) -> str:
    """Fixed, neutral rule: the most recent forecast of the last period for this station."""
    cases = RUNS[suite].cases()
    cases = cases[cases["eligible"] & (cases["station"] == station)]
    last = cases[cases["split"] == RUNS[suite].splits[-1]]
    return (last if len(last) else cases).sort_values("origin")["case_id"].iloc[-1]


def on_experiment(suite: str, variable: str | None):
    """Experiment changed: reset variable (kept if available), station and date to sensible defaults."""
    vchoices = variable_choices(suite)
    variable = variable if variable in [v for _, v in vchoices] else "t"
    stations = station_choices(suite)
    first_example = (RUNS[suite].manifest.get("ui_examples") or [""])[0]
    station = first_example.split("/")[1] if first_example.count("/") == 2 else stations[0][1]
    return (gr.update(choices=vchoices, value=variable), gr.update(choices=stations, value=station),
            gr.update(choices=date_choices(suite, station), value=default_case(suite, station)),
            scope_html(suite), gr.update(visible=has_checkpoints(suite)),
            gr.update(visible=has_setups(suite), choices=setup_choices(suite),
                      value=(EXPERIMENTS[suite].get("matrix") or {}).get("main")))


def on_station(suite: str, station: str):
    return gr.update(choices=date_choices(suite, station), value=default_case(suite, station))


def tiles_html(suite: str, variable: str, size: str | None = None, setup: str | None = None) -> str:
    """Headline numbers for the selected experiment, checkpoint and input setup (saved score tables)."""
    run, e = RUNS[suite], experiment_view(suite, size, setup)
    h = headline(run, variable, models=e["headline"])
    not_run = "".join(f"<div class='tile missing' title='{m}'><div class='v'>not run</div><div class='n'>"
                      f"{model_name(m)}</div><div class='d'>no saved result in this experiment; nothing is "
                      "substituted</div></div>" for m in e["missing"])
    if h.empty:
        return "<div class='note'>No headline systems for this variable.</div>"
    u = unit(variable)
    split = run.splits[-1]
    ref = e["reference"]
    ref_mae = h.set_index("model")["mae"].get(ref)
    maes = h.set_index("model")["mae"]

    def versus(value: float, other: str) -> str:
        ratio = value / maes[other]
        if ratio >= 1.5:
            return f"{ratio:.1f}× the error of {model_name(other)}"
        return f"{abs(ratio - 1):.0%} {'higher' if ratio > 1 else 'lower'} error than {model_name(other)}"

    out = []
    for r in h.itertuples():
        color = model_style(r.model)[0]
        parts = []
        other = e.get("compare", {}).get(r.model)
        if other in maes.index:
            parts.append(versus(r.mae, other))
        if r.model != ref and ref_mae is not None:
            parts.append(versus(r.mae, ref))
        detail = "reference" if r.model == ref else " · ".join(parts)
        out.append(f"<div class='tile{' ref' if r.model == ref else ''}' title='{r.model}'>"
                   f"<div class='v'>{r.mae:.2f} {u}</div>"
                   f"<div class='n'><span class='sw' style='background:{color}'></span>{r.name}</div>"
                   f"<div class='d'>{detail}</div></div>")
    band = e.get("band")
    cov = h.set_index("model")["coverage80"].get(band) if band in set(h["model"]) else None
    if cov is not None and pd.notna(cov):
        out.append(f"<div class='tile' title='{band}'><div class='v'>{cov:.0%}</div>"
                   f"<div class='n'>of observations inside the 80% range</div>"
                   f"<div class='d'>{model_name(band)} · nominal 80%</div></div>")
    n = int(h["n"].max())
    split_txt = f" · {SPLIT_NAMES.get(split, split)}" if len(run.splits) > 1 else ""
    return (f"<div class='tiles'>{''.join(out)}{not_run}</div><div class='note'><b>Mean absolute error</b> over {n:,} "
            f"forecast–observation pairs, all stations and lead times{split_txt}; lower is better.</div>")


def systems_update(suite, case_id, variable, pref_by, band, extra, size=None, setup=None):
    """Systems that forecast this variable; keeps the user's choices per experiment and the band."""
    run, e = RUNS[suite], experiment_view(suite, size, setup)
    saved = _saved_models(run, variable)
    exploratory = list(_extra_runs(extra, case_id))
    choices = saved + exploratory
    pref = (pref_by or {}).get(suite)
    if pref is None:
        value = [m for m in e["default_systems"] if m in saved] + exploratory
    else:
        value = [m for m in choices if m in pref]
    preds = run.predictions()
    prob = [m for m in saved if m in value and (preds.loc[preds["model"] == m, "output_type"] == "quantiles").any()]
    prob += [m for m in exploratory if m in value]
    if band is None or (band != "" and band not in prob):     # None: not chosen yet; "": user chose none
        band = e["band"] if e["band"] in prob else (prob[0] if prob else "")
    return (gr.update(choices=[(model_name(m), m) for m in choices], value=value),
            gr.update(choices=[("none", "")] + [(model_name(m), m) for m in prob], value=band))


def update_pref(suite, case_id, variable, selected, pref_by, extra, size=None, setup=None):
    run, e = RUNS[suite], experiment_view(suite, size, setup)
    choices = _saved_models(run, variable) + list(_extra_runs(extra, case_id))
    pref_by = dict(pref_by or {})
    base = pref_by.get(suite, list(e["default_systems"]))
    pref_by[suite] = [m for m in base if m not in choices] + list(selected or [])
    return pref_by


def switch_view(suite, size, setup, prev, pref_by):
    """Size or input setup changed: the chosen systems that were the previous view's Timestral and
    Timestral + AROME follow the new view (only to saved systems; others are left as chosen)."""
    prev = prev or {}
    new = {"size": size, "setup": setup}
    if not (pref_by or {}).get(suite) or not has_checkpoints(suite):
        return pref_by, new
    models = set(RUNS[suite].models)
    old_view = experiment_view(suite, prev.get("size"), prev.get("setup"))
    new_view = experiment_view(suite, size, setup)
    mapping = {o: n for o, n in zip(old_view["headline"], new_view["headline"]) if o != n and n in models}
    swapped = [mapping.get(m, m) for m in pref_by[suite]]
    return {**pref_by, suite: list(dict.fromkeys(swapped))}, new


def _extra_runs(extra: dict | None, case_id: str) -> dict:
    return extra["runs"] if extra and extra.get("case") == case_id else {}


def keep_extra_for_case(case_id, extra):
    return extra if extra and extra.get("case") == case_id else {"case": case_id, "runs": {}}


def overview(suite, case_id, variable, systems, band_model, history_hours, extra=None, error_mode="signed"):
    run = RUNS[suite]
    rows, obs, case = case_forecast(run, case_id, variable)
    exploratory = [df[df["variable"] == variable] for df in _extra_runs(extra, case_id).values()]
    if exploratory:
        rows = pd.concat([rows, *exploratory], ignore_index=True)
        for col in ("origin", "valid_time"):    # saved (ns) and fresh timestamps must share one dtype
            rows[col] = pd.to_datetime(rows[col], utc=True).astype("datetime64[us, UTC]")
    systems = systems or []
    fig = plots.forecast_figure(rows, obs, case, variable, systems, band_model or None, int(history_hours),
                                error_mode=error_mode or "off")
    sc = rows[rows["model"].isin(systems)].merge(obs.rename(columns={"time": "valid_time"}), on="valid_time")
    sc = sc.assign(err=(sc["point"] - sc["observed"]).abs()).groupby("model", sort=False)["err"].mean()
    sc = sc.reindex([m for m in systems if m in sc.index])
    this = ("<table class='mini'><thead><tr><th>System</th><th>Mean error, this forecast</th></tr></thead><tbody>"
            + "".join(f"<tr><td><span class='sw' style='background:{model_style(m)[0]}'></span>{model_name(m)}</td>"
                      f"<td>{v:.2f} {unit(variable)}</td></tr>" for m, v in sc.items())
            + "</tbody></table>")
    desc = "\n".join(f"- **{model_name(m)}**: {model_description(m)}" for m in systems)
    origin = pd.Timestamp(case["origin"])
    arome = (f" AROME run used: {case['arome_run'][:16].replace('T', ' ')} UTC (in the archive at "
             f"{case['arome_available_at'][11:16]} UTC)." if isinstance(case.get("arome_run"), str) else "")
    assumed = (" AROME availability is inferred retrospectively from archive timestamps."
               if isinstance(case.get("arome_run"), str) else "")
    caption = (f"**{station_label(case['station']).split(' ', 1)[-1]}**, forecast issued "
               f"{origin:%a %d %b %Y, %H:%M} UTC (historical replay: inputs restricted to data assumed available "
               f"at the forecast time).{assumed}{arome} Shaded ranges show where the probabilistic system "
               "expects the outcome 80% / 90% of the time; AROME and the simple references give single values."
               + (" Lower panel: forecast minus observation" + (" (absolute)" if error_mode == "absolute" else "")
                  + ", from the same point forecast the benchmark scores." if error_mode in ("signed", "absolute")
                  else ""))
    path = Path(tempfile.gettempdir()) / f"forecast_{case_id.replace('/', '_')}_{variable}.csv"
    forecast_csv(rows, obs).to_csv(path, index=False)
    return fig, caption, this, desc, str(path), plots.station_map(run.stations(), case["station"])


@zerogpu.gpu(duration=60)
def _exploratory_on_device(obs, case, size, context_days, calendar, arome_mode, arome_df, allow_download):
    """One labelled Timestral run; on ZeroGPU this function holds a GPU for its duration."""
    from weather_lab.forecast import exploratory_forecast
    from weather_lab.models import MODELS

    device = zerogpu.inference_device()
    try:
        return exploratory_forecast(obs, case, ["t", "pmer", "u", "v"], size, context_days, calendar,
                                    arome_mode=arome_mode, arome_df=arome_df, allow_download=allow_download,
                                    device=device)
    finally:
        if device == "cuda" and zerogpu.on_zerogpu():
            MODELS.release("cuda")   # the GPU is detached after this call; never keep GPU-resident state


EXPLORE_SETUPS = {"No AROME input": None, "AROME forecast + archived AROME history": "archived",
                  "AROME forecast + observed history (substituted)": "substituted_observations",
                  "AROME forecast only (diagnostic)": "missing"}


def run_exploratory(suite, case_id, size, context_days, calendar, setup_label, allow_download, pref_by, extra,
                    ckpt=None, setup=None):
    from weather_lab import predcache
    from weather_lab.forecast import exploratory_forecast

    run = RUNS[suite]
    _, _, case = case_forecast(run, case_id, "t")
    obs = run.observations()
    obs = obs[obs["station"] == case["station"]]
    extra = keep_extra_for_case(case_id, extra)
    arome_mode = EXPLORE_SETUPS.get(setup_label)
    arome_df = run.arome_extract() if arome_mode else None
    try:
        from weather_lab.forecast import ensure_checkpoint

        # download on CPU first, so a cold 837 MB fetch never consumes the GPU window or quota
        ensure_checkpoint(size, bool(allow_download))
        args = (obs, case, size, int(context_days), bool(calendar), arome_mode, arome_df, bool(allow_download))
        if zerogpu.on_zerogpu():
            # Cache lookup on CPU first: a hit never requests a GPU.
            frame, info = exploratory_forecast(obs, case, ["t", "pmer", "u", "v"], size, int(context_days),
                                               bool(calendar), arome_mode=arome_mode, arome_df=arome_df,
                                               device="cuda", lookup_only=True)
            if frame is None:
                frame, info = _exploratory_on_device(*args)
        else:
            frame, info = _exploratory_on_device(*args)
    except (PermissionError, ValueError) as e:
        return extra, pref_by, f"⚠️ {e}"
    except Exception as e:  # noqa: BLE001 - e.g. ZeroGPU quota/queue errors raised from the GPU worker
        return extra, pref_by, f"⚠️ exploratory run failed: {type(e).__name__}: {e}"
    mid = frame["model"].iloc[0]
    extra = {"case": case_id, "runs": {**extra["runs"], mid: frame}}
    pref_by = dict(pref_by or {})
    base = pref_by.get(suite, list(experiment_view(suite, ckpt, setup)["default_systems"]))
    pref_by[suite] = base + ([mid] if mid not in base else [])
    where = predcache.location()[1]
    return extra, pref_by, (f"Added **{model_name(mid)}** to the chart (outside the frozen benchmark; never scored). "
                            + ("Reused a cached prediction (no model run). " if info["cache_hit"]
                               else "Computed now and cached. ") + f"Cache: {where}.")


# --------------------------------------------------------------------------- benchmark

LEADS = [("All lead times", "all"), ("6 h ahead", "6"), ("24 h ahead", "24"), ("48 h ahead", "48")] + [
    (f"{h} h ahead", str(h)) for h in range(3, 49, 3) if h not in (6, 24, 48)]
METRIC_CHOICES = [("Mean absolute error", "mae"), ("RMSE", "rmse"), ("Pinball loss", "pinball"),
                  ("80% range coverage", "coverage80"), ("90% range coverage", "coverage90"),
                  ("80% range width", "width80"), ("90% range width", "width90")]
LEAD_VIEWS = [("At each lead", "at"), ("Cumulative: all leads up to H", "cumulative")]
ERROR_MODES = [("Signed (forecast − observation)", "signed"), ("Absolute", "absolute"), ("Off", "off")]


def bench_controls(suite: str):
    run = RUNS[suite]
    splits = run.splits
    stations = [("All stations", "ALL")] + station_choices(suite)
    seasons = sorted(set(run.tables["metrics_point"]["season"]), key=lambda s: list(SEASON_NAMES).index(s))
    return (gr.update(choices=[(SPLIT_NAMES.get(s, s), s) for s in splits], value=splits[-1],
                      visible=len(splits) > 1),
            gr.update(choices=stations, value="ALL"),
            gr.update(choices=[(SEASON_NAMES.get(s, s), s) for s in seasons], value="ALL"))


def _select(df, split, variable, station, season, lead):
    m = (df["split"] == split) & (df["variable"] == variable) & (df["station"] == station) & (df["season"] == season)
    m &= df["lead_h"].isna() if lead == "all" else (df["lead_h"] == int(lead))
    return df[m]


def finding(suite: str, variable: str, split: str | None = None, setup: str | None = None) -> str:
    """One sentence on the AROME comparison for the selected input setup, built only from clauses the saved
    all-lead scores support (no assumption that a larger checkpoint or more inputs must help)."""
    run, e = RUNS[suite], EXPERIMENTS[suite]
    if "arome-raw" not in run.models:
        return ""
    mx = e.get("matrix")
    setup = setup if mx and setup in mx["setups"] else (mx or {}).get("main", "archived")
    split = split or run.splits[-1]
    sel = lambda d: d[(d["split"] == split) & (d["variable"] == variable) & (d["station"] == "ALL")  # noqa: E731
                      & (d["season"] == "ALL") & d["lead_h"].isna()]
    allmae = sel(run.tables["metrics_point"]).set_index("model")["mae"]
    cover = sel(run.tables["metrics_probabilistic"]).set_index("model")["coverage80"]
    mae = allmae[[m for m in allmae.index if m not in DIFFERENT_INPUT]]
    if mae.empty:
        return ""
    clauses, diffs, notes = [], [], []
    if "arome-raw" not in mae.index:
        clauses.append("AROME is not scored for this variable")
    elif mae.idxmin() == "arome-raw":
        clauses.append("AROME remains most accurate")
    else:
        clauses.append(f"{model_name(mae.idxmin())} is more accurate than AROME")
    sizes = mx["sizes"] if mx else list(CHECKPOINTS)
    ids = {s: (f"timestral-{s.lower()}-mv-cal",
               matrix_id(suite, s, setup) or f"timestral-{s.lower()}-mv-cal-arome-archived") for s in sizes}
    pairs = {s: (b, c) for s, (b, c) in ids.items() if b in allmae.index and c in allmae.index}
    missing = [s for s, (b, c) in ids.items() if b in allmae.index and c not in allmae.index]
    setup_txt = SETUP_NAMES[setup][0] if mx else "AROME"
    if pairs:
        better = [s for s, (b, c) in pairs.items() if allmae[c] < allmae[b]]
        diffs += [abs(allmae[c] - allmae[b]) / allmae[b] for b, c in pairs.values()]
        verb = f"adding the {setup_txt}" if mx else "conditioning on AROME"
        cond = (f"{verb} improves Timestral" if len(better) == len(pairs) else
                f"{verb} improves Timestral {', '.join(better)} only" if better
                else f"{verb} does not improve Timestral")
        conds = [c for _, c in pairs.values()]
        if len(conds) == 2:
            small, large = conds
            diffs.append(abs(allmae[large] - allmae[small]) / allmae[small])
            cond += (", with the larger checkpoint performing better" if allmae[large] < allmae[small]
                     else ", and the larger checkpoint does not perform better")
        clauses.append(cond)
        low = [f"{model_name(c)} {cover[c]:.0%}" for _, c in pairs.values() if c in cover and cover[c] < 0.6]
        if low:
            notes.append(f"Its 80% ranges are far too narrow (observations inside: {', '.join(low)}): overconfident "
                         "probabilistic forecasts.")
        if setup == "substobs":
            notes.append("Observed history is a different input construction, not real AROME history.")
    if missing:
        notes.append(f"Not run: {', '.join(model_name(ids[s][1]) for s in missing)}.")
    if diffs and min(diffs) < 0.03:
        notes.append("Some of these differences are a few percent, within what this small sample can resolve; see "
                     "the paired differences under Advanced.")
    sample = e.get("sample", "this experiment")
    return (f"<div class='finding'><b>In {sample}</b> ({variable_name(variable, unit=False).lower()}, mean "
            f"absolute error over all stations and lead times): {'; '.join(clauses)}.{' ' if notes else ''}"
            f"{' '.join(notes)}</div>")


def lead_view(suite, variable, metric, station, split, statuses, size=None, setup=None, cumulative=False):
    """Score against lead, recomputed from saved predictions on the matched cases of the compared systems."""
    from weather_lab.horizon import by_lead, lead_ranges, scored_rows

    run, e = RUNS[suite], experiment_view(suite, size, setup)
    split = split or run.splits[-1]
    models = [m for m in _saved_models(run, variable) if training_status(m) in (statuses or STATUSES)]
    kind = plots.METRICS[metric][2]
    if kind == "prob":
        preds = run.predictions()
        prob = set(preds.loc[preds["output_type"] == "quantiles", "model"])
        models = [m for m in models if m in prob]
    if not models:
        return plots.metric_by_lead(pd.DataFrame(columns=["model"]), metric, variable, []), ""
    rows = scored_rows(run.predictions(), run.targets(), run.cases(), variable, models, metric, station, split)
    table = by_lead(rows, metric, cumulative=bool(cumulative))
    primary = list(dict.fromkeys(e["headline"] + [e["reference"]]))
    fig = plots.metric_by_lead(table, metric, variable, models, primary, bool(cumulative))
    n_per = table.groupby("lead_h")["n_cases"].max()
    blocks = int(table["n_blocks"].max()) if len(table) else 0
    ref = e["reference"] if kind == "point" else e.get("prob_reference", "climatology")
    lines = []
    if plots.METRICS[metric][1] is not None and ref in set(table["model"]):
        label = plots.METRICS[metric][0].lower()
        what = "is closer to nominal coverage than" if metric.startswith("coverage") else f"has a lower {label} than"
        for m in [x for x in primary if x != ref and x in set(table["model"])]:
            rng = lead_ranges(table, metric, m, ref)
            txt = ", ".join(f"{a}–{b} h" if a != b else f"{a} h" for a, b in rng) if rng else "no lead"
            lines.append(f"{model_name(m)} {what} {model_name(ref)} at {txt}")
    where = station_label(station).split(" ", 1)[-1] if station != "ALL" else "all stations"
    view = "all leads up to H" if cumulative else "at each lead H"
    ci = (f"Error bars: 95% block-bootstrap interval of the mean ({blocks} two-day blocks), an uncertainty of the "
          "score, not a forecast range." if blocks >= 10 else
          f"No interval shown: the cases span {blocks} two-day blocks, fewer than the 10 the project requires.")
    rng_n = f"{int(n_per.min())}" + (f"–{int(n_per.max())}" if n_per.min() != n_per.max() else "")
    text = (f"**How the error evolves** ({view}, {where}): " + ("; ".join(lines) + ". " if lines else "")
            + f"Matched cases: every system uses the same forecasts at each lead ({rng_n} per lead); leads of one "
            "forecast are not independent. "
            + ("Cumulative = mean over every (forecast, lead) pair up to H"
               + (" (root of the mean square for RMSE)" if metric == "rmse" else "") + ", not the value at H. "
               if cumulative else "") + ci)
    return fig, text


def matrix_html(suite: str, variable: str, split: str | None = None) -> str:
    """Input setup x checkpoint: saved scores, 'not run' (possible, no result) or 'unsupported input'."""
    from weather_lab.models import FAMILIES

    run, e = RUNS[suite], EXPERIMENTS[suite]
    mx = e.get("matrix")
    if not mx:
        return "<div class='note'>This experiment has no input-setup matrix.</div>"
    split = split or run.splits[-1]
    sel = lambda d: d[(d["split"] == split) & (d["variable"] == variable) & (d["station"] == "ALL")  # noqa: E731
                      & (d["season"] == "ALL") & d["lead_h"].isna()].set_index("model")
    pt, pr = sel(run.tables["metrics_point"]), sel(run.tables["metrics_probabilistic"])
    u = unit(variable)
    planned = [f for f in FAMILIES.values() if f.status == "planned"]
    head = "".join(f"<th>Timestral {s}</th>" for s in mx["sizes"]) + "".join(f"<th>{f.name}</th>" for f in planned)
    body = []
    for key in mx["setups"]:
        cells = []
        for s in mx["sizes"]:
            mid = matrix_id(suite, s, key)
            if mid in pt.index:
                prob = (f" · pinball {pr.loc[mid, 'pinball']:.2f} · 80% cov. {pr.loc[mid, 'coverage80']:.0%}"
                        if mid in pr.index else "")
                cells.append(f"<td title='{mid}'>MAE {pt.loc[mid, 'mae']:.2f} {u}{prob}</td>")
            else:
                cells.append(f"<td title='{mid}'><i>not run</i></td>")
        for f in planned:
            ok = key == "none" or f.known_covariates
            cells.append(f"<td><i>{'not integrated yet' if ok else 'unsupported input'}</i></td>")
        body.append(f"<tr><td><b>{SETUP_NAMES[key][0]}</b></td>{''.join(cells)}</tr>")
    return (f"<table class='mini matrix'><thead><tr><th>Input setup ({variable_name(variable, unit=False).lower()}, "
            f"all leads)</th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
            "<div class='note'>Within a row only the checkpoint differs: same cases, observations, AROME runs, "
            "history length, calendar inputs and scoring. <i>not run</i>: the model can take this input but no "
            "result is saved; <i>unsupported input</i>: the model family cannot take it.</div>")


def benchmark(suite, variable, metric, lead, split, station, season, statuses, size=None, setup=None,
              lead_mode="at"):
    run, e = RUNS[suite], experiment_view(suite, size, setup)
    cumulative = lead_mode == "cumulative"
    split = split or run.splits[-1]
    kind = plots.METRICS[metric][2]
    src = run.tables["metrics_point" if kind == "point" else "metrics_probabilistic"]
    keep = lambda d: d[d["model"].map(training_status).isin(statuses or STATUSES)]  # noqa: E731
    t = keep(_select(src, split, variable, station, season, lead))
    ref = e["reference"] if e["reference"] in set(t["model"]) else None
    hatched = frozenset(m for m in t["model"] if m in DIFFERENT_INPUT)
    bars = plots.comparison_bars(t, metric, variable, ref, hatched)
    lines, horizon_txt = lead_view(suite, variable, metric, station, split, statuses, size, setup, cumulative)
    n = int(t["n"].max()) if len(t) else 0
    na = run.tables.get("probabilistic_not_applicable", pd.DataFrame())
    na_names = sorted({model_name(m) for m in na.get("model", [])}) if kind == "prob" else []
    lead_txt = next(lbl for lbl, v in LEADS if v == lead)
    where = (station_label(station).split(" ", 1)[-1] if station != "ALL" else "all stations") + \
        (f", {SEASON_NAMES[season].lower()}" if season != "ALL" else "")
    summary = (f"**{variable_name(variable)} · {lead_txt.lower()} · {where}"
               + (f" · {SPLIT_NAMES.get(split, split)}" if len(run.splits) > 1 else "") + f"** — every system is "
               f"scored on the same {n:,} forecast–observation pairs."
               + (f" Outlined: the reference ({model_name(ref)})." if ref else "")
               + (f" Single-value systems ({', '.join(na_names)}) have no probabilistic score." if na_names else "")
               + (" Hatched: " + " and ".join(model_name(m) for m in sorted(hatched))
                  + (" use" if len(hatched) > 1 else " uses") + " station observations in place of past AROME values,"
                  " a different input from real AROME history." if hatched else "")
               + " In the lead-time chart, click a legend entry to show or hide a system."
               + (" The lead-time chart covers every season." if season != "ALL" else ""))
    return (finding(suite, variable, split, setup), summary, bars, lines, horizon_txt,
            *detail_tables(run, split, variable, station, season, lead, statuses))


def detail_tables(run, split, variable, station, season, lead, statuses):
    u = unit(variable)
    keep = lambda d: d[d["model"].map(training_status).isin(statuses or STATUSES)]  # noqa: E731
    p = keep(_select(run.tables["metrics_point"], split, variable, station, season, lead)).sort_values("mae")
    point = pd.DataFrame({"System": p["model"].map(model_name), f"MAE ({u})": p["mae"].round(2),
                          f"RMSE ({u})": p["rmse"].round(2), f"Bias ({u})": p["bias"].round(2),
                          "vs persistence": p["skill_mae_vs_persistence"].map(lambda x: f"{x:+.0%}"),
                          "Pairs": p["n"]})
    q = keep(_select(run.tables["metrics_probabilistic"], split, variable, station, season, lead)).sort_values(
        "pinball")
    prob = pd.DataFrame({"System": q["model"].map(model_name), f"Pinball ({u})": q["pinball"].round(3),
                         "80% coverage": q["coverage80"].map("{:.0%}".format),
                         "90% coverage": q["coverage90"].map("{:.0%}".format),
                         f"80% width ({u})": q["width80"].round(1), "Pairs": q["n"]})
    return point, prob


def advanced_tables(suite, variable, split):
    run = RUNS[suite]
    split = split or run.splits[-1]
    a = run.tables["availability"]
    a = a[(a["split"] == split) & (a["variable"] == variable)]
    avail = pd.DataFrame({"System": a["model"].map(model_name), "Cases": a["cases_in_suite"],
                          "Eligible": a["cases_eligible"], "Targets observed": a["observed_targets"],
                          "Failed": a["failed_predictions"], "Scored pairs": a["valid_pairs"]})
    d = run.tables["paired_differences"]
    d = d[(d["split"] == split) & (d["variable"] == variable)]
    paired = pd.DataFrame({"System": d["model"].map(model_name), "vs": d["reference"].map(model_name),
                           "Score": d["loss"].map({"abs_err": "abs. error", "pinball": "pinball"}),
                           "Lead": d["lead_h"].map(lambda x: "all" if pd.isna(x) else f"{int(x)} h"),
                           "Mean difference": d["mean_diff"].round(3),
                           "95% interval": [f"[{lo:.3f}, {hi:.3f}]" if pd.notna(lo) else "n/a (too few blocks)"
                                            for lo, hi in zip(d["ci95_low"], d["ci95_high"])],
                           "Blocks": d["n_blocks"]})
    rt = pd.DataFrame([{"System": model_name(s["id"]), "Per forecast (ms, batch)": s["per_case_ms"],
                        "Single forecast (ms)": s.get("warm_single_case_ms_median"),
                        "Parameters": f"{s['n_parameters'] / 1e6:.0f}M" if s.get("n_parameters") else "",
                        "Peak memory (MB)": s["peak_rss_mb_after"]} for s in run.run["systems"]])
    env = run.run["environment"]
    hw = (f"CPU timings on {env.get('cpu')} ({env['machine']}), torch {env['torch']}, {env['torch_threads']} "
          "threads. Peak memory is process-wide and cumulative across systems.")
    return avail, paired, rt, hw


def downloads(suite: str):
    p = RUNS[suite].path
    return str(p / "metrics_point.csv"), str(p / "metrics_probabilistic.csv"), str(p / "manifest.json")


# --------------------------------------------------------------------------- methods

def methods_md() -> str:
    exp_rows = "\n".join(
        f"| {EXPERIMENTS[s]['name']} | {EXPERIMENTS[s]['status']} | {EXPERIMENTS[s]['scope']} | "
        f"{EXPERIMENTS[s]['caveat']} | `{s}` |"
        for s in RUNS)
    seen = sorted({m for r in RUNS.values() for m in r.models}, key=lambda m: list(MODELS).index(m)
                  if m in MODELS else 99)
    sys_rows = "\n".join(f"| {model_name(m)} | `{m}` | {model_description(m)} |" for m in seen)
    return f"""
## What is being tested

{QUESTION}

Each **experiment** is a frozen set of forecast cases (stations × forecast times), scored separately and never
pooled. Forecasts are replayed from inputs restricted to data assumed available at the forecast time. Timestral
is used as released (no local training).

| Experiment | Status | Scope | Caveat | Identifier |
|---|---|---|---|---|
{exp_rows}

**AROME availability is an assumption.** An AROME run counts as available at a forecast time if Météo-France's
archive file was already published then, judged from the archive's Last-Modified timestamps after the fact (a
retrospective reconstruction, not a live record). Details: `docs/availability_evidence.md`.

## Systems

| Name in the app | Exact identifier | What it is |
|---|---|---|
{sys_rows}

**Time inputs, not calibration.** "Timestral 28M" receives hour-of-day and time-of-year values as known inputs
(identifiers ending in `-cal`, for *calendar*). No system is calibrated or trained locally: Timestral runs as
released. **AROME history**: Timestral only uses an AROME input if it also sees that input's past; "archived
history" uses earlier AROME runs available at the time, "observed history" substitutes station observations (a
different input). Details: `docs/diagnostics/arome_covariates.md`.

## Scores

- **Mean absolute error / RMSE**: average size of the error of the single-value forecast (the median for
  Timestral), in the variable's unit.
- **Pinball loss**: accuracy of the whole predicted distribution (21 quantiles), in the variable's unit; lower is
  better. Only for probabilistic systems.
- **80% range coverage**: share of observations inside the predicted 80% range; ideally 80%.
- All systems in a table are scored on the same forecast–observation pairs.
"""


def cache_md() -> str:
    """Checkpoint files on disk, models loaded in this process, saved predictions, exploratory cache."""
    from weather_lab.status import cache_status

    st = cache_status()
    ck = "\n".join(f"| {r['checkpoint']} | `{r['repo']}@{r['revision']}` | {'on disk' if r['on_disk'] else 'absent'} "
                   f"| {r['mb']} MB |" for r in st["checkpoints"])
    loaded = ", ".join(f"{r['repo'].split('/')[-1]} on {r['device']} (~{r['approx_mb']} MB)" for r in st["loaded"]) \
        or "none (benchmark views read saved files; a model loads only for an exploratory run that misses the cache)"
    runs = "\n".join(f"| {EXPERIMENTS.get(s, {}).get('name', s)} | `{r.run['run_id'][:60]}` | {len(r.models)} | "
                     f"{r.source} |" for s, r in RUNS.items())
    pc = st["prediction_cache"]
    ms = st["model_stats"]
    return f"""
**Checkpoint files** (Hub cache `{st['hub_cache']}`; offline mode: {st['offline']}). Every input setup of a
checkpoint shares these files.

| Checkpoint | Pinned revision | Files | Size |
|---|---|---|---|
{ck}

**Loaded in this process:** {loaded}. Loads so far: {ms['loads']}, reuses: {ms['load_hits']}, evictions:
{ms['evictions']}, downloads: {ms['downloads']}.

**Saved benchmark predictions** (shown without loading any model):

| Experiment | Run | Systems | Source |
|---|---|---|---|
{runs}

**Exploratory prediction cache:** {pc['entries']} entries ({pc['mb']} MB) at `{pc['path']}`: {pc['persistence']}.
Hits {pc['hits']}, misses {pc['misses']} in this process. Entries are keyed by checkpoint revision, input data
and masks, input setup, history length, horizon, quantiles, device and dtype; never reused across revisions or
setups. What survives restarts and rebuilds: `docs/caching.md`.
"""


def integrity_md() -> str:
    rows = "\n".join(
        f"| {EXPERIMENTS[s]['name']} | `{r.run['run_id']}` | {r.source} | `{r.manifest['protocol_sha256']}` | "
        f"{r.manifest['n_eligible']}/{r.manifest['n_cases']} | `{r.run['code_revision'][:12]}` |"
        for s, r in RUNS.items())
    return f"""| Experiment | Run | Source | Protocol hash | Eligible cases | Code |
|---|---|---|---|---|---|
{rows}

The *AROME vs Timestral · v1* experiment is a superseded version kept unchanged; enable "show superseded
versions" in the Overview's advanced panel to browse it. Fitting/adaptation status filters, availability,
paired differences and timings are in the Benchmark tab's advanced section."""


def protocol_md() -> str:
    p = ROOT / "docs" / "experiment_protocol.md"
    return p.read_text() if p.exists() else "See docs/experiment_protocol.md."


# --------------------------------------------------------------------------- layout

def failure_ui(state: dict) -> gr.Blocks:
    """Retryable failure state: explain what failed and offer a retry with progress."""
    import os
    import threading

    from weather_lab.bootstrap import BootstrapError, bootstrap, summarize_failure

    def retry(progress=gr.Progress()):
        try:
            bootstrap(progress=lambda f, m: progress(f, desc=m))
        except BootstrapError:
            from weather_lab.bootstrap import _state

            return summarize_failure(_state()) + f" (attempt {_state().get('attempts')})"
        threading.Timer(1.0, lambda: os.execv(sys.executable, [sys.executable] + sys.argv)).start()
        return "Setup complete. Restarting the app: reload this page in a few seconds."

    with gr.Blocks(title="Weather-lab: setup") as ui:
        gr.Markdown("# Weather-lab\nThe app fetches its data snapshot and model weights at startup.")
        status = gr.Markdown(summarize_failure(state))
        gr.Button("Retry setup", variant="primary").click(retry, None, status, concurrency_limit=1)
    return ui


def fallback_notice(state: dict | None) -> str | None:
    """One-line notice when bootstrap installed an earlier snapshot because the pinned one failed."""
    snap = (state or {}).get("snapshot") or {}
    if not snap.get("fallback"):
        return None
    err = str(snap.get("primary_error") or "unknown error")
    low = err.lower()
    reason = ("file not found in the storage bucket" if "not found" in low else
              "checksum mismatch" if "checksum" in low else
              (err[:117] + "...") if len(err) > 120 else err)
    return (f"Showing data snapshot `{snap.get('id')}` (fallback): pinned snapshot `{snap.get('primary_id')}` "
            f"could not be fetched — {reason}. Results shown are from the earlier snapshot.")


def build() -> gr.Blocks:
    with gr.Blocks(title="Weather-lab") as demo:
        if DEFAULT_SUITE is None:
            gr.Markdown("# Weather-lab\nNo saved results found. Run `uv run weather-lab bootstrap`.")
            return demo
        gr.Markdown(f"# Weather-lab\n{INTRO}", elem_id="hero")
        if notice := fallback_notice(BOOT):
            gr.Markdown(f"**Note:** {notice}", elem_classes="note", elem_id="snapshot-fallback")
        with gr.Row(equal_height=True):
            exp = gr.Dropdown(experiment_choices(), value=DEFAULT_SUITE, label="Experiment", scale=2, min_width=240)
            var = gr.Dropdown(variable_choices(DEFAULT_SUITE), value="t", label="Weather variable", scale=1,
                              min_width=180)
        scope = gr.HTML(scope_html(DEFAULT_SUITE), elem_id="scope")
        pref = gr.State({})
        extra = gr.State(EMPTY_EXTRA)
        with gr.Tabs(selected="overview"):
            # ---------------- overview
            with gr.Tab("Overview", id="overview"):
                with gr.Row(equal_height=True):
                    station = gr.Dropdown(label="Station", choices=[], scale=1, min_width=180)
                    ckpt = gr.Radio(list(CHECKPOINTS), value=CHECKPOINTS[0], label="Timestral model", scale=1,
                                    min_width=150, visible=has_checkpoints(DEFAULT_SUITE))
                    setup = gr.Dropdown(setup_choices(DEFAULT_SUITE), label="Timestral + AROME input", scale=2,
                                        value=(EXPERIMENTS[DEFAULT_SUITE].get("matrix") or {}).get("main"),
                                        min_width=230, visible=has_setups(DEFAULT_SUITE))
                    date = gr.Dropdown(label="Forecast issued", choices=[], scale=2, min_width=220)
                fplot = gr.Plot(show_label=False)
                tiles = gr.HTML()
                caption = gr.Markdown(elem_classes="note")
                with gr.Row():
                    with gr.Column(scale=2, min_width=280):
                        this_tab = gr.HTML()
                    with gr.Column(scale=3, min_width=280):
                        sys_desc = gr.Markdown(elem_classes="note")
                with gr.Accordion("Advanced: systems, ranges, history, exploratory run", open=False):
                    with gr.Row():
                        with gr.Column(scale=2, min_width=260):
                            systems = gr.CheckboxGroup(label="Systems shown", choices=[])
                            band = gr.Dropdown(label="Shaded 80% / 90% range for", choices=[], value=None)
                            history = gr.Slider(12, 240, value=48, step=12, label="Observed history shown (hours)")
                            err_mode = gr.Radio(ERROR_MODES, value="signed", label="Error panel under the chart")
                            show_old = gr.Checkbox(False, label="Show superseded experiment versions")
                            dl_case = gr.DownloadButton("Download this forecast (CSV)", size="sm")
                        with gr.Column(scale=1, min_width=220):
                            smap = gr.Plot(show_label=False)
                    gr.Markdown("**Exploratory run** — Timestral once on this forecast with custom settings; shown "
                                "on the chart, never saved or scored.", elem_classes="note")
                    with gr.Row():
                        e_size = gr.Dropdown(["Timestral-5M", "Timestral-28M", "Timestral-209M"],
                                             value="Timestral-28M", label="Checkpoint", min_width=160)
                        e_ctx = gr.Slider(7, 30, value=14, step=1, label="History read (days)", min_width=160)
                        e_cal = gr.Checkbox(value=True, label="Time-of-day / season inputs")
                        e_setup = gr.Dropdown(list(EXPLORE_SETUPS), value="No AROME input", label="AROME input",
                                              min_width=240)
                        e_dl = gr.Checkbox(value=False, label="Allow weight download")
                    e_btn = gr.Button("Run exploratory forecast", size="sm")
                    e_msg = gr.Markdown(elem_classes="note")

            # ---------------- benchmark
            with gr.Tab("Benchmark", id="benchmark"):
                with gr.Row(equal_height=True):
                    metric = gr.Dropdown(METRIC_CHOICES, value="mae", label="Score", min_width=180)
                    lead = gr.Dropdown(LEADS, value="all", label="Lead time (bars)", min_width=160)
                    b_station = gr.Dropdown(label="Station", choices=[], min_width=180)
                    lead_mode = gr.Radio(LEAD_VIEWS, value="at", label="Lead-time chart", min_width=260)
                    split = gr.Radio([], label="Period", visible=False, min_width=200)
                b_finding = gr.HTML()
                b_summary = gr.Markdown(elem_classes="note")
                bars = gr.Plot(show_label=False)
                lines = gr.Plot(show_label=False)
                b_horizon = gr.Markdown(elem_classes="note")
                with gr.Accordion("Experiment matrix: input setup × model", open=False):
                    matrix = gr.HTML()
                with gr.Accordion("Detailed tables and downloads", open=False):
                    point_tab = gr.Dataframe(interactive=False, label="Single-value accuracy", wrap=False)
                    prob_tab = gr.Dataframe(interactive=False, label="Probabilistic accuracy", wrap=False)
                    with gr.Row():
                        dl_point = gr.DownloadButton("Point scores (CSV)", size="sm")
                        dl_prob = gr.DownloadButton("Probabilistic scores (CSV)", size="sm")
                        dl_manifest = gr.DownloadButton("Case manifest (JSON)", size="sm")
                with gr.Accordion("Advanced: filters, availability, uncertainty, timings", open=False):
                    with gr.Row():
                        b_season = gr.Dropdown(label="Season (bars)", choices=[], min_width=180)
                    statuses = gr.CheckboxGroup(STATUSES, value=STATUSES, label="Fitting / adaptation status")
                    avail_tab = gr.Dataframe(interactive=False, label="Availability", wrap=False)
                    paired_tab = gr.Dataframe(interactive=False, label="Paired differences (block bootstrap)",
                                              wrap=False)
                    hw = gr.Markdown(elem_classes="note")
                    rt_tab = gr.Dataframe(interactive=False, label="CPU cost", wrap=False)

            # ---------------- methods
            with gr.Tab("Methods", id="methods"):
                gr.Markdown(methods_md())
                with gr.Accordion("Runs, versions and integrity", open=False):
                    gr.Markdown(integrity_md())
                with gr.Accordion("Caches and model files", open=False):
                    cache_box = gr.Markdown()
                    cache_btn = gr.Button("Refresh status", size="sm")
                with gr.Accordion("Full experiment protocol", open=False):
                    gr.Markdown(protocol_md())

        # ---------------- wiring
        view = gr.State({})
        ov_inputs = [exp, date, var, systems, band, history, extra, err_mode]
        ov_outputs = [fplot, caption, this_tab, sys_desc, dl_case, smap]
        sys_inputs = [exp, date, var, pref, band, extra, ckpt, setup]
        b_inputs = [exp, var, metric, lead, split, b_station, b_season, statuses, ckpt, setup, lead_mode]
        b_outputs = [b_finding, b_summary, bars, lines, b_horizon, point_tab, prob_tab]
        exp_outputs = [var, station, date, scope, ckpt, setup]
        tile_inputs = [exp, var, ckpt, setup]
        remember = lambda size, stp: {"size": size, "setup": stp}  # noqa: E731

        def refresh_overview(event):
            return (event.then(keep_extra_for_case, [date, extra], extra)
                    .then(systems_update, sys_inputs, [systems, band])
                    .then(overview, ov_inputs, ov_outputs)
                    .then(tiles_html, tile_inputs, tiles))

        def refresh_benchmark(event):
            return (event.then(benchmark, b_inputs, b_outputs)
                    .then(matrix_html, [exp, var], matrix)
                    .then(advanced_tables, [exp, var, split], [avail_tab, paired_tab, rt_tab, hw])
                    .then(downloads, exp, [dl_point, dl_prob, dl_manifest]))

        on_exp = exp.input(on_experiment, [exp, var], exp_outputs).then(remember, [ckpt, setup], view).then(
            bench_controls, exp, [split, b_station, b_season])
        refresh_benchmark(refresh_overview(on_exp))
        refresh_benchmark(var.input(systems_update, sys_inputs, [systems, band])
                          .then(overview, ov_inputs, ov_outputs).then(tiles_html, tile_inputs, tiles))
        refresh_overview(station.input(on_station, [exp, station], date))
        date.input(keep_extra_for_case, [date, extra], extra).then(
            systems_update, sys_inputs, [systems, band]).then(overview, ov_inputs, ov_outputs)
        for c in (ckpt, setup):
            refresh_benchmark(c.input(switch_view, [exp, ckpt, setup, view, pref], [pref, view])
                              .then(systems_update, sys_inputs, [systems, band]).then(overview, ov_inputs, ov_outputs)
                              .then(tiles_html, tile_inputs, tiles))
        systems.input(update_pref, [exp, date, var, systems, pref, extra, ckpt, setup], pref).then(
            systems_update, sys_inputs, [systems, band]).then(overview, ov_inputs, ov_outputs)
        for c in (band, history, err_mode):
            c.input(overview, ov_inputs, ov_outputs)
        show_old.input(lambda s, cur: gr.update(choices=experiment_choices(s), value=cur), [show_old, exp], exp)
        e_btn.click(run_exploratory, [exp, date, e_size, e_ctx, e_cal, e_setup, e_dl, pref, extra, ckpt, setup],
                    [extra, pref, e_msg], concurrency_limit=1).then(
            systems_update, sys_inputs, [systems, band]).then(overview, ov_inputs, ov_outputs)
        for c in (metric, lead, split, b_station, b_season, statuses, lead_mode):
            c.input(benchmark, b_inputs, b_outputs)
        split.input(advanced_tables, [exp, var, split], [avail_tab, paired_tab, rt_tab, hw])
        cache_btn.click(cache_md, None, cache_box)

        refresh_benchmark(refresh_overview(
            demo.load(on_experiment, [exp, var], exp_outputs).then(remember, [ckpt, setup], view).then(
                bench_controls, exp, [split, b_station, b_season]))).then(cache_md, None, cache_box)
    return demo


def init(run_bootstrap: bool = True) -> gr.Blocks:
    """Fetch/verify artifacts (unless disabled), load saved runs and build the UI."""
    global BOOT, RUNS, DEFAULT_SUITE
    BOOT = startup_bootstrap() if run_bootstrap else None
    RUNS = available_runs()
    DEFAULT_SUITE = DEFAULT_EXPERIMENT if DEFAULT_EXPERIMENT in RUNS else next(iter(RUNS), None)
    return failure_ui(BOOT) if (BOOT or {}).get("status") == "failed" and not RUNS else build()


def main() -> None:
    demo = init()
    demo.queue(default_concurrency_limit=4).launch(
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="stone",
                             font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]),
        css=CSS)


if __name__ == "__main__":
    main()
