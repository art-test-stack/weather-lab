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
import numpy as np
import pandas as pd

from weather_lab.common import ROOT, VARIABLES
from weather_lab_app import plots
from weather_lab_app.labels import (CHECKPOINTS, DEFAULT_EXPERIMENT, DIFFERENT_INPUT, EXPERIMENTS, MODELS,
                                    SEASON_NAMES, SETUP_NAMES, SPLIT_NAMES, model_description, model_name,
                                    model_style, unit, variable_name, with_checkpoint)
from weather_lab.forecast import qcol
from weather_lab_app.results import (arome_history_text, available_runs, case_forecast, forecast_csv, headline,
                                     headline_to, input_history, station_label, training_status)


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
STATUSES = ["pretrained, no local fine-tuning", "fitted on fitting period", "external NWP (issued run)",
            "post-processing of AROME (recent errors)", "no fitting"]
EMPTY_EXTRA = {"case": None, "runs": {}, "settings": {}, "active": None}
DEFAULT_STATION = "07149"            # Paris-Orly, by stable station id
DEFAULT_SIZE = CHECKPOINTS[-1]       # the largest integrated Timestral checkpoint (209M)
EXPLORE_MAX_HOURS = 120              # app limit for exploratory runs (Timestral itself sets no horizon limit)
CONTEXT_DAYS = (7, 90)               # range of the Observed context control
AROME_MODES = {"none": None, "archived": "archived", "substobs": "substituted_observations", "futureonly": "missing"}

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
#applied .applied {font-size:.84rem; color:#3d3c39;}
#applied .badge {display:inline-block; padding:1px 8px; border-radius:9px; font-weight:600; font-size:.78rem;
                 margin-right:4px;}
#applied .badge.saved {background:#e8f0fb; color:#184f95;}
#applied .badge.expl {background:#efe9fb; color:#4b2a91;}
#forecast-panel {padding:4px 6px; border-radius:10px;}
#applied .pending {font-size:.84rem; color:#7a4b00; background:#fff4e0; border-radius:6px; padding:3px 8px;
                   margin-top:3px;}
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
    return [(SETUP_NAMES[k][0], k) for k in mx.get("setups", {})]


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


def default_station(suite: str) -> str:
    """Paris-Orly (by station id) when the experiment has it, else its first UI example / first station."""
    stations = [s for _, s in station_choices(suite)]
    if DEFAULT_STATION in stations:
        return DEFAULT_STATION
    first_example = (RUNS[suite].manifest.get("ui_examples") or [""])[0]
    return first_example.split("/")[1] if first_example.count("/") == 2 else stations[0]


def on_experiment(suite: str, variable: str | None, station: str | None = None, size: str | None = None,
                  setup: str | None = None):
    """Experiment changed: keep the user's variable, station, model and AROME input where the new experiment
    has them; otherwise defaults. Context and horizon take the new run's recorded values."""
    vchoices = variable_choices(suite)
    variable = variable if variable in [v for _, v in vchoices] else "t"
    stations = station_choices(suite)
    station = station if station in [s for _, s in stations] else default_station(suite)
    choices = setup_choices(suite)
    setup = setup if setup in [k for _, k in choices] else e_main(suite)
    size = size if size in CHECKPOINTS else DEFAULT_SIZE
    return (gr.update(choices=vchoices, value=variable), gr.update(choices=vchoices, value=variable),
            gr.update(choices=stations, value=station),
            gr.update(choices=date_choices(suite, station), value=default_case(suite, station)),
            scope_html(suite), gr.update(visible=has_checkpoints(suite), value=size),
            gr.update(visible=has_setups(suite), choices=choices, value=setup),
            gr.update(value=default_context(suite, size, setup)),
            gr.update(value=RUNS[suite].horizon_hours))


def on_station(suite: str, station: str):
    return gr.update(choices=date_choices(suite, station), value=default_case(suite, station))


def tiles_html(suite: str, variable: str, size: str | None = None, setup: str | None = None,
               horizon: int | None = None) -> str:
    """Benchmark aggregate cards for the selected experiment, checkpoint and input setup (saved score tables).

    With `horizon` below the run's inference horizon they are recomputed from the saved predictions on leads
    <= horizon and labelled as an interactive slice; the frozen full-horizon scores stay in the saved tables.
    """
    run, e = RUNS[suite], experiment_view(suite, size, setup)
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
    what = (f"<b>Benchmark aggregate, interactive slice</b>: mean absolute error over {n:,} forecast–observation "
            f"pairs with lead ≤ {int(horizon)} h of the saved {run.horizon_hours} h forecasts, all stations"
            f"{split_txt}, recomputed from saved predictions. Frozen full-horizon scores: Benchmark tab."
            if sliced else
            f"<b>Benchmark aggregate</b> (frozen): mean absolute error over {n:,} forecast–observation pairs, all "
            f"stations and lead times{split_txt}; lower is better.")
    return f"<div class='tiles'>{''.join(out)}{not_run}</div><div class='note'>{what}</div>"


def saved_settings(suite: str, size: str | None, setup: str | None) -> dict:
    """Settings of the saved run behind the current view, from its provenance (run.json params, manifest)."""
    run, e = RUNS[suite], experiment_view(suite, size, setup)
    mid = e["band"] or next((m for m in e["headline"] if m.startswith("timestral") and m in run.models), None)
    p = run.params(mid) if mid else {}
    return {"model": mid, "context_days": p.get("context_days"), "calendar": p.get("calendar"),
            "horizon_h": run.horizon_hours, "missing": e["missing"]}


def default_context(suite: str, size: str | None, setup: str | None) -> int:
    """Context-control default: the active saved run's recorded context (never the validation choice of
    another run); the suite protocol's value when the view's system was not run."""
    days = saved_settings(suite, size, setup)["context_days"]
    return int(days or RUNS[suite].manifest["protocol"]["forecast"].get("context_days", 30))


def context_md(days, suite: str) -> str:
    step = RUNS[suite].step_hours
    return (f"= **{int(days) * 24} hours = {int(days) * 24 // step} {'three-hourly' if step == 3 else ''} steps**. "
            "The history supplied to inference. A saved replay's context comes from its run and is read-only; a "
            "different value prepares an exploratory run.")


def horizon_md(hours, suite: str) -> str:
    run = RUNS[suite]
    hours, step = int(hours), run.step_hours
    saved = run.horizon_hours
    where = (f"display/scoring slice of the saved {saved} h forecasts (no new inference)" if hours <= saved else
             f"beyond the saved {saved} h: needs **Run forecast** (exploratory). AROME runs end {saved} h after "
             "these forecast times, so AROME inputs allow at most that; Timestral alone up to "
             f"{EXPLORE_MAX_HOURS} h (app limit)")
    return f"= **{hours // step} forecast steps** of {step} h; {where}."


def settings_text(st: dict, step: int = 3) -> str:
    setup = next((k for k, v in AROME_MODES.items() if v == st.get("arome_mode")), "none")
    return (f"Timestral {st.get('checkpoint', '?').split('-')[-1]} · {SETUP_NAMES[setup][0]} · context "
            f"{st.get('context_days')} d ({st.get('context_steps')} steps) · inference horizon "
            f"{st.get('horizon_steps', 0) * st.get('step_hours', step)} h"
            + ("" if st.get("calendar", True) else " · no time inputs"))


# --------------------------------------------------------------------------- forecast configuration & execution
#
# One configuration (draft = the panel controls) and one applied result (`applied` State). Everything the
# Overview renders — curves, overlays, interval source, error panel, cards, captions, badge — derives from
# `applied` only. A draft edit is applied at once when it resolves without inference (a saved benchmark
# forecast, or a cached exploratory one); otherwise it is pending until **Run forecast**, which takes the
# same resolution path and runs inference only on a miss. A run result is applied only if the draft still
# equals the configuration it was computed for (obsolete responses are ignored).

CALENDAR = True          # every saved Timestral system uses calendar inputs; the panel does not change it


def _setup(suite: str, setup: str | None) -> str | None:
    return setup if has_setups(suite) else None


def primary_saved(suite: str, size: str | None, setup: str | None) -> str | None:
    """The saved Timestral system a (size, input setup) selects in this experiment, or None."""
    run, e = RUNS[suite], experiment_view(suite, size, setup)
    mid = e["band"] or next((m for m in e["headline"] if m.startswith("timestral") and m in run.models), None)
    if has_setups(suite):
        mid = matrix_id(suite, size or DEFAULT_SIZE, setup or "none")
    return mid


def _size_of(mid: str | None) -> str | None:
    for s in CHECKPOINTS:
        if mid and mid.startswith(f"timestral-{s.lower()}-"):
            return s
    return None


def draft(suite, case_id, size, setup, ctx, horizon) -> dict:
    """The configuration the panel describes. Experiments without a size selector use their system's size."""
    if not has_checkpoints(suite):
        size = _size_of(primary_saved(suite, size, setup)) or size
    return {"suite": suite, "case": case_id, "size": size, "setup": _setup(suite, setup), "ctx": int(ctx),
            "horizon": int(horizon)}


_CFG_KEYS = ("suite", "case", "size", "setup", "ctx", "horizon")


def _same(a: dict | None, b: dict | None) -> bool:
    return bool(a) and bool(b) and all(a.get(k) == b.get(k) for k in _CFG_KEYS)


def horizon_limit(suite: str, case_id: str, setup: str | None) -> tuple[int, str]:
    """(largest supported horizon in hours, why) for this forecast origin and input setup."""
    run, mode = RUNS[suite], AROME_MODES.get(_setup(suite, setup))
    if not mode:
        return EXPLORE_MAX_HOURS, (f"Timestral alone: up to {EXPLORE_MAX_HOURS} h (app limit); saved forecasts "
                                   f"cover {run.horizon_hours} h")
    case = run.cases().set_index("case_id").loc[case_id]
    ext = run.arome_extract()
    if ext is None or not isinstance(case.get("arome_run"), str):
        return 0, "no saved AROME run for this forecast, so AROME inputs are unavailable"
    from weather_lab.forecast import arome_coverage_steps

    origin, step = pd.Timestamp(case["origin"]), run.step_hours
    sub = ext[(ext["run"] == pd.Timestamp(case["arome_run"])) & (ext["station"] == case["station"])]
    piv = sub.pivot_table(index="valid_time", columns="variable", values="value")
    times = [origin + pd.Timedelta(hours=step * (k + 1)) for k in range(EXPLORE_MAX_HOURS // step)]
    rows = piv.reindex(times).T.to_numpy(dtype=float) if len(piv) else np.full((1, len(times)), np.nan)
    hours = arome_coverage_steps(rows) * step
    return hours, (f"AROME run {case['arome_run'][:16].replace('T', ' ')} UTC ends {hours} h after this forecast "
                   "time; AROME inputs are never extended. 'No AROME input' allows up to "
                   f"{EXPLORE_MAX_HOURS} h")


def horizon_control(suite, case_id, setup, horizon):
    """Constrain the horizon control to the supported maximum (value clamped) and explain the limit."""
    limit, why = horizon_limit(suite, case_id, setup)
    value = min(int(horizon), limit) if limit else int(horizon)
    step = RUNS[suite].step_hours
    info = (f"= **{value // step} steps** of {step} h. Maximum here: **{limit} h** — {why}." if limit else
            f"⚠️ {why}.")
    return gr.update(maximum=max(limit, step), value=max(value, step)), info, gr.update(interactive=limit > 0)


def saved_match(d: dict) -> str | None:
    """The saved benchmark system whose recorded settings equal the draft (horizon may be a slice)."""
    run = RUNS[d["suite"]]
    mid = primary_saved(d["suite"], d["size"], d["setup"])
    if not mid or mid not in run.models or d["case"] not in set(run.cases()["case_id"]):
        return None
    p = run.params(mid)
    ok = (p.get("context_days") == d["ctx"] and bool(p.get("calendar", CALENDAR)) == CALENDAR
          and d["horizon"] <= run.horizon_hours and d["horizon"] % run.step_hours == 0)
    return mid if ok else None


def _explore_inputs(d: dict):
    run = RUNS[d["suite"]]
    _, _, case = case_forecast(run, d["case"], "t")
    obs = run.full_observations()
    mode = AROME_MODES.get(d["setup"])
    return obs[obs["station"] == case["station"]], case, mode, (run.arome_extract() if mode else None)


def _device() -> str:
    return "cuda" if zerogpu.on_zerogpu() else "cpu"


def make_applied(d: dict, source: str, primary: str, prev: dict | None = None, frame=None, info=None) -> dict:
    """The applied result: configuration, source, primary system and the exploratory runs of this case."""
    same_case = bool(prev) and prev.get("suite") == d["suite"] and prev.get("case") == d["case"]
    runs = dict(prev.get("exploratory", {})) if same_case else {}
    settings = {}
    if frame is not None:
        runs[primary] = frame
        settings = {**info["settings"], "cache_hit": info["cache_hit"], "key": info["key"]}
    run = RUNS[d["suite"]]
    shown = min(d["horizon"], int(frame["lead_h"].max()) if frame is not None else run.horizon_hours)
    return {**d, "source": source, "primary": primary, "settings": settings, "exploratory": runs,
            "settings_by": {**(prev.get("settings_by", {}) if same_case else {}),
                            **({primary: settings} if settings else {})}, "shown_h": shown}


def resolve(d: dict, prev: dict | None = None, lookup_cache: bool = True) -> dict | None:
    """Saved benchmark forecast, else cached exploratory forecast, else None (needs inference)."""
    mid = saved_match(d)
    if mid:
        return make_applied(d, "saved", mid, prev)
    if not lookup_cache:
        return None
    from weather_lab.forecast import exploratory_forecast

    try:
        obs, case, mode, arome_df = _explore_inputs(d)
        step = RUNS[d["suite"]].step_hours
        frame, info = exploratory_forecast(obs, case, ["t", "pmer", "u", "v"], f"Timestral-{d['size']}", d["ctx"],
                                           CALENDAR, arome_mode=mode, arome_df=arome_df, device=_device(),
                                           lookup_only=True, horizon_steps=d["horizon"] // step)
    except (ValueError, KeyError):
        return None
    return make_applied(d, "exploratory", frame["model"].iloc[0], prev, frame, info) if frame is not None else None


def on_draft(suite, case_id, size, setup, ctx, horizon, applied):
    """A draft edit: apply it at once when it resolves without inference, else leave it pending."""
    d = draft(suite, case_id, size, setup, ctx, horizon)
    if _same(d, applied):
        return applied
    limit, _ = horizon_limit(suite, case_id, setup)
    if d["horizon"] > limit:
        return applied
    return resolve(d, applied) or applied


def run_forecast(suite, case_id, size, setup, ctx, horizon, applied):
    """Run forecast: the same resolution path; inference only on a miss. Returns ({draft, applied} | None, msg)."""
    from weather_lab import predcache
    from weather_lab.forecast import ensure_checkpoint, exploratory_forecast

    d = draft(suite, case_id, size, setup, ctx, horizon)
    keep = " The displayed result is unchanged."
    limit, why = horizon_limit(suite, case_id, setup)
    step = RUNS[suite].step_hours
    if not step <= d["horizon"] <= limit or d["horizon"] % step:
        return None, f"⚠️ Horizon {d['horizon']} h is not supported here: {why}.{keep}"
    hit = resolve(d, applied)
    if hit:
        src = "saved benchmark forecast" if hit["source"] == "saved" else "cached exploratory forecast"
        return {"draft": d, "applied": hit}, f"Applied the {src} for this configuration (no model run)."
    name = f"Timestral-{d['size']}"
    try:
        obs, case, mode, arome_df = _explore_inputs(d)
        ensure_checkpoint(name, False)       # weights must be cached (Space: preloaded); a cold fetch is not run here
        args = (obs, case, name, d["ctx"], CALENDAR, mode, arome_df, False, d["horizon"] // step)
        frame, info = _exploratory_on_device(*args)
    except (PermissionError, ValueError) as e:
        return None, f"⚠️ {e}.{keep}"
    except Exception as e:  # noqa: BLE001 - e.g. ZeroGPU quota/queue errors raised from the GPU worker
        return None, f"⚠️ Run forecast failed: {type(e).__name__}: {e}.{keep}"
    new = make_applied(d, "exploratory", frame["model"].iloc[0], applied, frame, info)
    return {"draft": d, "applied": new}, (f"Computed {settings_text(new['settings'])} and cached it "
                                          f"({predcache.location()[1]}; key {info['key']}).")


def apply_if_current(result, suite, case_id, size, setup, ctx, horizon, applied, msg):
    """Apply a run result atomically, unless the configuration changed while it was running."""
    if not result:
        return applied, msg
    if not _same(result["draft"], draft(suite, case_id, size, setup, ctx, horizon)):
        return applied, ("A result for an earlier configuration arrived after the settings changed and was not "
                         "applied (it is cached: Run forecast with those settings shows it at once).")
    return result["applied"], msg


def applied_text(a: dict) -> str:
    run = RUNS[a["suite"]]
    if a["source"] == "saved":
        p = run.params(a["primary"])
        ctx = p.get("context_days")
        return (f"{model_name(a['primary'])} · context {ctx} d ({int(ctx) * 24 // run.step_hours} steps) · shown to "
                f"{a['shown_h']} h of the saved {run.horizon_hours} h forecast")
    return f"{settings_text(a['settings'])} · shown to {a['shown_h']} h"


def status_html(applied, suite, case_id, size, setup, ctx, horizon) -> str:
    """Result badge (what is displayed, from `applied`) and the pending draft differences."""
    if not applied:
        return ""
    a = applied
    badge = ("<span class='badge saved'>Saved benchmark forecast</span>" if a["source"] == "saved" else
             f"<span class='badge expl'>Exploratory · {'cached' if a['settings'].get('cache_hit') else 'computed'}"
             "</span>")
    run = RUNS[a["suite"]]
    rid = f" <code>{run.run['run_id'][:40]}</code>" if a["source"] == "saved" else " outside the frozen benchmark"
    d = draft(suite, case_id, size, setup, ctx, horizon)
    diff = []
    if d["suite"] != a["suite"] or d["case"] != a["case"]:
        diff.append("station / forecast time")
    if d["size"] != a["size"]:
        diff.append(f"model {d['size']}")
    if d["setup"] != a["setup"]:
        diff.append(f"input: {SETUP_NAMES.get(d['setup'] or 'none', (d['setup'],))[0]}")
    if d["ctx"] != a["ctx"]:
        diff.append(f"context {d['ctx']} d")
    if d["horizon"] != a["horizon"]:
        diff.append(f"horizon {d['horizon']} h")
    out = f"<div class='applied'>{badge}{rid}: {applied_text(a)}</div>"
    if diff:
        out += ("<div class='pending'>Pending, not applied: " + ", ".join(diff) + ". The chart, cards and "
                "captions still show the result above. Press <b>Run forecast</b>.</div>")
    return out


def _view_systems(a: dict, variable: str) -> tuple[list[str], list[str], list[str]]:
    """(choices, defaults, probabilistic) for the applied result's view."""
    run, e = RUNS[a["suite"]], experiment_view(a["suite"], a["size"], a["setup"] or e_main(a["suite"]))
    saved = _saved_models(run, variable)
    exploratory = list(a.get("exploratory", {}))
    defaults = [m for m in e["default_systems"] if m in saved]
    if a["source"] == "saved" and a["primary"] in saved and a["primary"] not in defaults:
        defaults.append(a["primary"])
    if a["source"] == "exploratory":
        defaults.append(a["primary"])
    preds = run.predictions()
    prob = {m for m in saved if (preds.loc[preds["model"] == m, "output_type"] == "quantiles").any()} | set(exploratory)
    return saved + exploratory, defaults, [m for m in saved + exploratory if m in prob]


def e_main(suite: str) -> str | None:
    return (EXPERIMENTS[suite].get("matrix") or {}).get("main")


def update_pref(applied, variable, selected, pref_by):
    """Overlays are stored as additions/removals relative to the applied view's defaults, so they survive
    model / input changes without mixing configurations."""
    if not applied:
        return pref_by
    _, defaults, _ = _view_systems(applied, variable)
    sel = set(selected or [])
    pref_by = dict(pref_by or {})
    pref_by[applied["suite"]] = {"added": sorted(sel - set(defaults)), "removed": sorted(set(defaults) - sel)}
    return pref_by


def select_systems(a: dict, variable: str, pref_by: dict | None, band: str | None):
    """(choices, shown overlays, probabilistic shown, interval source) for the applied result. The applied
    primary system is always the interval source when it is probabilistic and shown."""
    choices, defaults, prob = _view_systems(a, variable)
    p = (pref_by or {}).get(a["suite"]) or {}
    value = [m for m in choices if (m in defaults and m not in p.get("removed", [])) or m in p.get("added", [])]
    shown_prob = [m for m in value if m in prob]
    if a["primary"] in shown_prob and (a["source"] == "exploratory" or band in (None, "") or band not in shown_prob):
        band = a["primary"]
    elif band not in shown_prob and band != "":
        band = shown_prob[0] if shown_prob else ""
    return choices, value, shown_prob, band


def render(applied, variable, pref_by, band, error_mode):
    """Every Overview output from the applied result, in one step (no partial updates)."""
    a = applied
    choices, value, shown_prob, band = select_systems(a, variable, pref_by, band)
    fig, caption, this, desc, csv, smap = overview(a, variable, value, band, error_mode)
    return (gr.update(choices=[(model_name(m), m) for m in choices], value=value),
            gr.update(choices=[("none", "")] + [(model_name(m), m) for m in shown_prob], value=band),
            fig, caption, this, desc, csv, smap, cards_html(a, variable))


def _window(rows: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return rows[rows["lead_h"] <= int(horizon)]


def forecast_scores(rows: pd.DataFrame, obs: pd.DataFrame, systems: list[str]) -> tuple[pd.DataFrame, int]:
    """Per system on THIS forecast: leads covered, and on the leads where every listed system has a point
    forecast and the observation exists (shared window): MAE, 80% coverage and width."""
    r = rows[rows["model"].isin(systems)].merge(obs.rename(columns={"time": "valid_time"}), on="valid_time",
                                                how="left")
    covers = r.groupby("model")["lead_h"].max()
    ok = r.dropna(subset=["point", "observed"])
    shared = set.intersection(*(set(g["lead_h"]) for _, g in ok.groupby("model"))) if len(ok) else set()
    have = set(ok["model"])
    shared = shared if have == set(covers.index) else set()
    s = ok[ok["lead_h"].isin(shared)].copy()
    s["abs_err"] = (s["point"] - s["observed"]).abs()
    if qcol(0.1) in s:
        s["in80"] = ((s["observed"] >= s[qcol(0.1)]) & (s["observed"] <= s[qcol(0.9)])).where(s[qcol(0.1)].notna())
        s["width80"] = s[qcol(0.9)] - s[qcol(0.1)]
    out = pd.DataFrame({"covers_h": covers})
    if len(s):
        agg = s.groupby("model").agg(mae=("abs_err", "mean"), n=("abs_err", "size"),
                                     **({"cov80": ("in80", "mean"), "width80": ("width80", "mean")}
                                        if "in80" in s else {}))
        out = out.join(agg)
    return out.reindex([m for m in systems if m in out.index]), len(shared)


def this_forecast_html(scores: pd.DataFrame, n_shared: int, variable: str) -> str:
    u = unit(variable)
    observable = n_shared > 0
    cell = lambda x, f: f(x) if observable and pd.notna(x) else "Not yet observable" if not observable else "–"  # noqa: E731
    rows = "".join(
        f"<tr><td><span class='sw' style='background:{model_style(m)[0]}'></span>{model_name(m)}</td>"
        f"<td>{int(r.covers_h)} h</td><td>{cell(r.get('mae'), lambda v: f'{v:.2f} {u}')}</td>"
        f"<td>{cell(r.get('cov80'), lambda v: f'{v:.0%}')}</td></tr>" for m, r in scores.iterrows())
    return ("<table class='mini'><thead><tr><th>System</th><th>Shown to</th><th>Mean abs. error</th>"
            f"<th>80% coverage</th></tr></thead><tbody>{rows}</tbody></table><div class='note'>This forecast only, "
            f"on the {n_shared} lead(s) where every listed system has a forecast and the observation exists"
            + ("" if observable else ": none observed yet") + ".</div>")


def cards_html(applied, variable) -> str:
    """Primary cards: the applied exploratory forecast (labelled), or the benchmark aggregate of the applied
    saved view over its displayed slice."""
    a = applied
    if a["source"] == "saved":
        return tiles_html(a["suite"], variable, a["size"], a["setup"] or e_main(a["suite"]), a["shown_h"])
    run, mid, st = RUNS[a["suite"]], a["primary"], a["settings"]
    frame = a["exploratory"][mid]
    rows, _, case = case_forecast(run, a["case"], variable)
    obs = _station_obs(run, case["station"], variable)
    f = _window(frame[frame["variable"] == variable], a["shown_h"])
    u = unit(variable)
    v = f.merge(obs.rename(columns={"time": "valid_time"}), on="valid_time", how="left")
    seen = v.dropna(subset=["observed"])
    n_obs, n_leads = len(seen), len(f)
    head = (f"<div class='note'><b>Current forecast · Exploratory</b> — {settings_text(st)}; shown to "
            f"{a['shown_h']} h; {n_obs} of {n_leads} leads observed so far.</div>")
    color = model_style(mid)[0]
    nyo = "<div class='v'>Not yet observable</div>"
    mae = f"<div class='v'>{(seen['point'] - seen['observed']).abs().mean():.2f} {u}</div>" if n_obs else nyo
    cards = [f"<div class='tile ref' title='{mid}'>{mae}<div class='n'><span class='sw' style='background:{color}'>"
             f"</span>Mean absolute error · this forecast</div><div class='d'>{n_obs} observed lead(s)</div></div>"]
    if qcol(0.1) in f and f[qcol(0.1)].notna().any():
        width = (f[qcol(0.9)] - f[qcol(0.1)]).mean()
        inside = ((seen["observed"] >= seen[qcol(0.1)]) & (seen["observed"] <= seen[qcol(0.9)])).mean()
        cards.append(f"<div class='tile'>{f'<div class=v>{inside:.0%}</div>' if n_obs else nyo}<div class='n'>"
                     f"inside its 80% range</div><div class='d'>nominal 80%; {n_obs} observation(s) — too few to "
                     "judge calibration</div></div>")
        cards.append(f"<div class='tile'><div class='v'>{width:.1f} {u}</div><div class='n'>mean 80% range width"
                     f"</div><div class='d'>over its {n_leads} lead(s)</div></div>")
    others = [m for m in ["arome-raw", primary_saved(a["suite"], a["size"], a["setup"])]
              if m and m in set(rows["model"])]
    sc, n_shared = forecast_scores(pd.concat([_window(rows, a["shown_h"]), f], ignore_index=True), obs,
                                   [mid] + others)
    for m in others:
        val = sc.loc[m, "mae"] if "mae" in sc and m in sc.index else None
        cards.append(f"<div class='tile' title='{m}'><div class='v'>"
                     + (f"{val:.2f} {u}" if val is not None and pd.notna(val) else "Not yet observable")
                     + f"</div><div class='n'><span class='sw' style='background:{model_style(m)[0]}'></span>"
                     f"{model_name(m)}</div><div class='d'>saved forecast, on the {n_shared} lead(s) both cover and "
                     "observed" + (f"; the exploratory forecast scores {sc.loc[mid, 'mae']:.2f} {u} there"
                                   if "mae" in sc and pd.notna(sc.loc[mid, "mae"]) else "") + "</div></div>")
    return (head + f"<div class='tiles'>{''.join(cards)}</div><div class='note'>One forecast, outside the frozen "
            "benchmark: not saved to results and never scored there. The benchmark aggregates over all cases are in "
            "the Benchmark tab.</div>")


def _station_obs(run, station: str, variable: str) -> pd.DataFrame:
    obs = run.full_observations()
    return obs[obs["station"] == station][["time", variable]].rename(columns={variable: "observed"})


def overview(applied, variable, systems, band_model, error_mode="absolute"):
    a = applied
    run = RUNS[a["suite"]]
    rows, _, case = case_forecast(run, a["case"], variable)
    obs = _station_obs(run, case["station"], variable)
    exploratory = [df[df["variable"] == variable] for df in a.get("exploratory", {}).values()]
    if exploratory:
        rows = pd.concat([rows, *exploratory], ignore_index=True)
        for col in ("origin", "valid_time"):    # saved (ns) and fresh timestamps must share one dtype
            rows[col] = pd.to_datetime(rows[col], utc=True).astype("datetime64[us, UTC]")
    rows = _window(rows, a["shown_h"])
    systems = [m for m in (systems or []) if m in set(rows["model"])]
    ctx_days = a["settings"]["context_days"] if a["source"] == "exploratory" else run.params(a["primary"]).get(
        "context_days", a["ctx"])
    fig = plots.forecast_figure(rows, obs, case, variable, systems, band_model or None,
                                context_hours=int(ctx_days) * 24, error_mode=error_mode or "off")
    scores, n_shared = forecast_scores(rows, obs, systems)
    this = this_forecast_html(scores, n_shared, variable)
    desc = "\n".join(f"- **{model_name(m)}**: {model_description(m)}" for m in systems)
    origin = pd.Timestamp(case["origin"])
    plotted = int(rows.loc[rows["model"].isin(systems), "lead_h"].max()) if systems else 0
    arome = (f" AROME run used: {case['arome_run'][:16].replace('T', ' ')} UTC (in the archive at "
             f"{case['arome_available_at'][11:16]} UTC)." if isinstance(case.get("arome_run"), str) else "")
    assumed = (" AROME availability is inferred retrospectively from archive timestamps."
               if isinstance(case.get("arome_run"), str) else "")
    last = plots.last_observation(obs, origin)
    gap = ("" if last is None or last[0] == origin else
           f" The last observation before the forecast time is at {last[0]:%a %d %b %H:%M} UTC, "
           f"{(origin - last[0]) / pd.Timedelta(hours=1):.0f} h earlier; the dotted lines start there.")
    err_txt = {"absolute": " Lower panel: absolute error |forecast − observation|",
               "signed": " Lower panel: signed error, forecast − observation (above zero: too high)"}
    caption = (f"**{station_label(case['station']).split(' ', 1)[-1]}**, forecast issued "
               f"{origin:%a %d %b %Y, %H:%M} UTC (historical replay: inputs restricted to data assumed available "
               f"at the forecast time).{assumed}{arome} Model input: **{ctx_days} days** of observations "
               f"({int(ctx_days) * 24 // run.step_hours} steps); the chart opens on the last 48 h — zoom out to see "
               f"the rest. Plotted: leads 3–{plotted} h. Shaded ranges show where the probabilistic system expects "
               "the outcome 80% / 90% of the time; AROME and the simple references give single values. Dotted lines "
               "join the last observation to each forecast's first value: a drawing aid, not a prediction, never "
               "scored or exported." + gap
               + (err_txt[error_mode] + ", from the same point forecast the benchmark scores." if error_mode in
                  err_txt else ""))
    path = Path(tempfile.gettempdir()) / f"forecast_{a['case'].replace('/', '_')}_{variable}.csv"
    saved_only = rows[~rows["model"].astype(str).str.startswith("exploratory")]   # exploratory rows never exported
    forecast_csv(saved_only, obs).to_csv(path, index=False)
    return fig, caption, this, desc, str(path), plots.station_map(run.stations(), case["station"])


@zerogpu.gpu(duration=60)
def _exploratory_on_device(obs, case, size, context_days, calendar, arome_mode, arome_df, allow_download,
                           horizon_steps=16):
    """One labelled Timestral run; on ZeroGPU this function holds a GPU for its duration."""
    from weather_lab.forecast import exploratory_forecast
    from weather_lab.models import MODELS

    device = zerogpu.inference_device()
    try:
        return exploratory_forecast(obs, case, ["t", "pmer", "u", "v"], size, context_days, calendar,
                                    arome_mode=arome_mode, arome_df=arome_df, allow_download=allow_download,
                                    device=device, horizon_steps=horizon_steps)
    finally:
        if device == "cuda" and zerogpu.on_zerogpu():
            MODELS.release("cuda")   # the GPU is detached after this call; never keep GPU-resident state


def initial_applied(suite: str) -> dict:
    """The applied state at startup / experiment change: the saved forecast of the default configuration."""
    st = default_station(suite)
    case = default_case(suite, st)
    setup = e_main(suite)
    d = draft(suite, case, DEFAULT_SIZE, setup, default_context(suite, DEFAULT_SIZE, setup), RUNS[suite].horizon_hours)
    return resolve(d, None, lookup_cache=False) or make_applied(d, "saved", primary_saved(suite, d["size"], setup)
                                                                 or "", None)


# --------------------------------------------------------------------------- benchmark

LEADS = [("All lead times", "all"), ("6 h ahead", "6"), ("24 h ahead", "24"), ("48 h ahead", "48")] + [
    (f"{h} h ahead", str(h)) for h in range(3, 49, 3) if h not in (6, 24, 48)]
METRIC_CHOICES = [("Mean absolute error", "mae"), ("RMSE", "rmse"), ("Pinball loss", "pinball"),
                  ("80% range coverage", "coverage80"), ("90% range coverage", "coverage90"),
                  ("80% range width", "width80"), ("90% range width", "width90")]
LEAD_VIEWS = [("At each lead", "at"), ("Cumulative: all leads up to H", "cumulative")]
ERROR_MODES = [("Absolute |forecast − observation|", "absolute"), ("Signed (forecast − observation)", "signed"),
               ("Off", "off")]


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


NEGLIGIBLE = 0.01   # relative MAE change below which the finding says "essentially unchanged"


def is_pilot(model: str) -> bool:
    """Context-enrichment / post-processing pilot systems (development diagnostics, features-dev-v1)."""
    return "-f-" in model or model == "arome-biascorr"


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
    mae = allmae[[m for m in allmae.index if m not in DIFFERENT_INPUT and not is_pilot(m)]]
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
    if setup == "none" and len(pairs) == 2:          # nothing added: compare the checkpoints only
        (small, _), (large, _) = pairs.values()
        diffs.append(abs(allmae[large] - allmae[small]) / allmae[small])
        clauses.append("without AROME input, the larger Timestral checkpoint "
                       + ("performs better" if allmae[large] < allmae[small] else "does not perform better"))
    elif pairs:
        rel ={s: (allmae[c] - allmae[b]) / allmae[b] for s, (b, c) in pairs.items()}
        better = [s for s, r in rel.items() if r <= -NEGLIGIBLE]
        diffs += [abs(r) for r in rel.values()]
        verb = f"adding the {setup_txt}" if mx else "conditioning on AROME"
        cond = (f"{verb} leaves Timestral's error essentially unchanged (within {NEGLIGIBLE:.0%})"
                if all(abs(r) < NEGLIGIBLE for r in rel.values()) else
                f"{verb} improves Timestral" if len(better) == len(pairs) else
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
    ih = input_history(run)
    if len(ih):
        ctxs = {ih.loc[m, "requested_days"] for pair in pairs.values() for m in pair if m in ih.index}
        if len(ctxs) > 1:
            notes.append("The compared systems use different input histories, so more than the input setup or "
                         "checkpoint changes.")
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
        ih = input_history(run)
        ids = [matrix_id(suite, s, key) for s in mx["sizes"]]
        days = [ih.loc[m, "requested_days"] if m in ih.index else None for m in ids if m in pt.index]
        if not days:
            hist = "–"
        elif any(d is None or pd.isna(d) for d in days):
            hist = "Unknown"
        elif len(set(days)) == 1:
            m0 = next(m for m in ids if m in ih.index)
            hist = f"{days[0]:g} d each ({int(ih.loc[m0, 'requested_steps'])} steps); AROME: {arome_history_text(ih.loc[m0])}"
        else:
            hist = "⚠ contexts differ (" + " vs ".join(f"{d:g} d" for d in days) + "): more than checkpoint size changes"
        body.append(f"<tr><td><b>{SETUP_NAMES[key][0]}</b></td>{''.join(cells)}<td>{hist}</td></tr>")
    return (f"<table class='mini matrix'><thead><tr><th>Input setup ({variable_name(variable, unit=False).lower()}, "
            f"all leads)</th>{head}<th>Input history</th></tr></thead><tbody>{''.join(body)}</tbody></table>"
            "<div class='note'>Within a row only the checkpoint differs: same cases, observations, AROME runs, "
            "history length, calendar inputs and scoring. <i>not run</i>: the model can take this input but no "
            "result is saved; <i>unsupported input</i>: the model family cannot take it.</div>")


def context_label(run, models: list[str]) -> str:
    """Actual model input history of the compared systems, from the run's provenance (not the chart)."""
    ih = input_history(run)
    ts = [m for m in models if m in ih.index and m.startswith("timestral")]
    if not ts:
        return ""
    days = {ih.loc[m, "requested_days"] for m in ts}
    if None in days or any(pd.isna(d) for d in days):
        hist = "**Input history: Unknown** for " + ", ".join(model_name(m) for m in ts if pd.isna(
            ih.loc[m, "requested_days"])) + " (not recorded in this run)."
    elif len(days) == 1:
        lo, hi = int(ih.loc[ts, "effective_min"].min()), int(ih.loc[ts, "effective_max"].max())
        steps = int(ih.loc[ts[0], "requested_steps"])
        hist = (f"**Input history:** every Timestral system here reads {days.pop():g} days of station observations "
                f"({steps} × {run.step_hours} h steps; observed {lo if lo == hi else f'{lo}–{hi}'} of {steps} per case).")
    else:
        hist = ("**Input history differs between systems** (" + "; ".join(
            f"{model_name(m)} {ih.loc[m, 'requested_days']:g} d" for m in ts) + "): comparisons across them change "
            "more than one thing.")
    arome = sorted({arome_history_text(ih.loc[m]) for m in ts if ih.loc[m, "arome_history"] not in ("", "not used")})
    return hist + (" AROME history (separate from the observation context): " + "; ".join(arome) + "."
                   if arome else "")


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
               + (" The lead-time chart covers every season." if season != "ALL" else "")
               + "\n\n" + context_label(run, list(t["model"])))
    return (finding(suite, variable, split, setup), summary, bars, lines, horizon_txt,
            *detail_tables(run, split, variable, station, season, lead, statuses))


def benchmark_applied(suite, variable, metric, lead, split, station, season, statuses, applied, lead_mode="at"):
    """Benchmark for the applied model / input setup (never the unapplied draft)."""
    a = applied if applied and applied.get("suite") == suite else {}
    return benchmark(suite, variable, metric, lead, split, station, season, statuses, a.get("size", DEFAULT_SIZE),
                     a.get("setup") or e_main(suite), lead_mode)


def detail_tables(run, split, variable, station, season, lead, statuses):
    u = unit(variable)
    keep = lambda d: d[d["model"].map(training_status).isin(statuses or STATUSES)]  # noqa: E731
    p = keep(_select(run.tables["metrics_point"], split, variable, station, season, lead)).sort_values("mae")
    ih = input_history(run)
    hist = lambda m: ih.loc[m, "summary"] if m in ih.index else "Unknown"  # noqa: E731
    ahist = lambda m: arome_history_text(ih.loc[m]) if m in ih.index else "Unknown"  # noqa: E731
    point = pd.DataFrame({"System": p["model"].map(model_name), f"MAE ({u})": p["mae"].round(2),
                          f"RMSE ({u})": p["rmse"].round(2), f"Bias ({u})": p["bias"].round(2),
                          "vs persistence": p["skill_mae_vs_persistence"].map(lambda x: f"{x:+.0%}"),
                          "Pairs": p["n"], "Input history": p["model"].map(hist),
                          "AROME history": p["model"].map(ahist)})
    q = keep(_select(run.tables["metrics_probabilistic"], split, variable, station, season, lead)).sort_values(
        "pinball")
    prob = pd.DataFrame({"System": q["model"].map(model_name), f"Pinball ({u})": q["pinball"].round(3),
                         "80% coverage": q["coverage80"].map("{:.0%}".format),
                         "90% coverage": q["coverage90"].map("{:.0%}".format),
                         f"80% width ({u})": q["width80"].round(1), "Pairs": q["n"],
                         "Input history": q["model"].map(hist)})
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
        scope = gr.HTML(scope_html(DEFAULT_SUITE), elem_id="scope")
        pref = gr.State({})
        a0 = initial_applied(DEFAULT_SUITE)
        applied = gr.State(a0)
        pending_run = gr.State(None)
        with gr.Tabs(selected="overview"):
            # ---------------- overview
            with gr.Tab("Overview", id="overview"):
                with gr.Group(elem_id="forecast-panel"):
                    with gr.Row(equal_height=True):
                        station = gr.Dropdown(station_choices(DEFAULT_SUITE), value=default_station(DEFAULT_SUITE),
                                              label="Station", scale=2, min_width=160)
                        date = gr.Dropdown(date_choices(DEFAULT_SUITE, default_station(DEFAULT_SUITE)),
                                           value=a0["case"], label="Forecast issued", scale=3, min_width=210)
                        var = gr.Dropdown(variable_choices(DEFAULT_SUITE), value="t", label="Weather variable",
                                          scale=2, min_width=150)
                        ckpt = gr.Radio(list(CHECKPOINTS), value=DEFAULT_SIZE, label="Timestral model", scale=2,
                                        min_width=150, visible=has_checkpoints(DEFAULT_SUITE))
                        setup = gr.Dropdown(setup_choices(DEFAULT_SUITE), label="Input configuration", scale=3,
                                            value=e_main(DEFAULT_SUITE), min_width=230,
                                            visible=has_setups(DEFAULT_SUITE))
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=4, min_width=240):
                            ctx = gr.Slider(*CONTEXT_DAYS, value=a0["ctx"], step=1, label="Observed context (days)")
                            ctx_info = gr.Markdown(context_md(a0["ctx"], DEFAULT_SUITE), elem_classes="note")
                        with gr.Column(scale=4, min_width=240):
                            h_upd, h_info0, _ = horizon_control(DEFAULT_SUITE, a0["case"], e_main(DEFAULT_SUITE),
                                                                a0["horizon"])
                            horizon = gr.Slider(3, h_upd["maximum"], value=a0["horizon"], step=3,
                                                label="Forecast horizon (hours)")
                            horizon_info = gr.Markdown(h_info0, elem_classes="note")
                        with gr.Column(scale=2, min_width=160):
                            e_btn = gr.Button("Run forecast", variant="primary")
                            e_msg = gr.Markdown(elem_classes="note")
                status = gr.HTML(status_html(a0, DEFAULT_SUITE, a0["case"], a0["size"], a0["setup"], a0["ctx"],
                                             a0["horizon"]), elem_id="applied")
                fplot = gr.Plot(show_label=False)
                tiles = gr.HTML()
                caption = gr.Markdown(elem_classes="note")
                with gr.Row():
                    with gr.Column(scale=2, min_width=280):
                        this_tab = gr.HTML()
                    with gr.Column(scale=3, min_width=280):
                        sys_desc = gr.Markdown(elem_classes="note")
                with gr.Accordion("Advanced: comparison overlays, uncertainty shading, error display, downloads",
                                  open=False):
                    with gr.Row():
                        with gr.Column(scale=2, min_width=260):
                            systems = gr.CheckboxGroup(label="Comparison overlays", choices=[])
                            band = gr.Dropdown(label="Shaded 80% / 90% range for", choices=[], value=None)
                            err_mode = gr.Radio(ERROR_MODES, value="absolute", label="Error panel under the chart")
                            show_old = gr.Checkbox(False, label="Show superseded experiment versions")
                            dl_case = gr.DownloadButton("Download this forecast (CSV, saved systems)", size="sm")
                        with gr.Column(scale=1, min_width=220):
                            smap = gr.Plot(show_label=False)

            # ---------------- benchmark
            with gr.Tab("Benchmark", id="benchmark"):
                with gr.Row(equal_height=True):
                    b_var = gr.Dropdown(variable_choices(DEFAULT_SUITE), value="t", label="Weather variable",
                                        min_width=160)
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
        draft_in = [exp, date, ckpt, setup, ctx, horizon]
        render_out = [systems, band, fplot, caption, this_tab, sys_desc, dl_case, smap, tiles]
        render_in = [applied, var, pref, band, err_mode]
        b_inputs = [exp, b_var, metric, lead, split, b_station, b_season, statuses, applied, lead_mode]
        b_outputs = [b_finding, b_summary, bars, lines, b_horizon, point_tab, prob_tab]
        exp_outputs = [var, b_var, station, date, scope, ckpt, setup, ctx, horizon]

        def show(event):
            """Render the applied result (one step) and the badge/pending line."""
            return event.then(render, render_in, render_out).then(status_html, [applied] + draft_in, status)

        def refresh_benchmark(event):
            return (event.then(benchmark_applied, b_inputs, b_outputs)
                    .then(matrix_html, [exp, b_var], matrix)
                    .then(advanced_tables, [exp, b_var, split], [avail_tab, paired_tab, rt_tab, hw])
                    .then(downloads, exp, [dl_point, dl_prob, dl_manifest]))

        def limit(event):
            return event.then(horizon_control, [exp, date, setup, horizon], [horizon, horizon_info, e_btn])

        def draft_changed(event):
            return show(limit(event).then(on_draft, draft_in + [applied], applied))

        def on_exp_chain(event):
            event = event.then(bench_controls, exp, [split, b_station, b_season]).then(
                context_md, [ctx, exp], ctx_info)
            return refresh_benchmark(draft_changed(event))

        exp_in = [exp, var, station, ckpt, setup]
        on_exp_chain(exp.input(on_experiment, exp_in, exp_outputs))
        draft_changed(station.input(on_station, [exp, station], date))
        for c in (date, setup):
            refresh_benchmark(draft_changed(c.input(lambda: None, None, None)))
        refresh_benchmark(draft_changed(ckpt.input(lambda: None, None, None)))
        show(ctx.input(context_md, [ctx, exp], ctx_info).then(on_draft, draft_in + [applied], applied))
        draft_changed(horizon.input(lambda: None, None, None))      # constrain + explain, then resolve
        var.input(lambda v: v, var, b_var).then(render, render_in, render_out).then(
            benchmark_applied, b_inputs, b_outputs)
        refresh_benchmark(b_var.input(lambda v: v, b_var, var)).then(render, render_in, render_out)
        systems.input(update_pref, [applied, var, systems, pref], pref).then(render, render_in, render_out)
        for c in (band, err_mode):
            c.input(render, render_in, render_out)
        show_old.input(lambda s, cur: gr.update(choices=experiment_choices(s), value=cur), [show_old, exp], exp)
        refresh_benchmark(show(
            e_btn.click(lambda: "Running…", None, e_msg)
            .then(run_forecast, draft_in + [applied], [pending_run, e_msg], concurrency_limit=1)
            .then(apply_if_current, [pending_run] + draft_in + [applied, e_msg], [applied, e_msg])))
        for c in (metric, lead, split, b_station, b_season, statuses, lead_mode):
            c.input(benchmark_applied, b_inputs, b_outputs)
        split.input(advanced_tables, [exp, b_var, split], [avail_tab, paired_tab, rt_tab, hw])
        cache_btn.click(cache_md, None, cache_box)

        on_exp_chain(demo.load(on_experiment, exp_in, exp_outputs)).then(cache_md, None, cache_box)
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
