"""Forecast drafts, captured execution, applied state and lightweight control feedback."""

from __future__ import annotations

from weather_lab_app import zerogpu, results

import threading
from functools import lru_cache
import gradio as gr
import numpy as np
import pandas as pd
from weather_lab.common import VARIABLES
from weather_lab_app.labels import CHECKPOINTS, EXPERIMENTS, SETUP_NAMES, model_name, variable_name, with_checkpoint
from weather_lab_app.results import case_forecast, station_label, calendar_case, calendar_value, calendar_info
from weather_lab import features as FEATS
from weather_lab.forecast_config import PREDICTION_TARGETS, Availability, ForecastConfig, Issue, check, describe, from_system


DEFAULT_STATION = "07149"            # Paris-Orly, by stable station id


DEFAULT_SIZE = CHECKPOINTS[-1]       # the largest integrated Timestral checkpoint (209M)


FORECAST_MODELS = [(f"Timestral-{s}", s) for s in CHECKPOINTS] + [("TFC t0-gamma", "t0-gamma")]


EXPLORE_MAX_HOURS = 120              # app limit for exploratory runs (Timestral itself sets no horizon limit)


CONTEXT_DAYS = (7, 90)               # range of the Observed context control


AROME_MODES = {"none": None, "archived": "archived", "substobs": "substituted_observations", "futureonly": "missing"}


def experiment_choices(show_superseded: bool = False) -> list[tuple[str, str]]:
    return [(EXPERIMENTS[s]["name"], s) for s in results.RUNS if not EXPERIMENTS[s].get("benchmark_only") and (show_superseded or not EXPERIMENTS[s].get("superseded"))]


def variable_choices(suite: str) -> list[tuple[str, str]]:
    run = results.RUNS[suite]
    have = set(run.manifest["protocol"]["variables"]) if run.run.get("exploration") else set(run.predictions()["variable"])
    return [(variable_name(v, unit=False), v) for v in VARIABLES if v in have]


def scope_html(suite: str) -> str:
    e, run = EXPERIMENTS[suite], results.RUNS[suite]
    cls = "status validation" if e["status"] == "Validation" else "status"
    return (f"<div class='scope'><span class='{cls}'>{e['status']}</span>{e.get('short', e['scope'])} · "
            "saved experiment scope and availability rules apply</div>")


def has_checkpoints(suite: str) -> bool:
    """True when a headline 28M system has a saved 209M twin not already in the headline (selector shown)."""
    if has_setups(suite):
        return True
    models, head = set(results.RUNS[suite].models), EXPERIMENTS[suite]["headline"]
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
    e, models = EXPERIMENTS[suite], set(results.RUNS[suite].models)
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


def station_choices(suite: str) -> list[tuple[str, str]]:
    return [(station_label(s).split(" ", 1)[-1], s) for s in results.RUNS[suite].manifest["protocol"]["stations"]]


def default_case(suite: str, station: str) -> str:
    """Fixed, neutral rule: the most recent forecast of the last period for this station."""
    cases = results.RUNS[suite].cases()
    cases = cases[cases["eligible"] & (cases["station"] == station)]
    last = cases[cases["split"] == results.RUNS[suite].splits[-1]]
    return (last if len(last) else cases).sort_values("origin")["case_id"].iloc[-1]


def default_station(suite: str) -> str:
    """Paris-Orly (by station id) when the experiment has it, else its first UI example / first station."""
    stations = [s for _, s in station_choices(suite)]
    if DEFAULT_STATION in stations:
        return DEFAULT_STATION
    first_example = (results.RUNS[suite].manifest.get("ui_examples") or [""])[0]
    return first_example.split("/")[1] if first_example.count("/") == 2 else stations[0]


def on_station(suite: str, station: str):
    run = results.RUNS[suite]
    return gr.update(value=calendar_value(run, default_case(suite, station)), info=calendar_info(run, station))


def saved_settings(suite: str, size: str | None, setup: str | None) -> dict:
    """Settings of the saved run behind the current view, from its provenance (run.json params, manifest)."""
    run, e = results.RUNS[suite], experiment_view(suite, size, setup)
    mid = e["band"] or next((m for m in e["headline"] if m.startswith("timestral") and m in run.models), None)
    p = run.params(mid) if mid else {}
    return {"model": mid, "context_days": p.get("context_days"), "calendar": p.get("calendar"),
            "horizon_h": run.horizon_hours, "missing": e["missing"]}


def default_context(suite: str, size: str | None, setup: str | None) -> int:
    """Context-control default: the active saved run's recorded context (never the validation choice of
    another run); the suite protocol's value when the view's system was not run."""
    days = saved_settings(suite, size, setup)["context_days"]
    return int(days or results.RUNS[suite].manifest["protocol"]["forecast"].get("context_days", 30))


def context_md(days, suite: str) -> str:
    step = results.RUNS[suite].step_hours
    return (f"= {int(days) * 24} hours = {int(days) * 24 // step} {'three-hourly ' if step == 3 else ''}steps "
            "(read-only conversion).")


def settings_text(st: dict, step: int = 3) -> str:
    setup = next((k for k, v in AROME_MODES.items() if v == st.get("arome_mode")), "none")
    feats_txt = ", ".join(FEATURE_LABELS.get(f, f) for f in st.get("features") or []) or "no additional features"
    method = "AROME + predicted error · forecast lead input · " if st.get("prediction_target") == "arome_error" else ""
    cond = "no additional AROME covariates" if method and setup == "none" else SETUP_NAMES[setup][0]
    retrieval = (f"RAG {'AROME errors · ' if st.get('retrieval_kind') == 'arome_error' else ''}"
                 f"{st['retrieval_method']} · hourly archive · observations delayed 3 h · ") if st.get("retrieval_method") else ""
    return (f"{st.get('checkpoint', '?')} · {retrieval}{method}{cond} · {feats_txt} · "
            f"calendar {'on' if st.get('calendar', True) else 'off'} · context {st.get('context_days')} d "
            f"({st.get('context_steps')} steps) · inference horizon {st.get('horizon_steps', 0) * st.get('step_hours', step)} h")


CALENDAR = True          # default: every saved Timestral system of the main experiments uses calendar inputs


def primary_saved(suite: str, size: str | None, setup: str | None) -> str | None:
    """The saved Timestral system a (size, input setup) selects in this experiment's view, or None."""
    run, e = results.RUNS[suite], experiment_view(suite, size, setup)
    if has_setups(suite):
        return matrix_id(suite, size or DEFAULT_SIZE, setup or "none")
    return e["band"] or next((m for m in e["headline"] if m.startswith("timestral") and m in run.models), None)


CONDITIONING_CHOICES = [("No AROME conditioning" if k == "none" else SETUP_NAMES[k][0], k) for k in AROME_MODES]


FEATURE_LABELS = {n: g.label for n, g in FEATS.GROUPS.items()}


FEATURE_CHOICES = [(g.label, n) for n, g in FEATS.GROUPS.items()]


DRAFT_KEYS = ("suite", "station", "case", "variable", "size", "conditioning", "features", "calendar", "ctx",
              "horizon", "resid_vars", "atmos_fields", "tend_lags", "overlays", "band", "error_mode", "prediction_target",
              "metric", "retrieval_method", "retrieval_index")


FORECAST_KEYS = DRAFT_KEYS[:13] + ("prediction_target", "retrieval_method", "retrieval_index")


PRESENTATION_KEYS = ("variable", "overlays", "band", "error_mode", "metric")


COUNTERS = {"applied": 0, "runs": 0}      # forecast applications / Run forecast clicks (this process)


_SUBMISSION_LOCK = threading.Lock()


DRAFT_LABELS = {"suite": "experiment", "station": "station", "case": "forecast time", "variable": "variable",
                "size": "model", "conditioning": "AROME conditioning", "features": "feature groups",
                "calendar": "calendar inputs", "ctx": "context", "horizon": "horizon",
                "resid_vars": "AROME-error variables", "atmos_fields": "dew point / cloud fields",
                "tend_lags": "tendency lags", "overlays": "comparison overlays", "band": "shaded range",
                "error_mode": "error panel", "prediction_target": "forecast method", "metric": "main metric",
                "retrieval_method": "retrieval method", "retrieval_index": "retrieval index"}


def make_draft(*vals) -> dict:
    """An immutable-by-convention copy of the forecast block's control values."""
    d = dict(zip(DRAFT_KEYS, vals))
    if d["suite"] in results.RUNS:
        d["case"] = calendar_case(results.RUNS[d["suite"]], d["station"], d["case"])
    d.setdefault("prediction_target", "weather")
    d.setdefault("metric", "mae")
    d.setdefault("retrieval_method", "none")
    d["retrieval_index"] = d.get("retrieval_index") or ""
    if d["metric"] not in ("mae", "rmse"):
        raise ValueError("Main metric must be MAE or RMSE")
    d["features"] = [n for n in FEATS.GROUPS if n in (d.get("features") or [])]
    for k in ("resid_vars", "atmos_fields", "overlays"):
        d[k] = sorted(d.get(k) or [])
    d["tend_lags"] = sorted(int(x) for x in d.get("tend_lags") or [])
    d["ctx"], d["horizon"], d["calendar"] = int(d["ctx"] or 0), int(d["horizon"] or 0), bool(d["calendar"])
    d["band"] = d.get("band") or ""
    return d


def feature_spec(d: dict) -> dict:
    p = {"resid": {"variables": d["resid_vars"]}, "atmos": {"fields": d["atmos_fields"]},
         "tend": {"lags_h": d["tend_lags"]}, "solar": {}}
    return {n: p.get(n, {}) for n in d["features"]}


def availability(suite: str, case_id: str) -> Availability:
    """What data exist for this case: issued AROME coverage, archived AROME, auxiliary obs, coordinates.
    Metadata only (the in-memory saved AROME extract and file presence); no inputs are built."""
    return _availability(suite, case_id, id(results.RUNS[suite]))


@lru_cache(maxsize=4096)
def _availability(suite: str, case_id: str, run_id: int) -> Availability:
    from weather_lab.forecast import arome_coverage_steps

    run = results.RUNS[suite]
    case = run.case(case_id)
    origin, step = pd.Timestamp(case["origin"]), run.step_hours
    ext = run.arome_extract()
    issued, has_hist, notes = 0, False, {}
    if ext is None:
        notes["arome"] = notes["arome_history"] = "this experiment has no saved AROME forecasts"
    else:
        sub = ext[ext["station"] == case["station"]]
        if isinstance(case.get("arome_run"), str):
            run_sub = sub[sub["run"] == pd.Timestamp(case["arome_run"])]
            piv = run_sub.pivot_table(index="valid_time", columns="variable", values="value")
            times = [origin + pd.Timedelta(hours=step * (k + 1)) for k in range(EXPLORE_MAX_HOURS // step)]
            rows = piv.reindex(times).T.to_numpy(dtype=float) if len(piv) else np.full((1, len(times)), np.nan)
            issued = arome_coverage_steps(rows) * step
        else:
            notes["arome"] = "no AROME run was available at this forecast time"
        q = sub[(sub["available_at"] <= origin) & (sub["available_at"] <= sub["valid_time"])
                & (sub["valid_time"] <= origin) & (sub["valid_time"] > sub["run"])]
        has_hist = len(q) > 0
        if not has_hist:
            notes["arome_history"] = ("no archived AROME forecast had been issued for a time before this forecast "
                                      "time (the archive starts later); needed even without AROME conditioning")
    aux = _aux_path(suite, id(run)) is not None
    if not aux:
        notes["aux_history"] = "no dew point / cloud observation file is installed for this experiment"
    return Availability(issued, case.get("arome_run") if issued else None, has_hist, aux,
                        case["station"] in run.station_coords(), notes)


@lru_cache(maxsize=16)
def _aux_path(suite: str, run_id: int):
    return results.RUNS[suite].aux_path()


def horizon_limit(suite: str, case_id: str, conditioning: str | None,
                  prediction_target: str = "weather") -> tuple[int, str]:
    """(largest supported horizon in hours, why) for this forecast origin and AROME conditioning."""
    run = results.RUNS[suite]
    correction = prediction_target == "arome_error"
    if not AROME_MODES.get(conditioning or "none") and not correction:
        return EXPLORE_MAX_HOURS, (f"Timestral without AROME conditioning: up to {EXPLORE_MAX_HOURS} h (app limit); "
                                   f"saved forecasts cover {run.horizon_hours} h")
    av = availability(suite, case_id)
    if not av.arome_issued_h:
        return 0, av.notes.get("arome", "no saved AROME run for this forecast") + ", so the AROME baseline is unavailable"
    if correction:
        return av.arome_issued_h, (f"AROME error correction needs the issued AROME baseline, which ends "
                                   f"{av.arome_issued_h} h after this forecast time")
    return av.arome_issued_h, (f"AROME run {av.arome_run[:16].replace('T', ' ')} UTC ends {av.arome_issued_h} h after "
                               "this forecast time; AROME inputs are never extended. Without AROME conditioning: up "
                               f"to {EXPLORE_MAX_HOURS} h")


def horizon_control(suite, case_id, conditioning, horizon, prediction_target="weather"):
    """Slider maximum from the supported limit. A value above it is reduced and the reduction is stated."""
    limit, why = horizon_limit(suite, case_id, conditioning, prediction_target)
    step = results.RUNS[suite].step_hours
    if not limit:
        return gr.update(minimum=step, step=step, maximum=EXPLORE_MAX_HOURS), f"⚠️ {why}."
    value = min(int(horizon), limit)
    cut = f" **Reduced from {int(horizon)} h** to the maximum." if value < int(horizon) else ""
    return (gr.update(minimum=step, step=step, maximum=limit, value=value),
            f"= {value // step} steps of {step} h. Maximum here: **{limit} h** — {why}.{cut}")


def draft_config(d: dict) -> tuple[ForecastConfig | None, list]:
    """(configuration, every capability / availability / validity issue) for a draft. No data loading."""
    run = results.RUNS[d["suite"]]
    try:
        spec = FEATS.normalize(feature_spec(d))
    except ValueError as e:
        name = next((n for n in d["features"] if f"'{n}'" in str(e)), "features")
        return None, [Issue("invalid", name, str(e).split(": ", 1)[-1])]
    correction = d["prediction_target"] == "arome_error"
    retrieval = d["retrieval_method"] != "none"
    cfg = ForecastConfig.make("t0-gamma" if d["size"] == "t0-gamma" else f"Timestral-{d['size']}",
                              d["conditioning"] or "none", spec, calendar=d["calendar"],
                              context_days=d["ctx"], horizon_h=d["horizon"], step_hours=1 if retrieval else run.step_hours,
                              prediction_target=d["prediction_target"],
                              targets=("t", "pmer") if correction or retrieval else ("t", "pmer", "u", "v"),
                              retrieval_method=d["retrieval_method"], retrieval_index=d["retrieval_index"] if retrieval else "")
    try:
        case = run.case(d["case"])
        if case.station != d["station"]:
            raise ValueError("Forecast origin belongs to a different station")
    except (ValueError, KeyError) as exc:
        return cfg, [Issue("invalid", "forecast time", str(exc))]
    # RAG resolves availability from its selected hourly archive on Run, never from the frozen suite.
    issues = check(cfg, None if retrieval else availability(d["suite"], d["case"]), EXPLORE_MAX_HOURS, CONTEXT_DAYS)
    if retrieval and d["variable"] not in cfg.targets:
        issues.append(Issue("unsupported", "retrieval", "select temperature or mean sea-level pressure"))
    if correction and d["variable"] not in cfg.targets:
        issues.append(Issue("unavailable", "prediction_target", "the installed AROME archive supplies correction "
                            "baselines for temperature and mean sea-level pressure; no wind baseline is installed"))
    return cfg, issues


def _systems_as_configs(suite: str) -> dict[str, ForecastConfig]:
    run = results.RUNS[suite]
    out = {}
    for s in run.run["systems"]:
        c = from_system(s, run.horizon_hours, run.step_hours)
        if c is not None:
            out[s["id"]] = c
    return out


def saved_match(d: dict, cfg: ForecastConfig) -> str | None:
    """The saved benchmark system whose recorded configuration equals the draft's (horizon may be a slice)."""
    run = results.RUNS[d["suite"]]
    if run.run.get("exploration"):
        return None
    if d["case"] not in set(run.cases()["case_id"]) or cfg.horizon_h > run.horizon_hours:
        return None
    hits = [m for m, c in _systems_as_configs(d["suite"]).items() if c.forecast_part() == cfg.forecast_part()]
    return hits[0] if hits else None


def _explore_inputs(d: dict, cfg: ForecastConfig):
    run = results.RUNS[d["suite"]]
    _, _, case = case_forecast(run, d["case"], "t")
    obs = run.full_observations()
    needs = {dep for name in cfg.feature_names for dep in FEATS.GROUPS[name].needs}
    needs_arome = bool(cfg.arome_mode) or "arome_history" in needs or cfg.prediction_target == "arome_error"
    station_obs = obs[obs["station"] == case["station"]]
    coords = run.station_coords()
    if run.run.get("exploration"):
        positions = station_obs[station_obs.time.le(case.origin)].dropna(subset=["LAT", "LON"])
        if len(positions):
            coords = {case.station: (float(positions.iloc[-1].LAT), float(positions.iloc[-1].LON))}
    return (station_obs, case, run.arome_extract() if needs_arome else None,
            run.aux_observations() if "aux_history" in needs else None, coords)


def _device() -> str:
    return "cuda" if zerogpu.on_zerogpu() else "cpu"


def _explore_kwargs(cfg: ForecastConfig, arome_df, aux_df, coords, step: int) -> dict:
    return {"arome_mode": cfg.arome_mode, "arome_df": arome_df, "horizon_steps": cfg.horizon_h // step,
            "step_hours": step, "features": cfg.feature_spec, "aux_df": aux_df, "station_coords": coords,
            "prediction_target": cfg.prediction_target}


def make_applied(d: dict, cfg: ForecastConfig, source: str, primary: str, prev: dict | None = None, frame=None,
                 info=None) -> dict:
    """The applied result (forecast part): configuration, source, primary system, this case's exploratory runs."""
    same_case = bool(prev) and prev.get("suite") == d["suite"] and prev.get("case") == d["case"]
    runs = dict(prev.get("exploratory", {})) if same_case else {}
    settings = {}
    if frame is not None:
        runs[primary] = frame
        settings = {**info["settings"], "cache_hit": info["cache_hit"], "key": info["key"]}
    run = results.RUNS[d["suite"]]
    shown = min(cfg.horizon_h, int(frame["lead_h"].max()) if frame is not None else run.horizon_hours)
    configs_by = dict(prev.get("configs_by", {})) if same_case else {}
    configs_by[primary] = cfg
    comparison = None
    if same_case and cfg.prediction_target == "arome_error":
        comparison = (prev["primary"] if prev["config"].prediction_target == "weather" else prev.get("comparison"))
    return {"suite": d["suite"], "case": d["case"], "size": d["size"],
            "setup": cfg.conditioning if has_setups(d["suite"]) else None, "config": cfg, "source": source,
            "primary": primary, "settings": settings, "exploratory": runs,
            "settings_by": {**(prev.get("settings_by", {}) if same_case else {}), **({primary: settings} if settings else {})},
            "shown_h": shown, "configs_by": configs_by, "comparison": comparison}


def present(new: dict, d: dict, prev: dict | None) -> dict:
    """Attach the draft's presentation (variable, overlays, interval source, error display) and the draft."""
    variable = d["variable"] if d["variable"] in [v for _, v in variable_choices(new["suite"])] else "t"
    pref = dict((prev or {}).get("view", {}).get("pref", {}))
    if prev and prev.get("view") and prev["suite"] == new["suite"]:
        pref = update_pref(prev, prev["view"]["variable"], d["overlays"], pref)
    new_forecast = not prev or prev.get("primary") != new["primary"] or prev.get("case") != new["case"]
    if new_forecast and new["config"].prediction_target == "arome_error":
        p = dict(pref.get(new["suite"], {}))
        keep = {new["primary"], "arome-raw", new.get("comparison")}
        p["removed"] = [m for m in p.get("removed", []) if m not in keep]
        pref[new["suite"]] = p
    choices, shown, prob, band = select_systems(new, variable, pref, d["band"], keep_band=not new_forecast)
    new["view"] = {"variable": variable, "pref": pref, "shown": shown, "choices": choices, "prob": prob,
                   "band": band, "error_mode": d["error_mode"] or "absolute", "metric": d["metric"]}
    new["draft"] = dict(d)
    return new


def begin_forecast(*vals):
    """Capture the click's draft and claim this session until its result has finished rendering."""
    *dvals, applied, submission = vals
    request = {"draft": make_draft(*dvals), "applied": applied}
    with _SUBMISSION_LOCK:
        if submission["busy"]:
            raise gr.Error("A forecast is already running in this session.")
        submission["busy"] = True
    return request, gr.update(interactive=False, value="Running…"), "Running…"


def run_captured(request):
    try:
        return run_forecast(*[request["draft"][k] for k in DRAFT_KEYS], request["applied"])
    except Exception as e:  # validation failures must also reach the session's cleanup callback
        return None, f"⚠️ Run forecast failed: {type(e).__name__}: {e}. The displayed result is unchanged."


def finish_forecast(*vals):
    with _SUBMISSION_LOCK:
        vals[-1]["busy"] = False
    return draft_feedback(*vals)


def run_forecast(*vals):
    """Run forecast: validate, then reuse / saved / cached / infer. Returns ({draft, applied} | None, message)."""
    from weather_lab import predcache
    from weather_lab.forecast import ensure_checkpoint, exploratory_forecast

    *dvals, applied = vals
    COUNTERS["runs"] += 1
    d = make_draft(*dvals)
    keep = " The displayed result is unchanged."
    cfg, issues = draft_config(d)
    if issues:
        return None, "⚠️ Cannot run this configuration: " + "; ".join(i.text() for i in issues) + "." + keep
    prev_cfg = (applied or {}).get("config")
    if (applied and applied["suite"] == d["suite"] and applied["case"] == d["case"] and prev_cfg is not None
            and prev_cfg.forecast_part() == cfg.forecast_part() and prev_cfg.horizon_h == cfg.horizon_h):
        new = {k: v for k, v in applied.items() if k not in ("view", "draft")}
        return {"draft": d, "applied": present(new, d, applied)}, "Applied the presentation settings (same forecast: no model run)."
    mid = saved_match(d, cfg)
    if mid:
        saved_cfg = from_system(next(s for s in results.RUNS[d["suite"]].run["systems"] if s["id"] == mid), cfg.horizon_h,
                                cfg.step_hours)
        new = make_applied(d, saved_cfg, "saved", mid, applied)
        return {"draft": d, "applied": present(new, d, applied)}, "Applied the saved benchmark forecast (no model run)."
    step = results.RUNS[d["suite"]].step_hours
    try:
        if cfg.retrieval_method != "none":
            from weather_lab.rag_forecast import forecast

            case = results.RUNS[d["suite"]].case(d["case"])
            obs, kw = None, {}
            frame, info = forecast(case, cfg, device=_device(), lookup_only=True)
        else:
            obs, case, arome_df, aux_df, coords = _explore_inputs(d, cfg)
            kw = _explore_kwargs(cfg, arome_df, aux_df, coords, step)
            frame, info = exploratory_forecast(obs, case, list(cfg.targets), cfg.model, cfg.context_days, cfg.calendar,
                                               device=_device(), lookup_only=True, **kw)
        if frame is None:
            try:
                ensure_checkpoint(cfg.model, False)
            except PermissionError as e:
                raise PermissionError(f"{e}. Install pinned weights with weather-lab models prefetch") from e
            frame, info = _exploratory_on_device(obs, case, cfg, kw)
    except (PermissionError, ValueError) as e:
        return None, f"⚠️ {e}.{keep}"
    except Exception as e:  # noqa: BLE001 - e.g. ZeroGPU quota/queue errors raised from the GPU worker
        return None, f"⚠️ Run forecast failed: {type(e).__name__}: {e}.{keep}"
    new = present(make_applied(d, cfg, "exploratory", frame["model"].iloc[0], applied, frame, info), d, applied)
    how = ("Applied the cached exploratory forecast (no model run)." if info["cache_hit"] else
           f"Computed {settings_text(new['settings'])} and cached it ({predcache.location()[1]}; key {info['key']}).")
    return {"draft": d, "applied": new}, how


def apply_if_current(result, *vals):
    """Apply a run result atomically, unless the draft changed while it was running. -> (applied, msg, changed)."""
    *dvals, applied, msg = vals
    if not result:
        return applied, msg, False
    if result["draft"] != make_draft(*dvals):
        return applied, ("A result for an earlier configuration arrived after the settings changed and was not "
                         "applied. Click Run forecast to apply the current settings."), False
    COUNTERS["applied"] += 1
    return result["applied"], msg, True


def applied_text(a: dict) -> str:
    cfg = a["config"]
    head = describe(cfg, {k: v[0] for k, v in SETUP_NAMES.items()}).replace(f"horizon {cfg.horizon_h} h",
                                                                             f"shown to {a['shown_h']} h")
    head += f" · main metric {a.get('view', {}).get('metric', 'mae').upper()}"
    if a["source"] == "saved":
        return f"{model_name(a['primary'])} = {head} (saved {results.RUNS[a['suite']].horizon_hours} h forecast)"
    return head


def pending_items(a: dict, d: dict) -> list[str]:
    """Readable differences between the draft and the applied draft (presentation compared with what is shown)."""
    ad = a.get("draft") or {}
    out = []
    for k in DRAFT_KEYS:
        if k in ("overlays", "band"):
            continue
        if k == "station" and d.get("case") != ad.get("case"):
            continue
        if d.get(k) != ad.get(k):
            v = d.get(k)
            val = (", ".join(FEATURE_LABELS[f] for f in v) or "none") if k == "features" else \
                ("on" if v else "off") if k == "calendar" else f"{v} d" if k == "ctx" else f"{v} h" if k == "horizon" \
                else SETUP_NAMES.get(v, (v,))[0] if k == "conditioning" else v if isinstance(v, (str, int)) else \
                ", ".join(map(str, v)) or "none"
            if k == "prediction_target":
                val = PREDICTION_TARGETS.get(v, v)
            if k == "metric":
                val = v.upper()
            if k == "case" and v in set(results.RUNS[d["suite"]].cases()["case_id"]):
                c = results.RUNS[d["suite"]].cases().set_index("case_id").loc[v]
                val = f"{station_label(c['station']).split(' ', 1)[-1]}, {c['origin']:%a %d %b %H:%M} UTC"
            out.append(f"{DRAFT_LABELS[k]} → {val}")
    view = a.get("view") or {}
    if set(d.get("overlays") or []) != set(view.get("shown") or []):
        out.append("comparison overlays")
    if (d.get("band") or "") != (view.get("band") or ""):
        out.append("shaded range")
    return out


def status_html(applied, *dvals) -> str:
    """Result summary (from `applied` only) and, separately, pending draft changes and why they cannot run."""
    if not applied:
        out = "<div class='pending'>No forecast has run. Select a UTC date and click <b>Run forecast</b>.</div>"
        if dvals:
            _, issues = draft_config(make_draft(*dvals))
            out += "".join(f"<div class='issues'>⛔ {i.text()}</div>" for i in issues)
        return out
    a = applied
    badge = ("<span class='badge saved'>Saved benchmark forecast</span>" if a["source"] == "saved" else
             f"<span class='badge expl'>Exploratory · {'cached' if a['settings'].get('cache_hit') else 'computed'}"
             "</span>")
    run = results.RUNS[a["suite"]]
    rid = f" <code>{run.run['run_id'][:40]}</code>" if a["source"] == "saved" else " outside the frozen benchmark"
    out = f"<div class='applied'>{badge}{rid}: {applied_text(a)}</div>"
    if not dvals:
        return out
    d = make_draft(*dvals)
    diff = pending_items(a, d)
    if diff:
        out += ("<div class='pending'>Changes pending — click <b>Run forecast</b>: " + "; ".join(diff)
                + ". The chart, cards and captions still show the result above.</div>")
    _, issues = draft_config(d)
    if issues:
        out += ("<div class='issues'>" + "".join(f"<div>⛔ <b>{i.subject_label()}</b> — {i.kind}: {i.message}</div>"
                                                 for i in issues) + "</div>")
    return out


def draft_feedback(*vals):
    """Every control edit: lightweight feedback only (no resolution, inference, input building or redraw)."""
    *dvals, applied, submission = vals
    d = make_draft(*dvals)
    if d["retrieval_method"] != "none":
        h_upd, h_info = gr.update(maximum=48), "Hourly RAG supports 24 or 48 h, with a 3 h observation delay."
    else:
        try:
            h_upd, h_info = horizon_control(d["suite"], d["case"], d["conditioning"], d["horizon"], d["prediction_target"])
        except (ValueError, KeyError):
            h_upd, h_info = gr.update(), ""
    if isinstance(h_upd, dict) and "value" in h_upd:
        dvals = list(dvals)
        dvals[DRAFT_KEYS.index("horizon")] = h_upd["value"]
        d = make_draft(*dvals)
    _, issues = draft_config(d)
    context_info = "RAG requires 30 days = 720 hourly steps from the index's saved archive." if d["retrieval_method"] != "none" else context_md(d["ctx"], d["suite"])
    return (status_html(applied, *dvals), h_upd, h_info, context_info,
            gr.update(interactive=not issues and not submission["busy"],
                      value="Running…" if submission["busy"] else "Run forecast"), feature_note(d))


def feature_note(d: dict) -> str:
    """Draft feature selection in words (roles; parameters live under Advanced)."""
    method = ("**AROME + predicted error:** the selected model predicts observation − archived AROME, then adds the issued "
              "AROME baseline. Forecast lead is an input; Calendar inputs add hour of day and time of year. "
              "The context is a maximum window: missing archived errors stay missing. "
              if d["prediction_target"] == "arome_error" else "")
    if not d["features"]:
        return method + "Feature groups: **no additional features** (a valid configuration). Parameters: Advanced."
    parts = [f"{FEATS.GROUPS[n].label} ({'history only' if FEATS.GROUPS[n].role == 'past' else 'known in advance'})"
             for n in d["features"]]
    return (method + "Feature groups: " + "; ".join(parts) + ". Parameters: Advanced. Inference only: no weights change; "
            "feature versions and parameters enter the cache key and provenance.")


def _view_systems(a: dict, variable: str) -> tuple[list[str], list[str], list[str]]:
    """(choices, defaults, probabilistic) for the applied result's view."""
    run, e = results.RUNS[a["suite"]], experiment_view(a["suite"], a["size"], a["setup"] or e_main(a["suite"]))
    saved = _saved_models(run, variable)
    exploratory = [m for m, frame in a.get("exploratory", {}).items() if variable in set(frame["variable"])]
    defaults = [m for m in e["default_systems"] if m in saved]
    if a["source"] == "saved" and a["primary"] in saved and a["primary"] not in defaults:
        defaults.append(a["primary"])
    if a["source"] == "exploratory":
        defaults.append(a["primary"])
    if a["config"].prediction_target == "arome_error":
        defaults += [m for m in ("arome-raw", a.get("comparison")) if m in saved + exploratory and m not in defaults]
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


def select_systems(a: dict, variable: str, pref_by: dict | None, band: str | None, keep_band: bool = False):
    """(choices, shown overlays, probabilistic shown, interval source). For a new forecast the applied primary
    system is the interval source when it is probabilistic and shown; a presentation-only Run keeps the
    selected source when it is still shown."""
    choices, defaults, prob = _view_systems(a, variable)
    p = (pref_by or {}).get(a["suite"]) or {}
    value = [m for m in choices if (m in defaults and m not in p.get("removed", [])) or m in p.get("added", [])]
    shown_prob = [m for m in value if m in prob]
    if keep_band and (band in shown_prob or band == ""):
        return choices, value, shown_prob, band
    if a["primary"] in shown_prob:
        band = a["primary"]
    elif band not in shown_prob:
        band = shown_prob[0] if shown_prob else ""
    return choices, value, shown_prob, band


@zerogpu.gpu(duration=60)
def _exploratory_on_device(obs, case, cfg, kw):
    """One labelled Timestral run; on ZeroGPU this function holds a GPU for its duration."""
    from weather_lab.forecast import exploratory_forecast
    from weather_lab.models import MODELS

    device = zerogpu.inference_device()
    try:
        if cfg.retrieval_method != "none":
            from weather_lab.rag_forecast import forecast

            return forecast(case, cfg, device=device)
        return exploratory_forecast(obs, case, list(cfg.targets), cfg.model, cfg.context_days, cfg.calendar,
                                    allow_download=False, device=device, **kw)
    finally:
        if device == "cuda" and zerogpu.on_zerogpu():
            MODELS.release("cuda")   # the GPU is detached after this call; never keep GPU-resident state


ERROR_MODES = [("Absolute |forecast − observation|", "absolute"), ("Signed (forecast − observation)", "signed"),
               ("Off", "off")]


def default_draft(suite: str) -> dict:
    """The forecast block's initial values for an experiment (Paris-Orly, Timestral-209M, main AROME setup)."""
    st = default_station(suite)
    cond = e_main(suite) or "none"
    g = FEATS.GROUPS
    return make_draft(suite, st, default_case(suite, st), "t", DEFAULT_SIZE, cond, [], CALENDAR,
                      default_context(suite, DEFAULT_SIZE, cond), results.RUNS[suite].horizon_hours,
                      list(g["resid"].defaults["variables"]), list(g["atmos"].defaults["fields"]),
                      list(g["tend"].defaults["lags_h"]), [], "", "absolute", "weather")


def initial_applied(suite: str) -> dict | None:
    """The applied state at page load: the saved forecast of the default configuration (shown once)."""
    if results.RUNS[suite].run.get("exploration"):
        return None
    d = default_draft(suite)
    cfg, _ = draft_config(d)
    mid = saved_match(d, cfg)
    if mid:
        cfg = from_system(next(s for s in results.RUNS[suite].run["systems"] if s["id"] == mid), cfg.horizon_h, cfg.step_hours)
    a = present(make_applied(d, cfg, "saved", mid or primary_saved(suite, d["size"], d["conditioning"]) or ""), d, None)
    a["draft"] = {**d, "overlays": sorted(a["view"]["shown"]), "band": a["view"]["band"]}
    return a


def on_experiment(suite: str, variable: str | None, station: str | None = None, size: str | None = None,
                  conditioning: str | None = None):
    """Experiment changed: draft only. Keep the user's variable, station, model and conditioning where the new
    experiment has them; context and horizon take the new run's recorded values. Nothing is applied."""
    vchoices = variable_choices(suite)
    variable = variable if variable in [v for _, v in vchoices] else "t"
    stations = station_choices(suite)
    station = station if station in [s for _, s in stations] else default_station(suite)
    size = size if size in {value for _, value in FORECAST_MODELS} else DEFAULT_SIZE
    conditioning = conditioning if conditioning in AROME_MODES else (e_main(suite) or "none")
    return (gr.update(choices=vchoices, value=variable),
            gr.update(choices=stations, value=station),
            gr.update(value=calendar_value(results.RUNS[suite], default_case(suite, station)), info=calendar_info(results.RUNS[suite], station)),
            scope_html(suite), gr.update(value=default_context(suite, size, conditioning)),
            gr.update(value=results.RUNS[suite].horizon_hours))
