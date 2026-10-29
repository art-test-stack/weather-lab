"""Independent development Benchmark execution and saved-result presentation."""

from __future__ import annotations

from weather_lab_app import zerogpu, plots, forecast_state as forecast, results

import threading
from pathlib import Path
import gradio as gr
import pandas as pd
from weather_lab_app.labels import CHECKPOINTS, DIFFERENT_INPUT, EXPERIMENTS, SEASON_NAMES, SETUP_NAMES, SPLIT_NAMES, model_name, unit, variable_name
from weather_lab_app.results import arome_history_text, input_history, station_label, training_status, load_run
from weather_lab.forecast_config import ForecastConfig, check, describe
import copy
from weather_lab import development, features, models
from weather_lab.common import read_json


KEYS = ("suite", "size", "prediction_target", "conditioning", "features", "calendar", "ctx", "horizon",
        "resid_vars", "atmos_fields", "tend_lags", "variable", "metric", "lead", "split", "station", "season",
        "statuses", "lead_mode", "saved", "retrieval_method", "retrieval_index", "precision")


_LOCK = threading.Lock()


def draft(*values):
    return copy.deepcopy(dict(zip(KEYS, values)))


def configuration(d):


    model = "t0-gamma" if d["size"] == "t0-gamma" else f"Timestral-{d['size']}"
    return ForecastConfig.make(model, d["conditioning"], forecast.feature_spec(d),
                               calendar=bool(d["calendar"]), context_days=int(d["ctx"]), horizon_h=int(d["horizon"]),
                               step_hours=results.RUNS[d["suite"]].step_hours,
                               targets=("t", "pmer") if d["prediction_target"] == "arome_error" else tuple(results.RUNS[d["suite"]].manifest["protocol"].get("variables", ("t", "pmer", "u", "v"))),
                               prediction_target=d["prediction_target"], retrieval_method=d.get("retrieval_method", "none"),
                               retrieval_index=(d.get("retrieval_index") or "") if d.get("retrieval_method", "none") != "none" else "",
                               precision=d.get("precision", "native"))


def defaults(suite):


    d = forecast.default_draft(suite)
    d.update(variable="t", metric="mae", lead="all", split=results.RUNS[suite].splits[-1], station="ALL", season="ALL",
             statuses=STATUSES, lead_mode="at", saved="", precision="native")
    return {k: copy.deepcopy(d[k]) for k in KEYS}


def choices(suite):
    return [("New development run", "")] + development.saved_runs(suite)


def index_choices(suite):
    from weather_lab.rag_store import Store


    store = Store()
    archive = results.RUNS[suite].run.get("archive_id")
    found = []
    for label, ident in store.list("indexes"):
        meta = read_json(store.root / "indexes" / ident / "index.json")
        if meta.get("archive_id") == archive and meta["retrieval"].get("stride_hours") == 1:
            found.append((label, ident))
    return found


def feedback(*values):


    *vals, applied, guard = values
    d = draft(*vals)
    pending = not applied or d != applied["draft"]
    issues = []
    try:
        cfg = configuration(d)
        issues = check(cfg, max_horizon_h=results.RUNS[d["suite"]].horizon_hours, context_range=forecast.CONTEXT_DAYS)
        if d["variable"] not in cfg.targets:
            raise ValueError("The selected forecast method does not target this weather variable")
    except (ValueError, TypeError) as exc:
        issues = [str(exc)]
    status = ("Changes pending — click Run benchmark. The displayed results are unchanged." if pending else
              "Settings match the displayed result. Run benchmark replays the selected saved run." if d["saved"] else
              "Selected model has not run. Click Run benchmark to evaluate all hourly windows." if applied and applied["kind"] == "reference" else
              "Settings match the displayed result. Rerun creates a new development run.")
    if issues:
        status += "\n\n" + "; ".join(i.text() if hasattr(i, "text") else i for i in issues)
    return status, gr.update(interactive=not issues and not guard["busy"],
                             value="Running…" if guard["busy"] else "Run benchmark" if pending or d["saved"] or (applied and applied["kind"] == "reference") else "Rerun")


def begin(*values):
    *vals, applied, guard = values
    with _LOCK:
        if guard["busy"]:
            raise gr.Error("A benchmark is already running in this session")
        guard["busy"] = True
    return {"draft": draft(*vals), "applied": applied}, gr.update(interactive=False, value="Running…"), "Running…"


@zerogpu.gpu(duration=60)
def predict_on_device(adapter, batches):
    adapter.device = zerogpu.inference_device()
    try:
        return development.predict_cases(adapter, batches)
    finally:
        if zerogpu.on_zerogpu():
            models.MODELS.release("cuda")


def execute(request, progress=gr.Progress()):


    d, previous = request["draft"], request["applied"]
    try:
        cfg = configuration(d)
        issues = check(cfg, max_horizon_h=results.RUNS[d["suite"]].horizon_hours, context_range=forecast.CONTEXT_DAYS)
        if issues or d["variable"] not in cfg.targets:
            raise ValueError("; ".join(i.text() for i in issues) or "This method does not target the selected variable")
        if d["saved"]:
            if d["saved"] not in {p for _, p in development.saved_runs(d["suite"])}:
                raise ValueError("Select a saved development run of this suite")
            try:
                rec = development.verify_run(Path(d["saved"]))
            except ValueError as exc:
                raise ValueError("A saved development artifact failed its checksum") from exc
            if rec["forecast_config"] != cfg.as_dict():
                raise ValueError("Settings differ from the selected saved run; choose New development run")
            path, kind, message = d["saved"], "development", "Replayed the saved development benchmark."
        elif (previous and previous["kind"] == "development" and previous["draft"]["suite"] == d["suite"]
              and previous["config"] == cfg and previous["draft"] != d):
            path, kind, message = previous["path"], "development", "Applied the display settings; no new inference."
        else:
            path = str(development.run(results.RUNS[d["suite"]].path, cfg, predict=predict_on_device, progress=progress))
            kind, message = "development", "Saved a separate development benchmark. Frozen results are unchanged."
        return {"draft": d, "config": cfg, "path": path, "kind": kind}, message
    except Exception as exc:
        return None, f"Benchmark failed: {type(exc).__name__}: {exc}. The displayed result is unchanged."


def apply(result, *values):
    *vals, previous, message = values
    if result is None:
        return previous, message, False
    if result["draft"] != draft(*vals):
        return previous, "Settings changed during execution. The run was saved; click Run benchmark to apply current settings.", False
    return result, message, True


def finish(*values):
    *vals, applied, guard = values
    with _LOCK:
        guard["busy"] = False
    d = draft(*vals)
    return *feedback(*values), gr.update(choices=choices(d["suite"]), value=d["saved"])


def on_suite(suite, variable):


    controls = bench_controls(suite)
    variables = forecast.variable_choices(suite)
    return (gr.update(choices=variables, value=variable if variable in dict((v, k) for k, v in variables) else "t"),
            *controls, gr.update(choices=choices(suite), value=""),
            gr.update(maximum=results.RUNS[suite].horizon_hours, value=results.RUNS[suite].horizon_hours))


def on_saved(path, suite):
    if not path:
        return tuple(gr.skip() for _ in range(13))
    if path not in {p for _, p in development.saved_runs(suite)}:
        raise gr.Error("Select a saved development run of this suite")
    c = read_json(Path(path) / "run.json")["forecast_config"]
    spec = {f["name"]: f["params"] for f in c["features"]}
    size = "t0-gamma" if c["model"] == "t0-gamma" else c["model"].removeprefix("Timestral-")
    return (size, c["prediction_target"], c["conditioning"], list(spec), c["calendar"],
            c["context_days"], c["horizon_h"], spec.get("resid", features.GROUPS["resid"].defaults)["variables"],
            spec.get("atmos", features.GROUPS["atmos"].defaults)["fields"],
            spec.get("tend", features.GROUPS["tend"].defaults)["lags_h"], c.get("retrieval_method", "none"), c.get("retrieval_index", ""),
            c.get("precision", {}).get("policy", "native"))


def render(applied, changed=True):


    if not changed:
        return tuple(gr.skip() for _ in range(19))
    d = applied["draft"]
    run = load_run(Path(applied["path"]))
    if applied["kind"] == "frozen":
        out = benchmark(d["suite"], d["variable"], d["metric"], d["lead"], d["split"], d["station"], d["season"],
                            d["statuses"], d["size"], d["conditioning"], d["lead_mode"])
        matrix = matrix_html(d["suite"], d["variable"], d["split"], d["metric"])
        label = "Frozen saved results. Run benchmark executes the separate configuration above."
    else:
        out = benchmark(d["suite"], d["variable"], d["metric"], d["lead"], d["split"], d["station"], d["season"],
                            d["statuses"], d["size"], d["conditioning"], d["lead_mode"], run=run)
        matrix = "Development run: one selected configuration. " + describe(applied["config"])
        label = "" if applied["kind"] == "reference" else (f"Development benchmark · {run.run['run_id']} · {describe(applied['config'])} · "
                 f"checkpoint revision {run.run['checkpoint']['revision'][:12]}")
    if applied["kind"] == "reference":
        matrix = "Reference forecasts only. The selected foundation model has not run."
        label = "2026 hourly benchmark · persistence reference · selected model not run"
    advanced = advanced_tables(d["suite"], d["variable"], d["split"], run=run)
    if applied["kind"] in ("development", "reference"):
        advanced = (pd.read_csv(run.path / "case_availability.csv", dtype={"station": str}), *advanced[1:3],
                    "Reference timings were copied from the source run. Development-model timings are not measured "
                    "separately; these are not comparable hardware measurements.")
    return (label, *out, matrix, *advanced, str(run.path / "metrics_point.csv"),
            str(run.path / "metrics_probabilistic.csv"), str(run.path / "predictions.parquet"),
            str(run.path / "manifest.json"), str(run.path / "run.json"),
            str(run.path / "case_availability.csv") if applied["kind"] in ("development", "reference") else None)


STATUSES = ["pretrained, no local fine-tuning", "fitted on fitting period", "external NWP (issued run)",
            "post-processing of AROME (recent errors)", "no fitting"]


LEADS = [("All lead times", "all"), ("6 h ahead", "6"), ("24 h ahead", "24"), ("48 h ahead", "48")] + [
    (f"{h} h ahead", str(h)) for h in range(3, 49, 3) if h not in (6, 24, 48)]


METRIC_CHOICES = [("Mean absolute error", "mae"), ("RMSE", "rmse"), ("Pinball loss", "pinball"),
                  ("80% range coverage", "coverage80"), ("90% range coverage", "coverage90"),
                  ("80% range width", "width80"), ("90% range width", "width90")]


LEAD_VIEWS = [("At each lead", "at"), ("Cumulative: all leads up to H", "cumulative")]


def bench_controls(suite: str):
    run = results.RUNS[suite]
    splits = run.splits
    stations = [("All stations", "ALL")] + forecast.station_choices(suite)
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


def finding(suite: str, variable: str, split: str | None = None, setup: str | None = None,
            metric: str = "mae") -> str:
    """One sentence on the AROME comparison for the selected input setup, built only from clauses the saved
    all-lead scores support (no assumption that a larger checkpoint or more inputs must help)."""
    run, e = results.RUNS[suite], EXPERIMENTS[suite]
    if "arome-raw" not in run.models or metric not in ("mae", "rmse"):
        return ""
    mx = e.get("matrix")
    setup = setup if mx and setup in mx["setups"] else (mx or {}).get("main", "archived")
    split = split or run.splits[-1]
    sel = lambda d: d[(d["split"] == split) & (d["variable"] == variable) & (d["station"] == "ALL")  # noqa: E731
                      & (d["season"] == "ALL") & d["lead_h"].isna()]
    allmae = sel(run.tables["metrics_point"]).set_index("model")[metric]
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
               forecast.matrix_id(suite, s, setup) or f"timestral-{s.lower()}-mv-cal-arome-archived") for s in sizes}
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
    score_name = "mean absolute error" if metric == "mae" else "root mean squared error"
    return (f"<div class='finding'><b>In {sample}</b> ({variable_name(variable, unit=False).lower()}, "
            f"{score_name} over all stations and lead times): {'; '.join(clauses)}.{' ' if notes else ''}"
            f"{' '.join(notes)}</div>")


def lead_view(suite, variable, metric, station, split, statuses, size=None, setup=None, cumulative=False, run=None):
    """Score against lead, recomputed from saved predictions on the matched cases of the compared systems."""
    from weather_lab.horizon import by_lead, lead_ranges, scored_rows

    run, e = run or results.RUNS[suite], forecast.experiment_view(suite, size, setup)
    if run.run.get("development") or run.run.get("hourly_benchmark"):
        e = {**e, "headline": run.models,
             "reference": "arome-raw" if "arome-raw" in forecast._saved_models(run, variable) else "persistence"}
    split = split or run.splits[-1]
    all_models = forecast._saved_models(run, variable)
    models = [m for m in all_models if training_status(m) in (statuses or STATUSES)]
    kind = plots.METRICS[metric][2]
    if kind == "prob":
        preds = run.predictions()
        prob = set(preds.loc[preds["output_type"] == "quantiles", "model"])
        models = [m for m in models if m in prob]
        all_models = [m for m in all_models if m in prob]
    if not models:
        return plots.metric_by_lead(pd.DataFrame(columns=["model"]), metric, variable, []), ""
    rows = scored_rows(run.predictions(), run.targets(), run.cases(), variable,
                       all_models if run.run.get("development") or run.run.get("hourly_benchmark") else models, metric, station, split)
    if rows.empty:
        return plots.metric_by_lead(pd.DataFrame(columns=["model"]), metric, variable, []), "No matched observed pairs."
    table = by_lead(rows, metric, cumulative=bool(cumulative))
    table = table[table.model.isin(models)]
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


def matrix_html(suite: str, variable: str, split: str | None = None, metric: str = "mae") -> str:
    """Input setup x checkpoint: saved scores, 'not run' (possible, no result) or 'unsupported input'."""
    from weather_lab.models import FAMILIES

    run, e = results.RUNS[suite], EXPERIMENTS[suite]
    mx = e.get("matrix")
    if not mx:
        return "<div class='note'>This experiment has no input-setup matrix.</div>"
    split = split or run.splits[-1]
    sel = lambda d: d[(d["split"] == split) & (d["variable"] == variable) & (d["station"] == "ALL")  # noqa: E731
                      & (d["season"] == "ALL") & d["lead_h"].isna()].set_index("model")
    pt, pr = sel(run.tables["metrics_point"]), sel(run.tables["metrics_probabilistic"])
    metric_name, _, kind = plots.METRICS[metric]
    scores = pt if kind == "point" else pr
    u = unit(variable)
    planned = [f for f in FAMILIES.values() if f.status == "planned"]
    head = "".join(f"<th>Timestral {s}</th>" for s in mx["sizes"]) + "".join(f"<th>{f.name}</th>" for f in planned)
    body = []
    for key in mx["setups"]:
        cells = []
        for s in mx["sizes"]:
            mid = forecast.matrix_id(suite, s, key)
            if mid in pt.index:
                value = scores.loc[mid, metric] if mid in scores.index else None
                score = ("not applicable" if value is None or pd.isna(value) else f"{value:.0%}"
                         if metric.startswith("coverage") else f"{value:.2f} {u}")
                prob = (f" · pinball {pr.loc[mid, 'pinball']:.2f} · 80% cov. {pr.loc[mid, 'coverage80']:.0%}"
                        if kind == "point" and mid in pr.index else "")
                cells.append(f"<td title='{mid}'>{metric_name}: {score}{prob}</td>")
            else:
                cells.append(f"<td title='{mid}'><i>not run</i></td>")
        for f in planned:
            ok = key == "none" or f.known_covariates
            cells.append(f"<td><i>{'not integrated yet' if ok else 'unsupported input'}</i></td>")
        ih = input_history(run)
        ids = [forecast.matrix_id(suite, s, key) for s in mx["sizes"]]
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
              lead_mode="at", run=None):
    run, e = run or results.RUNS[suite], forecast.experiment_view(suite, size, setup)
    development = run.run.get("development") or run.run.get("hourly_benchmark")
    if development:
        e = {**e, "reference": "arome-raw" if "arome-raw" in forecast._saved_models(run, variable) else "persistence"}
    cumulative = lead_mode == "cumulative"
    split = split or run.splits[-1]
    kind = plots.METRICS[metric][2]
    src = run.tables["metrics_point" if kind == "point" else "metrics_probabilistic"]
    keep = lambda d: d[d["model"].map(training_status).isin(statuses or STATUSES)]  # noqa: E731
    t = keep(_select(src, split, variable, station, season, lead))
    ref = e["reference"] if e["reference"] in set(t["model"]) else None
    hatched = frozenset(m for m in t["model"] if m in DIFFERENT_INPUT)
    bars = plots.comparison_bars(t, metric, variable, ref, hatched)
    lines, horizon_txt = lead_view(suite, variable, metric, station, split, statuses, size, setup, cumulative, run)
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
    result_finding = ("<div class='finding'>Development benchmark on the selected suite's cases; "
                      "no local fine-tuning or held-out performance claim.</div>" if development else
                      finding(suite, variable, split, setup, metric))
    return (result_finding, summary, bars, lines, horizon_txt,
            *detail_tables(run, split, variable, station, season, lead, statuses, metric))


def detail_tables(run, split, variable, station, season, lead, statuses, metric="mae"):
    u = unit(variable)
    keep = lambda d: d[d["model"].map(training_status).isin(statuses or STATUSES)]  # noqa: E731
    primary = metric if metric in ("mae", "rmse") else "mae"
    p = keep(_select(run.tables["metrics_point"], split, variable, station, season, lead)).sort_values(primary)
    ih = input_history(run)
    hist = lambda m: ih.loc[m, "summary"] if m in ih.index else "Unknown"  # noqa: E731
    ahist = lambda m: arome_history_text(ih.loc[m]) if m in ih.index else "Unknown"  # noqa: E731
    point = pd.DataFrame({"System": p["model"].map(model_name), f"MAE ({u})": p["mae"].round(2),
                          f"RMSE ({u})": p["rmse"].round(2), f"Bias ({u})": p["bias"].round(2),
                          "vs persistence": p["skill_mae_vs_persistence"].map(lambda x: f"{x:+.0%}"),
                          "Pairs": p["n"], "Input history": p["model"].map(hist),
                          "AROME history": p["model"].map(ahist)})
    if primary == "rmse":
        point = point[["System", f"RMSE ({u})", f"MAE ({u})", *point.columns[3:]]]
    q = keep(_select(run.tables["metrics_probabilistic"], split, variable, station, season, lead)).sort_values(
        "pinball")
    prob = pd.DataFrame({"System": q["model"].map(model_name), f"Pinball ({u})": q["pinball"].round(3),
                         "80% coverage": q["coverage80"].map("{:.0%}".format),
                         "90% coverage": q["coverage90"].map("{:.0%}".format),
                         f"80% width ({u})": q["width80"].round(1), "Pairs": q["n"],
                         "Input history": q["model"].map(hist)})
    return point, prob


def advanced_tables(suite, variable, split, run=None):
    run = run or results.RUNS[suite]
    split = split or run.splits[-1]
    a = run.tables["availability"]
    a = a[(a["split"] == split) & (a["variable"] == variable)]
    avail = pd.DataFrame({"System": a["model"].map(model_name), "Cases": a["cases_in_suite"],
                          "Eligible": a["cases_eligible"], "Targets observed": a["observed_targets"],
                          "Failed": a["failed_predictions"], "Scored pairs": a["valid_pairs"]})
    d = run.tables["paired_differences"]
    d = d.reindex(columns=["split", "variable", "model", "reference", "loss", "lead_h", "mean_diff",
                           "ci95_low", "ci95_high", "n_blocks"])
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
