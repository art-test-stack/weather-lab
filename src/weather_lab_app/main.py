"""Weather-lab: Gradio explorer for saved benchmark results.

Reads saved runs only (runs/<suite>/<latest> when present, otherwise the packaged
offline example in data/examples/). It never runs the benchmark. The exploratory
panel runs a single labelled Timestral forecast outside the frozen benchmark.
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
from weather_lab_app.results import (SUITES, available_runs, case_forecast, filter_metrics, forecast_csv,
                                 station_label, training_status)



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
STATUSES = ["pretrained, no local fine-tuning", "fitted on fitting period", "external NWP (issued run)", "no fitting"]
POINT_COLS = ["model", "training", "n", "mae", "rmse", "bias", "mae_ref", "skill_mae_vs_persistence"]
PROB_COLS = ["model", "training", "n", "pinball", "skill_pinball_vs_climatology", "coverage80", "coverage90",
             "width80", "width90"]

CSS = """
.gradio-container {max-width: 1280px !important; margin: auto;}
#hero h1 {font-size: 1.7rem; margin-bottom: .2rem;}
#hero p {color: #52514e; margin-top: 0;}
.badge {display:inline-block; padding:2px 8px; border-radius:10px; background:#eef4fb; color:#184f95;
        font-size:.8rem; margin-right:6px;}
.warn {display:inline-block; padding:2px 8px; border-radius:10px; background:#fff4e0; color:#7a4b00; font-size:.8rem;}
"""


HEADERS = {"mae": "MAE", "rmse": "RMSE", "mae_ref": "MAE persistence", "skill_mae_vs_persistence": "MAE skill",
           "pinball": "pinball", "skill_pinball_vs_climatology": "pinball skill", "coverage80": "cov. 80%",
           "coverage90": "cov. 90%", "width80": "width 80%", "width90": "width 90%"}


def _round(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[[c for c in cols if c in df]].copy()
    return out.round(3).rename(columns=HEADERS)


def suite_header(suite: str) -> str:
    run = RUNS[suite]
    m = run.manifest
    tag = "integration study" if suite.startswith("arome-issued") else "integration check (smoke)"
    return (f"<span class='badge'>{SUITES[suite]}</span><span class='warn'>{tag}: not a performance claim</span>"
            f"<br><small>run <code>{run.run['run_id']}</code> · {run.source} · "
            f"{m['n_eligible']}/{m['n_cases']} eligible cases · {len(m['protocol']['stations'])} stations · "
            f"leads 3-48 h · manifest protocol <code>{m['protocol_sha256']}</code> · local fine-tuning: none</small>")


# --------------------------------------------------------------------------- scorecard callbacks

def scorecard_controls(suite: str):
    run = RUNS[suite]
    stations = ["ALL"] + run.manifest["protocol"]["stations"]
    splits = run.splits
    point = run.tables["metrics_point"]
    variables = [v for v in VARIABLES if v in set(point["variable"])]
    return (gr.update(choices=splits, value=splits[-1], interactive=len(splits) > 1),
            gr.update(choices=variables, value="t"),
            gr.update(choices=[(station_label(s) if s != "ALL" else "All stations", s) for s in stations], value="ALL"),
            gr.update(choices=sorted(set(point["season"])), value="ALL"),
            suite_header(suite))


def scorecard(suite, split, variable, station, season, lead, statuses):
    run = RUNS[suite]
    point = filter_metrics(run.tables["metrics_point"], split, variable, station, season, lead, statuses)
    prob = filter_metrics(run.tables["metrics_probabilistic"], split, variable, station, season, lead, statuses)
    point_all = filter_metrics(run.tables["metrics_point"], split, variable, station, season, "all", statuses)
    by_lead = run.tables["metrics_point"]
    by_lead = by_lead[(by_lead["split"] == split) & (by_lead["variable"] == variable) & (by_lead["station"] == station)
                      & (by_lead["season"] == season)]
    by_lead = by_lead[by_lead["model"].map(training_status).isin(statuses or STATUSES)]
    prob_all = filter_metrics(run.tables["metrics_probabilistic"], split, variable, station, season, "all", statuses)
    na = run.tables.get("probabilistic_not_applicable", pd.DataFrame())
    na = na[(na["split"] == split) & (na["variable"] == variable)] if len(na) else na
    na_md = "".join(f"\n- **{r.model}**: {r.note}" for r in na.itertuples()) if len(na) else ""
    avail = run.tables["availability"]
    avail = avail[(avail["split"] == split) & (avail["variable"] == variable)].drop(columns=["split", "variable"])
    paired = run.tables["paired_differences"]
    paired = paired[(paired["split"] == split) & (paired["variable"] == variable)].round(3)
    n_common = int(point_all["n"].max()) if len(point_all) else 0
    lead_txt = "all leads" if lead == "all" else f"lead {lead} h"
    summary = (f"**{VARIABLES[variable][0]} ({VARIABLES[variable][1]}) · {split} · {lead_txt}** — every row uses the "
               f"same common valid set of **{n_common if lead == 'all' else (int(point['n'].max()) if len(point) else 0)}** "
               f"(case, lead) pairs where the target was observed and every system forecast it. Point errors use the "
               f"median for quantile systems.")
    ci_note = ("Paired differences are model − reference on the common set; a 95% block-bootstrap interval "
               "(blocks of 2 consecutive origin days) is shown only with ≥ 10 blocks.")
    return (summary, _round(point, POINT_COLS), _round(prob, PROB_COLS),
            "**Probabilistic metrics not applicable:**" + na_md if na_md else "",
            plots.skill_vs_lead(by_lead, variable), plots.coverage(prob_all), avail, paired, ci_note,
            runtime_table(suite), case_table(suite, split, variable))


def runtime_table(suite: str) -> pd.DataFrame:
    run = RUNS[suite]
    rows = []
    for s in run.run["systems"]:
        rows.append({"model": s["id"], "output": s["output"], "revision": s["revision"],
                     "batch wall time (s)": s["wall_s"], "per case (ms)": s["per_case_ms"],
                     "warm single case (ms)": s.get("warm_single_case_ms_median"),
                     "first load (s)": s.get("first_load_s"), "parameters": s.get("n_parameters"),
                     "process peak RSS after (MB)": s["peak_rss_mb_after"]})
    return pd.DataFrame(rows)


def hardware_note(suite: str) -> str:
    env = RUNS[suite].run["environment"]
    return (f"Measured on {env.get('cpu')} ({env['machine']}), torch {env['torch']} with {env['torch_threads']} threads, "
            f"batch size {RUNS[suite].run['systems'][-1].get('batch_size', 'n/a')}, context "
            f"{RUNS[suite].run['history_steps']} steps. Peak RSS is process-wide and cumulative across systems.")


def case_table(suite: str, split: str, variable: str) -> pd.DataFrame:
    cs = RUNS[suite].tables["case_scores"]
    cs = cs[(cs["split"] == split) & (cs["variable"] == variable)]
    if cs.empty:
        return pd.DataFrame()
    piv = cs.pivot_table(index=["case_id", "station", "season"], columns="model", values="mae").round(2).reset_index()
    return piv


def downloads(suite: str):
    p = RUNS[suite].path
    return str(p / "metrics_point.csv"), str(p / "metrics_probabilistic.csv"), str(p / "manifest.json")


# --------------------------------------------------------------------------- explorer callbacks

def explorer_controls(suite: str, case_id: str | None = None):
    run = RUNS[suite]
    cases = run.cases()
    cases = cases[cases["eligible"]]
    examples = run.manifest.get("ui_examples") or list(cases["case_id"][:1])
    case_id = case_id if case_id in set(cases["case_id"]) else examples[0]
    station = case_id.split("/")[1]
    preds = run.predictions()
    variables = [v for v in VARIABLES if v in set(preds["variable"])]
    systems = run.models
    default_sys = [m for m in systems if m in ("persistence", "climatology", "arome-raw", "timestral-28m-mv-cal",
                                               "timestral-28m-mv-cal-arome", "timestral-28m-mv-cal-arome-archived")]
    prob_models = [m for m in systems if (preds.loc[preds["model"] == m, "output_type"] == "quantiles").any()]
    band_default = next((m for m in ("timestral-28m-mv-cal-arome-archived", "timestral-28m-mv-cal-arome",
                                     "timestral-28m-mv-cal") if m in prob_models), prob_models[0])
    st_cases = cases[cases["station"] == station]
    return (gr.update(choices=[(station_label(s), s) for s in run.manifest["protocol"]["stations"]], value=station),
            gr.update(choices=_case_choices(st_cases, examples), value=case_id),
            gr.update(choices=variables, value="t"),
            gr.update(choices=systems, value=default_sys),
            gr.update(choices=[("none", "")] + prob_models, value=band_default))


def _case_choices(cases: pd.DataFrame, examples: list[str]) -> list[tuple[str, str]]:
    return [(f"{c.origin:%Y-%m-%d %H}Z · {c.split} · {c.season}" + ("  ★ example" if c.case_id in examples else ""),
             c.case_id) for c in cases.sort_values("origin").itertuples()]


def station_changed(suite: str, station: str):
    run = RUNS[suite]
    cases = run.cases()
    cases = cases[cases["eligible"] & (cases["station"] == station)]
    examples = run.manifest.get("ui_examples") or []
    first = next((e for e in examples if e.split("/")[1] == station), cases["case_id"].iloc[0])
    return gr.update(choices=_case_choices(cases, examples), value=first)


def explore(suite, case_id, variable, systems, band_model, extra=None):
    run = RUNS[suite]
    rows, obs, case = case_forecast(run, case_id, variable)
    if extra is not None and len(extra):
        extra = extra[extra["variable"] == variable]
    fig = plots.forecast_figure(rows, obs, case, variable, systems or [], band_model or None, extra=extra)
    sc = rows[rows["model"].isin(systems or [])].merge(obs.rename(columns={"time": "valid_time"}), on="valid_time")
    sc = sc.assign(abs_err=(sc["point"] - sc["observed"]).abs()).groupby("model")["abs_err"].agg(["mean", "count"])
    sc = sc.rename(columns={"mean": "case MAE", "count": "verified leads"}).round(2).reset_index()
    info = (f"**Case** `{case_id}` · {case['split']} split · season {case['season']} · inputs: observations at or "
            f"before the origin" + (" + issued AROME run (known-future covariate for the conditioned system)"
                                    if isinstance(case.get("arome_run"), str) else "")
            + ". Bands are drawn only for probabilistic systems; deterministic systems (persistence, seasonal naive, "
              "raw AROME) have none. Quantiles are marginal per lead, not coherent scenarios.")
    path = Path(tempfile.gettempdir()) / f"forecast_{case_id.replace('/', '_')}_{variable}.csv"
    forecast_csv(rows, obs).to_csv(path, index=False)
    return fig, plots.station_map(run.stations(), case["station"]), sc, info, str(path)


@zerogpu.gpu(duration=60)
def _exploratory_on_device(obs, case, size, context_days, calendar, allow_download):
    """One labelled Timestral run; on ZeroGPU this function holds a GPU for its duration."""
    from weather_lab.forecast import exploratory_forecast

    return exploratory_forecast(obs, case, ["t", "pmer", "u", "v"], size, context_days, calendar,
                                allow_download=allow_download, device=zerogpu.inference_device())


def run_exploratory(suite, case_id, variable, systems, band_model, size, context_days, calendar, allow_download):
    run = RUNS[suite]
    _, _, case = case_forecast(run, case_id, variable)
    obs = run.observations()
    obs = obs[obs["station"] == case["station"]]
    try:
        from weather_lab.forecast import ensure_checkpoint

        # download on CPU first, so a cold 837 MB fetch never consumes the GPU window or quota
        ensure_checkpoint(size, bool(allow_download))
        extra = _exploratory_on_device(obs, case, size, int(context_days), bool(calendar), bool(allow_download))
        msg = (f"Exploratory run **outside the frozen benchmark**: {extra['model'].iloc[0]} "
               f"(`{extra['model_revision'].iloc[0]}`). Not saved, not scored.")
    except (PermissionError, ValueError) as e:
        extra, msg = None, f"⚠️ {e}"
    except Exception as e:  # noqa: BLE001 - e.g. ZeroGPU quota/queue errors raised from the GPU worker
        extra, msg = None, f"⚠️ exploratory run failed: {type(e).__name__}: {e}"
    fig, *_ = explore(suite, case_id, variable, systems, band_model, extra)
    return fig, msg


def open_case(suite: str, case_id: str):
    return (gr.Tabs(selected="explorer"), gr.update(value=suite), *explorer_controls(suite, case_id))


# --------------------------------------------------------------------------- layout

def methods_md() -> str:
    proto = (ROOT / "docs" / "experiment_protocol.md")
    text = proto.read_text() if proto.exists() else "See docs/experiment_protocol.md."
    prov = ["\n\n## Runs shown in this app\n", "| suite | run | source | code | Timestral package | checkpoint |",
            "|---|---|---|---|---|---|"]
    for suite, run in RUNS.items():
        ckpt = sorted({s["revision"] for s in run.run["systems"] if s["kind"] == "timestral"})
        prov.append(f"| {suite} | `{run.run['run_id']}` | {run.source} | `{run.run['code_revision'][:12]}` | "
                    f"`{run.run['timestral_package'][:12]}` | {', '.join(ckpt)} |")
    return text + "\n".join(prov)


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
        # restart the process so the full explorer is built from the fetched artifacts
        threading.Timer(1.0, lambda: os.execv(sys.executable, [sys.executable] + sys.argv)).start()
        return "Setup complete. Restarting the app: reload this page in a few seconds."

    with gr.Blocks(title="Weather-lab: setup") as ui:
        gr.Markdown("# Weather-lab\nThe app fetches its data snapshot and model weights at startup.")
        status = gr.Markdown(summarize_failure(state))
        gr.Button("Retry setup", variant="primary").click(retry, None, status, concurrency_limit=1)
    return ui


def build() -> gr.Blocks:
    with gr.Blocks(title="Weather-lab") as demo:
        if DEFAULT_SUITE is None:
            gr.Markdown("# Weather-lab\nNo saved runs found. Run `uv run weather-lab evaluate --config "
                        "configs/smoke.yaml` or restore `data/examples/`.")
            return demo
        gr.Markdown("# Weather-lab\nShort-range **probabilistic weather forecasting** at French SYNOP "
                    "stations: Mistral's Timestral (zero-shot, no local fine-tuning) against persistence, "
                    "climatology and issued AROME forecasts, scored on frozen case sets.", elem_id="hero")
        suite_choices = [(SUITES[s], s) for s in RUNS]
        with gr.Tabs(selected="scorecard") as tabs:
            # ---------------- scorecard
            with gr.Tab("Benchmark scorecard", id="scorecard"):
                with gr.Row():
                    suite = gr.Radio(suite_choices, value=DEFAULT_SUITE, label="Suite (scored separately, never pooled)")
                header = gr.HTML()
                with gr.Row():
                    split = gr.Dropdown(label="Split", choices=[])
                    variable = gr.Dropdown(label="Variable", choices=[])
                    station = gr.Dropdown(label="Station", choices=[])
                    season = gr.Dropdown(label="Season", choices=[])
                    lead = gr.Dropdown(["all", "6", "24", "48"] + [str(h) for h in range(3, 49, 3)
                                                                  if h not in (6, 24, 48)],
                                       value="all", label="Lead (h)")
                statuses = gr.CheckboxGroup(STATUSES, value=STATUSES, label="Fitting / adaptation status")
                summary = gr.Markdown()
                with gr.Row(equal_height=True):
                    skill_plot = gr.Plot(label="Skill vs lead")
                    cov_plot = gr.Plot(label="Coverage")
                gr.Markdown("### Point scorecard (all systems)")
                point_tab = gr.Dataframe(interactive=False, wrap=True)
                gr.Markdown("### Probabilistic scorecard (quantile systems only)")
                prob_tab = gr.Dataframe(interactive=False, wrap=True)
                na_md = gr.Markdown()
                with gr.Accordion("Availability: expected, eligible, observed, failed", open=False):
                    avail_tab = gr.Dataframe(interactive=False)
                with gr.Accordion("Paired differences vs reference", open=False):
                    ci_md = gr.Markdown()
                    paired_tab = gr.Dataframe(interactive=False)
                with gr.Accordion("CPU latency and memory", open=False):
                    hw_md = gr.Markdown()
                    rt_tab = gr.Dataframe(interactive=False)
                gr.Markdown("### Scored cases · per-case MAE (common set) — pick one to open it in the explorer")
                cases_tab = gr.Dataframe(interactive=False, max_height=320)
                with gr.Row():
                    case_pick = gr.Dropdown(label="Case", choices=[], scale=3)
                    open_btn = gr.Button("Open in forecast explorer →", variant="primary", scale=1)
                with gr.Row():
                    dl_point = gr.DownloadButton("Point scores (CSV)")
                    dl_prob = gr.DownloadButton("Probabilistic scores (CSV)")
                    dl_manifest = gr.DownloadButton("Case manifest (JSON)")

            # ---------------- explorer
            with gr.Tab("Forecast explorer", id="explorer"):
                with gr.Row():
                    x_suite = gr.Radio(suite_choices, value=DEFAULT_SUITE, label="Suite")
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        x_map = gr.Plot(label="Stations")
                        x_station = gr.Dropdown(label="Station", choices=[])
                        x_case = gr.Dropdown(label="Historical forecast origin", choices=[])
                        x_var = gr.Dropdown(label="Variable", choices=[])
                        x_systems = gr.CheckboxGroup(label="Systems", choices=[])
                        x_band = gr.Dropdown(label="80%/90% bands for", choices=[])
                    with gr.Column(scale=3):
                        x_plot = gr.Plot(label="Forecast")
                        x_info = gr.Markdown()
                        x_scores = gr.Dataframe(interactive=False)
                        x_dl = gr.DownloadButton("Forecast for this case (CSV)")
                        with gr.Accordion("Exploratory run — outside the frozen benchmark", open=False):
                            gr.Markdown("Runs Timestral once on this case with custom settings. The result is "
                                        "labelled exploratory, never written to `runs/` and never scored.")
                            with gr.Row():
                                e_size = gr.Dropdown(["Timestral-5M", "Timestral-28M", "Timestral-209M"],
                                                     value="Timestral-28M", label="Checkpoint")
                                e_ctx = gr.Slider(7, 30, value=14, step=1, label="Context (days)")
                                e_cal = gr.Checkbox(value=True, label="Calendar covariates")
                                e_dl = gr.Checkbox(value=False, label="Allow weight download")
                            e_btn = gr.Button("Run exploratory forecast")
                            e_msg = gr.Markdown()

            # ---------------- methods
            with gr.Tab("Methods", id="methods"):
                gr.Markdown(methods_md())

        sc_inputs = [suite, split, variable, station, season, lead, statuses]
        sc_outputs = [summary, point_tab, prob_tab, na_md, skill_plot, cov_plot, avail_tab, paired_tab, ci_md,
                      rt_tab, cases_tab]

        def refresh_cases(s, sp, v):
            t = case_table(s, sp, v)
            ids = t["case_id"].tolist() if len(t) else []
            return gr.update(choices=ids, value=ids[0] if ids else None)

        suite.change(scorecard_controls, suite, [split, variable, station, season, header]).then(
            scorecard, sc_inputs, sc_outputs).then(downloads, suite, [dl_point, dl_prob, dl_manifest]).then(
            hardware_note, suite, hw_md).then(refresh_cases, [suite, split, variable], case_pick)
        for c in (split, variable, station, season, lead, statuses):
            c.input(scorecard, sc_inputs, sc_outputs)
        for c in (split, variable):
            c.input(refresh_cases, [suite, split, variable], case_pick)

        x_outputs = [x_plot, x_map, x_scores, x_info, x_dl]
        x_inputs = [x_suite, x_case, x_var, x_systems, x_band]
        x_suite.input(explorer_controls, x_suite, [x_station, x_case, x_var, x_systems, x_band]).then(
            explore, x_inputs, x_outputs)
        x_station.input(station_changed, [x_suite, x_station], x_case).then(explore, x_inputs, x_outputs)
        for c in (x_case, x_var, x_systems, x_band):
            c.input(explore, x_inputs, x_outputs)
        open_btn.click(open_case, [suite, case_pick], [tabs, x_suite, x_station, x_case, x_var, x_systems, x_band]
                       ).then(explore, x_inputs, x_outputs)
        e_btn.click(run_exploratory, x_inputs + [e_size, e_ctx, e_cal, e_dl], [x_plot, e_msg],
                    concurrency_limit=1)

        def init():
            return (*scorecard_controls(DEFAULT_SUITE),)

        demo.load(init, None, [split, variable, station, season, header]).then(
            scorecard, sc_inputs, sc_outputs).then(downloads, suite, [dl_point, dl_prob, dl_manifest]).then(
            hardware_note, suite, hw_md).then(refresh_cases, [suite, split, variable], case_pick).then(
            explorer_controls, x_suite, [x_station, x_case, x_var, x_systems, x_band]).then(
            explore, x_inputs, x_outputs)
    return demo


def init(run_bootstrap: bool = True) -> gr.Blocks:
    """Fetch/verify artifacts (unless disabled), load saved runs and build the UI."""
    global BOOT, RUNS, DEFAULT_SUITE
    BOOT = startup_bootstrap() if run_bootstrap else None
    RUNS = available_runs()
    DEFAULT_SUITE = "synop-historical-v1" if "synop-historical-v1" in RUNS else next(iter(RUNS), None)
    return failure_ui(BOOT) if (BOOT or {}).get("status") == "failed" and not RUNS else build()


def main() -> None:
    demo = init()
    demo.queue(default_concurrency_limit=4).launch(
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="stone",
                             font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]),
        css=CSS)


if __name__ == "__main__":
    main()
