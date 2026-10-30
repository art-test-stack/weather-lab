"""Weather-lab startup and Gradio application composition."""

from __future__ import annotations

from weather_lab_app import zerogpu, plots, forecast_state as forecast, forecast_view as view, benchmark as bench, results

import sys
from pathlib import Path
import gradio as gr
from weather_lab_app.labels import DEFAULT_EXPERIMENT
from weather_lab_app.results import available_runs, calendar_value, calendar_info, hourly_exploration
from weather_lab import development, features as FEATS
from weather_lab.forecast_config import PREDICTION_TARGETS


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


DEFAULT_SUITE: str | None = None
INTRO = ("Météo-France's **AROME** and pretrained time-series foundational models (TSFMs; inference only), "
         "including forecasts conditioned on AROME, at French stations. Hourly exploration covers 2022 "
         "through the latest recorded observations.")


FONT_DIR = Path(__file__).with_name("assets") / "fonts"


CSS = "\n".join(
    "@font-face {font-family: 'Latin Modern Roman'; "
    f"src: url('/gradio_api/file={FONT_DIR / ('lmroman10-' + face + '.otf')}') format('opentype'); "
    f"font-weight: {weight}; font-style: {style}; font-display: swap;}}"
    for face, weight, style in (("regular", 400, "normal"), ("bold", 700, "normal"),
                                ("italic", 400, "italic"), ("bolditalic", 700, "italic"))
) + """
.gradio-container {max-width: 1240px !important; margin: auto;}
#hero-header {position: relative;}
#hero {padding-right: 56px;}
#github-link {position: absolute; top: 0; right: 0; width: 44px; min-width: 44px;}
#hero h1 {font-size: 1.45rem; margin: 0 0 .1rem 0;}
#hero p {color: #3d3c39; margin: 0; font-size: .98rem;}
#github-link a {display: flex; align-items: center; justify-content: center; width: 44px; height: 44px;
                color: #0b0b0b; border-radius: 8px;}
#github-link a:hover, #github-link a:focus-visible {background: #e6e5e1;}
.benchmark-loading {background: #fff0ef; border: 1px solid #ebbbb6; border-radius: 12px; padding: 16px 20px;}
.benchmark-loading p, .benchmark-loading h3 {color: #7c2923;}
.benchmark-loading p {font-size: 1.1rem;}
.benchmark-loading h3 {font-size: 1.3rem;}
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
#applied .issues {font-size:.84rem; color:#8a1c1c; background:#fdeceb; border-radius:6px; padding:3px 8px;
                  margin-top:3px;}
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
    gr.set_static_paths(paths=[FONT_DIR])
    with gr.Blocks(title="Weather-lab") as demo:
        if DEFAULT_SUITE is None:
            gr.Markdown("# Weather-lab\nNo saved results found. Run `uv run weather-lab bootstrap`.")
            return demo
        with gr.Row(equal_height=False, elem_id="hero-header"):
            gr.Markdown(f"# Weather-lab\n{INTRO}", elem_id="hero", scale=1)
            gr.HTML('<a href="https://github.com/art-test-stack/weather-lab" target="_blank" '
                    'rel="noopener noreferrer" aria-label="Weather-lab on GitHub" title="View on GitHub">'
                    '<svg viewBox="0 0 24 24" width="28" height="28" fill="currentColor" aria-hidden="true">'
                    '<path d="M12 .297C5.37.297 0 5.67 0 12.297c0 5.303 3.438 9.8 8.205 11.385'
                    '.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.043-1.61-4.043-1.61'
                    '-.546-1.387-1.333-1.756-1.333-1.756-1.09-.745.083-.73.083-.73 1.205.084 1.838'
                    ' 1.237 1.838 1.237 1.07 1.835 2.809 1.305 3.495.998.108-.776.418-1.305.762'
                    '-1.605-2.665-.3-5.466-1.334-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303'
                    '-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02'
                    '.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12'
                    ' 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81'
                    ' 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092'
                    ' 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg></a>',
                    elem_id="github-link", scale=0, min_width=44)
        gr.Markdown("The tabs are:\n\n"
                    "- **Forecast:** Choose a window, a model to evaluate against raw AROME, model parameters, "
                    "feature engineering, and an output type, then run on the selected window.\n"
                    "- **Benchmark:** Visualize and compare saved model configurations against the AROME baseline "
                    "on forecast windows up to 48 hours.")
        if notice := fallback_notice(BOOT):
            gr.Markdown(f"**Note:** {notice}", elem_classes="note", elem_id="snapshot-fallback")
        a0 = forecast.initial_applied(DEFAULT_SUITE)
        d0 = a0["draft"] if a0 else forecast.default_draft(DEFAULT_SUITE)
        applied = gr.State(a0)
        submission = gr.State({"busy": False})
        captured_request = gr.State(None)
        pending_run = gr.State(None)
        changed = gr.State(True)
        g = FEATS.GROUPS
        with gr.Tabs(selected="overview"):
            # ---------------- forecast
            with gr.Tab("Forecast", id="overview"):
                exp = gr.Dropdown(forecast.experiment_choices(), value=DEFAULT_SUITE, label="Experiment")
                scope = gr.HTML(forecast.scope_html(DEFAULT_SUITE), elem_id="scope")
                with gr.Group(elem_id="forecast-panel"):
                    with gr.Row(equal_height=True):
                        station = gr.Dropdown(forecast.station_choices(DEFAULT_SUITE), value=d0["station"], label="Station",
                                              scale=2, min_width=160)
                        date = gr.DateTime(calendar_value(results.RUNS[DEFAULT_SUITE], d0["case"]), type="string", timezone="UTC",
                                           label="Forecast issued (UTC)", include_time=True,
                                           info=calendar_info(results.RUNS[DEFAULT_SUITE], d0["station"]), scale=3, min_width=210)
                        var = gr.Dropdown(forecast.variable_choices(DEFAULT_SUITE), value=d0["variable"],
                                          label="Weather variable", scale=2, min_width=150)
                        ckpt = gr.Radio(forecast.FORECAST_MODELS, value=d0["size"],
                                        label="Model (checkpoint)", scale=3, min_width=230)
                    method = gr.Radio([(label, key) for key, label in PREDICTION_TARGETS.items()],
                                      value=d0["prediction_target"], label="Forecast method")
                    from weather_lab.rag_store import Store

                    with gr.Row():
                        retrieval = gr.Radio([("None", "none"), ("RAG · levels", "level"), ("RAG · shape", "shape")],
                                             value="none", label="Retrieval augmentation")
                        retrieval_index = gr.Dropdown(Store().list("indexes"), label="Saved hourly retrieval index")
                        refresh_indexes = gr.Button("Refresh retrieval indexes", size="sm")
                    gr.Markdown("RAG uses the selected station/time with the index's hourly archive: **30-day context**, "
                                "**24 or 48 h**, **temperature/pressure**, direct or AROME-error prediction and no additional features. "
                                "AROME conditioning uses genuine issued runs. **AROME-error indexes require archived "
                                "conditioning** and retrieve only cases fully verified before this forecast. Missing "
                                "archive coverage is unavailable. Create an index or "
                                "install t0 weights (~1 GB) explicitly with the CLI. No automatic downloads.",
                                elem_classes="note")
                    with gr.Row(equal_height=True):
                        cond = gr.Dropdown(forecast.CONDITIONING_CHOICES, value=d0["conditioning"], label="AROME conditioning",
                                           scale=3, min_width=240)
                        feats_cb = gr.CheckboxGroup(forecast.FEATURE_CHOICES, value=d0["features"],
                                                    label="Additional feature groups (none = no additional features)",
                                                    scale=5, min_width=300)
                        calendar = gr.Checkbox(d0["calendar"], label="Calendar inputs (hour of day, time of year)",
                                               scale=2, min_width=170)
                    feat_note = gr.Markdown(forecast.feature_note(d0), elem_classes="note")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=240):
                            ctx = gr.Slider(*forecast.CONTEXT_DAYS, value=d0["ctx"], step=1, label="Observed context (days)")
                            ctx_info = gr.Markdown(forecast.context_md(d0["ctx"], DEFAULT_SUITE), elem_classes="note")
                        with gr.Column(scale=1, min_width=240):
                            h_upd, h_info0 = forecast.horizon_control(DEFAULT_SUITE, d0["case"], d0["conditioning"],
                                                             d0["horizon"])
                            horizon = gr.Slider(results.RUNS[DEFAULT_SUITE].step_hours, h_upd["maximum"], value=d0["horizon"], step=results.RUNS[DEFAULT_SUITE].step_hours,
                                                label="Forecast horizon (hours)")
                            horizon_info = gr.Markdown(h_info0, elem_classes="note")
                    main_metric = gr.Radio([("MAE", "mae"), ("RMSE", "rmse")], value=d0["metric"],
                                           label="Main metric", info="Both scores are shown; apply changes with Run forecast.")
                    with gr.Accordion("Advanced: feature parameters, overlays, uncertainty shading, error display",
                                      open=False, elem_id="advanced"):
                        with gr.Row():
                            resid_vars = gr.CheckboxGroup(list(g["resid"].choices["variables"]),
                                                          value=d0["resid_vars"], label="Recent AROME errors: variables")
                            atmos_fields = gr.CheckboxGroup([("dew point", "td"), ("cloud cover", "cloud")],
                                                            value=d0["atmos_fields"], label="Dew point / cloud: fields")
                            tend_lags = gr.CheckboxGroup([(f"{x} h", x) for x in g["tend"].choices["lags_h"]],
                                                         value=d0["tend_lags"], label="Tendencies: lags")
                        with gr.Row():
                            systems = gr.CheckboxGroup(label="Comparison overlays", choices=[], scale=3)
                            with gr.Column(scale=2, min_width=220):
                                band = gr.Dropdown(label="Shaded 80% / 90% range for", choices=[], value=None)
                                err_mode = gr.Radio(forecast.ERROR_MODES, value=d0["error_mode"],
                                                    label="Error panel under the chart")
                                show_old = gr.Checkbox(False, label="Show superseded experiment versions")
                    with gr.Row(equal_height=True):
                        e_btn = gr.Button("Run forecast", variant="primary", scale=1, min_width=160,
                                          elem_id="run-forecast")
                        e_msg = gr.Markdown(elem_classes="note", elem_id="run-msg")
                status = gr.HTML(forecast.status_html(a0, *[d0[k] for k in forecast.DRAFT_KEYS]), elem_id="applied")
                fplot = gr.Plot(show_label=False)
                tiles = gr.HTML()
                caption = gr.Markdown(elem_classes="note")
                with gr.Row():
                    with gr.Column(scale=2, min_width=280):
                        this_tab = gr.HTML()
                        dl_case = gr.DownloadButton("Download the applied forecast (CSV)", size="sm")
                    with gr.Column(scale=2, min_width=280):
                        sys_desc = gr.Markdown(elem_classes="note")
                    with gr.Column(scale=1, min_width=200):
                        smap = gr.Plot(show_label=False)

            # ---------------- benchmark
            with gr.Tab("Benchmark", id="benchmark"):
                gr.Markdown("### ⚠️ DATA LOADING\n"
                            "Wait 30 seconds to 1 minute for the benchmark results to load on the page.\n\n"
                            "Once the data and plots have loaded, the page layout may not adjust correctly, "
                            "preventing you from scrolling to the bottom. To fix this, open and close a "
                            "drop-down menu, such as one below a plot.", elem_classes="benchmark-loading")
                benchmark_suite = "hourly-2026-v3" if "hourly-2026-v3" in results.RUNS else next(
                    (s for s, r in results.RUNS.items() if not r.run.get("exploration")), DEFAULT_SUITE)

                saved_panel = bench.build(benchmark_suite, results.RUNS[benchmark_suite], demo)

        # ---------------- wiring
        # Draft controls, in DRAFT_KEYS order. Their edits only reach `draft_feedback` (never `applied`).
        draft_in = [exp, station, date, var, ckpt, cond, feats_cb, calendar, ctx, horizon, resid_vars, atmos_fields,
                    tend_lags, systems, band, err_mode, method, main_metric, retrieval, retrieval_index]
        feedback_out = [status, horizon, horizon_info, ctx_info, e_btn, feat_note]
        render_out = [systems, band, fplot, caption, this_tab, sys_desc, dl_case, smap, tiles]
        exp_outputs = [var, station, date, scope, ctx, horizon]

        def feedback(event):
            return event.then(forecast.draft_feedback, draft_in + [applied, submission], feedback_out, queue=False,
                              show_progress="hidden")

        # Forecast edits never apply or redraw either tab's results.
        feedback(exp.input(forecast.on_experiment, [exp, var, station, ckpt, cond], exp_outputs, show_progress="hidden"))
        feedback(station.input(forecast.on_station, [exp, station], date, show_progress="hidden"))
        date.change(forecast.draft_feedback, draft_in + [applied, submission], feedback_out, queue=False, show_progress="hidden")
        for c in (var, ckpt, cond, feats_cb, calendar, ctx, horizon, resid_vars, atmos_fields, tend_lags,
                  systems, band, err_mode, method, main_metric, retrieval, retrieval_index):
            c.input(forecast.draft_feedback, draft_in + [applied, submission], feedback_out, queue=False,
                    show_progress="hidden")
        def refresh_retrieval(selected):
            choices = Store().list("indexes")
            return gr.update(choices=choices, value=selected if selected in {v for _, v in choices} else None)

        feedback(refresh_indexes.click(refresh_retrieval, retrieval_index, retrieval_index))
        show_old.input(lambda s, cur: gr.update(choices=forecast.experiment_choices(s), value=cur), [show_old, exp], exp,
                       show_progress="hidden")

        # Run forecast: the only path that resolves and applies a forecast (one at a time per session)
        (e_btn.click(forecast.begin_forecast, draft_in + [applied, submission],
                     [captured_request, e_btn, e_msg], trigger_mode="once", queue=False)
         .success(forecast.run_captured, captured_request, [pending_run, e_msg], concurrency_limit=1)
         .then(forecast.apply_if_current, [pending_run] + draft_in + [applied, e_msg], [applied, e_msg, changed])
         .then(view.render, [applied, changed], render_out)
         .then(forecast.finish_forecast, draft_in + [applied, submission], feedback_out, queue=False))
        # Page load displays the Forecast state; saved Benchmark owns its load chain.
        feedback(demo.load(view.render, [applied], render_out))
        feedback(demo.load(refresh_retrieval, retrieval_index, retrieval_index))
        # handles for tests (event-graph checks): the draft controls and the applied outputs
        demo.wl = {"draft": draft_in, "applied": applied, "run": e_btn, "render": render_out, "status": status,
                   "submission": submission, "captured_request": captured_request,
                   "saved_benchmark": saved_panel,
                   "advanced": [resid_vars, atmos_fields, tend_lags, systems, band, err_mode, show_old]}
    return demo


def init(run_bootstrap: bool = True) -> gr.Blocks:
    """Fetch/verify artifacts (unless disabled), load saved runs and build the UI."""
    global BOOT, DEFAULT_SUITE
    BOOT = startup_bootstrap() if run_bootstrap else None
    results.RUNS = available_runs()
    from weather_lab.hourly_benchmark import configuration
    from weather_lab.rag_store import Store
    from weather_lab_app.results import load_run

    annual = configuration()
    if annual.get("artifact_id"):
        results.RUNS[annual["suite"]] = load_run(Store().get("suites", annual["artifact_id"]))
    hourly = hourly_exploration()
    if hourly:
        results.RUNS[hourly.suite] = hourly
    DEFAULT_SUITE = hourly.suite if hourly else DEFAULT_EXPERIMENT if DEFAULT_EXPERIMENT in results.RUNS else next(iter(results.RUNS), None)
    return failure_ui(BOOT) if (BOOT or {}).get("status") == "failed" and not results.RUNS else build()


def main() -> None:
    demo = init()
    demo.queue(default_concurrency_limit=4).launch(
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="stone", font=plots.FONT.split(", ")),
        css=CSS,
        allowed_paths=[str(r.path) for r in results.RUNS.values() if r.run.get("hourly_benchmark")]
                      + [str(development.location())])


if __name__ == "__main__":
    main()
