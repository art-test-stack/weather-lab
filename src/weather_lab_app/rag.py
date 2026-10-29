"""Explicit RAG research actions and read-only saved-result browsing."""
from __future__ import annotations

import threading

from weather_lab_app import zerogpu  # before gradio/torch

import gradio as gr
import pandas as pd

from weather_lab import models
from weather_lab.common import read_json
from weather_lab.rag import config
from weather_lab.rag_store import Store
from weather_lab_app.labels import EXPERIMENTS, model_name

_LOCK = threading.Lock()


@zerogpu.gpu(duration=60)
def predict_on_device(adapter, batches):
    from weather_lab.rag_evaluate import predict_cpu

    adapter.device = zerogpu.inference_device()
    try:
        return predict_cpu(adapter, batches)
    finally:
        if zerogpu.on_zerogpu():
            models.MODELS.release("cuda")


def _status():
    store = Store()
    location = "Space bucket mounted" if store.mounted else "Bucket API with a verified local cache"
    return (f"{location}. Artifacts are immutable; each id records its source hashes. "
            "Collection, index building and evaluation run only after their buttons are clicked.")


def refresh():
    store = Store()
    return (_status(), *[gr.update(choices=store.list(kind), value=None) for kind in ("archives", "indexes", "runs")])


def action(kind, archive=None, index=None, horizon=24, progress=gr.Progress()):
    if not _LOCK.acquire(blocking=False):
        raise gr.Error("A RAG research action is already running")
    try:
        store, cfg = Store(), config()
        if kind == "collect":
            from weather_lab.rag_archive import collect
            path = collect(cfg, store)
        elif kind in ("index", "error_index"):
            if not archive:
                raise ValueError("Select a saved archive first")
            if kind == "error_index":
                from weather_lab.common import ROOT
                from weather_lab.rag_arome_error import build

                cfg = config(ROOT / "configs/rag_arome_error_hourly_v3.yaml")
            else:
                from weather_lab.rag_index import build
            path = build(archive, cfg, store)
        elif kind == "evaluate":
            if not index:
                raise ValueError("Select a saved retrieval index first")
            from weather_lab.rag_evaluate import run
            from weather_lab.rag_index import Index

            # Retrieval-only methods stay on CPU. Foundation-model chunks request ZeroGPU explicitly.
            def predict(adapter, batches):
                if not hasattr(adapter, "checkpoint"):
                    from weather_lab.rag_evaluate import predict_cpu
                    return predict_cpu(adapter, batches)
                return predict_on_device(adapter, batches)
            selected = Index(store.get("indexes", index))
            if archive and selected.meta["archive_id"] != archive:
                raise ValueError("The selected index belongs to a different archive; select its archive or build a new index")
            path = run(index, cfg, int(horizon), store, predict=predict, progress=progress)
        elif kind == "prefetch":
            for name in cfg["foundation_models"]:
                models.prefetch(models.CHECKPOINTS[name])
            return "Selected model weights are cached. No evaluation has run.", *refresh()[1:]
        else:
            raise ValueError(kind)
        selections = {"archives": archive, "indexes": index, "runs": None}
        if kind == "collect":
            selections["indexes"] = None
        selections[{"collect": "archives", "index": "indexes", "error_index": "indexes", "evaluate": "runs"}[kind]] = path.name
        return (f"Saved {kind}: {path.name}. " + _status(),
                *[gr.update(choices=store.list(k), value=selections[k]) for k in ("archives", "indexes", "runs")])
    except Exception as exc:
        raise gr.Error(f"RAG {kind} failed: {type(exc).__name__}: {exc}") from exc
    finally:
        _LOCK.release()


def show(ident):
    if not ident:
        return "Select a saved RAG run.", pd.DataFrame(), pd.DataFrame(), None, None, None
    path = Store().get("runs", ident)
    rec = read_json(path / "run.json")
    scores = pd.read_csv(path / "metrics_point.csv")
    common = scores[(scores.scope == "all_systems_common") & scores.lead_h.eq(0)]
    common = common.copy()
    common["model"] = common.model.map(model_name)
    note = (f"**Saved development replay · {rec['horizon_hours']} h · {rec['product']}**  \n"
            f"{rec['requested_cases']:,} requested cases; {rec['cases_with_context']:,} with causal context. "
            f"Status: {rec['status']}.  \n{rec['observation_availability']}  \n"
            "Table: every system on the same verified targets. An empty support means no joint comparison is available. "
            "The CSV also contains paired AROME comparisons; their supports may differ.")
    return (note, common, pd.read_csv(path / "availability.csv"), str(path / "metrics_point.csv"),
            str(path / "predictions.parquet"), str(path / "run.json"))


def build():
    cfg = config()
    store = Store()
    with gr.Tab("RAG research", id="rag"):
        gr.Markdown("### RAG data and experiments\n"
                    "Manage the saved weather data and retrieval indexes used by **Forecast**, or run a separate "
                    "batch evaluation. For a single station and forecast time, use the **Forecast** tab.")
        gr.Markdown("### Available data and indexes\n"
                    "Choose a saved dataset and index. **Refresh** picks up data published from another machine.")
        status = gr.Markdown(_status())
        with gr.Row():
            archive = gr.Dropdown(store.list("archives"), label="Saved weather dataset")
            index = gr.Dropdown(store.list("indexes"), label="Retrieval index")
            refresh_button = gr.Button("Refresh", size="sm")
        gr.Markdown("**Weather-analogue indexes** retrieve similar observed weather from 2022–2025. "
                    "Use them in Forecast or the weather-analogue evaluation below.\n\n"
                    "**AROME-error indexes** retrieve earlier issued forecasts and their verified errors. "
                    "Use them in **Forecast → Archived AROME conditioning**. Only cases completely verified "
                    "before the forecast time can be retrieved. Batch evaluation of this index type is not available here.")
        with gr.Accordion("Prepare data", open=False):
            stations = ", ".join(s["name"] for s in cfg["stations"])
            gr.Markdown("#### Collect a new dataset\n"
                        f"The collection configuration requests **{stations}**, from **2022 onward**, with temperature "
                        "and pressure. Collection can take several minutes. Completed datasets are saved to the bucket; "
                        "downloads stay on local disk until collection finishes.")
            collect = gr.Button("Collect weather data (up to 300 MiB)")
            gr.Markdown("#### Build an index from the selected dataset\n"
                        "Index construction runs on CPU with a one-hour sliding stride. Weather analogues use pre-2026 observations; "
                        "AROME errors use genuine issued runs and verified outcomes for Orly and Blagnac.")
            with gr.Row():
                create = gr.Button("Build weather-analogue index")
                error_index = gr.Button("Build AROME-error index")
            gr.Markdown("#### Model setup\n"
                        "Install pinned model weights before running new forecasts or evaluations. This downloads up to "
                        "**1.86 GB** for Timestral-209M and t0-gamma. Forecasting and evaluation never silently download weights.")
            prefetch = gr.Button("Download model weights (up to 1.86 GB)")
            gr.Markdown("#### Coverage and replay assumptions\n"
                        "Named AROME runs from Open-Meteo start **2 April 2026**; earlier dates remain unavailable. "
                        "Stitched historical series are stored separately and never used as issued forecasts. "
                        "Original publication times are unavailable: this pilot explicitly assumes a 3-hour lag "
                        "for station observations and the selected AROME runs.")
        gr.Markdown("### Run or view evaluations\n"
                    "#### Run a weather-analogue evaluation\n"
                    f"**{EXPERIMENTS[cfg['experiment']]['name']}** compares Timestral-209M and t0-gamma, with and "
                    "without weather retrieval, against AROME across the configured hourly 2026 windows. "
                    "This batch job can take much longer than a single forecast. Missing data stay visible; "
                    "results are labelled as development experiments.")
        with gr.Row():
            horizon = gr.Radio([24, 48], value=24, label="Evaluation horizon (hours)")
            run = gr.Button("Run weather-analogue evaluation", variant="primary")
        gr.Markdown("#### View a saved evaluation\n"
                    "Choose a completed evaluation to open its scores, availability table and downloads.")
        with gr.Row():
            saved = gr.Dropdown(store.list("runs"), label="Saved evaluation")
            load = gr.Button("Open saved evaluation")
        note = gr.Markdown("Select a saved run and click Load saved results.")
        scores = gr.Dataframe(label="Shared-support MAE / RMSE, separated by variable", interactive=False)
        availability = gr.Dataframe(label="Case and system availability", interactive=False)
        with gr.Row():
            dl_scores = gr.DownloadButton("All point scores")
            dl_predictions = gr.DownloadButton("Predictions")
            dl_provenance = gr.DownloadButton("Run provenance")
        outputs = [status, archive, index, saved]
        collect.click(lambda: action("collect"), outputs=outputs, concurrency_id="rag-research", concurrency_limit=1)
        create.click(lambda a: action("index", archive=a), inputs=archive, outputs=outputs,
                     concurrency_id="rag-research", concurrency_limit=1)
        error_index.click(lambda a: action("error_index", archive=a), inputs=archive, outputs=outputs,
                          concurrency_id="rag-research", concurrency_limit=1)
        run.click(lambda a, i, h: action("evaluate", a, i, h), inputs=[archive, index, horizon], outputs=outputs,
                  concurrency_id="rag-research", concurrency_limit=1)
        prefetch.click(lambda: action("prefetch"), outputs=outputs, concurrency_id="rag-research", concurrency_limit=1)
        refresh_button.click(refresh, outputs=outputs)
        load.click(show, inputs=saved, outputs=[note, scores, availability, dl_scores, dl_predictions, dl_provenance])
    return outputs
