"""Independent draft/apply controls for explicit full-suite development benchmarks."""
from __future__ import annotations

import copy
import threading
from pathlib import Path

import gradio as gr
import pandas as pd

from weather_lab import development, features, models
from weather_lab.common import read_json, sha256_file
from weather_lab.forecast_config import ForecastConfig, check, describe
from weather_lab_app import zerogpu
from weather_lab_app.results import load_run

KEYS = ("suite", "size", "prediction_target", "conditioning", "features", "calendar", "ctx", "horizon",
        "resid_vars", "atmos_fields", "tend_lags", "variable", "metric", "lead", "split", "station", "season",
        "statuses", "lead_mode", "saved")
_LOCK = threading.Lock()


def draft(*values):
    return copy.deepcopy(dict(zip(KEYS, values)))


def configuration(d):
    from weather_lab_app import main as app

    return ForecastConfig.make(f"Timestral-{d['size']}", d["conditioning"], app.feature_spec(d),
                               calendar=bool(d["calendar"]), context_days=int(d["ctx"]), horizon_h=int(d["horizon"]),
                               step_hours=app.RUNS[d["suite"]].step_hours,
                               targets=("t", "pmer") if d["prediction_target"] == "arome_error" else ("t", "pmer", "u", "v"),
                               prediction_target=d["prediction_target"])


def defaults(suite):
    from weather_lab_app import main as app

    d = app.default_draft(suite)
    d.update(variable="t", metric="mae", lead="all", split=app.RUNS[suite].splits[-1], station="ALL", season="ALL",
             statuses=app.STATUSES, lead_mode="at", saved="")
    return {k: copy.deepcopy(d[k]) for k in KEYS}


def choices(suite):
    return [("New development run", "")] + development.saved_runs(suite)


def feedback(*values):
    from weather_lab_app import main as app

    *vals, applied, guard = values
    d = draft(*vals)
    pending = not applied or d != applied["draft"]
    issues = []
    try:
        cfg = configuration(d)
        issues = check(cfg, max_horizon_h=app.RUNS[d["suite"]].horizon_hours, context_range=app.CONTEXT_DAYS)
        if d["variable"] not in cfg.targets:
            raise ValueError("The selected forecast method does not target this weather variable")
    except (ValueError, TypeError) as exc:
        issues = [str(exc)]
    status = ("Changes pending — click Run benchmark. The displayed results are unchanged." if pending else
              "Settings match the displayed result. Run benchmark replays the selected saved run." if d["saved"] else
              "Settings match the displayed result. Rerun creates a new development run.")
    if issues:
        status += "\n\n" + "; ".join(i.text() if hasattr(i, "text") else i for i in issues)
    return status, gr.update(interactive=not issues and not guard["busy"],
                             value="Running…" if guard["busy"] else "Run benchmark" if pending or d["saved"] else "Rerun")


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
    from weather_lab_app import main as app

    d, previous = request["draft"], request["applied"]
    try:
        cfg = configuration(d)
        issues = check(cfg, max_horizon_h=app.RUNS[d["suite"]].horizon_hours, context_range=app.CONTEXT_DAYS)
        if issues or d["variable"] not in cfg.targets:
            raise ValueError("; ".join(i.text() for i in issues) or "This method does not target the selected variable")
        if d["saved"]:
            if d["saved"] not in {p for _, p in development.saved_runs(d["suite"])}:
                raise ValueError("Select a saved development run of this suite")
            rec = read_json(Path(d["saved"]) / "run.json")
            for name, digest in rec["files_sha256"].items():
                if sha256_file(Path(d["saved"]) / name) != digest:
                    raise ValueError("A saved development artifact failed its checksum")
            if rec["forecast_config"] != cfg.as_dict():
                raise ValueError("Settings differ from the selected saved run; choose New development run")
            path, kind, message = d["saved"], "development", "Replayed the saved development benchmark."
        elif (previous and previous["kind"] == "development" and previous["draft"]["suite"] == d["suite"]
              and previous["config"] == cfg and previous["draft"] != d):
            path, kind, message = previous["path"], "development", "Applied the display settings; no new inference."
        else:
            path = str(development.run(app.RUNS[d["suite"]].path, cfg, predict=predict_on_device, progress=progress))
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
    from weather_lab_app import main as app

    controls = app.bench_controls(suite)
    variables = app.variable_choices(suite)
    return (gr.update(choices=variables, value=variable if variable in dict((v, k) for k, v in variables) else "t"),
            *controls, gr.update(choices=choices(suite), value=""),
            gr.update(maximum=app.RUNS[suite].horizon_hours, value=app.RUNS[suite].horizon_hours))


def on_saved(path, suite):
    if not path:
        return tuple(gr.skip() for _ in range(10))
    if path not in {p for _, p in development.saved_runs(suite)}:
        raise gr.Error("Select a saved development run of this suite")
    c = read_json(Path(path) / "run.json")["forecast_config"]
    spec = {f["name"]: f["params"] for f in c["features"]}
    return (c["model"].split("-")[-1], c["prediction_target"], c["conditioning"], list(spec), c["calendar"],
            c["context_days"], c["horizon_h"], spec.get("resid", features.GROUPS["resid"].defaults)["variables"],
            spec.get("atmos", features.GROUPS["atmos"].defaults)["fields"],
            spec.get("tend", features.GROUPS["tend"].defaults)["lags_h"])


def render(applied, changed=True):
    from weather_lab_app import main as app

    if not changed:
        return tuple(gr.skip() for _ in range(19))
    d = applied["draft"]
    run = load_run(Path(applied["path"]))
    if applied["kind"] == "frozen":
        out = app.benchmark(d["suite"], d["variable"], d["metric"], d["lead"], d["split"], d["station"], d["season"],
                            d["statuses"], d["size"], d["conditioning"], d["lead_mode"])
        matrix = app.matrix_html(d["suite"], d["variable"], d["split"], d["metric"])
        label = "Frozen saved results. Run benchmark executes the separate configuration above."
    else:
        out = app.benchmark(d["suite"], d["variable"], d["metric"], d["lead"], d["split"], d["station"], d["season"],
                            d["statuses"], d["size"], d["conditioning"], d["lead_mode"], run=run)
        matrix = "Development run: one selected configuration. " + describe(applied["config"])
        label = (f"Development benchmark · {run.run['run_id']} · {describe(applied['config'])} · "
                 f"checkpoint revision {run.run['checkpoint']['revision'][:12]}")
    advanced = app.advanced_tables(d["suite"], d["variable"], d["split"], run=run)
    if applied["kind"] == "development":
        advanced = (pd.read_csv(run.path / "case_availability.csv", dtype={"station": str}), *advanced[1:3],
                    "Reference timings were copied from the source run. Development-model timings are not measured "
                    "separately; these are not comparable hardware measurements.")
    return (label, *out, matrix, *advanced, str(run.path / "metrics_point.csv"),
            str(run.path / "metrics_probabilistic.csv"), str(run.path / "predictions.parquet"),
            str(run.path / "manifest.json"), str(run.path / "run.json"),
            str(run.path / "case_availability.csv") if applied["kind"] == "development" else None)
