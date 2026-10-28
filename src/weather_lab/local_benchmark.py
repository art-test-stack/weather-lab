"""Local-only matrix execution; inference and later bucket publication are separate commands."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from weather_lab import development, models
from weather_lab.common import ROOT, environment, read_json, sha256_file, write_json
from weather_lab.forecast_config import ForecastConfig, check
from weather_lab.rag_store import Store
from weather_lab.suite import cases_from_manifest

CONFIG = ROOT / "configs/hourly_matrix_2026_bf16_v2.yaml"
BATCH_SIZE = 8
KEYS = ["case_id", "variable", "lead_h"]
POINT_COLUMNS = [*KEYS, "station", "origin", "valid_time", "model", "point"]


def configurations(config):
    result = []
    for model in config["checkpoints"]:
        for setup in config["setups"]:
            cfg = ForecastConfig.make(model, setup.get("conditioning", "none"), setup.get("features", []),
                calendar=setup["calendar"], context_days=config["context_days"], horizon_h=config["horizon_h"],
                step_hours=1, targets=("t", "pmer"), prediction_target=setup.get("prediction_target", "weather"),
                precision=config.get("precision", "native"), retrieval_method=setup.get("retrieval", "none"),
                retrieval_index=config[setup["index"]] if setup.get("index") else "")
            issues = check(cfg, max_horizon_h=48)
            if issues:
                raise ValueError("; ".join(i.text() for i in issues))
            result.append((f"{model}/{setup['name']}", cfg))
    if len({name for name, _ in result}) != len(result):
        raise ValueError("matrix configuration names must be unique")
    return result


def verify_run(path):
    rec = read_json(path / "run.json")
    for name, digest in rec["files_sha256"].items():
        if Path(name).name != name or sha256_file(path / name) != digest:
            raise ValueError(f"saved run failed verification: {path / name}")
    return rec


def completed(out, config, configs):
    found = {}
    for path in sorted((out / "runs").glob("*/run.json")):
        rec = read_json(path)
        if rec.get("source_suite") != config["suite"]:
            raise ValueError("output contains results from another source suite; choose a new output directory")
        for name, cfg in configs:
            if rec["forecast_config"] == cfg.as_dict():
                verify_run(path.parent)
                found[name] = path.parent
                break
    return found


def plan(config, device):
    files = ("local_benchmark.py", "development.py", "forecast.py", "forecast_config.py", "hourly_benchmark.py",
             "features.py", "t0_adapter.py", "rag_adapters.py", "rag_index.py", "rag_arome_error.py", "rag_evaluate.py",
             "arome_inputs.py", "precision.py", "models.py", "suite.py", "common.py", "evaluate.py")
    runtime = environment()
    return {"matrix": config["matrix"], "suite": config["suite"], "source_artifact": config["source_artifact"],
            "device": device, "inference_batch_size": BATCH_SIZE,
            "configurations": {n: c.as_dict() for n, c in configurations(config)},
            "checkpoints": {m: {"repo": models.CHECKPOINTS[m].repo, "revision": models.CHECKPOINTS[m].revision,
                                 "sha256": models.CHECKPOINTS[m].sha256} for m in config["checkpoints"]},
            "runtime": {k: runtime[k] for k in ("python", "torch", "numpy", "pandas", "torch_threads")},
            "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
            "code_sha256": {n: sha256_file(Path(__file__).with_name(n)) for n in files}}


def run(config, out, device="cpu", predict=None, log=print):
    import torch

    if device == "cuda" and not torch.cuda.is_available() or device == "mps" and not torch.backends.mps.is_available():
        raise ValueError(f"{device} is unavailable; choose an available device explicitly")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "matrix_plan.json"
    requested = plan(config, device)
    if frozen.exists() and read_json(frozen) != requested:
        raise ValueError("this output's matrix plan is frozen; resume with identical code/settings or choose a new output directory")
    if not frozen.exists():
        pending = out / "matrix_plan.pending"
        write_json(pending, requested)
        pending.replace(frozen)
    configs = configurations(config)
    done = completed(out, config, configs)
    source = Store().get("suites", config["source_artifact"])
    if read_json(source / "run.json")["suite"] != config["suite"]:
        raise ValueError("matrix source artifact differs from its suite")
    for key in ("weather_index", "error_index"):
        meta = read_json(Store().get("indexes", config[key]) / "index.json")
        if (meta["archive_id"] != read_json(source / "run.json")["archive_id"]
                or meta["variables"] != ["t", "pmer"] or meta["context_hours"] != config["context_days"] * 24
                or meta["horizon_hours"] < config["horizon_h"] + 3 or meta["observation_delay_hours"] != 3
                or meta["retrieval"].get("stride_hours") != 1):
            raise ValueError("matrix index and source archive, variables, context, horizon or delay differ")
    # Explicit setup is separate: a long run never silently downloads a later model's weights.
    for name in config["checkpoints"]:
        models.prefetch(models.CHECKPOINTS[name], allow_download=False)

    def worker(adapter, batches):
        adapter.device = device
        results = predict(adapter, batches) if predict else development.predict_cases(adapter, batches, batch_size=BATCH_SIZE)
        if results and all(result is None or not np.isfinite(result.point).any() for result, _ in results):
            raise RuntimeError("Every forecast in an inference batch failed; stopping this setup. "
                               "Completed setups are preserved. First failure: " + (results[0][1] or "no finite predictions"))
        return results

    for number, (name, cfg) in enumerate(configs, 1):
        if name in done:
            log(f"[{number}/{len(configs)}] Verified completed setup: {name}", flush=True)
        else:
            log(f"[{number}/{len(configs)}] Running locally on {device}: {name}", flush=True)
            def progress(fraction, desc):
                if not hasattr(progress, "last") or fraction - progress.last >= .02 or fraction == 1:
                    progress.last = fraction
                    log(f"  {name}: {desc}", flush=True)
            done[name] = development.run(source, cfg, predict=worker, progress=progress, root=out / "runs")
        rec = read_json(done[name] / "run.json")
        # Small discovery record; the original scored files and run provenance stay unchanged.
        write_json(done[name] / "matrix_setup.json", {"configuration_id": name, "matrix": config["matrix"],
                   "source_suite": rec["source_suite"], "created_at": rec["created_at"],
                   "forecast_config": rec["forecast_config"], "run_json_sha256": sha256_file(done[name] / "run.json"),
                   "device": device})
    models.MODELS.release()
    compare(config, out, log=log)
    return done


def _primary(path):
    rec = read_json(path / "run.json")
    frame = pd.read_parquet(path / "predictions.parquet", columns=POINT_COLUMNS)
    return frame[frame.model.eq(rec["systems"][-1]["id"])].copy()


def compare(config, out, log=print):
    """Stream the common support first, then score each configuration on exactly those same pairs."""
    from weather_lab.evaluate import _add_skill, _cube, _row_scores

    configs = configurations(config)
    found = completed(Path(out), config, configs)
    if set(found) != {n for n, _ in configs}:
        raise ValueError("shared matrix comparison requires every configured setup to be complete")
    source = Store().get("suites", config["source_artifact"])
    targets = pd.read_parquet(source / "targets.parquet")
    ar = pd.read_parquet(source / "arome_reference.parquet", columns=POINT_COLUMNS)
    persistence = pd.read_parquet(source / "predictions.parquet", columns=POINT_COLUMNS)
    initial = ar[ar.point.notna()].merge(targets, on=KEYS, suffixes=("", "_truth"))
    initial = initial[initial.observed.notna()]
    support = pd.MultiIndex.from_frame(initial[KEYS])
    support = support.intersection(pd.MultiIndex.from_frame(persistence[persistence.point.notna()][KEYS]))
    for name, _ in configs:
        frame = _primary(found[name])
        support = support.intersection(pd.MultiIndex.from_frame(frame[frame.point.notna()][KEYS]))
    support = support.sort_values()
    if support.empty:
        raise ValueError("no common observed pairs across all matrix setups and AROME; inspect case_availability.csv, no ranking was generated")
    destination = Path(out) / "comparisons"
    destination.mkdir(exist_ok=True)
    support.to_frame(index=False).to_parquet(destination / "common_support.parquet", index=False)
    support_sha = sha256_file(destination / "common_support.parquet")
    cases = cases_from_manifest(read_json(source / "manifest.json"))
    tables = []
    def score_frame(label, frame):
        frame = frame[pd.MultiIndex.from_frame(frame[KEYS]).isin(support)].copy()
        frame["model"], frame["output_type"] = label, "point"
        frame = frame.merge(targets[KEYS + ["observed"]], on=KEYS).merge(cases[["case_id", "split", "season"]], on="case_id")
        table = _cube(_row_scores(frame), {"mae": ("abs_err", "mean"), "rmse": ("sq_err", "mean"), "bias": ("err", "mean")})
        table["rmse"] = np.sqrt(table.rmse)
        return table
    tables.extend([score_frame("arome-raw", ar), score_frame("persistence", persistence)])
    for name, _ in configs:
        tables.append(score_frame(name, _primary(found[name])))
    scores = _add_skill(pd.concat(tables, ignore_index=True), "mae", "arome-raw")
    scores["support_sha256"] = support_sha
    scores["scope"] = "all configured setups plus AROME and persistence on identical observed pairs per variable"
    scores.to_csv(destination / "shared_arome_metrics.csv", index=False)
    write_json(destination / "provenance.json", {"matrix": config["matrix"], "suite": config["suite"],
               "source_artifact": config["source_artifact"], "support_sha256": support_sha,
               "scorer_sha256": sha256_file(Path(__file__)),
               "scores_sha256": sha256_file(destination / "shared_arome_metrics.csv"),
               "configurations": {n: {"run_id": read_json(p / "run.json")["run_id"],
                                      "run_json_sha256": sha256_file(p / "run.json")} for n, p in found.items()},
               "claim": "Development replay; descriptive point scores; overlapping windows are dependent; no held-out or fine-tuning claim"})
    log(f"Saved shared-support AROME comparison: {destination / 'shared_arome_metrics.csv'}", flush=True)


def publish(config, out):
    """Explicit later upload: data first, run completion records last; never runs inference."""
    import subprocess
    from weather_lab.publish import DEFAULT_BUCKET

    out = Path(out).resolve()
    found = completed(out, config, configurations(config))
    provenance_path = out / "comparisons/provenance.json"
    provenance = read_json(provenance_path)
    if (provenance["support_sha256"] != sha256_file(out / "comparisons/common_support.parquet")
            or provenance["scores_sha256"] != sha256_file(out / "comparisons/shared_arome_metrics.csv")):
        raise ValueError("shared comparison checksum differs; rerun local-benchmark compare before publishing")
    expected = {name for name, _ in configurations(config)}
    if set(found) != expected or set(provenance["configurations"]) != expected:
        raise ValueError("publish requires all setups and their completed shared-support comparison")
    for name, path in found.items():
        if provenance["configurations"][name]["run_json_sha256"] != sha256_file(path / "run.json"):
            raise ValueError("comparison provenance differs from the saved runs")
    prefix = f"hf://buckets/{DEFAULT_BUCKET}"
    base = ["hf", "buckets", "sync", str(out / "runs"), f"{prefix}/development-runs"]
    subprocess.run(base + ["--exclude", "*/run.json"], check=True)
    subprocess.run(base + ["--include", "*/run.json"], check=True)
    ident = config["matrix"] + "-" + sha256_file(provenance_path)[:16]
    subprocess.run(["hf", "buckets", "sync", str(out / "comparisons"),
                    f"{prefix}/benchmark-matrices/{ident}"], check=True)
    print(f"Published {len(found)} saved runs and {ident}. Reload Benchmark to discover the runs.")


def command(args):
    config = yaml.safe_load(Path(args.config).read_text())
    if args.local_action == "plan":
        for name, cfg in configurations(config):
            print(f"{name}: {cfg.as_dict()}")
        print(f"{len(configurations(config))} local configurations; no inference or uploads")
    elif args.local_action == "run":
        run(config, Path(args.out), args.device)
    elif args.local_action == "compare":
        compare(config, Path(args.out))
    else:
        publish(config, Path(args.out))


def parser(sub):
    root = sub.add_parser("local-benchmark", help="local hourly matrix; only the explicit publish command uploads results")
    actions = root.add_subparsers(dest="local_action", required=True)
    for name in ("plan", "run", "compare", "publish"):
        action = actions.add_parser(name)
        action.add_argument("--config", default=str(CONFIG))
        action.add_argument("--out", default="runs/local-hourly-2026")
        if name == "run":
            action.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
        action.set_defaults(func=command)
