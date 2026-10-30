"""Run frozen suites and independently rescore their saved predictions using weather_lab.scoring."""

from __future__ import annotations

import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from weather_lab import arome as arome_mod
from weather_lab.arome_inputs import build_frozen_arome_inputs
from weather_lab.baselines import Climatology, Persistence, SeasonalNaive
from weather_lab.common import (Config, environment, git_revision, load_config, now_utc, read_json, sha256_file,
                                write_json)
from weather_lab.data import load_observations
from weather_lab.forecast import Adapter, AromeBiasCorrected, AromeRaw, CaseBatch, Timestral
from weather_lab.forecast_config import ForecastConfig, _FROM_MODE, build_adapter
# Local ctx90 reproduction scripts still import common and score from evaluate.
from weather_lab.scoring import common, score
from weather_lab.suite import SEASONS, case_data_checksum, cases_from_manifest, observation_delay, split_bounds

TAIL_QUANTILE = 0.9


# --------------------------------------------------------------------------- adapters

def _foundation(mid: str, spec: dict, cfg: Config, kind: str) -> Adapter:
    from weather_lab.models import CHECKPOINTS

    ck = CHECKPOINTS["t0-gamma"]
    defaults = (cfg["timestral"] if kind == "timestral" else
                {"checkpoint": ck.repo, "revision": ck.revision, **(cfg.get("t0") or {})})
    checkpoint = spec.pop("checkpoint", defaults["checkpoint"])
    revision = spec.pop("revision", defaults["revision"])
    mode, step = spec.pop("arome", None), cfg["forecast"].get("step_hours", 3)
    request = ForecastConfig.make(checkpoint, _FROM_MODE[mode], spec.pop("features", ()),
                                  targets=tuple(spec.pop("targets")), calendar=spec.pop("calendar", False),
                                  context_days=spec.pop("context_days", cfg["forecast"]["context_days"]),
                                  horizon_h=step * cfg["forecast"].get("horizon_steps", 16), step_hours=step,
                                  prediction_target=spec.pop("prediction_target", "weather"))
    batch_size = spec.pop("batch_size", defaults.get("batch_size", 32 if kind == "timestral" else 8))
    adapter = build_adapter(request, mid, variables=cfg["variables"], revision=revision, arome=mode,
                            arome_fields=(cfg.get("arome") or {}).get("fields"), batch_size=batch_size, **spec)
    if adapter.kind != kind:
        raise ValueError(f"Checkpoint {checkpoint!r} is {adapter.kind}, not configured kind {kind}")
    return adapter


# Config `kind` -> constructor(id, remaining spec, config). A new model family adds one entry (and its
# adapter class); scoring, storage, caching and the app read only the shared output schema.
ADAPTERS = {
    "persistence": lambda mid, spec, cfg: Persistence(mid, cfg["variables"]),
    "seasonal_naive": lambda mid, spec, cfg: SeasonalNaive(mid, cfg["variables"]),
    "climatology": lambda mid, spec, cfg: Climatology(mid, cfg["variables"], **spec),
    "arome": lambda mid, spec, cfg: AromeRaw(mid, cfg["variables"], fields=cfg["arome"]["fields"]),
    "timestral": lambda mid, spec, cfg: _foundation(mid, spec, cfg, "timestral"),
    "t0": lambda mid, spec, cfg: _foundation(mid, spec, cfg, "t0"),
    "arome_biascorr": lambda mid, spec, cfg: AromeBiasCorrected(mid, cfg["variables"], fields=cfg["arome"]["fields"],
                                                                **spec),
}


def build_adapters(cfg: Config) -> list[Adapter]:
    out: list[Adapter] = []
    for spec in cfg["models"]:
        spec = dict(spec)
        mid, kind = spec.pop("id"), spec.pop("kind")
        if kind not in ADAPTERS:
            raise ValueError(f"unknown model kind {kind!r} (registered: {sorted(ADAPTERS)})")
        out.append(ADAPTERS[kind](mid, spec, cfg))
    return out


# --------------------------------------------------------------------------- batches

def make_batch(cfg: Config, obs: pd.DataFrame, cases: pd.DataFrame, history_steps: int,
               arome_df: pd.DataFrame | None = None, aux_df: pd.DataFrame | None = None,
               station_coords: dict | None = None) -> tuple[CaseBatch, np.ndarray]:
    """Inputs (history <= origin, issued AROME) and, separately, the verifying targets."""
    f = cfg["forecast"]
    step = pd.Timedelta(hours=f["step_hours"])
    delay = observation_delay(cfg)
    h = f["horizon_steps"] + int(delay / step)
    variables = cfg["variables"]
    n = len(cases)
    hist = np.full((n, len(variables), history_steps), np.nan, dtype=np.float32)
    targets = np.full((n, len(variables), h), np.nan, dtype=np.float32)
    per_station = {s: g.set_index("time")[variables] for s, g in obs.groupby("station")}
    for i, c in enumerate(cases.itertuples()):
        g = per_station[c.station]
        start = g.index[0]
        history_end = c.origin - delay
        pos = int((history_end - start) / step)
        if pos < 0 or pos >= len(g) or g.index[pos] != history_end:
            raise ValueError(f"{c.case_id}: observations do not cover the shifted history endpoint")
        if pos - history_steps + 1 < 0 or pos + h >= len(g):
            raise ValueError(f"{c.case_id}: observations do not cover context + horizon")
        hist[i] = g.iloc[pos - history_steps + 1: pos + 1].to_numpy(dtype=np.float32).T
        targets[i] = g.iloc[pos + 1: pos + h + 1].to_numpy(dtype=np.float32).T
    origins = (cases["origin"] - delay).to_numpy(dtype="datetime64[ns]")
    future = origins[:, None] + (np.arange(1, h + 1) * np.timedelta64(f["step_hours"], "h"))[None, :]
    arome_arr, hist_args = None, {}
    if arome_df is not None:
        arome_arr, hist_args = build_frozen_arome_inputs(arome_df, cases, origins, variables, cfg["arome"]["fields"], h,
                                            f["step_hours"], history_steps)
    aux_args = {}
    if aux_df is not None:
        from weather_lab.features import AUX_VARIABLES

        aux = np.full((n, len(AUX_VARIABLES), history_steps), np.nan, dtype=np.float32)
        per_aux = {s: g.set_index("time")[AUX_VARIABLES] for s, g in aux_df.groupby("station")}
        for i, c in enumerate(cases.itertuples()):
            times = pd.date_range(end=c.origin - delay, periods=history_steps, freq=step)
            aux[i] = per_aux[c.station].reindex(times).to_numpy(dtype=np.float32).T
        aux_args = {"aux_history": aux, "aux_variables": list(AUX_VARIABLES)}
    batch = CaseBatch(cases.reset_index(drop=True), variables, hist, origins, future, f["step_hours"], arome_arr,
                      **hist_args, **aux_args, station_coords=station_coords)
    return batch, targets


def targets_frame(batch: CaseBatch, targets: np.ndarray) -> pd.DataFrame:
    n, v, h = targets.shape
    c, j, k = (x.ravel() for x in np.meshgrid(np.arange(n), np.arange(v), np.arange(h), indexing="ij"))
    frame = pd.DataFrame({
        "case_id": batch.cases["case_id"].to_numpy()[c],
        "variable": np.array(batch.variables)[j],
        "lead_h": batch.lead_hours[c, k],
        "valid_time": pd.to_datetime(batch.future_times[c, k], utc=True),
        "observed": targets[c, j, k],
    })
    return frame.loc[frame["lead_h"] > 0].reset_index(drop=True)


def tail_thresholds(fit_obs: pd.DataFrame) -> dict:
    """Warm-tail thresholds: station x season P90 of fitting-period temperature (not a heatwave definition)."""
    df = fit_obs.assign(season=fit_obs["time"].dt.month.map(SEASONS))
    q = df.groupby(["station", "season"])["t"].quantile(TAIL_QUANTILE)
    return {f"{s}|{season}": round(float(v), 3) for (s, season), v in q.items()}


def observation_windows(obs: pd.DataFrame, cases: pd.DataFrame, history_steps: int, cfg: Config) -> pd.DataFrame:
    """Observations over each eligible case's context + horizon (for plots and exploratory reruns)."""
    step = pd.Timedelta(hours=cfg["forecast"]["step_hours"])
    delay = observation_delay(cfg)
    keep = []
    for station, g in cases.groupby("station"):
        o = obs[obs["station"] == station]
        m = np.zeros(len(o), dtype=bool)
        for origin in g["origin"]:
            m |= ((o["time"] > origin - delay - history_steps * step)
                  & (o["time"] <= origin + cfg["forecast"]["horizon_steps"] * step)).to_numpy()
        keep.append(o[m])
    return pd.concat(keep, ignore_index=True)[["station", "time"] + cfg["variables"]]


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 1e6 if sys.platform == "darwin" else r / 1e3, 1)  # bytes on macOS, KiB on Linux


# --------------------------------------------------------------------------- run

def _same_system(a: Adapter, rec: dict) -> bool:
    """A saved system record describes exactly this adapter (kind, inputs, variables, params, revision)."""
    import json

    want = a.describe()
    norm = lambda x: json.loads(json.dumps(x, sort_keys=True, default=str))  # noqa: E731
    revision = a.revision
    if isinstance(a, Timestral):
        revision = f"{a.checkpoint}@{a.revision[:12]}"
    elif a.kind == "t0":
        revision = f"{a.checkpoint}@{a.revision}"
    return all(norm(want[k]) == norm(rec.get(k)) for k in ("id", "kind", "output", "inputs", "variables", "params")) \
        and rec.get("revision") == revision


def reusable_predictions(prior_dir: Path, cfg: Config, manifest: dict, data_sha: str, adapters: list[Adapter],
                         targets: pd.DataFrame, log=print) -> dict[str, tuple[pd.DataFrame, dict]]:
    """Saved predictions of `prior_dir` that can stand in for a fresh run: same frozen manifest, case data,
    observations, AROME extract and verifying targets, and an identical system description."""
    prior = read_json(prior_dir / "run.json")
    checks = {"manifest": prior["manifest_sha256"] == sha256_file(cfg.manifest_path),
              "case data": prior["data"]["case_data_sha256"] == manifest["case_data_sha256"],
              "observations": prior["data"]["observations"]["sha256"] == data_sha,
              "arome extract": (prior["data"].get("arome") or {}).get("sha256") == (manifest.get("arome") or {}).get(
                  "sha256")}
    old_t = pd.read_parquet(prior_dir / "targets.parquet")
    checks["targets"] = old_t.reset_index(drop=True).equals(targets.reset_index(drop=True))
    if not all(checks.values()):
        raise RuntimeError(f"cannot reuse {prior_dir.name}: differs in {[k for k, ok in checks.items() if not ok]}")
    preds = pd.read_parquet(prior_dir / "predictions.parquet")
    out = {}
    for a in adapters:
        rec = next((r for r in prior["systems"] if r["id"] == a.id), None)
        if rec is None or not _same_system(a, rec):
            continue
        rows = preds[preds["model"] == a.id].drop(columns="run_id")
        expect = len(targets[targets["variable"].isin(a.variables)])
        if len(rows) != expect:
            raise RuntimeError(f"{prior_dir.name}/{a.id}: {len(rows)} rows, expected {expect}")
        out[a.id] = (rows, rec)
        log(f"  {a.id:<36} reused from {prior['run_id']} ({len(rows)} rows, identical description)")
    return out


def run_evaluation(cfg: Config, log=print, reuse_from: Path | None = None) -> Path:
    if not cfg.manifest_path.exists():
        raise FileNotFoundError(f"{cfg.manifest_path} missing: run `weather-lab data prepare` first")
    manifest = read_json(cfg.manifest_path)
    obs, data_meta = load_observations(cfg)
    cases_all = cases_from_manifest(manifest)
    if case_data_checksum(obs, cases_all, cfg) != manifest["case_data_sha256"]:
        raise RuntimeError("observations read by the frozen cases changed since the manifest was written")
    cases = cases_all[cases_all["eligible"]].reset_index(drop=True)
    adapters = build_adapters(cfg)
    arome_df = None
    if any("arome" in a.inputs for a in adapters):
        arome_df, arome_meta = arome_mod.load_arome(cfg)
        if arome_meta["sha256"] != manifest["arome"]["sha256"]:
            raise RuntimeError("AROME station extract differs from the frozen manifest")

    fit_start, fit_end = split_bounds(cfg)["fit"]
    fit_obs = obs[(obs["time"] >= fit_start) & (obs["time"] <= fit_end)]
    step_h = cfg["forecast"]["step_hours"]
    history_steps = max([a.context_steps(step_h) for a in adapters if hasattr(a, "context_steps")]
                        + [cfg["forecast"]["context_days"] * 24 // step_h])
    for a in adapters:
        if hasattr(a, "attach_arome") and arome_df is not None:
            a.attach_arome(arome_df)
    aux_df = aux_meta = None
    from weather_lab.features import GROUPS

    if any("aux_history" in GROUPS[name].needs for a in adapters for name in getattr(a, "features", [])):
        from weather_lab.data import load_aux

        aux_df, aux_meta = load_aux(cfg, log)
    coords = {s["id"]: (s["lat"], s["lon"]) for s in manifest["dataset"]["stations"]}
    batch, targets = make_batch(cfg, obs, cases, history_steps, arome_df, aux_df, coords)
    log(f"suite {cfg.suite} [{cfg.get('label')}]: {len(cases)}/{len(cases_all)} eligible cases, "
        f"{len(adapters)} systems, history {history_steps} steps")

    tag = "-".join(x for x in (cfg.get("variant", "run"), cfg.get("experiment")) if x)
    run_id = f"{tag}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{cfg.hash()[:8]}"
    run_dir = cfg.dir("runs") / cfg.suite / run_id
    run_dir.mkdir(parents=True)
    frames, systems = [], []
    rss0 = peak_rss_mb()
    tgt = targets_frame(batch, targets)
    reused = reusable_predictions(reuse_from, cfg, manifest, data_meta["sha256"], adapters, tgt, log) \
        if reuse_from else {}
    for a in adapters:
        if a.id in reused:
            rows, rec = reused[a.id]
            systems.append(rec | {"reused_from": rec.get("reused_from") or read_json(reuse_from / "run.json")[
                "run_id"]})
            frames.append(rows)
            continue
        t0 = time.perf_counter()
        if isinstance(a, Climatology):
            a.fit(fit_obs)
        fb = a.predict(batch)
        fb.validate(batch)
        elapsed = time.perf_counter() - t0
        rec = a.describe() | {"revision": fb.revision, "wall_s": round(elapsed, 3),
                              "per_case_ms": round(1000 * elapsed / max(len(cases), 1), 3),
                              "peak_rss_mb_after": peak_rss_mb()} | fb.info
        if cfg["forecast"].get("observation_delay_hours", 0):
            rec["input_alignment"] = {
                "observation_delay_hours": cfg["forecast"]["observation_delay_hours"],
                "generated_steps": batch.horizon, "exported_steps": cfg["forecast"]["horizon_steps"],
                "bridge": "discard leads at or before the decision origin",
            }
        if isinstance(a, Timestral):
            rec |= a.latency_probe(batch)
        systems.append(rec)
        frames.append(fb.to_frame(batch, cfg.suite))
        log(f"  {a.id:<36} {fb.output:<9} {elapsed:7.2f} s  "
            f"{np.isfinite(fb.point).mean():.1%} finite")
    preds = pd.concat(frames, ignore_index=True)
    preds.insert(1, "run_id", run_id)
    thresholds = tail_thresholds(fit_obs)

    preds.to_parquet(run_dir / "predictions.parquet", index=False)
    tgt.to_parquet(run_dir / "targets.parquet", index=False)
    shutil.copy(cfg.manifest_path, run_dir / "manifest.json")
    shutil.copy(cfg.path, run_dir / "config.yaml")
    write_json(run_dir / "tail_thresholds.json", thresholds)
    observation_windows(obs, cases, history_steps, cfg).to_parquet(run_dir / "observation_windows.parquet",
                                                                   index=False)
    tables = score_run(run_dir)
    for name, df in tables.items():
        df.to_csv(run_dir / f"{name}.csv", index=False)

    timestral_pkg = _timestral_commit()
    write_json(run_dir / "run.json", {
        "run_id": run_id, "suite": cfg.suite, "variant": cfg.get("variant"), "label": cfg.get("label"),
        "created_at": now_utc(), "config_file": str(cfg.path.name), "config_hash": cfg.hash(),
        "code_revision": git_revision(), "timestral_package": timestral_pkg,
        "manifest_sha256": sha256_file(cfg.manifest_path), "protocol_sha256": manifest["protocol_sha256"],
        "data": {"observations": {"file": data_meta["file"], "sha256": data_meta["sha256"]},
                 "case_data_sha256": manifest["case_data_sha256"], "sources": data_meta["sources"],
                 "arome": None if arome_df is None else {"sha256": manifest["arome"]["sha256"]},
                 **({"aux_observations": {"file": aux_meta["file"], "sha256": aux_meta["sha256"],
                                          "sources": aux_meta["sources"]}} if aux_meta else {})},
        "fit_period": [str(fit_start), str(fit_end)],
        "n_cases": len(cases_all), "n_eligible": len(cases), "history_steps": history_steps,
        "systems": systems, "environment": environment(), "peak_rss_mb_start": rss0,
        "local_fine_tuning": "none", "files": sorted(p.name for p in run_dir.iterdir()),
        "reused_systems_from": read_json(reuse_from / "run.json")["run_id"] if reused else None,
        "model_cache": _model_stats(),
    })
    log(f"saved run {run_dir}")
    return run_dir


def _model_stats() -> dict:
    from weather_lab.models import MODELS, STATS

    return {**STATS, "loaded_at_end": MODELS.status()}


def _timestral_commit() -> str:
    try:
        from importlib.metadata import distribution
        import json as _json

        d = distribution("timestral")
        txt = d.read_text("direct_url.json")
        return _json.loads(txt)["vcs_info"]["commit_id"] if txt else d.version
    except Exception:  # noqa: BLE001 - provenance best effort
        return "unknown"


# --------------------------------------------------------------------------- saved-file rescoring

def score_run(run_dir: Path) -> dict[str, pd.DataFrame]:
    """Recompute every score table from the files saved in `run_dir` (no model calls)."""
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    targets = pd.read_parquet(run_dir / "targets.parquet")
    manifest = read_json(run_dir / "manifest.json")
    cases = cases_from_manifest(manifest)
    thresholds = read_json(run_dir / "tail_thresholds.json")
    cfg = load_config(run_dir / "config.yaml")
    return score(preds, targets, cases, thresholds, cfg["forecast"]["highlight_leads_h"], cfg.get("paired_comparisons"))


def rescore(run_dir: Path, log=print) -> bool:
    tables = score_run(run_dir)
    out = run_dir / "rescored"
    out.mkdir(exist_ok=True)
    identical = True
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
        saved = run_dir / f"{name}.csv"
        if saved.exists():
            same = (out / f"{name}.csv").read_bytes() == saved.read_bytes()
            identical &= same
            log(f"  {name:<32} {'identical' if same else 'DIFFERENT'} ({len(df)} rows)")
    log("rescoring reproduced the saved scores byte-for-byte" if identical else "rescoring DIFFERS from saved scores")
    return identical
