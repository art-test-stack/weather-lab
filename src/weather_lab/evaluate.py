"""Run a frozen suite, persist predictions, and score them (re-runnable from saved files).

Scoring conventions (also in docs/experiment_protocol.md):
- point errors use each system's point forecast; for quantile systems that is the median;
- pinball loss: mean over the returned levels of max(tau (y - q), (tau - 1)(y - q)), physical units;
- 80 % / 90 % central intervals: [q0.10, q0.90] / [q0.05, q0.95]; deterministic systems get none;
- every comparison table is computed on the common valid set: the (case, lead) pairs of a
  variable where the target is observed and every compared system produced a finite forecast;
- scores are never pooled across splits, suites or variables.
"""

from __future__ import annotations

import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from weather_lab import arome as arome_mod
from weather_lab.baselines import Climatology, Persistence, SeasonalNaive
from weather_lab.common import (Config, environment, git_revision, load_config, now_utc, read_json, sha256_file,
                                write_json)
from weather_lab.data import load_observations
from weather_lab.forecast import QUANTILE_LEVELS, Adapter, AromeBiasCorrected, AromeRaw, CaseBatch, qcol
from weather_lab.forecast_config import ForecastConfig, _FROM_MODE, build_adapter
from weather_lab.suite import SEASONS, case_data_checksum, cases_from_manifest, split_bounds

POINT_REFERENCE = "persistence"
PROB_REFERENCE = "climatology"
INTERVALS = {80: (0.1, 0.9), 90: (0.05, 0.95)}
MIN_BLOCKS_FOR_CI = 10
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
    h = f["horizon_steps"]
    variables = cfg["variables"]
    n = len(cases)
    hist = np.full((n, len(variables), history_steps), np.nan, dtype=np.float32)
    targets = np.full((n, len(variables), h), np.nan, dtype=np.float32)
    per_station = {s: g.set_index("time")[variables] for s, g in obs.groupby("station")}
    for i, c in enumerate(cases.itertuples()):
        g = per_station[c.station]
        start = g.index[0]
        pos = int((c.origin - start) / step)
        assert g.index[pos] == c.origin
        if pos - history_steps + 1 < 0 or pos + h >= len(g):
            raise ValueError(f"{c.case_id}: observations do not cover context + horizon")
        hist[i] = g.iloc[pos - history_steps + 1: pos + 1].to_numpy(dtype=np.float32).T
        targets[i] = g.iloc[pos + 1: pos + h + 1].to_numpy(dtype=np.float32).T
    origins = cases["origin"].to_numpy(dtype="datetime64[ns]")
    future = origins[:, None] + (np.arange(1, h + 1) * np.timedelta64(f["step_hours"], "h"))[None, :]
    arome_arr, hist_args = None, {}
    if arome_df is not None:
        arome_arr, hist_args = arome_inputs(arome_df, cases, origins, variables, cfg["arome"]["fields"], h,
                                            f["step_hours"], history_steps)
    aux_args = {}
    if aux_df is not None:
        from weather_lab.features import AUX_VARIABLES

        aux = np.full((n, len(AUX_VARIABLES), history_steps), np.nan, dtype=np.float32)
        per_aux = {s: g.set_index("time")[AUX_VARIABLES] for s, g in aux_df.groupby("station")}
        for i, c in enumerate(cases.itertuples()):
            times = pd.date_range(end=c.origin, periods=history_steps, freq=step)   # <= origin only
            aux[i] = per_aux[c.station].reindex(times).to_numpy(dtype=np.float32).T
        aux_args = {"aux_history": aux, "aux_variables": list(AUX_VARIABLES)}
    batch = CaseBatch(cases.reset_index(drop=True), variables, hist, origins, future, f["step_hours"], arome_arr,
                      **hist_args, **aux_args, station_coords=station_coords)
    return batch, targets


def arome_inputs(arome_df: pd.DataFrame, cases: pd.DataFrame, origins: np.ndarray, variables: list[str],
                 fields: list[str], h: int, step_hours: int, history_steps: int) -> tuple[np.ndarray, dict]:
    """Issued AROME over the horizon (run named per case, available at its origin) and the archived
    AROME history with provenance, as `CaseBatch` fields. Shared by the benchmark and exploratory runs."""
    n, step = len(cases), pd.Timedelta(hours=step_hours)
    arome_arr = np.full((n, len(variables), h), np.nan, dtype=np.float32)
    a = arome_df.set_index(["run", "station", "variable", "valid_time"])["value"]
    avail = arome_df.groupby("run")["available_at"].max()
    for i, c in enumerate(cases.itertuples()):
        run = pd.Timestamp(c.arome_run)
        if avail[run] > c.origin:
            raise ValueError(f"{c.case_id}: AROME run {run} not available at origin")
        for j, var in enumerate(variables):
            if var not in fields:
                continue
            for k in range(h):
                arome_arr[i, j, k] = a.get((run, c.station, var, c.origin + (k + 1) * step), np.nan)
    hist_times = origins[:, None] + (np.arange(-(history_steps - 1), 1) * np.timedelta64(step_hours, "h"))[None]
    vals, runs, leads = arome_mod.archived_history(arome_df, cases["station"].to_numpy(), origins, hist_times,
                                                   variables)
    return arome_arr, {"arome_history": vals, "arome_history_run": runs, "arome_history_lead_h": leads}


def targets_frame(batch: CaseBatch, targets: np.ndarray) -> pd.DataFrame:
    n, v, h = targets.shape
    c, j, k = (x.ravel() for x in np.meshgrid(np.arange(n), np.arange(v), np.arange(h), indexing="ij"))
    return pd.DataFrame({
        "case_id": batch.cases["case_id"].to_numpy()[c],
        "variable": np.array(batch.variables)[j],
        "lead_h": (k + 1) * batch.step_hours,
        "valid_time": pd.to_datetime(batch.future_times[c, k], utc=True),
        "observed": targets[c, j, k],
    })


def tail_thresholds(fit_obs: pd.DataFrame) -> dict:
    """Warm-tail thresholds: station x season P90 of fitting-period temperature (not a heatwave definition)."""
    df = fit_obs.assign(season=fit_obs["time"].dt.month.map(SEASONS))
    q = df.groupby(["station", "season"])["t"].quantile(TAIL_QUANTILE)
    return {f"{s}|{season}": round(float(v), 3) for (s, season), v in q.items()}


def observation_windows(obs: pd.DataFrame, cases: pd.DataFrame, history_steps: int, cfg: Config) -> pd.DataFrame:
    """Observations over each eligible case's context + horizon (for plots and exploratory reruns)."""
    step = pd.Timedelta(hours=cfg["forecast"]["step_hours"])
    keep = []
    for station, g in cases.groupby("station"):
        o = obs[obs["station"] == station]
        m = np.zeros(len(o), dtype=bool)
        for origin in g["origin"]:
            m |= ((o["time"] > origin - history_steps * step)
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
    revision = f"{a.checkpoint}@{a.revision[:12]}" if isinstance(a, Timestral) else a.revision
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
    history_steps = max([a.context_steps(step_h) for a in adapters if isinstance(a, Timestral)]
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


# --------------------------------------------------------------------------- scoring

def pinball(y: np.ndarray, q: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Mean pinball loss over levels; y (N,), q (N, Q) -> (N,)."""
    d = y[:, None] - q
    return np.mean(np.maximum(levels * d, (levels - 1) * d), axis=1)


def _row_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row losses for one variable; df has observed, point and (maybe) quantile columns."""
    out = df.copy()
    err = out["point"] - out["observed"]
    out["abs_err"] = err.abs()
    out["sq_err"] = err ** 2
    out["err"] = err
    qcols = [qcol(l) for l in QUANTILE_LEVELS]
    has_q = out["output_type"].eq("quantiles").to_numpy()
    out["pinball"] = np.nan
    if has_q.any():
        sub = out.loc[has_q]
        out.loc[has_q, "pinball"] = pinball(sub["observed"].to_numpy(), sub[qcols].to_numpy(), np.array(QUANTILE_LEVELS))
        for nominal, (lo, hi) in INTERVALS.items():
            l, u = sub[qcol(lo)].to_numpy(), sub[qcol(hi)].to_numpy()
            y = sub["observed"].to_numpy()
            out.loc[has_q, f"in{nominal}"] = ((y >= l) & (y <= u)).astype(float)
            out.loc[has_q, f"width{nominal}"] = u - l
    return out


def _cube(df: pd.DataFrame, metrics: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Aggregate over split x model x variable x {station, ALL} x {season, ALL} x {lead, ALL}."""
    frames = []
    for st in (True, False):
        for se in (True, False):
            for ld in (True, False):
                keys = ["split", "model", "variable"] + (["station"] if st else []) + (["season"] if se else []) \
                    + (["lead_h"] if ld else [])
                g = df.groupby(keys, observed=True).agg(**metrics, n=("observed", "size")).reset_index()
                if not st:
                    g["station"] = "ALL"
                if not se:
                    g["season"] = "ALL"
                if not ld:
                    g["lead_h"] = pd.NA
                frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    out["lead_h"] = out["lead_h"].astype("Int64")
    return out


def _add_skill(table: pd.DataFrame, metric: str, ref: str) -> pd.DataFrame:
    keys = ["split", "variable", "station", "season", "lead_h"]
    r = table[table["model"] == ref][keys + [metric]].rename(columns={metric: f"{metric}_ref"})
    out = table.merge(r, on=keys, how="left")
    out[f"skill_{metric}_vs_{ref}"] = 1 - out[metric] / out[f"{metric}_ref"]
    return out


def _bootstrap(df: pd.DataFrame, loss: str, ref: str, highlight: list[int], n_boot: int = 2000,
               block_days: int = 2, seed: int = 0) -> pd.DataFrame:
    """Paired mean loss difference (model - ref) with a block bootstrap over origin dates.

    Blocks group consecutive origin days (`block_days`, matching the 48 h horizon overlap of daily
    origins); all stations of a day share a block. A CI is reported only with >= MIN_BLOCKS_FOR_CI blocks.
    """
    rng = np.random.default_rng(seed)
    rows = []
    df = df.assign(day=df["origin"].dt.floor("D"))
    for (split, var), g in df.groupby(["split", "variable"]):
        days = np.sort(g["day"].unique())
        block_of = {d: i // block_days for i, d in enumerate(days)}
        piv = g.pivot_table(index=["case_id", "lead_h", "day"], columns="model", values=loss)
        if ref not in piv:
            continue
        for model in [m for m in piv.columns if m != ref]:
            for lead in [None] + highlight:
                p = piv if lead is None else piv[piv.index.get_level_values("lead_h") == lead]
                d = (p[model] - p[ref]).dropna()
                if d.empty:
                    continue
                blocks = d.groupby(d.index.get_level_values("day").map(block_of)).agg(["sum", "count"])
                nb = len(blocks)
                rec = {"split": split, "variable": var, "model": model, "reference": ref, "loss": loss,
                       "lead_h": lead, "n_pairs": len(d), "n_blocks": nb, "mean_diff": d.mean(),
                       "ci95_low": np.nan, "ci95_high": np.nan}
                if nb >= MIN_BLOCKS_FOR_CI:
                    idx = rng.integers(0, nb, size=(n_boot, nb))
                    s, c = blocks["sum"].to_numpy()[idx].sum(1), blocks["count"].to_numpy()[idx].sum(1)
                    rec["ci95_low"], rec["ci95_high"] = np.percentile(s / c, [2.5, 97.5])
                rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out):
        out["lead_h"] = out["lead_h"].astype("Int64")
    return out


def score(preds: pd.DataFrame, targets: pd.DataFrame, cases: pd.DataFrame, thresholds: dict,
          highlight: list[int], pairs: list[list[str]] | None = None) -> dict[str, pd.DataFrame]:
    meta = cases[["case_id", "split", "season", "origin"]]
    df = preds.drop(columns=["origin"]).merge(targets[["case_id", "variable", "lead_h", "observed"]],
                                              on=["case_id", "variable", "lead_h"], how="left")
    df = df.merge(meta, on="case_id", how="left")
    df = _row_scores(df)

    # availability: expected = all manifest cases x leads (ineligible cases count as unavailable)
    n_leads = targets["lead_h"].nunique()
    avail = []
    for (split, model, var), g in df.groupby(["split", "model", "variable"]):
        n_all = int((cases["split"] == split).sum())
        n_elig = int(((cases["split"] == split) & cases["eligible"]).sum())
        avail.append({"split": split, "model": model, "variable": var, "output_type": g["output_type"].iloc[0],
                      "cases_in_suite": n_all, "cases_eligible": n_elig, "cases_ineligible": n_all - n_elig,
                      "expected_targets": n_all * n_leads, "eligible_targets": n_elig * n_leads,
                      "observed_targets": int(g["observed"].notna().sum()),
                      "predicted": int(g["point"].notna().sum()),
                      "failed_predictions": int(g["point"].isna().sum()),
                      "valid_pairs": int((g["point"].notna() & g["observed"].notna()).sum())})
    avail = pd.DataFrame(avail)

    point_rows = common(df, "point")
    point = _cube(point_rows, {"mae": ("abs_err", "mean"), "rmse": ("sq_err", "mean"), "bias": ("err", "mean")})
    point["rmse"] = np.sqrt(point["rmse"])
    point = _add_skill(point, "mae", POINT_REFERENCE)
    point = point.merge(avail.groupby(["split", "variable"])["eligible_targets"].first().reset_index(),
                        on=["split", "variable"], how="left")

    prob_rows = common(df[df["output_type"] == "quantiles"], "pinball")
    prob = _cube(prob_rows, {"pinball": ("pinball", "mean"), "coverage80": ("in80", "mean"),
                             "coverage90": ("in90", "mean"), "width80": ("width80", "mean"),
                             "width90": ("width90", "mean")})
    prob = _add_skill(prob, "pinball", PROB_REFERENCE)
    na = avail[avail["output_type"] == "point"][["split", "model", "variable"]].assign(
        note="not applicable: deterministic point forecast (no predictive distribution)")

    paired = pd.concat([_bootstrap(point_rows, "abs_err", POINT_REFERENCE, highlight),
                        _bootstrap(prob_rows, "pinball", PROB_REFERENCE, highlight)], ignore_index=True)

    t = point_rows[point_rows["variable"] == "t"].copy()
    t["threshold"] = np.array([thresholds.get(f"{s}|{SEASONS[v.month]}", np.nan)
                               for s, v in zip(t["station"], t["valid_time"])], dtype=float)
    tail = t[t["observed"] > t["threshold"]]
    matched = _matched(point_rows, prob_rows, pairs or [], highlight)
    tail_tab = tail.groupby(["split", "model"]).agg(n=("observed", "size"), mae=("abs_err", "mean"),
                                                    bias=("err", "mean"), pinball=("pinball", "mean")).reset_index()
    tail_tab["definition"] = "obs t > station x season P90 of fitting period (warm tail, not a heatwave definition)"

    # per-case summary for the UI's case browser
    case_tab = point_rows.groupby(["case_id", "station", "split", "season", "variable", "model"]).agg(
        mae=("abs_err", "mean"), n=("abs_err", "size")).reset_index()
    return {"metrics_point": point, "metrics_probabilistic": prob, "probabilistic_not_applicable": na,
            "availability": avail, "paired_differences": paired, "tail_warm_t": tail_tab, "case_scores": case_tab,
            **({"paired_matched": matched} if pairs else {})}


def common(sub: pd.DataFrame, col: str) -> pd.DataFrame:
    """Common valid set per (split, variable): pairs where the target is observed and every system in
    `sub` produced a finite value of `col`."""
    ok = sub[col].notna() & sub["observed"].notna()
    n_models = sub.groupby(["split", "variable"])["model"].transform("nunique")
    good = sub[ok].groupby(["split", "variable", "case_id", "lead_h"])["model"].transform("nunique")
    return sub[ok][good == n_models[ok]]


def _matched(point_rows: pd.DataFrame, prob_rows: pd.DataFrame, pairs: list[list[str]],
             highlight: list[int]) -> pd.DataFrame:
    """Paired differences for declared (model, reference) pairs on the run's common valid sets."""
    out = []
    for model, ref in pairs:
        for rows, loss in ((point_rows, "abs_err"), (prob_rows, "pinball")):
            sub = rows[rows["model"].isin([model, ref])]
            if sub["model"].nunique() == 2:
                out.append(_bootstrap(sub, loss, ref, highlight))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


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
