"""Read saved runs for the UI. Nothing here runs a benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from weather_lab.common import ROOT, STATION_NAMES, read_json
from weather_lab.forecast import QUANTILE_LEVELS, qcol

from weather_lab_app.labels import EXPERIMENT_ORDER, EXPERIMENTS, model_name


def training_status(model: str) -> str:
    if model.startswith(("timestral", "t0")):
        return "pretrained, no local fine-tuning"
    if model == "climatology":
        return "fitted on fitting period"
    if model == "arome-raw":
        return "external NWP (issued run)"
    if model == "arome-biascorr":
        return "post-processing of AROME (recent errors)"
    return "no fitting"


@dataclass
class Run:
    suite: str
    path: Path
    source: str                    # "local run" | "cached example"
    run: dict
    manifest: dict
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    issued_times: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def label(self) -> str:
        return f"{self.run['run_id']} · {self.run['label']} · {self.source}"

    @property
    def splits(self) -> list[str]:
        if self.run.get("exploration"):
            return ["exploratory"]
        return sorted(self.tables["metrics_point"]["split"].unique(), key=lambda s: ["validation", "test"].index(s))

    @property
    def models(self) -> list[str]:
        return [s["id"] for s in self.run["systems"]]

    def predictions(self) -> pd.DataFrame:
        if self.run.get("exploration"):
            return pd.DataFrame(columns=["case_id", "station", "origin", "valid_time", "lead_h", "variable",
                                         "model", "point", "output_type", "model_revision"])
        return _parquet(self.path / "predictions.parquet")

    def point_predictions(self, model: str, variable: str) -> pd.DataFrame:
        """Projected saved points: cache only the selected system/variable, without quantiles."""
        from weather_lab.saved_results import POINT_COLUMNS

        return _parquet(self.path / "predictions.parquet", POINT_COLUMNS,
                        (("model", "==", model), ("variable", "==", variable)))

    def issued_reference(self, case_id: str, variable: str) -> pd.DataFrame:
        from weather_lab.saved_results import POINT_COLUMNS

        return _parquet(self.path / "predictions.parquet", (*POINT_COLUMNS, "arome_run", "arome_available_at"),
                        (("model", "==", "arome-raw"), ("variable", "==", variable), ("case_id", "==", case_id)))

    def case_availability(self) -> pd.DataFrame:
        return _case_availability(self.path)

    def targets(self) -> pd.DataFrame:
        return _parquet(self.path / "targets.parquet")

    def observations(self) -> pd.DataFrame:
        if self.run.get("exploration"):
            from weather_lab.forecast_data import OBSERVATIONS

            return _parquet(self.path / OBSERVATIONS)
        return _parquet(self.path / "observation_windows.parquet")

    def cases(self) -> pd.DataFrame:
        df = pd.DataFrame(self.manifest["cases"])
        df["origin"] = pd.to_datetime(df["origin"], utc=True)
        return df

    def case(self, case_id: str) -> pd.Series:
        if self.run.get("exploration"):
            from weather_lab.forecast_data import make_case

            parts = str(case_id).split("/")
            if len(parts) != 3 or parts[0] != self.suite:
                raise ValueError("Select an exact UTC hour in the forecast calendar")
            try:
                origin = pd.to_datetime(parts[2], format="%Y%m%dT%H", utc=True)
            except ValueError:
                raise ValueError("Select an exact UTC hour in the forecast calendar") from None
            return make_case(self.run["hourly_data"], parts[1], origin, self.issued_times)
        return self.cases().set_index("case_id").loc[case_id]

    def stations(self) -> list[dict]:
        return self.manifest["dataset"]["stations"]

    @property
    def step_hours(self) -> int:
        return int(self.manifest["protocol"]["forecast"]["step_hours"])

    @property
    def horizon_hours(self) -> int:
        """Inference horizon of every saved forecast in this run (frozen in the suite)."""
        return int(self.manifest["protocol"]["forecast"]["horizon_steps"]) * self.step_hours

    def params(self, model: str) -> dict:
        return next((s.get("params") or {} for s in self.run["systems"] if s["id"] == model), {})

    def full_observations(self) -> pd.DataFrame:
        """The run's whole observation file when installed and unchanged (same SHA-256 as recorded in
        run.json), else the saved per-case windows. Used for longer exploratory contexts/horizons."""
        if self.run.get("exploration"):
            return self.observations()
        meta = (self.run.get("data") or {}).get("observations") or {}
        p = ROOT / "data" / "processed" / str(meta.get("file", ""))
        if meta.get("file") and p.exists() and _sha256(p) == meta.get("sha256"):
            return _parquet(p)
        return self.observations()

    def aux_path(self) -> Path | None:
        """The auxiliary SYNOP file (dew point, cloud cover) for this run's stations and years, if installed:
        the file recorded in run.json (when unchanged), else the one `data.aux_path` names for its config."""
        if self.run.get("exploration"):
            return None  # hourly archive has no prepared dew-point/cloud fields; never substitute SYNOP
        meta = (self.run.get("data") or {}).get("aux_observations") or {}
        p = ROOT / "data" / "processed" / str(meta.get("file", ""))
        if meta.get("file") and p.exists():
            return p if _sha256(p) == meta.get("sha256") else None
        cfg = self.path / "config.yaml"
        if not cfg.exists():
            return None
        from weather_lab.common import load_config
        from weather_lab.data import aux_path

        p = aux_path(load_config(cfg))
        return p if p.exists() else None

    def aux_observations(self) -> pd.DataFrame | None:
        p = self.aux_path()
        return _parquet(p) if p else None

    def station_coords(self) -> dict:
        return {s["id"]: (s["lat"], s["lon"]) for s in self.stations() if "lat" in s and "lon" in s}

    def arome_extract(self) -> pd.DataFrame | None:
        """The AROME station extract the run's AROME inputs came from (for exploratory reruns), if present."""
        if self.run.get("exploration"):
            meta = self.run["hourly_data"].get("arome")
            return _parquet(self.path / meta["file"]) if meta else None
        for p in (self.path / "arome_station_extract.parquet",
                  ROOT / "data" / "processed" / f"arome_{self.suite}.parquet"):
            if p.exists():
                return _parquet(p)
        return None


@lru_cache(maxsize=16)
def _parquet(path: Path, columns: tuple | None = None, filters: tuple = ()) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns, filters=list(filters) or None)


@lru_cache(maxsize=8)
def _case_availability(path: Path) -> pd.DataFrame:
    p = path / "case_availability.csv"
    return pd.read_csv(p, dtype={"station": str}) if p.exists() else pd.DataFrame(columns=["case_id", "status", "reason"])


@lru_cache(maxsize=16)
def _sha256(path: Path) -> str:
    from weather_lab.common import sha256_file

    return sha256_file(path)


def find_run(suite: str, prefer_local: bool = True) -> Path | None:
    """Latest local run of a benchmark (a combined multi-candidate run wins), else the installed example."""
    local = [p.parent for p in (ROOT / "runs" / suite).glob("*/run.json")]
    if prefer_local and local:
        meta = {d: read_json(d / "run.json") for d in local}
        combined = [d for d in local if "combined_from" in meta[d]]
        return max(combined or local, key=lambda d: (meta[d]["created_at"], d.name))
    example = ROOT / "data" / "examples" / suite
    return example if (example / "run.json").exists() else None


@lru_cache(maxsize=8)
def load_run(path: Path) -> Run:
    source = "cached example (offline replay)" if "examples" in path.parts else "local run"
    run = read_json(path / "run.json")
    tables = {}
    for name in ("metrics_point", "metrics_probabilistic", "availability", "paired_differences", "tail_warm_t",
                 "case_scores", "probabilistic_not_applicable"):
        p = path / f"{name}.csv"
        if p.exists():
            df = pd.read_csv(p, dtype={"station": str})
            if "lead_h" in df:
                df["lead_h"] = df["lead_h"].astype("Int64")
            tables[name] = df
    return Run(run["suite"], path, source, run, read_json(path / "manifest.json"), tables)


def saved_configurations(suite: str) -> dict[str, dict]:
    """Small discovery records only; no prediction reads, network calls or inference."""
    from weather_lab.development import location, saved_runs

    found, seen = {}, set()
    def priority(item):
        p = Path(item[1])
        complete_matrix = (p.parents[1] / "comparisons/provenance.json").is_file()
        return not complete_matrix, not p.is_relative_to(location())

    for label, path in sorted(saved_runs(suite), key=priority):
        p = Path(path)
        summary = p / "matrix_setup.json"
        rec = read_json(summary if summary.exists() else p / "run.json")
        c = rec["forecast_config"]
        digest = rec.get("run_json_sha256")
        if digest and digest in seen:
            continue  # the same immutable record may exist locally and on the bucket mount
        if c["step_hours"] == 1 and c["horizon_h"] == 48:
            if digest:
                seen.add(digest)
            found[path] = {"label": rec.get("configuration_id", label).replace("/", " · ").replace("-prediction", " prediction"),
                           "config": c, "summary": rec}
    labels = [r["label"] for r in found.values()]
    for path, record in found.items():
        if labels.count(record["label"]) > 1:
            record["label"] += f" · {Path(path).name}"
    return found


@lru_cache(maxsize=4)
def _matrix_files(directory: Path):
    """Verify and cache the small frozen comparison files, never read model predictions."""
    from weather_lab.common import sha256_file

    record = read_json(directory / "provenance.json")
    for filename, key in (("shared_arome_metrics.csv", "scores_sha256"), ("common_support.parquet", "support_sha256")):
        if sha256_file(directory / filename) != record[key]:
            raise ValueError(f"Frozen matrix checksum differs: {filename}")
    scores = pd.read_csv(directory / "shared_arome_metrics.csv", dtype={"station": str},
                         usecols=["split", "model", "variable", "station", "season", "lead_h", "mae", "n", "support_sha256"])
    if not scores.support_sha256.eq(record["support_sha256"]).all():
        raise ValueError("Frozen matrix scores refer to different common support")
    return record, scores, _parquet(directory / "common_support.parquet")


def saved_matrix(suite: str, catalog: dict):
    """Discover local comparisons and their explicitly published bucket counterpart."""
    from weather_lab.development import location

    directories = {Path(p).parents[1] / "comparisons" for p in catalog}
    directories.update(p.parent for p in (location().parent / "benchmark-matrices").glob("*/provenance.json"))
    candidates = []
    for directory in sorted(directories):
        if not (directory / "provenance.json").is_file():
            continue
        record = read_json(directory / "provenance.json")
        if record.get("suite") != suite:
            continue
        paths = {name: next((path for path, r in catalog.items()
                            if r["summary"].get("run_json_sha256") == c["run_json_sha256"]
                            and r["summary"].get("configuration_id") == name), None)
                 for name, c in record["configurations"].items()}
        candidates.append((sum(p is not None for p in paths.values()), directory, paths))
    if not candidates:
        return None
    _, directory, paths = max(candidates, key=lambda c: c[0])
    record, scores, support = _matrix_files(directory)
    for name, path in paths.items():
        if path and _sha256(Path(path) / "run.json") != record["configurations"][name]["run_json_sha256"]:
            raise ValueError("Frozen matrix configuration differs from installed run provenance")
    return {"provenance": record, "scores": scores, "support": support, "paths": paths}


@lru_cache(maxsize=8)
def verified_run(path: Path) -> Run:
    """Verify immutable saved payloads once per cached run, reusing the regular UI loader."""
    from weather_lab.development import verify_run

    verify_run(path)
    return load_run(path)


def available_runs(prefer_local: bool = True) -> dict[str, Run]:
    out = {}
    for suite in EXPERIMENT_ORDER:
        p = find_run(suite, prefer_local)
        if p is not None:
            out[suite] = load_run(p)
    return out


def hourly_exploration() -> Run | None:
    """Verified local hourly data, with no saved predictions or benchmark case manifest."""
    from weather_lab.forecast_data import OBSERVATIONS, SUITE, make_case
    from weather_lab.rag_archive import issued_forecasts

    path = ROOT / "data/processed"
    meta_path = (path / OBSERVATIONS).with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    meta = read_json(meta_path)
    if not meta.get("exploration_only") or _sha256(path / OBSERVATIONS) != meta["sha256"]:
        raise ValueError("Hourly exploration observations failed their checksum")
    cases = [make_case(meta, s["id"], s["last"]).to_dict() for s in meta["stations"]]
    manifest = {"cases": cases, "dataset": {"stations": meta["stations"]},
                "protocol": {"stations": [s["id"] for s in meta["stations"]], "variables": ["t", "pmer", "u", "v"],
                             "forecast": {"step_hours": 1, "horizon_steps": 48, "context_days": 30}}}
    run = Run(SUITE, path, "hourly observations (no saved forecasts)",
              {"run_id": meta["sha256"], "label": "Hourly exploration", "exploration": True,
               "hourly_data": meta, "systems": []}, manifest,
              {"metrics_point": pd.DataFrame({"split": ["exploratory"]})})
    run.observations()  # preload once; calendar edits use metadata, not file reads
    if meta.get("arome"):
        if _sha256(path / meta["arome"]["file"]) != meta["arome"]["sha256"]:
            raise ValueError("Hourly issued AROME extract failed its checksum")
        issued = issued_forecasts(run.arome_extract())
        run.issued_times = issued.groupby(["station", "run"], as_index=False).available_at.max()
    return run


def calendar_case(run: Run, station: str, value) -> str:
    """Normalize a UTC calendar value to a case key; invalid values reach draft validation."""
    if value is None or value == "":
        return ""
    if isinstance(value, str) and "/" in value:
        return value
    try:
        origin = pd.Timestamp(value)
        origin = origin.tz_localize("UTC") if origin.tzinfo is None else origin.tz_convert("UTC")
        if pd.isna(origin) or origin != origin.floor("h"):
            return str(value)
    except (ValueError, TypeError):
        return str(value)
    if run.run.get("exploration"):
        return f"{run.suite}/{station}/{origin:%Y%m%dT%H}"
    cases = run.cases()
    match = cases[cases.station.eq(station) & cases.origin.eq(origin)]
    return match.case_id.iloc[0] if len(match) else str(value)


def calendar_value(run: Run, case_id: str) -> str:
    return run.case(case_id).origin.strftime("%Y-%m-%d %H:%M:%S")


def calendar_info(run: Run, station: str) -> str:
    if run.run.get("exploration"):
        meta = run.run["hourly_data"]
        record = next(s for s in meta["stations"] if s["id"] == station)
        return f"UTC · hourly · {pd.Timestamp(meta['start']):%Y-%m-%d} through {pd.Timestamp(record['last']):%Y-%m-%d %H:%M}. Missing/QC-rejected observations remain missing."
    return f"UTC · saved {run.step_hours}-hourly replay origins only. For any hourly date from 2022, select Hourly exploration."


def headline(run: Run, variable: str, split: str | None = None, models: list[str] | None = None) -> pd.DataFrame:
    """All-lead, all-station scores of the headline systems (default: the experiment's), from saved tables."""
    exp = EXPERIMENTS.get(run.suite, {})
    point, prob = run.tables["metrics_point"], run.tables["metrics_probabilistic"]
    split = split or run.splits[-1]
    sel = lambda d: d[(d["split"] == split) & (d["variable"] == variable) & (d["station"] == "ALL")  # noqa: E731
                      & (d["season"] == "ALL") & d["lead_h"].isna()]
    pt = sel(point).set_index("model")
    pr = sel(prob).set_index("model")
    rows = []
    for m in exp.get("headline", []) if models is None else models:
        if m in pt.index:
            rows.append({"model": m, "name": model_name(m), "mae": pt.loc[m, "mae"], "rmse": pt.loc[m, "rmse"],
                         "n": int(pt.loc[m, "n"]),
                         "coverage80": pr.loc[m, "coverage80"] if m in pr.index else None})
    return pd.DataFrame(rows)


def headline_to(run: Run, variable: str, models: list[str], horizon_h: int, split: str | None = None) -> pd.DataFrame:
    """`headline` recomputed from saved predictions on leads <= horizon_h, on the scorer's common valid set
    over every saved system of the run (so at the full horizon it equals the saved tables)."""
    from weather_lab.horizon import scored_rows

    split = split or run.splits[-1]
    preds = run.predictions()
    have = preds.loc[preds["variable"] == variable, "model"].unique().tolist()
    out = {}
    for metric, col in (("mae", "abs_err"), ("coverage80", "in80")):
        cand = have if metric == "mae" else [m for m in have if (preds.loc[preds["model"] == m, "output_type"]
                                                                 == "quantiles").any()]
        if not cand:
            continue
        rows = scored_rows(preds, run.targets(), run.cases(), variable, cand, "pinball" if metric != "mae" else "mae",
                           split=split)
        rows = rows[rows["lead_h"] <= horizon_h]
        out[metric] = rows.groupby("model")[col].agg(["mean", "size"])
        if metric == "mae":
            out["rmse"] = rows.groupby("model")["sq_err"].mean().pow(0.5)
    pt = out.get("mae", pd.DataFrame(columns=["mean", "size"]))
    pr = out.get("coverage80", pd.DataFrame(columns=["mean", "size"]))
    return pd.DataFrame([{"model": m, "name": model_name(m), "mae": pt.loc[m, "mean"], "rmse": out["rmse"].loc[m],
                          "n": int(pt.loc[m, "size"]),
                          "coverage80": pr.loc[m, "mean"] if m in pr.index else None}
                         for m in models if m in pt.index])


def station_label(sid: str) -> str:
    return f"{sid} {STATION_NAMES.get(sid, '')}".strip()


def filter_metrics(df: pd.DataFrame, split: str, variable: str, station: str, season: str,
                   lead: str, statuses: list[str] | None = None) -> pd.DataFrame:
    m = (df["split"] == split) & (df["variable"] == variable) & (df["station"] == station) & (df["season"] == season)
    m &= df["lead_h"].isna() if lead == "all" else (df["lead_h"] == int(lead))
    out = df[m].copy()
    out.insert(1, "training", out["model"].map(training_status))
    if statuses:
        out = out[out["training"].isin(statuses)]
    return out


def case_forecast(run: Run, case_id: str, variable: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """(forecast rows for every system, observations window, case record)."""
    p = run.predictions()
    rows = p[(p["case_id"] == case_id) & (p["variable"] == variable)].sort_values(["model", "lead_h"])
    case = run.case(case_id)
    obs = run.observations()
    obs = obs[obs["station"] == case["station"]][["time", variable]].rename(columns={variable: "observed"})
    return rows, obs, case


def forecast_csv(rows: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    cols = ["case_id", "station", "origin", "valid_time", "lead_h", "variable", "unit", "model", "model_revision",
            "output_type", "point", "arome_run", "arome_available_at", "arome_baseline", "predicted_error"] \
        + [qcol(q) for q in QUANTILE_LEVELS if qcol(q) in rows]
    out = rows[[c for c in cols if c in rows]].merge(obs.rename(columns={"time": "valid_time"}), on="valid_time",
                                                       how="left")
    return out


# --------------------------------------------------------------------------- input history (provenance)

NON_HISTORY = {"persistence": "last observation at or before the forecast time",
               "seasonal-naive-24h": "observation 24 h before each valid time",
               "climatology": "no case history (quantiles fitted on the fitting period)",
               "arome-raw": "none (issued AROME run)"}
AROME_HISTORY = {None: "not used", "archived": "archived AROME runs available at the time",
                 "substituted_observations": "station observations substituted (not AROME)",
                 "observations": "station observations substituted (not AROME)",
                 "missing": "missing (past segment left empty)"}


def steps_text(days: float, step_hours: int) -> str:
    hours = int(round(days * 24))
    return f"{days:g} days = {hours} hours = {hours // step_hours} {'three-hourly' if step_hours == 3 else f'{step_hours}-hourly'} steps"


def input_history(run: Run) -> pd.DataFrame:
    if "systems" not in (getattr(run, "run", None) or {}):        # no provenance recorded: everything Unknown
        return pd.DataFrame(columns=["summary", "requested_days", "arome_history"]).rename_axis("model")
    return _input_history(run.path) if run.path.exists() else _compute_input_history(run)


@lru_cache(maxsize=8)
def _input_history(path: Path) -> pd.DataFrame:
    return _compute_input_history(load_run(path))


def _compute_input_history(run: Run) -> pd.DataFrame:
    """Per system: requested and effective target history, cadence and AROME-history construction/coverage,
    recomputed from the run's own provenance (run.json params, manifest, saved observation windows, AROME
    extract). Unknown when the run did not record a context."""
    step = pd.Timedelta(hours=run.step_hours)
    cases = run.cases()
    cases = cases[cases["eligible"]] if "eligible" in cases else cases
    obs = run.observations()
    arome = run.arome_extract()
    rows = []
    for sysrec in run.run["systems"]:
        mid, params = sysrec["id"], sysrec.get("params") or {}
        base = {"model": mid, "cadence_h": run.step_hours, "requested_days": None, "requested_steps": None,
                "effective_min": None, "effective_max": None, "cases_short": 0, "n_cases": len(cases),
                "arome_history": "", "arome_cov_min": None, "arome_cov_max": None, "arome_cov_median": None}
        if not mid.startswith("timestral"):
            rows.append({**base, "summary": NON_HISTORY.get(mid, "see Methods")})
            continue
        days = params.get("context_days")
        if days is None:
            rows.append({**base, "summary": "Unknown (context not recorded in this run)"})
            continue
        steps = int(days * 24 // run.step_hours)
        delay = pd.Timedelta(hours=(sysrec.get("input_alignment") or {}).get("observation_delay_hours", 0))
        variables = [v for v in sysrec.get("variables", []) if v in obs]
        counts = []
        for c in cases.itertuples():
            end = c.origin - delay
            o = obs[(obs["station"] == c.station) & (obs["time"] > end - steps * step) & (obs["time"] <= end)]
            counts.append(int(o[variables].notna().all(axis=1).sum()) if variables else 0)
        counts = pd.Series(counts, dtype=int)
        mode = params.get("arome")
        cov = None
        if mode == "archived" and arome is not None and len(cases):
            from weather_lab import arome as arome_mod

            origins = cases["origin"].dt.tz_convert(None).to_numpy().astype("datetime64[ns]")
            times = origins[:, None] - delay.to_timedelta64() + (
                np.arange(-(steps - 1), 1) * np.timedelta64(run.step_hours, "h"))[None]
            vals, _, _ = arome_mod.archived_history(arome, cases["station"].to_numpy(), origins, times, ["t"])
            cov = pd.Series(np.isfinite(vals[:, 0]).mean(axis=1))
        rows.append({**base, "requested_days": days, "requested_steps": steps,
                     "effective_min": int(counts.min()), "effective_max": int(counts.max()),
                     "cases_short": int((counts < steps).sum()), "arome_history": AROME_HISTORY.get(mode, str(mode)),
                     "arome_cov_min": None if cov is None else float(cov.min()),
                     "arome_cov_max": None if cov is None else float(cov.max()),
                     "arome_cov_median": None if cov is None else float(cov.median()),
                     "summary": history_text(days, steps, int(counts.min()), int(counts.max()), run.step_hours)})
    return pd.DataFrame(rows).set_index("model")


def history_text(days: float, steps: int, lo: int, hi: int, step_hours: int) -> str:
    eff = f"{lo}" if lo == hi else f"{lo}–{hi}"
    return f"{days:g} d ({int(days * 24)} h, {steps} × {step_hours} h steps; observed {eff} of {steps} per case)"


def arome_history_text(r: pd.Series) -> str:
    if not r["arome_history"] or r["arome_history"] == "not used":
        return "not used"
    if r["arome_cov_min"] is None or pd.isna(r["arome_cov_min"]):
        return r["arome_history"]
    return (f"{r['arome_history']}: {r['arome_cov_min']:.0%}–{r['arome_cov_max']:.0%} of the window "
            f"(median {r['arome_cov_median']:.0%})")
