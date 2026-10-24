"""Read saved runs for the UI. Nothing here runs a benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from weather_lab.common import ROOT, STATION_NAMES, read_json
from weather_lab.forecast import QUANTILE_LEVELS, qcol

SUITES = {
    "synop-historical-v1": "Station history · 2023–24 (SYNOP smoke)",
    "arome-issued-v2": "AROME comparison · Oct 2026",
    "arome-issued-v1": "AROME comparison · Oct 2026 (v1, superseded)",
}
SUITE_DESCRIPTIONS = {
    "synop-historical-v1": "Historical observation suite: 2 stations, semi-monthly origins in 2023 (validation) and "
                           "2024 (test).",
    "arome-issued-v2": "Issued AROME runs from the 14-day archive: 5 stations, 12 daily origins; historical AROME "
                       "input explicit per system.",
    "arome-issued-v1": "Earlier version of the AROME comparison, kept unchanged: its conditioned system substituted "
                       "station observations for historical AROME.",
}
# Fixed entity -> colour (reference categorical palette, fixed order; colour follows the system).
# 209M variants share their 28M counterpart's hue with a dashed line as secondary encoding.
MODEL_STYLE = {
    "timestral-28m-mv-cal": ("#2a78d6", "solid"),
    "timestral-209m-mv-cal": ("#2a78d6", "dash"),
    "arome-raw": ("#eb6834", "solid"),
    "timestral-28m-mv-cal-arome": ("#1baf7a", "solid"),
    "climatology": ("#eda100", "dot"),
    "persistence": ("#e87ba4", "dash"),
    "seasonal-naive-24h": ("#008300", "dashdot"),
    "timestral-28m-mv": ("#6250d6", "solid"),
    "timestral-28m-t": ("#e34948", "solid"),
    "timestral-28m-mv-cal-arome-nanpast": ("#8a8984", "longdash"),
    "timestral-28m-mv-cal-arome-archived": ("#1baf7a", "solid"),
    "timestral-209m-mv-cal-arome-archived": ("#1baf7a", "dash"),
    "timestral-28m-mv-cal-arome-futureonly": ("#6250d6", "dash"),
    "timestral-28m-mv-cal-arome-substobs": ("#e34948", "dot"),
}
EXPLORATORY_STYLE = ("#52514e", "dot")


EXPLORATORY_SIZE_STYLE = {"5M": ("#a3a29c", "dot"), "28M": ("#6b6a65", "dash"), "209M": ("#3d3c39", "longdashdot")}


def style(model: str) -> tuple[str, str]:
    if model.startswith("exploratory"):   # session-only runs: neutral greys, by checkpoint size
        return next((v for k, v in EXPLORATORY_SIZE_STYLE.items() if f"-{k} " in model), EXPLORATORY_STYLE)
    return MODEL_STYLE.get(model, EXPLORATORY_STYLE)


def training_status(model: str) -> str:
    if model.startswith("timestral"):
        return "pretrained, no local fine-tuning"
    if model == "climatology":
        return "fitted on fitting period"
    if model == "arome-raw":
        return "external NWP (issued run)"
    return "no fitting"


@dataclass
class Run:
    suite: str
    path: Path
    source: str                    # "local run" | "cached example"
    run: dict
    manifest: dict
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.run['run_id']} · {self.run['label']} · {self.source}"

    @property
    def splits(self) -> list[str]:
        return sorted(self.tables["metrics_point"]["split"].unique(), key=lambda s: ["validation", "test"].index(s))

    @property
    def models(self) -> list[str]:
        return [s["id"] for s in self.run["systems"]]

    def predictions(self) -> pd.DataFrame:
        return _parquet(self.path / "predictions.parquet")

    def targets(self) -> pd.DataFrame:
        return _parquet(self.path / "targets.parquet")

    def observations(self) -> pd.DataFrame:
        return _parquet(self.path / "observation_windows.parquet")

    def cases(self) -> pd.DataFrame:
        df = pd.DataFrame(self.manifest["cases"])
        df["origin"] = pd.to_datetime(df["origin"], utc=True)
        return df

    def stations(self) -> list[dict]:
        return self.manifest["dataset"]["stations"]


@lru_cache(maxsize=16)
def _parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def find_run(suite: str, prefer_local: bool = True) -> Path | None:
    local = sorted((ROOT / "runs" / suite).glob("*/run.json"))
    if prefer_local and local:
        return local[-1].parent
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


def available_runs(prefer_local: bool = True) -> dict[str, Run]:
    out = {}
    for suite in SUITES:
        p = find_run(suite, prefer_local)
        if p is not None:
            out[suite] = load_run(p)
    return out


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
    case = run.cases().set_index("case_id").loc[case_id]
    obs = run.observations()
    obs = obs[obs["station"] == case["station"]][["time", variable]].rename(columns={variable: "observed"})
    return rows, obs, case


def forecast_csv(rows: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    cols = ["case_id", "station", "origin", "valid_time", "lead_h", "variable", "unit", "model", "model_revision",
            "output_type", "point"] + [qcol(q) for q in QUANTILE_LEVELS if qcol(q) in rows]
    out = rows[[c for c in cols if c in rows]].merge(obs.rename(columns={"time": "valid_time"}), on="valid_time",
                                                       how="left")
    return out
