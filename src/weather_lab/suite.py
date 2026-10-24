"""Suite definition: origins, cases, eligibility and the frozen manifest.

A manifest is written once by `data prepare` from observable case descriptors only
(context missingness, AROME run availability), before any forecast is scored. Re-running
`prepare` with a different protocol or different case data is refused: create a new suite
version instead of silently changing the benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from weather_lab.common import Config, now_utc, read_json, sha256_text, ts, write_json

SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def split_bounds(cfg: Config) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    return {k: (ts(a), ts(b)) for k, (a, b) in cfg["splits"].items()}


def build_origins(cfg: Config) -> list[tuple[pd.Timestamp, str]]:
    """Deterministic origins; a target window must lie wholly inside its evaluation split."""
    o, f = cfg["origins"], cfg["forecast"]
    horizon = pd.Timedelta(hours=f["step_hours"] * f["horizon_steps"])
    bounds = split_bounds(cfg)
    out = []
    for split in o["splits"]:
        start, end = bounds[split]
        if o["rule"] == "explicit":
            candidates = [ts(f"{d}T{o['hour_utc']:02d}:00Z") for d in o["dates"]]
        else:
            days = pd.date_range(start.normalize(), end.normalize(), freq="D", tz="UTC")
            if o["rule"] == "semi_monthly":
                days = days[(days.day == 1) | (days.day == 15)]
            elif o["rule"] != "daily":
                raise ValueError(f"unknown origin rule {o['rule']}")
            candidates = [d + pd.Timedelta(hours=o["hour_utc"]) for d in days]
        for origin in candidates:
            # first target at origin + 1 step, last at origin + horizon: both inside the split
            if origin >= start and origin + horizon <= end:
                out.append((origin, split))
    return out


def case_id(suite: str, station: str, origin: pd.Timestamp) -> str:
    return f"{suite}/{station}/{origin:%Y%m%dT%H}"


def build_cases(cfg: Config, obs: pd.DataFrame, arome_runs: list[dict] | None = None) -> pd.DataFrame:
    f, e = cfg["forecast"], cfg["eligibility"]
    step = pd.Timedelta(hours=f["step_hours"])
    t_ctx = f["context_days"] * 24 // f["step_hours"]
    wide = {s: g.set_index("time") for s, g in obs.groupby("station")}
    runs = {r["origin"]: r for r in (arome_runs or [])}
    rows = []
    for origin, split in build_origins(cfg):
        for station in cfg.stations:
            g = wide[station]
            ctx = g.loc[origin - (t_ctx - 1) * step: origin]
            fut = g.loc[origin + step: origin + f["horizon_steps"] * step]
            row = {"case_id": case_id(cfg.suite, station, origin), "station": station, "origin": origin,
                   "split": split, "season": SEASONS[origin.month]}
            complete_axis = len(ctx) == t_ctx and len(fut) == f["horizon_steps"]
            for var in cfg["variables"]:
                row[f"ctx_missing_{var}"] = round(1 - ctx[var].notna().sum() / t_ctx, 4)
            # Eligibility uses inputs only (fixed before scoring); missing targets are handled at scoring.
            ev = e["variable"]
            recent = ctx.loc[origin - pd.Timedelta(hours=e["require_observation_within_hours"]) + step:, ev]
            reasons = []
            if not complete_axis:
                reasons.append("time axis does not cover context+horizon")
            if 1 - row[f"ctx_missing_{ev}"] < e["min_context_observed_fraction"]:
                reasons.append(f"{ev} context observed < {e['min_context_observed_fraction']:.0%}")
            # optional: the fraction must also hold within each shorter window (common set across contexts)
            for days in e.get("windows_days", []):
                w = ctx.loc[origin - (days * 24 // f["step_hours"] - 1) * step: origin, ev]
                frac = w.notna().sum() / (days * 24 // f["step_hours"])
                row[f"ctx_observed_{ev}_{days}d"] = round(float(frac), 4)
                if frac < e["min_context_observed_fraction"]:
                    reasons.append(f"{ev} observed < {e['min_context_observed_fraction']:.0%} in last {days} d")
            if recent.notna().sum() == 0:
                reasons.append(f"no {ev} observation in last {e['require_observation_within_hours']} h")
            if arome_runs is not None:
                r = runs.get(origin.strftime("%Y-%m-%dT%H:%M:%SZ"))
                if r is None or r.get("status") != "ok":
                    reasons.append(f"no qualifying AROME run ({(r or {}).get('reason', 'not fetched')})")
                else:
                    row["arome_run"] = r["run"]
                    row["arome_available_at"] = r["available_at"]
            row["eligible"] = not reasons
            row["exclusion_reason"] = "; ".join(reasons)
            rows.append(row)
    return pd.DataFrame(rows)


def ui_examples(cases: pd.DataFrame) -> list[str]:
    """Fixed rule, decided before predictions exist: for each station x season, the earliest
    eligible origin of the last evaluation split listed (test where present)."""
    ok = cases[cases["eligible"]]
    if ok.empty:
        return []
    last_split = ok["split"].iloc[-1]
    ok = ok[ok["split"] == last_split].sort_values("origin")
    return ok.groupby(["station", "season"], sort=True).head(1)["case_id"].tolist()


def case_data_checksum(obs: pd.DataFrame, cases: pd.DataFrame, cfg: Config) -> str:
    """Checksum of exactly the observations the cases read (context + verifying targets)."""
    f = cfg["forecast"]
    step = pd.Timedelta(hours=f["step_hours"])
    t_ctx = f["context_days"] * 24 // f["step_hours"]
    parts = []
    wide = {s: g.set_index("time") for s, g in obs.groupby("station")}
    for c in cases.itertuples():
        g = wide[c.station].loc[c.origin - (t_ctx - 1) * step: c.origin + f["horizon_steps"] * step,
                                cfg["variables"]]
        parts.append((np.round(g.to_numpy(dtype=float), 4) + 0.0).tobytes())  # +0.0 folds -0.0
    return sha256_text(b"".join(parts).hex())


def protocol(cfg: Config) -> dict:
    return {k: cfg.get(k) for k in ("suite", "variant", "splits", "origins", "forecast", "eligibility", "variables")} \
        | {"stations": cfg.stations, "arome": cfg.get("arome")}


def write_manifest(cfg: Config, obs: pd.DataFrame, data_meta: dict, cases: pd.DataFrame,
                   arome_meta: dict | None = None, log=print) -> dict:
    f = cfg["forecast"]
    proto = protocol(cfg)
    manifest = {
        "suite": cfg.suite,
        "variant": cfg.get("variant"),
        "label": cfg.get("label"),
        "created_at": now_utc(),
        "protocol": proto,
        "protocol_sha256": sha256_text(str(sorted(proto.items())))[:16],
        "leads_h": [f["step_hours"] * k for k in range(1, f["horizon_steps"] + 1)],
        "dataset": {"file": data_meta["file"], "sha256": data_meta["sha256"], "sources": data_meta["sources"],
                    "license": data_meta["license"], "stations": data_meta["stations"]},
        "case_data_sha256": case_data_checksum(obs, cases, cfg),
        "observation_availability": "observations valid at or before the origin are inputs (assumes SYNOP "
                                    "reports are available at their validity time; no dissemination latency "
                                    "modelled: historical replay, not an operational backtest)",
        "eligibility": cfg["eligibility"],
        "n_cases": int(len(cases)),
        "n_eligible": int(cases["eligible"].sum()),
        "cases": _records(cases),
        "ui_examples": ui_examples(cases),
        "ui_example_rule": "per station x season, earliest eligible origin in the test split",
    }
    if arome_meta is not None:
        manifest["arome"] = {k: arome_meta[k] for k in ("file", "sha256", "runs", "grid", "package", "fields",
                                                         "extraction", "availability_rule", "license", "sources")}
        manifest["arome"]["availability_assumption"] = (
            "a run is used for an origin only if every needed file's Last-Modified time in the public archive "
            "is <= origin; leads beyond the run's 51 h are never extrapolated")
    from weather_lab.specs import check_manifest

    check_manifest(cfg, manifest)  # a clean checkout must reproduce the committed spec
    path = cfg.manifest_path
    if path.exists():
        old = read_json(path)
        same = (old["protocol_sha256"] == manifest["protocol_sha256"]
                and old["case_data_sha256"] == manifest["case_data_sha256"]
                and [c["case_id"] for c in old["cases"]] == [c["case_id"] for c in manifest["cases"]]
                and [c["eligible"] for c in old["cases"]] == [c["eligible"] for c in manifest["cases"]])
        if not same:
            raise RuntimeError(f"{path} is frozen and the new protocol/case data differ. Create a new suite "
                               f"version (new manifest path) instead of changing the benchmark.")
        log(f"manifest unchanged: {path.name} (frozen {old['created_at']})")
        return old
    write_json(path, manifest)
    log(f"froze manifest {path.name}: {manifest['n_eligible']}/{manifest['n_cases']} eligible cases")
    return manifest


def _records(cases: pd.DataFrame) -> list[dict]:
    out = cases.copy()
    out["origin"] = out["origin"].map(lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return out.replace({np.nan: None}).to_dict(orient="records")


def cases_from_manifest(manifest: dict) -> pd.DataFrame:
    df = pd.DataFrame(manifest["cases"])
    df["origin"] = pd.to_datetime(df["origin"], utc=True)
    df["station"] = df["station"].astype(str)
    return df
