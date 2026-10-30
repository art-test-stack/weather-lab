"""Météo-France "Archive Synop OMM" download, cleaning and regularisation.

Facts used here come from the archive's technical description
(OBSERVATIONS_Descriptif_Technique_Données_SYNOP_OMM.pdf, version 2026-04-20) and from
inspecting the files themselves; see docs/sources.md:

- one `synop_YYYY.csv.gz` per year, `;`-separated, one row per station and validity time;
- `geo_id_wmo` is the 5-character WMO station id (leading zero kept as a string);
- `t` is in K, `pmer` (mean-sea-level pressure) and `pres` (station pressure) in Pa,
  `ff` (10-min mean wind) in m/s, `dd` (10-min mean wind direction) in degrees;
- the files carry no quality-flag columns; a missing value is an empty field;
- cadence for metropolitan stations is 3-hourly (00, 03, ..., 21 UTC).
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from weather_lab.common import Config, now_utc, read_json, sha256_file, write_json

DATASET_API = "https://www.data.gouv.fr/api/1/datasets/archive-synop-omm/"
DATASET_PAGE = "https://www.data.gouv.fr/datasets/archive-synop-omm"
LICENCE = "Licence Ouverte / Open Licence v2.0 (Etalab), data.gouv.fr licence id 'lov2'"
STEP = pd.Timedelta(hours=3)
RAW_COLUMNS = ["geo_id_wmo", "name", "lat", "lon", "insert_time", "validity_time", "pmer", "pres", "dd", "ff", "t"]
OBS_VARIABLES = ["t", "pmer", "u", "v"]


# --------------------------------------------------------------------------- fetch

def resolve_resources(session: requests.Session | None = None) -> dict:
    """Resolve current resource URLs from the data.gouv.fr dataset metadata."""
    s = session or requests.Session()
    meta = s.get(DATASET_API, timeout=60).json()
    out: dict = {"dataset_id": meta["id"], "license": meta.get("license"), "years": {}}
    for r in meta["resources"]:
        title = r["title"]
        if title.startswith("synop_") and title[6:].isdigit():
            out["years"][int(title[6:])] = {"url": r["url"], "resource_id": r["id"],
                                            "last_modified": r.get("last_modified")}
        elif "liste_stations" in title:
            out["stations"] = {"url": r["url"], "resource_id": r["id"]}
        elif "Descriptif_Technique" in title:
            out["technical_description"] = {"url": r["url"], "resource_id": r["id"]}
    return out


def _download(url: str, dest: Path, session: requests.Session) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        headers = {k: r.headers.get(k) for k in ("ETag", "Last-Modified", "Content-Length")}
        with open(tmp, "wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
    tmp.rename(dest)
    return {"url": url, "retrieved_at": now_utc(), "bytes": dest.stat().st_size, "sha256": sha256_file(dest),
            "http": headers}


def fetch_synop(cfg: Config, log=print) -> list[dict]:
    """Download each configured year once into `paths.raw`; returns provenance records."""
    raw = cfg.dir("raw")
    s = requests.Session()
    resources = resolve_resources(s)
    records = []
    targets = [(f"synop_{y}.csv.gz", resources["years"][y]["url"]) for y in cfg["data"]["years"]]
    targets.append(("synop_stations.csv", resources["stations"]["url"]))
    for name, url in targets:
        dest = raw / name
        sidecar = raw / f"{name}.provenance.json"
        if dest.exists() and sidecar.exists():
            rec = read_json(sidecar)
            log(f"cached  {name} ({rec['bytes'] / 1e6:.1f} MB)")
        else:
            log(f"fetch   {url}")
            rec = _download(url, dest, s)
            rec["dataset"] = DATASET_PAGE
            rec["license"] = LICENCE
            write_json(sidecar, rec)
            log(f"saved   {name} ({rec['bytes'] / 1e6:.1f} MB, sha256 {rec['sha256'][:12]})")
        records.append({"file": name, **rec})
    return records


# --------------------------------------------------------------------------- clean

def wind_components(ff: np.ndarray, dd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Meteorological convention: `dd` is the direction the wind blows FROM, clockwise from north.

    u (eastward) = -ff sin(dd), v (northward) = -ff cos(dd). A westerly (dd=270) gives u>0, v=0.
    Calm (ff == 0) gives u = v = 0 whatever dd. dd outside [0, 360] (e.g. variable-direction
    codes) or dd == 0 with ff > 0 (inconsistent; SYNOP reports north as 360) is treated as missing.
    """
    ff = np.asarray(ff, dtype=float)
    dd = np.asarray(dd, dtype=float)
    rad = np.deg2rad(dd)
    u = -ff * np.sin(rad)
    v = -ff * np.cos(rad)
    bad = (dd < 0) | (dd > 360) | ((dd == 0) & (ff > 0))
    calm = ff == 0
    u = np.where(calm, 0.0, np.where(bad, np.nan, u))
    v = np.where(calm, 0.0, np.where(bad, np.nan, v))
    return u, v


def read_year(path: Path, stations: list[str]) -> pd.DataFrame:
    """Read one annual file, keeping only the configured stations (chunked)."""
    frames = []
    with gzip.open(path, "rb") as f:
        text = io.TextIOWrapper(f, encoding="utf-8")
        for chunk in pd.read_csv(text, sep=";", usecols=RAW_COLUMNS, dtype={"geo_id_wmo": str, "name": str},
                                 chunksize=200_000):
            frames.append(chunk[chunk["geo_id_wmo"].isin(stations)])
    return pd.concat(frames, ignore_index=True)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Deduplicate, convert units and derive wind components. Returns (frame, report)."""
    df = raw.copy()
    df["time"] = pd.to_datetime(df["validity_time"], utc=True)
    df["insert_time"] = pd.to_datetime(df["insert_time"], utc=True, errors="coerce")
    df["_row"] = np.arange(len(df))
    # Duplicates: keep the most recently inserted record; ties (or no insert_time) -> last row in file order.
    df = df.sort_values(["geo_id_wmo", "time", "insert_time", "_row"], na_position="first", kind="stable")
    n_dup = int(df.duplicated(["geo_id_wmo", "time"], keep="last").sum())
    df = df.drop_duplicates(["geo_id_wmo", "time"], keep="last")
    on_grid = (df["time"].dt.minute == 0) & (df["time"].dt.hour % 3 == 0)
    n_off = int((~on_grid).sum())
    df = df[on_grid]
    out = pd.DataFrame({
        "station": df["geo_id_wmo"].astype(str).to_numpy(),
        "time": df["time"].to_numpy(),
        "t": df["t"].to_numpy(dtype=float) - 273.15,              # K -> °C
        "pmer": df["pmer"].to_numpy(dtype=float) / 100.0,          # Pa -> hPa (sea level, not station pressure)
        "ff": df["ff"].to_numpy(dtype=float),                      # m/s, 10-min mean (not gust)
        "dd": df["dd"].to_numpy(dtype=float),
    })
    out["u"], out["v"] = wind_components(out["ff"].to_numpy(), out["dd"].to_numpy())
    report = {"duplicates_dropped": n_dup, "off_grid_rows_dropped": n_off,
              "dd_zero_with_wind": int(((out["dd"] == 0) & (out["ff"] > 0)).sum())}
    return out, report


def regularise(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Reindex every station on a common 3-hourly UTC axis; missing rows stay NaN."""
    axis = pd.date_range(start, end, freq=STEP, tz="UTC")
    frames = []
    for station, g in df.groupby("station", sort=True):
        g = g.set_index("time").reindex(axis)
        g["station"] = station
        frames.append(g.rename_axis("time").reset_index())
    cols = ["station", "time", "t", "pmer", "u", "v", "ff", "dd"]
    return pd.concat(frames, ignore_index=True)[cols]


def processed_path(cfg: Config) -> Path:
    years = cfg["data"]["years"]
    return cfg.dir("processed") / f"synop_{'-'.join(cfg.stations)}_{min(years)}-{max(years)}.parquet"


def prepare_synop(cfg: Config, log=print) -> tuple[pd.DataFrame, dict]:
    raw_dir = cfg.dir("raw")
    stations = cfg.stations
    frames, sources, reports = [], [], {}
    for y in cfg["data"]["years"]:
        path = raw_dir / f"synop_{y}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `weather-lab data fetch --config {cfg.path}` first")
        prov = read_json(raw_dir / f"synop_{y}.csv.gz.provenance.json")
        sources.append({"file": path.name, "url": prov["url"], "sha256": prov["sha256"],
                        "retrieved_at": prov["retrieved_at"]})
        raw = read_year(path, stations)
        cleaned, rep = clean(raw)
        reports[y] = rep
        frames.append((y, cleaned, raw))
        log(f"read    synop_{y}: {len(raw)} rows for {len(stations)} stations, {rep}")
    obs = pd.concat([c for _, c, _ in frames], ignore_index=True)
    years = cfg["data"]["years"]
    # Axis: span of configured years, truncated at the last observation actually present
    # (the current-year file ends at its last daily update).
    start = pd.Timestamp(f"{min(years)}-01-01", tz="UTC")
    end = min(pd.Timestamp(f"{max(years)}-12-31 21:00", tz="UTC"), obs["time"].max())
    # Non-contiguous years (e.g. 2022-2024 + 2026) leave a NaN gap: honest missingness, never filled.
    obs = regularise(obs, start, end)

    station_meta = _station_metadata(raw_dir / "synop_stations.csv", pd.concat([r for _, _, r in frames]))
    coverage = (obs.groupby("station")[OBS_VARIABLES].apply(lambda g: g.notna().mean()).round(4))
    for s in station_meta:
        s["coverage"] = coverage.loc[s["id"]].to_dict() if s["id"] in coverage.index else {}
    out = processed_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    obs.to_parquet(out, index=False)
    meta = {
        "created_at": now_utc(),
        "file": out.name,
        "sha256": sha256_file(out),
        "rows": len(obs),
        "time_axis": {"start": str(start), "end": str(end), "step": "3h UTC"},
        "stations": station_meta,
        "sources": sources,
        "license": LICENCE,
        "cleaning": reports,
        "processing": [
            "kept configured stations only (geo_id_wmo as 5-char string)",
            "duplicates on (station, validity_time): keep latest insert_time, then last row in file order",
            "dropped rows not on the 3-hourly UTC grid",
            "t: K - 273.15 -> °C; pmer: Pa / 100 -> hPa (mean-sea-level, station pressure `pres` unused)",
            "u, v from ff (10-min mean, m/s; gusts unused) and dd with the meteorological FROM convention",
            "reindexed to a common 3-hourly UTC axis; missing values remain NaN (no filling)",
        ],
    }
    write_json(out.with_suffix(".meta.json"), meta)
    log(f"wrote   {out.relative_to(cfg.dir('processed').parent.parent)} ({len(obs)} rows, sha256 {meta['sha256'][:12]})")
    return obs, meta


def _station_metadata(station_csv: Path, raw: pd.DataFrame) -> list[dict]:
    listing = pd.read_csv(station_csv, sep=";", dtype={"ID": str})
    out = []
    for sid in sorted(raw["geo_id_wmo"].unique()):
        row = listing[listing["ID"] == sid]
        if len(row):
            r = row.iloc[0]
            out.append({"id": sid, "name": r["Nom"], "lat": float(r["Latitude"]), "lon": float(r["Longitude"]),
                        "altitude_m": float(r["Altitude"])})
        else:
            r = raw[raw["geo_id_wmo"] == sid].iloc[0]
            out.append({"id": sid, "name": r["name"], "lat": float(r["lat"]), "lon": float(r["lon"]),
                        "altitude_m": None})
    return out


def load_observations(cfg: Config) -> tuple[pd.DataFrame, dict]:
    path = processed_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `weather-lab data prepare --config {cfg.path}` first")
    return pd.read_parquet(path), read_json(path.with_suffix(".meta.json"))


# --------------------------------------------------------------------------- auxiliary observations (features)

AUX_RAW_COLUMNS = ["geo_id_wmo", "insert_time", "validity_time", "td", "n"]


def aux_path(cfg: Config) -> Path:
    years = cfg["data"]["years"]
    return cfg.dir("processed") / f"synop_aux_{'-'.join(cfg.stations)}_{min(years)}-{max(years)}.parquet"


def prepare_aux(cfg: Config, log=print) -> tuple[pd.DataFrame, dict]:
    """Dew point and total cloud cover on the same 3-hourly axis as the main observations, for the
    context-enrichment features only. A separate file: the frozen observation files are untouched.

    td: K - 273.15 -> °C. n (total cloud cover, %): values above 100 (101 = sky obscured) -> missing.
    Duplicates follow `clean` (latest insert_time, then last row). Nothing is filled.
    """
    raw_dir, stations = cfg.dir("raw"), cfg.stations
    frames, sources = [], []
    for y in cfg["data"]["years"]:
        path = raw_dir / f"synop_{y}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `weather-lab data fetch --config {cfg.path}` first")
        prov = read_json(raw_dir / f"synop_{y}.csv.gz.provenance.json")
        sources.append({"file": path.name, "sha256": prov["sha256"]})
        with gzip.open(path, "rb") as f:
            text = io.TextIOWrapper(f, encoding="utf-8")
            for chunk in pd.read_csv(text, sep=";", usecols=AUX_RAW_COLUMNS, dtype={"geo_id_wmo": str},
                                     chunksize=200_000):
                frames.append(chunk[chunk["geo_id_wmo"].isin(stations)])
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["validity_time"], utc=True)
    df["insert_time"] = pd.to_datetime(df["insert_time"], utc=True, errors="coerce")
    df["_row"] = np.arange(len(df))
    df = df.sort_values(["geo_id_wmo", "time", "insert_time", "_row"], na_position="first", kind="stable")
    df = df.drop_duplicates(["geo_id_wmo", "time"], keep="last")
    df = df[(df["time"].dt.minute == 0) & (df["time"].dt.hour % 3 == 0)]
    cloud = df["n"].to_numpy(dtype=float)
    out = pd.DataFrame({"station": df["geo_id_wmo"].to_numpy(), "time": df["time"].to_numpy(),
                        "td": df["td"].to_numpy(dtype=float) - 273.15,
                        "cloud": np.where(cloud > 100, np.nan, cloud)})
    obs, _ = load_observations(cfg)                       # same axis as the frozen observations
    out = obs[["station", "time"]].merge(out, on=["station", "time"], how="left")
    path = aux_path(cfg)
    out.to_parquet(path, index=False)
    meta = {"created_at": now_utc(), "file": path.name, "sha256": sha256_file(path), "rows": len(out),
            "sources": sources, "license": LICENCE,
            "coverage": out.groupby("station")[["td", "cloud"]].apply(lambda g: g.notna().mean()).round(4)
            .to_dict(orient="index"),
            "processing": ["td: K - 273.15 -> °C", "cloud: total cloud cover n (%), values > 100 -> missing",
                           "duplicates: latest insert_time, then last row; 3-hourly grid; no filling",
                           "axis: identical to the frozen observation file of the same config"]}
    write_json(path.with_suffix(".meta.json"), meta)
    log(f"wrote   {path.name} ({len(out)} rows, sha256 {meta['sha256'][:12]})")
    return out, meta


def load_aux(cfg: Config, log=print) -> tuple[pd.DataFrame, dict]:
    path = aux_path(cfg)
    if not path.exists():
        return prepare_aux(cfg, log)
    return pd.read_parquet(path), read_json(path.with_suffix(".meta.json"))
