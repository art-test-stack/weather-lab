"""Issued AROME forecasts at stations, from Météo-France's public 14-day PNT archive.

Only the GRIB2 messages needed (2 m temperature and mean-sea-level pressure at the
benchmark leads) are downloaded, using HTTP byte ranges on the archive's object store.
Each message is stored byte-for-byte as published, with the source URL, byte range,
ETag, Last-Modified and SHA-256, so the subset is a verifiable excerpt of the original
file (full files are ~25-70 MB each and are not stored, to respect the download budget).

Availability: the archive's `Last-Modified` header is the time the file appeared in the
archive. A run qualifies for an origin only if every file it needs was there by then.
This is evidence from the archive, not an operational dissemination log (other channels
may publish earlier), so the rule is conservative.
"""

from __future__ import annotations

import hashlib
import struct
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from weather_lab.common import Config, now_utc, read_json, sha256_file, sha256_text, ts, write_json

LICENCE = "Licence Ouverte / Open Licence v2.0 (Etalab), data.gouv.fr licence id 'lov2'"
DATASETS = {
    "archive": "https://www.data.gouv.fr/datasets/donnees-pnt-retention-14-jours",
    "package": "https://www.data.gouv.fr/datasets/paquets-arome-resolution-0-025deg",
}
# 0.025 deg packages group hourly leads into these files.
GROUPS_0025 = [(0, 6), (7, 12), (13, 18), (19, 24), (25, 30), (31, 36), (37, 42), (43, 48), (49, 51)]
# GRIB2 identification: (discipline, category, number, first-surface type, surface value).
FIELDS = {
    "t": {"key": (0, 0, 0, 103, 2), "grib": "2 m temperature (K)", "convert": lambda k: k - 273.15},
    "pmer": {"key": (0, 3, 1, 101, 0), "grib": "pressure reduced to MSL (Pa)", "convert": lambda pa: pa / 100.0},
}


def run_id(init: pd.Timestamp) -> str:
    return init.strftime("%Y-%m-%dT%H:%M:%SZ")


def file_for_lead(base_url: str, init: pd.Timestamp, lead_h: int, package: str = "SP1") -> tuple[str, str]:
    for a, b in GROUPS_0025:
        if a <= lead_h <= b:
            rid = run_id(init)
            name = f"arome__0025__{package}__{a:02d}H{b:02d}H__{rid}.grib2"
            return name, f"{base_url}/{rid}/arome/0025/{package}/{name}"
    raise ValueError(f"lead {lead_h} h outside AROME 0.025 range")


def required_leads(origin: pd.Timestamp, init: pd.Timestamp, horizon_steps: int, step_h: int) -> list[int]:
    offset = int((origin - init) / pd.Timedelta(hours=1))
    return [offset + step_h * k for k in range(1, horizon_steps + 1)]


def run_availability(last_modified: list[str], origin: pd.Timestamp) -> tuple[pd.Timestamp, bool]:
    """A run is usable at `origin` only if every file it needs was in the archive by then."""
    complete = max(pd.Timestamp(parsedate_to_datetime(h)).tz_convert("UTC") for h in last_modified)
    return complete, bool(complete <= origin)


# --------------------------------------------------------------------------- GRIB2 scanning

def parse_header(b: bytes) -> dict:
    """Identify a GRIB2 message from its first bytes (sections 0-5)."""
    if b[:4] != b"GRIB" or b[7] != 2:
        raise ValueError("not a GRIB2 message")
    info = {"discipline": b[6], "length": struct.unpack(">Q", b[8:16])[0]}
    p = 16
    while p + 5 <= len(b):
        sec_len = struct.unpack(">I", b[p:p + 4])[0]
        sec = b[p + 4]
        if sec == 4:
            info["template"] = struct.unpack(">H", b[p + 7:p + 9])[0]
            info["category"], info["number"] = b[p + 9], b[p + 10]
            info["time_unit"] = b[p + 17]
            info["forecast_time"] = struct.unpack(">i", b[p + 18:p + 22])[0]
            info["surface_type"] = b[p + 22]
            info["surface_value"] = struct.unpack(">I", b[p + 24:p + 28])[0]
        if sec == 5 or sec_len == 0:
            break
        p += sec_len
    return info


def scan_messages(url: str, session: requests.Session) -> tuple[list[dict], dict]:
    head = session.head(url, timeout=60)
    head.raise_for_status()
    size = int(head.headers["Content-Length"])
    http = {k: head.headers.get(k) for k in ("ETag", "Last-Modified", "Content-Length")}
    msgs, off = [], 0
    while off < size:
        r = session.get(url, headers={"Range": f"bytes={off}-{off + 1023}"}, timeout=60)
        r.raise_for_status()
        info = parse_header(r.content)
        info["offset"] = off
        msgs.append(info)
        off += info["length"]
    return msgs, http


def _select(msgs: list[dict], field: str, lead_h: int) -> dict | None:
    disc, cat, num, stype, sval = FIELDS[field]["key"]
    for m in msgs:
        if (m["discipline"], m.get("category"), m.get("number"), m.get("surface_type"), m.get("surface_value")) == \
                (disc, cat, num, stype, sval) and m.get("template") == 0 and m.get("time_unit") == 1 \
                and m.get("forecast_time") == lead_h:
            return m
    return None


# --------------------------------------------------------------------------- fetch

def fetch_run(cfg: Config, origin: pd.Timestamp, log=print) -> dict:
    """Fetch the needed messages of the run assigned to `origin`; returns the run record."""
    a = cfg["arome"]
    init = origin.normalize() + pd.Timedelta(hours=a["run_hour_utc"])
    leads = required_leads(origin, init, cfg["forecast"]["horizon_steps"], cfg["forecast"]["step_hours"])
    rid = run_id(init)
    out_dir = cfg.dir("raw") / "arome" / rid.replace(":", "")
    record_path = out_dir / "run.json"
    if record_path.exists():
        rec = read_json(record_path)
        log(f"cached  AROME run {rid}: {rec['status']}")
        return rec
    rec = {"run": rid, "origin": run_id(origin), "leads_h": leads, "fields": a["fields"], "files": [],
           "retrieved_at": now_utc(), "status": "ok", "license": LICENCE, "sources": DATASETS}
    if max(leads) > a["max_lead_h"]:
        rec.update(status="unavailable", reason=f"needs lead {max(leads)} h > max {a['max_lead_h']} h")
        write_json(record_path, rec)
        return rec
    by_file: dict[str, tuple[str, list[int]]] = {}
    for lead in leads:
        name, url = file_for_lead(a["base_url"], init, lead, a["package"])
        by_file.setdefault(name, (url, []))[1].append(lead)

    def work(item):
        name, (url, file_leads) = item
        s = requests.Session()
        try:
            msgs, http = scan_messages(url, s)
        except requests.HTTPError as e:
            return {"file": name, "url": url, "error": f"HTTP {e.response.status_code}"}
        wanted = []
        for lead in file_leads:
            for field in a["fields"]:
                m = _select(msgs, field, lead)
                wanted.append((field, lead, m))
        chunks, parts = [], []
        for field, lead, m in wanted:
            if m is None:
                parts.append({"field": field, "lead_h": lead, "missing": True})
                continue
            r = s.get(url, headers={"Range": f"bytes={m['offset']}-{m['offset'] + m['length'] - 1}"}, timeout=120)
            r.raise_for_status()
            if len(r.content) != m["length"]:
                raise IOError(f"short read for {name} at {m['offset']}")
            chunks.append(r.content)
            parts.append({"field": field, "lead_h": lead, "offset": m["offset"], "length": m["length"],
                          "sha256": hashlib.sha256(r.content).hexdigest()})
        out_dir.mkdir(parents=True, exist_ok=True)
        subset = out_dir / name.replace(".grib2", ".subset.grib2")
        subset.write_bytes(b"".join(chunks))
        return {"file": name, "url": url, "http": http, "n_messages_in_file": len(msgs), "messages": parts,
                "subset": subset.name, "subset_sha256": sha256_file(subset), "subset_bytes": subset.stat().st_size}

    with ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(work, sorted(by_file.items())))
    rec["files"] = files
    errors = [f for f in files if "error" in f]
    missing = [p for f in files for p in f.get("messages", []) if p.get("missing")]
    if errors or missing:
        rec.update(status="unavailable", reason=f"{len(errors)} file errors, {len(missing)} missing messages",
                   errors=errors)
    else:
        available_at, ok = run_availability([f["http"]["Last-Modified"] for f in files], origin)
        rec["available_at"] = run_id(available_at)
        rec["latency_h"] = round((available_at - init).total_seconds() / 3600, 3)
        if not ok:
            rec.update(status="unavailable", reason=f"run complete in archive at {rec['available_at']} > origin")
    rec["bytes_downloaded"] = sum(f.get("subset_bytes", 0) for f in files)
    write_json(record_path, rec)
    log(f"fetched AROME run {rid}: {rec['status']} ({rec['bytes_downloaded'] / 1e6:.1f} MB, "
        f"available {rec.get('available_at')})")
    return rec


def fetch_constants(cfg: Config, log=print) -> Path:
    dest = cfg.dir("raw") / "arome" / "constant-eurw1s40.grib2"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(cfg["arome"]["constant_url"], timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        write_json(dest.with_suffix(".provenance.json"), {"url": cfg["arome"]["constant_url"],
                                                          "retrieved_at": now_utc(), "sha256": sha256_file(dest),
                                                          "license": LICENCE})
        log(f"fetched AROME constant fields ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def fetch_arome(cfg: Config, log=print) -> list[dict]:
    fetch_constants(cfg, log)
    return [fetch_run(cfg, origin, log) for origin in arome_origins(cfg)]


def arome_origins(cfg: Config) -> list[pd.Timestamp]:
    o = cfg["origins"]
    return [ts(f"{d}T{o['hour_utc']:02d}:00Z") for d in o["dates"]]


# --------------------------------------------------------------------------- extract

def _messages(path: Path):
    import eccodes

    with open(path, "rb") as f:
        while (h := eccodes.codes_grib_new_from_file(f)) is not None:
            try:
                yield h
            finally:
                eccodes.codes_release(h)


def _grid(h) -> dict:
    import eccodes

    g = {k: eccodes.codes_get(h, k) for k in ("Ni", "Nj", "latitudeOfFirstGridPointInDegrees",
                                               "longitudeOfFirstGridPointInDegrees", "iDirectionIncrementInDegrees",
                                               "jDirectionIncrementInDegrees", "jScansPositively", "gridType")}
    if g["gridType"] != "regular_ll":
        raise ValueError(f"unexpected grid {g['gridType']}")
    return g


def nearest_index(grid: dict, lat: float, lon: float) -> tuple[int, int, float, float]:
    """Nearest grid point on a regular lat/lon grid -> (j, i, grid_lat, grid_lon)."""
    dj = grid["jDirectionIncrementInDegrees"] * (1 if grid["jScansPositively"] else -1)
    lon0 = (grid["longitudeOfFirstGridPointInDegrees"] + 180) % 360 - 180   # GRIB stores 348 for -12
    j = int(round((lat - grid["latitudeOfFirstGridPointInDegrees"]) / dj))
    i = int(round((lon - lon0) / grid["iDirectionIncrementInDegrees"]))
    if not (0 <= j < grid["Nj"] and 0 <= i < grid["Ni"]):
        raise ValueError(f"({lat}, {lon}) outside AROME grid")
    return j, i, round(grid["latitudeOfFirstGridPointInDegrees"] + j * dj, 6), \
        round(lon0 + i * grid["iDirectionIncrementInDegrees"], 6)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return float(2 * 6371.0 * np.arcsin(np.sqrt(a)))


def station_constants(const_path: Path, stations: list[dict]) -> dict[str, dict]:
    """Model terrain height and land-sea mask at each station's nearest grid point."""
    import eccodes

    out: dict[str, dict] = {s["id"]: {} for s in stations}
    for h in _messages(const_path):
        short = eccodes.codes_get(h, "shortName")
        grid = _grid(h)
        vals = eccodes.codes_get_values(h).reshape(grid["Nj"], grid["Ni"])
        for s in stations:
            j, i, _, _ = nearest_index(grid, s["lat"], s["lon"])
            out[s["id"]][short] = float(vals[j, i])
    return out


def extract(cfg: Config, stations: list[dict], log=print) -> tuple[pd.DataFrame, list[dict]]:
    """Station values (°C / hPa) for every fetched run, plus per-run records."""
    import eccodes

    raw = cfg.dir("raw") / "arome"
    const = station_constants(raw / "constant-eurw1s40.grib2", stations)
    rows, runs = [], []
    for origin in arome_origins(cfg):
        init = origin.normalize() + pd.Timedelta(hours=cfg["arome"]["run_hour_utc"])
        rec_path = raw / run_id(init).replace(":", "") / "run.json"
        if not rec_path.exists():
            runs.append({"run": run_id(init), "origin": run_id(origin), "status": "unavailable",
                         "reason": "not fetched (rolled out of the archive before download?)"})
            continue
        rec = read_json(rec_path)
        runs.append({k: rec.get(k) for k in ("run", "origin", "status", "reason", "available_at", "latency_h",
                                             "bytes_downloaded")})
        if rec["status"] != "ok":
            continue
        for f in rec["files"]:
            for h in _messages(rec_path.parent / f["subset"]):
                key = tuple(eccodes.codes_get(h, k, ktype=int) for k in (
                    "discipline", "parameterCategory", "parameterNumber", "typeOfFirstFixedSurface",
                    "scaledValueOfFirstFixedSurface"))
                field = next(k for k, v in FIELDS.items() if v["key"] == key)
                lead = eccodes.codes_get(h, "forecastTime", ktype=int)
                data_date = eccodes.codes_get(h, "dataDate", ktype=int)
                data_time = eccodes.codes_get(h, "dataTime", ktype=int)
                msg_init = pd.Timestamp(f"{data_date}{data_time:04d}", tz="UTC")
                if msg_init != init:
                    raise ValueError(f"message init {msg_init} != expected {init}")
                grid = _grid(h)
                vals = eccodes.codes_get_values(h).reshape(grid["Nj"], grid["Ni"])
                for s in stations:
                    j, i, glat, glon = nearest_index(grid, s["lat"], s["lon"])
                    rows.append({
                        "run": init, "available_at": ts(rec["available_at"]), "lead_h": lead,
                        "valid_time": init + pd.Timedelta(hours=lead), "station": s["id"], "variable": field,
                        "value": float(FIELDS[field]["convert"](vals[j, i])),
                        "grid_lat": glat, "grid_lon": glon,
                        "grid_distance_km": round(haversine_km(s["lat"], s["lon"], glat, glon), 3),
                        "model_height_m": const[s["id"]].get("h", const[s["id"]].get("orog")),
                        "land_sea_mask": const[s["id"]].get("lsm"), "station_altitude_m": s["altitude_m"],
                        "source_file": f["file"],
                    })
    df = pd.DataFrame(rows)
    log(f"extracted {len(df)} AROME station values from {sum(r['status'] == 'ok' for r in runs)} runs")
    return df, runs


def prepare_arome(cfg: Config, stations: list[dict], log=print) -> tuple[pd.DataFrame, dict]:
    df, runs = extract(cfg, stations, log)
    out = cfg.dir("processed") / f"arome_{cfg.suite}.parquet"
    df.to_parquet(out, index=False)
    meta = {"created_at": now_utc(), "file": out.name, "sha256": sha256_file(out), "runs": runs,
            "grid": "EURW1S40 (0.025 deg regular lat/lon)", "package": cfg["arome"]["package"],
            "fields": {k: FIELDS[k]["grib"] for k in cfg["arome"]["fields"]},
            "extraction": "nearest grid point (regular lat/lon index rounding); no height or lapse-rate correction; "
                          "model terrain height and land-sea mask recorded per station",
            "availability_rule": cfg["arome"]["availability_rule"], "license": LICENCE, "sources": DATASETS,
            "config_sha": sha256_text(str(cfg["arome"]))[:12]}
    write_json(out.with_suffix(".meta.json"), meta)
    return df, meta


def load_arome(cfg: Config) -> tuple[pd.DataFrame, dict]:
    path = cfg.dir("processed") / f"arome_{cfg.suite}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `weather-lab data prepare --config {cfg.path}` first")
    return pd.read_parquet(path), read_json(path.with_suffix(".meta.json"))
