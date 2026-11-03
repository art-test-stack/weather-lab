"""Explicit official-source acquisition; no inference or verification interpolation."""
from __future__ import annotations

import base64
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import requests
import yaml

from weather_lab import arome
from weather_lab.arome_fields import decode_station_records, step_values
from weather_lab.common import ROOT, now_utc, sha256_file
from weather_lab.forecast_data import CONFIG as STATIONS_CONFIG
from weather_lab.hourly_benchmark import configuration
from weather_lab.hourly_observations import read_hourly
from weather_lab.prospective_store import digest, encoded
from weather_lab.rag_archive import OBS_BASE

BASE = "https://meteofrance-pnt.s3.rbx.io.cloud.ovh.net/pnt"
PRODUCT = "meteofrance_arome_france"
TEMP = ROOT / ".tmp/codex/prospective-collection"


def stations():
    aliases = {s["id"]: s for s in yaml.safe_load(STATIONS_CONFIG.read_text())["stations"]}
    return [aliases[s["id"]] | s for s in configuration()["stations"]]


def rows_json(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def event(archive, task, status, **details):
    record = {"task": task, "status": status, "collected_at": now_utc(), **details}
    archive.put("status", digest(encoded(record)), record)


def collect_forecasts(archive, start, end, *, session=None, tree=None, decode=decode_station_records,
                      budget=500 * 2**20):
    session = session or requests.Session()
    tree = arome.archive_tree(session) if tree is None else tree
    start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
    consumed, successes, failures = 0, 0, []
    TEMP.mkdir(parents=True, exist_ok=True)
    for rid, models in sorted(tree.items()):
        run = pd.to_datetime(rid, utc=True)
        if not start <= run <= end or "SP1" not in models.get("arome", {}).get("0025", {}):
            continue
        # Discover issues from the catalog; hourly leads are never treated as new issues.
        for low, high in arome.GROUPS_0025:
            name, url = arome.file_for_lead(BASE, run, low)
            key = f"{rid}/{name}"
            if archive.get("forecasts", key) is not None:
                continue
            try:
                messages, http = arome.scan_messages(url, session)
                parts, payload = [], bytearray()
                for lead in range(max(1, low), high + 1):
                    for field in ("t", "pmer"):
                        message = arome._select(messages, field, lead)
                        if message is None:
                            continue
                        stop = message["offset"] + message["length"] - 1
                        response = session.get(url, headers={"Range": f"bytes={message['offset']}-{stop}"}, timeout=120)
                        response.raise_for_status()
                        raw = response.content
                        consumed += len(raw)
                        if consumed > budget:
                            raise ValueError("prospective AROME download exceeds 500 MiB budget")
                        if response.status_code != 206 or len(raw) != message["length"]:
                            raise ValueError("AROME server did not honor exact GRIB byte range")
                        # Recheck the source identity after HEAD/scanning; reject changed objects.
                        if http.get("ETag") and response.headers.get("ETag") != http["ETag"]:
                            raise ValueError("AROME source changed during range download")
                        payload.extend(raw)
                        parts.append({"field": field, "lead_h": lead, "offset": message["offset"],
                                      "length": len(raw), "sha256": digest(raw)})
                if not payload:
                    raise ValueError("no requested hourly GRIB messages published")
                collected = now_utc()
                published = pd.Timestamp(parsedate_to_datetime(http["Last-Modified"]))
                if published < run or published > pd.Timestamp(collected):
                    raise ValueError("AROME publication timestamp outside issue/collection bounds")
                with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
                    path = Path(tmp) / "subset.grib2"
                    path.write_bytes(payload)
                    native = decode(path, stations(), run, collected)
                    values = step_values(native, range(max(1, low), high + 1), 1)
                values = values[values.variable.isin(["t", "pmer"]) & np.isfinite(values.value)].copy()
                values["product"], values["source_kind"] = PRODUCT, "issued_run"
                values["availability_basis"] = "actual_first_collection"
                complete = len(parts) == 2 * (high - max(1, low) + 1)
                record = {"run": rid, "product": PRODUCT, "collected_at": collected,
                          "published_at": published.isoformat(), "source_url": url, "http": http,
                          "messages": parts, "source_sha256": digest(payload),
                          "grib_subset_base64": base64.b64encode(payload).decode(),
                          "rows": rows_json(values), "retrospective": run < pd.Timestamp(collected).floor("D"),
                          "complete": complete}
                # A partial file remains retryable: completed records are never replaced.
                if complete:
                    archive.put("forecasts", key, record)
                else:
                    archive.put("forecasts", key + "/partial/" + record["source_sha256"], record)
                successes += 1
            except (requests.RequestException, ValueError, KeyError) as exc:
                if "budget" in str(exc):
                    raise
                failures.append({"run": rid, "file": name, "reason": str(exc)})
    event(archive, "forecasts", "partial" if failures else "ok", successes=successes,
          failures=failures, start=str(start), end=str(end), downloaded_bytes=consumed)
    return successes


def collect_observations(archive, start, end, *, session=None, budget=500 * 2**20):
    """Version actual hourly station samples; publication delays remain genuine gaps."""
    session = session or requests.Session()
    start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
    if start.year < 2025 or end.year > 2026 or start > end:
        raise ValueError("this source period covers 2025–2026 only")
    TEMP.mkdir(parents=True, exist_ok=True)
    consumed, count, failures = 0, 0, []
    previous = archive.records("observations")
    with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
        for department in sorted({s["department"] for s in stations()}):
            url = f"{OBS_BASE}/H_{department}_latest-2025-2026.csv.gz"
            try:
                path = Path(tmp) / f"{department}.csv.gz"
                with session.get(url, stream=True, timeout=120) as response:
                    response.raise_for_status()
                    with path.open("wb") as stream:
                        for block in response.iter_content(1 << 20):
                            consumed += len(block)
                            if consumed > budget:
                                raise ValueError("prospective observation download exceeds 500 MiB budget")
                            stream.write(block)
                    http = {k: response.headers.get(k) for k in ("ETag", "Last-Modified")}
                collected, source_hash = now_utc(), sha256_file(path)
                for station in (s for s in stations() if s["department"] == department):
                    frame, metadata = read_hourly(path, station["id"], station["national_id"], start, end)
                    frame = frame[frame.NUM_POSTE.notna()]
                    for day, group in frame.groupby(frame.time.dt.strftime("%Y-%m-%d")):
                        rows = rows_json(group)
                        values_hash = digest(encoded(rows))
                        identity = f"{station['id']}/{day}"
                        versions = [r["value"] for r in previous if r["value"]["identity"] == identity]
                        latest = max(versions, key=lambda r: r["collected_at"]) if versions else None
                        if latest and latest["values_sha256"] == values_hash:
                            continue
                        rec = {"identity": identity, "collected_at": collected, "values_sha256": values_hash,
                               "source_url": url, "source_sha256": source_hash, "http": http,
                               "source_kind": "genuine_hourly_station_observations", "rows": rows,
                               "metadata": metadata, "retrospective": True}
                        archive.put("observations", identity + "/" + collected, rec)
                        count += 1
            except requests.RequestException as exc:
                failures.append({"department": department, "reason": str(exc)})
    event(archive, "observations", "partial" if failures else "ok", versions=count, failures=failures,
          start=str(start), end=str(end), downloaded_bytes=consumed)
    return count


def backfill_saved(archive, start, end, source):
    """Import a checksum-verified saved archive, retaining its historical assumptions."""
    from weather_lab.rag_archive import issued_forecasts
    from weather_lab.rag_store import Store

    Store.verify(source)
    path = source / "issued.parquet"
    if not path.exists():
        path = source / "arome_station_extract.parquet"
    frame = issued_forecasts(pd.read_parquet(path))
    lo, hi = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
    frame = frame[frame.run.between(lo, hi) & frame.station.isin([s["id"] for s in stations()])]
    source_hash = sha256_file(path)
    for run, group in frame.groupby("run"):
        record = {"run": run.isoformat(), "product": PRODUCT, "collected_at": now_utc(),
                  "retrospective": True, "source_sha256": source_hash,
                  "source_artifact": source.name, "rows": rows_json(group), "complete": False}
        key = f"historical/{source_hash}/{run.isoformat()}"
        if archive.get("forecasts", key) is None:
            archive.put("forecasts", key, record)
    event(archive, "backfill", "ok", source_artifact=source.name, start=str(lo), end=str(hi),
          limitation="Source availability assumptions retained; this is retrospective evidence only")
