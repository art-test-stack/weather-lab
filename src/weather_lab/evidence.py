"""Availability evidence: re-query archived resources, and record first-seen times for new runs.

`Last-Modified` on the archive object store is the time an object was (last) written there.
These utilities test what it can and cannot support, and provide a conservative,
evidence-backed alternative for runs observed live (`first_seen_available_at`). Nothing
here runs on a schedule; `observe` is invoked manually.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

from weather_lab import arome as A
from weather_lab.common import ROOT, Config, now_utc, read_json, write_json

EVIDENCE = ROOT / "data" / "evidence"
SYNOP_BASE = "https://meteofrance.s3.sbg.io.cloud.ovh.net/data/OBS/SYNOP/"


def _head(s: requests.Session, url: str) -> dict:
    r = s.head(url, timeout=60, allow_redirects=True)
    return {"status": r.status_code, "retrieved_at": now_utc(),
            **{k: r.headers.get(k) for k in ("ETag", "Last-Modified", "Content-Length")}}


def requery_arome(cfg: Config, s: requests.Session) -> list[dict]:
    """Re-query every fetched AROME file: headers now vs at fetch time, and message checksums."""
    out = []
    for rec_path in sorted((cfg.dir("raw") / "arome").glob("*/run.json")):
        rec = read_json(rec_path)
        for f in rec["files"]:
            now = _head(s, f["url"])
            msgs = []
            if now["status"] == 200:
                for m in f["messages"][:2]:     # two messages per file: content re-check by byte range
                    r = s.get(f["url"], headers={"Range": f"bytes={m['offset']}-{m['offset'] + m['length'] - 1}"},
                              timeout=120)
                    msgs.append({"field": m["field"], "lead_h": m["lead_h"], "offset": m["offset"],
                                 "sha256_then": m["sha256"], "sha256_now": hashlib.sha256(r.content).hexdigest()})
            out.append({"run": rec["run"], "file": f["file"], "url": f["url"],
                        "then": {"retrieved_at": rec["retrieved_at"], **f["http"]}, "now": now, "messages": msgs,
                        "valid_times": [str(pd.Timestamp(rec["run"]) + pd.Timedelta(hours=m["lead_h"]))
                                        for m in f["messages"]][:1]})
    return out


def full_checksums(urls: list[str], s: requests.Session) -> list[dict]:
    """Full-content SHA-256 and MD5 for sampled resources (MD5 compared with the S3 ETag)."""
    out = []
    for url in urls:
        h, md5, n = hashlib.sha256(), hashlib.md5(), 0
        with s.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            headers = {k: r.headers.get(k) for k in ("ETag", "Last-Modified", "Content-Length")}
            for block in r.iter_content(1 << 20):
                h.update(block)
                md5.update(block)
                n += len(block)
        etag = (headers["ETag"] or "").strip('"')
        out.append({"url": url, "retrieved_at": now_utc(), "bytes": n, "sha256": h.hexdigest(), "md5": md5.hexdigest(),
                    **headers, "etag_is_content_md5": etag == md5.hexdigest()})
    return out


def catalog_vs_object(s: requests.Session, n: int = 3) -> list[dict]:
    """data.gouv.fr catalogue timestamps vs object-store headers for the same AROME resources."""
    meta = s.get("https://www.data.gouv.fr/api/1/datasets/paquets-arome-resolution-0-025deg/", timeout=60).json()
    out = []
    for r in [r for r in meta["resources"] if r["title"].endswith(".grib2")][:n]:
        out.append({"title": r["title"], "catalog_created_at": r.get("created_at"),
                    "catalog_last_modified": r.get("last_modified"), "url": r["url"], "object": _head(s, r["url"])})
    return out


def synop_headers(years: list[int], s: requests.Session) -> list[dict]:
    return [{"file": f"synop_{y}.csv.gz", "url": f"{SYNOP_BASE}synop_{y}.csv.gz", **_head(s, f"{SYNOP_BASE}synop_{y}.csv.gz")}
            for y in years]


def collect(cfg: Config, sample_urls: list[str] | None = None, log=print) -> Path:
    s = requests.Session()
    rec = {"collected_at": now_utc(), "arome_requery": requery_arome(cfg, s),
           "catalog_vs_object": catalog_vs_object(s),
           "synop_headers": synop_headers([1996, 2010, 2022, 2023, 2024, 2025], s)}
    if sample_urls:
        rec["full_checksums"] = full_checksums(sample_urls, s)
    path = EVIDENCE / f"availability_{rec['collected_at'].replace(':', '')}.json"
    write_json(path, rec)
    log(f"saved {path}")
    return path


# --------------------------------------------------------------------------- first seen (manual)

def observe(cfg: Config, last_runs: int = 1, log=print) -> list[dict]:
    """Record first_seen_available_at for the newest runs whose required messages are retrievable now.

    For each run: scan the needed files, download the required messages (byte ranges) and
    record the time the last one was retrieved. An existing record keeps its earlier time.
    This bounds availability from above for runs observed live; it reconstructs nothing.
    """
    a = cfg["arome"]
    tree = A.archive_tree()
    runs = sorted(r for r in tree if pd.Timestamp(r).hour == a["run_hour_utc"]
                  and len(tree[r].get("arome", {}).get(a["grid"], {}).get(a["package"], [])) == len(A.GROUPS_0025))
    log_path = EVIDENCE / "first_seen.jsonl"
    seen = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            if r.get("complete"):
                seen.setdefault(r["run"], r)
    s = requests.Session()
    out = []
    for rid in runs[-last_runs:]:
        if rid in seen:
            log(f"{rid}: already recorded first seen at {seen[rid]['first_seen_available_at']}")
            out.append(seen[rid])
            continue
        init = pd.Timestamp(rid)
        leads = A.required_leads(init.normalize() + pd.Timedelta(hours=cfg["origins"]["hour_utc"]), init,
                                 cfg["forecast"]["horizon_steps"], cfg["forecast"]["step_hours"])
        files, complete = [], True
        by_file: dict[str, tuple[str, list[int]]] = {}
        for lead in leads:
            name, url = A.file_for_lead(a["base_url"], init, lead, a["package"])
            by_file.setdefault(name, (url, []))[1].append(lead)
        for name, (url, fl) in sorted(by_file.items()):
            msgs, http = A.scan_messages(url, s)
            got = []
            for lead in fl:
                for field in a["fields"]:
                    m = A._select(msgs, field, lead)
                    if m is None:
                        complete = False
                        continue
                    r = s.get(url, headers={"Range": f"bytes={m['offset']}-{m['offset'] + m['length'] - 1}"},
                              timeout=120)
                    complete &= r.status_code == 206 and len(r.content) == m["length"]
                    got.append({"field": field, "lead_h": lead, "sha256": hashlib.sha256(r.content).hexdigest()})
            files.append({"file": name, "url": url, "http": http, "messages": got})
        rec = {"run": rid, "complete": complete, "first_seen_available_at": now_utc() if complete else None,
               "observed_by": "weather-lab arome observe (manual)", "files": files}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"{rid}: complete={complete} first_seen_available_at={rec['first_seen_available_at']}")
        out.append(rec)
    return out
