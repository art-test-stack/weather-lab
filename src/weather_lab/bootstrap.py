"""Idempotent setup shared by local use and deployment: fetch, verify, prepare, reuse cache.

Two modes:

- `official`: download from the original providers (data.gouv.fr SYNOP files, the
  Météo-France 14-day AROME archive, the Hugging Face Hub), regenerate each suite's
  manifest (validated against the committed spec), and reproduce the saved results
  with real inference when they are absent (validated against the spec's score hashes).
  AROME runs that have rolled out of the 14-day archive make this mode fail fast.
- `snapshot`: download one immutable archive (URL + SHA-256) holding manifests,
  prepared observations, the AROME station extract, saved runs and evidence files;
  verify every member against its index; install; plus pinned weights from the Hub.

Every step checks what is already present (by checksum) and skips it, so restarting
is cheap. Failures raise `BootstrapError` with the failing step and whether a retry
can help; the state is recorded in `data/bootstrap_state.json`.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import shutil
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import requests
import yaml

from weather_lab.common import ROOT, Config, load_config, now_utc, read_json, sha256_file, write_json

DEPLOY_CONFIG = ROOT / "configs" / "deploy.yaml"
STATE_PATH = ROOT / "data" / "bootstrap_state.json"
CACHE = ROOT / "data" / "cache"
Progress = Callable[[float, str], None]


class BootstrapError(RuntimeError):
    def __init__(self, step: str, message: str, retryable: bool = True, hint: str = "") -> None:
        super().__init__(f"[{step}] {message}" + (f" — {hint}" if hint else ""))
        self.step, self.message, self.retryable, self.hint = step, message, retryable, hint


@dataclass
class Plan:
    mode: str
    suites: list[Config]
    snapshot: dict
    weights: list[dict]


def _print_progress(frac: float, msg: str) -> None:
    print(f"[bootstrap {frac:5.0%}] {msg}", flush=True)


def load_deploy(path: Path = DEPLOY_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_plan(mode: str = "auto", snapshot_url: str | None = None, snapshot_sha256: str | None = None,
              suites: list[str] | None = None, deploy: dict | None = None) -> Plan:
    d = (deploy or load_deploy())["bootstrap"]
    snap = dict(d.get("snapshot") or {})
    snap["url"] = snapshot_url or os.environ.get("WEATHER_LAB_SNAPSHOT_URL") or snap.get("url")
    snap["sha256"] = snapshot_sha256 or os.environ.get("WEATHER_LAB_SNAPSHOT_SHA256") or snap.get("sha256")
    mode = os.environ.get("WEATHER_LAB_BOOTSTRAP_MODE", mode) if mode == "auto" else mode
    if mode == "auto":
        mode = "snapshot" if snap.get("url") else "official"
    if mode not in ("official", "snapshot"):
        raise ValueError(f"unknown bootstrap mode {mode!r}")
    cfgs = [load_config(ROOT / p) for p in (suites or d["suites"])]
    return Plan(mode, cfgs, snap, d.get("weights", []))


# --------------------------------------------------------------------------- state

def _state() -> dict:
    return read_json(STATE_PATH) if STATE_PATH.exists() else {"attempts": 0, "steps": {}}


def _save(state: dict) -> None:
    write_json(STATE_PATH, state)


# --------------------------------------------------------------------------- downloads

def download(url: str, dest: Path, sha256: str | None, progress: Progress, label: str, lo: float, hi: float) -> Path:
    """Download (http(s) or file://) with progress; verify SHA-256; reuse a verified cached copy."""
    if dest.exists() and (sha256 is None or sha256_file(dest) == sha256):
        progress(hi, f"{label}: cached ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    try:
        if url.startswith("file://"):
            shutil.copyfile(url[7:], tmp)
            h.update(tmp.read_bytes())
        else:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total, done, last = int(r.headers.get("Content-Length", 0)), 0, 0.0
                with open(tmp, "wb") as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
                        h.update(block)
                        done += len(block)
                        if total and time.time() - last > 1:
                            last = time.time()
                            progress(lo + (hi - lo) * done / total, f"{label}: {done / 1e6:.0f}/{total / 1e6:.0f} MB")
    except (requests.RequestException, OSError) as e:
        tmp.unlink(missing_ok=True)
        raise BootstrapError(label, f"download failed: {e}", retryable=True) from e
    if sha256 is not None and h.hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        raise BootstrapError(label, f"checksum mismatch (got {h.hexdigest()[:12]}, expected {sha256[:12]})",
                             retryable=True, hint="the remote file changed or the transfer was corrupted")
    tmp.rename(dest)
    progress(hi, f"{label}: downloaded {dest.stat().st_size / 1e6:.1f} MB, sha256 verified")
    return dest


def ensure_weights(weights: list[dict], progress: Progress, lo: float, hi: float) -> list[dict]:
    from huggingface_hub import hf_hub_download

    out = []
    for k, w in enumerate(weights):
        a = lo + (hi - lo) * k / max(len(weights), 1)
        progress(a, f"weights {w['repo']}@{w['revision'][:8]}")
        try:
            hf_hub_download(w["repo"], "config.json", revision=w["revision"])
            path = Path(hf_hub_download(w["repo"], "model.safetensors", revision=w["revision"]))
        except Exception as e:  # noqa: BLE001 - network / hub errors are retryable
            raise BootstrapError("weights", f"{w['repo']}: {e}", retryable=True) from e
        digest = sha256_file(path.resolve())
        if digest != w["sha256"]:
            raise BootstrapError("weights", f"{w['repo']} sha256 {digest[:12]} != pinned {w['sha256'][:12]}",
                                 retryable=False)
        out.append({"repo": w["repo"], "revision": w["revision"], "path": str(path), "sha256_ok": True})
    progress(hi, f"weights verified ({len(weights)} checkpoint(s))")
    return out


# --------------------------------------------------------------------------- snapshot

SNAPSHOT_GLOBS = ["benchmarks/manifests/*.json", "data/examples/*/*", "data/processed/*.parquet",
                  "data/processed/*.meta.json", "data/evidence/*", "experiments/**/*"]


def build_snapshot(out_dir: Path, snapshot_id: str, log=print) -> dict:
    """Deterministic tar.gz of the local artifacts the app and suites need (no weights, no raw files)."""
    files = sorted({p for g in SNAPSHOT_GLOBS for p in ROOT.glob(g) if p.is_file() and p.name != "README.md"
                    and "/rescored/" not in str(p)})
    index = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files]
    meta = {"snapshot_id": snapshot_id, "files": index}  # no timestamp: byte-reproducible archive
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{snapshot_id}.tar.gz"
    with open(target, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w") as tar:
        def add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
        add("index.json", (yaml.safe_dump(meta, sort_keys=False)).encode())
        for p in files:
            add(str(p.relative_to(ROOT)), p.read_bytes())
    digest = sha256_file(target)
    (out_dir / f"{snapshot_id}.tar.gz.sha256").write_text(f"{digest}  {target.name}\n")
    log(f"snapshot {target} ({target.stat().st_size / 1e6:.1f} MB, {len(files)} files) sha256 {digest}")
    return {"id": snapshot_id, "path": str(target), "sha256": digest, "files": len(files),
            "bytes": target.stat().st_size}


def install_snapshot(snap: dict, progress: Progress, lo: float, hi: float) -> dict:
    if not snap.get("url") or not snap.get("sha256"):
        raise BootstrapError("snapshot", "no immutable snapshot location configured", retryable=False,
                             hint="set bootstrap.snapshot.url and .sha256 in configs/deploy.yaml (or "
                                  "WEATHER_LAB_SNAPSHOT_URL / WEATHER_LAB_SNAPSHOT_SHA256), or use --mode official")
    archive = download(snap["url"], CACHE / "snapshots" / f"{snap['sha256'][:16]}.tar.gz", snap["sha256"],
                       progress, "snapshot", lo, lo + 0.8 * (hi - lo))
    installed = reused = 0
    with tarfile.open(archive, "r:gz") as tar:
        index = yaml.safe_load(tar.extractfile("index.json").read())
        for entry in index["files"]:
            dest = ROOT / entry["path"]
            if not dest.resolve().is_relative_to(ROOT) or ".." in Path(entry["path"]).parts:
                raise BootstrapError("snapshot", f"unsafe member path {entry['path']}", retryable=False)
            if dest.exists() and dest.stat().st_size == entry["bytes"] and sha256_file(dest) == entry["sha256"]:
                reused += 1
                continue
            data = tar.extractfile(entry["path"]).read()
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise BootstrapError("snapshot", f"member {entry['path']} fails its index checksum", retryable=False)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            installed += 1
    progress(hi, f"snapshot {index['snapshot_id']}: {installed} files installed, {reused} reused")
    return {"snapshot_id": index["snapshot_id"], "installed": installed, "reused": reused}


# --------------------------------------------------------------------------- official

def prepare_suite(cfg: Config, log=print) -> dict:
    """data prepare: clean observations, AROME extract, cases and the frozen (spec-checked) manifest."""
    from weather_lab.data import load_observations, prepare_synop
    from weather_lab.suite import build_cases, write_manifest

    prepare_synop(cfg, log)
    obs, meta = load_observations(cfg)  # checksum what later stages will read back from Parquet
    arome_meta = None
    if cfg.get("arome"):
        from weather_lab.arome import load_arome, prepare_arome
        from weather_lab.example import restore_arome_extract

        if restore_arome_extract(cfg, log):
            _, arome_meta = load_arome(cfg)
        else:
            _, arome_meta = prepare_arome(cfg, meta["stations"], log)
        cases = build_cases(cfg, obs, arome_meta["runs"])
    else:
        cases = build_cases(cfg, obs)
    return write_manifest(cfg, obs, meta, cases, arome_meta, log)


def arome_runs_missing(cfg: Config) -> list[str]:
    """Runs the suite needs that are neither cached locally nor still in the 14-day archive."""
    from weather_lab.arome import arome_origins, run_id

    need = []
    for origin in arome_origins(cfg):
        init = origin.normalize() + pd.Timedelta(hours=cfg["arome"]["run_hour_utc"])
        if not (cfg.dir("raw") / "arome" / run_id(init).replace(":", "") / "run.json").exists():
            need.append(run_id(init))
    if not need:
        return []
    meta = requests.get("https://www.data.gouv.fr/api/1/datasets/donnees-pnt-retention-14-jours/", timeout=60).json()
    tree_url = next(r["url"] for r in meta["resources"] if r["url"].endswith(".json"))
    tree = requests.get(tree_url, timeout=120).json()["pnt"]
    return [r for r in need if not tree.get(r, {}).get("arome", {}).get("0025")]


def official_suite(cfg: Config, progress: Progress, lo: float, hi: float, log=print) -> dict:
    from weather_lab.data import fetch_synop
    from weather_lab.specs import check_scores, load_spec

    span = hi - lo
    progress(lo, f"{cfg.suite}: SYNOP files")
    try:
        fetch_synop(cfg, log)
    except (requests.RequestException, OSError, KeyError) as e:
        raise BootstrapError("synop", str(e), retryable=True) from e
    spec = load_spec(cfg) or {}
    for name, digest in spec.get("source_files", {}).items():
        if sha256_file(cfg.dir("raw") / name) != digest:
            raise BootstrapError("synop", f"{name} differs from the frozen suite's source", retryable=False,
                                 hint="the provider re-published the file; freeze a new suite version")
    if cfg.get("arome"):
        progress(lo + 0.25 * span, f"{cfg.suite}: AROME runs")
        packaged = (ROOT / "data" / "examples" / cfg.suite / "arome_station_extract.parquet").exists()
        if not packaged:
            missing = arome_runs_missing(cfg)
            if missing:
                raise BootstrapError("arome", f"{len(missing)} required runs are no longer in the 14-day archive "
                                              f"(first: {missing[0]})", retryable=False,
                                     hint="use --mode snapshot with the immutable snapshot of this suite")
            from weather_lab.arome import fetch_arome

            fetch_arome(cfg, log)
    from weather_lab.data import processed_path

    if cfg.manifest_path.exists() and processed_path(cfg).exists():
        progress(lo + 0.45 * span, f"{cfg.suite}: manifest and prepared data present (verified at evaluation)")
    else:
        progress(lo + 0.45 * span, f"{cfg.suite}: prepare + manifest")
        prepare_suite(cfg, log)
    example = ROOT / "data" / "examples" / cfg.suite
    result = {"suite": cfg.suite, "results": "reused"}
    if not (example / "run.json").exists() or \
            read_json(example / "manifest.json")["protocol_sha256"] != read_json(cfg.manifest_path)["protocol_sha256"]:
        progress(lo + 0.6 * span, f"{cfg.suite}: reproducing saved results (real inference)")
        from weather_lab.evaluate import run_evaluation
        from weather_lab.example import export_example

        run_dir = run_evaluation(cfg, log)
        ok = check_scores(cfg, run_dir)
        if ok and not all(ok.values()):
            raise BootstrapError("reproduce", f"scores differ from the spec: {ok}", retryable=False)
        export_example([run_dir], log)
        result.update(results="recomputed", scores_match_spec=ok)
    progress(hi, f"{cfg.suite}: ready")
    return result


# --------------------------------------------------------------------------- entry point

def ready(plan: Plan) -> bool:
    """Cheap check: every suite has saved results and a manifest."""
    return all((ROOT / "data" / "examples" / c.suite / "run.json").exists() and c.manifest_path.exists()
               for c in plan.suites)


def bootstrap(mode: str = "auto", snapshot_url: str | None = None, snapshot_sha256: str | None = None,
              suites: list[str] | None = None, weights: bool = True, progress: Progress | None = None,
              log=print) -> dict:
    progress = progress or _print_progress
    plan = make_plan(mode, snapshot_url, snapshot_sha256, suites)
    state = _state()
    state.update(attempts=state.get("attempts", 0) + 1, mode=plan.mode, started_at=now_utc(), status="running")
    state.pop("error", None)
    _save(state)
    t0 = time.perf_counter()
    try:
        progress(0.0, f"mode {plan.mode}: {', '.join(c.suite for c in plan.suites)}")
        if plan.mode == "snapshot":
            state["steps"]["snapshot"] = install_snapshot(plan.snapshot, progress, 0.0, 0.6)
        else:
            n = len(plan.suites)
            state["steps"]["suites"] = [official_suite(c, progress, 0.6 * k / n, 0.6 * (k + 1) / n, log)
                                        for k, c in enumerate(plan.suites)]
        if weights:
            state["steps"]["weights"] = ensure_weights(plan.weights, progress, 0.6, 0.95)
        if not ready(plan):
            missing = [c.suite for c in plan.suites if not (ROOT / "data" / "examples" / c.suite / "run.json").exists()]
            raise BootstrapError("verify", f"saved results missing for {missing}", retryable=False,
                                 hint="the snapshot does not contain these suites")
    except BootstrapError as e:
        state.update(status="failed", error={"step": e.step, "message": e.message, "retryable": e.retryable,
                                             "hint": e.hint}, finished_at=now_utc())
        _save(state)
        raise
    state.update(status="ready", finished_at=now_utc(), seconds=round(time.perf_counter() - t0, 2))
    _save(state)
    progress(1.0, f"ready in {state['seconds']} s")
    return state


# --------------------------------------------------------------------------- deployment output

SPACE_INCLUDE = ("app.py", "requirements.txt", "pyproject.toml", "src/", "configs/", "benchmarks/specs/", "docs/",
                 "AGENTS.md")


def git_intended_files() -> list[str]:
    import subprocess

    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split()
    return sorted(f for f in out if (ROOT / f).is_file())


def build_space(out: Path, log=print) -> Path:
    """Space repository content: GitHub-intended code + README with generated Spaces front matter."""
    deploy = load_deploy()
    if out.exists():
        shutil.rmtree(out)
    files = [f for f in git_intended_files() if f.startswith(SPACE_INCLUDE)]
    for f in files:
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, out / f)
    front = yaml.safe_dump(deploy["space"], sort_keys=False, allow_unicode=True, width=1000).strip()
    body = (ROOT / "README.md").read_text()
    (out / "README.md").write_text(f"---\n{front}\n---\n\n{body}")
    log(f"Space content in {out}: {len(files) + 1} files, no data or weights (fetched by bootstrap at startup)")
    return out


def summarize_failure(state: dict) -> str:
    e = state.get("error") or {}
    return (f"Setup failed at step **{e.get('step')}**: {e.get('message')}. "
            + (f"{e.get('hint')}. " if e.get("hint") else "")
            + ("A retry may succeed." if e.get("retryable") else "Retrying will not help until the cause is fixed."))
