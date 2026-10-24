"""Publish data snapshots to the Hugging Face Storage Bucket and keep the pin in sync.

`publish_snapshot` (run where the data lives): build -> upload archive + .sha256 -> verify the
remote copy -> rewrite `bootstrap.snapshot` in configs/bootstrap.yaml (new primary; previous
primary and existing fallbacks kept only if they are really in the bucket). An existing snapshot
id is never overwritten with different content. `check_pinned` is a read-only guard for CI.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from weather_lab.bootstrap import BOOTSTRAP_CONFIG, BUCKET_PREFIX, build_snapshot, load_bootstrap_config
from weather_lab.common import ROOT

DEFAULT_BUCKET = "ai-testack/weather-lab-storage"


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def remote_sha256(bucket: str, name: str, api=None) -> str | None:
    """SHA-256 recorded in the bucket's `<name>.sha256`, or None if the archive or its checksum is absent."""
    api = api or _api()
    found = {f.path for f in api.get_bucket_paths_info(bucket, [name, f"{name}.sha256"])}
    if name not in found or f"{name}.sha256" not in found:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "x.sha256"
        api.download_bucket_files(bucket, [(f"{name}.sha256", str(dest))], raise_on_missing_files=True)
        return dest.read_text().split()[0]


def _split(url: str) -> tuple[str, str]:
    owner, name, *rest = url[len(BUCKET_PREFIX):].split("/")
    return f"{owner}/{name}", "/".join(rest)


def check_pinned(config: Path = BOOTSTRAP_CONFIG, api=None, log=print) -> bool:
    """CI guard: the pinned primary snapshot exists in its bucket with the pinned checksum."""
    snap = load_bootstrap_config(config)["bootstrap"]["snapshot"]
    if not str(snap.get("url", "")).startswith(BUCKET_PREFIX):
        log(f"pinned snapshot is not a bucket URL ({snap.get('url')}); nothing to check")
        return bool(snap.get("url"))
    bucket, name = _split(snap["url"])
    remote = remote_sha256(bucket, name, api)
    if remote is None:
        log(f"FAIL: {name} (or its .sha256) is not in bucket {bucket}. Run `weather-lab snapshot publish` first.")
        return False
    if remote != snap["sha256"]:
        log(f"FAIL: {name} in {bucket} has sha256 {remote[:12]}…, pinned {snap['sha256'][:12]}…")
        return False
    log(f"ok: {name} is in {bucket} with the pinned sha256 {remote[:12]}…")
    return True


def write_pin(primary: dict, fallbacks: list[dict], mounts: dict, config: Path = BOOTSTRAP_CONFIG) -> None:
    """Rewrite only the `snapshot:` block of the bootstrap config, keeping everything else verbatim."""
    lines = config.read_text().splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if l.startswith("  snapshot:"))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("  ") and not lines[i].startswith("    "))
    block = [
        "  snapshot:\n",
        f"    id: {primary['id']}\n",
        "    # Public Hugging Face Storage Bucket. Buckets are mutable and unversioned, so integrity rests on\n",
        "    # the pinned SHA-256: never overwrite a snapshot file; publish a new id instead\n",
        "    # (`weather-lab snapshot publish` does this and rewrites this block).\n",
        f"    url: {primary['url']}\n",
        f"    sha256: {primary['sha256']}\n",
        "    # Where the bucket is attached as a volume (the Space mounts it at /data). Used when present,\n",
        "    # otherwise the file is downloaded through the Hub API. Both paths are checksum-verified.\n",
        "    mounts:\n",
        *[f"      {k}: {v}\n" for k, v in mounts.items()],
        "    # Earlier immutable snapshots, tried in order only when the pinned one above cannot be fetched or\n",
        "    # fails its checksum (the app then shows which snapshot it is using). Each entry is pinned by its\n",
        "    # SHA-256, so a fallback can only ever install a known snapshot. Mounts above apply to them too.\n",
        "    fallbacks:\n" if fallbacks else "    fallbacks: []\n",
    ]
    for f in fallbacks:
        block += [f"      - id: {f['id']}\n", f"        url: {f['url']}\n", f"        sha256: {f['sha256']}\n"]
    config.write_text("".join(lines[:start] + block + lines[end:]))


def publish_snapshot(snapshot_id: str, bucket: str = DEFAULT_BUCKET, out_dir: Path | None = None,
                     config: Path = BOOTSTRAP_CONFIG, dry_run: bool = False, api=None, log=print) -> dict:
    api = api or (None if dry_run else _api())
    out_dir = out_dir or ROOT / "dist" / "snapshot"
    built = build_snapshot(out_dir, snapshot_id, log)
    name = f"{snapshot_id}.tar.gz"
    archive = Path(built["path"])
    url = f"{BUCKET_PREFIX}{bucket}/{name}"
    primary = {"id": snapshot_id, "url": url, "sha256": built["sha256"]}
    snap = load_bootstrap_config(config)["bootstrap"]["snapshot"]
    if dry_run:
        log(f"dry run: would upload {name} ({built['bytes'] / 1e6:.1f} MB) and {name}.sha256 to {bucket}, "
            f"then pin it in {config.relative_to(ROOT)}")
        return {"primary": primary, "uploaded": False, "dry_run": True}
    existing = remote_sha256(bucket, name, api)
    if existing is not None and existing != built["sha256"]:
        raise RuntimeError(f"{name} already exists in {bucket} with different content ({existing[:12]}…); "
                           "snapshot ids are immutable, choose a new --id")
    if existing is None:
        log(f"uploading {name} ({built['bytes'] / 1e6:.1f} MB) to {bucket}")
        api.batch_bucket_files(bucket, add=[(str(archive), name), (str(archive) + ".sha256", f"{name}.sha256")])
    else:
        log(f"{name} is already in {bucket} with the same content; not re-uploading")
    # verify the remote copy end to end (download and hash), not just the .sha256 sidecar
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / name
        api.download_bucket_files(bucket, [(name, str(dest))], raise_on_missing_files=True)
        got = hashlib.sha256(dest.read_bytes()).hexdigest()
    if got != built["sha256"] or remote_sha256(bucket, name, api) != built["sha256"]:
        raise RuntimeError(f"remote verification failed for {name}: got {got[:12]}…")
    log(f"verified {name} in {bucket} (sha256 {got[:12]}…)")
    # fallbacks: previous primary first, then existing fallbacks; keep only entries present in the bucket
    candidates = ([{k: snap[k] for k in ("id", "url", "sha256")}] if snap.get("url") else []) + list(
        snap.get("fallbacks") or [])
    fallbacks, seen = [], {snapshot_id}
    for c in candidates:
        if c["id"] in seen or not str(c["url"]).startswith(BUCKET_PREFIX):
            continue
        seen.add(c["id"])
        b, n = _split(c["url"])
        if remote_sha256(b, n, api) == c["sha256"]:
            fallbacks.append(c)
        else:
            log(f"dropping fallback {c['id']}: not in {b} with its pinned checksum")
    write_pin(primary, fallbacks, snap.get("mounts") or {}, config)
    log(f"pinned {snapshot_id} in {config.relative_to(ROOT)} (fallbacks: {[f['id'] for f in fallbacks] or 'none'}); "
        "commit it and push to deploy")
    return {"primary": primary, "fallbacks": fallbacks, "uploaded": existing is None}
