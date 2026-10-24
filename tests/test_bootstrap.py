"""Bootstrap guards, spec validation and refreshed-suite ids (synthetic inputs, no network)."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from weather_lab.arome import GROUPS_0025, refresh_suite
from weather_lab.bootstrap import BootstrapError, download
from weather_lab.common import ROOT, load_config
from weather_lab.specs import check_manifest, load_spec


def _tree(days):
    files = [f"g{k}" for k in range(len(GROUPS_0025))]
    return {f"{d}T03:00:00Z": {"arome": {"0025": {"SP1": files}}} for d in days}


def test_refreshed_dates_get_a_new_suite_id(tmp_path):
    base = ROOT / "configs" / "arome_smoke.yaml"
    same = load_config(base)["origins"]["dates"]
    assert refresh_suite(base, tmp_path, now=pd.Timestamp("2026-10-24T19:00Z"), tree=_tree(same), log=lambda *_: 0) \
        is None
    later = [f"2026-10-{d}" for d in range(15, 27)]
    path = refresh_suite(base, tmp_path, now=pd.Timestamp("2026-10-30T12:00Z"), tree=_tree(later), log=lambda *_: 0)
    cfg = yaml.safe_load(open(path))
    assert cfg["suite"] == "arome-issued-r20261015-20261026" and cfg["suite"] != "arome-issued-v1"
    assert cfg["manifest"].endswith("arome-issued-r20261015-20261026.json")
    # 2026-10-27/28 targets are not yet verifiable at the given time and incomplete runs never qualify
    assert cfg["origins"]["dates"][-1] == "2026-10-26"


def test_download_rejects_checksum_mismatch(tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"snapshot")
    with pytest.raises(BootstrapError) as e:
        download(f"file://{src}", tmp_path / "b.bin", "0" * 64, lambda *_: None, "snapshot", 0, 1)
    assert e.value.retryable and not (tmp_path / "b.bin").exists()


def test_download_reuses_verified_cache(tmp_path):
    import hashlib

    src = tmp_path / "a.bin"
    src.write_bytes(b"snapshot")
    digest = hashlib.sha256(b"snapshot").hexdigest()
    dest = download(f"file://{src}", tmp_path / "b.bin", digest, lambda *_: None, "snapshot", 0, 1)
    src.unlink()  # a second call must not need the source again
    assert download(f"file://{src}", dest, digest, lambda *_: None, "snapshot", 0, 1) == dest


def test_spec_rejects_different_case_data():
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    spec = load_spec(cfg)
    manifest = {k: spec[k] for k in ("protocol_sha256", "case_data_sha256", "n_cases", "n_eligible")}
    manifest["dataset"] = {"sources": []}
    check_manifest(cfg, manifest)  # reproduces the spec: accepted
    manifest["case_data_sha256"] = "x"
    with pytest.raises(RuntimeError, match="does not reproduce"):
        check_manifest(cfg, manifest)


def test_bucket_url_prefers_mounted_volume_and_verifies_it(tmp_path, monkeypatch):
    import hashlib

    from weather_lab import bootstrap as B

    mount = tmp_path / "data"
    mount.mkdir()
    (mount / "snap.tar.gz").write_bytes(b"payload")
    url = "hf://buckets/owner/store/snap.tar.gz"
    assert B.resolve_bucket(url, {"owner/store": str(mount)}) == ("owner/store", "snap.tar.gz", mount / "snap.tar.gz")
    assert B.resolve_bucket(url, {"owner/store": str(tmp_path / "absent")})[2] is None

    def no_network(*a, **k):
        raise AssertionError("mounted file must be used without downloading")

    monkeypatch.setattr("huggingface_hub.download_bucket_files", no_network)
    digest = hashlib.sha256(b"payload").hexdigest()
    got = B.download(url, tmp_path / "out.tar.gz", digest, lambda *_: None, "snapshot", 0, 1,
                     mounts={"owner/store": str(mount)})
    assert got.read_bytes() == b"payload"
    with pytest.raises(BootstrapError):          # a mutable bucket file that changed is rejected
        B.download(url, tmp_path / "other.tar.gz", "0" * 64, lambda *_: None, "snapshot", 0, 1,
                   mounts={"owner/store": str(mount)})


def test_bucket_url_without_mount_uses_hub_api(tmp_path, monkeypatch):
    import hashlib

    from weather_lab import bootstrap as B

    calls = []

    def fake(bucket_id, files, raise_on_missing_files=False, token=None):
        calls.append((bucket_id, files[0][0]))
        Path(files[0][1]).write_bytes(b"remote")

    monkeypatch.setattr("huggingface_hub.download_bucket_files", fake)
    out = B.download("hf://buckets/owner/store/dir/snap.tar.gz", tmp_path / "o.tgz",
                     hashlib.sha256(b"remote").hexdigest(), lambda *_: None, "snapshot", 0, 1)
    assert calls == [("owner/store", "dir/snap.tar.gz")] and out.read_bytes() == b"remote"


# --------------------------------------------------------------------------- snapshot fallbacks

def _tarball(path: Path, snapshot_id: str, files: dict[str, bytes]) -> str:
    """Tiny archive in `build_snapshot`'s format (yaml index.json + members); returns its sha256."""
    import gzip
    import hashlib
    import io
    import tarfile

    index = {"snapshot_id": snapshot_id,
             "files": [{"path": p, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for p, b in files.items()]}
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w") as tar:
        for name, data in [("index.json", yaml.safe_dump(index).encode()), *files.items()]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point bootstrap at a tmp root/cache/state and return (B, root, src dir, run(snap) -> state)."""
    from weather_lab import bootstrap as B

    root = (tmp_path / "root").resolve()
    src = tmp_path / "src"
    root.mkdir()
    src.mkdir()
    monkeypatch.setattr(B, "ROOT", root)
    monkeypatch.setattr(B, "CACHE", root / "data" / "cache")
    monkeypatch.setattr(B, "STATE_PATH", root / "data" / "bootstrap_state.json")

    def run(snap):
        monkeypatch.setattr(B, "make_plan", lambda *a, **k: B.Plan("snapshot", [], snap, []))
        return B.bootstrap(weights=False, progress=lambda *_: None, log=lambda *_: None)

    return B, root, src, run


def _snap(src: Path, name: str, sha: str, fallbacks=()) -> dict:
    return {"id": name, "url": f"file://{src / (name + '.tar.gz')}", "sha256": sha, "fallbacks": list(fallbacks)}


def test_missing_primary_falls_back_to_first_fallback(sandbox):
    B, root, src, run = sandbox
    v3 = _tarball(src / "v3.tar.gz", "v3", {"data/x.txt": b"three"})
    v2 = _tarball(src / "v2.tar.gz", "v2", {"data/x.txt": b"two"})
    snap = _snap(src, "v4", "1" * 64, [_snap(src, "v3", v3), _snap(src, "v2", v2)])
    state = run(snap)
    assert state["status"] == "ready" and (root / "data" / "x.txt").read_bytes() == b"three"
    assert state["snapshot"]["id"] == "v3" and state["snapshot"]["fallback"] is True
    assert state["snapshot"]["primary_id"] == "v4" and "download failed" in state["snapshot"]["primary_error"]
    saved = B.read_json(B.STATE_PATH)
    assert saved["snapshot"] == state["snapshot"] and saved["steps"]["snapshot"]["snapshot_id"] == "v3"
    # idempotent: the cached verified fallback archive is reused even once its source is gone
    (src / "v3.tar.gz").unlink()
    again = run(snap)
    assert again["snapshot"]["id"] == "v3" and again["steps"]["snapshot"]["reused"] == 1


def test_primary_with_wrong_checksum_uses_fallback(sandbox):
    B, root, src, run = sandbox
    _tarball(src / "v4.tar.gz", "v4", {"data/x.txt": b"tampered"})
    v3 = _tarball(src / "v3.tar.gz", "v3", {"data/x.txt": b"three"})
    state = run(_snap(src, "v4", "2" * 64, [_snap(src, "v3", v3)]))
    assert state["snapshot"]["id"] == "v3" and state["snapshot"]["fallback"]
    assert "checksum mismatch" in state["snapshot"]["primary_error"]
    assert (root / "data" / "x.txt").read_bytes() == b"three"


def test_all_candidates_failing_raises(sandbox):
    B, root, src, run = sandbox
    _tarball(src / "v3.tar.gz", "v3", {"data/x.txt": b"three"})
    with pytest.raises(BootstrapError) as e:
        run(_snap(src, "v4", "1" * 64, [_snap(src, "v3", "3" * 64), _snap(src, "v2", "4" * 64)]))
    assert "v4" in e.value.message and "v3" in e.value.message and "v2" in e.value.message
    saved = B.read_json(B.STATE_PATH)
    assert saved["status"] == "failed" and "snapshot" not in saved and not (root / "data" / "x.txt").exists()


def test_present_primary_uses_no_fallback(sandbox):
    B, root, src, run = sandbox
    v4 = _tarball(src / "v4.tar.gz", "v4", {"data/x.txt": b"four"})
    v3 = _tarball(src / "v3.tar.gz", "v3", {"data/x.txt": b"three"})
    state = run(_snap(src, "v4", v4, [_snap(src, "v3", v3)]))
    assert state["snapshot"] == {"id": "v4", "fallback": False, "primary_id": "v4", "primary_error": None}
    assert (root / "data" / "x.txt").read_bytes() == b"four"
    assert not (B.CACHE / "snapshots" / f"{v3[:16]}.tar.gz").exists()


def test_plan_env_override_keeps_fallbacks(monkeypatch):
    from weather_lab.bootstrap import load_bootstrap_config, make_plan

    deploy = load_bootstrap_config()
    monkeypatch.setenv("WEATHER_LAB_SNAPSHOT_URL", "file:///tmp/other.tar.gz")
    monkeypatch.setenv("WEATHER_LAB_SNAPSHOT_SHA256", "5" * 64)
    plan = make_plan("snapshot", deploy=deploy)
    assert plan.snapshot["url"] == "file:///tmp/other.tar.gz" and plan.snapshot["sha256"] == "5" * 64
    assert plan.snapshot["id"] == "other"
    assert plan.snapshot["fallbacks"] == deploy["bootstrap"]["snapshot"]["fallbacks"]


def test_app_notice_only_for_fallback():
    from weather_lab_app.main import fallback_notice

    assert fallback_notice(None) is None
    assert fallback_notice({"snapshot": {"id": "v4", "fallback": False}}) is None
    msg = fallback_notice({"snapshot": {"id": "v3", "fallback": True, "primary_id": "v4",
                                        "primary_error": "download failed: 1 file(s) not found in bucket 'a/b': v4"}})
    assert msg.startswith("Showing data snapshot `v3` (fallback): pinned snapshot `v4` could not be fetched")
    assert "not found" in msg and msg.endswith("Results shown are from the earlier snapshot.")


class FakeBucketApi:
    """In-memory stand-in for the HfApi bucket calls used by weather_lab.publish."""

    def __init__(self, files=None):
        self.files = dict(files or {})        # name -> bytes
        self.uploads = []

    def get_bucket_paths_info(self, bucket, paths):
        from types import SimpleNamespace

        return [SimpleNamespace(path=p) for p in paths if p in self.files]

    def download_bucket_files(self, bucket, files, raise_on_missing_files=False, token=None):
        for remote, local in files:
            if remote not in self.files:
                raise FileNotFoundError(remote)
            Path(local).write_bytes(self.files[remote])

    def batch_bucket_files(self, bucket, add=None, copy=None, delete=None, token=None):
        for src, name in add or []:
            self.files[name] = Path(src).read_bytes()
            self.uploads.append(name)


def _publish_env(tmp_path, monkeypatch):
    import hashlib

    from weather_lab import bootstrap as B
    from weather_lab import publish as P

    cfg = tmp_path / "bootstrap.yaml"
    old = b"old snapshot"
    old_sha = hashlib.sha256(old).hexdigest()
    cfg.write_text(f"""bootstrap:
  suites: [configs/smoke.yaml]
  mode: auto
  snapshot:
    id: old
    url: hf://buckets/o/b/old.tar.gz
    sha256: {old_sha}
    mounts:
      o/b: /data
    fallbacks:
      - id: ghost
        url: hf://buckets/o/b/ghost.tar.gz
        sha256: {'0' * 64}
  weights: []
""")
    monkeypatch.setattr(P, "ROOT", tmp_path)
    monkeypatch.setattr(B, "ROOT", tmp_path)
    (tmp_path / "data" / "examples" / "s").mkdir(parents=True)
    (tmp_path / "data" / "examples" / "s" / "run.json").write_text("{}")
    api = FakeBucketApi({"old.tar.gz": old, "old.tar.gz.sha256": f"{old_sha}  old.tar.gz\n".encode()})
    return P, cfg, api, old_sha


def test_publish_uploads_verifies_and_pins(tmp_path, monkeypatch):
    P, cfg, api, old_sha = _publish_env(tmp_path, monkeypatch)
    res = P.publish_snapshot("new", "o/b", out_dir=tmp_path / "dist", config=cfg, api=api, log=lambda *_: 0)
    assert res["uploaded"] and api.uploads == ["new.tar.gz", "new.tar.gz.sha256"]
    snap = yaml.safe_load(cfg.read_text())["bootstrap"]["snapshot"]
    assert snap["id"] == "new" and snap["url"] == "hf://buckets/o/b/new.tar.gz" and snap["sha256"] == res["primary"]["sha256"]
    assert [f["id"] for f in snap["fallbacks"]] == ["old"]           # previous primary kept; "ghost" not in bucket
    assert snap["mounts"] == {"o/b": "/data"}
    assert yaml.safe_load(cfg.read_text())["bootstrap"]["suites"] == ["configs/smoke.yaml"]   # rest untouched
    assert P.check_pinned(cfg, api=api, log=lambda *_: 0)
    # publishing the same id again with identical content is a no-op upload
    again = P.publish_snapshot("new", "o/b", out_dir=tmp_path / "dist", config=cfg, api=api, log=lambda *_: 0)
    assert again["uploaded"] is False and api.uploads == ["new.tar.gz", "new.tar.gz.sha256"]


def test_publish_refuses_to_overwrite_an_existing_id(tmp_path, monkeypatch):
    P, cfg, api, _ = _publish_env(tmp_path, monkeypatch)
    api.files["taken.tar.gz"] = b"other"
    api.files["taken.tar.gz.sha256"] = b"f" * 64 + b"  taken.tar.gz\n"
    with pytest.raises(RuntimeError, match="immutable"):
        P.publish_snapshot("taken", "o/b", out_dir=tmp_path / "dist", config=cfg, api=api, log=lambda *_: 0)
    assert yaml.safe_load(cfg.read_text())["bootstrap"]["snapshot"]["id"] == "old"   # pin unchanged


def test_check_pinned_fails_when_snapshot_missing(tmp_path, monkeypatch):
    P, cfg, api, _ = _publish_env(tmp_path, monkeypatch)
    assert P.check_pinned(cfg, api=api, log=lambda *_: 0)
    del api.files["old.tar.gz"]
    assert not P.check_pinned(cfg, api=api, log=lambda *_: 0)
