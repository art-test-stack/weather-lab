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
