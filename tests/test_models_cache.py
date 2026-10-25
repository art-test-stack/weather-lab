"""Checkpoint files, loaded-model cache and exploratory prediction cache (no network, no weights)."""

import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from weather_lab import models, predcache


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """A Hub cache in tmp_path: snapshot_download writes the inference files of the asked revision."""
    calls, store = [], {}

    def local_dir(ck):
        return store.get(ck.ident)

    def snapshot_download(repo, revision, allow_patterns):
        calls.append((repo, revision, tuple(allow_patterns)))
        d = tmp_path / repo.replace("/", "--") / revision
        d.mkdir(parents=True)
        for f in allow_patterns:
            (d / f).write_bytes(b"x")
        store[f"{repo}@{revision}"] = d
        return str(d)

    monkeypatch.setattr(models, "local_dir", local_dir)
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    monkeypatch.setattr(models, "_verify", lambda ck, folder: None)
    monkeypatch.delenv("WEATHER_LAB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return calls


def test_prefetch_downloads_only_inference_files_once_per_revision(fake_hub, monkeypatch):
    ck = models.CHECKPOINTS["Timestral-209M"]
    with pytest.raises(PermissionError):
        models.prefetch(ck, allow_download=False)
    assert models.prefetch(ck)["downloaded"] is True
    assert models.prefetch(ck)["downloaded"] is False                  # every later variant reuses the files
    assert fake_hub == [(ck.repo, ck.revision, ("config.json", "model.safetensors"))]
    monkeypatch.setenv("WEATHER_LAB_OFFLINE", "1")
    assert models.prefetch(ck)["downloaded"] is False                  # offline + cached: fine
    with pytest.raises(models.CheckpointError, match="offline"):
        models.prefetch(models.CHECKPOINTS["Timestral-5M"])            # offline + absent: actionable error
    assert len(fake_hub) == 1


def test_resolve_pins_full_revisions():
    assert models.resolve("209M").revision == "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"
    assert models.resolve("mistralai/Timestral-28M").name == "Timestral-28M"
    with pytest.raises(models.CheckpointError):
        models.resolve("mistralai/Timestral-28M", "main")
    with pytest.raises(models.CheckpointError):
        models.resolve("someone/model")


def test_verify_rejects_a_corrupted_file(tmp_path):
    ck = models.CHECKPOINTS["Timestral-5M"]
    for f in ck.files:
        (tmp_path / f).write_bytes(b"truncated")
    with pytest.raises(models.CheckpointError, match="prefetch Timestral-5M"):
        models._verify(ck, tmp_path)


def test_model_cache_reuses_and_bounds_loaded_models(monkeypatch):
    loads = []

    def fake_load(self, ck, device, dtype):
        loads.append((ck.name, device))
        return models.Loaded(object(), {}, threading.Lock(), 100_000_000)

    monkeypatch.setattr(models.ModelCache, "_load", fake_load)
    cache = models.ModelCache(max_models=2, max_mb=10_000)
    a = cache.get(models.CHECKPOINTS["Timestral-28M"])
    assert cache.get(models.CHECKPOINTS["Timestral-28M"]) is a          # same revision/device/dtype: one load
    cache.get(models.CHECKPOINTS["Timestral-209M"])
    cache.get(models.CHECKPOINTS["Timestral-5M"])                       # third model evicts the least recent
    assert [s["repo"] for s in cache.status()] == ["mistralai/Timestral-209M", "mistralai/Timestral-5M"]
    cache.get(models.CHECKPOINTS["Timestral-209M"], device="cuda")
    assert cache.release("cuda") == 1                                   # ZeroGPU: GPU copies never kept
    assert loads.count(("Timestral-28M", "cpu")) == 1 and len(loads) == 4
    small = models.ModelCache(max_models=5, max_mb=150)                  # RAM budget evicts too
    small.get(models.CHECKPOINTS["Timestral-28M"])
    small.get(models.CHECKPOINTS["Timestral-209M"])
    assert len(small.status()) == 1


def test_variants_of_one_checkpoint_share_one_loaded_model(monkeypatch):
    from weather_lab.evaluate import build_adapters
    from weather_lab.common import load_config, ROOT

    seen = []
    monkeypatch.setattr(models.MODELS, "get", lambda ck, device="cpu", dtype="float32": seen.append(ck.ident))
    adapters = [a for a in build_adapters(load_config(ROOT / "configs" / "arome_v2_matrix.yaml"))
                if a.kind == "timestral"]
    keys = {(a.checkpoint, a.revision) for a in adapters}
    assert len(adapters) == 8 and len(keys) == 2                        # 4 input setups x 2 checkpoints


def _frame(v):
    return pd.DataFrame({"point": [v]})


def test_prediction_cache_hits_misses_and_single_flight(tmp_path):
    base = {"checkpoint": "mistralai/Timestral-209M", "revision": "9874d252", "arome_mode": "archived"}
    ctx = np.array([[1.0, np.nan, 3.0]], dtype=np.float32)
    k = predcache.make_key(base, {"context": ctx})
    assert predcache.make_key(dict(base), {"context": ctx.copy()}) == k
    assert predcache.make_key(base | {"revision": "other"}, {"context": ctx}) != k
    assert predcache.make_key(base | {"arome_mode": "missing"}, {"context": ctx}) != k
    assert predcache.make_key(base, {"context": np.array([[1.0, 2.0, 3.0]], np.float32)}) != k   # mask matters
    calls = []

    def slow():
        calls.append(1)
        time.sleep(0.2)
        return _frame(1.0)

    out = []
    threads = [threading.Thread(target=lambda: out.append(predcache.get_or_compute(k, base, slow, tmp_path)))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and sorted(h for _, h in out) == [False, True, True, True]
    assert not list(tmp_path.rglob("*.tmp"))
    frame, hit = predcache.get_or_compute(k, base, slow, tmp_path)
    assert hit and frame["point"].iloc[0] == 1.0 and len(calls) == 1
