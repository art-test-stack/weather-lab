"""Optional t0 loader uses the shared cache without touching bootstrap or old pins."""
import sys
from types import SimpleNamespace

import pytest

from weather_lab import models


def test_t0_loader_checks_runtime_before_fetch_and_reuses_loaded_weights(monkeypatch):
    ck = models.CHECKPOINTS["t0-gamma"]
    fetches, loads = [], []
    monkeypatch.setattr("importlib.metadata.version", lambda _: "0.5.0")
    monkeypatch.setattr(models, "prefetch", lambda c: fetches.append(c.ident) or
                        {"path": "/local/pinned", "downloaded": False})
    cache = models.ModelCache()
    with pytest.raises(models.CheckpointError, match="pinned"):
        cache.get(ck)
    assert not fetches

    class Fake:
        config = SimpleNamespace(xpos_scale_layout="halves", scaler_zero_no_value=False,
                                 quantile_levels=models.TIMESTRAL_QUANTILES)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            loads.append((path, kwargs))
            return cls()

        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

        def parameters(self):
            return [SimpleNamespace(numel=lambda: 10)]

    monkeypatch.setattr("importlib.metadata.version", lambda _: "0.6.0")
    monkeypatch.setitem(sys.modules, "t0", SimpleNamespace(T0Forecaster=Fake))
    item = cache.get(ck)
    assert cache.get(ck) is item and len(loads) == len(fetches) == 1
    assert loads[0][1] == {"local_files_only": True}
    assert item.info["tfc_t0"] == "0.6.0" and item.info["quantile_levels"] == list(models.TIMESTRAL_QUANTILES)
    cache.release()
    Fake.config = SimpleNamespace(xpos_scale_layout="interleaved", scaler_zero_no_value=True)
    with pytest.raises(models.CheckpointError, match="conventions"):
        cache.get(ck)


def test_t0_factory_capabilities_and_timestral_explorer_remain_separate():
    from weather_lab.evaluate import build_adapters
    from weather_lab.forecast import _registry
    from weather_lab.forecast_config import ForecastConfig, check

    cfg = {"variables": ["t", "pmer", "u", "v"], "forecast": {"context_days": 30},
           "models": [{"id": "t0-research", "kind": "t0", "targets": ["t"],
                       "calendar": True, "trim_leading_missing": True}]}
    adapter, = build_adapters(cfg)
    assert adapter.kind == "t0" and adapter.options["trim_leading_missing"]
    assert adapter.revision == models.CHECKPOINTS["t0-gamma"].revision
    assert "t0-gamma" not in _registry()
    issues = check(ForecastConfig.make("t0-gamma", "substobs"))
    assert any(i.kind == "unsupported" and i.subject == "conditioning" for i in issues)
