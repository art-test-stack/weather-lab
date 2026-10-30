"""Cache status in one place: checkpoint files on disk, models loaded in this process, saved benchmark
predictions and the exploratory prediction cache. Reads only local state; never contacts the Hub."""

from __future__ import annotations

from weather_lab.common import ROOT, read_json


def saved_runs() -> list[dict]:
    rows = []
    for source, base in (("local run", ROOT / "runs"), ("installed example", ROOT / "data" / "examples")):
        for p in sorted(base.glob("*/*/run.json")) + sorted(base.glob("*/run.json")):
            r = read_json(p)
            if (p.parent / "predictions.parquet").exists():
                rows.append({"suite": r["suite"], "run_id": r["run_id"], "systems": len(r["systems"]),
                             "source": source, "path": str(p.parent.relative_to(ROOT))})
    return rows


def cache_status() -> dict:
    from weather_lab import models, predcache

    return {"offline": models.offline(), "hub_cache": str(models.hub_cache()),
            "checkpoints": models.checkpoint_status(), "loaded": models.MODELS.status(),
            "model_stats": dict(models.STATS), "saved_runs": saved_runs(), "prediction_cache": predcache.status()}
