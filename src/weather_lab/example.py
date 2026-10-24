"""Package saved runs as a small, attributed real-data example for offline UI replay.

`data/examples/<suite>/` receives the run's predictions, targets, scores, manifest and
provenance, plus the observation windows the explorer plots. For the AROME suite the
station-level AROME extract is included too, so the suite can be re-prepared after the
14-day archive has rolled over.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from weather_lab.common import ROOT, Config, load_config, now_utc, read_json, write_json
from weather_lab.data import DATASET_PAGE, LICENCE

EXAMPLES = ROOT / "data" / "examples"


def export_example(run_dirs: list[Path], log=print) -> None:
    for run_dir in run_dirs:
        run = read_json(run_dir / "run.json")
        cfg = load_config(run_dir / "config.yaml")
        dest = EXAMPLES / run["suite"]
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for p in run_dir.iterdir():
            if p.is_file():
                shutil.copy(p, dest / p.name)
        extra = []
        if cfg.get("arome"):
            src = cfg.dir("processed") / f"arome_{cfg.suite}.parquet"
            shutil.copy(src, dest / "arome_station_extract.parquet")
            shutil.copy(src.with_suffix(".meta.json"), dest / "arome_station_extract.meta.json")
            extra.append("arome_station_extract.parquet: AROME 0.025 deg SP1 2 m temperature and MSLP at the "
                         "nearest grid point, from the Météo-France 14-day PNT archive (see its .meta.json)")
        from weather_lab.data import load_observations

        _, meta = load_observations(cfg)
        write_json(dest / "ATTRIBUTION.json", {
            "exported_at": now_utc(), "run_id": run["run_id"], "suite": run["suite"], "label": run["label"],
            "observations": {"source": "Météo-France, Archive Synop OMM", "dataset": DATASET_PAGE,
                             "files": meta["sources"], "license": LICENCE,
                             "processing": meta["processing"], "coverage": meta["time_axis"],
                             "stations": meta["stations"]},
            "predictions": "computed by this project; Timestral outputs from the pinned open-weight checkpoint "
                           "(Modified MIT licence, see docs/sources.md)",
            "extra": extra,
        })
        size = sum(p.stat().st_size for p in dest.iterdir())
        log(f"exported {run_dir.name} -> {dest.relative_to(ROOT)} ({size / 1e6:.1f} MB)")


def restore_arome_extract(cfg: Config, log=print) -> bool:
    """If the raw AROME subsets are absent (archive rolled over), reuse the packaged extract."""
    raw_runs = list((cfg.dir("raw") / "arome").glob("*/run.json"))
    packaged = EXAMPLES / cfg.suite / "arome_station_extract.parquet"
    if raw_runs or not packaged.exists():
        return False
    target = cfg.dir("processed") / f"arome_{cfg.suite}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(packaged, target)
    shutil.copy(packaged.with_suffix(".meta.json"), target.with_suffix(".meta.json"))
    log(f"raw AROME subsets absent: restored packaged station extract {packaged.relative_to(ROOT)}")
    return True
