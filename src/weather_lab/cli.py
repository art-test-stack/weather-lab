"""weather-lab command line: data fetch / data prepare / evaluate / score / export-example."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from weather_lab.common import load_config


def cmd_fetch(args) -> None:
    from weather_lab.data import fetch_synop

    cfg = load_config(args.config)
    fetch_synop(cfg)
    if cfg.get("arome"):
        from weather_lab.arome import fetch_arome

        recs = fetch_arome(cfg)
        ok = sum(r["status"] == "ok" for r in recs)
        print(f"AROME: {ok}/{len(recs)} runs qualify; "
              f"{sum(r.get('bytes_downloaded', 0) for r in recs) / 1e6:.1f} MB of GRIB2 messages")


def cmd_prepare(args) -> None:
    from weather_lab.data import load_observations, prepare_synop
    from weather_lab.suite import build_cases, write_manifest

    cfg = load_config(args.config)
    prepare_synop(cfg)
    obs, meta = load_observations(cfg)  # checksum what later stages will read back from Parquet
    arome_meta = None
    if cfg.get("arome"):
        from weather_lab.arome import prepare_arome
        from weather_lab.example import restore_arome_extract

        if not restore_arome_extract(cfg):
            _, arome_meta = prepare_arome(cfg, meta["stations"])
        else:
            from weather_lab.arome import load_arome

            _, arome_meta = load_arome(cfg)
        cases = build_cases(cfg, obs, arome_meta["runs"])
    else:
        cases = build_cases(cfg, obs)
    write_manifest(cfg, obs, meta, cases, arome_meta)


def cmd_evaluate(args) -> None:
    from weather_lab.evaluate import run_evaluation

    run_evaluation(load_config(args.config))


def cmd_score(args) -> None:
    from weather_lab.evaluate import rescore

    ok = rescore(Path(args.run_dir))
    sys.exit(0 if ok else 1)


def cmd_export(args) -> None:
    from weather_lab.example import export_example

    export_example([Path(p) for p in args.run_dirs])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="weather-lab", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data", help="download / prepare observations and AROME extracts")
    dsub = data.add_subparsers(dest="data_command", required=True)
    for name, fn, help_ in [("fetch", cmd_fetch, "download annual SYNOP files (+ AROME messages) once"),
                            ("prepare", cmd_prepare, "clean, regularise and freeze the suite manifest")]:
        sp = dsub.add_parser(name, help=help_)
        sp.add_argument("--config", required=True)
        sp.set_defaults(func=fn)
    ev = sub.add_parser("evaluate", help="run every system on the frozen suite and save predictions + scores")
    ev.add_argument("--config", required=True)
    ev.set_defaults(func=cmd_evaluate)
    sc = sub.add_parser("score", help="recompute scores from a saved run directory (no model calls)")
    sc.add_argument("--run-dir", required=True)
    sc.set_defaults(func=cmd_score)
    ex = sub.add_parser("export-example", help="package saved runs as the offline UI example (data/examples)")
    ex.add_argument("run_dirs", nargs="+")
    ex.set_defaults(func=cmd_export)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
