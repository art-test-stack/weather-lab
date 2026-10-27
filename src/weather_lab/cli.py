"""weather-lab command line: data, evaluation, specs, bootstrap, snapshots, diagnostics and evidence."""

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
    from weather_lab.bootstrap import prepare_suite

    prepare_suite(load_config(args.config))


def cmd_evaluate(args) -> None:
    from weather_lab.evaluate import run_evaluation

    run_evaluation(load_config(args.config), reuse_from=Path(args.reuse) if args.reuse else None)


def cmd_score(args) -> None:
    from weather_lab.evaluate import rescore

    ok = rescore(Path(args.run_dir))
    sys.exit(0 if ok else 1)


def cmd_export(args) -> None:
    from weather_lab.example import export_example

    export_example([Path(p) for p in args.run_dirs])


def cmd_bootstrap(args) -> None:
    from weather_lab.bootstrap import BootstrapError, bootstrap

    try:
        bootstrap(args.mode, args.snapshot_url, args.snapshot_sha256, args.suite or None, weights=not args.no_weights)
    except BootstrapError as e:
        print(f"bootstrap FAILED: {e}\n{'retryable: run the same command again' if e.retryable else 'not retryable'}")
        sys.exit(2)


def cmd_spec_freeze(args) -> None:
    from weather_lab.specs import freeze_spec

    freeze_spec(load_config(args.config), Path(args.run_dir) if args.run_dir else None, args.note)


def cmd_spec_add_reference(args) -> None:
    from weather_lab.specs import add_score_reference

    add_score_reference(load_config(args.config), Path(args.run_dir))


def cmd_snapshot_publish(args) -> None:
    from weather_lab.publish import publish_snapshot

    publish_snapshot(args.id, args.bucket, dry_run=args.dry_run)


def cmd_snapshot_check(args) -> None:
    from weather_lab.publish import check_pinned

    sys.exit(0 if check_pinned() else 1)


def cmd_snapshot_build(args) -> None:
    from weather_lab.bootstrap import build_snapshot

    build_snapshot(Path(args.out), args.id)


def cmd_arome_refresh(args) -> None:
    from weather_lab.arome import refresh_suite

    refresh_suite(Path(args.base), Path(args.out_dir))


def cmd_diag_inputs(args) -> None:
    import pandas as pd

    from weather_lab.diagnostics import audit_table, lead_and_coverage_summary, write_report

    cfg = load_config(args.config)
    tab = audit_table(cfg, args.case, args.variable)
    write_report(f"arome_inputs/{args.case.replace('/', '_')}", tab)
    hist = tab[tab["segment"] == "history"]
    first = hist["arome_run"].notna().to_numpy().argmax() if hist["arome_run"].notna().any() else len(hist)
    show = pd.concat([hist.iloc[max(first - 2, 0): first + 2], hist.tail(3), tab[tab["segment"] == "future"].iloc[
        [0, 1, 7, 15]]])
    cols = ["valid_time", "segment", "observed", "arome_archived", "arome_run", "lead_h", "availability",
            "input_substobs_mode_v1"]
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print(show[cols].assign(valid_time=show["valid_time"].dt.strftime("%m-%d %H:%MZ"),
                                arome_run=show["arome_run"].str[5:16]).round(2).to_string(index=False))
    summary = lead_and_coverage_summary(cfg)
    write_report("arome_inputs/summary", summary)
    print({k: summary[k] for k in ("history_coverage_mean", "history_lead_h", "future_lead_h")})


def cmd_diag_covariates(args) -> None:
    import json

    from weather_lab.diagnostics import sensitivity, write_report

    res = sensitivity(load_config(args.config), args.delta, args.case or None)
    p = write_report("covariate_sensitivity" + ("_case" if args.case else ""), res)
    print(json.dumps(res["pairs"], indent=1))
    print(f"saved {p}")


def cmd_models_prefetch(args) -> None:
    from weather_lab.models import CHECKPOINTS, CheckpointError, prefetch

    names = args.names or ["Timestral-28M", "Timestral-209M"]
    try:
        for n in names:
            r = prefetch(CHECKPOINTS[n if n in CHECKPOINTS else f"Timestral-{n}"])
            print(f"{r['checkpoint']:<16} {r['repo']}@{r['revision'][:12]} "
                  f"{'downloaded' if r['downloaded'] else 'already cached'} ({r['bytes'] / 1e6:.0f} MB, sha256 verified)"
                  f"\n{'':<16} {r['path']}")
    except CheckpointError as e:
        print(f"prefetch FAILED: {e}")
        sys.exit(2)


def cmd_models_status(args) -> None:
    import json

    from weather_lab.status import cache_status

    st = cache_status()
    if args.json:
        print(json.dumps(st, indent=1, default=str))
        return
    print(f"offline mode: {st['offline']} · Hub cache: {st['hub_cache']}")
    print("checkpoint files on disk:")
    for r in st["checkpoints"]:
        print(f"  {r['checkpoint']:<16} {'on disk' if r['on_disk'] else 'absent ':<8} {r['mb']:>5} MB  {r['revision']}")
    print(f"models loaded in this process: {len(st['loaded'])} (a fresh CLI process holds none)")
    print("saved benchmark predictions:")
    for r in st["saved_runs"]:
        print(f"  {r['suite']:<24} {r['run_id']:<48} {r['systems']:>3} systems  {r['source']}")
    pc = st["prediction_cache"]
    print(f"exploratory prediction cache: {pc['entries']} entries, {pc['mb']} MB at {pc['path']} ({pc['persistence']})")


def cmd_diag_forecast_only(args) -> None:
    import json

    from weather_lab.diagnostics import forecast_only_response, write_report

    sizes = tuple(args.checkpoint or ("Timestral-28M", "Timestral-209M"))
    res = forecast_only_response(load_config(args.config), sizes, args.delta, args.lead)
    p = write_report("forecast_only_response", res)
    print(json.dumps(res["checkpoints"], indent=1))
    print(f"saved {p}")


def cmd_arome_evidence(args) -> None:
    from weather_lab.evidence import collect

    collect(load_config(args.config), args.full_checksum)


def cmd_arome_observe(args) -> None:
    from weather_lab.evidence import observe

    observe(load_config(args.config), args.last_runs)


def cmd_compare(args) -> None:
    from weather_lab.compare import compare, freeze_selection

    if args.export_combined:
        from weather_lab.compare import export_combined

        export_combined(Path(args.rule), Path(args.export_combined))
        return
    res = compare(Path(args.rule))
    if args.freeze:
        for p in freeze_selection(res, Path(args.rule)):
            print(f"wrote {p}")


def cmd_features_list(args) -> None:
    from weather_lab.features import GROUPS

    for g in GROUPS.values():
        print(f"{g.name:6} v{g.version} {g.role:5} rows={','.join(g.rows())}  needs={','.join(g.needs) or '-'}  "
              f"{g.description}")
        print(f"       availability: {g.availability}\n       missing: {g.missing}")
        if g.defaults:
            print("       parameters: " + "; ".join(f"{k}={list(v) if isinstance(v, tuple) else v} (allowed "
                                                  f"{list(g.choices[k])})" for k, v in g.defaults.items()))


def cmd_features_select(args) -> None:
    import json

    from weather_lab.feature_selection import freeze

    res = freeze([Path(d) for d in args.run_dir], Path(args.rule), Path(args.out))
    print(json.dumps(res, indent=1, default=str))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="weather-lab", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    from weather_lab.rag import parser as rag_parser

    rag_parser(sub)
    data = sub.add_parser("data", help="download / prepare observations and AROME extracts")
    dsub = data.add_subparsers(dest="data_command", required=True)
    for name, fn, help_ in [("fetch", cmd_fetch, "download annual SYNOP files (+ AROME messages) once"),
                            ("prepare", cmd_prepare, "clean, regularise and freeze the suite manifest")]:
        sp = dsub.add_parser(name, help=help_)
        sp.add_argument("--config", required=True)
        sp.set_defaults(func=fn)
    ev = sub.add_parser("evaluate", help="run every system on the frozen suite and save predictions + scores")
    ev.add_argument("--config", required=True)
    ev.add_argument("--reuse", metavar="RUN_DIR", help="copy predictions of identical systems from an earlier run of "
                    "the same frozen manifest (verified: manifest, data, AROME, targets, system description)")
    ev.set_defaults(func=cmd_evaluate)
    sc = sub.add_parser("score", help="recompute scores from a saved run directory (no model calls)")
    sc.add_argument("--run-dir", required=True)
    sc.set_defaults(func=cmd_score)
    ex = sub.add_parser("export-example", help="package saved runs as the offline UI example (data/examples)")
    ex.add_argument("run_dirs", nargs="+")
    ex.set_defaults(func=cmd_export)
    bs = sub.add_parser("bootstrap", help="fetch/verify/prepare everything the app and suites need (idempotent)")
    bs.add_argument("--mode", choices=["auto", "official", "snapshot"], default="auto")
    bs.add_argument("--snapshot-url")
    bs.add_argument("--snapshot-sha256")
    bs.add_argument("--suite", action="append", help="config path(s); default: configs/bootstrap.yaml")
    bs.add_argument("--no-weights", action="store_true")
    bs.set_defaults(func=cmd_bootstrap)
    spec = sub.add_parser("spec", help="committed YAML specs of frozen suites")
    ssub = spec.add_subparsers(dest="spec_command", required=True)
    sf = ssub.add_parser("freeze", help="write benchmarks/specs/<manifest>.yaml from the frozen manifest")
    sf.add_argument("--config", required=True)
    sf.add_argument("--run-dir", help="reference run whose score files a re-evaluation must reproduce")
    sf.add_argument("--note")
    sf.set_defaults(func=cmd_spec_freeze)
    sa = ssub.add_parser("add-reference", help="record a run's exact score hashes for its torch/platform")
    sa.add_argument("--config", required=True)
    sa.add_argument("--run-dir", required=True)
    sa.set_defaults(func=cmd_spec_add_reference)
    snap = sub.add_parser("snapshot", help="immutable data snapshot for snapshot-mode bootstrap")
    snsub = snap.add_subparsers(dest="snapshot_command", required=True)
    sb = snsub.add_parser("build")
    sb.add_argument("--out", default="dist/snapshot")
    sb.add_argument("--id", required=True)
    sb.set_defaults(func=cmd_snapshot_build)
    sp = snsub.add_parser("publish", help="build, upload to the bucket, verify, and pin in configs/bootstrap.yaml")
    sp.add_argument("--id", required=True, help="new immutable snapshot id, e.g. wl-snapshot-v5-20261026")
    sp.add_argument("--bucket", default="ai-testack/weather-lab-storage")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_snapshot_publish)
    sc2 = snsub.add_parser("check", help="read-only: the pinned snapshot exists in its bucket with the pinned sha256")
    sc2.set_defaults(func=cmd_snapshot_check)
    ar = sub.add_parser("arome", help="AROME archive utilities")
    asub = ar.add_subparsers(dest="arome_command", required=True)
    rf = asub.add_parser("refresh-suite", help="write a NEW suite config for the dates now in the 14-day archive")
    rf.add_argument("--base", default="configs/arome_smoke.yaml")
    rf.add_argument("--out-dir", default="configs")
    rf.set_defaults(func=cmd_arome_refresh)
    ev = asub.add_parser("evidence", help="re-query fetched resources: headers, checksums, catalogue timestamps")
    ev.add_argument("--config", default="configs/arome_smoke.yaml")
    ev.add_argument("--full-checksum", action="append", help="URL to download fully for SHA-256/MD5 vs ETag")
    ev.set_defaults(func=cmd_arome_evidence)
    ob = asub.add_parser("observe", help="manually record first_seen_available_at for the newest complete run(s)")
    ob.add_argument("--config", default="configs/arome_smoke.yaml")
    ob.add_argument("--last-runs", type=int, default=1)
    ob.set_defaults(func=cmd_arome_observe)
    mo = sub.add_parser("models", help="pinned checkpoints: prefetch and cache status (WEATHER_LAB_OFFLINE=1: no Hub)")
    msub = mo.add_subparsers(dest="models_command", required=True)
    mp = msub.add_parser("prefetch", help="download + verify inference files of pinned checkpoints (default 28M 209M)")
    mp.add_argument("names", nargs="*", help="e.g. Timestral-28M 209M")
    mp.set_defaults(func=cmd_models_prefetch)
    ms = msub.add_parser("status", help="checkpoint files on disk, loaded models, saved and cached predictions")
    ms.add_argument("--json", action="store_true")
    ms.set_defaults(func=cmd_models_status)
    fe = sub.add_parser("features", help="context-enrichment feature groups: list, select on validation")
    fsub = fe.add_subparsers(dest="features_command", required=True)
    fl = fsub.add_parser("list", help="feature groups, roles, rows and versions")
    fl.set_defaults(func=cmd_features_list)
    fs = fsub.add_parser("select", help="apply configs/features/selection_rule.yaml to validation runs and freeze it")
    fs.add_argument("--run-dir", action="append", required=True, help="validation run(s) (repeat for a combination)")
    fs.add_argument("--rule", default="configs/features/selection_rule.yaml")
    fs.add_argument("--out", default="configs/features/frozen_selection.yaml")
    fs.set_defaults(func=cmd_features_select)
    dg = sub.add_parser("diagnose", help="audits and sensitivity checks")
    dgs = dg.add_subparsers(dest="diag_command", required=True)
    di = dgs.add_parser("arome-inputs", help="table of what the AROME covariate contains for one case")
    di.add_argument("--config", required=True)
    di.add_argument("--case", required=True)
    di.add_argument("--variable", default="t")
    di.set_defaults(func=cmd_diag_inputs)
    dc = dgs.add_parser("covariates", help="A/B/C AROME covariate comparison with a future-only perturbation")
    dc.add_argument("--config", required=True)
    dc.add_argument("--case", action="append")
    dc.add_argument("--delta", type=float, default=5.0)
    dc.set_defaults(func=cmd_diag_covariates)
    fo = dgs.add_parser("forecast-only", help="does a checkpoint use future-only AROME? reorder/local perturbations")
    fo.add_argument("--config", default="configs/arome_v2_matrix.yaml")
    fo.add_argument("--checkpoint", action="append", default=None)
    fo.add_argument("--delta", type=float, default=5.0)
    fo.add_argument("--lead", type=int, default=24)
    fo.set_defaults(func=cmd_diag_forecast_only)
    cp = sub.add_parser("compare", help="score candidate runs on one common set and apply a pre-declared rule")
    cp.add_argument("--rule", default="configs/context/selection_rule.yaml")
    cp.add_argument("--freeze", action="store_true", help="write configs/frozen/... and the frozen test config")
    cp.add_argument("--export-combined", metavar="DIR", help="write one run directory with all candidates (for the app)")
    cp.set_defaults(func=cmd_compare)
    p.add_argument("--offline", action="store_true", help="no Hub requests (same as WEATHER_LAB_OFFLINE=1)")
    args = p.parse_args(argv)
    if args.offline:
        import os

        os.environ["WEATHER_LAB_OFFLINE"] = os.environ["HF_HUB_OFFLINE"] = "1"
    args.func(args)


if __name__ == "__main__":
    main()

