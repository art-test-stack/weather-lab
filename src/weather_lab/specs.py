"""Compact, committed YAML specifications of frozen suites.

Manifests (JSON) are generated locally and are not committed. A spec records the
hashes a regenerated manifest must reproduce (protocol, exact case data, case counts),
the fixed-year source files, the AROME station extract and, optionally, the score
files a re-evaluation must reproduce. `data prepare` validates against it, so a clean
checkout regenerates the same frozen suite or fails loudly.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from weather_lab.common import ROOT, Config, now_utc, read_json, sha256_file

SPEC_DIR = ROOT / "benchmarks" / "specs"
SCORE_FILES = ("metrics_point.csv", "metrics_probabilistic.csv", "availability.csv")


def spec_path(cfg: Config) -> Path:
    return SPEC_DIR / f"{cfg.manifest_path.stem}.yaml"


def load_spec(cfg: Config) -> dict | None:
    p = spec_path(cfg)
    if not p.exists():
        return None
    with open(p) as f:
        return yaml.safe_load(f)


def environment_key(env: dict) -> str:
    """Exact score hashes depend on the numerical environment: torch version, OS and CPU arch."""
    plat = env.get("sys_platform") or ("darwin" if "macOS" in env.get("platform", "") else
                                      "linux" if "Linux" in env.get("platform", "") else "unknown")
    return f"torch-{env['torch']}/{plat}-{env['machine']}"


def systems_key(models: list[dict]) -> str:
    """Score files depend on which systems were run: hash the configured model list."""
    import json

    from weather_lab.common import sha256_text

    return "systems-" + sha256_text(json.dumps(models, sort_keys=True))[:12]


def reference_key(run_dir: Path) -> str:
    """`<environment>|<systems>` for a saved run (its run.json environment and config.yaml models)."""
    from weather_lab.common import load_config

    env = environment_key(read_json(run_dir / "run.json")["environment"])
    return f"{env}|{systems_key(load_config(run_dir / 'config.yaml')['models'])}"


def _score_references(spec: dict) -> dict[str, dict]:
    """`expected_scores` as {environment key: {file: sha256}} (legacy flat specs are converted)."""
    exp = spec.get("expected_scores") or {}
    if exp and all(isinstance(v, str) for v in exp.values()):   # legacy: frozen from the reference run
        return {spec.get("expected_scores_environment", "torch-2.14.1/darwin-arm64"): exp}
    return exp


def add_score_reference(cfg: Config, run_dir: Path, log=print) -> str:
    """Record a run's exact score hashes under its environment key; frozen fields stay untouched."""
    p = spec_path(cfg)
    spec = yaml.safe_load(p.read_text())
    run = read_json(run_dir / "run.json")
    if run["protocol_sha256"] != spec["protocol_sha256"]:
        raise RuntimeError("run does not belong to this frozen suite")
    refs = _score_references(spec)
    legacy = next(iter(refs)) if "reference_run" in spec else None
    if legacy is not None:   # fold the legacy single reference into the per-environment map
        spec.setdefault("reference_runs", {})[legacy] = spec.pop("reference_run")
    key = reference_key(run_dir)
    refs[key] = {f: sha256_file(run_dir / f) for f in SCORE_FILES}
    spec.pop("expected_scores_environment", None)
    spec["expected_scores"] = refs
    spec.setdefault("reference_runs", {})[key] = run["run_id"]
    header = p.read_text().splitlines()[0]
    with open(p, "w") as f:
        f.write(header + "\n")
        yaml.safe_dump(spec, f, sort_keys=False, allow_unicode=True)
    log(f"{p.name}: score reference for {key} from {run['run_id']}")
    return key


def freeze_spec(cfg: Config, run_dir: Path | None = None, note: str | None = None, log=print) -> Path:
    """Write the spec from the frozen manifest (and optionally a reference run's scores)."""
    m = read_json(cfg.manifest_path)
    spec = {
        "suite": m["suite"],
        "variant": m.get("variant"),
        "config": str(cfg.path.relative_to(ROOT)),
        "manifest": str(cfg.manifest_path.relative_to(ROOT)),
        "note": note,
        "frozen_at": m["created_at"],
        "protocol_sha256": m["protocol_sha256"],
        "case_data_sha256": m["case_data_sha256"],
        "n_cases": m["n_cases"],
        "n_eligible": m["n_eligible"],
        # fixed-year source files; the current-year file grows daily and is checked through case data
        "source_files": {s["file"]: s["sha256"] for s in m["dataset"]["sources"]
                         if not s["file"].startswith(f"synop_{m['created_at'][:4]}")},
    }
    if "arome" in m:
        spec["arome_extract_sha256"] = m["arome"]["sha256"]
    if run_dir is not None:
        run = read_json(run_dir / "run.json")
        key = reference_key(run_dir)
        spec["reference_runs"] = {key: run["run_id"]}
        spec["expected_scores"] = {key: {f: sha256_file(run_dir / f) for f in SCORE_FILES}}
    spec["written_at"] = now_utc()
    p = spec_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write("# Generated by `weather-lab spec freeze`; edit only by freezing a new suite version.\n")
        yaml.safe_dump(spec, f, sort_keys=False, allow_unicode=True)
    log(f"wrote spec {p.relative_to(ROOT)}")
    return p


def check_manifest(cfg: Config, manifest: dict) -> None:
    """Raise if a (re)generated manifest does not reproduce the committed spec."""
    spec = load_spec(cfg)
    if spec is None:
        return
    problems = [f"{k}: spec {spec[k]} != generated {manifest[k]}"
                for k in ("protocol_sha256", "case_data_sha256", "n_cases", "n_eligible") if spec[k] != manifest[k]]
    sources = {s["file"]: s["sha256"] for s in manifest["dataset"]["sources"]}
    problems += [f"source {f}: sha256 differs from spec" for f, h in spec.get("source_files", {}).items()
                 if f in sources and sources[f] != h]
    if "arome_extract_sha256" in spec and manifest.get("arome", {}).get("sha256") != spec["arome_extract_sha256"]:
        problems.append("AROME station extract differs from spec")
    if problems:
        raise RuntimeError(f"manifest does not reproduce {spec_path(cfg).name}: " + "; ".join(problems))


def check_scores(cfg: Config, run_dir: Path) -> dict:
    """Exact comparison with the reference recorded for the run's environment.

    Returns {file: bool}, or {} when the spec has no reference for that environment and system set (the result
    is then unverified, not failed: floating-point results legitimately differ across torch
    versions and platforms at the 1e-7 level).
    """
    spec = load_spec(cfg) or {}
    ref = _score_references(spec).get(reference_key(run_dir), {})
    return {f: sha256_file(run_dir / f) == h for f, h in ref.items()}
