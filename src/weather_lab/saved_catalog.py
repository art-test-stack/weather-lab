"""Shared discovery of the exact saved configurations displayed by Benchmark."""
from pathlib import Path

from weather_lab.common import read_json


def saved_configurations(suite: str) -> dict[str, dict]:
    """Small discovery records only; no prediction reads, network calls or inference."""
    from weather_lab.development import location, saved_runs

    found, seen = {}, set()
    def priority(item):
        p = Path(item[1])
        complete_matrix = (p.parents[1] / "comparisons/provenance.json").is_file()
        return not complete_matrix, not p.is_relative_to(location())

    for label, path in sorted(saved_runs(suite), key=priority):
        p = Path(path)
        summary = p / "matrix_setup.json"
        rec = read_json(summary if summary.exists() else p / "run.json")
        c = rec["forecast_config"]
        digest = rec.get("run_json_sha256")
        if digest and digest in seen:
            continue  # the same immutable record may exist locally and on the bucket mount
        if c["step_hours"] == 1 and c["horizon_h"] == 48:
            if digest:
                seen.add(digest)
            found[path] = {"label": rec.get("configuration_id", label).replace("/", " · ").replace("-prediction", " prediction"),
                           "config": c, "summary": rec}
    labels = [r["label"] for r in found.values()]
    for path, record in found.items():
        if labels.count(record["label"]) > 1:
            record["label"] += f" · {Path(path).name}"
    return found

