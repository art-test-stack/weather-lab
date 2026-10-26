"""Reproducible, explicitly invoked research stages; frozen runs and app caches stay read-only."""
from __future__ import annotations

import argparse
import json
import resource
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from weather_lab.common import ROOT, environment, now_utc, read_json, sha256_file, write_json
from weather_lab.evaluate import INTERVALS, score, score_run
from weather_lab.forecast import QUANTILE_LEVELS, qcol
from weather_lab.suite import cases_from_manifest

PROTOCOL = ROOT / "docs/experiments/practical_improvements_v1.yaml"
PAIR = ["case_id", "variable", "lead_h"]


def causal_errors(arome: pd.DataFrame, past: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Join earlier issued forecasts to already available observations, never evaluation targets."""
    if past.time.gt(origin).any():
        raise ValueError("Correction observations extend beyond the availability cutoff")
    if arome.duplicated(["station", "run", "variable", "valid_time"]).any():
        raise ValueError("Duplicate issued AROME values")
    implied = (arome.valid_time - arome.run).dt.total_seconds() / 3600
    if not np.array_equal(implied, arome.lead_h):
        raise ValueError("AROME run lead does not match init and valid time")
    fields = sorted(set(arome.variable) & set(past.columns))
    obs = past.melt(id_vars=["station", "time"], value_vars=fields, var_name="variable", value_name="observation")
    if obs.duplicated(["station", "time", "variable"]).any():
        raise ValueError("Duplicate observations")
    a = arome[arome.available_at.le(origin) & arome.available_at.le(arome.valid_time)
              & arome.valid_time.le(origin) & arome.run.lt(arome.valid_time)]
    out = a.merge(obs, left_on=["station", "valid_time", "variable"],
                  right_on=["station", "time", "variable"], validate="many_to_one")
    out["error"] = out.observation - out.value
    return out[np.isfinite(out.error)].copy()


def median_adjustment(errors: pd.DataFrame, station: str, run: pd.Timestamp, valid: pd.Timestamp,
                      variable: str, origin: pd.Timestamp) -> dict:
    """Approved C1: exact run lead/cycle, seven days, median pooling and fixed support rules."""
    if errors.valid_time.gt(origin).any() or errors.available_at.gt(origin).any():
        raise ValueError("Correction history contains a future label or a late forecast")
    lead = (valid - run).total_seconds() / 3600
    pool = errors[errors.variable.eq(variable) & errors.lead_h.eq(lead) & errors.run.dt.hour.eq(run.hour)
                  & errors.run.lt(run) & errors.run.ge(origin - pd.Timedelta(days=7))]
    local = pool[pool.station.eq(station)].error
    n, pooled_n = len(local), len(pool)
    if pooled_n < 10:
        correction, policy = 0.0, "zero_insufficient_pool"
    elif n < 3:
        correction, policy = float(pool.error.median()), "pooled_insufficient_local"
    else:
        correction = float((n * local.median() + 5 * pool.error.median()) / (n + 5))
        policy = "local_shrunk_to_pool"
    return {"correction": correction, "local_n": n, "pooled_n": pooled_n, "policy": policy}


def nudge(arome: pd.DataFrame, past: pd.DataFrame, station: str, run: pd.Timestamp,
          valid: pd.Timestamp, variable: str, origin: pd.Timestamp, tau: int) -> dict:
    if tau not in (6, 12, 24) or past.time.gt(origin).any():
        raise ValueError("Nudging requires a declared tau and observations available by origin")
    obs = past[past.station.eq(station) & past[variable].notna()].sort_values("time")
    if obs.empty:
        return {"correction": 0.0, "policy": "raw_no_observation", "observation_age_h": np.nan}
    last = obs.iloc[-1]
    age = (origin - last.time).total_seconds() / 3600
    rows = arome[arome.station.eq(station) & arome.run.eq(run) & arome.variable.eq(variable)
                 & arome.valid_time.eq(last.time) & arome.available_at.le(origin)]
    if rows.empty or not np.isfinite(rows.value.iloc[0]):
        return {"correction": 0.0, "policy": "raw_no_selected_run_value_at_observation", "observation_age_h": age}
    if len(rows) != 1:
        raise ValueError("Ambiguous selected-run value for nudging")
    elapsed = (valid - last.time).total_seconds() / 3600
    return {"correction": float(np.exp(-elapsed / tau) * (last[variable] - rows.value.iloc[0])),
            "policy": "nudged", "observation_age_h": age}


def point_blend(raw: pd.DataFrame, model: pd.DataFrame, alpha: float) -> pd.DataFrame:
    if alpha not in (0, .25, .5, .75, 1):
        raise ValueError("Blend alpha is outside the frozen grid")
    out = raw.merge(model[PAIR + ["point"]].rename(columns={"point": "model_median"}),
                    on=PAIR, how="left", validate="one_to_one")
    out["point"] += alpha * (out.model_median - out.point)
    out["model"] = f"c3-blend-alpha{alpha:g}"
    out["output_type"] = "point"
    out["model_revision"] = "point-blend-v1"
    for q in QUANTILE_LEVELS:
        out[qcol(q)] = np.nan
    out["policy"] = np.where(out.point.notna(), "point_blend", "unavailable_missing_component")
    return out


def period_cases(manifest: dict) -> pd.DataFrame:
    cases = cases_from_manifest(manifest)
    cases["source_split"] = cases.split
    cases["split"] = np.select([cases.origin.lt(pd.Timestamp("2026-10-14", tz="UTC")),
                                 cases.origin.lt(pd.Timestamp("2026-10-16", tz="UTC"))],
                                ["development_warmup", "purged"], default="development_selection")
    return cases


def matched_rows(predictions: pd.DataFrame, targets: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    if predictions.duplicated(["model"] + PAIR).any() or targets.duplicated(PAIR).any():
        raise ValueError("Duplicate prediction or target pairs")
    rows = predictions.merge(targets[PAIR + ["observed"]], on=PAIR, validate="many_to_one")
    rows = rows.merge(cases[["case_id", "split", "season", "eligible"]], on="case_id", validate="many_to_one")
    rows["target_observed"] = np.isfinite(rows.observed)
    rows["prediction_finite"] = np.isfinite(rows.point)
    rows["err"] = rows.point - rows.observed
    rows["abs_err"] = abs(rows.err)
    rows["sq_err"] = rows.err ** 2
    return rows


def summaries(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full coverage and paired raw comparisons, separately per period and physical variable."""
    records, leads = [], []
    raw = rows[rows.model.eq("arome-raw")][PAIR + ["point"]].rename(columns={"point": "raw_point"})
    for period, period_rows in [(p, g) for p, g in rows.groupby("split")] + [("all_development_descriptive", rows)]:
        for (var, model), g in period_rows.groupby(["variable", "model"]):
            eligible = g[g.eligible]
            valid = eligible[eligible.target_observed & eligible.prediction_finite]
            pair = valid.merge(raw, on=PAIR, validate="one_to_one").dropna(subset="raw_point")
            pair["delta_abs"] = pair.abs_err - abs(pair.raw_point - pair.observed)
            base = {"period": period, "variable": var, "model": model,
                    "eligible_pairs": len(eligible), "valid_pairs": len(valid), "common_raw_pairs": len(pair),
                    "station_balanced_mae": valid.groupby("station").abs_err.mean().mean(),
                    "micro_mae": valid.abs_err.mean(), "rmse": np.sqrt(valid.sq_err.mean()), "bias": valid.err.mean(),
                    "common_raw_mae": pair.groupby("station").abs_err.mean().mean(),
                    "delta_mae_vs_raw": pair.groupby("station").delta_abs.mean().mean(),
                    "coverage_fraction": len(valid) / len(eligible) if len(eligible) else np.nan,
                    "blocks_2d": pair.origin.dt.floor("2D").nunique(),
                    "blocks_7d": pair.origin.dt.floor("7D").nunique(),
                    "uncertainty": "withheld_fewer_than_10_blocks", "ci95_low": np.nan, "ci95_high": np.nan}
            if g.output_type.eq("quantiles").all():
                q = valid[[qcol(x) for x in QUANTILE_LEVELS]].to_numpy()
                error = valid.observed.to_numpy()[:, None] - q
                base["pinball"] = np.maximum(np.array(QUANTILE_LEVELS) * error,
                                             (np.array(QUANTILE_LEVELS) - 1) * error).mean()
                for nominal, (lo, hi) in INTERVALS.items():
                    base[f"coverage{nominal}"] = valid.observed.between(valid[qcol(lo)], valid[qcol(hi)]).mean()
                    base[f"width{nominal}"] = (valid[qcol(hi)] - valid[qcol(lo)]).mean()
            records.append(base)
            for lead, h in pair.groupby("lead_h"):
                leads.append({**base, "lead_h": lead, "valid_pairs": len(h),
                              "eligible_pairs": int(eligible.lead_h.eq(lead).sum()), "common_raw_pairs": len(h),
                              "station_balanced_mae": h.groupby("station").abs_err.mean().mean(),
                              "common_raw_mae": h.groupby("station").abs_err.mean().mean(),
                              "micro_mae": h.abs_err.mean(), "rmse": np.sqrt(h.sq_err.mean()), "bias": h.err.mean(),
                              "delta_mae_vs_raw": h.delta_abs.mean(),
                              **{k: np.nan for k in ("pinball", "coverage80", "coverage90", "width80", "width90")}})
                if g.output_type.eq("quantiles").all():
                    error = h.observed.to_numpy()[:, None] - h[[qcol(x) for x in QUANTILE_LEVELS]].to_numpy()
                    leads[-1]["pinball"] = np.maximum(np.array(QUANTILE_LEVELS) * error,
                                                      (np.array(QUANTILE_LEVELS) - 1) * error).mean()
                    for nominal, (lo, hi) in INTERVALS.items():
                        leads[-1][f"coverage{nominal}"] = h.observed.between(h[qcol(lo)], h[qcol(hi)]).mean()
                        leads[-1][f"width{nominal}"] = (h[qcol(hi)] - h[qcol(lo)]).mean()
    return pd.DataFrame(records), pd.DataFrame(leads)


def independent_rescore(out: Path) -> dict:
    """Independent project scorer reads persisted files; compare against the research arithmetic."""
    predictions = pd.read_parquet(out / "predictions.parquet")
    targets = pd.read_parquet(out / "targets.parquet")
    cases = pd.read_parquet(out / "cases.parquet")
    table = pd.read_csv(out / "summary.csv")
    checked, max_difference = 0, 0.0
    for model in predictions.model.unique():
        pred = predictions[predictions.model.eq(model)]
        if pred.output_type.eq("point").all():
            # The frozen scorer needs a distribution to construct its probability tables.
            pred = pd.concat([pred, predictions[predictions.model.eq("climatology")]], ignore_index=True)
        independent = score(pred, targets, cases, {}, [6, 24, 48])
        standard = independent["metrics_point"]
        standard = standard[standard.model.eq(model)]
        # Aggregate dimensions use ALL and a null lead in the existing scorer.
        for row in table[table.model.eq(model) & table.period.ne("all_development_descriptive")].itertuples():
            s = standard[standard.split.eq(row.period) & standard.variable.eq(row.variable)
                         & standard.station.eq("ALL") & standard.season.eq("ALL") & standard.lead_h.isna()]
            if len(s) != 1:
                raise AssertionError(f"Missing independent score for {model}/{row.period}/{row.variable}")
            for metric, ours in (("mae", row.micro_mae), ("rmse", row.rmse), ("bias", row.bias)):
                delta = abs(float(s.iloc[0][metric]) - ours)
                max_difference = max(max_difference, delta)
                assert delta <= 1e-10, (model, metric, delta)
                checked += 1
            if predictions[predictions.model.eq(model)].output_type.eq("quantiles").all():
                prob = independent["metrics_probabilistic"]
                s = prob[prob.split.eq(row.period) & prob.variable.eq(row.variable)
                         & prob.station.eq("ALL") & prob.season.eq("ALL") & prob.lead_h.isna()]
                assert len(s) == 1
                for metric in ("pinball", "coverage80", "coverage90", "width80", "width90"):
                    delta = abs(float(s.iloc[0][metric]) - getattr(row, metric))
                    max_difference = max(max_difference, delta)
                    assert delta <= 1e-10, (model, metric, delta)
                    checked += 1
    return {"passed": True, "numeric_tolerance": 1e-10, "max_absolute_difference": max_difference,
            "independently_checked_values": checked, "method": "weather_lab.evaluate.score on persisted predictions, targets and period cases"}


def save_outputs(out: Path, predictions: pd.DataFrame, targets: pd.DataFrame, cases: pd.DataFrame,
                 provenance: dict) -> None:
    out.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    targets.to_parquet(out / "targets.parquet", index=False)
    cases.to_parquet(out / "cases.parquet", index=False)
    rows = matched_rows(predictions, targets, cases)
    rows.to_parquet(out / "verification_pairs.parquet", index=False)
    summary, leads = summaries(rows)
    summary.to_csv(out / "summary.csv", index=False)
    leads.to_csv(out / "leads.csv", index=False)
    valid = rows[rows.eligible & rows.target_observed & rows.prediction_finite]
    valid.groupby(["split", "variable", "model", "station", "season"]).agg(
        n=("err", "size"), mae=("abs_err", "mean"), mse=("sq_err", "mean"), bias=("err", "mean")).reset_index().assign(
            rmse=lambda x: np.sqrt(x.mse)).to_csv(out / "station_season.csv", index=False)
    predictions.groupby(["model", "variable", "policy"], dropna=False).size().rename("n_pairs").reset_index().to_csv(
        out / "fallbacks.csv", index=False)
    standard = score(predictions, targets, cases, {}, [6, 24, 48])
    for name, frame in standard.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    write_json(out / "verification.json", independent_rescore(out))
    write_json(out / "run.json", provenance)
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(rows=3, cols=2, subplot_titles=("MAE (°C)", "RMSE (°C)", "Forecast − observation bias (°C)",
                                                        "Paired ΔMAE vs raw (°C)", "Pinball (°C)", "80% marginal coverage"))
    for model, g in leads[leads.period.eq("development_selection") & leads.variable.eq("t")].groupby("model"):
        for i, metric in enumerate(("micro_mae", "rmse", "bias", "delta_mae_vs_raw", "pinball", "coverage80")):
            fig.add_trace(go.Scatter(x=g.lead_h, y=g.get(metric), name=model, legendgroup=model,
                                    showlegend=i == 0), row=i // 2 + 1, col=i % 2 + 1)
    fig.update_layout(title="October selection: descriptive development evidence, no confirmation", height=1000)
    fig.write_html(out / "lead_curves.html", include_plotlyjs=True)


def run_corrections(out: Path) -> None:
    started, cpu0 = time.perf_counter(), time.process_time()
    destination = out
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    protocol = yaml.safe_load(PROTOCOL.read_text())
    source = ROOT / protocol["source"]["run"]
    source_hashes = {name: sha256_file(source / name) for name in protocol["source"]["hashes"]}
    assert source_hashes == protocol["source"]["hashes"], "Frozen source differs from protocol"
    for name in ("observations", "arome"):
        item = protocol["source"][name]
        assert sha256_file(ROOT / item["file"]) == item["sha256"], name
    baseline_verification = {name: frame.to_csv(index=False).encode() == (source / f"{name}.csv").read_bytes()
                             for name, frame in score_run(source).items()}
    assert all(baseline_verification.values())
    cases = period_cases(read_json(source / "manifest.json"))
    saved = pd.read_parquet(source / "predictions.parquet")
    refs = ["persistence", "seasonal-naive-24h", "climatology", "arome-raw", "arome-biascorr",
            "timestral-209m-mv-cal", "timestral-209m-mv-cal-arome-archived"]
    frames = [saved[saved.model.isin(refs)].assign(policy="verified_saved_prediction")]
    targets = pd.read_parquet(source / "targets.parquet")
    obs = pd.read_parquet(ROOT / protocol["source"]["observations"]["file"])
    arome = pd.read_parquet(ROOT / protocol["source"]["arome"]["file"])
    raw = saved[saved.model.eq("arome-raw")].copy()
    generated = []
    benchmark = None
    for number, (origin, origin_rows) in enumerate(raw.groupby("origin")):
        small_cpu, small_wall = time.process_time(), time.perf_counter()
        past = obs[obs.time.le(origin)]
        history = causal_errors(arome, past, origin)
        for row in origin_rows.to_dict("records"):
            run = pd.Timestamp(row["arome_run"])
            for model, adjustment in [("c1-median-pooled-v1", median_adjustment(
                    history, row["station"], run, row["valid_time"], row["variable"], origin))] + [
                    (f"c2-nudge-tau{tau}", nudge(arome, past, row["station"], run, row["valid_time"],
                                                row["variable"], origin, tau)) for tau in (6, 12, 24)]:
                generated.append({**row, **adjustment, "point": row["point"] + adjustment["correction"],
                                  "model": model, "model_revision": model, "output_type": "point"})
        if number == 0:
            benchmark = {"cases": int(origin_rows.case_id.nunique()),
                         "cpu_s": time.process_time() - small_cpu,
                         "wall_s": time.perf_counter() - small_wall}
            benchmark["projected_correction_cpu_s"] = benchmark["cpu_s"] * raw.origin.nunique()
            if benchmark["projected_correction_cpu_s"] > 10 * 60:
                raise RuntimeError("Small-batch cost exceeds the correction allocation")
            print("Small-batch estimate:", json.dumps(benchmark), flush=True)
        if time.process_time() - cpu0 > 55 * 60:
            raise RuntimeError("CPU budget reserve reached before the next origin")
    frames.append(pd.DataFrame(generated))
    model = saved[saved.model.eq("timestral-209m-mv-cal")]
    frames.extend(point_blend(raw, model, alpha) for alpha in (0, .25, .5, .75, 1))
    predictions = pd.concat(frames, ignore_index=True)
    predictions["suite"] = protocol["programme"]
    provenance = {"programme": protocol["programme"], "stage": "corrections", "created_at": now_utc(),
                  "protocol_sha256": sha256_file(PROTOCOL), "source_hashes": source_hashes,
                  "baseline_rescore_byte_identical": baseline_verification, "environment": environment(),
                  "code_sha256": sha256_file(Path(__file__)), "model_loads": 0, "downloads": 0,
                  "small_batch_benchmark": benchmark,
                  "cpu_s_before_outputs": time.process_time() - cpu0, "wall_s_before_outputs": time.perf_counter() - started}
    scratch = ROOT / ".tmp/codex/practical-improvements"
    scratch.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="corrections-", dir=scratch)) / "complete"
    save_outputs(out, predictions, targets, cases, provenance)
    table = pd.read_csv(out / "summary.csv")
    selected = table[table.period.eq("development_selection") & table.variable.eq("t")].set_index("model")
    simple = ["arome-raw", "arome-biascorr", "c1-median-pooled-v1", "c2-nudge-tau6", "c2-nudge-tau12", "c2-nudge-tau24"]
    reference = min(simple, key=lambda m: (selected.loc[m, "station_balanced_mae"], simple.index(m)))
    candidates = [m for m in selected.index if m.startswith(("timestral-209", "c3-blend"))]
    winner = min(candidates, key=lambda m: (selected.loc[m, "station_balanced_mae"], m))
    improvement = 1 - selected.loc[winner, "station_balanced_mae"] / selected.loc[reference, "station_balanced_mae"]
    write_json(out / "selection.json", {"period": "development_selection", "reference": reference,
               "best_foundation_candidate": winner, "relative_improvement": improvement,
               "frozen_candidate": winner if improvement >= .02 else None,
               "decision": "freeze_for_prospective_confirmation" if improvement >= .02 else "no_improvement",
               "confirmation_result": None})
    inventory = arome.groupby(["station", "run", "available_at", "variable", "lead_h"]).agg(
        valid_time=("valid_time", "first"), observed_values=("value", "count")).reset_index()
    inventory.to_csv(out / "archive_inventory.csv", index=False)
    provenance.update(cpu_s=time.process_time() - cpu0, wall_s=time.perf_counter() - started,
                      peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    assert provenance["cpu_s"] < 60 * 60
    assert source_hashes == {name: sha256_file(source / name) for name in source_hashes}
    write_json(out / "run.json", provenance)
    destination.parent.mkdir(parents=True, exist_ok=True)
    out.rename(destination)
    out.parent.rmdir()
    print(json.dumps(read_json(destination / "selection.json"), indent=2))
    print(f"Saved {destination}; CPU {provenance['cpu_s']:.2f} s, wall {provenance['wall_s']:.2f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("corrections", "rescore"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "corrections":
        run_corrections(args.output)
    else:
        print(json.dumps(independent_rescore(args.output), indent=2))


if __name__ == "__main__":
    main()
