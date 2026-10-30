"""Numerical scoring on explicit common supports; no run discovery, model execution or storage.

Scoring conventions (also in docs/experiment_protocol.md):
- point errors use each system's point forecast; for quantile systems that is the median;
- pinball loss: mean over the returned levels of max(tau (y - q), (tau - 1)(y - q)), physical units;
- 80 % / 90 % central intervals: [q0.10, q0.90] / [q0.05, q0.95]; deterministic systems get none;
- every comparison table is computed on the common valid set: the (case, lead) pairs of a
  variable where the target is observed and every compared system produced a finite forecast;
- scores are never pooled across splits, suites or variables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from weather_lab.forecast import QUANTILE_LEVELS, qcol
from weather_lab.suite import SEASONS

POINT_REFERENCE = "persistence"
PROB_REFERENCE = "climatology"
INTERVALS = {80: (0.1, 0.9), 90: (0.05, 0.95)}
MIN_BLOCKS_FOR_CI = 10


def pinball(y: np.ndarray, q: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Mean pinball loss over levels; y (N,), q (N, Q) -> (N,)."""
    d = y[:, None] - q
    return np.mean(np.maximum(levels * d, (levels - 1) * d), axis=1)


def _row_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row losses for one variable; df has observed, point and (maybe) quantile columns."""
    out = df.copy()
    err = out["point"] - out["observed"]
    out["abs_err"] = err.abs()
    out["sq_err"] = err ** 2
    out["err"] = err
    qcols = [qcol(l) for l in QUANTILE_LEVELS]
    has_q = out["output_type"].eq("quantiles").to_numpy()
    out["pinball"] = np.nan
    if not has_q.any():
        for nominal in INTERVALS:
            out[f"in{nominal}"] = np.nan
            out[f"width{nominal}"] = np.nan
    if has_q.any():
        sub = out.loc[has_q]
        out.loc[has_q, "pinball"] = pinball(sub["observed"].to_numpy(), sub[qcols].to_numpy(), np.array(QUANTILE_LEVELS))
        for nominal, (lo, hi) in INTERVALS.items():
            l, u = sub[qcol(lo)].to_numpy(), sub[qcol(hi)].to_numpy()
            y = sub["observed"].to_numpy()
            out.loc[has_q, f"in{nominal}"] = ((y >= l) & (y <= u)).astype(float)
            out.loc[has_q, f"width{nominal}"] = u - l
    return out


def _cube(df: pd.DataFrame, metrics: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Aggregate over split x model x variable x {station, ALL} x {season, ALL} x {lead, ALL}."""
    frames = []
    for st in (True, False):
        for se in (True, False):
            for ld in (True, False):
                keys = ["split", "model", "variable"] + (["station"] if st else []) + (["season"] if se else []) \
                    + (["lead_h"] if ld else [])
                g = df.groupby(keys, observed=True).agg(**metrics, n=("observed", "size")).reset_index()
                if not st:
                    g["station"] = "ALL"
                if not se:
                    g["season"] = "ALL"
                if not ld:
                    g["lead_h"] = pd.NA
                frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    out["lead_h"] = out["lead_h"].astype("Int64")
    return out


def _add_skill(table: pd.DataFrame, metric: str, ref: str) -> pd.DataFrame:
    keys = ["split", "variable", "station", "season", "lead_h"]
    r = table[table["model"] == ref][keys + [metric]].rename(columns={metric: f"{metric}_ref"})
    out = table.merge(r, on=keys, how="left")
    out[f"skill_{metric}_vs_{ref}"] = 1 - out[metric] / out[f"{metric}_ref"]
    return out


def _bootstrap(df: pd.DataFrame, loss: str, ref: str, highlight: list[int], n_boot: int = 2000,
               block_days: int = 2, seed: int = 0) -> pd.DataFrame:
    """Paired mean loss difference (model - ref) with a block bootstrap over origin dates.

    Blocks group consecutive origin days (`block_days`, matching the 48 h horizon overlap of daily
    origins); all stations of a day share a block. A CI is reported only with >= MIN_BLOCKS_FOR_CI blocks.
    """
    rng = np.random.default_rng(seed)
    rows = []
    df = df.assign(day=df["origin"].dt.floor("D"))
    for (split, var), g in df.groupby(["split", "variable"]):
        days = np.sort(g["day"].unique())
        block_of = {d: i // block_days for i, d in enumerate(days)}
        piv = g.pivot_table(index=["case_id", "lead_h", "day"], columns="model", values=loss)
        if ref not in piv:
            continue
        for model in [m for m in piv.columns if m != ref]:
            for lead in [None] + highlight:
                p = piv if lead is None else piv[piv.index.get_level_values("lead_h") == lead]
                d = (p[model] - p[ref]).dropna()
                if d.empty:
                    continue
                blocks = d.groupby(d.index.get_level_values("day").map(block_of)).agg(["sum", "count"])
                nb = len(blocks)
                rec = {"split": split, "variable": var, "model": model, "reference": ref, "loss": loss,
                       "lead_h": lead, "n_pairs": len(d), "n_blocks": nb, "mean_diff": d.mean(),
                       "ci95_low": np.nan, "ci95_high": np.nan}
                if nb >= MIN_BLOCKS_FOR_CI:
                    idx = rng.integers(0, nb, size=(n_boot, nb))
                    s, c = blocks["sum"].to_numpy()[idx].sum(1), blocks["count"].to_numpy()[idx].sum(1)
                    rec["ci95_low"], rec["ci95_high"] = np.percentile(s / c, [2.5, 97.5])
                rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out):
        out["lead_h"] = out["lead_h"].astype("Int64")
    return out


def score(preds: pd.DataFrame, targets: pd.DataFrame, cases: pd.DataFrame, thresholds: dict,
          highlight: list[int], pairs: list[list[str]] | None = None) -> dict[str, pd.DataFrame]:
    meta = cases[["case_id", "split", "season", "origin"]]
    df = preds.drop(columns=["origin"]).merge(targets[["case_id", "variable", "lead_h", "observed"]],
                                              on=["case_id", "variable", "lead_h"], how="left")
    df = df.merge(meta, on="case_id", how="left")
    df = _row_scores(df)

    # availability: expected = all manifest cases x leads (ineligible cases count as unavailable)
    n_leads = targets["lead_h"].nunique()
    avail = []
    for (split, model, var), g in df.groupby(["split", "model", "variable"]):
        n_all = int((cases["split"] == split).sum())
        n_elig = int(((cases["split"] == split) & cases["eligible"]).sum())
        avail.append({"split": split, "model": model, "variable": var, "output_type": g["output_type"].iloc[0],
                      "cases_in_suite": n_all, "cases_eligible": n_elig, "cases_ineligible": n_all - n_elig,
                      "expected_targets": n_all * n_leads, "eligible_targets": n_elig * n_leads,
                      "observed_targets": int(g["observed"].notna().sum()),
                      "predicted": int(g["point"].notna().sum()),
                      "failed_predictions": int(g["point"].isna().sum()),
                      "valid_pairs": int((g["point"].notna() & g["observed"].notna()).sum())})
    avail = pd.DataFrame(avail)

    point_rows = common(df, "point")
    point = _cube(point_rows, {"mae": ("abs_err", "mean"), "rmse": ("sq_err", "mean"), "bias": ("err", "mean")})
    point["rmse"] = np.sqrt(point["rmse"])
    point = _add_skill(point, "mae", POINT_REFERENCE)
    point = point.merge(avail.groupby(["split", "variable"])["eligible_targets"].first().reset_index(),
                        on=["split", "variable"], how="left")

    prob_rows = common(df[df["output_type"] == "quantiles"], "pinball")
    prob = _cube(prob_rows, {"pinball": ("pinball", "mean"), "coverage80": ("in80", "mean"),
                             "coverage90": ("in90", "mean"), "width80": ("width80", "mean"),
                             "width90": ("width90", "mean")})
    prob = _add_skill(prob, "pinball", PROB_REFERENCE)
    na = avail[avail["output_type"] == "point"][["split", "model", "variable"]].assign(
        note="not applicable: deterministic point forecast (no predictive distribution)")

    paired = pd.concat([_bootstrap(point_rows, "abs_err", POINT_REFERENCE, highlight),
                        _bootstrap(prob_rows, "pinball", PROB_REFERENCE, highlight)], ignore_index=True)

    t = point_rows[point_rows["variable"] == "t"].copy()
    t["threshold"] = np.array([thresholds.get(f"{s}|{SEASONS[v.month]}", np.nan)
                               for s, v in zip(t["station"], t["valid_time"])], dtype=float)
    tail = t[t["observed"] > t["threshold"]]
    matched = _matched(point_rows, prob_rows, pairs or [], highlight)
    tail_tab = tail.groupby(["split", "model"]).agg(n=("observed", "size"), mae=("abs_err", "mean"),
                                                    bias=("err", "mean"), pinball=("pinball", "mean")).reset_index()
    tail_tab["definition"] = "obs t > station x season P90 of fitting period (warm tail, not a heatwave definition)"

    # per-case summary for the UI's case browser
    case_tab = point_rows.groupby(["case_id", "station", "split", "season", "variable", "model"]).agg(
        mae=("abs_err", "mean"), n=("abs_err", "size")).reset_index()
    return {"metrics_point": point, "metrics_probabilistic": prob, "probabilistic_not_applicable": na,
            "availability": avail, "paired_differences": paired, "tail_warm_t": tail_tab, "case_scores": case_tab,
            **({"paired_matched": matched} if pairs else {})}


def common(sub: pd.DataFrame, col: str) -> pd.DataFrame:
    """Common valid set per (split, variable): pairs where the target is observed and every system in
    `sub` produced a finite value of `col`."""
    ok = sub[col].notna() & sub["observed"].notna()
    n_models = sub.groupby(["split", "variable"])["model"].transform("nunique")
    good = sub[ok].groupby(["split", "variable", "case_id", "lead_h"])["model"].transform("nunique")
    return sub[ok][good == n_models[ok]]


def _matched(point_rows: pd.DataFrame, prob_rows: pd.DataFrame, pairs: list[list[str]],
             highlight: list[int]) -> pd.DataFrame:
    """Paired differences for declared (model, reference) pairs on the run's common valid sets."""
    out = []
    for model, ref in pairs:
        for rows, loss in ((point_rows, "abs_err"), (prob_rows, "pinball")):
            sub = rows[rows["model"].isin([model, ref])]
            if sub["model"].nunique() == 2:
                out.append(_bootstrap(sub, loss, ref, highlight))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
