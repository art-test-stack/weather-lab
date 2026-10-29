"""Error through the forecast window, recomputed from saved predictions (no model calls).

Uses the scorer's own row losses (`scoring._row_scores`) and common valid set (`scoring.common`), so
"at lead H" values equal the saved per-lead score tables. Two views:

- **at lead H**: mean over the matched (case) pairs at that lead only;
- **cumulative to H**: mean over every matched (case, lead) pair with lead <= H (RMSE: root of that
  mean square). It is a different quantity from the value at H and is labelled as such.

Leads of one forecast are not independent samples. A 95 % interval of the mean is given only when the
cases span at least `MIN_BLOCKS_FOR_CI` temporal blocks (block bootstrap over origin days, as for the
paired differences); otherwise none is shown. These intervals describe uncertainty of an aggregate
score, not a forecast range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from weather_lab.scoring import MIN_BLOCKS_FOR_CI, _row_scores, common

METRIC_COLS = {"mae": "abs_err", "rmse": "sq_err", "bias": "err", "pinball": "pinball", "coverage80": "in80",
               "coverage90": "in90", "width80": "width80", "width90": "width90"}
PROBABILISTIC = {"pinball", "coverage80", "coverage90", "width80", "width90"}


def scored_rows(preds: pd.DataFrame, targets: pd.DataFrame, cases: pd.DataFrame, variable: str,
                models: list[str], metric: str, station: str = "ALL", split: str | None = None) -> pd.DataFrame:
    """Row losses of `models` on their common valid set (the systems compared, matched case by case)."""
    p = preds[(preds["variable"] == variable) & preds["model"].isin(models)]
    if metric in PROBABILISTIC:
        p = p[p["output_type"] == "quantiles"]
    if station != "ALL":
        p = p[p["station"] == station]
    meta = cases[["case_id", "split", "origin"]]
    df = p.drop(columns=["origin"]).merge(targets[["case_id", "variable", "lead_h", "observed"]],
                                          on=["case_id", "variable", "lead_h"], how="left").merge(meta, on="case_id")
    if split:
        df = df[df["split"] == split]
    df = _row_scores(df)
    return common(df, "pinball" if metric in PROBABILISTIC else "point")


def _finish(metric: str, x: float) -> float:
    return float(np.sqrt(x)) if metric == "rmse" else float(x)


def _block_ci(g: pd.DataFrame, col: str, metric: str, n_boot: int = 1000, block_days: int = 2,
              seed: int = 0) -> tuple[float, float, int]:
    days = g["origin"].dt.floor("D")
    order = {d: i // block_days for i, d in enumerate(np.sort(days.unique()))}
    blocks = g[col].groupby(days.map(order)).agg(["sum", "count"])
    nb = len(blocks)
    if nb < MIN_BLOCKS_FOR_CI:
        return np.nan, np.nan, nb
    idx = np.random.default_rng(seed).integers(0, nb, size=(n_boot, nb))
    means = blocks["sum"].to_numpy()[idx].sum(1) / blocks["count"].to_numpy()[idx].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return _finish(metric, lo), _finish(metric, hi), nb


def by_lead(rows: pd.DataFrame, metric: str, cumulative: bool = False, intervals: bool = True) -> pd.DataFrame:
    """One row per (model, lead): value, pairs used, and (when blocks allow) a 95 % interval of the mean."""
    col = METRIC_COLS[metric]
    out = []
    for model, g in rows.groupby("model", sort=False):
        for lead in sorted(g["lead_h"].unique()):
            sel = g[g["lead_h"] <= lead] if cumulative else g[g["lead_h"] == lead]
            lo, hi, nb = _block_ci(sel, col, metric) if intervals else (np.nan, np.nan, 0)
            out.append({"model": model, "lead_h": int(lead), metric: _finish(metric, sel[col].mean()),
                        "n": len(sel), "n_cases": sel["case_id"].nunique(), "ci95_low": lo, "ci95_high": hi,
                        "n_blocks": nb})
    return pd.DataFrame(out)


def lead_ranges(table: pd.DataFrame, metric: str, model: str, reference: str) -> list[tuple[int, int]]:
    """Contiguous lead ranges where `model` scores better than `reference` (descriptive, no test)."""
    piv = table.pivot(index="lead_h", columns="model", values=metric)
    if model not in piv or reference not in piv:
        return []
    better = (piv[model] - 0.8).abs() < (piv[reference] - 0.8).abs() if metric.startswith("coverage") \
        else piv[model] < piv[reference]
    ranges, start = [], None
    leads = list(piv.index)
    for i, lead in enumerate(leads):
        if better.loc[lead] and start is None:
            start = lead
        if start is not None and (not better.loc[lead] or i == len(leads) - 1):
            ranges.append((start, lead if better.loc[lead] else leads[i - 1]))
            start = None
    return ranges
