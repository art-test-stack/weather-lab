"""Audits of what the AROME covariate actually contains, and covariate sensitivity checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from weather_lab import arome as arome_mod
from weather_lab.common import ROOT, Config, read_json, write_json
from weather_lab.data import load_observations
from weather_lab.evaluate import build_adapters, make_batch
from weather_lab.forecast import Timestral
from weather_lab.suite import cases_from_manifest

OUT = ROOT / "experiments"


def load_suite_batch(cfg: Config, history_steps: int | None = None):
    manifest = read_json(cfg.manifest_path)
    obs, _ = load_observations(cfg)
    cases = cases_from_manifest(manifest)
    cases = cases[cases["eligible"]].reset_index(drop=True)
    arome_df, arome_meta = arome_mod.load_arome(cfg)
    steps = history_steps or cfg["forecast"]["context_days"] * 24 // cfg["forecast"]["step_hours"]
    batch, targets = make_batch(cfg, obs, cases, steps, arome_df)
    return batch, targets, arome_meta


def audit_table(cfg: Config, case_id: str, variable: str = "t") -> pd.DataFrame:
    """One row per valid time of one case: what each AROME covariate variant receives."""
    batch, targets, arome_meta = load_suite_batch(cfg)
    i = int(np.flatnonzero(batch.cases["case_id"].to_numpy() == case_id)[0])
    j = batch.variables.index(variable)
    case = batch.cases.iloc[i]
    avail = {r["run"]: r["available_at"] for r in arome_meta["runs"]}
    hist_t = batch.history_times()[i]
    rows = []
    for k, vt in enumerate(hist_t):
        run = batch.arome_history_run[i, k]
        has = not np.isnat(run)
        rid = pd.Timestamp(run, tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ") if has else None
        rows.append({
            "valid_time": pd.Timestamp(vt, tz="UTC"), "segment": "history",
            "observed": float(batch.history[i, j, k]),
            "arome_archived": float(batch.arome_history[i, j, k]),
            "arome_run": rid, "lead_h": batch.arome_history_lead_h[i, k] if has else np.nan,
            "availability": f"archive Last-Modified {avail[rid][11:16]}Z <= min(origin, valid time)" if has
            else "none: no fetched run available before both origin and valid time",
            "input_archived_mode": float(batch.arome_history[i, j, k]),
            "input_missing_mode": np.nan,
            "input_substobs_mode_v1": float(batch.history[i, j, k]),
        })
    run_future = case["arome_run"]
    origin = pd.Timestamp(case["origin"])
    for k, vt in enumerate(batch.future_times[i]):
        vt = pd.Timestamp(vt, tz="UTC")
        a = float(batch.arome[i, j, k])
        rows.append({
            "valid_time": vt, "segment": "future", "observed": float(targets[i, j, k]),
            "arome_archived": a, "arome_run": run_future,
            "lead_h": (vt - pd.Timestamp(run_future)).total_seconds() / 3600,
            "availability": f"archive Last-Modified {case['arome_available_at'][11:16]}Z <= origin "
                            f"{origin:%H:%M}Z",
            "input_archived_mode": a, "input_missing_mode": a, "input_substobs_mode_v1": a,
        })
    return pd.DataFrame(rows)


def lead_and_coverage_summary(cfg: Config) -> dict:
    """Historical vs future lead distributions, history coverage, and AROME error by lead."""
    batch, targets, _ = load_suite_batch(cfg)
    j = batch.variables.index("t")
    has = ~np.isnat(batch.arome_history_run)
    hist_leads = batch.arome_history_lead_h[has]
    fut_leads = np.array([(pd.Timestamp(vt, tz="UTC") - pd.Timestamp(r)).total_seconds() / 3600
                          for r, row in zip(batch.cases["arome_run"], batch.future_times) for vt in row])
    hist_err = np.abs(batch.arome_history[:, j][has] - batch.history[:, j][has])
    fut_err = np.abs(batch.arome[:, j] - targets[:, j]).ravel()
    fut_leads_2d = fut_leads.reshape(batch.arome[:, j].shape).ravel()

    def by_lead(leads, err):
        df = pd.DataFrame({"lead_h": leads, "abs_err": err}).dropna()
        return df.groupby("lead_h")["abs_err"].agg(["mean", "size"]).round(3).reset_index().to_dict("records")

    cov = has.mean(axis=1)
    per_origin = pd.DataFrame({"origin": batch.cases["origin"], "coverage": cov}).groupby("origin")["coverage"].mean()
    return {
        "history_steps": int(has.shape[1]),
        "history_coverage_mean": round(float(cov.mean()), 4),
        "history_coverage_by_origin": {f"{k:%Y-%m-%d}": round(float(v), 3) for k, v in per_origin.items()},
        "history_lead_h": {"min": float(hist_leads.min()), "max": float(hist_leads.max()),
                           "median": float(np.median(hist_leads)),
                           "counts": pd.Series(hist_leads).value_counts().sort_index().to_dict()},
        "future_lead_h": {"min": float(fut_leads.min()), "max": float(fut_leads.max()),
                          "median": float(np.median(fut_leads))},
        "t_abs_err_history_by_lead": by_lead(hist_leads, hist_err),
        "t_abs_err_future_by_lead": by_lead(fut_leads_2d, fut_err),
    }


# --------------------------------------------------------------------------- sensitivity (item 3)

def _variant(cfg: Config, mode: str | None) -> Timestral:
    base = next(a for a in build_adapters(cfg) if isinstance(a, Timestral) and a.arome_mode is None)
    return Timestral(f"diag-{mode}", cfg["variables"], targets=base.variables, checkpoint=base.checkpoint,
                     revision=base.revision, context_days=base.context_days, calendar=base.calendar, arome=mode,
                     arome_fields=cfg["arome"]["fields"], batch_size=base.batch_size)


def _quantiles(model: Timestral, ctx: np.ndarray, known: np.ndarray | None, h: int) -> np.ndarray:
    import torch

    from weather_lab.models import MODELS, resolve

    item = MODELS.get(resolve(model.checkpoint, model.revision))
    fc = item.forecaster
    with item.lock, torch.inference_mode():
        return fc.forecast(ctx, prediction_length=h, known_covariates=known, clip_nonnegative=False
                           ).quantiles.cpu().numpy()


def trace_rows(model: Timestral, ctx: np.ndarray, known: np.ndarray | None, h: int) -> list[dict]:
    """What the pinned upstream code does with each row: masks, key validity, scaling stats."""
    import torch
    from timestral.masks import compute_enc_key_validity
    from timestral.scaler import instance_norm_stats

    from weather_lab.forecast import load_timestral

    fc, _ = load_timestral(model.checkpoint, model.revision)
    inputs, mask = fc._as_context(ctx, None)
    kc = fc._as_covariates(known, inputs.shape[0], inputs.shape[-1] + h, "known_covariates")
    rows, rmask, roles, kf, kfm, n_tgt = fc._assemble_rows(inputs, mask, h, None, kc)
    b, v, t = rows.shape
    ps = fc.patch_size
    kf = kf.reshape(b, v, -1) if kf is not None else torch.zeros(b, v, ps)
    kfm = kfm.reshape(b, v, -1) if kfm is not None else torch.zeros(b, v, ps, dtype=torch.bool)
    full = torch.cat([rows, kf], -1)
    fmask = torch.cat([rmask, kfm], -1)
    n = full.shape[-1] // ps
    out = []
    for r in range(v):
        obs_patch = fmask[:, r].unflatten(-1, (n, ps)).any(-1)
        key_valid = compute_enc_key_validity(obs_patch, fc.config.attend_missing_patches)
        loc, scale, _ = instance_norm_stats(full[:, r], fmask[:, r], fc.config.minimum_scale)
        out.append({
            "row": r, "role": "target" if r < n_tgt else "known covariate",
            "observed_context_steps_mean": float(rmask[:, r].sum(-1).float().mean()),
            "observed_future_steps_mean": float(kfm[:, r].sum(-1).float().mean()),
            "valid_context_key_patches_mean": float(key_valid[:, : n - 1].sum(-1).float().mean()),
            "context_patches": n - 1,
            "rows_with_zero_valid_context_keys": int((key_valid[:, : n - 1].sum(-1) == 0).sum()),
            "instance_loc_mean": float(loc.mean()) if r >= n_tgt else None,
            "instance_scale_mean": float(scale.mean()) if r >= n_tgt else None,
        })
    return out


def sensitivity(cfg: Config, delta: float = 5.0, case_ids: list[str] | None = None) -> dict:
    """A / B / C and +delta (°C, future temperature covariate only) comparisons; all else fixed."""
    batch, _, _ = load_suite_batch(cfg)
    if case_ids:
        keep = np.flatnonzero(batch.cases["case_id"].isin(case_ids).to_numpy())
        batch = batch.subset(keep)
    h = batch.horizon
    mA, mB, mC = _variant(cfg, None), _variant(cfg, "missing"), _variant(cfg, "archived")
    ctx, kA = mA.build_inputs(batch)
    ctxB, kB = mB.build_inputs(batch)
    ctxC, kC = mC.build_inputs(batch)
    assert np.array_equal(ctx, ctxB) and np.array_equal(ctx, ctxC), "observations must be identical"
    n_cal = kA.shape[1]
    t_row = n_cal + mB.arome_fields.index("t")       # AROME temperature row index in the known block
    t_ctx = ctx.shape[-1]

    def perturbed(k):
        k = k.copy()
        k[:, t_row, t_ctx:] += delta                  # future segment of the temperature covariate only
        return k

    # Probe for the mechanism: B with ONE observed historical value at the first context step.
    kB1 = kB.copy()
    kB1[:, t_row, 0] = batch.arome_history[:, batch.variables.index("t"), -t_ctx]
    first_has = np.isfinite(kB1[:, t_row, 0])
    fill = np.nanmean(batch.history[:, batch.variables.index("t"), -t_ctx:], axis=1)
    kB1[~first_has, t_row, 0] = fill[~first_has]      # a single historical value, only to probe masking

    q = {"A": _quantiles(mA, ctx, kA, h), "B": _quantiles(mB, ctx, kB, h), "C": _quantiles(mC, ctx, kC, h),
         "B+d": _quantiles(mB, ctx, perturbed(kB), h), "C+d": _quantiles(mC, ctx, perturbed(kC), h),
         "B1": _quantiles(mB, ctx, kB1, h), "B1+d": _quantiles(mB, ctx, perturbed(kB1), h)}
    t_idx = mA.variables.index("t")

    def diff(a, b):
        d = np.abs(q[a] - q[b])
        return {"max_abs_all_vars": float(d.max()), "max_abs_t": float(d[:, t_idx].max()),
                "mean_abs_t_median": float(d[:, t_idx, :, 10].mean())}

    pairs = {"A vs B (future-only AROME added)": diff("A", "B"),
             "A vs C (archived history + future AROME added)": diff("A", "C"),
             "B vs C": diff("B", "C"),
             f"B vs B+{delta:g}C (future t covariate perturbed)": diff("B", "B+d"),
             f"C vs C+{delta:g}C (future t covariate perturbed)": diff("C", "C+d"),
             f"B1 vs B1+{delta:g}C (probe: one observed historical value)": diff("B1", "B1+d")}
    return {"n_cases": len(batch.cases), "delta_c": delta, "pairs": pairs,
            "trace_B": trace_rows(mB, ctx, kB, h), "trace_C": trace_rows(mC, ctx, kC, h),
            "trace_B1": trace_rows(mB, ctx, kB1, h),
            "history_coverage_C": float(np.isfinite(kC[:, t_row, :t_ctx]).mean())}


def write_report(name: str, obj: dict | pd.DataFrame) -> Path:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        p = d / "table.csv"
        obj.to_csv(p, index=False)
    else:
        p = d / "result.json"
        write_json(p, obj)
    return p


# --------------------------------------------------------------------------- forecast-only response

def normalized_known_row(fc, ctx: np.ndarray, known: np.ndarray, h: int, known_row: int) -> tuple[np.ndarray, np.ndarray]:
    """(normalized values, observed mask), each (N, T + H), of one known-covariate row exactly as the
    pinned upstream code scales it (non-causal instance mean/std over the row's observed positions)."""
    import torch
    from timestral.scaler import instance_norm_stats

    inputs, mask = fc._as_context(ctx, None)
    kc = fc._as_covariates(known, inputs.shape[0], inputs.shape[-1] + h, "known_covariates")
    rows, rmask, roles, kf, kfm, n_tgt = fc._assemble_rows(inputs, mask, h, None, kc)
    b, v, _ = rows.shape
    full = torch.cat([rows, kf.reshape(b, v, -1)], -1)[:, n_tgt + known_row]
    fmask = torch.cat([rmask, kfm.reshape(b, v, -1)], -1)[:, n_tgt + known_row]
    loc, scale, _ = instance_norm_stats(full, fmask, fc.config.minimum_scale)
    z = torch.where(fmask, (full - loc) / scale, torch.zeros_like(full))
    return z.cpu().numpy(), fmask.cpu().numpy()


def forecast_only_response(cfg: Config, sizes: tuple[str, ...] = ("Timestral-28M", "Timestral-209M"),
                           delta: float = 5.0, local_lead_h: int = 24, case_ids: list[str] | None = None) -> dict:
    """Does a checkpoint use the issued AROME temperature when its history segment is missing?

    Perturbations of the FUTURE segment of the AROME temperature row only, masks fixed:
    - reorder: the 16 issued values reversed in time (same values, same mask);
    - local: +delta at one lead only;
    - uniform: +delta at every lead (shown for reference: instance normalisation removes it).
    For each, the change of the normalised covariate actually seen by the model is reported next to the
    change of the temperature quantiles, for the forecast-only setup and, as a control, the archived one.
    """
    from weather_lab.models import CHECKPOINTS as REGISTRY, MODELS

    batch, _, _ = load_suite_batch(cfg)
    if case_ids:
        batch = batch.subset(np.flatnonzero(batch.cases["case_id"].isin(case_ids).to_numpy()))
    h = batch.horizon
    out = {"n_cases": len(batch.cases), "delta": delta, "local_lead_h": local_lead_h, "checkpoints": {}}
    for size in sizes:
        ck = REGISTRY[size]
        fc = MODELS.get(ck).forecaster
        res = {}
        for mode in ("missing", "archived"):
            m = Timestral(f"diag-{mode}", cfg["variables"], targets=cfg["variables"], checkpoint=ck.repo,
                          revision=ck.revision, context_days=cfg["forecast"]["context_days"], calendar=True,
                          arome=mode, arome_fields=cfg["arome"]["fields"])
            ctx, known = m.build_inputs(batch)
            row = known.shape[1] - len(m.arome_fields) + m.arome_fields.index("t")
            t_ctx, k = ctx.shape[-1], local_lead_h // batch.step_hours - 1
            variants = {"reorder": known.copy(), "local": known.copy(), "uniform": known.copy()}
            variants["reorder"][:, row, t_ctx:] = known[:, row, t_ctx:][:, ::-1]
            variants["local"][:, row, t_ctx + k] += delta
            variants["uniform"][:, row, t_ctx:] += delta
            z0, m0 = normalized_known_row(fc, ctx, known, h, row)
            q0 = _quantiles(m, ctx, known, h)
            ti = m.variables.index("t")
            for name, kv in variants.items():
                z1, m1 = normalized_known_row(fc, ctx, kv, h, row)
                assert np.array_equal(m0, m1), "perturbation changed the observation mask"
                q1 = _quantiles(m, ctx, kv, h)
                d = np.abs(q1[:, ti] - q0[:, ti])                      # (N, H, Q)
                res[f"{mode}/{name}"] = {
                    "normalized_input_max_abs_change": float(np.abs(z1 - z0).max()),
                    "t_quantiles_max_abs_change_c": float(d.max()),
                    "t_median_mean_abs_change_c": float(d[..., 10].mean()),
                    "t_median_mean_abs_change_at_lead_c": float(d[:, k, 10].mean()),
                    "other_variables_max_abs_change": float(np.abs(np.delete(q1 - q0, ti, axis=1)).max()),
                }
        out["checkpoints"][size] = {"revision": ck.revision, "responses": res}
    return out
