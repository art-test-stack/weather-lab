import numpy as np
import pandas as pd
import pytest

from weather_lab.forecast import QUANTILE_LEVELS, qcol
from weather_lab.practical_distribution import apply_distribution, fit_distribution


def fixture():
    rows = []
    for day in range(10, 14):
        origin = pd.Timestamp(f"2026-10-{day}T06:00Z")
        for station in ("07110", "07149", "07190", "07630", "07690"):
            rows.append(dict(case_id=f"{station}/{day}", station=station, origin=origin,
                             valid_time=origin + pd.Timedelta(hours=3), lead_h=3, variable="t", point=-2.,
                             arome_run=origin - pd.Timedelta(hours=3),
                             arome_available_at=origin - pd.Timedelta(hours=1),
                             observation_available_at=origin + pd.Timedelta(hours=3),
                             observed=float(day - 10)))
    return pd.DataFrame(rows), dict(warmup_start=pd.Timestamp("2026-10-10T06:00Z"),
                                    warmup_end=pd.Timestamp("2026-10-13T06:00Z"),
                                    cutoff=pd.Timestamp("2026-10-16T06:00Z"), source_identity={"source": "fixture"})


def future(rows):
    out = rows.iloc[:5].copy()
    for name in ("origin", "valid_time", "arome_run", "arome_available_at"):
        out[name] += pd.Timedelta(days=6)
    return out


def test_fit_is_pooled_exact_actual_lead_and_frozen_before_selection():
    rows, args = fixture()
    fit = fit_distribution(rows, **args)
    group = fit.groups.iloc[0]
    assert group.support == 20 and group.station_count == 5
    assert group.actual_run_lead_h == 6 and group.run_cycle_utc == 3
    assert group[qcol(.5)] == 3.5
    out = apply_distribution(future(rows), fit)
    assert out.point.eq(1.5).all()
    q = out[[qcol(x) for x in QUANTILE_LEVELS]].to_numpy()
    assert (np.diff(q, axis=1) >= 0).all() and (q[:, 0] < q[:, -1]).all()
    assert apply_distribution(future(rows).assign(variable="u"), fit).point.isna().all()
    wrong_lead = future(rows).assign(valid_time=lambda x: x.valid_time + pd.Timedelta(hours=3), lead_h=6)
    assert apply_distribution(wrong_lead, fit).point.isna().all()
    assert fit.identity != fit_distribution(rows.assign(observed=rows.observed + 1), **args).identity
    assert fit.identity != fit_distribution(rows, **{**args, "source_identity": {"source": "changed"}}).identity


def test_purging_late_labels_and_fit_application_cutoffs():
    rows, args = fixture()
    with pytest.raises(ValueError, match="boundary"):
        fit_distribution(rows.assign(valid_time=args["cutoff"], lead_h=lambda x:
                                      (x.valid_time-x.origin).dt.total_seconds()/3600), **args)
    with pytest.raises(ValueError, match="boundary"):
        fit_distribution(rows, **{**args, "warmup_end": args["warmup_end"] - pd.Timedelta(days=1)})
    fit = fit_distribution(rows, **args)
    with pytest.raises(ValueError, match="warm-up"):
        apply_distribution(rows, fit)
    late = fit_distribution(rows.assign(observation_available_at=args["cutoff"]), **args)
    assert apply_distribution(future(rows), late).point.isna().all()
    with pytest.raises(ValueError, match="availability"):
        fit_distribution(rows.assign(arome_available_at=rows.origin + pd.Timedelta(hours=1)), **args)


def test_insufficient_support_leaves_distribution_unavailable_without_fake_bands():
    rows, args = fixture()
    fit = fit_distribution(rows.iloc[1:], **args)
    assert fit.groups.support.iloc[0] == 19
    out = apply_distribution(future(rows), fit)
    assert out.point.isna().all()
    assert out[[qcol(q) for q in QUANTILE_LEVELS]].isna().all().all()


def test_persisted_predictions_rescore_independently(tmp_path):
    from weather_lab.common import read_json
    from weather_lab.practical import save_outputs

    rows, args = fixture()
    fit = fit_distribution(rows, **args)
    raw = future(rows).assign(model="arome-raw", model_revision="fixture", output_type="point", unit="°C", policy="raw")
    targets = raw[["case_id", "variable", "lead_h", "valid_time", "observed"]]
    raw = raw.drop(columns=["observed", "observation_available_at"])
    distribution = apply_distribution(raw, fit)
    climatology = distribution.assign(model="climatology")
    predictions = pd.concat([raw, distribution, climatology], ignore_index=True)
    cases = raw[["case_id", "origin"]].assign(split="development_selection", season="autumn", eligible=True)
    out = tmp_path / "scored"
    save_outputs(out, predictions, targets, cases, {"fixture": True})
    verification = read_json(out / "verification.json")
    assert verification["passed"] and verification["independently_checked_values"] > 0
    assert verification["max_absolute_difference"] <= 1e-10
