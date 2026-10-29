"""New information rows preserve causal roles, units, masks and native semantics."""
from dataclasses import replace
import importlib.util

import numpy as np
import pandas as pd
import pytest

from weather_lab.common import ROOT
from weather_lab.forecast import CaseBatch
from weather_lab.models import CHECKPOINTS


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/error-features-v1" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extra = load("extra_information", "adapters.py")
collector = load("issued_information_collector", "collect.py")


def fixture():
    origins = np.array(["2026-10-21T06"], dtype="datetime64[ns]")
    times = origins[:, None] + np.arange(-15, 1) * np.timedelta64(3, "h")
    history = np.stack([np.arange(16.) + 10, np.arange(16.) + 1000,
                        np.linspace(-3, 3, 16), np.linspace(2, -2, 16)])[None].astype(np.float32)
    history[0, 2, 0] = np.nan
    runs = times - np.timedelta64(6, "h")
    cases = pd.DataFrame({"case_id": ["c"], "station": ["07149"], "arome_run": ["2026-10-21T03:00Z"]})
    batch = CaseBatch(cases, ["t", "pmer", "u", "v"], history, origins,
        origins[:, None] + np.arange(1, 3) * np.timedelta64(3, "h"), 3,
        arome=np.array([[[10., 11.], [1000., 1001.], [0., 0.], [0., 0.]]], np.float32),
        arome_history=history + 1, arome_history_run=runs, arome_history_lead_h=np.full((1, 16), 6.),
        aux_history=np.array([[[2.] * 16, [50.] * 16]], np.float32), aux_variables=["td", "cloud"],
        station_coords={"07149": (48.7, 2.4)})
    issued = pd.DataFrame([{"station": "07149", "run": pd.Timestamp("2026-10-21T03:00Z"),
        "available_at": pd.Timestamp("2026-10-21T06:00Z"), "valid_time": pd.Timestamp(t, tz="UTC"),
        "variable": field, "value": value, "product": "meteofrance_arome_france"}
        for t in batch.future_times[0] for field, value in zip(extra.FIELDS, (60., 75., 100.))])
    return batch, issued


def test_native_translations_keep_wind_past_and_issued_weather_future_only():
    batch, issued = fixture()
    snapshot = batch.history.copy()
    extras = extra.ExtraInputs(True, issued)
    for family, cls in (("Timestral-209M", extra.ExtraTimestral), ("t0-gamma", extra.ExtraT0)):
        ck = CHECKPOINTS[family]
        kwargs = dict(context_days=2, calendar=False, arome="missing", arome_fields=["t", "pmer"],
                      features=["resid", "atmos", "solar", "tend"], extras=extras, extra_identity="fixture")
        if family.startswith("Timestral"):
            kwargs["prediction_target"] = "arome_error"
        model = cls("fixture", batch.variables, ["t", "pmer"], ck.repo, ck.revision, **kwargs)
        built = model.build_inputs(batch)
        context, known = built[:2]
        np.testing.assert_array_equal(context[:, :2], batch.history[:, :2] - batch.arome_history[:, :2])
        weather = known[:, -4:-1] if family.startswith("Timestral") else known[:, -3:]
        np.testing.assert_array_equal(weather[:, :, -2:], np.array([[[60, 60], [75, 75], [100, 100]]]))
        assert np.isnan(weather[:, :, :16]).all()
        wind = model.feature_rows(batch)[0][:, -2:] if family.startswith("Timestral") else context[:, -2:]
        np.testing.assert_array_equal(wind, batch.history[:, 2:])
        changed = batch.history.copy()
        changed[0, 2, -1] += 1
        altered = replace(batch, history=changed)
        if family == "t0-gamma":
            assert model.input_identity(batch) != model.input_identity(altered)
        else:
            assert not np.array_equal(model.feature_rows(batch)[0], model.feature_rows(altered)[0], equal_nan=True)
        source_changed = issued.copy()
        source_changed.loc[source_changed.variable.eq("cloud_cover"), "value"] += 1
        old_known = model.build_inputs(batch)[1].copy()
        old_key = model.input_identity(batch) if family == "t0-gamma" else None
        model.extras = extra.ExtraInputs(True, source_changed)
        assert not np.array_equal(old_known, model.build_inputs(batch)[1], equal_nan=True)
        if family == "t0-gamma":
            assert old_key != model.input_identity(batch)
    np.testing.assert_array_equal(batch.history, snapshot)


def test_future_fields_refuse_late_missing_duplicate_or_other_product():
    batch, issued = fixture()
    for source in (issued.assign(available_at=pd.Timestamp("2026-10-21T07:00Z")),
                   issued.iloc[:-1], pd.concat([issued, issued.iloc[:1]]), issued.assign(product="other")):
        with pytest.raises(ValueError):
            extra.ExtraInputs(False, source).rows(batch, 16)
    changed = batch.history.copy()
    changed[:, 2] = np.nan
    with pytest.raises(ValueError, match="entirely missing"):
        extra.ExtraInputs(True, None).rows(replace(batch, history=changed), 16)


def test_collector_validates_units_grid_time_and_preserves_missing_lead_zero():
    run = pd.Timestamp("2026-10-21T03:00Z")
    times = pd.date_range(run, periods=52, freq="h")
    fields = {"temperature_2m": [10.] * 52, "pressure_msl": [1000.] * 52,
              "relative_humidity_2m": [60.] * 52, "cloud_cover": [None] + [75.] * 51,
              "shortwave_radiation": [None] + [100.] * 51}
    response = [{"location_id": 7149, "latitude": 48.7, "longitude": 2.4, "utc_offset_seconds": 0,
                 "hourly_units": {"temperature_2m": "°C", "pressure_msl": "hPa", **collector.FIELDS},
                 "hourly": {"time": times.strftime("%Y-%m-%dT%H:%M").tolist(), **fields}}]
    anchors = pd.DataFrame([{"station": "07149", "run": run, "valid_time": t, "variable": v,
        "value": value, "grid_lat": 48.7, "grid_lon": 2.4}
        for t in times[6::3] for v, value in (("t", 10.), ("pmer", 1000.))])
    rows, _ = collector.validate(response, run, {"07149": {}}, anchors)
    frame = pd.DataFrame(rows)
    assert frame.value.isna().sum() == 2
    assert frame.available_at.eq(run + pd.Timedelta(hours=3)).all()
    response[0]["hourly_units"]["shortwave_radiation"] = "J/m²"
    with pytest.raises(ValueError, match="units"):
        collector.validate(response, run, {"07149": {}}, anchors)
    response[0]["hourly_units"]["shortwave_radiation"] = "W/m²"
    response[0]["latitude"] += .025
    with pytest.raises(ValueError, match="grid"):
        collector.validate(response, run, {"07149": {}}, anchors)
