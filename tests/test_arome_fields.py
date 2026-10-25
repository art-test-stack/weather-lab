"""Surface field identification and exact forecast-step values, without weather downloads."""

import eccodes
import numpy as np
import pandas as pd
import pytest

from weather_lab.arome_fields import FIELDS, decode_station_records, field_for_header, parse_field_header, step_values


def test_statistical_headers_use_interval_ends_and_do_not_match_instantaneous_fields():
    for name, statistic in (("gust", 2), ("precip", 1), ("shortwave", 1), ("t", 3)):
        handle = eccodes.codes_grib_new_from_samples("GRIB2")
        try:
            field = FIELDS[name]
            eccodes.codes_set(handle, "productDefinitionTemplateNumber", 8)
            for key, value in zip(("discipline", "parameterCategory", "parameterNumber", "typeOfFirstFixedSurface",
                                   "scaledValueOfFirstFixedSurface"), field.key):
                eccodes.codes_set(handle, key, value)
            eccodes.codes_set(handle, "typeOfStatisticalProcessing", statistic)
            eccodes.codes_set(handle, "startStep", 3)
            eccodes.codes_set(handle, "endStep", 6)
            info = parse_field_header(eccodes.codes_get_message(handle))
        finally:
            eccodes.codes_release(handle)
        assert (info["start_h"], info["end_h"]) == (3, 6)
        assert field_for_header(info) == (None if name == "t" else name)


def test_station_decoding_respects_bitmap_and_source_run(tmp_path):
    path = tmp_path / "surface.grib2"
    with path.open("wb") as stream:
        for name in ("t", "precip"):
            handle = eccodes.codes_grib_new_from_samples("GRIB2")
            try:
                eccodes.codes_set(handle, "dataDate", 20261012)
                eccodes.codes_set(handle, "dataTime", 300)
                for key, value in zip(("discipline", "parameterCategory", "parameterNumber",
                                       "typeOfFirstFixedSurface", "scaledValueOfFirstFixedSurface"), FIELDS[name].key):
                    eccodes.codes_set(handle, key, value)
                if name == "precip":
                    eccodes.codes_set(handle, "productDefinitionTemplateNumber", 8)
                    eccodes.codes_set(handle, "typeOfStatisticalProcessing", 1)
                    eccodes.codes_set(handle, "startStep", 0)
                    eccodes.codes_set(handle, "endStep", 3)
                else:
                    eccodes.codes_set(handle, "forecastTime", 3)
                eccodes.codes_set(handle, "bitmapPresent", 1)
                eccodes.codes_set(handle, "missingValue", 9999.)
                values = np.full(eccodes.codes_get(handle, "numberOfDataPoints"), 283.15 if name == "t" else 2.)
                values[0] = 9999.
                eccodes.codes_set_values(handle, values)
                eccodes.codes_write(handle, stream)
            finally:
                eccodes.codes_release(handle)
    stations = [{"id": "missing", "lat": 60., "lon": 0.}, {"id": "present", "lat": 60., "lon": 2.}]
    records = decode_station_records(path, stations, "2026-10-12T03:00Z", AVAILABLE)
    assert records[records["station"] == "missing"]["value"].isna().all()
    values = step_values(records, [3]).set_index(["station", "variable"])["value"]
    assert values["present", "t"] == pytest.approx(10.) and values["present", "precip"] == 2.
    with pytest.raises(ValueError, match="run differs"):
        decode_station_records(path, stations, "2026-10-11T03:00Z", AVAILABLE)


def _records(rows):
    return pd.DataFrame([{"run": "2026-10-12T03:00Z", "station": "07149", "variable": name,
                          "start_h": start, "end_h": end, "value": value, "available_at": available}
                         for name, start, end, value, available in rows])


AVAILABLE = "2026-10-12T05:45Z"


def test_units_accumulation_differences_and_complete_gust_windows():
    frame = _records([
        ("t", 3, 3, 283.15, AVAILABLE), ("pmer", 3, 3, 100_000., AVAILABLE),
        ("td", 3, 3, 278.15, AVAILABLE), ("rh", 3, 3, 75., AVAILABLE),
        ("precip", 0, 3, 1., AVAILABLE), ("precip", 0, 6, 4., AVAILABLE),
        ("snowfall", 0, 3, .5, AVAILABLE), ("snowfall", 0, 6, .75, AVAILABLE),
        ("shortwave", 0, 3, 108_000., AVAILABLE), ("shortwave", 0, 6, 324_000., AVAILABLE),
        ("gust", 0, 1, 5., AVAILABLE), ("gust", 1, 2, 8., AVAILABLE), ("gust", 2, 3, 6., AVAILABLE),
    ])
    values = step_values(frame, [3, 6]).set_index(["variable", "lead_h"])["value"]
    assert values["t", 3] == pytest.approx(10.) and values["td", 3] == pytest.approx(5.)
    assert values["pmer", 3] == 1000. and values["rh", 3] == 75.
    assert values["precip", 3] == 1. and values["precip", 6] == 3.
    assert values["snowfall", 3] == .5 and values["snowfall", 6] == .25
    assert values["shortwave", 3] == 10. and values["shortwave", 6] == 20.
    assert values["gust", 3] == 8. and np.isnan(values["gust", 6])


def test_missing_intervals_stay_missing_and_publication_covers_every_used_boundary():
    frame = _records([
        ("precip", 0, 6, 4., AVAILABLE), ("shortwave", 0, 6, 324_000., AVAILABLE),
        ("gust", 0, 1, 5., AVAILABLE), ("gust", 2, 3, 6., AVAILABLE),
    ])
    assert step_values(frame, [3, 6])["value"].isna().all()
    frame = _records([("precip", 0, 3, 1., "2026-10-12T06:30Z"), ("precip", 0, 6, 4., AVAILABLE)])
    result = step_values(frame, [6]).iloc[0]
    assert result["value"] == 3. and result["available_at"] == pd.Timestamp("2026-10-12T06:30Z")
    with pytest.raises(ValueError, match="negative"):
        step_values(_records([("precip", 0, 3, 4., AVAILABLE), ("precip", 0, 6, 1., AVAILABLE)]), [6])
