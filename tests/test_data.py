"""Unit conversions, wind convention and identifier preservation.

Inputs here are SYNTHETIC fixtures written to exercise code paths; they are not scientific evidence.
"""

import gzip

import numpy as np
import pandas as pd
import pytest

from weather_lab.data import RAW_COLUMNS, clean, read_year, regularise, wind_components


def test_wind_components_meteorological_convention():
    # westerly (from 270 deg) blows towards the east: u > 0, v = 0
    u, v = wind_components(np.array([10.0]), np.array([270.0]))
    assert u[0] == pytest.approx(10.0) and v[0] == pytest.approx(0.0, abs=1e-9)
    # northerly (from 360 deg) blows towards the south: v < 0
    u, v = wind_components(np.array([5.0]), np.array([360.0]))
    assert u[0] == pytest.approx(0.0, abs=1e-9) and v[0] == pytest.approx(-5.0)
    # south-westerly 225 deg: both components positive and equal
    u, v = wind_components(np.array([2.0]), np.array([225.0]))
    assert u[0] == pytest.approx(np.sqrt(2)) and v[0] == pytest.approx(np.sqrt(2))


def test_wind_calm_and_invalid_direction():
    u, v = wind_components(np.array([0.0, 4.0, 4.0]), np.array([0.0, 0.0, 990.0]))
    assert (u[0], v[0]) == (0.0, 0.0)            # calm
    assert np.isnan(u[1]) and np.isnan(v[1])     # dd=0 with wind: inconsistent -> missing
    assert np.isnan(u[2]) and np.isnan(v[2])     # out-of-range direction code -> missing


def _raw(rows):
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    df["geo_id_wmo"] = df["geo_id_wmo"].astype(str)
    return df


def test_clean_units_and_duplicates():
    raw = _raw([
        ["07149", "ORLY", 48.7, 2.4, "2024-01-01T00:10:00Z", "2024-01-01T00:00:00Z", 101325, 100000, 270, 3.0, 273.15],
        # duplicate validity time, inserted later with a corrected temperature: the later insert wins
        ["07149", "ORLY", 48.7, 2.4, "2024-01-01T01:00:00Z", "2024-01-01T00:00:00Z", 101325, 100000, 270, 3.0, 274.15],
        ["07149", "ORLY", 48.7, 2.4, None, "2024-01-01T03:00:00Z", None, 100000, None, None, 300.0],
        ["07149", "ORLY", 48.7, 2.4, None, "2024-01-01T04:00:00Z", 101000, 100000, 90, 1.0, 280.0],  # off grid
    ])
    out, report = clean(raw)
    assert report["duplicates_dropped"] == 1 and report["off_grid_rows_dropped"] == 1
    first = out.iloc[0]
    assert first["t"] == pytest.approx(1.0)                 # 274.15 K -> 1.0 °C (later insert kept)
    assert first["pmer"] == pytest.approx(1013.25)          # Pa -> hPa, sea-level pressure (not `pres`)
    assert first["u"] == pytest.approx(3.0)
    second = out.iloc[1]
    assert np.isnan(second["pmer"]) and np.isnan(second["u"])  # empty fields stay missing
    assert second["t"] == pytest.approx(26.85)


def test_station_ids_keep_leading_zeros(tmp_path):
    header = ";".join(["lat", "lon", "geo_id_wmo", "geo_id_wigos", "name", "reference_time", "insert_time",
                       "validity_time", "pmer", "pres", "dd", "ff", "t"])
    lines = [header,
             "48.7;2.4;07149;0-20000-0-07149;ORLY;;;2024-01-01T00:00:00Z;101000;100000;180;2.0;280.0",
             "43.6;1.4;07630;0-20000-0-07630;TOULOUSE;;;2024-01-01T00:00:00Z;101500;99800;90;1.0;282.0",
             "50.1;1.8;07005;0-20000-0-07005;ABBEVILLE;;;2024-01-01T00:00:00Z;;;;;"]
    path = tmp_path / "synop_2024.csv.gz"
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")
    df = read_year(path, ["07149", "07630"])
    assert sorted(df["geo_id_wmo"]) == ["07149", "07630"]   # strings, leading zero preserved, others dropped
    out, _ = clean(df)
    assert set(out["station"]) == {"07149", "07630"}


def test_regularise_keeps_missing_rows():
    df = pd.DataFrame({"station": ["07149", "07149"],
                       "time": pd.to_datetime(["2024-01-01T00:00Z", "2024-01-01T09:00Z"]),
                       "t": [1.0, 2.0], "pmer": [1000.0, 1001.0], "u": [0.0, 0.0], "v": [0.0, 0.0],
                       "ff": [0.0, 0.0], "dd": [0.0, 0.0]})
    out = regularise(df, pd.Timestamp("2024-01-01T00:00Z"), pd.Timestamp("2024-01-01T09:00Z"))
    assert len(out) == 4 and out["t"].isna().sum() == 2       # gaps are NaN, never filled
