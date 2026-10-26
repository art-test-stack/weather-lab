import gzip

import pandas as pd
import pytest

from weather_lab.hourly_observations import SOURCE_COLUMNS, read_hourly


def _write(path, rows):
    frame = pd.DataFrame(rows, columns=SOURCE_COLUMNS)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, sep=";", index=False)
    return path


def _row(hour, *, t="12.5", qt="0", pmer="1008.2", qpmer="1", ff="10", qff="0",
         dd="270", qdd="1", national_id="29075001", name="Brest", lat="48.4", lon="-4.5", alti="52"):
    return [national_id, name, lat, lon, alti, f"20260101{hour:02d}", t, qt, pmer, qpmer, ff, qff, dd, qdd]


def test_read_hourly_keeps_units_qc_raw_and_hourly_gaps(tmp_path):
    path = _write(tmp_path / "hourly.csv.gz", [
        _row(0),
        _row(1, qt="1", qpmer="2", t="13", pmer="1007"),
        _row(3, qt="9", qpmer="0", qff="2", qdd="9", t="14", pmer="1006", ff="8", dd="90"),
        _row(4, qt="", qpmer="unknown", qff="0", qdd="0", t="15", pmer="1005", ff="0", dd=""),
    ])

    frame, meta = read_hourly(path, "07110", "29075001", pd.Timestamp("2026-01-01"),
                              pd.Timestamp("2026-01-01 04:00Z"))

    assert frame.station.tolist() == ["07110"] * 5
    assert frame.time.tolist() == list(pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC"))
    assert frame.loc[0, ["t", "pmer", "ff", "dd"]].tolist() == [12.5, 1008.2, 10.0, 270.0]
    assert frame.loc[0, "u"] == pytest.approx(10)
    assert frame.loc[0, "v"] == pytest.approx(0, abs=1e-12)
    assert frame.loc[1, "t"] == 13.0
    assert pd.isna(frame.loc[1, "pmer"])
    assert frame.loc[1, "PMER_raw"] == "1007" and frame.loc[1, "QPMER"] == "2"
    assert frame.loc[2, ["t", "pmer", "ff", "dd"]].isna().all()
    assert (frame.loc[3, ["T_raw", "QT", "FF_raw", "QFF"]].tolist() == ["14", "9", "8", "2"])
    assert frame.loc[3, ["t", "ff", "dd"]].isna().all()
    assert frame.loc[3, "pmer"] == 1006.0
    assert frame.loc[4, ["t", "pmer"]].isna().all()
    assert pd.isna(frame.loc[4, "QT"])
    assert frame.loc[4, ["u", "v"]].tolist() == [0.0, 0.0]
    assert pd.isna(frame.loc[4, "dd"])
    assert frame.loc[0, "NUM_POSTE"] == "29075001"
    assert meta["national_id"] == "29075001" and meta["station_alias"] == "07110"
    assert meta["units"] == {"t": "degC", "pmer": "hPa", "u": "m/s", "v": "m/s",
                             "ff": "m/s", "dd": "degree"}
    assert meta["observed_fraction"]["t"] == 0.4
    assert meta["quality_counts"]["T"] == {"0": 1, "1": 1, "9": 1, "<missing>": 1}
    assert "causal replay assumption" in meta["availability"]["limitation"]


def test_rejects_duplicate_timestamps_and_missing_station(tmp_path):
    duplicate = _write(tmp_path / "duplicate.csv.gz", [_row(0), _row(0, t="99")])
    with pytest.raises(ValueError, match="duplicate hourly timestamps"):
        read_hourly(duplicate, "07110", "29075001", pd.Timestamp("2026-01-01", tz="UTC"),
                    pd.Timestamp("2026-01-01 01:00Z"))

    absent = _write(tmp_path / "absent.csv.gz", [_row(0)])
    with pytest.raises(ValueError, match="national_id 29075002 is absent"):
        read_hourly(absent, "07110", "29075002", pd.Timestamp("2026-01-01", tz="UTC"),
                    pd.Timestamp("2026-01-01", tz="UTC"))


def test_existing_station_with_no_rows_in_window_returns_missing_axis(tmp_path):
    path = _write(tmp_path / "hourly.csv.gz", [_row(0)])

    frame, meta = read_hourly(path, "07110", "29075001", pd.Timestamp("2026-01-02"),
                              pd.Timestamp("2026-01-02 02:00Z"))

    assert frame.station.tolist() == ["07110"] * 3
    assert frame.time.tolist() == list(pd.date_range("2026-01-02", periods=3, freq="h", tz="UTC"))
    assert frame[["t", "pmer", "u", "v", "ff", "dd"]].isna().all().all()
    assert meta["observed_fraction"] == {name: 0.0 for name in ("t", "pmer", "ff", "dd", "u", "v")}


@pytest.mark.parametrize("station,national_id,start,end,message", [
    ("7110", "29075001", "2026-01-01", "2026-01-01", "5-digit WMO"),
    ("07110", 29075001, "2026-01-01", "2026-01-01", "8-digit string"),
    ("07110", "29075001", "2026-01-01 00:30", "2026-01-01 01:00", "exact UTC hour"),
    ("07110", "29075001", "2026-01-01 02:00", "2026-01-01 01:00", "at or before end"),
])
def test_validates_ids_and_hourly_window(tmp_path, station, national_id, start, end, message):
    path = _write(tmp_path / "hourly.csv.gz", [_row(0)])
    with pytest.raises(ValueError, match=message):
        read_hourly(path, station, national_id, pd.Timestamp(start), pd.Timestamp(end))
