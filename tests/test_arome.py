"""AROME run availability, lead alignment, GRIB2 header parsing and grid lookup (no network)."""

import struct

import pandas as pd
import pytest

from weather_lab.arome import file_for_lead, nearest_index, parse_header, required_leads, run_availability


def test_required_leads_for_06z_origin_from_03z_run():
    origin, init = pd.Timestamp("2026-10-15T06:00Z"), pd.Timestamp("2026-10-15T03:00Z")
    leads = required_leads(origin, init, horizon_steps=16, step_h=3)
    assert leads[0] == 6 and leads[-1] == 51 and len(leads) == 16   # origin + 3 h ... origin + 48 h
    # valid times line up with the benchmark's target times
    assert [init + pd.Timedelta(hours=h) for h in leads][0] == origin + pd.Timedelta(hours=3)


def test_file_groups():
    name, url = file_for_lead("https://x/pnt", pd.Timestamp("2026-10-15T03:00Z"), 51)
    assert name == "arome__0025__SP1__49H51H__2026-10-15T03:00:00Z.grib2"
    assert url.endswith("/2026-10-15T03:00:00Z/arome/0025/SP1/" + name)
    assert file_for_lead("u", pd.Timestamp("2026-10-15T03:00Z"), 6)[0].startswith("arome__0025__SP1__00H06H")
    with pytest.raises(ValueError):
        file_for_lead("u", pd.Timestamp("2026-10-15T03:00Z"), 52)   # never extrapolate past 51 h


def test_run_availability_requires_every_file_before_origin():
    origin = pd.Timestamp("2026-10-15T06:00Z")
    ok_headers = ["Thu, 15 Oct 2026 04:47:00 GMT", "Thu, 15 Oct 2026 05:45:19 GMT"]
    complete, ok = run_availability(ok_headers, origin)
    assert ok and complete == pd.Timestamp("2026-10-15T05:45:19Z")
    late = ok_headers + ["Thu, 15 Oct 2026 06:00:01 GMT"]
    assert run_availability(late, origin)[1] is False
    # initialisation time alone proves nothing: a run published after the origin is rejected
    assert run_availability(["Thu, 15 Oct 2026 07:00:00 GMT"], origin)[1] is False


def _grib_header(cat, num, ft, stype, sval, template=0):
    sec0 = b"GRIB" + b"\x00\x00" + bytes([0, 2]) + struct.pack(">Q", 1000)
    sec1 = struct.pack(">IB", 21, 1) + b"\x00" * 16
    sec3 = struct.pack(">IB", 72, 3) + b"\x00" * 67
    body4 = bytearray(34 - 5)
    body4[0:2] = (0).to_bytes(2, "big")            # number of coordinate values
    body4[2:4] = template.to_bytes(2, "big")
    body4[4], body4[5] = cat, num
    body4[12] = 1                                  # hours
    body4[13:17] = ft.to_bytes(4, "big", signed=True)
    body4[17] = stype
    body4[18] = 0
    body4[19:23] = sval.to_bytes(4, "big")
    sec4 = struct.pack(">IB", 34, 4) + bytes(body4)
    sec5 = struct.pack(">IB", 21, 5) + struct.pack(">I", 665681) + struct.pack(">H", 42) + b"\x00" * 10
    return sec0 + sec1 + sec3 + sec4 + sec5


def test_parse_header_identifies_t2m_and_mslp():
    t2m = parse_header(_grib_header(0, 0, 24, 103, 2))
    assert (t2m["category"], t2m["number"], t2m["surface_type"], t2m["surface_value"], t2m["forecast_time"]) == \
        (0, 0, 103, 2, 24)
    assert t2m["length"] == 1000
    mslp = parse_header(_grib_header(3, 1, 51, 101, 0))
    assert (mslp["category"], mslp["number"], mslp["surface_type"], mslp["forecast_time"]) == (3, 1, 101, 51)


def test_nearest_index_handles_longitude_wrap():
    grid = {"Ni": 1121, "Nj": 717, "latitudeOfFirstGridPointInDegrees": 55.4,
            "longitudeOfFirstGridPointInDegrees": 348.0, "iDirectionIncrementInDegrees": 0.025,
            "jDirectionIncrementInDegrees": 0.025, "jScansPositively": 0}
    j, i, lat, lon = nearest_index(grid, 48.444167, -4.412)          # Brest-Guipavas
    assert (lat, lon) == (48.45, -4.4)
    with pytest.raises(ValueError):
        nearest_index(grid, 60.0, 0.0)
