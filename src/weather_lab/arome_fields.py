"""Surface AROME field definitions and conversion to forecast-step inputs.

Verified against the 0.025° SP1/SP2 headers of the 2026-10-21 03 UTC run. Input definitions
are versioned separately from the frozen benchmark extractor in ``arome.FIELDS``.
"""

from __future__ import annotations

import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from weather_lab.arome import parse_header


@dataclass(frozen=True)
class AromeField:
    name: str
    label: str
    unit: str
    package: str
    key: tuple[int, ...]             # discipline, category, number, surface type, surface value
    method: str = "instant"         # instant, sum per step, mean over step, maximum over step
    scale: float = 1.0
    offset: float = 0.0
    observation: str | None = None
    version: int = 1


FIELDS = {f.name: f for f in (
    AromeField("t", "Temperature at 2 m", "°C", "SP1", (0, 0, 0, 103, 2), offset=-273.15, observation="t"),
    AromeField("pmer", "Mean sea-level pressure", "hPa", "SP1", (0, 3, 1, 101, 0), scale=0.01,
               observation="pmer"),
    AromeField("u", "Eastward wind at 10 m", "m/s", "SP1", (0, 2, 2, 103, 10), observation="u"),
    AromeField("v", "Northward wind at 10 m", "m/s", "SP1", (0, 2, 3, 103, 10), observation="v"),
    AromeField("gust", "Maximum wind gust at 10 m", "m/s", "SP1", (0, 2, 22, 103, 10), "max"),
    AromeField("rh", "Relative humidity at 2 m", "%", "SP1", (0, 1, 1, 103, 2)),
    AromeField("td", "Dew point at 2 m", "°C", "SP2", (0, 0, 6, 103, 2), offset=-273.15, observation="td"),
    AromeField("cloud", "Total cloud cover", "%", "SP1", (0, 6, 1, 1, 0), observation="cloud"),
    AromeField("precip", "Total precipitation per step", "mm", "SP1", (0, 1, 52, 1, 0), "sum"),
    AromeField("snowfall", "Snowfall per step (water equivalent)", "mm", "SP1", (0, 1, 66, 1, 0), "sum"),
    AromeField("shortwave", "Mean downward solar radiation over step", "W/m²", "SP1", (0, 4, 7, 1, 0), "mean"),
)}
DEFAULT_FIELDS = ("t", "pmer")


def normalize(fields) -> tuple[str, ...]:
    unknown = set(fields) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown AROME input fields: {sorted(unknown)}")
    return tuple(name for name in FIELDS if name in fields)


def definition(fields) -> list[dict]:
    return [asdict(FIELDS[name]) for name in normalize(fields)]


def parse_field_header(raw: bytes) -> dict:
    """Read instantaneous and statistical GRIB2 headers; never infer an accumulation's end from its start."""
    info = parse_header(raw)
    if info["time_unit"] != 1:
        raise ValueError("AROME field header does not express lead times in hours")
    info.update(start_h=info["forecast_time"], end_h=info["forecast_time"], statistic="instant")
    if info["template"] == 0:
        return info
    if info["template"] != 8:
        raise ValueError(f"unsupported AROME product template {info['template']}")
    pos = 16
    while pos + 5 <= len(raw):
        length = struct.unpack(">I", raw[pos:pos + 4])[0]
        if raw[pos + 4] == 4:
            section = raw[pos:pos + length]
            if len(section) < 58 or section[41] != 1 or section[47] != 2 or section[48] != 1:
                raise ValueError("unsupported AROME statistical interval")
            statistics = {0: "mean", 1: "sum", 2: "max", 3: "min"}
            if section[46] not in statistics:
                raise ValueError("unsupported AROME statistical process")
            info.update(end_h=info["start_h"] + struct.unpack(">I", section[49:53])[0],
                        statistic=statistics[section[46]])
            return info
        if length == 0:
            break
        pos += length
    raise ValueError("missing AROME statistical section")


def field_for_header(info: dict) -> str | None:
    key = tuple(info[k] for k in ("discipline", "category", "number", "surface_type", "surface_value"))
    for name, field in FIELDS.items():
        statistic = "sum" if field.method == "mean" else field.method
        if field.key == key and info["statistic"] == statistic:
            return name
    return None


def decode_station_records(path: Path, stations: list[dict], run, available_at) -> pd.DataFrame:
    """Decode native station values and interval bounds from a verified GRIB subset."""
    import eccodes

    from weather_lab.arome import _grid, _messages, nearest_index

    run, available_at = pd.to_datetime(run, utc=True), pd.to_datetime(available_at, utc=True)
    if pd.isna(run) or pd.isna(available_at) or available_at < run:
        raise ValueError("AROME source needs a publication time at or after its run")
    rows = []
    for handle in _messages(path):
        info = parse_field_header(eccodes.codes_get_message(handle))
        name = field_for_header(info)
        if name is None:
            continue
        stamp = f"{eccodes.codes_get(handle, 'dataDate'):08d}{eccodes.codes_get(handle, 'dataTime'):04d}"
        if pd.Timestamp(stamp, tz="UTC") != run:
            raise ValueError("AROME message run differs from its source record")
        grid = _grid(handle)
        values = eccodes.codes_get_values(handle).reshape(grid["Nj"], grid["Ni"])
        bitmap = (eccodes.codes_get_array(handle, "bitmap").reshape(values.shape)
                  if eccodes.codes_get(handle, "bitmapPresent") else None)
        for station in stations:
            j, i, lat, lon = nearest_index(grid, station["lat"], station["lon"])
            value = float(values[j, i]) if bitmap is None or bitmap[j, i] else np.nan
            if np.isinf(value):
                raise ValueError("infinite AROME source value")
            rows.append({"run": run, "available_at": available_at, "station": station["id"],
                         "variable": name, "start_h": info["start_h"], "end_h": info["end_h"], "value": value,
                         "grid_lat": lat, "grid_lon": lon, "source_file": path.name})
    return pd.DataFrame(rows, columns=["run", "available_at", "station", "variable", "start_h", "end_h", "value",
                                       "grid_lat", "grid_lon", "source_file"])


def _window(rows: pd.DataFrame, start: int, end: int, method: str):
    """(value, source rows), or missing when the requested interval cannot be reconstructed exactly."""
    exact = rows[(rows["start_h"] == start) & (rows["end_h"] == end)]
    if len(exact) == 1:
        return float(exact["value"].iloc[0]), exact
    if method in ("sum", "mean"):
        right = rows[(rows["end_h"] == end) & (rows["start_h"] < start)]
        for _, r in right.iterrows():
            left = rows[(rows["end_h"] == start) & (rows["start_h"] == r["start_h"])]
            if len(left) == 1:
                return float(r["value"] - left["value"].iloc[0]), pd.concat([left, r.to_frame().T])
    parts = rows[(rows["start_h"] >= start) & (rows["end_h"] <= end)].sort_values("start_h")
    cursor = start
    for row in parts.itertuples():
        if row.start_h != cursor or row.end_h <= cursor:
            return np.nan, parts.iloc[:0]
        cursor = row.end_h
    if cursor != end or parts.empty:
        return np.nan, parts.iloc[:0]
    return float(parts["value"].max() if method == "max" else parts["value"].sum()), parts


def step_values(records: pd.DataFrame, leads_h, step_hours: int = 3) -> pd.DataFrame:
    """Native station records -> version-1 values on exact forecast-step windows.

    Accumulation differences use the same run and accumulation start. Gusts require a complete
    tiling of the step, or its exact maximum. Incomplete intervals stay missing, never zero-filled.
    Publication time is the latest timestamp of every record needed to construct the value.
    """
    if step_hours <= 0 or any(lead < step_hours for lead in leads_h):
        raise ValueError("AROME leads must cover a full positive forecast step")
    out = []
    records = records.copy()
    records["available_at"] = pd.to_datetime(records["available_at"], utc=True)
    records["run"] = pd.to_datetime(records["run"], utc=True)
    if np.isinf(records["value"]).any():
        raise ValueError("infinite AROME source value")
    if records.duplicated(["run", "station", "variable", "start_h", "end_h"]).any():
        raise ValueError("duplicate AROME source intervals")
    for (run, station, name), rows in records.groupby(["run", "station", "variable"]):
        field = FIELDS[name]
        rows = rows.dropna(subset=["value", "available_at"])
        for lead in leads_h:
            if field.method == "instant":
                sources = rows[(rows["start_h"] == lead) & (rows["end_h"] == lead)]
                value = float(sources["value"].iloc[0]) if len(sources) == 1 else np.nan
            else:
                value, sources = _window(rows, lead - step_hours, lead, field.method)
                if np.isfinite(value) and value < 0:
                    raise ValueError(f"negative AROME {name} interval at lead {lead} h")
                if field.method == "mean":
                    value /= step_hours * 3600
            out.append({"run": run, "station": station, "variable": name, "lead_h": lead,
                        "valid_time": run + pd.Timedelta(hours=lead), "value": value * field.scale + field.offset,
                        "available_at": sources["available_at"].max() if len(sources) else pd.NaT,
                        "step_hours": step_hours, "field_version": field.version})
    return pd.DataFrame(out)
