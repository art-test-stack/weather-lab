"""Reader for the validated hourly station archive.

This source has its own units, identifiers, and QC policy; keep it separate from
the frozen SYNOP OMM cleaner in :mod:`weather_lab.data`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from weather_lab.common import sha256_file
from weather_lab.data import wind_components

SOURCE_COLUMNS = ["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI", "AAAAMMJJHH",
                  "T", "QT", "PMER", "QPMER", "FF", "QFF", "DD", "QDD"]
VALUE_COLUMNS = {"T": "t", "PMER": "pmer", "FF": "ff", "DD": "dd"}
GOOD_QUALITY = {"0", "1"}
CHUNK_SIZE = 200_000


def _utc_hour(value: pd.Timestamp, label: str) -> pd.Timestamp:
    value = pd.Timestamp(value)
    value = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
    if value.minute or value.second or value.microsecond or value.nanosecond:
        raise ValueError(f"{label} must be on an exact UTC hour: {value}")
    return value


def read_hourly(path: Path, station: str, national_id: str, start: pd.Timestamp,
                end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Read one national station ID and regularise it on an inclusive hourly UTC axis.

    Values with QC 0 or 1 are accepted. Other or missing QC values mask the
    analysed value while the source value and QC flag remain in the output.
    """
    path = Path(path)
    if not (isinstance(station, str) and len(station) == 5 and station.isdigit()):
        raise ValueError(f"station alias {station!r} must be a 5-digit WMO string")
    if not (isinstance(national_id, str) and len(national_id) == 8 and national_id.isdigit()):
        raise ValueError(f"national_id {national_id!r} must be an exact 8-digit string")
    start, end = _utc_hour(start, "start"), _utc_hour(end, "end")
    if start > end:
        raise ValueError(f"start ({start}) must be at or before end ({end})")
    if not path.is_file():
        raise FileNotFoundError(path)

    frames = []
    found = False
    for chunk in pd.read_csv(path, sep=";", usecols=SOURCE_COLUMNS,
                             dtype={"NUM_POSTE": "string", "NOM_USUEL": "string",
                                    "AAAAMMJJHH": "string", "QT": "string", "QPMER": "string",
                                    "QFF": "string", "QDD": "string", "T": "string", "PMER": "string",
                                    "FF": "string", "DD": "string"},
                             chunksize=CHUNK_SIZE, compression="infer", encoding="utf-8"):
        selected = chunk.loc[chunk["NUM_POSTE"] == national_id].copy()
        if selected.empty:
            continue
        found = True
        selected["time"] = pd.to_datetime(selected["AAAAMMJJHH"], format="%Y%m%d%H", utc=True,
                                           errors="coerce")
        if selected["time"].isna().any():
            bad = selected.loc[selected["time"].isna(), "AAAAMMJJHH"].iloc[0]
            raise ValueError(f"invalid AAAAMMJJHH timestamp {bad!r} for national_id {national_id}")
        selected = selected.loc[selected["time"].between(start, end)]
        if not selected.empty:
            frames.append(selected)
    if not found:
        raise ValueError(f"national_id {national_id} is absent from {path}")

    source = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SOURCE_COLUMNS + ["time"])
    source["time"] = pd.to_datetime(source["time"], utc=True)
    if source["time"].duplicated().any():
        times = source.loc[source["time"].duplicated(keep=False), "time"].sort_values().unique()
        raise ValueError(f"duplicate hourly timestamps for national_id {national_id}: {times[:3].tolist()}")
    for field in ("LAT", "LON", "ALTI"):
        source[field] = pd.to_numeric(source[field], errors="coerce")

    out = pd.DataFrame({
        "station": station,
        "time": pd.date_range(start, end, freq="h"),
    }).merge(source, on="time", how="left", validate="one_to_one")
    # A national station number can have changing names or coordinates; retain those per observation.
    for raw, value in VALUE_COLUMNS.items():
        qc = "Q" + raw
        original = pd.to_numeric(out[raw], errors="coerce")
        valid = out[qc].isin(GOOD_QUALITY) & np.isfinite(original)
        out[f"{raw}_raw"] = out[raw]
        out[value] = original.where(valid)
    out["u"], out["v"] = wind_components(out["ff"].to_numpy(), out["dd"].to_numpy())
    columns = ["station", "time", "t", "pmer", "u", "v", "ff", "dd",
               "T_raw", "QT", "PMER_raw", "QPMER", "FF_raw", "QFF", "DD_raw", "QDD",
               "NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI"]
    out = out[columns]

    in_window = source
    quality_counts = {}
    for raw in VALUE_COLUMNS:
        qc = "Q" + raw
        counts = in_window[qc].fillna("<missing>").astype(str).value_counts(dropna=False)
        quality_counts[raw] = {str(code): int(count) for code, count in counts.items()}
    source_stations = (in_window[["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI"]]
                       .drop_duplicates().astype(object).where(pd.notna, None).to_dict("records"))
    observed_fraction = {name: round(float(out[name].notna().mean()), 6)
                         for name in ("t", "pmer", "ff", "dd", "u", "v")}
    metadata = {
        "file": path.name,
        "sha256": sha256_file(path),
        "national_id": national_id,
        "station_alias": station,
        "source_stations": source_stations,
        "time_axis": {"start": start.isoformat(), "end": end.isoformat(), "step": "1h UTC"},
        "units": {"t": "degC", "pmer": "hPa", "u": "m/s", "v": "m/s", "ff": "m/s", "dd": "degree"},
        "observed_fraction": observed_fraction,
        "quality_counts": quality_counts,
        "quality_rule": "accept QC codes 0 and 1; mask code 2, code 9, missing, and unknown codes",
        "availability": {
            "available_at": None,
            "limitation": "The source has no original observation publication or revision timestamps; declare a separate causal replay assumption before benchmarking.",
        },
    }
    return out, metadata
