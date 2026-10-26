"""Frozen schedule and candidate identities for the AROME-error context study."""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from weather_lab.forecast_config import ForecastConfig
from weather_lab.suite import case_id

SUITE = "ar-error-context-v1"
MAX_HORIZON_H = 48


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    start = pd.Timestamp(value)
    if pd.isna(start) or start.tzinfo is None:
        raise ValueError("collection_start must be timezone-aware at 06:00 UTC")
    start = start.tz_convert("UTC")
    if (start.hour, start.minute, start.second, start.microsecond, start.nanosecond) != (6, 0, 0, 0, 0):
        raise ValueError("collection_start must be at 06:00 UTC")
    return start


def _iso(value: pd.Timestamp) -> str:
    return value.isoformat().replace("+00:00", "Z")


def schedule(collection_start: str | pd.Timestamp) -> dict:
    """Return frozen half-open collection and evaluation windows anchored at 06:00 UTC."""
    start = _utc(collection_start)
    validation_start = start + pd.Timedelta(days=90)
    validation_end = validation_start + pd.Timedelta(days=30)
    purge_end = validation_end + pd.Timedelta(days=2)
    confirmation_end = purge_end + pd.Timedelta(days=60)
    last_validation_label = validation_end - pd.Timedelta(days=1) + pd.Timedelta(hours=MAX_HORIZON_H)
    last_confirmation_label = confirmation_end - pd.Timedelta(days=1) + pd.Timedelta(hours=MAX_HORIZON_H)
    if last_validation_label >= purge_end:
        raise ValueError("Validation labels must precede confirmation origins")
    return {
        "collection_start": _iso(start),
        "window_convention": "half-open [start, end)",
        "history_window": {"start": _iso(start), "end": _iso(validation_start)},
        "validation_window": {"start": _iso(validation_start), "end": _iso(validation_end)},
        "purge_window": {"start": _iso(validation_end), "end": _iso(purge_end)},
        "confirmation_window": {"start": _iso(purge_end), "end": _iso(confirmation_end)},
        "last_validation_label_time": _iso(last_validation_label),
        "last_confirmation_label_time": _iso(last_confirmation_label),
        "calibration": None,
    }


def origins(collection_start: str | pd.Timestamp, stations: list[str]) -> pd.DataFrame:
    """Build every validation and confirmation case; do not trim by model context or horizon."""
    if (not isinstance(stations, list) or not stations
            or any(not isinstance(s, str) or len(s) != 5 or not s.isdigit() for s in stations)
            or len(set(stations)) != len(stations)):
        raise ValueError("stations must be unique five-digit strings")
    windows = schedule(collection_start)
    rows = []
    for split in ("validation", "confirmation"):
        window = windows[f"{split}_window"]
        first, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
        for origin in pd.date_range(first, end - pd.Timedelta(days=1), freq="D"):
            for station in stations:
                rows.append({"case_id": case_id(SUITE, station, origin), "station": station,
                             "origin": origin, "split": split})
    return pd.DataFrame(rows, columns=["case_id", "station", "origin", "split"])


def candidates() -> dict[str, ForecastConfig]:
    """Return the four frozen native-horizon Timestral error configurations."""
    return {
        f"timestral209m-error-ctx{context}-h{horizon}-v1": ForecastConfig.make(
            "Timestral-209M", "futureonly", calendar=False, context_days=context,
            horizon_h=horizon, targets=("t", "pmer"), step_hours=3, prediction_target="arome_error")
        for context in (30, 90) for horizon in (24, 48)
    }


def candidate_identity(config: ForecastConfig, provenance: dict) -> str:
    """Hash the complete forecast config, including native horizon, and caller-supplied provenance."""
    payload = json.dumps({"config": config.as_dict(), "provenance": provenance}, sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()
