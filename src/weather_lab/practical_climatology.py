"""Fixed pooled harmonic climatology and reversible standalone anomaly targets."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json

import numpy as np
import pandas as pd

from weather_lab.common import VARIABLES
from weather_lab.forecast import CaseBatch, ForecastBatch


def harmonic_design(stations, times, fitted_stations: tuple[str, ...]) -> np.ndarray:
    """Station intercepts and shared first/second annual and UTC daily harmonics."""
    stations = np.asarray(stations)
    time = pd.DatetimeIndex(pd.to_datetime(np.asarray(times).ravel(), utc=True))
    if stations.shape != np.asarray(times).shape or not set(stations.ravel()) <= set(fitted_stations):
        raise ValueError("Climatology requires known stations aligned with timestamps")
    if time.isna().any():
        raise ValueError("Climatology timestamps cannot be missing")
    hour = np.asarray(time.hour + time.minute / 60 + time.second / 3600)
    annual = 2 * np.pi * (np.asarray(time.day_of_year) - 1 + hour / 24) / 365.25
    daily = 2 * np.pi * hour / 24
    return np.column_stack([*(stations.ravel() == s for s in fitted_stations),
                            *(f(k * phase) for phase in (annual, daily)
                              for k in (1, 2) for f in (np.sin, np.cos))]).astype(float)


@dataclass
class HarmonicClimatology:
    stations: tuple[str, ...]
    variables: list[str]
    coefficients: np.ndarray
    identity: str
    provenance: dict

    @classmethod
    def fit(cls, observations: pd.DataFrame, variables: list[str], *, source_fingerprint: str,
            start="2018-01-01T00:00:00Z", end="2022-12-31T21:00:00Z") -> "HarmonicClimatology":
        if not variables or not set(variables) <= VARIABLES.keys():
            raise ValueError("Unknown target variable or units")
        data = observations.sort_values(["station", "time"]).reset_index(drop=True)
        if data.empty or data.time.lt(pd.Timestamp(start)).any() or data.time.gt(pd.Timestamp(end)).any():
            raise ValueError("Climatology fitting observations extend outside the declared training interval")
        if data.duplicated(["station", "time"]).any():
            raise ValueError("Duplicate climatology training observations")
        stations = tuple(sorted(data.station.unique()))
        design = harmonic_design(data.station.to_numpy(), data.time.to_numpy(), stations)
        penalty = np.diag([0.] * len(stations) + [1.] * 8)
        coefficients, support = [], {}
        for variable in variables:
            finite = np.isfinite(data[variable])
            for station, group in data[finite].groupby("station"):
                if len(group) < 1000 or group.time.dt.month.nunique() != 12:
                    raise ValueError(f"Insufficient annual climatology support for {station}/{variable}")
            counts = data[finite].groupby("station").size()
            if set(counts.index) != set(stations):
                raise ValueError(f"Missing station training support for {variable}")
            x = design[finite]
            coefficients.append(np.linalg.solve(x.T @ x + penalty, x.T @ data.loc[finite, variable].to_numpy()))
            support[variable] = {str(s): int(n) for s, n in counts.items()}
        coefficients = np.asarray(coefficients)
        provenance = {"version": "pooled-harmonic-v1", "source_fingerprint": source_fingerprint,
                      "training_start": start, "training_end": end, "ridge_penalty": 1,
                      "station_intercepts_penalized": False, "annual_harmonics": 2, "daily_harmonics": 2,
                      "time_basis": "UTC; annual fractional day /365.25", "stations": stations,
                      "variables": variables, "units": {v: VARIABLES[v][1] for v in variables},
                      "support": support,
                      "training_values_sha256": hashlib.sha256(pd.util.hash_pandas_object(
                          data[["station", "time"] + variables], index=False).values.tobytes()).hexdigest(),
                      "coefficients": coefficients.tolist()}
        identity = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        return cls(stations, list(variables), coefficients, identity, provenance)

    def values(self, stations, times) -> np.ndarray:
        """Return physical-unit climatology as (case, variable, time)."""
        times = np.asarray(times)
        if times.ndim != 2 or len(stations) != len(times):
            raise ValueError("Climatology expects one time grid per station case")
        station_grid = np.broadcast_to(np.asarray(stations)[:, None], times.shape)
        x = harmonic_design(station_grid, times, self.stations)
        return (x @ self.coefficients.T).reshape(*times.shape, len(self.variables)).transpose(0, 2, 1)

    def anomaly_batch(self, batch: CaseBatch) -> CaseBatch:
        if batch.variables != self.variables:
            raise ValueError("Climatology variable order must match the model input")
        if pd.Timestamp(batch.origins.min(), tz="UTC") <= pd.Timestamp(self.provenance["training_end"]):
            raise ValueError("Climatology fit labels overlap the forecast period")
        baseline = self.values(batch.cases.station.to_numpy(), batch.history_times())
        history = (batch.history - baseline).astype(np.float32)
        assert np.array_equal(np.isnan(history), np.isnan(batch.history))
        return replace(batch, history=history)

    def restore(self, forecast: ForecastBatch, batch: CaseBatch) -> ForecastBatch:
        if forecast.variables != self.variables:
            raise ValueError("Climatology variable order must match the forecast")
        baseline = self.values(batch.cases.station.to_numpy(), batch.future_times)
        return replace(forecast, point=forecast.point + baseline,
                       quantiles=None if forecast.quantiles is None else forecast.quantiles + baseline[..., None],
                       info={**forecast.info, "target_transform": "pooled-harmonic-anomaly-v1",
                             "transform_fit_identity": self.identity,
                             "inverse_transform": "add future climatology to each physical-unit quantile"})

    def predict(self, batch: CaseBatch) -> ForecastBatch:
        return ForecastBatch("m1-pooled-harmonic-climatology-v1", self.identity, "point", self.variables,
                             self.values(batch.cases.station.to_numpy(), batch.future_times),
                             info={"transform_fit_identity": self.identity})
