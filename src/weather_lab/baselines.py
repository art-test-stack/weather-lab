"""Reference forecasts: persistence, 24 h seasonal naive and empirical-quantile climatology."""

from __future__ import annotations

import numpy as np
import pandas as pd

from weather_lab.forecast import QUANTILE_LEVELS, Adapter, CaseBatch, ForecastBatch


class Persistence(Adapter):
    """Last available observation at or before the origin, repeated at every lead."""

    kind = "persistence"

    def predict(self, batch: CaseBatch) -> ForecastBatch:
        idx = [batch.variables.index(v) for v in self.variables]
        hist = batch.history[:, idx]
        finite = np.isfinite(hist)
        # index of last finite value along time; -1 if none
        last = np.where(finite.any(-1), hist.shape[-1] - 1 - np.argmax(finite[..., ::-1], axis=-1), -1)
        vals = np.take_along_axis(hist, np.maximum(last, 0)[..., None], axis=-1)[..., 0]
        vals = np.where(last >= 0, vals, np.nan)
        point = np.repeat(vals[..., None], batch.horizon, axis=-1)
        return ForecastBatch(self.id, "n/a", "point", self.variables, point)


class SeasonalNaive(Adapter):
    """Repeat the last observed 24 h profile: value at valid_time - 24 h * ceil(lead / 24 h).

    Only observations at or before the origin are used; a missing source value stays missing.
    """

    kind = "seasonal_naive"

    def predict(self, batch: CaseBatch) -> ForecastBatch:
        idx = [batch.variables.index(v) for v in self.variables]
        hist = batch.history[:, idx]
        period = 24 // batch.step_hours
        k = np.arange(1, batch.horizon + 1)
        back = period * np.ceil(k / period).astype(int)
        src = hist.shape[-1] - 1 + k - back          # <= T-1, i.e. at or before the origin
        if src.min() < 0:
            raise ValueError("history too short for seasonal naive")
        return ForecastBatch(self.id, "n/a", "point", self.variables, hist[..., src])


class Climatology(Adapter):
    """Empirical quantiles of fitting-period observations.

    Grouping: station x variable x UTC hour slot x day-of-year window (+/- `window_days`,
    circular, no further smoothing). If fewer than `min_samples` finite values fall in the
    window it is widened to 2x, then the forecast is missing. Point forecast = median.
    """

    kind = "climatology"
    output = "quantiles"

    def __init__(self, id: str, variables: list[str], window_days: int = 15, min_samples: int = 20, **p):
        super().__init__(id, variables, window_days=window_days, min_samples=min_samples, **p)
        self.window, self.min_samples = window_days, min_samples
        self.table: dict[tuple[str, str], np.ndarray] = {}   # -> (366, 24, Q) NaN where unfitted
        self.fit_period: tuple[str, str] | None = None

    def fit(self, fit_obs: pd.DataFrame) -> None:
        self.fit_period = (str(fit_obs["time"].min()), str(fit_obs["time"].max()))
        levels = np.array(QUANTILE_LEVELS)
        for station, g in fit_obs.groupby("station"):
            doy = g["time"].dt.dayofyear.to_numpy()
            hour = g["time"].dt.hour.to_numpy()
            for var in self.variables:
                vals = g[var].to_numpy(dtype=float)
                table = np.full((367, 24, len(levels)), np.nan)
                for hr in np.unique(hour):
                    sel = (hour == hr) & np.isfinite(vals)
                    v_h, d_h = vals[sel], doy[sel]
                    for d in range(1, 367):
                        dist = np.abs(d_h - d)
                        dist = np.minimum(dist, 366 - dist)
                        for w in (self.window, 2 * self.window):
                            sample = v_h[dist <= w]
                            if len(sample) >= self.min_samples:
                                table[d, hr] = np.quantile(sample, levels)
                                break
                self.table[(station, var)] = table

    def predict(self, batch: CaseBatch) -> ForecastBatch:
        if not self.table:
            raise RuntimeError("climatology not fitted")
        n, h = len(batch.cases), batch.horizon
        q = np.full((n, len(self.variables), h, len(QUANTILE_LEVELS)), np.nan)
        times = pd.DatetimeIndex(batch.future_times.ravel())
        doy = times.dayofyear.to_numpy().reshape(n, h)
        hour = times.hour.to_numpy().reshape(n, h)
        for i, station in enumerate(batch.cases["station"].to_numpy()):
            for j, var in enumerate(self.variables):
                table = self.table.get((station, var))
                if table is not None:
                    q[i, j] = table[doy[i], hour[i]]
        point = q[..., QUANTILE_LEVELS.index(0.5)]
        return ForecastBatch(self.id, f"fit {self.fit_period[0][:10]}..{self.fit_period[1][:10]}", "quantiles",
                             self.variables, point, q, QUANTILE_LEVELS)
