"""Bounded, issued AROME patch information; no cell selection by verification error."""
from __future__ import annotations

import numpy as np
import pandas as pd

from weather_lab.arome import haversine_km, nearest_index
from weather_lab.arome_inputs import build_arome_inputs
from weather_lab.forecast import Timestral


ROWS = ("patch_mean_t", "patch_sd_t", "patch_east_gradient_t", "patch_north_gradient_t")
DEFINITION = {
    "name": "arome-spatial-patch", "version": 1, "role": "known",
    "geometry": "nearest-point-centred 3x3; all nine cells within 5 km of station",
    "rows": list(ROWS), "units": ["degC", "degC", "degC/km", "degC/km"],
    "statistics": "arithmetic mean; population SD; centred east/north finite differences",
    "availability": "issued future from selected run; archived history published by valid time and origin",
    "missing": "any absent/nonfinite cell makes all patch statistics unavailable; no replacement",
    "static_metadata": "saved beside inputs, not supplied as constant normalized rows",
    "mask": "NaN; wholly missing selected future rows refused; partial historical gaps retained",
}


def cell_coords(grid: dict, j: int, i: int) -> tuple[float, float]:
    dj = grid["jDirectionIncrementInDegrees"] * (1 if grid["jScansPositively"] else -1)
    return (grid["latitudeOfFirstGridPointInDegrees"] + j * dj,
            (grid["longitudeOfFirstGridPointInDegrees"] + 180) % 360 - 180
            + i * grid["iDirectionIncrementInDegrees"])


def patch_cells(grid: dict, lat: float, lon: float) -> list[tuple[int, int]]:
    j, i, _, _ = nearest_index(grid, lat, lon)
    cells = [(jj, ii) for jj in range(j - 1, j + 2) for ii in range(i - 1, i + 2)]
    if any(not (0 <= jj < grid["Nj"] and 0 <= ii < grid["Ni"])
           or haversine_km(lat, lon, *cell_coords(grid, jj, ii)) > 5 for jj, ii in cells):
        return []
    return cells


def patch_statistics(values: np.ndarray, grid: dict, cells: list[tuple[int, int]]) -> np.ndarray:
    if len(cells) != 9:
        return np.full(4, np.nan)
    patch = np.array([values[j, i] for j, i in cells]).reshape(3, 3)
    if not np.isfinite(patch).all():
        return np.full(4, np.nan)
    j, i = cells[4]
    lat, lon = cell_coords(grid, j, i)
    dx = haversine_km(*cell_coords(grid, j, i - 1), *cell_coords(grid, j, i + 1))
    dy = haversine_km(*cell_coords(grid, j - 1, i), *cell_coords(grid, j + 1, i))
    north_sign = 1 if grid["jScansPositively"] else -1
    return np.array([patch.mean(), patch.std(ddof=0), (patch[1, 2] - patch[1, 0]) / dx,
                     north_sign * (patch[2, 1] - patch[0, 1]) / dy])


def spatial_rows(frame: pd.DataFrame, batch, context_steps: int):
    inputs = build_arome_inputs(frame, batch.cases, batch.origins,
                                batch.history_times(context_steps), batch.future_times, ROWS)
    if not np.isfinite(inputs.future).all():
        raise ValueError("Selected run lacks complete bounded spatial patch information; candidate unavailable")
    rows = np.concatenate([inputs.history, inputs.future], axis=-1)
    return rows, {"definition": DEFINITION, "historical_observed_fraction":
                  np.isfinite(inputs.history).mean(axis=(0, 2)).tolist(),
                  "future_observed_fraction": np.isfinite(inputs.future).mean(axis=(0, 2)).tolist()}


class SpatialTimestral(Timestral):
    """Native known-covariate extension; source forecasts live on this adapter, never on weights."""

    def __init__(self, *args, spatial_frame: pd.DataFrame, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_frame = spatial_frame.copy()
        self.params["spatial_definition"] = DEFINITION

    def build_inputs(self, batch):
        context, known = super().build_inputs(batch)
        patch, _ = spatial_rows(self.spatial_frame, batch, self.context_steps(batch.step_hours))
        return context, patch if known is None else np.concatenate([known, patch], axis=1)

    def predict(self, batch):
        result = super().predict(batch)
        result.info["spatial_inputs"] = spatial_rows(
            self.spatial_frame, batch, self.context_steps(batch.step_hours))[1]
        return result
