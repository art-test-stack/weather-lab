import numpy as np
import pandas as pd
import pytest

from weather_lab.forecast import CaseBatch
from weather_lab.practical_spatial import ROWS, patch_cells, patch_statistics, spatial_rows


def test_fixed_patch_planar_units_bound_and_missing_cells():
    grid = dict(Ni=5, Nj=5, latitudeOfFirstGridPointInDegrees=48., longitudeOfFirstGridPointInDegrees=2.,
                iDirectionIncrementInDegrees=.025, jDirectionIncrementInDegrees=.025, jScansPositively=1)
    cells = patch_cells(grid, 48.05, 2.05)
    values = np.fromfunction(lambda j, i: 10 + 2 * i + 3 * j, (5, 5))
    stats = patch_statistics(values, grid, cells)
    assert stats[0] == 20
    assert stats[1] == pytest.approx(np.std([values[j, i] for j, i in cells]))
    assert 1 < stats[2] < 1.1  # 2 degC per ~1.86 km longitude increment
    assert 1 < stats[3] < 1.1  # 3 degC per ~2.78 km latitude increment
    reversed_grid = {**grid, "latitudeOfFirstGridPointInDegrees": 48.1, "jScansPositively": 0}
    assert patch_statistics(values[::-1], reversed_grid, patch_cells(reversed_grid, 48.05, 2.05))[3] == pytest.approx(stats[3])
    assert patch_cells(grid, 48., 2.) == []
    values[1, 1] = np.nan
    assert np.isnan(patch_statistics(values, grid, cells)).all()
    assert np.isnan(patch_statistics(values, grid, [])).all()


def test_spatial_roles_cutoff_and_selected_run_alignment():
    origin = pd.Timestamp("2026-10-16T06:00Z")
    run = origin - pd.Timedelta(hours=3)
    cases = pd.DataFrame([{"station": "07149", "arome_run": run}])
    batch = CaseBatch(cases, ["t"], np.zeros((1, 1, 2)),
                      np.array([origin.to_datetime64()]),
                      np.array([[(origin + pd.Timedelta(hours=3)).to_datetime64()]]), 3)
    records = []
    for field in ROWS:
        for valid, available, source, value in (
                (origin - pd.Timedelta(hours=3), origin - pd.Timedelta(hours=4), run - pd.Timedelta(days=1), 1),
                (origin, origin + pd.Timedelta(hours=1), run, 99),  # late: cannot enter context
                (origin + pd.Timedelta(hours=3), origin - pd.Timedelta(hours=1), run, 2),
                (origin + pd.Timedelta(hours=3), origin - pd.Timedelta(hours=1), run - pd.Timedelta(days=1), 88)):
            records.append(dict(station="07149", run=source, valid_time=valid, available_at=available,
                                variable=field, value=value))
    frame = pd.DataFrame(records)
    rows, report = spatial_rows(frame, batch, 2)
    assert np.array_equal(rows[0, :, 0], np.ones(4))
    assert np.isnan(rows[0, :, 1]).all()
    assert np.array_equal(rows[0, :, 2], np.full(4, 2))
    assert report["historical_observed_fraction"] == [.5] * 4
    with pytest.raises(ValueError, match="candidate unavailable"):
        spatial_rows(frame[frame.run.ne(run)], batch, 2)
