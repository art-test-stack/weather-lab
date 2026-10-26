import numpy as np
import pytest

from weather_lab.practical_grid import bilinear_cells, land_cell, weighted


def test_planar_interpolation_and_land_distance_bound():
    grid = dict(Ni=3, Nj=3, latitudeOfFirstGridPointInDegrees=48., longitudeOfFirstGridPointInDegrees=2.,
                iDirectionIncrementInDegrees=.025, jDirectionIncrementInDegrees=.025, jScansPositively=1)
    values = np.array([[0., 1., 2.], [2., 3., 4.], [4., 5., 6.]])
    assert weighted(values, bilinear_cells(grid, 48.0125, 2.0125)) == pytest.approx(1.5)
    mask = np.zeros((3, 3)); mask[0, 1] = 1
    assert land_cell(grid, mask, 48., 2.) == [(0, 1, 1.)]
    assert land_cell(grid, np.zeros((3, 3)), 48., 2.) == []
    assert np.isnan(weighted(values, []))
    assert bilinear_cells(grid, 49., 2.) == []
