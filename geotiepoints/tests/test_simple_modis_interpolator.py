#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Python-geotiepoints developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Tests for simple MODIS interpolators."""

import numpy as np
import pytest
import dask
import dask.array as da

from geotiepoints.simple_modis_interpolator import modis_1km_to_250m, modis_1km_to_500m
from .test_modisinterpolator import (
    assert_geodetic_distance,
    load_1km_lonlat_as_xarray_dask,
    load_1km_lonlat_as_dask,
    load_1km_lonlat_as_numpy,
    load_500m_lonlat_expected_as_xarray_dask,
    load_250m_lonlat_expected_as_xarray_dask,
)

from .utils import CustomScheduler


@pytest.mark.parametrize(
    ("input_func", "exp_func", "interp_func", "dist_max"),
    [
        (load_1km_lonlat_as_xarray_dask, load_500m_lonlat_expected_as_xarray_dask, modis_1km_to_500m, 16),
        (load_1km_lonlat_as_xarray_dask, load_250m_lonlat_expected_as_xarray_dask, modis_1km_to_250m, 27.35),
        (load_1km_lonlat_as_dask, load_500m_lonlat_expected_as_xarray_dask, modis_1km_to_500m, 16),
        (load_1km_lonlat_as_dask, load_250m_lonlat_expected_as_xarray_dask, modis_1km_to_250m, 27.35),
        (load_1km_lonlat_as_numpy, load_500m_lonlat_expected_as_xarray_dask, modis_1km_to_500m, 16),
        (load_1km_lonlat_as_numpy, load_250m_lonlat_expected_as_xarray_dask, modis_1km_to_250m, 27.35),
    ]
)
def test_basic_interp(input_func, exp_func, interp_func, dist_max):
    lon1, lat1 = input_func()
    lons_exp, lats_exp = exp_func()

    # when working with dask arrays, we shouldn't compute anything
    with dask.config.set(scheduler=CustomScheduler(0)):
        lons, lats = interp_func(lon1, lat1)

    if hasattr(lons, "compute"):
        lons, lats = da.compute(lons, lats)
    assert_geodetic_distance(lons, lats, lons_exp, lats_exp, dist_max)
    assert not np.any(np.isnan(lons))
    assert not np.any(np.isnan(lats))


def test_nonstandard_scan_size():
    lon1, lat1 = load_1km_lonlat_as_xarray_dask()
    # remove 1 row from the end
    lon1 = lon1[:-1]
    lat1 = lat1[:-1]

    pytest.raises(ValueError, modis_1km_to_250m, lon1, lat1)
