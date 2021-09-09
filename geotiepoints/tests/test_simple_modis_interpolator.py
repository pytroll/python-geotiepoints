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

import os
import numpy as np
import pytest
import h5py
import dask
import dask.array as da
import xarray as xr

from geotiepoints.simple_modis_interpolator import modis_1km_to_250m, modis_1km_to_500m
from .utils import CustomScheduler

FILENAME_DATA = os.path.join(
    os.path.dirname(__file__), '../../testdata/modis_test_data.h5')


def _to_dask(arr):
    return da.from_array(arr, chunks=4096)


def _to_da(arr):
    return xr.DataArray(_to_dask(arr), dims=['y', 'x'])


def _load_h5_lonlat_vars(lon_var, lat_var):
    h5f = h5py.File(FILENAME_DATA, 'r')
    lon1 = h5f[lon_var]
    lat1 = h5f[lat_var]
    return lon1, lat1


def _load_1km_lonlat_as_numpy():
    lon1, lat1 = _load_h5_lonlat_vars('lon_1km', 'lat_1km')
    return lon1[:], lat1[:]


def _load_1km_lonlat_as_dask():
    lon1, lat1 = _load_h5_lonlat_vars('lon_1km', 'lat_1km')
    return _to_dask(lon1), _to_dask(lat1)


def _load_1km_lonlat_as_xarray_dask():
    lon1, lat1 = _load_h5_lonlat_vars('lon_1km', 'lat_1km')
    return _to_da(lon1), _to_da(lat1)


def _load_500m_lonlat_expected_as_xarray_dask():
    h5f = h5py.File(FILENAME_DATA, 'r')
    lon500 = _to_da(h5f['lon_500m'])
    lat500 = _to_da(h5f['lat_500m'])
    return lon500, lat500


def _load_250m_lonlat_expected_as_xarray_dask():
    h5f = h5py.File(FILENAME_DATA, 'r')
    lon250 = _to_da(h5f['lon_250m'])
    lat250 = _to_da(h5f['lat_250m'])
    return lon250, lat250


@pytest.mark.parametrize(
    ("input_func", "exp_func", "interp_func"),
    [
        (_load_1km_lonlat_as_xarray_dask, _load_500m_lonlat_expected_as_xarray_dask, modis_1km_to_500m),
        (_load_1km_lonlat_as_xarray_dask, _load_250m_lonlat_expected_as_xarray_dask, modis_1km_to_250m),
        (_load_1km_lonlat_as_dask, _load_500m_lonlat_expected_as_xarray_dask, modis_1km_to_500m),
        (_load_1km_lonlat_as_dask, _load_250m_lonlat_expected_as_xarray_dask, modis_1km_to_250m),
        (_load_1km_lonlat_as_numpy, _load_500m_lonlat_expected_as_xarray_dask, modis_1km_to_500m),
        (_load_1km_lonlat_as_numpy, _load_250m_lonlat_expected_as_xarray_dask, modis_1km_to_250m),
    ]
)
def test_basic_interp(input_func, exp_func, interp_func):
    lon1, lat1 = input_func()
    lons_exp, lats_exp = exp_func()

    # when working with dask arrays, we shouldn't compute anything
    with dask.config.set(scheduler=CustomScheduler(0)):
        lons, lats = interp_func(lon1, lat1)

    if hasattr(lons, "compute"):
        lons, lats = da.compute(lons, lats)
    # our "truth" values are from the modisinterpolator results
    atol = 0.038  # 1e-2
    rtol = 0
    np.testing.assert_allclose(lons_exp, lons, atol=atol, rtol=rtol)
    np.testing.assert_allclose(lats_exp, lats, atol=atol, rtol=rtol)
    assert not np.any(np.isnan(lons))
    assert not np.any(np.isnan(lats))


def test_nonstandard_scan_size():
    lon1, lat1 = _load_1km_lonlat_as_xarray_dask()
    # remove 1 row from the end
    lon1 = lon1[:-1]
    lat1 = lat1[:-1]

    pytest.raises(ValueError, modis_1km_to_250m, lon1, lat1)


# def test_poles_datum(self):
#     import xarray as xr
#     h5f = h5py.File(FILENAME_DATA, 'r')
#     orig_lon = _to_da(h5f['lon_1km'])
#     lon1 = orig_lon + 180
#     lon1 = xr.where(lon1 > 180, lon1 - 360, lon1)
#     lat1 = _to_da(h5f['lat_1km'])
#     satz1 = _to_da(h5f['satz_1km'])
#
#     lat5 = lat1[2::5, 2::5]
#     lon5 = lon1[2::5, 2::5]
#
#     satz5 = satz1[2::5, 2::5]
#     lons, lats = modis_5km_to_1km(lon5, lat5, satz5)
#     lons = lons + 180
#     lons = xr.where(lons > 180, lons - 360, lons)
#     np.testing.assert_allclose(orig_lon, lons, atol=1e-2)
#     np.testing.assert_allclose(lat1, lats, atol=1e-2)
