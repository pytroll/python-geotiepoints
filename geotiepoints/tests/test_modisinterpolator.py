#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2022 Python-geotiepoints developers
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
"""Tests for MODIS interpolators."""

import numpy as np
from pyproj import Geod
import h5py
import os
import dask.array as da
import xarray as xr
from geotiepoints.modisinterpolator import (modis_1km_to_250m,
                                            modis_1km_to_500m,
                                            modis_5km_to_1km,
                                            modis_5km_to_500m,
                                            modis_5km_to_250m)
FILENAME_DATA = os.path.join(
    os.path.dirname(__file__), '../../testdata/modis_test_data.h5')


def _to_dask(arr):
    return da.from_array(arr, chunks=4096)


def _to_da(arr):
    return xr.DataArray(_to_dask(arr), dims=['y', 'x'])


def _load_h5_geo_vars(*var_names):
    h5f = h5py.File(FILENAME_DATA, 'r')
    return tuple(h5f[var_name] for var_name in var_names)


def load_1km_lonlat_as_numpy():
    lon1, lat1 = _load_h5_geo_vars('lon_1km', 'lat_1km')
    return lon1[:], lat1[:]


def load_1km_lonlat_as_dask():
    lon1, lat1 = _load_h5_geo_vars('lon_1km', 'lat_1km')
    return _to_dask(lon1), _to_dask(lat1)


def load_1km_lonlat_as_xarray_dask():
    lon1, lat1 = _load_h5_geo_vars('lon_1km', 'lat_1km')
    return _to_da(lon1), _to_da(lat1)


def load_1km_lonlat_satz_as_xarray_dask():
    lon1, lat1, satz1 = _load_h5_geo_vars('lon_1km', 'lat_1km', 'satz_1km')
    return _to_da(lon1), _to_da(lat1), _to_da(satz1)


def load_500m_lonlat_expected_as_xarray_dask():
    h5f = h5py.File(FILENAME_DATA, 'r')
    lon500 = _to_da(h5f['lon_500m'])
    lat500 = _to_da(h5f['lat_500m'])
    return lon500, lat500


def load_250m_lonlat_expected_as_xarray_dask():
    h5f = h5py.File(FILENAME_DATA, 'r')
    lon250 = _to_da(h5f['lon_250m'])
    lat250 = _to_da(h5f['lat_250m'])
    return lon250, lat250


def assert_geodetic_distance(
        lons_actual: np.ndarray,
        lats_actual: np.ndarray,
        lons_desired: np.ndarray,
        lats_desired: np.ndarray,
        max_distance_diff: float,
) -> None:
    """Check that the geodetic distance between two sets of coordinates is smaller than a threshold.

    Args:
        lons_actual: Longitude array produced by interpolation being tested.
        lats_actual: Latitude array produced by interpolation being tested.
        lons_desired: Longitude array of expected/truth coordinates.
        lats_desired: Latitude array of expected/truth coordinates.
        max_distance_diff: Limit of allowed distance difference in meters.

    """
    g = Geod(ellps="WGS84")
    _, _, dist = g.inv(lons_actual, lats_actual, lons_desired, lats_desired)
    print(dist.min(), dist.max())
    np.testing.assert_array_less(dist, max_distance_diff)  # meters


class TestModisInterpolator:
    def test_modis(self):
        lon1, lat1, satz1 = load_1km_lonlat_satz_as_xarray_dask()
        lon500, lat500 = load_500m_lonlat_expected_as_xarray_dask()
        lon250, lat250 = load_250m_lonlat_expected_as_xarray_dask()

        lons, lats = modis_1km_to_250m(lon1, lat1, satz1)
        assert_geodetic_distance(lons, lats, lon250, lat250, 1)

        lons, lats = modis_1km_to_500m(lon1, lat1, satz1)
        assert_geodetic_distance(lons, lats, lon500, lat500, 1)

        lat5 = lat1[2::5, 2::5]
        lon5 = lon1[2::5, 2::5]
        satz5 = satz1[2::5, 2::5]
        lons, lats = modis_5km_to_1km(lon5, lat5, satz5)
        assert_geodetic_distance(lons, lats, lon1, lat1, 25)

        # 5km to 500m
        lons, lats = modis_5km_to_500m(lon5, lat5, satz5)
        assert lons.shape == lon500.shape
        assert lats.shape == lat500.shape
        # self.assertTrue(np.allclose(lon500, lons, atol=1e-2))
        # self.assertTrue(np.allclose(lat500, lats, atol=1e-2))

        # 5km to 250m
        lons, lats = modis_5km_to_250m(lon5, lat5, satz5)
        assert lons.shape == lon250.shape
        assert lats.shape == lat250.shape
        # self.assertTrue(np.allclose(lon250, lons, atol=1e-2))
        # self.assertTrue(np.allclose(lat250, lats, atol=1e-2))

        # Test level 2
        lat5 = lat1[2::5, 2:-5:5]
        lon5 = lon1[2::5, 2:-5:5]
        satz5 = satz1[2::5, 2:-5:5]
        lons, lats = modis_5km_to_1km(lon5, lat5, satz5)
        assert_geodetic_distance(lons, lats, lon1, lat1, 106.0)

        # Test nans issue (#19)
        satz1 = _to_da(abs(np.linspace(-65.4, 65.4, 1354)).repeat(20).reshape(-1, 20).T)
        lons, lats = modis_1km_to_500m(lon1, lat1, satz1)
        assert not np.any(np.isnan(lons.compute()))
        assert not np.any(np.isnan(lats.compute()))

    def test_poles_datum(self):
        import xarray as xr
        h5f = h5py.File(FILENAME_DATA, 'r')
        orig_lon = _to_da(h5f['lon_1km'])
        lon1 = orig_lon + 180
        lon1 = xr.where(lon1 > 180, lon1 - 360, lon1)
        lat1 = _to_da(h5f['lat_1km'])
        satz1 = _to_da(h5f['satz_1km'])

        lat5 = lat1[2::5, 2::5]
        lon5 = lon1[2::5, 2::5]
        satz5 = satz1[2::5, 2::5]
        lons, lats = modis_5km_to_1km(lon5, lat5, satz5)

        lons = lons + 180
        lons = xr.where(lons > 180, lons - 360, lons)
        assert_geodetic_distance(lons, lats, orig_lon, lat1, 25.0)
