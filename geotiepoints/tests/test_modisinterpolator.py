#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 Martin Raspaud

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import numpy as np
import h5py
import os
from geotiepoints.modisinterpolator import (modis_1km_to_250m,
                                            modis_1km_to_500m,
                                            modis_5km_to_1km,
                                            modis_5km_to_500m,
                                            modis_5km_to_250m)
FILENAME_DATA = os.path.join(
    os.path.dirname(__file__), '../../testdata/modis_test_data.h5')

def to_da(arr):
    import xarray as xr
    import dask.array as da

    return xr.DataArray(da.from_array(arr, chunks=4096), dims=['y', 'x'])


class TestModisInterpolator(unittest.TestCase):
    def test_modis(self):
        h5f = h5py.File(FILENAME_DATA, 'r')
        lon1 = to_da(h5f['lon_1km'])
        lat1 = to_da(h5f['lat_1km'])
        satz1 = to_da(h5f['satz_1km'])

        lon250 = to_da(h5f['lon_250m'])
        lon500 = to_da(h5f['lon_500m'])

        lat250 = to_da(h5f['lat_250m'])
        lat500 = to_da(h5f['lat_500m'])

        lons, lats = modis_1km_to_250m(lon1, lat1, satz1)
        self.assertTrue(np.allclose(lon250, lons, atol=1e-2))
        self.assertTrue(np.allclose(lat250, lats, atol=1e-2))

        lons, lats = modis_1km_to_500m(lon1, lat1, satz1)
        self.assertTrue(np.allclose(lon500, lons, atol=1e-2))
        self.assertTrue(np.allclose(lat500, lats, atol=1e-2))

        lat5 = lat1[2::5, 2::5]
        lon5 = lon1[2::5, 2::5]

        satz5 = satz1[2::5, 2::5]
        lons, lats = modis_5km_to_1km(lon5, lat5, satz5)
        self.assertTrue(np.allclose(lon1, lons, atol=1e-2))
        self.assertTrue(np.allclose(lat1, lats, atol=1e-2))

        # 5km to 500m
        lons, lats = modis_5km_to_500m(lon5, lat5, satz5)
        self.assertEqual(lon500.shape, lons.shape)
        self.assertEqual(lat500.shape, lats.shape)
        # self.assertTrue(np.allclose(lon500, lons, atol=1e-2))
        # self.assertTrue(np.allclose(lat500, lats, atol=1e-2))

        # 5km to 250m
        lons, lats = modis_5km_to_250m(lon5, lat5, satz5)
        self.assertEqual(lon250.shape, lons.shape)
        self.assertEqual(lat250.shape, lats.shape)
        # self.assertTrue(np.allclose(lon250, lons, atol=1e-2))
        # self.assertTrue(np.allclose(lat250, lats, atol=1e-2))

        # Test level 2
        lat5 = lat1[2::5, 2:-5:5]
        lon5 = lon1[2::5, 2:-5:5]

        satz5 = satz1[2::5, 2:-5:5]
        lons, lats = modis_5km_to_1km(lon5, lat5, satz5)
        self.assertTrue(np.allclose(lon1, lons, atol=1e-2))
        self.assertTrue(np.allclose(lat1, lats, atol=1e-2))

    def test_poles_datum(self):
        import xarray as xr
        h5f = h5py.File(FILENAME_DATA, 'r')
        orig_lon = to_da(h5f['lon_1km'])
        lon1 = orig_lon + 180
        lon1 = xr.where(lon1 > 180, lon1 - 360, lon1)
        lat1 = to_da(h5f['lat_1km'])
        satz1 = to_da(h5f['satz_1km'])

        lat5 = lat1[2::5, 2::5]
        lon5 = lon1[2::5, 2::5]

        satz5 = satz1[2::5, 2::5]
        lons, lats = modis_5km_to_1km(lon5, lat5, satz5)
        lons = lons + 180
        lons = xr.where(lons > 180, lons - 360, lons)
        self.assertTrue(np.allclose(orig_lon, lons, atol=1e-2))
        self.assertTrue(np.allclose(lat1, lats, atol=1e-2))


def suite():
    """The suite for MODIS"""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestModisInterpolator))

    return mysuite

if __name__ == "__main__":
    unittest.main()
