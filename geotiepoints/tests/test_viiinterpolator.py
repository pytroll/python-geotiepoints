#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2021 Python-geotiepoints developers
#
# This file is part of python-geotiepoints.
#
# python-geotiepoints is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# python-geotiepoints is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# python-geotiepoints.  If not, see <http://www.gnu.org/licenses/>.
"""Test of the interpolation of geographical tiepoints for the VII products.

It follows the description provided in document "EPS-SG VII Level 1B Product Format Specification".

"""

import unittest
import numpy as np
import xarray as xr
from geotiepoints.viiinterpolator import tie_points_interpolation, tie_points_geo_interpolation


TEST_N_SCANS = 2
TEST_TIE_POINTS_FACTOR = 2
TEST_SCAN_ALT_TIE_POINTS = 3
TEST_VALID_ALT_TIE_POINTS = TEST_SCAN_ALT_TIE_POINTS * TEST_N_SCANS
TEST_INVALID_ALT_TIE_POINTS = TEST_SCAN_ALT_TIE_POINTS * TEST_N_SCANS + 1
TEST_ACT_TIE_POINTS = 4

# Results of latitude/longitude interpolation with simple interpolation on coordinates
TEST_LON_1 = np.array(
    [[-12., -11.5, -11., -10.5, -9., -8.5, -8., -7.5],
     [-9., -8.5, -8., -7.5, -6., -5.5, -5., -4.5],
     [-6., -5.5, -5., -4.5, -3., -2.5, -2., -1.5],
     [-3., -2.5, -2., -1.5, 0., 0.5, 1., 1.5],
     [0., 0.5, 1., 1.5, 3., 3.5, 4., 4.5],
     [3., 3.5, 4., 4.5, 6., 6.5, 7., 7.5]]
)
TEST_LAT_1 = np.array(
    [[0., 0.5, 1., 1.5, 3., 3.5, 4., 4.5],
     [3., 3.5, 4., 4.5, 6., 6.5, 7., 7.5],
     [6., 6.5, 7., 7.5, 9., 9.5, 10., 10.5],
     [9., 9.5, 10., 10.5, 12., 12.5, 13., 13.5],
     [12., 12.5, 13., 13.5, 15., 15.5, 16., 16.5],
     [15., 15.5, 16., 16.5, 18., 18.5, 19., 19.5]]
)

# Results of latitude/longitude interpolation on cartesian coordinates (latitude above 60 degrees)
TEST_LON_2 = np.array(
    [[-12., -11.50003808, -11., -10.50011426, -9., -8.50026689, -8., -7.50034342],
     [-9.00824726, -8.50989187, -8.01100418, -7.51272848, -6.01653996, -5.51842688, -5.01932226, -4.52129225],
     [-6., -5.50049716, -5., -4.50057447, -3., -2.50073021, -2., -1.50080874],
     [-3.02492451, -2.52706443, -2.02774808, -1.52997501, -0.03344942, 0.46414517, 0.96366893, 1.4611719],
     [0., 0.49903263, 1., 1.49895241, 3., 3.49878988, 4., 4.49870746],
     [2.9578336, 3.45514812, 3.9548757, 4.4520932, 5.94886832, 6.44588569, 6.94581415, 7.44272818]]
)

TEST_LAT_2 = np.array(
    [[0., 0.49998096, 1., 1.49994287, 3., 3.49986656, 4., 4.4998283],
     [2.99588485, 3.49506416, 3.99450923, 4.49364876, 5.99174708, 6.49080542, 6.99035883, 7.4893757],
     [6., 6.49975143, 7., 7.49971278, 9., 9.49963492, 10., 10.49959566],
     [8.98756357, 9.48649563, 9.98615477, 10.4850434, 11.98331018, 12.48210974, 12.98187246, 13.4806263],
     [12., 12.49951634, 13., 13.49947623, 15., 15.49939498, 16., 16.49935377],
     [14.97896116, 15.47762097, 15.97748548, 16.47609689, 17.97448854, 18.47300011, 18.97296495, 19.47142496]]
)

# Results of latitude/longitude interpolation on cartesian coordinates (longitude with a 360 degrees step)
TEST_LON_3 = np.array(
    [[-12., -11.50444038, -11., -10.50459822, -9., -8.50493209, -8., -7.50510905],
     [-9.17477341, -8.68280267, -8.18102962, -7.68936161, -6.19433153, -5.70332248, -5.20141997, -4.71077058],
     [-6., -5.50548573, -5., -4.50568668, -3., -2.50611746, -2., -1.506349],
     [-3.2165963, -2.72673687, -2.22474246, -1.73531828, -0.24232275, 0.24613534, 0.7481615, 1.23608137],
     [0., 0.49315061, 1., 1.49287934, 3., 3.49228746, 4., 4.49196335],
     [2.72743411, 3.21414435, 3.71610443, 4.20213182, 5.69115252, 6.17562189, 6.67735289, 7.16092853]]
)

TEST_LAT_3 = np.array(
    [[45., 45.49777998, 46., 46.49770107, 48., 48.49753416, 49., 49.49744569],
     [47.91286617, 48.40886652, 48.90975264, 49.40560282, 50.90313463, 51.39865815, 51.8996091, 52.39495445],
     [51., 51.49725738, 52., 52.49715691, 54., 54.50311196, 55., 55.50299846],
     [54.11364234, 54.61463583, 55.10950687, 55.61043728, 57.10152529, 57.60232834, 58.09766771, 58.59840655],
     [57., 57.50277937, 58., 58.50267347, 60., 60.50246826, 61., 61.5023687],
     [60.09019289, 60.59080228, 61.0865661, 61.58711028, 63.07951189, 63.57992468, 64.07607638, 64.57642295]]
)


class TestViiInterpolator(unittest.TestCase):
    """Test the vii_utils module."""

    def setUp(self):
        """Set up the test."""
        # Create the arrays for the interpolation test
        # The first has a valid number of n_tie_alt points (multiple of SCAN_ALT_TIE_POINTS)
        self.valid_data_for_interpolation = xr.DataArray(
            np.arange(
                TEST_VALID_ALT_TIE_POINTS * TEST_ACT_TIE_POINTS,
                dtype=np.float64,
            ).reshape(TEST_ACT_TIE_POINTS, TEST_VALID_ALT_TIE_POINTS),
            dims=('num_tie_points_act', 'num_tie_points_alt'),
        )
        # The second has an invalid number of n_tie_alt points (not multiple of SCAN_ALT_TIE_POINTS)
        self.invalid_data_for_interpolation = xr.DataArray(
            np.arange(
                TEST_INVALID_ALT_TIE_POINTS * TEST_ACT_TIE_POINTS,
                dtype=np.float64,
            ).reshape(TEST_ACT_TIE_POINTS, TEST_INVALID_ALT_TIE_POINTS),
            dims=('num_tie_points_act', 'num_tie_points_alt'),
        )
        # Then two arrays containing valid longitude and latitude data
        self.longitude = xr.DataArray(
            np.linspace(
                -12,
                11,
                num=TEST_VALID_ALT_TIE_POINTS * TEST_ACT_TIE_POINTS,
                dtype=np.float64,
            ).reshape(TEST_ACT_TIE_POINTS, TEST_VALID_ALT_TIE_POINTS),
            dims=('num_tie_points_act', 'num_tie_points_alt'),
        )
        self.latitude = xr.DataArray(
            np.linspace(
                0,
                23,
                num=TEST_VALID_ALT_TIE_POINTS * TEST_ACT_TIE_POINTS,
                dtype=np.float64,
            ).reshape(TEST_ACT_TIE_POINTS, TEST_VALID_ALT_TIE_POINTS),
            dims=('num_tie_points_act', 'num_tie_points_alt'),
        )
        # Then one containing latitude data above 60 degrees
        self.latitude_over60 = xr.DataArray(
            np.linspace(
                45,
                68,
                num=TEST_VALID_ALT_TIE_POINTS * TEST_ACT_TIE_POINTS,
                dtype=np.float64,
            ).reshape(TEST_ACT_TIE_POINTS, TEST_VALID_ALT_TIE_POINTS),
            dims=('num_tie_points_act', 'num_tie_points_alt'),
        )
        # Then one containing longitude data with a 360 degrees step
        self.longitude_over360 = xr.DataArray(
            np.linspace(
                -12,
                11,
                num=TEST_VALID_ALT_TIE_POINTS * TEST_ACT_TIE_POINTS,
                dtype=np.float64,
            ).reshape(TEST_ACT_TIE_POINTS, TEST_VALID_ALT_TIE_POINTS) % 360.,
            dims=('num_tie_points_act', 'num_tie_points_alt'),
        )

    def tearDown(self):
        """Tear down the test."""
        # Nothing to do
        pass

    def test_tie_points_interpolation(self):
        """# Test the interpolation routine with valid and invalid input."""
        # Test the interpolation routine with valid input
        result_valid = tie_points_interpolation(
            [self.valid_data_for_interpolation],
            TEST_SCAN_ALT_TIE_POINTS,
            TEST_TIE_POINTS_FACTOR
        )[0]

        act_points_interp = (TEST_ACT_TIE_POINTS - 1) * TEST_TIE_POINTS_FACTOR
        num_scans = TEST_VALID_ALT_TIE_POINTS // TEST_SCAN_ALT_TIE_POINTS
        scan_alt_points_interp = (TEST_SCAN_ALT_TIE_POINTS - 1) * TEST_TIE_POINTS_FACTOR

        # It is easier to check the delta between interpolated points, which must be 1/8 of the original delta
        # Across the track, it is possible to check the delta on the entire array
        delta_axis_0 = 1.0 * TEST_VALID_ALT_TIE_POINTS / TEST_TIE_POINTS_FACTOR
        self.assertTrue(np.allclose(
            np.diff(result_valid, axis=0),
            np.ones((act_points_interp - 1, num_scans * scan_alt_points_interp)) * delta_axis_0
            ))

        delta_axis_1 = 1.0 / TEST_TIE_POINTS_FACTOR
        # Along the track, it is necessary to check the delta on each scan separately
        for i in range(num_scans):
            first_index = i*(TEST_SCAN_ALT_TIE_POINTS-1)*TEST_TIE_POINTS_FACTOR
            last_index = (i+1)*(TEST_SCAN_ALT_TIE_POINTS-1)*TEST_TIE_POINTS_FACTOR
            result_per_scan = result_valid[:, first_index:last_index]
            self.assertTrue(np.allclose(
                np.diff(result_per_scan, axis=1),
                np.ones((act_points_interp, (TEST_SCAN_ALT_TIE_POINTS-1)*TEST_TIE_POINTS_FACTOR - 1)) * delta_axis_1
                ))

        self.assertEqual(len(result_valid.coords), 0)

        # Test the interpolation routine with invalid input
        with self.assertRaises(ValueError):
            tie_points_interpolation(
                [self.invalid_data_for_interpolation],
                TEST_SCAN_ALT_TIE_POINTS,
                TEST_TIE_POINTS_FACTOR
            )[0]

    def test_tie_points_geo_interpolation(self):
        """# Test the coordinates interpolation routine with valid and invalid input."""
        # Test the interpolation routine with valid input
        lon, lat = tie_points_geo_interpolation(
            self.longitude,
            self.latitude,
            TEST_SCAN_ALT_TIE_POINTS,
            TEST_TIE_POINTS_FACTOR
        )
        self.assertTrue(np.allclose(lon, TEST_LON_1))
        self.assertTrue(np.allclose(lat, TEST_LAT_1))

        lon, lat = tie_points_geo_interpolation(
            self.longitude_over360,
            self.latitude,
            TEST_SCAN_ALT_TIE_POINTS,
            TEST_TIE_POINTS_FACTOR
        )
        self.assertTrue(np.allclose(lon, TEST_LON_2))
        self.assertTrue(np.allclose(lat, TEST_LAT_2))

        lon, lat = tie_points_geo_interpolation(
            self.longitude,
            self.latitude_over60,
            TEST_SCAN_ALT_TIE_POINTS,
            TEST_TIE_POINTS_FACTOR
        )
        self.assertTrue(np.allclose(lon, TEST_LON_3))
        self.assertTrue(np.allclose(lat, TEST_LAT_3))

        # Test the interpolation routine with invalid input (different dimensions of the two arrays)
        with self.assertRaises(ValueError):
            tie_points_geo_interpolation(
                self.longitude,
                self.invalid_data_for_interpolation,
                TEST_SCAN_ALT_TIE_POINTS,
                TEST_TIE_POINTS_FACTOR
            )
