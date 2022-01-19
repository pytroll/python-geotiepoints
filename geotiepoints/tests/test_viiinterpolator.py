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

It follows the description provided in document "EPS-SG VII Level 1B Product Format Specification V4A".
This version is compatible for vii (METimage) test data version V2 (Jan 2022). It is not back compatible
with V1.

"""

import unittest
import numpy as np
import xarray as xr
import pytest
from geotiepoints.viiinterpolator import tie_points_interpolation, tie_points_geo_interpolation


TEST_N_SCANS = 2
TEST_TIE_POINTS_FACTOR = 2
TEST_SCAN_ALT_TIE_POINTS = 3
TEST_VALID_ALT_TIE_POINTS = TEST_SCAN_ALT_TIE_POINTS * TEST_N_SCANS
TEST_INVALID_ALT_TIE_POINTS = TEST_SCAN_ALT_TIE_POINTS * TEST_N_SCANS + 1
TEST_ACT_TIE_POINTS = 4

# Results of latitude/longitude interpolation with simple interpolation on coordinates
TEST_LON_1 = np.array(
    [[-12., -11.5, -11., -10.5, -10.0, -9.5],
     [-10., -9.5, -9., -8.5, -8.0, -7.5],
     [-8., -7.5, -7., -6.5, -6.0, -5.5],
     [-6., -5.5, -5., -4.5, -4.0, -3.5],
     [0., 0.5, 1., 1.5, 2.0, 2.5],
     [2., 2.5, 3., 3.5, 4.0, 4.5],
     [4., 4.5, 5., 5.5, 6.0, 6.5],
     [6., 6.5, 7., 7.5, 8.0, 8.5]]
)
TEST_LAT_1 = np.array(
    [[0., 0.5, 1., 1.5, 2., 2.5],
     [2., 2.5, 3., 3.5, 4., 4.5],
     [4., 4.5, 5., 5.5, 6., 6.5],
     [6., 6.5, 7., 7.5, 8., 8.5],
     [12., 12.5, 13., 13.5, 14.0, 14.5],
     [14., 14.5, 15., 15.5, 16., 16.5],
     [16., 16.5, 17., 17.5, 18., 18.5],
     [18., 18.5, 19., 19.5, 20., 20.5]]
)

# Results of latitude/longitude interpolation on cartesian coordinates (latitude above 60 degrees)
TEST_LON_2 = np.array(
    [[-12., -11.50003808, -11., -10.50011426, -10., -9.50019052],
     [-10.00243991, -9.5032411, -9.00366173, -8.50454031, -8.00488578, -7.5058423],
     [-8., -7.50034342, -7., -6.50042016, -6., -5.50049716],
     [-6.00734362, -5.50845783, -5.00857895, -4.50977302, -4.00981958, -3.51109426],
     [0., 0.49903263, 1., 1.49895241, 2., 2.49887151],
     [1.98257947, 2.48080192, 2.98127841, 3.47941324, 3.97996512, 4.47801105],
     [4., 4.49870746, 5., 5.49862418, 6., 6.49853998],
     [5.97729789, 6.47516183, 6.97594186, 7.47371256, 7.97456943, 8.47224525]]
)

TEST_LAT_2 = np.array(
    [[0., 0.49998096, 1., 1.49994287, 2., 2.49990475],
     [1.99878116, 2.49838091, 2.99817081, 3.49773189, 3.99755935, 4.49708148],
     [4., 4.4998283, 5., 5.49978993, 6., 6.49975143],
     [5.99633155, 6.4957749, 6.99571447, 7.49511791, 7.99509473, 8.49445789],
     [12., 12.49951634, 13., 13.49947623, 14., 14.499435779],
     [13.99129786, 14.4904098, 14.99064796, 15.48971613, 15.98999196, 16.48901572],
     [16., 16.49935377, 17., 17.49931213, 18., 18.49927003],
     [17.98865968, 18.48759253, 18.98798235, 19.48686863, 19.98729684, 20.48613573]]
)

# Results of latitude/longitude interpolation on cartesian coordinates (longitude with a 360 degrees step)
TEST_LON_3 = np.array(
    [[-12., -11.50444038, -11., -10.50459822, -10., -9.50476197],
     [-10.07492627, -9.58101155, -9.07759836, -8.5839056, - 8.0803761, -7.58691614],
     [-8., -7.50510905, -7., -6.5052934, -6., -5.50548573],
     [-6.0862821, -5.59332416, -5.08942935, -4.59674283, -4.09272043, -3.60032066],
     [0., 0.49315061, 1., 1.49287934, 2., 2.49259217],
     [1.88371709, 2.3739768, 2.87898193, 3.36879304, 3.87395153, 4.36327924],
     [4., 4.49196335, 5., 5.49161771, 6., 6.49124808],
     [5.86287282, 6.35111105, 6.85674573, 7.3443667, 7.85016382, 8.33710998]]
)

TEST_LAT_3 = np.array(
    [[45., 45.49777998, 46., 46.49770107, 47., 47.4976192],
     [46.96258417, 47.4595462, 47.9612508, 48.4581022, 48.95986481, 49.45660021],
     [49., 49.49744569, 50., 50.49735352, 51., 51.49725738],
     [50.95691833, 51.45340364, 51.95534841, 52.45169855, 52.95370691, 53.55477452],
     [57., 57.50277937, 58., 58.50267347, 59., 59.50256981],
     [59.04185095, 59.54357898, 60.04020844, 60.54185139, 61.03859839, 61.54015723],
     [61., 61.5023687, 62., 62.502271, 63., 63.50217506],
     [63.03546804, 63.53686131, 64.03394419, 64.53525587, 65.03244567, 65.53367647]]
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
            ).reshape(TEST_VALID_ALT_TIE_POINTS, TEST_ACT_TIE_POINTS),
            dims=('num_tie_points_alt', 'num_tie_points_act'),
        )
        # The second has an invalid number of n_tie_alt points (not multiple of SCAN_ALT_TIE_POINTS)
        self.invalid_data_for_interpolation = xr.DataArray(
            np.arange(
                TEST_INVALID_ALT_TIE_POINTS * TEST_ACT_TIE_POINTS,
                dtype=np.float64,
            ).reshape(TEST_INVALID_ALT_TIE_POINTS, TEST_ACT_TIE_POINTS),
            dims=('num_tie_points_alt', 'num_tie_points_act'),
        )
        # Then two arrays containing valid longitude and latitude data
        self.longitude = xr.DataArray(
            np.linspace(
                -12,
                11,
                num=TEST_VALID_ALT_TIE_POINTS * TEST_ACT_TIE_POINTS,
                dtype=np.float64,
            ).reshape(TEST_VALID_ALT_TIE_POINTS, TEST_ACT_TIE_POINTS),
            dims=('num_tie_points_alt', 'num_tie_points_act'),
        )
        self.latitude = xr.DataArray(
            np.linspace(
                0,
                23,
                num=TEST_VALID_ALT_TIE_POINTS * TEST_ACT_TIE_POINTS,
                dtype=np.float64,
            ).reshape(TEST_VALID_ALT_TIE_POINTS, TEST_ACT_TIE_POINTS),
            dims=('num_tie_points_alt', 'num_tie_points_act'),
        )
        # Then one containing latitude data above 60 degrees
        self.latitude_over60 = xr.DataArray(
            np.linspace(
                45,
                68,
                num=TEST_VALID_ALT_TIE_POINTS * TEST_ACT_TIE_POINTS,
                dtype=np.float64,
            ).reshape(TEST_VALID_ALT_TIE_POINTS, TEST_ACT_TIE_POINTS),
            dims=('num_tie_points_alt', 'num_tie_points_act'),
        )
        # Then one containing longitude data with a 360 degrees step
        self.longitude_over360 = xr.DataArray(
            np.linspace(
                -12,
                11,
                num=TEST_VALID_ALT_TIE_POINTS * TEST_ACT_TIE_POINTS,
                dtype=np.float64,
            ).reshape(TEST_VALID_ALT_TIE_POINTS, TEST_ACT_TIE_POINTS) % 360.,
            dims=('num_tie_points_alt', 'num_tie_points_act'),
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

        # Across the track
        delta_axis_0 = [0., 0.5, 1., 1.5, 2., 2.5]
        self.assertTrue(np.allclose(result_valid[0, :], delta_axis_0))
        # Along track
        delta_axis_1 = [0., 2., 4., 6., 12., 14., 16., 18]
        self.assertTrue(np.allclose(result_valid[:, 0], delta_axis_1))

        # Test the interpolation routine with invalid input
        pytest.raises(ValueError, tie_points_interpolation,
                      [self.invalid_data_for_interpolation],
                      TEST_SCAN_ALT_TIE_POINTS,
                      TEST_TIE_POINTS_FACTOR)

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
