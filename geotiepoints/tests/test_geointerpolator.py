#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2013-2021 Python-geotiepoints developers
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
"""Tests for GeoInterpolator."""

import unittest

import numpy as np
import pytest
from pyresample.geometry import SwathDefinition

from geotiepoints.geointerpolator import GeoInterpolator, GeoGridInterpolator, GeoSplineInterpolator

TIES_EXP1 = np.array([[6384905.78040055, 6381081.08333225, 6371519.34066148,
                       6328950.00792935, 6253610.69157758, 6145946.19489936,
                       6124413.29556372],
                      [6377591.95940176, 6370997., 6354509.6014956,
                       6305151.62592155, 6223234.99818839, 6109277.14889072,
                       6086485.57903118],
                      [6359307.40690478, 6345786.79166939, 6311985.2535809,
                       6245655.67090206, 6147295.76471541, 6017604.5338691,
                       5991666.28769983],
                      [6351993.58590599, 6335702.70833714, 6294975.51441502,
                       6221857.28889426, 6116920.07132621, 5980935.48786045,
                       5953738.5711673],
                      [6338032.26190294, 6320348.4990906, 6276139.09205974,
                       6199670.56624433, 6091551.90273768, 5952590.38414781,
                       5924798.08042984],
                      [6290665.5946295, 6270385.16249031, 6219684.08214232,
                       6137100.75832981, 6023313.2794414, 5879194.72399075,
                       5850371.01290062],
                      [6172248.92644589, 6145476.82098957, 6078546.55734877,
                       5980676.23854351, 5852716.72120069, 5695705.57359808,
                       5664303.34407756],
                      [6124882.25917245, 6095513.48438928, 6022091.54743135,
                       5918106.430629, 5784478.09790441, 5622309.91344102,
                       5589876.27654834]])

TIES_EXP2 = np.array([[6372937.31273379, 6370997., 6366146.21816553,
                       6351605.98629588, 6327412.61244969, 6293626.50067273,
                       6286869.27831734],
                      [6353136.46335726, 6345786.79166939, 6327412.61244969,
                       6299445.69529922, 6261968.60390423, 6215087.60607344,
                       6205711.40650728]])

TIES_EXP4 = np.array([[6381081.08333225, 6381639.66045187, 6372470.10269454,
                       6353590.21586788, 6325042.05851245],
                      [6370997., 6366146.21816553, 6351605.98629588,
                       6327412.61244969, 6293626.50067273],
                      [6345786.79166939, 6327412.61244969, 6299445.69529922,
                       6261968.60390423, 6215087.60607344],
                      [6335702.70833714, 6311919.17016336, 6278581.57890056,
                       6235791.00048604, 6183672.04823372]])

TIES_EXP5 = np.array([[6381081.08333225, 6371519.34066148, 6328950.00792935,
                       6253610.69157758, 6145946.19489936],
                      [6370997., 6354509.6014956, 6305151.62592155,
                       6223234.99818839, 6109277.14889072],
                      [6345786.79166939, 6311985.2535809, 6245655.67090206,
                       6147295.76471541, 6017604.5338691],
                      [6270385.16249031, 6219684.08214232, 6137100.75832981,
                       6023313.2794414, 5879194.72399075],
                      [6145476.82098957, 6078546.55734877, 5980676.23854351,
                       5852716.72120069, 5695705.57359808],
                      [6095513.48438928, 6022091.54743135, 5918106.430629,
                       5784478.09790441, 5622309.91344102]])

TIES_EXP6 = np.array([[6381081.08333225, 6371519.34066148, 6328950.00792935,
                       6253610.69157758, 6145946.19489936],
                      [6370997., 6354509.6014956, 6305151.62592155,
                       6223234.99818839, 6109277.14889072],
                      [6345786.79166939, 6311985.2535809, 6245655.67090206,
                       6147295.76471541, 6017604.5338691],
                      [6335702.70833714, 6294975.51441502, 6221857.28889426,
                       6116920.07132621, 5980935.48786045],
                      [6320348.4990906, 6276139.09205974, 6199670.56624433,
                       6091551.90273768, 5952590.38414781],
                      [6270385.16249031, 6219684.08214232, 6137100.75832981,
                       6023313.2794414, 5879194.72399075],
                      [6145476.82098957, 6078546.55734877, 5980676.23854351,
                       5852716.72120069, 5695705.57359808],
                      [6095513.48438928, 6022091.54743135, 5918106.430629,
                       5784478.09790441, 5622309.91344102]])

TIES_EXP7 = np.array([[6372937.31273379, 6370997., 6366146.21816553,
                       6351605.98629588, 6327412.61244969, 6293626.50067273,
                       6286869.27831734],
                      [6353136.46335726, 6345786.79166939, 6327412.61244969,
                       6299445.69529922, 6261968.60390423, 6215087.60607344,
                       6205711.40650728]])


class TestGeoInterpolator(unittest.TestCase):
    """Class for unit testing the ancillary interpolation functions."""

    def test_fillborders(self):
        """Test filling borders."""
        lons = np.arange(20).reshape((4, 5), order="F")
        lats = np.arange(20).reshape((4, 5), order="C")
        lines = np.array([2, 7, 12, 17]) / 5.0
        cols = np.array([2, 7, 12, 17, 22])
        hlines = np.arange(20) / 5.0
        hcols = np.arange(24)
        satint = GeoInterpolator(
            (lons, lats), (lines, cols), (hlines, hcols), chunk_size=10)
        satint.fill_borders('x', 'y')

        np.testing.assert_allclose(satint.tie_data[0], TIES_EXP1)
        np.testing.assert_allclose(satint.row_indices, np.array(
            [0, 2, 7, 9, 10, 12, 17, 19]) / 5.0)
        self.assertTrue(
            np.allclose(satint.col_indices, np.array([0, 2, 7, 12, 17, 22, 23])))

    def test_extrapolate_cols(self):
        """Test extrapolating columns."""
        lons = np.arange(10).reshape((2, 5), order="F")
        lats = np.arange(10).reshape((2, 5), order="C")
        lines = np.array([2, 7])
        cols = np.array([2, 7, 12, 17, 22])
        hlines = np.arange(10)
        hcols = np.arange(24)
        satint = GeoInterpolator((lons, lats), (lines, cols), (hlines, hcols))

        self.assertTrue(np.allclose(satint._extrapolate_cols(satint.tie_data[0]),
                                    TIES_EXP2))

    def test_fill_col_borders(self):
        """Test filling the column borders."""
        lons = np.arange(10).reshape((2, 5), order="F")
        lats = np.arange(10).reshape((2, 5), order="C")
        lines = np.array([2, 7])
        cols = np.array([2, 7, 12, 17, 22])
        hlines = np.arange(10)
        hcols = np.arange(24)
        satint = GeoInterpolator((lons, lats), (lines, cols), (hlines, hcols))
        satint._fill_col_borders()
        np.testing.assert_allclose(satint.tie_data[0], TIES_EXP7)
        np.testing.assert_allclose(satint.col_indices,
                                   np.array([0, 2, 7, 12, 17, 22, 23]))

    def test_extrapolate_rows(self):
        """Test extrapolation of rows."""
        lons = np.arange(10).reshape((2, 5), order="F")
        lats = np.arange(10).reshape((2, 5), order="C")
        lines = np.array([2, 7])
        cols = np.array([2, 7, 12, 17, 22])
        hlines = np.arange(10)
        hcols = np.arange(24)
        satint = GeoInterpolator((lons, lats), (lines, cols), (hlines, hcols))
        np.testing.assert_allclose(satint._extrapolate_rows(satint.tie_data[0],
                                                            hlines, -0.4, 9.4),
                                   TIES_EXP4)

    def test_fill_row_borders(self):
        """Test filling the row borders."""
        lons = np.arange(20).reshape((4, 5), order="F")
        lats = np.arange(20).reshape((4, 5), order="C")
        lines = np.array([2, 7, 12, 17]) / 5.0
        cols = np.array([2, 7, 12, 17, 22])
        hlines = np.arange(20) / 5.0
        hcols = np.arange(24)
        satint = GeoInterpolator((lons, lats), (lines, cols), (hlines, hcols))
        satint._fill_row_borders()
        np.testing.assert_allclose(satint.tie_data[0],
                                   TIES_EXP5)
        np.testing.assert_allclose(satint.row_indices,
                                   np.array([0, 2, 7, 12, 17, 19]) / 5.0)
        satint = GeoInterpolator((lons, lats), (lines, cols),
                                 (hlines, hcols), chunk_size=10)
        satint._fill_row_borders()
        np.testing.assert_allclose(satint.tie_data[0],
                                   TIES_EXP6)
        np.testing.assert_allclose(satint.row_indices,
                                   np.array([0, 2, 7, 9, 10, 12, 17, 19]) / 5.0)


TIE_LONS = np.array([[1, 2, 3, 4],
                     [1, 2, 3, 4],
                     [1, 2, 3, 4],
                     [1, 2, 3, 4],
                     [1, 2, 3, 4]])

TIE_LATS = np.array([[1, 1, 1, 1],
                     [2, 2, 2, 2],
                     [3, 3, 3, 3],
                     [4, 4, 4, 4],
                     [5, 5, 5, 5]])


class TestGeoGridInterpolator:
    """Test the GeoGridInterpolator."""

    @pytest.mark.parametrize("args", ((TIE_LONS, TIE_LATS),
                                      [SwathDefinition(TIE_LONS, TIE_LATS)]
                                      ))
    def test_geogrid_interpolation(self, args):
        """Test that the interpolator works with both explicit tie-point arrays and swath definition objects."""
        x_points = np.array([0, 1, 3, 7])
        y_points = np.array([0, 1, 3, 7, 15])

        interpolator = GeoGridInterpolator((y_points, x_points), *args)

        fine_x_points = np.arange(8)
        fine_y_points = np.arange(16)

        lons, lats = interpolator.interpolate((fine_y_points, fine_x_points))

        lons_expected = np.array([1., 2., 2.5, 3., 3.25, 3.5, 3.75, 4.])
        lats_expected = np.array([1., 2., 2.5, 3., 3.25, 3.5, 3.75, 4., 4.125,
                                  4.25, 4.375, 4.5, 4.625, 4.75, 4.875, 5.])

        np.testing.assert_allclose(lons[0, :], lons_expected, rtol=5e-5)
        np.testing.assert_allclose(lats[:, 0], lats_expected, rtol=5e-5)

    def test_geogrid_interpolation_counts_its_arguments(self):
        """Test that an arbitrary number of argument is not allowed in the interpolator."""
        with pytest.raises(ValueError):
            _ = GeoGridInterpolator((None, None), None, None, None)

    def test_geogrid_interpolation_to_shape(self):
        """Test that the interpolator works with both explicit tie-point arrays and swath definition objects."""
        x_points = np.array([0, 1, 3, 7])
        y_points = np.array([0, 1, 3, 7, 15])

        interpolator = GeoGridInterpolator((y_points, x_points), TIE_LONS, TIE_LATS)

        lons, lats = interpolator.interpolate_to_shape((16, 8))

        lons_expected = np.array([1., 2., 2.5, 3., 3.25, 3.5, 3.75, 4.])
        lats_expected = np.array([1., 2., 2.5, 3., 3.25, 3.5, 3.75, 4., 4.125,
                                  4.25, 4.375, 4.5, 4.625, 4.75, 4.875, 5.])

        np.testing.assert_allclose(lons[0, :], lons_expected, rtol=5e-5)
        np.testing.assert_allclose(lats[:, 0], lats_expected, rtol=5e-5)

    def test_geogrid_interpolation_preserves_dtype(self):
        """Test that the interpolator works with both explicit tie-point arrays and swath definition objects."""
        x_points = np.array([0, 1, 3, 7])
        y_points = np.array([0, 1, 3, 7, 15])

        interpolator = GeoGridInterpolator((y_points, x_points),
                                           TIE_LONS.astype(np.float32), TIE_LATS.astype(np.float32))

        lons, lats = interpolator.interpolate_to_shape((16, 8))

        assert lons.dtype == np.float32
        assert lats.dtype == np.float32

    def test_chunked_geogrid_interpolation(self):
        """Test that the interpolator works with both explicit tie-point arrays and swath definition objects."""
        dask = pytest.importorskip("dask")

        x_points = np.array([0, 1, 3, 7])
        y_points = np.array([0, 1, 3, 7, 15])

        interpolator = GeoGridInterpolator((y_points, x_points),
                                           TIE_LONS.astype(np.float32), TIE_LATS.astype(np.float32))

        lons, lats = interpolator.interpolate_to_shape((16, 8), chunks=4)

        assert lons.chunks == ((4, 4, 4, 4), (4, 4))
        assert lats.chunks == ((4, 4, 4, 4), (4, 4))

        with dask.config.set({"array.chunk-size": 64}):

            lons, lats = interpolator.interpolate_to_shape((16, 8), chunks="auto")
            assert lons.chunks == ((4, 4, 4, 4), (4, 4))
            assert lats.chunks == ((4, 4, 4, 4), (4, 4))

    def test_geogrid_interpolation_can_extrapolate(self):
        """Test that the interpolator can also extrapolate given the right parameters."""
        x_points = np.array([0, 1, 3, 7])
        y_points = np.array([0, 1, 3, 7, 15])

        interpolator = GeoGridInterpolator((y_points, x_points), TIE_LONS, TIE_LATS,
                                           bounds_error=False, fill_value=None)

        lons, lats = interpolator.interpolate_to_shape((16, 16), method="cubic")

        assert lons.shape == (16, 16)


class TestGeoSplineInterpolator:
    """Test the GeoGridInterpolator."""

    @pytest.mark.parametrize("args", ((TIE_LONS, TIE_LATS),
                                      [SwathDefinition(TIE_LONS, TIE_LATS)]
                                      ))
    def test_geospline_interpolation(self, args):
        """Test that the interpolator works with both explicit tie-point arrays and swath definition objects."""
        x_points = np.array([0, 1, 3, 7])
        y_points = np.array([0, 1, 3, 7, 15])

        interpolator = GeoSplineInterpolator((y_points, x_points), *args, kx=1, ky=1)

        fine_x_points = np.arange(8)
        fine_y_points = np.arange(16)

        lons, lats = interpolator.interpolate((fine_y_points, fine_x_points))

        lons_expected = np.array([1., 2., 2.5, 3., 3.25, 3.5, 3.75, 4.])
        lats_expected = np.array([1., 2., 2.5, 3., 3.25, 3.5, 3.75, 4., 4.125,
                                  4.25, 4.375, 4.5, 4.625, 4.75, 4.875, 5.])

        np.testing.assert_allclose(lons[0, :], lons_expected, rtol=5e-5)
        np.testing.assert_allclose(lats[:, 0], lats_expected, rtol=5e-5)

    def test_geospline_interpolation_to_shape(self):
        """Test that the interpolator works with both explicit tie-point arrays and swath definition objects."""
        x_points = np.array([0, 1, 3, 7])
        y_points = np.array([0, 1, 3, 7, 15])

        interpolator = GeoSplineInterpolator((y_points, x_points), TIE_LONS, TIE_LATS, kx=1, ky=1)

        lons, lats = interpolator.interpolate_to_shape((16, 8))

        lons_expected = np.array([1., 2., 2.5, 3., 3.25, 3.5, 3.75, 4.])
        lats_expected = np.array([1., 2., 2.5, 3., 3.25, 3.5, 3.75, 4., 4.125,
                                  4.25, 4.375, 4.5, 4.625, 4.75, 4.875, 5.])

        np.testing.assert_allclose(lons[0, :], lons_expected, rtol=5e-5)
        np.testing.assert_allclose(lats[:, 0], lats_expected, rtol=5e-5)

    def test_geospline_interpolation_preserves_dtype(self):
        """Test that the interpolator works with both explicit tie-point arrays and swath definition objects."""
        x_points = np.array([0, 1, 3, 7])
        y_points = np.array([0, 1, 3, 7, 15])

        interpolator = GeoGridInterpolator((y_points, x_points),
                                           TIE_LONS.astype(np.float32), TIE_LATS.astype(np.float32))

        lons, lats = interpolator.interpolate_to_shape((16, 8))

        assert lons.dtype == np.float32
        assert lats.dtype == np.float32

    def test_chunked_geospline_interpolation(self):
        """Test that the interpolator works with both explicit tie-point arrays and swath definition objects."""
        dask = pytest.importorskip("dask")

        x_points = np.array([0, 1, 3, 7])
        y_points = np.array([0, 1, 3, 7, 15])

        interpolator = GeoGridInterpolator((y_points, x_points),
                                           TIE_LONS.astype(np.float32), TIE_LATS.astype(np.float32))

        lons, lats = interpolator.interpolate_to_shape((16, 8), chunks=4)

        assert lons.chunks == ((4, 4, 4, 4), (4, 4))
        assert lats.chunks == ((4, 4, 4, 4), (4, 4))

        with dask.config.set({"array.chunk-size": 64}):

            lons, lats = interpolator.interpolate_to_shape((16, 8), chunks="auto")
            assert lons.chunks == ((4, 4, 4, 4), (4, 4))
            assert lats.chunks == ((4, 4, 4, 4), (4, 4))
