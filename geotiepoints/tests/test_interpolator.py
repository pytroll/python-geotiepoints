#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2013, 2017 Martin Raspaud

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

"""
"""

import unittest
import numpy as np

from geotiepoints.interpolator import Interpolator

TIES_EXP1 = np.array([[-2.00000000e+00,  -4.00000000e-01,   3.60000000e+00,
                       7.60000000e+00,   1.16000000e+01,   1.56000000e+01,
                       1.64000000e+01],
                      [-1.60000000e+00,   0.00000000e+00,   4.00000000e+00,
                       8.00000000e+00,   1.20000000e+01,   1.60000000e+01,
                       1.68000000e+01],
                      [-6.00000000e-01,   1.00000000e+00,   5.00000000e+00,
                       9.00000000e+00,   1.30000000e+01,   1.70000000e+01,
                       1.78000000e+01],
                      [-2.00000000e-01,   1.40000000e+00,   5.40000000e+00,
                       9.40000000e+00,   1.34000000e+01,   1.74000000e+01,
                       1.82000000e+01],
                      [4.44089210e-16,   1.60000000e+00,   5.60000000e+00,
                       9.60000000e+00,   1.36000000e+01,   1.76000000e+01,
                       1.84000000e+01],
                      [4.00000000e-01,   2.00000000e+00,   6.00000000e+00,
                       1.00000000e+01,   1.40000000e+01,   1.80000000e+01,
                       1.88000000e+01],
                      [1.40000000e+00,   3.00000000e+00,   7.00000000e+00,
                       1.10000000e+01,   1.50000000e+01,   1.90000000e+01,
                       1.98000000e+01],
                      [1.80000000e+00,   3.40000000e+00,   7.40000000e+00,
                       1.14000000e+01,   1.54000000e+01,   1.94000000e+01,
                       2.02000000e+01]])


class TestInterpolator(unittest.TestCase):

    # def test_fill_borders(self):
    #     lons = np.arange(20).reshape((4, 5), order="F")
    #     lats = np.arange(20).reshape((4, 5), order="C")
    #     lines = np.array([2, 7, 12, 17])
    #     cols = np.array([2, 7, 12, 17, 22])
    #     hlines = np.arange(20)
    #     hcols = np.arange(24)
    #     satint = Interpolator(
    #         (lons, lats), (lines, cols), (hlines, hcols), chunk_size=10)
    #     satint.fill_borders('x', 'y')
    #     self.assertTrue(np.allclose(satint.tie_data[0], TIES_EXP1))

    #     self.assertTrue(np.allclose(satint.row_indices,
    #                                 np.array([0,  2,  7,  9, 10, 12, 17, 19])))
    #     self.assertTrue(np.allclose(satint.col_indices,
    #                                 np.array([0,  2,  7, 12, 17, 22, 23])))

    def test_extrapolate_cols(self):
        lons = np.arange(10).reshape((2, 5), order="F")
        lats = np.arange(10).reshape((2, 5), order="C")
        lines = np.array([2, 7])
        cols = np.array([2, 7, 12, 17, 22])
        hlines = np.arange(10)
        hcols = np.arange(24)
        satint = Interpolator((lons, lats), (lines, cols), (hlines, hcols))
        self.assertTrue(np.allclose(satint._extrapolate_cols(satint.tie_data[0]),
                                    np.array([[-0.8,  0.,  2.,  4.,  6.,  8.,  8.4],
                                              [0.2,  1.,  3.,  5.,  7.,  9.,  9.4]])))

    def test_fill_col_borders(self):
        lons = np.arange(10).reshape((2, 5), order="F")
        lats = np.arange(10).reshape((2, 5), order="C")
        lines = np.array([2, 7])
        cols = np.array([2, 7, 12, 17, 22])
        hlines = np.arange(10)
        hcols = np.arange(24)
        satint = Interpolator((lons, lats), (lines, cols), (hlines, hcols))
        satint._fill_col_borders()
        self.assertTrue(np.allclose(satint.tie_data[0],
                                    np.array([[-0.8,  0.,  2.,  4.,  6.,  8.,  8.4],
                                              [0.2,  1.,  3.,  5.,  7.,  9.,  9.4]])))
        self.assertTrue(np.allclose(satint.col_indices,
                                    np.array([0,  2,  7, 12, 17, 22, 23])))

    def test_extrapolate_rows(self):
        lons = np.arange(10).reshape((2, 5), order="F")
        lats = np.arange(10).reshape((2, 5), order="C")
        lines = np.array([2, 7])
        cols = np.array([2, 7, 12, 17, 22])
        hlines = np.arange(10)
        hcols = np.arange(24)
        satint = Interpolator((lons, lats), (lines, cols), (hlines, hcols))
        first_idx = satint.hrow_indices[0]
        last_idx = satint.hrow_indices[-1]
        self.assertTrue(np.allclose(satint._extrapolate_rows(lons,
                                                             lines,
                                                             first_idx, last_idx),
                                    np.array([[-0.4,  1.6,  3.6,  5.6,  7.6],
                                              [0.,  2.,  4.,  6.,  8.],
                                              [1.,  3.,  5.,  7.,  9.],
                                              [1.4,  3.4,  5.4,  7.4,  9.4]])))

    def test_results_in_c_order(self):
        lons = np.arange(10).reshape((2, 5), order="F")
        lats = np.arange(10).reshape((2, 5), order="C")
        lines = np.array([2, 7])
        cols = np.array([2, 7, 12, 17, 22])
        hlines = np.arange(10)
        hcols = np.arange(24)
        satint = Interpolator((lons, lats), (lines, cols), (hlines, hcols))
        interp_results = satint.interpolate()
        assert interp_results[0].flags['C_CONTIGUOUS'] is True

    # def test_fill_row_borders(self):
    #     lons = np.arange(20).reshape((4, 5), order="F")
    #     lats = np.arange(20).reshape((4, 5), order="C")
    #     lines = np.array([2, 7, 12, 17])
    #     cols = np.array([2, 7, 12, 17, 22])
    #     hlines = np.arange(20)
    #     hcols = np.arange(24)
    #     satint = Interpolator((lons, lats), (lines, cols), (hlines, hcols))
    #     satint._fill_row_borders()
    #     self.assertTrue(np.allclose(satint.tie_data[0],
    #                                 np.array([[-0.4,   3.6,   7.6,  11.6,  15.6],
    #                                           [0.,   4.,   8.,  12.,  16.],
    #                                           [1.,   5.,   9.,  13.,  17.],
    #                                           [2.,   6.,  10.,  14.,  18.],
    #                                           [3.,   7.,  11.,  15.,  19.],
    #                                           [3.4,   7.4,  11.4,  15.4,  19.4]])))
    #     self.assertTrue(np.allclose(satint.row_indices,
    #                                 np.array([0,  2,  7, 12, 17, 19])))
    #     satint = Interpolator(
    #         (lons, lats), (lines, cols), (hlines, hcols), chunk_size=10)
    #     satint._fill_row_borders()
    #     self.assertTrue(np.allclose(satint.tie_data[0],
    #                                 np.array([[-0.4,   3.6,   7.6,  11.6,  15.6],
    #                                           [0.,   4.,   8.,  12.,  16.],
    #                                           [1.,   5.,   9.,  13.,  17.],
    #                                           [1.4,   5.4,   9.4,
    #                                               13.4,  17.4],
    #                                           [1.6,   5.6,   9.6,
    #                                               13.6,  17.6],
    #                                           [2.,   6.,  10.,  14.,  18.],
    #                                           [3.,   7.,  11.,  15.,  19.],
    #                                           [3.4,   7.4,  11.4,  15.4,  19.4]])))
    #     self.assertTrue(np.allclose(satint.row_indices,
    # np.array([0,  2,  7,  9, 10, 12, 17, 19])))


def suite():
    """The suite for Interpolator"""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestInterpolator))

    return mysuite
