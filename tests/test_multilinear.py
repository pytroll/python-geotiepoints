#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Adam.Dybbroe

# Author(s):

#   Adam.Dybbroe <a000680@c20671.ad.smhi.se>

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

"""Test the multilinear interpolation Cython implementation
"""

import unittest
import numpy as np

from geotiepoints.multilinear import MultilinearInterpolator

ARR1 = np.array([[0.99819991,  0.46872418,  0.39356122,  0.33170364,  0.79171217,
                  0.31300346,  0.69075206,  0.47492703,  0.2012821,  0.57347047,
                  0.12934437,  0.7384143,  0.84650654,  0.63113452,  0.35514309,
                  0.36773267,  0.42887193,  0.83856559,  0.1081537,  0.33634562,
                  0.25445117,  0.30121727,  0.19697695,  0.92937056,  0.40359487,
                  0.10237384,  0.37803665,  0.94699248,  0.08045698,  0.01366914],
                 [0.72748552,  0.70872219,  0.36883461,  0.52914895,  0.22308535,
                  0.4444687,  0.52394334,  0.52870835,  0.31756298,  0.13776131,
                  0.11812231,  0.46974149,  0.12318789,  0.76525517,  0.97328814,
                  0.6580273,  0.93059119,  0.32785305,  0.57033161,  0.80133526,
                  0.4311177,  0.44957946,  0.81073879,  0.79356296,  0.77565555,
                  0.90520185,  0.76064422,  0.78609587,  0.43915797,  0.50745485]])


RES1 = np.array([[0.7281677,  0.92679761,  0.75864538,  0.90459892,  0.67275753,
                  1.19414627,  0.96261924,  0.95958145,  1.25475254,  0.92185766,
                  0.48662567,  0.59148584,  0.49659237,  1.06516992,  1.02603276,
                  0.97956055,  0.23856018,  0.55233847,  0.3944975,  1.02927981,
                  1.00561313,  0.85410339,  0.90513151,  1.15393943,  0.81019541,
                  0.49423621,  0.8488052,  1.13126666,  0.46031945,  0.84240936],
                 [0.69743384,  0.88834789,  0.68115593,  0.8640305,  0.60768955,
                  1.09207047,  0.87763397,  0.92367035,  1.13837765,  0.9193805,
                  0.46966199,  0.54809891,  0.47045774,  0.99002514,  0.95705429,
                  0.89400699,  0.23406337,  0.51765791,  0.39250097,  0.9393847,
                  0.99415483,  0.81551342,  0.90068752,  1.05466585,  0.77916419,
                  0.48278904,  0.79960525,  1.02986508,  0.43767303,  0.77857132]])


def assertNumpyArraysEqual(self, other):
    if self.shape != other.shape:
        raise AssertionError("Shapes don't match")
    if not np.allclose(self, other):
        raise AssertionError("Elements don't match!")


class TestMultilinearInterpolator(unittest.TestCase):

    """Class for unit testing the multilinear interpolation method
    """

    def setUp(self):
        pass

    def test_multilinear_interp(self):
        """Test the multilinear interpolation"""

        smin = [-1, -1]
        smax = [1, 1]
        orders = [5, 5]

        f = lambda x: np.row_stack([
            np.sqrt(x[0, :]**2 + x[1, :]**2),
            np.power(x[0, :]**3 + x[1, :]**3, 1.0 / 3.0)
        ])

        interp = MultilinearInterpolator(smin, smax, orders)
        interp.set_values(f(interp.grid))

        result = interp(ARR1)
        # exact_values = f(ARR1)

        assertNumpyArraysEqual(result, RES1)

    def tearDown(self):
        """Clean up"""
        return


def suite():
    """The suite for Multilinear Interpolator"""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestMultilinearInterpolator))

    return mysuite

if __name__ == '__main__':
    unittest.main()
