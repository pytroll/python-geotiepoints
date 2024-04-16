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
"""Tests for Interpolator."""

import unittest
import numpy as np
import pytest
from unittest import mock

from geotiepoints.interpolator import Interpolator
from geotiepoints.interpolator import SingleGridInterpolator

TIES_EXP1 = np.array([[-2.00000000e+00, -4.00000000e-01, 3.60000000e+00,
                       7.60000000e+00, 1.16000000e+01, 1.56000000e+01,
                       1.64000000e+01],
                      [-1.60000000e+00, 0.00000000e+00, 4.00000000e+00,
                       8.00000000e+00, 1.20000000e+01, 1.60000000e+01,
                       1.68000000e+01],
                      [-6.00000000e-01, 1.00000000e+00, 5.00000000e+00,
                       9.00000000e+00, 1.30000000e+01, 1.70000000e+01,
                       1.78000000e+01],
                      [-2.00000000e-01, 1.40000000e+00, 5.40000000e+00,
                       9.40000000e+00, 1.34000000e+01, 1.74000000e+01,
                       1.82000000e+01],
                      [4.44089210e-16, 1.60000000e+00, 5.60000000e+00,
                       9.60000000e+00, 1.36000000e+01, 1.76000000e+01,
                       1.84000000e+01],
                      [4.00000000e-01, 2.00000000e+00, 6.00000000e+00,
                       1.00000000e+01, 1.40000000e+01, 1.80000000e+01,
                       1.88000000e+01],
                      [1.40000000e+00, 3.00000000e+00, 7.00000000e+00,
                       1.10000000e+01, 1.50000000e+01, 1.90000000e+01,
                       1.98000000e+01],
                      [1.80000000e+00, 3.40000000e+00, 7.40000000e+00,
                       1.14000000e+01, 1.54000000e+01, 1.94000000e+01,
                       2.02000000e+01]])


class TestInterpolator(unittest.TestCase):
    """Test the interpolator."""

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
        """Test extrapolation of the columns."""
        lons = np.arange(10).reshape((2, 5), order="F")
        lats = np.arange(10).reshape((2, 5), order="C")
        lines = np.array([2, 7])
        cols = np.array([2, 7, 12, 17, 22])
        hlines = np.arange(10)
        hcols = np.arange(24)
        satint = Interpolator((lons, lats), (lines, cols), (hlines, hcols))
        self.assertTrue(np.allclose(satint._extrapolate_cols(satint.tie_data[0]),
                                    np.array([[-0.8, 0., 2., 4., 6., 8., 8.4],
                                              [0.2, 1., 3., 5., 7., 9., 9.4]])))

    def test_fill_col_borders(self):
        """Test filling the column borders."""
        lons = np.arange(10).reshape((2, 5), order="F")
        lats = np.arange(10).reshape((2, 5), order="C")
        lines = np.array([2, 7])
        cols = np.array([2, 7, 12, 17, 22])
        hlines = np.arange(10)
        hcols = np.arange(24)
        satint = Interpolator((lons, lats), (lines, cols), (hlines, hcols))
        satint._fill_col_borders()
        self.assertTrue(np.allclose(satint.tie_data[0],
                                    np.array([[-0.8, 0., 2., 4., 6., 8., 8.4],
                                              [0.2, 1., 3., 5., 7., 9., 9.4]])))
        self.assertTrue(np.allclose(satint.col_indices,
                                    np.array([0, 2, 7, 12, 17, 22, 23])))

    def test_extrapolate_rows(self):
        """Test extrapolation of rows."""
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
                                    np.array([[-0.4, 1.6, 3.6, 5.6, 7.6],
                                              [0., 2., 4., 6., 8.],
                                              [1., 3., 5., 7., 9.],
                                              [1.4, 3.4, 5.4, 7.4, 9.4]])))

    def test_results_in_c_order(self):
        """Test that the results are in C order."""
        lons = np.arange(10).reshape((2, 5), order="F")
        lats = np.arange(10).reshape((2, 5), order="C")
        lines = np.array([2, 7])
        cols = np.array([2, 7, 12, 17, 22])
        hlines = np.arange(10)
        hcols = np.arange(24)
        satint = Interpolator((lons, lats), (lines, cols), (hlines, hcols))
        interp_results = satint.interpolate()
        assert interp_results[0].flags['C_CONTIGUOUS'] is True


@pytest.fixture
def grid_interpolator():
    """Return an instance of SingleGridInterpolator for testing."""
    xpoints = np.array([0, 3, 7, 15])
    ypoints = np.array([0, 3, 7, 15, 31])
    data = np.array([[0, 1, 0, 1],
                     [2, 2, 2, 1],
                     [0, 3, 3, 3],
                     [1, 2, 1, 2],
                     [4, 4, 4, 4]],
                    dtype=np.float64)

    return SingleGridInterpolator((ypoints, xpoints), data)


class TestSingleGridInterpolator:
    """Test for the SingleGridInterpolator."""

    def setup_method(self):
        """Set up the tests."""
        self.expected = np.array([[0.00000000e+00, 5.91666667e-01, 9.09722222e-01,
                                   1.00000000e+00, 9.08333333e-01, 6.80555556e-01,
                                   3.62500000e-01, 1.11022302e-16, -3.61111111e-01,
                                   -6.75000000e-01, -8.95833333e-01, -9.77777778e-01,
                                   -8.75000000e-01, -5.41666667e-01, 6.80555556e-02,
                                   1.00000000e+00],
                                  [1.29188830e+00, 1.36660572e+00, 1.37255612e+00,
                                   1.32180851e+00, 1.22643190e+00, 1.09849528e+00,
                                   9.50067677e-01, 7.93218085e-01, 6.40015514e-01,
                                   5.02528970e-01, 3.92827460e-01, 3.22979990e-01,
                                   3.05055566e-01, 3.51123195e-01, 4.73251884e-01,
                                   6.83510638e-01],
                                  [1.90990691e+00, 1.81115953e+00, 1.72929138e+00,
                                   1.66103090e+00, 1.60310651e+00, 1.55224664e+00,
                                   1.50517970e+00, 1.45863412e+00, 1.40933833e+00,
                                   1.35402074e+00, 1.28940979e+00, 1.21223389e+00,
                                   1.11922148e+00, 1.00710096e+00, 8.72600776e-01,
                                   7.12449341e-01],
                                  [2.00000000e+00, 1.99166667e+00, 1.99305556e+00,
                                   2.00000000e+00, 2.00833333e+00, 2.01388889e+00,
                                   2.01250000e+00, 2.00000000e+00, 1.97222222e+00,
                                   1.92500000e+00, 1.85416667e+00, 1.75555556e+00,
                                   1.62500000e+00, 1.45833333e+00, 1.25138889e+00,
                                   1.00000000e+00],
                                  [1.70811170e+00, 1.97446571e+00, 2.17697619e+00,
                                   2.32104863e+00, 2.41208851e+00, 2.45550132e+00,
                                   2.45669253e+00, 2.42106763e+00, 2.35403210e+00,
                                   2.26099144e+00, 2.14735111e+00, 2.01851661e+00,
                                   1.87989341e+00, 1.73688701e+00, 1.59490288e+00,
                                   1.45934650e+00],
                                  [1.18018617e+00, 1.82589523e+00, 2.29418084e+00,
                                   2.60650963e+00, 2.78434820e+00, 2.84916319e+00,
                                   2.82242122e+00, 2.72558891e+00, 2.58013287e+00,
                                   2.40751974e+00, 2.22921614e+00, 2.06668868e+00,
                                   1.94140399e+00, 1.87482869e+00, 1.88842940e+00,
                                   2.00367275e+00],
                                  [5.62167553e-01, 1.61229380e+00, 2.35779706e+00,
                                   2.83871581e+00, 3.09508855e+00, 3.16695379e+00,
                                   3.09435002e+00, 2.91731573e+00, 2.67588943e+00,
                                   2.41010961e+00, 2.16001476e+00, 1.96564339e+00,
                                   1.86703400e+00, 1.90422507e+00, 2.11725511e+00,
                                   2.54616261e+00],
                                  [-5.55111512e-16, 1.40000000e+00, 2.38095238e+00,
                                   3.00000000e+00, 3.31428571e+00, 3.38095238e+00,
                                   3.25714286e+00, 3.00000000e+00, 2.66666667e+00,
                                   2.31428571e+00, 2.00000000e+00, 1.78095238e+00,
                                   1.71428571e+00, 1.85714286e+00, 2.26666667e+00,
                                   3.00000000e+00],
                                  [-3.87768035e-01, 1.24340379e+00, 2.37526033e+00,
                                   3.07734929e+00, 3.41921836e+00, 3.47041522e+00,
                                   3.30048758e+00, 2.97898313e+00, 2.57544955e+00,
                                   2.15943455e+00, 1.80048581e+00, 1.56815104e+00,
                                   1.53197791e+00, 1.76151412e+00, 2.32630738e+00,
                                   3.29590536e+00],
                                  [-5.92170878e-01, 1.14910071e+00, 2.34627828e+00,
                                   3.07636778e+00, 3.41637520e+00, 3.44330649e+00,
                                   3.23416762e+00, 2.86596457e+00, 2.41570330e+00,
                                   1.96038977e+00, 1.57702995e+00, 1.34262982e+00,
                                   1.33419533e+00, 1.62873246e+00, 2.30324718e+00,
                                   3.43474544e+00],
                                  [-6.31638547e-01, 1.11173771e+00, 2.29804954e+00,
                                   3.00731383e+00, 3.31954746e+00, 3.31476733e+00,
                                   3.07299031e+00, 2.67423329e+00, 2.19851317e+00,
                                   1.72584681e+00, 1.33625112e+00, 1.10974297e+00,
                                   1.12633925e+00, 1.46605685e+00, 2.20891265e+00,
                                   3.43492354e+00],
                                  [-5.24601064e-01, 1.12596172e+00, 2.23461746e+00,
                                   2.88044580e+00, 3.14252639e+00, 3.09993890e+00,
                                   2.83176297e+00, 2.41707827e+00, 1.93496443e+00,
                                   1.46450113e+00, 1.08476800e+00, 8.74844707e-01,
                                   9.13810903e-01, 1.28074624e+00, 2.05473038e+00,
                                   3.31484296e+00],
                                  [-2.89488447e-01, 1.18641968e+00, 2.16002535e+00,
                                   2.70602204e+00, 2.89910321e+00, 2.81396236e+00,
                                   2.52529294e+00, 2.10778846e+00, 1.63614237e+00,
                                   1.18504816e+00, 8.29199299e-01, 6.43289272e-01,
                                   7.02011552e-01, 1.08005962e+00, 1.85212694e+00,
                                   3.09290701e+00],
                                  [5.52692819e-02, 1.28775854e+00, 2.07831656e+00,
                                   2.49430091e+00, 2.60306915e+00, 2.47197885e+00,
                                   2.16838755e+00, 1.75965283e+00, 1.31313224e+00,
                                   8.96183350e-01, 5.76163712e-01, 4.20430892e-01,
                                   4.96342449e-01, 8.71255945e-01, 1.61252894e+00,
                                   2.78751900e+00],
                                  [4.91242104e-01, 1.42462522e+00, 1.99353441e+00,
                                   2.25554078e+00, 2.26821545e+00, 2.08912953e+00,
                                   1.77585413e+00, 1.38596036e+00, 9.77019326e-01,
                                   6.06602149e-01, 3.32279936e-01, 2.11623798e-01,
                                   3.02204847e-01, 6.61594194e-01, 1.34736295e+00,
                                   2.41708223e+00],
                                  [1.00000000e+00, 1.59166667e+00, 1.90972222e+00,
                                   2.00000000e+00, 1.90833333e+00, 1.68055556e+00,
                                   1.36250000e+00, 1.00000000e+00, 6.38888889e-01,
                                   3.25000000e-01, 1.04166667e-01, 2.22222222e-02,
                                   1.25000000e-01, 4.58333333e-01, 1.06805556e+00,
                                   2.00000000e+00],
                                  [1.56311295e+00, 1.78352982e+00, 1.83092334e+00,
                                   1.73793693e+00, 1.53721403e+00, 1.26139808e+00,
                                   9.43132501e-01, 6.15060731e-01, 3.09826203e-01,
                                   6.00723485e-02, -1.01557398e-01, -1.42419605e-01,
                                   -2.98708387e-02, 2.68732333e-01, 7.86033344e-01,
                                   1.55467563e+00],
                                  [2.16215093e+00, 1.99486162e+00, 1.76118108e+00,
                                   1.47960993e+00, 1.16864878e+00, 8.46798254e-01,
                                   5.32558960e-01, 2.44431516e-01, 9.16537382e-04,
                                   -1.79485360e-01, -2.78273562e-01, -2.76947452e-01,
                                   -1.57006415e-01, 1.00050164e-01, 5.12722901e-01,
                                   1.09951241e+00],
                                  [2.77868393e+00, 2.22030900e+00, 1.70453878e+00,
                                   1.23527736e+00, 8.16428811e-01, 4.51897232e-01,
                                   1.45586708e-01, -9.85986773e-02, -2.76754836e-01,
                                   -3.84977682e-01, -4.19363128e-01, -3.76007089e-01,
                                   -2.51005477e-01, -4.04542058e-02, 2.59550811e-01,
                                   6.52913659e-01],
                                  [3.39428191e+00, 2.45451890e+00, 1.66503977e+00,
                                   1.01519757e+00, 4.94345351e-01, 9.18361693e-02,
                                   -2.02976925e-01, -4.00740881e-01, -5.12102647e-01,
                                   -5.47709170e-01, -5.18207399e-01, -4.34244283e-01,
                                   -3.06466769e-01, -1.45521806e-01, 3.79436582e-02,
                                   2.33282675e-01],
                                  [3.99051488e+00, 2.69213826e+00, 1.64672737e+00,
                                   8.29628926e-01, 2.16189635e-01, -2.18243782e-01,
                                   -4.98324609e-01, -6.48706129e-01, -6.94041625e-01,
                                   -6.58984380e-01, -5.68187679e-01, -4.46304804e-01,
                                   -3.17989039e-01, -2.07893666e-01, -1.40671971e-01,
                                   -1.40977235e-01],
                                  [4.54895279e+00, 2.92781402e+00, 1.65364493e+00,
                                   6.88829787e-01, -4.24710692e-03, -4.63201469e-01,
                                   -7.25649013e-01, -8.29205452e-01, -8.11486499e-01,
                                   -7.10107867e-01, -5.62685270e-01, -4.06834420e-01,
                                   -2.80171032e-01, -2.20310818e-01, -2.64869492e-01,
                                   -4.51462766e-01],
                                  [5.05116564e+00, 3.15619312e+00, 1.68983575e+00,
                                   6.03058511e-01, -1.53173643e-01, -6.27895738e-01,
                                   -8.70142807e-01, -9.28949884e-01, -8.53351999e-01,
                                   -6.92384185e-01, -4.95081475e-01, -3.10478902e-01,
                                   -1.87611496e-01, -1.75514291e-01, -3.23222319e-01,
                                   -6.79770612e-01],
                                  [5.47872340e+00, 3.37192249e+00, 1.75934319e+00,
                                   5.82573455e-01, -2.16798741e-01, -6.97185434e-01,
                                   -9.16998661e-01, -9.34650456e-01, -8.08552854e-01,
                                   -5.97117890e-01, -3.58757599e-01, -1.51884016e-01,
                                   -3.49091764e-02, -6.62451151e-02, -3.04303867e-01,
                                   -8.07497467e-01],
                                  [5.81319606e+00, 3.56964908e+00, 1.86621056e+00,
                                   6.37632979e-01, -1.81331170e-01, -6.55929406e-01,
                                   -8.51409244e-01, -8.33018201e-01, -6.66003793e-01,
                                   -4.15613535e-01, -1.47094943e-01, 7.43044672e-02,
                                   1.83337180e-01, 1.14755679e-01, -1.96687551e-01,
                                   -8.16240027e-01],
                                  [6.03615359e+00, 3.74401982e+00, 2.01448119e+00,
                                   7.78495441e-01, -3.29796987e-02, -4.88986498e-01,
                                   -6.58567226e-01, -6.10764153e-01, -4.14619546e-01,
                                   -1.39175676e-01, 1.46525189e-01, 3.73440779e-01,
                                   4.72528826e-01, 3.74747061e-01, 1.10532133e-02,
                                   -6.87594985e-01],
                                  [6.12916597e+00, 3.88968165e+00, 2.20819842e+00,
                                   1.01541920e+00, 2.42046904e-01, -1.81215558e-01,
                                   -3.23665277e-01, -2.54599342e-01, -4.33148426e-02,
                                   2.40891132e-01, 5.28721493e-01, 7.50879151e-01,
                                   8.38067017e-01, 7.20988000e-01, 3.30345012e-01,
                                   -4.03159036e-01],
                                  [6.07380319e+00, 4.00128150e+00, 2.45140557e+00,
                                   1.35866261e+00, 6.57539871e-01, 2.82524568e-01,
                                   1.68103934e-01, 2.48765198e-01, 4.58995588e-01,
                                   7.33282336e-01, 1.00611267e+00, 1.21197381e+00,
                                   1.28535300e+00, 1.16073747e+00, 7.72614430e-01,
                                   5.54711246e-02],
                                  [5.85163522e+00, 4.07346633e+00, 2.74814597e+00,
                                   1.81848404e+00, 1.22729043e+00, 9.17375033e-01,
                                   8.31547737e-01, 9.12618434e-01, 1.10339702e+00,
                                   1.34669338e+00, 1.58531741e+00, 1.76207900e+00,
                                   1.81978804e+00, 1.70125443e+00, 1.34928805e+00,
                                   7.06698803e-01],
                                  [5.44423205e+00, 4.10088306e+00, 3.10246296e+00,
                                   2.40514184e+00, 1.96508982e+00, 1.73847699e+00,
                                   1.68147346e+00, 1.75024934e+00, 1.90097472e+00,
                                   2.08981971e+00, 2.27295441e+00, 2.40654894e+00,
                                   2.44677339e+00, 2.34979786e+00, 2.07179247e+00,
                                   1.56892730e+00],
                                  [4.83316365e+00, 4.07817864e+00, 3.51839986e+00,
                                   3.12889438e+00, 2.88472927e+00, 2.76097160e+00,
                                   2.73268844e+00, 2.77494687e+00, 2.86281395e+00,
                                   2.97135677e+00, 3.07564238e+00, 3.15073786e+00,
                                   3.17171029e+00, 3.11362673e+00, 2.95155425e+00,
                                   2.66055994e+00],
                                  [4.00000000e+00, 4.00000000e+00, 4.00000000e+00,
                                   4.00000000e+00, 4.00000000e+00, 4.00000000e+00,
                                   4.00000000e+00, 4.00000000e+00, 4.00000000e+00,
                                   4.00000000e+00, 4.00000000e+00, 4.00000000e+00,
                                   4.00000000e+00, 4.00000000e+00, 4.00000000e+00,
                                   4.00000000e+00]])

    def test_interpolate(self, grid_interpolator):
        """Test that interpolation is working."""
        fine_x = np.arange(16)
        fine_y = np.arange(32)

        res = grid_interpolator.interpolate((fine_y, fine_x), method="cubic_legacy")
        np.testing.assert_allclose(res, self.expected, atol=2e-9)

    def test_interpolate_slices(self, grid_interpolator):
        """Test that interpolation from slices is working."""
        res = grid_interpolator.interpolate_slices((slice(0, 32), slice(0, 16)), method="cubic_legacy")
        np.testing.assert_allclose(res, self.expected, atol=2e-9)

    @pytest.mark.parametrize("chunks, expected_chunks", [(10, (10, 10)),
                                                         ((10, 5), (10, 5))])
    def test_interpolate_dask(self, grid_interpolator, chunks, expected_chunks):
        """Test that interpolation with dask is working."""
        da = pytest.importorskip("dask.array")

        fine_x = np.arange(16)
        fine_y = np.arange(32)

        with mock.patch.object(grid_interpolator,
                               "interpolate_numpy",
                               wraps=grid_interpolator.interpolate_numpy) as interpolate:
            res = grid_interpolator.interpolate((fine_y, fine_x), method="cubic_legacy", chunks=chunks)
            assert not interpolate.called

            assert isinstance(res, da.Array)
            v_chunk, h_chunk = expected_chunks
            assert res.chunks[0][0] == v_chunk
            assert res.chunks[1][0] == h_chunk

            np.testing.assert_allclose(res, self.expected, atol=2e-9)
            assert interpolate.called

    def test_interpolate_preserves_dtype(self):
        """Test that interpolation is preserving the dtype."""
        xpoints = np.array([0, 3, 7, 15])
        ypoints = np.array([0, 3, 7, 15, 31])
        data = np.array([[0, 1, 0, 1],
                        [2, 2, 2, 1],
                        [0, 3, 3, 3],
                        [1, 2, 1, 2],
                        [4, 4, 4, 4]],
                        dtype=np.float32)

        grid_interpolator = SingleGridInterpolator((ypoints, xpoints), data)
        fine_x = np.arange(16)
        fine_y = np.arange(32)

        res = grid_interpolator.interpolate((fine_y, fine_x), method="cubic_legacy")
        assert res.dtype == data.dtype
