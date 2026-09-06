"""Test the multilinear interpolation Cython implementation."""

import unittest
import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

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

RES1 = np.array([[1.25149566,  0.86801702,  0.59233148,  0.66310018,  0.86093089,
                  0.59448266,  0.88145882,  0.72057176,  0.44395844,  0.62692691,
                  0.22956686,  0.89308545,  0.8823242,  1.01691582,  1.06050602,
                  0.78964244,  1.04244001,  0.93481853,  0.61242017,  0.90454307,
                  0.5570492,  0.59214105,  0.87052131,  1.24339837,  0.9031915,
                  0.93282679,  0.88212567,  1.25090173,  0.47821939,  0.51308049],
                 [1.13891573,  0.79536539,  0.54753759,  0.61188808,  0.82629032,
                  0.55155224,  0.8025709,  0.64724795,  0.42423377,  0.60563713,
                  0.22485215,  0.82022895,  0.86317281,  0.92860053,  1.00514142,
                  0.73270743,  0.9756435,  0.88385772,  0.59570712,  0.85230456,
                  0.52319791,  0.55035212,  0.83992216,  1.12698577,  0.84054343,
                  0.91689093,  0.82346408,  1.13391799,  0.46731605,  0.5109711]])


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

        f = lambda x: np.vstack([
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


def _reference_function(x):
    """Smooth function of the ``d`` coordinates in the rows of ``x``."""
    return np.sum(np.sin(x + np.arange(x.shape[0])[:, None]), axis=0)


@pytest.mark.parametrize("d", [1, 2, 3, 4, 5])
def test_multilinear_matches_scipy(d):
    """Compare each dimensionality of the Cython kernels against scipy."""
    smin = [-1.0] * d
    smax = [1.0] * d
    orders = [4] * d

    interp = MultilinearInterpolator(smin, smax, orders)
    interp.set_values(np.atleast_2d(_reference_function(interp.grid)))

    axes = [np.linspace(smin[i], smax[i], orders[i]) for i in range(d)]
    scipy_interp = RegularGridInterpolator(
        axes, interp.values[0].reshape(orders), method="linear")

    rng = np.random.default_rng(1234)
    # in-bounds points only: outside the grid the Cython kernels extrapolate
    # linearly, which is an intentional difference from scipy's handling
    points = rng.uniform(-1.0, 1.0, (d, 500))

    result = interp(points)
    expected = scipy_interp(points.T)

    np.testing.assert_allclose(result[0], expected, rtol=1e-12, atol=1e-12)


def test_multilinear_too_many_dimensions():
    """A 6-D input is rejected with a message that matches the guard."""
    interp = MultilinearInterpolator([-1.0] * 6, [1.0] * 6, [4] * 6)
    interp.set_values(np.zeros((1, 4 ** 6)))
    with pytest.raises(Exception, match="strictly greater than 5"):
        interp(np.zeros((6, 3)))


def suite():
    """The suite for Multilinear Interpolator"""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestMultilinearInterpolator))

    return mysuite

if __name__ == '__main__':
    unittest.main()
