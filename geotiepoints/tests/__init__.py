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

"""The tests package"""

from geotiepoints import interpolator

from geotiepoints.tests import (test_geointerpolator,
                                test_modis,
                                test_satelliteinterpolator,
                                test_interpolator,
                                test_multilinear,
                                test_modisinterpolator)


import sys
if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

import doctest
import os
TRAVIS = os.environ.get("TRAVIS", False)


def suite():
    """The global test suite.
    """
    mysuite = unittest.TestSuite()
    if not TRAVIS:
        # Test sphinx documentation pages:
        mysuite.addTests(doctest.DocFileSuite('../../doc/source/index.rst'))
        # # Test the documentation strings
        mysuite.addTests(doctest.DocTestSuite(interpolator))

    # Use the unittests also
    mysuite.addTests(test_geointerpolator.suite())
    mysuite.addTests(test_modis.suite())
    mysuite.addTests(test_satelliteinterpolator.suite())
    mysuite.addTests(test_interpolator.suite())
    mysuite.addTests(test_multilinear.suite())
    mysuite.addTests(test_modisinterpolator.suite())
    return mysuite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
