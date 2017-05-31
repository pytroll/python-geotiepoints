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

"""
"""

import h5py
import numpy as np
from geotiepoints import modis5kmto1km

FILENAME_FULL = 'testdata/test_5_to_1_geoloc_full.h5'
FILENAME_5KM = 'testdata/test_5_to_1_geoloc_5km.h5'

with h5py.File(FILENAME_FULL) as h5f:
    glons = h5f['longitude'][:] / 1000.
    glats = h5f['latitude'][:] / 1000.

with h5py.File(FILENAME_5KM) as h5f:
    lons = h5f['longitude'][:] / 1000.
    lats = h5f['latitude'][:] / 1000.

tlons, tlats = modis5kmto1km(lons, lats)

print np.allclose(tlons, glons, atol=0.05)
print np.allclose(tlats, glats, atol=0.05)
