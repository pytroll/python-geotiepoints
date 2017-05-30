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

import numpy as np
from geotiepoints.geointerpolator import GeoInterpolator

lons = np.arange(20).reshape((4, 5), order="F")
lats = np.arange(20).reshape((4, 5), order="C")
lines = np.array([2, 7, 12, 17]) / 5.0
cols = np.array([2, 7, 12, 17, 22])
hlines = np.arange(20) / 5.0
hcols = np.arange(24)
satint = GeoInterpolator(
    (lons, lats), (lines, cols), (hlines, hcols), chunk_size=10)
satint.fill_borders('x', 'y')
