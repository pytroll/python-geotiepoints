#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2013 Martin Raspaud

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

"""Geographical interpolation (lon/lats).
"""

import numpy as np
from numpy import arccos, sign, rad2deg, sqrt, arcsin
from geotiepoints.interpolator import Interpolator

EARTH_RADIUS = 6370997.0

class GeoInterpolator(Interpolator):
    """
    Handles interpolation of geolocation from a grid of tie points.  It is
    preferable to have tie-points out till the edges if the tiepoint grid, but
    a method is provided to extrapolate linearly the tiepoints to the borders
    of the grid. The extrapolation is done automatically if it seems necessary.

    Uses numpy, scipy, and optionally pyresample

    The constructor takes in the tiepointed data as *data*, the
    *tiepoint_grid* and the desired *final_grid*. As optional arguments, one
    can provide *kx_* and *ky_* as interpolation orders (in x and y directions
    respectively), and the *chunksize* if the data has to be handled by pieces
    along the y axis (this affects how the extrapolator behaves). If
    *chunksize* is set, don't forget to adjust the interpolation orders
    accordingly: the interpolation is indeed done globaly (not chunkwise).
    """
    def __init__(self, lon_lat_data, *args, **kwargs):

        Interpolator.__init__(self, None, *args, **kwargs)
        self.lon_tiepoint = None
        self.lat_tiepoint = None
        try:
            # Maybe it's a pyresample object ?
            self.set_tiepoints(lon_lat_data.lons, lon_lat_data.lats)
            xyz = lon_lat_data.get_cartesian_coords()
            self.tie_data = [xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]]

        except AttributeError:
            self.set_tiepoints(lon_lat_data[0], lon_lat_data[1])
            lons_rad = np.radians(self.lon_tiepoint)
            lats_rad = np.radians(self.lat_tiepoint)
            x__ = EARTH_RADIUS * np.cos(lats_rad) * np.cos(lons_rad)
            y__ = EARTH_RADIUS * np.cos(lats_rad) * np.sin(lons_rad)
            z__ = EARTH_RADIUS * np.sin(lats_rad)
            self.tie_data = [x__, y__, z__]


        self.new_data = []
        for num in range(len(self.tie_data)):
            self.new_data.append([])


    def set_tiepoints(self, lon, lat):
        """Defines the lon,lat tie points.
        """
        self.lon_tiepoint = lon
        self.lat_tiepoint = lat

    def interpolate(self):
        newx, newy, newz = Interpolator.interpolate(self)
        lon = get_lons_from_cartesian(newx, newy)
        lat = get_lats_from_cartesian(newx, newy, newz)
        return lon, lat

def get_lons_from_cartesian(x__, y__):
    """Get longitudes from cartesian coordinates.
    """
    return rad2deg(arccos(x__ / sqrt(x__ ** 2 + y__ ** 2))) * sign(y__)
    
def get_lats_from_cartesian(x__, y__, z__, thr=0.8):
    """Get latitudes from cartesian coordinates.
    """
    # if we are at low latitudes - small z, then get the
    # latitudes only from z. If we are at high latitudes (close to the poles)
    # then derive the latitude using x and y:

    lats = np.where(np.logical_and(np.less(z__, thr * EARTH_RADIUS), 
                                   np.greater(z__, -1. * thr * EARTH_RADIUS)),
                    90 - rad2deg(arccos(z__/EARTH_RADIUS)),
                    sign(z__) *
                    (90 - rad2deg(arcsin(sqrt(x__ ** 2 + y__ ** 2)
                                         / EARTH_RADIUS))))
    return lats

