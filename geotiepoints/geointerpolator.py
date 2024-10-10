#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
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
"""Geographical interpolation (lon/lats)."""

import numpy as np
from geotiepoints.interpolator import Interpolator, MultipleGridInterpolator, MultipleSplineInterpolator


EARTH_RADIUS = 6370997.0


class GeoInterpolator(Interpolator):
    """Handles interpolation of geolocation from a grid of tie points.

    It is
    preferable to have tie-points out till the edges if the tiepoint grid, but
    a method is provided to extrapolate linearly the tiepoints to the borders
    of the grid. The extrapolation is done automatically if it seems necessary.

    Uses numpy, scipy, and optionally pyresample.

    The constructor takes in the tiepointed data as *data*, the
    *tiepoint_grid* and the desired *final_grid*. As optional arguments, one
    can provide *kx_* and *ky_* as interpolation orders (in x and y directions
    respectively), and the *chunksize* if the data has to be handled by pieces
    along the y axis (this affects how the extrapolator behaves). If
    *chunksize* is set, don't forget to adjust the interpolation orders
    accordingly: the interpolation is indeed done globaly (not chunkwise).

    """

    def __init__(self, lon_lat_data, *args, **kwargs):
        try:
            # Maybe it's a pyresample object ?
            self.lon_tiepoint = lon_lat_data.lons
            self.lat_tiepoint = lon_lat_data.lats
            xyz = lon_lat_data.get_cartesian_coords()
            tie_data = [xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]]
        except AttributeError:
            self.lon_tiepoint = lon_lat_data[0]
            self.lat_tiepoint = lon_lat_data[1]
            x__, y__, z__ = lonlat2xyz(self.lon_tiepoint, self.lat_tiepoint)
            tie_data = [x__, y__, z__]

        super().__init__(tie_data, *args, **kwargs)

    def interpolate(self):
        """Run the interpolation."""
        newx, newy, newz = super().interpolate()
        lon, lat = xyz2lonlat(newx, newy, newz)
        return lon, lat


def lonlat2xyz(lons, lats, radius=EARTH_RADIUS):
    """Convert lons and lats to cartesian coordinates."""
    lons_rad = np.deg2rad(lons)
    lats_rad = np.deg2rad(lats)
    x_coords = radius * np.cos(lats_rad) * np.cos(lons_rad)
    y_coords = radius * np.cos(lats_rad) * np.sin(lons_rad)
    z_coords = radius * np.sin(lats_rad)
    return x_coords, y_coords, z_coords


def xyz2lonlat(x__, y__, z__, radius=EARTH_RADIUS, thr=0.8, low_lat_z=True):
    """Get longitudes from cartesian coordinates."""
    lons = np.rad2deg(np.arccos(x__ / np.sqrt(x__ ** 2 + y__ ** 2))) * np.sign(y__)
    lats = np.sign(z__) * (90 - np.rad2deg(np.arcsin(np.sqrt(x__ ** 2 + y__ ** 2) / radius)))
    if low_lat_z:
        # if we are at low latitudes - small z, then get the
        # latitudes only from z. If we are at high latitudes (close to the poles)
        # then derive the latitude using x and y:
        normalized_z = z__ / radius
        lat_mask_cond = abs(normalized_z) < thr
        lat_z_only = 90 - np.rad2deg(np.arccos(normalized_z))
        lats = np.where(lat_mask_cond, lat_z_only, lats)

    return lons, lats


def _work_with_lonlats(klass):
    """Adapt MultipleInterpolator classes to work with geographical coordinates."""

    class GeoKlass(klass):

        def __init__(self, tie_points, *data, **interpolator_init_kwargs):
            """Set up the interpolator."""
            data = to_xyz(data)
            super().__init__(tie_points, *data, **interpolator_init_kwargs)

        def interpolate(self, fine_points, **interpolator_call_kwargs):
            """Interpolate to *fine_points*."""
            x, y, z = super().interpolate(fine_points, **interpolator_call_kwargs)
            return xyz2lonlat(x, y, z)

    return GeoKlass


def to_xyz(data):
    """Convert data to cartesian.

    Data can be a class with a `get_cartesian_coords` method, or a tuple of (lon, lat) arrays.
    """
    if len(data) == 1:
        xyz = data[0].get_cartesian_coords()
        data = [xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]]
    elif len(data) == 2:
        data = lonlat2xyz(*data)
    else:
        raise ValueError("Either pass lon/lats or a pyresample definition.")
    return data


GeoGridInterpolator = _work_with_lonlats(MultipleGridInterpolator)
GeoSplineInterpolator = _work_with_lonlats(MultipleSplineInterpolator)
