#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Python-geotiepoints developers
#
# This file is part of python-geotiepoints.
#
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
"""Interpolate MODIS 1km navigation arrays to 250m and 500m resolutions.

The code used here is a rewrite of the IDL function ``MODIS_GEO_INTERP_250``
used by Liam Gumley. It has been modified to convert coordinates to cartesian
(X, Y, Z) coordinates first to avoid problems with the anti-meridian and poles.
This code was originally part of the CSPP Polar2Grid project, but has been
moved here for integration with Satpy and newer versions of Polar2Grid.

This algorithm differs from the one in ``modisinterpolator`` as it only
requires the original longitude and latitude arrays. This is useful in the
case of reading the 250m or 500m MODIS L1b files or any MODIS L2 files without
including the MOD03 geolocation file as there is no SensorZenith angle dataset
in these files.

"""

from ._modis_utils import scanline_mapblocks
from ._simple_modis_interpolator import interpolate_geolocation_cartesian as interp_cython


@scanline_mapblocks
def interpolate_geolocation_cartesian(lon_array, lat_array, coarse_resolution, fine_resolution):
    """Interpolate MODIS navigation from 1000m resolution to 250m.

    Python rewrite of the IDL function ``MODIS_GEO_INTERP_250`` but converts to cartesian (X, Y, Z) coordinates
    first to avoid problems with the anti-meridian/poles.

    Arguments:
        lon_array: Longitude data as a 2D numpy, dask, or xarray DataArray object.
            The input data is expected to represent 1000m geolocation.
        lat_array: Latitude data as a 2D numpy, dask, or xarray DataArray object.
            The input data is expected to represent 1000m geolocation.
        res_factor (int): Expansion factor for the function. Should be 2 for
            500m output or 4 for 250m output.

    Returns:
        A two-element tuple (lon, lat).

    """
    return interp_cython(
        lon_array, lat_array, coarse_resolution, fine_resolution
    )


def modis_1km_to_250m(lon1, lat1):
    """Interpolate MODIS geolocation from 1km to 250m resolution."""
    return interpolate_geolocation_cartesian(
        lon1,
        lat1,
        coarse_resolution=1000,
        fine_resolution=250,
    )


def modis_1km_to_500m(lon1, lat1):
    """Interpolate MODIS geolocation from 1km to 500m resolution."""
    return interpolate_geolocation_cartesian(
        lon1,
        lat1,
        coarse_resolution=1000,
        fine_resolution=500)
