#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2020 Python-geotiepoints developers
#
# This file is part of python-geotiepoints.
#
# python-geotiepoints is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# python-geotiepoints is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# python-geotiepoints.  If not, see <http://www.gnu.org/licenses/>.

"""Interpolation of geographical tiepoints for the VII products.
It follows the description provided in document "EPS-SG VII Level 1B Product Format Specification".
Tiepoints are typically subsampled by a factor 8 with respect to the pixels, along and across the satellite track.
Because of the bowtie effect, tiepoints at the scan edges are not continuous between neighbouring scans,
therefore each scan has its own edge tiepoints in the along-track direction.
However, for computational efficiency, the edge tie points that lie outside the original data point raster
are excluded from the interpolation grid which is then carried out per granule, rather than per scan
at a (very) small geolocation accuracy cost at the swath edge (investigation to quantify this ongoing).
The interpolation functions are implemented for xarray.DataArrays as input.
This version works with vii test data V2 to be released Jan 2022 which has the data stored
in alt, act (row,col) format instead of act,alt (col,row)
"""

import xarray as xr
import dask.array as da
import numpy as np

# MEAN EARTH RADIUS AS DEFINED BY IUGG
MEAN_EARTH_RADIUS = 6371008.7714  # [m]


def tie_points_interpolation(data_on_tie_points, scan_alt_tie_points, tie_points_factor):
    """Interpolate the data from the tie points to the pixel points.
    The data are provided as a list of xarray DataArray objects, allowing to interpolate on several arrays
    at the same time; however the individual arrays must have exactly the same dimensions.
    Args:
        data_on_tie_points: list of xarray DataArray objects containing the values defined on the tie points.
        scan_alt_tie_points: number of tie points along the satellite track for each scan
        tie_points_factor: sub-sampling factor of tie points wrt pixel points
    Returns:
        list of xarray DataArray objects containing the interpolated values on the pixel points.
    """
    # Extract the dimensions of the tie points array across and along track
    n_tie_alt, n_tie_act = data_on_tie_points[0].shape
    dim_alt, dim_act = data_on_tie_points[0].dims

    # Check that the number of tie points along track is multiple of the number of tie points per scan
    if n_tie_alt % scan_alt_tie_points != 0:
        raise ValueError("The number of tie points in the along-route dimension must be a multiple of %d",
                         scan_alt_tie_points)

    # Compute the number of scans
    n_scans = n_tie_alt // scan_alt_tie_points

    # Compute the dimensions of the pixel points array across and along track
    n_pixel_act = (n_tie_act - 1) * tie_points_factor
    n_pixel_alt = (n_tie_alt - 1) * tie_points_factor

    # Create the grids used for interpolation across the track
    tie_grid_act = da.arange(0, n_pixel_act + 1, tie_points_factor)
    pixel_grid_act = da.arange(0, n_pixel_act)

    # Create the grids used for the interpolation along the track (must not include the spurious points between scans)
    tie_grid_alt = da.arange(0, n_pixel_alt + 1, tie_points_factor)
    n_pixel_alt_per_scan = (scan_alt_tie_points - 1) * tie_points_factor
    pixel_grid_alt = []

    for j_scan in range(n_scans):
        start_index_scan = j_scan * scan_alt_tie_points * tie_points_factor
        pixel_grid_alt.append(da.arange(start_index_scan, start_index_scan + n_pixel_alt_per_scan))
    pixel_grid_alt = da.concatenate(pixel_grid_alt)

    # Loop on all arrays
    data_on_pixel_points = []
    for data in data_on_tie_points:

        if data.shape != (n_tie_alt, n_tie_act) or data.dims != (dim_alt, dim_act):
            raise ValueError("The dimensions of the arrays are not consistent")

        # Interpolate using the xarray interp function twice: first across, then along the scan
        # (much faster than interpolating directly in the two dimensions)
        data = data.assign_coords({dim_alt: tie_grid_alt, dim_act: tie_grid_act})
        data_pixel = data.interp({dim_alt: pixel_grid_alt}, assume_sorted=True) \
                         .interp({dim_act: pixel_grid_act}, assume_sorted=True).drop_vars([dim_alt, dim_act])

        data_on_pixel_points.append(data_pixel)

    return data_on_pixel_points



def tie_points_geo_interpolation(longitude, latitude,
                                 scan_alt_tie_points, tie_points_factor,
                                 lat_threshold_use_cartesian=60.,
                                 z_threshold_use_xy=0.8):
    """Interpolate the geographical position from the tie points to the pixel points.

    The longitude and latitude values are provided as xarray DataArray objects.

    Args:
        data_on_tie_points: list of xarray DataArray objects containing the values defined on the tie points.
        scan_alt_tie_points: number of tie points along the satellite track for each scan
        tie_points_factor: sub-sampling factor of tie points wrt pixel points

    Returns:
        list of xarray DataArray objects containing the interpolated values on the pixel points.

    Args:
        longitude: xarray DataArray containing the longitude values defined on the tie points (degrees).
        latitude: xarray DataArray containing the latitude values defined on the tie points (degrees).
        scan_alt_tie_points: number of tie points along the satellite track for each scan.
        tie_points_factor: sub-sampling factor of tie points wrt pixel points.
        lat_threshold_use_cartesian: latitude threshold to use cartesian coordinates.
        z_threshold_use_xy: z threshold to compute latitude from x and y in cartesian coordinates.

    Returns:
        two xarray DataArray objects containing the interpolated longitude and latitude values on the pixel points.

    """
    # Check that the two arrays have the same dimensions
    if longitude.shape != latitude.shape:
        raise ValueError("The dimensions of longitude and latitude don't match")

    # Determine if the interpolation should be done in cartesian or geodetic coordinates
    to_cart = np.max(np.fabs(latitude)) > lat_threshold_use_cartesian or (np.max(longitude) - np.min(longitude)) > 180.

    if to_cart:

        x, y, z = _lonlat2xyz(longitude, latitude)

        interp_x, interp_y, interp_z = tie_points_interpolation([x, y, z],
                                                                scan_alt_tie_points,
                                                                tie_points_factor)

        interp_longitude, interp_latitude = _xyz2lonlat(interp_x, interp_y, interp_z, z_threshold_use_xy)

    else:

        interp_longitude, interp_latitude = tie_points_interpolation([longitude, latitude],
                                                                     scan_alt_tie_points,
                                                                     tie_points_factor)

    return interp_longitude, interp_latitude


def _lonlat2xyz(lons, lats):
    """Convert longitudes and latitudes to cartesian coordinates.

    Args:
        lons: array containing the longitude values in degrees.
        lats: array containing the latitude values in degrees.

    Returns:
        tuple of arrays containing the x, y, and z values in meters.

    """
    lons_rad = np.deg2rad(lons)
    lats_rad = np.deg2rad(lats)
    x_coords = MEAN_EARTH_RADIUS * np.cos(lats_rad) * np.cos(lons_rad)
    y_coords = MEAN_EARTH_RADIUS * np.cos(lats_rad) * np.sin(lons_rad)
    z_coords = MEAN_EARTH_RADIUS * np.sin(lats_rad)
    return x_coords, y_coords, z_coords


def _xyz2lonlat(x_coords, y_coords, z_coords, z_threshold_use_xy=0.8):
    """Get longitudes and latitudes from cartesian coordinates.

    Args:
        x_coords: array containing the x values in meters.
        y_coords: array containing the y values in meters.
        z_coords: array containing the z values in meters.
        z_threshold_use_xy: z threshold to compute latitude from x and y in cartesian coordinates.

    Returns:
        tuple of arrays containing the longitude and latitude values in degrees.

    """
    r = np.sqrt(x_coords ** 2 + y_coords ** 2)
    thr_z = z_threshold_use_xy * MEAN_EARTH_RADIUS
    lons = np.rad2deg(np.arccos(x_coords / r)) * np.sign(y_coords)
    # Compute latitude from z at low z and from x and y at high z
    lats = xr.where(
        np.logical_and(np.less(z_coords, thr_z), np.greater(z_coords, -thr_z)),
        90. - np.rad2deg(np.arccos(z_coords / MEAN_EARTH_RADIUS)),
        np.sign(z_coords) * (90. - np.rad2deg(np.arcsin(r / MEAN_EARTH_RADIUS)))
    )
    return lons, lats
