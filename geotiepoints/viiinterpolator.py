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
Due to the bowtie effect, the pixel sampling pattern is discontinuous at the swath edge. As a result, each scan has
the interpolation is more accurate when performed intra-scan and not inter-scan with discontinuous sampling regions.
This version uses all tie points for individual scans which are then subsequently re-assembled into granules
before return as per PFS (although at visual inspection level there appears to be little difference in the resulting
images). It is also modified to work with vii test data V2 to be released Jan 2022 which has the data stored
in alt, act (row,col) format instead of act,alt (col,row) """

import xarray as xr
import dask.array as da
import numpy as np
from satpy.dataset import combine_metadata

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
    n_pixel_alt=(scan_alt_tie_points-1) * tie_points_factor * n_scans

    # Create the grids used for interpolation across the track
    tie_grid_act = da.arange(0, n_pixel_act + 1, tie_points_factor)
    pixels_grid_act = da.arange(0, n_pixel_act)

    # Create the grids used for the interpolation along the track for the current scan
    n_pixel_alt_scan = (scan_alt_tie_points - 1) * tie_points_factor
    tie_grid_alt_scan = da.arange(0, n_pixel_alt_scan + 1, tie_points_factor)
    pixels_grid_alt_scan = da.arange(0, n_pixel_alt_scan)

    data_on_pixel_points = []

    # loop over granules
    for data in data_on_tie_points:

        # create an xarray to hold the data on pixel grid (note the coords get renamed later so keep the old names)
        rads= da.zeros((n_pixel_alt, n_pixel_act))
        pix_act=da.zeros(n_pixel_act)
        pix_alt=da.zeros(n_pixel_alt)

        data_on_pixel_points_granule=xr.DataArray(rads, dims=['num_tie_points_alt', 'num_tie_points_act'],
                            coords={'num_tie_points_alt': pix_alt, 'num_tie_points_act': pix_act})
        data_on_pixel_points_granule.attrs = combine_metadata(data)


        # loop over scans
        for j_scan in range(n_scans):
            index_tie_points_start = j_scan * scan_alt_tie_points
            index_tie_points_end = (j_scan * scan_alt_tie_points)+scan_alt_tie_points
            index_pixel_start = j_scan * (scan_alt_tie_points-1) * tie_points_factor
            index_pixel_end = (j_scan * (scan_alt_tie_points-1) * tie_points_factor)+(scan_alt_tie_points-1) * tie_points_factor

            data_on_tie_points_scan = data[index_tie_points_start:index_tie_points_end, :]

            data_on_tie_points_scan = data_on_tie_points_scan.assign_coords(
                {dim_alt: tie_grid_alt_scan, dim_act: tie_grid_act})
            data_pixel = data_on_tie_points_scan.interp({dim_alt: pixels_grid_alt_scan}, assume_sorted=True) \
                .interp({dim_act: pixels_grid_act}, assume_sorted=True).drop_vars([dim_alt, dim_act])


            # put the interpolated data in the pixel resolution granule
            data_on_pixel_points_granule[index_pixel_start:index_pixel_end,:] = data_pixel


        data_on_pixel_points.append(data_on_pixel_points_granule)

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
