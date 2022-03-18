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

from functools import wraps

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from .geointerpolator import lonlat2xyz, xyz2lonlat
from . import _simple_modis_interpolator

try:
    import dask.array as da
except ImportError:
    # if dask can't be imported then we aren't going to be given dask arrays
    da = None

try:
    import xarray as xr
except ImportError:
    xr = None


def scanline_mapblocks(func):
    """Convert dask array inputs to appropriate map_blocks calls.

    This function, applied as a decorator, will call the wrapped function
    using dask's ``map_blocks``. It will rechunk dask array inputs when
    necessary to make sure that the input chunks are entire scanlines to
    avoid incorrect interpolation.

    """
    @wraps(func)
    def _wrapper(*args, coarse_resolution=None, fine_resolution=None, **kwargs):
        if coarse_resolution is None or fine_resolution is None:
            raise ValueError("'coarse_resolution' and 'fine_resolution' are required keyword arguments.")
        first_arr = [arr for arr in args if hasattr(arr, "ndim")][0]
        if first_arr.ndim != 2 or first_arr.ndim != 2:
            raise ValueError("Expected 2D input arrays.")
        if hasattr(first_arr, "compute"):
            # assume it is dask or xarray with dask, ensure proper chunk size
            # if DataArray get just the dask array
            dask_args = _extract_dask_arrays_from_args(args)
            rows_per_scan = _simple_modis_interpolator.rows_per_scan_for_resolution(coarse_resolution)
            rechunked_args = _rechunk_dask_arrays_if_needed(dask_args, rows_per_scan)
            results = _call_map_blocks_interp(
                func,
                coarse_resolution,
                fine_resolution,
                *rechunked_args,
                **kwargs
            )
            if hasattr(first_arr, "dims"):
                # recreate DataArrays
                results = _results_to_data_arrays(first_arr.dims, *results)
            return results
        return func(
            *args,
            coarse_resolution=coarse_resolution,
            fine_resolution=fine_resolution,
            **kwargs
        )

    return _wrapper


def _extract_dask_arrays_from_args(args):
    return [arr_obj.data if hasattr(arr_obj, "dims") else arr_obj for arr_obj in args]


def _call_map_blocks_interp(func, coarse_resolution, fine_resolution, *args, **kwargs):
    first_arr = [arr for arr in args if hasattr(arr, "ndim")][0]
    res_factor = coarse_resolution // fine_resolution
    new_row_chunks = tuple(x * res_factor for x in first_arr.chunks[0])
    fine_pixels_per_1km = {250: 4, 500: 2, 1000: 1}[fine_resolution]
    fine_scan_width = 1354 * fine_pixels_per_1km
    new_col_chunks = (fine_scan_width,)
    wrapped_func = _map_blocks_handler(func)
    res = da.map_blocks(wrapped_func, *args,
                        coarse_resolution=coarse_resolution,
                        fine_resolution=fine_resolution,
                        **kwargs,
                        new_axis=[0],
                        chunks=(2, new_row_chunks, new_col_chunks),
                        dtype=first_arr.dtype,
                        meta=np.empty((2, 2, 2), dtype=first_arr.dtype))
    return tuple(res[idx] for idx in range(res.shape[0]))


def _results_to_data_arrays(dims, *results):
    new_results = []
    for result in results:
        if not isinstance(result, da.Array):
            continue
        data_arr = xr.DataArray(result, dims=dims)
        new_results.append(data_arr)
    return new_results


def _rechunk_dask_arrays_if_needed(args, rows_per_scan: int):
    # take current chunk size and get a relatively similar chunk size
    first_arr = [arr for arr in args if hasattr(arr, "ndim")][0]
    row_chunks = first_arr.chunks[0]
    col_chunks = first_arr.chunks[1]
    num_rows = first_arr.shape[0]
    num_cols = first_arr.shape[-1]
    good_row_chunks = all(x % rows_per_scan == 0 for x in row_chunks)
    good_col_chunks = len(col_chunks) == 1 and col_chunks[0] != num_cols
    all_orig_chunks = [arr.chunks for arr in args if hasattr(arr, "chunks")]

    if num_rows % rows_per_scan != 0:
        raise ValueError("Input longitude/latitude data does not consist of "
                         "whole scans (10 rows per scan).")
    all_same_chunks = all(
        all_orig_chunks[0] == some_chunks
        for some_chunks in all_orig_chunks[1:]
    )
    if good_row_chunks and good_col_chunks and all_same_chunks:
        return args

    new_row_chunks = (row_chunks[0] // rows_per_scan) * rows_per_scan
    new_args = [arr.rechunk((new_row_chunks, -1)) if hasattr(arr, "chunks") else arr for arr in args]
    return new_args


def _map_blocks_handler(func):
    @wraps(func)
    def _map_blocks_wrapper(*args, **kwargs):
        results = func(*args, **kwargs)
        return np.concatenate(
            tuple(result[np.newaxis] for result in results),
            axis=0)
    return _map_blocks_wrapper


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
    return _simple_modis_interpolator.interpolate_geolocation_cartesian(
        lon_array, lat_array, coarse_resolution, fine_resolution
    )


def _calc_slope_offset_250(result_array, y, start_idx, offset):
    m = (result_array[start_idx + offset + 3, :] - result_array[start_idx + offset, :]) / \
        (y[offset + 3, 0] - y[offset, 0])
    b = result_array[start_idx + offset + 3, :] - m * y[offset + 3, 0]
    return m, b


def _calc_slope_offset_500(result_array, y, start_idx, offset):
    m = (result_array[start_idx + offset + 1, :] - result_array[start_idx + offset, :]) / \
        (y[offset + 1, 0] - y[offset, 0])
    b = result_array[start_idx + offset + 1, :] - m * y[offset + 1, 0]
    return m, b


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
