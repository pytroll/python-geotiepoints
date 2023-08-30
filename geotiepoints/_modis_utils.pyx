# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False, initializedcheck=False, nonecheck=False
from functools import wraps

cimport cython
cimport numpy as np
from libc.math cimport asin, sin, cos, sqrt, acos, M_PI
import numpy as np

np.import_array()

try:
    import dask.array as da
except ImportError:
    # if dask can't be imported then we aren't going to be given dask arrays
    da = None

try:
    import xarray as xr
except ImportError:
    xr = None


DEF EARTH_RADIUS = 6370997.0


cdef void lonlat2xyz(
        floating[:, ::1] lons,
        floating[:, ::1] lats,
        floating[:, :, ::1] xyz,
) noexcept nogil:
    """Convert lons and lats to cartesian coordinates."""
    cdef Py_ssize_t i, j, k
    cdef floating lon_rad, lat_rad
    for i in range(lons.shape[0]):
        for j in range(lons.shape[1]):
            lon_rad = deg2rad(lons[i, j])
            lat_rad = deg2rad(lats[i, j])
            xyz[i, j, 0] = EARTH_RADIUS * cos(lat_rad) * cos(lon_rad)
            xyz[i, j, 1] = EARTH_RADIUS * cos(lat_rad) * sin(lon_rad)
            xyz[i, j, 2] = EARTH_RADIUS * sin(lat_rad)


cdef void xyz2lonlat(
        floating[:, :, ::1] xyz,
        floating[:, ::1] lons,
        floating[:, ::1] lats,
        bint low_lat_z=True,
        floating thr=0.8) noexcept nogil:
    """Get longitudes from cartesian coordinates."""
    cdef Py_ssize_t i, j
    cdef np.float64_t x, y, z
    for i in range(xyz.shape[0]):
        for j in range(xyz.shape[1]):
            # 64-bit precision matters apparently
            x = <np.float64_t>xyz[i, j, 0]
            y = <np.float64_t>xyz[i, j, 1]
            z = <np.float64_t>xyz[i, j, 2]
            lons[i, j] = rad2deg(acos(x / sqrt(x ** 2 + y ** 2))) * _sign(y)
            # if we are at low latitudes - small z, then get the
            # latitudes only from z. If we are at high latitudes (close to the poles)
            # then derive the latitude using x and y:
            if low_lat_z and (z < thr * EARTH_RADIUS) and (z > -1.0 * thr * EARTH_RADIUS):
                lats[i, j] = 90 - rad2deg(acos(z / EARTH_RADIUS))
            else:
                lats[i, j] = _sign(z) * (90 - rad2deg(asin(sqrt(x ** 2 + y ** 2) / EARTH_RADIUS)))


cdef inline int _sign(floating x) noexcept nogil:
    return 1 if x > 0 else (-1 if x < 0 else 0)


cdef inline floating rad2deg(floating x) noexcept nogil:
    return x * (180.0 / M_PI)


cdef inline floating deg2rad(floating x) noexcept nogil:
    return x * (M_PI / 180.0)


def rows_per_scan_for_resolution(res):
    return {
        5000: 2,
        1000: 10,
        500: 20,
        250: 40,
    }[res]


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
            rows_per_scan = rows_per_scan_for_resolution(coarse_resolution)
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
    num_cols = first_arr.shape[1]
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


