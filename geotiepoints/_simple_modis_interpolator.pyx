cimport cython

cimport numpy as np
from scipy.ndimage.interpolation import map_coordinates
from .geointerpolator import lonlat2xyz, xyz2lonlat
from libc.math cimport asin, sin, cos, sqrt, acos, M_PI
import numpy as np

ctypedef fused floating:
    np.float32_t
    np.float64_t


def rows_per_scan_for_resolution(res):
    return {
        5000: 2,
        1000: 10,
        500: 20,
        250: 40,
    }[res]


def interpolate_geolocation_cartesian(
        np.ndarray[floating, ndim=2] lon_array,
        np.ndarray[floating, ndim=2] lat_array,
        unsigned int coarse_resolution,
        unsigned int fine_resolution):
    cdef unsigned int rows_per_scan = rows_per_scan_for_resolution(coarse_resolution)
    cdef unsigned int res_factor = coarse_resolution // fine_resolution
    cdef Py_ssize_t num_rows = lon_array.shape[0]
    cdef Py_ssize_t num_cols = lon_array.shape[1]
    cdef unsigned int num_scans = num_rows // rows_per_scan
    cdef np.ndarray[floating, ndim=2] x_in, y_in, z_in
    # TODO: Use lon/lat views and cython version of lonlat2xyz
    x_in, y_in, z_in = lonlat2xyz(lon_array, lat_array)

    cdef np.ndarray[floating, ndim=3] coordinates = np.empty(
        (2, res_factor * rows_per_scan, res_factor * num_cols), dtype=lon_array.dtype)
    cdef floating[:, :, ::1] coordinates_view = coordinates
    _compute_xy_coordinate_arrays(num_cols, res_factor, rows_per_scan, coordinates_view)

    new_x = np.empty((num_rows * res_factor, num_cols * res_factor), dtype=lon_array.dtype)
    new_y = new_x.copy()
    new_z = new_x.copy()
    nav_arrays = [(x_in, new_x), (y_in, new_y), (z_in, new_z)]

    # Interpolate each scan, one at a time, otherwise the math doesn't work well
    for scan_idx in range(num_scans):
        # Calculate indexes
        j0 = rows_per_scan * scan_idx
        j1 = j0 + rows_per_scan
        k0 = rows_per_scan * res_factor * scan_idx
        k1 = k0 + rows_per_scan * res_factor

        for nav_array, result_array in nav_arrays:
            # Use bilinear interpolation for all 250 meter pixels
            map_coordinates(nav_array[j0:j1, :], coordinates, output=result_array[k0:k1, :], order=1, mode='nearest')

            if res_factor == 4:
                # Use linear extrapolation for the first two 250 meter pixels along track
                m, b = _calc_slope_offset_250(result_array, coordinates[0], k0, 2)
                result_array[k0 + 0, :] = m * coordinates[0, 0, 0] + b
                result_array[k0 + 1, :] = m * coordinates[0, 1, 0] + b

                # Use linear extrapolation for the last  two 250 meter pixels along track
                # m = (result_array[k0 + 37, :] - result_array[k0 + 34, :]) / (y[37, 0] - y[34, 0])
                # b = result_array[k0 + 37, :] - m * y[37, 0]
                m, b = _calc_slope_offset_250(result_array, coordinates[0], k0, 34)
                result_array[k0 + 38, :] = m * coordinates[0, 38, 0] + b
                result_array[k0 + 39, :] = m * coordinates[0, 39, 0] + b
            else:
                # 500m
                # Use linear extrapolation for the first two 250 meter pixels along track
                m, b = _calc_slope_offset_500(result_array, coordinates[0], k0, 1)
                result_array[k0 + 0, :] = m * coordinates[0, 0, 0] + b

                # Use linear extrapolation for the last two 250 meter pixels along track
                m, b = _calc_slope_offset_500(result_array, coordinates[0], k0, 17)
                result_array[k0 + 19, :] = m * coordinates[0, 19, 0] + b

    new_lons, new_lats = xyz2lonlat(new_x, new_y, new_z, low_lat_z=True)
    return new_lons.astype(lon_array.dtype), new_lats.astype(lon_array.dtype)


cdef void _compute_xy_coordinate_arrays(
        Py_ssize_t num_cols,
        unsigned int res_factor,
        unsigned int rows_per_scan,
        floating[:, :, ::1] coordinates,
) nogil:
    cdef Py_ssize_t i, j
    for j in range(coordinates.shape[1]):
        for i in range(coordinates.shape[2]):
            # y coordinate - 0.375 for 250m, 0.25 for 500m
            coordinates[0, j, i] = j * (1.0 / res_factor) - (res_factor * (1.0 / 16) + (1.0 / 8))
            # x coordinate
            coordinates[1, j, i] = i * (1.0 / res_factor)


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


