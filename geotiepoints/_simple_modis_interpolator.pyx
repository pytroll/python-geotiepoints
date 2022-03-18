cimport cython

from ._modis_utils cimport floating
from ._modis_utils cimport xyz2lonlat as xyz2lonlat_cython
from ._modis_utils import rows_per_scan_for_resolution
from libc.math cimport asin, sin, cos, sqrt, acos, M_PI
cimport numpy as np
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from .geointerpolator import lonlat2xyz
from .geointerpolator import xyz2lonlat


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
    _compute_xy_coordinate_arrays(res_factor, coordinates_view)

    cdef np.ndarray[floating, ndim=3] xyz_result = np.empty(
        (num_rows * res_factor, num_cols * res_factor, 3), dtype=lon_array.dtype)
    cdef floating[:, :, ::1] xyz_result_view = xyz_result
    cdef list xyz_input = [x_in, y_in, z_in]

    # Interpolate each scan, one at a time, otherwise the math doesn't work well
    cdef Py_ssize_t scan_idx, j0, j1, k0, k1, comp_index
    for scan_idx in range(num_scans):
        # Calculate indexes
        j0 = rows_per_scan * scan_idx
        j1 = j0 + rows_per_scan
        k0 = rows_per_scan * res_factor * scan_idx
        k1 = k0 + rows_per_scan * res_factor

        for comp_index in range(3):
            nav_array = xyz_input[comp_index]
            result_array = xyz_result[:, :, comp_index]
            # Use bilinear interpolation for all 250 meter pixels
            map_coordinates(nav_array[j0:j1, :], coordinates,
                            output=xyz_result[k0:k1, :, comp_index],
                            order=1, mode='nearest')

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

    cdef np.ndarray[floating, ndim=2] new_lons = np.empty((xyz_result_view.shape[0], xyz_result_view.shape[1]),
                                                          dtype=lon_array.dtype)
    cdef np.ndarray[floating, ndim=2] new_lats = np.empty((xyz_result_view.shape[0], xyz_result_view.shape[1]),
                                                          dtype=lon_array.dtype)
    cdef floating[:, ::1] new_lons_view = new_lons
    cdef floating[:, ::1] new_lats_view = new_lats
    xyz2lonlat_cython(xyz_result_view, new_lons_view, new_lats_view, low_lat_z=True)
    return new_lons, new_lats



cdef void _compute_xy_coordinate_arrays(
        unsigned int res_factor,
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


