# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False, initializedcheck=False, nonecheck=False
cimport cython

from ._modis_utils cimport floating
from ._modis_utils cimport lonlat2xyz, xyz2lonlat
from ._modis_utils import rows_per_scan_for_resolution
cimport numpy as np
import numpy as np
from scipy.ndimage import map_coordinates

np.import_array()

def interpolate_geolocation_cartesian(
        np.ndarray[floating, ndim=2] lon_array,
        np.ndarray[floating, ndim=2] lat_array,
        unsigned int coarse_resolution,
        unsigned int fine_resolution):
    lon_array = np.ascontiguousarray(lon_array)
    lat_array = np.ascontiguousarray(lat_array)
    cdef unsigned int rows_per_scan = rows_per_scan_for_resolution(coarse_resolution)
    cdef unsigned int res_factor = coarse_resolution // fine_resolution
    cdef Py_ssize_t num_rows = lon_array.shape[0]
    cdef Py_ssize_t num_cols = lon_array.shape[1]
    cdef unsigned int num_scans = num_rows // rows_per_scan

    # SciPy's map_coordinates requires the x/y dimension to be first
    cdef np.ndarray[floating, ndim=3] coordinates = np.empty(
        (2, res_factor * rows_per_scan, res_factor * num_cols), dtype=lon_array.dtype)
    cdef floating[:, :, ::1] coordinates_view = coordinates
    _compute_yx_coordinate_arrays(res_factor, coordinates_view)

    cdef np.ndarray[floating, ndim=3] xyz_result = np.empty(
        (res_factor * rows_per_scan, num_cols * res_factor, 3), dtype=lon_array.dtype)
    cdef floating[:, :, ::1] xyz_result_view = xyz_result
    cdef np.ndarray[floating, ndim=3] xyz_in = np.empty(
        (rows_per_scan, num_cols, 3), dtype=lon_array.dtype)
    cdef floating[:, :, ::1] xyz_in_view = xyz_in
    cdef floating[:, ::1] lon_in_view = lon_array
    cdef floating[:, ::1] lat_in_view = lat_array

    cdef np.ndarray[floating, ndim=2] new_lons = np.empty((res_factor * num_rows, res_factor * num_cols),
                                                          dtype=lon_array.dtype)
    cdef np.ndarray[floating, ndim=2] new_lats = np.empty((res_factor * num_rows, res_factor * num_cols),
                                                          dtype=lon_array.dtype)
    cdef floating[:, ::1] new_lons_view = new_lons
    cdef floating[:, ::1] new_lats_view = new_lats

    # Interpolate each scan, one at a time, otherwise the math doesn't work well
    cdef Py_ssize_t scan_idx, j0, j1, k0, k1, comp_index
    with nogil:
        for scan_idx in range(num_scans):
            # Calculate indexes
            j0 = rows_per_scan * scan_idx
            j1 = j0 + rows_per_scan
            k0 = rows_per_scan * res_factor * scan_idx
            k1 = k0 + rows_per_scan * res_factor
            lonlat2xyz(lon_in_view[j0:j1, :], lat_in_view[j0:j1, :], xyz_in_view)

            _compute_interpolated_xyz_scan(
                res_factor, coordinates_view, xyz_in_view,
                xyz_result_view)

            xyz2lonlat(xyz_result_view, new_lons_view[k0:k1], new_lats_view[k0:k1], low_lat_z=True)
    return new_lons, new_lats


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _compute_yx_coordinate_arrays(
        unsigned int res_factor,
        floating[:, :, ::1] coordinates,
) noexcept nogil:
    cdef Py_ssize_t i, j
    for j in range(coordinates.shape[1]):
        for i in range(coordinates.shape[2]):
            # y coordinate - 0.375 for 250m, 0.25 for 500m
            coordinates[0, j, i] = j * (1.0 / res_factor) - (res_factor * (1.0 / 16) + (1.0 / 8))
            # x coordinate
            coordinates[1, j, i] = i * (1.0 / res_factor)


@cython.boundscheck(False)
cdef void _compute_interpolated_xyz_scan(
        unsigned int res_factor,
        floating[:, :, ::1] coordinates_view,
        floating[:, :, ::1] xyz_input_view,
        floating[:, :, ::1] xyz_result_view,
) noexcept nogil:
    cdef Py_ssize_t comp_index
    cdef floating[:, :] input_view, result_view
    with gil:
        for comp_index in range(3):
            input_view = xyz_input_view[:, :, comp_index]
            result_view = xyz_result_view[:, :, comp_index]
            _call_map_coordinates(
                input_view,
                coordinates_view,
                result_view,
            )

    if res_factor == 4:
        for comp_index in range(3):
            result_view = xyz_result_view[:, :, comp_index]
            _extrapolate_xyz_rightmost_columns(result_view, 3)
            _interpolate_xyz_250(
                result_view,
                coordinates_view,
            )
    else:
        for comp_index in range(3):
            result_view = xyz_result_view[:, :, comp_index]
            _extrapolate_xyz_rightmost_columns(result_view, 1)
            _interpolate_xyz_500(
                result_view,
                coordinates_view,
            )


cdef void _call_map_coordinates(
        floating[:, :] nav_array_view,
        floating[:, :, ::1] coordinates_view,
        floating[:, :] result_view,
):
    cdef np.ndarray[floating, ndim=2] nav_array = np.asarray(nav_array_view)
    cdef np.ndarray[floating, ndim=3] coordinates_array = np.asarray(coordinates_view)
    cdef np.ndarray[floating, ndim=2] result_array = np.asarray(result_view)
    # Use bilinear interpolation for all 250 meter pixels
    map_coordinates(nav_array, coordinates_array,
                    output=result_array,
                    order=1, mode='nearest')


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _extrapolate_xyz_rightmost_columns(
        floating[:, :] result_view,
        int num_columns,
) noexcept nogil:
    cdef Py_ssize_t row_idx, col_offset
    cdef floating last_interp_col_diff
    for row_idx in range(result_view.shape[0]):
        last_interp_col_diff = result_view[row_idx, result_view.shape[1] - num_columns - 1] - \
                               result_view[row_idx, result_view.shape[1] - num_columns - 2]
        for col_offset in range(num_columns):
            # map_coordinates repeated the last columns value, we now add more to it as an "extrapolation"
            result_view[row_idx, result_view.shape[1] - num_columns + col_offset] += last_interp_col_diff * (col_offset + 1)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _interpolate_xyz_250(
        floating[:, :] result_view,
        floating[:, :, ::1] coordinates_view,
) noexcept nogil:
    cdef Py_ssize_t col_idx
    cdef floating m, b
    cdef floating[:] result_col_view
    cdef floating[:, ::1] y_coordinates = coordinates_view[0]
    for col_idx in range(result_view.shape[1]):
        result_col_view = result_view[:, col_idx]
        # Use linear extrapolation for the first two 250 meter pixels along track
        m = _calc_slope_250(result_col_view,
                            y_coordinates,
                            2)
        b = _calc_offset_250(result_col_view,
                             y_coordinates,
                             m,
                             2)
        result_view[0, col_idx] = m * y_coordinates[0, 0] + b
        result_view[1, col_idx] = m * y_coordinates[1, 0] + b

        # Use linear extrapolation for the last  two 250 meter pixels along track
        # m = (result_array[k0 + 37, :] - result_array[k0 + 34, :]) / (y[37, 0] - y[34, 0])
        # b = result_array[k0 + 37, :] - m * y[37, 0]
        m = _calc_slope_250(result_col_view,
                            y_coordinates,
                            34)
        b = _calc_offset_250(result_col_view,
                             y_coordinates,
                             m,
                             34)
        result_view[38, col_idx] = m * y_coordinates[38, 0] + b
        result_view[39, col_idx] = m * y_coordinates[39, 0] + b


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _interpolate_xyz_500(
        floating[:, :] result_view,
        floating[:, :, ::1] coordinates_view,
) noexcept nogil:
    cdef Py_ssize_t col_idx
    cdef floating m, b
    for col_idx in range(result_view.shape[1]):
        # Use linear extrapolation for the first two 250 meter pixels along track
        m = _calc_slope_500(
            result_view[:, col_idx],
            coordinates_view[0],
            1)
        b = _calc_offset_500(
            result_view[:, col_idx],
            coordinates_view[0],
            m,
            1)
        result_view[0, col_idx] = m * coordinates_view[0, 0, 0] + b

        # Use linear extrapolation for the last two 250 meter pixels along track
        m = _calc_slope_500(
            result_view[:, col_idx],
            coordinates_view[0],
            17)
        b = _calc_offset_500(
            result_view[:, col_idx],
            coordinates_view[0],
            m,
            17)
        result_view[19, col_idx] = m * coordinates_view[0, 19, 0] + b


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline floating _calc_slope_250(
        floating[:] result_view,
        floating[:, ::1] y,
        Py_ssize_t offset,
) noexcept nogil:
    return (result_view[offset + 3] - result_view[offset]) / \
           (y[offset + 3, 0] - y[offset, 0])


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline floating _calc_offset_250(
        floating[:] result_view,
        floating[:, ::1] y,
        floating m,
        Py_ssize_t offset,
) noexcept nogil:
    return result_view[offset + 3] - m * y[offset + 3, 0]


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline floating _calc_slope_500(
        floating[:] result_view,
        floating[:, ::1] y,
        Py_ssize_t offset,
) noexcept nogil:
    return (result_view[offset + 1] - result_view[offset]) / \
           (y[offset + 1, 0] - y[offset, 0])


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline floating _calc_offset_500(
        floating[:] result_view,
        floating[:, ::1] y,
        floating m,
        Py_ssize_t offset,
) noexcept nogil:
    return result_view[offset + 1] - m * y[offset + 1, 0]
