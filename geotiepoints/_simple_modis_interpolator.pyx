cimport cython

from ._modis_utils cimport floating
from ._modis_utils cimport xyz2lonlat as xyz2lonlat_cython
from ._modis_utils import rows_per_scan_for_resolution
cimport numpy as np
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from .geointerpolator import lonlat2xyz


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
    _compute_xy_coordinate_arrays(res_factor, coordinates_view)

    cdef np.ndarray[floating, ndim=3] xyz_result = np.empty(
        (res_factor * rows_per_scan, num_cols * res_factor, 3), dtype=lon_array.dtype)
    cdef floating[:, :, ::1] xyz_result_view = xyz_result
    cdef np.ndarray[floating, ndim=2] x_in, y_in, z_in
    cdef list xyz_input

    cdef np.ndarray[floating, ndim=2] new_lons = np.empty((res_factor * num_rows, res_factor * num_cols),
                                                          dtype=lon_array.dtype)
    cdef np.ndarray[floating, ndim=2] new_lats = np.empty((res_factor * num_rows, res_factor * num_cols),
                                                          dtype=lon_array.dtype)
    cdef floating[:, ::1] new_lons_view = new_lons
    cdef floating[:, ::1] new_lats_view = new_lats

    # Interpolate each scan, one at a time, otherwise the math doesn't work well
    cdef Py_ssize_t scan_idx, j0, j1, k0, k1, comp_index
    for scan_idx in range(num_scans):
        # Calculate indexes
        j0 = rows_per_scan * scan_idx
        j1 = j0 + rows_per_scan
        k0 = rows_per_scan * res_factor * scan_idx
        k1 = k0 + rows_per_scan * res_factor

        # TODO: Use lon/lat views and cython version of lonlat2xyz
        #   Use .reshape or similar to get a 3D view of the lon/lat arrays and 4D view of the xyz 3D array (scan, rows, cols, xyz)
        #   If we did this, could we declare all memory views as `::1`
        x_in, y_in, z_in = lonlat2xyz(lon_array[j0:j1], lat_array[j0:j1])
        xyz_input = [x_in, y_in, z_in]

        _compute_interpolated_xyz_scan(
            res_factor, coordinates, xyz_input,
            xyz_result_view)

        xyz2lonlat_cython(xyz_result_view, new_lons_view[k0:k1], new_lats_view[k0:k1], low_lat_z=True)
    return new_lons, new_lats


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
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


cdef void _compute_interpolated_xyz_scan(
        unsigned int res_factor,
        np.ndarray[floating, ndim=3] coordinates,
        list xyz_input,
        floating[:, :, ::1] xyz_result_view,
):
    cdef Py_ssize_t comp_index
    cdef np.ndarray[floating, ndim=2] nav_array, result_array
    cdef floating[:, :] result_view
    cdef floating[:, :, :] coordinates_view = coordinates
    for comp_index in range(3):
        nav_array = xyz_input[comp_index]
        result_view = xyz_result_view[:, :, comp_index]
        result_array = np.asarray(result_view)
        # Use bilinear interpolation for all 250 meter pixels
        map_coordinates(nav_array, coordinates,
                        output=result_array,
                        order=1, mode='nearest')
        if res_factor == 4:
            _interpolate_xyz_250(
                result_view,
                coordinates_view,
            )
        else:
            _interpolate_xyz_500(
                result_view,
                coordinates_view,
            )


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void _interpolate_xyz_250(
        floating[:, :] result_view,
        floating[:, :, :] coordinates_view,
) nogil:
    cdef Py_ssize_t col_idx
    cdef floating m, b
    cdef floating[:] result_col_view
    cdef floating[:, :] y_coordinates = coordinates_view[0]
    # FIXME: This doesn't need to run for every *row*, it only uses a few of them at a time.
    for col_idx in range(result_view.shape[1]):
        result_col_view = result_view[:, col_idx]
        # Use linear extrapolation for the first two 250 meter pixels along track
        # m = _calc_slope_250(result_view[:, col_idx],
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
cdef void _interpolate_xyz_500(
        floating[:, :] result_view,
        floating[:, :, :] coordinates_view,
) nogil:
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
cdef inline floating _calc_slope_250(
        floating[:] result_view,
        floating[:, :] y,
        Py_ssize_t offset,
) nogil:
    return (result_view[offset + 3] - result_view[offset]) / \
           (y[offset + 3, 0] - y[offset, 0])


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef inline floating _calc_offset_250(
        floating[:] result_view,
        floating[:, :] y,
        floating m,
        Py_ssize_t offset,
) nogil:
    return result_view[offset + 3] - m * y[offset + 3, 0]


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef inline floating _calc_slope_500(
        floating[:] result_view,
        floating[:, :] y,
        Py_ssize_t offset,
) nogil:
    return (result_view[offset + 1] - result_view[offset]) / \
           (y[offset + 1, 0] - y[offset, 0])


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef inline floating _calc_offset_500(
        floating[:] result_view,
        floating[:, :] y,
        floating m,
        Py_ssize_t offset,
) nogil:
    return result_view[offset + 1] - m * y[offset + 1, 0]
