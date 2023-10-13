# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False, initializedcheck=False, nonecheck=False
cimport cython
from ._modis_utils cimport lonlat2xyz, xyz2lonlat, floating, deg2rad
from .simple_modis_interpolator import scanline_mapblocks

from libc.math cimport asin, sin, cos, sqrt
cimport numpy as np
import numpy as np

DEF R = 6370.997
# Aqua altitude in km
DEF H = 709.0

np.import_array()


@scanline_mapblocks
def interpolate(
        np.ndarray[floating, ndim=2] lon1,
        np.ndarray[floating, ndim=2] lat1,
        np.ndarray[floating, ndim=2] satz1,
        unsigned int coarse_resolution=0,
        unsigned int fine_resolution=0,
        unsigned int coarse_scan_width=0,
):
    """Helper function to interpolate scan-aligned arrays.

    This function's decorator runs this function for each dask block/chunk of
    scans. The arrays are scan-aligned meaning they are an even number of scans
    (N rows per scan) and contain the entire scan width.

    """
    if coarse_resolution == 5000 and coarse_scan_width not in (0, 270, 271):
        raise NotImplementedError(
            "Can't interpolate if 5km tiepoints have less than 270 columns."
        )
    interp = MODISInterpolator(coarse_resolution, fine_resolution, coarse_scan_width=coarse_scan_width or 0)
    return interp.interpolate(lon1, lat1, satz1)


cdef void _compute_expansion_alignment(floating[:, :] satz_a, floating [:, :] satz_b, int scan_width,
                                       floating[:, ::1] c_expansion, floating[:, ::1] c_alignment) noexcept nogil:
    """Fill in expansion and alignment.
    
    Input angles should be in degrees and will be converted to radians.
    
    """
    cdef Py_ssize_t i, j
    cdef floating satz_a_rad, satz_b_rad, phi_a, phi_b, theta_a, theta_b, phi, zeta, theta, denominator, sin_beta_2, d, e
    for i in range(satz_a.shape[0]):
        for j in range(satz_a.shape[1]):
            satz_a_rad = deg2rad(satz_a[i, j])
            satz_b_rad = deg2rad(satz_b[i, j])
            phi_a = _compute_phi(satz_a_rad)
            phi_b = _compute_phi(satz_b_rad)
            theta_a = _compute_theta(satz_a_rad, phi_a)
            theta_b = _compute_theta(satz_b_rad, phi_b)
            phi = (phi_a + phi_b) / 2
            zeta = _compute_zeta(phi)
            theta = _compute_theta(zeta, phi)
            # Workaround for tiepoints symmetrical about the subsatellite-track
            denominator = theta_a * 2 if theta_a == theta_b else theta_a - theta_b

            c_expansion[i, j] = 4 * (((theta_a + theta_b) / 2 - theta) / denominator)

            sin_beta_2 = scan_width / (2 * H)
            d = ((R + H) / R * cos(phi) - cos(zeta)) * sin_beta_2
            e = cos(zeta) - sqrt(cos(zeta) ** 2 - d ** 2)

            c_alignment[i, j] = 4 * e * sin(zeta) / denominator


cdef inline floating _compute_phi(floating zeta) noexcept nogil:
    return asin(R * sin(zeta) / (R + H))


cdef inline floating _compute_theta(floating zeta, floating phi) noexcept nogil:
    return zeta - phi


cdef inline floating _compute_zeta(floating phi) noexcept nogil:
    return asin((R + H) * sin(phi) / R)


cdef inline floating[:, :] _get_upper_left_corner(floating[:, ::1] arr) noexcept nogil:
    return arr[:arr.shape[0] - 1, :arr.shape[1] - 1]


cdef inline floating[:, :] _get_upper_right_corner(floating[:, ::1] arr) noexcept nogil:
    return arr[:arr.shape[0] - 1, 1:]


cdef class MODISInterpolator:
    """Helper class for MODIS interpolation.

    Not intended for public use. Use ``modis_X_to_Y`` functions instead.

    """

    cdef int _coarse_scan_length
    cdef int _coarse_scan_width
    cdef int _coarse_pixels_per_1km
    cdef int _fine_pixels_per_coarse_pixel
    cdef int _fine_scan_width
    cdef int _fine_scan_length
    cdef int _coarse_resolution
    cdef int _fine_resolution
    cdef Py_ssize_t _factor_5km

    def __cinit__(self, unsigned int coarse_resolution, unsigned int fine_resolution, unsigned int coarse_scan_width=0):
        if coarse_resolution == 1000:
            self._coarse_scan_length = 10
            self._coarse_scan_width = 1354
        elif coarse_resolution == 5000:
            self._coarse_scan_length = 2
            if coarse_scan_width == 0:
                self._coarse_scan_width = 271
            else:
                self._coarse_scan_width = coarse_scan_width
        self._coarse_pixels_per_1km = coarse_resolution // 1000

        cdef int fine_pixels_per_1km = 1000 // fine_resolution
        self._fine_pixels_per_coarse_pixel = fine_pixels_per_1km * self._coarse_pixels_per_1km
        self._fine_scan_width = 1354 * fine_pixels_per_1km
        self._coarse_resolution = coarse_resolution
        self._fine_resolution = fine_resolution
        # partial rows/columns to repeat: 5km->1km => 2, 5km->500m => 4, 5km->250m => 8
        self._factor_5km = self._fine_pixels_per_coarse_pixel // self._coarse_pixels_per_1km * 2

    cdef tuple interpolate(
            self,
            np.ndarray[floating, ndim=2] lon1,
            np.ndarray[floating, ndim=2] lat1,
            np.ndarray[floating, ndim=2] satz1):
        """Interpolate MODIS geolocation from 'coarse_resolution' to 'fine_resolution'."""
        lon1 = np.ascontiguousarray(lon1)
        lat1 = np.ascontiguousarray(lat1)
        satz1 = np.ascontiguousarray(satz1)

        cdef int num_fine_scan_rows = self._coarse_scan_length * self._fine_pixels_per_coarse_pixel
        cdef int num_fine_scan_cols = self._fine_scan_width
        cdef np.ndarray[floating, ndim=2] a_track = np.empty((num_fine_scan_rows, num_fine_scan_cols), dtype=lon1.dtype)
        cdef np.ndarray[floating, ndim=2] a_scan = np.empty((num_fine_scan_rows, num_fine_scan_cols), dtype=lon1.dtype)
        cdef floating[:, ::1] a_track_view = a_track
        cdef floating[:, ::1] a_scan_view = a_scan

        cdef tuple coords_xy = self._get_coords()
        cdef np.ndarray[floating, ndim=1] x = coords_xy[0]
        cdef np.ndarray[floating, ndim=1] y = coords_xy[1]
        cdef floating[::1] x_view = x
        cdef floating[::1] y_view = y

        cdef np.ndarray[floating, ndim=2] tmp_tiepoint_a, tmp_tiepoint_b, tmp_tiepoint_c, tmp_tiepoint_d
        cdef floating[:, ::1] tmp_tiepoint_a_view, tmp_tiepoint_b_view, tmp_tiepoint_c_view, tmp_tiepoint_d_view
        tmp_tiepoint_a = np.empty((num_fine_scan_rows, num_fine_scan_cols), dtype=lon1.dtype)
        tmp_tiepoint_b = np.empty((num_fine_scan_rows, num_fine_scan_cols), dtype=lon1.dtype)
        tmp_tiepoint_c = np.empty((num_fine_scan_rows, num_fine_scan_cols), dtype=lon1.dtype)
        tmp_tiepoint_d = np.empty((num_fine_scan_rows, num_fine_scan_cols), dtype=lon1.dtype)
        tmp_tiepoint_a_view = tmp_tiepoint_a
        tmp_tiepoint_b_view = tmp_tiepoint_b
        tmp_tiepoint_c_view = tmp_tiepoint_c
        tmp_tiepoint_d_view = tmp_tiepoint_d

        cdef floating[:, :, ::1] xyz_coarse_view = np.empty((self._coarse_scan_length, self._coarse_scan_width, 3), dtype=lon1.dtype)
        cdef floating[:, :, ::1] xyz_fine_view = np.empty((num_fine_scan_rows, num_fine_scan_cols, 3), dtype=lon1.dtype)

        cdef unsigned int scans = satz1.shape[0] // self._coarse_scan_length
        cdef np.ndarray[floating, ndim=2] new_lons = np.empty((satz1.shape[0] * self._fine_pixels_per_coarse_pixel, self._fine_scan_width), dtype=lon1.dtype)
        cdef np.ndarray[floating, ndim=2] new_lats = np.empty((satz1.shape[0] * self._fine_pixels_per_coarse_pixel, self._fine_scan_width), dtype=lon1.dtype)
        cdef floating[:, ::1] coarse_lons = lon1
        cdef floating[:, ::1] coarse_lats = lat1
        cdef floating[:, ::1] coarse_satz = satz1
        cdef floating[:, ::1] fine_lons = new_lons
        cdef floating[:, ::1] fine_lats = new_lats
        with nogil:
            self._interpolate_lons_lats(
                scans,
                coarse_lons,
                coarse_lats,
                coarse_satz,
                x_view,
                y_view,
                a_track_view,
                a_scan_view,
                tmp_tiepoint_a_view,
                tmp_tiepoint_b_view,
                tmp_tiepoint_c_view,
                tmp_tiepoint_d_view,
                xyz_coarse_view,
                xyz_fine_view,
                fine_lons,
                fine_lats,
            )
        return new_lons, new_lats

    cdef tuple _get_coords(self):
        cdef np.ndarray[np.float32_t, ndim=1] x, y
        cdef np.float32_t[::1] x_view, y_view
        if self._coarse_scan_length == 10:
            x = np.arange(self._fine_scan_width, dtype=np.float32) % self._fine_pixels_per_coarse_pixel
            y = np.empty((self._coarse_scan_length * self._fine_pixels_per_coarse_pixel,), dtype=np.float32)
            x_view = x
            y_view = y
            with nogil:
                self._get_coords_1km(x_view, y_view)
        else:
            x, y = self._get_coords_5km()
        return x, y

    cdef void _get_coords_1km(
            self,
            floating[::1] x_view,
            floating[::1] y_view,
    ) noexcept nogil:
        cdef unsigned int scan_idx
        cdef int i
        cdef int fine_idx
        cdef int half_scan_length = self._fine_pixels_per_coarse_pixel // 2
        cdef unsigned int fine_pixels_per_scan = self._coarse_scan_length * self._fine_pixels_per_coarse_pixel
        for fine_idx in range(fine_pixels_per_scan):
            if fine_idx < half_scan_length:
                y_view[fine_idx] = -half_scan_length + 0.5 + fine_idx
            elif fine_idx >= fine_pixels_per_scan - half_scan_length:
                y_view[fine_idx] = (self._fine_pixels_per_coarse_pixel + 0.5) + (half_scan_length - (fine_pixels_per_scan - fine_idx))
            else:
                y_view[fine_idx] = ((fine_idx + half_scan_length) % self._fine_pixels_per_coarse_pixel) + 0.5

        for i in range(self._fine_pixels_per_coarse_pixel):
            x_view[(self._fine_scan_width - self._fine_pixels_per_coarse_pixel) + i] = self._fine_pixels_per_coarse_pixel + i

    @cython.wraparound(True)
    cdef tuple _get_coords_5km(self):
        cdef np.ndarray[np.float32_t, ndim=1] y = np.arange(self._fine_pixels_per_coarse_pixel * self._coarse_scan_length, dtype=np.float32) - 2
        cdef np.ndarray[np.float32_t, ndim=1] x = (np.arange(self._fine_scan_width, dtype=np.float32) - 2) % self._fine_pixels_per_coarse_pixel
        x[0] = -2
        x[1] = -1
        if self._coarse_scan_width == 271:
            x[-2] = 5
            x[-1] = 6
        else:
            # self._coarse_scan_width == 270
            x[-7] = 5
            x[-6] = 6
            x[-5] = 7
            x[-4] = 8
            x[-3] = 9
            x[-2] = 10
            x[-1] = 11
        return x, y

    cdef void _interpolate_lons_lats(self,
                                     unsigned int scans,
                                     floating[:, ::1] coarse_lons,
                                     floating[:, ::1] coarse_lats,
                                     floating[:, ::1] coarse_satz,
                                     floating[::1] x,
                                     floating[::1] y,
                                     floating[:, ::1] a_track,
                                     floating[:, ::1] a_scan,
                                     floating[:, ::1] tmp_fine_a,
                                     floating[:, ::1] tmp_fine_b,
                                     floating[:, ::1] tmp_fine_c,
                                     floating[:, ::1] tmp_fine_d,
                                     floating[:, :, ::1] coarse_xyz,
                                     floating[:, :, ::1] fine_xyz,
                                     floating[:, ::1] fine_lons,
                                     floating[:, ::1] fine_lats,
                                     ) noexcept nogil:
        cdef floating[:, ::1] lons_scan, lats_scan, satz_scan
        cdef floating[:, ::1] new_lons_scan, new_lats_scan
        cdef Py_ssize_t scan_idx
        for scan_idx in range(0, scans):
            lons_scan = coarse_lons[scan_idx * self._coarse_scan_length:(scan_idx + 1) * self._coarse_scan_length]
            lats_scan = coarse_lats[scan_idx * self._coarse_scan_length:(scan_idx + 1) * self._coarse_scan_length]
            satz_scan = coarse_satz[scan_idx * self._coarse_scan_length:(scan_idx + 1) * self._coarse_scan_length]
            new_lons_scan = fine_lons[scan_idx * self._coarse_scan_length * self._fine_pixels_per_coarse_pixel:(scan_idx + 1) * self._coarse_scan_length * self._fine_pixels_per_coarse_pixel]
            new_lats_scan = fine_lats[scan_idx * self._coarse_scan_length * self._fine_pixels_per_coarse_pixel:(scan_idx + 1) * self._coarse_scan_length * self._fine_pixels_per_coarse_pixel]
            self._get_atrack_ascan(
                satz_scan,
                x,
                y,
                tmp_fine_a[:self._coarse_scan_length - 1, :self._coarse_scan_width - 1],
                tmp_fine_b[:self._coarse_scan_length - 1, :self._coarse_scan_width - 1],
                tmp_fine_c,
                tmp_fine_d,
                a_track,
                a_scan)
            lonlat2xyz(lons_scan, lats_scan, coarse_xyz)
            self._compute_fine_xyz(a_track, a_scan, coarse_xyz,
                                   tmp_fine_a, tmp_fine_b, tmp_fine_c, tmp_fine_d,
                                   fine_xyz)

            xyz2lonlat(fine_xyz, new_lons_scan, new_lats_scan)

    cdef void _get_atrack_ascan(
            self,
            floating[:, ::1] satz,
            floating[::1] x,
            floating[::1] y,
            floating[:, ::1] c_exp_coarse,
            floating[:, ::1] c_ali_coarse,
            floating[:, ::1] c_exp_fine,
            floating[:, ::1] c_ali_fine,
            floating[:, ::1] a_track,
            floating[:, ::1] a_scan
    ) noexcept nogil:
        cdef floating[:, :] satz_a_view = _get_upper_left_corner(satz)
        cdef floating[:, :] satz_b_view = _get_upper_right_corner(satz)
        cdef Py_ssize_t scan_idx
        _compute_expansion_alignment(
            satz_a_view,
            satz_b_view,
            self._coarse_pixels_per_1km,
            c_exp_coarse,
            c_ali_coarse)

        cdef floating[:, :] c_exp_view2 = c_exp_coarse
        self._expand_tiepoint_array(c_exp_view2, c_exp_fine)

        cdef floating[:, :] c_ali_view2 = c_ali_coarse
        self._expand_tiepoint_array(c_ali_view2, c_ali_fine)

        self._calculate_atrack_ascan(
            x, y,
            c_exp_fine, c_ali_fine,
            a_track, a_scan)

    cdef void _calculate_atrack_ascan(
            self,
            floating[::1] coords_x,
            floating[::1] coords_y,
            floating[:, ::1] c_exp_full,
            floating[:, ::1] c_ali_full,
            floating[:, ::1] a_track,
            floating[:, ::1] a_scan,
    ) noexcept nogil:
        cdef Py_ssize_t i, j
        cdef floating s_s, s_t
        for j in range(coords_y.shape[0]):
            for i in range(coords_x.shape[0]):
                s_s = coords_x[i] / self._fine_pixels_per_coarse_pixel
                s_t = coords_y[j] / self._fine_pixels_per_coarse_pixel
                a_track[j, i] = s_t
                a_scan[j, i] = s_s + s_s * (1 - s_s) * c_exp_full[j, i] + s_t * (1 - s_t) * c_ali_full[j, i]

    cdef void _compute_fine_xyz(
            self,
            floating[:, ::1] a_track_view,
            floating[:, ::1] a_scan_view,
            floating[:, :, ::1] xyz_view,
            floating[:, ::1] data_tiepoint_a_view,
            floating[:, ::1] data_tiepoint_b_view,
            floating[:, ::1] data_tiepoint_c_view,
            floating[:, ::1] data_tiepoint_d_view,
            floating[:, :, ::1] xyz_comp_view,
    ) noexcept nogil:
        cdef Py_ssize_t k
        cdef floating[:, :] comp_a_view, comp_b_view, comp_c_view, comp_d_view
        for k in range(3):  # xyz
            comp_a_view = xyz_view[:xyz_view.shape[0] - 1, :xyz_view.shape[1] - 1, k]  # upper left
            comp_b_view = xyz_view[:xyz_view.shape[0] - 1, 1:, k]  # upper right
            comp_c_view = xyz_view[1:, 1:, k]  # lower right
            comp_d_view = xyz_view[1:, :xyz_view.shape[1] - 1, k]  # lower left
            self._expand_tiepoint_array(comp_a_view, data_tiepoint_a_view)
            self._expand_tiepoint_array(comp_b_view, data_tiepoint_b_view)
            self._expand_tiepoint_array(comp_c_view, data_tiepoint_c_view)
            self._expand_tiepoint_array(comp_d_view, data_tiepoint_d_view)
            self._compute_fine_xyz_component(
                a_track_view,
                a_scan_view,
                data_tiepoint_a_view,
                data_tiepoint_b_view,
                data_tiepoint_c_view,
                data_tiepoint_d_view,
                xyz_comp_view,
                k,
            )

    cdef void _compute_fine_xyz_component(
            self,
            floating[:, ::1] a_track_view,
            floating[:, ::1] a_scan_view,
            floating[:, ::1] data_a_2d_view,
            floating[:, ::1] data_b_2d_view,
            floating[:, ::1] data_c_2d_view,
            floating[:, ::1] data_d_2d_view,
            floating[:, :, ::1] xyz_comp_view,
            Py_ssize_t k,
    ) noexcept nogil:
        cdef Py_ssize_t i, j
        cdef floating scan1_tmp, scan2_tmp, atrack1, ascan1
        for i in range(a_scan_view.shape[0]):
            for j in range(a_scan_view.shape[1]):
                atrack1 = a_track_view[i, j]
                ascan1 = a_scan_view[i, j]
                scan1_tmp = (1 - ascan1) * data_a_2d_view[i, j] + ascan1 * data_b_2d_view[i, j]
                scan2_tmp = (1 - ascan1) * data_d_2d_view[i, j] + ascan1 * data_c_2d_view[i, j]
                xyz_comp_view[i, j, k] = (1 - atrack1) * scan1_tmp + atrack1 * scan2_tmp

    cdef void _expand_tiepoint_array(
            self,
            floating[:, :] input_arr,
            floating[:, ::1] output_arr,
    ) noexcept nogil:
        if self._coarse_scan_length == 10:
            self._expand_tiepoint_array_1km(input_arr, output_arr)
        else:
            self._expand_tiepoint_array_5km(input_arr, output_arr)

    cdef void _expand_tiepoint_array_1km(self, floating[:, :] input_arr, floating[:, ::1] expanded_arr) noexcept nogil:
        self._expand_tiepoint_array_1km_main(input_arr, expanded_arr)
        self._expand_tiepoint_array_1km_right_column(input_arr, expanded_arr)

    cdef void _expand_tiepoint_array_1km_main(self, floating[:, :] input_arr, floating[:, ::1] expanded_arr) noexcept nogil:
        cdef floating tiepoint_value
        cdef Py_ssize_t row_idx, col_idx, length_repeat_cycle, width_repeat_cycle, half_coarse_pixel_fine_offset, row_offset, col_offset
        cdef Py_ssize_t row_repeat_offset, col_repeat_offset
        half_coarse_pixel_fine_offset = self._fine_pixels_per_coarse_pixel // 2
        for row_idx in range(input_arr.shape[0]):
            row_offset = row_idx * self._fine_pixels_per_coarse_pixel + half_coarse_pixel_fine_offset
            for col_idx in range(input_arr.shape[1]):
                col_offset = col_idx * self._fine_pixels_per_coarse_pixel
                tiepoint_value = input_arr[row_idx, col_idx]
                self._expand_tiepoint_array_1km_with_repeat(tiepoint_value, expanded_arr, row_offset, col_offset)

        row_idx = 0
        row_offset = row_idx * self._fine_pixels_per_coarse_pixel
        for col_idx in range(input_arr.shape[1]):
            col_offset = col_idx * self._fine_pixels_per_coarse_pixel
            tiepoint_value = input_arr[row_idx, col_idx]
            self._expand_tiepoint_array_1km_with_repeat(tiepoint_value, expanded_arr, row_offset, col_offset)

        row_idx = input_arr.shape[0] - 1
        row_offset = row_idx * self._fine_pixels_per_coarse_pixel + self._fine_pixels_per_coarse_pixel
        for col_idx in range(input_arr.shape[1]):
            col_offset = col_idx * self._fine_pixels_per_coarse_pixel
            tiepoint_value = input_arr[row_idx, col_idx]
            self._expand_tiepoint_array_1km_with_repeat(tiepoint_value, expanded_arr, row_offset, col_offset)

    cdef void _expand_tiepoint_array_1km_right_column(
            self,
            floating[:, :] input_arr,
            floating[:, ::1] expanded_arr
    ) noexcept nogil:
        cdef floating tiepoint_value
        cdef Py_ssize_t row_idx, col_idx, length_repeat_cycle, width_repeat_cycle, half_coarse_pixel_fine_offset, row_offset, col_offset
        cdef Py_ssize_t row_repeat_offset, col_repeat_offset
        half_coarse_pixel_fine_offset = self._fine_pixels_per_coarse_pixel // 2
        col_idx = input_arr.shape[1] - 1
        col_offset = col_idx * self._fine_pixels_per_coarse_pixel + self._fine_pixels_per_coarse_pixel
        for row_idx in range(input_arr.shape[0]):
            row_offset = row_idx * self._fine_pixels_per_coarse_pixel + half_coarse_pixel_fine_offset
            tiepoint_value = input_arr[row_idx, col_idx]
            self._expand_tiepoint_array_1km_with_repeat(tiepoint_value, expanded_arr, row_offset, col_offset)

        self._expand_tiepoint_array_1km_right_column_top_row(input_arr, expanded_arr)
        self._expand_tiepoint_array_1km_right_column_bottom_row(input_arr, expanded_arr)

    cdef void _expand_tiepoint_array_1km_right_column_top_row(
            self,
            floating[:, :] input_arr,
            floating[:, ::1] expanded_arr
    ) noexcept nogil:
        cdef floating tiepoint_value
        cdef Py_ssize_t row_idx, col_idx, row_offset, col_offset
        row_idx = 0
        col_idx = input_arr.shape[1] - 1
        tiepoint_value = input_arr[row_idx, col_idx]
        row_offset = row_idx * self._fine_pixels_per_coarse_pixel
        col_offset = col_idx * self._fine_pixels_per_coarse_pixel + self._fine_pixels_per_coarse_pixel
        self._expand_tiepoint_array_1km_with_repeat(tiepoint_value, expanded_arr, row_offset, col_offset)

    cdef void _expand_tiepoint_array_1km_right_column_bottom_row(
            self,
            floating[:, :] input_arr,
            floating[:, ::1] expanded_arr
    ) noexcept nogil:
        cdef floating tiepoint_value
        cdef Py_ssize_t row_idx, col_idx, row_offset, col_offset
        row_idx = input_arr.shape[0] - 1
        col_idx = input_arr.shape[1] - 1
        tiepoint_value = input_arr[row_idx, col_idx]
        row_offset = row_idx * self._fine_pixels_per_coarse_pixel + self._fine_pixels_per_coarse_pixel
        col_offset = col_idx * self._fine_pixels_per_coarse_pixel + self._fine_pixels_per_coarse_pixel
        self._expand_tiepoint_array_1km_with_repeat(tiepoint_value, expanded_arr, row_offset, col_offset)

    cdef void _expand_tiepoint_array_1km_with_repeat(
            self,
            floating tiepoint_value,
            floating[:, ::1] expanded_arr,
            Py_ssize_t row_offset,
            Py_ssize_t col_offset,
    ) noexcept nogil:
        cdef Py_ssize_t length_repeat_cycle, width_repeat_cycle
        cdef Py_ssize_t row_repeat_offset, col_repeat_offset
        for length_repeat_cycle in range(self._fine_pixels_per_coarse_pixel):
            row_repeat_offset = row_offset + length_repeat_cycle
            for width_repeat_cycle in range(self._fine_pixels_per_coarse_pixel):
                col_repeat_offset = col_offset + width_repeat_cycle
                expanded_arr[row_repeat_offset, col_repeat_offset] = tiepoint_value

    cdef void _expand_tiepoint_array_5km(self, floating[:, :] input_arr, floating[:, ::1] expanded_arr) noexcept nogil:
        self._expand_tiepoint_array_5km_main(input_arr, expanded_arr)
        self._expand_tiepoint_array_5km_left(input_arr, expanded_arr)
        if self._coarse_scan_width == 270:
            self._expand_tiepoint_array_5km_270_extra_column(input_arr, expanded_arr)
        self._expand_tiepoint_array_5km_right(input_arr, expanded_arr)

    cdef void _expand_tiepoint_array_5km_main(self, floating[:, :] input_arr, floating[:, ::1] expanded_arr) noexcept nogil:
        cdef floating tiepoint_value
        cdef Py_ssize_t row_idx, col_idx, row_offset, col_offset
        for row_idx in range(input_arr.shape[0]):
            row_offset = row_idx * self._fine_pixels_per_coarse_pixel * 2
            for col_idx in range(input_arr.shape[1]):
                col_offset = col_idx * self._fine_pixels_per_coarse_pixel + self._factor_5km
                tiepoint_value = input_arr[row_idx, col_idx]
                self._expand_tiepoint_array_5km_with_repeat(
                    tiepoint_value,
                    expanded_arr,
                    row_offset,
                    col_offset,
                    self._fine_pixels_per_coarse_pixel)

    cdef void _expand_tiepoint_array_5km_270_extra_column(
            self,
            floating[:, :] input_arr,
            floating[:, ::1] expanded_arr
    ) noexcept nogil:
        """Copy an extra coarse pixel column between the main copied area and the right-most columns."""
        cdef floating tiepoint_value
        cdef Py_ssize_t row_idx, col_idx, row_offset, col_offset
        col_idx = input_arr.shape[1] - 1
        col_offset = col_idx * self._fine_pixels_per_coarse_pixel + self._fine_pixels_per_coarse_pixel + self._factor_5km
        for row_idx in range(input_arr.shape[0]):
            row_offset = row_idx * self._fine_pixels_per_coarse_pixel * 2
            tiepoint_value = input_arr[row_idx, col_idx]
            self._expand_tiepoint_array_5km_with_repeat(
                tiepoint_value,
                expanded_arr,
                row_offset,
                col_offset,
                self._fine_pixels_per_coarse_pixel)

    cdef void _expand_tiepoint_array_5km_left(
            self,
            floating[:, :] input_arr,
            floating[:, ::1] expanded_arr,
    ) noexcept nogil:
        self._expand_tiepoint_array_5km_edges(input_arr, expanded_arr, 0, 0)

    cdef void _expand_tiepoint_array_5km_right(
            self,
            floating[:, :] input_arr,
            floating[:, ::1] expanded_arr,
    ) noexcept nogil:
        self._expand_tiepoint_array_5km_edges(
            input_arr,
            expanded_arr,
            input_arr.shape[1] - 1,
            expanded_arr.shape[1] - self._factor_5km)

    cdef void _expand_tiepoint_array_5km_edges(
            self,
            floating[:, :] input_arr,
            floating[:, ::1] expanded_arr,
            Py_ssize_t course_col_idx,
            Py_ssize_t fine_col_idx,
    ) noexcept nogil:
        cdef floating tiepoint_value
        cdef Py_ssize_t row_idx, row_offset
        for row_idx in range(input_arr.shape[0]):
            row_offset = row_idx * self._fine_pixels_per_coarse_pixel * 2
            tiepoint_value = input_arr[row_idx, course_col_idx]
            self._expand_tiepoint_array_5km_with_repeat(
                tiepoint_value,
                expanded_arr,
                row_offset,
                fine_col_idx,
                self._factor_5km)

    cdef void _expand_tiepoint_array_5km_with_repeat(
            self,
            floating tiepoint_value,
            floating[:, ::1] expanded_arr,
            Py_ssize_t row_offset,
            Py_ssize_t col_offset,
            Py_ssize_t col_width,
    ) noexcept nogil:
        cdef Py_ssize_t length_repeat_cycle, width_repeat_cycle
        for length_repeat_cycle in range(self._fine_pixels_per_coarse_pixel * 2):
            for width_repeat_cycle in range(col_width):
                expanded_arr[row_offset + length_repeat_cycle,
                             col_offset + width_repeat_cycle] = tiepoint_value
