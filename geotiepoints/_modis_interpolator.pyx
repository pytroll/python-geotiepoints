from libc.math cimport fmin, fmax, floor
cimport cython
from cython.parallel import prange,parallel
from .simple_modis_interpolator import scanline_mapblocks

cimport numpy as np
from libc.math cimport asin, sin, cos, sqrt
import numpy as np

# ctypedef fused floating:
#     float
#     double

ctypedef fused floating:
    np.float32_t
    np.float64_t

EARTH_RADIUS = 6370997.0
# cdef np.float32_t R = 6371.0
# # Aqua scan width and altitude in km
# cdef np.float32_t scan_width = 10.00017
# cdef np.float32_t H = 705.0
DEF R = 6371.0
# Aqua scan width and altitude in km
DEF scan_width = 10.00017
DEF H = 705.0


def lonlat2xyz(
        lons, lats,
        # floating[:, :] lons, floating[:, :] lats,
        radius=EARTH_RADIUS):
    """Convert lons and lats to cartesian coordinates."""
    lons_rad = np.deg2rad(lons)
    lats_rad = np.deg2rad(lats)
    x_coords = radius * np.cos(lats_rad) * np.cos(lons_rad)
    y_coords = radius * np.cos(lats_rad) * np.sin(lons_rad)
    z_coords = radius * np.sin(lats_rad)
    return x_coords, y_coords, z_coords


def xyz2lonlat(
        # floating[:, :] x__,
        # floating[:, :] y__,
        # floating[:, :] z__,
        # np.ndarray[floating, ndim=2] x__,
        # np.ndarray[floating, ndim=2] y__,
        # np.ndarray[floating, ndim=2] z__,
        x__,
        y__,
        z__,
        radius=EARTH_RADIUS,
        thr=0.8,
        bint low_lat_z=True):
    """Get longitudes from cartesian coordinates."""
    lons = np.rad2deg(np.arccos(x__ / np.sqrt(x__ ** 2 + y__ ** 2))) * np.sign(y__)
    lats = np.sign(z__) * (90 - np.rad2deg(np.arcsin(np.sqrt(x__ ** 2 + y__ ** 2) / radius)))
    if low_lat_z:
        # if we are at low latitudes - small z, then get the
        # latitudes only from z. If we are at high latitudes (close to the poles)
        # then derive the latitude using x and y:
        lat_mask_cond = np.logical_and(
            np.less(z__, thr * radius),
            np.greater(z__, -1. * thr * radius))
        lat_z_only = 90 - np.rad2deg(np.arccos(z__ / radius))
        lats = np.where(lat_mask_cond, lat_z_only, lats)

    return lons, lats


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
    interp = Interpolator(coarse_resolution, fine_resolution, coarse_scan_width=coarse_scan_width or 0)
    return interp.interpolate(lon1, lat1, satz1)


cdef inline floating _compute_phi(floating zeta) nogil:
    return asin(R * sin(zeta) / (R + H))


cdef inline floating _compute_theta(floating zeta, floating phi) nogil:
    return zeta - phi


cdef inline floating _compute_zeta(floating phi) nogil:
    return asin((R + H) * sin(phi) / R)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void _compute_expansion_alignment(floating[:, :, ::1] satz_a, floating [:, :, ::1] satz_b,
                                       floating[:, :, ::1] c_expansion, floating[:, :, ::1] c_alignment) nogil:
    """All angles in radians."""
    cdef Py_ssize_t i, j, k
    cdef floating phi_a, phi_b, theta_a, theta_b, phi, zeta, theta, denominator, sin_beta_2, d, e
    for i in range(satz_a.shape[0]):
        for j in range(satz_a.shape[1]):
            for k in range(satz_a.shape[2]):
                phi_a = _compute_phi(satz_a[i, j, k])
                phi_b = _compute_phi(satz_b[i, j, k])
                theta_a = _compute_theta(satz_a[i, j, k], phi_a)
                theta_b = _compute_theta(satz_b[i, j, k], phi_b)
                phi = (phi_a + phi_b) / 2
                zeta = _compute_zeta(phi)
                theta = _compute_theta(zeta, phi)
                # Workaround for tiepoints symmetrical about the subsatellite-track
                denominator = theta_a * 2 if theta_a == theta_b else theta_a - theta_b

                c_expansion[i, j, k] = 4 * (((theta_a + theta_b) / 2 - theta) / denominator)

                sin_beta_2 = scan_width / (2 * H)
                d = ((R + H) / R * cos(phi) - cos(zeta)) * sin_beta_2
                e = cos(zeta) - sqrt(cos(zeta) ** 2 - d ** 2)

                c_alignment[i, j, k] = 4 * e * sin(zeta) / denominator


cdef tuple _get_corners(np.ndarray[floating, ndim=3] arr):
    arr_a = arr[:, :-1, :-1]
    arr_b = arr[:, :-1, 1:]
    arr_c = arr[:, 1:, 1:]
    arr_d = arr[:, 1:, :-1]
    return arr_a, arr_b, arr_c, arr_d


cdef class Interpolator:
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

    def __cinit__(self, unsigned int coarse_resolution, unsigned int fine_resolution, unsigned int coarse_scan_width=0):
        if coarse_resolution == 1000:
            coarse_scan_length = 10
            coarse_scan_width = 1354
        elif coarse_resolution == 5000:
            coarse_scan_length = 2
            if coarse_scan_width == 0:
                coarse_scan_width = 271
            else:
                coarse_scan_width = coarse_scan_width
        self._coarse_scan_length = coarse_scan_length
        self._coarse_scan_width = coarse_scan_width
        self._coarse_pixels_per_1km = coarse_resolution // 1000

        fine_pixels_per_1km = {
            250: 4,
            500: 2,
            1000: 1,
        }[fine_resolution]
        self._fine_pixels_per_coarse_pixel = fine_pixels_per_1km * self._coarse_pixels_per_1km
        self._fine_scan_width = 1354 * fine_pixels_per_1km
        # TODO: This is actually the same as fine_pixels_per_coarse_pixel I think?
        self._fine_scan_length = fine_pixels_per_1km * 10 // coarse_scan_length
        self._coarse_resolution = coarse_resolution
        self._fine_resolution = fine_resolution

    cdef tuple _get_coords(self, unsigned int scans):
        if self._coarse_scan_length == 10:
            return self._get_coords_1km(scans)
        return self._get_coords_5km(scans)

    cdef np.ndarray[floating, ndim=2] _expand_tiepoint_array(self, np.ndarray[floating, ndim=3] arr):
        if self._coarse_scan_length == 10:
            return self._expand_tiepoint_array_1km(arr)
        return self._expand_tiepoint_array_5km(arr)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef interpolate(
            self,
            np.ndarray[floating, ndim=2] lon1,
            np.ndarray[floating, ndim=2] lat1,
            np.ndarray[floating, ndim=2] satz1):
        """Interpolate MODIS geolocation from 'coarse_resolution' to 'fine_resolution'."""
        lon1 = np.ascontiguousarray(lon1)
        la11 = np.ascontiguousarray(lat1)
        satz1 = np.ascontiguousarray(satz1)

        cdef unsigned int scans = satz1.shape[0] // self._coarse_scan_length
        # reshape to (num scans, rows per scan, columns per scan)
        cdef np.ndarray[floating, ndim=3] satz1_3d = satz1.reshape((-1, self._coarse_scan_length, self._coarse_scan_width))
        print("Satz1 dtype: ", satz1.dtype, lon1.dtype, lat1.dtype, satz1_3d.dtype)

        corners = _get_corners(satz1_3d)
        cdef np.ndarray[floating, ndim=3] satz_a = np.deg2rad(np.ascontiguousarray(corners[0]))
        cdef np.ndarray[floating, ndim=3] satz_b = np.deg2rad(np.ascontiguousarray(corners[1]))
        cdef floating[:, :, ::1] satz_a_view = satz_a
        cdef floating[:, :, ::1] satz_b_view = satz_b
        cdef np.ndarray[floating, ndim=3] c_exp = np.empty((satz_a.shape[0], satz_a.shape[1], satz_b.shape[2]), dtype=satz_a.dtype)
        cdef np.ndarray[floating, ndim=3] c_ali = np.empty((satz_a.shape[0], satz_a.shape[1], satz_b.shape[2]), dtype=satz_a.dtype)
        cdef floating[:, :, ::1] c_exp_view = c_exp
        cdef floating[:, :, ::1] c_ali_view = c_ali
        _compute_expansion_alignment(satz_a_view, satz_b_view, c_exp_view, c_ali_view)

        coords_xy = self._get_coords(scans)
        cdef np.ndarray[floating, ndim=1] x = coords_xy[0]
        cdef np.ndarray[floating, ndim=1] y = coords_xy[1]
        i_rs, i_rt = np.meshgrid(x, y)
        print("Coords: ", x.dtype, y.dtype, i_rs.dtype, i_rt.dtype)

        p_os = 0
        p_ot = 0
        cdef np.ndarray[floating, ndim=2] s_s = (p_os + i_rs) * 1.0 / self._fine_pixels_per_coarse_pixel
        cdef np.ndarray[floating, ndim=2] s_t = (p_ot + i_rt) * 1.0 / self._fine_scan_length

        cdef np.ndarray[floating, ndim=2] c_exp_full = self._expand_tiepoint_array(c_exp)
        cdef np.ndarray[floating, ndim=2] c_ali_full = self._expand_tiepoint_array(c_ali)

        cdef np.ndarray[floating, ndim=2] a_track = s_t
        cdef np.ndarray[floating, ndim=2] a_scan = s_s + s_s * (1 - s_s) * c_exp_full + s_t * (1 - s_t) * c_ali_full

        res = []
        datasets = lonlat2xyz(lon1, lat1)
        cdef np.ndarray[floating, ndim=3] data
        cdef floating[:, :, :] data_view
        cdef np.ndarray[floating, ndim=3] data_a, data_b, data_c, data_d
        cdef np.ndarray[floating, ndim=2] data_a_2d, data_b_2d, data_c_2d, data_d_2d
        # cdef np.ndarray[np.float64_t, ndim=2] comp_arr_2d
        cdef np.ndarray[floating, ndim=2] comp_arr_2d
        for data_2d in datasets:
            data = data_2d.reshape((-1, self._coarse_scan_length, self._coarse_scan_width))
            corners = _get_corners(data)
            data_a = corners[0]
            data_b = corners[1]
            data_c = corners[2]
            data_d = corners[3]
            data_a_2d = self._expand_tiepoint_array(data_a)
            data_b_2d = self._expand_tiepoint_array(data_b)
            data_c_2d = self._expand_tiepoint_array(data_c)
            data_d_2d = self._expand_tiepoint_array(data_d)

            data_1 = (1 - a_scan) * data_a_2d + a_scan * data_b_2d
            data_2 = (1 - a_scan) * data_d_2d + a_scan * data_c_2d
            comp_arr_2d = (1 - a_track) * data_1 + a_track * data_2

            res.append(comp_arr_2d.astype(np.float64))
        new_lons, new_lats = xyz2lonlat(*res)
        return new_lons.astype(lon1.dtype), new_lats.astype(lat1.dtype)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef tuple _get_coords_1km(self, unsigned int scans):
        cdef int half_scan_length = self._fine_scan_length // 2
        cdef np.ndarray[np.float32_t, ndim=1] y = np.empty((scans * self._coarse_scan_length * self._fine_scan_length,), dtype=np.float32)
        cdef np.float32_t[::1] y_view = y
        cdef unsigned int scan_idx
        cdef int i
        cdef int fine_idx
        cdef unsigned int fine_pixels_per_scan = self._coarse_scan_length * self._fine_scan_length
        for scan_idx in range(scans):
            for i in range(fine_pixels_per_scan):
                fine_idx = scan_idx * fine_pixels_per_scan + i
                if i < half_scan_length:
                    y_view[fine_idx] = (-half_scan_length + 0.5) - i
                elif i > fine_pixels_per_scan - half_scan_length:
                    y_view[fine_idx] = (self._fine_scan_length + 0.5) + (fine_pixels_per_scan - i)
                else:
                    y_view[fine_idx] = (i % self._fine_scan_length) + 0.5

        cdef np.ndarray[np.float32_t, ndim=1] x = np.arange(self._fine_scan_width, dtype=np.float32) % self._fine_pixels_per_coarse_pixel
        cdef np.float32_t[::1] x_view = x
        for i in range(self._fine_pixels_per_coarse_pixel):
            x_view[(self._fine_scan_width - self._fine_pixels_per_coarse_pixel) + i] = self._fine_pixels_per_coarse_pixel + i
        return x, y

    cdef tuple _get_coords_5km(self, unsigned int scans):
        cdef np.ndarray[np.float32_t, ndim=1] y = np.arange(self._fine_scan_length * self._coarse_scan_length, dtype=np.float32) - 2
        y = np.tile(y, scans)

        cdef np.ndarray[np.float32_t, ndim=1] x = (np.arange(self._fine_scan_width, dtype=np.float32) - 2) % self._fine_pixels_per_coarse_pixel
        x[0] = -2
        x[1] = -1
        if self._coarse_scan_width == 271:
            x[-2] = 5
            x[-1] = 6
        else:
            # elif self._coarse_scan_width == 270:
            x[-7] = 5
            x[-6] = 6
            x[-5] = 7
            x[-4] = 8
            x[-3] = 9
            x[-2] = 10
            x[-1] = 11
        return x, y

    cdef np.ndarray[floating, ndim=2] _expand_tiepoint_array_1km_orig(self, np.ndarray[floating, ndim=3] arr):
        arr = np.repeat(arr, self._fine_scan_length, axis=1)
        arr = np.concatenate(
            (arr[:, :self._fine_scan_length // 2, :], arr, arr[:, -(self._fine_scan_length // 2):, :]), axis=1
        )
        cdef np.ndarray[floating, ndim=2] arr_2d = np.repeat(arr.reshape((-1, self._coarse_scan_width - 1)), self._fine_pixels_per_coarse_pixel, axis=1)
        return np.hstack((arr_2d, arr_2d[:, -self._fine_pixels_per_coarse_pixel:]))


    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef np.ndarray[floating, ndim=2] _expand_tiepoint_array_1km(self, np.ndarray[floating, ndim=3] npy_arr):
        # TODO: Replace shape multiplication with self._fine_pixel_length and self._fine_pixel_width
        cdef floating[:, :, :] arr = npy_arr
        cdef np.ndarray[floating, ndim=2] arr_2d = np.empty(
            (
                arr.shape[0] * (arr.shape[1] + 1) * self._fine_pixels_per_coarse_pixel,
                (arr.shape[2] + 1) * self._fine_pixels_per_coarse_pixel
            ),
            dtype=npy_arr.dtype)
        cdef floating[:, ::1] arr_2d_view = arr_2d
        cdef floating tiepoint_value
        cdef Py_ssize_t scan_idx, row_idx, col_idx, length_repeat_cycle, width_repeat_cycle, half_scan_offset, scan_offset, row_offset, col_offset
        # FIXME: This is not an actual scan length
        half_scan_offset = self._fine_scan_length // 2
        for scan_idx in range(arr.shape[0]):
            scan_offset = scan_idx * self._fine_scan_length * self._coarse_scan_length
            for row_idx in range(arr.shape[1]):
                row_offset = row_idx * self._fine_pixels_per_coarse_pixel
                for col_idx in range(arr.shape[2]):
                    col_offset = col_idx * self._fine_pixels_per_coarse_pixel
                    tiepoint_value = arr[scan_idx, row_idx, col_idx]
                    for length_repeat_cycle in range(self._fine_pixels_per_coarse_pixel):
                        for width_repeat_cycle in range(self._fine_pixels_per_coarse_pixel):
                            # main "center" scan portion
                            arr_2d_view[scan_offset + row_offset + length_repeat_cycle + half_scan_offset,
                                        col_offset + width_repeat_cycle] = tiepoint_value
                            if row_offset < half_scan_offset:
                                # copy of top half of the scan
                                arr_2d_view[scan_offset + row_offset + length_repeat_cycle,
                                            col_offset + width_repeat_cycle] = tiepoint_value
                            elif row_offset >= (((arr.shape[1] - 1) * self._fine_pixels_per_coarse_pixel) - (self._fine_scan_length // 2)):
                                # copy of bottom half of the scan
                                # TODO: Clean this up
                                arr_2d_view[scan_offset + row_offset + length_repeat_cycle + half_scan_offset + self._fine_scan_length // 2,
                                            col_offset + width_repeat_cycle] = tiepoint_value
                            if col_idx == arr.shape[2] - 1:
                                # there is one less coarse column than needed by the fine resolution
                                # copy last coarse column as the last fine coarse column
                                # this last coarse column will be both the second to last and the last
                                # fine resolution columns
                                arr_2d_view[scan_offset + row_offset + length_repeat_cycle + half_scan_offset,
                                            col_offset + self._fine_pixels_per_coarse_pixel + width_repeat_cycle] = tiepoint_value
                                # also need the top and bottom half copies
                                if row_offset < half_scan_offset:
                                    # copy of top half of the scan
                                    arr_2d_view[scan_offset + row_offset + length_repeat_cycle,
                                                col_offset + self._fine_pixels_per_coarse_pixel + width_repeat_cycle] = tiepoint_value
                                elif row_offset >= (((arr.shape[1] - 1) * self._fine_pixels_per_coarse_pixel) - (self._fine_scan_length // 2)):
                                    # TODO: Clean this up
                                    # copy of bottom half of the scan
                                    arr_2d_view[scan_offset + row_offset + length_repeat_cycle + half_scan_offset + self._fine_scan_length // 2,
                                                col_offset + self._fine_pixels_per_coarse_pixel + width_repeat_cycle] = tiepoint_value
        return arr_2d

    cdef np.ndarray[floating, ndim=2] _expand_tiepoint_array_5km(self, np.ndarray[floating, ndim=3] arr):
        arr = np.repeat(arr, self._fine_scan_length * 2, axis=1)
        cdef np.ndarray[floating, ndim=2] arr_2d = np.repeat(arr.reshape((-1, self._coarse_scan_width - 1)), self._fine_pixels_per_coarse_pixel, axis=1)
        factor = self._fine_pixels_per_coarse_pixel // self._coarse_pixels_per_1km
        if self._coarse_scan_width == 271:
            return np.hstack((arr_2d[:, :2 * factor], arr_2d, arr_2d[:, -2 * factor:]))
        else:
            return np.hstack(
                (
                    arr_2d[:, :2 * factor],
                    arr_2d,
                    arr_2d[:, -self._fine_pixels_per_coarse_pixel:],
                    arr_2d[:, -2 * factor:],
                )
            )
