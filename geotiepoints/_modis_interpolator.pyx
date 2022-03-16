from libc.math cimport fmin, fmax, floor
cimport cython
from .simple_modis_interpolator import scanline_mapblocks

cimport numpy as np
from libc.math cimport asin, sin, cos, sqrt, acos, M_PI
import numpy as np

ctypedef fused floating:
    np.float32_t
    np.float64_t

DEF EARTH_RADIUS = 6370997.0
# cdef np.float32_t R = 6371.0
# # Aqua scan width and altitude in km
# cdef np.float32_t scan_width = 10.00017
# cdef np.float32_t H = 705.0
DEF R = 6371.0
# Aqua scan width and altitude in km
DEF scan_width = 10.00017
DEF H = 705.0


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void lonlat2xyz(
        floating[:, :, ::1] lons,
        floating[:, :, ::1] lats,
        floating[:, :, :, ::1] xyz,
) nogil:
    """Convert lons and lats to cartesian coordinates."""
    cdef Py_ssize_t i, j, k
    cdef floating lon_rad, lat_rad
    for i in range(lons.shape[0]):
        for j in range(lons.shape[1]):
            for k in range(lons.shape[2]):
                lon_rad = _deg2rad64(lons[i, j, k])
                lat_rad = _deg2rad64(lats[i, j, k])
                xyz[i, j, k, 0] = EARTH_RADIUS * cos(lat_rad) * cos(lon_rad)
                xyz[i, j, k, 1] = EARTH_RADIUS * cos(lat_rad) * sin(lon_rad)
                xyz[i, j, k, 2] = EARTH_RADIUS * sin(lat_rad)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void xyz2lonlat(
        floating[:, :, ::1] xyz,
        floating[:, ::1] lons,
        floating[:, ::1] lats,
        floating thr=0.8,
        bint low_lat_z=True) nogil:
    """Get longitudes from cartesian coordinates."""
    cdef Py_ssize_t i, j
    cdef np.float64_t x, y, z
    for i in range(xyz.shape[0]):
        for j in range(xyz.shape[1]):
            # 64-bit precision matters apparently
            x = <np.float64_t>xyz[i, j, 0]
            y = <np.float64_t>xyz[i, j, 1]
            z = <np.float64_t>xyz[i, j, 2]
            lons[i, j] = _rad2deg64(acos(x / sqrt(x ** 2 + y ** 2))) * _sign(y)
            # if we are at low latitudes - small z, then get the
            # latitudes only from z. If we are at high latitudes (close to the poles)
            # then derive the latitude using x and y:
            if low_lat_z and (z < thr * EARTH_RADIUS) and (z > -1.0 * thr * EARTH_RADIUS):
                lats[i, j] = 90 - _rad2deg64(acos(z / EARTH_RADIUS))
            else:
                lats[i, j] = _sign(z) * (90 - _rad2deg64(asin(sqrt(x ** 2 + y ** 2) / EARTH_RADIUS)))


cdef inline int _sign(floating x) nogil:
    return 1 if x > 0 else (-1 if x < 0 else 0)


cdef inline floating _rad2deg(floating x) nogil:
    return x * (180.0 / M_PI)


cdef inline floating _deg2rad(floating x) nogil:
    return x * (M_PI / 180.0)


cdef inline np.float64_t _rad2deg64(np.float64_t x) nogil:
    return x * (180.0 / M_PI)


cdef inline np.float64_t _deg2rad64(np.float64_t x) nogil:
    return x * (M_PI / 180.0)


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
    """Fill in expansion and alignment.
    
    Input angles should be in degrees and will be converted to radians.
    
    """
    cdef Py_ssize_t i, j, k
    cdef floating satz_a_rad, satz_b_rad, phi_a, phi_b, theta_a, theta_b, phi, zeta, theta, denominator, sin_beta_2, d, e
    for i in range(satz_a.shape[0]):
        for j in range(satz_a.shape[1]):
            for k in range(satz_a.shape[2]):
                satz_a_rad = _deg2rad(satz_a[i, j, k])
                satz_b_rad = _deg2rad(satz_b[i, j, k])
                phi_a = _compute_phi(satz_a_rad)
                phi_b = _compute_phi(satz_b_rad)
                theta_a = _compute_theta(satz_a_rad, phi_a)
                theta_b = _compute_theta(satz_b_rad, phi_b)
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


cdef floating[:, :, ::1] _get_upper_left_corner(floating[:, :, ::1] arr):
    cdef floating[:, :, ::1] ret = arr[:, :-1, :-1].copy()
    return ret


cdef floating[:, :, ::1] _get_upper_right_corner(floating[:, :, ::1] arr):
    cdef floating[:, :, ::1] ret = arr[:, :-1, 1:].copy()
    return ret


cdef floating[:, :, ::1] _get_lower_right_corner(floating[:, :, ::1] arr):
    cdef floating[:, :, ::1] ret = arr[:, 1:, 1:].copy()
    return ret


cdef floating[:, :, ::1] _get_lower_left_corner(floating[:, :, ::1] arr):
    cdef floating[:, :, ::1] ret = arr[:, 1:, :-1].copy()
    return ret


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
        cdef np.ndarray[floating, ndim=2] expanded_arr
        cdef floating[:, ::1] expanded_arr_view
        # contiguous in the sensor zenith case, every 3rd in the lon/lat/xyz case
        cdef floating[:, :, :] input_arr_view = arr
        if self._coarse_scan_length == 10:
            expanded_arr = np.empty(
                (
                    arr.shape[0] * (arr.shape[1] + 1) * self._fine_pixels_per_coarse_pixel,
                    (arr.shape[2] + 1) * self._fine_pixels_per_coarse_pixel
                ),
                dtype=arr.dtype)
            expanded_arr_view = expanded_arr
            self._expand_tiepoint_array_1km(input_arr_view, expanded_arr_view)
            return expanded_arr
        return self._expand_tiepoint_array_5km(arr)

    cdef interpolate(
            self,
            np.ndarray[floating, ndim=2] lon1,
            np.ndarray[floating, ndim=2] lat1,
            np.ndarray[floating, ndim=2] satz1):
        """Interpolate MODIS geolocation from 'coarse_resolution' to 'fine_resolution'."""
        lon1 = np.ascontiguousarray(lon1)
        la11 = np.ascontiguousarray(lat1)
        satz1 = np.ascontiguousarray(satz1)

        cdef np.ndarray[floating, ndim=2] a_track, a_scan, new_lons, new_lats
        a_track, a_scan = self._get_atrack_ascan(satz1)
        new_lons, new_lats = self._interpolate_lons_lats(lon1, lat1, a_track, a_scan)
        return new_lons, new_lats

    cdef tuple _get_atrack_ascan(self, np.ndarray[floating, ndim=2] satz1):
        cdef unsigned int scans = satz1.shape[0] // self._coarse_scan_length
        # reshape to (num scans, rows per scan, columns per scan)
        cdef floating[:, :, ::1] satz1_3d = satz1.reshape((-1, self._coarse_scan_length, self._coarse_scan_width))

        cdef floating[:, :, ::1] satz_a_view = _get_upper_left_corner(satz1_3d)
        cdef floating[:, :, ::1] satz_b_view = _get_upper_right_corner(satz1_3d)
        cdef np.ndarray[floating, ndim=3] c_exp = np.empty(
            (satz_a_view.shape[0], satz_a_view.shape[1], satz_a_view.shape[2]),
            dtype=satz1.dtype)
        cdef np.ndarray[floating, ndim=3] c_ali = np.empty(
            (satz_a_view.shape[0], satz_a_view.shape[1], satz_a_view.shape[2]),
            dtype=satz1.dtype)
        cdef floating[:, :, ::1] c_exp_view = c_exp
        cdef floating[:, :, ::1] c_ali_view = c_ali
        _compute_expansion_alignment(satz_a_view, satz_b_view, c_exp_view, c_ali_view)

        cdef np.ndarray[floating, ndim=2] c_exp_full = self._expand_tiepoint_array(c_exp)
        cdef np.ndarray[floating, ndim=2] c_ali_full = self._expand_tiepoint_array(c_ali)

        coords_xy = self._get_coords(scans)
        cdef np.ndarray[floating, ndim=1] x = coords_xy[0]
        cdef np.ndarray[floating, ndim=1] y = coords_xy[1]
        i_rs, i_rt = np.meshgrid(x, y)
        cdef np.ndarray[floating, ndim=2] s_s = i_rs / self._fine_pixels_per_coarse_pixel
        cdef np.ndarray[floating, ndim=2] s_t = i_rt / self._fine_scan_length

        cdef np.ndarray[floating, ndim=2] a_track = s_t
        cdef np.ndarray[floating, ndim=2] a_scan = s_s + s_s * (1 - s_s) * c_exp_full + s_t * (1 - s_t) * c_ali_full
        return a_track, a_scan

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef tuple _interpolate_lons_lats(self,
                                      np.ndarray[floating, ndim=2] lon1,
                                      np.ndarray[floating, ndim=2] lat1,
                                      np.ndarray[floating, ndim=2] a_track,
                                      np.ndarray[floating, ndim=2] a_scan):
        res = []
        cdef np.ndarray[floating, ndim=3] lon1_3d, lat1_3d
        lon1_3d = lon1.reshape((-1, self._coarse_scan_length, self._coarse_scan_width))
        lat1_3d = lat1.reshape((-1, self._coarse_scan_length, self._coarse_scan_width))
        cdef floating[:, :, ::1] lon1_3d_view, lat1_3d_view
        lon1_3d_view = lon1_3d
        lat1_3d_view = lat1_3d
        # cdef np.ndarray[floating, ndim=3] lon1_a, lon1_b, lon1_c, lon1_d, lat1_a, lat1_b, lat1_c, lat1_d
        cdef floating[:, :, ::1] lon1_a, lon1_b, lon1_c, lon1_d, lat1_a, lat1_b, lat1_c, lat1_d
        lon1_a = _get_upper_left_corner(lon1_3d_view)
        lon1_b = _get_upper_right_corner(lon1_3d_view)
        lon1_c = _get_lower_right_corner(lon1_3d_view)
        lon1_d = _get_lower_left_corner(lon1_3d_view)
        lat1_a = _get_upper_left_corner(lat1_3d_view)
        lat1_b = _get_upper_right_corner(lat1_3d_view)
        lat1_c = _get_lower_right_corner(lat1_3d_view)
        lat1_d = _get_lower_left_corner(lat1_3d_view)
        cdef np.ndarray[floating, ndim=4] xyz_a = np.empty((lon1_a.shape[0], lon1_a.shape[1], lon1_a.shape[2], 3), dtype=lon1.dtype)
        cdef np.ndarray[floating, ndim=4] xyz_b = np.empty((lon1_b.shape[0], lon1_b.shape[1], lon1_b.shape[2], 3), dtype=lon1.dtype)
        cdef np.ndarray[floating, ndim=4] xyz_c = np.empty((lon1_c.shape[0], lon1_c.shape[1], lon1_c.shape[2], 3), dtype=lon1.dtype)
        cdef np.ndarray[floating, ndim=4] xyz_d = np.empty((lon1_d.shape[0], lon1_d.shape[1], lon1_d.shape[2], 3), dtype=lon1.dtype)
        cdef floating[:, :, :, ::1] xyz_a_view, xyz_b_view, xyz_c_view, xyz_d_view
        xyz_a_view = xyz_a
        xyz_b_view = xyz_b
        xyz_c_view = xyz_c
        xyz_d_view = xyz_d
        lonlat2xyz(lon1_a, lat1_a, xyz_a_view)
        lonlat2xyz(lon1_b, lat1_b, xyz_b_view)
        lonlat2xyz(lon1_c, lat1_c, xyz_c_view)
        lonlat2xyz(lon1_d, lat1_d, xyz_d_view)

        cdef np.ndarray[floating, ndim=3] data
        # cdef np.ndarray[floating, ndim=3] data_a, data_b, data_c, data_d
        cdef np.ndarray[floating, ndim=2] data_a_2d, data_b_2d, data_c_2d, data_d_2d
        cdef np.ndarray[floating, ndim=3] comp_arr_2d = np.empty((a_scan.shape[0], a_scan.shape[1], 3), dtype=lon1.dtype)
        cdef floating[:, :, ::1] xyz_comp_view = comp_arr_2d
        # cdef np.ndarray[floating, ndim=3] xyz_comp_a, xyz_comp_b, xyz_comp_c, xyz_comp_d
        cdef Py_ssize_t i, j, k
        cdef floating scan1_tmp, scan2_tmp, atrack1, ascan1
        cdef floating[:, ::1] data_a_2d_view, data_b_2d_view, data_c_2d_view, data_d_2d_view
        cdef np.ndarray[floating, ndim=3] comp_a, comp_b, comp_c, comp_d
        print(a_scan.shape[0], a_scan.shape[1], a_track.shape[0], a_track.shape[1], comp_arr_2d.shape[0], comp_arr_2d.shape[1], comp_arr_2d.shape[2])
        print(xyz_a.shape[0], xyz_a.shape[1], xyz_a.shape[2], xyz_a.shape[3])
        for k in range(3):  # xyz
            # data_a_2d = self._expand_tiepoint_array(xyz_a[:, :, :, k])
            # data_b_2d = self._expand_tiepoint_array(xyz_b[:, :, :, k])
            # data_c_2d = self._expand_tiepoint_array(xyz_c[:, :, :, k])
            # data_d_2d = self._expand_tiepoint_array(xyz_d[:, :, :, k])
            comp_a = xyz_a[:, :, :, k]
            comp_b = xyz_b[:, :, :, k]
            comp_c = xyz_c[:, :, :, k]
            comp_d = xyz_d[:, :, :, k]
            data_a_2d = self._expand_tiepoint_array(comp_a)
            data_b_2d = self._expand_tiepoint_array(comp_b)
            data_c_2d = self._expand_tiepoint_array(comp_c)
            data_d_2d = self._expand_tiepoint_array(comp_d)
            data_a_2d_view = data_a_2d
            data_b_2d_view = data_b_2d
            data_c_2d_view = data_c_2d
            data_d_2d_view = data_d_2d
            print(data_a_2d.shape[0], data_a_2d.shape[1], data_b_2d.shape[0], data_b_2d.shape[1])
            # TODO: assign views
            for i in range(a_scan.shape[0]):
                for j in range(a_scan.shape[1]):
                    atrack1 = a_track[i, j]
                    ascan1 = a_scan[i, j]
                    scan1_tmp = (1 - ascan1) * data_a_2d_view[i, j] + ascan1 * data_b_2d_view[i, j]
                    scan2_tmp = (1 - ascan1) * data_d_2d_view[i, j] + ascan1 * data_c_2d_view[i, j]
                    xyz_comp_view[i, j, k] = (1 - atrack1) * scan1_tmp + atrack1 * scan2_tmp

        # for xyz_comp_a, xyz_comp_b, xyz_comp_c, xyz_comp_d  in zip(xyz_a, xyz_b, xyz_c, xyz_d):
        #     data_a_2d = self._expand_tiepoint_array(xyz_comp_a)
        #     data_b_2d = self._expand_tiepoint_array(xyz_comp_b)
        #     data_c_2d = self._expand_tiepoint_array(xyz_comp_c)
        #     data_d_2d = self._expand_tiepoint_array(xyz_comp_d)
        #
        #     data_1 = (1 - a_scan) * data_a_2d + a_scan * data_b_2d
        #     data_2 = (1 - a_scan) * data_d_2d + a_scan * data_c_2d
        #     comp_arr_2d = (1 - a_track) * data_1 + a_track * data_2
        #     res.append(comp_arr_2d)
        cdef np.ndarray[floating, ndim=2] new_lons = np.empty((a_scan.shape[0], a_scan.shape[1]), dtype=lon1.dtype)
        cdef np.ndarray[floating, ndim=2] new_lats = np.empty((a_scan.shape[0], a_scan.shape[1]), dtype=lon1.dtype)
        cdef floating[:, ::1] new_lons_view = new_lons
        cdef floating[:, ::1] new_lats_view = new_lats
        xyz2lonlat(xyz_comp_view, new_lons_view, new_lats_view)
        return new_lons, new_lats

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
    cdef void _expand_tiepoint_array_1km(self, floating[:, :, :] arr, floating[:, ::1] arr_2d_view) nogil:
        # TODO: Replace shape multiplication with self._fine_pixel_length and self._fine_pixel_width
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
