from libc.math cimport fmin, fmax, floor
cimport cython
from cython.parallel import prange,parallel
from .simple_modis_interpolator import scanline_mapblocks

cimport numpy as np
import numpy as np

# ctypedef fused floating:
#     float
#     double

ctypedef fused floating:
    np.float32_t
    np.float64_t

EARTH_RADIUS = 6370997.0
R = 6371.0
# Aqua scan width and altitude in km
scan_width = 10.00017
H = 705.0


def lonlat2xyz(
        lons, lats,
        # floating[:, :] lons, floating[:, :] lats,
        floating radius=EARTH_RADIUS):
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
        floating radius=EARTH_RADIUS,
        floating thr=0.8,
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
    interp = Interpolator(coarse_resolution, fine_resolution, coarse_scan_width=coarse_scan_width or 0)
    return interp.interpolate(lon1, lat1, satz1)


def _compute_phi(zeta):
    return np.arcsin(R * np.sin(zeta) / (R + H))


def _compute_theta(zeta, phi):
    return zeta - phi


def _compute_zeta(phi):
    return np.arcsin((R + H) * np.sin(phi) / R)


def _compute_expansion_alignment(satz_a, satz_b):
    """All angles in radians."""
    zeta_a = satz_a
    zeta_b = satz_b

    phi_a = _compute_phi(zeta_a)
    phi_b = _compute_phi(zeta_b)
    theta_a = _compute_theta(zeta_a, phi_a)
    theta_b = _compute_theta(zeta_b, phi_b)
    phi = (phi_a + phi_b) / 2
    zeta = _compute_zeta(phi)
    theta = _compute_theta(zeta, phi)
    # Workaround for tiepoints symetrical about the subsatellite-track
    denominator = np.where(theta_a == theta_b, theta_a * 2, theta_a - theta_b)

    c_expansion = 4 * (((theta_a + theta_b) / 2 - theta) / denominator)

    sin_beta_2 = scan_width / (2 * H)

    d = ((R + H) / R * np.cos(phi) - np.cos(zeta)) * sin_beta_2
    e = np.cos(zeta) - np.sqrt(np.cos(zeta) ** 2 - d ** 2)

    c_alignment = 4 * e * np.sin(zeta) / denominator

    return c_expansion, c_alignment


def _get_corners(arr):
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
            np.ndarray[floating, ndim=2] satz1_):
        """Interpolate MODIS geolocation from 'coarse_resolution' to 'fine_resolution'."""
        cdef unsigned int scans = satz1_.shape[0] // self._coarse_scan_length
        # reshape to (num scans, rows per scan, columns per scan)
        cdef np.ndarray[floating, ndim=3] satz1 = satz1_.reshape((-1, self._coarse_scan_length, self._coarse_scan_width))
        # cdef floating [:, :, :] satz1 = satz1_.reshape((-1, self._coarse_scan_length, self._coarse_scan_width))

        # satz_a, satz_b = _get_corners(np.deg2rad(satz1))[:2]
        corners = _get_corners(np.deg2rad(satz1))
        cdef np.ndarray[floating, ndim=3] satz_a = corners[0]
        cdef np.ndarray[floating, ndim=3] satz_b = corners[1]
        # c_exp, c_ali = _compute_expansion_alignment(satz_a, satz_b)
        exp_alignments = _compute_expansion_alignment(satz_a, satz_b)
        cdef np.ndarray[floating, ndim=3] c_exp = exp_alignments[0]
        cdef np.ndarray[floating, ndim=3] c_ali = exp_alignments[1]

        # x, y = self._get_coords(scans)
        coords_xy = self._get_coords(scans)
        cdef np.ndarray[floating, ndim=1] x = coords_xy[0]
        cdef np.ndarray[floating, ndim=1] y = coords_xy[1]
        i_rs, i_rt = np.meshgrid(x, y)

        p_os = 0
        p_ot = 0
        s_s = (p_os + i_rs) * 1.0 / self._fine_pixels_per_coarse_pixel
        s_t = (p_ot + i_rt) * 1.0 / self._fine_scan_length

        cdef np.ndarray[floating, ndim=2] c_exp_full = self._expand_tiepoint_array(c_exp)
        cdef np.ndarray[floating, ndim=2] c_ali_full = self._expand_tiepoint_array(c_ali)

        a_track = s_t
        a_scan = s_s + s_s * (1 - s_s) * c_exp_full + s_t * (1 - s_t) * c_ali_full

        res = []
        datasets = lonlat2xyz(lon1, lat1)
        cdef np.ndarray[floating, ndim=3] data
        cdef np.ndarray[floating, ndim=3] data_a, data_b, data_c, data_d
        cdef np.ndarray[floating, ndim=2] data_a_2d, data_b_2d, data_c_2d, data_d_2d
        cdef np.ndarray[floating, ndim=2] comp_arr_2d
        for data_2d in datasets:
            data = data_2d.reshape((-1, self._coarse_scan_length, self._coarse_scan_width))
            # data_a, data_b, data_c, data_d = _get_corners(data)
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

            res.append(comp_arr_2d)
        new_lons, new_lats = xyz2lonlat(*res)
        return new_lons.astype(lon1.dtype), new_lats.astype(lat1.dtype)

    cdef _get_coords_1km(self, unsigned int scans):
        cdef np.ndarray[np.float32_t, ndim=1] y = (np.arange((self._coarse_scan_length + 1) * self._fine_scan_length, dtype=np.float32) % self._fine_scan_length) + 0.5
        cdef int half_scan_length = self._fine_scan_length // 2
        cdef np.ndarray[np.float32_t, ndim=1] y2 = y[half_scan_length:-half_scan_length]
        y2[:half_scan_length] = np.arange(-self._fine_scan_length / 2 + 0.5, 0)
        y2[-half_scan_length:] = np.arange(self._fine_scan_length + 0.5, self._fine_scan_length * 3 / 2)
        cdef np.ndarray[np.float32_t, ndim=1] y3 = np.tile(y2, scans)

        cdef np.ndarray[np.float32_t, ndim=1] x = np.arange(self._fine_scan_width, dtype=np.float32) % self._fine_pixels_per_coarse_pixel
        x[-self._fine_pixels_per_coarse_pixel:] = np.arange(
            self._fine_pixels_per_coarse_pixel,
            self._fine_pixels_per_coarse_pixel * 2)
        return x, y3

    # def _get_coords_5km(self, unsigned int scans):
    cdef _get_coords_5km(self, unsigned int scans):
        y = np.arange(self._fine_scan_length * self._coarse_scan_length) - 2
        y = np.tile(y, scans)

        x = (np.arange(self._fine_scan_width) - 2) % self._fine_pixels_per_coarse_pixel
        x[0] = -2
        x[1] = -1
        if self._coarse_scan_width == 271:
            x[-2] = 5
            x[-1] = 6
        elif self._coarse_scan_width == 270:
            x[-7] = 5
            x[-6] = 6
            x[-5] = 7
            x[-4] = 8
            x[-3] = 9
            x[-2] = 10
            x[-1] = 11
        else:
            raise NotImplementedError(
                "Can't interpolate if 5km tiepoints have less than 270 columns."
            )
        return x, y

    cdef np.ndarray[floating, ndim=2] _expand_tiepoint_array_1km(self, np.ndarray[floating, ndim=3] arr):
        arr = np.repeat(arr, self._fine_scan_length, axis=1)
        arr = np.concatenate(
            (arr[:, :self._fine_scan_length // 2, :], arr, arr[:, -(self._fine_scan_length // 2):, :]), axis=1
        )
        cdef np.ndarray[floating, ndim=2] arr_2d = np.repeat(arr.reshape((-1, self._coarse_scan_width - 1)), self._fine_pixels_per_coarse_pixel, axis=1)
        return np.hstack((arr_2d, arr_2d[:, -self._fine_pixels_per_coarse_pixel:]))

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
