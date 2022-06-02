#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 PyTroll community

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

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

"""Interpolation of MODIS data using satellite zenith angle.

Interpolation of geographical tiepoints using the second order interpolation
scheme implemented in the CVIIRS software, as described here:
Compact VIIRS SDR Product Format User Guide (V1J)
https://www.eumetsat.int/media/45988
and
Anders Meier Soerensen, Stephan Zinke,
A tie-point zone group compaction schema for the geolocation data of S-NPP and NOAA-20 VIIRS SDRs to reduce file sizes
in memory-sensitive environments,
Applied Computing and Geosciences, Volume 6, 2020, 100025, ISSN 2590-1974,
https://doi.org/10.1016/j.acags.2020.100025.
(https://www.sciencedirect.com/science/article/pii/S2590197420300070)
"""

import numpy as np
import warnings

from .geointerpolator import lonlat2xyz, xyz2lonlat
from .simple_modis_interpolator import scanline_mapblocks

R = 6371.
# Aqua altitude in km
H = 709.


def _compute_phi(zeta):
    return np.arcsin(R * np.sin(zeta) / (R + H))


def _compute_theta(zeta, phi):
    return zeta - phi


def _compute_zeta(phi):
    return np.arcsin((R + H) * np.sin(phi) / R)


def _compute_expansion_alignment(satz_a, satz_b, scan_width):
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


@scanline_mapblocks
def _interpolate(
        lon1,
        lat1,
        satz1,
        coarse_resolution=None,
        fine_resolution=None,
        coarse_scan_width=None,
):
    """Helper function to interpolate scan-aligned arrays.

    This function's decorator runs this function for each dask block/chunk of
    scans. The arrays are scan-aligned meaning they are an even number of scans
    (N rows per scan) and contain the entire scan width.

    """
    interp = _MODISInterpolator(coarse_resolution, fine_resolution, coarse_scan_width=coarse_scan_width)
    return interp.interpolate(lon1, lat1, satz1)


class _MODISInterpolator:
    """Helper class for MODIS interpolation.

    Not intended for public use. Use ``modis_X_to_Y`` functions instead.

    """
    def __init__(self, coarse_resolution, fine_resolution, coarse_scan_width=None):
        if coarse_resolution == 1000:
            coarse_scan_length = 10
            coarse_scan_width = 1354
            self._get_coords = self._get_coords_1km
            self._expand_tiepoint_array = self._expand_tiepoint_array_1km
        elif coarse_resolution == 5000:
            coarse_scan_length = 2
            self._get_coords = self._get_coords_5km
            self._expand_tiepoint_array = self._expand_tiepoint_array_5km
            if coarse_scan_width is None:
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

    def interpolate(self, lon1, lat1, satz1):
        """Interpolate MODIS geolocation from 'coarse_resolution' to 'fine_resolution'."""
        scans = satz1.shape[0] // self._coarse_scan_length
        # reshape to (num scans, rows per scan, columns per scan)
        satz1 = satz1.reshape((-1, self._coarse_scan_length, self._coarse_scan_width))

        satz_a, satz_b = _get_corners(np.deg2rad(satz1))[:2]
        c_exp, c_ali = _compute_expansion_alignment(satz_a, satz_b, self._coarse_pixels_per_1km)

        x, y = self._get_coords(scans)
        i_rs, i_rt = np.meshgrid(x, y)

        p_os = 0
        p_ot = 0
        s_s = (p_os + i_rs) * 1.0 / self._fine_pixels_per_coarse_pixel
        s_t = (p_ot + i_rt) * 1.0 / self._fine_scan_length

        c_exp_full = self._expand_tiepoint_array(c_exp)
        c_ali_full = self._expand_tiepoint_array(c_ali)

        a_track = s_t
        a_scan = s_s + s_s * (1 - s_s) * c_exp_full + s_t * (1 - s_t) * c_ali_full

        res = []
        datasets = lonlat2xyz(lon1, lat1)
        for data in datasets:
            data = data.reshape((-1, self._coarse_scan_length, self._coarse_scan_width))
            data_a, data_b, data_c, data_d = _get_corners(data)
            data_a = self._expand_tiepoint_array(data_a)
            data_b = self._expand_tiepoint_array(data_b)
            data_c = self._expand_tiepoint_array(data_c)
            data_d = self._expand_tiepoint_array(data_d)

            data_1 = (1 - a_scan) * data_a + a_scan * data_b
            data_2 = (1 - a_scan) * data_d + a_scan * data_c
            data = (1 - a_track) * data_1 + a_track * data_2

            res.append(data)
        new_lons, new_lats = xyz2lonlat(*res)
        return new_lons.astype(lon1.dtype), new_lats.astype(lat1.dtype)

    def _get_coords_1km(self, scans):
        y = (
            np.arange((self._coarse_scan_length + 1) * self._fine_scan_length) % self._fine_scan_length
        ) + 0.5
        half_scan_length = self._fine_scan_length // 2
        y = y[half_scan_length:-half_scan_length]
        y[:half_scan_length] = np.arange(-self._fine_scan_length / 2 + 0.5, 0)
        y[-half_scan_length:] = np.arange(self._fine_scan_length + 0.5, self._fine_scan_length * 3 / 2)
        y = np.tile(y, scans)

        x = np.arange(self._fine_scan_width) % self._fine_pixels_per_coarse_pixel
        x[-self._fine_pixels_per_coarse_pixel:] = np.arange(
            self._fine_pixels_per_coarse_pixel,
            self._fine_pixels_per_coarse_pixel * 2)
        return x, y

    def _get_coords_5km(self, scans):
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

    def _expand_tiepoint_array_1km(self, arr):
        arr = np.repeat(arr, self._fine_scan_length, axis=1)
        arr = np.concatenate(
            (arr[:, :self._fine_scan_length // 2, :], arr, arr[:, -(self._fine_scan_length // 2):, :]), axis=1
        )
        arr = np.repeat(arr.reshape((-1, self._coarse_scan_width - 1)), self._fine_pixels_per_coarse_pixel, axis=1)
        return np.hstack((arr, arr[:, -self._fine_pixels_per_coarse_pixel:]))

    def _expand_tiepoint_array_5km(self, arr):
        arr = np.repeat(arr, self._fine_scan_length * 2, axis=1)
        arr = np.repeat(arr.reshape((-1, self._coarse_scan_width - 1)), self._fine_pixels_per_coarse_pixel, axis=1)
        factor = self._fine_pixels_per_coarse_pixel // self._coarse_pixels_per_1km
        if self._coarse_scan_width == 271:
            return np.hstack((arr[:, :2 * factor], arr, arr[:, -2 * factor:]))
        else:
            return np.hstack(
                (
                    arr[:, :2 * factor],
                    arr,
                    arr[:, -self._fine_pixels_per_coarse_pixel:],
                    arr[:, -2 * factor:],
                )
            )


def modis_1km_to_250m(lon1, lat1, satz1):
    """Interpolate MODIS geolocation from 1km to 250m resolution."""
    return _interpolate(lon1, lat1, satz1,
                        coarse_resolution=1000,
                        fine_resolution=250)


def modis_1km_to_500m(lon1, lat1, satz1):
    """Interpolate MODIS geolocation from 1km to 500m resolution."""
    return _interpolate(lon1, lat1, satz1,
                        coarse_resolution=1000,
                        fine_resolution=500)


def modis_5km_to_1km(lon1, lat1, satz1):
    """Interpolate MODIS geolocation from 5km to 1km resolution."""
    return _interpolate(lon1, lat1, satz1,
                        coarse_resolution=5000,
                        fine_resolution=1000,
                        coarse_scan_width=lon1.shape[1])


def modis_5km_to_500m(lon1, lat1, satz1):
    """Interpolate MODIS geolocation from 5km to 500m resolution."""
    warnings.warn(
        "Interpolating 5km geolocation to 500m resolution " "may result in poor quality"
    )
    return _interpolate(lon1, lat1, satz1,
                        coarse_resolution=5000,
                        fine_resolution=500,
                        coarse_scan_width=lon1.shape[1])


def modis_5km_to_250m(lon1, lat1, satz1):
    """Interpolate MODIS geolocation from 5km to 250m resolution."""
    warnings.warn(
        "Interpolating 5km geolocation to 250m resolution " "may result in poor quality"
    )
    return _interpolate(lon1, lat1, satz1,
                        coarse_resolution=5000,
                        fine_resolution=250,
                        coarse_scan_width=lon1.shape[1])
