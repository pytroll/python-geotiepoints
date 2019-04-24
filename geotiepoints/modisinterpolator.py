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

"""Interpolation of geographical tiepoints using the second order interpolation
scheme implemented in the CVIIRS software, as described here:
Compact VIIRS SDR Product Format User Guide (V1J)
http://www.eumetsat.int/website/wcm/idc/idcplg?IdcService=GET_FILE&dDocName=PDF_DMT_708025&RevisionSelectionMethod=LatestReleased&Rendition=Web
"""

import xarray as xr
import dask.array as da
import numpy as np

R = 6371.
# Aqua scan width and altitude in km
scan_width = 10.00017
H = 705.


def compute_phi(zeta):
    return np.arcsin(R * np.sin(zeta) / (R + H))


def compute_theta(zeta, phi):
    return zeta - phi


def compute_zeta(phi):
    return np.arcsin((R + H) * np.sin(phi) / R)


def compute_expansion_alignment(satz_a, satz_b, satz_c, satz_d):
    """All angles in radians."""
    zeta_a = satz_a
    zeta_b = satz_b

    phi_a = compute_phi(zeta_a)
    phi_b = compute_phi(zeta_b)
    theta_a = compute_theta(zeta_a, phi_a)
    theta_b = compute_theta(zeta_b, phi_b)
    phi = (phi_a + phi_b) / 2
    zeta = compute_zeta(phi)
    theta = compute_theta(zeta, phi)

    c_expansion = 4 * (((theta_a + theta_b) / 2 - theta) / (theta_a - theta_b))

    sin_beta_2 = scan_width / (2 * H)

    d = ((R + H) / R * np.cos(phi) - np.cos(zeta)) * sin_beta_2
    e = np.cos(zeta) - np.sqrt(np.cos(zeta) ** 2 - d ** 2)

    c_alignment = 4 * e * np.sin(zeta) / (theta_a - theta_b)

    return c_expansion, c_alignment


def get_corners(arr):
    arr_a = arr[:, :-1, :-1]
    arr_b = arr[:, :-1, 1:]
    arr_c = arr[:, 1:, 1:]
    arr_d = arr[:, 1:, :-1]
    return arr_a, arr_b, arr_c, arr_d


class ModisInterpolator():

    def __init__(self, cres, fres, cscan_full_width=None):
        if cres == 1000:
            self.cscan_len = 10
            self.cscan_width = 1
            self.cscan_full_width = 1354
        elif cres == 5000:
            self.cscan_len = 2
            self.cscan_width = 5
            if cscan_full_width is None:
                self.cscan_full_width = 271
            else:
                self.cscan_full_width = cscan_full_width

        if fres == 250:
            self.fscan_width = 4 * self.cscan_width
            self.fscan_full_width = 1354 * 4
            self.fscan_len = 4 * 10 // self.cscan_len
            self.get_coords = self._get_coords_1km
            self.expand_tiepoint_array = self._expand_tiepoint_array_1km
        elif fres == 500:
            self.fscan_width = 2 * self.cscan_width
            self.fscan_full_width = 1354 * 2
            self.fscan_len = 2 * 10 // self.cscan_len
            self.get_coords = self._get_coords_1km
            self.expand_tiepoint_array = self._expand_tiepoint_array_1km
        elif fres == 1000:
            self.fscan_width = 1 * self.cscan_width
            self.fscan_full_width = 1354
            self.fscan_len = 1 * 10 // self.cscan_len
            self.get_coords = self._get_coords_5km
            self.expand_tiepoint_array = self._expand_tiepoint_array_5km

    def _expand_tiepoint_array_1km(self, arr, lines, cols):
        arr = da.repeat(arr, lines, axis=1)
        arr = da.concatenate((arr[:, :lines//2, :], arr, arr[:, -(lines//2):, :]), axis=1)
        arr = da.repeat(arr.reshape((-1, self.cscan_full_width - 1)), cols, axis=1)
        return da.hstack((arr, arr[:, -cols:]))

    def _get_coords_1km(self, scans):
        y = (np.arange((self.cscan_len + 1) * self.fscan_len) % self.fscan_len) + .5
        y = y[self.fscan_len // 2:-(self.fscan_len // 2)]
        y[:self.fscan_len//2] = np.arange(-self.fscan_len/2 + .5, 0)
        y[-(self.fscan_len//2):] = np.arange(self.fscan_len + .5, self.fscan_len * 3 / 2)
        y = np.tile(y, scans)

        x = np.arange(self.fscan_full_width) % self.fscan_width
        x[-self.fscan_width:] = np.arange(self.fscan_width, self.fscan_width * 2)
        return x, y

    def _expand_tiepoint_array_5km(self, arr, lines, cols):
        arr = da.repeat(arr, lines * 2, axis=1)
        arr = da.repeat(arr.reshape((-1, self.cscan_full_width - 1)), cols, axis=1)
        if self.cscan_full_width == 271:
            return da.hstack((arr[:, :2], arr, arr[:, -2:]))
        else:
            return da.hstack((arr[:, :2], arr, arr[:, -5:], arr[:, -2:]))

    def _get_coords_5km(self, scans):
        y = np.arange(self.fscan_len * self.cscan_len) - 2
        y = np.tile(y, scans)

        x = (np.arange(self.fscan_full_width) - 2) % self.fscan_width
        x[0] = -2
        x[1] = -1
        if self.cscan_full_width == 271:
            x[-2] = 5
            x[-1] = 6
        elif self.cscan_full_width == 270:
            x[-7] = 5
            x[-6] = 6
            x[-5] = 7
            x[-4] = 8
            x[-3] = 9
            x[-2] = 10
            x[-1] = 11
        else:
            raise NotImplementedError("Can't interpolate if 5km tiepoints have less than 270 columns.")
        return x, y

    def interpolate(self, lon1, lat1, satz1):
        cscan_len = self.cscan_len
        cscan_full_width = self.cscan_full_width

        fscan_width = self.fscan_width
        fscan_len = self.fscan_len

        scans = satz1.shape[0] // cscan_len
        satz1 = satz1.data

        satz1 = satz1.reshape((-1, cscan_len, cscan_full_width))

        satz_a, satz_b, satz_c, satz_d = get_corners(da.deg2rad(satz1))

        c_exp, c_ali = compute_expansion_alignment(satz_a, satz_b, satz_c, satz_d)

        x, y = self.get_coords(scans)
        i_rs, i_rt = da.meshgrid(x, y)

        p_os = 0
        p_ot = 0

        s_s = (p_os + i_rs) * 1. / fscan_width
        s_t = (p_ot + i_rt) * 1. / fscan_len

        cols = fscan_width
        lines = fscan_len

        c_exp_full = self.expand_tiepoint_array(c_exp, lines, cols)
        c_ali_full = self.expand_tiepoint_array(c_ali, lines, cols)

        a_track = s_t
        a_scan = (s_s + s_s * (1 - s_s) * c_exp_full + s_t*(1 - s_t) * c_ali_full)

        res = []

        sublat = lat1[::16, ::16]
        sublon = lon1[::16, ::16]
        to_cart = abs(sublat).max() > 60 or (sublon.max() - sublon.min()) > 180

        if to_cart:
            datasets = lonlat2xyz(lon1, lat1)
        else:
            datasets = [lon1, lat1]

        for data in datasets:
            data_attrs = data.attrs
            dims = data.dims
            data = data.data
            data = data.reshape((-1, cscan_len, cscan_full_width))
            data_a, data_b, data_c, data_d = get_corners(data)
            data_a = self.expand_tiepoint_array(data_a, lines, cols)
            data_b = self.expand_tiepoint_array(data_b, lines, cols)
            data_c = self.expand_tiepoint_array(data_c, lines, cols)
            data_d = self.expand_tiepoint_array(data_d, lines, cols)

            data_1 = (1 - a_scan) * data_a + a_scan * data_b
            data_2 = (1 - a_scan) * data_d + a_scan * data_c
            data = (1 - a_track) * data_1 + a_track * data_2

            res.append(xr.DataArray(data, attrs=data_attrs, dims=dims))

        if to_cart:
            return xyz2lonlat(*res)
        else:
            return res


def modis_1km_to_250m(lon1, lat1, satz1):

    interp = ModisInterpolator(1000, 250)
    return interp.interpolate(lon1, lat1, satz1)


def modis_1km_to_500m(lon1, lat1, satz1):

    interp = ModisInterpolator(1000, 500)
    return interp.interpolate(lon1, lat1, satz1)


def modis_5km_to_1km(lon1, lat1, satz1):

    interp = ModisInterpolator(5000, 1000, lon1.shape[1])
    return interp.interpolate(lon1, lat1, satz1)

def lonlat2xyz(lons, lats):
    """Convert lons and lats to cartesian coordinates."""
    R = 6370997.0
    x_coords = R * da.cos(da.deg2rad(lats)) * da.cos(da.deg2rad(lons))
    y_coords = R * da.cos(da.deg2rad(lats)) * da.sin(da.deg2rad(lons))
    z_coords = R * da.sin(da.deg2rad(lats))
    return x_coords, y_coords, z_coords

def xyz2lonlat(x__, y__, z__):
    """Get longitudes from cartesian coordinates.
    """
    R = 6370997.0
    lons = da.rad2deg(da.arccos(x__ / da.sqrt(x__ ** 2 + y__ ** 2))) * da.sign(y__)
    lats = da.sign(z__) * (90 - da.rad2deg(da.arcsin(da.sqrt(x__ ** 2 + y__ ** 2) / R)))

    return lons, lats
