#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2013-2018 PyTroll community

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

"""Generic interpolation routines.
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline, splev, splrep


def generic_modis5kmto1km(*data5km):
    """Getting 1km data for modis from 5km tiepoints.
    """
    cols5km = np.arange(2, 1354, 5)
    cols1km = np.arange(1354)
    lines = data5km[0].shape[0] * 5
    rows5km = np.arange(2, lines, 5)
    rows1km = np.arange(lines)

    along_track_order = 1
    cross_track_order = 3

    satint = Interpolator(list(data5km),
                          (rows5km, cols5km),
                          (rows1km, cols1km),
                          along_track_order,
                          cross_track_order,
                          chunk_size=10)
    satint.fill_borders("y", "x")
    return satint.interpolate()

# NOTE: extrapolate on a sphere ?


def _linear_extrapolate(pos, data, xev):
    """

    >>> import numpy as np
    >>> pos = np.array([1, 2])
    >>> data = np.arange(10).reshape((2, 5), order="F")
    >>> xev = 5
    >>> retv = _linear_extrapolate(pos, data, xev)
    >>> print([val for val in retv])
    [4.0, 6.0, 8.0, 10.0, 12.0]
    >>> xev = 0
    >>> retv = _linear_extrapolate(pos, data, xev)
    >>> print([val for val in retv])
    [-1.0, 1.0, 3.0, 5.0, 7.0]
    """
    if len(data) != 2 or len(pos) != 2:
        raise ValueError("len(pos) and the number of lines of data"
                         " must be 2.")

    return data[1] + ((xev - pos[1]) / (1.0 * (pos[0] - pos[1])) *
                      (data[0] - data[1]))


class Interpolator(object):

    """
    Handles interpolation of data from a grid of tie points.  It is preferable
    to have tie-points out till the edges if the tiepoint grid, but a method is
    provided to extrapolate linearly the tiepoints to the borders of the
    grid. The extrapolation is done automatically if it seems necessary.

    Uses numpy and scipy.

    The constructor takes in the tiepointed data as *data*, the
    *tiepoint_grid* and the desired *final_grid*. As optional arguments, one
    can provide *kx_* and *ky_* as interpolation orders (in x and y directions
    respectively), and the *chunksize* if the data has to be handled by pieces
    along the y axis (this affects how the extrapolator behaves). If
    *chunksize* is set, don't forget to adjust the interpolation orders
    accordingly: the interpolation is indeed done globaly (not chunkwise).
    """

    def __init__(self, data, tiepoint_grid, final_grid,
                 kx_=1, ky_=1, chunk_size=0):
        self.row_indices = tiepoint_grid[0]
        self.col_indices = tiepoint_grid[1]
        self.hrow_indices = final_grid[0]
        self.hcol_indices = final_grid[1]
        self.chunk_size = chunk_size
        if not isinstance(data, (tuple, list)):
            self.tie_data = [data]
        else:
            self.tie_data = list(data)

        self.new_data = []
        for num in range(len(self.tie_data)):
            self.new_data.append([])

        self.kx_, self.ky_ = kx_, ky_

    def fill_borders(self, *args):
        """Extrapolate tiepoint lons and lats to fill in the border of the
        chunks.
        """

        to_run = []
        cases = {"y": self._fill_row_borders,
                 "x": self._fill_col_borders}
        for dim in args:
            try:
                to_run.append(cases[dim])
            except KeyError:
                raise NameError("Unrecognized dimension: " + str(dim))

        for fun in to_run:
            fun()

    def _extrapolate_cols(self, data, first=True, last=True):
        """Extrapolate the column of data, to get the first and last together
        with the data.

        """

        if first:
            pos = self.col_indices[:2]
            first_column = _linear_extrapolate(pos,
                                               (data[:, 0], data[:, 1]),
                                               self.hcol_indices[0])
        if last:
            pos = self.col_indices[-2:]
            last_column = _linear_extrapolate(pos,
                                              (data[:, -2], data[:, -1]),
                                              self.hcol_indices[-1])

        if first and last:
            return np.hstack((np.expand_dims(first_column, 1),
                              data,
                              np.expand_dims(last_column, 1)))
        elif first:
            return np.hstack((np.expand_dims(first_column, 1),
                              data))
        elif last:
            return np.hstack((data,
                              np.expand_dims(last_column, 1)))
        else:
            return data

    def _fill_col_borders(self):
        """Add the first and last column to the data by extrapolation.
        """

        first = True
        last = True
        if self.col_indices[0] == self.hcol_indices[0]:
            first = False
        if self.col_indices[-1] == self.hcol_indices[-1]:
            last = False
        for num, data in enumerate(self.tie_data):
            self.tie_data[num] = self._extrapolate_cols(data, first, last)

        if first and last:
            self.col_indices = np.concatenate((np.array([self.hcol_indices[0]]),
                                               self.col_indices,
                                               np.array([self.hcol_indices[-1]])))
        elif first:
            self.col_indices = np.concatenate((np.array([self.hcol_indices[0]]),
                                               self.col_indices))
        elif last:
            self.col_indices = np.concatenate((self.col_indices,
                                               np.array([self.hcol_indices[-1]])))

    def _extrapolate_rows(self, data, row_indices, first_index, last_index):
        """Extrapolate the rows of data, to get the first and last together
        with the data.
        """

        pos = row_indices[:2]
        first_row = _linear_extrapolate(pos,
                                        (data[0, :], data[1, :]),
                                        first_index)
        pos = row_indices[-2:]
        last_row = _linear_extrapolate(pos,
                                       (data[-2, :], data[-1, :]),
                                       last_index)
        return np.vstack((np.expand_dims(first_row, 0),
                          data,
                          np.expand_dims(last_row, 0)))

    def _fill_row_borders(self):
        """Add the first and last rows to the data by extrapolation.
        """
        lines = len(self.hrow_indices)
        chunk_size = self.chunk_size or lines
        factor = len(self.hrow_indices) / len(self.row_indices)

        tmp_data = []
        for num in range(len(self.tie_data)):
            tmp_data.append([])
        row_indices = []

        for index in range(0, lines, chunk_size):
            indices = np.logical_and(self.row_indices >= index / factor,
                                     self.row_indices < (index
                                                         + chunk_size) / factor)
            ties = np.argwhere(indices).squeeze()
            tiepos = self.row_indices[indices].squeeze()

            for num, data in enumerate(self.tie_data):
                to_extrapolate = data[ties, :]
                if len(to_extrapolate) > 0:
                    extrapolated = self._extrapolate_rows(to_extrapolate,
                                                          tiepos,
                                                          self.hrow_indices[
                                                              index],
                                                          self.hrow_indices[index + chunk_size - 1])
                    tmp_data[num].append(extrapolated)

            row_indices.append(np.array([self.hrow_indices[index]]))
            row_indices.append(tiepos)
            row_indices.append(np.array([self.hrow_indices[index
                                                           + chunk_size - 1]]))

        for num in range(len(self.tie_data)):
            self.tie_data[num] = np.vstack(tmp_data[num])
        self.row_indices = np.concatenate(row_indices)

    def _interp(self):
        """Interpolate the cartesian coordinates.
        """
        if np.all(self.hrow_indices == self.row_indices):
            return self._interp1d()

        xpoints, ypoints = np.meshgrid(self.hrow_indices,
                                       self.hcol_indices)

        for num, data in enumerate(self.tie_data):
            spl = RectBivariateSpline(self.row_indices,
                                      self.col_indices,
                                      data,
                                      s=0,
                                      kx=self.kx_,
                                      ky=self.ky_)

            new_data_ = spl.ev(xpoints.ravel(), ypoints.ravel())
            self.new_data[num] = new_data_.reshape(xpoints.shape).T.copy(order='C')

    def _interp1d(self):
        """Interpolate in one dimension.
        """
        lines = len(self.hrow_indices)

        for num, data in enumerate(self.tie_data):
            self.new_data[num] = np.empty((len(self.hrow_indices),
                                           len(self.hcol_indices)),
                                          data.dtype)

            for cnt in range(lines):
                tck = splrep(self.col_indices, data[cnt, :], k=self.ky_, s=0)
                self.new_data[num][cnt, :] = splev(
                    self.hcol_indices, tck, der=0)

    def interpolate(self):
        """Do the interpolation, and return resulting longitudes and latitudes.
        """
        self._interp()

        return self.new_data
