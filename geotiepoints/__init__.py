#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2010-2013.

# Author(s):
 
#   Adam Dybbroe <adam.dybbroe@smhise>
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Panu Lahtinen <panu.lahtinen@fmi.fi>

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

"""Interpolation of geographical tiepoints.
"""

import numpy as np
from numpy import arccos, sign, rad2deg, sqrt, arcsin
from scipy.interpolate import RectBivariateSpline, splrep, splev
from multiprocessing import Pool

from geotiepoints.geointerpolator import GeoInterpolator as SatelliteInterpolator

EARTH_RADIUS = 6370997.0


def get_scene_splits(nlines_swath, nlines_scan, n_cpus):
    """Calculate the line numbers where the swath will be split in smaller
    granules for parallel processing"""

    nscans = nlines_swath / nlines_scan
    if nscans < n_cpus:
        nscans_subscene = 1
    else:
        nscans_subscene = nscans / n_cpus
    nlines_subscene = nscans_subscene * nlines_scan

    return range(nlines_subscene, nlines_swath, nlines_subscene)


def metop20kmto1km(lons20km, lats20km):
    """Getting 1km geolocation for metop avhrr from 20km tiepoints.
    """
    cols20km = np.array([0] + range(4, 2048, 20) + [2047])
    cols1km = np.arange(2048)
    lines = lons20km.shape[0]
    rows20km = np.arange(lines)
    rows1km = np.arange(lines)

    along_track_order = 1
    cross_track_order = 3

    satint = SatelliteInterpolator((lons20km, lats20km),
                                   (rows20km, cols20km),
                                   (rows1km, cols1km),
                                   along_track_order,
                                   cross_track_order)
    return satint.interpolate()

def modis5kmto1km(lons5km, lats5km):
    """Getting 1km geolocation for modis from 5km tiepoints.
    """
    cols5km = np.arange(2, 1354, 5)
    cols1km = np.arange(1354)
    lines = lons5km.shape[0] * 5
    rows5km = np.arange(2, lines, 5)
    rows1km = np.arange(lines)

    along_track_order = 1
    cross_track_order = 3

    satint = SatelliteInterpolator((lons5km, lats5km),
                                   (rows5km, cols5km),
                                   (rows1km, cols1km),
                                   along_track_order,
                                   cross_track_order,
                                   chunk_size=10)
    satint.fill_borders("y", "x")
    lons1km, lats1km = satint.interpolate()
    return lons1km, lats1km

def _multi(fun, lons, lats, chunk_size, cores=1):
    """Work on multiple cores.
    """
    pool = Pool(processes=cores)

    splits = get_scene_splits(lons.shape[0], chunk_size, cores)

    lons_parts = np.vsplit(lons, splits)
    lats_parts = np.vsplit(lats, splits)
    
    results = [pool.apply_async(fun,
                                (lons_parts[i],
                                 lats_parts[i]))
               for i in range(len(lons_parts))]
    
    pool.close()
    pool.join()

    lons, lats = zip(*(res.get() for res in results))

    return np.vstack(lons), np.vstack(lats)
    
def modis1kmto500m(lons1km, lats1km, cores=1):
    """Getting 500m geolocation for modis from 1km tiepoints.
    """
    if cores > 1:
        return _multi(modis1kmto500m, lons1km, lats1km, 10, cores)
    
    cols1km = np.arange(0, 2708, 2)
    cols500m = np.arange(2708)
    lines = lons1km.shape[0] * 2
    rows1km = np.arange(0.5, lines, 2)
    rows500m = np.arange(lines)

    along_track_order = 1
    cross_track_order = 3
    
    satint = SatelliteInterpolator((lons1km, lats1km),
                                   (rows1km, cols1km),
                                   (rows500m, cols500m),
                                   along_track_order,
                                   cross_track_order,
                                   chunk_size=20)
    satint.fill_borders("y", "x")
    lons500m, lats500m = satint.interpolate()
    return lons500m, lats500m



def modis1kmto250m(lons1km, lats1km, cores=1):
    """Getting 250m geolocation for modis from 1km tiepoints.
    """
    if cores > 1:
        return _multi(modis1kmto250m, lons1km, lats1km, 10, cores)
    
    cols1km = np.arange(0, 5416, 4)
    cols250m = np.arange(5416)

    along_track_order = 1
    cross_track_order = 3
    
    lines = lons1km.shape[0] * 4
    rows1km = np.arange(1.5, lines, 4)
    rows250m = np.arange(lines)

    satint = SatelliteInterpolator((lons1km, lats1km),
                                   (rows1km, cols1km),
                                   (rows250m, cols250m),
                                   along_track_order,
                                   cross_track_order,
                                   chunk_size=40)
    satint.fill_borders("y", "x")
    lons250m, lats250m = satint.interpolate()

    return lons250m, lats250m


