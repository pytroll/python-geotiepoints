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

import os, sys

import numpy as np
from numpy import arccos, sign, rad2deg, sqrt, arcsin
from scipy.interpolate import RectBivariateSpline, splrep, splev
from multiprocessing import Pool



EARTH_RADIUS = 6370997.0

# import logging
# LOG = logging.getLogger(__name__)

# #: Default time format
# _DEFAULT_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# #: Default log format
# _DEFAULT_LOG_FORMAT = '[%(levelname)s: %(asctime)s : %(name)s] %(message)s'

# _PYTHON_GEOTIEPOINTS_LOGFILE = os.environ.get('PYTHON_GEOTIEPOINTS_LOGFILE', None)
# if _PYTHON_GEOTIEPOINTS_LOGFILE:
#     ndays = int(OPTIONS.get("log_rotation_days", 1))
#     ncount = int(OPTIONS.get("log_rotation_backup", 5))
#     handler = handlers.TimedRotatingFileHandler(_PYTHON_GEOTIEPOINTS_LOGFILE,
#                                                 when='midnight', 
#                                                 interval=ndays, 
#                                                 backupCount=ncount, 
#                                                 encoding=None, 
#                                                 delay=False, 
#                                                 utc=True)

# else:
#     handler = logging.StreamHandler(sys.stderr)

# formatter = logging.Formatter(fmt=_DEFAULT_LOG_FORMAT,
#                               datefmt=_DEFAULT_TIME_FORMAT)
# handler.setFormatter(formatter)

# handler.setLevel(logging.DEBUG)
# LOG.setLevel(logging.DEBUG)
# LOG.addHandler(handler)

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


# def get_cpus(n_cpus):
#     """Get the number of CPUs to use for parallel proccessing"""
#     from multiprocessing import cpu_count

#     ncpus_available = cpu_count()
#     LOG.debug('Number of CPUs detected = %d' % ncpus_available)
#     if n_cpus:
#         if n_cpus > ncpus_available:
#             LOG.warning("Asking to use more CPUs than what is available!")
#             LOG.info("Setting number of CPUs to %d" % ncpus_available)
#             n_cpus = ncpus_available
#     else:
#         n_cpus = ncpus_available

#     LOG.debug('Using %d CPUs...' % n_cpus)
#     return n_cpus

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


# def modis1kmto250m(lons1km, lats1km, parallel=True, ncpus=None):
#     """Getting 250m geolocation for modis from 1km tiepoints.
#     Using multiprocessing in case of several cpus available
#     """
#     from multiprocessing import Process, Queue
#     #from threading import Thread as Process
#     #from Queue import Queue

#     if (ncpus and ncpus == 1) or not parallel:
#         return _modis1kmto250m(lons1km, lats1km, False)

#     num_of_cpus = get_cpus(ncpus)

#     # A Modis 1km scan spans 10 lines:
#     scene_splits = get_scene_splits(lons1km.shape[0], 
#                                     10, num_of_cpus)
#     LOG.debug("Scene-splits: " + str(scene_splits))

#     # Cut the swath in pieces and do processing in parallel:
#     scenes = []
#     queuelist = []
#     lons_subscenes = np.vsplit(lons1km, scene_splits)
#     lats_subscenes = np.vsplit(lats1km, scene_splits)
#     LOG.debug("len(lons_subscenes): " + str(len(lons_subscenes)))
#     for longs in lons_subscenes: 
#         LOG.debug("lons-subscene shapes: " + str(longs.shape))

#     for idx in range(len(lons_subscenes)):
#         lons = lons_subscenes[idx]
#         lats = lats_subscenes[idx]
#         #LOG.debug("Line number: " + str(scene_splits[idx]))
#         LOG.debug("Shape of lon/lat arrays: " + str(lons.shape))
        
#         queuelist.append(Queue())
#         scene = Process(target=_modis1kmto250m, 
#                         args=(lons, lats, queuelist[idx]))
#         scenes.append(scene)

#     LOG.debug("Number of queues: " + str(len(queuelist)))
#     LOG.debug("Number of processes: " + str(len(scenes)))

#     # Go through the Process list:
#     starts = [ scene.start() for scene in scenes ]
#     LOG.debug("%d processes startet" % len(starts))

#     results = [ que.get() for que in queuelist ]
#     LOG.debug("%d results retrieved" % len(results))

#     joins = [ scene.join() for scene in scenes ]
#     LOG.debug("%d processes joined" % len(joins))

#     lonlist = [ res[0] for res in results ]
#     latlist = [ res[1] for res in results ]
        
#     lons250m = np.concatenate(lonlist)
#     lats250m = np.concatenate(latlist)

#     return lons250m, lats250m


# def faster_modis1kmto250m(lons1km, lats1km, parallel=True, ncpus=None):
#     """Getting 250m geolocation for modis from 1km tiepoints.
#     Using multiprocesing to speed up performance.
#     """
#     cols1km = np.arange(0, 5416, 4)
#     cols250m = np.arange(5416)

#     along_track_order = 1
#     cross_track_order = 3
    
#     from multiprocessing import Process, Queue

#     def ipol_scene(scene_lons, scene_lats, que=None):
#         lines = scene_lons.shape[0] * 4
#         rows1km = np.arange(1.5, lines, 4)
#         rows250m = np.arange(lines)

#         satint = SatelliteInterpolator((scene_lons, scene_lats),
#                                        (rows1km, cols1km),
#                                        (rows250m, cols250m),
#                                        along_track_order,
#                                        cross_track_order,
#                                        chunk_size=40)
#         satint.fill_borders("y", "x")
#         lons_250m, lats_250m = satint.interpolate()
#         if que:
#             que.put((lons_250m, lats_250m))

#         return (lons_250m, lats_250m)

#     if (ncpus and ncpus == 1) or not parallel:
#         return ipol_scene(lons1km, lats1km, False)

#     num_of_cpus = get_cpus(ncpus)

#     # A Modis 1km scan spans 10 lines:
#     scene_splits = get_scene_splits(lons1km.shape[0], 
#                                     10, num_of_cpus)

#     # Cut the swath in pieces and do processing in parallel:
#     scenes = []
#     queuelist = []
#     lons_subscenes = np.vsplit(lons1km, scene_splits)
#     lats_subscenes = np.vsplit(lats1km, scene_splits)
#     for idx in range(len(scene_splits)):
#         lons = lons_subscenes[idx]
#         lats = lats_subscenes[idx]
#         LOG.debug("Line number: " + str(scene_splits[idx]))
            
#         queuelist.append(Queue())
#         scene = Process(target=ipol_scene, 
#                         args=(lons, lats, queuelist[idx]))
#         scenes.append(scene)

#     LOG.debug("Number of queues: " + str(len(queuelist)))
#     LOG.debug("Number of processes: " + str(len(scenes)))

#     # Go through the Process list:
#     starts = [ scene.start() for scene in scenes ]
#     LOG.debug("%d processes startet" % len(starts))

#     results = [ que.get() for que in queuelist ]
#     LOG.debug("%d results retrieved" % len(results))

#     joins = [ scene.join() for scene in scenes ]
#     LOG.debug("%d processes joined" % len(joins))

#     lonlist = [ res[0] for res in results ]
#     latlist = [ res[1] for res in results ]
        
#     lons250m = np.concatenate(lonlist)
#     lats250m = np.concatenate(latlist)

#     return lons250m, lats250m


# NOTE: extrapolate on a sphere ?
def _linear_extrapolate(pos, data, xev):
    """

    >>> import numpy as np
    >>> pos = np.array([1, 2])
    >>> data = np.arange(10).reshape((2, 5), order="F")
    >>> xev = 5
    >>> _linear_extrapolate(pos, data, xev)
    array([  4.,   6.,   8.,  10.,  12.])
    >>> xev = 0
    >>> _linear_extrapolate(pos, data, xev)
    array([-1.,  1.,  3.,  5.,  7.])
    """
    if len(data) != 2 or len(pos) != 2:
        raise ValueError("len(pos) and the number of lines of data"
                         " must be 2.")

    return data[1] + ((xev - pos[1]) / (1.0 * (pos[0] - pos[1])) *
                      (data[0] - data[1]))

class SatelliteInterpolator(object):
    """
    Handles interpolation of geolocation data from a grid of tie points.  It is
    preferable to have tie-points out till the edges if the tiepoint grid, but
    a method is provided to extrapolate linearly the tiepoints to the borders
    of the grid.

    Uses numpy, scipy, and pyresample

    The constructor takes in the tiepointed lon/lat data as *lon_lat_data*, the
    *tiepoint_grid* and the desired *final_grid*. As optional arguments, one
    can provide *kx_* and *ky_* as interpolation orders (in x and y directions
    respectively), and the *chunksize* if the data has to be handled by pieces
    along the y axis (this affects how the extrapolator behaves). If
    *chunksize* is set, don't forget to adjust the interpolation orders
    accordingly: the interpolation is indeed done globaly (not chunkwise).
    """

    def __init__(self, lon_lat_data, tiepoint_grid, final_grid,
                 kx_=1, ky_=1, chunk_size=0):
        self.row_indices = tiepoint_grid[0]
        self.col_indices = tiepoint_grid[1]
        self.hrow_indices = final_grid[0]
        self.hcol_indices = final_grid[1]
        self.chunk_size = chunk_size
        self.lon_tiepoint = None
        self.lat_tiepoint = None
        self.longitude = None
        self.latitude = None

        try:
            # Maybe it's a pyresample object ?
            self.set_tiepoints(lon_lat_data.lons, lon_lat_data.lats)
            xyz = lon_lat_data.get_cartesian_coords()
            self.x__ = xyz[:, :, 0]
            self.y__ = xyz[:, :, 1]
            self.z__ = xyz[:, :, 2]

        except AttributeError:
            self.set_tiepoints(lon_lat_data[0], lon_lat_data[1])
            lons_rad = np.radians(self.lon_tiepoint)
            lats_rad = np.radians(self.lat_tiepoint)
            self.x__ = EARTH_RADIUS * np.cos(lats_rad) * np.cos(lons_rad)
            self.y__ = EARTH_RADIUS * np.cos(lats_rad) * np.sin(lons_rad)
            self.z__ = EARTH_RADIUS * np.sin(lats_rad)

        self.newx = None
        self.newy = None
        self.newz = None

        self.kx_, self.ky_ = kx_, ky_
        
    def set_tiepoints(self, lon, lat):
        """Defines the lon,lat tie points.
        """
        self.lon_tiepoint = lon
        self.lat_tiepoint = lat

    def fill_borders(self, *args):
        """Extrapolate tiepoint lons and lats to fill in the border of the
        chunks.

        >>> import numpy as np
        >>> lons = np.arange(20).reshape((4, 5), order="F")
        >>> lats = np.arange(20).reshape((4, 5), order="C")
        >>> lines = np.array([2, 7, 12, 17])
        >>> cols = np.array([2, 7, 12, 17, 22])
        >>> hlines = np.arange(20)
        >>> hcols = np.arange(24)
        >>> satint = SatelliteInterpolator((lons, lats), (lines, cols), (hlines, hcols), chunk_size=10)
        >>> satint.fill_borders('x', 'y')
        >>> satint.x__
        array([[ 6384905.78040055,  6381081.08333225,  6371519.34066148,
                 6328950.00792935,  6253610.69157758,  6145946.19489936,
                 6124413.29556372],
               [ 6377591.95940176,  6370997.        ,  6354509.6014956 ,
                 6305151.62592155,  6223234.99818839,  6109277.14889072,
                 6086485.57903118],
               [ 6359307.40690478,  6345786.79166939,  6311985.2535809 ,
                 6245655.67090206,  6147295.76471541,  6017604.5338691 ,
                 5991666.28769983],
               [ 6351993.58590599,  6335702.70833714,  6294975.51441502,
                 6221857.28889426,  6116920.07132621,  5980935.48786045,
                 5953738.5711673 ],
               [ 6338032.26190294,  6320348.4990906 ,  6276139.09205974,
                 6199670.56624433,  6091551.90273768,  5952590.38414781,
                 5924798.08042984],
               [ 6290665.5946295 ,  6270385.16249031,  6219684.08214232,
                 6137100.75832981,  6023313.2794414 ,  5879194.72399075,
                 5850371.01290062],
               [ 6172248.92644589,  6145476.82098957,  6078546.55734877,
                 5980676.23854351,  5852716.72120069,  5695705.57359808,
                 5664303.34407756],
               [ 6124882.25917245,  6095513.48438928,  6022091.54743135,
                 5918106.430629  ,  5784478.09790441,  5622309.91344102,
                 5589876.27654834]])
        >>> satint.row_indices
        array([ 0,  2,  7,  9, 10, 12, 17, 19])
        >>> satint.col_indices
        array([ 0,  2,  7, 12, 17, 22, 23])
        """
        
        to_run = []
        cases = {"y": self._fill_row_borders,
                 "x": self._fill_col_borders}
        for dim in args:
            try:
                to_run.append(cases[dim])
            except KeyError:
                raise NameError("Unrecognized dimension: "+str(dim))

        for fun in to_run:
            fun()


    def _extrapolate_cols(self, data, first=True, last=True):
        """Extrapolate the column of data, to get the first and last together
        with the data.

        >>> import numpy as np
        >>> lons = np.arange(10).reshape((2, 5), order="F")
        >>> lats = np.arange(10).reshape((2, 5), order="C")
        >>> lines = np.array([2, 7])
        >>> cols = np.array([2, 7, 12, 17, 22])
        >>> hlines = np.arange(10)
        >>> hcols = np.arange(24)
        >>> satint = SatelliteInterpolator((lons, lats), (lines, cols), (hlines, hcols))
        >>> satint._extrapolate_cols(satint.x__)
        array([[ 6372937.31273379,  6370997.        ,  6366146.21816553,
                 6351605.98629588,  6327412.61244969,  6293626.50067273,
                 6286869.27831734],
               [ 6353136.46335726,  6345786.79166939,  6327412.61244969,
                 6299445.69529922,  6261968.60390423,  6215087.60607344,
                 6205711.40650728]])
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

        >>> import numpy as np
        >>> lons = np.arange(10).reshape((2, 5), order="F")
        >>> lats = np.arange(10).reshape((2, 5), order="C")
        >>> lines = np.array([2, 7])
        >>> cols = np.array([2, 7, 12, 17, 22])
        >>> hlines = np.arange(10)
        >>> hcols = np.arange(24)
        >>> satint = SatelliteInterpolator((lons, lats), (lines, cols), (hlines, hcols))
        >>> satint._fill_col_borders()
        >>> satint.x__
        array([[ 6372937.31273379,  6370997.        ,  6366146.21816553,
                 6351605.98629588,  6327412.61244969,  6293626.50067273,
                 6286869.27831734],
               [ 6353136.46335726,  6345786.79166939,  6327412.61244969,
                 6299445.69529922,  6261968.60390423,  6215087.60607344,
                 6205711.40650728]])
        >>> satint.col_indices
        array([ 0,  2,  7, 12, 17, 22, 23])
        """
        
        first = True
        last = True
        if self.col_indices[0] == self.hcol_indices[0]:
            first = False
        if self.col_indices[-1] == self.hcol_indices[-1]:
            last = False        

        self.x__ = self._extrapolate_cols(self.x__, first, last)
        self.y__ = self._extrapolate_cols(self.y__, first, last)
        self.z__ = self._extrapolate_cols(self.z__, first, last)
  
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


    def _extrapolate_rows(self, data):
        """Extrapolate the rows of data, to get the first and last together
        with the data.

        >>> import numpy as np
        >>> lons = np.arange(10).reshape((2, 5), order="F")
        >>> lats = np.arange(10).reshape((2, 5), order="C")
        >>> lines = np.array([2, 7])
        >>> cols = np.array([2, 7, 12, 17, 22])
        >>> hlines = np.arange(10)
        >>> hcols = np.arange(24)
        >>> satint = SatelliteInterpolator((lons, lats), (lines, cols), (hlines, hcols))
        >>> satint._extrapolate_rows(satint.x__)
        array([[ 6381081.08333225,  6381639.66045187,  6372470.10269454,
                 6353590.21586788,  6325042.05851245],
               [ 6370997.        ,  6366146.21816553,  6351605.98629588,
                 6327412.61244969,  6293626.50067273],
               [ 6345786.79166939,  6327412.61244969,  6299445.69529922,
                 6261968.60390423,  6215087.60607344],
               [ 6335702.70833714,  6311919.17016336,  6278581.57890056,
                 6235791.00048604,  6183672.04823372]])
        """

        pos = self.row_indices[:2]
        first_row = _linear_extrapolate(pos,
                                        (data[0, :], data[1, :]),
                                        self.hrow_indices[0])
        pos = self.row_indices[-2:]
        last_row = _linear_extrapolate(pos,
                                       (data[-2, :], data[-1, :]),
                                       self.hrow_indices[-1])

        return np.vstack((np.expand_dims(first_row, 0),
                          data,
                          np.expand_dims(last_row, 0))) 

    def _fill_row_borders(self):
        """Add the first and last rows to the data by extrapolation.

        >>> import numpy as np
        >>> lons = np.arange(20).reshape((4, 5), order="F")
        >>> lats = np.arange(20).reshape((4, 5), order="C")
        >>> lines = np.array([2, 7, 12, 17])
        >>> cols = np.array([2, 7, 12, 17, 22])
        >>> hlines = np.arange(20)
        >>> hcols = np.arange(24)
        >>> satint = SatelliteInterpolator((lons, lats), (lines, cols), (hlines, hcols))
        >>> satint._fill_row_borders()
        >>> satint.x__
        array([[ 6381081.08333225,  6371519.34066148,  6328950.00792935,
                 6253610.69157758,  6145946.19489936],
               [ 6370997.        ,  6354509.6014956 ,  6305151.62592155,
                 6223234.99818839,  6109277.14889072],
               [ 6345786.79166939,  6311985.2535809 ,  6245655.67090206,
                 6147295.76471541,  6017604.5338691 ],
               [ 6270385.16249031,  6219684.08214232,  6137100.75832981,
                 6023313.2794414 ,  5879194.72399075],
               [ 6145476.82098957,  6078546.55734877,  5980676.23854351,
                 5852716.72120069,  5695705.57359808],
               [ 6095513.48438928,  6022091.54743135,  5918106.430629  ,
                 5784478.09790441,  5622309.91344102]])
        >>> satint.row_indices
        array([ 0,  2,  7, 12, 17, 19])
        >>> satint = SatelliteInterpolator((lons, lats), (lines, cols), (hlines, hcols), chunk_size=10)
        >>> satint._fill_row_borders()
        >>> satint.x__
        array([[ 6381081.08333225,  6371519.34066148,  6328950.00792935,
                 6253610.69157758,  6145946.19489936],
               [ 6370997.        ,  6354509.6014956 ,  6305151.62592155,
                 6223234.99818839,  6109277.14889072],
               [ 6345786.79166939,  6311985.2535809 ,  6245655.67090206,
                 6147295.76471541,  6017604.5338691 ],
               [ 6335702.70833714,  6294975.51441502,  6221857.28889426,
                 6116920.07132621,  5980935.48786045],
               [ 6320348.4990906 ,  6276139.09205974,  6199670.56624433,
                 6091551.90273768,  5952590.38414781],
               [ 6270385.16249031,  6219684.08214232,  6137100.75832981,
                 6023313.2794414 ,  5879194.72399075],
               [ 6145476.82098957,  6078546.55734877,  5980676.23854351,
                 5852716.72120069,  5695705.57359808],
               [ 6095513.48438928,  6022091.54743135,  5918106.430629  ,
                 5784478.09790441,  5622309.91344102]])
         >>> satint.row_indices
         array([ 0,  2,  7,  9, 10, 12, 17, 19])
        """
        lines = len(self.hrow_indices)
        chunk_size = self.chunk_size or lines

        x__, y__, z__ = [], [], []
        row_indices = []
        for index in range(0, lines, chunk_size):
            ties = np.argwhere(np.logical_and(self.row_indices >= index,
                                              self.row_indices < index
                                              + chunk_size)).squeeze()
            tiepos = self.row_indices[np.logical_and(self.row_indices >= index,
                                                     self.row_indices < index
                                                     + chunk_size)].squeeze()
            x__.append(self._extrapolate_rows(self.x__[ties, :]))
            y__.append(self._extrapolate_rows(self.y__[ties, :]))
            z__.append(self._extrapolate_rows(self.z__[ties, :]))
            row_indices.append(np.array([self.hrow_indices[index]]))
            row_indices.append(tiepos)
            row_indices.append(np.array([self.hrow_indices[index
                                                           + chunk_size - 1]]))
        self.x__ = np.vstack(x__)
        self.y__ = np.vstack(y__)
        self.z__ = np.vstack(z__)

        self.row_indices = np.concatenate(row_indices)
    
    def _interp(self):
        """Interpolate the cartesian coordinates.
        """
        if np.all(self.hrow_indices == self.row_indices):
            return self._interp1d()
        
        xpoints, ypoints = np.meshgrid(self.hrow_indices,
                                       self.hcol_indices)

        spl = RectBivariateSpline(self.row_indices,
                                  self.col_indices,
                                  self.x__,
                                  s=0,
                                  kx=self.kx_,
                                  ky=self.ky_)

        self.newx = spl.ev(xpoints.ravel(), ypoints.ravel())
        self.newx = self.newx.reshape(xpoints.shape).T

        spl = RectBivariateSpline(self.row_indices,
                                  self.col_indices,
                                  self.y__,
                                  s=0,
                                  kx=self.kx_,
                                  ky=self.ky_)

        self.newy = spl.ev(xpoints.ravel(), ypoints.ravel())
        self.newy = self.newy.reshape(xpoints.shape).T

        spl = RectBivariateSpline(self.row_indices,
                                  self.col_indices,
                                  self.z__,
                                  s=0,
                                  kx=self.kx_,
                                  ky=self.ky_)

        self.newz = spl.ev(xpoints.ravel(), ypoints.ravel())
        self.newz = self.newz.reshape(xpoints.shape).T


    def _interp1d(self):
        """Interpolate in one dimension.
        """
        lines = len(self.hrow_indices)

        self.newx = np.empty((len(self.hrow_indices),
                              len(self.hcol_indices)),
                             self.x__.dtype)

        self.newy = np.empty((len(self.hrow_indices),
                              len(self.hcol_indices)),
                             self.y__.dtype)

        self.newz = np.empty((len(self.hrow_indices),
                              len(self.hcol_indices)),
                             self.z__.dtype)


        for cnt in range(lines):
            tck = splrep(self.col_indices, self.x__[cnt, :], k=self.ky_, s=0)
            self.newx[cnt, :] = splev(self.hcol_indices, tck, der=0)

            tck = splrep(self.col_indices, self.y__[cnt, :], k=self.ky_, s=0)
            self.newy[cnt, :] = splev(self.hcol_indices, tck, der=0)

            tck = splrep(self.col_indices, self.z__[cnt, :], k=self.ky_, s=0)
            self.newz[cnt, :] = splev(self.hcol_indices, tck, der=0)


    def interpolate(self):
        """Do the interpolation, and return resulting longitudes and latitudes.
        """
        self._interp()

        self.longitude = get_lons_from_cartesian(self.newx, self.newy)

        self.latitude = get_lats_from_cartesian(self.newx, self.newy, self.newz)

        return self.longitude, self.latitude

def get_lons_from_cartesian(x__, y__):
    """Get longitudes from cartesian coordinates.
    """
    return rad2deg(arccos(x__ / sqrt(x__ ** 2 + y__ ** 2))) * sign(y__)
    
def get_lats_from_cartesian(x__, y__, z__, thr=0.8):
    """Get latitudes from cartesian coordinates.
    """
    # if we are at low latitudes - small z, then get the
    # latitudes only from z. If we are at high latitudes (close to the poles)
    # then derive the latitude using x and y:

    lats = np.where(np.logical_and(np.less(z__, thr * EARTH_RADIUS), 
                                   np.greater(z__, -1. * thr * EARTH_RADIUS)),
                    90 - rad2deg(arccos(z__/EARTH_RADIUS)),
                    sign(z__) *
                    (90 - rad2deg(arcsin(sqrt(x__ ** 2 + y__ ** 2)
                                         / EARTH_RADIUS))))
    return lats

