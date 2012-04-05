#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010-2011.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Adam Dybbroe <adam.dybbroe@smhise>
#   Martin Raspaud <martin.raspaud@smhi.se>

from numpy import (meshgrid, mgrid, ceil,
                   arccos, sign, rad2deg, sqrt, fabs,
                   cos, sin, tan, arcsin, arctan)

EARTH_RADIUS = 6371000.0


class SatelliteInterpolator(object):
    """
    Handles interpolation of satellite geolocation data from a grid of tie points.
    It is preferable to have tie-points out till the edges if the satellite swath.
    The interpolation, is strictly an interpolation, so if the satellite swath goes beyond 
    the tie point grid this area outside the tie point grid will not be 
    interpolated/extrapolated.

    Uses numpy, Scipy, and pyresample

    >>> import numpy as np
    >>> coldim, rowdim = 2030, 1354
    >>> tiepoints = np.arange(0, coldim, 5), np.arange(0, rowdim, 5)
    >>> geodata = SatelliteInterpolator(tiepoints)
    # Get some longitudes and latitudes, e.g. from a HDF MODIS file...
    >>> geodata.lon_tiepoint = longitudes
    >>> geodata.lat_tiepoint = latitudes
    >>> geodata.interpolate()

    """

    def __init__(self, tiepoint_grid):
        self.row_indices = tiepoint_grid[0]
        self.col_indices = tiepoint_grid[1]
        self.lon_tiepoint = None
        self.lat_tiepoint = None
        self.longitude = None
        self.latitude = None

    def set_tiepoints(self, lon, lat):
        """Defines the lon,lat tie points"""
        self.lon_tiepoint = lon
        self.lat_tiepoint = lat

    def interpolate(self, method="linear"):
        """
        Upsample the tiepoint lonlat data to full resolution lonlat.
        Uses pyresample and Scipy to go from lon,lat space to x,y,z (cartesian) space,
        the interpolate in the 3D cartesian space, and then go back again to
        polar coordinates (lon,lat)

        """

        from pyresample import geometry
        from scipy.interpolate import griddata

        if self.lon_tiepoint is None or self.lat_tiepoint is None:
            raise ValueError('lon/lat values are not defined')

        swath = geometry.BaseDefinition(self.lon_tiepoint,
                                        self.lat_tiepoint)

        # Convert to cartesian coordinates
        
        xyz = swath.get_cartesian_coords()
        col_indices, row_indices = meshgrid(self.col_indices, self.row_indices)
        col_indices = col_indices.reshape(-1)
        row_indices = row_indices.reshape(-1)
        
        hcol_indices, hrow_indices = mgrid[ceil(self.col_indices[0]):
                                               self.col_indices[-1] + 1,
                                           ceil(self.row_indices[0]):
                                               self.row_indices[-1] + 1]
        orig_col_size = hcol_indices.shape[0]
        orig_row_size = hcol_indices.shape[1]

        hcol_indices = hcol_indices.reshape(-1)
        hrow_indices = hrow_indices.reshape(-1)

        # Interpolate x, y, and z        
        x_new = griddata((col_indices, row_indices),
                         xyz[:, :, 0].reshape(-1),
                         (hcol_indices, hrow_indices),
                         method=method)

        y_new = griddata((col_indices, row_indices),
                         xyz[:, :, 1].reshape(-1),
                         (hcol_indices, hrow_indices),
                         method=method)

        z_new = griddata((col_indices, row_indices),
                         xyz[:, :, 2].reshape(-1),
                         (hcol_indices, hrow_indices),
                         method=method)
        #orig_col_size = self.col_indices[-1] + 1
        #orig_row_size = self.row_indices[-1] + 1

        # Back to lon and lat
        self.longitude = get_lons_from_cartesian(x_new, y_new)
        self.longitude = self.longitude.reshape(orig_col_size, orig_row_size).transpose()
        #self.latitude = get_lats_from_cartesian(z_new)
        self.latitude = get_lats_from_cartesian(x_new, y_new, z_new)
        self.latitude = self.latitude.reshape(orig_col_size, orig_row_size).transpose()




    def interpolate_bivariate(self, output_grid=None):
        """
        Upsample the tiepoint lonlat data to full resolution lonlat.  Uses
        pyresample and Scipy to go from lon,lat space to x,y,z (cartesian)
        space, the interpolate in the 3D cartesian space, and then go back
        again to polar coordinates (lon,lat)
        """

        from pyresample import geometry
        from scipy.interpolate import SmoothBivariateSpline

        if self.lon_tiepoint is None or self.lat_tiepoint is None:
            raise ValueError('lon/lat values are not defined')

        swath = geometry.BaseDefinition(self.lon_tiepoint,
                                        self.lat_tiepoint)

        # Convert to cartesian coordinates
        xyz = swath.get_cartesian_coords()
        col_indices, row_indices = meshgrid(self.col_indices, self.row_indices)
        col_indices = col_indices.ravel()
        row_indices = row_indices.ravel()
        
        try:
            hcol_indices, hrow_indices = output_grid
        except TypeError:
            hcol_indices, hrow_indices = mgrid[ceil(self.col_indices[0]):
                                                   self.col_indices[-1] + 1,
                                               ceil(self.row_indices[0]):
                                                   self.row_indices[-1] + 1]

        orig_col_size = hcol_indices.shape[0]
        orig_row_size = hcol_indices.shape[1]

        hcol_indices = hcol_indices.ravel()
        hrow_indices = hrow_indices.ravel()

        # Interpolate x, y, and z 
        spl = SmoothBivariateSpline(col_indices, row_indices,
                                    xyz[:, :, 0].ravel(),
                                    kx=2, ky=1)
        x_new = spl.ev(hcol_indices, hrow_indices)

        spl = SmoothBivariateSpline(col_indices, row_indices,
                                    xyz[:, :, 1].ravel(),
                                    kx=2, ky=1)
        y_new = spl.ev(hcol_indices, hrow_indices)

        spl = SmoothBivariateSpline(col_indices, row_indices,
                                    xyz[:, :, 2].ravel(),
                                    kx=2, ky=1)
        z_new = spl.ev(hcol_indices, hrow_indices)



        # Back to lon and lat
        self.longitude = get_lons_from_cartesian(x_new, y_new)
        self.longitude = self.longitude.reshape(orig_col_size, orig_row_size).transpose()
        #self.latitude = get_lats_from_cartesian(z_new)
        self.latitude = get_lats_from_cartesian(x_new, y_new, z_new)
        self.latitude = self.latitude.reshape(orig_col_size, orig_row_size).transpose()



def get_lons_from_cartesian(x__, y__):
    """Get longitudes from cartesian coordinates"""
    #import geographic_tools
    #return geographic_tools.get_lons_from_cartesian(x__, y__)
    return rad2deg(arccos(x__/sqrt(x__**2 + y__**2)))*sign(y__)
    
def get_lats_from_cartesian(x__, y__, z__, thr=0.8):
    """Get latitudes from cartesian coordinates"""
    # if we are at low latitudes - small z, then get the
    # latitudes only from z. If we are at high latitudes (close to the poles)
    # then derive the latitude using x and y:

    #import geographic_tools
    #return geographic_tools.get_lats_from_cartesian(x__, y__, z__, thr)
    import numpy as np
    lats = np.where(np.logical_and(np.less(z__, thr * EARTH_RADIUS), 
                                   np.greater(z__, -1. * thr * EARTH_RADIUS)),
                    90 - rad2deg(arccos(z__/EARTH_RADIUS)),
                    sign(z__) * (90 - rad2deg(arcsin(sqrt(x__**2 + y__**2)/EARTH_RADIUS))))
    return lats
    #return sign(z__) * (90 - rad2deg(arcsin(sqrt(x__**2 + y__**2)/EARTH_RADIUS)))

#def get_lats_from_cartesian(z__):
#    """Get latitudes from cartesian coordinates"""
#    return 90 - rad2deg(arccos(z__/EARTH_RADIUS))


