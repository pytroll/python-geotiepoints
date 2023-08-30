# Copyright (c) 2018-2023 Python-geotiepoints developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
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

import warnings

from ._modis_interpolator import interpolate


def modis_1km_to_250m(lon1, lat1, satz1):
    """Interpolate MODIS geolocation from 1km to 250m resolution."""
    return interpolate(lon1, lat1, satz1,
                       coarse_resolution=1000,
                       fine_resolution=250)


def modis_1km_to_500m(lon1, lat1, satz1):
    """Interpolate MODIS geolocation from 1km to 500m resolution."""
    return interpolate(lon1, lat1, satz1,
                       coarse_resolution=1000,
                       fine_resolution=500)


def modis_5km_to_1km(lon1, lat1, satz1):
    """Interpolate MODIS geolocation from 5km to 1km resolution."""
    return interpolate(lon1, lat1, satz1,
                       coarse_resolution=5000,
                       fine_resolution=1000,
                       coarse_scan_width=lon1.shape[1])


def modis_5km_to_500m(lon1, lat1, satz1):
    """Interpolate MODIS geolocation from 5km to 500m resolution."""
    warnings.warn(
        "Interpolating 5km geolocation to 500m resolution " "may result in poor quality"
    )
    return interpolate(lon1, lat1, satz1,
                       coarse_resolution=5000,
                       fine_resolution=500,
                       coarse_scan_width=lon1.shape[1])


def modis_5km_to_250m(lon1, lat1, satz1):
    """Interpolate MODIS geolocation from 5km to 250m resolution."""
    warnings.warn(
        "Interpolating 5km geolocation to 250m resolution " "may result in poor quality"
    )
    return interpolate(lon1, lat1, satz1,
                       coarse_resolution=5000,
                       fine_resolution=250,
                       coarse_scan_width=lon1.shape[1])
