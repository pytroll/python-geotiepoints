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
