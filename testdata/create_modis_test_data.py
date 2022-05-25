#!/usr/bin/env python
"""Generate MODIS interpolation test data from real input data.

This script is used to generate the "testdata/modis_test_data.h5" file that is
used to validate the various modis interpolation algorithms in
python-geotiepoints. The test data file consists of 1km "truth" longitude and
latitude arrays from an input MOD03 HDF4 file and interpolated longitude and
latitude arrays at 500m and 250m resolution. The interpolation is done using
the CVIIRS based algorithm in ``geotiepoints/modisinterpolator.py``.
The CVIIRS algorithm was used as opposed to the "simple" or other interpolation
methods due to the smoother interpolation between pixels (no linear "steps").

MOD03 files geolocation data is terrain corrected. This means that the
interpolation methods currently in python-geotiepoints can't produce an
exact matching result for a round trip test of 1km (truth) ->
5km (every 5th pixel) -> 1km (interpolation result).
The input MOD03 test data was chosen due to its lack of varying terrain
(almost entirely ocean view) to minimize error/differences between the
1km truth and 1km interpolation results.

To limit size of the test data file and the reduce the execution time of tests
the test data is limited to the last 2 scans (20 rows of 1km data) of the
provided input data.

"""
import os
import sys
from datetime import datetime

import h5py
import numpy as np
from pyhdf.SD import SD, SDC
import xarray as xr
import dask.array as da

from geotiepoints.modisinterpolator import modis_1km_to_500m, modis_1km_to_250m


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="Input MOD03 geolocation HDF4 filename to read 1km lon/lat data from.")
    parser.add_argument("-o", "--output", default="modis_test_data.h5",
                        help="Output test data HDF5 filename being created")
    args = parser.parse_args()

    num_1km_rows = 20
    lons_1km, lats_1km, satz_1km = _get_1km_lon_lat_satz_from_mod03(args.input)
    lons_1km = lons_1km[-num_1km_rows:]
    lats_1km = lats_1km[-num_1km_rows:]
    satz_1km = satz_1km[-num_1km_rows:]
    lons_1km = xr.DataArray(da.from_array(lons_1km), dims=("y", "x"))
    lats_1km = xr.DataArray(da.from_array(lats_1km), dims=("y", "x"))
    satz_1km = xr.DataArray(da.from_array(satz_1km), dims=("y", "x"))

    with h5py.File(args.output, "w") as output_h:
        lons_500m, lats_500m = modis_1km_to_500m(lons_1km, lats_1km, satz_1km)
        lons_500m = lons_500m.astype(np.float32, copy=False)
        lats_500m = lats_500m.astype(np.float32, copy=False)

        lons_250m, lats_250m = modis_1km_to_250m(lons_1km, lats_1km, satz_1km)
        lons_250m = lons_250m.astype(np.float32, copy=False)
        lats_250m = lats_250m.astype(np.float32, copy=False)

        output_h.create_dataset("lon_1km", data=lons_1km, compression="gzip", compression_opts=9)
        output_h.create_dataset("lat_1km", data=lats_1km, compression="gzip", compression_opts=9)
        output_h.create_dataset("satz_1km", data=satz_1km, compression="gzip", compression_opts=9)
        output_h.create_dataset("lon_500m", data=lons_500m, compression="gzip", compression_opts=9)
        output_h.create_dataset('lat_500m', data=lats_500m, compression="gzip", compression_opts=9)
        output_h.create_dataset("lon_250m", data=lons_250m, compression="gzip", compression_opts=9)
        output_h.create_dataset("lat_250m", data=lats_250m, compression="gzip", compression_opts=9)
        output_h.attrs["1km_data_origin"] = os.path.basename(args.input)
        output_h.attrs["description"] = (
            "MODIS interpolation test data for the python-geotiepoints package. "
            "The 1 km data is taken directly from a MOD03 file. The 250m and "
            "500m is generated using the cviirs-based algorithm in "
            "`geotiepoints/modisinterpolator.py`. For more information see "
            "the generation script in `testdata/create_modis_test_data.py` in "
            "the python-geotiepoints git repository."
        )
        output_h.attrs["creation_date"] = datetime.utcnow().strftime("%Y-%m-%d")


def _get_1km_lon_lat_satz_from_mod03(hdf4_filename: str) -> tuple:
    h = SD(hdf4_filename, mode=SDC.READ)
    lon_var = h.select("Longitude")
    lat_var = h.select("Latitude")
    sat_zen_var = h.select("SensorZenith")

    # ensure 32-bit float
    lon_data = lon_var[:].astype(np.float32, copy=False)
    lat_data = lat_var[:].astype(np.float32, copy=False)
    sat_zen_attrs = sat_zen_var.attributes()
    scale_factor = sat_zen_attrs.get("scale_factor", 1.0)
    add_offset = sat_zen_attrs.get("add_offset", 0.0)
    sat_zen_data = (sat_zen_var[:] * scale_factor + add_offset).astype(np.float32, copy=False)

    return lon_data, lat_data, sat_zen_data


if __name__ == "__main__":
    sys.exit(main())
