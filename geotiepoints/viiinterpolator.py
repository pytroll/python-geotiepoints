"""Interpolation of geographical tiepoints for the VII products.

It follows the description provided in document "EPS-SG VII Level 1B Product Format Specification".
Tiepoints are typically subsampled by a factor 8 with respect to the pixels, along and across the satellite track.
Because of the bowtie effect, tiepoints at the scan edges are not continuous between neighbouring scans,
therefore each scan has its own edge tiepoints in the along-track direction.
The interpolation is carried out per scan, so the tie points of one scan never affect the pixels of another.
Pixels that coincide with a tie point take its value, the others are linearly interpolated between
the two neighbouring tie points, first along then across the track.
The interpolation functions are implemented for xarray.DataArrays as input, backed by numpy or dask arrays.
Dask arrays are interpolated chunk by chunk, each chunk holding whole scans of tie points across the whole
width of the swath; they are rechunked that way when needed. The results do not depend on the chunk size.
This version works with vii test data V2 to be released Jan 2022 which has the data stored
in alt, act (row,col) format instead of act,alt (col,row)
"""

import dask.array as da
import numpy as np
import xarray as xr

# MEAN EARTH RADIUS AS DEFINED BY IUGG
MEAN_EARTH_RADIUS = 6371008.7714  # [m]


def tie_points_interpolation(data_on_tie_points, scan_alt_tie_points, tie_points_factor):
    """Interpolate the data from the tie points to the pixel points.

    The data are provided as a list of xarray DataArray objects, allowing to interpolate on several arrays
    at the same time; however the individual arrays must have exactly the same dimensions.

    Args:
        data_on_tie_points: list of xarray DataArray objects containing the values defined on the tie points.
        scan_alt_tie_points: number of tie points along the satellite track for each scan
        tie_points_factor: sub-sampling factor of tie points wrt pixel points

    Returns:
        list of xarray DataArray objects containing the interpolated values on the pixel points.
        They keep the dimension names, name and attributes of the tie point arrays.

    """
    n_tie_alt, n_tie_act = data_on_tie_points[0].shape
    dims = data_on_tie_points[0].dims
    _check_whole_scans(n_tie_alt, scan_alt_tie_points)

    data_on_pixel_points = []
    for data in data_on_tie_points:
        if data.shape != (n_tie_alt, n_tie_act) or data.dims != dims:
            raise ValueError("The dimensions of the arrays are not consistent")
        pixels = _map_scan_blocks(_interpolate_scans, [data.data], scan_alt_tie_points, tie_points_factor)
        data_on_pixel_points.append(_pixels_to_data_array(pixels, data))
    return data_on_pixel_points


def tie_points_geo_interpolation(longitude, latitude,
                                 scan_alt_tie_points, tie_points_factor,
                                 lat_threshold_use_cartesian=60.,
                                 z_threshold_use_xy=0.8):
    """Interpolate the geographical position from the tie points to the pixel points.

    The interpolation is done on the longitude and latitude values, or on cartesian coordinates if the
    tie points reach latitudes above ``lat_threshold_use_cartesian`` or span more than 180 degrees of
    longitude. The choice is made from all the tie points, so it is the same for every dask chunk.
    For dask arrays it is made when the result is computed, so nothing is computed by this function.

    Args:
        longitude: xarray DataArray containing the longitude values defined on the tie points (degrees).
        latitude: xarray DataArray containing the latitude values defined on the tie points (degrees).
        scan_alt_tie_points: number of tie points along the satellite track for each scan.
        tie_points_factor: sub-sampling factor of tie points wrt pixel points.
        lat_threshold_use_cartesian: latitude threshold to use cartesian coordinates.
        z_threshold_use_xy: z threshold to compute latitude from x and y in cartesian coordinates.

    Returns:
        two xarray DataArray objects containing the interpolated longitude and latitude values on the pixel points.
        They keep the dimension names, name and attributes of the tie point arrays.

    """
    # Check that the two arrays have the same dimensions
    if longitude.shape != latitude.shape:
        raise ValueError("The dimensions of longitude and latitude don't match")
    _check_whole_scans(longitude.shape[0], scan_alt_tie_points)

    # Lazy for dask arrays: computed once with the result and given to every chunk
    max_abs_lat = abs(latitude).max().data
    lon_range = (longitude.max() - longitude.min()).data
    interp_lonlat = _map_scan_blocks(_geo_interpolate_scans, [longitude.data, latitude.data],
                                     scan_alt_tie_points, tie_points_factor,
                                     reductions=(max_abs_lat, lon_range), n_outputs=2,
                                     lat_threshold_use_cartesian=lat_threshold_use_cartesian,
                                     z_threshold_use_xy=z_threshold_use_xy)
    return _pixels_to_data_array(interp_lonlat[0], longitude), _pixels_to_data_array(interp_lonlat[1], latitude)


def _check_whole_scans(n_tie_alt, scan_alt_tie_points):
    """Check that the number of tie points along track is a multiple of the number of tie points per scan."""
    if n_tie_alt % scan_alt_tie_points != 0:
        raise ValueError("The number of tie points in the along-route dimension must be a multiple of "
                         f"{scan_alt_tie_points}")


def _use_cartesian(max_abs_lat, lon_range, lat_threshold_use_cartesian):
    """Check if the geographical interpolation must be done on cartesian coordinates.

    That is the case at high latitudes, or when the longitudes cross the antimeridian.

    Args:
        max_abs_lat: largest absolute latitude of all the tie points (degrees), NaN if there are none.
        lon_range: difference between the largest and smallest longitudes of all the tie points (degrees).
        lat_threshold_use_cartesian: latitude threshold to use cartesian coordinates.

    """
    return bool(max_abs_lat > lat_threshold_use_cartesian or lon_range > 180.)


def _map_scan_blocks(func, tie_arrays, scan_alt_tie_points, tie_points_factor, reductions=(), n_outputs=None,
                     **kwargs):
    """Call ``func`` on whole scans of tie points, chunk by chunk for dask arrays.

    Args:
        func: function taking the numpy tie point arrays, the ``reductions``, ``scan_alt_tie_points`` and
            ``tie_points_factor``, and returning the pixel array, or ``n_outputs`` pixel arrays stacked along a
            new first dimension.
        tie_arrays: numpy or dask arrays of the tie points, all with the same shape.
        scan_alt_tie_points: number of tie points along the satellite track for each scan.
        tie_points_factor: sub-sampling factor of tie points wrt pixel points.
        reductions: numpy or dask 0-d arrays computed from all the tie points. Dask ones are computed once and
            the same values are given to every chunk.
        n_outputs: number of stacked pixel arrays returned by ``func``, or None if it returns only one.
        kwargs: other keyword arguments passed to ``func``.

    Returns:
        numpy or dask array of the pixel values, stacked along the first dimension if ``n_outputs`` is given.

    """
    if not any(isinstance(arr, da.Array) for arr in tie_arrays):
        return func(*tie_arrays, *reductions, scan_alt_tie_points, tie_points_factor, **kwargs)

    tie_arrays = _rechunk_to_whole_scans([da.asarray(arr) for arr in tie_arrays], scan_alt_tie_points)
    row_chunks, col_chunks = tie_arrays[0].chunks
    pixel_chunks = (
        tuple(rows // scan_alt_tie_points * (scan_alt_tie_points - 1) * tie_points_factor for rows in row_chunks),
        ((col_chunks[0] - 1) * tie_points_factor,),
    )
    dtype = _pixel_dtype(*tie_arrays)
    if n_outputs is None:
        map_kwargs = {"chunks": pixel_chunks}
    else:
        map_kwargs = {"chunks": ((n_outputs,),) + pixel_chunks, "new_axis": 0}
    return da.map_blocks(func, *tie_arrays, *reductions, scan_alt_tie_points, tie_points_factor, **kwargs,
                         dtype=dtype, meta=np.array((), dtype=dtype), **map_kwargs)


def _rechunk_to_whole_scans(tie_arrays, scan_alt_tie_points):
    """Rechunk dask tie point arrays to whole scans along the track and the whole swath width across it.

    Arrays already chunked this way, like the ones of Satpy's METimage readers, are returned unchanged.

    """
    row_chunks, col_chunks = tie_arrays[0].chunks
    whole_scans = all(rows % scan_alt_tie_points == 0 for rows in row_chunks) and len(col_chunks) == 1
    if whole_scans and all(arr.chunks == tie_arrays[0].chunks for arr in tie_arrays[1:]):
        return tie_arrays
    rows = max(row_chunks[0] // scan_alt_tie_points, 1) * scan_alt_tie_points
    return [arr.rechunk((rows, -1)) for arr in tie_arrays]


def _pixel_dtype(*tie_arrays):
    """Get the data type of the pixels: floating point tie points keep theirs, integers become float64."""
    return np.result_type(*(arr.dtype for arr in tie_arrays), 1.0)


def _pixels_to_data_array(pixels, tie_points):
    """Wrap pixel values in a DataArray like the tie points, keeping only the coordinates not on their dimensions."""
    coords = {name: coord for name, coord in tie_points.coords.items() if not set(coord.dims) & set(tie_points.dims)}
    return xr.DataArray(pixels, dims=tie_points.dims, coords=coords, name=tie_points.name, attrs=tie_points.attrs)


def _interpolate_scans(tie_points, scan_alt_tie_points, tie_points_factor, out=None):
    """Interpolate whole scans of tie points to pixel points, first along then across the track.

    Pixels that coincide with a tie point get its value. The others are linearly interpolated between
    the two neighbouring tie points of the same scan. Taking the tie point value, instead of giving a
    zero weight to a neighbouring tie point, keeps an invalid (NaN) neighbour from invalidating the pixel.

    Args:
        tie_points: numpy array of whole scans (rows) of tie points over the whole swath width (columns).
        scan_alt_tie_points: number of tie points along the satellite track for each scan.
        tie_points_factor: sub-sampling factor of tie points wrt pixel points.
        out: C-contiguous numpy array to write the pixel values to, or None to create a new one.

    Returns:
        C-contiguous numpy array of the pixel values.

    """
    dtype = _pixel_dtype(tie_points)
    n_tie_alt, n_tie_act = tie_points.shape
    n_scans = n_tie_alt // scan_alt_tie_points
    # Position of the pixels between two tie points, excluding the one on the first tie point
    weights = np.arange(1, tie_points_factor, dtype=dtype) / tie_points_factor

    # Along the track, only between the tie points of the same scan:
    # (scans, tie point intervals per scan, pixels per interval, tie point columns)
    scans = tie_points.astype(dtype, copy=False).reshape(n_scans, scan_alt_tie_points, 1, n_tie_act)
    along = np.empty((n_scans, scan_alt_tie_points - 1, tie_points_factor, n_tie_act), dtype=dtype)
    _interpolate_intervals(scans[:, :-1], scans[:, 1:], weights[:, np.newaxis], along)
    along = along.reshape(-1, n_tie_act)

    # Across the track: (pixel rows, tie point intervals, pixels per interval)
    if out is None:
        out = np.empty((along.shape[0], (n_tie_act - 1) * tie_points_factor), dtype=dtype)
    across = out.reshape(along.shape[0], n_tie_act - 1, tie_points_factor)
    _interpolate_intervals(along[:, :-1, np.newaxis], along[:, 1:, np.newaxis], weights, across)
    return out


def _interpolate_intervals(start, end, weights, out):
    """Linearly interpolate tie point intervals into ``out``, whose third dimension holds the pixels of an interval.

    The first pixel of each interval gets the value of the start tie point, the others are interpolated
    using ``weights``. The values are written in place to avoid temporary arrays as large as ``out``.

    """
    out[:, :, :1] = start
    interpolated = out[:, :, 1:]
    np.multiply(end - start, weights, out=interpolated)
    interpolated += start


def _geo_interpolate_scans(longitude, latitude, max_abs_lat, lon_range, scan_alt_tie_points, tie_points_factor,
                           lat_threshold_use_cartesian, z_threshold_use_xy):
    """Interpolate whole scans of longitude and latitude tie points to pixel points.

    Args:
        longitude: numpy array of longitude tie points (degrees).
        latitude: numpy array of latitude tie points (degrees).
        max_abs_lat: largest absolute latitude of all the tie points, not only these ones (degrees).
        lon_range: longitude range of all the tie points, not only these ones (degrees).
        scan_alt_tie_points: number of tie points along the satellite track for each scan.
        tie_points_factor: sub-sampling factor of tie points wrt pixel points.
        lat_threshold_use_cartesian: latitude threshold to use cartesian coordinates.
        z_threshold_use_xy: z threshold to compute latitude from x and y in cartesian coordinates.

    Returns:
        numpy array of the pixel longitudes and latitudes, stacked along the first dimension.

    """
    n_tie_alt, n_tie_act = longitude.shape
    shape = (2, n_tie_alt // scan_alt_tie_points * (scan_alt_tie_points - 1) * tie_points_factor,
             (n_tie_act - 1) * tie_points_factor)
    lonlat = np.empty(shape, dtype=_pixel_dtype(longitude, latitude))
    if _use_cartesian(max_abs_lat, lon_range, lat_threshold_use_cartesian):
        x_coords, y_coords, z_coords = (_interpolate_scans(coords, scan_alt_tie_points, tie_points_factor)
                                        for coords in _lonlat2xyz(longitude, latitude))
        _xyz2lonlat(x_coords, y_coords, z_coords, z_threshold_use_xy, out=lonlat)
    else:
        _interpolate_scans(longitude, scan_alt_tie_points, tie_points_factor, out=lonlat[0])
        _interpolate_scans(latitude, scan_alt_tie_points, tie_points_factor, out=lonlat[1])
    return lonlat


def _lonlat2xyz(lons, lats):
    """Convert longitudes and latitudes to cartesian coordinates.

    Args:
        lons: array containing the longitude values in degrees.
        lats: array containing the latitude values in degrees.

    Returns:
        tuple of arrays containing the x, y, and z values in meters.

    """
    lons_rad = np.deg2rad(lons)
    lats_rad = np.deg2rad(lats)
    x_coords = MEAN_EARTH_RADIUS * np.cos(lats_rad) * np.cos(lons_rad)
    y_coords = MEAN_EARTH_RADIUS * np.cos(lats_rad) * np.sin(lons_rad)
    z_coords = MEAN_EARTH_RADIUS * np.sin(lats_rad)
    return x_coords, y_coords, z_coords


def _xyz2lonlat(x_coords, y_coords, z_coords, z_threshold_use_xy, out):
    """Get longitudes and latitudes from cartesian coordinates.

    The computations are done in place as much as possible to limit the number of temporary pixel arrays.

    Args:
        x_coords: numpy array containing the x values in meters.
        y_coords: numpy array containing the y values in meters.
        z_coords: numpy array containing the z values in meters.
        z_threshold_use_xy: z threshold to compute latitude from x and y in cartesian coordinates.
        out: numpy array to write the longitude and latitude values in degrees to, stacked along the first dimension.

    """
    lons, lats = out
    r = np.sqrt(x_coords ** 2 + y_coords ** 2)
    np.divide(x_coords, r, out=lons)
    np.arccos(lons, out=lons)
    np.rad2deg(lons, out=lons)
    lons *= np.sign(y_coords)
    # Compute latitude from z at low z and from x and y at high z
    np.divide(z_coords, MEAN_EARTH_RADIUS, out=lats)
    np.arccos(lats, out=lats)
    np.rad2deg(lats, out=lats)
    np.subtract(90., lats, out=lats)
    thr_z = z_threshold_use_xy * MEAN_EARTH_RADIUS
    use_xy = ~np.logical_and(np.less(z_coords, thr_z), np.greater(z_coords, -thr_z))
    if use_xy.any():
        lats[use_xy] = np.sign(z_coords[use_xy]) * (90. - np.rad2deg(np.arcsin(r[use_xy] / MEAN_EARTH_RADIUS)))
