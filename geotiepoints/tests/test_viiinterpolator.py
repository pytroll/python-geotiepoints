"""Test of the interpolation of geographical tiepoints for the VII products.

It follows the description provided in document "EPS-SG VII Level 1B Product Format Specification V4A".
This version is compatible for vii (METimage) test data version V2 (Jan 2022). It is not back compatible
with V1.

"""

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from geotiepoints.viiinterpolator import tie_points_interpolation, tie_points_geo_interpolation

from .utils import CustomScheduler


TEST_N_SCANS = 2
TEST_TIE_POINTS_FACTOR = 2
TEST_SCAN_ALT_TIE_POINTS = 3
TEST_VALID_ALT_TIE_POINTS = TEST_SCAN_ALT_TIE_POINTS * TEST_N_SCANS
TEST_INVALID_ALT_TIE_POINTS = TEST_SCAN_ALT_TIE_POINTS * TEST_N_SCANS + 1
TEST_ACT_TIE_POINTS = 4
TEST_SCAN_ALT_PIXELS = (TEST_SCAN_ALT_TIE_POINTS - 1) * TEST_TIE_POINTS_FACTOR
TEST_ACT_PIXELS = (TEST_ACT_TIE_POINTS - 1) * TEST_TIE_POINTS_FACTOR

# Results of latitude/longitude interpolation with simple interpolation on coordinates
TEST_LON_1 = np.array(
    [[-12., -11.5, -11., -10.5, -10.0, -9.5],
     [-10., -9.5, -9., -8.5, -8.0, -7.5],
     [-8., -7.5, -7., -6.5, -6.0, -5.5],
     [-6., -5.5, -5., -4.5, -4.0, -3.5],
     [0., 0.5, 1., 1.5, 2.0, 2.5],
     [2., 2.5, 3., 3.5, 4.0, 4.5],
     [4., 4.5, 5., 5.5, 6.0, 6.5],
     [6., 6.5, 7., 7.5, 8.0, 8.5]]
)
TEST_LAT_1 = np.array(
    [[0., 0.5, 1., 1.5, 2., 2.5],
     [2., 2.5, 3., 3.5, 4., 4.5],
     [4., 4.5, 5., 5.5, 6., 6.5],
     [6., 6.5, 7., 7.5, 8., 8.5],
     [12., 12.5, 13., 13.5, 14.0, 14.5],
     [14., 14.5, 15., 15.5, 16., 16.5],
     [16., 16.5, 17., 17.5, 18., 18.5],
     [18., 18.5, 19., 19.5, 20., 20.5]]
)

# Results of latitude/longitude interpolation on cartesian coordinates (longitude with a 360 degrees step)
TEST_LON_2 = np.array(
    [[-12., -11.50003808, -11., -10.50011426, -10., -9.50019052],
     [-10.00243991, -9.5032411, -9.00366173, -8.50454031, -8.00488578, -7.5058423],
     [-8., -7.50034342, -7., -6.50042016, -6., -5.50049716],
     [-6.00734362, -5.50845783, -5.00857895, -4.50977302, -4.00981958, -3.51109426],
     [0., 0.49903263, 1., 1.49895241, 2., 2.49887151],
     [1.98257947, 2.48080192, 2.98127841, 3.47941324, 3.97996512, 4.47801105],
     [4., 4.49870746, 5., 5.49862418, 6., 6.49853998],
     [5.97729789, 6.47516183, 6.97594186, 7.47371256, 7.97456943, 8.47224525]]
)

TEST_LAT_2 = np.array(
    [[0., 0.49998096, 1., 1.49994287, 2., 2.49990475],
     [1.99878116, 2.49838091, 2.99817081, 3.49773189, 3.99755935, 4.49708148],
     [4., 4.4998283, 5., 5.49978993, 6., 6.49975143],
     [5.99633155, 6.4957749, 6.99571447, 7.49511791, 7.99509473, 8.49445789],
     [12., 12.49951634, 13., 13.49947623, 14., 14.499435779],
     [13.99129786, 14.4904098, 14.99064796, 15.48971613, 15.98999196, 16.48901572],
     [16., 16.49935377, 17., 17.49931213, 18., 18.49927003],
     [17.98865968, 18.48759253, 18.98798235, 19.48686863, 19.98729684, 20.48613573]]
)

# Results of latitude/longitude interpolation on cartesian coordinates (latitude above 60 degrees)
TEST_LON_3 = np.array(
    [[-12., -11.50444038, -11., -10.50459822, -10., -9.50476197],
     [-10.07492627, -9.58101155, -9.07759836, -8.5839056, - 8.0803761, -7.58691614],
     [-8., -7.50510905, -7., -6.5052934, -6., -5.50548573],
     [-6.0862821, -5.59332416, -5.08942935, -4.59674283, -4.09272043, -3.60032066],
     [0., 0.49315061, 1., 1.49287934, 2., 2.49259217],
     [1.88371709, 2.3739768, 2.87898193, 3.36879304, 3.87395153, 4.36327924],
     [4., 4.49196335, 5., 5.49161771, 6., 6.49124808],
     [5.86287282, 6.35111105, 6.85674573, 7.3443667, 7.85016382, 8.33710998]]
)

TEST_LAT_3 = np.array(
    [[45., 45.49777998, 46., 46.49770107, 47., 47.4976192],
     [46.96258417, 47.4595462, 47.9612508, 48.4581022, 48.95986481, 49.45660021],
     [49., 49.49744569, 50., 50.49735352, 51., 51.49725738],
     [50.95691833, 51.45340364, 51.95534841, 52.45169855, 52.95370691, 53.55477452],
     [57., 57.50277937, 58., 58.50267347, 59., 59.50256981],
     [59.04185095, 59.54357898, 60.04020844, 60.54185139, 61.03859839, 61.54015723],
     [61., 61.5023687, 62., 62.502271, 63., 63.50217506],
     [63.03546804, 63.53686131, 64.03394419, 64.53525587, 65.03244567, 65.53367647]]
)


@pytest.fixture(params=[False, True], ids=["numpy", "dask"])
def use_dask(request):
    """Run a test with numpy-backed and with dask-backed tie points."""
    return request.param


def _tie_points_data_array(data, use_dask):
    """Wrap tie point values in a DataArray with the VII tie point dimensions.

    Dask arrays are chunked along the track in whole scans like Satpy's METimage readers do.

    """
    if use_dask:
        data = da.from_array(data, chunks=(TEST_SCAN_ALT_TIE_POINTS, -1))
    return xr.DataArray(data, dims=('num_tie_points_alt', 'num_tie_points_act'))


def _arange_tie_points(n_tie_alt, use_dask):
    """Create tie points counting up from 0 with ``n_tie_alt`` points along the track."""
    data = np.arange(n_tie_alt * TEST_ACT_TIE_POINTS, dtype=np.float64).reshape(n_tie_alt, TEST_ACT_TIE_POINTS)
    return _tie_points_data_array(data, use_dask)


def _linspace_tie_points(start, stop, n_tie_alt=TEST_VALID_ALT_TIE_POINTS):
    """Create evenly spaced tie point values from ``start`` to ``stop`` with ``n_tie_alt`` points along the track."""
    data = np.linspace(start, stop, num=n_tie_alt * TEST_ACT_TIE_POINTS, dtype=np.float64)
    return data.reshape(n_tie_alt, TEST_ACT_TIE_POINTS)


def _assert_pixel_array(data_arr, use_dask):
    """Check the array type, chunks, and memory layout of interpolated data.

    Satpy's METimage readers expect one chunk of pixel rows per chunk of tie point rows, each
    spanning the whole width of the swath. Consumers like pyresample's EWA resampling require
    C-contiguous arrays. Dask blocks are checked individually as computing the whole array would
    concatenate them into a new C-contiguous array and hide the problem.

    """
    if use_dask:
        assert isinstance(data_arr.data, da.Array)
        assert data_arr.chunks == ((TEST_SCAN_ALT_PIXELS,) * TEST_N_SCANS, (TEST_ACT_PIXELS,))
        blocks = dask.compute(*data_arr.data.to_delayed().ravel())
    else:
        assert isinstance(data_arr.data, np.ndarray)
        blocks = [data_arr.data]
    assert all(block.flags.c_contiguous for block in blocks)


def test_tie_points_interpolation(use_dask):
    """Test the interpolation routine with valid input."""
    data = _arange_tie_points(TEST_VALID_ALT_TIE_POINTS, use_dask)
    with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
        result = tie_points_interpolation([data], TEST_SCAN_ALT_TIE_POINTS, TEST_TIE_POINTS_FACTOR)[0]

    _assert_pixel_array(result, use_dask)
    # Across the track
    np.testing.assert_allclose(result[0, :], [0., 0.5, 1., 1.5, 2., 2.5])
    # Along the track
    np.testing.assert_allclose(result[:, 0], [0., 2., 4., 6., 12., 14., 16., 18.])


def test_tie_points_interpolation_invalid_alt_tie_points(use_dask):
    """Test that the number of tie points along the track must be a multiple of the tie points per scan."""
    data = _arange_tie_points(TEST_INVALID_ALT_TIE_POINTS, use_dask)
    with pytest.raises(ValueError, match="must be a multiple"):
        tie_points_interpolation([data], TEST_SCAN_ALT_TIE_POINTS, TEST_TIE_POINTS_FACTOR)


@pytest.mark.parametrize(
    ("longitude", "latitude", "exp_lon", "exp_lat"),
    [
        pytest.param(_linspace_tie_points(-12, 11), _linspace_tie_points(0, 23), TEST_LON_1, TEST_LAT_1,
                     id="lonlat"),
        pytest.param(_linspace_tie_points(-12, 11) % 360., _linspace_tie_points(0, 23), TEST_LON_2, TEST_LAT_2,
                     id="cartesian_lon_360_step"),
        pytest.param(_linspace_tie_points(-12, 11), _linspace_tie_points(45, 68), TEST_LON_3, TEST_LAT_3,
                     id="cartesian_lat_over_60"),
    ],
)
def test_tie_points_geo_interpolation(longitude, latitude, exp_lon, exp_lat, use_dask):
    """Test the coordinates interpolation routine in geodetic and cartesian coordinates."""
    with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
        lon, lat = tie_points_geo_interpolation(
            _tie_points_data_array(longitude, use_dask),
            _tie_points_data_array(latitude, use_dask),
            TEST_SCAN_ALT_TIE_POINTS,
            TEST_TIE_POINTS_FACTOR
        )

    _assert_pixel_array(lon, use_dask)
    _assert_pixel_array(lat, use_dask)
    np.testing.assert_allclose(lon, exp_lon)
    np.testing.assert_allclose(lat, exp_lat)


def test_tie_points_geo_interpolation_mismatched_shapes(use_dask):
    """Test that longitude and latitude must have the same shape."""
    longitude = _tie_points_data_array(_linspace_tie_points(-12, 11), use_dask)
    latitude = _arange_tie_points(TEST_INVALID_ALT_TIE_POINTS, use_dask)
    with pytest.raises(ValueError, match="don't match"):
        tie_points_geo_interpolation(longitude, latitude, TEST_SCAN_ALT_TIE_POINTS, TEST_TIE_POINTS_FACTOR)


@pytest.mark.parametrize(
    ("nan_tie_rows", "nan_tie_cols", "exp_nan_rows", "exp_nan_cols"),
    [
        pytest.param([0, 1, 2], [], [0, 1, 2, 3], [], id="missing_first_scan"),
        pytest.param([2], [], [3], [], id="edge_tie_row_of_first_scan"),
        pytest.param([1], [], [1, 2, 3], [], id="middle_tie_row_of_first_scan"),
        pytest.param([3], [], [4, 5], [], id="first_tie_row_of_second_scan"),
        pytest.param([], [1], [], [1, 2, 3], id="tie_column"),
    ],
)
def test_tie_points_interpolation_invalid_tie_points(nan_tie_rows, nan_tie_cols, exp_nan_rows, exp_nan_cols,
                                                     use_dask):
    """Test that invalid (NaN) tie points only invalidate the pixels interpolated from them.

    Pixels of other scans and pixels coinciding with a valid tie point stay valid.

    """
    data = np.arange(TEST_VALID_ALT_TIE_POINTS * TEST_ACT_TIE_POINTS, dtype=np.float64)
    data = data.reshape(TEST_VALID_ALT_TIE_POINTS, TEST_ACT_TIE_POINTS)
    data[nan_tie_rows, :] = np.nan
    data[:, nan_tie_cols] = np.nan
    result = tie_points_interpolation([_tie_points_data_array(data, use_dask)],
                                      TEST_SCAN_ALT_TIE_POINTS, TEST_TIE_POINTS_FACTOR)[0]

    exp_nan = np.zeros(result.shape, dtype=bool)
    exp_nan[exp_nan_rows, :] = True
    exp_nan[:, exp_nan_cols] = True
    np.testing.assert_array_equal(np.isnan(result.values), exp_nan)


def test_tie_points_interpolation_keeps_tie_point_values(use_dask):
    """Test that pixels coinciding with a tie point get exactly its value."""
    data = np.random.default_rng(42).uniform(-180, 180, (TEST_VALID_ALT_TIE_POINTS, TEST_ACT_TIE_POINTS))
    result = tie_points_interpolation([_tie_points_data_array(data, use_dask)],
                                      TEST_SCAN_ALT_TIE_POINTS, TEST_TIE_POINTS_FACTOR)[0]

    # Every tie point but the edge ones, at the end of each scan and of the swath width, coincides with a pixel
    tie_rows = [scan * TEST_SCAN_ALT_TIE_POINTS + row
                for scan in range(TEST_N_SCANS) for row in range(TEST_SCAN_ALT_TIE_POINTS - 1)]
    pixel_rows = [scan * TEST_SCAN_ALT_PIXELS + row * TEST_TIE_POINTS_FACTOR
                  for scan in range(TEST_N_SCANS) for row in range(TEST_SCAN_ALT_TIE_POINTS - 1)]
    np.testing.assert_array_equal(result.values[pixel_rows, ::TEST_TIE_POINTS_FACTOR], data[tie_rows, :-1])


@pytest.mark.parametrize(
    ("dtype", "exp_dtype"),
    [(np.float64, np.float64), (np.float32, np.float32), (np.int16, np.float64)],
)
def test_tie_points_interpolation_dtype_and_metadata(dtype, exp_dtype, use_dask):
    """Test that floating point tie points keep their type, integers become float64, and metadata is kept."""
    data = _arange_tie_points(TEST_VALID_ALT_TIE_POINTS, use_dask).astype(dtype)
    data = data.rename("tie_data").assign_attrs(units="1").assign_coords(scalar=1.0)
    result = tie_points_interpolation([data], TEST_SCAN_ALT_TIE_POINTS, TEST_TIE_POINTS_FACTOR)[0]

    assert result.dtype == exp_dtype
    assert result.values.dtype == exp_dtype
    assert result.name == "tie_data"
    assert result.attrs == {"units": "1"}
    assert list(result.coords) == ["scalar"]
    assert result.dims == data.dims


@pytest.mark.parametrize("lat_range", [(0, 23), (45, 68)], ids=["lonlat", "cartesian"])
def test_tie_points_geo_interpolation_float32(lat_range, use_dask):
    """Test that 32-bit floating point longitudes and latitudes stay 32-bit."""
    longitude = _tie_points_data_array(_linspace_tie_points(-12, 11).astype(np.float32), use_dask)
    latitude = _tie_points_data_array(_linspace_tie_points(*lat_range).astype(np.float32), use_dask)
    lon, lat = tie_points_geo_interpolation(longitude, latitude, TEST_SCAN_ALT_TIE_POINTS, TEST_TIE_POINTS_FACTOR)

    for data_arr in (lon, lat):
        assert data_arr.dtype == np.float32
        assert data_arr.values.dtype == np.float32


TEST_CHUNKS = [
    pytest.param((TEST_SCAN_ALT_TIE_POINTS, -1), id="one_scan"),
    pytest.param((2 * TEST_SCAN_ALT_TIE_POINTS, -1), id="two_scans"),
    pytest.param((-1, -1), id="single_chunk"),
    pytest.param((3 * TEST_SCAN_ALT_TIE_POINTS, -1), id="uneven_scans"),
    pytest.param((TEST_SCAN_ALT_TIE_POINTS - 1, -1), id="partial_scans"),
    pytest.param((2 * TEST_SCAN_ALT_TIE_POINTS, TEST_ACT_TIE_POINTS // 2), id="partial_width"),
]
TEST_N_SCANS_CHUNKED = 4


def _assert_same_as_numpy(result, expected):
    """Check that dask results are made of whole scan chunks across the swath width and equal the numpy ones."""
    assert isinstance(result.data, da.Array)
    assert all(rows % TEST_SCAN_ALT_PIXELS == 0 for rows in result.chunks[0])
    assert result.chunks[1] == (result.shape[1],)
    np.testing.assert_array_equal(result.values, expected.values)


@pytest.mark.parametrize("chunks", TEST_CHUNKS)
def test_tie_points_interpolation_independent_of_chunks(chunks):
    """Test that dask results are the same as numpy ones, whatever the chunks of the tie points."""
    data = np.random.default_rng(42).uniform(
        -180, 180, (TEST_N_SCANS_CHUNKED * TEST_SCAN_ALT_TIE_POINTS, TEST_ACT_TIE_POINTS))
    expected = tie_points_interpolation([_tie_points_data_array(data, False)],
                                        TEST_SCAN_ALT_TIE_POINTS, TEST_TIE_POINTS_FACTOR)[0]
    dask_data = xr.DataArray(da.from_array(data, chunks=chunks), dims=('num_tie_points_alt', 'num_tie_points_act'))
    with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
        result = tie_points_interpolation([dask_data], TEST_SCAN_ALT_TIE_POINTS, TEST_TIE_POINTS_FACTOR)[0]

    _assert_same_as_numpy(result, expected)


@pytest.mark.parametrize("chunks", TEST_CHUNKS)
@pytest.mark.parametrize("lat_range", [(0, 23), (45, 68)], ids=["lonlat", "cartesian"])
def test_tie_points_geo_interpolation_independent_of_chunks(lat_range, chunks):
    """Test that dask results are the same as numpy ones, whatever the chunks of the tie points.

    The choice of cartesian coordinates must be the same for every chunk too: in the cartesian case,
    only the chunks of the last scans reach latitudes above 60 degrees on their own.

    """
    n_tie_alt = TEST_N_SCANS_CHUNKED * TEST_SCAN_ALT_TIE_POINTS
    longitude = _linspace_tie_points(-12, 11, n_tie_alt)
    latitude = _linspace_tie_points(*lat_range, n_tie_alt)
    exp_lon, exp_lat = tie_points_geo_interpolation(
        _tie_points_data_array(longitude, False), _tie_points_data_array(latitude, False),
        TEST_SCAN_ALT_TIE_POINTS, TEST_TIE_POINTS_FACTOR)
    dims = ('num_tie_points_alt', 'num_tie_points_act')
    with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
        lon, lat = tie_points_geo_interpolation(
            xr.DataArray(da.from_array(longitude, chunks=chunks), dims=dims),
            xr.DataArray(da.from_array(latitude, chunks=chunks), dims=dims),
            TEST_SCAN_ALT_TIE_POINTS, TEST_TIE_POINTS_FACTOR)

    _assert_same_as_numpy(lon, exp_lon)
    _assert_same_as_numpy(lat, exp_lat)
