# python-geotiepoints — Agent Guide

python-geotiepoints interpolates (and extrapolates) a coarse grid of geolocation **tiepoints** into a
full-resolution lon/lat grid. Its primary consumer is [Satpy](https://github.com/pytroll/satpy), whose
readers call almost every public module here (see "Downstream API" below) — so treat public function
and class signatures as load-bearing and avoid breaking them.

The library is old and low-churn; recent commits are mostly dependabot bumps. Conventions drifted over
the years, so *do not* assume a pattern found in one module holds in another. This guide records what
is expensive to rediscover.

## Four independent lineages

These share almost no code and are **not** interchangeable.

| Lineage | Modules | Notes |
|---|---|---|
| Generic spline/grid | `interpolator.py`, `geointerpolator.py` | Two generations in one file (below) |
| MODIS Cython | `modisinterpolator.py` + `_modis_interpolator.pyx`; `simple_modis_interpolator.py` + `_simple_modis_interpolator.pyx`; `_modis_utils.pyx` | Modern MODIS path |
| EPS-SG VII | `viiinterpolator.py` | **Not VIIRS.** Pure xarray, no Cython |
| Multilinear | `multilinear.py` + `multilinear_cython.pyx` | Regular cartesian grid; unrelated to geolocation |

Non-obvious details:

- **`interpolator.py` holds two generations.** *Gen 1* is `Interpolator`: stateful, mutates
  `self.tie_data`/`self.row_indices` in `fill_borders()`, numpy-only, and despite the `ABC` import it
  is a plain class. *Gen 2* is `AbstractSingleInterpolator` → `SingleGridInterpolator` /
  `SingleSplineInterpolator` and `AbstractMultipleInterpolator` → `MultipleGridInterpolator` /
  `MultipleSplineInterpolator`: functional, no mutation, and they accept `chunks=` for lazy dask
  output. **New work belongs in Gen 2.**
- **`GeoGridInterpolator` and `GeoSplineInterpolator` are built dynamically** by the
  `_work_with_lonlats(klass)` class factory (`geointerpolator.py:78`). Grepping for the class body only
  finds `GeoKlass`; the names are bound at `geointerpolator.py:111-112`.
- **`viiinterpolator` is EPS-SG VII** (Visible/Infrared Imager), not VIIRS. Confusingly, the *CVIIRS*
  algorithm lives in `modisinterpolator.py`.
- **`basic_interpolator.py` is dead code.** It calls `DataFrame.as_matrix()`
  (`basic_interpolator.py:57`), removed in pandas 1.0, so it cannot run. Nothing imports it and it has
  no tests.
- **The `__init__.py` API is legacy but live.** `SatelliteInterpolator` (an alias of `GeoInterpolator`),
  `metop20kmto1km`, `modis5kmto1km`, `modis1kmto500m`, `modis1kmto250m`, `get_scene_splits`. It emits
  no deprecation warnings and Satpy still calls it. It is also the *only* thing `doc/source/index.rst`
  documents — none of the modern modules appear in the docs, so the code is the reference.

## Domain primer

**Scans and the bowtie.** These instruments sweep in *scans*, each of which is N detector rows.
Interpolation must be done per-scan and must **never** cross a scan boundary: the geometry is
discontinuous there (the "bowtie" effect), and interpolating across it produces garbage. Rows per scan
depend on resolution and are centralized in `rows_per_scan_for_resolution()` (`_modis_utils.pyx`):
`{5000: 2, 1000: 10, 500: 20, 250: 40}`. Every dask chunk must therefore contain whole scans.

**Why lon/lat → xyz → interpolate → lon/lat.** Interpolating degrees directly breaks at the
antimeridian (the 180/−180 wrap) and is degenerate near the poles, where longitude carries almost no
information. Cartesian xyz on a sphere is continuous everywhere, so every lineage converts in,
interpolates, and converts back. Converting back has its own trap: `xyz2lonlat` derives latitude from
`z` alone at low latitudes but from `sqrt(x²+y²)` near the poles, switched on `thr=0.8` of the Earth
radius, because `arccos(z/R)` is ill-conditioned near ±90°. `viiinterpolator` goes further and skips
the cartesian round trip entirely below 60° when the data does not cross the antimeridian.

**MODIS tiepoint layout.** A 1km MODIS swath is 1354 columns wide. 5km tiepoints are 271 columns
(MOD03-style) or 270 (some L2 products); `MODISInterpolator._get_coords_5km()` special-cases the two
with different right-edge `x` values, and anything else raises `NotImplementedError`. The recurring
"every 5th pixel starting at index 2" convention shows up as `np.arange(2, 1354, 5)` and `lon1[2::5, 2::5]`.

**Satellite-zenith geometry.** `_compute_expansion_alignment` (`_modis_interpolator.pyx`) derives
per-cell *expansion* and *alignment* coefficients from the satellite zenith angles at the corners of
each coarse cell, then bilinearly blends the four corner xyz values. It uses `R = 6370.997` km and
`H = 709.0` km (Aqua orbit altitude). The `theta_a == theta_b` branch is a deliberate fix for tiepoints
symmetric about the sub-satellite track (GH #19 NaN bug), guarded by
`test_sat_angle_based_interp_nan_handling`.

**Which MODIS path to use.** Prefer `modisinterpolator` when a SensorZenith dataset is available — it
is the more accurate algorithm (test tolerances 5 m for 1km→500m, 8.3 m for 1km→250m).
`simple_modis_interpolator` exists solely because 250m/500m L1b and L2 files ship no satz dataset; it
needs only lon/lat, supports only 1km→250m/500m, and is less accurate (16 m and 27.35 m).

**Terrain correction.** MOD03 geolocation is terrain-corrected, so a 1km→5km→1km round trip can never
be exact. That is why the test fixture is an ocean scene and why tests assert *geodetic distance*
(`assert_geodetic_distance`, via `pyproj.Geod`) rather than exact array equality.

## Dask and xarray: three unrelated mechanisms

1. **`scanline_mapblocks`** (`_modis_utils.pyx`) drives *both* MODIS paths. numpy in → numpy out
   (called directly); dask in → `da.map_blocks`; `DataArray` in → `DataArray` out — but **coords and
   attrs are dropped**, only `dims` is reattached. It rechunks rows to whole multiples of
   `rows_per_scan` and forces full-width column chunks, and it raises `ValueError` if the input is not
   whole scans. To return *two* arrays from `map_blocks` it stacks `(lon, lat)` into one
   `(2, rows, cols)` array via `new_axis=[0]` and splits the result back apart.
   `coarse_resolution` and `fine_resolution` are **required keyword-only** arguments.
2. **`AbstractSingleInterpolator.interpolate_dask`** (`interpolator.py`) hand-builds a dask graph
   (`normalize_chunks` + `tokenize` + a task dict), deliberately *not* `map_blocks`. Enabled by passing
   `chunks=` to `interpolate()` / `interpolate_to_shape()`; `chunks="auto"` works.
3. **`viiinterpolator`** relies entirely on `xarray.DataArray.interp()` and *requires* xarray input. It
   interpolates across-track then along-track as two 1-D passes because that is much faster than 2-D.

Everything else (the `__init__.py` legacy helpers, Gen-1 `Interpolator`/`GeoInterpolator`,
`multilinear`) is numpy-only; `__init__.py` parallelizes with `multiprocessing.Pool` instead.

**Laziness is enforced.** Tests wrap calls in `dask.config.set(scheduler=CustomScheduler(0))`
(`geotiepoints/tests/utils.py`), which raises if anything computes. New dask-facing code must build a
graph without computing.

## Cython conventions

- Every `.pyx`/`.pxd` opens with the identical directive line:
  `# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False, initializedcheck=False, nonecheck=False`
- **`wraparound=False` means negative indexing is banned** and silently wrong if used. There are past
  bugfix commits for exactly this. The sole exception is `_get_coords_5km`, explicitly annotated
  `@cython.wraparound(True)` (`_modis_interpolator.pyx:231`) so it can use `x[-2]`, `x[-7]`.
- **Two different `floating` fused types.** `_modis_utils.pxd` defines its own (float32/float64) and the
  MODIS `.pyx` files `cimport` it; `multilinear_cython.pyx` uses `from cython cimport floating`, the
  builtin, which *also* includes `long double`.
- C-contiguous typed memoryviews (`floating[:, ::1]`) throughout, with `np.ascontiguousarray()` at
  entry points, `noexcept nogil` on `cdef` helpers, and `with nogil:` around the per-scan loops.
  `_simple_modis_interpolator` re-acquires the GIL to call `scipy.ndimage.map_coordinates`.
- `_modis_utils.xyz2lonlat` deliberately upcasts to float64 internally even for float32 input
  ("64-bit precision matters apparently").
- Free-threading support is intentional: `freethreading_compatible=True` in `setup.py`, cp314t wheels,
  and a `Free Threading :: 1 - Unstable` classifier.
- `_simple_modis_interpolator.pyx` redundantly re-applies the file-level directives as per-function
  decorators; `_modis_interpolator.pyx` does not.

## Build, test, lint

```bash
pip install -e .
python setup.py build_ext --inplace --cython-coverage --force   # required before running tests
pytest geotiepoints/tests
pytest --cov=geotiepoints geotiepoints/tests --cov-report=xml   # what CI runs
make -C doc doctest
```

`--cython-coverage` is a **custom `setup.py` flag** that adds the `linetrace`/`profile` directives and
`CYTHON_TRACE` macros; without it Cython line coverage is empty.

- Tests load HDF5 fixtures by path relative to the test file (`../../testdata/`), so they only work
  from a source checkout, never from an installed wheel.
- There is **no `conftest.py` and no pytest configuration at all** — no markers, no ini options.
- `test_simple_modis_interpolator.py` imports its loaders and `assert_geodetic_distance` from
  `test_modisinterpolator.py`. Preserve that cross-module dependency.
- `testdata/create_modis_test_data.py` regenerates fixtures but needs `pyhdf` plus a real MOD03 file;
  it is not run in CI.
- Linting is **flake8 only**, configured in `setup.cfg` (`max-line-length = 120`, `ignore = D107`).
  There is no pre-commit config and no ruff config.
- CI: ubuntu/macos/windows × Python 3.11/3.12/3.13, plus an experimental nightly-dependency job.
  `python_requires >= 3.11`.
- There is no `[project]` table — package metadata still lives in `setup.py`.
- `geotiepoints/version.py` is versioneer-generated; never edit it. Release steps are in `RELEASING.md`.

## Known defects and traps

Verified as of this writing; fix them only when the task calls for it.

- `_modis_utils.pyx:103` — `if first_arr.ndim != 2 or first_arr.ndim != 2:` tests the same condition
  twice; the second was presumably meant to validate another array.
- `_modis_utils.pyx:173` — `good_col_chunks = len(col_chunks) == 1 and col_chunks[0] != num_cols` is
  always `False` (a single column chunk must equal `num_cols`), so the rechunk branch always runs.
  The `!=` looks like it should be `==`.
- `_modis_utils.pyx:178` — the error message hardcodes "(10 rows per scan)" regardless of resolution.
- `_modis_interpolator.pyx:4` imports `scanline_mapblocks` from `.simple_modis_interpolator` rather than
  from `._modis_utils` where it is defined, coupling the two MODIS front-ends for no reason.
- `multilinear_cython.pyx` uses `prange`, but `setup.py` compiles with only `-O3` and no
  `-fopenmp`/`/openmp`, so those loops are serial. `multilinear_interpolation_5d` is unreachable — the
  dispatcher raises for `d > 4`.
- **Three Earth radii**: `6370997.0` (`__init__.py`, `geointerpolator.py`, `_modis_utils.pyx`),
  `6370.997` km (`_modis_interpolator.pyx`), `6371008.7714` (`viiinterpolator.py`).
- `AbstractMultipleInterpolator.interpolate` (inherited by both `Multiple*Interpolator` classes)
  returns a **generator**, not a tuple.
- `simple_modis_interpolator.interpolate_geolocation_cartesian`'s docstring documents a `res_factor`
  argument that no longer exists.
- `DEF` compile-time constants (`DEF R`, `DEF H`, `DEF EARTH_RADIUS`) are deprecated in Cython 3.
- `interpolator.py`'s `Interpolator` docstring describes `kx_`/`ky_` as orders "in x and y", but
  `_interp` passes `kx=self.kx_` to the *row* (first) axis of `RectBivariateSpline`.

## Roadmap

Direction the maintainer intends to take. Context for work that is asked for — not authorization to
start it unprompted.

- **Linting and consistency.** Adopt ruff + pre-commit; unify docstring style (Google is the most
  common today, but `multilinear.py` is NumPy-style and Gen-1 `interpolator.py`/`geointerpolator.py`
  are bare prose); deduplicate the three copies of `lonlat2xyz`/`xyz2lonlat` and the Earth radius
  constants; reconcile module naming (`modisinterpolator.py`/`viiinterpolator.py` vs
  `simple_modis_interpolator.py`/`basic_interpolator.py`); document or rename the undocumented `h`
  prefix (`hrow_indices`/`hcol_indices` = high-resolution) and the `kx_`/`x__` trailing/double
  underscore habits; fix the `course_col_idx` typo (should be `coarse_`); remove dead code.
- **Interpolator performance.** The MODIS Cython kernels and the `scanline_mapblocks` chunking are the
  hot spots. Any change must hold the existing geodetic-distance tolerances and must not introduce dask
  computes (`CustomScheduler(0)` will fail the tests if it does).

## Downstream API (Satpy call sites)

| geotiepoints entry point | Satpy module |
|---|---|
| `SatelliteInterpolator` | `readers/hrpt.py`, `readers/nwcsaf_nc.py` |
| `metop20kmto1km` | `readers/eps_l1b.py` |
| `modisinterpolator.modis_1km_to_250m/500m`, `modis_5km_to_1km` | `readers/core/hdfeos.py` |
| `simple_modis_interpolator.modis_1km_to_250m/500m` | `readers/core/hdfeos.py` |
| `geointerpolator.lonlat2xyz`/`xyz2lonlat`, `interpolator.MultipleSplineInterpolator` | `readers/sar_c_safe.py` |
| `geointerpolator.GeoSplineInterpolator` | `readers/sgli_l1b.py` |
| `geointerpolator.GeoInterpolator` | `readers/ici_l1b_nc.py`, `readers/aapp_l1b.py` |
| `interpolator.Interpolator` | `readers/olci_nc.py`, `readers/aapp_l1b.py` |
| `viiinterpolator.tie_points_interpolation`, `tie_points_geo_interpolation` | `readers/core/metimage_nc.py` |
| `multilinear.MultilinearInterpolator` | `readers/msi_safe.py` |
