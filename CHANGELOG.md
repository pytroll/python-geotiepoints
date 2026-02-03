## Version 1.9.0 (2026/02/03)

### Issues Closed

* [Issue 105](https://github.com/pytroll/python-geotiepoints/issues/105) - Relicense to Apache Version 2 ([PR 114](https://github.com/pytroll/python-geotiepoints/pull/114) by [@djhoese](https://github.com/djhoese))

In this release 1 issue was closed.

### Pull Requests Merged

#### Features added

* [PR 114](https://github.com/pytroll/python-geotiepoints/pull/114) - Relicense to Apache Version 2.0 ([105](https://github.com/pytroll/python-geotiepoints/issues/105))

In this release 1 pull request was closed.


## Version 1.8.0 (2025/09/10)

### Issues Closed

* [Issue 37](https://github.com/pytroll/python-geotiepoints/issues/37) - Issues with interpolation on a Sentinel Tie Point Grid. 

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 103](https://github.com/pytroll/python-geotiepoints/pull/103) - Fix gcc compiler warnings in MODIS interpolator

#### Features added

* [PR 102](https://github.com/pytroll/python-geotiepoints/pull/102) - Add free-threading compatibility and drop Python 3.10 support

In this release 2 pull requests were closed.


## Version 1.7.5 (2024/10/12)

### Issues Closed

* [Issue 79](https://github.com/pytroll/python-geotiepoints/issues/79) - Test failure with scipy 1.13.x

In this release 1 issue was closed.

### Pull Requests Merged

#### Features added

* [PR 85](https://github.com/pytroll/python-geotiepoints/pull/85) - Add a spline interpolator for 2d arrays

In this release 1 pull request was closed.


## Version 1.7.4 (2024/06/26)

### Pull Requests Merged

#### Bugs fixed

* [PR 76](https://github.com/pytroll/python-geotiepoints/pull/76) - Fix numpy 2 dtype issues

In this release 1 pull request was closed.


## Version 1.7.3 (2024/04/15)

### Pull Requests Merged

#### Bugs fixed

* [PR 74](https://github.com/pytroll/python-geotiepoints/pull/74) - Build wheels with numpy 2.0rc1 and fix scipy 1.13.0 compatibility

In this release 1 pull request was closed.


## Version 1.7.2 (2024/02/14)

### Pull Requests Merged

#### Bugs fixed

* [PR 63](https://github.com/pytroll/python-geotiepoints/pull/63) - Deploy to test pypi only on tags

#### Features added

* [PR 70](https://github.com/pytroll/python-geotiepoints/pull/70) - Build wheels with numpy 2

In this release 2 pull requests were closed.


## Version 1.7.1 (2023/11/28)

### Pull Requests Merged

#### Bugs fixed

* [PR 62](https://github.com/pytroll/python-geotiepoints/pull/62) - Fix python versions in deploy ci

In this release 1 pull request was closed.


## Version 1.7.0 (2023/11/21)

### Issues Closed

* [Issue 56](https://github.com/pytroll/python-geotiepoints/issues/56) - Upgrade to Cython 3.0 and check annotations ([PR 57](https://github.com/pytroll/python-geotiepoints/pull/57) by [@djhoese](https://github.com/djhoese))
* [Issue 47](https://github.com/pytroll/python-geotiepoints/issues/47) - Help wanted: verify the interpolation of MERSI-2 1000M GEO to 250M GEO
* [Issue 23](https://github.com/pytroll/python-geotiepoints/issues/23) - Docstring headers still include authors.
* [Issue 21](https://github.com/pytroll/python-geotiepoints/issues/21) - Interpolation of MODIS lat/lons is incorrect
* [Issue 18](https://github.com/pytroll/python-geotiepoints/issues/18) - Make the interpolators dask-compatible

In this release 5 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 60](https://github.com/pytroll/python-geotiepoints/pull/60) - Add missing noexcept on cython function
* [PR 46](https://github.com/pytroll/python-geotiepoints/pull/46) - Fix tests on i386 architectures

#### Features added

* [PR 61](https://github.com/pytroll/python-geotiepoints/pull/61) - Fix geogrid chunking to accept "auto" and to preserve dtype
* [PR 57](https://github.com/pytroll/python-geotiepoints/pull/57) - Upgrade to Cython 3+ in building ([56](https://github.com/pytroll/python-geotiepoints/issues/56))

In this release 4 pull requests were closed.


## Version 1.6.0 (2023/03/17)


### Pull Requests Merged

#### Bugs fixed

* [PR 45](https://github.com/pytroll/python-geotiepoints/pull/45) - Fix VII interpolator compatibility with future versions of xarray

#### Features added

* [PR 44](https://github.com/pytroll/python-geotiepoints/pull/44) - Add interpolators based on scipy's RegularGridInterpolator

In this release 2 pull requests were closed.


## Version 1.5.1 (2022/12/09)

### Pull Requests Merged

#### Bugs fixed

* [PR 43](https://github.com/pytroll/python-geotiepoints/pull/43) - Fix deprecation for numpy array equality

In this release 1 pull request was closed.


## Version 1.5.0 (2022/10/25)

### Pull Requests Merged

#### Features added

* [PR 38](https://github.com/pytroll/python-geotiepoints/pull/38) - Rewrite simple and tiepoint modis interpolation in cython

In this release 1 pull request was closed.


## Version 1.4.1 (2022/06/08)

### Issues Closed

* [Issue 39](https://github.com/pytroll/python-geotiepoints/issues/39) - MODIS Interpolation Comparisons ([PR 41](https://github.com/pytroll/python-geotiepoints/pull/41) by [@djhoese](https://github.com/djhoese))

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 41](https://github.com/pytroll/python-geotiepoints/pull/41) - Fix MODIS cviirs-based interpolation ([39](https://github.com/pytroll/python-geotiepoints/issues/39))

#### Features added

* [PR 35](https://github.com/pytroll/python-geotiepoints/pull/35) - Optimize angle-based modis interpolation for dask

In this release 2 pull requests were closed.


## Version 1.4.0 (2022/02/21)

### Pull Requests Merged

#### Features added

* [PR 34](https://github.com/pytroll/python-geotiepoints/pull/34) - Updated interpolator for vii tie points for test data version V2 

In this release 1 pull request was closed.


## Version 1.3.1 (2022/02/04)

### Pull Requests Merged

#### Bugs fixed

* [PR 33](https://github.com/pytroll/python-geotiepoints/pull/33) - Fix deprecated use of np.int

#### Features added

* [PR 32](https://github.com/pytroll/python-geotiepoints/pull/32) - Change tested Python versions to 3.8, 3.9 and 3.10

In this release 2 pull requests were closed.


## Version 1.3.0 (2021/09/12)

### Pull Requests Merged

#### Features added

* [PR 31](https://github.com/pytroll/python-geotiepoints/pull/31) - Add simple lon/lat based MODIS interpolation

In this release 1 pull request was closed.


## Version 1.2.1 (2021/03/08)

### Issues Closed

* [Issue 29](https://github.com/pytroll/python-geotiepoints/issues/29) - C extension does not compile on py3.9 without re-cythonizing ([PR 30](https://github.com/pytroll/python-geotiepoints/pull/30))
* [Issue 28](https://github.com/pytroll/python-geotiepoints/issues/28) - I'm trying to install pycups on mac os using the treminal, but I'm getting “building wheel for pycups (setup.py) … error”
* [Issue 27](https://github.com/pytroll/python-geotiepoints/issues/27) - MNT: Stop using ci-helpers in appveyor.yml ([PR 30](https://github.com/pytroll/python-geotiepoints/pull/30))
* [Issue 26](https://github.com/pytroll/python-geotiepoints/issues/26) - pip install pysbrl --no-binary=pysbrl gives error

In this release 4 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 30](https://github.com/pytroll/python-geotiepoints/pull/30) - Switch build system to require Cython and build extensions on install ([29](https://github.com/pytroll/python-geotiepoints/issues/29), [27](https://github.com/pytroll/python-geotiepoints/issues/27))

#### Features added

* [PR 30](https://github.com/pytroll/python-geotiepoints/pull/30) - Switch build system to require Cython and build extensions on install ([29](https://github.com/pytroll/python-geotiepoints/issues/29), [27](https://github.com/pytroll/python-geotiepoints/issues/27))

In this release 2 pull requests were closed.


## Version 1.2.0 (2020/06/05)


### Pull Requests Merged

#### Bugs fixed

* [PR 19](https://github.com/pytroll/python-geotiepoints/pull/19) - Fix interpolation of symetrical tiepoints

#### Features added

* [PR 22](https://github.com/pytroll/python-geotiepoints/pull/22) - Add VII interpolator.
* [PR 16](https://github.com/pytroll/python-geotiepoints/pull/16) - Add MODIS 5km to 500m and 250m interpolation

In this release 3 pull requests were closed.


## Version 1.1.8 (2019/04/24)

### Issues Closed

### Pull Requests Merged

#### Bugs fixed

* [PR 14](https://github.com/pytroll/python-geotiepoints/pull/14) - Fix modis interpolation in tricky places

#### Features added

* [PR 15](https://github.com/pytroll/python-geotiepoints/pull/15) - Add support for modis l2 geolocation interpolation

In this release 2 pull requests were closed.


## Version v1.1.7 (2018/10/09)

### Issues Closed

* [Issue 8](https://github.com/pytroll/python-geotiepoints/issues/8) - When I install this package,it said 'Failed building wheel for python-geotiepoints'.

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 12](https://github.com/pytroll/python-geotiepoints/pull/12) - Fix python 3 compatibility for the metop interpolator

#### Features added

* [PR 13](https://github.com/pytroll/python-geotiepoints/pull/13) - Switch to versioneer and loghub
* [PR 11](https://github.com/pytroll/python-geotiepoints/pull/11) - Add cviirs-based fast modis interpolator ([405](https://github.com/pytroll/satpy/issues/405))

In this release 3 pull requests were closed.
