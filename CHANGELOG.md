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
