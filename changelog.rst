Changelog
=========


v1.1.6 (2018-09-25)
-------------------
- Update changelog. [David Hoese]
- Bump version: 1.1.5 → 1.1.6. [David Hoese]
- Merge pull request #10 from pytroll/bugfix-travis-37. [David Hoese]

  Fix python 3.7 environment on travis
- Add generic language settings for osx environments on travis. [David
  Hoese]
- Remove generic language setting for travis and added OSX 3.7 env.
  [David Hoese]
- Merge pull request #9 from AmitAronovitch/py37-support. [David Hoese]

  support Python 3.7
- Add py3.7 on linux test in travis. [Amit Aronovitch]
- Rebuild multilinear_cython.c with Cython 0.28 (supports py3.7) [Amit
  Aronovitch]
- Add templates for issues and PRs. [Adam.Dybbroe]


v1.1.5 (2018-05-21)
-------------------
- Update changelog. [davidh-ssec]
- Bump version: 1.1.4 → 1.1.5. [davidh-ssec]
- Remove wheel deployment on travis. [davidh-ssec]
- Add skip_existing to travis deploy. [davidh-ssec]


v1.1.4 (2018-05-21)
-------------------
- Update changelog. [davidh-ssec]
- Bump version: 1.1.3 → 1.1.4. [davidh-ssec]
- Merge pull request #7 from pytroll/bugfix-packaging. [David Hoese]

  Add MANIFEST to include cython files and update travis to use ci-helpers
- Replace 'Cython' dependency with 'cython' in conda environment on
  travis. [davidh-ssec]
- Fix adding osx to travis tests by using a matrix. [davidh-ssec]
- Add osx to travis tests and add python_requires to setup.py. [davidh-
  ssec]
- Add appveyor config. [davidh-ssec]
- Add MANIFEST to include cython files and update travis to use ci-
  helpers. [davidh-ssec]
- Fix travis/slack integration. [Adam.Dybbroe]


v1.1.3 (2018-03-12)
-------------------
- Update changelog. [Adam.Dybbroe]
- Bump version: 1.1.2 → 1.1.3. [Adam.Dybbroe]
- Add unittest2 as a test-requirement. [Adam.Dybbroe]

  setuptools installs h5py2.8.0rc from PyPI, which in turn requires
  unittest2, however, which is not in the h5py requirements!

- Fix unit tests for Python 3. [Adam.Dybbroe]
- Merge branch 'new_release' into develop. [Adam.Dybbroe]
- Build and test Python 3.4, 3.5, 3.6 on Travis. [Adam.Dybbroe]
- Fix doc tests. [Adam.Dybbroe]
- Merge pull request #6 from mitkin/develop. [Adam Dybbroe]

  Set "C" order of interpolated arrays explicitly
- Set "C" order of interpolated arrays explicitly. [Mikhail Itkin]

  It appears that scipy's spline interpolator returns an array that are in F and C
  order simultaneously. Transposing and viewing the array converts it into
  F-contiguous array. By specifying the order explicitly we convert it to
  C order



v1.1.2 (2017-12-01)
-------------------
- Update changelog. [Adam.Dybbroe]
- Bump version: 1.1.1 → 1.1.2. [Adam.Dybbroe]
- Fix bumpversion file. [Adam.Dybbroe]
- Go back one version number - bumpversion will bump it. [Adam.Dybbroe]
- Fix author mail address. [Adam.Dybbroe]
- Add separate version file. [Adam.Dybbroe]
- Bugfix documentation - code example. [Adam.Dybbroe]


v1.1.1 (2017-05-31)
-------------------
- Update changelog. [Adam.Dybbroe]
- Bump version: 1.1.0 → 1.1.1. [Adam.Dybbroe]
- Merge branch 'bugfix_201608_change' into develop. [Adam.Dybbroe]
- Fix tests for modis data interpolation. [Adam.Dybbroe]
- Add h5py to test_requires. [Martin Raspaud]
- Fix modis interpolators. [Martin Raspaud]
- Fix temporary mytest code. [Adam.Dybbroe]
- Comment out test that fails in the post aug2016 ode change.
  [Adam.Dybbroe]
- Add unittest for modis5kmto1km. Make testing of code before and after
  the august 2016 change possible. [Adam.Dybbroe]
- Add Cython generated C-code and make installation possible without
  having Cython and numpy header files available. [Adam.Dybbroe]

  Looked at how it was done for pyresample.



v1.1.0 (2017-05-19)
-------------------
- Update changelog. [Adam.Dybbroe]
- Bump version: 1.0.0 → 1.1.0. [Adam.Dybbroe]
- Merge pull request #3 from pytroll/multilinear-cython. [Adam Dybbroe]

  Multilinear cython
- Fix unittests. [Martin Raspaud]
- Merge remote-tracking branch 'origin/multilinear-cython' into
  multilinear-cython. [Martin Raspaud]
- Remove pyresample from the list of required packages. [Adam.Dybbroe]
- Fix extrapolation after lowres indices are in highres numberspace.
  [Martin Raspaud]
- Add back and fix the test_extrapolate_rows test. [Adam.Dybbroe]
- Restructure test-suite. Comment out tests that hasn't been maintained.
  [Adam.Dybbroe]
- Add badges to frontpage. [Adam.Dybbroe]
- Add unittest for multilinear interpolation. [Adam.Dybbroe]
- Add Cython to requirements file. [Adam.Dybbroe]
- Add requirements file. [Adam.Dybbroe]
- Rename README file. [Adam.Dybbroe]
- Prepare for travis, and clean up. [Adam.Dybbroe]
- Add fast multilinear interpolation on regular grid with Cython.
  [Adam.Dybbroe]


v1.0.0 (2016-10-27)
-------------------

Fix
~~~
- Bugfix: new_data attr was not initialized correctly in
  GeoInterpolator. [Martin Raspaud]

Other
~~~~~
- Update changelog. [Martin Raspaud]
- Bump version: 0.3.0 → 1.0.0. [Martin Raspaud]
- Add .bumpversion.cfg and .gitchangelog.rc. [Martin Raspaud]
- Fix row extrapolation in the chunked case. [Martin Raspaud]
- Merge pull request #1 from mitkin/develop. [Adam Dybbroe]

  [setup.py] added missing dependency Pandas
- [setup.py] added missing dependency Pandas. [Mikhail Itkin]

  `basic_interpolator` imports pandas, which was not in the `install_requires`
  this commit adds `pandas` to the `install_requires` in setup.py

- Add setup.cfg for rpm building. [Martin Raspaud]
- Removed dependency to memory profiler. [HelgeDMI]
- Basic bilinear interpolation of geotie points, which is even running
  on my local machine on the biggest Sentinel-1 input files (ca. 530MB).
  I have to add a test and test data. [Rolf-Helge Pfeiffer]
- Bump up version number to v0.3.0. [Martin Raspaud]
- Update documentation with new interface. [Martin Raspaud]
- Major reorganization and tests. [Martin Raspaud]

  * A new generic Interpolator has been introduced.
  * The SatelliteInterpolator is renamed to GeoInterpolator
  * The GeoInterpolator uses the generic Interpolator
  * SatelliteInterpolator is an alias for GeoInterpolator
  * Added regular unittests instead of heavy doctests.

- Merge branch 'multicore-feature' into develop. [Martin Raspaud]

  Conflicts:
  	tests/test_modis.py

- Cleanup. [Martin Raspaud]
- Core number fix. [Martin Raspaud]
- Remove unneeded arguments. [Martin Raspaud]
- Generalize multiprocessing. [Martin Raspaud]
- Bug fixing. [Adam Dybbroe]
- Adding util functions for cpu-setting and scene splitting. Cleaning up
  a bit. [Adam Dybbroe]
- Adding multiprocessing capability to the modis 1km to 250 meter
  interpolation. [Adam Dybbroe]
- Test multicore interpolation. [Martin Raspaud]
- Merge branch 'develop' of github.com:adybbroe/python-geotiepoints into
  develop. [Martin Raspaud]
- Merge branch 'develop' of github.com:adybbroe/python-geotiepoints into
  develop. [Martin Raspaud]
- Merge branch 'release-0.2' into develop. [Adam Dybbroe]
- Merge github.com:adybbroe/python-geotiepoints into develop. [Martin
  Raspaud]
- Tell about automatic extrapolation. [Martin Raspaud]
- Bump up version number. [Martin Raspaud]
- Merge branch 'release-0.2' [Adam Dybbroe]
- Autodocs: More mockup... [Adam Dybbroe]
- Mockup to avoid import errors when using autodoc. [Adam Dybbroe]
- Conf.py pythonpath settings. [Adam Dybbroe]
- Docs... [Adam Dybbroe]
- Docs... [Adam Dybbroe]
- Autodocs... [Adam Dybbroe]
- Fixing for autodoc... [Adam Dybbroe]
- Merge branch 'master' into release-0.2. [Adam Dybbroe]
- Clean up and try prepare for ReadTheDocs. [Adam Dybbroe]
- Merge branch 'release-0.2' [Adam Dybbroe]
- Testdata. [Adam Dybbroe]
- Temporary fix of file paths in tests. [Adam Dybbroe]
- Testdata added. [Adam Dybbroe]
- Test-code and data added. [Adam Dybbroe]
- Fixing bug in fill_borders. MODIS 250 meter fixed. [Adam Dybbroe]
- Added more documentation - examples and images. [Adam Dybbroe]
- Added documentation. [Martin Raspaud]


v0.1.0 (2012-05-15)
-------------------
- Doc: Added a few things in the readme. [Martin Raspaud]
- Fixing urls. [Martin Raspaud]
- Prepare for pypi. [Martin Raspaud]
- Merge branch 'master' of https://github.com/adybbroe/python-
  geotiepoints. [Adam Dybbroe]
- Initial commit. [Adam Dybbroe]
- Changing dir name also. [Martin Raspaud]
- Changed the name of the project to python-geotiepoints. [Martin
  Raspaud]
- Removed dependency to pyresample, and cleaned up. [Martin Raspaud]
- Cleanup a bit. [Martin Raspaud]
- Merge branch 'develop' of /data/proj/SAF/GIT/geo_interpolator into
  develop. [Martin Raspaud]
- Added GPLv3 license text. [Adam Dybbroe]
- Added metop interpolator and 1d interpolation. [Martin Raspaud]
- Documentation. [Martin Raspaud]
- Fixed documentation. [Martin Raspaud]
- Cleanup. [Martin Raspaud]
- Added modis functions and orders are now passed to interpolator
  constructor. [Martin Raspaud]
- Cleanup. [Martin Raspaud]
- Cleaning and bugfixing. Seems to work. [Martin Raspaud]

  Tested against real data.

- WIP: Reshaped SatelliteInterpolator, and added modis5kmto1km function.
  [Martin Raspaud]

  Relatively untested version. Should be functional though.

- Added a setup.py and renamed for consistency. [Martin Raspaud]
- Initial commit. [Martin Raspaud]


