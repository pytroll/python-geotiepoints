Changelog
=========

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


