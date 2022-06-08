#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2012-2018 PyTroll community

# Author(s):

#   Adam Dybbroe <adam.dybbroe@smhi.se>
#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Setting up the geo_interpolator project.
"""

import sys

from setuptools import Extension, setup
import versioneer
import numpy as np
from Cython.Build import cythonize

requirements = ['numpy', 'scipy', 'pandas']
test_requires = ['pytest', 'pytest-cov', 'h5py', 'xarray', 'dask', 'pyproj']

if sys.platform.startswith("win"):
    extra_compile_args = []
else:
    extra_compile_args = ["-O3"]

EXTENSIONS = [
    Extension(
        'geotiepoints.multilinear_cython',
        sources=['geotiepoints/multilinear_cython.pyx'],
        extra_compile_args=extra_compile_args,
        include_dirs=[np.get_include()],
    ),
]

cmdclass = versioneer.get_cmdclass()

with open('README.md', 'r') as readme:
    README = readme.read()

if __name__ == "__main__":
    setup(name='python-geotiepoints',
          version=versioneer.get_version(),
          description='Interpolation of geographic tiepoints in Python',
          long_description=README,
          long_description_content_type='text/markdown',
          author='Adam Dybbroe, Martin Raspaud',
          author_email='martin.raspaud@smhi.se',
          classifiers=["Development Status :: 4 - Beta",
                       "Intended Audience :: Science/Research",
                       "License :: OSI Approved :: GNU General Public License v3 " +
                       "or later (GPLv3+)",
                       "Operating System :: OS Independent",
                       "Programming Language :: Python",
                       "Topic :: Scientific/Engineering"],
          url="https://github.com/pytroll/python-geotiepoints",
          packages=['geotiepoints'],
          # packages=find_packages(),
          setup_requires=['numpy', 'cython'],
          python_requires='>=3.7',
          cmdclass=cmdclass,
          install_requires=requirements,
          ext_modules=cythonize(EXTENSIONS),
          tests_require=test_requires,
          zip_safe=False
          )
