# Copyright (c) 2012-2023 Python-geotiepoints developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Setting up the geo_interpolator project."""

import sys

from setuptools import setup, find_packages
import versioneer
import numpy as np
from Cython.Build import build_ext
from Cython.Distutils import Extension

requirements = ['numpy', 'scipy', 'pandas']
test_requires = ['pytest', 'pytest-cov', 'h5py', 'xarray', 'dask', 'pyproj', "pyresample"]

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
    Extension(
        'geotiepoints._modis_interpolator',
        sources=['geotiepoints/_modis_interpolator.pyx'],
        extra_compile_args=extra_compile_args,
        include_dirs=[np.get_include()],
    ),
    Extension(
        'geotiepoints._simple_modis_interpolator',
        sources=['geotiepoints/_simple_modis_interpolator.pyx'],
        extra_compile_args=extra_compile_args,
        include_dirs=[np.get_include()],
    ),
    Extension(
        'geotiepoints._modis_utils',
        sources=['geotiepoints/_modis_utils.pyx'],
        extra_compile_args=extra_compile_args,
        include_dirs=[np.get_include()],
    ),
]


try:
    sys.argv.remove("--cython-coverage")
    cython_coverage = True
except ValueError:
    cython_coverage = False


cython_directives = {
    "language_level": "3",
}
define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
if cython_coverage:
    print("Enabling directives/macros for Cython coverage support")
    cython_directives.update({
        "linetrace": True,
        "profile": True,
    })
    define_macros.extend([
        ("CYTHON_TRACE", "1"),
        ("CYTHON_TRACE_NOGIL", "1"),
    ])
for ext in EXTENSIONS:
    ext.define_macros = define_macros
    ext.cython_directives.update(cython_directives)

cmdclass = versioneer.get_cmdclass(cmdclass={"build_ext": build_ext})

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
          classifiers=["Development Status :: 5 - Production/Stable",
                       "Intended Audience :: Science/Research",
                       "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
                       "Operating System :: OS Independent",
                       "Programming Language :: Python",
                       "Programming Language :: Cython",
                       "Topic :: Scientific/Engineering"],
          url="https://github.com/pytroll/python-geotiepoints",
          packages=find_packages(),
          python_requires='>=3.10',
          cmdclass=cmdclass,
          install_requires=requirements,
          ext_modules=EXTENSIONS,
          tests_require=test_requires,
          zip_safe=False
          )
