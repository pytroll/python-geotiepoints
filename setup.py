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

import os
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext
import versioneer

requirements = ['numpy', 'scipy', 'pandas'],
# unittest2 is required by h5py 2.8.0rc:
test_requires = ['h5py', 'unittest2', 'xarray', 'dask']

if sys.platform.startswith("win"):
    extra_compile_args = []
else:
    extra_compile_args = ["-O3"]

EXTENSIONS = [
    Extension(
        'geotiepoints.multilinear_cython',
        sources=['geotiepoints/multilinear_cython.pyx'],
        extra_compile_args=extra_compile_args
    ),
    Extension(
        'geotiepoints.multilinear_cython',
        sources=['geotiepoints/multilinear_cython.pyx',
                 'geotiepoints/multilinear_cython.c'],
        language='c', extra_compile_args=extra_compile_args,
        depends=[]
    )

]

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None


def set_builtin(name, value):
    if isinstance(__builtins__, dict):
        __builtins__[name] = value
    else:
        setattr(__builtins__, name, value)


cmdclass = versioneer.get_cmdclass()
versioneer_build_ext = cmdclass.get('build_ext', _build_ext)


class build_ext(versioneer_build_ext):

    """Work around to bootstrap numpy includes in to extensions.

    Copied from:

        http://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py

    """

    def finalize_options(self):
        versioneer_build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        set_builtin('__NUMPY_SETUP__', False)
        import numpy
        self.include_dirs.append(numpy.get_include())


cmdclass['build_ext'] = build_ext

if __name__ == "__main__":
    if not os.getenv("USE_CYTHON", False) or cythonize is None:
        print(
            "Cython will not be used. Use environment variable 'USE_CYTHON=True' to use it")

        def cythonize(extensions, **dummy):
            """Fake function to compile from C/C++ files instead of compiling .pyx files with cython.
            """
            for extension in extensions:
                sources = []
                for sfile in extension.sources:
                    path, ext = os.path.splitext(sfile)
                    if ext in ('.pyx', '.py'):
                        if extension.language == 'c++':
                            ext = '.cpp'
                        else:
                            ext = '.c'
                        sfile = path + ext
                    sources.append(sfile)
                extension.sources[:] = sources
            return extensions

    setup(name='python-geotiepoints',
          version=versioneer.get_version(),
          description='Interpolation of geographic tiepoints in Python',
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
          setup_requires=['numpy'],
          python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*',
          cmdclass=cmdclass,

          install_requires=requirements,
          ext_modules=cythonize(EXTENSIONS),

          test_suite='geotiepoints.tests.suite',
          tests_require=test_requires,
          zip_safe=False
          )
