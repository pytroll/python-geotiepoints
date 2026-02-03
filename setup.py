"""Setting up the geo_interpolator project."""

import sys

from setuptools import setup, find_packages
import versioneer
import numpy as np
from Cython.Build import build_ext
from Cython.Distutils import Extension

requirements = ['numpy', 'scipy', 'pandas']
test_requires = ['pytest', 'pytest-cov', 'h5py', 'xarray', 'dask[array]', 'pyproj', "pyresample"]

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
    "freethreading_compatible": True,
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
          classifiers=[
              "Development Status :: 5 - Production/Stable",
              "Intended Audience :: Science/Research",
              "Operating System :: OS Independent",
              "Programming Language :: Python",
              "Programming Language :: Cython",
              "Topic :: Scientific/Engineering",
              "Programming Language :: Python :: Free Threading :: 1 - Unstable",
          ],
          license="Apache-2.0",
          license_files=["LICENSE.txt"],
          url="https://github.com/pytroll/python-geotiepoints",
          packages=find_packages(),
          python_requires='>=3.11',
          cmdclass=cmdclass,
          install_requires=requirements,
          ext_modules=EXTENSIONS,
          extras_require={
              "tests": test_requires,
          },
          zip_safe=False
          )
