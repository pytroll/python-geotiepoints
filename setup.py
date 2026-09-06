"""Setting up the geo_interpolator project."""

import os
import re
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

# Only this extension uses ``prange``; the MODIS extensions are deliberately
# built without OpenMP so they don't gain a needless libgomp dependency.
OMP_EXTENSION = "geotiepoints.multilinear_cython"

OMP_SETTING_TABLE = {
    '1': 'probe',
    '0': None,
    'gcc': 'gomp',
    'gomp': 'gomp',
    'clang': 'omp',
    'omp': 'omp',
    'msvc': 'msvc',
    'probe': 'probe',
}

OMP_COMPILE_ARGS = {
    'gomp': ['-fopenmp'],
    'omp': ['-Xpreprocessor', '-fopenmp'],
    'msvc': ['/openmp'],
}

OMP_LINK_ARGS = {
    'gomp': ['-lgomp'],
    'omp': ['-lomp'],
    'msvc': [],
}

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


class build_ext_subclass(build_ext):
    """Add OpenMP flags to the one extension that uses ``prange``."""

    def build_extensions(self):
        omp_compile_args, omp_link_args = _omp_compile_link_args(self.compiler.compiler_type)
        for ext in self.extensions:
            if ext.name != OMP_EXTENSION:
                continue
            ext.extra_compile_args = list(ext.extra_compile_args or []) + omp_compile_args
            ext.extra_link_args = list(ext.extra_link_args or []) + omp_link_args
        build_ext.build_extensions(self)


def _omp_compile_link_args(compiler):
    """Get the OpenMP compile and link arguments for this compiler and platform."""
    try:
        use_omp = OMP_SETTING_TABLE[os.environ.get('USE_OMP', 'probe')]
    except KeyError:
        raise ValueError("Unknown USE_OMP value %r, expected one of: %s"
                         % (os.environ.get('USE_OMP'), ", ".join(sorted(OMP_SETTING_TABLE))))

    compile_args = []
    link_args = []
    if use_omp == "probe":
        use_omp, compile_args, link_args = _probe_omp_for_compiler_and_platform(compiler)

    print(f"Will use {use_omp} for OpenMP." if use_omp else "OpenMP support not available.")
    compile_args = compile_args + OMP_COMPILE_ARGS.get(use_omp, [])
    link_args = link_args + OMP_LINK_ARGS.get(use_omp, [])
    print(f"Compiler: {compiler} / OpenMP: {use_omp} / "
          f"OpenMP compile args: {compile_args} / OpenMP link args: {link_args}")
    return compile_args, link_args


def _probe_omp_for_compiler_and_platform(compiler):
    compile_args = []
    link_args = []
    if compiler == "msvc":
        use_omp = "msvc"
    elif _is_conda_interpreter():
        # Conda provides its own compiler which does support openmp
        use_omp = "gomp"
    elif _is_macOS():
        # OpenMP is not supported with system clang but homebrew and macports have libomp packages
        compile_args, link_args = _macOS_omp_options_from_probe()
        if not (compile_args or link_args):
            print("Probe for libomp failed, skipping use of OpenMP with clang.")
            print("It may be possible to build with OpenMP using USE_OMP=clang with CFLAGS and "
                  "LDFLAGS explicit settings to use libomp.")
            use_omp = None
        else:
            use_omp = "omp"
    else:
        use_omp = "gomp"
    return use_omp, compile_args, link_args


def _is_conda_interpreter():
    """Is the running interpreter from Anaconda, miniconda, or conda-forge?

    Modern conda-forge builds don't always mention conda in ``sys.version``, so
    the environment variable is checked first.

    """
    if os.environ.get("CONDA_PREFIX"):
        return True
    return 'conda' in sys.version or 'Continuum' in sys.version


def _is_macOS():
    return 'darwin' in sys.platform


def _macOS_omp_options_from_probe():
    """Get common include and library paths for libomp installation on macOS.

    For example ``(['-I/opt/local/include/libomp'], ['-L/opt/local/lib/libomp'])``.

    """
    for cmd in ["brew ls --verbose libomp", "port contents libomp"]:
        inc, lib = _compile_link_paths_from_manifest(cmd)
        if inc and lib:
            return [f"-I{inc}"], [f"-L{lib}"]
    return [], []


def _compile_link_paths_from_manifest(cmd):
    """Parse include and library paths from macOS package managers.

    Example executions::

        # Homebrew
        $ brew ls --verbose libomp
        /opt/homebrew/Cellar/libomp/15.0.7/include/omp.h
        /opt/homebrew/Cellar/libomp/15.0.7/lib/libomp.dylib

        # MacPorts
        $ port contents libomp
        Port libomp contains:
          /opt/local/include/libomp/omp.h
          /opt/local/lib/libomp/libomp.dylib

    """
    from subprocess import run
    query = run(cmd, shell=True, check=False, capture_output=True)
    if query.returncode != 0:
        return None, None
    manifest = query.stdout.decode("UTF-8")
    # find all the unique directories mentioned in the manifest
    dirs = set(os.path.split(filename)[0] for filename in re.findall(r'^\s*(/.*?)\s*$', manifest, re.MULTILINE))
    # find a unique libdir and incdir
    inc = tuple(d for d in dirs if re.search(r'/include(\W|$)', d))
    lib = tuple(d for d in dirs if re.search(r'/lib(\W|$)', d))
    # only return success if there's no ambiguity
    return (inc + lib) if len(inc) == 1 and len(lib) == 1 else (None, None)


cmdclass = versioneer.get_cmdclass(cmdclass={"build_ext": build_ext_subclass})

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
