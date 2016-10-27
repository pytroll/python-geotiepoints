#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2012, 2013 Adam Dybbroe, Martin Raspaud

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

from setuptools import setup

setup(name='python-geotiepoints',
      version="v1.0.0",
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
      url="https://github.com/adybbroe/python-geotiepoints",
      packages = ['geotiepoints'],      
      install_requires=['numpy', 'scipy', 'pyresample', 'pandas'],
      zip_safe = False
      )

