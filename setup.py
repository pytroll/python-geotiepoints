#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2012 SMHI

# Author(s):

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

setup(name='pyresample',
      version="v0.1",
      description='Interpolation geographic tiepoints in Python',
      author='Martin Raspaud',
      author_email='martin.raspaud@smhi.se',
      package_dir = {'geo_interpolator': 'geo_interpolator'},
      packages = ['geo_interpolator'],      
      install_requires=['numpy', 'scipy'],
      zip_safe = False
      )

