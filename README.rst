python-geotiepoints
===================

Python-geotiepoints is a python module that interpolates (and extrapolates if
needed) geographical tiepoints into a larger geographical grid. This is usefull
when the full resolution lon/lat grid is needed while only a lower resolution
grid of tiepoints was provided.

Some helper functions are provided to accomodate for satellite data, but the
package should be generic enough to be used for any kind of data.

In addition we have added a fast multilinear interpolation of regular gridded
data using Cython.

Adam & Martin
May 2017, Norrköping, Sweden
