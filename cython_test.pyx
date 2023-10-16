# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False, initializedcheck=False, nonecheck=False
cimport cython
import numpy as np
cimport numpy as np

np.import_array()

def test_func():
    cdef np.ndarray[float, ndim=2] arr = np.zeros((5, 5), dtype=np.float32)
    cdef float[:, ::1] arr_view = arr
    _run(arr_view)

cdef void _run(float[:, ::1] arr_view) noexcept nogil:
    cdef float[:, :] tmp = _get_upper_left_corner(arr_view)

cdef inline float[:, :] _get_upper_left_corner(float[:, ::1] arr) noexcept nogil:
    return arr[:1, :1]
