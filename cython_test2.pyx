# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False, initializedcheck=False, nonecheck=False
cimport cython
import numpy as np
cimport numpy as np

np.import_array()

def test_func():
    cdef np.ndarray[float, ndim=2] arr = np.zeros((5, 5), dtype=np.float32)
    cdef float[:, ::1] arr_view = arr
    t = Test(5.0)
    t.call_me(arr_view)


cdef class Test:

    cdef float _a

    def __cinit__(self, float a):
        self._a = a

    cdef void call_me(self, float[:, ::1] my_arr) noexcept:
        with nogil:
            self._call_me(my_arr)

    cdef void _call_me(self, float[:, ::1] my_arr) noexcept nogil:
        cdef Py_ssize_t idx
        cdef float[:, :] my_arr2 = _get_upper_left_corner(my_arr)
        for idx in range(my_arr2.shape[0]):
            my_arr2[idx, 0] = self._a


cdef inline float[:, :] _get_upper_left_corner(float[:, ::1] arr) noexcept nogil:
    return arr[:3, :3]
