# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False, initializedcheck=False, nonecheck=False
cimport numpy as np

ctypedef fused floating:
    np.float32_t
    np.float64_t

cdef void lonlat2xyz(
        floating[:, ::1] lons,
        floating[:, ::1] lats,
        floating[:, :, ::1] xyz,
) noexcept nogil

cdef void xyz2lonlat(
        floating[:, :, ::1] xyz,
        floating[:, ::1] lons,
        floating[:, ::1] lats,
        bint low_lat_z=*,
        floating thr=*,
) noexcept nogil

cdef floating rad2deg(floating x) noexcept nogil
cdef floating deg2rad(floating x) noexcept nogil
