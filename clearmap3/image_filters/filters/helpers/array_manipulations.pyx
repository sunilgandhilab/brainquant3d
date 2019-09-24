# cython: infer_types=True
# cython: language_level=3
import numpy as np
cimport cython

ctypedef fused im_type:
    unsigned long #np.uint64
    unsigned int # np.uint32
    unsigned short # np.uint16
    unsigned char # np.utin8

@cython.boundscheck(False)
@cython.wraparound(False)
def min_threshold_3d(im_type[:,:,:] arr, int min):
    """ Sets all variables below threshold to 0
     """

    cdef Py_ssize_t x_max = arr.shape[0]
    cdef Py_ssize_t y_max = arr.shape[1]
    cdef Py_ssize_t z_max = arr.shape[2]
    cdef Py_ssize_t x, y, z

    cdef im_type[:,:,:] arr_view = arr

    for z in range(z_max):
        for y in range(y_max):
            for x in range(x_max):
                if arr_view[x,y,z] < min:
                    arr_view[x,y,z] = 0

    return arr

@cython.boundscheck(False)
@cython.wraparound(False)
def incriment_nonzero_3d(im_type[:,:,:] arr, int incriment):
    """ Adds 'increment' to all nonzero values
     """

    cdef Py_ssize_t x_max = arr.shape[0]
    cdef Py_ssize_t y_max = arr.shape[1]
    cdef Py_ssize_t z_max = arr.shape[2]
    cdef Py_ssize_t x, y, z

    cdef im_type[:,:,:] arr_view = arr

    for z in range(z_max):
        for y in range(y_max):
            for x in range(x_max):
                if arr_view[x,y,z] != 0:
                    arr_view[x,y,z] = arr_view[x,y,z] + incriment

    return arr
