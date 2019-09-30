# cython: language_level = 3

import numpy as np
import sys

cimport numpy as cnp
cimport cython
from posix.mman cimport *

ctypedef cnp.int32_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def nonzero_coords(cnp.ndarray[DTYPE_t, ndim=3] image, coords_filename):
    """ generates linear indices of all non zero values in array
    """

    # Expects an image of type int32
    cdef long zmax = image.shape[0]
    cdef long ymax = image.shape[1]
    cdef long xmax = image.shape[2]

    cdef long z, y, x, idx
    cdef DTYPE_t image_val

    byteorder = sys.byteorder

    cdef DTYPE_t *mmap_image

    cdef int i = 0

    with open(coords_filename, 'w+b') as coords_fd, open(image.filename, 'rb') as image_fd:
        mmap_image = <DTYPE_t *> mmap(NULL, image.size*sizeof(DTYPE_t), PROT_READ, MAP_SHARED,
                                      image_fd.fileno(), 0)

        for z in range(zmax):
            for y in range(ymax):
                for x in range(xmax):
                    idx = (z * ymax * xmax) + (y * xmax) + x + image.offset
                    image_val = mmap_image[idx]
                    if image_val != 0: # If value is nonzero, record the coordinate
                        coords_fd.write((idx).to_bytes(8, byteorder, signed=True))
                        i += 1


    return np.memmap(coords_filename, dtype='int64') # Return handle to coords file
