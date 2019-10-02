# cython: language_level = 3

import numpy as np
import sys

cimport numpy as cnp
cimport cython
from posix.mman cimport *

ctypedef cnp.int32_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _nonzero_coords(cnp.ndarray[DTYPE_t, ndim=3] image, coords_filename):
    """ generates linear indices of all non zero values in array
    """

    # Expects an image of type int32
    cdef long zmax = image.shape[0]
    cdef long ymax = image.shape[1]
    cdef long xmax = image.shape[2]

    cdef long z, y, x, idx
    cdef DTYPE_t image_val

    byteorder = sys.byteorder

    cdef char *mmapped_image
    cdef DTYPE_t *mmapped_image_offset

    cdef int i = 0

    with open(coords_filename, 'w+b') as coords_fd, open(image.filename, 'rb') as image_fd:

        for z in range(zmax):

            mmapped_image = <char *> mmap(NULL,
                                          image.size * sizeof(DTYPE_t) + image.offset,
                                          PROT_READ,
                                          MAP_SHARED,
                                          image_fd.fileno(),
                                          0)
            mmapped_image += image.offset
            mmapped_image_offset = <DTYPE_t *> mmapped_image

            for y in range(ymax):
                for x in range(xmax):
                    idx = (z * ymax * xmax) + (y * xmax) + x
                    image_val = mmapped_image_offset[idx]
                    if image_val != 0: # If value is nonzero, record the coordinate
                        coords_fd.write((idx).to_bytes(8, byteorder, signed=True))
                        i += 1

            munmap(mmapped_image, image.size * sizeof(DTYPE_t) + image.offset)

    return np.memmap(coords_filename, dtype='int64') # Return handle to coords file
