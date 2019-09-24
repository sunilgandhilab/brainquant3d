# cython: language_level = 3

import numpy as np
cimport numpy as cnp
cimport cython

from posix.mman cimport *

from libcpp.limits cimport numeric_limits

ctypedef fused INPUT_DTYPE:
    cnp.uint8_t
    cnp.uint16_t
    cnp.int32_t
    cnp.uint32_t

ctypedef fused OUTPUT_DTYPE:
    cnp.uint8_t
    cnp.uint16_t
    cnp.int32_t
    cnp.uint32_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _threshold_3d(cnp.ndarray[INPUT_DTYPE, ndim=3] image, cnp.ndarray[OUTPUT_DTYPE, ndim=3] output, int val):
    """Performs a simple threshold on 'image'.

    Parameters
    ----------
    image: numpy.memmap array (3D) of integers
        Image to be thresholded. Image must have exactly 3 dimensions.
    val: integer or float
        Value to be used as the threshold.
    output: numpy.memmap array (3D)
        Binary matrix where values >= 'val' = max(dtype) and values < 'val' = 0.
    """

    cdef long zmax = image.shape[0]
    cdef long ymax = image.shape[1]
    cdef long xmax = image.shape[2]

    cdef INPUT_DTYPE *mmapped_image
    cdef OUTPUT_DTYPE *mmapped_output

    image_fd = open(image.filename, 'rb')
    output_fd = open(output.filename, 'r+b')

    cdef INPUT_DTYPE image_val

    cdef OUTPUT_DTYPE maxval = numeric_limits[OUTPUT_DTYPE].max()

    cdef Py_ssize_t z, y, x
    cdef long idx

    for z in range(zmax):
        # Allocate memory at the start of each new frame
        mmapped_image = <INPUT_DTYPE *> mmap(NULL, image.size*sizeof(INPUT_DTYPE), PROT_READ, MAP_SHARED, image_fd.fileno(), image.offset)
        mmapped_output = <OUTPUT_DTYPE *> mmap(NULL, output.size*sizeof(OUTPUT_DTYPE), PROT_READ|PROT_WRITE, MAP_SHARED, output_fd.fileno(), output.offset)

        for y in range(ymax):
            for x in range(xmax):
                idx = (z * ymax * xmax) + (y * xmax) + x
                image_val = mmapped_image[idx]
                if image_val < val:
                    mmapped_output[idx] = 0
                else:
                    mmapped_output[idx] = maxval

        # Deallocate mapped memory after processing each frome
        munmap(mmapped_image, image.size*sizeof(INPUT_DTYPE))
        munmap(mmapped_output, output.size*sizeof(OUTPUT_DTYPE))
