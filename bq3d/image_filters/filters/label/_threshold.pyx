# cython: language_level = 3

cimport numpy as cnp
cimport cython

from posix.mman cimport *

from libcpp.limits cimport numeric_limits

ctypedef fused INPUT_DTYPE:
    cnp.float32_t
    cnp.float64_t
    cnp.uint8_t
    cnp.uint16_t
    cnp.int32_t
    cnp.uint32_t

ctypedef fused OUTPUT_DTYPE:
    cnp.float32_t
    cnp.float64_t
    cnp.uint8_t
    cnp.uint16_t
    cnp.int32_t
    cnp.uint32_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _threshold(cnp.ndarray[INPUT_DTYPE, ndim=3] image,
                  cnp.ndarray[OUTPUT_DTYPE, ndim=3] output,
                  INPUT_DTYPE val):
    """Performs a simple threshold on 'image'.

    Parameters
cnp    ----------
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

    cdef char *mmapped_image
    cdef INPUT_DTYPE *mmapped_image_offset
    cdef char *mmapped_output
    cdef OUTPUT_DTYPE *mmapped_output_offset

    image_fd = open(image.filename, 'rb')
    output_fd = open(output.filename, 'r+b')

    cdef INPUT_DTYPE image_val
    cdef OUTPUT_DTYPE maxval = numeric_limits[OUTPUT_DTYPE].max()

    cdef Py_ssize_t z, y, x
    cdef long idx

    for z in range(zmax):
        # Allocate memory at the start of each new frame
        mmapped_image = <char *> mmap(NULL,
                                      image.size * sizeof(INPUT_DTYPE) + image.offset,
                                      PROT_READ,
                                      MAP_SHARED,
                                      image_fd.fileno(),
                                      0)
        # Increment original pointer by <offset> bytes and then typecast to input datatype
        mmapped_image += image.offset
        mmapped_image_offset = <INPUT_DTYPE*>mmapped_image

        mmapped_output = <char *> mmap(NULL,
                                       output.size * sizeof(OUTPUT_DTYPE) + output.offset,
                                       PROT_READ|PROT_WRITE,
                                       MAP_SHARED,
                                       output_fd.fileno(),
                                       0)
        # Increment original pointer by <offset> bytes and then typecast to input datatype
        mmapped_output += output.offset
        mmapped_output_offset = <OUTPUT_DTYPE*>mmapped_output

        for y in range(ymax):
            for x in range(xmax):
                idx = (z * ymax * xmax) + (y * xmax) + x
                image_val = mmapped_image_offset[idx]
                if image_val < val:
                    mmapped_output_offset[idx] = 0
                else:
                    mmapped_output_offset[idx] = maxval

        # Deallocate mapped memory after processing each frome
        mmapped_image -= image.offset
        munmap(mmapped_image, image.size * sizeof(INPUT_DTYPE) + image.offset)
        munmap(mmapped_image_offset, image.size * sizeof(INPUT_DTYPE))

        mmapped_output -= output.offset
        munmap(mmapped_output, output.size * sizeof(OUTPUT_DTYPE) + output.offset)
        munmap(mmapped_output_offset, output.size * sizeof(OUTPUT_DTYPE))

