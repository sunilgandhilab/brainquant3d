# cython: language_level = 3

cimport numpy as cnp
cimport cython

from posix.mman cimport *
from libc.math cimport sqrt

ctypedef fused DTYPE:
    cnp.float32_t
    cnp.float64_t
    cnp.uint8_t
    cnp.uint16_t
    cnp.int32_t
    cnp.uint32_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
def _standardize(cnp.ndarray[DTYPE, ndim=3] image, cnp.ndarray[cnp.float32_t, ndim=3] output):
    """Standardizes images.

    Parameters
cnp    ----------
    image: numpy.memmap array
        Input image.
    output: numpy.memmap array
        Output image.
    """

    cdef long zmax = image.shape[0]
    cdef long ymax = image.shape[1]
    cdef long xmax = image.shape[2]

    cdef char *mmapped_image
    cdef DTYPE *mmapped_image_offset
    cdef char *mmapped_output
    cdef cnp.float32_t *mmapped_output_offset

    image_fd = open(image.filename, 'rb')
    output_fd = open(output.filename, 'r+b')

    cdef DTYPE image_val

    cdef Py_ssize_t z, y, x
    cdef long idx

    cdef long numvals = image.size
    cdef double sumvals = 0.0
    cdef double mean, sumx, std

    # Calculate mean
    for z in range(zmax):
        # Allocate memory at the start of each new frame
        mmapped_image = <char *> mmap(NULL,
                                      image.size * sizeof(DTYPE) + image.offset,
                                      PROT_READ,
                                      MAP_SHARED,
                                      image_fd.fileno(),
                                      0)
        # Increment original pointer by <offset> bytes and then typecast to input datatype
        mmapped_image += image.offset
        mmapped_image_offset = <DTYPE*>mmapped_image

        for y in range(ymax):
            for x in range(xmax):
                idx = (z * ymax * xmax) + (y * xmax) + x
                image_val = mmapped_image_offset[idx]
                sumvals += image_val

        # Deallocate mapped memory after processing each frome
        mmapped_image -= image.offset
        munmap(mmapped_image, image.size * sizeof(DTYPE) + image.offset)
        munmap(mmapped_image_offset, image.size * sizeof(DTYPE))

    mean = sumvals / numvals

    # Calculate standard deviation
    for z in range(zmax):
        # Allocate memory at the start of each new frame
        mmapped_image = <char *> mmap(NULL,
                                      image.size * sizeof(DTYPE) + image.offset,
                                      PROT_READ,
                                      MAP_SHARED,
                                      image_fd.fileno(),
                                      0)
        # Increment original pointer by <offset> bytes and then typecast to input datatype
        mmapped_image += image.offset
        mmapped_image_offset = <DTYPE*>mmapped_image

        for y in range(ymax):
            for x in range(xmax):
                idx = (z * ymax * xmax) + (y * xmax) + x
                image_val = mmapped_image_offset[idx]
                sumx += (image_val - mean) ** 2

        # Deallocate mapped memory after processing each frome
        mmapped_image -= image.offset
        munmap(mmapped_image, image.size * sizeof(DTYPE) + image.offset)
        munmap(mmapped_image_offset, image.size * sizeof(DTYPE))

    std = sqrt(sumx / numvals)

    # Write standardized pixel values to new file
    for z in range(zmax):
        # Allocate memory at the start of each new frame
        mmapped_image = <char *> mmap(NULL,
                                      image.size * sizeof(DTYPE) + image.offset,
                                      PROT_READ,
                                      MAP_SHARED,
                                      image_fd.fileno(),
                                      0)
        # Increment original pointer by <offset> bytes and then typecast to input datatype
        mmapped_image += image.offset
        mmapped_image_offset = <DTYPE*>mmapped_image

        mmapped_output = <char *> mmap(NULL,
                                       output.size * sizeof(cnp.float32_t) + output.offset,
                                       PROT_READ|PROT_WRITE,
                                       MAP_SHARED,
                                       output_fd.fileno(),
                                       0)
        # Increment original pointer by <offset> bytes and then typecast to input datatype
        mmapped_output += output.offset
        mmapped_output_offset = <cnp.float32_t*>mmapped_output

        for y in range(ymax):
            for x in range(xmax):
                idx = (z * ymax * xmax) + (y * xmax) + x
                image_val = mmapped_image_offset[idx]
                mmapped_output_offset[idx] = (image_val - mean) / std

        # Deallocate mapped memory after processing each frome
        mmapped_image -= image.offset
        munmap(mmapped_image, image.size * sizeof(DTYPE) + image.offset)
        munmap(mmapped_image_offset, image.size * sizeof(DTYPE))


        mmapped_output -= output.offset
        munmap(mmapped_output, output.size * sizeof(cnp.float32_t) + output.offset)
        munmap(mmapped_output_offset, output.size * sizeof(cnp.float32_t))



