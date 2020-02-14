# cython: language_level = 3

"""watershed.pyx - cython implementation of guts of watershed

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Original author: Lee Kamentsky
"""

import numpy as np
cimport numpy as cnp
cimport cython


from libc.math cimport sqrt

from posix.mman cimport *

ctypedef cnp.int32_t DTYPE_INT32_t
ctypedef cnp.int8_t DTYPE_BOOL_t

ctypedef fused INPUT_DTYPE:
    cnp.float32_t
    cnp.float64_t
    cnp.uint8_t
    cnp.uint16_t
    cnp.int32_t
    cnp.uint32_t

include "heap_watershed.pxi"


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
@cython.unraisable_tracebacks(False)
cdef inline double _euclid_dist(Py_ssize_t pt0, Py_ssize_t pt1,
                                cnp.intp_t[::1] strides) nogil:
    """Return the Euclidean distance between raveled points pt0 and pt1."""
    cdef double result = 0
    cdef double curr = 0
    for i in range(strides.shape[0]):
        curr = (pt0 // strides[i]) - (pt1 // strides[i])
        result += curr * curr
        pt0 = pt0 % strides[i]
        pt1 = pt1 % strides[i]
    return sqrt(result)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.unraisable_tracebacks(False)
cdef inline DTYPE_BOOL_t _diff_neighbors(DTYPE_INT32_t[::1] output,
                                         cnp.intp_t[::1] structure,
                                         unsigned char[::1] mask,
                                         Py_ssize_t index) nogil:
    """
    Return ``True`` and set ``mask[index]`` to ``False`` if the neighbors of
    ``index`` (as given by the offsets in ``structure``) have more than one
    distinct nonzero label.
    """
    cdef:
        Py_ssize_t i, neighbor_index
        DTYPE_INT32_t neighbor_label0, neighbor_label1
        Py_ssize_t nneighbors = structure.shape[0]

    if not mask[index]:
        return True

    neighbor_label0, neighbor_label1 = 0, 0
    for i in range(nneighbors):
        neighbor_index = structure[i] + index
        if mask[neighbor_index]:  # neighbor not a watershed line
            if not neighbor_label0:
                neighbor_label0 = output[neighbor_index]
            else:
                neighbor_label1 = output[neighbor_index]
                if neighbor_label1 and neighbor_label1 != neighbor_label0:
                    mask[index] = 0
                    return True
    return False

@cython.boundscheck(False)
@cython.wraparound(False)
def _watershed(cnp.ndarray[INPUT_DTYPE, ndim=3] image,
                      cnp.ndarray[cnp.int64_t, ndim=1] marker_locations,
                      cnp.intp_t[::1] structure,
                      cnp.ndarray[cnp.uint8_t, ndim=3] mask,
                      cnp.intp_t[::1] strides,
                      cnp.ndarray[cnp.int32_t, ndim=3] output,
                      DTYPE_BOOL_t invert=False):
    """Perform watershed algorithm using a raveled image and neighborhood.

    Parameters
    ----------

    image : numpy.memmap object
        3D image.
    marker_locations : numpy.memmap object
        The raveled coordinates of the initial markers (aka seeds) for the
        watershed. NOTE: these should *all* point to nonzero entries in the
        output, or the algorithm will never terminate and blow up your memory!
    structure : array of int
        A list of coordinate offsets to compute the raveled coordinates of each
        neighbor from the raveled coordinates of the current pixel.
    mask : numpy.memmap object
        An array of the same shape as `image` where each pixel contains a
        nonzero value if it is to be considered for flooding with watershed,
        zero otherwise. NOTE: it is *essential* that the border pixels (those
        with neighbors falling outside the volume) are all set to zero, or
        segfaults could occur.
    strides : array of int
        An array representing the number of steps to move along each dimension.
        This is used in computing the Euclidean distance between raveled
        indices.
    output : numpy.memmap object
        The output array, which must already contain nonzero entries at all the
        seed locations.
    invert : bool
        Parameter indicating whether to invert the image. Default: False.
    """
    cdef Heapitem elem
    cdef Heapitem new_elem
    cdef Py_ssize_t nneighbors = structure.shape[0]
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t age = 1
    cdef Py_ssize_t index = 0
    cdef Py_ssize_t neighbor_index = 0

    cdef Heap *hp = <Heap *> heap_from_numpy2()

    cdef char *mmapped_image
    cdef INPUT_DTYPE *mmapped_image_offset
    cdef char *mmapped_marker_locations
    cdef long *mmapped_marker_locations_offset
    cdef char *mmapped_mask
    cdef unsigned char *mmapped_mask_offset
    cdef char *mmapped_output
    cdef DTYPE_INT32_t *mmapped_output_offset

    image_fd = open(image.filename, 'r+b')
    marker_fd = open(marker_locations.filename, 'r+b')
    mask_fd = open(mask.filename, 'r+b')
    output_fd = open(output.filename, 'r+b')

    ############################# MMAP Files #############################
    mmapped_image = <char *> mmap(NULL,
                                  image.size * sizeof(INPUT_DTYPE) + image.offset,
                                  PROT_READ|PROT_WRITE,
                                  MAP_SHARED,
                                  image_fd.fileno(),
                                  0)
    mmapped_image += image.offset
    mmapped_image_offset = <INPUT_DTYPE *> mmapped_image

    mmapped_marker_locations = <char *> mmap(NULL,
                                             marker_locations.size * sizeof(long) + marker_locations.offset,
                                             PROT_READ|PROT_WRITE,
                                             MAP_SHARED,
                                             marker_fd.fileno(),
                                             0)
    mmapped_marker_locations += marker_locations.offset
    mmapped_marker_locations_offset = <long *> mmapped_marker_locations

    mmapped_mask = <char *> mmap(NULL,
                                 mask.size * sizeof(unsigned char) + mask.offset,
                                 PROT_READ|PROT_WRITE,
                                 MAP_SHARED,
                                 mask_fd.fileno(),
                                 0)
    mmapped_mask += mask.offset
    mmapped_mask_offset = <unsigned char *> mmapped_mask

    cdef unsigned char[::1] memview_mask = <unsigned char[:mask.size]> mmapped_mask_offset

    mmapped_output = <char *> mmap(NULL,
                                   output.size * sizeof(DTYPE_INT32_t) + output.offset,
                                   PROT_READ|PROT_WRITE,
                                   MAP_SHARED,
                                   output_fd.fileno(),
                                   0)
    mmapped_output += output.offset
    mmapped_output_offset = <DTYPE_INT32_t *> mmapped_output

    cdef DTYPE_INT32_t[::1] memview_output = <DTYPE_INT32_t [:output.size]> mmapped_output_offset
    ######################################################################

    cdef INPUT_DTYPE factor = -1 if invert else 1

    cdef long marker_size = marker_locations.size

    with nogil:
        for i in range(marker_size):
            index = mmapped_marker_locations_offset[i]
            elem.value = factor * mmapped_image_offset[index]
            elem.age = 0
            elem.index = index
            elem.source = index
            heappush(hp, &elem)

        while hp.items > 0:
            heappop(hp, &elem)

            for i in range(nneighbors):
                # get the flattened address of the neighbor
                neighbor_index = structure[i] + elem.index

                if not mmapped_mask_offset[neighbor_index]:
                    # this branch includes basin boundaries, aka watershed lines
                    # neighbor is not in mask
                    continue

                if mmapped_output_offset[neighbor_index]:
                    # pre-labeled neighbor is not added to the queue.
                    continue

                age += 1
                new_elem.value = factor * mmapped_image_offset[neighbor_index]

                mmapped_output_offset[neighbor_index] = mmapped_output_offset[elem.index]

                new_elem.age = age
                new_elem.index = neighbor_index
                new_elem.source = elem.source

                heappush(hp, &new_elem)

    mmapped_image -= image.offset
    munmap(mmapped_image, image.size * sizeof(INPUT_DTYPE) + image.offset)
    munmap(mmapped_image_offset, image.size * sizeof(INPUT_DTYPE))

    mmapped_marker_locations -= marker_locations.offset
    munmap(mmapped_marker_locations, marker_locations.size * sizeof(long) + marker_locations.offset)
    munmap(mmapped_marker_locations_offset, marker_locations.size * sizeof(long))

    mmapped_mask -= mask.offset
    munmap(mmapped_mask, mask.size * sizeof(unsigned char) + mask.offset)
    munmap(mmapped_mask_offset, mask.size * sizeof(unsigned char))

    mmapped_output -= output.offset
    munmap(mmapped_output, output.size * sizeof(DTYPE_INT32_t) + output.offset)
    munmap(mmapped_output_offset, output.size * sizeof(DTYPE_INT32_t))

    heap_done(hp)
