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
from libc.math cimport sqrt

from posix.mman cimport *
from libcpp cimport bool

cimport numpy as cnp
cimport cython


ctypedef cnp.int32_t DTYPE_INT32_t
ctypedef cnp.int8_t DTYPE_BOOL_t


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
def watershed_3d(image,
                      marker_locations,
                      cnp.intp_t[::1] structure,
                      mask,
                      cnp.intp_t[::1] strides,
                      cnp.double_t compactness,
                      output,
                      DTYPE_BOOL_t wsl,
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
    compactness : float
        A value greater than 0 implements the compact watershed algorithm
        (see .py file).
    output : numpy.memmap object
        The output array, which must already contain nonzero entries at all the
        seed locations.
    wsl : bool
        Parameter indicating whether the watershed line is calculated.
        If wsl is set to True, the watershed line is calculated.
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
    cdef DTYPE_BOOL_t compact = (compactness > 0)

    cdef Heap *hp = <Heap *> heap_from_numpy2()

    ############################# MMAP Files #############################
    image_fd = open(image.filename, 'r+b')
    cdef long image_length = np.prod(image.shape)
    cdef unsigned short *mmapped_image = <unsigned short *> mmap(NULL, image_length * sizeof(unsigned short), PROT_READ, MAP_SHARED, image_fd.fileno(), image.offset)

    marker_fd = open(marker_locations.filename, 'r+b')
    cdef long marker_length = np.prod(marker_locations.shape)
    cdef long *mmapped_marker_locations = <long *> mmap(NULL, marker_length * sizeof(long), PROT_READ, MAP_SHARED, marker_fd.fileno(), 0)

    mask_fd = open(mask.filename, 'r+b')
    cdef long mask_length = np.prod(mask.shape)
    cdef unsigned char *mmapped_mask = <unsigned char *> mmap(NULL, mask_length * sizeof(unsigned char), PROT_READ, MAP_SHARED, mask_fd.fileno(), mask.offset)
    cdef unsigned char[::1] memview_mask = <unsigned char[:mask_length]> mmapped_mask

    output_fd = open(output.filename, 'r+b')
    cdef long output_length = np.prod(output.shape)
    cdef DTYPE_INT32_t *mmapped_output = <DTYPE_INT32_t *> mmap(NULL, output_length * sizeof(DTYPE_INT32_t), PROT_READ|PROT_WRITE, MAP_SHARED, output_fd.fileno(), output.offset)
    cdef DTYPE_INT32_t[::1] memview_output = <DTYPE_INT32_t [:output_length]> mmapped_output
    ######################################################################

    cdef long factor = -1 if invert else 1

    with nogil:
        for i in range(marker_length):
            index = mmapped_marker_locations[i]
            elem.value = factor * mmapped_image[index]
            elem.age = 0
            elem.index = index
            elem.source = index
            heappush(hp, &elem)

        while hp.items > 0:
            heappop(hp, &elem)

            if compact or wsl:
                # in the compact case, we need to label pixels as they come off
                # the heap, because the same pixel can be pushed twice, *and* the
                # later push can have lower cost because of the compactness.
                #
                # In the case of preserving watershed lines, a similar argument
                # applies: we can only observe that all neighbors have been labeled
                # as the pixel comes off the heap. Trying to do so at push time
                # is a bug.
                if mmapped_output[elem.index] and elem.index != elem.source:
                    # non-marker, already visited from another neighbor
                    continue
                if wsl:
                    # if the current element has different-labeled neighbors and we
                    # want to preserve watershed lines, we mask it and move on

                    #if _diff_neighbors(mmapped_output, structure, mmapped_mask, elem.index):
                    if _diff_neighbors(memview_output, structure, memview_mask, elem.index):
                        continue
                mmapped_output[elem.index] = mmapped_output[elem.source]

            for i in range(nneighbors):
                # get the flattened address of the neighbor
                neighbor_index = structure[i] + elem.index

                if not mmapped_mask[neighbor_index]:
                    # this branch includes basin boundaries, aka watershed lines
                    # neighbor is not in mask
                    continue

                if mmapped_output[neighbor_index]:
                    # pre-labeled neighbor is not added to the queue.
                    continue

                age += 1
                new_elem.value = factor * mmapped_image[neighbor_index]
                if compact:
                    new_elem.value += (compactness *
                                       _euclid_dist(neighbor_index, elem.source,
                                                    strides))
                elif not wsl:
                    # in the simplest watershed case (no compactness and no
                    # watershed lines), we can label a pixel at the time that
                    # we push it onto the heap, because it can't be reached with
                    # lower cost later.
                    # This results in a very significant performance gain, see:
                    # https://github.com/scikit-image/scikit-image/issues/2636
                    mmapped_output[neighbor_index] = mmapped_output[elem.index]
                new_elem.age = age
                new_elem.index = neighbor_index
                new_elem.source = elem.source

                heappush(hp, &new_elem)

    heap_done(hp)
