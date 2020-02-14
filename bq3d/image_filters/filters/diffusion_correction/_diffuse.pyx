# cython: language_level = 3

cimport numpy as cnp
cimport cython

from posix.mman cimport *
from libc.stdio cimport printf

ctypedef cnp.int32_t DTYPE_INT32_t
ctypedef cnp.int8_t DTYPE_BOOL_t
ctypedef cnp.float32_t DTYPE_OUT

ctypedef fused INPUT_DTYPE:
    cnp.float32_t
    cnp.float64_t
    cnp.uint8_t
    cnp.uint16_t
    cnp.int32_t
    cnp.uint32_t

include "heap_diffuse.pxi"

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _diffuse(cnp.ndarray[INPUT_DTYPE, ndim=3] mask,
                      cnp.ndarray[DTYPE_OUT, ndim=3] image,
                      cnp.ndarray[cnp.int64_t, ndim=1] seeds,
                      cnp.intp_t[::1] structure,
                      cnp.intp_t[::1] strides,
                      DTYPE_OUT threshold,
                      float k_const):
    """Perform watershed algorithm using a raveled image and neighborhood.

    Parameters
    ----------

    mask : numpy.memmap object
        3D image mask. will only diffuse in mask
    image : float, numpy.memmap object
        An array of the same shape as `mask` to diffuse. diffusion will occur from 'seeds'.
        values outside mask will be used for calculating a voxels value and should be set to 1.
    seeds : numpy.memmap object
        The raveled coordinates of the initial markers (aka seeds) for the
        watershed. NOTE: these should *all* point to nonzero entries in the
        image, or the algorithm will never terminate and blow up your memory!
    structure : array of int
        A list of coordinate offsets to compute the raveled coordinates of each
        neighbor from the raveled coordinates of the current pixel.
    strides : array of int
        An array representing the number of steps to move along each dimension.
        This is used in computing the Euclidean distance between raveled
        indices.
    threshold:
        value at which to stop computing diffusion
    k_const : float
        diffusion constant. higher will diffuse faster.
    """

    cdef Heapitem elem
    cdef Heapitem new_elem
    cdef Py_ssize_t nneighbors = structure.shape[0]
    cdef Py_ssize_t i,j = 0
    cdef Py_ssize_t age = 1
    cdef Py_ssize_t index = 0
    cdef Py_ssize_t neighbor_index = 0
    cdef DTYPE_OUT mean, value = 0

    cdef Heap *hp = <Heap *> heap_from_numpy2()

    cdef char *mmapped_mask
    cdef unsigned char *mmapped_mask_offset
    cdef char *mmapped_image
    cdef DTYPE_OUT *mmapped_image_offset

    mask_fd = open(mask.filename, 'r+b')
    image_fd = open(image.filename, 'r+b')

    ############################# MMAP Files #############################
    mmapped_mask = <char *> mmap(NULL,
                                 mask.size * sizeof(unsigned char) + mask.offset,
                                 PROT_READ|PROT_WRITE,
                                 MAP_SHARED,
                                 mask_fd.fileno(),
                                 0)
    mmapped_mask += mask.offset
    mmapped_mask_offset = <unsigned char *> mmapped_mask

    cdef unsigned char[::1] memview_mask = <unsigned char[:mask.size]> mmapped_mask_offset

    mmapped_image = <char *> mmap(NULL,
                                   image.size * sizeof(DTYPE_OUT) + image.offset,
                                   PROT_READ|PROT_WRITE,
                                   MAP_SHARED,
                                   image_fd.fileno(),
                                   0)
    mmapped_image += image.offset
    mmapped_image_offset = <DTYPE_OUT *> mmapped_image

    cdef DTYPE_OUT[::1] memview_image = <DTYPE_OUT [:image.size]> mmapped_image_offset
    ######################################################################

    cdef long nseeds = seeds.size

    # set diffusion value outside mask to 1. assumes full concentration outside.
    #TODO: do this once at start

    with nogil:
        for i in range(nseeds):
            index = seeds[i]
            mmapped_image_offset[index] = 1

            elem.parent_value = 1
            elem.age = 0
            elem.index = index
            elem.source = index
            heappush(hp, &elem)

        j = 0
        while hp.items > 0:
            heappop(hp, &elem)
            j += 1

            # calculate value
            mean = 0
            for i in range(nneighbors):
                # get the flattened address of the neighbor
                neighbor_index = structure[i] + elem.index
                mean += mmapped_image_offset[neighbor_index]
            mean = mean / nneighbors
            value = ((mean * k_const) + mmapped_image_offset[elem.index]) / 2

            if value < threshold:
                continue

            mmapped_image_offset[elem.index] = value

            #if age % 1000000 == 0:
            #    printf('surround: %f ', mean)
            #    printf('final: %f\n', value)

            for i in range(nneighbors):
                # get the flattened address of the neighbor
                neighbor_index = structure[i] + elem.index

                # if outside mask or already added to calculated
                if mmapped_mask_offset[neighbor_index] != 1:
                    continue

                # prevents indices in the heap from being readded
                # but needs to be reset to 0 on next iteration
                mmapped_mask_offset[neighbor_index] = 2

                age += 1
                new_elem.parent_value = value
                new_elem.age = age
                new_elem.index = neighbor_index
                new_elem.source = elem.source
                heappush(hp, &new_elem)

    print(f'reached theshold in: {j}')

    # cleanup values set to 2
    for i in range(mask.size):
        if mmapped_mask_offset[i] > 1:
            mmapped_mask_offset[i] = 1

    mmapped_mask -= mask.offset
    munmap(mmapped_mask, mask.size * sizeof(unsigned char) + mask.offset)
    munmap(mmapped_mask_offset, mask.size * sizeof(unsigned char))

    mmapped_image -= image.offset
    munmap(mmapped_image, image.size * sizeof(DTYPE_OUT) + image.offset)
    munmap(mmapped_image_offset, image.size * sizeof(DTYPE_OUT))

    heap_done(hp)
