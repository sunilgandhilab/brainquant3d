# cython: language_level = 3

import numpy as np
cimport numpy as np
cimport cython

from posix.mman cimport mmap, PROT_READ, PROT_WRITE, MAP_SHARED

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def overlap(label_0, label_1, output):
    """
    label_0: High thresholded, labeled, and size filtered
    label_1: Low Threshold, labeled

    All 3 input images must be the same shape.

    'output' can be the same image as 'label_1', in which case 'label_1' will be overwritten.
    """

    cdef long zmax = label_0.shape[0]
    cdef long ymax = label_0.shape[1]
    cdef long xmax = label_0.shape[2]

    label_0_fd = open(label_0.filename, 'rb')
    cdef long label_0_length = np.prod(label_0.shape)
    cdef int *mmapped_label_0 = <int *> mmap(NULL, label_0_length * sizeof(int), PROT_READ, MAP_SHARED, label_0_fd.fileno(), label_0.offset)

    label_1_fd = open(label_1.filename, 'rb')
    cdef long label_1_length = np.prod(label_1.shape)
    cdef int *mmapped_label_1 = <int *> mmap(NULL, label_1_length * sizeof(int), PROT_READ, MAP_SHARED, label_1_fd.fileno(), label_1.offset)

    output_fd = open(output.filename, 'r+b')
    cdef long output_length = np.prod(output.shape)
    cdef int *mmapped_output = <int *> mmap(NULL, output_length * sizeof(int), PROT_READ|PROT_WRITE, MAP_SHARED, output_fd.fileno(), output.offset)

    cdef unsigned char[::1 ] lookup = np.zeros(2**32-1, dtype='uint8')
    cdef unsigned char lookup_val

    cdef Py_ssize_t x, y, z

    cdef long idx
    cdef int label_0_val, label_1_val

    # Compare labels and generate lookup to keep labels that overlap
    for z in range(zmax):
        for y in range(ymax):
            for x in range(xmax):
                idx = (z * ymax * xmax) + (y * xmax) + x
                label_0_val = mmapped_label_0[idx]
                label_1_val = mmapped_label_1[idx]
                if label_0_val != 0 and label_1_val != 0:
                    lookup[label_1_val] = 1

    # Remove labels that did not overlap
    for z in range(zmax):
        for y in range(ymax):
            for x in range(xmax):
                idx = (z * ymax * xmax) + (y * xmax) + x
                label_1_val = mmapped_label_1[idx]
                lookup_val = lookup[label_1_val]
                if lookup_val != 0:
                    mmapped_output[idx] = label_1_val
                else:
                    mmapped_output[idx] = 0

