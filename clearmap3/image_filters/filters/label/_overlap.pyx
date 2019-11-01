# cython: language_level = 3

import numpy as np
cimport numpy as np
cimport cython

from posix.mman cimport mmap, munmap, PROT_READ, PROT_WRITE, MAP_SHARED

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _overlap(label_0, label_1, output):
    """
    label_0: High thresholded, labeled, and size filtered
    label_1: Low Threshold, labeled

    All 3 input images must be the same shape.

    'output' can be the same image as 'label_1', in which case 'label_1' will be overwritten.
    """

    cdef long zmax = label_0.shape[0]
    cdef long ymax = label_0.shape[1]
    cdef long xmax = label_0.shape[2]

    cdef char *mmapped_label_0
    cdef int *mmapped_label_0_offset
    cdef char *mmapped_label_1
    cdef int *mmapped_label_1_offset
    cdef char *mmapped_output
    cdef int *mmapped_output_offset

    label_0_fd = open(label_0.filename, 'r+b')
    label_1_fd = open(label_1.filename, 'r+b')
    output_fd = open(output.filename, 'r+b')

    cdef unsigned char[::1 ] lookup = np.zeros(2**32-1, dtype='uint8')
    cdef unsigned char lookup_val

    cdef Py_ssize_t x, y, z

    cdef long idx
    cdef int label_0_val, label_1_val

    # Compare labels and generate lookup to keep labels that overlap
    for z in range(zmax):

        mmapped_label_0 = <char *> mmap(NULL,
                                        label_0.size * sizeof(int) + label_0.offset,
                                        PROT_READ|PROT_WRITE,
                                        MAP_SHARED,
                                        label_0_fd.fileno(),
                                        0)
        # Increment original pointer by <offset> bytes and then typecast to input datatype
        mmapped_label_0 += label_0.offset
        mmapped_label_0_offset = <int*> mmapped_label_0

        mmapped_label_1 = <char *> mmap(NULL,
                                        label_1.size * sizeof(int) + label_1.offset,
                                        PROT_READ|PROT_WRITE,
                                        MAP_SHARED,
                                        label_1_fd.fileno(),
                                        0)
        # Increment original pointer by <offset> bytes and then typecast to input datatype

        mmapped_label_1 += label_1.offset
        mmapped_label_1_offset = <int*> mmapped_label_1

        for y in range(ymax):
            for x in range(xmax):
                idx = (z * ymax * xmax) + (y * xmax) + x
                label_0_val = mmapped_label_0_offset[idx]
                label_1_val = mmapped_label_1_offset[idx]
                if label_0_val != 0 and label_1_val != 0:
                    lookup[label_1_val] = 1

        # Deallocate mapped memory after processing each frome
        mmapped_label_0 -= label_0.offset
        munmap(mmapped_label_0, label_0.size * sizeof(int) + label_0.offset)
        munmap(mmapped_label_0_offset, label_0.size * sizeof(int))
        mmapped_label_1 -= label_1.offset
        munmap(mmapped_label_1, label_1.size * sizeof(int) + label_1.offset)
        munmap(mmapped_label_1_offset, label_1.size * sizeof(int))

    # Remove labels that did not overlap
    for z in range(zmax):

        mmapped_label_1 = <char *> mmap(NULL,
                                        label_1.size * sizeof(int) + label_1.offset,
                                        PROT_READ|PROT_WRITE,
                                        MAP_SHARED,
                                        label_1_fd.fileno(),
                                        0)
        # Increment original pointer by <offset> bytes and then typecast to input datatype
        mmapped_label_1 += label_1.offset
        mmapped_label_1_offset = <int*> mmapped_label_1

        mmapped_output = <char *> mmap(NULL,
                                      output.size * sizeof(int) + output.offset,
                                      PROT_READ|PROT_WRITE,
                                      MAP_SHARED,
                                      output_fd.fileno(),
                                      0)
        # Increment original pointer by <offset> bytes and then typecast to input datatype
        mmapped_output += output.offset
        mmapped_output_offset = <int*> mmapped_output

        for y in range(ymax):
            for x in range(xmax):
                idx = (z * ymax * xmax) + (y * xmax) + x
                label_1_val = mmapped_label_1_offset[idx]
                lookup_val = lookup[label_1_val]
                if lookup_val != 0:
                    mmapped_output_offset[idx] = label_1_val
                else:
                    mmapped_output_offset[idx] = 0

        # Deallocate mapped memory after processing each frome
        mmapped_label_1 -= label_1.offset
        munmap(mmapped_label_1, label_1.size * sizeof(int) + label_1.offset)
        munmap(mmapped_label_1_offset, label_1.size * sizeof(int))
        mmapped_output -= output.offset
        munmap(mmapped_output, output.size * sizeof(int) + output.offset)
        munmap(mmapped_output_offset, output.size * sizeof(int))


