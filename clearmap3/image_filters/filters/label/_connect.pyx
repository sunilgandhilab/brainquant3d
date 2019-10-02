import numpy as np
cimport cython
cimport numpy as np

from libcpp.map cimport map
from libcpp.pair cimport pair

from _connect cimport *

DTYPE = np.int32

ctypedef np.int32_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _connect(unsigned char[:,:,::1] img, int[:,:,::1] out):

    cdef int zmax = img.shape[0]
    cdef int ymax = img.shape[1]
    cdef int xmax = img.shape[2]

    cdef long size = 2**32-1

    cdef int[::1] new_labels_lookup = np.zeros(size, dtype=DTYPE)
    cdef int[::1] lookup = np.zeros(size, dtype=DTYPE)

    cdef map[int, map[int, int]] rev_lookups

    cdef Mat slice_0, slice_1, _labeled_0, _labeled_1
    cdef int[:,::1] labeled_0 = np.zeros((ymax, xmax), dtype=DTYPE)
    cdef int[:,::1] labeled_1 = np.zeros((ymax, xmax), dtype=DTYPE)

    cdef int x, y, z, i, last_label, val, lookup_val, next_lookup_val

    cdef DTYPE_t a_val, b_val, out_val

    _labeled_0.create(ymax, xmax, CV_32S)
    _labeled_1.create(ymax, xmax, CV_32S)

    for z in range(zmax-1):
        # Read frames, convert to cv::Mat objects, label, then convert to memory-view
        if z == 0:
            np2cvMat(img[z], slice_0)
            last_label = connectedComponents(slice_0, _labeled_0)
            cvMat2np(_labeled_0, labeled_0)
            out[z] = labeled_0
        else:
            labeled_0 = out[z]
        np2cvMat(img[z+1], slice_1)
        connectedComponents(slice_1, _labeled_1)
        cvMat2np(_labeled_1, labeled_1)

        # Before generating lookup, shift labels to begin after last label from first slice
        new_labels_lookup = np.zeros(size, dtype=DTYPE) # Reset
        for y in range(ymax):
            for x in range(xmax):
                b_val = labeled_1[y,x]
                if b_val == 0:
                    pass
                elif not new_labels_lookup[b_val]:
                    last_label += 1
                    new_labels_lookup[b_val] = last_label
        for y in range(ymax):
            for x in range(xmax):
                b_val = labeled_1[y,x]
                lookup_val = new_labels_lookup[b_val]
                if b_val == 0:
                    pass
                elif lookup_val == 0:
                    out[z+1,y,x] = b_val
                else:
                    out[z+1,y,x] = lookup_val

        # Generate forward lookup
        for y in range(ymax):
            for x in range(xmax):
                a_val = labeled_0[y,x]
                b_val = out[z+1,y,x]
                if a_val == 0 or b_val == 0:
                    pass
                else:
                    lookup_val = lookup[b_val]
                    if lookup_val == 0:
                        lookup[b_val] = a_val

        # 1st pass to map secondary values
        for y in range(ymax):
            for x in range(xmax):
                b_val = out[z+1,y,x]
                lookup_val = lookup[b_val]
                if b_val == 0:
                    pass
                elif lookup_val == 0:
                    out[z+1,y,x] = b_val
                else:
                    out[z+1,y,x] = lookup_val

        # Generate reverse lookup
        for y in range(ymax):
            for x in range(xmax):
                a_val = labeled_0[y,x]
                out_val = out[z+1,y,x]
                if out_val == 0 or a_val == 0 or out_val == a_val:
                    pass
                else:
                    lookup_val = rev_lookups[z][a_val]
                    if lookup_val == 0:
                        rev_lookups[z][a_val] = out_val

    # Generate final lookup
    cdef int[::1] final_lookup = np.zeros(last_label+1, dtype=DTYPE)
    cdef pair[int, int] lookup_pair, next_lookup_pair
    cdef int lookup_idx

    for rev_lookup in rev_lookups:
        for lookup_pair in rev_lookup.second:
            lookup_key = lookup_pair.first
            lookup_val = lookup_pair.second
            final_lookup[lookup_key] = lookup_val
            next_lookup_key = lookup_val
            for next_lookup in rev_lookups:
                lookup_val = next_lookup.second[next_lookup_key]
                if lookup_val != 0:
                    final_lookup[lookup_key] = lookup_val
                    next_lookup_key = lookup_val


    # Apply final lookup
    for z in range(zmax):
        for y in range(ymax):
            for x in range(xmax):
                val = out[z,y,x]
                lookup_val = final_lookup[val]
                if val == 0:
                    pass
                elif lookup_val == 0:
                    out[z,y,x] = val
                else:
                    out[z,y,x] = lookup_val

    return last_label
