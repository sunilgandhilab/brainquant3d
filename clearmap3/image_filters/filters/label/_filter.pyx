import numpy as np
cimport numpy as np

from libcpp.map cimport map
from libcpp.pair cimport pair

def _size_filter(int[:,:,::1] img, int minsize, int maxsize, int[:,:,::1] out):

    cdef int zmax = img.shape[0]
    cdef int ymax = img.shape[1]
    cdef int xmax = img.shape[2]

    cdef map[int, int] areas

    cdef map[int, int] mlookup
    cdef int[::1] arrlookup = np.zeros(2**32-1, dtype=np.int32)

    cdef pair[int, int] i

    cdef int val, lookup_val, counts

    cdef Py_ssize_t z, y, x

    # Count pixels in each label
    for z in range(zmax):
        for y in range(ymax):
            for x in range(xmax):
                val = img[z,y,x]
                if val != 0:
                    areas[val] +=1

    # Filter labels by size (counts)
    original_count = 0
    for i in areas:
        counts = i.second
        if maxsize >= counts >= minsize:
            mlookup[i.first] = i.second
            arrlookup[i.first] = 1
        original_count += 1

    # Apply Lookup
    for z in range(zmax):
        for y in range(ymax):
            for x in range(xmax):
                val = img[z,y,x]
                lookup_val = arrlookup[val]
                if lookup_val != 0:
                    out[z,y,x] = val

    # Return total original label count and dictionary of final labels and respective counts
    label_counts = {i.first: i.second for i in mlookup if i.second != 0}
    return original_count, label_counts


def _label_by_size(int[:,:,::1] img, int[:,:,::1] out):

    cdef int zmax = img.shape[0]
    cdef int ymax = img.shape[1]
    cdef int xmax = img.shape[2]

    cdef map[int, int] areas

    cdef map[int, int] mlookup
    cdef int[::1] arrlookup = np.zeros(2**32-1, dtype=np.int32)

    cdef pair[int, int] i

    cdef int val, lookup_val

    cdef Py_ssize_t z, y, x

    # Count pixels in each label
    for z in range(zmax):
        for y in range(ymax):
            for x in range(xmax):
                val = img[z,y,x]
                if val != 0:
                    areas[val] +=1

    # generate LUT
    for i in areas:
        mlookup[i.first] = i.second
        arrlookup[i.first] = i.second

    # Apply Lookup
    for z in range(zmax):
        for y in range(ymax):
            for x in range(xmax):
                val = img[z,y,x]
                lookup_val = arrlookup[val]
                out[z,y,x] = lookup_val
