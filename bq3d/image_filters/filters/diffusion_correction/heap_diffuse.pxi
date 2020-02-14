cimport numpy as cnp

cdef struct Heapitem:
    cnp.float64_t parent_value
    cnp.int32_t age
    Py_ssize_t index
    Py_ssize_t source


cdef inline int smaller(Heapitem *a, Heapitem *b) nogil:
    # iterates smallest to largest
    if a.age != b.age:
        return a.age < b.age # oldest has lower age
    return a.parent_value < b.parent_value

include "../helpers/heap/heap_general.pxi"
