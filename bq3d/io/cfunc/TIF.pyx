# cython: language_level = 3
cimport numpy as cnp

from posix.mman cimport *

ctypedef fused INPUT_DTYPE:
    cnp.float32_t
    cnp.float64_t
    cnp.uint8_t
    cnp.uint16_t
    cnp.int32_t
    cnp.uint32_t

cdef unsigned char * memmap(cnp.ndarray[INPUT_DTYPE] array):
    """ takes a python memory map array and converts it to a c mmap array.

    Args:
        array: numpy memmap array

    Returns:
        c array with offset
    """
    cdef unsigned char *mmapped_offset
    cdef char *mmapped
    mmapped = <char *> mmap(NULL,
                                 array.size * sizeof(INPUT_DTYPE) + array.offset,
                                 PROT_READ|PROT_WRITE,
                                 MAP_SHARED,
                                 array.fileno(),
                                 0)
    mmapped += array.offset
    mmapped_offset = <unsigned char *> mmapped

    return mmapped_offset