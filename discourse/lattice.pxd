cdef class Transition:
    cdef int to
    cdef list labels
    cdef list idxs
    cdef int position

