# -*- mode: python -*-

cdef extern from "td_capture.h":
    cppclass MemoryTarget:
        MemoryTarget(size_t)
        void *buffer
        size_t len

    struct in_addr:
        pass

    struct CaptureSpec:
        in_addr source_addr
        unsigned short source_port
        unsigned short dest_port

        int channels
        int bits_per_sample

        bint want_native_format

    struct CaptureResult:
        bint ok
        char *error

        int BIG # A lie, but it works
        int LITTLE

        int endian
        int channels
        int bits_per_sample
        int shift

    cdef CaptureResult Capture(MemoryTarget *target,
                               CaptureSpec spec,
                               int ntries,
                               bint verbose) nogil

    MemoryTarget *NewMemoryTarget(size_t len) except NULL

    enum: NPY_BEHAVED # Work around cython bug in numpy.pxd?
