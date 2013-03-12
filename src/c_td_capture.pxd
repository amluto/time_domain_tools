# -*- mode: python -*-

# Time domain tools for CASPER
# Copyright (C) 2011 Massachusetts Institute of Technology
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 2 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
# USA.

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
