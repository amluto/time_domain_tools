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

import numpy
cimport numpy
numpy.import_array()
cimport cpython
cimport c_td_capture
cimport libc.string
import socket

cdef extern from "workaround.h":
    object SaneArrayFromData(numpy.dtype, int,
                             numpy.npy_intp *,
                             numpy.npy_intp *, void *, int,
                             object)

#cdef extern from "Python.h":
#    int PyBuffer_FillInfo(Py_buffer *view, object obj, void *buf,
#                          Py_ssize_t len, int readonly,
#                          int flags) except -1

cdef class CaptureSpec:
    cdef c_td_capture.CaptureSpec cspec
    cdef public unsigned short source_port
    cdef public unsigned short dest_port
    cdef public int channels
    cdef public int bits_per_sample
    cdef public bint want_native_format

    def __init__(CaptureSpec self):
        self.source_port = 0
        self.dest_port = 0
        self.channels = 0
        self.bits_per_sample = 0
        self.want_native_format = True
        #self.source_addr = '0.0.0.0'

    property source_addr:
        def __get__(CaptureSpec self):
            s = <char*>&self.cspec.source_addr
            return socket.inet_ntop(socket.AF_INET, s[0:4])
        def __set__(CaptureSpec self, val):
            out = socket.inet_pton(socket.AF_INET, val)
            libc.string.memcpy(&self.cspec.source_addr, <char*>out, 4)

    cdef sync(CaptureSpec self):
        self.cspec.source_port = self.source_port
        self.cspec.dest_port = self.dest_port
        self.cspec.channels = self.channels
        self.cspec.bits_per_sample = self.bits_per_sample
        self.cspec.want_native_format = self.want_native_format

def Capture(CaptureSpec spec, size_t nsamples, ntries = 2, verbose = False):
    spec.sync()

    if spec.bits_per_sample % 8 != 0:
        raise ValueError, 'bits_per_sample must be a multiple of 8'

    cdef size_t len = nsamples * spec.channels * spec.bits_per_sample / 8
    if len == 0:
        raise ValueError, 'You asked for zero bytes'

    cdef TimeDomainData data = TimeDomainData(len)

    cdef c_td_capture.CaptureResult result
    with nogil:
        result = c_td_capture.Capture(data.target, spec.cspec,
                                      ntries, verbose)

    if not result.ok:
        raise RuntimeError(result.error)

    data.channels = result.channels
    data.shift = result.shift

    if result.bits_per_sample == 8:
        dtype = numpy.dtype(numpy.int8)
    elif result.bits_per_sample == 16:
        dtype = numpy.dtype(numpy.int16)
    else:
        raise RuntimeError, "python wrapper can't handle data format"

    if result.endian == result.BIG:
        dtype = dtype.newbyteorder('>')
    elif result.endian == result.LITTLE:
        dtype = dtype.newbyteorder('<')
    else:
        raise RuntimeError, "unknown byte order"

    data.dtype = dtype
    return data

cdef class TimeDomainData:
    cdef c_td_capture.MemoryTarget *target
    cdef readonly numpy.npy_intp channels
    cdef numpy.dtype dtype
    cdef readonly int shift

    def __cinit__(TimeDomainData self):
        self.target = NULL
        self.channels = 1

    def __dealloc__(TimeDomainData self):
        del self.target

    def __init__(TimeDomainData self, size_t len,
                 numpy.dtype dtype = numpy.dtype(numpy.int8)):
        if not self.target:
            self.target = c_td_capture.NewMemoryTarget(len)
        self.dtype = dtype

    property data:
        """A numpy array containing the data."""
        def __get__(TimeDomainData self):
            # Set up a descriptor
            dt = self.dtype

            cdef numpy.npy_intp len[2]
            len[0] = self.target.len / (dt.itemsize * self.channels)
            len[1] = self.channels
            arr = SaneArrayFromData(dt, 2, &len[0], <numpy.npy_intp*>NULL,
                                    self.target.buffer,
                                    c_td_capture.NPY_BEHAVED,
                                    self)
            return arr

    def __reduce__(TimeDomainData self):
        raise TypeError('TimeDomainData cannot (yet?) be pickled')
