"""Time-domain analysis functions"""

from __future__ import division
import scipy.signal
import math
import numpy
import weakref

def _asfloat(x):
    """Internal helper to coerce an array to floating-point."""
    if isinstance(x, numpy.ndarray) and x.dtype == numpy.float64:
        return x
    else:
        return numpy.asarray(x, numpy.float64)

def _as_float_or_complex(x):
    """Internal helper to coerce an array to floating-point or complex."""
    if (isinstance(x, numpy.ndarray)
        and x.dtype in (numpy.float64, numpy.complex128)):
        return x
    else:
        return numpy.asarray(x, numpy.complex128)

_time_coord_cache = weakref.WeakValueDictionary()

def _time_coord(fs, num_samples, offset = 0, dec = 1):
    spec = (num_samples, offset, dec)
    try:
        return _time_coord_cache[spec]
    except KeyError:
        ret = numpy.arange(offset, num_samples + offset, dec) / fs
        ret.setflags(write = False)
        ret = ret[:] # Make sure it never becomes writeable.
        _time_coord_cache[spec] = ret
        return ret

def mean_power(signal):
    signal = _as_float_or_complex(signal)
    if signal.ndim != 1:
        raise TypeError, 'signal must be one-dimensional'

    # This is the least inefficient way I can think of.
    return numpy.linalg.norm(signal, ord = 2)**2 / len(signal)

class ToneEstimate(object):
    def __init__(self, estimator, f, data, tone):
        self.__est = estimator
        self.__fs = estimator._fs
        self.__f = f
        self.__datalen = len(data)
        self.__t_data = None
        self.__t_tone = None

        self.tone = tone
        self.total_power = mean_power(data)
        self.inband_power = mean_power(self.tone) / 2
        self.inband_noise = ((self.total_power - self.inband_power)
                             / (1 - estimator.fractional_band)
                             * estimator.fractional_band)
        self.est_tone_power = self.inband_power - self.inband_noise

    # We compute t_data and t_tone as lazily as possible.
    @property
    def t_data(self):
        if self.__t_data is None:
            self.__t_data = _time_coord(self.__fs, self.__datalen)
        return self.__t_data
    @property
    def t_tone(self):
        if self.__t_tone is None:
            self.__t_tone = _time_coord(self.__fs,
                                        len(self.tone) * self.__est._dec,
                                        self.__est.offset, self.__est._dec)
        return self.__t_tone

class ToneEstimator(object):
    def __init__(self, fs, bw):
        self._fs = fs
        self._bw = bw
        self._dec = 1 # Decimation factor

        # Generate a symmetric FIR filter.
        # Some scipy versions give a bogus warning.  Ignore it.
        self._nyquist = fs / 2
        cutoff = bw / self._nyquist

        firlen = 10.0 / cutoff
        # Round to a power of 2 (so fftconvolve can be super fast)
        firlen = 2**int(numpy.ceil(numpy.log2(firlen)))

        old_err = numpy.seterr(invalid='ignore')
        self._filter = scipy.signal.firwin(
            firlen,
            cutoff = cutoff)
        numpy.seterr(**old_err)

        self.offset = (len(self._filter) - 1) / 2
        self.fractional_band = bw / self._nyquist

    def estimate_tone(self, f, data):
        """Returns a ToneEstimate for the cosine wave at frequency f.

        Note that the mean square of the tone is *twice* the mean square of
        the original cosine wave."""
        f = float(f)
        data = _asfloat(data)

        if data.ndim != 1:
            raise TypeError, 'data must be one-dimensional'

        baseband = 2 * data * numpy.exp(-2j * math.pi * f / self._fs
                                         * numpy.arange(0, len(data)))
        if len(data) < len(self._filter):
            raise (ValueError,
                   'You need at least %d samples for specified bandwidth'
                   % len(self._filter))

        tone = scipy.signal.fftconvolve(baseband, self._filter, mode='valid')
        if self._dec != 1:
            tone = tone[::self._dec]

        return ToneEstimate(self, f, data, tone)

class ToneDecimatingEstimator(ToneEstimator):
    def __init__(self, fs, bw):
        super(ToneDecimatingEstimator, self).__init__(fs, bw)

        cutoff = self._bw / self._nyquist
        self._dec = int(2.0 / cutoff) # Oversample by 2 to minimize aliasing.

    def estimate_tone(self, f, data):
        """Returns a ToneEstimate for the cosine wave at frequency f.

        Note that the mean square of the tone is *twice* the mean square of
        the original cosine wave."""
        f = float(f)
        data = _asfloat(data)

        if data.ndim != 1:
            raise TypeError, 'data must be one-dimensional'

        baseband = 2 * data * numpy.exp(-2j * math.pi * f / self._fs
                                         * numpy.arange(0, len(data)))
        if len(data) < len(self._filter):
            raise (ValueError,
                   'You need at least %d samples for specified bandwidth'
                   % len(self._filter))

        valid_len = (len(data) - len(self._filter) + self._dec) // self._dec

        tone = numpy.zeros(valid_len, dtype = baseband.dtype)
        for i in xrange(valid_len):
            pos = self._dec * i
            tone[i] = numpy.dot(self._filter,
                                baseband[pos:pos+len(self._filter)])

        return ToneEstimate(self, f, data, tone)
