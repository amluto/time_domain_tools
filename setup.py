# Run as:
#    python setup.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy.distutils.misc_util

ext_modules = [
    Extension("td_capture",
              sources = ["src/py_td_capture.pyx"],
              extra_objects = ["src/td_capture-td_lib.o", "src/td_capture-convert.o"],
              libraries = ["rt"],
              language = 'c++',
              include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs(),
              )
]

setup(
    name = 'TimeDomainTools',
    description = 'Time-domain capture and analysis tools',
    author = 'Andrew Lutomirski',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    py_modules = ['td_analysis'], # Ugly hack
)
