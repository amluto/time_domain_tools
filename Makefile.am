ACLOCAL_AMFLAGS = -I m4

td_capturedir=./
td_capture_PROGRAMS=td_capture
td_capture_SOURCES=src/td_capture.cc src/td_lib.cc src/convert.cc
td_capture_CXXFLAGS=-O2 -fPIC

# distutils dependency checks are rather broken, so delete its intermediates.
mostlyclean-local:
	rm -f *.so src/py_*.cpp
.PHONY: mostlyclean-local

python: all mostlyclean-local
	python setup.py build_ext --inplace
.PHONY: python
