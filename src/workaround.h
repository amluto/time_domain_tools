#include <numpy/arrayobject.h>

static PyObject *SaneArrayFromData(PyArray_Descr *descr,
				   int nd, npy_intp *dims, npy_intp *stride,
				   void *data, int flags,
				   PyObject *base)
{
	Py_INCREF(descr);  /* NewFromDescr steals a reference. */
	PyObject *arr =
		PyArray_NewFromDescr(&PyArray_Type, descr, nd, dims, stride,
				     data, flags, NULL);
	if (!arr)
		return NULL;

	/* Give the array a reference to base. */
	Py_INCREF(base);
	PyArray_BASE((PyArrayObject*)arr) = base;

	return arr;
}

static MemoryTarget *NewMemoryTarget(size_t len)
{
	try {
		return new MemoryTarget(len);
	} catch (std::runtime_error &) {
		PyErr_SetString(PyExc_MemoryError,
				"failed to allocate MemoryTarget");
		return 0;
	}
}
