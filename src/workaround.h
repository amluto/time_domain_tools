/*
 * Time domain tools for CASPER
 * Copyright (C) 2011 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * version 2 as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
 * USA.
 */

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
