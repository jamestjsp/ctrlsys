/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#define PY_SSIZE_T_CLEAN
/* Define unique symbol for NumPy Array API to avoid multiple static definitions */
#define PY_ARRAY_UNIQUE_SYMBOL SLICOT_ARRAY_API
#define NO_IMPORT_ARRAY

#include "py_wrappers.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>



PyObject* py_tg01ad(PyObject* self, PyObject* args) {
    const char *job;
    i32 l, n, m, p;
    f64 thresh;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj;
    PyArrayObject *a_array, *e_array, *b_array, *c_array;
    i32 info = 0;
    i32 lda, lde, ldb, ldc;

    if (!PyArg_ParseTuple(args, "siiiidOOOO", &job,
                          &l, &n, &m, &p, &thresh,
                          &a_obj, &e_obj, &b_obj, &c_obj)) {
        return NULL;
    }

    if (l < 0 || n < 0 || m < 0 || p < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative");
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || b_array == NULL || c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *e_dims = PyArray_DIMS(e_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    lda = (l > 0) ? (i32)a_dims[0] : 1;
    lde = (l > 0) ? (i32)e_dims[0] : 1;
    ldb = (m > 0 && l > 0) ? (i32)b_dims[0] : 1;
    ldc = (p > 0) ? (i32)c_dims[0] : 1;

    i32 ldwork = 3 * (l + n);
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    npy_intp lscale_dims[1] = {l};
    npy_intp rscale_dims[1] = {n};

    PyObject *lscale_array = PyArray_SimpleNew(1, lscale_dims, NPY_DOUBLE);
    PyObject *rscale_array = PyArray_SimpleNew(1, rscale_dims, NPY_DOUBLE);
    if (lscale_array == NULL || rscale_array == NULL) {
        Py_XDECREF(lscale_array);
        Py_XDECREF(rscale_array);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output arrays");
        return NULL;
    }

    f64 *lscale = (l > 0) ? (f64*)PyArray_DATA((PyArrayObject*)lscale_array) : NULL;
    f64 *rscale = (n > 0) ? (f64*)PyArray_DATA((PyArrayObject*)rscale_array) : NULL;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    tg01ad(job, l, n, m, p, thresh, a_data, lda, e_data, lde,
           b_data, ldb, c_data, ldc, lscale, rscale, dwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("(OOOOOOi)", a_array, e_array, b_array, c_array,
                                     lscale_array, rscale_array, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(lscale_array);
    Py_DECREF(rscale_array);

    return result;
}



PyObject* py_tg01bd(PyObject* self, PyObject* args) {
    const char *jobe, *compq, *compz;
    i32 ilo, ihi;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj, *q_obj, *z_obj;
    PyArrayObject *a_array = NULL, *e_array = NULL, *b_array = NULL, *c_array = NULL;
    PyArrayObject *q_array = NULL, *z_array = NULL;
    i32 info = 0;

    if (!PyArg_ParseTuple(args, "sssiiOOOOOO", &jobe, &compq, &compz,
                          &ilo, &ihi,
                          &a_obj, &e_obj, &b_obj, &c_obj, &q_obj, &z_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    z_array = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || b_array == NULL ||
        c_array == NULL || q_array == NULL || z_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(q_array);
        Py_XDECREF(z_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    i32 n = (i32)a_dims[0];
    i32 m = (PyArray_NDIM(b_array) >= 2) ? (i32)b_dims[1] : 0;
    i32 p = (i32)c_dims[0];

    i32 lda = n > 0 ? n : 1;
    i32 lde = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldq = n > 0 ? n : 1;
    i32 ldz = n > 0 ? n : 1;

    i32 jrow_len = ihi + 1 - ilo;
    i32 jcol_len = n + 1 - ilo;
    bool ilq = (*compq == 'I' || *compq == 'i' || *compq == 'V' || *compq == 'v');
    i32 ni = ilq ? n : jcol_len;
    i32 ldwork = jrow_len + (ni > m ? ni : m);
    if (ldwork < 1) ldwork = 1;
    ldwork = ldwork > n ? ldwork : n;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *q_data = (f64*)PyArray_DATA(q_array);
    f64 *z_data = (f64*)PyArray_DATA(z_array);

    tg01bd(jobe, compq, compz, n, m, p, ilo, ihi,
           a_data, lda, e_data, lde, b_data, ldb, c_data, ldc,
           q_data, ldq, z_data, ldz, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(q_array);
    PyArray_ResolveWritebackIfCopy(z_array);

    PyObject *result = Py_BuildValue("(OOOOOOi)",
                                     a_array, e_array, b_array, c_array,
                                     q_array, z_array, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);

    return result;
}



PyObject* py_tg01fd(PyObject* self, PyObject* args) {
    const char *compq, *compz, *joba;
    i32 l, n, m, p;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj;
    f64 tol;
    PyArrayObject *a_array, *e_array, *b_array, *c_array;
    i32 ranke = 0, rnka22 = 0, info = 0;
    i32 lda, lde, ldb, ldc, ldq, ldz;

    if (!PyArg_ParseTuple(args, "sssiiiiOOOOd", &compq, &compz, &joba,
                          &l, &n, &m, &p, &a_obj, &e_obj, &b_obj, &c_obj, &tol)) {
        return NULL;
    }

    if (l < 0 || n < 0 || m < 0 || p < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative");
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || b_array == NULL || c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *e_dims = PyArray_DIMS(e_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    lda = (l > 0) ? (i32)a_dims[0] : 1;
    lde = (l > 0) ? (i32)e_dims[0] : 1;
    ldb = (l > 0) ? (i32)b_dims[0] : 1;
    ldc = (p > 0) ? (i32)c_dims[0] : 1;
    ldq = (l > 0) ? l : 1;
    ldz = (n > 0) ? n : 1;

    i32 ln = (l < n) ? l : n;
    i32 temp1 = n + p;
    i32 temp2 = (3 * n - 1 > m) ? 3 * n - 1 : m;
    temp2 = (temp2 > l) ? temp2 : l;
    temp2 = ln + temp2;
    i32 ldwork = (temp1 > temp2) ? temp1 : temp2;
    ldwork = (ldwork > 1) ? ldwork : 1;

    i32 *iwork = (n > 0) ? (i32*)malloc(n * sizeof(i32)) : NULL;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    f64 *q = NULL;
    f64 *z = NULL;

    bool compq_needed = (compq[0] == 'I' || compq[0] == 'i' || compq[0] == 'U' || compq[0] == 'u');
    bool compz_needed = (compz[0] == 'I' || compz[0] == 'i' || compz[0] == 'U' || compz[0] == 'u');

    npy_intp q_dims[2] = {l, l};
    npy_intp z_dims[2] = {n, n};
    npy_intp q_strides[2] = {sizeof(f64), l * sizeof(f64)};
    npy_intp z_strides[2] = {sizeof(f64), n * sizeof(f64)};
    
    PyObject *q_array = NULL;
    PyObject *z_array = NULL;

    if (compq_needed && l > 0) {
        q_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (q_array == NULL) {
             free(iwork); free(dwork);
             Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(c_array);
             return PyErr_NoMemory();
        }
        q = (f64*)PyArray_DATA((PyArrayObject*)q_array);
        memset(q, 0, l * l * sizeof(f64));
    } else {
        q_array = PyArray_EMPTY(2, q_dims, NPY_DOUBLE, 1);
    }
    
    if (compz_needed && n > 0) {
        z_array = PyArray_New(&PyArray_Type, 2, z_dims, NPY_DOUBLE, z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (z_array == NULL) {
            free(iwork); free(dwork);
            Py_DECREF(q_array);
            Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(c_array);
            return PyErr_NoMemory();
        }
        z = (f64*)PyArray_DATA((PyArrayObject*)z_array);
        memset(z, 0, n * n * sizeof(f64));
    } else {
        z_array = PyArray_EMPTY(2, z_dims, NPY_DOUBLE, 1);
    }

    if (dwork == NULL || (n > 0 && iwork == NULL)) {
        free(iwork);
        free(dwork);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    tg01fd(compq, compz, joba, l, n, m, p, a_data, lda, e_data, lde,
           b_data, ldb, c_data, ldc, q, ldq, z, ldz, &ranke, &rnka22,
           tol, iwork, dwork, ldwork, &info);

    // q_array and z_array already created

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("(OOOOOOiii)", a_array, e_array, b_array, c_array,
                                     q_array, z_array, ranke, rnka22, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);

    return result;
}


PyObject* py_tg01hx(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"compq", "compz", "l", "n", "m", "p", "n1", "lbe",
                             "a", "e", "b", "c", "tol", "q", NULL};
    const char *compq, *compz;
    i32 l, n, m, p, n1, lbe;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj;
    PyObject *q_obj = Py_None;
    f64 tol;
    PyArrayObject *a_array = NULL, *e_array = NULL, *b_array = NULL, *c_array = NULL;
    i32 nr = 0, nrblck = 0, info = 0;
    i32 lda, lde, ldb, ldc, ldq, ldz;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssiiiiiiOOOOd|O", kwlist,
                                      &compq, &compz, &l, &n, &m, &p, &n1, &lbe,
                                      &a_obj, &e_obj, &b_obj, &c_obj, &tol, &q_obj)) {
        return NULL;
    }

    if (l < 0 || n < 0 || m < 0 || p < 0 || n1 < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative");
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || b_array == NULL || c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *e_dims = PyArray_DIMS(e_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    lda = (l > 0) ? (i32)a_dims[0] : 1;
    lde = (l > 0) ? (i32)e_dims[0] : 1;
    ldb = (l > 0) ? (i32)b_dims[0] : 1;
    ldc = (p > 0) ? (i32)c_dims[0] : 1;
    ldq = (l > 0) ? l : 1;
    ldz = (n > 0) ? n : 1;

    i32 dwork_size = n;
    dwork_size = (dwork_size > l) ? dwork_size : l;
    dwork_size = (dwork_size > 2 * m) ? dwork_size : 2 * m;
    if (dwork_size < 1) dwork_size = 1;

    i32 *iwork = (m > 0) ? (i32*)malloc(m * sizeof(i32)) : NULL;
    f64 *dwork = (f64*)malloc(dwork_size * sizeof(f64));
    i32 *rtau = (n1 > 0) ? (i32*)malloc(n1 * sizeof(i32)) : NULL;
    f64 *q = NULL;
    f64 *z = NULL;

    bool compq_needed = (compq[0] == 'I' || compq[0] == 'i' || compq[0] == 'U' || compq[0] == 'u');
    bool compz_needed = (compz[0] == 'I' || compz[0] == 'i' || compz[0] == 'U' || compz[0] == 'u');

    npy_intp q_dims[2] = {l, l};
    npy_intp z_dims[2] = {n, n};
    npy_intp rtau_dims[1] = {n1};
    npy_intp q_strides[2] = {sizeof(f64), l * sizeof(f64)};
    npy_intp z_strides[2] = {sizeof(f64), n * sizeof(f64)};

    PyObject *q_array = NULL;
    PyObject *z_array = NULL;
    PyObject *rtau_array = NULL;

    if (compq_needed && l > 0) {
        q_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (q_array == NULL) {
             free(iwork); free(dwork); free(rtau);
             Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(c_array);
             return PyErr_NoMemory();
        }
        q = (f64*)PyArray_DATA((PyArrayObject*)q_array);
        memset(q, 0, l * l * sizeof(f64));

        if (q_obj != Py_None && (compq[0] == 'U' || compq[0] == 'u')) {
            PyArrayObject *q_input = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
            if (q_input != NULL) {
                // Should check dimensions of q_input? Input validation is usually good.
                // Assuming it matches l*l.
                memcpy(q, PyArray_DATA(q_input), l * l * sizeof(f64));
                Py_DECREF(q_input);
            }
        }
    } else {
        q_array = PyArray_EMPTY(2, q_dims, NPY_DOUBLE, 1);
    }

    if (compz_needed && n > 0) {
        z_array = PyArray_New(&PyArray_Type, 2, z_dims, NPY_DOUBLE, z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (z_array == NULL) {
             free(iwork); free(dwork); free(rtau);
             Py_DECREF(q_array);
             Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(c_array);
             return PyErr_NoMemory();
        }
        z = (f64*)PyArray_DATA((PyArrayObject*)z_array);
        memset(z, 0, n * n * sizeof(f64));
    } else {
        z_array = PyArray_EMPTY(2, z_dims, NPY_DOUBLE, 1);
    }

    if (dwork == NULL || (m > 0 && iwork == NULL) || (n1 > 0 && rtau == NULL)) {
        free(iwork);
        free(dwork);
        free(rtau);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    tg01hx(compq, compz, l, n, m, p, n1, lbe, a_data, lda, e_data, lde,
           b_data, ldb, c_data, ldc, q, ldq, z, ldz, &nr, &nrblck, rtau,
           tol, iwork, dwork, &info);

    // q_array and z_array already created

    if (n1 > 0) {
        rtau_array = PyArray_SimpleNew(1, rtau_dims, NPY_INT32);
        memcpy(PyArray_DATA((PyArrayObject*)rtau_array), rtau, n1 * sizeof(i32));
        free(rtau);
    } else {
        rtau_array = PyArray_EMPTY(1, rtau_dims, NPY_INT32, 0);
    }

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("(OOOOOOiiOi)", a_array, e_array, b_array, c_array,
                                     q_array, z_array, nr, nrblck, rtau_array, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);
    Py_DECREF(rtau_array);

    return result;
}

PyObject* py_tg01dd(PyObject* self, PyObject* args) {
    const char *compz;
    PyObject *a_obj, *e_obj, *c_obj, *z_obj = Py_None;
    PyArrayObject *a_array = NULL, *e_array = NULL, *c_array = NULL;
    i32 info = 0;

    if (!PyArg_ParseTuple(args, "sOOO|O", &compz, &a_obj, &e_obj, &c_obj, &z_obj)) {
        return NULL;
    }

    char compz_char = toupper(compz[0]);
    bool compute_z = (compz_char == 'I' || compz_char == 'U');

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(c_array);
        PyErr_SetString(PyExc_TypeError, "Failed to convert input arrays");
        return NULL;
    }

    i32 l = (i32)PyArray_DIM(a_array, 0);
    i32 n = (i32)PyArray_DIM(a_array, 1);
    i32 p = (i32)PyArray_DIM(c_array, 0);

    if (PyArray_DIM(e_array, 0) != l || PyArray_DIM(e_array, 1) != n) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_ValueError, "E must have same dimensions as A");
        return NULL;
    }
    if (PyArray_DIM(c_array, 1) != n) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_ValueError, "C must have same number of columns as A");
        return NULL;
    }

    i32 lda = l > 1 ? l : 1;
    i32 lde = l > 1 ? l : 1;
    i32 ldc = p > 1 ? p : 1;
    i32 ldz = compute_z ? (n > 1 ? n : 1) : 1;

    PyArrayObject *z_array = NULL;
    f64 *z_data = NULL;

    if (compute_z) {
        if (compz_char == 'U') {
            if (z_obj == Py_None) {
                Py_DECREF(a_array);
                Py_DECREF(e_array);
                Py_DECREF(c_array);
                PyErr_SetString(PyExc_ValueError, "Z matrix required when compz='U'");
                return NULL;
            }
            z_array = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE,
                                                       NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            if (z_array == NULL) {
                Py_DECREF(a_array);
                Py_DECREF(e_array);
                Py_DECREF(c_array);
                PyErr_SetString(PyExc_TypeError, "Failed to convert Z array");
                return NULL;
            }
            if (PyArray_DIM(z_array, 0) != n || PyArray_DIM(z_array, 1) != n) {
                Py_DECREF(a_array);
                Py_DECREF(e_array);
                Py_DECREF(c_array);
                Py_DECREF(z_array);
                PyErr_SetString(PyExc_ValueError, "Z must be N-by-N");
                return NULL;
            }
            z_data = (f64*)PyArray_DATA(z_array);
        } else {
            npy_intp z_dims[2] = {n, n};
            npy_intp z_strides[2] = {sizeof(f64), n * sizeof(f64)};
            z_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, z_dims, NPY_DOUBLE, z_strides,
                                                  NULL, 0, NPY_ARRAY_FARRAY, NULL);
            if (z_array == NULL) {
                Py_DECREF(a_array);
                Py_DECREF(e_array);
                Py_DECREF(c_array);
                PyErr_SetString(PyExc_MemoryError, "Failed to allocate Z array");
                return NULL;
            }
            z_data = (f64*)PyArray_DATA(z_array);
        }
    }

    i32 ln = l < n ? l : n;
    i32 max_lnp = l > n ? l : n;
    max_lnp = max_lnp > p ? max_lnp : p;
    i32 ldwork = ln + max_lnp;
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(c_array);
        Py_XDECREF(z_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    tg01dd(compz, l, n, p, a_data, lda, e_data, lde, c_data, ldc,
           z_data, ldz, dwork, ldwork, &info);

    free(dwork);

    if (info < 0) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(c_array);
        Py_XDECREF(z_array);
        PyErr_Format(PyExc_ValueError, "Illegal value for argument %d", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    if (compz_char == 'U' && z_array != NULL) {
        PyArray_ResolveWritebackIfCopy(z_array);
    }

    PyObject *result;
    if (compute_z) {
        result = Py_BuildValue("(OOOOi)", a_array, e_array, c_array, z_array, info);
        Py_DECREF(z_array);
    } else {
        result = Py_BuildValue("(OOOOi)", a_array, e_array, c_array, Py_None, info);
    }

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(c_array);

    return result;
}


PyObject* py_tg01az(PyObject* self, PyObject* args) {
    const char *job;
    i32 l, n, m, p;
    f64 thresh;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj;
    PyArrayObject *a_array, *e_array, *b_array, *c_array;
    i32 info = 0;
    i32 lda, lde, ldb, ldc;

    if (!PyArg_ParseTuple(args, "siiiidOOOO", &job,
                          &l, &n, &m, &p, &thresh,
                          &a_obj, &e_obj, &b_obj, &c_obj)) {
        return NULL;
    }

    if (l < 0 || n < 0 || m < 0 || p < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative");
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_COMPLEX128,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_COMPLEX128,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_COMPLEX128,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_COMPLEX128,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || b_array == NULL || c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *e_dims = PyArray_DIMS(e_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    lda = (l > 0) ? (i32)a_dims[0] : 1;
    lde = (l > 0) ? (i32)e_dims[0] : 1;
    ldb = (m > 0 && l > 0) ? (i32)b_dims[0] : 1;
    ldc = (p > 0) ? (i32)c_dims[0] : 1;

    i32 ldwork = 3 * (l + n);
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    npy_intp lscale_dims[1] = {l};
    npy_intp rscale_dims[1] = {n};

    PyObject *lscale_array = PyArray_SimpleNew(1, lscale_dims, NPY_DOUBLE);
    PyObject *rscale_array = PyArray_SimpleNew(1, rscale_dims, NPY_DOUBLE);
    if (lscale_array == NULL || rscale_array == NULL) {
        Py_XDECREF(lscale_array);
        Py_XDECREF(rscale_array);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output arrays");
        return NULL;
    }

    f64 *lscale = (l > 0) ? (f64*)PyArray_DATA((PyArrayObject*)lscale_array) : NULL;
    f64 *rscale = (n > 0) ? (f64*)PyArray_DATA((PyArrayObject*)rscale_array) : NULL;

    c128 *a_data = (c128*)PyArray_DATA(a_array);
    c128 *e_data = (c128*)PyArray_DATA(e_array);
    c128 *b_data = (c128*)PyArray_DATA(b_array);
    c128 *c_data = (c128*)PyArray_DATA(c_array);

    tg01az(job, l, n, m, p, thresh, a_data, lda, e_data, lde,
           b_data, ldb, c_data, ldc, lscale, rscale, dwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("(OOOOOOi)", a_array, e_array, b_array, c_array,
                                     lscale_array, rscale_array, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(lscale_array);
    Py_DECREF(rscale_array);

    return result;
}


PyObject* py_tg01cd(PyObject* self, PyObject* args) {
    const char *compq;
    PyObject *a_obj, *e_obj, *b_obj;
    PyObject *q_obj = NULL;
    PyArrayObject *a_array, *e_array, *b_array;
    i32 info = 0;

    if (!PyArg_ParseTuple(args, "sOOO|O", &compq, &a_obj, &e_obj, &b_obj, &q_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || b_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    i32 l = (i32)a_dims[0];
    i32 n = (PyArray_NDIM(a_array) > 1) ? (i32)a_dims[1] : 0;
    i32 m = (PyArray_NDIM(b_array) > 1) ? (i32)b_dims[1] : 0;

    i32 lda = (l > 0) ? l : 1;
    i32 lde = (l > 0) ? l : 1;
    i32 ldb = (l > 0) ? l : 1;
    i32 ldq = (l > 0) ? l : 1;

    char compq_c = toupper(compq[0]);
    bool compq_needed = (compq_c == 'I' || compq_c == 'U');

    i32 ln = (l < n) ? l : n;
    i32 maxlnm = l;
    if (n > maxlnm) maxlnm = n;
    if (m > maxlnm) maxlnm = m;
    i32 ldwork = (ln + maxlnm > 1) ? ln + maxlnm : 1;
    ldwork *= 4;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    npy_intp q_dims[2] = {l, l};
    npy_intp q_strides[2] = {sizeof(f64), l * sizeof(f64)};
    PyObject *q_array = NULL;
    f64 *q = NULL;

    if (compq_needed && l > 0) {
        q_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (q_array == NULL) {
            free(dwork);
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(b_array);
            return PyErr_NoMemory();
        }
        q = (f64*)PyArray_DATA((PyArrayObject*)q_array);
        memset(q, 0, l * l * sizeof(f64));

        if (compq_c == 'U' && q_obj != NULL && q_obj != Py_None) {
            PyArrayObject *q_input = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
            if (q_input != NULL) {
                memcpy(q, PyArray_DATA(q_input), l * l * sizeof(f64));
                Py_DECREF(q_input);
            }
        }
    } else {
        q_array = PyArray_EMPTY(2, q_dims, NPY_DOUBLE, 1);
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    tg01cd(compq, l, n, m, a_data, lda, e_data, lde, b_data, ldb,
           q, ldq, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    PyObject *result = Py_BuildValue("(OOOOi)", a_array, e_array, b_array, q_array, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(q_array);

    return result;
}


PyObject* py_tg01gd(PyObject* self, PyObject* args) {
    const char *jobs;
    i32 l, n, m, p;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj, *d_obj;
    f64 tol;
    PyArrayObject *a_array, *e_array, *b_array, *c_array, *d_array;
    i32 lr = 0, nr = 0, ranke = 0, infred = 0, info = 0;
    i32 lda, lde, ldb, ldc, ldd;

    if (!PyArg_ParseTuple(args, "siiiiOOOOOd", &jobs,
                          &l, &n, &m, &p, &a_obj, &e_obj, &b_obj, &c_obj, &d_obj, &tol)) {
        return NULL;
    }

    if (l < 0 || n < 0 || m < 0 || p < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative");
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || b_array == NULL ||
        c_array == NULL || d_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *e_dims = PyArray_DIMS(e_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);
    npy_intp *d_dims = PyArray_DIMS(d_array);

    lda = (l > 0) ? (i32)a_dims[0] : 1;
    lde = (l > 0) ? (i32)e_dims[0] : 1;
    ldb = (l > 0) ? (i32)b_dims[0] : 1;
    ldc = (p > 0) ? (i32)c_dims[0] : 1;
    ldd = (p > 0) ? (i32)d_dims[0] : 1;

    i32 ln = (l < n) ? l : n;
    i32 temp1 = n + p;
    i32 temp2 = 3 * n - 1;
    temp2 = (temp2 > m) ? temp2 : m;
    temp2 = (temp2 > l) ? temp2 : l;
    temp2 = (ln > 0) ? ln + temp2 : 1;
    i32 lwrmin = (temp1 > temp2) ? temp1 : temp2;
    lwrmin = (lwrmin > 1) ? lwrmin : 1;

    i32 lspace_size = lwrmin + l * (2 * n + m) + p * n;
    i32 ldwork = (lspace_size > lwrmin) ? lspace_size : lwrmin;
    ldwork = (ldwork > 1) ? ldwork : 1;

    i32 *iwork = (n > 0) ? (i32*)malloc(n * sizeof(i32)) : NULL;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (dwork == NULL || (n > 0 && iwork == NULL)) {
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);

    tg01gd(jobs, l, n, m, p, a_data, lda, e_data, lde,
           b_data, ldb, c_data, ldc, d_data, ldd,
           &lr, &nr, &ranke, &infred, tol, iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    PyObject *result = Py_BuildValue("(OOOOOiiiii)", a_array, e_array, b_array, c_array, d_array,
                                     lr, nr, ranke, infred, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    return result;
}


PyObject* py_tg01ed(PyObject* self, PyObject* args) {
    const char *joba;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj;
    f64 tol;
    PyArrayObject *a_array, *e_array, *b_array, *c_array;
    i32 ranke = 0, rnka22 = 0, info = 0;

    if (!PyArg_ParseTuple(args, "sOOOOd", &joba, &a_obj, &e_obj, &b_obj, &c_obj, &tol)) {
        return NULL;
    }

    char joba_c = toupper(joba[0]);
    if (joba_c != 'N' && joba_c != 'R') {
        PyErr_SetString(PyExc_ValueError, "joba must be 'N' or 'R'");
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || b_array == NULL || c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    i32 l = (i32)a_dims[0];
    i32 n = (PyArray_NDIM(a_array) > 1) ? (i32)a_dims[1] : 0;
    i32 m = (PyArray_NDIM(b_array) > 1 && b_dims[0] > 0) ? (i32)b_dims[1] : 0;
    i32 p = (i32)c_dims[0];

    i32 lda = (l > 0) ? l : 1;
    i32 lde = (l > 0) ? l : 1;
    i32 ldb = (l > 0) ? l : 1;
    i32 ldc = (p > 0) ? p : 1;
    i32 ldq = (l > 0) ? l : 1;
    i32 ldz = (n > 0) ? n : 1;

    i32 ln = (l < n) ? l : n;
    i32 temp1 = 3 * ln + ((l > n) ? l : n);
    i32 temp2 = 5 * ln;
    temp1 = (temp1 > temp2) ? temp1 : temp2;
    temp1 = (temp1 > m) ? temp1 : m;
    temp1 = (temp1 > p) ? temp1 : p;
    i32 ldwork = ln + temp1;
    if (ldwork < 1) ldwork = 1;
    ldwork *= 2;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    npy_intp q_dims[2] = {l, l};
    npy_intp q_strides[2] = {sizeof(f64), l * sizeof(f64)};
    npy_intp z_dims[2] = {n, n};
    npy_intp z_strides[2] = {sizeof(f64), n * sizeof(f64)};

    PyObject *q_array = NULL;
    PyObject *z_array = NULL;
    f64 *q = NULL;
    f64 *z = NULL;

    if (l > 0) {
        q_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (q_array == NULL) {
            free(dwork);
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            return PyErr_NoMemory();
        }
        q = (f64*)PyArray_DATA((PyArrayObject*)q_array);
    } else {
        q_array = PyArray_EMPTY(2, q_dims, NPY_DOUBLE, 1);
        if (q_array == NULL) {
            free(dwork);
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            return PyErr_NoMemory();
        }
    }

    if (n > 0) {
        z_array = PyArray_New(&PyArray_Type, 2, z_dims, NPY_DOUBLE, z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (z_array == NULL) {
            free(dwork);
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            Py_DECREF(q_array);
            return PyErr_NoMemory();
        }
        z = (f64*)PyArray_DATA((PyArrayObject*)z_array);
    } else {
        z_array = PyArray_EMPTY(2, z_dims, NPY_DOUBLE, 1);
        if (z_array == NULL) {
            free(dwork);
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            Py_DECREF(q_array);
            return PyErr_NoMemory();
        }
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    tg01ed(joba, l, n, m, p, a_data, lda, e_data, lde,
           b_data, ldb, c_data, ldc, q, ldq, z, ldz,
           &ranke, &rnka22, tol, dwork, ldwork, &info);

    free(dwork);

    if (info < 0) {
        PyArray_ResolveWritebackIfCopy(a_array);
        PyArray_ResolveWritebackIfCopy(e_array);
        PyArray_ResolveWritebackIfCopy(b_array);
        PyArray_ResolveWritebackIfCopy(c_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        PyErr_Format(PyExc_RuntimeError, "tg01ed: illegal value in argument %d", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("(OOOOOOiii)", a_array, e_array, b_array, c_array,
                                     q_array, z_array, ranke, rnka22, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);

    return result;
}


PyObject* py_tg01fz(PyObject* self, PyObject* args) {
    const char *compq, *compz, *joba;
    i32 l, n, m, p;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj;
    f64 tol;
    PyArrayObject *a_array, *e_array, *b_array, *c_array;
    i32 ranke = 0, rnka22 = 0, info = 0;
    i32 lda, lde, ldb, ldc, ldq, ldz;

    if (!PyArg_ParseTuple(args, "sssiiiiOOOOd", &compq, &compz, &joba,
                          &l, &n, &m, &p, &a_obj, &e_obj, &b_obj, &c_obj, &tol)) {
        return NULL;
    }

    if (l < 0 || n < 0 || m < 0 || p < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative");
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_CDOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_CDOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_CDOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_CDOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || b_array == NULL || c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *e_dims = PyArray_DIMS(e_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    lda = (l > 0) ? (i32)a_dims[0] : 1;
    lde = (l > 0) ? (i32)e_dims[0] : 1;
    ldb = (l > 0) ? (i32)b_dims[0] : 1;
    ldc = (p > 0) ? (i32)c_dims[0] : 1;
    ldq = (l > 0) ? l : 1;
    ldz = (n > 0) ? n : 1;

    i32 ln = (l < n) ? l : n;
    i32 temp1 = n + p;
    i32 temp2 = (3 * n - 1 > m) ? 3 * n - 1 : m;
    temp2 = (temp2 > l) ? temp2 : l;
    temp2 = ln + temp2;
    i32 lzwork = (temp1 > temp2) ? temp1 : temp2;
    lzwork = (lzwork > 1) ? lzwork : 1;

    i32 *iwork = (n > 0) ? (i32*)malloc(n * sizeof(i32)) : NULL;
    f64 *dwork = (f64*)malloc(2 * n * sizeof(f64));
    c128 *zwork = (c128*)malloc(lzwork * sizeof(c128));
    c128 *q = NULL;
    c128 *z = NULL;

    bool compq_needed = (compq[0] == 'I' || compq[0] == 'i' || compq[0] == 'U' || compq[0] == 'u');
    bool compz_needed = (compz[0] == 'I' || compz[0] == 'i' || compz[0] == 'U' || compz[0] == 'u');

    npy_intp q_dims[2] = {l, l};
    npy_intp z_dims[2] = {n, n};
    npy_intp q_strides[2] = {sizeof(c128), l * sizeof(c128)};
    npy_intp z_strides[2] = {sizeof(c128), n * sizeof(c128)};
    
    PyObject *q_array = NULL;
    PyObject *z_array = NULL;

    if (compq_needed && l > 0) {
        q_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_CDOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (q_array == NULL) {
             free(iwork); free(dwork); free(zwork);
             Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(c_array);
             return PyErr_NoMemory();
        }
        q = (c128*)PyArray_DATA((PyArrayObject*)q_array);
        memset(q, 0, l * l * sizeof(c128));
    } else {
        q_array = PyArray_EMPTY(2, q_dims, NPY_CDOUBLE, 1);
    }
    
    if (compz_needed && n > 0) {
        z_array = PyArray_New(&PyArray_Type, 2, z_dims, NPY_CDOUBLE, z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (z_array == NULL) {
            free(iwork); free(dwork); free(zwork);
            Py_DECREF(q_array);
            Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(c_array);
            return PyErr_NoMemory();
        }
        z = (c128*)PyArray_DATA((PyArrayObject*)z_array);
        memset(z, 0, n * n * sizeof(c128));
    } else {
        z_array = PyArray_EMPTY(2, z_dims, NPY_CDOUBLE, 1);
    }

    if (dwork == NULL || zwork == NULL || (n > 0 && iwork == NULL)) {
        free(iwork);
        free(dwork);
        free(zwork);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    c128 *a_data = (c128*)PyArray_DATA(a_array);
    c128 *e_data = (c128*)PyArray_DATA(e_array);
    c128 *b_data = (c128*)PyArray_DATA(b_array);
    c128 *c_data = (c128*)PyArray_DATA(c_array);

    tg01fz(compq, compz, joba, l, n, m, p, a_data, lda, e_data, lde,
           b_data, ldb, c_data, ldc, q, ldq, z, ldz, &ranke, &rnka22,
           tol, iwork, dwork, zwork, lzwork, &info);

    free(iwork);
    free(dwork);
    free(zwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("(OOOOOOiii)", a_array, e_array, b_array, c_array,
                                     q_array, z_array, ranke, rnka22, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);

    return result;
}

PyObject* py_tg01hd(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"jobcon", "compq", "compz", "n", "m", "p",
                             "a", "e", "b", "c", "tol", "q", "z", NULL};
    const char *jobcon, *compq, *compz;
    i32 n, m, p;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj;
    PyObject *q_obj = Py_None, *z_obj = Py_None;
    f64 tol;
    PyArrayObject *a_array = NULL, *e_array = NULL, *b_array = NULL, *c_array = NULL;
    i32 ncont = 0, niucon = 0, nrblck = 0, info = 0;
    i32 lda, lde, ldb, ldc, ldq, ldz;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssiiiOOOOd|OO", kwlist,
                                      &jobcon, &compq, &compz, &n, &m, &p,
                                      &a_obj, &e_obj, &b_obj, &c_obj, &tol,
                                      &q_obj, &z_obj)) {
        return NULL;
    }

    if (n < 0 || m < 0 || p < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative");
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || b_array == NULL || c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *e_dims = PyArray_DIMS(e_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    lda = (n > 0) ? (i32)a_dims[0] : 1;
    lde = (n > 0) ? (i32)e_dims[0] : 1;
    ldb = (n > 0) ? (i32)b_dims[0] : 1;
    ldc = (p > 0) ? (i32)c_dims[0] : 1;
    ldq = (n > 0) ? n : 1;
    ldz = (n > 0) ? n : 1;

    i32 dwork_size = n;
    dwork_size = (dwork_size > 2 * m) ? dwork_size : 2 * m;
    if (dwork_size < 1) dwork_size = 1;

    i32 *iwork = (m > 0) ? (i32*)malloc(m * sizeof(i32)) : NULL;
    f64 *dwork = (f64*)malloc(dwork_size * sizeof(f64));
    i32 *rtau = (n > 0) ? (i32*)malloc(n * sizeof(i32)) : NULL;
    f64 *q = NULL;
    f64 *z = NULL;

    bool compq_needed = (compq[0] == 'I' || compq[0] == 'i' || compq[0] == 'U' || compq[0] == 'u');
    bool compz_needed = (compz[0] == 'I' || compz[0] == 'i' || compz[0] == 'U' || compz[0] == 'u');

    npy_intp q_dims[2] = {n, n};
    npy_intp z_dims[2] = {n, n};
    npy_intp rtau_dims[1] = {n};
    npy_intp q_strides[2] = {sizeof(f64), n * sizeof(f64)};
    npy_intp z_strides[2] = {sizeof(f64), n * sizeof(f64)};

    PyObject *q_array = NULL;
    PyObject *z_array = NULL;
    PyObject *rtau_array = NULL;

    if (compq_needed && n > 0) {
        q_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (q_array == NULL) {
             free(iwork); free(dwork); free(rtau);
             Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(c_array);
             return PyErr_NoMemory();
        }
        q = (f64*)PyArray_DATA((PyArrayObject*)q_array);
        memset(q, 0, n * n * sizeof(f64));

        if (q_obj != Py_None && (compq[0] == 'U' || compq[0] == 'u')) {
            PyArrayObject *q_input = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
            if (q_input != NULL) {
                memcpy(q, PyArray_DATA(q_input), n * n * sizeof(f64));
                Py_DECREF(q_input);
            }
        }
    } else {
        q_array = PyArray_EMPTY(2, q_dims, NPY_DOUBLE, 1);
    }

    if (compz_needed && n > 0) {
        z_array = PyArray_New(&PyArray_Type, 2, z_dims, NPY_DOUBLE, z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (z_array == NULL) {
             free(iwork); free(dwork); free(rtau);
             Py_DECREF(q_array);
             Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(c_array);
             return PyErr_NoMemory();
        }
        z = (f64*)PyArray_DATA((PyArrayObject*)z_array);
        memset(z, 0, n * n * sizeof(f64));

        if (z_obj != Py_None && (compz[0] == 'U' || compz[0] == 'u')) {
            PyArrayObject *z_input = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
            if (z_input != NULL) {
                memcpy(z, PyArray_DATA(z_input), n * n * sizeof(f64));
                Py_DECREF(z_input);
            }
        }
    } else {
        z_array = PyArray_EMPTY(2, z_dims, NPY_DOUBLE, 1);
    }

    if (dwork == NULL || (m > 0 && iwork == NULL) || (n > 0 && rtau == NULL)) {
        free(iwork);
        free(dwork);
        free(rtau);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    tg01hd(jobcon, compq, compz, n, m, p, a_data, lda, e_data, lde,
           b_data, ldb, c_data, ldc, q, ldq, z, ldz, &ncont, &niucon,
           &nrblck, rtau, tol, iwork, dwork, &info);

    if (n > 0) {
        rtau_array = PyArray_SimpleNew(1, rtau_dims, NPY_INT32);
        memcpy(PyArray_DATA((PyArrayObject*)rtau_array), rtau, n * sizeof(i32));
        free(rtau);
    } else {
        rtau_array = PyArray_EMPTY(1, rtau_dims, NPY_INT32, 0);
    }

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("(OOOOOOiiiOi)", a_array, e_array, b_array, c_array,
                                     q_array, z_array, ncont, niucon, nrblck, rtau_array, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);
    Py_DECREF(rtau_array);

    return result;
}

PyObject* py_tg01hu(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"compq", "compz", "l", "n", "m1", "m2", "p", "n1", "lbe",
                             "a", "e", "b", "c", "tol", "q", "z", NULL};
    const char *compq, *compz;
    i32 l, n, m1, m2, p, n1, lbe;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj;
    PyObject *q_obj = Py_None, *z_obj = Py_None;
    f64 tol;
    PyArrayObject *a_array = NULL, *e_array = NULL, *b_array = NULL, *c_array = NULL;
    i32 nr = 0, nrblck = 0, info = 0;
    i32 lda, lde, ldb, ldc, ldq, ldz;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssiiiiiiiOOOOd|OO", kwlist,
                                      &compq, &compz, &l, &n, &m1, &m2, &p, &n1, &lbe,
                                      &a_obj, &e_obj, &b_obj, &c_obj, &tol,
                                      &q_obj, &z_obj)) {
        return NULL;
    }

    if (l < 0 || n < 0 || m1 < 0 || m2 < 0 || p < 0 || n1 < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative");
        return NULL;
    }

    i32 m = m1 + m2;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || b_array == NULL || c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *e_dims = PyArray_DIMS(e_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    lda = (l > 0) ? (i32)a_dims[0] : 1;
    lde = (l > 0) ? (i32)e_dims[0] : 1;
    ldb = (l > 0) ? (i32)b_dims[0] : 1;
    ldc = (p > 0) ? (i32)c_dims[0] : 1;
    ldq = (l > 0) ? l : 1;
    ldz = (n > 0) ? n : 1;

    i32 dwork_size = n;
    dwork_size = (dwork_size > l) ? dwork_size : l;
    dwork_size = (dwork_size > 2 * m) ? dwork_size : 2 * m;
    if (n1 > 0 && lbe > 0 && n1 > 2) {
        i32 max_lnm = l;
        if (n > max_lnm) max_lnm = n;
        if (m > max_lnm) max_lnm = m;
        if (n1 + max_lnm > dwork_size) dwork_size = n1 + max_lnm;
    }
    if (dwork_size < 1) dwork_size = 1;

    i32 *iwork = (m > 0) ? (i32*)malloc(m * sizeof(i32)) : NULL;
    f64 *dwork = (f64*)malloc(dwork_size * sizeof(f64));
    i32 *rtau = (2 * n1 > 0) ? (i32*)malloc(2 * n1 * sizeof(i32)) : NULL;
    f64 *q = NULL;
    f64 *z = NULL;

    bool compq_needed = (compq[0] == 'I' || compq[0] == 'i' || compq[0] == 'U' || compq[0] == 'u');
    bool compz_needed = (compz[0] == 'I' || compz[0] == 'i' || compz[0] == 'U' || compz[0] == 'u');

    npy_intp q_dims[2] = {l, l};
    npy_intp z_dims[2] = {n, n};
    npy_intp rtau_dims[1] = {2 * n1 > 0 ? 2 * n1 : 1};
    npy_intp q_strides[2] = {sizeof(f64), l * sizeof(f64)};
    npy_intp z_strides[2] = {sizeof(f64), n * sizeof(f64)};

    PyObject *q_array = NULL;
    PyObject *z_array = NULL;
    PyObject *rtau_array = NULL;

    if (compq_needed && l > 0) {
        q_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (q_array == NULL) {
             free(iwork); free(dwork); free(rtau);
             Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(c_array);
             return PyErr_NoMemory();
        }
        q = (f64*)PyArray_DATA((PyArrayObject*)q_array);
        memset(q, 0, l * l * sizeof(f64));

        if (q_obj != Py_None && (compq[0] == 'U' || compq[0] == 'u')) {
            PyArrayObject *q_input = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
            if (q_input != NULL) {
                memcpy(q, PyArray_DATA(q_input), l * l * sizeof(f64));
                Py_DECREF(q_input);
            }
        }
    } else {
        q_array = PyArray_EMPTY(2, q_dims, NPY_DOUBLE, 1);
    }

    if (compz_needed && n > 0) {
        z_array = PyArray_New(&PyArray_Type, 2, z_dims, NPY_DOUBLE, z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (z_array == NULL) {
             free(iwork); free(dwork); free(rtau);
             Py_DECREF(q_array);
             Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(c_array);
             return PyErr_NoMemory();
        }
        z = (f64*)PyArray_DATA((PyArrayObject*)z_array);
        memset(z, 0, n * n * sizeof(f64));

        if (z_obj != Py_None && (compz[0] == 'U' || compz[0] == 'u')) {
            PyArrayObject *z_input = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
            if (z_input != NULL) {
                memcpy(z, PyArray_DATA(z_input), n * n * sizeof(f64));
                Py_DECREF(z_input);
            }
        }
    } else {
        z_array = PyArray_EMPTY(2, z_dims, NPY_DOUBLE, 1);
    }

    if (dwork == NULL || (m > 0 && iwork == NULL) || (2 * n1 > 0 && rtau == NULL)) {
        free(iwork);
        free(dwork);
        free(rtau);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    tg01hu(compq, compz, l, n, m1, m2, p, n1, lbe, a_data, lda, e_data, lde,
           b_data, ldb, c_data, ldc, q, ldq, z, ldz, &nr, &nrblck, rtau,
           tol, iwork, dwork, dwork_size, &info);

    if (2 * n1 > 0) {
        rtau_array = PyArray_SimpleNew(1, rtau_dims, NPY_INT32);
        memcpy(PyArray_DATA((PyArrayObject*)rtau_array), rtau, 2 * n1 * sizeof(i32));
        free(rtau);
    } else {
        rtau_array = PyArray_EMPTY(1, rtau_dims, NPY_INT32, 0);
    }

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("(OOOOOOiiOi)", a_array, e_array, b_array, c_array,
                                     q_array, z_array, nr, nrblck, rtau_array, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);
    Py_DECREF(rtau_array);

    return result;
}

PyObject* py_tg01hy(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"compq", "compz", "l", "n", "m", "p", "n1", "lbe",
                             "a", "e", "b", "c", "tol", "q", NULL};
    const char *compq, *compz;
    i32 l, n, m, p, n1, lbe;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj;
    PyObject *q_obj = Py_None;
    f64 tol;
    PyArrayObject *a_array = NULL, *e_array = NULL, *b_array = NULL, *c_array = NULL;
    i32 nr = 0, nrblck = 0, info = 0;
    i32 lda, lde, ldb, ldc, ldq, ldz;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssiiiiiiOOOOd|O", kwlist,
                                      &compq, &compz, &l, &n, &m, &p, &n1, &lbe,
                                      &a_obj, &e_obj, &b_obj, &c_obj, &tol, &q_obj)) {
        return NULL;
    }

    if (l < 0 || n < 0 || m < 0 || p < 0 || n1 < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative");
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || b_array == NULL || c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *e_dims = PyArray_DIMS(e_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    lda = (l > 0) ? (i32)a_dims[0] : 1;
    lde = (l > 0) ? (i32)e_dims[0] : 1;
    ldb = (l > 0) ? (i32)b_dims[0] : 1;
    ldc = (p > 0) ? (i32)c_dims[0] : 1;
    ldq = (l > 0) ? l : 1;
    ldz = (n > 0) ? n : 1;

    i32 dwork_size = n;
    dwork_size = (dwork_size > l) ? dwork_size : l;
    i32 tmp = 2 * (m + n1 - 1);
    dwork_size = (dwork_size > tmp) ? dwork_size : tmp;
    if (dwork_size < 1) dwork_size = 1;

    i32 *iwork = (m > 0) ? (i32*)malloc(m * sizeof(i32)) : NULL;
    f64 *dwork = (f64*)malloc(dwork_size * sizeof(f64));
    i32 *rtau = (n1 > 0) ? (i32*)malloc(n1 * sizeof(i32)) : NULL;
    f64 *q = NULL;
    f64 *z = NULL;

    bool compq_needed = (compq[0] == 'I' || compq[0] == 'i' || compq[0] == 'U' || compq[0] == 'u');
    bool compz_needed = (compz[0] == 'I' || compz[0] == 'i' || compz[0] == 'U' || compz[0] == 'u');

    npy_intp q_dims[2] = {l, l};
    npy_intp z_dims[2] = {n, n};
    npy_intp rtau_dims[1] = {n1};
    npy_intp q_strides[2] = {sizeof(f64), l * sizeof(f64)};
    npy_intp z_strides[2] = {sizeof(f64), n * sizeof(f64)};

    PyObject *q_array = NULL;
    PyObject *z_array = NULL;
    PyObject *rtau_array = NULL;

    if (compq_needed && l > 0) {
        q_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (q_array == NULL) {
             free(iwork); free(dwork); free(rtau);
             Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(c_array);
             return PyErr_NoMemory();
        }
        q = (f64*)PyArray_DATA((PyArrayObject*)q_array);
        memset(q, 0, l * l * sizeof(f64));

        if (q_obj != Py_None && (compq[0] == 'U' || compq[0] == 'u')) {
            PyArrayObject *q_input = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
            if (q_input != NULL) {
                memcpy(q, PyArray_DATA(q_input), l * l * sizeof(f64));
                Py_DECREF(q_input);
            }
        }
    } else {
        q_array = PyArray_EMPTY(2, q_dims, NPY_DOUBLE, 1);
    }

    if (compz_needed && n > 0) {
        z_array = PyArray_New(&PyArray_Type, 2, z_dims, NPY_DOUBLE, z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (z_array == NULL) {
             free(iwork); free(dwork); free(rtau);
             Py_DECREF(q_array);
             Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(c_array);
             return PyErr_NoMemory();
        }
        z = (f64*)PyArray_DATA((PyArrayObject*)z_array);
        memset(z, 0, n * n * sizeof(f64));
    } else {
        z_array = PyArray_EMPTY(2, z_dims, NPY_DOUBLE, 1);
    }

    if (dwork == NULL || (m > 0 && iwork == NULL) || (n1 > 0 && rtau == NULL)) {
        free(iwork);
        free(dwork);
        free(rtau);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    tg01hy(compq, compz, l, n, m, p, n1, lbe, a_data, lda, e_data, lde,
           b_data, ldb, c_data, ldc, q, ldq, z, ldz, &nr, &nrblck, rtau,
           tol, iwork, dwork, dwork_size, &info);

    if (n1 > 0) {
        rtau_array = PyArray_SimpleNew(1, rtau_dims, NPY_INT32);
        memcpy(PyArray_DATA((PyArrayObject*)rtau_array), rtau, n1 * sizeof(i32));
        free(rtau);
    } else {
        rtau_array = PyArray_EMPTY(1, rtau_dims, NPY_INT32, 0);
    }

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *hy_result = Py_BuildValue("(OOOOOOiiOi)", a_array, e_array, b_array, c_array,
                                     q_array, z_array, nr, nrblck, rtau_array, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);
    Py_DECREF(rtau_array);

    return hy_result;
}

PyObject* py_tg01kd(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static char *kwlist[] = {"jobe", "compc", "compq", "compz", "a", "e", "b", "c", "q", "z", "incc", NULL};

    const char *jobe, *compc, *compq, *compz;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj;
    PyObject *q_obj = Py_None, *z_obj = Py_None;
    i32 incc = 1;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssssOOOO|OOi", kwlist,
                                     &jobe, &compc, &compq, &compz,
                                     &a_obj, &e_obj, &b_obj, &c_obj,
                                     &q_obj, &z_obj, &incc)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_INOUT_ARRAY2);
    if (a_array == NULL) return NULL;

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 lda = (n > 1) ? n : 1;

    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_INOUT_ARRAY2);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }
    i32 lde = (n > 1) ? n : 1;
    bool unite = (jobe[0] == 'I' || jobe[0] == 'i');
    if (unite) {
        lde = 1;
    }

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_INOUT_ARRAY2);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_INOUT_ARRAY2);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        return NULL;
    }

    bool withq = (compq[0] == 'I' || compq[0] == 'i' || compq[0] == 'U' || compq[0] == 'u');
    bool withz = (compz[0] == 'I' || compz[0] == 'i' || compz[0] == 'U' || compz[0] == 'u');
    bool lupdq = (compq[0] == 'U' || compq[0] == 'u');
    bool lupdz = (compz[0] == 'U' || compz[0] == 'u');

    i32 ldq = (withq && n > 1) ? n : 1;
    i32 ldz = (withz && n > 1) ? n : 1;

    npy_intp q_dims[2] = {n > 0 ? n : 1, n > 0 ? n : 1};
    npy_intp z_dims[2] = {n > 0 ? n : 1, n > 0 ? n : 1};
    npy_intp q_strides[2] = {sizeof(f64), (n > 0 ? n : 1) * sizeof(f64)};
    npy_intp z_strides[2] = {sizeof(f64), (n > 0 ? n : 1) * sizeof(f64)};

    PyObject *q_array_out = NULL;
    PyObject *z_array_out = NULL;
    f64 *q_data = NULL;
    f64 *z_data = NULL;

    if (withq && n > 0) {
        q_array_out = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (q_array_out == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            return PyErr_NoMemory();
        }
        q_data = (f64*)PyArray_DATA((PyArrayObject*)q_array_out);
        memset(q_data, 0, n * n * sizeof(f64));

        if (lupdq && q_obj != Py_None) {
            PyArrayObject *q_input = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
            if (q_input != NULL) {
                memcpy(q_data, PyArray_DATA(q_input), n * n * sizeof(f64));
                Py_DECREF(q_input);
            }
        }
    } else {
        q_array_out = PyArray_EMPTY(2, q_dims, NPY_DOUBLE, 1);
        if (q_array_out != NULL) {
            q_data = (f64*)PyArray_DATA((PyArrayObject*)q_array_out);
        }
    }

    if (withz && n > 0) {
        z_array_out = PyArray_New(&PyArray_Type, 2, z_dims, NPY_DOUBLE, z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (z_array_out == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            Py_DECREF(q_array_out);
            return PyErr_NoMemory();
        }
        z_data = (f64*)PyArray_DATA((PyArrayObject*)z_array_out);
        memset(z_data, 0, n * n * sizeof(f64));

        if (lupdz && z_obj != Py_None) {
            PyArrayObject *z_input = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
            if (z_input != NULL) {
                memcpy(z_data, PyArray_DATA(z_input), n * n * sizeof(f64));
                Py_DECREF(z_input);
            }
        }
    } else {
        z_array_out = PyArray_EMPTY(2, z_dims, NPY_DOUBLE, 1);
        if (z_array_out != NULL) {
            z_data = (f64*)PyArray_DATA((PyArrayObject*)z_array_out);
        }
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    i32 info = 0;
    tg01kd(jobe, compc, compq, compz, n, a_data, lda, e_data, lde,
           b_data, c_data, incc, q_data, ldq, z_data, ldz, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("(OOOOOOi)", a_array, e_array, b_array, c_array,
                                     q_array_out, z_array_out, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(q_array_out);
    Py_DECREF(z_array_out);

    return result;
}

PyObject* py_tg01jd(PyObject* self, PyObject* args) {
    const char *job, *systyp, *equil;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj;
    double tol;

    if (!PyArg_ParseTuple(args, "sssOOOOd", &job, &systyp, &equil,
                          &a_obj, &e_obj, &b_obj, &c_obj, &tol)) {
        return NULL;
    }

    PyArrayObject *a_in = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    PyArrayObject *e_in = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    PyArrayObject *b_in = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    PyArrayObject *c_in = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);

    if (!a_in || !e_in || !b_in || !c_in) {
        Py_XDECREF(a_in);
        Py_XDECREF(e_in);
        Py_XDECREF(b_in);
        Py_XDECREF(c_in);
        PyErr_SetString(PyExc_ValueError, "Failed to convert input arrays");
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_in, 0);
    i32 m = (i32)PyArray_DIM(b_in, 1);
    i32 p = (i32)PyArray_DIM(c_in, 0);

    i32 lda = (n > 1) ? n : 1;
    i32 lde = (n > 1) ? n : 1;
    i32 ldb = (n > 1) ? n : 1;
    i32 maxmp = (m > p) ? m : p;
    i32 ldc = (maxmp > 1) ? maxmp : 1;
    if (n == 0) ldc = 1;

    bool ljobir = (job[0] == 'I' || job[0] == 'i');
    bool ljobo = ljobir || (job[0] == 'O' || job[0] == 'o');
    bool lsysr = (systyp[0] == 'R' || systyp[0] == 'r');
    bool lequil = (equil[0] == 'S' || equil[0] == 's');

    i32 c_mult = (ljobir || lsysr) ? 2 : 1;
    i32 iwork_size = c_mult * n + maxmp;
    if (iwork_size < 1) iwork_size = 1;

    i32 min_ldwork;
    if (lequil) {
        min_ldwork = (8 * n > 2 * maxmp) ? 8 * n : 2 * maxmp;
    } else {
        min_ldwork = (n > 2 * maxmp) ? n : 2 * maxmp;
    }
    i32 large_ldwork = 2 * n * n + n * m + n * p + min_ldwork;
    i32 ldwork = large_ldwork > min_ldwork ? large_ldwork : min_ldwork;
    if (ldwork < 1) ldwork = 1;

    i32 *iwork = (i32*)calloc(iwork_size, sizeof(i32));
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        Py_DECREF(a_in);
        Py_DECREF(e_in);
        Py_DECREF(b_in);
        Py_DECREF(c_in);
        return PyErr_NoMemory();
    }

    npy_intp a_dims[2] = {n, n};
    npy_intp a_strides[2] = {sizeof(f64), n * sizeof(f64)};
    PyObject *a_array = PyArray_New(&PyArray_Type, 2, a_dims, NPY_DOUBLE, a_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    npy_intp e_dims[2] = {n, n};
    npy_intp e_strides[2] = {sizeof(f64), n * sizeof(f64)};
    PyObject *e_array = PyArray_New(&PyArray_Type, 2, e_dims, NPY_DOUBLE, e_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    i32 b_cols = ljobo ? maxmp : m;
    npy_intp b_dims[2] = {n, b_cols};
    npy_intp b_strides[2] = {sizeof(f64), ldb * sizeof(f64)};
    PyObject *b_array = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    npy_intp c_dims[2] = {ldc, n};
    npy_intp c_strides[2] = {sizeof(f64), ldc * sizeof(f64)};
    PyObject *c_array = PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE, c_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (!a_array || !e_array || !b_array || !c_array) {
        free(iwork);
        free(dwork);
        Py_DECREF(a_in);
        Py_DECREF(e_in);
        Py_DECREF(b_in);
        Py_DECREF(c_in);
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        return PyErr_NoMemory();
    }

    f64 *a_data = (f64*)PyArray_DATA((PyArrayObject*)a_array);
    f64 *e_data = (f64*)PyArray_DATA((PyArrayObject*)e_array);
    f64 *b_data = (f64*)PyArray_DATA((PyArrayObject*)b_array);
    f64 *c_data = (f64*)PyArray_DATA((PyArrayObject*)c_array);

    memcpy(a_data, PyArray_DATA(a_in), n * n * sizeof(f64));
    memcpy(e_data, PyArray_DATA(e_in), n * n * sizeof(f64));

    if (n > 0 && b_cols > 0) { memset(b_data, 0, ldb * b_cols * sizeof(f64)); }
    for (i32 j = 0; j < m; j++) {
        memcpy(&b_data[j * ldb], &((f64*)PyArray_DATA(b_in))[j * n], n * sizeof(f64));
    }

    if (ldc > 0 && n > 0) { memset(c_data, 0, ldc * n * sizeof(f64)); }
    for (i32 j = 0; j < n; j++) {
        memcpy(&c_data[j * ldc], &((f64*)PyArray_DATA(c_in))[j * p], p * sizeof(f64));
    }

    Py_DECREF(a_in);
    Py_DECREF(e_in);
    Py_DECREF(b_in);
    Py_DECREF(c_in);

    i32 nr = 0;
    i32 infred[7] = {0};
    i32 info = 0;

    tg01jd(job, systyp, equil, n, m, p, a_data, lda, e_data, lde,
           b_data, ldb, c_data, ldc, &nr, infred, tol, iwork, dwork, ldwork, &info);

    npy_intp infred_dims[1] = {7};
    PyObject *infred_array = PyArray_SimpleNew(1, infred_dims, NPY_INT32);
    memcpy(PyArray_DATA((PyArrayObject*)infred_array), infred, 7 * sizeof(i32));

    i32 nblck = (infred[6] > 0) ? infred[6] : 0;
    npy_intp iwork_out_dims[1] = {nblck > 0 ? nblck : 1};
    PyObject *iwork_array = PyArray_SimpleNew(1, iwork_out_dims, NPY_INT32);
    if (nblck > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)iwork_array), iwork, nblck * sizeof(i32));
    }

    free(iwork);
    free(dwork);

    npy_intp ar_dims[2] = {nr, nr};
    npy_intp ar_strides[2] = {sizeof(f64), nr * sizeof(f64)};
    PyObject *ar_array = PyArray_New(&PyArray_Type, 2, ar_dims, NPY_DOUBLE, ar_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    npy_intp er_dims[2] = {nr, nr};
    npy_intp er_strides[2] = {sizeof(f64), nr * sizeof(f64)};
    PyObject *er_array = PyArray_New(&PyArray_Type, 2, er_dims, NPY_DOUBLE, er_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    npy_intp br_dims[2] = {nr, m};
    npy_intp br_strides[2] = {sizeof(f64), nr * sizeof(f64)};
    PyObject *br_array = PyArray_New(&PyArray_Type, 2, br_dims, NPY_DOUBLE, br_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    npy_intp cr_dims[2] = {p, nr};
    npy_intp cr_strides[2] = {sizeof(f64), p * sizeof(f64)};
    PyObject *cr_array = PyArray_New(&PyArray_Type, 2, cr_dims, NPY_DOUBLE, cr_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (!ar_array || !er_array || !br_array || !cr_array) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(infred_array);
        Py_DECREF(iwork_array);
        Py_XDECREF(ar_array);
        Py_XDECREF(er_array);
        Py_XDECREF(br_array);
        Py_XDECREF(cr_array);
        return PyErr_NoMemory();
    }

    f64 *ar_data = (f64*)PyArray_DATA((PyArrayObject*)ar_array);
    f64 *er_data = (f64*)PyArray_DATA((PyArrayObject*)er_array);
    f64 *br_data = (f64*)PyArray_DATA((PyArrayObject*)br_array);
    f64 *cr_data = (f64*)PyArray_DATA((PyArrayObject*)cr_array);

    for (i32 j = 0; j < nr; j++) {
        memcpy(&ar_data[j * nr], &a_data[j * lda], nr * sizeof(f64));
        memcpy(&er_data[j * nr], &e_data[j * lde], nr * sizeof(f64));
    }
    for (i32 j = 0; j < m; j++) {
        memcpy(&br_data[j * nr], &b_data[j * ldb], nr * sizeof(f64));
    }
    for (i32 j = 0; j < nr; j++) {
        memcpy(&cr_data[j * p], &c_data[j * ldc], p * sizeof(f64));
    }

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);

    PyObject *result = Py_BuildValue("(OOOOiOOi)", ar_array, er_array, br_array, cr_array,
                                     nr, infred_array, iwork_array, info);

    Py_DECREF(ar_array);
    Py_DECREF(er_array);
    Py_DECREF(br_array);
    Py_DECREF(cr_array);
    Py_DECREF(infred_array);
    Py_DECREF(iwork_array);

    return result;
}

PyObject* py_tg01id(PyObject* self, PyObject* args) {
    const char *jobobs, *compq, *compz;
    f64 tol;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj;
    PyArrayObject *a_array = NULL, *e_array = NULL, *b_array = NULL, *c_array = NULL;
    i32 info = 0;

    if (!PyArg_ParseTuple(args, "sssOOOOd", &jobobs, &compq, &compz,
                          &a_obj, &e_obj, &b_obj, &c_obj, &tol)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || b_array == NULL || c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        PyErr_SetString(PyExc_ValueError, "Failed to convert input arrays");
        return NULL;
    }

    i32 ndim_a = PyArray_NDIM(a_array);
    i32 ndim_b = PyArray_NDIM(b_array);
    i32 ndim_c = PyArray_NDIM(c_array);

    i32 n, m, p;

    if (ndim_a == 2) {
        n = (i32)PyArray_DIM(a_array, 0);
    } else if (ndim_a == 0) {
        n = 0;
    } else {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_ValueError, "A must be a 2D array");
        return NULL;
    }

    if (n > 0) {
        if (ndim_b >= 2) {
            m = (i32)PyArray_DIM(b_array, 1);
        } else if (ndim_b == 1) {
            m = 1;
        } else {
            m = 0;
        }

        if (ndim_c >= 2) {
            p = (i32)PyArray_DIM(c_array, 0);
        } else if (ndim_c == 1) {
            p = 1;
        } else {
            p = 0;
        }
    } else {
        m = 0;
        p = 0;
    }

    i32 lda = (n > 1) ? n : 1;
    i32 lde = (n > 1) ? n : 1;
    i32 maxmp = (m > p) ? m : p;
    i32 ldb = (n > 1) ? n : 1;
    i32 ldc = (maxmp > 1) ? maxmp : 1;

    bool ilq = (compq[0] == 'I' || compq[0] == 'i' || compq[0] == 'U' || compq[0] == 'u');
    bool ilz = (compz[0] == 'I' || compz[0] == 'i' || compz[0] == 'U' || compz[0] == 'u');

    i32 ldq = ilq ? ((n > 1) ? n : 1) : 1;
    i32 ldz = ilz ? ((n > 1) ? n : 1) : 1;

    f64 *b_work = NULL, *c_work = NULL;
    bool need_b_work = (maxmp > m && n > 0);
    bool need_c_work = (maxmp > p && n > 0);

    if (need_b_work) {
        b_work = (f64*)calloc(ldb * maxmp, sizeof(f64));
        if (b_work == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate B work array");
            return NULL;
        }
        f64 *b_data = (f64*)PyArray_DATA(b_array);
        for (i32 j = 0; j < m; j++) {
            for (i32 i = 0; i < n; i++) {
                b_work[i + j * ldb] = b_data[i + j * ldb];
            }
        }
    }

    if (need_c_work) {
        c_work = (f64*)calloc(ldc * n, sizeof(f64));
        if (c_work == NULL) {
            free(b_work);
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate C work array");
            return NULL;
        }
        f64 *c_data = (f64*)PyArray_DATA(c_array);
        i32 orig_ldc = (p > 1) ? p : 1;
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < p; i++) {
                c_work[i + j * ldc] = c_data[i + j * orig_ldc];
            }
        }
    }

    npy_intp q_dims[2] = {n, n};
    npy_intp z_dims[2] = {n, n};
    npy_intp q_strides[2] = {sizeof(f64), ldq * sizeof(f64)};
    npy_intp z_strides[2] = {sizeof(f64), ldz * sizeof(f64)};

    PyObject *q_array = NULL, *z_array = NULL;

    if (ilq && n > 0) {
        q_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (q_array == NULL) {
            free(b_work);
            free(c_work);
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate Q array");
            return NULL;
        }
    } else {
        npy_intp one_dims[2] = {1, 1};
        q_array = PyArray_ZEROS(2, one_dims, NPY_DOUBLE, 1);
    }

    if (ilz && n > 0) {
        z_array = PyArray_New(&PyArray_Type, 2, z_dims, NPY_DOUBLE, z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (z_array == NULL) {
            free(b_work);
            free(c_work);
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            Py_DECREF(q_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate Z array");
            return NULL;
        }
    } else {
        npy_intp one_dims[2] = {1, 1};
        z_array = PyArray_ZEROS(2, one_dims, NPY_DOUBLE, 1);
    }

    f64 *q = (f64*)PyArray_DATA((PyArrayObject*)q_array);
    f64 *z = (f64*)PyArray_DATA((PyArrayObject*)z_array);

    i32 nobsv = 0, niuobs = 0, nlblck = 0;

    i32 ctau_size = (n > 1) ? n : 1;
    i32 *ctau = (i32*)calloc(ctau_size, sizeof(i32));
    i32 iwork_size = (p > 1) ? p : 1;
    i32 *iwork = (i32*)calloc(iwork_size, sizeof(i32));
    i32 dwork_size = (n > 2*p) ? n : 2*p;
    if (dwork_size < 1) dwork_size = 1;
    f64 *dwork = (f64*)calloc(dwork_size, sizeof(f64));

    if (ctau == NULL || iwork == NULL || dwork == NULL) {
        free(ctau);
        free(iwork);
        free(dwork);
        free(b_work);
        free(c_work);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data_ptr = need_b_work ? b_work : (f64*)PyArray_DATA(b_array);
    f64 *c_data_ptr = need_c_work ? c_work : (f64*)PyArray_DATA(c_array);
    i32 actual_ldc = need_c_work ? ldc : ((p > 1) ? p : 1);

    tg01id(jobobs, compq, compz, n, m, p,
           a_data, lda, e_data, lde, b_data_ptr, ldb, c_data_ptr, actual_ldc,
           q, ldq, z, ldz, &nobsv, &niuobs, &nlblck, ctau,
           tol, iwork, dwork, &info);

    if (need_b_work) {
        f64 *b_out = (f64*)PyArray_DATA(b_array);
        for (i32 j = 0; j < m; j++) {
            for (i32 i = 0; i < n; i++) {
                b_out[i + j * ldb] = b_work[i + j * ldb];
            }
        }
    }

    if (need_c_work) {
        f64 *c_out = (f64*)PyArray_DATA(c_array);
        i32 orig_ldc = (p > 1) ? p : 1;
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < p; i++) {
                c_out[i + j * orig_ldc] = c_work[i + j * ldc];
            }
        }
    }

    npy_intp ctau_dims[1] = {n};
    PyObject *ctau_array = NULL;
    if (n > 0) {
        ctau_array = PyArray_SimpleNew(1, ctau_dims, NPY_INT32);
        memcpy(PyArray_DATA((PyArrayObject*)ctau_array), ctau, n * sizeof(i32));
    } else {
        npy_intp zero_dims[1] = {0};
        ctau_array = PyArray_EMPTY(1, zero_dims, NPY_INT32, 0);
    }

    free(ctau);
    free(iwork);
    free(dwork);
    free(b_work);
    free(c_work);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *id_result = Py_BuildValue("(OOOOOOiiiOi)",
                                     a_array, e_array, b_array, c_array,
                                     q_array, z_array,
                                     nobsv, niuobs, nlblck, ctau_array, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);
    Py_DECREF(ctau_array);

    return id_result;
}


PyObject* py_tg01kz(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *jobe, *compc, *compq, *compz;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj;
    PyObject *q_obj = Py_None, *z_obj = Py_None;
    i32 incc = 1;
    PyArrayObject *a_array, *e_array, *b_array, *c_array;
    i32 info = 0;

    static char *kwlist[] = {"jobe", "compc", "compq", "compz",
                             "a", "e", "b", "c", "incc", "q", "z", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssssOOOO|iOO", kwlist,
                                     &jobe, &compc, &compq, &compz,
                                     &a_obj, &e_obj, &b_obj, &c_obj,
                                     &incc, &q_obj, &z_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_COMPLEX128,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_COMPLEX128,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_COMPLEX128,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_COMPLEX128,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || b_array == NULL || c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        return NULL;
    }

    int ndim_a = PyArray_NDIM(a_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);

    i32 n;
    if (ndim_a == 0) {
        n = 0;
    } else if (ndim_a == 1) {
        n = (i32)a_dims[0];
    } else {
        n = (i32)a_dims[0];
    }

    if (n < 0) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_ValueError, "N must be non-negative");
        return NULL;
    }

    bool unite = (jobe[0] == 'I' || jobe[0] == 'i');
    i32 lda = (n > 0) ? n : 1;
    i32 lde = unite ? 1 : ((n > 0) ? n : 1);

    bool liniq = (compq[0] == 'I' || compq[0] == 'i');
    bool lupdq = (compq[0] == 'U' || compq[0] == 'u');
    bool liniz = (compz[0] == 'I' || compz[0] == 'i');
    bool lupdz = (compz[0] == 'U' || compz[0] == 'u');
    bool withq = liniq || lupdq;
    bool withz = liniz || lupdz;

    i32 ldq = withq ? ((n > 0) ? n : 1) : 1;
    i32 ldz = withz ? ((n > 0) ? n : 1) : 1;

    npy_intp q_dims[2] = {n, n};
    npy_intp z_dims[2] = {n, n};
    npy_intp q_strides[2] = {sizeof(c128), ldq * sizeof(c128)};
    npy_intp z_strides[2] = {sizeof(c128), ldz * sizeof(c128)};

    PyObject *q_array = NULL, *z_array = NULL;

    if (lupdq && q_obj != Py_None) {
        q_array = PyArray_FROM_OTF(q_obj, NPY_COMPLEX128,
                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (q_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            return NULL;
        }
    } else if (withq && n > 0) {
        q_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_COMPLEX128, q_strides,
                              NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (q_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate Q array");
            return NULL;
        }
    } else {
        npy_intp one_dims[2] = {1, 1};
        q_array = PyArray_ZEROS(2, one_dims, NPY_COMPLEX128, 1);
    }

    if (lupdz && z_obj != Py_None) {
        z_array = PyArray_FROM_OTF(z_obj, NPY_COMPLEX128,
                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (z_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            Py_DECREF(q_array);
            return NULL;
        }
    } else if (withz && n > 0) {
        z_array = PyArray_New(&PyArray_Type, 2, z_dims, NPY_COMPLEX128, z_strides,
                              NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (z_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            Py_DECREF(q_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate Z array");
            return NULL;
        }
    } else {
        npy_intp one_dims[2] = {1, 1};
        z_array = PyArray_ZEROS(2, one_dims, NPY_COMPLEX128, 1);
    }

    c128 *a_data = (c128*)PyArray_DATA(a_array);
    c128 *e_data = (c128*)PyArray_DATA(e_array);
    c128 *b_data = (c128*)PyArray_DATA(b_array);
    c128 *c_data = (c128*)PyArray_DATA(c_array);
    c128 *q_data = (c128*)PyArray_DATA((PyArrayObject*)q_array);
    c128 *z_data = (c128*)PyArray_DATA((PyArrayObject*)z_array);

    tg01kz(jobe, compc, compq, compz, n,
           a_data, lda, e_data, lde, b_data,
           c_data, incc, q_data, ldq, z_data, ldz, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    if (lupdq && q_obj != Py_None) {
        PyArray_ResolveWritebackIfCopy((PyArrayObject*)q_array);
    }
    if (lupdz && z_obj != Py_None) {
        PyArray_ResolveWritebackIfCopy((PyArrayObject*)z_array);
    }

    PyObject *result = Py_BuildValue("(OOOOOOi)", a_array, e_array, b_array, c_array,
                                     q_array, z_array, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);

    return result;
}

PyObject* py_tg01nx(PyObject* self, PyObject* args) {
    (void)self;

    const char *jobt;
    i32 n, m, p, ndim;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj, *q_obj, *z_obj;

    if (!PyArg_ParseTuple(args, "siiiiOOOOOO",
                          &jobt, &n, &m, &p, &ndim,
                          &a_obj, &e_obj, &b_obj, &c_obj, &q_obj, &z_obj)) {
        return NULL;
    }

    PyObject *a_array = PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyObject *e_array = PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyObject *b_array = PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyObject *c_array = PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyObject *q_array = PyArray_FROM_OTF(q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyObject *z_array = PyArray_FROM_OTF(z_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!a_array || !e_array || !b_array || !c_array || !q_array || !z_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(q_array);
        Py_XDECREF(z_array);
        return NULL;
    }

    i32 lda = (n > 1) ? n : 1;
    i32 lde = (n > 1) ? n : 1;
    i32 ldb = (n > 1) ? n : 1;
    i32 ldc = (p > 1) ? p : 1;
    i32 ldq = (n > 1) ? n : 1;
    i32 ldz = (n > 1) ? n : 1;

    f64 *a_data = (f64*)PyArray_DATA((PyArrayObject*)a_array);
    f64 *e_data = (f64*)PyArray_DATA((PyArrayObject*)e_array);
    f64 *b_data = (f64*)PyArray_DATA((PyArrayObject*)b_array);
    f64 *c_data = (f64*)PyArray_DATA((PyArrayObject*)c_array);
    f64 *q_data = (f64*)PyArray_DATA((PyArrayObject*)q_array);
    f64 *z_data = (f64*)PyArray_DATA((PyArrayObject*)z_array);

    i32 *iwork = (i32*)PyMem_Calloc(n + 6, sizeof(i32));
    if (!iwork) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        return PyErr_NoMemory();
    }

    i32 info;
    tg01nx(jobt, n, m, p, ndim, a_data, lda, e_data, lde, b_data, ldb,
           c_data, ldc, q_data, ldq, z_data, ldz, iwork, &info);

    PyMem_Free(iwork);

    PyArray_ResolveWritebackIfCopy((PyArrayObject*)a_array);
    PyArray_ResolveWritebackIfCopy((PyArrayObject*)e_array);
    PyArray_ResolveWritebackIfCopy((PyArrayObject*)b_array);
    PyArray_ResolveWritebackIfCopy((PyArrayObject*)c_array);
    PyArray_ResolveWritebackIfCopy((PyArrayObject*)q_array);
    PyArray_ResolveWritebackIfCopy((PyArrayObject*)z_array);

    PyObject *res = Py_BuildValue("(OOOOOOi)", a_array, e_array, b_array, c_array,
                                  q_array, z_array, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);

    return res;
}

PyObject* py_tg01jy(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;
    const char *job, *systyp, *equil, *cksing, *restor;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj;
    PyObject *tol_obj = Py_None;

    static char *kwlist[] = {"job", "systyp", "equil", "cksing", "restor",
                             "a", "e", "b", "c", "tol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssssOOOO|O:tg01jy", kwlist,
                                     &job, &systyp, &equil, &cksing, &restor,
                                     &a_obj, &e_obj, &b_obj, &c_obj, &tol_obj)) {
        return NULL;
    }

    PyArrayObject *a_in = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    PyArrayObject *e_in = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    PyArrayObject *b_in = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    PyArrayObject *c_in = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);

    if (!a_in || !e_in || !b_in || !c_in) {
        Py_XDECREF(a_in);
        Py_XDECREF(e_in);
        Py_XDECREF(b_in);
        Py_XDECREF(c_in);
        PyErr_SetString(PyExc_ValueError, "Failed to convert input arrays");
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_in, 0);
    i32 m = (i32)PyArray_DIM(b_in, 1);
    i32 p = (i32)PyArray_DIM(c_in, 0);

    f64 tol[3] = {0.0, 0.0, 0.0};
    if (tol_obj != Py_None) {
        PyArrayObject *tol_arr = (PyArrayObject*)PyArray_FROM_OTF(tol_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (!tol_arr) {
            Py_DECREF(a_in);
            Py_DECREF(e_in);
            Py_DECREF(b_in);
            Py_DECREF(c_in);
            PyErr_SetString(PyExc_ValueError, "tol must be array of 3 floats");
            return NULL;
        }
        npy_intp ntol = PyArray_SIZE(tol_arr);
        f64 *tol_data = (f64*)PyArray_DATA(tol_arr);
        for (npy_intp i = 0; i < ntol && i < 3; i++) {
            tol[i] = tol_data[i];
        }
        Py_DECREF(tol_arr);
    }

    i32 lda = (n > 1) ? n : 1;
    i32 lde = (n > 1) ? n : 1;
    i32 ldb = (n > 1) ? n : 1;
    i32 maxmp = (m > p) ? m : p;
    i32 ldc = (maxmp > 1) ? maxmp : 1;
    if (n == 0) ldc = 1;

    bool ljobir = (job[0] == 'I' || job[0] == 'i');
    bool ljobo = ljobir || (job[0] == 'O' || job[0] == 'o');
    bool lsysr = (systyp[0] == 'R' || systyp[0] == 'r');
    bool lequil = (equil[0] == 'S' || equil[0] == 's');
    bool lsing = (cksing[0] == 'C' || cksing[0] == 'c');
    bool lrestor = (restor[0] == 'R' || restor[0] == 'r');

    i32 c_mult = (ljobir || lsysr) ? 2 : 1;
    i32 liwork = c_mult * n + maxmp;
    if (liwork < 1) liwork = 1;

    i32 nn = n * n;
    i32 k = n * (2*n + m + p);
    i32 ldwork;
    if (lrestor) {
        i32 t1 = 2 * (k + maxmp + n - 1);
        i32 t2 = nn + 4*n;
        ldwork = (t1 > t2) ? t1 : t2;
    } else {
        i32 t1 = 2 * (maxmp + n - 1);
        i32 t2 = nn + 4*n;
        ldwork = (t1 > t2) ? t1 : t2;
    }
    if (lequil) {
        i32 equil_min = 8*n > 2*maxmp ? 8*n : 2*maxmp;
        if (ldwork < equil_min) ldwork = equil_min;
    }
    if (lsing) {
        i32 n23 = (n > 23) ? n : 23;
        i32 sing_min = 2*nn + 10*n + n23;
        if (ldwork < sing_min) ldwork = sing_min;
    }
    if (ldwork < 1) ldwork = 1;

    i32 *iwork = (i32*)calloc(liwork, sizeof(i32));
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        Py_DECREF(a_in);
        Py_DECREF(e_in);
        Py_DECREF(b_in);
        Py_DECREF(c_in);
        return PyErr_NoMemory();
    }

    i32 b_cols = ljobo ? maxmp : m;
    npy_intp a_dims[2] = {n, n};
    npy_intp a_strides[2] = {sizeof(f64), n * sizeof(f64)};
    PyObject *a_array = PyArray_New(&PyArray_Type, 2, a_dims, NPY_DOUBLE, a_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    npy_intp e_dims[2] = {n, n};
    npy_intp e_strides[2] = {sizeof(f64), n * sizeof(f64)};
    PyObject *e_array = PyArray_New(&PyArray_Type, 2, e_dims, NPY_DOUBLE, e_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    npy_intp b_dims[2] = {n, b_cols};
    npy_intp b_strides[2] = {sizeof(f64), ldb * sizeof(f64)};
    PyObject *b_array = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    npy_intp c_dims[2] = {ldc, n};
    npy_intp c_strides[2] = {sizeof(f64), ldc * sizeof(f64)};
    PyObject *c_array = PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE, c_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (!a_array || !e_array || !b_array || !c_array) {
        free(iwork);
        free(dwork);
        Py_DECREF(a_in);
        Py_DECREF(e_in);
        Py_DECREF(b_in);
        Py_DECREF(c_in);
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        return PyErr_NoMemory();
    }

    f64 *a_data = (f64*)PyArray_DATA((PyArrayObject*)a_array);
    f64 *e_data = (f64*)PyArray_DATA((PyArrayObject*)e_array);
    f64 *b_data = (f64*)PyArray_DATA((PyArrayObject*)b_array);
    f64 *c_data = (f64*)PyArray_DATA((PyArrayObject*)c_array);

    if (n > 0) {
        memcpy(a_data, PyArray_DATA(a_in), n * n * sizeof(f64));
        memcpy(e_data, PyArray_DATA(e_in), n * n * sizeof(f64));

        if (n > 0 && b_cols > 0) { memset(b_data, 0, ldb * b_cols * sizeof(f64)); }
        for (i32 j = 0; j < m; j++) {
            memcpy(&b_data[j * ldb], &((f64*)PyArray_DATA(b_in))[j * n], n * sizeof(f64));
        }

        if (ldc > 0 && n > 0) { memset(c_data, 0, ldc * n * sizeof(f64)); }
        for (i32 j = 0; j < n; j++) {
            memcpy(&c_data[j * ldc], &((f64*)PyArray_DATA(c_in))[j * p], p * sizeof(f64));
        }
    }

    Py_DECREF(a_in);
    Py_DECREF(e_in);
    Py_DECREF(b_in);
    Py_DECREF(c_in);

    i32 nr = 0;
    i32 infred[7] = {0};
    i32 info = 0;

    tg01jy(job, systyp, equil, cksing, restor, n, m, p,
           a_data, lda, e_data, lde, b_data, ldb, c_data, ldc,
           &nr, infred, tol, iwork, dwork, ldwork, &info);

    npy_intp infred_dims[1] = {7};
    PyObject *infred_array = PyArray_SimpleNew(1, infred_dims, NPY_INT32);
    memcpy(PyArray_DATA((PyArrayObject*)infred_array), infred, 7 * sizeof(i32));

    i32 nblck = (infred[6] > 0) ? infred[6] : 0;
    npy_intp iwork_out_dims[1] = {nblck > 0 ? nblck : 1};
    PyObject *iwork_array = PyArray_SimpleNew(1, iwork_out_dims, NPY_INT32);
    if (nblck > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)iwork_array), iwork, nblck * sizeof(i32));
    }

    free(iwork);
    free(dwork);

    npy_intp ar_dims[2] = {nr, nr};
    npy_intp ar_strides[2] = {sizeof(f64), nr * sizeof(f64)};
    PyObject *ar_array = PyArray_New(&PyArray_Type, 2, ar_dims, NPY_DOUBLE, ar_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    npy_intp er_dims[2] = {nr, nr};
    npy_intp er_strides[2] = {sizeof(f64), nr * sizeof(f64)};
    PyObject *er_array = PyArray_New(&PyArray_Type, 2, er_dims, NPY_DOUBLE, er_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    npy_intp br_dims[2] = {nr, m};
    npy_intp br_strides[2] = {sizeof(f64), nr * sizeof(f64)};
    PyObject *br_array = PyArray_New(&PyArray_Type, 2, br_dims, NPY_DOUBLE, br_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    npy_intp cr_dims[2] = {p, nr};
    npy_intp cr_strides[2] = {sizeof(f64), p * sizeof(f64)};
    PyObject *cr_array = PyArray_New(&PyArray_Type, 2, cr_dims, NPY_DOUBLE, cr_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (!ar_array || !er_array || !br_array || !cr_array) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(infred_array);
        Py_DECREF(iwork_array);
        Py_XDECREF(ar_array);
        Py_XDECREF(er_array);
        Py_XDECREF(br_array);
        Py_XDECREF(cr_array);
        return PyErr_NoMemory();
    }

    f64 *ar_data = (f64*)PyArray_DATA((PyArrayObject*)ar_array);
    f64 *er_data = (f64*)PyArray_DATA((PyArrayObject*)er_array);
    f64 *br_data = (f64*)PyArray_DATA((PyArrayObject*)br_array);
    f64 *cr_data = (f64*)PyArray_DATA((PyArrayObject*)cr_array);

    for (i32 j = 0; j < nr; j++) {
        memcpy(&ar_data[j * nr], &a_data[j * lda], nr * sizeof(f64));
        memcpy(&er_data[j * nr], &e_data[j * lde], nr * sizeof(f64));
    }
    for (i32 j = 0; j < m; j++) {
        memcpy(&br_data[j * nr], &b_data[j * ldb], nr * sizeof(f64));
    }
    for (i32 j = 0; j < nr; j++) {
        memcpy(&cr_data[j * p], &c_data[j * ldc], p * sizeof(f64));
    }

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);

    PyObject *result = Py_BuildValue("(OOOOiOOi)", ar_array, er_array, br_array, cr_array,
                                     nr, infred_array, iwork_array, info);

    Py_DECREF(ar_array);
    Py_DECREF(er_array);
    Py_DECREF(br_array);
    Py_DECREF(cr_array);
    Py_DECREF(infred_array);
    Py_DECREF(iwork_array);

    return result;
}

PyObject* py_tg01oa(PyObject* self, PyObject* args) {
    (void)self;
    const char *jobe;
    PyObject *dcba_obj, *e_obj;
    PyArrayObject *dcba_array, *e_array;
    i32 info = 0;

    if (!PyArg_ParseTuple(args, "sOO", &jobe, &dcba_obj, &e_obj)) {
        return NULL;
    }

    bool unite = (jobe[0] == 'I' || jobe[0] == 'i');
    bool upper = (jobe[0] == 'U' || jobe[0] == 'u');
    if (!unite && !upper) {
        PyErr_Format(PyExc_ValueError, "jobe must be 'U' or 'I', got '%s'", jobe);
        return NULL;
    }

    dcba_array = (PyArrayObject*)PyArray_FROM_OTF(dcba_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (dcba_array == NULL || e_array == NULL) {
        Py_XDECREF(dcba_array);
        Py_XDECREF(e_array);
        PyErr_SetString(PyExc_ValueError, "Failed to convert input arrays");
        return NULL;
    }

    npy_intp *dcba_dims = PyArray_DIMS(dcba_array);
    i32 n1 = (i32)dcba_dims[0];
    i32 n = n1 - 1;
    i32 lddcba = n1;

    npy_intp *e_dims = PyArray_DIMS(e_array);
    i32 lde = (i32)e_dims[0];
    if (lde < 1) lde = 1;

    f64 *dcba_data = (f64*)PyArray_DATA(dcba_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);

    tg01oa(jobe, n, dcba_data, lddcba, e_data, lde, &info);

    PyArray_ResolveWritebackIfCopy(dcba_array);
    PyArray_ResolveWritebackIfCopy(e_array);

    PyObject *result = Py_BuildValue("(OOi)", dcba_array, e_array, info);

    Py_DECREF(dcba_array);
    Py_DECREF(e_array);

    return result;
}

PyObject* py_tg01ob(PyObject* self, PyObject* args) {
    (void)self;

    const char *jobe;
    PyObject *dcba_obj, *e_obj;

    if (!PyArg_ParseTuple(args, "sOO", &jobe, &dcba_obj, &e_obj)) {
        return NULL;
    }

    PyArrayObject *dcba_array = (PyArrayObject*)PyArray_FROM_OTF(
        dcba_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(
        e_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (dcba_array == NULL || e_array == NULL) {
        Py_XDECREF(dcba_array);
        Py_XDECREF(e_array);
        return NULL;
    }

    int ndim_dcba = PyArray_NDIM(dcba_array);
    npy_intp *dcba_dims = PyArray_DIMS(dcba_array);

    i32 n1, n;
    if (ndim_dcba == 2 && dcba_dims[0] > 0) {
        n1 = (i32)dcba_dims[0];
        n = n1 - 1;
    } else if (ndim_dcba == 2 && dcba_dims[0] == 0) {
        n1 = 0;
        n = -1;
    } else {
        Py_DECREF(dcba_array);
        Py_DECREF(e_array);
        PyErr_SetString(PyExc_ValueError, "DCBA must be a 2D array");
        return NULL;
    }

    i32 info = 0;

    bool unite = (jobe[0] == 'I' || jobe[0] == 'i');
    i32 lddcba = (n1 > 0) ? n1 : 1;
    i32 lde = unite ? 1 : ((n > 0) ? n : 1);

    c128 *dcba_data = (c128*)PyArray_DATA(dcba_array);
    c128 *e_data = (c128*)PyArray_DATA(e_array);

    tg01ob(jobe, n, dcba_data, lddcba, e_data, lde, &info);

    PyArray_ResolveWritebackIfCopy(dcba_array);
    PyArray_ResolveWritebackIfCopy(e_array);

    PyObject *result = Py_BuildValue("(OOi)", dcba_array, e_array, info);

    Py_DECREF(dcba_array);
    Py_DECREF(e_array);

    return result;
}

PyObject* py_tg01ld(PyObject* self, PyObject* args) {
    (void)self;

    const char *job, *joba, *compq, *compz;
    i32 n, m, p;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj;
    f64 tol;
    PyArrayObject *a_array, *e_array, *b_array, *c_array;
    i32 nf = 0, nd = 0, niblck = 0, info = 0;
    i32 lda, lde, ldb, ldc, ldq, ldz;

    if (!PyArg_ParseTuple(args, "ssssiiiOOOOd", &job, &joba, &compq, &compz,
                          &n, &m, &p, &a_obj, &e_obj, &b_obj, &c_obj, &tol)) {
        return NULL;
    }

    if (n < 0 || m < 0 || p < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative");
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || b_array == NULL || c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *e_dims = PyArray_DIMS(e_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    lda = (n > 0) ? (i32)a_dims[0] : 1;
    lde = (n > 0) ? (i32)e_dims[0] : 1;
    ldb = (n > 0) ? (i32)b_dims[0] : 1;
    ldc = (p > 0) ? (i32)c_dims[0] : 1;
    ldq = (n > 0) ? n : 1;
    ldz = (n > 0) ? n : 1;

    i32 temp1 = (3 * n > m) ? 3 * n : m;
    temp1 = (temp1 > p) ? temp1 : p;
    i32 ldwork = (n > 0) ? (n + temp1) : 1;
    ldwork = (ldwork > 1) ? ldwork : 1;

    i32 *iwork = (n > 0) ? (i32*)malloc(n * sizeof(i32)) : NULL;
    i32 *iblck = (n > 0) ? (i32*)malloc(n * sizeof(i32)) : NULL;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    f64 *q = NULL;
    f64 *z = NULL;

    bool compq_needed = (compq[0] == 'I' || compq[0] == 'i' || compq[0] == 'U' || compq[0] == 'u');
    bool compz_needed = (compz[0] == 'I' || compz[0] == 'i' || compz[0] == 'U' || compz[0] == 'u');

    npy_intp q_dims[2] = {n, n};
    npy_intp z_dims[2] = {n, n};
    npy_intp q_strides[2] = {sizeof(f64), n * sizeof(f64)};
    npy_intp z_strides[2] = {sizeof(f64), n * sizeof(f64)};

    PyObject *q_array = NULL;
    PyObject *z_array = NULL;

    if (compq_needed && n > 0) {
        q_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (q_array == NULL) {
             free(iwork); free(iblck); free(dwork);
             Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(c_array);
             return PyErr_NoMemory();
        }
        q = (f64*)PyArray_DATA((PyArrayObject*)q_array);
        memset(q, 0, n * n * sizeof(f64));
    } else {
        q_array = PyArray_EMPTY(2, q_dims, NPY_DOUBLE, 1);
    }

    if (compz_needed && n > 0) {
        z_array = PyArray_New(&PyArray_Type, 2, z_dims, NPY_DOUBLE, z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (z_array == NULL) {
            free(iwork); free(iblck); free(dwork);
            Py_DECREF(q_array);
            Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(c_array);
            return PyErr_NoMemory();
        }
        z = (f64*)PyArray_DATA((PyArrayObject*)z_array);
        memset(z, 0, n * n * sizeof(f64));
    } else {
        z_array = PyArray_EMPTY(2, z_dims, NPY_DOUBLE, 1);
    }

    if (dwork == NULL || (n > 0 && (iwork == NULL || iblck == NULL))) {
        free(iwork);
        free(iblck);
        free(dwork);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    tg01ld(job, joba, compq, compz, n, m, p, a_data, lda, e_data, lde,
           b_data, ldb, c_data, ldc, q, ldq, z, ldz, &nf, &nd, &niblck, iblck,
           tol, iwork, dwork, ldwork, &info);

    npy_intp iblck_dims[1] = {(niblck > 0) ? niblck : 1};
    PyObject *iblck_array = PyArray_SimpleNew(1, iblck_dims, NPY_INT32);
    if (niblck > 0 && iblck != NULL) {
        memcpy(PyArray_DATA((PyArrayObject*)iblck_array), iblck, niblck * sizeof(i32));
    }

    free(iwork);
    free(iblck);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("(OOOOOOiiiOi)", a_array, e_array, b_array, c_array,
                                     q_array, z_array, nf, nd, niblck, iblck_array, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);
    Py_DECREF(iblck_array);

    return result;
}

PyObject* py_tg01ly(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;

    int compq_int, compz_int;
    i32 ranke, rnka22;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj, *q_obj, *z_obj;
    f64 tol = 0.0;
    i32 ldwork = 0;
    PyArrayObject *a_array, *e_array, *b_array, *c_array, *q_array, *z_array;
    i32 nf = 0, niblck = 0, info = 0;
    i32 n, m, p;
    i32 lda, lde, ldb, ldc, ldq, ldz;

    static char *kwlist[] = {"compq", "compz", "ranke", "rnka22", "a", "e", "b", "c", "q", "z", "tol", "ldwork", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ppiiOOOOOO|di", kwlist,
                                     &compq_int, &compz_int, &ranke, &rnka22,
                                     &a_obj, &e_obj, &b_obj, &c_obj, &q_obj, &z_obj,
                                     &tol, &ldwork)) {
        return NULL;
    }

    bool compq = (bool)compq_int;
    bool compz = (bool)compz_int;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    z_array = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || b_array == NULL || c_array == NULL ||
        q_array == NULL || z_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(q_array);
        Py_XDECREF(z_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    n = (i32)a_dims[0];
    m = (PyArray_NDIM(b_array) == 1) ? 1 : (i32)b_dims[1];
    p = (i32)c_dims[0];

    lda = (n > 0) ? n : 1;
    lde = (n > 0) ? n : 1;
    ldb = (n > 0) ? n : 1;
    ldc = (p > 0) ? p : 1;
    ldq = (compq && n > 0) ? n : 1;
    ldz = (compz && n > 0) ? n : 1;

    i32 nd = n - ranke;
    i32 minwrk;
    if (ranke == n) {
        minwrk = 1;
    } else {
        i32 max_nm = n > m ? n : m;
        minwrk = 4 * nd - 1;
        i32 alt = nd + max_nm;
        if (alt > minwrk) minwrk = alt;
    }

    if (ldwork <= 0 && ldwork != -1) {
        ldwork = minwrk + n;
    }

    i32 *iwork = (n > 0) ? (i32*)malloc(n * sizeof(i32)) : NULL;
    i32 *iblck = (n > 0) ? (i32*)malloc(n * sizeof(i32)) : NULL;
    f64 *dwork = (f64*)malloc((ldwork > 0 ? ldwork : 1) * sizeof(f64));

    if (dwork == NULL || (n > 0 && (iwork == NULL || iblck == NULL))) {
        free(iwork);
        free(iblck);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *q_data = (f64*)PyArray_DATA(q_array);
    f64 *z_data = (f64*)PyArray_DATA(z_array);

    tg01ly(compq, compz, n, m, p, ranke, rnka22,
           a_data, lda, e_data, lde, b_data, ldb, c_data, ldc,
           q_data, ldq, z_data, ldz,
           &nf, &niblck, iblck, tol, iwork, dwork, ldwork, &info);

    npy_intp iblck_dims[1] = {(niblck > 0) ? niblck : 1};
    PyObject *iblck_array = PyArray_SimpleNew(1, iblck_dims, NPY_INT32);
    if (niblck > 0 && iblck != NULL) {
        memcpy(PyArray_DATA((PyArrayObject*)iblck_array), iblck, niblck * sizeof(i32));
    }

    free(iwork);
    free(iblck);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(q_array);
    PyArray_ResolveWritebackIfCopy(z_array);

    PyObject *result = Py_BuildValue("(OOOOOOiiOi)",
                                     a_array, e_array, b_array, c_array,
                                     q_array, z_array, nf, niblck, iblck_array, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);
    Py_DECREF(iblck_array);

    return result;
}

PyObject* py_tg01md(PyObject* self, PyObject* args) {
    (void)self;

    const char *job;
    i32 n, m, p;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj;
    f64 tol;
    PyArrayObject *a_array, *e_array, *b_array, *c_array;
    i32 nf = 0, nd = 0, niblck = 0, info = 0;
    i32 lda, lde, ldb, ldc, ldq, ldz;

    if (!PyArg_ParseTuple(args, "siiiOOOOd", &job,
                          &n, &m, &p, &a_obj, &e_obj, &b_obj, &c_obj, &tol)) {
        return NULL;
    }

    if (n < 0 || m < 0 || p < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative");
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || b_array == NULL || c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *e_dims = PyArray_DIMS(e_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    lda = (n > 0) ? (i32)a_dims[0] : 1;
    lde = (n > 0) ? (i32)e_dims[0] : 1;
    ldb = (n > 0) ? (i32)b_dims[0] : 1;
    ldc = (p > 0) ? (i32)c_dims[0] : 1;
    ldq = (n > 0) ? n : 1;
    ldz = (n > 0) ? n : 1;

    i32 ldwork = (n > 0) ? (4 * n) : 1;
    ldwork = (ldwork > 1) ? ldwork : 1;

    i32 *iwork = (n > 0) ? (i32*)malloc(n * sizeof(i32)) : NULL;
    i32 *iblck = (n > 0) ? (i32*)malloc(n * sizeof(i32)) : NULL;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    npy_intp q_dims[2] = {n, n};
    npy_intp z_dims[2] = {n, n};
    npy_intp q_strides[2] = {sizeof(f64), n * sizeof(f64)};
    npy_intp z_strides[2] = {sizeof(f64), n * sizeof(f64)};
    npy_intp eig_dims[1] = {n};

    PyObject *q_array = NULL;
    PyObject *z_array = NULL;
    PyObject *alphar_array = NULL;
    PyObject *alphai_array = NULL;
    PyObject *beta_array = NULL;

    if (n > 0) {
        q_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        z_array = PyArray_New(&PyArray_Type, 2, z_dims, NPY_DOUBLE, z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        alphar_array = PyArray_EMPTY(1, eig_dims, NPY_DOUBLE, 0);
        alphai_array = PyArray_EMPTY(1, eig_dims, NPY_DOUBLE, 0);
        beta_array = PyArray_EMPTY(1, eig_dims, NPY_DOUBLE, 0);
    } else {
        npy_intp zero_dims[2] = {0, 0};
        npy_intp zero_vec[1] = {0};
        q_array = PyArray_EMPTY(2, zero_dims, NPY_DOUBLE, 1);
        z_array = PyArray_EMPTY(2, zero_dims, NPY_DOUBLE, 1);
        alphar_array = PyArray_EMPTY(1, zero_vec, NPY_DOUBLE, 0);
        alphai_array = PyArray_EMPTY(1, zero_vec, NPY_DOUBLE, 0);
        beta_array = PyArray_EMPTY(1, zero_vec, NPY_DOUBLE, 0);
    }

    if (q_array == NULL || z_array == NULL || alphar_array == NULL ||
        alphai_array == NULL || beta_array == NULL ||
        dwork == NULL || (n > 0 && (iwork == NULL || iblck == NULL))) {
        free(iwork);
        free(iblck);
        free(dwork);
        Py_XDECREF(q_array);
        Py_XDECREF(z_array);
        Py_XDECREF(alphar_array);
        Py_XDECREF(alphai_array);
        Py_XDECREF(beta_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *q = (n > 0) ? (f64*)PyArray_DATA((PyArrayObject*)q_array) : NULL;
    f64 *z = (n > 0) ? (f64*)PyArray_DATA((PyArrayObject*)z_array) : NULL;
    f64 *alphar = (n > 0) ? (f64*)PyArray_DATA((PyArrayObject*)alphar_array) : NULL;
    f64 *alphai = (n > 0) ? (f64*)PyArray_DATA((PyArrayObject*)alphai_array) : NULL;
    f64 *beta = (n > 0) ? (f64*)PyArray_DATA((PyArrayObject*)beta_array) : NULL;

    if (n > 0) {
        memset(q, 0, n * n * sizeof(f64));
        memset(z, 0, n * n * sizeof(f64));
    }

    tg01md(job, n, m, p, a_data, lda, e_data, lde, b_data, ldb, c_data, ldc,
           alphar, alphai, beta, q, ldq, z, ldz,
           &nf, &nd, &niblck, iblck, tol, iwork, dwork, ldwork, &info);

    npy_intp iblck_dims[1] = {(niblck > 0) ? niblck : 1};
    PyObject *iblck_array = PyArray_SimpleNew(1, iblck_dims, NPY_INT32);
    if (niblck > 0 && iblck != NULL) {
        memcpy(PyArray_DATA((PyArrayObject*)iblck_array), iblck, niblck * sizeof(i32));
    }

    free(iwork);
    free(iblck);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("(OOOOOOOOOiiiOi)",
                                     a_array, e_array, b_array, c_array,
                                     alphar_array, alphai_array, beta_array,
                                     q_array, z_array,
                                     nf, nd, niblck, iblck_array, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);
    Py_DECREF(iblck_array);

    return result;
}

PyObject* py_tg01od(PyObject* self, PyObject* args) {
    (void)self;
    const char *jobe;
    PyObject *dcba_obj, *e_obj;
    f64 tol = 0.0;
    i32 info = 0;
    i32 nz = 0;
    f64 g = 0.0;

    if (!PyArg_ParseTuple(args, "sOO|d", &jobe, &dcba_obj, &e_obj, &tol)) {
        return NULL;
    }

    bool descr = (jobe[0] == 'G' || jobe[0] == 'g');
    bool ident = (jobe[0] == 'I' || jobe[0] == 'i');
    if (!descr && !ident) {
        PyErr_Format(PyExc_ValueError, "jobe must be 'G' or 'I', got '%s'", jobe);
        return NULL;
    }

    PyArrayObject *dcba_array = (PyArrayObject*)PyArray_FROM_OTF(dcba_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (dcba_array == NULL || e_array == NULL) {
        Py_XDECREF(dcba_array);
        Py_XDECREF(e_array);
        PyErr_SetString(PyExc_ValueError, "Failed to convert input arrays");
        return NULL;
    }

    npy_intp *dcba_dims = PyArray_DIMS(dcba_array);
    i32 n1 = (i32)dcba_dims[0];
    i32 n = n1 - 1;
    i32 lddcba = n1;

    npy_intp *e_dims = PyArray_DIMS(e_array);
    i32 lde = (i32)e_dims[0];
    if (lde < 1) lde = 1;

    i32 ldwork;
    if (descr) {
        ldwork = 2 * n + 1;
        if (ldwork < 1) ldwork = 1;
    } else {
        ldwork = n + 1;
        if (ldwork < 1) ldwork = 1;
    }

    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(dcba_array);
        Py_DECREF(e_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    f64 *dcba_data = (f64*)PyArray_DATA(dcba_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);

    tg01od(jobe, n, dcba_data, lddcba, e_data, lde, &nz, &g, tol, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(dcba_array);
    PyArray_ResolveWritebackIfCopy(e_array);

    PyObject *result = Py_BuildValue("(OOidi)", dcba_array, e_array, nz, g, info);

    Py_DECREF(dcba_array);
    Py_DECREF(e_array);

    return result;
}


PyObject* py_tg01pd(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;
    const char *dico, *stdom, *jobae, *compq, *compz;
    i32 n, m, p, nlow, nsup;
    f64 alpha;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj;
    PyObject *q_obj = NULL, *z_obj = NULL;
    i32 info = 0;

    static char *kwlist[] = {"dico", "stdom", "jobae", "compq", "compz",
                             "n", "m", "p", "nlow", "nsup", "alpha",
                             "a", "e", "b", "c", "q", "z", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssssiiiiidOOOO|OO", kwlist,
                                     &dico, &stdom, &jobae, &compq, &compz,
                                     &n, &m, &p, &nlow, &nsup, &alpha,
                                     &a_obj, &e_obj, &b_obj, &c_obj,
                                     &q_obj, &z_obj)) {
        return NULL;
    }

    bool ljobg = (jobae[0] == 'G' || jobae[0] == 'g');
    bool compq_u = (compq[0] == 'U' || compq[0] == 'u');
    bool compz_u = (compz[0] == 'U' || compz[0] == 'u');

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || b_array == NULL || c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        PyErr_SetString(PyExc_ValueError, "Failed to convert input arrays");
        return NULL;
    }

    i32 lda = n > 1 ? n : 1;
    i32 lde = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldc = p > 1 ? p : 1;
    i32 ldq = n > 1 ? n : 1;
    i32 ldz = n > 1 ? n : 1;

    npy_intp q_dims[2] = {n > 0 ? n : 1, n > 0 ? n : 1};
    npy_intp q_strides[2] = {sizeof(f64), (n > 0 ? n : 1) * (npy_intp)sizeof(f64)};
    PyObject *q_array_new = NULL;
    PyObject *z_array_new = NULL;

    f64 *q_data = NULL;
    f64 *z_data = NULL;

    if (compq_u && q_obj != NULL) {
        PyArrayObject *q_in = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE,
                                                       NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (q_in == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            PyErr_SetString(PyExc_ValueError, "Failed to convert Q array");
            return NULL;
        }
        q_array_new = (PyObject*)q_in;
        q_data = (f64*)PyArray_DATA((PyArrayObject*)q_array_new);
    } else {
        q_array_new = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides,
                                   NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (q_array_new == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            return PyErr_NoMemory();
        }
        q_data = (f64*)PyArray_DATA((PyArrayObject*)q_array_new);
    }

    if (compz_u && z_obj != NULL) {
        PyArrayObject *z_in = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE,
                                                       NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (z_in == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            Py_DECREF(q_array_new);
            PyErr_SetString(PyExc_ValueError, "Failed to convert Z array");
            return NULL;
        }
        z_array_new = (PyObject*)z_in;
        z_data = (f64*)PyArray_DATA((PyArrayObject*)z_array_new);
    } else {
        z_array_new = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides,
                                   NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (z_array_new == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            Py_DECREF(q_array_new);
            return PyErr_NoMemory();
        }
        z_data = (f64*)PyArray_DATA((PyArrayObject*)z_array_new);
    }

    i32 ldwork;
    if (n == 0) {
        ldwork = 1;
    } else if (ljobg) {
        ldwork = 8 * n + 16;
    } else {
        ldwork = 4 * n + 16;
    }

    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(q_array_new);
        Py_DECREF(z_array_new);
        return PyErr_NoMemory();
    }

    npy_intp eig_dims[1] = {n > 0 ? n : 1};
    PyObject *alphar_array = PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);
    PyObject *alphai_array = PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);
    PyObject *beta_array = PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);

    if (alphar_array == NULL || alphai_array == NULL || beta_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(q_array_new);
        Py_DECREF(z_array_new);
        Py_XDECREF(alphar_array);
        Py_XDECREF(alphai_array);
        Py_XDECREF(beta_array);
        free(dwork);
        return PyErr_NoMemory();
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *alphar_data = (f64*)PyArray_DATA((PyArrayObject*)alphar_array);
    f64 *alphai_data = (f64*)PyArray_DATA((PyArrayObject*)alphai_array);
    f64 *beta_data = (f64*)PyArray_DATA((PyArrayObject*)beta_array);

    i32 ndim = 0;

    tg01pd(dico, stdom, jobae, compq, compz, n, m, p, nlow, nsup, alpha,
           a_data, lda, e_data, lde, b_data, ldb, c_data, ldc,
           q_data, ldq, z_data, ldz, &ndim, alphar_data, alphai_data, beta_data,
           dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    if (compq_u && q_obj != NULL) {
        PyArray_ResolveWritebackIfCopy((PyArrayObject*)q_array_new);
    }
    if (compz_u && z_obj != NULL) {
        PyArray_ResolveWritebackIfCopy((PyArrayObject*)z_array_new);
    }

    PyObject *result = Py_BuildValue("(OOOOOOiOOOi)",
                                     a_array, e_array, b_array, c_array,
                                     q_array_new, z_array_new, ndim,
                                     alphar_array, alphai_array, beta_array, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(q_array_new);
    Py_DECREF(z_array_new);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);

    return result;
}

PyObject* py_tg01oz(PyObject* self, PyObject* args) {
    (void)self;
    const char *jobe;
    PyObject *dcba_obj, *e_obj;
    f64 tol = 0.0;
    i32 info = 0;
    i32 nz = 0;
    c128 g = 0.0 + 0.0 * I;

    if (!PyArg_ParseTuple(args, "sOO|d", &jobe, &dcba_obj, &e_obj, &tol)) {
        return NULL;
    }

    bool descr = (jobe[0] == 'G' || jobe[0] == 'g');
    bool ident = (jobe[0] == 'I' || jobe[0] == 'i');
    if (!descr && !ident) {
        PyErr_Format(PyExc_ValueError, "jobe must be 'G' or 'I', got '%s'", jobe);
        return NULL;
    }

    PyArrayObject *dcba_array = (PyArrayObject*)PyArray_FROM_OTF(dcba_obj, NPY_COMPLEX128,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_COMPLEX128,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (dcba_array == NULL || e_array == NULL) {
        Py_XDECREF(dcba_array);
        Py_XDECREF(e_array);
        PyErr_SetString(PyExc_ValueError, "Failed to convert input arrays");
        return NULL;
    }

    npy_intp *dcba_dims = PyArray_DIMS(dcba_array);
    i32 n1 = (i32)dcba_dims[0];
    i32 n = n1 - 1;
    i32 lddcba = n1;

    npy_intp *e_dims = PyArray_DIMS(e_array);
    i32 lde = (i32)e_dims[0];
    if (lde < 1) lde = 1;

    i32 lzwork;
    if (descr) {
        lzwork = 2 * n + 1;
        if (lzwork < 1) lzwork = 1;
    } else {
        lzwork = n + 1;
        if (lzwork < 1) lzwork = 1;
    }

    c128 *zwork = (c128*)PyMem_Calloc(lzwork, sizeof(c128));
    if (zwork == NULL) {
        Py_DECREF(dcba_array);
        Py_DECREF(e_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    c128 *dcba_data = (c128*)PyArray_DATA(dcba_array);
    c128 *e_data = (c128*)PyArray_DATA(e_array);

    tg01oz(jobe, n, dcba_data, lddcba, e_data, lde, &nz, &g, tol, zwork, lzwork, &info);

    PyMem_Free(zwork);

    PyArray_ResolveWritebackIfCopy(dcba_array);
    PyArray_ResolveWritebackIfCopy(e_array);

    PyObject *result = Py_BuildValue("(OOiDi)", dcba_array, e_array, nz, &g, info);

    Py_DECREF(dcba_array);
    Py_DECREF(e_array);

    return result;
}

PyObject* py_tg01wd(PyObject* self, PyObject* args) {
    (void)self;
    i32 n, m, p;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj;
    i32 info = 0;

    if (!PyArg_ParseTuple(args, "iiiOOOO", &n, &m, &p, &a_obj, &e_obj, &b_obj, &c_obj)) {
        return NULL;
    }

    i32 max_1_n = (1 > n) ? 1 : n;
    i32 max_1_p = (1 > p) ? 1 : p;

    if (n < 0) {
        info = -1;
    } else if (m < 0) {
        info = -2;
    } else if (p < 0) {
        info = -3;
    }

    if (info != 0) {
        npy_intp zero_dims[2] = {0, 0};
        npy_intp zero_1d[1] = {0};
        PyObject *a_empty = PyArray_New(&PyArray_Type, 2, zero_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        PyObject *e_empty = PyArray_New(&PyArray_Type, 2, zero_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        PyObject *b_empty = PyArray_New(&PyArray_Type, 2, zero_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        PyObject *c_empty = PyArray_New(&PyArray_Type, 2, zero_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        PyObject *q_empty = PyArray_New(&PyArray_Type, 2, zero_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        PyObject *z_empty = PyArray_New(&PyArray_Type, 2, zero_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        PyObject *alphar_empty = PyArray_New(&PyArray_Type, 1, zero_1d, NPY_DOUBLE, NULL, NULL, 0, 0, NULL);
        PyObject *alphai_empty = PyArray_New(&PyArray_Type, 1, zero_1d, NPY_DOUBLE, NULL, NULL, 0, 0, NULL);
        PyObject *beta_empty = PyArray_New(&PyArray_Type, 1, zero_1d, NPY_DOUBLE, NULL, NULL, 0, 0, NULL);
        PyObject *result = Py_BuildValue("(OOOOOOOOOi)",
                                         a_empty, e_empty, b_empty, c_empty,
                                         q_empty, z_empty,
                                         alphar_empty, alphai_empty, beta_empty, info);
        Py_DECREF(a_empty);
        Py_DECREF(e_empty);
        Py_DECREF(b_empty);
        Py_DECREF(c_empty);
        Py_DECREF(q_empty);
        Py_DECREF(z_empty);
        Py_DECREF(alphar_empty);
        Py_DECREF(alphai_empty);
        Py_DECREF(beta_empty);
        return result;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || b_array == NULL || c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        PyErr_SetString(PyExc_ValueError, "Failed to convert input arrays");
        return NULL;
    }

    i32 lda = max_1_n;
    i32 lde = max_1_n;
    i32 ldb = max_1_n;
    i32 ldc = max_1_p;
    i32 ldq = max_1_n;
    i32 ldz = max_1_n;

    npy_intp q_dims[2] = {n, n};
    npy_intp q_strides[2] = {sizeof(f64), ldq * (npy_intp)sizeof(f64)};
    PyObject *q_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE,
                                    q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *z_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE,
                                    q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    npy_intp eig_dims[1] = {n};
    PyObject *alphar_array = PyArray_New(&PyArray_Type, 1, eig_dims, NPY_DOUBLE,
                                         NULL, NULL, 0, 0, NULL);
    PyObject *alphai_array = PyArray_New(&PyArray_Type, 1, eig_dims, NPY_DOUBLE,
                                         NULL, NULL, 0, 0, NULL);
    PyObject *beta_array = PyArray_New(&PyArray_Type, 1, eig_dims, NPY_DOUBLE,
                                       NULL, NULL, 0, 0, NULL);

    if (q_array == NULL || z_array == NULL ||
        alphar_array == NULL || alphai_array == NULL || beta_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_XDECREF(q_array);
        Py_XDECREF(z_array);
        Py_XDECREF(alphar_array);
        Py_XDECREF(alphai_array);
        Py_XDECREF(beta_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output arrays");
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *q_data = (f64*)PyArray_DATA((PyArrayObject*)q_array);
    f64 *z_data = (f64*)PyArray_DATA((PyArrayObject*)z_array);
    f64 *alphar_data = (f64*)PyArray_DATA((PyArrayObject*)alphar_array);
    f64 *alphai_data = (f64*)PyArray_DATA((PyArrayObject*)alphai_array);
    f64 *beta_data = (f64*)PyArray_DATA((PyArrayObject*)beta_array);

    i32 ldwork = 8 * n + 16;
    if (ldwork < 1) ldwork = 1;
    if (n * m > ldwork) ldwork = n * m;
    if (p * n > ldwork) ldwork = p * n;

    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        Py_DECREF(alphar_array);
        Py_DECREF(alphai_array);
        Py_DECREF(beta_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    tg01wd(n, m, p, a_data, lda, e_data, lde, b_data, ldb, c_data, ldc,
           q_data, ldq, z_data, ldz, alphar_data, alphai_data, beta_data,
           dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("(OOOOOOOOOi)",
                                     a_array, e_array, b_array, c_array,
                                     q_array, z_array,
                                     alphar_array, alphai_array, beta_array, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);

    return result;
}

PyObject* py_tg01qd(PyObject* self, PyObject* args) {
    (void)self;

    const char *dico, *stdom, *jobfi;
    i32 n, m, p;
    f64 alpha, tol;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj;
    PyArrayObject *a_array, *e_array, *b_array, *c_array;
    i32 n1 = 0, n2 = 0, n3 = 0, nd = 0, niblck = 0, info = 0;
    i32 lda, lde, ldb, ldc, ldq, ldz;

    if (!PyArg_ParseTuple(args, "sssiiidOOOOd", &dico, &stdom, &jobfi,
                          &n, &m, &p, &alpha, &a_obj, &e_obj, &b_obj, &c_obj, &tol)) {
        return NULL;
    }

    if (n < 0 || m < 0 || p < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative");
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || b_array == NULL || c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *e_dims = PyArray_DIMS(e_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    lda = (n > 0) ? (i32)a_dims[0] : 1;
    lde = (n > 0) ? (i32)e_dims[0] : 1;
    ldb = (n > 0) ? (i32)b_dims[0] : 1;
    ldc = (p > 0) ? (i32)c_dims[0] : 1;
    ldq = (n > 0) ? n : 1;
    ldz = (n > 0) ? n : 1;

    bool order = (stdom[0] == 'S' || stdom[0] == 's' ||
                  stdom[0] == 'U' || stdom[0] == 'u');
    i32 ldwork;
    if (n == 0) {
        ldwork = 1;
    } else if (order) {
        ldwork = 4 * n + 16;
    } else {
        ldwork = 4 * n;
    }
    if (ldwork < 1) ldwork = 1;

    i32 *iwork = (n > 0) ? (i32*)calloc(n, sizeof(i32)) : NULL;
    i32 *iblck = (n > 0) ? (i32*)calloc(n, sizeof(i32)) : NULL;
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));

    npy_intp q_dims[2] = {n, n};
    npy_intp z_dims[2] = {n, n};
    npy_intp q_strides[2] = {sizeof(f64), n * sizeof(f64)};
    npy_intp z_strides[2] = {sizeof(f64), n * sizeof(f64)};
    npy_intp eig_dims[1] = {n};

    PyObject *q_array = NULL;
    PyObject *z_array = NULL;
    PyObject *alphar_array = NULL;
    PyObject *alphai_array = NULL;
    PyObject *beta_array = NULL;

    if (n > 0) {
        q_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        z_array = PyArray_New(&PyArray_Type, 2, z_dims, NPY_DOUBLE, z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        alphar_array = PyArray_EMPTY(1, eig_dims, NPY_DOUBLE, 0);
        alphai_array = PyArray_EMPTY(1, eig_dims, NPY_DOUBLE, 0);
        beta_array = PyArray_EMPTY(1, eig_dims, NPY_DOUBLE, 0);
    } else {
        npy_intp zero_dims[2] = {0, 0};
        npy_intp zero_vec[1] = {0};
        q_array = PyArray_EMPTY(2, zero_dims, NPY_DOUBLE, 1);
        z_array = PyArray_EMPTY(2, zero_dims, NPY_DOUBLE, 1);
        alphar_array = PyArray_EMPTY(1, zero_vec, NPY_DOUBLE, 0);
        alphai_array = PyArray_EMPTY(1, zero_vec, NPY_DOUBLE, 0);
        beta_array = PyArray_EMPTY(1, zero_vec, NPY_DOUBLE, 0);
    }

    if (q_array == NULL || z_array == NULL || alphar_array == NULL ||
        alphai_array == NULL || beta_array == NULL ||
        dwork == NULL || (n > 0 && (iwork == NULL || iblck == NULL))) {
        free(iwork);
        free(iblck);
        free(dwork);
        Py_XDECREF(q_array);
        Py_XDECREF(z_array);
        Py_XDECREF(alphar_array);
        Py_XDECREF(alphai_array);
        Py_XDECREF(beta_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *q = (n > 0) ? (f64*)PyArray_DATA((PyArrayObject*)q_array) : NULL;
    f64 *z = (n > 0) ? (f64*)PyArray_DATA((PyArrayObject*)z_array) : NULL;
    f64 *alphar = (n > 0) ? (f64*)PyArray_DATA((PyArrayObject*)alphar_array) : NULL;
    f64 *alphai = (n > 0) ? (f64*)PyArray_DATA((PyArrayObject*)alphai_array) : NULL;
    f64 *beta = (n > 0) ? (f64*)PyArray_DATA((PyArrayObject*)beta_array) : NULL;

    if (n > 0) {
        memset(q, 0, n * n * sizeof(f64));
        memset(z, 0, n * n * sizeof(f64));
    }

    tg01qd(dico, stdom, jobfi, n, m, p, alpha, a_data, lda, e_data, lde,
           b_data, ldb, c_data, ldc, &n1, &n2, &n3, &nd, &niblck, iblck,
           q, ldq, z, ldz, alphar, alphai, beta, tol, iwork, dwork, ldwork, &info);

    npy_intp iblck_dims[1] = {(niblck > 0) ? niblck : 1};
    PyObject *iblck_array = PyArray_SimpleNew(1, iblck_dims, NPY_INT32);
    if (niblck > 0 && iblck != NULL) {
        memcpy(PyArray_DATA((PyArrayObject*)iblck_array), iblck, niblck * sizeof(i32));
    }

    free(iwork);
    free(iblck);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("(OOOOiiiiiOOOOOOi)",
                                     a_array, e_array, b_array, c_array,
                                     n1, n2, n3, nd, niblck, iblck_array,
                                     q_array, z_array,
                                     alphar_array, alphai_array, beta_array, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);
    Py_DECREF(iblck_array);

    return result;
}

PyObject* py_tg01nd(PyObject* self, PyObject* args) {
    (void)self;

    const char *job, *jobt;
    i32 n, m, p;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj;
    f64 tol;
    PyArrayObject *a_array, *e_array, *b_array, *c_array;
    i32 nf = 0, nd = 0, niblck = 0, info = 0;
    i32 lda, lde, ldb, ldc, ldq, ldz;

    if (!PyArg_ParseTuple(args, "ssiiiOOOOd", &job, &jobt,
                          &n, &m, &p, &a_obj, &e_obj, &b_obj, &c_obj, &tol)) {
        return NULL;
    }

    if (n < 0 || m < 0 || p < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative");
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || b_array == NULL || c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *e_dims = PyArray_DIMS(e_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    lda = (n > 0) ? (i32)a_dims[0] : 1;
    lde = (n > 0) ? (i32)e_dims[0] : 1;
    ldb = (n > 0) ? (i32)b_dims[0] : 1;
    ldc = (p > 0) ? (i32)c_dims[0] : 1;
    ldq = (n > 0) ? n : 1;
    ldz = (n > 0) ? n : 1;

    i32 ldwork = (n > 0) ? (4 * n) : 1;
    ldwork = (ldwork > 1) ? ldwork : 1;

    i32 *iwork = (n > 0) ? (i32*)malloc((n + 6) * sizeof(i32)) : NULL;
    i32 *iblck = (n > 0) ? (i32*)malloc(n * sizeof(i32)) : NULL;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    npy_intp q_dims[2] = {n, n};
    npy_intp z_dims[2] = {n, n};
    npy_intp q_strides[2] = {sizeof(f64), n * sizeof(f64)};
    npy_intp z_strides[2] = {sizeof(f64), n * sizeof(f64)};
    npy_intp eig_dims[1] = {n};

    PyObject *q_array = NULL;
    PyObject *z_array = NULL;
    PyObject *alphar_array = NULL;
    PyObject *alphai_array = NULL;
    PyObject *beta_array = NULL;

    if (n > 0) {
        q_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        z_array = PyArray_New(&PyArray_Type, 2, z_dims, NPY_DOUBLE, z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        alphar_array = PyArray_EMPTY(1, eig_dims, NPY_DOUBLE, 0);
        alphai_array = PyArray_EMPTY(1, eig_dims, NPY_DOUBLE, 0);
        beta_array = PyArray_EMPTY(1, eig_dims, NPY_DOUBLE, 0);
    } else {
        npy_intp zero_dims[2] = {0, 0};
        npy_intp zero_vec[1] = {0};
        q_array = PyArray_EMPTY(2, zero_dims, NPY_DOUBLE, 1);
        z_array = PyArray_EMPTY(2, zero_dims, NPY_DOUBLE, 1);
        alphar_array = PyArray_EMPTY(1, zero_vec, NPY_DOUBLE, 0);
        alphai_array = PyArray_EMPTY(1, zero_vec, NPY_DOUBLE, 0);
        beta_array = PyArray_EMPTY(1, zero_vec, NPY_DOUBLE, 0);
    }

    if (q_array == NULL || z_array == NULL || alphar_array == NULL ||
        alphai_array == NULL || beta_array == NULL ||
        dwork == NULL || (n > 0 && (iwork == NULL || iblck == NULL))) {
        free(iwork);
        free(iblck);
        free(dwork);
        Py_XDECREF(q_array);
        Py_XDECREF(z_array);
        Py_XDECREF(alphar_array);
        Py_XDECREF(alphai_array);
        Py_XDECREF(beta_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *q = (n > 0) ? (f64*)PyArray_DATA((PyArrayObject*)q_array) : NULL;
    f64 *z = (n > 0) ? (f64*)PyArray_DATA((PyArrayObject*)z_array) : NULL;
    f64 *alphar = (n > 0) ? (f64*)PyArray_DATA((PyArrayObject*)alphar_array) : NULL;
    f64 *alphai = (n > 0) ? (f64*)PyArray_DATA((PyArrayObject*)alphai_array) : NULL;
    f64 *beta = (n > 0) ? (f64*)PyArray_DATA((PyArrayObject*)beta_array) : NULL;

    if (n > 0) {
        memset(q, 0, n * n * sizeof(f64));
        memset(z, 0, n * n * sizeof(f64));
    }

    tg01nd(job, jobt, n, m, p, a_data, lda, e_data, lde, b_data, ldb, c_data, ldc,
           alphar, alphai, beta, q, ldq, z, ldz,
           &nf, &nd, &niblck, iblck, tol, iwork, dwork, ldwork, &info);

    npy_intp iblck_dims[1] = {(niblck > 0) ? niblck : 1};
    PyObject *iblck_array = PyArray_SimpleNew(1, iblck_dims, NPY_INT32);
    if (niblck > 0 && iblck != NULL) {
        memcpy(PyArray_DATA((PyArrayObject*)iblck_array), iblck, niblck * sizeof(i32));
    }

    free(iwork);
    free(iblck);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("(OOOOOOOOOiiiOi)",
                                     a_array, e_array, b_array, c_array,
                                     alphar_array, alphai_array, beta_array,
                                     q_array, z_array,
                                     nf, nd, niblck, iblck_array, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);
    Py_DECREF(iblck_array);

    return result;
}

