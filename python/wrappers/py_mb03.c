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
#include <float.h>



/* Python wrapper for mb03oy */
PyObject* py_mb03oy(PyObject* self, PyObject* args) {
    i32 m, n, lda;
    f64 rcond, svlmax;
    PyObject *a_obj;
    PyArrayObject *a_array;
    i32 rank = 0, info = 0;

    if (!PyArg_ParseTuple(args, "iiOdd", &m, &n, &a_obj, &rcond, &svlmax)) {
        return NULL;
    }

    /* Validate dimensions before allocation */
    if (m < 0 || n < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative (m=%d, n=%d)", m, n);
        return NULL;
    }

    /* Convert to NumPy array - preserve Fortran-order (column-major) */
    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    /* Extract leading dimension (ensure lda >= 1 even if m=0) */
    npy_intp *a_dims = PyArray_DIMS(a_array);
    lda = (i32)a_dims[0];
    if (lda < 1) lda = 1;

    /* Allocate output arrays (handle n=0 edge case) */
    i32 mn = (m < n) ? m : n;
    i32 dwork_size = (n > 0) ? (3*n - 1) : 1;
    f64 *dwork = (f64*)malloc(dwork_size * sizeof(f64));

    if (dwork == NULL) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    /* Create output NumPy arrays - let NumPy allocate memory */
    npy_intp sval_dims[1] = {3};
    npy_intp jpvt_dims[1] = {n > 0 ? n : 0};
    npy_intp tau_dims[1] = {mn};

    PyObject *sval_array = PyArray_SimpleNew(1, sval_dims, NPY_DOUBLE);
    PyObject *jpvt_array = (n > 0) ? PyArray_SimpleNew(1, jpvt_dims, NPY_INT32) : PyArray_EMPTY(1, jpvt_dims, NPY_INT32, 0);
    PyObject *tau_array = (mn > 0) ? PyArray_SimpleNew(1, tau_dims, NPY_DOUBLE) : PyArray_EMPTY(1, tau_dims, NPY_DOUBLE, 0);

    if (sval_array == NULL || jpvt_array == NULL || tau_array == NULL) {
        free(dwork);
        Py_XDECREF(sval_array);
        Py_XDECREF(jpvt_array);
        Py_XDECREF(tau_array);
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output arrays");
        return NULL;
    }

    f64 *sval = (f64*)PyArray_DATA((PyArrayObject*)sval_array);
    i32 *jpvt = (n > 0) ? (i32*)PyArray_DATA((PyArrayObject*)jpvt_array) : NULL;
    f64 *tau = (mn > 0) ? (f64*)PyArray_DATA((PyArrayObject*)tau_array) : NULL;

    /* Call C function */
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    mb03oy(m, n, a_data, lda, rcond, svlmax, &rank, sval, jpvt, tau, dwork, &info);

    free(dwork);

    /* Resolve writebackifcopy before decref */
    PyArray_ResolveWritebackIfCopy(a_array);

    /* Build result tuple */
    PyObject *result = Py_BuildValue("(OiiOOO)", a_array, rank, info,
                                     sval_array, jpvt_array, tau_array);

    Py_DECREF(a_array);
    Py_DECREF(sval_array);
    Py_DECREF(jpvt_array);
    Py_DECREF(tau_array);

    return result;
}



/* Python wrapper for mb03od */
PyObject* py_mb03od(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *jobqr = "Q";
    i32 m, n, lda;
    f64 rcond, svlmax;
    PyObject *a_obj;
    PyArrayObject *a_array;
    i32 rank = 0, info = 0;

    static char *kwlist[] = {"m", "n", "a", "rcond", "svlmax", "jobqr", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiOdd|s", kwlist,
                                     &m, &n, &a_obj, &rcond, &svlmax, &jobqr)) {
        return NULL;
    }

    if (m < 0 || n < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative (m=%d, n=%d)", m, n);
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    lda = (i32)a_dims[0];
    if (lda < 1) lda = 1;

    i32 mn = (m < n) ? m : n;
    f64 *tau = (mn > 0) ? (f64*)malloc(mn * sizeof(f64)) : NULL;

    bool ljobqr = (*jobqr == 'Q' || *jobqr == 'q');
    i32 ldwork = ljobqr ? (3*n + 1) : ((2*mn > 1) ? 2*mn : 1);
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (dwork == NULL || (mn > 0 && tau == NULL)) {
        free(tau); free(dwork);
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    npy_intp sval_dims[1] = {3};
    npy_intp jpvt_dims[1] = {n > 0 ? n : 0};

    PyObject *sval_array = PyArray_SimpleNew(1, sval_dims, NPY_DOUBLE);
    PyObject *jpvt_array = (n > 0) ? PyArray_SimpleNew(1, jpvt_dims, NPY_INT32) : PyArray_EMPTY(1, jpvt_dims, NPY_INT32, 0);

    if (sval_array == NULL || jpvt_array == NULL) {
        free(tau); free(dwork);
        Py_XDECREF(sval_array);
        Py_XDECREF(jpvt_array);
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output arrays");
        return NULL;
    }

    f64 *sval = (f64*)PyArray_DATA((PyArrayObject*)sval_array);
    i32 *jpvt = (n > 0) ? (i32*)PyArray_DATA((PyArrayObject*)jpvt_array) : NULL;
    if (n > 0) memset(jpvt, 0, n * sizeof(i32));

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    mb03od(jobqr, m, n, a_data, lda, jpvt, rcond, svlmax, tau, &rank, sval, dwork, ldwork, &info);

    free(tau);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);

    PyObject *result = Py_BuildValue("OiOi", jpvt_array, rank, sval_array, info);

    Py_DECREF(a_array);
    Py_DECREF(sval_array);
    Py_DECREF(jpvt_array);

    return result;
}



/* Python wrapper for mb03ba */
PyObject* py_mb03ba(PyObject* self, PyObject* args) {
    i32 k, h;
    PyObject *s_obj;

    if (!PyArg_ParseTuple(args, "iiO", &k, &h, &s_obj)) {
        return NULL;
    }

    if (k < 1) {
        PyErr_SetString(PyExc_ValueError, "K must be >= 1");
        return NULL;
    }

    PyArrayObject *s_array = (PyArrayObject*)PyArray_FROM_OTF(s_obj, NPY_INT32,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (s_array == NULL) return NULL;

    const i32 *s_data = (const i32*)PyArray_DATA(s_array);

    npy_intp dims[1] = {k};
    PyArrayObject *amap_array = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_INT32, 0);
    PyArrayObject *qmap_array = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_INT32, 0);
    if (amap_array == NULL || qmap_array == NULL) {
        Py_XDECREF(amap_array);
        Py_XDECREF(qmap_array);
        Py_DECREF(s_array);
        return NULL;
    }

    i32 *amap_data = (i32*)PyArray_DATA(amap_array);
    i32 *qmap_data = (i32*)PyArray_DATA(qmap_array);
    i32 smult = 0;

    mb03ba(k, h, s_data, &smult, amap_data, qmap_data);

    Py_DECREF(s_array);

    PyObject *result = Py_BuildValue("(iOO)", smult, amap_array, qmap_array);
    Py_DECREF(amap_array);
    Py_DECREF(qmap_array);
    return result;
}


/* Python wrapper for mb03ab */
PyObject* py_mb03ab(PyObject* self, PyObject* args) {
    const char *shft;
    i32 k, n;
    PyObject *amap_obj, *s_obj, *a_obj;
    i32 sinv;
    f64 w1, w2;

    if (!PyArg_ParseTuple(args, "siiOOiOdd", &shft, &k, &n, &amap_obj, &s_obj,
                          &sinv, &a_obj, &w1, &w2)) {
        return NULL;
    }

    if (k < 1) {
        PyErr_SetString(PyExc_ValueError, "K must be >= 1");
        return NULL;
    }
    if (n < 2) {
        PyErr_SetString(PyExc_ValueError, "N must be >= 2");
        return NULL;
    }

    PyArrayObject *amap_array = (PyArrayObject*)PyArray_FROM_OTF(
        amap_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (amap_array == NULL) return NULL;

    PyArrayObject *s_array = (PyArrayObject*)PyArray_FROM_OTF(
        s_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (s_array == NULL) {
        Py_DECREF(amap_array);
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (a_array == NULL) {
        Py_DECREF(amap_array);
        Py_DECREF(s_array);
        return NULL;
    }

    const i32 *amap_data = (const i32*)PyArray_DATA(amap_array);
    const i32 *s_data = (const i32*)PyArray_DATA(s_array);
    const f64 *a_data = (const f64*)PyArray_DATA(a_array);

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda1 = (i32)a_dims[0];
    i32 lda2 = (i32)a_dims[1];

    f64 c1 = 0.0, s1 = 0.0, c2 = 0.0, s2 = 0.0;

    mb03ab(shft, k, n, amap_data, s_data, sinv, a_data, lda1, lda2, w1, w2,
           &c1, &s1, &c2, &s2);

    Py_DECREF(amap_array);
    Py_DECREF(s_array);
    Py_DECREF(a_array);

    return Py_BuildValue("(dddd)", c1, s1, c2, s2);
}


/* Python wrapper for mb03ad */
PyObject* py_mb03ad(PyObject* self, PyObject* args) {
    const char *shft;
    i32 k, n;
    PyObject *amap_obj, *s_obj, *a_obj;
    i32 sinv;

    if (!PyArg_ParseTuple(args, "siiOOiO", &shft, &k, &n, &amap_obj, &s_obj,
                          &sinv, &a_obj)) {
        return NULL;
    }

    if (k < 1) {
        PyErr_SetString(PyExc_ValueError, "K must be >= 1");
        return NULL;
    }
    if (n < 2) {
        PyErr_SetString(PyExc_ValueError, "N must be >= 2");
        return NULL;
    }

    PyArrayObject *amap_array = (PyArrayObject*)PyArray_FROM_OTF(
        amap_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (amap_array == NULL) return NULL;

    PyArrayObject *s_array = (PyArrayObject*)PyArray_FROM_OTF(
        s_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (s_array == NULL) {
        Py_DECREF(amap_array);
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (a_array == NULL) {
        Py_DECREF(amap_array);
        Py_DECREF(s_array);
        return NULL;
    }

    const i32 *amap_data = (const i32*)PyArray_DATA(amap_array);
    const i32 *s_data = (const i32*)PyArray_DATA(s_array);
    const f64 *a_data = (const f64*)PyArray_DATA(a_array);

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda1 = (i32)a_dims[0];
    i32 lda2 = (i32)a_dims[1];

    f64 c1 = 0.0, s1 = 0.0, c2 = 0.0, s2 = 0.0;

    mb03ad(shft, k, n, amap_data, s_data, sinv, a_data, lda1, lda2,
           &c1, &s1, &c2, &s2);

    Py_DECREF(amap_array);
    Py_DECREF(s_array);
    Py_DECREF(a_array);

    return Py_BuildValue("(dddd)", c1, s1, c2, s2);
}


/* Python wrapper for mb03ae */
PyObject* py_mb03ae(PyObject* self, PyObject* args) {
    const char *shft;
    i32 k, n;
    PyObject *amap_obj, *s_obj, *a_obj;
    i32 sinv;

    if (!PyArg_ParseTuple(args, "siiOOiO", &shft, &k, &n, &amap_obj, &s_obj,
                          &sinv, &a_obj)) {
        return NULL;
    }

    if (k < 1) {
        PyErr_SetString(PyExc_ValueError, "K must be >= 1");
        return NULL;
    }
    if (n < 2) {
        PyErr_SetString(PyExc_ValueError, "N must be >= 2");
        return NULL;
    }

    PyArrayObject *amap_array = (PyArrayObject*)PyArray_FROM_OTF(
        amap_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (amap_array == NULL) return NULL;

    PyArrayObject *s_array = (PyArrayObject*)PyArray_FROM_OTF(
        s_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (s_array == NULL) {
        Py_DECREF(amap_array);
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (a_array == NULL) {
        Py_DECREF(amap_array);
        Py_DECREF(s_array);
        return NULL;
    }

    const i32 *amap_data = (const i32*)PyArray_DATA(amap_array);
    const i32 *s_data = (const i32*)PyArray_DATA(s_array);
    const f64 *a_data = (const f64*)PyArray_DATA(a_array);

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda1 = (i32)a_dims[0];
    i32 lda2 = (i32)a_dims[1];

    f64 c1 = 0.0, s1 = 0.0, c2 = 0.0, s2 = 0.0;

    mb03ae(shft, k, n, amap_data, s_data, sinv, a_data, lda1, lda2,
           &c1, &s1, &c2, &s2);

    Py_DECREF(amap_array);
    Py_DECREF(s_array);
    Py_DECREF(a_array);

    return Py_BuildValue("(dddd)", c1, s1, c2, s2);
}


/* Python wrapper for mb03vd */
PyObject* py_mb03vd(PyObject* self, PyObject* args) {
    i32 n, p, ilo, ihi;
    PyObject *a_obj;
    PyArrayObject *a_array;
    i32 info = 0;

    if (!PyArg_ParseTuple(args, "iiiiO", &n, &p, &ilo, &ihi, &a_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda1 = (i32)a_dims[0];
    i32 lda2 = (i32)a_dims[1];
    if (lda1 < 1) lda1 = 1;
    if (lda2 < 1) lda2 = 1;

    i32 ldtau = (n > 1) ? (n - 1) : 1;
    i32 ldwork = (n > 0) ? n : 1;

    npy_intp tau_dims[2] = {ldtau, p > 0 ? p : 1};
    npy_intp tau_strides[2] = {sizeof(f64), ldtau * sizeof(f64)};

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (dwork == NULL) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    PyObject *tau_array = PyArray_New(&PyArray_Type, 2, tau_dims, NPY_DOUBLE,
                                      tau_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (tau_array == NULL) {
        free(dwork);
        Py_DECREF(a_array);
        return NULL;
    }
    f64 *tau = (f64*)PyArray_DATA((PyArrayObject*)tau_array);
    memset(tau, 0, ldtau * (p > 0 ? p : 1) * sizeof(f64));

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    mb03vd(n, p, ilo, ihi, a_data, lda1, lda2, tau, ldtau, dwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);

    PyObject *result = Py_BuildValue("OOi", a_array, tau_array, info);
    Py_DECREF(a_array);
    Py_DECREF(tau_array);
    return result;
}



/* Python wrapper for mb03vy */
PyObject* py_mb03vy(PyObject* self, PyObject* args) {
    i32 n, p, ilo, ihi;
    PyObject *a_obj, *tau_obj;
    PyArrayObject *a_array, *tau_array;
    i32 info = 0;

    if (!PyArg_ParseTuple(args, "iiiiOO", &n, &p, &ilo, &ihi, &a_obj, &tau_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    tau_array = (PyArrayObject*)PyArray_FROM_OTF(tau_obj, NPY_DOUBLE,
                                                 NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (tau_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda1 = (i32)a_dims[0];
    i32 lda2 = (i32)a_dims[1];
    if (lda1 < 1) lda1 = 1;
    if (lda2 < 1) lda2 = 1;

    i32 ldtau = (n > 1) ? (n - 1) : 1;
    i32 ldwork = (n > 0) ? n : 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(tau_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work array");
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *tau_data = (f64*)PyArray_DATA(tau_array);

    mb03vy(n, p, ilo, ihi, a_data, lda1, lda2, tau_data, ldtau, dwork, ldwork, &info);

    free(dwork);

    npy_intp q_dims[3] = {n > 0 ? n : 1, n > 0 ? n : 1, p > 0 ? p : 1};
    npy_intp q_strides[3] = {sizeof(f64), (n > 0 ? n : 1) * sizeof(f64),
                             (n > 0 ? n : 1) * (n > 0 ? n : 1) * sizeof(f64)};
    i32 q_size = (n > 0 ? n : 1) * (n > 0 ? n : 1) * (p > 0 ? p : 1);

    PyObject *q_array = PyArray_New(&PyArray_Type, 3, q_dims, NPY_DOUBLE,
                                    q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (q_array == NULL) {
        PyArray_ResolveWritebackIfCopy(a_array);
        Py_DECREF(a_array);
        Py_DECREF(tau_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output array");
        return NULL;
    }
    f64 *q_data = (f64*)PyArray_DATA((PyArrayObject*)q_array);
    memcpy(q_data, a_data, q_size * sizeof(f64));

    PyArray_ResolveWritebackIfCopy(a_array);

    Py_DECREF(a_array);
    Py_DECREF(tau_array);

    PyObject *result = Py_BuildValue("Oi", q_array, info);
    Py_DECREF(q_array);
    return result;
}



/* Python wrapper for mb03ya */
PyObject* py_mb03ya(PyObject* self, PyObject* args) {
    int wantt_int, wantq_int, wantz_int;
    i32 ilo, ihi, iloq, ihiq, pos;
    PyObject *a_obj, *b_obj, *q_obj, *z_obj;
    PyArrayObject *a_array = NULL, *b_array = NULL, *q_array = NULL, *z_array = NULL;
    i32 info = 0;

    if (!PyArg_ParseTuple(args, "iiiiiiiiOOOO",
                          &wantt_int, &wantq_int, &wantz_int,
                          &ilo, &ihi, &iloq, &ihiq, &pos,
                          &a_obj, &b_obj, &q_obj, &z_obj)) {
        return NULL;
    }

    bool wantt = wantt_int != 0;
    bool wantq = wantq_int != 0;
    bool wantz = wantz_int != 0;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (q_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    z_array = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (z_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(q_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 n = (i32)a_dims[0];
    i32 lda = n > 1 ? n : 1;
    i32 ldb = lda;
    i32 ldq = lda;
    i32 ldz = lda;

    f64 *a = (f64*)PyArray_DATA(a_array);
    f64 *b = (f64*)PyArray_DATA(b_array);
    f64 *q = (f64*)PyArray_DATA(q_array);
    f64 *z = (f64*)PyArray_DATA(z_array);

    mb03ya(wantt, wantq, wantz, n, ilo, ihi, iloq, ihiq, pos,
           a, lda, b, ldb, q, ldq, z, ldz, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(q_array);
    PyArray_ResolveWritebackIfCopy(z_array);

    PyObject *result = Py_BuildValue("OOOOi", a_array, b_array, q_array, z_array, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);
    return result;
}



/* Python wrapper for mb03yt */
PyObject* py_mb03yt(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj;
    PyArrayObject *a_array = NULL, *b_array = NULL;

    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda = (i32)a_dims[0];
    i32 ldb = lda;
    if (lda < 2) lda = 2;
    if (ldb < 2) ldb = 2;

    f64 alphar[2], alphai[2], beta[2];
    f64 csl, snl, csr, snr;

    f64 *a = (f64*)PyArray_DATA(a_array);
    f64 *b = (f64*)PyArray_DATA(b_array);

    mb03yt(a, lda, b, ldb, alphar, alphai, beta, &csl, &snl, &csr, &snr);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    npy_intp dims[1] = {2};
    PyObject *alphar_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyObject *alphai_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyObject *beta_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    memcpy(PyArray_DATA((PyArrayObject*)alphar_array), alphar, 2 * sizeof(f64));
    memcpy(PyArray_DATA((PyArrayObject*)alphai_array), alphai, 2 * sizeof(f64));
    memcpy(PyArray_DATA((PyArrayObject*)beta_array), beta, 2 * sizeof(f64));

    PyObject *result = Py_BuildValue("OOOOOdddd",
                                     a_array, b_array,
                                     alphar_array, alphai_array, beta_array,
                                     csl, snl, csr, snr);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);
    return result;
}



/* Python wrapper for mb03yd */
PyObject* py_mb03yd(PyObject* self, PyObject* args, PyObject* kwargs) {
    int wantt_int, wantq_int, wantz_int;
    i32 n, ilo, ihi, iloq, ihiq;
    PyObject *a_obj, *b_obj, *q_obj, *z_obj;
    PyArrayObject *a_array = NULL, *b_array = NULL, *q_array = NULL, *z_array = NULL;
    i32 info = 0;

    static char *kwlist[] = {"wantt", "wantq", "wantz", "n", "ilo", "ihi",
                             "iloq", "ihiq", "a", "b", "q", "z", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "pppiiiiiOOOO", kwlist,
                                     &wantt_int, &wantq_int, &wantz_int,
                                     &n, &ilo, &ihi, &iloq, &ihiq,
                                     &a_obj, &b_obj, &q_obj, &z_obj)) {
        return NULL;
    }

    bool wantt = wantt_int != 0;
    bool wantq = wantq_int != 0;
    bool wantz = wantz_int != 0;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (q_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    z_array = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (z_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(q_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda = (i32)a_dims[0];
    i32 ldb = lda;
    i32 ldq = lda;
    i32 ldz = lda;

    if (lda < 1) lda = 1;
    if (ldb < 1) ldb = 1;
    if (ldq < 1) ldq = 1;
    if (ldz < 1) ldz = 1;

    i32 ldwork = n > 1 ? n : 1;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        PyErr_NoMemory();
        return NULL;
    }

    npy_intp dims[1] = {n > 0 ? n : 1};
    PyObject *alphar_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyObject *alphai_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyObject *beta_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    if (alphar_array == NULL || alphai_array == NULL || beta_array == NULL) {
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        Py_XDECREF(alphar_array);
        Py_XDECREF(alphai_array);
        Py_XDECREF(beta_array);
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_array);
    f64 *b = (f64*)PyArray_DATA(b_array);
    f64 *q = (f64*)PyArray_DATA(q_array);
    f64 *z = (f64*)PyArray_DATA(z_array);
    f64 *alphar = (f64*)PyArray_DATA((PyArrayObject*)alphar_array);
    f64 *alphai = (f64*)PyArray_DATA((PyArrayObject*)alphai_array);
    f64 *beta = (f64*)PyArray_DATA((PyArrayObject*)beta_array);

    mb03yd(wantt, wantq, wantz, n, ilo, ihi, iloq, ihiq,
           a, lda, b, ldb, q, ldq, z, ldz,
           alphar, alphai, beta, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(q_array);
    PyArray_ResolveWritebackIfCopy(z_array);

    PyObject *result = Py_BuildValue("OOOi", alphar_array, alphai_array, beta_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);
    return result;
}



/* Python wrapper for mb03xu */
PyObject* py_mb03xu(PyObject* self, PyObject* args) {
    int ltra_int, ltrb_int;
    i32 n, k, nb;
    PyObject *a_obj, *b_obj, *g_obj, *q_obj;
    PyArrayObject *a_array = NULL, *b_array = NULL, *g_array = NULL, *q_array = NULL;

    if (!PyArg_ParseTuple(args, "ppiiiOOOO",
                          &ltra_int, &ltrb_int, &n, &k, &nb,
                          &a_obj, &b_obj, &g_obj, &q_obj)) {
        return NULL;
    }

    bool ltra = (ltra_int != 0);
    bool ltrb = (ltrb_int != 0);

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    g_array = (PyArrayObject*)PyArray_FROM_OTF(g_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || b_array == NULL || g_array == NULL || q_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(g_array);
        Py_XDECREF(q_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *g_dims = PyArray_DIMS(g_array);
    npy_intp *q_dims = PyArray_DIMS(q_array);

    i32 lda = (i32)a_dims[0];
    i32 ldb = (i32)b_dims[0];
    i32 ldg = (i32)g_dims[0];
    i32 ldq = (i32)q_dims[0];

    i32 kn = k + n;
    i32 nb2 = 2 * nb;
    i32 ldxa = (n > 1) ? n : 1;
    i32 ldxb = (kn > 1) ? kn : 1;
    i32 ldxg = (kn > 1) ? kn : 1;
    i32 ldxq = (n > 1) ? n : 1;
    i32 ldya = (kn > 1) ? kn : 1;
    i32 ldyb = (n > 1) ? n : 1;
    i32 ldyg = (kn > 1) ? kn : 1;
    i32 ldyq = (n > 1) ? n : 1;
    i32 ldwork = (nb > 0) ? 5 * nb : 1;

    npy_intp xa_dims[2] = {ldxa, nb2 > 0 ? nb2 : 1};
    npy_intp xb_dims[2] = {ldxb, nb2 > 0 ? nb2 : 1};
    npy_intp xg_dims[2] = {ldxg, nb2 > 0 ? nb2 : 1};
    npy_intp xq_dims[2] = {ldxq, nb2 > 0 ? nb2 : 1};
    npy_intp ya_dims[2] = {ldya, nb2 > 0 ? nb2 : 1};
    npy_intp yb_dims[2] = {ldyb, nb2 > 0 ? nb2 : 1};
    npy_intp yg_dims[2] = {ldyg, nb2 > 0 ? nb2 : 1};
    npy_intp yq_dims[2] = {ldyq, nb2 > 0 ? nb2 : 1};
    npy_intp csl_dims[1] = {nb2 > 0 ? nb2 : 1};
    npy_intp csr_dims[1] = {nb2 > 0 ? nb2 : 1};
    npy_intp taul_dims[1] = {nb > 0 ? nb : 1};
    npy_intp taur_dims[1] = {nb > 0 ? nb : 1};

    npy_intp xa_strides[2] = {sizeof(f64), ldxa * sizeof(f64)};
    npy_intp xb_strides[2] = {sizeof(f64), ldxb * sizeof(f64)};
    npy_intp xg_strides[2] = {sizeof(f64), ldxg * sizeof(f64)};
    npy_intp xq_strides[2] = {sizeof(f64), ldxq * sizeof(f64)};
    npy_intp ya_strides[2] = {sizeof(f64), ldya * sizeof(f64)};
    npy_intp yb_strides[2] = {sizeof(f64), ldyb * sizeof(f64)};
    npy_intp yg_strides[2] = {sizeof(f64), ldyg * sizeof(f64)};
    npy_intp yq_strides[2] = {sizeof(f64), ldyq * sizeof(f64)};

    PyObject *xa_array = PyArray_New(&PyArray_Type, 2, xa_dims, NPY_DOUBLE, xa_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *xb_array = PyArray_New(&PyArray_Type, 2, xb_dims, NPY_DOUBLE, xb_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *xg_array = PyArray_New(&PyArray_Type, 2, xg_dims, NPY_DOUBLE, xg_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *xq_array = PyArray_New(&PyArray_Type, 2, xq_dims, NPY_DOUBLE, xq_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *ya_array = PyArray_New(&PyArray_Type, 2, ya_dims, NPY_DOUBLE, ya_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *yb_array = PyArray_New(&PyArray_Type, 2, yb_dims, NPY_DOUBLE, yb_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *yg_array = PyArray_New(&PyArray_Type, 2, yg_dims, NPY_DOUBLE, yg_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *yq_array = PyArray_New(&PyArray_Type, 2, yq_dims, NPY_DOUBLE, yq_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *csl_array = PyArray_New(&PyArray_Type, 1, csl_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
    PyObject *csr_array = PyArray_New(&PyArray_Type, 1, csr_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
    PyObject *taul_array = PyArray_New(&PyArray_Type, 1, taul_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
    PyObject *taur_array = PyArray_New(&PyArray_Type, 1, taur_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);

    if (xa_array == NULL || xb_array == NULL || xg_array == NULL || xq_array == NULL ||
        ya_array == NULL || yb_array == NULL || yg_array == NULL || yq_array == NULL ||
        csl_array == NULL || csr_array == NULL || taul_array == NULL || taur_array == NULL) {
        Py_XDECREF(xa_array); Py_XDECREF(xb_array); Py_XDECREF(xg_array); Py_XDECREF(xq_array);
        Py_XDECREF(ya_array); Py_XDECREF(yb_array); Py_XDECREF(yg_array); Py_XDECREF(yq_array);
        Py_XDECREF(csl_array); Py_XDECREF(csr_array); Py_XDECREF(taul_array); Py_XDECREF(taur_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(g_array);
        Py_DECREF(q_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    f64 *xa = (f64*)PyArray_DATA((PyArrayObject*)xa_array);
    f64 *xb = (f64*)PyArray_DATA((PyArrayObject*)xb_array);
    f64 *xg = (f64*)PyArray_DATA((PyArrayObject*)xg_array);
    f64 *xq = (f64*)PyArray_DATA((PyArrayObject*)xq_array);
    f64 *ya = (f64*)PyArray_DATA((PyArrayObject*)ya_array);
    f64 *yb = (f64*)PyArray_DATA((PyArrayObject*)yb_array);
    f64 *yg = (f64*)PyArray_DATA((PyArrayObject*)yg_array);
    f64 *yq = (f64*)PyArray_DATA((PyArrayObject*)yq_array);
    f64 *csl = (f64*)PyArray_DATA((PyArrayObject*)csl_array);
    f64 *csr = (f64*)PyArray_DATA((PyArrayObject*)csr_array);
    f64 *taul = (f64*)PyArray_DATA((PyArrayObject*)taul_array);
    f64 *taur = (f64*)PyArray_DATA((PyArrayObject*)taur_array);

    memset(xa, 0, ldxa * (nb2 > 0 ? nb2 : 1) * sizeof(f64));
    memset(xb, 0, ldxb * (nb2 > 0 ? nb2 : 1) * sizeof(f64));
    memset(xg, 0, ldxg * (nb2 > 0 ? nb2 : 1) * sizeof(f64));
    memset(xq, 0, ldxq * (nb2 > 0 ? nb2 : 1) * sizeof(f64));
    memset(ya, 0, ldya * (nb2 > 0 ? nb2 : 1) * sizeof(f64));
    memset(yb, 0, ldyb * (nb2 > 0 ? nb2 : 1) * sizeof(f64));
    memset(yg, 0, ldyg * (nb2 > 0 ? nb2 : 1) * sizeof(f64));
    memset(yq, 0, ldyq * (nb2 > 0 ? nb2 : 1) * sizeof(f64));
    memset(csl, 0, (nb2 > 0 ? nb2 : 1) * sizeof(f64));
    memset(csr, 0, (nb2 > 0 ? nb2 : 1) * sizeof(f64));
    memset(taul, 0, (nb > 0 ? nb : 1) * sizeof(f64));
    memset(taur, 0, (nb > 0 ? nb : 1) * sizeof(f64));

    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(xa_array); Py_DECREF(xb_array); Py_DECREF(xg_array); Py_DECREF(xq_array);
        Py_DECREF(ya_array); Py_DECREF(yb_array); Py_DECREF(yg_array); Py_DECREF(yq_array);
        Py_DECREF(csl_array); Py_DECREF(csr_array); Py_DECREF(taul_array); Py_DECREF(taur_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(g_array);
        Py_DECREF(q_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *g_data = (f64*)PyArray_DATA(g_array);
    f64 *q_data = (f64*)PyArray_DATA(q_array);

    mb03xu(ltra, ltrb, n, k, nb,
           a_data, lda, b_data, ldb, g_data, ldg, q_data, ldq,
           xa, ldxa, xb, ldxb, xg, ldxg, xq, ldxq,
           ya, ldya, yb, ldyb, yg, ldyg, yq, ldyq,
           csl, csr, taul, taur, dwork);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(g_array);
    PyArray_ResolveWritebackIfCopy(q_array);

    PyObject *result = Py_BuildValue("(OOOOOOOOOOOOOOOOi)",
        a_array, b_array, g_array, q_array,
        xa_array, xb_array, xg_array, xq_array,
        ya_array, yb_array, yg_array, yq_array,
        csl_array, csr_array, taul_array, taur_array, 0);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(g_array);
    Py_DECREF(q_array);
    Py_DECREF(xa_array);
    Py_DECREF(xb_array);
    Py_DECREF(xg_array);
    Py_DECREF(xq_array);
    Py_DECREF(ya_array);
    Py_DECREF(yb_array);
    Py_DECREF(yg_array);
    Py_DECREF(yq_array);
    Py_DECREF(csl_array);
    Py_DECREF(csr_array);
    Py_DECREF(taul_array);
    Py_DECREF(taur_array);

    return result;
}



/* Python wrapper for mb03xp */
PyObject* py_mb03xp(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *job_str, *compq_str, *compz_str;
    i32 n, ilo, ihi;
    PyObject *a_obj, *b_obj;
    PyArrayObject *a_array = NULL, *b_array = NULL;
    PyArrayObject *q_array = NULL, *z_array = NULL;
    i32 info = 0;

    static char *kwlist[] = {"job", "compq", "compz", "n", "ilo", "ihi",
                             "a", "b", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssiiiOO", kwlist,
                                     &job_str, &compq_str, &compz_str,
                                     &n, &ilo, &ihi, &a_obj, &b_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda = (i32)a_dims[0];
    if (lda < 1) lda = 1;
    i32 ldb = lda;
    i32 ldq = (n > 1) ? n : 1;
    i32 ldz = ldq;
    i32 n_alloc = (n > 0) ? n : 1;

    bool wantq = (compq_str[0] == 'I' || compq_str[0] == 'i' ||
                  compq_str[0] == 'V' || compq_str[0] == 'v');
    bool wantz = (compz_str[0] == 'I' || compz_str[0] == 'i' ||
                  compz_str[0] == 'V' || compz_str[0] == 'v');

    npy_intp q_dims[2] = {n_alloc, n_alloc};

    f64 *q_data = NULL;
    f64 *z_data = NULL;

    if (wantq) {
        if (n > 0) {
            q_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE,
                                                  NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            if (q_array == NULL) {
                Py_DECREF(a_array);
                Py_DECREF(b_array);
                return NULL;
            }
            q_data = (f64*)PyArray_DATA(q_array);
            memset(q_data, 0, PyArray_NBYTES(q_array));
        } else {
            // Allocate dummy for mb03xp call if n=0? But n_alloc=1.
            // If n=0, mb03xp usually doesn't touch q.
            // But we can allocate a small dummy buffer on stack or malloc if strict.
            // Or just pass NULL if n=0?
            // Existing code allocated 1 element.
            q_data = (f64*)calloc(1, sizeof(f64)); // Dummy
        }
    } else {
        q_data = (f64*)calloc(1, sizeof(f64)); // Dummy
    }

    if (wantz) {
        if (n > 0) {
            z_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE,
                                                  NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            if (z_array == NULL) {
                Py_XDECREF(q_array);
                if (wantq && n <= 0) free(q_data);
                if (!wantq) free(q_data);
                Py_DECREF(a_array);
                Py_DECREF(b_array);
                return NULL;
            }
            z_data = (f64*)PyArray_DATA(z_array);
            memset(z_data, 0, PyArray_NBYTES(z_array));
        } else {
            z_data = (f64*)calloc(1, sizeof(f64));
        }
    } else {
        z_data = (f64*)calloc(1, sizeof(f64));
    }

    i32 ldwork = n > 1 ? n : 1;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        if (n <= 0 || !wantq) free(q_data);
        if (n <= 0 || !wantz) free(z_data);
        Py_XDECREF(q_array);
        Py_XDECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        PyErr_NoMemory();
        return NULL;
    }

    npy_intp dims[1] = {n_alloc};
    PyObject *alphar_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyObject *alphai_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyObject *beta_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    if (alphar_array == NULL || alphai_array == NULL || beta_array == NULL) {
        free(dwork);
        if (n <= 0 || !wantq) free(q_data);
        if (n <= 0 || !wantz) free(z_data);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_XDECREF(q_array);
        Py_XDECREF(z_array);
        Py_XDECREF(alphar_array);
        Py_XDECREF(alphai_array);
        Py_XDECREF(beta_array);
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_array);
    f64 *b = (f64*)PyArray_DATA(b_array);
    f64 *alphar = (f64*)PyArray_DATA((PyArrayObject*)alphar_array);
    f64 *alphai = (f64*)PyArray_DATA((PyArrayObject*)alphai_array);
    f64 *beta = (f64*)PyArray_DATA((PyArrayObject*)beta_array);

    mb03xp(job_str, compq_str, compz_str, n, ilo, ihi,
           a, lda, b, ldb, q_data, ldq, z_data, ldz,
           alphar, alphai, beta, dwork, ldwork, &info);

    free(dwork);
    if (!wantq || n <= 0) free(q_data);
    if (!wantz || n <= 0) free(z_data);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    if (wantq && n > 0) {
        // Already allocated and populated
    } else {
        npy_intp zero_dims[2] = {0, 0};
        // PyArray_New with zero dims, no data
        Py_XDECREF(q_array); // If it was allocated as 1x1 dummy or handling n=0 logic
        q_array = (PyArrayObject*)PyArray_ZEROS(2, zero_dims, NPY_DOUBLE, 1);
    }

    if (wantz && n > 0) {
        // Already allocated and populated
    } else {
        npy_intp zero_dims[2] = {0, 0};
        // PyArray_New with zero dims
        Py_XDECREF(z_array);
        z_array = (PyArrayObject*)PyArray_ZEROS(2, zero_dims, NPY_DOUBLE, 1);
    }

    PyObject *result = Py_BuildValue("OOOOOOOi",
                                     a_array, b_array, q_array, z_array,
                                     alphar_array, alphai_array, beta_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);
    return result;
}



/* Python wrapper for mb03xd */
PyObject* py_mb03xd(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *balanc_str, *job_str, *jobu_str, *jobv_str;
    i32 n;
    PyObject *a_obj, *qg_obj;
    PyArrayObject *a_array = NULL, *qg_array = NULL;
    i32 info = 0;

    static char *kwlist[] = {"balanc", "job", "jobu", "jobv", "n", "a", "qg", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssssiOO", kwlist,
                                     &balanc_str, &job_str, &jobu_str, &jobv_str,
                                     &n, &a_obj, &qg_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    qg_array = (PyArrayObject*)PyArray_FROM_OTF(qg_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (qg_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda = (i32)a_dims[0];
    if (lda < 1) lda = 1;
    npy_intp *qg_dims = PyArray_DIMS(qg_array);
    i32 ldqg = (i32)qg_dims[0];
    if (ldqg < 1) ldqg = 1;

    i32 n_alloc = (n > 0) ? n : 1;

    bool wantu = (jobu_str[0] == 'U' || jobu_str[0] == 'u');
    bool wantv = (jobv_str[0] == 'V' || jobv_str[0] == 'v');
    bool wantg = (job_str[0] == 'G' || job_str[0] == 'g');
    (void)wantg;

    i32 ldt = n_alloc;
    i32 ldu1 = wantu ? n_alloc : 1;
    i32 ldu2 = ldu1;
    i32 ldv1 = wantv ? n_alloc : 1;
    i32 ldv2 = ldv1;

    f64 *t = (f64*)malloc(ldt * n_alloc * sizeof(f64));
    f64 *u1 = (f64*)malloc(ldu1 * n_alloc * sizeof(f64));
    f64 *u2 = (f64*)malloc(ldu2 * n_alloc * sizeof(f64));
    f64 *v1 = (f64*)malloc(ldv1 * n_alloc * sizeof(f64));
    f64 *v2 = (f64*)malloc(ldv2 * n_alloc * sizeof(f64));
    f64 *scale = (f64*)malloc(n_alloc * sizeof(f64));

    if (t == NULL || u1 == NULL || u2 == NULL || v1 == NULL || v2 == NULL || scale == NULL) {
        free(t); free(u1); free(u2); free(v1); free(v2); free(scale);
        Py_DECREF(a_array);
        Py_DECREF(qg_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 ldwork;
    if (wantg) {
        if (wantu && wantv) {
            ldwork = 7 * n + n * n;
        } else if (!wantu && !wantv) {
            i32 v1t = 7 * n + n * n;
            i32 v2t = 2 * n + 3 * n * n;
            ldwork = (v1t > v2t) ? v1t : v2t;
        } else {
            ldwork = 7 * n + 2 * n * n;
        }
    } else {
        if (!wantu && !wantv) {
            ldwork = 7 * n + n * n;
        } else {
            ldwork = 8 * n;
        }
    }
    if (ldwork < 2) ldwork = 2;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        free(t); free(u1); free(u2); free(v1); free(v2); free(scale);
        Py_DECREF(a_array);
        Py_DECREF(qg_array);
        PyErr_NoMemory();
        return NULL;
    }

    // Allocate n+1 for wr/wi: mb03xd may write wr[n]/wi[n] for complex eigenvalue pairs
    npy_intp dims[1] = {n_alloc};
    npy_intp dims_alloc[1] = {n_alloc + 1};
    PyObject *wr_array = PyArray_SimpleNew(1, dims_alloc, NPY_DOUBLE);
    PyObject *wi_array = PyArray_SimpleNew(1, dims_alloc, NPY_DOUBLE);

    if (wr_array == NULL || wi_array == NULL) {
        free(dwork); free(t); free(u1); free(u2); free(v1); free(v2); free(scale);
        Py_DECREF(a_array);
        Py_DECREF(qg_array);
        Py_XDECREF(wr_array);
        Py_XDECREF(wi_array);
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_array);
    f64 *qg = (f64*)PyArray_DATA(qg_array);
    f64 *wr = (f64*)PyArray_DATA((PyArrayObject*)wr_array);
    f64 *wi = (f64*)PyArray_DATA((PyArrayObject*)wi_array);

    i32 ilo;
    mb03xd(balanc_str, job_str, jobu_str, jobv_str, n, a, lda, qg, ldqg,
           t, ldt, u1, ldu1, u2, ldu2, v1, ldv1, v2, ldv2,
           wr, wi, &ilo, scale, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(qg_array);

    npy_intp t_dims[2] = {n_alloc, n_alloc};
    npy_intp t_strides[2] = {sizeof(f64), ldt * sizeof(f64)};

    PyObject *t_array = PyArray_New(&PyArray_Type, 2, t_dims, NPY_DOUBLE,
                                    t_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (t_array == NULL) {
        free(t); free(u1); free(u2); free(v1); free(v2); free(scale);
        Py_DECREF(a_array);
        Py_DECREF(qg_array);
        Py_DECREF(wr_array);
        Py_DECREF(wi_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)t_array), t, ldt * n_alloc * sizeof(f64));
    free(t);

    PyObject *u1_array_out, *u2_array_out, *v1_array_out, *v2_array_out;
    npy_intp u_dims[2] = {ldu1, n_alloc};
    npy_intp u_strides[2] = {sizeof(f64), ldu1 * sizeof(f64)};
    npy_intp v_dims[2] = {ldv1, n_alloc};
    npy_intp v_strides[2] = {sizeof(f64), ldv1 * sizeof(f64)};

    if (wantu && n > 0) {
        u1_array_out = PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE,
                                   u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        u2_array_out = PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE,
                                   u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (u1_array_out == NULL || u2_array_out == NULL) {
            free(u1); free(u2); free(v1); free(v2); free(scale);
            Py_DECREF(a_array);
            Py_DECREF(qg_array);
            Py_DECREF(wr_array);
            Py_DECREF(wi_array);
            Py_DECREF(t_array);
            Py_XDECREF(u1_array_out);
            Py_XDECREF(u2_array_out);
            return NULL;
        }
        memcpy(PyArray_DATA((PyArrayObject*)u1_array_out), u1, ldu1 * n_alloc * sizeof(f64));
        memcpy(PyArray_DATA((PyArrayObject*)u2_array_out), u2, ldu2 * n_alloc * sizeof(f64));
        free(u1); free(u2);
    } else {
        free(u1); free(u2);
        npy_intp zero_dims[2] = {1, 1};
        u1_array_out = PyArray_ZEROS(2, zero_dims, NPY_DOUBLE, 1);
        u2_array_out = PyArray_ZEROS(2, zero_dims, NPY_DOUBLE, 1);
    }

    if (wantv && n > 0) {
        v1_array_out = PyArray_New(&PyArray_Type, 2, v_dims, NPY_DOUBLE,
                                   v_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        v2_array_out = PyArray_New(&PyArray_Type, 2, v_dims, NPY_DOUBLE,
                                   v_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (v1_array_out == NULL || v2_array_out == NULL) {
            free(v1); free(v2); free(scale);
            Py_DECREF(a_array);
            Py_DECREF(qg_array);
            Py_DECREF(wr_array);
            Py_DECREF(wi_array);
            Py_DECREF(t_array);
            Py_DECREF(u1_array_out);
            Py_DECREF(u2_array_out);
            Py_XDECREF(v1_array_out);
            Py_XDECREF(v2_array_out);
            return NULL;
        }
        memcpy(PyArray_DATA((PyArrayObject*)v1_array_out), v1, ldv1 * n_alloc * sizeof(f64));
        memcpy(PyArray_DATA((PyArrayObject*)v2_array_out), v2, ldv2 * n_alloc * sizeof(f64));
        free(v1); free(v2);
    } else {
        free(v1); free(v2);
        npy_intp zero_dims[2] = {1, 1};
        v1_array_out = PyArray_ZEROS(2, zero_dims, NPY_DOUBLE, 1);
        v2_array_out = PyArray_ZEROS(2, zero_dims, NPY_DOUBLE, 1);
    }

    npy_intp scale_dims[1] = {n_alloc};
    PyObject *scale_array = PyArray_SimpleNew(1, scale_dims, NPY_DOUBLE);
    if (scale_array == NULL) {
        free(scale);
        Py_DECREF(a_array);
        Py_DECREF(qg_array);
        Py_DECREF(wr_array);
        Py_DECREF(wi_array);
        Py_DECREF(t_array);
        Py_DECREF(u1_array_out);
        Py_DECREF(u2_array_out);
        Py_DECREF(v1_array_out);
        Py_DECREF(v2_array_out);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)scale_array), scale, n_alloc * sizeof(f64));
    free(scale);

    // Trim wr/wi from n+1 back to n (extra element was safety margin for complex pairs)
    PyArray_Dims new_dims = {dims, 1};
    PyArray_Resize((PyArrayObject*)wr_array, &new_dims, 1, NPY_ANYORDER);
    PyArray_Resize((PyArrayObject*)wi_array, &new_dims, 1, NPY_ANYORDER);

    PyObject *result = Py_BuildValue("OOOOOOOOOiOi",
                                     a_array, t_array, qg_array,
                                     u1_array_out, u2_array_out,
                                     v1_array_out, v2_array_out,
                                     wr_array, wi_array, ilo, scale_array, info);

    Py_DECREF(a_array);
    Py_DECREF(t_array);
    Py_DECREF(qg_array);
    Py_DECREF(u1_array_out);
    Py_DECREF(u2_array_out);
    Py_DECREF(v1_array_out);
    Py_DECREF(v2_array_out);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);
    Py_DECREF(scale_array);
    return result;
}



/* Python wrapper for mb03ry */
PyObject* py_mb03ry(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj, *c_obj;
    f64 pmax;
    PyArrayObject *a_array = NULL, *b_array = NULL, *c_array = NULL;
    i32 info = 0;

    if (!PyArg_ParseTuple(args, "OOOd", &a_obj, &b_obj, &c_obj, &pmax)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    int a_ndim = PyArray_NDIM(a_array);
    int b_ndim = PyArray_NDIM(b_array);
    int c_ndim = PyArray_NDIM(c_array);

    i32 m, n, lda, ldb, ldc;

    if (a_ndim == 2) {
        npy_intp *a_dims = PyArray_DIMS(a_array);
        m = (i32)a_dims[0];
        lda = m > 0 ? m : 1;
    } else if (a_ndim == 0 || (a_ndim == 1 && PyArray_SIZE(a_array) == 0)) {
        m = 0;
        lda = 1;
    } else {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_ValueError, "a must be 2-dimensional");
        return NULL;
    }

    if (b_ndim == 2) {
        npy_intp *b_dims = PyArray_DIMS(b_array);
        n = (i32)b_dims[0];
        ldb = n > 0 ? n : 1;
    } else if (b_ndim == 0 || (b_ndim == 1 && PyArray_SIZE(b_array) == 0)) {
        n = 0;
        ldb = 1;
    } else {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_ValueError, "b must be 2-dimensional");
        return NULL;
    }

    if (c_ndim == 2) {
        npy_intp *c_dims = PyArray_DIMS(c_array);
        ldc = (i32)c_dims[0];
        if (ldc < 1) ldc = 1;
    } else {
        ldc = 1;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    mb03ry(m, n, pmax, a_data, lda, b_data, ldb, c_data, ldc, &info);

    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("Oi", c_array, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    return result;
}



/* Python wrapper for mb03wd */
PyObject* py_mb03wd(PyObject* self, PyObject* args) {
    const char *job, *compz;
    i32 n, p, ilo, ihi, iloz, ihiz;
    PyObject *h_obj, *z_obj = NULL;
    PyArrayObject *h_array = NULL, *z_array = NULL;
    i32 info = 0;

    if (!PyArg_ParseTuple(args, "ssiiiiiiO|O", &job, &compz, &n, &p, &ilo, &ihi,
                          &iloz, &ihiz, &h_obj, &z_obj)) {
        return NULL;
    }

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }
    if (p < 1) {
        PyErr_SetString(PyExc_ValueError, "p must be at least 1");
        return NULL;
    }
    if (ilo < 1 || ilo > (n > 0 ? n : 1)) {
        PyErr_SetString(PyExc_ValueError, "Invalid ilo");
        return NULL;
    }
    if (ihi < ilo || ihi > n) {
        PyErr_SetString(PyExc_ValueError, "Invalid ihi");
        return NULL;
    }

    h_array = (PyArrayObject*)PyArray_FROM_OTF(h_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (h_array == NULL) {
        return NULL;
    }

    npy_intp *h_dims = PyArray_DIMS(h_array);
    i32 ldh1 = (i32)h_dims[0];
    i32 ldh2 = (i32)h_dims[1];
    if (ldh1 < 1) ldh1 = 1;
    if (ldh2 < 1) ldh2 = 1;

    bool wantz = (*compz == 'I' || *compz == 'i' || *compz == 'V' || *compz == 'v');
    i32 ldz1 = 1, ldz2 = 1;

    if (wantz) {
        if (*compz == 'V' || *compz == 'v') {
            if (z_obj == NULL || z_obj == Py_None) {
                Py_DECREF(h_array);
                PyErr_SetString(PyExc_ValueError, "z required when compz='V'");
                return NULL;
            }
            z_array = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE,
                                                       NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            if (z_array == NULL) {
                Py_DECREF(h_array);
                return NULL;
            }
            npy_intp *z_dims = PyArray_DIMS(z_array);
            ldz1 = (i32)z_dims[0];
            ldz2 = (i32)z_dims[1];
        } else {
            npy_intp z_dims[3] = {n > 0 ? n : 1, n > 0 ? n : 1, p > 0 ? p : 1};
            z_array = (PyArrayObject*)PyArray_ZEROS(3, z_dims, NPY_DOUBLE, 1);
            if (z_array == NULL) {
                Py_DECREF(h_array);
                PyErr_SetString(PyExc_MemoryError, "Failed to allocate z array");
                return NULL;
            }
            ldz1 = n > 0 ? n : 1;
            ldz2 = n > 0 ? n : 1;
        }
    } else {
        npy_intp z_dims[3] = {1, 1, 1};
        z_array = (PyArrayObject*)PyArray_ZEROS(3, z_dims, NPY_DOUBLE, 1);
        if (z_array == NULL) {
            Py_DECREF(h_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate z array");
            return NULL;
        }
    }

    i32 ldwork = ihi - ilo + p - 1;
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (dwork == NULL) {
        Py_DECREF(h_array);
        Py_DECREF(z_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    npy_intp wr_dims[1] = {n > 0 ? n : 1};
    PyObject *wr_array = PyArray_SimpleNew(1, wr_dims, NPY_DOUBLE);
    PyObject *wi_array = PyArray_SimpleNew(1, wr_dims, NPY_DOUBLE);

    if (wr_array == NULL || wi_array == NULL) {
        free(dwork);
        Py_DECREF(h_array);
        Py_DECREF(z_array);
        Py_XDECREF(wr_array);
        Py_XDECREF(wi_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output arrays");
        return NULL;
    }

    f64 *wr = (f64*)PyArray_DATA((PyArrayObject*)wr_array);
    f64 *wi = (f64*)PyArray_DATA((PyArrayObject*)wi_array);
    memset(wr, 0, (n > 0 ? n : 1) * sizeof(f64));
    memset(wi, 0, (n > 0 ? n : 1) * sizeof(f64));

    f64 *h_data = (f64*)PyArray_DATA(h_array);
    f64 *z_data = (f64*)PyArray_DATA(z_array);

    mb03wd(job, compz, n, p, ilo, ihi, iloz, ihiz, h_data, ldh1, ldh2,
           z_data, ldz1, ldz2, wr, wi, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(h_array);
    if (*compz == 'V' || *compz == 'v') {
        PyArray_ResolveWritebackIfCopy(z_array);
    }

    PyObject *result = Py_BuildValue("OOOOi", h_array, z_array, wr_array, wi_array, info);
    Py_DECREF(h_array);
    Py_DECREF(z_array);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);
    return result;
}



/* Python wrapper for mb03qd */
PyObject* py_mb03qd(PyObject* self, PyObject* args) {
    char *dico, *stdom, *jobu;
    int nlow, nsup;
    double alpha;
    PyObject *a_obj, *u_obj = NULL;
    PyArrayObject *a_array, *u_array = NULL;
    i32 ndim, info;

    if (!PyArg_ParseTuple(args, "sssOiid|O", &dico, &stdom, &jobu,
                          &a_obj, &nlow, &nsup, &alpha, &u_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 lda = n;
    i32 ldu = n;

    f64* a = (f64*)PyArray_DATA(a_array);

    f64* u = NULL;
    bool u_allocated = false;

    if (u_obj != NULL && u_obj != Py_None) {
        u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (u_array == NULL) {
            Py_DECREF(a_array);
            return NULL;
        }
        u = (f64*)PyArray_DATA(u_array);
    } else {
        u = (f64*)malloc(n * n * sizeof(f64));
        if (u == NULL) {
            Py_DECREF(a_array);
            return PyErr_NoMemory();
        }
        u_allocated = true;
    }

    f64* dwork = (f64*)malloc(n * sizeof(f64));
    if (dwork == NULL) {
        if (u_allocated) free(u);
        if (u_array) Py_DECREF(u_array);
        Py_DECREF(a_array);
        return PyErr_NoMemory();
    }

    mb03qd(dico, stdom, jobu, n, (i32)nlow, (i32)nsup, alpha, a, lda, u, ldu, &ndim, dwork, &info);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);

    PyObject* u_result;
    if (u_array != NULL) {
        PyArray_ResolveWritebackIfCopy(u_array);
        u_result = (PyObject*)u_array;
        Py_INCREF(u_result);
    } else {
        npy_intp dims[2] = {n, n};
        npy_intp strides[2] = {sizeof(f64), n * sizeof(f64)};
        u_result = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (u_result == NULL) {
            free(u);
            Py_DECREF(a_array);
            return NULL;
        }
        memcpy(PyArray_DATA((PyArrayObject*)u_result), u, n * n * sizeof(f64));
        free(u);
    }

    PyObject* result = Py_BuildValue("OOii", a_array, u_result, (int)ndim, (int)info);

    Py_DECREF(a_array);
    if (u_array) Py_DECREF(u_array);
    Py_DECREF(u_result);

    return result;
}



/* Python wrapper for mb03rd */
PyObject* py_mb03rd(PyObject* self, PyObject* args) {
    char *jobx, *sort;
    double pmax, tol = 0.0;
    PyObject *a_obj, *x_obj = NULL;
    PyArrayObject *a_array, *x_array = NULL;
    i32 nblcks, info;

    if (!PyArg_ParseTuple(args, "ssOd|Od", &jobx, &sort,
                          &a_obj, &pmax, &x_obj, &tol)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    i32 n = (i32)PyArray_DIM(a_array, 0);
    if (n == 0) {
        PyArray_ResolveWritebackIfCopy(a_array);
        npy_intp dims0[2] = {0, 0};
        PyObject* a_result = (PyObject*)a_array;
        PyObject* x_result = PyArray_ZEROS(2, dims0, NPY_DOUBLE, 1);
        npy_intp dims1[1] = {0};
        PyObject* blsize_result = PyArray_ZEROS(1, dims1, NPY_INT32, 0);
        PyObject* wr_result = PyArray_ZEROS(1, dims1, NPY_DOUBLE, 0);
        PyObject* wi_result = PyArray_ZEROS(1, dims1, NPY_DOUBLE, 0);

        PyObject* result = Py_BuildValue("OOiOOOi", a_result, x_result, 0,
                                         blsize_result, wr_result, wi_result, 0);
        Py_DECREF(a_array);
        Py_DECREF(x_result);
        Py_DECREF(blsize_result);
        Py_DECREF(wr_result);
        Py_DECREF(wi_result);
        return result;
    }

    i32 lda = n;
    i32 ldx = (jobx[0] == 'U' || jobx[0] == 'u') ? n : 1;

    f64* a = (f64*)PyArray_DATA(a_array);

    f64* x = NULL;
    bool x_allocated = false;
    bool ljobx = (jobx[0] == 'U' || jobx[0] == 'u');

    if (ljobx) {
        ldx = n;
        if (x_obj != NULL && x_obj != Py_None) {
            x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            if (x_array == NULL) {
                Py_DECREF(a_array);
                return NULL;
            }
            x = (f64*)PyArray_DATA(x_array);
        } else {
            x = (f64*)malloc(n * n * sizeof(f64));
            if (x == NULL) {
                Py_DECREF(a_array);
                return PyErr_NoMemory();
            }
            x_allocated = true;
            for (i32 i = 0; i < n * n; i++) {
                x[i] = 0.0;
            }
            for (i32 i = 0; i < n; i++) {
                x[i + i * n] = 1.0;
            }
        }
    } else {
        ldx = 1;
    }

    i32* blsize = (i32*)malloc(n * sizeof(i32));
    f64* wr = (f64*)malloc(n * sizeof(f64));
    f64* wi = (f64*)malloc(n * sizeof(f64));
    f64* dwork = (f64*)malloc(n * sizeof(f64));

    if (blsize == NULL || wr == NULL || wi == NULL || dwork == NULL) {
        if (blsize) free(blsize);
        if (wr) free(wr);
        if (wi) free(wi);
        if (dwork) free(dwork);
        if (x_allocated) free(x);
        if (x_array) Py_DECREF(x_array);
        Py_DECREF(a_array);
        return PyErr_NoMemory();
    }

    mb03rd(jobx, sort, n, pmax, a, lda, x, ldx, &nblcks, blsize, wr, wi, tol, dwork, &info);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);

    PyObject* x_result;
    if (ljobx) {
        if (x_array != NULL) {
            PyArray_ResolveWritebackIfCopy(x_array);
            x_result = (PyObject*)x_array;
            Py_INCREF(x_result);
        } else {
            npy_intp dims[2] = {n, n};
            npy_intp strides[2] = {sizeof(f64), n * sizeof(f64)};
            x_result = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            if (x_result == NULL) {
                free(x);
                free(blsize);
                free(wr);
                free(wi);
                Py_DECREF(a_array);
                return NULL;
            }
            memcpy(PyArray_DATA((PyArrayObject*)x_result), x, n * n * sizeof(f64));
            free(x);
            x_allocated = false;
        }
    } else {
        npy_intp dims[2] = {0, 0};
        x_result = PyArray_ZEROS(2, dims, NPY_DOUBLE, 1);
    }

    npy_intp blsize_dims[1] = {n};
    PyObject* blsize_result = PyArray_SimpleNew(1, blsize_dims, NPY_INT32);
    memcpy(PyArray_DATA((PyArrayObject*)blsize_result), blsize, n * sizeof(i32));
    free(blsize);

    npy_intp eig_dims[1] = {n};
    PyObject* wr_result = PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);
    PyObject* wi_result = PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);
    if (wr_result == NULL || wi_result == NULL) {
        free(wr);
        free(wi);
        Py_DECREF(a_array);
        Py_DECREF(x_result);
        Py_DECREF(blsize_result);
        Py_XDECREF(wr_result);
        Py_XDECREF(wi_result);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)wr_result), wr, n * sizeof(f64));
    memcpy(PyArray_DATA((PyArrayObject*)wi_result), wi, n * sizeof(f64));
    free(wr);
    free(wi);

    PyObject* result = Py_BuildValue("OOiOOOi", a_array, x_result, (int)nblcks,
                                     blsize_result, wr_result, wi_result, (int)info);

    Py_DECREF(a_array);
    if (x_array) Py_DECREF(x_array);
    Py_DECREF(x_result);
    Py_DECREF(blsize_result);
    Py_DECREF(wr_result);
    Py_DECREF(wi_result);
    if (x_allocated) free(x);

    return result;
}



PyObject* py_mb03ud(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"n", "a", "jobq", "jobp", "ldwork", NULL};
    i32 n;
    PyObject *a_obj;
    char *jobq_str = "N";
    char *jobp_str = "N";
    i32 ldwork = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iO|ssi", kwlist,
                                     &n, &a_obj, &jobq_str, &jobp_str, &ldwork)) {
        return NULL;
    }

    if (n < 0) {
        PyErr_Format(PyExc_ValueError, "n must be non-negative (n=%d)", n);
        return NULL;
    }

    char jobq = jobq_str[0];
    char jobp = jobp_str[0];

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda = (i32)a_dims[0];

    bool wantq = (jobq == 'V' || jobq == 'v');
    bool wantp = (jobp == 'V' || jobp == 'v');

    i32 ldq = wantq ? (n > 0 ? n : 1) : 1;

    f64 *q = NULL;
    PyArrayObject *q_array = NULL;
    if (wantq) {
        npy_intp q_dims[2] = {n, n};
        npy_intp q_strides[2] = {sizeof(f64), n * sizeof(f64)};
        q_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE,
                                              q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (q_array == NULL) {
            Py_DECREF(a_array);
            return NULL;
        }
        q = (f64*)PyArray_DATA(q_array);
        memset(q, 0, n * n * sizeof(f64));
    }

    npy_intp sv_dims[1] = {n};
    PyArrayObject *sv_array = (PyArrayObject*)PyArray_SimpleNew(1, sv_dims, NPY_DOUBLE);
    if (sv_array == NULL) {
        Py_XDECREF(q_array);
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate sv");
        return NULL;
    }

    f64 *sv = (f64*)PyArray_DATA(sv_array);
    if (n > 0) memset(sv, 0, n * sizeof(f64));

    i32 minwork = n > 0 ? 5 * n : 1;
    if (ldwork == 0) ldwork = minwork;

    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(sv_array);
        Py_XDECREF(q_array);
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *q_data = wantq ? q : NULL;

    i32 info;
    mb03ud(jobq, jobp, n, a_data, lda, q_data, ldq, sv, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);

    if (info < 0) {
        Py_DECREF(sv_array);
        Py_XDECREF(q_array);
        Py_DECREF(a_array);
        PyErr_Format(PyExc_ValueError, "MB03UD: invalid parameter at position %d", -info);
        return NULL;
    }

    PyObject *p_array = wantp ? (PyObject*)a_array : Py_None;
    PyObject *q_result = wantq ? (PyObject*)q_array : Py_None;

    if (!wantp) Py_INCREF(Py_None);
    if (!wantq) Py_INCREF(Py_None);

    PyObject *result = Py_BuildValue("OOOi", sv_array, p_array, q_result, info);

    Py_DECREF(sv_array);
    if (wantq) Py_DECREF(q_array);
    if (wantp) Py_DECREF(a_array);

    return result;
}



/* Python wrapper for mb03py */
PyObject* py_mb03py(PyObject* self, PyObject* args) {
    i32 m, n;
    f64 rcond, svlmax;
    PyObject *a_obj;

    if (!PyArg_ParseTuple(args, "iiOdd", &m, &n, &a_obj, &rcond, &svlmax)) {
        return NULL;
    }

    if (m < 0 || n < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative (m=%d, n=%d)", m, n);
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda = (i32)a_dims[0];
    if (lda < 1) lda = 1;

    i32 mn = (m < n) ? m : n;
    i32 dwork_size = (m > 0) ? (3*m - 1) : 1;
    if (dwork_size < 1) dwork_size = 1;
    f64 *dwork = (f64*)malloc(dwork_size * sizeof(f64));

    if (!dwork) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    npy_intp sval_dims[1] = {3};
    npy_intp jpvt_dims[1] = {m > 0 ? m : 0};
    npy_intp tau_dims[1] = {mn};

    PyObject *sval_array = PyArray_SimpleNew(1, sval_dims, NPY_DOUBLE);
    PyObject *jpvt_array = (m > 0) ? PyArray_SimpleNew(1, jpvt_dims, NPY_INT32)
                                   : PyArray_EMPTY(1, jpvt_dims, NPY_INT32, 0);
    PyObject *tau_array = (mn > 0) ? PyArray_SimpleNew(1, tau_dims, NPY_DOUBLE)
                                   : PyArray_EMPTY(1, tau_dims, NPY_DOUBLE, 0);

    if (!sval_array || !jpvt_array || !tau_array) {
        free(dwork);
        Py_XDECREF(sval_array);
        Py_XDECREF(jpvt_array);
        Py_XDECREF(tau_array);
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output arrays");
        return NULL;
    }

    f64 *sval = (f64*)PyArray_DATA((PyArrayObject*)sval_array);
    i32 *jpvt = (m > 0) ? (i32*)PyArray_DATA((PyArrayObject*)jpvt_array) : NULL;
    f64 *tau = (mn > 0) ? (f64*)PyArray_DATA((PyArrayObject*)tau_array) : NULL;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    i32 rank = 0, info = 0;

    mb03py(m, n, a_data, lda, rcond, svlmax, &rank, sval, jpvt, tau, dwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);

    PyObject *result = Py_BuildValue("(OiiOOO)", a_array, rank, info,
                                     sval_array, jpvt_array, tau_array);

    Py_DECREF(a_array);
    Py_DECREF(sval_array);
    Py_DECREF(jpvt_array);
    Py_DECREF(tau_array);

    return result;
}



/* Python wrapper for mb3oyz - Complex rank-revealing QR with column pivoting */
PyObject* py_mb3oyz(PyObject* self, PyObject* args) {
    i32 m, n;
    f64 rcond, svlmax;
    PyObject *a_obj;

    if (!PyArg_ParseTuple(args, "iiOdd", &m, &n, &a_obj, &rcond, &svlmax)) {
        return NULL;
    }

    if (m < 0 || n < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative (m=%d, n=%d)", m, n);
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    i32 lda = m > 0 ? m : 1;
    i32 mn = m < n ? m : n;

    i32 dwork_size = n > 0 ? 2 * n : 1;
    f64 *dwork = (f64*)malloc(dwork_size * sizeof(f64));
    i32 zwork_size = n > 0 ? 3 * n - 1 : 1;
    if (zwork_size < 1) zwork_size = 1;
    c128 *zwork = (c128*)malloc(zwork_size * sizeof(c128));

    if (!dwork || !zwork) {
        free(dwork); free(zwork);
        Py_DECREF(a_array);
        PyErr_NoMemory();
        return NULL;
    }

    npy_intp sval_dims[1] = {3};
    npy_intp jpvt_dims[1] = {n > 0 ? n : 0};
    npy_intp tau_dims[1] = {mn};

    PyObject *sval_array = PyArray_SimpleNew(1, sval_dims, NPY_DOUBLE);
    PyObject *jpvt_array = n > 0 ? PyArray_SimpleNew(1, jpvt_dims, NPY_INT32) : PyArray_EMPTY(1, jpvt_dims, NPY_INT32, 0);
    PyObject *tau_array = mn > 0 ? PyArray_SimpleNew(1, tau_dims, NPY_CDOUBLE) : PyArray_EMPTY(1, tau_dims, NPY_CDOUBLE, 0);

    if (!sval_array || !jpvt_array || !tau_array) {
        free(dwork); free(zwork);
        Py_XDECREF(sval_array);
        Py_XDECREF(jpvt_array);
        Py_XDECREF(tau_array);
        Py_DECREF(a_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *sval = (f64*)PyArray_DATA((PyArrayObject*)sval_array);
    i32 *jpvt = n > 0 ? (i32*)PyArray_DATA((PyArrayObject*)jpvt_array) : NULL;
    c128 *tau = mn > 0 ? (c128*)PyArray_DATA((PyArrayObject*)tau_array) : NULL;

    c128 *a_data = (c128*)PyArray_DATA(a_array);
    i32 rank = 0;
    i32 info = slicot_mb3oyz(m, n, a_data, lda, rcond, svlmax, &rank, sval, jpvt, tau, dwork, zwork);

    PyArray_ResolveWritebackIfCopy(a_array);

    free(dwork);
    free(zwork);

    PyObject *result = Py_BuildValue("(OiiOOO)", a_array, rank, info, sval_array, jpvt_array, tau_array);
    Py_DECREF(a_array);
    Py_DECREF(sval_array);
    Py_DECREF(jpvt_array);
    Py_DECREF(tau_array);
    return result;
}



/* Python wrapper for mb3pyz - Complex rank-revealing RQ with row pivoting */
PyObject* py_mb3pyz(PyObject* self, PyObject* args) {
    i32 m, n;
    f64 rcond, svlmax;
    PyObject *a_obj;

    if (!PyArg_ParseTuple(args, "iiOdd", &m, &n, &a_obj, &rcond, &svlmax)) {
        return NULL;
    }

    if (m < 0 || n < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative (m=%d, n=%d)", m, n);
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    i32 lda = m > 0 ? m : 1;
    i32 mn = m < n ? m : n;

    i32 dwork_size = m > 0 ? 2 * m : 1;
    f64 *dwork = (f64*)malloc(dwork_size * sizeof(f64));
    i32 zwork_size = m > 0 ? 3 * m - 1 : 1;
    if (zwork_size < 1) zwork_size = 1;
    c128 *zwork = (c128*)malloc(zwork_size * sizeof(c128));

    if (!dwork || !zwork) {
        free(dwork); free(zwork);
        Py_DECREF(a_array);
        PyErr_NoMemory();
        return NULL;
    }

    npy_intp sval_dims[1] = {3};
    npy_intp jpvt_dims[1] = {m > 0 ? m : 0};
    npy_intp tau_dims[1] = {mn};

    PyObject *sval_array = PyArray_SimpleNew(1, sval_dims, NPY_DOUBLE);
    PyObject *jpvt_array = m > 0 ? PyArray_SimpleNew(1, jpvt_dims, NPY_INT32) : PyArray_EMPTY(1, jpvt_dims, NPY_INT32, 0);
    PyObject *tau_array = mn > 0 ? PyArray_SimpleNew(1, tau_dims, NPY_CDOUBLE) : PyArray_EMPTY(1, tau_dims, NPY_CDOUBLE, 0);

    if (!sval_array || !jpvt_array || !tau_array) {
        free(dwork); free(zwork);
        Py_XDECREF(sval_array);
        Py_XDECREF(jpvt_array);
        Py_XDECREF(tau_array);
        Py_DECREF(a_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *sval = (f64*)PyArray_DATA((PyArrayObject*)sval_array);
    i32 *jpvt = m > 0 ? (i32*)PyArray_DATA((PyArrayObject*)jpvt_array) : NULL;
    c128 *tau = mn > 0 ? (c128*)PyArray_DATA((PyArrayObject*)tau_array) : NULL;

    c128 *a_data = (c128*)PyArray_DATA(a_array);
    i32 rank = 0;
    i32 info = slicot_mb3pyz(m, n, a_data, lda, rcond, svlmax, &rank, sval, jpvt, tau, dwork, zwork);

    PyArray_ResolveWritebackIfCopy(a_array);

    free(dwork);
    free(zwork);

    PyObject *result = Py_BuildValue("(OiiOOO)", a_array, rank, info, sval_array, jpvt_array, tau_array);
    Py_DECREF(a_array);
    Py_DECREF(sval_array);
    Py_DECREF(jpvt_array);
    Py_DECREF(tau_array);
    return result;
}



/* Python wrapper for mb03qy */
PyObject* py_mb03qy(PyObject* self, PyObject* args) {
    i32 n, l;
    PyObject *a_obj, *u_obj;

    if (!PyArg_ParseTuple(args, "iiOO", &n, &l, &a_obj, &u_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (u_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 lda = (i32)PyArray_DIM(a_array, 0);
    i32 ldu = (i32)PyArray_DIM(u_array, 0);

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *u_data = (f64*)PyArray_DATA(u_array);

    f64 e1 = 0.0, e2 = 0.0;
    i32 info = 0;

    mb03qy(n, l, a_data, lda, u_data, ldu, &e1, &e2, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(u_array);

    PyObject *result = Py_BuildValue("(OOddi)", a_array, u_array, e1, e2, info);
    Py_DECREF(a_array);
    Py_DECREF(u_array);
    return result;
}



/* Python wrapper for mb03qx */
PyObject* py_mb03qx(PyObject* self, PyObject* args) {
    PyObject *t_obj;

    if (!PyArg_ParseTuple(args, "O", &t_obj)) {
        return NULL;
    }

    PyArrayObject *t_array = (PyArrayObject*)PyArray_FROM_OTF(t_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (t_array == NULL) return NULL;

    i32 n = (i32)PyArray_DIM(t_array, 0);
    i32 ldt = n;
    const f64 *t_data = (const f64*)PyArray_DATA(t_array);

    npy_intp dims[1] = {n};
    PyArrayObject *wr_array = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    PyArrayObject *wi_array = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (wr_array == NULL || wi_array == NULL) {
        Py_XDECREF(wr_array);
        Py_XDECREF(wi_array);
        Py_DECREF(t_array);
        return NULL;
    }

    f64 *wr_data = (f64*)PyArray_DATA(wr_array);
    f64 *wi_data = (f64*)PyArray_DATA(wi_array);
    i32 info = 0;

    mb03qx(n, t_data, ldt, wr_data, wi_data, &info);

    Py_DECREF(t_array);

    PyObject *result = Py_BuildValue("(OOi)", wr_array, wi_array, info);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);
    return result;
}



/* Python wrapper for mb03rx */
PyObject* py_mb03rx(PyObject* self, PyObject* args) {
    const char* jobv;
    i32 kl, ku;
    PyObject *a_obj, *x_obj, *wr_obj, *wi_obj;

    if (!PyArg_ParseTuple(args, "siiOOOO", &jobv, &kl, &ku, &a_obj, &x_obj, &wr_obj, &wi_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (x_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *wr_array = (PyArrayObject*)PyArray_FROM_OTF(wr_obj, NPY_DOUBLE,
                                                                NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (wr_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(x_array);
        return NULL;
    }

    PyArrayObject *wi_array = (PyArrayObject*)PyArray_FROM_OTF(wi_obj, NPY_DOUBLE,
                                                                NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (wi_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(x_array);
        Py_DECREF(wr_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 lda = n;
    i32 ldx = (i32)PyArray_DIM(x_array, 0);

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *x_data = (f64*)PyArray_DATA(x_array);
    f64 *wr_data = (f64*)PyArray_DATA(wr_array);
    f64 *wi_data = (f64*)PyArray_DATA(wi_array);

    f64* dwork = (f64*)malloc(n * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(x_array);
        Py_DECREF(wr_array);
        Py_DECREF(wi_array);
        return PyErr_NoMemory();
    }

    mb03rx(jobv, n, kl, &ku, a_data, lda, x_data, ldx, wr_data, wi_data, dwork);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(x_array);
    PyArray_ResolveWritebackIfCopy(wr_array);
    PyArray_ResolveWritebackIfCopy(wi_array);

    PyObject *result = Py_BuildValue("(OOOOi)", a_array, x_array, wr_array, wi_array, ku);
    Py_DECREF(a_array);
    Py_DECREF(x_array);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);
    return result;
}



/* Python wrapper for mb03bd */
PyObject* py_mb03bd(PyObject* self, PyObject* args) {
    const char *job, *defl, *compq;
    i32 k, n, h, ilo, ihi;
    PyObject *s_obj, *a_obj;
    PyArrayObject *s_array = NULL, *a_array = NULL;
    i32 info = 0, iwarn = 0;

    if (!PyArg_ParseTuple(args, "sssiiiiiOO", &job, &defl, &compq,
                          &k, &n, &h, &ilo, &ihi, &s_obj, &a_obj)) {
        return NULL;
    }

    if (k < 1) {
        PyErr_SetString(PyExc_ValueError, "K must be >= 1");
        return NULL;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "N must be >= 0");
        return NULL;
    }
    if (h < 1 || h > k) {
        PyErr_SetString(PyExc_ValueError, "H must satisfy 1 <= H <= K");
        return NULL;
    }

    s_array = (PyArrayObject*)PyArray_FROM_OTF(s_obj, NPY_INT32,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (s_array == NULL) return NULL;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        Py_DECREF(s_array);
        return NULL;
    }

    const i32 *s_data = (const i32*)PyArray_DATA(s_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda1 = (n > 0) ? (i32)a_dims[0] : 1;
    i32 lda2 = (n > 0 && PyArray_NDIM(a_array) >= 2) ? (i32)a_dims[1] : 1;
    if (lda1 < 1) lda1 = 1;
    if (lda2 < 1) lda2 = 1;

    char compq_upper = (char)toupper((unsigned char)compq[0]);
    bool wantq = (compq_upper == 'U' || compq_upper == 'I' || compq_upper == 'P');

    i32 ldq1 = wantq ? (n > 0 ? n : 1) : 1;
    i32 ldq2 = wantq ? (n > 0 ? n : 1) : 1;

    npy_intp q_dims[3] = {ldq1, ldq2, wantq ? k : 1};

    f64 *q_data = NULL;
    npy_intp n_dim = (n > 0) ? n : 0;
    npy_intp alphar_dims[1] = {n_dim};
    npy_intp alphai_dims[1] = {n_dim};
    npy_intp beta_dims[1] = {n_dim};
    npy_intp scal_dims[1] = {n_dim};

    PyObject *q_array = NULL;
    if (wantq && n > 0) {
        q_array = PyArray_New(&PyArray_Type, 3, q_dims, NPY_DOUBLE,
                              NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        q_data = (f64*)PyArray_DATA((PyArrayObject*)q_array);
        memset(q_data, 0, PyArray_NBYTES((PyArrayObject*)q_array));
    } else {
        q_data = (f64*)calloc(1, sizeof(f64)); // Dummy
    }

    PyObject *alphar_array = NULL, *alphai_array = NULL, *beta_array = NULL, *scal_array = NULL;
    f64 *alphar_data = NULL, *alphai_data = NULL, *beta_data = NULL;
    i32 *scal_data = NULL;

    if (n > 0) {
        alphar_array = PyArray_SimpleNew(1, alphar_dims, NPY_DOUBLE);
        alphai_array = PyArray_SimpleNew(1, alphai_dims, NPY_DOUBLE);
        beta_array = PyArray_SimpleNew(1, beta_dims, NPY_DOUBLE);
        scal_array = PyArray_SimpleNew(1, scal_dims, NPY_INT32);
        
        if (alphar_array && alphai_array && beta_array && scal_array) {
            alphar_data = (f64*)PyArray_DATA((PyArrayObject*)alphar_array);
            alphai_data = (f64*)PyArray_DATA((PyArrayObject*)alphai_array);
            beta_data = (f64*)PyArray_DATA((PyArrayObject*)beta_array);
            scal_data = (i32*)PyArray_DATA((PyArrayObject*)scal_array);
            memset(alphar_data, 0, PyArray_NBYTES((PyArrayObject*)alphar_array));
            memset(alphai_data, 0, PyArray_NBYTES((PyArrayObject*)alphai_array));
            memset(beta_data, 0, PyArray_NBYTES((PyArrayObject*)beta_array));
            memset(scal_data, 0, PyArray_NBYTES((PyArrayObject*)scal_array));
        }
    }

    i32 liwork = 2 * k + n;
    i32 ldwork = k + ((2 * n > 8 * k) ? 2 * n : 8 * k);
    if (ldwork < 1) ldwork = 1;

    i32 *iwork = (i32*)calloc(liwork > 0 ? liwork : 1, sizeof(i32));
    f64 *dwork = (f64*)calloc(ldwork > 0 ? ldwork : 1, sizeof(f64));

    i32 *qind = (i32*)calloc(k, sizeof(i32));

    if ((wantq && n > 0 && q_array == NULL) || iwork == NULL || dwork == NULL || qind == NULL ||
        (n > 0 && (alphar_array == NULL || alphai_array == NULL ||
                   beta_array == NULL || scal_array == NULL))) {
        if (wantq && n > 0 && q_array == NULL) { /* nothing */ }
        else if (wantq && n <= 0) free(q_data);
        
        Py_XDECREF(q_array);
        Py_XDECREF(alphar_array);
        Py_XDECREF(alphai_array);
        Py_XDECREF(beta_array);
        Py_XDECREF(scal_array);
        free(iwork); free(dwork); free(qind);
        if (!wantq) free(q_data); 
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        return PyErr_NoMemory();
    }

    mb03bd(job, defl, compq, qind, k, n, h, ilo, ihi, s_data,
           a_data, lda1, lda2, q_data, ldq1, ldq2,
           alphar_data, alphai_data, beta_data, scal_data,
           iwork, liwork, dwork, ldwork, &iwarn, &info);

    free(iwork);
    free(dwork);
    free(qind);

    PyArray_ResolveWritebackIfCopy(a_array);

    PyObject *q_array_out = NULL;
    if (wantq && n > 0) {
        // Already allocated
        q_array_out = (PyObject*)q_array;
    } else {
        if (!wantq || n <= 0) free(q_data);
        npy_intp empty_dims[3] = {0, 0, 0};
        q_array_out = PyArray_EMPTY(3, empty_dims, NPY_DOUBLE, 1);
    }

    // alphar/alphai/beta/scal already declared at function start
    if (n <= 0) {
        npy_intp empty_dims[1] = {0};
        alphar_array = PyArray_EMPTY(1, empty_dims, NPY_DOUBLE, 0);
        alphai_array = PyArray_EMPTY(1, empty_dims, NPY_DOUBLE, 0);
        beta_array = PyArray_EMPTY(1, empty_dims, NPY_DOUBLE, 0);
        scal_array = PyArray_EMPTY(1, empty_dims, NPY_INT32, 0);
    }

    PyObject *result = Py_BuildValue("(OOOOOOii)",
                                     a_array, q_array_out,
                                     alphar_array, alphai_array, beta_array, scal_array,
                                     iwarn, info);

    Py_DECREF(s_array);
    Py_DECREF(a_array);
    Py_DECREF(q_array_out);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);
    Py_DECREF(scal_array);

    return result;
}


/* Python wrapper for mb03bb */
PyObject* py_mb03bb(PyObject* self, PyObject* args) {
    i32 k, sinv;
    PyObject *amap_obj, *s_obj, *a_obj;

    if (!PyArg_ParseTuple(args, "iOOiO", &k, &amap_obj, &s_obj, &sinv, &a_obj)) {
        return NULL;
    }

    if (k < 1) {
        PyErr_SetString(PyExc_ValueError, "K must be >= 1");
        return NULL;
    }

    PyArrayObject *amap_array = (PyArrayObject*)PyArray_FROM_OTF(
        amap_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (amap_array == NULL) return NULL;

    PyArrayObject *s_array = (PyArrayObject*)PyArray_FROM_OTF(
        s_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (s_array == NULL) {
        Py_DECREF(amap_array);
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (a_array == NULL) {
        Py_DECREF(amap_array);
        Py_DECREF(s_array);
        return NULL;
    }

    const i32 *amap_data = (const i32*)PyArray_DATA(amap_array);
    const i32 *s_data = (const i32*)PyArray_DATA(s_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda1 = (i32)a_dims[0];
    i32 lda2 = (i32)a_dims[1];
    if (lda1 < 2) lda1 = 2;
    if (lda2 < 2) lda2 = 2;

    f64 base = 2.0;
    f64 lgbas = log(base);
    f64 ulp = 2.220446049250313e-16;

    npy_intp out_dims[1] = {2};
    f64 *alphar_data = (f64*)calloc(2, sizeof(f64));
    f64 *alphai_data = (f64*)calloc(2, sizeof(f64));
    f64 *beta_data = (f64*)calloc(2, sizeof(f64));
    i32 *scal_data = (i32*)calloc(2, sizeof(i32));

    i32 ldwork = 8 * k;
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (alphar_data == NULL || alphai_data == NULL || beta_data == NULL ||
        scal_data == NULL || dwork == NULL) {
        free(alphar_data);
        free(alphai_data);
        free(beta_data);
        free(scal_data);
        free(dwork);
        Py_DECREF(amap_array);
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        return PyErr_NoMemory();
    }

    i32 info = 0;
    mb03bb(base, lgbas, ulp, k, amap_data, s_data, sinv,
           a_data, lda1, lda2, alphar_data, alphai_data,
           beta_data, scal_data, dwork, &info);

    free(dwork);
    Py_DECREF(amap_array);
    Py_DECREF(s_array);
    Py_DECREF(a_array);

    PyObject *alphar_array = PyArray_SimpleNew(1, out_dims, NPY_DOUBLE);
    PyObject *alphai_array = PyArray_SimpleNew(1, out_dims, NPY_DOUBLE);
    PyObject *beta_array = PyArray_SimpleNew(1, out_dims, NPY_DOUBLE);
    PyObject *scal_array = PyArray_SimpleNew(1, out_dims, NPY_INT32);
    
    if (alphar_array && alphai_array && beta_array && scal_array) {
        memcpy(PyArray_DATA((PyArrayObject*)alphar_array), alphar_data, 2*sizeof(f64));
        memcpy(PyArray_DATA((PyArrayObject*)alphai_array), alphai_data, 2*sizeof(f64));
        memcpy(PyArray_DATA((PyArrayObject*)beta_array), beta_data, 2*sizeof(f64));
        memcpy(PyArray_DATA((PyArrayObject*)scal_array), scal_data, 2*sizeof(i32));
    }
 
    free(alphar_data);
    free(alphai_data);
    free(beta_data);
    free(scal_data); // Free the temp buffers allocated with calloc

    PyObject *result = Py_BuildValue("(OOOOi)",
                                     alphar_array, alphai_array, beta_array, scal_array, info);

    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);
    Py_DECREF(scal_array);

    return result;
}


PyObject* py_mb03ag(PyObject* self, PyObject* args) {
    const char *shft;
    i32 k, n;
    PyObject *amap_obj, *s_obj, *a_obj;
    i32 sinv;

    if (!PyArg_ParseTuple(args, "siiOOiO", &shft, &k, &n, &amap_obj, &s_obj,
                          &sinv, &a_obj)) {
        return NULL;
    }

    if (k < 1) {
        PyErr_SetString(PyExc_ValueError, "K must be >= 1");
        return NULL;
    }
    if (n < 2) {
        PyErr_SetString(PyExc_ValueError, "N must be >= 2");
        return NULL;
    }

    PyArrayObject *amap_array = (PyArrayObject*)PyArray_FROM_OTF(
        amap_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (amap_array == NULL) return NULL;

    PyArrayObject *s_array = (PyArrayObject*)PyArray_FROM_OTF(
        s_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (s_array == NULL) {
        Py_DECREF(amap_array);
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (a_array == NULL) {
        Py_DECREF(amap_array);
        Py_DECREF(s_array);
        return NULL;
    }

    const i32 *amap_data = (const i32*)PyArray_DATA(amap_array);
    const i32 *s_data = (const i32*)PyArray_DATA(s_array);
    const f64 *a_data = (const f64*)PyArray_DATA(a_array);

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda1 = (i32)a_dims[0];
    i32 lda2 = (i32)a_dims[1];

    i32 liwork = 2 * n;
    i32 ldwork = 2 * n * n;
    i32 *iwork = (i32*)calloc(liwork > 0 ? liwork : 1, sizeof(i32));
    f64 *dwork = (f64*)calloc(ldwork > 0 ? ldwork : 1, sizeof(f64));
    if (iwork == NULL || dwork == NULL) {
        free(iwork);
        free(dwork);
        Py_DECREF(amap_array);
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        return PyErr_NoMemory();
    }

    f64 c1 = 0.0, s1 = 0.0, c2 = 0.0, s2 = 0.0;

    mb03ag(shft, k, n, amap_data, s_data, sinv, a_data, lda1, lda2,
           &c1, &s1, &c2, &s2, iwork, dwork);

    free(iwork);
    free(dwork);
    Py_DECREF(amap_array);
    Py_DECREF(s_array);
    Py_DECREF(a_array);

    return Py_BuildValue("(dddd)", c1, s1, c2, s2);
}


PyObject* py_mb03ai(PyObject* self, PyObject* args) {
    const char *shft;
    i32 k, n;
    PyObject *amap_obj, *s_obj, *a_obj;
    i32 sinv;

    if (!PyArg_ParseTuple(args, "siiOOiO", &shft, &k, &n, &amap_obj, &s_obj,
                          &sinv, &a_obj)) {
        return NULL;
    }

    if (k < 1) {
        PyErr_SetString(PyExc_ValueError, "K must be >= 1");
        return NULL;
    }
    if (n < 2) {
        PyErr_SetString(PyExc_ValueError, "N must be >= 2");
        return NULL;
    }

    PyArrayObject *amap_array = (PyArrayObject*)PyArray_FROM_OTF(
        amap_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (amap_array == NULL) return NULL;

    PyArrayObject *s_array = (PyArrayObject*)PyArray_FROM_OTF(
        s_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (s_array == NULL) {
        Py_DECREF(amap_array);
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (a_array == NULL) {
        Py_DECREF(amap_array);
        Py_DECREF(s_array);
        return NULL;
    }

    const i32 *amap_data = (const i32*)PyArray_DATA(amap_array);
    const i32 *s_data = (const i32*)PyArray_DATA(s_array);
    const f64 *a_data = (const f64*)PyArray_DATA(a_array);

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda1 = (i32)a_dims[0];
    i32 lda2 = (i32)a_dims[1];

    i32 ldwork = n * (n + 2);
    f64 *dwork = (f64*)calloc(ldwork > 0 ? ldwork : 1, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(amap_array);
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        return PyErr_NoMemory();
    }

    f64 c1 = 0.0, s1 = 0.0, c2 = 0.0, s2 = 0.0;

    mb03ai(shft, k, n, amap_data, s_data, sinv, a_data, lda1, lda2,
           &c1, &s1, &c2, &s2, dwork);

    free(dwork);
    Py_DECREF(amap_array);
    Py_DECREF(s_array);
    Py_DECREF(a_array);

    return Py_BuildValue("(dddd)", c1, s1, c2, s2);
}

PyObject* py_mb03bc(PyObject* self, PyObject* args) {
    i32 k;
    PyObject *amap_obj, *s_obj, *a_obj, *macpar_obj;
    i32 sinv;

    if (!PyArg_ParseTuple(args, "iOOiOO", &k, &amap_obj, &s_obj, &sinv,
                          &a_obj, &macpar_obj)) {
        return NULL;
    }

    if (k < 1) {
        PyErr_SetString(PyExc_ValueError, "K must be >= 1");
        return NULL;
    }

    PyArrayObject *amap_array = (PyArrayObject*)PyArray_FROM_OTF(
        amap_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (amap_array == NULL) return NULL;

    PyArrayObject *s_array = (PyArrayObject*)PyArray_FROM_OTF(
        s_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (s_array == NULL) {
        Py_DECREF(amap_array);
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_INOUT_ARRAY2);
    if (a_array == NULL) {
        Py_DECREF(amap_array);
        Py_DECREF(s_array);
        return NULL;
    }

    PyArrayObject *macpar_array = (PyArrayObject*)PyArray_FROM_OTF(
        macpar_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (macpar_array == NULL) {
        Py_DECREF(amap_array);
        Py_DECREF(s_array);
        PyArray_DiscardWritebackIfCopy(a_array);
        Py_DECREF(a_array);
        return NULL;
    }

    const i32 *amap_data = (const i32*)PyArray_DATA(amap_array);
    const i32 *s_data = (const i32*)PyArray_DATA(s_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    const f64 *macpar_data = (const f64*)PyArray_DATA(macpar_array);

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda1 = (i32)a_dims[0];
    i32 lda2 = (i32)a_dims[1];

    i32 ldwork = 3 * (k - 1);
    f64 *dwork = (f64*)calloc(ldwork > 0 ? ldwork : 1, sizeof(f64));

    if (dwork == NULL) {
        Py_DECREF(amap_array);
        Py_DECREF(s_array);
        PyArray_DiscardWritebackIfCopy(a_array);
        Py_DECREF(a_array);
        Py_DECREF(macpar_array);
        return PyErr_NoMemory();
    }

    npy_intp cv_dims[1] = {k};
    npy_intp sv_dims[1] = {k};
    PyObject *cv_array = PyArray_SimpleNew(1, cv_dims, NPY_DOUBLE);
    PyObject *sv_array = PyArray_SimpleNew(1, sv_dims, NPY_DOUBLE);

    if (cv_array == NULL || sv_array == NULL) {
        Py_XDECREF(cv_array);
        Py_XDECREF(sv_array);
        free(dwork);
        Py_DECREF(amap_array);
        Py_DECREF(s_array);
        PyArray_DiscardWritebackIfCopy(a_array);
        Py_DECREF(a_array);
        Py_DECREF(macpar_array);
        return NULL;
    }

    f64 *cv = (f64*)PyArray_DATA((PyArrayObject*)cv_array);
    f64 *sv = (f64*)PyArray_DATA((PyArrayObject*)sv_array);
    memset(cv, 0, k * sizeof(f64));
    memset(sv, 0, k * sizeof(f64));

    mb03bc(k, amap_data, s_data, sinv, a_data, lda1, lda2, macpar_data,
           cv, sv, dwork);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);

    PyObject *result = Py_BuildValue("(OOO)", a_array, cv_array, sv_array);

    Py_DECREF(amap_array);
    Py_DECREF(s_array);
    Py_DECREF(a_array);
    Py_DECREF(macpar_array);
    Py_DECREF(cv_array);
    Py_DECREF(sv_array);

    return result;
}

PyObject* py_mb03be(PyObject* self, PyObject* args) {
    i32 k;
    PyObject *amap_obj, *s_obj, *a_obj;
    i32 sinv;

    if (!PyArg_ParseTuple(args, "iOOiO", &k, &amap_obj, &s_obj, &sinv, &a_obj)) {
        return NULL;
    }

    if (k < 1) {
        PyErr_SetString(PyExc_ValueError, "K must be >= 1");
        return NULL;
    }

    PyArrayObject *amap_array = (PyArrayObject*)PyArray_FROM_OTF(
        amap_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (amap_array == NULL) return NULL;

    PyArrayObject *s_array = (PyArrayObject*)PyArray_FROM_OTF(
        s_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (s_array == NULL) {
        Py_DECREF(amap_array);
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        Py_DECREF(amap_array);
        Py_DECREF(s_array);
        return NULL;
    }

    const i32 *amap_data = (const i32*)PyArray_DATA(amap_array);
    const i32 *s_data = (const i32*)PyArray_DATA(s_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda1 = (i32)a_dims[0];
    i32 lda2 = (i32)a_dims[1];

    mb03be(k, amap_data, s_data, sinv, a_data, lda1, lda2);

    PyArray_ResolveWritebackIfCopy(a_array);

    Py_INCREF(a_array);
    PyObject *result = (PyObject*)a_array;

    Py_DECREF(amap_array);
    Py_DECREF(s_array);
    Py_DECREF(a_array);

    return result;
}

PyObject* py_mb03bf(PyObject* self, PyObject* args) {
    i32 k;
    PyObject *amap_obj, *s_obj, *a_obj;
    i32 sinv;
    f64 ulp;

    if (!PyArg_ParseTuple(args, "iOOiOd", &k, &amap_obj, &s_obj, &sinv, &a_obj, &ulp)) {
        return NULL;
    }

    if (k < 1) {
        PyErr_SetString(PyExc_ValueError, "K must be >= 1");
        return NULL;
    }

    PyArrayObject *amap_array = (PyArrayObject*)PyArray_FROM_OTF(
        amap_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (amap_array == NULL) return NULL;

    PyArrayObject *s_array = (PyArrayObject*)PyArray_FROM_OTF(
        s_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (s_array == NULL) {
        Py_DECREF(amap_array);
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        Py_DECREF(amap_array);
        Py_DECREF(s_array);
        return NULL;
    }

    const i32 *amap_data = (const i32*)PyArray_DATA(amap_array);
    const i32 *s_data = (const i32*)PyArray_DATA(s_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda1 = (i32)a_dims[0];
    i32 lda2 = (i32)a_dims[1];

    mb03bf(k, amap_data, s_data, sinv, a_data, lda1, lda2, ulp);

    PyArray_ResolveWritebackIfCopy(a_array);

    Py_INCREF(a_array);
    PyObject *result = (PyObject*)a_array;

    Py_DECREF(amap_array);
    Py_DECREF(s_array);
    Py_DECREF(a_array);

    return result;
}

PyObject* py_mb03bg(PyObject* self, PyObject* args) {
    i32 k, n;
    PyObject *amap_obj, *s_obj, *a_obj;
    i32 sinv;

    if (!PyArg_ParseTuple(args, "iiOOiO", &k, &n, &amap_obj, &s_obj, &sinv, &a_obj)) {
        return NULL;
    }

    if (k < 1) {
        PyErr_SetString(PyExc_ValueError, "K must be >= 1");
        return NULL;
    }
    if (n < 2) {
        PyErr_SetString(PyExc_ValueError, "N must be >= 2");
        return NULL;
    }

    PyArrayObject *amap_array = (PyArrayObject*)PyArray_FROM_OTF(
        amap_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (amap_array == NULL) return NULL;

    PyArrayObject *s_array = (PyArrayObject*)PyArray_FROM_OTF(
        s_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (s_array == NULL) {
        Py_DECREF(amap_array);
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (a_array == NULL) {
        Py_DECREF(amap_array);
        Py_DECREF(s_array);
        return NULL;
    }

    const i32 *amap_data = (const i32*)PyArray_DATA(amap_array);
    const i32 *s_data = (const i32*)PyArray_DATA(s_array);
    const f64 *a_data = (const f64*)PyArray_DATA(a_array);

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda1 = (i32)a_dims[0];
    i32 lda2 = (i32)a_dims[1];

    npy_intp wr_dims[1] = {2};
    PyArrayObject *wr_array = (PyArrayObject*)PyArray_SimpleNew(1, wr_dims, NPY_DOUBLE);
    PyArrayObject *wi_array = (PyArrayObject*)PyArray_SimpleNew(1, wr_dims, NPY_DOUBLE);
    if (wr_array == NULL || wi_array == NULL) {
        Py_XDECREF(wr_array);
        Py_XDECREF(wi_array);
        Py_DECREF(amap_array);
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        return NULL;
    }

    f64 *wr_data = (f64*)PyArray_DATA(wr_array);
    f64 *wi_data = (f64*)PyArray_DATA(wi_array);

    mb03bg(k, n, amap_data, s_data, sinv, a_data, lda1, lda2, wr_data, wi_data);

    PyObject *result = Py_BuildValue("OO", wr_array, wi_array);

    Py_DECREF(amap_array);
    Py_DECREF(s_array);
    Py_DECREF(a_array);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);

    return result;
}

PyObject *py_mb03bz(PyObject *self, PyObject *args, PyObject *kwargs) {
    const char *job;
    const char *compq;
    int k, n, ilo, ihi;
    PyObject *s_obj, *a_obj;
    PyObject *q_obj = Py_None;

    static char *kwlist[] = {"job", "compq", "k", "n", "ilo", "ihi", "s", "a", "q", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssiiiiOO|O", kwlist,
                                     &job, &compq, &k, &n, &ilo, &ihi,
                                     &s_obj, &a_obj, &q_obj)) {
        return NULL;
    }

    PyArrayObject *s_array = (PyArrayObject *)PyArray_FROM_OTF(
        s_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (s_array == NULL) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        Py_DECREF(s_array);
        return NULL;
    }

    i32 *s_data = (i32 *)PyArray_DATA(s_array);
    c128 *a_data = (c128 *)PyArray_DATA(a_array);

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda1 = (i32)a_dims[0];
    i32 lda2 = (i32)a_dims[1];

    int wantq = (compq[0] == 'V' || compq[0] == 'v' ||
                 compq[0] == 'I' || compq[0] == 'i');

    PyArrayObject *q_array = NULL;
    c128 *q_data = NULL;
    i32 ldq1 = 1, ldq2 = 1;

    if (wantq) {
        if (q_obj != Py_None && q_obj != NULL) {
            q_array = (PyArrayObject *)PyArray_FROM_OTF(
                q_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        } else {
            npy_intp q_dims[3] = {n > 0 ? n : 1, n > 0 ? n : 1, k};
            q_array = (PyArrayObject *)PyArray_ZEROS(3, q_dims, NPY_COMPLEX128, 1);
        }
        if (q_array == NULL) {
            Py_DECREF(s_array);
            Py_DECREF(a_array);
            return NULL;
        }
        q_data = (c128 *)PyArray_DATA(q_array);
        npy_intp *q_dims_actual = PyArray_DIMS(q_array);
        ldq1 = (i32)q_dims_actual[0];
        ldq2 = (i32)q_dims_actual[1];
    } else {
        npy_intp q_dims[3] = {1, 1, 1};
        q_array = (PyArrayObject *)PyArray_ZEROS(3, q_dims, NPY_COMPLEX128, 1);
        if (q_array == NULL) {
            Py_DECREF(s_array);
            Py_DECREF(a_array);
            return NULL;
        }
        q_data = (c128 *)PyArray_DATA(q_array);
    }

    i32 ldwork = n > 1 ? n : 1;
    i32 lzwork = n > 1 ? n : 1;

    f64 *dwork = (f64 *)malloc(ldwork * sizeof(f64));
    c128 *zwork = (c128 *)malloc(lzwork * sizeof(c128));
    if (dwork == NULL || zwork == NULL) {
        free(dwork);
        free(zwork);
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        Py_DECREF(q_array);
        PyErr_NoMemory();
        return NULL;
    }

    npy_intp out_dims[1] = {n > 0 ? n : 1};
    PyArrayObject *alpha_array = (PyArrayObject *)PyArray_SimpleNew(1, out_dims, NPY_COMPLEX128);
    PyArrayObject *beta_array = (PyArrayObject *)PyArray_SimpleNew(1, out_dims, NPY_COMPLEX128);
    PyArrayObject *scal_array = (PyArrayObject *)PyArray_SimpleNew(1, out_dims, NPY_INT32);
    if (alpha_array == NULL || beta_array == NULL || scal_array == NULL) {
        free(dwork);
        free(zwork);
        Py_XDECREF(alpha_array);
        Py_XDECREF(beta_array);
        Py_XDECREF(scal_array);
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        Py_DECREF(q_array);
        return NULL;
    }

    c128 *alpha_data = (c128 *)PyArray_DATA(alpha_array);
    c128 *beta_data = (c128 *)PyArray_DATA(beta_array);
    i32 *scal_data = (i32 *)PyArray_DATA(scal_array);

    i32 info = 0;

    mb03bz(job, compq, k, n, ilo, ihi, s_data, a_data, lda1, lda2,
           q_data, ldq1, ldq2, alpha_data, beta_data, scal_data,
           dwork, ldwork, zwork, lzwork, &info);

    free(dwork);
    free(zwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    if (wantq && q_obj != Py_None && q_obj != NULL) {
        PyArray_ResolveWritebackIfCopy(q_array);
    }

    PyObject *result = Py_BuildValue("OOOOOi",
                                     a_array, q_array,
                                     alpha_array, beta_array, scal_array,
                                     (int)info);

    Py_DECREF(s_array);
    Py_DECREF(a_array);
    Py_DECREF(q_array);
    Py_DECREF(alpha_array);
    Py_DECREF(beta_array);
    Py_DECREF(scal_array);

    return result;
}

PyObject *py_mb03cd(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"uplo", "n1", "n2", "prec", "a", "b", "d", NULL};

    const char *uplo;
    int n1, n2;
    double prec;
    PyObject *a_obj = NULL, *b_obj = NULL, *d_obj = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siidOOO", kwlist,
                                     &uplo, &n1, &n2, &prec, &a_obj, &b_obj, &d_obj)) {
        return NULL;
    }

    if (uplo[0] != 'U' && uplo[0] != 'u' && uplo[0] != 'L' && uplo[0] != 'l') {
        PyErr_SetString(PyExc_ValueError, "uplo must be 'U' or 'L'");
        return NULL;
    }

    i32 m = n1 + n2;

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *d_array = (PyArrayObject *)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || b_array == NULL || d_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(d_array);
        return NULL;
    }

    f64 *a_data = (f64 *)PyArray_DATA(a_array);
    f64 *b_data = (f64 *)PyArray_DATA(b_array);
    f64 *d_data = (f64 *)PyArray_DATA(d_array);

    i32 lda = m, ldb = m, ldd = m;
    i32 ldq1 = m, ldq2 = m, ldq3 = m;

    npy_intp dims[2] = {m, m};

    // q1, q2 mallocs removed
    // q1, q2, q3 allocated via PyArray_New later

    // Checks moved to allocation site

    i32 ldwork = 0;
    if (m > 2) {
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            ldwork = 16 * n1 + 10 * n2 + 23;
        } else {
            ldwork = 10 * n1 + 16 * n2 + 23;
        }
    }

    f64 *dwork = NULL;
    if (ldwork > 0) {
        dwork = (f64 *)malloc(ldwork * sizeof(f64));
        if (dwork == NULL) {
            // q1/q2/q3 not allocated yet
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(d_array);
            PyErr_NoMemory();
            return NULL;
        }
    }

    i32 loc_n1 = n1, loc_n2 = n2;
    i32 info = 0;

    PyArrayObject *q1_array = (PyArrayObject *)PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                                           NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyArrayObject *q2_array = (PyArrayObject *)PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                                           NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyArrayObject *q3_array = (PyArrayObject *)PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                                           NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (q1_array == NULL || q2_array == NULL || q3_array == NULL) {
        Py_XDECREF(q1_array);
        Py_XDECREF(q2_array);
        Py_XDECREF(q3_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(d_array);
        free(dwork);
        return NULL;
    }

    f64 *q1 = (f64*)PyArray_DATA(q1_array);
    f64 *q2 = (f64*)PyArray_DATA(q2_array);
    f64 *q3 = (f64*)PyArray_DATA(q3_array);
    memset(q1, 0, PyArray_NBYTES(q1_array));
    memset(q2, 0, PyArray_NBYTES(q2_array));
    memset(q3, 0, PyArray_NBYTES(q3_array));

    mb03cd(uplo, &loc_n1, &loc_n2, prec, a_data, lda, b_data, ldb, d_data, ldd,
           q1, ldq1, q2, ldq2, q3, ldq3, dwork, ldwork, &info);

    if (dwork != NULL) {
        free(dwork);
    }

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    PyObject *result = Py_BuildValue("OOOOOOiii",
                                     a_array, b_array, d_array,
                                     q1_array, q2_array, q3_array,
                                     (int)loc_n1, (int)loc_n2, (int)info);
 
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(d_array);
    Py_DECREF(q1_array);
    Py_DECREF(q2_array);
    Py_DECREF(q3_array);
 
    return result;
}

PyObject *py_mb03cz(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *a_obj, *b_obj, *d_obj;
    if (!PyArg_ParseTuple(args, "OOO", &a_obj, &b_obj, &d_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_CDOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_CDOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *d_array = (PyArrayObject *)PyArray_FROM_OTF(
        d_obj, NPY_CDOUBLE, NPY_ARRAY_IN_FARRAY);

    if (a_array == NULL || b_array == NULL || d_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(d_array);
        return NULL;
    }

    if (PyArray_NDIM(a_array) != 2 || PyArray_NDIM(b_array) != 2 ||
        PyArray_NDIM(d_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "A, B, D must be 2D arrays");
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(d_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *d_dims = PyArray_DIMS(d_array);

    if (a_dims[0] < 2 || a_dims[1] < 2 ||
        b_dims[0] < 2 || b_dims[1] < 2 ||
        d_dims[0] < 2 || d_dims[1] < 2) {
        PyErr_SetString(PyExc_ValueError, "A, B, D must be at least 2x2");
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(d_array);
        return NULL;
    }

    i32 lda = (i32)a_dims[0];
    i32 ldb = (i32)b_dims[0];
    i32 ldd = (i32)d_dims[0];

    c128 *a_data = (c128 *)PyArray_DATA(a_array);
    c128 *b_data = (c128 *)PyArray_DATA(b_array);
    c128 *d_data = (c128 *)PyArray_DATA(d_array);

    f64 co1, co2, co3;
    c128 si1, si2, si3;

    mb03cz(a_data, lda, b_data, ldb, d_data, ldd,
           &co1, &si1, &co2, &si2, &co3, &si3);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(d_array);

    return Py_BuildValue("dDdDdD", co1, &si1, co2, &si2, co3, &si3);
}

/* Python wrapper for mb03dd */
PyObject* py_mb03dd(PyObject* self, PyObject* args) {
    const char *uplo;
    i32 n1, n2;
    f64 prec;
    PyObject *a_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "siidOO", &uplo, &n1, &n2, &prec, &a_obj, &b_obj)) {
        return NULL;
    }

    i32 m = n1 + n2;
    if (m < 2 || m > 4) {
        PyErr_SetString(PyExc_ValueError, "N1 + N2 must be 2, 3, or 4");
        return NULL;
    }
    if (n1 < 0 || n1 > 2) {
        PyErr_SetString(PyExc_ValueError, "N1 must be 0, 1, or 2");
        return NULL;
    }
    if (n2 < 0 || n2 > 2) {
        PyErr_SetString(PyExc_ValueError, "N2 must be 0, 1, or 2");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || b_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);

    if (a_dims[0] < m || a_dims[1] < m || b_dims[0] < m || b_dims[1] < m) {
        PyErr_SetString(PyExc_ValueError, "A and B must be at least (N1+N2) x (N1+N2)");
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    i32 lda = (i32)a_dims[0];
    i32 ldb = (i32)b_dims[0];

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    npy_intp q_dims[2] = {m, m};
    PyObject *q1_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE,
                                     NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *q2_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE,
                                     NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (q1_array == NULL || q2_array == NULL) {
        Py_XDECREF(q1_array);
        Py_XDECREF(q2_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    f64 *q1_data = (f64*)PyArray_DATA((PyArrayObject*)q1_array);
    f64 *q2_data = (f64*)PyArray_DATA((PyArrayObject*)q2_array);
    memset(q1_data, 0, PyArray_NBYTES((PyArrayObject*)q1_array));
    memset(q2_data, 0, PyArray_NBYTES((PyArrayObject*)q2_array));

    i32 ldwork;
    if (*uplo == 'U' || *uplo == 'u') {
        ldwork = 16 * n1 + 10 * n2 + 23;
    } else if (*uplo == 'T' || *uplo == 't') {
        ldwork = 7 * n1 + 7 * n2 + 16;
    } else {
        ldwork = 10 * n1 + 16 * n2 + 23;
    }
    if (m == 2) ldwork = 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (dwork == NULL) {
        Py_DECREF(q1_array);
        Py_DECREF(q2_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 ldq1 = m, ldq2 = m;
    i32 n1_out = n1, n2_out = n2;
    i32 info = 0;

    mb03dd(uplo, &n1_out, &n2_out, prec, a_data, lda, b_data, ldb,
           q1_data, ldq1, q2_data, ldq2, dwork, ldwork, &info);

    free(dwork);

    PyObject *result = Py_BuildValue("OOOOiii",
        a_array, b_array, q1_array, q2_array, n1_out, n2_out, info);
 
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(q1_array);
    Py_DECREF(q2_array);
 
    return result;
}

/* Python wrapper for mb03dz */
PyObject* py_mb03dz(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);

    if (a_array == NULL || b_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);

    if (a_dims[0] < 2 || a_dims[1] < 2 ||
        b_dims[0] < 2 || b_dims[1] < 2) {
        PyErr_SetString(PyExc_ValueError, "A and B must be at least 2x2");
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    i32 lda = (i32)a_dims[0];
    i32 ldb = (i32)b_dims[0];

    c128 *a_data = (c128 *)PyArray_DATA(a_array);
    c128 *b_data = (c128 *)PyArray_DATA(b_array);

    f64 co1, co2;
    c128 si1, si2;

    mb03dz(a_data, lda, b_data, ldb, &co1, &si1, &co2, &si2);

    Py_DECREF(a_array);
    Py_DECREF(b_array);

    return Py_BuildValue("dDdD", co1, &si1, co2, &si2);
}

/* Python wrapper for mb03ed */
PyObject* py_mb03ed(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj, *d_obj;
    int n_int;
    double prec;

    if (!PyArg_ParseTuple(args, "idOOO", &n_int, &prec, &a_obj, &b_obj, &d_obj)) {
        return NULL;
    }

    i32 n = (i32)n_int;

    if (n != 2 && n != 4) {
        PyErr_SetString(PyExc_ValueError, "N must be 2 or 4");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_ENSURECOPY);

    if (a_array == NULL || b_array == NULL || d_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(d_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *d_dims = PyArray_DIMS(d_array);

    if (a_dims[0] < n || a_dims[1] < n) {
        PyErr_SetString(PyExc_ValueError, "A dimensions insufficient");
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(d_array);
        return NULL;
    }
    if (b_dims[0] < n || b_dims[1] < n) {
        PyErr_SetString(PyExc_ValueError, "B dimensions insufficient");
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(d_array);
        return NULL;
    }
    if (d_dims[0] < n || d_dims[1] < n) {
        PyErr_SetString(PyExc_ValueError, "D dimensions insufficient");
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(d_array);
        return NULL;
    }

    i32 lda = (i32)a_dims[0];
    i32 ldb = (i32)b_dims[0];
    i32 ldd = (i32)d_dims[0];

    f64 *a_data = (f64 *)PyArray_DATA(a_array);
    f64 *b_data = (f64 *)PyArray_DATA(b_array);
    f64 *d_data = (f64 *)PyArray_DATA(d_array);

    npy_intp q_dims[2] = {n, n};
    PyObject *q1_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE,
                                     NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *q2_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE,
                                     NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *q3_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE,
                                     NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (q1_array == NULL || q2_array == NULL || q3_array == NULL) {
        Py_XDECREF(q1_array);
        Py_XDECREF(q2_array);
        Py_XDECREF(q3_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(d_array);
        return NULL;
    }

    f64 *q1_data = (f64*)PyArray_DATA((PyArrayObject*)q1_array);
    f64 *q2_data = (f64*)PyArray_DATA((PyArrayObject*)q2_array);
    f64 *q3_data = (f64*)PyArray_DATA((PyArrayObject*)q3_array);
    memset(q1_data, 0, PyArray_NBYTES((PyArrayObject*)q1_array));
    memset(q2_data, 0, PyArray_NBYTES((PyArrayObject*)q2_array));
    memset(q3_data, 0, PyArray_NBYTES((PyArrayObject*)q3_array));

    i32 ldwork = (n == 4) ? 200 : 1;
    f64 *dwork = (f64 *)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        free(q1_data);
        free(q2_data);
        free(q3_data);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(d_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info = 0;

    mb03ed(n, prec, a_data, lda, b_data, ldb,
           d_data, ldd, q1_data, n, q2_data, n, q3_data, n,
           dwork, ldwork, &info);

    free(dwork);

    PyObject *result = Py_BuildValue("OOOOi", d_array, q1_array, q2_array, q3_array, info);
 
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(d_array);
    Py_DECREF(q1_array);
    Py_DECREF(q2_array);
    Py_DECREF(q3_array);
 
    return result;

    // Duplicate removed
}

/* Python wrapper for mb03gz */
PyObject *py_mb03gz(PyObject *self, PyObject *args) {
    (void)self;

    Py_complex z11_py, z12_py, z22_py, h11_py, h12_py;
    if (!PyArg_ParseTuple(args, "DDDDD", &z11_py, &z12_py, &z22_py, &h11_py, &h12_py)) {
        return NULL;
    }

    c128 z11 = z11_py.real + z11_py.imag * I;
    c128 z12 = z12_py.real + z12_py.imag * I;
    c128 z22 = z22_py.real + z22_py.imag * I;
    c128 h11 = h11_py.real + h11_py.imag * I;
    c128 h12 = h12_py.real + h12_py.imag * I;

    f64 co1, co2;
    c128 si1, si2;

    mb03gz(z11, z12, z22, h11, h12, &co1, &si1, &co2, &si2);

    return Py_BuildValue("dDdD", co1, &si1, co2, &si2);
}

/* Python wrapper for mb03gd */
PyObject *py_mb03gd(PyObject *self, PyObject *args) {
    (void)self;

    int n_py;
    PyObject *b_obj, *d_obj, *macpar_obj;
    if (!PyArg_ParseTuple(args, "iOOO", &n_py, &b_obj, &d_obj, &macpar_obj)) {
        return NULL;
    }

    i32 n = (i32)n_py;

    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *d_array = (PyArrayObject *)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *macpar_array = (PyArrayObject *)PyArray_FROM_OTF(
        macpar_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!b_array || !d_array || !macpar_array) {
        Py_XDECREF(b_array);
        Py_XDECREF(d_array);
        Py_XDECREF(macpar_array);
        return NULL;
    }

    f64 *b = (f64 *)PyArray_DATA(b_array);
    f64 *d = (f64 *)PyArray_DATA(d_array);
    f64 *macpar = (f64 *)PyArray_DATA(macpar_array);

    i32 ldb = (i32)PyArray_DIM(b_array, 0);
    i32 ldd = (n > 0) ? (i32)PyArray_DIM(d_array, 0) : 1;
    
    npy_intp dims[2] = {n, n};
    PyObject *q_array = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                     NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *u_array = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                     NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    
    if (q_array == NULL || u_array == NULL) {
        Py_XDECREF(q_array);
        Py_XDECREF(u_array);
        Py_DECREF(b_array);
        Py_DECREF(d_array);
        Py_DECREF(macpar_array);
        return NULL;
    }

    f64 *q = (f64*)PyArray_DATA((PyArrayObject*)q_array);
    f64 *u = (f64*)PyArray_DATA((PyArrayObject*)u_array);
    memset(q, 0, PyArray_NBYTES((PyArrayObject*)q_array));
    memset(u, 0, PyArray_NBYTES((PyArrayObject*)u_array));
    
    i32 ldwork = (n == 4) ? 12 : 1;
    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));

    if (!dwork) {
        Py_DECREF(q_array);
        Py_DECREF(u_array);
        Py_DECREF(b_array);
        Py_DECREF(d_array);
        Py_DECREF(macpar_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info = 0;

    mb03gd(n, b, ldb, d, ldd, macpar, q, n, u, n, dwork, ldwork, &info);

    free(dwork);

    // Arrays already created

    Py_DECREF(b_array);
    Py_DECREF(d_array);
    Py_DECREF(macpar_array);

    PyObject *result = Py_BuildValue("OOi", q_array, u_array, (int)info);
    Py_DECREF(q_array);
    Py_DECREF(u_array);

    return result;
}

/* Python wrapper for mb03hd */
PyObject *py_mb03hd(PyObject *self, PyObject *args) {
    (void)self;

    int n_py;
    PyObject *a_obj, *b_obj, *macpar_obj;
    if (!PyArg_ParseTuple(args, "iOOO", &n_py, &a_obj, &b_obj, &macpar_obj)) {
        return NULL;
    }

    i32 n = (i32)n_py;

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *macpar_array = (PyArrayObject *)PyArray_FROM_OTF(
        macpar_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!a_array || !b_array || !macpar_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(macpar_array);
        return NULL;
    }

    f64 *a = (f64 *)PyArray_DATA(a_array);
    f64 *b = (f64 *)PyArray_DATA(b_array);
    f64 *macpar = (f64 *)PyArray_DATA(macpar_array);

    i32 lda = (n > 2) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 ldb = (i32)PyArray_DIM(b_array, 0);

    npy_intp q_dims[2] = {n, n};
    PyObject *q_array_out = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE,
                                        NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!q_array_out) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(macpar_array);
        PyErr_NoMemory();
        return NULL;
    }
    f64 *q = (f64*)PyArray_DATA((PyArrayObject*)q_array_out);
    memset(q, 0, PyArray_NBYTES((PyArrayObject*)q_array_out));

    f64 dwork[24];
    i32 info = 0;

    mb03hd(n, a, lda, b, ldb, macpar, q, n, dwork, &info);

    PyObject *q_array = (PyObject*)q_array_out;
    // PyArray_ENABLEFLAGS(q_array, NPY_ARRAY_OWNDATA); // Implicitly owned

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(macpar_array);

    return Py_BuildValue("Oi", q_array, (int)info);
}

/* Python wrapper for mb03hz */
PyObject *py_mb03hz(PyObject *self, PyObject *args) {
    (void)self;

    Py_complex s11_py, s12_py, h11_py, h12_py;
    if (!PyArg_ParseTuple(args, "DDDD", &s11_py, &s12_py, &h11_py, &h12_py)) {
        return NULL;
    }

    c128 s11 = s11_py.real + s11_py.imag * I;
    c128 s12 = s12_py.real + s12_py.imag * I;
    c128 h11 = h11_py.real + h11_py.imag * I;
    c128 h12 = h12_py.real + h12_py.imag * I;

    f64 co;
    c128 si;

    mb03hz(s11, s12, h11, h12, &co, &si);

    return Py_BuildValue("dD", co, &si);
}

/* Python wrapper for mb03kc */
PyObject *py_mb03kc(PyObject *self, PyObject *args) {
    (void)self;

    i32 k, khess, n, r, lda;
    PyObject *s_obj, *a_obj;

    if (!PyArg_ParseTuple(args, "iiiiOOi", &k, &khess, &n, &r, &s_obj, &a_obj, &lda)) {
        return NULL;
    }

    PyArrayObject *s_array = (PyArrayObject *)PyArray_FROM_OTF(
        s_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!s_array) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) {
        Py_DECREF(s_array);
        return NULL;
    }

    const i32 *s = (const i32 *)PyArray_DATA(s_array);
    f64 *a = (f64 *)PyArray_DATA(a_array);

    npy_intp v_dims[1] = {2 * k};
    npy_intp tau_dims[1] = {k};
    PyObject *v_array = PyArray_SimpleNew(1, v_dims, NPY_DOUBLE);
    PyObject *tau_array = PyArray_SimpleNew(1, tau_dims, NPY_DOUBLE);
    
    if (v_array == NULL || tau_array == NULL) {
        Py_XDECREF(v_array);
        Py_XDECREF(tau_array);
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        return NULL;
    }
    f64 *v = (f64*)PyArray_DATA((PyArrayObject*)v_array);
    f64 *tau = (f64*)PyArray_DATA((PyArrayObject*)tau_array);
    memset(v, 0, PyArray_NBYTES((PyArrayObject*)v_array));
    memset(tau, 0, PyArray_NBYTES((PyArrayObject*)tau_array));

    mb03kc(k, khess, n, r, s, a, lda, v, tau);

    // Arrays already allocated and populated

    PyArrayObject *a_ret = NULL;
    if (PyArray_ResolveWritebackIfCopy(a_array) < 0) {
        Py_DECREF(v_array);
        Py_DECREF(tau_array);
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *a_orig_dims = PyArray_DIMS(a_array);
    npy_intp a_dims[3] = {a_orig_dims[0], a_orig_dims[1], a_orig_dims[2]};
    npy_intp a_strides[3] = {sizeof(f64), a_orig_dims[0] * sizeof(f64),
                             a_orig_dims[0] * a_orig_dims[1] * sizeof(f64)};

    a_ret = (PyArrayObject *)PyArray_New(
        &PyArray_Type, 3, a_dims, NPY_DOUBLE, a_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!a_ret) {
        Py_DECREF(v_array);
        Py_DECREF(tau_array);
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)a_ret), PyArray_DATA(a_array), a_dims[0] * a_dims[1] * a_dims[2] * sizeof(f64));

    Py_DECREF(s_array);
    Py_DECREF(a_array);

    return Py_BuildValue("OOO", v_array, tau_array, a_ret);
}

/* Python wrapper for mb03ke */
PyObject* py_mb03ke(PyObject* self, PyObject* args, PyObject* kwargs) {
    int trana, tranb;
    i32 isgn, k, m, n;
    i32 ldwork_in = 0;
    PyObject *s_obj, *a_obj, *b_obj, *c_obj;
    PyArrayObject *s_array, *a_array, *b_array, *c_array;

    static char *kwlist[] = {"trana", "tranb", "isgn", "k", "m", "n",
                             "s", "a", "b", "c", "ldwork", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ppiiiiOOOO|i", kwlist,
                                     &trana, &tranb, &isgn, &k, &m, &n,
                                     &s_obj, &a_obj, &b_obj, &c_obj, &ldwork_in)) {
        return NULL;
    }

    if (k < 2) {
        PyErr_SetString(PyExc_ValueError, "k must be >= 2");
        return NULL;
    }
    if (m < 1 || m > 2) {
        PyErr_SetString(PyExc_ValueError, "m must be 1 or 2");
        return NULL;
    }
    if (n < 1 || n > 2) {
        PyErr_SetString(PyExc_ValueError, "n must be 1 or 2");
        return NULL;
    }
    if (isgn != 1 && isgn != -1) {
        PyErr_SetString(PyExc_ValueError, "isgn must be 1 or -1");
        return NULL;
    }

    s_array = (PyArrayObject*)PyArray_FROM_OTF(s_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (s_array == NULL) return NULL;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) {
        Py_DECREF(s_array);
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (b_array == NULL) {
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        return NULL;
    }

    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) {
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    i32 *s = (i32*)PyArray_DATA(s_array);
    f64 *a = (f64*)PyArray_DATA(a_array);
    f64 *b = (f64*)PyArray_DATA(b_array);
    f64 *c = (f64*)PyArray_DATA(c_array);

    i32 mn = m * n;
    i32 kmn = k * mn;

    i32 minwrk = (4 * k - 3) * mn * mn + kmn;
    i32 ldwork = (ldwork_in == -1) ? -1 : ((ldwork_in > minwrk) ? ldwork_in : minwrk);

    f64 prec = DBL_EPSILON;
    f64 sfmin = DBL_MIN;
    f64 smin = sfmin / prec;

    f64 *dwork = (f64*)malloc((ldwork > 1 ? ldwork : 1) * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 scale;
    i32 info = 0;

    mb03ke((bool)trana, (bool)tranb, isgn, k, m, n, prec, smin, s, a, b, c, &scale, dwork, ldwork, &info);

    if (ldwork_in == -1) {
        free(dwork);
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return Py_BuildValue("(OOdi)", Py_None, Py_None, scale, info);
    }

    free(dwork);

    npy_intp x_dims[1] = {kmn};
    PyArrayObject *x_array = (PyArrayObject*)PyArray_SimpleNew(1, x_dims, NPY_DOUBLE);
    if (x_array == NULL) {
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }
    memcpy(PyArray_DATA(x_array), c, kmn * sizeof(f64));

    Py_DECREF(s_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    Py_DECREF(c_array);

    return Py_BuildValue("(Odi)", x_array, scale, info);
}

PyObject* py_mb03md(PyObject* self, PyObject* args) {
    i32 n;
    i32 l_in;
    f64 theta_in;
    PyObject *q_obj, *e_obj, *q2_obj, *e2_obj;
    f64 pivmin, tol, reltol;

    if (!PyArg_ParseTuple(args, "iidOOOOddd",
                          &n, &l_in, &theta_in, &q_obj, &e_obj,
                          &q2_obj, &e2_obj, &pivmin, &tol, &reltol)) {
        return NULL;
    }

    PyArrayObject *q_array = (PyArrayObject*)PyArray_FROM_OTF(
        q_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (q_array == NULL) return NULL;

    PyArrayObject *e_array = NULL;
    if (n > 1) {
        e_array = (PyArrayObject*)PyArray_FROM_OTF(
            e_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (e_array == NULL) {
            Py_DECREF(q_array);
            return NULL;
        }
    }

    PyArrayObject *q2_array = (PyArrayObject*)PyArray_FROM_OTF(
        q2_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (q2_array == NULL) {
        Py_DECREF(q_array);
        Py_XDECREF(e_array);
        return NULL;
    }

    PyArrayObject *e2_array = NULL;
    if (n > 1) {
        e2_array = (PyArrayObject*)PyArray_FROM_OTF(
            e2_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (e2_array == NULL) {
            Py_DECREF(q_array);
            Py_DECREF(e_array);
            Py_DECREF(q2_array);
            return NULL;
        }
    }

    f64 *q = (f64*)PyArray_DATA(q_array);
    f64 *e = (n > 1) ? (f64*)PyArray_DATA(e_array) : NULL;
    f64 *q2 = (f64*)PyArray_DATA(q2_array);
    f64 *e2 = (n > 1) ? (f64*)PyArray_DATA(e2_array) : NULL;

    i32 l_out = l_in;
    f64 theta_out = theta_in;
    i32 iwarn = 0;
    i32 info = 0;

    mb03md(n, &l_out, &theta_out, q, e, q2, e2, pivmin, tol, reltol, &iwarn, &info);

    Py_DECREF(q_array);
    Py_XDECREF(e_array);
    Py_DECREF(q2_array);
    Py_XDECREF(e2_array);

    return Py_BuildValue("(diii)", theta_out, l_out, iwarn, info);
}

PyObject* py_mb03qv(PyObject* self, PyObject* args)
{
    PyObject *s_obj, *t_obj;

    if (!PyArg_ParseTuple(args, "OO", &s_obj, &t_obj)) {
        return NULL;
    }

    PyArrayObject *s_array = (PyArrayObject*)PyArray_FROM_OTF(
        s_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (s_array == NULL) {
        return NULL;
    }

    PyArrayObject *t_array = (PyArrayObject*)PyArray_FROM_OTF(
        t_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (t_array == NULL) {
        Py_DECREF(s_array);
        return NULL;
    }

    int ndim_s = PyArray_NDIM(s_array);
    int ndim_t = PyArray_NDIM(t_array);
    if (ndim_s != 2 || ndim_t != 2) {
        PyErr_SetString(PyExc_ValueError, "S and T must be 2D arrays");
        Py_DECREF(s_array);
        Py_DECREF(t_array);
        return NULL;
    }

    npy_intp *dims_s = PyArray_DIMS(s_array);
    npy_intp *dims_t = PyArray_DIMS(t_array);
    if (dims_s[0] != dims_s[1] || dims_t[0] != dims_t[1]) {
        PyErr_SetString(PyExc_ValueError, "S and T must be square matrices");
        Py_DECREF(s_array);
        Py_DECREF(t_array);
        return NULL;
    }

    if (dims_s[0] != dims_t[0]) {
        PyErr_SetString(PyExc_ValueError, "S and T must have the same dimensions");
        Py_DECREF(s_array);
        Py_DECREF(t_array);
        return NULL;
    }

    i32 n = (i32)dims_s[0];
    i32 lds = (n > 0) ? (i32)PyArray_STRIDE(s_array, 1) / sizeof(f64) : 1;
    i32 ldt = (n > 0) ? (i32)PyArray_STRIDE(t_array, 1) / sizeof(f64) : 1;

    f64 *s = (f64*)PyArray_DATA(s_array);
    f64 *t = (f64*)PyArray_DATA(t_array);

    npy_intp out_dims[1] = {n};
    PyArrayObject *alphar_array = (PyArrayObject*)PyArray_EMPTY(1, out_dims, NPY_DOUBLE, 0);
    PyArrayObject *alphai_array = (PyArrayObject*)PyArray_EMPTY(1, out_dims, NPY_DOUBLE, 0);
    PyArrayObject *beta_array = (PyArrayObject*)PyArray_EMPTY(1, out_dims, NPY_DOUBLE, 0);

    if (alphar_array == NULL || alphai_array == NULL || beta_array == NULL) {
        Py_DECREF(s_array);
        Py_DECREF(t_array);
        Py_XDECREF(alphar_array);
        Py_XDECREF(alphai_array);
        Py_XDECREF(beta_array);
        return NULL;
    }

    f64 *alphar = (f64*)PyArray_DATA(alphar_array);
    f64 *alphai = (f64*)PyArray_DATA(alphai_array);
    f64 *beta = (f64*)PyArray_DATA(beta_array);

    i32 info = 0;
    mb03qv(n, s, lds, t, ldt, alphar, alphai, beta, &info);

    Py_DECREF(s_array);
    Py_DECREF(t_array);

    PyObject *result = Py_BuildValue("(OOOi)",
        (PyObject*)alphar_array, (PyObject*)alphai_array,
        (PyObject*)beta_array, info);

    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);

    return result;
}

PyObject* py_mb03qw(PyObject* self, PyObject* args)
{
    i32 n, l;
    PyObject *a_obj, *e_obj, *u_obj, *v_obj;

    if (!PyArg_ParseTuple(args, "iiOOOO", &n, &l, &a_obj, &e_obj, &u_obj, &v_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(
        e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *u_array = (PyArrayObject*)PyArray_FROM_OTF(
        u_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *v_array = (PyArrayObject*)PyArray_FROM_OTF(
        v_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || u_array == NULL || v_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(u_array);
        Py_XDECREF(v_array);
        return NULL;
    }

    i32 lda = (n > 0) ? (i32)(PyArray_STRIDE(a_array, 1) / sizeof(f64)) : 1;
    i32 lde = (n > 0) ? (i32)(PyArray_STRIDE(e_array, 1) / sizeof(f64)) : 1;
    i32 ldu = (n > 0) ? (i32)(PyArray_STRIDE(u_array, 1) / sizeof(f64)) : 1;
    i32 ldv = (n > 0) ? (i32)(PyArray_STRIDE(v_array, 1) / sizeof(f64)) : 1;

    f64 *a = (f64*)PyArray_DATA(a_array);
    f64 *e = (f64*)PyArray_DATA(e_array);
    f64 *u = (f64*)PyArray_DATA(u_array);
    f64 *v = (f64*)PyArray_DATA(v_array);

    npy_intp eig_dims[1] = {2};
    PyArrayObject *alphar_array = (PyArrayObject*)PyArray_EMPTY(1, eig_dims, NPY_DOUBLE, 0);
    PyArrayObject *alphai_array = (PyArrayObject*)PyArray_EMPTY(1, eig_dims, NPY_DOUBLE, 0);
    PyArrayObject *beta_array = (PyArrayObject*)PyArray_EMPTY(1, eig_dims, NPY_DOUBLE, 0);

    if (alphar_array == NULL || alphai_array == NULL || beta_array == NULL) {
        PyArray_DiscardWritebackIfCopy(a_array);
        PyArray_DiscardWritebackIfCopy(e_array);
        PyArray_DiscardWritebackIfCopy(u_array);
        PyArray_DiscardWritebackIfCopy(v_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(u_array);
        Py_DECREF(v_array);
        Py_XDECREF(alphar_array);
        Py_XDECREF(alphai_array);
        Py_XDECREF(beta_array);
        return NULL;
    }

    f64 *alphar = (f64*)PyArray_DATA(alphar_array);
    f64 *alphai = (f64*)PyArray_DATA(alphai_array);
    f64 *beta = (f64*)PyArray_DATA(beta_array);

    i32 info = 0;
    mb03qw(n, l, a, lda, e, lde, u, ldu, v, ldv, alphar, alphai, beta, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(u_array);
    PyArray_ResolveWritebackIfCopy(v_array);

    PyObject *result = Py_BuildValue("(OOOOOOOi)",
        (PyObject*)a_array, (PyObject*)e_array,
        (PyObject*)u_array, (PyObject*)v_array,
        (PyObject*)alphar_array, (PyObject*)alphai_array,
        (PyObject*)beta_array, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(u_array);
    Py_DECREF(v_array);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);

    return result;
}

PyObject *py_mb03iz(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;

    const char *compq, *compu;
    int n;
    PyObject *a_obj, *c_obj, *d_obj, *b_obj, *f_obj;
    double tol;
    PyObject *q_obj = Py_None, *u1_obj = Py_None, *u2_obj = Py_None;

    static char *kwlist[] = {"compq", "compu", "n", "a", "c", "d", "b", "f",
                             "tol", "q", "u1", "u2", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssiOOOOOd|OOO", kwlist,
                                     &compq, &compu, &n, &a_obj, &c_obj,
                                     &d_obj, &b_obj, &f_obj, &tol,
                                     &q_obj, &u1_obj, &u2_obj)) {
        return NULL;
    }

    i32 m = n / 2;
    int lcmpq = (compq[0] == 'I' || compq[0] == 'i' ||
                 compq[0] == 'U' || compq[0] == 'u');
    int lcmpu = (compu[0] == 'I' || compu[0] == 'i' ||
                 compu[0] == 'U' || compu[0] == 'u');

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *c_array = (PyArrayObject *)PyArray_FROM_OTF(
        c_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *d_array = (PyArrayObject *)PyArray_FROM_OTF(
        d_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *f_array = (PyArrayObject *)PyArray_FROM_OTF(
        f_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || c_array == NULL || d_array == NULL ||
        b_array == NULL || f_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        Py_XDECREF(b_array);
        Py_XDECREF(f_array);
        return NULL;
    }

    i32 lda = (m > 0) ? (i32)(PyArray_STRIDE(a_array, 1) / sizeof(c128)) : 1;
    i32 ldc = (m > 0) ? (i32)(PyArray_STRIDE(c_array, 1) / sizeof(c128)) : 1;
    i32 ldd = (m > 0) ? (i32)(PyArray_STRIDE(d_array, 1) / sizeof(c128)) : 1;
    i32 ldb = (m > 0) ? (i32)(PyArray_STRIDE(b_array, 1) / sizeof(c128)) : 1;
    i32 ldf = (m > 0) ? (i32)(PyArray_STRIDE(f_array, 1) / sizeof(c128)) : 1;

    c128 *a = (c128 *)PyArray_DATA(a_array);
    c128 *c_data = (c128 *)PyArray_DATA(c_array);
    c128 *d = (c128 *)PyArray_DATA(d_array);
    c128 *b = (c128 *)PyArray_DATA(b_array);
    c128 *f = (c128 *)PyArray_DATA(f_array);

    PyArrayObject *q_array = NULL;
    c128 *q = NULL;
    i32 ldq = 1;

    if (lcmpq) {
        if (q_obj != Py_None && q_obj != NULL) {
            q_array = (PyArrayObject *)PyArray_FROM_OTF(
                q_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        } else {
            npy_intp q_dims[2] = {n > 0 ? n : 1, n > 0 ? n : 1};
            q_array = (PyArrayObject *)PyArray_ZEROS(2, q_dims, NPY_COMPLEX128, 1);
        }
        if (q_array == NULL) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(c_array);
            PyArray_DiscardWritebackIfCopy(d_array);
            PyArray_DiscardWritebackIfCopy(b_array);
            PyArray_DiscardWritebackIfCopy(f_array);
            Py_DECREF(a_array);
            Py_DECREF(c_array);
            Py_DECREF(d_array);
            Py_DECREF(b_array);
            Py_DECREF(f_array);
            return NULL;
        }
        q = (c128 *)PyArray_DATA(q_array);
        ldq = (n > 0) ? (i32)(PyArray_STRIDE(q_array, 1) / sizeof(c128)) : 1;
    } else {
        npy_intp q_dims[2] = {1, 1};
        q_array = (PyArrayObject *)PyArray_ZEROS(2, q_dims, NPY_COMPLEX128, 1);
        if (q_array == NULL) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(c_array);
            PyArray_DiscardWritebackIfCopy(d_array);
            PyArray_DiscardWritebackIfCopy(b_array);
            PyArray_DiscardWritebackIfCopy(f_array);
            Py_DECREF(a_array);
            Py_DECREF(c_array);
            Py_DECREF(d_array);
            Py_DECREF(b_array);
            Py_DECREF(f_array);
            return NULL;
        }
        q = (c128 *)PyArray_DATA(q_array);
    }

    PyArrayObject *u1_array = NULL;
    PyArrayObject *u2_array = NULL;
    c128 *u1 = NULL;
    c128 *u2 = NULL;
    i32 ldu1 = 1, ldu2 = 1;

    if (lcmpu) {
        if (u1_obj != Py_None && u1_obj != NULL) {
            u1_array = (PyArrayObject *)PyArray_FROM_OTF(
                u1_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        } else {
            npy_intp u1_dims[2] = {m > 0 ? m : 1, m > 0 ? m : 1};
            u1_array = (PyArrayObject *)PyArray_ZEROS(2, u1_dims, NPY_COMPLEX128, 1);
        }
        if (u2_obj != Py_None && u2_obj != NULL) {
            u2_array = (PyArrayObject *)PyArray_FROM_OTF(
                u2_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        } else {
            npy_intp u2_dims[2] = {m > 0 ? m : 1, m > 0 ? m : 1};
            u2_array = (PyArrayObject *)PyArray_ZEROS(2, u2_dims, NPY_COMPLEX128, 1);
        }
        if (u1_array == NULL || u2_array == NULL) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(c_array);
            PyArray_DiscardWritebackIfCopy(d_array);
            PyArray_DiscardWritebackIfCopy(b_array);
            PyArray_DiscardWritebackIfCopy(f_array);
            Py_DECREF(a_array);
            Py_DECREF(c_array);
            Py_DECREF(d_array);
            Py_DECREF(b_array);
            Py_DECREF(f_array);
            Py_DECREF(q_array);
            Py_XDECREF(u1_array);
            Py_XDECREF(u2_array);
            return NULL;
        }
        u1 = (c128 *)PyArray_DATA(u1_array);
        u2 = (c128 *)PyArray_DATA(u2_array);
        ldu1 = (m > 0) ? (i32)(PyArray_STRIDE(u1_array, 1) / sizeof(c128)) : 1;
        ldu2 = (m > 0) ? (i32)(PyArray_STRIDE(u2_array, 1) / sizeof(c128)) : 1;
    } else {
        npy_intp u_dims[2] = {1, 1};
        u1_array = (PyArrayObject *)PyArray_ZEROS(2, u_dims, NPY_COMPLEX128, 1);
        u2_array = (PyArrayObject *)PyArray_ZEROS(2, u_dims, NPY_COMPLEX128, 1);
        if (u1_array == NULL || u2_array == NULL) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(c_array);
            PyArray_DiscardWritebackIfCopy(d_array);
            PyArray_DiscardWritebackIfCopy(b_array);
            PyArray_DiscardWritebackIfCopy(f_array);
            Py_DECREF(a_array);
            Py_DECREF(c_array);
            Py_DECREF(d_array);
            Py_DECREF(b_array);
            Py_DECREF(f_array);
            Py_DECREF(q_array);
            Py_XDECREF(u1_array);
            Py_XDECREF(u2_array);
            return NULL;
        }
        u1 = (c128 *)PyArray_DATA(u1_array);
        u2 = (c128 *)PyArray_DATA(u2_array);
    }

    i32 neig = 0;
    i32 info = 0;

    mb03iz(compq, compu, n, a, lda, c_data, ldc, d, ldd, b, ldb, f, ldf,
           q, ldq, u1, ldu1, u2, ldu2, &neig, tol, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(f_array);
    if (lcmpq && q_obj != Py_None && q_obj != NULL) {
        PyArray_ResolveWritebackIfCopy(q_array);
    }
    if (lcmpu) {
        if (u1_obj != Py_None && u1_obj != NULL) {
            PyArray_ResolveWritebackIfCopy(u1_array);
        }
        if (u2_obj != Py_None && u2_obj != NULL) {
            PyArray_ResolveWritebackIfCopy(u2_array);
        }
    }

    PyObject *result = Py_BuildValue("(OOOOOOOOii)",
        (PyObject *)a_array, (PyObject *)c_array, (PyObject *)d_array,
        (PyObject *)b_array, (PyObject *)f_array, (PyObject *)q_array,
        (PyObject *)u1_array, (PyObject *)u2_array, (int)neig, (int)info);

    Py_DECREF(a_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(b_array);
    Py_DECREF(f_array);
    Py_DECREF(q_array);
    Py_DECREF(u1_array);
    Py_DECREF(u2_array);

    return result;
}

PyObject* py_mb03jd(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"compq", "n", "a", "d", "b", "f", "q", NULL};

    const char *compq;
    int n;
    PyObject *a_obj, *d_obj, *b_obj, *f_obj;
    PyObject *q_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siOOOO|O", kwlist,
                                     &compq, &n, &a_obj, &d_obj, &b_obj, &f_obj, &q_obj)) {
        return NULL;
    }

    bool liniq = (compq[0] == 'I' || compq[0] == 'i');
    bool lupdq = (compq[0] == 'U' || compq[0] == 'u');
    bool lcmpq = liniq || lupdq;

    i32 m = n / 2;

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *d_array = (PyArrayObject *)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(d_array);
        return NULL;
    }

    PyArrayObject *f_array = (PyArrayObject *)PyArray_FROM_OTF(
        f_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (f_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(d_array);
        Py_DECREF(b_array);
        return NULL;
    }

    f64 *a = (f64 *)PyArray_DATA(a_array);
    f64 *d = (f64 *)PyArray_DATA(d_array);
    f64 *b = (f64 *)PyArray_DATA(b_array);
    f64 *f = (f64 *)PyArray_DATA(f_array);

    i32 lda = (m > 0) ? (i32)(PyArray_STRIDE(a_array, 1) / sizeof(f64)) : 1;
    i32 ldd = (m > 0) ? (i32)(PyArray_STRIDE(d_array, 1) / sizeof(f64)) : 1;
    i32 ldb = (m > 0) ? (i32)(PyArray_STRIDE(b_array, 1) / sizeof(f64)) : 1;
    i32 ldf = (m > 0) ? (i32)(PyArray_STRIDE(f_array, 1) / sizeof(f64)) : 1;

    PyArrayObject *q_array = NULL;
    f64 *q = NULL;
    i32 ldq = 1;

    if (lcmpq) {
        if (q_obj != Py_None && q_obj != NULL && lupdq) {
            q_array = (PyArrayObject *)PyArray_FROM_OTF(
                q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        } else {
            npy_intp q_dims[2] = {n > 0 ? n : 1, n > 0 ? n : 1};
            q_array = (PyArrayObject *)PyArray_ZEROS(2, q_dims, NPY_DOUBLE, 1);
        }
        if (q_array == NULL) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(d_array);
            PyArray_DiscardWritebackIfCopy(b_array);
            PyArray_DiscardWritebackIfCopy(f_array);
            Py_DECREF(a_array);
            Py_DECREF(d_array);
            Py_DECREF(b_array);
            Py_DECREF(f_array);
            return NULL;
        }
        q = (f64 *)PyArray_DATA(q_array);
        ldq = (n > 0) ? (i32)(PyArray_STRIDE(q_array, 1) / sizeof(f64)) : 1;
    } else {
        npy_intp q_dims[2] = {1, 1};
        q_array = (PyArrayObject *)PyArray_ZEROS(2, q_dims, NPY_DOUBLE, 1);
        if (q_array == NULL) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(d_array);
            PyArray_DiscardWritebackIfCopy(b_array);
            PyArray_DiscardWritebackIfCopy(f_array);
            Py_DECREF(a_array);
            Py_DECREF(d_array);
            Py_DECREF(b_array);
            Py_DECREF(f_array);
            return NULL;
        }
        q = (f64 *)PyArray_DATA(q_array);
    }

    i32 liwork = n + 1;
    i32 *iwork = (i32 *)malloc(liwork * sizeof(i32));
    if (iwork == NULL) {
        PyArray_DiscardWritebackIfCopy(a_array);
        PyArray_DiscardWritebackIfCopy(d_array);
        PyArray_DiscardWritebackIfCopy(b_array);
        PyArray_DiscardWritebackIfCopy(f_array);
        Py_DECREF(a_array);
        Py_DECREF(d_array);
        Py_DECREF(b_array);
        Py_DECREF(f_array);
        Py_DECREF(q_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 ldwork;
    if (lcmpq) {
        ldwork = (4 * n + 32 > 108) ? 4 * n + 32 : 108;
    } else {
        ldwork = (2 * n + 32 > 108) ? 2 * n + 32 : 108;
    }
    f64 *dwork = (f64 *)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        free(iwork);
        PyArray_DiscardWritebackIfCopy(a_array);
        PyArray_DiscardWritebackIfCopy(d_array);
        PyArray_DiscardWritebackIfCopy(b_array);
        PyArray_DiscardWritebackIfCopy(f_array);
        Py_DECREF(a_array);
        Py_DECREF(d_array);
        Py_DECREF(b_array);
        Py_DECREF(f_array);
        Py_DECREF(q_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 neig = 0;
    i32 info = 0;

    mb03jd(compq, n, a, lda, d, ldd, b, ldb, f, ldf, q, ldq,
           &neig, iwork, liwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(f_array);
    if (lcmpq && q_obj != Py_None && q_obj != NULL && lupdq) {
        PyArray_ResolveWritebackIfCopy(q_array);
    }

    PyObject *result = Py_BuildValue("(OOOOOii)",
        (PyObject *)a_array, (PyObject *)d_array, (PyObject *)b_array,
        (PyObject *)f_array, (PyObject *)q_array, (int)neig, (int)info);

    Py_DECREF(a_array);
    Py_DECREF(d_array);
    Py_DECREF(b_array);
    Py_DECREF(f_array);
    Py_DECREF(q_array);

    return result;
}

PyObject* py_mb03jp(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"compq", "n", "a", "d", "b", "f", "q", NULL};

    const char *compq;
    int n;
    PyObject *a_obj, *d_obj, *b_obj, *f_obj;
    PyObject *q_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siOOOO|O", kwlist,
                                     &compq, &n, &a_obj, &d_obj, &b_obj, &f_obj, &q_obj)) {
        return NULL;
    }

    bool liniq = (compq[0] == 'I' || compq[0] == 'i');
    bool lupdq = (compq[0] == 'U' || compq[0] == 'u');
    bool lcmpq = liniq || lupdq;

    i32 m = n / 2;

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *d_array = (PyArrayObject *)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(d_array);
        return NULL;
    }

    PyArrayObject *f_array = (PyArrayObject *)PyArray_FROM_OTF(
        f_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (f_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(d_array);
        Py_DECREF(b_array);
        return NULL;
    }

    f64 *a = (f64 *)PyArray_DATA(a_array);
    f64 *d = (f64 *)PyArray_DATA(d_array);
    f64 *b = (f64 *)PyArray_DATA(b_array);
    f64 *f = (f64 *)PyArray_DATA(f_array);

    i32 lda = (m > 0) ? (i32)(PyArray_STRIDE(a_array, 1) / sizeof(f64)) : 1;
    i32 ldd = (m > 0) ? (i32)(PyArray_STRIDE(d_array, 1) / sizeof(f64)) : 1;
    i32 ldb = (m > 0) ? (i32)(PyArray_STRIDE(b_array, 1) / sizeof(f64)) : 1;
    i32 ldf = (m > 0) ? (i32)(PyArray_STRIDE(f_array, 1) / sizeof(f64)) : 1;

    PyArrayObject *q_array = NULL;
    f64 *q = NULL;
    i32 ldq = 1;

    if (lcmpq) {
        if (q_obj != Py_None && q_obj != NULL && lupdq) {
            q_array = (PyArrayObject *)PyArray_FROM_OTF(
                q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        } else {
            npy_intp q_dims[2] = {n > 0 ? n : 1, n > 0 ? n : 1};
            q_array = (PyArrayObject *)PyArray_ZEROS(2, q_dims, NPY_DOUBLE, 1);
        }
        if (q_array == NULL) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(d_array);
            PyArray_DiscardWritebackIfCopy(b_array);
            PyArray_DiscardWritebackIfCopy(f_array);
            Py_DECREF(a_array);
            Py_DECREF(d_array);
            Py_DECREF(b_array);
            Py_DECREF(f_array);
            return NULL;
        }
        q = (f64 *)PyArray_DATA(q_array);
        ldq = (n > 0) ? (i32)(PyArray_STRIDE(q_array, 1) / sizeof(f64)) : 1;
    } else {
        npy_intp q_dims[2] = {1, 1};
        q_array = (PyArrayObject *)PyArray_ZEROS(2, q_dims, NPY_DOUBLE, 1);
        if (q_array == NULL) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(d_array);
            PyArray_DiscardWritebackIfCopy(b_array);
            PyArray_DiscardWritebackIfCopy(f_array);
            Py_DECREF(a_array);
            Py_DECREF(d_array);
            Py_DECREF(b_array);
            Py_DECREF(f_array);
            return NULL;
        }
        q = (f64 *)PyArray_DATA(q_array);
    }

    i32 liwork = (3 * n - 3 > 1) ? 3 * n - 3 : 1;
    i32 *iwork = (i32 *)malloc(liwork * sizeof(i32));
    if (iwork == NULL) {
        PyArray_DiscardWritebackIfCopy(a_array);
        PyArray_DiscardWritebackIfCopy(d_array);
        PyArray_DiscardWritebackIfCopy(b_array);
        PyArray_DiscardWritebackIfCopy(f_array);
        Py_DECREF(a_array);
        Py_DECREF(d_array);
        Py_DECREF(b_array);
        Py_DECREF(f_array);
        Py_DECREF(q_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 ldwork;
    if (lcmpq) {
        ldwork = ((4 * n + 32 > 108) ? 4 * n + 32 : 108) + 5 * m;
    } else {
        ldwork = ((2 * n + 32 > 108) ? 2 * n + 32 : 108) + 5 * m;
    }
    f64 *dwork = (f64 *)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        free(iwork);
        PyArray_DiscardWritebackIfCopy(a_array);
        PyArray_DiscardWritebackIfCopy(d_array);
        PyArray_DiscardWritebackIfCopy(b_array);
        PyArray_DiscardWritebackIfCopy(f_array);
        Py_DECREF(a_array);
        Py_DECREF(d_array);
        Py_DECREF(b_array);
        Py_DECREF(f_array);
        Py_DECREF(q_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 neig = 0;
    i32 info = 0;

    mb03jp(compq, n, a, lda, d, ldd, b, ldb, f, ldf, q, ldq,
           &neig, iwork, liwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(f_array);
    if (lcmpq && q_obj != Py_None && q_obj != NULL && lupdq) {
        PyArray_ResolveWritebackIfCopy(q_array);
    }

    PyObject *result = Py_BuildValue("(OOOOOii)",
        (PyObject *)a_array, (PyObject *)d_array, (PyObject *)b_array,
        (PyObject *)f_array, (PyObject *)q_array, (int)neig, (int)info);

    Py_DECREF(a_array);
    Py_DECREF(d_array);
    Py_DECREF(b_array);
    Py_DECREF(f_array);
    Py_DECREF(q_array);

    return result;
}

PyObject *py_mb03jz(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;

    const char *compq;
    int n;
    PyObject *a_obj, *d_obj, *b_obj, *f_obj;
    double tol = -1.0;
    PyObject *q_obj = Py_None;

    static char *kwlist[] = {"compq", "n", "a", "d", "b", "f", "q", "tol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siOOOO|Od", kwlist,
                                     &compq, &n, &a_obj, &d_obj,
                                     &b_obj, &f_obj, &q_obj, &tol)) {
        return NULL;
    }

    i32 m = n / 2;
    int liniq = (compq[0] == 'I' || compq[0] == 'i');
    int lupdq = (compq[0] == 'U' || compq[0] == 'u');
    int lcmpq = liniq || lupdq;

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *d_array = (PyArrayObject *)PyArray_FROM_OTF(
        d_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *f_array = (PyArrayObject *)PyArray_FROM_OTF(
        f_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || d_array == NULL ||
        b_array == NULL || f_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(d_array);
        Py_XDECREF(b_array);
        Py_XDECREF(f_array);
        return NULL;
    }

    i32 lda = (m > 0) ? (i32)(PyArray_STRIDE(a_array, 1) / sizeof(c128)) : 1;
    i32 ldd = (m > 0) ? (i32)(PyArray_STRIDE(d_array, 1) / sizeof(c128)) : 1;
    i32 ldb = (m > 0) ? (i32)(PyArray_STRIDE(b_array, 1) / sizeof(c128)) : 1;
    i32 ldf = (m > 0) ? (i32)(PyArray_STRIDE(f_array, 1) / sizeof(c128)) : 1;

    c128 *a = (c128 *)PyArray_DATA(a_array);
    c128 *d = (c128 *)PyArray_DATA(d_array);
    c128 *b = (c128 *)PyArray_DATA(b_array);
    c128 *f = (c128 *)PyArray_DATA(f_array);

    PyArrayObject *q_array = NULL;
    c128 *q = NULL;
    i32 ldq = 1;

    if (lcmpq) {
        if (q_obj != Py_None && q_obj != NULL && lupdq) {
            q_array = (PyArrayObject *)PyArray_FROM_OTF(
                q_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        } else {
            npy_intp q_dims[2] = {n > 0 ? n : 1, n > 0 ? n : 1};
            q_array = (PyArrayObject *)PyArray_ZEROS(2, q_dims, NPY_COMPLEX128, 1);
        }
        if (q_array == NULL) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(d_array);
            PyArray_DiscardWritebackIfCopy(b_array);
            PyArray_DiscardWritebackIfCopy(f_array);
            Py_DECREF(a_array);
            Py_DECREF(d_array);
            Py_DECREF(b_array);
            Py_DECREF(f_array);
            return NULL;
        }
        q = (c128 *)PyArray_DATA(q_array);
        ldq = (n > 0) ? (i32)(PyArray_STRIDE(q_array, 1) / sizeof(c128)) : 1;
    } else {
        npy_intp q_dims[2] = {1, 1};
        q_array = (PyArrayObject *)PyArray_ZEROS(2, q_dims, NPY_COMPLEX128, 1);
        if (q_array == NULL) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(d_array);
            PyArray_DiscardWritebackIfCopy(b_array);
            PyArray_DiscardWritebackIfCopy(f_array);
            Py_DECREF(a_array);
            Py_DECREF(d_array);
            Py_DECREF(b_array);
            Py_DECREF(f_array);
            return NULL;
        }
        q = (c128 *)PyArray_DATA(q_array);
    }

    i32 neig = 0;
    i32 info = 0;

    mb03jz(compq, n, a, lda, d, ldd, b, ldb, f, ldf, q, ldq, &neig, tol, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(f_array);
    if (lcmpq && q_obj != Py_None && q_obj != NULL && lupdq) {
        PyArray_ResolveWritebackIfCopy(q_array);
    }

    PyObject *result_jz = Py_BuildValue("(OOOOOii)",
        (PyObject *)a_array, (PyObject *)d_array, (PyObject *)b_array,
        (PyObject *)f_array, (PyObject *)q_array, (int)neig, (int)info);

    Py_DECREF(a_array);
    Py_DECREF(d_array);
    Py_DECREF(b_array);
    Py_DECREF(f_array);
    Py_DECREF(q_array);

    return result_jz;
}

/* MB03KB: Swap pairs of adjacent diagonal blocks in generalized periodic Schur form */
PyObject* py_mb03kb(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"compq", "whichq", "ws", "k", "nc", "kschur",
                             "j1", "n1", "n2", "n", "ni", "s", "t", "ldt",
                             "ixt", "q", "ldq", "ixq", "tol", "ldwork", NULL};

    const char* compq;
    PyObject *whichq_obj, *n_obj, *ni_obj, *s_obj, *t_obj, *ldt_obj, *ixt_obj;
    PyObject *q_obj, *ldq_obj, *ixq_obj, *tol_obj;
    int ws_int, k, nc, kschur, j1, n1, n2;
    int ldwork = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sOpiiiiiiOOOOOOOOOO|i", kwlist,
            &compq, &whichq_obj, &ws_int, &k, &nc, &kschur, &j1, &n1, &n2,
            &n_obj, &ni_obj, &s_obj, &t_obj, &ldt_obj, &ixt_obj,
            &q_obj, &ldq_obj, &ixq_obj, &tol_obj, &ldwork)) {
        return NULL;
    }

    bool ws = (ws_int != 0);

    PyArrayObject *whichq_arr = (PyArrayObject *)PyArray_FROM_OTF(whichq_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *n_arr = (PyArrayObject *)PyArray_FROM_OTF(n_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ni_arr = (PyArrayObject *)PyArray_FROM_OTF(ni_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *s_arr = (PyArrayObject *)PyArray_FROM_OTF(s_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ldt_arr = (PyArrayObject *)PyArray_FROM_OTF(ldt_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ixt_arr = (PyArrayObject *)PyArray_FROM_OTF(ixt_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ldq_arr = (PyArrayObject *)PyArray_FROM_OTF(ldq_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ixq_arr = (PyArrayObject *)PyArray_FROM_OTF(ixq_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *tol_arr = (PyArrayObject *)PyArray_FROM_OTF(tol_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *t_arr = (PyArrayObject *)PyArray_FROM_OTF(t_obj, NPY_DOUBLE,
        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *q_arr = (PyArrayObject *)PyArray_FROM_OTF(q_obj, NPY_DOUBLE,
        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!whichq_arr || !n_arr || !ni_arr || !s_arr || !ldt_arr || !ixt_arr ||
        !ldq_arr || !ixq_arr || !tol_arr || !t_arr || !q_arr) {
        Py_XDECREF(whichq_arr);
        Py_XDECREF(n_arr);
        Py_XDECREF(ni_arr);
        Py_XDECREF(s_arr);
        Py_XDECREF(ldt_arr);
        Py_XDECREF(ixt_arr);
        Py_XDECREF(ldq_arr);
        Py_XDECREF(ixq_arr);
        Py_XDECREF(tol_arr);
        Py_XDECREF(t_arr);
        Py_XDECREF(q_arr);
        return NULL;
    }

    i32* whichq = (i32*)PyArray_DATA(whichq_arr);
    i32* n_data = (i32*)PyArray_DATA(n_arr);
    i32* ni = (i32*)PyArray_DATA(ni_arr);
    i32* s = (i32*)PyArray_DATA(s_arr);
    i32* ldt = (i32*)PyArray_DATA(ldt_arr);
    i32* ixt = (i32*)PyArray_DATA(ixt_arr);
    i32* ldq = (i32*)PyArray_DATA(ldq_arr);
    i32* ixq = (i32*)PyArray_DATA(ixq_arr);
    f64* tol = (f64*)PyArray_DATA(tol_arr);
    f64* t = (f64*)PyArray_DATA(t_arr);
    f64* q = (f64*)PyArray_DATA(q_arr);

    i32 mn = 0;
    for (i32 i = 0; i < k; i++) {
        if (n_data[i] > mn) mn = n_data[i];
    }
    if (mn <= 10) mn = 0;

    i32 minwrk;
    if (n1 == 1 && n2 == 1) {
        minwrk = k * 10 + mn;
    } else if (n1 == 1 && n2 == 2) {
        minwrk = k * 25 + mn;
    } else if (n1 == 2 && n2 == 1) {
        i32 opt1 = k * 23 + mn;
        i32 opt2 = k * 25 - 12;
        minwrk = (opt1 > opt2) ? opt1 : opt2;
    } else if (n1 == 2 && n2 == 2) {
        i32 opt1 = k * 42 + mn;
        i32 opt2 = k * 80 - 48;
        minwrk = (opt1 > opt2) ? opt1 : opt2;
    } else {
        minwrk = 1;
    }

    i32 alloc_ldwork;
    if (ldwork == -1) {
        alloc_ldwork = 1;
    } else if (ldwork <= 0) {
        ldwork = minwrk + 100;
        alloc_ldwork = ldwork;
    } else {
        alloc_ldwork = ldwork;
    }

    i32* iwork = (i32*)malloc(4 * k * sizeof(i32));
    f64* dwork = (f64*)malloc(alloc_ldwork * sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        Py_DECREF(whichq_arr);
        Py_DECREF(n_arr);
        Py_DECREF(ni_arr);
        Py_DECREF(s_arr);
        Py_DECREF(ldt_arr);
        Py_DECREF(ixt_arr);
        Py_DECREF(ldq_arr);
        Py_DECREF(ixq_arr);
        Py_DECREF(tol_arr);
        PyArray_DiscardWritebackIfCopy(t_arr);
        PyArray_DiscardWritebackIfCopy(q_arr);
        Py_DECREF(t_arr);
        Py_DECREF(q_arr);
        return PyErr_NoMemory();
    }

    i32 info = 0;

    mb03kb(compq, whichq, ws, k, nc, kschur, j1, n1, n2,
           n_data, ni, s, t, ldt, ixt, q, ldq, ixq,
           tol, iwork, dwork, ldwork, &info);

    PyArray_ResolveWritebackIfCopy(t_arr);
    PyArray_ResolveWritebackIfCopy(q_arr);

    free(iwork);
    free(dwork);

    PyObject* result = Py_BuildValue("(OOi)", (PyObject*)t_arr, (PyObject*)q_arr, (int)info);

    Py_DECREF(whichq_arr);
    Py_DECREF(n_arr);
    Py_DECREF(ni_arr);
    Py_DECREF(s_arr);
    Py_DECREF(ldt_arr);
    Py_DECREF(ixt_arr);
    Py_DECREF(ldq_arr);
    Py_DECREF(ixq_arr);
    Py_DECREF(tol_arr);
    Py_DECREF(t_arr);
    Py_DECREF(q_arr);

    return result;
}

PyObject* py_mb03qg(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static char* kwlist[] = {"dico", "stdom", "jobu", "jobv", "a", "e",
                             "nlow", "nsup", "alpha", "u", "v", NULL};

    const char* dico;
    const char* stdom;
    const char* jobu;
    const char* jobv;
    PyObject* a_obj;
    PyObject* e_obj;
    int nlow, nsup;
    double alpha;
    PyObject* u_obj = Py_None;
    PyObject* v_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssssOOiid|OO", kwlist,
                                     &dico, &stdom, &jobu, &jobv,
                                     &a_obj, &e_obj, &nlow, &nsup, &alpha,
                                     &u_obj, &v_obj)) {
        return NULL;
    }

    if (!(dico[0] == 'C' || dico[0] == 'c' || dico[0] == 'D' || dico[0] == 'd')) {
        PyErr_SetString(PyExc_ValueError, "dico must be 'C' or 'D'");
        return NULL;
    }
    if (!(stdom[0] == 'S' || stdom[0] == 's' || stdom[0] == 'U' || stdom[0] == 'u')) {
        PyErr_SetString(PyExc_ValueError, "stdom must be 'S' or 'U'");
        return NULL;
    }
    if (!(jobu[0] == 'I' || jobu[0] == 'i' || jobu[0] == 'U' || jobu[0] == 'u')) {
        PyErr_SetString(PyExc_ValueError, "jobu must be 'I' or 'U'");
        return NULL;
    }
    if (!(jobv[0] == 'I' || jobv[0] == 'i' || jobv[0] == 'U' || jobv[0] == 'u')) {
        PyErr_SetString(PyExc_ValueError, "jobv must be 'I' or 'U'");
        return NULL;
    }

    bool discr = (dico[0] == 'D' || dico[0] == 'd');
    if (discr && alpha < 0.0) {
        PyErr_SetString(PyExc_ValueError, "alpha must be >= 0 for discrete-time");
        return NULL;
    }

    PyArrayObject* a_arr = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject* e_arr = (PyArrayObject*)PyArray_FROM_OTF(
        e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!a_arr || !e_arr) {
        Py_XDECREF(a_arr);
        Py_XDECREF(e_arr);
        return NULL;
    }

    if (PyArray_NDIM(a_arr) != 2 || PyArray_NDIM(e_arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "A and E must be 2D arrays");
        PyArray_DiscardWritebackIfCopy(a_arr);
        PyArray_DiscardWritebackIfCopy(e_arr);
        Py_DECREF(a_arr);
        Py_DECREF(e_arr);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_arr, 0);
    if (PyArray_DIM(a_arr, 1) != n || PyArray_DIM(e_arr, 0) != n || PyArray_DIM(e_arr, 1) != n) {
        PyErr_SetString(PyExc_ValueError, "A and E must be square matrices of same size");
        PyArray_DiscardWritebackIfCopy(a_arr);
        PyArray_DiscardWritebackIfCopy(e_arr);
        Py_DECREF(a_arr);
        Py_DECREF(e_arr);
        return NULL;
    }

    PyArrayObject* u_arr = NULL;
    PyArrayObject* v_arr = NULL;
    bool jobu_init = (jobu[0] == 'I' || jobu[0] == 'i');
    bool jobv_init = (jobv[0] == 'I' || jobv[0] == 'i');

    if (jobu_init) {
        npy_intp dims[2] = {n, n};
        u_arr = (PyArrayObject*)PyArray_ZEROS(2, dims, NPY_DOUBLE, 1);
    } else {
        if (u_obj == Py_None) {
            PyErr_SetString(PyExc_ValueError, "u must be provided when jobu='U'");
            PyArray_DiscardWritebackIfCopy(a_arr);
            PyArray_DiscardWritebackIfCopy(e_arr);
            Py_DECREF(a_arr);
            Py_DECREF(e_arr);
            return NULL;
        }
        u_arr = (PyArrayObject*)PyArray_FROM_OTF(
            u_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    }

    if (jobv_init) {
        npy_intp dims[2] = {n, n};
        v_arr = (PyArrayObject*)PyArray_ZEROS(2, dims, NPY_DOUBLE, 1);
    } else {
        if (v_obj == Py_None) {
            PyErr_SetString(PyExc_ValueError, "v must be provided when jobv='U'");
            PyArray_DiscardWritebackIfCopy(a_arr);
            PyArray_DiscardWritebackIfCopy(e_arr);
            Py_XDECREF(u_arr);
            Py_DECREF(a_arr);
            Py_DECREF(e_arr);
            return NULL;
        }
        v_arr = (PyArrayObject*)PyArray_FROM_OTF(
            v_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    }

    if (!u_arr || !v_arr) {
        PyArray_DiscardWritebackIfCopy(a_arr);
        PyArray_DiscardWritebackIfCopy(e_arr);
        Py_XDECREF(u_arr);
        Py_XDECREF(v_arr);
        Py_DECREF(a_arr);
        Py_DECREF(e_arr);
        return NULL;
    }

    f64* a = (f64*)PyArray_DATA(a_arr);
    f64* e = (f64*)PyArray_DATA(e_arr);
    f64* u = (f64*)PyArray_DATA(u_arr);
    f64* v = (f64*)PyArray_DATA(v_arr);

    i32 lda = n > 0 ? n : 1;
    i32 lde = n > 0 ? n : 1;
    i32 ldu = n > 0 ? n : 1;
    i32 ldv = n > 0 ? n : 1;

    i32 ldwork = (n > 1) ? (4 * n + 16) : 1;
    f64* dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (!dwork) {
        PyArray_DiscardWritebackIfCopy(a_arr);
        PyArray_DiscardWritebackIfCopy(e_arr);
        if (!jobu_init) PyArray_DiscardWritebackIfCopy(u_arr);
        if (!jobv_init) PyArray_DiscardWritebackIfCopy(v_arr);
        Py_DECREF(a_arr);
        Py_DECREF(e_arr);
        Py_DECREF(u_arr);
        Py_DECREF(v_arr);
        return PyErr_NoMemory();
    }

    i32 ndim = 0;
    i32 info = 0;

    mb03qg(dico, stdom, jobu, jobv, n, (i32)nlow, (i32)nsup, alpha,
           a, lda, e, lde, u, ldu, v, ldv, &ndim, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_arr);
    PyArray_ResolveWritebackIfCopy(e_arr);
    if (!jobu_init) PyArray_ResolveWritebackIfCopy(u_arr);
    if (!jobv_init) PyArray_ResolveWritebackIfCopy(v_arr);

    PyObject* result = Py_BuildValue("(OOOOii)",
                                     (PyObject*)a_arr, (PyObject*)e_arr,
                                     (PyObject*)u_arr, (PyObject*)v_arr,
                                     (int)ndim, (int)info);

    Py_DECREF(a_arr);
    Py_DECREF(e_arr);
    Py_DECREF(u_arr);
    Py_DECREF(v_arr);

    return result;
}

PyObject* py_mb03ka(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"compq", "whichq", "ws", "k", "nc", "kschur",
                             "ifst", "ilst", "n", "ni", "s", "t", "ldt",
                             "ixt", "q", "ldq", "ixq", "tol", "ldwork", NULL};

    const char* compq;
    PyObject *whichq_obj, *n_obj, *ni_obj, *s_obj, *t_obj, *ldt_obj, *ixt_obj;
    PyObject *q_obj, *ldq_obj, *ixq_obj, *tol_obj;
    int ws_int, k, nc, kschur, ifst, ilst;
    int ldwork = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sOpiiiiiOOOOOOOOOO|i", kwlist,
            &compq, &whichq_obj, &ws_int, &k, &nc, &kschur, &ifst, &ilst,
            &n_obj, &ni_obj, &s_obj, &t_obj, &ldt_obj, &ixt_obj,
            &q_obj, &ldq_obj, &ixq_obj, &tol_obj, &ldwork)) {
        return NULL;
    }

    bool ws = (ws_int != 0);

    PyArrayObject *whichq_arr = (PyArrayObject *)PyArray_FROM_OTF(whichq_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *n_arr = (PyArrayObject *)PyArray_FROM_OTF(n_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ni_arr = (PyArrayObject *)PyArray_FROM_OTF(ni_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *s_arr = (PyArrayObject *)PyArray_FROM_OTF(s_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ldt_arr = (PyArrayObject *)PyArray_FROM_OTF(ldt_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ixt_arr = (PyArrayObject *)PyArray_FROM_OTF(ixt_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ldq_arr = (PyArrayObject *)PyArray_FROM_OTF(ldq_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ixq_arr = (PyArrayObject *)PyArray_FROM_OTF(ixq_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *tol_arr = (PyArrayObject *)PyArray_FROM_OTF(tol_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *t_arr = (PyArrayObject *)PyArray_FROM_OTF(t_obj, NPY_DOUBLE,
        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *q_arr = (PyArrayObject *)PyArray_FROM_OTF(q_obj, NPY_DOUBLE,
        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!whichq_arr || !n_arr || !ni_arr || !s_arr || !ldt_arr || !ixt_arr ||
        !ldq_arr || !ixq_arr || !tol_arr || !t_arr || !q_arr) {
        Py_XDECREF(whichq_arr);
        Py_XDECREF(n_arr);
        Py_XDECREF(ni_arr);
        Py_XDECREF(s_arr);
        Py_XDECREF(ldt_arr);
        Py_XDECREF(ixt_arr);
        Py_XDECREF(ldq_arr);
        Py_XDECREF(ixq_arr);
        Py_XDECREF(tol_arr);
        Py_XDECREF(t_arr);
        Py_XDECREF(q_arr);
        return NULL;
    }

    i32* whichq = (i32*)PyArray_DATA(whichq_arr);
    i32* n_data = (i32*)PyArray_DATA(n_arr);
    i32* ni = (i32*)PyArray_DATA(ni_arr);
    i32* s = (i32*)PyArray_DATA(s_arr);
    i32* ldt = (i32*)PyArray_DATA(ldt_arr);
    i32* ixt = (i32*)PyArray_DATA(ixt_arr);
    i32* ldq = (i32*)PyArray_DATA(ldq_arr);
    i32* ixq = (i32*)PyArray_DATA(ixq_arr);
    f64* tol = (f64*)PyArray_DATA(tol_arr);
    f64* t = (f64*)PyArray_DATA(t_arr);
    f64* q = (f64*)PyArray_DATA(q_arr);

    i32 mn = 0;
    for (i32 i = 0; i < k; i++) {
        if (n_data[i] > mn) mn = n_data[i];
    }
    if (mn <= 10) mn = 0;

    i32 minwrk;
    i32 opt1 = 42 * k + mn;
    i32 opt2 = 80 * k - 48;
    minwrk = (opt1 > opt2) ? opt1 : opt2;
    if (minwrk < 1) minwrk = 1;

    i32 alloc_ldwork;
    if (ldwork == -1) {
        alloc_ldwork = 1;
    } else if (ldwork <= 0) {
        ldwork = minwrk + 100;
        alloc_ldwork = ldwork;
    } else {
        alloc_ldwork = ldwork;
    }

    i32* iwork = (i32*)malloc(4 * k * sizeof(i32));
    f64* dwork = (f64*)malloc(alloc_ldwork * sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        Py_DECREF(whichq_arr);
        Py_DECREF(n_arr);
        Py_DECREF(ni_arr);
        Py_DECREF(s_arr);
        Py_DECREF(ldt_arr);
        Py_DECREF(ixt_arr);
        Py_DECREF(ldq_arr);
        Py_DECREF(ixq_arr);
        Py_DECREF(tol_arr);
        PyArray_DiscardWritebackIfCopy(t_arr);
        PyArray_DiscardWritebackIfCopy(q_arr);
        Py_DECREF(t_arr);
        Py_DECREF(q_arr);
        return PyErr_NoMemory();
    }

    i32 info = 0;
    i32 ifst_out = ifst;
    i32 ilst_out = ilst;

    mb03ka(compq, whichq, ws, k, nc, kschur, &ifst_out, &ilst_out,
           n_data, ni, s, t, ldt, ixt, q, ldq, ixq,
           tol, iwork, dwork, ldwork, &info);

    PyArray_ResolveWritebackIfCopy(t_arr);
    PyArray_ResolveWritebackIfCopy(q_arr);

    free(iwork);
    free(dwork);

    PyObject* result = Py_BuildValue("(OOiii)", (PyObject*)t_arr, (PyObject*)q_arr,
                                     (int)ifst_out, (int)ilst_out, (int)info);

    Py_DECREF(whichq_arr);
    Py_DECREF(n_arr);
    Py_DECREF(ni_arr);
    Py_DECREF(s_arr);
    Py_DECREF(ldt_arr);
    Py_DECREF(ixt_arr);
    Py_DECREF(ldq_arr);
    Py_DECREF(ixq_arr);
    Py_DECREF(tol_arr);
    Py_DECREF(t_arr);
    Py_DECREF(q_arr);

    return result;
}

PyObject *py_mb03rw(PyObject *self, PyObject *args) {
    (void)self;

    i32 m, n;
    f64 pmax;
    PyObject *a_obj, *b_obj, *c_obj;

    if (!PyArg_ParseTuple(args, "iidOOO", &m, &n, &pmax, &a_obj, &b_obj, &c_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_COMPLEX128, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_COMPLEX128, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *c_array = (PyArrayObject *)PyArray_FROM_OTF(
        c_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || b_array == NULL || c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        return NULL;
    }

    i32 lda = (m > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 ldb = (n > 0) ? (i32)PyArray_DIM(b_array, 0) : 1;
    i32 ldc = (m > 0) ? (i32)PyArray_DIM(c_array, 0) : 1;

    const c128 *a_data = (const c128 *)PyArray_DATA(a_array);
    const c128 *b_data = (const c128 *)PyArray_DATA(b_array);
    c128 *c_data = (c128 *)PyArray_DATA(c_array);

    i32 info = 0;
    mb03rw(m, n, pmax, a_data, lda, b_data, ldb, c_data, ldc, &info);

    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("Oi", (PyObject *)c_array, (int)info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);

    return result;
}

PyObject *py_mb03sd(PyObject *self, PyObject *args) {
    (void)self;

    const char *jobscl;
    PyObject *a_obj, *qg_obj;

    if (!PyArg_ParseTuple(args, "sOO", &jobscl, &a_obj, &qg_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *qg_array = (PyArrayObject *)PyArray_FROM_OTF(
        qg_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);

    if (a_array == NULL || qg_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(qg_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 lda = (n > 0) ? n : 1;
    i32 ldqg = (n > 0) ? (i32)PyArray_DIM(qg_array, 0) : 1;

    const f64 *a_data = (const f64 *)PyArray_DATA(a_array);
    const f64 *qg_data = (const f64 *)PyArray_DATA(qg_array);

    npy_intp wr_dims[1] = {n};
    PyArrayObject *wr_array = (PyArrayObject *)PyArray_EMPTY(1, wr_dims, NPY_DOUBLE, 0);
    PyArrayObject *wi_array = (PyArrayObject *)PyArray_EMPTY(1, wr_dims, NPY_DOUBLE, 0);

    if (wr_array == NULL || wi_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(qg_array);
        Py_XDECREF(wr_array);
        Py_XDECREF(wi_array);
        return NULL;
    }

    f64 *wr_data = (f64 *)PyArray_DATA(wr_array);
    f64 *wi_data = (f64 *)PyArray_DATA(wi_array);

    i32 n2 = n * n;
    i32 ldwork = (n2 + n > 1) ? (n2 + n) : 1;
    f64 *dwork = (f64 *)malloc(ldwork * sizeof(f64));
    if (dwork == NULL && ldwork > 0) {
        Py_DECREF(a_array);
        Py_DECREF(qg_array);
        Py_DECREF(wr_array);
        Py_DECREF(wi_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info = 0;
    mb03sd(jobscl, n, (f64 *)a_data, lda, (f64 *)qg_data, ldqg,
           wr_data, wi_data, dwork, ldwork, &info);

    free(dwork);

    PyObject *result = Py_BuildValue("OOi", wr_array, wi_array, (int)info);

    Py_DECREF(a_array);
    Py_DECREF(qg_array);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);

    return result;
}

PyObject* py_mb03kd(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;
    static char* kwlist[] = {"compq", "strong", "k", "nc", "kschur",
                             "n", "ni", "s", "select", "t", "ldt",
                             "ixt", "ldq", "ixq", "tol",
                             "q", "ldwork", NULL};

    const char* compq;
    const char* strong;
    PyObject *n_obj, *ni_obj, *s_obj, *select_obj, *t_obj, *ldt_obj, *ixt_obj;
    PyObject *ldq_obj, *ixq_obj;
    PyObject *q_obj = Py_None;
    int k, nc, kschur;
    double tol;
    int ldwork = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssiiiOOOOOOOOOd|Oi", kwlist,
            &compq, &strong, &k, &nc, &kschur,
            &n_obj, &ni_obj, &s_obj, &select_obj, &t_obj, &ldt_obj,
            &ixt_obj, &ldq_obj, &ixq_obj, &tol, &q_obj, &ldwork)) {
        return NULL;
    }

    PyArrayObject *n_arr = (PyArrayObject *)PyArray_FROM_OTF(n_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ni_arr = (PyArrayObject *)PyArray_FROM_OTF(ni_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *s_arr = (PyArrayObject *)PyArray_FROM_OTF(s_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *select_arr = (PyArrayObject *)PyArray_FROM_OTF(select_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ldt_arr = (PyArrayObject *)PyArray_FROM_OTF(ldt_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ixt_arr = (PyArrayObject *)PyArray_FROM_OTF(ixt_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ldq_arr = (PyArrayObject *)PyArray_FROM_OTF(ldq_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ixq_arr = (PyArrayObject *)PyArray_FROM_OTF(ixq_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *t_arr = (PyArrayObject *)PyArray_FROM_OTF(t_obj, NPY_DOUBLE,
        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!n_arr || !ni_arr || !s_arr || !select_arr || !ldt_arr || !ixt_arr ||
        !ldq_arr || !ixq_arr || !t_arr) {
        Py_XDECREF(n_arr);
        Py_XDECREF(ni_arr);
        Py_XDECREF(s_arr);
        Py_XDECREF(select_arr);
        Py_XDECREF(ldt_arr);
        Py_XDECREF(ixt_arr);
        Py_XDECREF(ldq_arr);
        Py_XDECREF(ixq_arr);
        Py_XDECREF(t_arr);
        return NULL;
    }

    i32* n_data = (i32*)PyArray_DATA(n_arr);
    i32* ni = (i32*)PyArray_DATA(ni_arr);
    i32* s = (i32*)PyArray_DATA(s_arr);
    bool* select_data = (bool*)PyArray_DATA(select_arr);
    i32* ldt = (i32*)PyArray_DATA(ldt_arr);
    i32* ixt = (i32*)PyArray_DATA(ixt_arr);
    i32* ldq = (i32*)PyArray_DATA(ldq_arr);
    i32* ixq = (i32*)PyArray_DATA(ixq_arr);
    f64* t = (f64*)PyArray_DATA(t_arr);

    bool compq_is_N = (compq[0] == 'N' || compq[0] == 'n');

    i32 total_q_size = 0;
    if (!compq_is_N) {
        for (i32 i = 0; i < k; i++) {
            i32 max_q = (ixq[i] - 1) + ldq[i] * n_data[i];
            if (max_q > total_q_size) total_q_size = max_q;
        }
    }
    if (total_q_size < 1) total_q_size = 1;

    PyArrayObject *q_arr = NULL;
    f64* q = NULL;

    if (compq_is_N) {
        npy_intp q_dims[1] = {1};
        q_arr = (PyArrayObject *)PyArray_EMPTY(1, q_dims, NPY_DOUBLE, 1);
        if (!q_arr) {
            Py_DECREF(n_arr);
            Py_DECREF(ni_arr);
            Py_DECREF(s_arr);
            Py_DECREF(select_arr);
            Py_DECREF(ldt_arr);
            Py_DECREF(ixt_arr);
            Py_DECREF(ldq_arr);
            Py_DECREF(ixq_arr);
            PyArray_DiscardWritebackIfCopy(t_arr);
            Py_DECREF(t_arr);
            return NULL;
        }
        q = (f64*)PyArray_DATA(q_arr);
    } else if (q_obj != Py_None) {
        q_arr = (PyArrayObject *)PyArray_FROM_OTF(q_obj, NPY_DOUBLE,
            NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!q_arr) {
            Py_DECREF(n_arr);
            Py_DECREF(ni_arr);
            Py_DECREF(s_arr);
            Py_DECREF(select_arr);
            Py_DECREF(ldt_arr);
            Py_DECREF(ixt_arr);
            Py_DECREF(ldq_arr);
            Py_DECREF(ixq_arr);
            PyArray_DiscardWritebackIfCopy(t_arr);
            Py_DECREF(t_arr);
            return NULL;
        }
        q = (f64*)PyArray_DATA(q_arr);
    } else {
        npy_intp q_dims[1] = {total_q_size};
        q_arr = (PyArrayObject *)PyArray_ZEROS(1, q_dims, NPY_DOUBLE, 1);
        if (!q_arr) {
            Py_DECREF(n_arr);
            Py_DECREF(ni_arr);
            Py_DECREF(s_arr);
            Py_DECREF(select_arr);
            Py_DECREF(ldt_arr);
            Py_DECREF(ixt_arr);
            Py_DECREF(ldq_arr);
            Py_DECREF(ixq_arr);
            PyArray_DiscardWritebackIfCopy(t_arr);
            Py_DECREF(t_arr);
            return NULL;
        }
        q = (f64*)PyArray_DATA(q_arr);
    }

    i32 mn = 0;
    for (i32 i = 0; i < k; i++) {
        if (n_data[i] > mn) mn = n_data[i];
    }
    if (mn <= 10) mn = 0;

    i32 minwrk;
    i32 opt1 = 42 * k + mn;
    i32 opt2 = 80 * k - 48;
    minwrk = (opt1 > opt2) ? opt1 : opt2;
    if (minwrk < 1) minwrk = 1;

    i32 alloc_ldwork;
    if (ldwork == -1) {
        alloc_ldwork = 1;
    } else if (ldwork <= 0) {
        ldwork = minwrk + 100;
        alloc_ldwork = ldwork;
    } else {
        alloc_ldwork = ldwork;
    }

    i32* iwork = (i32*)malloc(4 * k * sizeof(i32));
    f64* dwork = (f64*)malloc(alloc_ldwork * sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        Py_DECREF(n_arr);
        Py_DECREF(ni_arr);
        Py_DECREF(s_arr);
        Py_DECREF(select_arr);
        Py_DECREF(ldt_arr);
        Py_DECREF(ixt_arr);
        Py_DECREF(ldq_arr);
        Py_DECREF(ixq_arr);
        PyArray_DiscardWritebackIfCopy(t_arr);
        Py_DECREF(t_arr);
        if (q_arr) {
            if (!compq_is_N && q_obj != Py_None) {
                PyArray_DiscardWritebackIfCopy(q_arr);
            }
            Py_DECREF(q_arr);
        }
        return PyErr_NoMemory();
    }

    i32 info = 0;
    i32 m = 0;

    mb03kd(compq, NULL, strong, k, nc, kschur,
           n_data, ni, s, select_data, t, ldt, ixt,
           q, ldq, ixq, &m, tol, iwork, dwork, ldwork, &info);

    PyArray_ResolveWritebackIfCopy(t_arr);
    if (!compq_is_N && q_obj != Py_None && q_arr) {
        PyArray_ResolveWritebackIfCopy(q_arr);
    }

    free(iwork);
    free(dwork);

    PyObject* result = Py_BuildValue("(OOii)", (PyObject*)t_arr, (PyObject*)q_arr,
                                     (int)m, (int)info);

    Py_DECREF(n_arr);
    Py_DECREF(ni_arr);
    Py_DECREF(s_arr);
    Py_DECREF(select_arr);
    Py_DECREF(ldt_arr);
    Py_DECREF(ixt_arr);
    Py_DECREF(ldq_arr);
    Py_DECREF(ixq_arr);
    Py_DECREF(t_arr);
    if (q_arr) {
        Py_DECREF(q_arr);
    }

    return result;
}

PyObject *py_mb03rz(PyObject *self, PyObject *args, PyObject *kwargs) {
    const char *jobx;
    const char *sort;
    PyObject *a_obj;
    double pmax;
    double tol;
    PyObject *x_obj = Py_None;

    static char *kwlist[] = {"jobx", "sort", "a", "pmax", "tol", "x", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssOdd|O", kwlist,
                                     &jobx, &sort, &a_obj, &pmax, &tol, &x_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 n = (i32)a_dims[0];
    i32 lda = n > 0 ? n : 1;

    int wantx = (jobx[0] == 'U' || jobx[0] == 'u');
    PyArrayObject *x_array = NULL;
    c128 *x_data = NULL;
    i32 ldx = 1;

    if (wantx) {
        if (x_obj != Py_None && x_obj != NULL) {
            x_array = (PyArrayObject *)PyArray_FROM_OTF(
                x_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        } else {
            npy_intp x_dims[2] = {n > 0 ? n : 1, n > 0 ? n : 1};
            x_array = (PyArrayObject *)PyArray_ZEROS(2, x_dims, NPY_COMPLEX128, 1);
            if (x_array) {
                c128 *xptr = (c128 *)PyArray_DATA(x_array);
                for (i32 i = 0; i < n; i++) {
                    xptr[i + i * n] = 1.0 + 0.0*I;
                }
            }
        }
        if (x_array == NULL) {
            PyArray_DiscardWritebackIfCopy(a_array);
            Py_DECREF(a_array);
            return NULL;
        }
        x_data = (c128 *)PyArray_DATA(x_array);
        ldx = n > 0 ? n : 1;
    } else {
        npy_intp x_dims[2] = {1, 1};
        x_array = (PyArrayObject *)PyArray_ZEROS(2, x_dims, NPY_COMPLEX128, 1);
        if (x_array == NULL) {
            PyArray_DiscardWritebackIfCopy(a_array);
            Py_DECREF(a_array);
            return NULL;
        }
        x_data = (c128 *)PyArray_DATA(x_array);
    }

    npy_intp blsize_dims[1] = {n > 0 ? n : 1};
    PyArrayObject *blsize_array = (PyArrayObject *)PyArray_SimpleNew(1, blsize_dims, NPY_INT32);
    npy_intp w_dims[1] = {n > 0 ? n : 1};
    PyArrayObject *w_array = (PyArrayObject *)PyArray_SimpleNew(1, w_dims, NPY_COMPLEX128);

    if (blsize_array == NULL || w_array == NULL) {
        Py_XDECREF(blsize_array);
        Py_XDECREF(w_array);
        PyArray_DiscardWritebackIfCopy(a_array);
        Py_DECREF(a_array);
        if (wantx && x_obj != Py_None && x_obj != NULL) {
            PyArray_DiscardWritebackIfCopy(x_array);
        }
        Py_DECREF(x_array);
        return NULL;
    }

    c128 *a_data = (c128 *)PyArray_DATA(a_array);
    i32 *blsize_data = (i32 *)PyArray_DATA(blsize_array);
    c128 *w_data = (c128 *)PyArray_DATA(w_array);

    i32 nblcks = 0;
    i32 info = 0;

    mb03rz(jobx, sort, n, pmax, a_data, lda, x_data, ldx,
           &nblcks, blsize_data, w_data, tol, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    if (wantx && x_obj != Py_None && x_obj != NULL) {
        PyArray_ResolveWritebackIfCopy(x_array);
    }

    PyObject *result = Py_BuildValue("OOiOOi",
                                     a_array, x_array, (int)nblcks,
                                     blsize_array, w_array, (int)info);

    Py_DECREF(a_array);
    Py_DECREF(x_array);
    Py_DECREF(blsize_array);
    Py_DECREF(w_array);

    return result;
}

PyObject *py_mb03ts(PyObject *self, PyObject *args) {
    (void)self;

    int isham_int, wantu_int;
    int j1, n1, n2;
    PyObject *a_obj, *g_obj, *u1_obj, *u2_obj;

    if (!PyArg_ParseTuple(args, "iiOOOOiii", &isham_int, &wantu_int,
                          &a_obj, &g_obj, &u1_obj, &u2_obj, &j1, &n1, &n2)) {
        return NULL;
    }

    bool isham = isham_int != 0;
    bool wantu = wantu_int != 0;

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *g_array = (PyArrayObject *)PyArray_FROM_OTF(
        g_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *u1_array = (PyArrayObject *)PyArray_FROM_OTF(
        u1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *u2_array = (PyArrayObject *)PyArray_FROM_OTF(
        u2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || g_array == NULL || u1_array == NULL || u2_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(g_array);
        Py_XDECREF(u1_array);
        Py_XDECREF(u2_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 lda = (n > 0) ? n : 1;
    i32 ldg = (n > 0) ? (i32)PyArray_DIM(g_array, 0) : 1;
    i32 ldu1 = (n > 0 && wantu) ? (i32)PyArray_DIM(u1_array, 0) : 1;
    i32 ldu2 = (n > 0 && wantu) ? (i32)PyArray_DIM(u2_array, 0) : 1;

    f64 *a_data = (f64 *)PyArray_DATA(a_array);
    f64 *g_data = (f64 *)PyArray_DATA(g_array);
    f64 *u1_data = (f64 *)PyArray_DATA(u1_array);
    f64 *u2_data = (f64 *)PyArray_DATA(u2_array);

    i32 ldwork = (n > 0) ? n : 1;
    f64 *dwork = (f64 *)malloc(ldwork * sizeof(f64));
    if (dwork == NULL && ldwork > 0) {
        Py_DECREF(a_array);
        Py_DECREF(g_array);
        Py_DECREF(u1_array);
        Py_DECREF(u2_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info = 0;
    mb03ts(isham, wantu, n, a_data, lda, g_data, ldg,
           u1_data, ldu1, u2_data, ldu2, j1, n1, n2, dwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(g_array);
    PyArray_ResolveWritebackIfCopy(u1_array);
    PyArray_ResolveWritebackIfCopy(u2_array);

    PyObject *result = Py_BuildValue("OOOOi", a_array, g_array, u1_array, u2_array, (int)info);

    Py_DECREF(a_array);
    Py_DECREF(g_array);
    Py_DECREF(u1_array);
    Py_DECREF(u2_array);

    return result;
}

PyObject *py_mb03td(PyObject *self, PyObject *args) {
    (void)self;

    const char *typ, *compu;
    PyObject *select_obj, *lower_obj, *a_obj, *g_obj, *u1_obj, *u2_obj;

    if (!PyArg_ParseTuple(args, "ssOOOOOO", &typ, &compu,
                          &select_obj, &lower_obj, &a_obj, &g_obj, &u1_obj, &u2_obj)) {
        return NULL;
    }

    PyArrayObject *select_array = (PyArrayObject *)PyArray_FROM_OTF(
        select_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *lower_array = (PyArrayObject *)PyArray_FROM_OTF(
        lower_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *g_array = (PyArrayObject *)PyArray_FROM_OTF(
        g_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *u1_array = (PyArrayObject *)PyArray_FROM_OTF(
        u1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *u2_array = (PyArrayObject *)PyArray_FROM_OTF(
        u2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (select_array == NULL || lower_array == NULL || a_array == NULL ||
        g_array == NULL || u1_array == NULL || u2_array == NULL) {
        Py_XDECREF(select_array);
        Py_XDECREF(lower_array);
        Py_XDECREF(a_array);
        Py_XDECREF(g_array);
        Py_XDECREF(u1_array);
        Py_XDECREF(u2_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    bool wantu = (compu[0] == 'U' || compu[0] == 'u');

    i32 lda = (n > 0) ? n : 1;
    i32 ldg = (n > 0) ? (i32)PyArray_DIM(g_array, 0) : 1;
    i32 ldu1 = (n > 0 && wantu) ? (i32)PyArray_DIM(u1_array, 0) : 1;
    i32 ldu2 = (n > 0 && wantu) ? (i32)PyArray_DIM(u2_array, 0) : 1;

    bool *select_data = (bool *)PyArray_DATA(select_array);
    bool *lower_data = (bool *)PyArray_DATA(lower_array);
    f64 *a_data = (f64 *)PyArray_DATA(a_array);
    f64 *g_data = (f64 *)PyArray_DATA(g_array);
    f64 *u1_data = (f64 *)PyArray_DATA(u1_array);
    f64 *u2_data = (f64 *)PyArray_DATA(u2_array);

    i32 ldwork = (n > 0) ? n : 1;
    f64 *dwork = (f64 *)malloc(ldwork * sizeof(f64));

    if (dwork == NULL && ldwork > 0) {
        Py_DECREF(select_array);
        Py_DECREF(lower_array);
        Py_DECREF(a_array);
        Py_DECREF(g_array);
        Py_DECREF(u1_array);
        Py_DECREF(u2_array);
        PyErr_NoMemory();
        return NULL;
    }

    // Allocate output arrays via NumPy (safe pattern per BUG_PATTERN_OWNDATA_FREE.md)
    npy_intp dims_wr[1] = {n > 0 ? n : 0};
    PyObject *wr_array = PyArray_SimpleNew(1, dims_wr, NPY_DOUBLE);
    PyObject *wi_array = PyArray_SimpleNew(1, dims_wr, NPY_DOUBLE);

    if (wr_array == NULL || wi_array == NULL) {
        free(dwork);
        Py_XDECREF(wr_array);
        Py_XDECREF(wi_array);
        Py_DECREF(select_array);
        Py_DECREF(lower_array);
        Py_DECREF(a_array);
        Py_DECREF(g_array);
        Py_DECREF(u1_array);
        Py_DECREF(u2_array);
        return NULL;
    }

    f64 *wr = (n > 0) ? (f64 *)PyArray_DATA((PyArrayObject *)wr_array) : NULL;
    f64 *wi = (n > 0) ? (f64 *)PyArray_DATA((PyArrayObject *)wi_array) : NULL;
    
    // mb03td needs valid pointers even if n=0, use dummies
    f64 wr_dummy, wi_dummy;
    if (n <= 0) {
        wr = &wr_dummy;
        wi = &wi_dummy;
    }

    i32 m = 0;
    i32 info = 0;

    mb03td(typ, compu, select_data, lower_data, n, a_data, lda, g_data, ldg,
           u1_data, ldu1, u2_data, ldu2, wr, wi, &m, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(g_array);
    PyArray_ResolveWritebackIfCopy(u1_array);
    PyArray_ResolveWritebackIfCopy(u2_array);

    // Arrays already allocated by NumPy - no OWNDATA needed

    PyObject *result = Py_BuildValue("OOOOOOii", a_array, g_array, u1_array, u2_array,
                                      wr_array, wi_array, (int)m, (int)info);

    Py_DECREF(select_array);
    Py_DECREF(lower_array);
    Py_DECREF(a_array);
    Py_DECREF(g_array);
    Py_DECREF(u1_array);
    Py_DECREF(u2_array);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);

    return result;
}

/* Python wrapper for mb03vw */
PyObject* py_mb03vw(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *compq, *triu;
    PyObject *qind_obj = Py_None;
    PyObject *s_obj, *a_obj;
    PyObject *q_obj = Py_None;
    i32 n, k, h, ilo, ihi;
    i32 info = 0;

    static char *kwlist[] = {"compq", "qind", "triu", "n", "k", "h", "ilo", "ihi",
                             "s", "a", "q", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sOsiiiii|OOO", kwlist,
                                     &compq, &qind_obj, &triu, &n, &k, &h,
                                     &ilo, &ihi, &s_obj, &a_obj, &q_obj)) {
        return NULL;
    }

    PyArrayObject *s_array = (PyArrayObject*)PyArray_FROM_OTF(s_obj, NPY_INT32,
                                                              NPY_ARRAY_IN_ARRAY);
    if (s_array == NULL) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        Py_DECREF(s_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda1 = (i32)a_dims[0];
    i32 lda2 = (i32)a_dims[1];
    if (lda1 < 1) lda1 = 1;
    if (lda2 < 1) lda2 = 1;

    bool lcmpq = (compq[0] == 'U' || compq[0] == 'u' ||
                  compq[0] == 'I' || compq[0] == 'i');
    bool lparq = (compq[0] == 'P' || compq[0] == 'p');
    bool need_q = lcmpq || lparq;

    i32 ldq1 = need_q ? (n > 1 ? n : 1) : 1;
    i32 ldq2 = need_q ? (n > 1 ? n : 1) : 1;

    PyArrayObject *qind_array = NULL;
    i32 *qind_data = NULL;
    if (lparq && qind_obj != Py_None) {
        qind_array = (PyArrayObject*)PyArray_FROM_OTF(qind_obj, NPY_INT32,
                                                       NPY_ARRAY_IN_ARRAY);
        if (qind_array == NULL) {
            Py_DECREF(s_array);
            Py_DECREF(a_array);
            return NULL;
        }
        qind_data = (i32*)PyArray_DATA(qind_array);
    }

    PyArrayObject *q_array = NULL;
    f64 *q_data = NULL;

    if (need_q) {
        if (q_obj != Py_None && (compq[0] == 'U' || compq[0] == 'u' || lparq)) {
            q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE,
                                                       NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            if (q_array == NULL) {
                Py_XDECREF(qind_array);
                Py_DECREF(s_array);
                Py_DECREF(a_array);
                return NULL;
            }
            npy_intp *q_dims = PyArray_DIMS(q_array);
            ldq1 = (i32)q_dims[0];
            q_data = (f64*)PyArray_DATA(q_array);
        } else {
            npy_intp q_dims[3] = {ldq1, ldq2, k > 0 ? k : 1};
            // Use PyArray_New with NULL data to allocate F-contiguous array safely
            q_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 3, q_dims, NPY_DOUBLE,
                                                  NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            if (q_array == NULL) {
                Py_XDECREF(qind_array);
                Py_DECREF(s_array);
                Py_DECREF(a_array);
                return NULL;
            }
            q_data = (f64*)PyArray_DATA(q_array);
            memset(q_data, 0, PyArray_NBYTES(q_array));
        }
    }

    i32 liwork = k > 0 ? 3 * k : 1;
    i32 *iwork = (i32*)malloc(liwork * sizeof(i32));

    i32 m = ihi - ilo + 1;
    i32 maxval = (ihi > n - ilo + 1) ? ihi : (n - ilo + 1);
    i32 ldwork = (n < 1 || k < 1 || n == 1 || ilo == ihi) ? 1 : m + maxval;
    ldwork = ldwork < 64 ? 64 : ldwork;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (iwork == NULL || dwork == NULL) {
        free(iwork);
        free(dwork);
        Py_XDECREF(q_array);
        Py_XDECREF(qind_array);
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    i32 *s_data = (i32*)PyArray_DATA(s_array);

    mb03vw(compq, qind_data, triu, n, k, &h, ilo, ihi, s_data,
           a_data, lda1, lda2, q_data, ldq1, ldq2,
           iwork, liwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    if (info < 0) {
        Py_XDECREF(q_array);
        Py_XDECREF(qind_array);
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        PyErr_Format(PyExc_ValueError, "mb03vw: illegal argument %d", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(a_array);
    if (q_array != NULL && PyArray_FLAGS(q_array) & NPY_ARRAY_WRITEBACKIFCOPY) {
        PyArray_ResolveWritebackIfCopy(q_array);
    }

    PyObject *result;
    if (need_q) {
        result = Py_BuildValue("OOii", a_array, q_array, (int)h, (int)info);
    } else {
        result = Py_BuildValue("OOii", a_array, Py_None, (int)h, (int)info);
    }

    Py_XDECREF(qind_array);
    Py_DECREF(s_array);
    Py_DECREF(a_array);
    Py_XDECREF(q_array);

    return result;
}

/* Python wrapper for mb03wx */
PyObject* py_mb03wx(PyObject* self, PyObject* args) {
    (void)self;

    i32 n, p;
    PyObject *t_obj;

    if (!PyArg_ParseTuple(args, "iiO", &n, &p, &t_obj)) {
        return NULL;
    }

    PyArrayObject *t_array = (PyArrayObject*)PyArray_FROM_OTF(t_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_IN_FARRAY);
    if (t_array == NULL) {
        return NULL;
    }

    npy_intp *t_dims = PyArray_DIMS(t_array);
    i32 ldt1 = (i32)t_dims[0];
    i32 ldt2 = (i32)t_dims[1];
    if (ldt1 < 1) ldt1 = 1;
    if (ldt2 < 1) ldt2 = 1;

    const f64 *t_data = (const f64*)PyArray_DATA(t_array);

    npy_intp out_dims[1] = {n > 0 ? n : 0};
    PyObject *wr_array = PyArray_New(&PyArray_Type, 1, out_dims, NPY_DOUBLE,
                                     NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *wi_array = PyArray_New(&PyArray_Type, 1, out_dims, NPY_DOUBLE,
                                     NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (wr_array == NULL || wi_array == NULL) {
        Py_XDECREF(wr_array);
        Py_XDECREF(wi_array);
        Py_DECREF(t_array);
        return NULL;
    }

    f64 *wr_data = (f64*)PyArray_DATA((PyArrayObject*)wr_array);
    f64 *wi_data = (f64*)PyArray_DATA((PyArrayObject*)wi_array);

    i32 info = 0;
    mb03wx(n, p, t_data, ldt1, ldt2, wr_data, wi_data, &info);

    Py_DECREF(t_array);

    PyObject *result = Py_BuildValue("OOi", wr_array, wi_array, (int)info);

    Py_DECREF(wr_array);
    Py_DECREF(wi_array);

    return result;
}

/* Python wrapper for mb03fd */
PyObject* py_mb03fd(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;

    i32 n;
    f64 prec;
    PyObject *a_obj, *b_obj;

    static char *kwlist[] = {"n", "prec", "a", "b", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "idOO", kwlist,
                                     &n, &prec, &a_obj, &b_obj)) {
        return NULL;
    }

    if (n != 2 && n != 4) {
        PyErr_SetString(PyExc_ValueError, "mb03fd: n must be 2 or 4");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);

    if (PyArray_NDIM(a_array) != 2 || a_dims[0] < n || a_dims[1] < n) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        PyErr_SetString(PyExc_ValueError, "mb03fd: a must be at least n-by-n");
        return NULL;
    }

    if (PyArray_NDIM(b_array) != 2 || b_dims[0] < n || b_dims[1] < n) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        PyErr_SetString(PyExc_ValueError, "mb03fd: b must be at least n-by-n");
        return NULL;
    }

    i32 lda = (i32)a_dims[0];
    i32 ldb = (i32)b_dims[0];

    npy_intp q_dims[2] = {n, n};

    PyObject *q1_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE,
                                     NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *q2_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE,
                                     NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (q1_array == NULL || q2_array == NULL) {
        Py_XDECREF(q1_array);
        Py_XDECREF(q2_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    f64 *q1_data = (f64*)PyArray_DATA((PyArrayObject*)q1_array);
    f64 *q2_data = (f64*)PyArray_DATA((PyArrayObject*)q2_array);
    memset(q1_data, 0, PyArray_NBYTES((PyArrayObject*)q1_array));
    memset(q2_data, 0, PyArray_NBYTES((PyArrayObject*)q2_array));

    i32 ldwork = (n == 4) ? 128 : 1;
    f64 *dwork = NULL;
    if (n == 4) {
        dwork = (f64*)malloc(ldwork * sizeof(f64));
        if (dwork == NULL) {
            free(q1_data);
            free(q2_data);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
            return NULL;
        }
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    i32 ldq1 = n;
    i32 ldq2 = n;
    i32 info = 0;

    mb03fd(n, prec, a_data, lda, b_data, ldb,
           q1_data, ldq1, q2_data, ldq2,
           dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    PyObject *result = Py_BuildValue("OOOOi", a_array, b_array, q1_array, q2_array, (int)info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(q1_array);
    Py_DECREF(q2_array);

    return result;
}

PyObject *py_mb03fz(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;

    const char *compq, *compu, *orth;
    int n;
    PyObject *z_obj, *b_obj, *fg_obj;

    static char *kwlist[] = {"compq", "compu", "orth", "n", "z", "b", "fg", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssiOOO", kwlist,
                                     &compq, &compu, &orth, &n, &z_obj, &b_obj, &fg_obj)) {
        return NULL;
    }

    i32 m = n / 2;
    int lcmpq = (compq[0] == 'C' || compq[0] == 'c');
    int lcmpu = (compu[0] == 'C' || compu[0] == 'c');
    int lcmp = lcmpq || lcmpu;

    PyArrayObject *z_array = (PyArrayObject *)PyArray_FROM_OTF(
        z_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_in_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);
    PyArrayObject *fg_in_array = (PyArrayObject *)PyArray_FROM_OTF(
        fg_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);

    if (z_array == NULL || b_in_array == NULL || fg_in_array == NULL) {
        Py_XDECREF(z_array);
        Py_XDECREF(b_in_array);
        Py_XDECREF(fg_in_array);
        return NULL;
    }

    i32 ldz = (n > 0) ? (i32)(PyArray_STRIDE(z_array, 1) / sizeof(c128)) : 1;
    c128 *z_data = (c128 *)PyArray_DATA(z_array);

    c128 *b_in_data = (c128 *)PyArray_DATA(b_in_array);
    c128 *fg_in_data = (c128 *)PyArray_DATA(fg_in_array);
    i32 ldb_in = (m > 0) ? (i32)(PyArray_STRIDE(b_in_array, 1) / sizeof(c128)) : 1;
    i32 ldfg_in = (m > 0) ? (i32)(PyArray_STRIDE(fg_in_array, 1) / sizeof(c128)) : 1;

    i32 ldb = (n > 0) ? n : 1;
    i32 ldfg = (n > 0) ? n : 1;
    i32 ldd = (lcmp && n > 0) ? n : 1;
    i32 ldc = (lcmp && n > 0) ? n : 1;
    i32 ldq = lcmpq ? (n > 0 ? 2 * n : 1) : 1;
    i32 ldu = lcmpu ? (n > 0 ? n : 1) : 1;

    c128 *b_data = NULL;
    c128 *fg_data = NULL;
    c128 *d_data = NULL;
    c128 *c_data = NULL;
    c128 *q_data = NULL;
    c128 *u_data = NULL;

    if (n > 0) {
        b_data = (c128 *)PyMem_Calloc(n * n, sizeof(c128));
        fg_data = (c128 *)PyMem_Calloc(n * (n + 1), sizeof(c128));
        if (b_data == NULL || fg_data == NULL) {
            PyMem_Free(b_data);
            PyMem_Free(fg_data);
            Py_DECREF(z_array);
            Py_DECREF(b_in_array);
            Py_DECREF(fg_in_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate B/FG arrays");
            return NULL;
        }
        for (i32 j = 0; j < m; j++) {
            for (i32 i = 0; i < m; i++) {
                b_data[i + j * ldb] = b_in_data[i + j * ldb_in];
            }
        }
        i32 fg_ncols = m + 1;
        for (i32 j = 0; j < fg_ncols; j++) {
            for (i32 i = 0; i < m; i++) {
                fg_data[i + j * ldfg] = fg_in_data[i + j * ldfg_in];
            }
        }
    }

    Py_DECREF(b_in_array);
    Py_DECREF(fg_in_array);

    if (lcmp && n > 0) {
        d_data = (c128 *)PyMem_Calloc(n * n, sizeof(c128));
        c_data = (c128 *)PyMem_Calloc(n * n, sizeof(c128));
        if (d_data == NULL || c_data == NULL) {
            PyMem_Free(b_data);
            PyMem_Free(fg_data);
            PyMem_Free(d_data);
            PyMem_Free(c_data);
            Py_DECREF(z_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate D/C arrays");
            return NULL;
        }
    }

    if (n > 0) {
        q_data = (c128 *)PyMem_Calloc(2 * n * 2 * n, sizeof(c128));
        if (q_data == NULL) {
            PyMem_Free(b_data);
            PyMem_Free(fg_data);
            PyMem_Free(d_data);
            PyMem_Free(c_data);
            Py_DECREF(z_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate Q array");
            return NULL;
        }
    }

    if (n > 0) {
        u_data = (c128 *)PyMem_Calloc(n * 2 * n, sizeof(c128));
        if (u_data == NULL) {
            PyMem_Free(b_data);
            PyMem_Free(fg_data);
            PyMem_Free(d_data);
            PyMem_Free(c_data);
            PyMem_Free(q_data);
            Py_DECREF(z_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate U array");
            return NULL;
        }
    }

    f64 *alphar = (f64 *)PyMem_Calloc(n > 0 ? n : 1, sizeof(f64));
    f64 *alphai = (f64 *)PyMem_Calloc(n > 0 ? n : 1, sizeof(f64));
    f64 *beta = (f64 *)PyMem_Calloc(n > 0 ? n : 1, sizeof(f64));

    if (alphar == NULL || alphai == NULL || beta == NULL) {
        PyMem_Free(alphar);
        PyMem_Free(alphai);
        PyMem_Free(beta);
        PyMem_Free(b_data);
        PyMem_Free(fg_data);
        PyMem_Free(d_data);
        PyMem_Free(c_data);
        PyMem_Free(q_data);
        PyMem_Free(u_data);
        Py_DECREF(z_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate eigenvalue arrays");
        return NULL;
    }

    i32 liwork = 2 * n + 9;
    i32 *iwork = (i32 *)PyMem_Calloc(liwork > 0 ? liwork : 1, sizeof(i32));

    i32 c_coef = lcmpu ? 18 : (lcmpq ? 16 : 13);
    i32 max_6n_27 = (6 * n > 27) ? 6 * n : 27;
    i32 ldwork = c_coef * n * n + n + max_6n_27;
    if (ldwork < 1) ldwork = 1;
    f64 *dwork = (f64 *)PyMem_Calloc(ldwork, sizeof(f64));

    i32 lzwork;
    if (lcmpq) {
        lzwork = 8 * n + 28;
    } else if (lcmpu) {
        lzwork = 6 * n + 28;
    } else {
        lzwork = 1;
    }
    if (lzwork < 1) lzwork = 1;
    c128 *zwork = (c128 *)PyMem_Calloc(lzwork, sizeof(c128));

    bool *bwork = NULL;
    if (lcmp && n > 0) {
        bwork = (bool *)PyMem_Calloc(n, sizeof(bool));
    }

    if (iwork == NULL || dwork == NULL || zwork == NULL || (lcmp && n > 0 && bwork == NULL)) {
        PyMem_Free(iwork);
        PyMem_Free(dwork);
        PyMem_Free(zwork);
        PyMem_Free(bwork);
        PyMem_Free(alphar);
        PyMem_Free(alphai);
        PyMem_Free(beta);
        PyMem_Free(b_data);
        PyMem_Free(fg_data);
        PyMem_Free(d_data);
        PyMem_Free(c_data);
        PyMem_Free(q_data);
        PyMem_Free(u_data);
        Py_DECREF(z_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    i32 neig = 0;
    i32 info = 0;

    mb03fz(compq, compu, orth, n, z_data, ldz, b_data, ldb, fg_data, ldfg,
           &neig, d_data, ldd, c_data, ldc, q_data, ldq, u_data, ldu,
           alphar, alphai, beta, iwork, liwork, dwork, ldwork, zwork, lzwork,
           bwork, &info);

    PyMem_Free(iwork);
    PyMem_Free(dwork);
    PyMem_Free(zwork);
    PyMem_Free(bwork);

    PyArray_ResolveWritebackIfCopy(z_array);

    npy_intp alphar_dims[1] = {n > 0 ? n : 0};
    PyObject *alphar_array = PyArray_SimpleNew(1, alphar_dims, NPY_DOUBLE);
    PyObject *alphai_array = PyArray_SimpleNew(1, alphar_dims, NPY_DOUBLE);
    PyObject *beta_array = PyArray_SimpleNew(1, alphar_dims, NPY_DOUBLE);

    if (alphar_array == NULL || alphai_array == NULL || beta_array == NULL) {
        PyMem_Free(alphar);
        PyMem_Free(alphai);
        PyMem_Free(beta);
        PyMem_Free(b_data);
        PyMem_Free(fg_data);
        PyMem_Free(d_data);
        PyMem_Free(c_data);
        PyMem_Free(q_data);
        PyMem_Free(u_data);
        Py_XDECREF(alphar_array);
        Py_XDECREF(alphai_array);
        Py_XDECREF(beta_array);
        Py_DECREF(z_array);
        return NULL;
    }
    if (n > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)alphar_array), alphar, n * sizeof(f64));
        memcpy(PyArray_DATA((PyArrayObject*)alphai_array), alphai, n * sizeof(f64));
        memcpy(PyArray_DATA((PyArrayObject*)beta_array), beta, n * sizeof(f64));
    }
    PyMem_Free(alphar);
    PyMem_Free(alphai);
    PyMem_Free(beta);

    PyObject *b_array, *fg_array, *d_array, *c_array, *q_array, *u_array;

    if (n > 0) {
        npy_intp b_dims[2] = {n, n};
        b_array = PyArray_ZEROS(2, b_dims, NPY_COMPLEX128, 1);
        npy_intp fg_dims[2] = {n, n + 1};
        fg_array = PyArray_ZEROS(2, fg_dims, NPY_COMPLEX128, 1);
        if (b_array == NULL || fg_array == NULL) {
            PyMem_Free(b_data);
            PyMem_Free(fg_data);
            PyMem_Free(d_data);
            PyMem_Free(c_data);
            PyMem_Free(q_data);
            PyMem_Free(u_data);
            Py_XDECREF(b_array);
            Py_XDECREF(fg_array);
            Py_DECREF(alphar_array);
            Py_DECREF(alphai_array);
            Py_DECREF(beta_array);
            Py_DECREF(z_array);
            return NULL;
        }
        memcpy(PyArray_DATA((PyArrayObject*)b_array), b_data, n * n * sizeof(c128));
        memcpy(PyArray_DATA((PyArrayObject*)fg_array), fg_data, n * (n + 1) * sizeof(c128));
        if (lcmp) {
            npy_intp d_dims[2] = {n, n};
            d_array = PyArray_ZEROS(2, d_dims, NPY_COMPLEX128, 1);
            c_array = PyArray_ZEROS(2, d_dims, NPY_COMPLEX128, 1);
            if (d_array != NULL && c_array != NULL) {
                memcpy(PyArray_DATA((PyArrayObject*)d_array), d_data, n * n * sizeof(c128));
                memcpy(PyArray_DATA((PyArrayObject*)c_array), c_data, n * n * sizeof(c128));
            }
            PyMem_Free(d_data);
            d_data = NULL;
            PyMem_Free(c_data);
            c_data = NULL;
        }
        PyMem_Free(b_data);
        PyMem_Free(fg_data);
    } else {
        npy_intp b_dims[2] = {0, 0};
        b_array = PyArray_ZEROS(2, b_dims, NPY_COMPLEX128, 1);
        fg_array = PyArray_ZEROS(2, b_dims, NPY_COMPLEX128, 1);
    }

    // d_array and c_array are now handled in the n > 0 block above
    // Only create empty arrays if lcmp is false or n == 0
    if (!lcmp || n == 0) {
        npy_intp d_dims[2] = {0, 0};
        d_array = PyArray_ZEROS(2, d_dims, NPY_COMPLEX128, 1);
        c_array = PyArray_ZEROS(2, d_dims, NPY_COMPLEX128, 1);
    }

    if (lcmpq && n > 0) {
        npy_intp q_dims[2] = {2 * n, 2 * n};
        q_array = PyArray_ZEROS(2, q_dims, NPY_COMPLEX128, 1);
        if (q_array == NULL) {
            PyMem_Free(q_data);
            PyMem_Free(u_data);
            Py_DECREF(d_array);
            Py_DECREF(c_array);
            Py_DECREF(b_array);
            Py_DECREF(fg_array);
            Py_DECREF(alphar_array);
            Py_DECREF(alphai_array);
            Py_DECREF(beta_array);
            Py_DECREF(z_array);
            return NULL;
        }
        memcpy(PyArray_DATA((PyArrayObject*)q_array), q_data, 2 * n * 2 * n * sizeof(c128));
        PyMem_Free(q_data);
    } else {
        npy_intp q_dims[2] = {0, 0};
        q_array = PyArray_ZEROS(2, q_dims, NPY_COMPLEX128, 1);
        PyMem_Free(q_data);
    }

    if (lcmpu && n > 0) {
        npy_intp u_dims[2] = {n, 2 * n};
        u_array = PyArray_ZEROS(2, u_dims, NPY_COMPLEX128, 1);
        if (u_array == NULL) {
            PyMem_Free(u_data);
            Py_DECREF(q_array);
            Py_DECREF(d_array);
            Py_DECREF(c_array);
            Py_DECREF(b_array);
            Py_DECREF(fg_array);
            Py_DECREF(alphar_array);
            Py_DECREF(alphai_array);
            Py_DECREF(beta_array);
            Py_DECREF(z_array);
            return NULL;
        }
        memcpy(PyArray_DATA((PyArrayObject*)u_array), u_data, n * 2 * n * sizeof(c128));
        PyMem_Free(u_data);
    } else {
        npy_intp u_dims[2] = {0, 0};
        u_array = PyArray_ZEROS(2, u_dims, NPY_COMPLEX128, 1);
        PyMem_Free(u_data);
    }

    PyObject *result = Py_BuildValue("(OOOiOOOOOOOi)",
        (PyObject *)z_array, b_array, fg_array,
        (int)neig, d_array, c_array, q_array, u_array,
        alphar_array, alphai_array, beta_array, (int)info);

    Py_DECREF(z_array);
    Py_DECREF(b_array);
    Py_DECREF(fg_array);
    Py_DECREF(d_array);
    Py_DECREF(c_array);
    Py_DECREF(q_array);
    Py_DECREF(u_array);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);

    return result;
}

PyObject *py_mb03lz(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;

    const char *compq;
    const char *orth;
    int n;
    PyObject *a_obj, *de_obj, *b_obj, *fg_obj;

    static char *kwlist[] = {"compq", "orth", "n", "a", "de", "b", "fg", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssiOOOO", kwlist,
                                     &compq, &orth, &n, &a_obj, &de_obj,
                                     &b_obj, &fg_obj)) {
        return NULL;
    }

    i32 m = n / 2;
    i32 n2 = 2 * n;
    i32 nn = n * n;
    bool lcmpq = (compq[0] == 'C' || compq[0] == 'c');

    PyArrayObject *a_in_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);
    PyArrayObject *de_in_array = (PyArrayObject *)PyArray_FROM_OTF(
        de_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);
    PyArrayObject *b_in_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);
    PyArrayObject *fg_in_array = (PyArrayObject *)PyArray_FROM_OTF(
        fg_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);

    if (a_in_array == NULL || de_in_array == NULL ||
        b_in_array == NULL || fg_in_array == NULL) {
        Py_XDECREF(a_in_array);
        Py_XDECREF(de_in_array);
        Py_XDECREF(b_in_array);
        Py_XDECREF(fg_in_array);
        return NULL;
    }

    i32 lda_in = (m > 0) ? (i32)(PyArray_STRIDE(a_in_array, 1) / sizeof(c128)) : 1;
    i32 ldde_in = (m > 0) ? (i32)(PyArray_STRIDE(de_in_array, 1) / sizeof(c128)) : 1;
    i32 ldb_in = (m > 0) ? (i32)(PyArray_STRIDE(b_in_array, 1) / sizeof(c128)) : 1;
    i32 ldfg_in = (m > 0) ? (i32)(PyArray_STRIDE(fg_in_array, 1) / sizeof(c128)) : 1;

    c128 *a_in = (c128 *)PyArray_DATA(a_in_array);
    c128 *de_in = (c128 *)PyArray_DATA(de_in_array);
    c128 *b_in = (c128 *)PyArray_DATA(b_in_array);
    c128 *fg_in = (c128 *)PyArray_DATA(fg_in_array);

    i32 lda = (n > 0) ? n : 1;
    i32 ldde = (n > 0) ? n : 1;
    i32 ldb = (n > 0) ? n : 1;
    i32 ldfg = (n > 0) ? n : 1;
    i32 ldq = lcmpq ? (n > 0 ? n2 : 1) : 1;

    c128 *a = NULL;
    c128 *de = NULL;
    c128 *b = NULL;
    c128 *fg = NULL;
    c128 *q = NULL;

    if (n > 0) {
        a = (c128 *)PyMem_Calloc(lda * n, sizeof(c128));
        de = (c128 *)PyMem_Calloc(ldde * (n + 1), sizeof(c128));
        b = (c128 *)PyMem_Calloc(ldb * n, sizeof(c128));
        fg = (c128 *)PyMem_Calloc(ldfg * (n + 1), sizeof(c128));
        if (a == NULL || de == NULL || b == NULL || fg == NULL) {
            PyMem_Free(a); PyMem_Free(de); PyMem_Free(b); PyMem_Free(fg);
            Py_DECREF(a_in_array);
            Py_DECREF(de_in_array);
            Py_DECREF(b_in_array);
            Py_DECREF(fg_in_array);
            PyErr_NoMemory();
            return NULL;
        }
        for (i32 j = 0; j < m; j++) {
            for (i32 i = 0; i < m; i++) {
                a[i + j * lda] = a_in[i + j * lda_in];
                b[i + j * ldb] = b_in[i + j * ldb_in];
            }
        }
        for (i32 j = 0; j < m + 1; j++) {
            for (i32 i = 0; i < m; i++) {
                de[i + j * ldde] = de_in[i + j * ldde_in];
                fg[i + j * ldfg] = fg_in[i + j * ldfg_in];
            }
        }
    }

    Py_DECREF(a_in_array);
    Py_DECREF(de_in_array);
    Py_DECREF(b_in_array);
    Py_DECREF(fg_in_array);

    if (lcmpq && n > 0) {
        q = (c128 *)PyMem_Calloc(ldq * ldq, sizeof(c128));
        if (q == NULL) {
            PyMem_Free(a); PyMem_Free(de); PyMem_Free(b); PyMem_Free(fg);
            PyErr_NoMemory();
            return NULL;
        }
    }

    npy_intp eig_dims[1] = {n > 0 ? n : 1};
    PyObject *alphar_array = PyArray_ZEROS(1, eig_dims, NPY_DOUBLE, 0);
    PyObject *alphai_array = PyArray_ZEROS(1, eig_dims, NPY_DOUBLE, 0);
    PyObject *beta_array = PyArray_ZEROS(1, eig_dims, NPY_DOUBLE, 0);

    if (alphar_array == NULL || alphai_array == NULL || beta_array == NULL) {
        Py_XDECREF(alphar_array);
        Py_XDECREF(alphai_array);
        Py_XDECREF(beta_array);
        PyMem_Free(a); PyMem_Free(de); PyMem_Free(b); PyMem_Free(fg); PyMem_Free(q);
        return NULL;
    }

    f64 *alphar = (f64 *)PyArray_DATA((PyArrayObject *)alphar_array);
    f64 *alphai = (f64 *)PyArray_DATA((PyArrayObject *)alphai_array);
    f64 *beta_data = (f64 *)PyArray_DATA((PyArrayObject *)beta_array);

    i32 liwork = n + 1;
    i32 lbwork = lcmpq ? (n > 0 ? n : 1) : 1;
    i32 ldwork, lzwork;
    if (n == 0) {
        ldwork = 1;
        lzwork = 1;
    } else if (lcmpq) {
        ldwork = 11 * nn + n2;
        lzwork = 8 * n + 4;
    } else {
        ldwork = 4 * nn + n2 + (3 > n ? 3 : n);
        lzwork = 1;
    }

    i32 *iwork = (i32 *)PyMem_Calloc(liwork > 1 ? liwork : 1, sizeof(i32));
    f64 *dwork = (f64 *)PyMem_Calloc(ldwork > 1 ? ldwork : 1, sizeof(f64));
    c128 *zwork = (c128 *)PyMem_Calloc(lzwork > 1 ? lzwork : 1, sizeof(c128));
    bool *bwork = (bool *)PyMem_Calloc(lbwork > 1 ? lbwork : 1, sizeof(bool));

    if (iwork == NULL || dwork == NULL || zwork == NULL || bwork == NULL) {
        PyMem_Free(iwork);
        PyMem_Free(dwork);
        PyMem_Free(zwork);
        PyMem_Free(bwork);
        Py_DECREF(alphar_array);
        Py_DECREF(alphai_array);
        Py_DECREF(beta_array);
        PyMem_Free(a); PyMem_Free(de); PyMem_Free(b); PyMem_Free(fg); PyMem_Free(q);
        PyErr_NoMemory();
        return NULL;
    }

    i32 neig = 0;
    i32 info = 0;

    mb03lz(compq, orth, n, a, lda, de, ldde, b, ldb, fg, ldfg,
           &neig, q, ldq, alphar, alphai, beta_data,
           iwork, dwork, ldwork, zwork, lzwork, bwork, &info);

    PyMem_Free(iwork);
    PyMem_Free(dwork);
    PyMem_Free(zwork);
    PyMem_Free(bwork);

    PyObject *a_out_obj = NULL;
    PyObject *de_out_obj = NULL;
    PyObject *b_out_obj = NULL;
    PyObject *fg_out_obj = NULL;
    PyObject *q_out = NULL;

    if (lcmpq && n > 0) {
        npy_intp a_dims[2] = {n, n};
        a_out_obj = PyArray_ZEROS(2, a_dims, NPY_COMPLEX128, 1);
        memcpy(PyArray_DATA((PyArrayObject *)a_out_obj), a, n * n * sizeof(c128));
        PyMem_Free(a);

        npy_intp de_dims[2] = {n, n + 1};
        de_out_obj = PyArray_ZEROS(2, de_dims, NPY_COMPLEX128, 1);
        memcpy(PyArray_DATA((PyArrayObject *)de_out_obj), de, n * (n + 1) * sizeof(c128));
        PyMem_Free(de);

        npy_intp b_dims[2] = {n, n};
        b_out_obj = PyArray_ZEROS(2, b_dims, NPY_COMPLEX128, 1);
        memcpy(PyArray_DATA((PyArrayObject *)b_out_obj), b, n * n * sizeof(c128));
        PyMem_Free(b);

        npy_intp fg_dims[2] = {n, n + 1};
        fg_out_obj = PyArray_ZEROS(2, fg_dims, NPY_COMPLEX128, 1);
        memcpy(PyArray_DATA((PyArrayObject *)fg_out_obj), fg, n * (n + 1) * sizeof(c128));
        PyMem_Free(fg);

        if (neig > 0) {
            npy_intp q_out_dims[2] = {n, neig};
            q_out = PyArray_ZEROS(2, q_out_dims, NPY_COMPLEX128, 1);
            if (q_out == NULL) {
                Py_DECREF(alphar_array);
                Py_DECREF(alphai_array);
                Py_DECREF(beta_array);
                Py_XDECREF(a_out_obj);
                Py_XDECREF(de_out_obj);
                Py_XDECREF(b_out_obj);
                Py_XDECREF(fg_out_obj);
                PyMem_Free(q);
                PyErr_NoMemory();
                return NULL;
            }
            c128 *q_out_data = (c128 *)PyArray_DATA((PyArrayObject *)q_out);
            for (i32 j = 0; j < neig; j++) {
                for (i32 i = 0; i < n; i++) {
                    q_out_data[i + j * n] = q[i + j * ldq];
                }
            }
            PyMem_Free(q);
        } else {
            npy_intp q_out_dims[2] = {n, 0};
            q_out = PyArray_ZEROS(2, q_out_dims, NPY_COMPLEX128, 1);
            PyMem_Free(q);
        }
    } else {
        npy_intp a_dims[2] = {m > 0 ? m : 1, m > 0 ? m : 1};
        a_out_obj = PyArray_ZEROS(2, a_dims, NPY_COMPLEX128, 1);
        npy_intp de_dims[2] = {m > 0 ? m : 1, m > 0 ? m + 1 : 1};
        de_out_obj = PyArray_ZEROS(2, de_dims, NPY_COMPLEX128, 1);
        npy_intp b_dims[2] = {m > 0 ? m : 1, m > 0 ? m : 1};
        b_out_obj = PyArray_ZEROS(2, b_dims, NPY_COMPLEX128, 1);
        npy_intp fg_dims[2] = {m > 0 ? m : 1, m > 0 ? m + 1 : 1};
        fg_out_obj = PyArray_ZEROS(2, fg_dims, NPY_COMPLEX128, 1);
        npy_intp q_out_dims[2] = {n > 0 ? n : 1, 0};
        q_out = PyArray_ZEROS(2, q_out_dims, NPY_COMPLEX128, 1);
        PyMem_Free(a); PyMem_Free(de); PyMem_Free(b); PyMem_Free(fg);
    }

    PyObject *result = Py_BuildValue("(OOOOOOOOii)",
        a_out_obj, de_out_obj, b_out_obj, fg_out_obj, q_out,
        alphar_array, alphai_array, beta_array,
        (int)neig, (int)info);

    Py_DECREF(a_out_obj);
    Py_DECREF(de_out_obj);
    Py_DECREF(b_out_obj);
    Py_DECREF(fg_out_obj);
    Py_DECREF(q_out);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);

    return result;
}


PyObject* py_mb03pd(PyObject* self, PyObject* args) {
    const char *jobrq;
    i32 m, n;
    f64 rcond, svlmax;
    PyObject *a_obj;

    if (!PyArg_ParseTuple(args, "siiOdd", &jobrq, &m, &n, &a_obj, &rcond, &svlmax)) {
        return NULL;
    }

    if (m < 0 || n < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative (m=%d, n=%d)", m, n);
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda = (i32)a_dims[0];
    if (lda < 1) lda = 1;

    i32 mn = (m < n) ? m : n;

    bool ljobrq = (jobrq[0] == 'R' || jobrq[0] == 'r');
    i32 dwork_size = ljobrq ? (3 * m) : (3 * mn);
    if (dwork_size < 1) dwork_size = 1;
    f64 *dwork = (f64*)malloc(dwork_size * sizeof(f64));

    if (!dwork) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate work arrays");
        return NULL;
    }

    npy_intp sval_dims[1] = {3};
    npy_intp jpvt_dims[1] = {m > 0 ? m : 0};
    npy_intp tau_dims[1] = {mn};

    PyObject *sval_array = PyArray_SimpleNew(1, sval_dims, NPY_DOUBLE);
    PyObject *jpvt_array = (m > 0) ? PyArray_SimpleNew(1, jpvt_dims, NPY_INT32)
                                   : PyArray_EMPTY(1, jpvt_dims, NPY_INT32, 0);
    PyObject *tau_array = (mn > 0) ? PyArray_SimpleNew(1, tau_dims, NPY_DOUBLE)
                                   : PyArray_EMPTY(1, tau_dims, NPY_DOUBLE, 0);

    if (!sval_array || !jpvt_array || !tau_array) {
        free(dwork);
        Py_XDECREF(sval_array);
        Py_XDECREF(jpvt_array);
        Py_XDECREF(tau_array);
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output arrays");
        return NULL;
    }

    f64 *sval = (f64*)PyArray_DATA((PyArrayObject*)sval_array);
    i32 *jpvt = (m > 0) ? (i32*)PyArray_DATA((PyArrayObject*)jpvt_array) : NULL;
    f64 *tau = (mn > 0) ? (f64*)PyArray_DATA((PyArrayObject*)tau_array) : NULL;

    if (m > 0) memset(jpvt, 0, m * sizeof(i32));

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    i32 rank = 0, info = 0;

    mb03pd(jobrq, m, n, a_data, lda, jpvt, rcond, svlmax, tau, &rank, sval, dwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);

    PyObject *result = Py_BuildValue("(OiiOOO)", a_array, rank, info,
                                     sval_array, jpvt_array, tau_array);

    Py_DECREF(a_array);
    Py_DECREF(sval_array);
    Py_DECREF(jpvt_array);
    Py_DECREF(tau_array);

    return result;
}

/**
 * Python wrapper for MB03LD
 */
PyObject* py_mb03ld(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"compq", "orth", "n", "a", "de", "b", "fg", NULL};

    const char *compq, *orth;
    int n;
    PyObject *a_obj, *de_obj, *b_obj, *fg_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssiOOOO", kwlist,
                                     &compq, &orth, &n, &a_obj, &de_obj, &b_obj, &fg_obj)) {
        return NULL;
    }

    char c_compq = (char)toupper((unsigned char)compq[0]);
    char c_orth = (char)toupper((unsigned char)orth[0]);

    bool liniq = (c_compq == 'C');
    bool valid_orth = (c_orth == 'P' || c_orth == 'S' || c_orth == 'Q');

    if (c_compq != 'N' && c_compq != 'C') {
        PyErr_SetString(PyExc_ValueError, "compq must be 'N' or 'C'");
        return NULL;
    }
    if (liniq && !valid_orth) {
        PyErr_SetString(PyExc_ValueError, "orth must be 'P', 'S', or 'Q' when compq='C'");
        return NULL;
    }
    if (n < 0 || (n % 2) != 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative and even");
        return NULL;
    }

    i32 m = n / 2;
    i32 n2 = n * 2;
    i32 nn = n * n;
    i32 mm = m * m;

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *de_array = (PyArrayObject*)PyArray_FROM_OTF(
        de_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *fg_array = (PyArrayObject*)PyArray_FROM_OTF(
        fg_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!a_array || !de_array || !b_array || !fg_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(de_array);
        Py_XDECREF(b_array);
        Py_XDECREF(fg_array);
        return NULL;
    }

    i32 lda = (m > 1) ? m : 1;
    i32 ldde = (m > 1) ? m : 1;
    i32 ldb = (m > 1) ? m : 1;
    i32 ldfg = (m > 1) ? m : 1;
    i32 ldq = liniq ? n2 : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *de_data = (f64*)PyArray_DATA(de_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *fg_data = (f64*)PyArray_DATA(fg_array);

    i32 miniw, mindw;
    if (n == 0) {
        miniw = 1;
        mindw = 1;
    } else if (liniq) {
        miniw = (32 > n2 + 3) ? 32 : n2 + 3;
        i32 tmp = 8 * n + 32;
        mindw = 8 * nn + ((tmp > 272) ? tmp : 272);
    } else {
        i32 j;
        if ((m % 2) == 0) {
            i32 tmp = 4 * n;
            j = ((tmp > 32) ? tmp : 32) + 4;
        } else {
            i32 tmp = 4 * n;
            j = (tmp > 36) ? tmp : 36;
        }
        miniw = (n + 12 > n2 + 3) ? n + 12 : n2 + 3;
        mindw = 3 * mm + nn + j;
    }

    npy_intp alphar_dims[1] = {m > 0 ? m : 0};
    npy_intp alphai_dims[1] = {m > 0 ? m : 0};
    npy_intp beta_dims[1] = {m > 0 ? m : 0};

    PyObject *alphar_array = PyArray_SimpleNew(1, alphar_dims, NPY_DOUBLE);
    PyObject *alphai_array = PyArray_SimpleNew(1, alphai_dims, NPY_DOUBLE);
    PyObject *beta_array = PyArray_SimpleNew(1, beta_dims, NPY_DOUBLE);

    f64 *alphar = (f64*)PyArray_DATA((PyArrayObject*)alphar_array);
    f64 *alphai = (f64*)PyArray_DATA((PyArrayObject*)alphai_array);
    f64 *beta = (f64*)PyArray_DATA((PyArrayObject*)beta_array);
    memset(alphar, 0, PyArray_NBYTES((PyArrayObject*)alphar_array));
    memset(alphai, 0, PyArray_NBYTES((PyArrayObject*)alphai_array));
    memset(beta, 0, PyArray_NBYTES((PyArrayObject*)beta_array));

    i32 *iwork = (i32*)calloc(miniw, sizeof(i32));
    f64 *dwork = (f64*)calloc(mindw, sizeof(f64));
    i32 *bwork = (i32*)calloc(m > 0 ? m : 1, sizeof(i32));
 
    npy_intp q_dims[2] = {ldq, n2};
    PyObject *q_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE,
                                    NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    f64 *q_data = (f64*)PyArray_DATA((PyArrayObject*)q_array);
    if (q_array) memset(q_data, 0, PyArray_NBYTES((PyArrayObject*)q_array));

    if (!alphar_array || !alphai_array || !beta_array || !iwork || !dwork || !bwork || !q_array) {
        Py_XDECREF(alphar_array);
        Py_XDECREF(alphai_array);
        Py_XDECREF(beta_array);
        free(iwork); free(dwork); free(bwork);
        Py_XDECREF(q_array);
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(b_array);
        Py_DECREF(fg_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 neig = 0, info = 0;

    mb03ld(compq, orth, n, a_data, lda, de_data, ldde, b_data, ldb, fg_data, ldfg,
           &neig, q_data, ldq, alphar, alphai, beta, iwork, miniw, dwork, mindw, bwork, &info);

    free(iwork);
    free(dwork);
    free(bwork);

    // Arrays are already allocated and populated
 
    PyObject *result = Py_BuildValue("(OOOOiOOOOi)",
                                     a_array, de_array, b_array, fg_array, neig,
                                     q_array, alphar_array, alphai_array, beta_array, info);
 
    Py_DECREF(a_array);
    Py_DECREF(de_array);
    Py_DECREF(b_array);
    Py_DECREF(fg_array);
    Py_DECREF(q_array);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);

    return result;
}

PyObject *py_mb03lp(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;

    const char *compq;
    const char *orth;
    PyObject *a_obj, *de_obj, *b_obj, *fg_obj;

    static char *kwlist[] = {"compq", "orth", "a", "de", "b", "fg", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssOOOO", kwlist,
                                     &compq, &orth, &a_obj, &de_obj,
                                     &b_obj, &fg_obj)) {
        return NULL;
    }

    PyArrayObject *a_in_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    PyArrayObject *de_in_array = (PyArrayObject *)PyArray_FROM_OTF(
        de_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    PyArrayObject *b_in_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    PyArrayObject *fg_in_array = (PyArrayObject *)PyArray_FROM_OTF(
        fg_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);

    if (a_in_array == NULL || de_in_array == NULL ||
        b_in_array == NULL || fg_in_array == NULL) {
        Py_XDECREF(a_in_array);
        Py_XDECREF(de_in_array);
        Py_XDECREF(b_in_array);
        Py_XDECREF(fg_in_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_in_array);
    i32 m = (i32)a_dims[0];
    i32 n = 2 * m;
    i32 n2 = 2 * n;
    bool lcmpq = (compq[0] == 'C' || compq[0] == 'c');

    i32 lda_in = (m > 0) ? (i32)(PyArray_STRIDE(a_in_array, 1) / sizeof(f64)) : 1;
    i32 ldde_in = (m > 0) ? (i32)(PyArray_STRIDE(de_in_array, 1) / sizeof(f64)) : 1;
    i32 ldb_in = (m > 0) ? (i32)(PyArray_STRIDE(b_in_array, 1) / sizeof(f64)) : 1;
    i32 ldfg_in = (m > 0) ? (i32)(PyArray_STRIDE(fg_in_array, 1) / sizeof(f64)) : 1;

    f64 *a_in = (f64 *)PyArray_DATA(a_in_array);
    f64 *de_in = (f64 *)PyArray_DATA(de_in_array);
    f64 *b_in = (f64 *)PyArray_DATA(b_in_array);
    f64 *fg_in = (f64 *)PyArray_DATA(fg_in_array);

    i32 lda = (m > 0) ? m : 1;
    i32 ldde = (m > 0) ? m : 1;
    i32 ldb = (m > 0) ? m : 1;
    i32 ldfg = (m > 0) ? m : 1;
    i32 ldq = lcmpq ? (n2 > 0 ? n2 : 1) : 1;

    npy_intp a_dims2[2] = {m > 0 ? m : 0, m > 0 ? m : 0};
    npy_intp a_strides[2] = {sizeof(f64), lda * sizeof(f64)};
    npy_intp de_dims2[2] = {m > 0 ? m : 0, m > 0 ? m + 1 : 0};
    npy_intp de_strides[2] = {sizeof(f64), ldde * sizeof(f64)};

    PyObject *a_out_obj = NULL;
    PyObject *de_out_obj = NULL;
    PyObject *b_out_obj = NULL;
    PyObject *fg_out_obj = NULL;
    PyObject *q_out = NULL;
    f64 *a = NULL;
    f64 *de = NULL;
    f64 *b = NULL;
    f64 *fg = NULL;
    f64 *q = NULL;

    if (m > 0) {
        a_out_obj = PyArray_New(&PyArray_Type, 2, a_dims2, NPY_DOUBLE, a_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        de_out_obj = PyArray_New(&PyArray_Type, 2, de_dims2, NPY_DOUBLE, de_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        b_out_obj = PyArray_New(&PyArray_Type, 2, a_dims2, NPY_DOUBLE, a_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        fg_out_obj = PyArray_New(&PyArray_Type, 2, de_dims2, NPY_DOUBLE, de_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (a_out_obj == NULL || de_out_obj == NULL || b_out_obj == NULL || fg_out_obj == NULL) {
            Py_XDECREF(a_out_obj); Py_XDECREF(de_out_obj); Py_XDECREF(b_out_obj); Py_XDECREF(fg_out_obj);
            Py_DECREF(a_in_array);
            Py_DECREF(de_in_array);
            Py_DECREF(b_in_array);
            Py_DECREF(fg_in_array);
            PyErr_NoMemory();
            return NULL;
        }
        a = (f64 *)PyArray_DATA((PyArrayObject *)a_out_obj);
        de = (f64 *)PyArray_DATA((PyArrayObject *)de_out_obj);
        b = (f64 *)PyArray_DATA((PyArrayObject *)b_out_obj);
        fg = (f64 *)PyArray_DATA((PyArrayObject *)fg_out_obj);
        memset(a, 0, lda * m * sizeof(f64));
        memset(de, 0, ldde * (m + 1) * sizeof(f64));
        memset(b, 0, ldb * m * sizeof(f64));
        memset(fg, 0, ldfg * (m + 1) * sizeof(f64));
        for (i32 j = 0; j < m; j++) {
            for (i32 i = 0; i < m; i++) {
                a[i + j * lda] = a_in[i + j * lda_in];
                b[i + j * ldb] = b_in[i + j * ldb_in];
            }
        }
        for (i32 j = 0; j < m + 1; j++) {
            for (i32 i = 0; i < m; i++) {
                de[i + j * ldde] = de_in[i + j * ldde_in];
                fg[i + j * ldfg] = fg_in[i + j * ldfg_in];
            }
        }
    } else {
        a_out_obj = PyArray_ZEROS(2, a_dims2, NPY_DOUBLE, 1);
        de_out_obj = PyArray_ZEROS(2, de_dims2, NPY_DOUBLE, 1);
        b_out_obj = PyArray_ZEROS(2, a_dims2, NPY_DOUBLE, 1);
        fg_out_obj = PyArray_ZEROS(2, de_dims2, NPY_DOUBLE, 1);
    }

    Py_DECREF(a_in_array);
    Py_DECREF(de_in_array);
    Py_DECREF(b_in_array);
    Py_DECREF(fg_in_array);

    if (lcmpq && n2 > 0) {
        q = (f64 *)calloc(ldq * n2, sizeof(f64));
        if (q == NULL) {
            Py_DECREF(a_out_obj); Py_DECREF(de_out_obj); Py_DECREF(b_out_obj); Py_DECREF(fg_out_obj);
            PyErr_NoMemory();
            return NULL;
        }
    }

    npy_intp eig_dims[1] = {m > 0 ? m : 0};
    PyObject *alphar_array = PyArray_ZEROS(1, eig_dims, NPY_DOUBLE, 0);
    PyObject *alphai_array = PyArray_ZEROS(1, eig_dims, NPY_DOUBLE, 0);
    PyObject *beta_array = PyArray_ZEROS(1, eig_dims, NPY_DOUBLE, 0);

    if (alphar_array == NULL || alphai_array == NULL || beta_array == NULL) {
        Py_XDECREF(alphar_array);
        Py_XDECREF(alphai_array);
        Py_XDECREF(beta_array);
        Py_DECREF(a_out_obj); Py_DECREF(de_out_obj); Py_DECREF(b_out_obj); Py_DECREF(fg_out_obj);
        free(q);
        return NULL;
    }

    f64 *alphar = (f64 *)PyArray_DATA((PyArrayObject *)alphar_array);
    f64 *alphai = (f64 *)PyArray_DATA((PyArrayObject *)alphai_array);
    f64 *beta_data = (f64 *)PyArray_DATA((PyArrayObject *)beta_array);

    i32 liwork, ldwork;
    if (n == 0) {
        liwork = 1;
        ldwork = 1;
    } else if (lcmpq) {
        liwork = (32 > 3 * n2 - 3) ? 32 : (3 * n2 - 3);
        i32 nn = n * n;
        ldwork = 8 * nn + ((13 * n + 32) > 272 ? (13 * n + 32) : 272);
    } else {
        i32 l;
        if ((m % 2) == 0) {
            l = ((4 * n) > 32 ? (4 * n) : 32) + 4;
        } else {
            l = (4 * n) > 36 ? (4 * n) : 36;
        }
        liwork = ((n + 12) > (n2 + 3)) ? (n + 12) : (n2 + 3);
        ldwork = 3 * m * m + n * n + l;
    }

    i32 lbwork = m > 0 ? m : 1;
    i32 *iwork = (i32 *)calloc(liwork > 1 ? liwork : 1, sizeof(i32));
    f64 *dwork = (f64 *)calloc(ldwork > 1 ? ldwork : 1, sizeof(f64));
    i32 *bwork = (i32 *)calloc(lbwork, sizeof(i32));

    if (iwork == NULL || dwork == NULL || bwork == NULL) {
        free(iwork);
        free(dwork);
        free(bwork);
        Py_DECREF(alphar_array);
        Py_DECREF(alphai_array);
        Py_DECREF(beta_array);
        Py_DECREF(a_out_obj); Py_DECREF(de_out_obj); Py_DECREF(b_out_obj); Py_DECREF(fg_out_obj);
        free(q);
        PyErr_NoMemory();
        return NULL;
    }

    i32 neig = 0;
    i32 info = 0;

    mb03lp(compq, orth, n, a, lda, de, ldde, b, ldb, fg, ldfg,
           &neig, q, ldq, alphar, alphai, beta_data,
           iwork, liwork, dwork, ldwork, bwork, &info);

    free(iwork);
    free(dwork);
    free(bwork);

    if (lcmpq && neig > 0 && n2 > 0) {
        npy_intp q_out_dims[2] = {n2, neig};
        npy_intp q_strides2[2] = {sizeof(f64), ldq * sizeof(f64)};
        q_out = PyArray_New(&PyArray_Type, 2, q_out_dims, NPY_DOUBLE, q_strides2, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (q_out == NULL) {
            Py_DECREF(a_out_obj); Py_DECREF(de_out_obj); Py_DECREF(b_out_obj); Py_DECREF(fg_out_obj);
            Py_DECREF(alphar_array); Py_DECREF(alphai_array); Py_DECREF(beta_array);
            free(q);
            PyErr_NoMemory();
            return NULL;
        }
        f64 *q_data = (f64 *)PyArray_DATA((PyArrayObject *)q_out);
        for (i32 j = 0; j < neig; j++) {
            for (i32 i = 0; i < n2; i++) {
                q_data[i + j * n2] = q[i + j * ldq];
            }
        }
        free(q);
    } else {
        q_out = Py_None;
        Py_INCREF(Py_None);
        free(q);
    }

    PyObject *result = Py_BuildValue("(OOOOiOOOOi)",
        a_out_obj, de_out_obj, b_out_obj, fg_out_obj, (int)neig, q_out,
        alphar_array, alphai_array, beta_array, (int)info);

    Py_DECREF(a_out_obj);
    Py_DECREF(de_out_obj);
    Py_DECREF(b_out_obj);
    Py_DECREF(fg_out_obj);
    Py_DECREF(q_out);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);

    return result;
}

/*
 * MB03XS: Eigenvalues and real skew-Hamiltonian Schur form
 *
 * Computes eigenvalues and real skew-Hamiltonian Schur form of
 * W = [[A, G], [Q, A^T]] where G, Q are skew-symmetric.
 */
PyObject *py_mb03xs(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *a_obj = NULL, *qg_obj = NULL;
    char *jobu_str = "N";

    static char *kwlist[] = {"a", "qg", "jobu", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|s", kwlist,
                                     &a_obj, &qg_obj, &jobu_str)) {
        return NULL;
    }

    char jobu = jobu_str[0];
    bool compu = (jobu == 'U' || jobu == 'u');

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *qg_array = (PyArrayObject *)PyArray_FROM_OTF(
        qg_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!a_array || !qg_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(qg_array);
        PyErr_SetString(PyExc_ValueError, "Failed to convert input arrays");
        return NULL;
    }

    if (PyArray_NDIM(a_array) != 2 || PyArray_NDIM(qg_array) != 2) {
        Py_DECREF(a_array);
        Py_DECREF(qg_array);
        PyErr_SetString(PyExc_ValueError, "a and qg must be 2-dimensional");
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 lda = (n > 1) ? n : 1;
    i32 ldqg = (n > 1) ? n : 1;
    i32 ldu1, ldu2;

    if (compu) {
        ldu1 = (n > 1) ? n : 1;
        ldu2 = (n > 1) ? n : 1;
    } else {
        ldu1 = 1;
        ldu2 = 1;
    }

    i32 nn = n * n;
    i32 ldwork;
    if (compu) {
        ldwork = (nn + 5 * n > 1) ? nn + 5 * n : 1;
    } else {
        i32 tmp1 = 5 * n;
        i32 tmp2 = nn + n;
        ldwork = (tmp1 > tmp2) ? tmp1 : tmp2;
        if (ldwork < 1) ldwork = 1;
    }
    ldwork = (ldwork > 2 * nn) ? ldwork : 2 * nn;

    npy_intp a_dims[2] = {n, n};
    npy_intp a_strides[2] = {sizeof(f64), lda * sizeof(f64)};
    npy_intp qg_dims[2] = {n, n + 1};
    npy_intp qg_strides[2] = {sizeof(f64), ldqg * sizeof(f64)};
    npy_intp wr_dims[1] = {n > 0 ? n : 0};
    npy_intp u_dims[2] = {n, n};
    npy_intp u_strides[2] = {sizeof(f64), ldu1 * sizeof(f64)};

    PyObject *a_out_obj = NULL;
    PyObject *qg_out_obj = NULL;
    PyObject *u1_out_obj = NULL;
    PyObject *u2_out_obj = NULL;
    PyObject *wr_obj = NULL;
    PyObject *wi_obj = NULL;

    f64 *a_out = NULL;
    f64 *qg_out = NULL;
    f64 *u1 = NULL;
    f64 *u2 = NULL;
    f64 *wr = NULL;
    f64 *wi = NULL;

    if (n > 0) {
        a_out_obj = PyArray_New(&PyArray_Type, 2, a_dims, NPY_DOUBLE, a_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        qg_out_obj = PyArray_New(&PyArray_Type, 2, qg_dims, NPY_DOUBLE, qg_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        wr_obj = PyArray_New(&PyArray_Type, 1, wr_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
        wi_obj = PyArray_New(&PyArray_Type, 1, wr_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);

        if (!a_out_obj || !qg_out_obj || !wr_obj || !wi_obj) {
            Py_XDECREF(a_out_obj); Py_XDECREF(qg_out_obj); Py_XDECREF(wr_obj); Py_XDECREF(wi_obj);
            Py_DECREF(a_array);
            Py_DECREF(qg_array);
            PyErr_NoMemory();
            return NULL;
        }

        a_out = (f64 *)PyArray_DATA((PyArrayObject *)a_out_obj);
        qg_out = (f64 *)PyArray_DATA((PyArrayObject *)qg_out_obj);
        wr = (f64 *)PyArray_DATA((PyArrayObject *)wr_obj);
        wi = (f64 *)PyArray_DATA((PyArrayObject *)wi_obj);
        memset(a_out, 0, lda * n * sizeof(f64));
        memset(qg_out, 0, ldqg * (n + 1) * sizeof(f64));
        memset(wr, 0, n * sizeof(f64));
        memset(wi, 0, n * sizeof(f64));

        if (compu) {
            u1_out_obj = PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE, u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            u2_out_obj = PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE, u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            if (!u1_out_obj || !u2_out_obj) {
                Py_XDECREF(u1_out_obj); Py_XDECREF(u2_out_obj);
                Py_DECREF(a_out_obj); Py_DECREF(qg_out_obj); Py_DECREF(wr_obj); Py_DECREF(wi_obj);
                Py_DECREF(a_array);
                Py_DECREF(qg_array);
                PyErr_NoMemory();
                return NULL;
            }
            u1 = (f64 *)PyArray_DATA((PyArrayObject *)u1_out_obj);
            u2 = (f64 *)PyArray_DATA((PyArrayObject *)u2_out_obj);
            memset(u1, 0, ldu1 * n * sizeof(f64));
            memset(u2, 0, ldu2 * n * sizeof(f64));
        }
    } else {
        a_out_obj = PyArray_ZEROS(2, a_dims, NPY_DOUBLE, 1);
        qg_out_obj = PyArray_ZEROS(2, qg_dims, NPY_DOUBLE, 1);
        wr_obj = PyArray_ZEROS(1, wr_dims, NPY_DOUBLE, 1);
        wi_obj = PyArray_ZEROS(1, wr_dims, NPY_DOUBLE, 1);
        if (compu) {
            u1_out_obj = PyArray_ZEROS(2, u_dims, NPY_DOUBLE, 1);
            u2_out_obj = PyArray_ZEROS(2, u_dims, NPY_DOUBLE, 1);
        }
    }

    f64 *dwork = (f64 *)calloc(ldwork > 0 ? ldwork : 1, sizeof(f64));
    if (!dwork) {
        Py_DECREF(a_out_obj); Py_DECREF(qg_out_obj); Py_DECREF(wr_obj); Py_DECREF(wi_obj);
        Py_XDECREF(u1_out_obj); Py_XDECREF(u2_out_obj);
        Py_DECREF(a_array);
        Py_DECREF(qg_array);
        PyErr_NoMemory();
        return NULL;
    }

    if (n > 0) {
        f64 *a_in = (f64 *)PyArray_DATA(a_array);
        f64 *qg_in = (f64 *)PyArray_DATA(qg_array);

        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < n; i++) {
                a_out[i + j * lda] = a_in[i + j * lda];
            }
        }
        for (i32 j = 0; j < n + 1; j++) {
            for (i32 i = 0; i < n; i++) {
                qg_out[i + j * ldqg] = qg_in[i + j * ldqg];
            }
        }
    }

    i32 info = 0;
    char jobu_c[2] = {jobu, '\0'};

    mb03xs(jobu_c, n, a_out, lda, qg_out, ldqg,
           u1, ldu1, u2, ldu2, wr, wi, dwork, ldwork, &info);

    free(dwork);

    PyObject *result;
    if (compu) {
        result = Py_BuildValue("(OOOOOOi)",
            a_out_obj, qg_out_obj, u1_out_obj, u2_out_obj, wr_obj, wi_obj, (int)info);

        Py_DECREF(u1_out_obj);
        Py_DECREF(u2_out_obj);
    } else {
        result = Py_BuildValue("(OOOOi)",
            a_out_obj, qg_out_obj, wr_obj, wi_obj, (int)info);
    }

    Py_DECREF(a_array);
    Py_DECREF(qg_array);
    Py_DECREF(a_out_obj);
    Py_DECREF(qg_out_obj);
    Py_DECREF(wr_obj);
    Py_DECREF(wi_obj);

    return result;
}

/* MB03XZ: Eigenvalues of complex Hamiltonian matrix */
PyObject* py_mb03xz(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *a_obj = NULL, *qg_obj = NULL;
    const char *balanc = "N", *job = "E", *jobu = "N";

    static char *kwlist[] = {"a", "qg", "balanc", "job", "jobu", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|sss", kwlist,
                                      &a_obj, &qg_obj, &balanc, &job, &jobu)) {
        return NULL;
    }

    char balanc_up = (char)toupper((unsigned char)balanc[0]);
    char job_up = (char)toupper((unsigned char)job[0]);
    char jobu_up = (char)toupper((unsigned char)jobu[0]);

    if (balanc_up != 'N' && balanc_up != 'P' && balanc_up != 'S' && balanc_up != 'B') {
        PyErr_SetString(PyExc_ValueError, "balanc must be 'N', 'P', 'S', or 'B'");
        return NULL;
    }
    if (job_up != 'E' && job_up != 'S' && job_up != 'G') {
        PyErr_SetString(PyExc_ValueError, "job must be 'E', 'S', or 'G'");
        return NULL;
    }
    if (jobu_up != 'N' && jobu_up != 'U') {
        PyErr_SetString(PyExc_ValueError, "jobu must be 'N' or 'U'");
        return NULL;
    }

    bool wants = (job_up == 'S') || (job_up == 'G');
    bool wantg = (job_up == 'G');
    bool wantu = (jobu_up == 'U');
    bool wantus = wants && wantu;

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_COMPLEX128, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *qg_array = (PyArrayObject *)PyArray_FROM_OTF(
        qg_obj, NPY_COMPLEX128, NPY_ARRAY_IN_FARRAY);

    if (!a_array || !qg_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(qg_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 n2 = 2 * n;

    i32 k = wants ? n2 : n;
    i32 lda = k > 1 ? k : 1;
    i32 ldqg = k > 1 ? k : 1;
    i32 ldu1 = wantus ? n2 : 1;
    i32 ldu2 = wantus ? n2 : 1;

    i32 nn = n * n;

    i32 mindw, minzw;
    if (n == 0) {
        mindw = 2;
    } else if (wantu) {
        if (wants) {
            i32 m1 = 20 * nn + 12 * n;
            mindw = m1 > 2 ? m1 : 2;
        } else {
            mindw = 20 * nn + 12 * n + 2;
        }
    } else {
        if (wants) {
            i32 m1 = 12 * nn + 4 * n;
            i32 m2 = 8 * nn + 12 * n;
            i32 mx = m1 > m2 ? m1 : m2;
            mindw = mx > 2 ? mx : 2;
        } else {
            i32 m1 = 12 * nn + 4 * n;
            i32 m2 = 8 * nn + 12 * n;
            mindw = (m1 > m2 ? m1 : m2) + 2;
        }
    }

    if (wantg || wantu) {
        minzw = 12 * n - 2 > 1 ? 12 * n - 2 : 1;
    } else if (wants) {
        minzw = 12 * n - 6 > 1 ? 12 * n - 6 : 1;
    } else {
        minzw = 1;
    }

    c128 *a_out = (c128 *)calloc(lda * (wants ? n2 : n) + 1, sizeof(c128));
    c128 *qg_out = (c128 *)calloc(ldqg * (wantg ? n2 : n + 1) + 1, sizeof(c128));
    c128 *u1 = wantus ? (c128 *)calloc(ldu1 * n2, sizeof(c128)) : NULL;
    c128 *u2 = wantus ? (c128 *)calloc(ldu2 * n2, sizeof(c128)) : NULL;
    f64 *wr = (f64 *)calloc(n2 + 1, sizeof(f64));
    f64 *wi = (f64 *)calloc(n2 + 1, sizeof(f64));
    f64 *scale = (f64 *)calloc(n + 1, sizeof(f64));
    f64 *dwork = (f64 *)calloc(mindw + 1, sizeof(f64));
    c128 *zwork = (c128 *)calloc(minzw + 1, sizeof(c128));
    bool *bwork = (bool *)calloc(wants && n > 0 ? (2 * n - 1) : 1, sizeof(bool));

    if (!a_out || !qg_out || !wr || !wi || !scale || !dwork || !zwork || !bwork ||
        (wantus && (!u1 || !u2))) {
        free(a_out); free(qg_out); free(u1); free(u2);
        free(wr); free(wi); free(scale); free(dwork); free(zwork); free(bwork);
        Py_DECREF(a_array);
        Py_DECREF(qg_array);
        PyErr_NoMemory();
        return NULL;
    }

    c128 *a_in = (c128 *)PyArray_DATA(a_array);
    c128 *qg_in = (c128 *)PyArray_DATA(qg_array);
    i32 lda_in = (i32)PyArray_DIM(a_array, 0);
    i32 ldqg_in = (i32)PyArray_DIM(qg_array, 0);

    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < n; i++) {
            a_out[i + j * lda] = a_in[i + j * lda_in];
        }
    }
    i32 qg_ncols = (i32)PyArray_DIM(qg_array, 1);
    for (i32 j = 0; j < qg_ncols; j++) {
        for (i32 i = 0; i < n; i++) {
            qg_out[i + j * ldqg] = qg_in[i + j * ldqg_in];
        }
    }

    i32 ilo, info;
    mb03xz(balanc, job, jobu, n, a_out, lda, qg_out, ldqg,
           u1, ldu1, u2, ldu2, wr, wi, &ilo, scale,
           dwork, mindw, zwork, minzw, bwork, &info);

    free(dwork);
    free(zwork);
    free(bwork);
    free(scale);

    Py_DECREF(a_array);
    Py_DECREF(qg_array);

    if (info < 0) {
        free(a_out); free(qg_out); free(u1); free(u2); free(wr); free(wi);
        PyErr_Format(PyExc_ValueError, "MB03XZ: illegal argument %d", -info);
        return NULL;
    }

    PyObject *wr_obj, *wi_obj, *sc_obj, *gc_obj, *u1_obj, *u2_obj, *scale_obj;

    npy_intp wr_dims[1] = {n2};
    wr_obj = PyArray_EMPTY(1, wr_dims, NPY_DOUBLE, 1);
    wi_obj = PyArray_EMPTY(1, wr_dims, NPY_DOUBLE, 1);
    if (n2 > 0) {
        memcpy(PyArray_DATA((PyArrayObject *)wr_obj), wr, n2 * sizeof(f64));
        memcpy(PyArray_DATA((PyArrayObject *)wi_obj), wi, n2 * sizeof(f64));
    }
    free(wr); free(wi);

    if (wants && n2 > 0) {
        npy_intp sc_dims[2] = {n2, n2};
        sc_obj = PyArray_EMPTY(2, sc_dims, NPY_COMPLEX128, 1);
        c128 *sc_data = (c128 *)PyArray_DATA((PyArrayObject *)sc_obj);
        for (i32 j = 0; j < n2; j++) {
            for (i32 i = 0; i < n2; i++) {
                sc_data[i + j * n2] = a_out[i + j * lda];
            }
        }
    } else {
        npy_intp sc_dims[2] = {wants ? n2 : 0, wants ? n2 : 0};
        sc_obj = PyArray_ZEROS(2, sc_dims, NPY_COMPLEX128, 1);
    }
    free(a_out);

    if (wantg && n2 > 0) {
        npy_intp gc_dims[2] = {n2, n2};
        gc_obj = PyArray_EMPTY(2, gc_dims, NPY_COMPLEX128, 1);
        c128 *gc_data = (c128 *)PyArray_DATA((PyArrayObject *)gc_obj);
        for (i32 j = 0; j < n2; j++) {
            for (i32 i = 0; i < n2; i++) {
                gc_data[i + j * n2] = qg_out[i + j * ldqg];
            }
        }
    } else {
        npy_intp gc_dims[2] = {0, 0};
        gc_obj = PyArray_ZEROS(2, gc_dims, NPY_COMPLEX128, 1);
    }
    free(qg_out);

    if (wantus && n2 > 0) {
        npy_intp u_dims[2] = {n2, n2};
        u1_obj = PyArray_EMPTY(2, u_dims, NPY_COMPLEX128, 1);
        u2_obj = PyArray_EMPTY(2, u_dims, NPY_COMPLEX128, 1);
        c128 *u1_data = (c128 *)PyArray_DATA((PyArrayObject *)u1_obj);
        c128 *u2_data = (c128 *)PyArray_DATA((PyArrayObject *)u2_obj);
        for (i32 j = 0; j < n2; j++) {
            for (i32 i = 0; i < n2; i++) {
                u1_data[i + j * n2] = u1[i + j * ldu1];
                u2_data[i + j * n2] = u2[i + j * ldu2];
            }
        }
    } else {
        npy_intp u_dims[2] = {0, 0};
        u1_obj = PyArray_ZEROS(2, u_dims, NPY_COMPLEX128, 1);
        u2_obj = PyArray_ZEROS(2, u_dims, NPY_COMPLEX128, 1);
    }
    free(u1); free(u2);

    npy_intp scale_dims[1] = {n};
    scale_obj = PyArray_ZEROS(1, scale_dims, NPY_DOUBLE, 1);

    PyObject *result = Py_BuildValue("(OOOOOOiOi)",
        wr_obj, wi_obj, sc_obj, gc_obj, u1_obj, u2_obj, (int)ilo, scale_obj, (int)info);

    Py_DECREF(wr_obj);
    Py_DECREF(wi_obj);
    Py_DECREF(sc_obj);
    Py_DECREF(gc_obj);
    Py_DECREF(u1_obj);
    Py_DECREF(u2_obj);
    Py_DECREF(scale_obj);

    return result;
}

/* Python wrapper for mb03id */
PyObject* py_mb03id(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"compq", "compu", "n", "a", "c", "d", "b", "f", "q", "u1", "u2", NULL};

    const char *compq, *compu;
    int n;
    PyObject *a_obj, *c_obj, *d_obj, *b_obj, *f_obj;
    PyObject *q_obj = Py_None, *u1_obj_in = Py_None, *u2_obj_in = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssiOOOOO|OOO", kwlist,
                                     &compq, &compu, &n, &a_obj, &c_obj, &d_obj, &b_obj, &f_obj,
                                     &q_obj, &u1_obj_in, &u2_obj_in)) {
        return NULL;
    }

    bool liniq = (compq[0] == 'I' || compq[0] == 'i');
    bool lupdq = (compq[0] == 'U' || compq[0] == 'u');
    bool lcmpq = liniq || lupdq;

    bool liniu = (compu[0] == 'I' || compu[0] == 'i');
    bool lupdu = (compu[0] == 'U' || compu[0] == 'u');
    bool lcmpu = liniu || lupdu;

    i32 m = n / 2;

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *c_array = (PyArrayObject *)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject *)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(c_array);
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }

    PyArrayObject *f_array = (PyArrayObject *)PyArray_FROM_OTF(
        f_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (f_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(b_array);
        return NULL;
    }

    f64 *a = (f64 *)PyArray_DATA(a_array);
    f64 *c = (f64 *)PyArray_DATA(c_array);
    f64 *d = (f64 *)PyArray_DATA(d_array);
    f64 *b = (f64 *)PyArray_DATA(b_array);
    f64 *f = (f64 *)PyArray_DATA(f_array);

    i32 lda = (m > 0) ? (i32)(PyArray_STRIDE(a_array, 1) / sizeof(f64)) : 1;
    i32 ldc = (m > 0) ? (i32)(PyArray_STRIDE(c_array, 1) / sizeof(f64)) : 1;
    i32 ldd = (m > 0) ? (i32)(PyArray_STRIDE(d_array, 1) / sizeof(f64)) : 1;
    i32 ldb = (m > 0) ? (i32)(PyArray_STRIDE(b_array, 1) / sizeof(f64)) : 1;
    i32 ldf = (m > 0) ? (i32)(PyArray_STRIDE(f_array, 1) / sizeof(f64)) : 1;

    PyArrayObject *q_array = NULL;
    f64 *q = NULL;
    i32 ldq = 1;

    if (lcmpq) {
        if (q_obj != Py_None && q_obj != NULL && lupdq) {
            q_array = (PyArrayObject *)PyArray_FROM_OTF(
                q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        } else {
            npy_intp q_dims[2] = {n > 0 ? n : 1, n > 0 ? n : 1};
            q_array = (PyArrayObject *)PyArray_ZEROS(2, q_dims, NPY_DOUBLE, 1);
        }
        if (q_array == NULL) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(c_array);
            PyArray_DiscardWritebackIfCopy(d_array);
            PyArray_DiscardWritebackIfCopy(b_array);
            PyArray_DiscardWritebackIfCopy(f_array);
            Py_DECREF(a_array);
            Py_DECREF(c_array);
            Py_DECREF(d_array);
            Py_DECREF(b_array);
            Py_DECREF(f_array);
            return NULL;
        }
        q = (f64 *)PyArray_DATA(q_array);
        ldq = (n > 0) ? (i32)(PyArray_STRIDE(q_array, 1) / sizeof(f64)) : 1;
    } else {
        npy_intp q_dims[2] = {1, 1};
        q_array = (PyArrayObject *)PyArray_ZEROS(2, q_dims, NPY_DOUBLE, 1);
        if (q_array == NULL) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(c_array);
            PyArray_DiscardWritebackIfCopy(d_array);
            PyArray_DiscardWritebackIfCopy(b_array);
            PyArray_DiscardWritebackIfCopy(f_array);
            Py_DECREF(a_array);
            Py_DECREF(c_array);
            Py_DECREF(d_array);
            Py_DECREF(b_array);
            Py_DECREF(f_array);
            return NULL;
        }
        q = (f64 *)PyArray_DATA(q_array);
    }

    PyArrayObject *u1_array = NULL;
    f64 *u1 = NULL;
    i32 ldu1 = 1;

    if (lcmpu) {
        if (u1_obj_in != Py_None && u1_obj_in != NULL && lupdu) {
            u1_array = (PyArrayObject *)PyArray_FROM_OTF(
                u1_obj_in, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        } else {
            npy_intp u1_dims[2] = {m > 0 ? m : 1, m > 0 ? m : 1};
            u1_array = (PyArrayObject *)PyArray_ZEROS(2, u1_dims, NPY_DOUBLE, 1);
        }
        if (u1_array == NULL) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(c_array);
            PyArray_DiscardWritebackIfCopy(d_array);
            PyArray_DiscardWritebackIfCopy(b_array);
            PyArray_DiscardWritebackIfCopy(f_array);
            Py_DECREF(a_array);
            Py_DECREF(c_array);
            Py_DECREF(d_array);
            Py_DECREF(b_array);
            Py_DECREF(f_array);
            Py_DECREF(q_array);
            return NULL;
        }
        u1 = (f64 *)PyArray_DATA(u1_array);
        ldu1 = (m > 0) ? (i32)(PyArray_STRIDE(u1_array, 1) / sizeof(f64)) : 1;
    } else {
        npy_intp u1_dims[2] = {1, 1};
        u1_array = (PyArrayObject *)PyArray_ZEROS(2, u1_dims, NPY_DOUBLE, 1);
        if (u1_array == NULL) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(c_array);
            PyArray_DiscardWritebackIfCopy(d_array);
            PyArray_DiscardWritebackIfCopy(b_array);
            PyArray_DiscardWritebackIfCopy(f_array);
            Py_DECREF(a_array);
            Py_DECREF(c_array);
            Py_DECREF(d_array);
            Py_DECREF(b_array);
            Py_DECREF(f_array);
            Py_DECREF(q_array);
            return NULL;
        }
        u1 = (f64 *)PyArray_DATA(u1_array);
    }

    PyArrayObject *u2_array = NULL;
    f64 *u2 = NULL;
    i32 ldu2 = 1;

    if (lcmpu) {
        if (u2_obj_in != Py_None && u2_obj_in != NULL && lupdu) {
            u2_array = (PyArrayObject *)PyArray_FROM_OTF(
                u2_obj_in, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        } else {
            npy_intp u2_dims[2] = {m > 0 ? m : 1, m > 0 ? m : 1};
            u2_array = (PyArrayObject *)PyArray_ZEROS(2, u2_dims, NPY_DOUBLE, 1);
        }
        if (u2_array == NULL) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(c_array);
            PyArray_DiscardWritebackIfCopy(d_array);
            PyArray_DiscardWritebackIfCopy(b_array);
            PyArray_DiscardWritebackIfCopy(f_array);
            Py_DECREF(a_array);
            Py_DECREF(c_array);
            Py_DECREF(d_array);
            Py_DECREF(b_array);
            Py_DECREF(f_array);
            Py_DECREF(q_array);
            Py_DECREF(u1_array);
            return NULL;
        }
        u2 = (f64 *)PyArray_DATA(u2_array);
        ldu2 = (m > 0) ? (i32)(PyArray_STRIDE(u2_array, 1) / sizeof(f64)) : 1;
    } else {
        npy_intp u2_dims[2] = {1, 1};
        u2_array = (PyArrayObject *)PyArray_ZEROS(2, u2_dims, NPY_DOUBLE, 1);
        if (u2_array == NULL) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(c_array);
            PyArray_DiscardWritebackIfCopy(d_array);
            PyArray_DiscardWritebackIfCopy(b_array);
            PyArray_DiscardWritebackIfCopy(f_array);
            Py_DECREF(a_array);
            Py_DECREF(c_array);
            Py_DECREF(d_array);
            Py_DECREF(b_array);
            Py_DECREF(f_array);
            Py_DECREF(q_array);
            Py_DECREF(u1_array);
            return NULL;
        }
        u2 = (f64 *)PyArray_DATA(u2_array);
    }

    i32 liwork = n + 1;
    if (liwork < 1) liwork = 1;
    i32 *iwork = (i32 *)malloc(liwork * sizeof(i32));
    if (iwork == NULL) {
        PyArray_DiscardWritebackIfCopy(a_array);
        PyArray_DiscardWritebackIfCopy(c_array);
        PyArray_DiscardWritebackIfCopy(d_array);
        PyArray_DiscardWritebackIfCopy(b_array);
        PyArray_DiscardWritebackIfCopy(f_array);
        Py_DECREF(a_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(b_array);
        Py_DECREF(f_array);
        Py_DECREF(q_array);
        Py_DECREF(u1_array);
        Py_DECREF(u2_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 ldwork;
    if (lcmpq) {
        ldwork = (4 * n + 48 > 171) ? 4 * n + 48 : 171;
    } else {
        ldwork = (2 * n + 48 > 171) ? 2 * n + 48 : 171;
    }
    f64 *dwork = (f64 *)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        free(iwork);
        PyArray_DiscardWritebackIfCopy(a_array);
        PyArray_DiscardWritebackIfCopy(c_array);
        PyArray_DiscardWritebackIfCopy(d_array);
        PyArray_DiscardWritebackIfCopy(b_array);
        PyArray_DiscardWritebackIfCopy(f_array);
        Py_DECREF(a_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(b_array);
        Py_DECREF(f_array);
        Py_DECREF(q_array);
        Py_DECREF(u1_array);
        Py_DECREF(u2_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 neig = 0;
    i32 info = 0;

    mb03id(compq, compu, n, a, lda, c, ldc, d, ldd, b, ldb, f, ldf,
           q, ldq, u1, ldu1, u2, ldu2, &neig, iwork, liwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(f_array);
    if (lcmpq && q_obj != Py_None && q_obj != NULL && lupdq) {
        PyArray_ResolveWritebackIfCopy(q_array);
    }
    if (lcmpu && u1_obj_in != Py_None && u1_obj_in != NULL && lupdu) {
        PyArray_ResolveWritebackIfCopy(u1_array);
    }
    if (lcmpu && u2_obj_in != Py_None && u2_obj_in != NULL && lupdu) {
        PyArray_ResolveWritebackIfCopy(u2_array);
    }

    PyObject *result = Py_BuildValue("(OOOOOOOOii)",
        (PyObject *)a_array, (PyObject *)c_array, (PyObject *)d_array,
        (PyObject *)b_array, (PyObject *)f_array, (PyObject *)q_array,
        (PyObject *)u1_array, (PyObject *)u2_array, (int)neig, (int)info);

    Py_DECREF(a_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(b_array);
    Py_DECREF(f_array);
    Py_DECREF(q_array);
    Py_DECREF(u1_array);
    Py_DECREF(u2_array);

    return result;
}

PyObject *py_mb3jzp(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;

    const char *compq;
    int n;
    PyObject *a_obj, *d_obj, *b_obj, *f_obj;
    double tol = -1.0;
    PyObject *q_obj = Py_None;

    static char *kwlist[] = {"compq", "n", "a", "d", "b", "f", "q", "tol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siOOOO|Od", kwlist,
                                     &compq, &n, &a_obj, &d_obj,
                                     &b_obj, &f_obj, &q_obj, &tol)) {
        return NULL;
    }

    i32 m = n / 2;
    int liniq = (compq[0] == 'I' || compq[0] == 'i');
    int lupdq = (compq[0] == 'U' || compq[0] == 'u');
    int lcmpq = liniq || lupdq;

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *d_array = (PyArrayObject *)PyArray_FROM_OTF(
        d_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *f_array = (PyArrayObject *)PyArray_FROM_OTF(
        f_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || d_array == NULL ||
        b_array == NULL || f_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(d_array);
        Py_XDECREF(b_array);
        Py_XDECREF(f_array);
        return NULL;
    }

    i32 lda = (m > 0) ? (i32)(PyArray_STRIDE(a_array, 1) / sizeof(c128)) : 1;
    i32 ldd = (m > 0) ? (i32)(PyArray_STRIDE(d_array, 1) / sizeof(c128)) : 1;
    i32 ldb = (m > 0) ? (i32)(PyArray_STRIDE(b_array, 1) / sizeof(c128)) : 1;
    i32 ldf = (m > 0) ? (i32)(PyArray_STRIDE(f_array, 1) / sizeof(c128)) : 1;

    c128 *a = (c128 *)PyArray_DATA(a_array);
    c128 *d = (c128 *)PyArray_DATA(d_array);
    c128 *b = (c128 *)PyArray_DATA(b_array);
    c128 *f = (c128 *)PyArray_DATA(f_array);

    PyArrayObject *q_array = NULL;
    c128 *q = NULL;
    i32 ldq = 1;

    if (lcmpq) {
        if (q_obj != Py_None && q_obj != NULL && lupdq) {
            q_array = (PyArrayObject *)PyArray_FROM_OTF(
                q_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        } else {
            npy_intp q_dims[2] = {n > 0 ? n : 1, n > 0 ? n : 1};
            q_array = (PyArrayObject *)PyArray_ZEROS(2, q_dims, NPY_COMPLEX128, 1);
        }
        if (q_array == NULL) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(d_array);
            PyArray_DiscardWritebackIfCopy(b_array);
            PyArray_DiscardWritebackIfCopy(f_array);
            Py_DECREF(a_array);
            Py_DECREF(d_array);
            Py_DECREF(b_array);
            Py_DECREF(f_array);
            return NULL;
        }
        q = (c128 *)PyArray_DATA(q_array);
        ldq = (n > 0) ? (i32)(PyArray_STRIDE(q_array, 1) / sizeof(c128)) : 1;
    } else {
        npy_intp q_dims[2] = {1, 1};
        q_array = (PyArrayObject *)PyArray_ZEROS(2, q_dims, NPY_COMPLEX128, 1);
        if (q_array == NULL) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(d_array);
            PyArray_DiscardWritebackIfCopy(b_array);
            PyArray_DiscardWritebackIfCopy(f_array);
            Py_DECREF(a_array);
            Py_DECREF(d_array);
            Py_DECREF(b_array);
            Py_DECREF(f_array);
            return NULL;
        }
        q = (c128 *)PyArray_DATA(q_array);
    }

    i32 ldwork = (m > 0) ? m : 1;
    f64 *dwork = (f64 *)PyMem_Calloc(ldwork, sizeof(f64));
    c128 *zwork = (c128 *)PyMem_Calloc(ldwork, sizeof(c128));

    if (dwork == NULL || zwork == NULL) {
        PyMem_Free(dwork);
        PyMem_Free(zwork);
        PyArray_DiscardWritebackIfCopy(a_array);
        PyArray_DiscardWritebackIfCopy(d_array);
        PyArray_DiscardWritebackIfCopy(b_array);
        PyArray_DiscardWritebackIfCopy(f_array);
        Py_DECREF(a_array);
        Py_DECREF(d_array);
        Py_DECREF(b_array);
        Py_DECREF(f_array);
        Py_DECREF(q_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 neig = 0;
    i32 info = 0;

    mb3jzp(compq, n, a, lda, d, ldd, b, ldb, f, ldf, q, ldq, &neig, tol,
           dwork, zwork, &info);

    PyMem_Free(dwork);
    PyMem_Free(zwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(f_array);
    if (lcmpq && q_obj != Py_None && q_obj != NULL && lupdq) {
        PyArray_ResolveWritebackIfCopy(q_array);
    }

    PyObject *result_jzp = Py_BuildValue("(OOOOOii)",
        (PyObject *)a_array, (PyObject *)d_array, (PyObject *)b_array,
        (PyObject *)f_array, (PyObject *)q_array, (int)neig, (int)info);

    Py_DECREF(a_array);
    Py_DECREF(d_array);
    Py_DECREF(b_array);
    Py_DECREF(f_array);
    Py_DECREF(q_array);

    return result_jzp;
}

PyObject *py_mb3lzp(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"compq", "orth", "n", "a", "de", "b", "fg", "q", NULL};

    const char *compq = NULL;
    const char *orth = NULL;
    int n = 0;
    PyObject *a_obj = NULL;
    PyObject *de_obj = NULL;
    PyObject *b_obj = NULL;
    PyObject *fg_obj = NULL;
    PyObject *q_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssiOOOO|O", kwlist,
                                     &compq, &orth, &n, &a_obj, &de_obj,
                                     &b_obj, &fg_obj, &q_obj)) {
        return NULL;
    }

    if (n < 0 || (n % 2) != 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative and even");
        return NULL;
    }

    bool lcmpq = (compq[0] == 'C' || compq[0] == 'c');
    (void)lcmpq;

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *de_array = (PyArrayObject *)PyArray_FROM_OTF(
        de_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (de_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        return NULL;
    }

    PyArrayObject *fg_array = (PyArrayObject *)PyArray_FROM_OTF(
        fg_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (fg_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(b_array);
        return NULL;
    }

    i32 lda = (n > 0) ? (i32)(PyArray_STRIDE(a_array, 1) / sizeof(c128)) : 1;
    i32 ldde = (n > 0) ? (i32)(PyArray_STRIDE(de_array, 1) / sizeof(c128)) : 1;
    i32 ldb = (n > 0) ? (i32)(PyArray_STRIDE(b_array, 1) / sizeof(c128)) : 1;
    i32 ldfg = (n > 0) ? (i32)(PyArray_STRIDE(fg_array, 1) / sizeof(c128)) : 1;

    c128 *a = (c128 *)PyArray_DATA(a_array);
    c128 *de = (c128 *)PyArray_DATA(de_array);
    c128 *b = (c128 *)PyArray_DATA(b_array);
    c128 *fg = (c128 *)PyArray_DATA(fg_array);

    PyArrayObject *q_array = NULL;
    c128 *q = NULL;
    i32 ldq = 1;
    i32 n2 = 2 * n;

    if (lcmpq) {
        npy_intp q_dims[2] = {n2 > 0 ? n2 : 1, n2 > 0 ? n2 : 1};
        q_array = (PyArrayObject *)PyArray_ZEROS(2, q_dims, NPY_COMPLEX128, 1);
        if (q_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(de_array);
            Py_DECREF(b_array);
            Py_DECREF(fg_array);
            return NULL;
        }
        q = (c128 *)PyArray_DATA(q_array);
        ldq = (n2 > 0) ? (i32)(PyArray_STRIDE(q_array, 1) / sizeof(c128)) : 1;
    } else {
        npy_intp q_dims[2] = {1, 1};
        q_array = (PyArrayObject *)PyArray_ZEROS(2, q_dims, NPY_COMPLEX128, 1);
        if (q_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(de_array);
            Py_DECREF(b_array);
            Py_DECREF(fg_array);
            return NULL;
        }
        q = (c128 *)PyArray_DATA(q_array);
    }

    npy_intp eig_dims[1] = {n > 0 ? n : 1};
    PyArrayObject *alphar_array = (PyArrayObject *)PyArray_ZEROS(1, eig_dims, NPY_DOUBLE, 0);
    PyArrayObject *alphai_array = (PyArrayObject *)PyArray_ZEROS(1, eig_dims, NPY_DOUBLE, 0);
    PyArrayObject *beta_array = (PyArrayObject *)PyArray_ZEROS(1, eig_dims, NPY_DOUBLE, 0);

    if (alphar_array == NULL || alphai_array == NULL || beta_array == NULL) {
        Py_XDECREF(alphar_array);
        Py_XDECREF(alphai_array);
        Py_XDECREF(beta_array);
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(b_array);
        Py_DECREF(fg_array);
        Py_DECREF(q_array);
        return NULL;
    }

    f64 *alphar = (f64 *)PyArray_DATA(alphar_array);
    f64 *alphai = (f64 *)PyArray_DATA(alphai_array);
    f64 *beta_val = (f64 *)PyArray_DATA(beta_array);

    i32 nn = n * n;
    i32 maxn3 = (n > 3) ? n : 3;
    i32 ldwork;
    i32 lzwork;
    if (n == 0) {
        ldwork = 1;
        lzwork = 1;
    } else if (lcmpq) {
        ldwork = 11 * nn + 2 * n;
        lzwork = 8 * n + 4;
    } else {
        ldwork = 4 * nn + 2 * n + maxn3;
        lzwork = 1;
    }

    i32 liwork = n + 1;
    i32 lbwork = lcmpq ? (n > 1 ? n - 1 : 1) : 1;

    f64 *dwork = (f64 *)PyMem_Calloc(ldwork > 0 ? ldwork : 1, sizeof(f64));
    c128 *zwork = (c128 *)PyMem_Calloc(lzwork > 0 ? lzwork : 1, sizeof(c128));
    i32 *iwork = (i32 *)PyMem_Calloc(liwork > 0 ? liwork : 1, sizeof(i32));
    bool *bwork = (bool *)PyMem_Calloc(lbwork > 0 ? lbwork : 1, sizeof(bool));

    if (dwork == NULL || zwork == NULL || iwork == NULL || bwork == NULL) {
        PyMem_Free(dwork);
        PyMem_Free(zwork);
        PyMem_Free(iwork);
        PyMem_Free(bwork);
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(b_array);
        Py_DECREF(fg_array);
        Py_DECREF(q_array);
        Py_DECREF(alphar_array);
        Py_DECREF(alphai_array);
        Py_DECREF(beta_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 neig = 0;
    i32 info = 0;

    mb3lzp(compq, orth, n, a, lda, de, ldde, b, ldb, fg, ldfg, &neig, q, ldq,
           alphar, alphai, beta_val, iwork, dwork, ldwork, zwork, lzwork,
           bwork, &info);

    PyMem_Free(dwork);
    PyMem_Free(zwork);
    PyMem_Free(iwork);
    PyMem_Free(bwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(de_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(fg_array);

    PyObject *result;
    if (lcmpq) {
        result = Py_BuildValue("(OOOOOOOOii)",
            (PyObject *)a_array, (PyObject *)de_array,
            (PyObject *)b_array, (PyObject *)fg_array,
            (PyObject *)q_array,
            (PyObject *)alphar_array, (PyObject *)alphai_array,
            (PyObject *)beta_array,
            (int)neig, (int)info);
    } else {
        result = Py_BuildValue("(OOOii)",
            (PyObject *)alphar_array, (PyObject *)alphai_array,
            (PyObject *)beta_array,
            (int)neig, (int)info);
    }

    Py_DECREF(a_array);
    Py_DECREF(de_array);
    Py_DECREF(b_array);
    Py_DECREF(fg_array);
    Py_DECREF(q_array);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);

    return result;
}

PyObject *py_mb03lf(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"compq", "compu", "orth", "z", "b", "fg", "n", NULL};

    const char *compq = NULL;
    const char *compu = NULL;
    const char *orth = NULL;
    PyObject *z_obj = NULL;
    PyObject *b_obj = NULL;
    PyObject *fg_obj = NULL;
    int n_arg = -1;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssOOO|i", kwlist,
                                     &compq, &compu, &orth, &z_obj, &b_obj,
                                     &fg_obj, &n_arg)) {
        return NULL;
    }

    PyArrayObject *z_array = (PyArrayObject *)PyArray_FROM_OTF(
        z_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (z_array == NULL) return NULL;

    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (b_array == NULL) {
        Py_DECREF(z_array);
        return NULL;
    }

    PyArrayObject *fg_array = (PyArrayObject *)PyArray_FROM_OTF(
        fg_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (fg_array == NULL) {
        Py_DECREF(z_array);
        Py_DECREF(b_array);
        return NULL;
    }

    i32 n = (n_arg >= 0) ? (i32)n_arg : (i32)PyArray_DIM(z_array, 0);

    if (n < 0 || (n % 2) != 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative and even");
        Py_DECREF(z_array);
        Py_DECREF(b_array);
        Py_DECREF(fg_array);
        return NULL;
    }

    i32 m = n / 2;
    bool lcmpq = (compq[0] == 'C' || compq[0] == 'c');
    bool lcmpu = (compu[0] == 'C' || compu[0] == 'c');
    bool lcmp = lcmpq || lcmpu;

    i32 ldz = (n > 0) ? (i32)(PyArray_STRIDE(z_array, 1) / sizeof(f64)) : 1;
    i32 ldb = (m > 0) ? (i32)(PyArray_STRIDE(b_array, 1) / sizeof(f64)) : 1;
    i32 ldfg = (m > 0) ? (i32)(PyArray_STRIDE(fg_array, 1) / sizeof(f64)) : 1;

    f64 *z = (f64 *)PyArray_DATA(z_array);
    const f64 *b = (const f64 *)PyArray_DATA(b_array);
    const f64 *fg = (const f64 *)PyArray_DATA(fg_array);

    PyArrayObject *q_array = NULL;
    f64 *q = NULL;
    i32 ldq = 1;
    i32 n2 = 2 * n;

    if (lcmpq) {
        npy_intp q_dims[2] = {n2 > 0 ? n2 : 1, n2 > 0 ? n2 : 1};
        q_array = (PyArrayObject *)PyArray_ZEROS(2, q_dims, NPY_DOUBLE, 1);
        if (q_array == NULL) {
            Py_DECREF(z_array);
            Py_DECREF(b_array);
            Py_DECREF(fg_array);
            return NULL;
        }
        q = (f64 *)PyArray_DATA(q_array);
        ldq = (n2 > 0) ? (i32)(PyArray_STRIDE(q_array, 1) / sizeof(f64)) : 1;
    } else {
        npy_intp q_dims[2] = {1, 1};
        q_array = (PyArrayObject *)PyArray_ZEROS(2, q_dims, NPY_DOUBLE, 1);
        if (q_array == NULL) {
            Py_DECREF(z_array);
            Py_DECREF(b_array);
            Py_DECREF(fg_array);
            return NULL;
        }
        q = (f64 *)PyArray_DATA(q_array);
    }

    PyArrayObject *u_array = NULL;
    f64 *u = NULL;
    i32 ldu = 1;

    if (lcmpu) {
        npy_intp u_dims[2] = {n > 0 ? n : 1, n2 > 0 ? n2 : 1};
        u_array = (PyArrayObject *)PyArray_ZEROS(2, u_dims, NPY_DOUBLE, 1);
        if (u_array == NULL) {
            Py_DECREF(z_array);
            Py_DECREF(b_array);
            Py_DECREF(fg_array);
            Py_DECREF(q_array);
            return NULL;
        }
        u = (f64 *)PyArray_DATA(u_array);
        ldu = (n > 0) ? (i32)(PyArray_STRIDE(u_array, 1) / sizeof(f64)) : 1;
    } else {
        npy_intp u_dims[2] = {1, 1};
        u_array = (PyArrayObject *)PyArray_ZEROS(2, u_dims, NPY_DOUBLE, 1);
        if (u_array == NULL) {
            Py_DECREF(z_array);
            Py_DECREF(b_array);
            Py_DECREF(fg_array);
            Py_DECREF(q_array);
            return NULL;
        }
        u = (f64 *)PyArray_DATA(u_array);
    }

    npy_intp eig_dims[1] = {m > 0 ? m : 1};
    PyArrayObject *alphar_array = (PyArrayObject *)PyArray_ZEROS(1, eig_dims, NPY_DOUBLE, 0);
    PyArrayObject *alphai_array = (PyArrayObject *)PyArray_ZEROS(1, eig_dims, NPY_DOUBLE, 0);
    PyArrayObject *beta_array = (PyArrayObject *)PyArray_ZEROS(1, eig_dims, NPY_DOUBLE, 0);

    if (alphar_array == NULL || alphai_array == NULL || beta_array == NULL) {
        Py_XDECREF(alphar_array);
        Py_XDECREF(alphai_array);
        Py_XDECREF(beta_array);
        Py_DECREF(z_array);
        Py_DECREF(b_array);
        Py_DECREF(fg_array);
        Py_DECREF(q_array);
        Py_DECREF(u_array);
        return NULL;
    }

    f64 *alphar = (f64 *)PyArray_DATA(alphar_array);
    f64 *alphai = (f64 *)PyArray_DATA(alphai_array);
    f64 *beta_val = (f64 *)PyArray_DATA(beta_array);

    i32 nn = n * n;
    i32 liwork, ldwork;
    if (n == 0) {
        liwork = 1;
        ldwork = 1;
    } else if (lcmp) {
        liwork = (n2 + 1 > 48) ? n2 + 1 : 48;
        i32 i_coef, j_coef;
        if (lcmpq) {
            i_coef = 4;
            j_coef = 10;
        } else {
            i_coef = 2;
            j_coef = 7;
        }
        if (lcmpu) {
            i_coef += 1;
            j_coef += 1;
        }
        (void)i_coef;
        i32 temp = (m + 252 > 432) ? m + 252 : 432;
        ldwork = j_coef * nn + temp;
    } else {
        liwork = n + 18;
        i32 temp = (6 * n > 54) ? 6 * n : 54;
        ldwork = 2 * nn + 3 * (nn / 2) + temp;
    }

    i32 lbwork = m > 0 ? m : 1;

    f64 *dwork = (f64 *)PyMem_Calloc(ldwork > 0 ? ldwork : 1, sizeof(f64));
    i32 *iwork = (i32 *)PyMem_Calloc(liwork > 0 ? liwork : 1, sizeof(i32));
    bool *bwork = (bool *)PyMem_Calloc(lbwork > 0 ? lbwork : 1, sizeof(bool));

    if (dwork == NULL || iwork == NULL || bwork == NULL) {
        PyMem_Free(dwork);
        PyMem_Free(iwork);
        PyMem_Free(bwork);
        Py_DECREF(z_array);
        Py_DECREF(b_array);
        Py_DECREF(fg_array);
        Py_DECREF(q_array);
        Py_DECREF(u_array);
        Py_DECREF(alphar_array);
        Py_DECREF(alphai_array);
        Py_DECREF(beta_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 neig = 0;
    i32 iwarn = 0;
    i32 info = 0;

    mb03lf(compq, compu, orth, n, z, ldz, b, ldb, fg, ldfg, &neig, q, ldq,
           u, ldu, alphar, alphai, beta_val, iwork, liwork, dwork, ldwork,
           bwork, &iwarn, &info);

    PyMem_Free(dwork);
    PyMem_Free(iwork);
    PyMem_Free(bwork);

    PyArray_ResolveWritebackIfCopy(z_array);

    PyObject *result = Py_BuildValue("(OiOOOOOii)",
        (PyObject *)z_array,
        (int)neig,
        (PyObject *)q_array,
        (PyObject *)u_array,
        (PyObject *)alphar_array,
        (PyObject *)alphai_array,
        (PyObject *)beta_array,
        (int)iwarn,
        (int)info);

    Py_DECREF(z_array);
    Py_DECREF(b_array);
    Py_DECREF(fg_array);
    Py_DECREF(q_array);
    Py_DECREF(u_array);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);

    return result;
}

PyObject *py_mb03wa(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *a_obj, *b_obj, *q_obj, *z_obj;
    int wantq, wantz, n1, n2;

    if (!PyArg_ParseTuple(args, "ppiiOOOO", &wantq, &wantz, &n1, &n2,
                          &a_obj, &b_obj, &q_obj, &z_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *q_array = (PyArrayObject *)PyArray_FROM_OTF(
        q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *z_array = (PyArrayObject *)PyArray_FROM_OTF(
        z_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || b_array == NULL || q_array == NULL || z_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(q_array);
        Py_XDECREF(z_array);
        return NULL;
    }

    i32 lda = (i32)PyArray_DIM(a_array, 0);
    i32 ldb = (i32)PyArray_DIM(b_array, 0);
    i32 ldq = (i32)PyArray_DIM(q_array, 0);
    i32 ldz = (i32)PyArray_DIM(z_array, 0);

    f64 *a = (f64 *)PyArray_DATA(a_array);
    f64 *b = (f64 *)PyArray_DATA(b_array);
    f64 *q = (f64 *)PyArray_DATA(q_array);
    f64 *z = (f64 *)PyArray_DATA(z_array);

    i32 info = 0;

    mb03wa((bool)wantq, (bool)wantz, (i32)n1, (i32)n2,
           a, lda, b, ldb, q, ldq, z, ldz, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(q_array);
    PyArray_ResolveWritebackIfCopy(z_array);

    PyObject *result = Py_BuildValue("(OOOOi)",
        (PyObject *)a_array,
        (PyObject *)b_array,
        (PyObject *)q_array,
        (PyObject *)z_array,
        (int)info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);

    return result;
}

/* Python wrapper for mb03za */
PyObject* py_mb03za(PyObject* self, PyObject* args) {
    (void)self;
    const char *compc, *compu, *compv, *compw, *which;
    PyObject *select_obj, *a_obj, *b_obj, *c_obj;
    PyObject *u1_obj, *u2_obj, *v1_obj, *v2_obj, *w_obj;

    if (!PyArg_ParseTuple(args, "sssssOOOOOOOOO",
                          &compc, &compu, &compv, &compw, &which,
                          &select_obj, &a_obj, &b_obj, &c_obj,
                          &u1_obj, &u2_obj, &v1_obj, &v2_obj, &w_obj)) {
        return NULL;
    }

    PyArrayObject *select_array = (PyArrayObject *)PyArray_FROM_OTF(
        select_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *c_array = (PyArrayObject *)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *u1_array = (PyArrayObject *)PyArray_FROM_OTF(
        u1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *u2_array = (PyArrayObject *)PyArray_FROM_OTF(
        u2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *v1_array = (PyArrayObject *)PyArray_FROM_OTF(
        v1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *v2_array = (PyArrayObject *)PyArray_FROM_OTF(
        v2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *w_array = (PyArrayObject *)PyArray_FROM_OTF(
        w_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!select_array || !a_array || !b_array || !c_array ||
        !u1_array || !u2_array || !v1_array || !v2_array || !w_array) {
        Py_XDECREF(select_array);
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(u1_array);
        Py_XDECREF(u2_array);
        Py_XDECREF(v1_array);
        Py_XDECREF(v2_array);
        Py_XDECREF(w_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = (i32)PyArray_DIM(c_array, 0);
    if (ldc < 1) ldc = 1;
    i32 ldu1 = (i32)PyArray_DIM(u1_array, 0);
    if (ldu1 < 1) ldu1 = 1;
    i32 ldu2 = (i32)PyArray_DIM(u2_array, 0);
    if (ldu2 < 1) ldu2 = 1;
    i32 ldv1 = (i32)PyArray_DIM(v1_array, 0);
    if (ldv1 < 1) ldv1 = 1;
    i32 ldv2 = (i32)PyArray_DIM(v2_array, 0);
    if (ldv2 < 1) ldv2 = 1;
    i32 ldw = (i32)PyArray_DIM(w_array, 0);
    if (ldw < 1) ldw = 1;

    i32 *select = (i32 *)PyArray_DATA(select_array);
    f64 *a = (f64 *)PyArray_DATA(a_array);
    f64 *b = (f64 *)PyArray_DATA(b_array);
    f64 *c = (f64 *)PyArray_DATA(c_array);
    f64 *u1 = (f64 *)PyArray_DATA(u1_array);
    f64 *u2 = (f64 *)PyArray_DATA(u2_array);
    f64 *v1 = (f64 *)PyArray_DATA(v1_array);
    f64 *v2 = (f64 *)PyArray_DATA(v2_array);
    f64 *w = (f64 *)PyArray_DATA(w_array);

    i32 m_out = n;
    i32 ldwork = (4*n > 8*n) ? 4*n : 8*n;
    if (ldwork < 1) ldwork = 1;
    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));
    f64 *wr = (f64 *)calloc(n > 0 ? n : 1, sizeof(f64));
    f64 *wi = (f64 *)calloc(n > 0 ? n : 1, sizeof(f64));

    if (!dwork || !wr || !wi) {
        free(dwork);
        free(wr);
        free(wi);
        Py_DECREF(select_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(u1_array);
        Py_DECREF(u2_array);
        Py_DECREF(v1_array);
        Py_DECREF(v2_array);
        Py_DECREF(w_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info = 0;

    mb03za(compc, compu, compv, compw, which, select,
           n, a, lda, b, ldb, c, ldc, u1, ldu1, u2, ldu2,
           v1, ldv1, v2, ldv2, w, ldw, wr, wi, &m_out,
           dwork, ldwork, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(u1_array);
    PyArray_ResolveWritebackIfCopy(u2_array);
    PyArray_ResolveWritebackIfCopy(v1_array);
    PyArray_ResolveWritebackIfCopy(v2_array);
    PyArray_ResolveWritebackIfCopy(w_array);

    npy_intp dims_m = m_out > 0 ? m_out : 0;
    PyArrayObject *wr_array = (PyArrayObject *)PyArray_SimpleNew(1, &dims_m, NPY_DOUBLE);
    PyArrayObject *wi_array = (PyArrayObject *)PyArray_SimpleNew(1, &dims_m, NPY_DOUBLE);

    if (!wr_array || !wi_array) {
        free(dwork);
        free(wr);
        free(wi);
        Py_DECREF(select_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(u1_array);
        Py_DECREF(u2_array);
        Py_DECREF(v1_array);
        Py_DECREF(v2_array);
        Py_DECREF(w_array);
        Py_XDECREF(wr_array);
        Py_XDECREF(wi_array);
        PyErr_NoMemory();
        return NULL;
    }

    if (m_out > 0) {
        memcpy(PyArray_DATA(wr_array), wr, m_out * sizeof(f64));
        memcpy(PyArray_DATA(wi_array), wi, m_out * sizeof(f64));
    }

    free(dwork);
    free(wr);
    free(wi);

    PyObject *result = Py_BuildValue("(OOOOOOOOOOii)",
        (PyObject *)a_array,
        (PyObject *)b_array,
        (PyObject *)c_array,
        (PyObject *)u1_array,
        (PyObject *)u2_array,
        (PyObject *)v1_array,
        (PyObject *)v2_array,
        (PyObject *)w_array,
        (PyObject *)wr_array,
        (PyObject *)wi_array,
        (int)m_out,
        (int)info);

    Py_DECREF(select_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(u1_array);
    Py_DECREF(u2_array);
    Py_DECREF(v1_array);
    Py_DECREF(v2_array);
    Py_DECREF(w_array);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);

    return result;
}

PyObject* py_mb03zd(PyObject* self, PyObject* args) {
    (void)self;
    const char *which, *meth, *stab, *balanc, *ortbal;
    int n, mm, ilo;
    PyObject *scale_obj, *s_obj, *t_obj, *g_obj;
    PyObject *u1_obj, *u2_obj, *v1_obj, *v2_obj;

    if (!PyArg_ParseTuple(args, "sssssiiiOOOOOOOO",
                          &which, &meth, &stab, &balanc, &ortbal,
                          &n, &mm, &ilo,
                          &scale_obj, &s_obj, &t_obj, &g_obj,
                          &u1_obj, &u2_obj, &v1_obj, &v2_obj)) {
        return NULL;
    }

    PyArrayObject *scale_array = (PyArrayObject *)PyArray_FROM_OTF(
        scale_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *s_array = (PyArrayObject *)PyArray_FROM_OTF(
        s_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *t_array = (PyArrayObject *)PyArray_FROM_OTF(
        t_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *g_array = (PyArrayObject *)PyArray_FROM_OTF(
        g_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *u1_array = (PyArrayObject *)PyArray_FROM_OTF(
        u1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *u2_array = (PyArrayObject *)PyArray_FROM_OTF(
        u2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *v1_array = (PyArrayObject *)PyArray_FROM_OTF(
        v1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *v2_array = (PyArrayObject *)PyArray_FROM_OTF(
        v2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!scale_array || !s_array || !t_array || !g_array ||
        !u1_array || !u2_array || !v1_array || !v2_array) {
        Py_XDECREF(scale_array);
        Py_XDECREF(s_array);
        Py_XDECREF(t_array);
        Py_XDECREF(g_array);
        Py_XDECREF(u1_array);
        Py_XDECREF(u2_array);
        Py_XDECREF(v1_array);
        Py_XDECREF(v2_array);
        return NULL;
    }

    i32 lds = n > 0 ? n : 1;
    i32 ldt = n > 0 ? n : 1;
    i32 ldg = n > 0 ? n : 1;
    i32 ldu1 = n > 0 ? n : 1;
    i32 ldu2 = n > 0 ? n : 1;
    i32 ldv1 = n > 0 ? n : 1;
    i32 ldv2 = n > 0 ? n : 1;
    i32 n2 = 2 * n;
    i32 ldus = n2 > 0 ? n2 : 1;
    i32 lduu = n2 > 0 ? n2 : 1;

    f64 *scale = (f64 *)PyArray_DATA(scale_array);
    f64 *s = (f64 *)PyArray_DATA(s_array);
    f64 *t = (f64 *)PyArray_DATA(t_array);
    f64 *g = (f64 *)PyArray_DATA(g_array);
    f64 *u1 = (f64 *)PyArray_DATA(u1_array);
    f64 *u2 = (f64 *)PyArray_DATA(u2_array);
    f64 *v1 = (f64 *)PyArray_DATA(v1_array);
    f64 *v2 = (f64 *)PyArray_DATA(v2_array);

    i32 ldwork = 8 * n * n + 32 * n + 8;
    if (ldwork < 1) ldwork = 1;

    i32 *select = (i32 *)calloc(n2 > 0 ? n2 : 1, sizeof(i32));
    bool *lwork_arr = (bool *)calloc(n2 > 0 ? n2 : 1, sizeof(bool));
    i32 *iwork = (i32 *)calloc(n2 > 0 ? n2 : 1, sizeof(i32));
    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));
    f64 *wr = (f64 *)calloc(n > 0 ? n : 1, sizeof(f64));
    f64 *wi = (f64 *)calloc(n > 0 ? n : 1, sizeof(f64));
    f64 *us = (f64 *)calloc((mm > 0 && n2 > 0) ? n2 * mm : 1, sizeof(f64));
    f64 *uu = (f64 *)calloc((mm > 0 && n2 > 0) ? n2 * mm : 1, sizeof(f64));

    if (!select || !lwork_arr || !iwork || !dwork || !wr || !wi || !us || !uu) {
        free(select);
        free(lwork_arr);
        free(iwork);
        free(dwork);
        free(wr);
        free(wi);
        free(us);
        free(uu);
        Py_DECREF(scale_array);
        Py_DECREF(s_array);
        Py_DECREF(t_array);
        Py_DECREF(g_array);
        Py_DECREF(u1_array);
        Py_DECREF(u2_array);
        Py_DECREF(v1_array);
        Py_DECREF(v2_array);
        PyErr_NoMemory();
        return NULL;
    }

    for (i32 i = 0; i < n2; i++) {
        select[i] = 1;
    }

    i32 m_out = 0;
    i32 info = 0;

    mb03zd(which, meth, stab, balanc, ortbal, select,
           n, mm, ilo, scale,
           s, lds, t, ldt, g, ldg,
           u1, ldu1, u2, ldu2, v1, ldv1, v2, ldv2,
           &m_out, wr, wi,
           us, ldus, uu, lduu,
           lwork_arr, iwork, dwork, ldwork, &info);

    PyArray_ResolveWritebackIfCopy(s_array);
    PyArray_ResolveWritebackIfCopy(t_array);
    PyArray_ResolveWritebackIfCopy(g_array);
    PyArray_ResolveWritebackIfCopy(u1_array);
    PyArray_ResolveWritebackIfCopy(u2_array);
    PyArray_ResolveWritebackIfCopy(v1_array);
    PyArray_ResolveWritebackIfCopy(v2_array);

    npy_intp dims_m = m_out > 0 ? m_out : 0;
    PyArrayObject *wr_array = (PyArrayObject *)PyArray_SimpleNew(1, &dims_m, NPY_DOUBLE);
    PyArrayObject *wi_array = (PyArrayObject *)PyArray_SimpleNew(1, &dims_m, NPY_DOUBLE);

    npy_intp dims_us[2] = {n2 > 0 ? n2 : 0, m_out > 0 ? m_out : 0};
    PyArrayObject *us_array = (PyArrayObject *)PyArray_New(
        &PyArray_Type, 2, dims_us, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyArrayObject *uu_array = (PyArrayObject *)PyArray_New(
        &PyArray_Type, 2, dims_us, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (us_array) {
        f64 *us_ptr = (f64*)PyArray_DATA(us_array);
        memset(us_ptr, 0, PyArray_NBYTES(us_array));
    }
    if (uu_array) {
        f64 *uu_ptr = (f64*)PyArray_DATA(uu_array);
        memset(uu_ptr, 0, PyArray_NBYTES(uu_array));
    }
    // Need to copy results from temporary buffers to arrays?
    // Wait, mb03zd computes into `us` and `uu`.
    // I need to use `memcpy` if I allocate new arrays.
    // OR allocate arrays BEFORE calling mb03zd.
    // `us` and `uu` are outputs.
    // Logic in mb03zd: `us`, `uu` passed as `f64*`.
    // So I should allocate ARRAYS first, get pointers, then call mb03zd.
    // Re-doing the fix to be "Allocate Upfront".

    if (!wr_array || !wi_array || !us_array || !uu_array) {
        free(select);
        free(lwork_arr);
        free(iwork);
        free(dwork);
        free(wr);
        free(wi);
        if (!us_array) free(us);
        if (!uu_array) free(uu);
        Py_DECREF(scale_array);
        Py_DECREF(s_array);
        Py_DECREF(t_array);
        Py_DECREF(g_array);
        Py_DECREF(u1_array);
        Py_DECREF(u2_array);
        Py_DECREF(v1_array);
        Py_DECREF(v2_array);
        Py_XDECREF(wr_array);
        Py_XDECREF(wi_array);
        Py_XDECREF(us_array);
        Py_XDECREF(uu_array);
        PyErr_NoMemory();
        return NULL;
    }

    // Arrays implicitly own data

    if (m_out > 0) {
        memcpy(PyArray_DATA(wr_array), wr, m_out * sizeof(f64));
        memcpy(PyArray_DATA(wi_array), wi, m_out * sizeof(f64));
        if (n2 > 0 && us_array && uu_array) {
             memcpy(PyArray_DATA(us_array), us, n2 * m_out * sizeof(f64));
             memcpy(PyArray_DATA(uu_array), uu, n2 * m_out * sizeof(f64));
        }
    }

    free(us);
    free(uu);
    // Cleaned up

    free(select);
    free(lwork_arr);
    free(iwork);
    free(dwork);
    free(wr);
    free(wi);

    PyObject *result = Py_BuildValue("(iOOOOi)",
        (int)m_out,
        (PyObject *)wr_array,
        (PyObject *)wi_array,
        (PyObject *)us_array,
        (PyObject *)uu_array,
        (int)info);

    Py_DECREF(scale_array);
    Py_DECREF(s_array);
    Py_DECREF(t_array);
    Py_DECREF(g_array);
    Py_DECREF(u1_array);
    Py_DECREF(u2_array);
    Py_DECREF(v1_array);
    Py_DECREF(v2_array);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);
    Py_DECREF(us_array);
    Py_DECREF(uu_array);

    return result;
}


/* Python wrapper for mb03ah */
PyObject* py_mb03ah(PyObject* self, PyObject* args) {
    const char *shft;
    i32 k, n;
    PyObject *amap_obj, *s_obj, *a_obj;
    i32 sinv;

    if (!PyArg_ParseTuple(args, "siiOOiO", &shft, &k, &n, &amap_obj, &s_obj,
                          &sinv, &a_obj)) {
        return NULL;
    }

    if (k < 1) {
        PyErr_SetString(PyExc_ValueError, "K must be >= 1");
        return NULL;
    }
    if (n < 2) {
        PyErr_SetString(PyExc_ValueError, "N must be >= 2");
        return NULL;
    }

    PyArrayObject *amap_array = (PyArrayObject*)PyArray_FROM_OTF(
        amap_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (amap_array == NULL) return NULL;

    PyArrayObject *s_array = (PyArrayObject*)PyArray_FROM_OTF(
        s_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (s_array == NULL) {
        Py_DECREF(amap_array);
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (a_array == NULL) {
        Py_DECREF(amap_array);
        Py_DECREF(s_array);
        return NULL;
    }

    const i32 *amap_data = (const i32*)PyArray_DATA(amap_array);
    const i32 *s_data = (const i32*)PyArray_DATA(s_array);
    const f64 *a_data = (const f64*)PyArray_DATA(a_array);

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda1 = (i32)a_dims[0];
    i32 lda2 = (i32)a_dims[1];

    f64 c1 = 0.0, s1 = 0.0, c2 = 0.0, s2 = 0.0;

    mb03ah(shft, k, n, amap_data, s_data, sinv, a_data, lda1, lda2,
           &c1, &s1, &c2, &s2);

    Py_DECREF(amap_array);
    Py_DECREF(s_array);
    Py_DECREF(a_array);

    return Py_BuildValue("(ddddi)", c1, s1, c2, s2, 0);
}
