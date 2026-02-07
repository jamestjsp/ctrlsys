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



/* Python wrapper for mb04od */
PyObject* py_mb04od(PyObject* self, PyObject* args) {
    const char *uplo_str;
    char uplo;
    i32 n, m, p, ldr, lda, ldb, ldc;
    PyObject *r_obj, *a_obj, *b_obj, *c_obj;
    PyArrayObject *r_array, *a_array, *b_array, *c_array, *tau_array;
    f64 *dwork;
    i32 ldwork;

    if (!PyArg_ParseTuple(args, "siiiOOOO",
                          &uplo_str, &n, &m, &p,
                          &r_obj, &a_obj, &b_obj, &c_obj)) {
        return NULL;
    }

    uplo = uplo_str[0];

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        Py_DECREF(r_array);
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(a_array);
        return NULL;
    }

    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    npy_intp *r_dims = PyArray_DIMS(r_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    ldr = (i32)r_dims[0];
    lda = (i32)a_dims[0];
    ldb = (i32)b_dims[0];
    ldc = (i32)c_dims[0];

    npy_intp tau_dims[1] = {n > 0 ? n : 1};
    tau_array = (PyArrayObject*)PyArray_SimpleNew(1, tau_dims, NPY_DOUBLE);
    if (tau_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    ldwork = (n - 1) > m ? (n - 1) : m;
    if (ldwork < 1) ldwork = 1;
    dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(tau_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *r_data = (f64*)PyArray_DATA(r_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *tau_data = (f64*)PyArray_DATA(tau_array);

    mb04od(&uplo, n, m, p, r_data, ldr, a_data, lda,
           b_data, ldb, c_data, ldc, tau_data, dwork);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(r_array);
    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyObject *result = Py_BuildValue("OOOOO", r_array, a_array, b_array, c_array, tau_array);
    Py_DECREF(r_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(tau_array);
    return result;
}



PyObject* py_mb04kd(PyObject* self, PyObject* args) {
    const char *uplo_str;
    i32 n, m, p;
    PyObject *r_obj, *a_obj, *b_obj;
    PyArrayObject *r_array, *a_array, *b_array;

    if (!PyArg_ParseTuple(args, "siiiOOO", &uplo_str, &n, &m, &p,
                          &r_obj, &a_obj, &b_obj)) {
        return NULL;
    }

    if (uplo_str == NULL || uplo_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "uplo must be a non-empty string");
        return NULL;
    }
    char uplo = uplo_str[0];

    /* Convert input arrays - in-place modification */
    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) return NULL;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        Py_DECREF(r_array);
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(a_array);
        return NULL;
    }

    /* Get dimensions */
    npy_intp *r_dims = PyArray_DIMS(r_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);

    i32 ldr = (i32)r_dims[0];
    i32 lda = (i32)a_dims[0];
    i32 ldb = (i32)b_dims[0];
    i32 ldc = n > 0 ? n : 1;

    /* Allocate output arrays */
    npy_intp c_dims[2] = {n, m};
    npy_intp c_strides[2] = {sizeof(f64), ldc * sizeof(f64)};
    PyObject *c_array = PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE,
                                    c_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (c_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate C matrix");
        return NULL;
    }
    f64 *c_data = (f64*)PyArray_DATA((PyArrayObject*)c_array);
    if (n > 0 && m > 0) {
        memset(c_data, 0, ldc * m * sizeof(f64));
    }

    npy_intp tau_dims[1] = {n > 0 ? n : 1};
    PyObject *tau_array = PyArray_SimpleNew(1, tau_dims, NPY_DOUBLE);
    if (tau_array == NULL) {
        Py_DECREF(c_array);
        Py_DECREF(r_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate tau");
        return NULL;
    }
    f64 *tau = (f64*)PyArray_DATA((PyArrayObject*)tau_array);
    memset(tau, 0, (n > 0 ? n : 1) * sizeof(f64));

    f64 *dwork = (f64*)calloc(n > 0 ? n : 1, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(c_array);
        Py_DECREF(tau_array);
        Py_DECREF(r_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    /* Call C function */
    f64 *r_data = (f64*)PyArray_DATA(r_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    mb04kd(uplo, n, m, p, r_data, ldr, a_data, lda, b_data, ldb,
           c_data, ldc, tau, dwork);

    /* Resolve writeback */
    PyArray_ResolveWritebackIfCopy(r_array);
    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    free(dwork);

    /* Return (R_bar, A_out, D, C, tau) */
    PyObject *result = Py_BuildValue("OOOOO", r_array, a_array, b_array, c_array, tau_array);
    Py_DECREF(r_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(tau_array);

    return result;
}



/* Python wrapper for mb04ow */
PyObject* py_mb04ow(PyObject* self, PyObject* args, PyObject* kwargs) {
    i32 m, n, p, incx = 1, incd = 1;
    PyObject *a_obj, *t_obj, *x_obj, *b_obj, *c_obj, *d_obj;
    PyArrayObject *a_array, *t_array, *x_array, *b_array, *c_array, *d_array;

    static char *kwlist[] = {"m", "n", "p", "a", "t", "x", "b", "c", "d", "incx", "incd", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiiOOOOOO|ii", kwlist,
                                     &m, &n, &p, &a_obj, &t_obj, &x_obj, 
                                     &b_obj, &c_obj, &d_obj, &incx, &incd)) {
        return NULL;
    }
    
    if (m < 0 || n < 0 || p < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative");
        return NULL;
    }
    
    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    t_array = (PyArrayObject*)PyArray_FROM_OTF(t_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    
    if (!a_array || !t_array || !x_array || !b_array || !c_array || !d_array) {
         Py_XDECREF(a_array); Py_XDECREF(t_array); Py_XDECREF(x_array);
         Py_XDECREF(b_array); Py_XDECREF(c_array); Py_XDECREF(d_array);
         return NULL;
    }
    
    i32 lda = (i32)PyArray_DIM(a_array, 0);
    i32 ldt = (i32)PyArray_DIM(t_array, 0);
    i32 ldb = (i32)PyArray_DIM(b_array, 0);
    i32 ldc = (i32)PyArray_DIM(c_array, 0);
    
    if (lda < 1) lda = 1;
    if (ldt < 1) ldt = 1;
    if (ldb < 1) ldb = 1;
    if (ldc < 1) ldc = 1;
    
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *t_data = (f64*)PyArray_DATA(t_array);
    f64 *x_data = (f64*)PyArray_DATA(x_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    
    mb04ow(m, n, p, a_data, lda, t_data, ldt, x_data, incx, b_data, ldb, c_data, ldc, d_data, incd);
    
    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(t_array);
    PyArray_ResolveWritebackIfCopy(x_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);
    
    PyObject *result = Py_BuildValue("OOOOOO", a_array, t_array, x_array, b_array, c_array, d_array);
    
    Py_DECREF(a_array);
    Py_DECREF(t_array);
    Py_DECREF(x_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    return result;
}



PyObject* py_mb04dd(PyObject* self, PyObject* args) {
    const char *job;
    PyObject *a_obj, *qg_obj;
    PyArrayObject *a_array = NULL, *qg_array = NULL;
    i32 info = 0;

    if (!PyArg_ParseTuple(args, "sOO", &job, &a_obj, &qg_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    qg_array = (PyArrayObject*)PyArray_FROM_OTF(qg_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || qg_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(qg_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 n = (i32)a_dims[0];
    i32 lda = n > 0 ? n : 1;
    i32 ldqg = n > 0 ? n : 1;

    npy_intp scale_dims[1] = {n > 0 ? n : 1};
    PyObject *scale_array = PyArray_SimpleNew(1, scale_dims, NPY_DOUBLE);
    if (scale_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(qg_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate scale array");
        return NULL;
    }
    f64 *scale = (f64*)PyArray_DATA((PyArrayObject*)scale_array);

    i32 ilo;
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *qg_data = (f64*)PyArray_DATA(qg_array);

    mb04dd(job, n, a_data, lda, qg_data, ldqg, &ilo, scale, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(qg_array);

    PyObject *result = Py_BuildValue("(OOiOi)", a_array, qg_array, ilo, scale_array, info);

    Py_DECREF(a_array);
    Py_DECREF(qg_array);
    Py_DECREF(scale_array);

    return result;
}



PyObject* py_mb04id(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"n", "m", "p", "a", "b", "l", "ldwork", NULL};
    i32 n, m, p, l = 0;
    i32 ldwork = 0;
    PyObject *a_obj, *b_obj = NULL;
    PyArrayObject *a_array = NULL, *b_array = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiiO|Oii", kwlist,
                                     &n, &m, &p, &a_obj, &b_obj, &l, &ldwork)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda = (i32)a_dims[0];

    i32 ldb = n > 0 ? n : 1;
    bool has_b = (b_obj != NULL && l > 0);

    if (has_b) {
        b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (b_array == NULL) {
            Py_DECREF(a_array);
            return NULL;
        }
        npy_intp *b_dims = PyArray_DIMS(b_array);
        ldb = (i32)b_dims[0];
    }

    i32 minwork = 1;
    if (m > 1 && m - 1 > minwork) minwork = m - 1;
    if (m > p && m - p > minwork) minwork = m - p;
    if (l > minwork) minwork = l;

    if (ldwork == 0) ldwork = minwork;

    f64 *dwork = (f64*)calloc(ldwork > 0 ? ldwork : 1, sizeof(f64));
    if (dwork == NULL) {
        Py_XDECREF(b_array);
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    i32 minval = (n < m ? n : m);
    npy_intp tau_dims[1] = {minval > 0 ? minval : 1};
    PyObject *tau_array = PyArray_SimpleNew(1, tau_dims, NPY_DOUBLE);
    if (tau_array == NULL) {
        free(dwork);
        Py_XDECREF(b_array);
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate tau");
        return NULL;
    }
    f64 *tau = (f64*)PyArray_DATA((PyArrayObject*)tau_array);
    memset(tau, 0, (minval > 0 ? minval : 1) * sizeof(f64));

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = has_b ? (f64*)PyArray_DATA(b_array) : NULL;
    f64 dummy_b = 0.0;
    if (!has_b) {
        b_data = &dummy_b;
        ldb = 1;
    }

    i32 info;
    mb04id(n, m, p, l, a_data, lda, b_data, ldb, tau, dwork, ldwork, &info);

    f64 optimal_work = dwork[0];
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    if (has_b) {
        PyArray_ResolveWritebackIfCopy(b_array);
    }

    if (info != 0) {
        Py_DECREF(tau_array);
        Py_XDECREF(b_array);
        Py_DECREF(a_array);
        PyErr_Format(PyExc_ValueError, "mb04id failed with info=%d", info);
        return NULL;
    }

    PyObject *result;
    if (ldwork == -1) {
        if (has_b) {
            result = Py_BuildValue("OOOid", a_array, b_array, tau_array, info, optimal_work);
        } else {
            result = Py_BuildValue("OOid", a_array, tau_array, info, optimal_work);
        }
    } else {
        if (has_b) {
            result = Py_BuildValue("OOOi", a_array, b_array, tau_array, info);
        } else {
            result = Py_BuildValue("OOi", a_array, tau_array, info);
        }
    }

    Py_XDECREF(b_array);
    Py_DECREF(a_array);
    Py_DECREF(tau_array);

    return result;
}



PyObject* py_mb04iy(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"side", "trans", "a", "tau", "c", "p", NULL};
    const char *side, *trans;
    i32 p = 0;
    PyObject *a_obj, *tau_obj, *c_obj;
    PyArrayObject *a_array, *tau_array, *c_array;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssOOO|i", kwlist,
                                     &side, &trans, &a_obj, &tau_obj, &c_obj, &p)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    tau_array = (PyArrayObject*)PyArray_FROM_OTF(tau_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (tau_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(tau_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);
    npy_intp *tau_dims = PyArray_DIMS(tau_array);

    i32 lda = (i32)a_dims[0];
    i32 ldc = (i32)c_dims[0];
    i32 n = (i32)c_dims[0];
    i32 m = (i32)c_dims[1];
    i32 k = (i32)tau_dims[0];

    bool left = (side[0] == 'L' || side[0] == 'l');
    i32 ldwork = left ? (m > 0 ? m : 1) : (n > 0 ? n : 1);

    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(tau_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *tau_data = (f64*)PyArray_DATA(tau_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    i32 info;
    mb04iy(side, trans, n, m, k, p, a_data, lda, tau_data, c_data, ldc, dwork, ldwork, &info);

    free(dwork);

    // Resolve WRITEBACKIFCOPY for 'a' array before DECREF
    if (PyArray_ResolveWritebackIfCopy(a_array) < 0) {
        Py_DECREF(a_array);
        Py_DECREF(tau_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_RuntimeError, "Failed to resolve WRITEBACKIFCOPY for array 'a'");
        return NULL;
    }

    // Resolve WRITEBACKIFCOPY for 'c' array
    if (PyArray_ResolveWritebackIfCopy(c_array) < 0) {
        Py_DECREF(a_array);
        Py_DECREF(tau_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_RuntimeError, "Failed to resolve WRITEBACKIFCOPY for array 'c'");
        return NULL;
    }

    if (info != 0) {
        Py_DECREF(a_array);
        Py_DECREF(tau_array);
        Py_DECREF(c_array);
        PyErr_Format(PyExc_ValueError, "mb04iy failed with info=%d", info);
        return NULL;
    }

    Py_DECREF(a_array);
    Py_DECREF(tau_array);

    PyObject *result = Py_BuildValue("Oi", c_array, info);
    Py_DECREF(c_array);
    return result;
}



PyObject* py_mb04oy(PyObject* self, PyObject* args) {
    i32 m, n;
    f64 tau;
    PyObject *v_obj, *a_obj, *b_obj;
    PyArrayObject *v_array, *a_array, *b_array;

    if (!PyArg_ParseTuple(args, "iiOdOO", &m, &n, &v_obj, &tau, &a_obj, &b_obj)) {
        return NULL;
    }

    v_array = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (v_array == NULL) return NULL;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        Py_DECREF(v_array);
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(v_array);
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);

    i32 lda = (i32)a_dims[0];
    i32 ldb = (i32)b_dims[0];

    f64 *dwork = NULL;
    if (m + 1 >= 11) {
        dwork = (f64*)calloc(n > 0 ? n : 1, sizeof(f64));
        if (dwork == NULL) {
            Py_DECREF(v_array);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
            return NULL;
        }
    }

    f64 *v_data = (f64*)PyArray_DATA(v_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    SLC_MB04OY(m, n, v_data, tau, a_data, lda, b_data, ldb, dwork);

    if (dwork != NULL) {
        free(dwork);
    }

    Py_DECREF(v_array);

    PyObject *result = Py_BuildValue("OO", a_array, b_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    return result;
}



/* Python wrapper for mb04ny - Apply Householder reflector to [A B] from right */
PyObject* py_mb04ny(PyObject* self, PyObject* args) {
    i32 m, n, incv;
    f64 tau;
    PyObject *v_obj, *a_obj, *b_obj;
    PyArrayObject *v_array, *a_array, *b_array;

    if (!PyArg_ParseTuple(args, "iiOidOO", &m, &n, &v_obj, &incv, &tau, &a_obj, &b_obj)) {
        return NULL;
    }

    v_array = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (v_array == NULL) return NULL;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        Py_DECREF(v_array);
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(v_array);
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);

    i32 lda = (m > 0) ? (i32)a_dims[0] : 1;
    i32 ldb = (m > 0) ? (i32)b_dims[0] : 1;

    f64 *dwork = NULL;
    if (n + 1 >= 11) {
        dwork = (f64*)calloc(m > 0 ? m : 1, sizeof(f64));
        if (dwork == NULL) {
            Py_DECREF(v_array);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
            return NULL;
        }
    }

    f64 *v_data = (f64*)PyArray_DATA(v_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    SLC_MB04NY(m, n, v_data, incv, tau, a_data, lda, b_data, ldb, dwork);

    if (dwork != NULL) {
        free(dwork);
    }

    Py_DECREF(v_array);

    PyObject *result = Py_BuildValue("OO", a_array, b_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    return result;
}



/* Python wrapper for mb04py - Apply elementary reflector to matrix */
PyObject* py_mb04py(PyObject* self, PyObject* args) {
    const char* side_str;
    i32 m, n;
    f64 tau;
    PyObject *v_obj, *c_obj;
    PyArrayObject *v_array, *c_array;

    if (!PyArg_ParseTuple(args, "siiOdO", &side_str, &m, &n, &v_obj, &tau, &c_obj)) {
        return NULL;
    }

    char side = side_str[0];

    v_array = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (v_array == NULL) return NULL;

    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) {
        Py_DECREF(v_array);
        return NULL;
    }

    npy_intp *c_dims = PyArray_DIMS(c_array);
    i32 ldc = (i32)c_dims[0];

    i32 order = (side == 'L' || side == 'l') ? m : n;
    f64 *dwork = NULL;
    if (order >= 11) {
        i32 work_size = (side == 'L' || side == 'l') ? n : m;
        dwork = (f64*)calloc(work_size > 0 ? work_size : 1, sizeof(f64));
        if (dwork == NULL) {
            Py_DECREF(v_array);
            Py_DECREF(c_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
            return NULL;
        }
    }

    f64 *v_data = (f64*)PyArray_DATA(v_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    SLC_MB04PY(side, m, n, v_data, tau, c_data, ldc, dwork);

    if (dwork != NULL) {
        free(dwork);
    }

    Py_DECREF(v_array);

    PyObject *result = Py_BuildValue("O", c_array);
    Py_DECREF(c_array);
    return result;
}



/* Python wrapper for mb04nd - RQ factorization of structured block matrix */
PyObject* py_mb04nd(PyObject* self, PyObject* args) {
    const char* uplo_str;
    i32 n, m, p;
    PyObject *r_obj, *a_obj, *b_obj, *c_obj;
    PyArrayObject *r_array, *a_array, *b_array, *c_array;

    if (!PyArg_ParseTuple(args, "siiiOOOO", &uplo_str, &n, &m, &p,
                          &r_obj, &a_obj, &b_obj, &c_obj)) {
        return NULL;
    }

    char uplo = uplo_str[0];

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) return NULL;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        Py_DECREF(r_array);
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(a_array);
        return NULL;
    }

    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    npy_intp *r_dims = PyArray_DIMS(r_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    i32 ldr = (n > 0) ? (i32)r_dims[0] : 1;
    i32 lda = (n > 0) ? (i32)a_dims[0] : 1;
    i32 ldb = (m > 0) ? (i32)b_dims[0] : 1;
    i32 ldc = (m > 0) ? (i32)c_dims[0] : 1;

    npy_intp tau_dim = n > 0 ? n : 0;
    PyObject *tau_array = PyArray_SimpleNew(1, &tau_dim, NPY_DOUBLE);
    if (tau_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate tau");
        return NULL;
    }
    f64 *tau = (f64*)PyArray_DATA((PyArrayObject*)tau_array);
    if (n > 0) {
        memset(tau, 0, n * sizeof(f64));
    }

    i32 dwork_size = ((n - 1) > m) ? (n - 1) : m;
    if (dwork_size < 1) dwork_size = 1;
    f64 *dwork = (f64*)calloc(dwork_size, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(tau_array);
        Py_DECREF(r_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate dwork");
        return NULL;
    }

    f64 *r_data = (f64*)PyArray_DATA(r_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    SLC_MB04ND(&uplo, n, m, p, r_data, ldr, a_data, lda,
               b_data, ldb, c_data, ldc, tau, dwork);

    free(dwork);

    Py_DECREF(r_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);

    return tau_array;
}



/* Python wrapper for mb04zd */
PyObject* py_mb04zd(PyObject* self, PyObject* args) {
    const char *compu_str;
    char compu;
    i32 n;
    PyObject *a_obj, *qg_obj, *u_obj = NULL;
    PyArrayObject *a_array, *qg_array, *u_array = NULL;
    f64 *dwork = NULL;
    i32 info;

    if (!PyArg_ParseTuple(args, "siOO|O",
                          &compu_str, &n, &a_obj, &qg_obj, &u_obj)) {
        return NULL;
    }

    compu = compu_str[0];

    bool need_u = (compu == 'I' || compu == 'i' || compu == 'F' || compu == 'f' ||
                   compu == 'V' || compu == 'v' || compu == 'A' || compu == 'a');
    bool accum = (compu == 'V' || compu == 'v' || compu == 'A' || compu == 'a');

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    qg_array = (PyArrayObject*)PyArray_FROM_OTF(qg_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (qg_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *qg_dims = PyArray_DIMS(qg_array);
    i32 lda = (i32)a_dims[0];
    i32 ldqg = (i32)qg_dims[0];
    i32 ldu = (n > 0 && need_u) ? n : 1;

    if (accum && u_obj != NULL && u_obj != Py_None) {
        u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE,
                                                    NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (u_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(qg_array);
            return NULL;
        }
        ldu = (i32)PyArray_DIMS(u_array)[0];
    } else if (need_u && n > 0) {
        npy_intp u_dims[2] = {n, 2 * n};
        u_array = (PyArrayObject*)PyArray_ZEROS(2, u_dims, NPY_DOUBLE, 1);
        if (u_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(qg_array);
            PyErr_NoMemory();
            return NULL;
        }
        ldu = n;
    } else {
        npy_intp u_dims[2] = {1, 2};
        u_array = (PyArrayObject*)PyArray_ZEROS(2, u_dims, NPY_DOUBLE, 1);
        if (u_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(qg_array);
            PyErr_NoMemory();
            return NULL;
        }
        ldu = 1;
    }

    i32 ldwork = (n > 0) ? 2 * n : 1;
    dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(qg_array);
        Py_DECREF(u_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *qg_data = (f64*)PyArray_DATA(qg_array);
    f64 *u_data = (f64*)PyArray_DATA(u_array);

    mb04zd(&compu, n, a_data, lda, qg_data, ldqg, u_data, ldu, dwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(qg_array);
    if (accum && u_obj != NULL && u_obj != Py_None) {
        PyArray_ResolveWritebackIfCopy(u_array);
    }

    PyObject *result = Py_BuildValue("OOOi", a_array, qg_array, u_array, info);
    Py_DECREF(a_array);
    Py_DECREF(qg_array);
    Py_DECREF(u_array);

    return result;
}



/* Python wrapper for mb04tb (blocked version of mb04ts) */
PyObject* py_mb04tb(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *trana_str, *tranb_str;
    i32 n, ilo;
    int ldwork_arg = 0;
    PyObject *a_obj, *b_obj, *g_obj, *q_obj;
    PyArrayObject *a_array = NULL, *b_array = NULL, *g_array = NULL, *q_array = NULL;
    f64 *csl = NULL, *csr = NULL, *taul = NULL, *taur = NULL, *dwork = NULL;
    i32 info = 0;

    static char *kwlist[] = {"trana", "tranb", "n", "ilo", "a", "b", "g", "q",
                              "ldwork", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssiiOOOO|i", kwlist,
                                     &trana_str, &tranb_str, &n, &ilo,
                                     &a_obj, &b_obj, &g_obj, &q_obj, &ldwork_arg)) {
        return NULL;
    }

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
    i32 lda = (i32)a_dims[0];
    if (lda < 1) lda = 1;
    i32 ldb = lda;
    i32 ldg = lda;
    i32 ldq = lda;

    i32 ldwork;
    if (ldwork_arg == -1) {
        ldwork = -1;
        dwork = (f64*)malloc(sizeof(f64));
    } else if (ldwork_arg > 0) {
        ldwork = (i32)ldwork_arg;
        dwork = (f64*)malloc(ldwork * sizeof(f64));
    } else {
        ldwork = (n > 1) ? n : 1;
        dwork = (f64*)malloc(ldwork * sizeof(f64));
    }

    i32 csl_len = (n > 0) ? 2 * n : 2;
    i32 csr_len = (n > 1) ? 2 * (n - 1) : 0;
    i32 taul_len = (n > 0) ? n : 1;
    i32 taur_len = (n > 1) ? n - 1 : 0;

    npy_intp csl_dims[1] = {csl_len};
    npy_intp csr_dims[1] = {csr_len > 0 ? csr_len : 0};
    npy_intp taul_dims[1] = {taul_len};
    npy_intp taur_dims[1] = {taur_len > 0 ? taur_len : 0};

    PyObject *csl_array = PyArray_SimpleNew(1, csl_dims, NPY_DOUBLE);
    PyObject *taul_array = PyArray_SimpleNew(1, taul_dims, NPY_DOUBLE);
    PyObject *csr_array = (csr_len > 0) ? PyArray_SimpleNew(1, csr_dims, NPY_DOUBLE) : PyArray_ZEROS(1, csr_dims, NPY_DOUBLE, 0);
    PyObject *taur_array = (taur_len > 0) ? PyArray_SimpleNew(1, taur_dims, NPY_DOUBLE) : PyArray_ZEROS(1, taur_dims, NPY_DOUBLE, 0);

    if (csl_array == NULL || csr_array == NULL || taul_array == NULL || taur_array == NULL || dwork == NULL) {
        Py_XDECREF(csl_array);
        Py_XDECREF(csr_array);
        Py_XDECREF(taul_array);
        Py_XDECREF(taur_array);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(g_array);
        Py_DECREF(q_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    csl = (f64*)PyArray_DATA((PyArrayObject*)csl_array);
    taul = (f64*)PyArray_DATA((PyArrayObject*)taul_array);
    csr = (csr_len > 0) ? (f64*)PyArray_DATA((PyArrayObject*)csr_array) : NULL;
    taur = (taur_len > 0) ? (f64*)PyArray_DATA((PyArrayObject*)taur_array) : NULL;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *g_data = (f64*)PyArray_DATA(g_array);
    f64 *q_data = (f64*)PyArray_DATA(q_array);

    mb04tb(trana_str, tranb_str, n, ilo,
           a_data, lda, b_data, ldb, g_data, ldg, q_data, ldq,
           csl, csr, taul, taur, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(g_array);
    PyArray_ResolveWritebackIfCopy(q_array);

    PyObject *result = Py_BuildValue("OOOOOOOOi",
                                     a_array, b_array, g_array, q_array,
                                     csl_array, csr_array, taul_array, taur_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(g_array);
    Py_DECREF(q_array);
    Py_DECREF(csl_array);
    Py_DECREF(csr_array);
    Py_DECREF(taul_array);
    Py_DECREF(taur_array);

    return result;
}



/* Python wrapper for mb04ts */
PyObject* py_mb04ts(PyObject* self, PyObject* args) {
    const char *trana_str, *tranb_str;
    i32 n, ilo;
    PyObject *a_obj, *b_obj, *g_obj, *q_obj;
    PyArrayObject *a_array = NULL, *b_array = NULL, *g_array = NULL, *q_array = NULL;
    f64 *csl = NULL, *csr = NULL, *taul = NULL, *taur = NULL, *dwork = NULL;
    i32 info = 0;

    if (!PyArg_ParseTuple(args, "ssiiOOOO",
                          &trana_str, &tranb_str, &n, &ilo,
                          &a_obj, &b_obj, &g_obj, &q_obj)) {
        return NULL;
    }

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
    i32 lda = (i32)a_dims[0];
    i32 ldb = lda;
    i32 ldg = lda;
    i32 ldq = lda;
    i32 ldwork = (n > 1) ? n : 1;

    i32 csl_len = (n > 0) ? 2 * n : 2;
    i32 csr_len = (n > 1) ? 2 * (n - 1) : 0;
    i32 taul_len = (n > 0) ? n : 1;
    i32 taur_len = (n > 1) ? n - 1 : 0;

    npy_intp csl_dims[1] = {csl_len};
    npy_intp csr_dims[1] = {csr_len > 0 ? csr_len : 0};
    npy_intp taul_dims[1] = {taul_len};
    npy_intp taur_dims[1] = {taur_len > 0 ? taur_len : 0};

    dwork = (f64*)malloc(ldwork * sizeof(f64));
    PyObject *csl_array = PyArray_SimpleNew(1, csl_dims, NPY_DOUBLE);
    PyObject *taul_array = PyArray_SimpleNew(1, taul_dims, NPY_DOUBLE);
    PyObject *csr_array = (csr_len > 0) ? PyArray_SimpleNew(1, csr_dims, NPY_DOUBLE) : PyArray_ZEROS(1, csr_dims, NPY_DOUBLE, 0);
    PyObject *taur_array = (taur_len > 0) ? PyArray_SimpleNew(1, taur_dims, NPY_DOUBLE) : PyArray_ZEROS(1, taur_dims, NPY_DOUBLE, 0);

    if (csl_array == NULL || csr_array == NULL || taul_array == NULL || taur_array == NULL || dwork == NULL) {
        Py_XDECREF(csl_array);
        Py_XDECREF(csr_array);
        Py_XDECREF(taul_array);
        Py_XDECREF(taur_array);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(g_array);
        Py_DECREF(q_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    csl = (f64*)PyArray_DATA((PyArrayObject*)csl_array);
    taul = (f64*)PyArray_DATA((PyArrayObject*)taul_array);
    csr = (csr_len > 0) ? (f64*)PyArray_DATA((PyArrayObject*)csr_array) : NULL;
    taur = (taur_len > 0) ? (f64*)PyArray_DATA((PyArrayObject*)taur_array) : NULL;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *g_data = (f64*)PyArray_DATA(g_array);
    f64 *q_data = (f64*)PyArray_DATA(q_array);

    mb04ts(trana_str, tranb_str, n, ilo,
           a_data, lda, b_data, ldb, g_data, ldg, q_data, ldq,
           csl, csr, taul, taur, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(g_array);
    PyArray_ResolveWritebackIfCopy(q_array);

    PyObject *result = Py_BuildValue("OOOOOOOOi",
                                     a_array, b_array, g_array, q_array,
                                     csl_array, csr_array, taul_array, taur_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(g_array);
    Py_DECREF(q_array);
    Py_DECREF(csl_array);
    Py_DECREF(csr_array);
    Py_DECREF(taul_array);
    Py_DECREF(taur_array);

    return result;
}



/* Python wrapper for mb04qc */
PyObject* py_mb04qc(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *strab_str, *trana_str, *tranb_str, *tranq_str;
    const char *direct_str, *storev_str, *storew_str;
    PyObject *v_obj, *w_obj, *rs_obj, *t_obj, *a_obj, *b_obj;
    PyArrayObject *v_array = NULL, *w_array = NULL, *rs_array = NULL, *t_array = NULL;
    PyArrayObject *a_array = NULL, *b_array = NULL;
    f64 *dwork = NULL;

    static char *kwlist[] = {"strab", "trana", "tranb", "tranq", "direct",
                              "storev", "storew", "m", "n", "k",
                              "v", "w", "rs", "t", "a", "b", NULL};

    int m_arg, n_arg, k_arg;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssssssiiiOOOOOO", kwlist,
                                     &strab_str, &trana_str, &tranb_str, &tranq_str,
                                     &direct_str, &storev_str, &storew_str,
                                     &m_arg, &n_arg, &k_arg,
                                     &v_obj, &w_obj, &rs_obj, &t_obj, &a_obj, &b_obj)) {
        return NULL;
    }

    v_array = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    w_array = (PyArrayObject*)PyArray_FROM_OTF(w_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    rs_array = (PyArrayObject*)PyArray_FROM_OTF(rs_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    t_array = (PyArrayObject*)PyArray_FROM_OTF(t_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (v_array == NULL || w_array == NULL || rs_array == NULL || t_array == NULL ||
        a_array == NULL || b_array == NULL) {
        Py_XDECREF(v_array);
        Py_XDECREF(w_array);
        Py_XDECREF(rs_array);
        Py_XDECREF(t_array);
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        return NULL;
    }

    i32 m = (i32)m_arg;
    i32 n = (i32)n_arg;
    i32 k = (i32)k_arg;

    i32 ldv = (i32)PyArray_DIMS(v_array)[0];
    i32 ldw = (i32)PyArray_DIMS(w_array)[0];
    i32 ldrs = (i32)PyArray_DIMS(rs_array)[0];
    i32 ldt = (i32)PyArray_DIMS(t_array)[0];
    i32 lda = (i32)PyArray_DIMS(a_array)[0];
    i32 ldb = (i32)PyArray_DIMS(b_array)[0];

    bool la1b1 = (strab_str[0] == 'N' || strab_str[0] == 'n');
    i32 ldwork = la1b1 ? 9 * n * k : 8 * n * k;
    if (ldwork <= 0) ldwork = 1;
    dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (dwork == NULL) {
        Py_DECREF(v_array);
        Py_DECREF(w_array);
        Py_DECREF(rs_array);
        Py_DECREF(t_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    f64 *v_data = (f64*)PyArray_DATA(v_array);
    f64 *w_data = (f64*)PyArray_DATA(w_array);
    f64 *rs_data = (f64*)PyArray_DATA(rs_array);
    f64 *t_data = (f64*)PyArray_DATA(t_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    mb04qc(strab_str, trana_str, tranb_str, tranq_str, direct_str, storev_str, storew_str,
           m, n, k, v_data, ldv, w_data, ldw, rs_data, ldrs, t_data, ldt,
           a_data, lda, b_data, ldb, dwork);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    PyObject *result = Py_BuildValue("OO", a_array, b_array);

    Py_DECREF(v_array);
    Py_DECREF(w_array);
    Py_DECREF(rs_array);
    Py_DECREF(t_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);

    return result;
}



/* Python wrapper for mb04qf */
PyObject* py_mb04qf(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *direct_str, *storev_str, *storew_str;
    PyObject *v_obj, *w_obj, *cs_obj, *tau_obj;
    PyArrayObject *v_array = NULL, *w_array = NULL;
    PyArrayObject *cs_array = NULL, *tau_array = NULL;
    PyArrayObject *rs_array = NULL, *t_array = NULL;
    f64 *dwork = NULL;

    static char *kwlist[] = {"direct", "storev", "storew", "n", "k",
                              "v", "w", "cs", "tau", NULL};

    int n_arg = -1, k_arg = -1;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssiiOOOO", kwlist,
                                     &direct_str, &storev_str, &storew_str,
                                     &n_arg, &k_arg,
                                     &v_obj, &w_obj, &cs_obj, &tau_obj)) {
        return NULL;
    }

    v_array = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    w_array = (PyArrayObject*)PyArray_FROM_OTF(w_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    cs_array = (PyArrayObject*)PyArray_FROM_OTF(cs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    tau_array = (PyArrayObject*)PyArray_FROM_OTF(tau_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (v_array == NULL || w_array == NULL || cs_array == NULL || tau_array == NULL) {
        Py_XDECREF(v_array);
        Py_XDECREF(w_array);
        Py_XDECREF(cs_array);
        Py_XDECREF(tau_array);
        return NULL;
    }

    i32 n = (i32)n_arg;
    i32 k = (i32)k_arg;

    npy_intp *v_dims = PyArray_DIMS(v_array);
    i32 ldv = (i32)v_dims[0];
    i32 ldw = (i32)PyArray_DIMS(w_array)[0];

    i32 ldrs = (k > 1) ? k : 1;
    i32 ldt = (k > 1) ? k : 1;

    npy_intp rs_dims[2] = {ldrs, 6 * k};
    npy_intp t_dims[2] = {ldt, 9 * k};
    npy_intp rs_strides[2] = {sizeof(f64), ldrs * sizeof(f64)};
    npy_intp t_strides[2] = {sizeof(f64), ldt * sizeof(f64)};

    i32 ldwork = 3 * k;
    dwork = (f64*)malloc(ldwork * sizeof(f64));

    rs_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, rs_dims, NPY_DOUBLE,
                                            rs_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    t_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, t_dims, NPY_DOUBLE,
                                           t_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (rs_array == NULL || t_array == NULL || dwork == NULL) {
        Py_XDECREF(rs_array);
        Py_XDECREF(t_array);
        free(dwork);
        Py_DECREF(v_array);
        Py_DECREF(w_array);
        Py_DECREF(cs_array);
        Py_DECREF(tau_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }

    f64 *rs_data = (f64*)PyArray_DATA(rs_array);
    f64 *t_data = (f64*)PyArray_DATA(t_array);
    memset(rs_data, 0, ldrs * 6 * k * sizeof(f64));
    memset(t_data, 0, ldt * 9 * k * sizeof(f64));

    f64 *v_data = (f64*)PyArray_DATA(v_array);
    f64 *w_data = (f64*)PyArray_DATA(w_array);
    f64 *cs_data = (f64*)PyArray_DATA(cs_array);
    f64 *tau_data = (f64*)PyArray_DATA(tau_array);

    mb04qf(direct_str, storev_str, storew_str,
           n, k, v_data, ldv, w_data, ldw,
           cs_data, tau_data, rs_data, ldrs, t_data, ldt, dwork);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(v_array);
    PyArray_ResolveWritebackIfCopy(w_array);

    i32 info = 0;
    PyObject *result = Py_BuildValue("OOi", rs_array, t_array, info);

    Py_DECREF(v_array);
    Py_DECREF(w_array);
    Py_DECREF(cs_array);
    Py_DECREF(tau_array);
    Py_DECREF(rs_array);
    Py_DECREF(t_array);

    return result;
}



/* Python wrapper for mb04qb (blocked version of mb04qu) */
PyObject* py_mb04qb(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *tranc_str, *trand_str, *tranq_str, *storev_str, *storew_str;
    PyObject *c_obj, *d_obj, *v_obj, *w_obj, *cs_obj, *tau_obj;
    int m_arg = -1, n_arg = -1, k_arg = -1, ldwork_arg = -1;
    PyArrayObject *c_array = NULL, *d_array = NULL, *v_array = NULL, *w_array = NULL;
    PyArrayObject *cs_array = NULL, *tau_array = NULL;
    f64 *dwork = NULL;
    i32 info = 0;

    static char *kwlist[] = {"tranc", "trand", "tranq", "storev", "storew",
                              "m", "n", "k", "v", "w", "c", "d", "cs", "tau",
                              "ldwork", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssssiiiOOOOOO|i", kwlist,
                                     &tranc_str, &trand_str, &tranq_str,
                                     &storev_str, &storew_str,
                                     &m_arg, &n_arg, &k_arg,
                                     &v_obj, &w_obj, &c_obj, &d_obj,
                                     &cs_obj, &tau_obj, &ldwork_arg)) {
        return NULL;
    }

    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    v_array = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    w_array = (PyArrayObject*)PyArray_FROM_OTF(w_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    cs_array = (PyArrayObject*)PyArray_FROM_OTF(cs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    tau_array = (PyArrayObject*)PyArray_FROM_OTF(tau_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (c_array == NULL || d_array == NULL || v_array == NULL || w_array == NULL ||
        cs_array == NULL || tau_array == NULL) {
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        Py_XDECREF(v_array);
        Py_XDECREF(w_array);
        Py_XDECREF(cs_array);
        Py_XDECREF(tau_array);
        return NULL;
    }

    npy_intp *c_dims = PyArray_DIMS(c_array);
    npy_intp *v_dims = PyArray_DIMS(v_array);

    i32 m = (i32)m_arg;
    i32 n = (i32)n_arg;
    i32 k = (i32)k_arg;

    i32 ldc = (i32)c_dims[0];
    i32 ldd = (i32)PyArray_DIMS(d_array)[0];
    i32 ldv = (i32)v_dims[0];
    i32 ldw = (i32)PyArray_DIMS(w_array)[0];

    i32 ldwork;
    if (ldwork_arg == -1) {
        ldwork = -1;
        dwork = (f64*)malloc(sizeof(f64));
    } else if (ldwork_arg > 0) {
        ldwork = (i32)ldwork_arg;
        dwork = (f64*)malloc(ldwork * sizeof(f64));
    } else {
        ldwork = (n > 1) ? n : 1;
        dwork = (f64*)malloc(ldwork * sizeof(f64));
    }

    if (dwork == NULL) {
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(v_array);
        Py_DECREF(w_array);
        Py_DECREF(cs_array);
        Py_DECREF(tau_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    const f64 *v_data = (const f64*)PyArray_DATA(v_array);
    const f64 *w_data = (const f64*)PyArray_DATA(w_array);
    const f64 *cs_data = (const f64*)PyArray_DATA(cs_array);
    const f64 *tau_data = (const f64*)PyArray_DATA(tau_array);

    mb04qb(tranc_str, trand_str, tranq_str, storev_str, storew_str,
           m, n, k, v_data, ldv, w_data, ldw,
           c_data, ldc, d_data, ldd, cs_data, tau_data,
           dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    PyObject *result = Py_BuildValue("OOi", c_array, d_array, info);

    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(v_array);
    Py_DECREF(w_array);
    Py_DECREF(cs_array);
    Py_DECREF(tau_array);

    return result;
}



/* Python wrapper for mb04qu */
PyObject* py_mb04qu(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *tranc_str, *trand_str, *tranq_str, *storev_str, *storew_str;
    PyObject *c_obj, *d_obj, *v_obj, *w_obj, *cs_obj, *tau_obj;
    int k_arg = -1;
    PyArrayObject *c_array = NULL, *d_array = NULL, *v_array = NULL, *w_array = NULL;
    PyArrayObject *cs_array = NULL, *tau_array = NULL;
    f64 *dwork = NULL;
    i32 info = 0;

    static char *kwlist[] = {"tranc", "trand", "tranq", "storev", "storew",
                              "c", "d", "v", "w", "cs", "tau", "k", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssssOOOOOO|i", kwlist,
                                     &tranc_str, &trand_str, &tranq_str,
                                     &storev_str, &storew_str,
                                     &c_obj, &d_obj, &v_obj, &w_obj,
                                     &cs_obj, &tau_obj, &k_arg)) {
        return NULL;
    }

    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    v_array = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    w_array = (PyArrayObject*)PyArray_FROM_OTF(w_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    cs_array = (PyArrayObject*)PyArray_FROM_OTF(cs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    tau_array = (PyArrayObject*)PyArray_FROM_OTF(tau_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (c_array == NULL || d_array == NULL || v_array == NULL || w_array == NULL ||
        cs_array == NULL || tau_array == NULL) {
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        Py_XDECREF(v_array);
        Py_XDECREF(w_array);
        Py_XDECREF(cs_array);
        Py_XDECREF(tau_array);
        return NULL;
    }

    npy_intp *c_dims = PyArray_DIMS(c_array);
    npy_intp *v_dims = PyArray_DIMS(v_array);

    bool ltrc = (tranc_str[0] == 'T' || tranc_str[0] == 't' ||
                 tranc_str[0] == 'C' || tranc_str[0] == 'c');
    bool lcolv = (storev_str[0] == 'C' || storev_str[0] == 'c');

    i32 m, n, k;
    if (ltrc) {
        m = (i32)c_dims[1];
        n = (i32)c_dims[0];
    } else {
        m = (i32)c_dims[0];
        n = (i32)c_dims[1];
    }

    if (k_arg >= 0) {
        k = (i32)k_arg;
    } else if (lcolv) {
        k = (i32)v_dims[1];
    } else {
        k = (i32)v_dims[0];
    }

    i32 ldc = (i32)c_dims[0];
    i32 ldd = (i32)PyArray_DIMS(d_array)[0];
    i32 ldv = (i32)v_dims[0];
    i32 ldw = (i32)PyArray_DIMS(w_array)[0];

    i32 ldwork = (n > 1) ? n : 1;
    dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(v_array);
        Py_DECREF(w_array);
        Py_DECREF(cs_array);
        Py_DECREF(tau_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *v_data = (f64*)PyArray_DATA(v_array);
    f64 *w_data = (f64*)PyArray_DATA(w_array);
    f64 *cs_data = (f64*)PyArray_DATA(cs_array);
    f64 *tau_data = (f64*)PyArray_DATA(tau_array);

    mb04qu(tranc_str, trand_str, tranq_str, storev_str, storew_str,
           m, n, k, v_data, ldv, w_data, ldw,
           c_data, ldc, d_data, ldd, cs_data, tau_data,
           dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(v_array);
    PyArray_ResolveWritebackIfCopy(w_array);

    PyObject *result = Py_BuildValue("OOi", c_array, d_array, info);

    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(v_array);
    Py_DECREF(w_array);
    Py_DECREF(cs_array);
    Py_DECREF(tau_array);

    return result;
}



/* Python wrapper for mb04ox */
PyObject* py_mb04ox(PyObject* self, PyObject* args) {
    PyObject *a_obj, *x_obj;

    if (!PyArg_ParseTuple(args, "OO", &a_obj, &x_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (x_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 lda = n;
    i32 incx = 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *x_data = (f64*)PyArray_DATA(x_array);

    mb04ox(n, a_data, lda, x_data, incx);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(x_array);

    PyObject *result = Py_BuildValue("(OO)", a_array, x_array);
    Py_DECREF(a_array);
    Py_DECREF(x_array);
    return result;
}


/* Python wrapper for mb04ad */
PyObject* py_mb04ad(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"job", "compq1", "compq2", "compu1", "compu2",
                             "z", "h", "q1", "q2", "u11", "u12", "u21", "u22", NULL};
    const char *job, *compq1, *compq2, *compu1, *compu2;
    PyObject *z_obj, *h_obj;
    PyObject *q1_obj = Py_None, *q2_obj = Py_None;
    PyObject *u11_obj = Py_None, *u12_obj = Py_None;
    PyObject *u21_obj = Py_None, *u22_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssssOO|OOOOOO", kwlist,
                                     &job, &compq1, &compq2, &compu1, &compu2,
                                     &z_obj, &h_obj,
                                     &q1_obj, &q2_obj,
                                     &u11_obj, &u12_obj, &u21_obj, &u22_obj)) {
        return NULL;
    }

    PyArrayObject *z_array = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (z_array == NULL) return NULL;

    PyArrayObject *h_array = (PyArrayObject*)PyArray_FROM_OTF(h_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (h_array == NULL) {
        Py_DECREF(z_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(z_array, 0);
    i32 m = n / 2;
    i32 ldz = n > 1 ? n : 1;
    i32 ldh = n > 1 ? n : 1;
    i32 ldt = n > 1 ? n : 1;

    bool lcmpq1 = (*compq1 == 'I' || *compq1 == 'i' || *compq1 == 'U' || *compq1 == 'u');
    bool lcmpq2 = (*compq2 == 'I' || *compq2 == 'i' || *compq2 == 'U' || *compq2 == 'u');
    bool lcmpu1 = (*compu1 == 'I' || *compu1 == 'i' || *compu1 == 'U' || *compu1 == 'u');
    bool lcmpu2 = (*compu2 == 'I' || *compu2 == 'i' || *compu2 == 'U' || *compu2 == 'u');

    i32 ldq1 = lcmpq1 ? (n > 1 ? n : 1) : 1;
    i32 ldq2 = lcmpq2 ? (n > 1 ? n : 1) : 1;
    i32 ldu11 = lcmpu1 ? (m > 1 ? m : 1) : 1;
    i32 ldu12 = lcmpu1 ? (m > 1 ? m : 1) : 1;
    i32 ldu21 = lcmpu2 ? (m > 1 ? m : 1) : 1;
    i32 ldu22 = lcmpu2 ? (m > 1 ? m : 1) : 1;

    f64 *z_data = (f64*)PyArray_DATA(z_array);
    f64 *h_data = (f64*)PyArray_DATA(h_array);

    npy_intp t_dims[2] = {n, n};
    npy_intp t_strides[2] = {sizeof(f64), n * sizeof(f64)};
    PyObject *t_out = PyArray_New(&PyArray_Type, 2, t_dims, NPY_DOUBLE, t_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (t_out == NULL) {
        Py_DECREF(z_array);
        Py_DECREF(h_array);
        return NULL;
    }
    f64 *t_data = (f64*)PyArray_DATA((PyArrayObject*)t_out);
    if (n > 0) {
        memset(t_data, 0, n * n * sizeof(f64));
    }

    f64 *q1_data = NULL;
    f64 *q2_data = NULL;
    f64 *u11_data = NULL;
    f64 *u12_data = NULL;
    f64 *u21_data = NULL;
    f64 *u22_data = NULL;
    PyArrayObject *q1_array = NULL;
    PyArrayObject *q2_array = NULL;
    PyArrayObject *u11_array = NULL;
    PyArrayObject *u12_array = NULL;
    PyArrayObject *u21_array = NULL;
    PyArrayObject *u22_array = NULL;
    PyObject *q1_out_arr = NULL;
    PyObject *q2_out_arr = NULL;
    PyObject *u11_out_arr = NULL;
    PyObject *u12_out_arr = NULL;
    PyObject *u21_out_arr = NULL;
    PyObject *u22_out_arr = NULL;

    if (lcmpq1) {
        if (q1_obj != Py_None && (*compq1 == 'U' || *compq1 == 'u')) {
            q1_array = (PyArrayObject*)PyArray_FROM_OTF(q1_obj, NPY_DOUBLE,
                                                         NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            if (q1_array == NULL) {
                Py_DECREF(z_array);
                Py_DECREF(h_array);
                Py_DECREF(t_out);
                return NULL;
            }
            q1_data = (f64*)PyArray_DATA(q1_array);
        } else {
            npy_intp q1_dims[2] = {n, n};
            npy_intp q1_strides[2] = {sizeof(f64), n * sizeof(f64)};
            q1_out_arr = PyArray_New(&PyArray_Type, 2, q1_dims, NPY_DOUBLE, q1_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            if (q1_out_arr == NULL) {
                Py_DECREF(z_array);
                Py_DECREF(h_array);
                Py_DECREF(t_out);
                return NULL;
            }
            q1_data = (f64*)PyArray_DATA((PyArrayObject*)q1_out_arr);
            if (n > 0) { memset(q1_data, 0, n * n * sizeof(f64)); }
        }
    } else {
        q1_data = (f64*)calloc(1, sizeof(f64));
    }

    if (lcmpq2) {
        if (q2_obj != Py_None && (*compq2 == 'U' || *compq2 == 'u')) {
            q2_array = (PyArrayObject*)PyArray_FROM_OTF(q2_obj, NPY_DOUBLE,
                                                         NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            if (q2_array == NULL) {
                Py_DECREF(z_array);
                Py_DECREF(h_array);
                Py_DECREF(t_out);
                if (q1_array) { Py_DECREF(q1_array); } else { Py_XDECREF(q1_out_arr); }
                return NULL;
            }
            q2_data = (f64*)PyArray_DATA(q2_array);
        } else {
            npy_intp q2_dims[2] = {n, n};
            npy_intp q2_strides[2] = {sizeof(f64), n * sizeof(f64)};
            q2_out_arr = PyArray_New(&PyArray_Type, 2, q2_dims, NPY_DOUBLE, q2_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            if (q2_out_arr == NULL) {
                Py_DECREF(z_array);
                Py_DECREF(h_array);
                Py_DECREF(t_out);
                if (q1_array) { Py_DECREF(q1_array); } else { Py_XDECREF(q1_out_arr); }
                return NULL;
            }
            q2_data = (f64*)PyArray_DATA((PyArrayObject*)q2_out_arr);
            if (n > 0) { memset(q2_data, 0, n * n * sizeof(f64)); }
        }
    } else {
        q2_data = (f64*)calloc(1, sizeof(f64));
    }

    if (lcmpu1) {
        if (u11_obj != Py_None && (*compu1 == 'U' || *compu1 == 'u')) {
            u11_array = (PyArrayObject*)PyArray_FROM_OTF(u11_obj, NPY_DOUBLE,
                                                          NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            u12_array = (PyArrayObject*)PyArray_FROM_OTF(u12_obj, NPY_DOUBLE,
                                                          NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            if (u11_array == NULL || u12_array == NULL) {
                Py_DECREF(z_array);
                Py_DECREF(h_array);
                Py_DECREF(t_out);
                if (q1_array) { Py_DECREF(q1_array); } else { Py_XDECREF(q1_out_arr); }
                if (q2_array) { Py_DECREF(q2_array); } else { Py_XDECREF(q2_out_arr); }
                Py_XDECREF(u11_array);
                Py_XDECREF(u12_array);
                return NULL;
            }
            u11_data = (f64*)PyArray_DATA(u11_array);
            u12_data = (f64*)PyArray_DATA(u12_array);
        } else {
            npy_intp u_dims[2] = {m, m};
            npy_intp u_strides[2] = {sizeof(f64), m * sizeof(f64)};
            u11_out_arr = PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE, u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            u12_out_arr = PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE, u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            if (u11_out_arr == NULL || u12_out_arr == NULL) {
                Py_DECREF(z_array);
                Py_DECREF(h_array);
                Py_DECREF(t_out);
                if (q1_array) { Py_DECREF(q1_array); } else { Py_XDECREF(q1_out_arr); }
                if (q2_array) { Py_DECREF(q2_array); } else { Py_XDECREF(q2_out_arr); }
                Py_XDECREF(u11_out_arr);
                Py_XDECREF(u12_out_arr);
                return NULL;
            }
            u11_data = (f64*)PyArray_DATA((PyArrayObject*)u11_out_arr);
            u12_data = (f64*)PyArray_DATA((PyArrayObject*)u12_out_arr);
            if (m > 0) {
                memset(u11_data, 0, m * m * sizeof(f64));
                memset(u12_data, 0, m * m * sizeof(f64));
            }
        }
    } else {
        u11_data = (f64*)calloc(1, sizeof(f64));
        u12_data = (f64*)calloc(1, sizeof(f64));
    }

    if (lcmpu2) {
        if (u21_obj != Py_None && (*compu2 == 'U' || *compu2 == 'u')) {
            u21_array = (PyArrayObject*)PyArray_FROM_OTF(u21_obj, NPY_DOUBLE,
                                                          NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            u22_array = (PyArrayObject*)PyArray_FROM_OTF(u22_obj, NPY_DOUBLE,
                                                          NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            if (u21_array == NULL || u22_array == NULL) {
                Py_DECREF(z_array);
                Py_DECREF(h_array);
                Py_DECREF(t_out);
                if (q1_array) { Py_DECREF(q1_array); } else { Py_XDECREF(q1_out_arr); }
                if (q2_array) { Py_DECREF(q2_array); } else { Py_XDECREF(q2_out_arr); }
                if (u11_array) { Py_DECREF(u11_array); } else { Py_XDECREF(u11_out_arr); }
                if (u12_array) { Py_DECREF(u12_array); } else { Py_XDECREF(u12_out_arr); }
                Py_XDECREF(u21_array);
                Py_XDECREF(u22_array);
                return NULL;
            }
            u21_data = (f64*)PyArray_DATA(u21_array);
            u22_data = (f64*)PyArray_DATA(u22_array);
        } else {
            npy_intp u_dims[2] = {m, m};
            npy_intp u_strides[2] = {sizeof(f64), m * sizeof(f64)};
            u21_out_arr = PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE, u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            u22_out_arr = PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE, u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            if (u21_out_arr == NULL || u22_out_arr == NULL) {
                Py_DECREF(z_array);
                Py_DECREF(h_array);
                Py_DECREF(t_out);
                if (q1_array) { Py_DECREF(q1_array); } else { Py_XDECREF(q1_out_arr); }
                if (q2_array) { Py_DECREF(q2_array); } else { Py_XDECREF(q2_out_arr); }
                if (u11_array) { Py_DECREF(u11_array); } else { Py_XDECREF(u11_out_arr); }
                if (u12_array) { Py_DECREF(u12_array); } else { Py_XDECREF(u12_out_arr); }
                Py_XDECREF(u21_out_arr);
                Py_XDECREF(u22_out_arr);
                return NULL;
            }
            u21_data = (f64*)PyArray_DATA((PyArrayObject*)u21_out_arr);
            u22_data = (f64*)PyArray_DATA((PyArrayObject*)u22_out_arr);
            if (m > 0) {
                memset(u21_data, 0, m * m * sizeof(f64));
                memset(u22_data, 0, m * m * sizeof(f64));
            }
        }
    } else {
        u21_data = (f64*)calloc(1, sizeof(f64));
        u22_data = (f64*)calloc(1, sizeof(f64));
    }

    npy_intp eig_dims[1] = {m};
    PyObject *alphar_out = PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);
    PyObject *alphai_out = PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);
    PyObject *beta_out = PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);
    if ((alphar_out == NULL || alphai_out == NULL || beta_out == NULL) && m > 0) {
        Py_DECREF(z_array);
        Py_DECREF(h_array);
        Py_DECREF(t_out);
        if (q1_array) { Py_DECREF(q1_array); } else if (q1_out_arr) { Py_DECREF(q1_out_arr); } else if (q1_data) { free(q1_data); }
        if (q2_array) { Py_DECREF(q2_array); } else if (q2_out_arr) { Py_DECREF(q2_out_arr); } else if (q2_data) { free(q2_data); }
        if (u11_array) { Py_DECREF(u11_array); } else if (u11_out_arr) { Py_DECREF(u11_out_arr); } else if (u11_data) { free(u11_data); }
        if (u12_array) { Py_DECREF(u12_array); } else if (u12_out_arr) { Py_DECREF(u12_out_arr); } else if (u12_data) { free(u12_data); }
        if (u21_array) { Py_DECREF(u21_array); } else if (u21_out_arr) { Py_DECREF(u21_out_arr); } else if (u21_data) { free(u21_data); }
        if (u22_array) { Py_DECREF(u22_array); } else if (u22_out_arr) { Py_DECREF(u22_out_arr); } else if (u22_data) { free(u22_data); }
        Py_XDECREF(alphar_out);
        Py_XDECREF(alphai_out);
        Py_XDECREF(beta_out);
        PyErr_NoMemory();
        return NULL;
    }
    f64 *alphar_data = (f64*)PyArray_DATA((PyArrayObject*)alphar_out);
    f64 *alphai_data = (f64*)PyArray_DATA((PyArrayObject*)alphai_out);
    f64 *beta_data = (f64*)PyArray_DATA((PyArrayObject*)beta_out);
    if (m > 0) {
        memset(alphar_data, 0, m * sizeof(f64));
        memset(alphai_data, 0, m * sizeof(f64));
        memset(beta_data, 0, m * sizeof(f64));
    }

    i32 liwork = n + 18;
    i32 *iwork = (i32*)calloc(liwork, sizeof(i32));
    if (iwork == NULL) {
        Py_DECREF(z_array);
        Py_DECREF(h_array);
        Py_DECREF(t_out);
        if (q1_array) { Py_DECREF(q1_array); } else if (q1_out_arr) { Py_DECREF(q1_out_arr); } else if (q1_data) { free(q1_data); }
        if (q2_array) { Py_DECREF(q2_array); } else if (q2_out_arr) { Py_DECREF(q2_out_arr); } else if (q2_data) { free(q2_data); }
        if (u11_array) { Py_DECREF(u11_array); } else if (u11_out_arr) { Py_DECREF(u11_out_arr); } else if (u11_data) { free(u11_data); }
        if (u12_array) { Py_DECREF(u12_array); } else if (u12_out_arr) { Py_DECREF(u12_out_arr); } else if (u12_data) { free(u12_data); }
        if (u21_array) { Py_DECREF(u21_array); } else if (u21_out_arr) { Py_DECREF(u21_out_arr); } else if (u21_data) { free(u21_data); }
        if (u22_array) { Py_DECREF(u22_array); } else if (u22_out_arr) { Py_DECREF(u22_out_arr); } else if (u22_data) { free(u22_data); }
        Py_DECREF(alphar_out);
        Py_DECREF(alphai_out);
        Py_DECREF(beta_out);
        PyErr_NoMemory();
        return NULL;
    }

    bool ltri = (*job == 'T' || *job == 't');
    i32 ldwork;
    if (n == 0) {
        ldwork = 7;
    } else if (ltri || lcmpq1 || lcmpq2 || lcmpu1 || lcmpu2) {
        ldwork = 12 * m * m + (6 * n > 54 ? 6 * n : 54);
    } else {
        ldwork = 6 * m * m + (6 * n > 54 ? 6 * n : 54);
    }
    ldwork = ldwork > 7 ? ldwork : 7;

    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(z_array);
        Py_DECREF(h_array);
        Py_DECREF(t_out);
        if (q1_array) { Py_DECREF(q1_array); } else if (q1_out_arr) { Py_DECREF(q1_out_arr); } else if (q1_data) { free(q1_data); }
        if (q2_array) { Py_DECREF(q2_array); } else if (q2_out_arr) { Py_DECREF(q2_out_arr); } else if (q2_data) { free(q2_data); }
        if (u11_array) { Py_DECREF(u11_array); } else if (u11_out_arr) { Py_DECREF(u11_out_arr); } else if (u11_data) { free(u11_data); }
        if (u12_array) { Py_DECREF(u12_array); } else if (u12_out_arr) { Py_DECREF(u12_out_arr); } else if (u12_data) { free(u12_data); }
        if (u21_array) { Py_DECREF(u21_array); } else if (u21_out_arr) { Py_DECREF(u21_out_arr); } else if (u21_data) { free(u21_data); }
        if (u22_array) { Py_DECREF(u22_array); } else if (u22_out_arr) { Py_DECREF(u22_out_arr); } else if (u22_data) { free(u22_data); }
        Py_DECREF(alphar_out);
        Py_DECREF(alphai_out);
        Py_DECREF(beta_out);
        free(iwork);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info;
    mb04ad(job, compq1, compq2, compu1, compu2, n,
           z_data, ldz, h_data, ldh,
           q1_data, ldq1, q2_data, ldq2,
           u11_data, ldu11, u12_data, ldu12,
           u21_data, ldu21, u22_data, ldu22,
           t_data, ldt,
           alphar_data, alphai_data, beta_data,
           iwork, liwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(z_array);
    PyArray_ResolveWritebackIfCopy(h_array);
    if (q1_array) PyArray_ResolveWritebackIfCopy(q1_array);
    if (q2_array) PyArray_ResolveWritebackIfCopy(q2_array);
    if (u11_array) PyArray_ResolveWritebackIfCopy(u11_array);
    if (u12_array) PyArray_ResolveWritebackIfCopy(u12_array);
    if (u21_array) PyArray_ResolveWritebackIfCopy(u21_array);
    if (u22_array) PyArray_ResolveWritebackIfCopy(u22_array);

    PyObject *q1_out, *q2_out, *u11_out, *u12_out, *u21_out, *u22_out;

    if (lcmpq1) {
        if (q1_array) {
            q1_out = (PyObject*)q1_array;
            Py_INCREF(q1_out);
        } else {
            q1_out = q1_out_arr;
            Py_INCREF(q1_out);
        }
    } else {
        q1_out = Py_None;
        Py_INCREF(Py_None);
        free(q1_data);
    }

    if (lcmpq2) {
        if (q2_array) {
            q2_out = (PyObject*)q2_array;
            Py_INCREF(q2_out);
        } else {
            q2_out = q2_out_arr;
            Py_INCREF(q2_out);
        }
    } else {
        q2_out = Py_None;
        Py_INCREF(Py_None);
        free(q2_data);
    }

    if (lcmpu1) {
        if (u11_array) {
            u11_out = (PyObject*)u11_array;
            u12_out = (PyObject*)u12_array;
            Py_INCREF(u11_out);
            Py_INCREF(u12_out);
        } else {
            u11_out = u11_out_arr;
            u12_out = u12_out_arr;
            Py_INCREF(u11_out);
            Py_INCREF(u12_out);
        }
    } else {
        u11_out = Py_None;
        u12_out = Py_None;
        Py_INCREF(Py_None);
        Py_INCREF(Py_None);
        free(u11_data);
        free(u12_data);
    }

    if (lcmpu2) {
        if (u21_array) {
            u21_out = (PyObject*)u21_array;
            u22_out = (PyObject*)u22_array;
            Py_INCREF(u21_out);
            Py_INCREF(u22_out);
        } else {
            u21_out = u21_out_arr;
            u22_out = u22_out_arr;
            Py_INCREF(u21_out);
            Py_INCREF(u22_out);
        }
    } else {
        u21_out = Py_None;
        u22_out = Py_None;
        Py_INCREF(Py_None);
        Py_INCREF(Py_None);
        free(u21_data);
        free(u22_data);
    }

    PyObject *result = Py_BuildValue("(OOOOOOOOOOOOi)",
                                     t_out, z_array, h_array,
                                     q1_out, q2_out, u11_out, u12_out, u21_out, u22_out,
                                     alphar_out, alphai_out, beta_out, info);

    Py_DECREF(t_out);
    Py_DECREF(z_array);
    Py_DECREF(h_array);
    Py_DECREF(q1_out);
    Py_DECREF(q2_out);
    Py_DECREF(u11_out);
    Py_DECREF(u12_out);
    Py_DECREF(u21_out);
    Py_DECREF(u22_out);
    Py_DECREF(alphar_out);
    Py_DECREF(alphai_out);
    Py_DECREF(beta_out);

    if (q1_array) Py_DECREF(q1_array);
    if (q2_array) Py_DECREF(q2_array);
    if (u11_array) Py_DECREF(u11_array);
    if (u12_array) Py_DECREF(u12_array);
    if (u21_array) Py_DECREF(u21_array);
    if (u22_array) Py_DECREF(u22_array);

    return result;
}

/* Python wrapper for mb04bd */
PyObject* py_mb04bd(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *job_str, *compq1_str, *compq2_str;
    PyObject *a_obj, *de_obj, *c1_obj, *vw_obj;
    PyObject *q1_obj = NULL;

    static char *kwlist[] = {"job", "compq1", "compq2", "a", "de", "c1", "vw", "q1", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssOOOO|O", kwlist,
                                     &job_str, &compq1_str, &compq2_str,
                                     &a_obj, &de_obj, &c1_obj, &vw_obj, &q1_obj)) {
        return NULL;
    }

    char job = (char)toupper((unsigned char)job_str[0]);
    char compq1 = (char)toupper((unsigned char)compq1_str[0]);
    char compq2 = (char)toupper((unsigned char)compq2_str[0]);

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *de_array = (PyArrayObject*)PyArray_FROM_OTF(de_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (de_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c1_array = (PyArrayObject*)PyArray_FROM_OTF(c1_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c1_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        return NULL;
    }

    PyArrayObject *vw_array = (PyArrayObject*)PyArray_FROM_OTF(vw_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (vw_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(c1_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 m = (i32)a_dims[0];
    i32 n = 2 * m;

    i32 lda = m > 1 ? m : 1;
    i32 ldde = m > 1 ? m : 1;
    i32 ldc1 = m > 1 ? m : 1;
    i32 ldvw = m > 1 ? m : 1;
    i32 ldq1 = n > 1 ? n : 1;
    i32 ldq2 = n > 1 ? n : 1;
    i32 ldb = m > 1 ? m : 1;
    i32 ldf = m > 1 ? m : 1;
    i32 ldc2 = m > 1 ? m : 1;

    bool lcmpq1 = (compq1 == 'I' || compq1 == 'U');
    bool lcmpq2 = (compq2 == 'I' || compq2 == 'U');
    if (lcmpq1 || lcmpq2) {
        ldq1 = n > 1 ? n : 1;
        ldq2 = n > 1 ? n : 1;
    }

    PyArrayObject *q1_array = NULL;
    f64 *q1_data = NULL;
    PyObject *q1_out_arr = NULL;
    PyObject *q2_out_arr = NULL;
    PyObject *b_out = NULL;
    PyObject *f_out = NULL;
    PyObject *c2_out = NULL;

    npy_intp q_dims[2] = {n, n};
    npy_intp q_strides[2] = {sizeof(f64), n * sizeof(f64)};
    npy_intp b_dims[2] = {m > 0 ? m : 0, m > 0 ? m : 0};
    npy_intp b_strides[2] = {sizeof(f64), (m > 0 ? m : 1) * sizeof(f64)};

    if (lcmpq1) {
        if (compq1 == 'U' && q1_obj != NULL && q1_obj != Py_None) {
            q1_array = (PyArrayObject*)PyArray_FROM_OTF(q1_obj, NPY_DOUBLE,
                                                        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            if (q1_array == NULL) {
                Py_DECREF(a_array);
                Py_DECREF(de_array);
                Py_DECREF(c1_array);
                Py_DECREF(vw_array);
                return NULL;
            }
            q1_data = (f64*)PyArray_DATA(q1_array);
        } else {
            q1_out_arr = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            if (q1_out_arr == NULL) {
                Py_DECREF(a_array);
                Py_DECREF(de_array);
                Py_DECREF(c1_array);
                Py_DECREF(vw_array);
                return NULL;
            }
            q1_data = (f64*)PyArray_DATA((PyArrayObject*)q1_out_arr);
            memset(q1_data, 0, (size_t)ldq1 * n * sizeof(f64));
        }
    }

    f64 *q2_data = NULL;
    if (lcmpq2) {
        q2_out_arr = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (q2_out_arr == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(de_array);
            Py_DECREF(c1_array);
            Py_DECREF(vw_array);
            if (q1_array) Py_DECREF(q1_array);
            else Py_XDECREF(q1_out_arr);
            return NULL;
        }
        q2_data = (f64*)PyArray_DATA((PyArrayObject*)q2_out_arr);
        memset(q2_data, 0, (size_t)ldq2 * n * sizeof(f64));
    }

    b_out = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    f_out = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    c2_out = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (b_out == NULL || f_out == NULL || c2_out == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(c1_array);
        Py_DECREF(vw_array);
        if (q1_array) Py_DECREF(q1_array);
        else Py_XDECREF(q1_out_arr);
        Py_XDECREF(q2_out_arr);
        Py_XDECREF(b_out);
        Py_XDECREF(f_out);
        Py_XDECREF(c2_out);
        return NULL;
    }
    f64 *b_data = (f64*)PyArray_DATA((PyArrayObject*)b_out);
    f64 *f_data = (f64*)PyArray_DATA((PyArrayObject*)f_out);
    f64 *c2_data = (f64*)PyArray_DATA((PyArrayObject*)c2_out);
    memset(b_data, 0, (size_t)ldb * m * sizeof(f64));
    memset(f_data, 0, (size_t)ldf * m * sizeof(f64));
    memset(c2_data, 0, (size_t)ldc2 * m * sizeof(f64));

    npy_intp eig_dims[1] = {m > 0 ? m : 0};
    PyObject *alphar_out = PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);
    PyObject *alphai_out = PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);
    PyObject *beta_out = PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);

    if (alphar_out == NULL || alphai_out == NULL || beta_out == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(c1_array);
        Py_DECREF(vw_array);
        if (q1_array) Py_DECREF(q1_array);
        else Py_XDECREF(q1_out_arr);
        Py_XDECREF(q2_out_arr);
        Py_DECREF(b_out);
        Py_DECREF(f_out);
        Py_DECREF(c2_out);
        Py_XDECREF(alphar_out);
        Py_XDECREF(alphai_out);
        Py_XDECREF(beta_out);
        PyErr_NoMemory();
        return NULL;
    }
    f64 *alphar_data = (f64*)PyArray_DATA((PyArrayObject*)alphar_out);
    f64 *alphai_data = (f64*)PyArray_DATA((PyArrayObject*)alphai_out);
    f64 *beta_data = (f64*)PyArray_DATA((PyArrayObject*)beta_out);
    if (m > 0) {
        memset(alphar_data, 0, m * sizeof(f64));
        memset(alphai_data, 0, m * sizeof(f64));
        memset(beta_data, 0, m * sizeof(f64));
    }

    i32 liwork = n + 12;
    i32 *iwork = (i32*)calloc(liwork, sizeof(i32));
    if (iwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(c1_array);
        Py_DECREF(vw_array);
        if (q1_array) Py_DECREF(q1_array);
        else Py_XDECREF(q1_out_arr);
        Py_XDECREF(q2_out_arr);
        Py_DECREF(b_out);
        Py_DECREF(f_out);
        Py_DECREF(c2_out);
        Py_DECREF(alphar_out);
        Py_DECREF(alphai_out);
        Py_DECREF(beta_out);
        PyErr_NoMemory();
        return NULL;
    }

    i32 mm = m * m;
    i32 wsize;
    if ((m % 2) == 0) {
        wsize = (4 * n > 32 ? 4 * n : 32) + 4;
    } else {
        wsize = (4 * n > 36 ? 4 * n : 36);
    }
    i32 ldwork;
    if (job == 'T' || lcmpq1 || lcmpq2) {
        ldwork = 8 * mm + wsize;
    } else {
        ldwork = 4 * mm + wsize;
    }

    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(c1_array);
        Py_DECREF(vw_array);
        if (q1_array) Py_DECREF(q1_array);
        else Py_XDECREF(q1_out_arr);
        Py_XDECREF(q2_out_arr);
        Py_DECREF(b_out);
        Py_DECREF(f_out);
        Py_DECREF(c2_out);
        Py_DECREF(alphar_out);
        Py_DECREF(alphai_out);
        Py_DECREF(beta_out);
        free(iwork);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *de_data = (f64*)PyArray_DATA(de_array);
    f64 *c1_data = (f64*)PyArray_DATA(c1_array);
    f64 *vw_data = (f64*)PyArray_DATA(vw_array);

    i32 info;
    mb04bd(job_str, compq1_str, compq2_str, n,
           a_data, lda, de_data, ldde, c1_data, ldc1, vw_data, ldvw,
           q1_data, ldq1, q2_data, ldq2,
           b_data, ldb, f_data, ldf, c2_data, ldc2,
           alphar_data, alphai_data, beta_data,
           iwork, liwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(de_array);
    PyArray_ResolveWritebackIfCopy(c1_array);
    PyArray_ResolveWritebackIfCopy(vw_array);

    PyObject *q1_out, *q2_out;
    if (lcmpq1) {
        if (q1_array) {
            q1_out = (PyObject*)q1_array;
            Py_INCREF(q1_out);
        } else {
            q1_out = q1_out_arr;
            Py_INCREF(q1_out);
        }
    } else {
        q1_out = Py_None;
        Py_INCREF(Py_None);
    }

    if (lcmpq2) {
        q2_out = q2_out_arr;
        Py_INCREF(q2_out);
    } else {
        q2_out = Py_None;
        Py_INCREF(Py_None);
    }

    PyObject *result = Py_BuildValue("(OOOOOOOOOOOOi)",
                                     a_array, de_array, c1_array, vw_array,
                                     q1_out, q2_out, b_out, f_out, c2_out,
                                     alphar_out, alphai_out, beta_out, info);

    Py_DECREF(a_array);
    Py_DECREF(de_array);
    Py_DECREF(c1_array);
    Py_DECREF(vw_array);
    Py_DECREF(q1_out);
    Py_DECREF(q2_out);
    Py_DECREF(b_out);
    Py_DECREF(f_out);
    Py_DECREF(c2_out);
    Py_DECREF(alphar_out);
    Py_DECREF(alphai_out);
    Py_DECREF(beta_out);

    return result;
}

PyObject* py_mb04bp(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *job_str, *compq1_str, *compq2_str;
    PyObject *a_obj, *de_obj, *c1_obj, *vw_obj;
    PyObject *q1_obj = NULL;

    static char *kwlist[] = {"job", "compq1", "compq2", "a", "de", "c1", "vw", "q1", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssOOOO|O", kwlist,
                                     &job_str, &compq1_str, &compq2_str,
                                     &a_obj, &de_obj, &c1_obj, &vw_obj, &q1_obj)) {
        return NULL;
    }

    char job = (char)toupper((unsigned char)job_str[0]);
    char compq1 = (char)toupper((unsigned char)compq1_str[0]);
    char compq2 = (char)toupper((unsigned char)compq2_str[0]);

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *de_array = (PyArrayObject*)PyArray_FROM_OTF(de_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (de_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c1_array = (PyArrayObject*)PyArray_FROM_OTF(c1_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c1_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        return NULL;
    }

    PyArrayObject *vw_array = (PyArrayObject*)PyArray_FROM_OTF(vw_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (vw_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(c1_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 m = (i32)a_dims[0];
    i32 n = 2 * m;

    i32 lda = m > 1 ? m : 1;
    i32 ldde = m > 1 ? m : 1;
    i32 ldc1 = m > 1 ? m : 1;
    i32 ldvw = m > 1 ? m : 1;
    i32 ldq1 = n > 1 ? n : 1;
    i32 ldq2 = n > 1 ? n : 1;
    i32 ldb = m > 1 ? m : 1;
    i32 ldf = m > 1 ? m : 1;
    i32 ldc2 = m > 1 ? m : 1;

    bool lcmpq1 = (compq1 == 'I' || compq1 == 'U');
    bool lcmpq2 = (compq2 == 'I' || compq2 == 'U');
    if (lcmpq1 || lcmpq2) {
        ldq1 = n > 1 ? n : 1;
        ldq2 = n > 1 ? n : 1;
    }

    PyArrayObject *q1_array = NULL;
    f64 *q1_data = NULL;

    if (lcmpq1) {
        if (compq1 == 'U' && q1_obj != NULL && q1_obj != Py_None) {
            q1_array = (PyArrayObject*)PyArray_FROM_OTF(q1_obj, NPY_DOUBLE,
                                                        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            if (q1_array == NULL) {
                Py_DECREF(a_array);
                Py_DECREF(de_array);
                Py_DECREF(c1_array);
                Py_DECREF(vw_array);
                return NULL;
            }
            q1_data = (f64*)PyArray_DATA(q1_array);
        } else {
            q1_data = (f64*)calloc((size_t)ldq1 * n, sizeof(f64));
            if (q1_data == NULL) {
                Py_DECREF(a_array);
                Py_DECREF(de_array);
                Py_DECREF(c1_array);
                Py_DECREF(vw_array);
                PyErr_NoMemory();
                return NULL;
            }
        }
    }

    f64 *q2_data = NULL;
    if (lcmpq2) {
        q2_data = (f64*)calloc((size_t)ldq2 * n, sizeof(f64));
        if (q2_data == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(de_array);
            Py_DECREF(c1_array);
            Py_DECREF(vw_array);
            if (q1_array) Py_DECREF(q1_array);
            else if (q1_data) free(q1_data);
            PyErr_NoMemory();
            return NULL;
        }
    }

    npy_intp b_dims[2] = {m > 0 ? m : 0, m > 0 ? m : 0};
    npy_intp b_strides[2] = {sizeof(f64), (m > 0 ? m : 1) * sizeof(f64)};
    npy_intp eig_dims[1] = {m > 0 ? m : 0};

    PyObject *b_out = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *f_out = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *c2_out = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *alphar_out = PyArray_New(&PyArray_Type, 1, eig_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
    PyObject *alphai_out = PyArray_New(&PyArray_Type, 1, eig_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
    PyObject *beta_out = PyArray_New(&PyArray_Type, 1, eig_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);

    if (b_out == NULL || f_out == NULL || c2_out == NULL ||
        alphar_out == NULL || alphai_out == NULL || beta_out == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(c1_array);
        Py_DECREF(vw_array);
        if (q1_array) Py_DECREF(q1_array);
        else if (q1_data) free(q1_data);
        if (q2_data) free(q2_data);
        Py_XDECREF(b_out);
        Py_XDECREF(f_out);
        Py_XDECREF(c2_out);
        Py_XDECREF(alphar_out);
        Py_XDECREF(alphai_out);
        Py_XDECREF(beta_out);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *b_data = (f64*)PyArray_DATA((PyArrayObject*)b_out);
    f64 *f_data = (f64*)PyArray_DATA((PyArrayObject*)f_out);
    f64 *c2_data = (f64*)PyArray_DATA((PyArrayObject*)c2_out);
    f64 *alphar_data = (f64*)PyArray_DATA((PyArrayObject*)alphar_out);
    f64 *alphai_data = (f64*)PyArray_DATA((PyArrayObject*)alphai_out);
    f64 *beta_data = (f64*)PyArray_DATA((PyArrayObject*)beta_out);

    if (m > 0) {
        size_t b_size = (size_t)ldb * m;
        memset(b_data, 0, b_size * sizeof(f64));
        memset(f_data, 0, b_size * sizeof(f64));
        memset(c2_data, 0, b_size * sizeof(f64));
    }
    if (m > 0) {
        memset(alphar_data, 0, m * sizeof(f64));
        memset(alphai_data, 0, m * sizeof(f64));
        memset(beta_data, 0, m * sizeof(f64));
    }

    i32 liwork = n + 12;
    i32 *iwork = (i32*)calloc(liwork, sizeof(i32));
    if (iwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(c1_array);
        Py_DECREF(vw_array);
        if (q1_array) Py_DECREF(q1_array);
        else if (q1_data) free(q1_data);
        if (q2_data) free(q2_data);
        Py_DECREF(b_out);
        Py_DECREF(f_out);
        Py_DECREF(c2_out);
        Py_DECREF(alphar_out);
        Py_DECREF(alphai_out);
        Py_DECREF(beta_out);
        PyErr_NoMemory();
        return NULL;
    }

    i32 mm = m * m;
    i32 wsize;
    if ((m % 2) == 0) {
        wsize = (4 * n > 32 ? 4 * n : 32) + 4;
    } else {
        wsize = (4 * n > 36 ? 4 * n : 36);
    }
    i32 ldwork;
    if (job == 'T' || lcmpq1 || lcmpq2) {
        ldwork = 8 * mm + wsize;
    } else {
        ldwork = 4 * mm + wsize;
    }

    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(c1_array);
        Py_DECREF(vw_array);
        if (q1_array) Py_DECREF(q1_array);
        else if (q1_data) free(q1_data);
        if (q2_data) free(q2_data);
        Py_DECREF(b_out);
        Py_DECREF(f_out);
        Py_DECREF(c2_out);
        Py_DECREF(alphar_out);
        Py_DECREF(alphai_out);
        Py_DECREF(beta_out);
        free(iwork);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *de_data = (f64*)PyArray_DATA(de_array);
    f64 *c1_data = (f64*)PyArray_DATA(c1_array);
    f64 *vw_data = (f64*)PyArray_DATA(vw_array);

    i32 info = 0;
    mb04bp(job_str, compq1_str, compq2_str, n,
           a_data, lda, de_data, ldde, c1_data, ldc1, vw_data, ldvw,
           q1_data, ldq1, q2_data, ldq2,
           b_data, ldb, f_data, ldf, c2_data, ldc2,
           alphar_data, alphai_data, beta_data,
           iwork, liwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(de_array);
    PyArray_ResolveWritebackIfCopy(c1_array);
    PyArray_ResolveWritebackIfCopy(vw_array);

    npy_intp q_dims[2] = {n, n};
    npy_intp q_strides[2] = {sizeof(f64), n * sizeof(f64)};

    PyObject *q1_out_result, *q2_out_result;
    if (lcmpq1) {
        if (q1_array) {
            q1_out_result = (PyObject*)q1_array;
            Py_INCREF(q1_out_result);
        } else {
            q1_out_result = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            f64 *q1_out_data = (f64*)PyArray_DATA((PyArrayObject*)q1_out_result);
            if ((size_t)ldq1 * n > 0) memcpy(q1_out_data, q1_data, (size_t)ldq1 * n * sizeof(f64));
            free(q1_data);
        }
    } else {
        q1_out_result = Py_None;
        Py_INCREF(Py_None);
        if (q1_data && !q1_array) free(q1_data);
    }

    if (lcmpq2) {
        q2_out_result = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        f64 *q2_out_data = (f64*)PyArray_DATA((PyArrayObject*)q2_out_result);
        if ((size_t)ldq2 * n > 0) memcpy(q2_out_data, q2_data, (size_t)ldq2 * n * sizeof(f64));
        free(q2_data);
    } else {
        q2_out_result = Py_None;
        Py_INCREF(Py_None);
        if (q2_data) free(q2_data);
    }

    PyObject *result = Py_BuildValue("(OOOOOOOOOOOOi)",
                                     a_array, de_array, c1_array, vw_array,
                                     q1_out_result, q2_out_result, b_out, f_out, c2_out,
                                     alphar_out, alphai_out, beta_out, info);

    Py_DECREF(a_array);
    Py_DECREF(de_array);
    Py_DECREF(c1_array);
    Py_DECREF(vw_array);
    Py_DECREF(q1_out_result);
    Py_DECREF(q2_out_result);
    Py_DECREF(b_out);
    Py_DECREF(f_out);
    Py_DECREF(c2_out);
    Py_DECREF(alphar_out);
    Py_DECREF(alphai_out);
    Py_DECREF(beta_out);

    return result;
}

PyObject* py_mb04cd(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *compq1_str, *compq2_str, *compq3_str;
    PyObject *a_obj, *b_obj, *d_obj;
    PyObject *q1_obj = NULL, *q2_obj = NULL, *q3_obj = NULL;

    static char *kwlist[] = {"compq1", "compq2", "compq3", "a", "b", "d",
                             "q1", "q2", "q3", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssOOO|OOO", kwlist,
                                     &compq1_str, &compq2_str, &compq3_str,
                                     &a_obj, &b_obj, &d_obj,
                                     &q1_obj, &q2_obj, &q3_obj)) {
        return NULL;
    }

    char compq1 = (char)toupper((unsigned char)compq1_str[0]);
    char compq2 = (char)toupper((unsigned char)compq2_str[0]);
    char compq3 = (char)toupper((unsigned char)compq3_str[0]);

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 n = (i32)a_dims[0];

    i32 lda = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldd = n > 1 ? n : 1;
    i32 ldq1 = n > 1 ? n : 1;
    i32 ldq2 = n > 1 ? n : 1;
    i32 ldq3 = n > 1 ? n : 1;

    bool lcmpq1 = (compq1 == 'I' || compq1 == 'U');
    bool lcmpq2 = (compq2 == 'I' || compq2 == 'U');
    bool lcmpq3 = (compq3 == 'I' || compq3 == 'U');

    if (lcmpq1) ldq1 = n > 1 ? n : 1;
    if (lcmpq2) ldq2 = n > 1 ? n : 1;
    if (lcmpq3) ldq3 = n > 1 ? n : 1;

    PyArrayObject *q1_array = NULL, *q2_array = NULL, *q3_array = NULL;
    f64 *q1_data = NULL, *q2_data = NULL, *q3_data = NULL;
    PyObject *q1_out_arr = NULL, *q2_out_arr = NULL, *q3_out_arr = NULL;

    npy_intp q_dims[2] = {n, n};
    npy_intp q_strides[2] = {sizeof(f64), n * sizeof(f64)};

    if (lcmpq1) {
        if (compq1 == 'U' && q1_obj != NULL && q1_obj != Py_None) {
            q1_array = (PyArrayObject*)PyArray_FROM_OTF(q1_obj, NPY_DOUBLE,
                                                        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            if (q1_array == NULL) {
                Py_DECREF(a_array);
                Py_DECREF(b_array);
                Py_DECREF(d_array);
                return NULL;
            }
            q1_data = (f64*)PyArray_DATA(q1_array);
        } else {
            q1_out_arr = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            if (q1_out_arr == NULL) {
                Py_DECREF(a_array);
                Py_DECREF(b_array);
                Py_DECREF(d_array);
                return NULL;
            }
            q1_data = (f64*)PyArray_DATA((PyArrayObject*)q1_out_arr);
            memset(q1_data, 0, (size_t)ldq1 * n * sizeof(f64));
        }
    }

    if (lcmpq2) {
        if (compq2 == 'U' && q2_obj != NULL && q2_obj != Py_None) {
            q2_array = (PyArrayObject*)PyArray_FROM_OTF(q2_obj, NPY_DOUBLE,
                                                        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            if (q2_array == NULL) {
                Py_DECREF(a_array);
                Py_DECREF(b_array);
                Py_DECREF(d_array);
                if (q1_array) Py_DECREF(q1_array);
                else Py_XDECREF(q1_out_arr);
                return NULL;
            }
            q2_data = (f64*)PyArray_DATA(q2_array);
        } else {
            q2_out_arr = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            if (q2_out_arr == NULL) {
                Py_DECREF(a_array);
                Py_DECREF(b_array);
                Py_DECREF(d_array);
                if (q1_array) Py_DECREF(q1_array);
                else Py_XDECREF(q1_out_arr);
                return NULL;
            }
            q2_data = (f64*)PyArray_DATA((PyArrayObject*)q2_out_arr);
            memset(q2_data, 0, (size_t)ldq2 * n * sizeof(f64));
        }
    }

    if (lcmpq3) {
        if (compq3 == 'U' && q3_obj != NULL && q3_obj != Py_None) {
            q3_array = (PyArrayObject*)PyArray_FROM_OTF(q3_obj, NPY_DOUBLE,
                                                        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            if (q3_array == NULL) {
                Py_DECREF(a_array);
                Py_DECREF(b_array);
                Py_DECREF(d_array);
                if (q1_array) Py_DECREF(q1_array);
                else Py_XDECREF(q1_out_arr);
                if (q2_array) Py_DECREF(q2_array);
                else Py_XDECREF(q2_out_arr);
                return NULL;
            }
            q3_data = (f64*)PyArray_DATA(q3_array);
        } else {
            q3_out_arr = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            if (q3_out_arr == NULL) {
                Py_DECREF(a_array);
                Py_DECREF(b_array);
                Py_DECREF(d_array);
                if (q1_array) Py_DECREF(q1_array);
                else Py_XDECREF(q1_out_arr);
                if (q2_array) Py_DECREF(q2_array);
                else Py_XDECREF(q2_out_arr);
                return NULL;
            }
            q3_data = (f64*)PyArray_DATA((PyArrayObject*)q3_out_arr);
            memset(q3_data, 0, (size_t)ldq3 * n * sizeof(f64));
        }
    }

    i32 m = n / 2;
    i32 liwork = (m / 2 + 1 > 48) ? m / 2 + 1 : 48;
    i32 *iwork = (i32*)calloc(liwork, sizeof(i32));
    if (iwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(d_array);
        if (q1_array) Py_DECREF(q1_array);
        else Py_XDECREF(q1_out_arr);
        if (q2_array) Py_DECREF(q2_array);
        else Py_XDECREF(q2_out_arr);
        if (q3_array) Py_DECREF(q3_array);
        else Py_XDECREF(q3_out_arr);
        PyErr_NoMemory();
        return NULL;
    }

    i32 nn = n * n;
    i32 wsize = (m / 2 + 252 > 432) ? m / 2 + 252 : 432;
    i32 ldwork = 3 * nn + wsize;

    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(d_array);
        if (q1_array) Py_DECREF(q1_array);
        else Py_XDECREF(q1_out_arr);
        if (q2_array) Py_DECREF(q2_array);
        else Py_XDECREF(q2_out_arr);
        if (q3_array) Py_DECREF(q3_array);
        else Py_XDECREF(q3_out_arr);
        free(iwork);
        PyErr_NoMemory();
        return NULL;
    }

    bool *bwork = (bool*)calloc(m > 0 ? m : 1, sizeof(bool));
    if (bwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(d_array);
        if (q1_array) Py_DECREF(q1_array);
        else Py_XDECREF(q1_out_arr);
        if (q2_array) Py_DECREF(q2_array);
        else Py_XDECREF(q2_out_arr);
        if (q3_array) Py_DECREF(q3_array);
        else Py_XDECREF(q3_out_arr);
        free(iwork);
        free(dwork);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);

    i32 info;
    mb04cd(compq1_str, compq2_str, compq3_str, n,
           a_data, lda, b_data, ldb, d_data, ldd,
           q1_data, ldq1, q2_data, ldq2, q3_data, ldq3,
           iwork, liwork, dwork, ldwork, bwork, &info);

    free(iwork);
    free(dwork);
    free(bwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    PyObject *q1_out, *q2_out, *q3_out;

    if (lcmpq1) {
        if (q1_array) {
            PyArray_ResolveWritebackIfCopy(q1_array);
            q1_out = (PyObject*)q1_array;
            Py_INCREF(q1_out);
        } else {
            q1_out = q1_out_arr;
            Py_INCREF(q1_out);
        }
    } else {
        q1_out = Py_None;
        Py_INCREF(Py_None);
    }

    if (lcmpq2) {
        if (q2_array) {
            PyArray_ResolveWritebackIfCopy(q2_array);
            q2_out = (PyObject*)q2_array;
            Py_INCREF(q2_out);
        } else {
            q2_out = q2_out_arr;
            Py_INCREF(q2_out);
        }
    } else {
        q2_out = Py_None;
        Py_INCREF(Py_None);
    }

    if (lcmpq3) {
        if (q3_array) {
            PyArray_ResolveWritebackIfCopy(q3_array);
            q3_out = (PyObject*)q3_array;
            Py_INCREF(q3_out);
        } else {
            q3_out = q3_out_arr;
            Py_INCREF(q3_out);
        }
    } else {
        q3_out = Py_None;
        Py_INCREF(Py_None);
    }

    PyObject *result = Py_BuildValue("(OOOOOOi)",
                                     a_array, b_array, d_array,
                                     q1_out, q2_out, q3_out, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(d_array);
    Py_DECREF(q1_out);
    Py_DECREF(q2_out);
    Py_DECREF(q3_out);

    return result;
}

PyObject* py_mb04db(PyObject* self, PyObject* args) {
    const char *job, *sgn;
    i32 ilo;
    PyObject *lscale_obj, *rscale_obj, *v1_obj, *v2_obj;
    PyArrayObject *lscale_array = NULL, *rscale_array = NULL;
    PyArrayObject *v1_array = NULL, *v2_array = NULL;
    i32 info = 0;

    if (!PyArg_ParseTuple(args, "ssiOOOO", &job, &sgn, &ilo,
                          &lscale_obj, &rscale_obj, &v1_obj, &v2_obj)) {
        return NULL;
    }

    lscale_array = (PyArrayObject*)PyArray_FROM_OTF(lscale_obj, NPY_DOUBLE,
                                                    NPY_ARRAY_IN_FARRAY);
    rscale_array = (PyArrayObject*)PyArray_FROM_OTF(rscale_obj, NPY_DOUBLE,
                                                    NPY_ARRAY_IN_FARRAY);
    v1_array = (PyArrayObject*)PyArray_FROM_OTF(v1_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    v2_array = (PyArrayObject*)PyArray_FROM_OTF(v2_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (lscale_array == NULL || rscale_array == NULL ||
        v1_array == NULL || v2_array == NULL) {
        Py_XDECREF(lscale_array);
        Py_XDECREF(rscale_array);
        Py_XDECREF(v1_array);
        Py_XDECREF(v2_array);
        return NULL;
    }

    npy_intp *v1_dims = PyArray_DIMS(v1_array);
    i32 n = (i32)v1_dims[0];
    i32 m = PyArray_NDIM(v1_array) > 1 ? (i32)v1_dims[1] : 1;
    i32 ldv1 = n > 0 ? n : 1;
    i32 ldv2 = n > 0 ? n : 1;

    f64 *lscale_data = (f64*)PyArray_DATA(lscale_array);
    f64 *rscale_data = (f64*)PyArray_DATA(rscale_array);
    f64 *v1_data = (f64*)PyArray_DATA(v1_array);
    f64 *v2_data = (f64*)PyArray_DATA(v2_array);

    mb04db(job, sgn, n, ilo, lscale_data, rscale_data, m,
           v1_data, ldv1, v2_data, ldv2, &info);

    PyArray_ResolveWritebackIfCopy(v1_array);
    PyArray_ResolveWritebackIfCopy(v2_array);

    PyObject *result = Py_BuildValue("(OOi)", v1_array, v2_array, info);

    Py_DECREF(lscale_array);
    Py_DECREF(rscale_array);
    Py_DECREF(v1_array);
    Py_DECREF(v2_array);

    return result;
}

PyObject* py_mb04di(PyObject* self, PyObject* args) {
    const char *job, *sgn;
    i32 ilo;
    PyObject *scale_obj, *v1_obj, *v2_obj;
    PyArrayObject *scale_array = NULL;
    PyArrayObject *v1_array = NULL, *v2_array = NULL;
    i32 info = 0;

    if (!PyArg_ParseTuple(args, "ssiOOO", &job, &sgn, &ilo,
                          &scale_obj, &v1_obj, &v2_obj)) {
        return NULL;
    }

    scale_array = (PyArrayObject*)PyArray_FROM_OTF(scale_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_IN_FARRAY);
    v1_array = (PyArrayObject*)PyArray_FROM_OTF(v1_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    v2_array = (PyArrayObject*)PyArray_FROM_OTF(v2_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (scale_array == NULL || v1_array == NULL || v2_array == NULL) {
        Py_XDECREF(scale_array);
        Py_XDECREF(v1_array);
        Py_XDECREF(v2_array);
        return NULL;
    }

    npy_intp *v1_dims = PyArray_DIMS(v1_array);
    i32 n = (i32)v1_dims[0];
    i32 m = PyArray_NDIM(v1_array) > 1 ? (i32)v1_dims[1] : 1;
    i32 ldv1 = n > 0 ? n : 1;
    i32 ldv2 = n > 0 ? n : 1;

    f64 *scale_data = (f64*)PyArray_DATA(scale_array);
    f64 *v1_data = (f64*)PyArray_DATA(v1_array);
    f64 *v2_data = (f64*)PyArray_DATA(v2_array);

    mb04di(job, sgn, n, ilo, scale_data, m, v1_data, ldv1, v2_data, ldv2, &info);

    PyArray_ResolveWritebackIfCopy(v1_array);
    PyArray_ResolveWritebackIfCopy(v2_array);

    PyObject *result = Py_BuildValue("(OOi)", v1_array, v2_array, info);

    Py_DECREF(scale_array);
    Py_DECREF(v1_array);
    Py_DECREF(v2_array);

    return result;
}

PyObject* py_mb04dl(PyObject* self, PyObject* args) {
    const char *job;
    i32 n;
    f64 thresh;
    PyObject *a_obj, *b_obj;
    PyArrayObject *a_array = NULL, *b_array = NULL;
    i32 ilo, ihi, iwarn, info;

    if (!PyArg_ParseTuple(args, "sidOO", &job, &n, &thresh, &a_obj, &b_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || b_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    npy_intp n_dim = n > 0 ? n : 0;
    PyObject *lscale_obj = PyArray_SimpleNew(1, &n_dim, NPY_DOUBLE);
    PyObject *rscale_obj = PyArray_SimpleNew(1, &n_dim, NPY_DOUBLE);
    if (lscale_obj == NULL || rscale_obj == NULL) {
        Py_XDECREF(lscale_obj);
        Py_XDECREF(rscale_obj);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }
    f64 *lscale_data = (f64*)PyArray_DATA((PyArrayObject*)lscale_obj);
    f64 *rscale_data = (f64*)PyArray_DATA((PyArrayObject*)rscale_obj);

    i32 ldwork = (n > 0) ? 8 * n + 2 : 1;
    npy_intp dwork_dim = ldwork;
    PyObject *dwork_obj = PyArray_SimpleNew(1, &dwork_dim, NPY_DOUBLE);
    if (dwork_obj == NULL) {
        Py_DECREF(lscale_obj);
        Py_DECREF(rscale_obj);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }
    f64 *dwork_data = (f64*)PyArray_DATA((PyArrayObject*)dwork_obj);

    mb04dl(job, n, thresh, a_data, lda, b_data, ldb,
           &ilo, &ihi, lscale_data, rscale_data, dwork_data, &iwarn, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    PyObject *result = Py_BuildValue("(OOiiOOOii)",
                                     a_array, b_array, ilo, ihi,
                                     lscale_obj, rscale_obj, dwork_obj, iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(lscale_obj);
    Py_DECREF(rscale_obj);
    Py_DECREF(dwork_obj);

    return result;
}

PyObject* py_mb04dp(PyObject* self, PyObject* args) {
    const char *job;
    i32 n;
    f64 thresh;
    PyObject *a_obj, *de_obj, *c_obj, *vw_obj;
    PyArrayObject *a_array = NULL, *de_array = NULL, *c_array = NULL, *vw_array = NULL;
    i32 ilo, iwarn, info;

    if (!PyArg_ParseTuple(args, "sidOOOO", &job, &n, &thresh, &a_obj, &de_obj, &c_obj, &vw_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    de_array = (PyArrayObject*)PyArray_FROM_OTF(de_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    vw_array = (PyArrayObject*)PyArray_FROM_OTF(vw_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || de_array == NULL || c_array == NULL || vw_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(de_array);
        Py_XDECREF(c_array);
        Py_XDECREF(vw_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldde = n > 0 ? n : 1;
    i32 ldc = n > 0 ? n : 1;
    i32 ldvw = n > 0 ? n : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *de_data = (f64*)PyArray_DATA(de_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *vw_data = (f64*)PyArray_DATA(vw_array);

    npy_intp n_dim = n > 0 ? n : 0;
    PyObject *lscale_obj = PyArray_SimpleNew(1, &n_dim, NPY_DOUBLE);
    PyObject *rscale_obj = PyArray_SimpleNew(1, &n_dim, NPY_DOUBLE);
    if (lscale_obj == NULL || rscale_obj == NULL) {
        Py_XDECREF(lscale_obj);
        Py_XDECREF(rscale_obj);
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(c_array);
        Py_DECREF(vw_array);
        return NULL;
    }
    f64 *lscale_data = (f64*)PyArray_DATA((PyArrayObject*)lscale_obj);
    f64 *rscale_data = (f64*)PyArray_DATA((PyArrayObject*)rscale_obj);

    i32 ldwork = (n > 0) ? 8 * n + 2 : 1;
    npy_intp dwork_dim = ldwork;
    PyObject *dwork_obj = PyArray_SimpleNew(1, &dwork_dim, NPY_DOUBLE);
    if (dwork_obj == NULL) {
        Py_DECREF(lscale_obj);
        Py_DECREF(rscale_obj);
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(c_array);
        Py_DECREF(vw_array);
        return NULL;
    }
    f64 *dwork_data = (f64*)PyArray_DATA((PyArrayObject*)dwork_obj);

    mb04dp(job, n, thresh, a_data, lda, de_data, ldde, c_data, ldc, vw_data, ldvw,
           &ilo, lscale_data, rscale_data, dwork_data, &iwarn, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(de_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(vw_array);

    PyObject *result = Py_BuildValue("(OOOOiOOOii)",
                                     a_array, de_array, c_array, vw_array, ilo,
                                     lscale_obj, rscale_obj, dwork_obj, iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(de_array);
    Py_DECREF(c_array);
    Py_DECREF(vw_array);
    Py_DECREF(lscale_obj);
    Py_DECREF(rscale_obj);
    Py_DECREF(dwork_obj);

    return result;
}


PyObject* py_mb04ds(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static char *kwlist[] = {"job", "n", "a", "qg", NULL};

    const char *job = NULL;
    i32 n = 0;
    PyObject *a_obj = NULL;
    PyObject *qg_obj = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siOO", kwlist,
                                     &job, &n, &a_obj, &qg_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = NULL;
    PyArrayObject *qg_array = NULL;
    i32 ilo = 0;
    i32 info = 0;

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    qg_array = (PyArrayObject*)PyArray_FROM_OTF(qg_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || qg_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(qg_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldqg = n > 0 ? n : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *qg_data = (f64*)PyArray_DATA(qg_array);

    npy_intp n_dim = n > 0 ? n : 0;
    PyObject *scale_obj = PyArray_SimpleNew(1, &n_dim, NPY_DOUBLE);
    if (scale_obj == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(qg_array);
        return NULL;
    }
    f64 *scale_data = (f64*)PyArray_DATA((PyArrayObject*)scale_obj);

    mb04ds(job, n, a_data, lda, qg_data, ldqg, &ilo, scale_data, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(qg_array);

    PyObject *result = Py_BuildValue("(OOiOi)",
                                     a_array, qg_array, ilo, scale_obj, info);

    Py_DECREF(a_array);
    Py_DECREF(qg_array);
    Py_DECREF(scale_obj);

    return result;
}


PyObject* py_mb04dy(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static char *kwlist[] = {"jobscl", "a", "qg", NULL};

    const char *jobscl = NULL;
    PyObject *a_obj = NULL;
    PyObject *qg_obj = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sOO", kwlist,
                                     &jobscl, &a_obj, &qg_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = NULL;
    PyArrayObject *qg_array = NULL;
    i32 info = 0;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    qg_array = (PyArrayObject*)PyArray_FROM_OTF(qg_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || qg_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(qg_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 lda = n > 0 ? n : 1;
    i32 ldqg = n > 0 ? n : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *qg_data = (f64*)PyArray_DATA(qg_array);

    char job_upper = (char)toupper((unsigned char)jobscl[0]);
    bool is_norm = (job_upper == '1') || (job_upper == 'O');
    npy_intp d_dim = is_norm ? 1 : (n > 0 ? n : 0);

    PyObject *d_obj = PyArray_SimpleNew(1, &d_dim, NPY_DOUBLE);
    if (d_obj == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(qg_array);
        return NULL;
    }
    f64 *d_data = (f64*)PyArray_DATA((PyArrayObject*)d_obj);

    f64 *dwork = NULL;
    if (n > 0 && job_upper != 'N') {
        dwork = (f64*)malloc(n * sizeof(f64));
        if (dwork == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(qg_array);
            Py_DECREF(d_obj);
            PyErr_NoMemory();
            return NULL;
        }
    }

    mb04dy(jobscl, n, a_data, lda, qg_data, ldqg, d_data, dwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(qg_array);

    PyObject *result = Py_BuildValue("(OOOi)",
                                     a_array, qg_array, d_obj, info);

    Py_DECREF(a_array);
    Py_DECREF(qg_array);
    Py_DECREF(d_obj);

    return result;
}

PyObject* py_mb04dz(PyObject* self, PyObject* args) {
    const char *job;
    PyObject *a_obj, *qg_obj;
    PyArrayObject *a_array = NULL, *qg_array = NULL;
    i32 info = 0;

    if (!PyArg_ParseTuple(args, "sOO", &job, &a_obj, &qg_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_CDOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    qg_array = (PyArrayObject*)PyArray_FROM_OTF(qg_obj, NPY_CDOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || qg_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(qg_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 n = (i32)a_dims[0];
    i32 lda = n > 0 ? n : 1;
    i32 ldqg = n > 0 ? n : 1;

    npy_intp scale_dims[1] = {n > 0 ? n : 1};
    PyObject *scale_array = PyArray_New(&PyArray_Type, 1, scale_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
    if (scale_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(qg_array);
        PyErr_NoMemory();
        return NULL;
    }
    f64 *scale = (f64*)PyArray_DATA((PyArrayObject*)scale_array);

    i32 ilo;
    c128 *a_data = (c128*)PyArray_DATA(a_array);
    c128 *qg_data = (c128*)PyArray_DATA(qg_array);

    mb04dz(job, n, a_data, lda, qg_data, ldqg, &ilo, scale, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(qg_array);

    PyObject *result = Py_BuildValue("(OOiOi)", a_array, qg_array, ilo, scale_array, info);

    Py_DECREF(a_array);
    Py_DECREF(qg_array);
    Py_DECREF(scale_array);

    return result;
}

PyObject* py_mb04ed(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static char *kwlist[] = {"job", "compq", "compu", "z", "b", "fg", "u1", "u2", NULL};

    const char *job = NULL;
    const char *compq = NULL;
    const char *compu = NULL;
    PyObject *z_obj = NULL;
    PyObject *b_obj = NULL;
    PyObject *fg_obj = NULL;
    PyObject *u1_obj = Py_None;
    PyObject *u2_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssOOO|OO", kwlist,
                                     &job, &compq, &compu, &z_obj, &b_obj, &fg_obj,
                                     &u1_obj, &u2_obj)) {
        return NULL;
    }

    PyArrayObject *z_array = NULL;
    PyArrayObject *b_array = NULL;
    PyArrayObject *fg_array = NULL;
    PyArrayObject *u1_array = NULL;
    PyArrayObject *u2_array = NULL;
    i32 info = 0;

    z_array = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    fg_array = (PyArrayObject*)PyArray_FROM_OTF(fg_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (z_array == NULL || b_array == NULL || fg_array == NULL) {
        Py_XDECREF(z_array);
        Py_XDECREF(b_array);
        Py_XDECREF(fg_array);
        return NULL;
    }

    npy_intp *z_dims = PyArray_DIMS(z_array);
    i32 n = (i32)z_dims[0];
    i32 m = n / 2;

    i32 ldz = n > 0 ? n : 1;
    i32 ldb = m > 0 ? m : 1;
    i32 ldfg = m > 0 ? m : 1;
    i32 ldq = n > 0 ? n : 1;
    i32 ldu1 = 1;
    i32 ldu2 = 1;

    char compu_upper = (char)toupper((unsigned char)compu[0]);
    bool lcmpu = (compu_upper == 'I') || (compu_upper == 'U');

    if (lcmpu) {
        ldu1 = m > 0 ? m : 1;
        ldu2 = m > 0 ? m : 1;
    }

    f64 *z_data = (f64*)PyArray_DATA(z_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *fg_data = (f64*)PyArray_DATA(fg_array);

    npy_intp q_dims[2] = {n > 0 ? n : 0, n > 0 ? n : 0};
    npy_intp q_strides[2] = {sizeof(f64), n * (i32)sizeof(f64)};
    PyObject *q_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE,
                                    q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (q_array == NULL) {
        Py_DECREF(z_array);
        Py_DECREF(b_array);
        Py_DECREF(fg_array);
        PyErr_NoMemory();
        return NULL;
    }
    f64 *q_data = (f64*)PyArray_DATA((PyArrayObject*)q_array);

    npy_intp u1_dims[2] = {m > 0 ? m : 0, m > 0 ? m : 0};
    npy_intp u1_strides[2] = {sizeof(f64), m * (i32)sizeof(f64)};
    f64 *u1_data = NULL;
    f64 *u2_data = NULL;
    PyObject *u1_out_array = NULL;
    PyObject *u2_out_array = NULL;

    if (lcmpu && m > 0) {
        if (compu_upper == 'U') {
            u1_array = (PyArrayObject*)PyArray_FROM_OTF(u1_obj, NPY_DOUBLE,
                                                        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            u2_array = (PyArrayObject*)PyArray_FROM_OTF(u2_obj, NPY_DOUBLE,
                                                        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            if (u1_array == NULL || u2_array == NULL) {
                Py_XDECREF(u1_array);
                Py_XDECREF(u2_array);
                Py_DECREF(z_array);
                Py_DECREF(b_array);
                Py_DECREF(fg_array);
                Py_DECREF(q_array);
                return NULL;
            }
            u1_data = (f64*)PyArray_DATA(u1_array);
            u2_data = (f64*)PyArray_DATA(u2_array);
            u1_out_array = (PyObject*)u1_array;
            u2_out_array = (PyObject*)u2_array;
            Py_INCREF(u1_out_array);
            Py_INCREF(u2_out_array);
        } else {
            u1_out_array = PyArray_New(&PyArray_Type, 2, u1_dims, NPY_DOUBLE,
                                       u1_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            u2_out_array = PyArray_New(&PyArray_Type, 2, u1_dims, NPY_DOUBLE,
                                       u1_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            if (u1_out_array == NULL || u2_out_array == NULL) {
                Py_XDECREF(u1_out_array);
                Py_XDECREF(u2_out_array);
                Py_DECREF(z_array);
                Py_DECREF(b_array);
                Py_DECREF(fg_array);
                Py_DECREF(q_array);
                PyErr_NoMemory();
                return NULL;
            }
            u1_data = (f64*)PyArray_DATA((PyArrayObject*)u1_out_array);
            u2_data = (f64*)PyArray_DATA((PyArrayObject*)u2_out_array);
        }
    } else {
        u1_dims[0] = 0;
        u1_dims[1] = 0;
        u1_out_array = PyArray_SimpleNew(2, u1_dims, NPY_DOUBLE);
        u2_out_array = PyArray_SimpleNew(2, u1_dims, NPY_DOUBLE);
        if (u1_out_array == NULL || u2_out_array == NULL) {
            Py_XDECREF(u1_out_array);
            Py_XDECREF(u2_out_array);
            Py_DECREF(z_array);
            Py_DECREF(b_array);
            Py_DECREF(fg_array);
            Py_DECREF(q_array);
            return NULL;
        }
        u1_data = (f64*)PyArray_DATA((PyArrayObject*)u1_out_array);
        u2_data = (f64*)PyArray_DATA((PyArrayObject*)u2_out_array);
    }

    npy_intp eig_dim = m > 0 ? m : 0;
    PyObject *alphar_array = PyArray_SimpleNew(1, &eig_dim, NPY_DOUBLE);
    PyObject *alphai_array = PyArray_SimpleNew(1, &eig_dim, NPY_DOUBLE);
    PyObject *beta_array = PyArray_SimpleNew(1, &eig_dim, NPY_DOUBLE);

    if (alphar_array == NULL || alphai_array == NULL || beta_array == NULL) {
        Py_XDECREF(alphar_array);
        Py_XDECREF(alphai_array);
        Py_XDECREF(beta_array);
        Py_XDECREF(u1_array);
        Py_XDECREF(u2_array);
        Py_DECREF(u1_out_array);
        Py_DECREF(u2_out_array);
        Py_DECREF(z_array);
        Py_DECREF(b_array);
        Py_DECREF(fg_array);
        Py_DECREF(q_array);
        return NULL;
    }

    f64 *alphar_data = (f64*)PyArray_DATA((PyArrayObject*)alphar_array);
    f64 *alphai_data = (f64*)PyArray_DATA((PyArrayObject*)alphai_array);
    f64 *beta_data = (f64*)PyArray_DATA((PyArrayObject*)beta_array);

    i32 liwork = n + 9;
    i32 mm = m * m;
    char job_upper = (char)toupper((unsigned char)job[0]);
    char compq_upper = (char)toupper((unsigned char)compq[0]);
    bool ltri = (job_upper == 'T');
    bool lcmpq = (compq_upper == 'I');

    i32 ldwork;
    if (n == 0) {
        ldwork = 4;
    } else if (ltri || lcmpq || lcmpu) {
        ldwork = 6 * mm + (3 * n > 27 ? 3 * n : 27);
    } else {
        ldwork = 3 * mm + (3 * n > 27 ? 3 * n : 27);
    }
    ldwork = ldwork > 4 ? ldwork : 4;

    i32 *iwork = (i32*)malloc(liwork * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if ((iwork == NULL || dwork == NULL) && n > 0) {
        free(iwork);
        free(dwork);
        Py_XDECREF(u1_array);
        Py_XDECREF(u2_array);
        Py_DECREF(u1_out_array);
        Py_DECREF(u2_out_array);
        Py_DECREF(alphar_array);
        Py_DECREF(alphai_array);
        Py_DECREF(beta_array);
        Py_DECREF(z_array);
        Py_DECREF(b_array);
        Py_DECREF(fg_array);
        Py_DECREF(q_array);
        PyErr_NoMemory();
        return NULL;
    }

    mb04ed(job, compq, compu, n, z_data, ldz, b_data, ldb, fg_data, ldfg,
           q_data, ldq, u1_data, ldu1, u2_data, ldu2,
           alphar_data, alphai_data, beta_data,
           iwork, liwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(z_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(fg_array);

    if (compu_upper == 'U') {
        PyArray_ResolveWritebackIfCopy(u1_array);
        PyArray_ResolveWritebackIfCopy(u2_array);
        Py_XDECREF(u1_array);
        Py_XDECREF(u2_array);
        u1_out_array = (PyObject*)PyArray_FROM_OTF(u1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
        u2_out_array = (PyObject*)PyArray_FROM_OTF(u2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    }

    PyObject *result = Py_BuildValue("(OOOOOOOOOi)",
                                     z_array, b_array, fg_array, q_array,
                                     u1_out_array, u2_out_array,
                                     alphar_array, alphai_array, beta_array,
                                     info);

    Py_DECREF(z_array);
    Py_DECREF(b_array);
    Py_DECREF(fg_array);
    Py_DECREF(q_array);
    Py_DECREF(u1_out_array);
    Py_DECREF(u2_out_array);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);

    return result;
}

PyObject* py_mb04fd(PyObject* self, PyObject* args, PyObject* kwargs)
{
    const char *job;
    const char *compq;
    PyObject *a_obj, *de_obj, *b_obj, *fg_obj;
    PyObject *q_obj = Py_None;
    PyObject *n_obj = Py_None;

    static char *kwlist[] = {"job", "compq", "a", "de", "b", "fg", "q", "n", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssOOOO|OO", kwlist,
                                     &job, &compq, &a_obj, &de_obj, &b_obj, &fg_obj,
                                     &q_obj, &n_obj)) {
        return NULL;
    }

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

    char compq_upper = (char)toupper((unsigned char)compq[0]);

    i32 m = (i32)PyArray_DIM(a_array, 0);
    i32 n;

    if (n_obj != Py_None) {
        n = (i32)PyLong_AsLong(n_obj);
    } else {
        n = 2 * m;
    }

    i32 lda = (i32)PyArray_DIM(a_array, 0);
    i32 ldde = (i32)PyArray_DIM(de_array, 0);
    i32 ldb = (i32)PyArray_DIM(b_array, 0);
    i32 ldfg = (i32)PyArray_DIM(fg_array, 0);

    if (lda < 1) lda = 1;
    if (ldde < 1) ldde = 1;
    if (ldb < 1) ldb = 1;
    if (ldfg < 1) ldfg = 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *de_data = (f64*)PyArray_DATA(de_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *fg_data = (f64*)PyArray_DATA(fg_array);

    bool lcmpq = (compq_upper == 'I' || compq_upper == 'U');
    i32 ldq = lcmpq ? n : 1;
    if (ldq < 1) ldq = 1;

    PyArrayObject *q_array = NULL;
    f64 *q_data = NULL;

    if (compq_upper == 'U') {
        q_array = (PyArrayObject*)PyArray_FROM_OTF(
            q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!q_array) {
            Py_DECREF(a_array);
            Py_DECREF(de_array);
            Py_DECREF(b_array);
            Py_DECREF(fg_array);
            return NULL;
        }
        q_data = (f64*)PyArray_DATA(q_array);
    } else if (lcmpq) {
        npy_intp q_dims[2] = {n, n};
        npy_intp q_strides[2] = {sizeof(f64), n * sizeof(f64)};
        q_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE,
                                              q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!q_array) {
            Py_DECREF(a_array);
            Py_DECREF(de_array);
            Py_DECREF(b_array);
            Py_DECREF(fg_array);
            PyErr_NoMemory();
            return NULL;
        }
        q_data = (f64*)PyArray_DATA(q_array);
    } else {
        npy_intp q_dims[2] = {1, 1};
        q_array = (PyArrayObject*)PyArray_ZEROS(2, q_dims, NPY_DOUBLE, 1);
        q_data = (f64*)PyArray_DATA(q_array);
    }

    i32 m_out = n / 2;
    if (m_out < 0) m_out = 0;

    npy_intp eig_dim = m_out;
    PyArrayObject *alphar_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 1, &eig_dim, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
    PyArrayObject *alphai_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 1, &eig_dim, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
    PyArrayObject *beta_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 1, &eig_dim, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);

    if (!alphar_array || !alphai_array || !beta_array) {
        Py_XDECREF(alphar_array);
        Py_XDECREF(alphai_array);
        Py_XDECREF(beta_array);
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(b_array);
        Py_DECREF(fg_array);
        Py_DECREF(q_array);
        PyErr_NoMemory();
        return NULL;
    }
    f64 *alphar_data = (f64*)PyArray_DATA(alphar_array);
    f64 *alphai_data = (f64*)PyArray_DATA(alphai_array);
    f64 *beta_data = (f64*)PyArray_DATA(beta_array);

    i32 mm = m_out * m_out;
    i32 mindw;

    if (lcmpq) {
        mindw = 3 > 2 * mm + mm - 1 ? 3 : 2 * mm + mm - 1;
    } else if (job[0] == 'T' || job[0] == 't') {
        mindw = 3 > mm + m_out - 1 ? 3 : mm + m_out - 1;
    } else {
        mindw = 3 > m_out ? 3 : m_out;
    }

    i32 ldwork = mindw + m_out * m_out;
    i32 liwork = m_out + 2;

    i32 *iwork = (i32*)calloc(liwork, sizeof(i32));
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        Py_DECREF(alphar_array);
        Py_DECREF(alphai_array);
        Py_DECREF(beta_array);
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(b_array);
        Py_DECREF(fg_array);
        Py_DECREF(q_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info = 0;

    mb04fd(job, compq, n, a_data, lda, de_data, ldde, b_data, ldb, fg_data, ldfg,
           q_data, ldq, alphar_data, alphai_data, beta_data,
           iwork, dwork, ldwork, &info);

    npy_intp iwork_dim = m_out + 1;
    PyArrayObject *iwork_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 1, &iwork_dim, NPY_INT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
    if (iwork_array) {
        i32 *iwork_out = (i32*)PyArray_DATA(iwork_array);
        for (i32 ii = 0; ii <= m_out; ii++) {
            iwork_out[ii] = iwork[ii];
        }
    }

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(de_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(fg_array);

    if (compq_upper == 'U') {
        PyArray_ResolveWritebackIfCopy(q_array);
    }

    PyObject *result = Py_BuildValue("(OOOOOOOOOi)",
                                     a_array, de_array, b_array, fg_array, q_array,
                                     alphar_array, alphai_array, beta_array,
                                     iwork_array, info);

    Py_DECREF(a_array);
    Py_DECREF(de_array);
    Py_DECREF(b_array);
    Py_DECREF(fg_array);
    Py_DECREF(q_array);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);
    Py_XDECREF(iwork_array);

    return result;
}

PyObject* py_mb04gd(PyObject* self, PyObject* args) {
    PyObject *a_obj, *jpvt_obj;
    PyArrayObject *a_array = NULL, *jpvt_array = NULL;

    if (!PyArg_ParseTuple(args, "OO", &a_obj, &jpvt_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    jpvt_array = (PyArrayObject*)PyArray_FROM_OTF(jpvt_obj, NPY_INT32,
                                                  NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (jpvt_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 ndim_a = PyArray_NDIM(a_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);

    i32 m, n;
    if (ndim_a == 2) {
        m = (i32)a_dims[0];
        n = (i32)a_dims[1];
    } else if (ndim_a == 1) {
        m = (i32)a_dims[0];
        n = 1;
    } else {
        m = 0;
        n = 0;
    }

    i32 lda = (m > 1) ? m : 1;
    i32 k = (m < n) ? m : n;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    i32 *jpvt_data = (i32*)PyArray_DATA(jpvt_array);

    npy_intp tau_dim = k > 0 ? k : 0;
    PyArrayObject *tau_array = NULL;
    f64 *tau_data = NULL;
    if (tau_dim > 0) {
        tau_array = (PyArrayObject*)PyArray_New(
            &PyArray_Type, 1, &tau_dim, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
        if (tau_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(jpvt_array);
            PyErr_NoMemory();
            return NULL;
        }
        tau_data = (f64*)PyArray_DATA(tau_array);
        memset(tau_data, 0, tau_dim * sizeof(f64));
    } else {
        npy_intp zero_dim = 0;
        tau_array = (PyArrayObject*)PyArray_EMPTY(1, &zero_dim, NPY_DOUBLE, 0);
        if (tau_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(jpvt_array);
            PyErr_NoMemory();
            return NULL;
        }
    }

    i32 ldwork = 3 * m > 1 ? 3 * m : 1;
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(tau_array);
        Py_DECREF(a_array);
        Py_DECREF(jpvt_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info = 0;
    mb04gd(m, n, a_data, lda, jpvt_data, tau_data, dwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(jpvt_array);

    PyObject *result = Py_BuildValue("(OOOi)", a_array, jpvt_array, tau_array, info);

    Py_DECREF(a_array);
    Py_DECREF(jpvt_array);
    Py_DECREF(tau_array);

    return result;
}

PyObject* py_mb04iz(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static char *kwlist[] = {"a", "p", "b", "lzwork", NULL};

    PyObject *a_obj = NULL;
    PyObject *b_obj = Py_None;
    i32 p = 0;
    i32 lzwork_in = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|Oi", kwlist,
                                     &a_obj, &p, &b_obj, &lzwork_in)) {
        return NULL;
    }

    if (p < 0) {
        PyErr_SetString(PyExc_ValueError, "p must be non-negative");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL) {
        return NULL;
    }

    int ndim_a = PyArray_NDIM(a_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);

    i32 n, m;
    if (ndim_a == 2) {
        n = (i32)a_dims[0];
        m = (i32)a_dims[1];
    } else if (ndim_a == 1) {
        n = (i32)a_dims[0];
        m = 1;
    } else if (ndim_a == 0) {
        n = 0;
        m = 0;
    } else {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "a must be 1D or 2D");
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;

    PyArrayObject *b_array = NULL;
    i32 l = 0;
    i32 ldb = 1;
    c128 *b_data = NULL;

    if (b_obj != Py_None) {
        b_array = (PyArrayObject*)PyArray_FROM_OTF(
            b_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (b_array == NULL) {
            Py_DECREF(a_array);
            return NULL;
        }
        int ndim_b = PyArray_NDIM(b_array);
        npy_intp *b_dims = PyArray_DIMS(b_array);

        if (ndim_b == 2) {
            if ((i32)b_dims[0] != n) {
                Py_DECREF(a_array);
                Py_DECREF(b_array);
                PyErr_SetString(PyExc_ValueError, "b must have same number of rows as a");
                return NULL;
            }
            l = (i32)b_dims[1];
        } else if (ndim_b == 1) {
            if ((i32)b_dims[0] != n && n > 0) {
                Py_DECREF(a_array);
                Py_DECREF(b_array);
                PyErr_SetString(PyExc_ValueError, "b must have same length as rows of a");
                return NULL;
            }
            l = 1;
        } else {
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            PyErr_SetString(PyExc_ValueError, "b must be 1D or 2D");
            return NULL;
        }
        ldb = n > 0 ? n : 1;
        b_data = (c128*)PyArray_DATA(b_array);
    }

    i32 tau_len = (n < m ? n : m);
    if (tau_len < 0) tau_len = 0;

    c128 *tau_data = NULL;
    PyArrayObject *tau_array = NULL;
    if (tau_len > 0) {
        npy_intp tau_dim = tau_len;
        tau_array = (PyArrayObject*)PyArray_New(
            &PyArray_Type, 1, &tau_dim, NPY_CDOUBLE, 
            NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (tau_array == NULL) {
            Py_DECREF(a_array);
            Py_XDECREF(b_array);
            return NULL;
        }
        tau_data = (c128*)PyArray_DATA(tau_array);
        memset(tau_data, 0, tau_len * sizeof(c128));
    }

    i32 minwork = 1;
    if (m > 1 && m - 1 > minwork) minwork = m - 1;
    if (m > p && m - p > minwork) minwork = m - p;
    if (l > minwork) minwork = l;

    i32 lzwork = lzwork_in > 0 ? lzwork_in : minwork;
    bool workspace_query = (lzwork_in == -1);
    if (workspace_query) {
        lzwork = -1;
    }

    c128 *zwork = (c128*)malloc((lzwork > 0 ? lzwork : 1) * sizeof(c128));
    if (zwork == NULL) {
        Py_XDECREF(tau_array);
        Py_DECREF(a_array);
        Py_XDECREF(b_array);
        PyErr_NoMemory();
        return NULL;
    }

    c128 *a_data = (c128*)PyArray_DATA(a_array);
    i32 info = 0;

    mb04iz(n, m, p, l, a_data, lda, b_data, ldb, tau_data, zwork, lzwork, &info);

    i32 zwork_opt = (i32)creal(zwork[0]);
    free(zwork);

    if (info != 0 && !workspace_query) {
        Py_XDECREF(tau_array);
        Py_DECREF(a_array);
        Py_XDECREF(b_array);
        if (info < 0) {
            PyErr_Format(PyExc_ValueError, "MB04IZ: illegal value in argument %d", -info);
        } else {
            PyErr_Format(PyExc_RuntimeError, "MB04IZ: algorithm error, info = %d", info);
        }
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(a_array);
    if (b_array != NULL) {
        PyArray_ResolveWritebackIfCopy(b_array);
    }

    if (tau_array == NULL) {
        npy_intp zero_dim = 0;
        tau_array = (PyArrayObject*)PyArray_EMPTY(1, &zero_dim, NPY_CDOUBLE, 0);
        if (tau_array == NULL) {
            Py_DECREF(a_array);
            Py_XDECREF(b_array);
            return NULL;
        }
    }

    PyObject *result_dict = PyDict_New();
    if (result_dict == NULL) {
        Py_DECREF(a_array);
        Py_XDECREF(b_array);
        Py_DECREF(tau_array);
        return NULL;
    }

    PyDict_SetItemString(result_dict, "a", (PyObject*)a_array);
    PyDict_SetItemString(result_dict, "tau", (PyObject*)tau_array);
    PyDict_SetItemString(result_dict, "info", PyLong_FromLong(info));
    PyDict_SetItemString(result_dict, "zwork_opt", PyLong_FromLong(zwork_opt));

    if (b_array != NULL) {
        PyDict_SetItemString(result_dict, "b", (PyObject*)b_array);
    }

    Py_DECREF(a_array);
    Py_XDECREF(b_array);
    Py_DECREF(tau_array);

    return result_dict;
}

PyObject* py_mb04jd(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"n", "m", "p", "a", "b", "l", "ldwork", NULL};
    i32 n, m, p, l = 0;
    i32 ldwork = 0;
    PyObject *a_obj, *b_obj = NULL;
    PyArrayObject *a_array = NULL, *b_array = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiiO|Oii", kwlist,
                                     &n, &m, &p, &a_obj, &b_obj, &l, &ldwork)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda = (i32)a_dims[0];

    i32 ldb = l > 0 ? l : 1;
    bool has_b = (b_obj != NULL && l > 0);

    if (has_b) {
        b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (b_array == NULL) {
            Py_DECREF(a_array);
            return NULL;
        }
        npy_intp *b_dims = PyArray_DIMS(b_array);
        ldb = (i32)b_dims[0];
    }

    i32 minwork = 1;
    if (n > 1 && n - 1 > minwork) minwork = n - 1;
    if (n > p && n - p > minwork) minwork = n - p;
    if (l > minwork) minwork = l;

    if (ldwork == 0) ldwork = minwork;

    f64 *dwork = (f64*)calloc(ldwork > 0 ? ldwork : 1, sizeof(f64));
    if (dwork == NULL) {
        Py_XDECREF(b_array);
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    i32 minval = (n < m ? n : m);
    PyObject *tau_array = NULL;
    f64 *tau = NULL;

    if (minval > 0) {
        npy_intp tau_dims[1] = {minval};
        tau_array = PyArray_SimpleNew(1, tau_dims, NPY_DOUBLE);
        if (tau_array == NULL) {
            free(dwork);
            Py_XDECREF(b_array);
            Py_DECREF(a_array);
            return NULL;
        }
        tau = (f64*)PyArray_DATA((PyArrayObject*)tau_array);
        memset(tau, 0, minval * sizeof(f64));
    } else {
        // Handle empty tau array case if necessary or use dummy logic, 
        // but mb04jd expects tau to be pointer?
        // original code: tau = calloc(minval > 0 ? minval : 1, ...)
        tau = (f64*)calloc(1, sizeof(f64)); // Dummy for pointer
        if (tau == NULL) {
            free(dwork);
            Py_XDECREF(b_array);
            Py_DECREF(a_array);
            PyErr_NoMemory();
            return NULL;
        }
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = has_b ? (f64*)PyArray_DATA(b_array) : NULL;
    f64 dummy_b = 0.0;
    if (!has_b) {
        b_data = &dummy_b;
        ldb = 1;
    }

    i32 info;
    mb04jd(n, m, p, l, a_data, lda, b_data, ldb, tau, dwork, ldwork, &info);

    f64 optimal_work = dwork[0];
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    if (has_b) {
        PyArray_ResolveWritebackIfCopy(b_array);
    }

    if (info != 0) {
        if (tau_array) {
            Py_DECREF(tau_array);
        } else {
            free(tau);
        }
        Py_XDECREF(b_array);
        Py_DECREF(a_array);
        PyErr_Format(PyExc_ValueError, "mb04jd failed with info=%d", info);
        return NULL;
    }

    if (tau_array == NULL) {
        npy_intp tau_dims[1] = {1};
        tau_array = PyArray_New(&PyArray_Type, 1, tau_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
        if (tau_array == NULL) {
             free(tau);
             Py_XDECREF(b_array);
             Py_DECREF(a_array);
             PyErr_NoMemory();
             return NULL;
        }
        f64 *tau_out = (f64*)PyArray_DATA((PyArrayObject*)tau_array);
        tau_out[0] = tau[0];
        free(tau);
    }

    PyObject *res_jd;
    if (ldwork == -1) {
        if (has_b) {
            res_jd = Py_BuildValue("OOOid", a_array, b_array, tau_array, info, optimal_work);
        } else {
            res_jd = Py_BuildValue("OOid", a_array, tau_array, info, optimal_work);
        }
    } else {
        if (has_b) {
            res_jd = Py_BuildValue("OOOi", a_array, b_array, tau_array, info);
        } else {
            res_jd = Py_BuildValue("OOi", a_array, tau_array, info);
        }
    }

    Py_XDECREF(b_array);
    Py_DECREF(a_array);
    Py_DECREF(tau_array);

    return res_jd;
}

PyObject* py_mb04az(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"job", "compq", "compu", "z", "b", "fg", NULL};

    const char *job, *compq, *compu;
    PyObject *z_obj, *b_obj, *fg_obj;
    PyObject *b_out = NULL, *fg_out = NULL;
    PyObject *d_arr = NULL, *c_arr = NULL;
    PyObject *q_arr = NULL, *u_arr = NULL;
    PyObject *alphar_array = NULL, *alphai_array = NULL, *beta_py = NULL;
    f64 *alphar = NULL, *alphai = NULL, *beta_arr = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssOOO", kwlist,
                                     &job, &compq, &compu,
                                     &z_obj, &b_obj, &fg_obj)) {
        return NULL;
    }

    char job_upper = (char)toupper((unsigned char)job[0]);
    char compq_upper = (char)toupper((unsigned char)compq[0]);
    char compu_upper = (char)toupper((unsigned char)compu[0]);

    bool ltri = (job_upper == 'T');
    bool lcmpq = (compq_upper == 'C');
    bool lcmpu = (compu_upper == 'C');

    PyArrayObject *z_array = (PyArrayObject *)PyArray_FROM_OTF(
        z_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (z_array == NULL) return NULL;

    i32 n = (i32)PyArray_DIM(z_array, 0);
    i32 m = n / 2;
    i32 n2 = 2 * n;

    if (n < 0 || (n % 2) != 0) {
        Py_DECREF(z_array);
        PyErr_SetString(PyExc_ValueError, "N must be non-negative and even");
        return NULL;
    }

    PyArrayObject *b_in_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);
    if (b_in_array == NULL) {
        Py_DECREF(z_array);
        return NULL;
    }

    PyArrayObject *fg_in_array = (PyArrayObject *)PyArray_FROM_OTF(
        fg_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);
    if (fg_in_array == NULL) {
        Py_DECREF(b_in_array);
        Py_DECREF(z_array);
        return NULL;
    }

    c128 *z_data = (c128 *)PyArray_DATA(z_array);
    c128 *b_in_data = (c128 *)PyArray_DATA(b_in_array);
    c128 *fg_in_data = (c128 *)PyArray_DATA(fg_in_array);

    i32 ldz = (i32)PyArray_DIM(z_array, 0);
    if (ldz < 1) ldz = 1;

    i32 ldb, ldfg, k_b, p_fg;
    c128 *b_data = NULL, *fg_data = NULL;


    if (ltri) {
        ldb = (n > 1) ? n : 1;
        k_b = n;
        ldfg = (n > 1) ? n : 1;
        p_fg = (m + 1 > n) ? m + 1 : n;

        npy_intp b_dims[2] = {ldb, k_b};
        npy_intp b_strides[2] = {sizeof(c128), ldb * sizeof(c128)};
        b_out = PyArray_New(&PyArray_Type, 2, b_dims, NPY_COMPLEX128,
                                      b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

        npy_intp fg_dims[2] = {ldfg, p_fg};
        npy_intp fg_strides[2] = {sizeof(c128), ldfg * sizeof(c128)};
        fg_out = PyArray_New(&PyArray_Type, 2, fg_dims, NPY_COMPLEX128,
                                       fg_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

        if (b_out == NULL || fg_out == NULL) {
            Py_XDECREF(b_out); Py_XDECREF(fg_out);
            Py_DECREF(fg_in_array); Py_DECREF(b_in_array); Py_DECREF(z_array);
            return NULL;
        }
        b_data = (c128*)PyArray_DATA((PyArrayObject*)b_out);
        fg_data = (c128*)PyArray_DATA((PyArrayObject*)fg_out);
        if (ldb * k_b > 0) {
            memset(b_data, 0, ldb * k_b * sizeof(c128));
        }
        if (ldfg * p_fg > 0) {
            memset(fg_data, 0, ldfg * p_fg * sizeof(c128));
        }



        for (i32 j = 0; j < m; j++) {
            for (i32 i = 0; i < m; i++) {
                b_data[i + j * ldb] = b_in_data[i + j * m];
            }
        }
        i32 fg_in_ldfg = (i32)PyArray_DIM(fg_in_array, 0);
        i32 fg_in_cols = (i32)PyArray_DIM(fg_in_array, 1);
        for (i32 j = 0; j < fg_in_cols; j++) {
            for (i32 i = 0; i < m; i++) {
                fg_data[i + j * ldfg] = fg_in_data[i + j * fg_in_ldfg];
            }
        }
    } else {
        ldb = (m > 1) ? m : 1;
        ldfg = (m > 1) ? m : 1;
        b_data = b_in_data;
        fg_data = fg_in_data;
    }

    i32 ldd = ltri ? (n > 1 ? n : 1) : 1;
    i32 ldc = ltri ? (n > 1 ? n : 1) : 1;
    i32 ldq = lcmpq ? (n2 > 1 ? n2 : 1) : 1;
    i32 ldu = lcmpu ? (n > 1 ? n : 1) : 1;

    c128 *d_data = NULL, *c_data = NULL, *q_data = NULL, *u_data = NULL;
    c128 dummy_d, dummy_c, dummy_q, dummy_u;

    if (ltri) {
        npy_intp nn_dims[2] = {n > 0 ? n : 0, n > 0 ? n : 0};
        npy_intp nn_strides[2] = {sizeof(c128), n > 0 ? n * sizeof(c128) : sizeof(c128)};

        d_arr = PyArray_New(&PyArray_Type, 2, nn_dims, NPY_COMPLEX128,
                                      nn_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        c_arr = PyArray_New(&PyArray_Type, 2, nn_dims, NPY_COMPLEX128,
                                      nn_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

        if (d_arr == NULL || c_arr == NULL) {
            Py_XDECREF(d_arr); Py_XDECREF(c_arr);
            Py_XDECREF(b_out); Py_XDECREF(fg_out);
            Py_DECREF(fg_in_array);
            Py_DECREF(b_in_array);
            Py_DECREF(z_array);
            return NULL;
        }
        d_data = (c128*)PyArray_DATA((PyArrayObject*)d_arr);
        c_data = (c128*)PyArray_DATA((PyArrayObject*)c_arr);
        if (n > 0) {
            memset(d_data, 0, n * n * sizeof(c128));
            memset(c_data, 0, n * n * sizeof(c128));
        }
    } else {
        d_data = &dummy_d;
        c_data = &dummy_c;
    }

    if (lcmpq) {
        npy_intp q_dims[2] = {n2, n2};
        npy_intp q_strides[2] = {sizeof(c128), n2 * sizeof(c128)};
        q_arr = PyArray_New(&PyArray_Type, 2, q_dims, NPY_COMPLEX128,
                                      q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

        if (q_arr == NULL) {
            Py_XDECREF(d_arr); Py_XDECREF(c_arr);
            Py_XDECREF(b_out); Py_XDECREF(fg_out);
            Py_DECREF(fg_in_array);
            Py_DECREF(b_in_array);
            Py_DECREF(z_array);
            return NULL;
        }
        q_data = (c128*)PyArray_DATA((PyArrayObject*)q_arr);
        if (n2 > 0) {
            memset(q_data, 0, n2 * n2 * sizeof(c128));
        }
    } else {
        q_data = &dummy_q;
    }

    if (lcmpu) {
        npy_intp u_dims[2] = {n, n2};
        npy_intp u_strides[2] = {sizeof(c128), n * sizeof(c128)};
        u_arr = PyArray_New(&PyArray_Type, 2, u_dims, NPY_COMPLEX128,
                                      u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

        if (u_arr == NULL) {
            Py_XDECREF(q_arr);
            Py_XDECREF(d_arr); Py_XDECREF(c_arr);
            Py_XDECREF(b_out); Py_XDECREF(fg_out);
            Py_DECREF(fg_in_array);
            Py_DECREF(b_in_array);
            Py_DECREF(z_array);
            return NULL;
        }
        u_data = (c128*)PyArray_DATA((PyArrayObject*)u_arr);
        if (n > 0) {
            memset(u_data, 0, n * n2 * sizeof(c128));
        }
    } else {
        u_data = &dummy_u;
    }

    npy_intp n_dim[1] = {n};
    alphar_array = PyArray_SimpleNew(1, n_dim, NPY_DOUBLE);
    alphai_array = PyArray_SimpleNew(1, n_dim, NPY_DOUBLE);
    beta_py = PyArray_SimpleNew(1, n_dim, NPY_DOUBLE);

    if (alphar_array == NULL || alphai_array == NULL || beta_py == NULL) {
        Py_XDECREF(alphar_array); Py_XDECREF(alphai_array); Py_XDECREF(beta_py);
        Py_XDECREF(u_arr);
        Py_XDECREF(q_arr);
        Py_XDECREF(d_arr); Py_XDECREF(c_arr);
        Py_XDECREF(b_out); Py_XDECREF(fg_out);
        Py_DECREF(fg_in_array);
        Py_DECREF(b_in_array);
        Py_DECREF(z_array);
        return NULL;
    }
    alphar = (f64*)PyArray_DATA((PyArrayObject*)alphar_array);
    alphai = (f64*)PyArray_DATA((PyArrayObject*)alphai_array);
    beta_arr = (f64*)PyArray_DATA((PyArrayObject*)beta_py);

    if (n > 0) {
        memset(alphar, 0, n * sizeof(f64));
        memset(alphai, 0, n * sizeof(f64));
        memset(beta_arr, 0, n * sizeof(f64));
    }

    i32 liwork = n2 + 9;
    i32 *iwork = (i32 *)calloc(liwork, sizeof(i32));
    if (iwork == NULL) {
        Py_XDECREF(alphar_array); Py_XDECREF(alphai_array); Py_XDECREF(beta_py);
        Py_XDECREF(u_arr);
        Py_XDECREF(q_arr);
        Py_XDECREF(d_arr); Py_XDECREF(c_arr);
        Py_XDECREF(b_out); Py_XDECREF(fg_out);
        Py_DECREF(fg_in_array);
        Py_DECREF(b_in_array);
        Py_DECREF(z_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 nn = n * n;
    i32 mindw, minzw;
    if (n == 0) {
        mindw = 4;
        minzw = 1;
    } else {
        if (ltri) {
            minzw = lcmpq ? 4 * n2 + 28 : 3 * n2 + 28;
        } else {
            minzw = 1;
        }
        i32 j_coef;
        if (lcmpu) {
            j_coef = 18;
        } else if (lcmpq) {
            j_coef = 16;
        } else {
            j_coef = 13;
        }
        i32 max_3n2_27 = (3 * n2 > 27) ? 3 * n2 : 27;
        mindw = j_coef * nn + n + max_3n2_27;
    }

    i32 ldwork = mindw > 4 ? mindw : 4;
    i32 lzwork = minzw > 1 ? minzw : 1;

    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));
    c128 *zwork = (c128 *)PyMem_Calloc(lzwork, sizeof(c128));
    if (dwork == NULL || zwork == NULL) {
        free(dwork);
        PyMem_Free(zwork);
        free(iwork);
        Py_XDECREF(alphar_array); Py_XDECREF(alphai_array); Py_XDECREF(beta_py);
        Py_XDECREF(u_arr);
        Py_XDECREF(q_arr);
        Py_XDECREF(d_arr); Py_XDECREF(c_arr);
        Py_XDECREF(b_out); Py_XDECREF(fg_out);
        Py_DECREF(fg_in_array);
        Py_DECREF(b_in_array);
        Py_DECREF(z_array);
        PyErr_NoMemory();
        return NULL;
    }

    bool *bwork = NULL;
    if (ltri && n > 0) {
        bwork = (bool *)calloc(n, sizeof(bool));
        if (bwork == NULL) {
            free(dwork);
            PyMem_Free(zwork);
            free(iwork);
            Py_XDECREF(alphar_array); Py_XDECREF(alphai_array); Py_XDECREF(beta_py);
            Py_XDECREF(u_arr);
            Py_XDECREF(q_arr);
            Py_XDECREF(d_arr); Py_XDECREF(c_arr);
            Py_XDECREF(b_out); Py_XDECREF(fg_out);
            Py_DECREF(fg_in_array);
            Py_DECREF(b_in_array);
            Py_DECREF(z_array);
            PyErr_NoMemory();
            return NULL;
        }
    }

    i32 info;
    mb04az(job, compq, compu, n, z_data, ldz, b_data, ldb, fg_data, ldfg,
           d_data, ldd, c_data, ldc, q_data, ldq, u_data, ldu,
           alphar, alphai, beta_arr, iwork, liwork, dwork, ldwork,
           zwork, lzwork, bwork, &info);

    free(dwork);
    PyMem_Free(zwork);
    free(iwork);
    if (bwork) free(bwork);

    PyArray_ResolveWritebackIfCopy(z_array);

    if (info < 0) {
        Py_XDECREF(alphar_array); Py_XDECREF(alphai_array); Py_XDECREF(beta_py);
        Py_XDECREF(u_arr);
        Py_XDECREF(q_arr);
        Py_XDECREF(d_arr); Py_XDECREF(c_arr);
        Py_XDECREF(b_out); Py_XDECREF(fg_out);
        Py_DECREF(fg_in_array);
        Py_DECREF(b_in_array);
        Py_DECREF(z_array);
        PyErr_Format(PyExc_ValueError, "mb04az: parameter %d is invalid", -info);
        return NULL;
    }

    PyObject *result;

    if (ltri) {
        /* Arrays already created: b_out, fg_out, d_arr, c_arr (and maybe q_arr, u_arr) */

        if (lcmpq && lcmpu) {
            result = Py_BuildValue("OOOOOOOOOOi",
                z_array, b_out, fg_out, d_arr, c_arr,
                q_arr, u_arr, alphar_array, alphai_array, beta_py, info);
        } else if (lcmpq) {
            result = Py_BuildValue("OOOOOOOOOi",
                z_array, b_out, fg_out, d_arr, c_arr,
                q_arr, alphar_array, alphai_array, beta_py, info);
        } else if (lcmpu) {
            result = Py_BuildValue("OOOOOOOOOi",
                z_array, b_out, fg_out, d_arr, c_arr,
                u_arr, alphar_array, alphai_array, beta_py, info);
        } else {
            result = Py_BuildValue("OOOOOOOOi",
                z_array, b_out, fg_out, d_arr, c_arr,
                alphar_array, alphai_array, beta_py, info);
        }
        
        Py_XDECREF(b_out);
        Py_XDECREF(fg_out);
        Py_XDECREF(d_arr);
        Py_XDECREF(c_arr);
        Py_XDECREF(q_arr);
        Py_XDECREF(u_arr);
    } else {
        result = Py_BuildValue("OOOi",
            alphar_array, alphai_array, beta_py, info);
            
        // No extra arrays to free in this case, except possibly q_arr/u_arr if lcmpq/lcmpu was set but ltri was false?
        // Wait, q_arr/u_arr logic was independent of ltri in logic above?
        // Let's check:
        // if (lcmpq) { create q_arr } else { dummy }
        // if (lcmpu) { create u_arr } else { dummy }
        // So yes, q_arr and u_arr CAN be non-null here!
        Py_XDECREF(q_arr);
        Py_XDECREF(u_arr);
    }

    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_py);
    Py_DECREF(fg_in_array);
    Py_DECREF(b_in_array);
    Py_DECREF(z_array);

    return result;
}

PyObject* py_mb04hd(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"compq1", "compq2", "a", "b", "q1", "q2", NULL};

    const char *compq1, *compq2;
    PyObject *a_obj, *b_obj;
    PyObject *q1_obj = NULL, *q2_obj = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssOO|OO", kwlist,
                                     &compq1, &compq2, &a_obj, &b_obj,
                                     &q1_obj, &q2_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || b_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        PyErr_SetString(PyExc_ValueError, "Could not convert inputs to arrays");
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 lda = (n > 1) ? n : 1;
    i32 ldb = (n > 1) ? n : 1;

    char cq1 = toupper((unsigned char)compq1[0]);
    char cq2 = toupper((unsigned char)compq2[0]);
    bool lcmpq1 = (cq1 == 'I') || (cq1 == 'U');
    bool lcmpq2 = (cq2 == 'I') || (cq2 == 'U');

    i32 ldq1 = lcmpq1 ? (n > 1 ? n : 1) : 1;
    i32 ldq2 = lcmpq2 ? (n > 1 ? n : 1) : 1;

    f64 *a = (f64 *)PyArray_DATA(a_array);
    f64 *b = (f64 *)PyArray_DATA(b_array);
    f64 *q1_data = NULL;
    f64 *q2_data = NULL;
    PyArrayObject *q1_array = NULL;
    PyArrayObject *q2_array = NULL;
    PyObject *q1_out = NULL;
    PyObject *q2_out = NULL;

    npy_intp dims[2] = {n, n};
    npy_intp strides[2] = {sizeof(f64), n * sizeof(f64)};

    if (cq1 == 'U') {
        if (q1_obj == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            PyErr_SetString(PyExc_ValueError, "q1 required when compq1='U'");
            return NULL;
        }
        q1_array = (PyArrayObject *)PyArray_FROM_OTF(
            q1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (q1_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            return NULL;
        }
        q1_out = (PyObject *)q1_array;
        Py_INCREF(q1_out);
        q1_data = (f64 *)PyArray_DATA(q1_array);
    } else if (cq1 == 'I') {
        q1_out = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                             strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (q1_out == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            return NULL;
        }
        q1_data = (f64 *)PyArray_DATA((PyArrayObject *)q1_out);
        if (n > 0) memset(q1_data, 0, n * n * sizeof(f64));
    }

    if (cq2 == 'U') {
        if (q2_obj == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_XDECREF(q1_out);
            Py_XDECREF(q1_array);
            PyErr_SetString(PyExc_ValueError, "q2 required when compq2='U'");
            return NULL;
        }
        q2_array = (PyArrayObject *)PyArray_FROM_OTF(
            q2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (q2_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_XDECREF(q1_out);
            Py_XDECREF(q1_array);
            return NULL;
        }
        q2_out = (PyObject *)q2_array;
        Py_INCREF(q2_out);
        q2_data = (f64 *)PyArray_DATA(q2_array);
    } else if (cq2 == 'I') {
        q2_out = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                             strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (q2_out == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_XDECREF(q1_out);
            Py_XDECREF(q1_array);
            return NULL;
        }
        q2_data = (f64 *)PyArray_DATA((PyArrayObject *)q2_out);
        if (n > 0) memset(q2_data, 0, n * n * sizeof(f64));
    }

    i32 m = n / 2;
    i32 liwork = (m + 1 > 32) ? m + 1 : 32;
    i32 ldwork = 2 * n * n + (m + 168 > 272 ? m + 168 : 272);
    if (ldwork < 1) ldwork = 1;

    i32 *iwork = (i32 *)calloc(liwork, sizeof(i32));
    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));
    i32 *bwork = (i32 *)calloc(m > 0 ? m : 1, sizeof(i32));

    if ((iwork == NULL || dwork == NULL || bwork == NULL) && n > 0) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_XDECREF(q1_out);
        Py_XDECREF(q2_out);
        Py_XDECREF(q1_array);
        Py_XDECREF(q2_array);
        free(iwork);
        free(dwork);
        free(bwork);
        return PyErr_NoMemory();
    }

    i32 info = 0;

    mb04hd(compq1, compq2, n, a, lda, b, ldb,
           lcmpq1 ? q1_data : NULL, ldq1,
           lcmpq2 ? q2_data : NULL, ldq2,
           iwork, liwork, dwork, ldwork, bwork, &info);

    free(iwork);
    free(dwork);
    free(bwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    if (q1_array) PyArray_ResolveWritebackIfCopy(q1_array);
    if (q2_array) PyArray_ResolveWritebackIfCopy(q2_array);

    PyObject *result;

    if (!lcmpq1) {
        q1_out = Py_None;
        Py_INCREF(q1_out);
    }

    if (!lcmpq2) {
        q2_out = Py_None;
        Py_INCREF(q2_out);
    }

    result = Py_BuildValue("OOOOi", a_array, b_array, q1_out, q2_out, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(q1_out);
    Py_DECREF(q2_out);
    Py_XDECREF(q1_array);
    Py_XDECREF(q2_array);

    return result;
}


PyObject* py_mb04ld(PyObject* self, PyObject* args) {
    const char *uplo_str;
    i32 n, m, p;
    PyObject *l_obj, *a_obj, *b_obj;
    PyArrayObject *l_array, *a_array, *b_array;

    if (!PyArg_ParseTuple(args, "siiiOOO", &uplo_str, &n, &m, &p,
                          &l_obj, &a_obj, &b_obj)) {
        return NULL;
    }

    if (uplo_str == NULL || uplo_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "uplo must be a non-empty string");
        return NULL;
    }
    char uplo = uplo_str[0];

    l_array = (PyArrayObject*)PyArray_FROM_OTF(l_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (l_array == NULL) return NULL;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        Py_DECREF(l_array);
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(l_array);
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *l_dims = PyArray_DIMS(l_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);

    i32 ldl = (i32)l_dims[0];
    i32 lda = (i32)a_dims[0];
    i32 ldb = (i32)b_dims[0];
    i32 ldc = p > 0 ? p : 1;

    npy_intp c_dims[2] = {p, n};
    npy_intp c_strides[2] = {sizeof(f64), ldc * sizeof(f64)};
    
    PyObject *c_array = PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE,
                                    c_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (c_array == NULL) {
        Py_DECREF(l_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }
    f64 *c_data = (f64*)PyArray_DATA((PyArrayObject*)c_array);
    if (p > 0 && n > 0) memset(c_data, 0, ldc * n * sizeof(f64));

    npy_intp tau_dims[1] = {n > 0 ? n : 1};
    PyObject *tau_array = PyArray_SimpleNew(1, tau_dims, NPY_DOUBLE);
    if (tau_array == NULL) {
        Py_DECREF(c_array);
        Py_DECREF(l_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }
    f64 *tau = (f64*)PyArray_DATA((PyArrayObject*)tau_array);
    memset(tau, 0, (n > 0 ? n : 1) * sizeof(f64));

    f64 *dwork = (f64*)calloc(n > 0 ? n : 1, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(tau_array);
        Py_DECREF(c_array);
        Py_DECREF(l_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    f64 *l_data = (f64*)PyArray_DATA(l_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    mb04ld(uplo, n, m, p, l_data, ldl, a_data, lda, b_data, ldb,
           c_data, ldc, tau, dwork);

    PyArray_ResolveWritebackIfCopy(l_array);
    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    
    free(dwork);

    PyObject *result = Py_BuildValue("OOOOO", l_array, a_array, b_array, c_array, tau_array);
    Py_DECREF(l_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(tau_array);

    return result;
}

/* Python wrapper for mb04md - Balance a general real matrix */
PyObject* py_mb04md(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *a_obj;
    double maxred_in = 0.0;
    static char *kwlist[] = {"a", "maxred", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|d", kwlist,
                                     &a_obj, &maxred_in)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(a_array) != 2) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "A must be a 2-dimensional array");
        return NULL;
    }

    npy_intp *dims = PyArray_DIMS(a_array);
    i32 n = (i32)dims[0];

    if (dims[0] != dims[1]) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "A must be a square matrix");
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    
    npy_intp scale_dims[1] = {n > 0 ? n : 0};
    PyObject *scale_array = PyArray_SimpleNew(1, scale_dims, NPY_DOUBLE);
    if (scale_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }
    f64 *scale = (f64*)PyArray_DATA((PyArrayObject*)scale_array);
    if (n > 0) {
        memset(scale, 0, n * sizeof(f64));
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 maxred = maxred_in;
    i32 info = 0;

    mb04md(n, &maxred, a_data, lda, scale, &info);

    PyArray_ResolveWritebackIfCopy(a_array);

    if (info < 0) {
        Py_DECREF(scale_array);
        Py_DECREF(a_array);
        PyErr_Format(PyExc_ValueError, "MB04MD: illegal value in argument %d", -info);
        return NULL;
    }
    
    // scale_array already exists, no need to recreate or enable OWNDATA manually (SimpleNew handles it)

    PyObject *result = Py_BuildValue("OOdi", a_array, scale_array, maxred, info);
    Py_DECREF(a_array);
    Py_DECREF(scale_array);

    return result;
}

/**
 * Python wrapper for MB04PA - Reduce (skew-)Hamiltonian like matrix.
 *
 * Signature:
 *   a_out, qg_out, xa_out, xg_out, xq_out, ya_out, cs_out, tau_out = mb04pa(
 *       lham, n, k, nb, a, qg, xa, xg, xq, ya, cs, tau, dwork)
 */
PyObject* py_mb04pa(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static char* kwlist[] = {"lham", "n", "k", "nb", "a", "qg", "xa", "xg",
                             "xq", "ya", "cs", "tau", "dwork", NULL};
    int lham_in;
    i32 n, k, nb;
    PyObject *a_obj, *qg_obj, *xa_obj, *xg_obj, *xq_obj, *ya_obj;
    PyObject *cs_obj, *tau_obj, *dwork_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "piiiOOOOOOOOO", kwlist,
                                     &lham_in, &n, &k, &nb, &a_obj, &qg_obj,
                                     &xa_obj, &xg_obj, &xq_obj, &ya_obj,
                                     &cs_obj, &tau_obj, &dwork_obj)) {
        return NULL;
    }

    bool lham = (lham_in != 0);

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *qg_array = (PyArrayObject*)PyArray_FROM_OTF(
        qg_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *xa_array = (PyArrayObject*)PyArray_FROM_OTF(
        xa_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *xg_array = (PyArrayObject*)PyArray_FROM_OTF(
        xg_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *xq_array = (PyArrayObject*)PyArray_FROM_OTF(
        xq_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *ya_array = (PyArrayObject*)PyArray_FROM_OTF(
        ya_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *cs_array = (PyArrayObject*)PyArray_FROM_OTF(
        cs_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *tau_array = (PyArrayObject*)PyArray_FROM_OTF(
        tau_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *dwork_array = (PyArrayObject*)PyArray_FROM_OTF(
        dwork_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || qg_array == NULL || xa_array == NULL ||
        xg_array == NULL || xq_array == NULL || ya_array == NULL ||
        cs_array == NULL || tau_array == NULL || dwork_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(qg_array);
        Py_XDECREF(xa_array);
        Py_XDECREF(xg_array);
        Py_XDECREF(xq_array);
        Py_XDECREF(ya_array);
        Py_XDECREF(cs_array);
        Py_XDECREF(tau_array);
        Py_XDECREF(dwork_array);
        return NULL;
    }

    i32 lda = (i32)PyArray_DIM(a_array, 0);
    i32 ldqg = (i32)PyArray_DIM(qg_array, 0);
    i32 ldxa = (i32)PyArray_DIM(xa_array, 0);
    i32 ldxg = (i32)PyArray_DIM(xg_array, 0);
    i32 ldxq = (i32)PyArray_DIM(xq_array, 0);
    i32 ldya = (i32)PyArray_DIM(ya_array, 0);

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *qg_data = (f64*)PyArray_DATA(qg_array);
    f64 *xa_data = (f64*)PyArray_DATA(xa_array);
    f64 *xg_data = (f64*)PyArray_DATA(xg_array);
    f64 *xq_data = (f64*)PyArray_DATA(xq_array);
    f64 *ya_data = (f64*)PyArray_DATA(ya_array);
    f64 *cs_data = (f64*)PyArray_DATA(cs_array);
    f64 *tau_data = (f64*)PyArray_DATA(tau_array);
    f64 *dwork_data = (f64*)PyArray_DATA(dwork_array);

    mb04pa(lham, n, k, nb, a_data, lda, qg_data, ldqg,
           xa_data, ldxa, xg_data, ldxg, xq_data, ldxq,
           ya_data, ldya, cs_data, tau_data, dwork_data);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(qg_array);
    PyArray_ResolveWritebackIfCopy(xa_array);
    PyArray_ResolveWritebackIfCopy(xg_array);
    PyArray_ResolveWritebackIfCopy(xq_array);
    PyArray_ResolveWritebackIfCopy(ya_array);
    PyArray_ResolveWritebackIfCopy(cs_array);
    PyArray_ResolveWritebackIfCopy(tau_array);

    PyObject *result = Py_BuildValue("OOOOOOOO",
                                     a_array, qg_array, xa_array, xg_array,
                                     xq_array, ya_array, cs_array, tau_array);

    Py_DECREF(a_array);
    Py_DECREF(qg_array);
    Py_DECREF(xa_array);
    Py_DECREF(xg_array);
    Py_DECREF(xq_array);
    Py_DECREF(ya_array);
    Py_DECREF(cs_array);
    Py_DECREF(tau_array);
    Py_DECREF(dwork_array);

    return result;
}

PyObject* py_mb04pu(PyObject *self, PyObject *args)
{
    i32 n, ilo;
    PyObject *a_obj, *qg_obj;

    if (!PyArg_ParseTuple(args, "iiOO", &n, &ilo, &a_obj, &qg_obj)) {
        return NULL;
    }

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be >= 0");
        return NULL;
    }

    i32 max_1_n = (1 > n) ? 1 : n;
    if (ilo < 1 || ilo > max_1_n) {
        PyErr_SetString(PyExc_ValueError, "ilo must satisfy 1 <= ilo <= max(1,n)");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *qg_array = (PyArrayObject *)PyArray_FROM_OTF(
        qg_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || qg_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(qg_array);
        return NULL;
    }

    i32 lda = (n > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 ldqg = (n > 0) ? (i32)PyArray_DIM(qg_array, 0) : 1;

    i32 cs_size = (n > 1) ? 2 * n - 2 : 1;
    i32 tau_size = (n > 1) ? n - 1 : 1;
    // DLARF calls need workspace of size n (not n-1 as documented)
    i32 ldwork = (n > 1) ? n : 1;

    npy_intp cs_dims[1] = {cs_size};
    npy_intp tau_dims[1] = {tau_size};

    PyArrayObject *cs_array = (PyArrayObject *)PyArray_ZEROS(1, cs_dims, NPY_DOUBLE, 0);
    PyArrayObject *tau_array = (PyArrayObject *)PyArray_ZEROS(1, tau_dims, NPY_DOUBLE, 0);

    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));
    if (cs_array == NULL || tau_array == NULL || dwork == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(qg_array);
        Py_XDECREF(cs_array);
        Py_XDECREF(tau_array);
        free(dwork);
        return PyErr_NoMemory();
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *qg_data = (f64*)PyArray_DATA(qg_array);
    f64 *cs_data = (f64*)PyArray_DATA(cs_array);
    f64 *tau_data = (f64*)PyArray_DATA(tau_array);

    i32 info = 0;
    mb04pu(n, ilo, a_data, lda, qg_data, ldqg, cs_data, tau_data, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(qg_array);

    PyObject *result = Py_BuildValue("OOOOi", a_array, qg_array, cs_array, tau_array, info);

    Py_DECREF(a_array);
    Py_DECREF(qg_array);
    Py_DECREF(cs_array);
    Py_DECREF(tau_array);

    return result;
}

/* Python wrapper for mb04pb (blocked version) */
PyObject* py_mb04pb(PyObject *self, PyObject *args)
{
    i32 n, ilo;
    PyObject *a_obj, *qg_obj;

    if (!PyArg_ParseTuple(args, "iiOO", &n, &ilo, &a_obj, &qg_obj)) {
        return NULL;
    }

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be >= 0");
        return NULL;
    }

    i32 max_1_n = (1 > n) ? 1 : n;
    if (ilo < 1 || ilo > max_1_n) {
        PyErr_SetString(PyExc_ValueError, "ilo must satisfy 1 <= ilo <= max(1,n)");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *qg_array = (PyArrayObject *)PyArray_FROM_OTF(
        qg_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || qg_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(qg_array);
        return NULL;
    }

    i32 lda = (n > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 ldqg = (n > 0) ? (i32)PyArray_DIM(qg_array, 0) : 1;

    i32 cs_size = (n > 1) ? 2 * n - 2 : 1;
    i32 tau_size = (n > 1) ? n - 1 : 1;
    i32 ldwork = (n > 1) ? n : 1;

    npy_intp cs_dims[1] = {cs_size};
    npy_intp tau_dims[1] = {tau_size};

    PyArrayObject *cs_array = (PyArrayObject *)PyArray_ZEROS(1, cs_dims, NPY_DOUBLE, 0);
    PyArrayObject *tau_array = (PyArrayObject *)PyArray_ZEROS(1, tau_dims, NPY_DOUBLE, 0);

    i32 info = 0;
    f64 work_query;
    mb04pb(n, ilo, NULL, lda, NULL, ldqg, NULL, NULL, &work_query, -1, &info);
    ldwork = (i32)work_query;
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));
    if (cs_array == NULL || tau_array == NULL || dwork == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(qg_array);
        Py_XDECREF(cs_array);
        Py_XDECREF(tau_array);
        free(dwork);
        return PyErr_NoMemory();
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *qg_data = (f64*)PyArray_DATA(qg_array);
    f64 *cs_data = (f64*)PyArray_DATA(cs_array);
    f64 *tau_data = (f64*)PyArray_DATA(tau_array);

    info = 0;
    mb04pb(n, ilo, a_data, lda, qg_data, ldqg, cs_data, tau_data, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(qg_array);

    PyObject *result = Py_BuildValue("OOOOi", a_array, qg_array, cs_array, tau_array, info);

    Py_DECREF(a_array);
    Py_DECREF(qg_array);
    Py_DECREF(cs_array);
    Py_DECREF(tau_array);

    return result;
}

/* Python wrapper for mb04qs */
PyObject* py_mb04qs(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *tranc_str, *trand_str, *tranu_str;
    PyObject *c_obj, *d_obj, *v_obj, *w_obj, *cs_obj, *tau_obj;
    int m_arg = -1, n_arg = -1, ilo_arg = 1, ldwork_arg = 0;
    PyArrayObject *c_array = NULL, *d_array = NULL, *v_array = NULL, *w_array = NULL;
    PyArrayObject *cs_array = NULL, *tau_array = NULL;
    f64 *dwork = NULL;
    i32 info = 0;

    static char *kwlist[] = {"tranc", "trand", "tranu", "m", "n", "ilo",
                              "v", "w", "c", "d", "cs", "tau",
                              "ldwork", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssiiiOOOOOO|i", kwlist,
                                     &tranc_str, &trand_str, &tranu_str,
                                     &m_arg, &n_arg, &ilo_arg,
                                     &v_obj, &w_obj, &c_obj, &d_obj,
                                     &cs_obj, &tau_obj, &ldwork_arg)) {
        return NULL;
    }

    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    v_array = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    w_array = (PyArrayObject*)PyArray_FROM_OTF(w_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    cs_array = (PyArrayObject*)PyArray_FROM_OTF(cs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    tau_array = (PyArrayObject*)PyArray_FROM_OTF(tau_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (c_array == NULL || d_array == NULL || v_array == NULL || w_array == NULL ||
        cs_array == NULL || tau_array == NULL) {
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        Py_XDECREF(v_array);
        Py_XDECREF(w_array);
        Py_XDECREF(cs_array);
        Py_XDECREF(tau_array);
        return NULL;
    }

    npy_intp *c_dims = PyArray_DIMS(c_array);

    i32 m = (i32)m_arg;
    i32 n = (i32)n_arg;
    i32 ilo = (i32)ilo_arg;

    i32 ldc = (i32)c_dims[0];
    i32 ldd = (i32)PyArray_DIMS(d_array)[0];
    i32 ldv = (i32)PyArray_DIMS(v_array)[0];
    i32 ldw = (i32)PyArray_DIMS(w_array)[0];

    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    const f64 *v_data = (const f64*)PyArray_DATA(v_array);
    const f64 *w_data = (const f64*)PyArray_DATA(w_array);
    const f64 *cs_data = (const f64*)PyArray_DATA(cs_array);
    const f64 *tau_data = (const f64*)PyArray_DATA(tau_array);

    i32 ldwork;
    if (ldwork_arg == -1) {
        ldwork = -1;
        dwork = (f64*)malloc(sizeof(f64));
        if (dwork == NULL) {
            Py_DECREF(c_array);
            Py_DECREF(d_array);
            Py_DECREF(v_array);
            Py_DECREF(w_array);
            Py_DECREF(cs_array);
            Py_DECREF(tau_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
            return NULL;
        }
        mb04qs(tranc_str, trand_str, tranu_str, m, n, ilo,
               v_data, ldv, w_data, ldw,
               c_data, ldc, d_data, ldd, cs_data, tau_data,
               dwork, ldwork, &info);
    } else if (ldwork_arg > 0) {
        ldwork = (i32)ldwork_arg;
        dwork = (f64*)malloc(ldwork * sizeof(f64));
        if (dwork == NULL) {
            Py_DECREF(c_array);
            Py_DECREF(d_array);
            Py_DECREF(v_array);
            Py_DECREF(w_array);
            Py_DECREF(cs_array);
            Py_DECREF(tau_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
            return NULL;
        }
        mb04qs(tranc_str, trand_str, tranu_str, m, n, ilo,
               v_data, ldv, w_data, ldw,
               c_data, ldc, d_data, ldd, cs_data, tau_data,
               dwork, ldwork, &info);
    } else {
        f64 work_query;
        ldwork = -1;
        mb04qs(tranc_str, trand_str, tranu_str, m, n, ilo,
               v_data, ldv, w_data, ldw,
               c_data, ldc, d_data, ldd, cs_data, tau_data,
               &work_query, ldwork, &info);
        ldwork = (i32)work_query;
        if (ldwork < 1) ldwork = 1;
        dwork = (f64*)malloc(ldwork * sizeof(f64));
        if (dwork == NULL) {
            Py_DECREF(c_array);
            Py_DECREF(d_array);
            Py_DECREF(v_array);
            Py_DECREF(w_array);
            Py_DECREF(cs_array);
            Py_DECREF(tau_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
            return NULL;
        }
        mb04qs(tranc_str, trand_str, tranu_str, m, n, ilo,
               v_data, ldv, w_data, ldw,
               c_data, ldc, d_data, ldd, cs_data, tau_data,
               dwork, ldwork, &info);
    }

    free(dwork);

    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    PyObject *result = Py_BuildValue("OOi", c_array, d_array, info);

    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(v_array);
    Py_DECREF(w_array);
    Py_DECREF(cs_array);
    Py_DECREF(tau_array);

    return result;
}

/* Python wrapper for mb04rs */
PyObject* py_mb04rs(PyObject* self, PyObject* args) {
    i32 m, n;
    f64 pmax;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *e_obj, *f_obj;

    if (!PyArg_ParseTuple(args, "iidOOOOOO",
                          &m, &n, &pmax,
                          &a_obj, &b_obj, &c_obj, &d_obj, &e_obj, &f_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }

    PyArrayObject *f_array = (PyArrayObject*)PyArray_FROM_OTF(f_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (f_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(e_array);
        return NULL;
    }

    i32 lda = (m > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 ldb = (n > 0) ? (i32)PyArray_DIM(b_array, 0) : 1;
    i32 ldc = (m > 0) ? (i32)PyArray_DIM(c_array, 0) : 1;
    i32 ldd = (m > 0) ? (i32)PyArray_DIM(d_array, 0) : 1;
    i32 lde = (n > 0) ? (i32)PyArray_DIM(e_array, 0) : 1;
    i32 ldf = (m > 0) ? (i32)PyArray_DIM(f_array, 0) : 1;

    const f64 *a_data = (const f64*)PyArray_DATA(a_array);
    const f64 *b_data = (const f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    const f64 *d_data = (const f64*)PyArray_DATA(d_array);
    const f64 *e_data = (const f64*)PyArray_DATA(e_array);
    f64 *f_data = (f64*)PyArray_DATA(f_array);

    i32 liwork = m + n + 2;
    i32 *iwork = (i32*)PyMem_Calloc(liwork > 0 ? liwork : 1, sizeof(i32));
    if (iwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(e_array);
        Py_DECREF(f_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    f64 scale;
    i32 info;

    mb04rs(m, n, pmax, a_data, lda, b_data, ldb, c_data, ldc,
           d_data, ldd, e_data, lde, f_data, ldf, &scale, iwork, &info);

    PyMem_Free(iwork);

    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(f_array);

    PyObject *result = Py_BuildValue("OOdi", c_array, f_array, scale, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(e_array);
    Py_DECREF(f_array);

    return result;
}

/* Python wrapper for mb04ru */
PyObject* py_mb04ru(PyObject* self, PyObject* args) {
    i32 n, ilo;
    PyObject *a_obj, *qg_obj;

    if (!PyArg_ParseTuple(args, "iiOO", &n, &ilo, &a_obj, &qg_obj)) {
        return NULL;
    }

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be >= 0");
        return NULL;
    }

    if (ilo < 1 || (n > 0 && ilo > n + 1) || (n == 0 && ilo != 1)) {
        PyErr_SetString(PyExc_ValueError, "ilo out of valid range");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    PyArrayObject *qg_array = (PyArrayObject*)PyArray_FROM_OTF(
        qg_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (qg_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 lda = (n > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 ldqg = (n > 0) ? (i32)PyArray_DIM(qg_array, 0) : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *qg_data = (f64*)PyArray_DATA(qg_array);

    npy_intp cs_dim = (n > 1) ? 2 * (n - 1) : 0;
    npy_intp tau_dim = (n > 1) ? n - 1 : 0;

    PyArrayObject *cs_array = NULL;
    PyArrayObject *tau_array = NULL;
    f64 *cs_data = NULL;
    f64 *tau_data = NULL;

    if (cs_dim > 0) {
        cs_array = (PyArrayObject*)PyArray_ZEROS(1, &cs_dim, NPY_DOUBLE, 0);
        if (cs_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(qg_array);
            return NULL;
        }
        cs_data = (f64*)PyArray_DATA(cs_array);
    } else {
        npy_intp zero_dim = 0;
        cs_array = (PyArrayObject*)PyArray_ZEROS(1, &zero_dim, NPY_DOUBLE, 0);
        if (cs_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(qg_array);
            return NULL;
        }
    }

    if (tau_dim > 0) {
        tau_array = (PyArrayObject*)PyArray_ZEROS(1, &tau_dim, NPY_DOUBLE, 0);
        if (tau_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(qg_array);
            Py_DECREF(cs_array);
            return NULL;
        }
        tau_data = (f64*)PyArray_DATA(tau_array);
    } else {
        npy_intp zero_dim = 0;
        tau_array = (PyArrayObject*)PyArray_ZEROS(1, &zero_dim, NPY_DOUBLE, 0);
        if (tau_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(qg_array);
            Py_DECREF(cs_array);
            return NULL;
        }
    }

    i32 ldwork = (n > 1) ? n : 1;
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(qg_array);
        Py_DECREF(cs_array);
        Py_DECREF(tau_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    i32 info;
    mb04ru(n, ilo, a_data, lda, qg_data, ldqg, cs_data, tau_data, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(qg_array);

    PyObject *result = Py_BuildValue("OOOOi", a_array, qg_array, cs_array, tau_array, info);

    Py_DECREF(a_array);
    Py_DECREF(qg_array);
    Py_DECREF(cs_array);
    Py_DECREF(tau_array);

    return result;
}

/* Python wrapper for mb04rb */
PyObject* py_mb04rb(PyObject* self, PyObject* args) {
    i32 n, ilo;
    PyObject *a_obj, *qg_obj;

    if (!PyArg_ParseTuple(args, "iiOO", &n, &ilo, &a_obj, &qg_obj)) {
        return NULL;
    }

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be >= 0");
        return NULL;
    }

    if (ilo < 1 || (n > 0 && ilo > n + 1) || (n == 0 && ilo != 1)) {
        PyErr_SetString(PyExc_ValueError, "ilo out of valid range");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    PyArrayObject *qg_array = (PyArrayObject*)PyArray_FROM_OTF(
        qg_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (qg_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 lda = (n > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 ldqg = (n > 0) ? (i32)PyArray_DIM(qg_array, 0) : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *qg_data = (f64*)PyArray_DATA(qg_array);

    npy_intp cs_dim = (n > 1) ? 2 * (n - 1) : 0;
    npy_intp tau_dim = (n > 1) ? n - 1 : 0;

    PyArrayObject *cs_array = NULL;
    PyArrayObject *tau_array = NULL;
    f64 *cs_data = NULL;
    f64 *tau_data = NULL;

    if (cs_dim > 0) {
        cs_array = (PyArrayObject*)PyArray_ZEROS(1, &cs_dim, NPY_DOUBLE, 0);
        if (cs_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(qg_array);
            return NULL;
        }
        cs_data = (f64*)PyArray_DATA(cs_array);
    } else {
        npy_intp zero_dim = 0;
        cs_array = (PyArrayObject*)PyArray_ZEROS(1, &zero_dim, NPY_DOUBLE, 0);
        if (cs_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(qg_array);
            return NULL;
        }
    }

    if (tau_dim > 0) {
        tau_array = (PyArrayObject*)PyArray_ZEROS(1, &tau_dim, NPY_DOUBLE, 0);
        if (tau_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(qg_array);
            Py_DECREF(cs_array);
            return NULL;
        }
        tau_data = (f64*)PyArray_DATA(tau_array);
    } else {
        npy_intp zero_dim = 0;
        tau_array = (PyArrayObject*)PyArray_ZEROS(1, &zero_dim, NPY_DOUBLE, 0);
        if (tau_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(qg_array);
            Py_DECREF(cs_array);
            return NULL;
        }
    }

    i32 ldwork_query = -1;
    f64 work_query;
    i32 info;
    mb04rb(n, ilo, a_data, lda, qg_data, ldqg, cs_data, tau_data, &work_query, ldwork_query, &info);
    i32 ldwork = (i32)work_query;
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(qg_array);
        Py_DECREF(cs_array);
        Py_DECREF(tau_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    mb04rb(n, ilo, a_data, lda, qg_data, ldqg, cs_data, tau_data, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(qg_array);

    PyObject *result = Py_BuildValue("OOOOi", a_array, qg_array, cs_array, tau_array, info);

    Py_DECREF(a_array);
    Py_DECREF(qg_array);
    Py_DECREF(cs_array);
    Py_DECREF(tau_array);

    return result;
}

PyObject* py_mb04bz(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"job", "compq", "a", "de", "b", "fg", NULL};

    const char *job, *compq;
    PyObject *a_obj, *de_obj, *b_obj, *fg_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssOOOO", kwlist,
                                     &job, &compq, &a_obj, &de_obj, &b_obj, &fg_obj)) {
        return NULL;
    }

    char job_upper = (char)toupper((unsigned char)job[0]);
    char compq_upper = (char)toupper((unsigned char)compq[0]);

    bool ltri = (job_upper == 'T');
    bool lcmpq = (compq_upper == 'C');

    PyArrayObject *a_in_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);
    if (a_in_array == NULL) return NULL;

    i32 m = (i32)PyArray_DIM(a_in_array, 0);
    i32 n = 2 * m;
    i32 n2 = 2 * n;

    PyArrayObject *de_in_array = (PyArrayObject *)PyArray_FROM_OTF(
        de_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);
    if (de_in_array == NULL) {
        Py_DECREF(a_in_array);
        return NULL;
    }

    PyArrayObject *b_in_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);
    if (b_in_array == NULL) {
        Py_DECREF(de_in_array);
        Py_DECREF(a_in_array);
        return NULL;
    }

    PyArrayObject *fg_in_array = (PyArrayObject *)PyArray_FROM_OTF(
        fg_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);
    if (fg_in_array == NULL) {
        Py_DECREF(b_in_array);
        Py_DECREF(de_in_array);
        Py_DECREF(a_in_array);
        return NULL;
    }

    c128 *a_in_data = (c128 *)PyArray_DATA(a_in_array);
    c128 *de_in_data = (c128 *)PyArray_DATA(de_in_array);
    c128 *b_in_data = (c128 *)PyArray_DATA(b_in_array);
    c128 *fg_in_data = (c128 *)PyArray_DATA(fg_in_array);

    i32 lda, ldde, ldb, ldfg, k_dim;
    c128 *a_data = NULL, *de_data = NULL, *b_data = NULL, *fg_data = NULL;

    // Safety refactor: Use PyArray_SimpleNew logic
    PyObject *a_out = NULL, *de_out = NULL, *b_out = NULL, *fg_out = NULL;
    PyObject *q_out = NULL;

    // JOB='E' uses same compact input format as JOB='T'
    bool compact_input = (job_upper == 'T' || job_upper == 'E');

    if (compact_input) {
        k_dim = (n > 1) ? n : 1;
        lda = k_dim;
        ldde = k_dim;
        ldb = k_dim;
        ldfg = k_dim;
    } else {
        k_dim = (n > 1) ? n : 1;
        lda = k_dim;
        ldde = k_dim;
        ldb = k_dim;
        ldfg = k_dim;
    }

    npy_intp dims_n_n[2] = {lda, n > 0 ? n : 1}; // Use calculated lda (which is n usually)
    a_out = PyArray_SimpleNew(2, dims_n_n, NPY_COMPLEX128);
    de_out = PyArray_SimpleNew(2, dims_n_n, NPY_COMPLEX128);
    b_out = PyArray_SimpleNew(2, dims_n_n, NPY_COMPLEX128);
    fg_out = PyArray_SimpleNew(2, dims_n_n, NPY_COMPLEX128);

    if (!a_out || !de_out || !b_out || !fg_out) {
        Py_XDECREF(a_out); Py_XDECREF(de_out); Py_XDECREF(b_out); Py_XDECREF(fg_out);
        Py_DECREF(fg_in_array); Py_DECREF(b_in_array);
        Py_DECREF(de_in_array); Py_DECREF(a_in_array);
        return NULL;
    }

    a_data = (c128 *)PyArray_DATA((PyArrayObject *)a_out);
    de_data = (c128 *)PyArray_DATA((PyArrayObject *)de_out);
    b_data = (c128 *)PyArray_DATA((PyArrayObject *)b_out);
    fg_data = (c128 *)PyArray_DATA((PyArrayObject *)fg_out);
    
    // Zero initialize
    memset(a_data, 0, PyArray_NBYTES((PyArrayObject*)a_out));
    memset(de_data, 0, PyArray_NBYTES((PyArrayObject*)de_out));
    memset(b_data, 0, PyArray_NBYTES((PyArrayObject*)b_out));
    memset(fg_data, 0, PyArray_NBYTES((PyArrayObject*)fg_out));

    if (compact_input) {
        i32 a_in_lda = (i32)PyArray_DIM(a_in_array, 0);
        for (i32 j = 0; j < m; j++) {
            for (i32 i = 0; i < m; i++) {
                a_data[i + j * lda] = a_in_data[i + j * a_in_lda];
            }
        }
        i32 de_in_ldde = (i32)PyArray_DIM(de_in_array, 0);
        i32 de_in_cols = (i32)PyArray_DIM(de_in_array, 1);
        for (i32 j = 0; j < de_in_cols && j < n; j++) {
            for (i32 i = 0; i < m && i < ldde; i++) {
                de_data[i + j * ldde] = de_in_data[i + j * de_in_ldde];
            }
        }
        i32 b_in_ldb = (i32)PyArray_DIM(b_in_array, 0);
        for (i32 j = 0; j < m; j++) {
            for (i32 i = 0; i < m; i++) {
                b_data[i + j * ldb] = b_in_data[i + j * b_in_ldb];
            }
        }
        i32 fg_in_ldfg = (i32)PyArray_DIM(fg_in_array, 0);
        i32 fg_in_cols = (i32)PyArray_DIM(fg_in_array, 1);
        for (i32 j = 0; j < fg_in_cols && j < n; j++) {
            for (i32 i = 0; i < m && i < ldfg; i++) {
                fg_data[i + j * ldfg] = fg_in_data[i + j * fg_in_ldfg];
            }
        }
    } else {
        // Validate dimensions
        i32 a_in_lda = (i32)PyArray_DIM(a_in_array, 0);
        i32 a_in_cols = (i32)PyArray_DIM(a_in_array, 1);
        if (a_in_lda < n || a_in_cols < n) {
             Py_DECREF(a_out); Py_DECREF(de_out); Py_DECREF(b_out); Py_DECREF(fg_out);
             Py_DECREF(fg_in_array); Py_DECREF(b_in_array);
             Py_DECREF(de_in_array); Py_DECREF(a_in_array);
             PyErr_Format(PyExc_ValueError, "Input A dimensions must be at least (%d, %d) for JOB!=T", n, n);
             return NULL;
        }
        
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < n; i++) {
                a_data[i + j * lda] = a_in_data[i + j * a_in_lda];
            }
        }
        i32 de_in_ldde = (i32)PyArray_DIM(de_in_array, 0);
        i32 de_in_cols = (i32)PyArray_DIM(de_in_array, 1);
        for (i32 j = 0; j < de_in_cols && j < n; j++) {
            for (i32 i = 0; i < n && i < ldde; i++) {
                de_data[i + j * ldde] = de_in_data[i + j * de_in_ldde];
            }
        }
        i32 b_in_ldb = (i32)PyArray_DIM(b_in_array, 0);
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < n; i++) {
                b_data[i + j * ldb] = b_in_data[i + j * b_in_ldb];
            }
        }
        i32 fg_in_ldfg = (i32)PyArray_DIM(fg_in_array, 0);
        i32 fg_in_cols = (i32)PyArray_DIM(fg_in_array, 1);
        for (i32 j = 0; j < fg_in_cols && j < n; j++) {
            for (i32 i = 0; i < n && i < ldfg; i++) {
                fg_data[i + j * ldfg] = fg_in_data[i + j * fg_in_ldfg];
            }
        }
    }

    i32 ldq = lcmpq ? (n2 > 1 ? n2 : 1) : 1;
    c128 *q_data = NULL;
    c128 dummy_q;

    if (lcmpq) {
        npy_intp dims_q[2] = {ldq, n2 > 0 ? n2 : 1};
        q_out = PyArray_SimpleNew(2, dims_q, NPY_COMPLEX128);
        if (!q_out) {
             Py_DECREF(a_out); Py_DECREF(de_out); Py_DECREF(b_out); Py_DECREF(fg_out);
             Py_DECREF(fg_in_array); Py_DECREF(b_in_array);
             Py_DECREF(de_in_array); Py_DECREF(a_in_array);
             return NULL;
        }
        q_data = (c128 *)PyArray_DATA((PyArrayObject *)q_out);
        memset(q_data, 0, PyArray_NBYTES((PyArrayObject *)q_out));
    } else {
        q_data = &dummy_q;
        q_out = Py_None;
        Py_INCREF(q_out);
    }

    npy_intp n_dim[1] = {n};  // n=0 gives empty arrays
    PyObject *alphar_array = PyArray_SimpleNew(1, n_dim, NPY_DOUBLE);
    PyObject *alphai_array = PyArray_SimpleNew(1, n_dim, NPY_DOUBLE);
    PyObject *beta_py = PyArray_SimpleNew(1, n_dim, NPY_DOUBLE);

    if (!alphar_array || !alphai_array || !beta_py) {
        Py_XDECREF(alphar_array); Py_XDECREF(alphai_array); Py_XDECREF(beta_py);
        if (q_out) Py_DECREF(q_out);
        Py_XDECREF(a_out); Py_XDECREF(de_out);
        Py_XDECREF(b_out); Py_XDECREF(fg_out);
        Py_DECREF(fg_in_array); Py_DECREF(b_in_array);
        Py_DECREF(de_in_array); Py_DECREF(a_in_array);
        return NULL;
    }

    f64 *alphar = (f64 *)PyArray_DATA((PyArrayObject *)alphar_array);
    f64 *alphai = (f64 *)PyArray_DATA((PyArrayObject *)alphai_array);
    f64 *beta_arr = (f64 *)PyArray_DATA((PyArrayObject *)beta_py);
    // Initialize to zero? Not strictly necessary as output, but safe.
    memset(alphar, 0, PyArray_NBYTES((PyArrayObject *)alphar_array));
    memset(alphai, 0, PyArray_NBYTES((PyArrayObject *)alphai_array));
    memset(beta_arr, 0, PyArray_NBYTES((PyArrayObject *)beta_py));

    i32 liwork = 2 * n + 4;
    i32 *iwork = (i32 *)calloc(liwork > 1 ? liwork : 1, sizeof(i32));
    if (iwork == NULL) {
        Py_XDECREF(alphar_array); Py_XDECREF(alphai_array); Py_XDECREF(beta_py);
        Py_XDECREF(q_out);
        Py_XDECREF(a_out); Py_XDECREF(de_out);
        Py_XDECREF(b_out); Py_XDECREF(fg_out);
        Py_DECREF(fg_in_array); Py_DECREF(b_in_array);
        Py_DECREF(de_in_array); Py_DECREF(a_in_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 nn = n * n;
    i32 mindw, minzw;
    if (n == 0) {
        mindw = 3;
        minzw = 1;
    } else if (lcmpq) {
        mindw = 11 * nn + n2;
        minzw = 8 * n + 4;
    } else {
        if (ltri) {
            mindw = 5 * nn + 3 * n;
        } else {
            mindw = 4 * nn + 3 * n;
        }
        minzw = ltri ? (6 * n + 4) : 1;
    }

    i32 ldwork = mindw > 3 ? mindw : 3;
    i32 lzwork = minzw > 1 ? minzw : 1;

    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));
    c128 *zwork = (c128 *)PyMem_Calloc(lzwork, sizeof(c128));
    if (dwork == NULL || zwork == NULL) {
        free(dwork); PyMem_Free(zwork);
        free(iwork);
        Py_XDECREF(alphar_array); Py_XDECREF(alphai_array); Py_XDECREF(beta_py);
        Py_XDECREF(q_out);
        Py_XDECREF(a_out); Py_XDECREF(de_out);
        Py_XDECREF(b_out); Py_XDECREF(fg_out);
        Py_DECREF(fg_in_array); Py_DECREF(b_in_array);
        Py_DECREF(de_in_array); Py_DECREF(a_in_array);
        PyErr_NoMemory();
        return NULL;
    }

    bool *bwork = NULL;
    if (ltri && n > 0) {
        bwork = (bool *)calloc(n, sizeof(bool));
        if (bwork == NULL) {
            free(dwork); PyMem_Free(zwork);
            free(iwork);
            Py_XDECREF(alphar_array); Py_XDECREF(alphai_array); Py_XDECREF(beta_py);
            Py_XDECREF(q_out);
            Py_XDECREF(a_out); Py_XDECREF(de_out);
            Py_XDECREF(b_out); Py_XDECREF(fg_out);
            Py_DECREF(fg_in_array); Py_DECREF(b_in_array);
            Py_DECREF(de_in_array); Py_DECREF(a_in_array);
            PyErr_NoMemory();
            return NULL;
        }
    }

    i32 info;
    mb04bz(job, compq, n, a_data, lda, de_data, ldde, b_data, ldb, fg_data, ldfg,
           q_data, ldq, alphar, alphai, beta_arr, iwork, dwork, ldwork,
           zwork, lzwork, bwork, &info);

    free(dwork);
    PyMem_Free(zwork);
    free(iwork);
    if (bwork) free(bwork);

    Py_DECREF(fg_in_array);
    Py_DECREF(b_in_array);
    Py_DECREF(de_in_array);
    Py_DECREF(a_in_array);

    if (info < 0) {
        Py_XDECREF(alphar_array); Py_XDECREF(alphai_array); Py_XDECREF(beta_py);
        Py_XDECREF(q_out);
        Py_XDECREF(a_out); Py_XDECREF(de_out);
        Py_XDECREF(b_out); Py_XDECREF(fg_out);
        PyErr_Format(PyExc_ValueError, "mb04bz: parameter %d is invalid", -info);
        return NULL;
    }



    PyObject *result;
    if (ltri) {
        // JOB='T': Return full Schur form + eigenvalues
        result = Py_BuildValue("OOOOOOOOi", a_out, de_out, b_out, fg_out, q_out,
                               alphar_array, alphai_array, beta_py, info);
    } else if (job_upper == 'E') {
        // JOB='E': Return only eigenvalues
        result = Py_BuildValue("OOOi", alphar_array, alphai_array, beta_py, info);
    } else {
        // Default: Return full output
        result = Py_BuildValue("OOOOOOOOi", a_out, de_out, b_out, fg_out, q_out,
                               alphar_array, alphai_array, beta_py, info);
    }

    // Decree references as Py_BuildValue increments them
    Py_DECREF(a_out);
    Py_DECREF(de_out);
    Py_DECREF(b_out);
    Py_DECREF(fg_out);
    Py_DECREF(q_out);

    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_py);

    return result;
}

PyObject* py_mb04fp(PyObject* self, PyObject* args, PyObject* kwargs)
{
    const char *job;
    const char *compq;
    PyObject *a_obj, *de_obj, *b_obj, *fg_obj;
    PyObject *q_obj = Py_None;
    PyObject *n_obj = Py_None;

    static char *kwlist[] = {"job", "compq", "a", "de", "b", "fg", "q", "n", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssOOOO|OO", kwlist,
                                     &job, &compq, &a_obj, &de_obj, &b_obj, &fg_obj,
                                     &q_obj, &n_obj)) {
        return NULL;
    }

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

    char compq_upper = (char)toupper((unsigned char)compq[0]);

    i32 m = (i32)PyArray_DIM(a_array, 0);
    i32 n;

    if (n_obj != Py_None) {
        n = (i32)PyLong_AsLong(n_obj);
    } else {
        n = 2 * m;
    }

    i32 lda = (i32)PyArray_DIM(a_array, 0);
    i32 ldde = (i32)PyArray_DIM(de_array, 0);
    i32 ldb = (i32)PyArray_DIM(b_array, 0);
    i32 ldfg = (i32)PyArray_DIM(fg_array, 0);

    if (lda < 1) lda = 1;
    if (ldde < 1) ldde = 1;
    if (ldb < 1) ldb = 1;
    if (ldfg < 1) ldfg = 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *de_data = (f64*)PyArray_DATA(de_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *fg_data = (f64*)PyArray_DATA(fg_array);

    bool lcmpq = (compq_upper == 'I' || compq_upper == 'U');
    i32 ldq = lcmpq ? n : 1;
    if (ldq < 1) ldq = 1;

    PyArrayObject *q_array = NULL;
    f64 *q_data = NULL;

    if (compq_upper == 'U') {
        q_array = (PyArrayObject*)PyArray_FROM_OTF(
            q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!q_array) {
            Py_DECREF(a_array);
            Py_DECREF(de_array);
            Py_DECREF(b_array);
            Py_DECREF(fg_array);
            return NULL;
        }
        q_data = (f64*)PyArray_DATA(q_array);
    } else if (lcmpq) {
        npy_intp q_dims[2] = {n, n};
        npy_intp q_strides[2] = {sizeof(f64), n * sizeof(f64)};
        q_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE,
                                              q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!q_array) {
            Py_DECREF(a_array);
            Py_DECREF(de_array);
            Py_DECREF(b_array);
            Py_DECREF(fg_array);
            PyErr_NoMemory();
            return NULL;
        }
        q_data = (f64*)PyArray_DATA(q_array);
    } else {
        npy_intp q_dims[2] = {1, 1};
        q_array = (PyArrayObject*)PyArray_ZEROS(2, q_dims, NPY_DOUBLE, 1);
        q_data = (f64*)PyArray_DATA(q_array);
    }

    i32 m_out = n / 2;
    if (m_out < 0) m_out = 0;

    npy_intp eig_dim = m_out;
    PyArrayObject *alphar_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 1, &eig_dim, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
    PyArrayObject *alphai_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 1, &eig_dim, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
    PyArrayObject *beta_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 1, &eig_dim, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);

    if (!alphar_array || !alphai_array || !beta_array) {
        Py_XDECREF(alphar_array);
        Py_XDECREF(alphai_array);
        Py_XDECREF(beta_array);
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(b_array);
        Py_DECREF(fg_array);
        Py_DECREF(q_array);
        PyErr_NoMemory();
        return NULL;
    }
    f64 *alphar_data = (f64*)PyArray_DATA(alphar_array);
    f64 *alphai_data = (f64*)PyArray_DATA(alphai_array);
    f64 *beta_data = (f64*)PyArray_DATA(beta_array);

    i32 mm = m_out * m_out;
    i32 mindw;

    if (lcmpq) {
        mindw = 3 > 2 * mm + mm - 1 ? 3 : 2 * mm + mm - 1;
    } else if (job[0] == 'T' || job[0] == 't') {
        mindw = 3 > mm + m_out - 1 ? 3 : mm + m_out - 1;
    } else {
        i32 temp = 2 * n - 6;
        mindw = 3 > m_out ? 3 : m_out;
        mindw = mindw > temp ? mindw : temp;
    }

    i32 ldwork = mindw + m_out * m_out;
    if (lcmpq && ldwork < n * n) {
        ldwork = n * n;
    }
    i32 liwork = m_out + 2;

    i32 *iwork = (i32*)calloc(liwork, sizeof(i32));
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        Py_DECREF(alphar_array);
        Py_DECREF(alphai_array);
        Py_DECREF(beta_array);
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(b_array);
        Py_DECREF(fg_array);
        Py_DECREF(q_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info = 0;

    mb04fp(job, compq, n, a_data, lda, de_data, ldde, b_data, ldb, fg_data, ldfg,
           q_data, ldq, alphar_data, alphai_data, beta_data,
           iwork, dwork, ldwork, &info);

    npy_intp iwork_dim = m_out + 1;
    PyArrayObject *iwork_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 1, &iwork_dim, NPY_INT32, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
    if (iwork_array) {
        i32 *iwork_out = (i32*)PyArray_DATA(iwork_array);
        for (i32 ii = 0; ii <= m_out; ii++) {
            iwork_out[ii] = iwork[ii];
        }
    }

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(de_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(fg_array);

    if (compq_upper == 'U') {
        PyArray_ResolveWritebackIfCopy(q_array);
    }

    PyObject *result_fp = Py_BuildValue("(OOOOOOOOOi)",
                                     a_array, de_array, b_array, fg_array, q_array,
                                     alphar_array, alphai_array, beta_array,
                                     iwork_array, info);

    Py_DECREF(a_array);
    Py_DECREF(de_array);
    Py_DECREF(b_array);
    Py_DECREF(fg_array);
    Py_DECREF(q_array);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);
    Py_XDECREF(iwork_array);

    return result_fp;
}

PyObject* py_mb04tu(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *x_obj = NULL, *y_obj = NULL;
    double c, s;
    int n = -1, incx = 1, incy = 1;

    static char *kwlist[] = {"x", "y", "c", "s", "n", "incx", "incy", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOdd|iii", kwlist,
                                     &x_obj, &y_obj, &c, &s, &n, &incx, &incy)) {
        return NULL;
    }

    PyArrayObject *x_array = (PyArrayObject*)PyArray_FROM_OTF(
        x_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *y_array = (PyArrayObject*)PyArray_FROM_OTF(
        y_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!x_array || !y_array) {
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        return NULL;
    }

    if (n < 0) {
        npy_intp x_size = PyArray_SIZE(x_array);
        npy_intp y_size = PyArray_SIZE(y_array);
        i32 abs_incx = incx > 0 ? incx : -incx;
        i32 abs_incy = incy > 0 ? incy : -incy;
        i32 n_from_x = abs_incx > 0 ? (i32)((x_size - 1) / abs_incx + 1) : (i32)x_size;
        i32 n_from_y = abs_incy > 0 ? (i32)((y_size - 1) / abs_incy + 1) : (i32)y_size;
        n = n_from_x < n_from_y ? n_from_x : n_from_y;
    }

    f64 *x_data = (f64*)PyArray_DATA(x_array);
    f64 *y_data = (f64*)PyArray_DATA(y_array);

    mb04tu((i32)n, x_data, (i32)incx, y_data, (i32)incy, (f64)c, (f64)s);

    PyArray_ResolveWritebackIfCopy(x_array);
    PyArray_ResolveWritebackIfCopy(y_array);

    PyObject *result = Py_BuildValue("(OO)", x_array, y_array);
    Py_DECREF(x_array);
    Py_DECREF(y_array);

    return result;
}

PyObject* py_mb04tv(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *a_obj = NULL, *e_obj = NULL, *z_obj = NULL;
    int n, nra, nca, ifira, ifica;
    int updatz = 0;  // Default: don't update Z

    static char *kwlist[] = {"n", "nra", "nca", "ifira", "ifica", "a", "e", "z",
                             "updatz", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiiiiOOO|p", kwlist,
                                     &n, &nra, &nca, &ifira, &ifica,
                                     &a_obj, &e_obj, &z_obj, &updatz)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(
        e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *z_array = (PyArrayObject*)PyArray_FROM_OTF(
        z_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!a_array || !e_array || !z_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(z_array);
        return NULL;
    }

    i32 lda = (i32)PyArray_DIM(a_array, 0);
    i32 lde = PyArray_NDIM(e_array) >= 1 ? (i32)PyArray_DIM(e_array, 0) : 1;
    i32 ldz = (i32)PyArray_DIM(z_array, 0);

    if (lde == 0) {
        lde = 1;  // Minimum leading dimension
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *z_data = (f64*)PyArray_DATA(z_array);

    i32 info = 0;
    mb04tv((bool)updatz, (i32)n, (i32)nra, (i32)nca, (i32)ifira, (i32)ifica,
           a_data, lda, e_data, lde, z_data, ldz, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(z_array);

    PyObject *result = Py_BuildValue("(OOOi)", a_array, e_array, z_array, info);
    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(z_array);

    return result;
}

PyObject* py_mb04tw(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *a_obj = NULL, *e_obj = NULL, *q_obj = NULL;
    int m, n, nre, nce, ifire, ifice, ifica;
    int updatq = 0;

    static char *kwlist[] = {"m", "n", "nre", "nce", "ifire", "ifice", "ifica",
                             "a", "e", "q", "updatq", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiiiiiiOOO|p", kwlist,
                                     &m, &n, &nre, &nce, &ifire, &ifice, &ifica,
                                     &a_obj, &e_obj, &q_obj, &updatq)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(
        e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *q_array = (PyArrayObject*)PyArray_FROM_OTF(
        q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!a_array || !e_array || !q_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(q_array);
        return NULL;
    }

    i32 lda = (i32)PyArray_DIM(a_array, 0);
    i32 lde = (i32)PyArray_DIM(e_array, 0);
    i32 ldq = (i32)PyArray_DIM(q_array, 0);

    if (lda == 0) lda = 1;
    if (lde == 0) lde = 1;
    if (ldq == 0) ldq = 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *q_data = (f64*)PyArray_DATA(q_array);

    i32 info = 0;
    mb04tw((bool)updatq, (i32)m, (i32)n, (i32)nre, (i32)nce, (i32)ifire,
           (i32)ifice, (i32)ifica, a_data, lda, e_data, lde, q_data, ldq, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(q_array);

    PyObject *result = Py_BuildValue("(OOOi)", a_array, e_array, q_array, info);
    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(q_array);

    return result;
}

PyObject* py_mb04ty(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *inuk_obj = NULL, *imuk_obj = NULL;
    PyObject *a_obj = NULL, *e_obj = NULL, *q_obj = NULL, *z_obj = NULL;
    int m, n, nblcks;
    int updatq = 0, updatz = 0;

    static char *kwlist[] = {"m", "n", "nblcks", "inuk", "imuk", "a", "e", "q", "z",
                             "updatq", "updatz", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiiOOOOOO|pp", kwlist,
                                     &m, &n, &nblcks, &inuk_obj, &imuk_obj,
                                     &a_obj, &e_obj, &q_obj, &z_obj,
                                     &updatq, &updatz)) {
        return NULL;
    }

    PyArrayObject *inuk_array = (PyArrayObject*)PyArray_FROM_OTF(
        inuk_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    PyArrayObject *imuk_array = (PyArrayObject*)PyArray_FROM_OTF(
        imuk_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(
        e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *q_array = (PyArrayObject*)PyArray_FROM_OTF(
        q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *z_array = (PyArrayObject*)PyArray_FROM_OTF(
        z_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!inuk_array || !imuk_array || !a_array || !e_array || !q_array || !z_array) {
        Py_XDECREF(inuk_array);
        Py_XDECREF(imuk_array);
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(q_array);
        Py_XDECREF(z_array);
        return NULL;
    }

    i32 lda = (m > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 lde = (m > 0) ? (i32)PyArray_DIM(e_array, 0) : 1;
    i32 ldq = (m > 0) ? (i32)PyArray_DIM(q_array, 0) : 1;
    i32 ldz = (n > 0) ? (i32)PyArray_DIM(z_array, 0) : 1;

    i32 *inuk_data = (i32*)PyArray_DATA(inuk_array);
    i32 *imuk_data = (i32*)PyArray_DATA(imuk_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *q_data = (f64*)PyArray_DATA(q_array);
    f64 *z_data = (f64*)PyArray_DATA(z_array);

    i32 info = 0;
    mb04ty((bool)updatq, (bool)updatz, (i32)m, (i32)n, (i32)nblcks,
           inuk_data, imuk_data, a_data, lda, e_data, lde,
           q_data, ldq, z_data, ldz, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(q_array);
    PyArray_ResolveWritebackIfCopy(z_array);

    PyObject *result = Py_BuildValue("(OOOOi)", a_array, e_array, q_array, z_array, info);
    Py_DECREF(inuk_array);
    Py_DECREF(imuk_array);
    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);

    return result;
}

PyObject* py_mb04su(PyObject *self, PyObject *args)
{
    i32 m, n;
    PyObject *a_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "iiOO", &m, &n, &a_obj, &b_obj)) {
        return NULL;
    }

    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be >= 0");
        return NULL;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be >= 0");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || b_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        return NULL;
    }

    i32 lda = (m > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 ldb = (m > 0) ? (i32)PyArray_DIM(b_array, 0) : 1;

    i32 k = (m < n) ? m : n;
    i32 cs_size = (k > 0) ? 2 * k : 1;
    i32 tau_size = (k > 0) ? k : 1;
    i32 ldwork = (n > 0) ? n : 1;

    npy_intp cs_dims[1] = {cs_size};
    npy_intp tau_dims[1] = {tau_size};

    PyArrayObject *cs_array = (PyArrayObject *)PyArray_ZEROS(1, cs_dims, NPY_DOUBLE, 0);
    PyArrayObject *tau_array = (PyArrayObject *)PyArray_ZEROS(1, tau_dims, NPY_DOUBLE, 0);

    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));
    if (cs_array == NULL || tau_array == NULL || dwork == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(cs_array);
        Py_XDECREF(tau_array);
        free(dwork);
        return PyErr_NoMemory();
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *cs_data = (f64*)PyArray_DATA(cs_array);
    f64 *tau_data = (f64*)PyArray_DATA(tau_array);

    i32 info = 0;
    mb04su(m, n, a_data, lda, b_data, ldb, cs_data, tau_data, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    PyObject *result = Py_BuildValue("OOOOi", a_array, b_array, cs_array, tau_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(cs_array);
    Py_DECREF(tau_array);

    return result;
}

PyObject* py_mb04ud(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"jobq", "jobz", "m", "n", "a", "e", "q", "z", "tol", NULL};

    const char *jobq_str, *jobz_str;
    i32 m, n;
    PyObject *a_obj, *e_obj;
    PyObject *q_obj = Py_None;
    PyObject *z_obj = Py_None;
    double tol = 0.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssiiOO|OOd", kwlist,
            &jobq_str, &jobz_str, &m, &n, &a_obj, &e_obj, &q_obj, &z_obj, &tol)) {
        return NULL;
    }

    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be >= 0");
        return NULL;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be >= 0");
        return NULL;
    }

    char jobq = jobq_str[0];
    char jobz = jobz_str[0];
    bool ljobqi = (jobq == 'I' || jobq == 'i');
    bool ljobzi = (jobz == 'I' || jobz == 'i');
    bool updatq = ljobqi || (jobq == 'U' || jobq == 'u');
    bool updatz = ljobzi || (jobz == 'U' || jobz == 'u');

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *e_array = (PyArrayObject *)PyArray_FROM_OTF(
        e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        return NULL;
    }

    i32 lda = (m > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 lde = (m > 0) ? (i32)PyArray_DIM(e_array, 0) : 1;

    PyArrayObject *q_array = NULL;
    PyArrayObject *z_array = NULL;
    f64 *q_data = NULL;
    f64 *z_data = NULL;
    i32 ldq = 1;
    i32 ldz = 1;

    if (updatq) {
        if (ljobqi) {
            npy_intp q_dims[2] = {m > 0 ? m : 1, m > 0 ? m : 1};
            q_array = (PyArrayObject *)PyArray_ZEROS(2, q_dims, NPY_DOUBLE, 1);
            if (q_array == NULL) {
                Py_DECREF(a_array);
                Py_DECREF(e_array);
                return PyErr_NoMemory();
            }
        } else {
            q_array = (PyArrayObject *)PyArray_FROM_OTF(
                q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            if (q_array == NULL) {
                Py_DECREF(a_array);
                Py_DECREF(e_array);
                return NULL;
            }
        }
        q_data = (f64 *)PyArray_DATA(q_array);
        ldq = (m > 0) ? (i32)PyArray_DIM(q_array, 0) : 1;
    } else {
        npy_intp q_dims[2] = {1, 1};
        q_array = (PyArrayObject *)PyArray_ZEROS(2, q_dims, NPY_DOUBLE, 1);
        if (q_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            return PyErr_NoMemory();
        }
        q_data = (f64 *)PyArray_DATA(q_array);
        ldq = 1;
    }

    if (updatz) {
        if (ljobzi) {
            npy_intp z_dims[2] = {n > 0 ? n : 1, n > 0 ? n : 1};
            z_array = (PyArrayObject *)PyArray_ZEROS(2, z_dims, NPY_DOUBLE, 1);
            if (z_array == NULL) {
                Py_DECREF(a_array);
                Py_DECREF(e_array);
                Py_DECREF(q_array);
                return PyErr_NoMemory();
            }
        } else {
            z_array = (PyArrayObject *)PyArray_FROM_OTF(
                z_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            if (z_array == NULL) {
                Py_DECREF(a_array);
                Py_DECREF(e_array);
                Py_DECREF(q_array);
                return NULL;
            }
        }
        z_data = (f64 *)PyArray_DATA(z_array);
        ldz = (n > 0) ? (i32)PyArray_DIM(z_array, 0) : 1;
    } else {
        npy_intp z_dims[2] = {1, 1};
        z_array = (PyArrayObject *)PyArray_ZEROS(2, z_dims, NPY_DOUBLE, 1);
        if (z_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(q_array);
            return PyErr_NoMemory();
        }
        z_data = (f64 *)PyArray_DATA(z_array);
        ldz = 1;
    }

    i32 istair_size = (m > 0) ? m : 1;
    npy_intp istair_dims[1] = {istair_size};
    PyArrayObject *istair_array = (PyArrayObject *)PyArray_ZEROS(1, istair_dims, NPY_INT32, 0);
    if (istair_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        return PyErr_NoMemory();
    }

    i32 ldwork = (m > n) ? m : n;
    if (ldwork < 1) ldwork = 1;
    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        Py_DECREF(istair_array);
        return PyErr_NoMemory();
    }

    f64 *a_data = (f64 *)PyArray_DATA(a_array);
    f64 *e_data = (f64 *)PyArray_DATA(e_array);
    i32 *istair_data = (i32 *)PyArray_DATA(istair_array);

    i32 ranke = 0;
    i32 info = 0;

    mb04ud(jobq_str, jobz_str, m, n, a_data, lda, e_data, lde,
           q_data, ldq, z_data, ldz, &ranke, istair_data, tol, dwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    if (updatq && !ljobqi) {
        PyArray_ResolveWritebackIfCopy(q_array);
    }
    if (updatz && !ljobzi) {
        PyArray_ResolveWritebackIfCopy(z_array);
    }

    PyObject *result = Py_BuildValue("OOOOiOi", a_array, e_array, q_array, z_array, ranke, istair_array, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);
    Py_DECREF(istair_array);

    return result;
}

/* Python wrapper for mb04rt - blocked generalized Sylvester equation solver */
PyObject* py_mb04rt(PyObject* self, PyObject* args) {
    i32 m, n;
    f64 pmax;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *e_obj, *f_obj;

    if (!PyArg_ParseTuple(args, "iidOOOOOO",
                          &m, &n, &pmax,
                          &a_obj, &b_obj, &c_obj, &d_obj, &e_obj, &f_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }

    PyArrayObject *f_array = (PyArrayObject*)PyArray_FROM_OTF(f_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (f_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(e_array);
        return NULL;
    }

    i32 lda = (m > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 ldb = (n > 0) ? (i32)PyArray_DIM(b_array, 0) : 1;
    i32 ldc = (m > 0) ? (i32)PyArray_DIM(c_array, 0) : 1;
    i32 ldd = (m > 0) ? (i32)PyArray_DIM(d_array, 0) : 1;
    i32 lde = (n > 0) ? (i32)PyArray_DIM(e_array, 0) : 1;
    i32 ldf = (m > 0) ? (i32)PyArray_DIM(f_array, 0) : 1;

    const f64 *a_data = (const f64*)PyArray_DATA(a_array);
    const f64 *b_data = (const f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    const f64 *d_data = (const f64*)PyArray_DATA(d_array);
    const f64 *e_data = (const f64*)PyArray_DATA(e_array);
    f64 *f_data = (f64*)PyArray_DATA(f_array);

    i32 liwork = 2 * (m + n) + 6;
    i32 *iwork = (i32*)PyMem_Calloc(liwork > 0 ? liwork : 1, sizeof(i32));
    if (iwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(e_array);
        Py_DECREF(f_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    f64 scale;
    i32 info;

    mb04rt(m, n, pmax, a_data, lda, b_data, ldb, c_data, ldc,
           d_data, ldd, e_data, lde, f_data, ldf, &scale, iwork, &info);

    PyMem_Free(iwork);

    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(f_array);

    PyObject *result = Py_BuildValue("OOdi", c_array, f_array, scale, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(e_array);
    Py_DECREF(f_array);

    return result;
}

PyObject* py_mb04rv(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"m", "n", "pmax", "a", "b", "c", "d", "e", "f", NULL};

    i32 m, n;
    f64 pmax;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *e_obj, *f_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iidOOOOOO", kwlist,
                                     &m, &n, &pmax, &a_obj, &b_obj, &c_obj,
                                     &d_obj, &e_obj, &f_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_COMPLEX128, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_COMPLEX128, NPY_ARRAY_IN_FARRAY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_COMPLEX128,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_COMPLEX128, NPY_ARRAY_IN_FARRAY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_COMPLEX128, NPY_ARRAY_IN_FARRAY);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }

    PyArrayObject *f_array = (PyArrayObject*)PyArray_FROM_OTF(f_obj, NPY_COMPLEX128,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (f_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(e_array);
        return NULL;
    }

    i32 lda = (m > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 ldb = (n > 0) ? (i32)PyArray_DIM(b_array, 0) : 1;
    i32 ldc = (m > 0) ? (i32)PyArray_DIM(c_array, 0) : 1;
    i32 ldd = (m > 0) ? (i32)PyArray_DIM(d_array, 0) : 1;
    i32 lde = (n > 0) ? (i32)PyArray_DIM(e_array, 0) : 1;
    i32 ldf = (m > 0) ? (i32)PyArray_DIM(f_array, 0) : 1;

    const c128 *a_data = (const c128*)PyArray_DATA(a_array);
    const c128 *b_data = (const c128*)PyArray_DATA(b_array);
    c128 *c_data = (c128*)PyArray_DATA(c_array);
    const c128 *d_data = (const c128*)PyArray_DATA(d_array);
    const c128 *e_data = (const c128*)PyArray_DATA(e_array);
    c128 *f_data = (c128*)PyArray_DATA(f_array);

    f64 scale;
    i32 info;

    mb04rv(m, n, pmax, a_data, lda, b_data, ldb, c_data, ldc,
           d_data, ldd, e_data, lde, f_data, ldf, &scale, &info);

    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(f_array);

    PyObject *result = Py_BuildValue("OOdi", c_array, f_array, scale, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(e_array);
    Py_DECREF(f_array);

    return result;
}

PyObject* py_mb04rw(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"m", "n", "pmax", "a", "b", "c", "d", "e", "f", NULL};

    i32 m, n;
    f64 pmax;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *e_obj, *f_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iidOOOOOO", kwlist,
                                     &m, &n, &pmax, &a_obj, &b_obj, &c_obj,
                                     &d_obj, &e_obj, &f_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_COMPLEX128, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_COMPLEX128, NPY_ARRAY_IN_FARRAY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_COMPLEX128,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_COMPLEX128, NPY_ARRAY_IN_FARRAY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_COMPLEX128, NPY_ARRAY_IN_FARRAY);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }

    PyArrayObject *f_array = (PyArrayObject*)PyArray_FROM_OTF(f_obj, NPY_COMPLEX128,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (f_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(e_array);
        return NULL;
    }

    i32 lda = (m > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 ldb = (n > 0) ? (i32)PyArray_DIM(b_array, 0) : 1;
    i32 ldc = (m > 0) ? (i32)PyArray_DIM(c_array, 0) : 1;
    i32 ldd = (m > 0) ? (i32)PyArray_DIM(d_array, 0) : 1;
    i32 lde = (n > 0) ? (i32)PyArray_DIM(e_array, 0) : 1;
    i32 ldf = (m > 0) ? (i32)PyArray_DIM(f_array, 0) : 1;

    const c128 *a_data = (const c128*)PyArray_DATA(a_array);
    const c128 *b_data = (const c128*)PyArray_DATA(b_array);
    c128 *c_data = (c128*)PyArray_DATA(c_array);
    const c128 *d_data = (const c128*)PyArray_DATA(d_array);
    const c128 *e_data = (const c128*)PyArray_DATA(e_array);
    c128 *f_data = (c128*)PyArray_DATA(f_array);

    i32 iwork_size = m + n + 2;
    i32 *iwork = (i32*)PyMem_Calloc(iwork_size > 0 ? iwork_size : 1, sizeof(i32));
    if (iwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(e_array);
        Py_DECREF(f_array);
        return PyErr_NoMemory();
    }

    f64 scale;
    i32 info;

    mb04rw(m, n, pmax, a_data, lda, b_data, ldb, c_data, ldc,
           d_data, ldd, e_data, lde, f_data, ldf, &scale, iwork, &info);

    PyMem_Free(iwork);

    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(f_array);

    PyObject *result = Py_BuildValue("OOdi", c_array, f_array, scale, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(e_array);
    Py_DECREF(f_array);

    return result;
}

PyObject* py_mb04tt(PyObject* self, PyObject* args)
{
    int updatq_int, updatz_int;
    int m, n, ifira, ifica, nca;
    PyObject *a_obj, *e_obj, *q_obj, *z_obj, *istair_obj;
    double tol;

    if (!PyArg_ParseTuple(args, "ppiiiiiOOOOOd",
                          &updatq_int, &updatz_int, &m, &n, &ifira, &ifica, &nca,
                          &a_obj, &e_obj, &q_obj, &z_obj, &istair_obj, &tol)) {
        return NULL;
    }

    bool updatq = (bool)updatq_int;
    bool updatz = (bool)updatz_int;

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (q_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        return NULL;
    }

    PyArrayObject *z_array = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (z_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(q_array);
        return NULL;
    }

    PyArrayObject *istair_array = (PyArrayObject*)PyArray_FROM_OTF(istair_obj, NPY_INT32,
                                                                   NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (istair_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        return NULL;
    }

    i32 lda = (m > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 lde = (m > 0) ? (i32)PyArray_DIM(e_array, 0) : 1;
    i32 ldq = (m > 0 && updatq) ? (i32)PyArray_DIM(q_array, 0) : 1;
    i32 ldz = (n > 0 && updatz) ? (i32)PyArray_DIM(z_array, 0) : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *q_data = (f64*)PyArray_DATA(q_array);
    f64 *z_data = (f64*)PyArray_DATA(z_array);
    i32 *istair_data = (i32*)PyArray_DATA(istair_array);

    i32 *iwork = (i32*)PyMem_Calloc(n > 0 ? n : 1, sizeof(i32));
    if (iwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        Py_DECREF(istair_array);
        return PyErr_NoMemory();
    }

    i32 rank = 0;

    mb04tt(updatq, updatz, (i32)m, (i32)n, (i32)ifira, (i32)ifica, (i32)nca,
           a_data, lda, e_data, lde, q_data, ldq, z_data, ldz,
           istair_data, &rank, tol, iwork);

    PyMem_Free(iwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(q_array);
    PyArray_ResolveWritebackIfCopy(z_array);
    PyArray_ResolveWritebackIfCopy(istair_array);

    PyObject *result = Py_BuildValue("OOOOOii", a_array, e_array, q_array, z_array,
                                     istair_array, rank, 0);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);
    Py_DECREF(istair_array);

    return result;
}

PyObject* py_mb04wu(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"tranq1", "tranq2", "m", "n", "k", "q1", "q2", "cs", "tau", NULL};

    int tranq1_int, tranq2_int;
    i32 m, n, k;
    PyObject *q1_obj, *q2_obj, *cs_obj, *tau_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ppiiiOOOO", kwlist,
            &tranq1_int, &tranq2_int, &m, &n, &k, &q1_obj, &q2_obj, &cs_obj, &tau_obj)) {
        return NULL;
    }

    bool tranq1 = (bool)tranq1_int;
    bool tranq2 = (bool)tranq2_int;

    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be >= 0");
        return NULL;
    }
    if (n < 0 || n > m) {
        PyErr_SetString(PyExc_ValueError, "n must satisfy 0 <= n <= m");
        return NULL;
    }
    if (k < 0 || k > n) {
        PyErr_SetString(PyExc_ValueError, "k must satisfy 0 <= k <= n");
        return NULL;
    }

    PyArrayObject *q1_array = (PyArrayObject *)PyArray_FROM_OTF(
        q1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (q1_array == NULL) {
        return NULL;
    }

    PyArrayObject *q2_array = (PyArrayObject *)PyArray_FROM_OTF(
        q2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (q2_array == NULL) {
        Py_DECREF(q1_array);
        return NULL;
    }

    PyArrayObject *cs_array = (PyArrayObject *)PyArray_FROM_OTF(
        cs_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (cs_array == NULL) {
        Py_DECREF(q1_array);
        Py_DECREF(q2_array);
        return NULL;
    }

    PyArrayObject *tau_array = (PyArrayObject *)PyArray_FROM_OTF(
        tau_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (tau_array == NULL) {
        Py_DECREF(q1_array);
        Py_DECREF(q2_array);
        Py_DECREF(cs_array);
        return NULL;
    }

    i32 ldq1 = (i32)PyArray_DIM(q1_array, 0);
    i32 ldq2 = (i32)PyArray_DIM(q2_array, 0);
    if (ldq1 < 1) ldq1 = 1;
    if (ldq2 < 1) ldq2 = 1;

    i32 ldwork = (m + n > 1) ? (m + n) : 1;
    f64 *dwork = (f64 *)PyMem_Calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(q1_array);
        Py_DECREF(q2_array);
        Py_DECREF(cs_array);
        Py_DECREF(tau_array);
        return PyErr_NoMemory();
    }

    f64 *q1_data = (f64*)PyArray_DATA(q1_array);
    f64 *q2_data = (f64*)PyArray_DATA(q2_array);
    f64 *cs_data = (f64*)PyArray_DATA(cs_array);
    f64 *tau_data = (f64*)PyArray_DATA(tau_array);

    i32 info = 0;
    mb04wu(tranq1, tranq2, m, n, k, q1_data, ldq1, q2_data, ldq2,
           cs_data, tau_data, dwork, ldwork, &info);

    PyMem_Free(dwork);

    PyArray_ResolveWritebackIfCopy(q1_array);
    PyArray_ResolveWritebackIfCopy(q2_array);

    PyObject *result = Py_BuildValue("OOi", q1_array, q2_array, info);

    Py_DECREF(q1_array);
    Py_DECREF(q2_array);
    Py_DECREF(cs_array);
    Py_DECREF(tau_array);

    return result;
}

PyObject *py_mb04rd(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"jobx", "joby", "sort", "n", "pmax", "a", "b",
                             "x", "y", "tol", NULL};

    const char *jobx, *joby, *sort;
    int n;
    double pmax, tol;
    PyObject *a_obj, *b_obj, *x_obj, *y_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssidOOOOd:mb04rd", kwlist,
                                     &jobx, &joby, &sort, &n, &pmax,
                                     &a_obj, &b_obj, &x_obj, &y_obj, &tol)) {
        return NULL;
    }

    bool wantx = (jobx[0] == 'U' || jobx[0] == 'u');
    bool wanty = (joby[0] == 'U' || joby[0] == 'u');

    int requirements = NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY;

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, requirements);
    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, requirements);
    PyArrayObject *x_array = (PyArrayObject *)PyArray_FROM_OTF(
        x_obj, NPY_DOUBLE, requirements);
    PyArrayObject *y_array = (PyArrayObject *)PyArray_FROM_OTF(
        y_obj, NPY_DOUBLE, requirements);

    if (a_array == NULL || b_array == NULL || x_array == NULL || y_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        return NULL;
    }

    i32 lda = (n > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 ldb = (n > 0) ? (i32)PyArray_DIM(b_array, 0) : 1;
    i32 ldx = (n > 0 && wantx) ? (i32)PyArray_DIM(x_array, 0) : 1;
    i32 ldy = (n > 0 && wanty) ? (i32)PyArray_DIM(y_array, 0) : 1;

    f64 *a_data = (f64 *)PyArray_DATA(a_array);
    f64 *b_data = (f64 *)PyArray_DATA(b_array);
    f64 *x_data = (f64 *)PyArray_DATA(x_array);
    f64 *y_data = (f64 *)PyArray_DATA(y_array);

    i32 n_alloc = (n > 0) ? n : 1;
    i32 *blsize = (i32 *)calloc(n_alloc, sizeof(i32));
    f64 *alphar = (f64 *)calloc(n_alloc, sizeof(f64));
    f64 *alphai = (f64 *)calloc(n_alloc, sizeof(f64));
    f64 *beta_arr = (f64 *)calloc(n_alloc, sizeof(f64));
    i32 *iwork = (i32 *)calloc(n_alloc + 6, sizeof(i32));

    i32 ldwork = (n <= 1) ? 1 : 4 * n + 16;
    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));

    if (!blsize || !alphar || !alphai || !beta_arr || !iwork || !dwork) {
        free(blsize); free(alphar); free(alphai);
        free(beta_arr); free(iwork); free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        return PyErr_NoMemory();
    }

    i32 nblcks = 0;
    i32 info = 0;

    mb04rd(jobx, joby, sort, n, pmax, a_data, lda, b_data, ldb,
           x_data, ldx, y_data, ldy, &nblcks, blsize, alphar, alphai,
           beta_arr, tol, iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(x_array);
    PyArray_ResolveWritebackIfCopy(y_array);

    npy_intp blsize_dims[1] = {n_alloc};
    PyObject *blsize_array = PyArray_SimpleNew(1, blsize_dims, NPY_INT32);
    memcpy(PyArray_DATA((PyArrayObject *)blsize_array), blsize, n_alloc * sizeof(i32));
    free(blsize);

    npy_intp eig_dims[1] = {n_alloc};
    PyObject *alphar_array = PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);
    PyObject *alphai_array = PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);
    PyObject *beta_array = PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);
    memcpy(PyArray_DATA((PyArrayObject *)alphar_array), alphar, n_alloc * sizeof(f64));
    memcpy(PyArray_DATA((PyArrayObject *)alphai_array), alphai, n_alloc * sizeof(f64));
    memcpy(PyArray_DATA((PyArrayObject *)beta_array), beta_arr, n_alloc * sizeof(f64));
    free(alphar);
    free(alphai);
    free(beta_arr);

    PyObject *result = Py_BuildValue("OOOOiOOOOi",
                                     a_array, b_array, x_array, y_array,
                                     nblcks, blsize_array,
                                     alphar_array, alphai_array, beta_array,
                                     info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(blsize_array);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);

    return result;
}

/* Python wrapper for mb04tx */
PyObject* py_mb04tx(PyObject* self, PyObject* args, PyObject* kwds) {
    static char *kwlist[] = {"a", "e", "inuk", "imuk", "q", "z",
                             "m", "n", "updatq", "updatz", NULL};

    PyObject *a_obj, *e_obj, *inuk_obj, *imuk_obj;
    PyObject *q_obj = Py_None, *z_obj = Py_None;
    i32 m_arg = -1, n_arg = -1;
    int updatq = 1, updatz = 1;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOO|OOiipp", kwlist,
                                     &a_obj, &e_obj, &inuk_obj, &imuk_obj,
                                     &q_obj, &z_obj, &m_arg, &n_arg,
                                     &updatq, &updatz)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(
        e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *inuk_array = (PyArrayObject*)PyArray_FROM_OTF(
        inuk_obj, NPY_INT32, NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (inuk_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        return NULL;
    }

    PyArrayObject *imuk_array = (PyArrayObject*)PyArray_FROM_OTF(
        imuk_obj, NPY_INT32, NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (imuk_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(inuk_array);
        return NULL;
    }

    i32 nblcks = (i32)PyArray_DIM(inuk_array, 0);
    i32 m = (m_arg >= 0) ? m_arg : (i32)PyArray_DIM(a_array, 0);
    i32 n = (n_arg >= 0) ? n_arg : (i32)PyArray_DIM(a_array, 1);
    i32 lda = (m > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 lde = (m > 0) ? (i32)PyArray_DIM(e_array, 0) : 1;

    f64 *a_data = (f64 *)PyArray_DATA(a_array);
    f64 *e_data = (f64 *)PyArray_DATA(e_array);
    i32 *inuk_data = (i32 *)PyArray_DATA(inuk_array);
    i32 *imuk_data = (i32 *)PyArray_DATA(imuk_array);

    PyArrayObject *q_array = NULL;
    PyArrayObject *z_array = NULL;
    f64 *q_data = NULL;
    f64 *z_data = NULL;
    i32 ldq = 1, ldz = 1;

    if (updatq && q_obj != Py_None) {
        q_array = (PyArrayObject*)PyArray_FROM_OTF(
            q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (q_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(inuk_array);
            Py_DECREF(imuk_array);
            return NULL;
        }
        q_data = (f64 *)PyArray_DATA(q_array);
        ldq = (m > 0) ? (i32)PyArray_DIM(q_array, 0) : 1;
    }

    if (updatz && z_obj != Py_None) {
        z_array = (PyArrayObject*)PyArray_FROM_OTF(
            z_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (z_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(inuk_array);
            Py_DECREF(imuk_array);
            Py_XDECREF(q_array);
            return NULL;
        }
        z_data = (f64 *)PyArray_DATA(z_array);
        ldz = (n > 0) ? (i32)PyArray_DIM(z_array, 0) : 1;
    }

    i32 mnei[4];
    mb04tx((bool)updatq, (bool)updatz, m, n, &nblcks,
           inuk_data, imuk_data, a_data, lda, e_data, lde,
           q_data, ldq, z_data, ldz, mnei);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(inuk_array);
    PyArray_ResolveWritebackIfCopy(imuk_array);
    if (q_array) PyArray_ResolveWritebackIfCopy(q_array);
    if (z_array) PyArray_ResolveWritebackIfCopy(z_array);

    npy_intp mnei_dims[1] = {4};
    PyObject *mnei_array = PyArray_SimpleNew(1, mnei_dims, NPY_INT32);
    memcpy(PyArray_DATA((PyArrayObject *)mnei_array), mnei, 4 * sizeof(i32));

    PyObject *q_result = q_array ? (PyObject *)q_array : Py_None;
    PyObject *z_result = z_array ? (PyObject *)z_array : Py_None;
    if (!q_array) Py_INCREF(Py_None);
    if (!z_array) Py_INCREF(Py_None);

    PyObject *result = Py_BuildValue("OOOOiOOOi",
                                     a_array, e_array, q_result, z_result,
                                     nblcks, inuk_array, imuk_array, mnei_array,
                                     0);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(inuk_array);
    Py_DECREF(imuk_array);
    if (q_array) Py_DECREF(q_array);
    if (z_array) Py_DECREF(z_array);
    Py_DECREF(mnei_array);

    return result;
}

PyObject *py_mb04vx(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;

    static char *kwlist[] = {"a", "e", "q", "z", "nblcks", "inuk", "imuk",
                             "updatq", "updatz", NULL};
    PyObject *a_obj, *e_obj, *q_obj, *z_obj, *inuk_obj, *imuk_obj;
    int nblcks;
    int updatq = 0, updatz = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOiOO|pp", kwlist,
                                     &a_obj, &e_obj, &q_obj, &z_obj,
                                     &nblcks, &inuk_obj, &imuk_obj,
                                     &updatq, &updatz)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *e_array = (PyArrayObject *)PyArray_FROM_OTF(
        e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *inuk_array = (PyArrayObject *)PyArray_FROM_OTF(
        inuk_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (inuk_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        return NULL;
    }

    PyArrayObject *imuk_array = (PyArrayObject *)PyArray_FROM_OTF(
        imuk_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (imuk_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(inuk_array);
        return NULL;
    }

    i32 m = (PyArray_NDIM(a_array) >= 1) ? (i32)PyArray_DIM(a_array, 0) : 0;
    i32 n = (PyArray_NDIM(a_array) >= 2) ? (i32)PyArray_DIM(a_array, 1) : 0;
    i32 lda = (m > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 lde = (m > 0) ? (i32)PyArray_DIM(e_array, 0) : 1;

    f64 *a_data = (f64 *)PyArray_DATA(a_array);
    f64 *e_data = (f64 *)PyArray_DATA(e_array);
    i32 *inuk_data = (i32 *)PyArray_DATA(inuk_array);
    i32 *imuk_data = (i32 *)PyArray_DATA(imuk_array);

    f64 *q_data = NULL;
    f64 *z_data = NULL;
    i32 ldq = 1, ldz = 1;
    PyArrayObject *q_array = NULL;
    PyArrayObject *z_array = NULL;

    if (updatq) {
        q_array = (PyArrayObject *)PyArray_FROM_OTF(
            q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (q_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(inuk_array);
            Py_DECREF(imuk_array);
            return NULL;
        }
        q_data = (f64 *)PyArray_DATA(q_array);
        ldq = (m > 0) ? (i32)PyArray_DIM(q_array, 0) : 1;
    }

    if (updatz) {
        z_array = (PyArrayObject *)PyArray_FROM_OTF(
            z_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (z_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(inuk_array);
            Py_DECREF(imuk_array);
            Py_XDECREF(q_array);
            return NULL;
        }
        z_data = (f64 *)PyArray_DATA(z_array);
        ldz = (n > 0) ? (i32)PyArray_DIM(z_array, 0) : 1;
    }

    i32 mnei[3];
    mb04vx((bool)updatq, (bool)updatz, m, n, nblcks,
           inuk_data, imuk_data, a_data, lda, e_data, lde,
           q_data, ldq, z_data, ldz, mnei);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(inuk_array);
    PyArray_ResolveWritebackIfCopy(imuk_array);
    if (q_array) PyArray_ResolveWritebackIfCopy(q_array);
    if (z_array) PyArray_ResolveWritebackIfCopy(z_array);

    npy_intp mnei_dims[1] = {3};
    PyObject *mnei_array = PyArray_SimpleNew(1, mnei_dims, NPY_INT32);
    memcpy(PyArray_DATA((PyArrayObject *)mnei_array), mnei, 3 * sizeof(i32));

    PyObject *q_result = q_array ? (PyObject *)q_array : Py_None;
    PyObject *z_result = z_array ? (PyObject *)z_array : Py_None;
    if (!q_array) Py_INCREF(Py_None);
    if (!z_array) Py_INCREF(Py_None);

    PyObject *result = Py_BuildValue("OOOOOOO",
                                     a_array, e_array, q_result, z_result,
                                     inuk_array, imuk_array, mnei_array);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(inuk_array);
    Py_DECREF(imuk_array);
    if (q_array) Py_DECREF(q_array);
    if (z_array) Py_DECREF(z_array);
    Py_DECREF(mnei_array);

    return result;
}

PyObject* py_mb04wd(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"tranq1", "tranq2", "m", "n", "k", "q1", "q2", "cs", "tau", NULL};

    int tranq1_int, tranq2_int;
    i32 m, n, k;
    PyObject *q1_obj, *q2_obj, *cs_obj, *tau_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ppiiiOOOO", kwlist,
            &tranq1_int, &tranq2_int, &m, &n, &k, &q1_obj, &q2_obj, &cs_obj, &tau_obj)) {
        return NULL;
    }

    bool tranq1 = (bool)tranq1_int;
    bool tranq2 = (bool)tranq2_int;

    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be >= 0");
        return NULL;
    }
    if (n < 0 || n > m) {
        PyErr_SetString(PyExc_ValueError, "n must satisfy 0 <= n <= m");
        return NULL;
    }
    if (k < 0 || k > n) {
        PyErr_SetString(PyExc_ValueError, "k must satisfy 0 <= k <= n");
        return NULL;
    }

    PyArrayObject *q1_array = (PyArrayObject *)PyArray_FROM_OTF(
        q1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (q1_array == NULL) {
        return NULL;
    }

    PyArrayObject *q2_array = (PyArrayObject *)PyArray_FROM_OTF(
        q2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (q2_array == NULL) {
        Py_DECREF(q1_array);
        return NULL;
    }

    PyArrayObject *cs_array = (PyArrayObject *)PyArray_FROM_OTF(
        cs_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (cs_array == NULL) {
        Py_DECREF(q1_array);
        Py_DECREF(q2_array);
        return NULL;
    }

    PyArrayObject *tau_array = (PyArrayObject *)PyArray_FROM_OTF(
        tau_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (tau_array == NULL) {
        Py_DECREF(q1_array);
        Py_DECREF(q2_array);
        Py_DECREF(cs_array);
        return NULL;
    }

    i32 ldq1 = (i32)PyArray_DIM(q1_array, 0);
    i32 ldq2 = (i32)PyArray_DIM(q2_array, 0);
    if (ldq1 < 1) ldq1 = 1;
    if (ldq2 < 1) ldq2 = 1;

    i32 minwrk = (m + n > 1) ? (m + n) : 1;
    i32 nb = 32;
    i32 opt2 = 8 * n * nb + 15 * nb * nb;
    i32 ldwork = (minwrk > opt2) ? minwrk : opt2;
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(q1_array);
        Py_DECREF(q2_array);
        Py_DECREF(cs_array);
        Py_DECREF(tau_array);
        return PyErr_NoMemory();
    }

    f64 *q1_data = (f64*)PyArray_DATA(q1_array);
    f64 *q2_data = (f64*)PyArray_DATA(q2_array);
    f64 *cs_data = (f64*)PyArray_DATA(cs_array);
    f64 *tau_data = (f64*)PyArray_DATA(tau_array);

    i32 info = 0;
    mb04wd(tranq1, tranq2, m, n, k, q1_data, ldq1, q2_data, ldq2,
           cs_data, tau_data, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(q1_array);
    PyArray_ResolveWritebackIfCopy(q2_array);

    PyObject *result = Py_BuildValue("OOi", q1_array, q2_array, info);

    Py_DECREF(q1_array);
    Py_DECREF(q2_array);
    Py_DECREF(cs_array);
    Py_DECREF(tau_array);

    return result;
}

PyObject* py_mb04xy(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"jobu", "jobv", "m", "n", "x", "taup", "tauq",
                             "u", "v", "inul", NULL};
    const char *jobu, *jobv;
    i32 m, n;
    PyObject *x_obj, *taup_obj, *tauq_obj;
    PyObject *u_obj = Py_None, *v_obj = Py_None;
    PyObject *inul_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssiiOOO|OOO", kwlist,
                                     &jobu, &jobv, &m, &n, &x_obj, &taup_obj,
                                     &tauq_obj, &u_obj, &v_obj, &inul_obj)) {
        return NULL;
    }

    PyArrayObject *x_array = (PyArrayObject*)PyArray_FROM_OTF(
        x_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (x_array == NULL) {
        return NULL;
    }

    PyArrayObject *taup_array = (PyArrayObject*)PyArray_FROM_OTF(
        taup_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (taup_array == NULL) {
        Py_DECREF(x_array);
        return NULL;
    }

    PyArrayObject *tauq_array = (PyArrayObject*)PyArray_FROM_OTF(
        tauq_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (tauq_array == NULL) {
        Py_DECREF(x_array);
        Py_DECREF(taup_array);
        return NULL;
    }

    PyArrayObject *inul_array = (PyArrayObject*)PyArray_FROM_OTF(
        inul_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    if (inul_array == NULL) {
        Py_DECREF(x_array);
        Py_DECREF(taup_array);
        Py_DECREF(tauq_array);
        return NULL;
    }

    bool wantu = (jobu[0] == 'A' || jobu[0] == 'a' ||
                  jobu[0] == 'S' || jobu[0] == 's');
    bool wantv = (jobv[0] == 'A' || jobv[0] == 'a' ||
                  jobv[0] == 'S' || jobv[0] == 's');

    PyArrayObject *u_array = NULL;
    PyArrayObject *v_array = NULL;

    if (wantu) {
        if (u_obj == Py_None) {
            PyErr_SetString(PyExc_ValueError, "u is required when jobu='A' or 'S'");
            Py_DECREF(x_array);
            Py_DECREF(taup_array);
            Py_DECREF(tauq_array);
            Py_DECREF(inul_array);
            return NULL;
        }
        u_array = (PyArrayObject*)PyArray_FROM_OTF(
            u_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (u_array == NULL) {
            Py_DECREF(x_array);
            Py_DECREF(taup_array);
            Py_DECREF(tauq_array);
            Py_DECREF(inul_array);
            return NULL;
        }
    }

    if (wantv) {
        if (v_obj == Py_None) {
            PyErr_SetString(PyExc_ValueError, "v is required when jobv='A' or 'S'");
            Py_DECREF(x_array);
            Py_DECREF(taup_array);
            Py_DECREF(tauq_array);
            Py_DECREF(inul_array);
            if (u_array != NULL) Py_DECREF(u_array);
            return NULL;
        }
        v_array = (PyArrayObject*)PyArray_FROM_OTF(
            v_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (v_array == NULL) {
            Py_DECREF(x_array);
            Py_DECREF(taup_array);
            Py_DECREF(tauq_array);
            Py_DECREF(inul_array);
            if (u_array != NULL) Py_DECREF(u_array);
            return NULL;
        }
    }

    i32 ldx = (m > 0) ? (i32)PyArray_DIM(x_array, 0) : 1;
    i32 ldu = 1;
    i32 ldv = 1;

    f64 *x_data = (f64*)PyArray_DATA(x_array);
    f64 *taup_data = (f64*)PyArray_DATA(taup_array);
    f64 *tauq_data = (f64*)PyArray_DATA(tauq_array);
    bool *inul_data = (bool*)PyArray_DATA(inul_array);

    f64 *u_data = NULL;
    f64 *v_data = NULL;

    if (wantu) {
        ldu = (i32)PyArray_DIM(u_array, 0);
        if (ldu < 1) ldu = 1;
        u_data = (f64*)PyArray_DATA(u_array);
    }

    if (wantv) {
        ldv = (i32)PyArray_DIM(v_array, 0);
        if (ldv < 1) ldv = 1;
        v_data = (f64*)PyArray_DATA(v_array);
    }

    i32 info = 0;
    mb04xy(jobu, jobv, m, n, x_data, ldx, taup_data, tauq_data,
           u_data, ldu, v_data, ldv, inul_data, &info);

    PyArray_ResolveWritebackIfCopy(x_array);
    if (wantu) PyArray_ResolveWritebackIfCopy(u_array);
    if (wantv) PyArray_ResolveWritebackIfCopy(v_array);

    PyObject *u_result = Py_None;
    PyObject *v_result = Py_None;
    if (wantu) {
        u_result = (PyObject*)u_array;
    }
    if (wantv) {
        v_result = (PyObject*)v_array;
    }

    PyObject *result = Py_BuildValue("OOi", u_result, v_result, info);

    Py_DECREF(x_array);
    Py_DECREF(taup_array);
    Py_DECREF(tauq_array);
    Py_DECREF(inul_array);
    if (wantu) Py_DECREF(u_array);
    if (wantv) Py_DECREF(v_array);

    return result;
}

PyObject* py_mb04yw(PyObject* self, PyObject* args)
{
    (void)self;
    int qrit_int, updatu_int, updatv_int;
    int m, n, l, k;
    double shift;
    PyObject *d_obj, *e_obj, *u_obj, *v_obj;

    if (!PyArg_ParseTuple(args, "iiiiiiidOOOO",
                          &qrit_int, &updatu_int, &updatv_int,
                          &m, &n, &l, &k, &shift,
                          &d_obj, &e_obj, &u_obj, &v_obj)) {
        return NULL;
    }

    bool qrit = (qrit_int != 0);
    bool updatu = (updatu_int != 0);
    bool updatv = (updatv_int != 0);

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (d_array == NULL) return NULL;

    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(
        e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (e_array == NULL) {
        Py_DECREF(d_array);
        return NULL;
    }

    i32 p = (m < n) ? m : n;

    PyArrayObject *u_array = NULL;
    PyArrayObject *v_array = NULL;
    f64 *u_data = NULL;
    f64 *v_data = NULL;
    i32 ldu = 1;
    i32 ldv = 1;

    if (updatu) {
        if (u_obj == Py_None) {
            PyErr_SetString(PyExc_ValueError, "u is required when updatu=True");
            Py_DECREF(d_array);
            Py_DECREF(e_array);
            return NULL;
        }
        u_array = (PyArrayObject*)PyArray_FROM_OTF(
            u_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (u_array == NULL) {
            Py_DECREF(d_array);
            Py_DECREF(e_array);
            return NULL;
        }
        ldu = (m > 0) ? (i32)PyArray_DIM(u_array, 0) : 1;
        u_data = (f64*)PyArray_DATA(u_array);
    }

    if (updatv) {
        if (v_obj == Py_None) {
            PyErr_SetString(PyExc_ValueError, "v is required when updatv=True");
            Py_DECREF(d_array);
            Py_DECREF(e_array);
            if (u_array) Py_DECREF(u_array);
            return NULL;
        }
        v_array = (PyArrayObject*)PyArray_FROM_OTF(
            v_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (v_array == NULL) {
            Py_DECREF(d_array);
            Py_DECREF(e_array);
            if (u_array) Py_DECREF(u_array);
            return NULL;
        }
        ldv = (n > 0) ? (i32)PyArray_DIM(v_array, 0) : 1;
        v_data = (f64*)PyArray_DATA(v_array);
    }

    i32 ldwork = 1;
    if (updatu && updatv) {
        ldwork = 4 * p - 4;
    } else if (updatu || updatv) {
        ldwork = 2 * p - 2;
    }
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64*)PyMem_Calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(d_array);
        Py_DECREF(e_array);
        if (u_array) Py_DECREF(u_array);
        if (v_array) Py_DECREF(v_array);
        return PyErr_NoMemory();
    }

    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);

    mb04yw(qrit, updatu, updatv, m, n, l, k, shift,
           d_data, e_data, u_data, ldu, v_data, ldv, dwork);

    PyMem_Free(dwork);

    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    if (u_array) PyArray_ResolveWritebackIfCopy(u_array);
    if (v_array) PyArray_ResolveWritebackIfCopy(v_array);

    PyObject *u_result = Py_None;
    PyObject *v_result = Py_None;
    if (updatu) {
        u_result = (PyObject*)u_array;
    }
    if (updatv) {
        v_result = (PyObject*)v_array;
    }

    i32 info = 0;
    PyObject *result = Py_BuildValue("OOOOi",
                                     (PyObject*)d_array, (PyObject*)e_array,
                                     u_result, v_result, info);

    Py_DECREF(d_array);
    Py_DECREF(e_array);
    if (u_array) Py_DECREF(u_array);
    if (v_array) Py_DECREF(v_array);

    return result;
}

/* Python wrapper for mb04rz */
PyObject *py_mb04rz(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"jobx", "joby", "sort", "n", "pmax", "a", "b",
                             "x", "y", "tol", NULL};

    const char *jobx, *joby, *sort;
    int n;
    double pmax, tol;
    PyObject *a_obj, *b_obj, *x_obj, *y_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssidOOOOd:mb04rz", kwlist,
                                     &jobx, &joby, &sort, &n, &pmax,
                                     &a_obj, &b_obj, &x_obj, &y_obj, &tol)) {
        return NULL;
    }

    bool wantx = (jobx[0] == 'U' || jobx[0] == 'u');
    bool wanty = (joby[0] == 'U' || joby[0] == 'u');

    int requirements = NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY;

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_COMPLEX128, requirements);
    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_COMPLEX128, requirements);
    PyArrayObject *x_array = (PyArrayObject *)PyArray_FROM_OTF(
        x_obj, NPY_COMPLEX128, requirements);
    PyArrayObject *y_array = (PyArrayObject *)PyArray_FROM_OTF(
        y_obj, NPY_COMPLEX128, requirements);

    if (a_array == NULL || b_array == NULL || x_array == NULL || y_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        return NULL;
    }

    i32 lda = (n > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 ldb = (n > 0) ? (i32)PyArray_DIM(b_array, 0) : 1;
    i32 ldx = (n > 0 && wantx) ? (i32)PyArray_DIM(x_array, 0) : 1;
    i32 ldy = (n > 0 && wanty) ? (i32)PyArray_DIM(y_array, 0) : 1;

    c128 *a_data = (c128 *)PyArray_DATA(a_array);
    c128 *b_data = (c128 *)PyArray_DATA(b_array);
    c128 *x_data = (c128 *)PyArray_DATA(x_array);
    c128 *y_data = (c128 *)PyArray_DATA(y_array);

    i32 n_alloc = (n > 0) ? n : 1;
    i32 *blsize = (i32 *)PyMem_Calloc(n_alloc, sizeof(i32));
    c128 *alpha = (c128 *)PyMem_Calloc(n_alloc, sizeof(c128));
    c128 *beta_arr = (c128 *)PyMem_Calloc(n_alloc, sizeof(c128));
    i32 *iwork = (i32 *)PyMem_Calloc(n_alloc + 2, sizeof(i32));

    if (!blsize || !alpha || !beta_arr || !iwork) {
        PyMem_Free(blsize); PyMem_Free(alpha);
        PyMem_Free(beta_arr); PyMem_Free(iwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        return PyErr_NoMemory();
    }

    i32 nblcks = 0;
    i32 info = 0;

    mb04rz(jobx, joby, sort, n, pmax, a_data, lda, b_data, ldb,
           x_data, ldx, y_data, ldy, &nblcks, blsize, alpha,
           beta_arr, tol, iwork, &info);

    PyMem_Free(iwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(x_array);
    PyArray_ResolveWritebackIfCopy(y_array);

    npy_intp blsize_dims[1] = {n_alloc};
    PyObject *blsize_array = PyArray_SimpleNew(1, blsize_dims, NPY_INT32);
    memcpy(PyArray_DATA((PyArrayObject *)blsize_array), blsize, n_alloc * sizeof(i32));
    PyMem_Free(blsize);

    npy_intp eig_dims[1] = {n_alloc};
    PyObject *alpha_array = PyArray_SimpleNew(1, eig_dims, NPY_COMPLEX128);
    PyObject *beta_array = PyArray_SimpleNew(1, eig_dims, NPY_COMPLEX128);
    memcpy(PyArray_DATA((PyArrayObject *)alpha_array), alpha, n_alloc * sizeof(c128));
    memcpy(PyArray_DATA((PyArrayObject *)beta_array), beta_arr, n_alloc * sizeof(c128));
    PyMem_Free(alpha);
    PyMem_Free(beta_arr);

    PyObject *result = Py_BuildValue("OOOOiOOOi",
                                     a_array, b_array, x_array, y_array,
                                     nblcks, blsize_array,
                                     alpha_array, beta_array,
                                     info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(blsize_array);
    Py_DECREF(alpha_array);
    Py_DECREF(beta_array);

    return result;
}

/* Python wrapper for mb04vd - upper block triangular form for rectangular pencil sE-A */
PyObject* py_mb04vd(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"mode", "jobq", "jobz", "m", "n", "ranke",
                             "a", "e", "q", "z", "istair", "tol", NULL};

    const char *mode_str, *jobq_str, *jobz_str;
    i32 m, n, ranke;
    PyObject *a_obj, *e_obj, *q_obj, *z_obj, *istair_obj;
    double tol = 0.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssiiiOOOOO|d", kwlist,
            &mode_str, &jobq_str, &jobz_str, &m, &n, &ranke,
            &a_obj, &e_obj, &q_obj, &z_obj, &istair_obj, &tol)) {
        return NULL;
    }

    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be >= 0");
        return NULL;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be >= 0");
        return NULL;
    }
    if (ranke < 0) {
        PyErr_SetString(PyExc_ValueError, "ranke must be >= 0");
        return NULL;
    }

    char mode = mode_str[0];
    char jobq = jobq_str[0];
    char jobz = jobz_str[0];
    bool lmodes = (mode == 'S' || mode == 's');
    bool ljobqi = (jobq == 'I' || jobq == 'i');
    bool ljobzi = (jobz == 'I' || jobz == 'i');
    bool updatq = ljobqi || (jobq == 'U' || jobq == 'u');
    bool updatz = ljobzi || (jobz == 'U' || jobz == 'u');

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *e_array = (PyArrayObject *)PyArray_FROM_OTF(
        e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *istair_array = (PyArrayObject *)PyArray_FROM_OTF(
        istair_obj, NPY_INT32, NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_array == NULL || e_array == NULL || istair_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(istair_array);
        return NULL;
    }

    i32 lda = (m > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 lde = (m > 0) ? (i32)PyArray_DIM(e_array, 0) : 1;

    PyArrayObject *q_array = NULL;
    PyArrayObject *z_array = NULL;
    f64 *q_data = NULL;
    f64 *z_data = NULL;
    i32 ldq = 1;
    i32 ldz = 1;

    if (updatq) {
        if (ljobqi) {
            npy_intp q_dims[2] = {m > 0 ? m : 1, m > 0 ? m : 1};
            q_array = (PyArrayObject *)PyArray_ZEROS(2, q_dims, NPY_DOUBLE, 1);
            if (q_array == NULL) {
                Py_DECREF(a_array);
                Py_DECREF(e_array);
                Py_DECREF(istair_array);
                return PyErr_NoMemory();
            }
        } else {
            q_array = (PyArrayObject *)PyArray_FROM_OTF(
                q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            if (q_array == NULL) {
                Py_DECREF(a_array);
                Py_DECREF(e_array);
                Py_DECREF(istair_array);
                return NULL;
            }
        }
        q_data = (f64 *)PyArray_DATA(q_array);
        ldq = (m > 0) ? (i32)PyArray_DIM(q_array, 0) : 1;
    } else {
        npy_intp q_dims[2] = {1, 1};
        q_array = (PyArrayObject *)PyArray_ZEROS(2, q_dims, NPY_DOUBLE, 1);
        if (q_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(istair_array);
            return PyErr_NoMemory();
        }
        q_data = (f64 *)PyArray_DATA(q_array);
        ldq = 1;
    }

    if (updatz) {
        if (ljobzi) {
            npy_intp z_dims[2] = {n > 0 ? n : 1, n > 0 ? n : 1};
            z_array = (PyArrayObject *)PyArray_ZEROS(2, z_dims, NPY_DOUBLE, 1);
            if (z_array == NULL) {
                Py_DECREF(a_array);
                Py_DECREF(e_array);
                Py_DECREF(istair_array);
                Py_DECREF(q_array);
                return PyErr_NoMemory();
            }
        } else {
            z_array = (PyArrayObject *)PyArray_FROM_OTF(
                z_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            if (z_array == NULL) {
                Py_DECREF(a_array);
                Py_DECREF(e_array);
                Py_DECREF(istair_array);
                Py_DECREF(q_array);
                return NULL;
            }
        }
        z_data = (f64 *)PyArray_DATA(z_array);
        ldz = (n > 0) ? (i32)PyArray_DIM(z_array, 0) : 1;
    } else {
        npy_intp z_dims[2] = {1, 1};
        z_array = (PyArrayObject *)PyArray_ZEROS(2, z_dims, NPY_DOUBLE, 1);
        if (z_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(e_array);
            Py_DECREF(istair_array);
            Py_DECREF(q_array);
            return PyErr_NoMemory();
        }
        z_data = (f64 *)PyArray_DATA(z_array);
        ldz = 1;
    }

    i32 max_mn = (m > n) ? m : n;
    i32 imuk_size = (max_mn > m + 1) ? max_mn : (m + 1);
    if (imuk_size < 1) imuk_size = 1;
    i32 imuk0_size = lmodes ? (n > 1 ? n : 1) : 1;
    i32 iwork_size = (n > 0) ? n : 1;

    i32 *imuk = (i32 *)PyMem_Calloc(imuk_size, sizeof(i32));
    i32 *inuk = (i32 *)PyMem_Calloc(imuk_size, sizeof(i32));
    i32 *imuk0 = (i32 *)PyMem_Calloc(imuk0_size, sizeof(i32));
    i32 *iwork = (i32 *)PyMem_Calloc(iwork_size, sizeof(i32));

    if (!imuk || !inuk || !imuk0 || !iwork) {
        PyMem_Free(imuk); PyMem_Free(inuk);
        PyMem_Free(imuk0); PyMem_Free(iwork);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(istair_array);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        return PyErr_NoMemory();
    }

    f64 *a_data = (f64 *)PyArray_DATA(a_array);
    f64 *e_data = (f64 *)PyArray_DATA(e_array);
    i32 *istair_data = (i32 *)PyArray_DATA(istair_array);

    i32 nblcks = 0;
    i32 nblcki = 0;
    i32 mnei[3] = {0, 0, 0};
    i32 info = 0;

    mb04vd(mode_str, jobq_str, jobz_str, m, n, ranke,
           a_data, lda, e_data, lde,
           q_data, ldq, z_data, ldz,
           istair_data, &nblcks, &nblcki,
           imuk, inuk, imuk0, mnei,
           tol, iwork, &info);

    PyMem_Free(iwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(istair_array);
    if (updatq && !ljobqi) {
        PyArray_ResolveWritebackIfCopy(q_array);
    }
    if (updatz && !ljobzi) {
        PyArray_ResolveWritebackIfCopy(z_array);
    }

    npy_intp imuk_dims[1] = {imuk_size};
    PyObject *imuk_array = PyArray_SimpleNew(1, imuk_dims, NPY_INT32);
    memcpy(PyArray_DATA((PyArrayObject *)imuk_array), imuk, imuk_size * sizeof(i32));
    PyMem_Free(imuk);

    npy_intp inuk_dims[1] = {imuk_size};
    PyObject *inuk_array = PyArray_SimpleNew(1, inuk_dims, NPY_INT32);
    memcpy(PyArray_DATA((PyArrayObject *)inuk_array), inuk, imuk_size * sizeof(i32));
    PyMem_Free(inuk);

    npy_intp imuk0_dims[1] = {imuk0_size};
    PyObject *imuk0_array = PyArray_SimpleNew(1, imuk0_dims, NPY_INT32);
    memcpy(PyArray_DATA((PyArrayObject *)imuk0_array), imuk0, imuk0_size * sizeof(i32));
    PyMem_Free(imuk0);

    npy_intp mnei_dims[1] = {3};
    PyObject *mnei_array = PyArray_SimpleNew(1, mnei_dims, NPY_INT32);
    memcpy(PyArray_DATA((PyArrayObject *)mnei_array), mnei, 3 * sizeof(i32));

    PyObject *q_result = updatq ? (PyObject *)q_array : Py_None;
    PyObject *z_result = updatz ? (PyObject *)z_array : Py_None;

    if (!updatq) Py_INCREF(Py_None);
    if (!updatz) Py_INCREF(Py_None);

    PyObject *result = Py_BuildValue("OOOOiiOOOOi",
                                     a_array, e_array, q_result, z_result,
                                     nblcks, nblcki,
                                     imuk_array, inuk_array, imuk0_array, mnei_array,
                                     info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    if (updatq) Py_DECREF(q_array);
    if (updatz) Py_DECREF(z_array);
    Py_DECREF(imuk_array);
    Py_DECREF(inuk_array);
    Py_DECREF(imuk0_array);
    Py_DECREF(mnei_array);

    return result;
}

PyObject* py_mb04wp(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"n", "ilo", "u1", "u2", "cs", "tau", NULL};

    i32 n, ilo;
    PyObject *u1_obj, *u2_obj, *cs_obj, *tau_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiOOOO", kwlist,
            &n, &ilo, &u1_obj, &u2_obj, &cs_obj, &tau_obj)) {
        return NULL;
    }

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be >= 0");
        return NULL;
    }
    i32 max_1_n = (n > 1) ? n : 1;
    if (ilo < 1 || ilo > max_1_n) {
        PyErr_SetString(PyExc_ValueError, "ilo must satisfy 1 <= ilo <= max(1, n)");
        return NULL;
    }

    PyArrayObject *u1_array = (PyArrayObject *)PyArray_FROM_OTF(
        u1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (u1_array == NULL) {
        return NULL;
    }

    PyArrayObject *u2_array = (PyArrayObject *)PyArray_FROM_OTF(
        u2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (u2_array == NULL) {
        Py_DECREF(u1_array);
        return NULL;
    }

    PyArrayObject *cs_array = (PyArrayObject *)PyArray_FROM_OTF(
        cs_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (cs_array == NULL) {
        Py_DECREF(u1_array);
        Py_DECREF(u2_array);
        return NULL;
    }

    PyArrayObject *tau_array = (PyArrayObject *)PyArray_FROM_OTF(
        tau_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (tau_array == NULL) {
        Py_DECREF(u1_array);
        Py_DECREF(u2_array);
        Py_DECREF(cs_array);
        return NULL;
    }

    i32 ldu1 = (i32)PyArray_DIM(u1_array, 0);
    i32 ldu2 = (i32)PyArray_DIM(u2_array, 0);
    if (ldu1 < 1) ldu1 = 1;
    if (ldu2 < 1) ldu2 = 1;

    i32 nh = n - ilo;
    i32 minwrk = (1 > 2 * nh) ? 1 : 2 * nh;
    i32 nb = 32;
    i32 opt2 = 8 * nh * nb + 15 * nb * nb;
    i32 ldwork = (minwrk > opt2) ? minwrk : opt2;
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(u1_array);
        Py_DECREF(u2_array);
        Py_DECREF(cs_array);
        Py_DECREF(tau_array);
        return PyErr_NoMemory();
    }

    f64 *u1_data = (f64 *)PyArray_DATA(u1_array);
    f64 *u2_data = (f64 *)PyArray_DATA(u2_array);
    f64 *cs_data = (f64 *)PyArray_DATA(cs_array);
    f64 *tau_data = (f64 *)PyArray_DATA(tau_array);

    i32 info = 0;
    mb04wp(n, ilo, u1_data, ldu1, u2_data, ldu2,
           cs_data, tau_data, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(u1_array);
    PyArray_ResolveWritebackIfCopy(u2_array);

    PyObject *result = Py_BuildValue("OOi", u1_array, u2_array, info);

    Py_DECREF(u1_array);
    Py_DECREF(u2_array);
    Py_DECREF(cs_array);
    Py_DECREF(tau_array);

    return result;
}

PyObject* py_mb04wr(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"job", "trans", "n", "ilo", "q1", "q2", "cs", "tau", NULL};

    const char *job, *trans;
    i32 n, ilo;
    PyObject *q1_obj, *q2_obj, *cs_obj, *tau_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssiiOOOO", kwlist,
            &job, &trans, &n, &ilo, &q1_obj, &q2_obj, &cs_obj, &tau_obj)) {
        return NULL;
    }

    if (!(job[0] == 'U' || job[0] == 'u' || job[0] == 'V' || job[0] == 'v')) {
        PyErr_SetString(PyExc_ValueError, "job must be 'U' or 'V'");
        return NULL;
    }

    if (!(trans[0] == 'N' || trans[0] == 'n' || trans[0] == 'T' || trans[0] == 't' ||
          trans[0] == 'C' || trans[0] == 'c')) {
        PyErr_SetString(PyExc_ValueError, "trans must be 'N', 'T', or 'C'");
        return NULL;
    }

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be >= 0");
        return NULL;
    }
    i32 max_1_n = (n > 1) ? n : 1;
    if (ilo < 1 || ilo > max_1_n) {
        PyErr_SetString(PyExc_ValueError, "ilo must satisfy 1 <= ilo <= max(1, n)");
        return NULL;
    }

    PyArrayObject *q1_array = (PyArrayObject *)PyArray_FROM_OTF(
        q1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (q1_array == NULL) {
        return NULL;
    }

    PyArrayObject *q2_array = (PyArrayObject *)PyArray_FROM_OTF(
        q2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (q2_array == NULL) {
        Py_DECREF(q1_array);
        return NULL;
    }

    PyArrayObject *cs_array = (PyArrayObject *)PyArray_FROM_OTF(
        cs_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (cs_array == NULL) {
        Py_DECREF(q1_array);
        Py_DECREF(q2_array);
        return NULL;
    }

    PyArrayObject *tau_array = (PyArrayObject *)PyArray_FROM_OTF(
        tau_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (tau_array == NULL) {
        Py_DECREF(q1_array);
        Py_DECREF(q2_array);
        Py_DECREF(cs_array);
        return NULL;
    }

    i32 ldq1 = (i32)PyArray_DIM(q1_array, 0);
    i32 ldq2 = (i32)PyArray_DIM(q2_array, 0);
    if (ldq1 < 1) ldq1 = 1;
    if (ldq2 < 1) ldq2 = 1;

    bool compu = (job[0] == 'U' || job[0] == 'u');
    i32 nh = compu ? (n - ilo + 1) : (n - ilo);
    i32 minwrk = (1 > 2 * nh) ? 1 : 2 * nh;
    i32 nb = 32;
    i32 opt2 = 8 * nh * nb + 15 * nb * nb;
    i32 ldwork = (minwrk > opt2) ? minwrk : opt2;
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(q1_array);
        Py_DECREF(q2_array);
        Py_DECREF(cs_array);
        Py_DECREF(tau_array);
        return PyErr_NoMemory();
    }

    f64 *q1_data = (f64 *)PyArray_DATA(q1_array);
    f64 *q2_data = (f64 *)PyArray_DATA(q2_array);
    f64 *cs_data = (f64 *)PyArray_DATA(cs_array);
    f64 *tau_data = (f64 *)PyArray_DATA(tau_array);

    i32 info = 0;
    mb04wr(job, trans, n, ilo, q1_data, ldq1, q2_data, ldq2,
           cs_data, tau_data, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(q1_array);
    PyArray_ResolveWritebackIfCopy(q2_array);

    PyObject *result = Py_BuildValue("OOi", q1_array, q2_array, info);

    Py_DECREF(q1_array);
    Py_DECREF(q2_array);
    Py_DECREF(cs_array);
    Py_DECREF(tau_array);

    return result;
}

PyObject *py_mb04yd(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"jobu", "jobv", "m", "n", "rank", "theta", "q", "e",
                             "tol", "reltol", "u", "v", NULL};

    const char *jobu, *jobv;
    int m, n, rank_in;
    double theta_in, tol, reltol;
    PyObject *q_obj, *e_obj;
    PyObject *u_obj = Py_None, *v_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssiiidOOdd|OO:mb04yd", kwlist,
                                     &jobu, &jobv, &m, &n, &rank_in, &theta_in,
                                     &q_obj, &e_obj, &tol, &reltol,
                                     &u_obj, &v_obj)) {
        return NULL;
    }

    bool ljobui = (jobu[0] == 'I' || jobu[0] == 'i');
    bool ljobuu = (jobu[0] == 'U' || jobu[0] == 'u');
    bool ljobvi = (jobv[0] == 'I' || jobv[0] == 'i');
    bool ljobvu = (jobv[0] == 'U' || jobv[0] == 'u');
    bool ljobua = ljobui || ljobuu;
    bool ljobva = ljobvi || ljobvu;

    if (!ljobua && !(jobu[0] == 'N' || jobu[0] == 'n')) {
        PyErr_SetString(PyExc_ValueError, "jobu must be 'N', 'I', or 'U'");
        return NULL;
    }
    if (!ljobva && !(jobv[0] == 'N' || jobv[0] == 'n')) {
        PyErr_SetString(PyExc_ValueError, "jobv must be 'N', 'I', or 'U'");
        return NULL;
    }

    i32 p = (m < n) ? m : n;

    int requirements = NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY;

    PyArrayObject *q_array = (PyArrayObject *)PyArray_FROM_OTF(
        q_obj, NPY_DOUBLE, requirements);
    if (q_array == NULL) return NULL;

    PyArrayObject *e_array = NULL;
    if (p > 1) {
        e_array = (PyArrayObject *)PyArray_FROM_OTF(
            e_obj, NPY_DOUBLE, requirements);
        if (e_array == NULL) {
            Py_DECREF(q_array);
            return NULL;
        }
    }

    PyArrayObject *u_array = NULL;
    PyArrayObject *v_array = NULL;

    if (ljobuu) {
        if (u_obj == Py_None) {
            PyErr_SetString(PyExc_ValueError, "u must be provided when jobu='U'");
            Py_DECREF(q_array);
            Py_XDECREF(e_array);
            return NULL;
        }
        u_array = (PyArrayObject *)PyArray_FROM_OTF(
            u_obj, NPY_DOUBLE, requirements);
        if (u_array == NULL) {
            Py_DECREF(q_array);
            Py_XDECREF(e_array);
            return NULL;
        }
    } else if (ljobui) {
        npy_intp u_dims[2] = {m, p};
        npy_intp u_strides[2] = {sizeof(f64), m * sizeof(f64)};
        u_array = (PyArrayObject *)PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE,
                                                u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (u_array == NULL) {
            Py_DECREF(q_array);
            Py_XDECREF(e_array);
            PyErr_NoMemory();
            return NULL;
        }
        size_t u_size = (m > 0 && p > 0) ? (size_t)m * p : 0;
        if (u_size > 0) memset(PyArray_DATA(u_array), 0, u_size * sizeof(f64));
    }

    if (ljobvu) {
        if (v_obj == Py_None) {
            PyErr_SetString(PyExc_ValueError, "v must be provided when jobv='U'");
            Py_DECREF(q_array);
            Py_XDECREF(e_array);
            Py_XDECREF(u_array);
            return NULL;
        }
        v_array = (PyArrayObject *)PyArray_FROM_OTF(
            v_obj, NPY_DOUBLE, requirements);
        if (v_array == NULL) {
            Py_DECREF(q_array);
            Py_XDECREF(e_array);
            Py_XDECREF(u_array);
            return NULL;
        }
    } else if (ljobvi) {
        npy_intp v_dims[2] = {n, p};
        npy_intp v_strides[2] = {sizeof(f64), n * sizeof(f64)};
        v_array = (PyArrayObject *)PyArray_New(&PyArray_Type, 2, v_dims, NPY_DOUBLE,
                                                v_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (v_array == NULL) {
            Py_DECREF(q_array);
            Py_XDECREF(e_array);
            Py_XDECREF(u_array);
            PyErr_NoMemory();
            return NULL;
        }
        size_t v_size = (n > 0 && p > 0) ? (size_t)n * p : 0;
        if (v_size > 0) memset(PyArray_DATA(v_array), 0, v_size * sizeof(f64));
    }

    i32 ldu = 1;
    if (ljobua) {
        ldu = (m > 0) ? m : 1;
    }
    i32 ldv = 1;
    if (ljobva) {
        ldv = (n > 0) ? n : 1;
    }

    i32 ldwork;
    if (ljobua || ljobva) {
        ldwork = (p > 0) ? (6 * p - 5) : 1;
    } else {
        ldwork = (p > 0) ? (4 * p - 3) : 1;
    }
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));
    bool *inul = (bool *)calloc((p > 0) ? p : 1, sizeof(bool));
    if (dwork == NULL || inul == NULL) {
        free(dwork);
        free(inul);
        Py_DECREF(q_array);
        Py_XDECREF(e_array);
        Py_XDECREF(u_array);
        Py_XDECREF(v_array);
        return PyErr_NoMemory();
    }

    for (i32 i = 0; i < p; i++) {
        inul[i] = false;
    }

    f64 *q_data = (f64 *)PyArray_DATA(q_array);
    f64 *e_data = (e_array != NULL) ? (f64 *)PyArray_DATA(e_array) : NULL;
    f64 *u_data = (u_array != NULL) ? (f64 *)PyArray_DATA(u_array) : NULL;
    f64 *v_data = (v_array != NULL) ? (f64 *)PyArray_DATA(v_array) : NULL;

    f64 dummy_arr[1] = {0.0};
    if (e_data == NULL) e_data = dummy_arr;
    if (u_data == NULL) u_data = dummy_arr;
    if (v_data == NULL) v_data = dummy_arr;

    i32 rank_out = rank_in;
    f64 theta_out = theta_in;
    i32 iwarn = 0;
    i32 info = 0;

    mb04yd(jobu, jobv, m, n, &rank_out, &theta_out, q_data, e_data,
           u_data, ldu, v_data, ldv, inul, tol, reltol, dwork, ldwork,
           &iwarn, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(q_array);
    if (e_array != NULL) {
        PyArray_ResolveWritebackIfCopy(e_array);
    }
    if (ljobuu && u_array != NULL) {
        PyArray_ResolveWritebackIfCopy(u_array);
    }
    if (ljobvu && v_array != NULL) {
        PyArray_ResolveWritebackIfCopy(v_array);
    }

    npy_intp inul_dims[1] = {(p > 0) ? p : 0};
    PyObject *inul_array = PyArray_SimpleNew(1, inul_dims, NPY_BOOL);
    if (p > 0) {
        npy_bool *inul_dest = (npy_bool *)PyArray_DATA((PyArrayObject *)inul_array);
        for (i32 i = 0; i < p; i++) {
            inul_dest[i] = inul[i] ? NPY_TRUE : NPY_FALSE;
        }
    }
    free(inul);

    PyObject *e_ret = (e_array != NULL) ? (PyObject *)e_array : Py_None;
    if (e_ret == Py_None) Py_INCREF(Py_None);

    PyObject *result = NULL;
    if (ljobua && ljobva) {
        result = Py_BuildValue("OOOOdiOii", q_array, e_ret, u_array, v_array,
                               theta_out, rank_out, inul_array, iwarn, info);
    } else if (ljobua) {
        result = Py_BuildValue("OOOdiOii", q_array, e_ret, u_array,
                               theta_out, rank_out, inul_array, iwarn, info);
    } else if (ljobva) {
        result = Py_BuildValue("OOOdiOii", q_array, e_ret, v_array,
                               theta_out, rank_out, inul_array, iwarn, info);
    } else {
        result = Py_BuildValue("OOdiOii", q_array, e_ret, theta_out, rank_out,
                               inul_array, iwarn, info);
    }

    Py_DECREF(q_array);
    if (e_array != NULL) Py_DECREF(e_array);
    else if (e_ret != Py_None) Py_DECREF(e_ret);
    Py_XDECREF(u_array);
    Py_XDECREF(v_array);
    Py_DECREF(inul_array);

    return result;
}

/*
 * Python wrapper for MB4DBZ - Inverse balancing transformation for complex
 * skew-Hamiltonian/Hamiltonian eigenvectors.
 */
PyObject *py_mb4dbz(PyObject *self, PyObject *args) {
    const char *job;
    const char *sgn;
    int n_in, ilo_in;
    PyObject *lscale_obj, *rscale_obj, *v1_obj, *v2_obj;

    if (!PyArg_ParseTuple(args, "ssiiOOOO", &job, &sgn, &n_in, &ilo_in,
                          &lscale_obj, &rscale_obj, &v1_obj, &v2_obj)) {
        return NULL;
    }

    i32 n = (i32)n_in;
    i32 ilo = (i32)ilo_in;

    PyArrayObject *lscale_array = (PyArrayObject *)PyArray_FROM_OTF(
        lscale_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (lscale_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "lscale must be a valid array");
        return NULL;
    }

    PyArrayObject *rscale_array = (PyArrayObject *)PyArray_FROM_OTF(
        rscale_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (rscale_array == NULL) {
        Py_DECREF(lscale_array);
        PyErr_SetString(PyExc_ValueError, "rscale must be a valid array");
        return NULL;
    }

    PyArrayObject *v1_array = (PyArrayObject *)PyArray_FROM_OTF(
        v1_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (v1_array == NULL) {
        Py_DECREF(lscale_array);
        Py_DECREF(rscale_array);
        PyErr_SetString(PyExc_ValueError, "v1 must be a valid complex array");
        return NULL;
    }

    PyArrayObject *v2_array = (PyArrayObject *)PyArray_FROM_OTF(
        v2_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (v2_array == NULL) {
        Py_DECREF(lscale_array);
        Py_DECREF(rscale_array);
        Py_DECREF(v1_array);
        PyErr_SetString(PyExc_ValueError, "v2 must be a valid complex array");
        return NULL;
    }

    int ndim_v1 = PyArray_NDIM(v1_array);
    int ndim_v2 = PyArray_NDIM(v2_array);

    i32 m;
    if (ndim_v1 == 2 && ndim_v2 == 2) {
        m = (i32)PyArray_DIM(v1_array, 1);
    } else if (n == 0) {
        m = 0;
    } else {
        Py_DECREF(lscale_array);
        Py_DECREF(rscale_array);
        Py_DECREF(v1_array);
        Py_DECREF(v2_array);
        PyErr_SetString(PyExc_ValueError, "v1 and v2 must be 2D arrays");
        return NULL;
    }

    i32 ldv1 = (n > 0) ? (i32)PyArray_DIM(v1_array, 0) : 1;
    i32 ldv2 = (n > 0) ? (i32)PyArray_DIM(v2_array, 0) : 1;

    f64 *lscale = (f64 *)PyArray_DATA(lscale_array);
    f64 *rscale = (f64 *)PyArray_DATA(rscale_array);
    c128 *v1_data = (c128 *)PyArray_DATA(v1_array);
    c128 *v2_data = (c128 *)PyArray_DATA(v2_array);

    i32 info = 0;

    mb4dbz(job, sgn, n, ilo, lscale, rscale, m, v1_data, ldv1, v2_data, ldv2, &info);

    if (info < 0) {
        Py_DECREF(lscale_array);
        Py_DECREF(rscale_array);
        PyArray_DiscardWritebackIfCopy(v1_array);
        PyArray_DiscardWritebackIfCopy(v2_array);
        Py_DECREF(v1_array);
        Py_DECREF(v2_array);
        PyErr_Format(PyExc_ValueError, "mb4dbz: argument %d had illegal value", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(v1_array);
    PyArray_ResolveWritebackIfCopy(v2_array);

    PyObject *result = Py_BuildValue("OOi", v1_array, v2_array, info);

    Py_DECREF(lscale_array);
    Py_DECREF(rscale_array);
    Py_DECREF(v1_array);
    Py_DECREF(v2_array);

    return result;
}

/* Python wrapper for mb4dlz - balance complex pencil (A,B) */
PyObject *py_mb4dlz(PyObject *self, PyObject *args) {
    const char *job;
    int n_in;
    double thresh;
    PyObject *a_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "sidOO", &job, &n_in, &thresh, &a_obj, &b_obj)) {
        return NULL;
    }

    i32 n = (i32)n_in;

    // Convert input arrays
    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "a must be a valid complex array");
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "b must be a valid complex array");
        return NULL;
    }

    // Get leading dimensions
    i32 lda = (n > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 ldb = (n > 0) ? (i32)PyArray_DIM(b_array, 0) : 1;

    c128 *a_data = (c128 *)PyArray_DATA(a_array);
    c128 *b_data = (c128 *)PyArray_DATA(b_array);

    // Allocate output arrays
    npy_intp scale_dims[1] = {n > 0 ? n : 0};
    PyArrayObject *lscale_array = (PyArrayObject *)PyArray_New(
        &PyArray_Type, 1, scale_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
    PyArrayObject *rscale_array = (PyArrayObject *)PyArray_New(
        &PyArray_Type, 1, scale_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);

    if (lscale_array == NULL || rscale_array == NULL) {
        Py_XDECREF(lscale_array);
        Py_XDECREF(rscale_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        PyErr_NoMemory();
        return NULL;
    }
    f64 *lscale = (f64 *)PyArray_DATA(lscale_array);
    f64 *rscale = (f64 *)PyArray_DATA(rscale_array);
    if (n > 0) {
        memset(lscale, 0, n * sizeof(f64));
        memset(rscale, 0, n * sizeof(f64));
    }

    // Determine workspace size
    bool do_scale = (*job == 'S' || *job == 's' || *job == 'B' || *job == 'b');
    i32 ldwork;
    if (!do_scale || n == 0) {
        ldwork = 1;
    } else if (thresh >= 0.0) {
        ldwork = 6 * n;
    } else {
        ldwork = 9 * n;
    }
    f64 *dwork = (f64 *)calloc(ldwork > 0 ? ldwork : 1, sizeof(f64));

    if (dwork == NULL) {
        Py_DECREF(lscale_array);
        Py_DECREF(rscale_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 ilo = 0, ihi = 0, iwarn = 0, info = 0;

    mb4dlz(job, n, thresh, a_data, lda, b_data, ldb, &ilo, &ihi,
           lscale, rscale, dwork, &iwarn, &info);

    if (info < 0) {
        free(dwork);
        Py_DECREF(lscale_array);
        Py_DECREF(rscale_array);
        PyArray_DiscardWritebackIfCopy(a_array);
        PyArray_DiscardWritebackIfCopy(b_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        PyErr_Format(PyExc_ValueError, "mb4dlz: argument %d had illegal value", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    // Create output array for dwork (first 5 elements contain useful info)
    i32 dwork_out_len = do_scale && n > 0 ? 5 : 0;

    PyArrayObject *dwork_array;
    if (dwork_out_len > 0) {
        npy_intp dwork_dims[1] = {dwork_out_len};
        dwork_array = (PyArrayObject *)PyArray_New(
            &PyArray_Type, 1, dwork_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
        if (dwork_array == NULL) {
            free(dwork);
            Py_DECREF(lscale_array);
            Py_DECREF(rscale_array);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            PyErr_NoMemory();
            return NULL;
        }
        f64 *dwork_out = (f64 *)PyArray_DATA(dwork_array);
        for (i32 i = 0; i < dwork_out_len; i++) {
            dwork_out[i] = dwork[i];
        }
        free(dwork);
    } else {
        free(dwork);
        npy_intp dwork_dims[1] = {0};
        dwork_array = (PyArrayObject *)PyArray_EMPTY(1, dwork_dims, NPY_DOUBLE, 0);
    }

    PyObject *result = Py_BuildValue("OOiiOOOii",
                                     a_array, b_array, ilo, ihi,
                                     lscale_array, rscale_array, dwork_array,
                                     iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(lscale_array);
    Py_DECREF(rscale_array);
    Py_DECREF(dwork_array);

    return result;
}

/* Python wrapper for mb04xd */
PyObject* py_mb04xd(PyObject* self, PyObject* args) {
    const char *jobu_str, *jobv_str;
    PyObject *a_obj;
    i32 rank_in;
    f64 theta_in, tol, reltol;

    if (!PyArg_ParseTuple(args, "ssOiddd",
                          &jobu_str, &jobv_str, &a_obj,
                          &rank_in, &theta_in, &tol, &reltol)) {
        return NULL;
    }

    char jobu = jobu_str[0];
    char jobv = jobv_str[0];

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "a must be a valid array");
        return NULL;
    }

    if (PyArray_NDIM(a_array) != 2) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "a must be a 2D array");
        return NULL;
    }

    i32 m = (i32)PyArray_DIM(a_array, 0);
    i32 n = (i32)PyArray_DIM(a_array, 1);
    i32 lda = (m > 0) ? m : 1;
    i32 p = (m < n) ? m : n;
    i32 k = (m > n) ? m : n;

    f64 *a_data = (f64 *)PyArray_DATA(a_array);

    bool ljobua = (jobu == 'A' || jobu == 'a');
    bool ljobus = (jobu == 'S' || jobu == 's');
    bool ljobva = (jobv == 'A' || jobv == 'a');
    bool ljobvs = (jobv == 'S' || jobv == 's');
    bool wantu = ljobua || ljobus;
    bool wantv = ljobva || ljobvs;

    i32 ldu, ldv, u_cols, v_cols;
    f64 *u_data = NULL, *v_data = NULL;
    PyArrayObject *u_array = NULL, *v_array = NULL;

    if (wantu) {
        ldu = (m > 0) ? m : 1;
        u_cols = ljobua ? m : p;
        npy_intp u_dims[2] = {m, u_cols};
        npy_intp u_strides[2] = {sizeof(f64), m * sizeof(f64)};
        u_array = (PyArrayObject *)PyArray_New(
            &PyArray_Type, 2, u_dims, NPY_DOUBLE, u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (u_array == NULL) {
            Py_DECREF(a_array);
            PyErr_NoMemory();
            return NULL;
        }
        u_data = (f64 *)PyArray_DATA(u_array);
        size_t u_size = (size_t)m * (size_t)u_cols;
        if (u_size > 0) memset(u_data, 0, u_size * sizeof(f64));
    } else {
        ldu = 1;
        u_cols = 0;
    }

    if (wantv) {
        ldv = (n > 0) ? n : 1;
        v_cols = ljobva ? n : p;
        npy_intp v_dims[2] = {n, v_cols};
        npy_intp v_strides[2] = {sizeof(f64), n * sizeof(f64)};
        v_array = (PyArrayObject *)PyArray_New(
            &PyArray_Type, 2, v_dims, NPY_DOUBLE, v_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (v_array == NULL) {
            Py_DECREF(a_array);
            if (wantu) { Py_DECREF(u_array); }
            PyErr_NoMemory();
            return NULL;
        }
        v_data = (f64 *)PyArray_DATA(v_array);
        size_t v_size = (size_t)n * (size_t)v_cols;
        if (v_size > 0) memset(v_data, 0, v_size * sizeof(f64));
    } else {
        ldv = 1;
        v_cols = 0;
    }

    i32 q_len = (p > 0) ? (2 * p - 1) : 1;
    npy_intp q_dims[1] = {q_len};
    PyArrayObject *q_array = (PyArrayObject *)PyArray_New(
        &PyArray_Type, 1, q_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
    if (q_array == NULL) {
        Py_DECREF(a_array);
        if (wantu) { Py_DECREF(u_array); }
        if (wantv) { Py_DECREF(v_array); }
        PyErr_NoMemory();
        return NULL;
    }
    f64 *q_data = (f64 *)PyArray_DATA(q_array);
    if (q_len > 0) memset(q_data, 0, q_len * sizeof(f64));

    npy_intp inul_dims[1] = {k > 0 ? k : 1};
    PyArrayObject *inul_array = (PyArrayObject *)PyArray_New(
        &PyArray_Type, 1, inul_dims, NPY_BOOL, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
    if (inul_array == NULL) {
        Py_DECREF(a_array);
        if (wantu) { Py_DECREF(u_array); }
        if (wantv) { Py_DECREF(v_array); }
        Py_DECREF(q_array);
        PyErr_NoMemory();
        return NULL;
    }
    bool *inul_data = (bool *)PyArray_DATA(inul_array);
    if (k > 0) memset(inul_data, 0, (k > 0 ? k : 1) * sizeof(bool));

    i32 ldw, ldy;
    bool qr_mode = (m >= 6 * n && n > 0);
    if (qr_mode && wantu) {
        i32 t1 = 2 * n;
        i32 t2 = n * (n + 1) / 2;
        ldw = (t1 > t2) ? t1 : t2;
    } else {
        ldw = 0;
    }
    if (wantu || wantv) {
        ldy = 8 * p - 5;
    } else {
        ldy = 6 * p - 3;
    }
    i32 t1 = 2 * p + k;
    i32 t2 = (t1 > ldy) ? t1 : ldy;
    i32 ldwork = ldw + t2;
    if (ldwork < 1) ldwork = 1;
    ldwork = ldwork + ldw + 2 * p + 64;

    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        if (wantu) { Py_DECREF(u_array); }
        if (wantv) { Py_DECREF(v_array); }
        Py_DECREF(q_array);
        Py_DECREF(inul_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate DWORK");
        return NULL;
    }

    i32 rank = rank_in;
    f64 theta = theta_in;
    i32 iwarn = 0;
    i32 info = 0;

    f64 dummy_u, dummy_v;
    f64 *u_ptr = wantu ? u_data : &dummy_u;
    f64 *v_ptr = wantv ? v_data : &dummy_v;

    mb04xd(&jobu, &jobv, m, n, &rank, &theta, a_data, lda,
           u_ptr, ldu, v_ptr, ldv, q_data, inul_data,
           tol, reltol, dwork, ldwork, &iwarn, &info);

    free(dwork);

    PyArray_DiscardWritebackIfCopy(a_array);
    Py_DECREF(a_array);

    if (info < 0) {
        if (wantu) { Py_DECREF(u_array); }
        if (wantv) { Py_DECREF(v_array); }
        Py_DECREF(q_array);
        Py_DECREF(inul_array);
        PyErr_Format(PyExc_ValueError, "mb04xd: argument %d had illegal value", -info);
        return NULL;
    }

    PyObject *u_ret = wantu ? (PyObject *)u_array : Py_None;
    PyObject *v_ret = wantv ? (PyObject *)v_array : Py_None;
    if (!wantu) Py_INCREF(Py_None);
    if (!wantv) Py_INCREF(Py_None);

    PyObject *result = Py_BuildValue("idOOOOii",
                                     rank, theta, u_ret, v_ret, q_array, inul_array, iwarn, info);

    if (wantu) { Py_DECREF(u_array); }
    if (wantv) { Py_DECREF(v_array); }
    Py_DECREF(q_array);
    Py_DECREF(inul_array);

    return result;
}

/* Python wrapper for mb4dpz - Balance complex skew-Hamiltonian/Hamiltonian pencil */
PyObject *py_mb4dpz(PyObject *self, PyObject *args) {
    const char *job;
    int n_in;
    double thresh;
    PyObject *a_obj, *de_obj, *c_obj, *vw_obj;

    if (!PyArg_ParseTuple(args, "sidOOOO", &job, &n_in, &thresh,
                          &a_obj, &de_obj, &c_obj, &vw_obj)) {
        return NULL;
    }

    i32 n = (i32)n_in;

    // Convert input arrays
    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "a must be a valid complex array");
        return NULL;
    }

    PyArrayObject *de_array = (PyArrayObject *)PyArray_FROM_OTF(
        de_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (de_array == NULL) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "de must be a valid complex array");
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject *)PyArray_FROM_OTF(
        c_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        PyErr_SetString(PyExc_ValueError, "c must be a valid complex array");
        return NULL;
    }

    PyArrayObject *vw_array = (PyArrayObject *)PyArray_FROM_OTF(
        vw_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (vw_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_ValueError, "vw must be a valid complex array");
        return NULL;
    }

    // Get leading dimensions
    i32 lda = (n > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 ldde = (n > 0) ? (i32)PyArray_DIM(de_array, 0) : 1;
    i32 ldc = (n > 0) ? (i32)PyArray_DIM(c_array, 0) : 1;
    i32 ldvw = (n > 0) ? (i32)PyArray_DIM(vw_array, 0) : 1;

    c128 *a_data = (c128 *)PyArray_DATA(a_array);
    c128 *de_data = (c128 *)PyArray_DATA(de_array);
    c128 *c_data = (c128 *)PyArray_DATA(c_array);
    c128 *vw_data = (c128 *)PyArray_DATA(vw_array);

    // Allocate output arrays
    npy_intp scale_dims[1] = {n > 0 ? n : 0};
    PyArrayObject *lscale_array = (PyArrayObject *)PyArray_New(
        &PyArray_Type, 1, scale_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
    PyArrayObject *rscale_array = (PyArrayObject *)PyArray_New(
        &PyArray_Type, 1, scale_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);

    if (lscale_array == NULL || rscale_array == NULL) {
        Py_XDECREF(lscale_array);
        Py_XDECREF(rscale_array);
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(c_array);
        Py_DECREF(vw_array);
        PyErr_NoMemory();
        return NULL;
    }
    f64 *lscale = (f64 *)PyArray_DATA(lscale_array);
    f64 *rscale = (f64 *)PyArray_DATA(rscale_array);
    if (n > 0) {
        memset(lscale, 0, n * sizeof(f64));
        memset(rscale, 0, n * sizeof(f64));
    }

    // Determine workspace size
    bool do_scale = (*job == 'S' || *job == 's' || *job == 'B' || *job == 'b');
    i32 ldwork;
    if (!do_scale || n == 0) {
        ldwork = 1;
    } else if (thresh >= 0.0) {
        ldwork = 6 * n;
    } else {
        ldwork = 9 * n;
    }
    f64 *dwork = (f64 *)calloc(ldwork > 0 ? ldwork : 1, sizeof(f64));

    if (dwork == NULL) {
        Py_DECREF(lscale_array);
        Py_DECREF(rscale_array);
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(c_array);
        Py_DECREF(vw_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 ilo = 0, iwarn = 0, info = 0;

    mb4dpz(job, n, thresh, a_data, lda, de_data, ldde, c_data, ldc, vw_data, ldvw,
           &ilo, lscale, rscale, dwork, &iwarn, &info);

    if (info < 0) {
        free(dwork);
        Py_DECREF(lscale_array);
        Py_DECREF(rscale_array);
        PyArray_DiscardWritebackIfCopy(a_array);
        PyArray_DiscardWritebackIfCopy(de_array);
        PyArray_DiscardWritebackIfCopy(c_array);
        PyArray_DiscardWritebackIfCopy(vw_array);
        Py_DECREF(a_array);
        Py_DECREF(de_array);
        Py_DECREF(c_array);
        Py_DECREF(vw_array);
        PyErr_Format(PyExc_ValueError, "mb4dpz: argument %d had illegal value", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(de_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(vw_array);

    // Create output array for dwork (first 5 elements contain useful info)
    i32 dwork_out_len = do_scale && n > 0 ? 5 : 0;

    PyArrayObject *dwork_array;
    if (dwork_out_len > 0) {
        npy_intp dwork_dims[1] = {dwork_out_len};
        dwork_array = (PyArrayObject *)PyArray_New(
            &PyArray_Type, 1, dwork_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
        if (dwork_array == NULL) {
            free(dwork);
            Py_DECREF(lscale_array);
            Py_DECREF(rscale_array);
            Py_DECREF(a_array);
            Py_DECREF(de_array);
            Py_DECREF(c_array);
            Py_DECREF(vw_array);
            PyErr_NoMemory();
            return NULL;
        }
        f64 *dwork_out = (f64 *)PyArray_DATA(dwork_array);
        for (i32 i = 0; i < dwork_out_len; i++) {
            dwork_out[i] = dwork[i];
        }
        free(dwork);
    } else {
        free(dwork);
        npy_intp dwork_dims[1] = {0};
        dwork_array = (PyArrayObject *)PyArray_EMPTY(1, dwork_dims, NPY_DOUBLE, 0);
    }

    PyObject *result = Py_BuildValue("iOOOii",
                                     ilo,
                                     lscale_array, rscale_array, dwork_array,
                                     iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(de_array);
    Py_DECREF(c_array);
    Py_DECREF(vw_array);
    Py_DECREF(lscale_array);
    Py_DECREF(rscale_array);
    Py_DECREF(dwork_array);

    return result;
}

