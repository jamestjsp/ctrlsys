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

/* Python wrapper for mb01pd */
PyObject* py_mb01pd(PyObject* self, PyObject* args) {
    const char *scun_str, *type_str;
    i32 m, n, kl, ku, nbl, lda;
    f64 anrm;
    PyObject *a_obj, *nrows_obj = NULL;
    PyArrayObject *a_array, *nrows_array = NULL;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssiiiidiOO",
                          &scun_str, &type_str, &m, &n, &kl, &ku, &anrm,
                          &nbl, &nrows_obj, &a_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    lda = (i32)a_dims[0];

    i32 *nrows_ptr = NULL;
    if (nrows_obj != NULL && nrows_obj != Py_None && nbl > 0) {
        nrows_array = (PyArrayObject*)PyArray_FROM_OTF(nrows_obj, NPY_INT32,
                                                       NPY_ARRAY_IN_ARRAY);
        if (nrows_array == NULL) {
            Py_DECREF(a_array);
            return NULL;
        }
        nrows_ptr = (i32*)PyArray_DATA(nrows_array);
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    mb01pd(scun_str, type_str, m, n, kl, ku, anrm, nbl, nrows_ptr, a_data, lda, &info);

    if (nrows_array != NULL) {
        Py_DECREF(nrows_array);
    }

    PyArray_ResolveWritebackIfCopy(a_array);
    PyObject *result = Py_BuildValue("Oi", a_array, info);
    Py_DECREF(a_array);
    return result;
}



/* Python wrapper for mb01qd */
PyObject* py_mb01qd(PyObject* self, PyObject* args) {
    char type;
    i32 m, n, kl, ku, nbl, lda;
    f64 cfrom, cto;
    PyObject *a_obj, *nrows_obj = NULL;
    PyArrayObject *a_array, *nrows_array = NULL;
    i32 info;

    if (!PyArg_ParseTuple(args, "ciiiiddO|O",
                          &type, &m, &n, &kl, &ku, &cfrom, &cto,
                          &a_obj, &nrows_obj)) {
        return NULL;
    }

    /* Convert to NumPy arrays - preserve Fortran-order (column-major) */
    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    /* Extract leading dimension from array shape */
    npy_intp *a_dims = PyArray_DIMS(a_array);
    lda = (i32)a_dims[0];

    /* Handle optional nrows parameter */
    i32 *nrows_ptr = NULL;
    if (nrows_obj != NULL && nrows_obj != Py_None) {
        nrows_array = (PyArrayObject*)PyArray_FROM_OTF(nrows_obj, NPY_INT32,
                                                       NPY_ARRAY_IN_ARRAY);
        if (nrows_array == NULL) {
            Py_DECREF(a_array);
            return NULL;
        }
        nrows_ptr = (i32*)PyArray_DATA(nrows_array);
        nbl = (i32)PyArray_SIZE(nrows_array);
    } else {
        nbl = 0;
    }

    /* Call C function */
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    mb01qd(type, m, n, kl, ku, cfrom, cto, nbl, nrows_ptr, a_data, lda, &info);

    /* Clean up and return */
    if (nrows_array != NULL) {
        Py_DECREF(nrows_array);
    }

    PyArray_ResolveWritebackIfCopy(a_array);
    PyObject *result = Py_BuildValue("Oi", a_array, info);
    Py_DECREF(a_array);
    return result;
}



/* Python wrapper for mb01rx */
PyObject* py_mb01rx(PyObject* self, PyObject* args) {
    const char *side_str, *uplo_str, *trans_str;
    char side, uplo, trans;
    i32 m, n, ldr, lda, ldb;
    f64 alpha, beta;
    PyObject *r_obj, *a_obj, *b_obj;
    PyArrayObject *r_array, *a_array, *b_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "sssiiddOOO",
                          &side_str, &uplo_str, &trans_str, &m, &n, &alpha, &beta,
                          &r_obj, &a_obj, &b_obj)) {
        return NULL;
    }

    side = side_str[0];
    uplo = uplo_str[0];
    trans = trans_str[0];

    /* Convert to NumPy arrays - preserve Fortran-order (column-major) */
    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        Py_DECREF(r_array);
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (b_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(a_array);
        return NULL;
    }

    /* Extract leading dimensions from array shapes */
    npy_intp *r_dims = PyArray_DIMS(r_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);

    ldr = (i32)r_dims[0];
    lda = (i32)a_dims[0];
    ldb = (i32)b_dims[0];

    /* Call C function */
    f64 *r_data = (f64*)PyArray_DATA(r_array);
    const f64 *a_data = (const f64*)PyArray_DATA(a_array);
    const f64 *b_data = (const f64*)PyArray_DATA(b_array);

    info = slicot_mb01rx(side, uplo, trans, m, n, alpha, beta,
                         r_data, ldr, a_data, lda, b_data, ldb);

    /* Clean up and return */
    Py_DECREF(a_array);
    Py_DECREF(b_array);

    PyArray_ResolveWritebackIfCopy(r_array);
    PyObject *result = Py_BuildValue("Oi", r_array, info);
    Py_DECREF(r_array);
    return result;
}



/* Python wrapper for mb01ru */
PyObject* py_mb01ru(PyObject* self, PyObject* args) {
    const char *uplo_str, *trans_str;
    i32 m, n;
    f64 alpha, beta;
    PyObject *r_obj, *a_obj, *x_obj;
    PyArrayObject *r_array, *a_array, *x_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssiiddOOO",
                          &uplo_str, &trans_str, &m, &n, &alpha, &beta,
                          &r_obj, &a_obj, &x_obj)) {
        return NULL;
    }

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        Py_DECREF(r_array);
        return NULL;
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (x_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *r_dims = PyArray_DIMS(r_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *x_dims = PyArray_DIMS(x_array);

    i32 ldr = (i32)r_dims[0];
    i32 lda = (i32)a_dims[0];
    i32 ldx = (i32)x_dims[0];
    if (ldr < 1) ldr = 1;
    if (lda < 1) lda = 1;
    if (ldx < 1) ldx = 1;
    i32 ldwork = m * n;

    f64 *r_data = (f64*)PyArray_DATA(r_array);
    const f64 *a_data = (const f64*)PyArray_DATA(a_array);
    f64 *x_data = (f64*)PyArray_DATA(x_array);

    f64 *dwork = NULL;
    if (ldwork > 0) {
        dwork = (f64*)malloc(ldwork * sizeof(f64));
        if (dwork == NULL) {
            Py_DECREF(r_array);
            Py_DECREF(a_array);
            Py_DECREF(x_array);
            PyErr_NoMemory();
            return NULL;
        }
    }

    mb01ru(uplo_str, trans_str, m, n, alpha, beta,
           r_data, ldr, a_data, lda, x_data, ldx, dwork, ldwork, &info);

    free(dwork);

    Py_DECREF(a_array);
    PyArray_ResolveWritebackIfCopy(x_array);
    Py_DECREF(x_array);

    PyArray_ResolveWritebackIfCopy(r_array);
    PyObject *result = Py_BuildValue("Oi", r_array, info);
    Py_DECREF(r_array);
    return result;
}



/* Python wrapper for mb01ry */
PyObject* py_mb01ry(PyObject* self, PyObject* args) {
    const char *side_str, *uplo_str, *trans_str;
    i32 m;
    f64 alpha, beta;
    PyObject *r_obj, *h_obj, *b_obj;
    PyArrayObject *r_array, *h_array, *b_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "sssiddOOO",
                          &side_str, &uplo_str, &trans_str, &m, &alpha, &beta,
                          &r_obj, &h_obj, &b_obj)) {
        return NULL;
    }

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) {
        return NULL;
    }

    h_array = (PyArrayObject*)PyArray_FROM_OTF(h_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (h_array == NULL) {
        Py_DECREF(r_array);
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (b_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(h_array);
        return NULL;
    }

    npy_intp *r_dims = PyArray_DIMS(r_array);
    npy_intp *h_dims = PyArray_DIMS(h_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);

    i32 ldr = (i32)r_dims[0];
    i32 ldh = (i32)h_dims[0];
    i32 ldb = (i32)b_dims[0];
    if (ldr < 1) ldr = 1;
    if (ldh < 1) ldh = 1;
    if (ldb < 1) ldb = 1;

    f64 *r_data = (f64*)PyArray_DATA(r_array);
    f64 *h_data = (f64*)PyArray_DATA(h_array);
    const f64 *b_data = (const f64*)PyArray_DATA(b_array);

    f64 *dwork = NULL;
    if (m > 0 && beta != 0.0 && side_str[0] == 'L') {
        dwork = (f64*)malloc(m * sizeof(f64));
        if (dwork == NULL) {
            Py_DECREF(r_array);
            Py_DECREF(h_array);
            Py_DECREF(b_array);
            PyErr_NoMemory();
            return NULL;
        }
    }

    mb01ry(side_str, uplo_str, trans_str, m, alpha, beta,
           r_data, ldr, h_data, ldh, b_data, ldb, dwork, &info);

    free(dwork);

    Py_DECREF(b_array);
    PyArray_ResolveWritebackIfCopy(h_array);
    Py_DECREF(h_array);

    PyArray_ResolveWritebackIfCopy(r_array);
    PyObject *result = Py_BuildValue("Oi", r_array, info);
    Py_DECREF(r_array);
    return result;
}



/* Python wrapper for mb01ud */
PyObject* py_mb01ud(PyObject* self, PyObject* args) {
    const char *side_str, *trans_str;
    i32 m, n;
    f64 alpha;
    PyObject *h_obj, *a_obj;
    PyArrayObject *h_array, *a_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssiidOO",
                          &side_str, &trans_str, &m, &n, &alpha,
                          &h_obj, &a_obj)) {
        return NULL;
    }

    h_array = (PyArrayObject*)PyArray_FROM_OTF(h_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (h_array == NULL) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        Py_DECREF(h_array);
        return NULL;
    }

    npy_intp *h_dims = PyArray_DIMS(h_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);

    i32 ldh = (i32)h_dims[0];
    i32 lda = (i32)a_dims[0];
    i32 ldb = (m > 1) ? m : 1;
    if (ldh < 1) ldh = 1;
    if (lda < 1) lda = 1;

    npy_intp b_dims[2] = {m, n};
    npy_intp b_strides[2] = {sizeof(f64), m * sizeof(f64)};

    PyArrayObject *b_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (b_array == NULL) {
        Py_DECREF(h_array);
        Py_DECREF(a_array);
        PyErr_NoMemory();
        return NULL;
    }
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    f64 *h_data = (f64*)PyArray_DATA(h_array);
    const f64 *a_data = (const f64*)PyArray_DATA(a_array);

    mb01ud(side_str, trans_str, m, n, alpha, h_data, ldh, a_data, lda, b_data, ldb, &info);

    PyArray_ResolveWritebackIfCopy(h_array);
    Py_DECREF(h_array);
    Py_DECREF(a_array);

    if (info != 0) {
        Py_DECREF(b_array);
        return Py_BuildValue("Oi", Py_None, info);
    }

    PyObject *result = Py_BuildValue("Oi", b_array, info);
    Py_DECREF(b_array);
    return result;
}



/* Python wrapper for mb01rb */
PyObject* py_mb01rb(PyObject* self, PyObject* args) {
    const char *side_str, *uplo_str, *trans_str;
    char side, uplo, trans;
    i32 m, n, ldr, lda, ldb;
    f64 alpha, beta;
    PyObject *r_obj, *a_obj, *b_obj;
    PyArrayObject *r_array, *a_array, *b_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "sssiiddOOO",
                          &side_str, &uplo_str, &trans_str, &m, &n, &alpha, &beta,
                          &r_obj, &a_obj, &b_obj)) {
        return NULL;
    }

    side = side_str[0];
    uplo = uplo_str[0];
    trans = trans_str[0];

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        Py_DECREF(r_array);
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (b_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *r_dims = PyArray_DIMS(r_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);

    ldr = (i32)r_dims[0];
    lda = (i32)a_dims[0];
    ldb = (i32)b_dims[0];

    f64 *r_data = (f64*)PyArray_DATA(r_array);
    const f64 *a_data = (const f64*)PyArray_DATA(a_array);
    const f64 *b_data = (const f64*)PyArray_DATA(b_array);

    mb01rb(&side, &uplo, &trans, m, n, alpha, beta,
           r_data, ldr, a_data, lda, b_data, ldb, &info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);

    if (info < 0) {
        PyArray_DiscardWritebackIfCopy(r_array);
        Py_DECREF(r_array);
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(r_array);
    PyObject *result = Py_BuildValue("Oi", r_array, info);
    Py_DECREF(r_array);
    return result;
}



/* Python wrapper for mb01td */
PyObject* py_mb01td(PyObject* self, PyObject* args) {
    i32 n, lda, ldb;
    PyObject *a_obj, *b_obj;
    PyArrayObject *a_array, *b_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
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

    n = (i32)a_dims[0];
    lda = n > 1 ? n : 1;
    ldb = n > 1 ? n : 1;

    const f64 *a_data = (const f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    f64 *dwork = NULL;
    if (n > 1) {
        dwork = (f64*)malloc((n - 1) * sizeof(f64));
        if (dwork == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            return PyErr_NoMemory();
        }
    }

    mb01td(n, a_data, lda, b_data, ldb, dwork, &info);

    free(dwork);
    Py_DECREF(a_array);

    if (info < 0) {
        PyArray_DiscardWritebackIfCopy(b_array);
        Py_DECREF(b_array);
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(b_array);
    PyObject *result = Py_BuildValue("Oi", b_array, info);
    Py_DECREF(b_array);
    return result;
}



/* Python wrapper for mb01rd */
PyObject* py_mb01rd(PyObject* self, PyObject* args) {
    const char *uplo_str, *trans_str;
    char uplo, trans;
    i32 m, n;
    f64 alpha, beta;
    PyObject *r_obj, *a_obj, *x_obj;
    PyArrayObject *r_array, *a_array, *x_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssiiddOOO",
                          &uplo_str, &trans_str, &m, &n, &alpha, &beta,
                          &r_obj, &a_obj, &x_obj)) {
        return NULL;
    }

    uplo = uplo_str[0];
    trans = trans_str[0];

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        Py_DECREF(r_array);
        return NULL;
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (x_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *r_dims = PyArray_DIMS(r_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *x_dims = PyArray_DIMS(x_array);

    i32 ldr = (i32)r_dims[0];
    i32 lda = (i32)a_dims[0];
    i32 ldx = (i32)x_dims[0];

    f64 *r_data = (f64*)PyArray_DATA(r_array);
    const f64 *a_data = (const f64*)PyArray_DATA(a_array);
    f64 *x_data = (f64*)PyArray_DATA(x_array);

    i32 ldwork = (beta != 0.0 && m > 0 && n > 0) ? m * n : 1;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(a_array);
        Py_DECREF(x_array);
        return PyErr_NoMemory();
    }

    mb01rd(&uplo, &trans, m, n, alpha, beta, r_data, ldr, a_data, lda,
           x_data, ldx, dwork, ldwork, &info);

    free(dwork);
    Py_DECREF(a_array);

    PyArray_ResolveWritebackIfCopy(x_array);
    Py_DECREF(x_array);

    if (info < 0) {
        PyArray_DiscardWritebackIfCopy(r_array);
        Py_DECREF(r_array);
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(r_array);
    PyObject *result = Py_BuildValue("Oi", r_array, info);
    Py_DECREF(r_array);
    return result;
}



/* Python wrapper for mb01rw */
PyObject* py_mb01rw(PyObject* self, PyObject* args) {
    const char *uplo_str, *trans_str;
    char uplo, trans;
    i32 m, n;
    PyObject *a_obj, *z_obj;
    PyArrayObject *a_array, *z_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssiiOO",
                          &uplo_str, &trans_str, &m, &n,
                          &a_obj, &z_obj)) {
        return NULL;
    }

    uplo = uplo_str[0];
    trans = trans_str[0];

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    z_array = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (z_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *z_dims = PyArray_DIMS(z_array);

    i32 lda = (i32)a_dims[0];
    i32 ldz = (i32)z_dims[0];

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    const f64 *z_data = (const f64*)PyArray_DATA(z_array);

    i32 dwork_size = n > 0 ? n : 1;
    f64 *dwork = (f64*)malloc(dwork_size * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(z_array);
        return PyErr_NoMemory();
    }

    mb01rw(&uplo, &trans, m, n, a_data, lda, z_data, ldz, dwork, &info);

    free(dwork);
    Py_DECREF(z_array);

    if (info < 0) {
        PyArray_DiscardWritebackIfCopy(a_array);
        Py_DECREF(a_array);
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(a_array);
    PyObject *result = Py_BuildValue("Oi", a_array, info);
    Py_DECREF(a_array);
    return result;
}



/* Python wrapper for mb01uy */
PyObject* py_mb01uy(PyObject* self, PyObject* args) {
    char *side, *uplo, *trans;
    i32 m, n;
    f64 alpha;
    PyObject *t_obj, *a_obj;
    PyArrayObject *t_array, *a_array;
    i32 info;
    i32 ldwork = 0;
    Py_ssize_t nargs = PyTuple_Size(args);

    if (!PyArg_ParseTuple(args, "sssiidOO|i", &side, &uplo, &trans, &m, &n, &alpha, &t_obj, &a_obj, &ldwork)) {
        return NULL;
    }

    t_array = (PyArrayObject*)PyArray_FROM_OTF(t_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (t_array == NULL) return NULL;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) {
        Py_DECREF(t_array);
        return NULL;
    }

    i32 ldt = (m > n) ? m : n;
    if ((*side == 'R' || *side == 'r') && ldt < n) ldt = n;
    i32 lda = (i32)PyArray_DIM(a_array, 0);

    f64 *t_in = (f64*)PyArray_DATA(t_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);

    npy_intp t_dims[2] = {ldt, (ldt > n) ? ldt : n};
    npy_intp t_strides[2] = {sizeof(f64), ldt * sizeof(f64)};
    PyObject *t_out = PyArray_New(&PyArray_Type, 2, t_dims, NPY_DOUBLE, t_strides,
                                   NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (t_out == NULL) {
        Py_DECREF(t_array);
        Py_DECREF(a_array);
        return NULL;
    }

    f64 *t_data = (f64*)PyArray_DATA((PyArrayObject*)t_out);
    i32 k = (*side == 'L' || *side == 'l') ? m : n;
    for (i32 j = 0; j < k; j++) {
        for (i32 i = 0; i < k; i++) {
            t_data[i + j * ldt] = t_in[i + j * PyArray_DIM(t_array, 0)];
        }
    }

    if (nargs <= 8) {
        ldwork = m * n;
    }
    i32 dwork_len = (ldwork == -1) ? 1 : ldwork;
    if (dwork_len < 1) dwork_len = 1;
    f64 *dwork = (f64*)malloc((size_t)dwork_len * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(t_array);
        Py_DECREF(a_array);
        Py_DECREF(t_out);
        return PyErr_NoMemory();
    }

    mb01uy(side, uplo, trans, m, n, alpha, t_data, ldt, a_data, lda, dwork, ldwork, &info);
    f64 work = dwork[0];

    free(dwork);

    if (ldwork == -1) {
        PyObject *result = Py_BuildValue("Oid", Py_None, info, work);
        Py_DECREF(t_array);
        Py_DECREF(a_array);
        Py_DECREF(t_out);
        return result;
    }

    npy_intp result_dims[2] = {m, n};
    npy_intp result_strides[2] = {sizeof(f64), m * sizeof(f64)};
    PyObject *result_array = PyArray_New(&PyArray_Type, 2, result_dims, NPY_DOUBLE, result_strides,
                                          NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (result_array == NULL) {
        Py_DECREF(t_array);
        Py_DECREF(a_array);
        Py_DECREF(t_out);
        return NULL;
    }

    f64 *result_data = (f64*)PyArray_DATA((PyArrayObject*)result_array);
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < m; i++) {
            result_data[i + j * m] = t_data[i + j * ldt];
        }
    }

    PyObject *result;
    if (nargs <= 8) {
        result = Py_BuildValue("Oi", result_array, info);
    } else {
        result = Py_BuildValue("Oid", result_array, info, work);
    }
    Py_DECREF(t_array);
    Py_DECREF(a_array);
    Py_DECREF(t_out);
    Py_DECREF(result_array);

    return result;
}



/* Python wrapper for mb01sd */
PyObject* py_mb01sd(PyObject* self, PyObject* args) {
    const char *jobs_str;
    PyObject *a_obj, *r_obj, *c_obj;
    PyArrayObject *a_array, *r_array, *c_array;

    if (!PyArg_ParseTuple(args, "sOOO", &jobs_str, &a_obj, &r_obj, &c_obj)) {
        return NULL;
    }

    char jobs = jobs_str[0];

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (r_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(r_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 m = (i32)a_dims[0];
    i32 n = (i32)a_dims[1];
    i32 lda = m;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    const f64 *r_data = (const f64*)PyArray_DATA(r_array);
    const f64 *c_data = (const f64*)PyArray_DATA(c_array);

    mb01sd(jobs, m, n, a_data, lda, r_data, c_data);

    Py_DECREF(r_array);
    Py_DECREF(c_array);

    PyArray_ResolveWritebackIfCopy(a_array);

    PyObject *result = (PyObject*)a_array;
    return result;
}



/* Python wrapper for mb01kd */
PyObject* py_mb01kd(PyObject* self, PyObject* args) {
    const char *uplo_str, *trans_str;
    i32 n, k;
    f64 alpha, beta;
    PyObject *a_obj, *b_obj, *c_obj;
    PyArrayObject *a_array, *b_array, *c_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssiidOOdO",
                          &uplo_str, &trans_str, &n, &k, &alpha,
                          &a_obj, &b_obj, &beta, &c_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
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

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    i32 lda = (i32)a_dims[0];
    i32 ldb = (i32)b_dims[0];
    i32 ldc = (i32)c_dims[0];

    const f64 *a_data = (const f64*)PyArray_DATA(a_array);
    const f64 *b_data = (const f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    mb01kd(uplo_str, trans_str, n, k, alpha, a_data, lda, b_data, ldb,
           beta, c_data, ldc, &info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);

    PyArray_ResolveWritebackIfCopy(c_array);
    PyObject *result = Py_BuildValue("Oi", c_array, info);
    Py_DECREF(c_array);
    return result;
}



/* Python wrapper for mb01md */
PyObject* py_mb01md(PyObject* self, PyObject* args) {
    const char *uplo_str;
    i32 n, incx, incy;
    f64 alpha, beta;
    PyObject *a_obj, *x_obj, *y_obj;
    PyArrayObject *a_array, *x_array, *y_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "sidOOidOi",
                          &uplo_str, &n, &alpha, &a_obj, &x_obj, &incx,
                          &beta, &y_obj, &incy)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (x_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE,
                                               NPY_ARRAY_INOUT_ARRAY2);
    if (y_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(x_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda = (i32)a_dims[0];
    if (lda < 1) lda = 1;

    const f64 *a_data = (const f64*)PyArray_DATA(a_array);
    const f64 *x_data = (const f64*)PyArray_DATA(x_array);
    f64 *y_data = (f64*)PyArray_DATA(y_array);

    mb01md(uplo_str[0], n, alpha, a_data, lda, x_data, incx,
           beta, y_data, incy, &info);

    Py_DECREF(a_array);
    Py_DECREF(x_array);

    PyArray_ResolveWritebackIfCopy(y_array);
    PyObject *result = Py_BuildValue("Oi", y_array, info);
    Py_DECREF(y_array);
    return result;
}



/* Python wrapper for mb01nd */
PyObject* py_mb01nd(PyObject* self, PyObject* args) {
    const char *uplo_str;
    i32 n, incx, incy;
    f64 alpha;
    PyObject *x_obj, *y_obj, *a_obj;
    PyArrayObject *x_array, *y_array, *a_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "sidOiOiO",
                          &uplo_str, &n, &alpha, &x_obj, &incx, &y_obj, &incy, &a_obj)) {
        return NULL;
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (x_array == NULL) {
        return NULL;
    }

    y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (y_array == NULL) {
        Py_DECREF(x_array);
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda = (i32)a_dims[0];
    if (lda < 1) lda = 1;

    const f64 *x_data = (const f64*)PyArray_DATA(x_array);
    const f64 *y_data = (const f64*)PyArray_DATA(y_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);

    mb01nd(uplo_str[0], n, alpha, x_data, incx, y_data, incy, a_data, lda, &info);

    Py_DECREF(x_array);
    Py_DECREF(y_array);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyObject *result = Py_BuildValue("Oi", a_array, info);
    Py_DECREF(a_array);
    return result;
}



/* Python wrapper for mb01ld */
PyObject* py_mb01ld(PyObject* self, PyObject* args) {
    const char *uplo_str, *trans_str;
    i32 m, n;
    f64 alpha, beta;
    PyObject *r_obj, *a_obj, *x_obj;
    PyArrayObject *r_array, *a_array, *x_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssiiddOOO",
                          &uplo_str, &trans_str, &m, &n, &alpha, &beta,
                          &r_obj, &a_obj, &x_obj)) {
        return NULL;
    }

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        Py_DECREF(r_array);
        return NULL;
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (x_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *r_dims = PyArray_DIMS(r_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *x_dims = PyArray_DIMS(x_array);

    i32 ldr = (i32)r_dims[0];
    i32 lda = (i32)a_dims[0];
    i32 ldx = (i32)x_dims[0];

    f64 *r_data = (f64*)PyArray_DATA(r_array);
    const f64 *a_data = (const f64*)PyArray_DATA(a_array);
    f64 *x_data = (f64*)PyArray_DATA(x_array);

    i32 ldwork = m * (n > 1 ? n - 1 : 0);
    if (ldwork < n) ldwork = n;
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(a_array);
        Py_DECREF(x_array);
        return PyErr_NoMemory();
    }

    mb01ld(uplo_str, trans_str, m, n, alpha, beta,
           r_data, ldr, a_data, lda, x_data, ldx, dwork, ldwork, &info);

    free(dwork);

    Py_DECREF(a_array);
    PyArray_ResolveWritebackIfCopy(x_array);
    Py_DECREF(x_array);

    PyArray_ResolveWritebackIfCopy(r_array);
    PyObject *result = Py_BuildValue("Oi", r_array, info);
    Py_DECREF(r_array);
    return result;
}



/* Python wrapper for mb01oc */
PyObject* py_mb01oc(PyObject* self, PyObject* args) {
    const char *uplo_str, *trans_str;
    i32 n;
    f64 alpha, beta;
    PyObject *r_obj, *h_obj, *x_obj;
    PyArrayObject *r_array, *h_array, *x_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssiddOOO",
                          &uplo_str, &trans_str, &n, &alpha, &beta,
                          &r_obj, &h_obj, &x_obj)) {
        return NULL;
    }

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) {
        return NULL;
    }

    h_array = (PyArrayObject*)PyArray_FROM_OTF(h_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (h_array == NULL) {
        Py_DECREF(r_array);
        return NULL;
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (x_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(h_array);
        return NULL;
    }

    npy_intp *r_dims = PyArray_DIMS(r_array);
    npy_intp *h_dims = PyArray_DIMS(h_array);
    npy_intp *x_dims = PyArray_DIMS(x_array);

    i32 ldr = (i32)r_dims[0];
    i32 ldh = (i32)h_dims[0];
    i32 ldx = (i32)x_dims[0];
    if (ldr < 1) ldr = 1;
    if (ldh < 1) ldh = 1;
    if (ldx < 1) ldx = 1;

    f64 *r_data = (f64*)PyArray_DATA(r_array);
    const f64 *h_data = (const f64*)PyArray_DATA(h_array);
    const f64 *x_data = (const f64*)PyArray_DATA(x_array);

    mb01oc(uplo_str, trans_str, n, alpha, beta, r_data, ldr,
           h_data, ldh, x_data, ldx, &info);

    Py_DECREF(h_array);
    Py_DECREF(x_array);

    PyArray_ResolveWritebackIfCopy(r_array);
    PyObject *result = Py_BuildValue("Oi", r_array, info);
    Py_DECREF(r_array);
    return result;
}



/* Python wrapper for mb01od */
PyObject* py_mb01od(PyObject* self, PyObject* args) {
    const char *uplo_str, *trans_str;
    i32 n;
    f64 alpha, beta;
    PyObject *r_obj, *h_obj, *x_obj, *e_obj;
    PyArrayObject *r_array, *h_array, *x_array, *e_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssiddOOOO",
                          &uplo_str, &trans_str, &n, &alpha, &beta,
                          &r_obj, &h_obj, &x_obj, &e_obj)) {
        return NULL;
    }

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) {
        return NULL;
    }

    h_array = (PyArrayObject*)PyArray_FROM_OTF(h_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (h_array == NULL) {
        Py_DECREF(r_array);
        return NULL;
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (x_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(h_array);
        return NULL;
    }

    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (e_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(h_array);
        Py_DECREF(x_array);
        return NULL;
    }

    npy_intp *r_dims = PyArray_DIMS(r_array);
    npy_intp *h_dims = PyArray_DIMS(h_array);
    npy_intp *x_dims = PyArray_DIMS(x_array);
    npy_intp *e_dims = PyArray_DIMS(e_array);

    i32 ldr = (i32)r_dims[0];
    i32 ldh = (i32)h_dims[0];
    i32 ldx = (i32)x_dims[0];
    i32 lde = (i32)e_dims[0];
    if (ldr < 1) ldr = 1;
    if (ldh < 1) ldh = 1;
    if (ldx < 1) ldx = 1;
    if (lde < 1) lde = 1;

    i32 ldwork = (beta != 0.0 && n > 0) ? n * n : 1;
    f64 *dwork = NULL;
    if (ldwork > 0) {
        dwork = (f64*)malloc(ldwork * sizeof(f64));
        if (dwork == NULL) {
            Py_DECREF(r_array);
            Py_DECREF(h_array);
            Py_DECREF(x_array);
            Py_DECREF(e_array);
            PyErr_NoMemory();
            return NULL;
        }
    }

    f64 *r_data = (f64*)PyArray_DATA(r_array);
    f64 *h_data = (f64*)PyArray_DATA(h_array);
    f64 *x_data = (f64*)PyArray_DATA(x_array);
    const f64 *e_data = (const f64*)PyArray_DATA(e_array);

    mb01od(uplo_str, trans_str, n, alpha, beta, r_data, ldr,
           h_data, ldh, x_data, ldx, e_data, lde, dwork, ldwork, &info);

    free(dwork);

    Py_DECREF(e_array);
    PyArray_ResolveWritebackIfCopy(x_array);
    Py_DECREF(x_array);
    PyArray_ResolveWritebackIfCopy(h_array);
    Py_DECREF(h_array);

    PyArray_ResolveWritebackIfCopy(r_array);
    PyObject *result = Py_BuildValue("Oi", r_array, info);
    Py_DECREF(r_array);
    return result;
}



/* Python wrapper for mb01oe */
PyObject* py_mb01oe(PyObject* self, PyObject* args) {
    const char *uplo_str, *trans_str;
    i32 n;
    f64 alpha, beta;
    PyObject *r_obj, *h_obj, *e_obj;
    PyArrayObject *r_array, *h_array, *e_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssiddOOO",
                          &uplo_str, &trans_str, &n, &alpha, &beta,
                          &r_obj, &h_obj, &e_obj)) {
        return NULL;
    }

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) {
        return NULL;
    }

    h_array = (PyArrayObject*)PyArray_FROM_OTF(h_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (h_array == NULL) {
        Py_DECREF(r_array);
        return NULL;
    }

    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (e_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(h_array);
        return NULL;
    }

    npy_intp *r_dims = PyArray_DIMS(r_array);
    npy_intp *h_dims = PyArray_DIMS(h_array);
    npy_intp *e_dims = PyArray_DIMS(e_array);

    i32 ldr = (i32)r_dims[0];
    i32 ldh = (i32)h_dims[0];
    i32 lde = (i32)e_dims[0];
    if (ldr < 1) ldr = 1;
    if (ldh < 1) ldh = 1;
    if (lde < 1) lde = 1;

    f64 *r_data = (f64*)PyArray_DATA(r_array);
    const f64 *h_data = (const f64*)PyArray_DATA(h_array);
    const f64 *e_data = (const f64*)PyArray_DATA(e_array);

    mb01oe(uplo_str, trans_str, n, alpha, beta, r_data, ldr,
           h_data, ldh, e_data, lde, &info);

    Py_DECREF(h_array);
    Py_DECREF(e_array);

    PyArray_ResolveWritebackIfCopy(r_array);
    PyObject *result = Py_BuildValue("Oi", r_array, info);
    Py_DECREF(r_array);
    return result;
}



/* Python wrapper for mb01oh */
PyObject* py_mb01oh(PyObject* self, PyObject* args) {
    const char *uplo_str, *trans_str;
    i32 n;
    f64 alpha, beta;
    PyObject *r_obj, *h_obj, *a_obj;
    PyArrayObject *r_array, *h_array, *a_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssiddOOO",
                          &uplo_str, &trans_str, &n, &alpha, &beta,
                          &r_obj, &h_obj, &a_obj)) {
        return NULL;
    }

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) {
        return NULL;
    }

    h_array = (PyArrayObject*)PyArray_FROM_OTF(h_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (h_array == NULL) {
        Py_DECREF(r_array);
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(h_array);
        return NULL;
    }

    npy_intp *r_dims = PyArray_DIMS(r_array);
    npy_intp *h_dims = PyArray_DIMS(h_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);

    i32 ldr = (i32)r_dims[0];
    i32 ldh = (i32)h_dims[0];
    i32 lda = (i32)a_dims[0];
    if (ldr < 1) ldr = 1;
    if (ldh < 1) ldh = 1;
    if (lda < 1) lda = 1;

    f64 *r_data = (f64*)PyArray_DATA(r_array);
    const f64 *h_data = (const f64*)PyArray_DATA(h_array);
    const f64 *a_data = (const f64*)PyArray_DATA(a_array);

    mb01oh(uplo_str, trans_str, n, alpha, beta, r_data, ldr,
           h_data, ldh, a_data, lda, &info);

    Py_DECREF(h_array);
    Py_DECREF(a_array);

    PyArray_ResolveWritebackIfCopy(r_array);
    PyObject *result = Py_BuildValue("Oi", r_array, info);
    Py_DECREF(r_array);
    return result;
}



/* Python wrapper for mb01ot */
PyObject* py_mb01ot(PyObject* self, PyObject* args) {
    const char *uplo_str, *trans_str;
    i32 n;
    f64 alpha, beta;
    PyObject *r_obj, *e_obj, *t_obj;
    PyArrayObject *r_array, *e_array, *t_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssiddOOO",
                          &uplo_str, &trans_str, &n, &alpha, &beta,
                          &r_obj, &e_obj, &t_obj)) {
        return NULL;
    }

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) {
        return NULL;
    }

    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (e_array == NULL) {
        Py_DECREF(r_array);
        return NULL;
    }

    t_array = (PyArrayObject*)PyArray_FROM_OTF(t_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (t_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(e_array);
        return NULL;
    }

    npy_intp *r_dims = PyArray_DIMS(r_array);
    npy_intp *e_dims = PyArray_DIMS(e_array);
    npy_intp *t_dims = PyArray_DIMS(t_array);

    i32 ldr = (i32)r_dims[0];
    i32 lde = (i32)e_dims[0];
    i32 ldt = (i32)t_dims[0];
    if (ldr < 1) ldr = 1;
    if (lde < 1) lde = 1;
    if (ldt < 1) ldt = 1;

    f64 *r_data = (f64*)PyArray_DATA(r_array);
    const f64 *e_data = (const f64*)PyArray_DATA(e_array);
    const f64 *t_data = (const f64*)PyArray_DATA(t_array);

    mb01ot(uplo_str, trans_str, n, alpha, beta, r_data, ldr,
           e_data, lde, t_data, ldt, &info);

    Py_DECREF(e_array);
    Py_DECREF(t_array);

    PyArray_ResolveWritebackIfCopy(r_array);
    PyObject *result = Py_BuildValue("Oi", r_array, info);
    Py_DECREF(r_array);
    return result;
}



/* Python wrapper for mb01ss */
PyObject* py_mb01ss(PyObject* self, PyObject* args) {
    const char *jobs_str, *uplo_str;
    PyObject *a_obj, *d_obj;
    PyArrayObject *a_array, *d_array;

    if (!PyArg_ParseTuple(args, "ssOO", &jobs_str, &uplo_str, &a_obj, &d_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 n = (i32)a_dims[0];
    i32 lda = n > 0 ? n : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    const f64 *d_data = (const f64*)PyArray_DATA(d_array);

    mb01ss(jobs_str[0], uplo_str[0], n, a_data, lda, d_data);

    Py_DECREF(d_array);

    PyArray_ResolveWritebackIfCopy(a_array);

    PyObject *result = (PyObject*)a_array;
    return result;
}



/* Python wrapper for mb01os */
PyObject* py_mb01os(PyObject* self, PyObject* args) {
    const char *uplo_str, *trans_str;
    PyObject *h_obj, *x_obj;
    PyArrayObject *h_array, *x_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssOO",
                          &uplo_str, &trans_str, &h_obj, &x_obj)) {
        return NULL;
    }

    h_array = (PyArrayObject*)PyArray_FROM_OTF(h_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (h_array == NULL) {
        return NULL;
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (x_array == NULL) {
        Py_DECREF(h_array);
        return NULL;
    }

    npy_intp *h_dims = PyArray_DIMS(h_array);
    npy_intp *x_dims = PyArray_DIMS(x_array);

    i32 n = (i32)h_dims[0];
    i32 ldh = n > 0 ? n : 1;
    i32 ldx = (i32)x_dims[0];
    if (ldx < 1) ldx = 1;
    i32 ldp = n > 0 ? n : 1;

    const f64 *h_data = (const f64*)PyArray_DATA(h_array);
    const f64 *x_data = (const f64*)PyArray_DATA(x_array);

    npy_intp p_dims[2] = {n, n};
    npy_intp p_strides[2] = {sizeof(f64), n * sizeof(f64)};

    if (n == 0) {
        Py_DECREF(h_array);
        Py_DECREF(x_array);
        npy_intp empty_dims[2] = {0, 0};
        PyArrayObject *p_array = (PyArrayObject*)PyArray_EMPTY(2, empty_dims, NPY_DOUBLE, 1);
        PyObject *result = Py_BuildValue("Oi", p_array, 0);
        Py_DECREF(p_array);
        return result;
    }

    PyArrayObject *p_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, p_dims, NPY_DOUBLE, p_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (p_array == NULL) {
        Py_DECREF(h_array);
        Py_DECREF(x_array);
        PyErr_NoMemory();
        return NULL;
    }
    f64 *p_data = (f64*)PyArray_DATA(p_array);

    mb01os(uplo_str, trans_str, n, h_data, ldh, x_data, ldx, p_data, ldp, &info);

    Py_DECREF(h_array);
    Py_DECREF(x_array);

    if (info != 0) {
        Py_DECREF(p_array);
        npy_intp err_dims[2] = {n, n};
        PyArrayObject *err_array = (PyArrayObject*)PyArray_ZEROS(2, err_dims, NPY_DOUBLE, 1);
        PyObject *result = Py_BuildValue("Oi", err_array, info);
        Py_DECREF(err_array);
        return result;
    }

    PyObject *result = Py_BuildValue("Oi", p_array, info);
    Py_DECREF(p_array);
    return result;
}



/* Python wrapper for mb01oo */
PyObject* py_mb01oo(PyObject* self, PyObject* args) {
    const char *uplo_str, *trans_str;
    PyObject *h_obj, *x_obj, *e_obj;
    PyArrayObject *h_array, *x_array, *e_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssOOO",
                          &uplo_str, &trans_str, &h_obj, &x_obj, &e_obj)) {
        return NULL;
    }

    h_array = (PyArrayObject*)PyArray_FROM_OTF(h_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (h_array == NULL) {
        return NULL;
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (x_array == NULL) {
        Py_DECREF(h_array);
        return NULL;
    }

    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (e_array == NULL) {
        Py_DECREF(h_array);
        Py_DECREF(x_array);
        return NULL;
    }

    npy_intp *h_dims = PyArray_DIMS(h_array);
    npy_intp *x_dims = PyArray_DIMS(x_array);
    npy_intp *e_dims = PyArray_DIMS(e_array);

    i32 n = (i32)h_dims[0];
    i32 ldh = n > 0 ? n : 1;
    i32 ldx = (i32)x_dims[0];
    if (ldx < 1) ldx = 1;
    i32 lde = (i32)e_dims[0];
    if (lde < 1) lde = 1;
    i32 ldp = n > 0 ? n : 1;

    const f64 *h_data = (const f64*)PyArray_DATA(h_array);
    const f64 *x_data = (const f64*)PyArray_DATA(x_array);
    const f64 *e_data = (const f64*)PyArray_DATA(e_array);

    npy_intp p_dims[2] = {n, n};
    npy_intp p_strides[2] = {sizeof(f64), n * sizeof(f64)};

    if (n == 0) {
        Py_DECREF(h_array);
        Py_DECREF(x_array);
        Py_DECREF(e_array);
        npy_intp empty_dims[2] = {0, 0};
        PyArrayObject *p_array = (PyArrayObject*)PyArray_EMPTY(2, empty_dims, NPY_DOUBLE, 1);
        PyObject *result = Py_BuildValue("Oi", p_array, 0);
        Py_DECREF(p_array);
        return result;
    }

    PyArrayObject *p_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, p_dims, NPY_DOUBLE, p_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (p_array == NULL) {
        Py_DECREF(h_array);
        Py_DECREF(x_array);
        Py_DECREF(e_array);
        PyErr_NoMemory();
        return NULL;
    }
    f64 *p_data = (f64*)PyArray_DATA(p_array);

    mb01oo(uplo_str, trans_str, n, h_data, ldh, x_data, ldx, e_data, lde, p_data, ldp, &info);

    Py_DECREF(h_array);
    Py_DECREF(x_array);
    Py_DECREF(e_array);

    if (info != 0) {
        Py_DECREF(p_array);
        npy_intp err_dims[2] = {n, n};
        PyArrayObject *err_array = (PyArrayObject*)PyArray_ZEROS(2, err_dims, NPY_DOUBLE, 1);
        PyObject *result = Py_BuildValue("Oi", err_array, info);
        Py_DECREF(err_array);
        return result;
    }

    PyObject *result = Py_BuildValue("Oi", p_array, info);
    Py_DECREF(p_array);
    return result;
}



/* Python wrapper for mb01uw */
PyObject* py_mb01uw(PyObject* self, PyObject* args) {
    const char *side_str, *trans_str;
    i32 m, n;
    f64 alpha;
    PyObject *h_obj, *a_obj;
    PyArrayObject *h_array, *a_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssiidOO",
                          &side_str, &trans_str, &m, &n, &alpha,
                          &h_obj, &a_obj)) {
        return NULL;
    }

    h_array = (PyArrayObject*)PyArray_FROM_OTF(h_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (h_array == NULL) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        Py_DECREF(h_array);
        return NULL;
    }

    npy_intp *h_dims = PyArray_DIMS(h_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);

    i32 ldh = (i32)h_dims[0];
    i32 lda = (i32)a_dims[0];
    if (ldh < 1) ldh = 1;
    if (lda < 1) lda = 1;

    i32 ldwork = m * n;
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(h_array);
        Py_DECREF(a_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *h_data = (f64*)PyArray_DATA(h_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);

    mb01uw(side_str, trans_str, m, n, alpha, h_data, ldh, a_data, lda, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(h_array);
    Py_DECREF(h_array);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyObject *result = Py_BuildValue("Oi", a_array, info);
    Py_DECREF(a_array);
    return result;
}



/* Python wrapper for mb01ux */
PyObject* py_mb01ux(PyObject* self, PyObject* args) {
    const char *side_str, *uplo_str, *trans_str;
    i32 m, n;
    f64 alpha;
    PyObject *t_obj, *a_obj;
    PyArrayObject *t_array, *a_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "sssiidOO",
                          &side_str, &uplo_str, &trans_str, &m, &n, &alpha,
                          &t_obj, &a_obj)) {
        return NULL;
    }

    t_array = (PyArrayObject*)PyArray_FROM_OTF(t_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (t_array == NULL) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        Py_DECREF(t_array);
        return NULL;
    }

    npy_intp *t_dims = PyArray_DIMS(t_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);

    i32 ldt = (i32)t_dims[0];
    i32 lda = (i32)a_dims[0];
    if (ldt < 1) ldt = 1;
    if (lda < 1) lda = 1;

    i32 k = (side_str[0] == 'L' || side_str[0] == 'l') ? m : n;
    i32 ldwork = (k > 1) ? (k / 2) * ((side_str[0] == 'L' || side_str[0] == 'l') ? n : m) + k - 1 : 1;
    if (ldwork < 2 * (k - 1) && k > 1) ldwork = 2 * (k - 1);
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(t_array);
        Py_DECREF(a_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *t_data = (f64*)PyArray_DATA(t_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);

    mb01ux(side_str, uplo_str, trans_str, m, n, alpha, t_data, ldt, a_data, lda, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(t_array);
    Py_DECREF(t_array);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyObject *result = Py_BuildValue("Oi", a_array, info);
    Py_DECREF(a_array);
    return result;
}

/* Python wrapper for mb01uz (complex) */
PyObject* py_mb01uz(PyObject* self, PyObject* args) {
    const char *side_str, *uplo_str, *trans_str;
    i32 m, n;
    Py_complex alpha_py;
    PyObject *t_obj, *a_obj;
    PyArrayObject *t_array, *a_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "sssiiDOO",
                          &side_str, &uplo_str, &trans_str, &m, &n, &alpha_py,
                          &t_obj, &a_obj)) {
        return NULL;
    }

    c128 alpha = alpha_py.real + alpha_py.imag * I;

    t_array = (PyArrayObject*)PyArray_FROM_OTF(t_obj, NPY_COMPLEX128,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (t_array == NULL) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_COMPLEX128,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        Py_DECREF(t_array);
        return NULL;
    }

    npy_intp *t_dims = PyArray_DIMS(t_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);

    i32 ldt = (i32)t_dims[0];
    i32 lda = (i32)a_dims[0];
    if (ldt < 1) ldt = 1;
    if (lda < 1) lda = 1;

    char side_c = (char)toupper((unsigned char)side_str[0]);
    i32 k = (side_c == 'L') ? m : n;
    i32 lzwork = m * n;
    if (lzwork < k) lzwork = k;
    if (lzwork < 1) lzwork = 1;

    c128 *zwork = (c128*)malloc(lzwork * sizeof(c128));
    if (zwork == NULL) {
        Py_DECREF(t_array);
        Py_DECREF(a_array);
        PyErr_NoMemory();
        return NULL;
    }

    c128 *t_data = (c128*)PyArray_DATA(t_array);
    c128 *a_data = (c128*)PyArray_DATA(a_array);

    mb01uz(side_str, uplo_str, trans_str, m, n, alpha, t_data, ldt, a_data, lda, zwork, lzwork, &info);

    free(zwork);

    PyArray_ResolveWritebackIfCopy(t_array);
    PyObject *result = Py_BuildValue("Oi", t_array, info);
    Py_DECREF(t_array);

    PyArray_ResolveWritebackIfCopy(a_array);
    Py_DECREF(a_array);
    return result;
}


/* Python wrapper for mb01rh */
PyObject* py_mb01rh(PyObject* self, PyObject* args) {
    const char *uplo_str, *trans_str;
    i32 n;
    f64 alpha, beta;
    PyObject *r_obj, *h_obj, *x_obj;
    PyArrayObject *r_array, *h_array, *x_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssiddOOO",
                          &uplo_str, &trans_str, &n, &alpha, &beta,
                          &r_obj, &h_obj, &x_obj)) {
        return NULL;
    }

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) {
        return NULL;
    }

    h_array = (PyArrayObject*)PyArray_FROM_OTF(h_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (h_array == NULL) {
        Py_DECREF(r_array);
        return NULL;
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (x_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(h_array);
        return NULL;
    }

    npy_intp *r_dims = PyArray_DIMS(r_array);
    npy_intp *h_dims = PyArray_DIMS(h_array);
    npy_intp *x_dims = PyArray_DIMS(x_array);

    i32 ldr = (i32)r_dims[0];
    i32 ldh = (i32)h_dims[0];
    i32 ldx = (i32)x_dims[0];
    if (ldr < 1) ldr = 1;
    if (ldh < 1) ldh = 1;
    if (ldx < 1) ldx = 1;

    i32 ldwork = (beta != 0.0 && n > 0) ? n * n : 1;
    f64 *dwork = NULL;
    if (ldwork > 0) {
        dwork = (f64*)malloc(ldwork * sizeof(f64));
        if (dwork == NULL) {
            Py_DECREF(r_array);
            Py_DECREF(h_array);
            Py_DECREF(x_array);
            PyErr_NoMemory();
            return NULL;
        }
    }

    f64 *r_data = (f64*)PyArray_DATA(r_array);
    f64 *h_data = (f64*)PyArray_DATA(h_array);
    f64 *x_data = (f64*)PyArray_DATA(x_array);

    mb01rh(uplo_str, trans_str, n, alpha, beta, r_data, ldr,
           h_data, ldh, x_data, ldx, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(x_array);
    Py_DECREF(x_array);
    PyArray_ResolveWritebackIfCopy(h_array);
    Py_DECREF(h_array);

    PyArray_ResolveWritebackIfCopy(r_array);
    PyObject *result = Py_BuildValue("Oi", r_array, info);
    Py_DECREF(r_array);
    return result;
}


/* Python wrapper for mb01rt */
PyObject* py_mb01rt(PyObject* self, PyObject* args) {
    const char *uplo_str, *trans_str;
    i32 n;
    f64 alpha, beta;
    PyObject *r_obj, *e_obj, *x_obj;
    PyArrayObject *r_array, *e_array, *x_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssiddOOO",
                          &uplo_str, &trans_str, &n, &alpha, &beta,
                          &r_obj, &e_obj, &x_obj)) {
        return NULL;
    }

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) {
        return NULL;
    }

    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (e_array == NULL) {
        Py_DECREF(r_array);
        return NULL;
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (x_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(e_array);
        return NULL;
    }

    npy_intp *r_dims = PyArray_DIMS(r_array);
    npy_intp *e_dims = PyArray_DIMS(e_array);
    npy_intp *x_dims = PyArray_DIMS(x_array);

    i32 ldr = (i32)r_dims[0];
    i32 lde = (i32)e_dims[0];
    i32 ldx = (i32)x_dims[0];
    if (ldr < 1) ldr = 1;
    if (lde < 1) lde = 1;
    if (ldx < 1) ldx = 1;

    i32 ldwork = (beta != 0.0 && n > 0) ? n * n : 1;
    f64 *dwork = NULL;
    if (ldwork > 0) {
        dwork = (f64*)malloc(ldwork * sizeof(f64));
        if (dwork == NULL) {
            Py_DECREF(r_array);
            Py_DECREF(e_array);
            Py_DECREF(x_array);
            PyErr_NoMemory();
            return NULL;
        }
    }

    f64 *r_data = (f64*)PyArray_DATA(r_array);
    const f64 *e_data = (const f64*)PyArray_DATA(e_array);
    f64 *x_data = (f64*)PyArray_DATA(x_array);

    mb01rt(uplo_str, trans_str, n, alpha, beta, r_data, ldr,
           e_data, lde, x_data, ldx, dwork, ldwork, &info);

    free(dwork);

    Py_DECREF(e_array);
    PyArray_ResolveWritebackIfCopy(x_array);
    Py_DECREF(x_array);

    PyArray_ResolveWritebackIfCopy(r_array);
    PyObject *result = Py_BuildValue("Oi", r_array, info);
    Py_DECREF(r_array);
    return result;
}


/* Python wrapper for mb01xy */
PyObject* py_mb01xy(PyObject* self, PyObject* args) {
    const char *uplo_str;
    PyObject *a_obj;
    PyArrayObject *a_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "sO", &uplo_str, &a_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 n = (i32)a_dims[0];
    i32 lda = n > 0 ? n : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);

    mb01xy(uplo_str, n, a_data, lda, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyObject *result = Py_BuildValue("Oi", a_array, info);
    Py_DECREF(a_array);
    return result;
}


PyObject* py_mb01yd(PyObject* self, PyObject* args) {
    const char *uplo_str, *trans_str;
    i32 n, k, l;
    f64 alpha, beta;
    PyObject *a_obj, *c_obj;
    PyArrayObject *a_array, *c_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssiiiddOO",
                          &uplo_str, &trans_str, &n, &k, &l, &alpha, &beta,
                          &a_obj, &c_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    i32 lda = (i32)a_dims[0];
    i32 ldc = (i32)c_dims[0];
    if (lda < 1) lda = 1;
    if (ldc < 1) ldc = 1;

    const f64 *a_data = (const f64*)PyArray_DATA(a_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    mb01yd(uplo_str, trans_str, n, k, l, alpha, beta, a_data, lda, c_data, ldc, &info);

    Py_DECREF(a_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyObject *result = Py_BuildValue("Oi", c_array, info);
    Py_DECREF(c_array);
    return result;
}


/* Python wrapper for mb01xd */
PyObject* py_mb01xd(PyObject* self, PyObject* args) {
    const char *uplo_str;
    PyObject *a_obj;
    PyArrayObject *a_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "sO", &uplo_str, &a_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 n = (i32)a_dims[0];
    i32 lda = n > 0 ? n : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);

    mb01xd(uplo_str, n, a_data, lda, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyObject *result = Py_BuildValue("Oi", a_array, info);
    Py_DECREF(a_array);
    return result;
}


/* Python wrapper for mb01zd */
PyObject* py_mb01zd(PyObject* self, PyObject* args) {
    const char *side_str, *uplo_str, *trans_str, *diag_str;
    i32 m, n, l;
    f64 alpha;
    PyObject *t_obj, *h_obj;
    PyArrayObject *t_array, *h_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssssiiidOO",
                          &side_str, &uplo_str, &trans_str, &diag_str,
                          &m, &n, &l, &alpha, &t_obj, &h_obj)) {
        return NULL;
    }

    t_array = (PyArrayObject*)PyArray_FROM_OTF(t_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (t_array == NULL) {
        return NULL;
    }

    h_array = (PyArrayObject*)PyArray_FROM_OTF(h_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (h_array == NULL) {
        Py_DECREF(t_array);
        return NULL;
    }

    npy_intp *t_dims = PyArray_DIMS(t_array);
    npy_intp *h_dims = PyArray_DIMS(h_array);

    i32 ldt = (i32)t_dims[0];
    i32 ldh = (i32)h_dims[0];
    if (ldt < 1) ldt = 1;
    if (ldh < 1) ldh = 1;

    const f64 *t_data = (const f64*)PyArray_DATA(t_array);
    f64 *h_data = (f64*)PyArray_DATA(h_array);

    mb01zd(side_str, uplo_str, trans_str, diag_str, m, n, l, alpha, t_data, ldt, h_data, ldh, &info);

    Py_DECREF(t_array);
    PyArray_ResolveWritebackIfCopy(h_array);
    PyObject *result = Py_BuildValue("Oi", h_array, info);
    Py_DECREF(h_array);
    return result;
}

PyObject* py_mb01wd(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static char* kwlist[] = {"dico", "uplo", "trans", "hess", "n", "alpha", "beta",
                             "r", "a", "t", NULL};

    const char *dico_str = NULL;
    const char *uplo_str = NULL;
    const char *trans_str = NULL;
    const char *hess_str = NULL;
    int n;
    double alpha, beta;
    PyObject *r_obj = NULL;
    PyObject *a_obj = NULL;
    PyObject *t_obj = NULL;
    PyArrayObject *r_array = NULL;
    PyArrayObject *a_array = NULL;
    PyArrayObject *t_array = NULL;
    i32 info = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssssiddOOO:mb01wd", kwlist,
                                     &dico_str, &uplo_str, &trans_str, &hess_str,
                                     &n, &alpha, &beta, &r_obj, &a_obj, &t_obj)) {
        return NULL;
    }

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

    t_array = (PyArrayObject*)PyArray_FROM_OTF(t_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (t_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *r_dims = PyArray_DIMS(r_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *t_dims = PyArray_DIMS(t_array);

    i32 ldr = (i32)r_dims[0];
    i32 lda = (i32)a_dims[0];
    i32 ldt = (i32)t_dims[0];
    if (ldr < 1) ldr = 1;
    if (lda < 1) lda = 1;
    if (ldt < 1) ldt = 1;

    f64 *r_data = (f64*)PyArray_DATA(r_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    const f64 *t_data = (const f64*)PyArray_DATA(t_array);

    mb01wd(dico_str, uplo_str, trans_str, hess_str, n, alpha, beta,
           r_data, ldr, a_data, lda, t_data, ldt, &info);

    Py_DECREF(t_array);
    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(r_array);
    PyObject *result = Py_BuildValue("OOi", a_array, r_array, info);
    Py_DECREF(a_array);
    Py_DECREF(r_array);
    return result;
}
