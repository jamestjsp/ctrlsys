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



/* Python wrapper for tb01vd */
PyObject* py_tb01vd(PyObject* self, PyObject* args, PyObject* kwargs) {
    i32 n, m, l;
    const char *apply = "N";
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *x0_obj;
    PyArrayObject *a_array, *b_array, *c_array, *d_array, *x0_array;

    static char *kwlist[] = {"n", "m", "l", "a", "b", "c", "d", "x0", "apply", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiiOOOOO|s", kwlist,
                                     &n, &m, &l, &a_obj, &b_obj, &c_obj, &d_obj, &x0_obj, &apply)) {
        return NULL;
    }

    if (n < 0 || m < 0 || l < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative (n=%d, m=%d, l=%d)", n, m, l);
        return NULL;
    }

    if (!(apply[0] == 'A' || apply[0] == 'a' || apply[0] == 'N' || apply[0] == 'n')) {
        PyErr_Format(PyExc_ValueError, "apply must be 'A' or 'N', got '%s'", apply);
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

    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                               NPY_ARRAY_IN_FARRAY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    x0_array = (PyArrayObject*)PyArray_FROM_OTF(x0_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (x0_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = l > 0 ? l : 1;
    i32 ldd = l > 0 ? l : 1;
    i32 ltheta = n * (l + m + 1) + l * m;

    i32 ldwork_min1 = n * n * l + n * l + n;
    i32 max_nl = n > l ? n : l;
    i32 inner1 = n * n + n * max_nl + 6 * n + (n < l ? n : l);
    i32 inner2 = n * m;
    i32 inner_max = inner1 > inner2 ? inner1 : inner2;
    i32 ldwork_min2 = n * n + inner_max;
    i32 ldwork = ldwork_min1 > ldwork_min2 ? ldwork_min1 : ldwork_min2;
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(x0_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }

    npy_intp theta_dims[1] = {ltheta > 0 ? ltheta : 1};
    PyObject *theta_array = PyArray_SimpleNew(1, theta_dims, NPY_DOUBLE);
    if (theta_array == NULL) {
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(x0_array);
        return NULL;
    }
    f64 *theta_data = (f64*)PyArray_DATA((PyArrayObject*)theta_array);
    memset(theta_data, 0, (ltheta > 0 ? ltheta : 1) * sizeof(f64));

    i32 info;
    f64 scale = 0.0;
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *x0_data = (f64*)PyArray_DATA(x0_array);

    tb01vd(apply, n, m, l, a_data, lda, b_data, ldb, c_data, ldc,
           d_data, ldd, x0_data, theta_data, ltheta, &scale,
           dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(x0_array);

    PyObject *result = Py_BuildValue("OOOOdi", theta_array, a_array, b_array, c_array, scale, info);

    Py_DECREF(theta_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(x0_array);

    return result;
}



/* Python wrapper for tb01vy */
PyObject* py_tb01vy(PyObject* self, PyObject* args, PyObject* kwargs) {
    i32 n, m, l, ltheta;
    const char *apply = "N";
    PyObject *theta_obj;
    PyArrayObject *theta_array;

    static char *kwlist[] = {"n", "m", "l", "theta", "apply", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiiO|s", kwlist,
                                     &n, &m, &l, &theta_obj, &apply)) {
        return NULL;
    }

    if (n < 0 || m < 0 || l < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative (n=%d, m=%d, l=%d)", n, m, l);
        return NULL;
    }

    theta_array = (PyArrayObject*)PyArray_FROM_OTF(theta_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_IN_FARRAY);
    if (theta_array == NULL) return NULL;

    ltheta = (i32)PyArray_SIZE(theta_array);
    i32 required_ltheta = n * (l + m + 1) + l * m;

    if (ltheta < required_ltheta) {
        Py_DECREF(theta_array);
        PyErr_Format(PyExc_ValueError, "ltheta=%d is too small (need >= %d)", ltheta, required_ltheta);
        return NULL;
    }

    if (!(apply[0] == 'A' || apply[0] == 'a' || apply[0] == 'N' || apply[0] == 'n')) {
        Py_DECREF(theta_array);
        PyErr_Format(PyExc_ValueError, "apply must be 'A' or 'N', got '%s'", apply);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = l > 0 ? l : 1;
    i32 ldd = l > 0 ? l : 1;
    i32 ldwork = n * (n + l + 1);

    npy_intp a_dims[2] = {n, n};
    npy_intp a_strides[2] = {sizeof(f64), lda * sizeof(f64)};
    PyObject *a_array = PyArray_New(&PyArray_Type, 2, a_dims, NPY_DOUBLE,
                                    a_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (a_array == NULL) {
        Py_DECREF(theta_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }
    f64 *a_data = (f64*)PyArray_DATA((PyArrayObject*)a_array);
    if (lda > 0 && n > 0) { memset(a_data, 0, lda * n * sizeof(f64)); }

    npy_intp b_dims[2] = {n, m};
    npy_intp b_strides[2] = {sizeof(f64), ldb * sizeof(f64)};
    PyObject *b_array = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE,
                                    b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (b_array == NULL) {
        Py_XDECREF(a_array);
        Py_DECREF(theta_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }
    f64 *b_data = (f64*)PyArray_DATA((PyArrayObject*)b_array);
    if (n > 0 && m > 0) { memset(b_data, 0, ldb * m * sizeof(f64)); }

    npy_intp c_dims[2] = {l, n};
    npy_intp c_strides[2] = {sizeof(f64), ldc * sizeof(f64)};
    PyObject *c_array = PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE,
                                    c_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_DECREF(theta_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }
    f64 *c_data = (f64*)PyArray_DATA((PyArrayObject*)c_array);
    if (l > 0 && n > 0) { memset(c_data, 0, ldc * n * sizeof(f64)); }

    npy_intp d_dims[2] = {l, m};
    npy_intp d_strides[2] = {sizeof(f64), ldd * sizeof(f64)};
    PyObject *d_array = PyArray_New(&PyArray_Type, 2, d_dims, NPY_DOUBLE,
                                    d_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (d_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_DECREF(theta_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }
    f64 *d_data = (f64*)PyArray_DATA((PyArrayObject*)d_array);
    if (l > 0 && m > 0) { memset(d_data, 0, ldd * m * sizeof(f64)); }

    f64 *dwork = (f64*)malloc(ldwork > 0 ? ldwork * sizeof(f64) : sizeof(f64));
    if (dwork == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        Py_DECREF(theta_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }

    npy_intp x0_dims[1] = {n > 0 ? n : 1};
    PyObject *x0_array = PyArray_SimpleNew(1, x0_dims, NPY_DOUBLE);
    if (x0_array == NULL) {
        free(dwork);
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        Py_DECREF(theta_array);
        return NULL;
    }
    f64 *x0 = (f64*)PyArray_DATA((PyArrayObject*)x0_array);
    memset(x0, 0, (n > 0 ? n : 1) * sizeof(f64));

    i32 info;
    f64 *theta_data = (f64*)PyArray_DATA(theta_array);

    tb01vy(apply, n, m, l, theta_data, ltheta, a_data, lda, b_data, ldb,
           c_data, ldc, d_data, ldd, x0, dwork, ldwork, &info);

    free(dwork);

    PyObject *result = Py_BuildValue("OOOOOi", a_array, b_array, c_array, d_array, x0_array, info);

    Py_DECREF(theta_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(x0_array);

    return result;
}



/* Python wrapper for ma02ad */
PyObject* py_tb01wd(PyObject* self, PyObject* args) {
    i32 n, m, p;
    PyObject *a_obj, *b_obj, *c_obj;
    PyArrayObject *a_array, *b_array, *c_array;

    if (!PyArg_ParseTuple(args, "iiiOOO", &n, &m, &p, &a_obj, &b_obj, &c_obj)) {
        return NULL;
    }

    /* Validate dimensions */
    if (n < 0 || m < 0 || p < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative (n=%d, m=%d, p=%d)", n, m, p);
        return NULL;
    }

    /* Convert inputs to NumPy arrays - preserve Fortran-order */
    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
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

    /* Get leading dimensions */
    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldu = n > 0 ? n : 1;

    /* Allocate output arrays */
    npy_intp u_dims[2] = {n, n};
    npy_intp u_strides[2] = {sizeof(f64), ldu * sizeof(f64)};
    PyObject *u_array = PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE,
                                    u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (u_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }
    f64 *u_data = (f64*)PyArray_DATA((PyArrayObject*)u_array);
    memset(u_data, 0, ldu * n * sizeof(f64));

    npy_intp wr_dims[1] = {n > 0 ? n : 1};
    PyObject *wr_array = PyArray_SimpleNew(1, wr_dims, NPY_DOUBLE);
    if (wr_array == NULL) {
        Py_DECREF(u_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }
    f64 *wr = (f64*)PyArray_DATA((PyArrayObject*)wr_array);
    memset(wr, 0, (n > 0 ? n : 1) * sizeof(f64));

    npy_intp wi_dims[1] = {n > 0 ? n : 1};
    PyObject *wi_array = PyArray_SimpleNew(1, wi_dims, NPY_DOUBLE);
    if (wi_array == NULL) {
        Py_DECREF(wr_array);
        Py_DECREF(u_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }
    f64 *wi = (f64*)PyArray_DATA((PyArrayObject*)wi_array);
    memset(wi, 0, (n > 0 ? n : 1) * sizeof(f64));

    i32 ldwork = n > 0 ? 3 * n : 1;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(wi_array);
        Py_DECREF(wr_array);
        Py_DECREF(u_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    /* Call C function */
    i32 info;
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    tb01wd(n, m, p, a_data, lda, b_data, ldb, c_data, ldc,
           u_data, ldu, wr, wi, dwork, ldwork, &info);

    free(dwork);

    /* Build result tuple */
    PyObject *result = Py_BuildValue("OOOOOOi", a_array, b_array, c_array,
                                     u_array, wr_array, wi_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(u_array);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);

    return result;
}



/* Python wrapper for tb01ud */
PyObject* py_tb01ud(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *jobz_str;
    i32 n, m, p;
    PyObject *a_obj, *b_obj, *c_obj;
    f64 tol = 0.0;

    static char *kwlist[] = {"jobz", "n", "m", "p", "a", "b", "c", "tol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siiiOOO|d", kwlist,
                                     &jobz_str, &n, &m, &p, &a_obj, &b_obj, &c_obj, &tol)) {
        return NULL;
    }

    char jobz = (char)toupper((unsigned char)jobz_str[0]);
    if (jobz != 'N' && jobz != 'F' && jobz != 'I') {
        PyErr_SetString(PyExc_ValueError, "jobz must be 'N', 'F', or 'I'");
        return NULL;
    }

    if (n < 0) {
        PyErr_Format(PyExc_ValueError, "n must be non-negative (got %d)", n);
        return NULL;
    }
    if (m < 0) {
        PyErr_Format(PyExc_ValueError, "m must be non-negative (got %d)", m);
        return NULL;
    }
    if (p < 0) {
        PyErr_Format(PyExc_ValueError, "p must be non-negative (got %d)", p);
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!c_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldz = (jobz == 'I' || jobz == 'F') ? (n > 0 ? n : 1) : 1;

    i32 n_alloc = n > 0 ? n : 1;
    i32 m_alloc = m > 0 ? m : 1;

    npy_intp z_dims[2] = {n_alloc, n_alloc};
    npy_intp z_strides[2] = {sizeof(f64), ldz * sizeof(f64)};
    PyObject *z_array = PyArray_New(&PyArray_Type, 2, z_dims, NPY_DOUBLE,
                                    z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (z_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate z_array");
        return NULL;
    }
    f64 *z_data = (f64*)PyArray_DATA((PyArrayObject*)z_array);
    memset(z_data, 0, ldz * n_alloc * sizeof(f64));

    npy_intp tau_dims[1] = {n_alloc};
    PyObject *tau_array = PyArray_SimpleNew(1, tau_dims, NPY_DOUBLE);
    if (tau_array == NULL) {
        Py_XDECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }
    f64 *tau = (f64*)PyArray_DATA((PyArrayObject*)tau_array);
    memset(tau, 0, n_alloc * sizeof(f64));

    npy_intp nblk_dims[1] = {n_alloc};
    PyObject *nblk_array = PyArray_SimpleNew(1, nblk_dims, NPY_INT32);
    if (nblk_array == NULL) {
        Py_DECREF(tau_array);
        Py_XDECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }
    i32 *nblk = (i32*)PyArray_DATA((PyArrayObject*)nblk_array);
    memset(nblk, 0, n_alloc * sizeof(i32));

    i32 *iwork = (i32*)calloc(m_alloc, sizeof(i32));

    i32 ldwork = 1;
    if (n > ldwork) ldwork = n;
    if (3 * m > ldwork) ldwork = 3 * m;
    if (p > ldwork) ldwork = p;
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        Py_DECREF(nblk_array);
        Py_DECREF(tau_array);
        Py_XDECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    i32 ncont, indcon, info;
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    tb01ud(&jobz, n, m, p, a_data, lda, b_data, ldb, c_data, ldc,
           &ncont, &indcon, nblk, z_data, ldz, tau, tol,
           iwork, dwork, ldwork, &info);

    free(dwork);
    free(iwork);

    PyObject *result = Py_BuildValue("OOOiiOOOi",
                                     a_array, b_array, c_array,
                                     ncont, indcon, nblk_array,
                                     z_array, tau_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(z_array);
    Py_DECREF(tau_array);
    Py_DECREF(nblk_array);

    return result;
}



/* Python wrapper for tb01zd */
PyObject* py_tb01zd(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *jobz_str;
    i32 n, p;
    PyObject *a_obj, *b_obj, *c_obj;
    f64 tol = 0.0;

    static char *kwlist[] = {"jobz", "n", "p", "a", "b", "c", "tol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siiOOO|d", kwlist,
                                     &jobz_str, &n, &p, &a_obj, &b_obj, &c_obj, &tol)) {
        return NULL;
    }

    char jobz = (char)toupper((unsigned char)jobz_str[0]);
    if (jobz != 'N' && jobz != 'F' && jobz != 'I') {
        PyErr_SetString(PyExc_ValueError, "jobz must be 'N', 'F', or 'I'");
        return NULL;
    }

    if (n < 0) {
        PyErr_Format(PyExc_ValueError, "n must be non-negative (got %d)", n);
        return NULL;
    }
    if (p < 0) {
        PyErr_Format(PyExc_ValueError, "p must be non-negative (got %d)", p);
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!c_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldz = (jobz == 'I' || jobz == 'F') ? (n > 0 ? n : 1) : 1;

    i32 n_alloc = n > 0 ? n : 1;

    npy_intp z_dims[2] = {n_alloc, n_alloc};
    npy_intp z_strides[2] = {sizeof(f64), ldz * sizeof(f64)};
    PyObject *z_array = PyArray_New(&PyArray_Type, 2, z_dims, NPY_DOUBLE,
                                    z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (z_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }
    f64 *z_data = (f64*)PyArray_DATA((PyArrayObject*)z_array);
    memset(z_data, 0, ldz * n_alloc * sizeof(f64));

    npy_intp tau_dims[1] = {n_alloc};
    PyObject *tau_array = PyArray_SimpleNew(1, tau_dims, NPY_DOUBLE);
    if (tau_array == NULL) {
        Py_DECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }
    f64 *tau = (f64*)PyArray_DATA((PyArrayObject*)tau_array);
    memset(tau, 0, n_alloc * sizeof(f64));

    i32 ldwork = n > p ? n : p;
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!dwork) {
        Py_DECREF(tau_array);
        Py_DECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    i32 ncont, info;
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    tb01zd(&jobz, n, p, a_data, lda, b_data, c_data, ldc,
           &ncont, z_data, ldz, tau, tol, dwork, ldwork, &info);

    free(dwork);

    PyObject *result = Py_BuildValue("OOOiOOi",
                                     a_array, b_array, c_array,
                                     ncont, z_array, tau_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(z_array);
    Py_DECREF(tau_array);

    return result;
}



/* Python wrapper for tb01pd */
PyObject* py_tb01pd(PyObject* self, PyObject* args) {
    const char *job_str, *equil_str;
    PyObject *a_obj, *b_obj, *c_obj;
    f64 tol;

    if (!PyArg_ParseTuple(args, "ssOOOd", &job_str, &equil_str,
                          &a_obj, &b_obj, &c_obj, &tol)) {
        return NULL;
    }

    char job = (char)toupper((unsigned char)job_str[0]);
    char equil = (char)toupper((unsigned char)equil_str[0]);

    if (job != 'M' && job != 'C' && job != 'O') {
        PyErr_SetString(PyExc_ValueError, "job must be 'M', 'C', or 'O'");
        return NULL;
    }
    if (equil != 'S' && equil != 'N') {
        PyErr_SetString(PyExc_ValueError, "equil must be 'S' or 'N'");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!c_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    int a_ndim = PyArray_NDIM(a_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    i32 n, m, p;
    if (a_ndim == 2) {
        n = (i32)a_dims[0];
        if (a_dims[0] != a_dims[1]) {
            PyErr_SetString(PyExc_ValueError, "A must be square");
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            return NULL;
        }
    } else if (a_ndim == 0 || (a_ndim == 2 && a_dims[0] == 0)) {
        n = 0;
    } else {
        PyErr_SetString(PyExc_ValueError, "A must be 2D array");
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    if (n > 0) {
        if ((i32)b_dims[0] != n) {
            PyErr_SetString(PyExc_ValueError, "B must have N rows");
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            return NULL;
        }
        if ((i32)c_dims[1] != n) {
            PyErr_SetString(PyExc_ValueError, "C must have N columns");
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            return NULL;
        }
    }

    m = (i32)b_dims[1];
    p = (i32)c_dims[0];

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 maxmp = m > p ? m : p;
    i32 ldc = n > 0 ? (maxmp > 1 ? maxmp : 1) : 1;

    i32 ldwork = n + (n > 3 * maxmp ? n : 3 * maxmp);
    if (ldwork < 1) ldwork = 1;

    i32 iwork_size = n + maxmp;
    if (iwork_size < 1) iwork_size = 1;

    f64 *b_work = NULL;
    f64 *c_work = NULL;

    if (job != 'C' && (i32)b_dims[1] < maxmp) {
        b_work = (f64*)calloc(n * maxmp, sizeof(f64));
        if (!b_work) {
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            PyErr_NoMemory();
            return NULL;
        }
        f64 *b_data = (f64*)PyArray_DATA(b_array);
        for (i32 j = 0; j < m; j++) {
            for (i32 i = 0; i < n; i++) {
                b_work[i + j * n] = b_data[i + j * n];
            }
        }
    }

    if ((i32)c_dims[0] < maxmp) {
        c_work = (f64*)calloc(maxmp * n, sizeof(f64));
        if (!c_work) {
            free(b_work);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            PyErr_NoMemory();
            return NULL;
        }
        f64 *c_data = (f64*)PyArray_DATA(c_array);
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < p; i++) {
                c_work[i + j * maxmp] = c_data[i + j * p];
            }
        }
    }

    i32 *iwork = (i32*)calloc(iwork_size, sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        free(b_work);
        free(c_work);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = b_work ? b_work : (f64*)PyArray_DATA(b_array);
    f64 *c_data = c_work ? c_work : (f64*)PyArray_DATA(c_array);

    i32 nr = 0;
    i32 info = 0;

    char job_str_c[2] = {job, '\0'};
    char equil_str_c[2] = {equil, '\0'};

    tb01pd(job_str_c, equil_str_c, n, m, p, a_data, lda, b_data, ldb, c_data, ldc,
           &nr, tol, iwork, dwork, ldwork, &info);

    free(dwork);

    if (b_work) {
        f64 *b_orig = (f64*)PyArray_DATA(b_array);
        for (i32 j = 0; j < m; j++) {
            for (i32 i = 0; i < n; i++) {
                b_orig[i + j * n] = b_work[i + j * n];
            }
        }
        free(b_work);
    }
    if (c_work) {
        f64 *c_orig = (f64*)PyArray_DATA(c_array);
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < p; i++) {
                c_orig[i + j * p] = c_work[i + j * maxmp];
            }
        }
        free(c_work);
    }

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    npy_intp iwork_dims[1] = {n > 0 ? n : 0};
    PyObject *iwork_array = NULL;
    if (n > 0) {
        iwork_array = PyArray_SimpleNew(1, iwork_dims, NPY_INT32);
        if (!iwork_array) {
            free(iwork);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            return NULL;
        }
        memcpy(PyArray_DATA((PyArrayObject*)iwork_array), iwork, n * sizeof(i32));
        free(iwork);
    } else {
        iwork_array = PyArray_ZEROS(1, iwork_dims, NPY_INT32, 0);
        free(iwork);
        if (!iwork_array) {
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            return NULL;
        }
    }

    PyObject *result = Py_BuildValue("OOOiOi", a_array, b_array, c_array,
                                     (int)nr, iwork_array, (int)info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(iwork_array);

    return result;
}



/* Python wrapper for tb01xd */
PyObject* py_tb01xd(PyObject* self, PyObject* args) {
    const char *jobd_str;
    i32 n, m, p, kl, ku;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "siiiiiOOOO", &jobd_str, &n, &m, &p, &kl, &ku,
                          &a_obj, &b_obj, &c_obj, &d_obj)) {
        return NULL;
    }

    char jobd = (char)toupper((unsigned char)jobd_str[0]);
    if (jobd != 'D' && jobd != 'Z') {
        PyErr_SetString(PyExc_ValueError, "jobd must be 'D' or 'Z'");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!c_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!d_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);
    npy_intp *d_dims = PyArray_DIMS(d_array);

    i32 lda = PyArray_NDIM(a_array) >= 1 ? (i32)a_dims[0] : 1;
    i32 ldb = PyArray_NDIM(b_array) >= 1 ? (i32)b_dims[0] : 1;
    i32 ldc = PyArray_NDIM(c_array) >= 1 ? (i32)c_dims[0] : 1;
    i32 ldd = PyArray_NDIM(d_array) >= 1 ? (i32)d_dims[0] : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);

    i32 info;
    tb01xd(&jobd, n, m, p, kl, ku, a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    PyObject *result = Py_BuildValue("OOOOi", a_array, b_array, c_array, d_array, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    return result;
}



/* Python wrapper for tb01id */
PyObject* py_tb01id(PyObject* self, PyObject* args) {
    const char *job_str;
    PyObject *a_obj, *b_obj, *c_obj;
    f64 maxred_in;

    if (!PyArg_ParseTuple(args, "sOOOd", &job_str, &a_obj, &b_obj, &c_obj, &maxred_in)) {
        return NULL;
    }

    char job = (char)toupper((unsigned char)job_str[0]);
    if (job != 'A' && job != 'B' && job != 'C' && job != 'N') {
        PyErr_SetString(PyExc_ValueError, "job must be 'A', 'B', 'C', or 'N'");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!c_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    i32 n = (i32)a_dims[0];
    i32 m = PyArray_NDIM(b_array) >= 2 ? (i32)b_dims[1] : 0;
    i32 p = PyArray_NDIM(c_array) >= 1 ? (i32)c_dims[0] : 0;
    i32 lda = n > 0 ? n : 1;
    i32 ldb = (m > 0 && n > 0) ? n : 1;
    i32 ldc = p > 0 ? p : 1;

    f64 *scale = (f64*)malloc((n > 0 ? n : 1) * sizeof(f64));
    if (!scale) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    f64 maxred = maxred_in;
    i32 info = 0;

    tb01id(&job, n, m, p, &maxred, a_data, lda, b_data, ldb, c_data, ldc, scale, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    npy_intp scale_dims[1] = {n};
    PyObject *scale_array = PyArray_SimpleNew(1, scale_dims, NPY_DOUBLE);
    if (!scale_array) {
        free(scale);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)scale_array), scale, n * sizeof(f64));
    free(scale);

    PyObject *result = Py_BuildValue("OOOdOi", a_array, b_array, c_array, maxred, scale_array, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(scale_array);

    return result;
}



/* Python wrapper for tb01iz - Complex system matrix balancing */
PyObject* py_tb01iz(PyObject* self, PyObject* args) {
    const char *job_str;
    PyObject *a_obj, *b_obj, *c_obj;
    double maxred_in;

    if (!PyArg_ParseTuple(args, "sOOOd", &job_str, &a_obj, &b_obj, &c_obj, &maxred_in)) {
        return NULL;
    }

    char job = job_str[0];

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!c_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 n = (i32)a_dims[0];

    npy_intp *b_dims = PyArray_DIMS(b_array);
    i32 m = (PyArray_NDIM(b_array) >= 2) ? (i32)b_dims[1] : (n > 0 ? (i32)b_dims[0] : 0);
    if (PyArray_NDIM(b_array) < 2 && n > 0) {
        m = (i32)b_dims[1];
    }
    if (PyArray_NDIM(b_array) >= 2) {
        m = (i32)b_dims[1];
    } else if (n == 0) {
        m = 0;
    } else {
        m = 1;
    }

    npy_intp *c_dims = PyArray_DIMS(c_array);
    i32 p = (PyArray_NDIM(c_array) >= 1) ? (i32)c_dims[0] : 0;

    i32 lda = n > 0 ? n : 1;
    i32 ldb = (m > 0 && n > 0) ? n : 1;
    i32 ldc = p > 0 ? p : 1;

    f64 *scale = NULL;
    if (n > 0) {
        scale = (f64*)malloc(n * sizeof(f64));
        if (!scale) {
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            PyErr_NoMemory();
            return NULL;
        }
    }

    c128 *a_data = (c128*)PyArray_DATA(a_array);
    c128 *b_data = (c128*)PyArray_DATA(b_array);
    c128 *c_data = (c128*)PyArray_DATA(c_array);

    f64 maxred = maxred_in;
    i32 info = slicot_tb01iz(job, n, m, p, &maxred, a_data, lda, b_data, ldb, c_data, ldc, scale);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    npy_intp scale_dims[1] = {n};
    PyObject *scale_array = PyArray_SimpleNew(1, scale_dims, NPY_DOUBLE);
    if (!scale_array) {
        free(scale);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }
    if (n > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)scale_array), scale, n * sizeof(f64));
    }
    free(scale);

    PyObject *result = Py_BuildValue("OOOdOi", a_array, b_array, c_array, maxred, scale_array, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(scale_array);
    return result;
}



/* Python wrapper for tb03ad */
PyObject* py_tb03ad(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *leri_str, *equil_str;
    int n_py, m_py, p_py;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    double tol = 0.0;

    static char *kwlist[] = {"leri", "equil", "n", "m", "p", "a", "b", "c", "d", "tol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssiiiOOOO|d", kwlist,
                                     &leri_str, &equil_str, &n_py, &m_py, &p_py,
                                     &a_obj, &b_obj, &c_obj, &d_obj, &tol)) {
        return NULL;
    }

    char leri = (char)toupper((unsigned char)leri_str[0]);
    char equil = (char)toupper((unsigned char)equil_str[0]);

    if (leri != 'L' && leri != 'R') {
        PyErr_SetString(PyExc_ValueError, "leri must be 'L' or 'R'");
        return NULL;
    }
    if (equil != 'S' && equil != 'N') {
        PyErr_SetString(PyExc_ValueError, "equil must be 'S' or 'N'");
        return NULL;
    }

    i32 n = (i32)n_py;
    i32 m = (i32)m_py;
    i32 p = (i32)p_py;

    if (n < 0) {
        PyErr_Format(PyExc_ValueError, "n must be non-negative (got %d)", n);
        return NULL;
    }
    if (m < 0) {
        PyErr_Format(PyExc_ValueError, "m must be non-negative (got %d)", m);
        return NULL;
    }
    if (p < 0) {
        PyErr_Format(PyExc_ValueError, "p must be non-negative (got %d)", p);
        return NULL;
    }

    PyArrayObject *a_in = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (!a_in) return NULL;

    PyArrayObject *b_in = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (!b_in) {
        Py_DECREF(a_in);
        return NULL;
    }

    PyArrayObject *c_in = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (!c_in) {
        Py_DECREF(a_in);
        Py_DECREF(b_in);
        return NULL;
    }

    PyArrayObject *d_in = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (!d_in) {
        Py_DECREF(a_in);
        Py_DECREF(b_in);
        Py_DECREF(c_in);
        return NULL;
    }

    i32 pwork = (leri == 'R') ? m : p;
    i32 mwork = (leri == 'R') ? p : m;
    i32 maxmp = (m > p) ? m : p;
    i32 mplim = (1 > maxmp) ? 1 : maxmp;

    i32 n_alloc = (n > 0) ? n : 1;
    i32 pwork_alloc = (pwork > 0) ? pwork : 1;
    i32 mwork_alloc = (mwork > 0) ? mwork : 1;
    i32 maxmp_alloc = (maxmp > 0) ? maxmp : 1;

    i32 lda = n_alloc;
    i32 ldb = n_alloc;
    i32 ldc = mplim;
    i32 ldd = mplim;

    npy_intp a_dims[2] = {n_alloc, n_alloc};
    npy_intp b_dims[2] = {n_alloc, maxmp_alloc};
    npy_intp c_dims[2] = {mplim, n_alloc};
    npy_intp d_dims[2] = {mplim, maxmp_alloc};

    PyObject *a_array = PyArray_ZEROS(2, a_dims, NPY_DOUBLE, 1);
    PyObject *b_array = PyArray_ZEROS(2, b_dims, NPY_DOUBLE, 1);
    PyObject *c_array = PyArray_ZEROS(2, c_dims, NPY_DOUBLE, 1);
    PyObject *d_array = PyArray_ZEROS(2, d_dims, NPY_DOUBLE, 1);

    if (!a_array || !b_array || !c_array || !d_array) {
        Py_XDECREF(a_array); Py_XDECREF(b_array);
        Py_XDECREF(c_array); Py_XDECREF(d_array);
        Py_DECREF(a_in); Py_DECREF(b_in);
        Py_DECREF(c_in); Py_DECREF(d_in);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA((PyArrayObject*)a_array);
    f64 *b_data = (f64*)PyArray_DATA((PyArrayObject*)b_array);
    f64 *c_data = (f64*)PyArray_DATA((PyArrayObject*)c_array);
    f64 *d_data = (f64*)PyArray_DATA((PyArrayObject*)d_array);

    f64 *a_in_data = (f64*)PyArray_DATA(a_in);
    f64 *b_in_data = (f64*)PyArray_DATA(b_in);
    f64 *c_in_data = (f64*)PyArray_DATA(c_in);
    f64 *d_in_data = (f64*)PyArray_DATA(d_in);

    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < n; i++) {
            a_data[i + j * lda] = a_in_data[i + j * n];
        }
    }
    for (i32 j = 0; j < m; j++) {
        for (i32 i = 0; i < n; i++) {
            b_data[i + j * ldb] = b_in_data[i + j * n];
        }
    }
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < p; i++) {
            c_data[i + j * ldc] = c_in_data[i + j * p];
        }
    }
    for (i32 j = 0; j < m; j++) {
        for (i32 i = 0; i < p; i++) {
            d_data[i + j * ldd] = d_in_data[i + j * p];
        }
    }

    Py_DECREF(a_in);
    Py_DECREF(b_in);
    Py_DECREF(c_in);
    Py_DECREF(d_in);

    i32 ldpco1 = pwork_alloc;
    i32 ldpco2 = pwork_alloc;
    i32 ldqco1 = (leri == 'R') ? mplim : pwork_alloc;
    i32 ldqco2 = (leri == 'R') ? mplim : mwork_alloc;
    i32 ldvco1 = pwork_alloc;
    i32 ldvco2 = n_alloc;

    i32 kpcoef = n_alloc + 1;
    npy_intp pcoeff_dims[3] = {ldpco1, ldpco2, kpcoef};
    npy_intp qcoeff_dims[3] = {ldqco1, ldqco2, kpcoef};
    npy_intp vcoeff_dims[3] = {ldvco1, ldvco2, kpcoef};
    npy_intp index_dims[1] = {maxmp_alloc};
    npy_intp iwork_dims[1] = {n_alloc + maxmp_alloc};

    PyObject *pcoeff_array = PyArray_ZEROS(3, pcoeff_dims, NPY_DOUBLE, 1);
    PyObject *qcoeff_array = PyArray_ZEROS(3, qcoeff_dims, NPY_DOUBLE, 1);
    PyObject *vcoeff_array = PyArray_ZEROS(3, vcoeff_dims, NPY_DOUBLE, 1);
    PyObject *index_array = PyArray_ZEROS(1, index_dims, NPY_INT32, 0);
    PyObject *iwork_array = PyArray_ZEROS(1, iwork_dims, NPY_INT32, 0);

    i32 ldwork = n + n;
    if (n + 3 * maxmp > ldwork) ldwork = n + 3 * maxmp;
    if (pwork * (pwork + 2) > ldwork) ldwork = pwork * (pwork + 2);
    if (ldwork < 1) ldwork = 1;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!pcoeff_array || !qcoeff_array || !vcoeff_array || !index_array || !iwork_array || !dwork) {
        Py_XDECREF(pcoeff_array); Py_XDECREF(qcoeff_array); Py_XDECREF(vcoeff_array);
        Py_XDECREF(index_array); Py_XDECREF(iwork_array); free(dwork);
        Py_DECREF(a_array); Py_DECREF(b_array);
        Py_DECREF(c_array); Py_DECREF(d_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    f64 *pcoeff = (f64*)PyArray_DATA((PyArrayObject*)pcoeff_array);
    f64 *qcoeff = (f64*)PyArray_DATA((PyArrayObject*)qcoeff_array);
    f64 *vcoeff = (f64*)PyArray_DATA((PyArrayObject*)vcoeff_array);
    i32 *index_data = (i32*)PyArray_DATA((PyArrayObject*)index_array);
    i32 *iwork = (i32*)PyArray_DATA((PyArrayObject*)iwork_array);

    i32 nr, info;

    char leri_c[2] = {leri, '\0'};
    char equil_c[2] = {equil, '\0'};

    tb03ad(leri_c, equil_c, n, m, p,
           a_data, lda, b_data, ldb, c_data, ldc,
           d_data, ldd, &nr, index_data,
           pcoeff, ldpco1, ldpco2,
           qcoeff, ldqco1, ldqco2,
           vcoeff, ldvco1, ldvco2,
           tol, iwork, dwork, ldwork, &info);

    free(dwork);

    PyObject *result = Py_BuildValue("OOOiOOOOOi",
                                     a_array, b_array, c_array, nr,
                                     index_array, pcoeff_array, qcoeff_array,
                                     vcoeff_array, iwork_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(pcoeff_array);
    Py_DECREF(qcoeff_array);
    Py_DECREF(vcoeff_array);
    Py_DECREF(index_array);
    Py_DECREF(iwork_array);

    return result;
}



/* Python wrapper for tb03ay */
PyObject* py_tb03ay(PyObject* self, PyObject* args) {
    int nr_py;
    PyObject *a_obj, *nblk_obj, *vcoeff_obj, *pcoeff_obj;

    if (!PyArg_ParseTuple(args, "iOOOO", &nr_py, &a_obj, &nblk_obj,
                          &vcoeff_obj, &pcoeff_obj)) {
        return NULL;
    }

    i32 nr = (i32)nr_py;

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (!a_array) return NULL;

    PyArrayObject *nblk_array = (PyArrayObject*)PyArray_FROM_OTF(
        nblk_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (!nblk_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *vcoeff_array = (PyArrayObject*)PyArray_FROM_OTF(
        vcoeff_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!vcoeff_array) {
        Py_DECREF(a_array);
        Py_DECREF(nblk_array);
        return NULL;
    }

    PyArrayObject *pcoeff_array = (PyArrayObject*)PyArray_FROM_OTF(
        pcoeff_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!pcoeff_array) {
        Py_DECREF(a_array);
        Py_DECREF(nblk_array);
        Py_DECREF(vcoeff_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *vcoeff_dims = PyArray_DIMS(vcoeff_array);
    npy_intp *pcoeff_dims = PyArray_DIMS(pcoeff_array);

    i32 lda = (i32)a_dims[0];
    i32 indblk = (i32)PyArray_SIZE(nblk_array);
    i32 ldvco1 = (i32)vcoeff_dims[0];
    i32 ldvco2 = (i32)vcoeff_dims[1];
    i32 ldpco1 = (i32)pcoeff_dims[0];
    i32 ldpco2 = (i32)pcoeff_dims[1];

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    i32 *nblk_data = (i32*)PyArray_DATA(nblk_array);
    f64 *vcoeff_data = (f64*)PyArray_DATA(vcoeff_array);
    f64 *pcoeff_data = (f64*)PyArray_DATA(pcoeff_array);

    i32 info;
    tb03ay(nr, a_data, lda, indblk, nblk_data,
           vcoeff_data, ldvco1, ldvco2,
           pcoeff_data, ldpco1, ldpco2, &info);

    PyArray_ResolveWritebackIfCopy(vcoeff_array);
    PyArray_ResolveWritebackIfCopy(pcoeff_array);

    PyObject *result = Py_BuildValue("OOi", vcoeff_array, pcoeff_array, info);
    Py_DECREF(a_array);
    Py_DECREF(nblk_array);
    Py_DECREF(vcoeff_array);
    Py_DECREF(pcoeff_array);

    return result;
}



/* Python wrapper for tb04ad */
PyObject* py_tb04ad(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *rowcol_str;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    f64 tol1 = 0.0, tol2 = 0.0;

    static char *kwlist[] = {"rowcol", "a", "b", "c", "d", "tol1", "tol2", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sOOOO|dd", kwlist,
                                     &rowcol_str, &a_obj, &b_obj, &c_obj, &d_obj, &tol1, &tol2)) {
        return NULL;
    }

    char rowcol = (char)toupper((unsigned char)rowcol_str[0]);

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) { Py_DECREF(a_array); return NULL; }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!c_array) { Py_DECREF(a_array); Py_DECREF(b_array); return NULL; }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!d_array) { Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); return NULL; }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 m = (i32)PyArray_DIM(b_array, 1);
    i32 p = (i32)PyArray_DIM(c_array, 0);

    i32 maxmp = m > p ? m : p;
    i32 pwork = (rowcol == 'R') ? p : m;
    i32 mwork = (rowcol == 'R') ? m : p;

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = (rowcol == 'C') ? (maxmp > 0 ? maxmp : 1) : (p > 0 ? p : 1);
    i32 ldd = (rowcol == 'C') ? (maxmp > 0 ? maxmp : 1) : (p > 0 ? p : 1);
    i32 lddcoe = pwork > 0 ? pwork : 1;
    i32 lduco1 = pwork > 0 ? pwork : 1;
    i32 lduco2 = mwork > 0 ? mwork : 1;

    i32 n_alloc = n > 0 ? n : 1;
    i32 pwork_alloc = pwork > 0 ? pwork : 1;

    npy_intp index_dims[1] = {pwork_alloc};
    PyObject *index_array = PyArray_SimpleNew(1, index_dims, NPY_INT32);
    if (!index_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }
    i32 *index_data = (i32*)PyArray_DATA((PyArrayObject*)index_array);
    memset(index_data, 0, pwork_alloc * sizeof(i32));

    npy_intp dcoeff_dims[2] = {lddcoe, n_alloc + 1};
    npy_intp dcoeff_strides[2] = {sizeof(f64), lddcoe * sizeof(f64)};
    PyObject *dcoeff_array = PyArray_New(&PyArray_Type, 2, dcoeff_dims, NPY_DOUBLE,
                                         dcoeff_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!dcoeff_array) {
        Py_DECREF(index_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }
    f64 *dcoeff_data = (f64*)PyArray_DATA((PyArrayObject*)dcoeff_array);
    memset(dcoeff_data, 0, lddcoe * (n_alloc + 1) * sizeof(f64));

    npy_intp ucoeff_dims[3] = {lduco1, lduco2, n_alloc + 1};
    npy_intp ucoeff_strides[3] = {sizeof(f64), lduco1 * sizeof(f64), lduco1 * lduco2 * sizeof(f64)};
    PyObject *ucoeff_array = PyArray_New(&PyArray_Type, 3, ucoeff_dims, NPY_DOUBLE,
                                         ucoeff_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!ucoeff_array) {
        Py_DECREF(dcoeff_array);
        Py_DECREF(index_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }
    f64 *ucoeff_data = (f64*)PyArray_DATA((PyArrayObject*)ucoeff_array);
    memset(ucoeff_data, 0, lduco1 * lduco2 * (n_alloc + 1) * sizeof(f64));

    i32 *iwork = (i32*)calloc(n_alloc + maxmp + 1, sizeof(i32));

    i32 mp = (rowcol == 'R') ? m : p;
    i32 pm = (rowcol == 'R') ? p : m;
    i32 ldwork = n * (n + 1) +
                 (n * mp + 2 * n + (n > mp ? n : mp));
    i32 ldwork2 = 3 * mp;
    i32 ldwork3 = pm;
    if (ldwork2 > ldwork) ldwork = ldwork2;
    if (ldwork3 > ldwork) ldwork = ldwork3;
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        Py_DECREF(ucoeff_array);
        Py_DECREF(dcoeff_array);
        Py_DECREF(index_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    i32 nr, info;
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);

    tb04ad(&rowcol, n, m, p, a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           &nr, index_data, dcoeff_data, lddcoe, ucoeff_data, lduco1, lduco2,
           tol1, tol2, iwork, dwork, ldwork, &info);

    free(dwork);
    free(iwork);

    PyObject *result = Py_BuildValue("OOOOiOOOi",
                                     a_array, b_array, c_array, d_array,
                                     nr, index_array, dcoeff_array, ucoeff_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(index_array);
    Py_DECREF(dcoeff_array);
    Py_DECREF(ucoeff_array);

    return result;
}



/* Python wrapper for tb05ad */
PyObject* py_tb05ad(PyObject* self, PyObject* args) {
    const char *baleig_str, *inita_str;
    PyObject *a_obj, *b_obj, *c_obj;
    Py_complex freq_py;

    if (!PyArg_ParseTuple(args, "ssOOOD", &baleig_str, &inita_str,
                          &a_obj, &b_obj, &c_obj, &freq_py)) {
        return NULL;
    }

    char baleig = (char)toupper((unsigned char)baleig_str[0]);
    char inita = (char)toupper((unsigned char)inita_str[0]);
    c128 freq = freq_py.real + freq_py.imag * I;

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!c_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    i32 n = (i32)a_dims[0];
    i32 m = PyArray_NDIM(b_array) >= 2 ? (i32)b_dims[1] : (n > 0 ? 1 : 0);
    i32 p = PyArray_NDIM(c_array) >= 1 ? (i32)c_dims[0] : 0;

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldg = p > 0 ? p : 1;
    i32 ldhinv = n > 0 ? n : 1;

    bool lbalec = (baleig == 'C');
    bool lbalea = (baleig == 'A');
    bool linita = (inita == 'G');

    i32 ldwork;
    if (linita && !lbalec && !lbalea) {
        i32 max_nmp = n > m ? n : m;
        max_nmp = max_nmp > p ? max_nmp : p;
        ldwork = n - 1 + max_nmp;
    } else if (linita && (lbalec || lbalea)) {
        i32 max_nm1p1 = n > (m - 1) ? n : (m - 1);
        max_nm1p1 = max_nm1p1 > (p - 1) ? max_nm1p1 : (p - 1);
        ldwork = n + max_nm1p1;
    } else if (!linita && (lbalec || lbalea)) {
        ldwork = 2 * n;
    } else {
        ldwork = 1;
    }
    if (ldwork < 1) ldwork = 1;
    ldwork = ldwork > 3 * n ? ldwork : 3 * n;

    i32 lzwork = (lbalec || lbalea) ? n * (n + 2) : (n > 0 ? n * n : 1);
    if (lzwork < 1) lzwork = 1;

    npy_intp g_dims[2] = {p, m};
    npy_intp g_strides[2] = {sizeof(c128), ldg * sizeof(c128)};
    PyObject *g_array = PyArray_New(&PyArray_Type, 2, g_dims, NPY_CDOUBLE,
                                     g_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!g_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }
    c128 *g_data = (c128*)PyArray_DATA((PyArrayObject*)g_array);
    memset(g_data, 0, p * m * sizeof(c128));

    npy_intp eig_dims[1] = {n > 0 ? n : 1};
    PyObject *evre_array = PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);
    if (!evre_array) {
        Py_DECREF(g_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }
    f64 *evre = (f64*)PyArray_DATA((PyArrayObject*)evre_array);
    memset(evre, 0, (n > 0 ? n : 1) * sizeof(f64));

    PyObject *evim_array = PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);
    if (!evim_array) {
        Py_DECREF(evre_array);
        Py_DECREF(g_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }
    f64 *evim = (f64*)PyArray_DATA((PyArrayObject*)evim_array);
    memset(evim, 0, (n > 0 ? n : 1) * sizeof(f64));

    npy_intp hinvb_dims[2] = {n, m};
    npy_intp hinvb_strides[2] = {sizeof(c128), ldhinv * sizeof(c128)};
    PyObject *hinvb_array = PyArray_New(&PyArray_Type, 2, hinvb_dims, NPY_CDOUBLE,
                                         hinvb_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!hinvb_array) {
        Py_DECREF(evim_array);
        Py_DECREF(evre_array);
        Py_DECREF(g_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }
    c128 *hinvb_data = (c128*)PyArray_DATA((PyArrayObject*)hinvb_array);
    memset(hinvb_data, 0, n * m * sizeof(c128));

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    c128 *zwork = (c128*)malloc(lzwork * sizeof(c128));

    if (!dwork || !zwork) {
        free(dwork);
        free(zwork);
        Py_DECREF(hinvb_array);
        Py_DECREF(evim_array);
        Py_DECREF(evre_array);
        Py_DECREF(g_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 rcond = 0.0;

    i32 info = slicot_tb05ad(baleig, inita, n, m, p, freq,
                              a_data, lda, b_data, ldb, c_data, ldc,
                              &rcond, g_data, ldg, evre, evim,
                              hinvb_data, ldhinv, dwork, ldwork, zwork, lzwork);

    free(dwork);
    free(zwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("OdOOOi", g_array, rcond, evre_array, evim_array,
                                     hinvb_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(g_array);
    Py_DECREF(evre_array);
    Py_DECREF(evim_array);
    Py_DECREF(hinvb_array);

    return result;
}



/* Python wrapper for tb01ld */
PyObject* py_tb01ld(PyObject* self, PyObject* args) {
    const char *dico, *stdom, *joba;
    f64 alpha;
    PyObject *a_obj, *b_obj, *c_obj;

    if (!PyArg_ParseTuple(args, "sssdOOO", &dico, &stdom, &joba, &alpha, &a_obj, &b_obj, &c_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
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

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 m = (i32)PyArray_DIM(b_array, 1);
    i32 p = (i32)PyArray_DIM(c_array, 0);
    i32 lda = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldc = p > 1 ? p : 1;
    i32 ldu = n > 1 ? n : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    npy_intp u_dims[2] = {n, n};
    npy_intp wr_dims[1] = {n > 0 ? n : 1};
    npy_intp u_strides[2] = {sizeof(f64), n * sizeof(f64)};

    PyArrayObject *u_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE,
                                                          u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!u_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }
    f64 *u_data = (f64*)PyArray_DATA(u_array);
    memset(u_data, 0, n * n * sizeof(f64));

    PyArrayObject *wr_array = (PyArrayObject*)PyArray_SimpleNew(1, wr_dims, NPY_DOUBLE);
    if (!wr_array) {
        Py_DECREF(u_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }
    f64 *wr_data = (f64*)PyArray_DATA(wr_array);
    memset(wr_data, 0, (n > 0 ? n : 1) * sizeof(f64));

    PyArrayObject *wi_array = (PyArrayObject*)PyArray_SimpleNew(1, wr_dims, NPY_DOUBLE);
    if (!wi_array) {
        Py_DECREF(wr_array);
        Py_DECREF(u_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }
    f64 *wi_data = (f64*)PyArray_DATA(wi_array);
    memset(wi_data, 0, (n > 0 ? n : 1) * sizeof(f64));

    i32 ldwork = 3 * n > n * m ? (3 * n > n * p ? 3 * n : n * p) : (n * m > n * p ? n * m : n * p);
    if (ldwork < 1) ldwork = 1;
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (!dwork) {
        free(dwork);
        Py_DECREF(wi_array);
        Py_DECREF(wr_array);
        Py_DECREF(u_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    i32 ndim = 0;
    i32 info = 0;

    tb01ld(dico, stdom, joba, n, m, p, alpha, a_data, lda, b_data, ldb, c_data, ldc,
           &ndim, u_data, ldu, wr_data, wi_data, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("(OOOiOOOi)", a_array, b_array, c_array, ndim,
                                      u_array, wr_array, wi_array, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(u_array);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);
    return result;
}



PyObject* py_tb01kx(PyObject* self, PyObject* args) {
    i32 ndim;
    PyObject *a_obj, *b_obj, *c_obj, *u_obj;

    if (!PyArg_ParseTuple(args, "iOOOO", &ndim, &a_obj, &b_obj, &c_obj, &u_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
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

    PyArrayObject *u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (u_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 m = PyArray_NDIM(b_array) >= 2 ? (i32)PyArray_DIM(b_array, 1) : 0;
    i32 p = (i32)PyArray_DIM(c_array, 0);

    if (ndim < 0 || ndim > n) {
        PyErr_Format(PyExc_ValueError, "ndim must satisfy 0 <= ndim <= n, got ndim=%d, n=%d", ndim, n);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(u_array);
        return NULL;
    }

    i32 lda = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldc = p > 1 ? p : 1;
    i32 ldu = n > 1 ? n : 1;
    i32 ldv = n > 1 ? n : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *u_data = (f64*)PyArray_DATA(u_array);

    npy_intp v_dims[2] = {n, n};
    npy_intp v_strides[2] = {sizeof(f64), n * sizeof(f64)};

    PyArrayObject *v_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, v_dims, NPY_DOUBLE,
                                                          v_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!v_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(u_array);
        return PyErr_NoMemory();
    }
    f64 *v_data = (f64*)PyArray_DATA(v_array);
    if (n > 0) {
        memset(v_data, 0, (size_t)n * n * sizeof(f64));
    }

    i32 info = 0;

    tb01kx(n, m, p, ndim, a_data, lda, b_data, ldb, c_data, ldc,
           u_data, ldu, v_data, ldv, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(u_array);

    PyObject *result = Py_BuildValue("(OOOOOi)", a_array, b_array, c_array, u_array, v_array, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(u_array);
    Py_DECREF(v_array);
    return result;
}

/* Python wrapper for tb01kd */
PyObject* py_tb01kd(PyObject* self, PyObject* args) {
    const char *dico, *stdom, *joba;
    f64 alpha;
    PyObject *a_obj, *b_obj, *c_obj;

    if (!PyArg_ParseTuple(args, "sssdOOO", &dico, &stdom, &joba, &alpha, &a_obj, &b_obj, &c_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
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

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 m = (i32)PyArray_DIM(b_array, 1);
    i32 p = (i32)PyArray_DIM(c_array, 0);
    i32 lda = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldc = p > 1 ? p : 1;
    i32 ldu = n > 1 ? n : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    npy_intp u_dims[2] = {n, n};
    npy_intp wr_dims[1] = {n > 0 ? n : 1};
    npy_intp u_strides[2] = {sizeof(f64), n * sizeof(f64)};

    PyArrayObject *u_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE,
                                                          u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!u_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }
    f64 *u_data = (f64*)PyArray_DATA(u_array);
    memset(u_data, 0, n * n * sizeof(f64));

    PyArrayObject *wr_array = (PyArrayObject*)PyArray_SimpleNew(1, wr_dims, NPY_DOUBLE);
    if (!wr_array) {
        Py_DECREF(u_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }
    f64 *wr_data = (f64*)PyArray_DATA(wr_array);
    memset(wr_data, 0, (n > 0 ? n : 1) * sizeof(f64));

    PyArrayObject *wi_array = (PyArrayObject*)PyArray_SimpleNew(1, wr_dims, NPY_DOUBLE);
    if (!wi_array) {
        Py_DECREF(wr_array);
        Py_DECREF(u_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }
    f64 *wi_data = (f64*)PyArray_DATA(wi_array);
    memset(wi_data, 0, (n > 0 ? n : 1) * sizeof(f64));

    i32 ldwork = 3 * n > n * m ? (3 * n > n * p ? 3 * n : n * p) : (n * m > n * p ? n * m : n * p);
    if (ldwork < 1) ldwork = 1;
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (!dwork) {
        free(dwork);
        Py_DECREF(wi_array);
        Py_DECREF(wr_array);
        Py_DECREF(u_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    i32 ndim = 0;
    i32 info = 0;

    tb01kd(dico, stdom, joba, n, m, p, alpha, a_data, lda, b_data, ldb, c_data, ldc,
           &ndim, u_data, ldu, wr_data, wi_data, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("(OOOiOOOi)", a_array, b_array, c_array, ndim,
                                      u_array, wr_array, wi_array, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(u_array);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);
    return result;
}

PyObject* py_tb01md(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"jobu", "uplo", "a", "b", "u", NULL};

    const char* jobu = NULL;
    const char* uplo = NULL;
    PyObject* a_obj = NULL;
    PyObject* b_obj = NULL;
    PyObject* u_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssOO|O", kwlist,
                                     &jobu, &uplo, &a_obj, &b_obj, &u_obj)) {
        return NULL;
    }

    if (!jobu || strlen(jobu) != 1 ||
        (jobu[0] != 'N' && jobu[0] != 'n' &&
         jobu[0] != 'I' && jobu[0] != 'i' &&
         jobu[0] != 'U' && jobu[0] != 'u')) {
        PyErr_SetString(PyExc_ValueError, "jobu must be 'N', 'I', or 'U'");
        return NULL;
    }

    if (!uplo || strlen(uplo) != 1 ||
        (uplo[0] != 'U' && uplo[0] != 'u' &&
         uplo[0] != 'L' && uplo[0] != 'l')) {
        PyErr_SetString(PyExc_ValueError, "uplo must be 'U' or 'L'");
        return NULL;
    }

    bool ljoba = (jobu[0] == 'I' || jobu[0] == 'i' ||
                  jobu[0] == 'U' || jobu[0] == 'u');
    bool ljobu = (jobu[0] == 'U' || jobu[0] == 'u');

    PyArrayObject* a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    PyArrayObject* b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 m = (i32)PyArray_DIM(b_array, 1);
    i32 lda = (n > 1) ? n : 1;
    i32 ldb = (n > 1) ? n : 1;
    i32 ldu = ljoba ? ((n > 1) ? n : 1) : 1;

    f64* a_data = (f64*)PyArray_DATA(a_array);
    f64* b_data = (f64*)PyArray_DATA(b_array);

    PyArrayObject* u_array = NULL;
    f64* u_data = NULL;

    if (ljoba) {
        npy_intp u_dims[2] = {n, n};
        npy_intp u_strides[2] = {sizeof(f64), n * sizeof(f64)};

        if (ljobu && u_obj != Py_None) {
            u_array = (PyArrayObject*)PyArray_FROM_OTF(
                u_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            if (!u_array) {
                Py_DECREF(a_array);
                Py_DECREF(b_array);
                return NULL;
            }
        } else {
            u_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE,
                                                  u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            if (!u_array) {
                Py_DECREF(a_array);
                Py_DECREF(b_array);
                return PyErr_NoMemory();
            }
        }
        u_data = (f64*)PyArray_DATA(u_array);
    }

    i32 ldwork = (n > m - 1) ? n : (m - 1);
    if (ldwork < 1) ldwork = 1;
    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (!dwork) {
        if (u_array) Py_DECREF(u_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return PyErr_NoMemory();
    }

    i32 info = 0;
    tb01md(jobu, uplo, n, m, a_data, lda, b_data, ldb, u_data, ldu, dwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    if (ljobu && u_obj != Py_None && u_array) {
        PyArray_ResolveWritebackIfCopy(u_array);
    }

    PyObject* u_result = (u_array) ? (PyObject*)u_array : Py_None;
    if (!u_array) Py_INCREF(Py_None);

    PyObject* result = Py_BuildValue("(OOOi)", a_array, b_array, u_result, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    if (u_array) Py_DECREF(u_array);

    return result;
}

PyObject* py_tb01nd(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"jobu", "uplo", "a", "c", "u", NULL};

    const char* jobu = NULL;
    const char* uplo = NULL;
    PyObject* a_obj = NULL;
    PyObject* c_obj = NULL;
    PyObject* u_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssOO|O", kwlist,
                                     &jobu, &uplo, &a_obj, &c_obj, &u_obj)) {
        return NULL;
    }

    if (!jobu || strlen(jobu) != 1 ||
        (jobu[0] != 'N' && jobu[0] != 'n' &&
         jobu[0] != 'I' && jobu[0] != 'i' &&
         jobu[0] != 'U' && jobu[0] != 'u')) {
        PyErr_SetString(PyExc_ValueError, "jobu must be 'N', 'I', or 'U'");
        return NULL;
    }

    if (!uplo || strlen(uplo) != 1 ||
        (uplo[0] != 'U' && uplo[0] != 'u' &&
         uplo[0] != 'L' && uplo[0] != 'l')) {
        PyErr_SetString(PyExc_ValueError, "uplo must be 'U' or 'L'");
        return NULL;
    }

    bool ljoba = (jobu[0] == 'I' || jobu[0] == 'i' ||
                  jobu[0] == 'U' || jobu[0] == 'u');
    bool ljobu = (jobu[0] == 'U' || jobu[0] == 'u');

    PyArrayObject* a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    PyArrayObject* c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!c_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 p = (i32)PyArray_DIM(c_array, 0);
    i32 lda = (n > 1) ? n : 1;
    i32 ldc = (p > 1) ? p : 1;
    i32 ldu = ljoba ? ((n > 1) ? n : 1) : 1;

    f64* a_data = (f64*)PyArray_DATA(a_array);
    f64* c_data = (f64*)PyArray_DATA(c_array);

    PyArrayObject* u_array = NULL;
    f64* u_data = NULL;

    if (ljoba) {
        npy_intp u_dims[2] = {n, n};
        npy_intp u_strides[2] = {sizeof(f64), n * sizeof(f64)};

        if (ljobu && u_obj != Py_None) {
            u_array = (PyArrayObject*)PyArray_FROM_OTF(
                u_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
            if (!u_array) {
                Py_DECREF(a_array);
                Py_DECREF(c_array);
                return NULL;
            }
        } else {
            u_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE,
                                                  u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            if (!u_array) {
                Py_DECREF(a_array);
                Py_DECREF(c_array);
                return PyErr_NoMemory();
            }
        }
        u_data = (f64*)PyArray_DATA(u_array);
    }

    i32 ldwork = (n > p - 1) ? n : (p - 1);
    if (ldwork < 1) ldwork = 1;
    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (!dwork) {
        if (u_array) Py_DECREF(u_array);
        Py_DECREF(a_array);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    i32 info = 0;
    tb01nd(jobu, uplo, n, p, a_data, lda, c_data, ldc, u_data, ldu, dwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    if (ljobu && u_obj != Py_None && u_array) {
        PyArray_ResolveWritebackIfCopy(u_array);
    }

    PyObject* u_result = (u_array) ? (PyObject*)u_array : Py_None;
    if (!u_array) Py_INCREF(Py_None);

    PyObject* result = Py_BuildValue("(OOOi)", a_array, c_array, u_result, info);

    Py_DECREF(a_array);
    Py_DECREF(c_array);
    if (u_array) Py_DECREF(u_array);

    return result;
}

PyObject* py_tb01px(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"job", "equil", "n", "m", "p", "a", "b", "c", "tol", NULL};

    const char* job = NULL;
    const char* equil = NULL;
    i32 n, m, p;
    PyObject* a_obj = NULL;
    PyObject* b_obj = NULL;
    PyObject* c_obj = NULL;
    f64 tol = 0.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssiiiOOO|d", kwlist,
                                     &job, &equil, &n, &m, &p,
                                     &a_obj, &b_obj, &c_obj, &tol)) {
        return NULL;
    }

    if (!job || strlen(job) != 1 ||
        (job[0] != 'M' && job[0] != 'm' &&
         job[0] != 'C' && job[0] != 'c' &&
         job[0] != 'O' && job[0] != 'o')) {
        PyErr_SetString(PyExc_ValueError, "job must be 'M', 'C', or 'O'");
        return NULL;
    }

    if (!equil || strlen(equil) != 1 ||
        (equil[0] != 'S' && equil[0] != 's' &&
         equil[0] != 'N' && equil[0] != 'n')) {
        PyErr_SetString(PyExc_ValueError, "equil must be 'S' or 'N'");
        return NULL;
    }

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be >= 0");
        return NULL;
    }
    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be >= 0");
        return NULL;
    }
    if (p < 0) {
        PyErr_SetString(PyExc_ValueError, "p must be >= 0");
        return NULL;
    }

    PyArrayObject* a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    PyArrayObject* b_input = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!b_input) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject* c_input = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!c_input) {
        Py_DECREF(a_array);
        Py_DECREF(b_input);
        return NULL;
    }

    i32 maxmp = (m > p) ? m : p;
    i32 lda = (n > 1) ? n : 1;
    i32 ldb = (n > 1) ? n : 1;
    i32 ldc = (n > 0) ? ((maxmp > 1) ? maxmp : 1) : 1;

    bool need_padding = (job[0] != 'C' && job[0] != 'c') && (m != p);
    f64* b_data = NULL;
    f64* c_data = NULL;
    PyArrayObject* b_array = NULL;
    PyArrayObject* c_array = NULL;

    if (need_padding) {
        npy_intp b_dims[2] = {n > 0 ? n : 1, maxmp > 0 ? maxmp : 1};
        npy_intp b_strides[2] = {sizeof(f64), ldb * sizeof(f64)};
        b_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE,
                                              b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!b_array) {
            Py_DECREF(a_array);
            Py_DECREF(b_input);
            Py_DECREF(c_input);
            return PyErr_NoMemory();
        }
        b_data = (f64*)PyArray_DATA(b_array);
        memset(b_data, 0, n * maxmp * sizeof(f64));
        f64* b_input_data = (f64*)PyArray_DATA(b_input);
        for (i32 j = 0; j < m; j++) {
            for (i32 i = 0; i < n; i++) {
                b_data[i + j * ldb] = b_input_data[i + j * n];
            }
        }

        npy_intp c_dims[2] = {maxmp > 0 ? maxmp : 1, n > 0 ? n : 1};
        npy_intp c_strides[2] = {sizeof(f64), ldc * sizeof(f64)};
        c_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE,
                                              c_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!c_array) {
            Py_DECREF(a_array);
            Py_DECREF(b_input);
            Py_DECREF(c_input);
            Py_DECREF(b_array);
            return PyErr_NoMemory();
        }
        c_data = (f64*)PyArray_DATA(c_array);
        memset(c_data, 0, maxmp * n * sizeof(f64));
        f64* c_input_data = (f64*)PyArray_DATA(c_input);
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < p; i++) {
                c_data[i + j * ldc] = c_input_data[i + j * p];
            }
        }
    } else {
        b_array = (PyArrayObject*)PyArray_FROM_OTF(
            b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!b_array) {
            Py_DECREF(a_array);
            Py_DECREF(b_input);
            Py_DECREF(c_input);
            return NULL;
        }
        c_array = (PyArrayObject*)PyArray_FROM_OTF(
            c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!c_array) {
            Py_DECREF(a_array);
            Py_DECREF(b_input);
            Py_DECREF(c_input);
            Py_DECREF(b_array);
            return NULL;
        }
        b_data = (f64*)PyArray_DATA(b_array);
        c_data = (f64*)PyArray_DATA(c_array);
    }

    Py_DECREF(b_input);
    Py_DECREF(c_input);

    f64* a_data = (f64*)PyArray_DATA(a_array);

    i32 c_coeff = (job[0] == 'M' || job[0] == 'm') ? 2 : 1;
    i32 liwork = c_coeff * n + maxmp;
    if (liwork < 1) liwork = 1;
    i32* iwork = (i32*)PyMem_Calloc(liwork, sizeof(i32));
    if (!iwork) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    i32 max_n_3maxmp = (n > 3 * maxmp) ? n : 3 * maxmp;
    i32 ldwmin = n + max_n_3maxmp;
    if (ldwmin < 1) ldwmin = 1;
    i32 ldwork = ldwmin + n * (n + m + p);
    f64* dwork = (f64*)PyMem_Calloc(ldwork, sizeof(f64));
    if (!dwork) {
        PyMem_Free(iwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    i32 nr = 0;
    i32 infred[4] = {0, 0, 0, 0};
    i32 info = 0;

    tb01px(job, equil, n, m, p, a_data, lda, b_data, ldb, c_data, ldc,
           &nr, infred, tol, iwork, dwork, ldwork, &info);

    PyMem_Free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    if (!need_padding) {
        PyArray_ResolveWritebackIfCopy(b_array);
        PyArray_ResolveWritebackIfCopy(c_array);
    }

    npy_intp infred_dims[1] = {4};
    PyArrayObject* infred_array = (PyArrayObject*)PyArray_SimpleNew(1, infred_dims, NPY_INT32);
    if (!infred_array) {
        PyMem_Free(iwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }
    i32* infred_data = (i32*)PyArray_DATA(infred_array);
    for (int i = 0; i < 4; i++) {
        infred_data[i] = infred[i];
    }

    i32 nblocks = infred[3];
    if (nblocks < 0) nblocks = 0;
    npy_intp iwork_dims[1] = {nblocks > 0 ? nblocks : 1};
    PyArrayObject* iwork_array = (PyArrayObject*)PyArray_SimpleNew(1, iwork_dims, NPY_INT32);
    if (!iwork_array) {
        PyMem_Free(iwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(infred_array);
        return PyErr_NoMemory();
    }
    i32* iwork_data = (i32*)PyArray_DATA(iwork_array);
    for (int i = 0; i < (nblocks > 0 ? nblocks : 1); i++) {
        iwork_data[i] = (i < liwork) ? iwork[i] : 0;
    }

    PyMem_Free(iwork);

    PyObject* result = Py_BuildValue("(OOOiOOi)",
        a_array, b_array, c_array, nr, infred_array, iwork_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(infred_array);
    Py_DECREF(iwork_array);

    return result;
}

/* Python wrapper for tb01ty */
PyObject* py_tb01ty(PyObject* self, PyObject* args, PyObject* kwargs) {
    i32 mode, ioff, joff, nrow, ncol;
    f64 size;
    PyObject *x_obj;
    PyArrayObject *x_array;

    static char *kwlist[] = {"mode", "ioff", "joff", "nrow", "ncol", "size", "x", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiiiidO", kwlist,
                                     &mode, &ioff, &joff, &nrow, &ncol, &size, &x_obj)) {
        return NULL;
    }

    if (nrow < 0) {
        PyErr_Format(PyExc_ValueError, "nrow must be non-negative, got %d", nrow);
        return NULL;
    }
    if (ncol < 0) {
        PyErr_Format(PyExc_ValueError, "ncol must be non-negative, got %d", ncol);
        return NULL;
    }
    if (ioff < 0) {
        PyErr_Format(PyExc_ValueError, "ioff must be non-negative, got %d", ioff);
        return NULL;
    }
    if (joff < 0) {
        PyErr_Format(PyExc_ValueError, "joff must be non-negative, got %d", joff);
        return NULL;
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (x_array == NULL) return NULL;

    i32 ldx = (i32)PyArray_DIM(x_array, 0);
    i32 n_cols = (i32)PyArray_DIM(x_array, 1);

    if (ioff + nrow > ldx) {
        PyErr_Format(PyExc_ValueError, "ioff + nrow (%d) exceeds matrix rows (%d)",
                     ioff + nrow, ldx);
        Py_DECREF(x_array);
        return NULL;
    }
    if (joff + ncol > n_cols) {
        PyErr_Format(PyExc_ValueError, "joff + ncol (%d) exceeds matrix columns (%d)",
                     joff + ncol, n_cols);
        Py_DECREF(x_array);
        return NULL;
    }

    f64 *x = (f64*)PyArray_DATA(x_array);

    i32 bvect_len = (mode != 0) ? (joff + ncol) : (ioff + nrow);
    if (bvect_len < 1) bvect_len = 1;

    npy_intp bvect_dims[1] = {bvect_len};
    npy_intp bvect_strides[1] = {sizeof(f64)};
    PyArrayObject *bvect_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 1, bvect_dims, NPY_DOUBLE, bvect_strides,
        NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (bvect_array == NULL) {
        Py_DECREF(x_array);
        return PyErr_NoMemory();
    }
    f64 *bvect = (f64*)PyArray_DATA(bvect_array);

    for (i32 i = 0; i < bvect_len; i++) {
        bvect[i] = 1.0;
    }

    tb01ty(mode, ioff, joff, nrow, ncol, size, x, ldx, bvect);

    PyArray_ResolveWritebackIfCopy(x_array);

    i32 info = 0;
    PyObject *result = Py_BuildValue("(OOi)", x_array, bvect_array, info);
    Py_DECREF(x_array);
    Py_DECREF(bvect_array);
    return result;
}

PyObject* py_tb01ux(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *compz_str;
    i32 n, m, p;
    PyObject *a_obj, *b_obj, *c_obj;
    f64 tol = 0.0;

    static char *kwlist[] = {"compz", "n", "m", "p", "a", "b", "c", "tol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siiiOOO|d", kwlist,
                                     &compz_str, &n, &m, &p, &a_obj, &b_obj, &c_obj, &tol)) {
        return NULL;
    }

    char compz = (char)toupper((unsigned char)compz_str[0]);
    if (compz != 'N' && compz != 'I') {
        PyErr_SetString(PyExc_ValueError, "compz must be 'N' or 'I'");
        return NULL;
    }

    if (n < 0) {
        PyErr_Format(PyExc_ValueError, "n must be non-negative (got %d)", n);
        return NULL;
    }
    if (m < 0) {
        PyErr_Format(PyExc_ValueError, "m must be non-negative (got %d)", m);
        return NULL;
    }
    if (p < 0) {
        PyErr_Format(PyExc_ValueError, "p must be non-negative (got %d)", p);
        return NULL;
    }
    if (tol >= 1.0) {
        char msg[64];
        snprintf(msg, sizeof(msg), "tol must be < 1 (got %g)", tol);
        PyErr_SetString(PyExc_ValueError, msg);
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    PyArrayObject *b_input = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!b_input) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_input = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!c_input) {
        Py_DECREF(a_array);
        Py_DECREF(b_input);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 mp = m > p ? m : p;
    i32 ldc = (n > 0 && mp > 0) ? mp : 1;
    i32 ldz = (compz == 'I') ? (n > 0 ? n : 1) : 1;

    i32 n_alloc = n > 0 ? n : 1;
    i32 p_alloc = p > 0 ? p : 1;

    bool need_padding = (m != p);
    f64 *b_data = NULL;
    f64 *c_data = NULL;
    PyArrayObject *b_array = NULL;
    PyArrayObject *c_array = NULL;

    if (need_padding) {
        npy_intp b_dims[2] = {n > 0 ? n : 1, mp > 0 ? mp : 1};
        npy_intp b_strides[2] = {sizeof(f64), ldb * sizeof(f64)};
        b_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE,
                                              b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!b_array) {
            Py_DECREF(a_array);
            Py_DECREF(b_input);
            Py_DECREF(c_input);
            return PyErr_NoMemory();
        }
        b_data = (f64*)PyArray_DATA(b_array);
        memset(b_data, 0, n_alloc * mp * sizeof(f64));
        f64 *b_input_data = (f64*)PyArray_DATA(b_input);
        for (i32 j = 0; j < m; j++) {
            for (i32 i = 0; i < n; i++) {
                b_data[i + j * ldb] = b_input_data[i + j * n];
            }
        }

        npy_intp c_dims[2] = {mp > 0 ? mp : 1, n > 0 ? n : 1};
        npy_intp c_strides[2] = {sizeof(f64), ldc * sizeof(f64)};
        c_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE,
                                              c_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!c_array) {
            Py_DECREF(a_array);
            Py_DECREF(b_input);
            Py_DECREF(c_input);
            Py_DECREF(b_array);
            return PyErr_NoMemory();
        }
        c_data = (f64*)PyArray_DATA(c_array);
        memset(c_data, 0, mp * n_alloc * sizeof(f64));
        f64 *c_input_data = (f64*)PyArray_DATA(c_input);
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < p; i++) {
                c_data[i + j * ldc] = c_input_data[i + j * p];
            }
        }
    } else {
        b_array = (PyArrayObject*)PyArray_FROM_OTF(
            b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!b_array) {
            Py_DECREF(a_array);
            Py_DECREF(b_input);
            Py_DECREF(c_input);
            return NULL;
        }
        c_array = (PyArrayObject*)PyArray_FROM_OTF(
            c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!c_array) {
            Py_DECREF(a_array);
            Py_DECREF(b_input);
            Py_DECREF(c_input);
            Py_DECREF(b_array);
            return NULL;
        }
        b_data = (f64*)PyArray_DATA(b_array);
        c_data = (f64*)PyArray_DATA(c_array);
    }

    Py_DECREF(b_input);
    Py_DECREF(c_input);

    npy_intp z_dims[2] = {n_alloc, n_alloc};
    npy_intp z_strides[2] = {sizeof(f64), ldz * sizeof(f64)};
    PyObject *z_array = PyArray_New(&PyArray_Type, 2, z_dims, NPY_DOUBLE,
                                    z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (z_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate z_array");
        return NULL;
    }
    f64 *z_data = (f64*)PyArray_DATA((PyArrayObject*)z_array);
    memset(z_data, 0, ldz * n_alloc * sizeof(f64));

    npy_intp ctau_dims[1] = {n_alloc};
    PyObject *ctau_array = PyArray_SimpleNew(1, ctau_dims, NPY_INT32);
    if (ctau_array == NULL) {
        Py_XDECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }
    i32 *ctau = (i32*)PyArray_DATA((PyArrayObject*)ctau_array);
    memset(ctau, 0, n_alloc * sizeof(i32));

    i32 *iwork = (i32*)PyMem_Calloc(p_alloc, sizeof(i32));

    i32 ldwork = 1;
    if (n > ldwork) ldwork = n;
    if (3 * p > ldwork) ldwork = 3 * p;
    if (m > ldwork) ldwork = m;
    if (ldwork < 1) ldwork = 1;
    i32 total_dwork = n + ldwork;

    f64 *dwork = (f64*)PyMem_Malloc(total_dwork * sizeof(f64));

    if (!iwork || !dwork) {
        PyMem_Free(iwork);
        PyMem_Free(dwork);
        Py_DECREF(ctau_array);
        Py_XDECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    i32 nobsv, nlblck, info;
    f64 *a_data = (f64*)PyArray_DATA(a_array);

    tb01ux(&compz, n, m, p, a_data, lda, b_data, ldb, c_data, ldc,
           z_data, ldz, &nobsv, &nlblck, ctau, tol, iwork, dwork, &info);

    PyMem_Free(dwork);
    PyMem_Free(iwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    if (!need_padding) {
        PyArray_ResolveWritebackIfCopy(b_array);
        PyArray_ResolveWritebackIfCopy(c_array);
    }

    PyObject *result = Py_BuildValue("OOOiiOOi",
                                     a_array, b_array, c_array,
                                     nobsv, nlblck, ctau_array,
                                     z_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(z_array);
    Py_DECREF(ctau_array);

    return result;
}


/* Python wrapper for tb01uy */
PyObject* py_tb01uy(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *jobz_str;
    i32 n, m1, m2, p;
    PyObject *a_obj, *b_obj, *c_obj;
    f64 tol = 0.0;

    static char *kwlist[] = {"jobz", "n", "m1", "m2", "p", "a", "b", "c", "tol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siiiiOOO|d", kwlist,
                                     &jobz_str, &n, &m1, &m2, &p, &a_obj, &b_obj, &c_obj, &tol)) {
        return NULL;
    }

    char jobz = (char)toupper((unsigned char)jobz_str[0]);
    if (jobz != 'N' && jobz != 'F' && jobz != 'I') {
        PyErr_SetString(PyExc_ValueError, "jobz must be 'N', 'F', or 'I'");
        return NULL;
    }

    if (n < 0) {
        PyErr_Format(PyExc_ValueError, "n must be non-negative (got %d)", n);
        return NULL;
    }
    if (m1 < 0) {
        PyErr_Format(PyExc_ValueError, "m1 must be non-negative (got %d)", m1);
        return NULL;
    }
    if (m2 < 0) {
        PyErr_Format(PyExc_ValueError, "m2 must be non-negative (got %d)", m2);
        return NULL;
    }
    if (p < 0) {
        PyErr_Format(PyExc_ValueError, "p must be non-negative (got %d)", p);
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!c_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldz = (jobz == 'I' || jobz == 'F') ? (n > 0 ? n : 1) : 1;

    i32 n_alloc = n > 0 ? n : 1;
    i32 max_m1_m2 = m1 > m2 ? m1 : m2;
    if (max_m1_m2 < 1) max_m1_m2 = 1;

    npy_intp z_dims[2] = {n_alloc, n_alloc};
    npy_intp z_strides[2] = {sizeof(f64), ldz * sizeof(f64)};
    PyObject *z_array = PyArray_New(&PyArray_Type, 2, z_dims, NPY_DOUBLE,
                                    z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (z_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate z_array");
        return NULL;
    }
    f64 *z_data = (f64*)PyArray_DATA((PyArrayObject*)z_array);
    memset(z_data, 0, ldz * n_alloc * sizeof(f64));

    i32 tau_len = n_alloc;
    if (tau_len < 1) tau_len = 1;
    npy_intp tau_dims[1] = {tau_len};
    PyObject *tau_array = PyArray_SimpleNew(1, tau_dims, NPY_DOUBLE);
    if (tau_array == NULL) {
        Py_XDECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }
    f64 *tau = (f64*)PyArray_DATA((PyArrayObject*)tau_array);
    memset(tau, 0, tau_len * sizeof(f64));

    npy_intp nblk_dims[1] = {2 * n_alloc};
    PyObject *nblk_array = PyArray_SimpleNew(1, nblk_dims, NPY_INT32);
    if (nblk_array == NULL) {
        Py_DECREF(tau_array);
        Py_XDECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }
    i32 *nblk = (i32*)PyArray_DATA((PyArrayObject*)nblk_array);
    memset(nblk, 0, 2 * n_alloc * sizeof(i32));

    i32 *iwork = (i32*)calloc(max_m1_m2, sizeof(i32));

    i32 ldwork = 1;
    if (n > ldwork) ldwork = n;
    if (3 * max_m1_m2 > ldwork) ldwork = 3 * max_m1_m2;
    if (p > ldwork) ldwork = p;
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        Py_DECREF(nblk_array);
        Py_DECREF(tau_array);
        Py_XDECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    i32 ncont, indcon, info;
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    tb01uy(&jobz, n, m1, m2, p, a_data, lda, b_data, ldb, c_data, ldc,
           &ncont, &indcon, nblk, z_data, ldz, tau, tol,
           iwork, dwork, ldwork, &info);

    free(dwork);
    free(iwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("OOOiiOOOi",
                                     a_array, b_array, c_array,
                                     ncont, indcon, nblk_array,
                                     z_array, tau_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(z_array);
    Py_DECREF(tau_array);
    Py_DECREF(nblk_array);

    return result;
}


/* Python wrapper for tb01wx */
PyObject* py_tb01wx(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *compu = "I";
    i32 n, m, p;
    i32 ldwork_in = 0;
    PyObject *a_obj, *b_obj, *c_obj;
    PyObject *u_obj = Py_None;
    PyArrayObject *a_array, *b_array, *c_array;
    PyArrayObject *u_input_array = NULL;

    static char *kwlist[] = {"compu", "n", "m", "p", "a", "b", "c", "u", "ldwork", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siiiOOO|Oi", kwlist,
                                     &compu, &n, &m, &p, &a_obj, &b_obj, &c_obj,
                                     &u_obj, &ldwork_in)) {
        return NULL;
    }

    char compu_char = toupper(compu[0]);
    if (compu_char != 'N' && compu_char != 'I' && compu_char != 'U') {
        i32 info = -1;
        return Py_BuildValue("OOOOOi", Py_None, Py_None, Py_None, Py_None, Py_None, info);
    }
    if (n < 0) {
        i32 info = -2;
        return Py_BuildValue("OOOOOi", Py_None, Py_None, Py_None, Py_None, Py_None, info);
    }
    if (m < 0) {
        i32 info = -3;
        return Py_BuildValue("OOOOOi", Py_None, Py_None, Py_None, Py_None, Py_None, info);
    }
    if (p < 0) {
        i32 info = -4;
        return Py_BuildValue("OOOOOi", Py_None, Py_None, Py_None, Py_None, Py_None, info);
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

    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldu = n > 0 ? n : 1;

    bool compute_u = (compu_char != 'N');

    PyObject *u_array = NULL;
    f64 *u_data = NULL;

    if (compu_char == 'U' && u_obj != Py_None) {
        u_input_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (u_input_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            return NULL;
        }
        u_array = (PyObject*)u_input_array;
        u_data = (f64*)PyArray_DATA(u_input_array);
    } else if (compute_u) {
        npy_intp u_dims[2] = {n, n};
        npy_intp u_strides[2] = {sizeof(f64), ldu * sizeof(f64)};
        u_array = PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE,
                              u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (u_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            return NULL;
        }
        u_data = (f64*)PyArray_DATA((PyArrayObject*)u_array);
        memset(u_data, 0, ldu * n * sizeof(f64));
    } else {
        npy_intp u_dims[2] = {1, 1};
        u_array = PyArray_SimpleNew(2, u_dims, NPY_DOUBLE);
        if (u_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            return NULL;
        }
        u_data = (f64*)PyArray_DATA((PyArrayObject*)u_array);
        u_data[0] = 0.0;
        ldu = 1;
    }

    i32 max_nmp = n;
    if (m > max_nmp) max_nmp = m;
    if (p > max_nmp) max_nmp = p;

    i32 minwrk = n > 0 ? n - 1 + max_nmp : 1;
    i32 ldwork = ldwork_in;

    npy_intp dwork_dims[1] = {minwrk > 1 ? minwrk : 1};
    PyObject *dwork_array = PyArray_SimpleNew(1, dwork_dims, NPY_DOUBLE);
    if (dwork_array == NULL) {
        Py_DECREF(u_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }
    f64 *dwork = (f64*)PyArray_DATA((PyArrayObject*)dwork_array);
    memset(dwork, 0, dwork_dims[0] * sizeof(f64));

    if (ldwork < 0) {
        ldwork = -1;
    } else if (ldwork < minwrk) {
        ldwork = minwrk;
    }

    f64 *dwork_call = NULL;
    if (ldwork == -1) {
        dwork_call = dwork;
    } else {
        dwork_call = (f64*)malloc(ldwork * sizeof(f64));
        if (dwork_call == NULL) {
            Py_DECREF(dwork_array);
            Py_DECREF(u_array);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
            return NULL;
        }
    }

    i32 info;
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    tb01wx(&compu_char, n, m, p, a_data, lda, b_data, ldb, c_data, ldc,
           u_data, ldu, dwork_call, ldwork, &info);

    dwork[0] = dwork_call[0];

    if (ldwork != -1) {
        free(dwork_call);
    }

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    if (u_input_array != NULL) {
        PyArray_ResolveWritebackIfCopy(u_input_array);
    }

    PyObject *result = Py_BuildValue("OOOOOi", a_array, b_array, c_array,
                                     u_array, dwork_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(u_array);
    Py_DECREF(dwork_array);

    return result;
}


/* Python wrapper for tb01xz (complex version of tb01xd) */
PyObject* py_tb01xz(PyObject* self, PyObject* args) {
    const char *jobd_str;
    i32 n, m, p, kl, ku;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "siiiiiOOOO", &jobd_str, &n, &m, &p, &kl, &ku,
                          &a_obj, &b_obj, &c_obj, &d_obj)) {
        return NULL;
    }

    char jobd = (char)toupper((unsigned char)jobd_str[0]);
    if (jobd != 'D' && jobd != 'Z') {
        PyErr_SetString(PyExc_ValueError, "jobd must be 'D' or 'Z'");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!c_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!d_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);
    npy_intp *d_dims = PyArray_DIMS(d_array);

    i32 lda = PyArray_NDIM(a_array) >= 1 ? (i32)a_dims[0] : 1;
    i32 ldb = PyArray_NDIM(b_array) >= 1 ? (i32)b_dims[0] : 1;
    i32 ldc = PyArray_NDIM(c_array) >= 1 ? (i32)c_dims[0] : 1;
    i32 ldd = PyArray_NDIM(d_array) >= 1 ? (i32)d_dims[0] : 1;

    c128 *a_data = (c128*)PyArray_DATA(a_array);
    c128 *b_data = (c128*)PyArray_DATA(b_array);
    c128 *c_data = (c128*)PyArray_DATA(c_array);
    c128 *d_data = (c128*)PyArray_DATA(d_array);

    i32 info;
    tb01xz(&jobd, n, m, p, kl, ku, a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    PyObject *result = Py_BuildValue("OOOOi", a_array, b_array, c_array, d_array, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    return result;
}


/* Python wrapper for tb04bv */
PyObject* py_tb04bv(PyObject* self, PyObject* args) {
    const char *order_str;
    i32 p, m, md;
    PyObject *ign_obj, *igd_obj, *gn_obj, *gd_obj;
    f64 tol = 0.0;

    if (!PyArg_ParseTuple(args, "siiiOOOO|d", &order_str, &p, &m, &md,
                          &ign_obj, &igd_obj, &gn_obj, &gd_obj, &tol)) {
        return NULL;
    }

    char order = (char)toupper((unsigned char)order_str[0]);
    if (order != 'I' && order != 'D') {
        PyErr_SetString(PyExc_ValueError, "order must be 'I' or 'D'");
        return NULL;
    }

    PyArrayObject *ign_array = (PyArrayObject*)PyArray_FROM_OTF(
        ign_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!ign_array) return NULL;

    PyArrayObject *igd_array = (PyArrayObject*)PyArray_FROM_OTF(
        igd_obj, NPY_INT32, NPY_ARRAY_FARRAY);
    if (!igd_array) {
        Py_DECREF(ign_array);
        return NULL;
    }

    PyArrayObject *gn_array = (PyArrayObject*)PyArray_FROM_OTF(
        gn_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!gn_array) {
        Py_DECREF(ign_array);
        Py_DECREF(igd_array);
        return NULL;
    }

    PyArrayObject *gd_array = (PyArrayObject*)PyArray_FROM_OTF(
        gd_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!gd_array) {
        Py_DECREF(ign_array);
        Py_DECREF(igd_array);
        Py_DECREF(gn_array);
        return NULL;
    }

    i32 ldign = p > 0 ? p : 1;
    i32 ldigd = p > 0 ? p : 1;
    i32 ldd = p > 0 ? p : 1;

    i32 *ign_data = (i32*)PyArray_DATA(ign_array);
    const i32 *igd_data = (const i32*)PyArray_DATA(igd_array);
    f64 *gn_data = (f64*)PyArray_DATA(gn_array);
    const f64 *gd_data = (const f64*)PyArray_DATA(gd_array);

    npy_intp d_dims[2] = {ldd, m > 0 ? m : 1};
    npy_intp d_strides[2] = {sizeof(f64), ldd * sizeof(f64)};
    PyArrayObject *d_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, d_dims, NPY_DOUBLE, d_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!d_array) {
        Py_DECREF(ign_array);
        Py_DECREF(igd_array);
        Py_DECREF(gn_array);
        Py_DECREF(gd_array);
        return NULL;
    }
    f64 *d_data = (f64*)PyArray_DATA(d_array);

    i32 info;
    tb04bv(&order, p, m, md, ign_data, ldign, igd_data, ldigd,
           gn_data, gd_data, d_data, ldd, tol, &info);

    PyArray_ResolveWritebackIfCopy(ign_array);
    PyArray_ResolveWritebackIfCopy(gn_array);

    PyObject *result = Py_BuildValue("OOOi", ign_array, gn_array, d_array, info);
    Py_DECREF(ign_array);
    Py_DECREF(igd_array);
    Py_DECREF(gn_array);
    Py_DECREF(gd_array);
    Py_DECREF(d_array);

    return result;
}

/* TB01TD: Balance state-space (A,B,C,D) */
PyObject* py_tb01td(PyObject* self, PyObject* args)
{
    i32 n, m, p;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "iiiOOOO", &n, &m, &p,
                          &a_obj, &b_obj, &c_obj, &d_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!c_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!d_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldd = p > 0 ? p : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);

    /* Allocate output arrays */
    npy_intp scstat_dim = n > 0 ? n : 1;
    npy_intp scin_dim = m > 0 ? m : 1;
    npy_intp scout_dim = p > 0 ? p : 1;

    PyArrayObject *scstat_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 1, &scstat_dim, NPY_DOUBLE, NULL, NULL, 0, 0, NULL);
    if (!scstat_array) {
        Py_DECREF(a_array); Py_DECREF(b_array);
        Py_DECREF(c_array); Py_DECREF(d_array);
        return NULL;
    }

    PyArrayObject *scin_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 1, &scin_dim, NPY_DOUBLE, NULL, NULL, 0, 0, NULL);
    if (!scin_array) {
        Py_DECREF(a_array); Py_DECREF(b_array);
        Py_DECREF(c_array); Py_DECREF(d_array);
        Py_DECREF(scstat_array);
        return NULL;
    }

    PyArrayObject *scout_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 1, &scout_dim, NPY_DOUBLE, NULL, NULL, 0, 0, NULL);
    if (!scout_array) {
        Py_DECREF(a_array); Py_DECREF(b_array);
        Py_DECREF(c_array); Py_DECREF(d_array);
        Py_DECREF(scstat_array); Py_DECREF(scin_array);
        return NULL;
    }

    f64 *scstat_data = (f64*)PyArray_DATA(scstat_array);
    f64 *scin_data = (f64*)PyArray_DATA(scin_array);
    f64 *scout_data = (f64*)PyArray_DATA(scout_array);

    /* Allocate workspace */
    npy_intp dwork_dim = n > 0 ? n : 1;
    f64 *dwork = (f64*)calloc(dwork_dim, sizeof(f64));
    if (!dwork) {
        Py_DECREF(a_array); Py_DECREF(b_array);
        Py_DECREF(c_array); Py_DECREF(d_array);
        Py_DECREF(scstat_array); Py_DECREF(scin_array);
        Py_DECREF(scout_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 low, igh, info;

    tb01td(n, m, p, a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           &low, &igh, scstat_data, scin_data, scout_data, dwork, &info);

    free(dwork);

    /* Writeback modified arrays */
    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    PyObject *result = Py_BuildValue("iiOOOi",
                                      low, igh,
                                      scstat_array, scin_array, scout_array,
                                      info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(scstat_array);
    Py_DECREF(scin_array);
    Py_DECREF(scout_array);

    return result;
}


/* Python wrapper for tb04bw */
PyObject* py_tb04bw(PyObject* self, PyObject* args) {
    const char *order_str;
    i32 p, m, md;
    PyObject *ign_obj, *igd_obj, *gn_obj, *gd_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "siiiOOOOO", &order_str, &p, &m, &md,
                          &ign_obj, &igd_obj, &gn_obj, &gd_obj, &d_obj)) {
        return NULL;
    }

    char order = (char)toupper((unsigned char)order_str[0]);
    if (order != 'I' && order != 'D') {
        PyErr_SetString(PyExc_ValueError, "order must be 'I' or 'D'");
        return NULL;
    }

    PyArrayObject *ign_array = (PyArrayObject*)PyArray_FROM_OTF(
        ign_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!ign_array) return NULL;

    PyArrayObject *igd_array = (PyArrayObject*)PyArray_FROM_OTF(
        igd_obj, NPY_INT32, NPY_ARRAY_FARRAY);
    if (!igd_array) {
        Py_DECREF(ign_array);
        return NULL;
    }

    PyArrayObject *gn_array = (PyArrayObject*)PyArray_FROM_OTF(
        gn_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!gn_array) {
        Py_DECREF(ign_array);
        Py_DECREF(igd_array);
        return NULL;
    }

    PyArrayObject *gd_array = (PyArrayObject*)PyArray_FROM_OTF(
        gd_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!gd_array) {
        Py_DECREF(ign_array);
        Py_DECREF(igd_array);
        Py_DECREF(gn_array);
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!d_array) {
        Py_DECREF(ign_array);
        Py_DECREF(igd_array);
        Py_DECREF(gn_array);
        Py_DECREF(gd_array);
        return NULL;
    }

    i32 ldign = p > 0 ? p : 1;
    i32 ldigd = p > 0 ? p : 1;
    i32 ldd = p > 0 ? p : 1;

    i32 *ign_data = (i32*)PyArray_DATA(ign_array);
    const i32 *igd_data = (const i32*)PyArray_DATA(igd_array);
    f64 *gn_data = (f64*)PyArray_DATA(gn_array);
    const f64 *gd_data = (const f64*)PyArray_DATA(gd_array);
    const f64 *d_data = (const f64*)PyArray_DATA(d_array);

    i32 info;
    tb04bw(&order, p, m, md, ign_data, ldign, igd_data, ldigd,
           gn_data, gd_data, d_data, ldd, &info);

    PyArray_ResolveWritebackIfCopy(ign_array);
    PyArray_ResolveWritebackIfCopy(gn_array);

    PyObject *result = Py_BuildValue("OOi", ign_array, gn_array, info);
    Py_DECREF(ign_array);
    Py_DECREF(igd_array);
    Py_DECREF(gn_array);
    Py_DECREF(gd_array);
    Py_DECREF(d_array);

    return result;
}

PyObject* py_tb04bx(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static char* kwlist[] = {"ip", "iz", "a", "b", "c", "d", "pr", "pi", "zr", "zi", NULL};

    i32 ip, iz;
    PyObject *a_obj, *b_obj, *c_obj, *pr_obj, *pi_obj, *zr_obj, *zi_obj;
    f64 d;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiOOOdOOOO:tb04bx", kwlist,
                                     &ip, &iz, &a_obj, &b_obj, &c_obj, &d,
                                     &pr_obj, &pi_obj, &zr_obj, &zi_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!c_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *pr_array = (PyArrayObject*)PyArray_FROM_OTF(
        pr_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!pr_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    PyArrayObject *pi_array = (PyArrayObject*)PyArray_FROM_OTF(
        pi_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!pi_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(pr_array);
        return NULL;
    }

    PyArrayObject *zr_array = (PyArrayObject*)PyArray_FROM_OTF(
        zr_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!zr_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(pr_array);
        Py_DECREF(pi_array);
        return NULL;
    }

    PyArrayObject *zi_array = (PyArrayObject*)PyArray_FROM_OTF(
        zi_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!zi_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(pr_array);
        Py_DECREF(pi_array);
        Py_DECREF(zr_array);
        return NULL;
    }

    i32 lda = ip > 0 ? ip : 1;
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    const f64 *c_data = (const f64*)PyArray_DATA(c_array);
    const f64 *pr_data = (const f64*)PyArray_DATA(pr_array);
    const f64 *pi_data = (const f64*)PyArray_DATA(pi_array);
    const f64 *zr_data = (const f64*)PyArray_DATA(zr_array);
    const f64 *zi_data = (const f64*)PyArray_DATA(zi_array);

    i32 *iwork = NULL;
    if (ip > 0) {
        iwork = (i32*)calloc(ip, sizeof(i32));
        if (!iwork) {
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            Py_DECREF(pr_array);
            Py_DECREF(pi_array);
            Py_DECREF(zr_array);
            Py_DECREF(zi_array);
            return PyErr_NoMemory();
        }
    }

    f64 gain;
    tb04bx(ip, iz, a_data, lda, b_data, c_data, d,
           pr_data, pi_data, zr_data, zi_data, &gain, iwork);

    free(iwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(pr_array);
    Py_DECREF(pi_array);
    Py_DECREF(zr_array);
    Py_DECREF(zi_array);

    return PyFloat_FromDouble(gain);
}

PyObject* py_tb04bd(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"jobd", "order", "equil", "n", "m", "p", "md",
                             "a", "b", "c", "d", "tol", NULL};

    const char *jobd, *order, *equil;
    i32 n, m, p, md;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    f64 tol = 0.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssiiiiOOOO|d:tb04bd", kwlist,
                                     &jobd, &order, &equil, &n, &m, &p, &md,
                                     &a_obj, &b_obj, &c_obj, &d_obj, &tol)) {
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldd = p > 0 ? p : 1;
    i32 ldign = p > 0 ? p : 1;
    i32 ldigd = p > 0 ? p : 1;

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *c_array = (PyArrayObject *)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *d_array = (PyArrayObject *)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);

    if (!a_array || !b_array || !c_array || !d_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        return NULL;
    }

    f64 *a_data = (f64 *)PyArray_DATA(a_array);
    f64 *b_data = (f64 *)PyArray_DATA(b_array);
    f64 *c_data = (f64 *)PyArray_DATA(c_array);
    const f64 *d_data = (const f64 *)PyArray_DATA(d_array);

    npy_intp ign_dims[2] = {p, m};
    npy_intp ign_strides[2] = {sizeof(i32), ldign * sizeof(i32)};
    PyArrayObject *ign_array = (PyArrayObject *)PyArray_New(
        &PyArray_Type, 2, ign_dims, NPY_INT32, ign_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    npy_intp igd_dims[2] = {p, m};
    npy_intp igd_strides[2] = {sizeof(i32), ldigd * sizeof(i32)};
    PyArrayObject *igd_array = (PyArrayObject *)PyArray_New(
        &PyArray_Type, 2, igd_dims, NPY_INT32, igd_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    npy_intp gn_dims[1] = {p * m * md};
    PyArrayObject *gn_array = (PyArrayObject *)PyArray_New(
        &PyArray_Type, 1, gn_dims, NPY_DOUBLE, NULL, NULL, 0, 0, NULL);

    npy_intp gd_dims[1] = {p * m * md};
    PyArrayObject *gd_array = (PyArrayObject *)PyArray_New(
        &PyArray_Type, 1, gd_dims, NPY_DOUBLE, NULL, NULL, 0, 0, NULL);

    if (!ign_array || !igd_array || !gn_array || !gd_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_XDECREF(ign_array);
        Py_XDECREF(igd_array);
        Py_XDECREF(gn_array);
        Py_XDECREF(gd_array);
        return PyErr_NoMemory();
    }

    i32 *ign_data = (i32 *)PyArray_DATA(ign_array);
    i32 *igd_data = (i32 *)PyArray_DATA(igd_array);
    f64 *gn_data = (f64 *)PyArray_DATA(gn_array);
    f64 *gd_data = (f64 *)PyArray_DATA(gd_array);

    i32 liwork = n > 0 ? n : 1;
    i32 *iwork = (i32 *)calloc(liwork, sizeof(i32));

    // LDWORK >= MAX(1, N*(N+P) + MAX(N + MAX(N,P), N*(2*N+5)))
    i32 np_term = n * (n + p);
    i32 inner1 = n + (n > p ? n : p);
    i32 inner2 = n * (2 * n + 5);
    i32 ldwork = np_term + (inner1 > inner2 ? inner1 : inner2);
    if (ldwork < 1) ldwork = 1;
    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(ign_array);
        Py_DECREF(igd_array);
        Py_DECREF(gn_array);
        Py_DECREF(gd_array);
        return PyErr_NoMemory();
    }

    i32 info;
    tb04bd(jobd, order, equil, n, m, p, md,
           a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           ign_data, ldign, igd_data, ldigd,
           gn_data, gd_data, tol,
           iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    if (info != 0) {
        Py_DECREF(ign_array);
        Py_DECREF(igd_array);
        Py_DECREF(gn_array);
        Py_DECREF(gd_array);
        PyErr_Format(PyExc_RuntimeError, "tb04bd failed with info=%d", info);
        return NULL;
    }

    return Py_BuildValue("NNNNi",
                         (PyObject *)ign_array,
                         (PyObject *)igd_array,
                         (PyObject *)gn_array,
                         (PyObject *)gd_array,
                         info);
}

PyObject *py_tb04cd(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"jobd", "equil", "a", "b", "c", "d", "npz", "tol", NULL};

    const char *jobd_str, *equil_str;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    int npz_in;
    double tol = 0.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssOOOOi|d", kwlist,
                                     &jobd_str, &equil_str,
                                     &a_obj, &b_obj, &c_obj, &d_obj,
                                     &npz_in, &tol)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject *)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *c_array = (PyArrayObject *)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *d_array = (PyArrayObject *)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);

    if (!a_array || !b_array || !c_array || !d_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 m = (i32)PyArray_DIM(b_array, 1);
    i32 p = (i32)PyArray_DIM(c_array, 0);
    i32 npz = (i32)npz_in;

    f64 *a_data = (f64 *)PyArray_DATA(a_array);
    f64 *b_data = (f64 *)PyArray_DATA(b_array);
    f64 *c_data = (f64 *)PyArray_DATA(c_array);
    f64 *d_data = (f64 *)PyArray_DATA(d_array);

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldd = (jobd_str[0] == 'D' || jobd_str[0] == 'd') ? (p > 0 ? p : 1) : 1;

    // Allocate output arrays
    npy_intp nz_dims[2] = {p > 0 ? p : 1, m > 0 ? m : 1};
    npy_intp nz_strides[2] = {sizeof(i32), (p > 0 ? p : 1) * (npy_intp)sizeof(i32)};
    PyArrayObject *nz_array = (PyArrayObject *)PyArray_New(
        &PyArray_Type, 2, nz_dims, NPY_INT32, nz_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    npy_intp np_dims[2] = {p > 0 ? p : 1, m > 0 ? m : 1};
    npy_intp np_strides[2] = {sizeof(i32), (p > 0 ? p : 1) * (npy_intp)sizeof(i32)};
    PyArrayObject *np_array = (PyArrayObject *)PyArray_New(
        &PyArray_Type, 2, np_dims, NPY_INT32, np_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    npy_intp pz_len = (npy_intp)p * m * npz;
    if (pz_len < 1) pz_len = 1;
    npy_intp pz_dims[1] = {pz_len};

    PyArrayObject *zerosr_array = (PyArrayObject *)PyArray_SimpleNew(1, pz_dims, NPY_DOUBLE);
    PyArrayObject *zerosi_array = (PyArrayObject *)PyArray_SimpleNew(1, pz_dims, NPY_DOUBLE);
    PyArrayObject *polesr_array = (PyArrayObject *)PyArray_SimpleNew(1, pz_dims, NPY_DOUBLE);
    PyArrayObject *polesi_array = (PyArrayObject *)PyArray_SimpleNew(1, pz_dims, NPY_DOUBLE);

    npy_intp gains_dims[2] = {p > 0 ? p : 1, m > 0 ? m : 1};
    npy_intp gains_strides[2] = {sizeof(f64), (p > 0 ? p : 1) * (npy_intp)sizeof(f64)};
    PyArrayObject *gains_array = (PyArrayObject *)PyArray_New(
        &PyArray_Type, 2, gains_dims, NPY_DOUBLE, gains_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (!nz_array || !np_array || !zerosr_array || !zerosi_array ||
        !polesr_array || !polesi_array || !gains_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_XDECREF(nz_array);
        Py_XDECREF(np_array);
        Py_XDECREF(zerosr_array);
        Py_XDECREF(zerosi_array);
        Py_XDECREF(polesr_array);
        Py_XDECREF(polesi_array);
        Py_XDECREF(gains_array);
        return PyErr_NoMemory();
    }

    i32 *nz_data = (i32 *)PyArray_DATA(nz_array);
    i32 *np_data = (i32 *)PyArray_DATA(np_array);
    f64 *zerosr_data = (f64 *)PyArray_DATA(zerosr_array);
    f64 *zerosi_data = (f64 *)PyArray_DATA(zerosi_array);
    f64 *polesr_data = (f64 *)PyArray_DATA(polesr_array);
    f64 *polesi_data = (f64 *)PyArray_DATA(polesi_array);
    f64 *gains_data = (f64 *)PyArray_DATA(gains_array);

    i32 ldnz = p > 0 ? p : 1;
    i32 ldnp = p > 0 ? p : 1;
    i32 ldgain = p > 0 ? p : 1;

    i32 liwork = n > 0 ? n : 1;
    i32 *iwork = (i32 *)calloc(liwork, sizeof(i32));

    // LDWORK >= MAX(1, N*(N+P) + MAX(N + MAX(N,P), N*(2*N+3)))
    i32 np_term = n * (n + p);
    i32 inner1 = n + (n > p ? n : p);
    i32 inner2 = n * (2 * n + 3);
    i32 ldwork = np_term + (inner1 > inner2 ? inner1 : inner2);
    if (ldwork < 1) ldwork = 1;
    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(nz_array);
        Py_DECREF(np_array);
        Py_DECREF(zerosr_array);
        Py_DECREF(zerosi_array);
        Py_DECREF(polesr_array);
        Py_DECREF(polesi_array);
        Py_DECREF(gains_array);
        return PyErr_NoMemory();
    }

    i32 info;
    tb04cd(jobd_str, equil_str, n, m, p, npz,
           a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           nz_data, ldnz, np_data, ldnp,
           zerosr_data, zerosi_data, polesr_data, polesi_data,
           gains_data, ldgain, tol,
           iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    if (info != 0) {
        Py_DECREF(nz_array);
        Py_DECREF(np_array);
        Py_DECREF(zerosr_array);
        Py_DECREF(zerosi_array);
        Py_DECREF(polesr_array);
        Py_DECREF(polesi_array);
        Py_DECREF(gains_array);
        PyErr_Format(PyExc_RuntimeError, "tb04cd failed with info=%d", info);
        return NULL;
    }

    return Py_BuildValue("NNNNNNNi",
                         (PyObject *)nz_array,
                         (PyObject *)np_array,
                         (PyObject *)zerosr_array,
                         (PyObject *)zerosi_array,
                         (PyObject *)polesr_array,
                         (PyObject *)polesi_array,
                         (PyObject *)gains_array,
                         info);
}

