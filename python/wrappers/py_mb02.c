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



/* Python wrapper for mb02uv */
PyObject* py_mb02uv(PyObject* self, PyObject* args) {
    i32 n;
    PyObject *a_obj;
    PyArrayObject *a_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "iO", &n, &a_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda = (i32)a_dims[0];

    f64 *a_data = (f64*)PyArray_DATA(a_array);

    i32 ipiv_size = n > 0 ? n : 1;
    i32 *ipiv = (i32*)malloc(ipiv_size * sizeof(i32));
    i32 *jpiv = (i32*)malloc(ipiv_size * sizeof(i32));
    if (ipiv == NULL || jpiv == NULL) {
        free(ipiv);
        free(jpiv);
        Py_DECREF(a_array);
        return PyErr_NoMemory();
    }

    mb02uv(n, a_data, lda, ipiv, jpiv, &info);

    npy_intp pivot_dims[1] = {n};
    PyArrayObject *ipiv_array = (PyArrayObject*)PyArray_SimpleNew(1, pivot_dims, NPY_INT32);
    PyArrayObject *jpiv_array = (PyArrayObject*)PyArray_SimpleNew(1, pivot_dims, NPY_INT32);

    if (ipiv_array == NULL || jpiv_array == NULL) {
        free(ipiv);
        free(jpiv);
        Py_XDECREF(ipiv_array);
        Py_XDECREF(jpiv_array);
        Py_DECREF(a_array);
        return PyErr_NoMemory();
    }

    memcpy(PyArray_DATA(ipiv_array), ipiv, n * sizeof(i32));
    memcpy(PyArray_DATA(jpiv_array), jpiv, n * sizeof(i32));

    free(ipiv);
    free(jpiv);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyObject *result = Py_BuildValue("OOOi", a_array, ipiv_array, jpiv_array, info);
    Py_DECREF(a_array);
    Py_DECREF(ipiv_array);
    Py_DECREF(jpiv_array);
    return result;
}



/* Python wrapper for mb02uu */
PyObject* py_mb02uu(PyObject* self, PyObject* args) {
    i32 n;
    PyObject *a_obj, *rhs_obj, *ipiv_obj, *jpiv_obj;
    PyArrayObject *a_array, *rhs_array, *ipiv_array, *jpiv_array;

    if (!PyArg_ParseTuple(args, "iOOOO", &n, &a_obj, &rhs_obj, &ipiv_obj, &jpiv_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    rhs_array = (PyArrayObject*)PyArray_FROM_OTF(rhs_obj, NPY_DOUBLE,
                                                  NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (rhs_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    ipiv_array = (PyArrayObject*)PyArray_FROM_OTF(ipiv_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (ipiv_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(rhs_array);
        return NULL;
    }

    jpiv_array = (PyArrayObject*)PyArray_FROM_OTF(jpiv_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (jpiv_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(rhs_array);
        Py_DECREF(ipiv_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 lda = (i32)a_dims[0];

    const f64 *a_data = (const f64*)PyArray_DATA(a_array);
    f64 *rhs_data = (f64*)PyArray_DATA(rhs_array);
    const i32 *ipiv_data = (const i32*)PyArray_DATA(ipiv_array);
    const i32 *jpiv_data = (const i32*)PyArray_DATA(jpiv_array);

    f64 scale;

    mb02uu(n, a_data, lda, rhs_data, ipiv_data, jpiv_data, &scale);

    Py_DECREF(a_array);
    Py_DECREF(ipiv_array);
    Py_DECREF(jpiv_array);

    PyArray_ResolveWritebackIfCopy(rhs_array);
    PyObject *result = Py_BuildValue("Od", rhs_array, scale);
    Py_DECREF(rhs_array);
    return result;
}



/* Python wrapper for mb02yd */
PyObject* py_mb02yd(PyObject* self, PyObject* args) {
    char* cond;
    i32 n, rank_in;
    f64 tol;
    PyObject *r_obj, *ipvt_obj, *diag_obj, *qtb_obj;
    PyArrayObject *r_array, *ipvt_array, *diag_array, *qtb_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "siOOOOid", &cond, &n, &r_obj, &ipvt_obj, &diag_obj, &qtb_obj, &rank_in, &tol)) {
        return NULL;
    }

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) return NULL;

    ipvt_array = (PyArrayObject*)PyArray_FROM_OTF(ipvt_obj, NPY_INT32, NPY_ARRAY_IN_FARRAY);
    if (ipvt_array == NULL) {
        Py_DECREF(r_array);
        return NULL;
    }

    diag_array = (PyArrayObject*)PyArray_FROM_OTF(diag_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (diag_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(ipvt_array);
        return NULL;
    }

    qtb_array = (PyArrayObject*)PyArray_FROM_OTF(qtb_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (qtb_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(ipvt_array);
        Py_DECREF(diag_array);
        return NULL;
    }

    i32 ldr = (i32)PyArray_DIM(r_array, 0);
    f64* r_data = (f64*)PyArray_DATA(r_array);
    i32* ipvt_data = (i32*)PyArray_DATA(ipvt_array);
    f64* diag_data = (f64*)PyArray_DATA(diag_array);
    f64* qtb_data = (f64*)PyArray_DATA(qtb_array);

    bool econd = (*cond == 'E' || *cond == 'e');
    i32 ldwork = econd ? 4*n : 2*n;
    if (ldwork < 1) ldwork = 1;  /* Ensure at least 1 element for malloc */
    f64* dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(ipvt_array);
        Py_DECREF(diag_array);
        Py_DECREF(qtb_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    /* Allocate output array x */
    f64* x_data = (n > 0) ? (f64*)calloc(n, sizeof(f64)) : NULL;
    if (n > 0 && x_data == NULL) {
        free(dwork);
        Py_DECREF(r_array);
        Py_DECREF(ipvt_array);
        Py_DECREF(diag_array);
        Py_DECREF(qtb_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output array");
        return NULL;
    }

    i32 rank = rank_in;

    mb02yd(cond, n, r_data, ldr, ipvt_data, diag_data, qtb_data, &rank, x_data, tol, dwork, ldwork, &info);

    free(dwork);

    /* Resolve writebackifcopy before decref */
    PyArray_ResolveWritebackIfCopy(r_array);

    /* Create NumPy array for output */
    npy_intp x_dims[1] = {n > 0 ? n : 0};
    PyObject* x_array;
    if (n > 0) {
        x_array = PyArray_SimpleNew(1, x_dims, NPY_DOUBLE);
        if (x_array == NULL) {
            free(x_data);
            Py_DECREF(r_array);
            Py_DECREF(ipvt_array);
            Py_DECREF(diag_array);
            Py_DECREF(qtb_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to create output array");
            return NULL;
        }
        f64 *x_out_data = (f64*)PyArray_DATA((PyArrayObject*)x_array);
        memcpy(x_out_data, x_data, (size_t)n * sizeof(f64));
        free(x_data);
    } else {
        x_array = PyArray_EMPTY(1, x_dims, NPY_DOUBLE, 0);
        if (x_array == NULL) {
            Py_DECREF(r_array);
            Py_DECREF(ipvt_array);
            Py_DECREF(diag_array);
            Py_DECREF(qtb_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to create output array");
            return NULL;
        }
    }

    if (info < 0) {
        Py_DECREF(r_array);
        Py_DECREF(ipvt_array);
        Py_DECREF(diag_array);
        Py_DECREF(qtb_array);
        Py_DECREF(x_array);
        PyErr_Format(PyExc_ValueError, "mb02yd: parameter %d is invalid", -info);
        return NULL;
    }

    PyObject* result = Py_BuildValue("Oii", x_array, rank, info);

    Py_DECREF(r_array);
    Py_DECREF(ipvt_array);
    Py_DECREF(diag_array);
    Py_DECREF(qtb_array);
    Py_DECREF(x_array);

    return result;
}



PyObject* py_mb02ud(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"fact", "side", "trans", "jobp", "m", "n", "alpha", "rcond",
                             "r", "b", "q", "sv", "rank", "rp", "ldwork", NULL};

    char *fact_str, *side_str, *trans_str, *jobp_str;
    i32 m_in, n_in, rank_in = 0;
    f64 alpha, rcond;
    i32 ldwork = 0;
    PyObject *r_obj, *b_obj;
    PyObject *q_obj = NULL, *sv_obj = NULL, *rp_obj = NULL;
    PyArrayObject *r_array, *b_array;
    PyArrayObject *q_array = NULL, *sv_array = NULL;
    i32 info;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssssiiddOO|OOiOi", kwlist,
                                     &fact_str, &side_str, &trans_str, &jobp_str,
                                     &m_in, &n_in, &alpha, &rcond, &r_obj, &b_obj,
                                     &q_obj, &sv_obj, &rank_in, &rp_obj, &ldwork)) {
        return NULL;
    }

    char fact = fact_str[0];
    char side = side_str[0];
    char jobp = jobp_str[0];

    bool nfct = (fact == 'N' || fact == 'n');
    bool left = (side == 'L' || side == 'l');
    bool pinv = (jobp == 'P' || jobp == 'p');
    (void)rp_obj;

    i32 l = left ? m_in : n_in;

    if (m_in < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be non-negative");
        return NULL;
    }
    if (n_in < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) return NULL;

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(r_array);
        return NULL;
    }

    i32 ldr = (i32)PyArray_DIM(r_array, 0);
    i32 ldb = (i32)PyArray_DIM(b_array, 0);

    f64* r_data = (f64*)PyArray_DATA(r_array);
    f64* b_data = (f64*)PyArray_DATA(b_array);

    f64* q_data = NULL;
    f64* sv_data = NULL;
    i32 ldq = l > 0 ? l : 1;

    if (nfct) {
        q_data = (f64*)calloc(l > 0 ? l * l : 1, sizeof(f64));
        sv_data = (f64*)calloc(l > 0 ? l : 1, sizeof(f64));
        if ((l > 0 && q_data == NULL) || (l > 0 && sv_data == NULL)) {
            free(q_data);
            free(sv_data);
            Py_DECREF(r_array);
            Py_DECREF(b_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate Q or SV");
            return NULL;
        }
    } else {
        if (q_obj == NULL || sv_obj == NULL) {
            Py_DECREF(r_array);
            Py_DECREF(b_array);
            PyErr_SetString(PyExc_ValueError, "q and sv are required when fact='F'");
            return NULL;
        }
        q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (q_array == NULL) {
            Py_DECREF(r_array);
            Py_DECREF(b_array);
            return NULL;
        }
        sv_array = (PyArrayObject*)PyArray_FROM_OTF(sv_obj, NPY_DOUBLE,
                                                    NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (sv_array == NULL) {
            Py_DECREF(r_array);
            Py_DECREF(b_array);
            Py_DECREF(q_array);
            return NULL;
        }
        q_data = (f64*)PyArray_DATA(q_array);
        sv_data = (f64*)PyArray_DATA(sv_array);
        ldq = (i32)PyArray_DIM(q_array, 0);
    }

    f64* rp_data = NULL;
    i32 ldrp = 1;
    if (pinv) {
        ldrp = l > 0 ? l : 1;
        rp_data = (f64*)calloc(ldrp * l > 0 ? ldrp * l : 1, sizeof(f64));
        if (l > 0 && rp_data == NULL) {
            if (nfct) { free(q_data); free(sv_data); }
            else { Py_XDECREF(q_array); Py_XDECREF(sv_array); }
            Py_DECREF(r_array);
            Py_DECREF(b_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate RP");
            return NULL;
        }
    }

    i32 minwrk = nfct ? (5 * l > 1 ? 5 * l : 1) : (l > 1 ? l : 1);
    i32 mn = m_in * n_in;
    i32 optwork = minwrk > mn ? minwrk : mn;
    if (ldwork == 0) ldwork = optwork;
    if (ldwork < minwrk) ldwork = minwrk;

    f64* dwork = (f64*)calloc(ldwork > 0 ? ldwork : 1, sizeof(f64));
    if (dwork == NULL) {
        if (pinv) free(rp_data);
        if (nfct) { free(q_data); free(sv_data); }
        else { Py_XDECREF(q_array); Py_XDECREF(sv_array); }
        Py_DECREF(r_array);
        Py_DECREF(b_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    i32 rank = nfct ? 0 : rank_in;

    mb02ud(fact_str, side_str, trans_str, jobp_str, m_in, n_in, alpha, rcond,
           &rank, r_data, ldr, q_data, ldq, sv_data, b_data, ldb,
           rp_data, ldrp, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(r_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    if (!nfct) {
        PyArray_ResolveWritebackIfCopy(q_array);
        PyArray_ResolveWritebackIfCopy(sv_array);
    }

    if (info < 0) {
        if (pinv) free(rp_data);
        if (nfct) { free(q_data); free(sv_data); }
        else { Py_XDECREF(q_array); Py_XDECREF(sv_array); }
        Py_DECREF(r_array);
        Py_DECREF(b_array);
        PyErr_Format(PyExc_ValueError, "mb02ud: parameter %d is invalid", -info);
        return NULL;
    }

    npy_intp q_dims[2] = {l, l};
    npy_intp q_strides[2] = {sizeof(f64), l * sizeof(f64)};
    PyObject* q_out;
    if (nfct) {
        q_out = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE,
                            q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (q_out == NULL) {
            free(q_data);
            free(sv_data);
            if (pinv) free(rp_data);
            Py_DECREF(r_array);
            Py_DECREF(b_array);
            return NULL;
        }
        f64 *q_out_data = (f64*)PyArray_DATA((PyArrayObject*)q_out);
        memcpy(q_out_data, q_data, (size_t)l * (size_t)l * sizeof(f64));
        free(q_data);
    } else {
        q_out = (PyObject*)q_array;
    }

    npy_intp sv_dims[1] = {l};
    PyObject* sv_out;
    if (nfct) {
        sv_out = PyArray_SimpleNew(1, sv_dims, NPY_DOUBLE);
        if (sv_out == NULL) {
            free(sv_data);
            Py_DECREF(q_out);
            if (pinv) free(rp_data);
            Py_DECREF(r_array);
            Py_DECREF(b_array);
            return NULL;
        }
        f64 *sv_out_data = (f64*)PyArray_DATA((PyArrayObject*)sv_out);
        memcpy(sv_out_data, sv_data, (size_t)l * sizeof(f64));
        free(sv_data);
    } else {
        sv_out = (PyObject*)sv_array;
    }

    PyObject* rp_out;
    if (pinv && rank > 0) {
        npy_intp rp_dims[2] = {l, l};
        npy_intp rp_strides[2] = {sizeof(f64), l * sizeof(f64)};
        rp_out = PyArray_New(&PyArray_Type, 2, rp_dims, NPY_DOUBLE,
                             rp_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (rp_out == NULL) {
            free(rp_data);
            Py_DECREF(q_out);
            Py_DECREF(sv_out);
            Py_DECREF(r_array);
            Py_DECREF(b_array);
            return NULL;
        }
        f64 *rp_out_data = (f64*)PyArray_DATA((PyArrayObject*)rp_out);
        memcpy(rp_out_data, rp_data, (size_t)l * (size_t)l * sizeof(f64));
        free(rp_data);
    } else {
        if (pinv) free(rp_data);
        Py_INCREF(Py_None);
        rp_out = Py_None;
    }

    PyObject* result = Py_BuildValue("OOOiOi", b_array, q_out, sv_out, rank, rp_out, info);

    Py_DECREF(r_array);
    Py_DECREF(b_array);
    Py_DECREF(q_out);
    Py_DECREF(sv_out);
    Py_DECREF(rp_out);

    return result;
}



/* Python wrapper for mb02vd */
PyObject* py_mb02vd(PyObject* self, PyObject* args) {
    char* trans;
    PyObject *a_obj, *b_obj;
    PyArrayObject *a_array, *b_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "sOO", &trans, &a_obj, &b_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 m = (i32)PyArray_DIM(b_array, 0);
    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 lda = (n > 1) ? n : 1;
    i32 ldb = (m > 1) ? m : 1;

    f64* a = (f64*)PyArray_DATA(a_array);
    f64* b = (f64*)PyArray_DATA(b_array);

    i32* ipiv = (i32*)malloc(n * sizeof(i32));
    if (ipiv == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return PyErr_NoMemory();
    }

    mb02vd(trans, m, n, a, lda, ipiv, b, ldb, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    npy_intp ipiv_dims[1] = {n};
    PyObject* ipiv_array = PyArray_SimpleNew(1, ipiv_dims, NPY_INT32);
    memcpy(PyArray_DATA((PyArrayObject*)ipiv_array), ipiv, n * sizeof(i32));
    free(ipiv);

    PyObject* result = Py_BuildValue("OOi", b_array, ipiv_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(ipiv_array);

    return result;
}


PyObject* py_mb02wd(PyObject* self, PyObject* args) {
    char* form;
    int itmax;
    double tol;
    PyObject *a_obj, *b_obj, *x_obj;

    if (!PyArg_ParseTuple(args, "siOOOd", &form, &itmax, &a_obj, &b_obj, &x_obj, &tol)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (!b_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *x_array = (PyArrayObject*)PyArray_FROM_OTF(
        x_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!x_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 lda = n > 1 ? n : 1;
    i32 incb = 1;
    i32 incx = 1;
    i32 ldwork = 3 * n > 2 ? 3 * n : 2;

    f64* a = (f64*)PyArray_DATA(a_array);
    f64* b = (f64*)PyArray_DATA(b_array);
    f64* x = (f64*)PyArray_DATA(x_array);

    f64* dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (!dwork) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(x_array);
        return PyErr_NoMemory();
    }

    i32 iwarn = 0;
    i32 info = 0;

    mb02wd(form, NULL, n, NULL, 0, NULL, 0, itmax,
           a, lda, b, incb, x, incx, tol, dwork, ldwork, &iwarn, &info);

    PyArray_ResolveWritebackIfCopy(x_array);

    double iterations = dwork[0];
    double residual = dwork[1];
    free(dwork);

    PyObject* result = Py_BuildValue("(Oddii)", x_array, iterations, residual, iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(x_array);

    return result;
}


/* Python wrapper for mb02sd */
PyObject* py_mb02sd(PyObject* self, PyObject* args) {
    int n;
    PyObject *h_obj;
    PyArrayObject *h_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "iO", &n, &h_obj)) {
        return NULL;
    }

    h_array = (PyArrayObject*)PyArray_FROM_OTF(h_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (h_array == NULL) return NULL;

    i32 ldh = (n > 1) ? n : 1;
    if (n > 0) {
        ldh = (i32)PyArray_DIM(h_array, 0);
    }

    f64* h = (f64*)PyArray_DATA(h_array);

    i32* ipiv = NULL;
    if (n > 0) {
        ipiv = (i32*)malloc(n * sizeof(i32));
        if (ipiv == NULL) {
            Py_DECREF(h_array);
            return PyErr_NoMemory();
        }
    }

    mb02sd(n, h, ldh, ipiv, &info);

    PyArray_ResolveWritebackIfCopy(h_array);

    npy_intp ipiv_dims[1] = {n > 0 ? n : 1};
    PyObject* ipiv_array = PyArray_SimpleNew(1, ipiv_dims, NPY_INT32);
    if (n > 0 && ipiv != NULL) {
        memcpy(PyArray_DATA((PyArrayObject*)ipiv_array), ipiv, n * sizeof(i32));
        free(ipiv);
    }

    PyObject* result = Py_BuildValue("OOi", h_array, ipiv_array, info);

    Py_DECREF(h_array);
    Py_DECREF(ipiv_array);

    return result;
}



/* Python wrapper for mb02rd */
PyObject* py_mb02rd(PyObject* self, PyObject* args) {
    char* trans;
    int n, nrhs;
    PyObject *h_obj, *ipiv_obj, *b_obj;
    PyArrayObject *h_array, *ipiv_array, *b_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "siiOOO", &trans, &n, &nrhs, &h_obj, &ipiv_obj, &b_obj)) {
        return NULL;
    }

    h_array = (PyArrayObject*)PyArray_FROM_OTF(h_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (h_array == NULL) return NULL;

    ipiv_array = (PyArrayObject*)PyArray_FROM_OTF(ipiv_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (ipiv_array == NULL) {
        Py_DECREF(h_array);
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(h_array);
        Py_DECREF(ipiv_array);
        return NULL;
    }

    i32 ldh = (n > 1) ? n : 1;
    if (n > 0) {
        ldh = (i32)PyArray_DIM(h_array, 0);
    }
    i32 ldb = (n > 1) ? n : 1;
    if (n > 0) {
        ldb = (i32)PyArray_DIM(b_array, 0);
    }

    const f64* h = (const f64*)PyArray_DATA(h_array);
    const i32* ipiv = (const i32*)PyArray_DATA(ipiv_array);
    f64* b = (f64*)PyArray_DATA(b_array);

    mb02rd(trans, n, nrhs, h, ldh, ipiv, b, ldb, &info);

    PyArray_ResolveWritebackIfCopy(b_array);

    PyObject* result = Py_BuildValue("Oi", b_array, info);

    Py_DECREF(h_array);
    Py_DECREF(ipiv_array);
    Py_DECREF(b_array);

    return result;
}



/* Python wrapper for mb02pd */
PyObject* py_mb02pd(PyObject* self, PyObject* args) {
    const char *fact_str, *trans_str;
    PyObject *a_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "ssOO", &fact_str, &trans_str, &a_obj, &b_obj)) {
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

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 nrhs = (i32)PyArray_DIM(b_array, 1);
    i32 lda = n > 0 ? n : 1;
    i32 ldaf = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldx = n > 0 ? n : 1;

    npy_intp af_dims[2] = {n, n};
    npy_intp x_dims[2] = {n, nrhs};
    npy_intp rhs_dims[1] = {nrhs > 0 ? nrhs : 1};

    PyArrayObject *af_array = (PyArrayObject*)PyArray_ZEROS(2, af_dims, NPY_DOUBLE, 1);
    PyArrayObject *x_array = (PyArrayObject*)PyArray_ZEROS(2, x_dims, NPY_DOUBLE, 1);
    PyArrayObject *ferr_array = (PyArrayObject*)PyArray_ZEROS(1, rhs_dims, NPY_DOUBLE, 0);
    PyArrayObject *berr_array = (PyArrayObject*)PyArray_ZEROS(1, rhs_dims, NPY_DOUBLE, 0);

    if (af_array == NULL || x_array == NULL || ferr_array == NULL || berr_array == NULL) {
        Py_XDECREF(af_array);
        Py_XDECREF(x_array);
        Py_XDECREF(ferr_array);
        Py_XDECREF(berr_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    i32 *ipiv = (i32*)calloc(n > 0 ? n : 1, sizeof(i32));
    i32 *iwork = (i32*)calloc(n > 0 ? n : 1, sizeof(i32));
    f64 *r = (f64*)calloc(n > 0 ? n : 1, sizeof(f64));
    f64 *c = (f64*)calloc(n > 0 ? n : 1, sizeof(f64));
    f64 *dwork = (f64*)calloc(4 * n > 1 ? 4 * n : 1, sizeof(f64));

    if (ipiv == NULL || iwork == NULL || r == NULL || c == NULL || dwork == NULL) {
        PyErr_NoMemory();
        free(ipiv); free(iwork); free(r); free(c); free(dwork);
        Py_DECREF(af_array);
        Py_DECREF(x_array);
        Py_DECREF(ferr_array);
        Py_DECREF(berr_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *af_data = (f64*)PyArray_DATA(af_array);
    f64 *x_data = (f64*)PyArray_DATA(x_array);
    f64 *ferr_data = (f64*)PyArray_DATA(ferr_array);
    f64 *berr_data = (f64*)PyArray_DATA(berr_array);

    f64 rcond;
    i32 info;
    char equed = 'N';

    mb02pd(fact_str, trans_str, n, nrhs, a_data, lda, af_data, ldaf, ipiv,
           &equed, r, c, b_data, ldb, x_data, ldx, &rcond, ferr_data, berr_data,
           iwork, dwork, &info);

    free(ipiv);
    free(iwork);
    free(r);
    free(c);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    PyObject *result = Py_BuildValue("OOOdi", x_array, ferr_array, berr_array, rcond, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(af_array);
    Py_DECREF(x_array);
    Py_DECREF(ferr_array);
    Py_DECREF(berr_array);

    return result;
}



/* Python wrapper for mb02sz - Complex Hessenberg LU factorization */
PyObject* py_mb02sz(PyObject* self, PyObject* args) {
    PyObject *h_obj;

    if (!PyArg_ParseTuple(args, "O", &h_obj)) {
        return NULL;
    }

    PyArrayObject *h_array = (PyArrayObject*)PyArray_FROM_OTF(
        h_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!h_array) return NULL;

    npy_intp *h_dims = PyArray_DIMS(h_array);
    i32 n = (i32)h_dims[0];
    i32 ldh = (i32)h_dims[0];

    i32 *ipiv = (i32*)malloc(n * sizeof(i32));
    if (!ipiv) {
        Py_DECREF(h_array);
        PyErr_NoMemory();
        return NULL;
    }

    c128 *h_data = (c128*)PyArray_DATA(h_array);

    i32 info = slicot_mb02sz(n, h_data, ldh, ipiv);

    PyArray_ResolveWritebackIfCopy(h_array);

    npy_intp ipiv_dims[1] = {n};
    PyObject *ipiv_array = PyArray_SimpleNew(1, ipiv_dims, NPY_INT32);
    if (!ipiv_array) {
        free(ipiv);
        Py_DECREF(h_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)ipiv_array), ipiv, n * sizeof(i32));
    free(ipiv);

    PyObject *result = Py_BuildValue("OOi", h_array, ipiv_array, info);
    Py_DECREF(h_array);
    Py_DECREF(ipiv_array);
    return result;
}



/* Python wrapper for mb02rz - Solve complex Hessenberg system */
PyObject* py_mb02rz(PyObject* self, PyObject* args) {
    const char *trans_str;
    PyObject *h_obj, *ipiv_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "sOOO", &trans_str, &h_obj, &ipiv_obj, &b_obj)) {
        return NULL;
    }

    char trans = trans_str[0];

    PyArrayObject *h_array = (PyArrayObject*)PyArray_FROM_OTF(
        h_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!h_array) return NULL;

    PyArrayObject *ipiv_array = (PyArrayObject*)PyArray_FROM_OTF(
        ipiv_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (!ipiv_array) {
        Py_DECREF(h_array);
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) {
        Py_DECREF(h_array);
        Py_DECREF(ipiv_array);
        return NULL;
    }

    npy_intp *h_dims = PyArray_DIMS(h_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);

    i32 n = (i32)h_dims[0];
    i32 ldh = (i32)h_dims[0];
    i32 nrhs = (PyArray_NDIM(b_array) > 1) ? (i32)b_dims[1] : 1;
    i32 ldb = (i32)b_dims[0];

    const c128 *h_data = (const c128*)PyArray_DATA(h_array);
    const i32 *ipiv_data = (const i32*)PyArray_DATA(ipiv_array);
    c128 *b_data = (c128*)PyArray_DATA(b_array);

    i32 info = slicot_mb02rz(trans, n, nrhs, h_data, ldh, ipiv_data, b_data, ldb);

    PyArray_ResolveWritebackIfCopy(b_array);

    PyObject *result = Py_BuildValue("Oi", b_array, info);
    Py_DECREF(h_array);
    Py_DECREF(ipiv_array);
    Py_DECREF(b_array);
    return result;
}



/* Python wrapper for mb02tz - Condition estimation of complex Hessenberg matrix */
PyObject* py_mb02tz(PyObject* self, PyObject* args) {
    const char *norm_str;
    f64 hnorm;
    PyObject *h_obj, *ipiv_obj;

    if (!PyArg_ParseTuple(args, "sdOO", &norm_str, &hnorm, &h_obj, &ipiv_obj)) {
        return NULL;
    }

    char norm = norm_str[0];

    PyArrayObject *h_array = (PyArrayObject*)PyArray_FROM_OTF(
        h_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!h_array) return NULL;

    PyArrayObject *ipiv_array = (PyArrayObject*)PyArray_FROM_OTF(
        ipiv_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (!ipiv_array) {
        Py_DECREF(h_array);
        return NULL;
    }

    npy_intp *h_dims = PyArray_DIMS(h_array);
    i32 n = (i32)h_dims[0];
    i32 ldh = (i32)h_dims[0];

    f64 *dwork = (f64*)malloc(n * sizeof(f64));
    c128 *zwork = (c128*)malloc(2 * n * sizeof(c128));
    if (!dwork || !zwork) {
        free(dwork);
        free(zwork);
        Py_DECREF(h_array);
        Py_DECREF(ipiv_array);
        PyErr_NoMemory();
        return NULL;
    }

    const c128 *h_data = (const c128*)PyArray_DATA(h_array);
    const i32 *ipiv_data = (const i32*)PyArray_DATA(ipiv_array);
    f64 rcond = 0.0;

    i32 info = slicot_mb02tz(norm, n, hnorm, h_data, ldh, ipiv_data, &rcond, dwork, zwork);

    free(dwork);
    free(zwork);

    PyObject *result = Py_BuildValue("di", rcond, info);
    Py_DECREF(h_array);
    Py_DECREF(ipiv_array);
    return result;
}



/* Python wrapper for mb02ed - Solve symmetric positive definite block Toeplitz system */
PyObject* py_mb02ed(PyObject* self, PyObject* args) {
    const char *typet_str;
    i32 k, n, nrhs;
    PyObject *t_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "siiiOO", &typet_str, &k, &n, &nrhs, &t_obj, &b_obj)) {
        return NULL;
    }

    char typet_u = typet_str[0];
    if (typet_u >= 'a' && typet_u <= 'z') typet_u -= 32;
    (void)typet_u;

    if (k < 0 || n < 0 || nrhs < 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid dimensions");
        return NULL;
    }

    PyArrayObject *t_array = (PyArrayObject*)PyArray_FROM_OTF(
        t_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!t_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) {
        Py_DECREF(t_array);
        return NULL;
    }

    i32 ldt = (i32)PyArray_DIM(t_array, 0);
    i32 ldb = (i32)PyArray_DIM(b_array, 0);

    f64 *t_data = (f64*)PyArray_DATA(t_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    i32 ldwork = n * k * k + (n + 2) * k;
    if (ldwork < 1) ldwork = 1;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (!dwork) {
        Py_DECREF(t_array);
        Py_DECREF(b_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info;
    mb02ed(typet_str, k, n, nrhs, t_data, ldt, b_data, ldb, dwork, ldwork, &info);

    free(dwork);

    if (info < 0) {
        Py_DECREF(t_array);
        Py_DECREF(b_array);
        PyErr_Format(PyExc_RuntimeError, "mb02ed failed with info=%d", info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(t_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    PyObject *result = Py_BuildValue("OOi", b_array, t_array, info);
    Py_DECREF(t_array);
    Py_DECREF(b_array);
    return result;
}



/* Python wrapper for mb02cx - Bring first blocks of generator to proper form */
PyObject* py_mb02cx(PyObject* self, PyObject* args) {
    const char *typet_str;
    i32 p, q, k;
    PyObject *a_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "siiiOO", &typet_str, &p, &q, &k, &a_obj, &b_obj)) {
        return NULL;
    }

    char typet_u = typet_str[0];
    if (typet_u >= 'a' && typet_u <= 'z') typet_u -= 32;
    (void)typet_u;

    if (p < 0 || q < 0 || k < 0 || k > p) {
        PyErr_SetString(PyExc_ValueError, "Invalid dimensions");
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

    i32 lda = (i32)PyArray_DIM(a_array, 0);
    i32 ldb = (i32)PyArray_DIM(b_array, 0);

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    i32 minKQ = (k < q) ? k : q;
    i32 lcs = 2 * k + minKQ;
    f64 *cs = (f64*)calloc(lcs > 0 ? lcs : 1, sizeof(f64));
    i32 ldwork = k > 1 ? k : 1;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (!cs || !dwork) {
        free(cs);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info;
    mb02cx(typet_str, p, q, k, a_data, lda, b_data, ldb, cs, lcs, dwork, ldwork, &info);

    free(dwork);

    if (info < 0) {
        free(cs);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        PyErr_Format(PyExc_RuntimeError, "mb02cx failed with info=%d", info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    npy_intp cs_dims[1] = {lcs > 0 ? lcs : 1};
    PyArrayObject *cs_array = (PyArrayObject*)PyArray_SimpleNew(1, cs_dims, NPY_DOUBLE);
    if (!cs_array) {
        free(cs);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        PyErr_NoMemory();
        return NULL;
    }
    f64 *cs_out_data = (f64*)PyArray_DATA(cs_array);
    memcpy(cs_out_data, cs, (size_t)cs_dims[0] * sizeof(f64));
    free(cs);

    PyObject *result = Py_BuildValue("OOOi", a_array, b_array, cs_array, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(cs_array);
    return result;
}



/* Python wrapper for mb02cy - Apply hyperbolic transformations to generator columns/rows */
PyObject* py_mb02cy(PyObject* self, PyObject* args) {
    const char *typet_str, *strucg_str;
    i32 p, q, n, k;
    PyObject *a_obj, *b_obj, *h_obj, *cs_obj;

    if (!PyArg_ParseTuple(args, "ssiiiiOOOO", &typet_str, &strucg_str,
                          &p, &q, &n, &k, &a_obj, &b_obj, &h_obj, &cs_obj)) {
        return NULL;
    }

    char typet_u = typet_str[0];
    if (typet_u >= 'a' && typet_u <= 'z') typet_u -= 32;
    (void)typet_u;

    if (p < 0 || q < 0 || n < 0 || k < 0 || k > p) {
        PyErr_SetString(PyExc_ValueError, "Invalid dimensions");
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

    PyArrayObject *h_array = (PyArrayObject*)PyArray_FROM_OTF(
        h_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!h_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *cs_array = (PyArrayObject*)PyArray_FROM_OTF(
        cs_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!cs_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(h_array);
        return NULL;
    }

    i32 lda = (i32)PyArray_DIM(a_array, 0);
    i32 ldb = (i32)PyArray_DIM(b_array, 0);
    i32 ldh = (i32)PyArray_DIM(h_array, 0);
    i32 lcs = (i32)PyArray_SIZE(cs_array);

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *h_data = (f64*)PyArray_DATA(h_array);
    f64 *cs_data = (f64*)PyArray_DATA(cs_array);

    i32 ldwork = n > 1 ? n : 1;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (!dwork) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(h_array);
        Py_DECREF(cs_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info;
    mb02cy(typet_str, strucg_str, p, q, n, k, a_data, lda, b_data, ldb,
           h_data, ldh, cs_data, lcs, dwork, ldwork, &info);

    free(dwork);

    if (info != 0) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(h_array);
        Py_DECREF(cs_array);
        PyErr_Format(PyExc_RuntimeError, "mb02cy failed with info=%d", info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(h_array);
    PyArray_ResolveWritebackIfCopy(cs_array);

    PyObject *result = Py_BuildValue("OOi", a_array, b_array, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(h_array);
    Py_DECREF(cs_array);
    return result;
}


/* Python wrapper for mb02cu - Bring first part of generator to proper form */
PyObject* py_mb02cu(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"typeg", "k", "p", "q", "nb", "a1", "a2", "b", "tol", NULL};
    const char *typeg_str;
    i32 k, p, q, nb;
    f64 tol = 0.0;
    PyObject *a1_obj, *a2_obj, *b_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siiiiOOO|d", kwlist,
                                     &typeg_str, &k, &p, &q, &nb,
                                     &a1_obj, &a2_obj, &b_obj, &tol)) {
        return NULL;
    }

    char typeg_u = typeg_str[0];
    if (typeg_u >= 'a' && typeg_u <= 'z') typeg_u -= 32;
    bool lrdef = (typeg_u == 'D');
    bool lcol = (typeg_u == 'C');

    if (typeg_u != 'D' && typeg_u != 'C' && typeg_u != 'R') {
        PyErr_SetString(PyExc_ValueError, "TYPEG must be 'D', 'C', or 'R'");
        return NULL;
    }
    if (k < 0) {
        PyErr_SetString(PyExc_ValueError, "k must be non-negative");
        return NULL;
    }
    if (p < k) {
        PyErr_SetString(PyExc_ValueError, "p must be >= k");
        return NULL;
    }
    if (q < 0 || (lrdef && q < k)) {
        PyErr_SetString(PyExc_ValueError, "Invalid q");
        return NULL;
    }

    PyArrayObject *a1_array = (PyArrayObject*)PyArray_FROM_OTF(
        a1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a1_array) return NULL;

    PyArrayObject *a2_array = (PyArrayObject*)PyArray_FROM_OTF(
        a2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a2_array) {
        Py_DECREF(a1_array);
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) {
        Py_DECREF(a1_array);
        Py_DECREF(a2_array);
        return NULL;
    }

    i32 lda1 = k > 0 ? (i32)PyArray_DIM(a1_array, 0) : 1;
    i32 lda2 = (i32)PyArray_DIM(a2_array, 0);
    i32 ldb = (i32)PyArray_DIM(b_array, 0);

    f64 *a1_data = (f64*)PyArray_DATA(a1_array);
    f64 *a2_data = (f64*)PyArray_DATA(a2_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    i32 rnk = 0;
    i32 *ipvt = (i32*)calloc(k > 0 ? k : 1, sizeof(i32));
    if (!ipvt) {
        Py_DECREF(a1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 col2 = p - k;
    i32 lcs;
    if (lrdef) {
        if (col2 == 0) {
            lcs = 2 * k + k;
        } else {
            lcs = 4 * k + k;
        }
    } else if (lcol) {
        if (col2 > 0) {
            lcs = 5 * k + (k < q ? k : q);
        } else {
            lcs = 2 * k + (k < q ? k : q);
        }
    } else {
        if (col2 > 0) {
            lcs = 5 * k + (k < q ? k : q);
        } else {
            lcs = 2 * k + (k < q ? k : q);
        }
    }
    if (lcs < 1) lcs = 1;

    f64 *cs = (f64*)calloc(lcs, sizeof(f64));
    if (!cs) {
        free(ipvt);
        Py_DECREF(a1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 ldwork;
    if (lrdef) {
        ldwork = 4 * k > 1 ? 4 * k : 1;
    } else {
        i32 nbk = nb * k;
        ldwork = nbk > k ? nbk : k;
        ldwork = ldwork > 1 ? ldwork : 1;
    }

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (!dwork) {
        free(ipvt);
        free(cs);
        Py_DECREF(a1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info;
    mb02cu(typeg_str, k, p, q, nb, a1_data, lda1, a2_data, lda2,
           b_data, ldb, &rnk, ipvt, cs, tol, dwork, ldwork, &info);

    free(dwork);

    if (info < 0) {
        free(ipvt);
        free(cs);
        Py_DECREF(a1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b_array);
        PyErr_Format(PyExc_ValueError, "mb02cu: parameter %d is invalid", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(a1_array);
    PyArray_ResolveWritebackIfCopy(a2_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    npy_intp ipvt_dims[1] = {k > 0 ? k : 1};
    PyArrayObject *ipvt_array = (PyArrayObject*)PyArray_SimpleNew(1, ipvt_dims, NPY_INT32);
    if (!ipvt_array) {
        free(ipvt);
        free(cs);
        Py_DECREF(a1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b_array);
        PyErr_NoMemory();
        return NULL;
    }
    memcpy(PyArray_DATA(ipvt_array), ipvt, (k > 0 ? k : 1) * sizeof(i32));
    free(ipvt);

    npy_intp cs_dims[1] = {lcs};
    PyArrayObject *cs_out = (PyArrayObject*)PyArray_SimpleNew(1, cs_dims, NPY_DOUBLE);
    if (!cs_out) {
        free(cs);
        Py_DECREF(a1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b_array);
        Py_DECREF(ipvt_array);
        PyErr_NoMemory();
        return NULL;
    }
    f64 *cs_out_data = (f64*)PyArray_DATA(cs_out);
    memcpy(cs_out_data, cs, (size_t)lcs * sizeof(f64));
    free(cs);

    PyObject *result = Py_BuildValue("OOOiOOi", a1_array, a2_array, b_array, rnk, ipvt_array, cs_out, info);

    Py_DECREF(a1_array);
    Py_DECREF(a2_array);
    Py_DECREF(b_array);
    Py_DECREF(ipvt_array);
    Py_DECREF(cs_out);

    return result;
}


/* Python wrapper for mb02cv - Apply MB02CU transformations to generator columns/rows */
PyObject* py_mb02cv(PyObject* self, PyObject* args) {
    const char *typeg_str, *strucg_str;
    i32 k, n, p, q, nb, rnk;
    PyObject *a1_obj, *a2_obj, *b_obj, *f1_obj, *f2_obj, *g_obj, *cs_obj;

    if (!PyArg_ParseTuple(args, "ssiiiiiiOOOOOOO", &typeg_str, &strucg_str,
                          &k, &n, &p, &q, &nb, &rnk,
                          &a1_obj, &a2_obj, &b_obj, &f1_obj, &f2_obj, &g_obj, &cs_obj)) {
        return NULL;
    }

    PyArrayObject *a1_array = (PyArrayObject*)PyArray_FROM_OTF(
        a1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a1_array) return NULL;

    PyArrayObject *a2_array = (PyArrayObject*)PyArray_FROM_OTF(
        a2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a2_array) {
        Py_DECREF(a1_array);
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) {
        Py_DECREF(a1_array);
        Py_DECREF(a2_array);
        return NULL;
    }

    PyArrayObject *f1_array = (PyArrayObject*)PyArray_FROM_OTF(
        f1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!f1_array) {
        Py_DECREF(a1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *f2_array = (PyArrayObject*)PyArray_FROM_OTF(
        f2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!f2_array) {
        Py_DECREF(a1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b_array);
        Py_DECREF(f1_array);
        return NULL;
    }

    PyArrayObject *g_array = (PyArrayObject*)PyArray_FROM_OTF(
        g_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!g_array) {
        Py_DECREF(a1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b_array);
        Py_DECREF(f1_array);
        Py_DECREF(f2_array);
        return NULL;
    }

    PyArrayObject *cs_array = (PyArrayObject*)PyArray_FROM_OTF(
        cs_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!cs_array) {
        Py_DECREF(a1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b_array);
        Py_DECREF(f1_array);
        Py_DECREF(f2_array);
        Py_DECREF(g_array);
        return NULL;
    }

    i32 lda1 = (i32)PyArray_DIM(a1_array, 0);
    i32 lda2 = (i32)PyArray_DIM(a2_array, 0);
    i32 ldb = (i32)PyArray_DIM(b_array, 0);
    i32 ldf1 = (i32)PyArray_DIM(f1_array, 0);
    i32 ldf2 = (i32)PyArray_DIM(f2_array, 0);
    i32 ldg = (i32)PyArray_DIM(g_array, 0);

    f64 *a1_data = (f64*)PyArray_DATA(a1_array);
    f64 *a2_data = (f64*)PyArray_DATA(a2_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *f1_data = (f64*)PyArray_DATA(f1_array);
    f64 *f2_data = (f64*)PyArray_DATA(f2_array);
    f64 *g_data = (f64*)PyArray_DATA(g_array);
    const f64 *cs_data = (const f64*)PyArray_DATA(cs_array);

    char typeg_u = typeg_str[0];
    if (typeg_u >= 'a' && typeg_u <= 'z') typeg_u -= 32;
    bool lrdef = (typeg_u == 'D');

    i32 ldwork;
    if (lrdef) {
        ldwork = (n > 1) ? n : 1;
    } else {
        if (nb >= 1) {
            ldwork = (n + k) * nb;
            if (ldwork < 1) ldwork = 1;
        } else {
            ldwork = (n > 1) ? n : 1;
        }
    }

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (!dwork) {
        Py_DECREF(a1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b_array);
        Py_DECREF(f1_array);
        Py_DECREF(f2_array);
        Py_DECREF(g_array);
        Py_DECREF(cs_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info;
    mb02cv(typeg_str, strucg_str, k, n, p, q, nb, rnk,
           a1_data, lda1, a2_data, lda2, b_data, ldb,
           f1_data, ldf1, f2_data, ldf2, g_data, ldg,
           cs_data, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a1_array);
    PyArray_ResolveWritebackIfCopy(a2_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(f1_array);
    PyArray_ResolveWritebackIfCopy(f2_array);
    PyArray_ResolveWritebackIfCopy(g_array);

    PyObject *result = Py_BuildValue("OOOi", f1_array, f2_array, g_array, info);

    Py_DECREF(a1_array);
    Py_DECREF(a2_array);
    Py_DECREF(b_array);
    Py_DECREF(f1_array);
    Py_DECREF(f2_array);
    Py_DECREF(g_array);
    Py_DECREF(cs_array);

    return result;
}


/* Python wrapper for mb02od - Triangular matrix equation solver with condition estimation */
PyObject* py_mb02od(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"side", "uplo", "trans", "diag", "norm",
                             "alpha", "a", "b", "tol", NULL};

    const char *side_str, *uplo_str, *trans_str, *diag_str, *norm_str;
    f64 alpha;
    f64 tol = 0.0;
    PyObject *a_obj, *b_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssssdOO|d", kwlist,
                                     &side_str, &uplo_str, &trans_str,
                                     &diag_str, &norm_str, &alpha,
                                     &a_obj, &b_obj, &tol)) {
        return NULL;
    }

    char side = toupper((unsigned char)side_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);
    char trans = toupper((unsigned char)trans_str[0]);
    char diag = toupper((unsigned char)diag_str[0]);
    char norm_c = toupper((unsigned char)norm_str[0]);

    if (side != 'L' && side != 'R') {
        PyErr_SetString(PyExc_ValueError, "side must be 'L' or 'R'");
        return NULL;
    }
    if (uplo != 'U' && uplo != 'L') {
        PyErr_SetString(PyExc_ValueError, "uplo must be 'U' or 'L'");
        return NULL;
    }
    if (trans != 'N' && trans != 'T' && trans != 'C') {
        PyErr_SetString(PyExc_ValueError, "trans must be 'N', 'T', or 'C'");
        return NULL;
    }
    if (diag != 'U' && diag != 'N') {
        PyErr_SetString(PyExc_ValueError, "diag must be 'U' or 'N'");
        return NULL;
    }
    if (norm_c != '1' && norm_c != 'O' && norm_c != 'I') {
        PyErr_SetString(PyExc_ValueError, "norm must be '1', 'O', or 'I'");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 m = (i32)PyArray_DIM(b_array, 0);
    i32 n = (PyArray_NDIM(b_array) > 1) ? (i32)PyArray_DIM(b_array, 1) : 1;
    i32 nrowa = (side == 'L') ? m : n;

    i32 lda = (i32)PyArray_DIM(a_array, 0);
    i32 ldb = m > 0 ? m : 1;

    const f64 *a_data = (const f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    i32 *iwork = (i32*)malloc((nrowa > 0 ? nrowa : 1) * sizeof(i32));
    f64 *dwork = (f64*)malloc((nrowa > 0 ? 3 * nrowa : 1) * sizeof(f64));
    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 rcond = 0.0;
    i32 info = 0;

    mb02od(side_str, uplo_str, trans_str, diag_str, norm_str,
           m, n, alpha, a_data, lda, b_data, ldb,
           &rcond, tol, iwork, dwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(b_array);

    if (info < 0) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        PyErr_Format(PyExc_ValueError, "mb02od: parameter %d is invalid", -info);
        return NULL;
    }

    PyObject *result = Py_BuildValue("Odi", b_array, rcond, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);

    return result;
}


/* Python wrapper for mb02cd - Cholesky factorization of block Toeplitz matrix */
PyObject* py_mb02cd(PyObject* self, PyObject* args) {
    const char *job_str, *typet_str;
    i32 k, n;
    PyObject *t_obj;

    if (!PyArg_ParseTuple(args, "ssiiO", &job_str, &typet_str, &k, &n, &t_obj)) {
        return NULL;
    }

    char job_u = job_str[0];
    if (job_u >= 'a' && job_u <= 'z') job_u -= 32;
    char typet_u = typet_str[0];
    if (typet_u >= 'a' && typet_u <= 'z') typet_u -= 32;

    bool isrow = (typet_u == 'R');
    bool compl = (job_u == 'L') || (job_u == 'A');
    bool compg = (job_u == 'G') || (job_u == 'R') || compl;
    bool compr = (job_u == 'R') || (job_u == 'A') || (job_u == 'O');

    if (k < 0 || n < 0) {
        PyErr_SetString(PyExc_ValueError, "k and n must be non-negative");
        return NULL;
    }

    PyArrayObject *t_array = (PyArrayObject*)PyArray_FROM_OTF(
        t_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!t_array) return NULL;

    i32 ldt = (i32)PyArray_DIM(t_array, 0);
    f64 *t_data = (f64*)PyArray_DATA(t_array);

    i32 m = n * k;
    i32 twok = 2 * k;

    i32 ldg, ldr, ldl;
    if (compg) {
        ldg = isrow ? twok : m;
        ldg = ldg > 1 ? ldg : 1;
    } else {
        ldg = 1;
    }

    if (compr) {
        ldr = m > 1 ? m : 1;
    } else {
        ldr = 1;
    }

    if (compl) {
        ldl = m > 1 ? m : 1;
    } else {
        ldl = 1;
    }

    i32 g_cols = isrow ? m : twok;
    if (!compg) g_cols = 1;
    i32 g_size = ldg * g_cols;
    if (g_size < 1) g_size = 1;

    i32 r_size = ldr * m;
    if (r_size < 1) r_size = 1;

    i32 l_size = ldl * m;
    if (l_size < 1) l_size = 1;

    i32 lcs = 3 * (n - 1) * k;
    if (lcs < 1) lcs = 1;

    i32 ldwork = (n > 1) ? (n - 1) * k : 1;
    if (ldwork < 1) ldwork = 1;

    f64 *g_data = (f64*)calloc(g_size, sizeof(f64));
    f64 *r_data = (f64*)calloc(r_size, sizeof(f64));
    f64 *l_data = (f64*)calloc(l_size, sizeof(f64));
    f64 *cs_data = (f64*)calloc(lcs, sizeof(f64));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!g_data || !r_data || !l_data || !cs_data || !dwork) {
        free(g_data);
        free(r_data);
        free(l_data);
        free(cs_data);
        free(dwork);
        Py_DECREF(t_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info;
    mb02cd(job_str, typet_str, k, n, t_data, ldt, g_data, ldg, r_data, ldr,
           l_data, ldl, cs_data, lcs, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(t_array);
    Py_DECREF(t_array);

    if (info < 0) {
        free(g_data);
        free(r_data);
        free(l_data);
        free(cs_data);
        PyErr_Format(PyExc_ValueError, "mb02cd: parameter %d is invalid", -info);
        return NULL;
    }

    PyObject *g_out, *r_out, *l_out, *cs_out;

    if (compg) {
        npy_intp g_dims[2] = {ldg, g_cols};
        npy_intp g_strides[2] = {sizeof(f64), ldg * sizeof(f64)};
        g_out = PyArray_New(&PyArray_Type, 2, g_dims, NPY_DOUBLE,
                            g_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!g_out) {
            free(g_data);
            free(r_data);
            free(l_data);
            free(cs_data);
            return NULL;
        }
        f64 *g_out_ptr = (f64*)PyArray_DATA((PyArrayObject*)g_out);
        memcpy(g_out_ptr, g_data, (size_t)ldg * (size_t)g_cols * sizeof(f64));
        free(g_data);
    } else {
        free(g_data);
        Py_INCREF(Py_None);
        g_out = Py_None;
    }

    if (compr) {
        npy_intp r_dims[2] = {m, m};
        npy_intp r_strides[2] = {sizeof(f64), ldr * sizeof(f64)};
        r_out = PyArray_New(&PyArray_Type, 2, r_dims, NPY_DOUBLE,
                            r_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!r_out) {
            free(r_data);
            free(l_data);
            free(cs_data);
            Py_DECREF(g_out);
            return NULL;
        }
        f64 *r_out_ptr = (f64*)PyArray_DATA((PyArrayObject*)r_out);
        memcpy(r_out_ptr, r_data, (size_t)ldr * (size_t)m * sizeof(f64));
        free(r_data);
    } else {
        free(r_data);
        Py_INCREF(Py_None);
        r_out = Py_None;
    }

    if (compl) {
        npy_intp l_dims[2] = {m, m};
        npy_intp l_strides[2] = {sizeof(f64), ldl * sizeof(f64)};
        l_out = PyArray_New(&PyArray_Type, 2, l_dims, NPY_DOUBLE,
                            l_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!l_out) {
            free(l_data);
            free(cs_data);
            Py_DECREF(g_out);
            Py_DECREF(r_out);
            return NULL;
        }
        f64 *l_out_ptr = (f64*)PyArray_DATA((PyArrayObject*)l_out);
        memcpy(l_out_ptr, l_data, (size_t)ldl * (size_t)m * sizeof(f64));
        free(l_data);
    } else {
        free(l_data);
        Py_INCREF(Py_None);
        l_out = Py_None;
    }

    npy_intp cs_dims[1] = {lcs};
    cs_out = PyArray_SimpleNew(1, cs_dims, NPY_DOUBLE);
    if (!cs_out) {
        free(cs_data);
        Py_DECREF(g_out);
        Py_DECREF(r_out);
        Py_DECREF(l_out);
        return NULL;
    }
    f64 *cs_out_ptr = (f64*)PyArray_DATA((PyArrayObject*)cs_out);
    memcpy(cs_out_ptr, cs_data, (size_t)lcs * sizeof(f64));
    free(cs_data);

    PyObject *result = Py_BuildValue("OOOOi", g_out, r_out, l_out, cs_out, info);

    Py_DECREF(g_out);
    Py_DECREF(r_out);
    Py_DECREF(l_out);
    Py_DECREF(cs_out);

    return result;
}


/* Python wrapper for mb02dd - Update Cholesky factorization of block Toeplitz matrix */
PyObject* py_mb02dd(PyObject* self, PyObject* args) {
    const char *job_str, *typet_str;
    i32 k, m, n;
    PyObject *ta_obj, *t_obj, *g_obj, *r_obj, *cs_obj;

    if (!PyArg_ParseTuple(args, "ssiiiOOOOO", &job_str, &typet_str, &k, &m, &n,
                          &ta_obj, &t_obj, &g_obj, &r_obj, &cs_obj)) {
        return NULL;
    }

    char job_u = job_str[0];
    if (job_u >= 'a' && job_u <= 'z') job_u -= 32;
    char typet_u = typet_str[0];
    if (typet_u >= 'a' && typet_u <= 'z') typet_u -= 32;

    bool isrow = (typet_u == 'R');
    bool compl_flag = (job_u == 'A');
    bool compg = (job_u == 'R') || compl_flag;

    if (k < 0 || m < 0 || n < 0) {
        PyErr_SetString(PyExc_ValueError, "k, m, and n must be non-negative");
        return NULL;
    }

    PyArrayObject *ta_array = (PyArrayObject*)PyArray_FROM_OTF(
        ta_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *t_array = (PyArrayObject*)PyArray_FROM_OTF(
        t_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *g_array = (PyArrayObject*)PyArray_FROM_OTF(
        g_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *r_array = (PyArrayObject*)PyArray_FROM_OTF(
        r_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *cs_array = (PyArrayObject*)PyArray_FROM_OTF(
        cs_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!ta_array || !t_array || !g_array || !r_array || !cs_array) {
        Py_XDECREF(ta_array);
        Py_XDECREF(t_array);
        Py_XDECREF(g_array);
        Py_XDECREF(r_array);
        Py_XDECREF(cs_array);
        return NULL;
    }

    i32 ldta = (i32)PyArray_DIM(ta_array, 0);
    i32 ldt = (i32)PyArray_DIM(t_array, 0);
    i32 ldg = (i32)PyArray_DIM(g_array, 0);
    i32 ldr = (i32)PyArray_DIM(r_array, 0);
    i32 lcs = (i32)PyArray_SIZE(cs_array);

    f64 *ta_data = (f64*)PyArray_DATA(ta_array);
    f64 *t_data = (f64*)PyArray_DATA(t_array);
    f64 *g_data = (f64*)PyArray_DATA(g_array);
    f64 *r_data = (f64*)PyArray_DATA(r_array);
    f64 *cs_data = (f64*)PyArray_DATA(cs_array);

    i32 s = (n + m) * k;
    i32 twok = 2 * k;

    i32 ldl;
    if (compl_flag) {
        ldl = isrow ? (m * k > 1 ? m * k : 1) : (s > 1 ? s : 1);
    } else {
        ldl = 1;
    }

    i32 l_cols = isrow ? s : (m * k > 1 ? m * k : 1);
    if (!compl_flag) l_cols = 1;

    i32 l_size = ldl * l_cols;
    if (l_size < 1) l_size = 1;

    i32 ldwork = (n + m > 1) ? (n + m - 1) * k : 1;
    if (ldwork < 1) ldwork = 1;

    f64 *l_data = (f64*)calloc(l_size, sizeof(f64));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!l_data || !dwork) {
        free(l_data);
        free(dwork);
        Py_DECREF(ta_array);
        Py_DECREF(t_array);
        Py_DECREF(g_array);
        Py_DECREF(r_array);
        Py_DECREF(cs_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info;
    mb02dd(job_str, typet_str, k, m, n, ta_data, ldta, t_data, ldt,
           g_data, ldg, r_data, ldr, l_data, ldl, cs_data, lcs, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(ta_array);
    PyArray_ResolveWritebackIfCopy(t_array);
    PyArray_ResolveWritebackIfCopy(g_array);
    PyArray_ResolveWritebackIfCopy(r_array);
    PyArray_ResolveWritebackIfCopy(cs_array);

    if (info < 0) {
        free(l_data);
        Py_DECREF(ta_array);
        Py_DECREF(t_array);
        Py_DECREF(g_array);
        Py_DECREF(r_array);
        Py_DECREF(cs_array);
        PyErr_Format(PyExc_ValueError, "mb02dd: parameter %d is invalid", -info);
        return NULL;
    }

    PyObject *l_out;
    if (compl_flag) {
        npy_intp l_dims[2];
        if (isrow) {
            l_dims[0] = m * k;
            l_dims[1] = s;
        } else {
            l_dims[0] = s;
            l_dims[1] = m * k;
        }
        npy_intp l_strides[2] = {sizeof(f64), ldl * sizeof(f64)};
        l_out = PyArray_New(&PyArray_Type, 2, l_dims, NPY_DOUBLE,
                            l_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!l_out) {
            free(l_data);
            Py_DECREF(ta_array);
            Py_DECREF(t_array);
            Py_DECREF(g_array);
            Py_DECREF(r_array);
            Py_DECREF(cs_array);
            return NULL;
        }
        f64 *l_out_ptr = (f64*)PyArray_DATA((PyArrayObject*)l_out);
        memcpy(l_out_ptr, l_data, (size_t)ldl * (size_t)l_cols * sizeof(f64));
        free(l_data);
    } else {
        free(l_data);
        Py_INCREF(Py_None);
        l_out = Py_None;
    }

    npy_intp r_dims[2];
    i32 r_out_rows, r_out_cols;
    if (isrow) {
        r_out_rows = s;
        r_out_cols = m * k;
        r_dims[0] = r_out_rows;
        r_dims[1] = r_out_cols;
    } else {
        r_out_rows = m * k;
        r_out_cols = s;
        r_dims[0] = r_out_rows;
        r_dims[1] = r_out_cols;
    }

    npy_intp r_strides[2] = {sizeof(f64), r_out_rows * sizeof(f64)};
    PyObject *r_out = PyArray_New(&PyArray_Type, 2, r_dims, NPY_DOUBLE,
                                  r_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!r_out) {
        Py_DECREF(l_out);
        Py_DECREF(ta_array);
        Py_DECREF(t_array);
        Py_DECREF(g_array);
        Py_DECREF(r_array);
        Py_DECREF(cs_array);
        return NULL;
    }
    f64 *r_out_ptr = (f64*)PyArray_DATA((PyArrayObject*)r_out);
    for (i32 j = 0; j < r_out_cols; j++) {
        for (i32 i = 0; i < r_out_rows; i++) {
            r_out_ptr[i + j * r_out_rows] = r_data[i + j * ldr];
        }
    }

    PyObject *g_out;
    if (compg) {
        npy_intp g_dims[2];
        if (isrow) {
            g_dims[0] = twok;
            g_dims[1] = s;
        } else {
            g_dims[0] = s;
            g_dims[1] = twok;
        }
        npy_intp g_strides[2] = {sizeof(f64), ldg * sizeof(f64)};
        g_out = PyArray_New(&PyArray_Type, 2, g_dims, NPY_DOUBLE,
                            g_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!g_out) {
            Py_DECREF(l_out);
            Py_DECREF(r_out);
            Py_DECREF(ta_array);
            Py_DECREF(t_array);
            Py_DECREF(g_array);
            Py_DECREF(r_array);
            Py_DECREF(cs_array);
            return NULL;
        }

        f64 *g_out_data = (f64*)PyArray_DATA((PyArrayObject*)g_out);
        i32 g_rows = isrow ? twok : s;
        i32 g_cols = isrow ? s : twok;
        for (i32 j = 0; j < g_cols; j++) {
            for (i32 i = 0; i < g_rows; i++) {
                g_out_data[i + j * g_rows] = g_data[i + j * ldg];
            }
        }
    } else {
        Py_INCREF(Py_None);
        g_out = Py_None;
    }

    Py_DECREF(ta_array);
    Py_DECREF(t_array);
    Py_DECREF(g_array);
    Py_DECREF(r_array);
    Py_DECREF(cs_array);

    PyObject *ta_out;
    PyArrayObject *ta_new = (PyArrayObject*)PyArray_FROM_OTF(
        ta_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    ta_out = (PyObject*)ta_new;

    PyObject *cs_out;
    npy_intp cs_dims[1] = {lcs};
    cs_out = PyArray_SimpleNew(1, cs_dims, NPY_DOUBLE);
    memcpy(PyArray_DATA((PyArrayObject*)cs_out), cs_data, lcs * sizeof(f64));

    cs_array = (PyArrayObject*)PyArray_FROM_OTF(cs_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    memcpy(PyArray_DATA((PyArrayObject*)cs_out), PyArray_DATA(cs_array), lcs * sizeof(f64));
    Py_DECREF(cs_array);

    PyObject *result = Py_BuildValue("OOOOOi", ta_out, g_out, r_out, l_out, cs_out, info);

    Py_DECREF(ta_out);
    Py_DECREF(g_out);
    Py_DECREF(r_out);
    Py_DECREF(l_out);
    Py_DECREF(cs_out);

    return result;
}


PyObject* py_mb02fd(PyObject* self, PyObject* args) {
    char* typet_str;
    i32 k, n, p, s;
    PyObject *t_obj;
    PyArrayObject *t_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "siiiiO", &typet_str, &k, &n, &p, &s, &t_obj)) {
        return NULL;
    }

    char typet = toupper((unsigned char)typet_str[0]);
    bool isrow = (typet == 'R');

    if (k < 0 || n < 0 || p < 0 || p > n || s < 0 || s > (n - p)) {
        PyErr_SetString(PyExc_ValueError, "Invalid parameters");
        return NULL;
    }

    t_array = (PyArrayObject*)PyArray_FROM_OTF(t_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (t_array == NULL) return NULL;

    i32 ldt = (i32)PyArray_DIM(t_array, 0);
    f64 *t_data = (f64*)PyArray_DATA(t_array);

    i32 ldr;
    i32 r_rows, r_cols;
    if (isrow) {
        if (p == 0) {
            ldr = s * k > 1 ? s * k : 1;
            r_rows = s * k;
        } else {
            ldr = (s + 1) * k > 1 ? (s + 1) * k : 1;
            r_rows = (s + 1) * k;
        }
        r_cols = n * k;
    } else {
        if (p == 0) {
            ldr = n * k > 1 ? n * k : 1;
            r_rows = n * k;
        } else {
            ldr = (n - p + 1) * k > 1 ? (n - p + 1) * k : 1;
            r_rows = (n - p + 1) * k;
        }
        r_cols = s * k;
    }

    i32 ldwork;
    if (p == 0) {
        ldwork = (n + 1) * k;
    } else {
        ldwork = (n - p + 2) * k;
    }
    i32 fourk = 4 * k;
    if (ldwork < fourk) ldwork = fourk;
    if (ldwork < 1) ldwork = 1;

    f64 *r_data = (f64*)calloc(ldr * r_cols > 0 ? ldr * r_cols : 1, sizeof(f64));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (r_data == NULL || dwork == NULL) {
        free(r_data);
        free(dwork);
        Py_DECREF(t_array);
        return PyErr_NoMemory();
    }

    mb02fd(typet_str, k, n, p, s, t_data, ldt, r_data, ldr, dwork, ldwork, &info);

    free(dwork);
    PyArray_ResolveWritebackIfCopy(t_array);
    Py_DECREF(t_array);

    if (info < 0 && info != -11) {
        free(r_data);
        PyErr_Format(PyExc_ValueError, "mb02fd: parameter %d is invalid", -info);
        return NULL;
    }

    npy_intp r_dims[2] = {r_rows, r_cols};
    npy_intp r_strides[2] = {sizeof(f64), ldr * sizeof(f64)};
    PyObject *r_out = PyArray_New(&PyArray_Type, 2, r_dims, NPY_DOUBLE,
                                  r_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (r_out == NULL) {
        free(r_data);
        return PyErr_NoMemory();
    }
    f64 *r_out_ptr = (f64*)PyArray_DATA((PyArrayObject*)r_out);
    if (r_rows > 0 && r_cols > 0) {
        memcpy(r_out_ptr, r_data, (size_t)ldr * (size_t)r_cols * sizeof(f64));
    }
    free(r_data);

    PyObject *result = Py_BuildValue("Oi", r_out, info);
    Py_DECREF(r_out);

    return result;
}


PyObject* py_mb02gd(PyObject* self, PyObject* args) {
    char* typet_str;
    char* triu_str;
    i32 k, n, nl, p, s;
    PyObject *t_obj;
    PyArrayObject *t_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssiiiiiO", &typet_str, &triu_str, &k, &n, &nl, &p, &s, &t_obj)) {
        return NULL;
    }

    char typet = toupper((unsigned char)typet_str[0]);
    char triu = toupper((unsigned char)triu_str[0]);
    bool isrow = (typet == 'R');
    bool ltri = (triu == 'T');

    if (typet != 'R' && typet != 'C') {
        PyErr_SetString(PyExc_ValueError, "typet must be 'R' or 'C'");
        return NULL;
    }
    if (triu != 'N' && triu != 'T') {
        PyErr_SetString(PyExc_ValueError, "triu must be 'N' or 'T'");
        return NULL;
    }
    if (k < 0) {
        PyErr_SetString(PyExc_ValueError, "k must be >= 0");
        return NULL;
    }
    if ((ltri && n < 2) || (!ltri && n < 1)) {
        PyErr_SetString(PyExc_ValueError, "n must be >= 2 for triu='T', >= 1 for triu='N'");
        return NULL;
    }
    if (nl >= n || (ltri && nl < 1) || (!ltri && nl < 0)) {
        PyErr_SetString(PyExc_ValueError, "Invalid nl");
        return NULL;
    }
    if (p < 0 || p > n) {
        PyErr_SetString(PyExc_ValueError, "p must be in [0, n]");
        return NULL;
    }
    if (s < 0 || s > (n - p)) {
        PyErr_SetString(PyExc_ValueError, "s must be in [0, n-p]");
        return NULL;
    }

    t_array = (PyArrayObject*)PyArray_FROM_OTF(t_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (t_array == NULL) return NULL;

    i32 ldt = (i32)PyArray_DIM(t_array, 0);
    f64 *t_data = (f64*)PyArray_DATA(t_array);

    i32 lenr = (nl + 1) * k;
    i32 sizr = ltri ? (nl * k + 1) : lenr;
    i32 ldrb = sizr;

    i32 rb_rows = sizr;
    i32 rb_cols;
    if (isrow) {
        i32 min_val = p + nl + s;
        if (min_val > n) min_val = n;
        rb_cols = min_val * k;
    } else {
        i32 min_val = p + s;
        if (min_val > n) min_val = n;
        rb_cols = min_val * k;
    }

    i32 ldwork = 1 + (lenr + nl) * k;
    i32 fourk = 4 * k;
    if (ldwork < fourk) ldwork = fourk;
    if (ldwork < 1) ldwork = 1;
    ldwork += lenr * k;

    f64 *rb_data = (f64*)calloc(ldrb * rb_cols > 0 ? ldrb * rb_cols : 1, sizeof(f64));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (rb_data == NULL || dwork == NULL) {
        free(rb_data);
        free(dwork);
        Py_DECREF(t_array);
        return PyErr_NoMemory();
    }

    mb02gd(typet_str, triu_str, k, n, nl, p, s, t_data, ldt, rb_data, ldrb,
           dwork, ldwork, &info);

    free(dwork);
    PyArray_ResolveWritebackIfCopy(t_array);
    Py_DECREF(t_array);

    if (info < 0 && info != -13) {
        free(rb_data);
        PyErr_Format(PyExc_ValueError, "mb02gd: parameter %d is invalid", -info);
        return NULL;
    }

    npy_intp rb_dims[2] = {rb_rows, rb_cols};
    npy_intp rb_strides[2] = {sizeof(f64), ldrb * sizeof(f64)};
    PyObject *rb_out = PyArray_New(&PyArray_Type, 2, rb_dims, NPY_DOUBLE,
                                   rb_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (rb_out == NULL) {
        free(rb_data);
        return PyErr_NoMemory();
    }
    f64 *rb_out_ptr = (f64*)PyArray_DATA((PyArrayObject*)rb_out);
    memcpy(rb_out_ptr, rb_data, (size_t)ldrb * (size_t)rb_cols * sizeof(f64));
    free(rb_data);

    PyObject *result = Py_BuildValue("Oi", rb_out, info);
    Py_DECREF(rb_out);

    return result;
}


PyObject* py_mb02hd(PyObject* self, PyObject* args) {
    char* triu_str;
    i32 k, l, m, ml, n, nu, p, s;
    PyObject *tc_obj, *tr_obj;
    PyArrayObject *tc_array, *tr_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "siiiiiiiiOO", &triu_str, &k, &l, &m, &ml, &n, &nu, &p, &s, &tc_obj, &tr_obj)) {
        return NULL;
    }

    char triu = toupper((unsigned char)triu_str[0]);
    bool ltri = (triu == 'T');

    if (triu != 'N' && triu != 'T') {
        PyErr_SetString(PyExc_ValueError, "triu must be 'N' or 'T'");
        return NULL;
    }
    if (k < 0) {
        PyErr_SetString(PyExc_ValueError, "k must be >= 0");
        return NULL;
    }
    if (l < 0) {
        PyErr_SetString(PyExc_ValueError, "l must be >= 0");
        return NULL;
    }
    if (m < 1) {
        PyErr_SetString(PyExc_ValueError, "m must be >= 1");
        return NULL;
    }
    if (n < 1) {
        PyErr_SetString(PyExc_ValueError, "n must be >= 1");
        return NULL;
    }
    if (ml < 0 || ml >= m) {
        PyErr_SetString(PyExc_ValueError, "ml must be in [0, m-1]");
        return NULL;
    }
    if (nu < 0 || nu >= n) {
        PyErr_SetString(PyExc_ValueError, "nu must be in [0, n-1]");
        return NULL;
    }
    if (p < 0) {
        PyErr_SetString(PyExc_ValueError, "p must be >= 0");
        return NULL;
    }
    if (s < 0) {
        PyErr_SetString(PyExc_ValueError, "s must be >= 0");
        return NULL;
    }

    tc_array = (PyArrayObject*)PyArray_FROM_OTF(tc_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (tc_array == NULL) return NULL;

    tr_array = (PyArrayObject*)PyArray_FROM_OTF(tr_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (tr_array == NULL) {
        Py_DECREF(tc_array);
        return NULL;
    }

    i32 ldtc = (i32)PyArray_DIM(tc_array, 0);
    i32 ldtr = (i32)PyArray_DIM(tr_array, 0);
    f64 *tc_data = (f64*)PyArray_DATA(tc_array);
    f64 *tr_data = (f64*)PyArray_DATA(tr_array);

    i32 x = ml + nu + 1;
    if (x > n) x = n;
    i32 lenr = x * l;
    i32 sizr;
    if (ltri) {
        sizr = (ml + nu) * l + 1;
        if (sizr > n * l) sizr = n * l;
    } else {
        sizr = lenr;
    }
    i32 ldrb = sizr > 1 ? sizr : 1;

    i32 mk = m * k;
    i32 nl_val = n * l;
    i32 min_mk_nl = (mk < nl_val) ? mk : nl_val;

    i32 rb_cols = s * l;
    i32 max_cols = min_mk_nl - p * l;
    if (rb_cols > max_cols) rb_cols = max_cols;
    if (rb_cols < 0) rb_cols = 0;

    i32 ldwork;
    if (p == 0) {
        i32 opt1 = lenr * l + (2 * nu + 1) * l * k;
        i32 opt2 = 2 * lenr * (k + l) + (6 + x) * l;
        ldwork = 1 + ((opt1 > opt2) ? opt1 : opt2);
    } else {
        ldwork = 1 + 2 * lenr * (k + l) + (6 + x) * l;
    }
    if (ldwork < 1) ldwork = 1;

    f64 *rb_data = (f64*)calloc(ldrb * rb_cols > 0 ? ldrb * rb_cols : 1, sizeof(f64));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (rb_data == NULL || dwork == NULL) {
        free(rb_data);
        free(dwork);
        Py_DECREF(tc_array);
        Py_DECREF(tr_array);
        return PyErr_NoMemory();
    }

    mb02hd(triu_str, k, l, m, ml, n, nu, p, s, tc_data, ldtc, tr_data, ldtr,
           rb_data, ldrb, dwork, ldwork, &info);

    free(dwork);
    PyArray_ResolveWritebackIfCopy(tc_array);
    PyArray_ResolveWritebackIfCopy(tr_array);
    Py_DECREF(tc_array);
    Py_DECREF(tr_array);

    if (info < 0) {
        free(rb_data);
        PyErr_Format(PyExc_ValueError, "mb02hd: parameter %d is invalid", -info);
        return NULL;
    }

    npy_intp rb_dims[2] = {ldrb, rb_cols};
    npy_intp rb_strides[2] = {sizeof(f64), ldrb * sizeof(f64)};
    PyObject *rb_out = PyArray_New(&PyArray_Type, 2, rb_dims, NPY_DOUBLE,
                                   rb_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (rb_out == NULL) {
        free(rb_data);
        return PyErr_NoMemory();
    }
    f64 *rb_out_ptr = (f64*)PyArray_DATA((PyArrayObject*)rb_out);
    memcpy(rb_out_ptr, rb_data, (size_t)ldrb * (size_t)rb_cols * sizeof(f64));
    free(rb_data);

    PyObject *result = Py_BuildValue("Oi", rb_out, info);
    Py_DECREF(rb_out);

    return result;
}


/* Python wrapper for mb02kd */
PyObject* py_mb02kd(PyObject* self, PyObject* args) {
    char *ldblk, *trans;
    i32 k, l, m, n, r_param;
    f64 alpha, beta;
    PyObject *tc_obj, *tr_obj, *b_obj, *c_obj = Py_None;
    PyArrayObject *tc_array, *tr_array, *b_array, *c_array = NULL;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssiiiiiddOOO|O", &ldblk, &trans, &k, &l, &m, &n,
                          &r_param, &alpha, &beta, &tc_obj, &tr_obj, &b_obj, &c_obj)) {
        return NULL;
    }

    tc_array = (PyArrayObject*)PyArray_FROM_OTF(tc_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (tc_array == NULL) return NULL;

    tr_array = (PyArrayObject*)PyArray_FROM_OTF(tr_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (tr_array == NULL) {
        Py_DECREF(tc_array);
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (b_array == NULL) {
        Py_DECREF(tc_array);
        Py_DECREF(tr_array);
        return NULL;
    }

    i32 ldtc = (i32)PyArray_DIM(tc_array, 0);
    i32 ldtr = (i32)PyArray_DIM(tr_array, 0);
    i32 ldb = (i32)PyArray_DIM(b_array, 0);

    char trans_c = (char)toupper((unsigned char)trans[0]);
    bool ltran = (trans_c == 'T') || (trans_c == 'C');

    i32 k_safe = k > 0 ? k : 0;
    i32 l_safe = l > 0 ? l : 0;
    i32 m_safe = m > 0 ? m : 0;
    i32 n_safe = n > 0 ? n : 0;
    i32 r_safe = r_param > 0 ? r_param : 0;

    i32 mk = m_safe * k_safe;
    i32 nl = n_safe * l_safe;
    i32 c_rows = ltran ? nl : mk;
    i32 ldc = c_rows > 0 ? c_rows : 1;
    i32 alloc_size = ldc * (r_safe > 0 ? r_safe : 1);

    f64 *c_data = (f64*)malloc(alloc_size * sizeof(f64));
    if (c_data == NULL) {
        Py_DECREF(tc_array);
        Py_DECREF(tr_array);
        Py_DECREF(b_array);
        return PyErr_NoMemory();
    }

    if (c_obj != Py_None && c_obj != NULL) {
        c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
        if (c_array == NULL) {
            free(c_data);
            Py_DECREF(tc_array);
            Py_DECREF(tr_array);
            Py_DECREF(b_array);
            return NULL;
        }
        memcpy(c_data, PyArray_DATA(c_array), c_rows * r_param * sizeof(f64));
        Py_DECREF(c_array);
    }

    i32 len = 1;
    while (len < m + n - 1) {
        len = len * 2;
    }
    i32 ldwork = len * (k * l + k * r_param + l * r_param + 1) + 1;
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        free(c_data);
        Py_DECREF(tc_array);
        Py_DECREF(tr_array);
        Py_DECREF(b_array);
        return PyErr_NoMemory();
    }

    f64 *tc_data = (f64*)PyArray_DATA(tc_array);
    f64 *tr_data = (f64*)PyArray_DATA(tr_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    mb02kd(ldblk, trans, k, l, m, n, r_param, alpha, beta, tc_data, ldtc,
           tr_data, ldtr, b_data, ldb, c_data, ldc, dwork, ldwork, &info);

    free(dwork);
    Py_DECREF(tc_array);
    Py_DECREF(tr_array);
    Py_DECREF(b_array);

    npy_intp c_dims[2] = {c_rows, r_safe};
    npy_intp c_strides[2] = {sizeof(f64), ldc * sizeof(f64)};
    PyObject *c_out = PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE,
                                  c_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (c_out == NULL) {
        free(c_data);
        return PyErr_NoMemory();
    }
    f64 *c_out_ptr = (f64*)PyArray_DATA((PyArrayObject*)c_out);
    if (c_rows > 0 && r_safe > 0) {
        memcpy(c_out_ptr, c_data, (size_t)ldc * (size_t)r_safe * sizeof(f64));
    }
    free(c_data);

    PyObject *result = Py_BuildValue("Oi", c_out, info);
    Py_DECREF(c_out);

    return result;
}


PyObject* py_mb02md(PyObject* self, PyObject* args, PyObject* kwargs) {
    char* job;
    i32 m, n, l;
    f64 tol_in;
    i32 rank_in = 0;
    PyObject *c_obj;
    PyArrayObject *c_array = NULL;

    static char* kwlist[] = {"job", "m", "n", "l", "c", "tol", "rank", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siiiOd|i", kwlist,
                                      &job, &m, &n, &l, &c_obj, &tol_in, &rank_in)) {
        return NULL;
    }

    char job_c = (char)toupper((unsigned char)job[0]);
    if (job_c != 'R' && job_c != 'T' && job_c != 'B' && job_c != 'N') {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'R', 'T', 'B', or 'N'");
        return NULL;
    }

    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "M must be non-negative");
        return NULL;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "N must be non-negative");
        return NULL;
    }
    if (l < 0) {
        PyErr_SetString(PyExc_ValueError, "L must be non-negative");
        return NULL;
    }

    PyArrayObject *c_input = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY);
    if (c_input == NULL) {
        return NULL;
    }

    i32 nl = n + l;
    i32 k = (m > nl) ? m : nl;
    i32 minmnl = (m < nl) ? m : nl;

    npy_intp *c_in_dims = PyArray_DIMS(c_input);
    i32 c_in_rows = (i32)c_in_dims[0];
    i32 c_in_cols = (i32)c_in_dims[1];
    i32 ldc = (1 > k) ? 1 : k;

    npy_intp c_dims[2] = {ldc, nl > 0 ? nl : 1};
    npy_intp c_strides[2] = {sizeof(f64), ldc * sizeof(f64)};
    c_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE,
                                          c_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (c_array == NULL) {
        Py_DECREF(c_input);
        return PyErr_NoMemory();
    }
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    memset(c_data, 0, (size_t)ldc * nl * sizeof(f64));

    f64 *c_in_data = (f64*)PyArray_DATA(c_input);
    i32 copy_rows = (c_in_rows < m) ? c_in_rows : m;
    i32 copy_cols = (c_in_cols < nl) ? c_in_cols : nl;
    for (i32 j = 0; j < copy_cols; j++) {
        memcpy(&c_data[j * ldc], &c_in_data[j * c_in_rows], copy_rows * sizeof(f64));
    }
    Py_DECREF(c_input);

    i32 s_len = minmnl > 0 ? minmnl : 1;
    f64 *s = (f64*)malloc(s_len * sizeof(f64));
    if (s == NULL) {
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    i32 ldx = n > 1 ? n : 1;
    i32 x_size = ldx * (l > 0 ? l : 1);
    f64 *x = (f64*)malloc(x_size * sizeof(f64));
    if (x == NULL) {
        free(s);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    i32 iwork_size = l > 0 ? l : 1;
    i32 *iwork = (i32*)malloc(iwork_size * sizeof(i32));
    if (iwork == NULL) {
        free(s);
        free(x);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    i32 ldwork;
    i32 ldw = (3 * minmnl + k > 5 * minmnl) ? (3 * minmnl + k) : (5 * minmnl);
    if (m >= nl) {
        ldwork = (2 > ldw) ? 2 : ldw;
    } else {
        i32 tmp1 = m * nl + ldw;
        i32 tmp2 = 3 * l;
        ldwork = (2 > tmp1) ? 2 : tmp1;
        ldwork = (ldwork > tmp2) ? ldwork : tmp2;
    }
    ldwork = ldwork > 1 ? ldwork : 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        free(s);
        free(x);
        free(iwork);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    i32 rank = rank_in;
    f64 tol = tol_in;
    i32 iwarn = 0;
    i32 info = 0;

    mb02md(job, m, n, l, &rank, c_data, ldc, s, x, ldx, &tol,
           iwork, dwork, ldwork, &iwarn, &info);

    free(iwork);

    f64 rcond = dwork[1];
    free(dwork);

    npy_intp s_dims[1] = {minmnl > 0 ? minmnl : 0};
    PyObject *s_array = PyArray_SimpleNew(1, s_dims, NPY_DOUBLE);
    if (s_array == NULL) {
        free(s);
        free(x);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }
    if (minmnl > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)s_array), s, minmnl * sizeof(f64));
    }
    free(s);

    npy_intp x_dims[2] = {n, l > 0 ? l : 0};
    npy_intp x_strides[2] = {sizeof(f64), ldx * sizeof(f64)};
    PyObject *x_array = NULL;

    if (n > 0 && l > 0) {
        x_array = PyArray_New(&PyArray_Type, 2, x_dims, NPY_DOUBLE,
                              x_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (x_array == NULL) {
            free(x);
            Py_DECREF(c_array);
            Py_DECREF(s_array);
            return PyErr_NoMemory();
        }
        f64 *x_out_ptr = (f64*)PyArray_DATA((PyArrayObject*)x_array);
        memcpy(x_out_ptr, x, (size_t)ldx * (size_t)l * sizeof(f64));
        free(x);
    } else {
        free(x);
        npy_intp empty_dims[2] = {n, l};
        x_array = PyArray_ZEROS(2, empty_dims, NPY_DOUBLE, 1);
        if (x_array == NULL) {
            Py_DECREF(c_array);
            Py_DECREF(s_array);
            return PyErr_NoMemory();
        }
    }

    PyObject *result = Py_BuildValue("OOOidii", c_array, s_array, x_array,
                                     rank, rcond, iwarn, info);
    Py_DECREF(c_array);
    Py_DECREF(s_array);
    Py_DECREF(x_array);

    return result;
}


/* Python wrapper for mb02ny - Separate zero singular value of bidiagonal submatrix */
PyObject* py_mb02ny(PyObject* self, PyObject* args) {
    int updatu_int, updatv_int;
    i32 m, n, i_idx, k;
    PyObject *q_obj, *e_obj, *u_obj, *v_obj;

    if (!PyArg_ParseTuple(args, "ppiiiiOOOO",
                          &updatu_int, &updatv_int, &m, &n, &i_idx, &k,
                          &q_obj, &e_obj, &u_obj, &v_obj)) {
        return NULL;
    }

    bool updatu = (bool)updatu_int;
    bool updatv = (bool)updatv_int;

    if (m < 0 || n < 0) {
        PyErr_SetString(PyExc_ValueError, "m and n must be non-negative");
        return NULL;
    }

    PyArrayObject *q_array = (PyArrayObject*)PyArray_FROM_OTF(
        q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!q_array) return NULL;

    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(
        e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!e_array) {
        Py_DECREF(q_array);
        return NULL;
    }

    PyArrayObject *u_array = (PyArrayObject*)PyArray_FROM_OTF(
        u_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!u_array) {
        Py_DECREF(q_array);
        Py_DECREF(e_array);
        return NULL;
    }

    PyArrayObject *v_array = (PyArrayObject*)PyArray_FROM_OTF(
        v_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!v_array) {
        Py_DECREF(q_array);
        Py_DECREF(e_array);
        Py_DECREF(u_array);
        return NULL;
    }

    f64 *q_data = (f64*)PyArray_DATA(q_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *u_data = (f64*)PyArray_DATA(u_array);
    f64 *v_data = (f64*)PyArray_DATA(v_array);

    npy_intp *u_dims = PyArray_DIMS(u_array);
    npy_intp *v_dims = PyArray_DIMS(v_array);
    i32 ldu = (i32)u_dims[0];
    i32 ldv = (i32)v_dims[0];

    i32 ldwork_max = 1;
    if (updatu && updatv) {
        i32 ki = k - i_idx;
        i32 im1 = i_idx - 1;
        ldwork_max = 2 * ((ki > im1) ? ki : im1);
    } else if (updatu) {
        ldwork_max = 2 * (k - i_idx);
    } else if (updatv) {
        ldwork_max = 2 * (i_idx - 1);
    }
    if (ldwork_max < 1) ldwork_max = 1;

    f64 *dwork = (f64*)malloc(ldwork_max * sizeof(f64));
    if (!dwork) {
        Py_DECREF(q_array);
        Py_DECREF(e_array);
        Py_DECREF(u_array);
        Py_DECREF(v_array);
        return PyErr_NoMemory();
    }

    mb02ny(updatu, updatv, m, n, i_idx, k, q_data, e_data, u_data, ldu, v_data, ldv, dwork);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(q_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(u_array);
    PyArray_ResolveWritebackIfCopy(v_array);

    PyObject *result = Py_BuildValue("OOOO", q_array, e_array, u_array, v_array);
    Py_DECREF(q_array);
    Py_DECREF(e_array);
    Py_DECREF(u_array);
    Py_DECREF(v_array);

    return result;
}


PyObject* py_mb02qd(PyObject* self, PyObject* args) {
    char *job_str, *iniper_str;
    i32 m, n, nrhs;
    f64 rcond, svlmax;
    PyObject *a_obj, *b_obj, *y_obj = Py_None, *jpvt_obj = Py_None;
    PyArrayObject *a_array = NULL, *b_array = NULL, *y_array = NULL, *jpvt_array = NULL;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssiiiddOO|OO", &job_str, &iniper_str,
                          &m, &n, &nrhs, &rcond, &svlmax,
                          &a_obj, &b_obj, &y_obj, &jpvt_obj)) {
        return NULL;
    }

    char job = toupper(job_str[0]);
    char iniper = toupper(iniper_str[0]);
    bool leasts = (job == 'L');
    bool permut = (iniper == 'P');

    i32 mn = (m < n) ? m : n;
    (void)mn;  /* used in workspace calculation */

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 lda = (i32)PyArray_DIM(a_array, 0);
    i32 ldb = (i32)PyArray_DIM(b_array, 0);
    if (lda < 1) lda = 1;
    if (ldb < 1) ldb = 1;

    f64* a_data = (f64*)PyArray_DATA(a_array);
    f64* b_data = (f64*)PyArray_DATA(b_array);

    const f64* y_data = NULL;
    if (!leasts && y_obj != Py_None && nrhs > 0) {
        y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
        if (y_array == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            return NULL;
        }
        y_data = (const f64*)PyArray_DATA(y_array);
    }

    i32 jpvt_size = n > 0 ? n : 1;
    i32* jpvt = (i32*)calloc(jpvt_size, sizeof(i32));
    if (jpvt == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_XDECREF(y_array);
        return PyErr_NoMemory();
    }

    if (permut && jpvt_obj != Py_None) {
        jpvt_array = (PyArrayObject*)PyArray_FROM_OTF(jpvt_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
        if (jpvt_array == NULL) {
            free(jpvt);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_XDECREF(y_array);
            return NULL;
        }
        memcpy(jpvt, PyArray_DATA(jpvt_array), n * sizeof(i32));
        Py_DECREF(jpvt_array);
        jpvt_array = NULL;
    }

    i32 rank;
    f64 sval[3] = {0.0, 0.0, 0.0};

    i32 ldwork = (mn + 3*n + 1 > 2*mn + nrhs) ? (mn + 3*n + 1) : (2*mn + nrhs);
    if (ldwork < 1) ldwork = 1;

    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        free(jpvt);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_XDECREF(y_array);
        return PyErr_NoMemory();
    }

    mb02qd(job_str, iniper_str, m, n, nrhs, rcond, svlmax,
           a_data, lda, b_data, ldb, y_data, jpvt, &rank, sval, dwork, ldwork, &info);

    free(dwork);
    Py_XDECREF(y_array);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    npy_intp sval_dims[1] = {3};
    PyArrayObject *sval_array = (PyArrayObject*)PyArray_SimpleNew(1, sval_dims, NPY_DOUBLE);
    if (sval_array == NULL) {
        free(jpvt);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return PyErr_NoMemory();
    }
    memcpy(PyArray_DATA(sval_array), sval, 3 * sizeof(f64));

    npy_intp jpvt_dims[1] = {n > 0 ? n : 0};
    PyArrayObject *jpvt_out = (PyArrayObject*)PyArray_SimpleNew(1, jpvt_dims, NPY_INT32);
    if (jpvt_out == NULL) {
        free(jpvt);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(sval_array);
        return PyErr_NoMemory();
    }
    if (n > 0) {
        memcpy(PyArray_DATA(jpvt_out), jpvt, n * sizeof(i32));
    }
    free(jpvt);

    PyObject *result = Py_BuildValue("OiOOi", b_array, rank, sval_array, jpvt_out, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(sval_array);
    Py_DECREF(jpvt_out);

    return result;
}

PyObject* py_mb02jd(PyObject* self, PyObject* args) {
    (void)self;

    const char* job;
    int k, l, m, n, p, s;
    PyObject *tc_obj, *tr_obj;

    if (!PyArg_ParseTuple(args, "siiiiiiOO", &job, &k, &l, &m, &n, &p, &s, &tc_obj, &tr_obj)) {
        return NULL;
    }

    if (job[0] != 'Q' && job[0] != 'q' && job[0] != 'R' && job[0] != 'r') {
        PyErr_SetString(PyExc_ValueError, "job must be 'Q' or 'R'");
        return NULL;
    }

    if (k < 0) {
        PyErr_SetString(PyExc_ValueError, "k must be >= 0");
        return NULL;
    }
    if (l < 0) {
        PyErr_SetString(PyExc_ValueError, "l must be >= 0");
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
    if (p < 0) {
        PyErr_SetString(PyExc_ValueError, "p must be >= 0");
        return NULL;
    }
    if (s < 0) {
        PyErr_SetString(PyExc_ValueError, "s must be >= 0");
        return NULL;
    }

    i32 minmknl = (m * k < n * l) ? m * k : n * l;
    if (p * l >= minmknl + l) {
        PyErr_SetString(PyExc_ValueError, "p*l must be < min(m*k, n*l) + l");
        return NULL;
    }
    if ((p + s) * l >= minmknl + l) {
        PyErr_SetString(PyExc_ValueError, "(p+s)*l must be < min(m*k, n*l) + l");
        return NULL;
    }

    bool compq = (job[0] == 'Q' || job[0] == 'q');

    i32 mk = m * k;
    i32 ldtc = (mk > 1) ? mk : 1;
    i32 ldtr = (k > 1) ? k : 1;

    PyArrayObject *tc_array = (PyArrayObject*)PyArray_FROM_OTF(
        tc_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    if (tc_array == NULL) {
        return NULL;
    }

    PyArrayObject *tr_array = (PyArrayObject*)PyArray_FROM_OTF(
        tr_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    if (tr_array == NULL) {
        Py_DECREF(tc_array);
        return NULL;
    }

    f64* tc_data = (f64*)PyArray_DATA(tc_array);
    f64* tr_data = (f64*)PyArray_DATA(tr_array);

    i32 q_cols = s * l;
    i32 remain = minmknl - p * l;
    if (remain < q_cols) q_cols = remain;
    if (q_cols < 0) q_cols = 0;

    i32 ldq = compq ? ((mk > 1) ? mk : 1) : 1;
    i32 q_rows = compq ? mk : 1;

    i32 n_minus_p_plus_1 = n - p + 1;
    i32 min_n_term = (n < n_minus_p_plus_1) ? n : n_minus_p_plus_1;
    i32 ldr = (min_n_term * l > 1) ? min_n_term * l : 1;

    f64* q = (f64*)calloc((size_t)q_rows * (size_t)(q_cols > 0 ? q_cols : 1), sizeof(f64));
    if (q == NULL) {
        Py_DECREF(tc_array);
        Py_DECREF(tr_array);
        return PyErr_NoMemory();
    }

    f64* r = (f64*)calloc((size_t)ldr * (size_t)(q_cols > 0 ? q_cols : 1), sizeof(f64));
    if (r == NULL) {
        free(q);
        Py_DECREF(tc_array);
        Py_DECREF(tr_array);
        return PyErr_NoMemory();
    }

    i32 ldwork;
    i32 max_p1 = (p > 1) ? p : 1;
    if (compq) {
        i32 mk_term = mk;
        i32 nl_term = (n - max_p1) * l;
        i32 max_mk_nl = (mk_term > nl_term) ? mk_term : nl_term;
        ldwork = 1 + (mk + (n - 1) * l) * (l + 2 * k) + 6 * l + max_mk_nl;
    } else {
        i32 n_minus_max_p1 = n - max_p1;
        ldwork = 1 + (n - 1) * l * (l + 2 * k) + 6 * l + n_minus_max_p1 * l;
        if (p == 0) {
            i32 alt = mk * (l + 1) + l;
            if (alt > ldwork) ldwork = alt;
        }
    }
    if (ldwork < 1) ldwork = 1;

    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        free(q);
        free(r);
        Py_DECREF(tc_array);
        Py_DECREF(tr_array);
        return PyErr_NoMemory();
    }

    i32 info;
    mb02jd(job, k, l, m, n, p, s, tc_data, ldtc, tr_data, ldtr,
           q, ldq, r, ldr, dwork, ldwork, &info);

    free(dwork);
    Py_DECREF(tc_array);
    Py_DECREF(tr_array);

    if (info < 0) {
        free(q);
        free(r);
        PyErr_Format(PyExc_ValueError, "MB02JD: illegal value for argument %d", -info);
        return NULL;
    }

    npy_intp q_dims[2] = {q_rows, q_cols > 0 ? q_cols : 0};
    npy_intp q_strides[2] = {sizeof(f64), q_rows * sizeof(f64)};
    PyArrayObject *q_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (q_array == NULL) {
        free(q);
        free(r);
        return PyErr_NoMemory();
    }
    f64 *q_out_ptr = (f64*)PyArray_DATA(q_array);
    memcpy(q_out_ptr, q, (size_t)q_rows * (size_t)(q_cols > 0 ? q_cols : 0) * sizeof(f64));
    free(q);

    npy_intp r_dims[2] = {ldr, q_cols > 0 ? q_cols : 0};
    npy_intp r_strides[2] = {sizeof(f64), ldr * sizeof(f64)};
    PyArrayObject *r_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, r_dims, NPY_DOUBLE, r_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (r_array == NULL) {
        Py_DECREF(q_array);
        free(r);
        return PyErr_NoMemory();
    }
    f64 *r_out_ptr = (f64*)PyArray_DATA(r_array);
    memcpy(r_out_ptr, r, (size_t)ldr * (size_t)(q_cols > 0 ? q_cols : 0) * sizeof(f64));
    free(r);

    PyObject *result = Py_BuildValue("OOi", q_array, r_array, info);
    Py_DECREF(q_array);
    Py_DECREF(r_array);

    return result;
}

PyObject* py_mb02id(PyObject* self, PyObject* args, PyObject* kwargs)
{
    const char* job;
    i32 k, l, m, n;
    PyObject *tc_obj, *tr_obj, *b_obj, *c_obj;

    static char* kwlist[] = {"job", "k", "l", "m", "n", "tc", "tr", "b", "c", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siiiiOOOO", kwlist,
                                     &job, &k, &l, &m, &n,
                                     &tc_obj, &tr_obj, &b_obj, &c_obj)) {
        return NULL;
    }

    char job_u = (char)toupper((unsigned char)job[0]);
    bool compo = (job_u == 'O') || (job_u == 'A');
    bool compu = (job_u == 'U') || (job_u == 'A');

    if (!compo && !compu) {
        PyErr_SetString(PyExc_ValueError, "MB02ID: job must be 'O', 'U', or 'A'");
        return NULL;
    }
    if (k < 0) {
        PyErr_SetString(PyExc_ValueError, "MB02ID: k must be >= 0");
        return NULL;
    }
    if (l < 0) {
        PyErr_SetString(PyExc_ValueError, "MB02ID: l must be >= 0");
        return NULL;
    }
    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "MB02ID: m must be >= 0");
        return NULL;
    }
    i32 mk = m * k;
    i32 nl = n * l;
    if (n < 0 || nl > mk) {
        PyErr_SetString(PyExc_ValueError, "MB02ID: n must satisfy 0 <= n*l <= m*k");
        return NULL;
    }

    PyArrayObject *tc_array = (PyArrayObject*)PyArray_FROM_OTF(
        tc_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (tc_array == NULL) {
        return NULL;
    }

    PyArrayObject *tr_array = (PyArrayObject*)PyArray_FROM_OTF(
        tr_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (tr_array == NULL) {
        Py_DECREF(tc_array);
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(tc_array);
        Py_DECREF(tr_array);
        return NULL;
    }

    PyArrayObject *c_in_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (c_in_array == NULL) {
        Py_DECREF(tc_array);
        Py_DECREF(tr_array);
        Py_DECREF(b_array);
        return NULL;
    }

    i32 ldtc = (i32)PyArray_DIM(tc_array, 0);
    i32 ldtr = (i32)PyArray_DIM(tr_array, 0);
    i32 ldb = (i32)PyArray_DIM(b_array, 0);
    i32 rb = (PyArray_NDIM(b_array) > 1) ? (i32)PyArray_DIM(b_array, 1) : 1;
    i32 rc = (PyArray_NDIM(c_in_array) > 1) ? (i32)PyArray_DIM(c_in_array, 1) : 1;

    if (compo && rb < 0) {
        Py_DECREF(tc_array);
        Py_DECREF(tr_array);
        Py_DECREF(b_array);
        Py_DECREF(c_in_array);
        PyErr_SetString(PyExc_ValueError, "MB02ID: rb must be >= 0");
        return NULL;
    }
    if (compu && rc < 0) {
        Py_DECREF(tc_array);
        Py_DECREF(tr_array);
        Py_DECREF(b_array);
        Py_DECREF(c_in_array);
        PyErr_SetString(PyExc_ValueError, "MB02ID: rc must be >= 0");
        return NULL;
    }

    f64* tc_data = (f64*)PyArray_DATA(tc_array);
    f64* tr_data = (f64*)PyArray_DATA(tr_array);
    f64* b_data = (f64*)PyArray_DATA(b_array);

    i32 ldc = (mk > 1) ? mk : 1;
    f64* c_data = NULL;
    if (compu) {
        c_data = (f64*)calloc((size_t)ldc * rc, sizeof(f64));
        if (c_data == NULL) {
            Py_DECREF(tc_array);
            Py_DECREF(tr_array);
            Py_DECREF(b_array);
            Py_DECREF(c_in_array);
            return PyErr_NoMemory();
        }
        f64* c_in_data = (f64*)PyArray_DATA(c_in_array);
        for (i32 j = 0; j < rc; j++) {
            for (i32 i = 0; i < nl; i++) {
                c_data[i + j * ldc] = c_in_data[i + j * nl];
            }
        }
    } else {
        c_data = (f64*)calloc(1, sizeof(f64));
        if (c_data == NULL) {
            Py_DECREF(tc_array);
            Py_DECREF(tr_array);
            Py_DECREF(b_array);
            Py_DECREF(c_in_array);
            return PyErr_NoMemory();
        }
        ldc = 1;
    }
    Py_DECREF(c_in_array);

    i32 x_term1 = 2 * nl * (l + k) + (6 + n) * l;
    i32 x_term2 = (nl + mk + 1) * l + mk;
    i32 x = (x_term1 > x_term2) ? x_term1 : x_term2;
    i32 y = n * mk * l + nl;

    i32 minmn = (m < n) ? m : n;
    i32 ldwork;
    if (minmn <= 1) {
        ldwork = mk;
        if (compo && rb > ldwork) ldwork = rb;
        if (compu && rc > ldwork) ldwork = rc;
        ldwork = y + ldwork;
        if (ldwork < 1) ldwork = 1;
    } else {
        ldwork = x;
        if (compo) {
            i32 t = nl * rb + 1;
            if (t > ldwork) ldwork = t;
        }
        if (compu) {
            i32 t = nl * rc + 1;
            if (t > ldwork) ldwork = t;
        }
    }

    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(tc_array);
        Py_DECREF(tr_array);
        Py_DECREF(b_array);
        free(c_data);
        return PyErr_NoMemory();
    }

    i32 info;
    mb02id(job, k, l, m, n, rb, rc, tc_data, ldtc, tr_data, ldtr,
           b_data, ldb, c_data, ldc, dwork, ldwork, &info);

    free(dwork);
    PyArray_ResolveWritebackIfCopy(b_array);
    Py_DECREF(tc_array);
    Py_DECREF(tr_array);

    if (info < 0) {
        Py_DECREF(b_array);
        free(c_data);
        PyErr_Format(PyExc_ValueError, "MB02ID: illegal value for argument %d", -info);
        return NULL;
    }

    PyObject *c_out_array;
    if (compu) {
        npy_intp c_dims[2] = {mk, rc};
        npy_intp c_strides[2] = {sizeof(f64), mk * sizeof(f64)};
        c_out_array = PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE, c_strides,
                                   NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (c_out_array == NULL) {
            Py_DECREF(b_array);
            free(c_data);
            return NULL;
        }
        f64 *c_out_data = (f64*)PyArray_DATA((PyArrayObject*)c_out_array);
        memcpy(c_out_data, c_data, (size_t)mk * (size_t)rc * sizeof(f64));
        free(c_data);
    } else {
        free(c_data);
        npy_intp c_dims[2] = {0, 0};
        c_out_array = PyArray_ZEROS(2, c_dims, NPY_DOUBLE, 1);
        if (c_out_array == NULL) {
            Py_DECREF(b_array);
            return NULL;
        }
    }

    PyObject *result = Py_BuildValue("OOi", b_array, c_out_array, info);
    Py_DECREF(b_array);
    Py_DECREF(c_out_array);

    return result;
}

PyObject* py_mb02jx(PyObject* self, PyObject* args) {
    (void)self;

    const char* job;
    int k, l, m, n;
    PyObject *tc_obj, *tr_obj;
    double tol1, tol2;

    if (!PyArg_ParseTuple(args, "siiiiOOdd", &job, &k, &l, &m, &n, &tc_obj, &tr_obj, &tol1, &tol2)) {
        return NULL;
    }

    if (job[0] != 'Q' && job[0] != 'q' && job[0] != 'R' && job[0] != 'r') {
        PyErr_SetString(PyExc_ValueError, "job must be 'Q' or 'R'");
        return NULL;
    }

    if (k < 0) {
        PyErr_SetString(PyExc_ValueError, "k must be >= 0");
        return NULL;
    }
    if (l < 0) {
        PyErr_SetString(PyExc_ValueError, "l must be >= 0");
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

    bool compq = (job[0] == 'Q' || job[0] == 'q');

    i32 mk = m * k;
    i32 nl = n * l;
    i32 ldtc = (mk > 1) ? mk : 1;
    i32 ldtr = (k > 1) ? k : 1;

    PyArrayObject *tc_array = (PyArrayObject*)PyArray_FROM_OTF(
        tc_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    if (tc_array == NULL) {
        return NULL;
    }

    PyArrayObject *tr_array = (PyArrayObject*)PyArray_FROM_OTF(
        tr_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    if (tr_array == NULL) {
        Py_DECREF(tc_array);
        return NULL;
    }

    const f64* tc_data = (const f64*)PyArray_DATA(tc_array);
    const f64* tr_data = (const f64*)PyArray_DATA(tr_array);

    i32 ldq = compq ? ((mk > 1) ? mk : 1) : 1;
    i32 q_rows = compq ? mk : 1;
    i32 max_rnk = (mk < nl) ? mk : nl;
    if (max_rnk < 1) max_rnk = 1;

    i32 ldr = (nl > 1) ? nl : 1;

    f64* q = (f64*)calloc((size_t)q_rows * (size_t)max_rnk, sizeof(f64));
    if (q == NULL) {
        Py_DECREF(tc_array);
        Py_DECREF(tr_array);
        return PyErr_NoMemory();
    }

    f64* r = (f64*)calloc((size_t)ldr * (size_t)max_rnk, sizeof(f64));
    if (r == NULL) {
        free(q);
        Py_DECREF(tc_array);
        Py_DECREF(tr_array);
        return PyErr_NoMemory();
    }

    i32 jpvt_size = (mk < nl) ? mk : nl;
    if (jpvt_size < 1) jpvt_size = 1;
    i32* jpvt = (i32*)calloc(jpvt_size, sizeof(i32));
    if (jpvt == NULL) {
        free(q);
        free(r);
        Py_DECREF(tc_array);
        Py_DECREF(tr_array);
        return PyErr_NoMemory();
    }

    i32 ldwork;
    if (compq) {
        i32 term1 = (mk + (n - 1) * l) * (l + 2 * k) + 9 * l;
        i32 max_term = (mk > (n - 1) * l) ? mk : (n - 1) * l;
        ldwork = (3 > term1 + max_term) ? 3 : term1 + max_term;
    } else {
        i32 term1 = (n - 1) * l * (l + 2 * k + 1) + 9 * l;
        i32 term2 = mk * (l + 1) + l;
        ldwork = 3;
        if (term1 > ldwork) ldwork = term1;
        if (term2 > ldwork) ldwork = term2;
    }
    if (ldwork < 3) ldwork = 3;

    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        free(q);
        free(r);
        free(jpvt);
        Py_DECREF(tc_array);
        Py_DECREF(tr_array);
        return PyErr_NoMemory();
    }

    i32 rnk;
    i32 info;
    mb02jx(job, k, l, m, n, tc_data, ldtc, tr_data, ldtr,
           &rnk, q, ldq, r, ldr, jpvt, tol1, tol2, dwork, ldwork, &info);

    free(dwork);
    Py_DECREF(tc_array);
    Py_DECREF(tr_array);

    if (info < 0) {
        free(q);
        free(r);
        free(jpvt);
        PyErr_Format(PyExc_ValueError, "MB02JX: illegal value for argument %d", -info);
        return NULL;
    }

    i32 actual_rnk = (rnk > 0) ? rnk : 0;

    npy_intp q_dims[2] = {q_rows, actual_rnk};
    npy_intp q_strides[2] = {sizeof(f64), q_rows * sizeof(f64)};
    PyArrayObject *q_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (q_array == NULL) {
        free(q);
        free(r);
        free(jpvt);
        return PyErr_NoMemory();
    }
    f64 *q_out_data = (f64*)PyArray_DATA(q_array);
    for (i32 j = 0; j < actual_rnk; j++) {
        memcpy(q_out_data + j * q_rows, q + j * q_rows, (size_t)q_rows * sizeof(f64));
    }
    free(q);

    npy_intp r_dims[2] = {ldr, actual_rnk};
    npy_intp r_strides[2] = {sizeof(f64), ldr * sizeof(f64)};
    PyArrayObject *r_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, r_dims, NPY_DOUBLE, r_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (r_array == NULL) {
        Py_DECREF(q_array);
        free(r);
        free(jpvt);
        return PyErr_NoMemory();
    }
    f64 *r_out_data = (f64*)PyArray_DATA(r_array);
    for (i32 j = 0; j < actual_rnk; j++) {
        memcpy(r_out_data + j * ldr, r + j * ldr, (size_t)ldr * sizeof(f64));
    }
    free(r);

    npy_intp jpvt_dims[1] = {jpvt_size};
    PyArrayObject *jpvt_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 1, jpvt_dims, NPY_INT32, NULL, NULL, 0, 0, NULL);
    if (jpvt_array == NULL) {
        Py_DECREF(q_array);
        Py_DECREF(r_array);
        free(jpvt);
        return PyErr_NoMemory();
    }
    i32 *jpvt_out_data = (i32*)PyArray_DATA(jpvt_array);
    memcpy(jpvt_out_data, jpvt, (size_t)jpvt_size * sizeof(i32));
    free(jpvt);

    PyObject *result = Py_BuildValue("iOOOi", rnk, q_array, r_array, jpvt_array, info);
    Py_DECREF(q_array);
    Py_DECREF(r_array);
    Py_DECREF(jpvt_array);

    return result;
}


/* Python wrapper for mb02td */
PyObject* py_mb02td(PyObject* self, PyObject* args) {
    char* norm;
    int n;
    double hnorm;
    PyObject *h_obj, *ipiv_obj;
    PyArrayObject *h_array, *ipiv_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "sidOO", &norm, &n, &hnorm, &h_obj, &ipiv_obj)) {
        return NULL;
    }

    h_array = (PyArrayObject*)PyArray_FROM_OTF(h_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (h_array == NULL) return NULL;

    ipiv_array = (PyArrayObject*)PyArray_FROM_OTF(ipiv_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (ipiv_array == NULL) {
        Py_DECREF(h_array);
        return NULL;
    }

    i32 ldh = (n > 1) ? n : 1;
    if (n > 0) {
        ldh = (i32)PyArray_DIM(h_array, 0);
    }

    f64* h = (f64*)PyArray_DATA(h_array);
    i32* ipiv = (i32*)PyArray_DATA(ipiv_array);

    i32* iwork = NULL;
    f64* dwork = NULL;
    i32 ldwork = (n > 0) ? 3 * n : 1;
    i32 liwork = (n > 0) ? n : 1;

    iwork = (i32*)malloc(liwork * sizeof(i32));
    dwork = (f64*)malloc(ldwork * sizeof(f64));

    if ((n > 0 && iwork == NULL) || (n > 0 && dwork == NULL)) {
        free(iwork);
        free(dwork);
        Py_DECREF(h_array);
        Py_DECREF(ipiv_array);
        return PyErr_NoMemory();
    }

    f64 rcond;
    mb02td(norm, n, hnorm, h, ldh, ipiv, &rcond, iwork, dwork, &info);

    free(iwork);
    free(dwork);
    Py_DECREF(h_array);
    Py_DECREF(ipiv_array);

    return Py_BuildValue("di", rcond, info);
}

PyObject* py_mb02uw(PyObject* self, PyObject* args)
{
    int ltrans;
    PyObject *par_obj, *a_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "pOOO", &ltrans, &par_obj, &a_obj, &b_obj)) {
        return NULL;
    }

    PyArrayObject *par_array = (PyArrayObject*)PyArray_FROM_OTF(par_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (par_array == NULL) return NULL;

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        Py_DECREF(par_array);
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(par_array);
        Py_DECREF(a_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 m = (i32)PyArray_DIM(b_array, 1);
    i32 lda = (n > 1) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 ldb = (n > 1) ? (i32)PyArray_DIM(b_array, 0) : 1;

    f64* par = (f64*)PyArray_DATA(par_array);
    f64* a = (f64*)PyArray_DATA(a_array);
    f64* b = (f64*)PyArray_DATA(b_array);

    f64 scale;
    i32 iwarn;

    mb02uw((bool)ltrans, n, m, par, a, lda, b, ldb, &scale, &iwarn);

    PyArray_ResolveWritebackIfCopy(b_array);

    Py_DECREF(par_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);

    return Py_BuildValue("Odi", b_obj, scale, iwarn);
}

PyObject* py_mb02nd(PyObject* self, PyObject* args)
{
    i32 m, n, l, rank;
    f64 theta, tol, reltol;
    PyObject *c_obj;

    if (!PyArg_ParseTuple(args, "iiiidOdd",
                          &m, &n, &l, &rank, &theta, &c_obj, &tol, &reltol)) {
        return NULL;
    }

    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be non-negative");
        return NULL;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }
    if (l < 0) {
        PyErr_SetString(PyExc_ValueError, "l must be non-negative");
        return NULL;
    }

    i32 min_mn = (m < n) ? m : n;
    if (rank > min_mn) {
        PyErr_SetString(PyExc_ValueError, "rank must be <= min(m,n)");
        return NULL;
    }
    if (rank < 0 && theta < 0.0) {
        PyErr_SetString(PyExc_ValueError, "theta must be >= 0 when rank < 0");
        return NULL;
    }

    i32 nl = n + l;
    i32 p = (m < nl) ? m : nl;
    i32 k = (m > nl) ? m : nl;

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) return NULL;

    i32 ldc = (k > 1) ? k : 1;
    if (PyArray_NDIM(c_array) >= 1 && PyArray_DIM(c_array, 0) > 0) {
        ldc = (i32)PyArray_DIM(c_array, 0);
    }
    i32 ldc_min = (1 > k) ? 1 : k;
    if (ldc < ldc_min) {
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_ValueError, "c has insufficient leading dimension");
        return NULL;
    }

    f64* c = (f64*)PyArray_DATA(c_array);

    i32 ldx = (n > 1) ? n : 1;
    i32 x_dim1 = (n > 0) ? n : 1;
    i32 x_dim2 = (l > 0) ? l : 1;
    npy_intp x_dims[2] = {x_dim1, x_dim2};
    npy_intp x_strides[2] = {(npy_intp)sizeof(f64), (npy_intp)(x_dim1 * sizeof(f64))};
    f64* x = (f64*)calloc((size_t)x_dim1 * (size_t)x_dim2, sizeof(f64));
    if (x == NULL) {
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    i32 q_len = (p > 0) ? 2*p - 1 : 1;
    npy_intp q_dims[1] = {q_len};
    f64* q = (f64*)calloc((size_t)q_len, sizeof(f64));
    if (q == NULL) {
        free(x);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    bool* inul = (bool*)PyMem_Calloc((size_t)(nl > 0 ? nl : 1), sizeof(bool));
    if (inul == NULL) {
        free(q);
        free(x);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    bool* bwork = (bool*)PyMem_Calloc((size_t)(nl > 0 ? nl : 1), sizeof(bool));
    if (bwork == NULL) {
        PyMem_Free(inul);
        free(q);
        free(x);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    i32 iwork_len = n + 2*l;
    if (iwork_len < 1) iwork_len = 1;
    i32* iwork = (i32*)calloc((size_t)iwork_len, sizeof(i32));
    if (iwork == NULL) {
        PyMem_Free(bwork);
        PyMem_Free(inul);
        free(q);
        free(x);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    i32 lw;
    if (m >= nl) {
        lw = (nl * (nl - 1)) / 2;
    } else {
        lw = m * nl - (m * (m - 1)) / 2;
    }
    i32 term1 = 6 * nl - 5;
    i32 term2 = l * l + ((nl > 3*l) ? nl : 3*l);
    i32 term_max = (term1 > term2) ? term1 : term2;
    i32 ldwork = p + lw + term_max;
    i32 ldwork_alt = k + 2 * p;
    ldwork = (ldwork > ldwork_alt) ? ldwork : ldwork_alt;
    ldwork = (ldwork > 2) ? ldwork : 2;
    ldwork = (i32)(ldwork * 1.5);

    f64* dwork = (f64*)calloc((size_t)ldwork, sizeof(f64));
    if (dwork == NULL) {
        free(iwork);
        PyMem_Free(bwork);
        PyMem_Free(inul);
        free(q);
        free(x);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    i32 iwarn = 0, info = 0;

    mb02nd(m, n, l, &rank, &theta, c, ldc, x, ldx, q, inul,
           tol, reltol, iwork, dwork, ldwork, bwork, &iwarn, &info);

    free(dwork);
    free(iwork);
    PyMem_Free(bwork);

    PyArray_ResolveWritebackIfCopy(c_array);
    Py_DECREF(c_array);

    if (info < 0) {
        PyMem_Free(inul);
        free(q);
        free(x);
        PyErr_Format(PyExc_ValueError, "mb02nd: illegal value in argument %d", -info);
        return NULL;
    }

    PyArrayObject* x_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, x_dims, NPY_DOUBLE, x_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (x_array == NULL) {
        PyMem_Free(inul);
        free(q);
        free(x);
        return NULL;
    }
    f64 *x_out_data = (f64*)PyArray_DATA(x_array);
    memcpy(x_out_data, x, (size_t)x_dims[0] * (size_t)x_dims[1] * sizeof(f64));
    free(x);

    PyArrayObject* q_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 1, q_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
    if (q_array == NULL) {
        PyMem_Free(inul);
        free(q);
        Py_DECREF(x_array);
        return NULL;
    }
    f64 *q_out_data = (f64*)PyArray_DATA(q_array);
    memcpy(q_out_data, q, (size_t)q_dims[0] * sizeof(f64));
    free(q);

    npy_intp inul_dims[1] = {nl > 0 ? nl : 1};
    PyObject* inul_list = PyList_New(inul_dims[0]);
    if (inul_list == NULL) {
        PyMem_Free(inul);
        Py_DECREF(q_array);
        Py_DECREF(x_array);
        return NULL;
    }
    for (i32 i = 0; i < inul_dims[0]; i++) {
        PyList_SET_ITEM(inul_list, i, PyBool_FromLong(inul[i]));
    }
    PyMem_Free(inul);

    return Py_BuildValue("OidOOii", x_array, rank, theta, q_array, inul_list, iwarn, info);
}

PyObject* py_mb02nd_full(PyObject* self, PyObject* args)
{
    i32 m, n, l, rank;
    f64 theta, tol, reltol;
    PyObject *c_obj;

    if (!PyArg_ParseTuple(args, "iiiidOdd",
                          &m, &n, &l, &rank, &theta, &c_obj, &tol, &reltol)) {
        return NULL;
    }

    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be non-negative");
        return NULL;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }
    if (l < 0) {
        PyErr_SetString(PyExc_ValueError, "l must be non-negative");
        return NULL;
    }

    i32 min_mn = (m < n) ? m : n;
    if (rank > min_mn) {
        PyErr_SetString(PyExc_ValueError, "rank must be <= min(m,n)");
        return NULL;
    }
    if (rank < 0 && theta < 0.0) {
        PyErr_SetString(PyExc_ValueError, "theta must be >= 0 when rank < 0");
        return NULL;
    }

    i32 nl = n + l;
    i32 p = (m < nl) ? m : nl;
    i32 k = (m > nl) ? m : nl;

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) return NULL;

    i32 ldc = (k > 1) ? k : 1;
    if (PyArray_NDIM(c_array) >= 1 && PyArray_DIM(c_array, 0) > 0) {
        ldc = (i32)PyArray_DIM(c_array, 0);
    }
    i32 ldc_min = (1 > k) ? 1 : k;
    if (ldc < ldc_min) {
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_ValueError, "c has insufficient leading dimension");
        return NULL;
    }

    f64* c = (f64*)PyArray_DATA(c_array);

    i32 ldx = (n > 1) ? n : 1;
    i32 x_dim1 = (n > 0) ? n : 1;
    i32 x_dim2 = (l > 0) ? l : 1;
    npy_intp x_dims[2] = {x_dim1, x_dim2};
    npy_intp x_strides[2] = {(npy_intp)sizeof(f64), (npy_intp)(x_dim1 * sizeof(f64))};
    f64* x = (f64*)calloc((size_t)x_dim1 * (size_t)x_dim2, sizeof(f64));
    if (x == NULL) {
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    i32 q_len = (p > 0) ? 2*p - 1 : 1;
    npy_intp q_dims[1] = {q_len};
    f64* q = (f64*)calloc((size_t)q_len, sizeof(f64));
    if (q == NULL) {
        free(x);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    bool* inul = (bool*)PyMem_Calloc((size_t)(nl > 0 ? nl : 1), sizeof(bool));
    if (inul == NULL) {
        free(q);
        free(x);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    bool* bwork = (bool*)PyMem_Calloc((size_t)(nl > 0 ? nl : 1), sizeof(bool));
    if (bwork == NULL) {
        PyMem_Free(inul);
        free(q);
        free(x);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    i32 iwork_len = n + 2*l;
    if (iwork_len < 1) iwork_len = 1;
    i32* iwork = (i32*)calloc((size_t)iwork_len, sizeof(i32));
    if (iwork == NULL) {
        PyMem_Free(bwork);
        PyMem_Free(inul);
        free(q);
        free(x);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    i32 lw;
    if (m >= nl) {
        lw = (nl * (nl - 1)) / 2;
    } else {
        lw = m * nl - (m * (m - 1)) / 2;
    }
    i32 term1 = 6 * nl - 5;
    i32 term2 = l * l + ((nl > 3*l) ? nl : 3*l);
    i32 term_max = (term1 > term2) ? term1 : term2;
    i32 ldwork = p + lw + term_max;
    i32 ldwork_alt = k + 2 * p;
    ldwork = (ldwork > ldwork_alt) ? ldwork : ldwork_alt;
    ldwork = (ldwork > 2) ? ldwork : 2;
    ldwork = (i32)(ldwork * 1.5);

    f64* dwork = (f64*)calloc((size_t)ldwork, sizeof(f64));
    if (dwork == NULL) {
        free(iwork);
        PyMem_Free(bwork);
        PyMem_Free(inul);
        free(q);
        free(x);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    i32 iwarn = 0, info = 0;

    mb02nd(m, n, l, &rank, &theta, c, ldc, x, ldx, q, inul,
           tol, reltol, iwork, dwork, ldwork, bwork, &iwarn, &info);

    free(dwork);
    free(iwork);
    PyMem_Free(bwork);

    PyArray_ResolveWritebackIfCopy(c_array);
    Py_DECREF(c_array);

    if (info < 0) {
        PyMem_Free(inul);
        free(q);
        free(x);
        PyErr_Format(PyExc_ValueError, "mb02nd: illegal value in argument %d", -info);
        return NULL;
    }

    PyArrayObject* x_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, x_dims, NPY_DOUBLE, x_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (x_array == NULL) {
        PyMem_Free(inul);
        free(q);
        free(x);
        return NULL;
    }
    f64 *x_out_data = (f64*)PyArray_DATA(x_array);
    memcpy(x_out_data, x, (size_t)x_dims[0] * (size_t)x_dims[1] * sizeof(f64));
    free(x);

    PyArrayObject* q_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 1, q_dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
    if (q_array == NULL) {
        PyMem_Free(inul);
        free(q);
        Py_DECREF(x_array);
        return NULL;
    }
    f64 *q_out_data = (f64*)PyArray_DATA(q_array);
    memcpy(q_out_data, q, (size_t)q_dims[0] * sizeof(f64));
    free(q);

    npy_intp inul_dims_val = nl > 0 ? nl : 1;
    PyObject* inul_list = PyList_New(inul_dims_val);
    if (inul_list == NULL) {
        PyMem_Free(inul);
        Py_DECREF(q_array);
        Py_DECREF(x_array);
        return NULL;
    }
    for (i32 i = 0; i < inul_dims_val; i++) {
        PyList_SET_ITEM(inul_list, i, PyBool_FromLong(inul[i]));
    }
    PyMem_Free(inul);

    return Py_BuildValue("OidOOii", x_array, rank, theta, q_array, inul_list, iwarn, info);
}


/* Python wrapper for mb02xd (standard form only, FORM='S') */
PyObject* py_mb02xd(PyObject* self, PyObject* args) {
    const char *stor_str, *uplo_str;
    int m, n, nrhs;
    PyObject *a_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "ssiiiOO", &stor_str, &uplo_str,
                          &m, &n, &nrhs, &a_obj, &b_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 lda = m > 1 ? m : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldata = n > 1 ? n : 1;

    bool full = (toupper((unsigned char)stor_str[0]) == 'F');
    i32 ata_size = full ? (n * n) : (n * (n + 1) / 2);
    if (ata_size < 1) ata_size = 1;

    npy_intp ata_dims[2] = {n > 0 ? n : 1, n > 0 ? n : 1};
    npy_intp ata_strides[2] = {sizeof(f64), ldata * sizeof(f64)};
    PyObject *ata_array = PyArray_New(&PyArray_Type, 2, ata_dims, NPY_DOUBLE,
                                      ata_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (ata_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return PyErr_NoMemory();
    }

    i32 lipar = 0;
    i32 ldpar = 0;
    i32 ldwork = 1;
    f64 dwork_val = 0.0;
    i32 info = 0;

    mb02xd("S", stor_str, uplo_str, NULL,
           &m, &n, &nrhs, NULL, &lipar, NULL, &ldpar,
           (f64*)PyArray_DATA(a_array), &lda,
           (f64*)PyArray_DATA(b_array), &ldb,
           (f64*)PyArray_DATA((PyArrayObject*)ata_array), &ldata,
           &dwork_val, &ldwork, &info);

    PyArray_ResolveWritebackIfCopy(b_array);

    PyObject *result = Py_BuildValue("OOi", b_array, ata_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(ata_array);

    return result;
}
