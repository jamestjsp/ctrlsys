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



/* Python wrapper for tc01od */
PyObject* py_tc01od(PyObject* self, PyObject* args) {
    const char *leri_str;
    i32 m, p;
    PyObject *pcoeff_obj, *qcoeff_obj;
    PyArrayObject *pcoeff_array, *qcoeff_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "siiOO", &leri_str, &m, &p, &pcoeff_obj, &qcoeff_obj)) {
        return NULL;
    }

    char leri = leri_str[0];

    pcoeff_array = (PyArrayObject*)PyArray_FROM_OTF(pcoeff_obj, NPY_DOUBLE,
                                                     NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (pcoeff_array == NULL) {
        return NULL;
    }

    qcoeff_array = (PyArrayObject*)PyArray_FROM_OTF(qcoeff_obj, NPY_DOUBLE,
                                                     NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (qcoeff_array == NULL) {
        Py_DECREF(pcoeff_array);
        return NULL;
    }

    npy_intp *pcoeff_dims = PyArray_DIMS(pcoeff_array);
    npy_intp *qcoeff_dims = PyArray_DIMS(qcoeff_array);

    i32 ldpco1 = (i32)pcoeff_dims[0];
    i32 ldpco2 = (i32)pcoeff_dims[1];
    i32 indlim = (i32)pcoeff_dims[2];

    i32 ldqco1 = (i32)qcoeff_dims[0];
    i32 ldqco2 = (i32)qcoeff_dims[1];

    f64 *pcoeff_data = (f64*)PyArray_DATA(pcoeff_array);
    f64 *qcoeff_data = (f64*)PyArray_DATA(qcoeff_array);

    tc01od(leri, m, p, indlim, pcoeff_data, ldpco1, ldpco2,
           qcoeff_data, ldqco1, ldqco2, &info);

    PyArray_ResolveWritebackIfCopy(pcoeff_array);
    PyArray_ResolveWritebackIfCopy(qcoeff_array);

    PyObject *result = Py_BuildValue("OOi", pcoeff_array, qcoeff_array, info);
    Py_DECREF(pcoeff_array);
    Py_DECREF(qcoeff_array);

    return result;
}



/* Python wrapper for tc04ad - Polynomial matrix representation to state-space */
PyObject* py_tc04ad(PyObject* self, PyObject* args) {
    const char *leri_str;
    i32 m, p;
    PyObject *index_obj, *pcoeff_obj, *qcoeff_obj;
    PyArrayObject *index_array, *pcoeff_array, *qcoeff_array;
    i32 n, info;
    f64 rcond;

    if (!PyArg_ParseTuple(args, "siiOOO", &leri_str, &m, &p, &index_obj, &pcoeff_obj, &qcoeff_obj)) {
        return NULL;
    }

    char leri = (char)toupper((unsigned char)leri_str[0]);
    if (leri != 'L' && leri != 'R') {
        PyErr_SetString(PyExc_ValueError, "leri must be 'L' or 'R'");
        return NULL;
    }

    index_array = (PyArrayObject*)PyArray_FROM_OTF(index_obj, NPY_INT32,
                                                    NPY_ARRAY_CARRAY_RO);
    if (index_array == NULL) {
        return NULL;
    }

    pcoeff_array = (PyArrayObject*)PyArray_FROM_OTF(pcoeff_obj, NPY_DOUBLE,
                                                     NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (pcoeff_array == NULL) {
        Py_DECREF(index_array);
        return NULL;
    }

    qcoeff_array = (PyArrayObject*)PyArray_FROM_OTF(qcoeff_obj, NPY_DOUBLE,
                                                     NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (qcoeff_array == NULL) {
        Py_DECREF(index_array);
        Py_DECREF(pcoeff_array);
        return NULL;
    }

    i32 *index_data = (i32*)PyArray_DATA(index_array);
    npy_intp *pcoeff_dims = PyArray_DIMS(pcoeff_array);
    npy_intp *qcoeff_dims = PyArray_DIMS(qcoeff_array);

    i32 ldpco1 = (i32)pcoeff_dims[0];
    i32 ldpco2 = (i32)pcoeff_dims[1];
    i32 ldqco1 = (i32)qcoeff_dims[0];
    i32 ldqco2 = (i32)qcoeff_dims[1];

    i32 mindex = (m > p) ? m : p;
    i32 pwork = (leri == 'L') ? p : m;

    i32 n_sum = 0;
    for (i32 i = 0; i < pwork; i++) {
        n_sum += index_data[i];
    }

    i32 lda = (n_sum > 0) ? n_sum : 1;
    i32 ldb = (n_sum > 0) ? n_sum : 1;
    i32 ldc = (mindex > 0) ? mindex : 1;
    i32 ldd = (mindex > 0) ? mindex : 1;

    i32 ldwork = (mindex * (mindex + 4) > 1) ? mindex * (mindex + 4) : 1;
    i32 iwork_size = 2 * mindex > 1 ? 2 * mindex : 1;

    f64 *a = (f64*)calloc(lda * n_sum + 1, sizeof(f64));
    f64 *b = (f64*)calloc(ldb * mindex + 1, sizeof(f64));
    f64 *c = (f64*)calloc(ldc * (n_sum > 0 ? n_sum : 1) + 1, sizeof(f64));
    f64 *d = (f64*)calloc(ldd * mindex + 1, sizeof(f64));
    i32 *iwork = (i32*)malloc(iwork_size * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!a || !b || !c || !d || !iwork || !dwork) {
        free(a); free(b); free(c); free(d);
        free(iwork); free(dwork);
        Py_DECREF(index_array);
        Py_DECREF(pcoeff_array);
        Py_DECREF(qcoeff_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *pcoeff_data = (f64*)PyArray_DATA(pcoeff_array);
    f64 *qcoeff_data = (f64*)PyArray_DATA(qcoeff_array);

    tc04ad(leri, m, p, index_data, pcoeff_data, ldpco1, ldpco2,
           qcoeff_data, ldqco1, ldqco2, &n, &rcond, a, lda, b, ldb,
           c, ldc, d, ldd, iwork, dwork, ldwork, &info);

    PyArray_ResolveWritebackIfCopy(pcoeff_array);
    PyArray_ResolveWritebackIfCopy(qcoeff_array);

    free(iwork);
    free(dwork);
    Py_DECREF(pcoeff_array);
    Py_DECREF(qcoeff_array);
    Py_DECREF(index_array);

    if (info < 0) {
        free(a); free(b); free(c); free(d);
        PyErr_Format(PyExc_ValueError, "tc04ad: illegal value for argument %d", -info);
        return NULL;
    }

    npy_intp a_dims[2] = {n, n};
    npy_intp b_dims[2] = {n, m};
    npy_intp c_dims[2] = {p, n};
    npy_intp d_dims[2] = {p, m};

    npy_intp a_out_strides[2] = {sizeof(f64), n * (npy_intp)sizeof(f64)};
    npy_intp b_out_strides[2] = {sizeof(f64), n * (npy_intp)sizeof(f64)};
    npy_intp c_out_strides[2] = {sizeof(f64), p * (npy_intp)sizeof(f64)};
    npy_intp d_out_strides[2] = {sizeof(f64), p * (npy_intp)sizeof(f64)};

    PyObject *a_array, *b_array, *c_array, *d_array;

    if (n > 0) {
        a_array = PyArray_New(&PyArray_Type, 2, a_dims, NPY_DOUBLE, a_out_strides,
                              NULL, 0, NPY_ARRAY_FARRAY, NULL);
        b_array = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_out_strides,
                              NULL, 0, NPY_ARRAY_FARRAY, NULL);
        c_array = PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE, c_out_strides,
                              NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!a_array || !b_array || !c_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(b_array);
            Py_XDECREF(c_array);
            free(a); free(b); free(c); free(d);
            PyErr_NoMemory();
            return NULL;
        }
        f64 *a_out = (f64*)PyArray_DATA((PyArrayObject*)a_array);
        f64 *b_out = (f64*)PyArray_DATA((PyArrayObject*)b_array);
        f64 *c_out = (f64*)PyArray_DATA((PyArrayObject*)c_array);
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < n; i++) a_out[i + j*n] = a[i + j*lda];
        }
        for (i32 j = 0; j < m; j++) {
            for (i32 i = 0; i < n; i++) b_out[i + j*n] = b[i + j*ldb];
        }
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < p; i++) c_out[i + j*p] = c[i + j*ldc];
        }
        free(a); free(b); free(c);
    } else {
        npy_intp zero_a[2] = {0, 0};
        npy_intp zero_b[2] = {0, m};
        npy_intp zero_c[2] = {p, 0};
        a_array = PyArray_ZEROS(2, zero_a, NPY_DOUBLE, 1);
        b_array = PyArray_ZEROS(2, zero_b, NPY_DOUBLE, 1);
        c_array = PyArray_ZEROS(2, zero_c, NPY_DOUBLE, 1);
        free(a); free(b); free(c);
    }

    d_array = PyArray_New(&PyArray_Type, 2, d_dims, NPY_DOUBLE, d_out_strides,
                          NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!a_array || !b_array || !c_array || !d_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        free(d);
        PyErr_NoMemory();
        return NULL;
    }
    f64 *d_out = (f64*)PyArray_DATA((PyArrayObject*)d_array);
    for (i32 j = 0; j < m; j++) {
        for (i32 i = 0; i < p; i++) d_out[i + j*p] = d[i + j*ldd];
    }
    free(d);

    PyObject *result = Py_BuildValue("idOOOOi", n, rcond, a_array, b_array, c_array, d_array, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    return result;
}


/* Python wrapper for tc05ad - Transfer matrix frequency response */
PyObject* py_tc05ad(PyObject* self, PyObject* args) {
    const char *leri_str;
    i32 m, p;
    Py_complex sval_py;
    PyObject *index_obj, *pcoeff_obj, *qcoeff_obj;
    PyArrayObject *index_array, *pcoeff_array, *qcoeff_array;
    i32 info;
    f64 rcond;

    if (!PyArg_ParseTuple(args, "siiDOOO", &leri_str, &m, &p, &sval_py,
                          &index_obj, &pcoeff_obj, &qcoeff_obj)) {
        return NULL;
    }

    char leri = (char)toupper((unsigned char)leri_str[0]);
    if (leri != 'L' && leri != 'R') {
        PyErr_SetString(PyExc_ValueError, "leri must be 'L' or 'R'");
        return NULL;
    }

    c128 sval = sval_py.real + I * sval_py.imag;

    index_array = (PyArrayObject*)PyArray_FROM_OTF(index_obj, NPY_INT32,
                                                    NPY_ARRAY_CARRAY_RO);
    if (index_array == NULL) {
        return NULL;
    }

    pcoeff_array = (PyArrayObject*)PyArray_FROM_OTF(pcoeff_obj, NPY_DOUBLE,
                                                     NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (pcoeff_array == NULL) {
        Py_DECREF(index_array);
        return NULL;
    }

    qcoeff_array = (PyArrayObject*)PyArray_FROM_OTF(qcoeff_obj, NPY_DOUBLE,
                                                     NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (qcoeff_array == NULL) {
        Py_DECREF(index_array);
        Py_DECREF(pcoeff_array);
        return NULL;
    }

    i32 *index_data = (i32*)PyArray_DATA(index_array);
    npy_intp *pcoeff_dims = PyArray_DIMS(pcoeff_array);
    npy_intp *qcoeff_dims = PyArray_DIMS(qcoeff_array);

    i32 ldpco1 = (i32)pcoeff_dims[0];
    i32 ldpco2 = (i32)pcoeff_dims[1];
    i32 ldqco1 = (i32)qcoeff_dims[0];
    i32 ldqco2 = (i32)qcoeff_dims[1];

    i32 pwork = (leri == 'L') ? p : m;
    i32 mplim = (m > p) ? m : p;

    i32 liwork = pwork;
    i32 ldwork = 2 * pwork;
    i32 lzwork = pwork * (pwork + 2);

    i32 ldcfre;
    if (leri == 'L') {
        ldcfre = (1 > p) ? 1 : p;
    } else {
        ldcfre = (1 > mplim) ? 1 : mplim;
    }

    i32 *iwork = (i32*)PyMem_Calloc(liwork > 0 ? liwork : 1, sizeof(i32));
    f64 *dwork = (f64*)PyMem_Calloc(ldwork > 0 ? ldwork : 1, sizeof(f64));
    c128 *zwork = (c128*)PyMem_Calloc(lzwork > 0 ? lzwork : 1, sizeof(c128));

    if (!iwork || !dwork || !zwork) {
        PyMem_Free(iwork);
        PyMem_Free(dwork);
        PyMem_Free(zwork);
        Py_DECREF(index_array);
        Py_DECREF(pcoeff_array);
        Py_DECREF(qcoeff_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 out_rows = (leri == 'L') ? p : m;
    i32 out_cols = (leri == 'L') ? m : p;

    c128 *cfreqr_data = (c128*)PyMem_Calloc(ldcfre * mplim > 0 ? ldcfre * mplim : 1, sizeof(c128));
    if (cfreqr_data == NULL) {
        PyMem_Free(iwork);
        PyMem_Free(dwork);
        PyMem_Free(zwork);
        Py_DECREF(index_array);
        Py_DECREF(pcoeff_array);
        Py_DECREF(qcoeff_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *pcoeff_data = (f64*)PyArray_DATA(pcoeff_array);
    f64 *qcoeff_data = (f64*)PyArray_DATA(qcoeff_array);

    tc05ad(leri, m, p, sval, index_data, pcoeff_data, ldpco1, ldpco2,
           qcoeff_data, ldqco1, ldqco2, &rcond, cfreqr_data, ldcfre,
           iwork, dwork, zwork, &info);

    PyArray_ResolveWritebackIfCopy(pcoeff_array);
    PyArray_ResolveWritebackIfCopy(qcoeff_array);

    PyMem_Free(iwork);
    PyMem_Free(dwork);
    PyMem_Free(zwork);
    Py_DECREF(index_array);
    Py_DECREF(pcoeff_array);
    Py_DECREF(qcoeff_array);

    if (info < 0) {
        PyMem_Free(cfreqr_data);
        PyErr_Format(PyExc_ValueError, "tc05ad: illegal value for argument %d", -info);
        return NULL;
    }

    npy_intp cfreqr_dims[2] = {out_rows, out_cols};
    npy_intp cfreqr_strides[2] = {sizeof(c128), out_rows * (npy_intp)sizeof(c128)};
    PyObject *cfreqr_array = PyArray_New(&PyArray_Type, 2, cfreqr_dims, NPY_COMPLEX128,
                                         cfreqr_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (cfreqr_array == NULL) {
        PyMem_Free(cfreqr_data);
        PyErr_NoMemory();
        return NULL;
    }
    c128 *cfreqr_out = (c128*)PyArray_DATA((PyArrayObject*)cfreqr_array);
    for (i32 j = 0; j < out_cols; j++) {
        for (i32 i = 0; i < out_rows; i++) {
            cfreqr_out[i + j * out_rows] = cfreqr_data[i + j * ldcfre];
        }
    }
    PyMem_Free(cfreqr_data);

    PyObject *result = Py_BuildValue("dOi", rcond, cfreqr_array, info);
    Py_DECREF(cfreqr_array);

    return result;
}

