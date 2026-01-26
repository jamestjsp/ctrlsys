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



/* Python wrapper for sb03mv */
PyObject* py_sb03mv(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *t_obj, *b_obj;
    int ltran_int = 0, lupper_int = 1;
    PyArrayObject *t_array, *b_array;
    f64 scale, xnorm;
    i32 info;

    static char* kwlist[] = {"t", "b", "ltran", "lupper", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|pp", kwlist,
                                     &t_obj, &b_obj, &ltran_int, &lupper_int)) {
        return NULL;
    }

    t_array = (PyArrayObject*)PyArray_FROM_OTF(t_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (t_array == NULL) return NULL;

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (b_array == NULL) {
        Py_DECREF(t_array);
        return NULL;
    }

    f64* t = (f64*)PyArray_DATA(t_array);
    f64* b_data = (f64*)PyArray_DATA(b_array);

    npy_intp dims[2] = {2, 2};
    npy_intp strides[2] = {sizeof(f64), 2*sizeof(f64)};
    PyObject* x_array = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (x_array == NULL) {
        Py_DECREF(t_array);
        Py_DECREF(b_array);
        return PyErr_NoMemory();
    }
    f64* x = (f64*)PyArray_DATA((PyArrayObject*)x_array);

    sb03mv((bool)ltran_int, (bool)lupper_int, t, 2, b_data, 2, &scale, x, 2, &xnorm, &info);

    Py_DECREF(t_array);
    Py_DECREF(b_array);

    return Py_BuildValue("Oddi", x_array, scale, xnorm, info);
}



/* Python wrapper for sb03ov */
PyObject* py_sb03ov(PyObject* self, PyObject* args) {
    f64 a_re, a_im, b, small;
    f64 a[2], c[2], s;

    if (!PyArg_ParseTuple(args, "dddd", &a_re, &a_im, &b, &small)) {
        return NULL;
    }

    a[0] = a_re;
    a[1] = a_im;
    sb03ov(a, b, small, c, &s);

    return Py_BuildValue("(ddddi)", a[0], c[0], c[1], s, 0);
}



/* Python wrapper for sb03mx */
PyObject* py_sb03mx(PyObject* self, PyObject* args) {
    char* trana;
    PyObject *a_obj, *c_obj;
    PyArrayObject *a_array, *c_array;
    f64 scale;
    i32 info;

    if (!PyArg_ParseTuple(args, "sOO", &trana, &a_obj, &c_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) return NULL;

    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 lda = (n > 1) ? n : 1;
    i32 ldc = lda;

    f64* a = (f64*)PyArray_DATA(a_array);
    f64* c = (f64*)PyArray_DATA(c_array);

    f64* dwork = (f64*)malloc(2 * n * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    sb03mx(trana, n, a, lda, c, ldc, &scale, dwork, &info);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject* result = Py_BuildValue("Odi", c_array, scale, info);

    Py_DECREF(a_array);
    Py_DECREF(c_array);

    return result;
}



/* Python wrapper for sb03my */
PyObject* py_sb03my(PyObject* self, PyObject* args) {
    char* trana;
    PyObject *a_obj, *c_obj;
    PyArrayObject *a_array, *c_array;
    f64 scale;
    i32 info;

    if (!PyArg_ParseTuple(args, "sOO", &trana, &a_obj, &c_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) return NULL;

    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 lda = (n > 1) ? n : 1;
    i32 ldc = lda;

    f64* a = (f64*)PyArray_DATA(a_array);
    f64* c = (f64*)PyArray_DATA(c_array);

    sb03my(trana, n, a, lda, c, ldc, &scale, &info);

    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject* result = Py_BuildValue("Odi", c_array, scale, info);

    Py_DECREF(a_array);
    Py_DECREF(c_array);

    return result;
}



/* Python wrapper for sb03sx */
PyObject* py_sb03sx(PyObject* self, PyObject* args) {
    char *trana, *uplo, *lyapun;
    int n;
    double xanorm;
    PyObject *t_obj, *u_obj, *r_obj;
    PyArrayObject *t_array, *u_array, *r_array;
    f64 ferr;
    i32 info;

    if (!PyArg_ParseTuple(args, "sssidOOO", &trana, &uplo, &lyapun, &n, &xanorm,
                          &t_obj, &u_obj, &r_obj)) {
        return NULL;
    }

    t_array = (PyArrayObject*)PyArray_FROM_OTF(t_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (t_array == NULL) return NULL;

    u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (u_array == NULL) {
        Py_DECREF(t_array);
        return NULL;
    }

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) {
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        return NULL;
    }

    i32 ldt = (n > 1) ? n : 1;
    i32 ldu = ldt;
    i32 ldr = ldt;
    i32 nn = n * n;
    i32 ldwork = (n > 0) ? ((2 * nn > 3) ? 2 * nn : 3) : 0;
    i32 iwork_size = (nn > 0) ? nn : 1;
    i32 dwork_size = (ldwork > 0) ? ldwork : 1;

    f64* t = (f64*)PyArray_DATA(t_array);
    f64* u = (f64*)PyArray_DATA(u_array);
    f64* r = (f64*)PyArray_DATA(r_array);

    i32* iwork = (i32*)malloc(iwork_size * sizeof(i32));
    f64* dwork = (f64*)malloc(dwork_size * sizeof(f64));
    if (iwork == NULL || dwork == NULL) {
        free(iwork);
        free(dwork);
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        Py_DECREF(r_array);
        return PyErr_NoMemory();
    }

    sb03sx(trana, uplo, lyapun, n, xanorm, t, ldt, u, ldu, r, ldr, &ferr, iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(r_array);

    PyObject* result = Py_BuildValue("dOi", ferr, r_array, info);

    Py_DECREF(t_array);
    Py_DECREF(u_array);
    Py_DECREF(r_array);

    return result;
}



/* Python wrapper for sb03sy */
PyObject* py_sb03sy(PyObject* self, PyObject* args) {
    char *job, *trana, *lyapun;
    int n;
    PyObject *t_obj, *u_obj, *xa_obj;
    PyArrayObject *t_array, *u_array, *xa_array;
    f64 sepd = 0.0, thnorm = 0.0;
    i32 info;

    if (!PyArg_ParseTuple(args, "sssiOOO", &job, &trana, &lyapun, &n,
                          &t_obj, &u_obj, &xa_obj)) {
        return NULL;
    }

    t_array = (PyArrayObject*)PyArray_FROM_OTF(t_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (t_array == NULL) return NULL;

    u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (u_array == NULL) {
        Py_DECREF(t_array);
        return NULL;
    }

    xa_array = (PyArrayObject*)PyArray_FROM_OTF(xa_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (xa_array == NULL) {
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        return NULL;
    }

    i32 ldt = (n > 1) ? n : 1;
    i32 ldu = ldt;
    i32 ldxa = ldt;
    i32 nn = n * n;
    i32 ldwork = (n > 0) ? ((2 * nn > 3) ? 2 * nn : 3) : 0;
    i32 iwork_size = (nn > 0) ? nn : 1;
    i32 dwork_size = (ldwork > 0) ? ldwork : 1;

    const f64* t = (const f64*)PyArray_DATA(t_array);
    const f64* u = (const f64*)PyArray_DATA(u_array);
    const f64* xa = (const f64*)PyArray_DATA(xa_array);

    i32* iwork = (i32*)malloc(iwork_size * sizeof(i32));
    f64* dwork = (f64*)malloc(dwork_size * sizeof(f64));
    if (iwork == NULL || dwork == NULL) {
        free(iwork);
        free(dwork);
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        Py_DECREF(xa_array);
        return PyErr_NoMemory();
    }

    sb03sy(job, trana, lyapun, n, t, ldt, u, ldu, xa, ldxa, &sepd, &thnorm, iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyObject* result = Py_BuildValue("ddi", sepd, thnorm, info);

    Py_DECREF(t_array);
    Py_DECREF(u_array);
    Py_DECREF(xa_array);

    return result;
}



/* Python wrapper for sb03mw */
PyObject* py_sb03mw(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *t_obj, *b_obj;
    int ltran_int = 0, lupper_int = 1;
    PyArrayObject *t_array, *b_array;
    f64 scale, xnorm;
    i32 info;

    static char* kwlist[] = {"t", "b", "ltran", "lupper", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|pp", kwlist,
                                     &t_obj, &b_obj, &ltran_int, &lupper_int)) {
        return NULL;
    }

    t_array = (PyArrayObject*)PyArray_FROM_OTF(t_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (t_array == NULL) return NULL;

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (b_array == NULL) {
        Py_DECREF(t_array);
        return NULL;
    }

    f64* t = (f64*)PyArray_DATA(t_array);
    f64* b_data = (f64*)PyArray_DATA(b_array);

    npy_intp dims[2] = {2, 2};
    npy_intp strides[2] = {sizeof(f64), 2*sizeof(f64)};
    PyObject* x_array = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (x_array == NULL) {
        Py_DECREF(t_array);
        Py_DECREF(b_array);
        return PyErr_NoMemory();
    }
    f64* x = (f64*)PyArray_DATA((PyArrayObject*)x_array);
    memset(x, 0, 4 * sizeof(f64));

    sb03mw((bool)ltran_int, (bool)lupper_int, t, 2, b_data, 2, &scale, x, 2, &xnorm, &info);

    Py_DECREF(t_array);
    Py_DECREF(b_array);

    return Py_BuildValue("Oddi", x_array, scale, xnorm, info);
}



/* Python wrapper for sb02mu */
PyObject* py_sb02mu(PyObject* self, PyObject* args) {
    const char *dico_str, *hinv_str, *uplo_str;
    i32 n;
    PyObject *a_obj, *g_obj, *q_obj;
    PyArrayObject *a_array = NULL, *g_array = NULL, *q_array = NULL;
    PyArrayObject *s_array = NULL;
    i32 info;

    if (!PyArg_ParseTuple(args, "sssiOOO",
                          &dico_str, &hinv_str, &uplo_str,
                          &n, &a_obj, &g_obj, &q_obj)) {
        return NULL;
    }

    char dico = toupper((unsigned char)dico_str[0]);
    char hinv = toupper((unsigned char)hinv_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);

    bool discr = (dico == 'D');

    if (dico != 'C' && dico != 'D') {
        PyErr_SetString(PyExc_ValueError, "Parameter 1 (DICO) must be 'C' or 'D'");
        return NULL;
    }
    if (discr && hinv != 'D' && hinv != 'I') {
        PyErr_SetString(PyExc_ValueError, "Parameter 2 (HINV) must be 'D' or 'I'");
        return NULL;
    }
    if (uplo != 'U' && uplo != 'L') {
        PyErr_SetString(PyExc_ValueError, "Parameter 3 (UPLO) must be 'U' or 'L'");
        return NULL;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }

    i32 n2 = 2 * n;
    i32 lda = n > 0 ? n : 1;
    i32 ldg = n > 0 ? n : 1;
    i32 ldq = n > 0 ? n : 1;
    i32 lds = n2 > 0 ? n2 : 1;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) goto cleanup;
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    if (n > 0) lda = (i32)PyArray_DIM(a_array, 0);

    g_array = (PyArrayObject*)PyArray_FROM_OTF(g_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!g_array) goto cleanup;
    f64 *g_data = (f64*)PyArray_DATA(g_array);
    if (n > 0) ldg = (i32)PyArray_DIM(g_array, 0);

    q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!q_array) goto cleanup;
    f64 *q_data = (f64*)PyArray_DATA(q_array);
    if (n > 0) ldq = (i32)PyArray_DIM(q_array, 0);

    npy_intp s_dims[2] = {n2, n2};
    npy_intp s_strides[2] = {sizeof(f64), lds * sizeof(f64)};
    f64 *s_data = NULL;
    if (n > 0) {
        s_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, s_dims, NPY_DOUBLE,
                                               s_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!s_array) {
            PyErr_NoMemory();
            goto cleanup;
        }
        s_data = (f64*)PyArray_DATA(s_array);
    }

    i32 ldwork = discr ? (4 * n > 2 ? 4 * n : 2) : 1;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    i32 *iwork = (i32*)malloc((n2 > 0 ? n2 : 1) * sizeof(i32));

    if (!dwork || !iwork) {
        free(dwork);
        free(iwork);
        Py_XDECREF(s_array);
        s_array = NULL;
        PyErr_NoMemory();
        goto cleanup;
    }

    sb02mu(dico_str, hinv_str, uplo_str, n, a_data, lda, g_data, ldg,
           q_data, ldq, s_data, lds, iwork, dwork, ldwork, &info);

    f64 rcond = discr ? dwork[1] : 1.0;

    free(dwork);
    free(iwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    Py_DECREF(a_array);
    Py_DECREF(g_array);
    Py_DECREF(q_array);
    a_array = g_array = q_array = NULL;

    if (info < 0) {
        Py_XDECREF(s_array);
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    if (n == 0) {
        npy_intp zero_dims[2] = {0, 0};
        s_array = (PyArrayObject*)PyArray_ZEROS(2, zero_dims, NPY_DOUBLE, 1);
        if (!s_array) {
            return PyErr_NoMemory();
        }
    }

    PyObject *result = Py_BuildValue("Odi", s_array, rcond, info);
    Py_DECREF(s_array);
    return result;

cleanup:
    Py_XDECREF(a_array);
    Py_XDECREF(g_array);
    Py_XDECREF(q_array);
    return NULL;
}



/* Python wrapper for sb02md */
PyObject* py_sb02md(PyObject* self, PyObject* args) {
    const char *dico_str, *hinv_str, *uplo_str, *scal_str, *sort_str;
    i32 n;
    PyObject *a_obj, *g_obj, *q_obj;
    PyArrayObject *a_array = NULL, *g_array = NULL, *q_array = NULL;
    PyArrayObject *x_array = NULL, *s_array = NULL, *u_array = NULL;
    PyArrayObject *wr_array = NULL, *wi_array = NULL;
    i32 info;
    f64 rcond;

    if (!PyArg_ParseTuple(args, "sssssiOOO",
                          &dico_str, &hinv_str, &uplo_str, &scal_str, &sort_str,
                          &n, &a_obj, &g_obj, &q_obj)) {
        return NULL;
    }

    char dico = toupper((unsigned char)dico_str[0]);
    char hinv = toupper((unsigned char)hinv_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);
    char scal = toupper((unsigned char)scal_str[0]);
    char sort = toupper((unsigned char)sort_str[0]);

    bool discr = (dico == 'D');

    if (dico != 'C' && dico != 'D') {
        PyErr_SetString(PyExc_ValueError, "Parameter 1 (DICO) must be 'C' or 'D'");
        return NULL;
    }
    if (discr && hinv != 'D' && hinv != 'I') {
        PyErr_SetString(PyExc_ValueError, "Parameter 2 (HINV) must be 'D' or 'I'");
        return NULL;
    }
    if (uplo != 'U' && uplo != 'L') {
        PyErr_SetString(PyExc_ValueError, "Parameter 3 (UPLO) must be 'U' or 'L'");
        return NULL;
    }
    if (scal != 'G' && scal != 'N') {
        PyErr_SetString(PyExc_ValueError, "Parameter 4 (SCAL) must be 'G' or 'N'");
        return NULL;
    }
    if (sort != 'S' && sort != 'U') {
        PyErr_SetString(PyExc_ValueError, "Parameter 5 (SORT) must be 'S' or 'U'");
        return NULL;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }

    i32 n2 = 2 * n;
    i32 lda = n > 0 ? n : 1;
    i32 ldg = n > 0 ? n : 1;
    i32 ldq = n > 0 ? n : 1;
    i32 lds = n2 > 0 ? n2 : 1;
    i32 ldu = n2 > 0 ? n2 : 1;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) goto cleanup;
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    if (n > 0) lda = (i32)PyArray_DIM(a_array, 0);

    g_array = (PyArrayObject*)PyArray_FROM_OTF(g_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!g_array) goto cleanup;
    f64 *g_data = (f64*)PyArray_DATA(g_array);
    if (n > 0) ldg = (i32)PyArray_DIM(g_array, 0);

    q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!q_array) goto cleanup;
    f64 *q_data = (f64*)PyArray_DATA(q_array);
    if (n > 0) ldq = (i32)PyArray_DIM(q_array, 0);

    f64 *s_data = NULL;
    f64 *u_data = NULL;
    f64 *wr_data = NULL;
    f64 *wi_data = NULL;
    i32 *iwork = NULL;
    f64 *dwork = NULL;
    i32 *bwork = NULL;

    npy_intp wr_dims[1] = {n2};
    npy_intp s_dims[2] = {n2, n2};
    npy_intp s_strides[2] = {sizeof(f64), lds * sizeof(f64)};
    npy_intp u_dims[2] = {n2, n2};
    npy_intp u_strides[2] = {sizeof(f64), ldu * sizeof(f64)};

    if (n > 0) {
        s_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, s_dims, NPY_DOUBLE,
                                              s_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!s_array) { PyErr_NoMemory(); goto cleanup; }
        s_data = (f64*)PyArray_DATA(s_array);

        u_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE,
                                              u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!u_array) { PyErr_NoMemory(); goto cleanup; }
        u_data = (f64*)PyArray_DATA(u_array);

        wr_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 1, wr_dims, NPY_DOUBLE,
                                               NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
        if (!wr_array) { PyErr_NoMemory(); goto cleanup; }
        wr_data = (f64*)PyArray_DATA(wr_array);

        wi_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 1, wr_dims, NPY_DOUBLE,
                                               NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);
        if (!wi_array) { PyErr_NoMemory(); goto cleanup; }
        wi_data = (f64*)PyArray_DATA(wi_array);
    }

    i32 ldwork = discr ? (6 * n > 3 ? 6 * n : 3) : (6 * n > 2 ? 6 * n : 2);
    dwork = (f64*)malloc(ldwork * sizeof(f64));
    iwork = (i32*)malloc((n2 > 0 ? n2 : 1) * sizeof(i32));
    bwork = (i32*)malloc((n2 > 0 ? n2 : 1) * sizeof(i32));

    if (!dwork || !iwork || !bwork) {
        free(dwork); free(iwork); free(bwork);
        PyErr_NoMemory();
        goto cleanup;
    }

    sb02md(dico_str, hinv_str, uplo_str, scal_str, sort_str, n,
           a_data, lda, g_data, ldg, q_data, ldq, &rcond,
           wr_data, wi_data, s_data, lds, u_data, ldu,
           iwork, dwork, ldwork, bwork, &info);

    free(dwork);
    free(iwork);
    free(bwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(q_array);

    if (info < 0) {
        Py_DECREF(a_array);
        Py_DECREF(g_array);
        Py_DECREF(q_array);
        Py_XDECREF(s_array);
        Py_XDECREF(u_array);
        Py_XDECREF(wr_array);
        Py_XDECREF(wi_array);
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    if (n > 0) {
        npy_intp x_dims[2] = {n, n};
        x_array = (PyArrayObject*)PyArray_SimpleNew(2, x_dims, NPY_DOUBLE);
        if (!x_array) {
            Py_DECREF(a_array);
            Py_DECREF(g_array);
            Py_DECREF(q_array);
            Py_DECREF(s_array);
            Py_DECREF(u_array);
            Py_DECREF(wr_array);
            Py_DECREF(wi_array);
            return PyErr_NoMemory();
        }
        f64 *x_out = (f64*)PyArray_DATA(x_array);
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < n; i++) {
                x_out[i * n + j] = q_data[i + j * ldq];
            }
        }
    } else {
        npy_intp zero_dims[2] = {0, 0};
        x_array = (PyArrayObject*)PyArray_ZEROS(2, zero_dims, NPY_DOUBLE, 0);
        if (!x_array) {
            Py_DECREF(a_array);
            Py_DECREF(g_array);
            Py_DECREF(q_array);
            return PyErr_NoMemory();
        }
    }

    Py_DECREF(a_array);
    Py_DECREF(g_array);
    Py_DECREF(q_array);
    a_array = g_array = q_array = NULL;

    if (n == 0) {
        npy_intp zero_dims1[1] = {0};
        npy_intp zero_dims2[2] = {0, 0};
        wr_array = (PyArrayObject*)PyArray_ZEROS(1, zero_dims1, NPY_DOUBLE, 0);
        wi_array = (PyArrayObject*)PyArray_ZEROS(1, zero_dims1, NPY_DOUBLE, 0);
        s_array = (PyArrayObject*)PyArray_ZEROS(2, zero_dims2, NPY_DOUBLE, 1);
        u_array = (PyArrayObject*)PyArray_ZEROS(2, zero_dims2, NPY_DOUBLE, 1);
        if (!wr_array || !wi_array || !s_array || !u_array) {
            Py_XDECREF(wr_array);
            Py_XDECREF(wi_array);
            Py_XDECREF(s_array);
            Py_XDECREF(u_array);
            Py_DECREF(x_array);
            return PyErr_NoMemory();
        }
    }

    PyObject *result = Py_BuildValue("OdOOOOi", x_array, rcond, wr_array, wi_array,
                                     s_array, u_array, info);
    Py_DECREF(x_array);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);
    Py_DECREF(s_array);
    Py_DECREF(u_array);
    return result;

cleanup:
    Py_XDECREF(a_array);
    Py_XDECREF(g_array);
    Py_XDECREF(q_array);
    return NULL;
}



/* Python wrapper for sb02mt */
PyObject* py_sb02mt(PyObject* self, PyObject* args) {
    const char *jobg_str, *jobl_str, *fact_str, *uplo_str;
    i32 n, m;
    PyObject *a_obj, *b_obj, *q_obj, *r_obj, *l_obj, *g_obj;
    PyArrayObject *a_array = NULL, *b_array = NULL, *q_array = NULL;
    PyArrayObject *r_array = NULL, *l_array = NULL, *g_array = NULL;
    i32 oufact, info;

    if (!PyArg_ParseTuple(args, "ssssiiOOOOOO",
                          &jobg_str, &jobl_str, &fact_str, &uplo_str,
                          &n, &m, &a_obj, &b_obj, &q_obj, &r_obj, &l_obj, &g_obj)) {
        return NULL;
    }

    char jobg = toupper((unsigned char)jobg_str[0]);
    char jobl = toupper((unsigned char)jobl_str[0]);
    char fact = toupper((unsigned char)fact_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);

    bool ljobg = (jobg == 'G');
    bool ljobl = (jobl == 'N');
    bool lfactc = (fact == 'C');
    bool lfactu = (fact == 'U');
    bool lnfact = (!lfactc && !lfactu);

    if (!ljobg && jobg != 'N') {
        PyErr_SetString(PyExc_ValueError, "Parameter 1 (JOBG) must be 'G' or 'N'");
        return NULL;
    }
    if (!ljobl && jobl != 'Z') {
        PyErr_SetString(PyExc_ValueError, "Parameter 2 (JOBL) must be 'Z' or 'N'");
        return NULL;
    }
    if (lnfact && fact != 'N') {
        PyErr_SetString(PyExc_ValueError, "Parameter 3 (FACT) must be 'N', 'C', or 'U'");
        return NULL;
    }
    if (uplo != 'U' && uplo != 'L') {
        PyErr_SetString(PyExc_ValueError, "Parameter 4 (UPLO) must be 'U' or 'L'");
        return NULL;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }
    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be non-negative");
        return NULL;
    }

    i32 lda = 1, ldb = 1, ldq = 1, ldr = 1, ldl = 1, ldg = 1;
    f64 *a_data = NULL, *b_data = NULL, *q_data = NULL;
    f64 *r_data = NULL, *l_data = NULL, *g_data = NULL;

    if (ljobl && a_obj != Py_None) {
        a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!a_array) goto cleanup;
        lda = (i32)PyArray_DIM(a_array, 0);
        a_data = (f64*)PyArray_DATA(a_array);
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) goto cleanup;
    ldb = (i32)PyArray_DIM(b_array, 0);
    b_data = (f64*)PyArray_DATA(b_array);

    if (ljobl && q_obj != Py_None) {
        q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!q_array) goto cleanup;
        ldq = (i32)PyArray_DIM(q_array, 0);
        q_data = (f64*)PyArray_DATA(q_array);
    }

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!r_array) goto cleanup;
    ldr = (i32)PyArray_DIM(r_array, 0);
    r_data = (f64*)PyArray_DATA(r_array);

    if (ljobl && l_obj != Py_None) {
        l_array = (PyArrayObject*)PyArray_FROM_OTF(l_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!l_array) goto cleanup;
        ldl = (i32)PyArray_DIM(l_array, 0);
        l_data = (f64*)PyArray_DATA(l_array);
    }

    if (ljobg && g_obj != Py_None) {
        g_array = (PyArrayObject*)PyArray_FROM_OTF(g_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!g_array) goto cleanup;
        ldg = (i32)PyArray_DIM(g_array, 0);
        g_data = (f64*)PyArray_DATA(g_array);
    }

    i32 ldwork;
    if (lfactc) {
        ldwork = 1;
    } else if (lfactu) {
        ldwork = (ljobg || ljobl) ? (n * m > 1 ? n * m : 1) : 1;
    } else {
        if (ljobg || ljobl) {
            i32 nm = n * m;
            i32 tmp = 3 * m > nm ? 3 * m : nm;
            ldwork = tmp > 2 ? tmp : 2;
        } else {
            ldwork = 3 * m > 2 ? 3 * m : 2;
        }
    }

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    i32 *iwork = (i32*)malloc((m > 0 ? m : 1) * sizeof(i32));
    i32 *ipiv = (i32*)malloc((m > 0 ? m : 1) * sizeof(i32));

    if (!dwork || !iwork || !ipiv) {
        free(dwork); free(iwork); free(ipiv);
        PyErr_NoMemory();
        goto cleanup;
    }

    sb02mt(jobg_str, jobl_str, fact_str, uplo_str,
           n, m, a_data, lda, b_data, ldb, q_data, ldq,
           r_data, ldr, l_data, ldl, ipiv, &oufact, g_data, ldg,
           iwork, dwork, ldwork, &info);

    free(dwork);
    free(iwork);
    free(ipiv);

    if (a_array) PyArray_ResolveWritebackIfCopy(a_array);
    if (b_array) PyArray_ResolveWritebackIfCopy(b_array);
    if (q_array) PyArray_ResolveWritebackIfCopy(q_array);
    if (r_array) PyArray_ResolveWritebackIfCopy(r_array);
    if (l_array) PyArray_ResolveWritebackIfCopy(l_array);
    if (g_array) PyArray_ResolveWritebackIfCopy(g_array);

    if (info < 0) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(q_array);
        Py_XDECREF(r_array);
        Py_XDECREF(l_array);
        Py_XDECREF(g_array);
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    PyObject *result;
    if (ljobg && !ljobl) {
        result = Py_BuildValue("Oii", g_array, oufact, info);
    } else if (!ljobg && ljobl) {
        result = Py_BuildValue("OOOOii", a_array, b_array, q_array, l_array, oufact, info);
    } else if (ljobg && ljobl) {
        result = Py_BuildValue("OOOOOii", a_array, b_array, q_array, l_array, g_array, oufact, info);
    } else {
        result = Py_BuildValue("ii", oufact, info);
    }

    Py_XDECREF(a_array);
    Py_XDECREF(b_array);
    Py_XDECREF(q_array);
    Py_XDECREF(r_array);
    Py_XDECREF(l_array);
    Py_XDECREF(g_array);

    return result;

cleanup:
    Py_XDECREF(a_array);
    Py_XDECREF(b_array);
    Py_XDECREF(q_array);
    Py_XDECREF(r_array);
    Py_XDECREF(l_array);
    Py_XDECREF(g_array);
    return NULL;
}



/* Python wrapper for sb02nd */
PyObject* py_sb02nd(PyObject* self, PyObject* args) {
    const char *dico_str, *fact_str, *uplo_str, *jobl_str;
    i32 n, m, p;
    f64 rnorm;
    PyObject *a_obj, *b_obj, *r_obj, *ipiv_obj, *l_obj, *x_obj;
    PyArrayObject *a_array = NULL, *b_array = NULL, *r_array = NULL;
    PyArrayObject *ipiv_array = NULL, *l_array = NULL, *x_array = NULL;
    PyArrayObject *f_array = NULL;
    i32 oufact[2] = {0, 0};
    i32 info;

    if (!PyArg_ParseTuple(args, "ssssiiiOOOOOOd",
                          &dico_str, &fact_str, &uplo_str, &jobl_str,
                          &n, &m, &p, &a_obj, &b_obj, &r_obj,
                          &ipiv_obj, &l_obj, &x_obj, &rnorm)) {
        return NULL;
    }

    char dico = toupper((unsigned char)dico_str[0]);
    char fact = toupper((unsigned char)fact_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);
    char jobl = toupper((unsigned char)jobl_str[0]);

    bool discr = (dico == 'D');
    bool lfactc = (fact == 'C');
    bool lfactd = (fact == 'D');
    bool lfactu = (fact == 'U');
    bool withl = (jobl == 'N');
    bool lfacta = lfactc || lfactd || lfactu;
    bool lnfact = !lfacta;

    if (!discr && dico != 'C') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'D' or 'C'");
        return NULL;
    }
    if ((lnfact && fact != 'N') || (discr && lfactu)) {
        PyErr_SetString(PyExc_ValueError, "FACT must be 'N', 'D', 'C', or 'U' (U not for discrete)");
        return NULL;
    }
    if (uplo != 'U' && uplo != 'L') {
        PyErr_SetString(PyExc_ValueError, "UPLO must be 'U' or 'L'");
        return NULL;
    }
    if (!withl && jobl != 'Z') {
        PyErr_SetString(PyExc_ValueError, "JOBL must be 'Z' or 'N'");
        return NULL;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }
    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be non-negative");
        return NULL;
    }

    i32 lda = 1, ldb = 1, ldr = 1, ldl = 1, ldx = 1, ldf = 1;
    f64 *a_data = NULL, *b_data = NULL, *r_data = NULL;
    f64 *l_data = NULL, *x_data = NULL, *f_data = NULL;
    i32 *ipiv_data = NULL;

    if (discr && a_obj != Py_None) {
        a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!a_array) goto cleanup;
        lda = (i32)PyArray_DIM(a_array, 0);
        a_data = (f64*)PyArray_DATA(a_array);
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) goto cleanup;
    ldb = (i32)PyArray_DIM(b_array, 0);
    b_data = (f64*)PyArray_DATA(b_array);

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!r_array) goto cleanup;
    ldr = (i32)PyArray_DIM(r_array, 0);
    r_data = (f64*)PyArray_DATA(r_array);

    ipiv_array = (PyArrayObject*)PyArray_FROM_OTF(ipiv_obj, NPY_INT32,
                                                  NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!ipiv_array) goto cleanup;
    ipiv_data = (i32*)PyArray_DATA(ipiv_array);

    if (withl && l_obj != Py_None) {
        l_array = (PyArrayObject*)PyArray_FROM_OTF(l_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!l_array) goto cleanup;
        ldl = (i32)PyArray_DIM(l_array, 0);
        l_data = (f64*)PyArray_DATA(l_array);
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!x_array) goto cleanup;
    ldx = (i32)PyArray_DIM(x_array, 0);
    x_data = (f64*)PyArray_DATA(x_array);

    ldf = m > 1 ? m : 1;
    npy_intp f_dims[2] = {ldf, n};
    npy_intp f_strides[2] = {sizeof(f64), ldf * sizeof(f64)};
    f_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, f_dims, NPY_DOUBLE,
                                          f_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!f_array) {
        PyErr_NoMemory();
        goto cleanup;
    }
    f_data = (f64*)PyArray_DATA(f_array);

    i32 ldwork;
    if (discr) {
        if (lnfact) {
            i32 tmp = 3 * m > n ? 3 * m : n;
            ldwork = tmp > 2 ? tmp : 2;
        } else {
            i32 tmp1 = n + 3 * m + 2;
            i32 tmp2 = 4 * n + 1;
            ldwork = tmp1 > tmp2 ? tmp1 : tmp2;
        }
    } else {
        if (lfactu) {
            ldwork = 2 * m > 2 ? 2 * m : 2;
        } else {
            ldwork = 3 * m > 2 ? 3 * m : 2;
        }
    }
    i32 nm = n * m;
    ldwork = ldwork > nm ? ldwork : nm;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (!dwork) {
        PyErr_NoMemory();
        goto cleanup;
    }

    sb02nd(dico_str, fact_str, uplo_str, jobl_str,
           n, m, p, a_data, lda, b_data, ldb, r_data, ldr,
           ipiv_data, l_data, ldl, x_data, ldx, rnorm,
           f_data, ldf, oufact, dwork, ldwork, &info);

    f64 rcond = dwork[1];
    free(dwork);

    if (a_array) PyArray_ResolveWritebackIfCopy(a_array);
    if (b_array) PyArray_ResolveWritebackIfCopy(b_array);
    if (r_array) PyArray_ResolveWritebackIfCopy(r_array);
    if (ipiv_array) PyArray_ResolveWritebackIfCopy(ipiv_array);
    if (l_array) PyArray_ResolveWritebackIfCopy(l_array);
    if (x_array) PyArray_ResolveWritebackIfCopy(x_array);

    if (info < 0) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(r_array);
        Py_XDECREF(ipiv_array);
        Py_XDECREF(l_array);
        Py_XDECREF(x_array);
        Py_XDECREF(f_array);
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    npy_intp oufact_dims[1] = {2};
    PyArrayObject *oufact_array = (PyArrayObject*)PyArray_SimpleNew(1, oufact_dims, NPY_INT32);
    if (!oufact_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(r_array);
        Py_XDECREF(ipiv_array);
        Py_XDECREF(l_array);
        Py_XDECREF(x_array);
        Py_XDECREF(f_array);
        return NULL;
    }
    ((i32*)PyArray_DATA(oufact_array))[0] = oufact[0];
    ((i32*)PyArray_DATA(oufact_array))[1] = oufact[1];

    PyObject *result = Py_BuildValue("OOOOdi", f_array, r_array, x_array,
                                     oufact_array, rcond, info);

    Py_XDECREF(a_array);
    Py_XDECREF(b_array);
    Py_XDECREF(r_array);
    Py_XDECREF(ipiv_array);
    Py_XDECREF(l_array);
    Py_XDECREF(x_array);
    Py_XDECREF(f_array);
    Py_XDECREF(oufact_array);

    return result;

cleanup:
    Py_XDECREF(a_array);
    Py_XDECREF(b_array);
    Py_XDECREF(r_array);
    Py_XDECREF(ipiv_array);
    Py_XDECREF(l_array);
    Py_XDECREF(x_array);
    Py_XDECREF(f_array);
    return NULL;
}



/* Python wrapper for sb02rd */
PyObject* py_sb02rd(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *job_str, *dico_str, *hinv_str, *trana_str, *uplo_str;
    const char *scal_str, *sort_str, *fact_str, *lyapun_str;
    PyObject *a_obj, *q_obj, *g_obj;
    PyObject *t_obj = Py_None, *v_obj = Py_None;

    static char *kwlist[] = {"job", "dico", "hinv", "trana", "uplo", "scal",
                             "sort", "fact", "lyapun", "A", "Q", "G",
                             "T", "V", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssssssssOOO|OO", kwlist,
            &job_str, &dico_str, &hinv_str, &trana_str, &uplo_str,
            &scal_str, &sort_str, &fact_str, &lyapun_str,
            &a_obj, &q_obj, &g_obj, &t_obj, &v_obj)) {
        return NULL;
    }

    char job = toupper((unsigned char)job_str[0]);
    char dico = toupper((unsigned char)dico_str[0]);
    char hinv = toupper((unsigned char)hinv_str[0]);
    char trana = toupper((unsigned char)trana_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);
    if (job != 'X' && job != 'C' && job != 'E' && job != 'A') {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'X', 'C', 'E', or 'A'");
        return NULL;
    }
    if (dico != 'C' && dico != 'D') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }
    if (dico == 'D' && hinv != 'D' && hinv != 'I') {
        PyErr_SetString(PyExc_ValueError, "HINV must be 'D' or 'I' for discrete-time");
        return NULL;
    }
    if (trana != 'N' && trana != 'T' && trana != 'C') {
        PyErr_SetString(PyExc_ValueError, "TRANA must be 'N', 'T', or 'C'");
        return NULL;
    }
    if (uplo != 'U' && uplo != 'L') {
        PyErr_SetString(PyExc_ValueError, "UPLO must be 'U' or 'L'");
        return NULL;
    }

    PyArrayObject *a_array = NULL, *q_array = NULL, *g_array = NULL;
    PyArrayObject *t_array = NULL, *v_array = NULL;
    PyArrayObject *x_array = NULL, *s_array = NULL;
    PyArrayObject *wr_array = NULL, *wi_array = NULL;
    i32 info = 0;
    f64 sep = 0.0, rcond = 0.0, ferr = 0.0;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 lda = n > 1 ? n : 1;
    i32 n2 = 2 * n;

    q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!q_array) { Py_DECREF(a_array); return NULL; }

    g_array = (PyArrayObject*)PyArray_FROM_OTF(g_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!g_array) { Py_DECREF(a_array); Py_DECREF(q_array); return NULL; }

    i32 ldq = (i32)PyArray_DIM(q_array, 0);
    i32 ldg = (i32)PyArray_DIM(g_array, 0);

    f64 *t_data = NULL, *v_data = NULL;
    i32 ldt = 1, ldv = 1;
    bool t_allocated = false, v_allocated = false;

    if (t_obj != Py_None) {
        t_array = (PyArrayObject*)PyArray_FROM_OTF(t_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!t_array) goto cleanup;
        ldt = (i32)PyArray_DIM(t_array, 0);
        t_data = (f64*)PyArray_DATA(t_array);
    } else if (job != 'X' && n > 0) {
        ldt = n;
        t_data = (f64*)calloc(n * n, sizeof(f64));
        if (!t_data) { PyErr_NoMemory(); goto cleanup; }
        t_allocated = true;
    }

    if (v_obj != Py_None) {
        v_array = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!v_array) goto cleanup;
        ldv = (i32)PyArray_DIM(v_array, 0);
        v_data = (f64*)PyArray_DATA(v_array);
    } else if (job != 'X' && n > 0) {
        ldv = n;
        v_data = (f64*)calloc(n * n, sizeof(f64));
        if (!v_data) { PyErr_NoMemory(); goto cleanup; }
        v_allocated = true;
    }

    npy_intp x_dims[2] = {n, n};
    npy_intp x_strides[2] = {sizeof(f64), n * sizeof(f64)};
    x_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, x_dims, NPY_DOUBLE,
                                          x_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!x_array) { PyErr_NoMemory(); goto cleanup; }
    f64 *x_data = (f64*)PyArray_DATA(x_array);

    npy_intp s_dims[2] = {n2, n2};
    npy_intp s_strides[2] = {sizeof(f64), n2 * sizeof(f64)};
    s_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, s_dims, NPY_DOUBLE,
                                          s_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!s_array) { PyErr_NoMemory(); goto cleanup; }
    f64 *s_data = (f64*)PyArray_DATA(s_array);

    npy_intp wr_dims[1] = {n2};
    wr_array = (PyArrayObject*)PyArray_SimpleNew(1, wr_dims, NPY_DOUBLE);
    if (!wr_array) goto cleanup;

    wi_array = (PyArrayObject*)PyArray_SimpleNew(1, wr_dims, NPY_DOUBLE);
    if (!wi_array) goto cleanup;

    i32 nn = n * n;
    i32 ldwork = 5 + (4 * nn + 8 * n > 1 ? 4 * nn + 8 * n : 1);
    ldwork = ldwork > 1 ? ldwork : 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    i32 *iwork = (i32*)malloc((2 * n > 1 ? 2 * n : 1) * sizeof(i32));
    i32 *bwork = (i32*)malloc((n2 > 1 ? n2 : 1) * sizeof(i32));

    if ((!dwork || !iwork || !bwork) && n > 0) {
        free(dwork); free(iwork); free(bwork);
        PyErr_NoMemory();
        goto cleanup;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *q_data = (f64*)PyArray_DATA(q_array);
    f64 *g_data = (f64*)PyArray_DATA(g_array);
    f64 *wr_data = (f64*)PyArray_DATA(wr_array);
    f64 *wi_data = (f64*)PyArray_DATA(wi_array);
    i32 lds = n2 > 1 ? n2 : 1;
    i32 ldx = n > 1 ? n : 1;

    sb02rd(job_str, dico_str, hinv_str, trana_str, uplo_str,
           scal_str, sort_str, fact_str, lyapun_str,
           n, a_data, lda, t_data, ldt, v_data, ldv,
           g_data, ldg, q_data, ldq, x_data, ldx,
           &sep, &rcond, &ferr, wr_data, wi_data,
           s_data, lds, iwork, dwork, ldwork, bwork, &info);

    free(dwork);
    free(iwork);
    free(bwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(q_array);
    PyArray_ResolveWritebackIfCopy(g_array);
    if (t_array) PyArray_ResolveWritebackIfCopy(t_array);
    if (v_array) PyArray_ResolveWritebackIfCopy(v_array);

    if (t_allocated) free(t_data);
    if (v_allocated) free(v_data);

    if (info < 0) {
        Py_XDECREF(a_array);
        Py_XDECREF(q_array);
        Py_XDECREF(g_array);
        Py_XDECREF(t_array);
        Py_XDECREF(v_array);
        Py_XDECREF(x_array);
        Py_XDECREF(s_array);
        Py_XDECREF(wr_array);
        Py_XDECREF(wi_array);
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    PyObject *result = Py_BuildValue("OdddOOOi", x_array, sep, rcond, ferr,
                                     wr_array, wi_array, s_array, info);

    Py_XDECREF(a_array);
    Py_XDECREF(q_array);
    Py_XDECREF(g_array);
    Py_XDECREF(t_array);
    Py_XDECREF(v_array);
    Py_XDECREF(x_array);
    Py_XDECREF(s_array);
    Py_XDECREF(wr_array);
    Py_XDECREF(wi_array);

    return result;

cleanup:
    if (t_allocated) free(t_data);
    if (v_allocated) free(v_data);
    Py_XDECREF(a_array);
    Py_XDECREF(q_array);
    Py_XDECREF(g_array);
    Py_XDECREF(t_array);
    Py_XDECREF(v_array);
    Py_XDECREF(x_array);
    Py_XDECREF(s_array);
    Py_XDECREF(wr_array);
    Py_XDECREF(wi_array);
    return NULL;
}



/* Python wrapper for sb02ru */
PyObject* py_sb02ru(PyObject* self, PyObject* args) {
    const char *dico_str, *hinv_str, *trana_str, *uplo_str;
    PyObject *a_obj, *g_obj, *q_obj;

    if (!PyArg_ParseTuple(args, "ssssOOO", &dico_str, &hinv_str, &trana_str, &uplo_str,
                         &a_obj, &g_obj, &q_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                             NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *g_array = (PyArrayObject*)PyArray_FROM_OTF(g_obj, NPY_DOUBLE,
                                                             NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (g_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE,
                                                             NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (q_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(g_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 n2 = 2 * n;
    i32 lda = n > 0 ? n : 1;
    i32 ldg = n > 0 ? n : 1;
    i32 ldq = n > 0 ? n : 1;
    i32 lds = n2 > 0 ? n2 : 1;

    bool discr = (dico_str[0] == 'D' || dico_str[0] == 'd');
    i32 ldwork = discr ? (6 * n > 2 ? 6 * n : 2) : 0;
    i32 liwork = discr ? 2 * n : 0;

    npy_intp s_dims[2] = {n2, n2};
    PyArrayObject *s_array = (PyArrayObject*)PyArray_ZEROS(2, s_dims, NPY_DOUBLE, 1);
    if (s_array == NULL) goto cleanup;

    i32 *iwork = NULL;
    f64 *dwork = NULL;
    if (discr) {
        iwork = (i32*)calloc(liwork > 0 ? liwork : 1, sizeof(i32));
        dwork = (f64*)calloc(ldwork > 0 ? ldwork : 1, sizeof(f64));
        if (iwork == NULL || dwork == NULL) {
            PyErr_NoMemory();
            free(iwork);
            free(dwork);
            Py_DECREF(s_array);
            goto cleanup;
        }
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *g_data = (f64*)PyArray_DATA(g_array);
    f64 *q_data = (f64*)PyArray_DATA(q_array);
    f64 *s_data = (f64*)PyArray_DATA(s_array);

    i32 info;
    sb02ru(dico_str, hinv_str, trana_str, uplo_str, n, a_data, lda,
           g_data, ldg, q_data, ldq, s_data, lds, iwork, dwork, ldwork, &info);

    f64 rcond = discr ? dwork[0] : 0.0;
    f64 pivotg = discr ? dwork[1] : 0.0;

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(g_array);
    PyArray_ResolveWritebackIfCopy(q_array);

    PyObject *result = Py_BuildValue("Oddi", s_array, rcond, pivotg, info);

    Py_DECREF(a_array);
    Py_DECREF(g_array);
    Py_DECREF(q_array);
    Py_DECREF(s_array);

    return result;

cleanup:
    Py_DECREF(a_array);
    Py_DECREF(g_array);
    Py_DECREF(q_array);
    return NULL;
}



/* Python wrapper for sb02sd */
PyObject* py_sb02sd(PyObject* self, PyObject* args) {
    char *job, *fact, *trana, *uplo, *lyapun;
    int n;
    PyObject *a_obj, *t_obj, *u_obj, *g_obj, *q_obj, *x_obj;
    PyArrayObject *a_array, *t_array, *u_array, *g_array, *q_array, *x_array;
    f64 sepd = 0.0, rcond = 0.0, ferr = 0.0;
    i32 info;

    if (!PyArg_ParseTuple(args, "sssssiOOOOOO", &job, &fact, &trana, &uplo, &lyapun,
                          &n, &a_obj, &t_obj, &u_obj, &g_obj, &q_obj, &x_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) return NULL;

    t_array = (PyArrayObject*)PyArray_FROM_OTF(t_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (t_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (u_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(t_array);
        return NULL;
    }

    g_array = (PyArrayObject*)PyArray_FROM_OTF(g_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (g_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        return NULL;
    }

    q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (q_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        Py_DECREF(g_array);
        return NULL;
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (x_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        Py_DECREF(g_array);
        Py_DECREF(q_array);
        return NULL;
    }

    i32 lda = (n > 1) ? n : 1;
    i32 ldt = lda;
    i32 ldu = lda;
    i32 ldg = lda;
    i32 ldq = lda;
    i32 ldx = lda;
    i32 nn = n * n;

    char update = (lyapun[0] == 'O' || lyapun[0] == 'o');
    char nofact = (fact[0] == 'N' || fact[0] == 'n');
    char jobc = (job[0] == 'C' || job[0] == 'c');

    i32 lwa = update ? nn : 0;
    i32 ldw;
    if (jobc) {
        ldw = ((2 * nn > 3) ? 2 * nn : 3) + nn;
    } else {
        ldw = ((2 * nn > 3) ? 2 * nn : 3) + 2 * nn;
        if (!update) ldw += n;
    }
    if (nofact) {
        i32 t1 = lwa + 5 * n;
        ldw = (t1 > ldw) ? t1 : ldw;
    }
    i32 ldwork = (ldw > 1) ? ldw : 1;
    i32 iwork_size = (nn > 0) ? nn : 1;

    const f64* a = (const f64*)PyArray_DATA(a_array);
    f64* t = (f64*)PyArray_DATA(t_array);
    f64* u = (f64*)PyArray_DATA(u_array);
    const f64* g = (const f64*)PyArray_DATA(g_array);
    const f64* q = (const f64*)PyArray_DATA(q_array);
    const f64* x = (const f64*)PyArray_DATA(x_array);

    i32* iwork = (i32*)malloc(iwork_size * sizeof(i32));
    f64* dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (iwork == NULL || dwork == NULL) {
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        Py_DECREF(g_array);
        Py_DECREF(q_array);
        Py_DECREF(x_array);
        return PyErr_NoMemory();
    }

    sb02sd(job, fact, trana, uplo, lyapun, n, a, lda, t, ldt, u, ldu, g, ldg, q, ldq,
           x, ldx, &sepd, &rcond, &ferr, iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(t_array);
    PyArray_ResolveWritebackIfCopy(u_array);

    PyObject* result = Py_BuildValue("OOdddi", t_array, u_array, sepd, rcond, ferr, info);

    Py_DECREF(a_array);
    Py_DECREF(t_array);
    Py_DECREF(u_array);
    Py_DECREF(g_array);
    Py_DECREF(q_array);
    Py_DECREF(x_array);

    return result;
}



/* Python wrapper for sb02oy */
PyObject* py_sb02oy(PyObject* self, PyObject* args) {
    const char *type_str, *dico_str, *jobb_str, *fact_str, *uplo_str, *jobl_str, *jobe_str;
    i32 n, m, p;
    f64 tol;
    PyObject *a_obj, *b_obj, *q_obj, *r_obj, *l_obj, *e_obj;

    if (!PyArg_ParseTuple(args, "sssssssiiiOOOOOOd",
                          &type_str, &dico_str, &jobb_str, &fact_str, &uplo_str,
                          &jobl_str, &jobe_str, &n, &m, &p,
                          &a_obj, &b_obj, &q_obj, &r_obj, &l_obj, &e_obj, &tol)) {
        return NULL;
    }

    char type_c = toupper((unsigned char)type_str[0]);
    char dico = toupper((unsigned char)dico_str[0]);
    char jobb = toupper((unsigned char)jobb_str[0]);
    char fact = toupper((unsigned char)fact_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);
    char jobl = toupper((unsigned char)jobl_str[0]);
    char jobe = toupper((unsigned char)jobe_str[0]);

    bool optc = (type_c == 'O');
    bool discr = (dico == 'D');
    bool ljobb = (jobb == 'B');
    bool lfacn = (fact == 'N');
    bool lfacq = (fact == 'C');
    bool lfacr = (fact == 'D');
    bool lfacb = (fact == 'B');
    bool ljobl = (jobl == 'Z');
    bool ljobe = (jobe == 'I');

    if (!optc && type_c != 'S') {
        PyErr_SetString(PyExc_ValueError, "TYPE must be 'O' or 'S'");
        return NULL;
    }
    if (!discr && dico != 'C') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }
    if (!ljobb && jobb != 'G') {
        PyErr_SetString(PyExc_ValueError, "JOBB must be 'B' or 'G'");
        return NULL;
    }
    if (!lfacn && !lfacq && !lfacr && !lfacb) {
        PyErr_SetString(PyExc_ValueError, "FACT must be 'N', 'C', 'D', or 'B'");
        return NULL;
    }
    if (uplo != 'U' && uplo != 'L') {
        PyErr_SetString(PyExc_ValueError, "UPLO must be 'U' or 'L'");
        return NULL;
    }
    if (ljobb && !ljobl && jobl != 'N') {
        PyErr_SetString(PyExc_ValueError, "JOBL must be 'Z' or 'N'");
        return NULL;
    }
    if (!ljobe && jobe != 'N') {
        PyErr_SetString(PyExc_ValueError, "JOBE must be 'I' or 'N'");
        return NULL;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }
    if (ljobb && m < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be non-negative");
        return NULL;
    }

    PyArrayObject *a_array = NULL, *b_array = NULL, *q_array = NULL;
    PyArrayObject *r_array = NULL, *l_array = NULL, *e_array = NULL;
    PyArrayObject *af_out = NULL, *bf_out = NULL;

    i32 lda = 1, ldb = 1, ldq = 1, ldr = 1, ldl = 1, lde = 1;
    f64 *a_data = NULL, *b_data = NULL, *q_data = NULL;
    f64 *r_data = NULL, *l_data = NULL, *e_data = NULL;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!a_array) goto cleanup;
    lda = (i32)PyArray_DIM(a_array, 0);
    if (lda < 1) lda = 1;
    a_data = (f64*)PyArray_DATA(a_array);

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!b_array) goto cleanup;
    ldb = (i32)PyArray_DIM(b_array, 0);
    if (ldb < 1) ldb = 1;
    b_data = (f64*)PyArray_DATA(b_array);

    q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!q_array) goto cleanup;
    ldq = (i32)PyArray_DIM(q_array, 0);
    if (ldq < 1) ldq = 1;
    q_data = (f64*)PyArray_DATA(q_array);

    if (ljobb && r_obj != Py_None) {
        r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
        if (!r_array) goto cleanup;
        ldr = (i32)PyArray_DIM(r_array, 0);
        if (ldr < 1) ldr = 1;
        r_data = (f64*)PyArray_DATA(r_array);
    }

    if (ljobb && l_obj != Py_None) {
        l_array = (PyArrayObject*)PyArray_FROM_OTF(l_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
        if (!l_array) goto cleanup;
        ldl = (i32)PyArray_DIM(l_array, 0);
        if (ldl < 1) ldl = 1;
        l_data = (f64*)PyArray_DATA(l_array);
    }

    if (!ljobe && e_obj != Py_None) {
        e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
        if (!e_array) goto cleanup;
        lde = (i32)PyArray_DIM(e_array, 0);
        if (lde < 1) lde = 1;
        e_data = (f64*)PyArray_DATA(e_array);
    }

    i32 n2 = 2 * n;
    i32 nnm = ljobb ? (n2 + m) : n2;
    i32 ldaf = nnm > 1 ? nnm : 1;
    i32 ldbf;
    bool need_bf;

    if (!ljobb && !discr && ljobe) {
        need_bf = false;
        ldbf = 1;
    } else {
        need_bf = true;
        ldbf = nnm > 1 ? nnm : 1;
    }

    i32 af_cols = ljobb ? nnm : n2;
    npy_intp af_buf_size = ldaf * af_cols;
    PyObject *af_buffer = PyArray_ZEROS(1, &af_buf_size, NPY_DOUBLE, 0);
    if (!af_buffer) {
        PyErr_NoMemory();
        goto cleanup;
    }
    f64 *af_data = (f64*)PyArray_DATA((PyArrayObject*)af_buffer);

    npy_intp af_out_dims[2] = {n2, n2};
    npy_intp af_out_strides[2] = {sizeof(f64), ldaf * sizeof(f64)};
    af_out = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, af_out_dims, NPY_DOUBLE,
                                                        af_out_strides, af_data, 0, NPY_ARRAY_FARRAY, NULL);
    if (!af_out) {
        Py_DECREF(af_buffer);
        PyErr_NoMemory();
        goto cleanup;
    }
    PyArray_SetBaseObject(af_out, af_buffer);

    f64 *bf_data = NULL;
    if (need_bf) {
        npy_intp bf_buf_size = ldbf * n2;
        PyObject *bf_buffer = PyArray_ZEROS(1, &bf_buf_size, NPY_DOUBLE, 0);
        if (!bf_buffer) {
            Py_DECREF(af_out);
            PyErr_NoMemory();
            goto cleanup;
        }
        bf_data = (f64*)PyArray_DATA((PyArrayObject*)bf_buffer);

        npy_intp bf_out_dims[2] = {n2, n2};
        npy_intp bf_out_strides[2] = {sizeof(f64), ldbf * sizeof(f64)};
        bf_out = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, bf_out_dims, NPY_DOUBLE,
                                             bf_out_strides, bf_data, 0, NPY_ARRAY_FARRAY, NULL);
        if (!bf_out) {
            Py_DECREF(bf_buffer);
            Py_DECREF(af_out);
            PyErr_NoMemory();
            goto cleanup;
        }
        PyArray_SetBaseObject(bf_out, bf_buffer);
    }

    i32 ldwork;
    if (ljobb) {
        i32 req1 = nnm > 3*m ? nnm : 3*m;
        ldwork = req1 > 1 ? req1 : 1;
    } else {
        ldwork = 1;
    }

    i32 liwork = ljobb ? (m > 1 ? m : 1) : 1;

    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    i32 *iwork = (i32*)calloc(liwork, sizeof(i32));
    if (!dwork || !iwork) {
        free(dwork);
        free(iwork);
        Py_DECREF(af_out);
        Py_XDECREF(bf_out);
        PyErr_NoMemory();
        goto cleanup;
    }

    i32 info;
    sb02oy(type_str, dico_str, jobb_str, fact_str, uplo_str, jobl_str, jobe_str,
           n, m, p, a_data, lda, b_data, ldb, q_data, ldq,
           r_data, ldr, l_data, ldl, e_data, lde,
           af_data, ldaf, bf_data, ldbf, tol, iwork, dwork, ldwork, &info);

    f64 rcond = ljobb ? dwork[1] : 0.0;

    free(dwork);
    free(iwork);

    if (info < 0) {
        Py_DECREF(af_out);
        Py_XDECREF(bf_out);
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(q_array);
        Py_XDECREF(r_array);
        Py_XDECREF(l_array);
        Py_XDECREF(e_array);
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    PyObject *result;
    if (!need_bf) {
        result = Py_BuildValue("Oi", af_out, info);
    } else if (ljobb) {
        result = Py_BuildValue("OOdi", af_out, bf_out, rcond, info);
        Py_DECREF(bf_out);
    } else {
        result = Py_BuildValue("OOi", af_out, bf_out, info);
        Py_DECREF(bf_out);
    }

    Py_DECREF(af_out);
    Py_XDECREF(a_array);
    Py_XDECREF(b_array);
    Py_XDECREF(q_array);
    Py_XDECREF(r_array);
    Py_XDECREF(l_array);
    Py_XDECREF(e_array);

    return result;

cleanup:
    Py_XDECREF(a_array);
    Py_XDECREF(b_array);
    Py_XDECREF(q_array);
    Py_XDECREF(r_array);
    Py_XDECREF(l_array);
    Py_XDECREF(e_array);
    return NULL;
}



/* Python wrapper for sb02od */
PyObject* py_sb02od(PyObject* self, PyObject* args) {
    const char *dico_str, *jobb_str, *fact_str, *uplo_str, *jobl_str, *sort_str;
    i32 n, m, p;
    f64 tol;
    PyObject *a_obj, *b_obj, *q_obj, *r_obj, *l_obj;

    if (!PyArg_ParseTuple(args, "ssssssiiiOOOOOd",
                          &dico_str, &jobb_str, &fact_str, &uplo_str, &jobl_str,
                          &sort_str, &n, &m, &p,
                          &a_obj, &b_obj, &q_obj, &r_obj, &l_obj, &tol)) {
        return NULL;
    }

    char dico = toupper((unsigned char)dico_str[0]);
    char jobb = toupper((unsigned char)jobb_str[0]);
    char fact = toupper((unsigned char)fact_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);
    char jobl = toupper((unsigned char)jobl_str[0]);
    char sort = toupper((unsigned char)sort_str[0]);

    bool discr = (dico == 'D');
    bool ljobb = (jobb == 'B');
    bool lfacn = (fact == 'N');
    bool lfacq = (fact == 'C');
    bool lfacr = (fact == 'D');
    bool lfacb = (fact == 'B');
    bool ljobl = (jobl == 'Z');

    if (!discr && dico != 'C') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }
    if (!ljobb && jobb != 'G') {
        PyErr_SetString(PyExc_ValueError, "JOBB must be 'B' or 'G'");
        return NULL;
    }
    if (!lfacn && !lfacq && !lfacr && !lfacb) {
        PyErr_SetString(PyExc_ValueError, "FACT must be 'N', 'C', 'D', or 'B'");
        return NULL;
    }
    if (uplo != 'U' && uplo != 'L') {
        PyErr_SetString(PyExc_ValueError, "UPLO must be 'U' or 'L'");
        return NULL;
    }
    if (ljobb && !ljobl && jobl != 'N') {
        PyErr_SetString(PyExc_ValueError, "JOBL must be 'Z' or 'N'");
        return NULL;
    }
    if (sort != 'S' && sort != 'U') {
        PyErr_SetString(PyExc_ValueError, "SORT must be 'S' or 'U'");
        return NULL;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }
    if (ljobb && m < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be non-negative");
        return NULL;
    }

    PyArrayObject *a_array = NULL, *b_array = NULL, *q_array = NULL;
    PyArrayObject *r_array = NULL, *l_array = NULL;
    PyArrayObject *x_array = NULL, *s_array = NULL, *t_array = NULL, *u_array = NULL;
    PyArrayObject *alfar_array = NULL, *alfai_array = NULL, *beta_array = NULL;

    i32 lda = 1, ldb = 1, ldq = 1, ldr = 1, ldl = 1;
    f64 *a_data = NULL, *b_data = NULL, *q_data = NULL;
    f64 *r_data = NULL, *l_data = NULL;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!a_array) return NULL;
    lda = (i32)PyArray_DIM(a_array, 0);
    if (lda < 1) lda = 1;
    a_data = (f64*)PyArray_DATA(a_array);

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!b_array) goto cleanup;
    ldb = (i32)PyArray_DIM(b_array, 0);
    if (ldb < 1) ldb = 1;
    b_data = (f64*)PyArray_DATA(b_array);

    q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!q_array) goto cleanup;
    ldq = (i32)PyArray_DIM(q_array, 0);
    if (ldq < 1) ldq = 1;
    q_data = (f64*)PyArray_DATA(q_array);

    if (ljobb && r_obj != Py_None) {
        r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!r_array) goto cleanup;
        ldr = (i32)PyArray_DIM(r_array, 0);
        if (ldr < 1) ldr = 1;
        r_data = (f64*)PyArray_DATA(r_array);
    }

    if (ljobb && l_obj != Py_None) {
        l_array = (PyArrayObject*)PyArray_FROM_OTF(l_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!l_array) goto cleanup;
        ldl = (i32)PyArray_DIM(l_array, 0);
        if (ldl < 1) ldl = 1;
        l_data = (f64*)PyArray_DATA(l_array);
    }

    i32 nn = 2 * n;
    i32 nnm = ljobb ? (nn + m) : nn;
    i32 ldx = n > 1 ? n : 1;
    i32 lds = nnm > 1 ? nnm : 1;
    i32 ldt = nnm > 1 ? nnm : 1;
    i32 ldu = nn > 1 ? nn : 1;

    npy_intp x_dims[2] = {n, n};
    npy_intp x_strides[2] = {sizeof(f64), ldx * sizeof(f64)};
    x_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, x_dims, NPY_DOUBLE,
                                                         x_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!x_array) { PyErr_NoMemory(); goto cleanup; }
    f64 *x_data = (f64*)PyArray_DATA(x_array);
    memset(x_data, 0, ldx * n * sizeof(f64));

    i32 s_ncols = ljobb ? nnm : nn;
    npy_intp s_dims[2] = {lds, s_ncols};
    npy_intp s_strides[2] = {sizeof(f64), lds * sizeof(f64)};
    s_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, s_dims, NPY_DOUBLE,
                                                         s_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!s_array) { PyErr_NoMemory(); goto cleanup; }
    f64 *s_data = (f64*)PyArray_DATA(s_array);
    memset(s_data, 0, lds * s_ncols * sizeof(f64));

    npy_intp t_dims[2] = {ldt, nn};
    npy_intp t_strides[2] = {sizeof(f64), ldt * sizeof(f64)};
    t_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, t_dims, NPY_DOUBLE,
                                                         t_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!t_array) { PyErr_NoMemory(); goto cleanup; }
    f64 *t_data = (f64*)PyArray_DATA(t_array);
    memset(t_data, 0, ldt * nn * sizeof(f64));

    npy_intp u_dims[2] = {nn, nn};
    npy_intp u_strides[2] = {sizeof(f64), ldu * sizeof(f64)};
    u_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE,
                                                         u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!u_array) { PyErr_NoMemory(); goto cleanup; }
    f64 *u_data = (f64*)PyArray_DATA(u_array);
    memset(u_data, 0, ldu * nn * sizeof(f64));

    npy_intp eig_dims[1] = {nn};
    alfar_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 1, eig_dims, NPY_DOUBLE,
                                                             NULL, NULL, 0, 0, NULL);
    if (!alfar_array) { PyErr_NoMemory(); goto cleanup; }
    f64 *alfar = (f64*)PyArray_DATA(alfar_array);
    memset(alfar, 0, nn * sizeof(f64));

    alfai_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 1, eig_dims, NPY_DOUBLE,
                                                             NULL, NULL, 0, 0, NULL);
    if (!alfai_array) { PyErr_NoMemory(); goto cleanup; }
    f64 *alfai = (f64*)PyArray_DATA(alfai_array);
    memset(alfai, 0, nn * sizeof(f64));

    beta_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 1, eig_dims, NPY_DOUBLE,
                                                            NULL, NULL, 0, 0, NULL);
    if (!beta_array) { PyErr_NoMemory(); goto cleanup; }
    f64 *beta_arr = (f64*)PyArray_DATA(beta_array);
    memset(beta_arr, 0, nn * sizeof(f64));

    i32 ldwork;
    i32 req1 = 14*n + 23;
    i32 req2 = 16*n;
    i32 req3 = ljobb ? (nn + m > 3*m ? nn + m : 3*m) : 1;
    ldwork = req1 > req2 ? req1 : req2;
    ldwork = ldwork > req3 ? ldwork : req3;
    if (ldwork < 3) ldwork = 3;

    i32 liwork = ljobb ? (m > nn ? m : nn) : nn;
    if (liwork < 1) liwork = 1;

    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    i32 *iwork = (i32*)calloc(liwork, sizeof(i32));
    if (!dwork || !iwork) {
        free(dwork); free(iwork);
        PyErr_NoMemory();
        goto cleanup;
    }

    f64 rcond;
    i32 info;
    sb02od(dico_str, jobb_str, fact_str, uplo_str, jobl_str, sort_str,
           n, m, p, a_data, lda, b_data, ldb, q_data, ldq,
           r_data, ldr, l_data, ldl, &rcond, x_data, ldx,
           alfar, alfai, beta_arr, s_data, lds, t_data, ldt, u_data, ldu,
           tol, iwork, dwork, ldwork, &info);

    free(dwork);
    free(iwork);

    if (q_array) PyArray_ResolveWritebackIfCopy(q_array);
    if (r_array) PyArray_ResolveWritebackIfCopy(r_array);
    if (l_array) PyArray_ResolveWritebackIfCopy(l_array);

    if (info < 0) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(q_array);
        Py_XDECREF(r_array);
        Py_XDECREF(l_array);
        Py_DECREF(x_array);
        Py_DECREF(s_array);
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        Py_DECREF(alfar_array);
        Py_DECREF(alfai_array);
        Py_DECREF(beta_array);
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    PyObject *result = Py_BuildValue("OdOOOOOOi", x_array, rcond, alfar_array, alfai_array,
                                     beta_array, s_array, t_array, u_array, info);

    Py_DECREF(x_array);
    Py_DECREF(alfar_array);
    Py_DECREF(alfai_array);
    Py_DECREF(beta_array);
    Py_DECREF(s_array);
    Py_DECREF(t_array);
    Py_DECREF(u_array);

    Py_XDECREF(a_array);
    Py_XDECREF(b_array);
    Py_XDECREF(q_array);
    Py_XDECREF(r_array);
    Py_XDECREF(l_array);

    return result;

cleanup:
    Py_XDECREF(a_array);
    Py_XDECREF(b_array);
    Py_XDECREF(q_array);
    Py_XDECREF(r_array);
    Py_XDECREF(l_array);
    Py_XDECREF(x_array);
    Py_XDECREF(s_array);
    Py_XDECREF(t_array);
    Py_XDECREF(u_array);
    Py_XDECREF(alfar_array);
    Py_XDECREF(alfai_array);
    Py_XDECREF(beta_array);
    return NULL;
}



/* Python wrapper for sb03od */
PyObject* py_sb03od(PyObject* self, PyObject* args) {
    const char *dico_str, *fact_str, *trans_str;
    PyObject *a_obj, *b_obj, *q_obj = NULL;
    
    if (!PyArg_ParseTuple(args, "sssOO|O", &dico_str, &fact_str, &trans_str,
                         &a_obj, &b_obj, &q_obj)) {
        return NULL;
    }
    
    // Convert inputs to arrays
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
    i32 lda = n > 0 ? n : 1;
    i32 ldq = n > 0 ? n : 1;

    // Initialize q_array early (before any gotos)
    PyArrayObject *q_array = NULL;

    // Get B dimensions based on transpose flag
    i32 m, ldb;
    if (trans_str[0] == 'N' || trans_str[0] == 'n') {
        m = (i32)PyArray_DIM(b_array, 0);
        ldb = m > 0 ? m : 1;
        if (PyArray_DIM(b_array, 1) != n) {
            PyErr_SetString(PyExc_ValueError, "B shape mismatch for trans='N'");
            goto cleanup;
        }
    } else {
        m = (i32)PyArray_DIM(b_array, 1);
        ldb = n > 0 ? n : 1;
        if (PyArray_DIM(b_array, 0) != n) {
            PyErr_SetString(PyExc_ValueError, "B shape mismatch for trans='T'");
            goto cleanup;
        }
    }

    // Handle Q matrix for fact='F'
    bool nofact = (fact_str[0] == 'N' || fact_str[0] == 'n');
    if (!nofact) {
        if (q_obj == NULL) {
            PyErr_SetString(PyExc_ValueError, "Q matrix required when fact='F'");
            goto cleanup;
        }
        q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE,
                                                  NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (q_array == NULL) goto cleanup;
    } else {
        // Allocate Q for output
        npy_intp q_dims[2] = {n, n};
        q_array = (PyArrayObject*)PyArray_ZEROS(2, q_dims, NPY_DOUBLE, 1);
        if (q_array == NULL) goto cleanup;
    }
    
    // Allocate output arrays
    npy_intp wr_dims[1] = {n};
    PyArrayObject *wr_array = (PyArrayObject*)PyArray_ZEROS(1, wr_dims, NPY_DOUBLE, 0);
    if (wr_array == NULL) goto cleanup;
    
    PyArrayObject *wi_array = (PyArrayObject*)PyArray_ZEROS(1, wr_dims, NPY_DOUBLE, 0);
    if (wi_array == NULL) {
        Py_DECREF(wr_array);
        goto cleanup;
    }
    
    // Workspace query
    f64 dwork_query;
    i32 ldwork = -1;
    i32 info;
    f64 scale;
    
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);  
    f64 *q_data = (f64*)PyArray_DATA(q_array);
    f64 *wr_data = (f64*)PyArray_DATA(wr_array);
    f64 *wi_data = (f64*)PyArray_DATA(wi_array);
    
    sb03od(dico_str, fact_str, trans_str, n, m, a_data, lda, q_data, ldq,
           b_data, ldb, &scale, wr_data, wi_data, &dwork_query, ldwork, &info);
    
    if (info < 0 && info != -16) {
        PyErr_Format(PyExc_ValueError, "sb03od parameter error: info=%d", info);
        Py_DECREF(wr_array);
        Py_DECREF(wi_array);
        goto cleanup;
    }
    
    // Allocate workspace
    ldwork = (i32)dwork_query;
    if (m == 0) ldwork = 1;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        PyErr_NoMemory();
        Py_DECREF(wr_array);
        Py_DECREF(wi_array);
        goto cleanup;
    }
    
    // Actual computation
    sb03od(dico_str, fact_str, trans_str, n, m, a_data, lda, q_data, ldq,
           b_data, ldb, &scale, wr_data, wi_data, dwork, ldwork, &info);
    
    free(dwork);
    
    // Resolve writebacks
    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    if (!nofact && q_array) {
        PyArray_ResolveWritebackIfCopy(q_array);
    }
    
    // Build return value: (u, q, wr, wi, scale, info)
    // Note: B array now contains U (upper triangular Cholesky factor)
    PyObject *result = Py_BuildValue("OOOOdi", b_array, q_array, wr_array, wi_array, scale, info);
    
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(q_array);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);
    
    return result;

cleanup:
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    if (q_array) Py_DECREF(q_array);
    return NULL;
}



/* Python wrapper for sb10dd - H-infinity controller for discrete-time system */
PyObject* py_sb10dd(PyObject* self, PyObject* args, PyObject* kwds) {
    static char *kwlist[] = {"n", "m", "np", "ncon", "nmeas", "gamma", "a", "b", "c", "d", "tol", NULL};
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    int n, m, np_val, ncon, nmeas;
    double gamma, tol = 0.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiidOOOO|d", kwlist,
                                      &n, &m, &np_val, &ncon, &nmeas, &gamma,
                                      &a_obj, &b_obj, &c_obj, &d_obj, &tol)) {
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

    i32 n_ = (i32)n;
    i32 m_ = (i32)m;
    i32 np_ = (i32)np_val;
    i32 ncon_ = (i32)ncon;
    i32 nmeas_ = (i32)nmeas;

    i32 m1 = m_ - ncon_;
    i32 m2 = ncon_;
    i32 np1 = np_ - nmeas_;
    i32 np2 = nmeas_;

    i32 lda = n_ > 1 ? n_ : 1;
    i32 ldb = n_ > 1 ? n_ : 1;
    i32 ldc = np_ > 1 ? np_ : 1;
    i32 ldd = np_ > 1 ? np_ : 1;
    i32 ldak = n_ > 1 ? n_ : 1;
    i32 ldbk = n_ > 1 ? n_ : 1;
    i32 ldck = m2 > 1 ? m2 : 1;
    i32 lddk = m2 > 1 ? m2 : 1;
    i32 ldx = n_ > 1 ? n_ : 1;
    i32 ldz = n_ > 1 ? n_ : 1;

    i32 q = m1;
    if (m2 > q) q = m2;
    if (np1 > q) q = np1;
    if (np2 > q) q = np2;

    i32 iwb_ws = (n_ + np1 + 1) * (n_ + m2) +
                 ((3 * (n_ + m2) + n_ + np1) > (5 * (n_ + m2)) ?
                  (3 * (n_ + m2) + n_ + np1) : (5 * (n_ + m2)));
    i32 iwc_ws = (n_ + np2) * (n_ + m1 + 1) +
                 ((3 * (n_ + np2) + n_ + m1) > (5 * (n_ + np2)) ?
                  (3 * (n_ + np2) + n_ + m1) : (5 * (n_ + np2)));
    i32 iwd_ws = 13 * n_ * n_ + 2 * m_ * m_ + n_ * (8 * m_ + np2) + m1 * (m2 + np2) + 6 * n_ +
                 ((14 * n_ + 23) > 16 * n_ ?
                  ((14 * n_ + 23) > (2 * n_ + m_) ?
                   ((14 * n_ + 23) > 3 * m_ ? (14 * n_ + 23) : 3 * m_) :
                   ((2 * n_ + m_) > 3 * m_ ? (2 * n_ + m_) : 3 * m_)) :
                  (16 * n_ > (2 * n_ + m_) ?
                   (16 * n_ > 3 * m_ ? 16 * n_ : 3 * m_) :
                   ((2 * n_ + m_) > 3 * m_ ? (2 * n_ + m_) : 3 * m_)));
    i32 iwg_ws = 13 * n_ * n_ + m_ * m_ + (8 * n_ + m_ + m2 + 2 * np2) * (m2 + np2) + 6 * n_ +
                 n_ * (m_ + np2) +
                 ((14 * n_ + 23) > 16 * n_ ?
                  ((14 * n_ + 23) > (2 * n_ + m2 + np2) ?
                   ((14 * n_ + 23) > 3 * (m2 + np2) ? (14 * n_ + 23) : 3 * (m2 + np2)) :
                   ((2 * n_ + m2 + np2) > 3 * (m2 + np2) ? (2 * n_ + m2 + np2) : 3 * (m2 + np2))) :
                  (16 * n_ > (2 * n_ + m2 + np2) ?
                   (16 * n_ > 3 * (m2 + np2) ? 16 * n_ : 3 * (m2 + np2)) :
                   ((2 * n_ + m2 + np2) > 3 * (m2 + np2) ? (2 * n_ + m2 + np2) : 3 * (m2 + np2))));

    i32 ldwork = iwb_ws;
    if (iwc_ws > ldwork) ldwork = iwc_ws;
    if (iwd_ws > ldwork) ldwork = iwd_ws;
    if (iwg_ws > ldwork) ldwork = iwg_ws;
    if (ldwork < 1) ldwork = 1;

    i32 liwork = 2 * (m2 > n_ ? m2 : n_);
    if (m_ > liwork) liwork = m_;
    if (m2 + np2 > liwork) liwork = m2 + np2;
    if (n_ * n_ > liwork) liwork = n_ * n_;
    if (liwork < 1) liwork = 1;

    npy_intp ak_dims[2] = {n_, n_};
    npy_intp ak_strides[2] = {sizeof(f64), n_ * sizeof(f64)};
    PyObject *ak_array = PyArray_New(&PyArray_Type, 2, ak_dims, NPY_DOUBLE,
                                      ak_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!ak_array) { Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array); return PyErr_NoMemory(); }
    f64 *ak = (f64*)PyArray_DATA((PyArrayObject*)ak_array);

    npy_intp bk_dims[2] = {n_, nmeas_};
    npy_intp bk_strides[2] = {sizeof(f64), n_ * sizeof(f64)};
    PyObject *bk_array = PyArray_New(&PyArray_Type, 2, bk_dims, NPY_DOUBLE,
                                      bk_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!bk_array) { Py_DECREF(ak_array); Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array); return PyErr_NoMemory(); }
    f64 *bk = (f64*)PyArray_DATA((PyArrayObject*)bk_array);

    npy_intp ck_dims[2] = {ncon_, n_};
    npy_intp ck_strides[2] = {sizeof(f64), ncon_ * sizeof(f64)};
    PyObject *ck_array = PyArray_New(&PyArray_Type, 2, ck_dims, NPY_DOUBLE,
                                      ck_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!ck_array) { Py_DECREF(bk_array); Py_DECREF(ak_array); Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array); return PyErr_NoMemory(); }
    f64 *ck = (f64*)PyArray_DATA((PyArrayObject*)ck_array);

    npy_intp dk_dims[2] = {ncon_, nmeas_};
    npy_intp dk_strides[2] = {sizeof(f64), ncon_ * sizeof(f64)};
    PyObject *dk_array = PyArray_New(&PyArray_Type, 2, dk_dims, NPY_DOUBLE,
                                      dk_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!dk_array) { Py_DECREF(ck_array); Py_DECREF(bk_array); Py_DECREF(ak_array); Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array); return PyErr_NoMemory(); }
    f64 *dk = (f64*)PyArray_DATA((PyArrayObject*)dk_array);

    npy_intp x_dims[2] = {n_, n_};
    npy_intp x_strides[2] = {sizeof(f64), n_ * sizeof(f64)};
    PyObject *x_array = PyArray_New(&PyArray_Type, 2, x_dims, NPY_DOUBLE,
                                     x_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!x_array) { Py_DECREF(dk_array); Py_DECREF(ck_array); Py_DECREF(bk_array); Py_DECREF(ak_array); Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array); return PyErr_NoMemory(); }
    f64 *x = (f64*)PyArray_DATA((PyArrayObject*)x_array);

    npy_intp z_dims[2] = {n_, n_};
    npy_intp z_strides[2] = {sizeof(f64), n_ * sizeof(f64)};
    PyObject *z_array = PyArray_New(&PyArray_Type, 2, z_dims, NPY_DOUBLE,
                                     z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!z_array) { Py_DECREF(x_array); Py_DECREF(dk_array); Py_DECREF(ck_array); Py_DECREF(bk_array); Py_DECREF(ak_array); Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array); return PyErr_NoMemory(); }
    f64 *z = (f64*)PyArray_DATA((PyArrayObject*)z_array);

    npy_intp rcond_dims[1] = {8};
    PyObject *rcond_array = PyArray_New(&PyArray_Type, 1, rcond_dims, NPY_DOUBLE,
                                         NULL, NULL, 0, 0, NULL);
    if (!rcond_array) { Py_DECREF(z_array); Py_DECREF(x_array); Py_DECREF(dk_array); Py_DECREF(ck_array); Py_DECREF(bk_array); Py_DECREF(ak_array); Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array); return PyErr_NoMemory(); }
    f64 *rcond = (f64*)PyArray_DATA((PyArrayObject*)rcond_array);

    i32 *iwork = (i32*)malloc(liwork * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork); free(dwork);
        Py_DECREF(rcond_array); Py_DECREF(z_array); Py_DECREF(x_array);
        Py_DECREF(dk_array); Py_DECREF(ck_array); Py_DECREF(bk_array); Py_DECREF(ak_array);
        Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array);
        return PyErr_NoMemory();
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);

    i32 info = 0;

    sb10dd(n_, m_, np_, ncon_, nmeas_, gamma, a_data, lda, b_data, ldb,
           c_data, ldc, d_data, ldd, ak, ldak, bk, ldbk, ck, ldck,
           dk, lddk, x, ldx, z, ldz, rcond, tol, iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyObject *result = Py_BuildValue("OOOOOOOi", ak_array, bk_array, ck_array,
                                      dk_array, x_array, z_array, rcond_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    return result;
}



/* Python wrapper for sb10fd - H-infinity (sub)optimal controller for continuous-time system */
PyObject* py_sb10fd(PyObject* self, PyObject* args, PyObject* kwds) {
    static char *kwlist[] = {"n", "m", "np", "ncon", "nmeas", "gamma", "a", "b", "c", "d", "tol", NULL};
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    int n, m, np_val, ncon, nmeas;
    double gamma, tol = 0.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiidOOOO|d", kwlist,
                                      &n, &m, &np_val, &ncon, &nmeas, &gamma,
                                      &a_obj, &b_obj, &c_obj, &d_obj, &tol)) {
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

    i32 n_ = (i32)n;
    i32 m_ = (i32)m;
    i32 np_ = (i32)np_val;
    i32 ncon_ = (i32)ncon;
    i32 nmeas_ = (i32)nmeas;

    i32 m1 = m_ - ncon_;
    i32 m2 = ncon_;
    i32 np1 = np_ - nmeas_;
    i32 np2 = nmeas_;

    i32 lda = n_ > 1 ? n_ : 1;
    i32 ldb = n_ > 1 ? n_ : 1;
    i32 ldc = np_ > 1 ? np_ : 1;
    i32 ldd = np_ > 1 ? np_ : 1;
    i32 ldak = n_ > 1 ? n_ : 1;
    i32 ldbk = n_ > 1 ? n_ : 1;
    i32 ldck = m2 > 1 ? m2 : 1;
    i32 lddk = m2 > 1 ? m2 : 1;

    i32 nd1 = np1 - m2;
    if (nd1 < 0) nd1 = 0;
    i32 nd2 = m1 - np2;
    if (nd2 < 0) nd2 = 0;

    i32 lw1 = (n_ + np1 + 1) * (n_ + m2) +
              ((3*(n_ + m2) + n_ + np1) > (5*(n_ + m2)) ?
               (3*(n_ + m2) + n_ + np1) : (5*(n_ + m2)));

    i32 lw2 = (n_ + np2) * (n_ + m1 + 1) +
              ((3*(n_ + np2) + n_ + m1) > (5*(n_ + np2)) ?
               (3*(n_ + np2) + n_ + m1) : (5*(n_ + np2)));

    i32 n_max_m1 = n_ > m1 ? n_ : m1;
    i32 lw3_inner = np1 * n_max_m1;
    i32 lw3_a = lw3_inner > (3*m2 + np1) ? lw3_inner : (3*m2 + np1);
    i32 lw3_b = lw3_a > (5*m2) ? lw3_a : (5*m2);
    i32 lw3 = m2 + np1*np1 + lw3_b;

    i32 n_max_np1 = n_ > np1 ? n_ : np1;
    i32 lw4 = np2 + m1*m1 +
              ((n_max_np1 * m1) > (3*np2 + m1) ?
               ((n_max_np1 * m1) > (5*np2) ? (n_max_np1 * m1) : (5*np2)) :
               ((3*np2 + m1) > (5*np2) ? (3*np2 + m1) : (5*np2)));

    i32 inner5_1 = 10*n_*n_ + 12*n_ + 5;
    i32 inner5_2a = n_*m_ > inner5_1 ? n_*m_ : inner5_1;
    i32 inner5_2b = n_*np_ > inner5_1 ? n_*np_ : inner5_1;
    i32 inner5_3a = 3*n_*n_ + inner5_2a;
    i32 inner5_3b = 3*n_*n_ + inner5_2b;
    i32 inner5_4a = 2*m1 > inner5_3a ? 2*m1 : inner5_3a;
    i32 inner5_4b = 2*np1 > inner5_3b ? 2*np1 : inner5_3b;
    i32 inner5_5a = m_*m_ + inner5_4a;
    i32 inner5_5b = np_*np_ + inner5_4b;
    i32 inner5_6 = inner5_5a > inner5_5b ? inner5_5a : inner5_5b;
    i32 lw5 = 2*n_*n_ + n_*(m_ + np_) + (1 > inner5_6 ? 1 : inner5_6);

    i32 np2_or_n = np2 > n_ ? np2 : n_;
    i32 inner6_1a = 2*nd1 > ((nd1 + nd2)*np2) ? 2*nd1 : ((nd1 + nd2)*np2);
    i32 inner6_1b = nd1*nd1 + inner6_1a;
    i32 inner6_2a = 2*nd2 > (nd2*m2) ? 2*nd2 : (nd2*m2);
    i32 inner6_2b = nd2*nd2 + inner6_2a;
    i32 inner6_3 = inner6_1b > inner6_2b ? inner6_1b : inner6_2b;
    i32 inner6_4 = 3*n_ > inner6_3 ? 3*n_ : inner6_3;
    i32 inner6_5a = m2*m2 + 3*m2;
    i32 inner6_5b = np2 * (2*np2 + m2 + np2_or_n);
    i32 inner6_5 = inner6_5a > inner6_5b ? inner6_5a : inner6_5b;
    i32 inner6_6 = m2*np2 + inner6_5;
    i32 inner6_7 = 2*n_*m2 > inner6_6 ? 2*n_*m2 : inner6_6;
    i32 inner6_8 = n_*(2*np2 + m2) + inner6_7;
    i32 inner6_9 = inner6_4 > inner6_8 ? inner6_4 : inner6_8;
    i32 inner6_10 = m2*np2 + np2*np2 + m2*m2 + inner6_9;
    i32 lw6 = 2*n_*n_ + n_*(m_ + np_) + (1 > inner6_10 ? 1 : inner6_10);

    i32 lw_max = lw1;
    if (lw2 > lw_max) lw_max = lw2;
    if (lw3 > lw_max) lw_max = lw3;
    if (lw4 > lw_max) lw_max = lw4;
    if (lw5 > lw_max) lw_max = lw5;
    if (lw6 > lw_max) lw_max = lw6;
    if (1 > lw_max) lw_max = 1;

    i32 ldwork = n_*m_ + np_*(n_ + m_) + m2*m2 + np2*np2 + lw_max;
    if (ldwork < 1) ldwork = 1;

    i32 liwork = 2 * n_;
    if (2 * m1 > liwork) liwork = 2 * m1;
    if (2 * np1 > liwork) liwork = 2 * np1;
    if (2 * m2 > liwork) liwork = 2 * m2;
    if (n_ * n_ > liwork) liwork = n_ * n_;
    if (liwork < 1) liwork = 1;

    i32 lbwork = 2 * n_;
    if (lbwork < 1) lbwork = 1;

    npy_intp ak_dims[2] = {n_, n_};
    npy_intp ak_strides[2] = {sizeof(f64), n_ * sizeof(f64)};
    PyObject *ak_array = PyArray_New(&PyArray_Type, 2, ak_dims, NPY_DOUBLE,
                                      ak_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!ak_array) { Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array); return PyErr_NoMemory(); }
    f64 *ak = (f64*)PyArray_DATA((PyArrayObject*)ak_array);

    npy_intp bk_dims[2] = {n_, nmeas_};
    npy_intp bk_strides[2] = {sizeof(f64), n_ * sizeof(f64)};
    PyObject *bk_array = PyArray_New(&PyArray_Type, 2, bk_dims, NPY_DOUBLE,
                                      bk_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!bk_array) { Py_DECREF(ak_array); Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array); return PyErr_NoMemory(); }
    f64 *bk = (f64*)PyArray_DATA((PyArrayObject*)bk_array);

    npy_intp ck_dims[2] = {ncon_, n_};
    npy_intp ck_strides[2] = {sizeof(f64), ncon_ * sizeof(f64)};
    PyObject *ck_array = PyArray_New(&PyArray_Type, 2, ck_dims, NPY_DOUBLE,
                                      ck_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!ck_array) { Py_DECREF(bk_array); Py_DECREF(ak_array); Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array); return PyErr_NoMemory(); }
    f64 *ck = (f64*)PyArray_DATA((PyArrayObject*)ck_array);

    npy_intp dk_dims[2] = {ncon_, nmeas_};
    npy_intp dk_strides[2] = {sizeof(f64), ncon_ * sizeof(f64)};
    PyObject *dk_array = PyArray_New(&PyArray_Type, 2, dk_dims, NPY_DOUBLE,
                                      dk_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!dk_array) { Py_DECREF(ck_array); Py_DECREF(bk_array); Py_DECREF(ak_array); Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array); return PyErr_NoMemory(); }
    f64 *dk = (f64*)PyArray_DATA((PyArrayObject*)dk_array);

    npy_intp rcond_dims[1] = {4};
    PyObject *rcond_array = PyArray_New(&PyArray_Type, 1, rcond_dims, NPY_DOUBLE,
                                         NULL, NULL, 0, 0, NULL);
    if (!rcond_array) { Py_DECREF(dk_array); Py_DECREF(ck_array); Py_DECREF(bk_array); Py_DECREF(ak_array); Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array); return PyErr_NoMemory(); }
    f64 *rcond = (f64*)PyArray_DATA((PyArrayObject*)rcond_array);

    i32 *iwork = (i32*)malloc(liwork * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    i32 *bwork = (i32*)malloc(lbwork * sizeof(i32));

    if (!iwork || !dwork || !bwork) {
        free(iwork); free(dwork); free(bwork);
        Py_DECREF(rcond_array); Py_DECREF(dk_array); Py_DECREF(ck_array); Py_DECREF(bk_array); Py_DECREF(ak_array);
        Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array);
        return PyErr_NoMemory();
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);

    i32 info = 0;

    sb10fd(n_, m_, np_, ncon_, nmeas_, gamma, a_data, lda, b_data, ldb,
           c_data, ldc, d_data, ldd, ak, ldak, bk, ldbk, ck, ldck,
           dk, lddk, rcond, tol, iwork, dwork, ldwork, bwork, &info);

    free(iwork);
    free(dwork);
    free(bwork);

    PyObject *result = Py_BuildValue("OOOOOi", ak_array, bk_array, ck_array,
                                      dk_array, rcond_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    return result;
}



/* Python wrapper for sb10jd */
PyObject* py_sb10jd(PyObject* self, PyObject* args, PyObject* kwds) {
    static char *kwlist[] = {"a", "b", "c", "d", "e", "n", "m", "np", NULL};
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *e_obj;
    int n_arg = -1, m_arg = -1, np_arg = -1;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOO|iii", kwlist,
                                      &a_obj, &b_obj, &c_obj, &d_obj, &e_obj,
                                      &n_arg, &m_arg, &np_arg)) {
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

    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(
        e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!e_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);
    (void)PyArray_DIMS(d_array);

    i32 n = (n_arg == -1) ? (i32)a_dims[0] : (i32)n_arg;
    i32 m = (m_arg == -1) ? (PyArray_NDIM(b_array) >= 2 ? (i32)b_dims[1] : 0) : (i32)m_arg;
    i32 np = (np_arg == -1) ? (i32)c_dims[0] : (i32)np_arg;

    if (n < 0 || m < 0 || np < 0) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(e_array);
        PyErr_SetString(PyExc_ValueError, "n, m, np must be non-negative");
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = np > 0 ? np : 1;
    i32 ldd = np > 0 ? np : 1;
    i32 lde = n > 0 ? n : 1;

    i32 tmp = n + m + np;
    i32 tmp2 = tmp > 5 ? tmp : 5;
    i32 ldwork = 2 * n * n + 2 * n + n * tmp2;
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (!dwork) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(e_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);

    i32 nsys = 0;
    i32 info = 0;

    sb10jd(n, m, np, a_data, lda, b_data, ldb, c_data, ldc,
           d_data, ldd, e_data, lde, &nsys, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(e_array);

    Py_DECREF(e_array);

    PyObject *result = Py_BuildValue("OOOOii", a_array, b_array, c_array, d_array, nsys, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    return result;
}



/* Python wrapper for sb10pd - Normalize system for H-infinity controller design */
PyObject* py_sb10pd(PyObject* self, PyObject* args, PyObject* kwds) {
    static char *kwlist[] = {"n", "m", "np", "ncon", "nmeas", "a", "b", "c", "d", "tu", "ty", "tol", NULL};
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *tu_obj, *ty_obj;
    int n, m, np_val, ncon, nmeas;
    double tol = 0.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiOOOOOO|d", kwlist,
                                      &n, &m, &np_val, &ncon, &nmeas,
                                      &a_obj, &b_obj, &c_obj, &d_obj,
                                      &tu_obj, &ty_obj, &tol)) {
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

    PyArrayObject *tu_array = (PyArrayObject*)PyArray_FROM_OTF(
        tu_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!tu_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }

    PyArrayObject *ty_array = (PyArrayObject*)PyArray_FROM_OTF(
        ty_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!ty_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(tu_array);
        return NULL;
    }

    i32 n_ = (i32)n;
    i32 m_ = (i32)m;
    i32 np_ = (i32)np_val;
    i32 ncon_ = (i32)ncon;
    i32 nmeas_ = (i32)nmeas;

    i32 m1 = m_ - ncon_;
    i32 m2 = ncon_;
    i32 np1 = np_ - nmeas_;
    i32 np2 = nmeas_;

    i32 lda = n_ > 1 ? n_ : 1;
    i32 ldb = n_ > 1 ? n_ : 1;
    i32 ldc = np_ > 1 ? np_ : 1;
    i32 ldd = np_ > 1 ? np_ : 1;
    i32 ldtu = m2 > 1 ? m2 : 1;
    i32 ldty = np2 > 1 ? np2 : 1;

    i32 nm2 = n_ + m2;
    i32 nnp1 = n_ + np1;
    i32 nnp2 = n_ + np2;
    i32 nm1 = n_ + m1;

    i32 lw1_mat = (nnp1 + 1) * nm2;
    i32 lw1_svd1 = 3 * nm2 + nnp1;
    i32 lw1_svd2 = 5 * nm2;
    i32 lw1 = lw1_mat + (lw1_svd1 > lw1_svd2 ? lw1_svd1 : lw1_svd2);

    i32 lw2_mat = nnp2 * (nm1 + 1);
    i32 lw2_svd1 = 3 * nnp2 + nm1;
    i32 lw2_svd2 = 5 * nnp2;
    i32 lw2 = lw2_mat + (lw2_svd1 > lw2_svd2 ? lw2_svd1 : lw2_svd2);

    i32 lw3_np1m1 = np1 > m1 ? np1 : m1;
    i32 lw3_work1 = np1 * (n_ > lw3_np1m1 ? n_ : lw3_np1m1);
    i32 lw3_work2 = 3 * m2 + np1;
    i32 lw3_work3 = 5 * m2;
    i32 lw3_max = lw3_work1;
    if (lw3_work2 > lw3_max) lw3_max = lw3_work2;
    if (lw3_work3 > lw3_max) lw3_max = lw3_work3;
    i32 lw3 = m2 + np1 * np1 + lw3_max;

    i32 lw4_nnp1 = n_ > np1 ? n_ : np1;
    i32 lw4_work1 = lw4_nnp1 * m1;
    i32 lw4_work2 = 3 * np2 + m1;
    i32 lw4_work3 = 5 * np2;
    i32 lw4_max = lw4_work1;
    if (lw4_work2 > lw4_max) lw4_max = lw4_work2;
    if (lw4_work3 > lw4_max) lw4_max = lw4_work3;
    i32 lw4 = np2 + m1 * m1 + lw4_max;

    i32 ldwork = lw1;
    if (lw2 > ldwork) ldwork = lw2;
    if (lw3 > ldwork) ldwork = lw3;
    if (lw4 > ldwork) ldwork = lw4;
    if (ldwork < 1) ldwork = 1;

    npy_intp rcond_dims[1] = {2};
    PyObject *rcond_array = PyArray_New(&PyArray_Type, 1, rcond_dims, NPY_DOUBLE,
                                         NULL, NULL, 0, 0, NULL);
    if (!rcond_array) {
        Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array);
        Py_DECREF(d_array); Py_DECREF(tu_array); Py_DECREF(ty_array);
        return PyErr_NoMemory();
    }
    f64 *rcond = (f64*)PyArray_DATA((PyArrayObject*)rcond_array);

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (!dwork) {
        Py_DECREF(rcond_array);
        Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array);
        Py_DECREF(d_array); Py_DECREF(tu_array); Py_DECREF(ty_array);
        return PyErr_NoMemory();
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *tu_data = (f64*)PyArray_DATA(tu_array);
    f64 *ty_data = (f64*)PyArray_DATA(ty_array);

    i32 info = 0;

    sb10pd(n_, m_, np_, ncon_, nmeas_, a_data, lda, b_data, ldb,
           c_data, ldc, d_data, ldd, tu_data, ldtu, ty_data, ldty,
           rcond, tol, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(tu_array);
    PyArray_ResolveWritebackIfCopy(ty_array);

    PyObject *result = Py_BuildValue("OOOOOOi", b_array, c_array, d_array,
                                      tu_array, ty_array, rcond_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(tu_array);
    Py_DECREF(ty_array);

    return result;
}



/* Python wrapper for sb10hd - H2 optimal n-state controller */
PyObject* py_sb10hd(PyObject* self, PyObject* args, PyObject* kwds) {
    static char *kwlist[] = {"n", "m", "np", "ncon", "nmeas",
                              "a", "b", "c", "d", "tol", NULL};
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    int n, m, np_val, ncon, nmeas;
    double tol;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiOOOOd", kwlist,
                                      &n, &m, &np_val, &ncon, &nmeas,
                                      &a_obj, &b_obj, &c_obj, &d_obj, &tol)) {
        return NULL;
    }

    i32 n_ = (i32)n;
    i32 m_ = (i32)m;
    i32 np_ = (i32)np_val;
    i32 ncon_ = (i32)ncon;
    i32 nmeas_ = (i32)nmeas;
    i32 m1 = m_ - ncon_;
    i32 m2 = ncon_;
    i32 np1 = np_ - nmeas_;
    i32 np2 = nmeas_;

    if (n_ < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be >= 0");
        return NULL;
    }
    if (m_ < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be >= 0");
        return NULL;
    }
    if (np_ < 0) {
        PyErr_SetString(PyExc_ValueError, "np must be >= 0");
        return NULL;
    }
    if (ncon_ < 0 || ncon_ > m_ || m2 > np1) {
        PyErr_SetString(PyExc_ValueError, "ncon must be in [0, m] and ncon <= np-nmeas");
        return NULL;
    }
    if (nmeas_ < 0 || nmeas_ > np_ || np2 > m1) {
        PyErr_SetString(PyExc_ValueError, "nmeas must be in [0, np] and nmeas <= m-ncon");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!b_array) { Py_DECREF(a_array); return NULL; }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!c_array) { Py_DECREF(a_array); Py_DECREF(b_array); return NULL; }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!d_array) { Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); return NULL; }

    i32 lda = n_ > 1 ? n_ : 1;
    i32 ldb = n_ > 1 ? n_ : 1;
    i32 ldc = np_ > 1 ? np_ : 1;
    i32 ldd = np_ > 1 ? np_ : 1;
    i32 ldak = n_ > 1 ? n_ : 1;
    i32 ldbk = n_ > 1 ? n_ : 1;
    i32 ldck = m2 > 1 ? m2 : 1;
    i32 lddk = m2 > 1 ? m2 : 1;

    const f64 *a_data = (const f64*)PyArray_DATA(a_array);
    const f64 *b_data = (const f64*)PyArray_DATA(b_array);
    const f64 *c_data = (const f64*)PyArray_DATA(c_array);
    const f64 *d_data = (const f64*)PyArray_DATA(d_array);

    npy_intp ak_dims[2] = {n_ > 0 ? n_ : 0, n_ > 0 ? n_ : 0};
    npy_intp bk_dims[2] = {n_ > 0 ? n_ : 0, np2 > 0 ? np2 : 0};
    npy_intp ck_dims[2] = {m2 > 0 ? m2 : 0, n_ > 0 ? n_ : 0};
    npy_intp dk_dims[2] = {m2 > 0 ? m2 : 0, np2 > 0 ? np2 : 0};
    npy_intp rcond_dims[1] = {4};

    npy_intp ak_strides[2] = {sizeof(f64), ldak * sizeof(f64)};
    npy_intp bk_strides[2] = {sizeof(f64), ldbk * sizeof(f64)};
    npy_intp ck_strides[2] = {sizeof(f64), ldck * sizeof(f64)};
    npy_intp dk_strides[2] = {sizeof(f64), lddk * sizeof(f64)};

    PyObject *ak_array = PyArray_New(&PyArray_Type, 2, ak_dims, NPY_DOUBLE,
                                      ak_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *bk_array = PyArray_New(&PyArray_Type, 2, bk_dims, NPY_DOUBLE,
                                      bk_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *ck_array = PyArray_New(&PyArray_Type, 2, ck_dims, NPY_DOUBLE,
                                      ck_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *dk_array = PyArray_New(&PyArray_Type, 2, dk_dims, NPY_DOUBLE,
                                      dk_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *rcond_array = PyArray_New(&PyArray_Type, 1, rcond_dims, NPY_DOUBLE,
                                         NULL, NULL, 0, NPY_ARRAY_CARRAY, NULL);

    if (!ak_array || !bk_array || !ck_array || !dk_array || !rcond_array) {
        Py_XDECREF(ak_array); Py_XDECREF(bk_array); Py_XDECREF(ck_array);
        Py_XDECREF(dk_array); Py_XDECREF(rcond_array);
        Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *ak = (f64*)PyArray_DATA((PyArrayObject*)ak_array);
    f64 *bk = (f64*)PyArray_DATA((PyArrayObject*)bk_array);
    f64 *ck = (f64*)PyArray_DATA((PyArrayObject*)ck_array);
    f64 *dk = (f64*)PyArray_DATA((PyArrayObject*)dk_array);
    f64 *rcond = (f64*)PyArray_DATA((PyArrayObject*)rcond_array);

    i32 iwork_size = 2 * n_ > n_ * n_ ? 2 * n_ : n_ * n_;
    iwork_size = iwork_size > 1 ? iwork_size : 1;
    i32 *iwork = (i32*)malloc(iwork_size * sizeof(i32));

    i32 q = m1 > m2 ? m1 : m2;
    q = q > np1 ? q : np1;
    q = q > np2 ? q : np2;
    i32 ldwork = 2 * q * (3 * q + 2 * n_) +
                 ((q * (q + (n_ > 5 ? n_ : 5) + 1)) > (n_ * (14 * n_ + 12 + 2 * q) + 5) ?
                  (q * (q + (n_ > 5 ? n_ : 5) + 1)) : (n_ * (14 * n_ + 12 + 2 * q) + 5));
    ldwork = ldwork > 1 ? ldwork : 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    i32 bwork_size = 2 * n_ > 1 ? 2 * n_ : 1;
    i32 *bwork = (i32*)malloc(bwork_size * sizeof(i32));

    if (!iwork || !dwork || !bwork) {
        free(iwork); free(dwork); free(bwork);
        Py_DECREF(ak_array); Py_DECREF(bk_array); Py_DECREF(ck_array);
        Py_DECREF(dk_array); Py_DECREF(rcond_array);
        Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info = 0;
    sb10hd(n_, m_, np_, ncon_, nmeas_,
           a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           ak, ldak, bk, ldbk, ck, ldck, dk, lddk,
           rcond, tol, iwork, dwork, ldwork, bwork, &info);

    free(iwork);
    free(dwork);
    free(bwork);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    PyObject *result = Py_BuildValue("OOOOOi", ak_array, bk_array, ck_array, dk_array,
                                      rcond_array, info);

    Py_DECREF(ak_array);
    Py_DECREF(bk_array);
    Py_DECREF(ck_array);
    Py_DECREF(dk_array);
    Py_DECREF(rcond_array);

    return result;
}



/* Python wrapper for sb10wd - H2 controller from state feedback and output injection */
PyObject* py_sb10wd(PyObject* self, PyObject* args, PyObject* kwds) {
    static char *kwlist[] = {"n", "m", "np", "ncon", "nmeas",
                              "a", "b", "c", "d", "f", "h", "tu", "ty", NULL};
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *f_obj, *h_obj, *tu_obj, *ty_obj;
    int n, m, np_val, ncon, nmeas;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiOOOOOOOO", kwlist,
                                      &n, &m, &np_val, &ncon, &nmeas,
                                      &a_obj, &b_obj, &c_obj, &d_obj,
                                      &f_obj, &h_obj, &tu_obj, &ty_obj)) {
        return NULL;
    }

    i32 n_ = (i32)n;
    i32 m_ = (i32)m;
    i32 np_ = (i32)np_val;
    i32 ncon_ = (i32)ncon;
    i32 nmeas_ = (i32)nmeas;
    i32 m2 = ncon_;
    i32 np2 = nmeas_;

    if (n_ < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be >= 0");
        return NULL;
    }
    if (m_ < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be >= 0");
        return NULL;
    }
    if (np_ < 0) {
        PyErr_SetString(PyExc_ValueError, "np must be >= 0");
        return NULL;
    }
    if (ncon_ < 0 || ncon_ > m_) {
        PyErr_SetString(PyExc_ValueError, "ncon must be in [0, m]");
        return NULL;
    }
    if (nmeas_ < 0 || nmeas_ > np_) {
        PyErr_SetString(PyExc_ValueError, "nmeas must be in [0, np]");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!b_array) { Py_DECREF(a_array); return NULL; }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!c_array) { Py_DECREF(a_array); Py_DECREF(b_array); return NULL; }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!d_array) { Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); return NULL; }

    PyArrayObject *f_array = (PyArrayObject*)PyArray_FROM_OTF(
        f_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!f_array) { Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array); return NULL; }

    PyArrayObject *h_array = (PyArrayObject*)PyArray_FROM_OTF(
        h_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!h_array) { Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array); Py_DECREF(f_array); return NULL; }

    PyArrayObject *tu_array = (PyArrayObject*)PyArray_FROM_OTF(
        tu_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!tu_array) { Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array); Py_DECREF(f_array); Py_DECREF(h_array); return NULL; }

    PyArrayObject *ty_array = (PyArrayObject*)PyArray_FROM_OTF(
        ty_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!ty_array) { Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array); Py_DECREF(f_array); Py_DECREF(h_array); Py_DECREF(tu_array); return NULL; }

    i32 lda = n_ > 1 ? n_ : 1;
    i32 ldb = n_ > 1 ? n_ : 1;
    i32 ldc = np_ > 1 ? np_ : 1;
    i32 ldd = np_ > 1 ? np_ : 1;
    i32 ldf = m2 > 1 ? m2 : 1;
    i32 ldh = n_ > 1 ? n_ : 1;
    i32 ldtu = m2 > 1 ? m2 : 1;
    i32 ldty = np2 > 1 ? np2 : 1;
    i32 ldak = n_ > 1 ? n_ : 1;
    i32 ldbk = n_ > 1 ? n_ : 1;
    i32 ldck = m2 > 1 ? m2 : 1;
    i32 lddk = m2 > 1 ? m2 : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *f_data = (f64*)PyArray_DATA(f_array);
    f64 *h_data = (f64*)PyArray_DATA(h_array);
    f64 *tu_data = (f64*)PyArray_DATA(tu_array);
    f64 *ty_data = (f64*)PyArray_DATA(ty_array);

    npy_intp ak_dims[2] = {n_ > 0 ? n_ : 0, n_ > 0 ? n_ : 0};
    npy_intp bk_dims[2] = {n_ > 0 ? n_ : 0, np2 > 0 ? np2 : 0};
    npy_intp ck_dims[2] = {m2 > 0 ? m2 : 0, n_ > 0 ? n_ : 0};
    npy_intp dk_dims[2] = {m2 > 0 ? m2 : 0, np2 > 0 ? np2 : 0};

    npy_intp ak_strides[2] = {sizeof(f64), ldak * sizeof(f64)};
    npy_intp bk_strides[2] = {sizeof(f64), ldbk * sizeof(f64)};
    npy_intp ck_strides[2] = {sizeof(f64), ldck * sizeof(f64)};
    npy_intp dk_strides[2] = {sizeof(f64), lddk * sizeof(f64)};

    PyObject *ak_array = PyArray_New(&PyArray_Type, 2, ak_dims, NPY_DOUBLE,
                                      ak_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *bk_array = PyArray_New(&PyArray_Type, 2, bk_dims, NPY_DOUBLE,
                                      bk_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *ck_array = PyArray_New(&PyArray_Type, 2, ck_dims, NPY_DOUBLE,
                                      ck_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *dk_array = PyArray_New(&PyArray_Type, 2, dk_dims, NPY_DOUBLE,
                                      dk_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (!ak_array || !bk_array || !ck_array || !dk_array) {
        Py_XDECREF(ak_array); Py_XDECREF(bk_array); Py_XDECREF(ck_array); Py_XDECREF(dk_array);
        Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array);
        Py_DECREF(f_array); Py_DECREF(h_array); Py_DECREF(tu_array); Py_DECREF(ty_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *ak = (f64*)PyArray_DATA((PyArrayObject*)ak_array);
    f64 *bk = (f64*)PyArray_DATA((PyArrayObject*)bk_array);
    f64 *ck = (f64*)PyArray_DATA((PyArrayObject*)ck_array);
    f64 *dk = (f64*)PyArray_DATA((PyArrayObject*)dk_array);

    i32 info = 0;
    sb10wd(n_, m_, np_, ncon_, nmeas_,
           a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           f_data, ldf, h_data, ldh, tu_data, ldtu, ty_data, ldty,
           ak, ldak, bk, ldbk, ck, ldck, dk, lddk, &info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(f_array);
    Py_DECREF(h_array);
    Py_DECREF(tu_array);
    Py_DECREF(ty_array);

    PyObject *result = Py_BuildValue("OOOOi", ak_array, bk_array, ck_array, dk_array, info);

    Py_DECREF(ak_array);
    Py_DECREF(bk_array);
    Py_DECREF(ck_array);
    Py_DECREF(dk_array);

    return result;
}



/* Python wrapper for sb04nv */
PyObject* py_sb04nv(PyObject* self, PyObject* args) {
    const char *abschr_str, *ul_str;
    i32 indx;
    PyObject *c_obj, *ab_obj;

    if (!PyArg_ParseTuple(args, "ssiOO", &abschr_str, &ul_str, &indx, &c_obj, &ab_obj)) {
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (c_array == NULL) {
        return NULL;
    }

    PyArrayObject *ab_array = (PyArrayObject*)PyArray_FROM_OTF(ab_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (ab_array == NULL) {
        Py_DECREF(c_array);
        return NULL;
    }

    npy_intp *c_dims = PyArray_DIMS(c_array);
    i32 n = (i32)c_dims[0];
    i32 m = (i32)c_dims[1];
    i32 ldc = (n > 1) ? n : 1;

    npy_intp *ab_dims = PyArray_DIMS(ab_array);
    i32 ldab = (ab_dims[0] > 1) ? (i32)ab_dims[0] : 1;

    bool is_b = (abschr_str[0] == 'B' || abschr_str[0] == 'b');
    i32 d_len = is_b ? 2 * n : 2 * m;
    if (d_len == 0) d_len = 1;

    npy_intp d_dims[1] = {d_len};
    PyObject *d_array = PyArray_SimpleNew(1, d_dims, NPY_DOUBLE);
    if (d_array == NULL) {
        Py_DECREF(c_array);
        Py_DECREF(ab_array);
        PyErr_NoMemory();
        return NULL;
    }
    f64 *d_data = (f64*)PyArray_DATA((PyArrayObject*)d_array);

    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *ab_data = (f64*)PyArray_DATA(ab_array);

    sb04nv(abschr_str, ul_str, n, m, c_data, ldc, indx, ab_data, ldab, d_data);

    Py_DECREF(c_array);
    Py_DECREF(ab_array);
    return d_array;
}



/* Python wrapper for sb04nw */
PyObject* py_sb04nw(PyObject* self, PyObject* args) {
    const char *abschr_str, *ul_str;
    i32 indx;
    PyObject *c_obj, *ab_obj;

    if (!PyArg_ParseTuple(args, "ssiOO", &abschr_str, &ul_str, &indx, &c_obj, &ab_obj)) {
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (c_array == NULL) {
        return NULL;
    }

    PyArrayObject *ab_array = (PyArrayObject*)PyArray_FROM_OTF(ab_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (ab_array == NULL) {
        Py_DECREF(c_array);
        return NULL;
    }

    npy_intp *c_dims = PyArray_DIMS(c_array);
    i32 n = (i32)c_dims[0];
    i32 m = (i32)c_dims[1];
    i32 ldc = (n > 1) ? n : 1;

    npy_intp *ab_dims = PyArray_DIMS(ab_array);
    i32 ldab = (ab_dims[0] > 1) ? (i32)ab_dims[0] : 1;

    bool is_b = (abschr_str[0] == 'B' || abschr_str[0] == 'b');
    i32 d_len = is_b ? n : m;
    if (d_len == 0) d_len = 1;

    npy_intp d_dims[1] = {d_len};
    PyObject *d_array = PyArray_SimpleNew(1, d_dims, NPY_DOUBLE);
    if (d_array == NULL) {
        Py_DECREF(c_array);
        Py_DECREF(ab_array);
        PyErr_NoMemory();
        return NULL;
    }
    f64 *d_data = (f64*)PyArray_DATA((PyArrayObject*)d_array);

    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *ab_data = (f64*)PyArray_DATA(ab_array);

    sb04nw(abschr_str, ul_str, n, m, c_data, ldc, indx, ab_data, ldab, d_data);

    Py_DECREF(c_array);
    Py_DECREF(ab_array);
    return d_array;
}



/* Python wrapper for sb04nx */
PyObject* py_sb04nx(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *rc_str, *ul_str;
    f64 lambd1, lambd2, lambd3, lambd4, tol;
    PyObject *a_obj, *d_obj;

    static char *kwlist[] = {"rc", "ul", "a", "d", "lambd1", "lambd2", "lambd3", "lambd4", "tol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssOOddddd", kwlist,
                                      &rc_str, &ul_str, &a_obj, &d_obj,
                                      &lambd1, &lambd2, &lambd3, &lambd4, &tol)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 m = (i32)a_dims[0];
    i32 lda = (m > 1) ? m : 1;
    i32 m2 = 2 * m;
    if (m2 == 0) m2 = 1;
    i32 lddwor = (m2 > 1) ? m2 : 1;

    i32 *iwork = (i32*)malloc((m2 > 0 ? m2 : 1) * sizeof(i32));
    f64 *dwork = (f64*)malloc((lddwor * (m2 + 3) > 0 ? lddwor * (m2 + 3) : 1) * sizeof(f64));
    if (iwork == NULL || dwork == NULL) {
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(d_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    i32 info = 0;

    sb04nx(rc_str, ul_str, m, a_data, lda, lambd1, lambd2, lambd3, lambd4,
           d_data, tol, iwork, dwork, lddwor, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(d_array);
    Py_DECREF(a_array);

    PyObject *result = Py_BuildValue("Oi", d_array, info);
    Py_DECREF(d_array);
    return result;
}



/* Python wrapper for sb04ny */
PyObject* py_sb04ny(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *rc_str, *ul_str;
    f64 lambda, tol;
    PyObject *a_obj, *d_obj;

    static char *kwlist[] = {"rc", "ul", "a", "d", "lambda_", "tol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssOOdd", kwlist,
                                      &rc_str, &ul_str, &a_obj, &d_obj,
                                      &lambda, &tol)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 m = (i32)a_dims[0];
    i32 lda = (m > 1) ? m : 1;
    i32 lddwor = (m > 1) ? m : 1;

    i32 *iwork = (i32*)malloc((m > 0 ? m : 1) * sizeof(i32));
    f64 *dwork = (f64*)malloc((lddwor * (m + 3) > 0 ? lddwor * (m + 3) : 1) * sizeof(f64));
    if (iwork == NULL || dwork == NULL) {
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(d_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    i32 info = 0;

    sb04ny(rc_str, ul_str, m, a_data, lda, lambda, d_data, tol, iwork, dwork, lddwor, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(d_array);
    Py_DECREF(a_array);

    PyObject *result = Py_BuildValue("Oi", d_array, info);
    Py_DECREF(d_array);
    return result;
}



/* Python wrapper for sb04mr */
PyObject* py_sb04mr(PyObject* self, PyObject* args) {
    i32 m;
    PyObject *d_obj;

    if (!PyArg_ParseTuple(args, "iO", &m, &d_obj)) {
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (d_array == NULL) {
        return NULL;
    }

    i32 ipr_len = (m > 0) ? 2 * m : 1;
    npy_intp ipr_dims[1] = {ipr_len};
    PyArrayObject *ipr_array = (PyArrayObject*)PyArray_SimpleNew(1, ipr_dims, NPY_INT32);
    if (ipr_array == NULL) {
        Py_DECREF(d_array);
        return NULL;
    }

    f64 *d_data = (f64*)PyArray_DATA(d_array);
    i32 *ipr_data = (i32*)PyArray_DATA(ipr_array);
    i32 info = 0;

    sb04mr(m, d_data, ipr_data, &info);

    PyArray_ResolveWritebackIfCopy(d_array);

    PyObject *result = Py_BuildValue("OOi", d_array, ipr_array, info);
    Py_DECREF(d_array);
    Py_DECREF(ipr_array);
    return result;
}



/* Python wrapper for sb04mw */
PyObject* py_sb04mw(PyObject* self, PyObject* args) {
    i32 m;
    PyObject *d_obj;

    if (!PyArg_ParseTuple(args, "iO", &m, &d_obj)) {
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (d_array == NULL) {
        return NULL;
    }

    i32 ipr_len = (m > 0) ? 2 * m : 1;
    npy_intp ipr_dims[1] = {ipr_len};
    PyArrayObject *ipr_array = (PyArrayObject*)PyArray_SimpleNew(1, ipr_dims, NPY_INT32);
    if (ipr_array == NULL) {
        Py_DECREF(d_array);
        return NULL;
    }

    f64 *d_data = (f64*)PyArray_DATA(d_array);
    i32 *ipr_data = (i32*)PyArray_DATA(ipr_array);
    i32 info = 0;

    sb04mw(m, d_data, ipr_data, &info);

    PyArray_ResolveWritebackIfCopy(d_array);

    PyObject *result = Py_BuildValue("OOi", d_array, ipr_array, info);
    Py_DECREF(d_array);
    Py_DECREF(ipr_array);
    return result;
}



/* Python wrapper for sb04qr */
PyObject* py_sb04qr(PyObject* self, PyObject* args) {
    i32 m;
    PyObject *d_obj;

    if (!PyArg_ParseTuple(args, "iO", &m, &d_obj)) {
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (d_array == NULL) {
        return NULL;
    }

    i32 ipr_len = (m > 0) ? 2 * m : 1;
    npy_intp ipr_dims[1] = {ipr_len};
    PyArrayObject *ipr_array = (PyArrayObject*)PyArray_SimpleNew(1, ipr_dims, NPY_INT32);
    if (ipr_array == NULL) {
        Py_DECREF(d_array);
        return NULL;
    }

    f64 *d_data = (f64*)PyArray_DATA(d_array);
    i32 *ipr_data = (i32*)PyArray_DATA(ipr_array);
    i32 info = 0;

    sb04qr(m, d_data, ipr_data, &info);

    PyArray_ResolveWritebackIfCopy(d_array);

    PyObject *result = Py_BuildValue("OOi", d_array, ipr_array, info);
    Py_DECREF(d_array);
    Py_DECREF(ipr_array);
    return result;
}



/* Python wrapper for sb04md */
PyObject* py_sb04md(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj, *c_obj;

    if (!PyArg_ParseTuple(args, "OOO", &a_obj, &b_obj, &c_obj)) {
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

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);

    i32 n = (i32)a_dims[0];
    i32 m = (i32)b_dims[0];
    i32 lda = (n > 1) ? n : 1;
    i32 ldb = (m > 1) ? m : 1;
    i32 ldc = (n > 1) ? n : 1;
    i32 ldz = (m > 1) ? m : 1;

    i32 t1 = 2 * n * n + 8 * n;
    i32 t2 = 5 * m;
    i32 t3 = n + m;
    i32 ldwork = 1;
    if (t1 > ldwork) ldwork = t1;
    if (t2 > ldwork) ldwork = t2;
    if (t3 > ldwork) ldwork = t3;

    i32 liwork = (n > 0) ? 4 * n : 1;

    npy_intp z_dims[2] = {m > 0 ? m : 1, m > 0 ? m : 1};
    npy_intp z_strides[2] = {sizeof(f64), (m > 0 ? m : 1) * sizeof(f64)};
    PyArrayObject *z_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, z_dims,
        NPY_DOUBLE, z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (z_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_NoMemory();
        return NULL;
    }
    f64 *z_data = (f64*)PyArray_DATA(z_array);

    i32 *iwork = (i32*)malloc(liwork * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (iwork == NULL || dwork == NULL) {
        free(iwork);
        free(dwork);
        Py_DECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    i32 info = 0;

    sb04md(n, m, a_data, lda, b_data, ldb, c_data, ldc, z_data, ldz,
           iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    Py_DECREF(a_array);
    Py_DECREF(b_array);

    PyObject *result = Py_BuildValue("OOi", c_array, z_array, info);
    Py_DECREF(c_array);
    Py_DECREF(z_array);
    return result;
}


/* Python wrapper for sb04py */
PyObject* py_sb04py(PyObject* self, PyObject* args) {
    const char *trana_str, *tranb_str;
    int isgn;
    PyObject *a_obj, *b_obj, *c_obj;

    if (!PyArg_ParseTuple(args, "ssiOOO", &trana_str, &tranb_str, &isgn,
                          &a_obj, &b_obj, &c_obj)) {
        return NULL;
    }

    if (trana_str[0] != 'N' && trana_str[0] != 'n' &&
        trana_str[0] != 'T' && trana_str[0] != 't' &&
        trana_str[0] != 'C' && trana_str[0] != 'c') {
        PyErr_SetString(PyExc_ValueError, "trana must be 'N', 'T', or 'C'");
        return NULL;
    }

    if (tranb_str[0] != 'N' && tranb_str[0] != 'n' &&
        tranb_str[0] != 'T' && tranb_str[0] != 't' &&
        tranb_str[0] != 'C' && tranb_str[0] != 'c') {
        PyErr_SetString(PyExc_ValueError, "tranb must be 'N', 'T', or 'C'");
        return NULL;
    }

    if (isgn != 1 && isgn != -1) {
        PyErr_SetString(PyExc_ValueError, "isgn must be 1 or -1");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
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

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);

    i32 m = (i32)a_dims[0];
    i32 n = (i32)b_dims[0];
    i32 lda = (m > 1) ? m : 1;
    i32 ldb = (n > 1) ? n : 1;
    i32 ldc = (m > 1) ? m : 1;

    i32 ldwork = (m > 0) ? 2 * m : 1;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 scale = 0.0;
    i32 info = 0;

    sb04py(trana_str[0], tranb_str[0], isgn, m, n, a_data, lda, b_data, ldb,
           c_data, ldc, &scale, dwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(c_array);

    Py_DECREF(a_array);
    Py_DECREF(b_array);

    PyObject *result = Py_BuildValue("Odi", c_array, scale, info);
    Py_DECREF(c_array);
    return result;
}


/* Python wrapper for sb04qd */
PyObject* py_sb04qd(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj, *c_obj;

    if (!PyArg_ParseTuple(args, "OOO", &a_obj, &b_obj, &c_obj)) {
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

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);

    i32 n = (i32)a_dims[0];
    i32 m = (i32)b_dims[0];
    i32 lda = (n > 1) ? n : 1;
    i32 ldb = (m > 1) ? m : 1;
    i32 ldc = (n > 1) ? n : 1;
    i32 ldz = (m > 1) ? m : 1;

    i32 t1 = 2 * n * n + 9 * n;
    i32 t2 = 5 * m;
    i32 t3 = n + m;
    i32 ldwork = 1;
    if (t1 > ldwork) ldwork = t1;
    if (t2 > ldwork) ldwork = t2;
    if (t3 > ldwork) ldwork = t3;

    i32 liwork = (n > 0) ? 4 * n : 1;

    npy_intp z_dims[2] = {m > 0 ? m : 1, m > 0 ? m : 1};
    npy_intp z_strides[2] = {sizeof(f64), (m > 0 ? m : 1) * sizeof(f64)};
    PyArrayObject *z_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, z_dims,
        NPY_DOUBLE, z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (z_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_NoMemory();
        return NULL;
    }
    f64 *z_data = (f64*)PyArray_DATA(z_array);

    i32 *iwork = (i32*)malloc(liwork * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (iwork == NULL || dwork == NULL) {
        free(iwork);
        free(dwork);
        Py_DECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    i32 info = 0;

    sb04qd(n, m, a_data, lda, b_data, ldb, c_data, ldc, z_data, ldz,
           iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    Py_DECREF(a_array);
    Py_DECREF(b_array);

    PyObject *result = Py_BuildValue("OOi", c_array, z_array, info);
    Py_DECREF(c_array);
    Py_DECREF(z_array);
    return result;
}



/* Python wrapper for sb04nd */
PyObject* py_sb04nd(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *abschu_str, *ula_str, *ulb_str;
    PyObject *a_obj, *b_obj, *c_obj;
    f64 tol = 0.0;

    static char *kwlist[] = {"abschu", "ula", "ulb", "a", "b", "c", "tol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssOOO|d", kwlist,
                                      &abschu_str, &ula_str, &ulb_str,
                                      &a_obj, &b_obj, &c_obj, &tol)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
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

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    i32 n = (i32)a_dims[0];
    i32 m = (i32)b_dims[0];
    i32 lda = (n > 1) ? n : 1;
    i32 ldb = (m > 1) ? m : 1;
    i32 ldc = (c_dims[0] > 1) ? (i32)c_dims[0] : 1;

    i32 maxmn = (m > n) ? m : n;
    bool use_dtrsyl = (abschu_str[0] == 'S' || abschu_str[0] == 's') &&
                      (ula_str[0] == 'U' || ula_str[0] == 'u') &&
                      (ulb_str[0] == 'U' || ulb_str[0] == 'u');

    i32 ldwork = use_dtrsyl ? 0 : 2 * maxmn * (4 + 2 * maxmn);
    i32 liwork = use_dtrsyl ? 0 : 2 * maxmn;

    i32 *iwork = NULL;
    f64 *dwork = NULL;

    if (!use_dtrsyl) {
        iwork = (i32*)malloc((liwork > 0 ? liwork : 1) * sizeof(i32));
        dwork = (f64*)malloc((ldwork > 0 ? ldwork : 1) * sizeof(f64));
        if (iwork == NULL || dwork == NULL) {
            free(iwork);
            free(dwork);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            PyErr_NoMemory();
            return NULL;
        }
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    i32 info = 0;

    sb04nd(abschu_str, ula_str, ulb_str, n, m, a_data, lda, b_data, ldb,
           c_data, ldc, tol, iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(c_array);

    Py_DECREF(a_array);
    Py_DECREF(b_array);

    PyObject *result = Py_BuildValue("Oi", c_array, info);
    Py_DECREF(c_array);
    return result;
}



/* Python wrapper for sb01fy */
PyObject* py_sb01fy(PyObject* self, PyObject* args) {
    int discr;
    PyObject *a_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "pOO", &discr, &a_obj, &b_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (a_array == NULL) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 m = (i32)PyArray_DIM(b_array, 1);
    i32 lda = n;
    i32 ldb = n;
    i32 ldf = m > 1 ? m : 1;
    i32 ldv = m > 1 ? m : 1;

    const f64 *a_data = (const f64*)PyArray_DATA(a_array);
    const f64 *b_data = (const f64*)PyArray_DATA(b_array);

    npy_intp f_dims[2] = {m, n};
    npy_intp v_dims[2] = {m, m};
    npy_intp f_strides[2] = {sizeof(f64), m * sizeof(f64)};
    npy_intp v_strides[2] = {sizeof(f64), m * sizeof(f64)};

    PyArrayObject *f_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, f_dims, NPY_DOUBLE,
                                                          f_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyArrayObject *v_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, v_dims, NPY_DOUBLE,
                                                          v_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!f_array || !v_array) {
        Py_XDECREF(f_array);
        Py_XDECREF(v_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return PyErr_NoMemory();
    }
    f64 *f_data = (f64*)PyArray_DATA(f_array);
    f64 *v_data = (f64*)PyArray_DATA(v_array);
    memset(f_data, 0, m * n * sizeof(f64));
    memset(v_data, 0, m * m * sizeof(f64));

    i32 info = 0;
    sb01fy((bool)discr, n, m, a_data, lda, b_data, ldb, f_data, ldf, v_data, ldv, &info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);

    PyObject *result = Py_BuildValue("(OOi)", f_array, v_array, info);
    Py_DECREF(f_array);
    Py_DECREF(v_array);
    return result;
}



/* Python wrapper for sb01bx */
PyObject* py_sb01bx(PyObject* self, PyObject* args) {
    int reig_int;
    i32 n;
    f64 xr, xi;
    PyObject *wr_obj, *wi_obj;

    if (!PyArg_ParseTuple(args, "iiddOO", &reig_int, &n, &xr, &xi, &wr_obj, &wi_obj)) {
        return NULL;
    }

    bool reig = (reig_int != 0);

    PyArrayObject *wr_array = (PyArrayObject*)PyArray_FROM_OTF(wr_obj, NPY_DOUBLE,
                                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (wr_array == NULL) return NULL;

    PyArrayObject *wi_array = (PyArrayObject*)PyArray_FROM_OTF(wi_obj, NPY_DOUBLE,
                                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (wi_array == NULL) {
        Py_DECREF(wr_array);
        return NULL;
    }

    f64 *wr_data = (f64*)PyArray_DATA(wr_array);
    f64 *wi_data = (f64*)PyArray_DATA(wi_array);

    f64 s = 0.0, p = 0.0;

    sb01bx(reig, n, xr, xi, wr_data, wi_data, &s, &p);

    PyArray_ResolveWritebackIfCopy(wr_array);
    PyArray_ResolveWritebackIfCopy(wi_array);

    PyObject *result = Py_BuildValue("(OOdd)", wr_array, wi_array, s, p);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);
    return result;
}



/* Python wrapper for sb01by */
PyObject* py_sb01by(PyObject* self, PyObject* args) {
    i32 n, m;
    f64 s, p, tol;
    PyObject *a_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "iiddOOd", &n, &m, &s, &p, &a_obj, &b_obj, &tol)) {
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

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    npy_intp f_dims[2] = {m, n};
    npy_intp f_strides[2] = {sizeof(f64), m * sizeof(f64)};
    PyArrayObject *f_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, f_dims, NPY_DOUBLE,
                                                          f_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (f_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return PyErr_NoMemory();
    }
    f64 *f_data = (f64*)PyArray_DATA(f_array);
    memset(f_data, 0, m * n * sizeof(f64));

    f64 *dwork = (f64*)calloc(m > 0 ? m : 1, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(f_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return PyErr_NoMemory();
    }

    i32 info = 0;
    sb01by(n, m, s, p, a_data, b_data, f_data, tol, dwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);

    PyObject *result = Py_BuildValue("(Oi)", f_array, info);
    Py_DECREF(f_array);
    return result;
}



/* Python wrapper for sb01bd */
PyObject* py_sb01bd(PyObject* self, PyObject* args) {
    char* dico;
    i32 n, m, np_poles;
    f64 alpha, tol;
    PyObject *a_obj, *b_obj, *wr_obj, *wi_obj;

    if (!PyArg_ParseTuple(args, "siiidOOOOd",
                          &dico, &n, &m, &np_poles, &alpha,
                          &a_obj, &b_obj, &wr_obj, &wi_obj, &tol)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *wr_array = (PyArrayObject*)PyArray_FROM_OTF(wr_obj, NPY_DOUBLE,
                                                                 NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (wr_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *wi_array = (PyArrayObject*)PyArray_FROM_OTF(wi_obj, NPY_DOUBLE,
                                                                 NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (wi_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(wr_array);
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *wr_data = (f64*)PyArray_DATA(wr_array);
    f64 *wi_data = (f64*)PyArray_DATA(wi_array);

    i32 lda = (n > 1) ? n : 1;
    i32 ldb = lda;
    i32 ldf = (m > 1) ? m : 1;
    i32 ldz = lda;

    /* Allocate output arrays using NumPy (Fortran order) */
    npy_intp f_dims[2] = {m, n};
    npy_intp z_dims[2] = {n, n};

    PyArrayObject *f_array = (PyArrayObject*)PyArray_EMPTY(2, f_dims, NPY_DOUBLE, 1);  /* 1 = Fortran order */
    PyArrayObject *z_array = (PyArrayObject*)PyArray_EMPTY(2, z_dims, NPY_DOUBLE, 1);

    if (f_array == NULL || z_array == NULL) {
        Py_XDECREF(f_array);
        Py_XDECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(wr_array);
        Py_DECREF(wi_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *f_data = (f64*)PyArray_DATA(f_array);
    f64 *z_data = (f64*)PyArray_DATA(z_array);

    /* Initialize to zero (only if non-empty) */
    if (m > 0 && n > 0) {
        memset(f_data, 0, m * n * sizeof(f64));
    }
    if (n > 0) {
        memset(z_data, 0, n * n * sizeof(f64));
    }

    /* Allocate workspace */
    i32 ldwork = 1;
    if (5*m > ldwork) ldwork = 5*m;
    if (5*n > ldwork) ldwork = 5*n;
    if (2*n + 4*m > ldwork) ldwork = 2*n + 4*m;
    f64 *dwork = (f64*)calloc(ldwork > 0 ? ldwork : 1, sizeof(f64));

    if (dwork == NULL) {
        free(dwork);
        Py_DECREF(f_array);
        Py_DECREF(z_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(wr_array);
        Py_DECREF(wi_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 nfp, nap, nup, iwarn, info;
    sb01bd(dico, n, m, np_poles, alpha, a_data, lda, b_data, ldb,
           wr_data, wi_data, &nfp, &nap, &nup, f_data, ldf, z_data, ldz,
           tol, dwork, ldwork, &iwarn, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(wr_array);
    PyArray_ResolveWritebackIfCopy(wi_array);

    PyObject *result = Py_BuildValue("(OOOiiiOOii)",
                                      a_array, wr_array, wi_array,
                                      nfp, nap, nup, f_array, z_array,
                                      iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);
    Py_DECREF(f_array);
    Py_DECREF(z_array);

    return result;
}



/* Python wrapper for sb03md */
PyObject* py_sb03md(PyObject* self, PyObject* args) {
    char *dico, *job, *fact, *trana;
    i32 n;
    PyObject *a_obj, *c_obj, *u_obj = Py_None;

    if (!PyArg_ParseTuple(args, "ssssiOO|O",
                          &dico, &job, &fact, &trana, &n, &a_obj, &c_obj, &u_obj)) {
        return NULL;
    }

    bool nofact = (fact[0] == 'N' || fact[0] == 'n');
    bool wantx = (job[0] == 'X' || job[0] == 'x');
    bool wantsp = (job[0] == 'S' || job[0] == 's');
    bool cont = (dico[0] == 'C' || dico[0] == 'c');

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *c_array = NULL;
    if (!wantsp) {
        c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                                    NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (c_array == NULL) {
            Py_DECREF(a_array);
            return NULL;
        }
    }

    PyArrayObject *u_array = NULL;
    PyArrayObject *u_out_array = NULL;
    f64 *u_data = NULL;

    if (!nofact && u_obj != Py_None) {
        u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE,
                                                    NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (u_array == NULL) {
            Py_DECREF(a_array);
            if (c_array) Py_DECREF(c_array);
            return NULL;
        }
        u_data = (f64*)PyArray_DATA(u_array);
    } else {
        npy_intp u_dims[2] = {n, n};
        npy_intp u_strides[2] = {sizeof(f64), n * sizeof(f64)};
        u_out_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE,
                                                   u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (u_out_array == NULL) {
            Py_DECREF(a_array);
            if (c_array) Py_DECREF(c_array);
            return PyErr_NoMemory();
        }
        u_data = (f64*)PyArray_DATA(u_out_array);
        if (n > 0) { memset(u_data, 0, n * n * sizeof(f64)); }
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *c_data = c_array ? (f64*)PyArray_DATA(c_array) : NULL;

    i32 lda = (n > 1) ? n : 1;
    i32 ldu = lda;
    i32 ldc = wantsp ? 1 : lda;

    /* Allocate eigenvalue arrays */
    npy_intp eig_dims[1] = {n};
    PyArrayObject *wr_array = (PyArrayObject*)PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);
    PyArrayObject *wi_array = (PyArrayObject*)PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);
    if (!wr_array || !wi_array) {
        Py_XDECREF(wr_array);
        Py_XDECREF(wi_array);
        if (u_out_array) Py_DECREF(u_out_array);
        Py_DECREF(a_array);
        if (c_array) Py_DECREF(c_array);
        if (u_array) Py_DECREF(u_array);
        return PyErr_NoMemory();
    }
    f64 *wr = (f64*)PyArray_DATA(wr_array);
    f64 *wi = (f64*)PyArray_DATA(wi_array);
    if (n > 0) { memset(wr, 0, n * sizeof(f64)); }
    if (n > 0) { memset(wi, 0, n * sizeof(f64)); }
    i32 *iwork = (i32*)calloc((n > 0) ? n * n : 1, sizeof(i32));

    /* Compute workspace size */
    i32 nn = n * n;
    i32 minwrk;
    if (wantx) {
        if (nofact) {
            minwrk = nn > 3*n ? nn : 3*n;
        } else if (cont) {
            minwrk = nn;
        } else {
            minwrk = nn > 2*n ? nn : 2*n;
        }
    } else {
        if (cont) {
            if (nofact) {
                minwrk = 2*nn > 3*n ? 2*nn : 3*n;
            } else {
                minwrk = 2*nn;
            }
        } else {
            minwrk = 2*nn + 2*n;
        }
    }
    minwrk = minwrk > 1 ? minwrk : 1;

    f64 *dwork = (f64*)calloc(minwrk, sizeof(f64));

    if (iwork == NULL || dwork == NULL) {
        free(iwork);
        free(dwork);
        Py_DECREF(wr_array);
        Py_DECREF(wi_array);
        if (u_out_array) Py_DECREF(u_out_array);
        Py_DECREF(a_array);
        if (c_array) Py_DECREF(c_array);
        if (u_array) Py_DECREF(u_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 scale, sep = 0.0, ferr = 0.0;
    i32 info;

    sb03md(dico, job, fact, trana, n, a_data, lda, u_data, ldu,
           c_data, ldc, &scale, &sep, &ferr, wr, wi, iwork, dwork, minwrk, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    if (c_array) PyArray_ResolveWritebackIfCopy(c_array);
    if (u_array) PyArray_ResolveWritebackIfCopy(u_array);

    /* Use U output array if we allocated it, otherwise use input u_array */
    if (u_array != NULL) {
        u_out_array = u_array;
        Py_INCREF(u_out_array);
    }

    /* Return C as solution X (or as is if wantsp) */
    PyObject *x_out;
    if (wantsp) {
        Py_INCREF(Py_None);
        x_out = Py_None;
    } else {
        x_out = (PyObject*)c_array;
        Py_INCREF(x_out);
    }

    PyObject *result = Py_BuildValue("(OOOOOdddi)",
                                      x_out, a_array, u_out_array,
                                      wr_array, wi_array,
                                      scale, sep, ferr, info);

    if (!wantsp) Py_DECREF(c_array);
    if (wantsp) Py_DECREF(x_out);
    Py_DECREF(a_array);
    Py_DECREF(u_out_array);
    if (u_array) Py_DECREF(u_array);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);

    return result;
}



/* Python wrapper for sb03ou */
PyObject* py_sb03ou(PyObject* self, PyObject* args) {
    int discr, ltrans;
    PyObject *a_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "ppOO", &discr, &ltrans, &a_obj, &b_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (a_array == NULL) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 m;
    if (ltrans) {
        m = (i32)PyArray_DIM(b_array, 1);
    } else {
        m = (i32)PyArray_DIM(b_array, 0);
    }
    i32 lda = n > 1 ? n : 1;
    i32 ldb;
    if (ltrans) {
        ldb = n > 1 ? n : 1;
    } else {
        ldb = m > 1 ? m : 1;
    }
    i32 ldu = n > 1 ? n : 1;

    const f64 *a_data = (const f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    npy_intp u_dims[2] = {n, n};
    npy_intp u_strides[2] = {sizeof(f64), n * sizeof(f64)};
    PyArrayObject *u_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE,
                                                          u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!u_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return PyErr_NoMemory();
    }
    f64 *u_data = (f64*)PyArray_DATA(u_array);
    memset(u_data, 0, n * n * sizeof(f64));

    i32 mn = m < n ? m : n;
    f64 *tau = (f64*)calloc(mn > 1 ? mn : 1, sizeof(f64));
    i32 ldwork = 4 * n > 1 ? 4 * n : 1;
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (!tau || !dwork) {
        free(tau);
        free(dwork);
        Py_DECREF(u_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return PyErr_NoMemory();
    }

    f64 scale = 1.0;
    i32 info = 0;
    sb03ou((bool)discr, (bool)ltrans, n, m, a_data, lda, b_data, ldb,
           tau, u_data, ldu, &scale, dwork, ldwork, &info);

    free(tau);
    free(dwork);
    Py_DECREF(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    Py_DECREF(b_array);

    PyObject *result = Py_BuildValue("(Odi)", u_array, scale, info);
    Py_DECREF(u_array);
    return result;
}



/* Python wrapper for sb08dd */
PyObject* py_sb08dd(PyObject* self, PyObject* args) {
    const char *dico;
    f64 tol;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "sOOOOd", &dico, &a_obj, &b_obj, &c_obj, &d_obj, &tol)) {
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

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 m = (i32)PyArray_DIM(b_array, 1);
    i32 p = (i32)PyArray_DIM(c_array, 0);
    i32 lda = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldc = p > 1 ? p : 1;
    i32 ldd = p > 1 ? p : 1;
    i32 ldcr = m > 1 ? m : 1;
    i32 lddr = m > 1 ? m : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);

    npy_intp cr_dims[2] = {m, n};
    npy_intp dr_dims[2] = {m, m};
    npy_intp cr_strides[2] = {sizeof(f64), m * sizeof(f64)};
    npy_intp dr_strides[2] = {sizeof(f64), m * sizeof(f64)};

    PyArrayObject *cr_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, cr_dims, NPY_DOUBLE,
                                                           cr_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyArrayObject *dr_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dr_dims, NPY_DOUBLE,
                                                           dr_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!cr_array || !dr_array) {
        Py_XDECREF(cr_array);
        Py_XDECREF(dr_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return PyErr_NoMemory();
    }
    f64 *cr_data = (f64*)PyArray_DATA(cr_array);
    f64 *dr_data = (f64*)PyArray_DATA(dr_array);
    memset(cr_data, 0, m * n * sizeof(f64));
    memset(dr_data, 0, m * m * sizeof(f64));

    i32 min1 = n * (n + 5);
    i32 min2 = m * (m + 2);
    i32 min3 = 4 * m;
    i32 min4 = 4 * p;
    i32 ldwork = min1 > min2 ? min1 : min2;
    ldwork = ldwork > min3 ? ldwork : min3;
    ldwork = ldwork > min4 ? ldwork : min4;
    ldwork = ldwork > 1 ? ldwork : 1;
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (!dwork) {
        free(dwork);
        Py_DECREF(cr_array);
        Py_DECREF(dr_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return PyErr_NoMemory();
    }

    i32 nq = 0, nr = 0;
    i32 iwarn = 0, info = 0;

    sb08dd(dico, n, m, p, a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           &nq, &nr, cr_data, ldcr, dr_data, lddr, tol, dwork, ldwork, &iwarn, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    PyObject *result = Py_BuildValue("(OOOOiiOOii)", a_array, b_array, c_array, d_array,
                                      nq, nr, cr_array, dr_array, iwarn, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(cr_array);
    Py_DECREF(dr_array);
    return result;
}


PyObject* py_sb10vd(PyObject* self, PyObject* args) {
    int n, m, np, ncon, nmeas;
    PyObject *a_obj, *b_obj, *c_obj;

    if (!PyArg_ParseTuple(args, "iiOOO", &ncon, &nmeas, &a_obj, &b_obj, &c_obj)) {
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

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (!c_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    n = (i32)PyArray_DIM(a_array, 0);
    m = (i32)PyArray_DIM(b_array, 1);
    np = (i32)PyArray_DIM(c_array, 0);

    i32 lda = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldc = np > 1 ? np : 1;
    i32 ldf = ncon > 1 ? ncon : 1;
    i32 ldh = n > 1 ? n : 1;
    i32 ldx = n > 1 ? n : 1;
    i32 ldy = n > 1 ? n : 1;

    f64* a = (f64*)PyArray_DATA(a_array);
    f64* b = (f64*)PyArray_DATA(b_array);
    f64* c = (f64*)PyArray_DATA(c_array);

    npy_intp f_dims[2] = {ncon, n};
    npy_intp f_strides[2] = {sizeof(f64), ncon * sizeof(f64)};
    npy_intp h_dims[2] = {n, nmeas};
    npy_intp h_strides[2] = {sizeof(f64), n * sizeof(f64)};
    npy_intp x_dims[2] = {n, n};
    npy_intp x_strides[2] = {sizeof(f64), n * sizeof(f64)};
    npy_intp y_dims[2] = {n, n};
    npy_intp y_strides[2] = {sizeof(f64), n * sizeof(f64)};

    PyObject* f_array = PyArray_New(&PyArray_Type, 2, f_dims, NPY_DOUBLE,
                                     f_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject* h_array = PyArray_New(&PyArray_Type, 2, h_dims, NPY_DOUBLE,
                                     h_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject* x_array = PyArray_New(&PyArray_Type, 2, x_dims, NPY_DOUBLE,
                                     x_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject* y_array = PyArray_New(&PyArray_Type, 2, y_dims, NPY_DOUBLE,
                                     y_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (!f_array || !h_array || !x_array || !y_array) {
        Py_XDECREF(f_array); Py_XDECREF(h_array);
        Py_XDECREF(x_array); Py_XDECREF(y_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    f64* f_out = (f64*)PyArray_DATA((PyArrayObject*)f_array);
    f64* h_out = (f64*)PyArray_DATA((PyArrayObject*)h_array);
    f64* x_out = (f64*)PyArray_DATA((PyArrayObject*)x_array);
    f64* y_out = (f64*)PyArray_DATA((PyArrayObject*)y_array);
    f64 xycond[2] = {0.0, 0.0};

    i32 nn = n * n;
    i32 liwork = nn > 2 * n ? nn : 2 * n;
    if (liwork < 1) liwork = 1;
    i32 ldwork = 13 * nn + 12 * n + 5;
    if (ldwork < 1) ldwork = 1;

    i32* iwork = (i32*)malloc(liwork * sizeof(i32));
    f64* dwork = (f64*)malloc(ldwork * sizeof(f64));
    i32* bwork = (i32*)malloc(2 * n * sizeof(i32));

    if (!iwork || !dwork || !bwork) {
        free(iwork); free(dwork); free(bwork);
        Py_DECREF(f_array); Py_DECREF(h_array);
        Py_DECREF(x_array); Py_DECREF(y_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return PyErr_NoMemory();
    }

    i32 info = 0;
    sb10vd(n, m, np, ncon, nmeas, a, lda, b, ldb, c, ldc,
           f_out, ldf, h_out, ldh, x_out, ldx, y_out, ldy,
           xycond, iwork, dwork, ldwork, bwork, &info);

    free(iwork);
    free(dwork);
    free(bwork);

    PyObject* result = Py_BuildValue("(OOOOddi)", f_array, h_array, x_array, y_array,
                                      xycond[0], xycond[1], info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(f_array);
    Py_DECREF(h_array);
    Py_DECREF(x_array);
    Py_DECREF(y_array);

    return result;
}


PyObject* py_sb10rd(PyObject* self, PyObject* args) {
    int n, m, np, ncon, nmeas;
    double gamma;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *f_obj, *h_obj, *tu_obj, *ty_obj, *x_obj, *y_obj;

    if (!PyArg_ParseTuple(args, "iiiiidOOOOOOOOOO", &n, &m, &np, &ncon, &nmeas, &gamma,
                          &a_obj, &b_obj, &c_obj, &d_obj, &f_obj, &h_obj,
                          &tu_obj, &ty_obj, &x_obj, &y_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *f_array = (PyArrayObject*)PyArray_FROM_OTF(f_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *h_array = (PyArrayObject*)PyArray_FROM_OTF(h_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *tu_array = (PyArrayObject*)PyArray_FROM_OTF(tu_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *ty_array = (PyArrayObject*)PyArray_FROM_OTF(ty_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);

    if (!a_array || !b_array || !c_array || !d_array || !f_array ||
        !h_array || !tu_array || !ty_array || !x_array || !y_array) {
        Py_XDECREF(a_array); Py_XDECREF(b_array); Py_XDECREF(c_array);
        Py_XDECREF(d_array); Py_XDECREF(f_array); Py_XDECREF(h_array);
        Py_XDECREF(tu_array); Py_XDECREF(ty_array); Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        return NULL;
    }

    i32 lda = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldc = np > 1 ? np : 1;
    i32 ldd = np > 1 ? np : 1;
    i32 ldf = m > 1 ? m : 1;
    i32 ldh = n > 1 ? n : 1;
    i32 ldtu = ncon > 1 ? ncon : 1;
    i32 ldty = nmeas > 1 ? nmeas : 1;
    i32 ldx = n > 1 ? n : 1;
    i32 ldy = n > 1 ? n : 1;
    i32 ldak = n > 1 ? n : 1;
    i32 ldbk = n > 1 ? n : 1;
    i32 ldck = ncon > 1 ? ncon : 1;
    i32 lddk = ncon > 1 ? ncon : 1;

    npy_intp ak_dims[2] = {n, n};
    npy_intp ak_strides[2] = {sizeof(f64), n * sizeof(f64)};
    npy_intp bk_dims[2] = {n, nmeas};
    npy_intp bk_strides[2] = {sizeof(f64), n * sizeof(f64)};
    npy_intp ck_dims[2] = {ncon, n};
    npy_intp ck_strides[2] = {sizeof(f64), ncon * sizeof(f64)};
    npy_intp dk_dims[2] = {ncon, nmeas};
    npy_intp dk_strides[2] = {sizeof(f64), ncon * sizeof(f64)};

    PyObject* ak_arr = PyArray_New(&PyArray_Type, 2, ak_dims, NPY_DOUBLE,
                                    ak_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject* bk_arr = PyArray_New(&PyArray_Type, 2, bk_dims, NPY_DOUBLE,
                                    bk_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject* ck_arr = PyArray_New(&PyArray_Type, 2, ck_dims, NPY_DOUBLE,
                                    ck_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject* dk_arr = PyArray_New(&PyArray_Type, 2, dk_dims, NPY_DOUBLE,
                                    dk_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (!ak_arr || !bk_arr || !ck_arr || !dk_arr) {
        Py_XDECREF(ak_arr); Py_XDECREF(bk_arr); Py_XDECREF(ck_arr); Py_XDECREF(dk_arr);
        Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array);
        Py_DECREF(d_array); Py_DECREF(f_array); Py_DECREF(h_array);
        Py_DECREF(tu_array); Py_DECREF(ty_array); Py_DECREF(x_array);
        Py_DECREF(y_array);
        return PyErr_NoMemory();
    }

    f64* ak = (f64*)PyArray_DATA((PyArrayObject*)ak_arr);
    f64* bk = (f64*)PyArray_DATA((PyArrayObject*)bk_arr);
    f64* ck = (f64*)PyArray_DATA((PyArrayObject*)ck_arr);
    f64* dk = (f64*)PyArray_DATA((PyArrayObject*)dk_arr);
    memset(ak, 0, n * n * sizeof(f64));
    memset(bk, 0, n * nmeas * sizeof(f64));
    memset(ck, 0, ncon * n * sizeof(f64));
    memset(dk, 0, ncon * nmeas * sizeof(f64));

    i32 ldwork = 2 * n * n + 14 * n + 23;
    if (ldwork < 1) ldwork = 1;
    f64* dwork = (f64*)malloc(ldwork * sizeof(f64));
    i32* iwork = (i32*)malloc(20 * sizeof(i32));

    if (!dwork || !iwork) {
        free(dwork); free(iwork);
        Py_DECREF(ak_arr); Py_DECREF(bk_arr); Py_DECREF(ck_arr); Py_DECREF(dk_arr);
        Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array);
        Py_DECREF(d_array); Py_DECREF(f_array); Py_DECREF(h_array);
        Py_DECREF(tu_array); Py_DECREF(ty_array); Py_DECREF(x_array);
        Py_DECREF(y_array);
        return PyErr_NoMemory();
    }

    i32 info = 0;
    sb10rd(n, m, np, ncon, nmeas, gamma,
           (f64*)PyArray_DATA(a_array), lda,
           (f64*)PyArray_DATA(b_array), ldb,
           (f64*)PyArray_DATA(c_array), ldc,
           (f64*)PyArray_DATA(d_array), ldd,
           (f64*)PyArray_DATA(f_array), ldf,
           (f64*)PyArray_DATA(h_array), ldh,
           (f64*)PyArray_DATA(tu_array), ldtu,
           (f64*)PyArray_DATA(ty_array), ldty,
           (f64*)PyArray_DATA(x_array), ldx,
           (f64*)PyArray_DATA(y_array), ldy,
           ak, ldak, bk, ldbk, ck, ldck, dk, lddk,
           iwork, dwork, ldwork, &info);

    free(dwork);
    free(iwork);

    PyObject* result = Py_BuildValue("(OOOOi)", ak_arr, bk_arr, ck_arr, dk_arr, info);

    Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array);
    Py_DECREF(d_array); Py_DECREF(f_array); Py_DECREF(h_array);
    Py_DECREF(tu_array); Py_DECREF(ty_array); Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(ak_arr); Py_DECREF(bk_arr); Py_DECREF(ck_arr); Py_DECREF(dk_arr);

    return result;
}


PyObject* py_sb10zp(PyObject* self, PyObject* args) {
    int discfl;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "iOOOO", &discfl, &a_obj, &b_obj, &c_obj, &d_obj)) {
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

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 lda = n > 0 ? n : 1;

    i32 minwork1 = n * n + 5 * n;
    i32 min1n = n > 0 ? 1 : 0;
    i32 minwork2 = 6 * n + 1 + min1n;
    i32 ldwork = minwork1 > minwork2 ? minwork1 : minwork2;
    if (ldwork < 1) ldwork = 1;

    i32 iwork_size = n + 1;
    if (iwork_size < 2) iwork_size = 2;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    i32 *iwork = (i32*)malloc(iwork_size * sizeof(i32));
    if (!dwork || !iwork) {
        free(dwork);
        free(iwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return PyErr_NoMemory();
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);

    i32 info = 0;

    sb10zp((i32)discfl, &n, a_data, lda, b_data, c_data, d_data, iwork, dwork, ldwork, &info);

    free(dwork);
    free(iwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    PyObject *result = Py_BuildValue("OOOOii", a_array, b_array, c_array, d_array, n, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    return result;
}


PyObject* py_sb10ad(PyObject* self, PyObject* args) {
    int job, n, m, np_arg, ncon, nmeas;
    double gamma, gtol, actol;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "iiiiiiOOOOddd", &job, &n, &m, &np_arg, &ncon, &nmeas,
                          &a_obj, &b_obj, &c_obj, &d_obj, &gamma, &gtol, &actol)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);

    if (!a_array || !b_array || !c_array || !d_array) {
        Py_XDECREF(a_array); Py_XDECREF(b_array);
        Py_XDECREF(c_array); Py_XDECREF(d_array);
        return NULL;
    }

    i32 m1 = m - ncon;
    i32 m2 = ncon;
    i32 np1 = np_arg - nmeas;
    i32 np2 = nmeas;
    i32 n2 = 2 * n;

    i32 lda = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldc = np_arg > 1 ? np_arg : 1;
    i32 ldd = np_arg > 1 ? np_arg : 1;
    i32 ldak = n > 1 ? n : 1;
    i32 ldbk = n > 1 ? n : 1;
    i32 ldck = m2 > 1 ? m2 : 1;
    i32 lddk = m2 > 1 ? m2 : 1;
    i32 ldac = n2 > 1 ? n2 : 1;
    i32 ldbc = n2 > 1 ? n2 : 1;
    i32 ldcc = np1 > 1 ? np1 : 1;
    i32 lddc = np1 > 1 ? np1 : 1;

    npy_intp ak_dims[2] = {n, n};
    npy_intp ak_strides[2] = {sizeof(f64), n * sizeof(f64)};
    npy_intp bk_dims[2] = {n, np2};
    npy_intp bk_strides[2] = {sizeof(f64), n * sizeof(f64)};
    npy_intp ck_dims[2] = {m2, n};
    npy_intp ck_strides[2] = {sizeof(f64), m2 * sizeof(f64)};
    npy_intp dk_dims[2] = {m2, np2};
    npy_intp dk_strides[2] = {sizeof(f64), m2 * sizeof(f64)};
    npy_intp ac_dims[2] = {n2, n2};
    npy_intp ac_strides[2] = {sizeof(f64), n2 * sizeof(f64)};
    npy_intp bc_dims[2] = {n2, m1};
    npy_intp bc_strides[2] = {sizeof(f64), n2 * sizeof(f64)};
    npy_intp cc_dims[2] = {np1, n2};
    npy_intp cc_strides[2] = {sizeof(f64), np1 * sizeof(f64)};
    npy_intp dc_dims[2] = {np1, m1};
    npy_intp dc_strides[2] = {sizeof(f64), np1 * sizeof(f64)};

    PyObject* ak_arr = PyArray_New(&PyArray_Type, 2, ak_dims, NPY_DOUBLE, ak_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject* bk_arr = PyArray_New(&PyArray_Type, 2, bk_dims, NPY_DOUBLE, bk_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject* ck_arr = PyArray_New(&PyArray_Type, 2, ck_dims, NPY_DOUBLE, ck_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject* dk_arr = PyArray_New(&PyArray_Type, 2, dk_dims, NPY_DOUBLE, dk_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject* ac_arr = PyArray_New(&PyArray_Type, 2, ac_dims, NPY_DOUBLE, ac_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject* bc_arr = PyArray_New(&PyArray_Type, 2, bc_dims, NPY_DOUBLE, bc_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject* cc_arr = PyArray_New(&PyArray_Type, 2, cc_dims, NPY_DOUBLE, cc_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject* dc_arr = PyArray_New(&PyArray_Type, 2, dc_dims, NPY_DOUBLE, dc_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (!ak_arr || !bk_arr || !ck_arr || !dk_arr || !ac_arr || !bc_arr || !cc_arr || !dc_arr) {
        Py_XDECREF(ak_arr); Py_XDECREF(bk_arr); Py_XDECREF(ck_arr); Py_XDECREF(dk_arr);
        Py_XDECREF(ac_arr); Py_XDECREF(bc_arr); Py_XDECREF(cc_arr); Py_XDECREF(dc_arr);
        Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array);
        return PyErr_NoMemory();
    }

    f64* ak = (f64*)PyArray_DATA((PyArrayObject*)ak_arr); memset(ak, 0, n * n * sizeof(f64));
    f64* bk = (f64*)PyArray_DATA((PyArrayObject*)bk_arr); memset(bk, 0, n * np2 * sizeof(f64));
    f64* ck = (f64*)PyArray_DATA((PyArrayObject*)ck_arr); memset(ck, 0, m2 * n * sizeof(f64));
    f64* dk = (f64*)PyArray_DATA((PyArrayObject*)dk_arr); memset(dk, 0, m2 * np2 * sizeof(f64));
    f64* ac = (f64*)PyArray_DATA((PyArrayObject*)ac_arr); memset(ac, 0, n2 * n2 * sizeof(f64));
    f64* bc = (f64*)PyArray_DATA((PyArrayObject*)bc_arr); memset(bc, 0, n2 * m1 * sizeof(f64));
    f64* cc = (f64*)PyArray_DATA((PyArrayObject*)cc_arr); memset(cc, 0, np1 * n2 * sizeof(f64));
    f64* dc = (f64*)PyArray_DATA((PyArrayObject*)dc_arr); memset(dc, 0, np1 * m1 * sizeof(f64));
    f64 rcond[4] = {0.0, 0.0, 0.0, 0.0};

    i32 nn = n * n;
    i32 np11 = np1 - m2;
    i32 m11 = m1 - np2;

    i32 lw1 = n*m + np_arg*n + np_arg*m + m2*m2 + np2*np2;
    i32 lw2a = (n + np1 + 1)*(n + m2) + (3*(n + m2) + n + np1 > 5*(n + m2) ? 3*(n + m2) + n + np1 : 5*(n + m2));
    i32 lw2b = (n + np2)*(n + m1 + 1) + (3*(n + np2) + n + m1 > 5*(n + np2) ? 3*(n + np2) + n + m1 : 5*(n + np2));
    i32 lw2c_max = np1*(n > m1 ? n : m1);
    if (3*m2 + np1 > lw2c_max) lw2c_max = 3*m2 + np1;
    if (5*m2 > lw2c_max) lw2c_max = 5*m2;
    i32 lw2c = m2 + np1*np1 + lw2c_max;
    i32 lw2d_max = (n > np1 ? n : np1)*m1;
    if (3*np2 + m1 > lw2d_max) lw2d_max = 3*np2 + m1;
    if (5*np2 > lw2d_max) lw2d_max = 5*np2;
    i32 lw2d = np2 + m1*m1 + lw2d_max;
    i32 lw2 = lw2a; if (lw2b > lw2) lw2 = lw2b; if (lw2c > lw2) lw2 = lw2c; if (lw2d > lw2) lw2 = lw2d;

    i32 min_np11_m1 = np11 < m1 ? np11 : m1; if (min_np11_m1 < 1) min_np11_m1 = 1;
    i32 max_np11_m1 = np11 > m1 ? np11 : m1; if (max_np11_m1 < 1) max_np11_m1 = 1;
    i32 min_np1_m11 = np1 < m11 ? np1 : m11; if (min_np1_m11 < 1) min_np1_m11 = 1;
    i32 max_np1_m11 = np1 > m11 ? np1 : m11; if (max_np1_m11 < 1) max_np1_m11 = 1;
    i32 lw3a = np11*m1 + (4*min_np11_m1 + max_np11_m1 > 6*min_np11_m1 ? 4*min_np11_m1 + max_np11_m1 : 6*min_np11_m1);
    i32 lw3b = np1*m11 + (4*min_np1_m11 + max_np1_m11 > 6*min_np1_m11 ? 4*min_np1_m11 + max_np1_m11 : 6*min_np1_m11);
    i32 lw3 = lw3a > lw3b ? lw3a : lw3b; if (lw3 < 1) lw3 = 1;

    i32 lw4 = 2*m*m + np_arg*np_arg + 2*m*n + m*np_arg + 2*n*np_arg;
    i32 lw5 = 2*nn + m*n + n*np_arg;

    i32 lw6a_inner = n*m > 10*nn + 12*n + 5 ? n*m : 10*nn + 12*n + 5;
    i32 lw6a = m*m + (2*m1 > 3*nn + lw6a_inner ? 2*m1 : 3*nn + lw6a_inner);
    i32 lw6b_inner = n*np_arg > 10*nn + 12*n + 5 ? n*np_arg : 10*nn + 12*n + 5;
    i32 lw6b = np_arg*np_arg + (2*np1 > 3*nn + lw6b_inner ? 2*np1 : 3*nn + lw6b_inner);
    i32 lw6 = lw6a > lw6b ? lw6a : lw6b;

    i32 lw7a = np11 > 0 ? np11*np11 + (2*np11 > (np11 + m11)*np2 ? 2*np11 : (np11 + m11)*np2) : 1;
    i32 lw7b = m11 > 0 ? m11*m11 + (2*m11 > m11*m2 ? 2*m11 : m11*m2) : 1;
    i32 lw7c = 3*n;
    i32 lw7d_inner2 = m2*m2 + 3*m2 > np2*(2*np2 + m2 + (np2 > n ? np2 : n)) ? m2*m2 + 3*m2 : np2*(2*np2 + m2 + (np2 > n ? np2 : n));
    i32 lw7d_inner = 2*n*m2 > m2*np2 + lw7d_inner2 ? 2*n*m2 : m2*np2 + lw7d_inner2;
    i32 lw7d = n*(2*np2 + m2) + lw7d_inner;
    i32 lw7_max = lw7a > lw7b ? lw7a : lw7b; if (lw7c > lw7_max) lw7_max = lw7c; if (lw7d > lw7_max) lw7_max = lw7d;
    i32 lw7 = m2*np2 + np2*np2 + m2*m2 + lw7_max;

    i32 lw56_max = lw6 > lw7 ? lw6 : lw7;
    i32 lw_other = lw2; if (lw3 > lw_other) lw_other = lw3; if (lw4 > lw_other) lw_other = lw4; if (lw5 + lw56_max > lw_other) lw_other = lw5 + lw56_max;
    i32 ldwork = lw1 + (1 > lw_other ? 1 : lw_other);
    if (ldwork < 1) ldwork = 1;

    i32 max_dims = n; if (m1 > max_dims) max_dims = m1; if (np1 > max_dims) max_dims = np1; if (m2 > max_dims) max_dims = m2; if (np2 > max_dims) max_dims = np2;
    i32 liwork = 2*max_dims > nn ? 2*max_dims : nn;
    if (liwork < 1) liwork = 1;
    i32 lbwork = 2 * n;
    if (lbwork < 1) lbwork = 1;

    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));
    i32* iwork = (i32*)calloc(liwork, sizeof(i32));
    i32* bwork = (i32*)calloc(lbwork, sizeof(i32));

    if (!dwork || !iwork || !bwork) {
        free(dwork); free(iwork); free(bwork);
        Py_DECREF(ak_arr); Py_DECREF(bk_arr); Py_DECREF(ck_arr); Py_DECREF(dk_arr);
        Py_DECREF(ac_arr); Py_DECREF(bc_arr); Py_DECREF(cc_arr); Py_DECREF(dc_arr);
        Py_DECREF(a_array); Py_DECREF(b_array);
        Py_DECREF(c_array); Py_DECREF(d_array);
        return PyErr_NoMemory();
    }

    f64 gamma_out = gamma;
    i32 info = 0;

    sb10ad((i32)job, (i32)n, (i32)m, (i32)np_arg, (i32)ncon, (i32)nmeas, &gamma_out,
           (f64*)PyArray_DATA(a_array), lda,
           (f64*)PyArray_DATA(b_array), ldb,
           (f64*)PyArray_DATA(c_array), ldc,
           (f64*)PyArray_DATA(d_array), ldd,
           ak, ldak, bk, ldbk, ck, ldck, dk, lddk,
           ac, ldac, bc, ldbc, cc, ldcc, dc, lddc,
           rcond, gtol, actol, iwork, liwork, dwork, ldwork, bwork, lbwork, &info);

    free(dwork);
    free(iwork);
    free(bwork);

    npy_intp rcond_dims[1] = {4};
    PyObject* rcond_arr = PyArray_SimpleNew(1, rcond_dims, NPY_DOUBLE);
    memcpy(PyArray_DATA((PyArrayObject*)rcond_arr), rcond, 4 * sizeof(f64));

    PyObject* result = Py_BuildValue("(OOOOOOOOdOi)",
        ak_arr, bk_arr, ck_arr, dk_arr, ac_arr, bc_arr, cc_arr, dc_arr, gamma_out, rcond_arr, info);

    Py_DECREF(a_array); Py_DECREF(b_array);
    Py_DECREF(c_array); Py_DECREF(d_array);
    Py_DECREF(ak_arr); Py_DECREF(bk_arr);
    Py_DECREF(ck_arr); Py_DECREF(dk_arr);
    Py_DECREF(ac_arr); Py_DECREF(bc_arr);
    Py_DECREF(cc_arr); Py_DECREF(dc_arr);
    Py_DECREF(rcond_arr);

    return result;
}


PyObject* py_sb10ld(PyObject* self, PyObject* args) {
    int n, m, np_arg, ncon, nmeas;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *ak_obj, *bk_obj, *ck_obj, *dk_obj;

    if (!PyArg_ParseTuple(args, "iiiiiOOOOOOOO", &n, &m, &np_arg, &ncon, &nmeas,
                          &a_obj, &b_obj, &c_obj, &d_obj, &ak_obj, &bk_obj, &ck_obj, &dk_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *ak_array = (PyArrayObject*)PyArray_FROM_OTF(ak_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *bk_array = (PyArrayObject*)PyArray_FROM_OTF(bk_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *ck_array = (PyArrayObject*)PyArray_FROM_OTF(ck_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *dk_array = (PyArrayObject*)PyArray_FROM_OTF(dk_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);

    if (!a_array || !b_array || !c_array || !d_array ||
        !ak_array || !bk_array || !ck_array || !dk_array) {
        Py_XDECREF(a_array); Py_XDECREF(b_array);
        Py_XDECREF(c_array); Py_XDECREF(d_array);
        Py_XDECREF(ak_array); Py_XDECREF(bk_array);
        Py_XDECREF(ck_array); Py_XDECREF(dk_array);
        return NULL;
    }

    i32 m1 = m - ncon;
    i32 m2 = ncon;
    i32 np1 = np_arg - nmeas;
    i32 np2 = nmeas;
    i32 n2 = 2 * n;

    i32 lda = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldc = np_arg > 1 ? np_arg : 1;
    i32 ldd = np_arg > 1 ? np_arg : 1;
    i32 ldak = n > 1 ? n : 1;
    i32 ldbk = n > 1 ? n : 1;
    i32 ldck = m2 > 1 ? m2 : 1;
    i32 lddk = m2 > 1 ? m2 : 1;
    i32 ldac = n2 > 1 ? n2 : 1;
    i32 ldbc = n2 > 1 ? n2 : 1;
    i32 ldcc = np1 > 1 ? np1 : 1;
    i32 lddc = np1 > 1 ? np1 : 1;

    npy_intp ac_dims[2] = {n2, n2};
    npy_intp ac_strides[2] = {sizeof(f64), n2 * sizeof(f64)};
    npy_intp bc_dims[2] = {n2, m1};
    npy_intp bc_strides[2] = {sizeof(f64), n2 * sizeof(f64)};
    npy_intp cc_dims[2] = {np1, n2};
    npy_intp cc_strides[2] = {sizeof(f64), np1 * sizeof(f64)};
    npy_intp dc_dims[2] = {np1, m1};
    npy_intp dc_strides[2] = {sizeof(f64), np1 * sizeof(f64)};

    PyObject* ac_arr = PyArray_New(&PyArray_Type, 2, ac_dims, NPY_DOUBLE, ac_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject* bc_arr = PyArray_New(&PyArray_Type, 2, bc_dims, NPY_DOUBLE, bc_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject* cc_arr = PyArray_New(&PyArray_Type, 2, cc_dims, NPY_DOUBLE, cc_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject* dc_arr = PyArray_New(&PyArray_Type, 2, dc_dims, NPY_DOUBLE, dc_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (!ac_arr || !bc_arr || !cc_arr || !dc_arr) {
        Py_XDECREF(ac_arr); Py_XDECREF(bc_arr); Py_XDECREF(cc_arr); Py_XDECREF(dc_arr);
        Py_DECREF(a_array); Py_DECREF(b_array);
        Py_DECREF(c_array); Py_DECREF(d_array);
        Py_DECREF(ak_array); Py_DECREF(bk_array);
        Py_DECREF(ck_array); Py_DECREF(dk_array);
        return PyErr_NoMemory();
    }

    f64* ac = (f64*)PyArray_DATA((PyArrayObject*)ac_arr); memset(ac, 0, n2 * n2 * sizeof(f64));
    f64* bc = (f64*)PyArray_DATA((PyArrayObject*)bc_arr); memset(bc, 0, n2 * m1 * sizeof(f64));
    f64* cc = (f64*)PyArray_DATA((PyArrayObject*)cc_arr); memset(cc, 0, np1 * n2 * sizeof(f64));
    f64* dc = (f64*)PyArray_DATA((PyArrayObject*)dc_arr); memset(dc, 0, np1 * m1 * sizeof(f64));

    i32 ldwork = 2*m*m + np_arg*np_arg + 2*m*n + m*np_arg + 2*n*np_arg;
    if (ldwork < 1) ldwork = 1;

    i32 iwork_size = 2 * (m2 > np2 ? m2 : np2);
    if (iwork_size < 1) iwork_size = 1;

    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));
    i32* iwork = (i32*)calloc(iwork_size, sizeof(i32));

    if (!dwork || !iwork) {
        free(dwork); free(iwork);
        Py_DECREF(ac_arr); Py_DECREF(bc_arr); Py_DECREF(cc_arr); Py_DECREF(dc_arr);
        Py_DECREF(a_array); Py_DECREF(b_array);
        Py_DECREF(c_array); Py_DECREF(d_array);
        Py_DECREF(ak_array); Py_DECREF(bk_array);
        Py_DECREF(ck_array); Py_DECREF(dk_array);
        return PyErr_NoMemory();
    }

    i32 info = 0;

    sb10ld((i32)n, (i32)m, (i32)np_arg, (i32)ncon, (i32)nmeas,
           (f64*)PyArray_DATA(a_array), lda,
           (f64*)PyArray_DATA(b_array), ldb,
           (f64*)PyArray_DATA(c_array), ldc,
           (f64*)PyArray_DATA(d_array), ldd,
           (f64*)PyArray_DATA(ak_array), ldak,
           (f64*)PyArray_DATA(bk_array), ldbk,
           (f64*)PyArray_DATA(ck_array), ldck,
           (f64*)PyArray_DATA(dk_array), lddk,
           ac, ldac, bc, ldbc, cc, ldcc, dc, lddc,
           iwork, dwork, ldwork, &info);

    free(dwork);
    free(iwork);

    PyObject* result = Py_BuildValue("(OOOOi)", ac_arr, bc_arr, cc_arr, dc_arr, info);

    Py_DECREF(a_array); Py_DECREF(b_array);
    Py_DECREF(c_array); Py_DECREF(d_array);
    Py_DECREF(ak_array); Py_DECREF(bk_array);
    Py_DECREF(ck_array); Py_DECREF(dk_array);
    Py_DECREF(ac_arr); Py_DECREF(bc_arr);
    Py_DECREF(cc_arr); Py_DECREF(dc_arr);

    return result;
}


/* Module method definitions */
/* Python wrapper for sb10yd */
PyObject* slicot_sb10yd(PyObject* self, PyObject* args) {
    i32 discfl, flag, n, lendat;
    PyObject *rfrdat_obj, *ifrdat_obj, *omega_obj;
    f64 tol = 0.0;

    if (!PyArg_ParseTuple(args, "iiiOOOd|i", &discfl, &flag, &n, &rfrdat_obj, &ifrdat_obj, &omega_obj, &tol, &lendat)) {
        return NULL;
    }

    PyArrayObject *rfrdat_array = (PyArrayObject*)PyArray_FROM_OTF(rfrdat_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ifrdat_array = (PyArrayObject*)PyArray_FROM_OTF(ifrdat_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *omega_array = (PyArrayObject*)PyArray_FROM_OTF(omega_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!rfrdat_array || !ifrdat_array || !omega_array) {
        Py_XDECREF(rfrdat_array);
        Py_XDECREF(ifrdat_array);
        Py_XDECREF(omega_array);
        return NULL;
    }

    if (PyArray_DIM(rfrdat_array, 0) == 0) {
         Py_DECREF(rfrdat_array);
         Py_DECREF(ifrdat_array);
         Py_DECREF(omega_array);
         PyErr_SetString(PyExc_ValueError, "Empty input arrays");
         return NULL;
    }
    lendat = (i32)PyArray_DIM(rfrdat_array, 0);

    f64 *rfrdat = (f64*)PyArray_DATA(rfrdat_array);
    f64 *ifrdat = (f64*)PyArray_DATA(ifrdat_array);
    f64 *omega = (f64*)PyArray_DATA(omega_array);

    i32 lda = (n > 1) ? n : 1;
    f64 *a = (f64*)malloc(lda * (n > 0 ? n : 1) * sizeof(f64));
    f64 *b = (f64*)malloc((n > 0 ? n : 1) * sizeof(f64));
    f64 *c = (f64*)malloc((n > 0 ? n : 1) * sizeof(f64));
    f64 d_val = 0.0;

    i32 ldwork = -1;
    i32 lzwork = -1;
    i32 info = 0;
    
    i32 liwork = (2 * n + 1 > 2) ? 2 * n + 1 : 2;
    i32 *iwork = (i32*)malloc(liwork * sizeof(i32));

    i32 hnpts = 2048;
    i32 lw1 = 2 * lendat + 4 * hnpts;
    i32 lw2 = lendat + 6 * hnpts;
    i32 n2 = 2 * n + 1;
    i32 mn = (2 * lendat < n2) ? 2 * lendat : n2;
    i32 term1 = mn + 6 * n + 4;
    i32 term2 = 2 * mn + 1;
    i32 lw3 = 0;
    if (n > 0)
        lw3 = 2 * lendat * n2 + (2 * lendat > n2 ? 2 * lendat : n2) + (term1 > term2 ? term1 : term2);
    else 
        lw3 = 4 * lendat + 5;
    
    i32 lw4 = 0;
    if (flag == 1) {
        i32 t3 = n * n + 5 * n;
        i32 t4 = 6 * n + 1 + (1 < n ? 1 : n);
        lw4 = (t3 > t4) ? t3 : t4;
    }
    
    ldwork = lw1;
    if (lw2 > ldwork) ldwork = lw2;
    if (lw3 > ldwork) ldwork = lw3;
    if (lw4 > ldwork) ldwork = lw4;
    if (ldwork < 2) ldwork = 2;

    lzwork = (n > 0) ? lendat * (n2 + 2) : lendat;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    c128 *zwork = (c128*)malloc(lzwork * sizeof(c128));

    if (!a || !b || !c || !iwork || !dwork || !zwork) {
         free(a); free(b); free(c); free(iwork); free(dwork); free(zwork);
         Py_DECREF(rfrdat_array); Py_DECREF(ifrdat_array); Py_DECREF(omega_array);
         PyErr_NoMemory();
         return NULL;
    }

    sb10yd(discfl, flag, lendat, rfrdat, ifrdat, omega, &n, a, lda, b, c, &d_val,
           tol, iwork, dwork, ldwork, zwork, lzwork, &info);

    npy_intp dims_a[2] = {n, n};
    npy_intp dims_vec[1] = {n};

    PyObject *a_out, *b_out, *c_out;

    if (n > 0) {
        npy_intp strides_a[2] = {sizeof(f64), n * sizeof(f64)};
        a_out = PyArray_New(&PyArray_Type, 2, dims_a, NPY_DOUBLE, strides_a, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        b_out = PyArray_SimpleNew(1, dims_vec, NPY_DOUBLE);
        c_out = PyArray_SimpleNew(1, dims_vec, NPY_DOUBLE);

        if (a_out && b_out && c_out) {
            memcpy(PyArray_DATA((PyArrayObject*)a_out), a, n * n * sizeof(f64));
            memcpy(PyArray_DATA((PyArrayObject*)b_out), b, n * sizeof(f64));
            memcpy(PyArray_DATA((PyArrayObject*)c_out), c, n * sizeof(f64));
        }
    } else {
        npy_intp zdims[2] = {0, 0};
        a_out = PyArray_SimpleNew(2, zdims, NPY_DOUBLE);
        b_out = PyArray_SimpleNew(1, zdims, NPY_DOUBLE);
        c_out = PyArray_SimpleNew(1, zdims, NPY_DOUBLE);
    }
    
    free(a);
    free(b);
    free(c);
    free(iwork);
    free(dwork);
    free(zwork);
    Py_DECREF(rfrdat_array);
    Py_DECREF(ifrdat_array);
    Py_DECREF(omega_array);

    return Py_BuildValue("OOOdii", a_out, b_out, c_out, d_val, n, info);
}

PyObject* py_sb01dd(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"a", "b", "indcon", "nblk", "wr", "wi", "z", "y", "tol", NULL};

    PyObject *a_obj, *b_obj, *nblk_obj, *wr_obj, *wi_obj, *z_obj, *y_obj;
    int indcon;
    double tol = 0.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOiOOOOO|d", kwlist,
            &a_obj, &b_obj, &indcon, &nblk_obj, &wr_obj, &wi_obj, &z_obj, &y_obj, &tol)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *nblk_array = (PyArrayObject*)PyArray_FROM_OTF(nblk_obj, NPY_INT32,
        NPY_ARRAY_CARRAY_RO);
    PyArrayObject *wr_array = (PyArrayObject*)PyArray_FROM_OTF(wr_obj, NPY_DOUBLE,
        NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *wi_array = (PyArrayObject*)PyArray_FROM_OTF(wi_obj, NPY_DOUBLE,
        NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *z_array = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE,
        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE,
        NPY_ARRAY_CARRAY_RO);

    if (!a_array || !b_array || !nblk_array || !wr_array || !wi_array || !z_array || !y_array) {
        Py_XDECREF(a_array); Py_XDECREF(b_array); Py_XDECREF(nblk_array);
        Py_XDECREF(wr_array); Py_XDECREF(wi_array); Py_XDECREF(z_array); Py_XDECREF(y_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 m = (i32)PyArray_DIM(b_array, 1);
    i32 lda = (n > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 ldb = (n > 0) ? (i32)PyArray_DIM(b_array, 0) : 1;
    i32 ldz = (n > 0) ? (i32)PyArray_DIM(z_array, 0) : 1;
    i32 ldg = (m > 0) ? m : 1;

    f64 *a = (f64*)PyArray_DATA(a_array);
    f64 *b = (f64*)PyArray_DATA(b_array);
    i32 *nblk = (i32*)PyArray_DATA(nblk_array);
    f64 *wr = (f64*)PyArray_DATA(wr_array);
    f64 *wi = (f64*)PyArray_DATA(wi_array);
    f64 *z = (f64*)PyArray_DATA(z_array);
    f64 *y = (f64*)PyArray_DATA(y_array);

    i32 ldwork = (m*n > m*m + 2*n + 4*m + 1) ? m*n : m*m + 2*n + 4*m + 1;
    if (ldwork < 1) ldwork = 1;

    npy_intp dims_g[2] = {m, n};
    npy_intp strides_g[2] = {sizeof(f64), ldg * sizeof(f64)};
    PyObject *g_array = PyArray_New(&PyArray_Type, 2, dims_g, NPY_DOUBLE, strides_g,
                                     NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!g_array) {
        Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(nblk_array);
        Py_DECREF(wr_array); Py_DECREF(wi_array); Py_DECREF(z_array); Py_DECREF(y_array);
        return NULL;
    }
    f64 *g = (f64*)PyArray_DATA((PyArrayObject*)g_array);
    memset(g, 0, ldg * n * sizeof(f64));

    i32 *iwork = (i32*)calloc((size_t)(m > 0 ? m : 1), sizeof(i32));
    f64 *dwork = (f64*)calloc((size_t)ldwork, sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork); free(dwork);
        Py_DECREF(g_array);
        Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(nblk_array);
        Py_DECREF(wr_array); Py_DECREF(wi_array); Py_DECREF(z_array); Py_DECREF(y_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 count, info;
    sb01dd(n, m, indcon, a, lda, b, ldb, nblk, wr, wi, z, ldz, y,
           &count, g, ldg, tol, iwork, dwork, ldwork, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(wr_array);
    PyArray_ResolveWritebackIfCopy(wi_array);
    PyArray_ResolveWritebackIfCopy(z_array);
    
    // g_array already created

    free(iwork);
    free(dwork);

    PyObject *result = Py_BuildValue("OOOOii", a_array, b_array, z_array, g_array, count, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(nblk_array);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);
    Py_DECREF(z_array);
    Py_DECREF(y_array);

    return result;
}

/* Python wrapper for sb01md */
PyObject* py_sb01md(PyObject* self, PyObject* args) {
    (void)self;

    i32 ncont, n;
    PyObject *a_obj, *b_obj, *wr_obj, *wi_obj, *z_obj;

    if (!PyArg_ParseTuple(args, "iiOOOOO", &ncont, &n, &a_obj, &b_obj,
                          &wr_obj, &wi_obj, &z_obj)) {
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

    PyArrayObject *wr_array = (PyArrayObject*)PyArray_FROM_OTF(wr_obj, NPY_DOUBLE,
                                                                NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (wr_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *wi_array = (PyArrayObject*)PyArray_FROM_OTF(wi_obj, NPY_DOUBLE,
                                                                NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (wi_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(wr_array);
        return NULL;
    }

    PyArrayObject *z_array = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (z_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(wr_array);
        Py_DECREF(wi_array);
        return NULL;
    }

    i32 lda = (ncont > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 ldz = (n > 0) ? (i32)PyArray_DIM(z_array, 0) : 1;

    f64 *a = (f64*)PyArray_DATA(a_array);
    f64 *b = (f64*)PyArray_DATA(b_array);
    const f64 *wr = (const f64*)PyArray_DATA(wr_array);
    const f64 *wi = (const f64*)PyArray_DATA(wi_array);
    f64 *z = (f64*)PyArray_DATA(z_array);

    i32 ldwork = 3 * ncont;
    if (ldwork < 1) ldwork = 1;

    npy_intp g_dims[1] = {ncont > 0 ? ncont : 0};
    PyObject *g_array = PyArray_New(&PyArray_Type, 1, g_dims, NPY_DOUBLE, NULL,
                                     NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!g_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(wr_array);
        Py_DECREF(wi_array);
        Py_DECREF(z_array);
        return PyErr_NoMemory();
    }
    f64 *g = (f64*)PyArray_DATA((PyArrayObject*)g_array);
    if (ncont > 0) {
        memset(g, 0, (size_t)ncont * sizeof(f64));
    }

    f64 *dwork = (f64*)calloc((size_t)ldwork, sizeof(f64));

    if (!dwork) {
        free(dwork);
        Py_DECREF(g_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(wr_array);
        Py_DECREF(wi_array);
        Py_DECREF(z_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info;
    sb01md(ncont, n, a, lda, b, wr, wi, z, ldz, g, dwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(z_array);

    // g_array already created

    PyObject *result = Py_BuildValue("OOOOi", a_array, b_array, z_array, g_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);
    Py_DECREF(z_array);
    Py_DECREF(g_array);

    return result;
}

PyObject* py_sb02cx(PyObject* self, PyObject* args)
{
    (void)self;
    double reig, ieig;

    if (!PyArg_ParseTuple(args, "dd", &reig, &ieig)) {
        return NULL;
    }

    int result = sb02cx(&reig, &ieig);

    return PyBool_FromLong(result);
}

PyObject* py_sb02mr(PyObject* self, PyObject* args)
{
    (void)self;
    double reig, ieig;

    if (!PyArg_ParseTuple(args, "dd", &reig, &ieig)) {
        return NULL;
    }

    int result = sb02mr(&reig, &ieig);

    return PyBool_FromLong(result);
}

PyObject* py_sb02ms(PyObject* self, PyObject* args)
{
    (void)self;
    double reig, ieig;

    if (!PyArg_ParseTuple(args, "dd", &reig, &ieig)) {
        return NULL;
    }

    int result = sb02ms(&reig, &ieig);

    return PyBool_FromLong(result);
}

PyObject* py_sb02mv(PyObject* self, PyObject* args)
{
    (void)self;
    double reig, ieig;

    if (!PyArg_ParseTuple(args, "dd", &reig, &ieig)) {
        return NULL;
    }

    int result = sb02mv(&reig, &ieig);

    return PyBool_FromLong(result);
}

PyObject* py_sb02mw(PyObject* self, PyObject* args)
{
    (void)self;
    double reig, ieig;

    if (!PyArg_ParseTuple(args, "dd", &reig, &ieig)) {
        return NULL;
    }

    int result = sb02mw(&reig, &ieig);

    return PyBool_FromLong(result);
}

PyObject* py_sb02ou(PyObject* self, PyObject* args)
{
    (void)self;
    double alphar, alphai, beta;

    if (!PyArg_ParseTuple(args, "ddd", &alphar, &alphai, &beta)) {
        return NULL;
    }

    int result = sb02ou(&alphar, &alphai, &beta);

    return PyBool_FromLong(result);
}

PyObject* py_sb02mx(PyObject* self, PyObject* args) {
    (void)self;
    const char *jobg_str, *jobl_str, *fact_str, *uplo_str;
    const char *trans_str, *flag_str, *def_str;
    i32 n, m;
    PyObject *a_obj, *b_obj, *q_obj, *r_obj, *l_obj, *g_obj;
    PyArrayObject *a_array = NULL, *b_array = NULL, *q_array = NULL;
    PyArrayObject *r_array = NULL, *l_array = NULL, *g_array = NULL;
    i32 oufact, info;

    if (!PyArg_ParseTuple(args, "sssssssiiOOOOOO",
                          &jobg_str, &jobl_str, &fact_str, &uplo_str,
                          &trans_str, &flag_str, &def_str,
                          &n, &m, &a_obj, &b_obj, &q_obj, &r_obj, &l_obj, &g_obj)) {
        return NULL;
    }

    char jobg = toupper((unsigned char)jobg_str[0]);
    char jobl = toupper((unsigned char)jobl_str[0]);
    char fact = toupper((unsigned char)fact_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);
    char trans = toupper((unsigned char)trans_str[0]);
    char flag_ch = toupper((unsigned char)flag_str[0]);
    char def = toupper((unsigned char)def_str[0]);

    bool ljobg = (jobg == 'G');
    bool ljobl = (jobl == 'N');
    bool lfactc = (fact == 'C');
    bool lfactu = (fact == 'U');
    bool lnfact = (!lfactc && !lfactu);

    if (!ljobg && jobg != 'N') {
        PyErr_SetString(PyExc_ValueError, "Parameter 1 (JOBG) must be 'G' or 'N'");
        return NULL;
    }
    if (!ljobl && jobl != 'Z') {
        PyErr_SetString(PyExc_ValueError, "Parameter 2 (JOBL) must be 'Z' or 'N'");
        return NULL;
    }
    if (lnfact && fact != 'N') {
        PyErr_SetString(PyExc_ValueError, "Parameter 3 (FACT) must be 'N', 'C', or 'U'");
        return NULL;
    }
    if (uplo != 'U' && uplo != 'L') {
        PyErr_SetString(PyExc_ValueError, "Parameter 4 (UPLO) must be 'U' or 'L'");
        return NULL;
    }
    if (trans != 'N' && trans != 'T' && trans != 'C') {
        PyErr_SetString(PyExc_ValueError, "Parameter 5 (TRANS) must be 'N', 'T', or 'C'");
        return NULL;
    }
    if (flag_ch != 'P' && flag_ch != 'M') {
        PyErr_SetString(PyExc_ValueError, "Parameter 6 (FLAG) must be 'P' or 'M'");
        return NULL;
    }
    if (def != 'D' && def != 'I' && lnfact) {
        PyErr_SetString(PyExc_ValueError, "Parameter 7 (DEF) must be 'D' or 'I' when FACT='N'");
        return NULL;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }
    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be non-negative");
        return NULL;
    }

    i32 lda = 1, ldb = 1, ldq = 1, ldr = 1, ldl = 1, ldg = 1;
    f64 *a_data = NULL, *b_data = NULL, *q_data = NULL;
    f64 *r_data = NULL, *l_data = NULL, *g_data = NULL;

    if (ljobl && a_obj != Py_None) {
        a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!a_array) goto cleanup;
        lda = (i32)PyArray_DIM(a_array, 0);
        a_data = (f64*)PyArray_DATA(a_array);
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) goto cleanup;
    ldb = (i32)PyArray_DIM(b_array, 0);
    b_data = (f64*)PyArray_DATA(b_array);

    if (ljobl && q_obj != Py_None) {
        q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!q_array) goto cleanup;
        ldq = (i32)PyArray_DIM(q_array, 0);
        q_data = (f64*)PyArray_DATA(q_array);
    }

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!r_array) goto cleanup;
    ldr = (i32)PyArray_DIM(r_array, 0);
    r_data = (f64*)PyArray_DATA(r_array);

    if (ljobl && l_obj != Py_None) {
        l_array = (PyArrayObject*)PyArray_FROM_OTF(l_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!l_array) goto cleanup;
        ldl = (i32)PyArray_DIM(l_array, 0);
        l_data = (f64*)PyArray_DATA(l_array);
    }

    if (ljobg && g_obj != Py_None) {
        g_array = (PyArrayObject*)PyArray_FROM_OTF(g_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!g_array) goto cleanup;
        ldg = (i32)PyArray_DIM(g_array, 0);
        g_data = (f64*)PyArray_DATA(g_array);
    }

    i32 ldwork;
    if (lfactc) {
        ldwork = 1;
    } else if (lfactu) {
        ldwork = (ljobg || ljobl) ? ((n * m > 1) ? n * m : 1) : 1;
    } else {
        if (ljobg || ljobl) {
            i32 nm = n * m;
            i32 tmp = 3 * m > nm ? 3 * m : nm;
            ldwork = tmp > 2 ? tmp : 2;
        } else {
            ldwork = 3 * m > 2 ? 3 * m : 2;
        }
    }

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    i32 *iwork = (i32*)malloc((m > 0 ? m : 1) * sizeof(i32));
    i32 *ipiv = (i32*)malloc((m > 0 ? m : 1) * sizeof(i32));

    if (!dwork || !iwork || !ipiv) {
        free(dwork); free(iwork); free(ipiv);
        PyErr_NoMemory();
        goto cleanup;
    }

    sb02mx(jobg_str, jobl_str, fact_str, uplo_str, trans_str, flag_str, def_str,
           n, m, a_data, lda, b_data, ldb, q_data, ldq,
           r_data, ldr, l_data, ldl, ipiv, &oufact, g_data, ldg,
           iwork, dwork, ldwork, &info);

    free(dwork);
    free(iwork);
    free(ipiv);

    if (a_array) PyArray_ResolveWritebackIfCopy(a_array);
    if (b_array) PyArray_ResolveWritebackIfCopy(b_array);
    if (q_array) PyArray_ResolveWritebackIfCopy(q_array);
    if (r_array) PyArray_ResolveWritebackIfCopy(r_array);
    if (l_array) PyArray_ResolveWritebackIfCopy(l_array);
    if (g_array) PyArray_ResolveWritebackIfCopy(g_array);

    if (info < 0) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(q_array);
        Py_XDECREF(r_array);
        Py_XDECREF(l_array);
        Py_XDECREF(g_array);
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    PyObject *result;
    if (ljobg && !ljobl) {
        result = Py_BuildValue("Oii", g_array, oufact, info);
    } else if (!ljobg && ljobl) {
        result = Py_BuildValue("OOOOii", a_array, b_array, q_array, l_array, oufact, info);
    } else if (ljobg && ljobl) {
        result = Py_BuildValue("OOOOOii", a_array, b_array, q_array, l_array, g_array, oufact, info);
    } else {
        result = Py_BuildValue("ii", oufact, info);
    }

    Py_XDECREF(a_array);
    Py_XDECREF(b_array);
    Py_XDECREF(q_array);
    Py_XDECREF(r_array);
    Py_XDECREF(l_array);
    Py_XDECREF(g_array);

    return result;

cleanup:
    Py_XDECREF(a_array);
    Py_XDECREF(b_array);
    Py_XDECREF(q_array);
    Py_XDECREF(r_array);
    Py_XDECREF(l_array);
    Py_XDECREF(g_array);
    return NULL;
}

PyObject* py_sb02ov(PyObject* self, PyObject* args)
{
    (void)self;
    double alphar, alphai, beta;

    if (!PyArg_ParseTuple(args, "ddd", &alphar, &alphai, &beta)) {
        return NULL;
    }

    int result = sb02ov(&alphar, &alphai, &beta);

    return PyBool_FromLong(result);
}

PyObject* py_sb02ow(PyObject* self, PyObject* args)
{
    (void)self;
    double alphar, alphai, beta;

    if (!PyArg_ParseTuple(args, "ddd", &alphar, &alphai, &beta)) {
        return NULL;
    }

    int result = sb02ow(&alphar, &alphai, &beta);

    return PyBool_FromLong(result);
}

PyObject* py_sb02ox(PyObject* self, PyObject* args)
{
    (void)self;
    double alphar, alphai, beta;

    if (!PyArg_ParseTuple(args, "ddd", &alphar, &alphai, &beta)) {
        return NULL;
    }

    int result = sb02ox(&alphar, &alphai, &beta);

    return PyBool_FromLong(result);
}

/* Python wrapper for sb03mu */
PyObject* py_sb03mu(PyObject* self, PyObject* args) {
    (void)self;
    int ltranl_int, ltranr_int, isgn;
    PyObject *tl_obj, *tr_obj, *b_obj;
    PyArrayObject *tl_array = NULL, *tr_array = NULL, *b_array = NULL;

    if (!PyArg_ParseTuple(args, "ppiOOO", &ltranl_int, &ltranr_int, &isgn,
                          &tl_obj, &tr_obj, &b_obj)) {
        return NULL;
    }

    bool ltranl = (bool)ltranl_int;
    bool ltranr = (bool)ltranr_int;

    tl_array = (PyArrayObject*)PyArray_FROM_OTF(tl_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (tl_array == NULL) goto cleanup;

    tr_array = (PyArrayObject*)PyArray_FROM_OTF(tr_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (tr_array == NULL) goto cleanup;

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (b_array == NULL) goto cleanup;

    i32 n1 = (i32)PyArray_DIM(tl_array, 0);
    i32 n2 = (i32)PyArray_DIM(tr_array, 0);

    i32 ldtl = (n1 > 0) ? n1 : 1;
    i32 ldtr = (n2 > 0) ? n2 : 1;
    i32 ldb = (n1 > 0) ? n1 : 1;
    i32 ldx = (n1 > 0) ? n1 : 1;

    f64* tl = (f64*)PyArray_DATA(tl_array);
    f64* tr = (f64*)PyArray_DATA(tr_array);
    f64* b = (f64*)PyArray_DATA(b_array);

    f64* x = NULL;
    npy_intp x_dims[2] = {n1, n2};
    npy_intp x_strides[2] = {sizeof(f64), ldx * sizeof(f64)};
    
    PyArrayObject* x_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, x_dims, NPY_DOUBLE, x_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (x_array == NULL) {
        goto cleanup;
    }
    x = (f64*)PyArray_DATA(x_array);
    memset(x, 0, x_dims[0] * x_dims[1] * sizeof(f64));

    f64 scale, xnorm;
    i32 info;

    sb03mu(ltranl, ltranr, isgn, n1, n2, tl, ldtl, tr, ldtr, b, ldb,
           &scale, x, ldx, &xnorm, &info);

    // x_array already created

    PyObject* result = Py_BuildValue("Oddi", x_array, scale, xnorm, info);

    Py_DECREF(x_array);
    Py_DECREF(tl_array);
    Py_DECREF(tr_array);
    Py_DECREF(b_array);

    return result;

cleanup:
    Py_XDECREF(tl_array);
    Py_XDECREF(tr_array);
    Py_XDECREF(b_array);
    return NULL;
}


/* Python wrapper for sb03pd */
PyObject* py_sb03pd(PyObject* self, PyObject* args) {
    (void)self;
    char* job;
    char* fact;
    char* trana;
    PyObject *a_obj, *c_obj;
    PyObject *u_obj = Py_None;
    PyArrayObject *a_array = NULL, *c_array = NULL, *u_array = NULL;

    if (!PyArg_ParseTuple(args, "sssOO|O", &job, &fact, &trana, &a_obj, &c_obj, &u_obj)) {
        return NULL;
    }

    char job_c = (char)toupper((unsigned char)job[0]);
    char fact_c = (char)toupper((unsigned char)fact[0]);
    bool wantx = (job_c == 'X');
    bool wantsp = (job_c == 'S');
    bool wantbh = (job_c == 'B');
    bool nofact = (fact_c == 'N');

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 lda = (n > 1) ? n : 1;
    i32 ldu = lda;
    i32 ldc = wantsp ? 1 : lda;

    if (!wantsp) {
        c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (c_array == NULL) {
            Py_DECREF(a_array);
            return NULL;
        }
    }

    f64* u = NULL;
    PyArrayObject* u_out_array = NULL;
    npy_intp dims_nn[2] = {n, n};
    npy_intp strides_nn[2] = {sizeof(f64), ldu * sizeof(f64)}; /* ldu is used but defined later. Wait. */
    /* ldu = lda. defined at 5993. So valid. */

    if (nofact) {
        u_out_array = (PyArrayObject*)PyArray_New(
            &PyArray_Type, 2, dims_nn, NPY_DOUBLE, strides_nn, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (u_out_array == NULL) {
            Py_DECREF(a_array);
            Py_XDECREF(c_array);
            return PyErr_NoMemory();
        }
        u = (f64*)PyArray_DATA(u_out_array);
        if (n > 0) memset(u, 0, n * n * sizeof(f64));
    } else {
        if (u_obj == Py_None) {
            PyErr_SetString(PyExc_ValueError, "U matrix required when fact='F'");
            Py_DECREF(a_array);
            Py_XDECREF(c_array);
            return NULL;
        }
        u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (u_array == NULL) {
            Py_DECREF(a_array);
            Py_XDECREF(c_array);
            return NULL;
        }
        u = (f64*)PyArray_DATA(u_array);
        u_out_array = u_array;
        Py_INCREF(u_out_array);
    }

    f64* a = (f64*)PyArray_DATA(a_array);
    f64* c = wantsp ? NULL : (f64*)PyArray_DATA(c_array);

    npy_intp dims_n[1] = {n};
    PyArrayObject* wr_array = (PyArrayObject*)PyArray_SimpleNew(1, dims_n, NPY_DOUBLE);
    PyArrayObject* wi_array = (PyArrayObject*)PyArray_SimpleNew(1, dims_n, NPY_DOUBLE);
    if (!wr_array || !wi_array) {
        Py_XDECREF(wr_array); Py_XDECREF(wi_array);
        Py_XDECREF(u_out_array); Py_XDECREF(u_array);
        Py_DECREF(a_array); Py_XDECREF(c_array);
        return PyErr_NoMemory();
    }
    f64* wr = (f64*)PyArray_DATA(wr_array);
    f64* wi = (f64*)PyArray_DATA(wi_array);
    if (n > 0) {
        memset(wr, 0, n * sizeof(f64));
        memset(wi, 0, n * sizeof(f64));
    }

    i32* iwork = (i32*)calloc((n > 0 ? n * n : 1), sizeof(i32));

    i32 ldwork;
    if (wantx) {
        if (nofact) {
            ldwork = (n * n > 3 * n) ? n * n : 3 * n;
        } else {
            ldwork = (n * n > 2 * n) ? n * n : 2 * n;
        }
    } else {
        ldwork = 2 * n * n + 2 * n;
    }
    if (ldwork < 1) ldwork = 1;
    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (iwork == NULL || dwork == NULL) {
        free(iwork); free(dwork);
        Py_DECREF(wr_array); Py_DECREF(wi_array);
        Py_XDECREF(u_out_array); Py_XDECREF(u_array);
        Py_DECREF(a_array); Py_XDECREF(c_array);
        return PyErr_NoMemory();
    }

    f64 scale = 1.0, sepd = 0.0, ferr = 0.0;
    i32 info = 0;

    f64* c_work = wantsp ? dwork : c;
    i32 ldc_work = wantsp ? n : ldc;
    if (ldc_work < 1) ldc_work = 1;

    sb03pd(job, fact, trana, n, a, lda, u, ldu, c_work, ldc_work,
           &scale, &sepd, &ferr, wr, wi, iwork, dwork, ldwork, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    if (c_array != NULL) {
        PyArray_ResolveWritebackIfCopy(c_array);
    }
    if (u_array != NULL) {
        PyArray_ResolveWritebackIfCopy(u_array);
    }

    // u_out_array, wr_array, wi_array are already created and valid.
    // u_out_array logic handled above (CREATED or INCREF'd).
    // wait, u_out_array logic:
    // If nofact: Created new. Ref count 1.
    // If !nofact: u_out_array = u_array. INCREF. Ref count 2 (param + incref).
    // Logic matches original expectation?
    // Original:
    // !nofact => u_out_array = u_array; INCREF.
    // Return Py_BuildValue(..., u_out_array) -> Increments ref count.
    
    // We should be fine.

    free(iwork);
    free(dwork);

    PyObject* result;
    if (wantsp) {
        result = Py_BuildValue("OOOddOOi",
                               a_array, u_out_array, Py_None,
                               scale, sepd, wr_array, wi_array, info);
    } else if (wantbh) {
        result = Py_BuildValue("OOOdddOOi",
                               a_array, u_out_array, c_array,
                               scale, sepd, ferr, wr_array, wi_array, info);
    } else {
        result = Py_BuildValue("OOOdOOi",
                               a_array, u_out_array, c_array,
                               scale, wr_array, wi_array, info);
    }

    Py_DECREF(a_array);
    Py_XDECREF(c_array);
    Py_DECREF(u_out_array);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);

    return result;
}

/*
 * SB03QX - Estimate forward error bound for continuous-time Lyapunov equation
 *
 * Args:
 *   trana (str): 'N', 'T', or 'C'
 *   uplo (str): 'U' or 'L'
 *   lyapun (str): 'O' or 'R'
 *   n (int): Order of matrices
 *   xanorm (float): Max-norm of solution X
 *   t (ndarray): N-by-N upper quasi-triangular Schur form
 *   u (ndarray): N-by-N orthogonal matrix (if lyapun='O')
 *   r (ndarray): N-by-N residual matrix
 *
 * Returns:
 *   ferr (float): Forward error bound
 *   r (ndarray): Symmetrized residual matrix
 *   info (int): 0=success, N+1=nearly common eigenvalues
 */
PyObject* py_sb03qx(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"trana", "uplo", "lyapun", "n", "xanorm",
                             "t", "u", "r", NULL};

    const char* trana;
    const char* uplo;
    const char* lyapun;
    int n;
    double xanorm;
    PyObject* t_obj;
    PyObject* u_obj;
    PyObject* r_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssidOOO", kwlist,
                                     &trana, &uplo, &lyapun, &n, &xanorm,
                                     &t_obj, &u_obj, &r_obj)) {
        return NULL;
    }

    PyArrayObject* t_array = (PyArrayObject*)PyArray_FROM_OTF(
        t_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (t_array == NULL) {
        return NULL;
    }

    PyArrayObject* u_array = (PyArrayObject*)PyArray_FROM_OTF(
        u_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (u_array == NULL) {
        Py_DECREF(t_array);
        return NULL;
    }

    PyArrayObject* r_array = (PyArrayObject*)PyArray_FROM_OTF(
        r_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) {
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        return NULL;
    }

    i32 ldt = (n > 1) ? n : 1;
    i32 ldu = (n > 1) ? n : 1;
    i32 ldr = (n > 1) ? n : 1;
    i32 nn = n * n;
    i32 ldwork = 2 * nn;

    f64* t = (f64*)PyArray_DATA(t_array);
    f64* u = (f64*)PyArray_DATA(u_array);
    f64* r = (f64*)PyArray_DATA(r_array);

    i32* iwork = (i32*)calloc(nn > 1 ? nn : 1, sizeof(i32));
    f64* dwork = (f64*)calloc(ldwork > 1 ? ldwork : 1, sizeof(f64));
    if (iwork == NULL || dwork == NULL) {
        free(iwork);
        free(dwork);
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        Py_DECREF(r_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 ferr;
    i32 info;

    sb03qx(trana, uplo, lyapun, n, xanorm, t, ldt, u, ldu,
           r, ldr, &ferr, iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(r_array);

    PyObject* result = Py_BuildValue("dOi", ferr, r_array, info);

    Py_DECREF(t_array);
    Py_DECREF(u_array);
    Py_DECREF(r_array);

    return result;
}


/*
 * SB03QY - Estimate separation and 1-norm of Theta for continuous-time Lyapunov
 *
 * Args:
 *   job (str): 'S', 'T', or 'B'
 *   trana (str): 'N', 'T', or 'C'
 *   lyapun (str): 'O' or 'R'
 *   t (ndarray): N-by-N upper quasi-triangular Schur form
 *   u (ndarray): N-by-N orthogonal matrix (if lyapun='O')
 *   x (ndarray): N-by-N solution matrix (if job='T' or 'B')
 *
 * Returns:
 *   sep (float): Estimated separation (if job='S' or 'B')
 *   thnorm (float): Estimated 1-norm of Theta (if job='T' or 'B')
 *   info (int): 0=success, N+1=nearly common eigenvalues
 */
PyObject* py_sb03qy(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"job", "trana", "lyapun", "t", "u", "x", NULL};

    const char* job;
    const char* trana;
    const char* lyapun;
    PyObject* t_obj;
    PyObject* u_obj;
    PyObject* x_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssOOO", kwlist,
                                     &job, &trana, &lyapun,
                                     &t_obj, &u_obj, &x_obj)) {
        return NULL;
    }

    PyArrayObject* t_array = (PyArrayObject*)PyArray_FROM_OTF(
        t_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (t_array == NULL) {
        return NULL;
    }

    PyArrayObject* u_array = (PyArrayObject*)PyArray_FROM_OTF(
        u_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (u_array == NULL) {
        Py_DECREF(t_array);
        return NULL;
    }

    PyArrayObject* x_array = (PyArrayObject*)PyArray_FROM_OTF(
        x_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (x_array == NULL) {
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(t_array, 0);
    i32 ldt = (n > 1) ? n : 1;
    i32 ldu = (n > 1) ? n : 1;
    i32 ldx = (n > 1) ? n : 1;
    i32 nn = n * n;
    i32 ldwork = 2 * nn;

    f64* t = (f64*)PyArray_DATA(t_array);
    f64* u = (f64*)PyArray_DATA(u_array);
    f64* x = (f64*)PyArray_DATA(x_array);

    i32* iwork = (i32*)calloc(nn > 1 ? nn : 1, sizeof(i32));
    f64* dwork = (f64*)calloc(ldwork > 1 ? ldwork : 1, sizeof(f64));
    if (iwork == NULL || dwork == NULL) {
        free(iwork);
        free(dwork);
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        Py_DECREF(x_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 sep = 0.0;
    f64 thnorm = 0.0;
    i32 info;

    sb03qy(job, trana, lyapun, n, t, ldt, u, ldu, x, ldx,
           &sep, &thnorm, iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyObject* result = Py_BuildValue("ddi", sep, thnorm, info);

    Py_DECREF(t_array);
    Py_DECREF(u_array);
    Py_DECREF(x_array);

    return result;
}


/*
 * SB03RD - Solution of continuous-time Lyapunov equations with separation estimation.
 *
 * Solves: op(A)' * X + X * op(A) = scale * C
 * where op(A) = A or A^T, C is symmetric.
 *
 * Args:
 *   job (str): 'X' solution only, 'S' separation only, 'B' both
 *   fact (str): 'F' Schur provided, 'N' compute Schur
 *   trana (str): 'N', 'T', or 'C'
 *   a (ndarray): N-by-N matrix A
 *   c (ndarray): N-by-N symmetric matrix C
 *   u (ndarray, optional): N-by-N orthogonal matrix (if fact='F')
 *
 * Returns:
 *   a_out (ndarray): Schur form of A
 *   u (ndarray): Orthogonal Schur factor
 *   x (ndarray): Solution matrix (c overwritten)
 *   scale (float): Scale factor
 *   sep (float): Separation estimate
 *   ferr (float): Forward error bound
 *   wr (ndarray): Real parts of eigenvalues
 *   wi (ndarray): Imaginary parts of eigenvalues
 *   info (int): 0=success
 */
PyObject* py_sb03rd(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"job", "fact", "trana", "a", "c", "u", NULL};

    const char* job;
    const char* fact;
    const char* trana;
    PyObject* a_obj;
    PyObject* c_obj;
    PyObject* u_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssOO|O", kwlist,
                                     &job, &fact, &trana, &a_obj, &c_obj, &u_obj)) {
        return NULL;
    }

    bool nofact = (fact[0] == 'N' || fact[0] == 'n');
    bool wantx = (job[0] == 'X' || job[0] == 'x');
    bool wantsp = (job[0] == 'S' || job[0] == 's');

    PyArrayObject* a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 lda = (n > 1) ? n : 1;
    i32 ldu = (n > 1) ? n : 1;
    i32 ldc = wantsp ? 1 : ((n > 1) ? n : 1);

    PyArrayObject* c_array = NULL;
    f64* c_data = NULL;
    if (!wantsp) {
        c_array = (PyArrayObject*)PyArray_FROM_OTF(
            c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (c_array == NULL) {
            Py_DECREF(a_array);
            return NULL;
        }
        c_data = (f64*)PyArray_DATA(c_array);
    } else {
        c_data = (f64*)calloc(1, sizeof(f64));
        if (c_data == NULL) {
            Py_DECREF(a_array);
            PyErr_NoMemory();
            return NULL;
        }
    }

    PyArrayObject* u_array = NULL;
    f64* u_data = NULL;
    if (!nofact && u_obj != Py_None) {
        u_array = (PyArrayObject*)PyArray_FROM_OTF(
            u_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (u_array == NULL) {
            Py_DECREF(a_array);
            if (c_array) Py_DECREF(c_array);
            if (wantsp) free(c_data);
            return NULL;
        }
        u_data = (f64*)PyArray_DATA(u_array);
    }

    f64* a_data = (f64*)PyArray_DATA(a_array);

    i32 nn = n * n;

    PyArrayObject* u_out_array = NULL;
    if (u_array == NULL) {
        npy_intp u_dims[2] = {n, n};
        npy_intp u_strides[2] = {sizeof(f64), n * sizeof(f64)};
        u_out_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE,
                                                   u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (u_out_array == NULL) {
            if (wantsp) free(c_data);
            Py_DECREF(a_array);
            if (c_array) Py_DECREF(c_array);
            return NULL;
        }
        u_data = (f64*)PyArray_DATA(u_out_array);
        if (n * n > 0) memset(u_data, 0, n * n * sizeof(f64));
    }

    npy_intp eig_dims[1] = {n};
    npy_intp eig_strides[1] = {sizeof(f64)};
    PyArrayObject* wr_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 1, eig_dims, NPY_DOUBLE,
                                                           eig_strides, NULL, 0, NPY_ARRAY_CARRAY, NULL);
    PyArrayObject* wi_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 1, eig_dims, NPY_DOUBLE,
                                                           eig_strides, NULL, 0, NPY_ARRAY_CARRAY, NULL);
    if (wr_array == NULL || wi_array == NULL) {
        Py_XDECREF(wr_array);
        Py_XDECREF(wi_array);
        Py_XDECREF(u_out_array);
        if (wantsp) free(c_data);
        Py_DECREF(a_array);
        if (c_array) Py_DECREF(c_array);
        if (u_array) Py_DECREF(u_array);
        PyErr_NoMemory();
        return NULL;
    }
    f64* wr = (f64*)PyArray_DATA(wr_array);
    f64* wi = (f64*)PyArray_DATA(wi_array);
    if (n > 0) {
        memset(wr, 0, n * sizeof(f64));
        memset(wi, 0, n * sizeof(f64));
    }

    i32* iwork = (i32*)calloc((nn > 0) ? nn : 1, sizeof(i32));

    i32 minwrk;
    if (wantx) {
        minwrk = nofact ? ((nn > 3 * n) ? nn : 3 * n) : nn;
    } else {
        minwrk = nofact ? ((2 * nn > 3 * n) ? 2 * nn : 3 * n) : 2 * nn;
    }
    if (n == 0) minwrk = 1;
    minwrk = (minwrk > 1) ? minwrk : 1;

    f64* dwork = (f64*)calloc(minwrk, sizeof(f64));

    if (iwork == NULL || dwork == NULL) {
        free(iwork);
        free(dwork);
        Py_XDECREF(u_out_array);
        Py_DECREF(wr_array);
        Py_DECREF(wi_array);
        if (wantsp) free(c_data);
        Py_DECREF(a_array);
        if (c_array) Py_DECREF(c_array);
        if (u_array) Py_DECREF(u_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 scale = 0.0, sep = 0.0, ferr = 0.0;
    i32 info;

    sb03rd(job, fact, trana, n, a_data, lda, u_data, ldu, c_data, ldc,
           &scale, &sep, &ferr, wr, wi, iwork, dwork, minwrk, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    if (c_array) PyArray_ResolveWritebackIfCopy(c_array);
    if (u_array) PyArray_ResolveWritebackIfCopy(u_array);

    if (u_out_array == NULL) {
        u_out_array = u_array;
        Py_INCREF(u_out_array);
    }

    PyObject* x_out;
    if (wantsp) {
        free(c_data);
        npy_intp x_dims[2] = {n, n};
        x_out = PyArray_ZEROS(2, x_dims, NPY_DOUBLE, 1);
    } else {
        x_out = (PyObject*)c_array;
        Py_INCREF(x_out);
    }

    PyObject* result = Py_BuildValue("(OOOdddOOi)",
                                     a_array, u_out_array, x_out,
                                     scale, sep, ferr,
                                     wr_array, wi_array, info);

    if (!wantsp) Py_DECREF(c_array);
    if (wantsp) Py_DECREF(x_out);
    Py_DECREF(a_array);
    Py_DECREF(u_out_array);
    if (u_array) Py_DECREF(u_array);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);

    return result;
}


PyObject* py_sb03sd(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;

    static char* kwlist[] = {"job", "fact", "trana", "uplo", "lyapun",
                             "n", "scale", "a", "t", "u", "c", "x", NULL};

    const char* job;
    const char* fact;
    const char* trana;
    const char* uplo;
    const char* lyapun;
    int n;
    double scale;
    PyObject* a_obj;
    PyObject* t_obj;
    PyObject* u_obj;
    PyObject* c_obj;
    PyObject* x_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssssidOOOOO", kwlist,
                                     &job, &fact, &trana, &uplo, &lyapun,
                                     &n, &scale, &a_obj, &t_obj, &u_obj,
                                     &c_obj, &x_obj)) {
        return NULL;
    }

    // Validate mode parameters
    char job_c = toupper(job[0]);
    char fact_c = toupper(fact[0]);
    char trana_c = toupper(trana[0]);
    char uplo_c = toupper(uplo[0]);
    char lyapun_c = toupper(lyapun[0]);

    bool jobc = (job_c == 'C');
    bool jobe = (job_c == 'E');
    bool jobb = (job_c == 'B');
    bool nofact = (fact_c == 'N');

    if (!jobc && !jobe && !jobb) {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'C', 'E', or 'B'");
        return NULL;
    }
    if (fact_c != 'F' && fact_c != 'N') {
        PyErr_SetString(PyExc_ValueError, "FACT must be 'F' or 'N'");
        return NULL;
    }
    if (trana_c != 'N' && trana_c != 'T' && trana_c != 'C') {
        PyErr_SetString(PyExc_ValueError, "TRANA must be 'N', 'T', or 'C'");
        return NULL;
    }
    if (uplo_c != 'U' && uplo_c != 'L') {
        PyErr_SetString(PyExc_ValueError, "UPLO must be 'U' or 'L'");
        return NULL;
    }
    if (lyapun_c != 'O' && lyapun_c != 'R') {
        PyErr_SetString(PyExc_ValueError, "LYAPUN must be 'O' or 'R'");
        return NULL;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "N must be >= 0");
        return NULL;
    }
    if (scale < 0.0 || scale > 1.0) {
        PyErr_SetString(PyExc_ValueError, "SCALE must be in [0, 1]");
        return NULL;
    }

    // Convert arrays
    i32 lda = (n > 1) ? n : 1;
    i32 ldt = lda;
    i32 ldu = lda;
    i32 ldc = lda;
    i32 ldx = lda;

    PyArrayObject* a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    PyArrayObject* t_array = (PyArrayObject*)PyArray_FROM_OTF(
        t_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject* u_array = (PyArrayObject*)PyArray_FROM_OTF(
        u_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject* c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    PyArrayObject* x_array = (PyArrayObject*)PyArray_FROM_OTF(
        x_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_INOUT_ARRAY2);

    if (!a_array || !t_array || !u_array || !c_array || !x_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(t_array);
        Py_XDECREF(u_array);
        Py_XDECREF(c_array);
        Py_XDECREF(x_array);
        return NULL;
    }

    // Get data pointers
    const f64* a_data = (const f64*)PyArray_DATA(a_array);
    f64* t_data = (f64*)PyArray_DATA(t_array);
    f64* u_data = (f64*)PyArray_DATA(u_array);
    const f64* c_data = (const f64*)PyArray_DATA(c_array);
    f64* x_data = (f64*)PyArray_DATA(x_array);

    // Allocate workspace
    i32 nn = n * n;
    i32 ldw = (3 > 2 * nn) ? 3 : 2 * nn;
    ldw += nn;
    i32 ldwork;
    if (jobc) {
        if (nofact) {
            ldwork = (ldw > 5 * n) ? ldw : 5 * n;
        } else {
            ldwork = ldw;
        }
    } else {
        ldwork = ldw + 2 * n;
    }
    ldwork = (ldwork > 1) ? ldwork : 1;

    i32* iwork = (i32*)calloc(nn > 1 ? nn : 1, sizeof(i32));
    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        Py_DECREF(c_array);
        Py_DECREF(x_array);
        PyErr_NoMemory();
        return NULL;
    }

    // Call the routine
    f64 sepd = 0.0, rcond = 0.0, ferr = 0.0;
    i32 info = 0;

    sb03sd(job, fact, trana, uplo, lyapun, n, scale,
           a_data, lda, t_data, ldt, u_data, ldu,
           c_data, ldc, x_data, ldx, &sepd, &rcond, &ferr,
           iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    // Resolve writebacks for INOUT arrays
    PyArray_ResolveWritebackIfCopy(t_array);
    PyArray_ResolveWritebackIfCopy(u_array);
    PyArray_ResolveWritebackIfCopy(x_array);

    if (info < 0) {
        Py_DECREF(a_array);
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        Py_DECREF(c_array);
        Py_DECREF(x_array);
        PyErr_Format(PyExc_RuntimeError, "SB03SD: illegal argument %d", -info);
        return NULL;
    }

    // Build result
    PyObject* result = Py_BuildValue("dddi", sepd, rcond, ferr, info);

    Py_DECREF(a_array);
    Py_DECREF(t_array);
    Py_DECREF(u_array);
    Py_DECREF(c_array);
    Py_DECREF(x_array);

    return result;
}

/**
 * Python wrapper for SB04OD - Solve generalized Sylvester equations with separation estimation.
 */
PyObject* py_sb04od(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;

    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *e_obj, *f_obj;
    const char *reduce, *trans, *jobd;

    static char* kwlist[] = {"reduce", "trans", "jobd", "a", "b", "c", "d", "e", "f", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssOOOOOO", kwlist,
                                     &reduce, &trans, &jobd,
                                     &a_obj, &b_obj, &c_obj,
                                     &d_obj, &e_obj, &f_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *f_array = (PyArrayObject*)PyArray_FROM_OTF(f_obj, NPY_DOUBLE,
        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!a_array || !b_array || !c_array || !d_array || !e_array || !f_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        Py_XDECREF(e_array);
        Py_XDECREF(f_array);
        PyErr_SetString(PyExc_TypeError, "Failed to convert input arrays");
        return NULL;
    }

    i32 m = (i32)PyArray_DIM(a_array, 0);
    i32 n = (i32)PyArray_DIM(b_array, 0);
    i32 lda = (m > 0) ? m : 1;
    i32 ldb = (n > 0) ? n : 1;
    i32 ldc = (m > 0) ? m : 1;
    i32 ldd = (m > 0) ? m : 1;
    i32 lde = (n > 0) ? n : 1;
    i32 ldf = (m > 0) ? m : 1;

    f64* a_data = (f64*)PyArray_DATA(a_array);
    f64* b_data = (f64*)PyArray_DATA(b_array);
    f64* c_data = (f64*)PyArray_DATA(c_array);
    f64* d_data = (f64*)PyArray_DATA(d_array);
    f64* e_data = (f64*)PyArray_DATA(e_array);
    f64* f_data = (f64*)PyArray_DATA(f_array);

    bool lredur = (*reduce == 'R' || *reduce == 'r');
    bool lredua = (*reduce == 'A' || *reduce == 'a');
    bool lredub = (*reduce == 'B' || *reduce == 'b');
    bool lredra = lredur || lredua;
    bool lredrb = lredur || lredub;
    bool ltrann = (*trans == 'N' || *trans == 'n');
    bool ljobd = (*jobd == 'D' || *jobd == 'd');
    bool ljobf = (*jobd == 'F' || *jobd == 'f');

    i32 ldp = lredra ? (m > 0 ? m : 1) : 1;
    i32 ldq = lredra ? (m > 0 ? m : 1) : 1;
    i32 ldu = lredrb ? (n > 0 ? n : 1) : 1;
    i32 ldv = lredrb ? (n > 0 ? n : 1) : 1;

    i32 p_size = lredra ? (m > 0 ? m * m : 1) : 1;
    i32 q_size = lredra ? (m > 0 ? m * m : 1) : 1;
    i32 u_size = lredrb ? (n > 0 ? n * n : 1) : 1;
    i32 v_size = lredrb ? (n > 0 ? n * n : 1) : 1;

    PyObject *p_array = NULL, *q_array = NULL, *u_array_out = NULL, *v_array_out = NULL;
    f64 *p_data = NULL, *q_data = NULL, *u_data = NULL, *v_data = NULL;

    if (lredra && m > 0) {
        npy_intp dims[2] = {m, m};
        npy_intp strides[2] = {sizeof(f64), m * sizeof(f64)};
        p_array = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        q_array = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!p_array || !q_array) {
            Py_XDECREF(p_array);
            Py_XDECREF(q_array);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            Py_DECREF(d_array);
            Py_DECREF(e_array);
            Py_DECREF(f_array);
            return NULL;
        }
        p_data = (f64*)PyArray_DATA((PyArrayObject*)p_array);
        q_data = (f64*)PyArray_DATA((PyArrayObject*)q_array);
        if (m * m > 0) {
            memset(p_data, 0, m * m * sizeof(f64));
            memset(q_data, 0, m * m * sizeof(f64));
        }
    } else {
        p_data = (f64*)calloc(p_size, sizeof(f64));
        q_data = (f64*)calloc(q_size, sizeof(f64));
    }

    if (lredrb && n > 0) {
        npy_intp dims[2] = {n, n};
        npy_intp strides[2] = {sizeof(f64), n * sizeof(f64)};
        u_array_out = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        v_array_out = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!u_array_out || !v_array_out) {
            Py_XDECREF(u_array_out);
            Py_XDECREF(v_array_out);
            Py_XDECREF(p_array);
            Py_XDECREF(q_array);
            if (!p_array) free(p_data);
            if (!q_array) free(q_data);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            Py_DECREF(d_array);
            Py_DECREF(e_array);
            Py_DECREF(f_array);
            return NULL;
        }
        u_data = (f64*)PyArray_DATA((PyArrayObject*)u_array_out);
        v_data = (f64*)PyArray_DATA((PyArrayObject*)v_array_out);
        if (n * n > 0) {
            memset(u_data, 0, n * n * sizeof(f64));
            memset(v_data, 0, n * n * sizeof(f64));
        }
    } else {
        u_data = (f64*)calloc(u_size, sizeof(f64));
        v_data = (f64*)calloc(v_size, sizeof(f64));
    }

    i32 mn = (m > n) ? m : n;
    i32 minwrk;
    if (lredur) {
        i32 t1 = 11 * mn;
        i32 t2 = 10 * mn + 23;
        minwrk = (t1 > t2) ? t1 : t2;
        if (minwrk < 1) minwrk = 1;
    } else if (lredua) {
        i32 t1 = 11 * m;
        i32 t2 = 10 * m + 23;
        minwrk = (t1 > t2) ? t1 : t2;
        if (minwrk < 1) minwrk = 1;
    } else if (lredub) {
        i32 t1 = 11 * n;
        i32 t2 = 10 * n + 23;
        minwrk = (t1 > t2) ? t1 : t2;
        if (minwrk < 1) minwrk = 1;
    } else {
        minwrk = 1;
    }
    if (ltrann && (ljobd || ljobf)) {
        i32 t = 2 * m * n;
        if (minwrk < t) minwrk = t;
    }
    i32 ldwork = minwrk > (m * n) ? minwrk : (m * n);
    if (ldwork < 1) ldwork = 1;

    i32 liwork = m + n + 6;
    i32* iwork = (i32*)calloc(liwork, sizeof(i32));
    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (!p_data || !q_data || !u_data || !v_data || !iwork || !dwork) {
        if (!p_array) free(p_data);
        if (!q_array) free(q_data);
        if (!u_array_out) free(u_data);
        if (!v_array_out) free(v_data);
        Py_XDECREF(p_array);
        Py_XDECREF(q_array);
        Py_XDECREF(u_array_out);
        Py_XDECREF(v_array_out);
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(e_array);
        Py_DECREF(f_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 scale = 0.0, dif = 0.0;
    i32 info = 0;

    sb04od(reduce, trans, jobd, m, n,
           a_data, lda, b_data, ldb, c_data, ldc,
           d_data, ldd, e_data, lde, f_data, ldf,
           &scale, &dif,
           p_data, ldp, q_data, ldq,
           u_data, ldu, v_data, ldv,
           iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(f_array);

    if (info < 0) {
        if (!p_array) free(p_data);
        if (!q_array) free(q_data);
        if (!u_array_out) free(u_data);
        if (!v_array_out) free(v_data);
        Py_XDECREF(p_array);
        Py_XDECREF(q_array);
        Py_XDECREF(u_array_out);
        Py_XDECREF(v_array_out);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(e_array);
        Py_DECREF(f_array);
        PyErr_Format(PyExc_RuntimeError, "SB04OD: illegal argument %d", -info);
        return NULL;
    }

    if (!p_array) {
        free(p_data);
        free(q_data);
        Py_INCREF(Py_None);
        p_array = Py_None;
        Py_INCREF(Py_None);
        q_array = Py_None;
    }

    if (!u_array_out) {
        free(u_data);
        free(v_data);
        Py_INCREF(Py_None);
        u_array_out = Py_None;
        Py_INCREF(Py_None);
        v_array_out = Py_None;
    }

    PyObject* result = Py_BuildValue("OOOOOOddOOOOi",
        a_array, b_array, c_array, d_array, e_array, f_array,
        scale, dif,
        p_array, q_array, u_array_out, v_array_out,
        info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(e_array);
    Py_DECREF(f_array);
    Py_XDECREF(p_array);
    Py_XDECREF(q_array);
    Py_XDECREF(u_array_out);
    Py_XDECREF(v_array_out);

    return result;
}

PyObject* py_sb04ow(PyObject* self, PyObject* args) {
    i32 m, n;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *e_obj, *f_obj;

    if (!PyArg_ParseTuple(args, "iiOOOOOO", &m, &n, &a_obj, &b_obj, &c_obj,
                          &d_obj, &e_obj, &f_obj)) {
        return NULL;
    }

    if (m <= 0) {
        PyErr_SetString(PyExc_ValueError, "m must be > 0");
        return NULL;
    }
    if (n <= 0) {
        PyErr_SetString(PyExc_ValueError, "n must be > 0");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }

    PyArrayObject *f_array = (PyArrayObject*)PyArray_FROM_OTF(f_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (f_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(e_array);
        return NULL;
    }

    i32 lda = m > 1 ? m : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldc = m > 1 ? m : 1;
    i32 ldd = m > 1 ? m : 1;
    i32 lde = n > 1 ? n : 1;
    i32 ldf = m > 1 ? m : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *f_data = (f64*)PyArray_DATA(f_array);

    i32 iwork_len = m + n + 2;
    i32 *iwork = (i32*)PyMem_Calloc(iwork_len, sizeof(i32));
    if (iwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(e_array);
        Py_DECREF(f_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 scale;
    i32 info;

    sb04ow(m, n, a_data, lda, b_data, ldb, c_data, ldc,
           d_data, ldd, e_data, lde, f_data, ldf, &scale, iwork, &info);

    PyMem_Free(iwork);

    PyArrayObject *c_out = c_array;
    PyArrayObject *f_out = f_array;

    PyArray_ResolveWritebackIfCopy(c_out);
    PyArray_ResolveWritebackIfCopy(f_out);

    PyObject *result = Py_BuildValue("OOdi",
                                     (PyObject*)c_out,
                                     (PyObject*)f_out,
                                     scale, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(e_array);
    Py_DECREF(f_array);

    return result;
}

PyObject* py_sb04pd(PyObject* self, PyObject* args) {
    (void)self;

    const char *dico_str, *facta_str, *factb_str, *trana_str, *tranb_str;
    int isgn;
    PyObject *a_obj, *b_obj, *c_obj;

    if (!PyArg_ParseTuple(args, "sssssiOOO",
                          &dico_str, &facta_str, &factb_str,
                          &trana_str, &tranb_str, &isgn,
                          &a_obj, &b_obj, &c_obj)) {
        return NULL;
    }

    char dico = dico_str[0];
    char facta = facta_str[0];
    char factb = factb_str[0];
    char trana = trana_str[0];
    char tranb = tranb_str[0];

    bool schura = (facta == 'S' || facta == 's');
    bool schurb = (factb == 'S' || factb == 's');

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    i32 m = (i32)PyArray_DIM(a_array, 0);
    i32 n = (i32)PyArray_DIM(b_array, 0);

    i32 lda = m > 1 ? m : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldc = m > 1 ? m : 1;
    i32 ldu = schura ? 1 : (m > 1 ? m : 1);
    i32 ldv = schurb ? 1 : (n > 1 ? n : 1);

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    f64 *u_data = NULL;
    f64 *v_data = NULL;
    PyObject *u_array_obj = Py_None;
    PyObject *v_array_obj = Py_None;

    if (!schura) {
        npy_intp u_dims[2] = {m, m};
        npy_intp u_strides[2] = {sizeof(f64), m * sizeof(f64)};
        u_array_obj = PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE,
                                  u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (u_array_obj == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            return NULL;
        }
        u_data = (f64*)PyArray_DATA((PyArrayObject*)u_array_obj);
        if (m * m > 0) memset(u_data, 0, m * m * sizeof(f64));
    } else {
        Py_INCREF(Py_None);
    }

    if (!schurb) {
        npy_intp v_dims[2] = {n, n};
        npy_intp v_strides[2] = {sizeof(f64), n * sizeof(f64)};
        v_array_obj = PyArray_New(&PyArray_Type, 2, v_dims, NPY_DOUBLE,
                                  v_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (v_array_obj == NULL) {
            if (u_array_obj != Py_None) Py_DECREF(u_array_obj);
            else Py_DECREF(Py_None);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            return NULL;
        }
        v_data = (f64*)PyArray_DATA((PyArrayObject*)v_array_obj);
        if (n * n > 0) memset(v_data, 0, n * n * sizeof(f64));
    } else {
        Py_INCREF(Py_None);
    }

    i32 max1 = 3*m > 5*n ? 3*m : 5*n;
    i32 max2 = 2*n + 2*m > max1 ? 2*n + 2*m : max1;
    i32 ldwork = 1 + 2*m + max2;
    if (ldwork < 1) ldwork = 1;

    f64 *dwork = (f64*)PyMem_Calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        if (u_array_obj != Py_None) Py_DECREF(u_array_obj);
        else Py_DECREF(Py_None);
        if (v_array_obj != Py_None) Py_DECREF(v_array_obj);
        else Py_DECREF(Py_None);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 scale;
    i32 info;

    sb04pd(dico, facta, factb, trana, tranb, (i32)isgn, m, n,
           a_data, lda, u_data, ldu, b_data, ldb, v_data, ldv,
           c_data, ldc, &scale, dwork, ldwork, &info);

    PyMem_Free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("OOOOOdi",
                                     (PyObject*)c_array,
                                     (PyObject*)a_array,
                                     u_array_obj,
                                     (PyObject*)b_array,
                                     v_array_obj,
                                     scale, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    if (u_array_obj != Py_None) Py_DECREF(u_array_obj);
    else Py_DECREF(Py_None);
    if (v_array_obj != Py_None) Py_DECREF(v_array_obj);
    else Py_DECREF(Py_None);

    return result;
}


/* Python wrapper for sb04rv */
PyObject* py_sb04rv(PyObject* self, PyObject* args) {
    const char *abschr_str, *ul_str;
    i32 indx;
    PyObject *c_obj, *ab_obj, *ba_obj;

    if (!PyArg_ParseTuple(args, "ssiOOO", &abschr_str, &ul_str, &indx, &c_obj, &ab_obj, &ba_obj)) {
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (c_array == NULL) {
        return NULL;
    }

    PyArrayObject *ab_array = (PyArrayObject*)PyArray_FROM_OTF(ab_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (ab_array == NULL) {
        Py_DECREF(c_array);
        return NULL;
    }

    PyArrayObject *ba_array = (PyArrayObject*)PyArray_FROM_OTF(ba_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (ba_array == NULL) {
        Py_DECREF(c_array);
        Py_DECREF(ab_array);
        return NULL;
    }

    npy_intp *c_dims = PyArray_DIMS(c_array);
    i32 n = (i32)c_dims[0];
    i32 m = (i32)c_dims[1];
    i32 ldc = (n > 1) ? n : 1;

    npy_intp *ab_dims = PyArray_DIMS(ab_array);
    i32 ldab = (ab_dims[0] > 1) ? (i32)ab_dims[0] : 1;

    npy_intp *ba_dims = PyArray_DIMS(ba_array);
    i32 ldba = (ba_dims[0] > 1) ? (i32)ba_dims[0] : 1;

    bool is_b = (abschr_str[0] == 'B' || abschr_str[0] == 'b');
    i32 d_len = is_b ? 2 * n : 2 * m;
    i32 dwork_len = d_len;
    if (d_len == 0) d_len = 1;
    if (dwork_len == 0) dwork_len = 1;

    npy_intp d_dims[1] = {d_len};
    PyObject *d_array = PyArray_SimpleNew(1, d_dims, NPY_DOUBLE);
    if (d_array == NULL) {
        Py_DECREF(c_array);
        Py_DECREF(ab_array);
        Py_DECREF(ba_array);
        return NULL;
    }
    f64 *d_data = (f64*)PyArray_DATA((PyArrayObject*)d_array);

    f64 *dwork = (f64*)malloc(dwork_len * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(d_array);
        Py_DECREF(c_array);
        Py_DECREF(ab_array);
        Py_DECREF(ba_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *ab_data = (f64*)PyArray_DATA(ab_array);
    f64 *ba_data = (f64*)PyArray_DATA(ba_array);

    sb04rv(abschr_str, ul_str, n, m, c_data, ldc, indx, ab_data, ldab, ba_data, ldba, d_data, dwork);

    free(dwork);

    Py_DECREF(c_array);
    Py_DECREF(ab_array);
    Py_DECREF(ba_array);
    return d_array;
}

PyObject* py_sb04rw(PyObject* self, PyObject* args) {
    const char *abschr_str, *ul_str;
    i32 indx;
    PyObject *c_obj, *ab_obj, *ba_obj;

    if (!PyArg_ParseTuple(args, "ssiOOO", &abschr_str, &ul_str, &indx, &c_obj, &ab_obj, &ba_obj)) {
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (c_array == NULL) {
        return NULL;
    }

    PyArrayObject *ab_array = (PyArrayObject*)PyArray_FROM_OTF(ab_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (ab_array == NULL) {
        Py_DECREF(c_array);
        return NULL;
    }

    PyArrayObject *ba_array = (PyArrayObject*)PyArray_FROM_OTF(ba_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (ba_array == NULL) {
        Py_DECREF(c_array);
        Py_DECREF(ab_array);
        return NULL;
    }

    npy_intp *c_dims = PyArray_DIMS(c_array);
    i32 n = (i32)c_dims[0];
    i32 m = (i32)c_dims[1];
    i32 ldc = (n > 1) ? n : 1;

    npy_intp *ab_dims = PyArray_DIMS(ab_array);
    i32 ldab = (ab_dims[0] > 1) ? (i32)ab_dims[0] : 1;

    npy_intp *ba_dims = PyArray_DIMS(ba_array);
    i32 ldba = (ba_dims[0] > 1) ? (i32)ba_dims[0] : 1;

    bool is_b = (abschr_str[0] == 'B' || abschr_str[0] == 'b');
    i32 d_len = is_b ? n : m;
    i32 dwork_len = d_len;
    if (d_len == 0) d_len = 1;
    if (dwork_len == 0) dwork_len = 1;

    npy_intp d_dims[1] = {d_len};
    PyObject *d_array = PyArray_SimpleNew(1, d_dims, NPY_DOUBLE);
    if (d_array == NULL) {
        Py_DECREF(c_array);
        Py_DECREF(ab_array);
        Py_DECREF(ba_array);
        return NULL;
    }
    f64 *d_data = (f64*)PyArray_DATA((PyArrayObject*)d_array);

    f64 *dwork = (f64*)malloc(dwork_len * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(d_array);
        Py_DECREF(c_array);
        Py_DECREF(ab_array);
        Py_DECREF(ba_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *ab_data = (f64*)PyArray_DATA(ab_array);
    f64 *ba_data = (f64*)PyArray_DATA(ba_array);

    sb04rw(abschr_str, ul_str, n, m, c_data, ldc, indx, ab_data, ldab, ba_data, ldba, d_data, dwork);

    free(dwork);

    Py_DECREF(c_array);
    Py_DECREF(ab_array);
    Py_DECREF(ba_array);
    return d_array;
}

PyObject* py_sb04rx(PyObject* self, PyObject* args) {
    const char *rc_str, *ul_str;
    f64 lambd1, lambd2, lambd3, lambd4, tol;
    PyObject *a_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "ssOddddOd", &rc_str, &ul_str, &a_obj,
                          &lambd1, &lambd2, &lambd3, &lambd4, &d_obj, &tol)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 m = (i32)a_dims[0];
    i32 lda = (m > 1) ? m : 1;
    i32 m2 = 2 * m;
    i32 lddwor = (m2 > 1) ? m2 : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);

    i32 iwork_size = (m2 > 0) ? m2 : 1;
    i32 dwork_size = lddwor * (m2 + 3);
    if (dwork_size < 1) dwork_size = 1;

    i32 *iwork = (i32*)malloc(iwork_size * sizeof(i32));
    if (iwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(d_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *dwork = (f64*)malloc(dwork_size * sizeof(f64));
    if (dwork == NULL) {
        free(iwork);
        Py_DECREF(a_array);
        Py_DECREF(d_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info = 0;
    sb04rx(rc_str, ul_str, m, a_data, lda, lambd1, lambd2, lambd3, lambd4,
           d_data, tol, iwork, dwork, lddwor, &info);

    free(iwork);
    free(dwork);

    npy_intp out_dims[1] = {m2};
    PyObject *d_out_array = PyArray_SimpleNew(1, out_dims, NPY_DOUBLE);
    if (d_out_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(d_array);
        return NULL;
    }
    f64 *d_out = (f64*)PyArray_DATA((PyArrayObject*)d_out_array);
    if (m2 > 0) memcpy(d_out, d_data, m2 * sizeof(f64));

    PyArray_ResolveWritebackIfCopy(d_array);
    Py_DECREF(a_array);
    Py_DECREF(d_array);

    PyObject *result = Py_BuildValue("Oi", d_out_array, info);
    Py_DECREF(d_out_array);
    return result;
}

PyObject* py_sb04ry(PyObject* self, PyObject* args) {
    const char *rc_str, *ul_str;
    f64 lambda, tol;
    PyObject *a_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "ssOdOd", &rc_str, &ul_str, &a_obj,
                          &lambda, &d_obj, &tol)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 m = (i32)a_dims[0];
    i32 lda = (m > 1) ? m : 1;
    i32 lddwor = (m > 1) ? m : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);

    i32 iwork_size = (m > 0) ? m : 1;
    i32 dwork_size = lddwor * (m + 3);
    if (dwork_size < 1) dwork_size = 1;

    i32 *iwork = (i32*)malloc(iwork_size * sizeof(i32));
    if (iwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(d_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *dwork = (f64*)malloc(dwork_size * sizeof(f64));
    if (dwork == NULL) {
        free(iwork);
        Py_DECREF(a_array);
        Py_DECREF(d_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info = 0;
    sb04ry(rc_str, ul_str, m, a_data, lda, lambda, d_data, tol,
           iwork, dwork, lddwor, &info);

    free(iwork);
    free(dwork);

    npy_intp out_dims[1] = {m};
    PyObject *d_out_array = PyArray_SimpleNew(1, out_dims, NPY_DOUBLE);
    if (d_out_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(d_array);
        return NULL;
    }
    f64 *d_out = (f64*)PyArray_DATA((PyArrayObject*)d_out_array);
    if (m > 0) {
        memcpy(d_out, d_data, m * sizeof(f64));
    }

    PyArray_ResolveWritebackIfCopy(d_array);
    Py_DECREF(a_array);
    Py_DECREF(d_array);

    PyObject *result = Py_BuildValue("Oi", d_out_array, info);
    Py_DECREF(d_out_array);
    return result;
}

/* Python wrapper for sb02qd */
PyObject* py_sb02qd(PyObject* self, PyObject* args) {
    (void)self;
    char *job, *fact, *trana, *uplo, *lyapun;
    PyObject *a_obj, *t_obj, *u_obj, *g_obj, *q_obj, *x_obj;
    PyArrayObject *a_array, *t_array, *u_array, *g_array, *q_array, *x_array;
    f64 sep = 0.0, rcond = 0.0, ferr = 0.0;
    i32 info;

    if (!PyArg_ParseTuple(args, "sssssOOOOOO", &job, &fact, &trana, &uplo, &lyapun,
                          &a_obj, &t_obj, &u_obj, &g_obj, &q_obj, &x_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) return NULL;

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 n = (i32)a_dims[0];

    t_array = (PyArrayObject*)PyArray_FROM_OTF(t_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (t_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (u_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(t_array);
        return NULL;
    }

    g_array = (PyArrayObject*)PyArray_FROM_OTF(g_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (g_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        return NULL;
    }

    q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (q_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        Py_DECREF(g_array);
        return NULL;
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (x_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        Py_DECREF(g_array);
        Py_DECREF(q_array);
        return NULL;
    }

    i32 lda = (n > 1) ? n : 1;
    i32 ldt = lda;
    i32 ldu = lda;
    i32 ldg = lda;
    i32 ldq = lda;
    i32 ldx = lda;
    i32 nn = n * n;

    int update = (lyapun[0] == 'O' || lyapun[0] == 'o');
    int nofact = (fact[0] == 'N' || fact[0] == 'n');
    int jobc = (job[0] == 'C' || job[0] == 'c');
    int needac = update && !jobc;

    i32 lwa = needac ? nn : 0;
    i32 ldw;
    if (nofact) {
        if (jobc) {
            ldw = (5 * n > 2 * nn) ? 5 * n : 2 * nn;
        } else {
            i32 opt1 = lwa + 5 * n;
            i32 opt2 = 4 * nn;
            ldw = (opt1 > opt2) ? opt1 : opt2;
        }
    } else {
        if (jobc) {
            ldw = 2 * nn;
        } else {
            ldw = 4 * nn;
        }
    }
    i32 ldwork = (ldw > 1) ? ldw : 1;
    i32 iwork_size = (nn > 0) ? nn : 1;

    const f64* a = (const f64*)PyArray_DATA(a_array);
    f64* t = (f64*)PyArray_DATA(t_array);
    f64* u = (f64*)PyArray_DATA(u_array);
    const f64* g = (const f64*)PyArray_DATA(g_array);
    const f64* q = (const f64*)PyArray_DATA(q_array);
    const f64* x = (const f64*)PyArray_DATA(x_array);

    i32* iwork = (i32*)malloc(iwork_size * sizeof(i32));
    f64* dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (iwork == NULL || dwork == NULL) {
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        Py_DECREF(g_array);
        Py_DECREF(q_array);
        Py_DECREF(x_array);
        return PyErr_NoMemory();
    }

    sb02qd(job, fact, trana, uplo, lyapun, n, a, lda, t, ldt, u, ldu, g, ldg, q, ldq,
           x, ldx, &sep, &rcond, &ferr, iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    if (info < 0) {
        PyArray_ResolveWritebackIfCopy(t_array);
        PyArray_ResolveWritebackIfCopy(u_array);
        Py_DECREF(a_array);
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        Py_DECREF(g_array);
        Py_DECREF(q_array);
        Py_DECREF(x_array);
        PyErr_Format(PyExc_ValueError, "sb02qd: illegal value for argument %d", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(t_array);
    PyArray_ResolveWritebackIfCopy(u_array);

    PyObject* result = Py_BuildValue("dddOOi", sep, rcond, ferr, t_array, u_array, info);

    Py_DECREF(a_array);
    Py_DECREF(t_array);
    Py_DECREF(u_array);
    Py_DECREF(g_array);
    Py_DECREF(q_array);
    Py_DECREF(x_array);

    return result;
}


/*
 * SB03QD - Estimate conditioning and forward error bound for continuous-time Lyapunov.
 */
PyObject* py_sb03qd(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"job", "fact", "trana", "uplo", "lyapun",
                             "scale", "a", "t", "u", "c", "x", NULL};

    const char* job;
    const char* fact;
    const char* trana;
    const char* uplo;
    const char* lyapun;
    f64 scale;
    PyObject* a_obj;
    PyObject* t_obj;
    PyObject* u_obj;
    PyObject* c_obj;
    PyObject* x_obj;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssssdOOOOO", kwlist,
                                     &job, &fact, &trana, &uplo, &lyapun,
                                     &scale, &a_obj, &t_obj, &u_obj,
                                     &c_obj, &x_obj)) {
        return NULL;
    }

    PyArrayObject* a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (a_array == NULL) return NULL;

    PyArrayObject* t_array = (PyArrayObject*)PyArray_FROM_OTF(
        t_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (t_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject* u_array = (PyArrayObject*)PyArray_FROM_OTF(
        u_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (u_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(t_array);
        return NULL;
    }

    PyArrayObject* c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        return NULL;
    }

    PyArrayObject* x_array = (PyArrayObject*)PyArray_FROM_OTF(
        x_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (x_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        Py_DECREF(c_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(x_array, 0);

    char lyapun_c = (char)toupper((unsigned char)lyapun[0]);
    char fact_c = (char)toupper((unsigned char)fact[0]);
    bool update = (lyapun_c == 'O');
    bool nofact = (fact_c == 'N');

    i32 lda = (n > 1 && (update || nofact)) ? n : 1;
    i32 ldt = (n > 1) ? n : 1;
    i32 ldu = (n > 1 && update) ? n : 1;
    i32 ldc = (n > 1) ? n : 1;
    i32 ldx = (n > 1) ? n : 1;

    i32 nn = n * n;
    i32 ldwork;
    char job_c = (char)toupper((unsigned char)job[0]);
    if (job_c == 'C') {
        ldwork = 2 * nn;
    } else {
        ldwork = 3 * nn;
        if (lyapun_c == 'R') {
            ldwork = ldwork + n - 1;
        }
    }
    if (nofact) {
        i32 alt = 5 * n;
        if (alt > ldwork) ldwork = alt;
    }
    if (ldwork < 1) ldwork = 1;

    f64* a = (f64*)PyArray_DATA(a_array);
    f64* t = (f64*)PyArray_DATA(t_array);
    f64* u = (f64*)PyArray_DATA(u_array);
    f64* c = (f64*)PyArray_DATA(c_array);
    f64* x = (f64*)PyArray_DATA(x_array);

    i32* iwork = (i32*)calloc(nn > 1 ? nn : 1, sizeof(i32));
    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (iwork == NULL || dwork == NULL) {
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        Py_DECREF(c_array);
        Py_DECREF(x_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 sep = 0.0;
    f64 rcond = 0.0;
    f64 ferr = 0.0;
    i32 info;

    sb03qd(job, fact, trana, uplo, lyapun, n, scale, a, lda, t, ldt, u, ldu,
           c, ldc, x, ldx, &sep, &rcond, &ferr, iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    if (info < 0) {
        PyArray_ResolveWritebackIfCopy(t_array);
        PyArray_ResolveWritebackIfCopy(u_array);
        Py_DECREF(a_array);
        Py_DECREF(t_array);
        Py_DECREF(u_array);
        Py_DECREF(c_array);
        Py_DECREF(x_array);
        PyErr_Format(PyExc_ValueError, "sb03qd: illegal value for argument %d", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(t_array);
    PyArray_ResolveWritebackIfCopy(u_array);

    PyObject* result = Py_BuildValue("dddOOi", sep, rcond, ferr, t_array, u_array, info);

    Py_DECREF(a_array);
    Py_DECREF(t_array);
    Py_DECREF(u_array);
    Py_DECREF(c_array);
    Py_DECREF(x_array);

    return result;
}

PyObject* py_sb03ud(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;

    static char* kwlist[] = {"job", "fact", "trana", "uplo", "lyapun",
                             "a", "c", "t", "u", "x", "scale", NULL};

    const char* job;
    const char* fact;
    const char* trana;
    const char* uplo;
    const char* lyapun;
    PyObject* a_obj = Py_None;
    PyObject* c_obj = Py_None;
    PyObject* t_obj = Py_None;
    PyObject* u_obj = Py_None;
    PyObject* x_obj = Py_None;
    double scale_in = 1.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssss|OOOOOd", kwlist,
                                     &job, &fact, &trana, &uplo, &lyapun,
                                     &a_obj, &c_obj, &t_obj, &u_obj, &x_obj,
                                     &scale_in)) {
        return NULL;
    }

    char job_c = toupper(job[0]);
    char fact_c = toupper(fact[0]);
    char trana_c = toupper(trana[0]);
    char uplo_c = toupper(uplo[0]);
    char lyapun_c = toupper(lyapun[0]);

    bool jobx = (job_c == 'X');
    bool jobs = (job_c == 'S');
    bool jobc = (job_c == 'C');
    bool jobe = (job_c == 'E');
    bool joba = (job_c == 'A');
    bool nofact = (fact_c == 'N');
    bool update = (lyapun_c == 'O');

    if (!jobx && !jobs && !jobc && !jobe && !joba) {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'X', 'S', 'C', 'E', or 'A'");
        return NULL;
    }
    if (fact_c != 'F' && fact_c != 'N') {
        PyErr_SetString(PyExc_ValueError, "FACT must be 'F' or 'N'");
        return NULL;
    }
    if (trana_c != 'N' && trana_c != 'T' && trana_c != 'C') {
        PyErr_SetString(PyExc_ValueError, "TRANA must be 'N', 'T', or 'C'");
        return NULL;
    }
    if (uplo_c != 'U' && uplo_c != 'L') {
        PyErr_SetString(PyExc_ValueError, "UPLO must be 'U' or 'L'");
        return NULL;
    }
    if (lyapun_c != 'O' && lyapun_c != 'R') {
        PyErr_SetString(PyExc_ValueError, "LYAPUN must be 'O' or 'R'");
        return NULL;
    }

    i32 n = 0;
    PyArrayObject* a_array = NULL;
    PyArrayObject* c_array = NULL;
    PyArrayObject* t_array = NULL;
    PyArrayObject* u_array = NULL;
    PyArrayObject* x_array = NULL;

    if (nofact && a_obj != Py_None) {
        a_array = (PyArrayObject*)PyArray_FROM_OTF(
            a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
        if (!a_array) return NULL;
        n = (i32)PyArray_DIM(a_array, 0);
    } else if (!nofact && t_obj != Py_None) {
        t_array = (PyArrayObject*)PyArray_FROM_OTF(
            t_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_INOUT_ARRAY2);
        if (!t_array) return NULL;
        n = (i32)PyArray_DIM(t_array, 0);
    } else if (c_obj != Py_None) {
        c_array = (PyArrayObject*)PyArray_FROM_OTF(
            c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
        if (!c_array) return NULL;
        n = (i32)PyArray_DIM(c_array, 0);
    }

    if (n < 0) {
        Py_XDECREF(a_array);
        Py_XDECREF(c_array);
        Py_XDECREF(t_array);
        PyErr_SetString(PyExc_ValueError, "N must be >= 0");
        return NULL;
    }

    i32 lda = (n > 1) ? n : 1;
    i32 ldt = lda;
    i32 ldu = lda;
    i32 ldc = lda;
    i32 ldx = lda;

    if (nofact && !a_array && a_obj != Py_None) {
        a_array = (PyArrayObject*)PyArray_FROM_OTF(
            a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
        if (!a_array) {
            Py_XDECREF(c_array);
            Py_XDECREF(t_array);
            return NULL;
        }
    }

    if (!jobs && !c_array && c_obj != Py_None) {
        c_array = (PyArrayObject*)PyArray_FROM_OTF(
            c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
        if (!c_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(t_array);
            return NULL;
        }
    }

    f64* t_data = NULL;
    if (!nofact) {
        if (!t_array && t_obj != Py_None) {
            t_array = (PyArrayObject*)PyArray_FROM_OTF(
                t_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_INOUT_ARRAY2);
        }
        if (t_array) {
            t_data = (f64*)PyArray_DATA(t_array);
        }
    }

    if (nofact) {
        npy_intp dims[2] = {n, n};
        npy_intp strides[2] = {sizeof(f64), n * sizeof(f64)};
        t_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                               strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!t_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(c_array);
            return NULL;
        }
        t_data = (f64*)PyArray_DATA(t_array);
        if (n * n > 0) memset(t_data, 0, n * n * sizeof(f64));
    }

    f64* u_data = NULL;
    if (update) {
        if (!nofact && u_obj != Py_None) {
            u_array = (PyArrayObject*)PyArray_FROM_OTF(
                u_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_INOUT_ARRAY2);
            if (!u_array) {
                Py_XDECREF(a_array);
                Py_XDECREF(c_array);
                Py_XDECREF(t_array);
                return NULL;
            }
            u_data = (f64*)PyArray_DATA(u_array);
        } else {
            npy_intp dims[2] = {n, n};
            npy_intp strides[2] = {sizeof(f64), n * sizeof(f64)};
            u_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                                   strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            if (!u_array) {
                Py_XDECREF(a_array);
                Py_XDECREF(c_array);
                Py_XDECREF(t_array);
                return NULL;
            }
            u_data = (f64*)PyArray_DATA(u_array);
            if (n * n > 0) memset(u_data, 0, n * n * sizeof(f64));
        }
    } else {
        npy_intp dims[2] = {1, 1};
        npy_intp strides[2] = {sizeof(f64), sizeof(f64)};
        u_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                               strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!u_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(c_array);
            Py_XDECREF(t_array);
            return NULL;
        }
        u_data = (f64*)PyArray_DATA(u_array);
        memset(u_data, 0, sizeof(f64));
        ldu = 1;
    }

    f64* x_data = NULL;
    if (!jobs) {
        if ((jobc || jobe) && x_obj != Py_None) {
            x_array = (PyArrayObject*)PyArray_FROM_OTF(
                x_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_INOUT_ARRAY2);
            if (!x_array) {
                Py_XDECREF(a_array);
                Py_XDECREF(c_array);
                Py_XDECREF(t_array);
                Py_XDECREF(u_array);
                return NULL;
            }
            x_data = (f64*)PyArray_DATA(x_array);
        } else {
            npy_intp dims[2] = {n, n};
            npy_intp strides[2] = {sizeof(f64), n * sizeof(f64)};
            x_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                                   strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            if (!x_array) {
                Py_XDECREF(a_array);
                Py_XDECREF(c_array);
                Py_XDECREF(t_array);
                Py_XDECREF(u_array);
                return NULL;
            }
            x_data = (f64*)PyArray_DATA(x_array);
            if (n * n > 0) memset(x_data, 0, n * n * sizeof(f64));
        }
    } else {
        npy_intp dims[2] = {1, 1};
        npy_intp strides[2] = {sizeof(f64), sizeof(f64)};
        x_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                               strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!x_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(c_array);
            Py_XDECREF(t_array);
            Py_XDECREF(u_array);
            return NULL;
        }
        x_data = (f64*)PyArray_DATA(x_array);
        memset(x_data, 0, sizeof(f64));
        ldx = 1;
    }

    f64* a_data = a_array ? (f64*)PyArray_DATA(a_array) : NULL;
    f64* c_data = c_array ? (f64*)PyArray_DATA(c_array) : NULL;

    i32 nn = n * n;
    i32 ldwork;
    if (jobx) {
        if (nofact) {
            ldwork = (nn > 3 * n) ? nn : 3 * n;
            ldwork = (ldwork > 1) ? ldwork : 1;
        } else {
            ldwork = (nn > 2 * n) ? nn : 2 * n;
            ldwork = (ldwork > 1) ? ldwork : 1;
        }
    } else if (jobs) {
        ldwork = (2 * nn > 3) ? 2 * nn : 3;
    } else if (jobc) {
        ldwork = ((2 * nn > 3) ? 2 * nn : 3) + nn;
    } else {
        ldwork = ((2 * nn > 3) ? 2 * nn : 3) + nn + 2 * n;
    }

    if (nofact && ldwork < 3 * n) {
        ldwork = 3 * n;
    }
    ldwork = (ldwork > 1) ? ldwork : 1;

    npy_intp wr_dim[1] = {n};
    PyArrayObject* wr_array = (PyArrayObject*)PyArray_SimpleNew(1, wr_dim, NPY_DOUBLE);
    PyArrayObject* wi_array = (PyArrayObject*)PyArray_SimpleNew(1, wr_dim, NPY_DOUBLE);
    if (!wr_array || !wi_array) {
        Py_XDECREF(wr_array);
        Py_XDECREF(wi_array);
        Py_XDECREF(a_array);
        Py_XDECREF(c_array);
        Py_XDECREF(t_array);
        Py_XDECREF(u_array);
        Py_XDECREF(x_array);
        return NULL;
    }
    f64* wr = (f64*)PyArray_DATA(wr_array);
    f64* wi = (f64*)PyArray_DATA(wi_array);
    if (n > 0) {
        memset(wr, 0, n * sizeof(f64));
        memset(wi, 0, n * sizeof(f64));
    }

    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));
    i32* iwork = (i32*)calloc((nn > 1) ? nn : 1, sizeof(i32));

    if (!dwork || !iwork) {
        free(dwork);
        free(iwork);
        Py_XDECREF(wr_array);
        Py_XDECREF(wi_array);
        Py_XDECREF(a_array);
        Py_XDECREF(c_array);
        Py_XDECREF(t_array);
        Py_XDECREF(u_array);
        Py_XDECREF(x_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 scale = scale_in;
    f64 sepd = 0.0;
    f64 rcond = 0.0;
    f64 ferr = 0.0;
    i32 info = 0;

    if (!update) ldu = 1;
    if (jobs) {
        ldc = 1;
        ldx = 1;
    }

    sb03ud(job, fact, trana, uplo, lyapun, n, &scale,
           a_data, lda, t_data, ldt, u_data, ldu,
           c_data, ldc, x_data, ldx, &sepd, &rcond, &ferr,
           wr, wi, iwork, dwork, ldwork, &info);

    free(dwork);
    free(iwork);

    if (info < 0) {
        PyArray_ResolveWritebackIfCopy(t_array);
        PyArray_ResolveWritebackIfCopy(u_array);
        PyArray_ResolveWritebackIfCopy(x_array);
        Py_XDECREF(a_array);
        Py_XDECREF(c_array);
        Py_XDECREF(t_array);
        Py_XDECREF(u_array);
        Py_XDECREF(x_array);
        Py_XDECREF(wr_array);
        Py_XDECREF(wi_array);
        PyErr_Format(PyExc_ValueError, "sb03ud: illegal value for argument %d", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(t_array);
    PyArray_ResolveWritebackIfCopy(u_array);
    PyArray_ResolveWritebackIfCopy(x_array);

    PyObject* result = Py_BuildValue("OddddOOi", x_array, scale, sepd, rcond, ferr,
                                      wr_array, wi_array, info);

    Py_XDECREF(a_array);
    Py_XDECREF(c_array);
    Py_XDECREF(t_array);
    Py_XDECREF(u_array);
    Py_XDECREF(x_array);
    Py_XDECREF(wr_array);
    Py_XDECREF(wi_array);

    return result;
}

/* sb04rd - Discrete-time Sylvester equation solver */
PyObject* py_sb04rd(PyObject* self, PyObject* args) {
    const char* abschu;
    const char* ula;
    const char* ulb;
    PyObject* a_obj;
    PyObject* b_obj;
    PyObject* c_obj;
    double tol = 0.0;

    if (!PyArg_ParseTuple(args, "sssOOO|d", &abschu, &ula, &ulb, &a_obj, &b_obj, &c_obj, &tol)) {
        return NULL;
    }

    if (*abschu != 'A' && *abschu != 'a' && *abschu != 'B' && *abschu != 'b' &&
        *abschu != 'S' && *abschu != 's') {
        PyErr_SetString(PyExc_ValueError, "abschu must be 'A', 'B', or 'S'");
        return NULL;
    }
    if (*ula != 'U' && *ula != 'u' && *ula != 'L' && *ula != 'l') {
        PyErr_SetString(PyExc_ValueError, "ula must be 'U' or 'L'");
        return NULL;
    }
    if (*ulb != 'U' && *ulb != 'u' && *ulb != 'L' && *ulb != 'l') {
        PyErr_SetString(PyExc_ValueError, "ulb must be 'U' or 'L'");
        return NULL;
    }

    PyArrayObject* a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
        NPY_ARRAY_IN_FARRAY);
    PyArrayObject* b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
        NPY_ARRAY_IN_FARRAY);
    PyArrayObject* c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!a_array || !b_array || !c_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        PyErr_NoMemory();
        return NULL;
    }

    if (PyArray_NDIM(a_array) != 2 || PyArray_NDIM(b_array) != 2 || PyArray_NDIM(c_array) != 2) {
        PyArray_DiscardWritebackIfCopy(c_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_ValueError, "a, b, and c must be 2-dimensional arrays");
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 m = (i32)PyArray_DIM(b_array, 0);
    i32 lda = (n > 0) ? n : 1;
    i32 ldb = (m > 0) ? m : 1;
    i32 ldc = (n > 0) ? n : 1;

    if (PyArray_DIM(a_array, 1) != n) {
        PyArray_DiscardWritebackIfCopy(c_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_ValueError, "a must be square");
        return NULL;
    }
    if (PyArray_DIM(b_array, 1) != m) {
        PyArray_DiscardWritebackIfCopy(c_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_ValueError, "b must be square");
        return NULL;
    }
    if (PyArray_DIM(c_array, 0) != n || PyArray_DIM(c_array, 1) != m) {
        PyArray_DiscardWritebackIfCopy(c_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_SetString(PyExc_ValueError, "c must have dimensions (n, m)");
        return NULL;
    }

    f64* a_data = (f64*)PyArray_DATA(a_array);
    f64* b_data = (f64*)PyArray_DATA(b_array);
    f64* c_data = (f64*)PyArray_DATA(c_array);

    i32 maxmn = (m > n) ? m : n;
    bool both_schur_upper = (*abschu == 'S' || *abschu == 's') &&
                            (*ula == 'U' || *ula == 'u') &&
                            (*ulb == 'U' || *ulb == 'u');
    i32 ldwork;
    i32 liwork;

    if (both_schur_upper) {
        ldwork = 2 * n;
        liwork = 1;
    } else {
        ldwork = 2 * maxmn * (4 + 2 * maxmn);
        liwork = 2 * maxmn;
    }

    if (ldwork < 1) ldwork = 1;
    if (liwork < 1) liwork = 1;

    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));
    i32* iwork = (i32*)calloc(liwork, sizeof(i32));

    if (!dwork || !iwork) {
        free(dwork);
        free(iwork);
        PyArray_DiscardWritebackIfCopy(c_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info = 0;
    sb04rd(abschu, ula, ulb, n, m, a_data, lda, b_data, ldb, c_data, ldc,
           tol, iwork, dwork, ldwork, &info);

    free(dwork);
    free(iwork);

    if (info < 0) {
        PyArray_DiscardWritebackIfCopy(c_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_Format(PyExc_ValueError, "sb04rd: illegal value for argument %d", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject* result = Py_BuildValue("Oi", c_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);

    return result;
}


/* Python wrapper for sb06nd */
PyObject* py_sb06nd(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;

    int n, m, kmax;
    PyObject *a_obj, *b_obj, *kstair_obj, *u_obj;
    PyArrayObject *a_array = NULL, *b_array = NULL, *kstair_array = NULL, *u_array = NULL;

    static char* kwlist[] = {"n", "m", "kmax", "a", "b", "kstair", "u", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiiOOOO", kwlist,
                                     &n, &m, &kmax, &a_obj, &b_obj, &kstair_obj, &u_obj)) {
        return NULL;
    }

    /* Parameter validation */
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be >= 0");
        return NULL;
    }
    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be >= 0");
        return NULL;
    }
    if (kmax < 0 || kmax > n) {
        PyErr_SetString(PyExc_ValueError, "kmax must satisfy 0 <= kmax <= n");
        return NULL;
    }

    /* Handle empty cases */
    if (n == 0 || m == 0) {
        npy_intp f_dims[2] = {m, n};
        PyObject* f_array = PyArray_ZEROS(2, f_dims, NPY_DOUBLE, 1);  /* F-order */

        npy_intp a_dims[2] = {n, n};
        npy_intp b_dims[2] = {n, m};
        npy_intp u_dims[2] = {n, n};
        PyObject* a_out = PyArray_ZEROS(2, a_dims, NPY_DOUBLE, 1);
        PyObject* b_out = PyArray_ZEROS(2, b_dims, NPY_DOUBLE, 1);
        PyObject* u_out = PyArray_ZEROS(2, u_dims, NPY_DOUBLE, 1);

        return Py_BuildValue("NNNNi", a_out, b_out, u_out, f_array, 0);
    }

    /* Convert input arrays */
    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    kstair_array = (PyArrayObject*)PyArray_FROM_OTF(kstair_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (kstair_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (u_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(kstair_array);
        return NULL;
    }

    f64* a_data = (f64*)PyArray_DATA(a_array);
    f64* b_data = (f64*)PyArray_DATA(b_array);
    i32* kstair_data = (i32*)PyArray_DATA(kstair_array);
    f64* u_data = (f64*)PyArray_DATA(u_array);

    i32 lda = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldu = n > 1 ? n : 1;
    i32 ldf = m > 1 ? m : 1;

    /* Allocate output F array early */
    npy_intp f_dims[2] = {m, n};
    npy_intp f_strides[2] = {sizeof(f64), ldf * sizeof(f64)};
    PyObject* f_array = PyArray_New(&PyArray_Type, 2, f_dims, NPY_DOUBLE, f_strides,
                                    NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!f_array) {
        PyArray_DiscardWritebackIfCopy(a_array);
        PyArray_DiscardWritebackIfCopy(b_array);
        PyArray_DiscardWritebackIfCopy(u_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(kstair_array);
        Py_DECREF(u_array);
        return NULL;
    }
    f64* f_data = (f64*)PyArray_DATA((PyArrayObject*)f_array);
    if ((size_t)ldf * n > 0) memset(f_data, 0, (size_t)ldf * n * sizeof(f64));

    f64* dwork = (f64*)calloc(2 * n, sizeof(f64));
    if (!dwork) {
        Py_DECREF(f_array);
        PyArray_DiscardWritebackIfCopy(a_array);
        PyArray_DiscardWritebackIfCopy(b_array);
        PyArray_DiscardWritebackIfCopy(u_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(kstair_array);
        Py_DECREF(u_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info = 0;
    sb06nd(n, m, kmax, a_data, lda, b_data, ldb, kstair_data,
           u_data, ldu, f_data, ldf, dwork, &info);

    free(dwork);

    if (info < 0) {
        Py_DECREF(f_array);
        PyArray_DiscardWritebackIfCopy(a_array);
        PyArray_DiscardWritebackIfCopy(b_array);
        PyArray_DiscardWritebackIfCopy(u_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(kstair_array);
        Py_DECREF(u_array);
        PyErr_Format(PyExc_ValueError, "sb06nd: illegal value for argument %d", -info);
        return NULL;
    }

    /* Apply writebacks for modified arrays */
    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(u_array);

    PyObject* result = Py_BuildValue("OOOOi", a_array, b_array, u_array, f_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(kstair_array);
    Py_DECREF(u_array);

    return result;
}


/* Python wrapper for sb08cd - Left coprime factorization with inner denominator */
PyObject* py_sb08cd(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;

    const char* dico;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    f64 tol = 0.0;

    static char* kwlist[] = {"dico", "a", "b", "c", "d", "tol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sOOOO|d", kwlist,
                                     &dico, &a_obj, &b_obj, &c_obj, &d_obj, &tol)) {
        return NULL;
    }

    if (dico[0] != 'C' && dico[0] != 'c' && dico[0] != 'D' && dico[0] != 'd') {
        PyErr_SetString(PyExc_ValueError, "dico must be 'C' or 'D'");
        return NULL;
    }

    PyArrayObject* a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject* b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject* c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject* d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 m = (i32)PyArray_DIM(d_array, 1);
    i32 p = (i32)PyArray_DIM(d_array, 0);

    i32 lda = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 maxmp = m > p ? m : p;
    i32 ldc = maxmp > 1 ? maxmp : 1;
    i32 ldd = maxmp > 1 ? maxmp : 1;
    i32 ldbr = n > 1 ? n : 1;
    i32 lddr = p > 1 ? p : 1;

    f64* a_data = (f64*)PyArray_DATA(a_array);
    f64* b_data = (f64*)PyArray_DATA(b_array);
    f64* c_data = (f64*)PyArray_DATA(c_array);
    f64* d_data = (f64*)PyArray_DATA(d_array);

    npy_intp br_dims[2] = {n, p};
    npy_intp dr_dims[2] = {p, p};

    PyArrayObject* br_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, br_dims, NPY_DOUBLE,
                                                           NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyArrayObject* dr_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dr_dims, NPY_DOUBLE,
                                                           NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!br_array || !dr_array) {
        Py_XDECREF(br_array);
        Py_XDECREF(dr_array);
        PyArray_DiscardWritebackIfCopy(a_array);
        PyArray_DiscardWritebackIfCopy(b_array);
        PyArray_DiscardWritebackIfCopy(c_array);
        PyArray_DiscardWritebackIfCopy(d_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }
    f64* br_data = (f64*)PyArray_DATA(br_array);
    f64* dr_data = (f64*)PyArray_DATA(dr_array);
    size_t br_size = (size_t)n * (p > 1 ? p : 1);
    size_t dr_size = (size_t)p * (p > 1 ? p : 1);
    if (br_size > 0) memset(br_data, 0, br_size * sizeof(f64));
    if (dr_size > 0) memset(dr_data, 0, dr_size * sizeof(f64));

    i32 min1 = n * (n + 5);
    i32 min2 = p * (p + 2);
    i32 min3 = 4 * p;
    i32 min4 = 4 * m;
    i32 minwrk = min1 > min2 ? min1 : min2;
    minwrk = minwrk > min3 ? minwrk : min3;
    minwrk = minwrk > min4 ? minwrk : min4;
    minwrk = minwrk > 1 ? minwrk : 1;
    i32 ldwork = p * n + minwrk;

    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (!dwork) {
        free(dwork);
        Py_DECREF(br_array);
        Py_DECREF(dr_array);
        PyArray_DiscardWritebackIfCopy(a_array);
        PyArray_DiscardWritebackIfCopy(b_array);
        PyArray_DiscardWritebackIfCopy(c_array);
        PyArray_DiscardWritebackIfCopy(d_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return PyErr_NoMemory();
    }

    i32 nq = 0, nr = 0;
    i32 iwarn = 0, info = 0;

    sb08cd(dico, n, m, p, a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           &nq, &nr, br_data, ldbr, dr_data, lddr, tol, dwork, ldwork, &iwarn, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    if (info < 0) {
        Py_DECREF(br_array);
        Py_DECREF(dr_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        PyErr_Format(PyExc_ValueError, "sb08cd: illegal value for argument %d", -info);
        return NULL;
    }

    PyObject* result = Py_BuildValue("iiOOii", nq, nr, br_array, dr_array, iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(br_array);
    Py_DECREF(dr_array);

    return result;
}

PyObject* py_sb08fd(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *dico;
    int n, m, p;
    PyObject *alpha_obj, *a_obj, *b_obj, *c_obj, *d_obj;
    f64 tol = 0.0;

    static char *kwlist[] = {"dico", "n", "m", "p", "alpha", "a", "b", "c", "d", "tol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siiiOOOOO|d", kwlist,
                                      &dico, &n, &m, &p, &alpha_obj,
                                      &a_obj, &b_obj, &c_obj, &d_obj, &tol)) {
        return NULL;
    }

    PyArrayObject *alpha_array = (PyArrayObject*)PyArray_FROM_OTF(alpha_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (alpha_array == NULL) return NULL;

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        Py_DECREF(alpha_array);
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(alpha_array);
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c_array == NULL) {
        Py_DECREF(alpha_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (d_array == NULL) {
        Py_DECREF(alpha_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    i32 lda = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldc = p > 1 ? p : 1;
    i32 ldd = p > 1 ? p : 1;
    i32 ldcr = m > 1 ? m : 1;
    i32 lddr = m > 1 ? m : 1;

    f64 *alpha_data = (f64*)PyArray_DATA(alpha_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);

    i32 cr_size = m * n > 0 ? m * n : 1;
    i32 dr_size = m * m > 0 ? m * m : 1;

    i32 n_out = n > 0 ? n : 1;
    i32 m_out = m > 0 ? m : 1;
    npy_intp cr_dims[2] = {m_out, n_out};
    npy_intp dr_dims[2] = {m_out, m_out};
    npy_intp cr_strides[2] = {sizeof(f64), ldcr * sizeof(f64)};
    npy_intp dr_strides[2] = {sizeof(f64), lddr * sizeof(f64)};

    PyArrayObject *cr_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, cr_dims, NPY_DOUBLE,
                                                           cr_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyArrayObject *dr_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dr_dims, NPY_DOUBLE,
                                                           dr_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!cr_array || !dr_array) {
        Py_XDECREF(cr_array);
        Py_XDECREF(dr_array);
        Py_DECREF(alpha_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }
    f64 *cr_data = (f64*)PyArray_DATA(cr_array);
    f64 *dr_data = (f64*)PyArray_DATA(dr_array);
    if (cr_size > 0) memset(cr_data, 0, cr_size * sizeof(f64));
    if (dr_size > 0) memset(dr_data, 0, dr_size * sizeof(f64));

    i32 min1 = n * (n + 5);
    i32 min2 = 5 * m;
    i32 min3 = 4 * p;
    i32 ldwork = min1 > min2 ? min1 : min2;
    ldwork = ldwork > min3 ? ldwork : min3;
    ldwork = ldwork > 1 ? ldwork : 1;
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (!dwork) {
        free(dwork);
        Py_DECREF(cr_array);
        Py_DECREF(dr_array);
        Py_DECREF(alpha_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return PyErr_NoMemory();
    }

    i32 nq = 0, nr = 0;
    i32 iwarn = 0, info = 0;

    sb08fd(dico, n, m, p, alpha_data, a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           &nq, &nr, cr_data, ldcr, dr_data, lddr, tol, dwork, ldwork, &iwarn, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("(OOOOOiiii)", a_array, b_array, c_array, cr_array, dr_array,
                                      nq, nr, iwarn, info);
    Py_DECREF(alpha_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(cr_array);
    Py_DECREF(dr_array);
    return result;
}

PyObject* py_sb08gd(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *br_obj, *dr_obj;

    if (!PyArg_ParseTuple(args, "OOOOOO", &a_obj, &b_obj, &c_obj, &d_obj, &br_obj, &dr_obj)) {
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

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    PyArrayObject *br_array = (PyArrayObject*)PyArray_FROM_OTF(br_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (br_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }

    PyArrayObject *dr_array = (PyArrayObject*)PyArray_FROM_OTF(dr_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (dr_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(br_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 m = (i32)PyArray_DIM(b_array, 1);
    i32 p = (i32)PyArray_DIM(c_array, 0);
    i32 lda = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldc = p > 1 ? p : 1;
    i32 ldd = p > 1 ? p : 1;
    i32 ldbr = n > 1 ? n : 1;
    i32 lddr = p > 1 ? p : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *br_data = (f64*)PyArray_DATA(br_array);
    f64 *dr_data = (f64*)PyArray_DATA(dr_array);

    i32 ldwork = 4 * p > 1 ? 4 * p : 1;
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    i32 *iwork = (i32*)calloc(p > 1 ? p : 1, sizeof(i32));

    if (!dwork || !iwork) {
        free(dwork);
        free(iwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(br_array);
        Py_DECREF(dr_array);
        return PyErr_NoMemory();
    }

    i32 info = 0;

    sb08gd(n, m, p, a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           br_data, ldbr, dr_data, lddr, dwork, iwork, &info);

    f64 rcond = dwork[0];

    free(dwork);
    free(iwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(dr_array);

    PyObject *result = Py_BuildValue("(OOOOOdi)", a_array, b_array, c_array, d_array, dr_array,
                                      rcond, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(br_array);
    Py_DECREF(dr_array);
    return result;
}


/*
 * SB02PD - Continuous-time algebraic Riccati equation solver using matrix
 * sign function method with error bounds and condition estimates.
 */
PyObject* py_sb02pd(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;
    char *job_str = "X", *trana_str = "N", *uplo_str = "U";
    PyObject *a_obj, *g_obj, *q_obj;
    PyArrayObject *a_array, *g_array, *q_array;

    static char *kwlist[] = {"a", "g", "q", "job", "trana", "uplo", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|sss", kwlist,
                                      &a_obj, &g_obj, &q_obj, &job_str, &trana_str, &uplo_str)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) return NULL;

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 n = (i32)a_dims[0];

    g_array = (PyArrayObject*)PyArray_FROM_OTF(g_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (g_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (q_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(g_array);
        return NULL;
    }

    i32 lda = (n > 1) ? n : 1;
    i32 ldg = lda;
    i32 ldq = lda;
    i32 ldx = lda;

    int all = (job_str[0] == 'A' || job_str[0] == 'a');
    i32 n2 = 2 * n;
    i32 minwrk;
    if (all) {
        i32 opt1 = n2 * n2 + 8 * n + 1;
        i32 opt2 = 6 * n * n;
        minwrk = (opt1 > opt2) ? opt1 : opt2;
    } else {
        minwrk = n2 * n2 + 8 * n + 1;
    }
    i32 ldwork = (minwrk > 1) ? minwrk : 1;

    i32 liwork;
    if (all) {
        i32 opt1 = 2 * n;
        i32 opt2 = n * n;
        liwork = (opt1 > opt2) ? opt1 : opt2;
    } else {
        liwork = 2 * n;
    }
    if (liwork < 1) liwork = 1;

    const f64* a = (const f64*)PyArray_DATA(a_array);
    const f64* g = (const f64*)PyArray_DATA(g_array);
    const f64* q = (const f64*)PyArray_DATA(q_array);

    npy_intp x_dims[2] = {n, n};
    npy_intp x_strides[2] = {sizeof(f64), n * sizeof(f64)};
    PyObject* x_array = PyArray_New(&PyArray_Type, 2, x_dims, NPY_DOUBLE, x_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (x_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(g_array);
        Py_DECREF(q_array);
        return NULL;
    }
    f64* x = (f64*)PyArray_DATA((PyArrayObject*)x_array);
    if (n * n > 0) memset(x, 0, n * n * sizeof(f64));

    npy_intp wr_dims[1] = {n};
    PyObject* wr_array = PyArray_SimpleNew(1, wr_dims, NPY_DOUBLE);
    PyObject* wi_array = PyArray_SimpleNew(1, wr_dims, NPY_DOUBLE);
    if (wr_array == NULL || wi_array == NULL) {
        Py_XDECREF(wr_array);
        Py_XDECREF(wi_array);
        Py_DECREF(x_array);
        Py_DECREF(a_array);
        Py_DECREF(g_array);
        Py_DECREF(q_array);
        return NULL;
    }
    f64* wr = (f64*)PyArray_DATA((PyArrayObject*)wr_array);
    f64* wi = (f64*)PyArray_DATA((PyArrayObject*)wi_array);
    if (n > 0) {
        memset(wr, 0, n * sizeof(f64));
        memset(wi, 0, n * sizeof(f64));
    }

    i32* iwork = (i32*)calloc(liwork, sizeof(i32));
    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (iwork == NULL || dwork == NULL) {
        free(iwork);
        free(dwork);
        Py_DECREF(x_array);
        Py_DECREF(wr_array);
        Py_DECREF(wi_array);
        Py_DECREF(a_array);
        Py_DECREF(g_array);
        Py_DECREF(q_array);
        return PyErr_NoMemory();
    }

    f64 rcond = 0.0, ferr = 0.0;
    i32 info = 0;

    sb02pd(job_str, trana_str, uplo_str, n, a, lda, g, ldg, q, ldq, x, ldx,
           &rcond, &ferr, wr, wi, iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    if (info < 0) {
        Py_DECREF(x_array);
        Py_DECREF(wr_array);
        Py_DECREF(wi_array);
        Py_DECREF(a_array);
        Py_DECREF(g_array);
        Py_DECREF(q_array);
        PyErr_Format(PyExc_ValueError, "sb02pd: illegal value for argument %d", -info);
        return NULL;
    }

    Py_DECREF(a_array);
    Py_DECREF(g_array);
    Py_DECREF(q_array);

    PyObject* result = Py_BuildValue("OddOOi", x_array, rcond, ferr, wr_array, wi_array, info);
    Py_DECREF(x_array);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);

    return result;
}

/* sb03td - Continuous-time Lyapunov equation solver with condition/error estimation */
PyObject* py_sb03td(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;

    static char* kwlist[] = {"job", "fact", "trana", "uplo", "lyapun",
                             "a", "c", "t", "u", "x", "scale", NULL};

    const char* job;
    const char* fact;
    const char* trana;
    const char* uplo;
    const char* lyapun;
    PyObject* a_obj = Py_None;
    PyObject* c_obj = Py_None;
    PyObject* t_obj = Py_None;
    PyObject* u_obj = Py_None;
    PyObject* x_obj = Py_None;
    double scale_in = 1.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssss|OOOOOd", kwlist,
                                     &job, &fact, &trana, &uplo, &lyapun,
                                     &a_obj, &c_obj, &t_obj, &u_obj, &x_obj,
                                     &scale_in)) {
        return NULL;
    }

    char job_c = toupper(job[0]);
    char fact_c = toupper(fact[0]);
    char trana_c = toupper(trana[0]);
    char uplo_c = toupper(uplo[0]);
    char lyapun_c = toupper(lyapun[0]);

    bool jobx = (job_c == 'X');
    bool jobs = (job_c == 'S');
    bool jobc = (job_c == 'C');
    bool jobe = (job_c == 'E');
    bool joba = (job_c == 'A');
    bool nofact = (fact_c == 'N');
    bool update = (lyapun_c == 'O');

    if (!jobx && !jobs && !jobc && !jobe && !joba) {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'X', 'S', 'C', 'E', or 'A'");
        return NULL;
    }
    if (fact_c != 'F' && fact_c != 'N') {
        PyErr_SetString(PyExc_ValueError, "FACT must be 'F' or 'N'");
        return NULL;
    }
    if (trana_c != 'N' && trana_c != 'T' && trana_c != 'C') {
        PyErr_SetString(PyExc_ValueError, "TRANA must be 'N', 'T', or 'C'");
        return NULL;
    }
    if (uplo_c != 'U' && uplo_c != 'L') {
        PyErr_SetString(PyExc_ValueError, "UPLO must be 'U' or 'L'");
        return NULL;
    }
    if (lyapun_c != 'O' && lyapun_c != 'R') {
        PyErr_SetString(PyExc_ValueError, "LYAPUN must be 'O' or 'R'");
        return NULL;
    }

    i32 n = 0;
    PyArrayObject* a_array = NULL;
    PyArrayObject* c_array = NULL;
    PyArrayObject* t_array = NULL;
    PyArrayObject* u_array = NULL;
    PyArrayObject* x_array = NULL;

    if (nofact && a_obj != Py_None) {
        a_array = (PyArrayObject*)PyArray_FROM_OTF(
            a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
        if (!a_array) return NULL;
        n = (i32)PyArray_DIM(a_array, 0);
    } else if (!nofact && t_obj != Py_None) {
        t_array = (PyArrayObject*)PyArray_FROM_OTF(
            t_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_INOUT_ARRAY2);
        if (!t_array) return NULL;
        n = (i32)PyArray_DIM(t_array, 0);
    } else if (c_obj != Py_None) {
        c_array = (PyArrayObject*)PyArray_FROM_OTF(
            c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
        if (!c_array) return NULL;
        n = (i32)PyArray_DIM(c_array, 0);
    }

    if (n < 0) {
        Py_XDECREF(a_array);
        Py_XDECREF(c_array);
        Py_XDECREF(t_array);
        PyErr_SetString(PyExc_ValueError, "N must be >= 0");
        return NULL;
    }

    i32 lda = (n > 1) ? n : 1;
    i32 ldt = lda;
    i32 ldu = lda;
    i32 ldc = lda;
    i32 ldx = lda;

    if (nofact && !a_array && a_obj != Py_None) {
        a_array = (PyArrayObject*)PyArray_FROM_OTF(
            a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
        if (!a_array) {
            Py_XDECREF(c_array);
            Py_XDECREF(t_array);
            return NULL;
        }
    }

    if (!jobs && !c_array && c_obj != Py_None) {
        c_array = (PyArrayObject*)PyArray_FROM_OTF(
            c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
        if (!c_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(t_array);
            return NULL;
        }
    }

    f64* t_data = NULL;
    if (!nofact) {
        if (!t_array && t_obj != Py_None) {
            t_array = (PyArrayObject*)PyArray_FROM_OTF(
                t_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_INOUT_ARRAY2);
        }
        if (t_array) {
            t_data = (f64*)PyArray_DATA(t_array);
        }
    }

    if (nofact) {
        npy_intp dims[2] = {n, n};
        npy_intp strides[2] = {sizeof(f64), n * sizeof(f64)};
        t_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                               strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!t_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(c_array);
            return NULL;
        }
        t_data = (f64*)PyArray_DATA(t_array);
        if (n * n > 0) memset(t_data, 0, n * n * sizeof(f64));
    }

    f64* u_data = NULL;
    if (update) {
        if (!nofact && u_obj != Py_None) {
            u_array = (PyArrayObject*)PyArray_FROM_OTF(
                u_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_INOUT_ARRAY2);
            if (!u_array) {
                Py_XDECREF(a_array);
                Py_XDECREF(c_array);
                Py_XDECREF(t_array);
                return NULL;
            }
            u_data = (f64*)PyArray_DATA(u_array);
        } else {
            npy_intp dims[2] = {n, n};
            npy_intp strides[2] = {sizeof(f64), n * sizeof(f64)};
            u_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                                   strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            if (!u_array) {
                Py_XDECREF(a_array);
                Py_XDECREF(c_array);
                Py_XDECREF(t_array);
                return NULL;
            }
            u_data = (f64*)PyArray_DATA(u_array);
            if (n * n > 0) memset(u_data, 0, n * n * sizeof(f64));
        }
    } else {
        npy_intp dims[2] = {1, 1};
        npy_intp strides[2] = {sizeof(f64), sizeof(f64)};
        u_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                               strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!u_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(c_array);
            Py_XDECREF(t_array);
            return NULL;
        }
        u_data = (f64*)PyArray_DATA(u_array);
        memset(u_data, 0, sizeof(f64));
        ldu = 1;
    }

    f64* x_data = NULL;
    if (!jobs) {
        if ((jobc || jobe) && x_obj != Py_None) {
            x_array = (PyArrayObject*)PyArray_FROM_OTF(
                x_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_INOUT_ARRAY2);
            if (!x_array) {
                Py_XDECREF(a_array);
                Py_XDECREF(c_array);
                Py_XDECREF(t_array);
                Py_XDECREF(u_array);
                return NULL;
            }
            x_data = (f64*)PyArray_DATA(x_array);
        } else {
            npy_intp dims[2] = {n, n};
            npy_intp strides[2] = {sizeof(f64), n * sizeof(f64)};
            x_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                                   strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            if (!x_array) {
                Py_XDECREF(a_array);
                Py_XDECREF(c_array);
                Py_XDECREF(t_array);
                Py_XDECREF(u_array);
                return NULL;
            }
            x_data = (f64*)PyArray_DATA(x_array);
            if (n * n > 0) memset(x_data, 0, n * n * sizeof(f64));
        }
    } else {
        npy_intp dims[2] = {1, 1};
        npy_intp strides[2] = {sizeof(f64), sizeof(f64)};
        x_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                               strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!x_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(c_array);
            Py_XDECREF(t_array);
            Py_XDECREF(u_array);
            return NULL;
        }
        x_data = (f64*)PyArray_DATA(x_array);
        memset(x_data, 0, sizeof(f64));
        ldx = 1;
    }

    f64* a_data = a_array ? (f64*)PyArray_DATA(a_array) : NULL;
    f64* c_data = c_array ? (f64*)PyArray_DATA(c_array) : NULL;

    i32 nn = n * n;
    i32 ldwork;
    if (jobx) {
        if (nofact) {
            ldwork = (nn > 3 * n) ? nn : 3 * n;
            ldwork = (ldwork > 1) ? ldwork : 1;
        } else {
            ldwork = (nn > 1) ? nn : 1;
        }
    } else if (jobs || jobc) {
        ldwork = (2 * nn > 3 * n) ? 2 * nn : 3 * n;
        ldwork = (ldwork > 1) ? ldwork : 1;
    } else {
        if (update) {
            ldwork = 3 * nn;
        } else {
            ldwork = 3 * nn + n - 1;
        }
        ldwork = (ldwork > 1) ? ldwork : 1;
    }

    if (nofact && ldwork < 3 * n) {
        ldwork = 3 * n;
    }
    ldwork = (ldwork > 1) ? ldwork : 1;

    npy_intp wr_dim[1] = {n};
    PyArrayObject* wr_array = (PyArrayObject*)PyArray_SimpleNew(1, wr_dim, NPY_DOUBLE);
    PyArrayObject* wi_array = (PyArrayObject*)PyArray_SimpleNew(1, wr_dim, NPY_DOUBLE);
    if (!wr_array || !wi_array) {
        Py_XDECREF(wr_array);
        Py_XDECREF(wi_array);
        Py_XDECREF(a_array);
        Py_XDECREF(c_array);
        Py_XDECREF(t_array);
        Py_XDECREF(u_array);
        Py_XDECREF(x_array);
        return NULL;
    }
    f64* wr = (f64*)PyArray_DATA(wr_array);
    f64* wi = (f64*)PyArray_DATA(wi_array);
    if (n > 0) {
        memset(wr, 0, n * sizeof(f64));
        memset(wi, 0, n * sizeof(f64));
    }

    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));
    i32* iwork = (i32*)calloc((nn > 1) ? nn : 1, sizeof(i32));

    if (!dwork || !iwork) {
        free(dwork);
        free(iwork);
        Py_DECREF(wr_array);
        Py_DECREF(wi_array);
        Py_XDECREF(a_array);
        Py_XDECREF(c_array);
        Py_XDECREF(t_array);
        Py_XDECREF(u_array);
        Py_XDECREF(x_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 scale = scale_in;
    f64 sep = 0.0;
    f64 rcond = 0.0;
    f64 ferr = 0.0;
    i32 info = 0;

    if (!update) ldu = 1;
    if (jobs) {
        ldc = 1;
        ldx = 1;
    }

    sb03td(job, fact, trana, uplo, lyapun, n, &scale,
           a_data, lda, t_data, ldt, u_data, ldu,
           c_data, ldc, x_data, ldx, &sep, &rcond, &ferr,
           wr, wi, iwork, dwork, ldwork, &info);

    free(dwork);
    free(iwork);

    if (info < 0) {
        PyArray_ResolveWritebackIfCopy(t_array);
        PyArray_ResolveWritebackIfCopy(u_array);
        PyArray_ResolveWritebackIfCopy(x_array);
        Py_XDECREF(a_array);
        Py_XDECREF(c_array);
        Py_XDECREF(t_array);
        Py_XDECREF(u_array);
        Py_XDECREF(x_array);
        Py_XDECREF(wr_array);
        Py_XDECREF(wi_array);
        PyErr_Format(PyExc_ValueError, "sb03td: illegal value for argument %d", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(t_array);
    PyArray_ResolveWritebackIfCopy(u_array);
    PyArray_ResolveWritebackIfCopy(x_array);

    PyObject* result = Py_BuildValue("OOOOOddddi", x_array, t_array, u_array, wr_array, wi_array,
                                      scale, sep, rcond, ferr, info);

    Py_XDECREF(a_array);
    Py_XDECREF(c_array);
    Py_XDECREF(t_array);
    Py_XDECREF(u_array);
    Py_XDECREF(x_array);
    Py_XDECREF(wr_array);
    Py_XDECREF(wi_array);

    return result;
}


/* Python wrapper for sb08ed - Left coprime factorization with prescribed stability degree */
/* Python wrapper for sb08ed - Left coprime factorization with prescribed stability degree */
PyObject* py_sb08ed(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;

    const char* dico;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *alpha_obj;
    f64 tol = 0.0;

    static char* kwlist[] = {"dico", "a", "b", "c", "d", "alpha", "tol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sOOOOO|d", kwlist,
                                     &dico, &a_obj, &b_obj, &c_obj, &d_obj, &alpha_obj, &tol)) {
        return NULL;
    }

    if (dico[0] != 'C' && dico[0] != 'c' && dico[0] != 'D' && dico[0] != 'd') {
        PyErr_SetString(PyExc_ValueError, "dico must be 'C' or 'D'");
        return NULL;
    }

    PyArrayObject* alpha_array = (PyArrayObject*)PyArray_FROM_OTF(alpha_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (alpha_array == NULL) return NULL;

    f64* alpha_data = (f64*)PyArray_DATA(alpha_array);
    bool discr = (dico[0] == 'D' || dico[0] == 'd');
    if (discr) {
        if (alpha_data[0] < 0.0 || alpha_data[0] >= 1.0 ||
            alpha_data[1] < 0.0 || alpha_data[1] >= 1.0) {
            Py_DECREF(alpha_array);
            PyErr_SetString(PyExc_ValueError, "For discrete-time: 0 <= alpha < 1");
            return NULL;
        }
    } else {
        if (alpha_data[0] >= 0.0 || alpha_data[1] >= 0.0) {
            Py_DECREF(alpha_array);
            PyErr_SetString(PyExc_ValueError, "For continuous-time: alpha < 0");
            return NULL;
        }
    }

    PyArrayObject* a_array_in = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array_in == NULL) {
        Py_DECREF(alpha_array);
        return NULL;
    }

    PyArrayObject* b_array_in = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (b_array_in == NULL) {
        Py_DECREF(alpha_array);
        Py_DECREF(a_array_in);
        return NULL;
    }

    PyArrayObject* c_array_in = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (c_array_in == NULL) {
        Py_DECREF(alpha_array);
        Py_DECREF(a_array_in);
        Py_DECREF(b_array_in);
        return NULL;
    }

    PyArrayObject* d_array_in = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (d_array_in == NULL) {
        Py_DECREF(alpha_array);
        Py_DECREF(a_array_in);
        Py_DECREF(b_array_in);
        Py_DECREF(c_array_in);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array_in, 0);
    i32 m = (i32)PyArray_DIM(d_array_in, 1);
    i32 p = (i32)PyArray_DIM(d_array_in, 0);

    i32 lda = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 maxmp = m > p ? m : p;
    i32 ldc = maxmp > 1 ? maxmp : 1;
    i32 ldd = maxmp > 1 ? maxmp : 1;
    i32 ldbr = n > 1 ? n : 1;
    i32 lddr = p > 1 ? p : 1;

    /* For output arrays, ensure at least 1 for dimensions to avoid zero-size arrays */
    i32 p_out = p > 0 ? p : 1;
    i32 n_out = n > 0 ? n : 1;

    /* SLICOT requires B to be LDB x MAX(M,P), C to be LDC x N where LDC >= MAX(M,P),
       and D to be LDD x MAX(M,P) where LDD >= MAX(M,P) for workspace during dual computation.
       Create work arrays with proper dimensions and copy input data. */
    npy_intp a_dims[2] = {lda, n > 0 ? n : 1};
    npy_intp b_dims[2] = {ldb, maxmp > 0 ? maxmp : 1};
    npy_intp c_dims[2] = {ldc, n > 0 ? n : 1};
    npy_intp d_dims[2] = {ldd, maxmp > 0 ? maxmp : 1};

    PyArrayObject* a_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, a_dims, NPY_DOUBLE,
                                                          NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyArrayObject* b_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE,
                                                          NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyArrayObject* c_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE,
                                                          NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyArrayObject* d_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, d_dims, NPY_DOUBLE,
                                                          NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (!a_array || !b_array || !c_array || !d_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        Py_DECREF(alpha_array);
        Py_DECREF(a_array_in);
        Py_DECREF(b_array_in);
        Py_DECREF(c_array_in);
        Py_DECREF(d_array_in);
        return PyErr_NoMemory();
    }

    f64* a_data = (f64*)PyArray_DATA(a_array);
    f64* b_data = (f64*)PyArray_DATA(b_array);
    f64* c_data = (f64*)PyArray_DATA(c_array);
    f64* d_data = (f64*)PyArray_DATA(d_array);
    f64* a_in = (f64*)PyArray_DATA(a_array_in);
    f64* b_in = (f64*)PyArray_DATA(b_array_in);
    f64* c_in = (f64*)PyArray_DATA(c_array_in);
    f64* d_in = (f64*)PyArray_DATA(d_array_in);

    /* Zero-initialize work arrays */
    memset(a_data, 0, (size_t)a_dims[0] * a_dims[1] * sizeof(f64));
    memset(b_data, 0, (size_t)b_dims[0] * b_dims[1] * sizeof(f64));
    memset(c_data, 0, (size_t)c_dims[0] * c_dims[1] * sizeof(f64));
    memset(d_data, 0, (size_t)d_dims[0] * d_dims[1] * sizeof(f64));

    /* Copy input A (n x n) */
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < n; i++) {
            a_data[i + j * lda] = a_in[i + j * n];
        }
    }

    /* Copy input B (n x m) into work array (ldb x maxmp) */
    for (i32 j = 0; j < m; j++) {
        for (i32 i = 0; i < n; i++) {
            b_data[i + j * ldb] = b_in[i + j * n];
        }
    }

    /* Copy input C (p x n) into work array (ldc x n) */
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < p; i++) {
            c_data[i + j * ldc] = c_in[i + j * p];
        }
    }

    /* Copy input D (p x m) into work array (ldd x maxmp) */
    for (i32 j = 0; j < m; j++) {
        for (i32 i = 0; i < p; i++) {
            d_data[i + j * ldd] = d_in[i + j * p];
        }
    }

    npy_intp br_dims[2] = {n_out, p_out};
    npy_intp dr_dims[2] = {p_out, p_out};

    PyArrayObject* br_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, br_dims, NPY_DOUBLE,
                                                           NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyArrayObject* dr_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dr_dims, NPY_DOUBLE,
                                                           NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!br_array || !dr_array) {
        Py_XDECREF(br_array);
        Py_XDECREF(dr_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(alpha_array);
        Py_DECREF(a_array_in);
        Py_DECREF(b_array_in);
        Py_DECREF(c_array_in);
        Py_DECREF(d_array_in);
        return NULL;
    }
    f64* br_data = (f64*)PyArray_DATA(br_array);
    f64* dr_data = (f64*)PyArray_DATA(dr_array);
    memset(br_data, 0, (size_t)n_out * p_out * sizeof(f64));
    memset(dr_data, 0, (size_t)p_out * p_out * sizeof(f64));

    i32 min1 = n * (n + 5);
    i32 min2 = 5 * p;
    i32 min3 = 4 * m;
    i32 minwrk = min1 > min2 ? min1 : min2;
    minwrk = minwrk > min3 ? minwrk : min3;
    minwrk = minwrk > 1 ? minwrk : 1;
    i32 ldwork = n * p + minwrk;

    f64* dwork = (f64*)calloc(ldwork > 1 ? ldwork : 1, sizeof(f64));

    if (!dwork) {
        Py_DECREF(br_array);
        Py_DECREF(dr_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(alpha_array);
        Py_DECREF(a_array_in);
        Py_DECREF(b_array_in);
        Py_DECREF(c_array_in);
        Py_DECREF(d_array_in);
        return PyErr_NoMemory();
    }

    i32 nq = 0, nr = 0;
    i32 iwarn = 0, info = 0;

    sb08ed(dico, n, m, p, alpha_data, a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           &nq, &nr, br_data, ldbr, dr_data, lddr, tol, dwork, ldwork, &iwarn, &info);

    free(dwork);

    /* Copy results back to output arrays with original dimensions */
    PyArrayObject* a_out = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                             NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject* b_out = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                                             NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject* c_out = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                                             NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject* d_out = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                                             NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (a_out && b_out && c_out && d_out) {
        f64* a_out_data = (f64*)PyArray_DATA(a_out);
        f64* b_out_data = (f64*)PyArray_DATA(b_out);
        f64* c_out_data = (f64*)PyArray_DATA(c_out);
        f64* d_out_data = (f64*)PyArray_DATA(d_out);

        /* Copy back A (n x n) */
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < n; i++) {
                a_out_data[i + j * n] = a_data[i + j * lda];
            }
        }

        /* Copy back B (n x m) */
        for (i32 j = 0; j < m; j++) {
            for (i32 i = 0; i < n; i++) {
                b_out_data[i + j * n] = b_data[i + j * ldb];
            }
        }

        /* Copy back C (p x n) */
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < p; i++) {
                c_out_data[i + j * p] = c_data[i + j * ldc];
            }
        }

        /* Copy back D (p x m) */
        for (i32 j = 0; j < m; j++) {
            for (i32 i = 0; i < p; i++) {
                d_out_data[i + j * p] = d_data[i + j * ldd];
            }
        }

        PyArray_ResolveWritebackIfCopy(a_out);
        PyArray_ResolveWritebackIfCopy(b_out);
        PyArray_ResolveWritebackIfCopy(c_out);
        PyArray_ResolveWritebackIfCopy(d_out);
    }

    Py_XDECREF(a_out);
    Py_XDECREF(b_out);
    Py_XDECREF(c_out);
    Py_XDECREF(d_out);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    if (info < 0) {
        Py_DECREF(br_array);
        Py_DECREF(dr_array);
        Py_DECREF(alpha_array);
        Py_DECREF(a_array_in);
        Py_DECREF(b_array_in);
        Py_DECREF(c_array_in);
        Py_DECREF(d_array_in);
        PyErr_Format(PyExc_ValueError, "sb08ed: illegal value for argument %d", -info);
        return NULL;
    }

    PyObject* result = Py_BuildValue("iiOOii", nq, nr, br_array, dr_array, iwarn, info);

    Py_DECREF(alpha_array);
    Py_DECREF(a_array_in);
    Py_DECREF(b_array_in);
    Py_DECREF(c_array_in);
    Py_DECREF(d_array_in);
    Py_DECREF(br_array);
    Py_DECREF(dr_array);

    return result;
}

PyObject* py_sb08hd(PyObject* self, PyObject* args) {
    (void)self;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *cr_obj, *dr_obj;

    if (!PyArg_ParseTuple(args, "OOOOOO", &a_obj, &b_obj, &c_obj, &d_obj, &cr_obj, &dr_obj)) {
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

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    PyArrayObject *cr_array = (PyArrayObject*)PyArray_FROM_OTF(cr_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (cr_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }

    PyArrayObject *dr_array = (PyArrayObject*)PyArray_FROM_OTF(dr_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (dr_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(cr_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 m = (i32)PyArray_DIM(b_array, 1);
    i32 p = (i32)PyArray_DIM(c_array, 0);
    i32 lda = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldc = p > 1 ? p : 1;
    i32 ldd = p > 1 ? p : 1;
    i32 ldcr = m > 1 ? m : 1;
    i32 lddr = m > 1 ? m : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *cr_data = (f64*)PyArray_DATA(cr_array);
    f64 *dr_data = (f64*)PyArray_DATA(dr_array);

    i32 ldwork = 4 * m > 1 ? 4 * m : 1;
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (!dwork) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(cr_array);
        Py_DECREF(dr_array);
        return PyErr_NoMemory();
    }

    i32 info = 0;

    sb08hd(n, m, p, a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           cr_data, ldcr, dr_data, lddr, dwork, &info);

    f64 rcond = dwork[0];

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(dr_array);

    if (info < 0) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(cr_array);
        Py_DECREF(dr_array);
        PyErr_Format(PyExc_ValueError, "sb08hd: illegal value for argument %d", -info);
        return NULL;
    }

    PyObject *result = Py_BuildValue("di", rcond, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(cr_array);
    Py_DECREF(dr_array);
    return result;
}

/* Python wrapper for sb08my */
PyObject* py_sb08my(PyObject* self, PyObject* args) {
    PyObject *a_obj;
    f64 epsb;

    if (!PyArg_ParseTuple(args, "Od", &a_obj, &epsb)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) return NULL;

    i32 da = (i32)PyArray_DIM(a_array, 0) - 1;
    const f64* a = (f64*)PyArray_DATA(a_array);

    npy_intp dims[1] = {da + 1};
    PyArrayObject *b_array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }
    f64* b = (f64*)PyArray_DATA(b_array);

    f64 epsb_out = epsb;
    sb08my(da, a, b, &epsb_out);

    Py_DECREF(a_array);
    return Py_BuildValue("Od", b_array, epsb_out);
}

/* Python wrapper for sb08ny */
PyObject* py_sb08ny(PyObject* self, PyObject* args) {
    PyObject *a_obj;

    if (!PyArg_ParseTuple(args, "O", &a_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) return NULL;

    i32 da = (i32)PyArray_DIM(a_array, 0) - 1;
    const f64* a = (f64*)PyArray_DATA(a_array);

    npy_intp dims[1] = {da + 1};
    PyArrayObject *b_array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }
    f64* b = (f64*)PyArray_DATA(b_array);

    f64 epsb;
    sb08ny(da, a, b, &epsb);

    Py_DECREF(a_array);
    return Py_BuildValue("Od", b_array, epsb);
}

/* Python wrapper for sb08md */
PyObject* py_sb08md(PyObject* self, PyObject* args) {
    const char* acona;
    int da;
    PyObject *a_obj;

    if (!PyArg_ParseTuple(args, "siO", &acona, &da, &a_obj)) {
        return NULL;
    }

    if (acona[0] != 'A' && acona[0] != 'a' && acona[0] != 'B' && acona[0] != 'b') {
        PyErr_SetString(PyExc_ValueError, "ACONA must be 'A' or 'B'");
        return NULL;
    }

    if (da < 0) {
        PyErr_SetString(PyExc_ValueError, "DA must be non-negative");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    npy_intp a_len = PyArray_DIM(a_array, 0);
    if (a_len < da + 1) {
        PyErr_SetString(PyExc_ValueError, "Array A must have at least DA+1 elements");
        Py_DECREF(a_array);
        return NULL;
    }

    f64* a = (f64*)PyArray_DATA(a_array);

    npy_intp dims[1] = {da + 1};
    PyArrayObject *e_array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }
    f64* e = (f64*)PyArray_DATA(e_array);

    i32 ldwork = 5 * da + 5;
    f64* dwork = (f64*)calloc((size_t)ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 res;
    i32 info;
    sb08md(acona, (i32)da, a, &res, e, dwork, ldwork, &info);

    free(dwork);

    PyArrayObject *b_array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        return NULL;
    }
    memcpy(PyArray_DATA(b_array), a, (size_t)(da + 1) * sizeof(f64));

    PyArray_ResolveWritebackIfCopy(a_array);
    Py_DECREF(a_array);

    return Py_BuildValue("OOdi", e_array, b_array, res, info);
}

/* Python wrapper for sb08nd */
PyObject* py_sb08nd(PyObject* self, PyObject* args) {
    const char* acona;
    int da;
    PyObject *a_obj;

    if (!PyArg_ParseTuple(args, "siO", &acona, &da, &a_obj)) {
        return NULL;
    }

    if (acona[0] != 'A' && acona[0] != 'a' && acona[0] != 'B' && acona[0] != 'b') {
        PyErr_SetString(PyExc_ValueError, "ACONA must be 'A' or 'B'");
        return NULL;
    }

    if (da < 0) {
        PyErr_SetString(PyExc_ValueError, "DA must be non-negative");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    npy_intp a_len = PyArray_DIM(a_array, 0);
    if (a_len < da + 1) {
        PyErr_SetString(PyExc_ValueError, "Array A must have at least DA+1 elements");
        Py_DECREF(a_array);
        return NULL;
    }

    f64* a = (f64*)PyArray_DATA(a_array);

    npy_intp dims[1] = {da + 1};
    PyArrayObject *e_array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }
    f64* e = (f64*)PyArray_DATA(e_array);

    i32 ldwork = 5 * da + 5;
    f64* dwork = (f64*)calloc((size_t)ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 res;
    i32 info;
    sb08nd(acona, (i32)da, a, &res, e, dwork, ldwork, &info);

    free(dwork);

    PyArrayObject *b_array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        return NULL;
    }
    memcpy(PyArray_DATA(b_array), a, (size_t)(da + 1) * sizeof(f64));

    PyArray_ResolveWritebackIfCopy(a_array);
    Py_DECREF(a_array);

    return Py_BuildValue("OOdi", e_array, b_array, res, info);
}

/* Python wrapper for sb09md */
PyObject* py_sb09md(PyObject* self, PyObject* args) {
    int n, nc, nb;
    double tol;
    PyObject *h1_obj, *h2_obj;

    if (!PyArg_ParseTuple(args, "iiiOOd", &n, &nc, &nb, &h1_obj, &h2_obj, &tol)) {
        return NULL;
    }

    if (n < 0 || nc < 0 || nb < 0) {
        i32 info;
        if (n < 0) info = -1;
        else if (nc < 0) info = -2;
        else info = -3;

        npy_intp out_dims[2] = {(nc > 1 ? nc : 1), (nb > 1 ? nb : 1)};
        PyArrayObject *ss_array = (PyArrayObject*)PyArray_ZEROS(2, out_dims, NPY_DOUBLE, 1);
        PyArrayObject *se_array = (PyArrayObject*)PyArray_ZEROS(2, out_dims, NPY_DOUBLE, 1);
        PyArrayObject *pre_array = (PyArrayObject*)PyArray_ZEROS(2, out_dims, NPY_DOUBLE, 1);
        return Py_BuildValue("OOOi", ss_array, se_array, pre_array, info);
    }

    PyArrayObject *h1_array = (PyArrayObject*)PyArray_FROM_OTF(
        h1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    if (h1_array == NULL) return NULL;

    PyArrayObject *h2_array = (PyArrayObject*)PyArray_FROM_OTF(
        h2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    if (h2_array == NULL) {
        Py_DECREF(h1_array);
        return NULL;
    }

    npy_intp h1_rows = PyArray_DIM(h1_array, 0);
    npy_intp h1_cols = (PyArray_NDIM(h1_array) > 1) ? PyArray_DIM(h1_array, 1) : 1;

    if (nc > 0 && nb > 0 && n > 0) {
        if (h1_rows < nc || h1_cols < (npy_intp)(n * nb)) {
            PyErr_SetString(PyExc_ValueError, "H1 must be NC-by-(N*NB)");
            Py_DECREF(h1_array);
            Py_DECREF(h2_array);
            return NULL;
        }
    }

    f64* h1 = (f64*)PyArray_DATA(h1_array);
    f64* h2 = (f64*)PyArray_DATA(h2_array);
    i32 ldh1 = (i32)((h1_rows > 1) ? h1_rows : 1);
    i32 ldh2 = (i32)((PyArray_DIM(h2_array, 0) > 1) ? PyArray_DIM(h2_array, 0) : 1);

    npy_intp out_nc = (nc > 0) ? nc : 1;
    npy_intp out_nb = (nb > 0) ? nb : 1;
    npy_intp out_dims[2] = {out_nc, out_nb};
    npy_intp strides[2] = {(npy_intp)sizeof(f64), out_nc * (npy_intp)sizeof(f64)};

    PyArrayObject *ss_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, out_dims, NPY_DOUBLE, strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyArrayObject *se_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, out_dims, NPY_DOUBLE, strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyArrayObject *pre_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, out_dims, NPY_DOUBLE, strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (ss_array == NULL || se_array == NULL || pre_array == NULL) {
        Py_XDECREF(ss_array);
        Py_XDECREF(se_array);
        Py_XDECREF(pre_array);
        Py_DECREF(h1_array);
        Py_DECREF(h2_array);
        return NULL;
    }
    f64* ss = (f64*)PyArray_DATA(ss_array);
    f64* se = (f64*)PyArray_DATA(se_array);
    f64* pre = (f64*)PyArray_DATA(pre_array);
    size_t arr_size = (size_t)(out_nc * out_nb);
    if (arr_size > 0) {
        memset(ss, 0, arr_size * sizeof(f64));
        memset(se, 0, arr_size * sizeof(f64));
        memset(pre, 0, arr_size * sizeof(f64));
    }

    i32 ldss = (i32)out_nc;
    i32 ldse = (i32)out_nc;
    i32 ldpre = (i32)out_nc;
    i32 info;

    sb09md((i32)n, (i32)nc, (i32)nb, h1, ldh1, h2, ldh2,
           ss, ldss, se, ldse, pre, ldpre, (f64)tol, &info);

    Py_DECREF(h1_array);
    Py_DECREF(h2_array);

    return Py_BuildValue("OOOi", ss_array, se_array, pre_array, info);
}

/* Python wrapper for sb10id - Positive feedback controller for loop shaping */
PyObject* py_sb10id(PyObject* self, PyObject* args, PyObject* kwds) {
    (void)self;
    static char *kwlist[] = {"n", "m", "np", "a", "b", "c", "d", "factor", NULL};
    int n, m, np_val;
    double factor;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiOOOOd", kwlist,
                                      &n, &m, &np_val, &a_obj, &b_obj, &c_obj, &d_obj, &factor)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    if (!b_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    if (!c_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    if (!d_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    i32 n_ = (i32)n;
    i32 m_ = (i32)m;
    i32 np_ = (i32)np_val;

    i32 maxn1 = n_ > 1 ? n_ : 1;
    i32 maxm1 = m_ > 1 ? m_ : 1;
    i32 maxnp1 = np_ > 1 ? np_ : 1;

    i32 lda = maxn1;
    i32 ldb = maxn1;
    i32 ldc = maxnp1;
    i32 ldd = maxnp1;
    i32 ldak = maxn1;
    i32 ldbk = maxn1;
    i32 ldck = maxm1;
    i32 lddk = maxm1;

    f64 *a = (f64*)PyArray_DATA(a_array);
    f64 *b = (f64*)PyArray_DATA(b_array);
    f64 *c = (f64*)PyArray_DATA(c_array);
    f64 *d = (f64*)PyArray_DATA(d_array);

    npy_intp ak_dims[2] = {maxn1, n_};
    npy_intp ak_strides[2] = {sizeof(f64), ldak * sizeof(f64)};
    npy_intp bk_dims[2] = {maxn1, np_};
    npy_intp bk_strides[2] = {sizeof(f64), ldbk * sizeof(f64)};
    npy_intp ck_dims[2] = {maxm1, n_};
    npy_intp ck_strides[2] = {sizeof(f64), ldck * sizeof(f64)};
    npy_intp dk_dims[2] = {maxm1, np_};
    npy_intp dk_strides[2] = {sizeof(f64), lddk * sizeof(f64)};

    PyArrayObject *ak_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, ak_dims, NPY_DOUBLE, ak_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyArrayObject *bk_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, bk_dims, NPY_DOUBLE, bk_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyArrayObject *ck_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, ck_dims, NPY_DOUBLE, ck_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyArrayObject *dk_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, dk_dims, NPY_DOUBLE, dk_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (!ak_array || !bk_array || !ck_array || !dk_array) {
        Py_XDECREF(ak_array);
        Py_XDECREF(bk_array);
        Py_XDECREF(ck_array);
        Py_XDECREF(dk_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }
    f64 *ak = (f64*)PyArray_DATA(ak_array);
    f64 *bk = (f64*)PyArray_DATA(bk_array);
    f64 *ck = (f64*)PyArray_DATA(ck_array);
    f64 *dk = (f64*)PyArray_DATA(dk_array);
    if ((size_t)(maxn1 * n_) > 0) memset(ak, 0, (size_t)(maxn1 * n_) * sizeof(f64));
    if ((size_t)(maxn1 * np_) > 0) memset(bk, 0, (size_t)(maxn1 * np_) * sizeof(f64));
    if ((size_t)(maxm1 * n_) > 0) memset(ck, 0, (size_t)(maxm1 * n_) * sizeof(f64));
    if ((size_t)(maxm1 * np_) > 0) memset(dk, 0, (size_t)(maxm1 * np_) * sizeof(f64));

    f64 rcond[2] = {0.0, 0.0};

    i32 iwork_size = 2 * n_;
    if (n_ * n_ > iwork_size) iwork_size = n_ * n_;
    if (m_ > iwork_size) iwork_size = m_;
    if (np_ > iwork_size) iwork_size = np_;
    if (iwork_size < 1) iwork_size = 1;
    i32 *iwork = (i32*)calloc((size_t)iwork_size, sizeof(i32));

    i32 ldwork = 4*n_*n_ + m_*m_ + np_*np_ + 2*m_*n_ + n_*np_ + 4*n_;
    i32 inner_max = 4*n_*n_ + 8*n_;
    if (inner_max < 1) inner_max = 1;
    i32 tmp1 = 6*n_*n_ + 5 + inner_max;
    i32 tmp3 = n_*np_ + 2*n_;
    ldwork += (tmp1 > tmp3 ? tmp1 : tmp3);
    if (ldwork < 1) ldwork = 1;
    f64 *dwork = (f64*)calloc((size_t)ldwork, sizeof(f64));

    i32 bwork_size = 2 * n_;
    if (bwork_size < 1) bwork_size = 1;
    i32 *bwork = (i32*)calloc((size_t)bwork_size, sizeof(i32));

    if (!iwork || !dwork || !bwork) {
        free(iwork); free(dwork); free(bwork);
        Py_DECREF(ak_array);
        Py_DECREF(bk_array);
        Py_DECREF(ck_array);
        Py_DECREF(dk_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 nk = 0;
    i32 info = 0;

    sb10id(n_, m_, np_, a, lda, b, ldb, c, ldc, d, ldd, factor,
           &nk, ak, ldak, bk, ldbk, ck, ldck, dk, lddk, rcond,
           iwork, dwork, ldwork, bwork, &info);

    free(iwork);
    free(dwork);
    free(bwork);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    npy_intp rcond_dims[1] = {2};
    PyArrayObject *rcond_array = (PyArrayObject*)PyArray_SimpleNew(1, rcond_dims, NPY_DOUBLE);
    if (!rcond_array) {
        Py_DECREF(ak_array);
        Py_DECREF(bk_array);
        Py_DECREF(ck_array);
        Py_DECREF(dk_array);
        return NULL;
    }
    memcpy(PyArray_DATA(rcond_array), rcond, 2 * sizeof(f64));

    PyObject *result = Py_BuildValue("OOOOiOi", ak_array, bk_array, ck_array, dk_array, nk, rcond_array, info);
    Py_DECREF(ak_array);
    Py_DECREF(bk_array);
    Py_DECREF(ck_array);
    Py_DECREF(dk_array);
    Py_DECREF(rcond_array);

    return result;
}

PyObject* py_sb10kd(PyObject* self, PyObject* args, PyObject* kwds) {
    static char *kwlist[] = {"n", "m", "np", "a", "b", "c", "factor", NULL};
    PyObject *a_obj, *b_obj, *c_obj;
    int n, m, np_val;
    double factor;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiOOOd", kwlist,
                                      &n, &m, &np_val,
                                      &a_obj, &b_obj, &c_obj, &factor)) {
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

    i32 n_ = (i32)n;
    i32 m_ = (i32)m;
    i32 np_ = (i32)np_val;

    i32 lda = n_ > 1 ? n_ : 1;
    i32 ldb = n_ > 1 ? n_ : 1;
    i32 ldc = np_ > 1 ? np_ : 1;
    i32 ldak = n_ > 1 ? n_ : 1;
    i32 ldbk = n_ > 1 ? n_ : 1;
    i32 ldck = m_ > 1 ? m_ : 1;
    i32 lddk = m_ > 1 ? m_ : 1;

    f64 *a = (f64*)PyArray_DATA(a_array);
    f64 *b = (f64*)PyArray_DATA(b_array);
    f64 *c = (f64*)PyArray_DATA(c_array);

    npy_intp ak_dims[2] = {n_, n_};
    npy_intp ak_strides[2] = {sizeof(f64), ldak * sizeof(f64)};
    npy_intp bk_dims[2] = {n_, np_};
    npy_intp bk_strides[2] = {sizeof(f64), ldbk * sizeof(f64)};
    npy_intp ck_dims[2] = {m_, n_};
    npy_intp ck_strides[2] = {sizeof(f64), ldck * sizeof(f64)};
    npy_intp dk_dims[2] = {m_, np_};
    npy_intp dk_strides[2] = {sizeof(f64), lddk * sizeof(f64)};

    PyArrayObject *ak_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, ak_dims, NPY_DOUBLE, ak_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyArrayObject *bk_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, bk_dims, NPY_DOUBLE, bk_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyArrayObject *ck_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, ck_dims, NPY_DOUBLE, ck_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyArrayObject *dk_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, dk_dims, NPY_DOUBLE, dk_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (!ak_array || !bk_array || !ck_array || !dk_array) {
        Py_XDECREF(ak_array);
        Py_XDECREF(bk_array);
        Py_XDECREF(ck_array);
        Py_XDECREF(dk_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }
    f64 *ak = (f64*)PyArray_DATA(ak_array);
    f64 *bk = (f64*)PyArray_DATA(bk_array);
    f64 *ck = (f64*)PyArray_DATA(ck_array);
    f64 *dk = (f64*)PyArray_DATA(dk_array);
    if (n_ * n_ > 0) memset(ak, 0, n_ * n_ * sizeof(f64));
    if (n_ * np_ > 0) memset(bk, 0, n_ * np_ * sizeof(f64));
    if (m_ * n_ > 0) memset(ck, 0, m_ * n_ * sizeof(f64));
    if (m_ * np_ > 0) memset(dk, 0, m_ * np_ * sizeof(f64));

    f64 rcond[4];

    i32 t1 = 14*n_ + 23;
    i32 t2 = 16*n_;
    i32 t3 = 2*n_ + np_ + m_;
    i32 t4 = 3*(np_ + m_);
    i32 max1 = t1 > t2 ? t1 : t2;
    max1 = max1 > t3 ? max1 : t3;
    max1 = max1 > t4 ? max1 : t4;

    i32 s1 = n_*n_;
    i32 s2 = 11*n_*np_ + 2*m_*m_ + 8*np_*np_ + 8*m_*n_ + 4*m_*np_ + np_;
    i32 max2 = s1 > s2 ? s1 : s2;

    i32 ldwork = 15*n_*n_ + 6*n_ + max1 + max2;
    if (ldwork < 1) ldwork = 1;

    i32 liwork = 2*(n_ > (np_ + m_) ? n_ : (np_ + m_));
    if (liwork < 1) liwork = 1;

    i32 lbwork = 2*n_;
    if (lbwork < 1) lbwork = 1;

    i32 *iwork = (i32*)calloc(liwork, sizeof(i32));
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    i32 *bwork = (i32*)calloc(lbwork, sizeof(i32));

    if (!iwork || !dwork || !bwork) {
        free(iwork); free(dwork); free(bwork);
        Py_DECREF(ak_array);
        Py_DECREF(bk_array);
        Py_DECREF(ck_array);
        Py_DECREF(dk_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info = 0;

    sb10kd(n_, m_, np_, a, lda, b, ldb, c, ldc, factor,
           ak, ldak, bk, ldbk, ck, ldck, dk, lddk, rcond,
           iwork, dwork, ldwork, bwork, &info);

    free(iwork);
    free(dwork);
    free(bwork);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);

    npy_intp rcond_dims[1] = {4};
    PyArrayObject *rcond_array = (PyArrayObject*)PyArray_SimpleNew(1, rcond_dims, NPY_DOUBLE);
    if (!rcond_array) {
        Py_DECREF(ak_array);
        Py_DECREF(bk_array);
        Py_DECREF(ck_array);
        Py_DECREF(dk_array);
        return NULL;
    }
    memcpy(PyArray_DATA(rcond_array), rcond, 4 * sizeof(f64));

    PyObject *result = Py_BuildValue("OOOOOi", ak_array, bk_array, ck_array, dk_array, rcond_array, info);
    Py_DECREF(ak_array);
    Py_DECREF(bk_array);
    Py_DECREF(ck_array);
    Py_DECREF(dk_array);
    Py_DECREF(rcond_array);

    return result;
}

/* Python wrapper for sb10md */
PyObject* slicot_sb10md(PyObject* self, PyObject* args) {
    i32 nc, mp, lendat, f, ord, mnb;
    PyObject *nblock_obj, *itype_obj;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *omega_obj;
    f64 qutol;

    if (!PyArg_ParseTuple(args, "iiiiiiOOdOOOOO", &nc, &mp, &lendat, &f, &ord, &mnb,
                          &nblock_obj, &itype_obj, &qutol,
                          &a_obj, &b_obj, &c_obj, &d_obj, &omega_obj)) {
        return NULL;
    }

    PyArrayObject *nblock_array = (PyArrayObject*)PyArray_FROM_OTF(nblock_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *itype_array = (PyArrayObject*)PyArray_FROM_OTF(itype_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *omega_array = (PyArrayObject*)PyArray_FROM_OTF(omega_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!nblock_array || !itype_array || !a_array || !b_array || !c_array || !d_array || !omega_array) {
        Py_XDECREF(nblock_array);
        Py_XDECREF(itype_array);
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        Py_XDECREF(omega_array);
        return NULL;
    }

    i32 *nblock = (i32*)PyArray_DATA(nblock_array);
    i32 *itype = (i32*)PyArray_DATA(itype_array);
    f64 *a = (f64*)PyArray_DATA(a_array);
    f64 *b = (f64*)PyArray_DATA(b_array);
    f64 *c = (f64*)PyArray_DATA(c_array);
    f64 *d = (f64*)PyArray_DATA(d_array);
    f64 *omega = (f64*)PyArray_DATA(omega_array);

    i32 lda = nc > 1 ? nc : 1;
    i32 ldb = nc > 1 ? nc : 1;
    i32 ldc = mp > 1 ? mp : 1;
    i32 ldd = mp > 1 ? mp : 1;

    i32 mp_ord = mp * ord;
    i32 mp_f = mp + f;
    i32 ldad = qutol >= 0.0 && mp_ord > 0 ? mp_ord : 1;
    i32 ldbd = qutol >= 0.0 && mp_ord > 0 ? mp_ord : 1;
    i32 ldcd = qutol >= 0.0 && mp_f > 0 ? mp_f : 1;
    i32 lddd = qutol >= 0.0 && mp_f > 0 ? mp_f : 1;

    f64 *ad = (f64*)calloc(ldad * mp_ord > 0 ? ldad * mp_ord : 1, sizeof(f64));
    f64 *bd = (f64*)calloc(ldbd * mp_f > 0 ? ldbd * mp_f : 1, sizeof(f64));
    f64 *cd = (f64*)calloc(ldcd * mp_ord > 0 ? ldcd * mp_ord : 1, sizeof(f64));
    f64 *dd = (f64*)calloc(lddd * mp_f > 0 ? lddd * mp_f : 1, sizeof(f64));
    f64 *mju = (f64*)calloc(lendat, sizeof(f64));

    i32 ii = nc > 4 * mnb - 2 ? nc : 4 * mnb - 2;
    ii = ii > mp ? ii : mp;
    if (qutol >= 0.0 && 2 * ord + 1 > ii) ii = 2 * ord + 1;
    if (ii < 1) ii = 1;
    i32 liwork = ii;
    i32 *iwork = (i32*)calloc(liwork, sizeof(i32));

    i32 HNPTS = 2048;
    i32 lwa = mp * lendat + 2 * mnb + mp - 1;
    i32 lwb = lendat * (mp + 2) + ord * (ord + 2) + 1;
    i32 lw1 = 2 * lendat + 4 * HNPTS;
    i32 lw2 = lendat + 6 * HNPTS;
    i32 mn = 2 * lendat < 2 * ord + 1 ? 2 * lendat : 2 * ord + 1;
    i32 term1 = mn + 6 * ord + 4;
    i32 term2 = 2 * mn + 1;
    i32 lw3 = 2 * lendat * (2 * ord + 1) + (2 * lendat > 2 * ord + 1 ? 2 * lendat : 2 * ord + 1) +
              (term1 > term2 ? term1 : term2);
    i32 term3 = ord * ord + 5 * ord;
    i32 term4 = 6 * ord + 1 + (1 < ord ? 1 : ord);
    i32 lw4 = term3 > term4 ? term3 : term4;

    i32 lwa_t1 = nc + (nc > mp - 1 ? nc : mp - 1);
    i32 lwa_t2 = 2 * mp * mp * mnb - mp * mp + 9 * mnb * mnb + mp * mnb + 11 * mp + 33 * mnb - 11;
    i32 dlwmax = lwa + (lwa_t1 > lwa_t2 ? lwa_t1 : lwa_t2);

    if (qutol >= 0.0) {
        i32 dlw_term = lwb + lw1;
        if (lwb + lw2 > dlw_term) dlw_term = lwb + lw2;
        if (lwb + lw3 > dlw_term) dlw_term = lwb + lw3;
        if (lwb + lw4 > dlw_term) dlw_term = lwb + lw4;
        if (lwb + 2 * ord > dlw_term) dlw_term = lwb + 2 * ord;
        if (dlw_term > dlwmax) dlwmax = dlw_term;
    }
    i32 ldwork = dlwmax > 3 ? dlwmax : 3;
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));

    i32 icwrk = mp * mp + nc * mp;
    i32 clw1 = icwrk + nc * nc + 2 * nc;
    i32 clw2 = 6 * mp * mp * mnb + 13 * mp * mp + 6 * mnb + 6 * mp - 3;
    i32 clwmax = clw1 > clw2 ? clw1 : clw2;
    if (qutol >= 0.0) {
        i32 clw3 = lendat * (2 * ord + 3);
        i32 clw4 = ord * (ord + 3) + 1;
        i32 clw_d = clw3 > clw4 ? clw3 : clw4;
        if (clw_d > clwmax) clwmax = clw_d;
    }
    i32 lzwork = clwmax;
    c128 *zwork = (c128*)calloc(lzwork, sizeof(c128));

    if (!ad || !bd || !cd || !dd || !mju || !iwork || !dwork || !zwork) {
        free(ad); free(bd); free(cd); free(dd); free(mju);
        free(iwork); free(dwork); free(zwork);
        Py_DECREF(nblock_array);
        Py_DECREF(itype_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(omega_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 totord = 0;
    i32 info = 0;

    sb10md(nc, mp, lendat, f, &ord, mnb, nblock, itype, qutol,
           a, lda, b, ldb, c, ldc, d, ldd, omega,
           &totord, ad, ldad, bd, ldbd, cd, ldcd, dd, lddd, mju,
           iwork, liwork, dwork, ldwork, zwork, lzwork, &info);

    PyArrayObject *a_out = a_array;
    PyArrayObject *b_out = b_array;
    PyArrayObject *c_out = c_array;
    PyArrayObject *d_out = d_array;

    npy_intp ad_dims[2] = {totord, totord};
    npy_intp bd_dims[2] = {totord, mp_f};
    npy_intp cd_dims[2] = {mp_f, totord};
    npy_intp dd_dims[2] = {mp_f, mp_f};
    npy_intp mju_dims[1] = {lendat};

    PyArrayObject *ad_array, *bd_array, *cd_array, *dd_array_out, *mju_array;

    if (totord > 0 && qutol >= 0.0) {
        npy_intp ad_strides[2] = {sizeof(f64), totord * sizeof(f64)};
        ad_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, ad_dims, NPY_DOUBLE, ad_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        f64 *ad_out = (f64*)PyArray_DATA(ad_array);
        for (i32 j = 0; j < totord; j++) {
            memcpy(ad_out + j * totord, ad + j * ldad, totord * sizeof(f64));
        }
        free(ad);

        npy_intp bd_strides[2] = {sizeof(f64), totord * sizeof(f64)};
        bd_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, bd_dims, NPY_DOUBLE, bd_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        f64 *bd_out = (f64*)PyArray_DATA(bd_array);
        for (i32 j = 0; j < mp_f; j++) {
            memcpy(bd_out + j * totord, bd + j * ldbd, totord * sizeof(f64));
        }
        free(bd);

        npy_intp cd_strides[2] = {sizeof(f64), mp_f * sizeof(f64)};
        cd_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, cd_dims, NPY_DOUBLE, cd_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        f64 *cd_out = (f64*)PyArray_DATA(cd_array);
        for (i32 j = 0; j < totord; j++) {
            memcpy(cd_out + j * mp_f, cd + j * ldcd, mp_f * sizeof(f64));
        }
        free(cd);

        npy_intp dd_strides[2] = {sizeof(f64), mp_f * sizeof(f64)};
        dd_array_out = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dd_dims, NPY_DOUBLE, dd_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        f64 *dd_out = (f64*)PyArray_DATA(dd_array_out);
        for (i32 j = 0; j < mp_f; j++) {
            memcpy(dd_out + j * mp_f, dd + j * lddd, mp_f * sizeof(f64));
        }
        free(dd);
    } else {
        npy_intp zero_dims[2] = {0, 0};
        ad_array = (PyArrayObject*)PyArray_SimpleNew(2, zero_dims, NPY_DOUBLE);
        bd_array = (PyArrayObject*)PyArray_SimpleNew(2, zero_dims, NPY_DOUBLE);
        cd_array = (PyArrayObject*)PyArray_SimpleNew(2, zero_dims, NPY_DOUBLE);
        free(ad);
        free(bd);
        free(cd);
        if (qutol >= 0.0 && mp_f > 0) {
            npy_intp dd_strides[2] = {sizeof(f64), mp_f * sizeof(f64)};
            dd_array_out = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dd_dims, NPY_DOUBLE, dd_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
            f64 *dd_out = (f64*)PyArray_DATA(dd_array_out);
            for (i32 j = 0; j < mp_f; j++) {
                memcpy(dd_out + j * mp_f, dd + j * lddd, mp_f * sizeof(f64));
            }
            free(dd);
        } else {
            dd_array_out = (PyArrayObject*)PyArray_SimpleNew(2, zero_dims, NPY_DOUBLE);
            free(dd);
        }
    }

    mju_array = (PyArrayObject*)PyArray_SimpleNew(1, mju_dims, NPY_DOUBLE);
    memcpy(PyArray_DATA(mju_array), mju, lendat * sizeof(f64));
    free(mju);

    free(iwork);
    free(dwork);
    free(zwork);
    Py_DECREF(nblock_array);
    Py_DECREF(itype_array);

    PyObject *result = Py_BuildValue("OOOOiOOOOOi",
                                      a_out, b_out, c_out, d_out, totord,
                                      ad_array, bd_array, cd_array, dd_array_out, mju_array, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(omega_array);
    Py_DECREF(ad_array);
    Py_DECREF(bd_array);
    Py_DECREF(cd_array);
    Py_DECREF(dd_array_out);
    Py_DECREF(mju_array);

    return result;
}


/* Python wrapper for sb10sd - H2 optimal controller for discrete-time systems */
PyObject* py_sb10sd(PyObject* self, PyObject* args, PyObject* kwds) {
    static char *kwlist[] = {"n", "m", "np", "ncon", "nmeas", "a", "b", "c", "d", "tol", NULL};
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    int n, m, np_val, ncon, nmeas;
    double tol = 0.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiOOOO|d", kwlist,
                                      &n, &m, &np_val, &ncon, &nmeas,
                                      &a_obj, &b_obj, &c_obj, &d_obj, &tol)) {
        return NULL;
    }

    i32 n_ = (i32)n;
    i32 m_ = (i32)m;
    i32 np_ = (i32)np_val;
    i32 ncon_ = (i32)ncon;
    i32 nmeas_ = (i32)nmeas;

    i32 m1 = m_ - ncon_;
    i32 m2 = ncon_;
    i32 np1 = np_ - nmeas_;
    i32 np2 = nmeas_;

    if (n_ < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }
    if (m_ < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be non-negative");
        return NULL;
    }
    if (np_ < 0) {
        PyErr_SetString(PyExc_ValueError, "np must be non-negative");
        return NULL;
    }
    if (ncon_ < 0 || m1 < 0 || m2 > np1) {
        PyErr_SetString(PyExc_ValueError, "ncon must satisfy 0 <= ncon <= m and ncon <= np-nmeas");
        return NULL;
    }
    if (nmeas_ < 0 || np1 < 0 || np2 > m1) {
        PyErr_SetString(PyExc_ValueError, "nmeas must satisfy 0 <= nmeas <= np and nmeas <= m-ncon");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!b_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!c_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!d_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    i32 lda = n_ > 1 ? n_ : 1;
    i32 ldb = n_ > 1 ? n_ : 1;
    i32 ldc = np_ > 1 ? np_ : 1;
    i32 ldd = np_ > 1 ? np_ : 1;
    i32 ldak = n_ > 1 ? n_ : 1;
    i32 ldbk = n_ > 1 ? n_ : 1;
    i32 ldck = m2 > 1 ? m2 : 1;
    i32 lddk = m2 > 1 ? m2 : 1;
    i32 ldx = n_ > 1 ? n_ : 1;
    i32 ldy = n_ > 1 ? n_ : 1;

    i32 tmp1 = 14 * n_ + 23;
    i32 tmp2 = 16 * n_;
    i32 ws1 = 14 * n_ * n_ + 6 * n_ + (tmp1 > tmp2 ? tmp1 : tmp2);
    i32 tmp3 = 3 > m1 ? 3 : m1;
    i32 ws2 = m2 * (n_ + m2 + tmp3);
    i32 ws3 = np2 * (n_ + np2 + 3);
    i32 ldwork = ws1;
    if (ws2 > ldwork) ldwork = ws2;
    if (ws3 > ldwork) ldwork = ws3;
    if (ldwork < 1) ldwork = 1;

    i32 iw1 = m2 > (2 * n_) ? m2 : (2 * n_);
    i32 iw2 = n_ * n_;
    i32 iw3 = np2;
    i32 liwork = iw1 > iw2 ? iw1 : iw2;
    if (iw3 > liwork) liwork = iw3;
    if (liwork < 1) liwork = 1;

    f64 *ak = (f64*)calloc(ldak * n_, sizeof(f64));
    f64 *bk = (f64*)calloc(ldbk * np2, sizeof(f64));
    f64 *ck = (f64*)calloc(ldck * n_, sizeof(f64));
    f64 *dk = (f64*)calloc(lddk * np2, sizeof(f64));
    f64 *x = (f64*)calloc(ldx * n_, sizeof(f64));
    f64 *y = (f64*)calloc(ldy * n_, sizeof(f64));
    f64 *rcond = (f64*)malloc(4 * sizeof(f64));
    i32 *iwork = (i32*)malloc(liwork * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    bool *bwork = (bool*)malloc(2 * n_ * sizeof(bool));

    if (!ak || !bk || !ck || !dk || !x || !y || !rcond || !iwork || !dwork || !bwork) {
        free(ak); free(bk); free(ck); free(dk); free(x); free(y);
        free(rcond); free(iwork); free(dwork); free(bwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);

    i32 info = 0;

    sb10sd(n_, m_, np_, ncon_, nmeas_, a_data, lda, b_data, ldb,
           c_data, ldc, d_data, ldd, ak, ldak, bk, ldbk, ck, ldck,
           dk, lddk, x, ldx, y, ldy, rcond, tol, iwork, dwork,
           ldwork, bwork, &info);

    free(iwork);
    free(dwork);
    free(bwork);

    npy_intp ak_dims[2] = {n_, n_};
    npy_intp bk_dims[2] = {n_, np2};
    npy_intp ck_dims[2] = {m2, n_};
    npy_intp dk_dims[2] = {m2, np2};
    npy_intp x_dims[2] = {n_, n_};
    npy_intp y_dims[2] = {n_, n_};
    npy_intp rcond_dims[1] = {4};

    npy_intp ak_strides[2] = {sizeof(f64), n_ * sizeof(f64)};
    npy_intp bk_strides[2] = {sizeof(f64), n_ * sizeof(f64)};
    npy_intp ck_strides[2] = {sizeof(f64), m2 * sizeof(f64)};
    npy_intp dk_strides[2] = {sizeof(f64), m2 * sizeof(f64)};
    npy_intp x_strides[2] = {sizeof(f64), n_ * sizeof(f64)};
    npy_intp y_strides[2] = {sizeof(f64), n_ * sizeof(f64)};

    PyArrayObject *ak_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, ak_dims, NPY_DOUBLE,
                                                           ak_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    f64 *ak_out = (f64*)PyArray_DATA(ak_array);
    for (i32 j = 0; j < n_; j++) {
        memcpy(ak_out + j * n_, ak + j * ldak, n_ * sizeof(f64));
    }
    free(ak);

    PyArrayObject *bk_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, bk_dims, NPY_DOUBLE,
                                                           bk_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    f64 *bk_out = (f64*)PyArray_DATA(bk_array);
    for (i32 j = 0; j < np2; j++) {
        memcpy(bk_out + j * n_, bk + j * ldbk, n_ * sizeof(f64));
    }
    free(bk);

    PyArrayObject *ck_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, ck_dims, NPY_DOUBLE,
                                                           ck_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    f64 *ck_out = (f64*)PyArray_DATA(ck_array);
    for (i32 j = 0; j < n_; j++) {
        memcpy(ck_out + j * m2, ck + j * ldck, m2 * sizeof(f64));
    }
    free(ck);

    PyArrayObject *dk_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dk_dims, NPY_DOUBLE,
                                                           dk_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    f64 *dk_out = (f64*)PyArray_DATA(dk_array);
    for (i32 j = 0; j < np2; j++) {
        memcpy(dk_out + j * m2, dk + j * lddk, m2 * sizeof(f64));
    }
    free(dk);

    PyArrayObject *x_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, x_dims, NPY_DOUBLE,
                                                          x_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    f64 *x_out = (f64*)PyArray_DATA(x_array);
    for (i32 j = 0; j < n_; j++) {
        memcpy(x_out + j * n_, x + j * ldx, n_ * sizeof(f64));
    }
    free(x);

    PyArrayObject *y_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, y_dims, NPY_DOUBLE,
                                                          y_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    f64 *y_out = (f64*)PyArray_DATA(y_array);
    for (i32 j = 0; j < n_; j++) {
        memcpy(y_out + j * n_, y + j * ldy, n_ * sizeof(f64));
    }
    free(y);

    PyObject *rcond_array = PyArray_New(&PyArray_Type, 1, rcond_dims, NPY_DOUBLE,
                                         NULL, NULL, 0, 0, NULL);
    f64 *rcond_out = (f64*)PyArray_DATA((PyArrayObject*)rcond_array);
    memcpy(rcond_out, rcond, 4 * sizeof(f64));
    free(rcond);

    PyObject *result = Py_BuildValue("OOOOOOOi",
                                      ak_array, bk_array, ck_array, dk_array,
                                      x_array, y_array, rcond_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(ak_array);
    Py_DECREF(bk_array);
    Py_DECREF(ck_array);
    Py_DECREF(dk_array);
    Py_DECREF(x_array);
    Py_DECREF(y_array);

    return result;
}


PyObject* py_sb10td(PyObject* self, PyObject* args, PyObject* kwds) {
    int n, m, np, ncon, nmeas;
    double tol = 0.0;
    PyObject *d_obj, *tu_obj, *ty_obj, *ak_obj, *bk_obj, *ck_obj, *dk_obj;

    static char* kwlist[] = {"n", "m", "np", "ncon", "nmeas",
                             "d", "tu", "ty", "ak", "bk", "ck", "dk",
                             "tol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiOOOOOOO|d", kwlist,
                                      &n, &m, &np, &ncon, &nmeas,
                                      &d_obj, &tu_obj, &ty_obj,
                                      &ak_obj, &bk_obj, &ck_obj, &dk_obj,
                                      &tol)) {
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *tu_array = (PyArrayObject*)PyArray_FROM_OTF(tu_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *ty_array = (PyArrayObject*)PyArray_FROM_OTF(ty_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    PyArrayObject *ak_array = (PyArrayObject*)PyArray_FROM_OTF(ak_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *bk_array = (PyArrayObject*)PyArray_FROM_OTF(bk_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *ck_array = (PyArrayObject*)PyArray_FROM_OTF(ck_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *dk_array = (PyArrayObject*)PyArray_FROM_OTF(dk_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!d_array || !tu_array || !ty_array || !ak_array || !bk_array || !ck_array || !dk_array) {
        Py_XDECREF(d_array);
        Py_XDECREF(tu_array);
        Py_XDECREF(ty_array);
        Py_XDECREF(ak_array);
        Py_XDECREF(bk_array);
        Py_XDECREF(ck_array);
        Py_XDECREF(dk_array);
        return NULL;
    }

    i32 m2 = ncon;
    i32 np2 = nmeas;
    i32 ldd = np > 1 ? np : 1;
    i32 ldtu = m2 > 1 ? m2 : 1;
    i32 ldty = np2 > 1 ? np2 : 1;
    i32 ldak = n > 1 ? n : 1;
    i32 ldbk = n > 1 ? n : 1;
    i32 ldck = m2 > 1 ? m2 : 1;
    i32 lddk = m2 > 1 ? m2 : 1;

    i32 t1 = n * m2;
    i32 t2 = n * np2;
    i32 t3 = m2 * np2;
    i32 t4 = m2 * m2 + 4 * m2;
    i32 ldwork = t1;
    if (t2 > ldwork) ldwork = t2;
    if (t3 > ldwork) ldwork = t3;
    if (t4 > ldwork) ldwork = t4;
    if (ldwork < 1) ldwork = 1;

    i32 liwork = 2 * m2;
    if (liwork < 1) liwork = 1;

    f64* dwork = (f64*)malloc(ldwork * sizeof(f64));
    i32* iwork = (i32*)malloc(liwork * sizeof(i32));

    if (!dwork || !iwork) {
        free(dwork);
        free(iwork);
        Py_DECREF(d_array);
        Py_DECREF(tu_array);
        Py_DECREF(ty_array);
        Py_DECREF(ak_array);
        Py_DECREF(bk_array);
        Py_DECREF(ck_array);
        Py_DECREF(dk_array);
        return PyErr_NoMemory();
    }

    f64 rcond = 0.0;
    i32 info = 0;

    sb10td(n, m, np, ncon, nmeas,
           (f64*)PyArray_DATA(d_array), ldd,
           (f64*)PyArray_DATA(tu_array), ldtu,
           (f64*)PyArray_DATA(ty_array), ldty,
           (f64*)PyArray_DATA(ak_array), ldak,
           (f64*)PyArray_DATA(bk_array), ldbk,
           (f64*)PyArray_DATA(ck_array), ldck,
           (f64*)PyArray_DATA(dk_array), lddk,
           &rcond, tol, iwork, dwork, ldwork, &info);

    free(dwork);
    free(iwork);

    PyArray_ResolveWritebackIfCopy(ak_array);
    PyArray_ResolveWritebackIfCopy(bk_array);
    PyArray_ResolveWritebackIfCopy(ck_array);
    PyArray_ResolveWritebackIfCopy(dk_array);

    PyObject* result = Py_BuildValue("(OOOOdi)",
                                      ak_array, bk_array, ck_array, dk_array, rcond, info);

    Py_DECREF(d_array);
    Py_DECREF(tu_array);
    Py_DECREF(ty_array);
    Py_DECREF(ak_array);
    Py_DECREF(bk_array);
    Py_DECREF(ck_array);
    Py_DECREF(dk_array);

    return result;
}


PyObject* py_sb10zd(PyObject* self, PyObject* args) {
    (void)self;
    int n, m, np_arg;
    double factor, tol;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "iiiOOOOdd", &n, &m, &np_arg, &a_obj, &b_obj,
                          &c_obj, &d_obj, &factor, &tol)) {
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

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (!c_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (!d_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = np_arg > 0 ? np_arg : 1;
    i32 ldd = np_arg > 0 ? np_arg : 1;
    i32 ldak = n > 0 ? n : 1;
    i32 ldbk = n > 0 ? n : 1;
    i32 ldck = m > 0 ? m : 1;
    i32 lddk = m > 0 ? m : 1;

    npy_intp dims_ak[2] = {n, n};
    npy_intp dims_bk[2] = {n, np_arg};
    npy_intp dims_ck[2] = {m, n};
    npy_intp dims_dk[2] = {m, np_arg};
    npy_intp dims_rcond[1] = {6};

    f64* ak = (f64*)calloc(ldak * n, sizeof(f64));
    f64* bk = (f64*)calloc(ldbk * np_arg, sizeof(f64));
    f64* ck = (f64*)calloc(ldck * n, sizeof(f64));
    f64* dk = (f64*)calloc(lddk * np_arg, sizeof(f64));

    if ((!ak && n > 0) || (!bk && n > 0 && np_arg > 0) ||
        (!ck && m > 0 && n > 0) || (!dk && m > 0 && np_arg > 0)) {
        free(ak);
        free(bk);
        free(ck);
        free(dk);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return PyErr_NoMemory();
    }

    i32 tmp1 = 14*n + 23;
    i32 tmp2 = 16*n;
    i32 tmp3 = 2*m - 1;
    i32 tmp4 = 2*np_arg - 1;
    i32 maxval = tmp1;
    if (tmp2 > maxval) maxval = tmp2;
    if (tmp3 > maxval) maxval = tmp3;
    if (tmp4 > maxval) maxval = tmp4;
    if (maxval < 1) maxval = 1;

    i32 ldwork = 16*n*n + 5*m*m + 7*np_arg*np_arg + 6*m*n + 7*m*np_arg +
                 7*n*np_arg + 6*n + 2*(m + np_arg) + maxval;
    if (ldwork < 1) ldwork = 1;

    i32 iwork_size = 2 * ((n > m + np_arg) ? n : (m + np_arg));
    if (iwork_size < 2) iwork_size = 2;

    i32 bwork_size = 2 * n;
    if (bwork_size < 2) bwork_size = 2;

    f64* dwork = (f64*)malloc(ldwork * sizeof(f64));
    i32* iwork = (i32*)malloc(iwork_size * sizeof(i32));
    bool* bwork = (bool*)malloc(bwork_size * sizeof(bool));

    if (!dwork || !iwork || !bwork) {
        free(ak);
        free(bk);
        free(ck);
        free(dk);
        free(dwork);
        free(iwork);
        free(bwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return PyErr_NoMemory();
    }

    f64 rcond[6] = {0};
    i32 info = 0;

    sb10zd((i32)n, (i32)m, (i32)np_arg,
           (const f64*)PyArray_DATA(a_array), lda,
           (const f64*)PyArray_DATA(b_array), ldb,
           (const f64*)PyArray_DATA(c_array), ldc,
           (const f64*)PyArray_DATA(d_array), ldd,
           (f64)factor,
           ak, ldak, bk, ldbk, ck, ldck, dk, lddk,
           rcond, (f64)tol, iwork, dwork, ldwork, bwork, &info);

    free(dwork);
    free(iwork);
    free(bwork);

    npy_intp strides_ak[2] = {sizeof(f64), n * sizeof(f64)};
    npy_intp strides_bk[2] = {sizeof(f64), n * sizeof(f64)};
    npy_intp strides_ck[2] = {sizeof(f64), m * sizeof(f64)};
    npy_intp strides_dk[2] = {sizeof(f64), m * sizeof(f64)};

    PyObject* ak_array = PyArray_New(&PyArray_Type, 2, dims_ak, NPY_DOUBLE,
                                      strides_ak, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject* bk_array = PyArray_New(&PyArray_Type, 2, dims_bk, NPY_DOUBLE,
                                      strides_bk, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject* ck_array = PyArray_New(&PyArray_Type, 2, dims_ck, NPY_DOUBLE,
                                      strides_ck, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject* dk_array = PyArray_New(&PyArray_Type, 2, dims_dk, NPY_DOUBLE,
                                      strides_dk, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (!ak_array || !bk_array || !ck_array || !dk_array) {
        Py_XDECREF(ak_array);
        Py_XDECREF(bk_array);
        Py_XDECREF(ck_array);
        Py_XDECREF(dk_array);
        free(ak);
        free(bk);
        free(ck);
        free(dk);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }

    f64* ak_out = (f64*)PyArray_DATA((PyArrayObject*)ak_array);
    for (i32 j = 0; j < n; j++) {
        memcpy(ak_out + j * n, ak + j * ldak, n * sizeof(f64));
    }
    free(ak);

    f64* bk_out = (f64*)PyArray_DATA((PyArrayObject*)bk_array);
    for (i32 j = 0; j < np_arg; j++) {
        memcpy(bk_out + j * n, bk + j * ldbk, n * sizeof(f64));
    }
    free(bk);

    f64* ck_out = (f64*)PyArray_DATA((PyArrayObject*)ck_array);
    for (i32 j = 0; j < n; j++) {
        memcpy(ck_out + j * m, ck + j * ldck, m * sizeof(f64));
    }
    free(ck);

    f64* dk_out = (f64*)PyArray_DATA((PyArrayObject*)dk_array);
    for (i32 j = 0; j < np_arg; j++) {
        memcpy(dk_out + j * m, dk + j * lddk, m * sizeof(f64));
    }
    free(dk);

    PyObject* rcond_array = PyArray_SimpleNew(1, dims_rcond, NPY_DOUBLE);
    if (!rcond_array) {
        Py_DECREF(ak_array);
        Py_DECREF(bk_array);
        Py_DECREF(ck_array);
        Py_DECREF(dk_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)rcond_array), rcond, 6 * sizeof(f64));

    PyObject* result = Py_BuildValue("(OOOOOi)", ak_array, bk_array, ck_array,
                                      dk_array, rcond_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(ak_array);
    Py_DECREF(bk_array);
    Py_DECREF(ck_array);
    Py_DECREF(dk_array);
    Py_DECREF(rcond_array);

    return result;
}

PyObject* py_sb16bd(PyObject* self, PyObject* args) {
    const char *dico_str, *jobd_str, *jobmr_str, *jobcf_str, *equil_str, *ordsel_str;
    int n, m, p, ncr;
    double tol1, tol2;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *f_obj, *g_obj;

    if (!PyArg_ParseTuple(args, "ssssssiiiiOOOOOOdd",
                          &dico_str, &jobd_str, &jobmr_str, &jobcf_str,
                          &equil_str, &ordsel_str, &n, &m, &p, &ncr,
                          &a_obj, &b_obj, &c_obj, &d_obj, &f_obj, &g_obj,
                          &tol1, &tol2)) {
        return NULL;
    }

    if (dico_str[0] != 'C' && dico_str[0] != 'c' &&
        dico_str[0] != 'D' && dico_str[0] != 'd') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }

    if (jobd_str[0] != 'D' && jobd_str[0] != 'd' &&
        jobd_str[0] != 'Z' && jobd_str[0] != 'z') {
        PyErr_SetString(PyExc_ValueError, "JOBD must be 'D' or 'Z'");
        return NULL;
    }

    if (jobmr_str[0] != 'B' && jobmr_str[0] != 'b' &&
        jobmr_str[0] != 'F' && jobmr_str[0] != 'f' &&
        jobmr_str[0] != 'S' && jobmr_str[0] != 's' &&
        jobmr_str[0] != 'P' && jobmr_str[0] != 'p') {
        PyErr_SetString(PyExc_ValueError, "JOBMR must be 'B', 'F', 'S', or 'P'");
        return NULL;
    }

    if (jobcf_str[0] != 'L' && jobcf_str[0] != 'l' &&
        jobcf_str[0] != 'R' && jobcf_str[0] != 'r') {
        PyErr_SetString(PyExc_ValueError, "JOBCF must be 'L' or 'R'");
        return NULL;
    }

    if (equil_str[0] != 'S' && equil_str[0] != 's' &&
        equil_str[0] != 'N' && equil_str[0] != 'n') {
        PyErr_SetString(PyExc_ValueError, "EQUIL must be 'S' or 'N'");
        return NULL;
    }

    if (ordsel_str[0] != 'F' && ordsel_str[0] != 'f' &&
        ordsel_str[0] != 'A' && ordsel_str[0] != 'a') {
        PyErr_SetString(PyExc_ValueError, "ORDSEL must be 'F' or 'A'");
        return NULL;
    }

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "N must be >= 0");
        return NULL;
    }
    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "M must be >= 0");
        return NULL;
    }
    if (p < 0) {
        PyErr_SetString(PyExc_ValueError, "P must be >= 0");
        return NULL;
    }

    bool fixord = (ordsel_str[0] == 'F' || ordsel_str[0] == 'f');
    if (fixord && (ncr < 0 || ncr > n)) {
        PyErr_SetString(PyExc_ValueError, "NCR must be >= 0 and <= N when ORDSEL='F'");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!b_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!c_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    bool withd = (jobd_str[0] == 'D' || jobd_str[0] == 'd');
    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!d_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    PyArrayObject *f_array = (PyArrayObject*)PyArray_FROM_OTF(
        f_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!f_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }

    PyArrayObject *g_array = (PyArrayObject*)PyArray_FROM_OTF(
        g_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!g_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(f_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldd = withd ? (p > 0 ? p : 1) : 1;
    i32 ldf = m > 0 ? m : 1;
    i32 ldg = n > 0 ? n : 1;
    i32 lddc = m > 0 ? m : 1;

    i32 mp = m + p;
    i32 max_nmp = n;
    if (mp > max_nmp) max_nmp = mp;
    i32 lwr = n * (2 * n + max_nmp + 5) + (n * (n + 1)) / 2;
    if (lwr < 1) lwr = 1;

    bool left = (jobcf_str[0] == 'L' || jobcf_str[0] == 'l');
    i32 ldwork;
    if (fixord && ncr == n) {
        ldwork = p * n > 0 ? p * n : 1;
    } else if (left) {
        i32 temp = lwr > 4 * m ? lwr : 4 * m;
        ldwork = (n + m) * mp + temp;
    } else {
        i32 temp = lwr > 4 * p ? lwr : 4 * p;
        ldwork = (n + p) * mp + temp;
    }
    if (ldwork < 1) ldwork = 1;

    bool spa = (jobmr_str[0] == 'S' || jobmr_str[0] == 's' ||
                jobmr_str[0] == 'P' || jobmr_str[0] == 'p');
    bool bfree = (jobmr_str[0] == 'F' || jobmr_str[0] == 'f' ||
                  jobmr_str[0] == 'P' || jobmr_str[0] == 'p');
    i32 pm = 0;
    if (bfree && !spa) pm = n;
    if (spa) pm = 2 * n > 1 ? 2 * n : 1;
    i32 liwork = left ? (pm > m ? pm : m) : (pm > p ? pm : p);
    if (liwork < 1) liwork = 1;

    i32 dc_size = lddc * p;
    if (dc_size < 1) dc_size = 1;
    f64 *dc_data = (f64*)calloc(dc_size, sizeof(f64));
    f64 *hsv = (f64*)malloc(n > 0 ? n * sizeof(f64) : sizeof(f64));
    i32 *iwork = (i32*)malloc(liwork * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!dc_data || !hsv || !iwork || !dwork) {
        free(dc_data);
        free(hsv);
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(f_array);
        Py_DECREF(g_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *f_data = (f64*)PyArray_DATA(f_array);
    f64 *g_data = (f64*)PyArray_DATA(g_array);

    i32 ncr_out = ncr;
    i32 iwarn = 0;
    i32 info = 0;

    sb16bd(dico_str, jobd_str, jobmr_str, jobcf_str, equil_str, ordsel_str,
           n, m, p, &ncr_out, a_data, lda, b_data, ldb, c_data, ldc,
           d_data, ldd, f_data, ldf, g_data, ldg, dc_data, lddc, hsv,
           tol1, tol2, iwork, dwork, ldwork, &iwarn, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(f_array);
    PyArray_ResolveWritebackIfCopy(g_array);

    free(iwork);
    free(dwork);

    npy_intp ac_dims[2] = {ncr_out, ncr_out};
    npy_intp bc_dims[2] = {ncr_out, p};
    npy_intp cc_dims[2] = {m, ncr_out};
    npy_intp dc_out_dims[2] = {m, p};
    npy_intp hsv_dims[1] = {n > 0 ? n : 1};

    PyArrayObject *ac_array = (PyArrayObject*)PyArray_SimpleNew(2, ac_dims, NPY_DOUBLE);
    PyArrayObject *bc_array = (PyArrayObject*)PyArray_SimpleNew(2, bc_dims, NPY_DOUBLE);
    PyArrayObject *cc_array = (PyArrayObject*)PyArray_SimpleNew(2, cc_dims, NPY_DOUBLE);
    PyArrayObject *dc_out_array = (PyArrayObject*)PyArray_SimpleNew(2, dc_out_dims, NPY_DOUBLE);
    PyArrayObject *hsv_array = (PyArrayObject*)PyArray_SimpleNew(1, hsv_dims, NPY_DOUBLE);

    if (!ac_array || !bc_array || !cc_array || !dc_out_array || !hsv_array) {
        free(dc_data);
        free(hsv);
        Py_XDECREF(ac_array);
        Py_XDECREF(bc_array);
        Py_XDECREF(cc_array);
        Py_XDECREF(dc_out_array);
        Py_XDECREF(hsv_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(f_array);
        Py_DECREF(g_array);
        PyErr_NoMemory();
        return NULL;
    }

    for (i32 j = 0; j < ncr_out; j++) {
        for (i32 i = 0; i < ncr_out; i++) {
            ((f64*)PyArray_DATA(ac_array))[i * ncr_out + j] = a_data[j * lda + i];
        }
    }

    for (i32 j = 0; j < p; j++) {
        for (i32 i = 0; i < ncr_out; i++) {
            ((f64*)PyArray_DATA(bc_array))[i * p + j] = g_data[j * ldg + i];
        }
    }

    for (i32 j = 0; j < ncr_out; j++) {
        for (i32 i = 0; i < m; i++) {
            ((f64*)PyArray_DATA(cc_array))[i * ncr_out + j] = f_data[j * ldf + i];
        }
    }

    for (i32 j = 0; j < p; j++) {
        for (i32 i = 0; i < m; i++) {
            ((f64*)PyArray_DATA(dc_out_array))[i * p + j] = dc_data[j * lddc + i];
        }
    }

    memcpy(PyArray_DATA(hsv_array), hsv, (n > 0 ? n : 1) * sizeof(f64));

    free(dc_data);
    free(hsv);

    PyObject *result = Py_BuildValue("OOOOOiii", ac_array, bc_array, cc_array,
                                     dc_out_array, hsv_array, ncr_out, iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(f_array);
    Py_DECREF(g_array);
    Py_DECREF(ac_array);
    Py_DECREF(bc_array);
    Py_DECREF(cc_array);
    Py_DECREF(dc_out_array);
    Py_DECREF(hsv_array);

    return result;
}

PyObject* py_sb16cy(PyObject* self, PyObject* args) {
    const char *dico_str, *jobcf_str;
    int n, m, p;
    PyObject *a_obj, *b_obj, *c_obj, *f_obj, *g_obj;

    if (!PyArg_ParseTuple(args, "ssiiiOOOOO",
                          &dico_str, &jobcf_str, &n, &m, &p,
                          &a_obj, &b_obj, &c_obj, &f_obj, &g_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!b_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!c_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *f_array = (PyArrayObject*)PyArray_FROM_OTF(
        f_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!f_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    PyArrayObject *g_array = (PyArrayObject*)PyArray_FROM_OTF(
        g_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!g_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(f_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldf = m > 0 ? m : 1;
    i32 ldg = n > 0 ? n : 1;
    i32 lds = n > 0 ? n : 1;
    i32 ldr = n > 0 ? n : 1;

    bool leftw = (jobcf_str[0] == 'L' || jobcf_str[0] == 'l');
    i32 mp = leftw ? m : p;
    i32 max_n_mp = n > mp ? n : mp;
    i32 min_n_mp = n < mp ? n : mp;
    i32 ldwork = n * (n + max_n_mp + min_n_mp + 6);
    if (ldwork < 1) ldwork = 1;

    f64 *s_data = NULL;
    f64 *r_data = NULL;
    f64 *dwork = NULL;

    if (n > 0) {
        s_data = (f64*)calloc(lds * n, sizeof(f64));
        r_data = (f64*)calloc(ldr * n, sizeof(f64));
        dwork = (f64*)malloc(ldwork * sizeof(f64));

        if (!s_data || !r_data || !dwork) {
            free(s_data);
            free(r_data);
            free(dwork);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            Py_DECREF(f_array);
            Py_DECREF(g_array);
            PyErr_NoMemory();
            return NULL;
        }
    }

    f64 scalec = 1.0, scaleo = 1.0;
    i32 info = 0;

    sb16cy(dico_str, jobcf_str, n, m, p,
           (f64*)PyArray_DATA(a_array), lda,
           (f64*)PyArray_DATA(b_array), ldb,
           (f64*)PyArray_DATA(c_array), ldc,
           (f64*)PyArray_DATA(f_array), ldf,
           (f64*)PyArray_DATA(g_array), ldg,
           &scalec, &scaleo,
           s_data, lds,
           r_data, ldr,
           dwork, ldwork, &info);

    free(dwork);

    npy_intp s_dims[2] = {n, n};
    npy_intp r_dims[2] = {n, n};

    PyArrayObject *s_array = NULL;
    PyArrayObject *r_array = NULL;

    if (n > 0) {
        npy_intp s_strides[2] = {sizeof(f64), n * sizeof(f64)};
        s_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, s_dims, NPY_DOUBLE,
                                               s_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!s_array) {
            free(s_data);
            free(r_data);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            Py_DECREF(f_array);
            Py_DECREF(g_array);
            PyErr_NoMemory();
            return NULL;
        }
        f64 *s_out = (f64*)PyArray_DATA(s_array);
        for (i32 j = 0; j < n; j++) {
            memcpy(s_out + j * n, s_data + j * lds, n * sizeof(f64));
        }
        free(s_data);

        npy_intp r_strides[2] = {sizeof(f64), n * sizeof(f64)};
        r_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, r_dims, NPY_DOUBLE,
                                               r_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!r_array) {
            free(r_data);
            Py_DECREF(s_array);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            Py_DECREF(f_array);
            Py_DECREF(g_array);
            PyErr_NoMemory();
            return NULL;
        }
        f64 *r_out = (f64*)PyArray_DATA(r_array);
        for (i32 j = 0; j < n; j++) {
            memcpy(r_out + j * n, r_data + j * ldr, n * sizeof(f64));
        }
        free(r_data);
    } else {
        s_array = (PyArrayObject*)PyArray_SimpleNew(2, s_dims, NPY_DOUBLE);
        r_array = (PyArrayObject*)PyArray_SimpleNew(2, r_dims, NPY_DOUBLE);
        if (!s_array || !r_array) {
            Py_XDECREF(s_array);
            Py_XDECREF(r_array);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            Py_DECREF(f_array);
            Py_DECREF(g_array);
            PyErr_NoMemory();
            return NULL;
        }
    }

    PyObject *result = Py_BuildValue("OOddi", s_array, r_array, scalec, scaleo, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(f_array);
    Py_DECREF(g_array);
    Py_DECREF(s_array);
    Py_DECREF(r_array);

    return result;
}


PyObject* py_sb10ed(PyObject* self, PyObject* args, PyObject* kwds) {
    static char *kwlist[] = {"n", "m", "np", "ncon", "nmeas", "a", "b", "c", "d", "tol", NULL};
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    int n, m, np_val, ncon, nmeas;
    double tol = 0.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiOOOO|d", kwlist,
                                      &n, &m, &np_val, &ncon, &nmeas,
                                      &a_obj, &b_obj, &c_obj, &d_obj, &tol)) {
        return NULL;
    }

    i32 n_ = (i32)n;
    i32 m_ = (i32)m;
    i32 np_ = (i32)np_val;
    i32 ncon_ = (i32)ncon;
    i32 nmeas_ = (i32)nmeas;

    i32 m1 = m_ - ncon_;
    i32 m2 = ncon_;
    i32 np1 = np_ - nmeas_;
    i32 np2 = nmeas_;

    if (n_ < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }
    if (m_ < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be non-negative");
        return NULL;
    }
    if (np_ < 0) {
        PyErr_SetString(PyExc_ValueError, "np must be non-negative");
        return NULL;
    }
    if (ncon_ < 0 || m1 < 0 || m2 > np1) {
        PyErr_SetString(PyExc_ValueError, "ncon must satisfy 0 <= ncon <= m and ncon <= np-nmeas");
        return NULL;
    }
    if (nmeas_ < 0 || np1 < 0 || np2 > m1) {
        PyErr_SetString(PyExc_ValueError, "nmeas must satisfy 0 <= nmeas <= np and nmeas <= m-ncon");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!b_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!c_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!d_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    i32 lda = n_ > 1 ? n_ : 1;
    i32 ldb = n_ > 1 ? n_ : 1;
    i32 ldc = np_ > 1 ? np_ : 1;
    i32 ldd = np_ > 1 ? np_ : 1;
    i32 ldak = n_ > 1 ? n_ : 1;
    i32 ldbk = n_ > 1 ? n_ : 1;
    i32 ldck = m2 > 1 ? m2 : 1;
    i32 lddk = m2 > 1 ? m2 : 1;

    i32 lw1 = (n_ + np1 + 1) * (n_ + m2) +
              (3 * (n_ + m2) + n_ + np1 > 5 * (n_ + m2) ?
               3 * (n_ + m2) + n_ + np1 : 5 * (n_ + m2));
    i32 lw2 = (n_ + np2) * (n_ + m1 + 1) +
              (3 * (n_ + np2) + n_ + m1 > 5 * (n_ + np2) ?
               3 * (n_ + np2) + n_ + m1 : 5 * (n_ + np2));
    i32 tmp_lw3 = np1 * (n_ > m1 ? n_ : m1);
    i32 tmp2_lw3 = 3 * m2 + np1;
    i32 tmp3_lw3 = 5 * m2;
    i32 max_lw3 = tmp_lw3 > tmp2_lw3 ? tmp_lw3 : tmp2_lw3;
    max_lw3 = max_lw3 > tmp3_lw3 ? max_lw3 : tmp3_lw3;
    i32 lw3 = m2 + np1 * np1 + max_lw3;

    i32 tmp_lw4 = (n_ > np1 ? n_ : np1) * m1;
    i32 tmp2_lw4 = 3 * np2 + m1;
    i32 tmp3_lw4 = 5 * np2;
    i32 max_lw4 = tmp_lw4 > tmp2_lw4 ? tmp_lw4 : tmp2_lw4;
    max_lw4 = max_lw4 > tmp3_lw4 ? max_lw4 : tmp3_lw4;
    i32 lw4 = np2 + m1 * m1 + max_lw4;

    i32 tmp_lw5a = 14 * n_ + 23 > 16 * n_ ? 14 * n_ + 23 : 16 * n_;
    i32 tmp_lw5b = 14 * n_ * n_ + 6 * n_ + tmp_lw5a;
    i32 tmp_lw5c = m2 * (n_ + m2 + (3 > m1 ? 3 : m1));
    i32 tmp_lw5d = np2 * (n_ + np2 + 3);
    i32 max_lw5 = 1 > tmp_lw5b ? 1 : tmp_lw5b;
    max_lw5 = max_lw5 > tmp_lw5c ? max_lw5 : tmp_lw5c;
    max_lw5 = max_lw5 > tmp_lw5d ? max_lw5 : tmp_lw5d;
    i32 lw5 = 2 * n_ * n_ + max_lw5;

    i32 tmp_lw6a = n_ * m2;
    i32 tmp_lw6b = n_ * np2;
    i32 tmp_lw6c = m2 * np2;
    i32 tmp_lw6d = m2 * m2 + 4 * m2;
    i32 lw6 = tmp_lw6a > tmp_lw6b ? tmp_lw6a : tmp_lw6b;
    lw6 = lw6 > tmp_lw6c ? lw6 : tmp_lw6c;
    lw6 = lw6 > tmp_lw6d ? lw6 : tmp_lw6d;

    i32 minwrk = n_ * m_ + np_ * (n_ + m_) + m2 * m2 + np2 * np2;
    i32 maxlw = 1 > lw1 ? 1 : lw1;
    maxlw = maxlw > lw2 ? maxlw : lw2;
    maxlw = maxlw > lw3 ? maxlw : lw3;
    maxlw = maxlw > lw4 ? maxlw : lw4;
    maxlw = maxlw > lw5 ? maxlw : lw5;
    maxlw = maxlw > lw6 ? maxlw : lw6;
    i32 ldwork = minwrk + maxlw;
    if (ldwork < 1) ldwork = 1;

    i32 iw1 = 2 * m2;
    i32 iw2 = 2 * n_;
    i32 iw3 = n_ * n_;
    i32 iw4 = np2;
    i32 liwork = iw1 > iw2 ? iw1 : iw2;
    liwork = liwork > iw3 ? liwork : iw3;
    liwork = liwork > iw4 ? liwork : iw4;
    if (liwork < 1) liwork = 1;

    i32 bk_cols = np2 > 0 ? np2 : 1;
    i32 dk_cols = np2 > 0 ? np2 : 1;

    f64 *ak = (f64*)calloc(ldak * n_, sizeof(f64));
    f64 *bk = (f64*)calloc(ldbk * bk_cols, sizeof(f64));
    f64 *ck = (f64*)calloc(ldck * n_, sizeof(f64));
    f64 *dk = (f64*)calloc(lddk * dk_cols, sizeof(f64));
    f64 *rcond = (f64*)malloc(7 * sizeof(f64));
    i32 *iwork = (i32*)malloc(liwork * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    bool *bwork = (bool*)malloc(2 * n_ * sizeof(bool));

    if (!ak || !bk || !ck || !dk || !rcond || !iwork || !dwork || !bwork) {
        free(ak); free(bk); free(ck); free(dk);
        free(rcond); free(iwork); free(dwork); free(bwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);

    i32 info = 0;

    sb10ed(n_, m_, np_, ncon_, nmeas_, a_data, lda, b_data, ldb,
           c_data, ldc, d_data, ldd, ak, ldak, bk, ldbk, ck, ldck,
           dk, lddk, rcond, tol, iwork, dwork, ldwork, bwork, &info);

    free(iwork);
    free(dwork);
    free(bwork);

    PyArray_ResolveWritebackIfCopy(a_array);

    npy_intp ak_dims[2] = {n_, n_};
    npy_intp bk_dims[2] = {n_, np2};
    npy_intp ck_dims[2] = {m2, n_};
    npy_intp dk_dims[2] = {m2, np2};
    npy_intp rcond_dims[1] = {7};

    npy_intp ak_strides[2] = {sizeof(f64), n_ * sizeof(f64)};
    npy_intp bk_strides[2] = {sizeof(f64), n_ * sizeof(f64)};
    npy_intp ck_strides[2] = {sizeof(f64), m2 * sizeof(f64)};
    npy_intp dk_strides[2] = {sizeof(f64), m2 * sizeof(f64)};

    PyArrayObject *ak_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, ak_dims, NPY_DOUBLE,
                                                           ak_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    f64 *ak_out = (f64*)PyArray_DATA(ak_array);
    for (i32 j = 0; j < n_; j++) {
        memcpy(ak_out + j * n_, ak + j * ldak, n_ * sizeof(f64));
    }
    free(ak);

    PyArrayObject *bk_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, bk_dims, NPY_DOUBLE,
                                                           bk_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    f64 *bk_out = (f64*)PyArray_DATA(bk_array);
    for (i32 j = 0; j < np2; j++) {
        memcpy(bk_out + j * n_, bk + j * ldbk, n_ * sizeof(f64));
    }
    free(bk);

    PyArrayObject *ck_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, ck_dims, NPY_DOUBLE,
                                                           ck_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    f64 *ck_out = (f64*)PyArray_DATA(ck_array);
    for (i32 j = 0; j < n_; j++) {
        memcpy(ck_out + j * m2, ck + j * ldck, m2 * sizeof(f64));
    }
    free(ck);

    PyArrayObject *dk_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dk_dims, NPY_DOUBLE,
                                                           dk_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    f64 *dk_out = (f64*)PyArray_DATA(dk_array);
    for (i32 j = 0; j < np2; j++) {
        memcpy(dk_out + j * m2, dk + j * lddk, m2 * sizeof(f64));
    }
    free(dk);

    PyObject *rcond_array = PyArray_New(&PyArray_Type, 1, rcond_dims, NPY_DOUBLE,
                                         NULL, NULL, 0, 0, NULL);
    f64 *rcond_out = (f64*)PyArray_DATA((PyArrayObject*)rcond_array);
    memcpy(rcond_out, rcond, 7 * sizeof(f64));
    free(rcond);

    PyObject *result = Py_BuildValue("OOOOOi",
                                      ak_array, bk_array, ck_array, dk_array,
                                      rcond_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(ak_array);
    Py_DECREF(bk_array);
    Py_DECREF(ck_array);
    Py_DECREF(dk_array);

    return result;
}

PyObject* py_sb16ad(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {
        "dico", "jobc", "jobo", "jobmr", "weight", "equil", "ordsel",
        "n", "m", "p", "nc", "ncr", "alpha",
        "a", "b", "c", "d", "ac", "bc", "cc", "dc",
        "tol1", "tol2", NULL
    };

    const char *dico, *jobc, *jobo, *jobmr, *weight, *equil, *ordsel;
    int n_, m_, p_, nc_, ncr_;
    double alpha, tol1, tol2;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    PyObject *ac_obj, *bc_obj, *cc_obj, *dc_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssssssiiiiidOOOOOOOOdd", kwlist,
                                     &dico, &jobc, &jobo, &jobmr, &weight, &equil, &ordsel,
                                     &n_, &m_, &p_, &nc_, &ncr_, &alpha,
                                     &a_obj, &b_obj, &c_obj, &d_obj,
                                     &ac_obj, &bc_obj, &cc_obj, &dc_obj,
                                     &tol1, &tol2)) {
        return NULL;
    }

    if ((dico[0] != 'C' && dico[0] != 'c' && dico[0] != 'D' && dico[0] != 'd') ||
        (jobc[0] != 'S' && jobc[0] != 's' && jobc[0] != 'E' && jobc[0] != 'e') ||
        (jobo[0] != 'S' && jobo[0] != 's' && jobo[0] != 'E' && jobo[0] != 'e') ||
        (jobmr[0] != 'B' && jobmr[0] != 'b' && jobmr[0] != 'F' && jobmr[0] != 'f' &&
         jobmr[0] != 'S' && jobmr[0] != 's' && jobmr[0] != 'P' && jobmr[0] != 'p') ||
        (weight[0] != 'N' && weight[0] != 'n' && weight[0] != 'O' && weight[0] != 'o' &&
         weight[0] != 'I' && weight[0] != 'i' && weight[0] != 'P' && weight[0] != 'p') ||
        (equil[0] != 'S' && equil[0] != 's' && equil[0] != 'N' && equil[0] != 'n') ||
        (ordsel[0] != 'F' && ordsel[0] != 'f' && ordsel[0] != 'A' && ordsel[0] != 'a')) {
        PyErr_SetString(PyExc_ValueError, "Invalid mode parameter");
        return NULL;
    }

    if (n_ < 0 || m_ < 0 || p_ < 0 || nc_ < 0) {
        PyErr_SetString(PyExc_ValueError, "Dimensions must be non-negative");
        return NULL;
    }

    i32 n = n_, m = m_, p = p_, nc = nc_, ncr = ncr_;
    i32 lda = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldc = p > 1 ? p : 1;
    i32 ldd = p > 1 ? p : 1;
    i32 ldac = nc > 1 ? nc : 1;
    i32 ldbc = nc > 1 ? nc : 1;
    i32 ldcc = m > 1 ? m : 1;
    i32 lddc = m > 1 ? m : 1;

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *ac_array = (PyArrayObject*)PyArray_FROM_OTF(
        ac_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *bc_array = (PyArrayObject*)PyArray_FROM_OTF(
        bc_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *cc_array = (PyArrayObject*)PyArray_FROM_OTF(
        cc_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *dc_array = (PyArrayObject*)PyArray_FROM_OTF(
        dc_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!a_array || !b_array || !c_array || !d_array ||
        !ac_array || !bc_array || !cc_array || !dc_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        Py_XDECREF(ac_array);
        Py_XDECREF(bc_array);
        Py_XDECREF(cc_array);
        Py_XDECREF(dc_array);
        return NULL;
    }

    bool istab = (weight[0] == 'I' || weight[0] == 'i');
    bool ostab = (weight[0] == 'O' || weight[0] == 'o');
    bool perf = (weight[0] == 'P' || weight[0] == 'p');
    bool leftw = ostab || perf;
    bool rightw = istab || perf;
    bool frwght = leftw || rightw;

    i32 nnc = n + nc;
    i32 mp = m + p;
    i32 lw;

    if (frwght) {
        i32 max_nnc_m_p = nnc;
        if (m > max_nnc_m_p) max_nnc_m_p = m;
        if (p > max_nnc_m_p) max_nnc_m_p = p;
        i32 lw1 = nnc * (nnc + max_nnc_m_p + 7);
        i32 lw2 = mp * (mp + 4);
        lw = nnc * (nnc + 2 * mp) + (lw1 > lw2 ? lw1 : lw2);
    } else {
        i32 max_m_p = m > p ? m : p;
        lw = nc * (max_m_p + 5);
        if (equil[0] == 'S' || equil[0] == 's') {
            if (n > lw) lw = n;
        }
    }
    i32 lsqred = 2 * nc * nc + 5 * nc;
    if (lsqred < 1) lsqred = 1;
    i32 ldwork = 2 * nc * nc + (lw > lsqred ? lw : lsqred);
    if (ldwork < 1) ldwork = 1;

    i32 liwrk1 = 0;
    if (jobmr[0] == 'F' || jobmr[0] == 'f') {
        liwrk1 = nc;
    } else if (jobmr[0] == 'S' || jobmr[0] == 's' || jobmr[0] == 'P' || jobmr[0] == 'p') {
        liwrk1 = 2 * nc;
    }
    i32 liwrk2 = 0;
    if (frwght) {
        liwrk2 = 2 * mp;
    }
    i32 liwork = liwrk1 > liwrk2 ? liwrk1 : liwrk2;
    if (liwork < 1) liwork = 1;

    f64 *hsvc = (f64*)calloc(nc > 1 ? nc : 1, sizeof(f64));
    i32 *iwork = (i32*)malloc(liwork * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!hsvc || !iwork || !dwork) {
        free(hsvc);
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(ac_array);
        Py_DECREF(bc_array);
        Py_DECREF(cc_array);
        Py_DECREF(dc_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *ac_data = (f64*)PyArray_DATA(ac_array);
    f64 *bc_data = (f64*)PyArray_DATA(bc_array);
    f64 *cc_data = (f64*)PyArray_DATA(cc_array);
    f64 *dc_data = (f64*)PyArray_DATA(dc_array);

    i32 ncs = 0;
    i32 iwarn = 0;
    i32 info = 0;

    sb16ad(dico, jobc, jobo, jobmr, weight, equil, ordsel,
           n, m, p, nc, &ncr, alpha,
           a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           ac_data, ldac, bc_data, ldbc, cc_data, ldcc, dc_data, lddc,
           &ncs, hsvc, tol1, tol2, iwork, dwork, ldwork, &iwarn, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(ac_array);
    PyArray_ResolveWritebackIfCopy(bc_array);
    PyArray_ResolveWritebackIfCopy(cc_array);
    PyArray_ResolveWritebackIfCopy(dc_array);

    npy_intp hsvc_dims[1] = {nc > 1 ? nc : 1};
    PyObject *hsvc_array = PyArray_SimpleNew(1, hsvc_dims, NPY_DOUBLE);
    memcpy(PyArray_DATA((PyArrayObject*)hsvc_array), hsvc, (nc > 1 ? nc : 1) * sizeof(f64));
    free(hsvc);

    PyObject *result = Py_BuildValue("OOOOiiOii",
                                      ac_array, bc_array, cc_array, dc_array,
                                      ncr, ncs, hsvc_array,
                                      iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(ac_array);
    Py_DECREF(bc_array);
    Py_DECREF(cc_array);
    Py_DECREF(dc_array);
    Py_DECREF(hsvc_array);

    return result;
}

/* Python wrapper for sb16cd */
PyObject* py_sb16cd(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *dico, *jobd, *jobmr, *jobcf, *ordsel;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *f_obj, *g_obj;
    int ncr_in;
    double tol;

    static char* kwlist[] = {"dico", "jobd", "jobmr", "jobcf", "ordsel",
                             "a", "b", "c", "d", "f", "g", "ncr", "tol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssssOOOOOOid", kwlist,
            &dico, &jobd, &jobmr, &jobcf, &ordsel,
            &a_obj, &b_obj, &c_obj, &d_obj, &f_obj, &g_obj,
            &ncr_in, &tol)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                              NPY_ARRAY_FARRAY);
    PyArrayObject *f_array = (PyArrayObject*)PyArray_FROM_OTF(f_obj, NPY_DOUBLE,
                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *g_array = (PyArrayObject*)PyArray_FROM_OTF(g_obj, NPY_DOUBLE,
                              NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!a_array || !b_array || !c_array || !d_array || !f_array || !g_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        Py_XDECREF(f_array);
        Py_XDECREF(g_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 m = (i32)PyArray_DIM(b_array, 1);
    i32 p = (i32)PyArray_DIM(c_array, 0);

    i32 lda = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldc = p > 1 ? p : 1;
    i32 ldd = p > 1 ? p : 1;
    i32 ldf = m > 1 ? m : 1;
    i32 ldg = n > 1 ? n : 1;

    bool left = (jobcf[0] == 'L' || jobcf[0] == 'l');
    i32 mp = left ? m : p;
    i32 n_mp = n > mp ? n : mp;
    i32 n_m_p = m > p ? m : p;
    i32 min_n_mp = n < mp ? n : mp;
    i32 lw1 = 2 * n * n + 5 * n;
    i32 lw2 = n * n_m_p;
    i32 lw3 = n * (n + n_mp + min_n_mp + 6);
    i32 lw_max = lw1;
    if (lw2 > lw_max) lw_max = lw2;
    if (lw3 > lw_max) lw_max = lw3;
    if (1 > lw_max) lw_max = 1;
    i32 ldwork = 2 * n * n + lw_max;

    f64 *dwork = (f64*)calloc(ldwork > 1 ? ldwork : 1, sizeof(f64));
    i32 *iwork = (i32*)calloc(n > 1 ? n : 1, sizeof(i32));
    f64 *hsv = (f64*)calloc(n > 1 ? n : 1, sizeof(f64));

    if (!dwork || !iwork || !hsv) {
        free(dwork);
        free(iwork);
        free(hsv);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(f_array);
        Py_DECREF(g_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *f_data = (f64*)PyArray_DATA(f_array);
    f64 *g_data = (f64*)PyArray_DATA(g_array);

    i32 ncr = (i32)ncr_in;
    i32 iwarn = 0;
    i32 info = 0;

    sb16cd(dico, jobd, jobmr, jobcf, ordsel, n, m, p, &ncr,
           a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           f_data, ldf, g_data, ldg, hsv, tol,
           iwork, dwork, ldwork, &iwarn, &info);

    free(dwork);
    free(iwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(f_array);
    PyArray_ResolveWritebackIfCopy(g_array);

    npy_intp hsv_dims[1] = {n > 1 ? n : 1};
    PyObject *hsv_array = PyArray_SimpleNew(1, hsv_dims, NPY_DOUBLE);
    memcpy(PyArray_DATA((PyArrayObject*)hsv_array), hsv, (n > 1 ? n : 1) * sizeof(f64));
    free(hsv);

    PyObject *result = Py_BuildValue("OOOOiii",
                                      a_array, g_array, f_array, hsv_array,
                                      ncr, iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(f_array);
    Py_DECREF(g_array);
    Py_DECREF(hsv_array);

    return result;
}
