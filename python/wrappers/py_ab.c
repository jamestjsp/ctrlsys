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



/* Python wrapper for ab01md */
PyObject* py_ab01md(PyObject* self, PyObject* args) {
    const char *jobz_str;
    PyObject *a_obj, *b_obj;
    f64 tol;

    if (!PyArg_ParseTuple(args, "sOOd", &jobz_str, &a_obj, &b_obj, &tol)) {
        return NULL;
    }

    char jobz = (char)toupper((unsigned char)jobz_str[0]);
    if (jobz != 'N' && jobz != 'F' && jobz != 'I') {
        PyErr_SetString(PyExc_ValueError, "jobz must be 'N', 'F', or 'I'");
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

    int a_ndim = PyArray_NDIM(a_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);
    int b_ndim = PyArray_NDIM(b_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);

    i32 n;
    if (a_ndim == 2) {
        n = (i32)a_dims[0];
        if (a_dims[0] != a_dims[1]) {
            PyErr_SetString(PyExc_ValueError, "A must be square");
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            return NULL;
        }
    } else if (a_ndim == 0 || (a_ndim == 2 && a_dims[0] == 0)) {
        n = 0;
    } else {
        PyErr_SetString(PyExc_ValueError, "A must be 2D array");
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    npy_intp b_len = (b_ndim == 1) ? b_dims[0] : (b_ndim == 2 ? b_dims[0] : 0);
    if (n > 0 && b_len != n) {
        PyErr_SetString(PyExc_ValueError, "B must have length N");
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldz = (jobz == 'N') ? 1 : (n > 0 ? n : 1);
    i32 ldwork = n > 0 ? n : 1;

    f64 *z = (f64*)malloc(ldz * (n > 0 ? n : 1) * sizeof(f64));
    f64 *tau = (f64*)malloc((n > 0 ? n : 1) * sizeof(f64));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!z || !tau || !dwork) {
        free(z);
        free(tau);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    i32 ncont = 0;
    i32 info = 0;
    char jobz_str_c[2] = {jobz, '\0'};
    ab01md(jobz_str_c, n, a_data, lda, b_data, &ncont, z, ldz, tau, tol,
           dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    PyObject *z_array = NULL;
    if (jobz != 'N' && n > 0) {
        npy_intp z_dims[2] = {n, n};
        npy_intp z_strides[2] = {sizeof(f64), n * sizeof(f64)};
        z_array = PyArray_New(&PyArray_Type, 2, z_dims, NPY_DOUBLE,
                              z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (z_array) {
            memcpy(PyArray_DATA((PyArrayObject*)z_array), z, (size_t)n * n * sizeof(f64));
        } else {
            free(z);
            free(tau);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            return NULL;
        }
    } else {
        npy_intp z_dims[2] = {n > 0 ? n : 0, n > 0 ? n : 0};
        z_array = PyArray_ZEROS(2, z_dims, NPY_DOUBLE, 1);
        if (!z_array) {
            free(z);
            free(tau);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            return NULL;
        }
    }
    free(z);

    npy_intp tau_dims[1] = {n > 0 ? n : 0};
    npy_intp tau_strides[1] = {sizeof(f64)};
    PyObject *tau_array = NULL;
    if (n > 0) {
        tau_array = PyArray_New(&PyArray_Type, 1, tau_dims, NPY_DOUBLE,
                                tau_strides, NULL, 0, 0, NULL);
        if (tau_array) {
            memcpy(PyArray_DATA((PyArrayObject*)tau_array), tau, (size_t)n * sizeof(f64));
        } else {
            free(tau);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(z_array);
            return NULL;
        }
    } else {
        tau_array = PyArray_ZEROS(1, tau_dims, NPY_DOUBLE, 0);
        if (!tau_array) {
            free(tau);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(z_array);
            return NULL;
        }
    }
    free(tau);

    PyObject *result = Py_BuildValue("OOiOOi", a_array, b_array, (int)ncont,
                                     z_array, tau_array, (int)info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(z_array);
    Py_DECREF(tau_array);

    return result;
}



/* Python wrapper for ab08md */
PyObject* py_ab08md(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *equil_str;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    f64 tol = 0.0;

    static char *kwlist[] = {"equil", "n", "m", "p", "a", "b", "c", "d", "tol", NULL};

    i32 n, m, p;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siiiOOOO|d", kwlist,
            &equil_str, &n, &m, &p, &a_obj, &b_obj, &c_obj, &d_obj, &tol)) {
        return NULL;
    }

    char equil = (char)toupper((unsigned char)equil_str[0]);
    if (equil != 'S' && equil != 'N') {
        PyErr_SetString(PyExc_ValueError, "equil must be 'S' or 'N'");
        return NULL;
    }

    PyArrayObject *a_array = NULL, *b_array = NULL, *c_array = NULL, *d_array = NULL;

    if (n > 0) {
        a_array = (PyArrayObject*)PyArray_FROM_OTF(
            a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!a_array) return NULL;
    }

    if (n > 0 && m > 0) {
        b_array = (PyArrayObject*)PyArray_FROM_OTF(
            b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!b_array) {
            Py_XDECREF(a_array);
            return NULL;
        }
    }

    if (p > 0 && n > 0) {
        c_array = (PyArrayObject*)PyArray_FROM_OTF(
            c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!c_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(b_array);
            return NULL;
        }
    }

    if (p > 0 && m > 0) {
        d_array = (PyArrayObject*)PyArray_FROM_OTF(
            d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!d_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(b_array);
            Py_XDECREF(c_array);
            return NULL;
        }
    }

    i32 np = n + p;
    i32 nm = n + m;
    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldd = p > 0 ? p : 1;

    i32 mpm = (p < m) ? p : m;
    i32 mpn = (p < n) ? p : n;
    i32 max_mp = (m > p) ? m : p;

    i32 ldwork = np * nm;
    i32 t1 = mpm + ((3*m - 1 > n) ? (3*m - 1) : n);
    i32 t2 = mpn + ((3*p - 1 > np) ? ((3*p - 1 > nm) ? (3*p - 1) : nm) : ((np > nm) ? np : nm));
    i32 t3 = (t1 > 1) ? t1 : 1;
    t3 = (t3 > t2) ? t3 : t2;
    ldwork += t3;

    i32 liwork = 2*n + max_mp + 1;
    if (liwork < 1) liwork = 1;

    i32 *iwork = (i32*)malloc(liwork * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = a_array ? (f64*)PyArray_DATA(a_array) : NULL;
    f64 *b_data = b_array ? (f64*)PyArray_DATA(b_array) : NULL;
    f64 *c_data = c_array ? (f64*)PyArray_DATA(c_array) : NULL;
    f64 *d_data = d_array ? (f64*)PyArray_DATA(d_array) : NULL;

    f64 dummy = 0.0;
    if (!a_data) a_data = &dummy;
    if (!b_data) b_data = &dummy;
    if (!c_data) c_data = &dummy;
    if (!d_data) d_data = &dummy;

    i32 rank = 0;
    i32 info = 0;
    char equil_str_c[2] = {equil, '\0'};

    ab08md(equil_str_c, n, m, p, a_data, lda, b_data, ldb,
           c_data, ldc, d_data, ldd, &rank, tol, iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    if (a_array) PyArray_ResolveWritebackIfCopy(a_array);
    if (b_array) PyArray_ResolveWritebackIfCopy(b_array);
    if (c_array) PyArray_ResolveWritebackIfCopy(c_array);
    if (d_array) PyArray_ResolveWritebackIfCopy(d_array);

    Py_XDECREF(a_array);
    Py_XDECREF(b_array);
    Py_XDECREF(c_array);
    Py_XDECREF(d_array);

    return Py_BuildValue("ii", rank, info);
}



/* Python wrapper for ab08nd */
PyObject* py_ab08nd(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *equil_str;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    f64 tol = 0.0;

    static char *kwlist[] = {"equil", "n", "m", "p", "a", "b", "c", "d", "tol", NULL};

    i32 n, m, p;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siiiOOOO|d", kwlist,
            &equil_str, &n, &m, &p, &a_obj, &b_obj, &c_obj, &d_obj, &tol)) {
        return NULL;
    }

    char equil = (char)toupper((unsigned char)equil_str[0]);
    if (equil != 'S' && equil != 'N') {
        PyErr_SetString(PyExc_ValueError, "equil must be 'S' or 'N'");
        return NULL;
    }

    if (n < 0 || m < 0 || p < 0) {
        PyErr_SetString(PyExc_ValueError, "n, m, p must be non-negative");
        return NULL;
    }

    PyArrayObject *a_array = NULL, *b_array = NULL, *c_array = NULL, *d_array = NULL;

    if (n > 0) {
        a_array = (PyArrayObject*)PyArray_FROM_OTF(
            a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!a_array) return NULL;
    }

    if (n > 0 && m > 0) {
        b_array = (PyArrayObject*)PyArray_FROM_OTF(
            b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!b_array) {
            Py_XDECREF(a_array);
            return NULL;
        }
    }

    if (p > 0 && n > 0) {
        c_array = (PyArrayObject*)PyArray_FROM_OTF(
            c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!c_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(b_array);
            return NULL;
        }
    }

    if (p > 0 && m > 0) {
        d_array = (PyArrayObject*)PyArray_FROM_OTF(
            d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!d_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(b_array);
            Py_XDECREF(c_array);
            return NULL;
        }
    }

    i32 np = n + p;
    i32 nm = n + m;
    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldd = p > 0 ? p : 1;
    i32 ldaf = (n + m) > 0 ? (n + m) : 1;
    i32 ldbf = np > 0 ? np : 1;

    i32 ii = (p < m) ? p : m;
    i32 mpm = ii;
    i32 mpn = (p < n) ? p : n;
    i32 mmn = (m < n) ? m : n;
    i32 max_mp = (m > p) ? m : p;
    i32 max_nm = (n > m) ? n : m;
    i32 max_np = (n > p) ? n : p;

    i32 t1 = mpm + ((3*m - 1 > n) ? (3*m - 1) : n);
    i32 t2 = mpn + ((3*p - 1 > np) ? ((3*p - 1 > nm) ? (3*p - 1) : nm) : ((np > nm) ? np : nm));
    i32 t3 = mmn + ((3*m - 1 > nm) ? (3*m - 1) : nm);
    i32 ldwork = (t1 > 1) ? t1 : 1;
    ldwork = (ldwork > t2) ? ldwork : t2;
    ldwork = (ldwork > t3) ? ldwork : t3;

    i32 af_cols = n + ii;
    i32 bf_cols = nm;

    f64 *af = (f64*)calloc(ldaf * (af_cols > 0 ? af_cols : 1), sizeof(f64));
    f64 *bf = (f64*)calloc(ldbf * (bf_cols > 0 ? bf_cols : 1), sizeof(f64));
    i32 *infz = (i32*)calloc(n > 0 ? n : 1, sizeof(i32));
    i32 *kronr = (i32*)calloc(max_nm + 1, sizeof(i32));
    i32 *kronl = (i32*)calloc(max_np + 1, sizeof(i32));
    i32 *iwork = (i32*)malloc((max_mp > 1 ? max_mp : 1) * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!af || !bf || !infz || !kronr || !kronl || !iwork || !dwork) {
        free(af); free(bf); free(infz); free(kronr); free(kronl);
        free(iwork); free(dwork);
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = a_array ? (f64*)PyArray_DATA(a_array) : NULL;
    f64 *b_data = b_array ? (f64*)PyArray_DATA(b_array) : NULL;
    f64 *c_data = c_array ? (f64*)PyArray_DATA(c_array) : NULL;
    f64 *d_data = d_array ? (f64*)PyArray_DATA(d_array) : NULL;

    f64 dummy = 0.0;
    if (!a_data) a_data = &dummy;
    if (!b_data) b_data = &dummy;
    if (!c_data) c_data = &dummy;
    if (!d_data) d_data = &dummy;

    i32 nu = 0, rank = 0, dinfz = 0, nkror = 0, nkrol = 0, info = 0;
    char equil_str_c[2] = {equil, '\0'};

    ab08nd(equil_str_c, n, m, p, a_data, lda, b_data, ldb,
           c_data, ldc, d_data, ldd, &nu, &rank, &dinfz, &nkror, &nkrol,
           infz, kronr, kronl, af, ldaf, bf, ldbf, tol, iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    if (a_array) PyArray_ResolveWritebackIfCopy(a_array);
    if (b_array) PyArray_ResolveWritebackIfCopy(b_array);
    if (c_array) PyArray_ResolveWritebackIfCopy(c_array);
    if (d_array) PyArray_ResolveWritebackIfCopy(d_array);

    Py_XDECREF(a_array);
    Py_XDECREF(b_array);
    Py_XDECREF(c_array);
    Py_XDECREF(d_array);

    npy_intp infz_dims[1] = {n > 0 ? n : 0};
    npy_intp kronr_dims[1] = {nkror};
    npy_intp kronl_dims[1] = {nkrol};
    npy_intp af_dims[2] = {ldaf, af_cols > 0 ? af_cols : 1};
    npy_intp bf_dims[2] = {ldbf, bf_cols > 0 ? bf_cols : 1};
    npy_intp af_strides[2] = {sizeof(f64), ldaf * sizeof(f64)};
    npy_intp bf_strides[2] = {sizeof(f64), ldbf * sizeof(f64)};

    PyObject *infz_array = PyArray_SimpleNew(1, infz_dims, NPY_INT32);
    PyObject *kronr_array = PyArray_SimpleNew(1, kronr_dims, NPY_INT32);
    PyObject *kronl_array = PyArray_SimpleNew(1, kronl_dims, NPY_INT32);

    if (n > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)infz_array), infz, n * sizeof(i32));
    }
    if (nkror > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)kronr_array), kronr, nkror * sizeof(i32));
    }
    if (nkrol > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)kronl_array), kronl, nkrol * sizeof(i32));
    }

    PyObject *af_array = PyArray_New(&PyArray_Type, 2, af_dims, NPY_DOUBLE,
                                      af_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (af_array) {
        memcpy(PyArray_DATA((PyArrayObject*)af_array), af, (size_t)ldaf * (af_cols > 0 ? af_cols : 1) * sizeof(f64));
    }
    free(af);

    PyObject *bf_array = PyArray_New(&PyArray_Type, 2, bf_dims, NPY_DOUBLE,
                                      bf_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (bf_array) {
        memcpy(PyArray_DATA((PyArrayObject*)bf_array), bf, (size_t)ldbf * (bf_cols > 0 ? bf_cols : 1) * sizeof(f64));
    }
    free(bf);

    free(infz);
    free(kronr);
    free(kronl);

    PyObject *result = Py_BuildValue("(iiiiiOOOOOi)", nu, rank, dinfz, nkror, nkrol,
                                     infz_array, kronr_array, kronl_array, af_array, bf_array, info);

    Py_DECREF(infz_array);
    Py_DECREF(kronr_array);
    Py_DECREF(kronl_array);
    Py_DECREF(af_array);
    Py_DECREF(bf_array);

    return result;
}



/* Python wrapper for ab08nw */
PyObject* py_ab08nw(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *equil_str;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    f64 tol = 0.0;

    static char *kwlist[] = {"equil", "n", "m", "p", "a", "b", "c", "d", "tol", NULL};

    i32 n, m, p;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siiiOOOO|d", kwlist,
            &equil_str, &n, &m, &p, &a_obj, &b_obj, &c_obj, &d_obj, &tol)) {
        return NULL;
    }

    char equil = (char)toupper((unsigned char)equil_str[0]);
    if (equil != 'S' && equil != 'N') {
        PyErr_SetString(PyExc_ValueError, "equil must be 'S' or 'N'");
        return NULL;
    }

    if (n < 0 || m < 0 || p < 0) {
        PyErr_SetString(PyExc_ValueError, "n, m, p must be non-negative");
        return NULL;
    }

    PyArrayObject *a_array = NULL, *b_array = NULL, *c_array = NULL, *d_array = NULL;

    if (n > 0) {
        a_array = (PyArrayObject*)PyArray_FROM_OTF(
            a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!a_array) return NULL;
    }

    if (n > 0 && m > 0) {
        b_array = (PyArrayObject*)PyArray_FROM_OTF(
            b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!b_array) {
            Py_XDECREF(a_array);
            return NULL;
        }
    }

    if (p > 0 && n > 0) {
        c_array = (PyArrayObject*)PyArray_FROM_OTF(
            c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!c_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(b_array);
            return NULL;
        }
    }

    if (p > 0 && m > 0) {
        d_array = (PyArrayObject*)PyArray_FROM_OTF(
            d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!d_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(b_array);
            Py_XDECREF(c_array);
            return NULL;
        }
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldd = p > 0 ? p : 1;
    i32 lde = n > 0 ? n : 1;

    i32 max_mp = (m > p) ? m : p;
    i32 ldabcd = n + max_mp;
    i32 labcd2 = ldabcd * ldabcd;

    i32 ldwork;
    if (n == 0 && m == 0 && p == 0) {
        ldwork = 1;
    } else {
        // Workspace for original call with (n, m, p)
        i32 mpm = (p < m) ? p : m;
        i32 mpn = (p < n) ? p : n;
        i32 t1 = mpm + m + ((2*m > n) ? (2*m) : n) - 1;
        i32 t2 = mpn + ((ldabcd > (3*p - 1)) ? ldabcd : (3*p - 1));
        i32 work1 = (t1 > t2) ? t1 : t2;
        if (work1 < 1) work1 = 1;

        // Workspace for transformed call with (n, m, m) - when mu != mm
        // This matches Fortran's workspace query: AB08NY(.FALSE., N, M, M, ...)
        i32 mpm2 = m;  // min(m, m) = m
        i32 mpn2 = (m < n) ? m : n;
        i32 t1b = mpm2 + m + ((2*m > n) ? (2*m) : n) - 1;
        i32 t2b = mpn2 + ((ldabcd > (3*m - 1)) ? ldabcd : (3*m - 1));
        i32 work2 = (t1b > t2b) ? t1b : t2b;
        if (work2 < 1) work2 = 1;

        ldwork = (work1 > work2) ? work1 : work2;
        ldwork += labcd2;
    }

    f64 *e = (f64*)calloc(lde * (n > 0 ? n : 1), sizeof(f64));
    i32 *infz = (i32*)calloc(n + 1 > 0 ? n + 1 : 1, sizeof(i32));
    i32 *kronr = (i32*)calloc(n + 1 > 0 ? n + 1 : 1, sizeof(i32));
    i32 *infe = (i32*)calloc(n + 1 > 0 ? n + 1 : 1, sizeof(i32));
    i32 *kronl = (i32*)calloc(n + 1 > 0 ? n + 1 : 1, sizeof(i32));
    i32 iwork_size = (max_mp > n + 1) ? max_mp : (n + 1);
    if (iwork_size < 1) iwork_size = 1;
    i32 *iwork = (i32*)malloc(iwork_size * sizeof(i32));
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (!e || !infz || !kronr || !infe || !kronl || !iwork || !dwork) {
        free(e); free(infz); free(kronr); free(infe); free(kronl);
        free(iwork); free(dwork);
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = a_array ? (f64*)PyArray_DATA(a_array) : NULL;
    f64 *b_data = b_array ? (f64*)PyArray_DATA(b_array) : NULL;
    f64 *c_data = c_array ? (f64*)PyArray_DATA(c_array) : NULL;
    f64 *d_data = d_array ? (f64*)PyArray_DATA(d_array) : NULL;

    f64 dummy = 0.0;
    if (!a_data) a_data = &dummy;
    if (!b_data) b_data = &dummy;
    if (!c_data) c_data = &dummy;
    if (!d_data) d_data = &dummy;

    i32 nfz = 0, nrank = 0, niz = 0, dinfz = 0;
    i32 nkror = 0, ninfe = 0, nkrol = 0, info = 0;
    char equil_str_c[2] = {equil, '\0'};

    ab08nw(equil_str_c, n, m, p, a_data, lda, b_data, ldb,
           c_data, ldc, d_data, ldd, &nfz, &nrank, &niz, &dinfz,
           &nkror, &ninfe, &nkrol, infz, kronr, infe, kronl,
           e, lde, tol, iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    if (a_array) PyArray_ResolveWritebackIfCopy(a_array);
    if (b_array) PyArray_ResolveWritebackIfCopy(b_array);
    if (c_array) PyArray_ResolveWritebackIfCopy(c_array);
    if (d_array) PyArray_ResolveWritebackIfCopy(d_array);

    npy_intp e_dims[2] = {n, n};
    npy_intp e_strides[2] = {sizeof(f64), lde * sizeof(f64)};
    npy_intp infz_dims[1] = {dinfz > 0 ? dinfz : 0};
    npy_intp kronr_dims[1] = {nkror > 0 ? nkror : 0};
    npy_intp infe_dims[1] = {ninfe > 0 ? ninfe : 0};
    npy_intp kronl_dims[1] = {nkrol > 0 ? nkrol : 0};

    PyObject *af_array;
    if (a_array && n > 0) {
        af_array = (PyObject*)a_array;
        Py_INCREF(af_array);
    } else {
        npy_intp zero_dims[2] = {0, 0};
        af_array = PyArray_ZEROS(2, zero_dims, NPY_DOUBLE, 1);
    }

    PyObject *e_array;
    if (n > 0) {
        e_array = PyArray_New(&PyArray_Type, 2, e_dims, NPY_DOUBLE,
                              e_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (e_array) {
            memcpy(PyArray_DATA((PyArrayObject*)e_array), e, (size_t)lde * n * sizeof(f64));
        }
        free(e);
    } else {
        free(e);
        npy_intp zero_dims[2] = {0, 0};
        e_array = PyArray_ZEROS(2, zero_dims, NPY_DOUBLE, 1);
    }

    PyObject *infz_array = PyArray_SimpleNew(1, infz_dims, NPY_INT32);
    PyObject *kronr_array = PyArray_SimpleNew(1, kronr_dims, NPY_INT32);
    PyObject *infe_array = PyArray_SimpleNew(1, infe_dims, NPY_INT32);
    PyObject *kronl_array = PyArray_SimpleNew(1, kronl_dims, NPY_INT32);

    if (!af_array || !e_array || !infz_array || !kronr_array || !infe_array || !kronl_array) {
        free(infz);
        free(kronr);
        free(infe);
        free(kronl);
        Py_XDECREF(af_array);
        Py_XDECREF(e_array);
        Py_XDECREF(infz_array);
        Py_XDECREF(kronr_array);
        Py_XDECREF(infe_array);
        Py_XDECREF(kronl_array);
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        return NULL;
    }

    if (dinfz > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)infz_array), infz, dinfz * sizeof(i32));
    }
    if (nkror > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)kronr_array), kronr, nkror * sizeof(i32));
    }
    if (ninfe > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)infe_array), infe, ninfe * sizeof(i32));
    }
    if (nkrol > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)kronl_array), kronl, nkrol * sizeof(i32));
    }

    free(infz);
    free(kronr);
    free(infe);
    free(kronl);

    Py_XDECREF(b_array);
    Py_XDECREF(c_array);
    Py_XDECREF(d_array);

    PyObject *result = Py_BuildValue("(OOiiiiiiiOOOOi)",
                                     af_array, e_array,
                                     nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
                                     infz_array, kronr_array, infe_array, kronl_array, info);

    Py_XDECREF(a_array);
    Py_DECREF(af_array);
    Py_DECREF(e_array);
    Py_DECREF(infz_array);
    Py_DECREF(kronr_array);
    Py_DECREF(infe_array);
    Py_DECREF(kronl_array);

    return result;
}


/* Python wrapper for ab08nx */
PyObject* py_ab08nx(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *abcd_obj;
    i32 n, m, p, ro, sigma, ninfz;
    f64 svlmax, tol;

    static char *kwlist[] = {"n", "m", "p", "ro", "sigma", "svlmax", "abcd",
                             "ninfz", "tol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiiiidOid", kwlist,
            &n, &m, &p, &ro, &sigma, &svlmax, &abcd_obj, &ninfz, &tol)) {
        return NULL;
    }

    if (n < 0 || m < 0 || p < 0) {
        PyErr_SetString(PyExc_ValueError, "n, m, p must be non-negative");
        return NULL;
    }

    i32 np = n + p;
    i32 nm = n + m;

    PyArrayObject *abcd_array = (PyArrayObject*)PyArray_FROM_OTF(
        abcd_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!abcd_array) return NULL;

    i32 ldabcd = np > 1 ? np : 1;

    i32 max_mp = (m > p) ? m : p;
    i32 *iwork = (i32*)malloc((max_mp > 1 ? max_mp : 1) * sizeof(i32));
    i32 *infz = (i32*)calloc(n > 0 ? n : 1, sizeof(i32));
    i32 *kronl = (i32*)calloc(n + 1 > 0 ? n + 1 : 1, sizeof(i32));

    i32 mpm = (p < m) ? p : m;
    i32 mpn = (p < n) ? p : n;
    i32 t1 = mpm + ((3*m - 1 > n) ? (3*m - 1) : n);
    i32 t2 = mpn + ((3*p - 1 > np) ? ((3*p - 1 > nm) ? (3*p - 1) : nm) : ((np > nm) ? np : nm));
    i32 ldwork = (t1 > 1) ? t1 : 1;
    ldwork = (ldwork > t2) ? ldwork : t2;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!iwork || !infz || !kronl || !dwork) {
        free(iwork); free(infz); free(kronl); free(dwork);
        Py_DECREF(abcd_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *abcd_data = (f64*)PyArray_DATA(abcd_array);
    i32 mu = 0, nu = 0, nkrol = 0, info = 0;

    ab08nx(n, m, p, &ro, &sigma, svlmax, abcd_data, ldabcd,
           &ninfz, infz, kronl, &mu, &nu, &nkrol, tol, iwork, dwork, ldwork, &info);

    npy_intp infz_dims[1] = {n > 0 ? n : 0};
    npy_intp kronl_dims[1] = {nkrol > 0 ? nkrol : 0};

    PyObject *infz_array = PyArray_SimpleNew(1, infz_dims, NPY_INT32);
    PyObject *kronl_array = PyArray_SimpleNew(1, kronl_dims, NPY_INT32);

    if (n > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)infz_array), infz, n * sizeof(i32));
    }
    if (nkrol > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)kronl_array), kronl, nkrol * sizeof(i32));
    }

    free(iwork); free(infz); free(kronl); free(dwork);

    PyArray_ResolveWritebackIfCopy(abcd_array);

    PyObject *result = Py_BuildValue("(OiiiiiiOOi)", abcd_array, ro, sigma,
                                     ninfz, mu, nu, nkrol, infz_array, kronl_array, info);

    Py_DECREF(abcd_array);
    Py_DECREF(infz_array);
    Py_DECREF(kronl_array);

    return result;
}


/* Python wrapper for ab08ny */
PyObject* py_ab08ny(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *abcd_obj;
    i32 n, m, p, ninfz;
    f64 svlmax, tol;
    int first_int;

    static char *kwlist[] = {"first", "n", "m", "p", "svlmax", "abcd",
                             "ninfz", "tol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "piiidOid", kwlist,
            &first_int, &n, &m, &p, &svlmax, &abcd_obj, &ninfz, &tol)) {
        return NULL;
    }

    bool first = (first_int != 0);

    if (n < 0 || m < 0 || p < 0) {
        PyErr_SetString(PyExc_ValueError, "n, m, p must be non-negative");
        return NULL;
    }

    i32 np = n + p;

    PyArrayObject *abcd_array = (PyArrayObject*)PyArray_FROM_OTF(
        abcd_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!abcd_array) return NULL;

    i32 ldabcd = np > 1 ? np : 1;

    i32 max_mp = (m > p) ? m : p;
    i32 *iwork = (i32*)malloc((max_mp > 1 ? max_mp : 1) * sizeof(i32));
    i32 *infz = (i32*)calloc(n > 0 ? n : 1, sizeof(i32));
    i32 *kronl = (i32*)calloc(n + 1 > 0 ? n + 1 : 1, sizeof(i32));

    i32 mpm = (p < m) ? p : m;
    i32 mpn = (p < n) ? p : n;
    i32 ldwork;
    if ((p < 1) || ((n < 1) && (m < 1))) {
        ldwork = 1;
    } else {
        i32 t1 = mpm + m + ((2*m > n) ? (2*m) : n) - 1;
        i32 t2 = mpn + ((n + ((p > m) ? p : m) > 3*p - 1) ?
                        (n + ((p > m) ? p : m)) : (3*p - 1));
        ldwork = (t1 > t2) ? t1 : t2;
        ldwork = (ldwork > 1) ? ldwork : 1;
    }
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!iwork || !infz || !kronl || !dwork) {
        free(iwork); free(infz); free(kronl); free(dwork);
        Py_DECREF(abcd_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *abcd_data = (f64*)PyArray_DATA(abcd_array);
    i32 nr = 0, pr = 0, dinfz = 0, nkronl = 0, info = 0;

    ab08ny(first, n, m, p, svlmax, abcd_data, ldabcd,
           &ninfz, &nr, &pr, &dinfz, &nkronl, infz, kronl,
           tol, iwork, dwork, ldwork, &info);

    npy_intp infz_dims[1] = {n > 0 ? n : 0};
    npy_intp kronl_dims[1] = {nkronl > 0 ? nkronl : 0};

    PyObject *infz_array = PyArray_SimpleNew(1, infz_dims, NPY_INT32);
    PyObject *kronl_array = PyArray_SimpleNew(1, kronl_dims, NPY_INT32);

    if (n > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)infz_array), infz, n * sizeof(i32));
    }
    if (nkronl > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)kronl_array), kronl, nkronl * sizeof(i32));
    }

    free(iwork); free(infz); free(kronl); free(dwork);

    PyArray_ResolveWritebackIfCopy(abcd_array);

    PyObject *result = Py_BuildValue("(OiiiiiOOi)", abcd_array, ninfz, nr, pr,
                                     dinfz, nkronl, infz_array, kronl_array, info);

    Py_DECREF(abcd_array);
    Py_DECREF(infz_array);
    Py_DECREF(kronl_array);

    return result;
}


/* Python wrapper for ab01nd */
PyObject* py_ab01nd(PyObject* self, PyObject* args) {
    const char *jobz_str;
    PyObject *a_obj, *b_obj;
    f64 tol;

    if (!PyArg_ParseTuple(args, "sOOd", &jobz_str, &a_obj, &b_obj, &tol)) {
        return NULL;
    }

    char jobz = (char)toupper((unsigned char)jobz_str[0]);
    if (jobz != 'N' && jobz != 'F' && jobz != 'I') {
        PyErr_SetString(PyExc_ValueError, "jobz must be 'N', 'F', or 'I'");
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

    int a_ndim = PyArray_NDIM(a_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);
    int b_ndim = PyArray_NDIM(b_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);

    i32 n, m;
    if (a_ndim == 2) {
        n = (i32)a_dims[0];
        if (a_dims[0] != a_dims[1]) {
            PyErr_SetString(PyExc_ValueError, "A must be square");
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            return NULL;
        }
    } else if (a_ndim == 0 || (a_ndim == 2 && a_dims[0] == 0)) {
        n = 0;
    } else {
        PyErr_SetString(PyExc_ValueError, "A must be 2D array");
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    if (b_ndim == 2) {
        if (n > 0 && b_dims[0] != n) {
            PyErr_SetString(PyExc_ValueError, "B must have N rows");
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            return NULL;
        }
        m = (i32)b_dims[1];
    } else if (b_ndim == 0 || (b_ndim == 2 && b_dims[0] == 0)) {
        m = (b_ndim == 2) ? (i32)b_dims[1] : 0;
    } else {
        PyErr_SetString(PyExc_ValueError, "B must be 2D array");
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldz = (jobz == 'N') ? 1 : (n > 0 ? n : 1);
    i32 ldwork = 1;
    if (n > ldwork) ldwork = n;
    if (3 * m > ldwork) ldwork = 3 * m;
    if (ldwork < 1) ldwork = 1;

    f64 *z = (f64*)malloc(ldz * (n > 0 ? n : 1) * sizeof(f64));
    f64 *tau = (f64*)malloc((n > 0 ? n : 1) * sizeof(f64));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    i32 *iwork = (i32*)malloc((m > 0 ? m : 1) * sizeof(i32));
    i32 *nblk = (i32*)malloc((n > 0 ? n : 1) * sizeof(i32));

    if (!z || !tau || !dwork || !iwork || !nblk) {
        free(z);
        free(tau);
        free(dwork);
        free(iwork);
        free(nblk);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    i32 ncont = 0;
    i32 indcon = 0;
    i32 info = 0;
    char jobz_str_c[2] = {jobz, '\0'};
    ab01nd(jobz_str_c, n, m, a_data, lda, b_data, ldb, &ncont, &indcon,
           nblk, z, ldz, tau, tol, iwork, dwork, ldwork, &info);

    free(dwork);
    free(iwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    PyObject *z_array = NULL;
    if (jobz != 'N' && n > 0) {
        npy_intp z_dims[2] = {n, n};
        npy_intp z_strides[2] = {sizeof(f64), n * sizeof(f64)};
        z_array = PyArray_New(&PyArray_Type, 2, z_dims, NPY_DOUBLE,
                              z_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (z_array) {
            memcpy(PyArray_DATA((PyArrayObject*)z_array), z, (size_t)n * n * sizeof(f64));
        } else {
            free(z);
            free(tau);
            free(nblk);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            return NULL;
        }
    } else {
        npy_intp z_dims[2] = {n > 0 ? n : 0, n > 0 ? n : 0};
        z_array = PyArray_ZEROS(2, z_dims, NPY_DOUBLE, 1);
        if (!z_array) {
            free(z);
            free(tau);
            free(nblk);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            return NULL;
        }
    }
    free(z);

    npy_intp tau_dims[1] = {n > 0 ? n : 0};
    npy_intp tau_strides[1] = {sizeof(f64)};
    PyObject *tau_array = NULL;
    if (n > 0) {
        tau_array = PyArray_New(&PyArray_Type, 1, tau_dims, NPY_DOUBLE,
                                tau_strides, NULL, 0, 0, NULL);
        if (tau_array) {
            memcpy(PyArray_DATA((PyArrayObject*)tau_array), tau, (size_t)n * sizeof(f64));
        } else {
            free(tau);
            free(nblk);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(z_array);
            return NULL;
        }
    } else {
        tau_array = PyArray_ZEROS(1, tau_dims, NPY_DOUBLE, 0);
        if (!tau_array) {
            free(tau);
            free(nblk);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(z_array);
            return NULL;
        }
    }
    free(tau);

    npy_intp nblk_dims[1] = {n > 0 ? n : 0};
    PyObject *nblk_array = NULL;
    if (n > 0) {
        nblk_array = PyArray_New(&PyArray_Type, 1, nblk_dims, NPY_INT32,
                                 NULL, NULL, 0, 0, NULL);
        if (nblk_array) {
            memcpy(PyArray_DATA((PyArrayObject*)nblk_array), nblk, (size_t)n * sizeof(i32));
        } else {
            free(nblk);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(z_array);
            Py_DECREF(tau_array);
            return NULL;
        }
    } else {
        nblk_array = PyArray_ZEROS(1, nblk_dims, NPY_INT32, 0);
        if (!nblk_array) {
            free(nblk);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(z_array);
            Py_DECREF(tau_array);
            return NULL;
        }
    }
    free(nblk);

    PyObject *result = Py_BuildValue("OOiiOOOi", a_array, b_array, (int)ncont,
                                     (int)indcon, nblk_array, z_array,
                                     tau_array, (int)info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(z_array);
    Py_DECREF(tau_array);
    Py_DECREF(nblk_array);

    return result;
}



/* Python wrapper for ab01od */
PyObject* py_ab01od(PyObject* self, PyObject* args) {
    const char *stages_str, *jobu_str, *jobv_str;
    PyObject *a_obj, *b_obj;
    f64 tol;

    if (!PyArg_ParseTuple(args, "sssOOd", &stages_str, &jobu_str, &jobv_str,
                          &a_obj, &b_obj, &tol)) {
        return NULL;
    }

    char stages = (char)toupper((unsigned char)stages_str[0]);
    char jobu = (char)toupper((unsigned char)jobu_str[0]);
    char jobv = (char)toupper((unsigned char)jobv_str[0]);

    if (stages != 'F' && stages != 'B' && stages != 'A') {
        PyErr_SetString(PyExc_ValueError, "stages must be 'F', 'B', or 'A'");
        return NULL;
    }
    if (jobu != 'N' && jobu != 'I') {
        PyErr_SetString(PyExc_ValueError, "jobu must be 'N' or 'I'");
        return NULL;
    }
    if (stages != 'F' && jobv != 'N' && jobv != 'I') {
        PyErr_SetString(PyExc_ValueError, "jobv must be 'N' or 'I'");
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

    int a_ndim = PyArray_NDIM(a_array);
    npy_intp *a_dims = PyArray_DIMS(a_array);
    int b_ndim = PyArray_NDIM(b_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);

    i32 n, m;
    if (a_ndim == 2) {
        n = (i32)a_dims[0];
        if (a_dims[0] != a_dims[1]) {
            PyErr_SetString(PyExc_ValueError, "A must be square");
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            return NULL;
        }
    } else if (a_ndim == 0 || (a_ndim == 2 && a_dims[0] == 0)) {
        n = 0;
    } else {
        PyErr_SetString(PyExc_ValueError, "A must be 2D array");
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    if (b_ndim == 2) {
        if (n > 0 && b_dims[0] != n) {
            PyErr_SetString(PyExc_ValueError, "B must have N rows");
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            return NULL;
        }
        m = (i32)b_dims[1];
    } else if (b_ndim == 0 || (b_ndim == 2 && b_dims[0] == 0)) {
        m = (b_ndim == 2) ? (i32)b_dims[1] : 0;
    } else {
        PyErr_SetString(PyExc_ValueError, "B must be 2D array");
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldu = (jobu == 'N') ? 1 : (n > 0 ? n : 1);
    i32 ldv = (stages == 'F' || jobv == 'N') ? 1 : (m > 0 ? m : 1);

    i32 ldwork;
    if (stages != 'B') {
        ldwork = n + (n > 3 * m ? n : 3 * m);
    } else {
        ldwork = m + (n > m ? n : m);
    }
    if (ldwork < 1) ldwork = 1;

    i32 u_size = ldu * (n > 0 ? n : 1);
    i32 v_size = ldv * (m > 0 ? m : 1);

    f64 *u_data = (f64*)malloc(u_size * sizeof(f64));
    f64 *v_data = (f64*)malloc(v_size * sizeof(f64));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    i32 *iwork = (i32*)malloc((m > 0 ? m : 1) * sizeof(i32));
    i32 *kstair = (i32*)malloc((n > 0 ? n : 1) * sizeof(i32));

    if (!u_data || !v_data || !dwork || !iwork || !kstair) {
        free(u_data);
        free(v_data);
        free(dwork);
        free(iwork);
        free(kstair);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    i32 ncont = 0;
    i32 indcon = 0;
    i32 info = 0;

    char stages_str_c[2] = {stages, '\0'};
    char jobu_str_c[2] = {jobu, '\0'};
    char jobv_str_c[2] = {jobv, '\0'};

    ab01od(stages_str_c, jobu_str_c, jobv_str_c, n, m, a_data, lda, b_data, ldb,
           u_data, ldu, v_data, ldv, &ncont, &indcon, kstair, tol,
           iwork, dwork, ldwork, &info);

    free(dwork);
    free(iwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    PyObject *u_array = NULL;
    if (jobu == 'I' && n > 0) {
        npy_intp u_dims[2] = {n, n};
        npy_intp u_strides[2] = {sizeof(f64), n * sizeof(f64)};
        u_array = PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE,
                              u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (u_array) {
            memcpy(PyArray_DATA((PyArrayObject*)u_array), u_data, (size_t)n * n * sizeof(f64));
        } else {
            free(u_data);
            free(v_data);
            free(kstair);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            return NULL;
        }
    } else {
        npy_intp u_dims[2] = {n > 0 ? n : 0, n > 0 ? n : 0};
        u_array = PyArray_ZEROS(2, u_dims, NPY_DOUBLE, 1);
        if (!u_array) {
            free(u_data);
            free(v_data);
            free(kstair);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            return NULL;
        }
    }
    free(u_data);

    PyObject *v_array = NULL;
    if (stages != 'F' && jobv == 'I' && m > 0) {
        npy_intp v_dims[2] = {m, m};
        npy_intp v_strides[2] = {sizeof(f64), m * sizeof(f64)};
        v_array = PyArray_New(&PyArray_Type, 2, v_dims, NPY_DOUBLE,
                              v_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (v_array) {
            memcpy(PyArray_DATA((PyArrayObject*)v_array), v_data, (size_t)m * m * sizeof(f64));
        } else {
            free(v_data);
            free(kstair);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(u_array);
            return NULL;
        }
    } else {
        npy_intp v_dims[2] = {m > 0 ? m : 0, m > 0 ? m : 0};
        v_array = PyArray_ZEROS(2, v_dims, NPY_DOUBLE, 1);
        if (!v_array) {
            free(v_data);
            free(kstair);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(u_array);
            return NULL;
        }
    }
    free(v_data);

    npy_intp kstair_dims[1] = {n > 0 ? n : 0};
    PyObject *kstair_array = NULL;
    if (n > 0) {
        kstair_array = PyArray_New(&PyArray_Type, 1, kstair_dims, NPY_INT32,
                                   NULL, NULL, 0, 0, NULL);
        if (kstair_array) {
            memcpy(PyArray_DATA((PyArrayObject*)kstair_array), kstair, (size_t)n * sizeof(i32));
        } else {
            free(kstair);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(u_array);
            Py_DECREF(v_array);
            return NULL;
        }
    } else {
        kstair_array = PyArray_ZEROS(1, kstair_dims, NPY_INT32, 0);
        if (!kstair_array) {
            free(kstair);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(u_array);
            Py_DECREF(v_array);
            return NULL;
        }
    }
    free(kstair);

    PyObject *result = Py_BuildValue("OOOOiiOi", a_array, b_array, u_array, v_array,
                                     (int)ncont, (int)indcon, kstair_array, (int)info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(u_array);
    Py_DECREF(v_array);
    Py_DECREF(kstair_array);

    return result;
}



/* Python wrapper for ab04md */
PyObject* py_ab04md(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *type_str;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    f64 alpha = 1.0, beta = 1.0;

    static char *kwlist[] = {"type", "a", "b", "c", "d", "alpha", "beta", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sOOOO|dd", kwlist,
                                      &type_str, &a_obj, &b_obj, &c_obj, &d_obj,
                                      &alpha, &beta)) {
        return NULL;
    }

    char type = (char)toupper((unsigned char)type_str[0]);
    if (type != 'D' && type != 'C') {
        PyErr_SetString(PyExc_ValueError, "type must be 'D' or 'C'");
        return NULL;
    }
    if (alpha == 0.0) {
        PyErr_SetString(PyExc_ValueError, "alpha must be non-zero");
        return NULL;
    }
    if (beta == 0.0) {
        PyErr_SetString(PyExc_ValueError, "beta must be non-zero");
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

    i32 n = PyArray_NDIM(a_array) >= 1 ? (i32)a_dims[0] : 0;
    i32 m = PyArray_NDIM(b_array) >= 2 ? (i32)b_dims[1] : (PyArray_NDIM(b_array) == 1 ? 1 : 0);
    i32 p = PyArray_NDIM(c_array) >= 1 ? (i32)c_dims[0] : 0;

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldd = p > 0 ? p : 1;

    i32 ldwork = n > 0 ? n : 1;
    i32 *iwork = (i32*)malloc((n > 0 ? n : 1) * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
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

    i32 info = ab04md(type, n, m, p, alpha, beta,
                      a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
                      iwork, dwork, ldwork);

    free(iwork);
    free(dwork);

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



/* Python wrapper for ab05md */
PyObject* py_ab05md(PyObject* self, PyObject* args) {
    const char *uplo_str, *over_str;
    PyObject *a1_obj, *b1_obj, *c1_obj, *d1_obj;
    PyObject *a2_obj, *b2_obj, *c2_obj, *d2_obj;

    if (!PyArg_ParseTuple(args, "ssOOOOOOOO",
                          &uplo_str, &over_str,
                          &a1_obj, &b1_obj, &c1_obj, &d1_obj,
                          &a2_obj, &b2_obj, &c2_obj, &d2_obj)) {
        return NULL;
    }

    char uplo = (char)toupper((unsigned char)uplo_str[0]);
    char over = (char)toupper((unsigned char)over_str[0]);

    if (uplo != 'L' && uplo != 'U') {
        PyErr_SetString(PyExc_ValueError, "uplo must be 'L' or 'U'");
        return NULL;
    }
    if (over != 'N' && over != 'O') {
        PyErr_SetString(PyExc_ValueError, "over must be 'N' or 'O'");
        return NULL;
    }

    PyArrayObject *a1_array = (PyArrayObject*)PyArray_FROM_OTF(
        a1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!a1_array) return NULL;

    PyArrayObject *b1_array = (PyArrayObject*)PyArray_FROM_OTF(
        b1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!b1_array) {
        Py_DECREF(a1_array);
        return NULL;
    }

    PyArrayObject *c1_array = (PyArrayObject*)PyArray_FROM_OTF(
        c1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!c1_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        return NULL;
    }

    PyArrayObject *d1_array = (PyArrayObject*)PyArray_FROM_OTF(
        d1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!d1_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        return NULL;
    }

    PyArrayObject *a2_array = (PyArrayObject*)PyArray_FROM_OTF(
        a2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!a2_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        return NULL;
    }

    PyArrayObject *b2_array = (PyArrayObject*)PyArray_FROM_OTF(
        b2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!b2_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        return NULL;
    }

    PyArrayObject *c2_array = (PyArrayObject*)PyArray_FROM_OTF(
        c2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!c2_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b2_array);
        return NULL;
    }

    PyArrayObject *d2_array = (PyArrayObject*)PyArray_FROM_OTF(
        d2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!d2_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b2_array);
        Py_DECREF(c2_array);
        return NULL;
    }

    npy_intp *a1_dims = PyArray_DIMS(a1_array);
    npy_intp *b1_dims = PyArray_DIMS(b1_array);
    npy_intp *c1_dims = PyArray_DIMS(c1_array);
    npy_intp *a2_dims = PyArray_DIMS(a2_array);
    npy_intp *b2_dims = PyArray_DIMS(b2_array);
    npy_intp *c2_dims = PyArray_DIMS(c2_array);
    npy_intp *d2_dims = PyArray_DIMS(d2_array);

    i32 n1 = PyArray_NDIM(a1_array) >= 1 ? (i32)a1_dims[0] : 0;
    i32 m1 = PyArray_NDIM(b1_array) >= 2 ? (i32)b1_dims[1] : (PyArray_NDIM(b1_array) == 1 ? 1 : 0);
    i32 p1 = PyArray_NDIM(c1_array) >= 1 ? (i32)c1_dims[0] : 0;
    i32 n2 = PyArray_NDIM(a2_array) >= 1 ? (i32)a2_dims[0] : 0;
    i32 p2 = PyArray_NDIM(c2_array) >= 1 ? (i32)c2_dims[0] : 0;

    i32 b2_cols = PyArray_NDIM(b2_array) >= 2 ? (i32)b2_dims[1] : (PyArray_NDIM(b2_array) == 1 ? 1 : 0);
    i32 d2_cols = PyArray_NDIM(d2_array) >= 2 ? (i32)d2_dims[1] : (PyArray_NDIM(d2_array) == 1 ? 1 : 0);
    if (b2_cols != p1 || d2_cols != p1) {
        PyErr_SetString(PyExc_ValueError, "P1 dimension mismatch: C1 rows must equal B2 cols and D2 cols");
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b2_array);
        Py_DECREF(c2_array);
        Py_DECREF(d2_array);
        return NULL;
    }

    i32 n = n1 + n2;
    i32 lda1 = n1 > 0 ? n1 : 1;
    i32 ldb1 = n1 > 0 ? n1 : 1;
    i32 ldc1 = p1 > 0 ? p1 : 1;
    i32 ldd1 = p1 > 0 ? p1 : 1;
    i32 lda2 = n2 > 0 ? n2 : 1;
    i32 ldb2 = n2 > 0 ? n2 : 1;
    i32 ldc2 = p2 > 0 ? p2 : 1;
    i32 ldd2 = p2 > 0 ? p2 : 1;

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p2 > 0 ? p2 : 1;
    i32 ldd = p2 > 0 ? p2 : 1;

    i32 maxdim = n1;
    if (m1 > maxdim) maxdim = m1;
    if (n2 > maxdim) maxdim = n2;
    if (p2 > maxdim) maxdim = p2;
    i32 ldwork = (over == 'O') ? ((1 > p1 * maxdim) ? 1 : p1 * maxdim) : 1;

    f64 *a = (f64*)malloc(lda * n * sizeof(f64));
    f64 *b = (f64*)malloc(ldb * m1 * sizeof(f64));
    f64 *c = (f64*)malloc(ldc * n * sizeof(f64));
    f64 *d = (f64*)malloc(ldd * m1 * sizeof(f64));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!a || !b || !c || !d || !dwork) {
        free(a);
        free(b);
        free(c);
        free(d);
        free(dwork);
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b2_array);
        Py_DECREF(c2_array);
        Py_DECREF(d2_array);
        PyErr_NoMemory();
        return NULL;
    }

    const f64 *a1_data = (const f64*)PyArray_DATA(a1_array);
    const f64 *b1_data = (const f64*)PyArray_DATA(b1_array);
    const f64 *c1_data = (const f64*)PyArray_DATA(c1_array);
    const f64 *d1_data = (const f64*)PyArray_DATA(d1_array);
    const f64 *a2_data = (const f64*)PyArray_DATA(a2_array);
    const f64 *b2_data = (const f64*)PyArray_DATA(b2_array);
    const f64 *c2_data = (const f64*)PyArray_DATA(c2_array);
    const f64 *d2_data = (const f64*)PyArray_DATA(d2_array);

    i32 n_out;
    i32 info = ab05md(uplo, over, n1, m1, p1, n2, p2,
                      a1_data, lda1, b1_data, ldb1, c1_data, ldc1, d1_data, ldd1,
                      a2_data, lda2, b2_data, ldb2, c2_data, ldc2, d2_data, ldd2,
                      &n_out, a, lda, b, ldb, c, ldc, d, ldd,
                      dwork, ldwork);

    free(dwork);

    Py_DECREF(a1_array);
    Py_DECREF(b1_array);
    Py_DECREF(c1_array);
    Py_DECREF(d1_array);
    Py_DECREF(a2_array);
    Py_DECREF(b2_array);
    Py_DECREF(c2_array);
    Py_DECREF(d2_array);

    npy_intp a_dims[2] = {n, n};
    npy_intp a_strides[2] = {sizeof(f64), n * sizeof(f64)};
    PyObject *a_out = PyArray_New(&PyArray_Type, 2, a_dims, NPY_DOUBLE, a_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!a_out) {
        free(a);
        free(b);
        free(c);
        free(d);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)a_out), a, (size_t)lda * n * sizeof(f64));
    free(a);

    npy_intp b_dims[2] = {n, m1};
    npy_intp b_strides[2] = {sizeof(f64), n * sizeof(f64)};
    PyObject *b_out = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!b_out) {
        Py_DECREF(a_out);
        free(b);
        free(c);
        free(d);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)b_out), b, (size_t)n * m1 * sizeof(f64));
    free(b);

    npy_intp c_dims[2] = {p2, n};
    npy_intp c_strides[2] = {sizeof(f64), p2 * sizeof(f64)};
    PyObject *c_out = PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE, c_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!c_out) {
        Py_DECREF(a_out);
        Py_DECREF(b_out);
        free(c);
        free(d);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)c_out), c, (size_t)p2 * n * sizeof(f64));
    free(c);

    npy_intp d_dims[2] = {p2, m1};
    npy_intp d_strides[2] = {sizeof(f64), p2 * sizeof(f64)};
    PyObject *d_out = PyArray_New(&PyArray_Type, 2, d_dims, NPY_DOUBLE, d_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!d_out) {
        Py_DECREF(a_out);
        Py_DECREF(b_out);
        Py_DECREF(c_out);
        free(d);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)d_out), d, (size_t)p2 * m1 * sizeof(f64));
    free(d);

    PyObject *result = Py_BuildValue("OOOOii", a_out, b_out, c_out, d_out, n_out, info);
    Py_DECREF(a_out);
    Py_DECREF(b_out);
    Py_DECREF(c_out);
    Py_DECREF(d_out);

    return result;
}



/* Python wrapper for ab05nd */
PyObject* py_ab05nd(PyObject* self, PyObject* args) {
    const char *over_str;
    f64 alpha;
    PyObject *a1_obj, *b1_obj, *c1_obj, *d1_obj;
    PyObject *a2_obj, *b2_obj, *c2_obj, *d2_obj;

    if (!PyArg_ParseTuple(args, "sdOOOOOOOO",
                          &over_str, &alpha,
                          &a1_obj, &b1_obj, &c1_obj, &d1_obj,
                          &a2_obj, &b2_obj, &c2_obj, &d2_obj)) {
        return NULL;
    }

    char over = (char)toupper((unsigned char)over_str[0]);

    if (over != 'N' && over != 'O') {
        PyErr_SetString(PyExc_ValueError, "over must be 'N' or 'O'");
        return NULL;
    }

    PyArrayObject *a1_array = (PyArrayObject*)PyArray_FROM_OTF(
        a1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!a1_array) return NULL;

    PyArrayObject *b1_array = (PyArrayObject*)PyArray_FROM_OTF(
        b1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!b1_array) {
        Py_DECREF(a1_array);
        return NULL;
    }

    PyArrayObject *c1_array = (PyArrayObject*)PyArray_FROM_OTF(
        c1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!c1_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        return NULL;
    }

    PyArrayObject *d1_array = (PyArrayObject*)PyArray_FROM_OTF(
        d1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!d1_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        return NULL;
    }

    PyArrayObject *a2_array = (PyArrayObject*)PyArray_FROM_OTF(
        a2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!a2_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        return NULL;
    }

    PyArrayObject *b2_array = (PyArrayObject*)PyArray_FROM_OTF(
        b2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!b2_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        return NULL;
    }

    PyArrayObject *c2_array = (PyArrayObject*)PyArray_FROM_OTF(
        c2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!c2_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b2_array);
        return NULL;
    }

    PyArrayObject *d2_array = (PyArrayObject*)PyArray_FROM_OTF(
        d2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!d2_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b2_array);
        Py_DECREF(c2_array);
        return NULL;
    }

    npy_intp *a1_dims = PyArray_DIMS(a1_array);
    npy_intp *b1_dims = PyArray_DIMS(b1_array);
    npy_intp *c1_dims = PyArray_DIMS(c1_array);
    npy_intp *a2_dims = PyArray_DIMS(a2_array);
    npy_intp *b2_dims = PyArray_DIMS(b2_array);
    npy_intp *c2_dims = PyArray_DIMS(c2_array);
    npy_intp *d2_dims = PyArray_DIMS(d2_array);

    i32 n1 = PyArray_NDIM(a1_array) >= 1 ? (i32)a1_dims[0] : 0;
    i32 m1 = PyArray_NDIM(b1_array) >= 2 ? (i32)b1_dims[1] : (PyArray_NDIM(b1_array) == 1 ? 1 : 0);
    i32 p1 = PyArray_NDIM(c1_array) >= 1 ? (i32)c1_dims[0] : 0;
    i32 n2 = PyArray_NDIM(a2_array) >= 1 ? (i32)a2_dims[0] : 0;

    i32 b2_cols = PyArray_NDIM(b2_array) >= 2 ? (i32)b2_dims[1] : (PyArray_NDIM(b2_array) == 1 ? 1 : 0);
    i32 d2_cols = PyArray_NDIM(d2_array) >= 2 ? (i32)d2_dims[1] : (PyArray_NDIM(d2_array) == 1 ? 1 : 0);
    if (b2_cols != p1 || d2_cols != p1) {
        PyErr_SetString(PyExc_ValueError, "P1 dimension mismatch: C1 rows must equal B2 cols and D2 cols");
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b2_array);
        Py_DECREF(c2_array);
        Py_DECREF(d2_array);
        return NULL;
    }

    i32 c2_rows = PyArray_NDIM(c2_array) >= 1 ? (i32)c2_dims[0] : 0;
    i32 d2_rows = PyArray_NDIM(d2_array) >= 1 ? (i32)d2_dims[0] : 0;
    if (c2_rows != m1 || d2_rows != m1) {
        PyErr_SetString(PyExc_ValueError, "M1 dimension mismatch: B1 cols must equal C2 rows and D2 rows");
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b2_array);
        Py_DECREF(c2_array);
        Py_DECREF(d2_array);
        return NULL;
    }

    i32 n = n1 + n2;
    i32 lda1 = n1 > 0 ? n1 : 1;
    i32 ldb1 = n1 > 0 ? n1 : 1;
    i32 ldc1 = p1 > 0 ? p1 : 1;
    i32 ldd1 = p1 > 0 ? p1 : 1;
    i32 lda2 = n2 > 0 ? n2 : 1;
    i32 ldb2 = n2 > 0 ? n2 : 1;
    i32 ldc2 = m1 > 0 ? m1 : 1;
    i32 ldd2 = m1 > 0 ? m1 : 1;

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p1 > 0 ? p1 : 1;
    i32 ldd = p1 > 0 ? p1 : 1;

    i32 ldw = p1 * p1;
    if (m1 * m1 > ldw) ldw = m1 * m1;
    if (n1 * p1 > ldw) ldw = n1 * p1;
    if (over == 'O') {
        if (m1 > n * n2) {
            i32 ldw2 = m1 * (m1 + 1);
            if (ldw2 > ldw) ldw = ldw2;
        }
        ldw = n1 * p1 + ldw;
    }
    if (ldw < 1) ldw = 1;
    i32 ldwork = ldw;

    f64 *a = (f64*)malloc(lda * n * sizeof(f64));
    f64 *b = (f64*)malloc(ldb * m1 * sizeof(f64));
    f64 *c = (f64*)malloc(ldc * n * sizeof(f64));
    f64 *d = (f64*)malloc(ldd * m1 * sizeof(f64));
    i32 *iwork = (i32*)malloc((p1 > 0 ? p1 : 1) * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!a || !b || !c || !d || !iwork || !dwork) {
        free(a);
        free(b);
        free(c);
        free(d);
        free(iwork);
        free(dwork);
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b2_array);
        Py_DECREF(c2_array);
        Py_DECREF(d2_array);
        PyErr_NoMemory();
        return NULL;
    }

    const f64 *a1_data = (const f64*)PyArray_DATA(a1_array);
    const f64 *b1_data = (const f64*)PyArray_DATA(b1_array);
    const f64 *c1_data = (const f64*)PyArray_DATA(c1_array);
    const f64 *d1_data = (const f64*)PyArray_DATA(d1_array);
    const f64 *a2_data = (const f64*)PyArray_DATA(a2_array);
    const f64 *b2_data = (const f64*)PyArray_DATA(b2_array);
    const f64 *c2_data = (const f64*)PyArray_DATA(c2_array);
    const f64 *d2_data = (const f64*)PyArray_DATA(d2_array);

    i32 n_out;
    i32 info = ab05nd(over, n1, m1, p1, n2, alpha,
                      a1_data, lda1, b1_data, ldb1, c1_data, ldc1, d1_data, ldd1,
                      a2_data, lda2, b2_data, ldb2, c2_data, ldc2, d2_data, ldd2,
                      &n_out, a, lda, b, ldb, c, ldc, d, ldd,
                      iwork, dwork, ldwork);

    free(iwork);
    free(dwork);

    Py_DECREF(a1_array);
    Py_DECREF(b1_array);
    Py_DECREF(c1_array);
    Py_DECREF(d1_array);
    Py_DECREF(a2_array);
    Py_DECREF(b2_array);
    Py_DECREF(c2_array);
    Py_DECREF(d2_array);

    npy_intp a_dims[2] = {n, n};
    npy_intp a_strides[2] = {sizeof(f64), n * sizeof(f64)};
    PyObject *a_out = PyArray_New(&PyArray_Type, 2, a_dims, NPY_DOUBLE, a_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!a_out) {
        free(a);
        free(b);
        free(c);
        free(d);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)a_out), a, (size_t)lda * n * sizeof(f64));
    free(a);

    npy_intp b_dims[2] = {n, m1};
    npy_intp b_strides[2] = {sizeof(f64), n * sizeof(f64)};
    PyObject *b_out = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!b_out) {
        Py_DECREF(a_out);
        free(b);
        free(c);
        free(d);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)b_out), b, (size_t)n * m1 * sizeof(f64));
    free(b);

    npy_intp c_dims[2] = {p1, n};
    npy_intp c_strides[2] = {sizeof(f64), p1 * sizeof(f64)};
    PyObject *c_out = PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE, c_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!c_out) {
        Py_DECREF(a_out);
        Py_DECREF(b_out);
        free(c);
        free(d);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)c_out), c, (size_t)p1 * n * sizeof(f64));
    free(c);

    npy_intp d_dims[2] = {p1, m1};
    npy_intp d_strides[2] = {sizeof(f64), p1 * sizeof(f64)};
    PyObject *d_out = PyArray_New(&PyArray_Type, 2, d_dims, NPY_DOUBLE, d_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!d_out) {
        Py_DECREF(a_out);
        Py_DECREF(b_out);
        Py_DECREF(c_out);
        free(d);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)d_out), d, (size_t)p1 * m1 * sizeof(f64));
    free(d);

    PyObject *result = Py_BuildValue("OOOOii", a_out, b_out, c_out, d_out, n_out, info);
    Py_DECREF(a_out);
    Py_DECREF(b_out);
    Py_DECREF(c_out);
    Py_DECREF(d_out);

    return result;
}



/* Python wrapper for ab05od */
PyObject* py_ab05od(PyObject* self, PyObject* args) {
    const char *over_str;
    f64 alpha;
    PyObject *a1_obj, *b1_obj, *c1_obj, *d1_obj;
    PyObject *a2_obj, *b2_obj, *c2_obj, *d2_obj;

    if (!PyArg_ParseTuple(args, "sOOOOOOOOd",
                          &over_str,
                          &a1_obj, &b1_obj, &c1_obj, &d1_obj,
                          &a2_obj, &b2_obj, &c2_obj, &d2_obj,
                          &alpha)) {
        return NULL;
    }

    char over = (char)toupper((unsigned char)over_str[0]);

    if (over != 'N' && over != 'O') {
        PyErr_SetString(PyExc_ValueError, "over must be 'N' or 'O'");
        return NULL;
    }

    PyArrayObject *a1_array = (PyArrayObject*)PyArray_FROM_OTF(
        a1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!a1_array) return NULL;

    PyArrayObject *b1_array = (PyArrayObject*)PyArray_FROM_OTF(
        b1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!b1_array) {
        Py_DECREF(a1_array);
        return NULL;
    }

    PyArrayObject *c1_array = (PyArrayObject*)PyArray_FROM_OTF(
        c1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!c1_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        return NULL;
    }

    PyArrayObject *d1_array = (PyArrayObject*)PyArray_FROM_OTF(
        d1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!d1_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        return NULL;
    }

    PyArrayObject *a2_array = (PyArrayObject*)PyArray_FROM_OTF(
        a2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!a2_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        return NULL;
    }

    PyArrayObject *b2_array = (PyArrayObject*)PyArray_FROM_OTF(
        b2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!b2_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        return NULL;
    }

    PyArrayObject *c2_array = (PyArrayObject*)PyArray_FROM_OTF(
        c2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!c2_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b2_array);
        return NULL;
    }

    PyArrayObject *d2_array = (PyArrayObject*)PyArray_FROM_OTF(
        d2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!d2_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b2_array);
        Py_DECREF(c2_array);
        return NULL;
    }

    npy_intp *a1_dims = PyArray_DIMS(a1_array);
    npy_intp *b1_dims = PyArray_DIMS(b1_array);
    npy_intp *c1_dims = PyArray_DIMS(c1_array);
    npy_intp *a2_dims = PyArray_DIMS(a2_array);
    npy_intp *b2_dims = PyArray_DIMS(b2_array);
    npy_intp *c2_dims = PyArray_DIMS(c2_array);

    i32 n1 = PyArray_NDIM(a1_array) >= 1 ? (i32)a1_dims[0] : 0;
    i32 m1 = PyArray_NDIM(b1_array) >= 2 ? (i32)b1_dims[1] : (PyArray_NDIM(b1_array) == 1 ? 1 : 0);
    i32 p1 = PyArray_NDIM(c1_array) >= 1 ? (i32)c1_dims[0] : 0;
    i32 n2 = PyArray_NDIM(a2_array) >= 1 ? (i32)a2_dims[0] : 0;
    i32 m2 = PyArray_NDIM(b2_array) >= 2 ? (i32)b2_dims[1] : (PyArray_NDIM(b2_array) == 1 ? 1 : 0);

    i32 p1_c2 = PyArray_NDIM(c2_array) >= 1 ? (i32)c2_dims[0] : 0;
    if (n2 > 0 && p1_c2 != p1) {
        PyErr_SetString(PyExc_ValueError, "P1 dimension mismatch: C1 rows must equal C2 rows");
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b2_array);
        Py_DECREF(c2_array);
        Py_DECREF(d2_array);
        return NULL;
    }

    i32 n = n1 + n2;
    i32 m = m1 + m2;
    i32 lda1 = n1 > 0 ? n1 : 1;
    i32 ldb1 = n1 > 0 ? n1 : 1;
    i32 ldc1 = p1 > 0 ? p1 : 1;
    i32 ldd1 = p1 > 0 ? p1 : 1;
    i32 lda2 = n2 > 0 ? n2 : 1;
    i32 ldb2 = n2 > 0 ? n2 : 1;
    i32 ldc2 = p1 > 0 ? p1 : 1;
    i32 ldd2 = p1 > 0 ? p1 : 1;

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p1 > 0 ? p1 : 1;
    i32 ldd = p1 > 0 ? p1 : 1;

    f64 *a = (f64*)malloc((lda * n > 0 ? lda * n : 1) * sizeof(f64));
    f64 *b = (f64*)malloc((ldb * m > 0 ? ldb * m : 1) * sizeof(f64));
    f64 *c = (f64*)malloc((ldc * n > 0 ? ldc * n : 1) * sizeof(f64));
    f64 *d = (f64*)malloc((ldd * m > 0 ? ldd * m : 1) * sizeof(f64));

    if (!a || !b || !c || !d) {
        free(a);
        free(b);
        free(c);
        free(d);
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b2_array);
        Py_DECREF(c2_array);
        Py_DECREF(d2_array);
        PyErr_NoMemory();
        return NULL;
    }

    const f64 *a1_data = (const f64*)PyArray_DATA(a1_array);
    const f64 *b1_data = (const f64*)PyArray_DATA(b1_array);
    const f64 *c1_data = (const f64*)PyArray_DATA(c1_array);
    const f64 *d1_data = (const f64*)PyArray_DATA(d1_array);
    const f64 *a2_data = (const f64*)PyArray_DATA(a2_array);
    const f64 *b2_data = (const f64*)PyArray_DATA(b2_array);
    const f64 *c2_data = (const f64*)PyArray_DATA(c2_array);
    const f64 *d2_data = (const f64*)PyArray_DATA(d2_array);

    i32 n_out, m_out;
    i32 info;

    if (n1 == 0 && n2 == 0) {
        /* Manual calculation for zero-state systems to avoid potential issues in AB05OD */
        n_out = 0;
        m_out = m1 + m2;
        info = 0;

        /* D = [D1, alpha*D2] */
        if (p1 > 0) {
            /* Copy D1 */
            if (m1 > 0) {
                memcpy(d, d1_data, (size_t)ldd * m1 * sizeof(f64));
            }
            /* Copy alpha*D2 */
            if (m2 > 0) {
                for (i32 j = 0; j < m2; j++) {
                    for (i32 i = 0; i < p1; i++) {
                        d[i + (m1 + j) * ldd] = alpha * d2_data[i + j * ldd2];
                    }
                }
            }
        }
    } else {
        info = ab05od('N', n1, m1, p1, n2, m2, alpha,
                      a1_data, lda1, b1_data, ldb1, c1_data, ldc1, d1_data, ldd1,
                      a2_data, lda2, b2_data, ldb2, c2_data, ldc2, d2_data, ldd2,
                      &n_out, &m_out, a, lda, b, ldb, c, ldc, d, ldd);
    }
    (void)over;

    Py_DECREF(a1_array);
    Py_DECREF(b1_array);
    Py_DECREF(c1_array);
    Py_DECREF(d1_array);
    Py_DECREF(a2_array);
    Py_DECREF(b2_array);
    Py_DECREF(c2_array);
    Py_DECREF(d2_array);

    npy_intp a_dims[2] = {n, n};
    npy_intp a_strides[2] = {sizeof(f64), n * sizeof(f64)};
    PyObject *a_out = PyArray_New(&PyArray_Type, 2, a_dims, NPY_DOUBLE, a_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!a_out) {
        free(a);
        free(b);
        free(c);
        free(d);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)a_out), a, (size_t)lda * n * sizeof(f64));
    free(a);

    npy_intp b_dims[2] = {n, m};
    npy_intp b_strides[2] = {sizeof(f64), n * sizeof(f64)};
    PyObject *b_out = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!b_out) {
        Py_DECREF(a_out);
        free(b);
        free(c);
        free(d);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)b_out), b, (size_t)n * m * sizeof(f64));
    free(b);

    npy_intp c_dims[2] = {p1, n};
    npy_intp c_strides[2] = {sizeof(f64), p1 * sizeof(f64)};
    PyObject *c_out = PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE, c_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!c_out) {
        Py_DECREF(a_out);
        Py_DECREF(b_out);
        free(c);
        free(d);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)c_out), c, (size_t)ldc * n * sizeof(f64));
    free(c);

    npy_intp d_dims[2] = {p1, m};
    npy_intp d_strides[2] = {sizeof(f64), p1 * sizeof(f64)};
    PyObject *d_out = PyArray_New(&PyArray_Type, 2, d_dims, NPY_DOUBLE, d_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!d_out) {
        Py_DECREF(a_out);
        Py_DECREF(b_out);
        Py_DECREF(c_out);
        free(d);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)d_out), d, (size_t)ldd * m * sizeof(f64));
    free(d);

    PyObject *result = Py_BuildValue("OOOOiii", a_out, b_out, c_out, d_out, n_out, m_out, info);
    Py_DECREF(a_out);
    Py_DECREF(b_out);
    Py_DECREF(c_out);
    Py_DECREF(d_out);

    return result;
}



/* Python wrapper for ab05pd */
PyObject* py_ab05pd(PyObject* self, PyObject* args) {
    int n1, m, p, n2;
    f64 alpha;
    PyObject *a1_obj, *b1_obj, *c1_obj, *d1_obj;
    PyObject *a2_obj, *b2_obj, *c2_obj, *d2_obj;

    if (!PyArg_ParseTuple(args, "iiiidOOOOOOOO",
                          &n1, &m, &p, &n2, &alpha,
                          &a1_obj, &b1_obj, &c1_obj, &d1_obj,
                          &a2_obj, &b2_obj, &c2_obj, &d2_obj)) {
        return NULL;
    }

    PyArrayObject *a1_array = (PyArrayObject*)PyArray_FROM_OTF(
        a1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!a1_array) return NULL;

    PyArrayObject *b1_array = (PyArrayObject*)PyArray_FROM_OTF(
        b1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!b1_array) {
        Py_DECREF(a1_array);
        return NULL;
    }

    PyArrayObject *c1_array = (PyArrayObject*)PyArray_FROM_OTF(
        c1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!c1_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        return NULL;
    }

    PyArrayObject *d1_array = (PyArrayObject*)PyArray_FROM_OTF(
        d1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!d1_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        return NULL;
    }

    PyArrayObject *a2_array = (PyArrayObject*)PyArray_FROM_OTF(
        a2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!a2_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        return NULL;
    }

    PyArrayObject *b2_array = (PyArrayObject*)PyArray_FROM_OTF(
        b2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!b2_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        return NULL;
    }

    PyArrayObject *c2_array = (PyArrayObject*)PyArray_FROM_OTF(
        c2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!c2_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b2_array);
        return NULL;
    }

    PyArrayObject *d2_array = (PyArrayObject*)PyArray_FROM_OTF(
        d2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!d2_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b2_array);
        Py_DECREF(c2_array);
        return NULL;
    }

    i32 n = n1 + n2;
    i32 lda1 = n1 > 0 ? n1 : 1;
    i32 ldb1 = n1 > 0 ? n1 : 1;
    i32 ldc1 = p > 0 ? p : 1;
    i32 ldd1 = p > 0 ? p : 1;
    i32 lda2 = n2 > 0 ? n2 : 1;
    i32 ldb2 = n2 > 0 ? n2 : 1;
    i32 ldc2 = p > 0 ? p : 1;
    i32 ldd2 = p > 0 ? p : 1;

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldd = p > 0 ? p : 1;

    f64 *a = (f64*)malloc((lda * n > 0 ? lda * n : 1) * sizeof(f64));
    f64 *b = (f64*)malloc((ldb * m > 0 ? ldb * m : 1) * sizeof(f64));
    f64 *c = (f64*)malloc((ldc * n > 0 ? ldc * n : 1) * sizeof(f64));
    f64 *d = (f64*)malloc((ldd * m > 0 ? ldd * m : 1) * sizeof(f64));

    if (!a || !b || !c || !d) {
        free(a);
        free(b);
        free(c);
        free(d);
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b2_array);
        Py_DECREF(c2_array);
        Py_DECREF(d2_array);
        PyErr_NoMemory();
        return NULL;
    }

    const f64 *a1_data = (const f64*)PyArray_DATA(a1_array);
    const f64 *b1_data = (const f64*)PyArray_DATA(b1_array);
    const f64 *c1_data = (const f64*)PyArray_DATA(c1_array);
    const f64 *d1_data = (const f64*)PyArray_DATA(d1_array);
    const f64 *a2_data = (const f64*)PyArray_DATA(a2_array);
    const f64 *b2_data = (const f64*)PyArray_DATA(b2_array);
    const f64 *c2_data = (const f64*)PyArray_DATA(c2_array);
    const f64 *d2_data = (const f64*)PyArray_DATA(d2_array);

    i32 n_out;
    i32 info = ab05pd('N', n1, m, p, n2, alpha,
                      a1_data, lda1, b1_data, ldb1, c1_data, ldc1, d1_data, ldd1,
                      a2_data, lda2, b2_data, ldb2, c2_data, ldc2, d2_data, ldd2,
                      &n_out, a, lda, b, ldb, c, ldc, d, ldd);

    Py_DECREF(a1_array);
    Py_DECREF(b1_array);
    Py_DECREF(c1_array);
    Py_DECREF(d1_array);
    Py_DECREF(a2_array);
    Py_DECREF(b2_array);
    Py_DECREF(c2_array);
    Py_DECREF(d2_array);

    npy_intp a_dims[2] = {n, n};
    npy_intp a_strides[2] = {sizeof(f64), n * sizeof(f64)};
    PyObject *a_out = PyArray_New(&PyArray_Type, 2, a_dims, NPY_DOUBLE, a_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!a_out) {
        free(a);
        free(b);
        free(c);
        free(d);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)a_out), a, (size_t)lda * n * sizeof(f64));
    free(a);

    npy_intp b_dims[2] = {n, m};
    npy_intp b_strides[2] = {sizeof(f64), n * sizeof(f64)};
    PyObject *b_out = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!b_out) {
        Py_DECREF(a_out);
        free(b);
        free(c);
        free(d);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)b_out), b, (size_t)n * m * sizeof(f64));
    free(b);

    npy_intp c_dims[2] = {p, n};
    npy_intp c_strides[2] = {sizeof(f64), p * sizeof(f64)};
    PyObject *c_out = PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE, c_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!c_out) {
        Py_DECREF(a_out);
        Py_DECREF(b_out);
        free(c);
        free(d);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)c_out), c, (size_t)ldc * n * sizeof(f64));
    free(c);

    npy_intp d_dims[2] = {p, m};
    npy_intp d_strides[2] = {sizeof(f64), p * sizeof(f64)};
    PyObject *d_out = PyArray_New(&PyArray_Type, 2, d_dims, NPY_DOUBLE, d_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!d_out) {
        Py_DECREF(a_out);
        Py_DECREF(b_out);
        Py_DECREF(c_out);
        free(d);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)d_out), d, (size_t)ldd * m * sizeof(f64));
    free(d);

    PyObject *result = Py_BuildValue("iOOOOi", n_out, a_out, b_out, c_out, d_out, info);
    Py_DECREF(a_out);
    Py_DECREF(b_out);
    Py_DECREF(c_out);
    Py_DECREF(d_out);

    return result;
}



/* Python wrapper for ab05qd */
PyObject* py_ab05qd(PyObject* self, PyObject* args) {
    const char *over_str;
    PyObject *a1_obj, *b1_obj, *c1_obj, *d1_obj;
    PyObject *a2_obj, *b2_obj, *c2_obj, *d2_obj;

    if (!PyArg_ParseTuple(args, "sOOOOOOOO",
                          &over_str,
                          &a1_obj, &b1_obj, &c1_obj, &d1_obj,
                          &a2_obj, &b2_obj, &c2_obj, &d2_obj)) {
        return NULL;
    }

    char over = (char)toupper((unsigned char)over_str[0]);

    if (over != 'N' && over != 'O') {
        PyErr_SetString(PyExc_ValueError, "over must be 'N' or 'O'");
        return NULL;
    }

    PyArrayObject *a1_array = (PyArrayObject*)PyArray_FROM_OTF(
        a1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!a1_array) return NULL;

    PyArrayObject *b1_array = (PyArrayObject*)PyArray_FROM_OTF(
        b1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!b1_array) {
        Py_DECREF(a1_array);
        return NULL;
    }

    PyArrayObject *c1_array = (PyArrayObject*)PyArray_FROM_OTF(
        c1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!c1_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        return NULL;
    }

    PyArrayObject *d1_array = (PyArrayObject*)PyArray_FROM_OTF(
        d1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!d1_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        return NULL;
    }

    PyArrayObject *a2_array = (PyArrayObject*)PyArray_FROM_OTF(
        a2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!a2_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        return NULL;
    }

    PyArrayObject *b2_array = (PyArrayObject*)PyArray_FROM_OTF(
        b2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!b2_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        return NULL;
    }

    PyArrayObject *c2_array = (PyArrayObject*)PyArray_FROM_OTF(
        c2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!c2_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b2_array);
        return NULL;
    }

    PyArrayObject *d2_array = (PyArrayObject*)PyArray_FROM_OTF(
        d2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!d2_array) {
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b2_array);
        Py_DECREF(c2_array);
        return NULL;
    }

    npy_intp *a1_dims = PyArray_DIMS(a1_array);
    npy_intp *b1_dims = PyArray_DIMS(b1_array);
    npy_intp *c1_dims = PyArray_DIMS(c1_array);
    npy_intp *d1_dims = PyArray_DIMS(d1_array);
    npy_intp *a2_dims = PyArray_DIMS(a2_array);
    npy_intp *b2_dims = PyArray_DIMS(b2_array);
    npy_intp *c2_dims = PyArray_DIMS(c2_array);
    npy_intp *d2_dims = PyArray_DIMS(d2_array);

    i32 n1 = PyArray_NDIM(a1_array) >= 1 ? (i32)a1_dims[0] : 0;
    i32 m1 = PyArray_NDIM(b1_array) >= 2 ? (i32)b1_dims[1] : (PyArray_NDIM(b1_array) == 1 ? 1 : 0);
    i32 p1 = PyArray_NDIM(c1_array) >= 1 ? (i32)c1_dims[0] : 0;
    i32 n2 = PyArray_NDIM(a2_array) >= 1 ? (i32)a2_dims[0] : 0;
    i32 m2 = PyArray_NDIM(b2_array) >= 2 ? (i32)b2_dims[1] : (PyArray_NDIM(b2_array) == 1 ? 1 : 0);
    i32 p2 = PyArray_NDIM(c2_array) >= 1 ? (i32)c2_dims[0] : 0;

    (void)d1_dims;
    (void)d2_dims;

    i32 n = n1 + n2;
    i32 m = m1 + m2;
    i32 p = p1 + p2;
    i32 lda1 = n1 > 0 ? n1 : 1;
    i32 ldb1 = n1 > 0 ? n1 : 1;
    i32 ldc1 = p1 > 0 ? p1 : 1;
    i32 ldd1 = p1 > 0 ? p1 : 1;
    i32 lda2 = n2 > 0 ? n2 : 1;
    i32 ldb2 = n2 > 0 ? n2 : 1;
    i32 ldc2 = p2 > 0 ? p2 : 1;
    i32 ldd2 = p2 > 0 ? p2 : 1;

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldd = p > 0 ? p : 1;

    f64 *a = (f64*)malloc(lda * n * sizeof(f64));
    f64 *b = (f64*)malloc(ldb * m * sizeof(f64));
    f64 *c = (f64*)malloc(ldc * n * sizeof(f64));
    f64 *d = (f64*)malloc(ldd * m * sizeof(f64));

    if (!a || !b || !c || !d) {
        free(a);
        free(b);
        free(c);
        free(d);
        Py_DECREF(a1_array);
        Py_DECREF(b1_array);
        Py_DECREF(c1_array);
        Py_DECREF(d1_array);
        Py_DECREF(a2_array);
        Py_DECREF(b2_array);
        Py_DECREF(c2_array);
        Py_DECREF(d2_array);
        PyErr_NoMemory();
        return NULL;
    }

    const f64 *a1_data = (const f64*)PyArray_DATA(a1_array);
    const f64 *b1_data = (const f64*)PyArray_DATA(b1_array);
    const f64 *c1_data = (const f64*)PyArray_DATA(c1_array);
    const f64 *d1_data = (const f64*)PyArray_DATA(d1_array);
    const f64 *a2_data = (const f64*)PyArray_DATA(a2_array);
    const f64 *b2_data = (const f64*)PyArray_DATA(b2_array);
    const f64 *c2_data = (const f64*)PyArray_DATA(c2_array);
    const f64 *d2_data = (const f64*)PyArray_DATA(d2_array);

    i32 n_out, m_out, p_out;
    i32 info = ab05qd('N', n1, m1, p1, n2, m2, p2,
                      a1_data, lda1, b1_data, ldb1, c1_data, ldc1, d1_data, ldd1,
                      a2_data, lda2, b2_data, ldb2, c2_data, ldc2, d2_data, ldd2,
                      &n_out, &m_out, &p_out, a, lda, b, ldb, c, ldc, d, ldd);
    (void)over;

    Py_DECREF(a1_array);
    Py_DECREF(b1_array);
    Py_DECREF(c1_array);
    Py_DECREF(d1_array);
    Py_DECREF(a2_array);
    Py_DECREF(b2_array);
    Py_DECREF(c2_array);
    Py_DECREF(d2_array);

    npy_intp a_dims[2] = {n, n};
    npy_intp a_strides[2] = {sizeof(f64), n * sizeof(f64)};
    PyObject *a_out = PyArray_New(&PyArray_Type, 2, a_dims, NPY_DOUBLE, a_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!a_out) {
        free(a);
        free(b);
        free(c);
        free(d);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)a_out), a, (size_t)lda * n * sizeof(f64));
    free(a);

    npy_intp b_dims[2] = {n, m};
    npy_intp b_strides[2] = {sizeof(f64), n * sizeof(f64)};
    PyObject *b_out = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!b_out) {
        Py_DECREF(a_out);
        free(b);
        free(c);
        free(d);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)b_out), b, (size_t)n * m * sizeof(f64));
    free(b);

    npy_intp c_dims[2] = {p, n};
    npy_intp c_strides[2] = {sizeof(f64), p * sizeof(f64)};
    PyObject *c_out = PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE, c_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!c_out) {
        Py_DECREF(a_out);
        Py_DECREF(b_out);
        free(c);
        free(d);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)c_out), c, (size_t)p * n * sizeof(f64));
    free(c);

    npy_intp d_dims[2] = {p, m};
    npy_intp d_strides[2] = {sizeof(f64), p * sizeof(f64)};
    PyObject *d_out = PyArray_New(&PyArray_Type, 2, d_dims, NPY_DOUBLE, d_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!d_out) {
        Py_DECREF(a_out);
        Py_DECREF(b_out);
        Py_DECREF(c_out);
        free(d);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)d_out), d, (size_t)p * m * sizeof(f64));
    free(d);

    PyObject *result = Py_BuildValue("OOOOiiii", a_out, b_out, c_out, d_out, n_out, m_out, p_out, info);
    Py_DECREF(a_out);
    Py_DECREF(b_out);
    Py_DECREF(c_out);
    Py_DECREF(d_out);

    return result;
}


/* Python wrapper for ab05sd */
PyObject* py_ab05sd(PyObject* self, PyObject* args) {
    const char *fbtype_str, *jobd_str;
    i32 n, m, p;
    double alpha;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *f_obj;

    if (!PyArg_ParseTuple(args, "ssiiidOOOOO", &fbtype_str, &jobd_str,
                          &n, &m, &p, &alpha,
                          &a_obj, &b_obj, &c_obj, &d_obj, &f_obj)) {
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

    PyArrayObject *f_array = (PyArrayObject*)PyArray_FROM_OTF(
        f_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!f_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }

    i32 lda = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldc = p > 1 ? p : 1;
    i32 ldd = p > 1 ? p : 1;
    i32 ldf = m > 1 ? m : 1;

    f64 *a = (f64*)PyArray_DATA(a_array);
    f64 *b = (f64*)PyArray_DATA(b_array);
    f64 *c = (f64*)PyArray_DATA(c_array);
    f64 *d = (f64*)PyArray_DATA(d_array);
    f64 *f = (f64*)PyArray_DATA(f_array);

    i32 pp4p = p * p + 4 * p;
    i32 max1m = (1 > m) ? 1 : m;
    i32 ldwork = (max1m > pp4p) ? max1m : pp4p;
    if (ldwork < 1) ldwork = 1;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    i32 iwork_size = (p > 1) ? p : 1;
    i32 *iwork = (i32*)malloc(iwork_size * sizeof(i32));
    if (!dwork || !iwork) {
        free(dwork);
        free(iwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(f_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 rcond = 0.0;
    i32 info = 0;

    ab05sd(fbtype_str, jobd_str, n, m, p, alpha, a, lda, b, ldb,
           c, ldc, d, ldd, f, ldf, &rcond, iwork, dwork, ldwork, &info);

    free(dwork);
    free(iwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    PyObject *result = Py_BuildValue("OOOOdi", a_array, b_array, c_array, d_array, rcond, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(f_array);

    return result;
}


PyObject* py_ab05rd(PyObject* self, PyObject* args) {
    const char *fbtype_str, *jobd_str;
    i32 n, m, p, mv, pz;
    double alpha, beta;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *f_obj, *k_obj, *g_obj, *h_obj;

    if (!PyArg_ParseTuple(args, "ssiiiiiddOOOOOOOO", &fbtype_str, &jobd_str,
                          &n, &m, &p, &mv, &pz, &alpha, &beta,
                          &a_obj, &b_obj, &c_obj, &d_obj, &f_obj, &k_obj, &g_obj, &h_obj)) {
        return NULL;
    }

    char jobd_upper = (char)toupper((unsigned char)jobd_str[0]);
    bool ljobd = (jobd_upper == 'D');

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

    PyArrayObject *f_array = (PyArrayObject*)PyArray_FROM_OTF(
        f_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!f_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }

    PyArrayObject *k_array = (PyArrayObject*)PyArray_FROM_OTF(
        k_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!k_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(f_array);
        return NULL;
    }

    PyArrayObject *g_array = (PyArrayObject*)PyArray_FROM_OTF(
        g_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!g_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(f_array);
        Py_DECREF(k_array);
        return NULL;
    }

    PyArrayObject *h_array = (PyArrayObject*)PyArray_FROM_OTF(
        h_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!h_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(f_array);
        Py_DECREF(k_array);
        Py_DECREF(g_array);
        return NULL;
    }

    i32 lda = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldc = p > 1 ? p : 1;
    i32 ldd = p > 1 ? p : 1;
    i32 ldf = m > 1 ? m : 1;
    i32 ldk = m > 1 ? m : 1;
    i32 ldg = m > 1 ? m : 1;
    i32 ldh = pz > 1 ? pz : 1;
    i32 ldbc = n > 1 ? n : 1;
    i32 ldcc = pz > 1 ? pz : 1;
    i32 lddc = pz > 1 ? pz : 1;

    f64 *a = (f64*)PyArray_DATA(a_array);
    f64 *b = (f64*)PyArray_DATA(b_array);
    f64 *c = (f64*)PyArray_DATA(c_array);
    f64 *d = (f64*)PyArray_DATA(d_array);
    f64 *f = (f64*)PyArray_DATA(f_array);
    f64 *k = (f64*)PyArray_DATA(k_array);
    f64 *g = (f64*)PyArray_DATA(g_array);
    f64 *h_data = (f64*)PyArray_DATA(h_array);

    i32 pmv = p * mv;
    i32 pp4p = p * p + 4 * p;
    i32 max1m = (1 > m) ? 1 : m;
    i32 ldwork;
    if (ljobd) {
        ldwork = max1m;
        if (pmv > ldwork) ldwork = pmv;
        if (pp4p > ldwork) ldwork = pp4p;
        if (ldwork < 1) ldwork = 1;
    } else {
        ldwork = max1m;
        if (ldwork < 1) ldwork = 1;
    }

    i32 iwork_size = (2*p > 1) ? 2*p : 1;

    f64 *bc = (f64*)calloc((size_t)ldbc * (size_t)(mv > 1 ? mv : 1), sizeof(f64));
    f64 *cc = (f64*)calloc((size_t)ldcc * (size_t)(n > 1 ? n : 1), sizeof(f64));
    f64 *dc = ljobd ? (f64*)calloc((size_t)lddc * (size_t)(mv > 1 ? mv : 1), sizeof(f64)) : NULL;
    f64 *dwork = (f64*)malloc((size_t)ldwork * sizeof(f64));
    i32 *iwork = (i32*)malloc((size_t)iwork_size * sizeof(i32));

    if (!bc || !cc || (ljobd && !dc) || !dwork || !iwork) {
        free(bc);
        free(cc);
        free(dc);
        free(dwork);
        free(iwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(f_array);
        Py_DECREF(k_array);
        Py_DECREF(g_array);
        Py_DECREF(h_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 rcond = 0.0;
    i32 info = 0;

    ab05rd(fbtype_str, jobd_str, n, m, p, mv, pz, alpha, beta,
           a, lda, b, ldb, c, ldc, d, ldd,
           f, ldf, k, ldk, g, ldg, h_data, ldh,
           &rcond, bc, ldbc, cc, ldcc, dc, lddc,
           iwork, dwork, ldwork, &info);

    free(dwork);
    free(iwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    npy_intp bc_dims[2] = {n, mv};
    npy_intp bc_strides[2] = {sizeof(f64), ldbc * sizeof(f64)};
    PyObject *bc_out = PyArray_New(&PyArray_Type, 2, bc_dims, NPY_DOUBLE, bc_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!bc_out) {
        free(bc);
        free(cc);
        free(dc);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(f_array);
        Py_DECREF(k_array);
        Py_DECREF(g_array);
        Py_DECREF(h_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)bc_out), bc, (size_t)n * mv * sizeof(f64));
    free(bc);

    npy_intp cc_dims[2] = {pz, n};
    npy_intp cc_strides[2] = {sizeof(f64), ldcc * sizeof(f64)};
    PyObject *cc_out = PyArray_New(&PyArray_Type, 2, cc_dims, NPY_DOUBLE, cc_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!cc_out) {
        free(cc);
        free(dc);
        Py_DECREF(bc_out);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(f_array);
        Py_DECREF(k_array);
        Py_DECREF(g_array);
        Py_DECREF(h_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)cc_out), cc, (size_t)pz * n * sizeof(f64));
    free(cc);

    PyObject *dc_out;
    if (ljobd) {
        npy_intp dc_dims[2] = {pz, mv};
        npy_intp dc_strides[2] = {sizeof(f64), lddc * sizeof(f64)};
        dc_out = PyArray_New(&PyArray_Type, 2, dc_dims, NPY_DOUBLE, dc_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!dc_out) {
            free(dc);
            Py_DECREF(bc_out);
            Py_DECREF(cc_out);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            Py_DECREF(d_array);
            Py_DECREF(f_array);
            Py_DECREF(k_array);
            Py_DECREF(g_array);
            Py_DECREF(h_array);
            return NULL;
        }
        memcpy(PyArray_DATA((PyArrayObject*)dc_out), dc, (size_t)pz * mv * sizeof(f64));
        free(dc);
    } else {
        free(dc);
        npy_intp dc_dims[2] = {pz, mv};
        dc_out = PyArray_ZEROS(2, dc_dims, NPY_DOUBLE, 1);
        if (!dc_out) {
            Py_DECREF(bc_out);
            Py_DECREF(cc_out);
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            Py_DECREF(d_array);
            Py_DECREF(f_array);
            Py_DECREF(k_array);
            Py_DECREF(g_array);
            Py_DECREF(h_array);
            return NULL;
        }
    }

    PyObject *result = Py_BuildValue("OOOOdi", a_array, bc_out, cc_out, dc_out, rcond, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(f_array);
    Py_DECREF(k_array);
    Py_DECREF(g_array);
    Py_DECREF(h_array);
    Py_DECREF(bc_out);
    Py_DECREF(cc_out);
    Py_DECREF(dc_out);

    return result;
}


/* Python wrapper for ab13md */
PyObject* py_ab13md(PyObject* self, PyObject* args, PyObject* kwds) {
    static char *kwlist[] = {"z", "nblock", "itype", "fact", "x", NULL};
    PyObject *z_obj, *nblock_obj, *itype_obj;
    const char *fact_str = "N";
    PyObject *x_obj = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|sO", kwlist,
                                      &z_obj, &nblock_obj, &itype_obj,
                                      &fact_str, &x_obj)) {
        return NULL;
    }

    char fact = (char)toupper((unsigned char)fact_str[0]);

    PyArrayObject *z_array = (PyArrayObject*)PyArray_FROM_OTF(
        z_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!z_array) return NULL;

    PyArrayObject *nblock_array = (PyArrayObject*)PyArray_FROM_OTF(
        nblock_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (!nblock_array) {
        Py_DECREF(z_array);
        return NULL;
    }

    PyArrayObject *itype_array = (PyArrayObject*)PyArray_FROM_OTF(
        itype_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (!itype_array) {
        Py_DECREF(z_array);
        Py_DECREF(nblock_array);
        return NULL;
    }

    PyArrayObject *x_array = NULL;
    if (x_obj && x_obj != Py_None) {
        x_array = (PyArrayObject*)PyArray_FROM_OTF(
            x_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!x_array) {
            Py_DECREF(z_array);
            Py_DECREF(nblock_array);
            Py_DECREF(itype_array);
            return NULL;
        }
    }

    npy_intp *z_dims = PyArray_DIMS(z_array);
    i32 n = (i32)z_dims[0];
    i32 ldz = n > 0 ? n : 1;
    i32 m = (i32)PyArray_SIZE(nblock_array);

    i32 *nblock_data = (i32*)PyArray_DATA(nblock_array);
    i32 *itype_data = (i32*)PyArray_DATA(itype_array);

    i32 mr = 0;
    for (i32 i = 0; i < m; i++) {
        if (itype_data[i] == 1) mr++;
    }
    i32 mt = m + mr - 1;
    if (mt < 1) mt = 1;

    i32 minwrk = 2*n*n*m - n*n + 9*m*m + n*m + 11*n + 33*m - 11;
    if (minwrk < 1) minwrk = 1;
    i32 minzrk = 6*n*n*m + 12*n*n + 6*m + 6*n - 3;
    if (minzrk < 1) minzrk = 1;

    i32 iwork_size = (4*m - 2 > n) ? 4*m - 2 : n;
    if (iwork_size < 1) iwork_size = 1;

    f64 *x_data_alloc = NULL;
    f64 *x_data;
    if (x_array) {
        x_data = (f64*)PyArray_DATA(x_array);
    } else {
        x_data_alloc = (f64*)calloc(mt, sizeof(f64));
        x_data = x_data_alloc;
        if (!x_data) {
            Py_DECREF(z_array);
            Py_DECREF(nblock_array);
            Py_DECREF(itype_array);
            PyErr_NoMemory();
            return NULL;
        }
    }

    f64 *d_data = (f64*)malloc(n * sizeof(f64));
    f64 *g_data = (f64*)malloc(n * sizeof(f64));
    i32 *iwork = (i32*)malloc(iwork_size * sizeof(i32));
    f64 *dwork = (f64*)malloc(minwrk * sizeof(f64));
    c128 *zwork = (c128*)malloc(minzrk * sizeof(c128));

    if (!d_data || !g_data || !iwork || !dwork || !zwork) {
        free(x_data_alloc);
        free(d_data);
        free(g_data);
        free(iwork);
        free(dwork);
        free(zwork);
        Py_DECREF(z_array);
        Py_DECREF(nblock_array);
        Py_DECREF(itype_array);
        Py_XDECREF(x_array);
        PyErr_NoMemory();
        return NULL;
    }

    c128 *z_data = (c128*)PyArray_DATA(z_array);
    f64 bound;

    i32 info = ab13md(fact, n, z_data, ldz, m, nblock_data, itype_data,
                      x_data, &bound, d_data, g_data, iwork, dwork, minwrk,
                      zwork, minzrk);

    free(iwork);
    free(dwork);
    free(zwork);

    PyArray_ResolveWritebackIfCopy(z_array);
    if (x_array) {
        PyArray_ResolveWritebackIfCopy(x_array);
    }

    npy_intp d_dims[1] = {n};
    npy_intp x_dims[1] = {mt};

    PyObject *d_out = PyArray_SimpleNew(1, d_dims, NPY_DOUBLE);
    if (!d_out) {
        free(x_data_alloc);
        free(d_data);
        free(g_data);
        Py_DECREF(z_array);
        Py_DECREF(nblock_array);
        Py_DECREF(itype_array);
        Py_XDECREF(x_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)d_out), d_data, (size_t)n * sizeof(f64));
    free(d_data);

    PyObject *g_out = PyArray_SimpleNew(1, d_dims, NPY_DOUBLE);
    if (!g_out) {
        free(x_data_alloc);
        free(g_data);
        Py_DECREF(d_out);
        Py_DECREF(z_array);
        Py_DECREF(nblock_array);
        Py_DECREF(itype_array);
        Py_XDECREF(x_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)g_out), g_data, (size_t)n * sizeof(f64));
    free(g_data);

    PyObject *x_out;
    if (x_data_alloc) {
        x_out = PyArray_SimpleNew(1, x_dims, NPY_DOUBLE);
        if (x_out) {
            memcpy(PyArray_DATA((PyArrayObject*)x_out), x_data_alloc, (size_t)mt * sizeof(f64));
            free(x_data_alloc);
            x_data_alloc = NULL;
        }
    } else {
        x_out = (PyObject*)x_array;
        Py_INCREF(x_out);
    }
    if (!x_out) {
        Py_DECREF(d_out);
        Py_DECREF(g_out);
        Py_DECREF(z_array);
        Py_DECREF(nblock_array);
        Py_DECREF(itype_array);
        Py_XDECREF(x_array);
        return NULL;
    }

    PyObject *result = Py_BuildValue("dOOOi", bound, d_out, g_out, x_out, info);

    Py_DECREF(z_array);
    Py_DECREF(nblock_array);
    Py_DECREF(itype_array);
    Py_XDECREF(x_array);
    Py_DECREF(d_out);
    Py_DECREF(g_out);
    Py_DECREF(x_out);

    return result;
}



/* Python wrapper for ab07md */
PyObject* py_ab07md(PyObject* self, PyObject* args) {
    const char *jobd_str;
    i32 n, m, p;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "siiiOOOO", &jobd_str, &n, &m, &p,
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

    i32 info = ab07md(jobd, n, m, p, a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd);

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



/* Python wrapper for ab07nd */
PyObject* py_ab07nd(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "OOOO", &a_obj, &b_obj, &c_obj, &d_obj)) {
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
    npy_intp *c_dims = PyArray_DIMS(c_array);

    i32 n = PyArray_NDIM(a_array) >= 1 ? (i32)a_dims[0] : 0;
    i32 m = PyArray_NDIM(c_array) >= 1 ? (i32)c_dims[0] : 0;

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = m > 0 ? m : 1;
    i32 ldd = m > 0 ? m : 1;

    i32 minwrk = (1 > 4 * m) ? 1 : 4 * m;
    i32 iwork_size = (2 * m > 1) ? 2 * m : 1;
    i32 ldwork = (n * m > minwrk) ? n * m : minwrk;

    i32 *iwork = (i32*)malloc(iwork_size * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
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

    f64 rcond;
    i32 info = ab07nd(n, m, a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
                      &rcond, iwork, dwork, ldwork);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    PyObject *result = Py_BuildValue("OOOOdi", a_array, b_array, c_array, d_array, rcond, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    return result;
}



/* Python wrapper for ab09ad */
PyObject* py_ab09ad(PyObject* self, PyObject* args) {
    const char *dico_str, *job_str, *equil_str, *ordsel_str;
    int n, m, p, nr;
    double tol;
    PyObject *a_obj, *b_obj, *c_obj;

    if (!PyArg_ParseTuple(args, "ssssiiiiOOOd", &dico_str, &job_str, &equil_str,
                          &ordsel_str, &n, &m, &p, &nr, &a_obj, &b_obj, &c_obj, &tol)) {
        return NULL;
    }

    if (dico_str[0] != 'C' && dico_str[0] != 'c' &&
        dico_str[0] != 'D' && dico_str[0] != 'd') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }

    if (job_str[0] != 'B' && job_str[0] != 'b' &&
        job_str[0] != 'N' && job_str[0] != 'n') {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'B' or 'N'");
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
    if (fixord && (nr < 0 || nr > n)) {
        PyErr_SetString(PyExc_ValueError, "NR must be >= 0 and <= N when ORDSEL='F'");
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

    i32 max_nmp = n;
    if (m > max_nmp) max_nmp = m;
    if (p > max_nmp) max_nmp = p;
    i32 ldwork = n * (2 * n + max_nmp + 5) + (n * (n + 1)) / 2;
    if (ldwork < 1) ldwork = 1;

    bool bfree = (job_str[0] == 'N' || job_str[0] == 'n');
    i32 iwork_size = bfree ? n : 1;
    if (iwork_size < 1) iwork_size = 1;

    f64 *hsv = (f64*)malloc(n > 0 ? n * sizeof(f64) : sizeof(f64));
    i32 *iwork = (i32*)malloc(iwork_size * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!hsv || !iwork || !dwork) {
        free(hsv);
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    i32 nr_out = nr;
    i32 iwarn = 0;
    i32 info = 0;

    ab09ad(dico_str, job_str, equil_str, ordsel_str, n, m, p, &nr_out,
           a_data, lda, b_data, ldb, c_data, ldc, hsv, tol,
           iwork, dwork, ldwork, &iwarn, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    npy_intp hsv_dims[1] = {n > 0 ? n : 1};
    PyArrayObject *hsv_array = (PyArrayObject*)PyArray_SimpleNew(1, hsv_dims, NPY_DOUBLE);
    if (!hsv_array) {
        free(hsv);
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_NoMemory();
        return NULL;
    }
    memcpy(PyArray_DATA(hsv_array), hsv, (n > 0 ? n : 1) * sizeof(f64));

    free(hsv);
    free(iwork);
    free(dwork);

    PyObject *result = Py_BuildValue("OOOOiii", a_array, b_array, c_array,
                                     hsv_array, nr_out, iwarn, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(hsv_array);

    return result;
}



/* Python wrapper for ab09ax */
PyObject* py_ab09ax(PyObject* self, PyObject* args) {
    const char *dico_str, *job_str, *ordsel_str;
    int n, m, p, nr;
    double tol;
    PyObject *a_obj, *b_obj, *c_obj;

    if (!PyArg_ParseTuple(args, "sssiiiiOOOd", &dico_str, &job_str, &ordsel_str,
                          &n, &m, &p, &nr, &a_obj, &b_obj, &c_obj, &tol)) {
        return NULL;
    }

    if (dico_str[0] != 'C' && dico_str[0] != 'c' &&
        dico_str[0] != 'D' && dico_str[0] != 'd') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }

    if (job_str[0] != 'B' && job_str[0] != 'b' &&
        job_str[0] != 'N' && job_str[0] != 'n') {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'B' or 'N'");
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
    if (fixord && (nr < 0 || nr > n)) {
        PyErr_SetString(PyExc_ValueError, "NR must be >= 0 and <= N when ORDSEL='F'");
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
    i32 ldt = n > 0 ? n : 1;
    i32 ldti = n > 0 ? n : 1;

    i32 max_nmp = n;
    if (m > max_nmp) max_nmp = m;
    if (p > max_nmp) max_nmp = p;
    i32 ldwork = n * (max_nmp + 5) + (n * (n + 1)) / 2;
    if (ldwork < 1) ldwork = 1;

    bool bfree = (job_str[0] == 'N' || job_str[0] == 'n');
    i32 iwork_size = bfree ? n : 1;
    if (iwork_size < 1) iwork_size = 1;

    f64 *hsv = (f64*)malloc(n > 0 ? n * sizeof(f64) : sizeof(f64));
    f64 *t = (f64*)malloc((n > 0 ? n * n : 1) * sizeof(f64));
    f64 *ti = (f64*)malloc((n > 0 ? n * n : 1) * sizeof(f64));
    i32 *iwork = (i32*)malloc(iwork_size * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!hsv || !t || !ti || !iwork || !dwork) {
        free(hsv);
        free(t);
        free(ti);
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    i32 nr_out = nr;
    i32 iwarn = 0;
    i32 info = 0;

    ab09ax(dico_str, job_str, ordsel_str, n, m, p, &nr_out, a_data, lda, b_data, ldb,
           c_data, ldc, hsv, t, ldt, ti, ldti, tol, iwork, dwork, ldwork, &iwarn, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    npy_intp hsv_dims[1] = {n > 0 ? n : 1};
    PyArrayObject *hsv_array = (PyArrayObject*)PyArray_SimpleNew(1, hsv_dims, NPY_DOUBLE);
    if (!hsv_array) {
        free(hsv);
        free(t);
        free(ti);
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_NoMemory();
        return NULL;
    }
    memcpy(PyArray_DATA(hsv_array), hsv, (n > 0 ? n : 1) * sizeof(f64));

    npy_intp t_dims[2] = {n > 0 ? n : 1, n > 0 ? n : 1};
    npy_intp t_strides[2] = {sizeof(f64), (n > 0 ? n : 1) * sizeof(f64)};
    PyArrayObject *t_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, t_dims, NPY_DOUBLE,
                                                          t_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!t_array) {
        free(hsv);
        free(t);
        free(ti);
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(hsv_array);
        PyErr_NoMemory();
        return NULL;
    }
    {
        i32 actual_n = n > 0 ? n : 1;
        for (i32 j = 0; j < actual_n; j++) {
            memcpy((f64*)PyArray_DATA(t_array) + j * actual_n, t + j * ldt, (size_t)actual_n * sizeof(f64));
        }
    }
    free(t);

    npy_intp ti_dims[2] = {n > 0 ? n : 1, n > 0 ? n : 1};
    npy_intp ti_strides[2] = {sizeof(f64), (n > 0 ? n : 1) * sizeof(f64)};
    PyArrayObject *ti_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, ti_dims, NPY_DOUBLE,
                                                           ti_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!ti_array) {
        free(hsv);
        free(ti);
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(hsv_array);
        Py_DECREF(t_array);
        PyErr_NoMemory();
        return NULL;
    }
    {
        i32 actual_n = n > 0 ? n : 1;
        for (i32 j = 0; j < actual_n; j++) {
            memcpy((f64*)PyArray_DATA(ti_array) + j * actual_n, ti + j * ldti, (size_t)actual_n * sizeof(f64));
        }
    }
    free(ti);

    free(hsv);
    free(iwork);
    free(dwork);

    PyObject *result = Py_BuildValue("OOOOOOiii", a_array, b_array, c_array,
                                     hsv_array, t_array, ti_array, nr_out, iwarn, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(hsv_array);
    Py_DECREF(t_array);
    Py_DECREF(ti_array);

    return result;
}



/* Python wrapper for ab09dd */
PyObject* py_ab09dd(PyObject* self, PyObject* args) {
    const char *dico_str;
    int n, m, p, nr;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "siiiiOOOO", &dico_str, &n, &m, &p, &nr,
                          &a_obj, &b_obj, &c_obj, &d_obj)) {
        return NULL;
    }

    if (dico_str[0] != 'C' && dico_str[0] != 'c' &&
        dico_str[0] != 'D' && dico_str[0] != 'd') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
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
    if (nr < 0 || nr > n) {
        PyErr_SetString(PyExc_ValueError, "NR must be >= 0 and <= N");
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

    i32 ns = n - nr;
    i32 iwork_size = (2 * ns > 1) ? 2 * ns : 1;
    i32 dwork_size = (4 * ns > 1) ? 4 * ns : 1;

    i32 *iwork = (i32*)malloc(iwork_size * sizeof(i32));
    f64 *dwork = (f64*)malloc(dwork_size * sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
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

    f64 rcond;
    i32 info = ab09dd(dico_str, n, m, p, nr, a_data, lda, b_data, ldb,
                      c_data, ldc, d_data, ldd, &rcond, iwork, dwork);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    PyObject *result = Py_BuildValue("OOOOdi", a_array, b_array, c_array, d_array, rcond, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    return result;
}



/* Python wrapper for ab09bx */
PyObject* py_ab09bx(PyObject* self, PyObject* args) {
    const char *dico_str, *job_str, *ordsel_str;
    int n, m, p, nr_in;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    double tol1, tol2;

    if (!PyArg_ParseTuple(args, "sssiiiiOOOOdd", &dico_str, &job_str, &ordsel_str,
                          &n, &m, &p, &nr_in, &a_obj, &b_obj, &c_obj, &d_obj,
                          &tol1, &tol2)) {
        return NULL;
    }

    if (dico_str[0] != 'C' && dico_str[0] != 'c' &&
        dico_str[0] != 'D' && dico_str[0] != 'd') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }

    if (job_str[0] != 'B' && job_str[0] != 'b' &&
        job_str[0] != 'N' && job_str[0] != 'n') {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'B' or 'N'");
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
    if (fixord && (nr_in < 0 || nr_in > n)) {
        PyErr_SetString(PyExc_ValueError, "NR must be >= 0 and <= N for fixed order");
        return NULL;
    }

    if (tol2 > 0.0 && tol2 > tol1) {
        PyErr_SetString(PyExc_ValueError, "TOL2 must be <= TOL1 when TOL2 > 0");
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
    i32 ldt = n > 0 ? n : 1;
    i32 ldti = n > 0 ? n : 1;

    i32 maxnmp = n;
    if (m > maxnmp) maxnmp = m;
    if (p > maxnmp) maxnmp = p;
    i32 ldwork = n * (maxnmp + 5) + (n * (n + 1)) / 2;
    if (ldwork < 1) ldwork = 1;
    ldwork *= 2;

    i32 iwork_size = 2 * n > 1 ? 2 * n : 1;

    f64 *hsv = (f64*)malloc((size_t)(n > 0 ? n : 1) * sizeof(f64));
    f64 *t = (f64*)malloc((size_t)(ldt * (n > 0 ? n : 1)) * sizeof(f64));
    f64 *ti = (f64*)malloc((size_t)(ldti * (n > 0 ? n : 1)) * sizeof(f64));
    i32 *iwork = (i32*)malloc((size_t)iwork_size * sizeof(i32));
    f64 *dwork = (f64*)malloc((size_t)ldwork * sizeof(f64));

    if (!hsv || !t || !ti || !iwork || !dwork) {
        free(hsv);
        free(t);
        free(ti);
        free(iwork);
        free(dwork);
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

    i32 nr = nr_in;
    i32 iwarn, info;

    ab09bx(dico_str, job_str, ordsel_str, n, m, p, &nr, a_data, lda,
           b_data, ldb, c_data, ldc, d_data, ldd, hsv, t, ldt, ti, ldti,
           tol1, tol2, iwork, dwork, ldwork, &iwarn, &info);

    i32 nmin = iwork[0];

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    npy_intp hsv_dims[1] = {n > 0 ? n : 1};
    PyObject *hsv_array = PyArray_SimpleNew(1, hsv_dims, NPY_DOUBLE);
    if (!hsv_array) {
        free(hsv);
        free(t);
        free(ti);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)hsv_array), hsv, (size_t)(n > 0 ? n : 1) * sizeof(f64));
    free(hsv);

    npy_intp t_dims[2] = {ldt, n > 0 ? n : 1};
    npy_intp t_strides[2] = {sizeof(f64), ldt * sizeof(f64)};
    PyObject *t_array = PyArray_New(&PyArray_Type, 2, t_dims, NPY_DOUBLE,
                                     t_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!t_array) {
        free(t);
        free(ti);
        Py_DECREF(hsv_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)t_array), t, (size_t)ldt * (n > 0 ? n : 1) * sizeof(f64));
    free(t);

    npy_intp ti_dims[2] = {ldti, n > 0 ? n : 1};
    npy_intp ti_strides[2] = {sizeof(f64), ldti * sizeof(f64)};
    PyObject *ti_array = PyArray_New(&PyArray_Type, 2, ti_dims, NPY_DOUBLE,
                                      ti_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!ti_array) {
        free(ti);
        Py_DECREF(t_array);
        Py_DECREF(hsv_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)ti_array), ti, (size_t)ldti * (n > 0 ? n : 1) * sizeof(f64));
    free(ti);

    PyObject *result = Py_BuildValue("OOOOiOOOiii",
        a_array, b_array, c_array, d_array, nr, hsv_array, t_array, ti_array,
        nmin, iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(hsv_array);
    Py_DECREF(t_array);
    Py_DECREF(ti_array);

    return result;
}



/* Python wrapper for ab09bd */
PyObject* py_ab09bd(PyObject* self, PyObject* args) {
    const char *dico_str, *job_str, *equil_str, *ordsel_str;
    int n, m, p, nr_in;
    double tol1, tol2;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "ssssiiiiOOOOdd", &dico_str, &job_str, &equil_str,
                          &ordsel_str, &n, &m, &p, &nr_in,
                          &a_obj, &b_obj, &c_obj, &d_obj, &tol1, &tol2)) {
        return NULL;
    }

    if (dico_str[0] != 'C' && dico_str[0] != 'c' &&
        dico_str[0] != 'D' && dico_str[0] != 'd') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }

    if (job_str[0] != 'B' && job_str[0] != 'b' &&
        job_str[0] != 'N' && job_str[0] != 'n') {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'B' or 'N'");
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
    if (fixord && (nr_in < 0 || nr_in > n)) {
        PyErr_SetString(PyExc_ValueError, "NR must be >= 0 and <= N for fixed order");
        return NULL;
    }

    if (tol2 > 0.0 && tol2 > tol1) {
        PyErr_SetString(PyExc_ValueError, "TOL2 must be <= TOL1 when TOL2 > 0");
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

    i32 maxnmp = n;
    if (m > maxnmp) maxnmp = m;
    if (p > maxnmp) maxnmp = p;
    i32 ldwork = n * (2*n + maxnmp + 5) + (n * (n + 1)) / 2;
    if (ldwork < 1) ldwork = 1;
    ldwork *= 2;

    i32 iwork_size = 2 * n > 1 ? 2 * n : 1;

    f64 *hsv = (f64*)malloc((size_t)(n > 0 ? n : 1) * sizeof(f64));
    i32 *iwork = (i32*)malloc((size_t)iwork_size * sizeof(i32));
    f64 *dwork = (f64*)malloc((size_t)ldwork * sizeof(f64));

    if (!hsv || !iwork || !dwork) {
        free(hsv);
        free(iwork);
        free(dwork);
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

    i32 nr = nr_in;
    i32 iwarn, info;

    ab09bd(dico_str, job_str, equil_str, ordsel_str, n, m, p, &nr,
           a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           hsv, tol1, tol2, iwork, dwork, ldwork, &iwarn, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    npy_intp hsv_dims[1] = {n > 0 ? n : 1};
    PyObject *hsv_array = PyArray_SimpleNew(1, hsv_dims, NPY_DOUBLE);
    if (!hsv_array) {
        free(hsv);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)hsv_array), hsv, (size_t)(n > 0 ? n : 1) * sizeof(f64));
    free(hsv);

    PyObject *result = Py_BuildValue("OOOOiOii",
        a_array, b_array, c_array, d_array, nr, hsv_array, iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(hsv_array);

    return result;
}



/* Python wrapper for ab09md */
PyObject* py_ab09md(PyObject* self, PyObject* args) {
    const char *dico_str, *job_str, *equil_str, *ordsel_str;
    int n, m, p, nr_in;
    double alpha, tol;
    PyObject *a_obj, *b_obj, *c_obj;

    if (!PyArg_ParseTuple(args, "ssssiiiidOOOd", &dico_str, &job_str, &equil_str,
                          &ordsel_str, &n, &m, &p, &nr_in, &alpha,
                          &a_obj, &b_obj, &c_obj, &tol)) {
        return NULL;
    }

    if (dico_str[0] != 'C' && dico_str[0] != 'c' &&
        dico_str[0] != 'D' && dico_str[0] != 'd') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }

    if (job_str[0] != 'B' && job_str[0] != 'b' &&
        job_str[0] != 'N' && job_str[0] != 'n') {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'B' or 'N'");
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
    if (fixord && (nr_in < 0 || nr_in > n)) {
        PyErr_SetString(PyExc_ValueError, "NR must be >= 0 and <= N for fixed order");
        return NULL;
    }

    bool discr = (dico_str[0] == 'D' || dico_str[0] == 'd');
    if (discr && (alpha < 0.0 || alpha > 1.0)) {
        PyErr_SetString(PyExc_ValueError, "ALPHA must be in [0,1] for discrete-time");
        return NULL;
    }
    if (!discr && alpha > 0.0) {
        PyErr_SetString(PyExc_ValueError, "ALPHA must be <= 0 for continuous-time");
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

    i32 maxnmp = n;
    if (m > maxnmp) maxnmp = m;
    if (p > maxnmp) maxnmp = p;
    i32 ldwork = n * (2*n + maxnmp + 5) + (n * (n + 1)) / 2;
    if (ldwork < 1) ldwork = 1;
    ldwork *= 2;

    i32 iwork_size = n > 1 ? n : 1;

    f64 *hsv = (f64*)malloc((size_t)(n > 0 ? n : 1) * sizeof(f64));
    i32 *iwork = (i32*)malloc((size_t)iwork_size * sizeof(i32));
    f64 *dwork = (f64*)malloc((size_t)ldwork * sizeof(f64));

    if (!hsv || !iwork || !dwork) {
        free(hsv);
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    i32 nr = nr_in;
    i32 ns, iwarn, info;

    ab09md(dico_str, job_str, equil_str, ordsel_str, n, m, p, &nr, alpha,
           a_data, lda, b_data, ldb, c_data, ldc,
           &ns, hsv, tol, iwork, dwork, ldwork, &iwarn, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    npy_intp hsv_dims[1] = {n > 0 ? n : 1};
    PyObject *hsv_array = PyArray_SimpleNew(1, hsv_dims, NPY_DOUBLE);
    if (!hsv_array) {
        free(hsv);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)hsv_array), hsv, (size_t)(n > 0 ? n : 1) * sizeof(f64));
    free(hsv);

    PyObject *result = Py_BuildValue("OOOiOiii",
        a_array, b_array, c_array, ns, hsv_array, nr, iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(hsv_array);

    return result;
}



/* Python wrapper for ab09hd */
PyObject* py_ab09hd(PyObject* self, PyObject* args) {
    const char *dico_str, *job_str, *equil_str, *ordsel_str;
    int n, m, p, nr_in;
    double alpha, beta, tol1, tol2;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "ssssiiiiddOOOOdd", &dico_str, &job_str, &equil_str,
                          &ordsel_str, &n, &m, &p, &nr_in, &alpha, &beta,
                          &a_obj, &b_obj, &c_obj, &d_obj, &tol1, &tol2)) {
        return NULL;
    }

    if (dico_str[0] != 'C' && dico_str[0] != 'c' &&
        dico_str[0] != 'D' && dico_str[0] != 'd') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }

    if (job_str[0] != 'B' && job_str[0] != 'b' &&
        job_str[0] != 'F' && job_str[0] != 'f' &&
        job_str[0] != 'S' && job_str[0] != 's' &&
        job_str[0] != 'P' && job_str[0] != 'p') {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'B', 'F', 'S', or 'P'");
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
    if (p < 0 || (beta == 0.0 && p > m)) {
        PyErr_SetString(PyExc_ValueError, "P must be >= 0 (and <= M if BETA = 0)");
        return NULL;
    }

    bool fixord = (ordsel_str[0] == 'F' || ordsel_str[0] == 'f');
    if (fixord && (nr_in < 0 || nr_in > n)) {
        PyErr_SetString(PyExc_ValueError, "NR must be >= 0 and <= N for fixed order");
        return NULL;
    }

    bool discr = (dico_str[0] == 'D' || dico_str[0] == 'd');
    if (discr && (alpha < 0.0 || alpha > 1.0)) {
        PyErr_SetString(PyExc_ValueError, "ALPHA must be in [0,1] for discrete-time");
        return NULL;
    }
    if (!discr && alpha > 0.0) {
        PyErr_SetString(PyExc_ValueError, "ALPHA must be <= 0 for continuous-time");
        return NULL;
    }

    if (beta < 0.0) {
        PyErr_SetString(PyExc_ValueError, "BETA must be >= 0");
        return NULL;
    }

    if (tol2 > 0.0 && !fixord && tol2 > tol1) {
        PyErr_SetString(PyExc_ValueError, "TOL2 must be <= TOL1 when TOL2 > 0 and ORDSEL='A'");
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

    i32 mb = m;
    if (beta > 0.0) mb = m + p;

    i32 maxnmp = n;
    if (mb > maxnmp) maxnmp = mb;
    if (p > maxnmp) maxnmp = p;

    i32 lw1 = n * (maxnmp + 5);
    i32 lw2a = p * (mb + 2);
    i32 lw2b = 10 * n * (n + 1);
    i32 lw2 = 2 * n * p + (lw2a > lw2b ? lw2a : lw2b);
    i32 lw = lw1 > lw2 ? lw1 : lw2;
    if (lw < 2) lw = 2;
    i32 ldwork = 2 * n * n + mb * (n + p) + lw;
    if (ldwork < 2) ldwork = 2;
    ldwork *= 2;

    i32 iwork_size = 2 * n > 1 ? 2 * n : 1;
    i32 bwork_size = 2 * n > 1 ? 2 * n : 1;

    f64 *hsv = (f64*)malloc((size_t)(n > 0 ? n : 1) * sizeof(f64));
    i32 *iwork = (i32*)malloc((size_t)iwork_size * sizeof(i32));
    f64 *dwork = (f64*)malloc((size_t)ldwork * sizeof(f64));
    i32 *bwork = (i32*)malloc((size_t)bwork_size * sizeof(i32));

    if (!hsv || !iwork || !dwork || !bwork) {
        free(hsv);
        free(iwork);
        free(dwork);
        free(bwork);
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

    i32 nr = nr_in;
    i32 ns, iwarn, info;

    ab09hd(dico_str, job_str, equil_str, ordsel_str, n, m, p, &nr, alpha, beta,
           a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           &ns, hsv, tol1, tol2, iwork, dwork, ldwork, bwork, &iwarn, &info);

    free(iwork);
    free(dwork);
    free(bwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    npy_intp hsv_dims[1] = {n > 0 ? n : 1};
    PyObject *hsv_array = PyArray_SimpleNew(1, hsv_dims, NPY_DOUBLE);
    if (!hsv_array) {
        free(hsv);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)hsv_array), hsv, (size_t)(n > 0 ? n : 1) * sizeof(f64));
    free(hsv);

    PyObject *result = Py_BuildValue("OOOOiiOii",
        a_array, b_array, c_array, d_array, nr, ns, hsv_array, iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(hsv_array);

    return result;
}



/* Python wrapper for ab09hx */
PyObject* py_ab09hx(PyObject* self, PyObject* args) {
    const char *dico_str, *job_str, *ordsel_str;
    int n, m, p, nr_in;
    double tol1, tol2;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "sssiiiiOOOOdd", &dico_str, &job_str,
                          &ordsel_str, &n, &m, &p, &nr_in,
                          &a_obj, &b_obj, &c_obj, &d_obj, &tol1, &tol2)) {
        return NULL;
    }

    if (dico_str[0] != 'C' && dico_str[0] != 'c' &&
        dico_str[0] != 'D' && dico_str[0] != 'd') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }

    if (job_str[0] != 'B' && job_str[0] != 'b' &&
        job_str[0] != 'F' && job_str[0] != 'f' &&
        job_str[0] != 'S' && job_str[0] != 's' &&
        job_str[0] != 'P' && job_str[0] != 'p') {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'B', 'F', 'S', or 'P'");
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
    if (p < 0 || p > m) {
        PyErr_SetString(PyExc_ValueError, "P must satisfy 0 <= P <= M");
        return NULL;
    }

    bool fixord = (ordsel_str[0] == 'F' || ordsel_str[0] == 'f');
    if (fixord && (nr_in < 0 || nr_in > n)) {
        PyErr_SetString(PyExc_ValueError, "NR must be >= 0 and <= N for fixed order");
        return NULL;
    }

    if (tol2 > 0.0 && !fixord && tol2 > tol1) {
        PyErr_SetString(PyExc_ValueError, "TOL2 must be <= TOL1 when TOL2 > 0 and ORDSEL='A'");
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
    i32 ldt = n > 0 ? n : 1;
    i32 ldti = n > 0 ? n : 1;

    i32 maxnmp = n;
    if (m > maxnmp) maxnmp = m;
    if (p > maxnmp) maxnmp = p;

    i32 lw1 = n * (maxnmp + 5);
    i32 lw2a = p * (m + 2);
    i32 lw2b = 10 * n * (n + 1);
    i32 lw2 = 2 * n * p + (lw2a > lw2b ? lw2a : lw2b);
    i32 lw = lw1 > lw2 ? lw1 : lw2;
    if (lw < 2) lw = 2;
    i32 ldwork = lw;
    if (ldwork < 2) ldwork = 2;
    ldwork *= 2;

    i32 iwork_size = 2 * n > 1 ? 2 * n : 1;
    i32 bwork_size = 2 * n > 1 ? 2 * n : 1;
    i32 t_size = ldt * (n > 0 ? n : 1);
    i32 ti_size = ldti * (n > 0 ? n : 1);

    f64 *hsv = (f64*)malloc((size_t)(n > 0 ? n : 1) * sizeof(f64));
    f64 *t = (f64*)malloc((size_t)t_size * sizeof(f64));
    f64 *ti = (f64*)malloc((size_t)ti_size * sizeof(f64));
    i32 *iwork = (i32*)malloc((size_t)iwork_size * sizeof(i32));
    f64 *dwork = (f64*)malloc((size_t)ldwork * sizeof(f64));
    i32 *bwork = (i32*)malloc((size_t)bwork_size * sizeof(i32));

    if (!hsv || !t || !ti || !iwork || !dwork || !bwork) {
        free(hsv);
        free(t);
        free(ti);
        free(iwork);
        free(dwork);
        free(bwork);
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

    i32 nr = nr_in;
    i32 iwarn, info;

    ab09hx(dico_str, job_str, ordsel_str, n, m, p, &nr,
           a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           hsv, t, ldt, ti, ldti, tol1, tol2,
           iwork, dwork, ldwork, bwork, &iwarn, &info);

    i32 nmin = iwork[0];
    f64 rcond = dwork[1];

    free(iwork);
    free(dwork);
    free(bwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    npy_intp hsv_dims[1] = {n > 0 ? n : 1};
    PyObject *hsv_array = PyArray_SimpleNew(1, hsv_dims, NPY_DOUBLE);
    if (!hsv_array) {
        free(hsv);
        free(t);
        free(ti);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)hsv_array), hsv, (size_t)(n > 0 ? n : 1) * sizeof(f64));
    free(hsv);

    i32 actual_n = n > 0 ? n : 1;
    npy_intp t_dims[2] = {actual_n, actual_n};
    npy_intp t_strides[2] = {sizeof(f64), actual_n * sizeof(f64)};
    PyObject *t_array = PyArray_New(&PyArray_Type, 2, t_dims, NPY_DOUBLE,
                                    t_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!t_array) {
        free(t);
        free(ti);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(hsv_array);
        return NULL;
    }
    for (i32 j = 0; j < actual_n; j++) {
        memcpy((f64*)PyArray_DATA((PyArrayObject*)t_array) + j * actual_n, t + j * ldt, (size_t)actual_n * sizeof(f64));
    }
    free(t);

    npy_intp ti_dims[2] = {actual_n, actual_n};
    npy_intp ti_strides[2] = {sizeof(f64), actual_n * sizeof(f64)};
    PyObject *ti_array = PyArray_New(&PyArray_Type, 2, ti_dims, NPY_DOUBLE,
                                     ti_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!ti_array) {
        free(ti);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(hsv_array);
        Py_DECREF(t_array);
        return NULL;
    }
    for (i32 j = 0; j < actual_n; j++) {
        memcpy((f64*)PyArray_DATA((PyArrayObject*)ti_array) + j * actual_n, ti + j * ldti, (size_t)actual_n * sizeof(f64));
    }
    free(ti);

    PyObject *result = Py_BuildValue("OOOOiOOOidii",
        a_array, b_array, c_array, d_array, nr, hsv_array,
        t_array, ti_array, nmin, rcond, iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(hsv_array);
    Py_DECREF(t_array);
    Py_DECREF(ti_array);

    return result;
}



/* Python wrapper for ab09nd */
PyObject* py_ab09nd(PyObject* self, PyObject* args) {
    const char *dico_str, *job_str, *equil_str, *ordsel_str;
    int n, m, p, nr_in;
    double alpha, tol1, tol2;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "ssssiiiidOOOOdd", &dico_str, &job_str, &equil_str,
                          &ordsel_str, &n, &m, &p, &nr_in, &alpha,
                          &a_obj, &b_obj, &c_obj, &d_obj, &tol1, &tol2)) {
        return NULL;
    }

    if (dico_str[0] != 'C' && dico_str[0] != 'c' &&
        dico_str[0] != 'D' && dico_str[0] != 'd') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }

    if (job_str[0] != 'B' && job_str[0] != 'b' &&
        job_str[0] != 'N' && job_str[0] != 'n') {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'B' or 'N'");
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
    if (fixord && (nr_in < 0 || nr_in > n)) {
        PyErr_SetString(PyExc_ValueError, "NR must be >= 0 and <= N for fixed order");
        return NULL;
    }

    bool discr = (dico_str[0] == 'D' || dico_str[0] == 'd');
    if (discr && (alpha < 0.0 || alpha > 1.0)) {
        PyErr_SetString(PyExc_ValueError, "ALPHA must be in [0,1] for discrete-time");
        return NULL;
    }
    if (!discr && alpha > 0.0) {
        PyErr_SetString(PyExc_ValueError, "ALPHA must be <= 0 for continuous-time");
        return NULL;
    }

    if (tol2 > 0.0 && tol2 > tol1) {
        PyErr_SetString(PyExc_ValueError, "TOL2 must be <= TOL1 when TOL2 > 0");
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

    i32 maxnmp = n;
    if (m > maxnmp) maxnmp = m;
    if (p > maxnmp) maxnmp = p;
    i32 ldwork = n * (2*n + maxnmp + 5) + (n * (n + 1)) / 2;
    if (ldwork < 1) ldwork = 1;
    ldwork *= 2;

    i32 iwork_size = 2 * n > 1 ? 2 * n : 1;

    f64 *hsv = (f64*)malloc((size_t)(n > 0 ? n : 1) * sizeof(f64));
    i32 *iwork = (i32*)malloc((size_t)iwork_size * sizeof(i32));
    f64 *dwork = (f64*)malloc((size_t)ldwork * sizeof(f64));

    if (!hsv || !iwork || !dwork) {
        free(hsv);
        free(iwork);
        free(dwork);
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

    i32 nr = nr_in;
    i32 ns, iwarn, info;

    ab09nd(dico_str, job_str, equil_str, ordsel_str, n, m, p, &nr, alpha,
           a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           &ns, hsv, tol1, tol2, iwork, dwork, ldwork, &iwarn, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    npy_intp hsv_dims[1] = {n > 0 ? n : 1};
    PyObject *hsv_array = PyArray_SimpleNew(1, hsv_dims, NPY_DOUBLE);
    if (!hsv_array) {
        free(hsv);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)hsv_array), hsv, (size_t)(n > 0 ? n : 1) * sizeof(f64));
    free(hsv);

    PyObject *result = Py_BuildValue("OOOOiiOii",
        a_array, b_array, c_array, d_array, nr, ns, hsv_array, iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(hsv_array);

    return result;
}



/* Python wrapper for ab08mz - Normal rank of transfer function (complex) */
PyObject* py_ab08mz(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *equil_str;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    f64 tol = 0.0;

    static char *kwlist[] = {"equil", "n", "m", "p", "a", "b", "c", "d", "tol", NULL};

    i32 n, m, p;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siiiOOOO|d", kwlist,
            &equil_str, &n, &m, &p, &a_obj, &b_obj, &c_obj, &d_obj, &tol)) {
        return NULL;
    }

    char equil = (char)toupper((unsigned char)equil_str[0]);
    if (equil != 'S' && equil != 'N') {
        PyErr_SetString(PyExc_ValueError, "equil must be 'S' or 'N'");
        return NULL;
    }

    PyArrayObject *a_array = NULL, *b_array = NULL, *c_array = NULL, *d_array = NULL;

    if (n > 0) {
        a_array = (PyArrayObject*)PyArray_FROM_OTF(
            a_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!a_array) return NULL;
    }

    if (n > 0 && m > 0) {
        b_array = (PyArrayObject*)PyArray_FROM_OTF(
            b_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!b_array) {
            Py_XDECREF(a_array);
            return NULL;
        }
    }

    if (p > 0 && n > 0) {
        c_array = (PyArrayObject*)PyArray_FROM_OTF(
            c_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!c_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(b_array);
            return NULL;
        }
    }

    if (p > 0 && m > 0) {
        d_array = (PyArrayObject*)PyArray_FROM_OTF(
            d_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!d_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(b_array);
            Py_XDECREF(c_array);
            return NULL;
        }
    }

    i32 np = n + p;
    i32 nm = n + m;
    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldd = p > 0 ? p : 1;

    i32 mpm = (p < m) ? p : m;
    i32 mpn = (p < n) ? p : n;
    i32 max_mp = (m > p) ? m : p;

    i32 t1 = mpm + ((3*m - 1 > n) ? (3*m - 1) : n);
    i32 t2 = mpn + ((3*p - 1 > np) ? ((3*p - 1 > nm) ? (3*p - 1) : nm) : ((np > nm) ? np : nm));
    i32 t3 = (t1 > 1) ? t1 : 1;
    t3 = (t3 > t2) ? t3 : t2;
    i32 lzwork = np * nm + t3;
    if (lzwork < 1) lzwork = 1;

    i32 liwork = 2*n + max_mp + 1;
    if (liwork < 1) liwork = 1;

    i32 ldwork = 2 * max_mp;
    if (ldwork < 1) ldwork = 1;

    i32 *iwork = (i32*)calloc(liwork, sizeof(i32));
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    c128 *zwork = (c128*)calloc(lzwork, sizeof(c128));

    if (!iwork || !dwork || !zwork) {
        free(iwork);
        free(dwork);
        free(zwork);
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        PyErr_NoMemory();
        return NULL;
    }

    c128 *a_data = a_array ? (c128*)PyArray_DATA(a_array) : NULL;
    c128 *b_data = b_array ? (c128*)PyArray_DATA(b_array) : NULL;
    c128 *c_data = c_array ? (c128*)PyArray_DATA(c_array) : NULL;
    c128 *d_data = d_array ? (c128*)PyArray_DATA(d_array) : NULL;

    c128 dummy = 0.0;
    if (!a_data) a_data = &dummy;
    if (!b_data) b_data = &dummy;
    if (!c_data) c_data = &dummy;
    if (!d_data) d_data = &dummy;

    i32 rank = 0;
    i32 info = slicot_ab08mz(equil, n, m, p, a_data, lda, b_data, ldb,
                             c_data, ldc, d_data, ldd, &rank, tol,
                             iwork, dwork, zwork, lzwork);

    free(iwork);
    free(dwork);
    free(zwork);

    if (a_array) PyArray_ResolveWritebackIfCopy(a_array);
    if (b_array) PyArray_ResolveWritebackIfCopy(b_array);
    if (c_array) PyArray_ResolveWritebackIfCopy(c_array);
    if (d_array) PyArray_ResolveWritebackIfCopy(d_array);

    Py_XDECREF(a_array);
    Py_XDECREF(b_array);
    Py_XDECREF(c_array);
    Py_XDECREF(d_array);

    return Py_BuildValue("ii", rank, info);
}



/* Python wrapper for ab08nz - Regular pencil for invariant zeros (complex) */
PyObject* py_ab08nz(PyObject* self, PyObject* args) {
    const char *equil_str;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    double tol;

    if (!PyArg_ParseTuple(args, "siiiOOOOd", &equil_str, &(i32){0}, &(i32){0}, &(i32){0},
                          &a_obj, &b_obj, &c_obj, &d_obj, &tol)) {
        PyErr_Clear();
        if (!PyArg_ParseTuple(args, "sOOOOd", &equil_str, &a_obj, &b_obj, &c_obj, &d_obj, &tol)) {
            return NULL;
        }
    }

    char equil = (char)toupper((unsigned char)equil_str[0]);

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!b_array) { Py_DECREF(a_array); return NULL; }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!c_array) { Py_DECREF(a_array); Py_DECREF(b_array); return NULL; }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (!d_array) { Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); return NULL; }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    i32 n = PyArray_NDIM(a_array) >= 1 ? (i32)a_dims[0] : 0;
    i32 m = PyArray_NDIM(b_array) >= 2 ? (i32)b_dims[1] : 0;
    i32 p = PyArray_NDIM(c_array) >= 1 ? (i32)c_dims[0] : 0;

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldd = p > 0 ? p : 1;
    i32 ldaf = (n + m > 0) ? n + m : 1;
    i32 ldbf = (n + p > 0) ? n + p : 1;

    i32 minmp = (m < p) ? m : p;
    i32 af_cols = n + minmp;
    i32 bf_cols = n + m;

    c128 *af = (c128*)calloc((size_t)ldaf * (af_cols > 0 ? af_cols : 1), sizeof(c128));
    c128 *bf = (c128*)calloc((size_t)ldbf * (bf_cols > 0 ? bf_cols : 1), sizeof(c128));

    i32 infz_size = n > 0 ? n : 1;
    i32 kronr_size = ((n > m ? n : m) + 1);
    i32 kronl_size = ((n > p ? n : p) + 1);
    i32 iwork_size = (m > p ? m : p) > 0 ? (m > p ? m : p) : 1;
    i32 dwork_size = (n > 2 * (m > p ? m : p) ? n : 2 * (m > p ? m : p));
    if (dwork_size < 1) dwork_size = 1;

    i32 *infz = (i32*)calloc(infz_size, sizeof(i32));
    i32 *kronr = (i32*)calloc(kronr_size, sizeof(i32));
    i32 *kronl = (i32*)calloc(kronl_size, sizeof(i32));
    i32 *iwork = (i32*)calloc(iwork_size, sizeof(i32));
    f64 *dwork = (f64*)calloc(dwork_size, sizeof(f64));

    i32 s = (m > p) ? m : p;
    i32 lzwork = (s > n ? s : n) + ((3*s - 1 > n + s) ? 3*s - 1 : n + s);
    if (lzwork < 1) lzwork = 1;
    lzwork *= 2;
    c128 *zwork = (c128*)calloc(lzwork, sizeof(c128));

    if (!af || !bf || !infz || !kronr || !kronl || !iwork || !dwork || !zwork) {
        free(af); free(bf); free(infz); free(kronr); free(kronl);
        free(iwork); free(dwork); free(zwork);
        Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array);
        PyErr_NoMemory();
        return NULL;
    }

    c128 *a_data = (c128*)PyArray_DATA(a_array);
    c128 *b_data = (c128*)PyArray_DATA(b_array);
    c128 *c_data = (c128*)PyArray_DATA(c_array);
    c128 *d_data = (c128*)PyArray_DATA(d_array);

    i32 nu, rank, dinfz, nkror, nkrol;
    i32 info = slicot_ab08nz(equil, n, m, p, a_data, lda, b_data, ldb,
                             c_data, ldc, d_data, ldd, &nu, &rank, &dinfz,
                             &nkror, &nkrol, infz, kronr, kronl, af, ldaf,
                             bf, ldbf, tol, iwork, dwork, zwork, lzwork);

    npy_intp infz_dims[1] = {infz_size};
    PyObject *infz_array = PyArray_SimpleNew(1, infz_dims, NPY_INT32);
    memcpy(PyArray_DATA((PyArrayObject*)infz_array), infz, infz_size * sizeof(i32));

    npy_intp kronr_dims[1] = {kronr_size};
    PyObject *kronr_array = PyArray_SimpleNew(1, kronr_dims, NPY_INT32);
    memcpy(PyArray_DATA((PyArrayObject*)kronr_array), kronr, kronr_size * sizeof(i32));

    npy_intp kronl_dims[1] = {kronl_size};
    PyObject *kronl_array = PyArray_SimpleNew(1, kronl_dims, NPY_INT32);
    memcpy(PyArray_DATA((PyArrayObject*)kronl_array), kronl, kronl_size * sizeof(i32));

    i32 af_rows = ldaf;
    i32 af_actual_cols = af_cols > 0 ? af_cols : 1;
    npy_intp af_dims[2] = {af_rows, af_actual_cols};
    npy_intp af_strides[2] = {sizeof(c128), af_rows * sizeof(c128)};
    PyObject *af_array = PyArray_New(&PyArray_Type, 2, af_dims, NPY_CDOUBLE,
                                     af_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (af_array) {
        for (i32 j = 0; j < af_actual_cols; j++) {
            memcpy((c128*)PyArray_DATA((PyArrayObject*)af_array) + j * af_rows, af + j * ldaf, (size_t)af_rows * sizeof(c128));
        }
    }
    free(af);

    i32 bf_rows = ldbf;
    i32 bf_actual_cols = bf_cols > 0 ? bf_cols : 1;
    npy_intp bf_dims[2] = {bf_rows, bf_actual_cols};
    npy_intp bf_strides[2] = {sizeof(c128), bf_rows * sizeof(c128)};
    PyObject *bf_array = PyArray_New(&PyArray_Type, 2, bf_dims, NPY_CDOUBLE,
                                     bf_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (bf_array) {
        for (i32 j = 0; j < bf_actual_cols; j++) {
            memcpy((c128*)PyArray_DATA((PyArrayObject*)bf_array) + j * bf_rows, bf + j * ldbf, (size_t)bf_rows * sizeof(c128));
        }
    }
    free(bf);

    free(infz); free(kronr); free(kronl); free(iwork); free(dwork); free(zwork);
    Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array);

    return Py_BuildValue("iiiiiOOOOOi", nu, rank, dinfz, nkror, nkrol,
                         infz_array, kronr_array, kronl_array, af_array, bf_array, info);
}



/* Python wrapper for ab8nxz - Extract reduced system (complex) */
PyObject* py_ab8nxz(PyObject* self, PyObject* args) {
    i32 n, m, p, ro, sigma;
    f64 svlmax, tol;
    PyObject *abcd_obj;

    if (!PyArg_ParseTuple(args, "iiiiidOd", &n, &m, &p, &ro, &sigma, &svlmax, &abcd_obj, &tol)) {
        return NULL;
    }

    if (n < 0 || m < 0 || p < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative (n=%d, m=%d, p=%d)", n, m, p);
        return NULL;
    }

    PyArrayObject *abcd_array = (PyArrayObject*)PyArray_FROM_OTF(
        abcd_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!abcd_array) return NULL;

    i32 np = n + p;
    i32 ldabcd = np > 0 ? np : 1;

    i32 infz_size = n > 0 ? n : 1;
    i32 kronl_size = n + 1;
    i32 iwork_size = (m > p ? m : p);
    if (iwork_size < 1) iwork_size = 1;
    i32 dwork_size = 2 * iwork_size;

    i32 *infz = (i32*)calloc(infz_size, sizeof(i32));
    i32 *kronl = (i32*)calloc(kronl_size, sizeof(i32));
    i32 *iwork = (i32*)malloc(iwork_size * sizeof(i32));
    f64 *dwork = (f64*)malloc(dwork_size * sizeof(f64));

    i32 mpm = (p < m) ? p : m;
    i32 minpn = (p < n) ? p : n;
    i32 jwork1 = mpm + (3 * m - 1 > n ? 3 * m - 1 : n);
    i32 jwork2 = minpn + (3 * p - 1 > np ? 3 * p - 1 : (np > n + m ? np : n + m));
    i32 lzwork = jwork1 > jwork2 ? jwork1 : jwork2;
    if (lzwork < 1) lzwork = 1;
    c128 *zwork = (c128*)malloc(lzwork * sizeof(c128));

    if (!infz || !kronl || !iwork || !dwork || !zwork) {
        free(infz); free(kronl); free(iwork); free(dwork); free(zwork);
        Py_DECREF(abcd_array);
        PyErr_NoMemory();
        return NULL;
    }

    c128 *abcd_data = (c128*)PyArray_DATA(abcd_array);
    i32 ninfz = 0;
    i32 mu = 0, nu = 0, nkrol = 0;

    i32 info = slicot_ab8nxz(n, m, p, &ro, &sigma, svlmax, abcd_data, ldabcd,
                             &ninfz, infz, kronl, &mu, &nu, &nkrol, tol,
                             iwork, dwork, zwork, lzwork);

    PyArray_ResolveWritebackIfCopy(abcd_array);

    npy_intp infz_dims[1] = {n > 0 ? n : 0};
    npy_intp kronl_dims[1] = {nkrol > 0 ? nkrol : 0};

    PyObject *infz_array = PyArray_SimpleNew(1, infz_dims, NPY_INT32);
    PyObject *kronl_array = PyArray_SimpleNew(1, kronl_dims, NPY_INT32);

    if (!infz_array || !kronl_array) {
        Py_XDECREF(infz_array);
        Py_XDECREF(kronl_array);
        Py_DECREF(abcd_array);
        free(infz); free(kronl); free(iwork); free(dwork); free(zwork);
        return PyErr_NoMemory();
    }

    if (n > 0) memcpy(PyArray_DATA((PyArrayObject*)infz_array), infz, n * sizeof(i32));
    if (nkrol > 0) memcpy(PyArray_DATA((PyArrayObject*)kronl_array), kronl, nkrol * sizeof(i32));

    free(infz); free(kronl); free(iwork); free(dwork); free(zwork);

    PyObject *result = Py_BuildValue("(OiiiiiOOi)", abcd_array, ro, sigma, mu, nu, ninfz,
                                     infz_array, kronl_array, info);
    Py_DECREF(abcd_array);
    Py_DECREF(infz_array);
    Py_DECREF(kronl_array);
    return result;
}



/* Python wrapper for ab13bd */
PyObject* py_ab13bd(PyObject* self, PyObject* args) {
    const char *dico, *jobn;
    f64 tol;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "ssOOOOd", &dico, &jobn, &a_obj, &b_obj, &c_obj, &d_obj, &tol)) {
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
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
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

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);

    i32 mxnp = n > p ? n : p;
    i32 minnp = n < p ? n : p;
    i32 req1 = m * (n + m) + (n * (n + 5) > m * (m + 2) ? n * (n + 5) : m * (m + 2));
    req1 = req1 > 4 * p ? req1 : 4 * p;
    i32 req2 = n * (mxnp + 4) + minnp;
    i32 ldwork = req1 > req2 ? req1 : req2;
    ldwork = ldwork > 1 ? ldwork : 1;
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (!dwork) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return PyErr_NoMemory();
    }

    i32 nq = 0;
    i32 iwarn = 0, info = 0;

    f64 h2norm = ab13bd(dico, jobn, n, m, p, a_data, lda, b_data, ldb, c_data, ldc,
                         d_data, ldd, &nq, tol, dwork, ldwork, &iwarn, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    return Py_BuildValue("(diii)", h2norm, nq, iwarn, info);
}



/* Python wrapper for ab13dd */
PyObject* py_ab13dd(PyObject* self, PyObject* args) {
    const char *dico, *jobe, *equil, *jobd;
    i32 n, m, p;
    f64 tol;
    PyObject *fpeak_obj, *a_obj, *e_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "ssssiiiOOOOOOd",
                          &dico, &jobe, &equil, &jobd, &n, &m, &p,
                          &fpeak_obj, &a_obj, &e_obj, &b_obj, &c_obj, &d_obj, &tol)) {
        return NULL;
    }

    PyArrayObject *fpeak_array = (PyArrayObject*)PyArray_FROM_OTF(fpeak_obj, NPY_DOUBLE,
                                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (fpeak_array == NULL) return NULL;

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        Py_DECREF(fpeak_array);
        return NULL;
    }

    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (e_array == NULL) {
        Py_DECREF(fpeak_array);
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(fpeak_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (c_array == NULL) {
        Py_DECREF(fpeak_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (d_array == NULL) {
        Py_DECREF(fpeak_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    i32 lda = n > 1 ? n : 1;
    i32 lde = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldc = p > 1 ? p : 1;
    i32 ldd = p > 1 ? p : 1;

    f64 *fpeak_data = (f64*)PyArray_DATA(fpeak_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);

    i32 minpm = m < p ? m : p;
    i32 maxpm = m > p ? m : p;
    i32 nn = n * n;
    i32 pm = p + m;
    (void)pm;

    i32 ldwork = 15 * nn + p * p + m * m + (6 * n + 3) * (p + m) + 4 * p * m + n * m + 22 * n + 7 * minpm;
    if (ldwork < 1) ldwork = 1;

    i32 lcwork = (n + m) * (n + p) + 2 * minpm + maxpm;
    if (lcwork < 1) lcwork = 1;

    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    c128 *cwork = (c128*)calloc(lcwork, sizeof(c128));
    i32 *iwork = (i32*)calloc(n > 1 ? n : 1, sizeof(i32));

    if (!dwork || !cwork || !iwork) {
        free(dwork);
        free(cwork);
        free(iwork);
        Py_DECREF(fpeak_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return PyErr_NoMemory();
    }

    f64 gpeak[2] = {0.0, 1.0};
    i32 info = 0;

    ab13dd(dico, jobe, equil, jobd, n, m, p, fpeak_data,
           a_data, lda, e_data, lde, b_data, ldb, c_data, ldc, d_data, ldd,
           gpeak, tol, iwork, dwork, ldwork, cwork, lcwork, &info);

    free(dwork);
    free(cwork);
    free(iwork);

    PyArray_ResolveWritebackIfCopy(fpeak_array);
    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    npy_intp gpeak_dims[1] = {2};
    PyObject *gpeak_array = PyArray_SimpleNew(1, gpeak_dims, NPY_DOUBLE);
    if (gpeak_array == NULL) {
        Py_DECREF(fpeak_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }
    ((f64*)PyArray_DATA((PyArrayObject*)gpeak_array))[0] = gpeak[0];
    ((f64*)PyArray_DATA((PyArrayObject*)gpeak_array))[1] = gpeak[1];

    npy_intp fpeak_out_dims[1] = {2};
    PyObject *fpeak_out_array = PyArray_SimpleNew(1, fpeak_out_dims, NPY_DOUBLE);
    if (fpeak_out_array == NULL) {
        Py_DECREF(fpeak_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(gpeak_array);
        return NULL;
    }
    ((f64*)PyArray_DATA((PyArrayObject*)fpeak_out_array))[0] = fpeak_data[0];
    ((f64*)PyArray_DATA((PyArrayObject*)fpeak_out_array))[1] = fpeak_data[1];

    Py_DECREF(fpeak_array);
    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    PyObject *result = Py_BuildValue("(OOi)", gpeak_array, fpeak_out_array, info);
    Py_DECREF(gpeak_array);
    Py_DECREF(fpeak_out_array);

    return result;
}



/* Python wrapper for ab13dx */
PyObject* py_ab13dx(PyObject* self, PyObject* args) {
    const char *dico, *jobe, *jobd;
    i32 n, m, p;
    f64 omega;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "sssiiiOOOOOd", &dico, &jobe, &jobd, &n, &m, &p,
                          &a_obj, &e_obj, &b_obj, &c_obj, &d_obj, &omega)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    i32 lda = n > 1 ? n : 1;
    i32 lde = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldc = p > 1 ? p : 1;
    i32 ldd = p > 1 ? p : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);

    i32 minpm = (p < m) ? p : m;
    i32 maxpm = (p > m) ? p : m;
    i32 ldwork = (4 * minpm + maxpm > 6 * minpm) ? (4 * minpm + maxpm) : (6 * minpm);
    ldwork += p * m;
    ldwork = ldwork > 1 ? ldwork : 1;

    i32 lzwork = (n + m) * (n + p) + 2 * minpm + maxpm;
    lzwork = lzwork > 1 ? lzwork : 1;

    i32 *iwork = (i32*)calloc(n > 1 ? n : 1, sizeof(i32));
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    c128 *zwork = (c128*)calloc(lzwork, sizeof(c128));

    if (!iwork || !dwork || !zwork) {
        free(iwork);
        free(dwork);
        free(zwork);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return PyErr_NoMemory();
    }

    i32 info = 0;
    f64 result = ab13dx(dico, jobe, jobd, n, m, p, omega,
                        a_data, lda, e_data, lde, b_data, ldb,
                        c_data, ldc, d_data, ldd,
                        iwork, dwork, ldwork, zwork, lzwork, &info);

    free(iwork);
    free(dwork);
    free(zwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    return Py_BuildValue("(di)", result, info);
}



/* Python wrapper for ab13ed */
PyObject* py_ab13ed(PyObject* self, PyObject* args, PyObject* kwargs) {
    i32 n;
    f64 tol = 0.0;
    PyObject *a_obj;
    PyArrayObject *a_array;
    i32 info;
    f64 low, high;
    static char *kwlist[] = {"a", "tol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|d", kwlist, &a_obj, &tol)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(a_array) != 2) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "Matrix A must be 2D");
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    n = (i32)a_dims[0];
    if (a_dims[0] != a_dims[1]) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "Matrix A must be square");
        return NULL;
    }

    i32 lda = (i32)a_dims[0];
    f64 *a_data = (f64*)PyArray_DATA(a_array);

    /* Workspace allocation */
    i32 minwrk = 3 * n * (n + 1);
    i32 ldwork = (minwrk > 1) ? minwrk : 1;

    /* Allocate slightly more to be safe for optimal or just min */
    ldwork = (ldwork < 1024) ? 1024 : ldwork;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        return PyErr_NoMemory();
    }

    ab13ed(n, a_data, lda, &low, &high, tol, dwork, ldwork, &info);

    free(dwork);
    Py_DECREF(a_array);

    return Py_BuildValue("ddi", low, high, info);
}



/* Python wrapper for ab13fd */
PyObject* py_ab13fd(PyObject* self, PyObject* args, PyObject* kwargs) {
    i32 n;
    f64 tol = 0.0;
    PyObject *a_obj;
    PyArrayObject *a_array;
    i32 info;
    f64 beta, omega;
    static char *kwlist[] = {"a", "tol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|d", kwlist, &a_obj, &tol)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(a_array) != 2) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "Matrix A must be 2D");
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    n = (i32)a_dims[0];
    if (a_dims[0] != a_dims[1]) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "Matrix A must be square");
        return NULL;
    }

    i32 lda = n > 1 ? n : 1;
    f64 *a_data = (f64*)PyArray_DATA(a_array);

    /* Real workspace: 3*n*(n+2) minimum */
    i32 minwrk = 3 * n * (n + 2);
    i32 ldwork = (minwrk > 1) ? minwrk : 1;
    ldwork = (ldwork < 1024) ? 1024 : ldwork;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(a_array);
        return PyErr_NoMemory();
    }

    /* Complex workspace: n*(n+3) minimum */
    i32 mincwrk = n * (n + 3);
    i32 lcwork = (mincwrk > 1) ? mincwrk : 1;
    lcwork = (lcwork < 512) ? 512 : lcwork;

    c128 *cwork = (c128*)malloc(lcwork * sizeof(c128));
    if (cwork == NULL) {
        free(dwork);
        Py_DECREF(a_array);
        return PyErr_NoMemory();
    }

    ab13fd(n, a_data, lda, &beta, &omega, tol, dwork, ldwork, cwork, lcwork, &info);

    free(cwork);
    free(dwork);
    Py_DECREF(a_array);

    return Py_BuildValue("ddi", beta, omega, info);
}

/* Python wrapper for ab09cd */
PyObject* py_ab09cd(PyObject* self, PyObject* args) {
    const char *dico_str, *equil_str, *ordsel_str;
    int n, m, p, nr_in;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    double tol1, tol2;

    if (!PyArg_ParseTuple(args, "sssiiiiOOOOdd", &dico_str, &equil_str, &ordsel_str,
                          &n, &m, &p, &nr_in, &a_obj, &b_obj, &c_obj, &d_obj,
                          &tol1, &tol2)) {
        return NULL;
    }

    if (dico_str[0] != 'C' && dico_str[0] != 'c' &&
        dico_str[0] != 'D' && dico_str[0] != 'd') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
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
    if (fixord && (nr_in < 0 || nr_in > n)) {
        PyErr_SetString(PyExc_ValueError, "NR must be >= 0 and <= N for fixed order");
        return NULL;
    }

    if (tol2 > 0.0 && tol2 > tol1) {
        PyErr_SetString(PyExc_ValueError, "TOL2 must be <= TOL1 when TOL2 > 0");
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

    i32 maxnmp = n;
    if (m > maxnmp) maxnmp = m;
    if (p > maxnmp) maxnmp = p;
    i32 minnm = n < m ? n : m;

    i32 ldw1 = n * (2 * n + maxnmp + 5) + (n * (n + 1)) / 2;
    i32 tmp1 = 3 * m + 1;
    i32 tmp2 = minnm + p;
    i32 ldw2 = n * (m + p + 2) + 2 * m * p + minnm + (tmp1 > tmp2 ? tmp1 : tmp2);
    i32 ldwork = (ldw1 > ldw2 ? ldw1 : ldw2);
    if (ldwork < 1) ldwork = 1;
    ldwork *= 2;

    bool discr = (dico_str[0] == 'D' || dico_str[0] == 'd');
    i32 iwork_size = discr ? (n > m ? n : m) : m;
    if (iwork_size < 1) iwork_size = 1;

    f64 *hsv = (f64*)malloc((size_t)(n > 0 ? n : 1) * sizeof(f64));
    i32 *iwork = (i32*)malloc((size_t)iwork_size * sizeof(i32));
    f64 *dwork = (f64*)malloc((size_t)ldwork * sizeof(f64));

    if (!hsv || !iwork || !dwork) {
        free(hsv);
        free(iwork);
        free(dwork);
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

    i32 nr = nr_in;
    i32 iwarn, info;

    ab09cd(dico_str, equil_str, ordsel_str, n, m, p, &nr, a_data, lda, b_data, ldb,
           c_data, ldc, d_data, ldd, hsv, tol1, tol2, iwork, dwork, ldwork,
           &iwarn, &info);

    i32 nmin = iwork[0];

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    npy_intp hsv_dims[1] = {n > 0 ? n : 1};
    PyObject *hsv_array = PyArray_SimpleNew(1, hsv_dims, NPY_DOUBLE);
    if (!hsv_array) {
        free(hsv);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)hsv_array), hsv, (size_t)(n > 0 ? n : 1) * sizeof(f64));
    free(hsv);

    PyObject *result = Py_BuildValue("OOOOiOiii",
        a_array, b_array, c_array, d_array, nr, hsv_array, nmin, iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(hsv_array);

    return result;
}

/* Python wrapper for ab09ed */
PyObject* py_ab09ed(PyObject* self, PyObject* args) {
    const char *dico_str, *equil_str, *ordsel_str;
    int n, m, p, nr_in;
    double alpha, tol1, tol2;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "sssiiiidOOOOdd", &dico_str, &equil_str, &ordsel_str,
                          &n, &m, &p, &nr_in, &alpha, &a_obj, &b_obj, &c_obj, &d_obj,
                          &tol1, &tol2)) {
        return NULL;
    }

    if (dico_str[0] != 'C' && dico_str[0] != 'c' &&
        dico_str[0] != 'D' && dico_str[0] != 'd') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
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
    if (fixord && (nr_in < 0 || nr_in > n)) {
        PyErr_SetString(PyExc_ValueError, "NR must be >= 0 and <= N for fixed order");
        return NULL;
    }

    bool discr = (dico_str[0] == 'D' || dico_str[0] == 'd');
    if (discr && (alpha < 0.0 || alpha > 1.0)) {
        PyErr_SetString(PyExc_ValueError, "ALPHA must be in [0, 1] for discrete-time");
        return NULL;
    }
    if (!discr && alpha > 0.0) {
        PyErr_SetString(PyExc_ValueError, "ALPHA must be <= 0 for continuous-time");
        return NULL;
    }

    if (tol2 > 0.0 && tol2 > tol1) {
        PyErr_SetString(PyExc_ValueError, "TOL2 must be <= TOL1 when TOL2 > 0");
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

    i32 maxnmp = n;
    if (m > maxnmp) maxnmp = m;
    if (p > maxnmp) maxnmp = p;
    i32 minnm = n < m ? n : m;

    i32 ldw1 = n * (2 * n + maxnmp + 5) + (n * (n + 1)) / 2;
    i32 tmp1 = 3 * m + 1;
    i32 tmp2 = minnm + p;
    i32 ldw2 = n * (m + p + 2) + 2 * m * p + minnm + (tmp1 > tmp2 ? tmp1 : tmp2);
    i32 ldwork = (ldw1 > ldw2 ? ldw1 : ldw2);
    if (ldwork < 1) ldwork = 1;
    ldwork *= 2;

    i32 iwork_size = discr ? (n > m ? n : m) : m;
    if (iwork_size < 1) iwork_size = 1;

    f64 *hsv = (f64*)malloc((size_t)(n > 0 ? n : 1) * sizeof(f64));
    i32 *iwork = (i32*)malloc((size_t)iwork_size * sizeof(i32));
    f64 *dwork = (f64*)malloc((size_t)ldwork * sizeof(f64));

    if (!hsv || !iwork || !dwork) {
        free(hsv);
        free(iwork);
        free(dwork);
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

    i32 nr = nr_in;
    i32 ns, iwarn, info;

    ab09ed(dico_str, equil_str, ordsel_str, n, m, p, &nr, alpha,
           a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           &ns, hsv, tol1, tol2, iwork, dwork, ldwork, &iwarn, &info);

    i32 nmin = iwork[0];

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    npy_intp hsv_dims[1] = {n > 0 ? n : 1};
    PyObject *hsv_array = PyArray_SimpleNew(1, hsv_dims, NPY_DOUBLE);
    if (!hsv_array) {
        free(hsv);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)hsv_array), hsv, (size_t)(n > 0 ? n : 1) * sizeof(f64));
    free(hsv);

    PyObject *result = Py_BuildValue("OOOOiiOiii",
        a_array, b_array, c_array, d_array, nr, ns, hsv_array, nmin, iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(hsv_array);

    return result;
}

/* Python wrapper for ab09cx */
PyObject* py_ab09cx(PyObject* self, PyObject* args) {
    const char *dico_str, *ordsel_str;
    int n, m, p, nr_in;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    double tol1, tol2;

    if (!PyArg_ParseTuple(args, "ssiiiiOOOOdd", &dico_str, &ordsel_str,
                          &n, &m, &p, &nr_in, &a_obj, &b_obj, &c_obj, &d_obj,
                          &tol1, &tol2)) {
        return NULL;
    }

    if (dico_str[0] != 'C' && dico_str[0] != 'c' &&
        dico_str[0] != 'D' && dico_str[0] != 'd') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
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
    if (fixord && (nr_in < 0 || nr_in > n)) {
        PyErr_SetString(PyExc_ValueError, "NR must be >= 0 and <= N for fixed order");
        return NULL;
    }

    if (tol2 > 0.0 && tol2 > tol1) {
        PyErr_SetString(PyExc_ValueError, "TOL2 must be <= TOL1 when TOL2 > 0");
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

    i32 maxnmp = n;
    if (m > maxnmp) maxnmp = m;
    if (p > maxnmp) maxnmp = p;
    i32 minnm = n < m ? n : m;

    i32 ldw1 = n * (2 * n + maxnmp + 5) + (n * (n + 1)) / 2;
    i32 tmp1 = 3 * m + 1;
    i32 tmp2 = minnm + p;
    i32 ldw2 = n * (m + p + 2) + 2 * m * p + minnm + (tmp1 > tmp2 ? tmp1 : tmp2);
    i32 ldwork = (ldw1 > ldw2 ? ldw1 : ldw2);
    if (ldwork < 1) ldwork = 1;
    ldwork *= 2;

    bool discr = (dico_str[0] == 'D' || dico_str[0] == 'd');
    i32 iwork_size = discr ? (n > m ? n : m) : m;
    if (iwork_size < 1) iwork_size = 1;

    f64 *hsv = (f64*)malloc((size_t)(n > 0 ? n : 1) * sizeof(f64));
    i32 *iwork = (i32*)malloc((size_t)iwork_size * sizeof(i32));
    f64 *dwork = (f64*)malloc((size_t)ldwork * sizeof(f64));

    if (!hsv || !iwork || !dwork) {
        free(hsv);
        free(iwork);
        free(dwork);
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

    i32 nr = nr_in;
    i32 iwarn, info;

    ab09cx(dico_str, ordsel_str, n, m, p, &nr, a_data, lda, b_data, ldb,
           c_data, ldc, d_data, ldd, hsv, tol1, tol2, iwork, dwork, ldwork,
           &iwarn, &info);

    i32 nmin = iwork[0];

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    npy_intp hsv_dims[1] = {n > 0 ? n : 1};
    PyObject *hsv_array = PyArray_SimpleNew(1, hsv_dims, NPY_DOUBLE);
    if (!hsv_array) {
        free(hsv);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)hsv_array), hsv, (size_t)(n > 0 ? n : 1) * sizeof(f64));
    free(hsv);

    PyObject *result = Py_BuildValue("OOOOiOiii",
        a_array, b_array, c_array, d_array, nr, hsv_array, nmin, iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(hsv_array);

    return result;
}



/* Python wrapper for ab09ix */
PyObject* py_ab09ix(PyObject* self, PyObject* args) {
    const char *dico_str, *job_str, *fact_str, *ordsel_str;
    int n, m, p, nr_in;
    double scalec, scaleo, tol1, tol2;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *ti_obj, *t_obj;

    if (!PyArg_ParseTuple(args, "ssssiiiiddOOOOOOdd",
                          &dico_str, &job_str, &fact_str, &ordsel_str,
                          &n, &m, &p, &nr_in, &scalec, &scaleo,
                          &a_obj, &b_obj, &c_obj, &d_obj, &ti_obj, &t_obj,
                          &tol1, &tol2)) {
        return NULL;
    }

    if (dico_str[0] != 'C' && dico_str[0] != 'c' &&
        dico_str[0] != 'D' && dico_str[0] != 'd') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }

    if (job_str[0] != 'B' && job_str[0] != 'b' &&
        job_str[0] != 'F' && job_str[0] != 'f' &&
        job_str[0] != 'S' && job_str[0] != 's' &&
        job_str[0] != 'P' && job_str[0] != 'p') {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'B', 'F', 'S', or 'P'");
        return NULL;
    }

    if (fact_str[0] != 'S' && fact_str[0] != 's' &&
        fact_str[0] != 'N' && fact_str[0] != 'n') {
        PyErr_SetString(PyExc_ValueError, "FACT must be 'S' or 'N'");
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
    if (fixord && (nr_in < 0 || nr_in > n)) {
        PyErr_SetString(PyExc_ValueError, "NR must be >= 0 and <= N for fixed order");
        return NULL;
    }

    if (scalec <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "SCALEC must be > 0");
        return NULL;
    }
    if (scaleo <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "SCALEO must be > 0");
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

    PyArrayObject *ti_array = (PyArrayObject*)PyArray_FROM_OTF(
        ti_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!ti_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }

    PyArrayObject *t_array = (PyArrayObject*)PyArray_FROM_OTF(
        t_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!t_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(ti_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldd = p > 0 ? p : 1;
    i32 ldti = n > 0 ? n : 1;
    i32 ldt = n > 0 ? n : 1;

    i32 max_mp = (m > p) ? m : p;
    i32 ldwork = 2 * n * n + 5 * n;
    if (n * max_mp > ldwork) ldwork = n * max_mp;
    if (ldwork < 1) ldwork = 1;

    i32 iwork_size = 2 * n;
    if (iwork_size < 1) iwork_size = 1;

    f64 *hsv = (f64*)malloc((n > 0 ? n : 1) * sizeof(f64));
    i32 *iwork = (i32*)malloc(iwork_size * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!hsv || !iwork || !dwork) {
        free(hsv);
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(ti_array);
        Py_DECREF(t_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *ti_data = (f64*)PyArray_DATA(ti_array);
    f64 *t_data = (f64*)PyArray_DATA(t_array);

    i32 nr = nr_in;
    i32 nminr = 0;
    i32 iwarn = 0;
    i32 info = ab09ix(dico_str, job_str, fact_str, ordsel_str,
                      n, m, p, &nr, scalec, scaleo,
                      a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
                      ti_data, ldti, t_data, ldt, &nminr, hsv, tol1, tol2,
                      iwork, dwork, ldwork, &iwarn);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(ti_array);
    PyArray_ResolveWritebackIfCopy(t_array);

    npy_intp hsv_dims[1] = {n > 0 ? n : 1};
    PyObject *hsv_array = PyArray_SimpleNew(1, hsv_dims, NPY_DOUBLE);
    if (!hsv_array) {
        free(hsv);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(ti_array);
        Py_DECREF(t_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)hsv_array), hsv, (size_t)(n > 0 ? n : 1) * sizeof(f64));
    free(hsv);

    PyObject *result = Py_BuildValue("OOOOOOiiOii",
        a_array, b_array, c_array, d_array, ti_array, t_array,
        nr, nminr, hsv_array, iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(ti_array);
    Py_DECREF(t_array);
    Py_DECREF(hsv_array);

    return result;
}


PyObject* py_ab09hy(PyObject* self, PyObject* args) {
    int n, m, p;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "iiiOOOO",
                          &n, &m, &p, &a_obj, &b_obj, &c_obj, &d_obj)) {
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
    if (p < 0 || p > m) {
        PyErr_SetString(PyExc_ValueError, "P must satisfy 0 <= P <= M");
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

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldd = p > 0 ? p : 1;
    i32 lds = n > 0 ? n : 1;
    i32 ldr = n > 0 ? n : 1;

    i32 max_nmp = n > m ? n : m;
    if (p > max_nmp) max_nmp = p;

    i32 lw1 = n * (max_nmp + 5);
    i32 lw2_part = p * (m + 2) > 10 * n * (n + 1) ? p * (m + 2) : 10 * n * (n + 1);
    i32 lw2 = 2 * n * p + lw2_part;
    i32 ldwork = lw1 > lw2 ? lw1 : lw2;
    if (ldwork < 2) ldwork = 2;

    i32 iwork_size = 2 * n > 1 ? 2 * n : 1;
    i32 bwork_size = 2 * n > 1 ? 2 * n : 1;

    f64 *s_data = (f64*)calloc((size_t)lds * (n > 0 ? n : 1), sizeof(f64));
    f64 *r_data = (f64*)calloc((size_t)ldr * (n > 0 ? n : 1), sizeof(f64));
    i32 *iwork = (i32*)malloc(iwork_size * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    i32 *bwork = (i32*)malloc(bwork_size * sizeof(i32));

    if (!s_data || !r_data || !iwork || !dwork || !bwork) {
        free(s_data);
        free(r_data);
        free(iwork);
        free(dwork);
        free(bwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 scalec = 1.0, scaleo = 1.0;
    i32 info = 0;

    ab09hy(n, m, p, a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           &scalec, &scaleo, s_data, lds, r_data, ldr,
           iwork, dwork, ldwork, bwork, &info);

    f64 rcond = dwork[1];

    free(iwork);
    free(dwork);
    free(bwork);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    i32 actual_n = n > 0 ? n : 1;
    npy_intp s_dims_new[2] = {actual_n, actual_n};
    npy_intp s_strides_new[2] = {sizeof(f64), actual_n * sizeof(f64)};
    PyObject *s_array = PyArray_New(&PyArray_Type, 2, s_dims_new, NPY_DOUBLE,
                                    s_strides_new, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!s_array) {
        free(s_data);
        free(r_data);
        return NULL;
    }
    for (i32 j = 0; j < actual_n; j++) {
        memcpy((f64*)PyArray_DATA((PyArrayObject*)s_array) + j * actual_n, s_data + j * lds, (size_t)actual_n * sizeof(f64));
    }
    free(s_data);

    npy_intp r_dims_new[2] = {actual_n, actual_n};
    npy_intp r_strides_new[2] = {sizeof(f64), actual_n * sizeof(f64)};
    PyObject *r_array = PyArray_New(&PyArray_Type, 2, r_dims_new, NPY_DOUBLE,
                                    r_strides_new, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!r_array) {
        Py_DECREF(s_array);
        free(r_data);
        return NULL;
    }
    for (i32 j = 0; j < actual_n; j++) {
        memcpy((f64*)PyArray_DATA((PyArrayObject*)r_array) + j * actual_n, r_data + j * ldr, (size_t)actual_n * sizeof(f64));
    }
    free(r_data);

    PyObject *result = Py_BuildValue("OOdddi",
        s_array, r_array, scalec, scaleo, rcond, info);

    Py_DECREF(s_array);
    Py_DECREF(r_array);

    return result;
}


PyObject* py_ab09jx(PyObject* self, PyObject* args) {
    const char *dico_str, *stdom_str, *evtype_str;
    int n;
    double alpha, tolinf;
    PyObject *er_obj, *ei_obj, *ed_obj;

    if (!PyArg_ParseTuple(args, "sssidOOOd",
                          &dico_str, &stdom_str, &evtype_str,
                          &n, &alpha, &er_obj, &ei_obj, &ed_obj, &tolinf)) {
        return NULL;
    }

    if (dico_str[0] != 'C' && dico_str[0] != 'c' &&
        dico_str[0] != 'D' && dico_str[0] != 'd') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }

    if (stdom_str[0] != 'S' && stdom_str[0] != 's' &&
        stdom_str[0] != 'U' && stdom_str[0] != 'u') {
        PyErr_SetString(PyExc_ValueError, "STDOM must be 'S' or 'U'");
        return NULL;
    }

    if (evtype_str[0] != 'S' && evtype_str[0] != 's' &&
        evtype_str[0] != 'G' && evtype_str[0] != 'g' &&
        evtype_str[0] != 'R' && evtype_str[0] != 'r') {
        PyErr_SetString(PyExc_ValueError, "EVTYPE must be 'S', 'G', or 'R'");
        return NULL;
    }

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "N must be >= 0");
        return NULL;
    }

    if ((dico_str[0] == 'D' || dico_str[0] == 'd') && alpha < 0.0) {
        PyErr_SetString(PyExc_ValueError, "ALPHA must be >= 0 for discrete-time");
        return NULL;
    }

    if (tolinf < 0.0 || tolinf >= 1.0) {
        PyErr_SetString(PyExc_ValueError, "TOLINF must satisfy 0 <= TOLINF < 1");
        return NULL;
    }

    PyArrayObject *er_array = (PyArrayObject*)PyArray_FROM_OTF(
        er_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    PyArrayObject *ei_array = (PyArrayObject*)PyArray_FROM_OTF(
        ei_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    PyArrayObject *ed_array = (PyArrayObject*)PyArray_FROM_OTF(
        ed_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);

    if (!er_array || !ei_array || !ed_array) {
        Py_XDECREF(er_array);
        Py_XDECREF(ei_array);
        Py_XDECREF(ed_array);
        return NULL;
    }

    f64 *er = (f64*)PyArray_DATA(er_array);
    f64 *ei = (f64*)PyArray_DATA(ei_array);
    f64 *ed = (f64*)PyArray_DATA(ed_array);

    i32 info = ab09jx(dico_str, stdom_str, evtype_str, n, alpha, er, ei, ed, tolinf);

    Py_DECREF(er_array);
    Py_DECREF(ei_array);
    Py_DECREF(ed_array);

    if (info < 0) {
        PyErr_Format(PyExc_RuntimeError, "AB09JX: Parameter %d had an illegal value", -info);
        return NULL;
    }

    return Py_BuildValue("i", info);
}


PyObject* py_ab09jv(PyObject* self, PyObject* args) {
    const char *job_str, *dico_str, *jobev_str, *stbchk_str;
    int n, m, p, nv, pv;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    PyObject *av_obj, *ev_obj, *bv_obj, *cv_obj, *dv_obj;

    if (!PyArg_ParseTuple(args, "ssssiiiii" "OOOOOOOOO",
                          &job_str, &dico_str, &jobev_str, &stbchk_str,
                          &n, &m, &p, &nv, &pv,
                          &a_obj, &b_obj, &c_obj, &d_obj,
                          &av_obj, &ev_obj, &bv_obj, &cv_obj, &dv_obj)) {
        return NULL;
    }

    char job = (char)toupper((unsigned char)job_str[0]);
    char dico_c = (char)toupper((unsigned char)dico_str[0]);
    char jobev = (char)toupper((unsigned char)jobev_str[0]);
    char stbchk = (char)toupper((unsigned char)stbchk_str[0]);

    if (job != 'V' && job != 'C') {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'V' or 'C'");
        return NULL;
    }
    if (dico_c != 'C' && dico_c != 'D') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }
    if (jobev != 'G' && jobev != 'I') {
        PyErr_SetString(PyExc_ValueError, "JOBEV must be 'G' or 'I'");
        return NULL;
    }
    if (stbchk != 'C' && stbchk != 'N') {
        PyErr_SetString(PyExc_ValueError, "STBCHK must be 'C' or 'N'");
        return NULL;
    }
    if (n < 0 || m < 0 || p < 0 || nv < 0 || pv < 0) {
        PyErr_SetString(PyExc_ValueError, "N, M, P, NV, PV must be >= 0");
        return NULL;
    }

    bool conjs = (job == 'C');
    bool unitev = (jobev == 'I');

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *av_array = (PyArrayObject*)PyArray_FROM_OTF(
        av_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *ev_array = (PyArrayObject*)PyArray_FROM_OTF(
        ev_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *bv_array = (PyArrayObject*)PyArray_FROM_OTF(
        bv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *cv_array = (PyArrayObject*)PyArray_FROM_OTF(
        cv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *dv_array = (PyArrayObject*)PyArray_FROM_OTF(
        dv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);

    if (!a_array || !b_array || !c_array || !d_array ||
        !av_array || !ev_array || !bv_array || !cv_array || !dv_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        Py_XDECREF(av_array);
        Py_XDECREF(ev_array);
        Py_XDECREF(bv_array);
        Py_XDECREF(cv_array);
        Py_XDECREF(dv_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc_min = 1;
    if (p > ldc_min) ldc_min = p;
    if (pv > ldc_min) ldc_min = pv;
    i32 ldc = ldc_min;
    i32 ldd = ldc_min;
    i32 ldav = nv > 0 ? nv : 1;
    i32 ldev = unitev ? 1 : (nv > 0 ? nv : 1);
    i32 ldbv = nv > 0 ? nv : 1;
    i32 ldcv = conjs ? (p > 0 ? p : 1) : (pv > 0 ? pv : 1);
    i32 lddv = ldcv;

    i32 liwork = unitev ? 1 : (nv + n + 6);
    i32 *iwork = (i32*)calloc(liwork > 0 ? liwork : 1, sizeof(i32));
    if (!iwork) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(av_array);
        Py_DECREF(ev_array);
        Py_DECREF(bv_array);
        Py_DECREF(cv_array);
        Py_DECREF(dv_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 ldwork;
    if (unitev) {
        i32 lw_ia = (dico_c == 'D' && conjs) ? 2 * nv : 0;
        i32 lw1_inner = lw_ia;
        if (pv * n > lw1_inner) lw1_inner = pv * n;
        if (pv * m > lw1_inner) lw1_inner = pv * m;
        i32 lw1_a = nv * (nv + 5);
        i32 lw1_b = nv * n + lw1_inner;
        ldwork = (lw1_a > lw1_b) ? lw1_a : lw1_b;
        if (ldwork < 1) ldwork = 1;
    } else {
        i32 lw2_inner1 = 11 * nv + 16;
        if (p * nv > lw2_inner1) lw2_inner1 = p * nv;
        if (pv * nv > lw2_inner1) lw2_inner1 = pv * nv;
        i32 lw2_inner2 = nv * n + n * n;
        if (pv * n > lw2_inner2) lw2_inner2 = pv * n;
        if (pv * m > lw2_inner2) lw2_inner2 = pv * m;
        i32 lw2_a = 2 * nv * nv + lw2_inner1;
        i32 lw2_b = nv * n + lw2_inner2;
        ldwork = (lw2_a > lw2_b) ? lw2_a : lw2_b;
        if (ldwork < 1) ldwork = 1;
    }

    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (!dwork) {
        free(iwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(av_array);
        Py_DECREF(ev_array);
        Py_DECREF(bv_array);
        Py_DECREF(cv_array);
        Py_DECREF(dv_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *av_data = (f64*)PyArray_DATA(av_array);
    f64 *ev_data = (f64*)PyArray_DATA(ev_array);
    f64 *bv_data = (f64*)PyArray_DATA(bv_array);
    f64 *cv_data = (f64*)PyArray_DATA(cv_array);
    f64 *dv_data = (f64*)PyArray_DATA(dv_array);

    i32 info = 0;
    ab09jv(job_str, dico_str, jobev_str, stbchk_str,
           n, m, p, nv, pv,
           a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           av_data, ldav, ev_data, ldev, bv_data, ldbv, cv_data, ldcv,
           dv_data, lddv, iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    if (info < 0) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(av_array);
        Py_DECREF(ev_array);
        Py_DECREF(bv_array);
        Py_DECREF(cv_array);
        Py_DECREF(dv_array);
        PyErr_Format(PyExc_RuntimeError, "AB09JV: Parameter %d had an illegal value", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(av_array);
    PyArray_ResolveWritebackIfCopy(ev_array);
    PyArray_ResolveWritebackIfCopy(bv_array);
    PyArray_ResolveWritebackIfCopy(cv_array);

    PyObject *result = Py_BuildValue("OOOOOOi",
        c_array, d_array, av_array, ev_array, bv_array, cv_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(av_array);
    Py_DECREF(ev_array);
    Py_DECREF(bv_array);
    Py_DECREF(cv_array);
    Py_DECREF(dv_array);

    return result;
}

/* Python wrapper for ab09jw */
PyObject* py_ab09jw(PyObject* self, PyObject* args) {
    const char *job_str, *dico_str, *jobew_str, *stbchk_str;
    int n, m, p, nw, mw;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    PyObject *aw_obj, *ew_obj, *bw_obj, *cw_obj, *dw_obj;

    if (!PyArg_ParseTuple(args, "ssssiiiii" "OOOOOOOOO",
                          &job_str, &dico_str, &jobew_str, &stbchk_str,
                          &n, &m, &p, &nw, &mw,
                          &a_obj, &b_obj, &c_obj, &d_obj,
                          &aw_obj, &ew_obj, &bw_obj, &cw_obj, &dw_obj)) {
        return NULL;
    }

    char job = (char)toupper((unsigned char)job_str[0]);
    char dico_c = (char)toupper((unsigned char)dico_str[0]);
    char jobew = (char)toupper((unsigned char)jobew_str[0]);
    char stbchk = (char)toupper((unsigned char)stbchk_str[0]);

    if (job != 'W' && job != 'C') {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'W' or 'C'");
        return NULL;
    }
    if (dico_c != 'C' && dico_c != 'D') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }
    if (jobew != 'G' && jobew != 'I') {
        PyErr_SetString(PyExc_ValueError, "JOBEW must be 'G' or 'I'");
        return NULL;
    }
    if (stbchk != 'C' && stbchk != 'N') {
        PyErr_SetString(PyExc_ValueError, "STBCHK must be 'C' or 'N'");
        return NULL;
    }
    if (n < 0 || m < 0 || p < 0 || nw < 0 || mw < 0) {
        PyErr_SetString(PyExc_ValueError, "N, M, P, NW, MW must be >= 0");
        return NULL;
    }

    bool conjs = (job == 'C');
    bool unitew = (jobew == 'I');

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *aw_array = (PyArrayObject*)PyArray_FROM_OTF(
        aw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *ew_array = (PyArrayObject*)PyArray_FROM_OTF(
        ew_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *bw_array = (PyArrayObject*)PyArray_FROM_OTF(
        bw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *cw_array = (PyArrayObject*)PyArray_FROM_OTF(
        cw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *dw_array = (PyArrayObject*)PyArray_FROM_OTF(
        dw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);

    if (!a_array || !b_array || !c_array || !d_array ||
        !aw_array || !ew_array || !bw_array || !cw_array || !dw_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        Py_XDECREF(aw_array);
        Py_XDECREF(ew_array);
        Py_XDECREF(bw_array);
        Py_XDECREF(cw_array);
        Py_XDECREF(dw_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldd = p > 0 ? p : 1;
    i32 ldaw = nw > 0 ? nw : 1;
    i32 ldew = unitew ? 1 : (nw > 0 ? nw : 1);
    i32 ldbw = nw > 0 ? nw : 1;
    i32 ldcw = conjs ? (mw > 0 ? mw : 1) : (m > 0 ? m : 1);
    i32 lddw = ldcw;

    i32 liwork = unitew ? 1 : (nw + n + 6);
    i32 *iwork = (i32*)calloc(liwork > 0 ? liwork : 1, sizeof(i32));
    if (!iwork) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(aw_array);
        Py_DECREF(ew_array);
        Py_DECREF(bw_array);
        Py_DECREF(cw_array);
        Py_DECREF(dw_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 ldwork;
    if (unitew) {
        i32 lw_ia = (dico_c == 'D' && conjs) ? 2 * nw : 0;
        i32 lw1_inner = lw_ia;
        if (n * mw > lw1_inner) lw1_inner = n * mw;
        if (p * mw > lw1_inner) lw1_inner = p * mw;
        i32 lw1_a = nw * (nw + 5);
        i32 lw1_b = nw * n + lw1_inner;
        ldwork = (lw1_a > lw1_b) ? lw1_a : lw1_b;
        if (ldwork < 1) ldwork = 1;
    } else {
        i32 lw2_inner1 = 11 * nw + 16;
        if (nw * m > lw2_inner1) lw2_inner1 = nw * m;
        if (mw * nw > lw2_inner1) lw2_inner1 = mw * nw;
        i32 lw2_inner2 = nw * n + n * n;
        if (mw * n > lw2_inner2) lw2_inner2 = mw * n;
        if (p * mw > lw2_inner2) lw2_inner2 = p * mw;
        i32 lw2_a = 2 * nw * nw + lw2_inner1;
        i32 lw2_b = nw * n + lw2_inner2;
        ldwork = (lw2_a > lw2_b) ? lw2_a : lw2_b;
        if (ldwork < 1) ldwork = 1;
    }

    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (!dwork) {
        free(iwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(aw_array);
        Py_DECREF(ew_array);
        Py_DECREF(bw_array);
        Py_DECREF(cw_array);
        Py_DECREF(dw_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *aw_data = (f64*)PyArray_DATA(aw_array);
    f64 *ew_data = (f64*)PyArray_DATA(ew_array);
    f64 *bw_data = (f64*)PyArray_DATA(bw_array);
    f64 *cw_data = (f64*)PyArray_DATA(cw_array);
    f64 *dw_data = (f64*)PyArray_DATA(dw_array);

    i32 info = 0;
    ab09jw(job_str, dico_str, jobew_str, stbchk_str,
           n, m, p, nw, mw,
           a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           aw_data, ldaw, ew_data, ldew, bw_data, ldbw, cw_data, ldcw,
           dw_data, lddw, iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    if (info < 0) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(aw_array);
        Py_DECREF(ew_array);
        Py_DECREF(bw_array);
        Py_DECREF(cw_array);
        Py_DECREF(dw_array);
        PyErr_Format(PyExc_RuntimeError, "AB09JW: Parameter %d had an illegal value", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(aw_array);
    PyArray_ResolveWritebackIfCopy(ew_array);
    PyArray_ResolveWritebackIfCopy(bw_array);
    PyArray_ResolveWritebackIfCopy(cw_array);

    PyObject *result = Py_BuildValue("OOOOOOi",
        b_array, d_array, aw_array, ew_array, bw_array, cw_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(aw_array);
    Py_DECREF(ew_array);
    Py_DECREF(bw_array);
    Py_DECREF(cw_array);
    Py_DECREF(dw_array);

    return result;
}


PyObject* py_ab09kx(PyObject* self, PyObject* args) {
    const char *job_str, *dico_str, *weight_str;
    int n, nv, nw, m, p;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    PyObject *av_obj, *bv_obj, *cv_obj, *dv_obj;
    PyObject *aw_obj, *bw_obj, *cw_obj, *dw_obj;

    if (!PyArg_ParseTuple(args, "sssiiiii" "OOOOOOOOOOOO",
                          &job_str, &dico_str, &weight_str,
                          &n, &nv, &nw, &m, &p,
                          &a_obj, &b_obj, &c_obj, &d_obj,
                          &av_obj, &bv_obj, &cv_obj, &dv_obj,
                          &aw_obj, &bw_obj, &cw_obj, &dw_obj)) {
        return NULL;
    }

    char job = (char)toupper((unsigned char)job_str[0]);
    char dico_c = (char)toupper((unsigned char)dico_str[0]);
    char weight = (char)toupper((unsigned char)weight_str[0]);

    if (job != 'N' && job != 'C') {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'N' or 'C'");
        return NULL;
    }
    if (dico_c != 'C' && dico_c != 'D') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }
    if (weight != 'N' && weight != 'L' && weight != 'R' && weight != 'B') {
        PyErr_SetString(PyExc_ValueError, "WEIGHT must be 'N', 'L', 'R', or 'B'");
        return NULL;
    }
    if (n < 0 || nv < 0 || nw < 0 || m < 0 || p < 0) {
        PyErr_SetString(PyExc_ValueError, "N, NV, NW, M, P must be >= 0");
        return NULL;
    }

    bool leftw = (weight == 'L' || weight == 'B');
    bool rightw = (weight == 'R' || weight == 'B');

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *av_array = (PyArrayObject*)PyArray_FROM_OTF(
        av_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *bv_array = (PyArrayObject*)PyArray_FROM_OTF(
        bv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *cv_array = (PyArrayObject*)PyArray_FROM_OTF(
        cv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *dv_array = (PyArrayObject*)PyArray_FROM_OTF(
        dv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    PyArrayObject *aw_array = (PyArrayObject*)PyArray_FROM_OTF(
        aw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *bw_array = (PyArrayObject*)PyArray_FROM_OTF(
        bw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *cw_array = (PyArrayObject*)PyArray_FROM_OTF(
        cw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *dw_array = (PyArrayObject*)PyArray_FROM_OTF(
        dw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);

    if (!a_array || !b_array || !c_array || !d_array ||
        !av_array || !bv_array || !cv_array || !dv_array ||
        !aw_array || !bw_array || !cw_array || !dw_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        Py_XDECREF(av_array);
        Py_XDECREF(bv_array);
        Py_XDECREF(cv_array);
        Py_XDECREF(dv_array);
        Py_XDECREF(aw_array);
        Py_XDECREF(bw_array);
        Py_XDECREF(cw_array);
        Py_XDECREF(dw_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldd = p > 0 ? p : 1;
    i32 ldav = leftw ? (nv > 0 ? nv : 1) : 1;
    i32 ldbv = leftw ? (nv > 0 ? nv : 1) : 1;
    i32 ldcv = leftw ? (p > 0 ? p : 1) : 1;
    i32 lddv = leftw ? (p > 0 ? p : 1) : 1;
    i32 ldaw = rightw ? (nw > 0 ? nw : 1) : 1;
    i32 ldbw = rightw ? (nw > 0 ? nw : 1) : 1;
    i32 ldcw = rightw ? (m > 0 ? m : 1) : 1;
    i32 lddw = rightw ? (m > 0 ? m : 1) : 1;

    bool conjs = (job == 'C');
    bool discr = (dico_c == 'D');
    i32 ia = (discr && conjs) ? 2 * nv : 0;
    i32 ib = (discr && conjs) ? 2 * nw : 0;

    i32 ldwork = 1;
    if (leftw) {
        i32 t1 = nv * (nv + 5);
        i32 t2a = (ia > p * n) ? ia : p * n;
        i32 t2 = (t2a > p * m) ? t2a : p * m;
        i32 t3 = nv * n + t2;
        ldwork = (ldwork > t1) ? ldwork : t1;
        ldwork = (ldwork > t3) ? ldwork : t3;
    }
    if (rightw) {
        i32 t1 = nw * (nw + 5);
        i32 t2a = (ib > m * n) ? ib : m * n;
        i32 t2 = (t2a > p * m) ? t2a : p * m;
        i32 t3 = nw * n + t2;
        ldwork = (ldwork > t1) ? ldwork : t1;
        ldwork = (ldwork > t3) ? ldwork : t3;
    }

    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (!dwork) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(av_array);
        Py_DECREF(bv_array);
        Py_DECREF(cv_array);
        Py_DECREF(dv_array);
        Py_DECREF(aw_array);
        Py_DECREF(bw_array);
        Py_DECREF(cw_array);
        Py_DECREF(dw_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *av_data = (f64*)PyArray_DATA(av_array);
    f64 *bv_data = (f64*)PyArray_DATA(bv_array);
    f64 *cv_data = (f64*)PyArray_DATA(cv_array);
    f64 *dv_data = (f64*)PyArray_DATA(dv_array);
    f64 *aw_data = (f64*)PyArray_DATA(aw_array);
    f64 *bw_data = (f64*)PyArray_DATA(bw_array);
    f64 *cw_data = (f64*)PyArray_DATA(cw_array);
    f64 *dw_data = (f64*)PyArray_DATA(dw_array);

    i32 iwarn = 0, info = 0;
    ab09kx(job_str, dico_str, weight_str,
           n, nv, nw, m, p,
           a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           av_data, ldav, bv_data, ldbv, cv_data, ldcv, dv_data, lddv,
           aw_data, ldaw, bw_data, ldbw, cw_data, ldcw, dw_data, lddw,
           dwork, ldwork, &iwarn, &info);

    free(dwork);

    if (info < 0) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(av_array);
        Py_DECREF(bv_array);
        Py_DECREF(cv_array);
        Py_DECREF(dv_array);
        Py_DECREF(aw_array);
        Py_DECREF(bw_array);
        Py_DECREF(cw_array);
        Py_DECREF(dw_array);
        PyErr_Format(PyExc_RuntimeError, "AB09KX: Parameter %d had an illegal value", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(av_array);
    PyArray_ResolveWritebackIfCopy(bv_array);
    PyArray_ResolveWritebackIfCopy(cv_array);
    PyArray_ResolveWritebackIfCopy(aw_array);
    PyArray_ResolveWritebackIfCopy(bw_array);
    PyArray_ResolveWritebackIfCopy(cw_array);

    PyObject *result = Py_BuildValue("OOOOOOOOOii",
        b_array, c_array, d_array,
        av_array, bv_array, cv_array,
        aw_array, bw_array, cw_array,
        iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(av_array);
    Py_DECREF(bv_array);
    Py_DECREF(cv_array);
    Py_DECREF(dv_array);
    Py_DECREF(aw_array);
    Py_DECREF(bw_array);
    Py_DECREF(cw_array);
    Py_DECREF(dw_array);

    return result;
}


/* Python wrapper for ab09jd - Frequency-weighted Hankel-norm approximation with invertible weights */
PyObject* py_ab09jd(PyObject* self, PyObject* args) {
    const char *jobv_str, *jobw_str, *jobinv_str, *dico_str, *equil_str, *ordsel_str;
    int n, nv, nw, m, p, nr_in;
    double alpha, tol1, tol2;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    PyObject *av_obj, *bv_obj, *cv_obj, *dv_obj;
    PyObject *aw_obj, *bw_obj, *cw_obj, *dw_obj;

    if (!PyArg_ParseTuple(args, "ssssssiiiiiid" "OOOOOOOOOOOO" "dd",
                          &jobv_str, &jobw_str, &jobinv_str, &dico_str, &equil_str, &ordsel_str,
                          &n, &nv, &nw, &m, &p, &nr_in, &alpha,
                          &a_obj, &b_obj, &c_obj, &d_obj,
                          &av_obj, &bv_obj, &cv_obj, &dv_obj,
                          &aw_obj, &bw_obj, &cw_obj, &dw_obj,
                          &tol1, &tol2)) {
        return NULL;
    }

    char jobv = (char)toupper((unsigned char)jobv_str[0]);
    char jobw = (char)toupper((unsigned char)jobw_str[0]);
    char jobinv = (char)toupper((unsigned char)jobinv_str[0]);
    char dico_c = (char)toupper((unsigned char)dico_str[0]);
    char equil = (char)toupper((unsigned char)equil_str[0]);
    char ordsel = (char)toupper((unsigned char)ordsel_str[0]);

    if (jobv != 'N' && jobv != 'V' && jobv != 'I' && jobv != 'C' && jobv != 'R') {
        PyErr_SetString(PyExc_ValueError, "JOBV must be 'N', 'V', 'I', 'C', or 'R'");
        return NULL;
    }
    if (jobw != 'N' && jobw != 'W' && jobw != 'I' && jobw != 'C' && jobw != 'R') {
        PyErr_SetString(PyExc_ValueError, "JOBW must be 'N', 'W', 'I', 'C', or 'R'");
        return NULL;
    }
    if (jobinv != 'N' && jobinv != 'I' && jobinv != 'A') {
        PyErr_SetString(PyExc_ValueError, "JOBINV must be 'N', 'I', or 'A'");
        return NULL;
    }
    if (dico_c != 'C' && dico_c != 'D') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }
    if (equil != 'S' && equil != 'N') {
        PyErr_SetString(PyExc_ValueError, "EQUIL must be 'S' or 'N'");
        return NULL;
    }
    if (ordsel != 'F' && ordsel != 'A') {
        PyErr_SetString(PyExc_ValueError, "ORDSEL must be 'F' or 'A'");
        return NULL;
    }
    if (n < 0 || nv < 0 || nw < 0 || m < 0 || p < 0) {
        PyErr_SetString(PyExc_ValueError, "N, NV, NW, M, P must be >= 0");
        return NULL;
    }

    bool leftw = (jobv != 'N');
    bool rightw = (jobw != 'N');

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *av_array = (PyArrayObject*)PyArray_FROM_OTF(
        av_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *bv_array = (PyArrayObject*)PyArray_FROM_OTF(
        bv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *cv_array = (PyArrayObject*)PyArray_FROM_OTF(
        cv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *dv_array = (PyArrayObject*)PyArray_FROM_OTF(
        dv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *aw_array = (PyArrayObject*)PyArray_FROM_OTF(
        aw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *bw_array = (PyArrayObject*)PyArray_FROM_OTF(
        bw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *cw_array = (PyArrayObject*)PyArray_FROM_OTF(
        cw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *dw_array = (PyArrayObject*)PyArray_FROM_OTF(
        dw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!a_array || !b_array || !c_array || !d_array ||
        !av_array || !bv_array || !cv_array || !dv_array ||
        !aw_array || !bw_array || !cw_array || !dw_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        Py_XDECREF(av_array);
        Py_XDECREF(bv_array);
        Py_XDECREF(cv_array);
        Py_XDECREF(dv_array);
        Py_XDECREF(aw_array);
        Py_XDECREF(bw_array);
        Py_XDECREF(cw_array);
        Py_XDECREF(dw_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldd = p > 0 ? p : 1;
    i32 ldav = leftw ? (nv > 0 ? nv : 1) : 1;
    i32 ldbv = leftw ? (nv > 0 ? nv : 1) : 1;
    i32 ldcv = leftw ? (p > 0 ? p : 1) : 1;
    i32 lddv = leftw ? (p > 0 ? p : 1) : 1;
    i32 ldaw = rightw ? (nw > 0 ? nw : 1) : 1;
    i32 ldbw = rightw ? (nw > 0 ? nw : 1) : 1;
    i32 ldcw = rightw ? (m > 0 ? m : 1) : 1;
    i32 lddw = rightw ? (m > 0 ? m : 1) : 1;

    i32 nvp = nv + p;
    i32 nwm = nw + m;

    i32 maxnmp = n;
    if (m > maxnmp) maxnmp = m;
    if (p > maxnmp) maxnmp = p;
    i32 minnm = n < m ? n : m;

    i32 ldwork = 1;
    if (leftw) {
        i32 t1 = 2 * nvp * (nvp + p) + p * p;
        i32 t2a = 2 * nvp * nvp + (11 * nvp + 16 > p * nvp ? 11 * nvp + 16 : p * nvp);
        i32 t2b = nvp * n + (nvp * n + n * n > p * n ? (nvp * n + n * n > p * m ? nvp * n + n * n : p * m) : (p * n > p * m ? p * n : p * m));
        ldwork = ldwork > (t1 + (t2a > t2b ? t2a : t2b)) ? ldwork : (t1 + (t2a > t2b ? t2a : t2b));
    }
    if (rightw) {
        i32 t1 = 2 * nwm * (nwm + m) + m * m;
        i32 t2a = 2 * nwm * nwm + (11 * nwm + 16 > m * nwm ? 11 * nwm + 16 : m * nwm);
        i32 t2b = nwm * n + (nwm * n + n * n > m * n ? (nwm * n + n * n > p * m ? nwm * n + n * n : p * m) : (m * n > p * m ? m * n : p * m));
        ldwork = ldwork > (t1 + (t2a > t2b ? t2a : t2b)) ? ldwork : (t1 + (t2a > t2b ? t2a : t2b));
    }
    i32 ldw3 = n * (2 * n + maxnmp + 5) + (n * (n + 1)) / 2;
    i32 ldw4 = n * (m + p + 2) + 2 * m * p + minnm;
    i32 t1 = 3 * m + 1;
    i32 t2 = minnm + p;
    ldw4 += (t1 > t2) ? t1 : t2;
    ldwork = (ldwork > ldw3) ? ldwork : ldw3;
    ldwork = (ldwork > ldw4) ? ldwork : ldw4;

    i32 liwork = 1;
    if (m > liwork) liwork = m;
    if (n > liwork) liwork = n;
    if (leftw && 2 * p + nvp + n + 6 > liwork) liwork = 2 * p + nvp + n + 6;
    if (rightw && 2 * m + nwm + n + 6 > liwork) liwork = 2 * m + nwm + n + 6;

    f64 *dwork = (f64*)calloc(ldwork + 1, sizeof(f64));
    i32 *iwork = (i32*)calloc(liwork + 1, sizeof(i32));
    f64 *hsv = (f64*)calloc(n > 0 ? n : 1, sizeof(f64));

    if (!dwork || !iwork || !hsv) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(av_array);
        Py_DECREF(bv_array);
        Py_DECREF(cv_array);
        Py_DECREF(dv_array);
        Py_DECREF(aw_array);
        Py_DECREF(bw_array);
        Py_DECREF(cw_array);
        Py_DECREF(dw_array);
        free(dwork);
        free(iwork);
        free(hsv);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *av_data = (f64*)PyArray_DATA(av_array);
    f64 *bv_data = (f64*)PyArray_DATA(bv_array);
    f64 *cv_data = (f64*)PyArray_DATA(cv_array);
    f64 *dv_data = (f64*)PyArray_DATA(dv_array);
    f64 *aw_data = (f64*)PyArray_DATA(aw_array);
    f64 *bw_data = (f64*)PyArray_DATA(bw_array);
    f64 *cw_data = (f64*)PyArray_DATA(cw_array);
    f64 *dw_data = (f64*)PyArray_DATA(dw_array);

    i32 nr = nr_in;
    i32 ns = 0;
    i32 iwarn = 0, info = 0;

    ab09jd(jobv_str, jobw_str, jobinv_str, dico_str, equil_str, ordsel_str,
           n, nv, nw, m, p, &nr, alpha,
           a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           av_data, ldav, bv_data, ldbv, cv_data, ldcv, dv_data, lddv,
           aw_data, ldaw, bw_data, ldbw, cw_data, ldcw, dw_data, lddw,
           &ns, hsv, tol1, tol2, iwork, dwork, ldwork, &iwarn, &info);

    free(dwork);
    free(iwork);

    if (info < 0) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(av_array);
        Py_DECREF(bv_array);
        Py_DECREF(cv_array);
        Py_DECREF(dv_array);
        Py_DECREF(aw_array);
        Py_DECREF(bw_array);
        Py_DECREF(cw_array);
        Py_DECREF(dw_array);
        free(hsv);
        PyErr_Format(PyExc_RuntimeError, "AB09JD: Parameter %d had an illegal value", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(av_array);
    PyArray_ResolveWritebackIfCopy(bv_array);
    PyArray_ResolveWritebackIfCopy(cv_array);
    PyArray_ResolveWritebackIfCopy(dv_array);
    PyArray_ResolveWritebackIfCopy(aw_array);
    PyArray_ResolveWritebackIfCopy(bw_array);
    PyArray_ResolveWritebackIfCopy(cw_array);
    PyArray_ResolveWritebackIfCopy(dw_array);

    npy_intp hsv_dims[1] = {n > 0 ? n : 1};
    PyObject *hsv_array = PyArray_SimpleNew(1, hsv_dims, NPY_DOUBLE);
    if (hsv_array) {
        memcpy(PyArray_DATA((PyArrayObject*)hsv_array), hsv, (size_t)(n > 0 ? n : 1) * sizeof(f64));
    }
    free(hsv);

    PyObject *result = Py_BuildValue("OOOOOOOOOOiiOii",
        a_array, b_array, c_array, d_array,
        av_array, bv_array, cv_array,
        aw_array, bw_array, cw_array,
        nr, ns, hsv_array,
        iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(av_array);
    Py_DECREF(bv_array);
    Py_DECREF(cv_array);
    Py_DECREF(dv_array);
    Py_DECREF(aw_array);
    Py_DECREF(bw_array);
    Py_DECREF(cw_array);
    Py_DECREF(dw_array);
    Py_XDECREF(hsv_array);

    return result;
}


/* Python wrapper for ab09kd - Frequency-weighted Hankel-norm approximation */
PyObject* py_ab09kd(PyObject* self, PyObject* args) {
    const char *job_str, *dico_str, *weight_str, *equil_str, *ordsel_str;
    int n, nv, nw, m, p, nr_in;
    double alpha, tol1, tol2;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    PyObject *av_obj, *bv_obj, *cv_obj, *dv_obj;
    PyObject *aw_obj, *bw_obj, *cw_obj, *dw_obj;

    if (!PyArg_ParseTuple(args, "sssssiiiiiid" "OOOOOOOOOOOO" "dd",
                          &job_str, &dico_str, &weight_str, &equil_str, &ordsel_str,
                          &n, &nv, &nw, &m, &p, &nr_in, &alpha,
                          &a_obj, &b_obj, &c_obj, &d_obj,
                          &av_obj, &bv_obj, &cv_obj, &dv_obj,
                          &aw_obj, &bw_obj, &cw_obj, &dw_obj,
                          &tol1, &tol2)) {
        return NULL;
    }

    char job = (char)toupper((unsigned char)job_str[0]);
    char dico_c = (char)toupper((unsigned char)dico_str[0]);
    char weight = (char)toupper((unsigned char)weight_str[0]);
    char equil = (char)toupper((unsigned char)equil_str[0]);
    char ordsel = (char)toupper((unsigned char)ordsel_str[0]);

    if (job != 'N' && job != 'C') {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'N' or 'C'");
        return NULL;
    }
    if (dico_c != 'C' && dico_c != 'D') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }
    if (weight != 'N' && weight != 'L' && weight != 'R' && weight != 'B') {
        PyErr_SetString(PyExc_ValueError, "WEIGHT must be 'N', 'L', 'R', or 'B'");
        return NULL;
    }
    if (equil != 'S' && equil != 'N') {
        PyErr_SetString(PyExc_ValueError, "EQUIL must be 'S' or 'N'");
        return NULL;
    }
    if (ordsel != 'F' && ordsel != 'A') {
        PyErr_SetString(PyExc_ValueError, "ORDSEL must be 'F' or 'A'");
        return NULL;
    }
    if (n < 0 || nv < 0 || nw < 0 || m < 0 || p < 0) {
        PyErr_SetString(PyExc_ValueError, "N, NV, NW, M, P must be >= 0");
        return NULL;
    }

    bool leftw = (weight == 'L' || weight == 'B');
    bool rightw = (weight == 'R' || weight == 'B');

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *av_array = (PyArrayObject*)PyArray_FROM_OTF(
        av_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *bv_array = (PyArrayObject*)PyArray_FROM_OTF(
        bv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *cv_array = (PyArrayObject*)PyArray_FROM_OTF(
        cv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *dv_array = (PyArrayObject*)PyArray_FROM_OTF(
        dv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *aw_array = (PyArrayObject*)PyArray_FROM_OTF(
        aw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *bw_array = (PyArrayObject*)PyArray_FROM_OTF(
        bw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *cw_array = (PyArrayObject*)PyArray_FROM_OTF(
        cw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *dw_array = (PyArrayObject*)PyArray_FROM_OTF(
        dw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!a_array || !b_array || !c_array || !d_array ||
        !av_array || !bv_array || !cv_array || !dv_array ||
        !aw_array || !bw_array || !cw_array || !dw_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        Py_XDECREF(av_array);
        Py_XDECREF(bv_array);
        Py_XDECREF(cv_array);
        Py_XDECREF(dv_array);
        Py_XDECREF(aw_array);
        Py_XDECREF(bw_array);
        Py_XDECREF(cw_array);
        Py_XDECREF(dw_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldd = p > 0 ? p : 1;
    i32 ldav = leftw ? (nv > 0 ? nv : 1) : 1;
    i32 ldbv = leftw ? (nv > 0 ? nv : 1) : 1;
    i32 ldcv = leftw ? (p > 0 ? p : 1) : 1;
    i32 lddv = leftw ? (p > 0 ? p : 1) : 1;
    i32 ldaw = rightw ? (nw > 0 ? nw : 1) : 1;
    i32 ldbw = rightw ? (nw > 0 ? nw : 1) : 1;
    i32 ldcw = rightw ? (m > 0 ? m : 1) : 1;
    i32 lddw = rightw ? (m > 0 ? m : 1) : 1;

    bool conjs = (job == 'C');
    bool discr = (dico_c == 'D');
    i32 ia = (discr && conjs) ? 2 * nv : 0;
    i32 ib = (discr && conjs) ? 2 * nw : 0;

    i32 maxnmp = n;
    if (m > maxnmp) maxnmp = m;
    if (p > maxnmp) maxnmp = p;
    i32 minnm = n < m ? n : m;

    i32 ldwork = 1;
    if (leftw) {
        i32 t1 = nv * (nv + 5);
        i32 t2a = (ia > p * n) ? ia : p * n;
        i32 t2 = (t2a > p * m) ? t2a : p * m;
        i32 t3 = nv * n + t2;
        ldwork = (ldwork > t1) ? ldwork : t1;
        ldwork = (ldwork > t3) ? ldwork : t3;
    }
    if (rightw) {
        i32 t1 = nw * (nw + 5);
        i32 t2a = (ib > m * n) ? ib : m * n;
        i32 t2 = (t2a > p * m) ? t2a : p * m;
        i32 t3 = nw * n + t2;
        ldwork = (ldwork > t1) ? ldwork : t1;
        ldwork = (ldwork > t3) ? ldwork : t3;
    }
    i32 ldw3 = n * (2 * n + maxnmp + 5) + (n * (n + 1)) / 2;
    i32 ldw4 = n * (m + p + 2) + 2 * m * p + minnm;
    i32 t1 = 3 * m + 1;
    i32 t2 = minnm + p;
    ldw4 += (t1 > t2) ? t1 : t2;
    ldwork = (ldwork > ldw3) ? ldwork : ldw3;
    ldwork = (ldwork > ldw4) ? ldwork : ldw4;

    i32 liwork = 1;
    if (m > liwork) liwork = m;
    if (discr && n > liwork) liwork = n;
    if (weight == 'L' && 2 * p > liwork) liwork = 2 * p;
    if (weight == 'R' && 2 * m > liwork) liwork = 2 * m;
    if (weight == 'B') {
        i32 c = (2 * m > 2 * p) ? 2 * m : 2 * p;
        if (c > liwork) liwork = c;
    }

    f64 *dwork = (f64*)calloc(ldwork + 1, sizeof(f64));
    i32 *iwork = (i32*)calloc(liwork + 1, sizeof(i32));
    f64 *hsv = (f64*)calloc(n > 0 ? n : 1, sizeof(f64));

    if (!dwork || !iwork || !hsv) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(av_array);
        Py_DECREF(bv_array);
        Py_DECREF(cv_array);
        Py_DECREF(dv_array);
        Py_DECREF(aw_array);
        Py_DECREF(bw_array);
        Py_DECREF(cw_array);
        Py_DECREF(dw_array);
        free(dwork);
        free(iwork);
        free(hsv);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *av_data = (f64*)PyArray_DATA(av_array);
    f64 *bv_data = (f64*)PyArray_DATA(bv_array);
    f64 *cv_data = (f64*)PyArray_DATA(cv_array);
    f64 *dv_data = (f64*)PyArray_DATA(dv_array);
    f64 *aw_data = (f64*)PyArray_DATA(aw_array);
    f64 *bw_data = (f64*)PyArray_DATA(bw_array);
    f64 *cw_data = (f64*)PyArray_DATA(cw_array);
    f64 *dw_data = (f64*)PyArray_DATA(dw_array);

    i32 nr = nr_in;
    i32 ns = 0;
    i32 iwarn = 0, info = 0;

    ab09kd(job_str, dico_str, weight_str, equil_str, ordsel_str,
           n, nv, nw, m, p, &nr, alpha,
           a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           av_data, ldav, bv_data, ldbv, cv_data, ldcv, dv_data, lddv,
           aw_data, ldaw, bw_data, ldbw, cw_data, ldcw, dw_data, lddw,
           &ns, hsv, tol1, tol2, iwork, dwork, ldwork, &iwarn, &info);

    free(dwork);
    free(iwork);

    if (info < 0) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(av_array);
        Py_DECREF(bv_array);
        Py_DECREF(cv_array);
        Py_DECREF(dv_array);
        Py_DECREF(aw_array);
        Py_DECREF(bw_array);
        Py_DECREF(cw_array);
        Py_DECREF(dw_array);
        free(hsv);
        PyErr_Format(PyExc_RuntimeError, "AB09KD: Parameter %d had an illegal value", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(av_array);
    PyArray_ResolveWritebackIfCopy(bv_array);
    PyArray_ResolveWritebackIfCopy(cv_array);
    PyArray_ResolveWritebackIfCopy(dv_array);
    PyArray_ResolveWritebackIfCopy(aw_array);
    PyArray_ResolveWritebackIfCopy(bw_array);
    PyArray_ResolveWritebackIfCopy(cw_array);
    PyArray_ResolveWritebackIfCopy(dw_array);

    npy_intp hsv_dims[1] = {n > 0 ? n : 1};
    PyObject *hsv_array = PyArray_SimpleNew(1, hsv_dims, NPY_DOUBLE);
    if (hsv_array) {
        memcpy(PyArray_DATA((PyArrayObject*)hsv_array), hsv, (size_t)(n > 0 ? n : 1) * sizeof(f64));
    }
    free(hsv);

    PyObject *result = Py_BuildValue("OOOOOOOOOOiiOii",
        a_array, b_array, c_array, d_array,
        av_array, bv_array, cv_array,
        aw_array, bw_array, cw_array,
        nr, ns, hsv_array,
        iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(av_array);
    Py_DECREF(bv_array);
    Py_DECREF(cv_array);
    Py_DECREF(dv_array);
    Py_DECREF(aw_array);
    Py_DECREF(bw_array);
    Py_DECREF(cw_array);
    Py_DECREF(dw_array);
    Py_XDECREF(hsv_array);

    return result;
}


/* Python wrapper for ab13ad */
PyObject* py_ab13ad(PyObject* self, PyObject* args) {
    const char *dico_str, *equil_str;
    int n, m, p;
    double alpha;
    PyObject *a_obj, *b_obj, *c_obj;

    if (!PyArg_ParseTuple(args, "ssiiidOOO", &dico_str, &equil_str,
                          &n, &m, &p, &alpha, &a_obj, &b_obj, &c_obj)) {
        return NULL;
    }

    if (dico_str[0] != 'C' && dico_str[0] != 'c' &&
        dico_str[0] != 'D' && dico_str[0] != 'd') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }

    if (equil_str[0] != 'S' && equil_str[0] != 's' &&
        equil_str[0] != 'N' && equil_str[0] != 'n') {
        PyErr_SetString(PyExc_ValueError, "EQUIL must be 'S' or 'N'");
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

    bool discr = (dico_str[0] == 'D' || dico_str[0] == 'd');
    if (discr && (alpha < 0.0 || alpha > 1.0)) {
        PyErr_SetString(PyExc_ValueError, "ALPHA must be in [0, 1] for discrete-time");
        return NULL;
    }
    if (!discr && alpha > 0.0) {
        PyErr_SetString(PyExc_ValueError, "ALPHA must be <= 0 for continuous-time");
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

    i32 maxnmp = n;
    if (m > maxnmp) maxnmp = m;
    if (p > maxnmp) maxnmp = p;

    i32 ldwork = n > 0 ? n * (maxnmp + 5) + (n * (n + 1)) / 2 : 1;
    ldwork = ldwork > 1 ? ldwork : 1;
    ldwork *= 2;

    f64 *hsv = (f64*)malloc((size_t)(n > 0 ? n : 1) * sizeof(f64));
    f64 *dwork = (f64*)malloc((size_t)ldwork * sizeof(f64));

    if (!hsv || !dwork) {
        free(hsv);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    i32 ns;
    i32 info;

    f64 hankel_norm = ab13ad(dico_str, equil_str, n, m, p, alpha,
                             a_data, lda, b_data, ldb, c_data, ldc,
                             &ns, hsv, dwork, ldwork, &info);

    free(dwork);

    if (info < 0) {
        free(hsv);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_Format(PyExc_RuntimeError, "AB13AD: Parameter %d had an illegal value", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);

    npy_intp hsv_dims[1] = {n > 0 ? n : 1};
    PyObject *hsv_array = PyArray_SimpleNew(1, hsv_dims, NPY_DOUBLE);
    if (!hsv_array) {
        free(hsv);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)hsv_array), hsv, (size_t)(n > 0 ? n : 1) * sizeof(f64));
    free(hsv);

    PyObject *result = Py_BuildValue("diOi", hankel_norm, ns, hsv_array, info);
    Py_DECREF(hsv_array);

    return result;
}



/* Python wrapper for ab13cd */
PyObject* py_ab13cd(PyObject* self, PyObject* args) {
    (void)self;
    i32 n, m, np;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    f64 tol;

    if (!PyArg_ParseTuple(args, "iiiOOOOd", &n, &m, &np, &a_obj, &b_obj, &c_obj, &d_obj, &tol)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!a_array) return NULL;

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
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

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!d_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = np > 0 ? np : 1;
    i32 ldd = np > 0 ? np : 1;

    i32 maxmnp = m > np ? m : np;
    i32 minwrk = 4*n*n + 2*m*m + 3*m*n + m*np + 2*(n+np)*np + 10*n + 6*maxmnp;
    minwrk = minwrk > 2 ? minwrk : 2;
    i32 ldwork = minwrk * 2;

    i32 mincwr = (n + m) * (n + np) + 3 * maxmnp;
    mincwr = mincwr > 1 ? mincwr : 1;
    i32 lcwork = mincwr * 2;

    i32 iwork_size = n > 0 ? n : 1;
    i32 bwork_size = 2 * n > 0 ? 2 * n : 1;

    i32 *iwork = (i32*)malloc((size_t)iwork_size * sizeof(i32));
    f64 *dwork = (f64*)malloc((size_t)ldwork * sizeof(f64));
    c128 *cwork = (c128*)malloc((size_t)lcwork * sizeof(c128));
    i32 *bwork = (i32*)malloc((size_t)bwork_size * sizeof(i32));

    if (!iwork || !dwork || !cwork || !bwork) {
        free(iwork);
        free(dwork);
        free(cwork);
        free(bwork);
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

    f64 fpeak;
    i32 info;

    f64 hnorm = ab13cd(n, m, np, a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
                       tol, iwork, dwork, ldwork, cwork, lcwork, bwork, &fpeak, &info);

    free(iwork);
    free(dwork);
    free(cwork);
    free(bwork);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    return Py_BuildValue("ddi", hnorm, fpeak, info);
}

/* Python wrapper for ab09iy */
PyObject* py_ab09iy(PyObject* self, PyObject* args) {
    const char *dico_str, *jobc_str, *jobo_str, *weight_str;
    i32 n, m, p, nv, pv, nw, mw;
    f64 alphac, alphao;
    PyObject *a_obj, *b_obj, *c_obj;
    PyObject *av_obj, *bv_obj, *cv_obj, *dv_obj;
    PyObject *aw_obj, *bw_obj, *cw_obj, *dw_obj;

    if (!PyArg_ParseTuple(args, "ssssiiiiiiiddOOOOOOOOOOO",
                          &dico_str, &jobc_str, &jobo_str, &weight_str,
                          &n, &m, &p, &nv, &pv, &nw, &mw,
                          &alphac, &alphao,
                          &a_obj, &b_obj, &c_obj,
                          &av_obj, &bv_obj, &cv_obj, &dv_obj,
                          &aw_obj, &bw_obj, &cw_obj, &dw_obj)) {
        return NULL;
    }

    if (n < 0 || m < 0 || p < 0 || nv < 0 || pv < 0 || nw < 0 || mw < 0) {
        PyErr_SetString(PyExc_ValueError, "Dimensions must be non-negative");
        return NULL;
    }

    char dico = (char)toupper((unsigned char)dico_str[0]);
    char jobc = (char)toupper((unsigned char)jobc_str[0]);
    char jobo = (char)toupper((unsigned char)jobo_str[0]);
    char weight = (char)toupper((unsigned char)weight_str[0]);

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    PyArrayObject *av_array = (PyArrayObject*)PyArray_FROM_OTF(
        av_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    PyArrayObject *bv_array = (PyArrayObject*)PyArray_FROM_OTF(
        bv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    PyArrayObject *cv_array = (PyArrayObject*)PyArray_FROM_OTF(
        cv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    PyArrayObject *dv_array = (PyArrayObject*)PyArray_FROM_OTF(
        dv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    PyArrayObject *aw_array = (PyArrayObject*)PyArray_FROM_OTF(
        aw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    PyArrayObject *bw_array = (PyArrayObject*)PyArray_FROM_OTF(
        bw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    PyArrayObject *cw_array = (PyArrayObject*)PyArray_FROM_OTF(
        cw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);
    PyArrayObject *dw_array = (PyArrayObject*)PyArray_FROM_OTF(
        dw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_FORCECAST);

    if (!a_array || !b_array || !c_array ||
        !av_array || !bv_array || !cv_array || !dv_array ||
        !aw_array || !bw_array || !cw_array || !dw_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(av_array);
        Py_XDECREF(bv_array);
        Py_XDECREF(cv_array);
        Py_XDECREF(dv_array);
        Py_XDECREF(aw_array);
        Py_XDECREF(bw_array);
        Py_XDECREF(cw_array);
        Py_XDECREF(dw_array);
        PyErr_SetString(PyExc_ValueError, "Failed to convert input arrays");
        return NULL;
    }

    i32 lda = n > 1 ? n : 1;
    i32 ldb = n > 1 ? n : 1;
    i32 ldc = p > 1 ? p : 1;
    i32 ldav = nv > 1 ? nv : 1;
    i32 ldbv = nv > 1 ? nv : 1;
    i32 ldcv = pv > 1 ? pv : 1;
    i32 lddv = pv > 1 ? pv : 1;
    i32 ldaw = nw > 1 ? nw : 1;
    i32 ldbw = nw > 1 ? nw : 1;
    i32 ldcw = m > 1 ? m : 1;
    i32 lddw = m > 1 ? m : 1;
    i32 lds = n > 1 ? n : 1;
    i32 ldr = n > 1 ? n : 1;

    bool leftw = (weight == 'L' || weight == 'B');
    bool rightw = (weight == 'R' || weight == 'B');

    i32 nnv = n + nv;
    i32 nnw = n + nw;
    i32 lw = 1;
    if (leftw && pv > 0) {
        i32 max_nnv_pv = nnv > pv ? nnv : pv;
        i32 lwl = nnv * (nnv + max_nnv_pv + 5);
        lw = lw > lwl ? lw : lwl;
    } else {
        i32 lwl = n * (p + 5);
        lw = lw > lwl ? lw : lwl;
    }
    if (rightw && mw > 0) {
        i32 max_nnw_mw = nnw > mw ? nnw : mw;
        i32 lwr = nnw * (nnw + max_nnw_mw + 5);
        lw = lw > lwr ? lw : lwr;
    } else {
        i32 lwr = n * (m + 5);
        lw = lw > lwr ? lw : lwr;
    }
    i32 ldwork = lw * 2;

    f64 *s_data = (f64*)malloc((size_t)lds * n * sizeof(f64));
    f64 *r_data = (f64*)malloc((size_t)ldr * n * sizeof(f64));
    f64 *dwork = (f64*)malloc((size_t)ldwork * sizeof(f64));

    if (!s_data || !r_data || !dwork) {
        free(s_data);
        free(r_data);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(av_array);
        Py_DECREF(bv_array);
        Py_DECREF(cv_array);
        Py_DECREF(dv_array);
        Py_DECREF(aw_array);
        Py_DECREF(bw_array);
        Py_DECREF(cw_array);
        Py_DECREF(dw_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *av_data = (f64*)PyArray_DATA(av_array);
    f64 *bv_data = (f64*)PyArray_DATA(bv_array);
    f64 *cv_data = (f64*)PyArray_DATA(cv_array);
    f64 *dv_data = (f64*)PyArray_DATA(dv_array);
    f64 *aw_data = (f64*)PyArray_DATA(aw_array);
    f64 *bw_data = (f64*)PyArray_DATA(bw_array);
    f64 *cw_data = (f64*)PyArray_DATA(cw_array);
    f64 *dw_data = (f64*)PyArray_DATA(dw_array);

    f64 scalec, scaleo;
    i32 info;

    ab09iy(&dico, &jobc, &jobo, &weight,
           n, m, p, nv, pv, nw, mw,
           alphac, alphao,
           a_data, lda, b_data, ldb, c_data, ldc,
           av_data, ldav, bv_data, ldbv, cv_data, ldcv, dv_data, lddv,
           aw_data, ldaw, bw_data, ldbw, cw_data, ldcw, dw_data, lddw,
           &scalec, &scaleo,
           s_data, lds, r_data, ldr,
           dwork, ldwork, &info);

    free(dwork);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(av_array);
    Py_DECREF(bv_array);
    Py_DECREF(cv_array);
    Py_DECREF(dv_array);
    Py_DECREF(aw_array);
    Py_DECREF(bw_array);
    Py_DECREF(cw_array);
    Py_DECREF(dw_array);

    npy_intp dims_s[2] = {n, n};
    npy_intp strides_s_new[2] = {sizeof(f64), n * sizeof(f64)};
    PyObject *s_array = PyArray_New(&PyArray_Type, 2, dims_s, NPY_DOUBLE,
                                    strides_s_new, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (s_array == NULL) {
        free(s_data);
        free(r_data);
        return NULL;
    }
    for (i32 j = 0; j < n; j++) {
        memcpy((f64*)PyArray_DATA((PyArrayObject*)s_array) + j * n, s_data + j * lds, (size_t)n * sizeof(f64));
    }
    free(s_data);

    npy_intp dims_r[2] = {n, n};
    npy_intp strides_r_new[2] = {sizeof(f64), n * sizeof(f64)};
    PyObject *r_array = PyArray_New(&PyArray_Type, 2, dims_r, NPY_DOUBLE,
                                    strides_r_new, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (r_array == NULL) {
        free(r_data);
        Py_DECREF(s_array);
        return NULL;
    }
    for (i32 j = 0; j < n; j++) {
        memcpy((f64*)PyArray_DATA((PyArrayObject*)r_array) + j * n, r_data + j * ldr, (size_t)n * sizeof(f64));
    }
    free(r_data);

    return Py_BuildValue("OOddi", s_array, r_array, scalec, scaleo, info);
}

PyObject* py_ab13hd(PyObject* self, PyObject* args) {
    const char *dico, *jobe, *equil, *jobd, *ckprop, *reduce, *poles;
    i32 n, m, p, ranke;
    PyObject *fpeak_obj, *a_obj, *e_obj, *b_obj, *c_obj, *d_obj, *tol_obj;

    if (!PyArg_ParseTuple(args, "sssssssiiiiOOOOOOO",
                          &dico, &jobe, &equil, &jobd, &ckprop, &reduce, &poles,
                          &n, &m, &p, &ranke,
                          &fpeak_obj, &a_obj, &e_obj, &b_obj, &c_obj, &d_obj, &tol_obj)) {
        return NULL;
    }

    PyArrayObject *fpeak_array = (PyArrayObject*)PyArray_FROM_OTF(fpeak_obj, NPY_DOUBLE,
                                                                   NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (fpeak_array == NULL) return NULL;

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        Py_DECREF(fpeak_array);
        return NULL;
    }

    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (e_array == NULL) {
        Py_DECREF(fpeak_array);
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(fpeak_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (c_array == NULL) {
        Py_DECREF(fpeak_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (d_array == NULL) {
        Py_DECREF(fpeak_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    PyArrayObject *tol_array = (PyArrayObject*)PyArray_FROM_OTF(tol_obj, NPY_DOUBLE,
                                                                 NPY_ARRAY_FARRAY | NPY_ARRAY_IN_ARRAY);
    if (tol_array == NULL) {
        Py_DECREF(fpeak_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }

    f64 *fpeak_data = (f64*)PyArray_DATA(fpeak_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *tol_data = (f64*)PyArray_DATA(tol_array);

    i32 lda = (n > 0) ? (i32)PyArray_DIM(a_array, 0) : 1;
    i32 ldb = (n > 0) ? (i32)PyArray_DIM(b_array, 0) : 1;
    i32 ldc = (p > 0) ? (i32)PyArray_DIM(c_array, 0) : 1;
    i32 ldd = (p > 0) ? (i32)PyArray_DIM(d_array, 0) : 1;
    i32 lde;
    if (*jobe == 'I' || *jobe == 'i') {
        lde = 1;
    } else if (*jobe == 'G' || *jobe == 'g') {
        lde = (n > 0) ? (i32)PyArray_DIM(e_array, 0) : 1;
    } else {
        lde = (ranke > 0) ? (i32)PyArray_DIM(e_array, 0) : 1;
    }

    f64 gpeak[2] = {0.0, 1.0};
    i32 nr = n;
    i32 iwarn = 0;
    i32 info = 0;

    i32 minpm = (p < m) ? p : m;
    i32 maxpm = (p > m) ? p : m;
    i32 pm = p + m;

    i32 liwork;
    if (minpm == 0 || n == 0) {
        liwork = 1;
    } else if (!(*jobe == 'I' || *jobe == 'i') || (*dico == 'D' || *dico == 'd') ||
               (*jobd == 'D' || *jobd == 'd')) {
        i32 r = (pm % 2 == 1) ? 1 : 0;
        liwork = 2*n + m + p + r + 12;
    } else {
        liwork = n;
    }
    liwork = (liwork > 1) ? liwork : 1;

    i32 query_sz = (n > 2) ? n : 2;
    f64 *dwork_query = (f64*)calloc(query_sz, sizeof(f64));
    c128 *zwork_query = (c128*)calloc(2, sizeof(c128));
    if (dwork_query == NULL || zwork_query == NULL) {
        if (dwork_query) free(dwork_query);
        if (zwork_query) free(zwork_query);
        Py_DECREF(fpeak_array); Py_DECREF(a_array); Py_DECREF(e_array);
        Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(d_array);
        Py_DECREF(tol_array);
        PyErr_NoMemory();
        return NULL;
    }
    i32 info_query = 0;
    i32 nr_query = n;
    i32 iwarn_query = 0;
    ab13hd(dico, jobe, equil, jobd, ckprop, reduce, poles,
           n, m, p, ranke, fpeak_data, a_data, lda, e_data, lde,
           b_data, ldb, c_data, ldc, d_data, ldd, &nr_query, gpeak, tol_data,
           NULL, dwork_query, -1, zwork_query, -1, NULL, &iwarn_query, &info_query);

    i32 ldwork = (i32)dwork_query[0];
    if (ldwork < 1) ldwork = 1;
    i32 lzwork = (i32)creal(zwork_query[0]);
    if (lzwork < 1) lzwork = 1;
    free(dwork_query);
    free(zwork_query);

    i32 *iwork = (i32*)PyMem_Calloc(liwork, sizeof(i32));
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    c128 *zwork = (c128*)calloc(lzwork, sizeof(c128));
    bool *bwork = (bool*)PyMem_Calloc((n > 0 ? n : 1), sizeof(bool));

    if (iwork == NULL || dwork == NULL || zwork == NULL || bwork == NULL) {
        if (iwork) PyMem_Free(iwork);
        if (dwork) free(dwork);
        if (zwork) free(zwork);
        if (bwork) PyMem_Free(bwork);
        Py_DECREF(fpeak_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(tol_array);
        PyErr_NoMemory();
        return NULL;
    }

    nr = n;
    ab13hd(dico, jobe, equil, jobd, ckprop, reduce, poles,
           n, m, p, ranke, fpeak_data, a_data, lda, e_data, lde,
           b_data, ldb, c_data, ldc, d_data, ldd, &nr, gpeak, tol_data,
           iwork, dwork, ldwork, zwork, lzwork, bwork, &iwarn, &info);

    PyMem_Free(iwork);
    free(dwork);
    free(zwork);
    PyMem_Free(bwork);

    PyArray_ResolveWritebackIfCopy(fpeak_array);
    Py_DECREF(fpeak_array);
    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(tol_array);

    npy_intp gpeak_dims[1] = {2};
    PyObject *gpeak_array = PyArray_SimpleNew(1, gpeak_dims, NPY_DOUBLE);
    if (gpeak_array == NULL) return NULL;
    memcpy(PyArray_DATA((PyArrayObject*)gpeak_array), gpeak, 2 * sizeof(f64));

    npy_intp fpeak_out_dims[1] = {2};
    PyObject *fpeak_out_array = PyArray_SimpleNew(1, fpeak_out_dims, NPY_DOUBLE);
    if (fpeak_out_array == NULL) {
        Py_DECREF(gpeak_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)fpeak_out_array), fpeak_data, 2 * sizeof(f64));

    return Py_BuildValue("OOiii", gpeak_array, fpeak_out_array, nr, iwarn, info);
}


/* Python wrapper for ab09id - Frequency-weighted model reduction (B&T/SPA) */
PyObject* py_ab09id(PyObject* self, PyObject* args) {
    const char *dico_str, *jobc_str, *jobo_str, *job_str, *weight_str, *equil_str, *ordsel_str;
    int n, m, p, nv, pv, nw, mw, nr_in;
    double alpha, alphac, alphao, tol1, tol2;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    PyObject *av_obj, *bv_obj, *cv_obj, *dv_obj;
    PyObject *aw_obj, *bw_obj, *cw_obj, *dw_obj;

    if (!PyArg_ParseTuple(args, "sssssssiiiiiiiiddd" "OOOOOOOOOOOO" "dd",
                          &dico_str, &jobc_str, &jobo_str, &job_str, &weight_str, &equil_str, &ordsel_str,
                          &n, &m, &p, &nv, &pv, &nw, &mw, &nr_in,
                          &alpha, &alphac, &alphao,
                          &a_obj, &b_obj, &c_obj, &d_obj,
                          &av_obj, &bv_obj, &cv_obj, &dv_obj,
                          &aw_obj, &bw_obj, &cw_obj, &dw_obj,
                          &tol1, &tol2)) {
        return NULL;
    }

    char dico_c = (char)toupper((unsigned char)dico_str[0]);
    char jobc_c = (char)toupper((unsigned char)jobc_str[0]);
    char jobo_c = (char)toupper((unsigned char)jobo_str[0]);
    char job_c = (char)toupper((unsigned char)job_str[0]);
    char weight_c = (char)toupper((unsigned char)weight_str[0]);
    char equil_c = (char)toupper((unsigned char)equil_str[0]);
    char ordsel_c = (char)toupper((unsigned char)ordsel_str[0]);

    if (dico_c != 'C' && dico_c != 'D') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }
    if (jobc_c != 'S' && jobc_c != 'E') {
        PyErr_SetString(PyExc_ValueError, "JOBC must be 'S' or 'E'");
        return NULL;
    }
    if (jobo_c != 'S' && jobo_c != 'E') {
        PyErr_SetString(PyExc_ValueError, "JOBO must be 'S' or 'E'");
        return NULL;
    }
    if (job_c != 'B' && job_c != 'F' && job_c != 'S' && job_c != 'P') {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'B', 'F', 'S', or 'P'");
        return NULL;
    }
    if (weight_c != 'N' && weight_c != 'L' && weight_c != 'R' && weight_c != 'B') {
        PyErr_SetString(PyExc_ValueError, "WEIGHT must be 'N', 'L', 'R', or 'B'");
        return NULL;
    }
    if (equil_c != 'S' && equil_c != 'N') {
        PyErr_SetString(PyExc_ValueError, "EQUIL must be 'S' or 'N'");
        return NULL;
    }
    if (ordsel_c != 'F' && ordsel_c != 'A') {
        PyErr_SetString(PyExc_ValueError, "ORDSEL must be 'F' or 'A'");
        return NULL;
    }
    if (n < 0 || m < 0 || p < 0 || nv < 0 || pv < 0 || nw < 0 || mw < 0) {
        PyErr_SetString(PyExc_ValueError, "N, M, P, NV, PV, NW, MW must be >= 0");
        return NULL;
    }

    bool leftw = (weight_c == 'L' || weight_c == 'B');
    bool rightw = (weight_c == 'R' || weight_c == 'B');

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(
        d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *av_array = (PyArrayObject*)PyArray_FROM_OTF(
        av_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *bv_array = (PyArrayObject*)PyArray_FROM_OTF(
        bv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *cv_array = (PyArrayObject*)PyArray_FROM_OTF(
        cv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *dv_array = (PyArrayObject*)PyArray_FROM_OTF(
        dv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *aw_array = (PyArrayObject*)PyArray_FROM_OTF(
        aw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *bw_array = (PyArrayObject*)PyArray_FROM_OTF(
        bw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *cw_array = (PyArrayObject*)PyArray_FROM_OTF(
        cw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject *dw_array = (PyArrayObject*)PyArray_FROM_OTF(
        dw_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!a_array || !b_array || !c_array || !d_array ||
        !av_array || !bv_array || !cv_array || !dv_array ||
        !aw_array || !bw_array || !cw_array || !dw_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        Py_XDECREF(av_array);
        Py_XDECREF(bv_array);
        Py_XDECREF(cv_array);
        Py_XDECREF(dv_array);
        Py_XDECREF(aw_array);
        Py_XDECREF(bw_array);
        Py_XDECREF(cw_array);
        Py_XDECREF(dw_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldc = p > 0 ? p : 1;
    i32 ldd = p > 0 ? p : 1;
    i32 ldav = leftw ? (nv > 0 ? nv : 1) : 1;
    i32 ldbv = leftw ? (nv > 0 ? nv : 1) : 1;
    i32 ldcv = leftw ? (pv > 0 ? pv : 1) : 1;
    i32 lddv = leftw ? (pv > 0 ? pv : 1) : 1;
    i32 ldaw = rightw ? (nw > 0 ? nw : 1) : 1;
    i32 ldbw = rightw ? (nw > 0 ? nw : 1) : 1;
    i32 ldcw = rightw ? (m > 0 ? m : 1) : 1;
    i32 lddw = rightw ? (m > 0 ? m : 1) : 1;

    i32 nn = n * n;
    i32 nnv = n + nv;
    i32 nnw = n + nw;
    i32 ppv = (p > pv) ? p : pv;

    i32 lw = 1;
    if (leftw && pv > 0) {
        i32 term = nnv * (nnv + ((nnv > pv) ? nnv : pv) + 5);
        lw = (lw > term) ? lw : term;
    } else {
        i32 term = n * (p + 5);
        lw = (lw > term) ? lw : term;
    }
    if (rightw && mw > 0) {
        i32 term = nnw * (nnw + ((nnw > mw) ? nnw : mw) + 5);
        lw = (lw > term) ? lw : term;
    } else {
        i32 term = n * (m + 5);
        lw = (lw > term) ? lw : term;
    }
    lw = 2 * nn + ((lw > (2 * nn + 5 * n)) ? lw : (2 * nn + 5 * n));
    i32 nmp = n * ((m > p) ? m : p);
    lw = (lw > nmp) ? lw : nmp;

    if (leftw && nv > 0) {
        i32 nv5 = nv * (nv + 5);
        i32 pv2 = pv * (pv + 2);
        i32 p4 = 4 * ppv;
        i32 maxterm = (nv5 > pv2) ? nv5 : pv2;
        maxterm = (maxterm > p4) ? maxterm : p4;
        i32 lcf = pv * (nv + pv) + pv * nv + maxterm;
        if (pv == p) {
            i32 term1 = nv + ((nv > 3 * p) ? nv : 3 * p);
            i32 term = (lcf > term1) ? lcf : term1;
            lw = (lw > term) ? lw : term;
        } else {
            i32 ppv3 = (nv > 3 * ppv) ? nv : 3 * ppv;
            i32 term = ppv * (2 * nv + ppv) + ((lcf > (nv + ppv3)) ? lcf : (nv + ppv3));
            lw = (lw > term) ? lw : term;
        }
    }

    i32 m_mw = (m > mw) ? m : mw;
    if (rightw && nw > 0) {
        if (mw == m) {
            i32 term = nw + ((nw > 3 * m) ? nw : 3 * m);
            lw = (lw > term) ? lw : term;
        } else {
            i32 mw3 = (nw > 3 * m_mw) ? nw : 3 * m_mw;
            i32 term = 2 * nw * m_mw + nw + mw3;
            lw = (lw > term) ? lw : term;
        }
        i32 nw5 = nw * (nw + 5);
        i32 mw2 = mw * (mw + 2);
        i32 mw4 = 4 * mw;
        i32 m4 = 4 * m;
        i32 maxterm = (nw5 > mw2) ? nw5 : mw2;
        maxterm = (maxterm > mw4) ? maxterm : mw4;
        maxterm = (maxterm > m4) ? maxterm : m4;
        i32 term = mw * (nw + mw) + maxterm;
        lw = (lw > term) ? lw : term;
    }

    i32 ldwork = lw + 3 * n + 100;

    i32 liwrk1 = 0;
    if (job_c == 'F') liwrk1 = n;
    else if (job_c == 'S' || job_c == 'P') liwrk1 = 2 * n;

    i32 liwrk2 = 0;
    if (leftw && nv > 0) liwrk2 = nv + ppv;

    i32 liwrk3 = 0;
    if (rightw && nw > 0) liwrk3 = nw + m_mw;

    i32 liwork = 3;
    if (liwrk1 > liwork) liwork = liwrk1;
    if (liwrk2 > liwork) liwork = liwrk2;
    if (liwrk3 > liwork) liwork = liwrk3;

    f64 *dwork = (f64*)calloc(ldwork + 1, sizeof(f64));
    i32 *iwork = (i32*)calloc(liwork + 1, sizeof(i32));
    f64 *hsv = (f64*)calloc(n > 0 ? n : 1, sizeof(f64));

    if (!dwork || !iwork || !hsv) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(av_array);
        Py_DECREF(bv_array);
        Py_DECREF(cv_array);
        Py_DECREF(dv_array);
        Py_DECREF(aw_array);
        Py_DECREF(bw_array);
        Py_DECREF(cw_array);
        Py_DECREF(dw_array);
        free(dwork);
        free(iwork);
        free(hsv);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *av_data = (f64*)PyArray_DATA(av_array);
    f64 *bv_data = (f64*)PyArray_DATA(bv_array);
    f64 *cv_data = (f64*)PyArray_DATA(cv_array);
    f64 *dv_data = (f64*)PyArray_DATA(dv_array);
    f64 *aw_data = (f64*)PyArray_DATA(aw_array);
    f64 *bw_data = (f64*)PyArray_DATA(bw_array);
    f64 *cw_data = (f64*)PyArray_DATA(cw_array);
    f64 *dw_data = (f64*)PyArray_DATA(dw_array);

    i32 nr = nr_in;
    i32 ns = 0;
    i32 iwarn = 0, info = 0;

    ab09id(dico_str, jobc_str, jobo_str, job_str, weight_str, equil_str, ordsel_str,
           n, m, p, nv, pv, nw, mw, &nr, alpha, alphac, alphao,
           a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           av_data, ldav, bv_data, ldbv, cv_data, ldcv, dv_data, lddv,
           aw_data, ldaw, bw_data, ldbw, cw_data, ldcw, dw_data, lddw,
           &ns, hsv, tol1, tol2, &iwarn, &info);

    free(dwork);
    free(iwork);

    if (info < 0) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(av_array);
        Py_DECREF(bv_array);
        Py_DECREF(cv_array);
        Py_DECREF(dv_array);
        Py_DECREF(aw_array);
        Py_DECREF(bw_array);
        Py_DECREF(cw_array);
        Py_DECREF(dw_array);
        free(hsv);
        PyErr_Format(PyExc_RuntimeError, "AB09ID: Parameter %d had an illegal value", -info);
        return NULL;
    }

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(av_array);
    PyArray_ResolveWritebackIfCopy(bv_array);
    PyArray_ResolveWritebackIfCopy(cv_array);
    PyArray_ResolveWritebackIfCopy(dv_array);
    PyArray_ResolveWritebackIfCopy(aw_array);
    PyArray_ResolveWritebackIfCopy(bw_array);
    PyArray_ResolveWritebackIfCopy(cw_array);
    PyArray_ResolveWritebackIfCopy(dw_array);

    npy_intp hsv_dims[1] = {n > 0 ? n : 1};
    PyObject *hsv_array = PyArray_SimpleNew(1, hsv_dims, NPY_DOUBLE);
    if (hsv_array) {
        memcpy(PyArray_DATA((PyArrayObject*)hsv_array), hsv, (size_t)(n > 0 ? n : 1) * sizeof(f64));
    }
    free(hsv);

    PyObject *result = Py_BuildValue("OOOOOOOOOOiOiii",
        a_array, b_array, c_array, d_array,
        av_array, bv_array, cv_array,
        aw_array, bw_array, cw_array,
        ns, hsv_array, nr, iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(av_array);
    Py_DECREF(bv_array);
    Py_DECREF(cv_array);
    Py_DECREF(dv_array);
    Py_DECREF(aw_array);
    Py_DECREF(bw_array);
    Py_DECREF(cw_array);
    Py_DECREF(dw_array);
    Py_XDECREF(hsv_array);

    return result;
}

/* Python wrapper for ab09fd */
PyObject* py_ab09fd(PyObject* self, PyObject* args) {
    const char *dico_str, *jobcf_str, *fact_str, *jobmr_str, *equil_str, *ordsel_str;
    int n, m, p, nr_in;
    double alpha, tol1, tol2;
    PyObject *a_obj, *b_obj, *c_obj;

    if (!PyArg_ParseTuple(args, "ssssssiiiidOOOdd", &dico_str, &jobcf_str, &fact_str,
                          &jobmr_str, &equil_str, &ordsel_str, &n, &m, &p, &nr_in,
                          &alpha, &a_obj, &b_obj, &c_obj, &tol1, &tol2)) {
        return NULL;
    }

    if (dico_str[0] != 'C' && dico_str[0] != 'c' &&
        dico_str[0] != 'D' && dico_str[0] != 'd') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }

    if (jobcf_str[0] != 'L' && jobcf_str[0] != 'l' &&
        jobcf_str[0] != 'R' && jobcf_str[0] != 'r') {
        PyErr_SetString(PyExc_ValueError, "JOBCF must be 'L' or 'R'");
        return NULL;
    }

    if (fact_str[0] != 'S' && fact_str[0] != 's' &&
        fact_str[0] != 'I' && fact_str[0] != 'i') {
        PyErr_SetString(PyExc_ValueError, "FACT must be 'S' or 'I'");
        return NULL;
    }

    if (jobmr_str[0] != 'B' && jobmr_str[0] != 'b' &&
        jobmr_str[0] != 'N' && jobmr_str[0] != 'n') {
        PyErr_SetString(PyExc_ValueError, "JOBMR must be 'B' or 'N'");
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
    if (fixord && (nr_in < 0 || nr_in > n)) {
        PyErr_SetString(PyExc_ValueError, "NR must be >= 0 and <= N for fixed order");
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

    i32 maxmp = (m > p) ? m : p;
    bool left = (jobcf_str[0] == 'L' || jobcf_str[0] == 'l');
    bool stabd = (fact_str[0] == 'S' || fact_str[0] == 's');
    bool bfree = (jobmr_str[0] == 'N' || jobmr_str[0] == 'n');

    i32 lwr = 2*n*n + n*((n > (m+p)) ? n : (m+p)) + 5*n + (n*(n+1))/2;
    i32 lw1, lw2, lw3, lw4;

    lw1 = n*(2*maxmp + p) + maxmp*(maxmp + p);
    i32 lw1_add = n*p + ((n*(n+5) > 5*p) ? n*(n+5) : 5*p);
    lw1_add = (lw1_add > 4*m) ? lw1_add : 4*m;
    lw1 = lw1 + ((lw1_add > lwr) ? lw1_add : lwr);

    i32 lw2_add = n*p + ((n*(n+5) > p*(p+2)) ? n*(n+5) : p*(p+2));
    lw2_add = (lw2_add > 4*p) ? lw2_add : 4*p;
    lw2_add = (lw2_add > 4*m) ? lw2_add : 4*m;
    lw2 = n*(2*maxmp + p) + maxmp*(maxmp + p) + ((lw2_add > lwr) ? lw2_add : lwr);

    i32 lw3_add = (5*m > 4*p) ? 5*m : 4*p;
    lw3 = (n+m)*(m+p) + ((lw3_add > lwr) ? lw3_add : lwr);

    i32 lw4_add = (m*(m+2) > 4*m) ? m*(m+2) : 4*m;
    lw4_add = (lw4_add > 4*p) ? lw4_add : 4*p;
    lw4 = (n+m)*(m+p) + ((lw4_add > lwr) ? lw4_add : lwr);

    i32 ldwork;
    if (left && stabd) {
        ldwork = lw1 > 1 ? lw1 : 1;
    } else if (left && !stabd) {
        ldwork = lw2 > 1 ? lw2 : 1;
    } else if (!left && stabd) {
        ldwork = lw3 > 1 ? lw3 : 1;
    } else {
        ldwork = lw4 > 1 ? lw4 : 1;
    }
    ldwork = (i32)(ldwork * 1.5);
    if (ldwork < 1) ldwork = 1;

    i32 pm = left ? p : m;
    i32 iwork_size;
    if (bfree) {
        iwork_size = (n > pm) ? n : pm;
    } else {
        iwork_size = pm;
    }
    if (iwork_size < 1) iwork_size = 1;

    f64 *hsv = (f64*)malloc((size_t)(n > 0 ? n : 1) * sizeof(f64));
    i32 *iwork = (i32*)malloc((size_t)iwork_size * sizeof(i32));
    f64 *dwork = (f64*)malloc((size_t)ldwork * sizeof(f64));

    if (!hsv || !iwork || !dwork) {
        free(hsv);
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    i32 nr = nr_in;
    i32 nq, iwarn, info;

    ab09fd(dico_str, jobcf_str, fact_str, jobmr_str, equil_str, ordsel_str,
           n, m, p, &nr, alpha, a_data, lda, b_data, ldb, c_data, ldc,
           &nq, hsv, tol1, tol2, iwork, dwork, ldwork, &iwarn, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    npy_intp hsv_dims[1] = {n > 0 ? n : 1};
    PyObject *hsv_array = PyArray_SimpleNew(1, hsv_dims, NPY_DOUBLE);
    if (!hsv_array) {
        free(hsv);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)hsv_array), hsv, (size_t)(n > 0 ? n : 1) * sizeof(f64));
    free(hsv);

    PyObject *result = Py_BuildValue("OOOiiOii",
        a_array, b_array, c_array, nr, nq, hsv_array, iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(hsv_array);

    return result;
}

/* Python wrapper for ab09gd */
PyObject* py_ab09gd(PyObject* self, PyObject* args) {
    const char *dico_str, *jobcf_str, *fact_str, *jobmr_str, *equil_str, *ordsel_str;
    int n, m, p, nr_in;
    double alpha, tol1, tol2, tol3;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "ssssssiiiidOOOOddd", &dico_str, &jobcf_str, &fact_str,
                          &jobmr_str, &equil_str, &ordsel_str, &n, &m, &p, &nr_in,
                          &alpha, &a_obj, &b_obj, &c_obj, &d_obj, &tol1, &tol2, &tol3)) {
        return NULL;
    }

    if (dico_str[0] != 'C' && dico_str[0] != 'c' &&
        dico_str[0] != 'D' && dico_str[0] != 'd') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }

    if (jobcf_str[0] != 'L' && jobcf_str[0] != 'l' &&
        jobcf_str[0] != 'R' && jobcf_str[0] != 'r') {
        PyErr_SetString(PyExc_ValueError, "JOBCF must be 'L' or 'R'");
        return NULL;
    }

    if (fact_str[0] != 'S' && fact_str[0] != 's' &&
        fact_str[0] != 'I' && fact_str[0] != 'i') {
        PyErr_SetString(PyExc_ValueError, "FACT must be 'S' or 'I'");
        return NULL;
    }

    if (jobmr_str[0] != 'B' && jobmr_str[0] != 'b' &&
        jobmr_str[0] != 'N' && jobmr_str[0] != 'n') {
        PyErr_SetString(PyExc_ValueError, "JOBMR must be 'B' or 'N'");
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
    if (fixord && (nr_in < 0 || nr_in > n)) {
        PyErr_SetString(PyExc_ValueError, "NR must be >= 0 and <= N for fixed order");
        return NULL;
    }

    bool stabd = (fact_str[0] == 'S' || fact_str[0] == 's');
    bool discr = (dico_str[0] == 'D' || dico_str[0] == 'd');
    if (stabd && ((!discr && alpha >= 0.0) ||
                  (discr && (alpha < 0.0 || alpha >= 1.0)))) {
        PyErr_SetString(PyExc_ValueError,
            "ALPHA must be < 0 for continuous or 0 <= ALPHA < 1 for discrete when FACT='S'");
        return NULL;
    }

    if (tol2 > 0.0 && tol2 > tol1) {
        PyErr_SetString(PyExc_ValueError, "TOL2 must be <= TOL1 when both > 0");
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

    i32 maxmp = (m > p) ? m : p;
    bool left = (jobcf_str[0] == 'L' || jobcf_str[0] == 'l');

    i32 lwr = 2*n*n + n*((n > (m+p)) ? n : (m+p)) + 5*n + (n*(n+1))/2;
    i32 lw1, lw2, lw3, lw4;

    lw1 = n*(2*maxmp + p) + maxmp*(maxmp + p);
    i32 lw1_add = n*p + ((n*(n+5) > 5*p) ? n*(n+5) : 5*p);
    lw1_add = (lw1_add > 4*m) ? lw1_add : 4*m;
    lw1 = lw1 + ((lw1_add > lwr) ? lw1_add : lwr);

    i32 lw2_add = n*p + ((n*(n+5) > p*(p+2)) ? n*(n+5) : p*(p+2));
    lw2_add = (lw2_add > 4*p) ? lw2_add : 4*p;
    lw2_add = (lw2_add > 4*m) ? lw2_add : 4*m;
    lw2 = n*(2*maxmp + p) + maxmp*(maxmp + p) + ((lw2_add > lwr) ? lw2_add : lwr);

    i32 lw3_add = (5*m > 4*p) ? 5*m : 4*p;
    lw3 = (n+m)*(m+p) + ((lw3_add > lwr) ? lw3_add : lwr);

    i32 lw4_add = (m*(m+2) > 4*m) ? m*(m+2) : 4*m;
    lw4_add = (lw4_add > 4*p) ? lw4_add : 4*p;
    lw4 = (n+m)*(m+p) + ((lw4_add > lwr) ? lw4_add : lwr);

    i32 ldwork;
    if (left && stabd) {
        ldwork = lw1 > 1 ? lw1 : 1;
    } else if (left && !stabd) {
        ldwork = lw2 > 1 ? lw2 : 1;
    } else if (!left && stabd) {
        ldwork = lw3 > 1 ? lw3 : 1;
    } else {
        ldwork = lw4 > 1 ? lw4 : 1;
    }
    ldwork = (i32)(ldwork * 1.5);
    if (ldwork < 1) ldwork = 1;

    i32 pm = left ? p : m;
    i32 iwork_size = (2*n > pm) ? 2*n : pm;
    if (iwork_size < 1) iwork_size = 1;

    f64 *hsv = (f64*)malloc((size_t)(n > 0 ? n : 1) * sizeof(f64));
    i32 *iwork = (i32*)malloc((size_t)iwork_size * sizeof(i32));
    f64 *dwork = (f64*)malloc((size_t)ldwork * sizeof(f64));

    if (!hsv || !iwork || !dwork) {
        free(hsv);
        free(iwork);
        free(dwork);
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

    i32 nr = nr_in;
    i32 nq, iwarn, info;

    ab09gd(dico_str, jobcf_str, fact_str, jobmr_str, equil_str, ordsel_str,
           n, m, p, &nr, alpha, a_data, lda, b_data, ldb, c_data, ldc,
           d_data, ldd, &nq, hsv, tol1, tol2, tol3, iwork, dwork, ldwork, &iwarn, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);
    PyArray_ResolveWritebackIfCopy(d_array);

    npy_intp hsv_dims[1] = {n > 0 ? n : 1};
    PyObject *hsv_array = PyArray_SimpleNew(1, hsv_dims, NPY_DOUBLE);
    if (!hsv_array) {
        free(hsv);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)hsv_array), hsv, (size_t)(n > 0 ? n : 1) * sizeof(f64));
    free(hsv);

    PyObject *result = Py_BuildValue("OOOOiiOii",
        a_array, b_array, c_array, d_array, nr, nq, hsv_array, iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(hsv_array);

    return result;
}

/* Python wrapper for ab13id */
PyObject* py_ab13id(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *jobsys_str, *jobeig_str, *equil_str, *cksing_str, *restor_str, *update_str;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj, *tol_obj;
    i32 n_in = -1, m_in = -1, p_in = -1;

    static char *kwlist[] = {
        "jobsys", "jobeig", "equil", "cksing", "restor", "update",
        "a", "e", "b", "c", "tol", "n", "m", "p", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssssssOOOOO|iii", kwlist,
                                     &jobsys_str, &jobeig_str, &equil_str,
                                     &cksing_str, &restor_str, &update_str,
                                     &a_obj, &e_obj, &b_obj, &c_obj, &tol_obj,
                                     &n_in, &m_in, &p_in)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;

    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(
        e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!e_array) {
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!b_array) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!c_array) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *tol_array = (PyArrayObject*)PyArray_FROM_OTF(
        tol_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (!tol_array) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);

    i32 n = (n_in >= 0) ? n_in : (i32)a_dims[0];
    i32 m = (m_in >= 0) ? m_in : (i32)b_dims[1];
    i32 p = (p_in >= 0) ? p_in : (i32)c_dims[0];

    i32 lda = (i32)a_dims[0];
    i32 lde = (i32)PyArray_DIMS(e_array)[0];
    i32 ldb = (i32)b_dims[0];
    i32 ldc = (i32)c_dims[0];

    i32 maxmp = (m > p) ? m : p;
    bool lredc = (jobsys_str[0] == 'R' || jobsys_str[0] == 'r');
    bool lrema = (jobeig_str[0] == 'A' || jobeig_str[0] == 'a');
    bool lequil = (equil_str[0] == 'S' || equil_str[0] == 's');
    bool lsing = (cksing_str[0] == 'C' || cksing_str[0] == 'c');
    bool maxacc = (restor_str[0] == 'R' || restor_str[0] == 'r');
    bool lupd = (update_str[0] == 'U' || update_str[0] == 'u');
    bool lrupd = lrema || lupd;

    i32 liwork;
    if (lredc) {
        liwork = 2 * n + maxmp + 7;
    } else {
        liwork = n;
    }
    if (liwork < 1) liwork = 1;

    i32 ldwork;
    if (lredc) {
        i32 k = n * (2 * n + m + p);
        i32 minwrk;
        if (maxacc) {
            minwrk = 2 * (k + maxmp + n - 1);
        } else {
            minwrk = 2 * (maxmp + n - 1);
        }
        minwrk = (minwrk > n * n + 4 * n) ? minwrk : n * n + 4 * n;
        if (lsing) {
            i32 y = 2 * n * n + 10 * n + ((n > 23) ? n : 23);
            minwrk = (minwrk > y) ? minwrk : y;
        }
        ldwork = minwrk;
    } else {
        ldwork = 0;
    }

    if (lrupd) {
        i32 w1 = n * n + 4 * n + 4;
        i32 w2 = n + maxmp;
        ldwork = (ldwork > w1) ? ldwork : w1;
        ldwork = (ldwork > w2) ? ldwork : w2;
    } else {
        i32 w = 4 * n + 4;
        ldwork = (ldwork > w) ? ldwork : w;
    }
    if (lequil) {
        i32 w = 8 * n;
        ldwork = (ldwork > w) ? ldwork : w;
    }
    if (ldwork < 1) ldwork = 1;
    ldwork = (i32)(ldwork * 1.5);

    i32 *iwork = (i32*)malloc((size_t)liwork * sizeof(i32));
    f64 *dwork = (f64*)malloc((size_t)ldwork * sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(tol_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    const f64 *tol_data = (const f64*)PyArray_DATA(tol_array);

    i32 nr, ranke, iwarn, info;

    bool is_proper = ab13id(jobsys_str, jobeig_str, equil_str, cksing_str,
                            restor_str, update_str,
                            n, m, p, a_data, lda, e_data, lde,
                            b_data, ldb, c_data, ldc,
                            &nr, &ranke, tol_data, iwork, dwork, ldwork, &iwarn, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(c_array);

    PyObject *result = Py_BuildValue("OiiOOOOii",
        is_proper ? Py_True : Py_False, nr, ranke,
        a_array, e_array, b_array, c_array, iwarn, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(tol_array);

    return result;
}
