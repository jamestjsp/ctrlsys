/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void tb01vd(const char* apply, i32 n, i32 m, i32 l, f64* a, i32 lda,
            f64* b, i32 ldb, f64* c, i32 ldc, const f64* d, i32 ldd,
            f64* x0, f64* theta, i32 ltheta, f64* scale,
            f64* dwork, i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 two = 2.0;
    const f64 half = 0.5;
    const i32 int1 = 1;
    i32 i, j, k, in, ca, ia, it, iu, iwr, iwi, jwork, itau, ir, iq;
    i32 ldt, ldca, wrkopt;
    f64 piby2, ri, ti;
    bool lapply;

    lapply = (apply[0] == 'A' || apply[0] == 'a');

    *info = 0;

    if (!lapply && !(apply[0] == 'N' || apply[0] == 'n')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (l < 0) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (ldc < (l > 1 ? l : 1)) {
        *info = -10;
    } else if (ldd < (l > 1 ? l : 1)) {
        *info = -12;
    } else if (ltheta < n * (m + l + 1) + l * m) {
        *info = -15;
    } else {
        i32 ldwork_min1 = n * n * l + n * l + n;
        i32 max_nl = n > l ? n : l;
        i32 inner1 = n * n + n * max_nl + 6 * n + (n < l ? n : l);
        i32 inner2 = n * m;
        i32 inner_max = inner1 > inner2 ? inner1 : inner2;
        i32 ldwork_min2 = n * n + inner_max;
        i32 ldwork_min = ldwork_min1 > ldwork_min2 ? ldwork_min1 : ldwork_min2;
        if (ldwork_min < 1) ldwork_min = 1;
        if (ldwork < ldwork_min) {
            *info = -17;
        }
    }

    if (*info != 0) {
        return;
    }

    i32 max_nml = n;
    if (m > max_nml) max_nml = m;
    if (l > max_nml) max_nml = l;
    if (max_nml == 0) {
        dwork[0] = one;
        return;
    }

    if (n == 0) {
        SLC_DLACPY("F", &l, &m, d, &ldd, theta, &l);
        dwork[0] = one;
        return;
    }

    if (l == 0) {
        SLC_DLACPY("F", &n, &m, b, &ldb, theta, &n);
        SLC_DCOPY(&n, x0, &int1, &theta[n*m], &int1);
        dwork[0] = one;
        return;
    }

    wrkopt = 1;
    piby2 = two * atan(one);

    ldt = n > l ? n : l;
    ca = 0;
    ia = 0;
    it = ia + n * n;
    iu = it + ldt * n;
    iwr = iu + n * n;
    iwi = iwr + n;
    jwork = iwi + n;

    SLC_DLACPY("F", &n, &n, a, &lda, &dwork[ia], &n);
    SLC_DLACPY("F", &l, &n, c, &ldc, &dwork[it], &ldt);

    i32 ldwork_sb03 = ldwork - jwork;
    i32 info_sb03;
    sb03od("D", "N", "N", n, l, &dwork[ia], n, &dwork[iu], n, &dwork[it], ldt,
           scale, &dwork[iwr], &dwork[iwi], &dwork[jwork], ldwork_sb03, &info_sb03);

    if (info_sb03 != 0) {
        if (info_sb03 == 6) {
            *info = 3;
        } else {
            *info = 2;
        }
        return;
    }
    i32 opt_sb03 = (i32)dwork[jwork] + jwork;
    wrkopt = wrkopt > opt_sb03 ? wrkopt : opt_sb03;

    if (*scale == zero) {
        *info = 1;
        return;
    }

    SLC_DTRMM("L", "U", "N", "N", &n, &n, &one, &dwork[it], &ldt, a, &lda);
    SLC_DTRSM("R", "U", "N", "N", &n, &n, &one, &dwork[it], &ldt, a, &lda);

    if (m > 0) {
        f64 one_over_scale = one / (*scale);
        SLC_DTRMM("L", "U", "N", "N", &n, &m, &one_over_scale, &dwork[it], &ldt, b, &ldb);
    }

    SLC_DTRMV("U", "N", "N", &n, &dwork[it], &ldt, x0, &int1);
    f64 one_over_scale = one / (*scale);
    SLC_DSCAL(&n, &one_over_scale, x0, &int1);

    SLC_DTRSM("R", "U", "N", "N", &l, &n, scale, &dwork[it], &ldt, c, &ldc);

    ma02ad("F", l, n, c, ldc, &dwork[ca], n);

    for (i = 0; i < n - 1; i++) {
        i32 rows = n;
        i32 cols = l;
        i32 ld_src = n;
        i32 ld_dst = n;
        SLC_DGEMM("T", "N", &rows, &cols, &n, &one, a, &lda,
                  &dwork[ca + i * n * l], &ld_src, &zero, &dwork[ca + (i + 1) * n * l], &ld_dst);
    }

    itau = ca + n * n * l;
    jwork = itau + n;
    i32 cols_qr = l * n;
    SLC_DGEQRF(&n, &cols_qr, &dwork[ca], &n, &dwork[itau], &dwork[jwork], &ldwork_sb03, &info_sb03);
    i32 opt_qr = (i32)dwork[jwork] + jwork;
    wrkopt = wrkopt > opt_qr ? wrkopt : opt_qr;

    ir = n * n;
    if (l != 2) {
        SLC_DCOPY(&n, &dwork[itau], &int1, &dwork[ir + n * n], &int1);
    }
    SLC_DLACPY("L", &n, &n, &dwork[ca], &n, &dwork[ir], &n);
    itau = ir + n * n;
    jwork = itau + n;

    iq = 0;
    SLC_DLASET("F", &n, &n, &zero, &one, &dwork[iq], &n);

    for (i = 0; i < n; i++) {
        if (dwork[ir + i * (n + 1)] < zero) {
            dwork[iq + i * (n + 1)] = -one;
        }
    }

    ldwork_sb03 = ldwork - jwork;
    SLC_DORMQR("L", "N", &n, &n, &n, &dwork[ir], &n, &dwork[itau], &dwork[iq], &n,
               &dwork[jwork], &ldwork_sb03, &info_sb03);
    i32 opt_ormqr = (i32)dwork[jwork] + jwork;
    wrkopt = wrkopt > opt_ormqr ? wrkopt : opt_ormqr;
    jwork = ir;

    SLC_DGEMM("T", "N", &n, &n, &n, &one, &dwork[iq], &n, a, &lda, &zero, &dwork[jwork], &n);
    SLC_DGEMM("N", "N", &n, &n, &n, &one, &dwork[jwork], &n, &dwork[iq], &n, &zero, a, &lda);

    if (m > 0) {
        SLC_DLACPY("F", &n, &m, b, &ldb, &dwork[jwork], &n);
        SLC_DGEMM("T", "N", &n, &m, &n, &one, &dwork[iq], &n, &dwork[jwork], &n, &zero, b, &ldb);
    }

    SLC_DLACPY("F", &l, &n, c, &ldc, &dwork[jwork], &l);
    SLC_DGEMM("N", "N", &l, &n, &n, &one, &dwork[jwork], &l, &dwork[iq], &n, &zero, c, &ldc);

    SLC_DCOPY(&n, x0, &int1, &dwork[jwork], &int1);
    SLC_DGEMV("T", &n, &n, &one, &dwork[iq], &n, &dwork[jwork], &int1, &zero, x0, &int1);

    ldca = n + l;
    ca = 0;

    for (i = 0; i < n; i++) {
        SLC_DCOPY(&l, &c[i * ldc], &int1, &dwork[ca + i * ldca], &int1);
        SLC_DCOPY(&n, &a[i * lda], &int1, &dwork[ca + l + i * ldca], &int1);
    }

    jwork = ca + ldca * n;

    for (i = 0; i < n; i++) {
        SLC_DCOPY(&l, &dwork[ca + 1 + (n - 1 - i) * (ldca + 1)], &int1, &theta[i * l], &int1);
        ri = dwork[ca + (n - 1 - i) * (ldca + 1)];
        ti = SLC_DNRM2(&l, &theta[i * l], &int1);

        f64 neg_one = -one;
        i32 offset_ca = ca + n - 1 - i;
        SLC_DGEMV("T", &l, &n, &one, &dwork[offset_ca + 1], &ldca,
                  &theta[i * l], &int1, &zero, &dwork[jwork], &int1);

        if (ti > zero) {
            f64 coeff = (ri - one) / (ti * ti);
            SLC_DGER(&l, &n, &coeff, &theta[i * l], &int1,
                     &dwork[jwork], &int1, &dwork[offset_ca + 1], &ldca);
        } else {
            f64 neg_half = -half;
            SLC_DGER(&l, &n, &neg_half, &theta[i * l], &int1,
                     &dwork[jwork], &int1, &dwork[offset_ca + 1], &ldca);
        }

        SLC_DGER(&l, &n, &neg_one, &theta[i * l], &int1,
                 &dwork[offset_ca], &ldca, &dwork[offset_ca + 1], &ldca);

        SLC_DAXPY(&n, &ri, &dwork[offset_ca], &ldca, &dwork[jwork], &int1);

        for (j = 0; j < n; j++) {
            in = ca + n - 1 - i + j * ldca;
            for (k = in; k < in + l; k++) {
                dwork[k] = dwork[k + 1];
            }
            dwork[in + l] = dwork[jwork + j];
        }

        if (lapply && ti != zero) {
            f64 factor = tan(ti * piby2) / ti;
            SLC_DSCAL(&l, &factor, &theta[i * l], &int1);
        }
    }

    if (m > 0) {
        SLC_DLACPY("F", &n, &m, b, &ldb, &theta[n * l], &n);
        SLC_DLACPY("F", &l, &m, d, &ldd, &theta[n * (l + m)], &l);
    }

    SLC_DCOPY(&n, x0, &int1, &theta[n * (l + m) + l * m], &int1);

    dwork[0] = (f64)wrkopt;
}
