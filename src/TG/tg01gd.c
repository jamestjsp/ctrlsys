/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>
#include <string.h>

void tg01gd(
    const char* jobs,
    const i32 l, const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* d, const i32 ldd,
    i32* lr, i32* nr, i32* ranke, i32* infred,
    const f64 tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
)
{
    const f64 one = 1.0, zero = 0.0;

    bool lquery, lspace, sstype;
    i32 k, k1, kwa, kwb, kwc, kwe, kwr, ls, lwrmin;
    i32 ns, rnka22, wrkopt;
    f64 dum[1];
    i32 max1l = (1 > l) ? 1 : l;
    i32 max1p = (1 > p) ? 1 : p;
    i32 ln = (l < n) ? l : n;

    sstype = (jobs[0] == 'S' || jobs[0] == 's');

    *info = 0;

    if (!sstype && jobs[0] != 'D' && jobs[0] != 'd') {
        *info = -1;
    } else if (l < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (p < 0) {
        *info = -5;
    } else if (lda < max1l) {
        *info = -7;
    } else if (lde < max1l) {
        *info = -9;
    } else if (ldb < max1l) {
        *info = -11;
    } else if (ldc < max1p) {
        *info = -13;
    } else if (ldd < max1p) {
        *info = -15;
    } else if (tol >= one) {
        *info = -20;
    } else {
        if (ln == 0) {
            lwrmin = 1;
        } else {
            i32 temp1 = n + p;
            i32 temp2 = 3 * n - 1;
            temp2 = (temp2 > m) ? temp2 : m;
            temp2 = (temp2 > l) ? temp2 : l;
            temp2 = ln + temp2;
            lwrmin = (temp1 > temp2) ? temp1 : temp2;
        }
        lquery = (ldwork == -1);

        if (lquery) {
            i32 dummy_ranke, dummy_rnka22;
            tg01fd("N", "N", "R", l, n, m, p, a, lda, e, lde,
                   b, ldb, c, ldc, dum, 1, dum, 1,
                   &dummy_ranke, &dummy_rnka22, tol, iwork, dwork, -1, info);
            wrkopt = (lwrmin > (i32)dwork[0]) ? lwrmin : (i32)dwork[0];
        } else if (ldwork < lwrmin) {
            *info = -23;
        }
    }

    if (*info != 0) {
        return;
    } else if (lquery) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    *lr = l;
    *nr = n;
    if (ln == 0) {
        dwork[0] = one;
        *ranke = 0;
        *infred = -1;
        return;
    }

    lspace = (ldwork >= (lwrmin + l * (2 * n + m) + p * n)) && !sstype;

    if (lspace) {
        kwa = 0;
        kwe = kwa + l * n;
        kwb = kwe + l * n;
        kwc = kwb + l * m;
        kwr = kwc + p * n;
        SLC_DLACPY("F", &l, &n, a, &lda, &dwork[kwa], &l);
        SLC_DLACPY("F", &l, &n, e, &lde, &dwork[kwe], &l);
        SLC_DLACPY("F", &l, &m, b, &ldb, &dwork[kwb], &l);
        SLC_DLACPY("F", &p, &n, c, &ldc, &dwork[kwc], &max1p);
    } else {
        kwr = 0;
    }

    i32 ldwork_eff = ldwork - kwr;
    tg01fd("N", "N", "R", l, n, m, p, a, lda, e, lde,
           b, ldb, c, ldc, dum, 1, dum, 1,
           ranke, &rnka22, tol, iwork, &dwork[kwr], ldwork_eff, info);

    if (*info == 0) {
        *infred = rnka22;
        if (rnka22 > 0) {
            k = *ranke;
            k1 = l;
            if (n < k1) k1 = n;
            if (k + rnka22 < k1) k1 = k + rnka22;

            *lr = l - rnka22;
            *nr = n - rnka22;
            ls = *lr - *ranke;
            ns = *nr - *ranke;

            SLC_DTRSM("L", "U", "N", "N", &rnka22, ranke, &one,
                      &a[k + k * lda], &lda, &a[k], &lda);

            SLC_DTRSM("L", "U", "N", "N", &rnka22, &m, &one,
                      &a[k + k * lda], &lda, &b[k], &ldb);

            f64 neg_one = -one;
            SLC_DGEMM("N", "N", &p, &m, &rnka22, &neg_one,
                      &c[k * ldc], &ldc, &b[k], &ldb, &one, d, &ldd);

            SLC_DGEMM("N", "N", ranke, &m, &rnka22, &neg_one,
                      &a[k * lda], &lda, &b[k], &ldb, &one, b, &ldb);
            SLC_DLACPY("F", &ls, &m, &b[k1], &ldb, &b[k], &ldb);

            SLC_DGEMM("N", "N", &p, ranke, &rnka22, &neg_one,
                      &c[k * ldc], &ldc, &a[k], &lda, &one, c, &ldc);
            SLC_DLACPY("F", &p, &ns, &c[k1 * ldc], &ldc, &c[k * ldc], &ldc);

            SLC_DGEMM("N", "N", ranke, ranke, &rnka22, &neg_one,
                      &a[k * lda], &lda, &a[k], &lda, &one, a, &lda);

            SLC_DLACPY("F", &ls, nr, &a[k1], &lda, &a[k], &lda);
            SLC_DLACPY("F", ranke, &ns, &a[k1 * lda], &lda, &a[k * lda], &lda);
        } else {
            if (lspace) {
                SLC_DLACPY("F", &l, &n, &dwork[kwa], &l, a, &lda);
                SLC_DLACPY("F", &l, &n, &dwork[kwe], &l, e, &lde);
                SLC_DLACPY("F", &l, &m, &dwork[kwb], &l, b, &ldb);
                SLC_DLACPY("F", &p, &n, &dwork[kwc], &max1p, c, &ldc);
                *infred = -1;
            }
        }

        if (sstype) {
            SLC_DTRSM("L", "U", "N", "N", ranke, nr, &one, e, &lde, a, &lda);

            SLC_DTRSM("L", "U", "N", "N", ranke, &m, &one, e, &lde, b, &ldb);

            SLC_DLASET("F", ranke, ranke, &zero, &one, e, &lde);
        }
        dwork[0] = dwork[kwr];
    }
}
