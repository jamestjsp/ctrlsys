/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdbool.h>

void mb02ed(const char* typet, i32 k, i32 n, i32 nrhs,
            f64* t, i32 ldt, f64* b, i32 ldb,
            f64* dwork, i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 neg_one = -1.0;

    char typet_u = (char)toupper((unsigned char)typet[0]);
    bool isrow = (typet_u == 'R');

    *info = 0;

    if (!isrow && typet_u != 'C') {
        *info = -1;
    } else if (k < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (nrhs < 0) {
        *info = -4;
    } else if (ldt < 1 || (isrow && ldt < k) || (!isrow && ldt < n * k)) {
        *info = -6;
    } else if (ldb < 1 || (isrow && ldb < nrhs) || (!isrow && ldb < n * k)) {
        *info = -8;
    } else if (ldwork < (n * k * k + (n + 2) * k > 1 ? n * k * k + (n + 2) * k : 1)) {
        dwork[0] = (f64)(n * k * k + (n + 2) * k > 1 ? n * k * k + (n + 2) * k : 1);
        *info = -10;
    }

    if (*info != 0) {
        return;
    }

    i32 min_knrhs = k;
    if (n < min_knrhs) min_knrhs = n;
    if (nrhs < min_knrhs) min_knrhs = nrhs;
    if (min_knrhs == 0) {
        dwork[0] = one;
        return;
    }

    i32 maxwrk = 0;
    i32 startn = 0;
    i32 startt = n * k * k;
    i32 starth = startt + 3 * k;
    i32 ierr = 0;

    i32 nk = n * k;

    if (isrow) {
        SLC_DPOTRF("U", &k, t, &ldt, &ierr);
        if (ierr != 0) {
            *info = 1;
            return;
        }

        if (n > 1) {
            i32 cols = (n - 1) * k;
            SLC_DTRSM("L", "U", "T", "N", &k, &cols, &one, t, &ldt, &t[0 + k * ldt], &ldt);
        }

        SLC_DTRSM("R", "U", "N", "N", &nrhs, &k, &one, t, &ldt, b, &ldb);
        if (n > 1) {
            i32 cols = (n - 1) * k;
            SLC_DGEMM("N", "N", &nrhs, &cols, &k, &one, b, &ldb, &t[0 + k * ldt], &ldt, &neg_one, &b[0 + k * ldb], &ldb);
        }

        SLC_DLASET("A", &k, &k, &zero, &one, &dwork[startn], &k);
        SLC_DTRSM("L", "U", "T", "N", &k, &k, &one, t, &ldt, &dwork[startn], &k);
        if (n > 1) {
            i32 cols = (n - 1) * k;
            SLC_DLACPY("A", &k, &cols, &t[0 + k * ldt], &ldt, &dwork[startn + k * k], &k);
        }
        i32 last_blk = (n - 1) * k;
        SLC_DLACPY("A", &k, &k, &dwork[startn], &k, &t[0 + last_blk * ldt], &ldt);

        SLC_DTRMM("R", "L", "N", "N", &nrhs, &k, &one, &t[0 + last_blk * ldt], &ldt, b, &ldb);

        for (i32 i = 1; i < n; i++) {
            i32 startr = i * k;
            i32 starti = (n - 1 - i) * k;

            mb02cx("R", k, k, k, t, ldt, &dwork[startn + i * k * k], k,
                   &dwork[startt], 3 * k, &dwork[starth], ldwork - starth, &ierr);
            if (ierr != 0) {
                *info = 1;
                return;
            }
            maxwrk = (maxwrk > (i32)dwork[starth]) ? maxwrk : (i32)dwork[starth];

            if (n > i + 1) {
                i32 cols = (n - 1 - i) * k;
                mb02cy("R", "N", k, k, cols, k, &t[0 + k * ldt], ldt, &dwork[startn + (i + 1) * k * k], k,
                       &dwork[startn + i * k * k], k, &dwork[startt], 3 * k, &dwork[starth], ldwork - starth, &ierr);
                maxwrk = (maxwrk > (i32)dwork[starth]) ? maxwrk : (i32)dwork[starth];
            }

            SLC_DTRSM("R", "U", "N", "N", &nrhs, &k, &neg_one, t, &ldt, &b[0 + startr * ldb], &ldb);
            if (n > i + 1) {
                i32 cols = (n - 1 - i) * k;
                SLC_DGEMM("N", "N", &nrhs, &cols, &k, &one, &b[0 + startr * ldb], &ldb, &t[0 + k * ldt], &ldt, &one, &b[0 + (startr + k) * ldb], &ldb);
            }

            SLC_DLASET("A", &k, &k, &zero, &zero, &t[0 + starti * ldt], &ldt);
            i32 prev_cols = i * k;
            mb02cy("R", "N", k, k, prev_cols, k, &t[0 + starti * ldt], ldt, &dwork[startn], k,
                   &dwork[startn + i * k * k], k, &dwork[startt], 3 * k, &dwork[starth], ldwork - starth, &ierr);
            maxwrk = (maxwrk > (i32)dwork[starth]) ? maxwrk : (i32)dwork[starth];

            mb02cy("R", "T", k, k, k, k, &t[0 + last_blk * ldt], ldt, &dwork[startn + i * k * k], k,
                   &dwork[startn + i * k * k], k, &dwork[startt], 3 * k, &dwork[starth], ldwork - starth, &ierr);
            maxwrk = (maxwrk > (i32)dwork[starth]) ? maxwrk : (i32)dwork[starth];

            SLC_DGEMM("N", "N", &nrhs, &prev_cols, &k, &one, &b[0 + startr * ldb], &ldb, &t[0 + starti * ldt], &ldt, &one, b, &ldb);
            SLC_DTRMM("R", "L", "N", "N", &nrhs, &k, &one, &t[0 + last_blk * ldt], &ldt, &b[0 + startr * ldb], &ldb);
        }
    } else {
        SLC_DPOTRF("L", &k, t, &ldt, &ierr);
        if (ierr != 0) {
            *info = 1;
            return;
        }

        if (n > 1) {
            i32 rows = (n - 1) * k;
            SLC_DTRSM("R", "L", "T", "N", &rows, &k, &one, t, &ldt, &t[k], &ldt);
        }

        SLC_DTRSM("L", "L", "N", "N", &k, &nrhs, &one, t, &ldt, b, &ldb);
        if (n > 1) {
            i32 rows = (n - 1) * k;
            SLC_DGEMM("N", "N", &rows, &nrhs, &k, &one, &t[k], &ldt, b, &ldb, &neg_one, &b[k], &ldb);
        }

        SLC_DLASET("A", &k, &k, &zero, &one, &dwork[startn], &nk);
        SLC_DTRSM("R", "L", "T", "N", &k, &k, &one, t, &ldt, &dwork[startn], &nk);
        if (n > 1) {
            i32 rows = (n - 1) * k;
            SLC_DLACPY("A", &rows, &k, &t[k], &ldt, &dwork[startn + k], &nk);
        }
        i32 last_blk = (n - 1) * k;
        SLC_DLACPY("A", &k, &k, &dwork[startn], &nk, &t[last_blk], &ldt);

        SLC_DTRMM("L", "U", "N", "N", &k, &nrhs, &one, &t[last_blk], &ldt, b, &ldb);

        for (i32 i = 1; i < n; i++) {
            i32 startr = i * k;
            i32 starti = (n - 1 - i) * k;

            mb02cx("C", k, k, k, t, ldt, &dwork[startn + i * k], nk,
                   &dwork[startt], 3 * k, &dwork[starth], ldwork - starth, &ierr);
            if (ierr != 0) {
                *info = 1;
                return;
            }
            maxwrk = (maxwrk > (i32)dwork[starth]) ? maxwrk : (i32)dwork[starth];

            if (n > i + 1) {
                i32 rows = (n - 1 - i) * k;
                mb02cy("C", "N", k, k, rows, k, &t[k], ldt, &dwork[startn + (i + 1) * k], nk,
                       &dwork[startn + i * k], nk, &dwork[startt], 3 * k, &dwork[starth], ldwork - starth, &ierr);
                maxwrk = (maxwrk > (i32)dwork[starth]) ? maxwrk : (i32)dwork[starth];
            }

            SLC_DTRSM("L", "L", "N", "N", &k, &nrhs, &neg_one, t, &ldt, &b[startr], &ldb);
            if (n > i + 1) {
                i32 rows = (n - 1 - i) * k;
                SLC_DGEMM("N", "N", &rows, &nrhs, &k, &one, &t[k], &ldt, &b[startr], &ldb, &one, &b[startr + k], &ldb);
            }

            SLC_DLASET("A", &k, &k, &zero, &zero, &t[starti], &ldt);
            i32 prev_rows = i * k;
            mb02cy("C", "N", k, k, prev_rows, k, &t[starti], ldt, &dwork[startn], nk,
                   &dwork[startn + i * k], nk, &dwork[startt], 3 * k, &dwork[starth], ldwork - starth, &ierr);
            maxwrk = (maxwrk > (i32)dwork[starth]) ? maxwrk : (i32)dwork[starth];

            mb02cy("C", "T", k, k, k, k, &t[last_blk], ldt, &dwork[startn + i * k], nk,
                   &dwork[startn + i * k], nk, &dwork[startt], 3 * k, &dwork[starth], ldwork - starth, &ierr);
            maxwrk = (maxwrk > (i32)dwork[starth]) ? maxwrk : (i32)dwork[starth];

            SLC_DGEMM("N", "N", &prev_rows, &nrhs, &k, &one, &t[starti], &ldt, &b[startr], &ldb, &one, b, &ldb);
            SLC_DTRMM("L", "U", "N", "N", &k, &nrhs, &one, &t[last_blk], &ldt, &b[startr], &ldb);
        }
    }

    dwork[0] = (f64)(starth + maxwrk > 1 ? starth + maxwrk : 1);
}
