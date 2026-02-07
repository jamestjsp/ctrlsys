/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot/tg.h"
#include "slicot_blas.h"

#include <stdbool.h>

void tg01md(
    const char* job, const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* alphar, f64* alphai, f64* beta,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* nf, i32* nd, i32* niblck, i32* iblck,
    const f64 tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
)
{
    const f64 one = 1.0;
    const f64 zero = 0.0;

    bool lquery, trinf;
    i32 i, ihi, ilo, minwrk, nbc, nc, nr, wrkopt;
    f64 dum[1];
    i32 int1 = 1;
    i32 tg01ld_info;

    *info = 0;
    trinf = (job[0] == 'I' || job[0] == 'i');

    if (!(job[0] == 'F' || job[0] == 'f') && !trinf) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (p < 0) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (lde < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -10;
    } else if (ldc < (p > 1 ? p : 1)) {
        *info = -12;
    } else if (ldq < (n > 1 ? n : 1)) {
        *info = -17;
    } else if (ldz < (n > 1 ? n : 1)) {
        *info = -19;
    } else if (tol >= one) {
        *info = -24;
    } else {
        lquery = (ldwork == -1);
        if (n == 0) {
            minwrk = 1;
        } else {
            minwrk = 4 * n;
        }

        if (lquery) {
            ilo = 1;
            ihi = n;

            tg01ld(job, "H", "I", "I", n, 0, 0,
                   a, lda, e, lde, dum, ldb, dum, ldc,
                   q, ldq, z, ldz, nf, nd, niblck, iblck, tol, iwork,
                   dwork, -1, &tg01ld_info);
            wrkopt = (minwrk > (i32)dwork[0]) ? minwrk : (i32)dwork[0];

            SLC_DHGEQZ("S", "V", "V", &n, &ilo, &ihi,
                       a, &lda, e, &lde, alphar, alphai, beta,
                       q, &ldq, z, &ldz, dwork, &int1, info);
            i32 dhgeqz_work = (i32)dwork[0];
            wrkopt = (wrkopt > dhgeqz_work) ? wrkopt : dhgeqz_work;
        }

        if (ldwork < minwrk && !lquery) {
            *info = -27;
        }
    }

    if (*info != 0) {
        i32 neg_info = -(*info);
        SLC_XERBLA("TG01MD", &neg_info);
        return;
    } else if (lquery) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    *nf = 0;
    *nd = 0;
    *niblck = 0;
    if (n == 0) {
        dwork[0] = one;
        return;
    }

    tg01ld(job, "H", "I", "I", n, 0, 0,
           a, lda, e, lde, dum, ldb, dum, ldc,
           q, ldq, z, ldz, nf, nd, niblck, iblck, tol, iwork,
           dwork, ldwork, &tg01ld_info);

    if (tg01ld_info != 0) {
        *info = 1;
        return;
    }
    wrkopt = (minwrk > (i32)dwork[0]) ? minwrk : (i32)dwork[0];

    if (trinf) {
        ilo = n - *nf + 1;
        ihi = n;
    } else {
        ilo = 1;
        ihi = *nf;
    }

    SLC_DHGEQZ("S", "V", "V", &n, &ilo, &ihi,
               a, &lda, e, &lde, alphar, alphai, beta,
               q, &ldq, z, &ldz, dwork, &ldwork, info);

    if (*info != 0) {
        *info = 2;
        return;
    }
    i32 dhgeqz_work = (i32)dwork[0];
    wrkopt = (wrkopt > dhgeqz_work) ? wrkopt : dhgeqz_work;

    nbc = ldwork / n;
    if (nbc < 1) nbc = 1;
    if (nbc > m) nbc = m;

    for (i = 0; i < m; i += nbc) {
        nc = nbc;
        if (nc > m - i) nc = m - i;

        SLC_DGEMM("T", "N", &n, &nc, &n, &one, q, &ldq,
                  &b[i * ldb], &ldb, &zero, dwork, &n);
        SLC_DLACPY("A", &n, &nc, dwork, &n, &b[i * ldb], &ldb);
    }

    nbc = ldwork / n;
    if (nbc < 1) nbc = 1;
    if (nbc > p) nbc = p;

    for (i = 0; i < p; i += nbc) {
        nr = nbc;
        if (nr > p - i) nr = p - i;

        SLC_DGEMM("N", "N", &nr, &n, &n, &one,
                  &c[i], &ldc, z, &ldz, &zero, dwork, &nr);
        SLC_DLACPY("A", &nr, &n, dwork, &nr, &c[i], &ldc);
    }

    dwork[0] = (f64)wrkopt;
}
