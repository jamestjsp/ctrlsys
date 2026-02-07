/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot/tg.h"
#include "slicot/mb03.h"
#include "slicot_blas.h"

#include <stdbool.h>

void tg01qd(
    const char* dico, const char* stdom, const char* jobfi,
    const i32 n, const i32 m, const i32 p, const f64 alpha,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    i32* n1, i32* n2, i32* n3, i32* nd, i32* niblck, i32* iblck,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    f64* alphar, f64* alphai, f64* beta,
    const f64 tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
)
{
    const f64 one = 1.0;
    const f64 zero = 0.0;

    bool discr, lquery, order, redif, stab;
    i32 i, lw, minwrk, nb, nbc, nc, ndim, nf, ni, nlow, nr, nsup, wrkopt;
    f64 dum[1];
    i32 neg1 = -1;
    i32 tg01md_info, mb03qg_info;

    *info = 0;
    discr = (dico[0] == 'D' || dico[0] == 'd');
    redif = (jobfi[0] == 'I' || jobfi[0] == 'i');
    stab = (stdom[0] == 'S' || stdom[0] == 's');
    order = stab || (stdom[0] == 'U' || stdom[0] == 'u');

    if (!(dico[0] == 'C' || dico[0] == 'c') && !discr) {
        *info = -1;
    } else if (!order && !(stdom[0] == 'N' || stdom[0] == 'n')) {
        *info = -2;
    } else if (!(jobfi[0] == 'F' || jobfi[0] == 'f') && !redif) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (p < 0) {
        *info = -6;
    } else if (discr && alpha < zero) {
        *info = -7;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (lde < (n > 1 ? n : 1)) {
        *info = -11;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -13;
    } else if (ldc < (p > 1 ? p : 1)) {
        *info = -15;
    } else if (ldq < (n > 1 ? n : 1)) {
        *info = -23;
    } else if (ldz < (n > 1 ? n : 1)) {
        *info = -25;
    } else if (tol >= one) {
        *info = -29;
    } else {
        if (n == 0) {
            minwrk = 1;
        } else if (order) {
            minwrk = 4 * n + 16;
        } else {
            minwrk = 4 * n;
        }
        lquery = (ldwork == -1);

        i32 max_mp = (m > p) ? m : p;
        if (max_mp == 0) max_mp = 1;
        i32 qrf_info = 0;
        SLC_DGEQRF(&n, &max_mp, a, &lda, dum, dum, &neg1, &qrf_info);
        nb = (max_mp > 1) ? (i32)(dum[0] / max_mp) : 1;
        i32 n_max_mp = n * max_mp;
        i32 nb_nb = nb * nb;
        lw = (nb_nb < n_max_mp) ? nb_nb : n_max_mp;

        if (lquery) {
            tg01md(jobfi, n, 0, 0, a, lda, e, lde, dum, ldb, dum, ldc,
                   alphar, alphai, beta, q, ldq, z, ldz, &nf, nd, niblck,
                   iblck, tol, iwork, dwork, -1, &tg01md_info);
            wrkopt = minwrk;
            if ((i32)dwork[0] > wrkopt) wrkopt = (i32)dwork[0];
            if (lw > wrkopt) wrkopt = lw;

            if (order) {
                nlow = 1;
                nsup = n;
                mb03qg(dico, stdom, "U", "U", n, nlow, nsup, alpha,
                       a, lda, e, lde, q, ldq, z, ldz, &ndim,
                       dwork, -1, &mb03qg_info);
                if ((i32)dwork[0] > wrkopt) wrkopt = (i32)dwork[0];
            }
        } else if (ldwork < minwrk) {
            *info = -32;
        }
    }

    if (*info != 0) {
        i32 neg_info = -(*info);
        SLC_XERBLA("TG01QD", &neg_info);
        return;
    } else if (lquery) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    if (n == 0) {
        *n1 = 0;
        *n2 = 0;
        *n3 = 0;
        *nd = 0;
        *niblck = 0;
        dwork[0] = one;
        return;
    }

    tg01md(jobfi, n, 0, 0, a, lda, e, lde, dum, ldb, dum, ldc,
           alphar, alphai, beta, q, ldq, z, ldz, &nf, nd, niblck,
           iblck, tol, iwork, dwork, ldwork, &tg01md_info);

    if (tg01md_info != 0) {
        if (tg01md_info == 1) {
            *info = 1;
        } else {
            *info = 2;
        }
        return;
    }

    wrkopt = minwrk;
    if ((i32)dwork[0] > wrkopt) wrkopt = (i32)dwork[0];

    ni = n - nf;
    if (order) {
        if (redif) {
            nlow = ni + 1;
            nsup = n;
        } else {
            nlow = 1;
            nsup = (nf > 1) ? nf : 1;
        }

        mb03qg(dico, stdom, "U", "U", n, nlow, nsup, alpha,
               a, lda, e, lde, q, ldq, z, ldz, &ndim,
               dwork, ldwork, &mb03qg_info);

        if (mb03qg_info != 0) {
            *info = 3;
            return;
        }

        if ((i32)dwork[0] > wrkopt) wrkopt = (i32)dwork[0];

        if (redif) {
            *n1 = ni;
            *n2 = ndim;
            *n3 = n - *n1 - *n2;
        } else {
            *n1 = ndim;
            *n3 = ni;
            *n2 = n - *n1 - *n3;
        }

        mb03qv(n, a, lda, e, lde, alphar, alphai, beta, info);
    } else {
        if (redif) {
            *n1 = ni;
            *n3 = nf;
        } else {
            *n1 = nf;
            *n3 = ni;
        }
        *n2 = 0;
    }

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

    if (lw > wrkopt) wrkopt = lw;
    dwork[0] = (f64)wrkopt;
}
