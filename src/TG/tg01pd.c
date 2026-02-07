/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>
#include <stddef.h>

void tg01pd(
    const char* dico, const char* stdom, const char* jobae,
    const char* compq, const char* compz,
    const i32 n, const i32 m, const i32 p,
    const i32 nlow, const i32 nsup, const f64 alpha,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* ndim,
    f64* alphar, f64* alphai, f64* beta,
    f64* dwork, const i32 ldwork,
    i32* info
)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    bool discr, ljobg, lquery;
    i32 icompq, icompz, lw, minwrk, nb, nbc, nc, nr, sdim, wrkopt;
    i32 max_mp;

    if (compq[0] == 'I' || compq[0] == 'i') {
        icompq = 1;
    } else if (compq[0] == 'U' || compq[0] == 'u') {
        icompq = 2;
    } else {
        icompq = 0;
    }

    if (compz[0] == 'I' || compz[0] == 'i') {
        icompz = 1;
    } else if (compz[0] == 'U' || compz[0] == 'u') {
        icompz = 2;
    } else {
        icompz = 0;
    }

    *info = 0;
    discr = (dico[0] == 'D' || dico[0] == 'd');
    ljobg = (jobae[0] == 'G' || jobae[0] == 'g');

    if (!(dico[0] == 'C' || dico[0] == 'c' || discr)) {
        *info = -1;
    } else if (!(stdom[0] == 'S' || stdom[0] == 's' || stdom[0] == 'U' || stdom[0] == 'u')) {
        *info = -2;
    } else if (!(jobae[0] == 'S' || jobae[0] == 's' || ljobg)) {
        *info = -3;
    } else if (icompq <= 0 || (ljobg && icompq == 2)) {
        *info = -4;
    } else if (icompz <= 0 || (ljobg && icompz == 2)) {
        *info = -5;
    } else if (n < 0) {
        *info = -6;
    } else if (m < 0) {
        *info = -7;
    } else if (p < 0) {
        *info = -8;
    } else if ((ljobg && nlow != ((1 < n) ? 1 : n)) ||
               (!ljobg && nlow < 0)) {
        *info = -9;
    } else if ((ljobg && nsup != n) ||
               (!ljobg && (nsup < nlow || n < nsup))) {
        *info = -10;
    } else if (discr && alpha < zero) {
        *info = -11;
    } else if (lda < ((1 > n) ? 1 : n)) {
        *info = -13;
    } else if (lde < ((1 > n) ? 1 : n)) {
        *info = -15;
    } else if (ldb < ((1 > n) ? 1 : n)) {
        *info = -17;
    } else if (ldc < ((1 > p) ? 1 : p)) {
        *info = -19;
    } else if (ldq < ((1 > n) ? 1 : n)) {
        *info = -21;
    } else if (ldz < ((1 > n) ? 1 : n)) {
        *info = -23;
    } else {
        if (n == 0) {
            minwrk = 1;
        } else if (ljobg) {
            minwrk = 8 * n + 16;
        } else {
            minwrk = 4 * n + 16;
        }
        lquery = (ldwork == -1);

        max_mp = (m > p) ? m : p;
        if (max_mp < 1) max_mp = 1;

        i32 lwork_query_qrf = -1;
        SLC_DGEQRF(&n, &max_mp, a, &lda, dwork, dwork, &lwork_query_qrf, info);
        nb = (i32)(dwork[0]) / max_mp;
        lw = (nb * nb < n * max_mp) ? nb * nb : n * max_mp;

        if (lquery) {
            wrkopt = minwrk;
            if (ljobg) {
                i32 lwork_query = -1;
                i32 bwork_dummy = 0;
                SLC_DGGES("V", "V", "N", NULL, &n, a, &lda, e, &lde, &sdim,
                          alphar, alphai, beta, q, &ldq, z, &ldz,
                          dwork, &lwork_query, &bwork_dummy, info);
                if ((i32)(dwork[0]) > wrkopt) wrkopt = (i32)(dwork[0]);
            }
            i32 lwork_query2 = -1;
            mb03qg(dico, stdom, "U", "U", n, nlow, nsup, alpha,
                   a, lda, e, lde, q, ldq, z, ldz, ndim, dwork, lwork_query2, info);
            if ((i32)(dwork[0]) > wrkopt) wrkopt = (i32)(dwork[0]);
            if (lw > wrkopt) wrkopt = lw;
        } else if (ldwork < minwrk) {
            *info = -29;
        }
    }

    if (*info != 0) {
        return;
    } else if (lquery) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    *ndim = 0;
    if (n == 0) {
        dwork[0] = one;
        return;
    }

    if (ljobg) {
        i32 bwork_dummy = 0;
        SLC_DGGES("V", "V", "N", NULL, &n, a, &lda, e, &lde, &sdim,
                  alphar, alphai, beta, q, &ldq, z, &ldz,
                  dwork, &ldwork, &bwork_dummy, info);
        if (*info != 0) {
            *info = 1;
            return;
        }
        wrkopt = (minwrk > (i32)(dwork[0])) ? minwrk : (i32)(dwork[0]);
    } else {
        if (icompq == 1) {
            SLC_DLASET("F", &n, &n, &zero, &one, q, &ldq);
        }
        if (icompz == 1) {
            SLC_DLASET("F", &n, &n, &zero, &one, z, &ldz);
        }
        wrkopt = minwrk;
    }

    mb03qg(dico, stdom, "U", "U", n, nlow, nsup, alpha,
           a, lda, e, lde, q, ldq, z, ldz, ndim, dwork, ldwork, info);
    if (*info != 0) {
        *info = 2;
        return;
    }
    if ((i32)(dwork[0]) > wrkopt) wrkopt = (i32)(dwork[0]);

    mb03qv(n, a, lda, e, lde, alphar, alphai, beta, info);

    nbc = (ldwork / n > 1) ? ldwork / n : 1;
    nbc = (nbc < m) ? nbc : m;
    for (i32 i = 0; i < m; i += nbc) {
        nc = ((nbc < m - i) ? nbc : m - i);
        i32 col = i;
        SLC_DGEMM("T", "N", &n, &nc, &n, &one, q, &ldq,
                  &b[col * ldb], &ldb, &zero, dwork, &n);
        SLC_DLACPY("A", &n, &nc, dwork, &n, &b[col * ldb], &ldb);
    }

    nbc = (ldwork / n > 1) ? ldwork / n : 1;
    nbc = (nbc < p) ? nbc : p;
    for (i32 i = 0; i < p; i += nbc) {
        nr = ((nbc < p - i) ? nbc : p - i);
        i32 row = i;
        SLC_DGEMM("N", "N", &nr, &n, &n, &one,
                  &c[row], &ldc, z, &ldz, &zero, dwork, &nr);
        SLC_DLACPY("A", &nr, &n, dwork, &nr, &c[row], &ldc);
    }

    i32 opt = (wrkopt > lw) ? wrkopt : lw;
    dwork[0] = (f64)opt;
}
