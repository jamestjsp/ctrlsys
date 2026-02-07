/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

void mb01uy(
    const char* side, const char* uplo, const char* trans,
    const i32 m, const i32 n,
    const f64 alpha,
    f64* t, const i32 ldt,
    const f64* a, const i32 lda,
    f64* dwork, const i32 ldwork,
    i32* info
)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const i32 inc1 = 1;

    bool lside, luplo, ltran;
    i32 k, l, mn, wrkmin, nb;

    *info = 0;
    lside = (*side == 'L' || *side == 'l');
    luplo = (*uplo == 'U' || *uplo == 'u');
    ltran = (*trans == 'T' || *trans == 't' || *trans == 'C' || *trans == 'c');

    if (lside) {
        k = m;
        l = n;
    } else {
        k = n;
        l = m;
    }
    mn = (m < n) ? m : n;

    wrkmin = 1;
    if (alpha != zero && mn > 0) {
        wrkmin = (wrkmin > k) ? wrkmin : k;
    }

    if (ldwork == -1) {
        dwork[0] = (f64)(m * n);
        return;
    }

    if ((!lside && *side != 'R' && *side != 'r')) {
        *info = -1;
        return;
    }
    if ((!luplo && *uplo != 'L' && *uplo != 'l')) {
        *info = -2;
        return;
    }
    if ((!ltran && *trans != 'N' && *trans != 'n')) {
        *info = -3;
        return;
    }
    if (m < 0) {
        *info = -4;
        return;
    }
    if (n < 0) {
        *info = -5;
        return;
    }
    if (ldt < ((m > 1) ? m : 1) || (!lside && ldt < n)) {
        *info = -8;
        return;
    }
    if (lda < ((m > 1) ? m : 1)) {
        *info = -10;
        return;
    }
    if (ldwork < wrkmin) {
        dwork[0] = (f64)wrkmin;
        *info = -12;
        return;
    }

    if (mn == 0) {
        return;
    }

    if (alpha == zero) {
        SLC_DLASET("F", &m, &n, &zero, &zero, t, &ldt);
        return;
    }

    nb = (l > 0) ? ldwork / l : 1;
    if (nb < 1) nb = 1;
    if (nb > k) nb = k;

    if (ldwork >= m * n) {
        // Fast BLAS 3 path - enough workspace for full copy
        SLC_DLACPY("A", &m, &n, a, &lda, dwork, &m);
        SLC_DTRMM(side, uplo, trans, "N", &m, &n, &alpha, t, &ldt, dwork, &m);
        SLC_DLACPY("A", &m, &n, dwork, &m, t, &ldt);
    } else if (nb > 1) {
        // BLAS 3 loop path - blocked algorithm with panels of size NB
        const char *uploc, *tranc;
        char uploc_buf[2] = {0};

        if (ltran) {
            ma02ed(*uplo, k, t, ldt);
            if (luplo) {
                uploc = "L";
            } else {
                uploc = "U";
            }
            tranc = "N";
            luplo = !luplo;
            ltran = !ltran;
        } else {
            uploc_buf[0] = *uplo;
            uploc = uploc_buf;
            tranc = trans;
        }

        i32 bl = k / nb;
        if (bl < 1) bl = 1;
        i32 j = (k < nb * bl) ? k : nb * bl;

        if (lside) {
            if (luplo) {
                // Upper triangular, left side: compute last rows first
                i32 nr, ii;
                if (j == m) {
                    nr = nb;
                    ii = m - nb;  // 0-based
                    bl = bl - 1;
                } else {
                    nr = m - j;
                    ii = j;  // 0-based
                }
                SLC_DLACPY("A", &nr, &n, &a[ii], &lda, dwork, &nr);
                SLC_DTRMM("L", uploc, tranc, "N", &nr, &n, &alpha,
                          &t[ii + ii*ldt], &ldt, dwork, &nr);
                SLC_DLACPY("A", &nr, &n, dwork, &nr, &t[ii], &ldt);

                for (i32 i = 0; i < bl; i++) {
                    i32 ij = ii;
                    ii = ii - nb;
                    SLC_DLACPY("A", &nb, &n, &a[ii], &lda, dwork, &nb);
                    SLC_DTRMM("L", uploc, tranc, "N", &nb, &n,
                              &alpha, &t[ii + ii*ldt], &ldt, dwork, &nb);
                    i32 gemm_k = m - ij;
                    SLC_DGEMM(tranc, "N", &nb, &n, &gemm_k, &alpha,
                              &t[ii + ij*ldt], &ldt, &a[ij], &lda,
                              &one, dwork, &nb);
                    SLC_DLACPY("A", &nb, &n, dwork, &nb, &t[ii], &ldt);
                }
            } else {
                // Lower triangular, left side: compute first rows first
                i32 nr, ii;
                if (j == m) {
                    nr = nb;
                    bl = bl - 1;
                } else {
                    nr = m - j;
                }
                SLC_DLACPY("A", &nr, &n, a, &lda, dwork, &nr);
                SLC_DTRMM("L", uploc, tranc, "N", &nr, &n, &alpha,
                          t, &ldt, dwork, &nr);
                SLC_DLACPY("A", &nr, &n, dwork, &nr, t, &ldt);
                ii = nr;  // 0-based: next panel starts here

                for (i32 i = 0; i < bl; i++) {
                    SLC_DLACPY("A", &nb, &n, &a[ii], &lda, dwork, &nb);
                    SLC_DTRMM("L", uploc, tranc, "N", &nb, &n,
                              &alpha, &t[ii + ii*ldt], &ldt, dwork, &nb);
                    i32 gemm_k = ii;
                    SLC_DGEMM(tranc, "N", &nb, &n, &gemm_k, &alpha,
                              &t[ii], &ldt, a, &lda, &one, dwork, &nb);
                    SLC_DLACPY("A", &nb, &n, dwork, &nb, &t[ii], &ldt);
                    ii = ii + nb;
                }
            }
        } else {
            if (luplo) {
                // Upper triangular, right side: compute first columns first
                i32 nc, ii;
                ii = 0;
                if (j == n) {
                    nc = nb;
                    bl = bl - 1;
                } else {
                    nc = n - j;
                }
                SLC_DLACPY("A", &m, &nc, a, &lda, dwork, &m);
                SLC_DTRMM("R", uploc, tranc, "N", &m, &nc, &alpha,
                          t, &ldt, dwork, &m);
                SLC_DLACPY("A", &m, &nc, dwork, &m, t, &ldt);
                ii = ii + nc;

                for (i32 i = 0; i < bl; i++) {
                    i32 ij = ii;  // 0-based: ii already 0-based
                    SLC_DLACPY("A", &m, &nb, &a[ii*lda], &lda, dwork, &m);
                    SLC_DTRMM("R", uploc, tranc, "N", &m, &nb,
                              &alpha, &t[ii + ii*ldt], &ldt, dwork, &m);
                    i32 gemm_k = ij;
                    SLC_DGEMM(tranc, "N", &m, &nb, &gemm_k, &alpha,
                              a, &lda, &t[ii*ldt], &ldt, &one, dwork, &m);
                    SLC_DLACPY("A", &m, &nb, dwork, &m, &t[ii*ldt], &ldt);
                    ii = ii + nb;
                }
            } else {
                // Lower triangular, right side: compute last columns first
                i32 nc, ii;
                if (j == n) {
                    nc = nb;
                    ii = n - nb;  // 0-based
                    bl = bl - 1;
                } else {
                    nc = n - j;
                    ii = j;  // 0-based
                }
                SLC_DLACPY("A", &m, &nc, &a[ii*lda], &lda, dwork, &m);
                SLC_DTRMM("R", uploc, tranc, "N", &m, &nc, &alpha,
                          &t[ii + ii*ldt], &ldt, dwork, &m);
                SLC_DLACPY("A", &m, &nc, dwork, &m, &t[ii*ldt], &ldt);

                for (i32 i = 0; i < bl; i++) {
                    i32 ij = ii;
                    ii = ii - nb;
                    SLC_DLACPY("A", &m, &nb, &a[ii*lda], &lda, dwork, &m);
                    SLC_DTRMM("R", uploc, tranc, "N", &m, &nb,
                              &alpha, &t[ii + ii*ldt], &ldt, dwork, &m);
                    SLC_DGEMM(tranc, "N", &m, &nb, &nc, &alpha,
                              &a[ij*lda], &lda, &t[ij + ii*ldt], &ldt,
                              &one, dwork, &m);
                    SLC_DLACPY("A", &m, &nb, dwork, &m, &t[ii*ldt], &ldt);
                    nc = nc + nb;
                }
            }
        }
    } else {
        // BLAS 2 path - minimal workspace
        bool upnt = luplo && !ltran;
        bool lotr = ltran && !luplo;
        bool uptr = luplo && ltran;
        bool lont = !luplo && !ltran;

        if (luplo || lotr) {
            ma02ed(*uplo, k, t, ldt);
        }

        if (lside) {
            if (upnt || lotr) {
                for (i32 i = 0; i < m; i++) {
                    i32 len = m - i;
                    SLC_DCOPY(&len, &t[i + i*ldt], &inc1, dwork, &inc1);
                    SLC_DGEMV("T", &len, &n, &alpha, &a[i], &lda,
                              dwork, &inc1, &zero, &t[i], &ldt);
                }
            } else if (uptr || lont) {
                for (i32 i = 0; i < m; i++) {
                    i32 len = i + 1;
                    SLC_DCOPY(&len, &t[i], &ldt, dwork, &inc1);
                    SLC_DGEMV("T", &len, &n, &alpha, a, &lda,
                              dwork, &inc1, &zero, &t[i], &ldt);
                }
            }
        } else {
            if (upnt || lotr) {
                for (i32 i = 0; i < n; i++) {
                    i32 len = i + 1;
                    SLC_DCOPY(&len, &t[i*ldt], &inc1, dwork, &inc1);
                    SLC_DGEMV("N", &m, &len, &alpha, a, &lda,
                              dwork, &inc1, &zero, &t[i*ldt], &inc1);
                }
            } else if (uptr || lont) {
                for (i32 i = 0; i < n; i++) {
                    i32 len = n - i;
                    SLC_DCOPY(&len, &t[i + i*ldt], &inc1, dwork, &inc1);
                    SLC_DGEMV("N", &m, &len, &alpha, &a[i*lda], &lda,
                              dwork, &inc1, &zero, &t[i*ldt], &inc1);
                }
            }
        }
    }

    dwork[0] = (f64)(m * n);
}
