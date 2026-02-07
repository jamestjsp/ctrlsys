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
    } else {
        // BLAS 2 path - minimal workspace
        // Fill in T by symmetry, then use DGEMV
        bool upnt = luplo && !ltran;  // upper, no-trans
        bool lotr = ltran && !luplo;  // trans, lower
        bool uptr = luplo && ltran;   // upper, trans
        bool lont = !luplo && !ltran; // lower, no-trans

        if (luplo || lotr) {
            ma02ed(*uplo, k, t, ldt);
        }

        if (lside) {
            if (upnt || lotr) {
                // T(i,:) = alpha * T(i:m,i)' * A(i:m,:)
                for (i32 i = 0; i < m; i++) {
                    i32 len = m - i;
                    SLC_DCOPY(&len, &t[i + i*ldt], &inc1, dwork, &inc1);
                    SLC_DGEMV("T", &len, &n, &alpha, &a[i], &lda,
                              dwork, &inc1, &zero, &t[i], &ldt);
                }
            } else if (uptr || lont) {
                // T(i,:) = alpha * T(0:i,i)' * A(0:i,:)
                for (i32 i = 0; i < m; i++) {
                    i32 len = i + 1;
                    SLC_DCOPY(&len, &t[i], &ldt, dwork, &inc1);
                    SLC_DGEMV("T", &len, &n, &alpha, a, &lda,
                              dwork, &inc1, &zero, &t[i], &ldt);
                }
            }
        } else {
            if (upnt || lotr) {
                // T(:,i) = alpha * A(:,0:i) * T(0:i,i)
                for (i32 i = 0; i < n; i++) {
                    i32 len = i + 1;
                    SLC_DCOPY(&len, &t[i*ldt], &inc1, dwork, &inc1);
                    SLC_DGEMV("N", &m, &len, &alpha, a, &lda,
                              dwork, &inc1, &zero, &t[i*ldt], &inc1);
                }
            } else if (uptr || lont) {
                // T(:,i) = alpha * A(:,i:n) * T(i:n,i)
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
