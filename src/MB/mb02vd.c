/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

void mb02vd(
    const char* trans,
    const i32 m,
    const i32 n,
    f64* a,
    const i32 lda,
    i32* ipiv,
    f64* b,
    const i32 ldb,
    i32* info
)
{
    const f64 ONE = 1.0;

    bool tran = (*trans == 'T' || *trans == 't' || *trans == 'C' || *trans == 'c');
    bool notran = (*trans == 'N' || *trans == 'n');

    *info = 0;
    if (!tran && !notran) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -5;
    } else if (ldb < (m > 1 ? m : 1)) {
        *info = -8;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("MB02VD", &neginfo);
        return;
    }

    if (m == 0 || n == 0) {
        return;
    }

    SLC_DGETRF(&n, &n, a, &lda, ipiv, info);

    if (*info == 0) {
        if (tran) {
            ma02gd(m, b, ldb, 1, n, ipiv, 1);

            f64 alpha = ONE;
            SLC_DTRSM("Right", "Lower", "Transpose", "Unit", &m, &n, &alpha, a, &lda, b, &ldb);
            SLC_DTRSM("Right", "Upper", "Transpose", "NonUnit", &m, &n, &alpha, a, &lda, b, &ldb);
        } else {
            f64 alpha = ONE;
            SLC_DTRSM("Right", "Upper", "NoTranspose", "NonUnit", &m, &n, &alpha, a, &lda, b, &ldb);
            SLC_DTRSM("Right", "Lower", "NoTranspose", "Unit", &m, &n, &alpha, a, &lda, b, &ldb);

            ma02gd(m, b, ldb, 1, n, ipiv, -1);
        }
    }
}
