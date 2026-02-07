/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>

void sb08hd(
    const i32 n,
    const i32 m,
    const i32 p,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    const f64* cr,
    const i32 ldcr,
    f64* dr,
    const i32 lddr,
    f64* dwork,
    i32* info
)
{
    const f64 ONE = 1.0;
    const f64 MINUS_ONE = -1.0;

    f64 drnorm, rcond;
    i32 max_n_1 = (n > 1) ? n : 1;
    i32 max_p_1 = (p > 1) ? p : 1;
    i32 max_m_1 = (m > 1) ? m : 1;

    *info = 0;

    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (p < 0) {
        *info = -3;
    } else if (lda < max_n_1) {
        *info = -5;
    } else if (ldb < max_n_1) {
        *info = -7;
    } else if (ldc < max_p_1) {
        *info = -9;
    } else if (ldd < max_p_1) {
        *info = -11;
    } else if (ldcr < max_m_1) {
        *info = -13;
    } else if (lddr < max_m_1) {
        *info = -15;
    }

    if (*info != 0) {
        return;
    }

    if (m == 0) {
        dwork[0] = ONE;
        return;
    }

    // Allocate workspace for IWORK (pivot array)
    i32* iwork = (i32*)malloc(m * sizeof(i32));
    if (!iwork) {
        *info = -16;  // workspace allocation failure
        return;
    }

    // Factor DR: compute 1-norm first
    drnorm = SLC_DLANGE("1", &m, &m, dr, &lddr, dwork);

    // LU factorization: P*DR = L*U
    SLC_DGETRF(&m, &m, dr, &lddr, iwork, info);
    if (*info != 0) {
        *info = 1;
        dwork[0] = 0.0;
        free(iwork);
        return;
    }

    // Compute B = BQR * DR^{-1} using LU factorization P*DR = L*U
    // Solve B * (L*U) = BQR for B, with row permutation
    // Step 1: B := B * U^{-1}
    SLC_DTRSM("R", "U", "N", "N", &n, &m, &ONE, dr, &lddr, b, &ldb);
    // Step 2: B := B * L^{-1}
    SLC_DTRSM("R", "L", "N", "U", &n, &m, &ONE, dr, &lddr, b, &ldb);
    // Step 3: Apply column permutation (reverse of row pivoting)
    ma02gd(n, b, ldb, 1, m, iwork, -1);

    // Compute A = AQR - B * CR (where B is now BQR * DR^{-1})
    SLC_DGEMM("N", "N", &n, &n, &m, &MINUS_ONE, b, &ldb, cr, &ldcr, &ONE, a, &lda);

    // Compute D = DQ * DR^{-1}
    SLC_DTRSM("R", "U", "N", "N", &p, &m, &ONE, dr, &lddr, d, &ldd);
    SLC_DTRSM("R", "L", "N", "U", &p, &m, &ONE, dr, &lddr, d, &ldd);
    ma02gd(p, d, ldd, 1, m, iwork, -1);

    // Compute C = CQ - D * CR (where D is now DQ * DR^{-1})
    SLC_DGEMM("N", "N", &p, &n, &m, &MINUS_ONE, d, &ldd, cr, &ldcr, &ONE, c, &ldc);

    // Estimate reciprocal condition number of DR
    // Workspace: 4*M
    SLC_DGECON("1", &m, dr, &lddr, &drnorm, &rcond, dwork, iwork, info);

    f64 eps = SLC_DLAMCH("E");
    if (rcond <= eps) {
        *info = 2;
    }

    dwork[0] = rcond;
    free(iwork);
}
