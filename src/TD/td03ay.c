/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void td03ay(
    const i32 mwork,
    const i32 pwork,
    const i32* index,
    const f64* dcoeff,
    const i32 lddcoe,
    const f64* ucoeff,
    const i32 lduco1,
    const i32 lduco2,
    const i32 n,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    i32 i, ia, ibias, indcur, ja, jmax1, k;
    f64 absdia, absdmx, bignum, diag, smlnum, umax1, temp;

    *info = 0;

    SLC_DLASET("U", &n, &n, &ZERO, &ZERO, a, &lda);
    if (n > 1) {
        i32 nm1 = n - 1;
        SLC_DLASET("L", &nm1, &nm1, &ZERO, &ONE, &a[1], &lda);
    }

    SLC_DLASET("F", &pwork, &n, &ZERO, &ZERO, c, &ldc);

    smlnum = SLC_DLAMCH("S") / SLC_DLAMCH("P");
    bignum = ONE / smlnum;

    ibias = 2;
    ja = 0;

    for (i = 0; i < pwork; i++) {
        absdia = fabs(dcoeff[i]);

        jmax1 = SLC_IDAMAX(&mwork, &ucoeff[i], &lduco1) - 1;
        if (jmax1 < 0) jmax1 = 0;
        umax1 = fabs(ucoeff[i + jmax1 * lduco1]);

        if ((absdia < smlnum) ||
            (absdia < ONE && umax1 > absdia * bignum)) {
            *info = i + 1;
            return;
        }

        diag = ONE / dcoeff[i];
        indcur = index[i];

        if (indcur != 0) {
            ibias = ibias + indcur;
            ja = ja + indcur;

            if (indcur >= 1) {
                jmax1 = SLC_IDAMAX(&indcur, &dcoeff[i + lddcoe], &lddcoe) - 1;
                if (jmax1 < 0) jmax1 = 0;
                absdmx = fabs(dcoeff[i + (jmax1 + 1) * lddcoe]);

                if (absdia >= ONE) {
                    if (umax1 > ONE) {
                        if ((absdmx / absdia) > (bignum / umax1)) {
                            *info = i + 1;
                            return;
                        }
                    }
                } else {
                    if (umax1 > ONE) {
                        if (absdmx > (bignum * absdia) / umax1) {
                            *info = i + 1;
                            return;
                        }
                    }
                }
            }

            for (k = 1; k <= indcur; k++) {
                ia = ibias - k - 2;

                temp = -diag * dcoeff[i + k * lddcoe];
                a[ia + (ja - 1) * lda] = temp;

                SLC_DCOPY(&mwork, &ucoeff[i + k * lduco1 * lduco2], &lduco1,
                          &b[ia], &ldb);
                SLC_DAXPY(&mwork, &temp, &ucoeff[i], &lduco1, &b[ia], &ldb);
            }

            if (ja < n) {
                a[ja + (ja - 1) * lda] = ZERO;
            }

            c[i + (ja - 1) * ldc] = diag;
        }

        SLC_DCOPY(&mwork, &ucoeff[i], &lduco1, &d[i], &ldd);
        SLC_DSCAL(&mwork, &diag, &d[i], &ldd);
    }
}
