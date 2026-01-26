/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void tb03ay(i32 nr, const f64* a, i32 lda, i32 indblk, const i32* nblk,
            f64* vcoeff, i32 ldvco1, i32 ldvco2,
            f64* pcoeff, i32 ldpco1, i32 ldpco2, i32* info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 MONE = -1.0;
    const i32 int1 = 1;

    *info = 0;

    if (indblk == 0) {
        return;
    }

    i32 inplus = indblk + 1;
    i32 joff = nr;

    for (i32 l = 1; l <= indblk; l++) {
        i32 lwork = inplus - l;

        i32 ncol = nblk[lwork - 1];
        joff = joff - ncol;

        i32 lstart = joff + 1;
        i32 lstop = joff;

        for (i32 k = lwork + 1; k <= inplus; k++) {
            i32 nrow = nblk[k - 2];
            lstop = lstop + nrow;

            i32 nelem = lstop - lstart + 1;

            SLC_DGEMM("N", "N", &nrow, &ncol, &nelem, &ONE,
                      &vcoeff[(lstart - 1) * ldvco1 + (k - 1) * ldvco1 * ldvco2],
                      &ldvco1,
                      &a[joff + (lstart - 1) * lda],
                      &lda, &ZERO,
                      &pcoeff[(k - 1) * ldpco1 * ldpco2],
                      &ldpco1);
        }

        i32 nrow = ncol;

        for (i32 k = lwork; k <= indblk; k++) {
            i32 kplus = k + 1;

            for (i32 j = 1; j <= ncol; j++) {
                SLC_DSCAL(&nrow, &MONE, &pcoeff[(j - 1) * ldpco1 + (k - 1) * ldpco1 * ldpco2], &int1);
                SLC_DAXPY(&nrow, &ONE,
                          &vcoeff[(joff + j - 1) * ldvco1 + (kplus - 1) * ldvco1 * ldvco2],
                          &int1,
                          &pcoeff[(j - 1) * ldpco1 + (k - 1) * ldpco1 * ldpco2],
                          &int1);
            }

            if (k < inplus) { nrow = nblk[k - 1]; }
        }

        for (i32 j = 1; j <= ncol; j++) {
            SLC_DSCAL(&nrow, &MONE, &pcoeff[(j - 1) * ldpco1 + (inplus - 1) * ldpco1 * ldpco2], &int1);
        }

        if (lwork != 1) {
            i32 ioff = joff - nblk[lwork - 2];

            for (i32 i = 1; i <= ncol; i++) {
                if (a[(ioff + i - 1) + (joff + i - 1) * lda] == ZERO) {
                    *info = i;
                    return;
                }
            }

            nrow = nblk[lwork - 1];

            for (i32 k = lwork; k <= inplus; k++) {
                SLC_DLACPY("F", &nrow, &ncol,
                           &pcoeff[(k - 1) * ldpco1 * ldpco2],
                           &ldpco1,
                           &vcoeff[ioff * ldvco1 + (k - 1) * ldvco1 * ldvco2],
                           &ldvco1);
                SLC_DTRSM("R", "U", "N", "N",
                          &nrow, &ncol, &ONE,
                          &a[ioff + joff * lda],
                          &lda,
                          &vcoeff[ioff * ldvco1 + (k - 1) * ldvco1 * ldvco2],
                          &ldvco1);
                if (k < inplus) { nrow = nblk[k - 1]; }
            }
        }
    }
}
