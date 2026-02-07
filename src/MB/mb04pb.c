/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * MB04PB - Reduce Hamiltonian matrix to PVL form (blocked version)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

void mb04pb(i32 n, i32 ilo,
            f64 *a, i32 lda,
            f64 *qg, i32 ldqg,
            f64 *cs, f64 *tau,
            f64 *dwork, i32 ldwork,
            i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool lquery;
    i32 i, ib, ierr, minwrk, nb, nbmin, nh, nib, nnb;
    i32 nx, pdw, pxa, pxg, pxq, pya, wrkopt;

    *info = 0;
    i32 max_1_n = (1 > n) ? 1 : n;
    minwrk = (1 > n - 1) ? 1 : n - 1;

    nb = 0;
    wrkopt = 1;

    if (n < 0) {
        *info = -1;
    } else if (ilo < 1 || ilo > max_1_n) {
        *info = -2;
    } else if (lda < max_1_n) {
        *info = -4;
    } else if (ldqg < max_1_n) {
        *info = -6;
    } else {
        lquery = (ldwork == -1);
        if (ldwork < minwrk && !lquery) {
            dwork[0] = (f64)minwrk;
            *info = -10;
        } else {
            if (n == 0) {
                wrkopt = 1;
            } else {
                i32 gehrd_info = 0;
                f64 gehrd_work;
                i32 neg1 = -1;
                SLC_DGEHRD(&n, &ilo, &n, dwork, &n, dwork, &gehrd_work, &neg1, &gehrd_info);
                wrkopt = minwrk > (i32)gehrd_work ? minwrk : (i32)gehrd_work;
                nb = wrkopt / n;
                i32 blocked_work = 8 * n * nb + 3 * nb;
                wrkopt = wrkopt > blocked_work ? wrkopt : blocked_work;
            }
            if (lquery) {
                dwork[0] = (f64)wrkopt;
                return;
            }
        }
    }

    if (*info != 0) {
        return;
    }

    for (i = 0; i < ilo - 1; i++) {
        tau[i] = ZERO;
        cs[2 * i] = ONE;
        cs[2 * i + 1] = ZERO;
    }

    if (n <= ilo) {
        dwork[0] = ONE;
        return;
    }

    nh = n - ilo + 1;
    nbmin = 2;
    nx = nb;
    if (nb > 1 && nb < nh) {
        i32 nx_ue = ue01md(3, "MB04PB", " ", n, ilo, -1);
        nx = nb > nx_ue ? nb : nx_ue;
        if (nx < nh) {
            if (ldwork < wrkopt) {
                i32 nbmin_ue = ue01md(2, "MB04PB", " ", n, ilo, -1);
                nbmin = 2 > nbmin_ue ? 2 : nbmin_ue;
                nb = ldwork / (8 * n + 3);
            }
        }
    }

    nnb = n * nb;
    pxa = 0;
    pya = pxa + 2 * nnb;
    pxq = pya + 2 * nnb;
    pxg = pxq + 2 * nnb;
    pdw = pxg + 2 * nnb;

    if (nb < nbmin || nb >= nh) {
        i = ilo;
    } else {
        /* i is 1-based (same value as Fortran I) throughout blocked loop */
        for (i = ilo; i <= n - nx - 1; i += nb) {
            ib = nb < (n - i + 1) ? nb : (n - i + 1);
            nib = n * ib;

            mb04pa(true, n - i + 1, i - 1, ib,
                   &a[(i - 1) * lda], lda,
                   &qg[(i - 1) * ldqg], ldqg,
                   &dwork[pxa], n,
                   &dwork[pxg], n,
                   &dwork[pxq], n,
                   &dwork[pya], n,
                   &cs[2 * (i - 1)],
                   &tau[i - 1],
                   &dwork[pdw]);

            if (n > i + ib) {
                i32 niib = n - i - ib;
                i32 iib = i + ib;

                SLC_DGEMM("N", "T", &niib, &niib, &ib, &ONE,
                          &qg[iib + (i - 1) * ldqg], &ldqg,
                          &dwork[pxa + ib + 1], &n,
                          &ONE, &a[iib + iib * lda], &lda);

                SLC_DGEMM("N", "T", &niib, &niib, &ib, &ONE,
                          &a[iib + (i - 1) * lda], &lda,
                          &dwork[pxa + nib + ib + 1], &n,
                          &ONE, &a[iib + iib * lda], &lda);

                SLC_DGEMM("N", "T", &n, &niib, &ib, &ONE,
                          &dwork[pya], &n,
                          &qg[iib + (i - 1) * ldqg], &ldqg,
                          &ONE, &a[iib * lda], &lda);

                SLC_DGEMM("N", "T", &n, &niib, &ib, &ONE,
                          &dwork[pya + nib], &n,
                          &a[iib + (i - 1) * lda], &lda,
                          &ONE, &a[iib * lda], &lda);

                SLC_DSYR2K("L", "N", &niib, &ib, &ONE,
                           &dwork[pxq + ib + 1], &n,
                           &qg[iib + (i - 1) * ldqg], &ldqg,
                           &ONE, &qg[iib + iib * ldqg], &ldqg);

                SLC_DSYR2K("L", "N", &niib, &ib, &ONE,
                           &dwork[pxq + nib + ib + 1], &n,
                           &a[iib + (i - 1) * lda], &lda,
                           &ONE, &qg[iib + iib * ldqg], &ldqg);

                SLC_DGEMM("N", "T", &iib, &niib, &ib, &ONE,
                          &dwork[pxg], &n,
                          &qg[iib + (i - 1) * ldqg], &ldqg,
                          &ONE, &qg[(iib + 1) * ldqg], &ldqg);

                SLC_DGEMM("N", "T", &iib, &niib, &ib, &ONE,
                          &dwork[pxg + nib], &n,
                          &a[iib + (i - 1) * lda], &lda,
                          &ONE, &qg[(iib + 1) * ldqg], &ldqg);

                SLC_DSYR2K("U", "N", &niib, &ib, &ONE,
                           &dwork[pxg + ib + i], &n,
                           &qg[iib + (i - 1) * ldqg], &ldqg,
                           &ONE, &qg[iib + (iib + 1) * ldqg], &ldqg);

                SLC_DSYR2K("U", "N", &niib, &ib, &ONE,
                           &dwork[pxg + nib + ib + i], &n,
                           &a[iib + (i - 1) * lda], &lda,
                           &ONE, &qg[iib + (iib + 1) * ldqg], &ldqg);
            }
        }
    }

    mb04pu(n, i, a, lda, qg, ldqg, cs, tau, dwork, ldwork, &ierr);

    dwork[0] = (f64)wrkopt;
}
