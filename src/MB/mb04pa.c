/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

void mb04pa(const bool lham, const i32 n, const i32 k, const i32 nb,
            f64* a, const i32 lda, f64* qg, const i32 ldqg,
            f64* xa, const i32 ldxa, f64* xg, const i32 ldxg,
            f64* xq, const i32 ldxq, f64* ya, const i32 ldya,
            f64* cs, f64* tau, f64* dwork)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 HALF = 0.5;

    i32 i, j;
    i32 nb1, nb2;
    f64 aki, alpha, c, s, tauq, temp, ttemp;
    i32 int1 = 1;
    i32 dim_temp;

    if (n + k <= 0) {
        dwork[0] = ONE;
        return;
    }

    nb1 = nb + 1;
    nb2 = nb + nb1;

    if (lham) {
        for (i = 0; i < nb; i++) {
            i32 fi = i + 1;  // Fortran 1-based i

            alpha = qg[(k + fi) + (fi - 1) * ldqg];
            i32 min_i2n = (fi + 2 - 1 < n - 1) ? fi + 2 - 1 : n - 1;
            dim_temp = n - fi;
            SLC_DLARFG(&dim_temp, &alpha, &qg[(k + min_i2n + 1) + (fi - 1) * ldqg], &int1, &tauq);
            qg[(k + fi) + (fi - 1) * ldqg] = ONE;

            temp = -tauq * SLC_DDOT(&dim_temp, &qg[(k + fi) + (fi - 1) * ldqg], &int1,
                                     &a[(k + fi) + (fi - 1) * lda], &int1);
            SLC_DAXPY(&dim_temp, &temp, &qg[(k + fi) + (fi - 1) * ldqg], &int1,
                      &a[(k + fi) + (fi - 1) * lda], &int1);

            aki = a[(k + fi) + (fi - 1) * lda];
            SLC_DLARTG(&aki, &alpha, &c, &s, &a[(k + fi) + (fi - 1) * lda]);

            aki = a[(k + fi) + (fi - 1) * lda];
            SLC_DLARFG(&dim_temp, &aki, &a[(k + min_i2n + 1) + (fi - 1) * lda], &int1, &tau[i]);
            a[(k + fi) + (fi - 1) * lda] = ONE;

            // Update XA with first Householder reflection
            i32 nk = n + k;
            i32 im1 = fi - 1;
            SLC_DGEMV("T", &dim_temp, &dim_temp, &ONE, &a[(k + fi) + fi * lda], &lda,
                      &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &xa[fi + (fi - 1) * ldxa], &int1);
            if (im1 > 0) {
                SLC_DGEMV("T", &dim_temp, &im1, &ONE, &qg[(k + fi) + 0 * ldqg], &ldqg,
                          &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, dwork, &int1);
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &xa[fi + 0 * ldxa], &ldxa,
                          dwork, &int1, &ONE, &xa[fi + (fi - 1) * ldxa], &int1);
                SLC_DGEMV("T", &dim_temp, &im1, &ONE, &a[(k + fi) + 0 * lda], &lda,
                          &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &dwork[nb1 - 1], &int1);
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &xa[fi + (nb1 - 1) * ldxa], &ldxa,
                          &dwork[nb1 - 1], &int1, &ONE, &xa[fi + (fi - 1) * ldxa], &int1);
                SLC_DGEMV("T", &dim_temp, &im1, &ONE, &ya[(k + fi) + 0 * ldya], &ldya,
                          &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &xa[0 + (fi - 1) * ldxa], &int1);
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &qg[(k + fi) + 0 * ldqg], &ldqg,
                          &xa[0 + (fi - 1) * ldxa], &int1, &ONE, &xa[fi + (fi - 1) * ldxa], &int1);
                SLC_DGEMV("T", &dim_temp, &im1, &ONE, &ya[(k + fi) + (nb1 - 1) * ldya], &ldya,
                          &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &xa[0 + (fi - 1) * ldxa], &int1);
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &a[(k + fi) + 0 * lda], &lda,
                          &xa[0 + (fi - 1) * ldxa], &int1, &ONE, &xa[fi + (fi - 1) * ldxa], &int1);
            }
            f64 neg_tauq = -tauq;
            SLC_DSCAL(&dim_temp, &neg_tauq, &xa[fi + (fi - 1) * ldxa], &int1);

            // Update YA with first Householder reflection
            SLC_DGEMV("N", &nk, &dim_temp, &ONE, &a[0 + fi * lda], &lda,
                      &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &ya[0 + (fi - 1) * ldya], &int1);
            if (im1 > 0) {
                SLC_DGEMV("T", &dim_temp, &im1, &ONE, &xa[fi + 0 * ldxa], &ldxa,
                          &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &dwork[nb2 - 1], &int1);
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &qg[(k + fi) + 0 * ldqg], &ldqg,
                          &dwork[nb2 - 1], &int1, &ONE, &ya[(k + fi) + (fi - 1) * ldya], &int1);
                SLC_DGEMV("T", &dim_temp, &im1, &ONE, &xa[fi + (nb1 - 1) * ldxa], &ldxa,
                          &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &dwork[nb2 - 1], &int1);
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &a[(k + fi) + 0 * lda], &lda,
                          &dwork[nb2 - 1], &int1, &ONE, &ya[(k + fi) + (fi - 1) * ldya], &int1);
                SLC_DGEMV("N", &nk, &im1, &ONE, ya, &ldya,
                          dwork, &int1, &ONE, &ya[0 + (fi - 1) * ldya], &int1);
                SLC_DGEMV("N", &nk, &im1, &ONE, &ya[0 + (nb1 - 1) * ldya], &ldya,
                          &dwork[nb1 - 1], &int1, &ONE, &ya[0 + (fi - 1) * ldya], &int1);
            }
            SLC_DSCAL(&nk, &neg_tauq, &ya[0 + (fi - 1) * ldya], &int1);
            temp = -tauq * SLC_DDOT(&dim_temp, &qg[(k + fi) + (fi - 1) * ldqg], &int1,
                                    &ya[(k + fi) + (fi - 1) * ldya], &int1);
            SLC_DAXPY(&dim_temp, &temp, &qg[(k + fi) + (fi - 1) * ldqg], &int1,
                      &ya[(k + fi) + (fi - 1) * ldya], &int1);

            // Update (i+1)-th column of A
            SLC_DGEMV("N", &dim_temp, &fi, &ONE, &qg[(k + fi) + 0 * ldqg], &ldqg,
                      &xa[fi + 0 * ldxa], &ldxa, &ONE, &a[(k + fi) + fi * lda], &int1);
            if (im1 > 0) {
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &a[(k + fi) + 0 * lda], &lda,
                          &xa[fi + (nb1 - 1) * ldxa], &ldxa, &ONE, &a[(k + fi) + fi * lda], &int1);
            }
            SLC_DGEMV("N", &nk, &fi, &ONE, ya, &ldya,
                      &qg[(k + fi) + 0 * ldqg], &ldqg, &ONE, &a[0 + fi * lda], &int1);
            if (im1 > 0) {
                SLC_DGEMV("N", &nk, &im1, &ONE, &ya[0 + (nb1 - 1) * ldya], &ldya,
                          &a[(k + fi) + 0 * lda], &lda, &ONE, &a[0 + fi * lda], &int1);
            }

            // Update (i+1)-th row of A
            if (n > fi + 1) {
                i32 nmi1 = n - fi - 1;
                SLC_DGEMV("N", &nmi1, &fi, &ONE, &xa[(fi + 1) + 0 * ldxa], &ldxa,
                          &qg[(k + fi) + 0 * ldqg], &ldqg, &ONE, &a[(k + fi) + (fi + 1) * lda], &lda);
                if (im1 > 0) {
                    SLC_DGEMV("N", &nmi1, &im1, &ONE, &xa[(fi + 1) + (nb1 - 1) * ldxa], &ldxa,
                              &a[(k + fi) + 0 * lda], &lda, &ONE, &a[(k + fi) + (fi + 1) * lda], &lda);
                }
                SLC_DGEMV("N", &nmi1, &fi, &ONE, &qg[(k + fi + 1) + 0 * ldqg], &ldqg,
                          &ya[(k + fi) + 0 * ldya], &ldya, &ONE, &a[(k + fi) + (fi + 1) * lda], &lda);
                if (im1 > 0) {
                    SLC_DGEMV("N", &nmi1, &im1, &ONE, &a[(k + fi + 1) + 0 * lda], &lda,
                              &ya[(k + fi) + (nb1 - 1) * ldya], &ldya, &ONE, &a[(k + fi) + (fi + 1) * lda], &lda);
                }
            }

            // Annihilate updated parts in YA
            for (j = 0; j < fi; j++) {
                ya[(k + fi) + j * ldya] = ZERO;
            }
            for (j = 0; j < im1; j++) {
                ya[(k + fi) + (nb + j) * ldya] = ZERO;
            }

            // Update XQ with first Householder reflection
            SLC_DSYMV("L", &dim_temp, &ONE, &qg[(k + fi) + fi * ldqg], &ldqg,
                      &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &xq[fi + (fi - 1) * ldxq], &int1);
            if (im1 > 0) {
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &xq[fi + 0 * ldxq], &ldxq,
                          dwork, &int1, &ONE, &xq[fi + (fi - 1) * ldxq], &int1);
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &xq[fi + (nb1 - 1) * ldxq], &ldxq,
                          &dwork[nb1 - 1], &int1, &ONE, &xq[fi + (fi - 1) * ldxq], &int1);
                SLC_DGEMV("T", &dim_temp, &im1, &ONE, &xq[fi + 0 * ldxq], &ldxq,
                          &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &xq[0 + (fi - 1) * ldxq], &int1);
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &qg[(k + fi) + 0 * ldqg], &ldqg,
                          &xq[0 + (fi - 1) * ldxq], &int1, &ONE, &xq[fi + (fi - 1) * ldxq], &int1);
                SLC_DGEMV("T", &dim_temp, &im1, &ONE, &xq[fi + (nb1 - 1) * ldxq], &ldxq,
                          &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &xq[0 + (fi - 1) * ldxq], &int1);
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &a[(k + fi) + 0 * lda], &lda,
                          &xq[0 + (fi - 1) * ldxq], &int1, &ONE, &xq[fi + (fi - 1) * ldxq], &int1);
            }
            SLC_DSCAL(&dim_temp, &neg_tauq, &xq[fi + (fi - 1) * ldxq], &int1);
            f64 half_neg_tauq = -HALF * tauq;
            temp = half_neg_tauq * SLC_DDOT(&dim_temp, &qg[(k + fi) + (fi - 1) * ldqg], &int1,
                                            &xq[fi + (fi - 1) * ldxq], &int1);
            SLC_DAXPY(&dim_temp, &temp, &qg[(k + fi) + (fi - 1) * ldqg], &int1,
                      &xq[fi + (fi - 1) * ldxq], &int1);

            // Update (i+1)-th column and row of Q
            SLC_DGEMV("N", &dim_temp, &fi, &ONE, &qg[(k + fi) + 0 * ldqg], &ldqg,
                      &xq[fi + 0 * ldxq], &ldxq, &ONE, &qg[(k + fi) + fi * ldqg], &int1);
            if (im1 > 0) {
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &a[(k + fi) + 0 * lda], &lda,
                          &xq[fi + (nb1 - 1) * ldxq], &ldxq, &ONE, &qg[(k + fi) + fi * ldqg], &int1);
            }
            SLC_DGEMV("N", &dim_temp, &fi, &ONE, &xq[fi + 0 * ldxq], &ldxq,
                      &qg[(k + fi) + 0 * ldqg], &ldqg, &ONE, &qg[(k + fi) + fi * ldqg], &int1);
            if (im1 > 0) {
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &xq[fi + (nb1 - 1) * ldxq], &ldxq,
                          &a[(k + fi) + 0 * lda], &lda, &ONE, &qg[(k + fi) + fi * ldqg], &int1);
            }

            // Update XG with first Householder reflection
            i32 kpi = k + fi;
            SLC_DGEMV("N", &kpi, &dim_temp, &ONE, &qg[0 + (fi + 1) * ldqg], &ldqg,
                      &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &xg[0 + (fi - 1) * ldxg], &int1);
            SLC_DSYMV("U", &dim_temp, &ONE, &qg[(k + fi) + (fi + 1) * ldqg], &ldqg,
                      &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &xg[(k + fi) + (fi - 1) * ldxg], &int1);
            if (im1 > 0) {
                SLC_DGEMV("N", &nk, &im1, &ONE, xg, &ldxg,
                          dwork, &int1, &ONE, &xg[0 + (fi - 1) * ldxg], &int1);
                SLC_DGEMV("N", &nk, &im1, &ONE, &xg[0 + (nb1 - 1) * ldxg], &ldxg,
                          &dwork[nb1 - 1], &int1, &ONE, &xg[0 + (fi - 1) * ldxg], &int1);
                SLC_DGEMV("T", &dim_temp, &im1, &ONE, &xg[(k + fi) + 0 * ldxg], &ldxq,
                          &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &dwork[nb2 - 1], &int1);
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &qg[(k + fi) + 0 * ldqg], &ldqg,
                          &dwork[nb2 - 1], &int1, &ONE, &xg[(k + fi) + (fi - 1) * ldxg], &int1);
                SLC_DGEMV("T", &dim_temp, &im1, &ONE, &xg[(k + fi) + (nb1 - 1) * ldxg], &ldxq,
                          &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &dwork[nb2 - 1], &int1);
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &a[(k + fi) + 0 * lda], &lda,
                          &dwork[nb2 - 1], &int1, &ONE, &xg[(k + fi) + (fi - 1) * ldxg], &int1);
            }
            SLC_DSCAL(&nk, &neg_tauq, &xg[0 + (fi - 1) * ldxg], &int1);
            temp = half_neg_tauq * SLC_DDOT(&dim_temp, &qg[(k + fi) + (fi - 1) * ldqg], &int1,
                                            &xg[(k + fi) + (fi - 1) * ldxg], &int1);
            SLC_DAXPY(&dim_temp, &temp, &qg[(k + fi) + (fi - 1) * ldqg], &int1,
                      &xg[(k + fi) + (fi - 1) * ldxg], &int1);

            // Update (i+1)-th column and row of G
            SLC_DGEMV("N", &kpi, &fi, &ONE, xg, &ldxg,
                      &qg[(k + fi) + 0 * ldqg], &ldqg, &ONE, &qg[0 + (fi + 1) * ldqg], &int1);
            if (im1 > 0) {
                SLC_DGEMV("N", &kpi, &im1, &ONE, &xg[0 + (nb1 - 1) * ldxg], &ldxg,
                          &a[(k + fi) + 0 * lda], &lda, &ONE, &qg[0 + (fi + 1) * ldqg], &int1);
            }
            SLC_DGEMV("N", &dim_temp, &fi, &ONE, &xg[(k + fi) + 0 * ldxg], &ldxg,
                      &qg[(k + fi) + 0 * ldqg], &ldqg, &ONE, &qg[(k + fi) + (fi + 1) * ldqg], &ldqg);
            if (im1 > 0) {
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &xg[(k + fi) + (nb1 - 1) * ldxg], &ldxg,
                          &a[(k + fi) + 0 * lda], &lda, &ONE, &qg[(k + fi) + (fi + 1) * ldqg], &ldqg);
            }
            SLC_DGEMV("N", &dim_temp, &fi, &ONE, &qg[(k + fi) + 0 * ldqg], &ldqg,
                      &xg[(k + fi) + 0 * ldxg], &ldxg, &ONE, &qg[(k + fi) + (fi + 1) * ldqg], &ldqg);
            if (im1 > 0) {
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &a[(k + fi) + 0 * lda], &lda,
                          &xg[(k + fi) + (nb1 - 1) * ldxg], &ldxg, &ONE, &qg[(k + fi) + (fi + 1) * ldqg], &ldqg);
            }

            // Annihilate updated parts in XG
            for (j = 0; j < fi; j++) {
                xg[(k + fi) + j * ldxg] = ZERO;
            }
            for (j = 0; j < im1; j++) {
                xg[(k + fi) + (nb + j) * ldxg] = ZERO;
            }

            // Apply orthogonal symplectic Givens rotation
            SLC_DROT(&kpi, &a[0 + fi * lda], &int1, &qg[0 + (fi + 1) * ldqg], &int1, &c, &s);
            if (n > fi + 1) {
                i32 nmi1 = n - fi - 1;
                SLC_DROT(&nmi1, &a[(k + fi + 1) + fi * lda], &int1, &qg[(k + fi) + (fi + 2) * ldqg], &ldqg, &c, &s);
                SLC_DROT(&nmi1, &a[(k + fi) + (fi + 1) * lda], &lda, &qg[(k + fi + 1) + fi * ldqg], &int1, &c, &s);
            }
            temp = a[(k + fi) + fi * lda];
            ttemp = qg[(k + fi) + (fi + 1) * ldqg];
            a[(k + fi) + fi * lda] = c * temp + s * qg[(k + fi) + fi * ldqg];
            qg[(k + fi) + (fi + 1) * ldqg] = c * ttemp - s * temp;
            qg[(k + fi) + fi * ldqg] = -s * temp + c * qg[(k + fi) + fi * ldqg];
            ttemp = -s * ttemp - c * temp;
            temp = a[(k + fi) + fi * lda];
            qg[(k + fi) + fi * ldqg] = c * qg[(k + fi) + fi * ldqg] + s * ttemp;
            a[(k + fi) + fi * lda] = c * temp + s * qg[(k + fi) + (fi + 1) * ldqg];
            qg[(k + fi) + (fi + 1) * ldqg] = -s * temp + c * qg[(k + fi) + (fi + 1) * ldqg];
            cs[2 * i] = c;
            cs[2 * i + 1] = s;
            qg[(k + fi) + (fi - 1) * ldqg] = tauq;

            // Update XA with second Householder reflection
            SLC_DGEMV("T", &dim_temp, &dim_temp, &ONE, &a[(k + fi) + fi * lda], &lda,
                      &a[(k + fi) + (fi - 1) * lda], &int1, &ZERO, &xa[fi + (nb + fi - 1) * ldxa], &int1);
            if (n > fi + 1) {
                i32 nmi1 = n - fi - 1;
                SLC_DGEMV("T", &nmi1, &fi, &ONE, &qg[(k + fi + 1) + 0 * ldqg], &ldqg,
                          &a[(k + fi + 1) + (fi - 1) * lda], &int1, &ZERO, dwork, &int1);
                SLC_DGEMV("N", &nmi1, &fi, &ONE, &xa[(fi + 1) + 0 * ldxa], &ldxa,
                          dwork, &int1, &ONE, &xa[(fi + 1) + (nb + fi - 1) * ldxa], &int1);
                if (im1 > 0) {
                    SLC_DGEMV("T", &nmi1, &im1, &ONE, &a[(k + fi + 1) + 0 * lda], &lda,
                              &a[(k + fi + 1) + (fi - 1) * lda], &int1, &ZERO, &dwork[nb1 - 1], &int1);
                    SLC_DGEMV("N", &nmi1, &im1, &ONE, &xa[(fi + 1) + (nb1 - 1) * ldxa], &ldxa,
                              &dwork[nb1 - 1], &int1, &ONE, &xa[(fi + 1) + (nb + fi - 1) * ldxa], &int1);
                }
                SLC_DGEMV("T", &nmi1, &fi, &ONE, &ya[(k + fi + 1) + 0 * ldya], &ldya,
                          &a[(k + fi + 1) + (fi - 1) * lda], &int1, &ZERO, &xa[0 + (nb + fi - 1) * ldxa], &int1);
                SLC_DGEMV("N", &nmi1, &fi, &ONE, &qg[(k + fi + 1) + 0 * ldqg], &ldqg,
                          &xa[0 + (nb + fi - 1) * ldxa], &int1, &ONE, &xa[(fi + 1) + (nb + fi - 1) * ldxa], &int1);
                if (im1 > 0) {
                    SLC_DGEMV("T", &nmi1, &im1, &ONE, &ya[(k + fi + 1) + (nb1 - 1) * ldya], &ldya,
                              &a[(k + fi + 1) + (fi - 1) * lda], &int1, &ZERO, &xa[0 + (nb + fi - 1) * ldxa], &int1);
                    SLC_DGEMV("N", &nmi1, &im1, &ONE, &a[(k + fi + 1) + 0 * lda], &lda,
                              &xa[0 + (nb + fi - 1) * ldxa], &int1, &ONE, &xa[(fi + 1) + (nb + fi - 1) * ldxa], &int1);
                }
            }
            f64 neg_tau = -tau[i];
            SLC_DSCAL(&dim_temp, &neg_tau, &xa[fi + (nb + fi - 1) * ldxa], &int1);

            // Update YA with second Householder reflection
            SLC_DGEMV("N", &nk, &dim_temp, &ONE, &a[0 + fi * lda], &lda,
                      &a[(k + fi) + (fi - 1) * lda], &int1, &ZERO, &ya[0 + (nb + fi - 1) * ldya], &int1);
            if (n > fi + 1) {
                i32 nmi1 = n - fi - 1;
                SLC_DGEMV("T", &nmi1, &fi, &ONE, &xa[(fi + 1) + 0 * ldxa], &ldxa,
                          &a[(k + fi + 1) + (fi - 1) * lda], &int1, &ZERO, &dwork[nb2 - 1], &int1);
                SLC_DGEMV("N", &nmi1, &fi, &ONE, &qg[(k + fi + 1) + 0 * ldqg], &ldqg,
                          &dwork[nb2 - 1], &int1, &ONE, &ya[(k + fi + 1) + (nb + fi - 1) * ldya], &int1);
                if (im1 > 0) {
                    SLC_DGEMV("T", &nmi1, &im1, &ONE, &xa[(fi + 1) + (nb1 - 1) * ldxa], &ldxa,
                              &a[(k + fi + 1) + (fi - 1) * lda], &int1, &ZERO, &dwork[nb2 - 1], &int1);
                    SLC_DGEMV("N", &nmi1, &im1, &ONE, &a[(k + fi + 1) + 0 * lda], &lda,
                              &dwork[nb2 - 1], &int1, &ONE, &ya[(k + fi + 1) + (nb + fi - 1) * ldya], &int1);
                }
            }
            SLC_DGEMV("N", &nk, &fi, &ONE, ya, &ldya,
                      dwork, &int1, &ONE, &ya[0 + (nb + fi - 1) * ldya], &int1);
            if (im1 > 0) {
                SLC_DGEMV("N", &nk, &im1, &ONE, &ya[0 + (nb1 - 1) * ldya], &ldya,
                          &dwork[nb1 - 1], &int1, &ONE, &ya[0 + (nb + fi - 1) * ldya], &int1);
            }
            SLC_DSCAL(&nk, &neg_tau, &ya[0 + (nb + fi - 1) * ldya], &int1);
            temp = -tau[i] * SLC_DDOT(&dim_temp, &a[(k + fi) + (fi - 1) * lda], &int1,
                                       &ya[(k + fi) + (nb + fi - 1) * ldya], &int1);
            SLC_DAXPY(&dim_temp, &temp, &a[(k + fi) + (fi - 1) * lda], &int1,
                      &ya[(k + fi) + (nb + fi - 1) * ldya], &int1);

            // Update (i+1)-th column of A
            SLC_DAXPY(&nk, &ONE, &ya[0 + (nb + fi - 1) * ldya], &int1, &a[0 + fi * lda], &int1);
            f64 xa_val = xa[fi + (nb + fi - 1) * ldxa];
            SLC_DAXPY(&dim_temp, &xa_val, &a[(k + fi) + (fi - 1) * lda], &int1, &a[(k + fi) + fi * lda], &int1);

            // Update (i+1)-th row of A
            if (n > fi + 1) {
                i32 nmi1 = n - fi - 1;
                SLC_DAXPY(&nmi1, &ONE, &xa[(fi + 1) + (nb + fi - 1) * ldxa], &int1, &a[(k + fi) + (fi + 1) * lda], &lda);
                f64 ya_val = ya[(k + fi) + (nb + fi - 1) * ldya];
                SLC_DAXPY(&nmi1, &ya_val, &a[(k + fi + 1) + (fi - 1) * lda], &int1, &a[(k + fi) + (fi + 1) * lda], &lda);
            }

            // Annihilate updated parts in YA
            ya[(k + fi) + (nb + fi - 1) * ldya] = ZERO;

            // Update XQ with second Householder reflection
            SLC_DSYMV("L", &dim_temp, &ONE, &qg[(k + fi) + fi * ldqg], &ldqg,
                      &a[(k + fi) + (fi - 1) * lda], &int1, &ZERO, &xq[fi + (nb + fi - 1) * ldxq], &int1);
            if (n > fi + 1) {
                i32 nmi1 = n - fi - 1;
                SLC_DGEMV("N", &nmi1, &fi, &ONE, &xq[(fi + 1) + 0 * ldxq], &ldxq,
                          dwork, &int1, &ONE, &xq[(fi + 1) + (nb + fi - 1) * ldxq], &int1);
                if (im1 > 0) {
                    SLC_DGEMV("N", &nmi1, &im1, &ONE, &xq[(fi + 1) + (nb1 - 1) * ldxq], &ldxq,
                              &dwork[nb1 - 1], &int1, &ONE, &xq[(fi + 1) + (nb + fi - 1) * ldxq], &int1);
                }
                SLC_DGEMV("T", &nmi1, &fi, &ONE, &xq[(fi + 1) + 0 * ldxq], &ldxq,
                          &a[(k + fi + 1) + (fi - 1) * lda], &int1, &ZERO, &xq[0 + (nb + fi - 1) * ldxq], &int1);
                SLC_DGEMV("N", &nmi1, &fi, &ONE, &qg[(k + fi + 1) + 0 * ldqg], &ldqg,
                          &xq[0 + (nb + fi - 1) * ldxq], &int1, &ONE, &xq[(fi + 1) + (nb + fi - 1) * ldxq], &int1);
                if (im1 > 0) {
                    SLC_DGEMV("T", &nmi1, &im1, &ONE, &xq[(fi + 1) + (nb1 - 1) * ldxq], &ldxq,
                              &a[(k + fi + 1) + (fi - 1) * lda], &int1, &ZERO, &xq[0 + (nb + fi - 1) * ldxq], &int1);
                    SLC_DGEMV("N", &nmi1, &im1, &ONE, &a[(k + fi + 1) + 0 * lda], &lda,
                              &xq[0 + (nb + fi - 1) * ldxq], &int1, &ONE, &xq[(fi + 1) + (nb + fi - 1) * ldxq], &int1);
                }
            }
            SLC_DSCAL(&dim_temp, &neg_tau, &xq[fi + (nb + fi - 1) * ldxq], &int1);
            f64 half_neg_tau = -HALF * tau[i];
            temp = half_neg_tau * SLC_DDOT(&dim_temp, &a[(k + fi) + (fi - 1) * lda], &int1,
                                            &xq[fi + (nb + fi - 1) * ldxq], &int1);
            SLC_DAXPY(&dim_temp, &temp, &a[(k + fi) + (fi - 1) * lda], &int1, &xq[fi + (nb + fi - 1) * ldxq], &int1);

            // Update (i+1)-th column and row of Q
            SLC_DAXPY(&dim_temp, &ONE, &xq[fi + (nb + fi - 1) * ldxq], &int1, &qg[(k + fi) + fi * ldqg], &int1);
            f64 xq_val = xq[fi + (nb + fi - 1) * ldxq];
            SLC_DAXPY(&dim_temp, &xq_val, &a[(k + fi) + (fi - 1) * lda], &int1, &qg[(k + fi) + fi * ldqg], &int1);

            // Update XG with second Householder reflection
            SLC_DGEMV("N", &kpi, &dim_temp, &ONE, &qg[0 + (fi + 1) * ldqg], &ldqg,
                      &a[(k + fi) + (fi - 1) * lda], &int1, &ZERO, &xg[0 + (nb + fi - 1) * ldxg], &int1);
            SLC_DSYMV("U", &dim_temp, &ONE, &qg[(k + fi) + (fi + 1) * ldqg], &ldqg,
                      &a[(k + fi) + (fi - 1) * lda], &int1, &ZERO, &xg[(k + fi) + (nb + fi - 1) * ldxg], &int1);
            SLC_DGEMV("N", &nk, &fi, &ONE, xg, &ldxg,
                      dwork, &int1, &ONE, &xg[0 + (nb + fi - 1) * ldxg], &int1);
            if (im1 > 0) {
                SLC_DGEMV("N", &nk, &im1, &ONE, &xg[0 + (nb1 - 1) * ldxg], &ldxg,
                          &dwork[nb1 - 1], &int1, &ONE, &xg[0 + (nb + fi - 1) * ldxg], &int1);
            }
            if (n > fi + 1) {
                i32 nmi1 = n - fi - 1;
                SLC_DGEMV("T", &nmi1, &fi, &ONE, &xg[(k + fi + 1) + 0 * ldxg], &ldxq,
                          &a[(k + fi + 1) + (fi - 1) * lda], &int1, &ZERO, &dwork[nb2 - 1], &int1);
                SLC_DGEMV("N", &nmi1, &fi, &ONE, &qg[(k + fi + 1) + 0 * ldqg], &ldqg,
                          &dwork[nb2 - 1], &int1, &ONE, &xg[(k + fi + 1) + (nb + fi - 1) * ldxg], &int1);
                if (im1 > 0) {
                    SLC_DGEMV("T", &nmi1, &im1, &ONE, &xg[(k + fi + 1) + (nb1 - 1) * ldxg], &ldxq,
                              &a[(k + fi + 1) + (fi - 1) * lda], &int1, &ZERO, &dwork[nb2 - 1], &int1);
                    SLC_DGEMV("N", &nmi1, &im1, &ONE, &a[(k + fi + 1) + 0 * lda], &lda,
                              &dwork[nb2 - 1], &int1, &ONE, &xg[(k + fi + 1) + (nb + fi - 1) * ldxg], &int1);
                }
            }
            SLC_DSCAL(&nk, &neg_tau, &xg[0 + (nb + fi - 1) * ldxg], &int1);
            temp = half_neg_tau * SLC_DDOT(&dim_temp, &a[(k + fi) + (fi - 1) * lda], &int1,
                                           &xg[(k + fi) + (nb + fi - 1) * ldxg], &int1);
            SLC_DAXPY(&dim_temp, &temp, &a[(k + fi) + (fi - 1) * lda], &int1, &xg[(k + fi) + (nb + fi - 1) * ldxg], &int1);

            // Update (i+1)-th column and row of G
            SLC_DAXPY(&kpi, &ONE, &xg[0 + (nb + fi - 1) * ldxg], &int1, &qg[0 + (fi + 1) * ldqg], &int1);
            SLC_DAXPY(&dim_temp, &ONE, &xg[(k + fi) + (nb + fi - 1) * ldxg], &int1, &qg[(k + fi) + (fi + 1) * ldqg], &ldqg);
            f64 xg_val = xg[(k + fi) + (nb + fi - 1) * ldxg];
            SLC_DAXPY(&dim_temp, &xg_val, &a[(k + fi) + (fi - 1) * lda], &int1, &qg[(k + fi) + (fi + 1) * ldqg], &ldqg);

            // Annihilate updated parts in XG
            xg[(k + fi) + (nb + fi - 1) * ldxg] = ZERO;

            a[(k + fi) + (fi - 1) * lda] = aki;
        }
    } else {
        // Skew-Hamiltonian case (LHAM = false)
        for (i = 0; i < nb; i++) {
            i32 fi = i + 1;

            alpha = qg[(k + fi) + (fi - 1) * ldqg];
            i32 min_i2n = (fi + 2 - 1 < n - 1) ? fi + 2 - 1 : n - 1;
            dim_temp = n - fi;
            SLC_DLARFG(&dim_temp, &alpha, &qg[(k + min_i2n + 1) + (fi - 1) * ldqg], &int1, &tauq);
            qg[(k + fi) + (fi - 1) * ldqg] = ONE;

            temp = -tauq * SLC_DDOT(&dim_temp, &qg[(k + fi) + (fi - 1) * ldqg], &int1,
                                     &a[(k + fi) + (fi - 1) * lda], &int1);
            SLC_DAXPY(&dim_temp, &temp, &qg[(k + fi) + (fi - 1) * ldqg], &int1,
                      &a[(k + fi) + (fi - 1) * lda], &int1);

            aki = a[(k + fi) + (fi - 1) * lda];
            SLC_DLARTG(&aki, &alpha, &c, &s, &a[(k + fi) + (fi - 1) * lda]);

            aki = a[(k + fi) + (fi - 1) * lda];
            SLC_DLARFG(&dim_temp, &aki, &a[(k + min_i2n + 1) + (fi - 1) * lda], &int1, &tau[i]);
            a[(k + fi) + (fi - 1) * lda] = ONE;

            // Update XA with first Householder reflection
            i32 nk = n + k;
            i32 im1 = fi - 1;
            SLC_DGEMV("T", &dim_temp, &dim_temp, &ONE, &a[(k + fi) + fi * lda], &lda,
                      &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &xa[fi + (fi - 1) * ldxa], &int1);
            if (im1 > 0) {
                SLC_DGEMV("T", &dim_temp, &im1, &ONE, &qg[(k + fi) + 0 * ldqg], &ldqg,
                          &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, dwork, &int1);
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &xa[fi + 0 * ldxa], &ldxa,
                          dwork, &int1, &ONE, &xa[fi + (fi - 1) * ldxa], &int1);
                SLC_DGEMV("T", &dim_temp, &im1, &ONE, &a[(k + fi) + 0 * lda], &lda,
                          &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &dwork[nb1 - 1], &int1);
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &xa[fi + (nb1 - 1) * ldxa], &ldxa,
                          &dwork[nb1 - 1], &int1, &ONE, &xa[fi + (fi - 1) * ldxa], &int1);
                SLC_DGEMV("T", &dim_temp, &im1, &ONE, &ya[(k + fi) + 0 * ldya], &ldya,
                          &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &xa[0 + (fi - 1) * ldxa], &int1);
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &qg[(k + fi) + 0 * ldqg], &ldqg,
                          &xa[0 + (fi - 1) * ldxa], &int1, &ONE, &xa[fi + (fi - 1) * ldxa], &int1);
                SLC_DGEMV("T", &dim_temp, &im1, &ONE, &ya[(k + fi) + (nb1 - 1) * ldya], &ldya,
                          &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &xa[0 + (fi - 1) * ldxa], &int1);
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &a[(k + fi) + 0 * lda], &lda,
                          &xa[0 + (fi - 1) * ldxa], &int1, &ONE, &xa[fi + (fi - 1) * ldxa], &int1);
            }
            f64 neg_tauq = -tauq;
            SLC_DSCAL(&dim_temp, &neg_tauq, &xa[fi + (fi - 1) * ldxa], &int1);

            // Update YA with first Householder reflection
            SLC_DGEMV("N", &nk, &dim_temp, &ONE, &a[0 + fi * lda], &lda,
                      &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &ya[0 + (fi - 1) * ldya], &int1);
            if (im1 > 0) {
                SLC_DGEMV("T", &dim_temp, &im1, &ONE, &xa[fi + 0 * ldxa], &ldxa,
                          &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &dwork[nb2 - 1], &int1);
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &qg[(k + fi) + 0 * ldqg], &ldqg,
                          &dwork[nb2 - 1], &int1, &ONE, &ya[(k + fi) + (fi - 1) * ldya], &int1);
                SLC_DGEMV("T", &dim_temp, &im1, &ONE, &xa[fi + (nb1 - 1) * ldxa], &ldxa,
                          &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &dwork[nb2 - 1], &int1);
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &a[(k + fi) + 0 * lda], &lda,
                          &dwork[nb2 - 1], &int1, &ONE, &ya[(k + fi) + (fi - 1) * ldya], &int1);
                SLC_DGEMV("N", &nk, &im1, &ONE, ya, &ldya,
                          dwork, &int1, &ONE, &ya[0 + (fi - 1) * ldya], &int1);
                SLC_DGEMV("N", &nk, &im1, &ONE, &ya[0 + (nb1 - 1) * ldya], &ldya,
                          &dwork[nb1 - 1], &int1, &ONE, &ya[0 + (fi - 1) * ldya], &int1);
            }
            SLC_DSCAL(&nk, &neg_tauq, &ya[0 + (fi - 1) * ldya], &int1);
            temp = -tauq * SLC_DDOT(&dim_temp, &qg[(k + fi) + (fi - 1) * ldqg], &int1,
                                    &ya[(k + fi) + (fi - 1) * ldya], &int1);
            SLC_DAXPY(&dim_temp, &temp, &qg[(k + fi) + (fi - 1) * ldqg], &int1,
                      &ya[(k + fi) + (fi - 1) * ldya], &int1);

            // Update (i+1)-th column of A
            SLC_DGEMV("N", &dim_temp, &fi, &ONE, &qg[(k + fi) + 0 * ldqg], &ldqg,
                      &xa[fi + 0 * ldxa], &ldxa, &ONE, &a[(k + fi) + fi * lda], &int1);
            if (im1 > 0) {
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &a[(k + fi) + 0 * lda], &lda,
                          &xa[fi + (nb1 - 1) * ldxa], &ldxa, &ONE, &a[(k + fi) + fi * lda], &int1);
            }
            SLC_DGEMV("N", &nk, &fi, &ONE, ya, &ldya,
                      &qg[(k + fi) + 0 * ldqg], &ldqg, &ONE, &a[0 + fi * lda], &int1);
            if (im1 > 0) {
                SLC_DGEMV("N", &nk, &im1, &ONE, &ya[0 + (nb1 - 1) * ldya], &ldya,
                          &a[(k + fi) + 0 * lda], &lda, &ONE, &a[0 + fi * lda], &int1);
            }

            // Update (i+1)-th row of A
            if (n > fi + 1) {
                i32 nmi1 = n - fi - 1;
                SLC_DGEMV("N", &nmi1, &fi, &ONE, &xa[(fi + 1) + 0 * ldxa], &ldxa,
                          &qg[(k + fi) + 0 * ldqg], &ldqg, &ONE, &a[(k + fi) + (fi + 1) * lda], &lda);
                if (im1 > 0) {
                    SLC_DGEMV("N", &nmi1, &im1, &ONE, &xa[(fi + 1) + (nb1 - 1) * ldxa], &ldxa,
                              &a[(k + fi) + 0 * lda], &lda, &ONE, &a[(k + fi) + (fi + 1) * lda], &lda);
                }
                SLC_DGEMV("N", &nmi1, &fi, &ONE, &qg[(k + fi + 1) + 0 * ldqg], &ldqg,
                          &ya[(k + fi) + 0 * ldya], &ldya, &ONE, &a[(k + fi) + (fi + 1) * lda], &lda);
                if (im1 > 0) {
                    SLC_DGEMV("N", &nmi1, &im1, &ONE, &a[(k + fi + 1) + 0 * lda], &lda,
                              &ya[(k + fi) + (nb1 - 1) * ldya], &ldya, &ONE, &a[(k + fi) + (fi + 1) * lda], &lda);
                }
            }

            // Annihilate updated parts in YA
            for (j = 0; j < fi; j++) {
                ya[(k + fi) + j * ldya] = ZERO;
            }
            for (j = 0; j < im1; j++) {
                ya[(k + fi) + (nb + j) * ldya] = ZERO;
            }

            // Update XQ with first Householder reflection (skew case uses mb01md)
            i32 info_mb01md;
            mb01md('L', dim_temp, ONE, &qg[(k + fi) + fi * ldqg], ldqg,
                   &qg[(k + fi) + (fi - 1) * ldqg], 1, ZERO, &xq[fi + (fi - 1) * ldxq], 1, &info_mb01md);
            if (im1 > 0) {
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &xq[fi + 0 * ldxq], &ldxq,
                          dwork, &int1, &ONE, &xq[fi + (fi - 1) * ldxq], &int1);
                SLC_DGEMV("N", &dim_temp, &im1, &ONE, &xq[fi + (nb1 - 1) * ldxq], &ldxq,
                          &dwork[nb1 - 1], &int1, &ONE, &xq[fi + (fi - 1) * ldxq], &int1);
                SLC_DGEMV("T", &dim_temp, &im1, &ONE, &xq[fi + 0 * ldxq], &ldxq,
                          &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &xq[0 + (fi - 1) * ldxq], &int1);
                f64 neg_one = -ONE;
                SLC_DGEMV("N", &dim_temp, &im1, &neg_one, &qg[(k + fi) + 0 * ldqg], &ldqg,
                          &xq[0 + (fi - 1) * ldxq], &int1, &ONE, &xq[fi + (fi - 1) * ldxq], &int1);
                SLC_DGEMV("T", &dim_temp, &im1, &ONE, &xq[fi + (nb1 - 1) * ldxq], &ldxq,
                          &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &xq[0 + (fi - 1) * ldxq], &int1);
                SLC_DGEMV("N", &dim_temp, &im1, &neg_one, &a[(k + fi) + 0 * lda], &lda,
                          &xq[0 + (fi - 1) * ldxq], &int1, &ONE, &xq[fi + (fi - 1) * ldxq], &int1);
            }
            SLC_DSCAL(&dim_temp, &neg_tauq, &xq[fi + (fi - 1) * ldxq], &int1);
            f64 half_neg_tauq = -HALF * tauq;
            temp = half_neg_tauq * SLC_DDOT(&dim_temp, &qg[(k + fi) + (fi - 1) * ldqg], &int1,
                                            &xq[fi + (fi - 1) * ldxq], &int1);
            SLC_DAXPY(&dim_temp, &temp, &qg[(k + fi) + (fi - 1) * ldqg], &int1,
                      &xq[fi + (fi - 1) * ldxq], &int1);

            // Update (i+1)-th column and row of Q
            if (n > fi + 1) {
                i32 nmi1 = n - fi - 1;
                f64 neg_one = -ONE;
                SLC_DGEMV("N", &nmi1, &fi, &neg_one, &qg[(k + fi + 1) + 0 * ldqg], &ldqg,
                          &xq[fi + 0 * ldxq], &ldxq, &ONE, &qg[(k + fi + 1) + fi * ldqg], &int1);
                if (im1 > 0) {
                    SLC_DGEMV("N", &nmi1, &im1, &neg_one, &a[(k + fi + 1) + 0 * lda], &lda,
                              &xq[fi + (nb1 - 1) * ldxq], &ldxq, &ONE, &qg[(k + fi + 1) + fi * ldqg], &int1);
                }
                SLC_DGEMV("N", &nmi1, &fi, &ONE, &xq[(fi + 1) + 0 * ldxq], &ldxq,
                          &qg[(k + fi) + 0 * ldqg], &ldqg, &ONE, &qg[(k + fi + 1) + fi * ldqg], &int1);
                if (im1 > 0) {
                    SLC_DGEMV("N", &nmi1, &im1, &ONE, &xq[(fi + 1) + (nb1 - 1) * ldxq], &ldxq,
                              &a[(k + fi) + 0 * lda], &lda, &ONE, &qg[(k + fi + 1) + fi * ldqg], &int1);
                }
            }

            // Update XG with first Householder reflection (skew case uses mb01md)
            i32 kpi = k + fi;
            SLC_DGEMV("N", &kpi, &dim_temp, &ONE, &qg[0 + (fi + 1) * ldqg], &ldqg,
                      &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &xg[0 + (fi - 1) * ldxg], &int1);
            mb01md('U', dim_temp, ONE, &qg[(k + fi) + (fi + 1) * ldqg], ldqg,
                   &qg[(k + fi) + (fi - 1) * ldqg], 1, ZERO, &xg[(k + fi) + (fi - 1) * ldxg], 1, &info_mb01md);
            if (im1 > 0) {
                SLC_DGEMV("N", &nk, &im1, &ONE, xg, &ldxg,
                          dwork, &int1, &ONE, &xg[0 + (fi - 1) * ldxg], &int1);
                SLC_DGEMV("N", &nk, &im1, &ONE, &xg[0 + (nb1 - 1) * ldxg], &ldxg,
                          &dwork[nb1 - 1], &int1, &ONE, &xg[0 + (fi - 1) * ldxg], &int1);
                SLC_DGEMV("T", &dim_temp, &im1, &ONE, &xg[(k + fi) + 0 * ldxg], &ldxq,
                          &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &dwork[nb2 - 1], &int1);
                f64 neg_one = -ONE;
                SLC_DGEMV("N", &dim_temp, &im1, &neg_one, &qg[(k + fi) + 0 * ldqg], &ldqg,
                          &dwork[nb2 - 1], &int1, &ONE, &xg[(k + fi) + (fi - 1) * ldxg], &int1);
                SLC_DGEMV("T", &dim_temp, &im1, &ONE, &xg[(k + fi) + (nb1 - 1) * ldxg], &ldxq,
                          &qg[(k + fi) + (fi - 1) * ldqg], &int1, &ZERO, &dwork[nb2 - 1], &int1);
                SLC_DGEMV("N", &dim_temp, &im1, &neg_one, &a[(k + fi) + 0 * lda], &lda,
                          &dwork[nb2 - 1], &int1, &ONE, &xg[(k + fi) + (fi - 1) * ldxg], &int1);
            }
            SLC_DSCAL(&nk, &neg_tauq, &xg[0 + (fi - 1) * ldxg], &int1);
            temp = half_neg_tauq * SLC_DDOT(&dim_temp, &qg[(k + fi) + (fi - 1) * ldqg], &int1,
                                            &xg[(k + fi) + (fi - 1) * ldxg], &int1);
            SLC_DAXPY(&dim_temp, &temp, &qg[(k + fi) + (fi - 1) * ldqg], &int1,
                      &xg[(k + fi) + (fi - 1) * ldxg], &int1);

            // Update (i+1)-th column and row of G
            SLC_DGEMV("N", &kpi, &fi, &ONE, xg, &ldxg,
                      &qg[(k + fi) + 0 * ldqg], &ldqg, &ONE, &qg[0 + (fi + 1) * ldqg], &int1);
            if (im1 > 0) {
                SLC_DGEMV("N", &kpi, &im1, &ONE, &xg[0 + (nb1 - 1) * ldxg], &ldxg,
                          &a[(k + fi) + 0 * lda], &lda, &ONE, &qg[0 + (fi + 1) * ldqg], &int1);
            }
            if (n > fi + 1) {
                i32 nmi1 = n - fi - 1;
                f64 neg_one = -ONE;
                SLC_DGEMV("N", &nmi1, &fi, &neg_one, &xg[(k + fi + 1) + 0 * ldxg], &ldxg,
                          &qg[(k + fi) + 0 * ldqg], &ldqg, &ONE, &qg[(k + fi) + (fi + 2) * ldqg], &ldqg);
                if (im1 > 0) {
                    SLC_DGEMV("N", &nmi1, &im1, &neg_one, &xg[(k + fi + 1) + (nb1 - 1) * ldxg], &ldxg,
                              &a[(k + fi) + 0 * lda], &lda, &ONE, &qg[(k + fi) + (fi + 2) * ldqg], &ldqg);
                }
                SLC_DGEMV("N", &nmi1, &fi, &ONE, &qg[(k + fi + 1) + 0 * ldqg], &ldqg,
                          &xg[(k + fi) + 0 * ldxg], &ldxg, &ONE, &qg[(k + fi) + (fi + 2) * ldqg], &ldqg);
                if (im1 > 0) {
                    SLC_DGEMV("N", &nmi1, &im1, &ONE, &a[(k + fi + 1) + 0 * lda], &lda,
                              &xg[(k + fi) + (nb1 - 1) * ldxg], &ldxg, &ONE, &qg[(k + fi) + (fi + 2) * ldqg], &ldqg);
                }
            }

            // Annihilate updated parts in XG
            for (j = 0; j < fi; j++) {
                xg[(k + fi) + j * ldxg] = ZERO;
            }
            for (j = 0; j < im1; j++) {
                xg[(k + fi) + (nb + j) * ldxg] = ZERO;
            }

            // Apply orthogonal symplectic Givens rotation (skew case has different sign)
            SLC_DROT(&kpi, &a[0 + fi * lda], &int1, &qg[0 + (fi + 1) * ldqg], &int1, &c, &s);
            if (n > fi + 1) {
                i32 nmi1 = n - fi - 1;
                f64 neg_s = -s;
                SLC_DROT(&nmi1, &a[(k + fi + 1) + fi * lda], &int1, &qg[(k + fi) + (fi + 2) * ldqg], &ldqg, &c, &neg_s);
                SLC_DROT(&nmi1, &a[(k + fi) + (fi + 1) * lda], &lda, &qg[(k + fi + 1) + fi * ldqg], &int1, &c, &neg_s);
            }
            cs[2 * i] = c;
            cs[2 * i + 1] = s;
            qg[(k + fi) + (fi - 1) * ldqg] = tauq;

            // Update XA with second Householder reflection
            SLC_DGEMV("T", &dim_temp, &dim_temp, &ONE, &a[(k + fi) + fi * lda], &lda,
                      &a[(k + fi) + (fi - 1) * lda], &int1, &ZERO, &xa[fi + (nb + fi - 1) * ldxa], &int1);
            if (n > fi + 1) {
                i32 nmi1 = n - fi - 1;
                SLC_DGEMV("T", &nmi1, &fi, &ONE, &qg[(k + fi + 1) + 0 * ldqg], &ldqg,
                          &a[(k + fi + 1) + (fi - 1) * lda], &int1, &ZERO, dwork, &int1);
                SLC_DGEMV("N", &nmi1, &fi, &ONE, &xa[(fi + 1) + 0 * ldxa], &ldxa,
                          dwork, &int1, &ONE, &xa[(fi + 1) + (nb + fi - 1) * ldxa], &int1);
                if (im1 > 0) {
                    SLC_DGEMV("T", &nmi1, &im1, &ONE, &a[(k + fi + 1) + 0 * lda], &lda,
                              &a[(k + fi + 1) + (fi - 1) * lda], &int1, &ZERO, &dwork[nb1 - 1], &int1);
                    SLC_DGEMV("N", &nmi1, &im1, &ONE, &xa[(fi + 1) + (nb1 - 1) * ldxa], &ldxa,
                              &dwork[nb1 - 1], &int1, &ONE, &xa[(fi + 1) + (nb + fi - 1) * ldxa], &int1);
                }
                SLC_DGEMV("T", &nmi1, &fi, &ONE, &ya[(k + fi + 1) + 0 * ldya], &ldya,
                          &a[(k + fi + 1) + (fi - 1) * lda], &int1, &ZERO, &xa[0 + (nb + fi - 1) * ldxa], &int1);
                SLC_DGEMV("N", &nmi1, &fi, &ONE, &qg[(k + fi + 1) + 0 * ldqg], &ldqg,
                          &xa[0 + (nb + fi - 1) * ldxa], &int1, &ONE, &xa[(fi + 1) + (nb + fi - 1) * ldxa], &int1);
                if (im1 > 0) {
                    SLC_DGEMV("T", &nmi1, &im1, &ONE, &ya[(k + fi + 1) + (nb1 - 1) * ldya], &ldya,
                              &a[(k + fi + 1) + (fi - 1) * lda], &int1, &ZERO, &xa[0 + (nb + fi - 1) * ldxa], &int1);
                    SLC_DGEMV("N", &nmi1, &im1, &ONE, &a[(k + fi + 1) + 0 * lda], &lda,
                              &xa[0 + (nb + fi - 1) * ldxa], &int1, &ONE, &xa[(fi + 1) + (nb + fi - 1) * ldxa], &int1);
                }
            }
            f64 neg_tau = -tau[i];
            SLC_DSCAL(&dim_temp, &neg_tau, &xa[fi + (nb + fi - 1) * ldxa], &int1);

            // Update YA with second Householder reflection
            SLC_DGEMV("N", &nk, &dim_temp, &ONE, &a[0 + fi * lda], &lda,
                      &a[(k + fi) + (fi - 1) * lda], &int1, &ZERO, &ya[0 + (nb + fi - 1) * ldya], &int1);
            if (n > fi + 1) {
                i32 nmi1 = n - fi - 1;
                SLC_DGEMV("T", &nmi1, &fi, &ONE, &xa[(fi + 1) + 0 * ldxa], &ldxa,
                          &a[(k + fi + 1) + (fi - 1) * lda], &int1, &ZERO, &dwork[nb2 - 1], &int1);
                SLC_DGEMV("N", &nmi1, &fi, &ONE, &qg[(k + fi + 1) + 0 * ldqg], &ldqg,
                          &dwork[nb2 - 1], &int1, &ONE, &ya[(k + fi + 1) + (nb + fi - 1) * ldya], &int1);
                if (im1 > 0) {
                    SLC_DGEMV("T", &nmi1, &im1, &ONE, &xa[(fi + 1) + (nb1 - 1) * ldxa], &ldxa,
                              &a[(k + fi + 1) + (fi - 1) * lda], &int1, &ZERO, &dwork[nb2 - 1], &int1);
                    SLC_DGEMV("N", &nmi1, &im1, &ONE, &a[(k + fi + 1) + 0 * lda], &lda,
                              &dwork[nb2 - 1], &int1, &ONE, &ya[(k + fi + 1) + (nb + fi - 1) * ldya], &int1);
                }
            }
            SLC_DGEMV("N", &nk, &fi, &ONE, ya, &ldya,
                      dwork, &int1, &ONE, &ya[0 + (nb + fi - 1) * ldya], &int1);
            if (im1 > 0) {
                SLC_DGEMV("N", &nk, &im1, &ONE, &ya[0 + (nb1 - 1) * ldya], &ldya,
                          &dwork[nb1 - 1], &int1, &ONE, &ya[0 + (nb + fi - 1) * ldya], &int1);
            }
            SLC_DSCAL(&nk, &neg_tau, &ya[0 + (nb + fi - 1) * ldya], &int1);
            temp = -tau[i] * SLC_DDOT(&dim_temp, &a[(k + fi) + (fi - 1) * lda], &int1,
                                       &ya[(k + fi) + (nb + fi - 1) * ldya], &int1);
            SLC_DAXPY(&dim_temp, &temp, &a[(k + fi) + (fi - 1) * lda], &int1,
                      &ya[(k + fi) + (nb + fi - 1) * ldya], &int1);

            // Update (i+1)-th column of A
            SLC_DAXPY(&nk, &ONE, &ya[0 + (nb + fi - 1) * ldya], &int1, &a[0 + fi * lda], &int1);
            f64 xa_val = xa[fi + (nb + fi - 1) * ldxa];
            SLC_DAXPY(&dim_temp, &xa_val, &a[(k + fi) + (fi - 1) * lda], &int1, &a[(k + fi) + fi * lda], &int1);

            // Update (i+1)-th row of A
            if (n > fi + 1) {
                i32 nmi1 = n - fi - 1;
                SLC_DAXPY(&nmi1, &ONE, &xa[(fi + 1) + (nb + fi - 1) * ldxa], &int1, &a[(k + fi) + (fi + 1) * lda], &lda);
                f64 ya_val = ya[(k + fi) + (nb + fi - 1) * ldya];
                SLC_DAXPY(&nmi1, &ya_val, &a[(k + fi + 1) + (fi - 1) * lda], &int1, &a[(k + fi) + (fi + 1) * lda], &lda);
            }

            // Annihilate updated parts in YA
            ya[(k + fi) + (nb + fi - 1) * ldya] = ZERO;

            // Update XQ with second Householder reflection (skew case)
            mb01md('L', dim_temp, ONE, &qg[(k + fi) + fi * ldqg], ldqg,
                   &a[(k + fi) + (fi - 1) * lda], 1, ZERO, &xq[fi + (nb + fi - 1) * ldxq], 1, &info_mb01md);
            if (n > fi + 1) {
                i32 nmi1 = n - fi - 1;
                SLC_DGEMV("N", &nmi1, &fi, &ONE, &xq[(fi + 1) + 0 * ldxq], &ldxq,
                          dwork, &int1, &ONE, &xq[(fi + 1) + (nb + fi - 1) * ldxq], &int1);
                if (im1 > 0) {
                    SLC_DGEMV("N", &nmi1, &im1, &ONE, &xq[(fi + 1) + (nb1 - 1) * ldxq], &ldxq,
                              &dwork[nb1 - 1], &int1, &ONE, &xq[(fi + 1) + (nb + fi - 1) * ldxq], &int1);
                }
                SLC_DGEMV("T", &nmi1, &fi, &ONE, &xq[(fi + 1) + 0 * ldxq], &ldxq,
                          &a[(k + fi + 1) + (fi - 1) * lda], &int1, &ZERO, &xq[0 + (nb + fi - 1) * ldxq], &int1);
                f64 neg_one = -ONE;
                SLC_DGEMV("N", &nmi1, &fi, &neg_one, &qg[(k + fi + 1) + 0 * ldqg], &ldqg,
                          &xq[0 + (nb + fi - 1) * ldxq], &int1, &ONE, &xq[(fi + 1) + (nb + fi - 1) * ldxq], &int1);
                if (im1 > 0) {
                    SLC_DGEMV("T", &nmi1, &im1, &ONE, &xq[(fi + 1) + (nb1 - 1) * ldxq], &ldxq,
                              &a[(k + fi + 1) + (fi - 1) * lda], &int1, &ZERO, &xq[0 + (nb + fi - 1) * ldxq], &int1);
                    SLC_DGEMV("N", &nmi1, &im1, &neg_one, &a[(k + fi + 1) + 0 * lda], &lda,
                              &xq[0 + (nb + fi - 1) * ldxq], &int1, &ONE, &xq[(fi + 1) + (nb + fi - 1) * ldxq], &int1);
                }
            }
            SLC_DSCAL(&dim_temp, &neg_tau, &xq[fi + (nb + fi - 1) * ldxq], &int1);
            f64 half_neg_tau = -HALF * tau[i];
            temp = half_neg_tau * SLC_DDOT(&dim_temp, &a[(k + fi) + (fi - 1) * lda], &int1,
                                            &xq[fi + (nb + fi - 1) * ldxq], &int1);
            SLC_DAXPY(&dim_temp, &temp, &a[(k + fi) + (fi - 1) * lda], &int1, &xq[fi + (nb + fi - 1) * ldxq], &int1);

            // Update (i+1)-th column and row of Q
            if (n > fi + 1) {
                i32 nmi1 = n - fi - 1;
                SLC_DAXPY(&nmi1, &ONE, &xq[(fi + 1) + (nb + fi - 1) * ldxq], &int1, &qg[(k + fi + 1) + fi * ldqg], &int1);
                f64 neg_xq_val = -xq[fi + (nb + fi - 1) * ldxq];
                SLC_DAXPY(&nmi1, &neg_xq_val, &a[(k + fi + 1) + (fi - 1) * lda], &int1, &qg[(k + fi + 1) + fi * ldqg], &int1);
            }

            // Update XG with second Householder reflection (skew case)
            SLC_DGEMV("N", &kpi, &dim_temp, &ONE, &qg[0 + (fi + 1) * ldqg], &ldqg,
                      &a[(k + fi) + (fi - 1) * lda], &int1, &ZERO, &xg[0 + (nb + fi - 1) * ldxg], &int1);
            mb01md('U', dim_temp, ONE, &qg[(k + fi) + (fi + 1) * ldqg], ldqg,
                   &a[(k + fi) + (fi - 1) * lda], 1, ZERO, &xg[(k + fi) + (nb + fi - 1) * ldxg], 1, &info_mb01md);
            SLC_DGEMV("N", &nk, &fi, &ONE, xg, &ldxg,
                      dwork, &int1, &ONE, &xg[0 + (nb + fi - 1) * ldxg], &int1);
            if (im1 > 0) {
                SLC_DGEMV("N", &nk, &im1, &ONE, &xg[0 + (nb1 - 1) * ldxg], &ldxg,
                          &dwork[nb1 - 1], &int1, &ONE, &xg[0 + (nb + fi - 1) * ldxg], &int1);
            }
            if (n > fi + 1) {
                i32 nmi1 = n - fi - 1;
                SLC_DGEMV("T", &nmi1, &fi, &ONE, &xg[(k + fi + 1) + 0 * ldxg], &ldxq,
                          &a[(k + fi + 1) + (fi - 1) * lda], &int1, &ZERO, &dwork[nb2 - 1], &int1);
                f64 neg_one = -ONE;
                SLC_DGEMV("N", &nmi1, &fi, &neg_one, &qg[(k + fi + 1) + 0 * ldqg], &ldqg,
                          &dwork[nb2 - 1], &int1, &ONE, &xg[(k + fi + 1) + (nb + fi - 1) * ldxg], &int1);
                if (im1 > 0) {
                    SLC_DGEMV("T", &nmi1, &im1, &ONE, &xg[(k + fi + 1) + (nb1 - 1) * ldxg], &ldxq,
                              &a[(k + fi + 1) + (fi - 1) * lda], &int1, &ZERO, &dwork[nb2 - 1], &int1);
                    SLC_DGEMV("N", &nmi1, &im1, &neg_one, &a[(k + fi + 1) + 0 * lda], &lda,
                              &dwork[nb2 - 1], &int1, &ONE, &xg[(k + fi + 1) + (nb + fi - 1) * ldxg], &int1);
                }
            }
            SLC_DSCAL(&nk, &neg_tau, &xg[0 + (nb + fi - 1) * ldxg], &int1);
            temp = half_neg_tau * SLC_DDOT(&dim_temp, &a[(k + fi) + (fi - 1) * lda], &int1,
                                           &xg[(k + fi) + (nb + fi - 1) * ldxg], &int1);
            SLC_DAXPY(&dim_temp, &temp, &a[(k + fi) + (fi - 1) * lda], &int1, &xg[(k + fi) + (nb + fi - 1) * ldxg], &int1);

            // Update (i+1)-th column and row of G
            SLC_DAXPY(&kpi, &ONE, &xg[0 + (nb + fi - 1) * ldxg], &int1, &qg[0 + (fi + 1) * ldqg], &int1);
            if (n > fi + 1) {
                i32 nmi1 = n - fi - 1;
                f64 neg_one = -ONE;
                SLC_DAXPY(&nmi1, &neg_one, &xg[(k + fi + 1) + (nb + fi - 1) * ldxg], &int1, &qg[(k + fi) + (fi + 2) * ldqg], &ldqg);
                f64 xg_val = xg[(k + fi) + (nb + fi - 1) * ldxg];
                SLC_DAXPY(&nmi1, &xg_val, &a[(k + fi + 1) + (fi - 1) * lda], &int1, &qg[(k + fi) + (fi + 2) * ldqg], &ldqg);
            }

            // Annihilate updated parts in XG
            xg[(k + fi) + (nb + fi - 1) * ldxg] = ZERO;

            a[(k + fi) + (fi - 1) * lda] = aki;
        }
    }
}
