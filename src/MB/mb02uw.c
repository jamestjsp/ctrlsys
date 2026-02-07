/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB02UW - Solve 1x1 or 2x2 linear system with scaling and perturbation
 *
 * Solves A*X = s*B or A'*X = s*B where N=1 or 2, with automatic scaling
 * to prevent overflow and perturbation of near-singular matrices.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb02uw(const bool ltrans, const i32 n, const i32 m, const f64* par,
            const f64* a, const i32 lda, f64* b, const i32 ldb,
            f64* scale, i32* iwarn)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    static const bool zswap[4] = {false, false, true, true};
    static const bool rswap[4] = {false, true, false, true};
    static const i32 ipivot[4][4] = {
        {1, 2, 3, 4},
        {2, 1, 4, 3},
        {3, 4, 1, 2},
        {4, 3, 2, 1}
    };

    f64 c[2][2], cv[4];
    i32 i, j, icmax;
    f64 bbnd, bignum, bnorm, b1, b2, cmax, c21, c22;
    f64 cs, eps, l21, scalep, smini, smlnum, smin, temp, u11;
    f64 u11r, u12, u22, xnorm, x1, x2;

    *iwarn = 0;

    smin = par[2];
    eps = par[0];
    smlnum = TWO * par[1] / eps;
    bignum = ONE / smlnum;
    smini = (smin > smlnum) ? smin : smlnum;

    *scale = ONE;

    if (n == 1) {
        cs = a[0];
        cmax = fabs(cs);

        if (cmax < smini) {
            cs = smini;
            cmax = smini;
            *iwarn = 1;
        }

        i32 ldb_i = ldb;
        i32 idx = SLC_IDAMAX(&m, b, &ldb_i);
        bnorm = fabs(b[idx - 1]);

        if (cmax < ONE && bnorm > ONE) {
            if (bnorm > bignum * cmax) {
                *scale = ONE / bnorm;
            }
        }

        for (i = 0; i < m; i++) {
            b[i * ldb] = (b[i * ldb] * (*scale)) / cs;
        }
    } else {
        c[0][0] = a[0];
        c[1][1] = a[1 + lda];
        if (ltrans) {
            c[0][1] = a[1];
            c[1][0] = a[lda];
        } else {
            c[1][0] = a[1];
            c[0][1] = a[lda];
        }

        cv[0] = c[0][0];
        cv[1] = c[1][0];
        cv[2] = c[0][1];
        cv[3] = c[1][1];

        cmax = ZERO;
        icmax = 0;

        for (j = 0; j < 4; j++) {
            if (fabs(cv[j]) > cmax) {
                cmax = fabs(cv[j]);
                icmax = j + 1;
            }
        }

        if (cmax < smini) {
            i32 n_dim = n;
            i32 m_dim = m;
            bnorm = SLC_DLANGE("M", &n_dim, &m_dim, b, &ldb, cv);
            if (smini < ONE && bnorm > ONE) {
                if (bnorm > bignum * smini) {
                    *scale = ONE / bnorm;
                }
            }
            temp = (*scale) / smini;

            for (i = 0; i < m; i++) {
                b[i * ldb] = temp * b[i * ldb];
                b[1 + i * ldb] = temp * b[1 + i * ldb];
            }

            *iwarn = 1;
            return;
        }

        u11 = cv[icmax - 1];
        c21 = cv[ipivot[1][icmax - 1] - 1];
        u12 = cv[ipivot[2][icmax - 1] - 1];
        c22 = cv[ipivot[3][icmax - 1] - 1];
        u11r = ONE / u11;
        l21 = u11r * c21;
        u22 = c22 - u12 * l21;

        if (fabs(u22) < smini) {
            u22 = smini;
            *iwarn = 1;
        }

        scalep = ONE;

        for (i = 0; i < m; i++) {
            if (rswap[icmax - 1]) {
                b1 = b[1 + i * ldb];
                b2 = b[i * ldb];
            } else {
                b1 = b[i * ldb];
                b2 = b[1 + i * ldb];
            }
            b2 = b2 - l21 * b1;
            bbnd = fabs(b1 * (u22 * u11r));
            if (fabs(b2) > bbnd) {
                bbnd = fabs(b2);
            }
            if (bbnd > ONE && fabs(u22) < ONE) {
                if (bbnd >= bignum * fabs(u22)) {
                    *scale = ONE / bbnd;
                }
            }
            if (*scale < scalep) {
                *scale = (*scale < scalep) ? *scale : scalep;
            }
            if (*scale < scalep) {
                f64 ratio = (*scale) / scalep;
                for (j = 0; j < i; j++) {
                    b[j * ldb] = b[j * ldb] * ratio;
                    b[1 + j * ldb] = b[1 + j * ldb] * ratio;
                }
                scalep = *scale;
            }

            x2 = (b2 * (*scale)) / u22;
            x1 = ((*scale) * b1) * u11r - x2 * (u11r * u12);
            if (zswap[icmax - 1]) {
                b[i * ldb] = x2;
                b[1 + i * ldb] = x1;
            } else {
                b[i * ldb] = x1;
                b[1 + i * ldb] = x2;
            }
            xnorm = fabs(x1);
            if (fabs(x2) > xnorm) {
                xnorm = fabs(x2);
            }

            if (xnorm > ONE && cmax > ONE) {
                if (xnorm > bignum / cmax) {
                    temp = cmax / bignum;
                    b[i * ldb] = temp * b[i * ldb];
                    b[1 + i * ldb] = temp * b[1 + i * ldb];
                    *scale = temp * (*scale);
                }
            }
            scalep = *scale;
        }
    }
}
