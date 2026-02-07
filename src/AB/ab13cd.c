// SPDX-License-Identifier: BSD-3-Clause
//
// AB13CD - H-infinity norm of continuous-time stable system
//
// Translated from SLICOT AB13CD.f (Fortran 77)

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>

#define MAXIT 10
#define HUGE_VAL_SLICOT 1.0e30

static inline i32 max_i32(i32 a, i32 b) { return a > b ? a : b; }
static inline i32 min_i32(i32 a, i32 b) { return a < b ? a : b; }
static int select_stable(const f64* reig, const f64* ieig) {
    (void)ieig;
    return *reig < 0.0;
}

static int select_imaginary(const f64* reig, const f64* ieig) {
    (void)ieig;
    f64 eps = SLC_DLAMCH("Epsilon");
    f64 tol_ = 100.0 * eps;
    return fabs(*reig) < tol_;
}

f64 ab13cd(i32 n, i32 m, i32 np, const f64 *a, i32 lda, const f64 *b, i32 ldb,
           const f64 *c, i32 ldc, const f64 *d, i32 ldd, f64 tol,
           i32 *iwork, f64 *dwork, i32 ldwork, c128 *cwork, i32 lcwork,
           i32 *bwork, f64 *fpeak_out, i32 *info)
{
    const f64 zero = 0.0, one = 1.0, two = 2.0;
    i32 info2 = 0;

    *info = 0;

    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (np < 0) {
        *info = -3;
    } else if (lda < max_i32(1, n)) {
        *info = -5;
    } else if (ldb < max_i32(1, n)) {
        *info = -7;
    } else if (ldc < max_i32(1, np)) {
        *info = -9;
    } else if (ldd < max_i32(1, np)) {
        *info = -11;
    }

    i32 minwrk = max_i32(2, 4*n*n + 2*m*m + 3*m*n + m*np + 2*(n+np)*np + 10*n + 6*max_i32(m, np));
    if (*info == 0 && ldwork < minwrk) {
        *info = -15;
    }
    i32 mincwr = max_i32(1, (n + m) * (n + np) + 3 * max_i32(m, np));
    if (*info == 0 && lcwork < mincwr) {
        *info = -17;
    }

    if (*info != 0) {
        return zero;
    }

    if (m == 0 || np == 0) {
        *fpeak_out = zero;
        return zero;
    }

    i32 iw2 = n;
    i32 iw3 = iw2 + n;
    i32 iw4 = iw3 + n * n;
    i32 iw5 = iw4 + n * m;
    i32 iw6 = iw5 + np * m;
    i32 iwrk = iw6 + min_i32(np, m);

    f64 gammal = zero;
    f64 fpeak = HUGE_VAL_SLICOT;
    i32 lwamax = 0, lcwamx = 0;

    SLC_DLACPY("Full", &np, &m, d, &ldd, &dwork[iw5], &np);
    i32 ldnp = np;
    SLC_DGESVD("N", "N", &np, &m, &dwork[iw5], &ldnp, &dwork[iw6],
               dwork, &np, dwork, &m, &dwork[iwrk], &(i32){ldwork - iwrk}, &info2);
    if (info2 > 0) {
        *info = 4;
        return zero;
    }
    gammal = dwork[iw6];
    fpeak = HUGE_VAL_SLICOT;
    lwamax = (i32)dwork[iwrk] + iwrk;

    if (n == 0) {
        dwork[0] = two;
        dwork[1] = zero;
        cwork[0] = one;
        *fpeak_out = zero;
        return gammal;
    }

    SLC_DLACPY("Full", &n, &n, a, &lda, &dwork[iw3], &n);
    i32 ldn = n;
    i32 sdim = 0;
    SLC_DGEES("N", "S", select_stable, &n, &dwork[iw3], &ldn, &sdim, dwork,
              &dwork[iw2], dwork, &n, &dwork[iwrk], &(i32){ldwork - iwrk}, bwork, &info2);
    if (info2 > 0) {
        *info = 3;
        return zero;
    }
    if (sdim < n) {
        *info = 1;
        return zero;
    }
    lwamax = max_i32((i32)dwork[iwrk] + iwrk, lwamax);

    SLC_DLACPY("Full", &n, &n, a, &lda, &dwork[iw3], &n);
    SLC_DLACPY("Full", &n, &m, b, &ldb, &dwork[iw4], &n);
    SLC_DLACPY("Full", &np, &m, d, &ldd, &dwork[iw5], &np);
    SLC_DGESV(&n, &m, &dwork[iw3], &n, iwork, &dwork[iw4], &n, &info2);
    if (info2 > 0) {
        *info = 1;
        return zero;
    }
    f64 neg_one = -one;
    SLC_DGEMM("N", "N", &np, &m, &n, &neg_one, c, &ldc, &dwork[iw4], &n,
              &one, &dwork[iw5], &np);
    SLC_DGESVD("N", "N", &np, &m, &dwork[iw5], &np, &dwork[iw6],
               dwork, &np, dwork, &m, &dwork[iwrk], &(i32){ldwork - iwrk}, &info2);
    if (info2 > 0) {
        *info = 4;
        return zero;
    }
    if (gammal < dwork[iw6]) {
        gammal = dwork[iw6];
        fpeak = zero;
    }
    lwamax = max_i32((i32)dwork[iwrk] + iwrk, lwamax);

    SLC_DLACPY("Full", &n, &n, a, &lda, &dwork[iw3], &n);
    SLC_DGEES("N", "S", select_stable, &n, &dwork[iw3], &n, &sdim, dwork,
              &dwork[iw2], dwork, &n, &dwork[iwrk], &(i32){ldwork - iwrk}, bwork, &info2);

    bool complx = false;
    for (i32 i = 0; i < n; i++) {
        if (dwork[iw2 + i] != zero) {
            complx = true;
            break;
        }
    }

    f64 omega;
    if (!complx) {
        f64 wrmin = fabs(dwork[0]);
        for (i32 i = 1; i < n; i++) {
            if (wrmin > fabs(dwork[i])) wrmin = fabs(dwork[i]);
        }
        omega = wrmin;
    } else {
        f64 ratmax = zero;
        f64 wimax = zero;
        for (i32 i = 0; i < n; i++) {
            f64 den = hypot(dwork[i], dwork[iw2 + i]);
            f64 rat = fabs((dwork[iw2 + i] / dwork[i]) / den);
            if (ratmax < rat) {
                ratmax = rat;
                wimax = den;
            }
        }
        omega = wimax;
    }

    i32 icw2 = n * n;
    i32 icw3 = icw2 + n * m;
    i32 icw4 = icw3 + np * n;
    i32 icwrk = icw4 + np * m;

    c128 jimag = I;

    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < n; i++) {
            cwork[i + j * n] = -a[i + j * lda];
        }
        cwork[j + j * n] = jimag * omega - a[j + j * lda];
    }
    for (i32 j = 0; j < m; j++) {
        for (i32 i = 0; i < n; i++) {
            cwork[icw2 + i + j * n] = b[i + j * ldb];
        }
    }
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < np; i++) {
            cwork[icw3 + i + j * np] = c[i + j * ldc];
        }
    }
    for (i32 j = 0; j < m; j++) {
        for (i32 i = 0; i < np; i++) {
            cwork[icw4 + i + j * np] = d[i + j * ldd];
        }
    }

    SLC_ZGESV(&n, &m, cwork, &n, iwork, &cwork[icw2], &n, &info2);
    if (info2 > 0) {
        *info = 1;
        return zero;
    }

    c128 cone = 1.0 + 0.0 * I;
    SLC_ZGEMM("N", "N", &np, &m, &n, &cone, &cwork[icw3], &np,
              &cwork[icw2], &n, &cone, &cwork[icw4], &np);
    SLC_ZGESVD("N", "N", &np, &m, &cwork[icw4], &np, &dwork[iw6],
               cwork, &np, cwork, &m, &cwork[icwrk], &(i32){lcwork - icwrk},
               &dwork[iwrk], &info2);
    if (info2 > 0) {
        *info = 4;
        return zero;
    }
    if (gammal < dwork[iw6]) {
        gammal = dwork[iw6];
        fpeak = omega;
    }
    lcwamx = (i32)creal(cwork[icwrk]) + icwrk;

    iw2 = m * n;
    i32 iw3_ = iw2 + m * m;
    i32 iw4_ = iw3_ + np * np;
    i32 iw5_ = iw4_ + m * m;
    i32 iw6_ = iw5_ + m * n;
    i32 iw7 = iw6_ + m * n;
    i32 iw8 = iw7 + np * np;
    i32 iw9 = iw8 + np * n;
    i32 iw10 = iw9 + 4 * n * n;
    i32 iw11 = iw10 + 2 * n;
    i32 iw12 = iw11 + 2 * n;
    iwrk = iw12 + min_i32(np, m);

    SLC_DGEMM("T", "N", &m, &n, &np, &one, d, &ldd, c, &ldc, &zero, dwork, &m);
    SLC_DSYRK("U", "T", &m, &np, &one, d, &ldd, &zero, &dwork[iw2], &m);
    SLC_DSYRK("U", "N", &np, &m, &one, d, &ldd, &zero, &dwork[iw3_], &np);

    i32 iter = 0;
    f64 gamma = zero;
    f64 gammau = zero;

    while (iter < MAXIT) {
        iter++;
        gamma = (one + two * tol) * gammal;

        for (i32 j = 0; j < m; j++) {
            for (i32 i = 0; i <= j; i++) {
                dwork[iw4_ + i + j * m] = -dwork[iw2 + i + j * m];
            }
            dwork[iw4_ + j + j * m] = gamma * gamma - dwork[iw2 + j + j * m];
        }

        SLC_DLACPY("Full", &m, &n, dwork, &m, &dwork[iw5_], &m);
        SLC_DPOTRF("U", &m, &dwork[iw4_], &m, &info2);
        if (info2 > 0) {
            *info = 2;
            return zero;
        }
        SLC_DPOTRS("U", &m, &n, &dwork[iw4_], &m, &dwork[iw5_], &m, &info2);

        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < m; i++) {
                dwork[iw6_ + i + j * m] = b[j + i * ldb];
            }
        }
        SLC_DPOTRS("U", &m, &n, &dwork[iw4_], &m, &dwork[iw6_], &m, &info2);

        for (i32 j = 0; j < np; j++) {
            for (i32 i = 0; i <= j; i++) {
                dwork[iw7 + i + j * np] = -dwork[iw3_ + i + j * np];
            }
            dwork[iw7 + j + j * np] = gamma * gamma - dwork[iw3_ + j + j * np];
        }

        SLC_DLACPY("Full", &np, &n, c, &ldc, &dwork[iw8], &np);
        SLC_DPOSV("U", &np, &n, &dwork[iw7], &np, &dwork[iw8], &np, &info2);
        if (info2 > 0) {
            *info = 2;
            return zero;
        }

        i32 ld2n = 2 * n;
        SLC_DLACPY("Full", &n, &n, a, &lda, &dwork[iw9], &ld2n);
        SLC_DGEMM("N", "N", &n, &n, &m, &one, b, &ldb, &dwork[iw5_], &m,
                  &one, &dwork[iw9], &ld2n);

        f64 neg_gamma = -gamma;
        slicot_mb01rx('L', 'U', 'T', n, np, zero, neg_gamma, &dwork[iw9 + n], ld2n,
                      c, ldc, &dwork[iw8], np);
        ma02ed('U', n, &dwork[iw9 + n], ld2n);
        slicot_mb01rx('L', 'U', 'N', n, m, zero, gamma, &dwork[iw9 + 2*n*n], ld2n,
                      b, ldb, &dwork[iw6_], m);
        ma02ed('U', n, &dwork[iw9 + 2*n*n], ld2n);

        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < n; i++) {
                dwork[iw9 + 2*n*n + n + i + j * ld2n] = -dwork[iw9 + j + i * ld2n];
            }
        }

        SLC_DGEES("N", "S", select_imaginary, &ld2n, &dwork[iw9], &ld2n, &sdim,
                  &dwork[iw10], &dwork[iw11], dwork, &ld2n,
                  &dwork[iwrk], &(i32){ldwork - iwrk}, bwork, &info2);
        if (info2 > 0) {
            *info = 3;
            return zero;
        }
        lwamax = max_i32((i32)dwork[iwrk] + iwrk, lwamax);

        if (sdim == 0) {
            gammau = gamma;
            break;
        }

        i32 k = 0;
        for (i32 i = 0; i < sdim - 1; i += 2) {
            dwork[iw10 + k] = dwork[iw11 + i];
            k++;
        }

        if (k >= 2) {
            for (i32 j = 0; j < k - 1; j++) {
                for (i32 l = j + 1; l < k; l++) {
                    if (dwork[iw10 + j] > dwork[iw10 + l]) {
                        f64 temp = dwork[iw10 + j];
                        dwork[iw10 + j] = dwork[iw10 + l];
                        dwork[iw10 + l] = temp;
                    }
                }
            }

            for (i32 l = 0; l < k - 1; l++) {
                omega = (dwork[iw10 + l] + dwork[iw10 + l + 1]) / two;

                for (i32 j = 0; j < n; j++) {
                    for (i32 i = 0; i < n; i++) {
                        cwork[i + j * n] = -a[i + j * lda];
                    }
                    cwork[j + j * n] = jimag * omega - a[j + j * lda];
                }
                for (i32 j = 0; j < m; j++) {
                    for (i32 i = 0; i < n; i++) {
                        cwork[icw2 + i + j * n] = b[i + j * ldb];
                    }
                }
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = 0; i < np; i++) {
                        cwork[icw3 + i + j * np] = c[i + j * ldc];
                    }
                }
                for (i32 j = 0; j < m; j++) {
                    for (i32 i = 0; i < np; i++) {
                        cwork[icw4 + i + j * np] = d[i + j * ldd];
                    }
                }

                SLC_ZGESV(&n, &m, cwork, &n, iwork, &cwork[icw2], &n, &info2);
                if (info2 > 0) {
                    *info = 1;
                    return zero;
                }
                SLC_ZGEMM("N", "N", &np, &m, &n, &cone, &cwork[icw3], &np,
                          &cwork[icw2], &n, &cone, &cwork[icw4], &np);
                SLC_ZGESVD("N", "N", &np, &m, &cwork[icw4], &np, &dwork[iw6],
                           cwork, &np, cwork, &m, &cwork[icwrk], &(i32){lcwork - icwrk},
                           &dwork[iwrk], &info2);
                if (info2 > 0) {
                    *info = 4;
                    return zero;
                }
                if (gammal < dwork[iw6]) {
                    gammal = dwork[iw6];
                    fpeak = omega;
                }
                lcwamx = max_i32((i32)creal(cwork[icwrk]) + icwrk, lcwamx);
            }
        }
    }

    if (iter > MAXIT) {
        *info = 2;
        return zero;
    }

    f64 hnorm = (gammal + gammau) / two;

    dwork[0] = (f64)lwamax;
    dwork[1] = fpeak;
    cwork[0] = (c128)lcwamx;
    *fpeak_out = fpeak;

    return hnorm;
}
