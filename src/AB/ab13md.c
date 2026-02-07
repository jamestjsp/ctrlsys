/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

static int select_dummy(const c128* w) {
    (void)w;
    return 0;
}

i32 ab13md(
    char fact,
    i32 n,
    c128* z,
    i32 ldz,
    i32 m,
    const i32* nblock,
    const i32* itype,
    f64* x,
    f64* bound,
    f64* d,
    f64* g,
    i32* iwork,
    f64* dwork,
    i32 ldwork,
    c128* zwork,
    i32 lzwork
) {
    static const c128 CZERO = 0.0 + 0.0*I;
    static const c128 CONE = 1.0 + 0.0*I;
    static const c128 CIMAG = 0.0 + 1.0*I;
    static const f64 ZERO = 0.0;
    static const f64 ONE = 1.0;
    static const f64 TWO = 2.0;
    static const f64 FOUR = 4.0;
    static const f64 FIVE = 5.0;
    static const f64 EIGHT = 8.0;
    static const f64 TEN = 10.0;
    static const f64 FORTY = 40.0;
    static const f64 FIFTY = 50.0;
    static const f64 ALPHA = 100.0;
    static const f64 BETA = 0.01;
    static const f64 THETA = 0.01;
    static const f64 C1 = 1.0e-3;
    static const f64 C2 = 1.0e-2;
    static const f64 C3 = 0.25;
    static const f64 C4 = 0.9;
    static const f64 C5 = 1.5;
    static const f64 C6 = 10.0;
    static const f64 C7 = 100.0;
    static const f64 C8 = 1.0e3;
    static const f64 C9 = 1.0e4;

    i32 info = 0;
    i32 info2;
    i32 minwrk, minzrk;
    i32 i, j, k, l;
    i32 nsum, isum, mr, mt;
    i32 iter, sdim;
    i32 lwa, lwamax, lza, lzamax;
    f64 eps, tol, tol2, tol3, tol4, tol5, regpar;
    f64 znorm, znorm2, scale, tau, snorm;
    f64 svlam, c, dlambd, e, emax, emin;
    f64 phi, pp, prod, rat, rcond, temp;
    f64 colsum, rowsum, stsize;
    f64 hnorm, hnorm1, hn, delta;
    f64 t1, t2, t3, ynorm1, ynorm2;
    c128 detf, tempij, tempji;
    bool xfact, pos, gtest;
    i32 bwork[1];

    i32 iw2, iw3, iw4, iw5, iw6, iw7, iw8, iw9, iw10, iw11, iw12;
    i32 iw13, iw14, iw15, iw16, iw17, iw18, iw19, iw20, iw21, iw22;
    i32 iw23, iw24, iw25, iw26, iw27, iw28, iw29, iw30, iw31, iw32;
    i32 iw33, iwrk;
    i32 iz2, iz3, iz4, iz5, iz6, iz7, iz8, iz9, iz10, iz11;
    i32 iz12, iz13, iz14, iz15, iz16, iz17, iz18, iz19, iz20, iz21;
    i32 iz22, iz23, iz24, izwrk;

    i32 one_int = 1;

    minwrk = 2*n*n*m - n*n + 9*m*m + n*m + 11*n + 33*m - 11;
    minzrk = 6*n*n*m + 12*n*n + 6*m + 6*n - 3;

    xfact = (fact == 'F' || fact == 'f');
    if (!xfact && fact != 'N' && fact != 'n') {
        info = -1;
    } else if (n < 0) {
        info = -2;
    } else if (ldz < (n > 1 ? n : 1)) {
        info = -4;
    } else if (m < 1) {
        info = -5;
    } else if (ldwork < minwrk) {
        info = -14;
    } else if (lzwork < minzrk) {
        info = -16;
    }
    if (info != 0) {
        return info;
    }

    nsum = 0;
    isum = 0;
    mr = 0;
    for (i = 0; i < m; i++) {
        if (nblock[i] < 1) {
            return 1;
        }
        if (itype[i] == 1 && nblock[i] > 1) {
            return 3;
        }
        nsum += nblock[i];
        if (itype[i] == 1) mr++;
        if (itype[i] == 1 || itype[i] == 2) isum++;
    }
    if (nsum != n) {
        return 2;
    }
    if (isum != m) {
        return 4;
    }
    mt = m + mr - 1;

    lwamax = 0;
    lzamax = 0;

    SLC_DLASET("Full", &n, &one_int, &ONE, &ONE, d, &n);
    SLC_DLASET("Full", &n, &one_int, &ZERO, &ZERO, g, &n);

    znorm = SLC_ZLANGE("F", &n, &n, z, &ldz, dwork);
    if (znorm == ZERO) {
        *bound = ZERO;
        dwork[0] = ONE;
        zwork[0] = CONE;
        return 0;
    }

    SLC_ZLACPY("Full", &n, &n, z, &ldz, zwork, &n);

    if (nblock[0] == n) {
        if (itype[0] == 1) {
            if (cimag(z[0]) != ZERO) {
                *bound = ZERO;
            } else {
                *bound = fabs(creal(z[0]));
            }
            dwork[0] = ONE;
            zwork[0] = CONE;
        } else {
            i32 n2 = n*n;
            SLC_ZGESVD("N", "N", &n, &n, zwork, &n, dwork, zwork, &one_int,
                       zwork, &one_int, &zwork[n2], &lzwork,
                       &dwork[n], &info2);
            if (info2 > 0) {
                return 6;
            }
            *bound = dwork[0];
            lza = n2 + (i32)creal(zwork[n2]);
            dwork[0] = 5*n;
            zwork[0] = (f64)lza;
        }
        return 0;
    }

    eps = SLC_DLAMCH("P");

    tol = C7 * sqrt(eps);
    tol2 = C9 * eps;
    tol3 = C6 * eps;
    tol4 = C1;
    tol5 = C1;
    regpar = C8 * eps;

    iw2 = m*m;
    iw3 = iw2 + m;
    iw4 = iw3 + n;
    iw5 = iw4 + m;
    iw6 = iw5 + m;
    iw7 = iw6 + n;
    iw8 = iw7 + n;
    iw9 = iw8 + n*(m-1);
    iw10 = iw9 + n*n*mt;
    iw11 = iw10 + mt;
    iw12 = iw11 + mt*mt;
    iw13 = iw12 + n;
    iw14 = iw13 + mt + 1;
    iw15 = iw14 + mt + 1;
    iw16 = iw15 + mt + 1;
    iw17 = iw16 + mt + 1;
    iw18 = iw17 + mt + 1;
    iw19 = iw18 + mt;
    iw20 = iw19 + mt;
    iw21 = iw20 + mt;
    iw22 = iw21 + n;
    iw23 = iw22 + m - 1;
    iw24 = iw23 + mr;
    iw25 = iw24 + n;
    iw26 = iw25 + 2*mt;
    iw27 = iw26 + mt;
    iw28 = iw27 + mt;
    iw29 = iw28 + m - 1;
    iw30 = iw29 + mr;
    iw31 = iw30 + n + 2*mt;
    iw32 = iw31 + mt*mt;
    iw33 = iw32 + mt;
    iwrk = iw33 + mt + 1;

    iz2 = n*n;
    iz3 = iz2 + n*n;
    iz4 = iz3 + n*n;
    iz5 = iz4 + n*n;
    iz6 = iz5 + n*n;
    iz7 = iz6 + n*n*mt;
    iz8 = iz7 + n*n;
    iz9 = iz8 + n*n;
    iz10 = iz9 + n*n;
    iz11 = iz10 + mt;
    iz12 = iz11 + n*n;
    iz13 = iz12 + n;
    iz14 = iz13 + n*n;
    iz15 = iz14 + n;
    iz16 = iz15 + n*n;
    iz17 = iz16 + n;
    iz18 = iz17 + n*n;
    iz19 = iz18 + n*n*mt;
    iz20 = iz19 + mt;
    iz21 = iz20 + n*n*mt;
    iz22 = iz21 + n*n;
    iz23 = iz22 + n*n;
    iz24 = iz23 + n*n;
    izwrk = iz24 + mt;

    iwork[0] = 0;
    for (i = 1; i <= m; i++) {
        iwork[i] = iwork[i-1] + nblock[i-1];
    }

    if (!xfact) {
        SLC_DLASET("Full", &m, &m, &ZERO, &ZERO, dwork, &m);
        SLC_DLASET("Full", &m, &one_int, &ONE, &ONE, &dwork[iw2], &m);
        znorm = SLC_ZLANGE("F", &n, &n, zwork, &n, dwork);

        for (j = 0; j < m; j++) {
            for (i = 0; i < m; i++) {
                if (i != j) {
                    i32 rows = iwork[i+1] - iwork[i];
                    i32 cols = iwork[j+1] - iwork[j];
                    SLC_ZLACPY("Full", &rows, &cols,
                               &z[iwork[i] + iwork[j]*ldz], &ldz,
                               &zwork[iz2], &n);
                    i32 lzw = lzwork - izwrk;
                    SLC_ZGESVD("N", "N", &rows, &cols, &zwork[iz2],
                               &n, &dwork[iw3], zwork, &one_int, zwork, &one_int,
                               &zwork[izwrk], &lzw, &dwork[iwrk], &info2);
                    if (info2 > 0) {
                        return 6;
                    }
                    lza = (i32)creal(zwork[izwrk]);
                    if (lza > lzamax) lzamax = lza;
                    znorm2 = dwork[iw3];
                    dwork[i + j*m] = znorm2 + znorm*tol2;
                }
            }
        }

        SLC_DLASET("Full", &m, &one_int, &ZERO, &ZERO, &dwork[iw4], &m);

        int converged = 0;
        while (!converged) {
            for (i = 0; i < m; i++) {
                dwork[iw5 + i] = dwork[iw4 + i] - ONE;
            }
            hnorm = SLC_DLANGE("F", &m, &one_int, &dwork[iw5], &m, dwork);
            if (hnorm <= tol2) {
                converged = 1;
                break;
            }
            for (k = 0; k < m; k++) {
                colsum = ZERO;
                for (i = 0; i < m; i++) {
                    colsum += dwork[i + k*m];
                }
                rowsum = ZERO;
                for (j = 0; j < m; j++) {
                    rowsum += dwork[k + j*m];
                }
                rat = sqrt(colsum / rowsum);
                dwork[iw4 + k] = rat;
                for (i = 0; i < m; i++) {
                    dwork[i + k*m] /= rat;
                }
                for (j = 0; j < m; j++) {
                    dwork[k + j*m] *= rat;
                }
                dwork[iw2 + k] *= rat;
            }
        }

        scale = ONE / dwork[iw2];
        SLC_DSCAL(&m, &scale, &dwork[iw2], &one_int);
    } else {
        dwork[iw2] = ONE;
        for (i = 1; i < m; i++) {
            dwork[iw2 + i] = sqrt(x[i-1]);
        }
    }

    for (j = 0; j < m; j++) {
        for (i = 0; i < m; i++) {
            if (i != j) {
                i32 rows = iwork[i+1] - iwork[i];
                i32 cols = iwork[j+1] - iwork[j];
                SLC_ZLASCL("G", &m, &m, &dwork[iw2+j], &dwork[iw2+i],
                           &rows, &cols,
                           &zwork[iwork[i] + iwork[j]*n], &n, &info2);
            }
        }
    }

    SLC_ZLACPY("Full", &n, &n, zwork, &n, &zwork[iz2], &n);
    i32 lzw = lzwork - izwrk;
    SLC_ZGESVD("N", "N", &n, &n, &zwork[iz2], &n, &dwork[iw3],
               zwork, &one_int, zwork, &one_int, &zwork[izwrk], &lzw,
               &dwork[iwrk], &info2);
    if (info2 > 0) {
        return 6;
    }
    lza = (i32)creal(zwork[izwrk]);
    if (lza > lzamax) lzamax = lza;
    znorm = dwork[iw3];
    SLC_ZLASCL("G", &m, &m, &znorm, &ONE, &n, &n, zwork, &n, &info2);

    i32 n2 = n*n;
    i32 mt_int = mt;
    SLC_DLASET("Full", &n2, &mt_int, &ZERO, &ZERO, &dwork[iw9], &n2);

    for (i = 0; i < nblock[0]; i++) {
        dwork[iw6 + i] = ONE;
    }
    for (i = nblock[0]; i < n; i++) {
        dwork[iw6 + i] = ZERO;
    }

    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
            zwork[iz3 + i + j*n] = dwork[iw6 + i] * zwork[i + j*n];
        }
    }

    SLC_ZGEMM("C", "N", &n, &n, &n, &CONE, zwork, &n, &zwork[iz3], &n,
              &CZERO, &zwork[iz4], &n);

    SLC_ZLACPY("Full", &n, &n, &zwork[iz4], &n, &zwork[iz5], &n);

    SLC_DCOPY(&n, &dwork[iw6], &one_int, &dwork[iw7], &one_int);

    for (k = 1; k < m; k++) {
        for (i = 0; i < iwork[k]; i++) {
            dwork[iw6 + i] = ZERO;
        }
        for (i = iwork[k]; i < iwork[k] + nblock[k]; i++) {
            dwork[iw6 + i] = ONE;
        }
        if (k < m - 1) {
            for (i = iwork[k+1]; i < n; i++) {
                dwork[iw6 + i] = ZERO;
            }
        }

        for (j = 0; j < n; j++) {
            for (i = 0; i < n; i++) {
                zwork[iz3 + i + j*n] = dwork[iw6 + i] * zwork[i + j*n];
            }
        }

        SLC_ZGEMM("C", "N", &n, &n, &n, &CONE, zwork, &n, &zwork[iz3], &n,
                  &CZERO, &zwork[iz4], &n);

        SLC_ZCOPY(&n2, &zwork[iz4], &one_int, &zwork[iz6 + (k-1)*n2], &one_int);

        SLC_DCOPY(&n, &dwork[iw6], &one_int, &dwork[iw8 + (k-1)*n], &one_int);

        for (i = 0; i < n; i++) {
            dwork[iw9 + i + i*n + (k-1)*n2] = dwork[iw6 + i];
        }
    }

    l = 0;
    for (k = 0; k < m; k++) {
        if (itype[k] == 1) {
            l++;
            for (i = 0; i < iwork[k]; i++) {
                dwork[iw6 + i] = ZERO;
            }
            for (i = iwork[k]; i < iwork[k] + nblock[k]; i++) {
                dwork[iw6 + i] = ONE;
            }
            if (k < m - 1) {
                for (i = iwork[k+1]; i < n; i++) {
                    dwork[iw6 + i] = ZERO;
                }
            }

            for (j = 0; j < n; j++) {
                for (i = 0; i < n; i++) {
                    zwork[iz3 + i + j*n] = dwork[iw6 + i] * zwork[i + j*n];
                }
            }

            for (j = 0; j < n; j++) {
                for (i = 0; i <= j; i++) {
                    tempij = zwork[iz3 + i + j*n];
                    tempji = zwork[iz3 + j + i*n];
                    zwork[iz4 + i + j*n] = CIMAG * (tempij - conj(tempji));
                    zwork[iz4 + j + i*n] = CIMAG * (tempji - conj(tempij));
                }
            }

            SLC_ZCOPY(&n2, &zwork[iz4], &one_int, &zwork[iz6 + (m-2+l)*n2], &one_int);
        }
    }

    for (i = 0; i < m - 1; i++) {
        x[i] = ONE;
    }
    if (mr > 0) {
        if (!xfact) {
            for (i = 0; i < mr; i++) {
                x[m-1 + i] = ZERO;
            }
        } else {
            l = 0;
            for (k = 0; k < m; k++) {
                if (itype[k] == 1) {
                    x[m-1 + l] = x[m-1 + l] / (dwork[iw2 + k] * dwork[iw2 + k]);
                    l++;
                }
            }
        }
    }

    svlam = ONE / eps;
    c = ONE;

    SLC_DLASET("Full", &mt_int, &mt_int, &ZERO, &ONE, &dwork[iw11], &mt_int);

    iter = -1;

iter_loop:
    iter++;

    for (i = 0; i < mt; i++) {
        zwork[iz10 + i] = x[i];
    }
    SLC_ZCOPY(&n2, &zwork[iz5], &one_int, &zwork[iz7], &one_int);
    SLC_ZGEMV("N", &n2, &mt_int, &CONE, &zwork[iz6], &n2,
              &zwork[iz10], &one_int, &CONE, &zwork[iz7], &one_int);

    SLC_DCOPY(&n, &dwork[iw7], &one_int, &dwork[iw12], &one_int);
    i32 mm1 = m - 1;
    SLC_DGEMV("N", &n, &mm1, &ONE, &dwork[iw8], &n, x, &one_int, &ONE,
              &dwork[iw12], &one_int);
    for (i = 0; i < n; i++) {
        dwork[iw12 + i] = ONE / dwork[iw12 + i];
    }

    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
            zwork[iz11 + i + j*n] = dwork[iw12 + i] * zwork[iz7 + i + j*n];
        }
    }

    lzw = lzwork - izwrk;
    SLC_ZGEES("N", "N", select_dummy, &n, &zwork[iz11], &n, &sdim,
              &zwork[iz12], zwork, &n, &zwork[izwrk], &lzw,
              &dwork[iwrk], bwork, &info2);
    if (info2 > 0) {
        return 6;
    }
    lza = (i32)creal(zwork[izwrk]);
    if (lza > lzamax) lzamax = lza;

    e = creal(zwork[iz12]);
    for (i = 1; i < n; i++) {
        if (creal(zwork[iz12 + i]) > e) {
            e = creal(zwork[iz12 + i]);
        }
    }

    if (mr > 0) {
        snorm = fabs(x[m-1]);
        for (i = m; i < mt; i++) {
            if (fabs(x[i]) > snorm) snorm = fabs(x[i]);
        }
        if (snorm > FORTY) {
            tau = C7;
        } else if (snorm > EIGHT) {
            tau = FIFTY;
        } else if (snorm > FOUR) {
            tau = TEN;
        } else if (snorm > ONE) {
            tau = FIVE;
        } else {
            tau = TWO;
        }
    }

    if (iter == 0) {
        dlambd = e + C1;
    } else {
        dwork[iw13] = e;
        SLC_DCOPY(&mt_int, x, &one_int, &dwork[iw13+1], &one_int);
        dlambd = (ONE - THETA)*dwork[iw13] + THETA*dwork[iw14];
        SLC_DCOPY(&mt_int, &dwork[iw13+1], &one_int, &dwork[iw18], &one_int);
        SLC_DCOPY(&mt_int, &dwork[iw14+1], &one_int, &dwork[iw19], &one_int);

        l = 0;
theta_loop:
        for (i = 0; i < mt; i++) {
            f64 th_scaled = THETA / pow(2.0, l);
            x[i] = (ONE - th_scaled)*dwork[iw18+i] + th_scaled*dwork[iw19+i];
        }

        for (i = 0; i < mt; i++) {
            zwork[iz10 + i] = x[i];
        }
        SLC_ZCOPY(&n2, &zwork[iz5], &one_int, &zwork[iz9], &one_int);
        SLC_ZGEMV("N", &n2, &mt_int, &CONE, &zwork[iz6], &n2,
                  &zwork[iz10], &one_int, &CONE, &zwork[iz9], &one_int);

        SLC_DCOPY(&n, &dwork[iw7], &one_int, &dwork[iw21], &one_int);
        SLC_DGEMV("N", &n, &mm1, &ONE, &dwork[iw8], &n, x, &one_int, &ONE,
                  &dwork[iw21], &one_int);

        for (j = 0; j < n; j++) {
            for (i = 0; i < n; i++) {
                if (i == j) {
                    zwork[iz13 + i + i*n] = THETA*BETA*(dwork[iw14] - dwork[iw13])/TWO
                                          - dlambd*dwork[iw21+i] + zwork[iz9 + i + i*n];
                } else {
                    zwork[iz13 + i + j*n] = zwork[iz9 + i + j*n];
                }
            }
        }

        lzw = lzwork - izwrk;
        SLC_ZGEES("N", "N", select_dummy, &n, &zwork[iz13], &n, &sdim,
                  &zwork[iz14], zwork, &n, &zwork[izwrk], &lzw,
                  &dwork[iwrk], bwork, &info2);
        if (info2 > 0) {
            return 6;
        }
        lza = (i32)creal(zwork[izwrk]);
        if (lza > lzamax) lzamax = lza;

        emax = creal(zwork[iz14]);
        for (i = 1; i < n; i++) {
            if (creal(zwork[iz14 + i]) > emax) {
                emax = creal(zwork[iz14 + i]);
            }
        }

        if (emax <= ZERO) {
            goto set_y;
        } else {
            l++;
            goto theta_loop;
        }
    }

set_y:
    dwork[iw13] = dlambd;
    SLC_DCOPY(&mt_int, x, &one_int, &dwork[iw13+1], &one_int);

    if ((svlam - dlambd) < tol) {
        *bound = sqrt(e > ZERO ? e : ZERO) * znorm;
        for (i = 0; i < m - 1; i++) {
            x[i] = x[i] * dwork[iw2 + i + 1] * dwork[iw2 + i + 1];
        }

        for (i = 0; i < m - 1; i++) {
            dwork[iw20 + i] = sqrt(x[i]);
        }

        SLC_DCOPY(&n, &dwork[iw7], &one_int, d, &one_int);
        SLC_DGEMV("N", &n, &mm1, &ONE, &dwork[iw8], &n,
                  &dwork[iw20], &one_int, &ONE, d, &one_int);

        j = 0;
        l = 0;
        for (k = 0; k < m; k++) {
            j += nblock[k];
            if (itype[k] == 1) {
                x[m-1 + l] = x[m-1 + l] * dwork[iw2 + k] * dwork[iw2 + k];
                g[j - 1] = x[m-1 + l];
                l++;
            }
        }
        SLC_DSCAL(&n, &znorm, g, &one_int);

        dwork[0] = (f64)(minwrk - 5*n + lwamax);
        zwork[0] = (f64)(minzrk - 3*n + lzamax);
        return 0;
    }

    svlam = dlambd;

    for (k = 0; k < m; k++) {
        SLC_DCOPY(&mm1, x, &one_int, &dwork[iw22], &one_int);
        if (mr > 0) {
            SLC_DCOPY(&mr, &x[m-1], &one_int, &dwork[iw23], &one_int);
        }

        for (i = 0; i < mt; i++) {
            zwork[iz10 + i] = x[i];
        }
        SLC_ZCOPY(&n2, &zwork[iz5], &one_int, &zwork[iz7], &one_int);
        SLC_ZGEMV("N", &n2, &mt_int, &CONE, &zwork[iz6], &n2,
                  &zwork[iz10], &one_int, &CONE, &zwork[iz7], &one_int);

        SLC_DCOPY(&n, &dwork[iw7], &one_int, &dwork[iw24], &one_int);
        SLC_DGEMV("N", &n, &mm1, &ONE, &dwork[iw8], &n,
                  &dwork[iw22], &one_int, &ONE, &dwork[iw24], &one_int);

        for (j = 0; j < n; j++) {
            for (i = 0; i < n; i++) {
                if (i == j) {
                    zwork[iz15 + i + i*n] = dlambd*dwork[iw24+i] - zwork[iz7 + i + i*n];
                } else {
                    zwork[iz15 + i + j*n] = -zwork[iz7 + i + j*n];
                }
            }
        }
        SLC_ZLACPY("Full", &n, &n, &zwork[iz15], &n, &zwork[iz17], &n);

        lzw = lzwork - izwrk;
        SLC_ZGEES("N", "N", select_dummy, &n, &zwork[iz15], &n, &sdim,
                  &zwork[iz16], zwork, &n, &zwork[izwrk], &lzw,
                  &dwork[iwrk], bwork, &info2);
        if (info2 > 0) {
            return 6;
        }
        lza = (i32)creal(zwork[izwrk]);
        if (lza > lzamax) lzamax = lza;

        detf = CONE;
        for (i = 0; i < n; i++) {
            detf *= zwork[iz16 + i];
        }

        SLC_ZGETRF(&n, &n, &zwork[iz17], &n, iwork, &info2);
        if (info2 > 0) {
            return 5;
        }
        i32 lzw = lzwork - izwrk;
        SLC_ZGETRI(&n, &zwork[iz17], &n, iwork, &zwork[izwrk], &lzw, &info2);
        lza = (i32)creal(zwork[izwrk]);
        if (lza > lzamax) lzamax = lza;

        for (i = 0; i < m - 1; i++) {
            dwork[iw25 + i] = dwork[iw22 + i] - BETA;
            dwork[iw25 + m - 1 + i] = ALPHA - dwork[iw22 + i];
        }
        if (mr > 0) {
            for (i = 0; i < mr; i++) {
                dwork[iw25 + 2*(m-1) + i] = dwork[iw23 + i] + tau;
                dwork[iw25 + 2*(m-1) + mr + i] = tau - dwork[iw23 + i];
            }
        }
        prod = ONE;
        for (i = 0; i < 2*mt; i++) {
            prod *= dwork[iw25 + i];
        }
        temp = creal(detf);
        if (temp < eps) temp = eps;
        phi = -log(temp) - log(prod);

        for (j = 0; j < mt; j++) {
            for (i = 0; i < n2; i++) {
                zwork[iz18 + i + j*n2] = dlambd*dwork[iw9 + i + j*n2] - zwork[iz6 + i + j*n2];
            }
        }
        SLC_ZGEMV("C", &n2, &mt_int, &CONE, &zwork[iz18], &n2,
                  &zwork[iz17], &one_int, &CZERO, &zwork[iz19], &one_int);

        for (i = 0; i < m - 1; i++) {
            dwork[iw26 + i] = ONE/(dwork[iw22 + i] - BETA) - ONE/(ALPHA - dwork[iw22 + i]);
        }
        if (mr > 0) {
            for (i = 0; i < mr; i++) {
                dwork[iw26 + m - 1 + i] = ONE/(dwork[iw23 + i] + tau) - ONE/(tau - dwork[iw23 + i]);
            }
        }
        for (i = 0; i < mt; i++) {
            dwork[iw26 + i] = -creal(zwork[iz19 + i]) - dwork[iw26 + i];
        }

        SLC_DLACPY("Full", &mt_int, &mt_int, &dwork[iw11], &mt_int, &dwork[iw31], &mt_int);
        SLC_DCOPY(&mt_int, &dwork[iw26], &one_int, &dwork[iw27], &one_int);
        i32 ldw2 = ldwork - iwrk;
        SLC_DSYSV("U", &mt_int, &one_int, &dwork[iw31], &mt_int, iwork,
                  &dwork[iw27], &mt_int, &dwork[iwrk], &ldw2, &info2);
        if (info2 > 0) {
            return 5;
        }
        lwa = (i32)dwork[iwrk];
        if (lwa > lwamax) lwamax = lwa;

        stsize = ONE;

        SLC_DCOPY(&mm1, &dwork[iw27], &one_int, &dwork[iw28], &one_int);

        l = 0;
        for (i = 0; i < m - 1; i++) {
            if (dwork[iw28 + i] > ZERO) {
                l++;
                temp = (dwork[iw22 + i] - BETA) / dwork[iw28 + i];
                if (l == 1) {
                    stsize = (temp < stsize) ? temp : stsize;
                } else {
                    stsize = (temp < stsize) ? temp : stsize;
                }
            }
        }

        l = 0;
        for (i = 0; i < m - 1; i++) {
            if (dwork[iw28 + i] < ZERO) {
                l++;
                temp = (ALPHA - dwork[iw22 + i]) / (-dwork[iw28 + i]);
                if (l == 1 || temp < stsize) {
                    stsize = temp;
                }
            }
        }

        if (mr > 0) {
            SLC_DCOPY(&mr, &dwork[iw27 + m - 1], &one_int, &dwork[iw29], &one_int);

            l = 0;
            for (i = 0; i < mr; i++) {
                if (dwork[iw29 + i] > ZERO) {
                    l++;
                    temp = (dwork[iw23 + i] + tau) / dwork[iw29 + i];
                    if (l == 1 || temp < stsize) {
                        stsize = temp;
                    }
                }
            }

            l = 0;
            for (i = 0; i < mr; i++) {
                if (dwork[iw29 + i] < ZERO) {
                    l++;
                    temp = (tau - dwork[iw23 + i]) / (-dwork[iw29 + i]);
                    if (l == 1 || temp < stsize) {
                        stsize = temp;
                    }
                }
            }
        }

        stsize = C4 * stsize;

        if (stsize >= tol4) {
            for (i = 0; i < mt; i++) {
                dwork[iw20 + i] = x[i] - stsize * dwork[iw27 + i];
            }

            SLC_DCOPY(&mm1, &dwork[iw20], &one_int, &dwork[iw22], &one_int);
            if (mr > 0) {
                SLC_DCOPY(&mr, &dwork[iw20 + m - 1], &one_int, &dwork[iw23], &one_int);
            }

            for (i = 0; i < mt; i++) {
                zwork[iz10 + i] = dwork[iw20 + i];
            }
            SLC_ZCOPY(&n2, &zwork[iz5], &one_int, &zwork[iz7], &one_int);
            SLC_ZGEMV("N", &n2, &mt_int, &CONE, &zwork[iz6], &n2,
                      &zwork[iz10], &one_int, &CONE, &zwork[iz7], &one_int);

            SLC_DCOPY(&n, &dwork[iw7], &one_int, &dwork[iw24], &one_int);
            SLC_DGEMV("N", &n, &mm1, &ONE, &dwork[iw8], &n,
                      &dwork[iw22], &one_int, &ONE, &dwork[iw24], &one_int);

            for (j = 0; j < n; j++) {
                for (i = 0; i < n; i++) {
                    if (i == j) {
                        zwork[iz15 + i + i*n] = dlambd*dwork[iw24+i] - zwork[iz7 + i + i*n];
                    } else {
                        zwork[iz15 + i + j*n] = -zwork[iz7 + i + j*n];
                    }
                }
            }

            lzw = lzwork - izwrk;
            SLC_ZGEES("N", "N", select_dummy, &n, &zwork[iz15], &n, &sdim,
                      &zwork[iz16], zwork, &n, &zwork[izwrk], &lzw,
                      &dwork[iwrk], bwork, &info2);
            if (info2 > 0) {
                return 6;
            }
            lza = (i32)creal(zwork[izwrk]);
            if (lza > lzamax) lzamax = lza;

            emin = creal(zwork[iz16]);
            for (i = 1; i < n; i++) {
                if (creal(zwork[iz16 + i]) < emin) {
                    emin = creal(zwork[iz16 + i]);
                }
            }

            for (i = 0; i < n; i++) {
                dwork[iw30 + i] = creal(zwork[iz16 + i]);
            }
            for (i = 0; i < m - 1; i++) {
                dwork[iw30 + n + i] = dwork[iw22 + i] - BETA;
                dwork[iw30 + n + m - 1 + i] = ALPHA - dwork[iw22 + i];
            }
            if (mr > 0) {
                for (i = 0; i < mr; i++) {
                    dwork[iw30 + n + 2*(m-1) + i] = dwork[iw23 + i] + tau;
                    dwork[iw30 + n + 2*(m-1) + mr + i] = tau - dwork[iw23 + i];
                }
            }

            prod = ONE;
            for (i = 0; i < n + 2*mt; i++) {
                prod *= dwork[iw30 + i];
            }

            if (emin <= ZERO || (-log(prod)) >= phi) {
                stsize /= TEN;
            } else {
                SLC_DCOPY(&mt_int, &dwork[iw20], &one_int, x, &one_int);
            }
        }

        if (stsize < tol4) break;
    }

newton_loop:
    SLC_DCOPY(&mm1, x, &one_int, &dwork[iw22], &one_int);
    if (mr > 0) {
        SLC_DCOPY(&mr, &x[m-1], &one_int, &dwork[iw23], &one_int);
    }

    for (i = 0; i < mt; i++) {
        zwork[iz10 + i] = x[i];
    }
    SLC_ZCOPY(&n2, &zwork[iz5], &one_int, &zwork[iz7], &one_int);
    SLC_ZGEMV("N", &n2, &mt_int, &CONE, &zwork[iz6], &n2,
              &zwork[iz10], &one_int, &CONE, &zwork[iz7], &one_int);

    SLC_DCOPY(&n, &dwork[iw7], &one_int, &dwork[iw24], &one_int);
    SLC_DGEMV("N", &n, &mm1, &ONE, &dwork[iw8], &n,
              &dwork[iw22], &one_int, &ONE, &dwork[iw24], &one_int);

    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
            if (i == j) {
                zwork[iz15 + i + i*n] = dlambd*dwork[iw24+i] - zwork[iz7 + i + i*n];
            } else {
                zwork[iz15 + i + j*n] = -zwork[iz7 + i + j*n];
            }
        }
    }
    SLC_ZLACPY("Full", &n, &n, &zwork[iz15], &n, &zwork[iz17], &n);

    lzw = lzwork - izwrk;
    SLC_ZGEES("N", "N", select_dummy, &n, &zwork[iz15], &n, &sdim,
              &zwork[iz16], zwork, &n, &zwork[izwrk], &lzw,
              &dwork[iwrk], bwork, &info2);
    if (info2 > 0) {
        return 6;
    }
    lza = (i32)creal(zwork[izwrk]);
    if (lza > lzamax) lzamax = lza;

    detf = CONE;
    for (i = 0; i < n; i++) {
        detf *= zwork[iz16 + i];
    }

    SLC_ZGETRF(&n, &n, &zwork[iz17], &n, iwork, &info2);
    if (info2 > 0) {
        return 5;
    }
    i32 ldw3 = ldwork - iwrk;
    SLC_ZGETRI(&n, &zwork[iz17], &n, iwork, &zwork[izwrk], &ldw3, &info2);
    lza = (i32)creal(zwork[izwrk]);
    if (lza > lzamax) lzamax = lza;

    for (i = 0; i < m - 1; i++) {
        dwork[iw25 + i] = dwork[iw22 + i] - BETA;
        dwork[iw25 + m - 1 + i] = ALPHA - dwork[iw22 + i];
    }
    if (mr > 0) {
        for (i = 0; i < mr; i++) {
            dwork[iw25 + 2*(m-1) + i] = dwork[iw23 + i] + tau;
            dwork[iw25 + 2*(m-1) + mr + i] = tau - dwork[iw23 + i];
        }
    }
    prod = ONE;
    for (i = 0; i < 2*mt; i++) {
        prod *= dwork[iw25 + i];
    }
    temp = creal(detf);
    if (temp < eps) temp = eps;
    phi = -log(temp) - log(prod);

    for (j = 0; j < mt; j++) {
        for (i = 0; i < n2; i++) {
            zwork[iz18 + i + j*n2] = dlambd*dwork[iw9 + i + j*n2] - zwork[iz6 + i + j*n2];
        }
    }
    SLC_ZGEMV("C", &n2, &mt_int, &CONE, &zwork[iz18], &n2,
              &zwork[iz17], &one_int, &CZERO, &zwork[iz19], &one_int);

    for (i = 0; i < m - 1; i++) {
        dwork[iw26 + i] = ONE/(dwork[iw22 + i] - BETA) - ONE/(ALPHA - dwork[iw22 + i]);
    }
    if (mr > 0) {
        for (i = 0; i < mr; i++) {
            dwork[iw26 + m - 1 + i] = ONE/(dwork[iw23 + i] + tau) - ONE/(tau - dwork[iw23 + i]);
        }
    }
    for (i = 0; i < mt; i++) {
        dwork[iw26 + i] = -creal(zwork[iz19 + i]) - dwork[iw26 + i];
    }

    i32 nmt = n * mt;
    SLC_ZGEMM("N", "N", &n, &nmt, &n, &CONE, &zwork[iz17], &n,
              &zwork[iz18], &n, &CZERO, &zwork[iz20], &n);

    SLC_DLASET("Full", &mt_int, &mt_int, &ZERO, &ZERO, &dwork[iw11], &mt_int);

    for (k = 0; k < mt; k++) {
        SLC_ZCOPY(&n2, &zwork[iz20 + k*n2], &one_int, &zwork[iz22], &one_int);

        for (j = 0; j < n; j++) {
            for (i = 0; i < n; i++) {
                zwork[iz23 + i + j*n] = conj(zwork[iz22 + j + i*n]);
            }
        }

        i32 kp1 = k + 1;
        SLC_ZGEMV("C", &n2, &kp1, &CONE, &zwork[iz20], &n2,
                  &zwork[iz23], &one_int, &CZERO, &zwork[iz24], &one_int);

        for (j = 0; j <= k; j++) {
            dwork[iw11 + k + j*mt] = creal(conj(zwork[iz24 + j]));
        }
    }

    for (i = 0; i < m - 1; i++) {
        f64 diff1 = dwork[iw22 + i] - BETA;
        f64 diff2 = ALPHA - dwork[iw22 + i];
        dwork[iw10 + i] = ONE/(diff1*diff1) + ONE/(diff2*diff2);
    }
    if (mr > 0) {
        for (i = 0; i < mr; i++) {
            f64 sum1 = dwork[iw23 + i] + tau;
            f64 diff1 = tau - dwork[iw23 + i];
            dwork[iw10 + m - 1 + i] = ONE/(sum1*sum1) + ONE/(diff1*diff1);
        }
    }

    for (i = 0; i < mt; i++) {
        dwork[iw11 + i + i*mt] += dwork[iw10 + i];
    }

    for (j = 0; j < mt; j++) {
        for (i = 0; i < j; i++) {
            t1 = dwork[iw11 + i + j*mt];
            t2 = dwork[iw11 + j + i*mt];
            dwork[iw11 + i + j*mt] = t1 + t2;
            dwork[iw11 + j + i*mt] = t1 + t2;
        }
    }

hessian_rcond:
    hnorm = SLC_DLANGE("F", &mt_int, &mt_int, &dwork[iw11], &mt_int, dwork);

    SLC_DLACPY("Full", &mt_int, &mt_int, &dwork[iw11], &mt_int, &dwork[iw31], &mt_int);
    hnorm1 = SLC_DLANGE("1", &mt_int, &mt_int, &dwork[iw31], &mt_int, dwork);
    i32 ldw4 = ldwork - iwrk;
    SLC_DSYTRF("U", &mt_int, &dwork[iw31], &mt_int, iwork, &dwork[iwrk], &ldw4, &info2);
    if (info2 > 0) {
        return 5;
    }
    lwa = (i32)dwork[iwrk];
    if (lwa > lwamax) lwamax = lwa;

    SLC_DSYCON("U", &mt_int, &dwork[iw31], &mt_int, iwork, &hnorm1,
               &rcond, &dwork[iwrk], &iwork[mt], &info2);

    if (rcond < tol3) {
        for (i = 0; i < mt; i++) {
            dwork[iw11 + i + i*mt] += hnorm * regpar;
        }
        goto hessian_rcond;
    }

    SLC_DCOPY(&mt_int, &dwork[iw26], &one_int, &dwork[iw27], &one_int);
    SLC_DSYTRS("U", &mt_int, &one_int, &dwork[iw31], &mt_int, iwork,
               &dwork[iw27], &mt_int, &info2);

    gtest = false;
    for (i = 0; i < mt; i++) {
        dwork[iw20 + i] = x[i] - dwork[iw27 + i];
    }

    SLC_DCOPY(&mm1, &dwork[iw20], &one_int, &dwork[iw22], &one_int);
    if (mr > 0) {
        SLC_DCOPY(&mr, &dwork[iw20 + m - 1], &one_int, &dwork[iw23], &one_int);
    }

    for (i = 0; i < mt; i++) {
        zwork[iz10 + i] = dwork[iw20 + i];
    }
    SLC_ZCOPY(&n2, &zwork[iz5], &one_int, &zwork[iz7], &one_int);
    SLC_ZGEMV("N", &n2, &mt_int, &CONE, &zwork[iz6], &n2,
              &zwork[iz10], &one_int, &CONE, &zwork[iz7], &one_int);

    SLC_DCOPY(&n, &dwork[iw7], &one_int, &dwork[iw24], &one_int);
    SLC_DGEMV("N", &n, &mm1, &ONE, &dwork[iw8], &n,
              &dwork[iw22], &one_int, &ONE, &dwork[iw24], &one_int);

    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
            if (i == j) {
                zwork[iz15 + i + i*n] = dlambd*dwork[iw24+i] - zwork[iz7 + i + i*n];
            } else {
                zwork[iz15 + i + j*n] = -zwork[iz7 + i + j*n];
            }
        }
    }

    lzw = lzwork - izwrk;
    SLC_ZGEES("N", "N", select_dummy, &n, &zwork[iz15], &n, &sdim,
              &zwork[iz16], zwork, &n, &zwork[izwrk], &lzw,
              &dwork[iwrk], bwork, &info2);
    if (info2 > 0) {
        return 6;
    }
    lza = (i32)creal(zwork[izwrk]);
    if (lza > lzamax) lzamax = lza;

    for (i = 0; i < n; i++) {
        dwork[iw30 + i] = creal(zwork[iz16 + i]);
    }
    for (i = 0; i < m - 1; i++) {
        dwork[iw30 + n + i] = dwork[iw22 + i] - BETA;
        dwork[iw30 + n + m - 1 + i] = ALPHA - dwork[iw22 + i];
    }
    if (mr > 0) {
        for (i = 0; i < mr; i++) {
            dwork[iw30 + n + 2*(m-1) + i] = dwork[iw23 + i] + tau;
            dwork[iw30 + n + 2*(m-1) + mr + i] = tau - dwork[iw23 + i];
        }
    }

    emin = dwork[iw30];
    for (i = 0; i < n + 2*mt; i++) {
        if (dwork[iw30 + i] < emin) emin = dwork[iw30 + i];
    }

    if (emin <= ZERO) {
        gtest = false;
    } else {
        pp = SLC_DDOT(&mt_int, &dwork[iw26], &one_int, &dwork[iw27], &one_int);
        prod = ONE;
        for (i = 0; i < n + 2*mt; i++) {
            prod *= dwork[iw30 + i];
        }
        t1 = -log(prod);
        t2 = phi - C2*pp;
        t3 = phi - C4*pp;
        if (t1 >= t3 && t1 < t2) gtest = true;
    }

    pp = SLC_DDOT(&mt_int, &dwork[iw26], &one_int, &dwork[iw27], &one_int);
    delta = sqrt(pp);

    if (gtest || delta <= C3) {
        for (i = 0; i < mt; i++) {
            x[i] = x[i] - dwork[iw27 + i];
        }
    } else {
        for (i = 0; i < mt; i++) {
            x[i] = x[i] - dwork[iw27 + i] / (ONE + delta);
        }
    }

    if (delta < tol5) {
        goto center_found;
    }
    goto newton_loop;

center_found:
    dwork[iw14] = dlambd;
    SLC_DCOPY(&mt_int, x, &one_int, &dwork[iw14+1], &one_int);

    SLC_DCOPY(&mt_int, &dwork[iw14], &one_int, &dwork[iw15], &one_int);
    i32 mtp1 = mt + 1;
    SLC_DCOPY(&mtp1, &dwork[iw14], &one_int, &dwork[iw15], &one_int);

    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
            zwork[iz21 + i + j*n] = dwork[iw24 + i] * conj(zwork[iz17 + j + i*n]);
        }
    }
    SLC_ZGEMV("C", &n2, &mt_int, &CONE, &zwork[iz20], &n2,
              &zwork[iz21], &one_int, &CZERO, &zwork[iz24], &one_int);

    for (i = 0; i < mt; i++) {
        dwork[iw32 + i] = creal(zwork[iz24 + i]);
    }

    SLC_DLACPY("Full", &mt_int, &mt_int, &dwork[iw11], &mt_int, &dwork[iw31], &mt_int);
    i32 ldw5 = ldwork - iwrk;
    SLC_DSYSV("U", &mt_int, &one_int, &dwork[iw31], &mt_int, iwork,
              &dwork[iw32], &mt_int, &dwork[iwrk], &ldw5, &info2);
    if (info2 > 0) {
        return 5;
    }
    lwa = (i32)dwork[iwrk];
    if (lwa > lwamax) lwamax = lwa;

    hn = SLC_DLANGE("F", &mt_int, &one_int, &dwork[iw32], &mt_int, dwork);

    dwork[iw13] = dlambd - c / hn;
    for (i = 0; i < mt; i++) {
        dwork[iw13 + 1 + i] = x[i] + c * dwork[iw32 + i] / hn;
    }

    SLC_DCOPY(&mm1, &dwork[iw13+1], &one_int, &dwork[iw22], &one_int);
    if (mr > 0) {
        SLC_DCOPY(&mr, &dwork[iw13 + m], &one_int, &dwork[iw23], &one_int);
    }

    for (i = 0; i < mt; i++) {
        zwork[iz10 + i] = dwork[iw13 + 1 + i];
    }
    SLC_ZCOPY(&n2, &zwork[iz5], &one_int, &zwork[iz7], &one_int);
    SLC_ZGEMV("N", &n2, &mt_int, &CONE, &zwork[iz6], &n2,
              &zwork[iz10], &one_int, &CONE, &zwork[iz7], &one_int);

    SLC_DCOPY(&n, &dwork[iw7], &one_int, &dwork[iw24], &one_int);
    SLC_DGEMV("N", &n, &mm1, &ONE, &dwork[iw8], &n,
              &dwork[iw22], &one_int, &ONE, &dwork[iw24], &one_int);

    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
            if (i == j) {
                zwork[iz15 + i + i*n] = dwork[iw13]*dwork[iw24+i] - zwork[iz7 + i + i*n];
            } else {
                zwork[iz15 + i + j*n] = -zwork[iz7 + i + j*n];
            }
        }
    }

    lzw = lzwork - izwrk;
    SLC_ZGEES("N", "N", select_dummy, &n, &zwork[iz15], &n, &sdim,
              &zwork[iz16], zwork, &n, &zwork[izwrk], &lzw,
              &dwork[iwrk], bwork, &info2);
    if (info2 > 0) {
        return 6;
    }
    lza = (i32)creal(zwork[izwrk]);
    if (lza > lzamax) lzamax = lza;

    emin = creal(zwork[iz16]);
    for (i = 1; i < n; i++) {
        if (creal(zwork[iz16 + i]) < emin) {
            emin = creal(zwork[iz16 + i]);
        }
    }

    pos = true;
    for (i = 0; i < m - 1; i++) {
        dwork[iw25 + i] = dwork[iw22 + i] - BETA;
        dwork[iw25 + m - 1 + i] = ALPHA - dwork[iw22 + i];
    }
    if (mr > 0) {
        for (i = 0; i < mr; i++) {
            dwork[iw25 + 2*(m-1) + i] = dwork[iw23 + i] + tau;
            dwork[iw25 + 2*(m-1) + mr + i] = tau - dwork[iw23 + i];
        }
    }
    temp = dwork[iw25];
    for (i = 1; i < 2*mt; i++) {
        if (dwork[iw25 + i] < temp) temp = dwork[iw25 + i];
    }
    if (temp <= ZERO || emin <= ZERO) pos = false;

pos_loop:
    if (pos) {
        SLC_DCOPY(&mtp1, &dwork[iw13], &one_int, &dwork[iw17], &one_int);

        for (i = 0; i < mt + 1; i++) {
            dwork[iw13 + i] = dwork[iw13 + i] + C5*(dwork[iw13 + i] - dwork[iw15 + i]);
        }

        SLC_DCOPY(&mm1, &dwork[iw13+1], &one_int, &dwork[iw22], &one_int);
        if (mr > 0) {
            SLC_DCOPY(&mr, &dwork[iw13 + m], &one_int, &dwork[iw23], &one_int);
        }

        for (i = 0; i < mt; i++) {
            zwork[iz10 + i] = dwork[iw13 + 1 + i];
        }
        SLC_ZCOPY(&n2, &zwork[iz5], &one_int, &zwork[iz7], &one_int);
        SLC_ZGEMV("N", &n2, &mt_int, &CONE, &zwork[iz6], &n2,
                  &zwork[iz10], &one_int, &CONE, &zwork[iz7], &one_int);

        SLC_DCOPY(&n, &dwork[iw7], &one_int, &dwork[iw24], &one_int);
        SLC_DGEMV("N", &n, &mm1, &ONE, &dwork[iw8], &n,
                  &dwork[iw22], &one_int, &ONE, &dwork[iw24], &one_int);

        SLC_DCOPY(&mtp1, &dwork[iw17], &one_int, &dwork[iw15], &one_int);

        for (j = 0; j < n; j++) {
            for (i = 0; i < n; i++) {
                if (i == j) {
                    zwork[iz15 + i + i*n] = dwork[iw13]*dwork[iw24+i] - zwork[iz7 + i + i*n];
                } else {
                    zwork[iz15 + i + j*n] = -zwork[iz7 + i + j*n];
                }
            }
        }

        lzw = lzwork - izwrk;
        SLC_ZGEES("N", "N", select_dummy, &n, &zwork[iz15], &n, &sdim,
                  &zwork[iz16], zwork, &n, &zwork[izwrk], &lzw,
                  &dwork[iwrk], bwork, &info2);
        if (info2 > 0) {
            return 6;
        }
        lza = (i32)creal(zwork[izwrk]);
        if (lza > lzamax) lzamax = lza;

        emin = creal(zwork[iz16]);
        for (i = 1; i < n; i++) {
            if (creal(zwork[iz16 + i]) < emin) {
                emin = creal(zwork[iz16 + i]);
            }
        }

        pos = true;
        for (i = 0; i < m - 1; i++) {
            dwork[iw25 + i] = dwork[iw22 + i] - BETA;
            dwork[iw25 + m - 1 + i] = ALPHA - dwork[iw22 + i];
        }
        if (mr > 0) {
            for (i = 0; i < mr; i++) {
                dwork[iw25 + 2*(m-1) + i] = dwork[iw23 + i] + tau;
                dwork[iw25 + 2*(m-1) + mr + i] = tau - dwork[iw23 + i];
            }
        }
        temp = dwork[iw25];
        for (i = 1; i < 2*mt; i++) {
            if (dwork[iw25 + i] < temp) temp = dwork[iw25 + i];
        }
        if (temp <= ZERO || emin <= ZERO) pos = false;
        goto pos_loop;
    }

bisection_loop:
    for (i = 0; i < mt + 1; i++) {
        dwork[iw16 + i] = (dwork[iw13 + i] + dwork[iw15 + i]) / TWO;
    }

    SLC_DCOPY(&mm1, &dwork[iw16+1], &one_int, &dwork[iw22], &one_int);
    if (mr > 0) {
        SLC_DCOPY(&mr, &dwork[iw16 + m], &one_int, &dwork[iw23], &one_int);
    }

    for (i = 0; i < mt; i++) {
        zwork[iz10 + i] = dwork[iw16 + 1 + i];
    }
    SLC_ZCOPY(&n2, &zwork[iz5], &one_int, &zwork[iz7], &one_int);
    SLC_ZGEMV("N", &n2, &mt_int, &CONE, &zwork[iz6], &n2,
              &zwork[iz10], &one_int, &CONE, &zwork[iz7], &one_int);

    SLC_DCOPY(&n, &dwork[iw7], &one_int, &dwork[iw24], &one_int);
    SLC_DGEMV("N", &n, &mm1, &ONE, &dwork[iw8], &n,
              &dwork[iw22], &one_int, &ONE, &dwork[iw24], &one_int);

    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
            if (i == j) {
                zwork[iz15 + i + i*n] = dwork[iw16]*dwork[iw24+i] - zwork[iz7 + i + i*n];
            } else {
                zwork[iz15 + i + j*n] = -zwork[iz7 + i + j*n];
            }
        }
    }

    lzw = lzwork - izwrk;
    SLC_ZGEES("N", "N", select_dummy, &n, &zwork[iz15], &n, &sdim,
              &zwork[iz16], zwork, &n, &zwork[izwrk], &lzw,
              &dwork[iwrk], bwork, &info2);
    if (info2 > 0) {
        return 6;
    }
    lza = (i32)creal(zwork[izwrk]);
    if (lza > lzamax) lzamax = lza;

    emin = creal(zwork[iz16]);
    for (i = 1; i < n; i++) {
        if (creal(zwork[iz16 + i]) < emin) {
            emin = creal(zwork[iz16 + i]);
        }
    }

    pos = true;
    for (i = 0; i < m - 1; i++) {
        dwork[iw25 + i] = dwork[iw22 + i] - BETA;
        dwork[iw25 + m - 1 + i] = ALPHA - dwork[iw22 + i];
    }
    if (mr > 0) {
        for (i = 0; i < mr; i++) {
            dwork[iw25 + 2*(m-1) + i] = dwork[iw23 + i] + tau;
            dwork[iw25 + 2*(m-1) + mr + i] = tau - dwork[iw23 + i];
        }
    }
    temp = dwork[iw25];
    for (i = 1; i < 2*mt; i++) {
        if (dwork[iw25 + i] < temp) temp = dwork[iw25 + i];
    }
    if (temp <= ZERO || emin <= ZERO) pos = false;

    if (pos) {
        SLC_DCOPY(&mtp1, &dwork[iw16], &one_int, &dwork[iw15], &one_int);
    } else {
        SLC_DCOPY(&mtp1, &dwork[iw16], &one_int, &dwork[iw13], &one_int);
    }

    for (i = 0; i < mt + 1; i++) {
        dwork[iw33 + i] = dwork[iw13 + i] - dwork[iw15 + i];
    }
    ynorm1 = SLC_DLANGE("F", &mtp1, &one_int, &dwork[iw33], &mtp1, dwork);

    for (i = 0; i < mt + 1; i++) {
        dwork[iw33 + i] = dwork[iw13 + i] - dwork[iw14 + i];
    }
    ynorm2 = SLC_DLANGE("F", &mtp1, &one_int, &dwork[iw33], &mtp1, dwork);

    if (ynorm1 >= ynorm2 * THETA) {
        goto bisection_loop;
    }

    for (i = 0; i < mt + 1; i++) {
        dwork[iw33 + i] = dwork[iw15 + i] - dwork[iw14 + i];
    }
    c = SLC_DLANGE("F", &mtp1, &one_int, &dwork[iw33], &mtp1, dwork);

    SLC_DCOPY(&mt_int, &dwork[iw15+1], &one_int, x, &one_int);

    goto iter_loop;
}
