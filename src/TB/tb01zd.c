/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

static bool lsame(char ca, char cb) {
    if (ca >= 'a' && ca <= 'z') ca -= 32;
    if (cb >= 'a' && cb <= 'z') cb -= 32;
    return ca == cb;
}

static i32 imax(i32 a, i32 b) {
    return a > b ? a : b;
}

void tb01zd(const char* jobz, i32 n, i32 p,
            f64* a, i32 lda, f64* b,
            f64* c, i32 ldc, i32* ncont,
            f64* z, i32 ldz, f64* tau, f64 tol,
            f64* dwork, i32 ldwork, i32* info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool ljobf, ljobi, ljobz;
    i32 itau, j;
    f64 anorm, bnorm, fanorm, fbnorm, h, b1, thresh, toldef, wrkopt;
    i32 nblk_dummy[1] = {0};

    const i32 int1 = 1;

    *info = 0;
    ljobf = lsame(*jobz, 'F');
    ljobi = lsame(*jobz, 'I');
    ljobz = ljobf || ljobi;

    if (!ljobz && !lsame(*jobz, 'N')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (p < 0) {
        *info = -3;
    } else if (lda < imax(1, n)) {
        *info = -5;
    } else if (ldc < imax(1, p)) {
        *info = -8;
    } else if (ldz < 1 || (ljobz && ldz < n)) {
        *info = -11;
    } else if (ldwork < imax(1, imax(n, p))) {
        *info = -15;
    }

    if (*info != 0) {
        return;
    }

    *ncont = 0;
    dwork[0] = ONE;
    if (n == 0) {
        return;
    }

    wrkopt = ONE;

    anorm = SLC_DLANGE("M", &n, &n, a, &lda, dwork);
    bnorm = SLC_DLANGE("M", &n, &int1, b, &n, dwork);

    if (bnorm == ZERO) {
        if (ljobf) {
            SLC_DLASET("F", &n, &n, &ZERO, &ZERO, z, &ldz);
            SLC_DLASET("F", &n, &int1, &ZERO, &ZERO, tau, &n);
        } else if (ljobi) {
            SLC_DLASET("F", &n, &n, &ZERO, &ONE, z, &ldz);
        }
        return;
    }

    mb01pd("S", "G", n, n, 0, 0, anorm, 0, nblk_dummy, a, lda, info);
    mb01pd("S", "G", n, 1, 0, 0, bnorm, 0, nblk_dummy, b, n, info);

    fanorm = SLC_DLANGE("F", &n, &n, a, &lda, dwork);
    fbnorm = SLC_DLANGE("1", &n, &int1, b, &n, dwork);

    toldef = tol;
    if (toldef <= ZERO) {
        thresh = (f64)n * SLC_DLAMCH("E");
        toldef = thresh * (fanorm > fbnorm ? fanorm : fbnorm);
    } else {
        thresh = (f64)n * SLC_DLAMCH("E");
    }

    itau = 0;
    if (fbnorm > toldef) {
        if (n > 1) {
            SLC_DLARFG(&n, b, &b[1], &int1, &h);

            b1 = b[0];
            b[0] = ONE;

            SLC_DLARF("R", &n, &n, b, &int1, &h, a, &lda, dwork);
            SLC_DLARF("L", &n, &n, b, &int1, &h, a, &lda, dwork);

            SLC_DLARF("R", &p, &n, b, &int1, &h, c, &ldc, dwork);

            b[0] = b1;
            tau[0] = h;
            itau = 1;
        } else {
            b1 = b[0];
            tau[0] = ZERO;
        }

        i32 ilo = 1;
        i32 ihi = n;
        SLC_DGEHRD(&n, &ilo, &ihi, a, &lda, &tau[itau], dwork, &ldwork, info);
        wrkopt = dwork[0];

        SLC_DORMHR("R", "N", &p, &n, &ilo, &ihi, a, &lda, &tau[itau], c, &ldc, dwork, &ldwork, info);
        wrkopt = wrkopt > dwork[0] ? wrkopt : dwork[0];

        if (ljobz) {
            if (n > 1) {
                i32 n_m1 = n - 1;
                SLC_DLACPY("F", &n_m1, &int1, &b[1], &n_m1, &z[1], &ldz);
            }
            if (n > 2) {
                i32 n_m2 = n - 2;
                SLC_DLACPY("L", &n_m2, &n_m2, &a[2], &lda, &z[2 + ldz], &ldz);
            }
            if (ljobi) {
                SLC_DORGQR(&n, &n, &n, z, &ldz, tau, dwork, &ldwork, info);
                wrkopt = wrkopt > dwork[0] ? wrkopt : dwork[0];
            }
        }

        if (n > 2) {
            i32 n_m2 = n - 2;
            SLC_DLASET("L", &n_m2, &n_m2, &ZERO, &ZERO, &a[2], &lda);
        }
        if (n > 1) {
            i32 n_m1 = n - 1;
            SLC_DLASET("F", &n_m1, &int1, &ZERO, &ZERO, &b[1], &n_m1);
        }

        if (tol <= ZERO) {
            toldef = thresh * (fanorm > fabs(b1) ? fanorm : fabs(b1));
        }

        j = 0;
        while (j < n - 1) {
            if (fabs(a[(j + 1) + j * lda]) > toldef) {
                j++;
            } else {
                break;
            }
        }
        *ncont = j + 1;

        if (*ncont < n) {
            a[*ncont + (*ncont - 1) * lda] = ZERO;
        }

        mb01pd("U", "H", *ncont, *ncont, 0, 0, anorm, 0, nblk_dummy, a, lda, info);
        mb01pd("U", "G", 1, 1, 0, 0, bnorm, 0, nblk_dummy, b, n, info);
        if (*ncont < n) {
            i32 n_minus_ncont = n - *ncont;
            mb01pd("U", "G", n, n_minus_ncont, 0, 0, anorm, 0, nblk_dummy, &a[*ncont * lda], lda, info);
        }
    } else {
        mb01pd("U", "G", n, n, 0, 0, anorm, 0, nblk_dummy, a, lda, info);
        mb01pd("U", "G", n, 1, 0, 0, bnorm, 0, nblk_dummy, b, n, info);
        if (ljobf) {
            SLC_DLASET("F", &n, &n, &ZERO, &ZERO, z, &ldz);
            SLC_DLASET("F", &n, &int1, &ZERO, &ZERO, tau, &n);
        } else if (ljobi) {
            SLC_DLASET("F", &n, &n, &ZERO, &ONE, z, &ldz);
        }
    }

    dwork[0] = wrkopt;
}
