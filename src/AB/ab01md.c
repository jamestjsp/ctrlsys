/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <ctype.h>
#include <stddef.h>

void ab01md(const char* jobz, i32 n, f64* a, i32 lda, f64* b, i32* ncont,
            f64* z, i32 ldz, f64* tau, f64 tol, f64* dwork, i32 ldwork,
            i32* info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    char jobz_upper = (char)toupper((unsigned char)jobz[0]);
    bool ljobf = (jobz_upper == 'F');
    bool ljobi = (jobz_upper == 'I');
    bool ljobz = ljobf || ljobi;

    *info = 0;
    *ncont = 0;

    if (!ljobz && jobz_upper != 'N') {
        *info = -1;
        return;
    }
    if (n < 0) {
        *info = -2;
        return;
    }
    i32 max1n = (1 > n) ? 1 : n;
    if (lda < max1n) {
        *info = -4;
        return;
    }
    if (!ljobz && ldz < 1) {
        *info = -8;
        return;
    }
    if (ljobz && ldz < max1n) {
        *info = -8;
        return;
    }
    if (ldwork < max1n) {
        *info = -12;
        return;
    }

    dwork[0] = ONE;
    if (n == 0) {
        return;
    }

    f64 wrkopt = ONE;

    f64 anorm = SLC_DLANGE("M", &n, &n, a, &lda, dwork);
    i32 int1 = 1;
    f64 bnorm = SLC_DLANGE("M", &n, &int1, b, &n, dwork);

    if (bnorm == ZERO) {
        if (ljobf) {
            SLC_DLASET("F", &n, &n, &ZERO, &ZERO, z, &ldz);
            SLC_DLASET("F", &n, &int1, &ZERO, &ZERO, tau, &n);
        } else if (ljobi) {
            SLC_DLASET("F", &n, &n, &ZERO, &ONE, z, &ldz);
        }
        return;
    }

    i32 nbl = 0;
    i32 info_tmp = 0;
    mb01pd("S", "G", n, n, 0, 0, anorm, nbl, NULL, a, lda, &info_tmp);
    mb01pd("S", "G", n, 1, 0, 0, bnorm, nbl, NULL, b, n, &info_tmp);

    f64 fanorm = SLC_DLANGE("F", &n, &n, a, &lda, dwork);
    f64 fbnorm = SLC_DLANGE("1", &n, &int1, b, &n, dwork);

    f64 toldef = tol;
    f64 thresh = 0.0;
    if (toldef <= ZERO) {
        thresh = (f64)n * SLC_DLAMCH("E");
        toldef = thresh * (fanorm > fbnorm ? fanorm : fbnorm);
    }

    i32 itau = 0;
    f64 b1 = 0.0;

    if (fbnorm > toldef) {
        if (n > 1) {
            f64 h;
            SLC_DLARFG(&n, &b[0], &b[1], &int1, &h);

            b1 = b[0];
            b[0] = ONE;

            SLC_DLARF("R", &n, &n, b, &int1, &h, a, &lda, dwork);
            SLC_DLARF("L", &n, &n, b, &int1, &h, a, &lda, dwork);

            b[0] = b1;
            tau[0] = h;
            itau = 1;
        } else {
            b1 = b[0];
        }

        SLC_DGEHRD(&n, &int1, &n, a, &lda, &tau[itau], dwork, &ldwork, &info_tmp);
        wrkopt = dwork[0];

        if (ljobz) {
            if (n > 1) {
                i32 nm1 = n - 1;
                SLC_DLACPY("F", &nm1, &int1, &b[1], &nm1, &z[1], &ldz);
            }
            if (n > 2) {
                i32 nm2 = n - 2;
                SLC_DLACPY("L", &nm2, &nm2, &a[2], &lda, &z[2 + ldz], &ldz);
            }
            if (ljobi) {
                if (n == 1) {
                    z[0] = ONE;
                } else {
                    SLC_DORGQR(&n, &n, &n, z, &ldz, tau, dwork, &ldwork, &info_tmp);
                    f64 newopt = dwork[0];
                    wrkopt = (wrkopt > newopt) ? wrkopt : newopt;
                }
            }
        }

        if (n > 2) {
            i32 nm2 = n - 2;
            SLC_DLASET("L", &nm2, &nm2, &ZERO, &ZERO, &a[2], &lda);
        }
        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_DLASET("F", &nm1, &int1, &ZERO, &ZERO, &b[1], &nm1);
        }

        if (tol <= ZERO) {
            f64 abs_b1 = fabs(b1);
            toldef = thresh * (fanorm > abs_b1 ? fanorm : abs_b1);
        }

        i32 j = 0;
        while (j < n - 1) {
            if (fabs(a[(j + 1) + j * lda]) > toldef) {
                j++;
            } else {
                break;
            }
        }

        *ncont = j + 1;
        if (j < n - 1) {
            a[(j + 1) + j * lda] = ZERO;
        }

        mb01pd("U", "H", *ncont, *ncont, 0, 0, anorm, nbl, NULL, a, lda, &info_tmp);
        mb01pd("U", "G", 1, 1, 0, 0, bnorm, nbl, NULL, b, n, &info_tmp);
        if (*ncont < n) {
            i32 n_minus_ncont = n - *ncont;
            mb01pd("U", "G", n, n_minus_ncont, 0, 0, anorm, nbl, NULL,
                   &a[(*ncont) * lda], lda, &info_tmp);
        }
    } else {
        if (ljobf) {
            SLC_DLASET("F", &n, &n, &ZERO, &ZERO, z, &ldz);
            SLC_DLASET("F", &n, &int1, &ZERO, &ZERO, tau, &n);
        } else if (ljobi) {
            SLC_DLASET("F", &n, &n, &ZERO, &ONE, z, &ldz);
        }
        mb01pd("U", "G", n, n, 0, 0, anorm, nbl, NULL, a, lda, &info_tmp);
        mb01pd("U", "G", n, 1, 0, 0, bnorm, nbl, NULL, b, n, &info_tmp);
    }

    dwork[0] = wrkopt;
}
