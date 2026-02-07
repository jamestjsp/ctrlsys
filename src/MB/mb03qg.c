/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb03qg(const char *dico, const char *stdom, const char *jobu, const char *jobv,
            i32 n, i32 nlow, i32 nsup, f64 alpha,
            f64 *a, i32 lda, f64 *e, i32 lde,
            f64 *u, i32 ldu, f64 *v, i32 ldv,
            i32 *ndim, f64 *dwork, i32 ldwork, i32 *info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    bool discr, lstdom, lquery;
    i32 ib, l, lm1, minwrk, nup;
    f64 alphai[2], alphar[2], beta[2];
    f64 tlambd, tole, x;

    i32 maxn = (n > 1) ? n : 1;

    *info = 0;
    discr = (dico[0] == 'D' || dico[0] == 'd');
    lstdom = (stdom[0] == 'S' || stdom[0] == 's');

    if (!(dico[0] == 'C' || dico[0] == 'c' || discr)) {
        *info = -1;
    } else if (!(lstdom || stdom[0] == 'U' || stdom[0] == 'u')) {
        *info = -2;
    } else if (!(jobu[0] == 'I' || jobu[0] == 'i' ||
                 jobu[0] == 'U' || jobu[0] == 'u')) {
        *info = -3;
    } else if (!(jobv[0] == 'I' || jobv[0] == 'i' ||
                 jobv[0] == 'V' || jobv[0] == 'v' ||
                 jobv[0] == 'U' || jobv[0] == 'u')) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (nlow < 0) {
        *info = -6;
    } else if (nsup < nlow || n < nsup) {
        *info = -7;
    } else if (discr && alpha < ZERO) {
        *info = -8;
    } else if (lda < maxn) {
        *info = -10;
    } else if (lde < maxn) {
        *info = -12;
    } else if (ldu < maxn) {
        *info = -14;
    } else if (ldv < maxn) {
        *info = -16;
    } else {
        lquery = (ldwork == -1);
        if (n <= 1) {
            minwrk = 1;
        } else {
            minwrk = 4 * n + 16;
        }
        if (ldwork < minwrk && !lquery) {
            *info = -19;
        }
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("MB03QG", &neginfo);
        return;
    } else if (lquery) {
        dwork[0] = (f64)minwrk;
        return;
    }

    *ndim = 0;
    if (nsup == 0) {
        return;
    }

    if (nlow > 1) {
        if (a[(nlow - 1) + (nlow - 2) * lda] != ZERO) {
            *info = 1;
        }
    }
    if (nsup < n) {
        if (a[nsup + (nsup - 1) * lda] != ZERO) {
            *info = 1;
        }
    }
    if (*info != 0) {
        return;
    }

    if (jobu[0] == 'I' || jobu[0] == 'i') {
        SLC_DLASET("Full", &n, &n, &ZERO, &ONE, u, &ldu);
    }

    if (jobv[0] == 'I' || jobv[0] == 'i') {
        SLC_DLASET("Full", &n, &n, &ZERO, &ONE, v, &ldv);
    }

    tole = SLC_DLAMCH("Epsilon") *
           SLC_DLANTR("1", "Upper", "Non-unit", &n, &n, e, &lde, dwork);

    l = nsup;
    nup = nsup;

    while (l >= nlow) {
        ib = 1;
        if (l > nlow) {
            lm1 = l - 1;
            if (a[(l - 1) + (lm1 - 1) * lda] != ZERO) {
                i32 mb03qw_info;
                mb03qw(n, lm1, a, lda, e, lde, u, ldu, v, ldv,
                       alphar, alphai, beta, &mb03qw_info);
                if (a[(l - 1) + (lm1 - 1) * lda] != ZERO) {
                    ib = 2;
                }
            }
        }

        if (discr) {
            if (ib == 1) {
                tlambd = fabs(a[(l - 1) + (l - 1) * lda]);
                x = fabs(e[(l - 1) + (l - 1) * lde]);
            } else {
                tlambd = SLC_DLAPY2(&alphar[0], &alphai[0]);
                x = fabs(beta[0]);
            }
        } else {
            if (ib == 1) {
                x = e[(l - 1) + (l - 1) * lde];
                if (x < ZERO) {
                    tlambd = -a[(l - 1) + (l - 1) * lda];
                    x = -x;
                } else {
                    tlambd = a[(l - 1) + (l - 1) * lda];
                }
            } else {
                tlambd = alphar[0];
                x = beta[0];
                if (x < ZERO) {
                    tlambd = -tlambd;
                    x = -x;
                }
            }
        }

        if ((lstdom && tlambd < alpha * x && x > tole) ||
            (!lstdom && tlambd > alpha * x)) {
            *ndim = *ndim + ib;
            l = l - ib;
        } else {
            if (*ndim != 0) {
                i32 wantq = 1;
                i32 wantz = 1;
                i32 dtgexc_info;
                SLC_DTGEXC(&wantq, &wantz, &n, a, &lda, e, &lde, u, &ldu,
                           v, &ldv, &l, &nup, dwork, &ldwork, &dtgexc_info);
                if (dtgexc_info != 0) {
                    *info = 2;
                    return;
                }
                nup = nup - 1;
                l = l - 1;
            } else {
                nup = nup - ib;
                l = l - ib;
            }
        }
    }

    dwork[0] = (f64)minwrk;
}
