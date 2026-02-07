/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

void mb04db(const char *job, const char *sgn, i32 n, i32 ilo,
            const f64 *lscale, const f64 *rscale, i32 m,
            f64 *v1, i32 ldv1, f64 *v2, i32 ldv2, i32 *info)
{
    char j = job[0];
    char s = sgn[0];

    bool lperm = (j == 'P' || j == 'p' || j == 'B' || j == 'b');
    bool lscal = (j == 'S' || j == 's' || j == 'B' || j == 'b');
    bool lsgn = (s == 'N' || s == 'n');

    f64 neg_one = -1.0;

    *info = 0;

    if (!lperm && !lscal && !(j == 'N' || j == 'n')) {
        *info = -1;
    } else if (!lsgn && !(s == 'P' || s == 'p')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ilo < 1 || ilo > n + 1) {
        *info = -4;
    } else if (m < 0) {
        *info = -7;
    } else if (ldv1 < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (ldv2 < (n > 1 ? n : 1)) {
        *info = -11;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0 || (j == 'N' || j == 'n')) {
        return;
    }

    if (lscal) {
        for (i32 i = ilo - 1; i < n; i++) {
            SLC_DRSCL(&m, &lscale[i], &v1[i], &ldv1);
        }
        for (i32 i = ilo - 1; i < n; i++) {
            SLC_DRSCL(&m, &rscale[i], &v2[i], &ldv2);
        }
    }

    if (lperm) {
        for (i32 i = ilo - 2; i >= 0; i--) {
            i32 k = (i32)lscale[i];
            bool sysw = k > n;
            if (sysw) {
                k = k - n;
            }

            if (k - 1 != i) {
                SLC_DSWAP(&m, &v1[i], &ldv1, &v1[k - 1], &ldv1);
                SLC_DSWAP(&m, &v2[i], &ldv2, &v2[k - 1], &ldv2);
            }

            if (sysw) {
                SLC_DSWAP(&m, &v1[k - 1], &ldv1, &v2[k - 1], &ldv2);

                if (lsgn) {
                    SLC_DSCAL(&m, &neg_one, &v1[k - 1], &ldv1);
                } else {
                    SLC_DSCAL(&m, &neg_one, &v2[k - 1], &ldv2);
                }
            }
        }
    }
}
