// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"

void mb04di(const char *job, const char *sgn, i32 n, i32 ilo,
            const f64 *scale, i32 m, f64 *v1, i32 ldv1,
            f64 *v2, i32 ldv2, i32 *info) {

    const f64 NEG_ONE = -1.0;

    bool lperm = (*job == 'P' || *job == 'p' || *job == 'B' || *job == 'b');
    bool lscal = (*job == 'S' || *job == 's' || *job == 'B' || *job == 'b');
    bool lsgn = (*sgn == 'N' || *sgn == 'n');

    *info = 0;

    if (!lperm && !lscal && !(*job == 'N' || *job == 'n')) {
        *info = -1;
    } else if (!lsgn && !(*sgn == 'P' || *sgn == 'p')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ilo < 1 || ilo > n + 1) {
        *info = -4;
    } else if (m < 0) {
        *info = -6;
    } else if (ldv1 < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (ldv2 < (n > 1 ? n : 1)) {
        *info = -10;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0 || (*job == 'N' || *job == 'n')) {
        return;
    }

    if (lscal) {
        for (i32 i = ilo - 1; i < n; i++) {
            f64 s = scale[i];
            SLC_DSCAL(&m, &s, &v1[i], &ldv1);
        }
        for (i32 i = ilo - 1; i < n; i++) {
            f64 s = scale[i];
            SLC_DRSCL(&m, &s, &v2[i], &ldv2);
        }
    }

    if (lperm) {
        for (i32 i = ilo - 2; i >= 0; i--) {
            i32 k = (i32)scale[i];
            bool sysw = (k > n);
            if (sysw) {
                k = k - n;
            }

            k = k - 1;

            if (k < 0 || k >= n) {
                continue;
            }

            if (k != i) {
                SLC_DSWAP(&m, &v1[i], &ldv1, &v1[k], &ldv1);
                SLC_DSWAP(&m, &v2[i], &ldv2, &v2[k], &ldv2);
            }

            if (sysw) {
                SLC_DSWAP(&m, &v1[k], &ldv1, &v2[k], &ldv2);
                if (lsgn) {
                    SLC_DSCAL(&m, &NEG_ONE, &v1[k], &ldv1);
                } else {
                    SLC_DSCAL(&m, &NEG_ONE, &v2[k], &ldv2);
                }
            }
        }
    }
}
