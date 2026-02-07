// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>

void mb4dbz(const char *job, const char *sgn, i32 n, i32 ilo,
            const f64 *lscale, const f64 *rscale, i32 m,
            c128 *v1, i32 ldv1, c128 *v2, i32 ldv2, i32 *info) {

    const c128 NEG_ONE = -1.0 + 0.0*I;

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
        *info = -7;
    } else if (ldv1 < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (ldv2 < (n > 1 ? n : 1)) {
        *info = -11;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0 || (*job == 'N' || *job == 'n')) {
        return;
    }

    // Inverse scaling
    if (lscal) {
        for (i32 i = ilo - 1; i < n; i++) {
            SLC_ZDRSCL(&m, &lscale[i], &v1[i], &ldv1);
        }
        for (i32 i = ilo - 1; i < n; i++) {
            SLC_ZDRSCL(&m, &rscale[i], &v2[i], &ldv2);
        }
    }

    // Inverse permutation
    if (lperm) {
        for (i32 i = ilo - 2; i >= 0; i--) {
            i32 k = (i32)lscale[i];
            bool sysw = k > n;
            if (sysw) {
                k = k - n;
            }

            // Convert k from 1-based to 0-based
            k = k - 1;

            if (k >= 0 && k < n && k != i) {
                // Exchange rows k <-> i
                SLC_ZSWAP(&m, &v1[i], &ldv1, &v1[k], &ldv1);
                SLC_ZSWAP(&m, &v2[i], &ldv2, &v2[k], &ldv2);
            }

            if (sysw && k >= 0 && k < n) {
                // Exchange V1(k,:) <-> V2(k,:)
                SLC_ZSWAP(&m, &v1[k], &ldv1, &v2[k], &ldv2);

                if (lsgn) {
                    SLC_ZSCAL(&m, &NEG_ONE, &v1[k], &ldv1);
                } else {
                    SLC_ZSCAL(&m, &NEG_ONE, &v2[k], &ldv2);
                }
            }
        }
    }
}
