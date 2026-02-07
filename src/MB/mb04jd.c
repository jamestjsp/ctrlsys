// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb04jd(i32 n, i32 m, i32 p, i32 l, f64 *a, i32 lda, f64 *b, i32 ldb,
            f64 *tau, f64 *dwork, i32 ldwork, i32 *info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    *info = 0;

    i32 minwork = 1;
    if (n > 1 && n - 1 > minwork) minwork = n - 1;
    if (n > p && n - p > minwork) minwork = n - p;
    if (l > minwork) minwork = l;

    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (p < 0) {
        *info = -3;
    } else if (l < 0) {
        *info = -4;
    } else if (lda < (n > 0 ? n : 1)) {
        *info = -6;
    } else if (ldb < (l > 0 ? l : 1)) {
        *info = -8;
    } else if (ldwork < minwork) {
        *info = -11;
    }

    if (*info != 0) {
        return;
    }

    i32 minval = (m < n) ? m : n;
    if (minval == 0) {
        dwork[0] = one;
        return;
    } else if (m <= p + 1) {
        for (i32 i = 0; i < minval; i++) {
            tau[i] = zero;
        }
        dwork[0] = one;
        return;
    }

    f64 wrkopt = (n > 1) ? (f64)(n - 1) : one;
    if ((f64)l > wrkopt) wrkopt = (f64)l;

    i32 pmin = (p < n) ? p : n;

    for (i32 i = 0; i < pmin; i++) {
        i32 vec_len = m - p;
        i32 offset = i + i * lda;

        SLC_DLARFG(&vec_len, &a[offset], &a[offset + lda], &lda, &tau[i]);

        if (tau[i] != zero) {
            f64 first = a[offset];
            a[offset] = one;

            if (i < n - 1) {
                i32 nrows = n - i - 1;
                SLC_DLARF("Right", &nrows, &vec_len, &a[offset], &lda,
                          &tau[i], &a[offset + 1], &lda, dwork);
            }

            if (l > 0) {
                SLC_DLARF("Right", &l, &vec_len, &a[offset], &lda,
                          &tau[i], &b[i * ldb], &ldb, dwork);
            }

            a[offset] = first;
        }
    }

    if (n > p) {
        i32 nrows = n - p;
        i32 ncols = m - p;
        i32 offset = p + p * lda;
        i32 info_lqf = 0;

        SLC_DGELQF(&nrows, &ncols, &a[offset], &lda, &tau[p], dwork, &ldwork,
                   &info_lqf);
        if (dwork[0] > wrkopt) wrkopt = dwork[0];

        if (l > 0) {
            i32 minval2 = minval - p;
            i32 info_lq = 0;

            SLC_DORMLQ("Right", "Transpose", &l, &ncols, &minval2,
                       &a[offset], &lda, &tau[p], &b[p * ldb], &ldb,
                       dwork, &ldwork, &info_lq);
            if (dwork[0] > wrkopt) wrkopt = dwork[0];
        }
    }

    dwork[0] = wrkopt;
}
