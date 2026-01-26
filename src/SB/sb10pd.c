/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB10PD - Normalization of system for H-infinity controller design
 *
 * Reduces D12 and D21 of system P to unit diagonal form and transforms
 * B, C, D11 for H2/H-infinity controller computation.
 *
 * System partitioning:
 *     | A  | B1  B2  |
 * P = |----|---------|
 *     | C1 | D11 D12 |
 *     | C2 | D21 D22 |
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * Overlap-safe matrix copy using memmove.
 * Required because optimized BLAS DLACPY may use memcpy internally,
 * which has undefined behavior for overlapping regions.
 */
static void dlacpy_safe(i32 m, i32 n, const f64 *a, i32 lda, f64 *b, i32 ldb) {
    for (i32 j = 0; j < n; j++) {
        memmove(&b[j * ldb], &a[j * lda], (size_t)m * sizeof(f64));
    }
}

void sb10pd(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    const f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    f64* tu,
    const i32 ldtu,
    f64* ty,
    const i32 ldty,
    f64* rcond,
    const f64 tol,
    f64* dwork,
    const i32 ldwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const i32 int1 = 1;

    i32 m1 = m - ncon;
    i32 m2 = ncon;
    i32 np1 = np - nmeas;
    i32 np2 = nmeas;

    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (np < 0) {
        *info = -3;
    } else if (ncon < 0 || m1 < 0 || m2 > np1) {
        *info = -4;
    } else if (nmeas < 0 || np1 < 0 || np2 > m1) {
        *info = -5;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (ldc < (np > 1 ? np : 1)) {
        *info = -11;
    } else if (ldd < (np > 1 ? np : 1)) {
        *info = -13;
    } else if (ldtu < (m2 > 1 ? m2 : 1)) {
        *info = -15;
    } else if (ldty < (np2 > 1 ? np2 : 1)) {
        *info = -17;
    }

    if (*info == 0) {
        i32 nm2 = n + m2;
        i32 nnp1 = n + np1;
        i32 nnp2 = n + np2;
        i32 nm1 = n + m1;

        i32 lw1_mat = (nnp1 + 1) * nm2;
        i32 lw1_svd1 = 3 * nm2 + nnp1;
        i32 lw1_svd2 = 5 * nm2;
        i32 lw1 = lw1_mat + (lw1_svd1 > lw1_svd2 ? lw1_svd1 : lw1_svd2);

        i32 lw2_mat = nnp2 * (nm1 + 1);
        i32 lw2_svd1 = 3 * nnp2 + nm1;
        i32 lw2_svd2 = 5 * nnp2;
        i32 lw2 = lw2_mat + (lw2_svd1 > lw2_svd2 ? lw2_svd1 : lw2_svd2);

        i32 lw3_np1m1 = np1 > m1 ? np1 : m1;
        i32 lw3_work1 = np1 * (n > lw3_np1m1 ? n : lw3_np1m1);
        i32 lw3_work2 = 3 * m2 + np1;
        i32 lw3_work3 = 5 * m2;
        i32 lw3_max = lw3_work1;
        if (lw3_work2 > lw3_max) lw3_max = lw3_work2;
        if (lw3_work3 > lw3_max) lw3_max = lw3_work3;
        i32 lw3 = m2 + np1 * np1 + lw3_max;

        i32 lw4_nnp1 = n > np1 ? n : np1;
        i32 lw4_work1 = lw4_nnp1 * m1;
        i32 lw4_work2 = 3 * np2 + m1;
        i32 lw4_work3 = 5 * np2;
        i32 lw4_max = lw4_work1;
        if (lw4_work2 > lw4_max) lw4_max = lw4_work2;
        if (lw4_work3 > lw4_max) lw4_max = lw4_work3;
        i32 lw4 = np2 + m1 * m1 + lw4_max;

        i32 minwrk = lw1;
        if (lw2 > minwrk) minwrk = lw2;
        if (lw3 > minwrk) minwrk = lw3;
        if (lw4 > minwrk) minwrk = lw4;
        if (minwrk < 1) minwrk = 1;

        if (ldwork < minwrk) {
            *info = -21;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0 || np == 0 || m1 == 0 || m2 == 0 ||
        np1 == 0 || np2 == 0) {
        rcond[0] = one;
        rcond[1] = one;
        dwork[0] = one;
        return;
    }

    i32 nd1 = np1 - m2;
    i32 nd2 = m1 - np2;
    f64 eps = SLC_DLAMCH("Epsilon");
    f64 toll = tol;
    if (toll <= zero) {
        toll = sqrt(eps);
    }

    i32 info2 = 0;
    i32 lwamax = 0;

    i32 nnp1 = n + np1;
    i32 nm2 = n + m2;
    i32 iext = nm2;
    i32 iwrk = iext + nnp1 * nm2;

    SLC_DLACPY("Full", &n, &n, a, &lda, &dwork[iext], &nnp1);
    SLC_DLACPY("Full", &np1, &n, c, &ldc, &dwork[iext + n], &nnp1);
    i32 col_offset = iext + nnp1 * n;
    SLC_DLACPY("Full", &n, &m2, &b[m1 * ldb], &ldb, &dwork[col_offset], &nnp1);
    SLC_DLACPY("Full", &np1, &m2, &d[m1 * ldd], &ldd, &dwork[col_offset + n], &nnp1);

    i32 lwork = ldwork - iwrk;
    SLC_DGESVD("N", "N", &nnp1, &nm2, &dwork[iext], &nnp1, dwork,
               tu, &ldtu, ty, &ldty, &dwork[iwrk], &lwork, &info2);

    if (info2 != 0) {
        *info = 5;
        return;
    }

    if (dwork[nm2 - 1] / dwork[0] <= eps) {
        *info = 1;
        return;
    }

    lwamax = (i32)dwork[iwrk] + iwrk;

    i32 nnp2 = n + np2;
    i32 nm1 = n + m1;
    iext = nnp2;
    iwrk = iext + nnp2 * nm1;

    SLC_DLACPY("Full", &n, &n, a, &lda, &dwork[iext], &nnp2);
    SLC_DLACPY("Full", &np2, &n, &c[np1], &ldc, &dwork[iext + n], &nnp2);
    col_offset = iext + nnp2 * n;
    SLC_DLACPY("Full", &n, &m1, b, &ldb, &dwork[col_offset], &nnp2);
    SLC_DLACPY("Full", &np2, &m1, &d[np1], &ldd, &dwork[col_offset + n], &nnp2);

    lwork = ldwork - iwrk;
    SLC_DGESVD("N", "N", &nnp2, &nm1, &dwork[iext], &nnp2, dwork,
               tu, &ldtu, ty, &ldty, &dwork[iwrk], &lwork, &info2);

    if (info2 != 0) {
        *info = 5;
        return;
    }

    if (dwork[nnp2 - 1] / dwork[0] <= eps) {
        *info = 2;
        return;
    }

    i32 tmp = (i32)dwork[iwrk] + iwrk;
    if (tmp > lwamax) lwamax = tmp;

    i32 iq = m2;
    iwrk = iq + np1 * np1;

    SLC_DGESVD("A", "A", &np1, &m2, &d[m1 * ldd], &ldd, dwork,
               &dwork[iq], &np1, tu, &ldtu, &dwork[iwrk], &lwork, &info2);

    if (info2 != 0) {
        *info = 5;
        return;
    }

    rcond[0] = dwork[m2 - 1] / dwork[0];
    if (rcond[0] <= toll) {
        rcond[1] = zero;
        *info = 3;
        return;
    }

    tmp = (i32)dwork[iwrk] + iwrk;
    if (tmp > lwamax) lwamax = tmp;

    if (nd1 > 0) {
        SLC_DLACPY("Full", &np1, &m2, &dwork[iq], &np1, &d[m1 * ldd], &ldd);
        // Use memmove-based copy for potentially overlapping regions within dwork
        dlacpy_safe(np1, nd1, &dwork[iq + np1 * m2], np1, &dwork[iq], np1);
        SLC_DLACPY("Full", &np1, &m2, &d[m1 * ldd], &ldd, &dwork[iq + np1 * nd1], &np1);
    }

    for (i32 j = 0; j < m2 - 1; j++) {
        i32 jlen = j + 1;
        SLC_DSWAP(&jlen, &tu[(j + 1) * ldtu], &int1, &tu[j + 1], &ldtu);
    }

    for (i32 j = 0; j < m2; j++) {
        f64 scale = one / dwork[j];
        SLC_DSCAL(&m2, &scale, &tu[j * ldtu], &int1);
    }

    SLC_DGEMM("T", "N", &np1, &n, &np1, &one, &dwork[iq], &np1, c, &ldc,
              &zero, &dwork[iwrk], &np1);
    SLC_DLACPY("Full", &np1, &n, &dwork[iwrk], &np1, c, &ldc);
    tmp = iwrk + np1 * n;
    if (tmp > lwamax) lwamax = tmp;

    SLC_DGEMM("T", "N", &np1, &m1, &np1, &one, &dwork[iq], &np1, d, &ldd,
              &zero, &dwork[iwrk], &np1);
    SLC_DLACPY("Full", &np1, &m1, &dwork[iwrk], &np1, d, &ldd);
    tmp = iwrk + np1 * m1;
    if (tmp > lwamax) lwamax = tmp;

    iq = np2;
    iwrk = iq + m1 * m1;

    SLC_DGESVD("A", "A", &np2, &m1, &d[np1], &ldd, dwork, ty,
               &ldty, &dwork[iq], &m1, &dwork[iwrk], &lwork, &info2);

    if (info2 != 0) {
        *info = 5;
        return;
    }

    rcond[1] = dwork[np2 - 1] / dwork[0];
    if (rcond[1] <= toll) {
        *info = 4;
        return;
    }

    tmp = (i32)dwork[iwrk] + iwrk;
    if (tmp > lwamax) lwamax = tmp;

    if (nd2 > 0) {
        SLC_DLACPY("Full", &np2, &m1, &dwork[iq], &m1, &d[np1], &ldd);
        // Use memmove-based copy for potentially overlapping regions within dwork
        dlacpy_safe(nd2, m1, &dwork[iq + np2], m1, &dwork[iq], m1);
        SLC_DLACPY("Full", &np2, &m1, &d[np1], &ldd, &dwork[iq + nd2], &m1);
    }

    for (i32 j = 0; j < np2; j++) {
        f64 scale = one / dwork[j];
        SLC_DSCAL(&np2, &scale, &ty[j * ldty], &int1);
    }

    for (i32 j = 0; j < np2 - 1; j++) {
        i32 jlen = j + 1;
        SLC_DSWAP(&jlen, &ty[(j + 1) * ldty], &int1, &ty[j + 1], &ldty);
    }

    SLC_DGEMM("N", "T", &n, &m1, &m1, &one, b, &ldb, &dwork[iq], &m1,
              &zero, &dwork[iwrk], &n);
    SLC_DLACPY("Full", &n, &m1, &dwork[iwrk], &n, b, &ldb);
    tmp = iwrk + n * m1;
    if (tmp > lwamax) lwamax = tmp;

    SLC_DGEMM("N", "T", &np1, &m1, &m1, &one, d, &ldd, &dwork[iq], &m1,
              &zero, &dwork[iwrk], &np1);
    SLC_DLACPY("Full", &np1, &m1, &dwork[iwrk], &np1, d, &ldd);
    tmp = iwrk + np1 * m1;
    if (tmp > lwamax) lwamax = tmp;

    SLC_DGEMM("N", "N", &n, &m2, &m2, &one, &b[m1 * ldb], &ldb, tu, &ldtu,
              &zero, dwork, &n);
    SLC_DLACPY("Full", &n, &m2, dwork, &n, &b[m1 * ldb], &ldb);

    SLC_DGEMM("N", "N", &np2, &n, &np2, &one, ty, &ldty,
              &c[np1], &ldc, &zero, dwork, &np2);
    SLC_DLACPY("Full", &np2, &n, dwork, &np2, &c[np1], &ldc);

    tmp = n * m2;
    i32 tmp2 = np2 * n;
    if (tmp2 > tmp) tmp = tmp2;
    if (tmp > lwamax) lwamax = tmp;

    dwork[0] = (f64)lwamax;
}
