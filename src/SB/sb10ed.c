/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB10ED - H2 optimal state controller for discrete-time systems
 *
 * Computes the H2 optimal n-state controller:
 *     K = | AK | BK |
 *         |----|----|
 *         | CK | DK |
 *
 * for the discrete-time system:
 *     P = | A  | B1  B2  |
 *         |----|---------|
 *         | C1 |  0  D12 |
 *         | C2 | D21 D22 |
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>

void sb10ed(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    f64* ak,
    const i32 ldak,
    f64* bk,
    const i32 ldbk,
    f64* ck,
    const i32 ldck,
    f64* dk,
    const i32 lddk,
    f64* rcond,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* info
)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    i32 m1 = m - ncon;
    i32 m2 = ncon;
    i32 np1 = np - nmeas;
    i32 np2 = nmeas;
    i32 nl = n > 1 ? n : 1;
    i32 npl = np > 1 ? np : 1;
    i32 m2l = m2 > 1 ? m2 : 1;
    i32 nlp = np2 > 1 ? np2 : 1;

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
    } else if (lda < nl) {
        *info = -7;
    } else if (ldb < nl) {
        *info = -9;
    } else if (ldc < npl) {
        *info = -11;
    } else if (ldd < npl) {
        *info = -13;
    } else if (ldak < nl) {
        *info = -15;
    } else if (ldbk < nl) {
        *info = -17;
    } else if (ldck < m2l) {
        *info = -19;
    } else if (lddk < m2l) {
        *info = -21;
    } else {
        i32 lw1 = (n + np1 + 1) * (n + m2) +
                  (3 * (n + m2) + n + np1 > 5 * (n + m2) ?
                   3 * (n + m2) + n + np1 : 5 * (n + m2));
        i32 lw2 = (n + np2) * (n + m1 + 1) +
                  (3 * (n + np2) + n + m1 > 5 * (n + np2) ?
                   3 * (n + np2) + n + m1 : 5 * (n + np2));
        i32 tmp_lw3 = np1 * (n > m1 ? n : m1);
        i32 tmp2_lw3 = 3 * m2 + np1;
        i32 tmp3_lw3 = 5 * m2;
        i32 max_lw3 = tmp_lw3 > tmp2_lw3 ? tmp_lw3 : tmp2_lw3;
        max_lw3 = max_lw3 > tmp3_lw3 ? max_lw3 : tmp3_lw3;
        i32 lw3 = m2 + np1 * np1 + max_lw3;

        i32 tmp_lw4 = (n > np1 ? n : np1) * m1;
        i32 tmp2_lw4 = 3 * np2 + m1;
        i32 tmp3_lw4 = 5 * np2;
        i32 max_lw4 = tmp_lw4 > tmp2_lw4 ? tmp_lw4 : tmp2_lw4;
        max_lw4 = max_lw4 > tmp3_lw4 ? max_lw4 : tmp3_lw4;
        i32 lw4 = np2 + m1 * m1 + max_lw4;

        i32 tmp_lw5a = 14 * n + 23 > 16 * n ? 14 * n + 23 : 16 * n;
        i32 tmp_lw5b = 14 * n * n + 6 * n + tmp_lw5a;
        i32 tmp_lw5c = m2 * (n + m2 + (3 > m1 ? 3 : m1));
        i32 tmp_lw5d = np2 * (n + np2 + 3);
        i32 max_lw5 = 1 > tmp_lw5b ? 1 : tmp_lw5b;
        max_lw5 = max_lw5 > tmp_lw5c ? max_lw5 : tmp_lw5c;
        max_lw5 = max_lw5 > tmp_lw5d ? max_lw5 : tmp_lw5d;
        i32 lw5 = 2 * n * n + max_lw5;

        i32 tmp_lw6a = n * m2;
        i32 tmp_lw6b = n * np2;
        i32 tmp_lw6c = m2 * np2;
        i32 tmp_lw6d = m2 * m2 + 4 * m2;
        i32 lw6 = tmp_lw6a > tmp_lw6b ? tmp_lw6a : tmp_lw6b;
        lw6 = lw6 > tmp_lw6c ? lw6 : tmp_lw6c;
        lw6 = lw6 > tmp_lw6d ? lw6 : tmp_lw6d;

        i32 minwrk = n * m + np * (n + m) + m2 * m2 + np2 * np2;
        i32 maxlw = 1 > lw1 ? 1 : lw1;
        maxlw = maxlw > lw2 ? maxlw : lw2;
        maxlw = maxlw > lw3 ? maxlw : lw3;
        maxlw = maxlw > lw4 ? maxlw : lw4;
        maxlw = maxlw > lw5 ? maxlw : lw5;
        maxlw = maxlw > lw6 ? maxlw : lw6;
        minwrk += maxlw;

        if (ldwork < minwrk) {
            *info = -26;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 && (m2 == 0 || np2 == 0)) {
        rcond[0] = one;
        rcond[1] = one;
        rcond[2] = one;
        rcond[3] = one;
        rcond[4] = one;
        rcond[5] = one;
        rcond[6] = one;
        dwork[0] = one;
        return;
    }

    f64 toll = tol;
    if (toll <= zero) {
        toll = sqrt(SLC_DLAMCH("Epsilon"));
    }

    i32 iwc = n * m;
    i32 iwd = iwc + np * n;
    i32 iwtu = iwd + np * m;
    i32 iwty = iwtu + m2 * m2;
    i32 iwrk = iwty + np2 * np2;

    SLC_DLACPY("Full", &n, &m, b, &ldb, dwork, &nl);
    SLC_DLACPY("Full", &np, &n, c, &ldc, &dwork[iwc], &npl);
    SLC_DLACPY("Full", &np, &m, d, &ldd, &dwork[iwd], &npl);

    for (i32 i = 0; i < n; i++) {
        a[i + i * lda] -= one;
    }

    i32 info2 = 0;
    i32 ldwork_rem = ldwork - iwrk;
    sb10pd(n, m, np, ncon, nmeas, a, lda, dwork, nl,
           &dwork[iwc], npl, &dwork[iwd], npl, &dwork[iwtu], m2l,
           &dwork[iwty], nlp, rcond, toll, &dwork[iwrk], ldwork_rem, &info2);

    for (i32 i = 0; i < n; i++) {
        a[i + i * lda] += one;
    }

    if (info2 > 0) {
        *info = info2;
        return;
    }

    i32 lwamax = (i32)dwork[iwrk] + iwrk;

    i32 iwx = iwrk;
    i32 iwy = iwx + n * n;
    iwrk = iwy + n * n;

    ldwork_rem = ldwork - iwrk;
    sb10sd(n, m, np, ncon, nmeas, a, lda, dwork, nl,
           &dwork[iwc], npl, &dwork[iwd], npl, ak, ldak, bk, ldbk,
           ck, ldck, dk, lddk, &dwork[iwx], nl, &dwork[iwy], nl,
           &rcond[2], toll, iwork, &dwork[iwrk], ldwork_rem, bwork, &info2);

    if (info2 > 0) {
        *info = info2 + 5;
        return;
    }

    i32 tmp_max = (i32)dwork[iwrk] + iwrk;
    lwamax = lwamax > tmp_max ? lwamax : tmp_max;

    iwrk = iwx;

    ldwork_rem = ldwork - iwrk;
    sb10td(n, m, np, ncon, nmeas, &dwork[iwd], npl, &dwork[iwtu], m2l,
           &dwork[iwty], nlp, ak, ldak, bk, ldbk, ck, ldck, dk, lddk,
           &rcond[6], toll, iwork, &dwork[iwrk], ldwork_rem, &info2);

    if (info2 > 0) {
        *info = 10;
        return;
    }

    dwork[0] = (f64)lwamax;
}
