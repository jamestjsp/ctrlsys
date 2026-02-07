/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB10FD - H-infinity (sub)optimal state controller for continuous-time systems
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdio.h>

void sb10fd(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    const f64 gamma,
    const f64* a,
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
    i32* info)
{
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
    } else if (gamma < 0.0) {
        *info = -6;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -8;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -10;
    } else if (ldc < (1 > np ? 1 : np)) {
        *info = -12;
    } else if (ldd < (1 > np ? 1 : np)) {
        *info = -14;
    } else if (ldak < (1 > n ? 1 : n)) {
        *info = -16;
    } else if (ldbk < (1 > n ? 1 : n)) {
        *info = -18;
    } else if (ldck < (1 > m2 ? 1 : m2)) {
        *info = -20;
    } else if (lddk < (1 > m2 ? 1 : m2)) {
        *info = -22;
    } else {
        i32 nd1 = np1 - m2;
        i32 nd2 = m1 - np2;

        i32 lw1 = (n + np1 + 1) * (n + m2) +
                  ((3*(n + m2) + n + np1) > (5*(n + m2)) ?
                   (3*(n + m2) + n + np1) : (5*(n + m2)));

        i32 lw2 = (n + np2) * (n + m1 + 1) +
                  ((3*(n + np2) + n + m1) > (5*(n + np2)) ?
                   (3*(n + np2) + n + m1) : (5*(n + np2)));

        i32 n_max_m1 = n > m1 ? n : m1;
        i32 lw3_inner = np1 * n_max_m1;
        i32 lw3_a = lw3_inner > (3*m2 + np1) ? lw3_inner : (3*m2 + np1);
        i32 lw3_b = lw3_a > (5*m2) ? lw3_a : (5*m2);
        i32 lw3 = m2 + np1*np1 + lw3_b;

        i32 n_max_np1 = n > np1 ? n : np1;
        i32 lw4 = np2 + m1*m1 +
                  ((n_max_np1 * m1) > (3*np2 + m1) ?
                   ((n_max_np1 * m1) > (5*np2) ? (n_max_np1 * m1) : (5*np2)) :
                   ((3*np2 + m1) > (5*np2) ? (3*np2 + m1) : (5*np2)));

        i32 inner5_1 = 10*n*n + 12*n + 5;
        i32 inner5_2a = n*m > inner5_1 ? n*m : inner5_1;
        i32 inner5_2b = n*np > inner5_1 ? n*np : inner5_1;
        i32 inner5_3a = 3*n*n + inner5_2a;
        i32 inner5_3b = 3*n*n + inner5_2b;
        i32 inner5_4a = 2*m1 > inner5_3a ? 2*m1 : inner5_3a;
        i32 inner5_4b = 2*np1 > inner5_3b ? 2*np1 : inner5_3b;
        i32 inner5_5a = m*m + inner5_4a;
        i32 inner5_5b = np*np + inner5_4b;
        i32 inner5_6 = inner5_5a > inner5_5b ? inner5_5a : inner5_5b;
        i32 lw5 = 2*n*n + n*(m + np) + (1 > inner5_6 ? 1 : inner5_6);

        i32 np2_or_n = np2 > n ? np2 : n;
        i32 inner6_1a = 2*nd1 > ((nd1 + nd2)*np2) ? 2*nd1 : ((nd1 + nd2)*np2);
        i32 inner6_1b = nd1*nd1 + inner6_1a;
        i32 inner6_2a = 2*nd2 > (nd2*m2) ? 2*nd2 : (nd2*m2);
        i32 inner6_2b = nd2*nd2 + inner6_2a;
        i32 inner6_3 = inner6_1b > inner6_2b ? inner6_1b : inner6_2b;
        i32 inner6_4 = 3*n > inner6_3 ? 3*n : inner6_3;
        i32 inner6_5a = m2*m2 + 3*m2;
        i32 inner6_5b = np2 * (2*np2 + m2 + np2_or_n);
        i32 inner6_5 = inner6_5a > inner6_5b ? inner6_5a : inner6_5b;
        i32 inner6_6 = m2*np2 + inner6_5;
        i32 inner6_7 = 2*n*m2 > inner6_6 ? 2*n*m2 : inner6_6;
        i32 inner6_8 = n*(2*np2 + m2) + inner6_7;
        i32 inner6_9 = inner6_4 > inner6_8 ? inner6_4 : inner6_8;
        i32 inner6_10 = m2*np2 + np2*np2 + m2*m2 + inner6_9;
        i32 lw6 = 2*n*n + n*(m + np) + (1 > inner6_10 ? 1 : inner6_10);

        i32 lw_max = lw1;
        if (lw2 > lw_max) lw_max = lw2;
        if (lw3 > lw_max) lw_max = lw3;
        if (lw4 > lw_max) lw_max = lw4;
        if (lw5 > lw_max) lw_max = lw5;
        if (lw6 > lw_max) lw_max = lw6;
        if (1 > lw_max) lw_max = 1;

        i32 minwrk = n*m + np*(n + m) + m2*m2 + np2*np2 + lw_max;

        if (ldwork < minwrk) {
            *info = -27;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0 || np == 0 || m1 == 0 || m2 == 0 ||
        np1 == 0 || np2 == 0) {
        rcond[0] = 1.0;
        rcond[1] = 1.0;
        rcond[2] = 1.0;
        rcond[3] = 1.0;
        dwork[0] = 1.0;
        return;
    }

    f64 toll = tol;
    if (toll <= 0.0) {
        toll = sqrt(SLC_DLAMCH("Epsilon"));
    }

    i32 iwc = n * m;
    i32 iwd = iwc + np * n;
    i32 iwtu = iwd + np * m;
    i32 iwty = iwtu + m2 * m2;
    i32 iwrk = iwty + np2 * np2;

    SLC_DLACPY("Full", &n, &m, b, &ldb, dwork, &n);
    SLC_DLACPY("Full", &np, &n, c, &ldc, &dwork[iwc], &np);
    SLC_DLACPY("Full", &np, &m, d, &ldd, &dwork[iwd], &np);

    i32 ldwork_sub = ldwork - iwrk;
    i32 info2 = 0;
    sb10pd(n, m, np, ncon, nmeas, a, lda, dwork, n,
           &dwork[iwc], np, &dwork[iwd], np, &dwork[iwtu], m2,
           &dwork[iwty], np2, rcond, toll, &dwork[iwrk], ldwork_sub, &info2);

    if (info2 > 0) {
        *info = info2;
        return;
    }

    i32 lwamax = (i32)dwork[iwrk] + iwrk;

    i32 iwx = iwrk;
    i32 iwy = iwx + n * n;
    i32 iwf = iwy + n * n;
    i32 iwh = iwf + m * n;
    iwrk = iwh + n * np;

    ldwork_sub = ldwork - iwrk;
    sb10qd(n, m, np, ncon, nmeas, gamma, a, lda, dwork, n,
           &dwork[iwc], np, &dwork[iwd], np, &dwork[iwf], m,
           &dwork[iwh], n, &dwork[iwx], n, &dwork[iwy], n,
           &rcond[2], iwork, &dwork[iwrk], ldwork_sub, bwork, &info2);

    if (info2 > 0) {
        *info = info2 + 5;
        return;
    }

    i32 opt2 = (i32)dwork[iwrk] + iwrk;
    if (opt2 > lwamax) lwamax = opt2;

    ldwork_sub = ldwork - iwrk;
    sb10rd(n, m, np, ncon, nmeas, gamma, a, lda, dwork, n,
           &dwork[iwc], np, &dwork[iwd], np, &dwork[iwf], m,
           &dwork[iwh], n, &dwork[iwtu], m2, &dwork[iwty], np2,
           &dwork[iwx], n, &dwork[iwy], n, ak, ldak, bk, ldbk,
           ck, ldck, dk, lddk, iwork, &dwork[iwrk], ldwork_sub, &info2);

    if (info2 == 1) {
        *info = 6;
        return;
    } else if (info2 == 2) {
        *info = 9;
        return;
    }

    i32 opt3 = (i32)dwork[iwrk] + iwrk;
    if (opt3 > lwamax) lwamax = opt3;

    dwork[0] = (f64)lwamax;
}
