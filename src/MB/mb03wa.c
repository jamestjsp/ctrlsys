// SPDX-License-Identifier: BSD-3-Clause
/**
 * @file mb03wa.c
 * @brief Swap adjacent diagonal blocks in periodic real Schur form
 *
 * Swaps adjacent diagonal blocks A11*B11 and A22*B22 of size 1-by-1 or 2-by-2
 * in an upper (quasi) triangular matrix product A*B by orthogonal equivalence.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb03wa(
    const bool wantq,
    const bool wantz,
    const i32 n1,
    const i32 n2,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* q,
    const i32 ldq,
    f64* z,
    const i32 ldz,
    i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TEN = 10.0;
    const i32 LDST = 4;
    const bool WANDS = true;

    i32 i, linfo, m;
    f64 bqra21, brqa21, ddum, dnorm, dscale, dsum, eps;
    f64 f, g, sa, sb, scale, smlnum, ss, thresh, ws;

    i32 iwork[4];
    f64 ai[2], ar[2], be[2], dwork[32];
    f64 ir[16], ircop[16], li[16], licop[16];
    f64 s[16], scpy[16], t[16], taul[4], taur[4], tcpy[16];

    i32 int1 = 1;
    i32 int2 = 2;
    f64 dneone = -1.0;

    *info = 0;

    if (n1 <= 0 || n2 <= 0) {
        return;
    }
    m = n1 + n2;

    bool weak = false;
    bool dtrong = false;

    SLC_DLASET("All", &LDST, &LDST, &ZERO, &ZERO, li, &LDST);
    SLC_DLASET("All", &LDST, &LDST, &ZERO, &ZERO, ir, &LDST);
    SLC_DLACPY("Full", &m, &m, a, &lda, s, &LDST);
    SLC_DLACPY("Full", &m, &m, b, &ldb, t, &LDST);

    eps = SLC_DLAMCH("P");
    smlnum = SLC_DLAMCH("S") / eps;
    dscale = ZERO;
    dsum = ONE;
    SLC_DLACPY("Full", &m, &m, s, &LDST, dwork, &m);
    i32 msq = m * m;
    SLC_DLASSQ(&msq, dwork, &int1, &dscale, &dsum);
    SLC_DLACPY("Full", &m, &m, t, &LDST, dwork, &m);
    SLC_DLASSQ(&msq, dwork, &int1, &dscale, &dsum);
    dnorm = dscale * sqrt(dsum);
    thresh = fmax(TEN * eps * dnorm, smlnum);

    if (m == 2) {
        f = s[1 + 1 * LDST] * t[1 + 1 * LDST] - t[0] * s[0];
        g = -s[1 + 1 * LDST] * t[0 + 1 * LDST] - t[0] * s[0 + 1 * LDST];
        sb = fabs(t[0]);
        sa = fabs(s[1 + 1 * LDST]);
        SLC_DLARTG(&f, &g, &ir[0 + 1 * LDST], &ir[0], &ddum);
        ir[1 + 0 * LDST] = -ir[0 + 1 * LDST];
        ir[1 + 1 * LDST] = ir[0];
        SLC_DROT(&int2, &s[0], &int1, &s[0 + 1 * LDST], &int1, &ir[0], &ir[1 + 0 * LDST]);
        SLC_DROT(&int2, &t[0], &LDST, &t[1 + 0 * LDST], &LDST, &ir[0], &ir[1 + 0 * LDST]);
        if (sa >= sb) {
            SLC_DLARTG(&s[0], &s[1], &li[0], &li[1], &ddum);
        } else {
            SLC_DLARTG(&t[1 + 1 * LDST], &t[1], &li[0], &li[1], &ddum);
            li[1] = -li[1];
        }
        SLC_DROT(&int2, &s[0], &LDST, &s[1], &LDST, &li[0], &li[1]);
        SLC_DROT(&int2, &t[0], &int1, &t[0 + 1 * LDST], &int1, &li[0], &li[1]);
        li[1 + 1 * LDST] = li[0];
        li[0 + 1 * LDST] = -li[1];

        ws = fabs(s[1]) + fabs(t[1]);
        weak = ws <= thresh;
        if (!weak) {
            *info = 1;
            return;
        }

        if (WANDS) {
            SLC_DLACPY("Full", &m, &m, a, &lda, &dwork[msq], &m);
            SLC_DGEMM("No Transpose", "No Transpose", &m, &m, &m, &ONE, li, &LDST, s, &LDST, &ZERO, dwork, &m);
            SLC_DGEMM("No Transpose", "Transpose", &m, &m, &m, &dneone, dwork, &m, ir, &LDST, &ONE, &dwork[msq], &m);
            dscale = ZERO;
            dsum = ONE;
            SLC_DLASSQ(&msq, &dwork[msq], &int1, &dscale, &dsum);

            SLC_DLACPY("Full", &m, &m, b, &ldb, &dwork[msq], &m);
            SLC_DGEMM("No Transpose", "No Transpose", &m, &m, &m, &ONE, ir, &LDST, t, &LDST, &ZERO, dwork, &m);
            SLC_DGEMM("No Transpose", "Transpose", &m, &m, &m, &dneone, dwork, &m, li, &LDST, &ONE, &dwork[msq], &m);
            SLC_DLASSQ(&msq, &dwork[msq], &int1, &dscale, &dsum);
            ss = dscale * sqrt(dsum);
            dtrong = ss <= thresh;
            if (!dtrong) {
                *info = 1;
                return;
            }
        }

        SLC_DLACPY("All", &m, &m, s, &LDST, a, &lda);
        SLC_DLACPY("All", &m, &m, t, &LDST, b, &ldb);

        a[1] = ZERO;
        b[1] = ZERO;

        if (wantq) {
            SLC_DROT(&int2, &q[0], &int1, &q[0 + 1 * ldq], &int1, &li[0], &li[1]);
        }
        if (wantz) {
            SLC_DROT(&int2, &z[0], &int1, &z[0 + 1 * ldz], &int1, &ir[0], &ir[1 + 0 * LDST]);
        }

        return;
    } else {
        SLC_DLACPY("Full", &n1, &n2, &t[0 + n1 * LDST], &LDST, li, &LDST);
        SLC_DLACPY("Full", &n1, &n2, &s[0 + n1 * LDST], &LDST, &ir[n2 + n1 * LDST], &LDST);
        sb04ow(n1, n2, s, LDST, &s[n1 + n1 * LDST], LDST,
               &ir[n2 + n1 * LDST], LDST, t, LDST, &t[n1 + n1 * LDST], LDST,
               li, LDST, &scale, iwork, &linfo);
        if (linfo != 0) {
            *info = 1;
            return;
        }

        for (i = 0; i < n2; i++) {
            SLC_DSCAL(&n1, &dneone, &li[0 + i * LDST], &int1);
            li[n1 + i + i * LDST] = scale;
        }
        SLC_DGEQR2(&m, &n2, li, &LDST, taul, dwork, &linfo);
        SLC_DORG2R(&m, &m, &n2, li, &LDST, taul, dwork, &linfo);

        for (i = 0; i < n1; i++) {
            ir[n2 + i + i * LDST] = scale;
        }
        SLC_DGERQ2(&n1, &m, &ir[n2], &LDST, taur, dwork, &linfo);
        SLC_DORGR2(&m, &m, &n1, ir, &LDST, taur, dwork, &linfo);

        SLC_DGEMM("Transpose", "No Transpose", &m, &m, &m, &ONE, li, &LDST, s, &LDST, &ZERO, dwork, &m);
        SLC_DGEMM("No Transpose", "Transpose", &m, &m, &m, &ONE, dwork, &m, ir, &LDST, &ZERO, s, &LDST);
        SLC_DGEMM("No Transpose", "No Transpose", &m, &m, &m, &ONE, ir, &LDST, t, &LDST, &ZERO, dwork, &m);
        SLC_DGEMM("No Transpose", "No Transpose", &m, &m, &m, &ONE, dwork, &m, li, &LDST, &ZERO, t, &LDST);
        SLC_DLACPY("All", &m, &m, s, &LDST, scpy, &LDST);
        SLC_DLACPY("All", &m, &m, t, &LDST, tcpy, &LDST);
        SLC_DLACPY("All", &m, &m, ir, &LDST, ircop, &LDST);
        SLC_DLACPY("All", &m, &m, li, &LDST, licop, &LDST);

        SLC_DGEQR2(&m, &m, t, &LDST, taur, dwork, &linfo);
        SLC_DORM2R("Right", "No Transpose", &m, &m, &m, t, &LDST, taur, s, &LDST, dwork, &linfo);
        SLC_DORM2R("Left", "Transpose", &m, &m, &m, t, &LDST, taur, ir, &LDST, dwork, &linfo);

        dscale = ZERO;
        dsum = ONE;
        for (i = 0; i < n2; i++) {
            SLC_DLASSQ(&n1, &s[n2 + i * LDST], &int1, &dscale, &dsum);
        }
        brqa21 = dscale * sqrt(dsum);

        SLC_DGERQ2(&m, &m, tcpy, &LDST, taul, dwork, &linfo);
        SLC_DORMR2("Left", "No Transpose", &m, &m, &m, tcpy, &LDST, taul, scpy, &LDST, dwork, &linfo);
        SLC_DORMR2("Right", "Transpose", &m, &m, &m, tcpy, &LDST, taul, licop, &LDST, dwork, &linfo);

        dscale = ZERO;
        dsum = ONE;
        for (i = 0; i < n2; i++) {
            SLC_DLASSQ(&n1, &scpy[n2 + i * LDST], &int1, &dscale, &dsum);
        }
        bqra21 = dscale * sqrt(dsum);

        if (bqra21 <= brqa21 && bqra21 <= thresh) {
            SLC_DLACPY("All", &m, &m, scpy, &LDST, s, &LDST);
            SLC_DLACPY("All", &m, &m, tcpy, &LDST, t, &LDST);
            SLC_DLACPY("All", &m, &m, ircop, &LDST, ir, &LDST);
            SLC_DLACPY("All", &m, &m, licop, &LDST, li, &LDST);
        } else if (brqa21 >= thresh) {
            *info = 1;
            return;
        }

        i32 mm1 = m - 1;
        SLC_DLASET("Lower", &mm1, &mm1, &ZERO, &ZERO, &t[1], &LDST);

        if (WANDS) {
            SLC_DLACPY("All", &m, &m, a, &lda, &dwork[msq], &m);
            SLC_DGEMM("No Transpose", "No Transpose", &m, &m, &m, &ONE, li, &LDST, s, &LDST, &ZERO, dwork, &m);
            SLC_DGEMM("No Transpose", "No Transpose", &m, &m, &m, &dneone, dwork, &m, ir, &LDST, &ONE, &dwork[msq], &m);
            dscale = ZERO;
            dsum = ONE;
            SLC_DLASSQ(&msq, &dwork[msq], &int1, &dscale, &dsum);

            SLC_DLACPY("All", &m, &m, b, &ldb, &dwork[msq], &m);
            SLC_DGEMM("Transpose", "No Transpose", &m, &m, &m, &ONE, ir, &LDST, t, &LDST, &ZERO, dwork, &m);
            SLC_DGEMM("No Transpose", "Transpose", &m, &m, &m, &dneone, dwork, &m, li, &LDST, &ONE, &dwork[msq], &m);
            SLC_DLASSQ(&msq, &dwork[msq], &int1, &dscale, &dsum);
            ss = dscale * sqrt(dsum);
            dtrong = ss <= thresh;
            if (!dtrong) {
                *info = 1;
                return;
            }
        }

        SLC_DLASET("All", &n1, &n2, &ZERO, &ZERO, &s[n2], &LDST);

        SLC_DLACPY("All", &m, &m, s, &LDST, a, &lda);
        SLC_DLACPY("All", &m, &m, t, &LDST, b, &ldb);
        SLC_DLASET("All", &LDST, &LDST, &ZERO, &ZERO, t, &LDST);

        SLC_DLASET("All", &m, &m, &ZERO, &ZERO, dwork, &m);
        dwork[0] = ONE;
        t[0] = ONE;
        if (n2 > 1) {
            mb03yt(a, lda, b, ldb, ar, ai, be, &dwork[0], &dwork[1], &t[0], &t[1]);
            dwork[m] = -dwork[1];
            dwork[m + 1] = dwork[0];
            t[n2 - 1 + (n2 - 1) * LDST] = t[0];
            t[0 + 1 * LDST] = -t[1];
        }
        dwork[msq - 1] = ONE;
        t[m - 1 + (m - 1) * LDST] = ONE;

        if (n1 > 1) {
            mb03yt(&a[n2 + n2 * lda], lda, &b[n2 + n2 * ldb], ldb, taur, taul, &dwork[msq],
                   &dwork[n2 * m + n2], &dwork[n2 * m + n2 + 1], &t[n2 + n2 * LDST], &t[m - 1 + (m - 2) * LDST]);
            dwork[msq - 1] = dwork[n2 * m + n2];
            dwork[msq - 2] = -dwork[n2 * m + n2 + 1];
            t[m - 1 + (m - 1) * LDST] = t[n2 + n2 * LDST];
            t[m - 2 + (m - 1) * LDST] = -t[m - 1 + (m - 2) * LDST];
        }

        SLC_DGEMM("Transpose", "No Transpose", &n2, &n1, &n2, &ONE, dwork, &m, &a[0 + n2 * lda], &lda, &ZERO, &dwork[msq], &n2);
        SLC_DLACPY("All", &n2, &n1, &dwork[msq], &n2, &a[0 + n2 * lda], &lda);
        SLC_DGEMM("Transpose", "No Transpose", &n2, &n1, &n2, &ONE, t, &LDST, &b[0 + n2 * ldb], &ldb, &ZERO, &dwork[msq], &n2);
        SLC_DLACPY("All", &n2, &n1, &dwork[msq], &n2, &b[0 + n2 * ldb], &ldb);
        SLC_DGEMM("No Transpose", "No Transpose", &m, &m, &m, &ONE, li, &LDST, dwork, &m, &ZERO, &dwork[msq], &m);
        SLC_DLACPY("All", &m, &m, &dwork[msq], &m, li, &LDST);
        SLC_DGEMM("No Transpose", "No Transpose", &n2, &n1, &n1, &ONE, &a[0 + n2 * lda], &lda, &t[n2 + n2 * LDST], &LDST, &ZERO, &dwork[msq], &m);
        SLC_DLACPY("All", &n2, &n1, &dwork[msq], &m, &a[0 + n2 * lda], &lda);
        SLC_DGEMM("No Transpose", "No Transpose", &n2, &n1, &n1, &ONE, &b[0 + n2 * ldb], &ldb, &dwork[n2 * m + n2], &m, &ZERO, &dwork[msq], &m);
        SLC_DLACPY("All", &n2, &n1, &dwork[msq], &m, &b[0 + n2 * ldb], &ldb);
        SLC_DGEMM("Transpose", "No Transpose", &m, &m, &m, &ONE, t, &LDST, ir, &LDST, &ZERO, dwork, &m);
        SLC_DLACPY("All", &m, &m, dwork, &m, ir, &LDST);

        if (wantq) {
            SLC_DGEMM("No Transpose", "No Transpose", &m, &m, &m, &ONE, q, &ldq, li, &LDST, &ZERO, dwork, &m);
            SLC_DLACPY("All", &m, &m, dwork, &m, q, &ldq);
        }

        if (wantz) {
            SLC_DGEMM("No Transpose", "Transpose", &m, &m, &m, &ONE, z, &ldz, ir, &LDST, &ZERO, dwork, &m);
            SLC_DLACPY("Full", &m, &m, dwork, &m, z, &ldz);
        }

        return;
    }
}
