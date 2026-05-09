/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB10SD - H2 optimal controller for normalized discrete-time systems
 *
 * Computes the H2 optimal controller matrices:
 *     K = | AK | BK |
 *         |----|----|
 *         | CK | DK |
 *
 * for the normalized discrete-time system from SB10PD.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>

void sb10sd(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
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
    f64* x,
    const i32 ldx,
    f64* y,
    const i32 ldy,
    f64* rcond,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
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
    } else if (ldak < (n > 1 ? n : 1)) {
        *info = -15;
    } else if (ldbk < (n > 1 ? n : 1)) {
        *info = -17;
    } else if (ldck < (m2 > 1 ? m2 : 1)) {
        *info = -19;
    } else if (lddk < (m2 > 1 ? m2 : 1)) {
        *info = -21;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -23;
    } else if (ldy < (n > 1 ? n : 1)) {
        *info = -25;
    }

    if (*info == 0) {
        i32 tmp1 = 14 * n + 23;
        i32 tmp2 = 16 * n;
        i32 ws1 = 14 * n * n + 6 * n + (tmp1 > tmp2 ? tmp1 : tmp2);

        i32 tmp3 = 3 > m1 ? 3 : m1;
        i32 ws2 = m2 * (n + m2 + tmp3);

        i32 ws3 = np2 * (n + np2 + 3);

        i32 minwrk = ws1;
        if (ws2 > minwrk) minwrk = ws2;
        if (ws3 > minwrk) minwrk = ws3;
        if (minwrk < 1) minwrk = 1;

        if (ldwork < minwrk) {
            *info = -30;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0 || np == 0 || m1 == 0 || m2 == 0 ||
        np1 == 0 || np2 == 0) {
        rcond[0] = one;
        rcond[1] = one;
        rcond[2] = one;
        rcond[3] = one;
        dwork[0] = one;
        return;
    }

    i32 nd1 = np1 - m2;
    i32 nd2 = m1 - np2;
    f64 toll = tol;
    if (toll <= zero) {
        toll = sqrt(SLC_DLAMCH("Epsilon"));
    }

    i32 iwq = 0;
    i32 iwg = iwq + n * n;
    i32 iwr = iwg + n * n;
    i32 iwi = iwr + 2 * n;
    i32 iwb = iwi + 2 * n;
    i32 iws = iwb + 2 * n;
    i32 iwt = iws + 4 * n * n;
    i32 iwu = iwt + 4 * n * n;
    i32 iwrk = iwu + 4 * n * n;
    i32 iwc = iwr;
    i32 iwv = iwc + n * n;

    f64 mone = -one;
    i32 lwamax = 0;
    i32 info2 = 0;
    f64 rcond2;
    f64 sepd, ferr;

    SLC_DLACPY("Full", &n, &n, a, &lda, ak, &ldak);
    SLC_DGEMM("N", "N", &n, &n, &m2, &mone, &b[m1 * ldb], &ldb,
              &c[nd1], &ldc, &one, ak, &ldak);

    if (nd1 > 0) {
        SLC_DSYRK("L", "T", &n, &nd1, &one, c, &ldc, &zero, &dwork[iwq], &n);
    } else {
        SLC_DLASET("L", &n, &n, &zero, &zero, &dwork[iwq], &n);
    }

    SLC_DSYRK("L", "N", &n, &m2, &one, &b[m1 * ldb], &ldb, &zero, &dwork[iwg], &n);

    i32 n2 = 2 * n;
    i32 lwork = ldwork - iwrk;
    sb02od("D", "G", "N", "L", "Z", "S", n, m2, np1, ak, ldak,
           &dwork[iwg], n, &dwork[iwq], n, &dwork[iwrk], m,
           &dwork[iwrk], n, &rcond2, x, ldx, &dwork[iwr],
           &dwork[iwi], &dwork[iwb], &dwork[iws], n2,
           &dwork[iwt], n2, &dwork[iwu], n2, toll, iwork,
           &dwork[iwrk], lwork, &info2);

    if (info2 > 0) {
        *info = 1;
        return;
    }
    lwamax = (i32)dwork[iwrk] + iwrk;

    iwrk = iwv + n * n;
    lwork = ldwork - iwrk;
    sb02sd("C", "N", "N", "L", "O", n, ak, ldak, &dwork[iwc],
           n, &dwork[iwv], n, &dwork[iwg], n, &dwork[iwq], n,
           x, ldx, &sepd, &rcond[2], &ferr, iwork, &dwork[iwrk],
           lwork, &info2);

    if (info2 > 0) {
        rcond[2] = zero;
    }
    i32 tmp = (i32)dwork[iwrk] + iwrk;
    if (tmp > lwamax) lwamax = tmp;

    i32 iw2 = m2 * n;
    iwrk = iw2 + m2 * m2;

    SLC_DGEMM("T", "N", &m2, &n, &n, &one, &b[m1 * ldb], &ldb, x, &ldx,
              &zero, dwork, &m2);

    SLC_DLASET("L", &m2, &m2, &zero, &one, &dwork[iw2], &m2);
    mb01rx("Left", "Lower", "N", m2, n, one, one, &dwork[iw2],
           m2, dwork, m2, &b[m1 * ldb], ldb, &info2);

    f64 anorm = SLC_DLANSY("I", "L", &m2, &dwork[iw2], &m2, &dwork[iwrk]);
    SLC_DPOTRF("L", &m2, &dwork[iw2], &m2, &info2);
    if (info2 > 0) {
        *info = 2;
        return;
    }
    SLC_DPOCON("L", &m2, &dwork[iw2], &m2, &anorm, &rcond[0],
               &dwork[iwrk], iwork, &info2);

    if (rcond[0] < toll) {
        *info = 2;
        return;
    }

    SLC_DLACPY("Full", &m2, &n, &c[nd1], &ldc, ck, &ldck);
    SLC_DGEMM("N", "N", &m2, &n, &n, &mone, dwork, &m2, a, &lda, &mone, ck, &ldck);

    SLC_DPOTRS("L", &m2, &n, &dwork[iw2], &m2, ck, &ldck, &info2);

    SLC_DLACPY("Full", &m2, &m1, &d[nd1], &ldd, &dwork[iwrk], &m2);
    SLC_DGEMM("N", "N", &m2, &m1, &n, &mone, dwork, &m2, b, &ldb, &mone,
              &dwork[iwrk], &m2);

    SLC_DPOTRS("L", &m2, &m1, &dwork[iw2], &m2, &dwork[iwrk], &m2, &info2);

    SLC_DLACPY("Full", &m2, &np2, &dwork[iwrk + nd2 * m2], &m2, dk, &lddk);

    iwrk = iwu + 4 * n * n;

    SLC_DLACPY("Full", &n, &n, a, &lda, ak, &ldak);
    SLC_DGEMM("N", "N", &n, &n, &np2, &mone, &b[nd2 * ldb], &ldb,
              &c[np1], &ldc, &one, ak, &ldak);

    for (i32 j = 0; j < n - 1; j++) {
        i32 jlen = j + 1;
        SLC_DSWAP(&jlen, &ak[(j + 1) * ldak], &int1, &ak[j + 1], &ldak);
    }

    if (nd2 > 0) {
        SLC_DSYRK("U", "N", &n, &nd2, &one, b, &ldb, &zero, &dwork[iwq], &n);
    } else {
        SLC_DLASET("U", &n, &n, &zero, &zero, &dwork[iwq], &n);
    }

    SLC_DSYRK("U", "T", &n, &np2, &one, &c[np1], &ldc, &zero, &dwork[iwg], &n);

    lwork = ldwork - iwrk;
    sb02od("D", "G", "N", "U", "Z", "S", n, np2, m1, ak, ldak,
           &dwork[iwg], n, &dwork[iwq], n, &dwork[iwrk], m,
           &dwork[iwrk], n, &rcond2, y, ldy, &dwork[iwr],
           &dwork[iwi], &dwork[iwb], &dwork[iws], n2,
           &dwork[iwt], n2, &dwork[iwu], n2, toll, iwork,
           &dwork[iwrk], lwork, &info2);

    if (info2 > 0) {
        *info = 3;
        return;
    }
    tmp = (i32)dwork[iwrk] + iwrk;
    if (tmp > lwamax) lwamax = tmp;

    iwrk = iwv + n * n;
    lwork = ldwork - iwrk;
    sb02sd("C", "N", "N", "U", "O", n, ak, ldak, &dwork[iwc],
           n, &dwork[iwv], n, &dwork[iwg], n, &dwork[iwq], n,
           y, ldy, &sepd, &rcond[3], &ferr, iwork, &dwork[iwrk],
           lwork, &info2);

    if (info2 > 0) {
        rcond[3] = zero;
    }
    tmp = (i32)dwork[iwrk] + iwrk;
    if (tmp > lwamax) lwamax = tmp;

    iw2 = n * np2;
    iwrk = iw2 + np2 * np2;

    SLC_DGEMM("N", "T", &n, &np2, &n, &one, y, &ldy, &c[np1], &ldc,
              &zero, dwork, &n);

    SLC_DLASET("U", &np2, &np2, &zero, &one, &dwork[iw2], &np2);
    mb01rx("Left", "Upper", "N", np2, n, one, one, &dwork[iw2],
           np2, &c[np1], ldc, dwork, n, &info2);

    anorm = SLC_DLANSY("I", "U", &np2, &dwork[iw2], &np2, &dwork[iwrk]);
    SLC_DPOTRF("U", &np2, &dwork[iw2], &np2, &info2);
    if (info2 > 0) {
        *info = 4;
        return;
    }
    SLC_DPOCON("U", &np2, &dwork[iw2], &np2, &anorm, &rcond[1],
               &dwork[iwrk], iwork, &info2);

    if (rcond[1] < toll) {
        *info = 4;
        return;
    }

    SLC_DLACPY("Full", &n, &np2, &b[nd2 * ldb], &ldb, bk, &ldbk);
    SLC_DGEMM("N", "N", &n, &np2, &n, &one, a, &lda, dwork, &n, &one, bk, &ldbk);

    SLC_DTRSM("R", "U", "N", "N", &n, &np2, &mone, &dwork[iw2], &np2, bk, &ldbk);
    SLC_DTRSM("R", "U", "T", "N", &n, &np2, &one, &dwork[iw2], &np2, bk, &ldbk);

    SLC_DGEMM("N", "N", &m2, &np2, &n, &one, ck, &ldck, dwork, &n, &one, dk, &lddk);

    SLC_DTRSM("R", "U", "N", "N", &m2, &np2, &one, &dwork[iw2], &np2, dk, &lddk);
    SLC_DTRSM("R", "U", "T", "N", &m2, &np2, &one, &dwork[iw2], &np2, dk, &lddk);

    SLC_DGEMM("N", "N", &m2, &n, &np2, &mone, dk, &lddk, &c[np1], &ldc,
              &one, ck, &ldck);

    SLC_DLACPY("Full", &n, &n, a, &lda, ak, &ldak);
    SLC_DGEMM("N", "N", &n, &n, &m2, &one, &b[m1 * ldb], &ldb, ck, &ldck,
              &one, ak, &ldak);
    SLC_DGEMM("N", "N", &n, &n, &np2, &one, bk, &ldbk, &c[np1], &ldc,
              &one, ak, &ldak);

    SLC_DGEMM("N", "N", &n, &np2, &m2, &one, &b[m1 * ldb], &ldb, dk, &lddk,
              &mone, bk, &ldbk);

    dwork[0] = (f64)lwamax;
}
