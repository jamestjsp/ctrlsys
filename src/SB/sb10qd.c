/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB10QD - State feedback and output injection matrices for H-infinity
 *          (sub)optimal controller
 *
 * Computes the state feedback matrix F and output injection matrix H
 * for an H-infinity controller using Glover's and Doyle's 1988 formulas.
 * Solves two Riccati equations with condition estimates.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>

void sb10qd(
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
    f64* f,
    const i32 ldf,
    f64* h,
    const i32 ldh,
    f64* x,
    const i32 ldx,
    f64* y,
    const i32 ldy,
    f64* xycond,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 negone = -1.0;
    i32 m1 = m - ncon;
    i32 m2 = ncon;
    i32 np1 = np - nmeas;
    i32 np2 = nmeas;
    i32 nn = n * n;

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
    } else if (gamma < zero) {
        *info = -6;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -10;
    } else if (ldc < (np > 1 ? np : 1)) {
        *info = -12;
    } else if (ldd < (np > 1 ? np : 1)) {
        *info = -14;
    } else if (ldf < (m > 1 ? m : 1)) {
        *info = -16;
    } else if (ldh < (n > 1 ? n : 1)) {
        *info = -18;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -20;
    } else if (ldy < (n > 1 ? n : 1)) {
        *info = -22;
    } else {
        i32 minwrk_x = m * m + (2 * m1 > 3 * nn + (n * m > 10 * nn + 12 * n + 5 ? n * m : 10 * nn + 12 * n + 5) ?
                               2 * m1 : 3 * nn + (n * m > 10 * nn + 12 * n + 5 ? n * m : 10 * nn + 12 * n + 5));
        i32 minwrk_y = np * np + (2 * np1 > 3 * nn + (n * np > 10 * nn + 12 * n + 5 ? n * np : 10 * nn + 12 * n + 5) ?
                                 2 * np1 : 3 * nn + (n * np > 10 * nn + 12 * n + 5 ? n * np : 10 * nn + 12 * n + 5));
        i32 minwrk = minwrk_x > minwrk_y ? minwrk_x : minwrk_y;
        if (minwrk < 1) minwrk = 1;
        if (ldwork < minwrk) {
            *info = -26;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0 || np == 0 || m1 == 0 || m2 == 0 || np1 == 0 || np2 == 0) {
        xycond[0] = one;
        xycond[1] = one;
        dwork[0] = one;
        return;
    }

    i32 nd1 = np1 - m2;
    i32 nd2 = m1 - np2;
    i32 n2 = 2 * n;

    f64 eps = SLC_DLAMCH("Epsilon");

    i32 iwa = m * m;
    i32 iwq = iwa + nn;
    i32 iwg = iwq + nn;
    i32 iw2 = iwg + nn;

    f64 gamma2 = gamma * gamma;
    f64 neg_gamma2 = -gamma2;
    SLC_DLASET("L", &m1, &m1, &zero, &neg_gamma2, dwork, &m);

    if (nd1 > 0) {
        SLC_DSYRK("L", "T", &m1, &nd1, &one, d, &ldd, &one, dwork, &m);
    }

    i32 iwrk = iwa;
    i32 lwamax = 0;
    i32 info2 = 0;
    f64 anorm = SLC_DLANSY("I", "L", &m1, dwork, &m, &dwork[iwrk]);

    i32 ldwork_rem = ldwork - iwrk;
    SLC_DSYTRF("L", &m1, dwork, &m, iwork, &dwork[iwrk], &ldwork_rem, &info2);
    if (info2 > 0) {
        *info = 1;
        return;
    }

    lwamax = (i32)dwork[iwrk] + iwrk;

    f64 rcond;
    SLC_DSYCON("L", &m1, dwork, &m, iwork, &anorm, &rcond, &dwork[iwrk], &iwork[m1], &info2);
    if (rcond < eps) {
        *info = 1;
        return;
    }

    SLC_DSYTRI("L", &m1, dwork, &m, iwork, &dwork[iwrk], &info2);

    SLC_DSYMM("R", "L", &m2, &m1, &negone, dwork, &m, &d[nd1], &ldd, &zero, &dwork[m1], &m);

    i32 m1_m = m1 * (m + 1);
    SLC_DLASET("Lower", &m2, &m2, &zero, &one, &dwork[m1_m], &m);

    slicot_mb01rx('R', 'L', 'T', m2, m1, one, negone, &dwork[m1_m], m, &d[nd1], ldd, &dwork[m1], m);

    SLC_DGEMM("T", "N", &m1, &n, &np1, &one, d, &ldd, c, &ldc, &zero, &dwork[iw2], &m);

    SLC_DLACPY("Full", &m2, &n, &c[nd1], &ldc, &dwork[iw2 + m1], &m);

    SLC_DSYMM("L", "L", &m, &n, &one, dwork, &m, &dwork[iw2], &m, &zero, f, &ldf);

    SLC_DLACPY("Full", &n, &n, a, &lda, &dwork[iwa], &n);
    SLC_DGEMM("N", "N", &n, &n, &m, &negone, b, &ldb, f, &ldf, &one, &dwork[iwa], &n);

    if (nd1 == 0) {
        SLC_DLASET("L", &n, &n, &zero, &zero, &dwork[iwq], &n);
    } else {
        SLC_DSYRK("L", "T", &n, &np1, &one, c, &ldc, &zero, &dwork[iwq], &n);
        slicot_mb01rx('L', 'L', 'T', n, m, one, negone, &dwork[iwq], n, &dwork[iw2], m, f, ldf);
    }

    iwrk = iw2;
    i32 ldwork_ru = m * n;
    mb01ru("Lower", "NoTranspose", n, m, zero, one, &dwork[iwg], n, b, ldb, dwork, m, &dwork[iwrk], ldwork_ru, &info2);

    i32 iwt = iw2;
    i32 iwv = iwt + nn;
    i32 iwr = iwv + nn;
    i32 iwi = iwr + n2;
    i32 iws = iwi + n2;
    iwrk = iws + 4 * nn;

    f64 sep, ferr;
    ldwork_rem = ldwork - iwrk;

    sb02rd("All", "Continuous", "NotUsed", "NoTranspose", "Lower", "GeneralScaling",
           "Stable", "NotFactored", "Original", n, &dwork[iwa], n, &dwork[iwt], n,
           &dwork[iwv], n, &dwork[iwg], n, &dwork[iwq], n, x, ldx, &sep, &xycond[0],
           &ferr, &dwork[iwr], &dwork[iwi], &dwork[iws], n2, iwork, &dwork[iwrk],
           ldwork_rem, bwork, &info2);

    if (info2 > 0) {
        *info = 2;
        return;
    }

    i32 tmp = (i32)dwork[iwrk] + iwrk;
    if (tmp > lwamax) lwamax = tmp;

    iwrk = iw2;
    SLC_DGEMM("T", "N", &m, &n, &n, &one, b, &ldb, x, &ldx, &zero, &dwork[iwrk], &m);
    SLC_DSYMM("L", "L", &m, &n, &negone, dwork, &m, &dwork[iwrk], &m, &negone, f, &ldf);

    iwa = np * np;
    iwq = iwa + nn;
    iwg = iwq + nn;
    iw2 = iwg + nn;

    SLC_DLASET("U", &np1, &np1, &zero, &neg_gamma2, dwork, &np);

    if (nd2 > 0) {
        SLC_DSYRK("U", "N", &np1, &nd2, &one, d, &ldd, &one, dwork, &np);
    }

    iwrk = iwa;
    anorm = SLC_DLANSY("I", "U", &np1, dwork, &np, &dwork[iwrk]);

    ldwork_rem = ldwork - iwrk;
    SLC_DSYTRF("U", &np1, dwork, &np, iwork, &dwork[iwrk], &ldwork_rem, &info2);
    if (info2 > 0) {
        *info = 1;
        return;
    }

    tmp = (i32)dwork[iwrk] + iwrk;
    if (tmp > lwamax) lwamax = tmp;

    SLC_DSYCON("U", &np1, dwork, &np, iwork, &anorm, &rcond, &dwork[iwrk], &iwork[np1], &info2);
    if (rcond < eps) {
        *info = 1;
        return;
    }

    SLC_DSYTRI("U", &np1, dwork, &np, iwork, &dwork[iwrk], &info2);

    i32 np1_np = np1 * np;
    SLC_DSYMM("L", "U", &np1, &np2, &negone, dwork, &np, &d[nd2 * ldd], &ldd, &zero, &dwork[np1_np], &np);

    i32 np1_np1 = np1 * (np + 1);
    SLC_DLASET("Full", &np2, &np2, &zero, &one, &dwork[np1_np1], &np);

    slicot_mb01rx('L', 'U', 'T', np2, np1, one, negone, &dwork[np1_np1], np, &d[nd2 * ldd], ldd, &dwork[np1_np], np);

    SLC_DGEMM("N", "T", &n, &np1, &m1, &one, b, &ldb, d, &ldd, &zero, &dwork[iw2], &n);

    SLC_DLACPY("Full", &n, &np2, &b[nd2 * ldb], &ldb, &dwork[iw2 + np1 * n], &n);

    SLC_DSYMM("R", "U", &n, &np, &one, dwork, &np, &dwork[iw2], &n, &zero, h, &ldh);

    SLC_DLACPY("Full", &n, &n, a, &lda, &dwork[iwa], &n);
    SLC_DGEMM("N", "N", &n, &n, &np, &negone, h, &ldh, c, &ldc, &one, &dwork[iwa], &n);

    if (nd2 == 0) {
        SLC_DLASET("U", &n, &n, &zero, &zero, &dwork[iwq], &n);
    } else {
        SLC_DSYRK("U", "N", &n, &m1, &one, b, &ldb, &zero, &dwork[iwq], &n);
        slicot_mb01rx('R', 'U', 'T', n, np, one, negone, &dwork[iwq], n, h, ldh, &dwork[iw2], n);
    }

    iwrk = iw2;
    i32 ldwork_ru2 = n * np;
    mb01ru("Upper", "Transpose", n, np, zero, one, &dwork[iwg], n, c, ldc, dwork, np, &dwork[iwrk], ldwork_ru2, &info2);

    iwt = iw2;
    iwv = iwt + nn;
    iwr = iwv + nn;
    iwi = iwr + n2;
    iws = iwi + n2;
    iwrk = iws + 4 * nn;

    ldwork_rem = ldwork - iwrk;

    sb02rd("All", "Continuous", "NotUsed", "Transpose", "Upper", "GeneralScaling",
           "Stable", "NotFactored", "Original", n, &dwork[iwa], n, &dwork[iwt], n,
           &dwork[iwv], n, &dwork[iwg], n, &dwork[iwq], n, y, ldy, &sep, &xycond[1],
           &ferr, &dwork[iwr], &dwork[iwi], &dwork[iws], n2, iwork, &dwork[iwrk],
           ldwork_rem, bwork, &info2);

    if (info2 > 0) {
        *info = 3;
        return;
    }

    tmp = (i32)dwork[iwrk] + iwrk;
    if (tmp > lwamax) lwamax = tmp;

    iwrk = iw2;
    SLC_DGEMM("N", "T", &n, &np, &n, &one, y, &ldy, c, &ldc, &zero, &dwork[iwrk], &n);
    SLC_DSYMM("R", "U", &n, &np, &negone, dwork, &np, &dwork[iwrk], &n, &negone, h, &ldh);

    dwork[0] = (f64)lwamax;
}
