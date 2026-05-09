/**
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

static int select_none(const f64* wr, const f64* wi) {
    (void)wr;
    (void)wi;
    return 0;
}

void sb10zd(i32 n, i32 m, i32 np, const f64* a, i32 lda,
            const f64* b, i32 ldb, const f64* c, i32 ldc,
            const f64* d, i32 ldd, f64 factor,
            f64* ak, i32 ldak, f64* bk, i32 ldbk,
            f64* ck, i32 ldck, f64* dk, i32 lddk,
            f64* rcond, f64 tol, i32* iwork, f64* dwork, i32 ldwork,
            i32* bwork, i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    i32 i, j;
    i32 i1, i2, i3, i4, i5, i6, i7, i8, i9, i10;
    i32 i11, i12, i13, i14, i15, i16, i17, i18, i19, i20;
    i32 i21, i22, i23, i24, i25, i26;
    i32 info2, iwrk, lwamax, minwrk, n2, ns, sdim;
    f64 anorm, gamma, toll;

    i32 int1 = 1, int0 = 0;

    *info = 0;

    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (np < 0) {
        *info = -3;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -5;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldc < (np > 1 ? np : 1)) {
        *info = -9;
    } else if (ldd < (np > 1 ? np : 1)) {
        *info = -11;
    } else if (factor < ONE) {
        *info = -12;
    } else if (ldak < (n > 1 ? n : 1)) {
        *info = -14;
    } else if (ldbk < (n > 1 ? n : 1)) {
        *info = -16;
    } else if (ldck < (m > 1 ? m : 1)) {
        *info = -18;
    } else if (lddk < (m > 1 ? m : 1)) {
        *info = -20;
    } else if (tol >= ONE) {
        *info = -22;
    }

    /* Compute workspace */
    i32 tmp1 = 14*n + 23;
    i32 tmp2 = 16*n;
    i32 tmp3 = 2*m - 1;
    i32 tmp4 = 2*np - 1;
    i32 maxval = tmp1;
    if (tmp2 > maxval) maxval = tmp2;
    if (tmp3 > maxval) maxval = tmp3;
    if (tmp4 > maxval) maxval = tmp4;

    minwrk = 16*n*n + 5*m*m + 7*np*np + 6*m*n + 7*m*np + 7*n*np +
             6*n + 2*(m + np) + maxval;

    if (ldwork < minwrk) {
        *info = -25;
    }

    if (*info != 0) {
        return;
    }

    /* Quick return if possible */
    if (n == 0 || m == 0 || np == 0) {
        rcond[0] = ONE;
        rcond[1] = ONE;
        rcond[2] = ONE;
        rcond[3] = ONE;
        rcond[4] = ONE;
        rcond[5] = ONE;
        dwork[0] = ONE;
        return;
    }

    /* Set default tolerance */
    if (tol <= ZERO) {
        toll = sqrt(SLC_DLAMCH("Epsilon"));
    } else {
        toll = tol;
    }

    /* Workspace usage */
    n2 = 2*n;
    i1  = n*n;
    i2  = i1 + n*n;
    i3  = i2 + np*np;
    i4  = i3 + m*m;
    i5  = i4 + np*np;
    i6  = i5 + m*m;
    i7  = i6 + m*n;
    i8  = i7 + m*n;
    i9  = i8 + n*n;
    i10 = i9 + n*n;
    i11 = i10 + n2;
    i12 = i11 + n2;
    i13 = i12 + n2;
    i14 = i13 + n2*n2;
    i15 = i14 + n2*n2;

    iwrk = i15 + n2*n2;
    lwamax = 0;

    /* Compute R1 = Ip + D*D' */
    SLC_DLASET("U", &np, &np, &ZERO, &ONE, &dwork[i2], &np);
    SLC_DSYRK("U", "N", &np, &m, &ONE, d, &ldd, &ONE, &dwork[i2], &np);
    SLC_DLACPY("U", &np, &np, &dwork[i2], &np, &dwork[i4], &np);

    /* Factorize R1 = R'*R */
    SLC_DPOTRF("U", &np, &dwork[i4], &np, &info2);

    /* Compute C'*R^{-1} in BK */
    ma02ad("F", np, n, c, ldc, bk, ldbk);
    SLC_DTRSM("R", "U", "N", "N", &n, &np, &ONE, &dwork[i4], &np, bk, &ldbk);

    /* Compute R2 = Im + D'*D */
    SLC_DLASET("U", &m, &m, &ZERO, &ONE, &dwork[i3], &m);
    SLC_DSYRK("U", "T", &m, &np, &ONE, d, &ldd, &ONE, &dwork[i3], &m);
    SLC_DLACPY("U", &m, &m, &dwork[i3], &m, &dwork[i5], &m);

    /* Factorize R2 = U'*U */
    SLC_DPOTRF("U", &m, &dwork[i5], &m, &info2);

    /* Compute (U^{-1})'*B' */
    ma02ad("F", n, m, b, ldb, &dwork[i6], m);
    SLC_DTRTRS("U", "T", "N", &m, &n, &dwork[i5], &m, &dwork[i6], &m, &info2);

    /* Compute D'*C */
    SLC_DGEMM("T", "N", &m, &n, &np, &ONE, d, &ldd, c, &ldc, &ZERO, &dwork[i7], &m);

    /* Compute (U^{-1})'*D'*C */
    SLC_DTRTRS("U", "T", "N", &m, &n, &dwork[i5], &m, &dwork[i7], &m, &info2);

    /* Compute Ar = A - B*R2^{-1}*D'*C */
    SLC_DLACPY("F", &n, &n, a, &lda, &dwork[i8], &n);
    f64 negone = -ONE;
    SLC_DGEMM("T", "N", &n, &n, &m, &negone, &dwork[i6], &m, &dwork[i7], &m,
              &ONE, &dwork[i8], &n);

    /* Compute Cr = C'*R1^{-1}*C */
    SLC_DSYRK("U", "N", &n, &np, &ONE, bk, &ldbk, &ZERO, &dwork[i9], &n);

    /* Compute Dr = B*R2^{-1}*B' in AK */
    SLC_DSYRK("U", "T", &n, &m, &ONE, &dwork[i6], &m, &ZERO, ak, &ldak);

    /* Solution of the Riccati equation Ar'*P*(In + Dr*P)^{-1}*Ar - P + Cr = 0 */
    sb02od("D", "G", "N", "U", "Z", "S", n, m, np, &dwork[i8], n,
           ak, ldak, &dwork[i9], n, dwork, m, dwork, n,
           &rcond[0], dwork, n, &dwork[i10], &dwork[i11],
           &dwork[i12], &dwork[i13], n2, &dwork[i14], n2,
           &dwork[i15], n2, -ONE, iwork, &dwork[iwrk], ldwork-iwrk+1,
           &info2);
    if (info2 != 0) {
        *info = 1;
        return;
    }
    i32 tmp = (i32)dwork[iwrk] + iwrk - 1;
    if (tmp > lwamax) lwamax = tmp;

    /* Transpose Ar: swap row j+1 with column j+1 (j=0..n-2) */
    for (j = 1; j <= n - 1; j++) {
        SLC_DSWAP(&j, &dwork[i8 + j], &n, &dwork[i8 + j*n], &int1);
    }

    /* Solution of the Riccati equation Ar*Q*(In + Cr*Q)^{-1}*Ar' - Q + Dr = 0 */
    sb02od("D", "G", "N", "U", "Z", "S", n, m, np, &dwork[i8], n,
           &dwork[i9], n, ak, ldak, dwork, m, dwork, n,
           &rcond[1], &dwork[i1], n, &dwork[i10], &dwork[i11],
           &dwork[i12], &dwork[i13], n2, &dwork[i14], n2,
           &dwork[i15], n2, -ONE, iwork, &dwork[iwrk], ldwork-iwrk+1,
           &info2);
    if (info2 != 0) {
        *info = 2;
        return;
    }
    tmp = (i32)dwork[iwrk] + iwrk - 1;
    if (tmp > lwamax) lwamax = tmp;

    /* Compute gamma */
    SLC_DGEMM("N", "N", &n, &n, &n, &ONE, &dwork[i1], &n, dwork, &n,
              &ZERO, &dwork[i8], &n);
    SLC_DGEES("N", "N", select_none, &n, &dwork[i8], &n, &sdim,
              &dwork[i10], &dwork[i11], &dwork[iwrk], &n,
              &dwork[iwrk], &ldwork, bwork, &info2);
    if (info2 != 0) {
        *info = 3;
        return;
    }
    tmp = (i32)dwork[iwrk] + iwrk - 1;
    if (tmp > lwamax) lwamax = tmp;

    gamma = ZERO;
    for (i = 0; i < n; i++) {
        if (dwork[i10 + i] > gamma) gamma = dwork[i10 + i];
    }
    gamma = factor * sqrt(ONE + gamma);

    /* Workspace usage (second phase) */
    i5  = i4 + np*np;
    i6  = i5 + m*m;
    i7  = i6 + np*np;
    i8  = i7 + np*np;
    i9  = i8 + np*np;
    i10 = i9 + np;
    i11 = i10 + np*np;
    i12 = i11 + m*m;
    i13 = i12 + m;

    iwrk = i13 + m*m;

    /* Compute eigenvalues and eigenvectors of R1 */
    SLC_DLACPY("U", &np, &np, &dwork[i2], &np, &dwork[i8], &np);
    i32 ldw = ldwork - iwrk + 1;
    SLC_DSYEV("V", "U", &np, &dwork[i8], &np, &dwork[i9], &dwork[iwrk], &ldw, &info2);
    if (info2 != 0) {
        *info = 3;
        return;
    }
    tmp = (i32)dwork[iwrk] + iwrk - 1;
    if (tmp > lwamax) lwamax = tmp;

    /* Compute R1^{-1/2} */
    for (j = 0; j < np; j++) {
        for (i = 0; i < np; i++) {
            dwork[i10 + i + j*np] = dwork[i8 + j + i*np] / sqrt(dwork[i9 + i]);
        }
    }
    SLC_DGEMM("N", "N", &np, &np, &np, &ONE, &dwork[i8], &np,
              &dwork[i10], &np, &ZERO, &dwork[i4], &np);

    /* Compute eigenvalues and eigenvectors of R2 */
    SLC_DLACPY("U", &m, &m, &dwork[i3], &m, &dwork[i11], &m);
    ldw = ldwork - iwrk + 1;
    SLC_DSYEV("V", "U", &m, &dwork[i11], &m, &dwork[i12], &dwork[iwrk], &ldw, &info2);
    if (info2 != 0) {
        *info = 3;
        return;
    }
    tmp = (i32)dwork[iwrk] + iwrk - 1;
    if (tmp > lwamax) lwamax = tmp;

    /* Compute R2^{-1/2} */
    for (j = 0; j < m; j++) {
        for (i = 0; i < m; i++) {
            dwork[i13 + i + j*m] = dwork[i11 + j + i*m] / sqrt(dwork[i12 + i]);
        }
    }
    SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &dwork[i11], &m,
              &dwork[i13], &m, &ZERO, &dwork[i5], &m);

    /* Compute R1 + C*Q*C' */
    SLC_DGEMM("N", "T", &n, &np, &n, &ONE, &dwork[i1], &n, c, &ldc,
              &ZERO, bk, &ldbk);
    mb01rx("L", "U", "N", np, n, ONE, ONE, &dwork[i2], np, c, ldc, bk, ldbk, &info2);
    SLC_DLACPY("U", &np, &np, &dwork[i2], &np, &dwork[i8], &np);

    /* Compute eigenvalues and eigenvectors of R1 + C*Q*C' */
    ldw = ldwork - iwrk + 1;
    SLC_DSYEV("V", "U", &np, &dwork[i8], &np, &dwork[i9], &dwork[iwrk], &ldw, &info2);
    if (info2 != 0) {
        *info = 3;
        return;
    }
    tmp = (i32)dwork[iwrk] + iwrk - 1;
    if (tmp > lwamax) lwamax = tmp;

    /* Compute (R1 + C*Q*C')^{-1} */
    for (j = 0; j < np; j++) {
        for (i = 0; i < np; i++) {
            dwork[i10 + i + j*np] = dwork[i8 + j + i*np] / dwork[i9 + i];
        }
    }
    SLC_DGEMM("N", "N", &np, &np, &np, &ONE, &dwork[i8], &np,
              &dwork[i10], &np, &ZERO, &dwork[i6], &np);

    /* Compute Z2^{-1} */
    for (j = 0; j < np; j++) {
        for (i = 0; i < np; i++) {
            dwork[i10 + i + j*np] = dwork[i8 + j + i*np] * sqrt(dwork[i9 + i]);
        }
    }
    SLC_DGEMM("N", "N", &np, &np, &np, &ONE, &dwork[i8], &np,
              &dwork[i10], &np, &ZERO, &dwork[i7], &np);

    /* Workspace usage (third phase) */
    i9  = i8 + n*np;
    i10 = i9 + n*np;
    i11 = i10 + np*m;
    i12 = i11 + (np + m)*(np + m);
    i13 = i12 + n*(np + m);
    i14 = i13 + n*(np + m);
    i15 = i14 + n*n;
    i16 = i15 + n*n;
    i17 = i16 + (np + m)*n;
    i18 = i17 + (np + m)*(np + m);
    i19 = i18 + (np + m)*n;
    i20 = i19 + m*n;
    i21 = i20 + m*np;
    i22 = i21 + np*n;
    i23 = i22 + n*n;
    i24 = i23 + n*np;
    i25 = i24 + np*np;
    i26 = i25 + m*m;

    iwrk = i26 + n*m;

    /* Compute A*Q*C' + B*D' */
    SLC_DGEMM("N", "T", &n, &np, &m, &ONE, b, &ldb, d, &ldd, &ZERO, &dwork[i8], &n);
    SLC_DGEMM("N", "N", &n, &np, &n, &ONE, a, &lda, bk, &ldbk, &ONE, &dwork[i8], &n);

    /* Compute H = -(A*Q*C' + B*D')*(R1 + C*Q*C')^{-1} */
    SLC_DGEMM("N", "N", &n, &np, &np, &negone, &dwork[i8], &n,
              &dwork[i6], &np, &ZERO, &dwork[i9], &n);

    /* Compute R1^{-1/2}*D */
    SLC_DGEMM("N", "N", &np, &m, &np, &ONE, &dwork[i4], &np, d, &ldd,
              &ZERO, &dwork[i10], &np);

    /* Compute Rx */
    i32 npm = np + m;
    for (j = 0; j < np; j++) {
        i32 joff = j + 1;  /* Fortran: DCOPY(J,...) where J=1..NP */
        SLC_DCOPY(&joff, &dwork[i2 + j*np], &int1, &dwork[i11 + j*npm], &int1);
        dwork[i11 + j + j*npm] = dwork[i2 + j + j*np] - gamma*gamma;
    }

    SLC_DGEMM("N", "N", &np, &m, &np, &ONE, &dwork[i7], &np,
              &dwork[i10], &np, &ZERO, &dwork[i11 + npm*np], &npm);
    SLC_DLASET("U", &m, &m, &ZERO, &ONE, &dwork[i11 + npm*np + np], &npm);

    /* Compute Bx */
    SLC_DGEMM("N", "N", &n, &np, &np, &negone, &dwork[i9], &n,
              &dwork[i7], &np, &ZERO, &dwork[i12], &n);
    SLC_DGEMM("N", "N", &n, &m, &m, &ONE, b, &ldb, &dwork[i5], &m,
              &ZERO, &dwork[i12 + n*np], &n);

    /* Compute Sx */
    SLC_DGEMM("T", "N", &n, &np, &np, &ONE, c, &ldc, &dwork[i7], &np,
              &ZERO, &dwork[i13], &n);
    SLC_DGEMM("T", "N", &n, &m, &np, &ONE, c, &ldc, &dwork[i10], &np,
              &ZERO, &dwork[i13 + n*np], &n);

    /* Compute (gamma^2 - 1)*In - P*Q */
    f64 g2m1 = gamma*gamma - ONE;
    SLC_DLASET("F", &n, &n, &ZERO, &g2m1, &dwork[i14], &n);
    SLC_DGEMM("N", "N", &n, &n, &n, &negone, dwork, &n, &dwork[i1], &n,
              &ONE, &dwork[i14], &n);

    /* Compute X = ((gamma^2 - 1)*In - P*Q)^{-1}*gamma^2*P */
    SLC_DLACPY("F", &n, &n, dwork, &n, &dwork[i15], &n);
    f64 g2 = gamma*gamma;
    SLC_DLASCL("G", &int0, &int0, &ONE, &g2, &n, &n, &dwork[i15], &n, &info2);
    anorm = SLC_DLANGE("1", &n, &n, &dwork[i14], &n, &dwork[iwrk]);
    SLC_DGETRF(&n, &n, &dwork[i14], &n, iwork, &info2);
    if (info2 > 0) {
        *info = 4;
        return;
    }
    SLC_DGECON("1", &n, &dwork[i14], &n, &anorm, &rcond[2], &dwork[iwrk], &iwork[n], &info2);

    if (rcond[2] < toll) {
        *info = 4;
        return;
    }
    SLC_DGETRS("N", &n, &n, &dwork[i14], &n, iwork, &dwork[i15], &n, &info2);

    /* Compute Bx'*X */
    SLC_DGEMM("T", "N", &npm, &n, &n, &ONE, &dwork[i12], &n,
              &dwork[i15], &n, &ZERO, &dwork[i16], &npm);

    /* Compute Rx + Bx'*X*Bx */
    SLC_DLACPY("U", &npm, &npm, &dwork[i11], &npm, &dwork[i17], &npm);
    mb01rx("L", "U", "N", npm, n, ONE, ONE, &dwork[i17], npm, &dwork[i16], npm, &dwork[i12], n, &info2);

    /* Compute -(Sx' + Bx'*X*A) */
    ma02ad("F", n, npm, &dwork[i13], n, &dwork[i18], npm);
    SLC_DGEMM("N", "N", &npm, &n, &n, &negone, &dwork[i16], &npm,
              a, &lda, &negone, &dwork[i18], &npm);

    /* Factorize Rx + Bx'*X*Bx */
    anorm = SLC_DLANSY("1", "U", &npm, &dwork[i17], &npm, &dwork[iwrk]);
    ldw = ldwork - iwrk;
    SLC_DSYTRF("U", &npm, &dwork[i17], &npm, iwork, &dwork[iwrk], &ldw, &info2);
    if (info2 != 0) {
        *info = 5;
        return;
    }
    SLC_DSYCON("U", &npm, &dwork[i17], &npm, iwork, &anorm, &rcond[3], &dwork[iwrk], &iwork[npm], &info2);

    if (rcond[3] < toll) {
        *info = 5;
        return;
    }

    /* Compute F = -(Rx + Bx'*X*Bx)^{-1}*(Sx' + Bx'*X*A) */
    SLC_DSYTRS("U", &npm, &n, &dwork[i17], &npm, iwork, &dwork[i18], &npm, &info2);

    /* Compute B'*X */
    SLC_DGEMM("T", "N", &m, &n, &n, &ONE, b, &ldb, &dwork[i15], &n,
              &ZERO, &dwork[i19], &m);

    /* Compute -(D' - B'*X*H) */
    for (j = 0; j < np; j++) {
        for (i = 0; i < m; i++) {
            dwork[i20 + i + j*m] = -d[j + i*ldd];
        }
    }
    SLC_DGEMM("N", "N", &m, &np, &n, &ONE, &dwork[i19], &m,
              &dwork[i9], &n, &ONE, &dwork[i20], &m);

    /* Compute C + Z2^{-1}*F1 */
    SLC_DLACPY("F", &np, &n, c, &ldc, &dwork[i21], &np);
    SLC_DGEMM("N", "N", &np, &n, &np, &ONE, &dwork[i7], &np,
              &dwork[i18], &npm, &ONE, &dwork[i21], &np);

    /* Compute R2 + B'*X*B */
    mb01rx("L", "U", "N", m, n, ONE, ONE, &dwork[i3], m, &dwork[i19], m, b, ldb, &info2);

    /* Factorize R2 + B'*X*B */
    SLC_DPOTRF("U", &m, &dwork[i3], &m, &info2);

    /* Compute Dk_hat = -(R2 + B'*X*B)^{-1}*(D' - B'*X*H) */
    SLC_DLACPY("F", &m, &np, &dwork[i20], &m, dk, &lddk);
    SLC_DPOTRS("U", &m, &np, &dwork[i3], &m, dk, &lddk, &info2);

    /* Compute Bk_hat = -H + B*Dk_hat */
    SLC_DLACPY("F", &n, &np, &dwork[i9], &n, &dwork[i23], &n);
    SLC_DGEMM("N", "N", &n, &np, &m, &ONE, b, &ldb, dk, &lddk,
              &negone, &dwork[i23], &n);

    /* Compute R2^{-1/2}*F2 */
    SLC_DGEMM("N", "N", &m, &n, &m, &ONE, &dwork[i5], &m,
              &dwork[i18 + np], &npm, &ZERO, ck, &ldck);

    /* Compute Ck_hat = R2^{-1/2}*F2 - Dk_hat*(C + Z2^{-1}*F1) */
    SLC_DGEMM("N", "N", &m, &n, &np, &negone, dk, &lddk,
              &dwork[i21], &np, &ONE, ck, &ldck);

    /* Compute Ak_hat = A + H*C + B*Ck_hat */
    SLC_DLACPY("F", &n, &n, a, &lda, ak, &ldak);
    SLC_DGEMM("N", "N", &n, &n, &np, &ONE, &dwork[i9], &n, c, &ldc,
              &ONE, ak, &ldak);
    SLC_DGEMM("N", "N", &n, &n, &m, &ONE, b, &ldb, ck, &ldck,
              &ONE, ak, &ldak);

    /* Compute Ip + D*Dk_hat */
    SLC_DLASET("Full", &np, &np, &ZERO, &ONE, &dwork[i24], &np);
    SLC_DGEMM("N", "N", &np, &np, &m, &ONE, d, &ldd, dk, &lddk,
              &ONE, &dwork[i24], &np);

    /* Compute Im + Dk_hat*D */
    SLC_DLASET("Full", &m, &m, &ZERO, &ONE, &dwork[i25], &m);
    SLC_DGEMM("N", "N", &m, &m, &np, &ONE, dk, &lddk, d, &ldd,
              &ONE, &dwork[i25], &m);

    /* Compute Ck = M*Ck_hat, M = (Im + Dk_hat*D)^{-1} */
    anorm = SLC_DLANGE("1", &m, &m, &dwork[i25], &m, &dwork[iwrk]);
    SLC_DGETRF(&m, &m, &dwork[i25], &m, iwork, &info2);
    if (info2 != 0) {
        *info = 7;
        return;
    }
    SLC_DGECON("1", &m, &dwork[i25], &m, &anorm, &rcond[5], &dwork[iwrk], &iwork[m], &info2);

    if (rcond[5] < toll) {
        *info = 7;
        return;
    }
    SLC_DGETRS("N", &m, &n, &dwork[i25], &m, iwork, ck, &ldck, &info2);

    /* Compute Dk = M*Dk_hat */
    SLC_DGETRS("N", &m, &np, &dwork[i25], &m, iwork, dk, &lddk, &info2);

    /* Compute Bk_hat*D */
    SLC_DGEMM("N", "N", &n, &m, &np, &ONE, &dwork[i23], &n, d, &ldd,
              &ZERO, &dwork[i26], &n);

    /* Compute Ak = Ak_hat - Bk_hat*D*Ck */
    SLC_DGEMM("N", "N", &n, &n, &m, &negone, &dwork[i26], &n, ck, &ldck,
              &ONE, ak, &ldak);

    /* Compute Bk = Bk_hat*(Ip + D*Dk_hat)^{-1} */
    anorm = SLC_DLANGE("1", &np, &np, &dwork[i24], &np, &dwork[iwrk]);
    SLC_DLACPY("Full", &n, &np, &dwork[i23], &n, bk, &ldbk);
    mb02vd("N", n, np, &dwork[i24], np, iwork, bk, ldbk, &info2);
    if (info2 != 0) {
        *info = 6;
        return;
    }
    SLC_DGECON("1", &np, &dwork[i24], &np, &anorm, &rcond[4], &dwork[iwrk], &iwork[np], &info2);

    if (rcond[4] < toll) {
        *info = 6;
        return;
    }

    /* Workspace usage (fourth phase) - closed-loop stability check */
    i2 = np*np;
    i3 = i2 + n*np;
    i4 = i3 + m*m;
    i5 = i4 + n*m;
    i6 = i5 + np*n;
    i7 = i6 + m*n;
    i8 = i7 + n2*n2;
    i9 = i8 + n2;

    iwrk = i9 + n2;

    /* Compute Ip - D*Dk */
    SLC_DLASET("Full", &np, &np, &ZERO, &ONE, dwork, &np);
    SLC_DGEMM("N", "N", &np, &np, &m, &negone, d, &ldd, dk, &lddk, &ONE, dwork, &np);

    /* Compute Bk*(Ip - D*Dk)^{-1} */
    SLC_DLACPY("Full", &n, &np, bk, &ldbk, &dwork[i2], &n);
    mb02vd("N", n, np, dwork, np, iwork, &dwork[i2], n, &info2);
    if (info2 != 0) {
        *info = 8;
        return;
    }

    /* Compute Im - Dk*D */
    SLC_DLASET("Full", &m, &m, &ZERO, &ONE, &dwork[i3], &m);
    SLC_DGEMM("N", "N", &m, &m, &np, &negone, dk, &lddk, d, &ldd, &ONE, &dwork[i3], &m);

    /* Compute B*(Im - Dk*D)^{-1} */
    SLC_DLACPY("Full", &n, &m, b, &ldb, &dwork[i4], &n);
    mb02vd("N", n, m, &dwork[i3], m, iwork, &dwork[i4], n, &info2);
    if (info2 != 0) {
        *info = 9;
        return;
    }

    /* Compute D*Ck */
    SLC_DGEMM("N", "N", &np, &n, &m, &ONE, d, &ldd, ck, &ldck, &ZERO, &dwork[i5], &np);

    /* Compute Dk*C */
    SLC_DGEMM("N", "N", &m, &n, &np, &ONE, dk, &lddk, c, &ldc, &ZERO, &dwork[i6], &m);

    /* Compute the closed-loop state matrix */
    SLC_DLACPY("F", &n, &n, a, &lda, &dwork[i7], &n2);
    SLC_DGEMM("N", "N", &n, &n, &m, &ONE, &dwork[i4], &n, &dwork[i6], &m, &ONE, &dwork[i7], &n2);
    SLC_DGEMM("N", "N", &n, &n, &m, &ONE, &dwork[i4], &n, ck, &ldck, &ZERO, &dwork[i7 + n2*n], &n2);
    SLC_DGEMM("N", "N", &n, &n, &np, &ONE, &dwork[i2], &n, c, &ldc, &ZERO, &dwork[i7 + n], &n2);
    SLC_DLACPY("F", &n, &n, ak, &ldak, &dwork[i7 + n2*n + n], &n2);
    SLC_DGEMM("N", "N", &n, &n, &np, &ONE, &dwork[i2], &n, &dwork[i5], &np, &ONE, &dwork[i7 + n2*n + n], &n2);

    /* Compute the closed-loop poles */
    ldw = ldwork - iwrk + 1;
    SLC_DGEES("N", "N", select_none, &n2, &dwork[i7], &n2, &sdim,
              &dwork[i8], &dwork[i9], &dwork[iwrk], &n2,
              &dwork[iwrk], &ldw, bwork, &info2);
    if (info2 != 0) {
        *info = 3;
        return;
    }
    tmp = (i32)dwork[iwrk] + iwrk - 1;
    if (tmp > lwamax) lwamax = tmp;

    /* Check stability of the closed-loop system */
    ns = 0;
    for (i = 0; i < n2; i++) {
        f64 absval = SLC_DLAPY2(&dwork[i8 + i], &dwork[i9 + i]);
        if (absval > ONE) ns++;
    }

    if (ns > 0) {
        *info = 10;
        return;
    }

    dwork[0] = (f64)lwamax;
}
