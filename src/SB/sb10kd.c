/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB10KD - Discrete-time loop shaping controller design
 *
 * Computes the matrices of the positive feedback controller
 *     K = [Ak, Bk; Ck, Dk]
 * for the shaped plant
 *     G = [A, B; C, 0]
 * using the McFarlane-Glover method.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>

static int sb10kd_select(const f64* reig, const f64* ieig) {
    (void)reig;
    (void)ieig;
    return 0;
}

void sb10kd(
    const i32 n,
    const i32 m,
    const i32 np,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64 factor,
    f64* ak,
    const i32 ldak,
    f64* bk,
    const i32 ldbk,
    f64* ck,
    const i32 ldck,
    f64* dk,
    const i32 lddk,
    f64* rcond,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    i32 i, j, n2, ns, sdim, info2;
    i32 i1, i2, i3, i4, i5, i6, i7, i8, i9;
    i32 i10, i11, i12, i13, i14, i15, i16, i17, i18, i19;
    i32 i20, i21, i22, i23, i24, i25, i26;
    i32 iwrk, lwa, lwamax, minwrk;
    f64 gamma, rnorm;
    f64 neg_one = -1.0;

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
    } else if (factor < one) {
        *info = -10;
    } else if (ldak < (n > 1 ? n : 1)) {
        *info = -12;
    } else if (ldbk < (n > 1 ? n : 1)) {
        *info = -14;
    } else if (ldck < (m > 1 ? m : 1)) {
        *info = -16;
    } else if (lddk < (m > 1 ? m : 1)) {
        *info = -18;
    }

    if (*info == 0) {
        i32 t1 = 14*n + 23;
        i32 t2 = 16*n;
        i32 t3 = 2*n + np + m;
        i32 t4 = 3*(np + m);
        i32 max1 = t1 > t2 ? t1 : t2;
        max1 = max1 > t3 ? max1 : t3;
        max1 = max1 > t4 ? max1 : t4;

        i32 s1 = n*n;
        i32 s2 = 11*n*np + 2*m*m + 8*np*np + 8*m*n + 4*m*np + np;
        i32 max2 = s1 > s2 ? s1 : s2;

        minwrk = 15*n*n + 6*n + max1 + max2;
        if (ldwork < minwrk) {
            *info = -22;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0 || np == 0) {
        rcond[0] = one;
        rcond[1] = one;
        rcond[2] = one;
        rcond[3] = one;
        dwork[0] = one;
        return;
    }

    n2 = 2*n;
    i1 = n*n;
    i2 = i1 + n*n;
    i3 = i2 + n*n;
    i4 = i3 + n*n;
    i5 = i4 + n2;
    i6 = i5 + n2;
    i7 = i6 + n2;
    i8 = i7 + n2*n2;
    i9 = i8 + n2*n2;

    iwrk = i9 + n2*n2;
    lwamax = 0;

    // Compute Cr = C'*C
    SLC_DSYRK("U", "T", &n, &np, &one, c, &ldc, &zero, &dwork[i2], &n);

    // Compute Dr = B*B'
    SLC_DSYRK("U", "N", &n, &m, &one, b, &ldb, &zero, &dwork[i3], &n);

    // Solution of P-Riccati: A'*P*(I + Dr*P)^{-1}*A - P + Cr = 0
    sb02od("D", "G", "N", "U", "Z", "S", n, m, np, a, lda,
           &dwork[i3], n, &dwork[i2], n, dwork, m, dwork,
           n, &rcond[0], dwork, n, &dwork[i4],
           &dwork[i5], &dwork[i6], &dwork[i7], n2,
           &dwork[i8], n2, &dwork[i9], n2, neg_one, iwork,
           &dwork[iwrk], ldwork - iwrk, &info2);
    if (info2 != 0) {
        *info = 1;
        return;
    }
    lwa = (i32)dwork[iwrk] + iwrk;
    lwamax = lwa > lwamax ? lwa : lwamax;

    // Transpose A in AK (workspace)
    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
            ak[i + j*ldak] = a[j + i*lda];
        }
    }

    // Solution of Q-Riccati: A*Q*(I + Cr*Q)^{-1}*A' - Q + Dr = 0
    sb02od("D", "G", "N", "U", "Z", "S", n, m, np, ak, ldak,
           &dwork[i2], n, &dwork[i3], n, dwork, m, dwork,
           n, &rcond[1], &dwork[i1], n, &dwork[i4],
           &dwork[i5], &dwork[i6], &dwork[i7], n2,
           &dwork[i8], n2, &dwork[i9], n2, neg_one, iwork,
           &dwork[iwrk], ldwork - iwrk, &info2);
    if (info2 != 0) {
        *info = 2;
        return;
    }
    lwa = (i32)dwork[iwrk] + iwrk;
    lwamax = lwa > lwamax ? lwa : lwamax;

    // Compute gamma: max eigenvalue of Q*P, then gamma = factor*sqrt(1+max_eig)
    SLC_DGEMM("N", "N", &n, &n, &n, &one, &dwork[i1], &n, dwork, &n,
              &zero, ak, &ldak);
    SLC_DGEES("N", "N", sb10kd_select, &n, ak, &ldak, &sdim, &dwork[i6],
              &dwork[i7], &dwork[iwrk], &n, &dwork[iwrk],
              &ldwork, bwork, &info2);
    if (info2 != 0) {
        *info = 4;
        return;
    }
    lwa = (i32)dwork[iwrk] + iwrk;
    lwamax = lwa > lwamax ? lwa : lwamax;

    gamma = zero;
    for (i = 0; i < n; i++) {
        if (dwork[i6 + i] > gamma) {
            gamma = dwork[i6 + i];
        }
    }
    gamma = factor * sqrt(one + gamma);

    // Workspace usage for second phase (i1, i2 unchanged from first phase)
    // i1 = n*n (Q solution), i2 = 2*n*n (now used for Q*C')
    i3 = i2 + n*np;
    i4 = i3 + np*np;
    i5 = i4 + np*np;
    i6 = i5 + np*np;
    i7 = i6 + np;
    i8 = i7 + np*np;
    i9 = i8 + np*np;
    i10 = i9 + np*np;
    i11 = i10 + n*np;
    i12 = i11 + n*np;
    i13 = i12 + (np+m)*(np+m);
    i14 = i13 + n*(np+m);
    i15 = i14 + n*(np+m);
    i16 = i15 + n*n;
    i17 = i16 + n2;
    i18 = i17 + n2;
    i19 = i18 + n2;
    i20 = i19 + (n2+np+m)*(n2+np+m);
    i21 = i20 + (n2+np+m)*n2;

    iwrk = i21 + n2*n2;

    // Compute Q*C'
    SLC_DGEMM("N", "T", &n, &np, &n, &one, &dwork[i1], &n, c, &ldc,
              &zero, &dwork[i2], &n);

    // Compute I_p + C*Q*C'
    SLC_DLASET("F", &np, &np, &zero, &one, &dwork[i3], &np);
    SLC_DGEMM("N", "N", &np, &np, &n, &one, c, &ldc, &dwork[i2], &n,
              &one, &dwork[i3], &np);

    // Eigendecomposition of I_p + C*Q*C'
    SLC_DLACPY("U", &np, &np, &dwork[i3], &np, &dwork[i5], &np);
    i32 lwork_syev = ldwork - iwrk;
    SLC_DSYEV("V", "U", &np, &dwork[i5], &np, &dwork[i6],
              &dwork[iwrk], &lwork_syev, &info2);
    if (info2 != 0) {
        *info = 4;
        return;
    }
    lwa = (i32)dwork[iwrk] + iwrk;
    lwamax = lwa > lwamax ? lwa : lwamax;

    // Compute (I_p + C*Q*C')^{-1} = V * D^{-1} * V'
    for (j = 0; j < np; j++) {
        for (i = 0; i < np; i++) {
            dwork[i9 + i + j*np] = dwork[i5 + j + i*np] / dwork[i6 + i];
        }
    }
    SLC_DGEMM("N", "N", &np, &np, &np, &one, &dwork[i5], &np,
              &dwork[i9], &np, &zero, &dwork[i4], &np);

    // Compute Z2 = V * D^{-1/2} * V'
    for (j = 0; j < np; j++) {
        for (i = 0; i < np; i++) {
            dwork[i9 + i + j*np] = dwork[i5 + j + i*np] / sqrt(dwork[i6 + i]);
        }
    }
    SLC_DGEMM("N", "N", &np, &np, &np, &one, &dwork[i5], &np,
              &dwork[i9], &np, &zero, &dwork[i7], &np);

    // Compute Z2^{-1} = V * D^{1/2} * V'
    for (j = 0; j < np; j++) {
        for (i = 0; i < np; i++) {
            dwork[i9 + i + j*np] = dwork[i5 + j + i*np] * sqrt(dwork[i6 + i]);
        }
    }
    SLC_DGEMM("N", "N", &np, &np, &np, &one, &dwork[i5], &np,
              &dwork[i9], &np, &zero, &dwork[i8], &np);

    // Compute A*Q*C'
    SLC_DGEMM("N", "N", &n, &np, &n, &one, a, &lda, &dwork[i2], &n,
              &zero, &dwork[i10], &n);

    // Compute H = -A*Q*C' * (I_p + C*Q*C')^{-1}
    SLC_DGEMM("N", "N", &n, &np, &np, &neg_one, &dwork[i10], &n,
              &dwork[i4], &np, &zero, &dwork[i11], &n);

    // Compute Rx = I_{np+m}
    i32 npm = np + m;
    SLC_DLASET("F", &npm, &npm, &zero, &one, &dwork[i12], &npm);

    // Rx(1:np,1:np) = I_p + C*Q*C' - gamma^2 * I_p
    for (j = 0; j < np; j++) {
        for (i = 0; i < np; i++) {
            dwork[i12 + i + j*npm] = dwork[i3 + i + j*np];
        }
        dwork[i12 + j + j*npm] = dwork[i3 + j + j*np] - gamma*gamma;
    }

    // Compute Bx = [-H*Z2^{-1}, B]
    SLC_DGEMM("N", "N", &n, &np, &np, &neg_one, &dwork[i11], &n,
              &dwork[i8], &np, &zero, &dwork[i13], &n);
    for (j = 0; j < m; j++) {
        for (i = 0; i < n; i++) {
            dwork[i13 + n*np + i + j*n] = b[i + j*ldb];
        }
    }

    // Compute Sx = [C'*Z2^{-1}, 0]
    SLC_DGEMM("T", "N", &n, &np, &np, &one, c, &ldc, &dwork[i8], &np,
              &zero, &dwork[i14], &n);
    SLC_DLASET("F", &n, &m, &zero, &zero, &dwork[i14 + n*np], &n);

    // Solve X-Riccati equation
    sb02od("D", "B", "C", "U", "N", "S", n, npm, np, a, lda,
           &dwork[i13], n, (f64*)c, ldc, &dwork[i12], npm,
           &dwork[i14], n, &rcond[2], &dwork[i15], n,
           &dwork[i16], &dwork[i17], &dwork[i18],
           &dwork[i19], n2+npm, &dwork[i20], n2+npm,
           &dwork[i21], n2, neg_one, iwork,
           &dwork[iwrk], ldwork - iwrk, &info2);
    if (info2 != 0) {
        *info = 3;
        return;
    }
    lwa = (i32)dwork[iwrk] + iwrk;
    lwamax = lwa > lwamax ? lwa : lwamax;

    i22 = i16;
    i23 = i22 + npm*n;
    i24 = i23 + npm*npm;
    i25 = i24 + npm*n;
    i26 = i25 + m*n;

    iwrk = i25;

    // Compute Bx'*X
    SLC_DGEMM("T", "N", &npm, &n, &n, &one, &dwork[i13], &n,
              &dwork[i15], &n, &zero, &dwork[i22], &npm);

    // Compute Rx + Bx'*X*Bx
    SLC_DLACPY("F", &npm, &npm, &dwork[i12], &npm, &dwork[i23], &npm);
    SLC_DGEMM("N", "N", &npm, &npm, &n, &one, &dwork[i22], &npm,
              &dwork[i13], &n, &one, &dwork[i23], &npm);

    // Compute -(Sx' + Bx'*X*A)
    for (j = 0; j < n; j++) {
        for (i = 0; i < npm; i++) {
            dwork[i24 + i + j*npm] = dwork[i14 + j + i*n];
        }
    }
    SLC_DGEMM("N", "N", &npm, &n, &n, &neg_one, &dwork[i22], &npm,
              a, &lda, &neg_one, &dwork[i24], &npm);

    // Factorize Rx + Bx'*X*Bx
    rnorm = SLC_DLANSY("1", "U", &npm, &dwork[i23], &npm, &dwork[iwrk]);
    i32 lwork_sytrf = ldwork - iwrk;
    SLC_DSYTRF("U", &npm, &dwork[i23], &npm, iwork, &dwork[iwrk], &lwork_sytrf, &info2);
    if (info2 != 0) {
        *info = 5;
        return;
    }
    lwa = (i32)dwork[iwrk] + iwrk;
    lwamax = lwa > lwamax ? lwa : lwamax;

    SLC_DSYCON("U", &npm, &dwork[i23], &npm, iwork, &rnorm,
               &rcond[3], &dwork[iwrk], &iwork[npm], &info2);

    // Solve for F = -(Rx + Bx'*X*Bx)^{-1} * (Sx' + Bx'*X*A)
    SLC_DSYTRS("U", &npm, &n, &dwork[i23], &npm, iwork, &dwork[i24], &npm, &info2);

    // Compute B'*X
    SLC_DGEMM("T", "N", &m, &n, &n, &one, b, &ldb, &dwork[i15], &n,
              &zero, &dwork[i25], &m);

    // Compute I_m + B'*X*B
    SLC_DLASET("F", &m, &m, &zero, &one, &dwork[i23], &m);
    SLC_DGEMM("N", "N", &m, &m, &n, &one, &dwork[i25], &m, b, &ldb,
              &one, &dwork[i23], &m);

    // Factorize I_m + B'*X*B (Cholesky)
    SLC_DPOTRF("U", &m, &dwork[i23], &m, &info2);

    // Solve (I_m + B'*X*B)^{-1} * B'*X
    SLC_DPOTRS("U", &m, &n, &dwork[i23], &m, &dwork[i25], &m, &info2);

    // Compute Dk = (I_m + B'*X*B)^{-1} * B'*X * H
    SLC_DGEMM("N", "N", &m, &np, &n, &one, &dwork[i25], &m,
              &dwork[i11], &n, &zero, dk, &lddk);

    // Compute Bk = -H + B*Dk
    SLC_DLACPY("F", &n, &np, &dwork[i11], &n, bk, &ldbk);
    SLC_DGEMM("N", "N", &n, &np, &m, &one, b, &ldb, dk, &lddk,
              &neg_one, bk, &ldbk);

    // Compute Dk*Z2^{-1}
    SLC_DGEMM("N", "N", &m, &np, &np, &one, dk, &lddk, &dwork[i8], &np,
              &zero, &dwork[i26], &m);

    // Compute F1 + Z2*C
    SLC_DLACPY("F", &np, &n, &dwork[i24], &npm, &dwork[i12], &np);
    SLC_DGEMM("N", "N", &np, &n, &np, &one, &dwork[i7], &np, c, &ldc,
              &one, &dwork[i12], &np);

    // Compute Ck = F2 - Dk*Z2^{-1}*(F1 + Z2*C)
    SLC_DLACPY("F", &m, &n, &dwork[i24 + np], &npm, ck, &ldck);
    SLC_DGEMM("N", "N", &m, &n, &np, &neg_one, &dwork[i26], &m,
              &dwork[i12], &np, &one, ck, &ldck);

    // Compute Ak = A + H*C + B*Ck
    SLC_DLACPY("F", &n, &n, a, &lda, ak, &ldak);
    SLC_DGEMM("N", "N", &n, &n, &np, &one, &dwork[i11], &n, c, &ldc,
              &one, ak, &ldak);
    SLC_DGEMM("N", "N", &n, &n, &m, &one, b, &ldb, ck, &ldck,
              &one, ak, &ldak);

    // Workspace usage for closed-loop stability check
    i1 = m*n;
    i2 = i1 + n2*n2;
    i3 = i2 + n2;

    iwrk = i3 + n2;

    // Compute Dk*C
    SLC_DGEMM("N", "N", &m, &n, &np, &one, dk, &lddk, c, &ldc,
              &zero, dwork, &m);

    // Form closed-loop state matrix
    SLC_DLACPY("F", &n, &n, a, &lda, &dwork[i1], &n2);
    SLC_DGEMM("N", "N", &n, &n, &m, &neg_one, b, &ldb, dwork, &m,
              &one, &dwork[i1], &n2);
    SLC_DGEMM("N", "N", &n, &n, &np, &neg_one, bk, &ldbk, c, &ldc,
              &zero, &dwork[i1 + n], &n2);
    SLC_DGEMM("N", "N", &n, &n, &m, &one, b, &ldb, ck, &ldck,
              &zero, &dwork[i1 + n2*n], &n2);
    SLC_DLACPY("F", &n, &n, ak, &ldak, &dwork[i1 + n2*n + n], &n2);

    // Compute closed-loop eigenvalues
    i32 lwork_dgees = ldwork - iwrk;
    SLC_DGEES("N", "N", sb10kd_select, &n2, &dwork[i1], &n2, &sdim,
              &dwork[i2], &dwork[i3], &dwork[iwrk], &n,
              &dwork[iwrk], &lwork_dgees, bwork, &info2);
    if (info2 != 0) {
        *info = 4;
        return;
    }
    lwa = (i32)dwork[iwrk] + iwrk;
    lwamax = lwa > lwamax ? lwa : lwamax;

    // Check closed-loop stability
    ns = 0;
    for (i = 0; i < n2; i++) {
        f64 abs_eig = SLC_DLAPY2(&dwork[i2 + i], &dwork[i3 + i]);
        if (abs_eig > one) {
            ns++;
        }
    }
    if (ns > 0) {
        *info = 6;
        return;
    }

    dwork[0] = (f64)lwamax;
}
