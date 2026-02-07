/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB10RD - H-infinity (sub)optimal controller matrices from F and H
 *
 * Computes the matrices of an H-infinity (sub)optimal controller:
 *
 *          | AK | BK |
 *      K = |----|----|
 *          | CK | DK |
 *
 * from the state feedback matrix F and output injection matrix H as
 * determined by SB10QD.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>

void sb10rd(
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
    const f64* f,
    const i32 ldf,
    const f64* h,
    const i32 ldh,
    const f64* tu,
    const i32 ldtu,
    const f64* ty,
    const i32 ldty,
    const f64* x,
    const i32 ldx,
    const f64* y,
    const i32 ldy,
    f64* ak,
    const i32 ldak,
    f64* bk,
    const i32 ldbk,
    f64* ck,
    const i32 ldck,
    f64* dk,
    const i32 lddk,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
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
    } else if (ldtu < (m2 > 1 ? m2 : 1)) {
        *info = -20;
    } else if (ldty < (np2 > 1 ? np2 : 1)) {
        *info = -22;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -24;
    } else if (ldy < (n > 1 ? n : 1)) {
        *info = -26;
    } else if (ldak < (n > 1 ? n : 1)) {
        *info = -28;
    } else if (ldbk < (n > 1 ? n : 1)) {
        *info = -30;
    } else if (ldck < (m2 > 1 ? m2 : 1)) {
        *info = -32;
    } else if (lddk < (m2 > 1 ? m2 : 1)) {
        *info = -34;
    } else {
        i32 nd1 = np1 - m2;
        i32 nd2 = m1 - np2;
        i32 t1 = nd1 * nd1 + (2*nd1 > (nd1 + nd2)*np2 ? 2*nd1 : (nd1 + nd2)*np2);
        i32 t2 = nd2 * nd2 + (2*nd2 > nd2*m2 ? 2*nd2 : nd2*m2);
        i32 t3 = 3 * n;
        i32 t4a = m2*m2 + 3*m2;
        i32 t4b = np2 * (2*np2 + m2 + (np2 > n ? np2 : n));
        i32 t4 = m2*np2 + (t4a > t4b ? t4a : t4b);
        i32 t5 = n * (2*np2 + m2) + (2*n*m2 > t4 ? 2*n*m2 : t4);
        i32 maxterm = t1;
        if (t2 > maxterm) maxterm = t2;
        if (t3 > maxterm) maxterm = t3;
        if (t5 > maxterm) maxterm = t5;
        i32 minwrk = m2*np2 + np2*np2 + m2*m2 + maxterm;
        if (minwrk < 1) minwrk = 1;

        if (ldwork < minwrk) {
            *info = -37;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0 || np == 0 || m1 == 0 || m2 == 0 || np1 == 0 || np2 == 0) {
        dwork[0] = one;
        return;
    }

    f64 eps = SLC_DLAMCH("Epsilon");

    i32 id11 = 0;
    i32 id21 = id11 + m2 * np2;
    i32 id12 = id21 + np2 * np2;
    i32 iw1 = id12 + m2 * m2;
    i32 iw2 = iw1 + (np1 - m2) * (np1 - m2);
    i32 iw3 = iw2 + (np1 - m2) * np2;
    i32 iw4;
    i32 iwrk = iw2;
    i32 nd1 = np1 - m2;
    i32 nd2 = m1 - np2;
    i32 lwamax = 0;
    i32 info2;

    // Set D11HAT := -D1122
    for (i32 j = 0; j < np2; j++) {
        for (i32 i = 0; i < m2; i++) {
            dwork[id11 + i + j * m2] = -d[(nd1 + i) + (nd2 + j) * ldd];
        }
    }

    // Set D21HAT := Inp2
    SLC_DLASET("Upper", &np2, &np2, &zero, &one, &dwork[id21], &np2);

    // Set D12HAT := Im2
    SLC_DLASET("Lower", &m2, &m2, &zero, &one, &dwork[id12], &m2);

    // Compute D11HAT, D21HAT, D12HAT
    f64 anorm, rcond;
    f64 gamma2 = gamma * gamma;

    if (nd1 > 0) {
        if (nd2 == 0) {
            // D21HAT'*D21HAT = Inp2 - D1112'*D1112/gamma^2
            f64 alpha = -one / gamma2;
            SLC_DSYRK("U", "T", &np2, &nd1, &alpha, d, &ldd, &one, &dwork[id21], &np2);
        } else {
            // gdum = gamma^2*Ind1 - D1111*D1111'
            SLC_DLASET("U", &nd1, &nd1, &zero, &gamma2, &dwork[iw1], &nd1);
            f64 minusone = -one;
            SLC_DSYRK("U", "N", &nd1, &nd2, &minusone, d, &ldd, &one, &dwork[iw1], &nd1);

            anorm = SLC_DLANSY("I", "U", &nd1, &dwork[iw1], &nd1, &dwork[iwrk]);

            i32 work_avail = ldwork - iwrk;
            SLC_DSYTRF("U", &nd1, &dwork[iw1], &nd1, iwork, &dwork[iwrk], &work_avail, &info2);
            if (info2 > 0) {
                *info = 1;
                return;
            }
            i32 used = (i32)dwork[iwrk] + iwrk;
            if (used > lwamax) lwamax = used;

            SLC_DSYCON("U", &nd1, &dwork[iw1], &nd1, iwork, &anorm, &rcond, &dwork[iwrk], &iwork[nd1], &info2);

            if (rcond < eps) {
                *info = 1;
                return;
            }

            // inv(gdum)*D1112
            SLC_DLACPY("Full", &nd1, &np2, &d[nd2 * ldd], &ldd, &dwork[iw2], &nd1);
            SLC_DSYTRS("U", &nd1, &np2, &dwork[iw1], &nd1, iwork, &dwork[iw2], &nd1, &info2);

            // D11HAT = -D1121*D1111'*inv(gdum)*D1112 - D1122
            SLC_DGEMM("T", "N", &nd2, &np2, &nd1, &one, d, &ldd, &dwork[iw2], &nd1, &zero, &dwork[iw3], &nd2);
            f64 minusone2 = -one;
            SLC_DGEMM("N", "N", &m2, &np2, &nd2, &minusone2, &d[nd1], &ldd, &dwork[iw3], &nd2, &one, &dwork[id11], &m2);

            // D21HAT'*D21HAT = Inp2 - D1112'*inv(gdum)*D1112
            slicot_mb01rx('L', 'U', 'T', np2, nd1, one, -one, &dwork[id21], np2, &d[nd2 * ldd], ldd, &dwork[iw2], nd1);

            iw2 = iw1 + nd2 * nd2;
            iwrk = iw2;

            // gdum = gamma^2*Ind2 - D1111'*D1111
            SLC_DLASET("L", &nd2, &nd2, &zero, &gamma2, &dwork[iw1], &nd2);
            f64 minusone3 = -one;
            SLC_DSYRK("L", "T", &nd2, &nd1, &minusone3, d, &ldd, &one, &dwork[iw1], &nd2);

            anorm = SLC_DLANSY("I", "L", &nd2, &dwork[iw1], &nd2, &dwork[iwrk]);

            work_avail = ldwork - iwrk;
            SLC_DSYTRF("L", &nd2, &dwork[iw1], &nd2, iwork, &dwork[iwrk], &work_avail, &info2);
            if (info2 > 0) {
                *info = 1;
                return;
            }
            used = (i32)dwork[iwrk] + iwrk;
            if (used > lwamax) lwamax = used;

            SLC_DSYCON("L", &nd2, &dwork[iw1], &nd2, iwork, &anorm, &rcond, &dwork[iwrk], &iwork[nd2], &info2);

            if (rcond < eps) {
                *info = 1;
                return;
            }

            // inv(gdum)*D1121'
            ma02ad("Full", m2, nd2, &d[nd1], ldd, &dwork[iw2], nd2);
            SLC_DSYTRS("L", &nd2, &m2, &dwork[iw1], &nd2, iwork, &dwork[iw2], &nd2, &info2);

            // D12HAT*D12HAT' = Im2 - D1121*inv(gdum)*D1121'
            slicot_mb01rx('L', 'L', 'N', m2, nd2, one, -one, &dwork[id12], m2, &d[nd1], ldd, &dwork[iw2], nd2);
        }
    } else {
        if (nd2 > 0) {
            // D12HAT*D12HAT' = Im2 - D1121*D1121'/gamma^2
            f64 alpha = -one / gamma2;
            SLC_DSYRK("L", "N", &m2, &nd2, &alpha, d, &ldd, &one, &dwork[id12], &m2);
        }
    }

    // D21HAT using Cholesky
    SLC_DPOTRF("U", &np2, &dwork[id21], &np2, &info2);
    if (info2 > 0) {
        *info = 1;
        return;
    }

    // D12HAT using Cholesky
    SLC_DPOTRF("L", &m2, &dwork[id12], &m2, &info2);
    if (info2 > 0) {
        *info = 1;
        return;
    }

    // Z = In - Y*X/gamma^2, then LU factorization in AK
    iwrk = iw1;
    SLC_DLASET("Full", &n, &n, &zero, &one, ak, &ldak);
    f64 minusgam2inv = -one / gamma2;
    SLC_DGEMM("N", "N", &n, &n, &n, &minusgam2inv, y, &ldy, x, &ldx, &one, ak, &ldak);

    anorm = SLC_DLANGE("1", &n, &n, ak, &ldak, &dwork[iwrk]);
    SLC_DGETRF(&n, &n, ak, &ldak, iwork, &info2);
    if (info2 > 0) {
        *info = 1;
        return;
    }

    SLC_DGECON("1", &n, ak, &ldak, &anorm, &rcond, &dwork[iwrk], &iwork[n], info);

    if (rcond < eps) {
        *info = 1;
        return;
    }

    i32 iwb = iw1;
    i32 iwc = iwb + n * np2;
    iw1 = iwc + (m2 + np2) * n;
    iw2 = iw1 + n * m2;

    // BK = (C2 + F12)'
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < np2; i++) {
            bk[j + i * ldbk] = c[(np1 + i) + j * ldc] + f[(nd2 + i) + j * ldf];
        }
    }

    // (C2 + F12)*Z'
    SLC_DGETRS("Transpose", &n, &np2, ak, &ldak, iwork, bk, &ldbk, &info2);

    // F2*Z'
    ma02ad("Full", m2, n, &f[m1], ldf, &dwork[iw1], n);
    SLC_DGETRS("Transpose", &n, &m2, ak, &ldak, iwork, &dwork[iw1], &n, &info2);

    // C1HAT' = (F2*Z)' - (D11HAT*(C2 + F12)*Z)'
    f64 minusone4 = -one;
    SLC_DGEMM("N", "T", &n, &m2, &np2, &minusone4, bk, &ldbk, &dwork[id11], &m2, &one, &dwork[iw1], &n);

    // CHAT
    i32 ldc_hat = m2 + np2;
    SLC_DGEMM("N", "T", &m2, &n, &m2, &one, tu, &ldtu, &dwork[iw1], &n, &zero, &dwork[iwc], &ldc_hat);
    ma02ad("Full", n, np2, bk, ldbk, &dwork[iwc + m2], ldc_hat);
    f64 minusone5 = -one;
    SLC_DTRMM("L", "U", "N", "N", &np2, &n, &minusone5, &dwork[id21], &np2, &dwork[iwc + m2], &ldc_hat);

    // B2 + H12
    for (i32 j = 0; j < m2; j++) {
        for (i32 i = 0; i < n; i++) {
            dwork[iw2 + i + j * n] = b[i + (m1 + j) * ldb] + h[i + (nd1 + j) * ldh];
        }
    }

    // AK = A + H*C
    SLC_DLACPY("Full", &n, &n, a, &lda, ak, &ldak);
    SLC_DGEMM("N", "N", &n, &n, &np, &one, h, &ldh, c, &ldc, &one, ak, &ldak);

    // AHAT = A + HC + (B2 + H12)*C1HAT
    SLC_DGEMM("N", "T", &n, &n, &m2, &one, &dwork[iw2], &n, &dwork[iw1], &n, &one, ak, &ldak);

    // B1HAT = -H2 + (B2 + H12)*D11HAT
    SLC_DLACPY("Full", &n, &np2, &h[np1 * ldh], &ldh, bk, &ldbk);
    f64 minusone6 = -one;
    SLC_DGEMM("N", "N", &n, &np2, &m2, &one, &dwork[iw2], &n, &dwork[id11], &m2, &minusone6, bk, &ldbk);

    // BHAT1
    SLC_DGEMM("N", "N", &n, &np2, &np2, &one, bk, &ldbk, ty, &ldty, &zero, &dwork[iwb], &n);

    // Tu*D11HAT
    SLC_DGEMM("N", "N", &m2, &np2, &m2, &one, tu, &ldtu, &dwork[id11], &m2, &zero, &dwork[iw1], &m2);

    // Tu*D11HAT*Ty in DK
    SLC_DGEMM("N", "N", &m2, &np2, &np2, &one, &dwork[iw1], &m2, ty, &ldty, &zero, dk, &lddk);

    // P = Im2 + Tu*D11HAT*Ty*D22
    iw2 = iw1 + m2 * np2;
    iwrk = iw2 + m2 * m2;
    SLC_DLASET("Full", &m2, &m2, &zero, &one, &dwork[iw2], &m2);
    SLC_DGEMM("N", "N", &m2, &m2, &np2, &one, dk, &lddk, &d[(np1) + (m1) * ldd], &ldd, &one, &dwork[iw2], &m2);

    anorm = SLC_DLANGE("1", &m2, &m2, &dwork[iw2], &m2, &dwork[iwrk]);
    SLC_DGETRF(&m2, &m2, &dwork[iw2], &m2, iwork, &info2);
    if (info2 > 0) {
        *info = 2;
        return;
    }

    SLC_DGECON("1", &m2, &dwork[iw2], &m2, &anorm, &rcond, &dwork[iwrk], &iwork[m2], &info2);

    if (rcond < eps) {
        *info = 2;
        return;
    }

    // CK = inv(P)*CHAT(1:M2,:)
    SLC_DLACPY("Full", &m2, &n, &dwork[iwc], &ldc_hat, ck, &ldck);
    SLC_DGETRS("NoTranspose", &m2, &n, &dwork[iw2], &m2, iwork, ck, &ldck, &info2);

    // Q = Inp2 + D22*Tu*D11HAT*Ty and LU factorization
    iw3 = iw2 + np2 * np2;
    iw4 = iw3 + np2 * m2;
    iwrk = iw4 + np2 * np2;

    SLC_DLASET("Full", &np2, &np2, &zero, &one, &dwork[iw2], &np2);
    SLC_DGEMM("N", "N", &np2, &np2, &m2, &one, &d[(np1) + (m1) * ldd], &ldd, dk, &lddk, &one, &dwork[iw2], &np2);
    SLC_DGETRF(&np2, &np2, &dwork[iw2], &np2, iwork, &info2);
    if (info2 > 0) {
        *info = 2;
        return;
    }

    // A1 = inv(Q)*D22 and inv(Q)
    SLC_DLACPY("Full", &np2, &m2, &d[(np1) + (m1) * ldd], &ldd, &dwork[iw3], &np2);
    SLC_DGETRS("NoTranspose", &np2, &m2, &dwork[iw2], &np2, iwork, &dwork[iw3], &np2, &info2);

    i32 work_avail2 = ldwork - iwrk;
    SLC_DGETRI(&np2, &dwork[iw2], &np2, iwork, &dwork[iwrk], &work_avail2, &info2);
    i32 used2 = (i32)dwork[iwrk] + iwrk;
    if (used2 > lwamax) lwamax = used2;

    // A2 = (inv(Ty) - inv(Q)*inv(Ty) - A1*Tu*D11HAT)*inv(D21HAT)
    SLC_DLACPY("Full", &np2, &np2, ty, &ldty, &dwork[iw4], &np2);
    SLC_DGETRF(&np2, &np2, &dwork[iw4], &np2, iwork, &info2);
    SLC_DGETRI(&np2, &dwork[iw4], &np2, iwork, &dwork[iwrk], &work_avail2, &info2);

    SLC_DLACPY("Full", &np2, &np2, &dwork[iw4], &np2, &dwork[iwrk], &np2);
    f64 minusone7 = -one;
    SLC_DGEMM("N", "N", &np2, &np2, &np2, &minusone7, &dwork[iw2], &np2, &dwork[iwrk], &np2, &one, &dwork[iw4], &np2);
    f64 minusone8 = -one;
    SLC_DGEMM("N", "N", &np2, &np2, &m2, &minusone8, &dwork[iw3], &np2, &dwork[iw1], &m2, &one, &dwork[iw4], &np2);
    SLC_DTRMM("R", "U", "N", "N", &np2, &np2, &one, &dwork[id21], &np2, &dwork[iw4], &np2);

    // [A1 A2]*CHAT
    i32 m2np2 = m2 + np2;
    SLC_DGEMM("N", "N", &np2, &n, &m2np2, &one, &dwork[iw3], &np2, &dwork[iwc], &ldc_hat, &zero, &dwork[iwrk], &np2);

    // AK := AHAT - BHAT1*[A1 A2]*CHAT
    f64 minusone9 = -one;
    SLC_DGEMM("N", "N", &n, &n, &np2, &minusone9, &dwork[iwb], &n, &dwork[iwrk], &np2, &one, ak, &ldak);

    // BK := BHAT1*inv(Q)
    SLC_DGEMM("N", "N", &n, &np2, &np2, &one, &dwork[iwb], &n, &dwork[iw2], &np2, &zero, bk, &ldbk);

    // DK := Tu*D11HAT*Ty*inv(Q)
    SLC_DGEMM("N", "N", &m2, &np2, &np2, &one, dk, &lddk, &dwork[iw2], &np2, &zero, &dwork[iw3], &m2);
    SLC_DLACPY("Full", &m2, &np2, &dwork[iw3], &m2, dk, &lddk);

    dwork[0] = (f64)lwamax;
}
