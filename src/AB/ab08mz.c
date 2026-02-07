/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>

i32 slicot_ab8nxz(i32 n, i32 m, i32 p, i32* ro, i32* sigma, f64 svlmax,
                  c128* abcd, i32 ldabcd, i32* ninfz, i32* infz, i32* kronl,
                  i32* mu, i32* nu, i32* nkrol, f64 tol, i32* iwork,
                  f64* dwork, c128* zwork, i32 lzwork);

i32 slicot_ab08mz(char equil, i32 n, i32 m, i32 p, c128* a, i32 lda,
                  c128* b, i32 ldb, c128* c, i32 ldc, c128* d, i32 ldd,
                  i32* rank, f64 tol, i32* iwork, f64* dwork, c128* zwork,
                  i32 lzwork) {
    const f64 DZERO = 0.0;
    const f64 DONE = 1.0;

    i32 np = n + p;
    i32 nm = n + m;
    i32 info = 0;
    char equil_up = (char)toupper((unsigned char)equil);
    bool lequil = (equil_up == 'S');
    bool lquery = (lzwork == -1);
    i32 wrkopt = np * nm;

    if (!lequil && equil_up != 'N') {
        info = -1;
    } else if (n < 0) {
        info = -2;
    } else if (m < 0) {
        info = -3;
    } else if (p < 0) {
        info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        info = -6;
    } else if (ldb < (n > 1 ? n : 1)) {
        info = -8;
    } else if (ldc < (p > 1 ? p : 1)) {
        info = -10;
    } else if (ldd < (p > 1 ? p : 1)) {
        info = -12;
    } else {
        i32 minpm = (p < m) ? p : m;
        i32 minpn = (p < n) ? p : n;
        i32 t1 = minpm + ((3 * m - 1 > n) ? 3 * m - 1 : n);
        i32 t2 = minpn + ((3 * p - 1 > np) ? ((3 * p - 1 > nm) ? 3 * p - 1 : nm) : ((np > nm) ? np : nm));
        i32 kw = wrkopt + ((t1 > 1) ? ((t1 > t2) ? t1 : t2) : ((1 > t2) ? 1 : t2));

        if (lquery) {
            f64 svlmax_q = DZERO;
            i32 ninfz_q = 0;
            i32 ro_q = p;
            i32 sigma_q = 0;
            i32 mu_q, nu_q, nkrol_q;
            slicot_ab8nxz(n, m, p, &ro_q, &sigma_q, svlmax_q, zwork, (np > 1 ? np : 1),
                         &ninfz_q, iwork, iwork, &mu_q, &nu_q, &nkrol_q, tol,
                         iwork, dwork, zwork, -1);
            i32 opt = wrkopt + (i32)creal(zwork[0]);
            wrkopt = (kw > opt) ? kw : opt;
        } else if (lzwork < kw) {
            info = -17;
        }
    }

    if (info != 0) {
        i32 xinfo = -info;
        SLC_XERBLA("AB08MZ", &xinfo);
        return info;
    } else if (lquery) {
        zwork[0] = (f64)wrkopt + 0.0*I;
        return 0;
    }

    i32 minmp = (m < p) ? m : p;
    if (minmp == 0) {
        *rank = 0;
        zwork[0] = DONE + 0.0*I;
        return 0;
    }

    for (i32 i = 0; i < 2 * n + 1; i++) {
        iwork[i] = 0;
    }

    i32 n_int = n, m_int = m, p_int = p;
    i32 ldnp = np;

    SLC_ZLACPY("Full", &n_int, &m_int, b, &ldb, zwork, &ldnp);
    SLC_ZLACPY("Full", &p_int, &m_int, d, &ldd, &zwork[n], &ldnp);
    SLC_ZLACPY("Full", &n_int, &n_int, a, &lda, &zwork[np * m], &ldnp);
    SLC_ZLACPY("Full", &p_int, &n_int, c, &ldc, &zwork[np * m + n], &ldnp);

    i32 kw = wrkopt;

    if (lequil) {
        f64 maxred = DZERO;
        slicot_tb01iz('A', n, m, p, &maxred, &zwork[np * m], np, zwork, np,
                     &zwork[np * m + n], np, dwork);
    }

    f64 thresh = sqrt((f64)(np * nm)) * SLC_DLAMCH("Precision");
    f64 toler = tol;
    if (toler < thresh) toler = thresh;

    f64 svlmax = SLC_ZLANGE("Frobenius", &np, &nm, zwork, &ldnp, dwork);

    i32 ro = p;
    i32 sigma = 0;
    i32 ninfz = 0;
    i32 mu, nu, nkrol;

    slicot_ab8nxz(n, m, p, &ro, &sigma, svlmax, zwork, np, &ninfz, iwork,
                 &iwork[n], &mu, &nu, &nkrol, toler, &iwork[2 * n + 1],
                 dwork, &zwork[kw], lzwork - kw + 1);

    *rank = mu;

    i32 opt = (i32)creal(zwork[kw]) + kw - 1;
    zwork[0] = (f64)((wrkopt > opt) ? wrkopt : opt) + 0.0*I;

    return 0;
}
