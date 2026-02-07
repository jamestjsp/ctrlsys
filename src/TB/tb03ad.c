/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>

void tb03ad(const char* leri, const char* equil, i32 n, i32 m, i32 p,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            f64* d, i32 ldd, i32* nr, i32* index,
            f64* pcoeff, i32 ldpco1, i32 ldpco2,
            f64* qcoeff, i32 ldqco1, i32 ldqco2,
            f64* vcoeff, i32 ldvco1, i32 ldvco2,
            f64 tol, i32* iwork, f64* dwork, i32 ldwork, i32* info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    *info = 0;

    char leri_char = (*leri == 'l' || *leri == 'L') ? 'L' :
                     ((*leri == 'r' || *leri == 'R') ? 'R' : '\0');
    char equil_char = (*equil == 's' || *equil == 'S') ? 'S' :
                      ((*equil == 'n' || *equil == 'N') ? 'N' : '\0');

    bool lleril = (leri_char == 'L');
    bool llerir = (leri_char == 'R');
    bool lequil = (equil_char == 'S');

    i32 maxmp = (m > p) ? m : p;
    i32 mplim = (1 > maxmp) ? 1 : maxmp;

    i32 pwork, mwork;
    if (llerir) {
        pwork = m;
        mwork = p;
    } else {
        pwork = p;
        mwork = m;
    }

    i32 max1n = (1 > n) ? 1 : n;
    i32 max1pwork = (1 > pwork) ? 1 : pwork;
    i32 max1mwork = (1 > mwork) ? 1 : mwork;

    if (!lleril && !llerir) {
        *info = -1;
    } else if (!lequil && equil_char != 'N') {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (p < 0) {
        *info = -5;
    } else if (lda < max1n) {
        *info = -7;
    } else if (ldb < max1n) {
        *info = -9;
    } else if (ldc < mplim) {
        *info = -11;
    } else if (ldd < mplim) {
        *info = -13;
    } else if (ldpco1 < max1pwork) {
        *info = -17;
    } else if (ldpco2 < max1pwork) {
        *info = -18;
    } else if (ldqco1 < max1pwork || (llerir && ldqco1 < mplim)) {
        *info = -20;
    } else if (ldqco2 < max1mwork || (llerir && ldqco2 < mplim)) {
        *info = -21;
    } else if (ldvco1 < max1pwork) {
        *info = -23;
    } else if (ldvco2 < max1n) {
        *info = -24;
    } else {
        i32 ldw_req1 = n + maxmp * 3;
        if (n > maxmp * 3) ldw_req1 = n + n;
        i32 ldw_req2 = pwork * (pwork + 2);
        i32 ldw_req = (ldw_req1 > ldw_req2) ? ldw_req1 : ldw_req2;
        if (ldw_req < 1) ldw_req = 1;
        if (ldwork < ldw_req) {
            *info = -28;
        }
    }

    if (*info != 0) {
        return;
    }

    i32 max_nmp = n;
    if (m > max_nmp) max_nmp = m;
    if (p > max_nmp) max_nmp = p;

    if (max_nmp == 0) {
        *nr = 0;
        dwork[0] = ONE;
        return;
    }

    if (llerir) {
        ab07md('D', n, m, p, a, lda, b, ldb, c, ldc, d, ldd);
    }

    if (lequil) {
        f64 maxred = ZERO;
        i32 local_info;
        tb01id("A", n, mwork, pwork, &maxred, a, lda, b, ldb, c, ldc,
               dwork, &local_info);
    }

    i32 iz = 0;
    i32 itau = 0;
    i32 jwork = itau + n;

    i32 ncont, indblk, local_info;
    tb01ud("N", n, mwork, pwork, a, lda, b, ldb, c, ldc,
           &ncont, &indblk, iwork, &dwork[iz], 1, &dwork[itau], tol,
           &iwork[n], &dwork[jwork], ldwork - jwork, &local_info);

    i32 wrkopt = (i32)dwork[jwork] + jwork;

    ab07md('Z', ncont, mwork, pwork, a, lda, b, ldb, c, ldc, dwork, 1);

    i32 nr_val;
    tb01ud("N", ncont, pwork, mwork, a, lda, b, ldb, c, ldc,
           &nr_val, &indblk, iwork, &dwork[iz], 1, &dwork[itau], tol,
           &iwork[n], &dwork[jwork], ldwork - jwork, &local_info);
    *nr = nr_val;

    i32 wtemp = (i32)dwork[jwork] + jwork;
    if (wtemp > wrkopt) wrkopt = wtemp;

    ab07md('Z', *nr, pwork, mwork, a, lda, b, ldb, c, ldc, dwork, 1);

    for (i32 i = indblk; i < n; i++) {
        iwork[i] = 0;
    }

    for (i32 k = 0; k < n + 1; k++) {
        SLC_DLASET("F", &pwork, &pwork, &ZERO, &ZERO,
                   &pcoeff[k * ldpco1 * ldpco2], &ldpco1);
        SLC_DLASET("F", &pwork, &mwork, &ZERO, &ZERO,
                   &qcoeff[k * ldqco1 * ldqco2], &ldqco1);
        i32 nr_loc = *nr;
        SLC_DLASET("F", &pwork, &nr_loc, &ZERO, &ZERO,
                   &vcoeff[k * ldvco1 * ldvco2], &ldvco1);
    }

    i32 inplus = indblk + 1;
    i32 istart = 0;
    i32 joff = *nr;

    for (i32 k = 1; k <= indblk; k++) {
        i32 kwork = inplus - k;
        i32 kplus = kwork + 1;
        i32 istop = iwork[kwork - 1];
        joff = joff - istop;

        for (i32 i = istart; i < istop; i++) {
            index[i] = kwork;
            vcoeff[i + (joff + i) * ldvco1 + (kplus - 1) * ldvco1 * ldvco2] = ONE;
        }

        istart = istop;
    }

    for (i32 i = istart; i < pwork; i++) {
        index[i] = 0;
        pcoeff[i + i * ldpco1] = ONE;
    }

    if (indblk <= 0) {
        *nr = 0;
        dwork[0] = (f64)wrkopt;
        if (llerir) {
            ab07md('Z', *nr, mwork, pwork, a, lda, b, ldb, c, ldc, dwork, 1);
        }
        return;
    }

    i32 nrow = iwork[indblk - 1];
    i32 ioff = *nr - nrow;
    i32 kmax = indblk - 1;
    itau = 0;
    i32 ifirst = 0;
    if (indblk > 2) {
        ifirst = ioff - iwork[kmax - 1];
    }

    i32 ncol = nrow;
    i32 nreflc = nrow;
    joff = ioff;
    for (i32 k = 1; k <= kmax; k++) {
        i32 kwork = indblk - k;
        ncol = nrow;
        nrow = iwork[kwork - 1];
        joff = ioff;
        ioff = ioff - nrow;
        nreflc = (nrow < ncol) ? nrow : ncol;
        jwork = itau + nreflc;
        if (kwork >= 2) {
            ifirst = ifirst - iwork[kwork - 2];
        }

        SLC_DGEQRF(&nrow, &ncol, &a[ioff + joff * lda], &lda, &dwork[itau],
                   &dwork[jwork], &(i32){ldwork - jwork}, &local_info);

        wtemp = (i32)dwork[jwork] + jwork;
        if (wtemp > wrkopt) wrkopt = wtemp;

        SLC_DORMQR("L", "T", &nrow, &joff, &nreflc,
                   &a[ioff + joff * lda], &lda, &dwork[itau], &a[ioff], &lda,
                   &dwork[jwork], &(i32){ldwork - jwork}, &local_info);

        wtemp = (i32)dwork[jwork] + jwork;
        if (wtemp > wrkopt) wrkopt = wtemp;

        SLC_DORMQR("L", "T", &nrow, &mwork, &nreflc,
                   &a[ioff + joff * lda], &lda, &dwork[itau], &b[ioff], &ldb,
                   &dwork[jwork], &(i32){ldwork - jwork}, &local_info);

        wtemp = (i32)dwork[jwork] + jwork;
        if (wtemp > wrkopt) wrkopt = wtemp;

        i32 nr_minus_if = *nr - ifirst;
        SLC_DORMQR("R", "N", &nr_minus_if, &nrow, &nreflc,
                   &a[ioff + joff * lda], &lda, &dwork[itau],
                   &a[ifirst + ioff * lda], &lda,
                   &dwork[jwork], &(i32){ldwork - jwork}, &local_info);

        wtemp = (i32)dwork[jwork] + jwork;
        if (wtemp > wrkopt) wrkopt = wtemp;

        if (k != kmax && nrow > 1) {
            i32 nrow_m1 = nrow - 1;
            SLC_DLASET("L", &nrow_m1, &ncol, &ZERO, &ZERO,
                       &a[(ioff + 1) + joff * lda], &lda);
        }
    }

    SLC_DORMQR("R", "N", &pwork, &nrow, &nreflc,
               &a[ioff + joff * lda], &lda, &dwork[itau], c, &ldc,
               &dwork[jwork], &(i32){ldwork - jwork}, &local_info);

    wtemp = (i32)dwork[jwork] + jwork;
    if (wtemp > wrkopt) wrkopt = wtemp;

    if (nrow > 1) {
        i32 nrow_m1 = nrow - 1;
        SLC_DLASET("L", &nrow_m1, &ncol, &ZERO, &ZERO,
                   &a[(ioff + 1) + joff * lda], &lda);
    }

    tb03ay(*nr, a, lda, indblk, iwork, vcoeff, ldvco1, ldvco2,
           pcoeff, ldpco1, ldpco2, &local_info);

    if (local_info != 0) {
        *info = 1;
        return;
    }

    i32 ic = 0;
    i32 irankc = iwork[0];
    i32 ldwric = (1 > pwork) ? 1 : pwork;
    SLC_DLACPY("F", &pwork, &irankc, c, &ldc, &dwork[ic], &ldwric);

    if (irankc < pwork) {
        itau = ic + ldwric * irankc;
        jwork = itau + irankc;

        SLC_DGEQRF(&pwork, &irankc, &dwork[ic], &ldwric, &dwork[itau],
                   &dwork[jwork], &(i32){ldwork - jwork}, &local_info);

        wtemp = (i32)dwork[jwork] + jwork;
        if (wtemp > wrkopt) wrkopt = wtemp;

        for (i32 i = 0; i < irankc; i++) {
            if (dwork[ic + i * ldwric + i] == ZERO) {
                *info = 2;
                return;
            }
        }

        nrow = irankc;
        for (i32 k = 0; k < inplus; k++) {
            SLC_DTRSM("R", "U", "N", "N", &nrow, &irankc, &ONE,
                      &dwork[ic], &ldwric,
                      &pcoeff[k * ldpco1 * ldpco2], &ldpco1);
            if (k < indblk) {
                nrow = iwork[k];
            }
        }

        nrow = pwork;
        for (i32 k = 0; k < inplus; k++) {
            SLC_DORMQR("R", "T", &nrow, &pwork, &irankc,
                       &dwork[ic], &ldwric, &dwork[itau],
                       &pcoeff[k * ldpco1 * ldpco2], &ldpco1,
                       &dwork[jwork], &(i32){ldwork - jwork}, &local_info);
            wtemp = (i32)dwork[jwork] + jwork;
            if (wtemp > wrkopt) wrkopt = wtemp;
            if (k < indblk) {
                nrow = iwork[k];
            }
        }
    } else {
        SLC_DGETRF(&pwork, &pwork, &dwork[ic], &ldwric, &iwork[n], &local_info);

        if (local_info != 0) {
            *info = 2;
            return;
        }

        nrow = irankc;
        for (i32 k = 0; k < inplus; k++) {
            SLC_DTRSM("R", "U", "N", "N", &nrow, &pwork, &ONE,
                      &dwork[ic], &ldwric,
                      &pcoeff[k * ldpco1 * ldpco2], &ldpco1);
            SLC_DTRSM("R", "L", "N", "U", &nrow, &pwork, &ONE,
                      &dwork[ic], &ldwric,
                      &pcoeff[k * ldpco1 * ldpco2], &ldpco1);
            ma02gd(nrow, &pcoeff[k * ldpco1 * ldpco2], ldpco1, 1, pwork,
                   &iwork[n], -1);
            if (k < indblk) {
                nrow = iwork[k];
            }
        }
    }

    nrow = pwork;
    for (i32 k = 0; k < inplus; k++) {
        i32 nr_loc = *nr;
        SLC_DGEMM("N", "N", &nrow, &mwork, &nr_loc, &ONE,
                  &vcoeff[k * ldvco1 * ldvco2], &ldvco1,
                  b, &ldb, &ZERO,
                  &qcoeff[k * ldqco1 * ldqco2], &ldqco1);
        SLC_DGEMM("N", "N", &nrow, &mwork, &pwork, &ONE,
                  &pcoeff[k * ldpco1 * ldpco2], &ldpco1,
                  d, &ldd, &ONE,
                  &qcoeff[k * ldqco1 * ldqco2], &ldqco1);
        if (k < indblk) {
            nrow = iwork[k];
        }
    }

    if (llerir) {
        ab07md('Z', *nr, mwork, pwork, a, lda, b, ldb, c, ldc, dwork, 1);

        i32 kpcoef = 0;
        for (i32 i = 0; i < pwork; i++) {
            if (index[i] > kpcoef) kpcoef = index[i];
        }
        kpcoef = kpcoef + 1;

        tc01od('L', mwork, pwork, kpcoef, pcoeff, ldpco1, ldpco2,
               qcoeff, ldqco1, ldqco2, &local_info);
    } else {
        tb01yd(*nr, m, p, a, lda, b, ldb, c, ldc, &local_info);
    }

    dwork[0] = (f64)wrkopt;
}
