/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <string.h>

static inline i32 max_i32(i32 a, i32 b) {
    return a > b ? a : b;
}

static inline i32 min_i32(i32 a, i32 b) {
    return a < b ? a : b;
}

void ab08nw(const char* equil, i32 n, i32 m, i32 p, f64* a, i32 lda,
            f64* b, i32 ldb, f64* c, i32 ldc, f64* d, i32 ldd,
            i32* nfz, i32* nrank, i32* niz, i32* dinfz, i32* nkror,
            i32* ninfe, i32* nkrol, i32* infz, i32* kronr, i32* infe,
            i32* kronl, f64* e, i32 lde, f64 tol, i32* iwork, f64* dwork,
            i32 ldwork, i32* info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    *info = 0;

    i32 ldabcd = n + max_i32(p, m);
    i32 labcd2 = ldabcd * ldabcd;
    bool lequil = (equil[0] == 'S' || equil[0] == 's');

    if (!lequil && equil[0] != 'N' && equil[0] != 'n') {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (p < 0) {
        *info = -4;
    } else if (lda < max_i32(1, n)) {
        *info = -6;
    } else if (ldb < 1 || (m > 0 && ldb < n)) {
        *info = -8;
    } else if (ldc < max_i32(1, p)) {
        *info = -10;
    } else if (ldd < max_i32(1, p)) {
        *info = -12;
    } else if (lde < max_i32(1, n)) {
        *info = -25;
    } else if (tol >= ONE) {
        *info = -26;
    } else {
        i32 mpn = min_i32(p, n);
        i32 mpm = min_i32(p, m);
        bool qret = (max_i32(n, max_i32(m, p)) == 0);
        bool lquery = (ldwork == -1);

        i32 jwork;
        if (qret) {
            jwork = 1;
        } else {
            jwork = max_i32(mpm + m + max_i32(2*m, n) - 1,
                           mpn + max_i32(ldabcd, 3*p - 1)) + labcd2;
        }

        if (lquery) {
            i32 wrkopt;
            if (qret) {
                wrkopt = 1;
            } else {
                f64 svlmax = ZERO;
                i32 niz_tmp = 0;
                i32 nu, mu, dinfz_tmp, nkrol_tmp, info_tmp;

                ab08ny(true, n, m, p, svlmax, dwork, ldabcd, &niz_tmp,
                       &nu, &mu, &dinfz_tmp, &nkrol_tmp, infz, kronl, tol,
                       iwork, dwork, -1, &info_tmp);
                wrkopt = max_i32(jwork, labcd2 + (i32)dwork[0]);

                i32 i1;
                ab08ny(false, n, m, m, svlmax, dwork, ldabcd, &niz_tmp,
                       &nu, &mu, &i1, nkror, iwork, kronr, tol, iwork,
                       dwork, -1, &info_tmp);
                wrkopt = max_i32(wrkopt, labcd2 + (i32)dwork[0]);

                sl_int mpm_int = mpm;
                sl_int n_plus_mpm = n + mpm;
                sl_int ldabcd_int = ldabcd;
                sl_int info_lapack;
                SLC_DTZRZF(&mpm_int, &n_plus_mpm, dwork, &ldabcd_int, dwork,
                           dwork, (sl_int*)&info_tmp, &info_lapack);
                wrkopt = max_i32(wrkopt, labcd2 + mpm + (i32)dwork[0]);

                sl_int n_int = n;
                SLC_DORMRZ("Right", "Transpose", &n_int, &n_plus_mpm, &mpm_int,
                           &n_int, dwork, &ldabcd_int, dwork, dwork, &ldabcd_int,
                           dwork, (sl_int*)&info_tmp, &info_lapack);
                wrkopt = max_i32(wrkopt, labcd2 + mpm + (i32)dwork[0]);
            }
            dwork[0] = (f64)wrkopt;
            return;
        } else if (ldwork < jwork) {
            *info = -29;
        }
    }

    if (*info != 0) {
        return;
    }

    *niz = 0;
    *nkrol = 0;
    *nkror = 0;
    *ninfe = 0;

    bool qret = (max_i32(n, max_i32(m, p)) == 0);
    if (qret) {
        *dinfz = 0;
        *nfz = 0;
        *nrank = 0;
        dwork[0] = ONE;
        return;
    }

    i32 wrkopt = 1;
    i32 kabcd = 0;
    i32 jwork = kabcd + labcd2;

    if (lequil) {
        f64 maxred = ZERO;
        i32 info_tmp;
        tb01id("A", n, m, p, &maxred, a, lda, b, ldb, c, ldc, dwork, &info_tmp);
        wrkopt = n;
    }

    i32 nn = n;
    i32 mm = m;
    i32 pp = p;

    sl_int n_int = n, m_int = m, p_int = p;
    sl_int lda_int = lda, ldb_int = ldb, ldc_int = ldc, ldd_int = ldd;
    sl_int ldabcd_int = ldabcd;

    SLC_DLACPY("Full", &n_int, &m_int, b, &ldb_int, &dwork[kabcd], &ldabcd_int);
    SLC_DLACPY("Full", &p_int, &m_int, d, &ldd_int, &dwork[kabcd + n], &ldabcd_int);
    SLC_DLACPY("Full", &n_int, &n_int, a, &lda_int, &dwork[kabcd + ldabcd * m], &ldabcd_int);
    SLC_DLACPY("Full", &p_int, &n_int, c, &ldc_int, &dwork[kabcd + ldabcd * m + n], &ldabcd_int);

    f64 toler = tol;
    if (toler <= ZERO) {
        toler = (f64)labcd2 * SLC_DLAMCH("Precision");
    }

    sl_int nn_plus_pp = nn + pp;
    sl_int nn_plus_mm = nn + mm;
    f64 svlmax = SLC_DLANGE("Frobenius", &nn_plus_pp, &nn_plus_mm, &dwork[kabcd],
                            &ldabcd_int, &dwork[jwork]);

    i32 nu, mu, info_tmp;
    ab08ny(true, nn, mm, pp, svlmax, &dwork[kabcd], ldabcd, niz,
           &nu, &mu, dinfz, nkrol, infz, kronl, toler, iwork,
           &dwork[jwork], ldwork - jwork, &info_tmp);

    wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);

    i32 nsinfe = mu;
    *nrank = nn + mu;

    i32 info_tb;
    tb01xd("D", nu, mm, mm, max_i32(0, nu - 1), max_i32(0, nu - 1),
           &dwork[kabcd + ldabcd * mm], ldabcd,
           &dwork[kabcd], ldabcd,
           &dwork[kabcd + ldabcd * mm + nu], ldabcd,
           &dwork[kabcd + nu], ldabcd, &info_tb);

    ma02bd('R', nu + mm, mm, &dwork[kabcd], ldabcd);
    ma02bd('L', mm, nu + mm, &dwork[kabcd + nu], ldabcd);

    if (mu != mm) {
        nn = nu;
        pp = mm;
        mm = mu;
        kabcd = kabcd + (pp - mm) * ldabcd;

        i32 i0, i1;
        ab08ny(false, nn, mm, pp, svlmax, &dwork[kabcd], ldabcd, &i0,
               &nu, &mu, &i1, nkror, iwork, kronr, toler, iwork,
               &dwork[jwork], ldwork - jwork, &info_tmp);

        wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);
    }

    if (min_i32(nu, mu) != 0) {
        i32 nu1 = nu + kabcd;
        i32 i1 = nu + mu;
        i32 itau = jwork;
        jwork = itau + mu;

        sl_int mu_int = mu;
        sl_int i1_int = i1;
        sl_int ldwork_minus_jwork = ldwork - jwork;
        sl_int info_lapack;

        SLC_DTZRZF(&mu_int, &i1_int, &dwork[nu1], &ldabcd_int, &dwork[itau],
                   &dwork[jwork], &ldwork_minus_jwork, &info_lapack);
        wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);

        sl_int nu_int = nu;
        SLC_DORMRZ("Right", "Transpose", &nu_int, &i1_int, &mu_int, &nu_int,
                   &dwork[nu1], &ldabcd_int, &dwork[itau], &dwork[kabcd],
                   &ldabcd_int, &dwork[jwork], &ldwork_minus_jwork, &info_lapack);
        wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);

        SLC_DLACPY("Full", &nu_int, &nu_int, &dwork[kabcd + mu * ldabcd],
                   &ldabcd_int, a, &lda_int);

        SLC_DLASET("Full", &nu_int, &mu_int, &ZERO, &ZERO, &dwork[kabcd], &ldabcd_int);
        SLC_DLASET("Full", &nu_int, &nu_int, &ZERO, &ONE, &dwork[kabcd + mu * ldabcd], &ldabcd_int);

        SLC_DORMRZ("Right", "Transpose", &nu_int, &i1_int, &mu_int, &nu_int,
                   &dwork[nu1], &ldabcd_int, &dwork[itau], &dwork[kabcd],
                   &ldabcd_int, &dwork[jwork], &ldwork_minus_jwork, &info_lapack);
        wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);

        sl_int lde_int = lde;
        SLC_DLACPY("Full", &nu_int, &nu_int, &dwork[kabcd + mu * ldabcd],
                   &ldabcd_int, e, &lde_int);
    } else {
        sl_int nu_int = nu;
        sl_int lde_int = lde;

        SLC_DLACPY("Full", &nu_int, &nu_int, &dwork[kabcd + mu * ldabcd],
                   &ldabcd_int, a, &lda_int);
        SLC_DLASET("Full", &nu_int, &nu_int, &ZERO, &ONE, e, &lde_int);
    }

    *nfz = nu;

    for (i32 i = 0; i < *nkror; i++) {
        iwork[i] = kronr[i];
    }

    i32 j = 0;
    for (i32 i = 0; i < *nkror; i++) {
        for (i32 ii = j; ii < j + iwork[i]; ii++) {
            kronr[ii] = i;
        }
        j = j + iwork[i];
    }
    *nkror = j;

    for (i32 i = 0; i < *nkrol; i++) {
        iwork[i] = kronl[i];
    }

    j = 0;
    for (i32 i = 0; i < *nkrol; i++) {
        for (i32 ii = j; ii < j + iwork[i]; ii++) {
            kronl[ii] = i;
        }
        j = j + iwork[i];
    }
    *nkrol = j;

    for (i32 i = 0; i < *dinfz; i++) {
        *ninfe = *ninfe + infz[i];
    }

    *ninfe = nsinfe - *ninfe;

    for (i32 i = 0; i < *ninfe; i++) {
        infe[i] = 1;
    }

    i32 ninfe_val = *ninfe;
    for (i32 i = 0; i < *dinfz; i++) {
        for (i32 ii = ninfe_val; ii < ninfe_val + infz[i]; ii++) {
            if (ii < n + 1) {
                infe[ii] = i + 2;
            }
        }
        ninfe_val = ninfe_val + infz[i];
    }
    *ninfe = ninfe_val;

    dwork[0] = (f64)wrkopt;
}
