/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <string.h>

static inline i32 max_i32(i32 a, i32 b) { return a > b ? a : b; }
static inline i32 min_i32(i32 a, i32 b) { return a < b ? a : b; }

void ag08bd(const char* equil, i32 l, i32 n, i32 m, i32 p,
            f64* a, i32 lda, f64* e, i32 lde,
            f64* b, i32 ldb, f64* c, i32 ldc, const f64* d, i32 ldd,
            i32* nfz, i32* nrank, i32* niz, i32* dinfz,
            i32* nkror, i32* ninfe, i32* nkrol,
            i32* infz, i32* kronr, i32* infe, i32* kronl,
            f64 tol, i32* iwork, f64* dwork, i32 ldwork, i32* info) {

    const f64 ONE = 1.0, ZERO = 0.0;

    bool lequil = (equil[0] == 'S' || equil[0] == 's');
    bool lquery = (ldwork == -1);
    i32 ldabcd = max_i32(l + p, n + m);
    i32 labcd2 = ldabcd * (n + m);

    *info = 0;

    if (!lequil && !(equil[0] == 'N' || equil[0] == 'n')) {
        *info = -1;
    } else if (l < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (p < 0) {
        *info = -5;
    } else if (lda < max_i32(1, l)) {
        *info = -7;
    } else if (lde < max_i32(1, l)) {
        *info = -9;
    } else if (ldb < 1 || (m > 0 && ldb < l)) {
        *info = -11;
    } else if (ldc < max_i32(1, p)) {
        *info = -13;
    } else if (ldd < max_i32(1, p)) {
        *info = -15;
    } else if (tol >= ONE) {
        *info = -27;
    } else {
        i32 i1 = min_i32(l, n);
        i32 ii = min_i32(m, p);
        i32 ldw = labcd2 + max_i32(1, 5 * ldabcd);
        if (lequil) {
            ldw = max_i32(4 * (l + n), ldw);
        }

        if (lquery) {
            i32 wrkopt = ldw;

            sl_int nn = 0, n2 = 0;
            sl_int qinfo = 0;
            sl_int lwork_query = -1;
            f64 dum = 0.0;

            tg01fd("N", "N", "N", l, n, m, p, a, lda, e, lde,
                   b, ldb, c, ldc, &dum, 1, &dum, 1,
                   &nn, &n2, tol, iwork, dwork, -1, &qinfo);
            wrkopt = max_i32(wrkopt, (i32)dwork[0]);

            i32 ldabcd_int = max_i32(1, ldabcd + i1);
            f64 svlmax = ZERO;
            i32 nu = 0, mu = 0, niz_tmp = 0, dinfz_tmp = 0, nkrol_tmp = 0;

            ag08by(true, i1, m + n, p + l, svlmax, dwork, ldabcd_int,
                   e, lde, &nu, &mu, &niz_tmp, &dinfz_tmp, &nkrol_tmp,
                   infz, kronl, tol, iwork, dwork, -1, &qinfo);
            wrkopt = max_i32(wrkopt, labcd2 + (i32)dwork[0]);

            ag08by(false, i1, ii, m + n, svlmax, dwork, ldabcd_int,
                   e, lde, &nu, &mu, &niz_tmp, &dinfz_tmp, &nkrol_tmp,
                   infz, kronl, tol, iwork, dwork, -1, &qinfo);
            wrkopt = max_i32(wrkopt, labcd2 + (i32)dwork[0]);

            sl_int j_int = max_i32(1, ldabcd);
            sl_int ii_int = ii, i1_int = i1, i1_ii = i1 + ii;
            SLC_DTZRZF(&ii_int, &i1_ii, dwork, &j_int, dwork, dwork, &lwork_query, &qinfo);
            wrkopt = max_i32(wrkopt, labcd2 + ii + (i32)dwork[0]);

            SLC_DORMRZ("Right", "Transpose", &i1_int, &i1_ii, &ii_int, &i1_int,
                       dwork, &j_int, dwork, dwork, &j_int, dwork, &lwork_query, &qinfo);
            wrkopt = max_i32(wrkopt, labcd2 + ii + (i32)dwork[0]);

            dwork[0] = (f64)wrkopt;
            return;
        } else if (ldwork < ldw) {
            *info = -30;
        }
    }

    if (*info != 0) {
        return;
    }

    *niz = 0;
    *nkrol = 0;
    *nkror = 0;

    if (max_i32(max_i32(l, n), max_i32(m, p)) == 0) {
        *nfz = 0;
        *dinfz = 0;
        *ninfe = 0;
        *nrank = 0;
        iwork[0] = 0;
        dwork[0] = ONE;
        return;
    }

    i32 wrkopt = 1;
    i32 kabcd = 0;
    i32 jwork = kabcd + labcd2;

    if (lequil) {
        tg01ad("A", l, n, m, p, ZERO, a, lda, e, lde, b, ldb,
               c, ldc, dwork, &dwork[l], &dwork[l + n], info);
        wrkopt = 4 * (l + n);
    }

    i32 nn = 0, n2 = 0;
    i32 tg01_info = 0;
    tg01fd("N", "N", "N", l, n, m, p, a, lda, e, lde, b, ldb,
           c, ldc, dwork, 1, dwork, 1, &nn, &n2, tol,
           iwork, dwork, ldwork, &tg01_info);
    wrkopt = max_i32(wrkopt, (i32)dwork[0]);

    n2 = n - nn;
    i32 mm = m + n2;
    i32 pp = p + (l - nn);

    sl_int l_int = l, p_int = p, m_int = m;
    sl_int nn_int = nn, n2_int = n2;
    sl_int lda_int = lda, ldc_int = ldc, ldb_int = ldb, ldd_int = ldd;
    sl_int ldabcd_int = ldabcd;

    SLC_DLACPY("Full", &l_int, &m_int, b, &ldb_int, &dwork[kabcd], &ldabcd_int);
    SLC_DLACPY("Full", &p_int, &m_int, d, &ldd_int, &dwork[kabcd + l], &ldabcd_int);

    sl_int a_offset_col = nn + 1 - 1;
    SLC_DLACPY("Full", &l_int, &n2_int, &a[a_offset_col * lda], &lda_int,
               &dwork[kabcd + ldabcd * m], &ldabcd_int);
    SLC_DLACPY("Full", &p_int, &n2_int, &c[a_offset_col * ldc], &ldc_int,
               &dwork[kabcd + ldabcd * m + l], &ldabcd_int);
    SLC_DLACPY("Full", &l_int, &nn_int, a, &lda_int,
               &dwork[kabcd + ldabcd * mm], &ldabcd_int);
    SLC_DLACPY("Full", &p_int, &nn_int, c, &ldc_int,
               &dwork[kabcd + ldabcd * mm + l], &ldabcd_int);

    f64 toler = tol;
    if (toler <= ZERO) {
        toler = (f64)((l + p) * (m + n)) * SLC_DLAMCH("Precision");
    }

    i32 nn_pp = nn + pp;
    i32 nn_mm = nn + mm;
    f64 svlmax = SLC_DLANGE("Frobenius", &nn_pp, &nn_mm, &dwork[kabcd], &ldabcd_int, &dwork[jwork]);

    i32 nu = 0, mu = 0, ag_nkrol = 0;
    i32 ldwork_ag = ldwork - jwork;
    ag08by(true, nn, mm, pp, svlmax, &dwork[kabcd], ldabcd,
           e, lde, &nu, &mu, niz, dinfz, &ag_nkrol,
           infz, kronl, toler, iwork, &dwork[jwork], ldwork_ag, info);

    wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);
    *nkrol = ag_nkrol;

    i32 nsinfe = mu;
    *nrank = nn + mu;

    tb01xd("D", nu, mm, mm, max_i32(0, nu - 1), max_i32(0, nu - 1),
           &dwork[kabcd + ldabcd * mm], ldabcd,
           &dwork[kabcd], ldabcd,
           &dwork[kabcd + ldabcd * mm + nu], ldabcd,
           &dwork[kabcd + nu], ldabcd, info);

    ma02bd('R', nu + mm, mm, &dwork[kabcd], ldabcd);
    ma02bd('L', mm, nu + mm, &dwork[kabcd + nu], ldabcd);
    ma02cd(nu, 0, max_i32(0, nu - 1), e, lde);

    if (mu != mm) {
        nn = nu;
        pp = mm;
        mm = mu;
        kabcd = kabcd + (pp - mm) * ldabcd;

        i32 i0_tmp = 0, i1_tmp = 0;
        i32 ag_nkror = 0;
        ag08by(false, nn, mm, pp, svlmax, &dwork[kabcd], ldabcd,
               e, lde, &nu, &mu, &i0_tmp, &i1_tmp, &ag_nkror,
               iwork, kronr, toler, iwork, &dwork[jwork], ldwork_ag, info);

        wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);
        *nkror = ag_nkror;
    }

    if (nu != 0) {
        i32 numu = nu + mu;
        i32 ipd = kabcd + nu;
        i32 itau = jwork;
        jwork = itau + mu;
        i32 ldwork_rz = ldwork - jwork;

        sl_int nu_int = nu, mu_int = mu, numu_int = numu;
        sl_int ldwork_rz_int = ldwork_rz;
        sl_int rzinfo = 0;

        SLC_DTZRZF(&mu_int, &numu_int, &dwork[ipd], &ldabcd_int, &dwork[itau],
                   &dwork[jwork], &ldwork_rz_int, &rzinfo);
        wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);

        SLC_DORMRZ("Right", "Transpose", &nu_int, &numu_int, &mu_int, &nu_int,
                   &dwork[ipd], &ldabcd_int, &dwork[itau], &dwork[kabcd],
                   &ldabcd_int, &dwork[jwork], &ldwork_rz_int, &rzinfo);
        wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);

        SLC_DLACPY("Full", &nu_int, &nu_int, &dwork[kabcd + ldabcd * mu], &ldabcd_int, a, &lda_int);

        SLC_DLASET("Full", &nu_int, &mu_int, &ZERO, &ZERO, &dwork[kabcd], &ldabcd_int);
        SLC_DLACPY("Full", &nu_int, &nu_int, e, &lde, &dwork[kabcd + ldabcd * mu], &ldabcd_int);

        SLC_DORMRZ("Right", "Transpose", &nu_int, &numu_int, &mu_int, &nu_int,
                   &dwork[ipd], &ldabcd_int, &dwork[itau], &dwork[kabcd],
                   &ldabcd_int, &dwork[jwork], &ldwork_rz_int, &rzinfo);

        sl_int lde_int = lde;
        SLC_DLACPY("Full", &nu_int, &nu_int, &dwork[kabcd + ldabcd * mu], &ldabcd_int, e, &lde_int);
    }

    *nfz = nu;

    for (i32 i = 0; i < *nkror; i++) {
        iwork[i] = kronr[i];
    }

    i32 j = 0;
    for (i32 i = 0; i < *nkror; i++) {
        for (i32 ii = j; ii < j + iwork[i]; ii++) {
            if (ii < n + m + 1) {
                kronr[ii] = i;
            }
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
            if (ii < l + p + 1) {
                kronl[ii] = i;
            }
        }
        j = j + iwork[i];
    }
    *nkrol = j;

    *ninfe = 0;
    for (i32 i = 0; i < *dinfz; i++) {
        *ninfe += infz[i];
    }
    *ninfe = nsinfe - *ninfe;

    for (i32 i = 0; i < *ninfe; i++) {
        infe[i] = 1;
    }

    i32 ninfe_accum = *ninfe;
    for (i32 i = 0; i < *dinfz; i++) {
        for (i32 ii = ninfe_accum; ii < ninfe_accum + infz[i]; ii++) {
            if (ii < 1 + min_i32(l + p, n + m)) {
                infe[ii] = i + 2;
            }
        }
        ninfe_accum += infz[i];
    }
    *ninfe = ninfe_accum;

    iwork[0] = nsinfe;
    dwork[0] = (f64)wrkopt;
}
