/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

static inline i32 max_i32(i32 a, i32 b) {
    return a > b ? a : b;
}

static inline i32 min_i32(i32 a, i32 b) {
    return a < b ? a : b;
}

void ab08ny(bool first, i32 n, i32 m, i32 p, f64 svlmax,
            f64* abcd, i32 ldabcd, i32* ninfz, i32* nr, i32* pr,
            i32* dinfz, i32* nkronl, i32* infz, i32* kronl,
            f64 tol, i32* iwork, f64* dwork, i32 ldwork, i32* info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    i32 pn = p + n;
    i32 mn = m + n;
    i32 mpn = min_i32(p, n);
    i32 mpm = min_i32(p, m);

    *info = 0;
    bool lquery = (ldwork == -1);

    if (n < 0) {
        *info = -2;
    } else if (m < 0 || (!first && m > p)) {
        *info = -3;
    } else if (p < 0) {
        *info = -4;
    } else if (svlmax < ZERO) {
        *info = -5;
    } else if (ldabcd < max_i32(1, pn)) {
        *info = -7;
    } else if (*ninfz < 0 || (first && *ninfz > 0)) {
        *info = -8;
    } else if (tol >= ONE) {
        *info = -15;
    } else {
        i32 jwork;
        if (min_i32(p, max_i32(n, m)) == 0) {
            jwork = 1;
        } else {
            jwork = max_i32(mpm + m + max_i32(2*m, n) - 1,
                           mpn + max_i32(n + max_i32(p, m), 3*p - 1));
        }

        if (lquery) {
            i32 wrkopt = jwork;

            if (m > 0) {
                sl_int p_int = p, mpm_int = mpm;
                sl_int n_int = n, ldabcd_int = ldabcd;
                sl_int lwork_query = -1;
                sl_int info_query = 0;

                mb04id(p, mpm, m - 1, n, abcd, ldabcd, abcd, ldabcd,
                       dwork, dwork, -1, &info_query);
                wrkopt = max_i32(jwork, mpm + (i32)dwork[0]);

                SLC_DORMQR("Left", "Transpose", &p_int, &n_int, &mpm_int,
                           abcd, &ldabcd_int, dwork, abcd, &ldabcd_int,
                           dwork, &lwork_query, &info_query);
                wrkopt = max_i32(wrkopt, mpm + (i32)dwork[0]);
            }

            sl_int pn_int = pn, n_int = n, mpn_int = mpn;
            sl_int mn_int = mn, ldabcd_int = ldabcd;
            sl_int lwork_query = -1;
            sl_int info_query = 0;

            SLC_DORMRQ("Right", "Transpose", &pn_int, &n_int, &mpn_int,
                       abcd, &ldabcd_int, dwork, abcd, &ldabcd_int,
                       dwork, &lwork_query, &info_query);
            wrkopt = max_i32(wrkopt, mpn + (i32)dwork[0]);

            SLC_DORMRQ("Left", "NoTranspose", &n_int, &mn_int, &mpn_int,
                       abcd, &ldabcd_int, dwork, abcd, &ldabcd_int,
                       dwork, &lwork_query, &info_query);
            wrkopt = max_i32(wrkopt, mpn + (i32)dwork[0]);

            dwork[0] = (f64)wrkopt;
            return;
        } else if (ldwork < jwork) {
            *info = -18;
        }
    }

    if (*info != 0) {
        return;
    }

    *pr = p;
    *nr = n;
    *dinfz = 0;
    *nkronl = 0;

    if (p == 0) {
        dwork[0] = ONE;
        return;
    }

    if (max_i32(n, m) == 0) {
        *pr = 0;
        *nkronl = 1;
        kronl[0] = p;
        dwork[0] = ONE;
        return;
    }

    i32 wrkopt = 1;
    f64 rcond = tol;
    if (rcond <= ZERO) {
        rcond = (f64)(pn * mn) * SLC_DLAMCH("Epsilon");
    }

    i32 sigma;
    if (first) {
        sigma = 0;
    } else {
        sigma = m;
    }

    i32 ro = p - sigma;
    i32 mp1 = m + 1;
    i32 mui = 0;
    i32 nblcks = 0;
    i32 itau = 0;

    f64 sval[3];
    i32 i, i1, icol, irc, irow, jwork, k, mnr, mntau, muim1, rank, ro1, taui;
    sl_int info_lapack;

    while (1) {
        if (*pr == 0) {
            goto L20;
        }

        ro1 = ro;
        mnr = m + *nr;

        if (m > 0) {
            irow = *nr;

            if (sigma > 0) {
                jwork = itau + sigma;

                mb04id(ro + sigma, sigma, sigma - 1, mnr - sigma,
                       &abcd[irow], ldabcd, &abcd[(sigma) * ldabcd + irow],
                       ldabcd, &dwork[itau], &dwork[jwork],
                       ldwork - jwork, info);

                sl_int ro_sigma_minus_1 = ro + sigma - 1;
                sl_int sigma_int = sigma;
                sl_int ldabcd_int = ldabcd;
                sl_int irow_plus_1 = irow + 1;
                SLC_DLASET("Lower", &ro_sigma_minus_1, &sigma_int, &ZERO, &ZERO,
                           &abcd[irow_plus_1], &ldabcd_int);

                wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);
            }

            if (first) {
                jwork = itau + min_i32(ro1, m - sigma);

                irow = min_i32(*nr + sigma, pn - 1);
                icol = min_i32(sigma, m - 1);

                mb03oy(ro1, m - sigma, &abcd[icol * ldabcd + irow], ldabcd,
                       rcond, svlmax, &rank, sval, iwork, &dwork[itau],
                       &dwork[jwork], info);
                wrkopt = max_i32(wrkopt, jwork + 3*(m - sigma) - 1);

                sl_int nr_plus_sigma = *nr + sigma;
                sl_int m_minus_sigma = m - sigma;
                sl_int forwrd = 1;
                sl_int ldabcd_int = ldabcd;
                SLC_DLAPMT(&forwrd, &nr_plus_sigma, &m_minus_sigma,
                           &abcd[icol * ldabcd], &ldabcd_int, iwork);

                if (rank > 0) {
                    sl_int ro1_int = ro1;
                    sl_int nr_int = *nr;
                    sl_int rank_int = rank;
                    sl_int ldabcd_int = ldabcd;
                    sl_int ldwork_minus_jwork = ldwork - jwork;

                    SLC_DORMQR("Left", "Transpose", &ro1_int, &nr_int, &rank_int,
                               &abcd[icol * ldabcd + irow], &ldabcd_int,
                               &dwork[itau], &abcd[mp1 * ldabcd - ldabcd + irow],
                               &ldabcd_int, &dwork[jwork], &ldwork_minus_jwork,
                               &info_lapack);
                    wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);

                    sl_int ro1_minus_1 = ro1 - 1;
                    sl_int min_ro1_rank = min_i32(ro1 - 1, rank);
                    i32 irow_idx = min_i32(irow + 1, pn - 1);
                    SLC_DLASET("Lower", &ro1_minus_1, &min_ro1_rank, &ZERO, &ZERO,
                               &abcd[icol * ldabcd + irow_idx], &ldabcd_int);

                    ro1 = ro1 - rank;
                }
            }

            if (ro1 == 0) {
                goto L30;
            }
        }

        sigma = *pr - ro1;
        nblcks++;
        taui = ro1;

        if (*nr <= 0) {
            *pr = sigma;
            rank = 0;
        } else {
            irc = *nr + sigma;
            i1 = irc;
            mntau = min_i32(taui, *nr);
            jwork = itau + mntau;

            mb03py(taui, *nr, &abcd[mp1 * ldabcd - ldabcd + i1], ldabcd,
                   rcond, svlmax, &rank, sval, iwork, &dwork[itau],
                   &dwork[jwork], info);
            wrkopt = max_i32(wrkopt, jwork + 3*taui - 1);

            if (rank > 0) {
                irow = i1 + taui - rank;

                sl_int irc_int = irc;
                sl_int nr_int = *nr;
                sl_int rank_int = rank;
                sl_int ldabcd_int = ldabcd;
                sl_int ldwork_minus_jwork = ldwork - jwork;

                SLC_DORMRQ("Right", "Transpose", &irc_int, &nr_int, &rank_int,
                           &abcd[mp1 * ldabcd - ldabcd + irow], &ldabcd_int,
                           &dwork[mntau - rank], &abcd[mp1 * ldabcd - ldabcd],
                           &ldabcd_int, &dwork[jwork], &ldwork_minus_jwork,
                           &info_lapack);
                wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);

                sl_int mnr_int = mnr;
                SLC_DORMRQ("Left", "NoTranspose", &nr_int, &mnr_int, &rank_int,
                           &abcd[mp1 * ldabcd - ldabcd + irow], &ldabcd_int,
                           &dwork[mntau - rank], abcd, &ldabcd_int,
                           &dwork[jwork], &ldwork_minus_jwork, &info_lapack);
                wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);

                sl_int nr_minus_rank = *nr - rank;
                SLC_DLASET("Full", &rank_int, &nr_minus_rank, &ZERO, &ZERO,
                           &abcd[mp1 * ldabcd - ldabcd + irow], &ldabcd_int);

                if (rank > 1) {
                    sl_int rank_minus_1 = rank - 1;
                    i32 col_offset = mp1 - 1 + *nr - rank;
                    SLC_DLASET("Lower", &rank_minus_1, &rank_minus_1, &ZERO, &ZERO,
                               &abcd[col_offset * ldabcd + irow + 1], &ldabcd_int);
                }
            }
        }

L20:
        mui = rank;
        *nr = *nr - mui;
        *pr = sigma + mui;

        kronl[nblcks - 1] = taui - mui;

        if (first && nblcks > 1) {
            infz[nblcks - 2] = muim1 - taui;
        }
        muim1 = mui;
        ro = mui;

        if (mui <= 0) {
            break;
        }
    }

L30:
    if (first) {
        if (mui == 0) {
            *dinfz = max_i32(0, nblcks - 1);
        } else {
            *dinfz = nblcks;
            infz[nblcks - 1] = mui;
        }
        k = *dinfz;

        for (i = k - 1; i >= 0; i--) {
            if (infz[i] != 0) {
                break;
            }
            *dinfz = *dinfz - 1;
        }

        for (i = 0; i < *dinfz; i++) {
            *ninfz = *ninfz + infz[i] * (i + 1);
        }
    }

    *nkronl = nblcks;

    for (i = nblcks - 1; i >= 0; i--) {
        if (kronl[i] != 0) {
            break;
        }
        *nkronl = *nkronl - 1;
    }

    dwork[0] = (f64)wrkopt;
}
