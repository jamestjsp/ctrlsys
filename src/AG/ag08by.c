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

void ag08by(bool first, i32 n, i32 m, i32 p, f64 svlmax,
            f64* abcd, i32 ldabcd, f64* e, i32 lde,
            i32* nr, i32* pr, i32* ninfz, i32* dinfz, i32* nkronl,
            i32* infz, i32* kronl, f64 tol, i32* iwork,
            f64* dwork, i32 ldwork, i32* info) {

    const i32 IMAX = 1, IMIN = 2;
    const f64 ONE = 1.0, ZERO = 0.0;

    bool lquery;
    i32 i, icol, ilast, irc, irow, ismax, ismin, itau;
    i32 j, jlast, jwork1, jwork2, k, mn, mn1, mnr;
    i32 mntau, mp1, mpm, mui, muim1, n1, nblcks, pn;
    i32 rank, ro, ro1, sigma, taui, wrkopt;
    f64 c, c1, c2, rcond, s, s1, s2, smax, smaxpr;
    f64 smin, sminpr, t, tolz, tt;
    f64 sval[3];

    lquery = (ldwork == -1);
    *info = 0;
    pn = p + n;
    mn = m + n;
    mpm = min_i32(p, m);

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
    } else if (lde < max_i32(1, n)) {
        *info = -9;
    } else if (tol > ONE) {
        *info = -17;
    } else {
        wrkopt = max_i32(1, 5 * p);
        if (p > 0) {
            if (m > 0) {
                wrkopt = max_i32(wrkopt, mn - 1);
                if (first) {
                    wrkopt = max_i32(wrkopt, mpm + max_i32(3 * m - 1, n));
                    if (lquery) {
                        sl_int p_int = p, n_int = n, mpm_int = mpm;
                        sl_int ldabcd_int = ldabcd, lwork_query = -1;
                        sl_int qinfo = 0;
                        SLC_DORMQR("Left", "Transpose", &p_int, &n_int, &mpm_int,
                                   abcd, &ldabcd_int, dwork, abcd, &ldabcd_int,
                                   dwork, &lwork_query, &qinfo);
                        wrkopt = max_i32(wrkopt, mpm + (i32)dwork[0]);
                    }
                }
            }
        }
        if (ldwork < wrkopt && !lquery) {
            *info = -20;
        }
    }

    if (*info != 0) {
        return;
    } else if (lquery) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    *pr = p;
    *nr = n;
    *dinfz = 0;
    *ninfz = 0;
    *nkronl = 0;

    if (p == 0) {
        dwork[0] = ONE;
        return;
    }
    if (n == 0 && m == 0) {
        *pr = 0;
        *nkronl = 1;
        kronl[0] = p;
        dwork[0] = ONE;
        return;
    }

    tolz = sqrt(SLC_DLAMCH("Epsilon"));
    rcond = tol;
    if (rcond <= ZERO) {
        rcond = (f64)(pn * mn) * SLC_DLAMCH("Epsilon");
    }

    if (first) {
        sigma = 0;
    } else {
        sigma = m;
    }
    ro = p - sigma;
    mp1 = m + 1;
    mui = 0;

    itau = 0;
    jwork1 = itau + mpm;
    ismin = 2 * p;
    ismax = ismin + p;
    jwork2 = ismax + p;
    nblcks = 0;
    wrkopt = 1;
    muim1 = 0;

    while (*pr != 0) {
        ro1 = ro;
        mnr = m + *nr;

        if (m > 0) {
            irow = *nr;
            for (icol = 0; icol < sigma; icol++) {
                irow++;
                sl_int ro_plus_1 = ro + 1;
                sl_int one = 1;
                sl_int mnr_minus_icol = mnr - icol;
                sl_int ldabcd_int = ldabcd;

                SLC_DLARFG(&ro_plus_1, &abcd[(icol) * ldabcd + irow - 1],
                           &abcd[(icol) * ldabcd + irow], &one, &t);

                SLC_DLATZM("L", &ro_plus_1, &mnr_minus_icol,
                           &abcd[(icol) * ldabcd + irow], &one, &t,
                           &abcd[(icol + 1) * ldabcd + irow - 1],
                           &abcd[(icol + 1) * ldabcd + irow],
                           &ldabcd_int, dwork);

                for (i32 ii = 0; ii < *pr - icol - 1; ii++) {
                    abcd[(icol) * ldabcd + irow + ii] = ZERO;
                }
            }
            wrkopt = max_i32(wrkopt, mn - 1);

            if (first) {
                irow = min_i32(*nr + sigma, pn - 1);
                icol = min_i32(sigma, m - 1);

                mb03oy(ro1, m - sigma,
                       &abcd[(icol) * ldabcd + irow],
                       ldabcd, rcond, svlmax, &rank, sval, iwork,
                       &dwork[itau], &dwork[jwork1], info);
                wrkopt = max_i32(wrkopt, jwork1 + 3 * m - 1);

                sl_int forwrd = 1;
                sl_int nr_plus_sigma = *nr + sigma;
                sl_int m_minus_sigma = m - sigma;
                sl_int ldabcd_int = ldabcd;
                SLC_DLAPMT(&forwrd, &nr_plus_sigma, &m_minus_sigma,
                           &abcd[(icol) * ldabcd], &ldabcd_int, iwork);

                if (rank > 0) {
                    sl_int ro1_int = ro1, nr_int = *nr, rank_int = rank;
                    sl_int ldabcd_int = ldabcd;
                    sl_int lwork_avail = ldwork - jwork1;
                    sl_int qinfo = 0;

                    SLC_DORMQR("Left", "Transpose", &ro1_int, &nr_int, &rank_int,
                               &abcd[(icol) * ldabcd + irow], &ldabcd_int,
                               &dwork[itau],
                               &abcd[(mp1 - 1) * ldabcd + irow], &ldabcd_int,
                               &dwork[jwork1], &lwork_avail, &qinfo);
                    wrkopt = max_i32(wrkopt, jwork1 + (i32)dwork[jwork1]);

                    sl_int ro1_minus_1 = ro1 - 1;
                    sl_int lower_rows = min_i32(ro1 - 1, rank);
                    sl_int irow_next = min_i32(irow + 1, pn - 1);
                    SLC_DLASET("Lower", &ro1_minus_1, &lower_rows, &ZERO, &ZERO,
                               &abcd[(icol) * ldabcd + irow_next], &ldabcd_int);

                    ro1 = ro1 - rank;
                }
            }

            if (ro1 == 0) {
                break;
            }
        }

        sigma = *pr - ro1;
        nblcks++;
        taui = ro1;

        if (*nr == 0) {
            *pr = sigma;
            rank = 0;
        } else {
            irc = *nr + sigma;
            n1 = *nr;

            if (taui > 1) {
                for (i = 0; i < taui; i++) {
                    sl_int nr_int = *nr;
                    sl_int ldabcd_int = ldabcd;
                    dwork[i] = SLC_DNRM2(&nr_int, &abcd[(mp1 - 1) * ldabcd + irc + i], &ldabcd_int);
                    dwork[p + i] = dwork[i];
                }
            }

            rank = 0;
            mntau = min_i32(taui, *nr);

            ilast = *nr + *pr - 1;
            jlast = m + *nr - 1;
            irow = ilast;
            icol = jlast;
            i = taui - 1;

            while (rank < mntau) {
                mn1 = m + n1;

                if (i > 0) {
                    sl_int i_int = i + 1;
                    sl_int one = 1;
                    j = SLC_IDAMAX(&i_int, dwork, &one) - 1;
                    if (j != i) {
                        dwork[j] = dwork[i];
                        dwork[p + j] = dwork[p + i];
                        sl_int n1_int = n1;
                        sl_int ldabcd_int = ldabcd;
                        SLC_DSWAP(&n1_int, &abcd[(mp1 - 1) * ldabcd + irow], &ldabcd_int,
                                  &abcd[(mp1 - 1) * ldabcd + irc + j], &ldabcd_int);
                    }
                }

                for (k = 0; k < n1 - 1; k++) {
                    j = m + k;

                    t = abcd[(j + 1) * ldabcd + irow];
                    SLC_DLARTG(&t, &abcd[(j) * ldabcd + irow], &c, &s, &abcd[(j + 1) * ldabcd + irow]);
                    abcd[(j) * ldabcd + irow] = ZERO;

                    sl_int irow_int = irow;
                    sl_int one = 1;
                    sl_int lde_int = lde;
                    sl_int ldabcd_int = ldabcd;
                    SLC_DROT(&irow_int, &abcd[(j + 1) * ldabcd], &one, &abcd[(j) * ldabcd], &one, &c, &s);
                    sl_int k_plus_2 = k + 2;
                    SLC_DROT(&k_plus_2, &e[(k + 1) * lde], &one, &e[(k) * lde], &one, &c, &s);

                    t = e[(k) * lde + k];
                    SLC_DLARTG(&t, &e[(k) * lde + k + 1], &c, &s, &e[(k) * lde + k]);
                    e[(k) * lde + k + 1] = ZERO;

                    sl_int n1_minus_k = n1 - k - 1;
                    sl_int mn1_int = mn1;
                    SLC_DROT(&n1_minus_k, &e[(k + 1) * lde + k], &lde_int, &e[(k + 1) * lde + k + 1], &lde_int, &c, &s);
                    SLC_DROT(&mn1_int, &abcd[k], &ldabcd_int, &abcd[k + 1], &ldabcd_int, &c, &s);
                }

                if (rank == 0) {
                    smax = fabs(abcd[(jlast) * ldabcd + ilast]);
                    if (smax == ZERO) {
                        goto L80;
                    }
                    smin = smax;
                    smaxpr = smax;
                    sminpr = smin;
                    c1 = ONE;
                    c2 = ONE;
                } else {
                    sl_int rank_int = rank;
                    for (j = 0; j < rank; j++) {
                        dwork[jwork2 + j] = abcd[(icol + 1 + j) * ldabcd + irow];
                    }
                    SLC_DLAIC1(&IMIN, &rank_int, &dwork[ismin], &smin,
                               &dwork[jwork2], &abcd[(icol) * ldabcd + irow],
                               &sminpr, &s1, &c1);
                    SLC_DLAIC1(&IMAX, &rank_int, &dwork[ismax], &smax,
                               &dwork[jwork2], &abcd[(icol) * ldabcd + irow],
                               &smaxpr, &s2, &c2);
                    wrkopt = max_i32(wrkopt, 5 * p);
                }

                if (svlmax * rcond <= smaxpr) {
                    if (svlmax * rcond <= sminpr) {
                        if (smaxpr * rcond < sminpr) {
                            if (n1 == 0) {
                                rank++;
                                goto L80;
                            }

                            if (n1 > 1 && i - 1 > 0) {
                                for (j = 0; j < i - 1; j++) {
                                    if (dwork[j] != ZERO) {
                                        t = fabs(abcd[(icol) * ldabcd + irc + j]) / dwork[j];
                                        t = (ONE + t) * (ONE - t);
                                        if (t < ZERO) t = ZERO;
                                        tt = t * pow(dwork[j] / dwork[p + j], 2);
                                        if (tt > tolz) {
                                            dwork[j] = dwork[j] * sqrt(t);
                                        } else {
                                            sl_int n1_minus_1 = n1 - 1;
                                            sl_int ldabcd_int = ldabcd;
                                            dwork[j] = SLC_DNRM2(&n1_minus_1, &abcd[(mp1 - 1) * ldabcd + irc + j], &ldabcd_int);
                                            dwork[p + j] = dwork[j];
                                        }
                                    }
                                }
                            }

                            for (j = 0; j < rank; j++) {
                                dwork[ismin + j] = s1 * dwork[ismin + j];
                                dwork[ismax + j] = s2 * dwork[ismax + j];
                            }

                            dwork[ismin + rank] = c1;
                            dwork[ismax + rank] = c2;
                            smin = sminpr;
                            smax = smaxpr;
                            rank++;
                            icol--;
                            irow--;
                            n1--;
                            i--;
                            continue;
                        }
                    }
                }
                break;
            }

L80:
            mui = rank;
            *nr = *nr - mui;
            *pr = sigma + mui;

            kronl[nblcks - 1] = taui - mui;

            if (first && nblcks > 1) {
                infz[nblcks - 2] = muim1 - taui;
            }
            muim1 = mui;
            ro = mui;

            if (mui == 0) {
                break;
            }
        }
    }

    if (first) {
        if (mui == 0) {
            *dinfz = max_i32(0, nblcks - 1);
        } else {
            *dinfz = nblcks;
            if (nblcks > 0) {
                infz[nblcks - 1] = mui;
            }
        }
        k = *dinfz;
        for (i = k - 1; i >= 0; i--) {
            if (infz[i] != 0) {
                break;
            }
            *dinfz = *dinfz - 1;
        }
        for (i = 0; i < *dinfz; i++) {
            *ninfz += infz[i] * (i + 1);
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
