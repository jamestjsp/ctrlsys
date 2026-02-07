/*
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"
#include "slicot_blas.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void mb02nd(i32 m, i32 n, i32 l, i32* rank, f64* theta,
            f64* c, i32 ldc, f64* x, i32 ldx, f64* q, bool* inul,
            f64 tol, f64 reltol, i32* iwork, f64* dwork, i32 ldwork,
            bool* bwork, i32* iwarn, i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 NEGONE = -1.0;

    i32 i, i1, ifail, ihoush, ij, ioff, itaup, itauq;
    i32 iwarm, j, j1, jf, jv, jwork, k, kf, kj, ldf, lw;
    i32 mc, minwrk, mj, mnl, nj, nl, p, wrkopt;
    f64 cs, eps, first, fnorm, hh, inprod, rcond, sn, temp;
    bool lfirst, lquery, sufwrk, useqr;
    f64 dummy[2];
    i32 int1 = 1, int0 = 0;

    *iwarn = 0;
    *info = 0;
    nl = n + l;
    k = (m > nl) ? m : nl;
    p = (m < nl) ? m : nl;

    if (m >= nl) {
        lw = (nl * (nl - 1)) / 2;
    } else {
        lw = m * nl - (m * (m - 1)) / 2;
    }
    jv = p + lw + ((6*nl - 5 > l*l + ((nl > 3*l) ? nl : 3*l)) ?
                   6*nl - 5 : l*l + ((nl > 3*l) ? nl : 3*l));

    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (l < 0) {
        *info = -3;
    } else if (*rank > ((m < n) ? m : n)) {
        *info = -4;
    } else if ((*rank < 0) && (*theta < ZERO)) {
        *info = -5;
    } else if (ldc < ((1 > k) ? 1 : k)) {
        *info = -7;
    } else if (ldx < ((1 > n) ? 1 : n)) {
        *info = -9;
    } else {
        i32 ilaenv_val = SLC_ILAENV(&(i32){6}, "DGESVD", "NN", &m, &nl, &int0, &int0);
        useqr = m >= ((nl > ilaenv_val) ? nl : ilaenv_val);
        lquery = (ldwork == -1);

        i32 term1 = k + 2*p;
        minwrk = (2 > term1) ? 2 : term1;
        minwrk = (minwrk > jv) ? minwrk : jv;
        wrkopt = minwrk;

        if (useqr) {
            mnl = nl;
        } else {
            mnl = m;
        }

        if (lquery) {
            if (useqr) {
                SLC_DGEQRF(&m, &nl, c, &ldc, dwork, dwork, &(i32){-1}, &ifail);
                i32 opt_geqrf = (i32)dwork[0];
                wrkopt = (wrkopt > nl + opt_geqrf) ? wrkopt : nl + opt_geqrf;
            }
            SLC_DGEBRD(&mnl, &nl, c, &ldc, q, q, dwork, dwork, dwork, &(i32){-1}, &ifail);
            i32 opt_gebrd = (i32)dwork[0];

            SLC_DORMBR("P", "L", "N", &nl, &nl, &mnl, dwork, &p, dwork, dwork, &nl, &dwork[1], &(i32){-1}, &ifail);
            i32 opt_ormbr = (i32)dwork[1];

            SLC_DGERQF(&l, &nl, dwork, &nl, dwork, dummy, &(i32){-1}, &ifail);
            i32 opt_gerqf = (i32)dummy[0];

            SLC_DORMRQ("R", "T", &n, &nl, &l, dwork, &nl, dwork, dwork, &nl, &dummy[1], &(i32){-1}, &ifail);
            i32 opt_ormrq = (i32)dummy[1];

            i32 max_temp = opt_ormbr;
            max_temp = (max_temp > opt_gerqf) ? max_temp : opt_gerqf;
            max_temp = (max_temp > opt_ormrq) ? max_temp : opt_ormrq;

            i32 opt1 = 2*p + opt_gebrd;
            i32 opt2 = p + p*nl + ((6*nl - 5 > nl*nl + ((max_temp > 3*l) ? max_temp : 3*l)) ?
                                    6*nl - 5 : nl*nl + ((max_temp > 3*l) ? max_temp : 3*l));
            wrkopt = (wrkopt > opt1) ? wrkopt : opt1;
            wrkopt = (wrkopt > opt2) ? wrkopt : opt2;
        }

        if (ldwork < minwrk && !lquery) {
            *info = -16;
        }
    }

    if (*info != 0) {
        i32 neg_info = -(*info);
        SLC_XERBLA("MB02ND", &neg_info);
        return;
    } else if (lquery) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    i32 minmnl = (m < nl) ? m : nl;
    if (minmnl == 0) {
        if (m == 0) {
            SLC_DLASET("F", &nl, &nl, &ZERO, &ONE, c, &ldc);
            SLC_DLASET("F", &n, &l, &ZERO, &ZERO, x, &ldx);
            for (i = 0; i < nl; i++) {
                inul[i] = true;
            }
        }
        if (*rank >= 0) {
            *theta = ZERO;
        }
        *rank = 0;
        dwork[0] = TWO;
        dwork[1] = ONE;
        return;
    }

    wrkopt = 2;

    eps = SLC_DLAMCH("P");
    lfirst = true;

    for (i = 0; i < p; i++) {
        inul[i] = false;
        bwork[i] = false;
    }

    for (i = p; i < nl; i++) {
        inul[i] = true;
        bwork[i] = false;
    }

    if (useqr) {
        itauq = 0;
        jwork = itauq + nl;
        i32 ldwork_geqrf = ldwork - jwork;
        SLC_DGEQRF(&m, &nl, c, &ldc, &dwork[itauq], &dwork[jwork],
                   &ldwork_geqrf, &ifail);
        i32 cur_opt = (i32)dwork[jwork] + jwork;
        wrkopt = (wrkopt > cur_opt) ? wrkopt : cur_opt;

        if (nl > 1) {
            i32 nl_m1 = nl - 1;
            SLC_DLASET("L", &nl_m1, &nl_m1, &ZERO, &ZERO, &c[1], &ldc);
        }
    }

    itaup = 0;
    itauq = itaup + p;
    jwork = itauq + p;
    i32 ldwork_gebrd = ldwork - jwork;
    SLC_DGEBRD(&mnl, &nl, c, &ldc, q, &q[p], &dwork[itauq],
               &dwork[itaup], &dwork[jwork], &ldwork_gebrd, &ifail);
    i32 cur_opt = (i32)dwork[jwork] + jwork;
    wrkopt = (wrkopt > cur_opt) ? wrkopt : cur_opt;

    if (m < nl) {
        ioff = 0;
        for (i = 0; i < p - 1; i++) {
            SLC_DLARTG(&q[i], &q[p + i], &cs, &sn, &temp);
            q[i] = temp;
            q[p + i] = sn * q[i + 1];
            q[i + 1] = cs * q[i + 1];
        }
    } else {
        ioff = 1;
    }

    ihoush = itauq;
    mc = nl - ioff;
    kf = ihoush + p * nl;
    i32 term2 = l*l + ((nl > 3*l) ? nl : 3*l);
    sufwrk = ldwork >= (kf + ((6*nl - 5 > term2) ? 6*nl - 5 : term2));

    if (sufwrk) {
        SLC_DLACPY("U", &p, &nl, c, &ldc, &dwork[ihoush], &p);
        kj = kf;
        wrkopt = (wrkopt > kf) ? wrkopt : kf;
    } else {
        kj = ihoush;
        i32 limit = (p < mc) ? p : mc;
        for (nj = 0; nj < limit; nj++) {
            j = mc - nj;
            SLC_DCOPY(&j, &c[nj + (nj + ioff) * ldc], &ldc, &dwork[kj], &int1);
            kj = kj + j;
        }
    }

    SLC_DLASET("F", &nl, &nl, &ZERO, &ONE, c, &ldc);
    jv = kj;
    iwarm = 0;

label_100:
    jwork = jv;
    i32 ldwork_mb04yd = ldwork - jwork;
    mb04yd("N", "U", p, nl, rank, theta, q, &q[p], dummy, 1, c, ldc,
           inul, tol, reltol, &dwork[jwork], ldwork_mb04yd, iwarn, info);
    wrkopt = (wrkopt > jwork + 6*nl - 5) ? wrkopt : jwork + 6*nl - 5;

    *iwarn = (*iwarn > iwarm) ? *iwarn : iwarm;
    if (*info > 0) {
        return;
    }

    k = 0;
    for (i = 0; i < nl; i++) {
        if (inul[i]) {
            iwork[k] = i + 1;
            k++;
        }
    }

    if (k < l) {
        *info = 2;
        return;
    }

    kf = k;

    if (sufwrk && lfirst) {
        ij = jv;
        for (j = 0; j < k; j++) {
            i32 col_idx = iwork[j] - 1;
            SLC_DCOPY(&nl, &c[col_idx * ldc], &int1, &dwork[ij], &int1);
            ij = ij + nl;
        }

        ij = jv;
        jwork = ij + nl * k;
        i32 ldwork_ormbr = ldwork - jwork;
        SLC_DORMBR("P", "L", "N", &nl, &k, &mnl, &dwork[ihoush], &p,
                   &dwork[itaup], &dwork[ij], &nl, &dwork[jwork],
                   &ldwork_ormbr, &ifail);
        cur_opt = (i32)dwork[jwork] + jwork;
        wrkopt = (wrkopt > cur_opt) ? wrkopt : cur_opt;

        for (i = 0; i < nl; i++) {
            if (inul[i] && !bwork[i]) {
                bwork[i] = true;
            }
        }
    } else {
        for (i = 0; i < nl; i++) {
            if (inul[i] && !bwork[i]) {
                kj = jv;
                i32 limit = (p < mc) ? p : mc;
                for (nj = limit - 1; nj >= 0; nj--) {
                    j = mc - nj;
                    kj = kj - j;
                    first = dwork[kj];
                    dwork[kj] = ONE;
                    SLC_DLARF("L", &j, &int1, &dwork[kj], &int1,
                              &dwork[itaup + nj], &c[(nj + ioff) + i * ldc], &ldc,
                              &dwork[jwork]);
                    dwork[kj] = first;
                }
                bwork[i] = true;
            }
        }
    }

    if (*rank <= 0) {
        *rank = 0;
    }
    if ((*rank == 0 || l == 0) && !((*rank == 0) && (l > 0))) {
        if (sufwrk && lfirst) {
            SLC_DLACPY("F", &nl, &k, &dwork[jv], &nl, c, &ldc);
        }
        dwork[0] = (f64)wrkopt;
        dwork[1] = ONE;
        return;
    }

    if (sufwrk && lfirst) {
        itauq = jwork;
        jwork = itauq + l;
        i32 ldwork_gerqf = ldwork - jwork;
        SLC_DGERQF(&l, &k, &dwork[jv + n], &nl, &dwork[itauq], &dwork[jwork],
                   &ldwork_gerqf, info);
        cur_opt = (i32)dwork[jwork] + jwork;
        wrkopt = (wrkopt > cur_opt) ? wrkopt : cur_opt;

        i32 ldwork_ormrq = ldwork - jwork;
        SLC_DORMRQ("R", "T", &n, &k, &l, &dwork[jv + n], &nl,
                   &dwork[itauq], &dwork[jv], &nl, &dwork[jwork],
                   &ldwork_ormrq, info);
        cur_opt = (i32)dwork[jwork] + jwork;
        wrkopt = (wrkopt > cur_opt) ? wrkopt : cur_opt;

        jf = jv + nl * (k - l) + n;
        ldf = nl;
        jwork = jf + ldf * l - n;
        i32 k_m_l = k - l;
        SLC_DLASET("F", &l, &k_m_l, &ZERO, &ZERO, &dwork[jv + n], &ldf);
        if (l > 1) {
            i32 l_m1 = l - 1;
            SLC_DLASET("L", &l_m1, &l_m1, &ZERO, &ZERO, &dwork[jf + 1], &ldf);
        }
        ij = jv;
        for (j = 0; j < k; j++) {
            i32 col_idx = iwork[j] - 1;
            SLC_DCOPY(&nl, &dwork[ij], &int1, &c[col_idx * ldc], &int1);
            ij = ij + nl;
        }
    } else {
        i = nl;
        jf = jv;
        ldf = l;
        jwork = jf + ldf * l;
        wrkopt = (wrkopt > jwork + nl) ? wrkopt : jwork + nl;

        while (k >= 1 && i > n) {
            for (j = 0; j < k; j++) {
                i32 col_idx = iwork[j] - 1;
                dwork[jwork + j] = c[(i - 1) + col_idx * ldc];
            }

            SLC_DLARFG(&k, &dwork[jwork + k - 1], &dwork[jwork], &int1, &temp);
            i32 col_idx_k = iwork[k - 1] - 1;
            c[(i - 1) + col_idx_k * ldc] = dwork[jwork + k - 1];

            if (temp != ZERO) {
                for (i1 = 0; i1 < i - 1; i1++) {
                    i32 col_idx_last = iwork[k - 1] - 1;
                    inprod = c[i1 + col_idx_last * ldc];
                    for (j = 0; j < k - 1; j++) {
                        i32 col_idx_j = iwork[j] - 1;
                        inprod += dwork[jwork + j] * c[i1 + col_idx_j * ldc];
                    }
                    hh = inprod * temp;
                    c[i1 + col_idx_last * ldc] -= hh;
                    for (j = 0; j < k - 1; j++) {
                        j1 = iwork[j] - 1;
                        c[i1 + j1 * ldc] -= dwork[jwork + j] * hh;
                        c[(i - 1) + j1 * ldc] = ZERO;
                    }
                }
            }
            i32 i_m_n = i - n;
            i32 col_idx_last = iwork[k - 1] - 1;
            SLC_DCOPY(&i_m_n, &c[n + col_idx_last * ldc], &int1,
                      &dwork[jf + (i - n - 1) * l], &int1);
            k--;
            i--;
        }
    }

    SLC_DTRCON("1", "U", "N", &l, &dwork[jf], &ldf, &rcond, &dwork[jwork],
               &iwork[kf], info);
    wrkopt = (wrkopt > jwork + 3*l) ? wrkopt : jwork + 3*l;

    for (j = 0; j < l; j++) {
        i32 col_idx = iwork[kf - l + j] - 1;
        SLC_DCOPY(&n, &c[col_idx * ldc], &int1, &x[j * ldx], &int1);
    }

    fnorm = SLC_DLANTR("1", "U", "N", &l, &l, &dwork[jf], &ldf, &dwork[jwork]);

    if (rcond <= eps * fnorm) {
        (*rank)--;
        goto label_340;
    }
    f64 ynorm = SLC_DLANGE("1", &n, &l, x, &ldx, &dwork[jwork]);
    if (fnorm <= eps * ynorm) {
        *rank -= l;
        goto label_340;
    } else {
        goto label_400;
    }

label_340:
    iwarm = 2;
    *theta = -ONE;

    if (sufwrk && lfirst) {
        if (p < nl) {
            kj = ihoush + nl * (nl - 1);
            mj = ihoush + p * (nl - 1);
            for (nj = 0; nj < nl; nj++) {
                for (j = p - 1; j >= 0; j--) {
                    dwork[kj + j] = dwork[mj + j];
                }
                kj -= nl;
                mj -= p;
            }
        }
        kj = ihoush;
        mj = ihoush + nl * ioff;
        i32 limit = (p < mc) ? p : mc;
        for (nj = 0; nj < limit; nj++) {
            for (j = 0; j <= mc - nj - 1; j++) {
                dwork[kj] = dwork[mj + j * p];
                kj++;
            }
            mj += nl + 1;
        }
        jv = kj;
        lfirst = false;
    }
    goto label_100;

label_400:
    SLC_DTRSM("R", "U", "N", "N", &n, &l, &NEGONE, &dwork[jf], &ldf, x, &ldx);

    dwork[0] = (f64)wrkopt;
    dwork[1] = rcond;
}
