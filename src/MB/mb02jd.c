/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB02JD - Full QR factorization of a block Toeplitz matrix of full rank
 *
 * Computes a lower triangular matrix R and a matrix Q with Q^T Q = I such that
 * T = Q R^T, where T is a K*M-by-L*N block Toeplitz matrix.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

void mb02jd(const char* job, const i32 k, const i32 l, const i32 m,
            const i32 n, const i32 p, const i32 s, const f64* tc, const i32 ldtc,
            const f64* tr, const i32 ldtr, f64* q, const i32 ldq,
            f64* r, const i32 ldr, f64* dwork, const i32 ldwork, i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    i32 int1 = 1;

    i32 colr, i, ierr, kk, len, nb, nbmin, pdq, pdw, pnq, pnr, pre, pt, rnk;
    i32 shfr, stps, wrkmin, wrkopt;
    bool compq, lquery;
    i32 ipvt[1];

    *info = 0;
    compq = (job[0] == 'Q' || job[0] == 'q');
    bool job_valid = compq || (job[0] == 'R' || job[0] == 'r');

    if (compq) {
        i32 max_p1 = (p > 1) ? p : 1;
        i32 mk_term = m * k;
        i32 nl_term = (n - max_p1) * l;
        i32 max_mk_nl = (mk_term > nl_term) ? mk_term : nl_term;
        wrkmin = 1 + (m * k + (n - 1) * l) * (l + 2 * k) + 6 * l + max_mk_nl;
    } else {
        i32 n_minus_max_p1 = n - ((p > 1) ? p : 1);
        wrkmin = 1 + (n - 1) * l * (l + 2 * k) + 6 * l + n_minus_max_p1 * l;
        if (p == 0) {
            i32 alt = m * k * (l + 1) + l;
            if (alt > wrkmin) {
                wrkmin = alt;
            }
        }
    }

    if (!job_valid) {
        *info = -1;
    } else if (k < 0) {
        *info = -2;
    } else if (l < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else {
        i32 minmknl = (m * k < n * l) ? m * k : n * l;
        if (p * l >= minmknl + l || p < 0) {
            *info = -6;
        } else if ((p + s) * l >= minmknl + l || s < 0) {
            *info = -7;
        } else if (ldtc < ((m * k > 1) ? m * k : 1)) {
            *info = -9;
        } else if (ldtr < ((k > 1) ? k : 1)) {
            *info = -11;
        } else if (ldq < 1 || (compq && ldq < m * k)) {
            *info = -13;
        } else {
            i32 n_minus_p_plus_1 = n - p + 1;
            i32 min_n_term = (n < n_minus_p_plus_1) ? n : n_minus_p_plus_1;
            i32 ldr_min = (min_n_term * l > 1) ? min_n_term * l : 1;
            if (ldr < ldr_min) {
                *info = -15;
            }
        }
    }

    if (*info == 0) {
        lquery = (ldwork == -1);
        if (lquery) {
            wrkopt = 1;
            if (m * k <= l) {
                pdw = m * k * l + m * k;
                i32 mk = m * k;
                SLC_DGEQRF(&mk, &l, dwork, &mk, dwork, dwork, &int1, &ierr);
                i32 opt1 = (i32)dwork[0] + pdw;
                if (opt1 > wrkopt) wrkopt = opt1;
                SLC_DORGQR(&mk, &mk, &mk, dwork, &mk, dwork, dwork, &int1, &ierr);
                opt1 = (i32)dwork[0] + pdw;
                if (opt1 > wrkopt) wrkopt = opt1;
                if (n > 1) {
                    pdw = m * k * m * k;
                    i32 nm1 = n - 1;
                    mb02kd("R", "T", k, l, m, nm1, m * k, ONE, ZERO,
                           (f64*)tc, ldtc, (f64*)tr, ldtr, dwork, m * k, r, ldr, dwork, ldwork, &ierr);
                    opt1 = (i32)dwork[0] + pdw;
                    if (opt1 > wrkopt) wrkopt = opt1;
                }
            } else if (p == 0) {
                if (compq) {
                    i32 mk = m * k;
                    SLC_DGEQRF(&mk, &l, q, &ldq, dwork, dwork, &int1, &ierr);
                    i32 opt1 = (i32)dwork[0] + l;
                    if (opt1 > wrkopt) wrkopt = opt1;
                    SLC_DORGQR(&mk, &l, &l, q, &ldq, dwork, dwork, &int1, &ierr);
                    opt1 = (i32)dwork[0] + l;
                    if (opt1 > wrkopt) wrkopt = opt1;
                    if (n > 1) {
                        i32 nm1 = n - 1;
                        mb02kd("R", "T", k, l, m, nm1, l, ONE, ZERO,
                               (f64*)tc, ldtc, (f64*)tr, ldtr, q, ldq, r, ldr, dwork, ldwork, &ierr);
                        opt1 = (i32)dwork[0];
                        if (opt1 > wrkopt) wrkopt = opt1;
                    }
                } else {
                    pdw = m * k * l;
                    i32 mk = m * k;
                    SLC_DGEQRF(&mk, &l, dwork, &mk, dwork, dwork, &int1, &ierr);
                    i32 opt1 = (i32)dwork[0] + pdw + l;
                    if (opt1 > wrkopt) wrkopt = opt1;
                    SLC_DORGQR(&mk, &l, &l, dwork, &mk, dwork, dwork, &int1, &ierr);
                    opt1 = (i32)dwork[0] + pdw + l;
                    if (opt1 > wrkopt) wrkopt = opt1;
                    if (n > 1) {
                        i32 nm1 = n - 1;
                        mb02kd("R", "T", k, l, m, nm1, l, ONE, ZERO,
                               (f64*)tc, ldtc, (f64*)tr, ldtr, dwork, m * k, r, ldr, dwork, ldwork, &ierr);
                        opt1 = (i32)dwork[0] + pdw;
                        if (opt1 > wrkopt) wrkopt = opt1;
                    }
                }
                pre = 1;
            }

            if (m * k > l || p > 0 || (p == 0 && n > 1)) {
                pre = (p > 1) ? p : 1;
                if (compq) {
                    pdw = (2 * k + l) * ((n - 1) * l + m * k) + 1;
                    i32 len1 = l + m * k;
                    i32 len2 = (n - pre + 1) * l;
                    len = (len1 > len2) ? len1 : len2;
                } else {
                    pdw = (2 * k + l) * (n - 1) * l + 1;
                    len = (n - pre + 1) * l;
                }
                i32 maxlen = (len > 1) ? len : 1;
                SLC_DGELQF(&maxlen, &l, dwork, &maxlen, dwork, dwork, &int1, &ierr);
                i32 opt1 = pdw + 6 * l + (i32)dwork[0];
                if (opt1 > wrkopt) wrkopt = opt1;
            }

            dwork[0] = (f64)wrkopt;
            return;
        } else if (ldwork < wrkmin) {
            dwork[0] = (f64)wrkmin;
            *info = -17;
        }
    }

    if (*info != 0) {
        i32 neg_info = -(*info);
        SLC_XERBLA("MB02JD", &neg_info);
        return;
    }

    i32 minmkls = m;
    if (n < minmkls) minmkls = n;
    if (k * l < minmkls) minmkls = k * l;
    if (s < minmkls) minmkls = s;
    if (minmkls == 0) {
        dwork[0] = ONE;
        return;
    }

    wrkopt = 1;
    if (m * k <= l) {
        i32 mk = m * k;
        SLC_DLACPY("A", &mk, &l, tc, &ldtc, dwork, &mk);
        pdw = m * k * l;
        i32 ldw_remain = ldwork - pdw - m * k;
        SLC_DGEQRF(&mk, &l, dwork, &mk, &dwork[pdw], &dwork[pdw + m * k], &ldw_remain, &ierr);
        i32 opt1 = (i32)dwork[pdw + m * k] + pdw + m * k;
        if (opt1 > wrkopt) wrkopt = opt1;
        ma02ad("U", mk, l, dwork, mk, r, ldr);
        ldw_remain = ldwork - pdw - m * k;
        SLC_DORGQR(&mk, &mk, &mk, dwork, &mk, &dwork[pdw], &dwork[pdw + m * k], &ldw_remain, &ierr);
        opt1 = (i32)dwork[pdw + m * k] + pdw + m * k;
        if (opt1 > wrkopt) wrkopt = opt1;
        if (compq) {
            SLC_DLACPY("A", &mk, &mk, dwork, &mk, q, &ldq);
        }
        pdw = m * k * m * k;
        if (n > 1) {
            i32 nm1 = n - 1;
            i32 ldw_remain2 = ldwork - pdw;
            mb02kd("R", "T", k, l, m, nm1, m * k, ONE, ZERO,
                   (f64*)tc, ldtc, (f64*)tr, ldtr, dwork, m * k, &r[l], ldr, &dwork[pdw], ldw_remain2, &ierr);
            opt1 = (i32)dwork[pdw] + pdw;
            if (opt1 > wrkopt) wrkopt = opt1;
        }
        dwork[0] = (f64)wrkopt;
        return;
    }

    if (p == 0) {
        i32 mk = m * k;

        if (compq) {
            SLC_DLACPY("A", &mk, &l, tc, &ldtc, q, &ldq);
            i32 ldw_remain = ldwork - l;
            SLC_DGEQRF(&mk, &l, q, &ldq, dwork, &dwork[l], &ldw_remain, &ierr);
            i32 opt1 = (i32)dwork[l] + l;
            if (opt1 > wrkopt) wrkopt = opt1;
            ma02ad("U", l, l, q, ldq, r, ldr);
            ldw_remain = ldwork - l;
            SLC_DORGQR(&mk, &l, &l, q, &ldq, dwork, &dwork[l], &ldw_remain, &ierr);
            opt1 = (i32)dwork[l] + l;
            if (opt1 > wrkopt) wrkopt = opt1;
            if (n > 1) {
                i32 nm1 = n - 1;
                mb02kd("R", "T", k, l, m, nm1, l, ONE, ZERO,
                       (f64*)tc, ldtc, (f64*)tr, ldtr, q, ldq, &r[l], ldr, dwork, ldwork, &ierr);
                opt1 = (i32)dwork[0];
                if (opt1 > wrkopt) wrkopt = opt1;
            }
        } else {
            pdw = m * k * l;
            SLC_DLACPY("A", &mk, &l, tc, &ldtc, dwork, &mk);
            i32 ldw_remain = ldwork - pdw - l;
            SLC_DGEQRF(&mk, &l, dwork, &mk, &dwork[pdw], &dwork[pdw + l], &ldw_remain, &ierr);
            i32 opt1 = (i32)dwork[pdw + l] + pdw + l;
            if (opt1 > wrkopt) wrkopt = opt1;
            ma02ad("U", l, l, dwork, mk, r, ldr);
            ldw_remain = ldwork - pdw - l;
            SLC_DORGQR(&mk, &l, &l, dwork, &mk, &dwork[pdw], &dwork[pdw + l], &ldw_remain, &ierr);
            opt1 = (i32)dwork[pdw + l] + pdw + l;
            if (opt1 > wrkopt) wrkopt = opt1;
            if (n > 1) {
                i32 nm1 = n - 1;
                mb02kd("R", "T", k, l, m, nm1, l, ONE, ZERO,
                       (f64*)tc, ldtc, (f64*)tr, ldtr, dwork, m * k, &r[l], ldr, &dwork[pdw], ldwork - pdw, &ierr);
                opt1 = (i32)dwork[pdw] + pdw;
                if (opt1 > wrkopt) wrkopt = opt1;
            }
        }

        if (n == 1) {
            dwork[0] = (f64)wrkopt;
            return;
        }

        pnr = (n - 1) * l * k + 1;
        i32 nm1l = (n - 1) * l;
        ma02ad("A", k, nm1l, tr, ldtr, &dwork[1], nm1l);

        i32 nm1ll = (n - 1) * l * l;
        SLC_DLACPY("A", &nm1l, &l, &r[l], &ldr, &dwork[pnr], &nm1l);

        pt = (m - 1) * k;
        pdw = pnr + nm1ll;

        i32 min_m_nm1 = (m < n - 1) ? m : n - 1;
        for (i = 0; i < min_m_nm1; i++) {
            ma02ad("A", k, l, &tc[pt], ldtc, &dwork[pdw], nm1l);
            pt -= k;
            pdw += l;
        }

        pt = 0;
        for (i = m; i < n - 1; i++) {
            ma02ad("A", k, l, &tr[pt * ldtr], ldtr, &dwork[pdw], nm1l);
            pt += l;
            pdw += l;
        }

        if (compq) {
            pdq = (2 * k + l) * nm1l + 1;
            pdw = (2 * k + l) * (nm1l + m * k) + 1;
            pnq = pdq + m * k * k;
            i32 mk = m * k;
            i32 m1k = (m - 1) * k;
            SLC_DLASET("A", &k, &k, &ZERO, &ONE, &dwork[pdq], &mk);
            SLC_DLASET("A", &m1k, &k, &ZERO, &ZERO, &dwork[pdq + k], &mk);
            SLC_DLACPY("A", &mk, &l, q, &ldq, &dwork[pnq], &mk);
            SLC_DLASET("A", &mk, &k, &ZERO, &ZERO, &dwork[pnq + m * l * k], &mk);
        } else {
            pdw = (2 * k + l) * nm1l + 1;
        }
        pre = 1;
        stps = s - 1;
    } else {
        pnr = (n - 1) * l * k + 1;
        if (compq) {
            pdq = (2 * k + l) * (n - 1) * l + 1;
            pdw = (2 * k + l) * ((n - 1) * l + m * k) + 1;
            pnq = pdq + m * k * k;
        } else {
            pdw = (2 * k + l) * (n - 1) * l + 1;
        }
        pre = p;
        stps = s;
    }

    if (compq) {
        i32 len1 = l + m * k;
        i32 len2 = (n - pre + 1) * l;
        len = (len1 > len2) ? len1 : len2;
    } else {
        len = (n - pre + 1) * l;
    }
    i32 maxlen = (len > 1) ? len : 1;
    i32 available = ldwork - pdw - 6 * l;
    nb = (available > 0) ? available / maxlen : 0;
    if (nb > l) nb = l;

    i32 ilaenv_result;
    {
        i32 neg1 = -1;
        char name[] = "DGELQF";
        char opts[] = " ";
        ilaenv_result = SLC_ILAENV(&int1, name, opts, &maxlen, &l, &neg1, &neg1);
    }
    nbmin = 2;
    if (ilaenv_result > nbmin) nbmin = ilaenv_result;
    if (nb < nbmin) nb = 0;

    colr = l;
    i32 nm1l = (n - 1) * l;
    len = (n - pre) * l;
    shfr = (pre - 1) * l;

    for (i = pre; i < pre + stps; i++) {
        i32 minmknl = (m * k < n * l) ? m * k : n * l;
        kk = l;
        if (minmknl - i * l < l) kk = minmknl - i * l;

        SLC_DLACPY("L", &len, &kk, &r[colr - l + (colr - l) * ldr], &ldr,
                   &r[colr + colr * ldr], &ldr);

        i32 kk_plus_k = kk + k;
        i32 l_plus_k = l + k;
        mb02cu("C", kk, kk_plus_k, l_plus_k, nb, &r[colr + colr * ldr], ldr,
               &dwork[shfr + 1], nm1l, &dwork[pnr + shfr], nm1l,
               &rnk, ipvt, &dwork[pdw], ZERO, &dwork[pdw + 6 * l], ldwork - pdw - 6 * l, &ierr);

        if (ierr != 0) {
            *info = 1;
            return;
        }

        if (len > kk) {
            i32 len_minus_kk = len - kk;
            i32 neg1 = -1;
            mb02cv("C", "N", kk, len_minus_kk, kk_plus_k, l_plus_k, nb, neg1,
                   &r[colr + colr * ldr], ldr, &dwork[shfr + 1], nm1l,
                   &dwork[pnr + shfr], nm1l,
                   &r[colr + kk + colr * ldr], ldr, &dwork[shfr + kk + 1], nm1l,
                   &dwork[pnr + shfr + kk], nm1l,
                   &dwork[pdw], &dwork[pdw + 6 * l], ldwork - pdw - 6 * l, &ierr);
        }

        if (compq) {
            i32 mk = m * k;
            SLC_DLASET("A", &k, &kk, &ZERO, &ZERO, &q[colr * ldq], &ldq);
            if (m > 1) {
                i32 m1k = (m - 1) * k;
                SLC_DLACPY("A", &m1k, &kk, &q[(colr - l) * ldq], &ldq, &q[k + colr * ldq], &ldq);
            }
            i32 neg1 = -1;
            mb02cv("C", "N", kk, mk, kk_plus_k, l_plus_k, nb, neg1,
                   &r[colr + colr * ldr], ldr, &dwork[shfr + 1], nm1l,
                   &dwork[pnr + shfr], nm1l,
                   &q[colr * ldq], ldq, &dwork[pdq], mk, &dwork[pnq], mk,
                   &dwork[pdw], &dwork[pdw + 6 * l], ldwork - pdw - 6 * l, &ierr);
        }

        len -= l;
        colr += l;
        shfr += l;
    }

    dwork[0] = (f64)wrkopt;
}
