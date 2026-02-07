/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB02JX - Low rank QR factorization with column pivoting of a block Toeplitz matrix
 *
 * Computes: T P = Q R^T
 * where R is lower trapezoidal, P is a block permutation matrix, Q^T Q = I.
 * The number of columns in R (RNK) is the numerical rank of T.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

void mb02jx(const char* job, const i32 k, const i32 l, const i32 m,
            const i32 n, const f64* tc, const i32 ldtc, const f64* tr,
            const i32 ldtr, i32* rnk, f64* q, const i32 ldq,
            f64* r, const i32 ldr, i32* jpvt, const f64 tol1, const f64 tol2,
            f64* dwork, const i32 ldwork, i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    i32 int1 = 1;

    i32 cpcol, gap, i, ierr, j, jj, jwork, kk, len, mk;
    i32 nzc, pdp, pdq, pdw, pnq, pnr, pp, ppr, pt, rdef;
    i32 rrdf, rrnk, wrkmin, wrkopt;
    f64 ltol1, ltol2, nrm, temp;
    bool compq, last;

    *info = 0;
    wrkopt = 3;
    mk = m * k;
    compq = (job[0] == 'Q' || job[0] == 'q');
    bool job_valid = compq || (job[0] == 'R' || job[0] == 'r');

    if (compq) {
        i32 term1 = (mk + (n - 1) * l) * (l + 2 * k) + 9 * l;
        i32 max_term = (mk > (n - 1) * l) ? mk : (n - 1) * l;
        wrkmin = (3 > term1 + max_term) ? 3 : term1 + max_term;
    } else {
        i32 term1 = (n - 1) * l * (l + 2 * k + 1) + 9 * l;
        i32 term2 = mk * (l + 1) + l;
        wrkmin = 3;
        if (term1 > wrkmin) wrkmin = term1;
        if (term2 > wrkmin) wrkmin = term2;
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
    } else if (ldtc < ((mk > 1) ? mk : 1)) {
        *info = -7;
    } else if (ldtr < ((k > 1) ? k : 1)) {
        *info = -9;
    } else if (ldq < 1 || (compq && ldq < mk)) {
        *info = -12;
    } else if (ldr < ((n * l > 1) ? n * l : 1)) {
        *info = -14;
    } else if (ldwork < wrkmin) {
        dwork[0] = (f64)wrkmin;
        *info = -19;
    }

    if (*info != 0) {
        i32 neg_info = -(*info);
        SLC_XERBLA("MB02JX", &neg_info);
        return;
    }

    i32 minval = m;
    if (n < minval) minval = n;
    if (k < minval) minval = k;
    if (l < minval) minval = l;
    if (minval == 0) {
        *rnk = 0;
        dwork[0] = (f64)wrkopt;
        dwork[1] = ZERO;
        dwork[2] = ZERO;
        return;
    }

    wrkopt = wrkmin;

    if (mk <= l) {
        SLC_DLACPY("A", &mk, &l, tc, &ldtc, dwork, &mk);
        pdw = mk * l;
        jwork = pdw + mk;
        i32 ldw_remain = ldwork - jwork;
        SLC_DGEQRF(&mk, &l, dwork, &mk, &dwork[pdw], &dwork[jwork], &ldw_remain, &ierr);
        i32 opt1 = (i32)dwork[jwork] + jwork;
        if (opt1 > wrkopt) wrkopt = opt1;
        ma02ad("U", mk, l, dwork, mk, r, ldr);
        ldw_remain = ldwork - jwork;
        SLC_DORGQR(&mk, &mk, &mk, dwork, &mk, &dwork[pdw], &dwork[jwork], &ldw_remain, &ierr);
        opt1 = (i32)dwork[jwork] + jwork;
        if (opt1 > wrkopt) wrkopt = opt1;
        if (compq) {
            SLC_DLACPY("A", &mk, &mk, dwork, &mk, q, &ldq);
        }
        pdw = mk * mk;
        if (n > 1) {
            i32 nm1 = n - 1;
            i32 ldw_remain2 = ldwork - pdw;
            mb02kd("R", "T", k, l, m, nm1, mk, ONE, ZERO,
                   (f64*)tc, ldtc, (f64*)tr, ldtr, dwork, mk, &r[l], ldr, &dwork[pdw], ldw_remain2, &ierr);
        }
        opt1 = (i32)dwork[pdw] + pdw;
        if (opt1 > wrkopt) wrkopt = opt1;

        for (i = 0; i < mk; i++) {
            jpvt[i] = i + 1;
        }

        *rnk = mk;
        dwork[0] = (f64)wrkopt;
        dwork[1] = ZERO;
        dwork[2] = ZERO;
        return;
    }

    for (i = 0; i < l; i++) {
        jpvt[i] = 0;
    }

    ltol1 = tol1;
    ltol2 = tol2;

    if (compq) {
        SLC_DLACPY("A", &mk, &l, tc, &ldtc, q, &ldq);
        i32 ldw_remain = ldwork - l;
        SLC_DGEQP3(&mk, &l, q, &ldq, jpvt, dwork, &dwork[l], &ldw_remain, &ierr);
        i32 opt1 = (i32)dwork[l] + l;
        if (opt1 > wrkopt) wrkopt = opt1;

        if (ltol1 < ZERO) {
            temp = ONE / sqrt((f64)l);
            SLC_DLASET("A", &l, &int1, &temp, &temp, &dwork[l], &int1);

            for (i = 0; i < 5; i++) {
                SLC_DTRMV("U", "N", "N", &l, q, &ldq, &dwork[l], &int1);
                SLC_DTRMV("U", "T", "N", &l, q, &ldq, &dwork[l], &int1);
                nrm = SLC_DNRM2(&l, &dwork[l], &int1);
                f64 inv_nrm = ONE / nrm;
                SLC_DSCAL(&l, &inv_nrm, &dwork[l], &int1);
            }

            ltol1 = sqrt(nrm * SLC_DLAMCH("E"));
        }

        i = l - 1;
        while (i >= 0 && fabs(q[i + i * ldq]) <= ltol1) {
            i--;
        }

        rrnk = i + 1;
        rrdf = l - rrnk;
        ma02ad("U", rrnk, l, q, ldq, r, ldr);
        if (rrnk > 1) {
            i32 lm1 = l - 1;
            i32 rrnkm1 = rrnk - 1;
            SLC_DLASET("U", &lm1, &rrnkm1, &ZERO, &ZERO, &r[ldr], &ldr);
        }
        ldw_remain = ldwork - l;
        SLC_DORGQR(&mk, &l, &rrnk, q, &ldq, dwork, &dwork[l], &ldw_remain, &ierr);
        opt1 = (i32)dwork[l] + l;
        if (opt1 > wrkopt) wrkopt = opt1;
        if (n > 1) {
            i32 nm1 = n - 1;
            mb02kd("R", "T", k, l, m, nm1, rrnk, ONE, ZERO,
                   (f64*)tc, ldtc, (f64*)tr, ldtr, q, ldq, &r[l], ldr, dwork, ldwork, &ierr);
            opt1 = (i32)dwork[0];
            if (opt1 > wrkopt) wrkopt = opt1;
        }
    } else {
        pdw = mk * l;
        jwork = pdw + l;
        SLC_DLACPY("A", &mk, &l, tc, &ldtc, dwork, &mk);
        i32 ldw_remain = ldwork - jwork;
        SLC_DGEQP3(&mk, &l, dwork, &mk, jpvt, &dwork[pdw], &dwork[jwork], &ldw_remain, &ierr);
        i32 opt1 = (i32)dwork[jwork] + jwork;
        if (opt1 > wrkopt) wrkopt = opt1;

        if (ltol1 < ZERO) {
            temp = ONE / sqrt((f64)l);
            SLC_DLASET("A", &l, &int1, &temp, &temp, &dwork[jwork], &int1);

            for (i = 0; i < 5; i++) {
                SLC_DTRMV("U", "N", "N", &l, dwork, &mk, &dwork[jwork], &int1);
                SLC_DTRMV("U", "T", "N", &l, dwork, &mk, &dwork[jwork], &int1);
                nrm = SLC_DNRM2(&l, &dwork[jwork], &int1);
                f64 inv_nrm = ONE / nrm;
                SLC_DSCAL(&l, &inv_nrm, &dwork[jwork], &int1);
            }

            ltol1 = sqrt(nrm * SLC_DLAMCH("E"));
        }

        rrnk = l;
        i = (l - 1) * mk + l - 1;
        while (rrnk > 0 && fabs(dwork[i]) <= ltol1) {
            rrnk--;
            i = i - mk - 1;
        }

        rrdf = l - rrnk;
        ma02ad("U", rrnk, l, dwork, mk, r, ldr);
        if (rrnk > 1) {
            i32 lm1 = l - 1;
            i32 rrnkm1 = rrnk - 1;
            SLC_DLASET("U", &lm1, &rrnkm1, &ZERO, &ZERO, &r[ldr], &ldr);
        }
        ldw_remain = ldwork - jwork;
        SLC_DORGQR(&mk, &l, &rrnk, dwork, &mk, &dwork[pdw], &dwork[jwork], &ldw_remain, &ierr);
        opt1 = (i32)dwork[jwork] + jwork;
        if (opt1 > wrkopt) wrkopt = opt1;
        if (n > 1) {
            i32 nm1 = n - 1;
            i32 ldw_remain2 = ldwork - pdw;
            mb02kd("R", "T", k, l, m, nm1, rrnk, ONE, ZERO,
                   (f64*)tc, ldtc, (f64*)tr, ldtr, dwork, mk, &r[l], ldr, &dwork[pdw], ldw_remain2, &ierr);
            opt1 = (i32)dwork[pdw] + pdw;
            if (opt1 > wrkopt) wrkopt = opt1;
        }
    }

    if (n == 1) {
        *rnk = rrnk;
        dwork[0] = (f64)wrkopt;
        dwork[1] = ltol1;
        dwork[2] = ZERO;
        return;
    }

    if (ltol2 < ZERO) {
        ltol2 = (f64)(n * l) * sqrt(SLC_DLAMCH("E"));
    }

    for (j = 0; j < l; j++) {
        for (i32 ii = 0; ii < rrnk; ii++) {
            r[(l + jpvt[j] - 1) + (rrnk + ii) * ldr] = r[j + ii * ldr];
        }
    }

    if (n > 2) {
        i32 nm2l = (n - 2) * l;
        SLC_DLACPY("A", &nm2l, &rrnk, &r[l], &ldr, &r[2 * l + rrnk * ldr], &ldr);
    }

    i32 nm1l = (n - 1) * l;
    i32 min_rrdf_k = (rrdf < k) ? rrdf : k;
    if (rrdf > 0) {
        ma02ad("A", min_rrdf_k, nm1l, tr, ldtr, &r[l + (2 * rrnk) * ldr], ldr);
    }
    if (k > rrdf) {
        ma02ad("A", k - rrdf, nm1l, &tr[rrdf], ldtr, dwork, nm1l);
    }

    i32 max_k_minus_rrdf = (k - rrdf > 0) ? k - rrdf : 0;
    pnr = nm1l * max_k_minus_rrdf;
    SLC_DLACPY("A", &nm1l, &rrnk, &r[l], &ldr, &dwork[pnr], &nm1l);

    pdw = pnr + nm1l * rrnk;
    pt = (m - 1) * k;

    i32 min_m_nm1 = (m < n - 1) ? m : (n - 1);
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
        pdq = pnr + nm1l * (rrnk + k);
        pnq = pdq + mk * max_k_minus_rrdf;
        pdw = pnq + mk * (rrnk + k);
        SLC_DLACPY("A", &mk, &rrnk, q, &ldq, &dwork[pnq], &mk);
        if (m > 1) {
            i32 m1k = (m - 1) * k;
            SLC_DLACPY("A", &m1k, &rrnk, q, &ldq, &q[k + (rrnk) * ldq], &ldq);
        }
        SLC_DLASET("A", &k, &rrnk, &ZERO, &ZERO, &q[rrnk * ldq], &ldq);
        if (rrdf > 0) {
            SLC_DLASET("A", &mk, &rrdf, &ZERO, &ONE, &q[(2 * rrnk) * ldq], &ldq);
        }
        SLC_DLASET("A", &rrdf, &max_k_minus_rrdf, &ZERO, &ZERO, &dwork[pdq], &mk);
        i32 mk_minus_rrdf = mk - rrdf;
        SLC_DLASET("A", &mk_minus_rrdf, &max_k_minus_rrdf, &ZERO, &ONE, &dwork[pdq + rrdf], &mk);
        SLC_DLASET("A", &mk, &k, &ZERO, &ZERO, &dwork[pnq + mk * rrnk], &mk);
    } else {
        pdw = pnr + nm1l * (rrnk + k);
    }
    ppr = 0;
    *rnk = rrnk;
    rdef = rrdf;
    len = n * l;
    gap = n * l - ((n * l < mk) ? n * l : mk);

    kk = l + k - rdef;
    if (kk > l) kk = l;
    if (kk > mk - l) kk = mk - l;

    for (i32 blk = l; blk < ((mk < n * l) ? mk : n * l); blk += l) {
        if (blk + l <= ((mk < n * l) ? mk : n * l)) {
            last = false;
        } else {
            last = true;
        }
        pp = kk + ((k - rdef > 0) ? k - rdef : 0);
        len -= l;

        i32 lkrdef = l + k - rdef;
        i32 neg1 = -1;
        mb02cu("D", kk, pp, lkrdef, neg1, &r[blk + (*rnk) * ldr], ldr,
               &dwork[ppr], nm1l, &dwork[pnr], nm1l,
               &rrnk, &jpvt[blk], &dwork[pdw], ltol1, &dwork[pdw + 5 * l],
               ldwork - pdw - 5 * l, &ierr);
        if (ierr != 0) {
            *info = 1;
            return;
        }

        pdp = pdw + 6 * l - blk;
        for (j = blk; j < blk + kk; j++) {
            jpvt[j] = jpvt[j] + blk;
            dwork[pdp + jpvt[j]] = (f64)(j + 1);
        }

        for (j = blk; j < blk + kk; j++) {
            temp = (f64)(j + 1);
            jj = j;
            while (dwork[pdp + jj] != temp) {
                jj++;
            }
            if (jj != j) {
                dwork[pdp + jj] = dwork[pdp + j];
                SLC_DSWAP(rnk, &r[j], &ldr, &r[jj], &ldr);
            }
        }

        for (j = blk + kk; j < blk + l; j++) {
            jpvt[j] = j + 1;
        }

        if (len > kk) {
            i32 len_minus_kk_minus_gap = len - kk - gap;
            i32 neg1 = -1;
            mb02cv("D", "N", kk, len_minus_kk_minus_gap, pp, lkrdef, neg1, rrnk,
                   &r[blk + (*rnk) * ldr], ldr, &dwork[ppr], nm1l, &dwork[pnr], nm1l,
                   &r[blk + kk + (*rnk) * ldr], ldr, &dwork[ppr + kk], nm1l,
                   &dwork[pnr + kk], nm1l, &dwork[pdw], &dwork[pdw + 5 * l],
                   ldwork - pdw - 5 * l, &ierr);
        }

        if (compq) {
            i32 neg1 = -1;
            mb02cv("D", "N", kk, mk, pp, lkrdef, neg1, rrnk,
                   &r[blk + (*rnk) * ldr], ldr, &dwork[ppr], nm1l, &dwork[pnr], nm1l,
                   &q[(*rnk) * ldq], ldq, &dwork[pdq], mk, &dwork[pnq], mk,
                   &dwork[pdw], &dwork[pdw + 5 * l], ldwork - pdw - 5 * l, &ierr);
        }

        nzc = 0;
        for (j = kk - 1; j >= rrnk; j--) {
            if (fabs(r[(blk + j) + (*rnk + j) * ldr]) <= ltol1) nzc++;
        }

        pt = pnr;
        for (j = rrnk; j < kk - nzc; j++) {
            temp = r[(blk + j) + (*rnk + j) * ldr];
            i32 count = len - j - 1 - gap;
            if (count > 0) {
                SLC_DSCAL(&count, &temp, &r[(blk + j + 1) + (*rnk + j) * ldr], &int1);
                f64 neg_dwork_pt_j = -dwork[pt + j];
                SLC_DAXPY(&count, &neg_dwork_pt_j, &dwork[pt + j + 1], &int1,
                          &r[(blk + j + 1) + (*rnk + j) * ldr], &int1);
                if (SLC_DNRM2(&count, &r[(blk + j + 1) + (*rnk + j) * ldr], &int1) >
                    ltol2 * fabs(temp)) {
                    *info = 2;
                    return;
                }
            }
            pt += nm1l;
        }

        rrdf = kk - rrnk;
        i32 blk_1 = blk;
        SLC_DLASET("A", &blk_1, &rrnk, &ZERO, &ZERO, &r[(*rnk) * ldr], &ldr);
        i32 lm1 = l - 1;
        i32 rrnkm1 = rrnk - 1;
        if (rrnkm1 > 0) {
            SLC_DLASET("U", &lm1, &rrnkm1, &ZERO, &ZERO, &r[blk + (*rnk + 1) * ldr], &ldr);
        }

        if (!last) {
            i32 new_kk = l + k - rdef - rrdf + nzc;
            if (new_kk > l) new_kk = l;
            if (new_kk > mk - blk - l) new_kk = mk - blk - l;

            if (new_kk <= 0) {
                *rnk = *rnk + rrnk;
                goto finish;
            }

            SLC_DLASET("A", &l, &rrdf, &ZERO, &ZERO, &r[blk + (*rnk + rrnk) * ldr], &ldr);

            if ((rrdf - nzc) > 0 && nzc > 0) {
                cpcol = (nzc < new_kk) ? nzc : new_kk;
                for (j = *rnk + rrnk; j < *rnk + rrnk + cpcol; j++) {
                    i32 len_minus_l = len - l;
                    if (len_minus_l > 0) {
                        SLC_DCOPY(&len_minus_l, &r[(blk + l) + (j + rrdf - nzc) * ldr], &int1,
                                  &r[(blk + l) + j * ldr], &int1);
                    }
                }
            }

            cpcol = (rrnk < new_kk - nzc) ? rrnk : (new_kk - nzc);
            if (cpcol > 0) {
                for (j = blk; j < blk + l; j++) {
                    for (i32 jj = 0; jj < cpcol; jj++) {
                        r[(jpvt[j] + l - 1) + (*rnk + rrnk + nzc + jj) * ldr] =
                            r[j + (*rnk + jj) * ldr];
                    }
                }

                if (len > 2 * l) {
                    i32 len_minus_2l = len - 2 * l;
                    SLC_DLACPY("A", &len_minus_2l, &cpcol, &r[(blk + l) + (*rnk) * ldr], &ldr,
                               &r[(blk + 2 * l) + (*rnk + rrnk + nzc) * ldr], &ldr);
                }
            }
            ppr += l;

            cpcol = (k - rdef < new_kk - rrnk - nzc) ? k - rdef : (new_kk - rrnk - nzc);
            if (cpcol > 0) {
                i32 len_minus_l = len - l;
                if (len_minus_l > 0) {
                    SLC_DLACPY("A", &len_minus_l, &cpcol, &dwork[ppr], &nm1l,
                               &r[(blk + l) + (*rnk + 2 * rrnk + nzc) * ldr], &ldr);
                }
                ppr += cpcol * nm1l;
            }
            pnr += (rrdf - nzc) * nm1l + l;

            if (compq) {
                if ((rrdf - nzc) > 0 && nzc > 0) {
                    cpcol = (nzc < new_kk) ? nzc : new_kk;
                    for (j = *rnk + rrnk; j < *rnk + rrnk + cpcol; j++) {
                        SLC_DCOPY(&mk, &q[(j + rrdf - nzc) * ldq], &int1, &q[j * ldq], &int1);
                    }
                }
                cpcol = (rrnk < new_kk - nzc) ? rrnk : (new_kk - nzc);
                if (cpcol > 0) {
                    SLC_DLASET("A", &k, &cpcol, &ZERO, &ZERO, &q[(*rnk + rrnk + nzc) * ldq], &ldq);
                    if (m > 1) {
                        i32 m1k = (m - 1) * k;
                        SLC_DLACPY("A", &m1k, &cpcol, &q[(*rnk) * ldq], &ldq,
                                   &q[k + (*rnk + rrnk + nzc) * ldq], &ldq);
                    }
                }
                cpcol = (k - rdef < new_kk - rrnk - nzc) ? k - rdef : (new_kk - rrnk - nzc);
                if (cpcol > 0) {
                    SLC_DLACPY("A", &mk, &cpcol, &dwork[pdq], &mk,
                               &q[(*rnk + 2 * rrnk + nzc) * ldq], &ldq);
                    pdq += cpcol * mk;
                }
                pnq += (rrdf - nzc) * mk;
            }
            kk = new_kk;
        }
        *rnk = *rnk + rrnk;
        rdef = rdef + rrdf - nzc;
    }

finish:
    dwork[0] = (f64)wrkopt;
    dwork[1] = ltol1;
    dwork[2] = ltol2;
}
