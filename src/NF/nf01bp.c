/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

/**
 * @brief Compute Levenberg-Marquardt parameter for Wiener system.
 *
 * @param[in] cond Condition estimation mode.
 * @param[in] n Order of matrix R.
 * @param[in] ipar Integer parameters.
 * @param[in] lipar Length of ipar.
 * @param[in,out] r Matrix R.
 * @param[in] ldr Leading dimension of R.
 * @param[in] ipvt Permutation matrix.
 * @param[in] diag Diagonal scaling.
 * @param[in] qtb Q'*b.
 * @param[in] delta Trust region radius.
 * @param[in,out] par LM parameter.
 * @param[in,out] ranks Ranks.
 * @param[out] x Solution.
 * @param[out] rx Residual.
 * @param[in] tol Tolerance.
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
void nf01bp(const char *cond, i32 n, const i32 *ipar, i32 lipar, f64 *r, i32 ldr,
            const i32 *ipvt, const f64 *diag, const f64 *qtb, f64 delta,
            f64 *par, i32 *ranks, f64 *x, f64 *rx, f64 tol, f64 *dwork,
            i32 ldwork, i32 *info)
{
    const i32 itmax = 10;
    const f64 p1 = 0.1, p001 = 0.001, zero = 0.0, one = 1.0;
    i32 bn, bsm, bsn, i, ibsn, iter, j, jw, k, l, lds, n2, nths, rank, st;
    f64 dmino, dwarf, dxnorm, fp, gnorm, parc, parl, paru, sum, temp, toldef;
    bool badrk, econd, ncond, sing, ucond;
    char condl[2] = {0, 0};
    i32 inc_1 = 1;

    econd = (cond[0] == 'E' || cond[0] == 'e');
    ncond = (cond[0] == 'N' || cond[0] == 'n');
    ucond = (cond[0] == 'U' || cond[0] == 'u');
    *info = 0;
    n2 = 2 * n;

    if (!(econd || ncond || ucond)) *info = -1;
    else if (n < 0) *info = -2;
    else if (lipar < 4) *info = -4;
    else if (ldr < (n > 0 ? n : 1)) *info = -6;
    else if (delta <= zero) *info = -10;
    else if (*par < zero) *info = -11;
    else {
        st = ipar[0];
        bn = ipar[1];
        bsm = ipar[2];
        bsn = ipar[3];
        nths = bn * bsn;

        if (st < 0 || bn < 0 || bsm < 0 || bsn < 0) *info = -3;
        else if (n != nths + st) *info = -2;
        else {
            if (n > 0) {
                dmino = diag[0];
                sing = false;
                for (j = 0; j < n; j++) {
                    if (diag[j] < dmino) dmino = diag[j];
                    sing = sing || (diag[j] == zero);
                }
            } else {
                dmino = zero;
                sing = false;
            }

            if (sing) *info = -8;
            else if (ucond) {
                badrk = false;
                if (bn <= 1 || bsn == 0) {
                    if (n > 0) badrk = (ranks[0] < 0 || ranks[0] > n);
                } else {
                    rank = 0;
                    for (k = 0; k < bn; k++) {
                        badrk = badrk || (ranks[k] < 0) || (ranks[k] > bsn);
                        rank += ranks[k];
                    }
                    if (st > 0) {
                        badrk = badrk || (ranks[bn] < 0) || (ranks[bn] > st);
                        rank += ranks[bn];
                    }
                }
                if (badrk) *info = -12;
            } else {
                jw = n2;
                if (bn <= 1 || bsn == 0) {
                    if (econd) jw = 4 * n;
                } else {
                    jw = st * nths + jw;
                    if (econd) jw = 2 * ((bsn > st) ? bsn : st) + jw;
                }
                if (ldwork < jw) *info = -17;
            }
        }
    }

    if (*info != 0) {
        i32 err_code = -(*info);
        SLC_XERBLA("NF01BP", &err_code);
        return;
    }

    if (n == 0) {
        *par = zero;
        return;
    }

    if (bn <= 1 || bsn == 0) {
        md03by(cond, n, r, ldr, ipvt, diag, qtb, delta, par, ranks, x, rx, tol, dwork, ldwork, info);
        return;
    }

    dwarf = SLC_DLAMCH("Underflow");

    SLC_DCOPY(&n, qtb, &inc_1, rx, &inc_1);
    nf01br(cond, "Upper", "No transpose", n, ipar, lipar, r, ldr, dwork, dwork, 1, rx, ranks, tol, dwork, ldwork, info);

    for (j = 0; j < n; j++) {
        l = ipvt[j] - 1;
        if (l < 0 || l >= n) {
            *info = -7;
            return;
        }
        x[l] = rx[j];
    }

    iter = 0;
    for (j = 0; j < n; j++) dwork[j] = diag[j] * x[j];

    dxnorm = SLC_DNRM2(&n, dwork, &inc_1);
    fp = dxnorm - delta;

    if (fp > p1 * delta) {
        lds = (st > 0 ? st : 1);
        jw = n2 + st * nths;

        if (ucond) {
            if (ldwork >= jw + 2 * ((bsn > st) ? bsn : st)) {
                condl[0] = 'E';
                toldef = (f64)n * SLC_DLAMCH("Epsilon");
            } else {
                condl[0] = 'N';
                toldef = tol;
            }
        } else {
            rank = 0;
            for (k = 0; k < bn; k++) rank += ranks[k];
            if (st > 0) rank += ranks[bn];
            condl[0] = cond[0];
            toldef = tol;
        }

        if (rank == n) {
            for (j = 0; j < n; j++) {
                l = ipvt[j] - 1;
                if (l < 0 || l >= n) {
                    *info = -7;
                    return;
                }
                rx[j] = diag[l] * (dwork[l] / dxnorm);
            }

            nf01br("Use ranks", "Upper", "Transpose", n, ipar, lipar, r, ldr, dwork, dwork, 1, rx, ranks, tol, dwork, ldwork, info);
            temp = SLC_DNRM2(&n, rx, &inc_1);
            parl = ((fp / delta) / temp) / temp;

            if ((condl[0] != 'U' && condl[0] != 'u') && dmino > zero) condl[0] = 'U';
        } else {
            parl = zero;
        }

        ibsn = 0;
        k = 0;
        for (j = 0; j < n; j++) {
            ibsn++;
            if (j < nths) {
                sum = SLC_DDOT(&ibsn, &r[k + (ibsn - 1) * ldr], &inc_1, &qtb[k], &inc_1);
                if (ibsn == bsn) {
                    ibsn = 0;
                    k += bsn;
                }
            } else if (j == nths) {
                sum = SLC_DDOT(&ibsn, &r[k + (ibsn - 1) * ldr], &inc_1, &qtb[k], &inc_1);
            } else {
                i32 len_jp1 = j + 1;
                sum = SLC_DDOT(&len_jp1, &r[0 + (ibsn - 1) * ldr], &inc_1, qtb, &inc_1);
            }
            l = ipvt[j] - 1;
            if (l < 0 || l >= n) {
                *info = -7;
                return;
            }
            rx[j] = sum / diag[l];
        }

        gnorm = SLC_DNRM2(&n, rx, &inc_1);
        paru = gnorm / delta;
        if (paru == zero) paru = dwarf / ((delta < p1) ? delta : p1) / p001;

        *par = (*par > parl) ? *par : parl;
        *par = (*par < paru) ? *par : paru;
        if (*par == zero) *par = gnorm / dxnorm;

        while (1) {
            iter++;

            if (*par == zero) *par = ((dwarf > p001 * paru) ? dwarf : p001 * paru);
            temp = sqrt(*par);

            for (j = 0; j < n; j++) rx[j] = temp * diag[j];

            nf01bq(condl, n, ipar, lipar, r, ldr, ipvt, rx, qtb, ranks, x, &toldef, dwork, ldwork, info);

            for (j = 0; j < n; j++) dwork[n + j] = diag[j] * x[j];

            dxnorm = SLC_DNRM2(&n, &dwork[n], &inc_1);
            temp = fp;
            fp = dxnorm - delta;

            if (!(fabs(fp) > p1 * delta && (parl != zero || fp > temp || temp >= zero) && iter < itmax)) break;

            for (j = 0; j < n; j++) {
                l = ipvt[j] - 1;
                if (l < 0 || l >= n) {
                    *info = -7;
                    return;
                }
                rx[j] = diag[l] * (dwork[n + l] / dxnorm);
            }

            nf01br("Use ranks", "Lower", "Transpose", n, ipar, lipar, r, ldr, dwork, &dwork[n2], lds, rx, ranks, tol, &dwork[jw], ldwork - jw, info);
            temp = SLC_DNRM2(&n, rx, &inc_1);
            parc = ((fp / delta) / temp) / temp;

            if (fp > zero) parl = (parl > *par) ? parl : *par;
            else if (fp < zero) paru = (paru < *par) ? paru : *par;

            *par = (parl > *par + parc) ? parl : *par + parc;
        }
    }

    for (j = 0; j < n; j++) {
        l = ipvt[j] - 1;
        if (l < 0 || l >= n) {
            *info = -7;
            return;
        }
        rx[j] = -x[l];
    }

    for (i = 0; i < nths; i += bsn) {
        SLC_DTRMV("Upper", "NoTranspose", "NonUnit", &bsn, &r[i], &ldr, &rx[i], &inc_1);
    }

    if (st > 0) {
        SLC_DGEMV("NoTranspose", &nths, &st, &one, &r[0 + bsn * ldr], &ldr, &rx[nths], &inc_1, &one, rx, &inc_1);
        SLC_DTRMV("Upper", "NoTranspose", "NonUnit", &st, &r[nths + bsn * ldr], &ldr, &rx[nths], &inc_1);
    }

    if (iter == 0) {
        *par = zero;
        i = 0;
        for (k = 0; k < bn; k++) {
            for (j = 0; j < bsn; j++) {
                dwork[i] = r[i + j * ldr];
                i32 len = bsn - j;
                SLC_DCOPY(&len, &r[i + j * ldr], &ldr, &r[i + j * ldr], &inc_1);
                i++;
            }
        }

        if (st > 0) {
            for (j = bsn; j < bsn + st; j++) {
                SLC_DCOPY(&nths, &r[0 + j * ldr], &inc_1, &dwork[n + j - bsn], &st);
                dwork[i] = r[i + j * ldr];
                i32 len = bsn + st - j;
                SLC_DCOPY(&len, &r[i + j * ldr], &ldr, &r[i + j * ldr], &inc_1);
                i++;
            }
        }
    } else {
        for (k = n; k < n + st * nths; k++) dwork[k] = dwork[k + n];
    }
}
