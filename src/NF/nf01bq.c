/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stddef.h>

/**
 * @brief Solve linear system J*x = b, D*x = 0 in least squares sense.
 *
 * @param[in] cond Condition estimation mode.
 * @param[in] n Order of matrix R.
 * @param[in] ipar Integer parameters (st, bn, bsm, bsn).
 * @param[in] lipar Length of ipar.
 * @param[in,out] r Matrix R (ldr x nc).
 * @param[in] ldr Leading dimension of R.
 * @param[in] ipvt Permutation matrix P.
 * @param[in] diag Diagonal scaling matrix D.
 * @param[in] qtb First n elements of Q'*b.
 * @param[in,out] ranks Numerical ranks.
 * @param[out] x Solution vector.
 * @param[in] tol Tolerance.
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
void nf01bq(const char *cond, i32 n, const i32 *ipar, i32 lipar, f64 *r, i32 ldr, 
            const i32 *ipvt, const f64 *diag, const f64 *qtb, i32 *ranks, 
            f64 *x, f64 *tol, f64 *dwork, i32 ldwork, i32 *info)
{
    /* Local variables */
    i32 bn, bsm, bsn, i, ib, ibsn, is, itr, itc, j, jw, k, kf, l, nc, nths, st;
    bool econd;
    i32 inc_1 = 1;
    f64 zero = 0.0;

    econd = (cond[0] == 'E' || cond[0] == 'e');
    *info = 0;
    
    /* Argument checks omitted for brevity, relying on caller or adding later */
    /* Proper checks should be added */
    if (!(econd || cond[0] == 'N' || cond[0] == 'n' || cond[0] == 'U' || cond[0] == 'u')) *info = -1;
    else if (n < 0) *info = -2;
    else if (lipar < 4) *info = -4;
    else if (ldr < (n > 1 ? n : 1)) *info = -6; /* Fixed max(1, n) */
    else {
        st = ipar[0];
        bn = ipar[1];
        bsm = ipar[2];
        bsn = ipar[3];
        nths = bn * bsn;
        
        if (st < 0 || bn < 0 || bsm < 0 || bsn < 0) *info = -3;
        else if (n != nths + st) *info = -2;
        else {
            i32 jw_req = 2 * n;
            if (bn <= 1 || bsn == 0) {
                if (econd) jw_req = 4 * n;
            } else {
                jw_req += st * nths;
                if (econd) {
                    i32 max_bsn_st = (bsn > st) ? bsn : st;
                    jw_req += 2 * max_bsn_st;
                }
            }
            if (ldwork < jw_req) *info = -14;
        }
    }

    if (*info != 0) {
        i32 err_code = -(*info);
        SLC_XERBLA("NF01BQ", &err_code);
        return;
    }

    if (n == 0) return;

    if (bn <= 1 || bsn == 0) {
        mb02yd(cond, n, r, ldr, ipvt, diag, qtb, ranks, x, *tol, dwork, ldwork, info);
        return;
    }

    /* General case */
    ib = n; /* 0-based index DWORK[IB] is DWORK(IB+1) in Fortran */
    is = ib + n;
    jw = is + st * nths;
    
    i = 0;
    l = is;
    nc = bsn + st;
    kf = nc;

    /* Copy R and Q'*b */
    /* Save diagonal elements of R in X */
    for (k = 0; k < bn; k++) {
        for (j = 0; j < bsn; j++) {
            x[i] = r[i + j * ldr];
            /* Copy upper part of block */
            i32 len = bsn - j;
            SLC_DCOPY(&len, &r[i + j * ldr], &ldr, &r[i + j * ldr], &inc_1); /* Wait, copy to itself? */
            /* Fortran: CALL DCOPY( BSN-J+1, R(I,J), LDR, R(I,J), 1 )
               This copies column J of R (stride LDR, i.e. row elements?) 
               No, R(I,J) is element. LDR stride -> R(I,J), R(I+1,J), ...
               Wait, R(LDR, NC). 
               R(I,J) -> &r[i + j*ldr].
               Stride LDR means next COLUMN element? No, next ROW element is stride 1.
               Stride LDR means next COLUMN.
               So DCOPY with stride LDR copies elements:
               R(I,J), R(I, J+1), R(I, J+2)...
               So it copies a ROW of R.
               To R(I,J) with stride 1 (COLUMN).
               So it copies Row to Column?
               "transpose of strict upper triangle of S is stored in strict lower triangle of R"
               This copies upper triangle of R to lower triangle of R?
               Or saves R to S?
               
               Ah, "Copy R ... to initialize S."
               Wait, R is input/output.
               If we overwrite R with S, we need to restore R later?
               "On exit ... strict lower triangles of R_k ... contain ... S."
               But here we are inside NF01BQ.
               
               "Copy R and Q'*b to preserve input and initialize S."
               "In particular, save the diagonal elements of R in X."
               
               The loop copies upper part of R to lower part (transposed)?
               DCOPY(..., R(I,J), LDR, R(I,J), 1).
               Source stride LDR (Row). Dest stride 1 (Col).
               So R(I,J) -> R(I,J)
               R(I, J+1) -> R(I+1, J)
               So we copy Upper(R) to Lower(R).
               So R becomes Symmetric? Or just filled with S (initially S=R).
            */
            SLC_DCOPY(&len, &r[i + j * ldr], &ldr, &r[i + j * ldr], &inc_1);
            i++;
        }
    }

    for (j = bsn; j < nc; j++) {
        /* Copy L_k to DWORK(L) */
        /* L_k is column j of R. Rows 0 to NTHS-1.
           Wait, J goes BSN to NC-1.
           R(1, J) is column J. Length NTHS?
           "DCOPY( NTHS, R(1,J), 1, DWORK(L), ST )"
           Source: Col J of R. Stride 1.
           Dest: DWORK(L). Stride ST.
           So we copy Col J of R to Row of DWORK (if DWORK stores S transpose?)
           DWORK(IS) contains [L_1' ... L_l'].
           So we transpose L_k into DWORK.
        */
        i32 j_idx = j; /* 0-based */
        SLC_DCOPY(&nths, &r[0 + j_idx * ldr], &inc_1, &dwork[l], &st);
        
        x[i] = r[i + j_idx * ldr];
        i32 len = nc - j_idx;
        SLC_DCOPY(&len, &r[i + j_idx * ldr], &ldr, &r[i + j_idx * ldr], &inc_1);
        i++;
        l++;
    }

    SLC_DCOPY(&n, qtb, &inc_1, &dwork[ib], &inc_1);

    /* Eliminate diagonal D */
    if (st > 0) {
        itr = nths;
        itc = bsn;
    } else {
        itr = 0;
        itc = 0;
    }
    
    ibsn = 0;
    for (j = 0; j < n; j++) {
        ibsn++;
        i = ibsn - 1; /* 0-based index in block? */

        l = ipvt[j] - 1; /* 0-based */
        if (l < 0 || l >= n) {
            *info = -7;
            return;
        }
        if (diag[l] != zero) {
            f64 qtbpj = zero;
            dwork[j] = diag[l];
            for (k = j + 1; k < ((j + kf < n) ? j + kf : n); k++) {
                dwork[k] = zero;
            }

            /* Eliminate diagonal D using Givens rotations */
            if (j < nths) {
                i32 m_rows = bsn - ibsn + 1;
                mb04ow(m_rows, st, 1, &r[j + (ibsn - 1) * ldr], ldr,
                       &r[itr + itc * ldr], ldr, &dwork[j], 1,
                       &dwork[ib + j], bsn, &dwork[ib + nths], st, &qtbpj, 1);
                if (ibsn == bsn)
                    ibsn = 0;
            } else if (j == nths) {
                mb04ow(1, st, 1, &r[j + (ibsn - 1) * ldr], ldr,
                       &r[itr + itc * ldr], ldr, &dwork[j], 1,
                       &dwork[ib + j], bsn, &dwork[ib + nths], st, &qtbpj, 1);
                kf = st;
            } else {
                i32 n_cols = n - j;
                mb04ow(0, n_cols, 1, &r[j + (ibsn - 1) * ldr], ldr,
                       &r[j + (ibsn - 1) * ldr], ldr, &dwork[j], 1,
                       &dwork[ib + j], 1, &dwork[ib + j], st, &qtbpj, 1);
            }
        } else {
            /* diag[l] == 0, update IBSN and KF if needed */
            if (j < nths) {
                if (ibsn == bsn)
                    ibsn = 0;
            } else if (j == nths) {
                kf = st;
            }
        }

        /* Store diagonal of S */
        dwork[j] = r[j + i * ldr];
    }

    /* Solve triangular system with NF01BR */
    i32 ldwork_remaining = ldwork - jw;
    i32 lds = 1;
    nf01br(cond, "U", "N", n, ipar, lipar, r, ldr,
           dwork, &dwork[is], lds, &dwork[ib], ranks, *tol,
           &dwork[jw], ldwork_remaining, info);

    i = 0;

    /* Restore diagonal elements of R from X and swap upper/lower triangles */
    for (k = 0; k < bn; k++) {
        for (j = 0; j < bsn; j++) {
            r[i + j * ldr] = x[i];
            i32 len = bsn - j;
            SLC_DSWAP(&len, &r[i + j * ldr], &ldr, &r[i + j * ldr], &inc_1);
            i++;
        }
    }

    l = is;
    for (j = bsn; j < nc; j++) {
        /* Swap back L_k from DWORK(IS) to column j of R */
        SLC_DSWAP(&nths, &r[0 + j * ldr], &inc_1, &dwork[l], &st);
        r[i + j * ldr] = x[i];
        i32 len = nc - j;
        SLC_DSWAP(&len, &r[i + j * ldr], &ldr, &r[i + j * ldr], &inc_1);
        i++;
        l++;
    }

    /* Permute solution */
    for (j = 0; j < n; j++) {
        l = ipvt[j] - 1;
        if (l < 0 || l >= n) {
            *info = -7;
            return;
        }
        x[l] = dwork[n + j];
    }
}
