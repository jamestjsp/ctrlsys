/**
 * @file mb04rb.c
 * @brief Reduction of skew-Hamiltonian matrix to PVL form (blocked).
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb04rb(
    const i32 n,
    const i32 ilo,
    f64* a, const i32 lda,
    f64* qg, const i32 ldqg,
    f64* cs,
    f64* tau,
    f64* dwork, const i32 ldwork,
    i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    i32 i, ierr, minwrk, nb = 1, wrkopt = 1;
    bool lquery;

    *info = 0;
    minwrk = (n > 1) ? n - 1 : 1;

    if (n < 0) {
        *info = -1;
    } else if (ilo < 1 || ilo > n + 1) {
        *info = -2;
    } else if (lda < ((n > 1) ? n : 1)) {
        *info = -4;
    } else if (ldqg < ((n > 1) ? n : 1)) {
        *info = -6;
    } else {
        lquery = (ldwork == -1);
        if (ldwork < minwrk && !lquery) {
            dwork[0] = (f64)minwrk;
            *info = -10;
        } else {
            if (n <= ilo) {
                wrkopt = 1;
            } else {
                i32 lwork_query = -1;
                SLC_DGEHRD(&n, &(i32){1}, &n, dwork, &n, dwork, dwork, &lwork_query, &ierr);
                wrkopt = (minwrk > (i32)dwork[0]) ? minwrk : (i32)dwork[0];
                nb = wrkopt / n;
                if (nb > n) nb = n;
                i32 opt_blocked = 8 * n * nb + 3 * nb;
                wrkopt = (wrkopt > opt_blocked) ? wrkopt : opt_blocked;
            }
            if (lquery) {
                dwork[0] = (f64)wrkopt;
                return;
            }
        }
    }

    if (*info != 0) {
        return;
    }

    for (i = 0; i < ilo - 1; i++) {
        tau[i] = ZERO;
        cs[2 * i] = ONE;
        cs[2 * i + 1] = ZERO;
    }

    if (n <= ilo) {
        dwork[0] = ONE;
        return;
    }

    i32 nh = n - ilo + 1;
    i32 nbmin = 2;
    i32 nx = 0;

    if (nb > 1 && nb < nh) {
        nx = nb;
        i32 ue_nx = ue01md(3, "MB04RB", " ", n, ilo, -1);
        if (ue_nx > nx) nx = ue_nx;

        if (nx < nh) {
            if (ldwork < wrkopt) {
                nbmin = ue01md(2, "MB04RB", " ", n, ilo, -1);
                if (nbmin < 2) nbmin = 2;
                nb = ldwork / (8 * n + 3);
            }
        }
    }

    /*
     * i tracks current column position using Fortran 1-based convention.
     * Fortran DWORK(PXA_f + expr) → C dwork[pxa + expr] since pxa = PXA_f - 1.
     * Fortran A(r,c) → C a[(r-1) + (c-1)*lda].
     * Fortran QG(r,c) → C qg[(r-1) + (c-1)*ldqg].
     */
    if (nb < nbmin || nb >= nh) {
        i = ilo;
    } else {
        i32 nnb = n * nb;
        i32 pxa = 0;
        i32 pya = pxa + 2 * nnb;
        i32 pxq = pya + 2 * nnb;
        i32 pxg = pxq + 2 * nnb;
        i32 pdw = pxg + 2 * nnb;

        for (i = ilo; i <= n - nx - 1; i += nb) {
            i32 ib = nb;
            if (n - i < ib) ib = n - i;
            i32 nib = n * ib;

            // MB04PA(.FALSE., N-I+1, I-1, IB, A(1,I), LDA, QG(1,I), LDQG,
            //         DWORK(PXA), N, DWORK(PXG), N, DWORK(PXQ), N,
            //         DWORK(PYA), N, CS(2*I-1), TAU(I), DWORK(PDW))
            mb04pa(false, n - i + 1, i - 1, ib,
                   &a[(i - 1) * lda], lda,
                   &qg[(i - 1) * ldqg], ldqg,
                   &dwork[pxa], n, &dwork[pxg], n,
                   &dwork[pxq], n, &dwork[pya], n,
                   &cs[2 * (i - 1)], &tau[i - 1],
                   &dwork[pdw]);

            if (n > i + ib) {
                i32 nrib = n - i - ib;
                i32 iib = i + ib;

                // Update A(1:n, i+ib+1:n):
                // DGEMM('N','T', N-I-IB, N-I-IB, IB, 1, QG(I+IB+1,I), LDQG,
                //        DWORK(PXA+IB+1), N, 1, A(I+IB+1,I+IB+1), LDA)
                SLC_DGEMM("N", "T", &nrib, &nrib, &ib, &ONE,
                          &qg[iib + (i - 1) * ldqg], &ldqg,
                          &dwork[pxa + ib + 1], &n,
                          &ONE, &a[iib + iib * lda], &lda);

                // DGEMM('N','T', N-I-IB, N-I-IB, IB, 1, A(I+IB+1,I), LDA,
                //        DWORK(PXA+NIB+IB+1), N, 1, A(I+IB+1,I+IB+1), LDA)
                SLC_DGEMM("N", "T", &nrib, &nrib, &ib, &ONE,
                          &a[iib + (i - 1) * lda], &lda,
                          &dwork[pxa + nib + ib + 1], &n,
                          &ONE, &a[iib + iib * lda], &lda);

                // DGEMM('N','T', N, N-I-IB, IB, 1, DWORK(PYA), N,
                //        QG(I+IB+1,I), LDQG, 1, A(1,I+IB+1), LDA)
                SLC_DGEMM("N", "T", &n, &nrib, &ib, &ONE,
                          &dwork[pya], &n,
                          &qg[iib + (i - 1) * ldqg], &ldqg,
                          &ONE, &a[iib * lda], &lda);

                // DGEMM('N','T', N, N-I-IB, IB, 1, DWORK(PYA+NIB), N,
                //        A(I+IB+1,I), LDA, 1, A(1,I+IB+1), LDA)
                SLC_DGEMM("N", "T", &n, &nrib, &ib, &ONE,
                          &dwork[pya + nib], &n,
                          &a[iib + (i - 1) * lda], &lda,
                          &ONE, &a[iib * lda], &lda);

                // Update Q(i+ib+1:n, i+ib+1:n):
                // MB01KD('L','N', N-I-IB, IB, 1, DWORK(PXQ+IB+1), N,
                //         QG(I+IB+1,I), LDQG, 1, QG(I+IB+1,I+IB+1), LDQG, IERR)
                mb01kd("L", "N", nrib, ib, ONE,
                       &dwork[pxq + ib + 1], n,
                       &qg[iib + (i - 1) * ldqg], ldqg,
                       ONE, &qg[iib + iib * ldqg], ldqg, &ierr);

                // MB01KD('L','N', N-I-IB, IB, 1, DWORK(PXQ+NIB+IB+1), N,
                //         A(I+IB+1,I), LDA, 1, QG(I+IB+1,I+IB+1), LDQG, IERR)
                mb01kd("L", "N", nrib, ib, ONE,
                       &dwork[pxq + nib + ib + 1], n,
                       &a[iib + (i - 1) * lda], lda,
                       ONE, &qg[iib + iib * ldqg], ldqg, &ierr);

                // Update G(1:n, 1:n):
                // DGEMM('N','T', I+IB, N-I-IB, IB, 1, DWORK(PXG), N,
                //        QG(I+IB+1,I), LDQG, 1, QG(1,I+IB+2), LDQG)
                SLC_DGEMM("N", "T", &iib, &nrib, &ib, &ONE,
                          &dwork[pxg], &n,
                          &qg[iib + (i - 1) * ldqg], &ldqg,
                          &ONE, &qg[(iib + 1) * ldqg], &ldqg);

                // DGEMM('N','T', I+IB, N-I-IB, IB, 1, DWORK(PXG+NIB), N,
                //        A(I+IB+1,I), LDA, 1, QG(1,I+IB+2), LDQG)
                SLC_DGEMM("N", "T", &iib, &nrib, &ib, &ONE,
                          &dwork[pxg + nib], &n,
                          &a[iib + (i - 1) * lda], &lda,
                          &ONE, &qg[(iib + 1) * ldqg], &ldqg);

                // MB01KD('U','N', N-I-IB, IB, 1, DWORK(PXG+IB+I), N,
                //         QG(I+IB+1,I), LDQG, 1, QG(I+IB+1,I+IB+2), LDQG, IERR)
                mb01kd("U", "N", nrib, ib, ONE,
                       &dwork[pxg + ib + i], n,
                       &qg[iib + (i - 1) * ldqg], ldqg,
                       ONE, &qg[iib + (iib + 1) * ldqg], ldqg, &ierr);

                // MB01KD('U','N', N-I-IB, IB, 1, DWORK(PXG+NIB+IB+I), N,
                //         A(I+IB+1,I), LDA, 1, QG(I+IB+1,I+IB+2), LDQG, IERR)
                mb01kd("U", "N", nrib, ib, ONE,
                       &dwork[pxg + nib + ib + i], n,
                       &a[iib + (i - 1) * lda], lda,
                       ONE, &qg[iib + (iib + 1) * ldqg], ldqg, &ierr);
            }
        }
    }

    mb04ru(n, i, a, lda, qg, ldqg, cs, tau, dwork, ldwork, &ierr);

    dwork[0] = (f64)wrkopt;
}
