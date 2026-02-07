/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * NF01BB - FCN routine for optimizing all parameters of a Wiener system.
 *
 * This is the FCN routine for optimizing all parameters (linear and nonlinear)
 * of a Wiener system using MD03AD Levenberg-Marquardt optimizer.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdio.h>

void nf01bb(i32 *iflag, i32 nfun, i32 lx, i32 *ipar, i32 lipar,
            const f64 *u, i32 ldu, const f64 *y, i32 ldy, f64 *x,
            i32 *nfevl, f64 *e, f64 *j, i32 *ldj, f64 *jte,
            f64 *dwork, i32 ldwork, i32 *info)
{
    const f64 MINUS_ONE = -1.0;
    i32 int1 = 1;

    i32 l = ipar[1];
    i32 m = ipar[4];
    i32 n = ipar[5];
    i32 nn = ipar[6];
    i32 nsmp = (l == 0) ? nfun : nfun / l;

    *info = 0;

    if (*iflag == 1) {
        /* Compute output of Wiener system and error functions.
         * U = input to linear part
         * Y = actual output
         * E = predicted - actual (error)
         * Workspace: NFUN + MAX(2*NN, (N+L)*(N+M) + 2*N + MAX(N*(N+L), N+M+L)) */

        nf01ad(nsmp, m, l, &ipar[5], lipar - 2, x, lx, (f64*)u, ldu,
               e, nsmp, dwork, ldwork, info);

        for (i32 i = 0; i < l; i++) {
            SLC_DAXPY(&nsmp, &MINUS_ONE, &y[i * ldy], &int1, &e[i * nsmp], &int1);
        }

        i32 jwork = (m > 0) ? (n * (n + l) > n + m + l ? n * (n + l) : n + m + l)
                            : (n * (n + l) > l ? n * (n + l) : l);
        i32 t1 = 2 * nn;
        i32 t2 = (n + l) * (n + m) + 2 * n + jwork;
        dwork[0] = (f64)(nfun + (t1 > t2 ? t1 : t2));

    } else if (*iflag == 2) {
        /* Compute Jacobian in compressed form.
         * Workspace: 2*NFUN + MAX(2*NN, (N+L)*(N+M) + 2*N + MAX(N*(N+L), N+M+L)) */

        nf01bd("C", nsmp, m, l, &ipar[5], lipar - 2, x, lx, (f64*)u, ldu,
               e, j, ldj, jte, dwork, ldwork, info);

        *nfevl = n * (m + l + 1) + l * m;

        i32 jwork = (m > 0) ? (n * (n + l) > n + m + l ? n * (n + l) : n + m + l)
                            : (n * (n + l) > l ? n * (n + l) : l);
        i32 t1 = 2 * nn;
        i32 t2 = (n + l) * (n + m) + 2 * n + jwork;
        dwork[0] = (f64)(2 * nfun + (t1 > t2 ? t1 : t2));

    } else if (*iflag == 3) {
        /* Set parameters for workspace sizing */
        i32 st = ipar[0];
        i32 bsn = ipar[3];

        *ldj = nfun;
        ipar[0] = nfun * (bsn + st);  /* length of J */

        i32 jwork = (m > 0) ? (n * (n + l) > n + m + l ? n * (n + l) : n + m + l)
                            : (n * (n + l) > l ? n * (n + l) : l);
        i32 t1 = 2 * nn;
        i32 t2 = (n + l) * (n + m) + 2 * n + jwork;
        ipar[1] = *ldj + (t1 > t2 ? t1 : t2);  /* workspace for FCN */
        ipar[2] = *ldj + ipar[1];              /* workspace for JPJ */
        ipar[3] = 0;
        ipar[4] = nfun;

    } else if (*iflag == 0) {
        /* Print intermediate results */
        f64 err = SLC_DNRM2(&nfun, e, &int1);
        printf(" Norm of current error = %15.6e\n", err);
    }
}
