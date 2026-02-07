/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * NF01BA - FCN routine for optimizing the parameters of the nonlinear part
 *          of a Wiener system (initialization phase), using MD03AD.
 *
 * This is the FCN routine for the nonlinear part initialization.
 * It is called for each output of the Wiener system.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdio.h>

void nf01ba(i32 *iflag, i32 nsmp, i32 n, i32 *ipar, i32 lipar,
            const f64 *z, i32 ldz, const f64 *y, i32 ldy, f64 *x,
            i32 *nfevl, f64 *e, f64 *j, i32 *ldj, f64 *jte,
            f64 *dwork, i32 ldwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 MINUS_ONE = -1.0;
    i32 int1 = 1;

    *info = 0;

    if (*iflag == 1) {
        /* Call NF01AY to compute the output y of the Wiener system (in E)
         * and then the error functions (also in E). The array Z must
         * contain the output of the linear part of the Wiener system, and
         * Y must contain the original output Y of the Wiener system.
         * IPAR[1] must contain the number of outputs.
         * Workspace: need 2*NN, NN = IPAR[2] (number of neurons). */
        i32 nz = ipar[1];  /* number of outputs (1 for single output call) */
        i32 nn = ipar[2];  /* number of neurons */
        i32 lipar_local = lipar - 2;

        nf01ay(nsmp, nz, 1, &ipar[2], lipar_local, x, n, z, ldz,
               e, nsmp, dwork, ldwork, info);

        /* e = e - y (compute error: predicted - actual) */
        SLC_DAXPY(&nsmp, &MINUS_ONE, y, &int1, e, &int1);
        dwork[0] = (f64)(2 * nn);

    } else if (*iflag == 2) {
        /* Call NF01BY to compute the Jacobian in a compressed form.
         * IPAR[1], IPAR[2] must have the same content as for IFLAG = 1.
         * Workspace: need 0. */
        i32 nz = ipar[1];
        i32 lipar_local = lipar - 2;

        nf01by("C", nsmp, nz, 1, &ipar[2], lipar_local, x, n, z, ldz,
               e, j, *ldj, jte, dwork, ldwork, info);
        *nfevl = 0;
        dwork[0] = ZERO;

    } else if (*iflag == 3) {
        /* Set the parameter LDJ, the length of the array J, and the sizes
         * of the workspace for FCN (IFLAG = 1 or 2), and JPJ. */
        *ldj = nsmp;
        ipar[0] = nsmp * n;           /* length of J */
        ipar[1] = 2 * ipar[2];        /* workspace for FCN */
        ipar[2] = 0;                  /* workspace for JPJ */
        ipar[3] = nsmp;               /* for packed storage */

    } else if (*iflag == 0) {
        /* Special call for printing intermediate results. */
        f64 err = SLC_DNRM2(&nsmp, e, &int1);
        printf(" Norm of current error = %15.6e\n", err);
    }
}
