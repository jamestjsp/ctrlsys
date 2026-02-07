/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdio.h>

/**
 * @brief Error function for Wiener system identification (FCN for MD03BD).
 *
 * This is the FCN routine for optimizing the parameters of the
 * nonlinear part of a Wiener system (initialization phase), using
 * SLICOT Library routine MD03BD.
 *
 * @param[in,out] iflag Integer indicating the action to be performed.
 * @param[in] nsmp Number of training samples (M in MD03BD).
 * @param[in] n Number of variables (N in MD03BD).
 * @param[in,out] ipar Integer parameters.
 * @param[in] lipar Length of ipar.
 * @param[in] z Input samples (DPAR1 in MD03BD).
 * @param[in] ldz Leading dimension of Z.
 * @param[in] y Output samples (DPAR2 in MD03BD).
 * @param[in] ldy Leading dimension of Y.
 * @param[in] x Current estimate of parameters.
 * @param[out] nfevl Number of function evaluations.
 * @param[out] e Error vector.
 * @param[out] j Jacobian matrix.
 * @param[in,out] ldj Leading dimension of J.
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
void nf01be(i32 *iflag, i32 nsmp, i32 n, i32 *ipar, i32 lipar, 
            f64 *z, i32 ldz, f64 *y, i32 ldy, f64 *x, 
            i32 *nfevl, f64 *e, f64 *j, i32 *ldj, 
            f64 *dwork, i32 ldwork, i32 *info)
{
    f64 zero = 0.0, neg_one = -1.0;
    i32 inc = 1;
    i32 nz, nn;
    f64 err;

    *info = 0;

    if (*iflag == 1) {
        /* Compute output y (in e) and error functions (also in e) */
        /* IPAR(2) -> ipar[1] is NZ (number of inputs to NN) */
        /* IPAR(3) -> ipar[2] starts NN parameters */
        /* NF01AY expects ipar for itself. 
           Fortran: CALL NF01AY( ..., IPAR(3), LIPAR-2, ... )
           So we pass &ipar[2].
        */
        nz = ipar[1];
        /* Check lipar >= 2 + needed by NF01AY? */
        
        nf01ay(nsmp, nz, 1, &ipar[2], lipar - 2, x, n, z, ldz, 
               e, nsmp, dwork, ldwork, info);
               
        /* E = E - Y */
        /* DAXPY( NSMP, -ONE, Y, 1, E, 1 ) */
        /* Y is nsmp x 1 (conceptually). Stride 1? Y is passed as DPAR2. */
        /* Assuming Y is vector of length nsmp. Stride 1. */
        SLC_DAXPY(&nsmp, &neg_one, y, &inc, e, &inc);
        
        /* DWORK(1) = 2*NN. NN is IPAR(3) -> ipar[2]? 
           Wait, NF01AY takes IPAR. 
           In NF01AY, IPAR(1) is NN.
           So ipar[2] should be NN.
        */
        nn = ipar[2];
        dwork[0] = (f64)(2 * nn);
        
    } else if (*iflag == 2) {
        /* Compute Jacobian */
        const char *cjte = "N";
        nz = ipar[1];
        
        nf01by(cjte, nsmp, nz, 1, &ipar[2], lipar - 2, x, n, z, ldz, 
               e, j, *ldj, dwork, dwork, ldwork, info); /* jte is dwork? */
        /* Fortran: CALL NF01BY( ..., J, LDJ, DWORK, DWORK, LDWORK, INFO )
           Arguments of NF01BY: ..., J, LDJ, JTE, DWORK, LDWORK, INFO.
           So JTE is passed as DWORK. And DWORK as DWORK.
           This implies JTE uses start of DWORK. And DWORK (workspace) also starts there?
           NF01BY documentation: "JTE: ... The array JTE is not referenced if CJTE = 'N'".
           So passing DWORK is safe if 'N'.
        */
        
        *nfevl = 0;
        dwork[0] = zero;
        
    } else if (*iflag == 3) {
        /* Initialization */
        *ldj = nsmp;
        nn = ipar[2]; /* Assuming ipar[2] is initialized before call? */
        /* In MD03BD, FCN is called with IFLAG=3 to initialize.
           User must set IPAR before calling MD03BD? Yes.
        */
        
        ipar[0] = nsmp * n; /* M * N */
        /* IPAR(2) = 2*IPAR(3) -> ipar[1] = 2 * ipar[2] ???
           Fortran: IPAR(2) = 2*IPAR(3).
           Wait, earlier IPAR(2) was NZ.
           Why overwrite NZ?
           
           Ah, "Set ... sizes of the workspace for FCN (IFLAG = 1 or 2), QRFACT and LMPARM."
           "IPAR(2) must contain the number of outputs." -> In description of IFLAG=1.
           
           Wait, maybe IPAR definitions change for MD03BD usage?
           MD03BD uses IPAR for workspace sizes.
           MD03BD calls FCN(3) to get sizes.
           
           In NF01BE:
           IPAR(1) = NSMP*N (Size of J?)
           IPAR(2) = 2*IPAR(3).
           IPAR(3) = 0.
           IPAR(4) = 4*N + 1.
           IPAR(5) = 4*N.
           
           This seems to set up IPAR for MD03BD internal use (LWORK requirements).
           But if IPAR(2) was NZ, it is now overwritten.
           This implies NZ must be restored or IPAR(2) is output of IFLAG=3?
           
           MD03BD documentation:
           "IPAR    (input/output) INTEGER array ...
                    On exit, IPAR contains the number of observations..."
           
           MD03BD calls FCN(3) at start.
           So NF01BE sets up IPAR.
           But NF01BE needs NN (IPAR(3)) to calculate workspace.
           
           So User sets IPAR(3) = NN before calling.
           NF01BE reads NN, sets IPAR(2) = workspace size for FCN?
           
           Wait, NF01BE code:
           IPAR(2) = 2*IPAR(3)  <- Workspace for NF01AY/BY? 
           NF01AY needs 2*NN workspace.
           So IPAR(2) in MD03BD context likely means "Workspace size for FCN".
           
           But in IFLAG=1 block:
           "IPAR(2) must contain the number of outputs."
           This contradicts.
           
           Wait, let's look at `NF01BE.f` again.
           IFLAG=1: "IPAR(2) must contain the number of outputs."
           IFLAG=3: "IPAR(2) = 2*IPAR(3)"
           
           If `NF01BE` is used with `MD03BD`, `MD03BD` calls IFLAG=3 FIRST.
           Then `MD03BD` allocates workspace based on IPAR.
           Then `MD03BD` calls IFLAG=1.
           So `IPAR(2)` will be `2*NN` when IFLAG=1 is called.
           
           BUT `NF01AY` is called with `IPAR(2)` as `NZ` (Arg 2).
           `CALL NF01AY( NSMP, IPAR(2), ... )`
           
           So `NZ` must be `2*NN`?
           That seems wrong. `NZ` is input sample length. `NN` is neurons.
           
           Maybe `IPAR(2)` in Fortran code IS NOT `NZ`?
           `CALL NF01AY( NSMP, IPAR(2), 1, IPAR(3), ... )`
           Doc of `NF01AY`: `SUBROUTINE NF01AY( NSMP, NZ, L, IPAR, ... )`.
           So 2nd arg is `NZ`.
           
           So `NF01BE` forces `NZ = 2*NN`?
           Or `IPAR(2)` is used for both?
           
           If `NZ` is fixed to `2*NN`, then it's specific to Wiener system structure?
           "Wiener system: linear state-space + static nonlinearity".
           Linear part output `z` (size `NZ`?) goes to Nonlinear part.
           Nonlinear part `NF01AY` takes `z`.
           
           If `NF01BE` sets `IPAR(2) = 2*IPAR(3)`, then `NZ` becomes `2*NN`.
           This implies the linear part outputs `2*NN` signals?
           
           Let's assume the Fortran code is correct and translate literally.
           If there's a logic issue in SLICOT reference, I might uncover it, but I should follow it.
        */
        
        ipar[0] = nsmp * n;
        ipar[1] = 2 * ipar[2];
        ipar[2] = 0;
        ipar[3] = 4 * n + 1;
        ipar[4] = 4 * n;
        
    } else if (*iflag == 0) {
        /* Print intermediate results */
        err = SLC_DNRM2(&nsmp, e, &inc);
        printf(" Norm of current error = %15.6E\n", err);
    }
}
