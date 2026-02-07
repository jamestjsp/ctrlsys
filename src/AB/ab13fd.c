/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>
#include <math.h>

void ab13fd(
    const i32 n,
    const f64* a,
    const i32 lda,
    f64* beta,
    f64* omega,
    const f64 tol,
    f64* dwork,
    const i32 ldwork,
    c128* cwork,
    const i32 lcwork,
    i32* info
)
{
    const i32 MAXIT = 50;
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const c128 CONE = 1.0 + 0.0 * I;

    *info = 0;
    i32 minwrk = 3 * n * (n + 2);

    if (n < 0) {
        *info = -1;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -3;
    } else if (ldwork < (minwrk > 1 ? minwrk : 1)) {
        *info = -8;
    } else if (lcwork < (n * (n + 3) > 1 ? n * (n + 3) : 1)) {
        *info = -10;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("AB13FD", &neginfo);
        return;
    }

    *omega = ZERO;
    if (n == 0) {
        *beta = ZERO;
        dwork[0] = ONE;
        cwork[0] = CONE;
        return;
    }

    i32 n2 = n * n;
    i32 igf = 0;
    i32 ia2 = igf + n2 + n;
    i32 iaa = ia2 + n2;
    i32 iwk = iaa + n2;
    i32 iwr = iaa;
    i32 iwi = iwr + n;

    bool sufwrk = (ldwork - iwk >= n2);

    f64 sfmn = SLC_DLAMCH("Safe minimum");
    f64 eps = SLC_DLAMCH("Epsilon");
    f64 tol1 = sqrt(eps * (f64)(2 * n)) *
               SLC_DLANGE("Frobenius", &n, &n, a, &lda, dwork);
    f64 tau = ONE + (tol > eps ? tol : eps);

    i32 kom = 2;
    f64 low = ZERO;

    SLC_DLACPY("All", &n, &n, a, &lda, &dwork[igf], &n);
    i32 ldwork_rem = ldwork - ia2;
    *beta = mb03ny(n, *omega, &dwork[igf], n, &dwork[igf + n2],
                   &dwork[ia2], ldwork_rem, cwork, lcwork, info);
    if (*info != 0) {
        return;
    }

    i32 lbest = minwrk;
    i32 opt_from_mb03ny = (i32)dwork[ia2] - ia2 + 1;
    if (opt_from_mb03ny > lbest) lbest = opt_from_mb03ny;
    if (4 * n2 + n > lbest) lbest = 4 * n2 + n;

    i32 itnum = 1;

    while (itnum <= MAXIT && *beta > tau * (tol1 > low ? tol1 : low)) {
        f64 sigma;
        if (kom == 2) {
            sigma = *beta / tau;
        } else {
            sigma = sqrt(*beta) * sqrt(tol1 > low ? tol1 : low);
        }

        SLC_DLACPY("Full", &n, &n, a, &lda, &dwork[iaa], &n);

        dwork[igf] = sigma;
        dwork[igf + n] = -sigma;

        i32 int1 = 1;
        for (i32 i = 1; i < n; i++) {
            dwork[igf + i] = ZERO;
        }

        for (i32 i = igf; i < ia2 - n - 1; i += n + 1) {
            SLC_DCOPY(&n, &dwork[i], &int1, &dwork[i + n + 1], &int1);
            dwork[i + n + 1 + n] = dwork[i + n];
        }

        f64 dummy2[1];
        i32 int1_ld = 1;
        mb04zd("N", n, &dwork[iaa], n, &dwork[igf], n, dummy2, 1,
               &dwork[iwk], info);

        i32 jwork = ia2;
        if (sufwrk) {
            jwork = iwk;
        }

        SLC_DLACPY("Lower", &n, &n, &dwork[igf], &n, &dwork[jwork], &n);
        ma02ed('L', n, &dwork[jwork], n);

        if (sufwrk) {
            SLC_DSYMM("Left", "Upper", &n, &n, &ONE, &dwork[igf + n], &n,
                      &dwork[jwork], &n, &ZERO, &dwork[ia2], &n);
        } else {
            for (i32 i = 0; i < n; i++) {
                SLC_DSYMV("Upper", &n, &ONE, &dwork[igf + n], &n,
                          &dwork[ia2 + n * i], &int1, &ZERO, &dwork[iwk], &int1);
                SLC_DCOPY(&n, &dwork[iwk], &int1, &dwork[ia2 + n * i], &int1);
            }
        }

        SLC_DGEMM("N", "N", &n, &n, &n, &ONE, &dwork[iaa], &n,
                  &dwork[iaa], &n, &ONE, &dwork[ia2], &n);

        jwork = iwi + n;
        i32 ilo, ihi;
        SLC_DGEBAL("Scale", &n, &dwork[ia2], &n, &ilo, &ihi, &dwork[jwork], info);

        SLC_DHSEQR("Eigenvalues", "NoSchurVectors", &n, &ilo, &ihi,
                   &dwork[ia2], &n, &dwork[iwr], &dwork[iwi], dummy2, &int1_ld,
                   &dwork[jwork], &n, info);

        if (*info != 0) {
            *info = 2;
            return;
        }

        kom = 0;
        f64 om = 0, om1 = 0, om2 = 0;
        for (i32 i = 0; i < n; i++) {
            f64 temp = fabs(dwork[iwi + i]);
            if (tol1 > sfmn) {
                temp = temp / tol1;
            }
            if (dwork[iwr + i] < ZERO && temp <= tol1) {
                kom++;
                om = sqrt(-dwork[iwr + i]);
                if (kom == 1) om1 = om;
                if (kom == 2) om2 = om;
            }
        }

        if (kom == 0) {
            low = sigma;
        } else {
            if (kom == 2) {
                om = om1 + (om2 - om1) / TWO;
            } else if (kom == 1 && itnum == 1) {
                om = om1 / TWO;
                kom = 2;
            }

            SLC_DLACPY("All", &n, &n, a, &lda, &dwork[igf], &n);
            f64 sv = mb03ny(n, om, &dwork[igf], n, &dwork[igf + n2],
                            &dwork[ia2], ldwork - ia2, cwork, lcwork, info);
            if (*info != 0) {
                return;
            }

            if (*beta > sv) {
                *beta = sv;
                *omega = om;
            } else {
                *info = 1;
                return;
            }
        }
        itnum++;
    }

    if (*beta > tau * (tol1 > low ? tol1 : low)) {
        *info = 1;
        return;
    }

    dwork[0] = (f64)lbest;
}
