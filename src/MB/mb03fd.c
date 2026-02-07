/*
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>

void mb03fd(i32 n, f64 prec, f64 *a, i32 lda, f64 *b, i32 ldb,
            f64 *q1, i32 ldq1, f64 *q2, i32 ldq2,
            f64 *dwork, i32 ldwork, i32 *info) {

    const f64 zero = 0.0, one = 1.0;

    *info = 0;

    if (n == 4) {
        f64 as[16], bs[16];
        i32 as_ld = 4, bs_ld = 4;
        i32 idum, ierr, ilo, ihi;
        f64 c_arr[4], r_arr[4];
        i32 bwork[4] = {0, 0, 0, 0};

        SLC_DLACPY("F", &n, &n, a, &lda, as, &as_ld);
        SLC_DLACPY("F", &n, &n, b, &ldb, bs, &bs_ld);

        i32 lwork_dgges = ldwork - 3 * n;
        SLC_DGGES("V", "V", "S", sb02ow, &n, b, &ldb, a, &lda, &idum,
                  dwork, &dwork[n], &dwork[2 * n],
                  q2, &ldq2, q1, &ldq1,
                  &dwork[3 * n], &lwork_dgges, bwork, info);

        if (*info != 0) {
            SLC_DLACPY("F", &n, &n, as, &as_ld, a, &lda);
            SLC_DLACPY("F", &n, &n, bs, &bs_ld, b, &ldb);

            mb04dl("B", n, zero, b, ldb, a, lda, &ilo, &ihi, c_arr, r_arr, dwork, &idum, &ierr);

            SLC_DGGES("V", "V", "S", sb02ow, &n, b, &ldb, a, &lda, &idum,
                      dwork, &dwork[n], &dwork[2 * n],
                      q2, &ldq2, q1, &ldq1,
                      &dwork[3 * n], &lwork_dgges, bwork, &ierr);

            if (ierr != 0) {
                if (*info >= 1 && *info <= 4) {
                    *info = 1;
                } else {
                    *info = 2;
                }
                return;
            }

            SLC_DGGBAK("B", "R", &n, &ilo, &ihi, c_arr, r_arr, &n, q1, &ldq1, info);
            SLC_DGGBAK("B", "L", &n, &ilo, &ihi, c_arr, r_arr, &n, q2, &ldq2, info);
        }
    } else {
        f64 a11, a22, b21, b12;
        f64 safmin, scala, scalb;
        f64 co, si, tmp;

        a11 = fabs(a[0]);
        a22 = fabs(a[1 + lda]);
        b21 = fabs(b[1]);
        b12 = fabs(b[ldb]);

        safmin = SLC_DLAMCH("S");
        scala = one / fmax(fmax(a11, a22), safmin);
        scalb = one / fmax(fmax(b12, b21), safmin);

        a11 = scala * a11;
        a22 = scala * a22;
        b21 = scalb * b21;
        b12 = scalb * b12;

        if (a11 <= prec) {
            q1[0] = one;
            q1[1] = zero;
            q1[ldq1] = zero;
            q1[1 + ldq1] = one;
            q2[0] = zero;
            q2[1] = one;
            q2[ldq2] = one;
            q2[1 + ldq2] = zero;
        } else if (a22 <= prec) {
            q1[0] = zero;
            q1[1] = one;
            q1[ldq1] = one;
            q1[1 + ldq1] = zero;
            q2[0] = one;
            q2[1] = zero;
            q2[ldq2] = zero;
            q2[1 + ldq2] = one;
        } else if (b21 <= prec) {
            q1[0] = one;
            q1[1] = zero;
            q1[ldq1] = zero;
            q1[1 + ldq1] = one;
            q2[0] = one;
            q2[1] = zero;
            q2[ldq2] = zero;
            q2[1 + ldq2] = one;
        } else if (b12 <= prec) {
            q1[0] = zero;
            q1[1] = one;
            q1[ldq1] = one;
            q1[1 + ldq1] = zero;
            q2[0] = zero;
            q2[1] = one;
            q2[ldq2] = one;
            q2[1 + ldq2] = zero;
        } else {
            f64 sign_product = copysign(one, a[0]) * copysign(one, a[1 + lda]) *
                               copysign(one, b[1]) * copysign(one, b[ldb]);
            if (sign_product > zero) {
                f64 sign_aa = copysign(one, a[0] * a[1 + lda]);
                SLC_DLARTG(&(f64){sign_aa * sqrt(a22 * b12)}, &(f64){sqrt(a11 * b21)}, &co, &si, &tmp);
                q1[0] = co;
                q1[1] = -si;
                q1[ldq1] = si;
                q1[1 + ldq1] = co;

                SLC_DLARTG(&(f64){sqrt(a11 * b12)}, &(f64){sqrt(a22 * b21)}, &co, &si, &tmp);
                q2[0] = co;
                q2[1] = -si;
                q2[ldq2] = si;
                q2[1 + ldq2] = co;
            } else {
                q1[0] = one;
                q1[1] = zero;
                q1[ldq1] = zero;
                q1[1 + ldq1] = one;
                q2[0] = one;
                q2[1] = zero;
                q2[ldq2] = zero;
                q2[1 + ldq2] = one;
            }
        }
    }
}
