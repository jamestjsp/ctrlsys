/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * BB04AD - Benchmark examples for (generalized) discrete-time Lyapunov equations
 *
 * Purpose:
 *   Generate benchmark examples of (generalized) discrete-time Lyapunov equations
 *       A^T X A - E^T X E = Y
 *
 *   In some examples, the right hand side has the form Y = -B^T B
 *   and the solution can be represented as X = U^T U.
 *
 *   E, A, Y, X, and U are real N-by-N matrices, and B is M-by-N.
 *   Note that E can be the identity matrix. For some examples, B, X, or U
 *   are not provided.
 *
 *   This routine is an implementation of the benchmark library
 *   DTLEX (Version 1.0) described in [1].
 *
 * References:
 *   [1] D. Kressner, V. Mehrmann, and T. Penzl.
 *       DTLEX - a Collection of Benchmark Examples for Discrete-Time
 *       Lyapunov Equations. SLICOT Working Note 1999-7, 1999.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <string.h>
#include <ctype.h>

#define ZERO 0.0
#define ONE 1.0
#define TWO 2.0
#define FOUR 4.0

static bool lsame_local(char a, char b) {
    return toupper((unsigned char)a) == toupper((unsigned char)b);
}

void bb04ad(const char* def, const i32* nr, f64* dpar, i32* ipar,
            bool* vec, i32* n, i32* m,
            f64* e, const i32 lde, f64* a, const i32 lda,
            f64* y, const i32 ldy, f64* b, const i32 ldb,
            f64* x, const i32 ldx, f64* u, const i32 ldu,
            char* note, f64* dwork, const i32 ldwork, i32* info)
{
    i32 i, j, k;
    f64 temp, ttemp, twobyn;
    i32 int1 = 1;
    f64 dbl0 = ZERO, dbl1 = ONE;

    static const bool vecdef[8] = {true, true, false, true, true, false, false, false};

    *info = 0;
    for (i = 0; i < 8; i++) {
        vec[i] = vecdef[i];
    }

    if (nr[0] == 4) {
        if (!(lsame_local(def[0], 'D') || lsame_local(def[0], 'N'))) {
            *info = -1;
            return;
        }

        if (nr[1] == 1) {
            strncpy(note, "DTLEX: Example 4.1", 70);
            if (lsame_local(def[0], 'D')) {
                ipar[0] = 10;
                dpar[0] = 1.5;
                dpar[1] = 1.5;
            }
            if ((dpar[0] <= ONE) || (dpar[1] <= ONE)) *info = -3;
            if (ipar[0] < 2) *info = -4;
            *n = ipar[0];
            *m = 1;
            if (lde < *n) *info = -9;
            if (lda < *n) *info = -11;
            if (ldy < *n) *info = -13;
            if (ldb < *m) *info = -15;
            if (ldx < *n) *info = -17;
            if (ldwork < (*n) * 2) *info = -22;
            if (*info != 0) return;

            vec[5] = true;
            vec[6] = true;
            twobyn = TWO / (f64)(*n);
            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            SLC_DLASET("A", n, n, &dbl0, &dbl0, a, &lda);
            SLC_DLASET("A", n, n, &dbl0, &dbl0, y, &ldy);
            f64 neg_twobyn = -twobyn;
            f64 b_diag = ONE - twobyn;
            SLC_DLASET("A", m, n, &neg_twobyn, &b_diag, b, &ldb);
            SLC_DLASET("A", n, n, &dbl0, &dbl0, x, &ldx);

            temp = ONE;
            for (i = 0; i < *n; i++) {
                a[i + i * lda] = (temp - ONE) / (temp + ONE);
                dwork[i] = ONE;
                temp *= dpar[0];
            }

            SLC_DGEMV("T", n, n, &dbl1, a, &lda, dwork, &int1, &dbl0, &dwork[*n], &int1);
            SLC_DGER(n, n, &neg_twobyn, dwork, &int1, &dwork[*n], &int1, a, &lda);

            SLC_DGEMV("N", n, n, &dbl1, a, &lda, dwork, &int1, &dbl0, &dwork[*n], &int1);
            SLC_DGER(n, n, &neg_twobyn, &dwork[*n], &int1, dwork, &int1, a, &lda);

            f64 col_scale = ONE;
            for (j = 0; j < *n; j++) {
                b[0 + j * ldb] = b[0 + j * ldb] * col_scale;
                f64 row_scale = col_scale;
                for (i = 0; i < *n; i++) {
                    a[i + j * lda] = a[i + j * lda] * row_scale;
                    row_scale *= dpar[1];
                }
                dwork[j] = ONE - TWO * (j % 2);
                col_scale /= dpar[1];
            }

            SLC_DGEMV("T", n, n, &dbl1, a, &lda, dwork, &int1, &dbl0, &dwork[*n], &int1);
            SLC_DGER(n, n, &neg_twobyn, dwork, &int1, &dwork[*n], &int1, a, &lda);

            SLC_DGEMV("N", n, n, &dbl1, a, &lda, dwork, &int1, &dbl0, &dwork[*n], &int1);
            SLC_DGER(n, n, &neg_twobyn, &dwork[*n], &int1, dwork, &int1, a, &lda);

            f64 dot = SLC_DDOT(n, b, &ldb, dwork, &int1);
            f64 factor = -twobyn * dot;
            SLC_DAXPY(n, &factor, dwork, &int1, b, &ldb);

            f64 neg_one = -ONE;
            SLC_DGER(n, n, &neg_one, b, &ldb, b, &ldb, y, &ldy);

            for (j = 0; j < *n; j++) {
                SLC_DAXPY(n, &neg_one, &y[0 + j * ldy], &int1, &x[0 + j * ldx], &int1);
            }

        } else if (nr[1] == 2) {
            strncpy(note, "DTLEX: Example 4.2", 70);
            if (lsame_local(def[0], 'D')) {
                ipar[0] = 10;
                dpar[0] = -0.5;
                dpar[1] = 1.5;
            }
            if ((dpar[0] <= -ONE) || (dpar[0] >= ONE) || (dpar[1] <= ONE)) *info = -3;
            if (ipar[0] < 2) *info = -4;
            *n = ipar[0];
            *m = 1;
            if (lde < *n) *info = -9;
            if (lda < *n) *info = -11;
            if (ldy < *n) *info = -13;
            if (ldb < *m) *info = -15;
            if (ldwork < (*n) * 2) *info = -22;
            if (*info != 0) return;

            vec[5] = true;
            twobyn = TWO / (f64)(*n);
            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            f64 lambda = dpar[0];
            SLC_DLASET("A", n, n, &dbl0, &lambda, a, &lda);
            SLC_DLASET("A", n, n, &dbl0, &dbl0, y, &ldy);
            f64 neg_twobyn = -twobyn;
            f64 b_diag = ONE - twobyn;
            SLC_DLASET("A", m, n, &neg_twobyn, &b_diag, b, &ldb);

            for (i = 0; i < *n - 1; i++) {
                dwork[i] = ONE;
                a[i + (i + 1) * lda] = ONE;
            }
            dwork[*n - 1] = ONE;

            SLC_DGEMV("T", n, n, &dbl1, a, &lda, dwork, &int1, &dbl0, &dwork[*n], &int1);
            SLC_DGER(n, n, &neg_twobyn, dwork, &int1, &dwork[*n], &int1, a, &lda);

            SLC_DGEMV("N", n, n, &dbl1, a, &lda, dwork, &int1, &dbl0, &dwork[*n], &int1);
            SLC_DGER(n, n, &neg_twobyn, &dwork[*n], &int1, dwork, &int1, a, &lda);

            f64 col_scale = ONE;
            for (j = 0; j < *n; j++) {
                b[0 + j * ldb] = b[0 + j * ldb] * col_scale;
                f64 row_scale = col_scale;
                for (i = 0; i < *n; i++) {
                    a[i + j * lda] = a[i + j * lda] * row_scale;
                    row_scale *= dpar[1];
                }
                dwork[j] = ONE - TWO * (j % 2);
                col_scale /= dpar[1];
            }

            SLC_DGEMV("T", n, n, &dbl1, a, &lda, dwork, &int1, &dbl0, &dwork[*n], &int1);
            SLC_DGER(n, n, &neg_twobyn, dwork, &int1, &dwork[*n], &int1, a, &lda);

            SLC_DGEMV("N", n, n, &dbl1, a, &lda, dwork, &int1, &dbl0, &dwork[*n], &int1);
            SLC_DGER(n, n, &neg_twobyn, &dwork[*n], &int1, dwork, &int1, a, &lda);

            f64 dot = SLC_DDOT(n, b, &ldb, dwork, &int1);
            f64 factor = -twobyn * dot;
            SLC_DAXPY(n, &factor, dwork, &int1, b, &ldb);

            f64 neg_one = -ONE;
            SLC_DGER(n, n, &neg_one, b, &ldb, b, &ldb, y, &ldy);

        } else if (nr[1] == 3) {
            strncpy(note, "DTLEX: Example 4.3", 70);
            if (lsame_local(def[0], 'D')) {
                ipar[0] = 10;
                dpar[0] = 10.0;
            }
            if (dpar[0] < ZERO) *info = -3;
            if (ipar[0] < 2) *info = -4;
            *n = ipar[0];
            *m = 0;
            if (lde < *n) *info = -9;
            if (lda < *n) *info = -11;
            if (ldy < *n) *info = -13;
            if (ldx < *n) *info = -17;
            if (*info != 0) return;

            vec[2] = true;
            vec[6] = true;
            temp = pow(TWO, -dpar[0]);
            SLC_DLASET("U", n, n, &dbl0, &dbl0, e, &lde);
            SLC_DLASET("L", n, n, &temp, &dbl1, e, &lde);
            SLC_DLASET("L", n, n, &dbl0, &dbl0, a, &lda);
            SLC_DLASET("U", n, n, &dbl1, &dbl0, a, &lda);
            SLC_DLASET("A", n, n, &dbl1, &dbl1, x, &ldx);

            for (i = 0; i < *n; i++) {
                a[i + i * lda] = (f64)(i + 1) + temp;
            }

            for (j = 0; j < *n; j++) {
                for (i = 0; i < *n; i++) {
                    y[i + j * ldy] = temp * temp * (f64)(1 - (*n - 1 - i) * (*n - 1 - j)) +
                                     temp * (f64)(3 * ((i + 1) + (j + 1)) - 2 * (*n + 1)) +
                                     FOUR * (f64)((i + 1) * (j + 1)) - TWO * (f64)((i + 1) + (j + 1));
                }
            }

        } else if (nr[1] == 4) {
            strncpy(note, "DTLEX: Example 4.4", 70);
            if (lsame_local(def[0], 'D')) {
                ipar[0] = 10;
                dpar[0] = 1.5;
            }
            if (dpar[0] < ONE) *info = -3;
            if (ipar[0] < 1) *info = -4;
            *n = ipar[0] * 3;
            *m = 1;
            if (lde < *n) *info = -9;
            if (lda < *n) *info = -11;
            if (ldy < *n) *info = -13;
            if (ldb < *m) *info = -15;
            if (*info != 0) return;

            vec[2] = true;
            vec[5] = true;
            SLC_DLASET("A", n, n, &dbl0, &dbl0, e, &lde);
            SLC_DLASET("A", n, n, &dbl0, &dbl0, a, &lda);

            for (i = 0; i < ipar[0]; i++) {
                ttemp = ONE - ONE / pow(dpar[0], (f64)(i + 1));
                temp = -ttemp / sqrt(TWO);
                for (j = 0; j < i; j++) {
                    for (k = 0; k < 3; k++) {
                        a[(*n - (i + 1) * 3 + 2) + ((j + 1) * 3 - 1 - k) * lda] = ttemp;
                        a[(*n - (i + 1) * 3 + 1) + ((j + 1) * 3 - 1 - k) * lda] = TWO * temp;
                    }
                }
                a[(*n - (i + 1) * 3 + 2) + ((i + 1) * 3 - 3) * lda] = ttemp;
                a[(*n - (i + 1) * 3 + 1) + ((i + 1) * 3 - 3) * lda] = TWO * temp;
                a[(*n - (i + 1) * 3 + 1) + ((i + 1) * 3 - 2) * lda] = TWO * temp;
                a[(*n - (i + 1) * 3 + 1) + ((i + 1) * 3 - 1) * lda] = temp;
                a[(*n - (i + 1) * 3 + 0) + ((i + 1) * 3 - 1) * lda] = temp;
            }

            for (j = 0; j < *n; j++) {
                if (j > 0) {
                    SLC_DAXPY(n, &dbl1, &a[(j - 1) + 0 * lda], &lda, &a[j + 0 * lda], &lda);
                }
                b[0 + j * ldb] = (f64)(j + 1);
                for (i = 0; i < *n; i++) {
                    i32 min_ij = (i + 1) < (j + 1) ? (i + 1) : (j + 1);
                    e[i + (*n - j - 1) * lde] = (f64)min_ij;
                    y[i + j * ldy] = -(f64)((i + 1) * (j + 1));
                }
            }

        } else {
            *info = -2;
        }
    } else {
        *info = -2;
    }
}
