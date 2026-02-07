/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * MB04CD - Reduce special block (anti-)diagonal skew-Hamiltonian/Hamiltonian
 *          pencil in factored form to generalized Schur form
 *
 * Purpose:
 *   To compute the transformed matrices A, B and D, using orthogonal
 *   matrices Q1, Q2 and Q3 for a real N-by-N regular pencil
 *
 *                ( A11   0  ) ( B11   0  )     (  0   D12 )
 *   aA*B - bD = a(          )(          ) - b (          ),
 *                (  0   A22 ) (  0   B22 )     ( D21   0  )
 *
 *   where A11, A22, B11, B22 and D12 are upper triangular, D21 is
 *   upper quasi-triangular and the generalized matrix product
 *      -1        -1    -1        -1
 *   A11   D12 B22   A22   D21 B11   is upper quasi-triangular, such
 *   that Q3' A Q2, Q2' B Q1 are upper triangular, Q3' D Q1 is upper
 *   quasi-triangular and the transformed pencil
 *   a(Q3' A B Q1) - b(Q3' D Q1) is in generalized Schur form.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int sb02ow_select(const f64 *alphar, const f64 *alphai, const f64 *beta) {
    const f64 zero = 0.0;
    return (*alphar < zero && *beta > zero) || (*alphar > zero && *beta < zero);
}

void mb04cd(const char *compq1, const char *compq2, const char *compq3,
            i32 n, f64 *a, i32 lda, f64 *b, i32 ldb, f64 *d, i32 ldd,
            f64 *q1, i32 ldq1, f64 *q2, i32 ldq2, f64 *q3, i32 ldq3,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork,
            bool *bwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 HUND2 = 200.0;

    char cq1_upper = (char)toupper((unsigned char)compq1[0]);
    char cq2_upper = (char)toupper((unsigned char)compq2[0]);
    char cq3_upper = (char)toupper((unsigned char)compq3[0]);

    bool liniq1 = (cq1_upper == 'I');
    bool liniq2 = (cq2_upper == 'I');
    bool liniq3 = (cq3_upper == 'I');
    bool lupdq1 = (cq1_upper == 'U');
    bool lupdq2 = (cq2_upper == 'U');
    bool lupdq3 = (cq3_upper == 'U');
    bool lcmpq1 = liniq1 || lupdq1;
    bool lcmpq2 = liniq2 || lupdq2;
    bool lcmpq3 = liniq3 || lupdq3;
    bool lquery = (ldwork == -1);

    i32 m = n / 2;
    i32 mm = m * m;
    i32 minwrk = 12 * mm + (m + 252 > 432 ? m + 252 : 432);

    i32 int1 = 1;
    i32 int0 = 0;

    *info = 0;

    if (cq1_upper != 'N' && !lcmpq1) {
        *info = -1;
    } else if (cq2_upper != 'N' && !lcmpq2) {
        *info = -2;
    } else if (cq3_upper != 'N' && !lcmpq3) {
        *info = -3;
    } else if (n < 0 || (n % 2) != 0) {
        *info = -4;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -6;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -8;
    } else if (ldd < (1 > n ? 1 : n)) {
        *info = -10;
    } else if (ldq1 < (1 > n ? 1 : n)) {
        *info = -12;
    } else if (ldq2 < (1 > n ? 1 : n)) {
        *info = -14;
    } else if (ldq3 < (1 > n ? 1 : n)) {
        *info = -16;
    } else if (liwork < (m + 1 > 48 ? m + 1 : 48)) {
        *info = -18;
    } else if (!lquery && ldwork < minwrk) {
        dwork[0] = (f64)minwrk;
        *info = -20;
    }

    if (*info != 0) {
        return;
    }

    if (n > 0 || lquery) {
        i32 i_val = (4 < n) ? 4 : n;
        sl_int bw[4] = {1, 1, 1, 1};
        i32 idum;
        f64 tmp2, tmp3;

        f64 opt1, opt2, opt3, opt4;
        {
            i32 lwork_query = -1;
            i32 info_query = 0;
            f64 work_query[4];

            SLC_DGGES("V", "V", "S", sb02ow_select, &i_val, a, &lda,
                      b, &ldb, &idum, work_query, work_query, work_query,
                      q1, &i_val, q2, &i_val, work_query, &lwork_query, bw, &info_query);
            opt1 = work_query[0];

            SLC_DGGES("V", "V", "N", sb02ow_select, &i_val, a, &lda,
                      b, &ldb, &idum, work_query, work_query, work_query,
                      q1, &i_val, q2, &i_val, work_query, &lwork_query, bw, &info_query);
            opt2 = work_query[0];

            i32 two = 2;
            SLC_DGGEV("N", "N", &two, a, &lda, b, &ldb,
                      work_query, work_query, work_query, &tmp2, &int1,
                      &tmp2, &int1, work_query, &lwork_query, &info_query);
            opt3 = work_query[0];

            SLC_DTGSEN(&int0, &int1, &int1, bw, &i_val, a, &lda, b, &ldb,
                       work_query, work_query, work_query, q1, &i_val, q2, &i_val,
                       &idum, &tmp2, &tmp2, &tmp3, work_query, &lwork_query,
                       &idum, &int1, &info_query);
            opt4 = work_query[0];
        }

        i32 opt_terms[6] = {
            28 + (i32)opt1,
            4 * m + 8,
            4 * n,
            24 + (i32)opt2,
            6 + (i32)opt3,
            12 + (i32)opt4
        };
        i32 max_opt = opt_terms[0];
        for (i32 i = 1; i < 6; i++) {
            if (opt_terms[i] > max_opt) max_opt = opt_terms[i];
        }
        if (4 * n > max_opt) max_opt = 4 * n;

        i32 optwrk = 96 + max_opt;
        if (minwrk > optwrk) optwrk = minwrk;

        if (lquery) {
            dwork[0] = (f64)optwrk;
            return;
        }
    }

    if (n == 0) {
        dwork[0] = ONE;
        return;
    }

    i32 ia11 = 0;
    i32 id12 = ia11 + mm;
    i32 ib22 = id12 + mm;
    i32 ia22 = ib22 + mm;
    i32 id21 = ia22 + mm;
    i32 ib11 = id21 + mm;
    i32 iv1 = ib11 + mm;
    i32 iv2 = iv1 + mm;
    i32 iv3 = iv2 + mm;
    i32 iv4 = iv3 + mm;
    i32 iv5 = iv4 + mm;
    i32 iv6 = iv5 + mm;
    i32 mp1 = m + 1;

    f64 ulp = SLC_DLAMCH("Precision");
    f64 base = SLC_DLAMCH("Base");
    f64 lgbas = log(base);

    i32 k = 6;
    i32 kschur = 5;
    iwork[2 * k] = -1;
    iwork[2 * k + 1] = 1;
    iwork[2 * k + 2] = -1;
    iwork[2 * k + 3] = -1;
    iwork[2 * k + 4] = 1;
    iwork[2 * k + 5] = -1;

    i32 iwork_out;
    mb03ba(k, kschur, &iwork[2 * k], &iwork_out, iwork, &iwork[k]);

    f64 dum = ZERO;
    for (i32 i = 0; i < mm * k; i++) {
        dwork[i] = ZERO;
    }

    SLC_DLACPY("U", &m, &m, a, &lda, dwork, &m);
    SLC_DLACPY("U", &m, &m, &a[mp1 - 1 + (mp1 - 1) * lda], &lda, &dwork[ia22], &m);
    SLC_DLACPY("U", &m, &m, b, &ldb, &dwork[ib11], &m);
    SLC_DLACPY("U", &m, &m, &b[mp1 - 1 + (mp1 - 1) * ldb], &ldb, &dwork[ib22], &m);
    SLC_DLACPY("U", &m, &m, &d[(mp1 - 1) * ldd], &ldd, &dwork[id12], &m);
    SLC_DLACPY("U", &m, &m, &d[mp1 - 1], &ldd, &dwork[id21], &m);

    if (m > 1) {
        i32 m_minus_1 = m - 1;
        i32 ldd_plus_1 = ldd + 1;
        SLC_DCOPY(&m_minus_1, &d[m + 1], &ldd_plus_1, &dwork[id21 + 1], &mp1);
    }

    i32 j = 0;
    i32 ia = iv1;
    i32 ib_idx = ia + 1;

    while (j < m) {
        if (j < m - 1) {
            if (dwork[id21 + (j + 1) + j * m] == ZERO) {
                ma01bd(base, lgbas, k, &iwork[2 * k], &dwork[j * m + j], mm,
                       &dwork[ia], &dwork[ib_idx], &iwork[3 * k]);
                bwork[j] = (dwork[ia] > ZERO) || (dwork[ib_idx] == ZERO);
                j++;
            } else {
                bwork[j] = true;
                bwork[j + 1] = true;
                j += 2;
            }
        } else {
            ma01bd(base, lgbas, k, &iwork[2 * k], &dwork[mm - 1], mm,
                   &dwork[ia], &dwork[ib_idx], &iwork[3 * k]);
            bwork[j] = (dwork[ia] > ZERO) || (dwork[ib_idx] == ZERO);
            j++;
        }
    }

    j = 0;
    while (j < m && bwork[j]) {
        j++;
    }

    i32 m1, m2, i1, i2, i3, m4;

    if (j != mp1 - 1) {
        i32 iwrk = 2 * iv1;
        i32 ib11_new = 0;
        i32 id21_new = ib11_new + mm;
        i32 ia22_new = id21_new + mm;
        i32 ib22_new = ia22_new + mm;
        i32 id12_new = ib22_new + mm;
        i32 ia11_new = id12_new + mm;

        kschur = 2;

        for (i32 i = 0; i < k; i++) {
            iwork[i] = m;
            iwork[k + i] = 0;
            iwork[3 * k + i] = 1 + i * mm;
        }

        for (i32 i = 0; i < mm * k; i++) {
            dwork[ib11_new + i] = ZERO;
        }

        SLC_DLACPY("U", &m, &m, &d[mp1 - 1], &ldd, &dwork[id21_new], &m);
        SLC_DLACPY("U", &m, &m, &d[(mp1 - 1) * ldd], &ldd, &dwork[id12_new], &m);
        SLC_DLACPY("U", &m, &m, a, &lda, &dwork[ia11_new], &m);
        SLC_DLACPY("U", &m, &m, &a[mp1 - 1 + (mp1 - 1) * lda], &lda, &dwork[ia22_new], &m);
        SLC_DLACPY("U", &m, &m, b, &ldb, &dwork[ib11_new], &m);
        SLC_DLACPY("U", &m, &m, &b[mp1 - 1 + (mp1 - 1) * ldb], &ldb, &dwork[ib22_new], &m);

        if (m > 1) {
            i32 m_minus_1 = m - 1;
            i32 ldd_plus_1 = ldd + 1;
            SLC_DCOPY(&m_minus_1, &d[m + 1], &ldd_plus_1, &dwork[id21_new + 1], &mp1);
        }

        i32 idum_arr[1] = {0};
        i32 ldwork_mb03kd = ldwork - iwrk;
        mb03kd("I", idum_arr, "N", k, m, kschur, iwork, &iwork[k], &iwork[2 * k],
               bwork, dwork, iwork, &iwork[3 * k], &dwork[iv1], iwork, &iwork[3 * k],
               &m1, HUND2, &iwork[4 * k], &dwork[iwrk], ldwork_mb03kd, info);

        if (*info > 0) {
            return;
        }

        m2 = m - m1;
        i1 = m1;
        i2 = i1 + m1;
        i3 = i2 + m2;
        m4 = 2 * m2;

        if (lupdq1) {
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, q1, &ldq1, &dwork[iv2], &m, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLACPY("F", &m, &m, &a[mp1 - 1], &lda, q1, &ldq1);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q1[mp1 - 1], &ldq1, &dwork[iv2], &m, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLACPY("F", &m, &m, &a[mp1 - 1], &lda, &q1[mp1 - 1], &ldq1);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q1[(mp1 - 1) * ldq1], &ldq1, &dwork[iv5], &m, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLACPY("F", &m, &m, &a[mp1 - 1], &lda, &q1[(mp1 - 1) * ldq1], &ldq1);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q1[mp1 - 1 + (mp1 - 1) * ldq1], &ldq1, &dwork[iv5], &m, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLACPY("F", &m, &m, &a[mp1 - 1], &lda, &q1[mp1 - 1 + (mp1 - 1) * ldq1], &ldq1);

            if (m2 > 0) {
                SLC_DLACPY("F", &m, &m, &q1[i1 * ldq1], &ldq1, &a[mp1 - 1], &lda);
                SLC_DLACPY("F", &m, &m1, &a[mp1 - 1 + m2 * lda], &lda, &q1[i1 * ldq1], &ldq1);
                SLC_DLACPY("F", &m, &m2, &a[mp1 - 1], &lda, &q1[i2 * ldq1], &ldq1);
                SLC_DLACPY("F", &m, &m, &q1[mp1 - 1 + i1 * ldq1], &ldq1, &a[mp1 - 1], &lda);
                SLC_DLACPY("F", &m, &m1, &a[mp1 - 1 + m2 * lda], &lda, &q1[mp1 - 1 + i1 * ldq1], &ldq1);
                SLC_DLACPY("F", &m, &m2, &a[mp1 - 1], &lda, &q1[mp1 - 1 + i2 * ldq1], &ldq1);
            }
        }

        if (lupdq2) {
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, q2, &ldq2, &dwork[iv1], &m, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLACPY("F", &m, &m, &a[mp1 - 1], &lda, q2, &ldq2);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q2[mp1 - 1], &ldq2, &dwork[iv1], &m, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLACPY("F", &m, &m, &a[mp1 - 1], &lda, &q2[mp1 - 1], &ldq2);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q2[(mp1 - 1) * ldq2], &ldq2, &dwork[iv4], &m, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLACPY("F", &m, &m, &a[mp1 - 1], &lda, &q2[(mp1 - 1) * ldq2], &ldq2);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q2[mp1 - 1 + (mp1 - 1) * ldq2], &ldq2, &dwork[iv4], &m, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLACPY("F", &m, &m, &a[mp1 - 1], &lda, &q2[mp1 - 1 + (mp1 - 1) * ldq2], &ldq2);

            if (m2 > 0) {
                SLC_DLACPY("F", &m, &m, &q2[i1 * ldq2], &ldq2, &a[mp1 - 1], &lda);
                SLC_DLACPY("F", &m, &m1, &a[mp1 - 1 + m2 * lda], &lda, &q2[i1 * ldq2], &ldq2);
                SLC_DLACPY("F", &m, &m2, &a[mp1 - 1], &lda, &q2[i2 * ldq2], &ldq2);
                SLC_DLACPY("F", &m, &m, &q2[mp1 - 1 + i1 * ldq2], &ldq2, &a[mp1 - 1], &lda);
                SLC_DLACPY("F", &m, &m1, &a[mp1 - 1 + m2 * lda], &lda, &q2[mp1 - 1 + i1 * ldq2], &ldq2);
                SLC_DLACPY("F", &m, &m2, &a[mp1 - 1], &lda, &q2[mp1 - 1 + i2 * ldq2], &ldq2);
            }
        }

        if (lupdq3) {
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, q3, &ldq3, &dwork[iv6], &m, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLACPY("F", &m, &m, &a[mp1 - 1], &lda, q3, &ldq3);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q3[mp1 - 1], &ldq3, &dwork[iv6], &m, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLACPY("F", &m, &m, &a[mp1 - 1], &lda, &q3[mp1 - 1], &ldq3);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q3[(mp1 - 1) * ldq3], &ldq3, &dwork[iv3], &m, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLACPY("F", &m, &m, &a[mp1 - 1], &lda, &q3[(mp1 - 1) * ldq3], &ldq3);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q3[mp1 - 1 + (mp1 - 1) * ldq3], &ldq3, &dwork[iv3], &m, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLACPY("F", &m, &m, &a[mp1 - 1], &lda, &q3[mp1 - 1 + (mp1 - 1) * ldq3], &ldq3);

            if (m2 > 0) {
                SLC_DLACPY("F", &m, &m, &q3[i1 * ldq3], &ldq3, &a[mp1 - 1], &lda);
                SLC_DLACPY("F", &m, &m1, &a[mp1 - 1 + m2 * lda], &lda, &q3[i1 * ldq3], &ldq3);
                SLC_DLACPY("F", &m, &m2, &a[mp1 - 1], &lda, &q3[i2 * ldq3], &ldq3);
                SLC_DLACPY("F", &m, &m, &q3[mp1 - 1 + i1 * ldq3], &ldq3, &a[mp1 - 1], &lda);
                SLC_DLACPY("F", &m, &m1, &a[mp1 - 1 + m2 * lda], &lda, &q3[mp1 - 1 + i1 * ldq3], &ldq3);
                SLC_DLACPY("F", &m, &m2, &a[mp1 - 1], &lda, &q3[mp1 - 1 + i2 * ldq3], &ldq3);
            }
        }

        if (m2 > 0) {
            SLC_DLASET("F", &m, &m, &ZERO, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLACPY("U", &m1, &m1, &dwork[ia11_new], &m, a, &lda);
            SLC_DLASET("F", &m1, &m1, &ZERO, &ZERO, &a[i1 * lda], &lda);
            SLC_DLACPY("U", &m1, &m1, &dwork[ia22_new], &m, &a[i1 + i1 * lda], &lda);
            SLC_DLACPY("F", &m1, &m2, &dwork[ia11_new + m * m1], &m, &a[i2 * lda], &lda);
            SLC_DLASET("F", &m1, &m2, &ZERO, &ZERO, &a[i1 + i2 * lda], &lda);
            SLC_DLACPY("U", &m2, &m2, &dwork[ia11_new + m * m1 + m1], &m, &a[i2 + i2 * lda], &lda);
            SLC_DLASET("F", &m1, &m2, &ZERO, &ZERO, &a[i3 * lda], &lda);
            SLC_DLACPY("F", &m1, &m2, &dwork[ia22_new + m * m1], &m, &a[i1 + i3 * lda], &lda);
            SLC_DLASET("F", &m2, &m2, &ZERO, &ZERO, &a[i2 + i3 * lda], &lda);
            SLC_DLACPY("U", &m2, &m2, &dwork[ia22_new + m * m1 + m1], &m, &a[i3 + i3 * lda], &lda);

            SLC_DLASET("F", &m, &m, &ZERO, &ZERO, &b[mp1 - 1], &ldb);
            SLC_DLACPY("U", &m1, &m1, &dwork[ib11_new], &m, b, &ldb);
            SLC_DLASET("F", &m1, &m1, &ZERO, &ZERO, &b[i1 * ldb], &ldb);
            SLC_DLACPY("U", &m1, &m1, &dwork[ib22_new], &m, &b[i1 + i1 * ldb], &ldb);
            SLC_DLACPY("F", &m1, &m2, &dwork[ib11_new + m * m1], &m, &b[i2 * ldb], &ldb);
            SLC_DLASET("F", &m1, &m2, &ZERO, &ZERO, &b[i1 + i2 * ldb], &ldb);
            SLC_DLACPY("U", &m2, &m2, &dwork[ib11_new + m * m1 + m1], &m, &b[i2 + i2 * ldb], &ldb);
            SLC_DLASET("F", &m1, &m2, &ZERO, &ZERO, &b[i3 * ldb], &ldb);
            SLC_DLACPY("F", &m1, &m2, &dwork[ib22_new + m * m1], &m, &b[i1 + i3 * ldb], &ldb);
            SLC_DLASET("F", &m2, &m2, &ZERO, &ZERO, &b[i2 + i3 * ldb], &ldb);
            SLC_DLACPY("U", &m2, &m2, &dwork[ib22_new + m * m1 + m1], &m, &b[i3 + i3 * ldb], &ldb);

            SLC_DLASET("F", &m1, &m1, &ZERO, &ZERO, d, &ldd);
            SLC_DLACPY("U", &m1, &m1, &dwork[id21_new], &m, &d[i1], &ldd);
            if (m1 > 1) {
                i32 m1_minus_1 = m1 - 1;
                SLC_DCOPY(&m1_minus_1, &dwork[id21_new + 1], &mp1, &d[i1 + 1], &(i32){ldd + 1});
            }
            if (m1 > 2) {
                i32 m1_minus_2 = m1 - 2;
                SLC_DLASET("L", &m1_minus_2, &m1_minus_2, &ZERO, &ZERO, &d[i1 + 2], &ldd);
            }
            SLC_DLASET("F", &m4, &m1, &ZERO, &ZERO, &d[i2], &ldd);
            SLC_DLACPY("U", &m1, &m1, &dwork[id12_new], &m, &d[i1 * ldd], &ldd);
            if (m1 > 1) {
                i32 m1_minus_1 = m1 - 1;
                SLC_DLASET("L", &m1_minus_1, &m1_minus_1, &ZERO, &ZERO, &d[1 + i1 * ldd], &ldd);
            }
            i32 n_minus_m1 = n - m1;
            SLC_DLASET("F", &n_minus_m1, &m1, &ZERO, &ZERO, &d[i1 + i1 * ldd], &ldd);
            SLC_DLASET("F", &m1, &m2, &ZERO, &ZERO, &d[i2 * ldd], &ldd);
            SLC_DLACPY("F", &m1, &m2, &dwork[id21_new + m * m1], &m, &d[i1 + i2 * ldd], &ldd);
            SLC_DLASET("F", &m2, &m2, &ZERO, &ZERO, &d[i2 + i2 * ldd], &ldd);
            SLC_DLACPY("U", &m2, &m2, &dwork[id21_new + m * m1 + m1], &m, &d[i3 + i2 * ldd], &ldd);
            if (i3 < n - 1) {
                i32 m2_minus_1 = m2 - 1;
                SLC_DCOPY(&m2_minus_1, &dwork[id21_new + m * m1 + i1], &mp1, &d[i3 + 1 + i2 * ldd], &(i32){ldd + 1});
            }
            if (m2 > 2) {
                i32 m2_minus_2 = m2 - 2;
                SLC_DLASET("L", &m2_minus_2, &m2_minus_2, &ZERO, &ZERO, &d[i3 + 2 + i2 * ldd], &ldd);
            }
            SLC_DLACPY("F", &m1, &m2, &dwork[id12_new + m * m1], &m, &d[i3 * ldd], &ldd);
            SLC_DLASET("F", &m1, &m2, &ZERO, &ZERO, &d[i1 + i3 * ldd], &ldd);
            SLC_DLACPY("F", &m2, &m2, &dwork[id12_new + m * m1 + m1], &m, &d[i2 + i3 * ldd], &ldd);
            SLC_DLASET("F", &m2, &m2, &ZERO, &ZERO, &d[i3 + i3 * ldd], &ldd);
        } else {
            SLC_DLASET("F", &m, &m, &ZERO, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLASET("F", &m, &m, &ZERO, &ZERO, &a[(mp1 - 1) * lda], &lda);
            SLC_DLASET("F", &m, &m, &ZERO, &ZERO, &b[mp1 - 1], &ldb);
            SLC_DLASET("F", &m, &m, &ZERO, &ZERO, &b[(mp1 - 1) * ldb], &ldb);
            SLC_DLASET("F", &m, &m, &ZERO, &ZERO, d, &ldd);
            SLC_DLASET("F", &m, &m, &ZERO, &ZERO, &d[mp1 - 1 + (mp1 - 1) * ldd], &ldd);
        }

        if (liniq1) {
            SLC_DLACPY("F", &m, &m1, &dwork[iv2], &m, q1, &ldq1);
            SLC_DLASET("F", &m, &m1, &ZERO, &ZERO, &q1[mp1 - 1], &ldq1);
            SLC_DLASET("F", &m, &m1, &ZERO, &ZERO, &q1[i1 * ldq1], &ldq1);
            SLC_DLACPY("F", &m, &m1, &dwork[iv5], &m, &q1[mp1 - 1 + i1 * ldq1], &ldq1);
            if (m2 > 0) {
                SLC_DLACPY("F", &m, &m2, &dwork[iv2 + m * m1], &m, &q1[i2 * ldq1], &ldq1);
                SLC_DLASET("F", &m, &m2, &ZERO, &ZERO, &q1[mp1 - 1 + i2 * ldq1], &ldq1);
                SLC_DLASET("F", &m, &m2, &ZERO, &ZERO, &q1[i3 * ldq1], &ldq1);
                SLC_DLACPY("F", &m, &m2, &dwork[iv5 + m * m1], &m, &q1[mp1 - 1 + i3 * ldq1], &ldq1);
            }
        }

        if (liniq2) {
            SLC_DLACPY("F", &m, &m1, &dwork[iv1], &m, q2, &ldq2);
            SLC_DLASET("F", &m, &m1, &ZERO, &ZERO, &q2[mp1 - 1], &ldq2);
            SLC_DLASET("F", &m, &m1, &ZERO, &ZERO, &q2[i1 * ldq2], &ldq2);
            SLC_DLACPY("F", &m, &m1, &dwork[iv4], &m, &q2[mp1 - 1 + i1 * ldq2], &ldq2);
            if (m2 > 0) {
                SLC_DLACPY("F", &m, &m2, &dwork[iv1 + m * m1], &m, &q2[i2 * ldq2], &ldq2);
                SLC_DLASET("F", &m, &m2, &ZERO, &ZERO, &q2[mp1 - 1 + i2 * ldq2], &ldq2);
                SLC_DLASET("F", &m, &m2, &ZERO, &ZERO, &q2[i3 * ldq2], &ldq2);
                SLC_DLACPY("F", &m, &m2, &dwork[iv4 + m * m1], &m, &q2[mp1 - 1 + i3 * ldq2], &ldq2);
            }
        }

        if (liniq3) {
            SLC_DLACPY("F", &m, &m1, &dwork[iv6], &m, q3, &ldq3);
            SLC_DLASET("F", &m, &m1, &ZERO, &ZERO, &q3[mp1 - 1], &ldq3);
            SLC_DLASET("F", &m, &m1, &ZERO, &ZERO, &q3[i1 * ldq3], &ldq3);
            SLC_DLACPY("F", &m, &m1, &dwork[iv3], &m, &q3[mp1 - 1 + i1 * ldq3], &ldq3);
            if (m2 > 0) {
                SLC_DLACPY("F", &m, &m2, &dwork[iv6 + m * m1], &m, &q3[i2 * ldq3], &ldq3);
                SLC_DLASET("F", &m, &m2, &ZERO, &ZERO, &q3[mp1 - 1 + i2 * ldq3], &ldq3);
                SLC_DLASET("F", &m, &m2, &ZERO, &ZERO, &q3[i3 * ldq3], &ldq3);
                SLC_DLACPY("F", &m, &m2, &dwork[iv3 + m * m1], &m, &q3[mp1 - 1 + i3 * ldq3], &ldq3);
            }
        }
    } else {
        m1 = m;
        m2 = 0;
        i1 = m1;
        i2 = i1 + m1;
        i3 = i2;
        m4 = 0;
        SLC_DLASET("F", &m, &m, &ZERO, &ZERO, &a[mp1 - 1], &lda);
        SLC_DLASET("F", &m, &m, &ZERO, &ZERO, &a[(mp1 - 1) * lda], &lda);
        SLC_DLASET("F", &m, &m, &ZERO, &ZERO, &b[mp1 - 1], &ldb);
        SLC_DLASET("F", &m, &m, &ZERO, &ZERO, &b[(mp1 - 1) * ldb], &ldb);
        SLC_DLASET("F", &m, &m, &ZERO, &ZERO, d, &ldd);
        SLC_DLASET("F", &m, &m, &ZERO, &ZERO, &d[mp1 - 1 + (mp1 - 1) * ldd], &ldd);
        if (liniq1) {
            SLC_DLASET("F", &n, &n, &ZERO, &ONE, q1, &ldq1);
        }
        if (liniq2) {
            SLC_DLASET("F", &n, &n, &ZERO, &ONE, q2, &ldq2);
        }
        if (liniq3) {
            SLC_DLASET("F", &n, &n, &ZERO, &ONE, q3, &ldq3);
        }
    }

    i32 r = 0;
    j = 0;
    while (j < m1) {
        if (j < m1 - 1) {
            r++;
            iwork[r - 1] = j;
            i32 d_idx = m1 + j + 1 + j * ldd;
            if (d[d_idx] == ZERO) {
                j++;
            } else {
                j += 2;
            }
        } else {
            r++;
            iwork[r - 1] = j;
            j++;
        }
    }
    iwork[r] = j;

    for (i32 k_idx = 0; k_idx < r; k_idx++) {
        i32 ib1 = iwork[k_idx];
        i32 ib2 = iwork[k_idx + 1];
        i32 dim1 = ib2 - ib1;
        i32 sdim_local = 2 * dim1;

        i32 iauple = 0;
        i32 ialole = iauple + dim1;
        i32 iaupri = dim1 * sdim_local;
        i32 ialori = iaupri + dim1;
        i32 ibuple = sdim_local * sdim_local;
        i32 iblole = ibuple + dim1;
        i32 ibupri = 3 * dim1 * sdim_local;
        i32 iblori = ibupri + dim1;
        i32 iduple = 2 * sdim_local * sdim_local;
        i32 idlole = iduple + dim1;
        i32 idupri = 5 * dim1 * sdim_local;
        i32 idlori = idupri + dim1;
        i32 i1uple = 3 * sdim_local * sdim_local;
        i32 i1lole = i1uple + dim1;
        i32 i1upri = 7 * dim1 * sdim_local;
        i32 i1lori = i1upri + dim1;
        i32 i2uple = 4 * sdim_local * sdim_local;
        i32 i2lole = i2uple + dim1;
        i32 i2upri = 9 * dim1 * sdim_local;
        i32 i2lori = i2upri + dim1;
        i32 i3uple = 5 * sdim_local * sdim_local;
        i32 i3lole = i3uple + dim1;
        i32 i3upri = 11 * dim1 * sdim_local;
        i32 i3lori = i3upri + dim1;

        if (dim1 == 1) {
            i32 stride1 = (lda + 1) * m1;
            i32 stride2 = (ldb + 1) * m1;
            i32 stride3 = (ldd - 1) * m1;
            SLC_DCOPY(&sdim_local, &a[ib1 + ib1 * lda], &stride1, &dwork[iauple], &(i32){sdim_local + 1});
            SLC_DCOPY(&sdim_local, &b[ib1 + ib1 * ldb], &stride2, &dwork[ibuple], &(i32){sdim_local + 1});
            SLC_DCOPY(&sdim_local, &d[m1 + ib1 + ib1 * ldd], &stride3, &dwork[idlole], &int1);
        } else {
            SLC_DLACPY("F", &dim1, &dim1, &a[ib1 + ib1 * lda], &lda, &dwork[iauple], &sdim_local);
            SLC_DLASET("F", &dim1, &dim1, &ZERO, &ZERO, &dwork[ialole], &sdim_local);
            SLC_DLASET("F", &dim1, &dim1, &ZERO, &ZERO, &dwork[iaupri], &sdim_local);
            SLC_DLACPY("F", &dim1, &dim1, &a[m1 + ib1 + (m1 + ib1) * lda], &lda, &dwork[ialori], &sdim_local);

            SLC_DLACPY("F", &dim1, &dim1, &b[ib1 + ib1 * ldb], &ldb, &dwork[ibuple], &sdim_local);
            SLC_DLASET("F", &dim1, &dim1, &ZERO, &ZERO, &dwork[iblole], &sdim_local);
            SLC_DLASET("F", &dim1, &dim1, &ZERO, &ZERO, &dwork[ibupri], &sdim_local);
            SLC_DLACPY("F", &dim1, &dim1, &b[m1 + ib1 + (m1 + ib1) * ldb], &ldb, &dwork[iblori], &sdim_local);

            SLC_DLASET("F", &dim1, &dim1, &ZERO, &ZERO, &dwork[iduple], &sdim_local);
            SLC_DLACPY("F", &dim1, &dim1, &d[m1 + ib1 + ib1 * ldd], &ldd, &dwork[idlole], &sdim_local);
            SLC_DLACPY("F", &dim1, &dim1, &d[ib1 + (m1 + ib1) * ldd], &ldd, &dwork[idupri], &sdim_local);
            SLC_DLASET("F", &dim1, &dim1, &ZERO, &ZERO, &dwork[idlori], &sdim_local);
        }

        i32 iwrk_ed = 6 * sdim_local * sdim_local;
        i32 ldwork_ed = ldwork - iwrk_ed;

        mb03ed(sdim_local, ulp, &dwork[iauple], sdim_local, &dwork[ibuple], sdim_local,
               &dwork[iduple], sdim_local, &dwork[i1uple], sdim_local,
               &dwork[i2uple], sdim_local, &dwork[i3uple], sdim_local,
               &dwork[iwrk_ed], ldwork_ed, info);

        if (*info > 0) {
            *info = 3;
            return;
        }

        i32 nr = ib2 - 1;

        if (dim1 == 2) {
            i32 itmp = iwrk_ed + dim1 * m;
            i32 itmp2 = itmp + dim1 * m;
            i32 itmp3 = itmp2 + dim1 * dim1;

            SLC_DLACPY("F", &nr, &dim1, &a[ib1 * lda], &lda, &dwork[iwrk_ed], &nr);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &dwork[iwrk_ed], &nr,
                      &dwork[i2uple], &sdim_local, &ZERO, &a[ib1 * lda], &lda);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &a[(m1 + ib1) * lda], &lda,
                      &dwork[i2lole], &sdim_local, &ONE, &a[ib1 * lda], &lda);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &dwork[iwrk_ed], &nr,
                      &dwork[i2upri], &sdim_local, &ZERO, &dwork[itmp], &nr);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &a[(m1 + ib1) * lda], &lda,
                      &dwork[i2lori], &sdim_local, &ONE, &dwork[itmp], &nr);
            SLC_DLACPY("F", &nr, &dim1, &dwork[itmp], &nr, &a[(m1 + ib1) * lda], &lda);

            SLC_DLACPY("F", &nr, &dim1, &a[i1 + ib1 * lda], &lda, &dwork[iwrk_ed], &nr);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &dwork[iwrk_ed], &nr,
                      &dwork[i2uple], &sdim_local, &ZERO, &a[i1 + ib1 * lda], &lda);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &a[i1 + (m1 + ib1) * lda], &lda,
                      &dwork[i2lole], &sdim_local, &ONE, &a[i1 + ib1 * lda], &lda);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &dwork[iwrk_ed], &nr,
                      &dwork[i2upri], &sdim_local, &ZERO, &dwork[itmp], &nr);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &a[i1 + (m1 + ib1) * lda], &lda,
                      &dwork[i2lori], &sdim_local, &ONE, &dwork[itmp], &nr);
            SLC_DLACPY("F", &nr, &dim1, &dwork[itmp], &nr, &a[i1 + (m1 + ib1) * lda], &lda);

            SLC_DLACPY("F", &dim1, &dim1, &a[m1 + ib1 + ib1 * lda], &lda, &dwork[itmp2], &dim1);
            SLC_DLACPY("F", &dim1, &dim1, &a[ib1 + (m1 + ib1) * lda], &lda, &dwork[itmp3], &dim1);
            SLC_DLASET("F", &dim1, &dim1, &ZERO, &ZERO, &a[m1 + ib1 + ib1 * lda], &lda);
            i32 ncols = m1 - ib2;
            if (ncols > 0) {
                SLC_DGEMM("T", "N", &dim1, &ncols, &dim1, &ONE, &dwork[i3upri], &sdim_local,
                          &a[ib1 + ib2 * lda], &lda, &ZERO, &a[m1 + ib1 + ib2 * lda], &lda);
            }
            i32 ncols2 = m1 - ib1;
            SLC_DGEMM("T", "N", &dim1, &ncols2, &dim1, &ONE, &dwork[i3uple], &sdim_local,
                      &a[ib1 + ib1 * lda], &lda, &ZERO, &dwork[itmp], &dim1);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &ONE, &dwork[i3lole], &sdim_local,
                      &dwork[itmp2], &dim1, &ONE, &dwork[itmp], &dim1);
            SLC_DLACPY("F", &dim1, &ncols2, &dwork[itmp], &dim1, &a[ib1 + ib1 * lda], &lda);

            SLC_DGEMM("T", "N", &dim1, &ncols2, &dim1, &ONE, &dwork[i3lole], &sdim_local,
                      &a[m1 + ib1 + (m1 + ib1) * lda], &lda, &ZERO, &a[ib1 + (m1 + ib1) * lda], &lda);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &ONE, &dwork[i3uple], &sdim_local,
                      &dwork[itmp3], &dim1, &ONE, &a[ib1 + (m1 + ib1) * lda], &lda);
            SLC_DGEMM("T", "N", &dim1, &ncols2, &dim1, &ONE, &dwork[i3lori], &sdim_local,
                      &a[m1 + ib1 + (m1 + ib1) * lda], &lda, &ZERO, &dwork[itmp], &dim1);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &ONE, &dwork[i3upri], &sdim_local,
                      &dwork[itmp3], &dim1, &ONE, &dwork[itmp], &dim1);
            SLC_DLACPY("F", &dim1, &ncols2, &dwork[itmp], &dim1, &a[m1 + ib1 + (m1 + ib1) * lda], &lda);

            SLC_DLACPY("F", &nr, &dim1, &b[ib1 * ldb], &ldb, &dwork[iwrk_ed], &nr);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &dwork[iwrk_ed], &nr,
                      &dwork[i1uple], &sdim_local, &ZERO, &b[ib1 * ldb], &ldb);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &b[(m1 + ib1) * ldb], &ldb,
                      &dwork[i1lole], &sdim_local, &ONE, &b[ib1 * ldb], &ldb);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &dwork[iwrk_ed], &nr,
                      &dwork[i1upri], &sdim_local, &ZERO, &dwork[itmp], &nr);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &b[(m1 + ib1) * ldb], &ldb,
                      &dwork[i1lori], &sdim_local, &ONE, &dwork[itmp], &nr);
            SLC_DLACPY("F", &nr, &dim1, &dwork[itmp], &nr, &b[(m1 + ib1) * ldb], &ldb);

            SLC_DLACPY("F", &nr, &dim1, &b[i1 + ib1 * ldb], &ldb, &dwork[iwrk_ed], &nr);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &dwork[iwrk_ed], &nr,
                      &dwork[i1uple], &sdim_local, &ZERO, &b[i1 + ib1 * ldb], &ldb);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &b[i1 + (m1 + ib1) * ldb], &ldb,
                      &dwork[i1lole], &sdim_local, &ONE, &b[i1 + ib1 * ldb], &ldb);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &dwork[iwrk_ed], &nr,
                      &dwork[i1upri], &sdim_local, &ZERO, &dwork[itmp], &nr);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &b[i1 + (m1 + ib1) * ldb], &ldb,
                      &dwork[i1lori], &sdim_local, &ONE, &dwork[itmp], &nr);
            SLC_DLACPY("F", &nr, &dim1, &dwork[itmp], &nr, &b[i1 + (m1 + ib1) * ldb], &ldb);

            SLC_DLACPY("F", &dim1, &dim1, &b[m1 + ib1 + ib1 * ldb], &ldb, &dwork[itmp2], &dim1);
            SLC_DLACPY("F", &dim1, &dim1, &b[ib1 + (m1 + ib1) * ldb], &ldb, &dwork[itmp3], &dim1);
            if (ncols > 0) {
                SLC_DGEMM("T", "N", &dim1, &ncols, &dim1, &ONE, &dwork[i2upri], &sdim_local,
                          &b[ib1 + ib2 * ldb], &ldb, &ZERO, &b[m1 + ib1 + ib2 * ldb], &ldb);
            }
            SLC_DLASET("F", &dim1, &dim1, &ZERO, &ZERO, &b[m1 + ib1 + ib1 * ldb], &ldb);
            SLC_DGEMM("T", "N", &dim1, &ncols2, &dim1, &ONE, &dwork[i2uple], &sdim_local,
                      &b[ib1 + ib1 * ldb], &ldb, &ZERO, &dwork[itmp], &dim1);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &ONE, &dwork[i2lole], &sdim_local,
                      &dwork[itmp2], &dim1, &ONE, &dwork[itmp], &dim1);
            SLC_DLACPY("F", &dim1, &ncols2, &dwork[itmp], &dim1, &b[ib1 + ib1 * ldb], &ldb);

            SLC_DGEMM("T", "N", &dim1, &ncols2, &dim1, &ONE, &dwork[i2lole], &sdim_local,
                      &b[m1 + ib1 + (m1 + ib1) * ldb], &ldb, &ZERO, &b[ib1 + (m1 + ib1) * ldb], &ldb);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &ONE, &dwork[i2uple], &sdim_local,
                      &dwork[itmp3], &dim1, &ONE, &b[ib1 + (m1 + ib1) * ldb], &ldb);
            SLC_DGEMM("T", "N", &dim1, &ncols2, &dim1, &ONE, &dwork[i2lori], &sdim_local,
                      &b[m1 + ib1 + (m1 + ib1) * ldb], &ldb, &ZERO, &dwork[itmp], &dim1);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &ONE, &dwork[i2upri], &sdim_local,
                      &dwork[itmp3], &dim1, &ONE, &dwork[itmp], &dim1);
            SLC_DLACPY("F", &dim1, &ncols2, &dwork[itmp], &dim1, &b[m1 + ib1 + (m1 + ib1) * ldb], &ldb);

            SLC_DLACPY("F", &nr, &dim1, &d[ib1 * ldd], &ldd, &dwork[iwrk_ed], &nr);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &dwork[iwrk_ed], &nr,
                      &dwork[i1uple], &sdim_local, &ZERO, &d[ib1 * ldd], &ldd);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &d[(m1 + ib1) * ldd], &ldd,
                      &dwork[i1lole], &sdim_local, &ONE, &d[ib1 * ldd], &ldd);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &dwork[iwrk_ed], &nr,
                      &dwork[i1upri], &sdim_local, &ZERO, &dwork[itmp], &nr);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &d[(m1 + ib1) * ldd], &ldd,
                      &dwork[i1lori], &sdim_local, &ONE, &dwork[itmp], &nr);
            SLC_DLACPY("F", &nr, &dim1, &dwork[itmp], &nr, &d[(m1 + ib1) * ldd], &ldd);

            SLC_DLACPY("F", &nr, &dim1, &d[i1 + ib1 * ldd], &ldd, &dwork[iwrk_ed], &nr);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &dwork[iwrk_ed], &nr,
                      &dwork[i1uple], &sdim_local, &ZERO, &d[i1 + ib1 * ldd], &ldd);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &d[i1 + (m1 + ib1) * ldd], &ldd,
                      &dwork[i1lole], &sdim_local, &ONE, &d[i1 + ib1 * ldd], &ldd);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &dwork[iwrk_ed], &nr,
                      &dwork[i1upri], &sdim_local, &ZERO, &dwork[itmp], &nr);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &d[i1 + (m1 + ib1) * ldd], &ldd,
                      &dwork[i1lori], &sdim_local, &ONE, &dwork[itmp], &nr);
            SLC_DLACPY("F", &nr, &dim1, &dwork[itmp], &nr, &d[i1 + (m1 + ib1) * ldd], &ldd);

            SLC_DLACPY("F", &dim1, &dim1, &d[ib1 + ib1 * ldd], &ldd, &dwork[itmp2], &dim1);
            SLC_DLACPY("F", &dim1, &dim1, &d[m1 + ib1 + (m1 + ib1) * ldd], &ldd, &dwork[itmp3], &dim1);
            SLC_DGEMM("T", "N", &dim1, &ncols2, &dim1, &ONE, &dwork[i3lole], &sdim_local,
                      &d[m1 + ib1 + ib1 * ldd], &ldd, &ZERO, &d[ib1 + ib1 * ldd], &ldd);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &ONE, &dwork[i3uple], &sdim_local,
                      &dwork[itmp2], &dim1, &ONE, &d[ib1 + ib1 * ldd], &ldd);
            SLC_DLASET("F", &dim1, &dim1, &ZERO, &ZERO, &d[m1 + ib1 + ib1 * ldd], &ldd);
            SLC_DGEMM("T", "N", &dim1, &ncols2, &dim1, &ONE, &dwork[i3lori], &sdim_local,
                      &d[m1 + ib1 + (ib1 + 1) * ldd], &ldd, &ZERO, &dwork[itmp], &dim1);
            SLC_DLACPY("F", &dim1, &ncols2, &dwork[itmp], &dim1, &d[m1 + ib1 + (ib1 + 1) * ldd], &ldd);

            SLC_DGEMM("T", "N", &dim1, &ncols2, &dim1, &ONE, &dwork[i3upri], &sdim_local,
                      &d[ib1 + (m1 + ib1) * ldd], &ldd, &ZERO, &d[m1 + ib1 + (m1 + ib1) * ldd], &ldd);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &ONE, &dwork[i3lori], &sdim_local,
                      &dwork[itmp3], &dim1, &ONE, &d[m1 + ib1 + (m1 + ib1) * ldd], &ldd);
            SLC_DGEMM("T", "N", &dim1, &ncols2, &dim1, &ONE, &dwork[i3uple], &sdim_local,
                      &d[ib1 + (m1 + ib1) * ldd], &ldd, &ZERO, &dwork[itmp], &dim1);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &ONE, &dwork[i3lole], &sdim_local,
                      &dwork[itmp3], &dim1, &ONE, &dwork[itmp], &dim1);
            SLC_DLACPY("F", &dim1, &ncols2, &dwork[itmp], &dim1, &d[ib1 + (m1 + ib1) * ldd], &ldd);

            itmp = iwrk_ed + n * dim1;

            if (lcmpq1) {
                SLC_DLACPY("F", &n, &dim1, &q1[ib1 * ldq1], &ldq1, &dwork[iwrk_ed], &n);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &dwork[iwrk_ed], &n,
                          &dwork[i1uple], &sdim_local, &ZERO, &q1[ib1 * ldq1], &ldq1);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &q1[(m1 + ib1) * ldq1], &ldq1,
                          &dwork[i1lole], &sdim_local, &ONE, &q1[ib1 * ldq1], &ldq1);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &dwork[iwrk_ed], &n,
                          &dwork[i1upri], &sdim_local, &ZERO, &dwork[itmp], &n);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &q1[(m1 + ib1) * ldq1], &ldq1,
                          &dwork[i1lori], &sdim_local, &ONE, &dwork[itmp], &n);
                SLC_DLACPY("F", &n, &dim1, &dwork[itmp], &n, &q1[(m1 + ib1) * ldq1], &ldq1);
            }

            if (lcmpq2) {
                SLC_DLACPY("F", &n, &dim1, &q2[ib1 * ldq2], &ldq2, &dwork[iwrk_ed], &n);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &dwork[iwrk_ed], &n,
                          &dwork[i2uple], &sdim_local, &ZERO, &q2[ib1 * ldq2], &ldq2);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &q2[(m1 + ib1) * ldq2], &ldq2,
                          &dwork[i2lole], &sdim_local, &ONE, &q2[ib1 * ldq2], &ldq2);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &dwork[iwrk_ed], &n,
                          &dwork[i2upri], &sdim_local, &ZERO, &dwork[itmp], &n);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &q2[(m1 + ib1) * ldq2], &ldq2,
                          &dwork[i2lori], &sdim_local, &ONE, &dwork[itmp], &n);
                SLC_DLACPY("F", &n, &dim1, &dwork[itmp], &n, &q2[(m1 + ib1) * ldq2], &ldq2);
            }

            if (lcmpq3) {
                SLC_DLACPY("F", &n, &dim1, &q3[ib1 * ldq3], &ldq3, &dwork[iwrk_ed], &n);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &dwork[iwrk_ed], &n,
                          &dwork[i3uple], &sdim_local, &ZERO, &q3[ib1 * ldq3], &ldq3);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &q3[(m1 + ib1) * ldq3], &ldq3,
                          &dwork[i3lole], &sdim_local, &ONE, &q3[ib1 * ldq3], &ldq3);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &dwork[iwrk_ed], &n,
                          &dwork[i3upri], &sdim_local, &ZERO, &dwork[itmp], &n);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &q3[(m1 + ib1) * ldq3], &ldq3,
                          &dwork[i3lori], &sdim_local, &ONE, &dwork[itmp], &n);
                SLC_DLACPY("F", &n, &dim1, &dwork[itmp], &n, &q3[(m1 + ib1) * ldq3], &ldq3);
            }
        } else {
            f64 tmp2_local, tmp3_local;

            SLC_DCOPY(&nr, &a[ib1 * lda], &int1, &dwork[iwrk_ed], &int1);
            SLC_DSCAL(&nr, &dwork[i2uple], &a[ib1 * lda], &int1);
            SLC_DAXPY(&nr, &dwork[i2lole], &a[(m1 + ib1) * lda], &int1, &a[ib1 * lda], &int1);
            SLC_DSCAL(&nr, &dwork[i2lori], &a[(m1 + ib1) * lda], &int1);
            SLC_DAXPY(&nr, &dwork[i2upri], &dwork[iwrk_ed], &int1, &a[(m1 + ib1) * lda], &int1);

            SLC_DCOPY(&nr, &a[i1 + ib1 * lda], &int1, &dwork[iwrk_ed], &int1);
            SLC_DSCAL(&nr, &dwork[i2uple], &a[i1 + ib1 * lda], &int1);
            SLC_DAXPY(&nr, &dwork[i2lole], &a[i1 + (m1 + ib1) * lda], &int1, &a[i1 + ib1 * lda], &int1);
            SLC_DSCAL(&nr, &dwork[i2lori], &a[i1 + (m1 + ib1) * lda], &int1);
            SLC_DAXPY(&nr, &dwork[i2upri], &dwork[iwrk_ed], &int1, &a[i1 + (m1 + ib1) * lda], &int1);

            tmp2_local = a[m1 + ib1 + ib1 * lda];
            tmp3_local = a[ib1 + (m1 + ib1) * lda];
            i32 ncols2 = m1 - ib1;
            if (m1 > ib1) {
                i32 ncols3 = m1 - ib1;
                SLC_DCOPY(&ncols3, &a[ib1 + (ib1 + 1) * lda], &lda, &a[m1 + ib1 + (ib1 + 1) * lda], &lda);
                SLC_DSCAL(&ncols3, &dwork[i3upri], &a[m1 + ib1 + (ib1 + 1) * lda], &lda);
            }
            a[m1 + ib1 + ib1 * lda] = ZERO;
            SLC_DSCAL(&ncols2, &dwork[i3uple], &a[ib1 + ib1 * lda], &lda);
            a[ib1 + ib1 * lda] += dwork[i3lole] * tmp2_local;

            SLC_DCOPY(&ncols2, &a[m1 + ib1 + (m1 + ib1) * lda], &lda, &a[ib1 + (m1 + ib1) * lda], &lda);
            SLC_DSCAL(&ncols2, &dwork[i3lole], &a[ib1 + (m1 + ib1) * lda], &lda);
            a[ib1 + (m1 + ib1) * lda] += dwork[i3uple] * tmp3_local;
            SLC_DSCAL(&ncols2, &dwork[i3lori], &a[m1 + ib1 + (m1 + ib1) * lda], &lda);
            a[m1 + ib1 + (m1 + ib1) * lda] += dwork[i3upri] * tmp3_local;

            SLC_DCOPY(&nr, &b[ib1 * ldb], &int1, &dwork[iwrk_ed], &int1);
            SLC_DSCAL(&nr, &dwork[i1uple], &b[ib1 * ldb], &int1);
            SLC_DAXPY(&nr, &dwork[i1lole], &b[(m1 + ib1) * ldb], &int1, &b[ib1 * ldb], &int1);
            SLC_DSCAL(&nr, &dwork[i1lori], &b[(m1 + ib1) * ldb], &int1);
            SLC_DAXPY(&nr, &dwork[i1upri], &dwork[iwrk_ed], &int1, &b[(m1 + ib1) * ldb], &int1);

            SLC_DCOPY(&nr, &b[i1 + ib1 * ldb], &int1, &dwork[iwrk_ed], &int1);
            SLC_DSCAL(&nr, &dwork[i1uple], &b[i1 + ib1 * ldb], &int1);
            SLC_DAXPY(&nr, &dwork[i1lole], &b[i1 + (m1 + ib1) * ldb], &int1, &b[i1 + ib1 * ldb], &int1);
            SLC_DSCAL(&nr, &dwork[i1lori], &b[i1 + (m1 + ib1) * ldb], &int1);
            SLC_DAXPY(&nr, &dwork[i1upri], &dwork[iwrk_ed], &int1, &b[i1 + (m1 + ib1) * ldb], &int1);

            tmp2_local = b[m1 + ib1 + ib1 * ldb];
            tmp3_local = b[ib1 + (m1 + ib1) * ldb];
            if (m1 > ib1) {
                i32 ncols3 = m1 - ib1;
                SLC_DCOPY(&ncols3, &b[ib1 + (ib1 + 1) * ldb], &ldb, &b[m1 + ib1 + (ib1 + 1) * ldb], &ldb);
                SLC_DSCAL(&ncols3, &dwork[i2upri], &b[m1 + ib1 + (ib1 + 1) * ldb], &ldb);
            }
            b[m1 + ib1 + ib1 * ldb] = ZERO;
            SLC_DSCAL(&ncols2, &dwork[i2uple], &b[ib1 + ib1 * ldb], &ldb);
            b[ib1 + ib1 * ldb] += dwork[i2lole] * tmp2_local;

            SLC_DCOPY(&ncols2, &b[m1 + ib1 + (m1 + ib1) * ldb], &ldb, &b[ib1 + (m1 + ib1) * ldb], &ldb);
            SLC_DSCAL(&ncols2, &dwork[i2lole], &b[ib1 + (m1 + ib1) * ldb], &ldb);
            b[ib1 + (m1 + ib1) * ldb] += dwork[i2uple] * tmp3_local;
            SLC_DSCAL(&ncols2, &dwork[i2lori], &b[m1 + ib1 + (m1 + ib1) * ldb], &ldb);
            b[m1 + ib1 + (m1 + ib1) * ldb] += dwork[i2upri] * tmp3_local;

            SLC_DCOPY(&nr, &d[ib1 * ldd], &int1, &dwork[iwrk_ed], &int1);
            SLC_DSCAL(&nr, &dwork[i1uple], &d[ib1 * ldd], &int1);
            SLC_DAXPY(&nr, &dwork[i1lole], &d[(m1 + ib1) * ldd], &int1, &d[ib1 * ldd], &int1);
            SLC_DSCAL(&nr, &dwork[i1lori], &d[(m1 + ib1) * ldd], &int1);
            SLC_DAXPY(&nr, &dwork[i1upri], &dwork[iwrk_ed], &int1, &d[(m1 + ib1) * ldd], &int1);

            SLC_DCOPY(&nr, &d[i1 + ib1 * ldd], &int1, &dwork[iwrk_ed], &int1);
            SLC_DSCAL(&nr, &dwork[i1uple], &d[i1 + ib1 * ldd], &int1);
            SLC_DAXPY(&nr, &dwork[i1lole], &d[i1 + (m1 + ib1) * ldd], &int1, &d[i1 + ib1 * ldd], &int1);
            SLC_DSCAL(&nr, &dwork[i1lori], &d[i1 + (m1 + ib1) * ldd], &int1);
            SLC_DAXPY(&nr, &dwork[i1upri], &dwork[iwrk_ed], &int1, &d[i1 + (m1 + ib1) * ldd], &int1);

            tmp2_local = d[ib1 + ib1 * ldd];
            tmp3_local = d[m1 + ib1 + (m1 + ib1) * ldd];
            SLC_DCOPY(&ncols2, &d[m1 + ib1 + ib1 * ldd], &ldd, &d[ib1 + ib1 * ldd], &ldd);
            SLC_DSCAL(&ncols2, &dwork[i3lole], &d[ib1 + ib1 * ldd], &ldd);
            d[ib1 + ib1 * ldd] += dwork[i3uple] * tmp2_local;
            d[m1 + ib1 + ib1 * ldd] = ZERO;
            SLC_DSCAL(&ncols2, &dwork[i3lori], &d[m1 + ib1 + (ib1 + 1) * ldd], &ldd);

            SLC_DCOPY(&ncols2, &d[ib1 + (m1 + ib1) * ldd], &ldd, &d[m1 + ib1 + (m1 + ib1) * ldd], &ldd);
            SLC_DSCAL(&ncols2, &dwork[i3upri], &d[m1 + ib1 + (m1 + ib1) * ldd], &ldd);
            d[m1 + ib1 + (m1 + ib1) * ldd] += dwork[i3lori] * tmp3_local;
            SLC_DSCAL(&ncols2, &dwork[i3uple], &d[ib1 + (m1 + ib1) * ldd], &ldd);
            d[ib1 + (m1 + ib1) * ldd] += dwork[i3lole] * tmp3_local;

            if (lcmpq1) {
                SLC_DCOPY(&n, &q1[ib1 * ldq1], &int1, &dwork[iwrk_ed], &int1);
                SLC_DSCAL(&n, &dwork[i1uple], &q1[ib1 * ldq1], &int1);
                SLC_DAXPY(&n, &dwork[i1lole], &q1[(m1 + ib1) * ldq1], &int1, &q1[ib1 * ldq1], &int1);
                SLC_DSCAL(&n, &dwork[i1lori], &q1[(m1 + ib1) * ldq1], &int1);
                SLC_DAXPY(&n, &dwork[i1upri], &dwork[iwrk_ed], &int1, &q1[(m1 + ib1) * ldq1], &int1);
            }

            if (lcmpq2) {
                SLC_DCOPY(&n, &q2[ib1 * ldq2], &int1, &dwork[iwrk_ed], &int1);
                SLC_DSCAL(&n, &dwork[i2uple], &q2[ib1 * ldq2], &int1);
                SLC_DAXPY(&n, &dwork[i2lole], &q2[(m1 + ib1) * ldq2], &int1, &q2[ib1 * ldq2], &int1);
                SLC_DSCAL(&n, &dwork[i2lori], &q2[(m1 + ib1) * ldq2], &int1);
                SLC_DAXPY(&n, &dwork[i2upri], &dwork[iwrk_ed], &int1, &q2[(m1 + ib1) * ldq2], &int1);
            }

            if (lcmpq3) {
                SLC_DCOPY(&n, &q3[ib1 * ldq3], &int1, &dwork[iwrk_ed], &int1);
                SLC_DSCAL(&n, &dwork[i3uple], &q3[ib1 * ldq3], &int1);
                SLC_DAXPY(&n, &dwork[i3lole], &q3[(m1 + ib1) * ldq3], &int1, &q3[ib1 * ldq3], &int1);
                SLC_DSCAL(&n, &dwork[i3lori], &q3[(m1 + ib1) * ldq3], &int1);
                SLC_DAXPY(&n, &dwork[i3upri], &dwork[iwrk_ed], &int1, &q3[(m1 + ib1) * ldq3], &int1);
            }
        }

        for (i32 j_idx = k_idx - 1; j_idx >= 0; j_idx--) {
            i32 ij1 = iwork[j_idx];
            i32 ij2 = iwork[j_idx + 1];
            i32 dim1_j = iwork[k_idx + 1] - iwork[k_idx];
            i32 dim2_j = ij2 - ij1;
            i32 sdim_j = dim1_j + dim2_j;

            i32 iauple_j = 0;
            i32 ialole_j = iauple_j + dim1_j;
            i32 iaupri_j = dim1_j * sdim_j;
            i32 ialori_j = iaupri_j + dim1_j;
            i32 ibuple_j = sdim_j * sdim_j;
            i32 iblole_j = ibuple_j + dim1_j;
            i32 ibupri_j = sdim_j * sdim_j + dim1_j * sdim_j;
            i32 iblori_j = ibupri_j + dim1_j;
            i32 iduple_j = 2 * sdim_j * sdim_j;
            i32 idlole_j = iduple_j + dim1_j;
            i32 idupri_j = 2 * sdim_j * sdim_j + dim1_j * sdim_j;
            i32 idlori_j = idupri_j + dim1_j;
            i32 i1uple_j = 3 * sdim_j * sdim_j;
            i32 i1lole_j = i1uple_j + dim1_j;
            i32 i1upri_j = 3 * sdim_j * sdim_j + dim1_j * sdim_j;
            i32 i1lori_j = i1upri_j + dim1_j;
            i32 i2uple_j = 4 * sdim_j * sdim_j;
            i32 i2lole_j = i2uple_j + dim1_j;
            i32 i2upri_j = 4 * sdim_j * sdim_j + dim1_j * sdim_j;
            i32 i2lori_j = i2upri_j + dim1_j;
            i32 i3uple_j = 5 * sdim_j * sdim_j;
            i32 i3lole_j = i3uple_j + dim1_j;
            i32 i3upri_j = 5 * sdim_j * sdim_j + dim1_j * sdim_j;
            i32 i3lori_j = i3upri_j + dim1_j;

            if (dim1_j == 2 && dim2_j == 2) {
                SLC_DLACPY("F", &dim1_j, &dim1_j, &a[ib1 + ib1 * lda], &lda, &dwork[iauple_j], &sdim_j);
                SLC_DLACPY("F", &dim2_j, &dim1_j, &a[m1 + ij1 + ib1 * lda], &lda, &dwork[ialole_j], &sdim_j);
                SLC_DLASET("F", &dim1_j, &dim2_j, &ZERO, &ZERO, &dwork[iaupri_j], &sdim_j);
                SLC_DLACPY("F", &dim2_j, &dim2_j, &a[m1 + ij1 + (m1 + ij1) * lda], &lda, &dwork[ialori_j], &sdim_j);

                SLC_DLACPY("F", &dim1_j, &dim1_j, &b[ib1 + ib1 * ldb], &ldb, &dwork[ibuple_j], &sdim_j);
                SLC_DLACPY("F", &dim2_j, &dim1_j, &b[m1 + ij1 + ib1 * ldb], &ldb, &dwork[iblole_j], &sdim_j);
                SLC_DLASET("F", &dim1_j, &dim2_j, &ZERO, &ZERO, &dwork[ibupri_j], &sdim_j);
                SLC_DLACPY("F", &dim2_j, &dim2_j, &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb, &dwork[iblori_j], &sdim_j);

                SLC_DLACPY("F", &dim1_j, &dim1_j, &d[ib1 + ib1 * ldd], &ldd, &dwork[iduple_j], &sdim_j);
                SLC_DLACPY("F", &dim2_j, &dim1_j, &d[m1 + ij1 + ib1 * ldd], &ldd, &dwork[idlole_j], &sdim_j);
                SLC_DLASET("F", &dim1_j, &dim2_j, &ZERO, &ZERO, &dwork[idupri_j], &sdim_j);
                SLC_DLACPY("F", &dim2_j, &dim2_j, &d[m1 + ij1 + (m1 + ij1) * ldd], &ldd, &dwork[idlori_j], &sdim_j);
            } else if (dim1_j == 1 && dim2_j == 2) {
                dwork[iauple_j] = a[ib1 + ib1 * lda];
                SLC_DCOPY(&dim2_j, &a[m1 + ij1 + ib1 * lda], &int1, &dwork[ialole_j], &int1);
                SLC_DCOPY(&dim2_j, &dum, &int0, &dwork[iaupri_j], &sdim_j);
                SLC_DLACPY("F", &dim2_j, &dim2_j, &a[m1 + ij1 + (m1 + ij1) * lda], &lda, &dwork[ialori_j], &sdim_j);

                dwork[ibuple_j] = b[ib1 + ib1 * ldb];
                SLC_DCOPY(&dim2_j, &b[m1 + ij1 + ib1 * ldb], &int1, &dwork[iblole_j], &int1);
                SLC_DCOPY(&dim2_j, &dum, &int0, &dwork[ibupri_j], &sdim_j);
                SLC_DLACPY("F", &dim2_j, &dim2_j, &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb, &dwork[iblori_j], &sdim_j);

                dwork[iduple_j] = d[ib1 + ib1 * ldd];
                SLC_DCOPY(&dim2_j, &d[m1 + ij1 + ib1 * ldd], &int1, &dwork[idlole_j], &int1);
                SLC_DCOPY(&dim2_j, &dum, &int0, &dwork[idupri_j], &sdim_j);
                SLC_DLACPY("F", &dim2_j, &dim2_j, &d[m1 + ij1 + (m1 + ij1) * ldd], &ldd, &dwork[idlori_j], &sdim_j);
            } else if (dim1_j == 2 && dim2_j == 1) {
                SLC_DLACPY("F", &dim1_j, &dim1_j, &a[ib1 + ib1 * lda], &lda, &dwork[iauple_j], &sdim_j);
                SLC_DCOPY(&dim1_j, &a[m1 + ij1 + ib1 * lda], &lda, &dwork[ialole_j], &sdim_j);
                SLC_DCOPY(&dim1_j, &dum, &int0, &dwork[iaupri_j], &int1);
                dwork[ialori_j] = a[m1 + ij1 + (m1 + ij1) * lda];

                SLC_DLACPY("F", &dim1_j, &dim1_j, &b[ib1 + ib1 * ldb], &ldb, &dwork[ibuple_j], &sdim_j);
                SLC_DCOPY(&dim1_j, &b[m1 + ij1 + ib1 * ldb], &ldb, &dwork[iblole_j], &sdim_j);
                SLC_DCOPY(&dim1_j, &dum, &int0, &dwork[ibupri_j], &int1);
                dwork[iblori_j] = b[m1 + ij1 + (m1 + ij1) * ldb];

                SLC_DLACPY("F", &dim1_j, &dim1_j, &d[ib1 + ib1 * ldd], &ldd, &dwork[iduple_j], &sdim_j);
                SLC_DCOPY(&dim1_j, &d[m1 + ij1 + ib1 * ldd], &ldd, &dwork[idlole_j], &sdim_j);
                SLC_DCOPY(&dim1_j, &dum, &int0, &dwork[idupri_j], &int1);
                dwork[idlori_j] = d[m1 + ij1 + (m1 + ij1) * ldd];
            } else {
                dwork[iauple_j] = a[ib1 + ib1 * lda];
                dwork[ialole_j] = a[m1 + ij1 + ib1 * lda];
                dwork[iaupri_j] = ZERO;
                dwork[ialori_j] = a[m1 + ij1 + (m1 + ij1) * lda];

                dwork[ibuple_j] = b[ib1 + ib1 * ldb];
                dwork[iblole_j] = b[m1 + ij1 + ib1 * ldb];
                dwork[ibupri_j] = ZERO;
                dwork[iblori_j] = b[m1 + ij1 + (m1 + ij1) * ldb];

                dwork[iduple_j] = d[ib1 + ib1 * ldd];
                dwork[idlole_j] = d[m1 + ij1 + ib1 * ldd];
                dwork[idupri_j] = ZERO;
                dwork[idlori_j] = d[m1 + ij1 + (m1 + ij1) * ldd];
            }

            i32 iwrk_j = 6 * sdim_j * sdim_j;
            i32 itmp_j = iwrk_j + 2 * n;
            i32 ldwork_j = ldwork - iwrk_j;

            mb03cd("L", &dim1_j, &dim2_j, ulp, &dwork[iauple_j], sdim_j, &dwork[ibuple_j], sdim_j,
                   &dwork[iduple_j], sdim_j, &dwork[i1uple_j], sdim_j, &dwork[i2uple_j], sdim_j,
                   &dwork[i3uple_j], sdim_j, &dwork[iwrk_j], ldwork_j, info);

            if (*info > 0) {
                if (*info <= 2) {
                    *info = 2;
                } else if (*info <= 4) {
                    *info = 3;
                } else {
                    *info = 4;
                }
                return;
            }

            i32 nrow_j = ij2 - 1;
            i32 nr_j = ib2 - 1;

            if (dim1_j == 2 && dim2_j == 2) {
                SLC_DLACPY("F", &nr_j, &dim1_j, &a[ib1 * lda], &lda, &dwork[iwrk_j], &nr_j);
                SLC_DGEMM("N", "N", &nr_j, &dim1_j, &dim1_j, &ONE, &dwork[iwrk_j], &nr_j,
                          &dwork[i2uple_j], &sdim_j, &ZERO, &a[ib1 * lda], &lda);
                i32 nr_minus_dim1 = nr_j - dim1_j;
                SLC_DGEMM("N", "N", &nr_minus_dim1, &dim1_j, &dim2_j, &ONE, &a[(m1 + ij1) * lda], &lda,
                          &dwork[i2lole_j], &sdim_j, &ONE, &a[ib1 * lda], &lda);
                SLC_DGEMM("N", "N", &nr_j, &dim2_j, &dim1_j, &ONE, &dwork[iwrk_j], &nr_j,
                          &dwork[i2upri_j], &sdim_j, &ZERO, &dwork[itmp_j], &nr_j);
                SLC_DGEMM("N", "N", &nr_minus_dim1, &dim2_j, &dim2_j, &ONE, &a[(m1 + ij1) * lda], &lda,
                          &dwork[i2lori_j], &sdim_j, &ONE, &dwork[itmp_j], &nr_j);
                SLC_DLACPY("F", &nr_j, &dim2_j, &dwork[itmp_j], &nr_j, &a[(m1 + ij1) * lda], &lda);

                SLC_DLACPY("F", &nrow_j, &dim1_j, &a[i1 + ib1 * lda], &lda, &dwork[iwrk_j], &nrow_j);
                SLC_DGEMM("N", "N", &nrow_j, &dim1_j, &dim1_j, &ONE, &dwork[iwrk_j], &nrow_j,
                          &dwork[i2uple_j], &sdim_j, &ZERO, &a[i1 + ib1 * lda], &lda);
                SLC_DGEMM("N", "N", &nrow_j, &dim1_j, &dim2_j, &ONE, &a[i1 + (m1 + ij1) * lda], &lda,
                          &dwork[i2lole_j], &sdim_j, &ONE, &a[i1 + ib1 * lda], &lda);
                SLC_DGEMM("N", "N", &nrow_j, &dim2_j, &dim1_j, &ONE, &dwork[iwrk_j], &nrow_j,
                          &dwork[i2upri_j], &sdim_j, &ZERO, &dwork[itmp_j], &nrow_j);
                SLC_DGEMM("N", "N", &nrow_j, &dim2_j, &dim2_j, &ONE, &a[i1 + (m1 + ij1) * lda], &lda,
                          &dwork[i2lori_j], &sdim_j, &ONE, &dwork[itmp_j], &nrow_j);
                SLC_DLACPY("F", &nrow_j, &dim2_j, &dwork[itmp_j], &nrow_j, &a[i1 + (m1 + ij1) * lda], &lda);

                i32 ncols_ib = m1 - ib1;
                SLC_DLACPY("F", &dim1_j, &ncols_ib, &a[ib1 + ib1 * lda], &lda, &dwork[iwrk_j], &dim1_j);
                SLC_DGEMM("T", "N", &dim1_j, &ncols_ib, &dim1_j, &ONE, &dwork[i3uple_j], &sdim_j,
                          &dwork[iwrk_j], &dim1_j, &ZERO, &a[ib1 + ib1 * lda], &lda);
                SLC_DGEMM("T", "N", &dim1_j, &ncols_ib, &dim2_j, &ONE, &dwork[i3lole_j], &sdim_j,
                          &a[m1 + ij1 + ib1 * lda], &lda, &ONE, &a[ib1 + ib1 * lda], &lda);
                SLC_DGEMM("T", "N", &dim2_j, &ncols_ib, &dim1_j, &ONE, &dwork[i3upri_j], &sdim_j,
                          &dwork[iwrk_j], &dim1_j, &ZERO, &dwork[itmp_j], &dim2_j);
                SLC_DGEMM("T", "N", &dim2_j, &ncols_ib, &dim2_j, &ONE, &dwork[i3lori_j], &sdim_j,
                          &a[m1 + ij1 + ib1 * lda], &lda, &ONE, &dwork[itmp_j], &dim2_j);
                SLC_DLACPY("F", &dim2_j, &ncols_ib, &dwork[itmp_j], &dim2_j, &a[m1 + ij1 + ib1 * lda], &lda);

                i32 ncols_ij = m1 - ij1;
                SLC_DLACPY("F", &dim1_j, &ncols_ij, &a[ib1 + (m1 + ij1) * lda], &lda, &dwork[iwrk_j], &dim1_j);
                SLC_DGEMM("T", "N", &dim1_j, &ncols_ij, &dim1_j, &ONE, &dwork[i3uple_j], &sdim_j,
                          &dwork[iwrk_j], &dim1_j, &ZERO, &a[ib1 + (m1 + ij1) * lda], &lda);
                SLC_DGEMM("T", "N", &dim1_j, &ncols_ij, &dim2_j, &ONE, &dwork[i3lole_j], &sdim_j,
                          &a[m1 + ij1 + (m1 + ij1) * lda], &lda, &ONE, &a[ib1 + (m1 + ij1) * lda], &lda);
                SLC_DGEMM("T", "N", &dim2_j, &ncols_ij, &dim1_j, &ONE, &dwork[i3upri_j], &sdim_j,
                          &dwork[iwrk_j], &dim1_j, &ZERO, &dwork[itmp_j], &dim2_j);
                SLC_DGEMM("T", "N", &dim2_j, &ncols_ij, &dim2_j, &ONE, &dwork[i3lori_j], &sdim_j,
                          &a[m1 + ij1 + (m1 + ij1) * lda], &lda, &ONE, &dwork[itmp_j], &dim2_j);
                SLC_DLACPY("F", &dim2_j, &ncols_ij, &dwork[itmp_j], &dim2_j, &a[m1 + ij1 + (m1 + ij1) * lda], &lda);

                if (m2 > 0) {
                    SLC_DLACPY("F", &dim1_j, &m4, &a[ib1 + i2 * lda], &lda, &dwork[iwrk_j], &dim1_j);
                    SLC_DGEMM("T", "N", &dim1_j, &m4, &dim1_j, &ONE, &dwork[i3uple_j], &sdim_j,
                              &dwork[iwrk_j], &dim1_j, &ZERO, &a[ib1 + i2 * lda], &lda);
                    SLC_DGEMM("T", "N", &dim1_j, &m4, &dim2_j, &ONE, &dwork[i3lole_j], &sdim_j,
                              &a[m1 + ij1 + i2 * lda], &lda, &ONE, &a[ib1 + i2 * lda], &lda);
                    SLC_DGEMM("T", "N", &dim2_j, &m4, &dim1_j, &ONE, &dwork[i3upri_j], &sdim_j,
                              &dwork[iwrk_j], &dim1_j, &ZERO, &dwork[itmp_j], &dim2_j);
                    SLC_DGEMM("T", "N", &dim2_j, &m4, &dim2_j, &ONE, &dwork[i3lori_j], &sdim_j,
                              &a[m1 + ij1 + i2 * lda], &lda, &ONE, &dwork[itmp_j], &dim2_j);
                    SLC_DLACPY("F", &dim2_j, &m4, &dwork[itmp_j], &dim2_j, &a[m1 + ij1 + i2 * lda], &lda);
                }

                SLC_DLACPY("F", &nr_j, &dim1_j, &b[ib1 * ldb], &ldb, &dwork[iwrk_j], &nr_j);
                SLC_DGEMM("N", "N", &nr_j, &dim1_j, &dim1_j, &ONE, &dwork[iwrk_j], &nr_j,
                          &dwork[i1uple_j], &sdim_j, &ZERO, &b[ib1 * ldb], &ldb);
                SLC_DGEMM("N", "N", &nr_minus_dim1, &dim1_j, &dim2_j, &ONE, &b[(m1 + ij1) * ldb], &ldb,
                          &dwork[i1lole_j], &sdim_j, &ONE, &b[ib1 * ldb], &ldb);
                SLC_DGEMM("N", "N", &nr_j, &dim2_j, &dim1_j, &ONE, &dwork[iwrk_j], &nr_j,
                          &dwork[i1upri_j], &sdim_j, &ZERO, &dwork[itmp_j], &nr_j);
                SLC_DGEMM("N", "N", &nr_minus_dim1, &dim2_j, &dim2_j, &ONE, &b[(m1 + ij1) * ldb], &ldb,
                          &dwork[i1lori_j], &sdim_j, &ONE, &dwork[itmp_j], &nr_j);
                SLC_DLACPY("F", &nr_j, &dim2_j, &dwork[itmp_j], &nr_j, &b[(m1 + ij1) * ldb], &ldb);

                SLC_DLACPY("F", &nrow_j, &dim1_j, &b[i1 + ib1 * ldb], &ldb, &dwork[iwrk_j], &nrow_j);
                SLC_DGEMM("N", "N", &nrow_j, &dim1_j, &dim1_j, &ONE, &dwork[iwrk_j], &nrow_j,
                          &dwork[i1uple_j], &sdim_j, &ZERO, &b[i1 + ib1 * ldb], &ldb);
                SLC_DGEMM("N", "N", &nrow_j, &dim1_j, &dim2_j, &ONE, &b[i1 + (m1 + ij1) * ldb], &ldb,
                          &dwork[i1lole_j], &sdim_j, &ONE, &b[i1 + ib1 * ldb], &ldb);
                SLC_DGEMM("N", "N", &nrow_j, &dim2_j, &dim1_j, &ONE, &dwork[iwrk_j], &nrow_j,
                          &dwork[i1upri_j], &sdim_j, &ZERO, &dwork[itmp_j], &nrow_j);
                SLC_DGEMM("N", "N", &nrow_j, &dim2_j, &dim2_j, &ONE, &b[i1 + (m1 + ij1) * ldb], &ldb,
                          &dwork[i1lori_j], &sdim_j, &ONE, &dwork[itmp_j], &nrow_j);
                SLC_DLACPY("F", &nrow_j, &dim2_j, &dwork[itmp_j], &nrow_j, &b[i1 + (m1 + ij1) * ldb], &ldb);

                SLC_DLACPY("F", &dim1_j, &ncols_ib, &b[ib1 + ib1 * ldb], &ldb, &dwork[iwrk_j], &dim1_j);
                SLC_DGEMM("T", "N", &dim1_j, &ncols_ib, &dim1_j, &ONE, &dwork[i2uple_j], &sdim_j,
                          &dwork[iwrk_j], &dim1_j, &ZERO, &b[ib1 + ib1 * ldb], &ldb);
                SLC_DGEMM("T", "N", &dim1_j, &ncols_ib, &dim2_j, &ONE, &dwork[i2lole_j], &sdim_j,
                          &b[m1 + ij1 + ib1 * ldb], &ldb, &ONE, &b[ib1 + ib1 * ldb], &ldb);
                SLC_DGEMM("T", "N", &dim2_j, &ncols_ib, &dim1_j, &ONE, &dwork[i2upri_j], &sdim_j,
                          &dwork[iwrk_j], &dim1_j, &ZERO, &dwork[itmp_j], &dim2_j);
                SLC_DGEMM("T", "N", &dim2_j, &ncols_ib, &dim2_j, &ONE, &dwork[i2lori_j], &sdim_j,
                          &b[m1 + ij1 + ib1 * ldb], &ldb, &ONE, &dwork[itmp_j], &dim2_j);
                SLC_DLACPY("F", &dim2_j, &ncols_ib, &dwork[itmp_j], &dim2_j, &b[m1 + ij1 + ib1 * ldb], &ldb);

                SLC_DLACPY("F", &dim1_j, &ncols_ij, &b[ib1 + (m1 + ij1) * ldb], &ldb, &dwork[iwrk_j], &dim1_j);
                SLC_DGEMM("T", "N", &dim1_j, &ncols_ij, &dim1_j, &ONE, &dwork[i2uple_j], &sdim_j,
                          &dwork[iwrk_j], &dim1_j, &ZERO, &b[ib1 + (m1 + ij1) * ldb], &ldb);
                SLC_DGEMM("T", "N", &dim1_j, &ncols_ij, &dim2_j, &ONE, &dwork[i2lole_j], &sdim_j,
                          &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb, &ONE, &b[ib1 + (m1 + ij1) * ldb], &ldb);
                SLC_DGEMM("T", "N", &dim2_j, &ncols_ij, &dim1_j, &ONE, &dwork[i2upri_j], &sdim_j,
                          &dwork[iwrk_j], &dim1_j, &ZERO, &dwork[itmp_j], &dim2_j);
                SLC_DGEMM("T", "N", &dim2_j, &ncols_ij, &dim2_j, &ONE, &dwork[i2lori_j], &sdim_j,
                          &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb, &ONE, &dwork[itmp_j], &dim2_j);
                SLC_DLACPY("F", &dim2_j, &ncols_ij, &dwork[itmp_j], &dim2_j, &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb);

                if (m2 > 0) {
                    SLC_DLACPY("F", &dim1_j, &m4, &b[ib1 + i2 * ldb], &ldb, &dwork[iwrk_j], &dim1_j);
                    SLC_DGEMM("T", "N", &dim1_j, &m4, &dim1_j, &ONE, &dwork[i2uple_j], &sdim_j,
                              &dwork[iwrk_j], &dim1_j, &ZERO, &b[ib1 + i2 * ldb], &ldb);
                    SLC_DGEMM("T", "N", &dim1_j, &m4, &dim2_j, &ONE, &dwork[i2lole_j], &sdim_j,
                              &b[m1 + ij1 + i2 * ldb], &ldb, &ONE, &b[ib1 + i2 * ldb], &ldb);
                    SLC_DGEMM("T", "N", &dim2_j, &m4, &dim1_j, &ONE, &dwork[i2upri_j], &sdim_j,
                              &dwork[iwrk_j], &dim1_j, &ZERO, &dwork[itmp_j], &dim2_j);
                    SLC_DGEMM("T", "N", &dim2_j, &m4, &dim2_j, &ONE, &dwork[i2lori_j], &sdim_j,
                              &b[m1 + ij1 + i2 * ldb], &ldb, &ONE, &dwork[itmp_j], &dim2_j);
                    SLC_DLACPY("F", &dim2_j, &m4, &dwork[itmp_j], &dim2_j, &b[m1 + ij1 + i2 * ldb], &ldb);
                }

                SLC_DLACPY("F", &nr_j, &dim1_j, &d[ib1 * ldd], &ldd, &dwork[iwrk_j], &nr_j);
                SLC_DGEMM("N", "N", &nr_j, &dim1_j, &dim1_j, &ONE, &dwork[iwrk_j], &nr_j,
                          &dwork[i1uple_j], &sdim_j, &ZERO, &d[ib1 * ldd], &ldd);
                SLC_DGEMM("N", "N", &nr_minus_dim1, &dim1_j, &dim2_j, &ONE, &d[(m1 + ij1) * ldd], &ldd,
                          &dwork[i1lole_j], &sdim_j, &ONE, &d[ib1 * ldd], &ldd);
                SLC_DGEMM("N", "N", &nr_j, &dim2_j, &dim1_j, &ONE, &dwork[iwrk_j], &nr_j,
                          &dwork[i1upri_j], &sdim_j, &ZERO, &dwork[itmp_j], &nr_j);
                SLC_DGEMM("N", "N", &nr_minus_dim1, &dim2_j, &dim2_j, &ONE, &d[(m1 + ij1) * ldd], &ldd,
                          &dwork[i1lori_j], &sdim_j, &ONE, &dwork[itmp_j], &nr_j);
                SLC_DLACPY("F", &nr_j, &dim2_j, &dwork[itmp_j], &nr_j, &d[(m1 + ij1) * ldd], &ldd);

                SLC_DLACPY("F", &nrow_j, &dim1_j, &d[i1 + ib1 * ldd], &ldd, &dwork[iwrk_j], &nrow_j);
                SLC_DGEMM("N", "N", &nrow_j, &dim1_j, &dim1_j, &ONE, &dwork[iwrk_j], &nrow_j,
                          &dwork[i1uple_j], &sdim_j, &ZERO, &d[i1 + ib1 * ldd], &ldd);
                SLC_DGEMM("N", "N", &nrow_j, &dim1_j, &dim2_j, &ONE, &d[i1 + (m1 + ij1) * ldd], &ldd,
                          &dwork[i1lole_j], &sdim_j, &ONE, &d[i1 + ib1 * ldd], &ldd);
                SLC_DGEMM("N", "N", &nrow_j, &dim2_j, &dim1_j, &ONE, &dwork[iwrk_j], &nrow_j,
                          &dwork[i1upri_j], &sdim_j, &ZERO, &dwork[itmp_j], &nrow_j);
                SLC_DGEMM("N", "N", &nrow_j, &dim2_j, &dim2_j, &ONE, &d[i1 + (m1 + ij1) * ldd], &ldd,
                          &dwork[i1lori_j], &sdim_j, &ONE, &dwork[itmp_j], &nrow_j);
                SLC_DLACPY("F", &nrow_j, &dim2_j, &dwork[itmp_j], &nrow_j, &d[i1 + (m1 + ij1) * ldd], &ldd);

                SLC_DLACPY("F", &dim1_j, &ncols_ib, &d[ib1 + ib1 * ldd], &ldd, &dwork[iwrk_j], &dim1_j);
                SLC_DGEMM("T", "N", &dim1_j, &ncols_ib, &dim1_j, &ONE, &dwork[i3uple_j], &sdim_j,
                          &dwork[iwrk_j], &dim1_j, &ZERO, &d[ib1 + ib1 * ldd], &ldd);
                SLC_DGEMM("T", "N", &dim1_j, &ncols_ib, &dim2_j, &ONE, &dwork[i3lole_j], &sdim_j,
                          &d[m1 + ij1 + ib1 * ldd], &ldd, &ONE, &d[ib1 + ib1 * ldd], &ldd);
                SLC_DGEMM("T", "N", &dim2_j, &ncols_ib, &dim1_j, &ONE, &dwork[i3upri_j], &sdim_j,
                          &dwork[iwrk_j], &dim1_j, &ZERO, &dwork[itmp_j], &dim2_j);
                SLC_DGEMM("T", "N", &dim2_j, &ncols_ib, &dim2_j, &ONE, &dwork[i3lori_j], &sdim_j,
                          &d[m1 + ij1 + ib1 * ldd], &ldd, &ONE, &dwork[itmp_j], &dim2_j);
                SLC_DLACPY("F", &dim2_j, &ncols_ib, &dwork[itmp_j], &dim2_j, &d[m1 + ij1 + ib1 * ldd], &ldd);

                SLC_DLACPY("F", &dim1_j, &ncols_ij, &d[ib1 + (m1 + ij1) * ldd], &ldd, &dwork[iwrk_j], &dim1_j);
                SLC_DGEMM("T", "N", &dim1_j, &ncols_ij, &dim1_j, &ONE, &dwork[i3uple_j], &sdim_j,
                          &dwork[iwrk_j], &dim1_j, &ZERO, &d[ib1 + (m1 + ij1) * ldd], &ldd);
                SLC_DGEMM("T", "N", &dim1_j, &ncols_ij, &dim2_j, &ONE, &dwork[i3lole_j], &sdim_j,
                          &d[m1 + ij1 + (m1 + ij1) * ldd], &ldd, &ONE, &d[ib1 + (m1 + ij1) * ldd], &ldd);
                SLC_DGEMM("T", "N", &dim2_j, &ncols_ij, &dim1_j, &ONE, &dwork[i3upri_j], &sdim_j,
                          &dwork[iwrk_j], &dim1_j, &ZERO, &dwork[itmp_j], &dim2_j);
                SLC_DGEMM("T", "N", &dim2_j, &ncols_ij, &dim2_j, &ONE, &dwork[i3lori_j], &sdim_j,
                          &d[m1 + ij1 + (m1 + ij1) * ldd], &ldd, &ONE, &dwork[itmp_j], &dim2_j);
                SLC_DLACPY("F", &dim2_j, &ncols_ij, &dwork[itmp_j], &dim2_j, &d[m1 + ij1 + (m1 + ij1) * ldd], &ldd);

                if (m2 > 0) {
                    SLC_DLACPY("F", &dim1_j, &m4, &d[ib1 + i2 * ldd], &ldd, &dwork[iwrk_j], &dim1_j);
                    SLC_DGEMM("T", "N", &dim1_j, &m4, &dim1_j, &ONE, &dwork[i3uple_j], &sdim_j,
                              &dwork[iwrk_j], &dim1_j, &ZERO, &d[ib1 + i2 * ldd], &ldd);
                    SLC_DGEMM("T", "N", &dim1_j, &m4, &dim2_j, &ONE, &dwork[i3lole_j], &sdim_j,
                              &d[m1 + ij1 + i2 * ldd], &ldd, &ONE, &d[ib1 + i2 * ldd], &ldd);
                    SLC_DGEMM("T", "N", &dim2_j, &m4, &dim1_j, &ONE, &dwork[i3upri_j], &sdim_j,
                              &dwork[iwrk_j], &dim1_j, &ZERO, &dwork[itmp_j], &dim2_j);
                    SLC_DGEMM("T", "N", &dim2_j, &m4, &dim2_j, &ONE, &dwork[i3lori_j], &sdim_j,
                              &d[m1 + ij1 + i2 * ldd], &ldd, &ONE, &dwork[itmp_j], &dim2_j);
                    SLC_DLACPY("F", &dim2_j, &m4, &dwork[itmp_j], &dim2_j, &d[m1 + ij1 + i2 * ldd], &ldd);
                }

                if (lcmpq1) {
                    SLC_DLACPY("F", &n, &dim1_j, &q1[ib1 * ldq1], &ldq1, &dwork[iwrk_j], &n);
                    SLC_DGEMM("N", "N", &n, &dim1_j, &dim1_j, &ONE, &dwork[iwrk_j], &n,
                              &dwork[i1uple_j], &sdim_j, &ZERO, &q1[ib1 * ldq1], &ldq1);
                    SLC_DGEMM("N", "N", &n, &dim1_j, &dim2_j, &ONE, &q1[(m1 + ij1) * ldq1], &ldq1,
                              &dwork[i1lole_j], &sdim_j, &ONE, &q1[ib1 * ldq1], &ldq1);
                    SLC_DGEMM("N", "N", &n, &dim2_j, &dim1_j, &ONE, &dwork[iwrk_j], &n,
                              &dwork[i1upri_j], &sdim_j, &ZERO, &dwork[itmp_j], &n);
                    SLC_DGEMM("N", "N", &n, &dim2_j, &dim2_j, &ONE, &q1[(m1 + ij1) * ldq1], &ldq1,
                              &dwork[i1lori_j], &sdim_j, &ONE, &dwork[itmp_j], &n);
                    SLC_DLACPY("F", &n, &dim2_j, &dwork[itmp_j], &n, &q1[(m1 + ij1) * ldq1], &ldq1);
                }

                if (lcmpq2) {
                    SLC_DLACPY("F", &n, &dim1_j, &q2[ib1 * ldq2], &ldq2, &dwork[iwrk_j], &n);
                    SLC_DGEMM("N", "N", &n, &dim1_j, &dim1_j, &ONE, &dwork[iwrk_j], &n,
                              &dwork[i2uple_j], &sdim_j, &ZERO, &q2[ib1 * ldq2], &ldq2);
                    SLC_DGEMM("N", "N", &n, &dim1_j, &dim2_j, &ONE, &q2[(m1 + ij1) * ldq2], &ldq2,
                              &dwork[i2lole_j], &sdim_j, &ONE, &q2[ib1 * ldq2], &ldq2);
                    SLC_DGEMM("N", "N", &n, &dim2_j, &dim1_j, &ONE, &dwork[iwrk_j], &n,
                              &dwork[i2upri_j], &sdim_j, &ZERO, &dwork[itmp_j], &n);
                    SLC_DGEMM("N", "N", &n, &dim2_j, &dim2_j, &ONE, &q2[(m1 + ij1) * ldq2], &ldq2,
                              &dwork[i2lori_j], &sdim_j, &ONE, &dwork[itmp_j], &n);
                    SLC_DLACPY("F", &n, &dim2_j, &dwork[itmp_j], &n, &q2[(m1 + ij1) * ldq2], &ldq2);
                }

                if (lcmpq3) {
                    SLC_DLACPY("F", &n, &dim1_j, &q3[ib1 * ldq3], &ldq3, &dwork[iwrk_j], &n);
                    SLC_DGEMM("N", "N", &n, &dim1_j, &dim1_j, &ONE, &dwork[iwrk_j], &n,
                              &dwork[i3uple_j], &sdim_j, &ZERO, &q3[ib1 * ldq3], &ldq3);
                    SLC_DGEMM("N", "N", &n, &dim1_j, &dim2_j, &ONE, &q3[(m1 + ij1) * ldq3], &ldq3,
                              &dwork[i3lole_j], &sdim_j, &ONE, &q3[ib1 * ldq3], &ldq3);
                    SLC_DGEMM("N", "N", &n, &dim2_j, &dim1_j, &ONE, &dwork[iwrk_j], &n,
                              &dwork[i3upri_j], &sdim_j, &ZERO, &dwork[itmp_j], &n);
                    SLC_DGEMM("N", "N", &n, &dim2_j, &dim2_j, &ONE, &q3[(m1 + ij1) * ldq3], &ldq3,
                              &dwork[i3lori_j], &sdim_j, &ONE, &dwork[itmp_j], &n);
                    SLC_DLACPY("F", &n, &dim2_j, &dwork[itmp_j], &n, &q3[(m1 + ij1) * ldq3], &ldq3);
                }
            } else if (dim1_j == 1 && dim2_j == 2) {
                i32 nr_minus_1 = nr_j - 1;
                SLC_DCOPY(&nr_j, &a[ib1 * lda], &int1, &dwork[iwrk_j], &int1);
                SLC_DGEMV("N", &nr_minus_1, &dim2_j, &ONE, &a[(m1 + ij1) * lda], &lda,
                          &dwork[i2lole_j], &int1, &dwork[i2uple_j], &a[ib1 * lda], &int1);
                a[nr_j - 1 + ib1 * lda] = dwork[i2uple_j] * a[nr_j - 1 + ib1 * lda];
                SLC_DGEMM("N", "N", &nr_minus_1, &dim2_j, &dim2_j, &ONE, &a[(m1 + ij1) * lda], &lda,
                          &dwork[i2lori_j], &sdim_j, &ZERO, &dwork[itmp_j], &nr_j);
                dwork[itmp_j + nr_j - 1] = ZERO;
                dwork[itmp_j + 2 * nr_j - 1] = ZERO;
                SLC_DAXPY(&nr_j, &dwork[i2upri_j], &dwork[iwrk_j], &int1, &dwork[itmp_j], &int1);
                SLC_DAXPY(&nr_j, &dwork[i2upri_j + sdim_j], &dwork[iwrk_j], &int1, &dwork[itmp_j + nr_j], &int1);
                SLC_DLACPY("F", &nr_j, &dim2_j, &dwork[itmp_j], &nr_j, &a[(m1 + ij1) * lda], &lda);

                SLC_DCOPY(&nrow_j, &a[i1 + ib1 * lda], &int1, &dwork[iwrk_j], &int1);
                SLC_DGEMV("N", &nrow_j, &dim2_j, &ONE, &a[i1 + (m1 + ij1) * lda], &lda,
                          &dwork[i2lole_j], &int1, &dwork[i2uple_j], &a[i1 + ib1 * lda], &int1);
                SLC_DGEMM("N", "N", &nrow_j, &dim2_j, &dim2_j, &ONE, &a[i1 + (m1 + ij1) * lda], &lda,
                          &dwork[i2lori_j], &sdim_j, &ZERO, &dwork[itmp_j], &nrow_j);
                SLC_DAXPY(&nrow_j, &dwork[i2upri_j], &dwork[iwrk_j], &int1, &dwork[itmp_j], &int1);
                SLC_DAXPY(&nrow_j, &dwork[i2upri_j + sdim_j], &dwork[iwrk_j], &int1, &dwork[itmp_j + nrow_j], &int1);
                SLC_DLACPY("F", &nrow_j, &dim2_j, &dwork[itmp_j], &nrow_j, &a[i1 + (m1 + ij1) * lda], &lda);

                i32 ncols_ib = m1 - ib1;
                SLC_DCOPY(&ncols_ib, &a[ib1 + ib1 * lda], &lda, &dwork[iwrk_j], &int1);
                SLC_DGEMV("T", &dim2_j, &ncols_ib, &ONE, &a[m1 + ij1 + ib1 * lda], &lda,
                          &dwork[i3lole_j], &int1, &dwork[i3uple_j], &a[ib1 + ib1 * lda], &lda);
                SLC_DGEMM("T", "N", &dim2_j, &ncols_ib, &dim2_j, &ONE, &dwork[i3lori_j], &sdim_j,
                          &a[m1 + ij1 + ib1 * lda], &lda, &ZERO, &dwork[itmp_j], &dim2_j);
                SLC_DAXPY(&ncols_ib, &dwork[i3upri_j], &dwork[iwrk_j], &int1, &dwork[itmp_j], &dim2_j);
                SLC_DAXPY(&ncols_ib, &dwork[i3upri_j + sdim_j], &dwork[iwrk_j], &int1, &dwork[itmp_j + 1], &dim2_j);
                SLC_DLACPY("F", &dim2_j, &ncols_ib, &dwork[itmp_j], &dim2_j, &a[m1 + ij1 + ib1 * lda], &lda);

                i32 ncols_ij = m1 - ij1;
                SLC_DCOPY(&ncols_ij, &a[ib1 + (m1 + ij1) * lda], &lda, &dwork[iwrk_j], &int1);
                SLC_DGEMV("T", &dim2_j, &ncols_ij, &ONE, &a[m1 + ij1 + (m1 + ij1) * lda], &lda,
                          &dwork[i3lole_j], &int1, &dwork[i3uple_j], &a[ib1 + (m1 + ij1) * lda], &lda);
                SLC_DGEMM("T", "N", &dim2_j, &ncols_ij, &dim2_j, &ONE, &dwork[i3lori_j], &sdim_j,
                          &a[m1 + ij1 + (m1 + ij1) * lda], &lda, &ZERO, &dwork[itmp_j], &dim2_j);
                SLC_DAXPY(&ncols_ij, &dwork[i3upri_j], &dwork[iwrk_j], &int1, &dwork[itmp_j], &dim2_j);
                SLC_DAXPY(&ncols_ij, &dwork[i3upri_j + sdim_j], &dwork[iwrk_j], &int1, &dwork[itmp_j + 1], &dim2_j);
                SLC_DLACPY("F", &dim2_j, &ncols_ij, &dwork[itmp_j], &dim2_j, &a[m1 + ij1 + (m1 + ij1) * lda], &lda);

                if (m2 > 0) {
                    SLC_DCOPY(&m4, &a[ib1 + i2 * lda], &lda, &dwork[iwrk_j], &int1);
                    SLC_DGEMV("T", &dim2_j, &m4, &ONE, &a[m1 + ij1 + i2 * lda], &lda,
                              &dwork[i3lole_j], &int1, &dwork[i3uple_j], &a[ib1 + i2 * lda], &lda);
                    SLC_DGEMM("T", "N", &dim2_j, &m4, &dim2_j, &ONE, &dwork[i3lori_j], &sdim_j,
                              &a[m1 + ij1 + i2 * lda], &lda, &ZERO, &dwork[itmp_j], &dim2_j);
                    SLC_DAXPY(&m4, &dwork[i3upri_j], &dwork[iwrk_j], &int1, &dwork[itmp_j], &dim2_j);
                    SLC_DAXPY(&m4, &dwork[i3upri_j + sdim_j], &dwork[iwrk_j], &int1, &dwork[itmp_j + 1], &dim2_j);
                    SLC_DLACPY("F", &dim2_j, &m4, &dwork[itmp_j], &dim2_j, &a[m1 + ij1 + i2 * lda], &lda);
                }

                SLC_DCOPY(&nr_j, &b[ib1 * ldb], &int1, &dwork[iwrk_j], &int1);
                SLC_DGEMV("N", &nr_minus_1, &dim2_j, &ONE, &b[(m1 + ij1) * ldb], &ldb,
                          &dwork[i1lole_j], &int1, &dwork[i1uple_j], &b[ib1 * ldb], &int1);
                b[nr_j - 1 + ib1 * ldb] = dwork[i1uple_j] * b[nr_j - 1 + ib1 * ldb];
                SLC_DGEMM("N", "N", &nr_minus_1, &dim2_j, &dim2_j, &ONE, &b[(m1 + ij1) * ldb], &ldb,
                          &dwork[i1lori_j], &sdim_j, &ZERO, &dwork[itmp_j], &nr_j);
                dwork[itmp_j + nr_j - 1] = ZERO;
                dwork[itmp_j + 2 * nr_j - 1] = ZERO;
                SLC_DAXPY(&nr_j, &dwork[i1upri_j], &dwork[iwrk_j], &int1, &dwork[itmp_j], &int1);
                SLC_DAXPY(&nr_j, &dwork[i1upri_j + sdim_j], &dwork[iwrk_j], &int1, &dwork[itmp_j + nr_j], &int1);
                SLC_DLACPY("F", &nr_j, &dim2_j, &dwork[itmp_j], &nr_j, &b[(m1 + ij1) * ldb], &ldb);

                SLC_DCOPY(&nrow_j, &b[i1 + ib1 * ldb], &int1, &dwork[iwrk_j], &int1);
                SLC_DGEMV("N", &nrow_j, &dim2_j, &ONE, &b[i1 + (m1 + ij1) * ldb], &ldb,
                          &dwork[i1lole_j], &int1, &dwork[i1uple_j], &b[i1 + ib1 * ldb], &int1);
                SLC_DGEMM("N", "N", &nrow_j, &dim2_j, &dim2_j, &ONE, &b[i1 + (m1 + ij1) * ldb], &ldb,
                          &dwork[i1lori_j], &sdim_j, &ZERO, &dwork[itmp_j], &nrow_j);
                SLC_DAXPY(&nrow_j, &dwork[i1upri_j], &dwork[iwrk_j], &int1, &dwork[itmp_j], &int1);
                SLC_DAXPY(&nrow_j, &dwork[i1upri_j + sdim_j], &dwork[iwrk_j], &int1, &dwork[itmp_j + nrow_j], &int1);
                SLC_DLACPY("F", &nrow_j, &dim2_j, &dwork[itmp_j], &nrow_j, &b[i1 + (m1 + ij1) * ldb], &ldb);

                SLC_DCOPY(&ncols_ib, &b[ib1 + ib1 * ldb], &ldb, &dwork[iwrk_j], &int1);
                SLC_DGEMV("T", &dim2_j, &ncols_ib, &ONE, &b[m1 + ij1 + ib1 * ldb], &ldb,
                          &dwork[i2lole_j], &int1, &dwork[i2uple_j], &b[ib1 + ib1 * ldb], &ldb);
                SLC_DGEMM("T", "N", &dim2_j, &ncols_ib, &dim2_j, &ONE, &dwork[i2lori_j], &sdim_j,
                          &b[m1 + ij1 + ib1 * ldb], &ldb, &ZERO, &dwork[itmp_j], &dim2_j);
                SLC_DAXPY(&ncols_ib, &dwork[i2upri_j], &dwork[iwrk_j], &int1, &dwork[itmp_j], &dim2_j);
                SLC_DAXPY(&ncols_ib, &dwork[i2upri_j + sdim_j], &dwork[iwrk_j], &int1, &dwork[itmp_j + 1], &dim2_j);
                SLC_DLACPY("F", &dim2_j, &ncols_ib, &dwork[itmp_j], &dim2_j, &b[m1 + ij1 + ib1 * ldb], &ldb);

                SLC_DCOPY(&ncols_ij, &b[ib1 + (m1 + ij1) * ldb], &ldb, &dwork[iwrk_j], &int1);
                SLC_DGEMV("T", &dim2_j, &ncols_ij, &ONE, &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb,
                          &dwork[i2lole_j], &int1, &dwork[i2uple_j], &b[ib1 + (m1 + ij1) * ldb], &ldb);
                SLC_DGEMM("T", "N", &dim2_j, &ncols_ij, &dim2_j, &ONE, &dwork[i2lori_j], &sdim_j,
                          &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb, &ZERO, &dwork[itmp_j], &dim2_j);
                SLC_DAXPY(&ncols_ij, &dwork[i2upri_j], &dwork[iwrk_j], &int1, &dwork[itmp_j], &dim2_j);
                SLC_DAXPY(&ncols_ij, &dwork[i2upri_j + sdim_j], &dwork[iwrk_j], &int1, &dwork[itmp_j + 1], &dim2_j);
                SLC_DLACPY("F", &dim2_j, &ncols_ij, &dwork[itmp_j], &dim2_j, &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb);

                if (m2 > 0) {
                    SLC_DCOPY(&m4, &b[ib1 + i2 * ldb], &ldb, &dwork[iwrk_j], &int1);
                    SLC_DGEMV("T", &dim2_j, &m4, &ONE, &b[m1 + ij1 + i2 * ldb], &ldb,
                              &dwork[i2lole_j], &int1, &dwork[i2uple_j], &b[ib1 + i2 * ldb], &ldb);
                    SLC_DGEMM("T", "N", &dim2_j, &m4, &dim2_j, &ONE, &dwork[i2lori_j], &sdim_j,
                              &b[m1 + ij1 + i2 * ldb], &ldb, &ZERO, &dwork[itmp_j], &dim2_j);
                    SLC_DAXPY(&m4, &dwork[i2upri_j], &dwork[iwrk_j], &int1, &dwork[itmp_j], &dim2_j);
                    SLC_DAXPY(&m4, &dwork[i2upri_j + sdim_j], &dwork[iwrk_j], &int1, &dwork[itmp_j + 1], &dim2_j);
                    SLC_DLACPY("F", &dim2_j, &m4, &dwork[itmp_j], &dim2_j, &b[m1 + ij1 + i2 * ldb], &ldb);
                }

                SLC_DCOPY(&nr_j, &d[ib1 * ldd], &int1, &dwork[iwrk_j], &int1);
                SLC_DGEMV("N", &nr_minus_1, &dim2_j, &ONE, &d[(m1 + ij1) * ldd], &ldd,
                          &dwork[i1lole_j], &int1, &dwork[i1uple_j], &d[ib1 * ldd], &int1);
                d[nr_j - 1 + ib1 * ldd] = dwork[i1uple_j] * d[nr_j - 1 + ib1 * ldd];
                SLC_DGEMM("N", "N", &nr_minus_1, &dim2_j, &dim2_j, &ONE, &d[(m1 + ij1) * ldd], &ldd,
                          &dwork[i1lori_j], &sdim_j, &ZERO, &dwork[itmp_j], &nr_j);
                dwork[itmp_j + nr_j - 1] = ZERO;
                dwork[itmp_j + 2 * nr_j - 1] = ZERO;
                SLC_DAXPY(&nr_j, &dwork[i1upri_j], &dwork[iwrk_j], &int1, &dwork[itmp_j], &int1);
                SLC_DAXPY(&nr_j, &dwork[i1upri_j + sdim_j], &dwork[iwrk_j], &int1, &dwork[itmp_j + nr_j], &int1);
                SLC_DLACPY("F", &nr_j, &dim2_j, &dwork[itmp_j], &nr_j, &d[(m1 + ij1) * ldd], &ldd);

                SLC_DCOPY(&nrow_j, &d[i1 + ib1 * ldd], &int1, &dwork[iwrk_j], &int1);
                SLC_DGEMV("N", &nrow_j, &dim2_j, &ONE, &d[i1 + (m1 + ij1) * ldd], &ldd,
                          &dwork[i1lole_j], &int1, &dwork[i1uple_j], &d[i1 + ib1 * ldd], &int1);
                SLC_DGEMM("N", "N", &nrow_j, &dim2_j, &dim2_j, &ONE, &d[i1 + (m1 + ij1) * ldd], &ldd,
                          &dwork[i1lori_j], &sdim_j, &ZERO, &dwork[itmp_j], &nrow_j);
                SLC_DAXPY(&nrow_j, &dwork[i1upri_j], &dwork[iwrk_j], &int1, &dwork[itmp_j], &int1);
                SLC_DAXPY(&nrow_j, &dwork[i1upri_j + sdim_j], &dwork[iwrk_j], &int1, &dwork[itmp_j + nrow_j], &int1);
                SLC_DLACPY("F", &nrow_j, &dim2_j, &dwork[itmp_j], &nrow_j, &d[i1 + (m1 + ij1) * ldd], &ldd);

                SLC_DCOPY(&ncols_ib, &d[ib1 + ib1 * ldd], &ldd, &dwork[iwrk_j], &int1);
                SLC_DGEMV("T", &dim2_j, &ncols_ib, &ONE, &d[m1 + ij1 + ib1 * ldd], &ldd,
                          &dwork[i3lole_j], &int1, &dwork[i3uple_j], &d[ib1 + ib1 * ldd], &ldd);
                SLC_DGEMM("T", "N", &dim2_j, &ncols_ib, &dim2_j, &ONE, &dwork[i3lori_j], &sdim_j,
                          &d[m1 + ij1 + ib1 * ldd], &ldd, &ZERO, &dwork[itmp_j], &dim2_j);
                SLC_DAXPY(&ncols_ib, &dwork[i3upri_j], &dwork[iwrk_j], &int1, &dwork[itmp_j], &dim2_j);
                SLC_DAXPY(&ncols_ib, &dwork[i3upri_j + sdim_j], &dwork[iwrk_j], &int1, &dwork[itmp_j + 1], &dim2_j);
                SLC_DLACPY("F", &dim2_j, &ncols_ib, &dwork[itmp_j], &dim2_j, &d[m1 + ij1 + ib1 * ldd], &ldd);

                SLC_DCOPY(&ncols_ij, &d[ib1 + (m1 + ij1) * ldd], &ldd, &dwork[iwrk_j], &int1);
                SLC_DGEMV("T", &dim2_j, &ncols_ij, &ONE, &d[m1 + ij1 + (m1 + ij1) * ldd], &ldd,
                          &dwork[i3lole_j], &int1, &dwork[i3uple_j], &d[ib1 + (m1 + ij1) * ldd], &ldd);
                SLC_DGEMM("T", "N", &dim2_j, &ncols_ij, &dim2_j, &ONE, &dwork[i3lori_j], &sdim_j,
                          &d[m1 + ij1 + (m1 + ij1) * ldd], &ldd, &ZERO, &dwork[itmp_j], &dim2_j);
                SLC_DAXPY(&ncols_ij, &dwork[i3upri_j], &dwork[iwrk_j], &int1, &dwork[itmp_j], &dim2_j);
                SLC_DAXPY(&ncols_ij, &dwork[i3upri_j + sdim_j], &dwork[iwrk_j], &int1, &dwork[itmp_j + 1], &dim2_j);
                SLC_DLACPY("F", &dim2_j, &ncols_ij, &dwork[itmp_j], &dim2_j, &d[m1 + ij1 + (m1 + ij1) * ldd], &ldd);

                if (m2 > 0) {
                    SLC_DCOPY(&m4, &d[ib1 + i2 * ldd], &ldd, &dwork[iwrk_j], &int1);
                    SLC_DGEMV("T", &dim2_j, &m4, &ONE, &d[m1 + ij1 + i2 * ldd], &ldd,
                              &dwork[i3lole_j], &int1, &dwork[i3uple_j], &d[ib1 + i2 * ldd], &ldd);
                    SLC_DGEMM("T", "N", &dim2_j, &m4, &dim2_j, &ONE, &dwork[i3lori_j], &sdim_j,
                              &d[m1 + ij1 + i2 * ldd], &ldd, &ZERO, &dwork[itmp_j], &dim2_j);
                    SLC_DAXPY(&m4, &dwork[i3upri_j], &dwork[iwrk_j], &int1, &dwork[itmp_j], &dim2_j);
                    SLC_DAXPY(&m4, &dwork[i3upri_j + sdim_j], &dwork[iwrk_j], &int1, &dwork[itmp_j + 1], &dim2_j);
                    SLC_DLACPY("F", &dim2_j, &m4, &dwork[itmp_j], &dim2_j, &d[m1 + ij1 + i2 * ldd], &ldd);
                }

                if (lcmpq1) {
                    SLC_DCOPY(&n, &q1[ib1 * ldq1], &int1, &dwork[iwrk_j], &int1);
                    SLC_DGEMV("N", &n, &dim2_j, &ONE, &q1[(m1 + ij1) * ldq1], &ldq1,
                              &dwork[i1lole_j], &int1, &dwork[i1uple_j], &q1[ib1 * ldq1], &int1);
                    SLC_DGEMM("N", "N", &n, &dim2_j, &dim2_j, &ONE, &q1[(m1 + ij1) * ldq1], &ldq1,
                              &dwork[i1lori_j], &sdim_j, &ZERO, &dwork[itmp_j], &n);
                    SLC_DAXPY(&n, &dwork[i1upri_j], &dwork[iwrk_j], &int1, &dwork[itmp_j], &int1);
                    SLC_DAXPY(&n, &dwork[i1upri_j + sdim_j], &dwork[iwrk_j], &int1, &dwork[itmp_j + n], &int1);
                    SLC_DLACPY("F", &n, &dim2_j, &dwork[itmp_j], &n, &q1[(m1 + ij1) * ldq1], &ldq1);
                }

                if (lcmpq2) {
                    SLC_DCOPY(&n, &q2[ib1 * ldq2], &int1, &dwork[iwrk_j], &int1);
                    SLC_DGEMV("N", &n, &dim2_j, &ONE, &q2[(m1 + ij1) * ldq2], &ldq2,
                              &dwork[i2lole_j], &int1, &dwork[i2uple_j], &q2[ib1 * ldq2], &int1);
                    SLC_DGEMM("N", "N", &n, &dim2_j, &dim2_j, &ONE, &q2[(m1 + ij1) * ldq2], &ldq2,
                              &dwork[i2lori_j], &sdim_j, &ZERO, &dwork[itmp_j], &n);
                    SLC_DAXPY(&n, &dwork[i2upri_j], &dwork[iwrk_j], &int1, &dwork[itmp_j], &int1);
                    SLC_DAXPY(&n, &dwork[i2upri_j + sdim_j], &dwork[iwrk_j], &int1, &dwork[itmp_j + n], &int1);
                    SLC_DLACPY("F", &n, &dim2_j, &dwork[itmp_j], &n, &q2[(m1 + ij1) * ldq2], &ldq2);
                }

                if (lcmpq3) {
                    SLC_DCOPY(&n, &q3[ib1 * ldq3], &int1, &dwork[iwrk_j], &int1);
                    SLC_DGEMV("N", &n, &dim2_j, &ONE, &q3[(m1 + ij1) * ldq3], &ldq3,
                              &dwork[i3lole_j], &int1, &dwork[i3uple_j], &q3[ib1 * ldq3], &int1);
                    SLC_DGEMM("N", "N", &n, &dim2_j, &dim2_j, &ONE, &q3[(m1 + ij1) * ldq3], &ldq3,
                              &dwork[i3lori_j], &sdim_j, &ZERO, &dwork[itmp_j], &n);
                    SLC_DAXPY(&n, &dwork[i3upri_j], &dwork[iwrk_j], &int1, &dwork[itmp_j], &int1);
                    SLC_DAXPY(&n, &dwork[i3upri_j + sdim_j], &dwork[iwrk_j], &int1, &dwork[itmp_j + n], &int1);
                    SLC_DLACPY("F", &n, &dim2_j, &dwork[itmp_j], &n, &q3[(m1 + ij1) * ldq3], &ldq3);
                }
            } else if (dim1_j == 2 && dim2_j == 1) {
                i32 nr_minus_1 = nr_j - 1;
                i32 ncols_ib = m1 - ib1;
                i32 ncols_ij = m1 - ij1;

                SLC_DLACPY("F", &nr_j, &dim1_j, &a[ib1 * lda], &lda, &dwork[iwrk_j], &nr_j);
                SLC_DGEMM("N", "N", &nr_j, &dim1_j, &dim1_j, &ONE, &dwork[iwrk_j], &nr_j,
                          &dwork[i2uple_j], &sdim_j, &ZERO, &a[ib1 * lda], &lda);
                SLC_DAXPY(&nr_minus_1, &dwork[i2lole_j], &a[(m1 + ij1) * lda], &int1, &a[ib1 * lda], &int1);
                SLC_DAXPY(&nr_minus_1, &dwork[i2lole_j + sdim_j], &a[(m1 + ij1) * lda], &int1, &a[1 + ib1 * lda], &int1);
                a[nr_j - 1 + (m1 + ij1) * lda] = ZERO;
                SLC_DGEMV("N", &nr_j, &dim1_j, &ONE, &dwork[iwrk_j], &nr_j, &dwork[i2upri_j], &int1,
                          &dwork[i2lori_j], &a[(m1 + ij1) * lda], &int1);

                if (nrow_j > 0) {
                    SLC_DLACPY("F", &nrow_j, &dim1_j, &a[i1 + ib1 * lda], &lda, &dwork[iwrk_j], &nrow_j);
                    SLC_DGEMM("N", "N", &nrow_j, &dim1_j, &dim1_j, &ONE, &dwork[iwrk_j], &nrow_j,
                              &dwork[i2uple_j], &sdim_j, &ZERO, &a[i1 + ib1 * lda], &lda);
                    SLC_DAXPY(&nrow_j, &dwork[i2lole_j], &a[i1 + (m1 + ij1) * lda], &int1, &a[i1 + ib1 * lda], &int1);
                    SLC_DAXPY(&nrow_j, &dwork[i2lole_j + sdim_j], &a[i1 + (m1 + ij1) * lda], &int1, &a[i1 + 1 + ib1 * lda], &int1);
                    SLC_DGEMV("N", &nrow_j, &dim1_j, &ONE, &dwork[iwrk_j], &nrow_j, &dwork[i2upri_j], &int1,
                              &dwork[i2lori_j], &a[i1 + (m1 + ij1) * lda], &int1);
                }

                SLC_DLACPY("F", &dim1_j, &ncols_ib, &a[ib1 + ib1 * lda], &lda, &dwork[iwrk_j], &dim1_j);
                SLC_DGEMM("T", "N", &dim1_j, &ncols_ib, &dim1_j, &ONE, &dwork[i3uple_j], &sdim_j,
                          &dwork[iwrk_j], &dim1_j, &ZERO, &a[ib1 + ib1 * lda], &lda);
                SLC_DAXPY(&ncols_ib, &dwork[i3lole_j], &a[m1 + ij1 + ib1 * lda], &lda, &a[ib1 + ib1 * lda], &lda);
                SLC_DAXPY(&ncols_ib, &dwork[i3lole_j + sdim_j], &a[m1 + ij1 + ib1 * lda], &lda, &a[ib1 + 1 + ib1 * lda], &lda);
                SLC_DGEMV("T", &dim1_j, &ncols_ib, &ONE, &dwork[iwrk_j], &dim1_j, &dwork[i3upri_j], &int1,
                          &dwork[i3lori_j], &a[m1 + ij1 + ib1 * lda], &lda);

                SLC_DLACPY("F", &dim1_j, &ncols_ij, &a[ib1 + (m1 + ij1) * lda], &lda, &dwork[iwrk_j], &dim1_j);
                SLC_DGEMM("T", "N", &dim1_j, &ncols_ij, &dim1_j, &ONE, &dwork[i3uple_j], &sdim_j,
                          &dwork[iwrk_j], &dim1_j, &ZERO, &a[ib1 + (m1 + ij1) * lda], &lda);
                SLC_DAXPY(&ncols_ij, &dwork[i3lole_j], &a[m1 + ij1 + (m1 + ij1) * lda], &lda, &a[ib1 + (m1 + ij1) * lda], &lda);
                SLC_DAXPY(&ncols_ij, &dwork[i3lole_j + sdim_j], &a[m1 + ij1 + (m1 + ij1) * lda], &lda, &a[ib1 + 1 + (m1 + ij1) * lda], &lda);
                SLC_DGEMV("T", &dim1_j, &ncols_ij, &ONE, &dwork[iwrk_j], &dim1_j, &dwork[i3upri_j], &int1,
                          &dwork[i3lori_j], &a[m1 + ij1 + (m1 + ij1) * lda], &lda);

                if (m2 > 0) {
                    SLC_DLACPY("F", &dim1_j, &m4, &a[ib1 + i2 * lda], &lda, &dwork[iwrk_j], &dim1_j);
                    SLC_DGEMM("T", "N", &dim1_j, &m4, &dim1_j, &ONE, &dwork[i3uple_j], &sdim_j,
                              &dwork[iwrk_j], &dim1_j, &ZERO, &a[ib1 + i2 * lda], &lda);
                    SLC_DAXPY(&m4, &dwork[i3lole_j], &a[m1 + ij1 + i2 * lda], &lda, &a[ib1 + i2 * lda], &lda);
                    SLC_DAXPY(&m4, &dwork[i3lole_j + sdim_j], &a[m1 + ij1 + i2 * lda], &lda, &a[ib1 + 1 + i2 * lda], &lda);
                    SLC_DGEMV("T", &dim1_j, &m4, &ONE, &dwork[iwrk_j], &dim1_j, &dwork[i3upri_j], &int1,
                              &dwork[i3lori_j], &a[m1 + ij1 + i2 * lda], &lda);
                }

                SLC_DLACPY("F", &nr_j, &dim1_j, &b[ib1 * ldb], &ldb, &dwork[iwrk_j], &nr_j);
                SLC_DGEMM("N", "N", &nr_j, &dim1_j, &dim1_j, &ONE, &dwork[iwrk_j], &nr_j,
                          &dwork[i1uple_j], &sdim_j, &ZERO, &b[ib1 * ldb], &ldb);
                SLC_DAXPY(&nr_minus_1, &dwork[i1lole_j], &b[(m1 + ij1) * ldb], &int1, &b[ib1 * ldb], &int1);
                SLC_DAXPY(&nr_minus_1, &dwork[i1lole_j + sdim_j], &b[(m1 + ij1) * ldb], &int1, &b[1 + ib1 * ldb], &int1);
                b[nr_j - 1 + (m1 + ij1) * ldb] = ZERO;
                SLC_DGEMV("N", &nr_j, &dim1_j, &ONE, &dwork[iwrk_j], &nr_j, &dwork[i1upri_j], &int1,
                          &dwork[i1lori_j], &b[(m1 + ij1) * ldb], &int1);

                if (nrow_j > 0) {
                    SLC_DLACPY("F", &nrow_j, &dim1_j, &b[i1 + ib1 * ldb], &ldb, &dwork[iwrk_j], &nrow_j);
                    SLC_DGEMM("N", "N", &nrow_j, &dim1_j, &dim1_j, &ONE, &dwork[iwrk_j], &nrow_j,
                              &dwork[i1uple_j], &sdim_j, &ZERO, &b[i1 + ib1 * ldb], &ldb);
                    SLC_DAXPY(&nrow_j, &dwork[i1lole_j], &b[i1 + (m1 + ij1) * ldb], &int1, &b[i1 + ib1 * ldb], &int1);
                    SLC_DAXPY(&nrow_j, &dwork[i1lole_j + sdim_j], &b[i1 + (m1 + ij1) * ldb], &int1, &b[i1 + 1 + ib1 * ldb], &int1);
                    SLC_DGEMV("N", &nrow_j, &dim1_j, &ONE, &dwork[iwrk_j], &nrow_j, &dwork[i1upri_j], &int1,
                              &dwork[i1lori_j], &b[i1 + (m1 + ij1) * ldb], &int1);
                }

                SLC_DLACPY("F", &dim1_j, &ncols_ib, &b[ib1 + ib1 * ldb], &ldb, &dwork[iwrk_j], &dim1_j);
                SLC_DGEMM("T", "N", &dim1_j, &ncols_ib, &dim1_j, &ONE, &dwork[i2uple_j], &sdim_j,
                          &dwork[iwrk_j], &dim1_j, &ZERO, &b[ib1 + ib1 * ldb], &ldb);
                SLC_DAXPY(&ncols_ib, &dwork[i2lole_j], &b[m1 + ij1 + ib1 * ldb], &ldb, &b[ib1 + ib1 * ldb], &ldb);
                SLC_DAXPY(&ncols_ib, &dwork[i2lole_j + sdim_j], &b[m1 + ij1 + ib1 * ldb], &ldb, &b[ib1 + 1 + ib1 * ldb], &ldb);
                SLC_DGEMV("T", &dim1_j, &ncols_ib, &ONE, &dwork[iwrk_j], &dim1_j, &dwork[i2upri_j], &int1,
                          &dwork[i2lori_j], &b[m1 + ij1 + ib1 * ldb], &ldb);

                SLC_DLACPY("F", &dim1_j, &ncols_ij, &b[ib1 + (m1 + ij1) * ldb], &ldb, &dwork[iwrk_j], &dim1_j);
                SLC_DGEMM("T", "N", &dim1_j, &ncols_ij, &dim1_j, &ONE, &dwork[i2uple_j], &sdim_j,
                          &dwork[iwrk_j], &dim1_j, &ZERO, &b[ib1 + (m1 + ij1) * ldb], &ldb);
                SLC_DAXPY(&ncols_ij, &dwork[i2lole_j], &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb, &b[ib1 + (m1 + ij1) * ldb], &ldb);
                SLC_DAXPY(&ncols_ij, &dwork[i2lole_j + sdim_j], &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb, &b[ib1 + 1 + (m1 + ij1) * ldb], &ldb);
                SLC_DGEMV("T", &dim1_j, &ncols_ij, &ONE, &dwork[iwrk_j], &dim1_j, &dwork[i2upri_j], &int1,
                          &dwork[i2lori_j], &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb);

                if (m2 > 0) {
                    SLC_DLACPY("F", &dim1_j, &m4, &b[ib1 + i2 * ldb], &ldb, &dwork[iwrk_j], &dim1_j);
                    SLC_DGEMM("T", "N", &dim1_j, &m4, &dim1_j, &ONE, &dwork[i2uple_j], &sdim_j,
                              &dwork[iwrk_j], &dim1_j, &ZERO, &b[ib1 + i2 * ldb], &ldb);
                    SLC_DAXPY(&m4, &dwork[i2lole_j], &b[m1 + ij1 + i2 * ldb], &ldb, &b[ib1 + i2 * ldb], &ldb);
                    SLC_DAXPY(&m4, &dwork[i2lole_j + sdim_j], &b[m1 + ij1 + i2 * ldb], &ldb, &b[ib1 + 1 + i2 * ldb], &ldb);
                    SLC_DGEMV("T", &dim1_j, &m4, &ONE, &dwork[iwrk_j], &dim1_j, &dwork[i2upri_j], &int1,
                              &dwork[i2lori_j], &b[m1 + ij1 + i2 * ldb], &ldb);
                }

                SLC_DLACPY("F", &nr_j, &dim1_j, &d[ib1 * ldd], &ldd, &dwork[iwrk_j], &nr_j);
                SLC_DGEMM("N", "N", &nr_j, &dim1_j, &dim1_j, &ONE, &dwork[iwrk_j], &nr_j,
                          &dwork[i1uple_j], &sdim_j, &ZERO, &d[ib1 * ldd], &ldd);
                SLC_DAXPY(&nr_minus_1, &dwork[i1lole_j], &d[(m1 + ij1) * ldd], &int1, &d[ib1 * ldd], &int1);
                SLC_DAXPY(&nr_minus_1, &dwork[i1lole_j + sdim_j], &d[(m1 + ij1) * ldd], &int1, &d[1 + ib1 * ldd], &int1);
                d[nr_j - 1 + (m1 + ij1) * ldd] = ZERO;
                SLC_DGEMV("N", &nr_j, &dim1_j, &ONE, &dwork[iwrk_j], &nr_j, &dwork[i1upri_j], &int1,
                          &dwork[i1lori_j], &d[(m1 + ij1) * ldd], &int1);

                if (nrow_j > 0) {
                    SLC_DLACPY("F", &nrow_j, &dim1_j, &d[i1 + ib1 * ldd], &ldd, &dwork[iwrk_j], &nrow_j);
                    SLC_DGEMM("N", "N", &nrow_j, &dim1_j, &dim1_j, &ONE, &dwork[iwrk_j], &nrow_j,
                              &dwork[i1uple_j], &sdim_j, &ZERO, &d[i1 + ib1 * ldd], &ldd);
                    SLC_DAXPY(&nrow_j, &dwork[i1lole_j], &d[i1 + (m1 + ij1) * ldd], &int1, &d[i1 + ib1 * ldd], &int1);
                    SLC_DAXPY(&nrow_j, &dwork[i1lole_j + sdim_j], &d[i1 + (m1 + ij1) * ldd], &int1, &d[i1 + 1 + ib1 * ldd], &int1);
                    SLC_DGEMV("N", &nrow_j, &dim1_j, &ONE, &dwork[iwrk_j], &nrow_j, &dwork[i1upri_j], &int1,
                              &dwork[i1lori_j], &d[i1 + (m1 + ij1) * ldd], &int1);
                }

                SLC_DLACPY("F", &dim1_j, &ncols_ib, &d[ib1 + ib1 * ldd], &ldd, &dwork[iwrk_j], &dim1_j);
                SLC_DGEMM("T", "N", &dim1_j, &ncols_ib, &dim1_j, &ONE, &dwork[i3uple_j], &sdim_j,
                          &dwork[iwrk_j], &dim1_j, &ZERO, &d[ib1 + ib1 * ldd], &ldd);
                SLC_DAXPY(&ncols_ib, &dwork[i3lole_j], &d[m1 + ij1 + ib1 * ldd], &ldd, &d[ib1 + ib1 * ldd], &ldd);
                SLC_DAXPY(&ncols_ib, &dwork[i3lole_j + sdim_j], &d[m1 + ij1 + ib1 * ldd], &ldd, &d[ib1 + 1 + ib1 * ldd], &ldd);
                SLC_DGEMV("T", &dim1_j, &ncols_ib, &ONE, &dwork[iwrk_j], &dim1_j, &dwork[i3upri_j], &int1,
                          &dwork[i3lori_j], &d[m1 + ij1 + ib1 * ldd], &ldd);

                SLC_DLACPY("F", &dim1_j, &ncols_ij, &d[ib1 + (m1 + ij1) * ldd], &ldd, &dwork[iwrk_j], &dim1_j);
                SLC_DGEMM("T", "N", &dim1_j, &ncols_ij, &dim1_j, &ONE, &dwork[i3uple_j], &sdim_j,
                          &dwork[iwrk_j], &dim1_j, &ZERO, &d[ib1 + (m1 + ij1) * ldd], &ldd);
                SLC_DAXPY(&ncols_ij, &dwork[i3lole_j], &d[m1 + ij1 + (m1 + ij1) * ldd], &ldd, &d[ib1 + (m1 + ij1) * ldd], &ldd);
                SLC_DAXPY(&ncols_ij, &dwork[i3lole_j + sdim_j], &d[m1 + ij1 + (m1 + ij1) * ldd], &ldd, &d[ib1 + 1 + (m1 + ij1) * ldd], &ldd);
                SLC_DGEMV("T", &dim1_j, &ncols_ij, &ONE, &dwork[iwrk_j], &dim1_j, &dwork[i3upri_j], &int1,
                          &dwork[i3lori_j], &d[m1 + ij1 + (m1 + ij1) * ldd], &ldd);

                if (m2 > 0) {
                    SLC_DLACPY("F", &dim1_j, &m4, &d[ib1 + i2 * ldd], &ldd, &dwork[iwrk_j], &dim1_j);
                    SLC_DGEMM("T", "N", &dim1_j, &m4, &dim1_j, &ONE, &dwork[i3uple_j], &sdim_j,
                              &dwork[iwrk_j], &dim1_j, &ZERO, &d[ib1 + i2 * ldd], &ldd);
                    SLC_DAXPY(&m4, &dwork[i3lole_j], &d[m1 + ij1 + i2 * ldd], &ldd, &d[ib1 + i2 * ldd], &ldd);
                    SLC_DAXPY(&m4, &dwork[i3lole_j + sdim_j], &d[m1 + ij1 + i2 * ldd], &ldd, &d[ib1 + 1 + i2 * ldd], &ldd);
                    SLC_DGEMV("T", &dim1_j, &m4, &ONE, &dwork[iwrk_j], &dim1_j, &dwork[i3upri_j], &int1,
                              &dwork[i3lori_j], &d[m1 + ij1 + i2 * ldd], &ldd);
                }

                if (lcmpq1) {
                    SLC_DLACPY("F", &n, &dim1_j, &q1[ib1 * ldq1], &ldq1, &dwork[iwrk_j], &n);
                    SLC_DGEMM("N", "N", &n, &dim1_j, &dim1_j, &ONE, &dwork[iwrk_j], &n,
                              &dwork[i1uple_j], &sdim_j, &ZERO, &q1[ib1 * ldq1], &ldq1);
                    SLC_DAXPY(&n, &dwork[i1lole_j], &q1[(m1 + ij1) * ldq1], &int1, &q1[ib1 * ldq1], &int1);
                    SLC_DAXPY(&n, &dwork[i1lole_j + sdim_j], &q1[(m1 + ij1) * ldq1], &int1, &q1[1 + ib1 * ldq1], &int1);
                    SLC_DGEMV("N", &n, &dim1_j, &ONE, &dwork[iwrk_j], &n, &dwork[i1upri_j], &int1,
                              &dwork[i1lori_j], &q1[(m1 + ij1) * ldq1], &int1);
                }

                if (lcmpq2) {
                    SLC_DLACPY("F", &n, &dim1_j, &q2[ib1 * ldq2], &ldq2, &dwork[iwrk_j], &n);
                    SLC_DGEMM("N", "N", &n, &dim1_j, &dim1_j, &ONE, &dwork[iwrk_j], &n,
                              &dwork[i2uple_j], &sdim_j, &ZERO, &q2[ib1 * ldq2], &ldq2);
                    SLC_DAXPY(&n, &dwork[i2lole_j], &q2[(m1 + ij1) * ldq2], &int1, &q2[ib1 * ldq2], &int1);
                    SLC_DAXPY(&n, &dwork[i2lole_j + sdim_j], &q2[(m1 + ij1) * ldq2], &int1, &q2[1 + ib1 * ldq2], &int1);
                    SLC_DGEMV("N", &n, &dim1_j, &ONE, &dwork[iwrk_j], &n, &dwork[i2upri_j], &int1,
                              &dwork[i2lori_j], &q2[(m1 + ij1) * ldq2], &int1);
                }

                if (lcmpq3) {
                    SLC_DLACPY("F", &n, &dim1_j, &q3[ib1 * ldq3], &ldq3, &dwork[iwrk_j], &n);
                    SLC_DGEMM("N", "N", &n, &dim1_j, &dim1_j, &ONE, &dwork[iwrk_j], &n,
                              &dwork[i3uple_j], &sdim_j, &ZERO, &q3[ib1 * ldq3], &ldq3);
                    SLC_DAXPY(&n, &dwork[i3lole_j], &q3[(m1 + ij1) * ldq3], &int1, &q3[ib1 * ldq3], &int1);
                    SLC_DAXPY(&n, &dwork[i3lole_j + sdim_j], &q3[(m1 + ij1) * ldq3], &int1, &q3[1 + ib1 * ldq3], &int1);
                    SLC_DGEMV("N", &n, &dim1_j, &ONE, &dwork[iwrk_j], &n, &dwork[i3upri_j], &int1,
                              &dwork[i3lori_j], &q3[(m1 + ij1) * ldq3], &int1);
                }
            } else {
                i32 nr_minus_1 = nr_j - 1;
                i32 ncols_ib = m1 - ib1;
                i32 ncols_ij = m1 - ij1;

                SLC_DCOPY(&nr_j, &a[ib1 * lda], &int1, &dwork[iwrk_j], &int1);
                SLC_DSCAL(&nr_j, &dwork[i2uple_j], &a[ib1 * lda], &int1);
                SLC_DAXPY(&nr_minus_1, &dwork[i2lole_j], &a[(m1 + ij1) * lda], &int1, &a[ib1 * lda], &int1);
                SLC_DSCAL(&nr_minus_1, &dwork[i2lori_j], &a[(m1 + ij1) * lda], &int1);
                SLC_DAXPY(&nr_minus_1, &dwork[i2upri_j], &dwork[iwrk_j], &int1, &a[(m1 + ij1) * lda], &int1);
                a[nr_j - 1 + (m1 + ij1) * lda] = dwork[i2upri_j] * dwork[iwrk_j + nr_j - 1];

                SLC_DCOPY(&nrow_j, &a[i1 + ib1 * lda], &int1, &dwork[iwrk_j], &int1);
                SLC_DSCAL(&nrow_j, &dwork[i2uple_j], &a[i1 + ib1 * lda], &int1);
                SLC_DAXPY(&nrow_j, &dwork[i2lole_j], &a[i1 + (m1 + ij1) * lda], &int1, &a[i1 + ib1 * lda], &int1);
                SLC_DSCAL(&nrow_j, &dwork[i2lori_j], &a[i1 + (m1 + ij1) * lda], &int1);
                SLC_DAXPY(&nrow_j, &dwork[i2upri_j], &dwork[iwrk_j], &int1, &a[i1 + (m1 + ij1) * lda], &int1);

                SLC_DCOPY(&ncols_ib, &a[ib1 + ib1 * lda], &lda, &dwork[iwrk_j], &int1);
                SLC_DSCAL(&ncols_ib, &dwork[i3uple_j], &a[ib1 + ib1 * lda], &lda);
                SLC_DAXPY(&ncols_ib, &dwork[i3lole_j], &a[m1 + ij1 + ib1 * lda], &lda, &a[ib1 + ib1 * lda], &lda);
                SLC_DSCAL(&ncols_ib, &dwork[i3lori_j], &a[m1 + ij1 + ib1 * lda], &lda);
                SLC_DAXPY(&ncols_ib, &dwork[i3upri_j], &dwork[iwrk_j], &int1, &a[m1 + ij1 + ib1 * lda], &lda);

                SLC_DCOPY(&ncols_ij, &a[ib1 + (m1 + ij1) * lda], &lda, &dwork[iwrk_j], &int1);
                SLC_DSCAL(&ncols_ij, &dwork[i3uple_j], &a[ib1 + (m1 + ij1) * lda], &lda);
                SLC_DAXPY(&ncols_ij, &dwork[i3lole_j], &a[m1 + ij1 + (m1 + ij1) * lda], &lda, &a[ib1 + (m1 + ij1) * lda], &lda);
                SLC_DSCAL(&ncols_ij, &dwork[i3lori_j], &a[m1 + ij1 + (m1 + ij1) * lda], &lda);
                SLC_DAXPY(&ncols_ij, &dwork[i3upri_j], &dwork[iwrk_j], &int1, &a[m1 + ij1 + (m1 + ij1) * lda], &lda);

                if (m2 > 0) {
                    SLC_DCOPY(&m4, &a[ib1 + i2 * lda], &lda, &dwork[iwrk_j], &int1);
                    SLC_DSCAL(&m4, &dwork[i3uple_j], &a[ib1 + i2 * lda], &lda);
                    SLC_DAXPY(&m4, &dwork[i3lole_j], &a[m1 + ij1 + i2 * lda], &lda, &a[ib1 + i2 * lda], &lda);
                    SLC_DSCAL(&m4, &dwork[i3lori_j], &a[m1 + ij1 + i2 * lda], &lda);
                    SLC_DAXPY(&m4, &dwork[i3upri_j], &dwork[iwrk_j], &int1, &a[m1 + ij1 + i2 * lda], &lda);
                }

                SLC_DCOPY(&nr_j, &b[ib1 * ldb], &int1, &dwork[iwrk_j], &int1);
                SLC_DSCAL(&nr_j, &dwork[i1uple_j], &b[ib1 * ldb], &int1);
                SLC_DAXPY(&nr_minus_1, &dwork[i1lole_j], &b[(m1 + ij1) * ldb], &int1, &b[ib1 * ldb], &int1);
                SLC_DSCAL(&nr_minus_1, &dwork[i1lori_j], &b[(m1 + ij1) * ldb], &int1);
                SLC_DAXPY(&nr_minus_1, &dwork[i1upri_j], &dwork[iwrk_j], &int1, &b[(m1 + ij1) * ldb], &int1);
                b[nr_j - 1 + (m1 + ij1) * ldb] = dwork[i1upri_j] * dwork[iwrk_j + nr_j - 1];

                SLC_DCOPY(&nrow_j, &b[i1 + ib1 * ldb], &int1, &dwork[iwrk_j], &int1);
                SLC_DSCAL(&nrow_j, &dwork[i1uple_j], &b[i1 + ib1 * ldb], &int1);
                SLC_DAXPY(&nrow_j, &dwork[i1lole_j], &b[i1 + (m1 + ij1) * ldb], &int1, &b[i1 + ib1 * ldb], &int1);
                SLC_DSCAL(&nrow_j, &dwork[i1lori_j], &b[i1 + (m1 + ij1) * ldb], &int1);
                SLC_DAXPY(&nrow_j, &dwork[i1upri_j], &dwork[iwrk_j], &int1, &b[i1 + (m1 + ij1) * ldb], &int1);

                SLC_DCOPY(&ncols_ib, &b[ib1 + ib1 * ldb], &ldb, &dwork[iwrk_j], &int1);
                SLC_DSCAL(&ncols_ib, &dwork[i2uple_j], &b[ib1 + ib1 * ldb], &ldb);
                SLC_DAXPY(&ncols_ib, &dwork[i2lole_j], &b[m1 + ij1 + ib1 * ldb], &ldb, &b[ib1 + ib1 * ldb], &ldb);
                SLC_DSCAL(&ncols_ib, &dwork[i2lori_j], &b[m1 + ij1 + ib1 * ldb], &ldb);
                SLC_DAXPY(&ncols_ib, &dwork[i2upri_j], &dwork[iwrk_j], &int1, &b[m1 + ij1 + ib1 * ldb], &ldb);

                SLC_DCOPY(&ncols_ij, &b[ib1 + (m1 + ij1) * ldb], &ldb, &dwork[iwrk_j], &int1);
                SLC_DSCAL(&ncols_ij, &dwork[i2uple_j], &b[ib1 + (m1 + ij1) * ldb], &ldb);
                SLC_DAXPY(&ncols_ij, &dwork[i2lole_j], &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb, &b[ib1 + (m1 + ij1) * ldb], &ldb);
                SLC_DSCAL(&ncols_ij, &dwork[i2lori_j], &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb);
                SLC_DAXPY(&ncols_ij, &dwork[i2upri_j], &dwork[iwrk_j], &int1, &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb);

                if (m2 > 0) {
                    SLC_DCOPY(&m4, &b[ib1 + i2 * ldb], &ldb, &dwork[iwrk_j], &int1);
                    SLC_DSCAL(&m4, &dwork[i2uple_j], &b[ib1 + i2 * ldb], &ldb);
                    SLC_DAXPY(&m4, &dwork[i2lole_j], &b[m1 + ij1 + i2 * ldb], &ldb, &b[ib1 + i2 * ldb], &ldb);
                    SLC_DSCAL(&m4, &dwork[i2lori_j], &b[m1 + ij1 + i2 * ldb], &ldb);
                    SLC_DAXPY(&m4, &dwork[i2upri_j], &dwork[iwrk_j], &int1, &b[m1 + ij1 + i2 * ldb], &ldb);
                }

                SLC_DCOPY(&nr_j, &d[ib1 * ldd], &int1, &dwork[iwrk_j], &int1);
                SLC_DSCAL(&nr_j, &dwork[i1uple_j], &d[ib1 * ldd], &int1);
                SLC_DAXPY(&nr_minus_1, &dwork[i1lole_j], &d[(m1 + ij1) * ldd], &int1, &d[ib1 * ldd], &int1);
                SLC_DSCAL(&nr_minus_1, &dwork[i1lori_j], &d[(m1 + ij1) * ldd], &int1);
                SLC_DAXPY(&nr_minus_1, &dwork[i1upri_j], &dwork[iwrk_j], &int1, &d[(m1 + ij1) * ldd], &int1);
                d[nr_j - 1 + (m1 + ij1) * ldd] = dwork[i1upri_j] * dwork[iwrk_j + nr_j - 1];

                SLC_DCOPY(&nrow_j, &d[i1 + ib1 * ldd], &int1, &dwork[iwrk_j], &int1);
                SLC_DSCAL(&nrow_j, &dwork[i1uple_j], &d[i1 + ib1 * ldd], &int1);
                SLC_DAXPY(&nrow_j, &dwork[i1lole_j], &d[i1 + (m1 + ij1) * ldd], &int1, &d[i1 + ib1 * ldd], &int1);
                SLC_DSCAL(&nrow_j, &dwork[i1lori_j], &d[i1 + (m1 + ij1) * ldd], &int1);
                SLC_DAXPY(&nrow_j, &dwork[i1upri_j], &dwork[iwrk_j], &int1, &d[i1 + (m1 + ij1) * ldd], &int1);

                SLC_DCOPY(&ncols_ib, &d[ib1 + ib1 * ldd], &ldd, &dwork[iwrk_j], &int1);
                SLC_DSCAL(&ncols_ib, &dwork[i3uple_j], &d[ib1 + ib1 * ldd], &ldd);
                SLC_DAXPY(&ncols_ib, &dwork[i3lole_j], &d[m1 + ij1 + ib1 * ldd], &ldd, &d[ib1 + ib1 * ldd], &ldd);
                SLC_DSCAL(&ncols_ib, &dwork[i3lori_j], &d[m1 + ij1 + ib1 * ldd], &ldd);
                SLC_DAXPY(&ncols_ib, &dwork[i3upri_j], &dwork[iwrk_j], &int1, &d[m1 + ij1 + ib1 * ldd], &ldd);

                SLC_DCOPY(&ncols_ij, &d[ib1 + (m1 + ij1) * ldd], &ldd, &dwork[iwrk_j], &int1);
                SLC_DSCAL(&ncols_ij, &dwork[i3uple_j], &d[ib1 + (m1 + ij1) * ldd], &ldd);
                SLC_DAXPY(&ncols_ij, &dwork[i3lole_j], &d[m1 + ij1 + (m1 + ij1) * ldd], &ldd, &d[ib1 + (m1 + ij1) * ldd], &ldd);
                SLC_DSCAL(&ncols_ij, &dwork[i3lori_j], &d[m1 + ij1 + (m1 + ij1) * ldd], &ldd);
                SLC_DAXPY(&ncols_ij, &dwork[i3upri_j], &dwork[iwrk_j], &int1, &d[m1 + ij1 + (m1 + ij1) * ldd], &ldd);

                if (m2 > 0) {
                    SLC_DCOPY(&m4, &d[ib1 + i2 * ldd], &ldd, &dwork[iwrk_j], &int1);
                    SLC_DSCAL(&m4, &dwork[i3uple_j], &d[ib1 + i2 * ldd], &ldd);
                    SLC_DAXPY(&m4, &dwork[i3lole_j], &d[m1 + ij1 + i2 * ldd], &ldd, &d[ib1 + i2 * ldd], &ldd);
                    SLC_DSCAL(&m4, &dwork[i3lori_j], &d[m1 + ij1 + i2 * ldd], &ldd);
                    SLC_DAXPY(&m4, &dwork[i3upri_j], &dwork[iwrk_j], &int1, &d[m1 + ij1 + i2 * ldd], &ldd);
                }

                if (lcmpq1) {
                    SLC_DCOPY(&n, &q1[ib1 * ldq1], &int1, &dwork[iwrk_j], &int1);
                    SLC_DSCAL(&n, &dwork[i1uple_j], &q1[ib1 * ldq1], &int1);
                    SLC_DAXPY(&n, &dwork[i1lole_j], &q1[(m1 + ij1) * ldq1], &int1, &q1[ib1 * ldq1], &int1);
                    SLC_DSCAL(&n, &dwork[i1lori_j], &q1[(m1 + ij1) * ldq1], &int1);
                    SLC_DAXPY(&n, &dwork[i1upri_j], &dwork[iwrk_j], &int1, &q1[(m1 + ij1) * ldq1], &int1);
                }

                if (lcmpq2) {
                    SLC_DCOPY(&n, &q2[ib1 * ldq2], &int1, &dwork[iwrk_j], &int1);
                    SLC_DSCAL(&n, &dwork[i2uple_j], &q2[ib1 * ldq2], &int1);
                    SLC_DAXPY(&n, &dwork[i2lole_j], &q2[(m1 + ij1) * ldq2], &int1, &q2[ib1 * ldq2], &int1);
                    SLC_DSCAL(&n, &dwork[i2lori_j], &q2[(m1 + ij1) * ldq2], &int1);
                    SLC_DAXPY(&n, &dwork[i2upri_j], &dwork[iwrk_j], &int1, &q2[(m1 + ij1) * ldq2], &int1);
                }

                if (lcmpq3) {
                    SLC_DCOPY(&n, &q3[ib1 * ldq3], &int1, &dwork[iwrk_j], &int1);
                    SLC_DSCAL(&n, &dwork[i3uple_j], &q3[ib1 * ldq3], &int1);
                    SLC_DAXPY(&n, &dwork[i3lole_j], &q3[(m1 + ij1) * ldq3], &int1, &q3[ib1 * ldq3], &int1);
                    SLC_DSCAL(&n, &dwork[i3lori_j], &q3[(m1 + ij1) * ldq3], &int1);
                    SLC_DAXPY(&n, &dwork[i3upri_j], &dwork[iwrk_j], &int1, &q3[(m1 + ij1) * ldq3], &int1);
                }
            }
        }
    }

    if (m2 > 1) {
        i32 m4_minus_2 = m4 - 2;
        SLC_DLACPY("F", &n, &m4_minus_2, &a[(i2 + 1) * lda], &lda, dwork, &n);
        for (i32 i = 0; i < m2 - 1; i++) {
            SLC_DCOPY(&n, &dwork[n * i], &int1, &a[(2 * (m1 + i) + 1) * lda], &int1);
            SLC_DCOPY(&n, &dwork[n * (m2 + i - 1)], &int1, &a[2 * (m1 + i) * lda], &int1);
        }

        SLC_DLACPY("F", &m4_minus_2, &m4, &a[i2 + 1 + i2 * lda], &lda, dwork, &m4_minus_2);
        for (i32 i = 0; i < m2 - 1; i++) {
            SLC_DCOPY(&m4, &dwork[i], &m4_minus_2, &a[2 * (m1 + i) + 1 + i2 * lda], &lda);
            SLC_DCOPY(&m4, &dwork[m2 + i - 1], &m4_minus_2, &a[2 * (m1 + i) + i2 * lda], &lda);
        }

        SLC_DLACPY("F", &n, &m4_minus_2, &b[(i2 + 1) * ldb], &ldb, dwork, &n);
        for (i32 i = 0; i < m2 - 1; i++) {
            SLC_DCOPY(&n, &dwork[n * i], &int1, &b[(2 * (m1 + i) + 1) * ldb], &int1);
            SLC_DCOPY(&n, &dwork[n * (m2 + i - 1)], &int1, &b[2 * (m1 + i) * ldb], &int1);
        }

        SLC_DLACPY("F", &m4_minus_2, &m4, &b[i2 + 1 + i2 * ldb], &ldb, dwork, &m4_minus_2);
        for (i32 i = 0; i < m2 - 1; i++) {
            SLC_DCOPY(&m4, &dwork[i], &m4_minus_2, &b[2 * (m1 + i) + 1 + i2 * ldb], &ldb);
            SLC_DCOPY(&m4, &dwork[m2 + i - 1], &m4_minus_2, &b[2 * (m1 + i) + i2 * ldb], &ldb);
        }

        SLC_DLACPY("F", &n, &m4_minus_2, &d[(i2 + 1) * ldd], &ldd, dwork, &n);
        for (i32 i = 0; i < m2 - 1; i++) {
            SLC_DCOPY(&n, &dwork[n * i], &int1, &d[(2 * (m1 + i) + 1) * ldd], &int1);
            SLC_DCOPY(&n, &dwork[n * (m2 + i - 1)], &int1, &d[2 * (m1 + i) * ldd], &int1);
        }

        SLC_DLACPY("F", &m4_minus_2, &m4, &d[i2 + 1 + i2 * ldd], &ldd, dwork, &m4_minus_2);
        for (i32 i = 0; i < m2 - 1; i++) {
            SLC_DCOPY(&m4, &dwork[i], &m4_minus_2, &d[2 * (m1 + i) + 1 + i2 * ldd], &ldd);
            SLC_DCOPY(&m4, &dwork[m2 + i - 1], &m4_minus_2, &d[2 * (m1 + i) + i2 * ldd], &ldd);
        }

        if (lcmpq1) {
            SLC_DLACPY("F", &n, &m4_minus_2, &q1[(i2 + 1) * ldq1], &ldq1, dwork, &n);
            for (i32 i = 0; i < m2 - 1; i++) {
                SLC_DCOPY(&n, &dwork[n * i], &int1, &q1[(2 * (m1 + i) + 1) * ldq1], &int1);
                SLC_DCOPY(&n, &dwork[n * (m2 + i - 1)], &int1, &q1[2 * (m1 + i) * ldq1], &int1);
            }
        }

        if (lcmpq2) {
            SLC_DLACPY("F", &n, &m4_minus_2, &q2[(i2 + 1) * ldq2], &ldq2, dwork, &n);
            for (i32 i = 0; i < m2 - 1; i++) {
                SLC_DCOPY(&n, &dwork[n * i], &int1, &q2[(2 * (m1 + i) + 1) * ldq2], &int1);
                SLC_DCOPY(&n, &dwork[n * (m2 + i - 1)], &int1, &q2[2 * (m1 + i) * ldq2], &int1);
            }
        }

        if (lcmpq3) {
            SLC_DLACPY("F", &n, &m4_minus_2, &q3[(i2 + 1) * ldq3], &ldq3, dwork, &n);
            for (i32 i = 0; i < m2 - 1; i++) {
                SLC_DCOPY(&n, &dwork[n * i], &int1, &q3[(2 * (m1 + i) + 1) * ldq3], &int1);
                SLC_DCOPY(&n, &dwork[n * (m2 + i - 1)], &int1, &q3[2 * (m1 + i) * ldq3], &int1);
            }
        }
    }
}
