/*
 * SPDX-License-Identifier: BSD-3-Clause
 * MB04HD - Reduce block (anti-)diagonal skew-Hamiltonian/Hamiltonian pencil
 *          to generalized Schur form
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

void mb04hd(const char *compq1, const char *compq2,
            i32 n, f64 *a, i32 lda, f64 *b, i32 ldb,
            f64 *q1, i32 ldq1, f64 *q2, i32 ldq2,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork,
            i32 *bwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 HUND2 = 200.0;

    char cq1 = (char)toupper((unsigned char)compq1[0]);
    char cq2 = (char)toupper((unsigned char)compq2[0]);

    bool liniq1 = (cq1 == 'I');
    bool liniq2 = (cq2 == 'I');
    bool lupdq1 = (cq1 == 'U');
    bool lupdq2 = (cq2 == 'U');
    bool lcmpq1 = liniq1 || lupdq1;
    bool lcmpq2 = liniq2 || lupdq2;
    bool lquery = (ldwork == -1);

    i32 m = n / 2;
    i32 mm = m * m;
    i32 minwrk = 2 * n * n + (m + 168 > 272 ? m + 168 : 272);

    i32 int1 = 1;
    i32 int0 = 0;

    *info = 0;

    if (cq1 != 'N' && !lcmpq1) {
        *info = -1;
    } else if (cq2 != 'N' && !lcmpq2) {
        *info = -2;
    } else if (n < 0 || (n % 2) != 0) {
        *info = -3;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -5;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -7;
    } else if (ldq1 < 1 || (lcmpq1 && ldq1 < (1 > n ? 1 : n))) {
        *info = -9;
    } else if (ldq2 < 1 || (lcmpq2 && ldq2 < (1 > n ? 1 : n))) {
        *info = -11;
    } else if (liwork < (m + 1 > 32 ? m + 1 : 32)) {
        *info = -13;
    } else if (!lquery && ldwork < minwrk) {
        dwork[0] = (f64)minwrk;
        *info = -15;
    }

    if (*info != 0) {
        return;
    }

    /* Compute optimal workspace */
    i32 itemp = (1 > n ? 1 : n);
    if (itemp > 4) itemp = 4;
    i32 bw[4] = {1, 1, 1, 1};
    i32 idum = 0;
    f64 dum[2];
    i32 neg1 = -1;

    SLC_DGGES("V", "V", "S", sb02ow_select, &itemp, a, &lda, b, &ldb,
              &idum, dwork, dwork, dwork, q1, &itemp, q2, &itemp,
              dwork, &neg1, bw, info);
    if (*info != 0) *info = 0;
    dum[0] = dwork[0];

    SLC_DGGES("V", "V", "N", sb02ow_select, &itemp, a, &lda, b, &ldb,
              &idum, dwork, dwork, dwork, q1, &itemp, q2, &itemp,
              &dwork[1], &neg1, bw, info);
    if (*info != 0) *info = 0;
    dum[1] = dwork[1];

    i32 optwrk = 64 + (12 + (i32)dum[0] > 4 * n ? 12 + (i32)dum[0] : 4 * n);
    i32 temp2 = 24 + (i32)dum[1] > 28 + 4 * itemp ? 24 + (i32)dum[1] : 28 + 4 * itemp;
    if (temp2 > optwrk) optwrk = temp2;
    if (minwrk > optwrk) optwrk = minwrk;

    if (lquery) {
        dwork[0] = (f64)optwrk;
        return;
    }

    /* Quick return */
    if (n == 0) {
        dwork[0] = ONE;
        return;
    }

    /* Workspace indices (Fortran 1-based converted to C 0-based) */
    i32 ia11 = 0;
    i32 ib12 = ia11 + mm;
    i32 ia22 = ib12 + mm;
    i32 ib21 = ia22 + mm;
    i32 iv1 = ib21 + mm;
    i32 iv2 = iv1 + mm;
    i32 iv3 = iv2 + mm;
    i32 iv4 = iv3 + mm;
    i32 mp1 = m + 1;

    /* Get machine parameters */
    f64 ulp = SLC_DLAMCH("P");
    f64 base = SLC_DLAMCH("B");
    f64 lgbas = log(base);

    /* Compute maps for MB03BA/MB03KD */
    i32 k = 4;
    i32 kschur = 4;
    iwork[2 * k] = -1;
    iwork[2 * k + 1] = 1;
    iwork[2 * k + 2] = -1;
    iwork[2 * k + 3] = 1;

    i32 iout;
    mb03ba(k, kschur, &iwork[2 * k], &iout, iwork, &iwork[k]);

    /* Store factors of formal matrix product */
    f64 dum0 = ZERO;
    SLC_DCOPY(&mm, &dum0, &int0, dwork, &int1);
    SLC_DCOPY(&mm, &dum0, &int0, &dwork[mm], &int1);
    SLC_DCOPY(&mm, &dum0, &int0, &dwork[2 * mm], &int1);
    SLC_DCOPY(&mm, &dum0, &int0, &dwork[3 * mm], &int1);

    SLC_DLACPY("U", &m, &m, a, &lda, dwork, &m);
    SLC_DLACPY("U", &m, &m, &a[mp1 - 1 + (mp1 - 1) * lda], &lda, &dwork[ia22], &m);
    SLC_DLACPY("U", &m, &m, &b[(mp1 - 1) * ldb], &ldb, &dwork[ib12], &m);
    SLC_DLACPY("U", &m, &m, &b[mp1 - 1], &ldb, &dwork[ib21], &m);

    if (m > 1) {
        i32 m_minus_1 = m - 1;
        i32 ldb_plus_1 = ldb + 1;
        i32 mp1_inc = mp1;
        SLC_DCOPY(&m_minus_1, &b[m + 1], &ldb_plus_1, &dwork[ib21 + 1], &mp1_inc);
    }

    /* Set BWORK according to eigenvalues */
    i32 j = 0;
    i32 ia_idx = iv1;
    i32 ib_idx = ia_idx + 1;

    while (j < m) {
        if (j < m - 1) {
            if (dwork[ib21 + (j + 1) + j * m] == ZERO) {
                ma01bd(base, lgbas, k, &iwork[2 * k], &dwork[j * m + j], mm,
                       &dwork[ia_idx], &dwork[ib_idx], &iwork[3 * k]);
                bwork[j] = (dwork[ia_idx] > ZERO) || (dwork[ib_idx] == ZERO);
                j++;
            } else {
                bwork[j] = true;
                bwork[j + 1] = true;
                j += 2;
            }
        } else if (j == m - 1) {
            ma01bd(base, lgbas, k, &iwork[2 * k], &dwork[mm - 1], mm,
                   &dwork[ia_idx], &dwork[ib_idx], &iwork[3 * k]);
            bwork[j] = (dwork[ia_idx] > ZERO) || (dwork[ib_idx] == ZERO);
            j++;
        }
    }

    /* Check if all BWORK[j] = true */
    j = 0;
    while (j < m && bwork[j]) {
        j++;
    }

    i32 m1, m2, i1, i2, i3, m4;

    if (j != m) {
        /* Apply periodic QZ algorithm for reordering */
        i32 iwrk = 2 * iv1;
        ib21 = 0;
        ia22 = ib21 + mm;
        ib12 = ia22 + mm;
        ia11 = ib12 + mm;

        kschur = 1;
        iwork[2 * k] = 1;
        iwork[2 * k + 1] = -1;
        iwork[2 * k + 2] = 1;
        iwork[2 * k + 3] = -1;

        for (i32 i = 0; i < k; i++) {
            iwork[i] = m;
            iwork[k + i] = 0;
            iwork[3 * k + i] = 1 + i * mm;
        }

        SLC_DCOPY(&mm, &dum0, &int0, &dwork[ib21], &int1);
        SLC_DCOPY(&mm, &dum0, &int0, &dwork[ia22], &int1);
        SLC_DCOPY(&mm, &dum0, &int0, &dwork[ib12], &int1);
        SLC_DCOPY(&mm, &dum0, &int0, &dwork[ia11], &int1);

        SLC_DLACPY("U", &m, &m, &b[mp1 - 1], &ldb, &dwork[ib21], &m);
        SLC_DLACPY("U", &m, &m, &b[(mp1 - 1) * ldb], &ldb, &dwork[ib12], &m);
        SLC_DLACPY("U", &m, &m, a, &lda, &dwork[ia11], &m);
        SLC_DLACPY("U", &m, &m, &a[mp1 - 1 + (mp1 - 1) * lda], &lda, &dwork[ia22], &m);

        if (m > 1) {
            i32 m_minus_1 = m - 1;
            i32 ldb_plus_1 = ldb + 1;
            i32 mp1_inc = mp1;
            SLC_DCOPY(&m_minus_1, &b[m + 1], &ldb_plus_1, &dwork[ib21 + 1], &mp1_inc);
        }

        i32 ldwork_mb03kd = ldwork - iwrk;
        mb03kd("I", &idum, "N", k, m, kschur,
               iwork, &iwork[k], &iwork[2 * k], (bool *)bwork,
               dwork, iwork, &iwork[3 * k], &dwork[iv1],
               iwork, &iwork[3 * k], &m1, HUND2, &iwork[4 * k],
               &dwork[iwrk], ldwork_mb03kd, info);

        if (*info > 0) return;

        m2 = m - m1;
        i1 = m1;
        i2 = i1 + m1;
        i3 = i2 + m2;
        m4 = 2 * m2;

        /* Update Q1 and Q2 if needed, then perform permutations */
        if (lupdq1) {
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, q1, &ldq1, &dwork[iv1], &m, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLACPY("F", &m, &m, &a[mp1 - 1], &lda, q1, &ldq1);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q1[mp1 - 1], &ldq1, &dwork[iv1], &m, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLACPY("F", &m, &m, &a[mp1 - 1], &lda, &q1[mp1 - 1], &ldq1);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q1[(mp1 - 1) * ldq1], &ldq1, &dwork[iv3], &m, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLACPY("F", &m, &m, &a[mp1 - 1], &lda, &q1[(mp1 - 1) * ldq1], &ldq1);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q1[mp1 - 1 + (mp1 - 1) * ldq1], &ldq1, &dwork[iv3], &m, &ZERO, &a[mp1 - 1], &lda);
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
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, q2, &ldq2, &dwork[iv4], &m, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLACPY("F", &m, &m, &a[mp1 - 1], &lda, q2, &ldq2);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q2[mp1 - 1], &ldq2, &dwork[iv4], &m, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLACPY("F", &m, &m, &a[mp1 - 1], &lda, &q2[mp1 - 1], &ldq2);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q2[(mp1 - 1) * ldq2], &ldq2, &dwork[iv2], &m, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLACPY("F", &m, &m, &a[mp1 - 1], &lda, &q2[(mp1 - 1) * ldq2], &ldq2);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q2[mp1 - 1 + (mp1 - 1) * ldq2], &ldq2, &dwork[iv2], &m, &ZERO, &a[mp1 - 1], &lda);
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

        /* Make permutations of A and B matrices */
        if (m2 > 0) {
            SLC_DLASET("F", &m, &m, &ZERO, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLACPY("U", &m1, &m1, &dwork[ia11], &m, a, &lda);
            SLC_DLASET("F", &m1, &m1, &ZERO, &ZERO, &a[i1 * lda], &lda);
            SLC_DLACPY("U", &m1, &m1, &dwork[ia22], &m, &a[i1 + i1 * lda], &lda);
            SLC_DLACPY("F", &m1, &m2, &dwork[ia11 + m * m1], &m, &a[i2 * lda], &lda);
            SLC_DLASET("F", &m1, &m2, &ZERO, &ZERO, &a[i1 + i2 * lda], &lda);
            SLC_DLACPY("U", &m2, &m2, &dwork[ia11 + m * m1 + m1], &m, &a[i2 + i2 * lda], &lda);
            SLC_DLASET("F", &m1, &m2, &ZERO, &ZERO, &a[i3 * lda], &lda);
            SLC_DLACPY("F", &m1, &m2, &dwork[ia22 + m * m1], &m, &a[i1 + i3 * lda], &lda);
            SLC_DLASET("F", &m2, &m2, &ZERO, &ZERO, &a[i2 + i3 * lda], &lda);
            SLC_DLACPY("U", &m2, &m2, &dwork[ia22 + m * m1 + m1], &m, &a[i3 + i3 * lda], &lda);

            SLC_DLASET("F", &m1, &m1, &ZERO, &ZERO, b, &ldb);
            SLC_DLACPY("U", &m1, &m1, &dwork[ib21], &m, &b[m1], &ldb);
            i32 m1_minus_1 = m1 - 1;
            i32 ldb_plus_1 = ldb + 1;
            SLC_DCOPY(&m1_minus_1, &dwork[ib21 + 1], &mp1, &b[m1 + 1], &ldb_plus_1);
            if (m1 > 2) {
                i32 m1_minus_2 = m1 - 2;
                SLC_DLASET("L", &m1_minus_2, &m1_minus_2, &ZERO, &ZERO, &b[m1 + 2], &ldb);
            }
            SLC_DLASET("F", &m4, &m1, &ZERO, &ZERO, &b[i2], &ldb);
            SLC_DLACPY("U", &m1, &m1, &dwork[ib12], &m, &b[i1 * ldb], &ldb);
            if (m1 > 1) {
                i32 m1_minus_1 = m1 - 1;
                SLC_DLASET("L", &m1_minus_1, &m1_minus_1, &ZERO, &ZERO, &b[1 + i1 * ldb], &ldb);
            }
            i32 n_minus_m1 = n - m1;
            SLC_DLASET("F", &n_minus_m1, &m1, &ZERO, &ZERO, &b[i1 + i1 * ldb], &ldb);
            SLC_DLASET("F", &m1, &m2, &ZERO, &ZERO, &b[i2 * ldb], &ldb);
            SLC_DLACPY("F", &m1, &m2, &dwork[ib21 + m * m1], &m, &b[i1 + i2 * ldb], &ldb);
            SLC_DLASET("F", &m2, &m2, &ZERO, &ZERO, &b[i2 + i2 * ldb], &ldb);
            SLC_DLACPY("U", &m2, &m2, &dwork[ib21 + m * m1 + m1], &m, &b[i3 + i2 * ldb], &ldb);
            if (i3 < n - 1) {
                i32 m2_minus_1 = m2 - 1;
                SLC_DCOPY(&m2_minus_1, &dwork[ib21 + m * m1 + i1], &mp1, &b[i3 + 1 + i2 * ldb], &ldb_plus_1);
            }
            if (m2 > 2) {
                i32 m2_minus_2 = m2 - 2;
                SLC_DLASET("L", &m2_minus_2, &m2_minus_2, &ZERO, &ZERO, &b[i3 + 2 + i2 * ldb], &ldb);
            }
            SLC_DLACPY("F", &m1, &m2, &dwork[ib12 + m * m1], &m, &b[i3 * ldb], &ldb);
            SLC_DLASET("F", &m1, &m2, &ZERO, &ZERO, &b[i1 + i3 * ldb], &ldb);
            SLC_DLACPY("F", &m2, &m2, &dwork[ib12 + m * m1 + m1], &m, &b[i2 + i3 * ldb], &ldb);
            SLC_DLASET("F", &m2, &m2, &ZERO, &ZERO, &b[i3 + i3 * ldb], &ldb);
        } else {
            SLC_DLASET("F", &m, &m, &ZERO, &ZERO, &a[mp1 - 1], &lda);
            SLC_DLASET("F", &m, &m, &ZERO, &ZERO, &a[(mp1 - 1) * lda], &lda);
            SLC_DLASET("F", &m, &m, &ZERO, &ZERO, b, &ldb);
            SLC_DLASET("F", &m, &m, &ZERO, &ZERO, &b[mp1 - 1 + (mp1 - 1) * ldb], &ldb);
        }

        /* Initialize Q1 and Q2 if requested */
        if (liniq1) {
            SLC_DLACPY("F", &m, &m1, &dwork[iv1], &m, q1, &ldq1);
            SLC_DLASET("F", &m, &m1, &ZERO, &ZERO, &q1[mp1 - 1], &ldq1);
            SLC_DLASET("F", &m, &m1, &ZERO, &ZERO, &q1[i1 * ldq1], &ldq1);
            SLC_DLACPY("F", &m, &m1, &dwork[iv3], &m, &q1[mp1 - 1 + i1 * ldq1], &ldq1);
            if (m2 > 0) {
                SLC_DLACPY("F", &m, &m2, &dwork[iv1 + m * m1], &m, &q1[i2 * ldq1], &ldq1);
                SLC_DLASET("F", &m, &m2, &ZERO, &ZERO, &q1[mp1 - 1 + i2 * ldq1], &ldq1);
                SLC_DLASET("F", &m, &m2, &ZERO, &ZERO, &q1[i3 * ldq1], &ldq1);
                SLC_DLACPY("F", &m, &m2, &dwork[iv3 + m * m1], &m, &q1[mp1 - 1 + i3 * ldq1], &ldq1);
            }
        }

        if (liniq2) {
            SLC_DLACPY("F", &m, &m1, &dwork[iv4], &m, q2, &ldq2);
            SLC_DLASET("F", &m, &m1, &ZERO, &ZERO, &q2[mp1 - 1], &ldq2);
            SLC_DLASET("F", &m, &m1, &ZERO, &ZERO, &q2[i1 * ldq2], &ldq2);
            SLC_DLACPY("F", &m, &m1, &dwork[iv2], &m, &q2[mp1 - 1 + i1 * ldq2], &ldq2);
            if (m2 > 0) {
                SLC_DLACPY("F", &m, &m2, &dwork[iv4 + m * m1], &m, &q2[i2 * ldq2], &ldq2);
                SLC_DLASET("F", &m, &m2, &ZERO, &ZERO, &q2[mp1 - 1 + i2 * ldq2], &ldq2);
                SLC_DLASET("F", &m, &m2, &ZERO, &ZERO, &q2[i3 * ldq2], &ldq2);
                SLC_DLACPY("F", &m, &m2, &dwork[iv2 + m * m1], &m, &q2[mp1 - 1 + i3 * ldq2], &ldq2);
            }
        }
    } else {
        /* All eigenvalues are positive - no reordering needed */
        m1 = m;
        m2 = 0;
        i1 = m1;
        i2 = i1 + m1;
        i3 = i2;
        m4 = 0;

        SLC_DLASET("F", &m, &m, &ZERO, &ZERO, &a[mp1 - 1], &lda);
        SLC_DLASET("F", &m, &m, &ZERO, &ZERO, &a[(mp1 - 1) * lda], &lda);
        SLC_DLASET("F", &m, &m, &ZERO, &ZERO, b, &ldb);
        SLC_DLASET("F", &m, &m, &ZERO, &ZERO, &b[mp1 - 1 + (mp1 - 1) * ldb], &ldb);

        if (liniq1) {
            SLC_DLASET("F", &n, &n, &ZERO, &ONE, q1, &ldq1);
        }
        if (liniq2) {
            SLC_DLASET("F", &n, &n, &ZERO, &ONE, q2, &ldq2);
        }
    }

    /* Count blocks in BB31 */
    i32 r = 0;
    j = 0;
    while (j < m1) {
        iwork[r] = j;
        r++;
        if (j < m1 - 1) {
            if (b[m1 + j + 1 + j * ldb] == ZERO) {
                j++;
            } else {
                j += 2;
            }
        } else {
            j++;
        }
    }
    iwork[r] = j;

    /* Triangularize upper left subpencil aAA1 - bBB1 */
    for (i32 ii = 0; ii < r; ii++) {
        i32 ib1 = iwork[ii];
        i32 ib2 = iwork[ii + 1];
        i32 dim1 = ib2 - ib1;
        i32 sdim = 2 * dim1;

        i32 iauple = 0;
        i32 iaupri = dim1 * sdim;
        i32 ialori = iaupri + dim1;
        i32 ibuple = sdim * sdim;
        i32 iblole = ibuple + dim1;
        i32 ibupri = 3 * dim1 * sdim;
        i32 iblori = ibupri + dim1;
        i32 i1uple = 2 * sdim * sdim;
        i32 i1lole = i1uple + dim1;
        i32 i1upri = 5 * dim1 * sdim;
        i32 i1lori = i1upri + dim1;
        i32 i2uple = 3 * sdim * sdim;
        i32 i2lole = i2uple + dim1;
        i32 i2upri = 7 * dim1 * sdim;
        i32 i2lori = i2upri + dim1;

        /* Generate input matrices for MB03FD */
        if (dim1 == 1) {
            i32 lda_plus_1_m1 = (lda + 1) * m1;
            i32 sdim_plus_1 = sdim + 1;
            SLC_DCOPY(&sdim, &a[ib1 + ib1 * lda], &lda_plus_1_m1, &dwork[iauple], &sdim_plus_1);
            i32 ldb_minus_1_m1 = (ldb - 1) * m1;
            SLC_DCOPY(&sdim, &b[m1 + ib1 + ib1 * ldb], &ldb_minus_1_m1, &dwork[iblole], &int1);
        } else {
            SLC_DLACPY("U", &dim1, &dim1, &a[ib1 + ib1 * lda], &lda, &dwork[iauple], &sdim);
            i32 sdim_minus_1 = sdim - 1;
            SLC_DLASET("L", &sdim_minus_1, &sdim_minus_1, &ZERO, &ZERO, &dwork[iauple + 1], &sdim);
            SLC_DLASET("F", &dim1, &dim1, &ZERO, &ZERO, &dwork[iaupri], &sdim);
            SLC_DLACPY("U", &dim1, &dim1, &a[m1 + ib1 + (m1 + ib1) * lda], &lda, &dwork[ialori], &sdim);
            dwork[ialori + 1] = ZERO;

            SLC_DLASET("F", &dim1, &dim1, &ZERO, &ZERO, &dwork[ibuple], &sdim);
            SLC_DLACPY("F", &dim1, &dim1, &b[m1 + ib1 + ib1 * ldb], &ldb, &dwork[iblole], &sdim);
            SLC_DLACPY("F", &dim1, &dim1, &b[ib1 + (m1 + ib1) * ldb], &ldb, &dwork[ibupri], &sdim);
            SLC_DLASET("F", &dim1, &dim1, &ZERO, &ZERO, &dwork[iblori], &sdim);
        }

        /* Perform eigenvalue exchange via MB03FD */
        i32 iwrk = 4 * sdim * sdim;
        i32 itmp = iwrk + m * dim1;
        i32 itmp2 = itmp + m * dim1;
        i32 itmp3 = itmp2 + dim1 * dim1;
        i32 ldwork_mb03fd = ldwork - iwrk;

        mb03fd(sdim, ulp, &dwork[iauple], sdim, &dwork[ibuple], sdim,
               &dwork[i1uple], sdim, &dwork[i2uple], sdim,
               &dwork[iwrk], ldwork_mb03fd, info);

        if (*info > 0) {
            if (*info < 3) *info = 2;
            return;
        }

        i32 nr = ib2;

        /* Apply updates based on block size */
        if (dim1 == 2) {
            /* Update A with 2x2 blocks */
            SLC_DLACPY("F", &nr, &dim1, &a[ib1 * lda], &lda, &dwork[iwrk], &nr);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &dwork[iwrk], &nr, &dwork[i1uple], &sdim, &ZERO, &a[ib1 * lda], &lda);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &a[(m1 + ib1) * lda], &lda, &dwork[i1lole], &sdim, &ONE, &a[ib1 * lda], &lda);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &dwork[iwrk], &nr, &dwork[i1upri], &sdim, &ZERO, &dwork[itmp], &nr);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &a[(m1 + ib1) * lda], &lda, &dwork[i1lori], &sdim, &ONE, &dwork[itmp], &nr);
            SLC_DLACPY("F", &nr, &dim1, &dwork[itmp], &nr, &a[(m1 + ib1) * lda], &lda);

            SLC_DLACPY("F", &nr, &dim1, &a[i1 + ib1 * lda], &lda, &dwork[iwrk], &nr);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &dwork[iwrk], &nr, &dwork[i1uple], &sdim, &ZERO, &a[i1 + ib1 * lda], &lda);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &a[i1 + (m1 + ib1) * lda], &lda, &dwork[i1lole], &sdim, &ONE, &a[i1 + ib1 * lda], &lda);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &dwork[iwrk], &nr, &dwork[i1upri], &sdim, &ZERO, &dwork[itmp], &nr);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &a[i1 + (m1 + ib1) * lda], &lda, &dwork[i1lori], &sdim, &ONE, &dwork[itmp], &nr);
            SLC_DLACPY("F", &nr, &dim1, &dwork[itmp], &nr, &a[i1 + (m1 + ib1) * lda], &lda);

            /* Continue with row updates */
            SLC_DLACPY("F", &dim1, &dim1, &a[m1 + ib1 + ib1 * lda], &lda, &dwork[itmp2], &dim1);
            SLC_DLACPY("F", &dim1, &dim1, &a[ib1 + (m1 + ib1) * lda], &lda, &dwork[itmp3], &dim1);
            SLC_DLASET("F", &dim1, &dim1, &ZERO, &ZERO, &a[m1 + ib1 + ib1 * lda], &lda);

            i32 m1_minus_ib2_plus_1 = m1 - ib2;
            SLC_DGEMM("T", "N", &dim1, &m1_minus_ib2_plus_1, &dim1, &ONE, &dwork[i2upri], &sdim, &a[ib1 + ib2 * lda], &lda, &ZERO, &a[m1 + ib1 + ib2 * lda], &lda);

            i32 m1_minus_ib1_plus_1 = m1 - ib1;
            SLC_DGEMM("T", "N", &dim1, &m1_minus_ib1_plus_1, &dim1, &ONE, &dwork[i2uple], &sdim, &a[ib1 + ib1 * lda], &lda, &ZERO, &dwork[itmp], &dim1);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &ONE, &dwork[i2lole], &sdim, &dwork[itmp2], &dim1, &ONE, &dwork[itmp], &dim1);
            SLC_DLACPY("F", &dim1, &m1_minus_ib1_plus_1, &dwork[itmp], &dim1, &a[ib1 + ib1 * lda], &lda);

            SLC_DGEMM("T", "N", &dim1, &m1_minus_ib1_plus_1, &dim1, &ONE, &dwork[i2lole], &sdim, &a[m1 + ib1 + (m1 + ib1) * lda], &lda, &ZERO, &a[ib1 + (m1 + ib1) * lda], &lda);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &ONE, &dwork[i2uple], &sdim, &dwork[itmp3], &dim1, &ONE, &a[ib1 + (m1 + ib1) * lda], &lda);
            SLC_DGEMM("T", "N", &dim1, &m1_minus_ib1_plus_1, &dim1, &ONE, &dwork[i2lori], &sdim, &a[m1 + ib1 + (m1 + ib1) * lda], &lda, &ZERO, &dwork[itmp], &dim1);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &ONE, &dwork[i2upri], &sdim, &dwork[itmp3], &dim1, &ONE, &dwork[itmp], &dim1);
            SLC_DLACPY("F", &dim1, &m1_minus_ib1_plus_1, &dwork[itmp], &dim1, &a[m1 + ib1 + (m1 + ib1) * lda], &lda);

            if (m2 > 0) {
                SLC_DGEMM("T", "N", &dim1, &m2, &dim1, &ONE, &dwork[i2upri], &sdim, &a[ib1 + i2 * lda], &lda, &ZERO, &a[m1 + ib1 + i2 * lda], &lda);
                SLC_DGEMM("T", "N", &dim1, &m2, &dim1, &ONE, &dwork[i2uple], &sdim, &a[ib1 + i2 * lda], &lda, &ZERO, &dwork[itmp], &dim1);
                SLC_DLACPY("F", &dim1, &m2, &dwork[itmp], &dim1, &a[ib1 + i2 * lda], &lda);
                SLC_DGEMM("T", "N", &dim1, &m2, &dim1, &ONE, &dwork[i2lole], &sdim, &a[m1 + ib1 + i3 * lda], &lda, &ZERO, &a[ib1 + i3 * lda], &lda);
                SLC_DGEMM("T", "N", &dim1, &m2, &dim1, &ONE, &dwork[i2lori], &sdim, &a[m1 + ib1 + i3 * lda], &lda, &ZERO, &dwork[itmp], &dim1);
                SLC_DLACPY("F", &dim1, &m2, &dwork[itmp], &dim1, &a[m1 + ib1 + i3 * lda], &lda);
            }

            /* Update B */
            SLC_DLACPY("F", &nr, &dim1, &b[ib1 * ldb], &ldb, &dwork[iwrk], &nr);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &dwork[iwrk], &nr, &dwork[i1uple], &sdim, &ZERO, &b[ib1 * ldb], &ldb);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &b[(m1 + ib1) * ldb], &ldb, &dwork[i1lole], &sdim, &ONE, &b[ib1 * ldb], &ldb);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &dwork[iwrk], &nr, &dwork[i1upri], &sdim, &ZERO, &dwork[itmp], &nr);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &b[(m1 + ib1) * ldb], &ldb, &dwork[i1lori], &sdim, &ONE, &dwork[itmp], &nr);
            SLC_DLACPY("F", &nr, &dim1, &dwork[itmp], &nr, &b[(m1 + ib1) * ldb], &ldb);

            SLC_DLACPY("F", &nr, &dim1, &b[i1 + ib1 * ldb], &ldb, &dwork[iwrk], &nr);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &dwork[iwrk], &nr, &dwork[i1uple], &sdim, &ZERO, &b[i1 + ib1 * ldb], &ldb);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &b[i1 + (m1 + ib1) * ldb], &ldb, &dwork[i1lole], &sdim, &ONE, &b[i1 + ib1 * ldb], &ldb);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &dwork[iwrk], &nr, &dwork[i1upri], &sdim, &ZERO, &dwork[itmp], &nr);
            SLC_DGEMM("N", "N", &nr, &dim1, &dim1, &ONE, &b[i1 + (m1 + ib1) * ldb], &ldb, &dwork[i1lori], &sdim, &ONE, &dwork[itmp], &nr);
            SLC_DLACPY("F", &nr, &dim1, &dwork[itmp], &nr, &b[i1 + (m1 + ib1) * ldb], &ldb);

            SLC_DLACPY("F", &dim1, &dim1, &b[ib1 + ib1 * ldb], &ldb, &dwork[itmp2], &dim1);
            SLC_DLACPY("F", &dim1, &dim1, &b[m1 + ib1 + (m1 + ib1) * ldb], &ldb, &dwork[itmp3], &dim1);
            SLC_DGEMM("T", "N", &dim1, &m1_minus_ib1_plus_1, &dim1, &ONE, &dwork[i2lole], &sdim, &b[m1 + ib1 + ib1 * ldb], &ldb, &ZERO, &b[ib1 + ib1 * ldb], &ldb);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &ONE, &dwork[i2uple], &sdim, &dwork[itmp2], &dim1, &ONE, &b[ib1 + ib1 * ldb], &ldb);
            SLC_DLASET("F", &dim1, &dim1, &ZERO, &ZERO, &b[m1 + ib1 + ib1 * ldb], &ldb);
            i32 ib1_plus_1 = ib1 + 1;
            SLC_DGEMM("T", "N", &dim1, &m1_minus_ib1_plus_1, &dim1, &ONE, &dwork[i2lori], &sdim, &b[m1 + ib1 + ib1_plus_1 * ldb], &ldb, &ZERO, &dwork[itmp], &dim1);
            SLC_DLACPY("F", &dim1, &m1_minus_ib1_plus_1, &dwork[itmp], &dim1, &b[m1 + ib1 + ib1_plus_1 * ldb], &ldb);

            SLC_DGEMM("T", "N", &dim1, &m1_minus_ib1_plus_1, &dim1, &ONE, &dwork[i2upri], &sdim, &b[ib1 + (m1 + ib1) * ldb], &ldb, &ZERO, &b[m1 + ib1 + (m1 + ib1) * ldb], &ldb);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &ONE, &dwork[i2lori], &sdim, &dwork[itmp3], &dim1, &ONE, &b[m1 + ib1 + (m1 + ib1) * ldb], &ldb);
            SLC_DGEMM("T", "N", &dim1, &m1_minus_ib1_plus_1, &dim1, &ONE, &dwork[i2uple], &sdim, &b[ib1 + (m1 + ib1) * ldb], &ldb, &ZERO, &dwork[itmp], &dim1);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &ONE, &dwork[i2lole], &sdim, &dwork[itmp3], &dim1, &ONE, &dwork[itmp], &dim1);
            SLC_DLACPY("F", &dim1, &m1_minus_ib1_plus_1, &dwork[itmp], &dim1, &b[ib1 + (m1 + ib1) * ldb], &ldb);

            if (m2 > 0) {
                SLC_DGEMM("T", "N", &dim1, &m2, &dim1, &ONE, &dwork[i2lole], &sdim, &b[m1 + ib1 + i2 * ldb], &ldb, &ZERO, &b[ib1 + i2 * ldb], &ldb);
                SLC_DGEMM("T", "N", &dim1, &m2, &dim1, &ONE, &dwork[i2lori], &sdim, &b[m1 + ib1 + i2 * ldb], &ldb, &ZERO, &dwork[itmp], &dim1);
                SLC_DLACPY("F", &dim1, &m2, &dwork[itmp], &dim1, &b[m1 + ib1 + i2 * ldb], &ldb);
                SLC_DGEMM("T", "N", &dim1, &m2, &dim1, &ONE, &dwork[i2upri], &sdim, &b[ib1 + i3 * ldb], &ldb, &ZERO, &b[m1 + ib1 + i3 * ldb], &ldb);
                SLC_DGEMM("T", "N", &dim1, &m2, &dim1, &ONE, &dwork[i2uple], &sdim, &b[ib1 + i3 * ldb], &ldb, &ZERO, &dwork[itmp], &dim1);
                SLC_DLACPY("F", &dim1, &m2, &dwork[itmp], &dim1, &b[ib1 + i3 * ldb], &ldb);
            }

            itmp = iwrk + n * dim1;

            /* Update Q1 */
            if (lcmpq1) {
                SLC_DLACPY("F", &n, &dim1, &q1[ib1 * ldq1], &ldq1, &dwork[iwrk], &n);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &dwork[iwrk], &n, &dwork[i1uple], &sdim, &ZERO, &q1[ib1 * ldq1], &ldq1);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &q1[(m1 + ib1) * ldq1], &ldq1, &dwork[i1lole], &sdim, &ONE, &q1[ib1 * ldq1], &ldq1);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &dwork[iwrk], &n, &dwork[i1upri], &sdim, &ZERO, &dwork[itmp], &n);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &q1[(m1 + ib1) * ldq1], &ldq1, &dwork[i1lori], &sdim, &ONE, &dwork[itmp], &n);
                SLC_DLACPY("F", &n, &dim1, &dwork[itmp], &n, &q1[(m1 + ib1) * ldq1], &ldq1);
            }

            /* Update Q2 */
            if (lcmpq2) {
                SLC_DLACPY("F", &n, &dim1, &q2[ib1 * ldq2], &ldq2, &dwork[iwrk], &n);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &dwork[iwrk], &n, &dwork[i2uple], &sdim, &ZERO, &q2[ib1 * ldq2], &ldq2);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &q2[(m1 + ib1) * ldq2], &ldq2, &dwork[i2lole], &sdim, &ONE, &q2[ib1 * ldq2], &ldq2);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &dwork[iwrk], &n, &dwork[i2upri], &sdim, &ZERO, &dwork[itmp], &n);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &q2[(m1 + ib1) * ldq2], &ldq2, &dwork[i2lori], &sdim, &ONE, &dwork[itmp], &n);
                SLC_DLACPY("F", &n, &dim1, &dwork[itmp], &n, &q2[(m1 + ib1) * ldq2], &ldq2);
            }
        } else {
            /* dim1 == 1: scalar updates */
            SLC_DCOPY(&nr, &a[ib1 * lda], &int1, &dwork[iwrk], &int1);
            SLC_DSCAL(&nr, &dwork[i1uple], &a[ib1 * lda], &int1);
            SLC_DAXPY(&nr, &dwork[i1lole], &a[(m1 + ib1) * lda], &int1, &a[ib1 * lda], &int1);
            SLC_DSCAL(&nr, &dwork[i1lori], &a[(m1 + ib1) * lda], &int1);
            SLC_DAXPY(&nr, &dwork[i1upri], &dwork[iwrk], &int1, &a[(m1 + ib1) * lda], &int1);

            SLC_DCOPY(&nr, &a[i1 + ib1 * lda], &int1, &dwork[iwrk], &int1);
            SLC_DSCAL(&nr, &dwork[i1uple], &a[i1 + ib1 * lda], &int1);
            SLC_DAXPY(&nr, &dwork[i1lole], &a[i1 + (m1 + ib1) * lda], &int1, &a[i1 + ib1 * lda], &int1);
            SLC_DSCAL(&nr, &dwork[i1lori], &a[i1 + (m1 + ib1) * lda], &int1);
            SLC_DAXPY(&nr, &dwork[i1upri], &dwork[iwrk], &int1, &a[i1 + (m1 + ib1) * lda], &int1);

            f64 tmp2 = a[m1 + ib1 + ib1 * lda];
            f64 tmp3 = a[ib1 + (m1 + ib1) * lda];
            if (m1 > ib1 + 1) {
                i32 m1_minus_ib1_minus_1 = m1 - ib1 - 1;
                SLC_DCOPY(&m1_minus_ib1_minus_1, &a[ib1 + (ib1 + 1) * lda], &lda, &a[m1 + ib1 + (ib1 + 1) * lda], &lda);
                SLC_DSCAL(&m1_minus_ib1_minus_1, &dwork[i2upri], &a[m1 + ib1 + (ib1 + 1) * lda], &lda);
            }
            a[m1 + ib1 + ib1 * lda] = ZERO;
            i32 m1_minus_ib1_plus_1 = m1 - ib1;
            SLC_DSCAL(&m1_minus_ib1_plus_1, &dwork[i2uple], &a[ib1 + ib1 * lda], &lda);
            a[ib1 + ib1 * lda] += dwork[i2lole] * tmp2;

            SLC_DCOPY(&m1_minus_ib1_plus_1, &a[m1 + ib1 + (m1 + ib1) * lda], &lda, &a[ib1 + (m1 + ib1) * lda], &lda);
            SLC_DSCAL(&m1_minus_ib1_plus_1, &dwork[i2lole], &a[ib1 + (m1 + ib1) * lda], &lda);
            a[ib1 + (m1 + ib1) * lda] += dwork[i2uple] * tmp3;
            SLC_DSCAL(&m1_minus_ib1_plus_1, &dwork[i2lori], &a[m1 + ib1 + (m1 + ib1) * lda], &lda);
            a[m1 + ib1 + (m1 + ib1) * lda] += dwork[i2upri] * tmp3;

            if (m2 > 0) {
                SLC_DCOPY(&m2, &a[ib1 + i2 * lda], &lda, &a[m1 + ib1 + i2 * lda], &lda);
                SLC_DSCAL(&m2, &dwork[i2upri], &a[m1 + ib1 + i2 * lda], &lda);
                SLC_DSCAL(&m2, &dwork[i2uple], &a[ib1 + i2 * lda], &lda);
                SLC_DCOPY(&m2, &a[m1 + ib1 + i3 * lda], &lda, &a[ib1 + i3 * lda], &lda);
                SLC_DSCAL(&m2, &dwork[i2lole], &a[ib1 + i3 * lda], &lda);
                SLC_DSCAL(&m2, &dwork[i2lori], &a[m1 + ib1 + i3 * lda], &lda);
            }

            /* Update B */
            SLC_DCOPY(&nr, &b[ib1 * ldb], &int1, &dwork[iwrk], &int1);
            SLC_DSCAL(&nr, &dwork[i1uple], &b[ib1 * ldb], &int1);
            SLC_DAXPY(&nr, &dwork[i1lole], &b[(m1 + ib1) * ldb], &int1, &b[ib1 * ldb], &int1);
            SLC_DSCAL(&nr, &dwork[i1lori], &b[(m1 + ib1) * ldb], &int1);
            SLC_DAXPY(&nr, &dwork[i1upri], &dwork[iwrk], &int1, &b[(m1 + ib1) * ldb], &int1);

            SLC_DCOPY(&nr, &b[i1 + ib1 * ldb], &int1, &dwork[iwrk], &int1);
            SLC_DSCAL(&nr, &dwork[i1uple], &b[i1 + ib1 * ldb], &int1);
            SLC_DAXPY(&nr, &dwork[i1lole], &b[i1 + (m1 + ib1) * ldb], &int1, &b[i1 + ib1 * ldb], &int1);
            SLC_DSCAL(&nr, &dwork[i1lori], &b[i1 + (m1 + ib1) * ldb], &int1);
            SLC_DAXPY(&nr, &dwork[i1upri], &dwork[iwrk], &int1, &b[i1 + (m1 + ib1) * ldb], &int1);

            tmp2 = b[ib1 + ib1 * ldb];
            tmp3 = b[m1 + ib1 + (m1 + ib1) * ldb];
            SLC_DCOPY(&m1_minus_ib1_plus_1, &b[m1 + ib1 + ib1 * ldb], &ldb, &b[ib1 + ib1 * ldb], &ldb);
            SLC_DSCAL(&m1_minus_ib1_plus_1, &dwork[i2lole], &b[ib1 + ib1 * ldb], &ldb);
            b[ib1 + ib1 * ldb] += dwork[i2uple] * tmp2;
            b[m1 + ib1 + ib1 * ldb] = ZERO;
            i32 ib1_plus_1 = ib1 + 1;
            SLC_DSCAL(&m1_minus_ib1_plus_1, &dwork[i2lori], &b[m1 + ib1 + ib1_plus_1 * ldb], &ldb);

            SLC_DCOPY(&m1_minus_ib1_plus_1, &b[ib1 + (m1 + ib1) * ldb], &ldb, &b[m1 + ib1 + (m1 + ib1) * ldb], &ldb);
            SLC_DSCAL(&m1_minus_ib1_plus_1, &dwork[i2upri], &b[m1 + ib1 + (m1 + ib1) * ldb], &ldb);
            b[m1 + ib1 + (m1 + ib1) * ldb] += dwork[i2lori] * tmp3;
            SLC_DSCAL(&m1_minus_ib1_plus_1, &dwork[i2uple], &b[ib1 + (m1 + ib1) * ldb], &ldb);
            b[ib1 + (m1 + ib1) * ldb] += dwork[i2lole] * tmp3;

            if (m2 > 0) {
                SLC_DCOPY(&m2, &b[m1 + ib1 + i2 * ldb], &ldb, &b[ib1 + i2 * ldb], &ldb);
                SLC_DSCAL(&m2, &dwork[i2lole], &b[ib1 + i2 * ldb], &ldb);
                SLC_DSCAL(&m2, &dwork[i2lori], &b[m1 + ib1 + i2 * ldb], &ldb);
                SLC_DCOPY(&m2, &b[ib1 + i3 * ldb], &ldb, &b[m1 + ib1 + i3 * ldb], &ldb);
                SLC_DSCAL(&m2, &dwork[i2upri], &b[m1 + ib1 + i3 * ldb], &ldb);
                SLC_DSCAL(&m2, &dwork[i2uple], &b[ib1 + i3 * ldb], &ldb);
            }

            itmp = iwrk + n;

            /* Update Q1 */
            if (lcmpq1) {
                SLC_DCOPY(&n, &q1[ib1 * ldq1], &int1, &dwork[iwrk], &int1);
                SLC_DSCAL(&n, &dwork[i1uple], &q1[ib1 * ldq1], &int1);
                SLC_DAXPY(&n, &dwork[i1lole], &q1[(m1 + ib1) * ldq1], &int1, &q1[ib1 * ldq1], &int1);
                SLC_DSCAL(&n, &dwork[i1lori], &q1[(m1 + ib1) * ldq1], &int1);
                SLC_DAXPY(&n, &dwork[i1upri], &dwork[iwrk], &int1, &q1[(m1 + ib1) * ldq1], &int1);
            }

            /* Update Q2 */
            if (lcmpq2) {
                SLC_DCOPY(&n, &q2[ib1 * ldq2], &int1, &dwork[iwrk], &int1);
                SLC_DSCAL(&n, &dwork[i2uple], &q2[ib1 * ldq2], &int1);
                SLC_DAXPY(&n, &dwork[i2lole], &q2[(m1 + ib1) * ldq2], &int1, &q2[ib1 * ldq2], &int1);
                SLC_DSCAL(&n, &dwork[i2lori], &q2[(m1 + ib1) * ldq2], &int1);
                SLC_DAXPY(&n, &dwork[i2upri], &dwork[iwrk], &int1, &q2[(m1 + ib1) * ldq2], &int1);
            }
        }

        /* Inner loop for MB03DD updates (Fortran lines 1175-2093) */
        for (i32 jj = ii - 1; jj >= 0; jj--) {
            i32 ij1 = iwork[jj];
            i32 ij2 = iwork[jj + 1];
            i32 dim1_inner = iwork[ii + 1] - iwork[ii];
            i32 dim2_inner = ij2 - ij1;
            i32 sdim_inner = dim1_inner + dim2_inner;

            i32 iauple_in = 0;
            i32 ialole_in = iauple_in + dim1_inner;
            i32 iaupri_in = dim1_inner * sdim_inner;
            i32 ialori_in = iaupri_in + dim1_inner;
            i32 ibuple_in = sdim_inner * sdim_inner;
            i32 iblole_in = ibuple_in + dim1_inner;
            i32 ibupri_in = sdim_inner * sdim_inner + dim1_inner * sdim_inner;
            i32 iblori_in = ibupri_in + dim1_inner;
            i32 i1uple_in = 2 * sdim_inner * sdim_inner;
            i32 i1lole_in = i1uple_in + dim1_inner;
            i32 i1upri_in = 2 * sdim_inner * sdim_inner + dim1_inner * sdim_inner;
            i32 i1lori_in = i1upri_in + dim1_inner;
            i32 i2uple_in = 3 * sdim_inner * sdim_inner;
            i32 i2lole_in = i2uple_in + dim1_inner;
            i32 i2upri_in = 3 * sdim_inner * sdim_inner + dim1_inner * sdim_inner;
            i32 i2lori_in = i2upri_in + dim1_inner;

            /* Generate input matrices for MB03DD */
            if (dim1_inner == 2 && dim2_inner == 2) {
                SLC_DLACPY("F", &dim1_inner, &dim1_inner, &a[ib1 + ib1 * lda], &lda, &dwork[iauple_in], &sdim_inner);
                SLC_DLACPY("F", &dim2_inner, &dim1_inner, &a[m1 + ij1 + ib1 * lda], &lda, &dwork[ialole_in], &sdim_inner);
                SLC_DLASET("F", &dim1_inner, &dim2_inner, &ZERO, &ZERO, &dwork[iaupri_in], &sdim_inner);
                SLC_DLACPY("F", &dim2_inner, &dim2_inner, &a[m1 + ij1 + (m1 + ij1) * lda], &lda, &dwork[ialori_in], &sdim_inner);

                SLC_DLACPY("F", &dim1_inner, &dim1_inner, &b[ib1 + ib1 * ldb], &ldb, &dwork[ibuple_in], &sdim_inner);
                SLC_DLACPY("F", &dim2_inner, &dim1_inner, &b[m1 + ij1 + ib1 * ldb], &ldb, &dwork[iblole_in], &sdim_inner);
                SLC_DLASET("F", &dim1_inner, &dim2_inner, &ZERO, &ZERO, &dwork[ibupri_in], &sdim_inner);
                SLC_DLACPY("F", &dim2_inner, &dim2_inner, &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb, &dwork[iblori_in], &sdim_inner);

            } else if (dim1_inner == 1 && dim2_inner == 2) {
                dwork[iauple_in] = a[ib1 + ib1 * lda];
                SLC_DCOPY(&dim2_inner, &a[m1 + ij1 + ib1 * lda], &int1, &dwork[ialole_in], &int1);
                dwork[iaupri_in] = ZERO;
                dwork[iaupri_in + sdim_inner] = ZERO;
                SLC_DLACPY("F", &dim2_inner, &dim2_inner, &a[m1 + ij1 + (m1 + ij1) * lda], &lda, &dwork[ialori_in], &sdim_inner);

                dwork[ibuple_in] = b[ib1 + ib1 * ldb];
                SLC_DCOPY(&dim2_inner, &b[m1 + ij1 + ib1 * ldb], &int1, &dwork[iblole_in], &int1);
                dwork[ibupri_in] = ZERO;
                dwork[ibupri_in + sdim_inner] = ZERO;
                SLC_DLACPY("F", &dim2_inner, &dim2_inner, &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb, &dwork[iblori_in], &sdim_inner);

            } else if (dim1_inner == 2 && dim2_inner == 1) {
                SLC_DLACPY("F", &dim1_inner, &dim1_inner, &a[ib1 + ib1 * lda], &lda, &dwork[iauple_in], &sdim_inner);
                SLC_DCOPY(&dim1_inner, &a[m1 + ij1 + ib1 * lda], &lda, &dwork[ialole_in], &sdim_inner);
                dwork[iaupri_in] = ZERO;
                dwork[iaupri_in + 1] = ZERO;
                dwork[ialori_in] = a[m1 + ij1 + (m1 + ij1) * lda];

                SLC_DLACPY("F", &dim1_inner, &dim1_inner, &b[ib1 + ib1 * ldb], &ldb, &dwork[ibuple_in], &sdim_inner);
                SLC_DCOPY(&dim1_inner, &b[m1 + ij1 + ib1 * ldb], &ldb, &dwork[iblole_in], &sdim_inner);
                dwork[ibupri_in] = ZERO;
                dwork[ibupri_in + 1] = ZERO;
                dwork[iblori_in] = b[m1 + ij1 + (m1 + ij1) * ldb];

            } else {
                dwork[iauple_in] = a[ib1 + ib1 * lda];
                dwork[ialole_in] = a[m1 + ij1 + ib1 * lda];
                dwork[iaupri_in] = ZERO;
                dwork[ialori_in] = a[m1 + ij1 + (m1 + ij1) * lda];

                dwork[ibuple_in] = b[ib1 + ib1 * ldb];
                dwork[iblole_in] = b[m1 + ij1 + ib1 * ldb];
                dwork[ibupri_in] = ZERO;
                dwork[iblori_in] = b[m1 + ij1 + (m1 + ij1) * ldb];
            }

            /* Perform upper triangularization via MB03DD */
            i32 iwrk_in = 4 * sdim_inner * sdim_inner;
            i32 itmp_in = iwrk_in + 2 * n;
            i32 ldwork_mb03dd = ldwork - iwrk_in;

            mb03dd("L", &dim1_inner, &dim2_inner, ulp, &dwork[ibuple_in], sdim_inner,
                   &dwork[iauple_in], sdim_inner, &dwork[i1uple_in], sdim_inner,
                   &dwork[i2uple_in], sdim_inner, &dwork[iwrk_in], ldwork_mb03dd, info);

            if (*info > 0) {
                if (*info <= 4) {
                    *info = 2;
                } else {
                    *info = 4;
                }
                return;
            }

            i32 nrow_in = ij2;
            i32 nr_in = ib2;

            if (dim1_inner == 2 && dim2_inner == 2) {
                /* Update A - column updates */
                SLC_DLACPY("F", &nr_in, &dim1_inner, &a[ib1 * lda], &lda, &dwork[iwrk_in], &nr_in);
                SLC_DGEMM("N", "N", &nr_in, &dim1_inner, &dim1_inner, &ONE, &dwork[iwrk_in], &nr_in, &dwork[i1uple_in], &sdim_inner, &ZERO, &a[ib1 * lda], &lda);
                i32 nr_minus_dim1 = nr_in - dim1_inner;
                SLC_DGEMM("N", "N", &nr_minus_dim1, &dim1_inner, &dim2_inner, &ONE, &a[(m1 + ij1) * lda], &lda, &dwork[i1lole_in], &sdim_inner, &ONE, &a[ib1 * lda], &lda);
                SLC_DGEMM("N", "N", &nr_in, &dim2_inner, &dim1_inner, &ONE, &dwork[iwrk_in], &nr_in, &dwork[i1upri_in], &sdim_inner, &ZERO, &dwork[itmp_in], &nr_in);
                SLC_DGEMM("N", "N", &nr_minus_dim1, &dim2_inner, &dim2_inner, &ONE, &a[(m1 + ij1) * lda], &lda, &dwork[i1lori_in], &sdim_inner, &ONE, &dwork[itmp_in], &nr_in);
                SLC_DLACPY("F", &nr_in, &dim2_inner, &dwork[itmp_in], &nr_in, &a[(m1 + ij1) * lda], &lda);

                SLC_DLACPY("F", &nrow_in, &dim1_inner, &a[i1 + ib1 * lda], &lda, &dwork[iwrk_in], &nrow_in);
                SLC_DGEMM("N", "N", &nrow_in, &dim1_inner, &dim1_inner, &ONE, &dwork[iwrk_in], &nrow_in, &dwork[i1uple_in], &sdim_inner, &ZERO, &a[i1 + ib1 * lda], &lda);
                SLC_DGEMM("N", "N", &nrow_in, &dim1_inner, &dim2_inner, &ONE, &a[i1 + (m1 + ij1) * lda], &lda, &dwork[i1lole_in], &sdim_inner, &ONE, &a[i1 + ib1 * lda], &lda);
                SLC_DGEMM("N", "N", &nrow_in, &dim2_inner, &dim1_inner, &ONE, &dwork[iwrk_in], &nrow_in, &dwork[i1upri_in], &sdim_inner, &ZERO, &dwork[itmp_in], &nrow_in);
                SLC_DGEMM("N", "N", &nrow_in, &dim2_inner, &dim2_inner, &ONE, &a[i1 + (m1 + ij1) * lda], &lda, &dwork[i1lori_in], &sdim_inner, &ONE, &dwork[itmp_in], &nrow_in);
                SLC_DLACPY("F", &nrow_in, &dim2_inner, &dwork[itmp_in], &nrow_in, &a[i1 + (m1 + ij1) * lda], &lda);

                /* Update A - row updates */
                i32 m1_minus_ib1_p1 = m1 - ib1;
                SLC_DLACPY("F", &dim1_inner, &m1_minus_ib1_p1, &a[ib1 + ib1 * lda], &lda, &dwork[iwrk_in], &dim1_inner);
                SLC_DGEMM("T", "N", &dim1_inner, &m1_minus_ib1_p1, &dim1_inner, &ONE, &dwork[i2uple_in], &sdim_inner, &dwork[iwrk_in], &dim1_inner, &ZERO, &a[ib1 + ib1 * lda], &lda);
                SLC_DGEMM("T", "N", &dim1_inner, &m1_minus_ib1_p1, &dim2_inner, &ONE, &dwork[i2lole_in], &sdim_inner, &a[m1 + ij1 + ib1 * lda], &lda, &ONE, &a[ib1 + ib1 * lda], &lda);
                SLC_DGEMM("T", "N", &dim2_inner, &m1_minus_ib1_p1, &dim1_inner, &ONE, &dwork[i2upri_in], &sdim_inner, &dwork[iwrk_in], &dim1_inner, &ZERO, &dwork[itmp_in], &dim2_inner);
                SLC_DGEMM("T", "N", &dim2_inner, &m1_minus_ib1_p1, &dim2_inner, &ONE, &dwork[i2lori_in], &sdim_inner, &a[m1 + ij1 + ib1 * lda], &lda, &ONE, &dwork[itmp_in], &dim2_inner);
                SLC_DLACPY("F", &dim2_inner, &m1_minus_ib1_p1, &dwork[itmp_in], &dim2_inner, &a[m1 + ij1 + ib1 * lda], &lda);

                i32 m1_minus_ij1_p1 = m1 - ij1;
                SLC_DLACPY("F", &dim1_inner, &m1_minus_ij1_p1, &a[ib1 + (m1 + ij1) * lda], &lda, &dwork[iwrk_in], &dim1_inner);
                SLC_DGEMM("T", "N", &dim1_inner, &m1_minus_ij1_p1, &dim1_inner, &ONE, &dwork[i2uple_in], &sdim_inner, &dwork[iwrk_in], &dim1_inner, &ZERO, &a[ib1 + (m1 + ij1) * lda], &lda);
                SLC_DGEMM("T", "N", &dim1_inner, &m1_minus_ij1_p1, &dim2_inner, &ONE, &dwork[i2lole_in], &sdim_inner, &a[m1 + ij1 + (m1 + ij1) * lda], &lda, &ONE, &a[ib1 + (m1 + ij1) * lda], &lda);
                SLC_DGEMM("T", "N", &dim2_inner, &m1_minus_ij1_p1, &dim1_inner, &ONE, &dwork[i2upri_in], &sdim_inner, &dwork[iwrk_in], &dim1_inner, &ZERO, &dwork[itmp_in], &dim2_inner);
                SLC_DGEMM("T", "N", &dim2_inner, &m1_minus_ij1_p1, &dim2_inner, &ONE, &dwork[i2lori_in], &sdim_inner, &a[m1 + ij1 + (m1 + ij1) * lda], &lda, &ONE, &dwork[itmp_in], &dim2_inner);
                SLC_DLACPY("F", &dim2_inner, &m1_minus_ij1_p1, &dwork[itmp_in], &dim2_inner, &a[m1 + ij1 + (m1 + ij1) * lda], &lda);

                if (m2 > 0) {
                    SLC_DLACPY("F", &dim1_inner, &m4, &a[ib1 + i2 * lda], &lda, &dwork[iwrk_in], &dim1_inner);
                    SLC_DGEMM("T", "N", &dim1_inner, &m4, &dim1_inner, &ONE, &dwork[i2uple_in], &sdim_inner, &dwork[iwrk_in], &dim1_inner, &ZERO, &a[ib1 + i2 * lda], &lda);
                    SLC_DGEMM("T", "N", &dim1_inner, &m4, &dim2_inner, &ONE, &dwork[i2lole_in], &sdim_inner, &a[m1 + ij1 + i2 * lda], &lda, &ONE, &a[ib1 + i2 * lda], &lda);
                    SLC_DGEMM("T", "N", &dim2_inner, &m4, &dim1_inner, &ONE, &dwork[i2upri_in], &sdim_inner, &dwork[iwrk_in], &dim1_inner, &ZERO, &dwork[itmp_in], &dim2_inner);
                    SLC_DGEMM("T", "N", &dim2_inner, &m4, &dim2_inner, &ONE, &dwork[i2lori_in], &sdim_inner, &a[m1 + ij1 + i2 * lda], &lda, &ONE, &dwork[itmp_in], &dim2_inner);
                    SLC_DLACPY("F", &dim2_inner, &m4, &dwork[itmp_in], &dim2_inner, &a[m1 + ij1 + i2 * lda], &lda);
                }

                /* Update B - column updates */
                SLC_DLACPY("F", &nr_in, &dim1_inner, &b[ib1 * ldb], &ldb, &dwork[iwrk_in], &nr_in);
                SLC_DGEMM("N", "N", &nr_in, &dim1_inner, &dim1_inner, &ONE, &dwork[iwrk_in], &nr_in, &dwork[i1uple_in], &sdim_inner, &ZERO, &b[ib1 * ldb], &ldb);
                SLC_DGEMM("N", "N", &nr_minus_dim1, &dim1_inner, &dim2_inner, &ONE, &b[(m1 + ij1) * ldb], &ldb, &dwork[i1lole_in], &sdim_inner, &ONE, &b[ib1 * ldb], &ldb);
                SLC_DGEMM("N", "N", &nr_in, &dim2_inner, &dim1_inner, &ONE, &dwork[iwrk_in], &nr_in, &dwork[i1upri_in], &sdim_inner, &ZERO, &dwork[itmp_in], &nr_in);
                SLC_DGEMM("N", "N", &nr_minus_dim1, &dim2_inner, &dim2_inner, &ONE, &b[(m1 + ij1) * ldb], &ldb, &dwork[i1lori_in], &sdim_inner, &ONE, &dwork[itmp_in], &nr_in);
                SLC_DLACPY("F", &nr_in, &dim2_inner, &dwork[itmp_in], &nr_in, &b[(m1 + ij1) * ldb], &ldb);

                SLC_DLACPY("F", &nrow_in, &dim1_inner, &b[i1 + ib1 * ldb], &ldb, &dwork[iwrk_in], &nrow_in);
                SLC_DGEMM("N", "N", &nrow_in, &dim1_inner, &dim1_inner, &ONE, &dwork[iwrk_in], &nrow_in, &dwork[i1uple_in], &sdim_inner, &ZERO, &b[i1 + ib1 * ldb], &ldb);
                SLC_DGEMM("N", "N", &nrow_in, &dim1_inner, &dim2_inner, &ONE, &b[i1 + (m1 + ij1) * ldb], &ldb, &dwork[i1lole_in], &sdim_inner, &ONE, &b[i1 + ib1 * ldb], &ldb);
                SLC_DGEMM("N", "N", &nrow_in, &dim2_inner, &dim1_inner, &ONE, &dwork[iwrk_in], &nrow_in, &dwork[i1upri_in], &sdim_inner, &ZERO, &dwork[itmp_in], &nrow_in);
                SLC_DGEMM("N", "N", &nrow_in, &dim2_inner, &dim2_inner, &ONE, &b[i1 + (m1 + ij1) * ldb], &ldb, &dwork[i1lori_in], &sdim_inner, &ONE, &dwork[itmp_in], &nrow_in);
                SLC_DLACPY("F", &nrow_in, &dim2_inner, &dwork[itmp_in], &nrow_in, &b[i1 + (m1 + ij1) * ldb], &ldb);

                /* Update B - row updates */
                SLC_DLACPY("F", &dim1_inner, &m1_minus_ib1_p1, &b[ib1 + ib1 * ldb], &ldb, &dwork[iwrk_in], &dim1_inner);
                SLC_DGEMM("T", "N", &dim1_inner, &m1_minus_ib1_p1, &dim1_inner, &ONE, &dwork[i2uple_in], &sdim_inner, &dwork[iwrk_in], &dim1_inner, &ZERO, &b[ib1 + ib1 * ldb], &ldb);
                SLC_DGEMM("T", "N", &dim1_inner, &m1_minus_ib1_p1, &dim2_inner, &ONE, &dwork[i2lole_in], &sdim_inner, &b[m1 + ij1 + ib1 * ldb], &ldb, &ONE, &b[ib1 + ib1 * ldb], &ldb);
                SLC_DGEMM("T", "N", &dim2_inner, &m1_minus_ib1_p1, &dim1_inner, &ONE, &dwork[i2upri_in], &sdim_inner, &dwork[iwrk_in], &dim1_inner, &ZERO, &dwork[itmp_in], &dim2_inner);
                SLC_DGEMM("T", "N", &dim2_inner, &m1_minus_ib1_p1, &dim2_inner, &ONE, &dwork[i2lori_in], &sdim_inner, &b[m1 + ij1 + ib1 * ldb], &ldb, &ONE, &dwork[itmp_in], &dim2_inner);
                SLC_DLACPY("F", &dim2_inner, &m1_minus_ib1_p1, &dwork[itmp_in], &dim2_inner, &b[m1 + ij1 + ib1 * ldb], &ldb);

                SLC_DLACPY("F", &dim1_inner, &m1_minus_ij1_p1, &b[ib1 + (m1 + ij1) * ldb], &ldb, &dwork[iwrk_in], &dim1_inner);
                SLC_DGEMM("T", "N", &dim1_inner, &m1_minus_ij1_p1, &dim1_inner, &ONE, &dwork[i2uple_in], &sdim_inner, &dwork[iwrk_in], &dim1_inner, &ZERO, &b[ib1 + (m1 + ij1) * ldb], &ldb);
                SLC_DGEMM("T", "N", &dim1_inner, &m1_minus_ij1_p1, &dim2_inner, &ONE, &dwork[i2lole_in], &sdim_inner, &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb, &ONE, &b[ib1 + (m1 + ij1) * ldb], &ldb);
                SLC_DGEMM("T", "N", &dim2_inner, &m1_minus_ij1_p1, &dim1_inner, &ONE, &dwork[i2upri_in], &sdim_inner, &dwork[iwrk_in], &dim1_inner, &ZERO, &dwork[itmp_in], &dim2_inner);
                SLC_DGEMM("T", "N", &dim2_inner, &m1_minus_ij1_p1, &dim2_inner, &ONE, &dwork[i2lori_in], &sdim_inner, &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb, &ONE, &dwork[itmp_in], &dim2_inner);
                SLC_DLACPY("F", &dim2_inner, &m1_minus_ij1_p1, &dwork[itmp_in], &dim2_inner, &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb);

                if (m2 > 0) {
                    SLC_DLACPY("F", &dim1_inner, &m4, &b[ib1 + i2 * ldb], &ldb, &dwork[iwrk_in], &dim1_inner);
                    SLC_DGEMM("T", "N", &dim1_inner, &m4, &dim1_inner, &ONE, &dwork[i2uple_in], &sdim_inner, &dwork[iwrk_in], &dim1_inner, &ZERO, &b[ib1 + i2 * ldb], &ldb);
                    SLC_DGEMM("T", "N", &dim1_inner, &m4, &dim2_inner, &ONE, &dwork[i2lole_in], &sdim_inner, &b[m1 + ij1 + i2 * ldb], &ldb, &ONE, &b[ib1 + i2 * ldb], &ldb);
                    SLC_DGEMM("T", "N", &dim2_inner, &m4, &dim1_inner, &ONE, &dwork[i2upri_in], &sdim_inner, &dwork[iwrk_in], &dim1_inner, &ZERO, &dwork[itmp_in], &dim2_inner);
                    SLC_DGEMM("T", "N", &dim2_inner, &m4, &dim2_inner, &ONE, &dwork[i2lori_in], &sdim_inner, &b[m1 + ij1 + i2 * ldb], &ldb, &ONE, &dwork[itmp_in], &dim2_inner);
                    SLC_DLACPY("F", &dim2_inner, &m4, &dwork[itmp_in], &dim2_inner, &b[m1 + ij1 + i2 * ldb], &ldb);
                }

                /* Update Q1 */
                if (lcmpq1) {
                    SLC_DLACPY("F", &n, &dim1_inner, &q1[ib1 * ldq1], &ldq1, &dwork[iwrk_in], &n);
                    SLC_DGEMM("N", "N", &n, &dim1_inner, &dim1_inner, &ONE, &dwork[iwrk_in], &n, &dwork[i1uple_in], &sdim_inner, &ZERO, &q1[ib1 * ldq1], &ldq1);
                    SLC_DGEMM("N", "N", &n, &dim1_inner, &dim2_inner, &ONE, &q1[(m1 + ij1) * ldq1], &ldq1, &dwork[i1lole_in], &sdim_inner, &ONE, &q1[ib1 * ldq1], &ldq1);
                    SLC_DGEMM("N", "N", &n, &dim2_inner, &dim1_inner, &ONE, &dwork[iwrk_in], &n, &dwork[i1upri_in], &sdim_inner, &ZERO, &dwork[itmp_in], &n);
                    SLC_DGEMM("N", "N", &n, &dim2_inner, &dim2_inner, &ONE, &q1[(m1 + ij1) * ldq1], &ldq1, &dwork[i1lori_in], &sdim_inner, &ONE, &dwork[itmp_in], &n);
                    SLC_DLACPY("F", &n, &dim2_inner, &dwork[itmp_in], &n, &q1[(m1 + ij1) * ldq1], &ldq1);
                }

                /* Update Q2 */
                if (lcmpq2) {
                    SLC_DLACPY("F", &n, &dim1_inner, &q2[ib1 * ldq2], &ldq2, &dwork[iwrk_in], &n);
                    SLC_DGEMM("N", "N", &n, &dim1_inner, &dim1_inner, &ONE, &dwork[iwrk_in], &n, &dwork[i2uple_in], &sdim_inner, &ZERO, &q2[ib1 * ldq2], &ldq2);
                    SLC_DGEMM("N", "N", &n, &dim1_inner, &dim2_inner, &ONE, &q2[(m1 + ij1) * ldq2], &ldq2, &dwork[i2lole_in], &sdim_inner, &ONE, &q2[ib1 * ldq2], &ldq2);
                    SLC_DGEMM("N", "N", &n, &dim2_inner, &dim1_inner, &ONE, &dwork[iwrk_in], &n, &dwork[i2upri_in], &sdim_inner, &ZERO, &dwork[itmp_in], &n);
                    SLC_DGEMM("N", "N", &n, &dim2_inner, &dim2_inner, &ONE, &q2[(m1 + ij1) * ldq2], &ldq2, &dwork[i2lori_in], &sdim_inner, &ONE, &dwork[itmp_in], &n);
                    SLC_DLACPY("F", &n, &dim2_inner, &dwork[itmp_in], &n, &q2[(m1 + ij1) * ldq2], &ldq2);
                }

            } else if (dim1_inner == 1 && dim2_inner == 2) {
                /* dim1=1, dim2=2 case - lines 1557-1763 in Fortran */
                /* Update A - column updates */
                SLC_DCOPY(&nr_in, &a[ib1 * lda], &int1, &dwork[iwrk_in], &int1);
                i32 nr_minus_1 = nr_in - 1;
                SLC_DGEMV("N", &nr_minus_1, &dim2_inner, &ONE, &a[(m1 + ij1) * lda], &lda, &dwork[i1lole_in], &int1, &dwork[i1uple_in], &a[ib1 * lda], &int1);
                a[nr_in - 1 + ib1 * lda] = dwork[i1uple_in] * a[nr_in - 1 + ib1 * lda];
                SLC_DGEMM("N", "N", &nr_minus_1, &dim2_inner, &dim2_inner, &ONE, &a[(m1 + ij1) * lda], &lda, &dwork[i1lori_in], &sdim_inner, &ZERO, &dwork[itmp_in], &nr_in);
                dwork[itmp_in + nr_in - 1] = ZERO;
                dwork[itmp_in + 2 * nr_in - 1] = ZERO;
                SLC_DAXPY(&nr_in, &dwork[i1upri_in], &dwork[iwrk_in], &int1, &dwork[itmp_in], &int1);
                SLC_DAXPY(&nr_in, &dwork[i1upri_in + sdim_inner], &dwork[iwrk_in], &int1, &dwork[itmp_in + nr_in], &int1);
                SLC_DLACPY("F", &nr_in, &dim2_inner, &dwork[itmp_in], &nr_in, &a[(m1 + ij1) * lda], &lda);

                SLC_DCOPY(&nrow_in, &a[i1 + ib1 * lda], &int1, &dwork[iwrk_in], &int1);
                SLC_DGEMV("N", &nrow_in, &dim2_inner, &ONE, &a[i1 + (m1 + ij1) * lda], &lda, &dwork[i1lole_in], &int1, &dwork[i1uple_in], &a[i1 + ib1 * lda], &int1);
                SLC_DGEMM("N", "N", &nrow_in, &dim2_inner, &dim2_inner, &ONE, &a[i1 + (m1 + ij1) * lda], &lda, &dwork[i1lori_in], &sdim_inner, &ZERO, &dwork[itmp_in], &nrow_in);
                SLC_DAXPY(&nrow_in, &dwork[i1upri_in], &dwork[iwrk_in], &int1, &dwork[itmp_in], &int1);
                SLC_DAXPY(&nrow_in, &dwork[i1upri_in + sdim_inner], &dwork[iwrk_in], &int1, &dwork[itmp_in + nrow_in], &int1);
                SLC_DLACPY("F", &nrow_in, &dim2_inner, &dwork[itmp_in], &nrow_in, &a[i1 + (m1 + ij1) * lda], &lda);

                /* Update A - row updates */
                i32 m1_minus_ib1_p1 = m1 - ib1;
                SLC_DCOPY(&m1_minus_ib1_p1, &a[ib1 + ib1 * lda], &lda, &dwork[iwrk_in], &int1);
                SLC_DGEMV("T", &dim2_inner, &m1_minus_ib1_p1, &ONE, &a[m1 + ij1 + ib1 * lda], &lda, &dwork[i2lole_in], &int1, &dwork[i2uple_in], &a[ib1 + ib1 * lda], &lda);
                SLC_DGEMM("T", "N", &dim2_inner, &m1_minus_ib1_p1, &dim2_inner, &ONE, &dwork[i2lori_in], &sdim_inner, &a[m1 + ij1 + ib1 * lda], &lda, &ZERO, &dwork[itmp_in], &dim2_inner);
                SLC_DAXPY(&m1_minus_ib1_p1, &dwork[i2upri_in], &dwork[iwrk_in], &int1, &dwork[itmp_in], &dim2_inner);
                SLC_DAXPY(&m1_minus_ib1_p1, &dwork[i2upri_in + sdim_inner], &dwork[iwrk_in], &int1, &dwork[itmp_in + 1], &dim2_inner);
                SLC_DLACPY("F", &dim2_inner, &m1_minus_ib1_p1, &dwork[itmp_in], &dim2_inner, &a[m1 + ij1 + ib1 * lda], &lda);

                i32 m1_minus_ij1_p1 = m1 - ij1;
                SLC_DCOPY(&m1_minus_ij1_p1, &a[ib1 + (m1 + ij1) * lda], &lda, &dwork[iwrk_in], &int1);
                SLC_DGEMV("T", &dim2_inner, &m1_minus_ij1_p1, &ONE, &a[m1 + ij1 + (m1 + ij1) * lda], &lda, &dwork[i2lole_in], &int1, &dwork[i2uple_in], &a[ib1 + (m1 + ij1) * lda], &lda);
                SLC_DGEMM("T", "N", &dim2_inner, &m1_minus_ij1_p1, &dim2_inner, &ONE, &dwork[i2lori_in], &sdim_inner, &a[m1 + ij1 + (m1 + ij1) * lda], &lda, &ZERO, &dwork[itmp_in], &dim2_inner);
                SLC_DAXPY(&m1_minus_ij1_p1, &dwork[i2upri_in], &dwork[iwrk_in], &int1, &dwork[itmp_in], &dim2_inner);
                SLC_DAXPY(&m1_minus_ij1_p1, &dwork[i2upri_in + sdim_inner], &dwork[iwrk_in], &int1, &dwork[itmp_in + 1], &dim2_inner);
                SLC_DLACPY("F", &dim2_inner, &m1_minus_ij1_p1, &dwork[itmp_in], &dim2_inner, &a[m1 + ij1 + (m1 + ij1) * lda], &lda);

                if (m2 > 0) {
                    SLC_DCOPY(&m4, &a[ib1 + i2 * lda], &lda, &dwork[iwrk_in], &int1);
                    SLC_DGEMV("T", &dim2_inner, &m4, &ONE, &a[m1 + ij1 + i2 * lda], &lda, &dwork[i2lole_in], &int1, &dwork[i2uple_in], &a[ib1 + i2 * lda], &lda);
                    SLC_DGEMM("T", "N", &dim2_inner, &m4, &dim2_inner, &ONE, &dwork[i2lori_in], &sdim_inner, &a[m1 + ij1 + i2 * lda], &lda, &ZERO, &dwork[itmp_in], &dim2_inner);
                    SLC_DAXPY(&m4, &dwork[i2upri_in], &dwork[iwrk_in], &int1, &dwork[itmp_in], &dim2_inner);
                    SLC_DAXPY(&m4, &dwork[i2upri_in + sdim_inner], &dwork[iwrk_in], &int1, &dwork[itmp_in + 1], &dim2_inner);
                    SLC_DLACPY("F", &dim2_inner, &m4, &dwork[itmp_in], &dim2_inner, &a[m1 + ij1 + i2 * lda], &lda);
                }

                /* Update B - column updates */
                SLC_DCOPY(&nr_in, &b[ib1 * ldb], &int1, &dwork[iwrk_in], &int1);
                SLC_DGEMV("N", &nr_minus_1, &dim2_inner, &ONE, &b[(m1 + ij1) * ldb], &ldb, &dwork[i1lole_in], &int1, &dwork[i1uple_in], &b[ib1 * ldb], &int1);
                b[nr_in - 1 + ib1 * ldb] = dwork[i1uple_in] * b[nr_in - 1 + ib1 * ldb];
                SLC_DGEMM("N", "N", &nr_minus_1, &dim2_inner, &dim2_inner, &ONE, &b[(m1 + ij1) * ldb], &ldb, &dwork[i1lori_in], &sdim_inner, &ZERO, &dwork[itmp_in], &nr_in);
                dwork[itmp_in + nr_in - 1] = ZERO;
                dwork[itmp_in + 2 * nr_in - 1] = ZERO;
                SLC_DAXPY(&nr_in, &dwork[i1upri_in], &dwork[iwrk_in], &int1, &dwork[itmp_in], &int1);
                SLC_DAXPY(&nr_in, &dwork[i1upri_in + sdim_inner], &dwork[iwrk_in], &int1, &dwork[itmp_in + nr_in], &int1);
                SLC_DLACPY("F", &nr_in, &dim2_inner, &dwork[itmp_in], &nr_in, &b[(m1 + ij1) * ldb], &ldb);

                SLC_DCOPY(&nrow_in, &b[i1 + ib1 * ldb], &int1, &dwork[iwrk_in], &int1);
                SLC_DGEMV("N", &nrow_in, &dim2_inner, &ONE, &b[i1 + (m1 + ij1) * ldb], &ldb, &dwork[i1lole_in], &int1, &dwork[i1uple_in], &b[i1 + ib1 * ldb], &int1);
                SLC_DGEMM("N", "N", &nrow_in, &dim2_inner, &dim2_inner, &ONE, &b[i1 + (m1 + ij1) * ldb], &ldb, &dwork[i1lori_in], &sdim_inner, &ZERO, &dwork[itmp_in], &nrow_in);
                SLC_DAXPY(&nrow_in, &dwork[i1upri_in], &dwork[iwrk_in], &int1, &dwork[itmp_in], &int1);
                SLC_DAXPY(&nrow_in, &dwork[i1upri_in + sdim_inner], &dwork[iwrk_in], &int1, &dwork[itmp_in + nrow_in], &int1);
                SLC_DLACPY("F", &nrow_in, &dim2_inner, &dwork[itmp_in], &nrow_in, &b[i1 + (m1 + ij1) * ldb], &ldb);

                /* Update B - row updates */
                SLC_DCOPY(&m1_minus_ib1_p1, &b[ib1 + ib1 * ldb], &ldb, &dwork[iwrk_in], &int1);
                SLC_DGEMV("T", &dim2_inner, &m1_minus_ib1_p1, &ONE, &b[m1 + ij1 + ib1 * ldb], &ldb, &dwork[i2lole_in], &int1, &dwork[i2uple_in], &b[ib1 + ib1 * ldb], &ldb);
                SLC_DGEMM("T", "N", &dim2_inner, &m1_minus_ib1_p1, &dim2_inner, &ONE, &dwork[i2lori_in], &sdim_inner, &b[m1 + ij1 + ib1 * ldb], &ldb, &ZERO, &dwork[itmp_in], &dim2_inner);
                SLC_DAXPY(&m1_minus_ib1_p1, &dwork[i2upri_in], &dwork[iwrk_in], &int1, &dwork[itmp_in], &dim2_inner);
                SLC_DAXPY(&m1_minus_ib1_p1, &dwork[i2upri_in + sdim_inner], &dwork[iwrk_in], &int1, &dwork[itmp_in + 1], &dim2_inner);
                SLC_DLACPY("F", &dim2_inner, &m1_minus_ib1_p1, &dwork[itmp_in], &dim2_inner, &b[m1 + ij1 + ib1 * ldb], &ldb);

                SLC_DCOPY(&m1_minus_ij1_p1, &b[ib1 + (m1 + ij1) * ldb], &ldb, &dwork[iwrk_in], &int1);
                SLC_DGEMV("T", &dim2_inner, &m1_minus_ij1_p1, &ONE, &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb, &dwork[i2lole_in], &int1, &dwork[i2uple_in], &b[ib1 + (m1 + ij1) * ldb], &ldb);
                SLC_DGEMM("T", "N", &dim2_inner, &m1_minus_ij1_p1, &dim2_inner, &ONE, &dwork[i2lori_in], &sdim_inner, &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb, &ZERO, &dwork[itmp_in], &dim2_inner);
                SLC_DAXPY(&m1_minus_ij1_p1, &dwork[i2upri_in], &dwork[iwrk_in], &int1, &dwork[itmp_in], &dim2_inner);
                SLC_DAXPY(&m1_minus_ij1_p1, &dwork[i2upri_in + sdim_inner], &dwork[iwrk_in], &int1, &dwork[itmp_in + 1], &dim2_inner);
                SLC_DLACPY("F", &dim2_inner, &m1_minus_ij1_p1, &dwork[itmp_in], &dim2_inner, &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb);

                if (m2 > 0) {
                    SLC_DCOPY(&m4, &b[ib1 + i2 * ldb], &ldb, &dwork[iwrk_in], &int1);
                    SLC_DGEMV("T", &dim2_inner, &m4, &ONE, &b[m1 + ij1 + i2 * ldb], &ldb, &dwork[i2lole_in], &int1, &dwork[i2uple_in], &b[ib1 + i2 * ldb], &ldb);
                    SLC_DGEMM("T", "N", &dim2_inner, &m4, &dim2_inner, &ONE, &dwork[i2lori_in], &sdim_inner, &b[m1 + ij1 + i2 * ldb], &ldb, &ZERO, &dwork[itmp_in], &dim2_inner);
                    SLC_DAXPY(&m4, &dwork[i2upri_in], &dwork[iwrk_in], &int1, &dwork[itmp_in], &dim2_inner);
                    SLC_DAXPY(&m4, &dwork[i2upri_in + sdim_inner], &dwork[iwrk_in], &int1, &dwork[itmp_in + 1], &dim2_inner);
                    SLC_DLACPY("F", &dim2_inner, &m4, &dwork[itmp_in], &dim2_inner, &b[m1 + ij1 + i2 * ldb], &ldb);
                }

                /* Update Q1 */
                if (lcmpq1) {
                    SLC_DCOPY(&n, &q1[ib1 * ldq1], &int1, &dwork[iwrk_in], &int1);
                    SLC_DGEMV("N", &n, &dim2_inner, &ONE, &q1[(m1 + ij1) * ldq1], &ldq1, &dwork[i1lole_in], &int1, &dwork[i1uple_in], &q1[ib1 * ldq1], &int1);
                    SLC_DGEMM("N", "N", &n, &dim2_inner, &dim2_inner, &ONE, &q1[(m1 + ij1) * ldq1], &ldq1, &dwork[i1lori_in], &sdim_inner, &ZERO, &dwork[itmp_in], &n);
                    SLC_DAXPY(&n, &dwork[i1upri_in], &dwork[iwrk_in], &int1, &dwork[itmp_in], &int1);
                    SLC_DAXPY(&n, &dwork[i1upri_in + sdim_inner], &dwork[iwrk_in], &int1, &dwork[itmp_in + n], &int1);
                    SLC_DLACPY("F", &n, &dim2_inner, &dwork[itmp_in], &n, &q1[(m1 + ij1) * ldq1], &ldq1);
                }

                /* Update Q2 */
                if (lcmpq2) {
                    SLC_DCOPY(&n, &q2[ib1 * ldq2], &int1, &dwork[iwrk_in], &int1);
                    SLC_DGEMV("N", &n, &dim2_inner, &ONE, &q2[(m1 + ij1) * ldq2], &ldq2, &dwork[i2lole_in], &int1, &dwork[i2uple_in], &q2[ib1 * ldq2], &int1);
                    SLC_DGEMM("N", "N", &n, &dim2_inner, &dim2_inner, &ONE, &q2[(m1 + ij1) * ldq2], &ldq2, &dwork[i2lori_in], &sdim_inner, &ZERO, &dwork[itmp_in], &n);
                    SLC_DAXPY(&n, &dwork[i2upri_in], &dwork[iwrk_in], &int1, &dwork[itmp_in], &int1);
                    SLC_DAXPY(&n, &dwork[i2upri_in + sdim_inner], &dwork[iwrk_in], &int1, &dwork[itmp_in + n], &int1);
                    SLC_DLACPY("F", &n, &dim2_inner, &dwork[itmp_in], &n, &q2[(m1 + ij1) * ldq2], &ldq2);
                }

            } else if (dim1_inner == 2 && dim2_inner == 1) {
                /* dim1=2, dim2=1 case - lines 1765-1959 in Fortran */
                /* Update A - column updates */
                SLC_DLACPY("F", &nr_in, &dim1_inner, &a[ib1 * lda], &lda, &dwork[iwrk_in], &nr_in);
                SLC_DGEMM("N", "N", &nr_in, &dim1_inner, &dim1_inner, &ONE, &dwork[iwrk_in], &nr_in, &dwork[i1uple_in], &sdim_inner, &ZERO, &a[ib1 * lda], &lda);
                i32 nr_minus_1 = nr_in - 1;
                SLC_DAXPY(&nr_minus_1, &dwork[i1lole_in], &a[(m1 + ij1) * lda], &int1, &a[ib1 * lda], &int1);
                SLC_DAXPY(&nr_minus_1, &dwork[i1lole_in + sdim_inner], &a[(m1 + ij1) * lda], &int1, &a[(ib1 + 1) * lda], &int1);
                a[nr_in - 1 + (m1 + ij1) * lda] = ZERO;
                SLC_DGEMV("N", &nr_in, &dim1_inner, &ONE, &dwork[iwrk_in], &nr_in, &dwork[i1upri_in], &int1, &dwork[i1lori_in], &a[(m1 + ij1) * lda], &int1);

                SLC_DLACPY("F", &nrow_in, &dim1_inner, &a[i1 + ib1 * lda], &lda, &dwork[iwrk_in], &nrow_in);
                SLC_DGEMM("N", "N", &nrow_in, &dim1_inner, &dim1_inner, &ONE, &dwork[iwrk_in], &nrow_in, &dwork[i1uple_in], &sdim_inner, &ZERO, &a[i1 + ib1 * lda], &lda);
                SLC_DAXPY(&nrow_in, &dwork[i1lole_in], &a[i1 + (m1 + ij1) * lda], &int1, &a[i1 + ib1 * lda], &int1);
                SLC_DAXPY(&nrow_in, &dwork[i1lole_in + sdim_inner], &a[i1 + (m1 + ij1) * lda], &int1, &a[i1 + (ib1 + 1) * lda], &int1);
                SLC_DGEMV("N", &nrow_in, &dim1_inner, &ONE, &dwork[iwrk_in], &nrow_in, &dwork[i1upri_in], &int1, &dwork[i1lori_in], &a[i1 + (m1 + ij1) * lda], &int1);

                /* Update A - row updates */
                i32 m1_minus_ib1_p1 = m1 - ib1;
                SLC_DLACPY("F", &dim1_inner, &m1_minus_ib1_p1, &a[ib1 + ib1 * lda], &lda, &dwork[iwrk_in], &dim1_inner);
                SLC_DGEMM("T", "N", &dim1_inner, &m1_minus_ib1_p1, &dim1_inner, &ONE, &dwork[i2uple_in], &sdim_inner, &dwork[iwrk_in], &dim1_inner, &ZERO, &a[ib1 + ib1 * lda], &lda);
                SLC_DAXPY(&m1_minus_ib1_p1, &dwork[i2lole_in], &a[m1 + ij1 + ib1 * lda], &lda, &a[ib1 + ib1 * lda], &lda);
                SLC_DAXPY(&m1_minus_ib1_p1, &dwork[i2lole_in + sdim_inner], &a[m1 + ij1 + ib1 * lda], &lda, &a[ib1 + 1 + ib1 * lda], &lda);
                SLC_DGEMV("T", &dim1_inner, &m1_minus_ib1_p1, &ONE, &dwork[iwrk_in], &dim1_inner, &dwork[i2upri_in], &int1, &dwork[i2lori_in], &a[m1 + ij1 + ib1 * lda], &lda);

                i32 m1_minus_ij1_p1 = m1 - ij1;
                SLC_DLACPY("F", &dim1_inner, &m1_minus_ij1_p1, &a[ib1 + (m1 + ij1) * lda], &lda, &dwork[iwrk_in], &dim1_inner);
                SLC_DGEMM("T", "N", &dim1_inner, &m1_minus_ij1_p1, &dim1_inner, &ONE, &dwork[i2uple_in], &sdim_inner, &dwork[iwrk_in], &dim1_inner, &ZERO, &a[ib1 + (m1 + ij1) * lda], &lda);
                SLC_DAXPY(&m1_minus_ij1_p1, &dwork[i2lole_in], &a[m1 + ij1 + (m1 + ij1) * lda], &lda, &a[ib1 + (m1 + ij1) * lda], &lda);
                SLC_DAXPY(&m1_minus_ij1_p1, &dwork[i2lole_in + sdim_inner], &a[m1 + ij1 + (m1 + ij1) * lda], &lda, &a[ib1 + 1 + (m1 + ij1) * lda], &lda);
                SLC_DGEMV("T", &dim1_inner, &m1_minus_ij1_p1, &ONE, &dwork[iwrk_in], &dim1_inner, &dwork[i2upri_in], &int1, &dwork[i2lori_in], &a[m1 + ij1 + (m1 + ij1) * lda], &lda);

                if (m2 > 0) {
                    SLC_DLACPY("F", &dim1_inner, &m4, &a[ib1 + i2 * lda], &lda, &dwork[iwrk_in], &dim1_inner);
                    SLC_DGEMM("T", "N", &dim1_inner, &m4, &dim1_inner, &ONE, &dwork[i2uple_in], &sdim_inner, &dwork[iwrk_in], &dim1_inner, &ZERO, &a[ib1 + i2 * lda], &lda);
                    SLC_DAXPY(&m4, &dwork[i2lole_in], &a[m1 + ij1 + i2 * lda], &lda, &a[ib1 + i2 * lda], &lda);
                    SLC_DAXPY(&m4, &dwork[i2lole_in + sdim_inner], &a[m1 + ij1 + i2 * lda], &lda, &a[ib1 + 1 + i2 * lda], &lda);
                    SLC_DGEMV("T", &dim1_inner, &m4, &ONE, &dwork[iwrk_in], &dim1_inner, &dwork[i2upri_in], &int1, &dwork[i2lori_in], &a[m1 + ij1 + i2 * lda], &lda);
                }

                /* Update B - column updates */
                SLC_DLACPY("F", &nr_in, &dim1_inner, &b[ib1 * ldb], &ldb, &dwork[iwrk_in], &nr_in);
                SLC_DGEMM("N", "N", &nr_in, &dim1_inner, &dim1_inner, &ONE, &dwork[iwrk_in], &nr_in, &dwork[i1uple_in], &sdim_inner, &ZERO, &b[ib1 * ldb], &ldb);
                SLC_DAXPY(&nr_minus_1, &dwork[i1lole_in], &b[(m1 + ij1) * ldb], &int1, &b[ib1 * ldb], &int1);
                SLC_DAXPY(&nr_minus_1, &dwork[i1lole_in + sdim_inner], &b[(m1 + ij1) * ldb], &int1, &b[(ib1 + 1) * ldb], &int1);
                b[nr_in - 1 + (m1 + ij1) * ldb] = ZERO;
                SLC_DGEMV("N", &nr_in, &dim1_inner, &ONE, &dwork[iwrk_in], &nr_in, &dwork[i1upri_in], &int1, &dwork[i1lori_in], &b[(m1 + ij1) * ldb], &int1);

                SLC_DLACPY("F", &nrow_in, &dim1_inner, &b[i1 + ib1 * ldb], &ldb, &dwork[iwrk_in], &nrow_in);
                SLC_DGEMM("N", "N", &nrow_in, &dim1_inner, &dim1_inner, &ONE, &dwork[iwrk_in], &nrow_in, &dwork[i1uple_in], &sdim_inner, &ZERO, &b[i1 + ib1 * ldb], &ldb);
                SLC_DAXPY(&nrow_in, &dwork[i1lole_in], &b[i1 + (m1 + ij1) * ldb], &int1, &b[i1 + ib1 * ldb], &int1);
                SLC_DAXPY(&nrow_in, &dwork[i1lole_in + sdim_inner], &b[i1 + (m1 + ij1) * ldb], &int1, &b[i1 + (ib1 + 1) * ldb], &int1);
                SLC_DGEMV("N", &nrow_in, &dim1_inner, &ONE, &dwork[iwrk_in], &nrow_in, &dwork[i1upri_in], &int1, &dwork[i1lori_in], &b[i1 + (m1 + ij1) * ldb], &int1);

                /* Update B - row updates */
                SLC_DLACPY("F", &dim1_inner, &m1_minus_ib1_p1, &b[ib1 + ib1 * ldb], &ldb, &dwork[iwrk_in], &dim1_inner);
                SLC_DGEMM("T", "N", &dim1_inner, &m1_minus_ib1_p1, &dim1_inner, &ONE, &dwork[i2uple_in], &sdim_inner, &dwork[iwrk_in], &dim1_inner, &ZERO, &b[ib1 + ib1 * ldb], &ldb);
                SLC_DAXPY(&m1_minus_ib1_p1, &dwork[i2lole_in], &b[m1 + ij1 + ib1 * ldb], &ldb, &b[ib1 + ib1 * ldb], &ldb);
                SLC_DAXPY(&m1_minus_ib1_p1, &dwork[i2lole_in + sdim_inner], &b[m1 + ij1 + ib1 * ldb], &ldb, &b[ib1 + 1 + ib1 * ldb], &ldb);
                SLC_DGEMV("T", &dim1_inner, &m1_minus_ib1_p1, &ONE, &dwork[iwrk_in], &dim1_inner, &dwork[i2upri_in], &int1, &dwork[i2lori_in], &b[m1 + ij1 + ib1 * ldb], &ldb);

                SLC_DLACPY("F", &dim1_inner, &m1_minus_ij1_p1, &b[ib1 + (m1 + ij1) * ldb], &ldb, &dwork[iwrk_in], &dim1_inner);
                SLC_DGEMM("T", "N", &dim1_inner, &m1_minus_ij1_p1, &dim1_inner, &ONE, &dwork[i2uple_in], &sdim_inner, &dwork[iwrk_in], &dim1_inner, &ZERO, &b[ib1 + (m1 + ij1) * ldb], &ldb);
                SLC_DAXPY(&m1_minus_ij1_p1, &dwork[i2lole_in], &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb, &b[ib1 + (m1 + ij1) * ldb], &ldb);
                SLC_DAXPY(&m1_minus_ij1_p1, &dwork[i2lole_in + sdim_inner], &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb, &b[ib1 + 1 + (m1 + ij1) * ldb], &ldb);
                SLC_DGEMV("T", &dim1_inner, &m1_minus_ij1_p1, &ONE, &dwork[iwrk_in], &dim1_inner, &dwork[i2upri_in], &int1, &dwork[i2lori_in], &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb);

                if (m2 > 0) {
                    SLC_DLACPY("F", &dim1_inner, &m4, &b[ib1 + i2 * ldb], &ldb, &dwork[iwrk_in], &dim1_inner);
                    SLC_DGEMM("T", "N", &dim1_inner, &m4, &dim1_inner, &ONE, &dwork[i2uple_in], &sdim_inner, &dwork[iwrk_in], &dim1_inner, &ZERO, &b[ib1 + i2 * ldb], &ldb);
                    SLC_DAXPY(&m4, &dwork[i2lole_in], &b[m1 + ij1 + i2 * ldb], &ldb, &b[ib1 + i2 * ldb], &ldb);
                    SLC_DAXPY(&m4, &dwork[i2lole_in + sdim_inner], &b[m1 + ij1 + i2 * ldb], &ldb, &b[ib1 + 1 + i2 * ldb], &ldb);
                    SLC_DGEMV("T", &dim1_inner, &m4, &ONE, &dwork[iwrk_in], &dim1_inner, &dwork[i2upri_in], &int1, &dwork[i2lori_in], &b[m1 + ij1 + i2 * ldb], &ldb);
                }

                /* Update Q1 */
                if (lcmpq1) {
                    SLC_DLACPY("F", &n, &dim1_inner, &q1[ib1 * ldq1], &ldq1, &dwork[iwrk_in], &n);
                    SLC_DGEMM("N", "N", &n, &dim1_inner, &dim1_inner, &ONE, &dwork[iwrk_in], &n, &dwork[i1uple_in], &sdim_inner, &ZERO, &q1[ib1 * ldq1], &ldq1);
                    SLC_DAXPY(&n, &dwork[i1lole_in], &q1[(m1 + ij1) * ldq1], &int1, &q1[ib1 * ldq1], &int1);
                    SLC_DAXPY(&n, &dwork[i1lole_in + sdim_inner], &q1[(m1 + ij1) * ldq1], &int1, &q1[(ib1 + 1) * ldq1], &int1);
                    SLC_DGEMV("N", &n, &dim1_inner, &ONE, &dwork[iwrk_in], &n, &dwork[i1upri_in], &int1, &dwork[i1lori_in], &q1[(m1 + ij1) * ldq1], &int1);
                }

                /* Update Q2 */
                if (lcmpq2) {
                    SLC_DLACPY("F", &n, &dim1_inner, &q2[ib1 * ldq2], &ldq2, &dwork[iwrk_in], &n);
                    SLC_DGEMM("N", "N", &n, &dim1_inner, &dim1_inner, &ONE, &dwork[iwrk_in], &n, &dwork[i2uple_in], &sdim_inner, &ZERO, &q2[ib1 * ldq2], &ldq2);
                    SLC_DAXPY(&n, &dwork[i2lole_in], &q2[(m1 + ij1) * ldq2], &int1, &q2[ib1 * ldq2], &int1);
                    SLC_DAXPY(&n, &dwork[i2lole_in + sdim_inner], &q2[(m1 + ij1) * ldq2], &int1, &q2[(ib1 + 1) * ldq2], &int1);
                    SLC_DGEMV("N", &n, &dim1_inner, &ONE, &dwork[iwrk_in], &n, &dwork[i2upri_in], &int1, &dwork[i2lori_in], &q2[(m1 + ij1) * ldq2], &int1);
                }

            } else {
                /* dim1=1, dim2=1 case - lines 1961-2091 in Fortran */
                /* Update A - column updates */
                i32 nr_minus_1 = nr_in - 1;
                SLC_DCOPY(&nr_in, &a[ib1 * lda], &int1, &dwork[iwrk_in], &int1);
                SLC_DSCAL(&nr_in, &dwork[i1uple_in], &a[ib1 * lda], &int1);
                SLC_DAXPY(&nr_minus_1, &dwork[i1lole_in], &a[(m1 + ij1) * lda], &int1, &a[ib1 * lda], &int1);
                SLC_DSCAL(&nr_minus_1, &dwork[i1lori_in], &a[(m1 + ij1) * lda], &int1);
                SLC_DAXPY(&nr_minus_1, &dwork[i1upri_in], &dwork[iwrk_in], &int1, &a[(m1 + ij1) * lda], &int1);
                a[nr_in - 1 + (m1 + ij1) * lda] = dwork[i1upri_in] * dwork[iwrk_in + nr_in - 1];

                SLC_DCOPY(&nrow_in, &a[i1 + ib1 * lda], &int1, &dwork[iwrk_in], &int1);
                SLC_DSCAL(&nrow_in, &dwork[i1uple_in], &a[i1 + ib1 * lda], &int1);
                SLC_DAXPY(&nrow_in, &dwork[i1lole_in], &a[i1 + (m1 + ij1) * lda], &int1, &a[i1 + ib1 * lda], &int1);
                SLC_DSCAL(&nrow_in, &dwork[i1lori_in], &a[i1 + (m1 + ij1) * lda], &int1);
                SLC_DAXPY(&nrow_in, &dwork[i1upri_in], &dwork[iwrk_in], &int1, &a[i1 + (m1 + ij1) * lda], &int1);

                /* Update A - row updates */
                i32 m1_minus_ib1_p1 = m1 - ib1;
                SLC_DCOPY(&m1_minus_ib1_p1, &a[ib1 + ib1 * lda], &lda, &dwork[iwrk_in], &int1);
                SLC_DSCAL(&m1_minus_ib1_p1, &dwork[i2uple_in], &a[ib1 + ib1 * lda], &lda);
                SLC_DAXPY(&m1_minus_ib1_p1, &dwork[i2lole_in], &a[m1 + ij1 + ib1 * lda], &lda, &a[ib1 + ib1 * lda], &lda);
                SLC_DSCAL(&m1_minus_ib1_p1, &dwork[i2lori_in], &a[m1 + ij1 + ib1 * lda], &lda);
                SLC_DAXPY(&m1_minus_ib1_p1, &dwork[i2upri_in], &dwork[iwrk_in], &int1, &a[m1 + ij1 + ib1 * lda], &lda);

                i32 m1_minus_ij1_p1 = m1 - ij1;
                SLC_DCOPY(&m1_minus_ij1_p1, &a[ib1 + (m1 + ij1) * lda], &lda, &dwork[iwrk_in], &int1);
                SLC_DSCAL(&m1_minus_ij1_p1, &dwork[i2uple_in], &a[ib1 + (m1 + ij1) * lda], &lda);
                SLC_DAXPY(&m1_minus_ij1_p1, &dwork[i2lole_in], &a[m1 + ij1 + (m1 + ij1) * lda], &lda, &a[ib1 + (m1 + ij1) * lda], &lda);
                SLC_DSCAL(&m1_minus_ij1_p1, &dwork[i2lori_in], &a[m1 + ij1 + (m1 + ij1) * lda], &lda);
                SLC_DAXPY(&m1_minus_ij1_p1, &dwork[i2upri_in], &dwork[iwrk_in], &int1, &a[m1 + ij1 + (m1 + ij1) * lda], &lda);

                if (m2 > 0) {
                    SLC_DCOPY(&m4, &a[ib1 + i2 * lda], &lda, &dwork[iwrk_in], &int1);
                    SLC_DSCAL(&m4, &dwork[i2uple_in], &a[ib1 + i2 * lda], &lda);
                    SLC_DAXPY(&m4, &dwork[i2lole_in], &a[m1 + ij1 + i2 * lda], &lda, &a[ib1 + i2 * lda], &lda);
                    SLC_DSCAL(&m4, &dwork[i2lori_in], &a[m1 + ij1 + i2 * lda], &lda);
                    SLC_DAXPY(&m4, &dwork[i2upri_in], &dwork[iwrk_in], &int1, &a[m1 + ij1 + i2 * lda], &lda);
                }

                /* Update B - column updates */
                SLC_DCOPY(&nr_in, &b[ib1 * ldb], &int1, &dwork[iwrk_in], &int1);
                SLC_DSCAL(&nr_in, &dwork[i1uple_in], &b[ib1 * ldb], &int1);
                SLC_DAXPY(&nr_minus_1, &dwork[i1lole_in], &b[(m1 + ij1) * ldb], &int1, &b[ib1 * ldb], &int1);
                SLC_DSCAL(&nr_minus_1, &dwork[i1lori_in], &b[(m1 + ij1) * ldb], &int1);
                SLC_DAXPY(&nr_minus_1, &dwork[i1upri_in], &dwork[iwrk_in], &int1, &b[(m1 + ij1) * ldb], &int1);
                b[nr_in - 1 + (m1 + ij1) * ldb] = dwork[i1upri_in] * dwork[iwrk_in + nr_in - 1];

                SLC_DCOPY(&nrow_in, &b[i1 + ib1 * ldb], &int1, &dwork[iwrk_in], &int1);
                SLC_DSCAL(&nrow_in, &dwork[i1uple_in], &b[i1 + ib1 * ldb], &int1);
                SLC_DAXPY(&nrow_in, &dwork[i1lole_in], &b[i1 + (m1 + ij1) * ldb], &int1, &b[i1 + ib1 * ldb], &int1);
                SLC_DSCAL(&nrow_in, &dwork[i1lori_in], &b[i1 + (m1 + ij1) * ldb], &int1);
                SLC_DAXPY(&nrow_in, &dwork[i1upri_in], &dwork[iwrk_in], &int1, &b[i1 + (m1 + ij1) * ldb], &int1);

                /* Update B - row updates */
                SLC_DCOPY(&m1_minus_ib1_p1, &b[ib1 + ib1 * ldb], &ldb, &dwork[iwrk_in], &int1);
                SLC_DSCAL(&m1_minus_ib1_p1, &dwork[i2uple_in], &b[ib1 + ib1 * ldb], &ldb);
                SLC_DAXPY(&m1_minus_ib1_p1, &dwork[i2lole_in], &b[m1 + ij1 + ib1 * ldb], &ldb, &b[ib1 + ib1 * ldb], &ldb);
                SLC_DSCAL(&m1_minus_ib1_p1, &dwork[i2lori_in], &b[m1 + ij1 + ib1 * ldb], &ldb);
                SLC_DAXPY(&m1_minus_ib1_p1, &dwork[i2upri_in], &dwork[iwrk_in], &int1, &b[m1 + ij1 + ib1 * ldb], &ldb);

                SLC_DCOPY(&m1_minus_ij1_p1, &b[ib1 + (m1 + ij1) * ldb], &ldb, &dwork[iwrk_in], &int1);
                SLC_DSCAL(&m1_minus_ij1_p1, &dwork[i2uple_in], &b[ib1 + (m1 + ij1) * ldb], &ldb);
                SLC_DAXPY(&m1_minus_ij1_p1, &dwork[i2lole_in], &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb, &b[ib1 + (m1 + ij1) * ldb], &ldb);
                SLC_DSCAL(&m1_minus_ij1_p1, &dwork[i2lori_in], &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb);
                SLC_DAXPY(&m1_minus_ij1_p1, &dwork[i2upri_in], &dwork[iwrk_in], &int1, &b[m1 + ij1 + (m1 + ij1) * ldb], &ldb);

                if (m2 > 0) {
                    SLC_DCOPY(&m4, &b[ib1 + i2 * ldb], &ldb, &dwork[iwrk_in], &int1);
                    SLC_DSCAL(&m4, &dwork[i2uple_in], &b[ib1 + i2 * ldb], &ldb);
                    SLC_DAXPY(&m4, &dwork[i2lole_in], &b[m1 + ij1 + i2 * ldb], &ldb, &b[ib1 + i2 * ldb], &ldb);
                    SLC_DSCAL(&m4, &dwork[i2lori_in], &b[m1 + ij1 + i2 * ldb], &ldb);
                    SLC_DAXPY(&m4, &dwork[i2upri_in], &dwork[iwrk_in], &int1, &b[m1 + ij1 + i2 * ldb], &ldb);
                }

                /* Update Q1 */
                if (lcmpq1) {
                    SLC_DCOPY(&n, &q1[ib1 * ldq1], &int1, &dwork[iwrk_in], &int1);
                    SLC_DSCAL(&n, &dwork[i1uple_in], &q1[ib1 * ldq1], &int1);
                    SLC_DAXPY(&n, &dwork[i1lole_in], &q1[(m1 + ij1) * ldq1], &int1, &q1[ib1 * ldq1], &int1);
                    SLC_DSCAL(&n, &dwork[i1lori_in], &q1[(m1 + ij1) * ldq1], &int1);
                    SLC_DAXPY(&n, &dwork[i1upri_in], &dwork[iwrk_in], &int1, &q1[(m1 + ij1) * ldq1], &int1);
                }

                /* Update Q2 */
                if (lcmpq2) {
                    SLC_DCOPY(&n, &q2[ib1 * ldq2], &int1, &dwork[iwrk_in], &int1);
                    SLC_DSCAL(&n, &dwork[i2uple_in], &q2[ib1 * ldq2], &int1);
                    SLC_DAXPY(&n, &dwork[i2lole_in], &q2[(m1 + ij1) * ldq2], &int1, &q2[ib1 * ldq2], &int1);
                    SLC_DSCAL(&n, &dwork[i2lori_in], &q2[(m1 + ij1) * ldq2], &int1);
                    SLC_DAXPY(&n, &dwork[i2upri_in], &dwork[iwrk_in], &int1, &q2[(m1 + ij1) * ldq2], &int1);
                }
            }
        } /* End of inner loop (jj) */
    }

    /* Triangularize lower right subpencil aAA2 - bBB2 */
    if (m2 > 1) {
        i32 m4_minus_2 = m4 - 2;
        SLC_DLACPY("F", &n, &m4_minus_2, &a[(i2 + 1) * lda], &lda, dwork, &n);
        for (i32 i = 0; i < m2 - 1; i++) {
            SLC_DCOPY(&n, &dwork[n * i], &int1, &a[(2 * (m1 + i) + 2) * lda], &int1);
            SLC_DCOPY(&n, &dwork[n * (m2 + i - 1)], &int1, &a[(2 * (m1 + i) + 1) * lda], &int1);
        }
        SLC_DLACPY("F", &m4_minus_2, &m4, &a[i2 + 1 + i2 * lda], &lda, dwork, &m4_minus_2);
        for (i32 i = 0; i < m2 - 1; i++) {
            SLC_DCOPY(&m4, &dwork[i], &m4_minus_2, &a[2 * (m1 + i) + 2 + i2 * lda], &lda);
            SLC_DCOPY(&m4, &dwork[m2 + i - 1], &m4_minus_2, &a[2 * (m1 + i) + 1 + i2 * lda], &lda);
        }

        SLC_DLACPY("F", &n, &m4_minus_2, &b[(i2 + 1) * ldb], &ldb, dwork, &n);
        for (i32 i = 0; i < m2 - 1; i++) {
            SLC_DCOPY(&n, &dwork[n * i], &int1, &b[(2 * (m1 + i) + 2) * ldb], &int1);
            SLC_DCOPY(&n, &dwork[n * (m2 + i - 1)], &int1, &b[(2 * (m1 + i) + 1) * ldb], &int1);
        }

        SLC_DLACPY("F", &m4_minus_2, &m4, &b[i2 + 1 + i2 * ldb], &ldb, dwork, &m4_minus_2);
        for (i32 i = 0; i < m2 - 1; i++) {
            SLC_DCOPY(&m4, &dwork[i], &m4_minus_2, &b[2 * (m1 + i) + 2 + i2 * ldb], &ldb);
            SLC_DCOPY(&m4, &dwork[m2 + i - 1], &m4_minus_2, &b[2 * (m1 + i) + 1 + i2 * ldb], &ldb);
        }

        if (lcmpq1) {
            SLC_DLACPY("F", &n, &m4_minus_2, &q1[(i2 + 1) * ldq1], &ldq1, dwork, &n);
            for (i32 i = 0; i < m2 - 1; i++) {
                SLC_DCOPY(&n, &dwork[n * i], &int1, &q1[(2 * (m1 + i) + 2) * ldq1], &int1);
                SLC_DCOPY(&n, &dwork[n * (m2 + i - 1)], &int1, &q1[(2 * (m1 + i) + 1) * ldq1], &int1);
            }
        }

        if (lcmpq2) {
            SLC_DLACPY("F", &n, &m4_minus_2, &q2[(i2 + 1) * ldq2], &ldq2, dwork, &n);
            for (i32 i = 0; i < m2 - 1; i++) {
                SLC_DCOPY(&n, &dwork[n * i], &int1, &q2[(2 * (m1 + i) + 2) * ldq2], &int1);
                SLC_DCOPY(&n, &dwork[n * (m2 + i - 1)], &int1, &q2[(2 * (m1 + i) + 1) * ldq2], &int1);
            }
        }
    }

    dwork[0] = (f64)optwrk;
}
