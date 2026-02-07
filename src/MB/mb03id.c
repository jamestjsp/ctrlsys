/*
 * SPDX-License-Identifier: BSD-3-Clause
 * MB03ID - Move eigenvalues with negative real parts to leading subpencil
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MB03ID_DEBUG (getenv("MB03ID_DEBUG") != NULL)

void mb03id(const char *compq, const char *compu, i32 n, f64 *a, i32 lda,
            f64 *c, i32 ldc, f64 *d, i32 ldd, f64 *b, i32 ldb, f64 *f, i32 ldf,
            f64 *q, i32 ldq, f64 *u1, i32 ldu1, f64 *u2, i32 ldu2, i32 *neig,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork, i32 *info)
{
    const f64 ZERO = 0.0, ONE = 1.0, HALF = 0.5, TEN = 10.0;
    i32 int1 = 1;

    char compq_c = (char)toupper((unsigned char)compq[0]);
    char compu_c = (char)toupper((unsigned char)compu[0]);

    i32 m = n / 2;
    bool liniq = (compq_c == 'I');
    bool lupdq = (compq_c == 'U');
    bool liniu = (compu_c == 'I');
    bool lupdu = (compu_c == 'U');
    bool lcmpq = liniq || lupdq;
    bool lcmpu = liniu || lupdu;

    i32 optdw;
    if (lcmpq) {
        optdw = (4*n + 48 > 171) ? 4*n + 48 : 171;
    } else {
        optdw = (2*n + 48 > 171) ? 2*n + 48 : 171;
    }

    *info = 0;

    if (compq_c != 'N' && !lcmpq) {
        *info = -1;
    } else if (compu_c != 'N' && !lcmpu) {
        *info = -2;
    } else if (n < 0 || (n % 2) != 0) {
        *info = -3;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -5;
    } else if (ldc < (m > 1 ? m : 1)) {
        *info = -7;
    } else if (ldd < (m > 1 ? m : 1)) {
        *info = -9;
    } else if (ldb < (m > 1 ? m : 1)) {
        *info = -11;
    } else if (ldf < (m > 1 ? m : 1)) {
        *info = -13;
    } else if (ldq < 1 || (lcmpq && ldq < n)) {
        *info = -15;
    } else if (ldu1 < 1 || (lcmpu && ldu1 < m)) {
        *info = -17;
    } else if (ldu2 < 1 || (lcmpu && ldu2 < m)) {
        *info = -19;
    } else if (liwork < n + 1) {
        *info = -22;
    } else if (ldwork < optdw) {
        *info = -24;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        *neig = 0;
        return;
    }

    f64 prec = SLC_DLAMCH("Precision");
    f64 base = SLC_DLAMCH("Base");
    f64 lgbas = log(base);
    f64 dn = (f64)n * prec;
    f64 tol_val = (m < 10 ? (f64)m : TEN) * prec;

    f64 par[2];
    par[0] = prec;
    par[1] = SLC_DLAMCH("Safe minimum");

    /* STEP 0: Partition blocks - determine location and size of diagonal blocks */
    i32 i_idx = 0;  /* 0-based index for C arrays */
    i32 j_idx = 0;
    i32 is_offset = m + 1;  /* Fortran IS = M + 1; C: iwork[is_offset + j] = IWORK(IS + J + 1) */

    f64 nrmb = SLC_DLANHS("One", &m, b, &ldb, dwork);

    if (MB03ID_DEBUG) {
        fprintf(stderr, "MB03ID: tol=%.16e, nrmb=%.16e, threshold=%.16e\n", tol_val, nrmb, tol_val * nrmb);
        fprintf(stderr, "MB03ID: B subdiagonals:\n");
        for (i32 ii = 0; ii < m - 1; ii++) {
            fprintf(stderr, "  B(%d,%d) = %.16e\n", ii + 2, ii + 1, b[(ii + 1) + ii * ldb]);
        }
    }

    /* WHILE(i_idx < m-1) */
    while (i_idx < m - 1) {
        iwork[j_idx] = i_idx + 1;  /* Store 1-based index */

        if (MB03ID_DEBUG) {
            fprintf(stderr, "MB03ID: i=%d, B(%d,%d)=%.16e, threshold=%.16e, is_small=%d\n",
                    i_idx + 1, i_idx + 2, i_idx + 1, b[(i_idx + 1) + i_idx * ldb],
                    tol_val * nrmb, fabs(b[(i_idx + 1) + i_idx * ldb]) <= tol_val * nrmb);
        }

        if (fabs(b[(i_idx + 1) + i_idx * ldb]) <= tol_val * nrmb) {
            /* 1-by-1 block */
            b[(i_idx + 1) + i_idx * ldb] = ZERO;
            f64 sign_val = a[i_idx + i_idx * lda] * b[i_idx + i_idx * ldb] * c[i_idx + i_idx * ldc];
            iwork[is_offset + j_idx] = (sign_val >= 0.0) ? 1 : -1;
            i_idx++;
        } else {
            /* 2-by-2 block */
            f64 u11_val = b[(i_idx + 1) + i_idx * ldb] * a[i_idx + (i_idx + 1) * lda];
            f64 u12_val = b[(i_idx + 1) + i_idx * ldb] * c[(i_idx + 1) + i_idx * ldc];
            f64 tmpa = b[(i_idx + 1) + (i_idx + 1) * ldb] * a[i_idx + i_idx * lda] - u11_val;
            f64 tmpc = b[i_idx + i_idx * ldb] * c[(i_idx + 1) + (i_idx + 1) * ldc] - u12_val;

            if (fabs(tmpa) <= prec * fabs(u11_val) &&
                fabs(tmpc) <= prec * fabs(u12_val)) {
                /* Severe cancellation - use periodic QZ */
                i32 idum[8];
                f64 prd[12];  /* 2x2x3 */
                i32 prd_ld = 2;

                idum[0] = 1;
                idum[1] = 2;
                idum[2] = 3;
                idum[3] = 1;
                idum[4] = -1;
                idum[5] = -1;

                /* Copy B block to prd[0:4] */
                i32 two = 2;
                SLC_DLACPY("Full", &two, &two, &b[i_idx + i_idx * ldb], &ldb, prd, &prd_ld);
                /* Copy A block to prd[4:8] */
                SLC_DLACPY("Upper", &two, &two, &a[i_idx + i_idx * lda], &lda, &prd[4], &prd_ld);
                /* Transpose C block to prd[8:12] */
                ma02ad("Lower", two, two, &c[i_idx + i_idx * ldc], ldc, &prd[8], prd_ld);
                prd[4 + 1] = ZERO;  /* prd(2,1,2) */
                prd[8 + 1] = ZERO;  /* prd(2,1,3) */

                i32 info_bb;
                i32 k_val = 3;
                mb03bb(base, lgbas, prec, k_val, idum, &idum[3], 1, prd, prd_ld, two,
                       dwork, &dwork[2], &dwork[4], &idum[6], &dwork[6], &info_bb);

                if (info_bb == 1) {
                    *info = 1;
                    return;
                }

                if (dwork[4] == ZERO || dwork[5] == ZERO ||
                    fabs(dwork[0]) <= dn * fabs(dwork[2])) {
                    iwork[is_offset + j_idx] = 0;
                } else {
                    iwork[is_offset + j_idx] = ma01cd(dwork[0], idum[6], dwork[1], idum[7]);
                }
            } else if (c[i_idx + i_idx * ldc] == ZERO ||
                       a[(i_idx + 1) + (i_idx + 1) * lda] == ZERO) {
                iwork[is_offset + j_idx] = 0;
            } else {
                f64 u11_calc = tmpa / a[(i_idx + 1) + (i_idx + 1) * lda] +
                               tmpc / c[i_idx + i_idx * ldc];
                if (u11_calc == ZERO) {
                    iwork[is_offset + j_idx] = 0;
                } else {
                    f64 sign1 = (u11_calc >= 0.0) ? ONE : -ONE;
                    f64 sign2 = (a[i_idx + i_idx * lda] * c[(i_idx + 1) + (i_idx + 1) * ldc] >= 0.0) ? ONE : -ONE;
                    iwork[is_offset + j_idx] = (i32)(sign1 * sign2);
                }
            }
            i_idx += 2;
        }
        j_idx++;
    }

    if (i_idx == m - 1) {
        /* Last 1-by-1 block */
        iwork[j_idx] = i_idx + 1;  /* 1-based */
        f64 sign_val = a[i_idx + i_idx * lda] * b[i_idx + i_idx * ldb] * c[i_idx + i_idx * ldc];
        iwork[is_offset + j_idx] = (sign_val >= 0.0) ? 1 : -1;
        j_idx++;
    }

    i32 r = j_idx;

    /* Initialize Q if appropriate */
    i32 iupd, upds;
    if (liniq) {
        iupd = m;  /* 0-based offset into Q columns */
        upds = m;
        SLC_DLASET("Full", &n, &n, &ZERO, &ONE, q, &ldq);
    } else if (lupdq) {
        iupd = 0;
        upds = n;
    }

    /* Initialize U1 and U2 if appropriate */
    if (liniu) {
        SLC_DLASET("Full", &m, &m, &ZERO, &ONE, u1, &ldu1);
        SLC_DLASET("Full", &m, &m, &ZERO, &ZERO, u2, &ldu2);
    }

    f64 a2 = 0.0, c2 = 0.0, f2 = 0.0;
    if (m > 1) {
        a2 = a[(m - 1) + (m - 2) * lda];
        c2 = c[(m - 2) + (m - 1) * ldc];
        f2 = f[(m - 1) + (m - 2) * ldf];
    }

    /* STEP 1: Reorder eigenvalues in subpencil aC'*A - bB */
    i32 mm = 0;
    i32 mp = r + 1;  /* Fortran: MP = J where J = R + 1 */

    /* Workspace pointers (0-based) */
    i32 iq1 = 0;
    i32 iq2 = iq1 + 16;
    i32 iq3 = iq2 + 16;
    i32 ia = iq3 + 16;
    i32 ib_w = ia + 16;
    i32 ic_w = ib_w + 16;
    i32 iwrk1 = ic_w + 16;
    i32 iwrk2 = ia;

    i32 ib3 = m;  /* 0-based position M (1-based M+1) */
    iwork[r] = ib3 + 1;  /* Store 1-based */

    if (MB03ID_DEBUG) {
        fprintf(stderr, "MB03ID: n=%d, m=%d, r=%d blocks\n", n, m, r);
        fprintf(stderr, "MB03ID: block boundaries (1-based):");
        for (i32 i = 0; i <= r; i++) fprintf(stderr, " iwork[%d]=%d", i, iwork[i]);
        fprintf(stderr, "\nMB03ID: signs:");
        for (i32 i = 0; i < r; i++) fprintf(stderr, " [%d]=%d", i, iwork[is_offset + i]);
        fprintf(stderr, "\n");
    }

    /* I. Reorder eigenvalues with negative real parts to top */
    for (i32 k = 0; k < r; k++) {
        if (iwork[is_offset + k] < 0) {
            for (i32 jj = k - 1; jj >= mm; jj--) {
                i32 ib1 = iwork[jj] - 1;       /* 0-based */
                i32 ib2 = iwork[jj + 1] - 1;   /* 0-based */
                ib3 = iwork[jj + 2] - 1;       /* 0-based */
                i32 dim1 = ib2 - ib1;
                i32 dim2 = ib3 - ib2;
                i32 sdim = dim1 + dim2;

                /* Copy blocks to DWORK */
                SLC_DLACPY("Upper", &sdim, &sdim, &a[ib1 + ib1 * lda], &lda, &dwork[ia], &sdim);
                ma02ad("Lower", sdim, sdim, &c[ib1 + ib1 * ldc], ldc, &dwork[ic_w], sdim);
                SLC_DLACPY("Upper", &sdim, &sdim, &b[ib1 + ib1 * ldb], &ldb, &dwork[ib_w], &sdim);

                /* Copy subdiagonal of B */
                i32 sdim_m1 = sdim - 1;
                i32 ldb_p1 = ldb + 1;
                i32 sdim_p1 = sdim + 1;
                SLC_DCOPY(&sdim_m1, &b[(ib1 + 1) + ib1 * ldb], &ldb_p1, &dwork[ib_w + 1], &sdim_p1);

                /* Set additional zeros */
                if (dim1 == 2) {
                    dwork[ia + 1] = ZERO;
                    dwork[ic_w + 1] = ZERO;
                }
                if (dim2 == 2) {
                    i32 i1 = sdim * (sdim - 1) - 1;
                    dwork[ia + i1] = ZERO;
                    dwork[ic_w + i1] = ZERO;
                }
                dwork[ib_w + sdim - 1] = ZERO;
                if (sdim == 4) {
                    dwork[ib_w + 2] = ZERO;
                    dwork[ib_w + 7] = ZERO;
                }

                /* Perform eigenvalue exchange via MB03CD */
                i32 info_cd;
                i32 ldwork_avail = ldwork - iwrk1;
                mb03cd("Upper", &dim1, &dim2, prec, &dwork[ic_w], sdim,
                       &dwork[ia], sdim, &dwork[ib_w], sdim, &dwork[iq1], sdim,
                       &dwork[iq2], sdim, &dwork[iq3], sdim, &dwork[iwrk1],
                       ldwork_avail, &info_cd);

                if (info_cd > 0) {
                    *info = 2;
                    return;
                }

                /* Copy transformed B back if sdim > 2 */
                if (sdim > 2) {
                    SLC_DLACPY("Upper", &sdim, &sdim, &dwork[ib_w], &sdim, &b[ib1 + ib1 * ldb], &ldb);
                    SLC_DCOPY(&sdim_m1, &dwork[ib_w + 1], &sdim_p1, &b[(ib1 + 1) + ib1 * ldb], &ldb_p1);
                }

                i32 nrows = ib1;
                i32 ncols = m - ib3;
                i32 nrow = ib3;
                i32 ncol = m - ib1;

                /* Save and clear lower triangle of A */
                f64 dum[12];  /* 3x4 */
                SLC_DLACPY("Lower", &sdim_m1, &sdim_m1, &a[(ib1 + 1) + ib1 * lda], &lda, dum, &int1);
                i32 three = 3;
                SLC_DLASET("Lower", &sdim_m1, &sdim_m1, &ZERO, &ZERO, &a[(ib1 + 1) + ib1 * lda], &lda);
                /* Save and clear upper triangle of C */
                SLC_DLACPY("Upper", &sdim_m1, &sdim_m1, &c[ib1 + (ib1 + 1) * ldc], &ldc, &dum[3], &three);
                SLC_DLASET("Upper", &sdim_m1, &sdim_m1, &ZERO, &ZERO, &c[ib1 + (ib1 + 1) * ldc], &ldc);

                /* Update A */
                SLC_DGEMM("N", "N", &nrow, &sdim, &sdim, &ONE, &a[ib1 * lda], &lda,
                          &dwork[iq1], &sdim, &ZERO, &dwork[iwrk2], &nrow);
                SLC_DLACPY("Full", &nrow, &sdim, &dwork[iwrk2], &nrow, &a[ib1 * lda], &lda);
                SLC_DGEMM("T", "N", &sdim, &ncol, &sdim, &ONE, &dwork[iq2], &sdim,
                          &a[ib1 + ib1 * lda], &lda, &ZERO, &dwork[iwrk2], &sdim);
                SLC_DLACPY("Full", &sdim, &ncol, &dwork[iwrk2], &sdim, &a[ib1 + ib1 * lda], &lda);
                SLC_DLACPY("Lower", &sdim_m1, &sdim_m1, dum, &three, &a[(ib1 + 1) + ib1 * lda], &lda);

                /* Update C */
                SLC_DGEMM("N", "N", &ncol, &sdim, &sdim, &ONE, &c[ib1 + ib1 * ldc], &ldc,
                          &dwork[iq3], &sdim, &ZERO, &dwork[iwrk2], &ncol);
                SLC_DLACPY("Full", &ncol, &sdim, &dwork[iwrk2], &ncol, &c[ib1 + ib1 * ldc], &ldc);
                SLC_DGEMM("T", "N", &sdim, &nrow, &sdim, &ONE, &dwork[iq2], &sdim,
                          &c[ib1 * ldc], &ldc, &ZERO, &dwork[iwrk2], &sdim);
                SLC_DLACPY("Full", &sdim, &nrow, &dwork[iwrk2], &sdim, &c[ib1 * ldc], &ldc);
                SLC_DLACPY("Upper", &sdim_m1, &sdim_m1, &dum[3], &three, &c[ib1 + (ib1 + 1) * ldc], &ldc);

                /* Update D */
                SLC_DGEMM("N", "N", &m, &sdim, &sdim, &ONE, &d[ib1 * ldd], &ldd,
                          &dwork[iq3], &sdim, &ZERO, &dwork[iwrk2], &m);
                SLC_DLACPY("Full", &m, &sdim, &dwork[iwrk2], &m, &d[ib1 * ldd], &ldd);
                SLC_DGEMM("T", "N", &sdim, &m, &sdim, &ONE, &dwork[iq2], &sdim,
                          &d[ib1], &ldd, &ZERO, &dwork[iwrk2], &sdim);
                SLC_DLACPY("Full", &sdim, &m, &dwork[iwrk2], &sdim, &d[ib1], &ldd);

                /* Update B */
                i32 nrow_b, ncol_b, ibs_b, ldw_b;
                if (sdim > 2) {
                    nrow_b = nrows;
                    ncol_b = ncols;
                    ibs_b = ib3;
                    ldw_b = (nrow_b > 1) ? nrow_b : 1;
                } else {
                    ibs_b = ib1;
                    ldw_b = nrow;
                    nrow_b = nrow;
                    ncol_b = ncol;
                }
                SLC_DGEMM("N", "N", &nrow_b, &sdim, &sdim, &ONE, &b[ib1 * ldb], &ldb,
                          &dwork[iq1], &sdim, &ZERO, &dwork[iwrk2], &ldw_b);
                SLC_DLACPY("Full", &nrow_b, &sdim, &dwork[iwrk2], &ldw_b, &b[ib1 * ldb], &ldb);
                SLC_DGEMM("T", "N", &sdim, &ncol_b, &sdim, &ONE, &dwork[iq3], &sdim,
                          &b[ib1 + ibs_b * ldb], &ldb, &ZERO, &dwork[iwrk2], &sdim);
                SLC_DLACPY("Full", &sdim, &ncol_b, &dwork[iwrk2], &sdim, &b[ib1 + ibs_b * ldb], &ldb);

                /* Update F */
                i32 nrows_f = nrows;
                i32 ncols_f = ncols;
                SLC_DGEMM("N", "N", &nrows_f, &sdim, &sdim, &ONE, &f[ib1 * ldf], &ldf,
                          &dwork[iq3], &sdim, &ZERO, &dwork[iwrk2], &ldw_b);
                SLC_DLACPY("Full", &nrows_f, &sdim, &dwork[iwrk2], &ldw_b, &f[ib1 * ldf], &ldf);
                SLC_DGEMM("T", "N", &sdim, &ncols_f, &sdim, &ONE, &dwork[iq3], &sdim,
                          &f[ib1 + ib3 * ldf], &ldf, &ZERO, &dwork[iwrk2], &sdim);
                SLC_DLACPY("Full", &sdim, &ncols_f, &dwork[iwrk2], &sdim, &f[ib1 + ib3 * ldf], &ldf);

                i32 ldwork_ru = ldwork - iwrk2;
                mb01ru("Upper", "Transpose", sdim, sdim, ZERO, ONE,
                       &f[ib1 + ib1 * ldf], ldf, &dwork[iq3], sdim,
                       &f[ib1 + ib1 * ldf], ldf, &dwork[iwrk2], ldwork_ru, &info_cd);
                i32 ldf_p1 = ldf + 1;
                SLC_DSCAL(&sdim, &HALF, &f[ib1 + ib1 * ldf], &ldf_p1);

                if (lcmpq) {
                    /* Update Q */
                    SLC_DGEMM("N", "N", &upds, &sdim, &sdim, &ONE, &q[ib1 * ldq], &ldq,
                              &dwork[iq1], &sdim, &ZERO, &dwork[iwrk2], &upds);
                    SLC_DLACPY("Full", &upds, &sdim, &dwork[iwrk2], &upds, &q[ib1 * ldq], &ldq);
                    SLC_DGEMM("N", "N", &upds, &sdim, &sdim, &ONE, &q[iupd + (m + ib1) * ldq], &ldq,
                              &dwork[iq3], &sdim, &ZERO, &dwork[iwrk2], &upds);
                    SLC_DLACPY("Full", &upds, &sdim, &dwork[iwrk2], &upds, &q[iupd + (m + ib1) * ldq], &ldq);
                }

                if (lcmpu) {
                    /* Update U1 */
                    SLC_DGEMM("N", "N", &m, &sdim, &sdim, &ONE, &u1[ib1 * ldu1], &ldu1,
                              &dwork[iq2], &sdim, &ZERO, &dwork[iwrk2], &m);
                    SLC_DLACPY("Full", &m, &sdim, &dwork[iwrk2], &m, &u1[ib1 * ldu1], &ldu1);
                }

                if (lupdu) {
                    /* Update U2 */
                    SLC_DGEMM("N", "N", &m, &sdim, &sdim, &ONE, &u2[ib1 * ldu2], &ldu2,
                              &dwork[iq2], &sdim, &ZERO, &dwork[iwrk2], &m);
                    SLC_DLACPY("Full", &m, &sdim, &dwork[iwrk2], &m, &u2[ib1 * ldu2], &ldu2);
                }

                /* Update index lists */
                i32 hlp = dim2 - dim1;
                if (hlp == 1) {
                    iwork[jj + 1] = ib1 + 2;  /* 1-based */
                } else if (hlp == -1) {
                    iwork[jj + 1] = ib1 + 3;  /* 1-based */
                }

                /* Swap sign indicators */
                i32 tmp_sign = iwork[is_offset + jj];
                iwork[is_offset + jj] = iwork[is_offset + jj + 1];
                iwork[is_offset + jj + 1] = tmp_sign;
            }
            mm++;
        }
    }

    if (MB03ID_DEBUG) {
        fprintf(stderr, "MB03ID: After STEP 1 (Loop I), mm=%d, mp=%d\n", mm, mp);
    }

    /* II. Reorder eigenvalues with positive real parts to bottom */
    if (MB03ID_DEBUG) {
        fprintf(stderr, "MB03ID: Loop II starting with mp=%d, processing blocks %d down to %d\n", mp, r - 1, mm);
    }
    for (i32 k = r - 1; k >= mm; k--) {
        if (MB03ID_DEBUG) {
            fprintf(stderr, "MB03ID: Loop II k=%d, sign=%d\n", k, iwork[is_offset + k]);
        }
        if (iwork[is_offset + k] > 0) {
            /* Fortran: DO J = K, MP-2. C: jj = k..mp-3 (0-based) */
            for (i32 jj = k; jj < mp - 2; jj++) {
                i32 ib1 = iwork[jj] - 1;
                i32 ib2 = iwork[jj + 1] - 1;
                ib3 = iwork[jj + 2] - 1;
                i32 dim1 = ib2 - ib1;
                i32 dim2 = ib3 - ib2;
                i32 sdim = dim1 + dim2;

                /* Copy blocks to DWORK */
                SLC_DLACPY("Upper", &sdim, &sdim, &a[ib1 + ib1 * lda], &lda, &dwork[ia], &sdim);
                ma02ad("Lower", sdim, sdim, &c[ib1 + ib1 * ldc], ldc, &dwork[ic_w], sdim);
                SLC_DLACPY("Upper", &sdim, &sdim, &b[ib1 + ib1 * ldb], &ldb, &dwork[ib_w], &sdim);

                i32 sdim_m1 = sdim - 1;
                i32 ldb_p1 = ldb + 1;
                i32 sdim_p1 = sdim + 1;
                SLC_DCOPY(&sdim_m1, &b[(ib1 + 1) + ib1 * ldb], &ldb_p1, &dwork[ib_w + 1], &sdim_p1);

                if (dim1 == 2) {
                    dwork[ia + 1] = ZERO;
                    dwork[ic_w + 1] = ZERO;
                }
                if (dim2 == 2) {
                    i32 i1 = sdim * (sdim - 1) - 1;
                    dwork[ia + i1] = ZERO;
                    dwork[ic_w + i1] = ZERO;
                }
                dwork[ib_w + sdim - 1] = ZERO;
                if (sdim == 4) {
                    dwork[ib_w + 2] = ZERO;
                    dwork[ib_w + 7] = ZERO;
                }

                i32 info_cd;
                i32 ldwork_avail = ldwork - iwrk1;
                mb03cd("Upper", &dim1, &dim2, prec, &dwork[ic_w], sdim,
                       &dwork[ia], sdim, &dwork[ib_w], sdim, &dwork[iq1], sdim,
                       &dwork[iq2], sdim, &dwork[iq3], sdim, &dwork[iwrk1],
                       ldwork_avail, &info_cd);

                if (info_cd > 0) {
                    *info = 2;
                    return;
                }

                if (sdim > 2) {
                    SLC_DLACPY("Upper", &sdim, &sdim, &dwork[ib_w], &sdim, &b[ib1 + ib1 * ldb], &ldb);
                    SLC_DCOPY(&sdim_m1, &dwork[ib_w + 1], &sdim_p1, &b[(ib1 + 1) + ib1 * ldb], &ldb_p1);
                }

                i32 nrows = ib1;
                i32 ncols = m - ib3;
                i32 nrow = ib3;
                i32 ncol = m - ib1;

                f64 dum[12];
                i32 three = 3;
                SLC_DLACPY("Lower", &sdim_m1, &sdim_m1, &a[(ib1 + 1) + ib1 * lda], &lda, dum, &three);
                SLC_DLASET("Lower", &sdim_m1, &sdim_m1, &ZERO, &ZERO, &a[(ib1 + 1) + ib1 * lda], &lda);
                SLC_DLACPY("Upper", &sdim_m1, &sdim_m1, &c[ib1 + (ib1 + 1) * ldc], &ldc, &dum[3], &three);
                SLC_DLASET("Upper", &sdim_m1, &sdim_m1, &ZERO, &ZERO, &c[ib1 + (ib1 + 1) * ldc], &ldc);

                /* Update A */
                SLC_DGEMM("N", "N", &nrow, &sdim, &sdim, &ONE, &a[ib1 * lda], &lda,
                          &dwork[iq1], &sdim, &ZERO, &dwork[iwrk2], &nrow);
                SLC_DLACPY("Full", &nrow, &sdim, &dwork[iwrk2], &nrow, &a[ib1 * lda], &lda);
                SLC_DGEMM("T", "N", &sdim, &ncol, &sdim, &ONE, &dwork[iq2], &sdim,
                          &a[ib1 + ib1 * lda], &lda, &ZERO, &dwork[iwrk2], &sdim);
                SLC_DLACPY("Full", &sdim, &ncol, &dwork[iwrk2], &sdim, &a[ib1 + ib1 * lda], &lda);
                SLC_DLACPY("Lower", &sdim_m1, &sdim_m1, dum, &three, &a[(ib1 + 1) + ib1 * lda], &lda);

                /* Update C */
                SLC_DGEMM("N", "N", &ncol, &sdim, &sdim, &ONE, &c[ib1 + ib1 * ldc], &ldc,
                          &dwork[iq3], &sdim, &ZERO, &dwork[iwrk2], &ncol);
                SLC_DLACPY("Full", &ncol, &sdim, &dwork[iwrk2], &ncol, &c[ib1 + ib1 * ldc], &ldc);
                SLC_DGEMM("T", "N", &sdim, &nrow, &sdim, &ONE, &dwork[iq2], &sdim,
                          &c[ib1 * ldc], &ldc, &ZERO, &dwork[iwrk2], &sdim);
                SLC_DLACPY("Full", &sdim, &nrow, &dwork[iwrk2], &sdim, &c[ib1 * ldc], &ldc);
                SLC_DLACPY("Upper", &sdim_m1, &sdim_m1, &dum[3], &three, &c[ib1 + (ib1 + 1) * ldc], &ldc);

                /* Update D */
                SLC_DGEMM("N", "N", &m, &sdim, &sdim, &ONE, &d[ib1 * ldd], &ldd,
                          &dwork[iq3], &sdim, &ZERO, &dwork[iwrk2], &m);
                SLC_DLACPY("Full", &m, &sdim, &dwork[iwrk2], &m, &d[ib1 * ldd], &ldd);
                SLC_DGEMM("T", "N", &sdim, &m, &sdim, &ONE, &dwork[iq2], &sdim,
                          &d[ib1], &ldd, &ZERO, &dwork[iwrk2], &sdim);
                SLC_DLACPY("Full", &sdim, &m, &dwork[iwrk2], &sdim, &d[ib1], &ldd);

                /* Update B */
                i32 nrow_b, ncol_b, ibs_b, ldw_b;
                if (sdim > 2) {
                    nrow_b = nrows;
                    ncol_b = ncols;
                    ibs_b = ib3;
                    ldw_b = (nrow_b > 1) ? nrow_b : 1;
                } else {
                    ibs_b = ib1;
                    ldw_b = nrow;
                    nrow_b = nrow;
                    ncol_b = ncol;
                }
                SLC_DGEMM("N", "N", &nrow_b, &sdim, &sdim, &ONE, &b[ib1 * ldb], &ldb,
                          &dwork[iq1], &sdim, &ZERO, &dwork[iwrk2], &ldw_b);
                SLC_DLACPY("Full", &nrow_b, &sdim, &dwork[iwrk2], &ldw_b, &b[ib1 * ldb], &ldb);
                SLC_DGEMM("T", "N", &sdim, &ncol_b, &sdim, &ONE, &dwork[iq3], &sdim,
                          &b[ib1 + ibs_b * ldb], &ldb, &ZERO, &dwork[iwrk2], &sdim);
                SLC_DLACPY("Full", &sdim, &ncol_b, &dwork[iwrk2], &sdim, &b[ib1 + ibs_b * ldb], &ldb);

                /* Update F */
                i32 nrows_f = nrows;
                i32 ncols_f = ncols;
                SLC_DGEMM("N", "N", &nrows_f, &sdim, &sdim, &ONE, &f[ib1 * ldf], &ldf,
                          &dwork[iq3], &sdim, &ZERO, &dwork[iwrk2], &ldw_b);
                SLC_DLACPY("Full", &nrows_f, &sdim, &dwork[iwrk2], &ldw_b, &f[ib1 * ldf], &ldf);
                SLC_DGEMM("T", "N", &sdim, &ncols_f, &sdim, &ONE, &dwork[iq3], &sdim,
                          &f[ib1 + ib3 * ldf], &ldf, &ZERO, &dwork[iwrk2], &sdim);
                SLC_DLACPY("Full", &sdim, &ncols_f, &dwork[iwrk2], &sdim, &f[ib1 + ib3 * ldf], &ldf);

                i32 ldwork_ru = ldwork - iwrk2;
                i32 info_cd_dummy;
                mb01ru("Upper", "Transpose", sdim, sdim, ZERO, ONE,
                       &f[ib1 + ib1 * ldf], ldf, &dwork[iq3], sdim,
                       &f[ib1 + ib1 * ldf], ldf, &dwork[iwrk2], ldwork_ru, &info_cd_dummy);
                i32 ldf_p1 = ldf + 1;
                SLC_DSCAL(&sdim, &HALF, &f[ib1 + ib1 * ldf], &ldf_p1);

                if (lcmpq) {
                    SLC_DGEMM("N", "N", &upds, &sdim, &sdim, &ONE, &q[ib1 * ldq], &ldq,
                              &dwork[iq1], &sdim, &ZERO, &dwork[iwrk2], &upds);
                    SLC_DLACPY("Full", &upds, &sdim, &dwork[iwrk2], &upds, &q[ib1 * ldq], &ldq);
                    SLC_DGEMM("N", "N", &upds, &sdim, &sdim, &ONE, &q[iupd + (m + ib1) * ldq], &ldq,
                              &dwork[iq3], &sdim, &ZERO, &dwork[iwrk2], &upds);
                    SLC_DLACPY("Full", &upds, &sdim, &dwork[iwrk2], &upds, &q[iupd + (m + ib1) * ldq], &ldq);
                }

                if (lcmpu) {
                    SLC_DGEMM("N", "N", &m, &sdim, &sdim, &ONE, &u1[ib1 * ldu1], &ldu1,
                              &dwork[iq2], &sdim, &ZERO, &dwork[iwrk2], &m);
                    SLC_DLACPY("Full", &m, &sdim, &dwork[iwrk2], &m, &u1[ib1 * ldu1], &ldu1);
                }

                if (lupdu) {
                    SLC_DGEMM("N", "N", &m, &sdim, &sdim, &ONE, &u2[ib1 * ldu2], &ldu2,
                              &dwork[iq2], &sdim, &ZERO, &dwork[iwrk2], &m);
                    SLC_DLACPY("Full", &m, &sdim, &dwork[iwrk2], &m, &u2[ib1 * ldu2], &ldu2);
                }

                i32 hlp = dim2 - dim1;
                if (hlp == 1) {
                    iwork[jj + 1] = ib1 + 2;
                } else if (hlp == -1) {
                    iwork[jj + 1] = ib1 + 3;
                }
            }
            mp--;
        }
    }

    /* STEP 2: Reorder remaining eigenvalues with negative real parts */
    if (MB03ID_DEBUG) {
        fprintf(stderr, "MB03ID: Before STEP 2, mm=%d, mp=%d, r=%d\n", mm, mp, r);
        fprintf(stderr, "MB03ID: STEP 2 loop will iterate from k=%d down to %d\n", r - 1, mp - 1);
    }
    i32 iquple = 0;
    i32 iuuple = iquple + 16;
    i32 izuple = iuuple + 16;
    i32 ihuple = izuple + 16;
    i32 iwrk5 = ihuple + 16;
    i32 iwrk3 = izuple;
    i32 iwrk4 = iwrk3 + 2*n;
    i32 itmp1 = iwrk3 + n;
    i32 itmp2 = itmp1 + 4;
    i32 itmp3 = itmp2 + 4;

    for (i32 k = r - 1; k >= mp - 1; k--) {
        /* I. Exchange eigenvalues between two diagonal blocks */
        i32 ir = iwork[r - 1] - 1;  /* 0-based */
        i32 dim1 = iwork[r] - iwork[r - 1];
        i32 sdim = 2 * dim1;

        if (dim1 == 2) {
            a[(m - 1) + ir * lda] = ZERO;
            c[ir + (m - 1) * ldc] = ZERO;
            f[(m - 1) + ir * ldf] = f[ir + (m - 1) * ldf];
        }

        /* Calculate submatrix positions */
        i32 izupri = izuple + dim1 * sdim;
        i32 izlori = izupri + dim1;
        i32 iuupri = iuuple + dim1 * sdim;
        i32 iqlole = iquple + dim1;
        i32 iqupri = iquple + dim1 * sdim;
        i32 iqlori = iqupri + dim1;

        /* Build input matrices for MB03GD */
        SLC_DLACPY("Upper", &dim1, &dim1, &a[ir + ir * lda], &lda, &dwork[izuple], &sdim);
        SLC_DLACPY("Full", &dim1, &dim1, &d[ir + ir * ldd], &ldd, &dwork[izupri], &sdim);
        SLC_DLACPY("Lower", &dim1, &dim1, &c[ir + ir * ldc], &ldc, &dwork[izlori], &sdim);
        SLC_DLACPY("Full", &dim1, &dim1, &b[ir + ir * ldb], &ldb, &dwork[ihuple], &sdim);
        SLC_DLACPY("Upper", &dim1, &dim1, &f[ir + ir * ldf], &ldf, &dwork[ihuple + dim1 * sdim], &sdim);

        if (dim1 == 2) {
            dwork[izuple + 1] = ZERO;
            dwork[izlori + sdim] = ZERO;
        }

        /* Perform eigenvalue exchange */
        i32 ldwork_gd = ldwork - iwrk5;
        i32 info_gd;
        mb03gd(sdim, &dwork[izuple], sdim, &dwork[ihuple], sdim, par,
               &dwork[iquple], sdim, &dwork[iuuple], sdim, &dwork[iwrk5], ldwork_gd, &info_gd);

        if (info_gd > 0) {
            *info = 3;
            return;
        }

        if (dim1 == 2) {
            /* Full 2x2 block case - extensive updates */
            /* Update A by transformations from right */
            SLC_DLACPY("Full", &m, &dim1, &a[ir * lda], &lda, &dwork[iwrk3], &m);
            SLC_DGEMM("N", "N", &m, &dim1, &dim1, &ONE, &dwork[iwrk3], &m,
                      &dwork[iquple], &sdim, &ZERO, &a[ir * lda], &lda);
            SLC_DGEMM("N", "N", &m, &dim1, &dim1, &ONE, &d[ir * ldd], &ldd,
                      &dwork[iqlole], &sdim, &ONE, &a[ir * lda], &lda);

            /* Update D by transformations from right */
            SLC_DGEMM("N", "N", &m, &dim1, &dim1, &ONE, &dwork[iwrk3], &m,
                      &dwork[iqupri], &sdim, &ZERO, &dwork[itmp1], &m);
            SLC_DGEMM("N", "N", &m, &dim1, &dim1, &ONE, &d[ir * ldd], &ldd,
                      &dwork[iqlori], &sdim, &ONE, &dwork[itmp1], &m);
            SLC_DLACPY("Full", &m, &dim1, &dwork[itmp1], &m, &d[ir * ldd], &ldd);

            /* Compute Cf*Q21 */
            SLC_DGEMM("N", "N", &dim1, &dim1, &dim1, &ONE, &c[ir + ir * ldc], &ldc,
                      &dwork[iqlole], &sdim, &ZERO, &dwork[itmp1], &dim1);

            /* Update C by transformations from right */
            SLC_DGEMM("N", "N", &dim1, &dim1, &dim1, &ONE, &c[ir + ir * ldc], &ldc,
                      &dwork[iqlori], &sdim, &ZERO, &dwork[iwrk3], &dim1);
            SLC_DLACPY("Full", &dim1, &dim1, &dwork[iwrk3], &dim1, &c[ir + ir * ldc], &ldc);

            /* Update A by transformations from left */
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &ONE, &dwork[iuuple], &sdim,
                      &a[ir + ir * lda], &lda, &ZERO, &dwork[iwrk3], &dim1);
            f64 neg_one = -ONE;
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &neg_one, &dwork[iuupri], &sdim,
                      &dwork[itmp1], &dim1, &ONE, &dwork[iwrk3], &dim1);
            SLC_DLACPY("Full", &dim1, &dim1, &dwork[iwrk3], &dim1, &a[ir + ir * lda], &lda);

            /* Update D by transformations from left */
            SLC_DLACPY("Full", &dim1, &m, &d[ir], &ldd, &dwork[iwrk3], &dim1);
            SLC_DGEMM("T", "N", &dim1, &m, &dim1, &ONE, &dwork[iuuple], &sdim,
                      &dwork[iwrk3], &dim1, &ZERO, &d[ir], &ldd);
            SLC_DGEMM("T", "N", &dim1, &m, &dim1, &neg_one, &dwork[iuupri], &sdim,
                      &c[ir], &ldc, &ONE, &d[ir], &ldd);

            /* Update C by transformations from left */
            SLC_DGEMM("T", "N", &dim1, &m, &dim1, &ONE, &dwork[iuupri], &sdim,
                      &dwork[iwrk3], &dim1, &ZERO, &dwork[itmp1], &dim1);
            SLC_DGEMM("T", "N", &dim1, &m, &dim1, &ONE, &dwork[iuuple], &sdim,
                      &c[ir], &ldc, &ONE, &dwork[itmp1], &dim1);
            SLC_DLACPY("Full", &dim1, &m, &dwork[itmp1], &dim1, &c[ir], &ldc);

            /* Update B by transformations from right */
            SLC_DLACPY("Full", &m, &dim1, &b[ir * ldb], &ldb, &dwork[iwrk3], &m);
            SLC_DGEMM("N", "N", &m, &dim1, &dim1, &ONE, &dwork[iwrk3], &m,
                      &dwork[iquple], &sdim, &ZERO, &b[ir * ldb], &ldb);
            SLC_DGEMM("N", "N", &m, &dim1, &dim1, &ONE, &f[ir * ldf], &ldf,
                      &dwork[iqlole], &sdim, &ONE, &b[ir * ldb], &ldb);

            /* Update F by transformations from right */
            SLC_DGEMM("N", "N", &m, &dim1, &dim1, &ONE, &dwork[iwrk3], &m,
                      &dwork[iqupri], &sdim, &ZERO, &dwork[itmp1], &m);
            SLC_DGEMM("N", "N", &m, &dim1, &dim1, &ONE, &f[ir * ldf], &ldf,
                      &dwork[iqlori], &sdim, &ONE, &dwork[itmp1], &m);
            SLC_DLACPY("Full", &m, &dim1, &dwork[itmp1], &m, &f[ir * ldf], &ldf);

            /* Compute Bf'*Q21 and Bf'*Q22 */
            i32 offset_b = m - dim1;
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &ONE, &dwork[iwrk3 + offset_b], &m,
                      &dwork[iqlole], &sdim, &ZERO, &dwork[itmp1], &dim1);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &ONE, &dwork[iwrk3 + offset_b], &m,
                      &dwork[iqlori], &sdim, &ZERO, &dwork[itmp2], &dim1);

            /* Update B by transformations from left */
            SLC_DLACPY("Full", &dim1, &dim1, &b[ir + ir * ldb], &ldb, &dwork[itmp3], &dim1);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &ONE, &dwork[iqupri], &sdim,
                      &dwork[itmp1], &dim1, &ZERO, &b[ir + ir * ldb], &ldb);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &ONE, &dwork[iqlori], &sdim,
                      &dwork[itmp3], &dim1, &ONE, &b[ir + ir * ldb], &ldb);

            /* Update F by transformations from left */
            slicot_mb01rx('L', 'U', 'T', dim1, dim1, ZERO, ONE, &dwork[itmp1], dim1,
                         &dwork[iqlori], sdim, &f[ir + ir * ldf], ldf);
            slicot_mb01rx('L', 'U', 'T', dim1, dim1, ONE, ONE, &dwork[itmp1], dim1,
                         &dwork[iqupri], sdim, &dwork[itmp2], dim1);
            SLC_DLACPY("Upper", &dim1, &dim1, &dwork[itmp1], &dim1, &f[ir + ir * ldf], &ldf);

            if (lcmpq) {
                /* Update Q */
                SLC_DLACPY("Full", &n, &dim1, &q[ir * ldq], &ldq, &dwork[iwrk4], &n);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &dwork[iwrk4], &n,
                          &dwork[iquple], &sdim, &ZERO, &q[ir * ldq], &ldq);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &q[(m + ir) * ldq], &ldq,
                          &dwork[iqlole], &sdim, &ONE, &q[ir * ldq], &ldq);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &dwork[iwrk4], &n,
                          &dwork[iqupri], &sdim, &ZERO, &dwork[iwrk3], &n);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &ONE, &q[(m + ir) * ldq], &ldq,
                          &dwork[iqlori], &sdim, &ONE, &dwork[iwrk3], &n);
                SLC_DLACPY("Full", &n, &dim1, &dwork[iwrk3], &n, &q[(m + ir) * ldq], &ldq);
            }

            if (lcmpu) {
                /* Update U */
                SLC_DLACPY("Full", &m, &dim1, &u1[ir * ldu1], &ldu1, &dwork[itmp1], &m);
                SLC_DGEMM("N", "N", &m, &dim1, &dim1, &ONE, &dwork[itmp1], &m,
                          &dwork[iuuple], &sdim, &ZERO, &u1[ir * ldu1], &ldu1);
                SLC_DGEMM("N", "N", &m, &dim1, &dim1, &neg_one, &u2[ir * ldu2], &ldu2,
                          &dwork[iuupri], &sdim, &ONE, &u1[ir * ldu1], &ldu1);
                SLC_DGEMM("N", "N", &m, &dim1, &dim1, &ONE, &dwork[itmp1], &m,
                          &dwork[iuupri], &sdim, &ZERO, &dwork[iwrk3], &m);
                SLC_DGEMM("N", "N", &m, &dim1, &dim1, &ONE, &u2[ir * ldu2], &ldu2,
                          &dwork[iuuple], &sdim, &ONE, &dwork[iwrk3], &m);
                SLC_DLACPY("Full", &m, &dim1, &dwork[iwrk3], &m, &u2[ir * ldu2], &ldu2);
            }
        } else {
            /* 1x1 block case */
            f64 u11 = dwork[iuuple];
            f64 u12 = dwork[iuupri];
            f64 q11 = dwork[iquple];
            f64 q21 = dwork[iqlole];
            f64 q12 = dwork[iqupri];
            f64 q22 = dwork[iqlori];

            /* Update A by transformations from right */
            SLC_DCOPY(&m, &a[ir * lda], &int1, &dwork[iwrk3], &int1);
            SLC_DSCAL(&m, &q11, &a[ir * lda], &int1);
            SLC_DAXPY(&m, &q21, &d[ir * ldd], &int1, &a[ir * lda], &int1);

            /* Update D by transformations from right */
            SLC_DSCAL(&m, &q22, &d[ir * ldd], &int1);
            SLC_DAXPY(&m, &q12, &dwork[iwrk3], &int1, &d[ir * ldd], &int1);

            /* Update C by transformations from right */
            f64 tmpc = c[ir + ir * ldc] * q21;
            c[ir + ir * ldc] *= q22;

            /* Update A by transformations from left */
            a[ir + ir * lda] = u11 * a[ir + ir * lda] - u12 * tmpc;

            /* Update D by transformations from left */
            SLC_DCOPY(&m, &d[ir], &ldd, &dwork[iwrk3], &int1);
            SLC_DSCAL(&m, &u11, &d[ir], &ldd);
            f64 neg_u12 = -u12;
            SLC_DAXPY(&m, &neg_u12, &c[ir], &ldc, &d[ir], &ldd);

            /* Update C by transformations from left */
            SLC_DSCAL(&m, &u11, &c[ir], &ldc);
            SLC_DAXPY(&m, &u12, &dwork[iwrk3], &int1, &c[ir], &ldc);

            /* Update B by transformations from right */
            i32 m_m1 = m - 1;
            SLC_DCOPY(&m_m1, &b[ir * ldb], &int1, &dwork[iwrk3], &int1);
            SLC_DSCAL(&m_m1, &q11, &b[ir * ldb], &int1);
            SLC_DAXPY(&m_m1, &q21, &f[ir * ldf], &int1, &b[ir * ldb], &int1);

            /* Update F by transformations from right */
            SLC_DSCAL(&m_m1, &q22, &f[ir * ldf], &int1);
            SLC_DAXPY(&m_m1, &q12, &dwork[iwrk3], &int1, &f[ir * ldf], &int1);

            /* Update B by transformations from left */
            b[(m - 1) + (m - 1) * ldb] = -b[(m - 1) + (m - 1) * ldb];

            if (lcmpq) {
                /* Update Q */
                SLC_DCOPY(&n, &q[ir * ldq], &int1, &dwork[iwrk4], &int1);
                SLC_DSCAL(&n, &q11, &q[ir * ldq], &int1);
                SLC_DAXPY(&n, &q21, &q[(ir + m) * ldq], &int1, &q[ir * ldq], &int1);
                SLC_DSCAL(&n, &q22, &q[(ir + m) * ldq], &int1);
                SLC_DAXPY(&n, &q12, &dwork[iwrk4], &int1, &q[(ir + m) * ldq], &int1);
            }

            if (lcmpu) {
                /* Update U */
                SLC_DCOPY(&m, &u1[ir * ldu1], &int1, &dwork[iwrk4], &int1);
                SLC_DSCAL(&m, &u11, &u1[ir * ldu1], &int1);
                SLC_DAXPY(&m, &neg_u12, &u2[ir * ldu2], &int1, &u1[ir * ldu1], &int1);
                SLC_DSCAL(&m, &u11, &u2[ir * ldu2], &int1);
                SLC_DAXPY(&m, &u12, &dwork[iwrk4], &int1, &u2[ir * ldu2], &int1);
            }
        }

        mm++;

        /* Inner loop for block swapping */
        for (i32 jj = r - 2; jj >= mm - 1; jj--) {
            i32 ib1 = iwork[jj] - 1;
            i32 ib2 = iwork[jj + 1] - 1;
            ib3 = iwork[jj + 2] - 1;
            dim1 = ib2 - ib1;
            i32 dim2 = ib3 - ib2;
            sdim = dim1 + dim2;

            /* Copy blocks to DWORK */
            SLC_DLACPY("Upper", &sdim, &sdim, &a[ib1 + ib1 * lda], &lda, &dwork[ia], &sdim);
            ma02ad("Lower", sdim, sdim, &c[ib1 + ib1 * ldc], ldc, &dwork[ic_w], sdim);
            SLC_DLACPY("Upper", &sdim, &sdim, &b[ib1 + ib1 * ldb], &ldb, &dwork[ib_w], &sdim);

            i32 sdim_m1 = sdim - 1;
            i32 ldb_p1 = ldb + 1;
            i32 sdim_p1 = sdim + 1;
            SLC_DCOPY(&sdim_m1, &b[(ib1 + 1) + ib1 * ldb], &ldb_p1, &dwork[ib_w + 1], &sdim_p1);

            if (dim1 == 2) {
                dwork[ia + 1] = ZERO;
                dwork[ic_w + 1] = ZERO;
            }
            if (dim2 == 2) {
                i32 i1 = sdim * (sdim - 1) - 1;
                dwork[ia + i1] = ZERO;
                dwork[ic_w + i1] = ZERO;
            }
            dwork[ib_w + sdim - 1] = ZERO;
            if (sdim == 4) {
                dwork[ib_w + 2] = ZERO;
                dwork[ib_w + 7] = ZERO;
            }

            /* Perform eigenvalue exchange */
            i32 info_cd;
            i32 ldwork_avail = ldwork - iwrk1;
            mb03cd("Upper", &dim1, &dim2, prec, &dwork[ic_w], sdim,
                   &dwork[ia], sdim, &dwork[ib_w], sdim, &dwork[iq1], sdim,
                   &dwork[iq2], sdim, &dwork[iq3], sdim, &dwork[iwrk1],
                   ldwork_avail, &info_cd);

            if (info_cd > 0) {
                *info = 2;
                return;
            }

            if (sdim > 2) {
                SLC_DLACPY("Upper", &sdim, &sdim, &dwork[ib_w], &sdim, &b[ib1 + ib1 * ldb], &ldb);
                SLC_DCOPY(&sdim_m1, &dwork[ib_w + 1], &sdim_p1, &b[(ib1 + 1) + ib1 * ldb], &ldb_p1);
            }

            i32 nrows = ib1;
            i32 ncols = m - ib3;
            i32 nrow = ib3;
            i32 ncol = m - ib1;

            f64 dum[12];
            i32 three = 3;
            SLC_DLACPY("Lower", &sdim_m1, &sdim_m1, &a[(ib1 + 1) + ib1 * lda], &lda, dum, &three);
            SLC_DLASET("Lower", &sdim_m1, &sdim_m1, &ZERO, &ZERO, &a[(ib1 + 1) + ib1 * lda], &lda);
            SLC_DLACPY("Upper", &sdim_m1, &sdim_m1, &c[ib1 + (ib1 + 1) * ldc], &ldc, &dum[3], &three);
            SLC_DLASET("Upper", &sdim_m1, &sdim_m1, &ZERO, &ZERO, &c[ib1 + (ib1 + 1) * ldc], &ldc);

            /* Update A */
            SLC_DGEMM("N", "N", &nrow, &sdim, &sdim, &ONE, &a[ib1 * lda], &lda,
                      &dwork[iq1], &sdim, &ZERO, &dwork[iwrk2], &nrow);
            SLC_DLACPY("Full", &nrow, &sdim, &dwork[iwrk2], &nrow, &a[ib1 * lda], &lda);
            SLC_DGEMM("T", "N", &sdim, &ncol, &sdim, &ONE, &dwork[iq2], &sdim,
                      &a[ib1 + ib1 * lda], &lda, &ZERO, &dwork[iwrk2], &sdim);
            SLC_DLACPY("Full", &sdim, &ncol, &dwork[iwrk2], &sdim, &a[ib1 + ib1 * lda], &lda);
            SLC_DLACPY("Lower", &sdim_m1, &sdim_m1, dum, &three, &a[(ib1 + 1) + ib1 * lda], &lda);

            /* Update C */
            SLC_DGEMM("N", "N", &ncol, &sdim, &sdim, &ONE, &c[ib1 + ib1 * ldc], &ldc,
                      &dwork[iq3], &sdim, &ZERO, &dwork[iwrk2], &ncol);
            SLC_DLACPY("Full", &ncol, &sdim, &dwork[iwrk2], &ncol, &c[ib1 + ib1 * ldc], &ldc);
            SLC_DGEMM("T", "N", &sdim, &nrow, &sdim, &ONE, &dwork[iq2], &sdim,
                      &c[ib1 * ldc], &ldc, &ZERO, &dwork[iwrk2], &sdim);
            SLC_DLACPY("Full", &sdim, &nrow, &dwork[iwrk2], &sdim, &c[ib1 * ldc], &ldc);
            SLC_DLACPY("Upper", &sdim_m1, &sdim_m1, &dum[3], &three, &c[ib1 + (ib1 + 1) * ldc], &ldc);

            /* Update D */
            SLC_DGEMM("N", "N", &m, &sdim, &sdim, &ONE, &d[ib1 * ldd], &ldd,
                      &dwork[iq3], &sdim, &ZERO, &dwork[iwrk2], &m);
            SLC_DLACPY("Full", &m, &sdim, &dwork[iwrk2], &m, &d[ib1 * ldd], &ldd);
            SLC_DGEMM("T", "N", &sdim, &m, &sdim, &ONE, &dwork[iq2], &sdim,
                      &d[ib1], &ldd, &ZERO, &dwork[iwrk2], &sdim);
            SLC_DLACPY("Full", &sdim, &m, &dwork[iwrk2], &sdim, &d[ib1], &ldd);

            /* Update B */
            i32 nrow_b, ncol_b, ibs_b, ldw_b;
            if (sdim > 2) {
                nrow_b = nrows;
                ncol_b = ncols;
                ibs_b = ib3;
                ldw_b = (nrow_b > 1) ? nrow_b : 1;
            } else {
                ibs_b = ib1;
                ldw_b = nrow;
                nrow_b = nrow;
                ncol_b = ncol;
            }
            SLC_DGEMM("N", "N", &nrow_b, &sdim, &sdim, &ONE, &b[ib1 * ldb], &ldb,
                      &dwork[iq1], &sdim, &ZERO, &dwork[iwrk2], &ldw_b);
            SLC_DLACPY("Full", &nrow_b, &sdim, &dwork[iwrk2], &ldw_b, &b[ib1 * ldb], &ldb);
            SLC_DGEMM("T", "N", &sdim, &ncol_b, &sdim, &ONE, &dwork[iq3], &sdim,
                      &b[ib1 + ibs_b * ldb], &ldb, &ZERO, &dwork[iwrk2], &sdim);
            SLC_DLACPY("Full", &sdim, &ncol_b, &dwork[iwrk2], &sdim, &b[ib1 + ibs_b * ldb], &ldb);

            /* Update F */
            i32 nrows_f = nrows;
            i32 ncols_f = ncols;
            SLC_DGEMM("N", "N", &nrows_f, &sdim, &sdim, &ONE, &f[ib1 * ldf], &ldf,
                      &dwork[iq3], &sdim, &ZERO, &dwork[iwrk2], &ldw_b);
            SLC_DLACPY("Full", &nrows_f, &sdim, &dwork[iwrk2], &ldw_b, &f[ib1 * ldf], &ldf);
            SLC_DGEMM("T", "N", &sdim, &ncols_f, &sdim, &ONE, &dwork[iq3], &sdim,
                      &f[ib1 + ib3 * ldf], &ldf, &ZERO, &dwork[iwrk2], &sdim);
            SLC_DLACPY("Full", &sdim, &ncols_f, &dwork[iwrk2], &sdim, &f[ib1 + ib3 * ldf], &ldf);

            i32 ldwork_ru = ldwork - iwrk2;
            i32 info_ru;
            mb01ru("Upper", "Transpose", sdim, sdim, ZERO, ONE,
                   &f[ib1 + ib1 * ldf], ldf, &dwork[iq3], sdim,
                   &f[ib1 + ib1 * ldf], ldf, &dwork[iwrk2], ldwork_ru, &info_ru);
            i32 ldf_p1 = ldf + 1;
            SLC_DSCAL(&sdim, &HALF, &f[ib1 + ib1 * ldf], &ldf_p1);

            if (lcmpq) {
                SLC_DGEMM("N", "N", &n, &sdim, &sdim, &ONE, &q[ib1 * ldq], &ldq,
                          &dwork[iq1], &sdim, &ZERO, &dwork[iwrk2], &n);
                SLC_DLACPY("Full", &n, &sdim, &dwork[iwrk2], &n, &q[ib1 * ldq], &ldq);
                SLC_DGEMM("N", "N", &n, &sdim, &sdim, &ONE, &q[(m + ib1) * ldq], &ldq,
                          &dwork[iq3], &sdim, &ZERO, &dwork[iwrk2], &n);
                SLC_DLACPY("Full", &n, &sdim, &dwork[iwrk2], &n, &q[(m + ib1) * ldq], &ldq);
            }

            if (lcmpu) {
                SLC_DGEMM("N", "N", &m, &sdim, &sdim, &ONE, &u1[ib1 * ldu1], &ldu1,
                          &dwork[iq2], &sdim, &ZERO, &dwork[iwrk2], &m);
                SLC_DLACPY("Full", &m, &sdim, &dwork[iwrk2], &m, &u1[ib1 * ldu1], &ldu1);
                SLC_DGEMM("N", "N", &m, &sdim, &sdim, &ONE, &u2[ib1 * ldu2], &ldu2,
                          &dwork[iq2], &sdim, &ZERO, &dwork[iwrk2], &m);
                SLC_DLACPY("Full", &m, &sdim, &dwork[iwrk2], &m, &u2[ib1 * ldu2], &ldu2);
            }

            /* Update index list */
            i32 hlp = dim2 - dim1;
            if (hlp == 1) {
                iwork[jj + 1] = ib1 + 2;
            } else if (hlp == -1) {
                iwork[jj + 1] = ib1 + 3;
            }
        }
    }

    /* Restore elements */
    if (m > 1) {
        a[(m - 1) + (m - 2) * lda] = a2;
        c[(m - 2) + (m - 1) * ldc] = c2;
        f[(m - 1) + (m - 2) * ldf] = f2;
    }

    if (MB03ID_DEBUG) {
        fprintf(stderr, "MB03ID: Final mm=%d, iwork[mm]=%d\n", mm, mm > 0 ? iwork[mm] : 0);
    }

    if (mm > 0) {
        *neig = iwork[mm] - 1;  /* Convert to count */
    } else {
        *neig = 0;
    }

    if (MB03ID_DEBUG) {
        fprintf(stderr, "MB03ID: Returning neig=%d\n", *neig);
    }
}
