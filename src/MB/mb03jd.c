/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB03JD - Eigenvalue reordering for real skew-Hamiltonian/Hamiltonian pencil
 *
 * Moves eigenvalues with strictly negative real parts of an N-by-N real
 * skew-Hamiltonian/Hamiltonian pencil aS - bH in structured Schur form to
 * the leading principal subpencil, while keeping the triangular form.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define MB03JD_DEBUG 0

void mb03jd(const char* compq, const i32 n, f64* a, const i32 lda,
            f64* d, const i32 ldd, f64* b, const i32 ldb,
            f64* f, const i32 ldf, f64* q, const i32 ldq,
            i32* neig, i32* iwork, const i32 liwork,
            f64* dwork, const i32 ldwork, i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 HALF = 0.5;
    const f64 TEN = 10.0;

    bool liniq, lupdq, lcmpq;
    i32 m, optdw;
    i32 dim1, dim2, hlp, i, ia, iauple, ib, ib1, ib2, ib3, ibuple, ibupri;
    i32 ic, ics, iq1, iq2, iqlole, iqlori, iquple, iqupri, ir, is;
    i32 itmp1, itmp2, itmp3, iupd, iwrk1, iwrk2, iwrk3, iwrk4, iwrk5;
    i32 j, k, ldw, mm, mp, ncol, ncols, nrow, nrows, r, sdim, upds;
    f64 a2, d1, d2, d3, f2, nrma, nrmb, prec, q11, q12, q21, q22, tmp, tol;
    f64 par[2];

    i32 int1 = 1;
    f64 dbl0 = 0.0, dbl1 = 1.0, neg1 = -1.0;

    m = n / 2;
    liniq = (compq[0] == 'I' || compq[0] == 'i');
    lupdq = (compq[0] == 'U' || compq[0] == 'u');
    lcmpq = liniq || lupdq;

    if (lcmpq) {
        optdw = (4 * n + 32 > 108) ? 4 * n + 32 : 108;
    } else {
        optdw = (2 * n + 32 > 108) ? 2 * n + 32 : 108;
    }

    *info = 0;
    if (!(compq[0] == 'N' || compq[0] == 'n' || lcmpq)) {
        *info = -1;
    } else if (n < 0 || (n % 2) != 0) {
        *info = -2;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -4;
    } else if (ldd < (1 > m ? 1 : m)) {
        *info = -6;
    } else if (ldb < (1 > m ? 1 : m)) {
        *info = -8;
    } else if (ldf < (1 > m ? 1 : m)) {
        *info = -10;
    } else if (ldq < 1 || (lcmpq && ldq < n)) {
        *info = -12;
    } else if (liwork < n + 1) {
        *info = -15;
    } else if (ldwork < optdw) {
        *info = -17;
    }

    if (*info != 0) {
        i32 neg_info = -(*info);
        SLC_XERBLA("MB03JD", &neg_info);
        return;
    }

    if (n == 0) {
        *neig = 0;
        return;
    }

    prec = SLC_DLAMCH("Precision");
    tol = ((f64)n < TEN ? (f64)n : TEN) * prec;

    par[0] = prec;
    par[1] = SLC_DLAMCH("Safe minimum");

    i = 1;
    j = 1;
    is = m + 1;

    nrma = SLC_DLANTR("One", "Upper", "Non-diag", &m, &m, a, &lda, dwork);
    nrmb = SLC_DLANHS("One", &m, b, &ldb, dwork);

    while (i <= m - 1) {
        iwork[j - 1] = i;
        if (fabs(b[i + (i - 1) * ldb]) <= tol * nrmb) {
            b[i + (i - 1) * ldb] = ZERO;
            if (fabs(a[(i - 1) + (i - 1) * lda]) <= tol * nrma ||
                fabs(b[(i - 1) + (i - 1) * ldb]) <= tol * nrmb) {
                iwork[is + j - 1] = 0;
#if MB03JD_DEBUG
                fprintf(stderr, "  Block %d (1x1 at i=%d): A=%e, B=%e -> sign=0 (singular)\n",
                        j, i, a[(i-1)+(i-1)*lda], b[(i-1)+(i-1)*ldb]);
#endif
            } else {
                f64 prod = a[(i - 1) + (i - 1) * lda] * b[(i - 1) + (i - 1) * ldb];
                iwork[is + j - 1] = (i32)(prod < 0 ? -ONE : (prod > 0 ? ONE : ZERO));
#if MB03JD_DEBUG
                fprintf(stderr, "  Block %d (1x1 at i=%d): A=%e, B=%e, prod=%e -> sign=%d\n",
                        j, i, a[(i-1)+(i-1)*lda], b[(i-1)+(i-1)*ldb], prod, iwork[is+j-1]);
#endif
            }
            i = i + 1;
        } else {
            if (a[(i - 1) + (i - 1) * lda] == ZERO || a[i + i * lda] == ZERO) {
                iwork[is + j - 1] = 0;
#if MB03JD_DEBUG
                fprintf(stderr, "  Block %d (2x2 at i=%d): A diagonal zero -> sign=0\n", j, i);
#endif
            } else {
                tmp = (b[(i - 1) + (i - 1) * ldb] -
                       (b[i + (i - 1) * ldb] / a[i + i * lda]) * a[(i - 1) + i * lda]) /
                          a[(i - 1) + (i - 1) * lda] +
                      b[i + i * ldb] / a[i + i * lda];
#if MB03JD_DEBUG
                f64 b12 = b[(i-1)+i*ldb];
                fprintf(stderr, "  Block %d (2x2 at i=%d): A11=%e, A12=%e, A22=%e, B11=%e, B12=%e, B21=%e, B22=%e, trace=%e, B12*B21=%e -> sign=%d\n",
                        j, i, a[(i-1)+(i-1)*lda], a[(i-1)+i*lda], a[i+i*lda],
                        b[(i-1)+(i-1)*ldb], b12, b[i+(i-1)*ldb], b[i+i*ldb],
                        tmp, b12 * b[i+(i-1)*ldb], (tmp == ZERO ? 0 : (tmp < 0 ? -1 : 1)));
#endif
                if (tmp == ZERO) {
                    // trace=0 can mean:
                    // 1. Pure imaginary eigenvalues +-i*mu (B12*B21 < 0) -> no negative real part
                    // 2. Real eigenvalue pair +-lambda (B12*B21 > 0) -> one negative real part
                    // Check B12*B21 to distinguish
                    f64 b12 = b[(i - 1) + i * ldb];
                    f64 b21 = b[i + (i - 1) * ldb];
                    f64 prod_b = b12 * b21;
                    if (prod_b > ZERO) {
                        // Real eigenvalue pair +-lambda: one is negative
                        // Treat as having negative eigenvalue so it gets processed in STEP 2
                        iwork[is + j - 1] = -1;
#if MB03JD_DEBUG
                        fprintf(stderr, "    -> B12*B21=%e > 0, real pair +-lambda, setting sign=-1\n", prod_b);
#endif
                    } else {
                        // Pure imaginary eigenvalues or zero
                        iwork[is + j - 1] = 0;
#if MB03JD_DEBUG
                        fprintf(stderr, "    -> B12*B21=%e <= 0, pure imaginary, keeping sign=0\n", prod_b);
#endif
                    }
                } else {
                    iwork[is + j - 1] = (i32)(tmp < 0 ? -ONE : ONE);
                }
            }
            i = i + 2;
        }
        j = j + 1;
    }

    if (i == m) {
        iwork[j - 1] = i;
        if (fabs(a[(i - 1) + (i - 1) * lda]) <= tol * nrma ||
            fabs(b[(i - 1) + (i - 1) * ldb]) <= tol * nrmb) {
            iwork[is + j - 1] = 0;
        } else {
            f64 prod = a[(i - 1) + (i - 1) * lda] * b[(i - 1) + (i - 1) * ldb];
            iwork[is + j - 1] = (i32)(prod < 0 ? -ONE : (prod > 0 ? ONE : ZERO));
        }
        j = j + 1;
    }

    r = j - 1;

#if MB03JD_DEBUG
    fprintf(stderr, "MB03JD: n=%d, m=%d, r=%d blocks\n", n, m, r);
    fprintf(stderr, "Block indices (iwork[0..r-1]):\n");
    for (int dbg_i = 0; dbg_i < r; dbg_i++) {
        fprintf(stderr, "  block %d: start=%d, sign=%d\n",
                dbg_i+1, iwork[dbg_i], iwork[is + dbg_i]);
    }
    fprintf(stderr, "  iwork[r]=%d (end marker)\n", iwork[r]);
#endif

    if (liniq) {
        iupd = m + 1;
        upds = m;
        SLC_DLASET("Full", &n, &n, &dbl0, &dbl1, q, &ldq);
    } else if (lupdq) {
        iupd = 1;
        upds = n;
    }

    if (m > 1) {
        d1 = d[(m - 2) + (m - 2) * ldd];
        d2 = d[(m - 1) + (m - 2) * ldd];
        d3 = d[(m - 1) + (m - 1) * ldd];
        a2 = a[(m - 1) + (m - 2) * lda];
        f2 = f[(m - 1) + (m - 2) * ldf];
    }

    mm = 0;
    mp = j;

    iq1 = 0;
    iq2 = iq1 + 16;
    ia = iq2 + 16;
    ib = ia + 16;
    iwrk1 = ib + 16;
    iwrk2 = ia;

    k = 1;
    ib3 = m + 1;
    iwork[r] = ib3;

#if MB03JD_DEBUG
    fprintf(stderr, "Before STEP 1: r=%d, ib3=%d, iwork[r]=%d\n", r, ib3, iwork[r]);
#endif

    while (k <= r) {
        if (iwork[is + k - 1] < 0) {
            for (j = k - 1; j >= mm + 1; j--) {
                ib1 = iwork[j - 1];
                ib2 = iwork[j];
                ib3 = iwork[j + 1];
                dim1 = ib2 - ib1;
                dim2 = ib3 - ib2;
                sdim = dim1 + dim2;

                SLC_DLACPY("Upper", &sdim, &sdim, &a[(ib1 - 1) + (ib1 - 1) * lda], &lda,
                           &dwork[ia], &sdim);
                i32 sdim_m1 = sdim - 1;
                SLC_DLASET("Lower", &sdim_m1, &sdim_m1, &dbl0, &dbl0,
                           &dwork[ia + 1], &sdim);
                SLC_DLACPY("Upper", &sdim, &sdim, &b[(ib1 - 1) + (ib1 - 1) * ldb], &ldb,
                           &dwork[ib], &sdim);
                i32 ldb_p1 = ldb + 1;
                i32 sdim_p1 = sdim + 1;
                SLC_DCOPY(&sdim_m1, &b[ib1 + (ib1 - 1) * ldb], &ldb_p1,
                          &dwork[ib + 1], &sdim_p1);
                i32 sdim_m2 = sdim - 2;
                SLC_DLASET("Lower", &sdim_m2, &sdim_m2, &dbl0, &dbl0,
                           &dwork[ib + 2], &sdim);

                i32 iwrk1_size = ldwork - iwrk1;
                mb03dd("Triangular", &dim1, &dim2, prec, &dwork[ib], sdim,
                       &dwork[ia], sdim, &dwork[iq1], sdim,
                       &dwork[iq2], sdim, &dwork[iwrk1], iwrk1_size, info);
                if (*info > 0) {
                    *info = 1;
                    return;
                }

                nrows = ib1 - 1;
                ncols = m - ib3 + 1;
                ics = ib3;

                if (sdim > 2) {
                    SLC_DLACPY("Upper", &sdim, &sdim, &dwork[ia], &sdim,
                               &a[(ib1 - 1) + (ib1 - 1) * lda], &lda);
                    SLC_DLACPY("Upper", &sdim, &sdim, &dwork[ib], &sdim,
                               &b[(ib1 - 1) + (ib1 - 1) * ldb], &ldb);
                    SLC_DCOPY(&sdim_m1, &dwork[ib + 1], &sdim_p1,
                              &b[ib1 + (ib1 - 1) * ldb], &ldb_p1);
                    nrow = nrows;
                    ncol = ncols;
                    ic = ics;
                    ldw = (1 > nrow) ? 1 : nrow;
                } else {
                    tmp = a[ib1 + (ib1 - 1) * lda];
                    a[ib1 + (ib1 - 1) * lda] = ZERO;
                    nrow = ib3 - 1;
                    ncol = m - ib1 + 1;
                    ic = ib1;
                    ldw = nrow;
                }

                SLC_DGEMM("N", "N", &nrow, &sdim, &sdim, &dbl1,
                          &a[(ib1 - 1) * lda], &lda, &dwork[iq1], &sdim,
                          &dbl0, &dwork[iwrk2], &ldw);
                SLC_DLACPY("Full", &nrow, &sdim, &dwork[iwrk2], &ldw,
                           &a[(ib1 - 1) * lda], &lda);
                SLC_DGEMM("T", "N", &sdim, &ncol, &sdim, &dbl1,
                          &dwork[iq2], &sdim, &a[(ib1 - 1) + (ic - 1) * lda], &lda,
                          &dbl0, &dwork[iwrk2], &sdim);
                SLC_DLACPY("Full", &sdim, &ncol, &dwork[iwrk2], &sdim,
                           &a[(ib1 - 1) + (ic - 1) * lda], &lda);
                if (sdim == 2) {
                    a[ib1 + (ib1 - 1) * lda] = tmp;
                }

                SLC_DGEMM("N", "N", &nrows, &sdim, &sdim, &dbl1,
                          &d[(ib1 - 1) * ldd], &ldd, &dwork[iq2], &sdim,
                          &dbl0, &dwork[iwrk2], &ldw);
                SLC_DLACPY("Full", &nrows, &sdim, &dwork[iwrk2], &ldw,
                           &d[(ib1 - 1) * ldd], &ldd);
                SLC_DGEMM("T", "N", &sdim, &ncols, &sdim, &dbl1,
                          &dwork[iq2], &sdim, &d[(ib1 - 1) + (ics - 1) * ldd], &ldd,
                          &dbl0, &dwork[iwrk2], &sdim);
                SLC_DLACPY("Full", &sdim, &ncols, &dwork[iwrk2], &sdim,
                           &d[(ib1 - 1) + (ics - 1) * ldd], &ldd);
                i32 iwrk2_size = ldwork - iwrk2;
                mb01ld("Upper", "Transpose", sdim, sdim, ZERO, ONE,
                       &d[(ib1 - 1) + (ib1 - 1) * ldd], ldd, &dwork[iq2], sdim,
                       &d[(ib1 - 1) + (ib1 - 1) * ldd], ldd, &dwork[iwrk2],
                       iwrk2_size, info);

                SLC_DGEMM("N", "N", &nrow, &sdim, &sdim, &dbl1,
                          &b[(ib1 - 1) * ldb], &ldb, &dwork[iq1], &sdim,
                          &dbl0, &dwork[iwrk2], &ldw);
                SLC_DLACPY("Full", &nrow, &sdim, &dwork[iwrk2], &ldw,
                           &b[(ib1 - 1) * ldb], &ldb);
                SLC_DGEMM("T", "N", &sdim, &ncol, &sdim, &dbl1,
                          &dwork[iq2], &sdim, &b[(ib1 - 1) + (ic - 1) * ldb], &ldb,
                          &dbl0, &dwork[iwrk2], &sdim);
                SLC_DLACPY("Full", &sdim, &ncol, &dwork[iwrk2], &sdim,
                           &b[(ib1 - 1) + (ic - 1) * ldb], &ldb);

                SLC_DGEMM("N", "N", &nrows, &sdim, &sdim, &dbl1,
                          &f[(ib1 - 1) * ldf], &ldf, &dwork[iq2], &sdim,
                          &dbl0, &dwork[iwrk2], &ldw);
                SLC_DLACPY("Full", &nrows, &sdim, &dwork[iwrk2], &ldw,
                           &f[(ib1 - 1) * ldf], &ldf);
                SLC_DGEMM("T", "N", &sdim, &ncols, &sdim, &dbl1,
                          &dwork[iq2], &sdim, &f[(ib1 - 1) + (ics - 1) * ldf], &ldf,
                          &dbl0, &dwork[iwrk2], &sdim);
                SLC_DLACPY("Full", &sdim, &ncols, &dwork[iwrk2], &sdim,
                           &f[(ib1 - 1) + (ics - 1) * ldf], &ldf);
                mb01ru("Upper", "Transpose", sdim, sdim, ZERO, ONE,
                       &f[(ib1 - 1) + (ib1 - 1) * ldf], ldf, &dwork[iq2], sdim,
                       &f[(ib1 - 1) + (ib1 - 1) * ldf], ldf, &dwork[iwrk2],
                       iwrk2_size, info);
                i32 ldf_p1 = ldf + 1;
                SLC_DSCAL(&sdim, &HALF, &f[(ib1 - 1) + (ib1 - 1) * ldf], &ldf_p1);

                if (lcmpq) {
                    SLC_DGEMM("N", "N", &upds, &sdim, &sdim, &dbl1,
                              &q[(ib1 - 1) * ldq], &ldq, &dwork[iq1], &sdim,
                              &dbl0, &dwork[iwrk2], &upds);
                    SLC_DLACPY("Full", &upds, &sdim, &dwork[iwrk2], &upds,
                               &q[(ib1 - 1) * ldq], &ldq);
                    SLC_DGEMM("N", "N", &upds, &sdim, &sdim, &dbl1,
                              &q[(iupd - 1) + (m + ib1 - 1) * ldq], &ldq,
                              &dwork[iq2], &sdim, &dbl0, &dwork[iwrk2], &upds);
                    SLC_DLACPY("Full", &upds, &sdim, &dwork[iwrk2], &upds,
                               &q[(iupd - 1) + (m + ib1 - 1) * ldq], &ldq);
                }

                hlp = dim2 - dim1;
                if (hlp == 1) {
                    iwork[j] = ib1 + 1;
                } else if (hlp == -1) {
                    iwork[j] = ib1 + 2;
                }

                hlp = iwork[is + j - 1];
                iwork[is + j - 1] = iwork[is + j];
                iwork[is + j] = hlp;
            }
            mm = mm + 1;
        }
        k = k + 1;
    }

#if MB03JD_DEBUG
    fprintf(stderr, "After STEP 1: mm=%d, mp=%d\n", mm, mp);
#endif

    k = r;
    while (k >= mm + 1) {
        if (iwork[is + k - 1] > 0) {
            for (j = k; j <= mp - 2; j++) {
                ib1 = iwork[j - 1];
                ib2 = iwork[j];
                ib3 = iwork[j + 1];
                dim1 = ib2 - ib1;
                dim2 = ib3 - ib2;
                sdim = dim1 + dim2;

                SLC_DLACPY("Upper", &sdim, &sdim, &a[(ib1 - 1) + (ib1 - 1) * lda], &lda,
                           &dwork[ia], &sdim);
                i32 sdim_m1 = sdim - 1;
                SLC_DLASET("Lower", &sdim_m1, &sdim_m1, &dbl0, &dbl0,
                           &dwork[ia + 1], &sdim);
                SLC_DLACPY("Upper", &sdim, &sdim, &b[(ib1 - 1) + (ib1 - 1) * ldb], &ldb,
                           &dwork[ib], &sdim);
                i32 ldb_p1 = ldb + 1;
                i32 sdim_p1 = sdim + 1;
                SLC_DCOPY(&sdim_m1, &b[ib1 + (ib1 - 1) * ldb], &ldb_p1,
                          &dwork[ib + 1], &sdim_p1);
                i32 sdim_m2 = sdim - 2;
                SLC_DLASET("Lower", &sdim_m2, &sdim_m2, &dbl0, &dbl0,
                           &dwork[ib + 2], &sdim);

                i32 iwrk1_size = ldwork - iwrk1;
                mb03dd("Triangular", &dim1, &dim2, prec, &dwork[ib], sdim,
                       &dwork[ia], sdim, &dwork[iq1], sdim,
                       &dwork[iq2], sdim, &dwork[iwrk1], iwrk1_size, info);
                if (*info > 0) {
                    *info = 1;
                    return;
                }

                nrows = ib1 - 1;
                ncols = m - ib3 + 1;
                ics = ib3;

                if (sdim > 2) {
                    SLC_DLACPY("Upper", &sdim, &sdim, &dwork[ia], &sdim,
                               &a[(ib1 - 1) + (ib1 - 1) * lda], &lda);
                    SLC_DLACPY("Upper", &sdim, &sdim, &dwork[ib], &sdim,
                               &b[(ib1 - 1) + (ib1 - 1) * ldb], &ldb);
                    SLC_DCOPY(&sdim_m1, &dwork[ib + 1], &sdim_p1,
                              &b[ib1 + (ib1 - 1) * ldb], &ldb_p1);
                    nrow = nrows;
                    ncol = ncols;
                    ic = ics;
                    ldw = (1 > nrow) ? 1 : nrow;
                } else {
                    tmp = a[ib1 + (ib1 - 1) * lda];
                    a[ib1 + (ib1 - 1) * lda] = ZERO;
                    nrow = ib3 - 1;
                    ncol = m - ib1 + 1;
                    ic = ib1;
                    ldw = nrow;
                }

                SLC_DGEMM("N", "N", &nrow, &sdim, &sdim, &dbl1,
                          &a[(ib1 - 1) * lda], &lda, &dwork[iq1], &sdim,
                          &dbl0, &dwork[iwrk2], &ldw);
                SLC_DLACPY("Full", &nrow, &sdim, &dwork[iwrk2], &ldw,
                           &a[(ib1 - 1) * lda], &lda);
                SLC_DGEMM("T", "N", &sdim, &ncol, &sdim, &dbl1,
                          &dwork[iq2], &sdim, &a[(ib1 - 1) + (ic - 1) * lda], &lda,
                          &dbl0, &dwork[iwrk2], &sdim);
                SLC_DLACPY("Full", &sdim, &ncol, &dwork[iwrk2], &sdim,
                           &a[(ib1 - 1) + (ic - 1) * lda], &lda);
                if (sdim == 2) {
                    a[ib1 + (ib1 - 1) * lda] = tmp;
                }

                SLC_DGEMM("N", "N", &nrows, &sdim, &sdim, &dbl1,
                          &d[(ib1 - 1) * ldd], &ldd, &dwork[iq2], &sdim,
                          &dbl0, &dwork[iwrk2], &ldw);
                SLC_DLACPY("Full", &nrows, &sdim, &dwork[iwrk2], &ldw,
                           &d[(ib1 - 1) * ldd], &ldd);
                SLC_DGEMM("T", "N", &sdim, &ncols, &sdim, &dbl1,
                          &dwork[iq2], &sdim, &d[(ib1 - 1) + (ics - 1) * ldd], &ldd,
                          &dbl0, &dwork[iwrk2], &sdim);
                SLC_DLACPY("Full", &sdim, &ncols, &dwork[iwrk2], &sdim,
                           &d[(ib1 - 1) + (ics - 1) * ldd], &ldd);
                i32 iwrk2_size = ldwork - iwrk2;
                mb01ld("Upper", "Transpose", sdim, sdim, ZERO, ONE,
                       &d[(ib1 - 1) + (ib1 - 1) * ldd], ldd, &dwork[iq2], sdim,
                       &d[(ib1 - 1) + (ib1 - 1) * ldd], ldd, &dwork[iwrk2],
                       iwrk2_size, info);

                SLC_DGEMM("N", "N", &nrow, &sdim, &sdim, &dbl1,
                          &b[(ib1 - 1) * ldb], &ldb, &dwork[iq1], &sdim,
                          &dbl0, &dwork[iwrk2], &ldw);
                SLC_DLACPY("Full", &nrow, &sdim, &dwork[iwrk2], &ldw,
                           &b[(ib1 - 1) * ldb], &ldb);
                SLC_DGEMM("T", "N", &sdim, &ncol, &sdim, &dbl1,
                          &dwork[iq2], &sdim, &b[(ib1 - 1) + (ic - 1) * ldb], &ldb,
                          &dbl0, &dwork[iwrk2], &sdim);
                SLC_DLACPY("Full", &sdim, &ncol, &dwork[iwrk2], &sdim,
                           &b[(ib1 - 1) + (ic - 1) * ldb], &ldb);

                SLC_DGEMM("N", "N", &nrows, &sdim, &sdim, &dbl1,
                          &f[(ib1 - 1) * ldf], &ldf, &dwork[iq2], &sdim,
                          &dbl0, &dwork[iwrk2], &ldw);
                SLC_DLACPY("Full", &nrows, &sdim, &dwork[iwrk2], &ldw,
                           &f[(ib1 - 1) * ldf], &ldf);
                SLC_DGEMM("T", "N", &sdim, &ncols, &sdim, &dbl1,
                          &dwork[iq2], &sdim, &f[(ib1 - 1) + (ics - 1) * ldf], &ldf,
                          &dbl0, &dwork[iwrk2], &sdim);
                SLC_DLACPY("Full", &sdim, &ncols, &dwork[iwrk2], &sdim,
                           &f[(ib1 - 1) + (ics - 1) * ldf], &ldf);
                mb01ru("Upper", "Transpose", sdim, sdim, ZERO, ONE,
                       &f[(ib1 - 1) + (ib1 - 1) * ldf], ldf, &dwork[iq2], sdim,
                       &f[(ib1 - 1) + (ib1 - 1) * ldf], ldf, &dwork[iwrk2],
                       iwrk2_size, info);
                i32 ldf_p1 = ldf + 1;
                SLC_DSCAL(&sdim, &HALF, &f[(ib1 - 1) + (ib1 - 1) * ldf], &ldf_p1);

                if (lcmpq) {
                    SLC_DGEMM("N", "N", &upds, &sdim, &sdim, &dbl1,
                              &q[(ib1 - 1) * ldq], &ldq, &dwork[iq1], &sdim,
                              &dbl0, &dwork[iwrk2], &upds);
                    SLC_DLACPY("Full", &upds, &sdim, &dwork[iwrk2], &upds,
                               &q[(ib1 - 1) * ldq], &ldq);
                    SLC_DGEMM("N", "N", &upds, &sdim, &sdim, &dbl1,
                              &q[(iupd - 1) + (m + ib1 - 1) * ldq], &ldq,
                              &dwork[iq2], &sdim, &dbl0, &dwork[iwrk2], &upds);
                    SLC_DLACPY("Full", &upds, &sdim, &dwork[iwrk2], &upds,
                               &q[(iupd - 1) + (m + ib1 - 1) * ldq], &ldq);
                }

                hlp = dim2 - dim1;
                if (hlp == 1) {
                    iwork[j] = ib1 + 1;
                } else if (hlp == -1) {
                    iwork[j] = ib1 + 2;
                }
            }
            mp = mp - 1;
        }
        k = k - 1;
    }

#if MB03JD_DEBUG
    fprintf(stderr, "After STEP 1.5: mm=%d, mp=%d\n", mm, mp);
    fprintf(stderr, "STEP 2 loop: k from %d down to %d\n", r, mp);
#endif

    iquple = 0;
    iauple = iquple + 16;
    ibuple = iauple + 8;
    iwrk5 = ibuple + 8;
    iwrk3 = iauple;
    iwrk4 = iwrk3 + 2 * n;
    itmp1 = iwrk3 + n;
    itmp2 = itmp1 + 4;
    itmp3 = itmp2 + 4;

    for (k = r; k >= mp; k--) {
        ir = iwork[r - 1];
        dim1 = iwork[r] - ir;
        sdim = 2 * dim1;

        if (dim1 == 2) {
            a[(m - 1) + (ir - 1) * lda] = ZERO;

            d[(ir - 1) + (ir - 1) * ldd] = ZERO;
            d[(m - 1) + (ir - 1) * ldd] = -d[(ir - 1) + (m - 1) * ldd];
            d[(m - 1) + (m - 1) * ldd] = ZERO;
            f[(m - 1) + (ir - 1) * ldf] = f[(ir - 1) + (m - 1) * ldf];
        }

        ibupri = ibuple + dim1 * dim1;
        iqlole = iquple + dim1;
        iqupri = iquple + dim1 * sdim;
        iqlori = iqupri + dim1;

        if (dim1 == 2) {
            SLC_DLACPY("Upper", &dim1, &dim1, &a[(ir - 1) + (ir - 1) * lda], &lda,
                       &dwork[iauple], &dim1);
            dwork[iauple + 6] = d[(ir - 1) + ir * ldd];
            SLC_DLACPY("Full", &dim1, &dim1, &b[(ir - 1) + (ir - 1) * ldb], &ldb,
                       &dwork[ibuple], &dim1);
            SLC_DLACPY("Upper", &dim1, &dim1, &f[(ir - 1) + (ir - 1) * ldf], &ldf,
                       &dwork[ibupri], &dim1);
        } else {
            dwork[ibuple] = b[(ir - 1) + (ir - 1) * ldb];
            dwork[ibupri] = f[(ir - 1) + (ir - 1) * ldf];
        }

        i32 info_hd = 0;
        mb03hd(sdim, &dwork[iauple], dim1, &dwork[ibuple], dim1, par,
               &dwork[iquple], sdim, &dwork[iwrk5], &info_hd);
        if (info_hd > 0) {
            *info = 2;
            return;
        }

        if (dim1 == 2) {
            SLC_DLACPY("Full", &m, &dim1, &a[(ir - 1) * lda], &lda,
                       &dwork[iwrk3], &m);
            SLC_DGEMM("N", "N", &m, &dim1, &dim1, &dbl1,
                      &dwork[iwrk3], &m, &dwork[iquple], &sdim,
                      &dbl0, &a[(ir - 1) * lda], &lda);
            SLC_DGEMM("N", "N", &m, &dim1, &dim1, &dbl1,
                      &d[(ir - 1) * ldd], &ldd, &dwork[iqlole], &sdim,
                      &dbl1, &a[(ir - 1) * lda], &lda);

            SLC_DGEMM("N", "N", &m, &dim1, &dim1, &dbl1,
                      &dwork[iwrk3], &m, &dwork[iqupri], &sdim,
                      &dbl0, &dwork[itmp1], &m);
            SLC_DGEMM("N", "N", &m, &dim1, &dim1, &dbl1,
                      &d[(ir - 1) * ldd], &ldd, &dwork[iqlori], &sdim,
                      &dbl1, &dwork[itmp1], &m);
            SLC_DLACPY("Full", &m, &dim1, &dwork[itmp1], &m, &d[(ir - 1) * ldd],
                       &ldd);

            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &dbl1,
                      &dwork[iwrk3 + m - dim1], &m, &dwork[iqlole], &sdim,
                      &dbl0, &dwork[itmp1], &dim1);
            SLC_DGEMV("T", &dim1, &dim1, &dbl1, &dwork[iwrk3 + m - dim1], &m,
                      &dwork[iqlori + sdim], &int1, &dbl0, &dwork[itmp2], &int1);

            SLC_DLACPY("Full", &dim1, &dim1, &a[(ir - 1) + (ir - 1) * lda], &lda,
                       &dwork[iwrk3], &dim1);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &neg1,
                      &dwork[iqupri], &sdim, &dwork[itmp1], &dim1,
                      &dbl0, &a[(ir - 1) + (ir - 1) * lda], &lda);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &dbl1,
                      &dwork[iqlori], &sdim, &dwork[iwrk3], &dim1,
                      &dbl1, &a[(ir - 1) + (ir - 1) * lda], &lda);

            f64 dot1 = SLC_DDOT(&dim1, &dwork[iqlori], &int1, &d[(ir - 1) + (m - 1) * ldd], &int1);
            f64 dot2 = SLC_DDOT(&dim1, &dwork[iqupri], &int1, &dwork[itmp2], &int1);
            d[(ir - 1) + (m - 1) * ldd] = dot1 - dot2;

            SLC_DLACPY("Full", &m, &dim1, &b[(ir - 1) * ldb], &ldb,
                       &dwork[iwrk3], &m);
            SLC_DGEMM("N", "N", &m, &dim1, &dim1, &dbl1,
                      &dwork[iwrk3], &m, &dwork[iquple], &sdim,
                      &dbl0, &b[(ir - 1) * ldb], &ldb);
            SLC_DGEMM("N", "N", &m, &dim1, &dim1, &dbl1,
                      &f[(ir - 1) * ldf], &ldf, &dwork[iqlole], &sdim,
                      &dbl1, &b[(ir - 1) * ldb], &ldb);

            SLC_DGEMM("N", "N", &m, &dim1, &dim1, &dbl1,
                      &dwork[iwrk3], &m, &dwork[iqupri], &sdim,
                      &dbl0, &dwork[itmp1], &m);
            SLC_DGEMM("N", "N", &m, &dim1, &dim1, &dbl1,
                      &f[(ir - 1) * ldf], &ldf, &dwork[iqlori], &sdim,
                      &dbl1, &dwork[itmp1], &m);
            SLC_DLACPY("Full", &m, &dim1, &dwork[itmp1], &m, &f[(ir - 1) * ldf],
                       &ldf);

            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &dbl1,
                      &dwork[iwrk3 + m - dim1], &m, &dwork[iqlole], &sdim,
                      &dbl0, &dwork[itmp1], &dim1);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &dbl1,
                      &dwork[iwrk3 + m - dim1], &m, &dwork[iqlori], &sdim,
                      &dbl0, &dwork[itmp2], &dim1);

            SLC_DLACPY("Full", &dim1, &dim1, &b[(ir - 1) + (ir - 1) * ldb], &ldb,
                       &dwork[itmp3], &dim1);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &dbl1,
                      &dwork[iqupri], &sdim, &dwork[itmp1], &dim1,
                      &dbl0, &b[(ir - 1) + (ir - 1) * ldb], &ldb);
            SLC_DGEMM("T", "N", &dim1, &dim1, &dim1, &dbl1,
                      &dwork[iqlori], &sdim, &dwork[itmp3], &dim1,
                      &dbl1, &b[(ir - 1) + (ir - 1) * ldb], &ldb);

            slicot_mb01rx('L', 'U', 'T', dim1, dim1, ZERO, ONE,
                          &dwork[itmp1], dim1, &dwork[iqlori], sdim,
                          &f[(ir - 1) + (ir - 1) * ldf], ldf);
            slicot_mb01rx('L', 'U', 'T', dim1, dim1, ONE, ONE,
                          &dwork[itmp1], dim1, &dwork[iqupri], sdim,
                          &dwork[itmp2], dim1);
            SLC_DLACPY("Upper", &dim1, &dim1, &dwork[itmp1], &dim1,
                       &f[(ir - 1) + (ir - 1) * ldf], &ldf);

            if (lcmpq) {
                SLC_DLACPY("Full", &n, &dim1, &q[(ir - 1) * ldq], &ldq,
                           &dwork[iwrk4], &n);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &dbl1,
                          &dwork[iwrk4], &n, &dwork[iquple], &sdim,
                          &dbl0, &q[(ir - 1) * ldq], &ldq);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &dbl1,
                          &q[(m + ir - 1) * ldq], &ldq, &dwork[iqlole], &sdim,
                          &dbl1, &q[(ir - 1) * ldq], &ldq);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &dbl1,
                          &dwork[iwrk4], &n, &dwork[iqupri], &sdim,
                          &dbl0, &dwork[iwrk3], &n);
                SLC_DGEMM("N", "N", &n, &dim1, &dim1, &dbl1,
                          &q[(m + ir - 1) * ldq], &ldq, &dwork[iqlori], &sdim,
                          &dbl1, &dwork[iwrk3], &n);
                SLC_DLACPY("Full", &n, &dim1, &dwork[iwrk3], &n,
                           &q[(m + ir - 1) * ldq], &ldq);
            }
        } else {
            q11 = dwork[iquple];
            q21 = dwork[iqlole];
            q12 = dwork[iqupri];
            q22 = dwork[iqlori];

            i32 m_m1 = m - 1;
            SLC_DCOPY(&m_m1, &a[(ir - 1) * lda], &int1, &dwork[iwrk3], &int1);
            SLC_DSCAL(&m_m1, &q11, &a[(ir - 1) * lda], &int1);
            SLC_DAXPY(&m_m1, &q21, &d[(ir - 1) * ldd], &int1, &a[(ir - 1) * lda], &int1);

            SLC_DSCAL(&m_m1, &q22, &d[(ir - 1) * ldd], &int1);
            SLC_DAXPY(&m_m1, &q12, &dwork[iwrk3], &int1, &d[(ir - 1) * ldd], &int1);

            SLC_DCOPY(&m_m1, &b[(ir - 1) * ldb], &int1, &dwork[iwrk3], &int1);
            SLC_DSCAL(&m_m1, &q11, &b[(ir - 1) * ldb], &int1);
            SLC_DAXPY(&m_m1, &q21, &f[(ir - 1) * ldf], &int1, &b[(ir - 1) * ldb], &int1);

            SLC_DSCAL(&m_m1, &q22, &f[(ir - 1) * ldf], &int1);
            SLC_DAXPY(&m_m1, &q12, &dwork[iwrk3], &int1, &f[(ir - 1) * ldf], &int1);

            b[(m - 1) + (m - 1) * ldb] = -b[(m - 1) + (m - 1) * ldb];

            if (lcmpq) {
                SLC_DCOPY(&n, &q[(ir - 1) * ldq], &int1, &dwork[iwrk4], &int1);
                SLC_DSCAL(&n, &q11, &q[(ir - 1) * ldq], &int1);
                SLC_DAXPY(&n, &q21, &q[(ir + m - 1) * ldq], &int1, &q[(ir - 1) * ldq], &int1);
                SLC_DSCAL(&n, &q22, &q[(ir + m - 1) * ldq], &int1);
                SLC_DAXPY(&n, &q12, &dwork[iwrk4], &int1, &q[(ir + m - 1) * ldq], &int1);
            }
        }

#if MB03JD_DEBUG
        fprintf(stderr, "STEP 2 iteration k=%d: mm before increment=%d, ir=%d, dim1=%d\n", k, mm, ir, dim1);
#endif
        mm = mm + 1;
#if MB03JD_DEBUG
        fprintf(stderr, "  After mm++: mm=%d, inner loop j from %d down to %d\n", mm, r-1, mm);
#endif
        for (j = r - 1; j >= mm; j--) {
            ib1 = iwork[j - 1];
            ib2 = iwork[j];
            ib3 = iwork[j + 1];
            dim1 = ib2 - ib1;
            dim2 = ib3 - ib2;
            sdim = dim1 + dim2;

            SLC_DLACPY("Upper", &sdim, &sdim, &a[(ib1 - 1) + (ib1 - 1) * lda], &lda,
                       &dwork[ia], &sdim);
            i32 sdim_m1 = sdim - 1;
            SLC_DLASET("Lower", &sdim_m1, &sdim_m1, &dbl0, &dbl0,
                       &dwork[ia + 1], &sdim);
            SLC_DLACPY("Upper", &sdim, &sdim, &b[(ib1 - 1) + (ib1 - 1) * ldb], &ldb,
                       &dwork[ib], &sdim);
            i32 ldb_p1 = ldb + 1;
            i32 sdim_p1 = sdim + 1;
            SLC_DCOPY(&sdim_m1, &b[ib1 + (ib1 - 1) * ldb], &ldb_p1,
                      &dwork[ib + 1], &sdim_p1);
            i32 sdim_m2 = sdim - 2;
            SLC_DLASET("Lower", &sdim_m2, &sdim_m2, &dbl0, &dbl0,
                       &dwork[ib + 2], &sdim);

            i32 iwrk1_size = ldwork - iwrk1;
            mb03dd("Triangular", &dim1, &dim2, prec, &dwork[ib], sdim,
                   &dwork[ia], sdim, &dwork[iq1], sdim,
                   &dwork[iq2], sdim, &dwork[iwrk1], iwrk1_size, info);
            if (*info > 0) {
                *info = 1;
                return;
            }

            nrows = ib1 - 1;
            ncols = m - ib3 + 1;
            ics = ib3;

            if (sdim > 2) {
                SLC_DLACPY("Upper", &sdim, &sdim, &dwork[ia], &sdim,
                           &a[(ib1 - 1) + (ib1 - 1) * lda], &lda);
                SLC_DLACPY("Upper", &sdim, &sdim, &dwork[ib], &sdim,
                           &b[(ib1 - 1) + (ib1 - 1) * ldb], &ldb);
                SLC_DCOPY(&sdim_m1, &dwork[ib + 1], &sdim_p1,
                          &b[ib1 + (ib1 - 1) * ldb], &ldb_p1);
                nrow = nrows;
                ncol = ncols;
                ic = ics;
                ldw = (1 > nrow) ? 1 : nrow;
            } else {
                tmp = a[ib1 + (ib1 - 1) * lda];
                a[ib1 + (ib1 - 1) * lda] = ZERO;
                nrow = ib3 - 1;
                ncol = m - ib1 + 1;
                ic = ib1;
                ldw = nrow;
            }

            SLC_DGEMM("N", "N", &nrow, &sdim, &sdim, &dbl1,
                      &a[(ib1 - 1) * lda], &lda, &dwork[iq1], &sdim,
                      &dbl0, &dwork[iwrk2], &ldw);
            SLC_DLACPY("Full", &nrow, &sdim, &dwork[iwrk2], &ldw,
                       &a[(ib1 - 1) * lda], &lda);
            SLC_DGEMM("T", "N", &sdim, &ncol, &sdim, &dbl1,
                      &dwork[iq2], &sdim, &a[(ib1 - 1) + (ic - 1) * lda], &lda,
                      &dbl0, &dwork[iwrk2], &sdim);
            SLC_DLACPY("Full", &sdim, &ncol, &dwork[iwrk2], &sdim,
                       &a[(ib1 - 1) + (ic - 1) * lda], &lda);
            if (sdim == 2) {
                a[ib1 + (ib1 - 1) * lda] = tmp;
            }

            SLC_DGEMM("N", "N", &nrows, &sdim, &sdim, &dbl1,
                      &d[(ib1 - 1) * ldd], &ldd, &dwork[iq2], &sdim,
                      &dbl0, &dwork[iwrk2], &ldw);
            SLC_DLACPY("Full", &nrows, &sdim, &dwork[iwrk2], &ldw,
                       &d[(ib1 - 1) * ldd], &ldd);
            SLC_DGEMM("T", "N", &sdim, &ncols, &sdim, &dbl1,
                      &dwork[iq2], &sdim, &d[(ib1 - 1) + (ics - 1) * ldd], &ldd,
                      &dbl0, &dwork[iwrk2], &sdim);
            SLC_DLACPY("Full", &sdim, &ncols, &dwork[iwrk2], &sdim,
                       &d[(ib1 - 1) + (ics - 1) * ldd], &ldd);
            i32 iwrk2_size = ldwork - iwrk2;
            mb01ld("Upper", "Transpose", sdim, sdim, ZERO, ONE,
                   &d[(ib1 - 1) + (ib1 - 1) * ldd], ldd, &dwork[iq2], sdim,
                   &d[(ib1 - 1) + (ib1 - 1) * ldd], ldd, &dwork[iwrk2],
                   iwrk2_size, info);

            SLC_DGEMM("N", "N", &nrow, &sdim, &sdim, &dbl1,
                      &b[(ib1 - 1) * ldb], &ldb, &dwork[iq1], &sdim,
                      &dbl0, &dwork[iwrk2], &ldw);
            SLC_DLACPY("Full", &nrow, &sdim, &dwork[iwrk2], &ldw,
                       &b[(ib1 - 1) * ldb], &ldb);
            SLC_DGEMM("T", "N", &sdim, &ncol, &sdim, &dbl1,
                      &dwork[iq2], &sdim, &b[(ib1 - 1) + (ic - 1) * ldb], &ldb,
                      &dbl0, &dwork[iwrk2], &sdim);
            SLC_DLACPY("Full", &sdim, &ncol, &dwork[iwrk2], &sdim,
                       &b[(ib1 - 1) + (ic - 1) * ldb], &ldb);

            SLC_DGEMM("N", "N", &nrows, &sdim, &sdim, &dbl1,
                      &f[(ib1 - 1) * ldf], &ldf, &dwork[iq2], &sdim,
                      &dbl0, &dwork[iwrk2], &ldw);
            SLC_DLACPY("Full", &nrows, &sdim, &dwork[iwrk2], &ldw,
                       &f[(ib1 - 1) * ldf], &ldf);
            SLC_DGEMM("T", "N", &sdim, &ncols, &sdim, &dbl1,
                      &dwork[iq2], &sdim, &f[(ib1 - 1) + (ics - 1) * ldf], &ldf,
                      &dbl0, &dwork[iwrk2], &sdim);
            SLC_DLACPY("Full", &sdim, &ncols, &dwork[iwrk2], &sdim,
                       &f[(ib1 - 1) + (ics - 1) * ldf], &ldf);
            mb01ru("Upper", "Transpose", sdim, sdim, ZERO, ONE,
                   &f[(ib1 - 1) + (ib1 - 1) * ldf], ldf, &dwork[iq2], sdim,
                   &f[(ib1 - 1) + (ib1 - 1) * ldf], ldf, &dwork[iwrk2],
                   iwrk2_size, info);
            i32 ldf_p1 = ldf + 1;
            SLC_DSCAL(&sdim, &HALF, &f[(ib1 - 1) + (ib1 - 1) * ldf], &ldf_p1);

            if (lcmpq) {
                SLC_DGEMM("N", "N", &n, &sdim, &sdim, &dbl1,
                          &q[(ib1 - 1) * ldq], &ldq, &dwork[iq1], &sdim,
                          &dbl0, &dwork[iwrk2], &n);
                SLC_DLACPY("Full", &n, &sdim, &dwork[iwrk2], &n,
                           &q[(ib1 - 1) * ldq], &ldq);
                SLC_DGEMM("N", "N", &n, &sdim, &sdim, &dbl1,
                          &q[(m + ib1 - 1) * ldq], &ldq, &dwork[iq2], &sdim,
                          &dbl0, &dwork[iwrk2], &n);
                SLC_DLACPY("Full", &n, &sdim, &dwork[iwrk2], &n,
                           &q[(m + ib1 - 1) * ldq], &ldq);
            }

            hlp = dim2 - dim1;
            if (hlp == 1) {
                iwork[j] = ib1 + 1;
            } else if (hlp == -1) {
                iwork[j] = ib1 + 2;
            }
        }
    }

    if (m > 1) {
        d[(m - 2) + (m - 2) * ldd] = d1;
        d[(m - 1) + (m - 2) * ldd] = d2;
        d[(m - 1) + (m - 1) * ldd] = d3;
        a[(m - 1) + (m - 2) * lda] = a2;
        f[(m - 1) + (m - 2) * ldf] = f2;
    }

    if (mm > 0) {
        *neig = iwork[mm] - 1;
    } else {
        *neig = 0;
    }

#if MB03JD_DEBUG
    fprintf(stderr, "Final: mm=%d, iwork[mm]=%d, neig=%d\n", mm, mm > 0 ? iwork[mm] : 0, *neig);
#endif
}
