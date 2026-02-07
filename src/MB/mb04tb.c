// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb04tb(const char *trana, const char *tranb, i32 n, i32 ilo,
            f64 *a, i32 lda, f64 *b, i32 ldb, f64 *g, i32 ldg,
            f64 *q, i32 ldq, f64 *csl, f64 *csr,
            f64 *taul, f64 *taur, f64 *dwork, i32 ldwork, i32 *info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool ltra = (trana[0] == 'T' || trana[0] == 't' ||
                 trana[0] == 'C' || trana[0] == 'c');
    bool ltrb = (tranb[0] == 'T' || tranb[0] == 't' ||
                 tranb[0] == 'C' || tranb[0] == 'c');

    *info = 0;
    i32 minwrk = (n > 1) ? n : 1;

    if (!ltra && !(trana[0] == 'N' || trana[0] == 'n')) {
        *info = -1;
    } else if (!ltrb && !(tranb[0] == 'N' || tranb[0] == 'n')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ilo < 1 || ilo > n + 1) {
        *info = -4;
    } else if (lda < minwrk) {
        *info = -6;
    } else if (ldb < minwrk) {
        *info = -8;
    } else if (ldg < minwrk) {
        *info = -10;
    } else if (ldq < minwrk) {
        *info = -12;
    }

    if (*info != 0) {
        return;
    }

    bool lquery = (ldwork == -1);

    if (ldwork < minwrk && !lquery) {
        dwork[0] = (f64)minwrk;
        *info = -18;
        return;
    }

    i32 nb = 1;
    i32 wrkopt;
    if (n == 0) {
        wrkopt = 1;
    } else {
        i32 qr_info;
        i32 neg1 = -1;
        f64 qr_work;
        SLC_DGEQRF(&n, &ilo, dwork, &minwrk, dwork, &qr_work, &neg1, &qr_info);
        wrkopt = (i32)qr_work;
        if (wrkopt < minwrk) wrkopt = minwrk;
        nb = wrkopt / ilo;
        if (nb < 1) nb = 1;
        i32 opt2 = 16 * n * nb + 5 * nb;
        if (opt2 > wrkopt) wrkopt = opt2;
    }

    if (lquery) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    for (i32 i = 0; i < ilo - 1; i++) {
        csl[2 * i] = ONE;
        csl[2 * i + 1] = ZERO;
        taul[i] = ZERO;
        if (i < n - 1) {
            csr[2 * i] = ONE;
            csr[2 * i + 1] = ZERO;
            taur[i] = ZERO;
        }
    }

    i32 nh = n - ilo + 1;
    if (nh == 0) {
        dwork[0] = ONE;
        return;
    }

    i32 nbmin = 2;
    i32 nx = nb;
    if (nb > 1 && nb < nh) {
        char opts[3] = { trana[0], tranb[0], '\0' };
        nx = ue01md(3, "MB04TB", opts, n, ilo, -1);
        if (nx < nb) nx = nb;
        if (nx < nh) {
            if (ldwork < wrkopt) {
                nbmin = ue01md(2, "MB04TB", opts, n, ilo, -1);
                if (nbmin < 2) nbmin = 2;
                nb = ldwork / (16 * n + 5);
                if (nb < 1) nb = 1;
            }
        }
    }

    i32 nnb = n * nb;
    i32 pyb = 0;
    i32 pyq = pyb + 2 * nnb;
    i32 pya = pyq + 2 * nnb;
    i32 pyg = pya + 2 * nnb;
    i32 pxq = pyg + 2 * nnb;
    i32 pxa = pxq + 2 * nnb;
    i32 pxg = pxa + 2 * nnb;
    i32 pxb = pxg + 2 * nnb;
    i32 pdw = pxb + 2 * nnb;

    i32 i_start = ilo;

    if (nb < nbmin || nb >= nh) {
        // Use unblocked code
        i_start = ilo;
    } else if (ltra && ltrb) {
        for (i32 i = ilo; i <= n - nx - 1; i += nb) {
            i32 ib = nb;
            if (ib > n - i) ib = n - i;
            i32 nib = n * ib;

            // 1-based Fortran: A(I,1) → C 0-based: a[(i-1) + 0*lda]
            mb03xu(ltra, ltrb, n - i + 1, i - 1, ib,
                   &a[(i - 1)], lda,
                   &b[(i - 1) * ldb], ldb,
                   g, ldg,
                   &q[(i - 1) + (i - 1) * ldq], ldq,
                   &dwork[pxa], n, &dwork[pxb], n, &dwork[pxg], n, &dwork[pxq], n,
                   &dwork[pya], n, &dwork[pyb], n, &dwork[pyg], n, &dwork[pyq], n,
                   &csl[2 * (i - 1)], &csr[2 * (i - 1)], &taul[i - 1], &taur[i - 1],
                   &dwork[pdw]);

            // Update A(i+ib+1:n, 1:n) — row indices i+ib..n-1 (0-based)
            i32 rows_a = n - i - ib;
            i32 cols_a = n - i - ib + 1;
            if (rows_a > 0 && cols_a > 0) {
                SLC_DGEMM("N", "T", &rows_a, &cols_a, &ib, &ONE,
                          &dwork[pxa + nb + 1], &n, &q[(i + ib - 1) + (i - 1) * ldq], &ldq,
                          &ONE, &a[(i + ib) + (i + ib - 1) * lda], &lda);
                SLC_DGEMM("N", "N", &rows_a, &cols_a, &ib, &ONE,
                          &dwork[pxa + nib + nb + 1], &n, &a[(i - 1) + (i + ib - 1) * lda], &lda,
                          &ONE, &a[(i + ib) + (i + ib - 1) * lda], &lda);
            }
            if (rows_a > 0) {
                SLC_DGEMM("T", "T", &rows_a, &n, &ib, &ONE,
                          &q[(i - 1) + (i + ib) * ldq], &ldq, &dwork[pya], &n,
                          &ONE, &a[(i + ib)], &lda);
                SLC_DGEMM("N", "T", &rows_a, &n, &ib, &ONE,
                          &b[(i + ib) + (i - 1) * ldb], &ldb, &dwork[pya + nib], &n,
                          &ONE, &a[(i + ib)], &lda);
            }

            // Update Q(i+ib:n, i+ib+1:n)
            i32 rows_q = n - i - ib + 1;
            i32 cols_q = n - i - ib;
            if (rows_q > 0 && cols_q > 0) {
                SLC_DGEMM("N", "T", &rows_q, &cols_q, &ib, &ONE,
                          &q[(i + ib - 1) + (i - 1) * ldq], &ldq, &dwork[pxq + nb + 1], &n,
                          &ONE, &q[(i + ib - 1) + (i + ib) * ldq], &ldq);
                SLC_DGEMM("T", "T", &rows_q, &cols_q, &ib, &ONE,
                          &a[(i - 1) + (i + ib - 1) * lda], &lda, &dwork[pxq + nib + nb + 1], &n,
                          &ONE, &q[(i + ib - 1) + (i + ib) * ldq], &ldq);
                SLC_DGEMM("N", "N", &rows_q, &cols_q, &ib, &ONE,
                          &dwork[pyq + nb], &n, &q[(i - 1) + (i + ib) * ldq], &ldq,
                          &ONE, &q[(i + ib - 1) + (i + ib) * ldq], &ldq);
                SLC_DGEMM("N", "T", &rows_q, &cols_q, &ib, &ONE,
                          &dwork[pyq + nib + nb], &n, &b[(i + ib) + (i - 1) * ldb], &ldb,
                          &ONE, &q[(i + ib - 1) + (i + ib) * ldq], &ldq);
            }

            // Update G
            if (rows_q > 0) {
                SLC_DGEMM("N", "T", &rows_q, &n, &ib, &ONE,
                          &q[(i + ib - 1) + (i - 1) * ldq], &ldq, &dwork[pxg], &n,
                          &ONE, &g[(i + ib - 1)], &ldg);
                SLC_DGEMM("T", "T", &rows_q, &n, &ib, &ONE,
                          &a[(i - 1) + (i + ib - 1) * lda], &lda, &dwork[pxg + nib], &n,
                          &ONE, &g[(i + ib - 1)], &ldg);
            }
            if (cols_q > 0) {
                SLC_DGEMM("N", "N", &n, &cols_q, &ib, &ONE,
                          &dwork[pyg], &n, &q[(i - 1) + (i + ib) * ldq], &ldq,
                          &ONE, &g[(i + ib) * ldg], &ldg);
                SLC_DGEMM("N", "T", &n, &cols_q, &ib, &ONE,
                          &dwork[pyg + nib], &n, &b[(i + ib) + (i - 1) * ldb], &ldb,
                          &ONE, &g[(i + ib) * ldg], &ldg);
            }

            // Update B(1:n, i+ib:n)
            if (rows_q > 0) {
                SLC_DGEMM("N", "T", &n, &rows_q, &ib, &ONE,
                          &dwork[pxb], &n, &q[(i + ib - 1) + (i - 1) * ldq], &ldq,
                          &ONE, &b[(i + ib - 1) * ldb], &ldb);
                SLC_DGEMM("N", "N", &n, &rows_q, &ib, &ONE,
                          &dwork[pxb + nib], &n, &a[(i - 1) + (i + ib - 1) * lda], &lda,
                          &ONE, &b[(i + ib - 1) * ldb], &ldb);
            }
            if (rows_a > 0 && rows_q > 0) {
                SLC_DGEMM("T", "T", &rows_a, &rows_q, &ib, &ONE,
                          &q[(i - 1) + (i + ib) * ldq], &ldq, &dwork[pyb + nb], &n,
                          &ONE, &b[(i + ib) + (i + ib - 1) * ldb], &ldb);
                SLC_DGEMM("N", "T", &rows_a, &rows_q, &ib, &ONE,
                          &b[(i + ib) + (i - 1) * ldb], &ldb, &dwork[pyb + nib + nb], &n,
                          &ONE, &b[(i + ib) + (i + ib - 1) * ldb], &ldb);
            }

            i_start = i + nb;
        }
    } else if (ltra) {
        for (i32 i = ilo; i <= n - nx - 1; i += nb) {
            i32 ib = nb;
            if (ib > n - i) ib = n - i;
            i32 nib = n * ib;

            mb03xu(ltra, ltrb, n - i + 1, i - 1, ib,
                   &a[(i - 1)], lda,
                   &b[(i - 1)], ldb,
                   g, ldg,
                   &q[(i - 1) + (i - 1) * ldq], ldq,
                   &dwork[pxa], n, &dwork[pxb], n, &dwork[pxg], n, &dwork[pxq], n,
                   &dwork[pya], n, &dwork[pyb], n, &dwork[pyg], n, &dwork[pyq], n,
                   &csl[2 * (i - 1)], &csr[2 * (i - 1)], &taul[i - 1], &taur[i - 1],
                   &dwork[pdw]);

            i32 rows_a = n - i - ib;
            i32 cols_a = n - i - ib + 1;
            if (rows_a > 0 && cols_a > 0) {
                SLC_DGEMM("N", "T", &rows_a, &cols_a, &ib, &ONE,
                          &dwork[pxa + nb + 1], &n, &q[(i + ib - 1) + (i - 1) * ldq], &ldq,
                          &ONE, &a[(i + ib) + (i + ib - 1) * lda], &lda);
                SLC_DGEMM("N", "N", &rows_a, &cols_a, &ib, &ONE,
                          &dwork[pxa + nib + nb + 1], &n, &a[(i - 1) + (i + ib - 1) * lda], &lda,
                          &ONE, &a[(i + ib) + (i + ib - 1) * lda], &lda);
            }
            if (rows_a > 0) {
                SLC_DGEMM("T", "T", &rows_a, &n, &ib, &ONE,
                          &q[(i - 1) + (i + ib) * ldq], &ldq, &dwork[pya], &n,
                          &ONE, &a[(i + ib)], &lda);
                SLC_DGEMM("T", "T", &rows_a, &n, &ib, &ONE,
                          &b[(i - 1) + (i + ib) * ldb], &ldb, &dwork[pya + nib], &n,
                          &ONE, &a[(i + ib)], &lda);
            }

            i32 rows_q = n - i - ib + 1;
            i32 cols_q = n - i - ib;
            if (rows_q > 0 && cols_q > 0) {
                SLC_DGEMM("N", "T", &rows_q, &cols_q, &ib, &ONE,
                          &q[(i + ib - 1) + (i - 1) * ldq], &ldq, &dwork[pxq + nb + 1], &n,
                          &ONE, &q[(i + ib - 1) + (i + ib) * ldq], &ldq);
                SLC_DGEMM("T", "T", &rows_q, &cols_q, &ib, &ONE,
                          &a[(i - 1) + (i + ib - 1) * lda], &lda, &dwork[pxq + nib + nb + 1], &n,
                          &ONE, &q[(i + ib - 1) + (i + ib) * ldq], &ldq);
                SLC_DGEMM("N", "N", &rows_q, &cols_q, &ib, &ONE,
                          &dwork[pyq + nb], &n, &q[(i - 1) + (i + ib) * ldq], &ldq,
                          &ONE, &q[(i + ib - 1) + (i + ib) * ldq], &ldq);
                SLC_DGEMM("N", "N", &rows_q, &cols_q, &ib, &ONE,
                          &dwork[pyq + nib + nb], &n, &b[(i - 1) + (i + ib) * ldb], &ldb,
                          &ONE, &q[(i + ib - 1) + (i + ib) * ldq], &ldq);
            }

            if (rows_q > 0) {
                SLC_DGEMM("N", "T", &rows_q, &n, &ib, &ONE,
                          &q[(i + ib - 1) + (i - 1) * ldq], &ldq, &dwork[pxg], &n,
                          &ONE, &g[(i + ib - 1)], &ldg);
                SLC_DGEMM("T", "T", &rows_q, &n, &ib, &ONE,
                          &a[(i - 1) + (i + ib - 1) * lda], &lda, &dwork[pxg + nib], &n,
                          &ONE, &g[(i + ib - 1)], &ldg);
            }
            if (cols_q > 0) {
                SLC_DGEMM("N", "N", &n, &cols_q, &ib, &ONE,
                          &dwork[pyg], &n, &q[(i - 1) + (i + ib) * ldq], &ldq,
                          &ONE, &g[(i + ib) * ldg], &ldg);
                SLC_DGEMM("N", "N", &n, &cols_q, &ib, &ONE,
                          &dwork[pyg + nib], &n, &b[(i - 1) + (i + ib) * ldb], &ldb,
                          &ONE, &g[(i + ib) * ldg], &ldg);
            }

            if (rows_q > 0) {
                SLC_DGEMM("N", "T", &rows_q, &n, &ib, &ONE,
                          &q[(i + ib - 1) + (i - 1) * ldq], &ldq, &dwork[pxb], &n,
                          &ONE, &b[(i + ib - 1)], &ldb);
                SLC_DGEMM("T", "T", &rows_q, &n, &ib, &ONE,
                          &a[(i - 1) + (i + ib - 1) * lda], &lda, &dwork[pxb + nib], &n,
                          &ONE, &b[(i + ib - 1)], &ldb);
            }
            if (rows_q > 0 && cols_q > 0) {
                SLC_DGEMM("N", "N", &rows_q, &cols_q, &ib, &ONE,
                          &dwork[pyb + nb], &n, &q[(i - 1) + (i + ib) * ldq], &ldq,
                          &ONE, &b[(i + ib - 1) + (i + ib) * ldb], &ldb);
                SLC_DGEMM("N", "N", &rows_q, &cols_q, &ib, &ONE,
                          &dwork[pyb + nib + nb], &n, &b[(i - 1) + (i + ib) * ldb], &ldb,
                          &ONE, &b[(i + ib - 1) + (i + ib) * ldb], &ldb);
            }

            i_start = i + nb;
        }
    } else if (ltrb) {
        for (i32 i = ilo; i <= n - nx - 1; i += nb) {
            i32 ib = nb;
            if (ib > n - i) ib = n - i;
            i32 nib = n * ib;

            mb03xu(ltra, ltrb, n - i + 1, i - 1, ib,
                   &a[(i - 1) * lda], lda,
                   &b[(i - 1) * ldb], ldb,
                   g, ldg,
                   &q[(i - 1) + (i - 1) * ldq], ldq,
                   &dwork[pxa], n, &dwork[pxb], n, &dwork[pxg], n, &dwork[pxq], n,
                   &dwork[pya], n, &dwork[pyb], n, &dwork[pyg], n, &dwork[pyq], n,
                   &csl[2 * (i - 1)], &csr[2 * (i - 1)], &taul[i - 1], &taur[i - 1],
                   &dwork[pdw]);

            i32 rows_q = n - i - ib + 1;
            i32 cols_q = n - i - ib;
            // Update A(1:n, i+ib+1:n)
            if (rows_q > 0 && cols_q > 0) {
                SLC_DGEMM("N", "T", &rows_q, &cols_q, &ib, &ONE,
                          &q[(i + ib - 1) + (i - 1) * ldq], &ldq, &dwork[pxa + nb + 1], &n,
                          &ONE, &a[(i + ib - 1) + (i + ib) * lda], &lda);
                SLC_DGEMM("N", "T", &rows_q, &cols_q, &ib, &ONE,
                          &a[(i + ib - 1) + (i - 1) * lda], &lda, &dwork[pxa + nib + nb + 1], &n,
                          &ONE, &a[(i + ib - 1) + (i + ib) * lda], &lda);
            }
            if (cols_q > 0) {
                SLC_DGEMM("N", "N", &n, &cols_q, &ib, &ONE,
                          &dwork[pya], &n, &q[(i - 1) + (i + ib) * ldq], &ldq,
                          &ONE, &a[(i + ib) * lda], &lda);
                SLC_DGEMM("N", "T", &n, &cols_q, &ib, &ONE,
                          &dwork[pya + nib], &n, &b[(i + ib) + (i - 1) * ldb], &ldb,
                          &ONE, &a[(i + ib) * lda], &lda);
            }

            // Update Q(i+ib:n, i+ib+1:n)
            if (rows_q > 0 && cols_q > 0) {
                SLC_DGEMM("N", "T", &rows_q, &cols_q, &ib, &ONE,
                          &q[(i + ib - 1) + (i - 1) * ldq], &ldq, &dwork[pxq + nb + 1], &n,
                          &ONE, &q[(i + ib - 1) + (i + ib) * ldq], &ldq);
                SLC_DGEMM("N", "T", &rows_q, &cols_q, &ib, &ONE,
                          &a[(i + ib - 1) + (i - 1) * lda], &lda, &dwork[pxq + nib + nb + 1], &n,
                          &ONE, &q[(i + ib - 1) + (i + ib) * ldq], &ldq);
                SLC_DGEMM("N", "N", &rows_q, &cols_q, &ib, &ONE,
                          &dwork[pyq + nb], &n, &q[(i - 1) + (i + ib) * ldq], &ldq,
                          &ONE, &q[(i + ib - 1) + (i + ib) * ldq], &ldq);
                SLC_DGEMM("N", "T", &rows_q, &cols_q, &ib, &ONE,
                          &dwork[pyq + nib + nb], &n, &b[(i + ib) + (i - 1) * ldb], &ldb,
                          &ONE, &q[(i + ib - 1) + (i + ib) * ldq], &ldq);
            }

            // Update G
            if (rows_q > 0) {
                SLC_DGEMM("N", "T", &rows_q, &n, &ib, &ONE,
                          &q[(i + ib - 1) + (i - 1) * ldq], &ldq, &dwork[pxg], &n,
                          &ONE, &g[(i + ib - 1)], &ldg);
                SLC_DGEMM("N", "T", &rows_q, &n, &ib, &ONE,
                          &a[(i + ib - 1) + (i - 1) * lda], &lda, &dwork[pxg + nib], &n,
                          &ONE, &g[(i + ib - 1)], &ldg);
            }
            if (cols_q > 0) {
                SLC_DGEMM("N", "N", &n, &cols_q, &ib, &ONE,
                          &dwork[pyg], &n, &q[(i - 1) + (i + ib) * ldq], &ldq,
                          &ONE, &g[(i + ib) * ldg], &ldg);
                SLC_DGEMM("N", "T", &n, &cols_q, &ib, &ONE,
                          &dwork[pyg + nib], &n, &b[(i + ib) + (i - 1) * ldb], &ldb,
                          &ONE, &g[(i + ib) * ldg], &ldg);
            }

            // Update B(1:n, i+ib:n)
            if (rows_q > 0) {
                SLC_DGEMM("N", "T", &n, &rows_q, &ib, &ONE,
                          &dwork[pxb], &n, &q[(i + ib - 1) + (i - 1) * ldq], &ldq,
                          &ONE, &b[(i + ib - 1) * ldb], &ldb);
                SLC_DGEMM("N", "T", &n, &rows_q, &ib, &ONE,
                          &dwork[pxb + nib], &n, &a[(i + ib - 1) + (i - 1) * lda], &lda,
                          &ONE, &b[(i + ib - 1) * ldb], &ldb);
            }
            i32 rows_a = n - i - ib;
            if (rows_a > 0 && rows_q > 0) {
                SLC_DGEMM("T", "T", &rows_a, &rows_q, &ib, &ONE,
                          &q[(i - 1) + (i + ib) * ldq], &ldq, &dwork[pyb + nb], &n,
                          &ONE, &b[(i + ib) + (i + ib - 1) * ldb], &ldb);
                SLC_DGEMM("N", "T", &rows_a, &rows_q, &ib, &ONE,
                          &b[(i + ib) + (i - 1) * ldb], &ldb, &dwork[pyb + nib + nb], &n,
                          &ONE, &b[(i + ib) + (i + ib - 1) * ldb], &ldb);
            }

            i_start = i + nb;
        }
    } else {
        // !ltra && !ltrb
        for (i32 i = ilo; i <= n - nx - 1; i += nb) {
            i32 ib = nb;
            if (ib > n - i) ib = n - i;
            i32 nib = n * ib;

            mb03xu(ltra, ltrb, n - i + 1, i - 1, ib,
                   &a[(i - 1) * lda], lda,
                   &b[(i - 1)], ldb,
                   g, ldg,
                   &q[(i - 1) + (i - 1) * ldq], ldq,
                   &dwork[pxa], n, &dwork[pxb], n, &dwork[pxg], n, &dwork[pxq], n,
                   &dwork[pya], n, &dwork[pyb], n, &dwork[pyg], n, &dwork[pyq], n,
                   &csl[2 * (i - 1)], &csr[2 * (i - 1)], &taul[i - 1], &taur[i - 1],
                   &dwork[pdw]);

            i32 rows_q = n - i - ib + 1;
            i32 cols_q = n - i - ib;
            // Update A(1:n, i+ib+1:n)
            if (rows_q > 0 && cols_q > 0) {
                SLC_DGEMM("N", "T", &rows_q, &cols_q, &ib, &ONE,
                          &q[(i + ib - 1) + (i - 1) * ldq], &ldq, &dwork[pxa + nb + 1], &n,
                          &ONE, &a[(i + ib - 1) + (i + ib) * lda], &lda);
                SLC_DGEMM("N", "T", &rows_q, &cols_q, &ib, &ONE,
                          &a[(i + ib - 1) + (i - 1) * lda], &lda, &dwork[pxa + nib + nb + 1], &n,
                          &ONE, &a[(i + ib - 1) + (i + ib) * lda], &lda);
            }
            if (cols_q > 0) {
                SLC_DGEMM("N", "N", &n, &cols_q, &ib, &ONE,
                          &dwork[pya], &n, &q[(i - 1) + (i + ib) * ldq], &ldq,
                          &ONE, &a[(i + ib) * lda], &lda);
                SLC_DGEMM("N", "N", &n, &cols_q, &ib, &ONE,
                          &dwork[pya + nib], &n, &b[(i - 1) + (i + ib) * ldb], &ldb,
                          &ONE, &a[(i + ib) * lda], &lda);
            }

            // Update Q(i+ib:n, i+ib+1:n)
            if (rows_q > 0 && cols_q > 0) {
                SLC_DGEMM("N", "T", &rows_q, &cols_q, &ib, &ONE,
                          &q[(i + ib - 1) + (i - 1) * ldq], &ldq, &dwork[pxq + nb + 1], &n,
                          &ONE, &q[(i + ib - 1) + (i + ib) * ldq], &ldq);
                SLC_DGEMM("N", "T", &rows_q, &cols_q, &ib, &ONE,
                          &a[(i + ib - 1) + (i - 1) * lda], &lda, &dwork[pxq + nib + nb + 1], &n,
                          &ONE, &q[(i + ib - 1) + (i + ib) * ldq], &ldq);
                SLC_DGEMM("N", "N", &rows_q, &cols_q, &ib, &ONE,
                          &dwork[pyq + nb], &n, &q[(i - 1) + (i + ib) * ldq], &ldq,
                          &ONE, &q[(i + ib - 1) + (i + ib) * ldq], &ldq);
                SLC_DGEMM("N", "N", &rows_q, &cols_q, &ib, &ONE,
                          &dwork[pyq + nib + nb], &n, &b[(i - 1) + (i + ib) * ldb], &ldb,
                          &ONE, &q[(i + ib - 1) + (i + ib) * ldq], &ldq);
            }

            // Update G
            if (rows_q > 0) {
                SLC_DGEMM("N", "T", &rows_q, &n, &ib, &ONE,
                          &q[(i + ib - 1) + (i - 1) * ldq], &ldq, &dwork[pxg], &n,
                          &ONE, &g[(i + ib - 1)], &ldg);
                SLC_DGEMM("N", "T", &rows_q, &n, &ib, &ONE,
                          &a[(i + ib - 1) + (i - 1) * lda], &lda, &dwork[pxg + nib], &n,
                          &ONE, &g[(i + ib - 1)], &ldg);
            }
            if (cols_q > 0) {
                SLC_DGEMM("N", "N", &n, &cols_q, &ib, &ONE,
                          &dwork[pyg], &n, &q[(i - 1) + (i + ib) * ldq], &ldq,
                          &ONE, &g[(i + ib) * ldg], &ldg);
                SLC_DGEMM("N", "N", &n, &cols_q, &ib, &ONE,
                          &dwork[pyg + nib], &n, &b[(i - 1) + (i + ib) * ldb], &ldb,
                          &ONE, &g[(i + ib) * ldg], &ldg);
            }

            // Update B(i+ib:n, 1:n)
            if (rows_q > 0) {
                SLC_DGEMM("N", "T", &rows_q, &n, &ib, &ONE,
                          &q[(i + ib - 1) + (i - 1) * ldq], &ldq, &dwork[pxb], &n,
                          &ONE, &b[(i + ib - 1)], &ldb);
                SLC_DGEMM("N", "T", &rows_q, &n, &ib, &ONE,
                          &a[(i + ib - 1) + (i - 1) * lda], &lda, &dwork[pxb + nib], &n,
                          &ONE, &b[(i + ib - 1)], &ldb);
            }
            if (rows_q > 0 && cols_q > 0) {
                SLC_DGEMM("N", "N", &rows_q, &cols_q, &ib, &ONE,
                          &dwork[pyb + nb], &n, &q[(i - 1) + (i + ib) * ldq], &ldq,
                          &ONE, &b[(i + ib - 1) + (i + ib) * ldb], &ldb);
                SLC_DGEMM("N", "N", &rows_q, &cols_q, &ib, &ONE,
                          &dwork[pyb + nib + nb], &n, &b[(i - 1) + (i + ib) * ldb], &ldb,
                          &ONE, &b[(i + ib - 1) + (i + ib) * ldb], &ldb);
            }

            i_start = i + nb;
        }
    }

    // Use unblocked code for the rest
    i32 ierr;
    mb04ts(trana, tranb, n, i_start, a, lda, b, ldb, g, ldg, q, ldq,
           csl, csr, taul, taur, dwork, ldwork, &ierr);

    dwork[0] = (f64)wrkopt;
}
