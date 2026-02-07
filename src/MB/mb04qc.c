/*
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>

void mb04qc(
    const char *strab, const char *trana, const char *tranb,
    const char *tranq, const char *direct, const char *storev,
    const char *storew, i32 m, i32 n, i32 k,
    const f64 *v, i32 ldv, const f64 *w, i32 ldw,
    const f64 *rs, i32 ldrs, const f64 *t, i32 ldt,
    f64 *a, i32 lda, f64 *b, i32 ldb,
    f64 *dwork)
{
    i32 i;
    i32 int1 = 1;
    f64 dbl0 = 0.0, dbl1 = 1.0;
    bool la1b1, lcolv, lcolw, ltra, ltrb, ltrq;
    f64 fact;
    i32 nk, nmk, km1;
    i32 pr1, pr2, pr3, ps1, ps2, ps3;
    i32 pt11, pt12, pt13, pt21, pt22, pt23, pt31, pt32, pt33;
    i32 pdw1, pdw2, pdw3, pdw4, pdw5, pdw6, pdw7, pdw8, pdw9;

    if (m <= 0 || n <= 0) {
        return;
    }

    la1b1 = (strab[0] == 'N' || strab[0] == 'n');
    lcolv = (storev[0] == 'C' || storev[0] == 'c');
    lcolw = (storew[0] == 'C' || storew[0] == 'c');
    ltra = (trana[0] == 'T' || trana[0] == 't' || trana[0] == 'C' || trana[0] == 'c');
    ltrb = (tranb[0] == 'T' || tranb[0] == 't' || tranb[0] == 'C' || tranb[0] == 'c');
    ltrq = (tranq[0] == 'T' || tranq[0] == 't' || tranq[0] == 'C' || tranq[0] == 'c');

    nk = n * k;
    nmk = n * (k - 1);
    km1 = k - 1;

    pr1 = 0;
    pr2 = pr1 + k;
    pr3 = pr2 + k;
    ps1 = pr3 + k;
    ps2 = ps1 + k;
    ps3 = ps2 + k;

    pt11 = 0;
    pt12 = pt11 + k;
    pt13 = pt12 + k;
    pt21 = pt13 + k;
    pt22 = pt21 + k;
    pt23 = pt22 + k;
    pt31 = pt23 + k;
    pt32 = pt31 + k;
    pt33 = pt32 + k;

    pdw1 = 0;
    pdw2 = pdw1 + nk;
    pdw3 = pdw2 + nk;
    pdw4 = pdw3 + nk;
    pdw5 = pdw4 + nk;
    pdw6 = pdw5 + nk;
    pdw7 = pdw6 + nk;
    pdw8 = pdw7 + nk;
    pdw9 = pdw8 + nk;

    if (la1b1) {
        if (ltra) {
            for (i = 0; i < k; i++) {
                SLC_DCOPY(&n, &a[0 + i * lda], &int1, &dwork[pdw7 + i * n], &int1);
            }
        } else {
            for (i = 0; i < n; i++) {
                SLC_DCOPY(&k, &a[0 + i * lda], &int1, &dwork[pdw7 + i], &n);
            }
        }

        SLC_DCOPY(&nk, &dwork[pdw7], &int1, &dwork[pdw1], &int1);
        if (lcolw) {
            SLC_DTRMM("R", "L", "N", "U", &n, &k, &dbl1, w, &ldw, &dwork[pdw1], &n);
        } else {
            SLC_DTRMM("R", "U", "T", "U", &n, &k, &dbl1, w, &ldw, &dwork[pdw1], &n);
        }

        SLC_DCOPY(&nk, &dwork[pdw7], &int1, &dwork[pdw2], &int1);
        if (lcolv) {
            SLC_DTRMM("R", "L", "N", "U", &n, &k, &dbl1, v, &ldv, &dwork[pdw2], &n);
        } else {
            SLC_DTRMM("R", "U", "T", "U", &n, &k, &dbl1, v, &ldv, &dwork[pdw2], &n);
        }
        fact = dbl1;
    } else {
        fact = dbl0;
    }

    i32 mk = m - k;
    if (m > k) {
        if (ltra && lcolw) {
            SLC_DGEMM("N", "N", &n, &k, &mk, &dbl1, &a[0 + k * lda], &lda,
                      &w[k + 0 * ldw], &ldw, &fact, &dwork[pdw1], &n);
        } else if (ltra) {
            SLC_DGEMM("N", "T", &n, &k, &mk, &dbl1, &a[0 + k * lda], &lda,
                      &w[0 + k * ldw], &ldw, &fact, &dwork[pdw1], &n);
        } else if (lcolw) {
            SLC_DGEMM("T", "N", &n, &k, &mk, &dbl1, &a[k + 0 * lda], &lda,
                      &w[k + 0 * ldw], &ldw, &fact, &dwork[pdw1], &n);
        } else {
            SLC_DGEMM("T", "T", &n, &k, &mk, &dbl1, &a[k + 0 * lda], &lda,
                      &w[0 + k * ldw], &ldw, &fact, &dwork[pdw1], &n);
        }
    } else if (!la1b1) {
        SLC_DLASET("A", &n, &k, &dbl0, &dbl0, &dwork[pdw1], &n);
    }

    if (m > k) {
        if (ltra && lcolv) {
            SLC_DGEMM("N", "N", &n, &k, &mk, &dbl1, &a[0 + k * lda], &lda,
                      &v[k + 0 * ldv], &ldv, &fact, &dwork[pdw2], &n);
        } else if (ltra) {
            SLC_DGEMM("N", "T", &n, &k, &mk, &dbl1, &a[0 + k * lda], &lda,
                      &v[0 + k * ldv], &ldv, &fact, &dwork[pdw2], &n);
        } else if (lcolv) {
            SLC_DGEMM("T", "N", &n, &k, &mk, &dbl1, &a[k + 0 * lda], &lda,
                      &v[k + 0 * ldv], &ldv, &fact, &dwork[pdw2], &n);
        } else {
            SLC_DGEMM("T", "T", &n, &k, &mk, &dbl1, &a[k + 0 * lda], &lda,
                      &v[0 + k * ldv], &ldv, &fact, &dwork[pdw2], &n);
        }
    } else if (!la1b1) {
        SLC_DLASET("A", &n, &k, &dbl0, &dbl0, &dwork[pdw2], &n);
    }

    if (ltrq) {
        SLC_DCOPY(&nk, &dwork[pdw1], &int1, &dwork[pdw3], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt11) * ldt], &ldt, &dwork[pdw3], &n);

        SLC_DCOPY(&nmk, &dwork[pdw2], &int1, &dwork[pdw4], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &km1, &dbl1, &t[0 + (pt31 + 1) * ldt], &ldt, &dwork[pdw4], &n);

        SLC_DAXPY(&nmk, &dbl1, &dwork[pdw4], &int1, &dwork[pdw3 + n], &int1);

        if (la1b1) {
            SLC_DCOPY(&nmk, &dwork[pdw7], &int1, &dwork[pdw8], &int1);
            SLC_DTRMM("R", "U", "N", "N", &n, &km1, &dbl1, &t[0 + (pt21 + 1) * ldt], &ldt, &dwork[pdw8], &n);
            SLC_DAXPY(&nmk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw3 + n], &int1);
        }

        SLC_DCOPY(&nk, &dwork[pdw1], &int1, &dwork[pdw4], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt12) * ldt], &ldt, &dwork[pdw4], &n);

        SLC_DCOPY(&nmk, &dwork[pdw2], &int1, &dwork[pdw5], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &km1, &dbl1, &t[0 + (pt32 + 1) * ldt], &ldt, &dwork[pdw5], &n);

        SLC_DAXPY(&nmk, &dbl1, &dwork[pdw5], &int1, &dwork[pdw4 + n], &int1);

        if (la1b1) {
            SLC_DCOPY(&nk, &dwork[pdw7], &int1, &dwork[pdw8], &int1);
            SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt22) * ldt], &ldt, &dwork[pdw8], &n);
            SLC_DAXPY(&nk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw4], &int1);
        }

        SLC_DCOPY(&nk, &dwork[pdw2], &int1, &dwork[pdw5], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt33) * ldt], &ldt, &dwork[pdw5], &n);

        SLC_DCOPY(&nk, &dwork[pdw1], &int1, &dwork[pdw6], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt13) * ldt], &ldt, &dwork[pdw6], &n);

        SLC_DAXPY(&nk, &dbl1, &dwork[pdw6], &int1, &dwork[pdw5], &int1);

        if (la1b1) {
            SLC_DCOPY(&nk, &dwork[pdw7], &int1, &dwork[pdw8], &int1);
            SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt23) * ldt], &ldt, &dwork[pdw8], &n);
            SLC_DAXPY(&nk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw5], &int1);
        }

        SLC_DCOPY(&nk, &dwork[pdw4], &int1, &dwork[pdw6], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &rs[(ps2) * ldrs], &ldrs, &dwork[pdw6], &n);

        SLC_DCOPY(&nmk, &dwork[pdw5], &int1, &dwork[pdw8], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &km1, &dbl1, &rs[0 + (ps1 + 1) * ldrs], &ldrs, &dwork[pdw8], &n);

        SLC_DAXPY(&nmk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw6 + n], &int1);

        SLC_DCOPY(&nk, &dwork[pdw4], &int1, &dwork[pdw8], &int1);
        SLC_DTRMM("R", "U", "N", "U", &n, &k, &dbl1, &rs[(pr2) * ldrs], &ldrs, &dwork[pdw8], &n);

        SLC_DAXPY(&nk, &dbl1, &dwork[pdw3], &int1, &dwork[pdw8], &int1);
        SLC_DAXPY(&nk, &dbl1, &dwork[pdw6], &int1, &dwork[pdw3], &int1);
        SLC_DCOPY(&nk, &dwork[pdw8], &int1, &dwork[pdw6], &int1);

        if (lcolv) {
            SLC_DTRMM("R", "L", "T", "U", &n, &k, &dbl1, v, &ldv, &dwork[pdw3], &n);
        } else {
            SLC_DTRMM("R", "U", "N", "U", &n, &k, &dbl1, v, &ldv, &dwork[pdw3], &n);
        }

        SLC_DCOPY(&nk, &dwork[pdw5], &int1, &dwork[pdw8], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &rs[(ps3) * ldrs], &ldrs, &dwork[pdw8], &n);

        SLC_DAXPY(&nk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw6], &int1);

        SLC_DCOPY(&nmk, &dwork[pdw4 + n], &int1, &dwork[pdw8], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &km1, &dbl1, &rs[0 + (pr3 + 1) * ldrs], &ldrs, &dwork[pdw8], &n);

        SLC_DAXPY(&nmk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw6 + n], &int1);

        SLC_DCOPY(&nk, &dwork[pdw6], &int1, &dwork[pdw8], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &rs[(pr1) * ldrs], &ldrs, &dwork[pdw8], &n);

        SLC_DAXPY(&nk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw6], &int1);

        if (lcolw) {
            SLC_DTRMM("R", "L", "T", "U", &n, &k, &dbl1, w, &ldw, &dwork[pdw6], &n);
        } else {
            SLC_DTRMM("R", "U", "N", "U", &n, &k, &dbl1, w, &ldw, &dwork[pdw6], &n);
        }

        if (ltra) {
            for (i = 0; i < k; i++) {
                SLC_DAXPY(&n, &dbl1, &dwork[pdw3 + i * n], &int1, &a[0 + i * lda], &int1);
                SLC_DAXPY(&n, &dbl1, &dwork[pdw6 + i * n], &int1, &a[0 + i * lda], &int1);
            }
        } else {
            for (i = 0; i < n; i++) {
                SLC_DAXPY(&k, &dbl1, &dwork[pdw3 + i], &n, &a[0 + i * lda], &int1);
                SLC_DAXPY(&k, &dbl1, &dwork[pdw6 + i], &n, &a[0 + i * lda], &int1);
            }
        }

        if (m > k) {
            if (lcolv) {
                SLC_DGEMM("N", "T", &mk, &n, &k, &dbl1, &v[k + 0 * ldv], &ldv,
                          &dwork[pdw3], &n, &dbl1, &a[ltra ? 0 + k * lda : k + 0 * lda], &lda);
            } else {
                SLC_DGEMM("T", "T", &mk, &n, &k, &dbl1, &v[0 + k * ldv], &ldv,
                          &dwork[pdw3], &n, &dbl1, &a[ltra ? 0 + k * lda : k + 0 * lda], &lda);
            }

            if (lcolw) {
                SLC_DGEMM("N", "T", &mk, &n, &k, &dbl1, &w[k + 0 * ldw], &ldw,
                          &dwork[pdw6], &n, &dbl1, &a[ltra ? 0 + k * lda : k + 0 * lda], &lda);
            } else {
                SLC_DGEMM("T", "T", &mk, &n, &k, &dbl1, &w[0 + k * ldw], &ldw,
                          &dwork[pdw6], &n, &dbl1, &a[ltra ? 0 + k * lda : k + 0 * lda], &lda);
            }
        }

        if (la1b1) {
            if (ltrb) {
                for (i = 0; i < k; i++) {
                    SLC_DCOPY(&n, &b[0 + i * ldb], &int1, &dwork[pdw9 + i * n], &int1);
                }
            } else {
                for (i = 0; i < n; i++) {
                    SLC_DCOPY(&k, &b[0 + i * ldb], &int1, &dwork[pdw9 + i], &n);
                }
            }

            SLC_DCOPY(&nk, &dwork[pdw9], &int1, &dwork[pdw1], &int1);
            if (lcolw) {
                SLC_DTRMM("R", "L", "N", "U", &n, &k, &dbl1, w, &ldw, &dwork[pdw1], &n);
            } else {
                SLC_DTRMM("R", "U", "T", "U", &n, &k, &dbl1, w, &ldw, &dwork[pdw1], &n);
            }

            SLC_DCOPY(&nk, &dwork[pdw9], &int1, &dwork[pdw2], &int1);
            if (lcolv) {
                SLC_DTRMM("R", "L", "N", "U", &n, &k, &dbl1, v, &ldv, &dwork[pdw2], &n);
            } else {
                SLC_DTRMM("R", "U", "T", "U", &n, &k, &dbl1, v, &ldv, &dwork[pdw2], &n);
            }
            fact = dbl1;
        } else {
            fact = dbl0;
        }

        if (m > k) {
            if (ltrb && lcolw) {
                SLC_DGEMM("N", "N", &n, &k, &mk, &dbl1, &b[0 + k * ldb], &ldb,
                          &w[k + 0 * ldw], &ldw, &fact, &dwork[pdw1], &n);
            } else if (ltrb) {
                SLC_DGEMM("N", "T", &n, &k, &mk, &dbl1, &b[0 + k * ldb], &ldb,
                          &w[0 + k * ldw], &ldw, &fact, &dwork[pdw1], &n);
            } else if (lcolw) {
                SLC_DGEMM("T", "N", &n, &k, &mk, &dbl1, &b[k + 0 * ldb], &ldb,
                          &w[k + 0 * ldw], &ldw, &fact, &dwork[pdw1], &n);
            } else {
                SLC_DGEMM("T", "T", &n, &k, &mk, &dbl1, &b[k + 0 * ldb], &ldb,
                          &w[0 + k * ldw], &ldw, &fact, &dwork[pdw1], &n);
            }
        } else if (!la1b1) {
            SLC_DLASET("A", &n, &k, &dbl0, &dbl0, &dwork[pdw1], &n);
        }

        if (m > k) {
            if (ltrb && lcolv) {
                SLC_DGEMM("N", "N", &n, &k, &mk, &dbl1, &b[0 + k * ldb], &ldb,
                          &v[k + 0 * ldv], &ldv, &fact, &dwork[pdw2], &n);
            } else if (ltrb) {
                SLC_DGEMM("N", "T", &n, &k, &mk, &dbl1, &b[0 + k * ldb], &ldb,
                          &v[0 + k * ldv], &ldv, &fact, &dwork[pdw2], &n);
            } else if (lcolv) {
                SLC_DGEMM("T", "N", &n, &k, &mk, &dbl1, &b[k + 0 * ldb], &ldb,
                          &v[k + 0 * ldv], &ldv, &fact, &dwork[pdw2], &n);
            } else {
                SLC_DGEMM("T", "T", &n, &k, &mk, &dbl1, &b[k + 0 * ldb], &ldb,
                          &v[0 + k * ldv], &ldv, &fact, &dwork[pdw2], &n);
            }
        } else if (!la1b1) {
            SLC_DLASET("A", &n, &k, &dbl0, &dbl0, &dwork[pdw2], &n);
        }

        SLC_DCOPY(&nk, &dwork[pdw1], &int1, &dwork[pdw3], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt11) * ldt], &ldt, &dwork[pdw3], &n);

        SLC_DCOPY(&nmk, &dwork[pdw2], &int1, &dwork[pdw4], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &km1, &dbl1, &t[0 + (pt31 + 1) * ldt], &ldt, &dwork[pdw4], &n);

        SLC_DAXPY(&nmk, &dbl1, &dwork[pdw4], &int1, &dwork[pdw3 + n], &int1);

        if (la1b1) {
            SLC_DCOPY(&nmk, &dwork[pdw9], &int1, &dwork[pdw8], &int1);
            SLC_DTRMM("R", "U", "N", "N", &n, &km1, &dbl1, &t[0 + (pt21 + 1) * ldt], &ldt, &dwork[pdw8], &n);
            SLC_DAXPY(&nmk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw3 + n], &int1);
        }

        SLC_DCOPY(&nk, &dwork[pdw1], &int1, &dwork[pdw4], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt12) * ldt], &ldt, &dwork[pdw4], &n);

        SLC_DCOPY(&nmk, &dwork[pdw2], &int1, &dwork[pdw5], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &km1, &dbl1, &t[0 + (pt32 + 1) * ldt], &ldt, &dwork[pdw5], &n);

        SLC_DAXPY(&nmk, &dbl1, &dwork[pdw5], &int1, &dwork[pdw4 + n], &int1);

        if (la1b1) {
            SLC_DCOPY(&nk, &dwork[pdw9], &int1, &dwork[pdw8], &int1);
            SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt22) * ldt], &ldt, &dwork[pdw8], &n);
            SLC_DAXPY(&nk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw4], &int1);
        }

        SLC_DCOPY(&nk, &dwork[pdw2], &int1, &dwork[pdw5], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt33) * ldt], &ldt, &dwork[pdw5], &n);

        SLC_DCOPY(&nk, &dwork[pdw1], &int1, &dwork[pdw6], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt13) * ldt], &ldt, &dwork[pdw6], &n);

        SLC_DAXPY(&nk, &dbl1, &dwork[pdw6], &int1, &dwork[pdw5], &int1);

        if (la1b1) {
            SLC_DCOPY(&nk, &dwork[pdw9], &int1, &dwork[pdw8], &int1);
            SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt23) * ldt], &ldt, &dwork[pdw8], &n);
            SLC_DAXPY(&nk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw5], &int1);
        }

        SLC_DCOPY(&nk, &dwork[pdw4], &int1, &dwork[pdw6], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &rs[(ps2) * ldrs], &ldrs, &dwork[pdw6], &n);

        SLC_DCOPY(&nmk, &dwork[pdw5], &int1, &dwork[pdw8], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &km1, &dbl1, &rs[0 + (ps1 + 1) * ldrs], &ldrs, &dwork[pdw8], &n);

        SLC_DAXPY(&nmk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw6 + n], &int1);

        f64 neg1 = -dbl1;
        SLC_DCOPY(&nk, &dwork[pdw4], &int1, &dwork[pdw8], &int1);
        SLC_DTRMM("R", "U", "N", "U", &n, &k, &dbl1, &rs[(pr2) * ldrs], &ldrs, &dwork[pdw8], &n);

        SLC_DAXPY(&nk, &neg1, &dwork[pdw3], &int1, &dwork[pdw8], &int1);
        SLC_DAXPY(&nk, &neg1, &dwork[pdw6], &int1, &dwork[pdw3], &int1);
        SLC_DCOPY(&nk, &dwork[pdw8], &int1, &dwork[pdw6], &int1);

        if (lcolv) {
            SLC_DTRMM("R", "L", "T", "U", &n, &k, &dbl1, v, &ldv, &dwork[pdw3], &n);
        } else {
            SLC_DTRMM("R", "U", "N", "U", &n, &k, &dbl1, v, &ldv, &dwork[pdw3], &n);
        }

        SLC_DCOPY(&nk, &dwork[pdw5], &int1, &dwork[pdw8], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &rs[(ps3) * ldrs], &ldrs, &dwork[pdw8], &n);

        SLC_DAXPY(&nk, &neg1, &dwork[pdw8], &int1, &dwork[pdw6], &int1);

        SLC_DCOPY(&nmk, &dwork[pdw4 + n], &int1, &dwork[pdw8], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &km1, &dbl1, &rs[0 + (pr3 + 1) * ldrs], &ldrs, &dwork[pdw8], &n);

        SLC_DAXPY(&nmk, &neg1, &dwork[pdw8], &int1, &dwork[pdw6 + n], &int1);

        SLC_DCOPY(&nk, &dwork[pdw6], &int1, &dwork[pdw8], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &rs[(pr1) * ldrs], &ldrs, &dwork[pdw8], &n);

        SLC_DAXPY(&nk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw6], &int1);

        if (lcolw) {
            SLC_DTRMM("R", "L", "T", "U", &n, &k, &dbl1, w, &ldw, &dwork[pdw6], &n);
        } else {
            SLC_DTRMM("R", "U", "N", "U", &n, &k, &dbl1, w, &ldw, &dwork[pdw6], &n);
        }

        if (ltrb) {
            for (i = 0; i < k; i++) {
                SLC_DAXPY(&n, &dbl1, &dwork[pdw3 + i * n], &int1, &b[0 + i * ldb], &int1);
                SLC_DAXPY(&n, &dbl1, &dwork[pdw6 + i * n], &int1, &b[0 + i * ldb], &int1);
            }
        } else {
            for (i = 0; i < n; i++) {
                SLC_DAXPY(&k, &dbl1, &dwork[pdw3 + i], &n, &b[0 + i * ldb], &int1);
                SLC_DAXPY(&k, &dbl1, &dwork[pdw6 + i], &n, &b[0 + i * ldb], &int1);
            }
        }

        if (m > k) {
            if (lcolv) {
                SLC_DGEMM("N", "T", &mk, &n, &k, &dbl1, &v[k + 0 * ldv], &ldv,
                          &dwork[pdw3], &n, &dbl1, &b[ltrb ? 0 + k * ldb : k + 0 * ldb], &ldb);
            } else {
                SLC_DGEMM("T", "T", &mk, &n, &k, &dbl1, &v[0 + k * ldv], &ldv,
                          &dwork[pdw3], &n, &dbl1, &b[ltrb ? 0 + k * ldb : k + 0 * ldb], &ldb);
            }

            if (lcolw) {
                SLC_DGEMM("N", "T", &mk, &n, &k, &dbl1, &w[k + 0 * ldw], &ldw,
                          &dwork[pdw6], &n, &dbl1, &b[ltrb ? 0 + k * ldb : k + 0 * ldb], &ldb);
            } else {
                SLC_DGEMM("T", "T", &mk, &n, &k, &dbl1, &w[0 + k * ldw], &ldw,
                          &dwork[pdw6], &n, &dbl1, &b[ltrb ? 0 + k * ldb : k + 0 * ldb], &ldb);
            }
        }
    } else {
        /* TRANQ = 'N' - apply Q (not Q^T) */
        /* Similar structure but different order of operations */
        /* This is a very long section - implementing later */

        /* Placeholder for TRANQ='N' case - apply Q */
        /* The structure mirrors the TRANQ='T' case but with different signs */
        /* and different order of operations on the block matrices */

        SLC_DCOPY(&nk, &dwork[pdw1], &int1, &dwork[pdw3], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt11) * ldt], &ldt, &dwork[pdw3], &n);

        SLC_DCOPY(&nmk, &dwork[pdw2], &int1, &dwork[pdw4], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &km1, &dbl1, &t[0 + (pt31 + 1) * ldt], &ldt, &dwork[pdw4], &n);

        SLC_DAXPY(&nmk, &dbl1, &dwork[pdw4], &int1, &dwork[pdw3 + n], &int1);

        if (la1b1) {
            SLC_DCOPY(&nmk, &dwork[pdw7], &int1, &dwork[pdw8], &int1);
            SLC_DTRMM("R", "U", "N", "N", &n, &km1, &dbl1, &t[0 + (pt21 + 1) * ldt], &ldt, &dwork[pdw8], &n);
            SLC_DAXPY(&nmk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw3 + n], &int1);
        }

        SLC_DCOPY(&nk, &dwork[pdw1], &int1, &dwork[pdw4], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt12) * ldt], &ldt, &dwork[pdw4], &n);

        SLC_DCOPY(&nmk, &dwork[pdw2], &int1, &dwork[pdw5], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &km1, &dbl1, &t[0 + (pt32 + 1) * ldt], &ldt, &dwork[pdw5], &n);

        SLC_DAXPY(&nmk, &dbl1, &dwork[pdw5], &int1, &dwork[pdw4 + n], &int1);

        if (la1b1) {
            SLC_DCOPY(&nk, &dwork[pdw7], &int1, &dwork[pdw8], &int1);
            SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt22) * ldt], &ldt, &dwork[pdw8], &n);
            SLC_DAXPY(&nk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw4], &int1);
        }

        SLC_DCOPY(&nk, &dwork[pdw2], &int1, &dwork[pdw5], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt33) * ldt], &ldt, &dwork[pdw5], &n);

        SLC_DCOPY(&nk, &dwork[pdw1], &int1, &dwork[pdw6], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt13) * ldt], &ldt, &dwork[pdw6], &n);

        SLC_DAXPY(&nk, &dbl1, &dwork[pdw6], &int1, &dwork[pdw5], &int1);

        if (la1b1) {
            SLC_DCOPY(&nk, &dwork[pdw7], &int1, &dwork[pdw8], &int1);
            SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt23) * ldt], &ldt, &dwork[pdw8], &n);
            SLC_DAXPY(&nk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw5], &int1);
        }

        SLC_DCOPY(&nk, &dwork[pdw4], &int1, &dwork[pdw6], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &rs[(ps2) * ldrs], &ldrs, &dwork[pdw6], &n);

        SLC_DCOPY(&nmk, &dwork[pdw5], &int1, &dwork[pdw8], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &km1, &dbl1, &rs[0 + (ps1 + 1) * ldrs], &ldrs, &dwork[pdw8], &n);

        SLC_DAXPY(&nmk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw6 + n], &int1);

        SLC_DCOPY(&nk, &dwork[pdw4], &int1, &dwork[pdw8], &int1);
        SLC_DTRMM("R", "U", "N", "U", &n, &k, &dbl1, &rs[(pr2) * ldrs], &ldrs, &dwork[pdw8], &n);

        SLC_DAXPY(&nk, &dbl1, &dwork[pdw3], &int1, &dwork[pdw8], &int1);
        SLC_DAXPY(&nk, &dbl1, &dwork[pdw6], &int1, &dwork[pdw3], &int1);
        SLC_DCOPY(&nk, &dwork[pdw8], &int1, &dwork[pdw6], &int1);

        if (lcolv) {
            SLC_DTRMM("R", "L", "T", "U", &n, &k, &dbl1, v, &ldv, &dwork[pdw3], &n);
        } else {
            SLC_DTRMM("R", "U", "N", "U", &n, &k, &dbl1, v, &ldv, &dwork[pdw3], &n);
        }

        SLC_DCOPY(&nk, &dwork[pdw5], &int1, &dwork[pdw8], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &rs[(ps3) * ldrs], &ldrs, &dwork[pdw8], &n);

        SLC_DAXPY(&nk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw6], &int1);

        SLC_DCOPY(&nmk, &dwork[pdw4 + n], &int1, &dwork[pdw8], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &km1, &dbl1, &rs[0 + (pr3 + 1) * ldrs], &ldrs, &dwork[pdw8], &n);

        SLC_DAXPY(&nmk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw6 + n], &int1);

        SLC_DCOPY(&nk, &dwork[pdw6], &int1, &dwork[pdw8], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &rs[(pr1) * ldrs], &ldrs, &dwork[pdw8], &n);

        SLC_DAXPY(&nk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw6], &int1);

        if (lcolw) {
            SLC_DTRMM("R", "L", "T", "U", &n, &k, &dbl1, w, &ldw, &dwork[pdw6], &n);
        } else {
            SLC_DTRMM("R", "U", "N", "U", &n, &k, &dbl1, w, &ldw, &dwork[pdw6], &n);
        }

        if (ltra) {
            for (i = 0; i < k; i++) {
                SLC_DAXPY(&n, &dbl1, &dwork[pdw3 + i * n], &int1, &a[0 + i * lda], &int1);
                SLC_DAXPY(&n, &dbl1, &dwork[pdw6 + i * n], &int1, &a[0 + i * lda], &int1);
            }
        } else {
            for (i = 0; i < n; i++) {
                SLC_DAXPY(&k, &dbl1, &dwork[pdw3 + i], &n, &a[0 + i * lda], &int1);
                SLC_DAXPY(&k, &dbl1, &dwork[pdw6 + i], &n, &a[0 + i * lda], &int1);
            }
        }

        if (m > k) {
            if (lcolv) {
                SLC_DGEMM("N", "T", &mk, &n, &k, &dbl1, &v[k + 0 * ldv], &ldv,
                          &dwork[pdw3], &n, &dbl1, &a[ltra ? 0 + k * lda : k + 0 * lda], &lda);
            } else {
                SLC_DGEMM("T", "T", &mk, &n, &k, &dbl1, &v[0 + k * ldv], &ldv,
                          &dwork[pdw3], &n, &dbl1, &a[ltra ? 0 + k * lda : k + 0 * lda], &lda);
            }

            if (lcolw) {
                SLC_DGEMM("N", "T", &mk, &n, &k, &dbl1, &w[k + 0 * ldw], &ldw,
                          &dwork[pdw6], &n, &dbl1, &a[ltra ? 0 + k * lda : k + 0 * lda], &lda);
            } else {
                SLC_DGEMM("T", "T", &mk, &n, &k, &dbl1, &w[0 + k * ldw], &ldw,
                          &dwork[pdw6], &n, &dbl1, &a[ltra ? 0 + k * lda : k + 0 * lda], &lda);
            }
        }

        if (la1b1) {
            if (ltrb) {
                for (i = 0; i < k; i++) {
                    SLC_DCOPY(&n, &b[0 + i * ldb], &int1, &dwork[pdw9 + i * n], &int1);
                }
            } else {
                for (i = 0; i < n; i++) {
                    SLC_DCOPY(&k, &b[0 + i * ldb], &int1, &dwork[pdw9 + i], &n);
                }
            }

            SLC_DCOPY(&nk, &dwork[pdw9], &int1, &dwork[pdw1], &int1);
            if (lcolw) {
                SLC_DTRMM("R", "L", "N", "U", &n, &k, &dbl1, w, &ldw, &dwork[pdw1], &n);
            } else {
                SLC_DTRMM("R", "U", "T", "U", &n, &k, &dbl1, w, &ldw, &dwork[pdw1], &n);
            }

            SLC_DCOPY(&nk, &dwork[pdw9], &int1, &dwork[pdw2], &int1);
            if (lcolv) {
                SLC_DTRMM("R", "L", "N", "U", &n, &k, &dbl1, v, &ldv, &dwork[pdw2], &n);
            } else {
                SLC_DTRMM("R", "U", "T", "U", &n, &k, &dbl1, v, &ldv, &dwork[pdw2], &n);
            }
            fact = dbl1;
        } else {
            fact = dbl0;
        }

        if (m > k) {
            if (ltrb && lcolw) {
                SLC_DGEMM("N", "N", &n, &k, &mk, &dbl1, &b[0 + k * ldb], &ldb,
                          &w[k + 0 * ldw], &ldw, &fact, &dwork[pdw1], &n);
            } else if (ltrb) {
                SLC_DGEMM("N", "T", &n, &k, &mk, &dbl1, &b[0 + k * ldb], &ldb,
                          &w[0 + k * ldw], &ldw, &fact, &dwork[pdw1], &n);
            } else if (lcolw) {
                SLC_DGEMM("T", "N", &n, &k, &mk, &dbl1, &b[k + 0 * ldb], &ldb,
                          &w[k + 0 * ldw], &ldw, &fact, &dwork[pdw1], &n);
            } else {
                SLC_DGEMM("T", "T", &n, &k, &mk, &dbl1, &b[k + 0 * ldb], &ldb,
                          &w[0 + k * ldw], &ldw, &fact, &dwork[pdw1], &n);
            }
        } else if (!la1b1) {
            SLC_DLASET("A", &n, &k, &dbl0, &dbl0, &dwork[pdw1], &n);
        }

        if (m > k) {
            if (ltrb && lcolv) {
                SLC_DGEMM("N", "N", &n, &k, &mk, &dbl1, &b[0 + k * ldb], &ldb,
                          &v[k + 0 * ldv], &ldv, &fact, &dwork[pdw2], &n);
            } else if (ltrb) {
                SLC_DGEMM("N", "T", &n, &k, &mk, &dbl1, &b[0 + k * ldb], &ldb,
                          &v[0 + k * ldv], &ldv, &fact, &dwork[pdw2], &n);
            } else if (lcolv) {
                SLC_DGEMM("T", "N", &n, &k, &mk, &dbl1, &b[k + 0 * ldb], &ldb,
                          &v[k + 0 * ldv], &ldv, &fact, &dwork[pdw2], &n);
            } else {
                SLC_DGEMM("T", "T", &n, &k, &mk, &dbl1, &b[k + 0 * ldb], &ldb,
                          &v[0 + k * ldv], &ldv, &fact, &dwork[pdw2], &n);
            }
        } else if (!la1b1) {
            SLC_DLASET("A", &n, &k, &dbl0, &dbl0, &dwork[pdw2], &n);
        }

        SLC_DCOPY(&nk, &dwork[pdw1], &int1, &dwork[pdw3], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt11) * ldt], &ldt, &dwork[pdw3], &n);

        SLC_DCOPY(&nmk, &dwork[pdw2], &int1, &dwork[pdw4], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &km1, &dbl1, &t[0 + (pt31 + 1) * ldt], &ldt, &dwork[pdw4], &n);

        SLC_DAXPY(&nmk, &dbl1, &dwork[pdw4], &int1, &dwork[pdw3 + n], &int1);

        if (la1b1) {
            SLC_DCOPY(&nmk, &dwork[pdw9], &int1, &dwork[pdw8], &int1);
            SLC_DTRMM("R", "U", "N", "N", &n, &km1, &dbl1, &t[0 + (pt21 + 1) * ldt], &ldt, &dwork[pdw8], &n);
            SLC_DAXPY(&nmk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw3 + n], &int1);
        }

        SLC_DCOPY(&nk, &dwork[pdw1], &int1, &dwork[pdw4], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt12) * ldt], &ldt, &dwork[pdw4], &n);

        SLC_DCOPY(&nmk, &dwork[pdw2], &int1, &dwork[pdw5], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &km1, &dbl1, &t[0 + (pt32 + 1) * ldt], &ldt, &dwork[pdw5], &n);

        SLC_DAXPY(&nmk, &dbl1, &dwork[pdw5], &int1, &dwork[pdw4 + n], &int1);

        if (la1b1) {
            SLC_DCOPY(&nk, &dwork[pdw9], &int1, &dwork[pdw8], &int1);
            SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt22) * ldt], &ldt, &dwork[pdw8], &n);
            SLC_DAXPY(&nk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw4], &int1);
        }

        SLC_DCOPY(&nk, &dwork[pdw2], &int1, &dwork[pdw5], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt33) * ldt], &ldt, &dwork[pdw5], &n);

        SLC_DCOPY(&nk, &dwork[pdw1], &int1, &dwork[pdw6], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt13) * ldt], &ldt, &dwork[pdw6], &n);

        SLC_DAXPY(&nk, &dbl1, &dwork[pdw6], &int1, &dwork[pdw5], &int1);

        if (la1b1) {
            SLC_DCOPY(&nk, &dwork[pdw9], &int1, &dwork[pdw8], &int1);
            SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &t[(pt23) * ldt], &ldt, &dwork[pdw8], &n);
            SLC_DAXPY(&nk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw5], &int1);
        }

        f64 neg1 = -dbl1;

        SLC_DCOPY(&nk, &dwork[pdw4], &int1, &dwork[pdw6], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &rs[(ps2) * ldrs], &ldrs, &dwork[pdw6], &n);

        SLC_DCOPY(&nmk, &dwork[pdw5], &int1, &dwork[pdw8], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &km1, &dbl1, &rs[0 + (ps1 + 1) * ldrs], &ldrs, &dwork[pdw8], &n);

        SLC_DAXPY(&nmk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw6 + n], &int1);

        SLC_DCOPY(&nk, &dwork[pdw4], &int1, &dwork[pdw8], &int1);
        SLC_DTRMM("R", "U", "N", "U", &n, &k, &dbl1, &rs[(pr2) * ldrs], &ldrs, &dwork[pdw8], &n);

        SLC_DAXPY(&nk, &neg1, &dwork[pdw3], &int1, &dwork[pdw8], &int1);
        SLC_DAXPY(&nk, &neg1, &dwork[pdw6], &int1, &dwork[pdw3], &int1);
        SLC_DCOPY(&nk, &dwork[pdw8], &int1, &dwork[pdw6], &int1);

        if (lcolv) {
            SLC_DTRMM("R", "L", "T", "U", &n, &k, &dbl1, v, &ldv, &dwork[pdw3], &n);
        } else {
            SLC_DTRMM("R", "U", "N", "U", &n, &k, &dbl1, v, &ldv, &dwork[pdw3], &n);
        }

        SLC_DCOPY(&nk, &dwork[pdw5], &int1, &dwork[pdw8], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &rs[(ps3) * ldrs], &ldrs, &dwork[pdw8], &n);

        SLC_DAXPY(&nk, &neg1, &dwork[pdw8], &int1, &dwork[pdw6], &int1);

        SLC_DCOPY(&nmk, &dwork[pdw4 + n], &int1, &dwork[pdw8], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &km1, &dbl1, &rs[0 + (pr3 + 1) * ldrs], &ldrs, &dwork[pdw8], &n);

        SLC_DAXPY(&nmk, &neg1, &dwork[pdw8], &int1, &dwork[pdw6 + n], &int1);

        SLC_DCOPY(&nk, &dwork[pdw6], &int1, &dwork[pdw8], &int1);
        SLC_DTRMM("R", "U", "N", "N", &n, &k, &dbl1, &rs[(pr1) * ldrs], &ldrs, &dwork[pdw8], &n);

        SLC_DAXPY(&nk, &dbl1, &dwork[pdw8], &int1, &dwork[pdw6], &int1);

        if (lcolw) {
            SLC_DTRMM("R", "L", "T", "U", &n, &k, &dbl1, w, &ldw, &dwork[pdw6], &n);
        } else {
            SLC_DTRMM("R", "U", "N", "U", &n, &k, &dbl1, w, &ldw, &dwork[pdw6], &n);
        }

        if (ltrb) {
            for (i = 0; i < k; i++) {
                SLC_DAXPY(&n, &dbl1, &dwork[pdw3 + i * n], &int1, &b[0 + i * ldb], &int1);
                SLC_DAXPY(&n, &dbl1, &dwork[pdw6 + i * n], &int1, &b[0 + i * ldb], &int1);
            }
        } else {
            for (i = 0; i < n; i++) {
                SLC_DAXPY(&k, &dbl1, &dwork[pdw3 + i], &n, &b[0 + i * ldb], &int1);
                SLC_DAXPY(&k, &dbl1, &dwork[pdw6 + i], &n, &b[0 + i * ldb], &int1);
            }
        }

        if (m > k) {
            if (lcolv) {
                SLC_DGEMM("N", "T", &mk, &n, &k, &dbl1, &v[k + 0 * ldv], &ldv,
                          &dwork[pdw3], &n, &dbl1, &b[ltrb ? 0 + k * ldb : k + 0 * ldb], &ldb);
            } else {
                SLC_DGEMM("T", "T", &mk, &n, &k, &dbl1, &v[0 + k * ldv], &ldv,
                          &dwork[pdw3], &n, &dbl1, &b[ltrb ? 0 + k * ldb : k + 0 * ldb], &ldb);
            }

            if (lcolw) {
                SLC_DGEMM("N", "T", &mk, &n, &k, &dbl1, &w[k + 0 * ldw], &ldw,
                          &dwork[pdw6], &n, &dbl1, &b[ltrb ? 0 + k * ldb : k + 0 * ldb], &ldb);
            } else {
                SLC_DGEMM("T", "T", &mk, &n, &k, &dbl1, &w[0 + k * ldw], &ldw,
                          &dwork[pdw6], &n, &dbl1, &b[ltrb ? 0 + k * ldb : k + 0 * ldb], &ldb);
            }
        }
    }
}
