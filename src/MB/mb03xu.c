// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <stddef.h>

// 1-based Fortran-style element access, kept literal to mirror MB03XU.f.
#define A_(r, c) a[((r) - 1) + (ptrdiff_t)((c) - 1) * lda]
#define B_(r, c) b[((r) - 1) + (ptrdiff_t)((c) - 1) * ldb]
#define G_(r, c) g[((r) - 1) + (ptrdiff_t)((c) - 1) * ldg]
#define Q_(r, c) q[((r) - 1) + (ptrdiff_t)((c) - 1) * ldq]
#define XA_(r, c) xa[((r) - 1) + (ptrdiff_t)((c) - 1) * ldxa]
#define XB_(r, c) xb[((r) - 1) + (ptrdiff_t)((c) - 1) * ldxb]
#define XG_(r, c) xg[((r) - 1) + (ptrdiff_t)((c) - 1) * ldxg]
#define XQ_(r, c) xq[((r) - 1) + (ptrdiff_t)((c) - 1) * ldxq]
#define YA_(r, c) ya[((r) - 1) + (ptrdiff_t)((c) - 1) * ldya]
#define YB_(r, c) yb[((r) - 1) + (ptrdiff_t)((c) - 1) * ldyb]
#define YG_(r, c) yg[((r) - 1) + (ptrdiff_t)((c) - 1) * ldyg]
#define YQ_(r, c) yq[((r) - 1) + (ptrdiff_t)((c) - 1) * ldyq]
#define DW(k) dwork[(k) - 1]
#define CSL_(k) csl[(k) - 1]
#define CSR_(k) csr[(k) - 1]
#define TAUL_(k) taul[(k) - 1]
#define TAUR_(k) taur[(k) - 1]

void mb03xu(bool ltra, bool ltrb, i32 n, i32 k, i32 nb,
            f64 *a, i32 lda, f64 *b, i32 ldb, f64 *g, i32 ldg,
            f64 *q, i32 ldq, f64 *xa, i32 ldxa, f64 *xb, i32 ldxb,
            f64 *xg, i32 ldxg, f64 *xq, i32 ldxq, f64 *ya, i32 ldya,
            f64 *yb, i32 ldyb, f64 *yg, i32 ldyg, f64 *yq, i32 ldyq,
            f64 *csl, f64 *csr, f64 *taul, f64 *taur, f64 *dwork) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    i32 int1 = 1;
    i32 i, j;
    f64 alpha, c, s, tauq, temp;

    if (n + k <= 0) {
        return;
    }

    i32 nb1 = nb + 1;
    i32 nb2 = nb + nb;
    i32 nb3 = nb2 + nb;
    i32 pdw = nb3 + nb + 1;

    if (ltra && ltrb) {
        for (i = 1; i <= nb; i++) {
            alpha = Q_(i, i);
            SLC_DLARFG(&(i32){n - i + 1}, &alpha, &Q_(i + 1, i), &int1, &tauq);
            Q_(i, i) = ONE;
            temp = -tauq*SLC_DDOT(&(i32){n - i + 1}, &Q_(i, i), &int1, &A_(i, k + i), &lda);
            SLC_DAXPY(&(i32){n - i + 1}, &temp, &Q_(i, i), &int1, &A_(i, k + i), &lda);
            temp = A_(i, k + i);
            SLC_DLARTG(&temp, &alpha, &c, &s, &A_(i, k + i));
            SLC_DLARFG(&(i32){n - i + 1}, &A_(i, k + i), &A_(i, k + i + 1), &lda, &TAUL_(i));
            temp = A_(i, k + i);
            A_(i, k + i) = ONE;
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){n - i}, &ONE, &Q_(i, i + 1), &ldq, &Q_(i, i), &int1, &ZERO, &XQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &Q_(i, 1), &ldq, &Q_(i, i), &int1, &ZERO, dwork, &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XQ_(i + 1, 1), &ldxq, dwork, &int1, &ONE, &XQ_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){i - 1}, &(i32){n - i + 1}, &ONE, &A_(1, k + i), &lda, &Q_(i, i), &int1, &ZERO, &DW(nb + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XQ_(i + 1, nb1), &ldxq, &DW(nb + 1), &int1, &ONE, &XQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YQ_(i, 1), &ldyq, &Q_(i, i), &int1, &ZERO, &XQ_(1, i), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &XQ_(1, i), &int1, &ONE, &XQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YQ_(i, nb1), &ldyq, &Q_(i, i), &int1, &ZERO, &XQ_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &XQ_(1, i + nb), &int1, &ONE, &XQ_(i + 1, i), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-tauq}, &XQ_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &XQ_(i + 1, 1), &ldxq, &Q_(i, 1), &ldq, &ONE, &Q_(i, i + 1), &ldq);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XQ_(i + 1, nb1), &ldxq, &A_(1, k + i), &int1, &ONE, &Q_(i, i + 1), &ldq);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &YQ_(i, 1), &ldyq, &ONE, &Q_(i, i + 1), &ldq);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &YQ_(i, nb1), &ldyq, &ONE, &Q_(i, i + 1), &ldq);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){n - i + 1}, &ONE, &A_(i + 1, k + i), &lda, &Q_(i, i), &int1, &ZERO, &XA_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XA_(i + 1, 1), &ldxa, dwork, &int1, &ONE, &XA_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XA_(i + 1, nb1), &ldxa, &DW(nb + 1), &int1, &ONE, &XA_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YA_(k + i, 1), &ldya, &Q_(i, i), &int1, &ZERO, &XA_(1, i), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &XA_(1, i), &int1, &ONE, &XA_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YA_(k + i, nb1), &ldya, &Q_(i, i), &int1, &ZERO, &XA_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &XA_(1, i), &int1, &ONE, &XA_(i + 1, i), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-tauq}, &XA_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &XA_(i + 1, 1), &ldxa, &Q_(i, 1), &ldq, &ONE, &A_(i + 1, k + i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XA_(i + 1, nb1), &ldxa, &A_(1, k + i), &int1, &ONE, &A_(i + 1, k + i), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &YA_(k + i, 1), &ldya, &ONE, &A_(i + 1, k + i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &YA_(k + i, nb1), &ldya, &ONE, &A_(i + 1, k + i), &int1);
            SLC_DROT(&(i32){n - i}, &A_(i + 1, k + i), &int1, &Q_(i, i + 1), &ldq, &c, &s);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){n - i}, &ONE, &Q_(i, i + 1), &ldq, &A_(i, k + i), &lda, &ZERO, &XQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &A_(i, k + i + 1), &lda, &ZERO, &DW(nb2 + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &XQ_(i + 1, 1), &ldxq, &DW(nb2 + 1), &int1, &ONE, &XQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){i - 1}, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &A_(i, k + i + 1), &lda, &ZERO, &DW(nb3 + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XQ_(i + 1, nb1), &ldxq, &DW(nb3 + 1), &int1, &ONE, &XQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, 1), &ldyq, &A_(i, k + i + 1), &lda, &ZERO, &XQ_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &XQ_(1, i + nb), &int1, &ONE, &XQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, nb1), &ldyq, &A_(i, k + i + 1), &lda, &ZERO, &XQ_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &XQ_(1, i + nb), &int1, &ONE, &XQ_(i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-TAUL_(i)}, &XQ_(i + 1, i + nb), &int1);
            SLC_DAXPY(&(i32){n - i}, &ONE, &XQ_(i + 1, i + nb), &int1, &Q_(i, i + 1), &ldq);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){n - i + 1}, &ONE, &A_(i + 1, k + i), &lda, &A_(i, k + i), &lda, &ZERO, &XA_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &XA_(i + 1, 1), &ldxa, &DW(nb2 + 1), &int1, &ONE, &XA_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XA_(i + 1, nb1), &ldxa, &DW(nb3 + 1), &int1, &ONE, &XA_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YA_(k + i + 1, 1), &ldya, &A_(i, k + i + 1), &lda, &ZERO, &XA_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &XA_(1, i + nb), &int1, &ONE, &XA_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YA_(k + i + 1, nb1), &ldya, &A_(i, k + i + 1), &lda, &ZERO, &XA_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &XA_(1, i + nb), &int1, &ONE, &XA_(i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-TAUL_(i)}, &XA_(i + 1, i + nb), &int1);
            SLC_DAXPY(&(i32){n - i}, &ONE, &XA_(i + 1, i + nb), &int1, &A_(i + 1, k + i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){k + n}, &ONE, &G_(k + i, 1), &ldg, &Q_(i, i), &int1, &ZERO, &XG_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XG_(1, 1), &ldxg, dwork, &int1, &ONE, &XG_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XG_(1, nb1), &ldxg, &DW(nb + 1), &int1, &ONE, &XG_(1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YG_(k + i, 1), &ldyg, &Q_(i, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &DW(pdw), &int1, &ONE, &XG_(k + i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YG_(k + i, nb1), &ldyg, &Q_(i, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &DW(pdw), &int1, &ONE, &XG_(k + i + 1, i), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-tauq}, &XG_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &XG_(1, 1), &ldxg, &Q_(i, 1), &ldq, &ONE, &G_(k + i, 1), &ldg);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XG_(1, nb1), &ldxg, &A_(1, k + i), &int1, &ONE, &G_(k + i, 1), &ldg);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &YG_(k + i, 1), &ldyg, &ONE, &G_(k + i, k + i + 1), &ldg);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &YG_(k + i, nb1), &ldyg, &ONE, &G_(k + i, k + i + 1), &ldg);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){n - i + 1}, &ONE, &B_(1, i), &ldb, &Q_(i, i), &int1, &ZERO, &XB_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XB_(1, 1), &ldxb, dwork, &int1, &ONE, &XB_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XB_(1, nb1), &ldxb, &DW(nb + 1), &int1, &ONE, &XB_(1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YB_(i, 1), &ldyb, &Q_(i, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &DW(pdw), &int1, &ONE, &XB_(k + i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YB_(i, nb1), &ldyb, &Q_(i, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &DW(pdw), &int1, &ONE, &XB_(k + i + 1, i), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-tauq}, &XB_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &XB_(1, 1), &ldxb, &Q_(i, 1), &ldq, &ONE, &B_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XB_(1, nb1), &ldxb, &A_(1, k + i), &int1, &ONE, &B_(1, i), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &YB_(i, 1), &ldyb, &ONE, &B_(k + i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &YB_(i, nb1), &ldyb, &ONE, &B_(k + i + 1, i), &int1);
            SLC_DROT(&(i32){k + n}, &G_(k + i, 1), &ldg, &B_(1, i), &int1, &c, &s);
            for (j = 1; j <= i - 1; j++) {
                YG_(k + i, j) = ZERO;
            }
            for (j = 1; j <= i - 1; j++) {
                YG_(k + i, nb + j) = ZERO;
            }
            for (j = 1; j <= i - 1; j++) {
                YA_(k + i, j) = ZERO;
            }
            for (j = 1; j <= i - 1; j++) {
                YA_(k + i, nb + j) = ZERO;
            }
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){k + n}, &ONE, &G_(k + i, 1), &ldg, &A_(i, k + i), &lda, &ZERO, &XG_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &XG_(1, 1), &ldxg, &DW(nb2 + 1), &int1, &ONE, &XG_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XG_(1, nb1), &ldxg, &DW(nb3 + 1), &int1, &ONE, &XG_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YG_(k + i + 1, 1), &ldyg, &A_(i, k + i + 1), &lda, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &DW(pdw), &int1, &ONE, &XG_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YG_(k + i + 1, nb1), &ldyg, &A_(i, k + i + 1), &lda, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &DW(pdw), &int1, &ONE, &XG_(k + i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-TAUL_(i)}, &XG_(1, i + nb), &int1);
            SLC_DAXPY(&(i32){k + n}, &ONE, &XG_(1, i + nb), &int1, &G_(k + i, 1), &ldg);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){n - i + 1}, &ONE, &B_(1, i), &ldb, &A_(i, k + i), &lda, &ZERO, &XB_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &XB_(1, 1), &ldxb, &DW(nb2 + 1), &int1, &ONE, &XB_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XB_(1, nb1), &ldxb, &DW(nb3 + 1), &int1, &ONE, &XB_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, 1), &ldyb, &A_(i, k + i + 1), &lda, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &DW(pdw), &int1, &ONE, &XB_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, nb1), &ldyb, &A_(i, k + i + 1), &lda, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &DW(pdw), &int1, &ONE, &XB_(k + i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-TAUL_(i)}, &XB_(1, i + nb), &int1);
            SLC_DAXPY(&(i32){k + n}, &ONE, &XB_(1, i + nb), &int1, &B_(1, i), &int1);
            A_(i, k + i) = temp;
            Q_(i, i) = tauq;
            CSL_(2*i - 1) = c;
            CSL_(2*i) = s;
            alpha = Q_(i, i + 1);
            SLC_DLARFG(&(i32){n - i}, &alpha, &Q_(i, i + 2), &ldq, &tauq);
            Q_(i, i + 1) = ONE;
            temp = -tauq*SLC_DDOT(&(i32){n - i}, &Q_(i, i + 1), &ldq, &B_(k + i + 1, i), &int1);
            SLC_DAXPY(&(i32){n - i}, &temp, &Q_(i, i + 1), &ldq, &B_(k + i + 1, i), &int1);
            temp = B_(k + i + 1, i);
            SLC_DLARTG(&temp, &alpha, &c, &s, &B_(k + i + 1, i));
            s = -s;
            SLC_DLARFG(&(i32){n - i}, &B_(k + i + 1, i), &B_(k + i + 2, i), &int1, &TAUR_(i));
            temp = B_(k + i + 1, i);
            B_(k + i + 1, i) = ONE;
            SLC_DGEMV("T", &(i32){n - i}, &(i32){n - i}, &ONE, &B_(k + i + 1, i + 1), &ldb, &Q_(i, i + 1), &ldq, &ZERO, &YB_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XB_(k + i + 1, 1), &ldxb, &Q_(i, i + 1), &ldq, &ZERO, &YB_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &YB_(1, i), &int1, &ONE, &YB_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XB_(k + i + 1, nb1), &ldxb, &Q_(i, i + 1), &ldq, &ZERO, &YB_(1, i), &int1);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &YB_(1, i), &int1, &ONE, &YB_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &Q_(i, i + 1), &ldq, &ZERO, dwork, &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, 1), &ldyb, dwork, &int1, &ONE, &YB_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &Q_(i, i + 1), &ldq, &ZERO, &DW(nb + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, nb1), &ldyb, &DW(nb + 1), &int1, &ONE, &YB_(i + 1, i), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-tauq}, &YB_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &XB_(k + i + 1, 1), &ldxb, &ONE, &B_(k + i + 1, i + 1), &ldb);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &XB_(k + i + 1, nb1), &ldxb, &ONE, &B_(k + i + 1, i + 1), &ldb);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &YB_(i + 1, 1), &ldyb, &Q_(1, i + 1), &int1, &ONE, &B_(k + i + 1, i + 1), &ldb);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, nb1), &ldyb, &B_(k + i + 1, 1), &ldb, &ONE, &B_(k + i + 1, i + 1), &ldb);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){n - i}, &ONE, &Q_(i + 1, i + 1), &ldq, &Q_(i, i + 1), &ldq, &ZERO, &YQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XQ_(i + 1, 1), &ldxq, &Q_(i, i + 1), &ldq, &ZERO, &YQ_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &YQ_(1, i), &int1, &ONE, &YQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XQ_(i + 1, nb1), &ldxq, &Q_(i, i + 1), &ldq, &ZERO, &YQ_(1, i), &int1);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &YQ_(1, i), &int1, &ONE, &YQ_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, 1), &ldyq, dwork, &int1, &ONE, &YQ_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, nb1), &ldyq, &DW(nb + 1), &int1, &ONE, &YQ_(i + 1, i), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-tauq}, &YQ_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &XQ_(i + 1, 1), &ldxq, &ONE, &Q_(i + 1, i + 1), &int1);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &XQ_(i + 1, nb1), &ldxq, &ONE, &Q_(i + 1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &YQ_(i + 1, 1), &ldyq, &Q_(1, i + 1), &int1, &ONE, &Q_(i + 1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, nb1), &ldyq, &B_(k + i + 1, 1), &ldb, &ONE, &Q_(i + 1, i + 1), &int1);
            SLC_DROT(&(i32){n - i}, &Q_(i + 1, i + 1), &int1, &B_(k + i + 1, i + 1), &ldb, &c, &s);
            for (j = 1; j <= i; j++) {
                XB_(k + i + 1, j) = ZERO;
            }
            for (j = 1; j <= i; j++) {
                XB_(k + i + 1, nb + j) = ZERO;
            }
            SLC_DGEMV("T", &(i32){n - i}, &(i32){n - i}, &ONE, &B_(k + i + 1, i + 1), &ldb, &B_(k + i + 1, i), &int1, &ZERO, &YB_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XB_(k + i + 2, 1), &ldxb, &B_(k + i + 2, i), &int1, &ZERO, &YB_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &YB_(1, i + nb), &int1, &ONE, &YB_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XB_(k + i + 2, nb1), &ldxb, &B_(k + i + 2, i), &int1, &ZERO, &YB_(1, i + nb), &int1);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &YB_(1, i + nb), &int1, &ONE, &YB_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &i, &(i32){n - i - 1}, &ONE, &Q_(1, i + 2), &ldq, &B_(k + i + 2, i), &int1, &ZERO, &DW(nb2 + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &YB_(i + 1, 1), &ldyb, &DW(nb2 + 1), &int1, &ONE, &YB_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &(i32){i - 1}, &ONE, &B_(k + i + 2, 1), &ldb, &B_(k + i + 2, i), &int1, &ZERO, &DW(nb3 + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, nb1), &ldyb, &DW(nb3 + 1), &int1, &ONE, &YB_(i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-TAUR_(i)}, &YB_(i + 1, i + nb), &int1);
            SLC_DAXPY(&(i32){n - i}, &ONE, &YB_(i + 1, i + nb), &int1, &B_(k + i + 1, i + 1), &ldb);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){n - i}, &ONE, &Q_(i + 1, i + 1), &ldq, &B_(k + i + 1, i), &int1, &ZERO, &YQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XQ_(i + 2, 1), &ldxq, &B_(k + i + 2, i), &int1, &ZERO, &YQ_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &YQ_(1, i + nb), &int1, &ONE, &YQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XQ_(i + 2, nb1), &ldxq, &B_(k + i + 2, i), &int1, &ZERO, &YQ_(1, i + nb), &int1);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &YQ_(1, i + nb), &int1, &ONE, &YQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &YQ_(i + 1, 1), &ldyq, &DW(nb2 + 1), &int1, &ONE, &YQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, nb1), &ldyq, &DW(nb3 + 1), &int1, &ONE, &YQ_(i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-TAUR_(i)}, &YQ_(i + 1, i + nb), &int1);
            SLC_DAXPY(&(i32){n - i}, &ONE, &YQ_(i + 1, i + nb), &int1, &Q_(i + 1, i + 1), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){k + n}, &ONE, &A_(i + 1, 1), &lda, &Q_(i, i + 1), &ldq, &ZERO, &YA_(1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XA_(i + 1, 1), &ldxa, &Q_(i, i + 1), &ldq, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &DW(pdw), &int1, &ONE, &YA_(k + i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XA_(i + 1, nb1), &ldxa, &Q_(i, i + 1), &ldq, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &DW(pdw), &int1, &ONE, &YA_(k + i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YA_(1, 1), &ldya, dwork, &int1, &ONE, &YA_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YA_(1, nb1), &ldya, &DW(nb + 1), &int1, &ONE, &YA_(1, i), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-tauq}, &YA_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &XA_(i + 1, 1), &ldxa, &ONE, &A_(i + 1, k + i + 1), &lda);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &XA_(i + 1, nb1), &ldxa, &ONE, &A_(i + 1, k + i + 1), &lda);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &YA_(1, 1), &ldya, &Q_(1, i + 1), &int1, &ONE, &A_(i + 1, 1), &lda);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YA_(1, nb1), &ldya, &B_(k + i + 1, 1), &ldb, &ONE, &A_(i + 1, 1), &lda);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){n - i}, &ONE, &G_(1, k + i + 1), &ldg, &Q_(i, i + 1), &ldq, &ZERO, &YG_(1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XG_(k + i + 1, 1), &ldxg, &Q_(i, i + 1), &ldq, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &DW(pdw), &int1, &ONE, &YG_(k + i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XG_(k + i + 1, nb1), &ldxg, &Q_(i, i + 1), &ldq, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &DW(pdw), &int1, &ONE, &YG_(k + i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YG_(1, 1), &ldyg, dwork, &int1, &ONE, &YG_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YG_(1, nb1), &ldyg, &DW(nb + 1), &int1, &ONE, &YG_(1, i), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-tauq}, &YG_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &XG_(k + i + 1, 1), &ldxg, &ONE, &G_(k + i + 1, k + i + 1), &int1);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &XG_(k + i + 1, nb1), &ldxg, &ONE, &G_(k + i + 1, k + i + 1), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &YG_(1, 1), &ldyg, &Q_(1, i + 1), &int1, &ONE, &G_(1, k + i + 1), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YG_(1, nb1), &ldyg, &B_(k + i + 1, 1), &ldb, &ONE, &G_(1, k + i + 1), &int1);
            for (j = 1; j <= i; j++) {
                XG_(k + i + 1, j) = ZERO;
            }
            for (j = 1; j <= i; j++) {
                XG_(k + i + 1, nb + j) = ZERO;
            }
            SLC_DROT(&(i32){k + n}, &A_(i + 1, 1), &lda, &G_(1, k + i + 1), &int1, &c, &s);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){k + n}, &ONE, &A_(i + 1, 1), &lda, &B_(k + i + 1, i), &int1, &ZERO, &YA_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XA_(i + 2, 1), &ldxa, &B_(k + i + 2, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &DW(pdw), &int1, &ONE, &YA_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XA_(i + 2, nb1), &ldxa, &B_(k + i + 2, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &DW(pdw), &int1, &ONE, &YA_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &YA_(1, 1), &ldya, &DW(nb2 + 1), &int1, &ONE, &YA_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YA_(1, nb1), &ldya, &DW(nb3 + 1), &int1, &ONE, &YA_(1, i + nb), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-TAUR_(i)}, &YA_(1, i + nb), &int1);
            SLC_DAXPY(&(i32){k + n}, &ONE, &YA_(1, i + nb), &int1, &A_(i + 1, 1), &lda);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){n - i}, &ONE, &G_(1, k + i + 1), &ldg, &B_(k + i + 1, i), &int1, &ZERO, &YG_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XG_(k + i + 2, 1), &ldxg, &B_(k + i + 2, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &DW(pdw), &int1, &ONE, &YG_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XG_(k + i + 2, nb1), &ldxg, &B_(k + i + 2, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &DW(pdw), &int1, &ONE, &YG_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &YG_(1, 1), &ldyg, &DW(nb2 + 1), &int1, &ONE, &YG_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YG_(1, nb1), &ldyg, &DW(nb3 + 1), &int1, &ONE, &YG_(1, i + nb), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-TAUR_(i)}, &YG_(1, i + nb), &int1);
            SLC_DAXPY(&(i32){k + n}, &ONE, &YG_(1, i + nb), &int1, &G_(1, k + i + 1), &int1);
            B_(k + i + 1, i) = temp;
            Q_(i, i + 1) = tauq;
            CSR_(2*i - 1) = c;
            CSR_(2*i) = s;
        }
    } else if (ltra) {
        for (i = 1; i <= nb; i++) {
            alpha = Q_(i, i);
            SLC_DLARFG(&(i32){n - i + 1}, &alpha, &Q_(i + 1, i), &int1, &tauq);
            Q_(i, i) = ONE;
            temp = -tauq*SLC_DDOT(&(i32){n - i + 1}, &Q_(i, i), &int1, &A_(i, k + i), &lda);
            SLC_DAXPY(&(i32){n - i + 1}, &temp, &Q_(i, i), &int1, &A_(i, k + i), &lda);
            temp = A_(i, k + i);
            SLC_DLARTG(&temp, &alpha, &c, &s, &A_(i, k + i));
            SLC_DLARFG(&(i32){n - i + 1}, &A_(i, k + i), &A_(i, k + i + 1), &lda, &TAUL_(i));
            temp = A_(i, k + i);
            A_(i, k + i) = ONE;
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){n - i}, &ONE, &Q_(i, i + 1), &ldq, &Q_(i, i), &int1, &ZERO, &XQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &Q_(i, 1), &ldq, &Q_(i, i), &int1, &ZERO, dwork, &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XQ_(i + 1, 1), &ldxq, dwork, &int1, &ONE, &XQ_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){i - 1}, &(i32){n - i + 1}, &ONE, &A_(1, k + i), &lda, &Q_(i, i), &int1, &ZERO, &DW(nb + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XQ_(i + 1, nb1), &ldxq, &DW(nb + 1), &int1, &ONE, &XQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YQ_(i, 1), &ldyq, &Q_(i, i), &int1, &ZERO, &XQ_(1, i), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &XQ_(1, i), &int1, &ONE, &XQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YQ_(i, nb1), &ldyq, &Q_(i, i), &int1, &ZERO, &XQ_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &XQ_(1, i + nb), &int1, &ONE, &XQ_(i + 1, i), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-tauq}, &XQ_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &XQ_(i + 1, 1), &ldxq, &Q_(i, 1), &ldq, &ONE, &Q_(i, i + 1), &ldq);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XQ_(i + 1, nb1), &ldxq, &A_(1, k + i), &int1, &ONE, &Q_(i, i + 1), &ldq);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &YQ_(i, 1), &ldyq, &ONE, &Q_(i, i + 1), &ldq);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &YQ_(i, nb1), &ldyq, &ONE, &Q_(i, i + 1), &ldq);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){n - i + 1}, &ONE, &A_(i + 1, k + i), &lda, &Q_(i, i), &int1, &ZERO, &XA_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XA_(i + 1, 1), &ldxa, dwork, &int1, &ONE, &XA_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XA_(i + 1, nb1), &ldxa, &DW(nb + 1), &int1, &ONE, &XA_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YA_(k + i, 1), &ldya, &Q_(i, i), &int1, &ZERO, &XA_(1, i), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &XA_(1, i), &int1, &ONE, &XA_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YA_(k + i, nb1), &ldya, &Q_(i, i), &int1, &ZERO, &XA_(1, i), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &XA_(1, i), &int1, &ONE, &XA_(i + 1, i), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-tauq}, &XA_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &XA_(i + 1, 1), &ldxa, &Q_(i, 1), &ldq, &ONE, &A_(i + 1, k + i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XA_(i + 1, nb1), &ldxa, &A_(1, k + i), &int1, &ONE, &A_(i + 1, k + i), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &YA_(k + i, 1), &ldya, &ONE, &A_(i + 1, k + i), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &YA_(k + i, nb1), &ldya, &ONE, &A_(i + 1, k + i), &int1);
            SLC_DROT(&(i32){n - i}, &A_(i + 1, k + i), &int1, &Q_(i, i + 1), &ldq, &c, &s);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){n - i}, &ONE, &Q_(i, i + 1), &ldq, &A_(i, k + i), &lda, &ZERO, &XQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &A_(i, k + i + 1), &lda, &ZERO, &DW(nb2 + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &XQ_(i + 1, 1), &ldxq, &DW(nb2 + 1), &int1, &ONE, &XQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){i - 1}, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &A_(i, k + i + 1), &lda, &ZERO, &DW(nb3 + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XQ_(i + 1, nb1), &ldxq, &DW(nb3 + 1), &int1, &ONE, &XQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, 1), &ldyq, &A_(i, k + i + 1), &lda, &ZERO, &XQ_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &XQ_(1, i + nb), &int1, &ONE, &XQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, nb1), &ldyq, &A_(i, k + i + 1), &lda, &ZERO, &XQ_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &XQ_(1, i + nb), &int1, &ONE, &XQ_(i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-TAUL_(i)}, &XQ_(i + 1, i + nb), &int1);
            SLC_DAXPY(&(i32){n - i}, &ONE, &XQ_(i + 1, i + nb), &int1, &Q_(i, i + 1), &ldq);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){n - i + 1}, &ONE, &A_(i + 1, k + i), &lda, &A_(i, k + i), &lda, &ZERO, &XA_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &XA_(i + 1, 1), &ldxa, &DW(nb2 + 1), &int1, &ONE, &XA_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XA_(i + 1, nb1), &ldxa, &DW(nb3 + 1), &int1, &ONE, &XA_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YA_(k + i + 1, 1), &ldya, &A_(i, k + i + 1), &lda, &ZERO, &XA_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &XA_(1, i + nb), &int1, &ONE, &XA_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YA_(k + i + 1, nb1), &ldya, &A_(i, k + i + 1), &lda, &ZERO, &XA_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &XA_(1, i + nb), &int1, &ONE, &XA_(i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-TAUL_(i)}, &XA_(i + 1, i + nb), &int1);
            SLC_DAXPY(&(i32){n - i}, &ONE, &XA_(i + 1, i + nb), &int1, &A_(i + 1, k + i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){k + n}, &ONE, &G_(k + i, 1), &ldg, &Q_(i, i), &int1, &ZERO, &XG_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XG_(1, 1), &ldxg, dwork, &int1, &ONE, &XG_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XG_(1, nb1), &ldxg, &DW(nb + 1), &int1, &ONE, &XG_(1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YG_(k + i, 1), &ldyg, &Q_(i, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &DW(pdw), &int1, &ONE, &XG_(k + i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YG_(k + i, nb1), &ldyg, &Q_(i, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &DW(pdw), &int1, &ONE, &XG_(k + i + 1, i), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-tauq}, &XG_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &XG_(1, 1), &ldxg, &Q_(i, 1), &ldq, &ONE, &G_(k + i, 1), &ldg);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XG_(1, nb1), &ldxg, &A_(1, k + i), &int1, &ONE, &G_(k + i, 1), &ldg);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &YG_(k + i, 1), &ldyg, &ONE, &G_(k + i, k + i + 1), &ldg);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &YG_(k + i, nb1), &ldyg, &ONE, &G_(k + i, k + i + 1), &ldg);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){k + n}, &ONE, &B_(i, 1), &ldb, &Q_(i, i), &int1, &ZERO, &XB_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XB_(1, 1), &ldxb, dwork, &int1, &ONE, &XB_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XB_(1, nb1), &ldxb, &DW(nb + 1), &int1, &ONE, &XB_(1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YB_(i, 1), &ldyb, &Q_(i, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &DW(pdw), &int1, &ONE, &XB_(k + i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YB_(i, nb1), &ldyb, &Q_(i, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &DW(pdw), &int1, &ONE, &XB_(k + i + 1, i), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-tauq}, &XB_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &XB_(1, 1), &ldxb, &Q_(i, 1), &ldq, &ONE, &B_(i, 1), &ldb);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XB_(1, nb1), &ldxb, &A_(1, k + i), &int1, &ONE, &B_(i, 1), &ldb);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &YB_(i, 1), &ldyb, &ONE, &B_(i, k + i + 1), &ldb);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &YB_(i, nb1), &ldyb, &ONE, &B_(i, k + i + 1), &ldb);
            SLC_DROT(&(i32){k + n}, &G_(k + i, 1), &ldg, &B_(i, 1), &ldb, &c, &s);
            for (j = 1; j <= i - 1; j++) {
                YG_(k + i, j) = ZERO;
            }
            for (j = 1; j <= i - 1; j++) {
                YG_(k + i, nb + j) = ZERO;
            }
            for (j = 1; j <= i - 1; j++) {
                YA_(k + i, j) = ZERO;
            }
            for (j = 1; j <= i - 1; j++) {
                YA_(k + i, nb + j) = ZERO;
            }
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){k + n}, &ONE, &G_(k + i, 1), &ldg, &A_(i, k + i), &lda, &ZERO, &XG_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &XG_(1, 1), &ldxg, &DW(nb2 + 1), &int1, &ONE, &XG_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XG_(1, nb1), &ldxg, &DW(nb3 + 1), &int1, &ONE, &XG_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YG_(k + i + 1, 1), &ldyg, &A_(i, k + i + 1), &lda, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &DW(pdw), &int1, &ONE, &XG_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YG_(k + i + 1, nb1), &ldyg, &A_(i, k + i + 1), &lda, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &DW(pdw), &int1, &ONE, &XG_(k + i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-TAUL_(i)}, &XG_(1, i + nb), &int1);
            SLC_DAXPY(&(i32){k + n}, &ONE, &XG_(1, i + nb), &int1, &G_(k + i, 1), &ldg);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){k + n}, &ONE, &B_(i, 1), &ldb, &A_(i, k + i), &lda, &ZERO, &XB_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &XB_(1, 1), &ldxb, &DW(nb2 + 1), &int1, &ONE, &XB_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XB_(1, nb1), &ldxb, &DW(nb3 + 1), &int1, &ONE, &XB_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, 1), &ldyb, &A_(i, k + i + 1), &lda, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &DW(pdw), &int1, &ONE, &XB_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, nb1), &ldyb, &A_(i, k + i + 1), &lda, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &DW(pdw), &int1, &ONE, &XB_(k + i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-TAUL_(i)}, &XB_(1, i + nb), &int1);
            SLC_DAXPY(&(i32){k + n}, &ONE, &XB_(1, i + nb), &int1, &B_(i, 1), &ldb);
            A_(i, k + i) = temp;
            Q_(i, i) = tauq;
            CSL_(2*i - 1) = c;
            CSL_(2*i) = s;
            alpha = Q_(i, i + 1);
            SLC_DLARFG(&(i32){n - i}, &alpha, &Q_(i, i + 2), &ldq, &tauq);
            Q_(i, i + 1) = ONE;
            temp = -tauq*SLC_DDOT(&(i32){n - i}, &Q_(i, i + 1), &ldq, &B_(i, k + i + 1), &ldb);
            SLC_DAXPY(&(i32){n - i}, &temp, &Q_(i, i + 1), &ldq, &B_(i, k + i + 1), &ldb);
            temp = B_(i, k + i + 1);
            SLC_DLARTG(&temp, &alpha, &c, &s, &B_(i, k + i + 1));
            s = -s;
            SLC_DLARFG(&(i32){n - i}, &B_(i, k + i + 1), &B_(i, k + i + 2), &ldb, &TAUR_(i));
            temp = B_(i, k + i + 1);
            B_(i, k + i + 1) = ONE;
            SLC_DGEMV("N", &(i32){n - i}, &(i32){n - i}, &ONE, &B_(i + 1, k + i + 1), &ldb, &Q_(i, i + 1), &ldq, &ZERO, &YB_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XB_(k + i + 1, 1), &ldxb, &Q_(i, i + 1), &ldq, &ZERO, &YB_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &YB_(1, i), &int1, &ONE, &YB_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XB_(k + i + 1, nb1), &ldxb, &Q_(i, i + 1), &ldq, &ZERO, &YB_(1, i), &int1);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &YB_(1, i), &int1, &ONE, &YB_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &Q_(i, i + 1), &ldq, &ZERO, dwork, &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, 1), &ldyb, dwork, &int1, &ONE, &YB_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &Q_(i, i + 1), &ldq, &ZERO, &DW(nb + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, nb1), &ldyb, &DW(nb + 1), &int1, &ONE, &YB_(i + 1, i), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-tauq}, &YB_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &XB_(k + i + 1, 1), &ldxb, &ONE, &B_(i + 1, k + i + 1), &int1);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &XB_(k + i + 1, nb1), &ldxb, &ONE, &B_(i + 1, k + i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &YB_(i + 1, 1), &ldyb, &Q_(1, i + 1), &int1, &ONE, &B_(i + 1, k + i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, nb1), &ldyb, &B_(1, k + i + 1), &int1, &ONE, &B_(i + 1, k + i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){n - i}, &ONE, &Q_(i + 1, i + 1), &ldq, &Q_(i, i + 1), &ldq, &ZERO, &YQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XQ_(i + 1, 1), &ldxq, &Q_(i, i + 1), &ldq, &ZERO, &YQ_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &YQ_(1, i), &int1, &ONE, &YQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XQ_(i + 1, nb1), &ldxq, &Q_(i, i + 1), &ldq, &ZERO, &YQ_(1, i), &int1);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &YQ_(1, i), &int1, &ONE, &YQ_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, 1), &ldyq, dwork, &int1, &ONE, &YQ_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, nb1), &ldyq, &DW(nb + 1), &int1, &ONE, &YQ_(i + 1, i), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-tauq}, &YQ_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &XQ_(i + 1, 1), &ldxq, &ONE, &Q_(i + 1, i + 1), &int1);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &XQ_(i + 1, nb1), &ldxq, &ONE, &Q_(i + 1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &YQ_(i + 1, 1), &ldyq, &Q_(1, i + 1), &int1, &ONE, &Q_(i + 1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, nb1), &ldyq, &B_(1, k + i + 1), &int1, &ONE, &Q_(i + 1, i + 1), &int1);
            SLC_DROT(&(i32){n - i}, &Q_(i + 1, i + 1), &int1, &B_(i + 1, k + i + 1), &int1, &c, &s);
            for (j = 1; j <= i; j++) {
                XB_(k + i + 1, j) = ZERO;
            }
            for (j = 1; j <= i; j++) {
                XB_(k + i + 1, nb + j) = ZERO;
            }
            SLC_DGEMV("N", &(i32){n - i}, &(i32){n - i}, &ONE, &B_(i + 1, k + i + 1), &ldb, &B_(i, k + i + 1), &ldb, &ZERO, &YB_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XB_(k + i + 2, 1), &ldxb, &B_(i, k + i + 2), &ldb, &ZERO, &YB_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &YB_(1, i + nb), &int1, &ONE, &YB_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XB_(k + i + 2, nb1), &ldxb, &B_(i, k + i + 2), &ldb, &ZERO, &YB_(1, i + nb), &int1);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &YB_(1, i + nb), &int1, &ONE, &YB_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &i, &(i32){n - i - 1}, &ONE, &Q_(1, i + 2), &ldq, &B_(i, k + i + 2), &ldb, &ZERO, &DW(nb2 + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &YB_(i + 1, 1), &ldyb, &DW(nb2 + 1), &int1, &ONE, &YB_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){i - 1}, &(i32){n - i - 1}, &ONE, &B_(1, k + i + 2), &ldb, &B_(i, k + i + 2), &ldb, &ZERO, &DW(nb3 + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, nb1), &ldyb, &DW(nb3 + 1), &int1, &ONE, &YB_(i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-TAUR_(i)}, &YB_(i + 1, i + nb), &int1);
            SLC_DAXPY(&(i32){n - i}, &ONE, &YB_(i + 1, i + nb), &int1, &B_(i + 1, k + i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){n - i}, &ONE, &Q_(i + 1, i + 1), &ldq, &B_(i, k + i + 1), &ldb, &ZERO, &YQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XQ_(i + 2, 1), &ldxq, &B_(i, k + i + 2), &ldb, &ZERO, &YQ_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &YQ_(1, i + nb), &int1, &ONE, &YQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XQ_(i + 2, nb1), &ldxq, &B_(i, k + i + 2), &ldb, &ZERO, &YQ_(1, i + nb), &int1);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &YQ_(1, i + nb), &int1, &ONE, &YQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &YQ_(i + 1, 1), &ldyq, &DW(nb2 + 1), &int1, &ONE, &YQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, nb1), &ldyq, &DW(nb3 + 1), &int1, &ONE, &YQ_(i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-TAUR_(i)}, &YQ_(i + 1, i + nb), &int1);
            SLC_DAXPY(&(i32){n - i}, &ONE, &YQ_(i + 1, i + nb), &int1, &Q_(i + 1, i + 1), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){k + n}, &ONE, &A_(i + 1, 1), &lda, &Q_(i, i + 1), &ldq, &ZERO, &YA_(1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XA_(i + 1, 1), &ldxa, &Q_(i, i + 1), &ldq, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &DW(pdw), &int1, &ONE, &YA_(k + i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XA_(i + 1, nb1), &ldxa, &Q_(i, i + 1), &ldq, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &DW(pdw), &int1, &ONE, &YA_(k + i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YA_(1, 1), &ldya, dwork, &int1, &ONE, &YA_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YA_(1, nb1), &ldya, &DW(nb + 1), &int1, &ONE, &YA_(1, i), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-tauq}, &YA_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &XA_(i + 1, 1), &ldxa, &ONE, &A_(i + 1, k + i + 1), &lda);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &XA_(i + 1, nb1), &ldxa, &ONE, &A_(i + 1, k + i + 1), &lda);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &YA_(1, 1), &ldya, &Q_(1, i + 1), &int1, &ONE, &A_(i + 1, 1), &lda);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YA_(1, nb1), &ldya, &B_(1, k + i + 1), &int1, &ONE, &A_(i + 1, 1), &lda);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){n - i}, &ONE, &G_(1, k + i + 1), &ldg, &Q_(i, i + 1), &ldq, &ZERO, &YG_(1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XG_(k + i + 1, 1), &ldxg, &Q_(i, i + 1), &ldq, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &DW(pdw), &int1, &ONE, &YG_(k + i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XG_(k + i + 1, nb1), &ldxg, &Q_(i, i + 1), &ldq, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &DW(pdw), &int1, &ONE, &YG_(k + i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YG_(1, 1), &ldyg, dwork, &int1, &ONE, &YG_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YG_(1, nb1), &ldyg, &DW(nb + 1), &int1, &ONE, &YG_(1, i), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-tauq}, &YG_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &XG_(k + i + 1, 1), &ldxg, &ONE, &G_(k + i + 1, k + i + 1), &int1);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &XG_(k + i + 1, nb1), &ldxg, &ONE, &G_(k + i + 1, k + i + 1), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &YG_(1, 1), &ldyg, &Q_(1, i + 1), &int1, &ONE, &G_(1, k + i + 1), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YG_(1, nb1), &ldyg, &B_(1, k + i + 1), &int1, &ONE, &G_(1, k + i + 1), &int1);
            for (j = 1; j <= i; j++) {
                XG_(k + i + 1, j) = ZERO;
            }
            for (j = 1; j <= i; j++) {
                XG_(k + i + 1, nb + j) = ZERO;
            }
            SLC_DROT(&(i32){k + n}, &A_(i + 1, 1), &lda, &G_(1, k + i + 1), &int1, &c, &s);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){k + n}, &ONE, &A_(i + 1, 1), &lda, &B_(i, k + i + 1), &ldb, &ZERO, &YA_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XA_(i + 2, 1), &ldxa, &B_(i, k + i + 2), &ldb, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &DW(pdw), &int1, &ONE, &YA_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XA_(i + 2, nb1), &ldxa, &B_(i, k + i + 2), &ldb, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &DW(pdw), &int1, &ONE, &YA_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &YA_(1, 1), &ldya, &DW(nb2 + 1), &int1, &ONE, &YA_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YA_(1, nb1), &ldya, &DW(nb3 + 1), &int1, &ONE, &YA_(1, i + nb), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-TAUR_(i)}, &YA_(1, i + nb), &int1);
            SLC_DAXPY(&(i32){k + n}, &ONE, &YA_(1, i + nb), &int1, &A_(i + 1, 1), &lda);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){n - i}, &ONE, &G_(1, k + i + 1), &ldg, &B_(i, k + i + 1), &ldb, &ZERO, &YG_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XG_(k + i + 2, 1), &ldxg, &B_(i, k + i + 2), &ldb, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &DW(pdw), &int1, &ONE, &YG_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XG_(k + i + 2, nb1), &ldxg, &B_(i, k + i + 2), &ldb, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &i, &(i32){n - i}, &ONE, &A_(1, k + i + 1), &lda, &DW(pdw), &int1, &ONE, &YG_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &YG_(1, 1), &ldyg, &DW(nb2 + 1), &int1, &ONE, &YG_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YG_(1, nb1), &ldyg, &DW(nb3 + 1), &int1, &ONE, &YG_(1, i + nb), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-TAUR_(i)}, &YG_(1, i + nb), &int1);
            SLC_DAXPY(&(i32){k + n}, &ONE, &YG_(1, i + nb), &int1, &G_(1, k + i + 1), &int1);
            B_(i, k + i + 1) = temp;
            Q_(i, i + 1) = tauq;
            CSR_(2*i - 1) = c;
            CSR_(2*i) = s;
        }
    } else if (ltrb) {
        for (i = 1; i <= nb; i++) {
            alpha = Q_(i, i);
            SLC_DLARFG(&(i32){n - i + 1}, &alpha, &Q_(i + 1, i), &int1, &tauq);
            Q_(i, i) = ONE;
            temp = -tauq*SLC_DDOT(&(i32){n - i + 1}, &Q_(i, i), &int1, &A_(k + i, i), &int1);
            SLC_DAXPY(&(i32){n - i + 1}, &temp, &Q_(i, i), &int1, &A_(k + i, i), &int1);
            temp = A_(k + i, i);
            SLC_DLARTG(&temp, &alpha, &c, &s, &A_(k + i, i));
            SLC_DLARFG(&(i32){n - i + 1}, &A_(k + i, i), &A_(k + i + 1, i), &int1, &TAUL_(i));
            temp = A_(k + i, i);
            A_(k + i, i) = ONE;
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){n - i}, &ONE, &Q_(i, i + 1), &ldq, &Q_(i, i), &int1, &ZERO, &XQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &Q_(i, 1), &ldq, &Q_(i, i), &int1, &ZERO, dwork, &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XQ_(i + 1, 1), &ldxq, dwork, &int1, &ONE, &XQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &A_(k + i, 1), &lda, &Q_(i, i), &int1, &ZERO, &DW(nb + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XQ_(i + 1, nb1), &ldxq, &DW(nb + 1), &int1, &ONE, &XQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YQ_(i, 1), &ldyq, &Q_(i, i), &int1, &ZERO, &XQ_(1, i), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &XQ_(1, i), &int1, &ONE, &XQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YQ_(i, nb1), &ldyq, &Q_(i, i), &int1, &ZERO, &XQ_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &XQ_(1, i + nb), &int1, &ONE, &XQ_(i + 1, i), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-tauq}, &XQ_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &XQ_(i + 1, 1), &ldxq, &Q_(i, 1), &ldq, &ONE, &Q_(i, i + 1), &ldq);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XQ_(i + 1, nb1), &ldxq, &A_(k + i, 1), &lda, &ONE, &Q_(i, i + 1), &ldq);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &YQ_(i, 1), &ldyq, &ONE, &Q_(i, i + 1), &ldq);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &YQ_(i, nb1), &ldyq, &ONE, &Q_(i, i + 1), &ldq);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){n - i}, &ONE, &A_(k + i, i + 1), &lda, &Q_(i, i), &int1, &ZERO, &XA_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XA_(i + 1, 1), &ldxa, dwork, &int1, &ONE, &XA_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XA_(i + 1, nb1), &ldxa, &DW(nb + 1), &int1, &ONE, &XA_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YA_(k + i, 1), &ldya, &Q_(i, i), &int1, &ZERO, &XA_(1, i), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &XA_(1, i), &int1, &ONE, &XA_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YA_(k + i, nb1), &ldya, &Q_(i, i), &int1, &ZERO, &XA_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &XA_(1, i), &int1, &ONE, &XA_(i + 1, i), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-tauq}, &XA_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &XA_(i + 1, 1), &ldxa, &Q_(i, 1), &ldq, &ONE, &A_(k + i, i + 1), &lda);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XA_(i + 1, nb1), &ldxa, &A_(k + i, 1), &lda, &ONE, &A_(k + i, i + 1), &lda);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &YA_(k + i, 1), &ldya, &ONE, &A_(k + i, i + 1), &lda);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &YA_(k + i, nb1), &ldya, &ONE, &A_(k + i, i + 1), &lda);
            SLC_DROT(&(i32){n - i}, &A_(k + i, i + 1), &lda, &Q_(i, i + 1), &ldq, &c, &s);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){n - i}, &ONE, &Q_(i, i + 1), &ldq, &A_(k + i, i), &int1, &ZERO, &XQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &A_(k + i + 1, i), &int1, &ZERO, &DW(nb2 + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &XQ_(i + 1, 1), &ldxq, &DW(nb2 + 1), &int1, &ONE, &XQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &A_(k + i + 1, 1), &lda, &A_(k + i + 1, i), &int1, &ZERO, &DW(nb3 + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XQ_(i + 1, nb1), &ldxq, &DW(nb3 + 1), &int1, &ONE, &XQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, 1), &ldyq, &A_(k + i + 1, i), &int1, &ZERO, &XQ_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &XQ_(1, i + nb), &int1, &ONE, &XQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, nb1), &ldyq, &A_(k + i + 1, i), &int1, &ZERO, &XQ_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &XQ_(1, i + nb), &int1, &ONE, &XQ_(i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-TAUL_(i)}, &XQ_(i + 1, i + nb), &int1);
            SLC_DAXPY(&(i32){n - i}, &ONE, &XQ_(i + 1, i + nb), &int1, &Q_(i, i + 1), &ldq);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){n - i}, &ONE, &A_(k + i, i + 1), &lda, &A_(k + i, i), &int1, &ZERO, &XA_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &XA_(i + 1, 1), &ldxa, &DW(nb2 + 1), &int1, &ONE, &XA_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XA_(i + 1, nb1), &ldxa, &DW(nb3 + 1), &int1, &ONE, &XA_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YA_(k + i + 1, 1), &ldya, &A_(k + i + 1, i), &int1, &ZERO, &XA_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &XA_(1, i + nb), &int1, &ONE, &XA_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YA_(k + i + 1, nb1), &ldya, &A_(k + i + 1, i), &int1, &ZERO, &XA_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &XA_(1, i + nb), &int1, &ONE, &XA_(i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-TAUL_(i)}, &XA_(i + 1, i + nb), &int1);
            SLC_DAXPY(&(i32){n - i}, &ONE, &XA_(i + 1, i + nb), &int1, &A_(k + i, i + 1), &lda);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){k + n}, &ONE, &G_(k + i, 1), &ldg, &Q_(i, i), &int1, &ZERO, &XG_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XG_(1, 1), &ldxg, dwork, &int1, &ONE, &XG_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XG_(1, nb1), &ldxg, &DW(nb + 1), &int1, &ONE, &XG_(1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YG_(k + i, 1), &ldyg, &Q_(i, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &DW(pdw), &int1, &ONE, &XG_(k + i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YG_(k + i, nb1), &ldyg, &Q_(i, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &DW(pdw), &int1, &ONE, &XG_(k + i + 1, i), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-tauq}, &XG_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &XG_(1, 1), &ldxg, &Q_(i, 1), &ldq, &ONE, &G_(k + i, 1), &ldg);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XG_(1, nb1), &ldxg, &A_(k + i, 1), &lda, &ONE, &G_(k + i, 1), &ldg);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &YG_(k + i, 1), &ldyg, &ONE, &G_(k + i, k + i + 1), &ldg);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &YG_(k + i, nb1), &ldyg, &ONE, &G_(k + i, k + i + 1), &ldg);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){n - i + 1}, &ONE, &B_(1, i), &ldb, &Q_(i, i), &int1, &ZERO, &XB_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XB_(1, 1), &ldxb, dwork, &int1, &ONE, &XB_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XB_(1, nb1), &ldxb, &DW(nb + 1), &int1, &ONE, &XB_(1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YB_(i, 1), &ldyb, &Q_(i, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &DW(pdw), &int1, &ONE, &XB_(k + i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YB_(i, nb1), &ldyb, &Q_(i, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &DW(pdw), &int1, &ONE, &XB_(k + i + 1, i), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-tauq}, &XB_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &XB_(1, 1), &ldxb, &Q_(i, 1), &ldq, &ONE, &B_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XB_(1, nb1), &ldxb, &A_(k + i, 1), &lda, &ONE, &B_(1, i), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &YB_(i, 1), &ldyb, &ONE, &B_(k + i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &YB_(i, nb1), &ldyb, &ONE, &B_(k + i + 1, i), &int1);
            SLC_DROT(&(i32){k + n}, &G_(k + i, 1), &ldg, &B_(1, i), &int1, &c, &s);
            for (j = 1; j <= i - 1; j++) {
                YG_(k + i, j) = ZERO;
            }
            for (j = 1; j <= i - 1; j++) {
                YG_(k + i, nb + j) = ZERO;
            }
            for (j = 1; j <= i - 1; j++) {
                YA_(k + i, j) = ZERO;
            }
            for (j = 1; j <= i - 1; j++) {
                YA_(k + i, nb + j) = ZERO;
            }
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){k + n}, &ONE, &G_(k + i, 1), &ldg, &A_(k + i, i), &int1, &ZERO, &XG_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &XG_(1, 1), &ldxg, &DW(nb2 + 1), &int1, &ONE, &XG_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XG_(1, nb1), &ldxg, &DW(nb3 + 1), &int1, &ONE, &XG_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YG_(k + i + 1, 1), &ldyg, &A_(k + i + 1, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &DW(pdw), &int1, &ONE, &XG_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YG_(k + i + 1, nb1), &ldyg, &A_(k + i + 1, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &DW(pdw), &int1, &ONE, &XG_(k + i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-TAUL_(i)}, &XG_(1, i + nb), &int1);
            SLC_DAXPY(&(i32){k + n}, &ONE, &XG_(1, i + nb), &int1, &G_(k + i, 1), &ldg);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){n - i + 1}, &ONE, &B_(1, i), &ldb, &A_(k + i, i), &int1, &ZERO, &XB_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &XB_(1, 1), &ldxb, &DW(nb2 + 1), &int1, &ONE, &XB_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XB_(1, nb1), &ldxb, &DW(nb3 + 1), &int1, &ONE, &XB_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, 1), &ldyb, &A_(k + i + 1, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &DW(pdw), &int1, &ONE, &XB_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, nb1), &ldyb, &A_(k + i + 1, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &DW(pdw), &int1, &ONE, &XB_(k + i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-TAUL_(i)}, &XB_(1, i + nb), &int1);
            SLC_DAXPY(&(i32){k + n}, &ONE, &XB_(1, i + nb), &int1, &B_(1, i), &int1);
            A_(k + i, i) = temp;
            Q_(i, i) = tauq;
            CSL_(2*i - 1) = c;
            CSL_(2*i) = s;
            alpha = Q_(i, i + 1);
            SLC_DLARFG(&(i32){n - i}, &alpha, &Q_(i, i + 2), &ldq, &tauq);
            Q_(i, i + 1) = ONE;
            temp = -tauq*SLC_DDOT(&(i32){n - i}, &Q_(i, i + 1), &ldq, &B_(k + i + 1, i), &int1);
            SLC_DAXPY(&(i32){n - i}, &temp, &Q_(i, i + 1), &ldq, &B_(k + i + 1, i), &int1);
            temp = B_(k + i + 1, i);
            SLC_DLARTG(&temp, &alpha, &c, &s, &B_(k + i + 1, i));
            s = -s;
            SLC_DLARFG(&(i32){n - i}, &B_(k + i + 1, i), &B_(k + i + 2, i), &int1, &TAUR_(i));
            temp = B_(k + i + 1, i);
            B_(k + i + 1, i) = ONE;
            SLC_DGEMV("T", &(i32){n - i}, &(i32){n - i}, &ONE, &B_(k + i + 1, i + 1), &ldb, &Q_(i, i + 1), &ldq, &ZERO, &YB_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XB_(k + i + 1, 1), &ldxb, &Q_(i, i + 1), &ldq, &ZERO, &YB_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &YB_(1, i), &int1, &ONE, &YB_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XB_(k + i + 1, nb1), &ldxb, &Q_(i, i + 1), &ldq, &ZERO, &YB_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &YB_(1, i), &int1, &ONE, &YB_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &Q_(i, i + 1), &ldq, &ZERO, dwork, &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, 1), &ldyb, dwork, &int1, &ONE, &YB_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &B_(k + i + 1, 1), &ldb, &Q_(i, i + 1), &ldq, &ZERO, &DW(nb + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, nb1), &ldyb, &DW(nb + 1), &int1, &ONE, &YB_(i + 1, i), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-tauq}, &YB_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &XB_(k + i + 1, 1), &ldxb, &ONE, &B_(k + i + 1, i + 1), &ldb);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &XB_(k + i + 1, nb1), &ldxb, &ONE, &B_(k + i + 1, i + 1), &ldb);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &YB_(i + 1, 1), &ldyb, &Q_(1, i + 1), &int1, &ONE, &B_(k + i + 1, i + 1), &ldb);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, nb1), &ldyb, &B_(k + i + 1, 1), &ldb, &ONE, &B_(k + i + 1, i + 1), &ldb);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){n - i}, &ONE, &Q_(i + 1, i + 1), &ldq, &Q_(i, i + 1), &ldq, &ZERO, &YQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XQ_(i + 1, 1), &ldxq, &Q_(i, i + 1), &ldq, &ZERO, &YQ_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &YQ_(1, i), &int1, &ONE, &YQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XQ_(i + 1, nb1), &ldxq, &Q_(i, i + 1), &ldq, &ZERO, &YQ_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &YQ_(1, i), &int1, &ONE, &YQ_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, 1), &ldyq, dwork, &int1, &ONE, &YQ_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, nb1), &ldyq, &DW(nb + 1), &int1, &ONE, &YQ_(i + 1, i), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-tauq}, &YQ_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &XQ_(i + 1, 1), &ldxq, &ONE, &Q_(i + 1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &XQ_(i + 1, nb1), &ldxq, &ONE, &Q_(i + 1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &YQ_(i + 1, 1), &ldyq, &Q_(1, i + 1), &int1, &ONE, &Q_(i + 1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, nb1), &ldyq, &B_(k + i + 1, 1), &ldb, &ONE, &Q_(i + 1, i + 1), &int1);
            SLC_DROT(&(i32){n - i}, &Q_(i + 1, i + 1), &int1, &B_(k + i + 1, i + 1), &ldb, &c, &s);
            for (j = 1; j <= i; j++) {
                XB_(k + i + 1, j) = ZERO;
            }
            for (j = 1; j <= i; j++) {
                XB_(k + i + 1, nb + j) = ZERO;
            }
            SLC_DGEMV("T", &(i32){n - i}, &(i32){n - i}, &ONE, &B_(k + i + 1, i + 1), &ldb, &B_(k + i + 1, i), &int1, &ZERO, &YB_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XB_(k + i + 2, 1), &ldxb, &B_(k + i + 2, i), &int1, &ZERO, &YB_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &YB_(1, i + nb), &int1, &ONE, &YB_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XB_(k + i + 2, nb1), &ldxb, &B_(k + i + 2, i), &int1, &ZERO, &YB_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &YB_(1, i + nb), &int1, &ONE, &YB_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &i, &(i32){n - i - 1}, &ONE, &Q_(1, i + 2), &ldq, &B_(k + i + 2, i), &int1, &ZERO, &DW(nb2 + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &YB_(i + 1, 1), &ldyb, &DW(nb2 + 1), &int1, &ONE, &YB_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &(i32){i - 1}, &ONE, &B_(k + i + 2, 1), &ldb, &B_(k + i + 2, i), &int1, &ZERO, &DW(nb3 + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, nb1), &ldyb, &DW(nb3 + 1), &int1, &ONE, &YB_(i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-TAUR_(i)}, &YB_(i + 1, i + nb), &int1);
            SLC_DAXPY(&(i32){n - i}, &ONE, &YB_(i + 1, i + nb), &int1, &B_(k + i + 1, i + 1), &ldb);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){n - i}, &ONE, &Q_(i + 1, i + 1), &ldq, &B_(k + i + 1, i), &int1, &ZERO, &YQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XQ_(i + 2, 1), &ldxq, &B_(k + i + 2, i), &int1, &ZERO, &YQ_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &YQ_(1, i + nb), &int1, &ONE, &YQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XQ_(i + 2, nb1), &ldxq, &B_(k + i + 2, i), &int1, &ZERO, &YQ_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &YQ_(1, i + nb), &int1, &ONE, &YQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &YQ_(i + 1, 1), &ldyq, &DW(nb2 + 1), &int1, &ONE, &YQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, nb1), &ldyq, &DW(nb3 + 1), &int1, &ONE, &YQ_(i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-TAUR_(i)}, &YQ_(i + 1, i + nb), &int1);
            SLC_DAXPY(&(i32){n - i}, &ONE, &YQ_(i + 1, i + nb), &int1, &Q_(i + 1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){n - i}, &ONE, &A_(1, i + 1), &lda, &Q_(i, i + 1), &ldq, &ZERO, &YA_(1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XA_(i + 1, 1), &ldxa, &Q_(i, i + 1), &ldq, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &DW(pdw), &int1, &ONE, &YA_(k + i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XA_(i + 1, nb1), &ldxa, &Q_(i, i + 1), &ldq, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &DW(pdw), &int1, &ONE, &YA_(k + i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YA_(1, 1), &ldya, dwork, &int1, &ONE, &YA_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YA_(1, nb1), &ldya, &DW(nb + 1), &int1, &ONE, &YA_(1, i), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-tauq}, &YA_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &XA_(i + 1, 1), &ldxa, &ONE, &A_(k + i + 1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &XA_(i + 1, nb1), &ldxa, &ONE, &A_(k + i + 1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &YA_(1, 1), &ldya, &Q_(1, i + 1), &int1, &ONE, &A_(1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YA_(1, nb1), &ldya, &B_(k + i + 1, 1), &ldb, &ONE, &A_(1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){n - i}, &ONE, &G_(1, k + i + 1), &ldg, &Q_(i, i + 1), &ldq, &ZERO, &YG_(1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XG_(k + i + 1, 1), &ldxg, &Q_(i, i + 1), &ldq, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &DW(pdw), &int1, &ONE, &YG_(k + i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XG_(k + i + 1, nb1), &ldxg, &Q_(i, i + 1), &ldq, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &DW(pdw), &int1, &ONE, &YG_(k + i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YG_(1, 1), &ldyg, dwork, &int1, &ONE, &YG_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YG_(1, nb1), &ldyg, &DW(nb + 1), &int1, &ONE, &YG_(1, i), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-tauq}, &YG_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &XG_(k + i + 1, 1), &ldxg, &ONE, &G_(k + i + 1, k + i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &XG_(k + i + 1, nb1), &ldxg, &ONE, &G_(k + i + 1, k + i + 1), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &YG_(1, 1), &ldyg, &Q_(1, i + 1), &int1, &ONE, &G_(1, k + i + 1), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YG_(1, nb1), &ldyg, &B_(k + i + 1, 1), &ldb, &ONE, &G_(1, k + i + 1), &int1);
            for (j = 1; j <= i; j++) {
                XG_(k + i + 1, j) = ZERO;
            }
            for (j = 1; j <= i; j++) {
                XG_(k + i + 1, nb + j) = ZERO;
            }
            SLC_DROT(&(i32){k + n}, &A_(1, i + 1), &int1, &G_(1, k + i + 1), &int1, &c, &s);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){n - i}, &ONE, &A_(1, i + 1), &lda, &B_(k + i + 1, i), &int1, &ZERO, &YA_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XA_(i + 2, 1), &ldxa, &B_(k + i + 2, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &DW(pdw), &int1, &ONE, &YA_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XA_(i + 2, nb1), &ldxa, &B_(k + i + 2, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &DW(pdw), &int1, &ONE, &YA_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &YA_(1, 1), &ldya, &DW(nb2 + 1), &int1, &ONE, &YA_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YA_(1, nb1), &ldya, &DW(nb3 + 1), &int1, &ONE, &YA_(1, i + nb), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-TAUR_(i)}, &YA_(1, i + nb), &int1);
            SLC_DAXPY(&(i32){k + n}, &ONE, &YA_(1, i + nb), &int1, &A_(1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){n - i}, &ONE, &G_(1, k + i + 1), &ldg, &B_(k + i + 1, i), &int1, &ZERO, &YG_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XG_(k + i + 2, 1), &ldxg, &B_(k + i + 2, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &DW(pdw), &int1, &ONE, &YG_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XG_(k + i + 2, nb1), &ldxg, &B_(k + i + 2, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &DW(pdw), &int1, &ONE, &YG_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &YG_(1, 1), &ldyg, &DW(nb2 + 1), &int1, &ONE, &YG_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YG_(1, nb1), &ldyg, &DW(nb3 + 1), &int1, &ONE, &YG_(1, i + nb), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-TAUR_(i)}, &YG_(1, i + nb), &int1);
            SLC_DAXPY(&(i32){k + n}, &ONE, &YG_(1, i + nb), &int1, &G_(1, k + i + 1), &int1);
            B_(k + i + 1, i) = temp;
            Q_(i, i + 1) = tauq;
            CSR_(2*i - 1) = c;
            CSR_(2*i) = s;
        }
    } else {
        for (i = 1; i <= nb; i++) {
            alpha = Q_(i, i);
            SLC_DLARFG(&(i32){n - i + 1}, &alpha, &Q_(i + 1, i), &int1, &tauq);
            Q_(i, i) = ONE;
            temp = -tauq*SLC_DDOT(&(i32){n - i + 1}, &Q_(i, i), &int1, &A_(k + i, i), &int1);
            SLC_DAXPY(&(i32){n - i + 1}, &temp, &Q_(i, i), &int1, &A_(k + i, i), &int1);
            temp = A_(k + i, i);
            SLC_DLARTG(&temp, &alpha, &c, &s, &A_(k + i, i));
            SLC_DLARFG(&(i32){n - i + 1}, &A_(k + i, i), &A_(k + i + 1, i), &int1, &TAUL_(i));
            temp = A_(k + i, i);
            A_(k + i, i) = ONE;
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){n - i}, &ONE, &Q_(i, i + 1), &ldq, &Q_(i, i), &int1, &ZERO, &XQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &Q_(i, 1), &ldq, &Q_(i, i), &int1, &ZERO, dwork, &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XQ_(i + 1, 1), &ldxq, dwork, &int1, &ONE, &XQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &A_(k + i, 1), &lda, &Q_(i, i), &int1, &ZERO, &DW(nb + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XQ_(i + 1, nb1), &ldxq, &DW(nb + 1), &int1, &ONE, &XQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YQ_(i, 1), &ldyq, &Q_(i, i), &int1, &ZERO, &XQ_(1, i), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &XQ_(1, i), &int1, &ONE, &XQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YQ_(i, nb1), &ldyq, &Q_(i, i), &int1, &ZERO, &XQ_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &XQ_(1, i + nb), &int1, &ONE, &XQ_(i + 1, i), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-tauq}, &XQ_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &XQ_(i + 1, 1), &ldxq, &Q_(i, 1), &ldq, &ONE, &Q_(i, i + 1), &ldq);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XQ_(i + 1, nb1), &ldxq, &A_(k + i, 1), &lda, &ONE, &Q_(i, i + 1), &ldq);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &YQ_(i, 1), &ldyq, &ONE, &Q_(i, i + 1), &ldq);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &YQ_(i, nb1), &ldyq, &ONE, &Q_(i, i + 1), &ldq);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){n - i}, &ONE, &A_(k + i, i + 1), &lda, &Q_(i, i), &int1, &ZERO, &XA_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XA_(i + 1, 1), &ldxa, dwork, &int1, &ONE, &XA_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XA_(i + 1, nb1), &ldxa, &DW(nb + 1), &int1, &ONE, &XA_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YA_(k + i, 1), &ldya, &Q_(i, i), &int1, &ZERO, &XA_(1, i), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &XA_(1, i), &int1, &ONE, &XA_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YA_(k + i, nb1), &ldya, &Q_(i, i), &int1, &ZERO, &XA_(1, i), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &XA_(1, i), &int1, &ONE, &XA_(i + 1, i), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-tauq}, &XA_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &XA_(i + 1, 1), &ldxa, &Q_(i, 1), &ldq, &ONE, &A_(k + i, i + 1), &lda);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XA_(i + 1, nb1), &ldxa, &A_(k + i, 1), &lda, &ONE, &A_(k + i, i + 1), &lda);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &YA_(k + i, 1), &ldya, &ONE, &A_(k + i, i + 1), &lda);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &YA_(k + i, nb1), &ldya, &ONE, &A_(k + i, i + 1), &lda);
            SLC_DROT(&(i32){n - i}, &A_(k + i, i + 1), &lda, &Q_(i, i + 1), &ldq, &c, &s);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){n - i}, &ONE, &Q_(i, i + 1), &ldq, &A_(k + i, i), &int1, &ZERO, &XQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &A_(k + i + 1, i), &int1, &ZERO, &DW(nb2 + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &XQ_(i + 1, 1), &ldxq, &DW(nb2 + 1), &int1, &ONE, &XQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &A_(k + i + 1, 1), &lda, &A_(k + i + 1, i), &int1, &ZERO, &DW(nb3 + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XQ_(i + 1, nb1), &ldxq, &DW(nb3 + 1), &int1, &ONE, &XQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, 1), &ldyq, &A_(k + i + 1, i), &int1, &ZERO, &XQ_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &XQ_(1, i + nb), &int1, &ONE, &XQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, nb1), &ldyq, &A_(k + i + 1, i), &int1, &ZERO, &XQ_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &XQ_(1, i + nb), &int1, &ONE, &XQ_(i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-TAUL_(i)}, &XQ_(i + 1, i + nb), &int1);
            SLC_DAXPY(&(i32){n - i}, &ONE, &XQ_(i + 1, i + nb), &int1, &Q_(i, i + 1), &ldq);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){n - i}, &ONE, &A_(k + i, i + 1), &lda, &A_(k + i, i), &int1, &ZERO, &XA_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &XA_(i + 1, 1), &ldxa, &DW(nb2 + 1), &int1, &ONE, &XA_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &XA_(i + 1, nb1), &ldxa, &DW(nb3 + 1), &int1, &ONE, &XA_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YA_(k + i + 1, 1), &ldya, &A_(k + i + 1, i), &int1, &ZERO, &XA_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &XA_(1, i + nb), &int1, &ONE, &XA_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YA_(k + i + 1, nb1), &ldya, &A_(k + i + 1, i), &int1, &ZERO, &XA_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &XA_(1, i + nb), &int1, &ONE, &XA_(i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-TAUL_(i)}, &XA_(i + 1, i + nb), &int1);
            SLC_DAXPY(&(i32){n - i}, &ONE, &XA_(i + 1, i + nb), &int1, &A_(k + i, i + 1), &lda);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){k + n}, &ONE, &G_(k + i, 1), &ldg, &Q_(i, i), &int1, &ZERO, &XG_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XG_(1, 1), &ldxg, dwork, &int1, &ONE, &XG_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XG_(1, nb1), &ldxg, &DW(nb + 1), &int1, &ONE, &XG_(1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YG_(k + i, 1), &ldyg, &Q_(i, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &DW(pdw), &int1, &ONE, &XG_(k + i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YG_(k + i, nb1), &ldyg, &Q_(i, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &DW(pdw), &int1, &ONE, &XG_(k + i + 1, i), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-tauq}, &XG_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &XG_(1, 1), &ldxg, &Q_(i, 1), &ldq, &ONE, &G_(k + i, 1), &ldg);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XG_(1, nb1), &ldxg, &A_(k + i, 1), &lda, &ONE, &G_(k + i, 1), &ldg);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &YG_(k + i, 1), &ldyg, &ONE, &G_(k + i, k + i + 1), &ldg);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &YG_(k + i, nb1), &ldyg, &ONE, &G_(k + i, k + i + 1), &ldg);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){k + n}, &ONE, &B_(i, 1), &ldb, &Q_(i, i), &int1, &ZERO, &XB_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XB_(1, 1), &ldxb, dwork, &int1, &ONE, &XB_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XB_(1, nb1), &ldxb, &DW(nb + 1), &int1, &ONE, &XB_(1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YB_(i, 1), &ldyb, &Q_(i, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &DW(pdw), &int1, &ONE, &XB_(k + i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){i - 1}, &ONE, &YB_(i, nb1), &ldyb, &Q_(i, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &DW(pdw), &int1, &ONE, &XB_(k + i + 1, i), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-tauq}, &XB_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &XB_(1, 1), &ldxb, &Q_(i, 1), &ldq, &ONE, &B_(i, 1), &ldb);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XB_(1, nb1), &ldxb, &A_(k + i, 1), &lda, &ONE, &B_(i, 1), &ldb);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &YB_(i, 1), &ldyb, &ONE, &B_(i, k + i + 1), &ldb);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &YB_(i, nb1), &ldyb, &ONE, &B_(i, k + i + 1), &ldb);
            SLC_DROT(&(i32){k + n}, &G_(k + i, 1), &ldg, &B_(i, 1), &ldb, &c, &s);
            for (j = 1; j <= i - 1; j++) {
                YG_(k + i, j) = ZERO;
            }
            for (j = 1; j <= i - 1; j++) {
                YG_(k + i, nb + j) = ZERO;
            }
            for (j = 1; j <= i - 1; j++) {
                YA_(k + i, j) = ZERO;
            }
            for (j = 1; j <= i - 1; j++) {
                YA_(k + i, nb + j) = ZERO;
            }
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){k + n}, &ONE, &G_(k + i, 1), &ldg, &A_(k + i, i), &int1, &ZERO, &XG_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &XG_(1, 1), &ldxg, &DW(nb2 + 1), &int1, &ONE, &XG_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XG_(1, nb1), &ldxg, &DW(nb3 + 1), &int1, &ONE, &XG_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YG_(k + i + 1, 1), &ldyg, &A_(k + i + 1, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &DW(pdw), &int1, &ONE, &XG_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YG_(k + i + 1, nb1), &ldyg, &A_(k + i + 1, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &DW(pdw), &int1, &ONE, &XG_(k + i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-TAUL_(i)}, &XG_(1, i + nb), &int1);
            SLC_DAXPY(&(i32){k + n}, &ONE, &XG_(1, i + nb), &int1, &G_(k + i, 1), &ldg);
            SLC_DGEMV("T", &(i32){n - i + 1}, &(i32){k + n}, &ONE, &B_(i, 1), &ldb, &A_(k + i, i), &int1, &ZERO, &XB_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &XB_(1, 1), &ldxb, &DW(nb2 + 1), &int1, &ONE, &XB_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &XB_(1, nb1), &ldxb, &DW(nb3 + 1), &int1, &ONE, &XB_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, 1), &ldyb, &A_(k + i + 1, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &DW(pdw), &int1, &ONE, &XB_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, nb1), &ldyb, &A_(k + i + 1, i), &int1, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("T", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &DW(pdw), &int1, &ONE, &XB_(k + i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-TAUL_(i)}, &XB_(1, i + nb), &int1);
            SLC_DAXPY(&(i32){k + n}, &ONE, &XB_(1, i + nb), &int1, &B_(i, 1), &ldb);
            A_(k + i, i) = temp;
            Q_(i, i) = tauq;
            CSL_(2*i - 1) = c;
            CSL_(2*i) = s;
            alpha = Q_(i, i + 1);
            SLC_DLARFG(&(i32){n - i}, &alpha, &Q_(i, i + 2), &ldq, &tauq);
            Q_(i, i + 1) = ONE;
            temp = -tauq*SLC_DDOT(&(i32){n - i}, &Q_(i, i + 1), &ldq, &B_(i, k + i + 1), &ldb);
            SLC_DAXPY(&(i32){n - i}, &temp, &Q_(i, i + 1), &ldq, &B_(i, k + i + 1), &ldb);
            temp = B_(i, k + i + 1);
            SLC_DLARTG(&temp, &alpha, &c, &s, &B_(i, k + i + 1));
            s = -s;
            SLC_DLARFG(&(i32){n - i}, &B_(i, k + i + 1), &B_(i, k + i + 2), &ldb, &TAUR_(i));
            temp = B_(i, k + i + 1);
            B_(i, k + i + 1) = ONE;
            SLC_DGEMV("N", &(i32){n - i}, &(i32){n - i}, &ONE, &B_(i + 1, k + i + 1), &ldb, &Q_(i, i + 1), &ldq, &ZERO, &YB_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XB_(k + i + 1, 1), &ldxb, &Q_(i, i + 1), &ldq, &ZERO, &YB_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &YB_(1, i), &int1, &ONE, &YB_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XB_(k + i + 1, nb1), &ldxb, &Q_(i, i + 1), &ldq, &ZERO, &YB_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &YB_(1, i), &int1, &ONE, &YB_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){i - 1}, &(i32){n - i}, &ONE, &Q_(1, i + 1), &ldq, &Q_(i, i + 1), &ldq, &ZERO, dwork, &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, 1), &ldyb, dwork, &int1, &ONE, &YB_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){i - 1}, &(i32){n - i}, &ONE, &B_(1, k + i + 1), &ldb, &Q_(i, i + 1), &ldq, &ZERO, &DW(nb + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, nb1), &ldyb, &DW(nb + 1), &int1, &ONE, &YB_(i + 1, i), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-tauq}, &YB_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &XB_(k + i + 1, 1), &ldxb, &ONE, &B_(i + 1, k + i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &XB_(k + i + 1, nb1), &ldxb, &ONE, &B_(i + 1, k + i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &YB_(i + 1, 1), &ldyb, &Q_(1, i + 1), &int1, &ONE, &B_(i + 1, k + i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, nb1), &ldyb, &B_(1, k + i + 1), &int1, &ONE, &B_(i + 1, k + i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){n - i}, &ONE, &Q_(i + 1, i + 1), &ldq, &Q_(i, i + 1), &ldq, &ZERO, &YQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XQ_(i + 1, 1), &ldxq, &Q_(i, i + 1), &ldq, &ZERO, &YQ_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &YQ_(1, i), &int1, &ONE, &YQ_(i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XQ_(i + 1, nb1), &ldxq, &Q_(i, i + 1), &ldq, &ZERO, &YQ_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &YQ_(1, i), &int1, &ONE, &YQ_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, 1), &ldyq, dwork, &int1, &ONE, &YQ_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, nb1), &ldyq, &DW(nb + 1), &int1, &ONE, &YQ_(i + 1, i), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-tauq}, &YQ_(i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &XQ_(i + 1, 1), &ldxq, &ONE, &Q_(i + 1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &XQ_(i + 1, nb1), &ldxq, &ONE, &Q_(i + 1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &YQ_(i + 1, 1), &ldyq, &Q_(1, i + 1), &int1, &ONE, &Q_(i + 1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, nb1), &ldyq, &B_(1, k + i + 1), &int1, &ONE, &Q_(i + 1, i + 1), &int1);
            SLC_DROT(&(i32){n - i}, &Q_(i + 1, i + 1), &int1, &B_(i + 1, k + i + 1), &int1, &c, &s);
            for (j = 1; j <= i; j++) {
                XB_(k + i + 1, j) = ZERO;
            }
            for (j = 1; j <= i; j++) {
                XB_(k + i + 1, nb + j) = ZERO;
            }
            SLC_DGEMV("N", &(i32){n - i}, &(i32){n - i}, &ONE, &B_(i + 1, k + i + 1), &ldb, &B_(i, k + i + 1), &ldb, &ZERO, &YB_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XB_(k + i + 2, 1), &ldxb, &B_(i, k + i + 2), &ldb, &ZERO, &YB_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &YB_(1, i + nb), &int1, &ONE, &YB_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XB_(k + i + 2, nb1), &ldxb, &B_(i, k + i + 2), &ldb, &ZERO, &YB_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &YB_(1, i + nb), &int1, &ONE, &YB_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &i, &(i32){n - i - 1}, &ONE, &Q_(1, i + 2), &ldq, &B_(i, k + i + 2), &ldb, &ZERO, &DW(nb2 + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &YB_(i + 1, 1), &ldyb, &DW(nb2 + 1), &int1, &ONE, &YB_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){i - 1}, &(i32){n - i - 1}, &ONE, &B_(1, k + i + 2), &ldb, &B_(i, k + i + 2), &ldb, &ZERO, &DW(nb3 + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YB_(i + 1, nb1), &ldyb, &DW(nb3 + 1), &int1, &ONE, &YB_(i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-TAUR_(i)}, &YB_(i + 1, i + nb), &int1);
            SLC_DAXPY(&(i32){n - i}, &ONE, &YB_(i + 1, i + nb), &int1, &B_(i + 1, k + i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){n - i}, &ONE, &Q_(i + 1, i + 1), &ldq, &B_(i, k + i + 1), &ldb, &ZERO, &YQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XQ_(i + 2, 1), &ldxq, &B_(i, k + i + 2), &ldb, &ZERO, &YQ_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &YQ_(1, i + nb), &int1, &ONE, &YQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XQ_(i + 2, nb1), &ldxq, &B_(i, k + i + 2), &ldb, &ZERO, &YQ_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &YQ_(1, i + nb), &int1, &ONE, &YQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &YQ_(i + 1, 1), &ldyq, &DW(nb2 + 1), &int1, &ONE, &YQ_(i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &(i32){i - 1}, &ONE, &YQ_(i + 1, nb1), &ldyq, &DW(nb3 + 1), &int1, &ONE, &YQ_(i + 1, i + nb), &int1);
            SLC_DSCAL(&(i32){n - i}, &(f64){-TAUR_(i)}, &YQ_(i + 1, i + nb), &int1);
            SLC_DAXPY(&(i32){n - i}, &ONE, &YQ_(i + 1, i + nb), &int1, &Q_(i + 1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){n - i}, &ONE, &A_(1, i + 1), &lda, &Q_(i, i + 1), &ldq, &ZERO, &YA_(1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XA_(i + 1, 1), &ldxa, &Q_(i, i + 1), &ldq, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &DW(pdw), &int1, &ONE, &YA_(k + i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XA_(i + 1, nb1), &ldxa, &Q_(i, i + 1), &ldq, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &DW(pdw), &int1, &ONE, &YA_(k + i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YA_(1, 1), &ldya, dwork, &int1, &ONE, &YA_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YA_(1, nb1), &ldya, &DW(nb + 1), &int1, &ONE, &YA_(1, i), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-tauq}, &YA_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &XA_(i + 1, 1), &ldxa, &ONE, &A_(k + i + 1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &XA_(i + 1, nb1), &ldxa, &ONE, &A_(k + i + 1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &YA_(1, 1), &ldya, &Q_(1, i + 1), &int1, &ONE, &A_(1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YA_(1, nb1), &ldya, &B_(1, k + i + 1), &int1, &ONE, &A_(1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){n - i}, &ONE, &G_(1, k + i + 1), &ldg, &Q_(i, i + 1), &ldq, &ZERO, &YG_(1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XG_(k + i + 1, 1), &ldxg, &Q_(i, i + 1), &ldq, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &DW(pdw), &int1, &ONE, &YG_(k + i + 1, i), &int1);
            SLC_DGEMV("T", &(i32){n - i}, &i, &ONE, &XG_(k + i + 1, nb1), &ldxg, &Q_(i, i + 1), &ldq, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &DW(pdw), &int1, &ONE, &YG_(k + i + 1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YG_(1, 1), &ldyg, dwork, &int1, &ONE, &YG_(1, i), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YG_(1, nb1), &ldyg, &DW(nb + 1), &int1, &ONE, &YG_(1, i), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-tauq}, &YG_(1, i), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &XG_(k + i + 1, 1), &ldxg, &ONE, &G_(k + i + 1, k + i + 1), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &XG_(k + i + 1, nb1), &ldxg, &ONE, &G_(k + i + 1, k + i + 1), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &YG_(1, 1), &ldyg, &Q_(1, i + 1), &int1, &ONE, &G_(1, k + i + 1), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YG_(1, nb1), &ldyg, &B_(1, k + i + 1), &int1, &ONE, &G_(1, k + i + 1), &int1);
            for (j = 1; j <= i; j++) {
                XG_(k + i + 1, j) = ZERO;
            }
            for (j = 1; j <= i; j++) {
                XG_(k + i + 1, nb + j) = ZERO;
            }
            SLC_DROT(&(i32){k + n}, &A_(1, i + 1), &int1, &G_(1, k + i + 1), &int1, &c, &s);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){n - i}, &ONE, &A_(1, i + 1), &lda, &B_(i, k + i + 1), &ldb, &ZERO, &YA_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XA_(i + 2, 1), &ldxa, &B_(i, k + i + 2), &ldb, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &DW(pdw), &int1, &ONE, &YA_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XA_(i + 2, nb1), &ldxa, &B_(i, k + i + 2), &ldb, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &DW(pdw), &int1, &ONE, &YA_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &YA_(1, 1), &ldya, &DW(nb2 + 1), &int1, &ONE, &YA_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YA_(1, nb1), &ldya, &DW(nb3 + 1), &int1, &ONE, &YA_(1, i + nb), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-TAUR_(i)}, &YA_(1, i + nb), &int1);
            SLC_DAXPY(&(i32){k + n}, &ONE, &YA_(1, i + nb), &int1, &A_(1, i + 1), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){n - i}, &ONE, &G_(1, k + i + 1), &ldg, &B_(i, k + i + 1), &ldb, &ZERO, &YG_(1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XG_(k + i + 2, 1), &ldxg, &B_(i, k + i + 2), &ldb, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &Q_(i + 1, 1), &ldq, &DW(pdw), &int1, &ONE, &YG_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("T", &(i32){n - i - 1}, &i, &ONE, &XG_(k + i + 2, nb1), &ldxg, &B_(i, k + i + 2), &ldb, &ZERO, &DW(pdw), &int1);
            SLC_DGEMV("N", &(i32){n - i}, &i, &ONE, &A_(k + i + 1, 1), &lda, &DW(pdw), &int1, &ONE, &YG_(k + i + 1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &i, &ONE, &YG_(1, 1), &ldyg, &DW(nb2 + 1), &int1, &ONE, &YG_(1, i + nb), &int1);
            SLC_DGEMV("N", &(i32){k + n}, &(i32){i - 1}, &ONE, &YG_(1, nb1), &ldyg, &DW(nb3 + 1), &int1, &ONE, &YG_(1, i + nb), &int1);
            SLC_DSCAL(&(i32){k + n}, &(f64){-TAUR_(i)}, &YG_(1, i + nb), &int1);
            SLC_DAXPY(&(i32){k + n}, &ONE, &YG_(1, i + nb), &int1, &G_(1, k + i + 1), &int1);
            B_(i, k + i + 1) = temp;
            Q_(i, i + 1) = tauq;
            CSR_(2*i - 1) = c;
            CSR_(2*i) = s;
        }
    }
}

#undef A_
#undef B_
#undef G_
#undef Q_
#undef XA_
#undef XB_
#undef XG_
#undef XQ_
#undef YA_
#undef YB_
#undef YG_
#undef YQ_
#undef DW
#undef CSL_
#undef CSR_
#undef TAUL_
#undef TAUR_
