/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB10ID - Positive feedback controller for loop shaping design
 *
 * Computes the matrices of the positive feedback controller
 *
 *         | Ak | Bk |
 *     K = |----|----|
 *         | Ck | Dk |
 *
 * for the shaped plant
 *
 *         | A | B |
 *     G = |---|---|
 *         | C | D |
 *
 * in the McFarlane/Glover Loop Shaping Design Procedure.
 *
 * References:
 * [1] McFarlane, D. and Glover, K.
 *     "A loop shaping design procedure using H_infinity synthesis."
 *     IEEE Trans. Automat. Control, vol. AC-37, no. 6, pp. 759-769, 1992.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>
#include <ctype.h>

static int select_dummy(const f64* reig, const f64* ieig) {
    (void)reig;
    (void)ieig;
    return 0;
}

void sb10id(
    const i32 n,
    const i32 m,
    const i32 np,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    const f64 factor,
    i32* nk,
    f64* ak,
    const i32 ldak,
    f64* bk,
    const i32 ldbk,
    f64* ck,
    const i32 ldck,
    f64* dk,
    const i32 lddk,
    f64* rcond,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 mone = -1.0;
    *info = 0;

    i32 maxn1 = n > 1 ? n : 1;
    i32 maxm1 = m > 1 ? m : 1;
    i32 maxnp1 = np > 1 ? np : 1;

    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (np < 0) {
        *info = -3;
    } else if (lda < maxn1) {
        *info = -5;
    } else if (ldb < maxn1) {
        *info = -7;
    } else if (ldc < maxnp1) {
        *info = -9;
    } else if (ldd < maxnp1) {
        *info = -11;
    } else if (factor < one) {
        *info = -12;
    } else if (ldak < maxn1) {
        *info = -15;
    } else if (ldbk < maxn1) {
        *info = -17;
    } else if (ldck < maxm1) {
        *info = -19;
    } else if (lddk < maxm1) {
        *info = -21;
    }

    i32 minwrk = 4*n*n + m*m + np*np + 2*m*n + n*np + 4*n;
    i32 tmp1 = 6*n*n + 5;
    i32 tmp2 = 4*n*n + 8*n;
    if (tmp2 > 1) tmp1 = tmp1 > tmp2 ? tmp1 : (tmp2 + 5);
    i32 tmp3 = n*np + 2*n;
    minwrk += (tmp1 > tmp3 ? tmp1 : tmp3);

    if (ldwork < minwrk) {
        *info = -25;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0 || np == 0) {
        rcond[0] = one;
        rcond[1] = one;
        dwork[0] = one;
        return;
    }

    i32 i1 = n * n;
    i32 i2 = i1 + n * n;
    i32 i3 = i2 + m * n;
    i32 i4 = i3 + m * n;
    i32 i5 = i4 + m * m;
    i32 i6 = i5 + np * np;
    i32 i7 = i6 + np * n;
    i32 i8 = i7 + n * n;
    i32 i9 = i8 + n * n;
    i32 i10 = i9 + n * n;
    i32 i11 = i10 + n * n;
    i32 i12 = i11 + 2 * n;
    i32 i13 = i12 + 2 * n;
    i32 iwrk = i13 + 4 * n * n;

    SLC_DGEMM("T", "N", &m, &n, &np, &one, d, &ldd, c, &ldc, &zero, &dwork[i2], &m);

    SLC_DLASET("U", &m, &m, &zero, &one, &dwork[i4], &m);
    SLC_DSYRK("U", "T", &m, &np, &one, d, &ldd, &one, &dwork[i4], &m);

    i32 info2;
    SLC_DPOTRF("U", &m, &dwork[i4], &m, &info2);

    SLC_DPOTRS("U", &m, &n, &dwork[i4], &m, &dwork[i2], &m, &info2);

    SLC_DLACPY("F", &n, &m, b, &ldb, &dwork[i3], &n);
    SLC_DTRSM("R", "U", "N", "N", &n, &m, &one, &dwork[i4], &m, &dwork[i3], &n);

    SLC_DLASET("U", &np, &np, &zero, &one, &dwork[i5], &np);
    SLC_DSYRK("U", "N", &np, &m, &one, d, &ldd, &one, &dwork[i5], &np);

    SLC_DPOTRF("U", &np, &dwork[i5], &np, &info2);

    SLC_DLACPY("F", &np, &n, c, &ldc, &dwork[i6], &np);
    SLC_DTRSM("L", "U", "T", "N", &np, &n, &one, &dwork[i5], &np, &dwork[i6], &np);

    SLC_DLACPY("F", &n, &n, a, &lda, &dwork[i7], &n);
    SLC_DGEMM("N", "N", &n, &n, &m, &mone, b, &ldb, &dwork[i2], &m, &one, &dwork[i7], &n);

    SLC_DSYRK("U", "T", &n, &np, &one, &dwork[i6], &np, &zero, &dwork[i8], &n);

    SLC_DSYRK("U", "N", &n, &m, &one, &dwork[i3], &n, &zero, &dwork[i9], &n);

    i32 n2 = 2 * n;
    f64 sep, ferr;
    sb02rd("A", "C", "D", "N", "U", "G", "S", "N", "O", n,
           &dwork[i7], n, &dwork[i10], n, ak, ldak,
           &dwork[i9], n, &dwork[i8], n, dwork, n, &sep,
           &rcond[0], &ferr, &dwork[i11], &dwork[i12],
           &dwork[i13], n2, iwork, &dwork[iwrk], ldwork - iwrk, bwork, &info2);

    if (info2 != 0) {
        *info = 1;
        return;
    }
    i32 lwa = (i32)dwork[iwrk] + iwrk;
    i32 lwamax = minwrk > lwa ? minwrk : lwa;

    sb02rd("A", "C", "D", "T", "U", "G", "S", "N", "O", n,
           &dwork[i7], n, &dwork[i10], n, ak, ldak,
           &dwork[i8], n, &dwork[i9], n, &dwork[i1], n, &sep,
           &rcond[1], &ferr, &dwork[i11], &dwork[i12],
           &dwork[i13], n2, iwork, &dwork[iwrk], ldwork - iwrk, bwork, &info2);

    if (info2 != 0) {
        *info = 2;
        return;
    }
    lwa = (i32)dwork[iwrk] + iwrk;
    lwamax = lwa > lwamax ? lwa : lwamax;

    SLC_DTRSM("R", "U", "T", "N", &n, &m, &one, &dwork[i4], &m, &dwork[i3], &n);
    SLC_DGEMM("T", "N", &m, &n, &n, &mone, &dwork[i3], &n, dwork, &n, &mone, &dwork[i2], &m);

    SLC_DGEMM("N", "N", &n, &n, &n, &one, dwork, &n, &dwork[i1], &n, &zero, &dwork[i7], &n);

    i32 sdim;
    SLC_DGEES("N", "N", select_dummy, &n, &dwork[i7], &n, &sdim,
              &dwork[i11], &dwork[i12], &dwork[iwrk], &n,
              &dwork[iwrk], &(i32){ldwork - iwrk}, bwork, &info2);

    if (info2 != 0) {
        *info = 3;
        return;
    }
    lwa = (i32)dwork[iwrk] + iwrk;
    lwamax = lwa > lwamax ? lwa : lwamax;

    f64 gamma = zero;
    for (i32 i = 0; i < n; i++) {
        if (dwork[i11 + i] > gamma) {
            gamma = dwork[i11 + i];
        }
    }
    gamma = factor * sqrt(one + gamma);

    i4 = i3 + n * n;
    i5 = i4 + n * n;

    SLC_DLACPY("F", &n, &n, a, &lda, &dwork[i4], &n);
    SLC_DGEMM("N", "N", &n, &n, &m, &one, b, &ldb, &dwork[i2], &m, &one, &dwork[i4], &n);

    f64 gamsq = gamma * gamma;
    f64 w1diag = one - gamsq;
    SLC_DLASET("F", &n, &n, &zero, &w1diag, &dwork[i3], &n);
    SLC_DGEMM("N", "N", &n, &n, &n, &one, &dwork[i1], &n, dwork, &n, &one, &dwork[i3], &n);

    SLC_DGEMM("N", "T", &n, &np, &n, &gamsq, &dwork[i1], &n, c, &ldc, &zero, bk, &ldbk);

    SLC_DLACPY("F", &np, &n, c, &ldc, &dwork[i5], &np);
    SLC_DGEMM("N", "N", &np, &n, &m, &one, d, &ldd, &dwork[i2], &m, &one, &dwork[i5], &np);

    SLC_DGEMM("N", "N", &n, &n, &n, &one, &dwork[i3], &n, &dwork[i4], &n, &zero, ak, &ldak);
    SLC_DGEMM("N", "N", &n, &n, &np, &one, bk, &ldbk, &dwork[i5], &np, &one, ak, &ldak);

    SLC_DGEMM("T", "N", &m, &n, &n, &one, b, &ldb, dwork, &n, &zero, ck, &ldck);

    for (i32 i = 0; i < m; i++) {
        for (i32 j = 0; j < np; j++) {
            dk[i + j * lddk] = -d[j + i * ldd];
        }
    }

    iwrk = i4;

    i32 ldwork2 = ldwork - iwrk;
    sb10jd(n, np, m, ak, ldak, bk, ldbk, ck, ldck, dk, lddk,
           &dwork[i3], n, nk, &dwork[iwrk], ldwork2, &info2);

    if (info2 != 0) {
        *info = 3;
        return;
    }
    lwa = (i32)dwork[iwrk] + iwrk;
    lwamax = lwa > lwamax ? lwa : lwamax;

    i32 nk_val = *nk;

    i2 = np * np;
    i3 = i2 + nk_val * np;
    i4 = i3 + m * m;
    i5 = i4 + n * m;
    i6 = i5 + np * nk_val;
    i7 = i6 + m * n;

    iwrk = i7 + (n + nk_val) * (n + nk_val);

    SLC_DLASET("F", &np, &np, &zero, &one, dwork, &np);
    SLC_DGEMM("N", "N", &np, &np, &m, &mone, d, &ldd, dk, &lddk, &one, dwork, &np);

    SLC_DLACPY("F", &nk_val, &np, bk, &ldbk, &dwork[i2], &nk_val);
    mb02vd("N", nk_val, np, dwork, np, iwork, &dwork[i2], nk_val, &info2);

    if (info2 != 0) {
        *info = 4;
        return;
    }

    SLC_DLASET("F", &m, &m, &zero, &one, &dwork[i3], &m);
    SLC_DGEMM("N", "N", &m, &m, &np, &mone, dk, &lddk, d, &ldd, &one, &dwork[i3], &m);

    SLC_DLACPY("F", &n, &m, b, &ldb, &dwork[i4], &n);
    mb02vd("N", n, m, &dwork[i3], m, iwork, &dwork[i4], n, &info2);

    if (info2 != 0) {
        *info = 5;
        return;
    }

    SLC_DGEMM("N", "N", &np, &nk_val, &m, &one, d, &ldd, ck, &ldck, &zero, &dwork[i5], &np);

    SLC_DGEMM("N", "N", &m, &n, &np, &one, dk, &lddk, c, &ldc, &zero, &dwork[i6], &m);

    i32 nnk = n + nk_val;
    SLC_DLACPY("F", &n, &n, a, &lda, &dwork[i7], &nnk);
    SLC_DGEMM("N", "N", &n, &n, &m, &one, &dwork[i4], &n, &dwork[i6], &m, &one, &dwork[i7], &nnk);

    SLC_DGEMM("N", "N", &nk_val, &n, &np, &one, &dwork[i2], &nk_val, c, &ldc, &zero, &dwork[i7 + n], &nnk);

    i32 offset_12 = i7 + nnk * n;
    SLC_DGEMM("N", "N", &n, &nk_val, &m, &one, &dwork[i4], &n, ck, &ldck, &zero, &dwork[offset_12], &nnk);

    i32 offset_22 = i7 + nnk * n + n;
    SLC_DLACPY("F", &nk_val, &nk_val, ak, &ldak, &dwork[offset_22], &nnk);
    SLC_DGEMM("N", "N", &nk_val, &nk_val, &np, &one, &dwork[i2], &nk_val, &dwork[i5], &np, &one, &dwork[offset_22], &nnk);

    SLC_DGEES("N", "N", select_dummy, &nnk, &dwork[i7], &nnk, &sdim,
              dwork, &dwork[nnk], &dwork[iwrk], &nnk,
              &dwork[iwrk], &(i32){ldwork - iwrk}, bwork, &info2);

    if (info2 != 0) {
        *info = 3;
        return;
    }
    lwa = (i32)dwork[iwrk] + iwrk;
    lwamax = lwa > lwamax ? lwa : lwamax;

    i32 ns = 0;
    for (i32 i = 0; i < nnk; i++) {
        if (dwork[i] >= zero) {
            ns++;
        }
    }
    if (ns > 0) {
        *info = 6;
        return;
    }

    dwork[0] = (f64)lwamax;
}
