/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SG03AY - Solve reduced generalized continuous-time Lyapunov equation
 *
 * Solves for X either the reduced generalized continuous-time
 * Lyapunov equation:
 *     A' * X * E + E' * X * A = SCALE * Y    (TRANS='N')
 * or
 *     A * X * E' + E * X * A' = SCALE * Y    (TRANS='T')
 *
 * where Y is symmetric. A is upper quasi-triangular, E is upper triangular.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <string.h>
#include <ctype.h>

void sg03ay(const char* trans, i32 n, const f64* a, i32 lda,
            const f64* e, i32 lde, f64* x, i32 ldx, f64* scale, i32* info)
{
    const f64 mone = -1.0;
    const f64 one = 1.0;
    const f64 zero = 0.0;

    i32 dimmat, i, info1, kb, kh, kl, lb, lh, ll;
    f64 ak11, ak12, ak21, ak22, al11, al12, al21, al22;
    f64 ek11, ek12, ek22, el11, el12, el22, scale1;

    i32 piv1[4], piv2[4];
    f64 mat[16], rhs[4], tm[4];

    bool notrns = (toupper(trans[0]) == 'N');

    *info = 0;
    if (!notrns && toupper(trans[0]) != 'T') {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (lde < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -8;
    }

    if (*info != 0) {
        return;
    }

    *scale = one;

    if (n == 0) return;

    if (notrns) {
        // Solve equation (1): A'*X*E + E'*X*A = scale*Y
        kl = 0;
        kb = 1;

        while (kl + kb <= n) {
            kl = kl + kb;
            if (kl == n) {
                kb = 1;
            } else {
                if (a[kl + (kl - 1) * lda] != zero) {
                    kb = 2;
                } else {
                    kb = 1;
                }
            }
            kh = kl + kb - 1;

            // Copy elements of solution already known by symmetry
            if (kl > 1) {
                for (i = kl; i <= kh; i++) {
                    i32 len = kl - 1;
                    i32 inc1 = 1;
                    SLC_DCOPY(&len, &x[(i - 1) * ldx], &inc1, &x[i - 1], &ldx);
                }
            }

            // Inner loop
            ll = kl - 1;
            lb = 1;

            while (ll + lb <= n) {
                ll = ll + lb;
                if (ll == n) {
                    lb = 1;
                } else {
                    if (a[ll + (ll - 1) * lda] != zero) {
                        lb = 2;
                    } else {
                        lb = 1;
                    }
                }
                lh = ll + lb - 1;

                // Update RHS (I)
                if (ll > 1) {
                    i32 nrows = kb;
                    i32 ncols = lb;
                    i32 k = ll - 1;
                    i32 lhkl1 = lh - kl + 1;
                    i32 lhkh1 = lh - kh + 1;
                    i32 two = 2;

                    SLC_DGEMM("N", "N", &nrows, &ncols, &k, &one,
                              &x[(kl - 1)], &ldx, &e[(ll - 1) * lde], &lde,
                              &zero, tm, &two);

                    SLC_DGEMM("T", "N", &lhkl1, &ncols, &nrows, &mone,
                              &a[(kl - 1) + (kl - 1) * lda], &lda, tm, &two,
                              &one, &x[(kl - 1) + (ll - 1) * ldx], &ldx);

                    SLC_DGEMM("N", "N", &nrows, &ncols, &k, &one,
                              &x[(kl - 1)], &ldx, &a[(ll - 1) * lda], &lda,
                              &zero, tm, &two);

                    SLC_DGEMM("T", "N", &lhkh1, &ncols, &nrows, &mone,
                              &e[(kl - 1) + (kh - 1) * lde], &lde, tm, &two,
                              &one, &x[(kh - 1) + (ll - 1) * ldx], &ldx);

                    if (kb == 2) {
                        f64 alpha = -e[(kl - 1) + (kl - 1) * lde];
                        SLC_DAXPY(&ncols, &alpha, tm, &two,
                                  &x[(kl - 1) + (ll - 1) * ldx], &ldx);
                    }
                }

                // Solve small Sylvester equations
                if (kb == 1 && lb == 1) {
                    dimmat = 1;
                    mat[0] = e[(ll - 1) + (ll - 1) * lde] * a[(kl - 1) + (kl - 1) * lda]
                           + a[(ll - 1) + (ll - 1) * lda] * e[(kl - 1) + (kl - 1) * lde];
                    rhs[0] = x[(kl - 1) + (ll - 1) * ldx];

                } else if (kb == 2 && lb == 1) {
                    dimmat = 2;

                    ak11 = a[(kl - 1) + (kl - 1) * lda];
                    ak12 = a[(kl - 1) + (kh - 1) * lda];
                    ak21 = a[(kh - 1) + (kl - 1) * lda];
                    ak22 = a[(kh - 1) + (kh - 1) * lda];

                    al11 = a[(ll - 1) + (ll - 1) * lda];

                    ek11 = e[(kl - 1) + (kl - 1) * lde];
                    ek12 = e[(kl - 1) + (kh - 1) * lde];
                    ek22 = e[(kh - 1) + (kh - 1) * lde];

                    el11 = e[(ll - 1) + (ll - 1) * lde];

                    mat[0 + 0 * 4] = el11 * ak11 + al11 * ek11;
                    mat[0 + 1 * 4] = el11 * ak21;
                    mat[1 + 0 * 4] = el11 * ak12 + al11 * ek12;
                    mat[1 + 1 * 4] = el11 * ak22 + al11 * ek22;

                    rhs[0] = x[(kl - 1) + (ll - 1) * ldx];
                    rhs[1] = x[(kh - 1) + (ll - 1) * ldx];

                } else if (kb == 1 && lb == 2) {
                    dimmat = 2;

                    ak11 = a[(kl - 1) + (kl - 1) * lda];

                    al11 = a[(ll - 1) + (ll - 1) * lda];
                    al12 = a[(ll - 1) + (lh - 1) * lda];
                    al21 = a[(lh - 1) + (ll - 1) * lda];
                    al22 = a[(lh - 1) + (lh - 1) * lda];

                    ek11 = e[(kl - 1) + (kl - 1) * lde];

                    el11 = e[(ll - 1) + (ll - 1) * lde];
                    el12 = e[(ll - 1) + (lh - 1) * lde];
                    el22 = e[(lh - 1) + (lh - 1) * lde];

                    mat[0 + 0 * 4] = el11 * ak11 + al11 * ek11;
                    mat[0 + 1 * 4] = al21 * ek11;
                    mat[1 + 0 * 4] = el12 * ak11 + al12 * ek11;
                    mat[1 + 1 * 4] = el22 * ak11 + al22 * ek11;

                    rhs[0] = x[(kl - 1) + (ll - 1) * ldx];
                    rhs[1] = x[(kl - 1) + (lh - 1) * ldx];

                } else {
                    dimmat = 4;

                    ak11 = a[(kl - 1) + (kl - 1) * lda];
                    ak12 = a[(kl - 1) + (kh - 1) * lda];
                    ak21 = a[(kh - 1) + (kl - 1) * lda];
                    ak22 = a[(kh - 1) + (kh - 1) * lda];

                    al11 = a[(ll - 1) + (ll - 1) * lda];
                    al12 = a[(ll - 1) + (lh - 1) * lda];
                    al21 = a[(lh - 1) + (ll - 1) * lda];
                    al22 = a[(lh - 1) + (lh - 1) * lda];

                    ek11 = e[(kl - 1) + (kl - 1) * lde];
                    ek12 = e[(kl - 1) + (kh - 1) * lde];
                    ek22 = e[(kh - 1) + (kh - 1) * lde];

                    el11 = e[(ll - 1) + (ll - 1) * lde];
                    el12 = e[(ll - 1) + (lh - 1) * lde];
                    el22 = e[(lh - 1) + (lh - 1) * lde];

                    mat[0 + 0 * 4] = el11 * ak11 + al11 * ek11;
                    mat[0 + 1 * 4] = el11 * ak21;
                    mat[0 + 2 * 4] = al21 * ek11;
                    mat[0 + 3 * 4] = zero;

                    mat[1 + 0 * 4] = el11 * ak12 + al11 * ek12;
                    mat[1 + 1 * 4] = el11 * ak22 + al11 * ek22;
                    mat[1 + 2 * 4] = al21 * ek12;
                    mat[1 + 3 * 4] = al21 * ek22;

                    mat[2 + 0 * 4] = el12 * ak11 + al12 * ek11;
                    mat[2 + 1 * 4] = el12 * ak21;
                    mat[2 + 2 * 4] = el22 * ak11 + al22 * ek11;
                    mat[2 + 3 * 4] = el22 * ak21;

                    mat[3 + 0 * 4] = el12 * ak12 + al12 * ek12;
                    mat[3 + 1 * 4] = el12 * ak22 + al12 * ek22;
                    mat[3 + 2 * 4] = el22 * ak12 + al22 * ek12;
                    mat[3 + 3 * 4] = el22 * ak22 + al22 * ek22;

                    rhs[0] = x[(kl - 1) + (ll - 1) * ldx];
                    if (kl == ll) {
                        rhs[1] = x[(kl - 1) + (kh - 1) * ldx];
                    } else {
                        rhs[1] = x[(kh - 1) + (ll - 1) * ldx];
                    }
                    rhs[2] = x[(kl - 1) + (lh - 1) * ldx];
                    rhs[3] = x[(kh - 1) + (lh - 1) * ldx];
                }

                i32 four = 4;
                mb02uv(dimmat, mat, four, piv1, piv2, &info1);
                if (info1 != 0) *info = 1;
                mb02uu(dimmat, mat, four, rhs, piv1, piv2, &scale1);

                // Scaling
                if (scale1 != one) {
                    for (i = 0; i < n; i++) {
                        i32 len = n;
                        i32 inc1 = 1;
                        SLC_DSCAL(&len, &scale1, &x[i * ldx], &inc1);
                    }
                    *scale = (*scale) * scale1;
                }

                if (lb == 1 && kb == 1) {
                    x[(kl - 1) + (ll - 1) * ldx] = rhs[0];
                } else if (lb == 1 && kb == 2) {
                    x[(kl - 1) + (ll - 1) * ldx] = rhs[0];
                    x[(kh - 1) + (ll - 1) * ldx] = rhs[1];
                } else if (lb == 2 && kb == 1) {
                    x[(kl - 1) + (ll - 1) * ldx] = rhs[0];
                    x[(kl - 1) + (lh - 1) * ldx] = rhs[1];
                } else {
                    x[(kl - 1) + (ll - 1) * ldx] = rhs[0];
                    x[(kh - 1) + (ll - 1) * ldx] = rhs[1];
                    x[(kl - 1) + (lh - 1) * ldx] = rhs[2];
                    x[(kh - 1) + (lh - 1) * ldx] = rhs[3];
                }

                // Update RHS (II)
                if (kl < ll) {
                    i32 nrows = kb;
                    i32 ncols = lb;
                    i32 lhkh = lh - kh;
                    i32 inc1 = 1;
                    i32 two = 2;

                    if (lb == 2) {
                        SLC_DGEMV("N", &nrows, &two, &one, &x[(kl - 1) + (ll - 1) * ldx],
                                  &ldx, &e[(ll - 1) + (lh - 1) * lde], &inc1, &zero, &tm[2], &inc1);
                    }
                    SLC_DCOPY(&nrows, &x[(kl - 1) + (ll - 1) * ldx], &inc1, tm, &inc1);
                    f64 ell = e[(ll - 1) + (ll - 1) * lde];
                    SLC_DSCAL(&nrows, &ell, tm, &inc1);

                    SLC_DGEMM("T", "N", &lhkh, &ncols, &nrows, &mone,
                              &a[(kl - 1) + kh * lda], &lda, tm, &two,
                              &one, &x[kh + (ll - 1) * ldx], &ldx);

                    SLC_DGEMM("N", "N", &nrows, &ncols, &ncols, &one,
                              &x[(kl - 1) + (ll - 1) * ldx], &ldx,
                              &a[(ll - 1) + (ll - 1) * lda], &lda, &zero, tm, &two);

                    SLC_DGEMM("T", "N", &lhkh, &ncols, &nrows, &mone,
                              &e[(kl - 1) + kh * lde], &lde, tm, &two,
                              &one, &x[kh + (ll - 1) * ldx], &ldx);
                }
            }
        }

    } else {
        // Solve equation (2): A*X*E' + E*X*A' = scale*Y
        ll = n + 1;

        while (ll > 1) {
            lh = ll - 1;
            if (lh == 1) {
                lb = 1;
            } else {
                if (a[(ll - 2) + (ll - 3) * lda] != zero) {
                    lb = 2;
                } else {
                    lb = 1;
                }
            }
            ll = ll - lb;

            // Copy elements of solution already known by symmetry
            if (lh < n) {
                for (i = ll; i <= lh; i++) {
                    i32 len = n - lh;
                    i32 inc1 = 1;
                    SLC_DCOPY(&len, &x[(i - 1) + lh * ldx], &ldx, &x[lh + (i - 1) * ldx], &inc1);
                }
            }

            // Inner loop
            kl = lh + 1;

            while (kl > 1) {
                kh = kl - 1;
                if (kh == 1) {
                    kb = 1;
                } else {
                    if (a[(kl - 2) + (kl - 3) * lda] != zero) {
                        kb = 2;
                    } else {
                        kb = 1;
                    }
                }
                kl = kl - kb;

                // Update RHS (I)
                if (kh < n) {
                    i32 nrows = kb;
                    i32 ncols = lb;
                    i32 nkh = n - kh;
                    i32 llkl1 = ll - kl + 1;
                    i32 lhkl1 = lh - kl + 1;
                    i32 two = 2;
                    i32 inc1 = 1;

                    SLC_DGEMM("N", "N", &nrows, &ncols, &nkh, &one,
                              &a[(kl - 1) + kh * lda], &lda, &x[kh + (ll - 1) * ldx],
                              &ldx, &zero, tm, &two);

                    SLC_DGEMM("N", "T", &nrows, &llkl1, &ncols, &mone, tm, &two,
                              &e[(kl - 1) + (ll - 1) * lde], &lde,
                              &one, &x[(kl - 1) + (kl - 1) * ldx], &ldx);

                    if (lb == 2) {
                        f64 alpha = -e[(lh - 1) + (lh - 1) * lde];
                        SLC_DAXPY(&nrows, &alpha, &tm[1], &inc1,
                                  &x[(kl - 1) + (lh - 1) * ldx], &inc1);
                    }

                    SLC_DGEMM("N", "N", &nrows, &ncols, &nkh, &one,
                              &e[(kl - 1) + kh * lde], &lde, &x[kh + (ll - 1) * ldx],
                              &ldx, &zero, tm, &two);

                    SLC_DGEMM("N", "T", &nrows, &lhkl1, &ncols, &mone, tm, &two,
                              &a[(kl - 1) + (ll - 1) * lda], &lda,
                              &one, &x[(kl - 1) + (kl - 1) * ldx], &ldx);
                }

                // Solve small Sylvester equations
                if (kb == 1 && lb == 1) {
                    dimmat = 1;
                    mat[0] = e[(ll - 1) + (ll - 1) * lde] * a[(kl - 1) + (kl - 1) * lda]
                           + a[(ll - 1) + (ll - 1) * lda] * e[(kl - 1) + (kl - 1) * lde];
                    rhs[0] = x[(kl - 1) + (ll - 1) * ldx];

                } else if (kb == 2 && lb == 1) {
                    dimmat = 2;

                    ak11 = a[(kl - 1) + (kl - 1) * lda];
                    ak12 = a[(kl - 1) + (kh - 1) * lda];
                    ak21 = a[(kh - 1) + (kl - 1) * lda];
                    ak22 = a[(kh - 1) + (kh - 1) * lda];

                    al11 = a[(ll - 1) + (ll - 1) * lda];

                    ek11 = e[(kl - 1) + (kl - 1) * lde];
                    ek12 = e[(kl - 1) + (kh - 1) * lde];
                    ek22 = e[(kh - 1) + (kh - 1) * lde];

                    el11 = e[(ll - 1) + (ll - 1) * lde];

                    mat[0 + 0 * 4] = el11 * ak11 + al11 * ek11;
                    mat[0 + 1 * 4] = el11 * ak12 + al11 * ek12;
                    mat[1 + 0 * 4] = el11 * ak21;
                    mat[1 + 1 * 4] = el11 * ak22 + al11 * ek22;

                    rhs[0] = x[(kl - 1) + (ll - 1) * ldx];
                    rhs[1] = x[(kh - 1) + (ll - 1) * ldx];

                } else if (kb == 1 && lb == 2) {
                    dimmat = 2;

                    ak11 = a[(kl - 1) + (kl - 1) * lda];

                    al11 = a[(ll - 1) + (ll - 1) * lda];
                    al12 = a[(ll - 1) + (lh - 1) * lda];
                    al21 = a[(lh - 1) + (ll - 1) * lda];
                    al22 = a[(lh - 1) + (lh - 1) * lda];

                    ek11 = e[(kl - 1) + (kl - 1) * lde];

                    el11 = e[(ll - 1) + (ll - 1) * lde];
                    el12 = e[(ll - 1) + (lh - 1) * lde];
                    el22 = e[(lh - 1) + (lh - 1) * lde];

                    mat[0 + 0 * 4] = el11 * ak11 + al11 * ek11;
                    mat[0 + 1 * 4] = el12 * ak11 + al12 * ek11;
                    mat[1 + 0 * 4] = al21 * ek11;
                    mat[1 + 1 * 4] = el22 * ak11 + al22 * ek11;

                    rhs[0] = x[(kl - 1) + (ll - 1) * ldx];
                    rhs[1] = x[(kl - 1) + (lh - 1) * ldx];

                } else {
                    dimmat = 4;

                    ak11 = a[(kl - 1) + (kl - 1) * lda];
                    ak12 = a[(kl - 1) + (kh - 1) * lda];
                    ak21 = a[(kh - 1) + (kl - 1) * lda];
                    ak22 = a[(kh - 1) + (kh - 1) * lda];

                    al11 = a[(ll - 1) + (ll - 1) * lda];
                    al12 = a[(ll - 1) + (lh - 1) * lda];
                    al21 = a[(lh - 1) + (ll - 1) * lda];
                    al22 = a[(lh - 1) + (lh - 1) * lda];

                    ek11 = e[(kl - 1) + (kl - 1) * lde];
                    ek12 = e[(kl - 1) + (kh - 1) * lde];
                    ek22 = e[(kh - 1) + (kh - 1) * lde];

                    el11 = e[(ll - 1) + (ll - 1) * lde];
                    el12 = e[(ll - 1) + (lh - 1) * lde];
                    el22 = e[(lh - 1) + (lh - 1) * lde];

                    mat[0 + 0 * 4] = el11 * ak11 + al11 * ek11;
                    mat[0 + 1 * 4] = el11 * ak12 + al11 * ek12;
                    mat[0 + 2 * 4] = el12 * ak11 + al12 * ek11;
                    mat[0 + 3 * 4] = el12 * ak12 + al12 * ek12;

                    mat[1 + 0 * 4] = el11 * ak21;
                    mat[1 + 1 * 4] = el11 * ak22 + al11 * ek22;
                    mat[1 + 2 * 4] = el12 * ak21;
                    mat[1 + 3 * 4] = el12 * ak22 + al12 * ek22;

                    mat[2 + 0 * 4] = al21 * ek11;
                    mat[2 + 1 * 4] = al21 * ek12;
                    mat[2 + 2 * 4] = el22 * ak11 + al22 * ek11;
                    mat[2 + 3 * 4] = el22 * ak12 + al22 * ek12;

                    mat[3 + 0 * 4] = zero;
                    mat[3 + 1 * 4] = al21 * ek22;
                    mat[3 + 2 * 4] = el22 * ak21;
                    mat[3 + 3 * 4] = el22 * ak22 + al22 * ek22;

                    rhs[0] = x[(kl - 1) + (ll - 1) * ldx];
                    if (kl == ll) {
                        rhs[1] = x[(kl - 1) + (kh - 1) * ldx];
                    } else {
                        rhs[1] = x[(kh - 1) + (ll - 1) * ldx];
                    }
                    rhs[2] = x[(kl - 1) + (lh - 1) * ldx];
                    rhs[3] = x[(kh - 1) + (lh - 1) * ldx];
                }

                i32 four = 4;
                mb02uv(dimmat, mat, four, piv1, piv2, &info1);
                if (info1 != 0) *info = 1;
                mb02uu(dimmat, mat, four, rhs, piv1, piv2, &scale1);

                // Scaling
                if (scale1 != one) {
                    for (i = 0; i < n; i++) {
                        i32 len = n;
                        i32 inc1 = 1;
                        SLC_DSCAL(&len, &scale1, &x[i * ldx], &inc1);
                    }
                    *scale = (*scale) * scale1;
                }

                if (lb == 1 && kb == 1) {
                    x[(kl - 1) + (ll - 1) * ldx] = rhs[0];
                } else if (lb == 1 && kb == 2) {
                    x[(kl - 1) + (ll - 1) * ldx] = rhs[0];
                    x[(kh - 1) + (ll - 1) * ldx] = rhs[1];
                } else if (lb == 2 && kb == 1) {
                    x[(kl - 1) + (ll - 1) * ldx] = rhs[0];
                    x[(kl - 1) + (lh - 1) * ldx] = rhs[1];
                } else {
                    x[(kl - 1) + (ll - 1) * ldx] = rhs[0];
                    x[(kh - 1) + (ll - 1) * ldx] = rhs[1];
                    x[(kl - 1) + (lh - 1) * ldx] = rhs[2];
                    x[(kh - 1) + (lh - 1) * ldx] = rhs[3];
                }

                // Update RHS (II)
                if (kl < ll) {
                    i32 nrows = kb;
                    i32 ncols = lb;
                    i32 llkl = ll - kl;
                    i32 two = 2;

                    SLC_DGEMM("N", "N", &nrows, &ncols, &nrows, &one,
                              &a[(kl - 1) + (kl - 1) * lda], &lda,
                              &x[(kl - 1) + (ll - 1) * ldx], &ldx, &zero, tm, &two);

                    SLC_DGEMM("N", "T", &nrows, &llkl, &ncols, &mone, tm, &two,
                              &e[(kl - 1) + (ll - 1) * lde], &lde,
                              &one, &x[(kl - 1) + (kl - 1) * ldx], &ldx);

                    SLC_DGEMV("T", &nrows, &ncols, &one, &x[(kl - 1) + (ll - 1) * ldx],
                              &ldx, &e[(kl - 1) + (kl - 1) * lde], &lde, &zero, tm, &two);

                    if (kb == 2) {
                        SLC_DCOPY(&ncols, &x[(kh - 1) + (ll - 1) * ldx], &ldx,
                                  &tm[1], &two);
                        f64 ekh = e[(kh - 1) + (kh - 1) * lde];
                        SLC_DSCAL(&ncols, &ekh, &tm[1], &two);
                    }

                    SLC_DGEMM("N", "T", &nrows, &llkl, &ncols, &mone, tm, &two,
                              &a[(kl - 1) + (ll - 1) * lda], &lda,
                              &one, &x[(kl - 1) + (kl - 1) * ldx], &ldx);
                }
            }
        }
    }
}
