/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SG03AX - Solve reduced generalized discrete-time Lyapunov equation
 *
 * Solves:
 *   TRANS='N':  A' * X * A - E' * X * E = scale * Y
 *   TRANS='T':  A * X * A' - E * X * E' = scale * Y
 *
 * where A is upper quasitriangular, E is upper triangular (generalized Schur form),
 * and Y is symmetric.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sg03ax(
    const char* trans,
    const i32 n,
    const f64* a,
    const i32 lda,
    const f64* e,
    const i32 lde,
    f64* x,
    const i32 ldx,
    f64* scale,
    i32* info
)
{
    const f64 MONE = -1.0;
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    bool notrns = (*trans == 'N' || *trans == 'n');

    *info = 0;
    if (!notrns && *trans != 'T' && *trans != 't') {
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
        i32 neginfo = -(*info);
        SLC_XERBLA("SG03AX", &neginfo);
        return;
    }

    *scale = ONE;

    if (n == 0) {
        return;
    }

    f64 mat[16], rhs[4], tm[4];
    i32 piv1[4], piv2[4];

    if (notrns) {
        i32 kl = -1;
        i32 kb = 1;

        while (kl + kb < n) {
            kl = kl + kb;
            if (kl == n - 1) {
                kb = 1;
            } else {
                if (a[(kl+1) + kl*lda] != ZERO) {
                    kb = 2;
                } else {
                    kb = 1;
                }
            }
            i32 kh = kl + kb - 1;

            if (kl > 0) {
                for (i32 i = kl; i <= kh; i++) {
                    for (i32 j = 0; j < kl; j++) {
                        x[i + j*ldx] = x[j + i*ldx];
                    }
                }
            }

            i32 ll = kl - 1;
            i32 lb = 1;

            while (ll + lb < n) {
                ll = ll + lb;
                if (ll == n - 1) {
                    lb = 1;
                } else {
                    if (a[(ll+1) + ll*lda] != ZERO) {
                        lb = 2;
                    } else {
                        lb = 1;
                    }
                }
                i32 lh = ll + lb - 1;

                if (ll > 0) {
                    i32 m_len = kb;
                    i32 n_len = lb;
                    i32 k_len = ll;
                    f64 alpha = ONE;
                    f64 beta = ZERO;
                    i32 ldtm = 2;
                    SLC_DGEMM("N", "N", &m_len, &n_len, &k_len, &alpha,
                              &x[kl], &ldx, &a[0 + ll*lda], &lda, &beta, tm, &ldtm);

                    i32 rows = lh - kl + 1;
                    alpha = MONE;
                    beta = ONE;
                    SLC_DGEMM("T", "N", &rows, &n_len, &m_len, &alpha,
                              &a[kl + kl*lda], &lda, tm, &ldtm, &beta, &x[kl + ll*ldx], &ldx);

                    alpha = ONE;
                    beta = ZERO;
                    SLC_DGEMM("N", "N", &m_len, &n_len, &k_len, &alpha,
                              &x[kl], &ldx, &e[0 + ll*lde], &lde, &beta, tm, &ldtm);

                    rows = lh - kh + 1;
                    beta = ONE;
                    SLC_DGEMM("T", "N", &rows, &n_len, &m_len, &alpha,
                              &e[kl + kh*lde], &lde, tm, &ldtm, &beta, &x[kh + ll*ldx], &ldx);

                    if (kb == 2) {
                        for (i32 j = 0; j < lb; j++) {
                            x[kl + (ll+j)*ldx] += e[kl + kl*lde] * tm[0 + j*2];
                        }
                    }
                }

                i32 dimmat;
                f64 ak11, ak12, ak21, ak22, al11, al12, al21, al22;
                f64 ek11, ek12, ek22, el11, el12, el22;

                if (kb == 1 && lb == 1) {
                    dimmat = 1;
                    mat[0] = a[ll + ll*lda] * a[kl + kl*lda] - e[ll + ll*lde] * e[kl + kl*lde];
                    rhs[0] = x[kl + ll*ldx];
                } else if (kb == 2 && lb == 1) {
                    dimmat = 2;
                    ak11 = a[kl + kl*lda]; ak12 = a[kl + kh*lda];
                    ak21 = a[kh + kl*lda]; ak22 = a[kh + kh*lda];
                    al11 = a[ll + ll*lda];
                    ek11 = e[kl + kl*lde]; ek12 = e[kl + kh*lde]; ek22 = e[kh + kh*lde];
                    el11 = e[ll + ll*lde];

                    mat[0 + 0*4] = al11*ak11 - el11*ek11;
                    mat[0 + 1*4] = al11*ak21;
                    mat[1 + 0*4] = al11*ak12 - el11*ek12;
                    mat[1 + 1*4] = al11*ak22 - el11*ek22;

                    rhs[0] = x[kl + ll*ldx];
                    rhs[1] = x[kh + ll*ldx];
                } else if (kb == 1 && lb == 2) {
                    dimmat = 2;
                    ak11 = a[kl + kl*lda];
                    al11 = a[ll + ll*lda]; al12 = a[ll + lh*lda];
                    al21 = a[lh + ll*lda]; al22 = a[lh + lh*lda];
                    ek11 = e[kl + kl*lde];
                    el11 = e[ll + ll*lde]; el12 = e[ll + lh*lde]; el22 = e[lh + lh*lde];

                    mat[0 + 0*4] = al11*ak11 - el11*ek11;
                    mat[0 + 1*4] = al21*ak11;
                    mat[1 + 0*4] = al12*ak11 - el12*ek11;
                    mat[1 + 1*4] = al22*ak11 - el22*ek11;

                    rhs[0] = x[kl + ll*ldx];
                    rhs[1] = x[kl + lh*ldx];
                } else {
                    dimmat = 4;
                    ak11 = a[kl + kl*lda]; ak12 = a[kl + kh*lda];
                    ak21 = a[kh + kl*lda]; ak22 = a[kh + kh*lda];
                    al11 = a[ll + ll*lda]; al12 = a[ll + lh*lda];
                    al21 = a[lh + ll*lda]; al22 = a[lh + lh*lda];
                    ek11 = e[kl + kl*lde]; ek12 = e[kl + kh*lde]; ek22 = e[kh + kh*lde];
                    el11 = e[ll + ll*lde]; el12 = e[ll + lh*lde]; el22 = e[lh + lh*lde];

                    mat[0 + 0*4] = al11*ak11 - el11*ek11;
                    mat[0 + 1*4] = al11*ak21;
                    mat[0 + 2*4] = al21*ak11;
                    mat[0 + 3*4] = al21*ak21;

                    mat[1 + 0*4] = al11*ak12 - el11*ek12;
                    mat[1 + 1*4] = al11*ak22 - el11*ek22;
                    mat[1 + 2*4] = al21*ak12;
                    mat[1 + 3*4] = al21*ak22;

                    mat[2 + 0*4] = al12*ak11 - el12*ek11;
                    mat[2 + 1*4] = al12*ak21;
                    mat[2 + 2*4] = al22*ak11 - el22*ek11;
                    mat[2 + 3*4] = al22*ak21;

                    mat[3 + 0*4] = al12*ak12 - el12*ek12;
                    mat[3 + 1*4] = al12*ak22 - el12*ek22;
                    mat[3 + 2*4] = al22*ak12 - el22*ek12;
                    mat[3 + 3*4] = al22*ak22 - el22*ek22;

                    rhs[0] = x[kl + ll*ldx];
                    if (kl == ll) {
                        rhs[1] = x[kl + kh*ldx];
                    } else {
                        rhs[1] = x[kh + ll*ldx];
                    }
                    rhs[2] = x[kl + lh*ldx];
                    rhs[3] = x[kh + lh*ldx];
                }

                i32 info1;
                mb02uv(dimmat, mat, 4, piv1, piv2, &info1);
                if (info1 != 0) *info = 1;

                f64 scale1;
                mb02uu(dimmat, mat, 4, rhs, piv1, piv2, &scale1);

                if (scale1 != ONE) {
                    for (i32 i = 0; i < n; i++) {
                        i32 nn = n;
                        i32 one = 1;
                        SLC_DSCAL(&nn, &scale1, &x[0 + i*ldx], &one);
                    }
                    *scale = (*scale) * scale1;
                }

                if (lb == 1 && kb == 1) {
                    x[kl + ll*ldx] = rhs[0];
                } else if (lb == 1 && kb == 2) {
                    x[kl + ll*ldx] = rhs[0];
                    x[kh + ll*ldx] = rhs[1];
                } else if (lb == 2 && kb == 1) {
                    x[kl + ll*ldx] = rhs[0];
                    x[kl + lh*ldx] = rhs[1];
                } else {
                    x[kl + ll*ldx] = rhs[0];
                    x[kh + ll*ldx] = rhs[1];
                    x[kl + lh*ldx] = rhs[2];
                    x[kh + lh*ldx] = rhs[3];
                }

                if (kl < ll) {
                    i32 m_len = kb;
                    i32 n_len = lb;
                    i32 k_len = lb;
                    f64 alpha = ONE;
                    f64 beta = ZERO;
                    i32 ldtm = 2;
                    SLC_DGEMM("N", "N", &m_len, &n_len, &k_len, &alpha,
                              &x[kl + ll*ldx], &ldx, &a[ll + ll*lda], &lda, &beta, tm, &ldtm);

                    i32 rows = lh - kh;
                    alpha = MONE;
                    beta = ONE;
                    SLC_DGEMM("T", "N", &rows, &n_len, &m_len, &alpha,
                              &a[kl + (kh+1)*lda], &lda, tm, &ldtm, &beta, &x[(kh+1) + ll*ldx], &ldx);

                    if (lb == 2) {
                        for (i32 i = 0; i < kb; i++) {
                            tm[i] = x[kl+i + ll*ldx];
                        }
                        for (i32 i = 0; i < kb; i++) {
                            tm[i] *= e[ll + ll*lde];
                        }
                    }

                    for (i32 i = 0; i < kb; i++) {
                        f64 dot = ZERO;
                        for (i32 j = 0; j < lb; j++) {
                            dot += x[(kl+i) + (ll+j)*ldx] * e[ll+j + lh*lde];
                        }
                        tm[i + (lb-1)*2] = dot;
                    }

                    rows = lh - kh;
                    alpha = ONE;
                    SLC_DGEMM("T", "N", &rows, &n_len, &m_len, &alpha,
                              &e[kl + (kh+1)*lde], &lde, tm, &ldtm, &beta, &x[(kh+1) + ll*ldx], &ldx);
                }
            }
        }
    } else {
        i32 ll = n;

        while (ll > 0) {
            i32 lh = ll - 1;
            i32 lb;
            if (lh == 0) {
                lb = 1;
            } else {
                if (a[ll-1 + (ll-2)*lda] != ZERO) {
                    lb = 2;
                } else {
                    lb = 1;
                }
            }
            ll = ll - lb;

            if (lh < n - 1) {
                for (i32 i = ll; i <= lh; i++) {
                    for (i32 j = lh + 1; j < n; j++) {
                        x[j + i*ldx] = x[i + j*ldx];
                    }
                }
            }

            i32 kl = lh + 1;

            while (kl > 0) {
                i32 kh = kl - 1;
                i32 kb;
                if (kh == 0) {
                    kb = 1;
                } else {
                    if (a[kl-1 + (kl-2)*lda] != ZERO) {
                        kb = 2;
                    } else {
                        kb = 1;
                    }
                }
                kl = kl - kb;

                if (kh < n - 1) {
                    i32 m_len = kb;
                    i32 n_len = lb;
                    i32 k_len = n - kh - 1;
                    f64 alpha = ONE;
                    f64 beta = ZERO;
                    i32 ldtm = 2;
                    SLC_DGEMM("N", "N", &m_len, &n_len, &k_len, &alpha,
                              &a[kl + (kh+1)*lda], &lda, &x[(kh+1) + ll*ldx], &ldx, &beta, tm, &ldtm);

                    i32 rows = lh - kl + 1;
                    alpha = MONE;
                    beta = ONE;
                    SLC_DGEMM("N", "T", &m_len, &rows, &n_len, &alpha,
                              tm, &ldtm, &a[kl + ll*lda], &lda, &beta, &x[kl + kl*ldx], &ldx);

                    alpha = ONE;
                    beta = ZERO;
                    SLC_DGEMM("N", "N", &m_len, &n_len, &k_len, &alpha,
                              &e[kl + (kh+1)*lde], &lde, &x[(kh+1) + ll*ldx], &ldx, &beta, tm, &ldtm);

                    rows = ll - kl + 1;
                    beta = ONE;
                    SLC_DGEMM("N", "T", &m_len, &rows, &n_len, &alpha,
                              tm, &ldtm, &e[kl + ll*lde], &lde, &beta, &x[kl + kl*ldx], &ldx);

                    if (lb == 2) {
                        for (i32 i = 0; i < kb; i++) {
                            x[kl+i + lh*ldx] += tm[i + 1*2] * e[lh + lh*lde];
                        }
                    }
                }

                i32 dimmat;
                f64 ak11, ak12, ak21, ak22, al11, al12, al21, al22;
                f64 ek11, ek12, ek22, el11, el12, el22;

                if (kb == 1 && lb == 1) {
                    dimmat = 1;
                    mat[0] = a[ll + ll*lda] * a[kl + kl*lda] - e[ll + ll*lde] * e[kl + kl*lde];
                    rhs[0] = x[kl + ll*ldx];
                } else if (kb == 2 && lb == 1) {
                    dimmat = 2;
                    ak11 = a[kl + kl*lda]; ak12 = a[kl + kh*lda];
                    ak21 = a[kh + kl*lda]; ak22 = a[kh + kh*lda];
                    al11 = a[ll + ll*lda];
                    ek11 = e[kl + kl*lde]; ek12 = e[kl + kh*lde]; ek22 = e[kh + kh*lde];
                    el11 = e[ll + ll*lde];

                    mat[0 + 0*4] = al11*ak11 - el11*ek11;
                    mat[0 + 1*4] = al11*ak12 - el11*ek12;
                    mat[1 + 0*4] = al11*ak21;
                    mat[1 + 1*4] = al11*ak22 - el11*ek22;

                    rhs[0] = x[kl + ll*ldx];
                    rhs[1] = x[kh + ll*ldx];
                } else if (kb == 1 && lb == 2) {
                    dimmat = 2;
                    ak11 = a[kl + kl*lda];
                    al11 = a[ll + ll*lda]; al12 = a[ll + lh*lda];
                    al21 = a[lh + ll*lda]; al22 = a[lh + lh*lda];
                    ek11 = e[kl + kl*lde];
                    el11 = e[ll + ll*lde]; el12 = e[ll + lh*lde]; el22 = e[lh + lh*lde];

                    mat[0 + 0*4] = al11*ak11 - el11*ek11;
                    mat[0 + 1*4] = al12*ak11 - el12*ek11;
                    mat[1 + 0*4] = al21*ak11;
                    mat[1 + 1*4] = al22*ak11 - el22*ek11;

                    rhs[0] = x[kl + ll*ldx];
                    rhs[1] = x[kl + lh*ldx];
                } else {
                    dimmat = 4;
                    ak11 = a[kl + kl*lda]; ak12 = a[kl + kh*lda];
                    ak21 = a[kh + kl*lda]; ak22 = a[kh + kh*lda];
                    al11 = a[ll + ll*lda]; al12 = a[ll + lh*lda];
                    al21 = a[lh + ll*lda]; al22 = a[lh + lh*lda];
                    ek11 = e[kl + kl*lde]; ek12 = e[kl + kh*lde]; ek22 = e[kh + kh*lde];
                    el11 = e[ll + ll*lde]; el12 = e[ll + lh*lde]; el22 = e[lh + lh*lde];

                    mat[0 + 0*4] = al11*ak11 - el11*ek11;
                    mat[0 + 1*4] = al11*ak12 - el11*ek12;
                    mat[0 + 2*4] = al12*ak11 - el12*ek11;
                    mat[0 + 3*4] = al12*ak12 - el12*ek12;

                    mat[1 + 0*4] = al11*ak21;
                    mat[1 + 1*4] = al11*ak22 - el11*ek22;
                    mat[1 + 2*4] = al12*ak21;
                    mat[1 + 3*4] = al12*ak22 - el12*ek22;

                    mat[2 + 0*4] = al21*ak11;
                    mat[2 + 1*4] = al21*ak12;
                    mat[2 + 2*4] = al22*ak11 - el22*ek11;
                    mat[2 + 3*4] = al22*ak12 - el22*ek12;

                    mat[3 + 0*4] = al21*ak21;
                    mat[3 + 1*4] = al21*ak22;
                    mat[3 + 2*4] = al22*ak21;
                    mat[3 + 3*4] = al22*ak22 - el22*ek22;

                    rhs[0] = x[kl + ll*ldx];
                    if (kl == ll) {
                        rhs[1] = x[kl + kh*ldx];
                    } else {
                        rhs[1] = x[kh + ll*ldx];
                    }
                    rhs[2] = x[kl + lh*ldx];
                    rhs[3] = x[kh + lh*ldx];
                }

                i32 info1;
                mb02uv(dimmat, mat, 4, piv1, piv2, &info1);
                if (info1 != 0) *info = 1;

                f64 scale1;
                mb02uu(dimmat, mat, 4, rhs, piv1, piv2, &scale1);

                if (scale1 != ONE) {
                    for (i32 i = 0; i < n; i++) {
                        i32 nn = n;
                        i32 one = 1;
                        SLC_DSCAL(&nn, &scale1, &x[0 + i*ldx], &one);
                    }
                    *scale = (*scale) * scale1;
                }

                if (lb == 1 && kb == 1) {
                    x[kl + ll*ldx] = rhs[0];
                } else if (lb == 1 && kb == 2) {
                    x[kl + ll*ldx] = rhs[0];
                    x[kh + ll*ldx] = rhs[1];
                } else if (lb == 2 && kb == 1) {
                    x[kl + ll*ldx] = rhs[0];
                    x[kl + lh*ldx] = rhs[1];
                } else {
                    x[kl + ll*ldx] = rhs[0];
                    x[kh + ll*ldx] = rhs[1];
                    x[kl + lh*ldx] = rhs[2];
                    x[kh + lh*ldx] = rhs[3];
                }

                if (kl < ll) {
                    i32 m_len = kb;
                    i32 n_len = lb;
                    i32 k_len = kb;
                    f64 alpha = ONE;
                    f64 beta = ZERO;
                    i32 ldtm = 2;
                    SLC_DGEMM("N", "N", &m_len, &n_len, &k_len, &alpha,
                              &a[kl + kl*lda], &lda, &x[kl + ll*ldx], &ldx, &beta, tm, &ldtm);

                    i32 cols = ll - kl;
                    alpha = MONE;
                    beta = ONE;
                    SLC_DGEMM("N", "T", &m_len, &cols, &n_len, &alpha,
                              tm, &ldtm, &a[kl + ll*lda], &lda, &beta, &x[kl + kl*ldx], &ldx);

                    SLC_DGEMV("T", &m_len, &n_len, (f64[]){ONE}, &x[kl + ll*ldx], &ldx,
                              &e[kl + kl*lde], &lde, (f64[]){ZERO}, tm, &ldtm);

                    if (kb == 2) {
                        for (i32 j = 0; j < lb; j++) {
                            tm[1 + j*2] = x[kh + (ll+j)*ldx];
                        }
                        for (i32 j = 0; j < lb; j++) {
                            tm[1 + j*2] *= e[kh + kh*lde];
                        }
                    }

                    cols = ll - kl;
                    alpha = ONE;
                    SLC_DGEMM("N", "T", &m_len, &cols, &n_len, &alpha,
                              tm, &ldtm, &e[kl + ll*lde], &lde, &beta, &x[kl + kl*ldx], &ldx);
                }
            }
        }
    }
}
