/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>
#include <string.h>

void sg03bw(
    const char* trans,
    const i32 m, const i32 n,
    const f64* a, const i32 lda,
    const f64* c, const i32 ldc,
    const f64* e, const i32 lde,
    const f64* d, const i32 ldd,
    f64* x, const i32 ldx,
    f64* scale,
    i32* info
)
{
    const f64 mone = -1.0, one = 1.0, zero = 0.0;

    bool notrns;
    f64 scale1;
    i32 dimmat, i, info1, j, ma, mai, maj, mb, me;
    f64 mat[16];
    f64 rhs[4], tm[4];
    i32 piv1[4], piv2[4];
    i32 int1 = 1, int0 = 0, ldmat = 4;

    notrns = (trans[0] == 'N' || trans[0] == 'n');

    *info = 0;

    if (!notrns && !(trans[0] == 'T' || trans[0] == 't')) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (n != 1 && n != 2) {
        *info = -3;
    } else if (lda < (m > 0 ? m : 1)) {
        *info = -5;
    } else if (ldc < (n > 0 ? n : 1)) {
        *info = -7;
    } else if (lde < (m > 0 ? m : 1)) {
        *info = -9;
    } else if (ldd < (n > 0 ? n : 1)) {
        *info = -11;
    } else if (ldx < (m > 0 ? m : 1)) {
        *info = -13;
    }

    if (*info != 0) {
        i32 neg_info = -*info;
        SLC_XERBLA("SG03BW", &neg_info);
        return;
    }

    *scale = one;

    if (m == 0)
        return;

    if (notrns) {
        me = 0;
        while (me != m) {
            ma = me + 1;
            if (ma == m) {
                me = m;
                mb = 1;
            } else {
                if (a[ma + (ma - 1) * lda] != zero) {
                    me = ma + 1;
                    mb = 2;
                } else {
                    me = ma;
                    mb = 1;
                }
            }

            if (n == 1) {
                dimmat = mb;
                for (i = 0; i < mb; i++) {
                    mai = ma + i;
                    for (j = 0; j < mb; j++) {
                        maj = ma + j;
                        mat[i + j * 4] = c[0] * a[(maj - 1) + (mai - 1) * lda];
                        if (maj <= mai)
                            mat[i + j * 4] += d[0] * e[(maj - 1) + (mai - 1) * lde];
                    }
                    rhs[i] = x[(mai - 1) + 0 * ldx];
                }
            } else {
                dimmat = 2 * mb;
                for (i = 0; i < mb; i++) {
                    mai = ma + i;
                    for (j = 0; j < mb; j++) {
                        maj = ma + j;
                        mat[i + j * 4] = c[0 + 0 * ldc] * a[(maj - 1) + (mai - 1) * lda];
                        mat[(mb + i) + j * 4] = c[0 + 1 * ldc] * a[(maj - 1) + (mai - 1) * lda];
                        mat[i + (mb + j) * 4] = c[1 + 0 * ldc] * a[(maj - 1) + (mai - 1) * lda];
                        mat[(mb + i) + (mb + j) * 4] = c[1 + 1 * ldc] * a[(maj - 1) + (mai - 1) * lda];
                        if (maj <= mai) {
                            mat[i + j * 4] += d[0 + 0 * ldd] * e[(maj - 1) + (mai - 1) * lde];
                            mat[(mb + i) + j * 4] += d[0 + 1 * ldd] * e[(maj - 1) + (mai - 1) * lde];
                            mat[i + (mb + j) * 4] += d[1 + 0 * ldd] * e[(maj - 1) + (mai - 1) * lde];
                            mat[(mb + i) + (mb + j) * 4] += d[1 + 1 * ldd] * e[(maj - 1) + (mai - 1) * lde];
                        }
                    }
                    rhs[i] = x[(mai - 1) + 0 * ldx];
                    rhs[mb + i] = x[(mai - 1) + 1 * ldx];
                }
            }

            SLC_DGETC2(&dimmat, mat, &ldmat, piv1, piv2, &info1);
            if (info1 != 0)
                *info = 1;
            SLC_DGESC2(&dimmat, mat, &ldmat, rhs, piv1, piv2, &scale1);
            if (scale1 != one) {
                *scale = scale1 * (*scale);
                SLC_DLASCL("G", &int0, &int0, &one, &scale1, &m, &n, x, &ldx, &info1);
            }

            SLC_DCOPY(&mb, rhs, &int1, &x[(ma - 1) + 0 * ldx], &int1);
            if (n == 2)
                SLC_DCOPY(&mb, &rhs[mb], &int1, &x[(ma - 1) + 1 * ldx], &int1);

            if (me < m) {
                i32 ldtm = 2;
                SLC_DGEMM("N", "N", &mb, &n, &n, &one, &x[(ma - 1) + 0 * ldx], &ldx, c, &ldc, &zero, tm, &ldtm);
                i32 mme = m - me;
                SLC_DGEMM("T", "N", &mme, &n, &mb, &mone, &a[(ma - 1) + me * lda], &lda, tm, &ldtm, &one, &x[me + 0 * ldx], &ldx);
                SLC_DGEMM("N", "N", &mb, &n, &n, &one, &x[(ma - 1) + 0 * ldx], &ldx, d, &ldd, &zero, tm, &ldtm);
                SLC_DGEMM("T", "N", &mme, &n, &mb, &mone, &e[(ma - 1) + me * lde], &lde, tm, &ldtm, &one, &x[me + 0 * ldx], &ldx);
            }
        }
    } else {
        ma = m + 1;
        while (ma != 1) {
            me = ma - 1;
            if (me == 1) {
                ma = 1;
                mb = 1;
            } else {
                // Check for 2x2 quasitriangular block (ensure me-1 >= 0)
                if (me > 1 && a[(me - 1) + (me - 1 - 1) * lda] != zero) {
                    ma = me - 1;
                    mb = 2;
                } else {
                    ma = me;
                    mb = 1;
                }
            }

            if (n == 1) {
                dimmat = mb;
                for (i = 0; i < mb; i++) {
                    mai = ma + i;
                    for (j = 0; j < mb; j++) {
                        maj = ma + j;
                        mat[i + j * 4] = c[0] * a[(mai - 1) + (maj - 1) * lda];
                        if (maj >= mai)
                            mat[i + j * 4] += d[0] * e[(mai - 1) + (maj - 1) * lde];
                    }
                    rhs[i] = x[(mai - 1) + 0 * ldx];
                }
            } else {
                dimmat = 2 * mb;
                for (i = 0; i < mb; i++) {
                    mai = ma + i;
                    for (j = 0; j < mb; j++) {
                        maj = ma + j;
                        mat[i + j * 4] = c[0 + 0 * ldc] * a[(mai - 1) + (maj - 1) * lda];
                        mat[(mb + i) + j * 4] = c[1 + 0 * ldc] * a[(mai - 1) + (maj - 1) * lda];
                        mat[i + (mb + j) * 4] = c[0 + 1 * ldc] * a[(mai - 1) + (maj - 1) * lda];
                        mat[(mb + i) + (mb + j) * 4] = c[1 + 1 * ldc] * a[(mai - 1) + (maj - 1) * lda];
                        if (maj >= mai) {
                            mat[i + j * 4] += d[0 + 0 * ldd] * e[(mai - 1) + (maj - 1) * lde];
                            mat[(mb + i) + j * 4] += d[1 + 0 * ldd] * e[(mai - 1) + (maj - 1) * lde];
                            mat[i + (mb + j) * 4] += d[0 + 1 * ldd] * e[(mai - 1) + (maj - 1) * lde];
                            mat[(mb + i) + (mb + j) * 4] += d[1 + 1 * ldd] * e[(mai - 1) + (maj - 1) * lde];
                        }
                    }
                    rhs[i] = x[(mai - 1) + 0 * ldx];
                    rhs[mb + i] = x[(mai - 1) + 1 * ldx];
                }
            }

            SLC_DGETC2(&dimmat, mat, &ldmat, piv1, piv2, &info1);
            if (info1 != 0)
                *info = 1;
            SLC_DGESC2(&dimmat, mat, &ldmat, rhs, piv1, piv2, &scale1);
            if (scale1 != one) {
                *scale = scale1 * (*scale);
                SLC_DLASCL("G", &int0, &int0, &one, &scale1, &m, &n, x, &ldx, &info1);
            }

            SLC_DCOPY(&mb, rhs, &int1, &x[(ma - 1) + 0 * ldx], &int1);
            if (n == 2)
                SLC_DCOPY(&mb, &rhs[mb], &int1, &x[(ma - 1) + 1 * ldx], &int1);

            if (ma > 1) {
                i32 ldtm = 2;
                i32 ma1 = ma - 1;
                SLC_DGEMM("N", "T", &mb, &n, &n, &one, &x[(ma - 1) + 0 * ldx], &ldx, c, &ldc, &zero, tm, &ldtm);
                SLC_DGEMM("N", "N", &ma1, &n, &mb, &mone, &a[0 + (ma - 1) * lda], &lda, tm, &ldtm, &one, x, &ldx);
                SLC_DGEMM("N", "T", &mb, &n, &n, &one, &x[(ma - 1) + 0 * ldx], &ldx, d, &ldd, &zero, tm, &ldtm);
                SLC_DGEMM("N", "N", &ma1, &n, &mb, &mone, &e[0 + (ma - 1) * lde], &lde, tm, &ldtm, &one, x, &ldx);
            }
        }
    }
}
