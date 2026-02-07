/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

static inline i32 max_i32(i32 a, i32 b) { return a > b ? a : b; }

void mb01od(const char* uplo_str, const char* trans_str, i32 n,
            f64 alpha, f64 beta, f64* r, i32 ldr,
            f64* h, i32 ldh, f64* x, i32 ldx,
            const f64* e, i32 lde, f64* dwork, i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 two = 2.0;
    const f64 half = 0.5;

    char uplo = uplo_str[0];
    char trans = trans_str[0];
    bool luplo = (uplo == 'U' || uplo == 'u');
    bool ltrans = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');

    *info = 0;

    if (!luplo && uplo != 'L' && uplo != 'l') {
        *info = -1;
    } else if (!ltrans && trans != 'N' && trans != 'n') {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ldr < max_i32(1, n)) {
        *info = -7;
    } else if (ldh < max_i32(1, n)) {
        *info = -9;
    } else if (ldx < max_i32(1, n)) {
        *info = -11;
    } else if (lde < max_i32(1, n)) {
        *info = -13;
    } else if ((beta != zero && ldwork < n * n) ||
               (beta == zero && ldwork < 0)) {
        *info = -15;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        return;
    }

    if (beta == zero) {
        if (alpha == zero) {
            const char* uplo_ptr = luplo ? "U" : "L";
            SLC_DLASET(uplo_ptr, &n, &n, &zero, &zero, r, &ldr);
        } else if (alpha != one) {
            const char* uplo_ptr = luplo ? "U" : "L";
            i32 izero = 0;
            SLC_DLASCL(uplo_ptr, &izero, &izero, &one, &alpha, &n, &n, r, &ldr, info);
        }
        return;
    }

    i32 inc1 = 1;
    SLC_DSCAL(&n, &half, x, &(i32){ldx + 1});

    if (!ltrans) {
        if (n > 2) {
            i32 len = n - 2;
            SLC_DSWAP(&len, &h[2 + 0 * ldh], &inc1, &h[2 + 1 * ldh], &(i32){ldh + 1});
        }

        if (luplo) {
            for (i32 j = 0; j < n - 1; j++) {
                i32 j1 = j + 1;
                i32 len_j = j + 1;

                SLC_DCOPY(&len_j, &x[0 + j * ldx], &inc1, &dwork[0 + j * n], &inc1);
                SLC_DTRMV("U", "N", "N", &len_j, h, &ldh, &dwork[0 + j * n], &inc1);

                for (i32 i = 1; i <= j; i++) {
                    dwork[i + j * n] += h[i + 0 * ldh] * x[(i - 1) + j * ldx];
                }
                dwork[j1 + j * n] = h[j1 + 0 * ldh] * x[j + j * ldx];
            }

            SLC_DCOPY(&n, &x[0 + (n - 1) * ldx], &inc1, &dwork[0 + (n - 1) * n], &inc1);
            SLC_DTRMV("U", "N", "N", &n, h, &ldh, &dwork[0 + (n - 1) * n], &inc1);

            for (i32 i = 1; i < n; i++) {
                dwork[i + (n - 1) * n] += h[i + 0 * ldh] * x[(i - 1) + (n - 1) * ldx];
            }
        } else {
            for (i32 j = 0; j < n - 1; j++) {
                i32 j1 = j + 1;
                i32 len_j = j + 1;

                SLC_DCOPY(&len_j, &x[j + 0 * ldx], &ldx, &dwork[0 + j * n], &inc1);
                SLC_DTRMV("U", "N", "N", &len_j, h, &ldh, &dwork[0 + j * n], &inc1);

                for (i32 i = 1; i <= j; i++) {
                    dwork[i + j * n] += h[i + 0 * ldh] * x[j + (i - 1) * ldx];
                }
                dwork[j1 + j * n] = h[j1 + 0 * ldh] * x[j + j * ldx];
            }

            SLC_DCOPY(&n, &x[(n - 1) + 0 * ldx], &ldx, &dwork[0 + (n - 1) * n], &inc1);
            SLC_DTRMV("U", "N", "N", &n, h, &ldh, &dwork[0 + (n - 1) * n], &inc1);

            for (i32 i = 1; i < n; i++) {
                dwork[i + (n - 1) * n] += h[i + 0 * ldh] * x[(n - 1) + (i - 1) * ldx];
            }
        }

        if (n > 2) {
            i32 len = n - 2;
            SLC_DSWAP(&len, &h[2 + 0 * ldh], &inc1, &h[2 + 1 * ldh], &(i32){ldh + 1});
        }
    } else {
        if (luplo) {
            for (i32 j = 0; j < n - 1; j++) {
                i32 j1 = j + 1;
                i32 len_j = j + 1;

                SLC_DCOPY(&len_j, &h[0 + j * ldh], &inc1, &dwork[0 + j * n], &inc1);
                SLC_DTRMV("U", "N", "N", &len_j, x, &ldx, &dwork[0 + j * n], &inc1);
                SLC_DAXPY(&len_j, &h[j1 + j * ldh], &x[0 + j1 * ldx], &inc1, &dwork[0 + j * n], &inc1);
                dwork[j1 + j * n] = h[j1 + j * ldh] * x[j1 + j1 * ldx];
            }

            SLC_DCOPY(&n, &h[0 + (n - 1) * ldh], &inc1, &dwork[0 + (n - 1) * n], &inc1);
            SLC_DTRMV("U", "N", "N", &n, x, &ldx, &dwork[0 + (n - 1) * n], &inc1);
        } else {
            for (i32 j = 0; j < n - 1; j++) {
                i32 j1 = j + 1;
                i32 len_j = j + 1;

                SLC_DCOPY(&len_j, &h[0 + j * ldh], &inc1, &dwork[0 + j * n], &inc1);
                SLC_DTRMV("L", "T", "N", &len_j, x, &ldx, &dwork[0 + j * n], &inc1);
                SLC_DAXPY(&len_j, &h[j1 + j * ldh], &x[j1 + 0 * ldx], &ldx, &dwork[0 + j * n], &inc1);
                dwork[j1 + j * n] = h[j1 + j * ldh] * x[j1 + j1 * ldx];
            }

            SLC_DCOPY(&n, &h[0 + (n - 1) * ldh], &inc1, &dwork[0 + (n - 1) * n], &inc1);
            SLC_DTRMV("L", "T", "N", &n, x, &ldx, &dwork[0 + (n - 1) * n], &inc1);
        }
    }

    mb01oe(uplo_str, trans_str, n, alpha, beta, r, ldr, dwork, n, e, lde, info);

    // Zero out strictly lower triangular part of dwork before reusing it
    // for E*U or E*L' computation (needed because first section leaves leftovers
    // in subdiagonal that would corrupt second mb01oe call for UPLO='L')
    for (i32 j = 0; j < n - 1; j++) {
        for (i32 i = j + 1; i < n; i++) {
            dwork[i + j * n] = zero;
        }
    }

    if (!ltrans) {
        if (luplo) {
            for (i32 j = 0; j < n; j++) {
                i32 len_j = j + 1;
                SLC_DCOPY(&len_j, &x[0 + j * ldx], &inc1, &dwork[0 + j * n], &inc1);
                SLC_DTRMV("U", "N", "N", &len_j, e, &lde, &dwork[0 + j * n], &inc1);
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                i32 len_j = j + 1;
                SLC_DCOPY(&len_j, &x[j + 0 * ldx], &ldx, &dwork[0 + j * n], &inc1);
                SLC_DTRMV("U", "N", "N", &len_j, e, &lde, &dwork[0 + j * n], &inc1);
            }
        }
    } else {
        if (luplo) {
            for (i32 j = 0; j < n; j++) {
                i32 len_j = j + 1;
                SLC_DCOPY(&len_j, &e[0 + j * lde], &inc1, &dwork[0 + j * n], &inc1);
                SLC_DTRMV("U", "N", "N", &len_j, x, &ldx, &dwork[0 + j * n], &inc1);
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                i32 len_j = j + 1;
                SLC_DCOPY(&len_j, &e[0 + j * lde], &inc1, &dwork[0 + j * n], &inc1);
                SLC_DTRMV("L", "T", "N", &len_j, x, &ldx, &dwork[0 + j * n], &inc1);
            }
        }
    }

    mb01oe(uplo_str, trans_str, n, one, beta, r, ldr, h, ldh, dwork, n, info);

    SLC_DSCAL(&n, &two, x, &(i32){ldx + 1});
}
