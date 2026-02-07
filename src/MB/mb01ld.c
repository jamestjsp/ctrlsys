/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>
#include <stdlib.h>

void mb01ld(const char* uplo_str, const char* trans_str, i32 m, i32 n,
            f64 alpha, f64 beta, f64* r, i32 ldr, const f64* a, i32 lda,
            f64* x, i32 ldx, f64* dwork, i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    i32 int1 = 1;

    char uplo = uplo_str[0];
    char trans = trans_str[0];
    bool upper = (uplo == 'U' || uplo == 'u');
    bool nottra = (trans == 'N' || trans == 'n');
    bool ltrans = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');

    *info = 0;

    if (!upper && uplo != 'L' && uplo != 'l') {
        *info = -1;
    } else if (!nottra && !ltrans) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ldr < (m > 1 ? m : 1)) {
        *info = -8;
    } else if (lda < 1 || (ltrans && lda < n) || (nottra && lda < m)) {
        *info = -10;
    } else if (ldx < (n > 1 ? n : 1) ||
               (ldx < m && upper && ldwork < m * (n - 1))) {
        *info = -12;
    } else if (ldwork < 0 ||
               (beta != zero && m > 1 && n > 1 && ldwork < n)) {
        *info = -14;
    }

    if (*info != 0) {
        return;
    }

    if (m <= 0) {
        return;
    }

    i32 m2 = (2 < m) ? 2 : m;
    i32 ii, jj;

    if (beta == zero || n <= 1) {
        if (upper) {
            ii = 0;
            jj = m2 - 1;
        } else {
            ii = m2 - 1;
            jj = 0;
        }

        if (alpha == zero) {
            i32 mm1 = m - 1;
            SLC_DLASET(&uplo, &mm1, &mm1, &zero, &zero, &r[ii + jj * ldr], &ldr);
        } else {
            if (alpha != one) {
                i32 mm1 = m - 1;
                i32 type = 0;
                SLC_DLASCL(&uplo, &type, &type, &one, &alpha, &mm1, &mm1,
                           &r[ii + jj * ldr], &ldr, info);
            }
        }
        return;
    }

    if (ldwork >= m * (n - 1)) {
        if (upper) {
            ii = 0;
            jj = m2 - 1;
        } else {
            ii = m2 - 1;
            jj = 0;
        }

        i32 nm1 = n - 1;

        if (nottra) {
            SLC_DLACPY("F", &m, &nm1, &a[ii * lda], &lda, dwork, &m);
            SLC_DTRMM("R", &uplo, "N", "N", &m, &nm1, &one,
                      &x[ii + jj * ldx], &ldx, dwork, &m);
            mb01kd(&uplo, &trans, m, nm1, beta, dwork, m, &a[jj * lda], lda,
                   alpha, r, ldr, info);
        } else {
            SLC_DLACPY("F", &nm1, &m, &a[jj], &lda, dwork, &nm1);
            SLC_DTRMM("L", &uplo, "N", "N", &nm1, &m, &one,
                      &x[ii + jj * ldx], &ldx, dwork, &nm1);
            mb01kd(&uplo, &trans, m, nm1, beta, &a[ii], lda, dwork, nm1,
                   alpha, r, ldr, info);
        }
    } else {
        if (nottra) {
            if (upper) {
                for (i32 j = 0; j < n - 1; j++) {
                    for (i32 i = 0; i < j; i++) {
                        dwork[i] = x[i + j * ldx];
                    }
                    dwork[j] = zero;
                    for (i32 i = j + 1; i < n; i++) {
                        dwork[i] = -x[j + i * ldx];
                    }
                    SLC_DGEMV(&trans, &m, &n, &one, a, &lda, dwork, &int1,
                              &zero, &x[j * ldx], &int1);
                }

                for (i32 i = 0; i < n - 1; i++) {
                    dwork[i] = x[i + (n - 1) * ldx];
                }
                i32 nm1 = n - 1;
                SLC_DGEMV(&trans, &m, &nm1, &one, a, &lda, dwork, &int1,
                          &zero, &x[(n - 1) * ldx], &int1);

                for (i32 i = 0; i < m - 1; i++) {
                    for (i32 jj = 0; jj < n; jj++) {
                        dwork[jj] = x[i + jj * ldx];
                    }
                    i32 mmi = m - i - 1;
                    SLC_DGEMV(&trans, &mmi, &n, &beta, &a[(i + 1) * lda], &lda,
                              dwork, &int1, &alpha, &r[i + (i + 1) * ldr], &ldr);
                }
            } else {
                for (i32 i = 0; i < n - 1; i++) {
                    for (i32 j = 0; j < i; j++) {
                        dwork[j] = x[i + j * ldx];
                    }
                    dwork[i] = zero;
                    for (i32 j = i + 1; j < n; j++) {
                        dwork[j] = -x[j + i * ldx];
                    }
                    SLC_DGEMV(&trans, &m, &n, &one, a, &lda, dwork, &int1,
                              &zero, &x[i], &ldx);
                }

                for (i32 j = 0; j < n - 1; j++) {
                    dwork[j] = x[(n - 1) + j * ldx];
                }
                i32 nm1 = n - 1;
                SLC_DGEMV(&trans, &m, &nm1, &one, a, &lda, dwork, &int1,
                          &zero, &x[(n - 1)], &ldx);

                for (i32 j = 0; j < m - 1; j++) {
                    for (i32 ii = 0; ii < n; ii++) {
                        dwork[ii] = x[ii + j * ldx];
                    }
                    i32 mmj = m - j - 1;
                    SLC_DGEMV(&trans, &mmj, &n, &beta, &a[(j + 1)], &lda,
                              dwork, &int1, &alpha, &r[(j + 1) + j * ldr], &int1);
                }
            }
        } else {
            if (upper) {
                for (i32 j = 0; j < n - 1; j++) {
                    for (i32 i = 0; i < j; i++) {
                        dwork[i] = x[i + j * ldx];
                    }
                    dwork[j] = zero;
                    for (i32 i = j + 1; i < n; i++) {
                        dwork[i] = -x[j + i * ldx];
                    }
                    SLC_DGEMV(&trans, &n, &m, &one, a, &lda, dwork, &int1,
                              &zero, &x[j * ldx], &int1);
                }

                i32 nm1 = n - 1;
                for (i32 i = 0; i < n - 1; i++) {
                    dwork[i] = x[i + (n - 1) * ldx];
                }
                SLC_DGEMV(&trans, &nm1, &m, &one, a, &lda, dwork, &int1,
                          &zero, &x[(n - 1) * ldx], &int1);

                for (i32 i = 0; i < m - 1; i++) {
                    for (i32 jj = 0; jj < n; jj++) {
                        dwork[jj] = x[i + jj * ldx];
                    }
                    i32 mmi = m - i - 1;
                    SLC_DGEMV(&trans, &n, &mmi, &beta, &a[(i + 1) * lda], &lda,
                              dwork, &int1, &alpha, &r[i + (i + 1) * ldr], &ldr);
                }
            } else {
                for (i32 i = 0; i < n - 1; i++) {
                    for (i32 j = 0; j < i; j++) {
                        dwork[j] = x[i + j * ldx];
                    }
                    dwork[i] = zero;
                    for (i32 j = i + 1; j < n; j++) {
                        dwork[j] = -x[j + i * ldx];
                    }
                    SLC_DGEMV(&trans, &n, &m, &one, a, &lda, dwork, &int1,
                              &zero, &x[i], &ldx);
                }

                i32 nm1 = n - 1;
                for (i32 j = 0; j < n - 1; j++) {
                    dwork[j] = x[(n - 1) + j * ldx];
                }
                SLC_DGEMV(&trans, &nm1, &m, &one, a, &lda, dwork, &int1,
                          &zero, &x[(n - 1)], &ldx);

                for (i32 j = 0; j < m - 1; j++) {
                    for (i32 ii = 0; ii < n; ii++) {
                        dwork[ii] = x[ii + j * ldx];
                    }
                    i32 mmj = m - j - 1;
                    SLC_DGEMV(&trans, &n, &mmj, &beta, &a[(j + 1) * lda], &lda,
                              dwork, &int1, &alpha, &r[(j + 1) + j * ldr], &int1);
                }
            }
        }
    }
}
