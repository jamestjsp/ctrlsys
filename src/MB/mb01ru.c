// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdbool.h>

void mb01ru(const char* uplo, const char* trans, i32 m, i32 n,
            f64 alpha, f64 beta, f64* r, i32 ldr, const f64* a, i32 lda,
            f64* x, i32 ldx, f64* dwork, i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 two = 2.0;
    const f64 half = 0.5;

    char uplo_c = (char)toupper((unsigned char)uplo[0]);
    char trans_c = (char)toupper((unsigned char)trans[0]);

    bool luplo = (uplo_c == 'U');
    bool ltrans = (trans_c == 'T' || trans_c == 'C');

    *info = 0;

    if (!luplo && uplo_c != 'L') {
        *info = -1;
    } else if (!ltrans && trans_c != 'N') {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ldr < (m > 1 ? m : 1)) {
        *info = -8;
    } else if (lda < 1 || (ltrans && lda < n) || (!ltrans && lda < m)) {
        *info = -10;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -12;
    } else if ((beta != zero && ldwork < m * n) || (beta == zero && ldwork < 0)) {
        *info = -14;
    }

    if (*info != 0) {
        return;
    }

    if (m == 0) {
        return;
    }

    if (beta == zero || n == 0) {
        if (alpha == zero) {
            SLC_DLASET(&uplo_c, &m, &m, &zero, &zero, r, &ldr);
        } else if (alpha != one) {
            i32 kl = 0, ku = 0;
            SLC_DLASCL(&uplo_c, &kl, &ku, &one, &alpha, &m, &m, r, &ldr, info);
        }
        return;
    }

    i32 ldx1 = ldx + 1;
    SLC_DSCAL(&n, &half, x, &ldx1);

    if (ltrans) {
        SLC_DLACPY("Full", &n, &m, a, &lda, dwork, &n);
        SLC_DTRMM("Left", &uplo_c, "NoTranspose", "Non-unit", &n, &m,
                  &one, x, &ldx, dwork, &n);
        SLC_DSYR2K(&uplo_c, &trans_c, &m, &n, &beta, dwork, &n, a, &lda, &alpha, r, &ldr);
    } else {
        SLC_DLACPY("Full", &m, &n, a, &lda, dwork, &m);
        SLC_DTRMM("Right", &uplo_c, "NoTranspose", "Non-unit", &m, &n,
                  &one, x, &ldx, dwork, &m);
        SLC_DSYR2K(&uplo_c, &trans_c, &m, &n, &beta, dwork, &m, a, &lda, &alpha, r, &ldr);
    }

    SLC_DSCAL(&n, &two, x, &ldx1);
}
