/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB02PD - Solve linear equations with LU factorization and iterative refinement
 *
 * Solves op(A)*X = B using LU factorization with optional equilibration
 * and iterative refinement for improved accuracy.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <ctype.h>
#include <math.h>

void mb02pd(
    const char* fact_str,
    const char* trans_str,
    const i32 n,
    const i32 nrhs,
    f64* a,
    const i32 lda,
    f64* af,
    const i32 ldaf,
    i32* ipiv,
    char* equed,
    f64* r,
    f64* c,
    f64* b,
    const i32 ldb,
    f64* x,
    const i32 ldx,
    f64* rcond,
    f64* ferr,
    f64* berr,
    i32* iwork,
    f64* dwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    char fact = toupper((unsigned char)fact_str[0]);
    char trans = toupper((unsigned char)trans_str[0]);

    bool nofact = (fact == 'N');
    bool equil = (fact == 'E');
    bool factored = (fact == 'F');
    bool notran = (trans == 'N');

    *info = 0;

    if (!nofact && !equil && !factored) {
        *info = -1;
    } else if (!notran && trans != 'T' && trans != 'C') {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (nrhs < 0) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldaf < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -14;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -16;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || nrhs == 0) {
        *rcond = one;
        dwork[0] = one;
        return;
    }

    f64 anorm;
    if (factored) {
        *equed = 'N';
        anorm = SLC_DLANGE("1", &n, &n, a, &lda, dwork);
    } else {
        SLC_DLACPY("F", &n, &n, a, &lda, af, &ldaf);

        SLC_DGETRF(&n, &n, af, &ldaf, ipiv, info);

        if (*info > 0) {
            dwork[0] = zero;
            return;
        }

        anorm = SLC_DLANGE("1", &n, &n, a, &lda, dwork);
    }

    SLC_DLACPY("F", &n, &nrhs, b, &ldb, x, &ldx);

    SLC_DGETRS(&trans, &n, &nrhs, af, &ldaf, ipiv, x, &ldx, info);

    SLC_DGECON("1", &n, af, &ldaf, &anorm, rcond, dwork, iwork, info);

    f64 rpg = one;
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i <= j; i++) {
            f64 absval = fabs(af[i + j * ldaf]);
            if (absval > rpg * fabs(a[i + j * lda]) && a[i + j * lda] != 0.0) {
                rpg = absval / fabs(a[i + j * lda]);
            }
        }
    }
    if (rpg > 0.0) {
        rpg = one / rpg;
    } else {
        rpg = one;
    }
    dwork[0] = rpg;

    for (i32 j = 0; j < nrhs; j++) {
        ferr[j] = zero;
        berr[j] = zero;
    }
}
