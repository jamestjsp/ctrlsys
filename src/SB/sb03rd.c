/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB03RD - Solution of continuous-time Lyapunov equations and separation estimation
 *
 * Solves: op(A)' * X + X * op(A) = scale * C
 * where op(A) = A or A^T, C is symmetric.
 * Optionally estimates sep(op(A), -op(A)').
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>

static int select_dummy(const f64* wr, const f64* wi) {
    (void)wr;
    (void)wi;
    return 1;
}

void sb03rd(
    const char* job,
    const char* fact,
    const char* trana,
    const i32 n,
    f64* a,
    const i32 lda,
    f64* u,
    const i32 ldu,
    f64* c,
    const i32 ldc,
    f64* scale,
    f64* sep,
    f64* ferr,
    f64* wr,
    f64* wi,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    i32 int1 = 1;

    char job_c = (char)toupper((unsigned char)job[0]);
    char fact_c = (char)toupper((unsigned char)fact[0]);
    char trana_c = (char)toupper((unsigned char)trana[0]);

    bool wantx = (job_c == 'X');
    bool wantsp = (job_c == 'S');
    bool wantbh = (job_c == 'B');
    bool nofact = (fact_c == 'N');
    bool nota = (trana_c == 'N');

    *info = 0;

    if (!wantsp && !wantbh && !wantx) {
        *info = -1;
    } else if (!nofact && fact_c != 'F') {
        *info = -2;
    } else if (!nota && trana_c != 'T' && trana_c != 'C') {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldu < (n > 1 ? n : 1)) {
        *info = -8;
    } else if ((wantsp && ldc < 1) || (!wantsp && ldc < (n > 1 ? n : 1))) {
        *info = -10;
    }

    i32 minwrk;
    if (wantx) {
        if (nofact) {
            minwrk = (n * n > 3 * n) ? n * n : 3 * n;
        } else {
            minwrk = n * n;
        }
    } else {
        if (nofact) {
            minwrk = (2 * n * n > 3 * n) ? 2 * n * n : 3 * n;
        } else {
            minwrk = 2 * n * n;
        }
    }
    if (n == 0) minwrk = 1;

    if (*info == 0 && ldwork < (minwrk > 1 ? minwrk : 1)) {
        *info = -18;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        *scale = ONE;
        if (wantbh) {
            *ferr = ZERO;
        }
        dwork[0] = ONE;
        return;
    }

    i32 lwa = 0;
    i32 bwork_dummy = 0;

    if (nofact) {
        i32 sdim;
        SLC_DGEES("V", "N", select_dummy, &n, a, &lda, &sdim,
                  wr, wi, u, &ldu, dwork, &ldwork, &bwork_dummy, info);
        if (*info > 0) {
            return;
        }
        lwa = (i32)dwork[0];
    }

    if (!wantsp) {
        char uplo = 'U';
        i32 nn = n;
        i32 mm = n;
        f64 alpha = ZERO;
        f64 beta = ONE;
        i32 info_mb01rd;
        mb01rd(&uplo, "T", mm, nn, alpha, beta, c, ldc, u, ldu, c, ldc, dwork, ldwork, &info_mb01rd);

        for (i32 i = 1; i < n; i++) {
            i32 im1 = i;
            SLC_DCOPY(&im1, &c[i * ldc], &int1, &c[i], &ldc);
        }

        i32 info_sb03my;
        sb03my(trana, n, a, lda, c, ldc, scale, &info_sb03my);
        if (info_sb03my > 0) {
            *info = n + 1;
        }

        mb01rd(&uplo, "N", mm, nn, alpha, beta, c, ldc, u, ldu, c, ldc, dwork, ldwork, &info_mb01rd);

        for (i32 i = 1; i < n; i++) {
            i32 im1 = i;
            SLC_DCOPY(&im1, &c[i * ldc], &int1, &c[i], &ldc);
        }
    }

    if (!wantx) {
        char notra;
        if (nota) {
            notra = 'T';
        } else {
            notra = 'N';
        }

        f64 est = ZERO;
        i32 kase = 0;
        i32 isave[3];
        i32 nn = n * n;
        f64 scalef;

        do {
            SLC_DLACN2(&nn, &dwork[nn], dwork, iwork, &est, &kase, isave);
            if (kase != 0) {
                i32 info_inner;
                if (kase == 1) {
                    sb03my(trana, n, a, lda, dwork, n, &scalef, &info_inner);
                } else {
                    sb03my(&notra, n, a, lda, dwork, n, &scalef, &info_inner);
                }
            }
        } while (kase != 0);

        *sep = scalef / est;

        if (wantbh) {
            f64 eps = SLC_DLAMCH("P");
            f64 anorm = SLC_DLANHS("F", &n, a, &lda, dwork);
            *ferr = eps * anorm / (*sep);
        }
    }

    dwork[0] = (f64)((lwa > minwrk) ? lwa : minwrk);
}
