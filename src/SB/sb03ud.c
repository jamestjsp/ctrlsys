// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file sb03ud.c
 * @brief Solve discrete-time Lyapunov equation with conditioning/error bounds.
 *
 * Solves: op(A)'*X*op(A) - X = scale*C
 * where op(A) = A or A', C and X are N-by-N symmetric matrices.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

static int select_none(const f64* wr, const f64* wi) {
    (void)wr;
    (void)wi;
    return 0;
}

void sb03ud(
    const char* job,
    const char* fact,
    const char* trana,
    const char* uplo,
    const char* lyapun,
    i32 n,
    f64* scale,
    f64* a,
    i32 lda,
    f64* t,
    i32 ldt,
    f64* u,
    i32 ldu,
    f64* c,
    i32 ldc,
    f64* x,
    i32 ldx,
    f64* sepd,
    f64* rcond,
    f64* ferr,
    f64* wr,
    f64* wi,
    i32* iwork,
    f64* dwork,
    i32 ldwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 HALF = 0.5;
    char job_c = (char)toupper((unsigned char)job[0]);
    char fact_c = (char)toupper((unsigned char)fact[0]);
    char trana_c = (char)toupper((unsigned char)trana[0]);
    char uplo_c = (char)toupper((unsigned char)uplo[0]);
    char lyapun_c = (char)toupper((unsigned char)lyapun[0]);

    bool jobx = (job_c == 'X');
    bool jobs = (job_c == 'S');
    bool jobc = (job_c == 'C');
    bool jobe = (job_c == 'E');
    bool joba = (job_c == 'A');
    bool nofact = (fact_c == 'N');
    bool notrna = (trana_c == 'N');
    bool lower = (uplo_c == 'L');
    bool update = (lyapun_c == 'O');

    i32 nn = n * n;
    i32 ldw;

    if (jobx) {
        if (nofact) {
            ldw = nn > 3 * n ? nn : 3 * n;
            ldw = ldw > 1 ? ldw : 1;
        } else {
            ldw = nn > 2 * n ? nn : 2 * n;
            ldw = ldw > 1 ? ldw : 1;
        }
    } else if (jobs) {
        ldw = 2 * nn > 3 ? 2 * nn : 3;
    } else if (jobc) {
        ldw = (2 * nn > 3 ? 2 * nn : 3) + nn;
    } else {
        ldw = (2 * nn > 3 ? 2 * nn : 3) + nn + 2 * n;
    }

    *info = 0;

    if (!jobx && !jobs && !jobc && !jobe && !joba) {
        *info = -1;
    } else if (!nofact && fact_c != 'F') {
        *info = -2;
    } else if (!notrna && trana_c != 'T' && trana_c != 'C') {
        *info = -3;
    } else if (!lower && uplo_c != 'U') {
        *info = -4;
    } else if (!update && lyapun_c != 'R') {
        *info = -5;
    } else if (n < 0) {
        *info = -6;
    } else if ((jobc || jobe) && (*scale < ZERO || *scale > ONE)) {
        *info = -7;
    } else if (lda < 1 || (lda < n && ((update && !jobx) || nofact))) {
        *info = -9;
    } else if (ldt < (n > 1 ? n : 1)) {
        *info = -11;
    } else if (ldu < 1 || (ldu < n && update)) {
        *info = -13;
    } else if (ldc < 1 || (!jobs && ldc < n)) {
        *info = -15;
    } else if (ldx < 1 || (!jobs && ldx < n)) {
        *info = -17;
    } else if (ldwork < ldw) {
        *info = -25;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("SB03UD", &neginfo);
        return;
    }

    if (n == 0) {
        if (jobx || joba)
            *scale = ONE;
        if (jobc || joba)
            *rcond = ONE;
        if (jobe || joba)
            *ferr = ZERO;
        dwork[0] = ONE;
        return;
    }

    char cfact;

    if (nofact) {
        SLC_DLACPY("F", &n, &n, a, &lda, t, &ldt);
        char sjob = update ? 'V' : 'N';
        i32 sdim_dummy = 0;
        i32* bwork = (i32*)malloc(n * sizeof(i32));
        if (!bwork) {
            *info = -25;
            return;
        }
        SLC_DGEES(&sjob, "N", select_none, &n, t, &ldt, &sdim_dummy,
                  wr, wi, u, &ldu, dwork, &ldwork, bwork, info);
        free(bwork);
        if (*info > 0) {
            return;
        }
        ldw = ldw > (i32)dwork[0] ? ldw : (i32)dwork[0];
        cfact = 'F';
    } else {
        cfact = fact_c;
    }

    if (jobx || joba) {
        SLC_DLACPY(uplo, &n, &n, c, &ldc, x, &ldx);

        if (update) {
            i32 dummy_info = 0;
            mb01ru(uplo, "T", n, n, ZERO, ONE, x, ldx, u, ldu, x, ldx, dwork, ldwork, &dummy_info);
            i32 ldxp1 = ldx + 1;
            SLC_DSCAL(&n, &HALF, x, &ldxp1);
        }

        ma02ed(uplo_c, n, x, ldx);

        i32 solver_info = 0;
        sb03mx(trana, n, t, ldt, x, ldx, scale, dwork, &solver_info);
        if (solver_info > 0) {
            *info = n + 1;
        }

        if (update) {
            i32 dummy_info = 0;
            mb01ru(uplo, "N", n, n, ZERO, ONE, x, ldx, u, ldu, x, ldx, dwork, ldwork, &dummy_info);
            i32 ldxp1 = ldx + 1;
            SLC_DSCAL(&n, &HALF, x, &ldxp1);

            ma02ed(uplo_c, n, x, ldx);
        }
    }

    if (jobs) {
        f64 thnorm;
        sb03sy("S", trana, lyapun, n, t, ldt, u, ldu, dwork, 1, sepd, &thnorm,
               iwork, dwork, ldwork, info);
    } else if (!jobx) {
        char jobl;
        if (joba) {
            jobl = 'B';
        } else {
            jobl = job_c;
        }
        sb03sd(&jobl, &cfact, trana, uplo, lyapun, n, *scale, a, lda, t, ldt,
               u, ldu, c, ldc, x, ldx, sepd, rcond, ferr, iwork, dwork, ldwork, info);
        ldw = ldw > (i32)dwork[0] ? ldw : (i32)dwork[0];
    }

    dwork[0] = (f64)ldw;
}
