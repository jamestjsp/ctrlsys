/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB03QY - Estimate separation and 1-norm of Theta for continuous-time Lyapunov equation
 *
 * Estimates sep(op(A), -op(A)') and/or the 1-norm of Theta, where
 * op(A) = A or A' and Omega, Theta are operators associated to
 * continuous-time Lyapunov equation: op(A)' * X + X * op(A) = C
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>

void sb03qy(
    const char* job,
    const char* trana,
    const char* lyapun,
    const i32 n,
    const f64* t,
    const i32 ldt,
    const f64* u,
    const i32 ldu,
    const f64* x,
    const i32 ldx,
    f64* sep,
    f64* thnorm,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 HALF = 0.5;

    char job_c = (char)toupper((unsigned char)job[0]);
    char trana_c = (char)toupper((unsigned char)trana[0]);
    char lyapun_c = (char)toupper((unsigned char)lyapun[0]);

    bool wants = (job_c == 'S');
    bool wantt = (job_c == 'T');
    bool notrna = (trana_c == 'N');
    bool update = (lyapun_c == 'O');

    i32 nn = n * n;
    *info = 0;

    if (!wants && !wantt && job_c != 'B') {
        *info = -1;
    } else if (!notrna && trana_c != 'T' && trana_c != 'C') {
        *info = -2;
    } else if (!update && lyapun_c != 'R') {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ldt < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldu < 1 || (update && ldu < n)) {
        *info = -8;
    } else if (ldx < 1 || (!wants && ldx < n)) {
        *info = -10;
    } else if (ldwork < 2 * nn) {
        *info = -15;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("SB03QY", &neginfo);
        return;
    }

    if (n == 0) {
        return;
    }

    i32 itmp = nn;  // 0-based offset for second workspace partition
    char tranat = notrna ? 'T' : 'N';

    i32 kase = 0;
    i32 isave[3] = {0, 0, 0};
    f64 est, scale;
    i32 info2;

    if (!wantt) {
        // Estimate sep(op(A), -op(A)')
        // Workspace: 2*N*N

        kase = 0;

        do {
            SLC_DLACN2(&nn, &dwork[itmp], dwork, iwork, &est, &kase, isave);

            if (kase != 0) {
                // Select the triangular part of symmetric matrix to be used
                f64 norm_upper = SLC_DLANSY("1", "U", &n, dwork, &n, &dwork[itmp]);
                f64 norm_lower = SLC_DLANSY("1", "L", &n, dwork, &n, &dwork[itmp]);
                char uplo = (norm_upper >= norm_lower) ? 'U' : 'L';

                if (update) {
                    // Transform: RHS := U' * RHS * U
                    mb01ru(&uplo, "T", n, n, ZERO, ONE, dwork, n, u, ldu,
                           dwork, n, &dwork[itmp], nn, &info2);
                    i32 np1 = n + 1;
                    SLC_DSCAL(&n, &HALF, dwork, &np1);
                }

                ma02ed(uplo, n, dwork, n);

                if (kase == 1) {
                    // Solve op(T)' * Y + Y * op(T) = scale * RHS
                    sb03my(trana, n, t, ldt, dwork, n, &scale, &info2);
                } else {
                    // Solve op(T) * W + W * op(T)' = scale * RHS
                    sb03my(&tranat, n, t, ldt, dwork, n, &scale, &info2);
                }

                if (info2 > 0) {
                    *info = n + 1;
                }

                if (update) {
                    // Transform back: Z := U * Z * U'
                    mb01ru(&uplo, "N", n, n, ZERO, ONE, dwork, n, u, ldu,
                           dwork, n, &dwork[itmp], nn, &info2);
                    i32 np1 = n + 1;
                    SLC_DSCAL(&n, &HALF, dwork, &np1);

                    ma02ed(uplo, n, dwork, n);
                }
            }
        } while (kase != 0);

        if (est > scale) {
            *sep = scale / est;
        } else {
            f64 bignum = ONE / SLC_DLAMCH("S");
            if (scale < est * bignum) {
                *sep = scale / est;
            } else {
                *sep = bignum;
            }
        }

        // Return if the equation is singular
        if (*sep == ZERO) {
            return;
        }
    }

    if (!wants) {
        // Estimate norm(Theta)
        // Workspace: 2*N*N

        kase = 0;

        do {
            SLC_DLACN2(&nn, &dwork[itmp], dwork, iwork, &est, &kase, isave);

            if (kase != 0) {
                // Select the triangular part of symmetric matrix to be used
                f64 norm_upper = SLC_DLANSY("1", "U", &n, dwork, &n, &dwork[itmp]);
                f64 norm_lower = SLC_DLANSY("1", "L", &n, dwork, &n, &dwork[itmp]);
                char uplo = (norm_upper >= norm_lower) ? 'U' : 'L';

                // Fill in the remaining triangle
                ma02ed(uplo, n, dwork, n);

                // Compute RHS = op(W)' * X + X * op(W)
                char uplo_str[2] = {uplo, '\0'};
                char tranat_str[2] = {tranat, '\0'};
                SLC_DSYR2K(uplo_str, tranat_str, &n, &n, &ONE, dwork, &n,
                           x, &ldx, &ZERO, &dwork[itmp], &n);
                SLC_DLACPY(uplo_str, &n, &n, &dwork[itmp], &n, dwork, &n);

                if (update) {
                    // Transform: RHS := U' * RHS * U
                    mb01ru(&uplo, "T", n, n, ZERO, ONE, dwork, n, u, ldu,
                           dwork, n, &dwork[itmp], nn, &info2);
                    i32 np1 = n + 1;
                    SLC_DSCAL(&n, &HALF, dwork, &np1);
                }

                ma02ed(uplo, n, dwork, n);

                if (kase == 1) {
                    // Solve op(T)' * Y + Y * op(T) = scale * RHS
                    sb03my(trana, n, t, ldt, dwork, n, &scale, &info2);
                } else {
                    // Solve op(T) * W + W * op(T)' = scale * RHS
                    sb03my(&tranat, n, t, ldt, dwork, n, &scale, &info2);
                }

                if (info2 > 0) {
                    *info = n + 1;
                }

                if (update) {
                    // Transform back: Z := U * Z * U'
                    mb01ru(&uplo, "N", n, n, ZERO, ONE, dwork, n, u, ldu,
                           dwork, n, &dwork[itmp], nn, &info2);
                    i32 np1 = n + 1;
                    SLC_DSCAL(&n, &HALF, dwork, &np1);

                    ma02ed(uplo, n, dwork, n);
                }
            }
        } while (kase != 0);

        if (est < scale) {
            *thnorm = est / scale;
        } else {
            f64 bignum = ONE / SLC_DLAMCH("S");
            if (est < scale * bignum) {
                *thnorm = est / scale;
            } else {
                *thnorm = bignum;
            }
        }
    }
}
