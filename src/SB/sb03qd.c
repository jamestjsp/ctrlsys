/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB03QD - Estimate conditioning and forward error bound for continuous-time Lyapunov equation
 *
 * Estimates the conditioning and computes an error bound on the solution of:
 *     op(A)' * X + X * op(A) = scale * C
 * where op(A) = A or A' and C is symmetric.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>

static int select_func(const f64* wr, const f64* wi) {
    (void)wr;
    (void)wi;
    return 0;
}

void sb03qd(
    const char* job,
    const char* fact,
    const char* trana,
    const char* uplo,
    const char* lyapun,
    const i32 n,
    const f64 scale,
    const f64* a,
    const i32 lda,
    f64* t,
    const i32 ldt,
    f64* u,
    const i32 ldu,
    const f64* c,
    const i32 ldc,
    const f64* x,
    const i32 ldx,
    f64* sep,
    f64* rcond,
    f64* ferr,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 THREE = 3.0;

    char job_c = (char)toupper((unsigned char)job[0]);
    char fact_c = (char)toupper((unsigned char)fact[0]);
    char trana_c = (char)toupper((unsigned char)trana[0]);
    char uplo_c = (char)toupper((unsigned char)uplo[0]);
    char lyapun_c = (char)toupper((unsigned char)lyapun[0]);

    bool jobc = (job_c == 'C');
    bool jobe = (job_c == 'E');
    bool jobb = (job_c == 'B');
    bool nofact = (fact_c == 'N');
    bool notrna = (trana_c == 'N');
    bool lower = (uplo_c == 'L');
    bool update = (lyapun_c == 'O');

    i32 nn = n * n;
    i32 ldw;
    if (jobc) {
        ldw = 2 * nn;
    } else {
        ldw = 3 * nn;
    }
    if (!(jobc || update)) {
        ldw = ldw + n - 1;
    }

    *info = 0;

    if (!jobb && !jobc && !jobe) {
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
    } else if (scale < ZERO || scale > ONE) {
        *info = -7;
    } else if (lda < 1 || (lda < n && (update || nofact))) {
        *info = -9;
    } else if (ldt < (n > 1 ? n : 1)) {
        *info = -11;
    } else if (ldu < 1 || (ldu < n && update)) {
        *info = -13;
    } else if (ldc < (n > 1 ? n : 1)) {
        *info = -15;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -17;
    }

    i32 iwrk;
    if (*info == 0) {
        if (nofact) {
            iwrk = (ldw > 5 * n) ? ldw : 5 * n;
        } else {
            iwrk = ldw;
        }
        iwrk = (iwrk > 1) ? iwrk : 1;

        bool lquery = (ldwork == -1);

        char sjob[2] = {update ? 'V' : 'N', '\0'};
        i32 wrkopt = 0;

        if (lquery) {
            if (nofact) {
                i32 sdim;
                i32 bwork[1] = {0};
                f64 query;
                i32 neg1 = -1;
                SLC_DGEES(sjob, "N", select_func, &n, t, &ldt, &sdim,
                          dwork, dwork, u, &ldu, &query, &neg1, bwork, info);
                wrkopt = iwrk > (i32)query + 2 * n ? iwrk : (i32)query + 2 * n;
            } else {
                wrkopt = iwrk;
            }
            if (!update) {
                wrkopt = (wrkopt > 4 * nn) ? wrkopt : 4 * nn;
            }
            dwork[0] = (f64)wrkopt;
            return;
        }

        if (ldwork < iwrk) {
            *info = -23;
        }
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("SB03QD", &neginfo);
        return;
    }

    if (n == 0) {
        if (!jobe) *rcond = ONE;
        if (!jobc) *ferr = ZERO;
        dwork[0] = ONE;
        return;
    }

    f64 xnorm = SLC_DLANSY("1", uplo, &n, x, &ldx, dwork);
    if (xnorm == ZERO) {
        if (!jobe) *rcond = ZERO;
        if (!jobc) *ferr = ZERO;
        dwork[0] = (f64)n;
        return;
    }

    f64 anorm;
    if (nofact || update) {
        anorm = SLC_DLANGE("1", &n, &n, a, &lda, dwork);
    } else {
        anorm = SLC_DLANHS("1", &n, t, &ldt, dwork);
    }

    i32 wrkopt = 0;

    if (anorm == ONE) {
        if (nofact || update) {
            SLC_DLACPY("F", &n, &n, a, &lda, dwork, &n);
        } else {
            SLC_DLACPY("F", &n, &n, t, &ldt, dwork, &n);
            if (n > 2) {
                i32 nm2 = n - 2;
                SLC_DLASET("L", &nm2, &nm2, &ZERO, &ZERO, &dwork[2], &n);
            }
        }
        dwork[nn] = ONE;
        i32 np1 = n + 1;
        i32 int0 = 0;
        f64 neg1 = -ONE;
        SLC_DAXPY(&n, &neg1, &dwork[nn], &int0, dwork, &np1);

        if (SLC_DLANGE("M", &n, &n, dwork, &n, dwork) == ZERO) {
            if (!jobe) {
                *sep = TWO;
                *rcond = ONE;
            }
            if (jobc) {
                dwork[0] = (f64)(nn + 1);
                return;
            } else {
                SLC_DLACPY(uplo, &n, &n, x, &ldx, dwork, &n);
                f64 factor = -scale / TWO;

                if (lower) {
                    for (i32 j = 0; j < n; j++) {
                        i32 len = n - j;
                        SLC_DAXPY(&len, &factor, &c[j + j * ldc], &(i32){1},
                                  &dwork[j * n + j], &(i32){1});
                    }
                } else {
                    for (i32 j = 0; j < n; j++) {
                        i32 len = j + 1;
                        SLC_DAXPY(&len, &factor, &c[j * ldc], &(i32){1},
                                  &dwork[j * n], &(i32){1});
                    }
                }

                f64 res_norm = SLC_DLANSY("1", uplo, &n, dwork, &n, &dwork[nn]);
                *ferr = (res_norm / xnorm < ONE) ? res_norm / xnorm : ONE;
                dwork[0] = (f64)(nn + n);
                return;
            }
        }
    } else if (anorm == ZERO) {
        if (!jobe) {
            *sep = ZERO;
            *rcond = ZERO;
        }
        if (!jobc) {
            *ferr = ONE;
        }
        dwork[0] = (f64)n;
        return;
    }

    f64 cnorm = SLC_DLANSY("1", uplo, &n, c, &ldc, dwork);

    i32 iabs = 0;
    i32 ixbs = iabs + nn;
    i32 ires = ixbs + nn;
    iwrk = ires + nn;

    if (nofact) {
        SLC_DLACPY("F", &n, &n, a, &lda, t, &ldt);
        char sjob[2] = {update ? 'V' : 'N', '\0'};
        i32 sdim;
        i32 bwork[1] = {0};
        i32 ldw_schur = ldwork - 2 * n;
        SLC_DGEES(sjob, "N", select_func, &n, t, &ldt, &sdim,
                  dwork, &dwork[n], u, &ldu, &dwork[2 * n], &ldw_schur, bwork, info);
        if (*info > 0) {
            return;
        }
        wrkopt = (i32)dwork[2 * n] + 2 * n;
    }

    if (!jobe) {
        f64 thnorm;
        sb03qy("B", trana, lyapun, n, t, ldt, u, ldu, x, ldx, sep, &thnorm, iwork, dwork, ldwork, info);

        wrkopt = (wrkopt > 2 * nn) ? wrkopt : 2 * nn;

        if (*sep == ZERO) {
            *rcond = ZERO;
            if (jobb) *ferr = ONE;
            dwork[0] = (f64)wrkopt;
            return;
        }

        f64 tmax = (*sep > xnorm) ? *sep : xnorm;
        tmax = (tmax > anorm) ? tmax : anorm;

        f64 temp, denom;
        if (tmax <= ONE) {
            temp = (*sep) * xnorm;
            denom = (scale * cnorm) + ((*sep) * anorm) * thnorm;
        } else {
            temp = ((*sep) / tmax) * (xnorm / tmax);
            denom = ((scale / tmax) * (cnorm / tmax)) +
                    (((*sep) / tmax) * (anorm / tmax)) * thnorm;
        }

        if (temp >= denom) {
            *rcond = ONE;
        } else {
            *rcond = temp / denom;
        }
    }

    if (!jobc) {
        char tranat = notrna ? 'T' : 'N';
        char tranat_str[2] = {tranat, '\0'};
        char uplo_str[2] = {uplo_c, '\0'};
        if (update) {
            SLC_DLACPY(uplo, &n, &n, c, &ldc, &dwork[ires], &n);
            f64 neg_scale = -scale;
            SLC_DSYR2K(uplo_str, tranat_str, &n, &n, &ONE, a, &lda, x, &ldx,
                       &neg_scale, &dwork[ires], &n);
        } else {
            i32 info2;
            mb01ud("R", trana, n, n, ONE, t, ldt, x, ldx, &dwork[ires], n, &info2);
            i32 jj = ires;
            if (lower) {
                for (i32 j = 0; j < n; j++) {
                    i32 len = n - j;
                    f64 neg_scale = -scale;
                    SLC_DAXPY(&len, &ONE, &dwork[jj], &n, &dwork[jj], &(i32){1});
                    SLC_DAXPY(&len, &neg_scale, &c[j + j * ldc], &(i32){1}, &dwork[jj], &(i32){1});
                    jj = jj + n + 1;
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    i32 len = j + 1;
                    f64 neg_scale = -scale;
                    SLC_DAXPY(&len, &ONE, &dwork[ires + j], &n, &dwork[jj], &(i32){1});
                    SLC_DAXPY(&len, &neg_scale, &c[j * ldc], &(i32){1}, &dwork[jj], &(i32){1});
                    jj = jj + n;
                }
            }
        }

        wrkopt = (wrkopt > 3 * nn) ? wrkopt : 3 * nn;

        f64 eps = SLC_DLAMCH("E");
        f64 epsn = eps * (f64)(n + 3);
        f64 temp = eps * THREE * scale;

        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < n; i++) {
                dwork[ixbs + j * n + i] = fabs(x[i + j * ldx]);
            }
        }

        if (lower) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = j; i < n; i++) {
                    dwork[ires + j * n + i] = temp * fabs(c[i + j * ldc]) +
                                              fabs(dwork[ires + j * n + i]);
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i <= j; i++) {
                    dwork[ires + j * n + i] = temp * fabs(c[i + j * ldc]) +
                                              fabs(dwork[ires + j * n + i]);
                }
            }
        }

        if (update) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < n; i++) {
                    dwork[iabs + j * n + i] = fabs(a[i + j * lda]);
                }
            }

            SLC_DSYR2K(uplo_str, tranat_str, &n, &n, &epsn, &dwork[iabs], &n,
                       &dwork[ixbs], &n, &ONE, &dwork[ires], &n);
        } else {
            for (i32 j = 0; j < n; j++) {
                i32 imax = (j + 1 < n) ? j + 1 : n - 1;
                for (i32 i = 0; i <= imax; i++) {
                    dwork[iabs + j * n + i] = fabs(t[i + j * ldt]);
                }
            }

            i32 info2;
            mb01uw("L", trana, n, n, epsn, &dwork[iabs], n, &dwork[ixbs], n,
                   &dwork[iwrk], ldwork - iwrk, &info2);

            i32 jj = ires;
            i32 jx = ixbs;
            if (lower) {
                for (i32 j = 0; j < n; j++) {
                    i32 len = n - j;
                    SLC_DAXPY(&len, &ONE, &dwork[jx], &n, &dwork[jx], &(i32){1});
                    SLC_DAXPY(&len, &ONE, &dwork[jx], &(i32){1}, &dwork[jj], &(i32){1});
                    jj = jj + n + 1;
                    jx = jx + n + 1;
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    i32 len = j + 1;
                    SLC_DAXPY(&len, &ONE, &dwork[ixbs + j], &n, &dwork[jx], &(i32){1});
                    SLC_DAXPY(&len, &ONE, &dwork[jx], &(i32){1}, &dwork[jj], &(i32){1});
                    jj = jj + n;
                    jx = jx + n;
                }
            }

            wrkopt = (wrkopt > 3 * nn + n - 1) ? wrkopt : 3 * nn + n - 1;
        }

        f64 xanorm = SLC_DLANSY("M", uplo, &n, x, &ldx, dwork);

        i32 info2;
        sb03qx(trana, uplo, lyapun, n, xanorm, t, ldt, u, ldu,
               &dwork[ires], n, ferr, iwork, dwork, ires, &info2);

        if (info2 == n + 1) {
            *info = n + 1;
        }
    }

    dwork[0] = (f64)wrkopt;
}
