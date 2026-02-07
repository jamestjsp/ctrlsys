/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB02QD - Estimate conditioning and forward error bound for continuous-time
 *          algebraic Riccati equation
 *
 * Estimates the conditioning and computes an error bound on the solution
 * of the real continuous-time matrix algebraic Riccati equation:
 *     op(A)'*X + X*op(A) + Q - X*G*X = 0
 * where op(A) = A or A'.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>
#include <stdbool.h>

static int select_dummy(const f64* wr, const f64* wi) {
    (void)wr;
    (void)wi;
    return 0;
}

void sb02qd(
    const char* job,
    const char* fact,
    const char* trana,
    const char* uplo,
    const char* lyapun,
    const i32 n,
    const f64* a,
    const i32 lda,
    f64* t,
    const i32 ldt,
    f64* u,
    const i32 ldu,
    const f64* g,
    const i32 ldg,
    const f64* q,
    const i32 ldq,
    const f64* x,
    const i32 ldx,
    f64* sep,
    f64* rcond,
    f64* ferr,
    i32* iwork,
    f64* dwork,
    i32 ldwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 FOUR = 4.0;
    const f64 HALF = 0.5;

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

    bool needac = update && !jobc;

    i32 nn = n * n;
    i32 lwa = needac ? nn : 0;
    i32 ldw;

    if (nofact) {
        if (jobc) {
            ldw = (5 * n > 2 * nn) ? 5 * n : 2 * nn;
        } else {
            i32 opt1 = lwa + 5 * n;
            i32 opt2 = 4 * nn;
            ldw = (opt1 > opt2) ? opt1 : opt2;
        }
    } else {
        if (jobc) {
            ldw = 2 * nn;
        } else {
            ldw = 4 * nn;
        }
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
    } else if (lda < 1 || (lda < n && (update || nofact))) {
        *info = -8;
    } else if (ldt < (n > 1 ? n : 1)) {
        *info = -10;
    } else if (ldu < 1 || (ldu < n && update)) {
        *info = -12;
    } else if (ldg < (n > 1 ? n : 1)) {
        *info = -14;
    } else if (ldq < (n > 1 ? n : 1)) {
        *info = -16;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -18;
    }

    bool lquery = (ldwork == -1);
    i32 wrkopt = 1;

    if (*info == 0) {
        const char* sjob = update ? "V" : "N";
        if (lquery && nofact) {
            i32 sdim;
            i32 bwork_dummy[1] = {0};
            i32 qwork = -1;
            SLC_DGEES(sjob, "N", select_dummy, &n, t, &ldt, &sdim,
                      dwork, dwork, u, &ldu, dwork, &qwork, bwork_dummy, info);
            wrkopt = (int)dwork[0] + lwa + 2 * n;
            if (wrkopt < ldw) wrkopt = ldw;
            if (wrkopt < 1) wrkopt = 1;
        }
        if (ldwork < (1 > ldw ? 1 : ldw) && !lquery) {
            *info = -24;
        }
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("SB02QD", &neginfo);
        return;
    }

    if (lquery) {
        dwork[0] = (f64)wrkopt;
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

    i32 ixbs = 0;
    i32 itmp = ixbs + nn;
    i32 iabs = itmp + nn;
    i32 ires = iabs + nn;

    wrkopt = 0;

    if (needac || nofact) {
        SLC_DLACPY("F", &n, &n, a, &lda, dwork, &n);

        if (notrna) {
            f64 neg1 = -ONE;
            SLC_DSYMM("L", uplo, &n, &n, &neg1, g, &ldg, x, &ldx, &ONE, dwork, &n);
        } else {
            f64 neg1 = -ONE;
            SLC_DSYMM("R", uplo, &n, &n, &neg1, g, &ldg, x, &ldx, &ONE, dwork, &n);
        }

        wrkopt = nn;
        if (nofact) {
            SLC_DLACPY("F", &n, &n, dwork, &n, t, &ldt);
        }
    } else {
        wrkopt = n;
    }

    if (nofact) {
        const char* sjob = update ? "V" : "N";
        i32 sdim;
        i32 bwork_dummy[1] = {0};
        i32 lwork = ldwork - lwa - 2 * n;

        SLC_DGEES(sjob, "N", select_dummy, &n, t, &ldt, &sdim,
                  &dwork[lwa], &dwork[lwa + n], u, &ldu,
                  &dwork[lwa + 2 * n], &lwork, bwork_dummy, info);

        if (*info > 0) {
            if (lwa > 0) {
                i32 two_n = 2 * n;
                i32 int1 = 1;
                SLC_DCOPY(&two_n, &dwork[lwa], &int1, dwork, &int1);
            }
            return;
        }

        i32 opt = (int)dwork[lwa + 2 * n] + lwa + 2 * n;
        if (opt > wrkopt) wrkopt = opt;
    }

    if (needac) {
        SLC_DLACPY("F", &n, &n, dwork, &n, &dwork[iabs], &n);
    }

    char tranat = notrna ? 'T' : 'N';

    if (!jobe) {
        f64 thnorm;
        sb03qy("B", trana, lyapun, n, t, ldt, u, ldu, x, ldx,
               sep, &thnorm, iwork, dwork, ldwork, info);

        i32 opt = lwa + 2 * nn;
        if (opt > wrkopt) wrkopt = opt;

        if (*sep == ZERO) {
            *rcond = ZERO;
            if (jobb) *ferr = ONE;
            dwork[0] = (f64)wrkopt;
            return;
        }

        i32 kase = 0;
        i32 isave[3] = {0, 0, 0};
        f64 est, scale;
        i32 info2;

        do {
            SLC_DLACN2(&nn, &dwork[itmp], dwork, iwork, &est, &kase, isave);

            if (kase != 0) {
                f64 upper_norm = SLC_DLANSY("1", "U", &n, dwork, &n, &dwork[itmp]);
                f64 lower_norm = SLC_DLANSY("1", "L", &n, dwork, &n, &dwork[itmp]);
                char loup = (upper_norm >= lower_norm) ? 'U' : 'L';

                mb01ru(&loup, "N", n, n, ZERO, ONE, dwork, n, x, ldx,
                       dwork, n, &dwork[itmp], nn, &info2);
                i32 np1 = n + 1;
                SLC_DSCAL(&n, &HALF, dwork, &np1);

                if (update) {
                    mb01ru(&loup, "T", n, n, ZERO, ONE, dwork, n, u, ldu,
                           dwork, n, &dwork[itmp], nn, &info2);
                    SLC_DSCAL(&n, &HALF, dwork, &np1);
                }

                ma02ed(loup, n, dwork, n);

                if (kase == 1) {
                    sb03my(trana, n, t, ldt, dwork, n, &scale, &info2);
                } else {
                    sb03my(&tranat, n, t, ldt, dwork, n, &scale, &info2);
                }

                if (update) {
                    mb01ru(&loup, "N", n, n, ZERO, ONE, dwork, n, u, ldu,
                           dwork, n, &dwork[itmp], nn, &info2);
                    SLC_DSCAL(&n, &HALF, dwork, &np1);

                    ma02ed(loup, n, dwork, n);
                }
            }
        } while (kase != 0);

        f64 pinorm;
        if (est < scale) {
            pinorm = est / scale;
        } else {
            f64 bignum = ONE / SLC_DLAMCH("S");
            if (est < scale * bignum) {
                pinorm = est / scale;
            } else {
                pinorm = bignum;
            }
        }

        f64 anorm;
        if (update) {
            anorm = SLC_DLANGE("1", &n, &n, a, &lda, dwork);
        } else {
            anorm = SLC_DLANHS("1", &n, t, &ldt, dwork);
        }

        f64 qnorm = SLC_DLANSY("1", uplo, &n, q, &ldq, dwork);
        f64 gnorm = SLC_DLANSY("1", uplo, &n, g, &ldg, dwork);

        f64 tmax = *sep;
        if (xnorm > tmax) tmax = xnorm;
        if (anorm > tmax) tmax = anorm;
        if (gnorm > tmax) tmax = gnorm;

        f64 temp, denom;
        if (tmax <= ONE) {
            temp = (*sep) * xnorm;
            denom = qnorm + (*sep * anorm) * thnorm + (*sep * gnorm) * pinorm;
        } else {
            temp = (*sep / tmax) * (xnorm / tmax);
            denom = ((ONE / tmax) * (qnorm / tmax)) +
                    ((*sep / tmax) * (anorm / tmax)) * thnorm +
                    ((*sep / tmax) * (gnorm / tmax)) * pinorm;
        }

        if (temp >= denom) {
            *rcond = ONE;
        } else {
            *rcond = temp / denom;
        }
    }

    if (!jobc) {
        f64 sig;
        i32 info2;

        if (update) {
            char uplo_str[2] = {uplo_c, '\0'};
            SLC_DLACPY(uplo_str, &n, &n, q, &ldq, &dwork[ires], &n);
            char tranat_str[2] = {tranat, '\0'};
            SLC_DSYR2K(uplo_str, tranat_str, &n, &n, &ONE, a, &lda, x, &ldx, &ONE, &dwork[ires], &n);
            sig = -ONE;
        } else {
            mb01ud("R", trana, n, n, ONE, t, ldt, x, ldx, &dwork[ires], n, &info2);

            i32 jj = ires;
            if (lower) {
                for (i32 j = 0; j < n; j++) {
                    i32 count = n - j;
                    i32 int1 = 1;
                    SLC_DAXPY(&count, &ONE, &dwork[jj], &n, &dwork[jj], &int1);
                    SLC_DAXPY(&count, &ONE, &q[j + j * ldq], &int1, &dwork[jj], &int1);
                    jj += n + 1;
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    i32 count = j + 1;
                    i32 int1 = 1;
                    SLC_DAXPY(&count, &ONE, &dwork[ires + j], &n, &dwork[jj], &int1);
                    SLC_DAXPY(&count, &ONE, &q[j * ldq], &int1, &dwork[jj], &int1);
                    jj += n;
                }
            }
            sig = ONE;
        }

        char uplo_str[2] = {uplo_c, '\0'};
        SLC_DLACPY(uplo_str, &n, &n, g, &ldg, &dwork[iabs], &n);
        mb01ru(&uplo_c, &tranat, n, n, ONE, sig, &dwork[ires], n, x, ldx,
               &dwork[iabs], n, &dwork[itmp], nn, &info2);

        f64 eps = SLC_DLAMCH("E");
        f64 epsn = eps * (f64)(n + 4);
        f64 temp = eps * FOUR;

        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < n; i++) {
                dwork[ixbs + j * n + i] = fabs(x[i + j * ldx]);
            }
        }

        if (lower) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = j; i < n; i++) {
                    dwork[ires + j * n + i] = temp * fabs(q[i + j * ldq]) +
                        fabs(dwork[ires + j * n + i]);
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i <= j; i++) {
                    dwork[ires + j * n + i] = temp * fabs(q[i + j * ldq]) +
                        fabs(dwork[ires + j * n + i]);
                }
            }
        }

        if (update) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < n; i++) {
                    dwork[iabs + j * n + i] = fabs(dwork[iabs + j * n + i]);
                }
            }

            char tranat_str[2] = {tranat, '\0'};
            SLC_DSYR2K(uplo_str, tranat_str, &n, &n, &epsn, &dwork[iabs], &n,
                       &dwork[ixbs], &n, &ONE, &dwork[ires], &n);
        } else {
            for (i32 j = 0; j < n; j++) {
                i32 count = (j + 1 < n) ? (j + 2) : n;
                for (i32 i = 0; i < count; i++) {
                    dwork[iabs + j * n + i] = fabs(t[i + j * ldt]);
                }
            }

            mb01ud("L", &tranat, n, n, epsn, &dwork[iabs], n,
                   &dwork[ixbs], n, &dwork[itmp], n, &info2);

            i32 jj = ires;
            i32 jx = itmp;
            if (lower) {
                for (i32 j = 0; j < n; j++) {
                    i32 count = n - j;
                    i32 int1 = 1;
                    SLC_DAXPY(&count, &ONE, &dwork[jx], &n, &dwork[jx], &int1);
                    SLC_DAXPY(&count, &ONE, &dwork[jx], &int1, &dwork[jj], &int1);
                    jj += n + 1;
                    jx += n + 1;
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    i32 count = j + 1;
                    i32 int1 = 1;
                    SLC_DAXPY(&count, &ONE, &dwork[itmp + j], &n, &dwork[jx], &int1);
                    SLC_DAXPY(&count, &ONE, &dwork[jx], &int1, &dwork[jj], &int1);
                    jj += n;
                    jx += n;
                }
            }
        }

        if (lower) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = j; i < n; i++) {
                    dwork[iabs + j * n + i] = fabs(g[i + j * ldg]);
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i <= j; i++) {
                    dwork[iabs + j * n + i] = fabs(g[i + j * ldg]);
                }
            }
        }

        f64 eps2 = eps * (f64)(2 * (n + 1));
        mb01ru(&uplo_c, trana, n, n, ONE, eps2, &dwork[ires], n,
               &dwork[ixbs], n, &dwork[iabs], n, &dwork[itmp], nn, &info2);

        i32 opt = 4 * nn;
        if (opt > wrkopt) wrkopt = opt;

        f64 xanorm = SLC_DLANSY("M", uplo, &n, x, &ldx, dwork);

        sb03qx(trana, uplo, lyapun, n, xanorm, t, ldt, u, ldu,
               &dwork[ires], n, ferr, iwork, dwork, ires, info);
    }

    dwork[0] = (f64)wrkopt;
}
