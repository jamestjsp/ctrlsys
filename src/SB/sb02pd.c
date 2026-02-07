/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB02PD - Continuous-time algebraic Riccati equation solver using matrix
 *          sign function method with error bounds and condition estimates
 *
 * Solves the real continuous-time matrix algebraic Riccati equation:
 *     op(A)'*X + X*op(A) + Q - X*G*X = 0
 * where op(A) = A or A' and G, Q are symmetric.
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

void sb02pd(
    const char* job,
    const char* trana,
    const char* uplo,
    const i32 n,
    const f64* a,
    const i32 lda,
    const f64* g,
    const i32 ldg,
    const f64* q,
    const i32 ldq,
    f64* x,
    const i32 ldx,
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
    const i32 MAXIT = 50;
    const f64 ZERO = 0.0;
    const f64 HALF = 0.5;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 TEN = 10.0;

    char job_c = (char)toupper((unsigned char)job[0]);
    char trana_c = (char)toupper((unsigned char)trana[0]);
    char uplo_c = (char)toupper((unsigned char)uplo[0]);

    bool all = (job_c == 'A');
    bool notrna = (trana_c == 'N');
    bool lower = (uplo_c == 'L');
    bool lquery = (ldwork == -1);

    *info = 0;

    if (!all && job_c != 'X') {
        *info = -1;
    } else if (trana_c != 'T' && trana_c != 'C' && !notrna) {
        *info = -2;
    } else if (!lower && uplo_c != 'U') {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -6;
    } else if (ldg < (1 > n ? 1 : n)) {
        *info = -8;
    } else if (ldq < (1 > n ? 1 : n)) {
        *info = -10;
    } else if (ldx < (1 > n ? 1 : n)) {
        *info = -12;
    }

    i32 n2 = 2 * n;
    i32 minwrk;
    i32 lwamax = 1;

    if (*info == 0) {
        if (all) {
            i32 opt1 = n2 * n2 + 8 * n + 1;
            i32 opt2 = 6 * n * n;
            minwrk = (opt1 > opt2) ? opt1 : opt2;
        } else {
            minwrk = n2 * n2 + 8 * n + 1;
        }

        i32 itau = n2 * n2;
        i32 iwrk = itau + n2;

        if (lquery) {
            i32 info2;
            i32 lw_neg1 = -1;

            SLC_DSYTRF(uplo, &n2, dwork, &n2, iwork, dwork, &lw_neg1, &info2);
            lwamax = (i32)dwork[0];

            SLC_DGEQP3(&n2, &n2, dwork, &n2, iwork, dwork, dwork, &lw_neg1, &info2);
            i32 qp3_opt = (i32)dwork[0];
            if (qp3_opt > lwamax) lwamax = qp3_opt;

            SLC_DORMQR("L", "N", &n2, &n, &n, dwork, &n2, dwork, dwork,
                      &n2, dwork, &lw_neg1, &info2);
            i32 qr_opt = (i32)dwork[0];
            i32 iwrk_opt = iwrk + (qr_opt > lwamax ? qr_opt : lwamax);
            if (iwrk_opt > minwrk) lwamax = iwrk_opt;
            else lwamax = minwrk;

            if (all) {
                i32 bwork_dummy[1] = {0};
                i32 sdim;
                SLC_DGEES("V", "N", select_dummy, &n, dwork, &n, &sdim, wr, wi,
                         dwork, &n, dwork, &lw_neg1, bwork_dummy, &info2);
                i32 gees_opt = n2 * n + (i32)dwork[0];
                if (gees_opt > lwamax) lwamax = gees_opt;

                f64 sep_tmp;
                sb02qd("B", "F", trana, uplo, "O", n, a, lda, dwork, n, dwork, n,
                       g, ldg, q, ldq, x, ldx, &sep_tmp, rcond, ferr, iwork, dwork, -1, &info2);
                i32 qd_opt = n2 * n + (i32)dwork[0];
                if (qd_opt > lwamax) lwamax = qd_opt;
            }
        }

        if (ldwork < minwrk && !lquery) {
            *info = -19;
        }
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("SB02PD", &neginfo);
        return;
    }

    if (lquery) {
        dwork[0] = (f64)lwamax;
        return;
    }

    if (n == 0) {
        if (all) {
            *rcond = ONE;
            *ferr = ZERO;
        }
        dwork[0] = ONE;
        return;
    }

    f64 eps = SLC_DLAMCH("P");
    f64 tol = TEN * (f64)n * eps;

    f64 qnorm2 = sqrt(SLC_DLANSY("1", uplo, &n, q, &ldq, dwork));
    f64 gnorm2 = sqrt(SLC_DLANSY("1", uplo, &n, g, &ldg, dwork));

    i32 ini, isv;
    char loup;

    if (lower) {
        ini = 0;
        isv = n2;
        loup = 'U';

        for (i32 j = 0; j < n; j++) {
            i32 ij = j * n2 + j;
            for (i32 i = j; i < n; i++) {
                dwork[ij] = -q[i + j * ldq];
                ij++;
            }

            if (notrna) {
                for (i32 i = 0; i < n; i++) {
                    dwork[ij] = -a[i + j * lda];
                    ij++;
                }
            } else {
                for (i32 i = 0; i < n; i++) {
                    dwork[ij] = -a[j + i * lda];
                    ij++;
                }
            }
        }

        for (i32 j = 0; j < n; j++) {
            i32 ij = (n + j) * n2 + n + j;
            for (i32 i = j; i < n; i++) {
                dwork[ij] = g[i + j * ldg];
                ij++;
            }
        }
    } else {
        ini = n2;
        isv = 0;
        loup = 'L';

        for (i32 j = 0; j < n; j++) {
            i32 ij = (j + 1) * n2;
            for (i32 i = 0; i <= j; i++) {
                dwork[ij] = -q[i + j * ldq];
                ij++;
            }
        }

        for (i32 j = 0; j < n; j++) {
            i32 ij = (n + j + 1) * n2;

            if (notrna) {
                for (i32 i = 0; i < n; i++) {
                    dwork[ij] = -a[j + i * lda];
                    ij++;
                }
            } else {
                for (i32 i = 0; i < n; i++) {
                    dwork[ij] = -a[i + j * lda];
                    ij++;
                }
            }

            for (i32 i = 0; i <= j; i++) {
                dwork[ij] = g[i + j * ldg];
                ij++;
            }
        }
    }

    i32 iscl = 0;
    if (qnorm2 > gnorm2 && gnorm2 > ZERO) {
        i32 info2;
        SLC_DLASCL(uplo, &(i32){0}, &(i32){0}, &qnorm2, &gnorm2, &n, &n, &dwork[ini], &n2, &info2);
        SLC_DLASCL(uplo, &(i32){0}, &(i32){0}, &gnorm2, &qnorm2, &n, &n, &dwork[n2 * n + n + ini], &n2, &info2);
        iscl = 1;
    }

    i32 itau = n2 * n2;
    i32 iwrk = itau + n2;

    for (i32 iter = 0; iter < MAXIT; iter++) {
        if (lower) {
            for (i32 i = 0; i < n2; i++) {
                i32 int1 = 1;
                i32 count = i + 1;
                SLC_DCOPY(&count, &dwork[i], &n2, &dwork[(i + 1) * n2], &int1);
            }
        } else {
            for (i32 i = 0; i < n2; i++) {
                i32 int1 = 1;
                i32 count = i + 1;
                SLC_DCOPY(&count, &dwork[(i + 1) * n2], &int1, &dwork[i], &n2);
            }
        }

        f64 hnorm = SLC_DLANSY("F", uplo, &n2, &dwork[ini], &n2, dwork);

        i32 info2;
        i32 lwork = ldwork - iwrk;
        SLC_DSYTRF(uplo, &n2, &dwork[ini], &n2, iwork, &dwork[iwrk], &lwork, &info2);

        if (info2 > 0) {
            *info = 1;
            return;
        }

        i32 opt = iwrk + (i32)dwork[iwrk];
        if (opt > lwamax) lwamax = opt;

        SLC_DSYTRI(uplo, &n2, &dwork[ini], &n2, iwork, &dwork[iwrk], &info2);

        if (lower) {
            for (i32 j = 0; j < n; j++) {
                i32 ij2 = (n + j) * n2 + n + j;
                for (i32 ij1 = j * n2 + j; ij1 < j * n2 + n; ij1++) {
                    f64 temp = dwork[ij1];
                    dwork[ij1] = -dwork[ij2];
                    dwork[ij2] = -temp;
                    ij2++;
                }

                i32 count = j;
                i32 int1 = 1;
                if (count > 0) {
                    SLC_DSWAP(&count, &dwork[n + j], &n2, &dwork[j * n2 + n], &int1);
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                i32 ij2 = (n + j + 1) * n2 + n;
                for (i32 ij1 = (j + 1) * n2; ij1 <= (j + 1) * n2 + j; ij1++) {
                    f64 temp = dwork[ij1];
                    dwork[ij1] = -dwork[ij2];
                    dwork[ij2] = -temp;
                    ij2++;
                }

                i32 count = j;
                i32 int1 = 1;
                if (count > 0) {
                    SLC_DSWAP(&count, &dwork[(n + 1) * n2 + j], &n2, &dwork[(n + j + 1) * n2], &int1);
                }
            }
        }

        f64 hinnrm = SLC_DLANSY("F", uplo, &n2, &dwork[ini], &n2, dwork);
        f64 scale = sqrt(hinnrm / hnorm);

        if (lower) {
            for (i32 j = 0; j < n2; j++) {
                i32 ji = j * n2 + j;
                for (i32 ij = ji; ij < (j + 1) * n2; ij++) {
                    dwork[ij] = (dwork[ij] / scale + dwork[ji] * scale) / TWO;
                    dwork[ji] = dwork[ji] - dwork[ij];
                    ji += n2;
                }
            }
        } else {
            for (i32 j = 0; j < n2; j++) {
                i32 ji = j;
                for (i32 ij = (j + 1) * n2; ij <= (j + 1) * n2 + j; ij++) {
                    dwork[ij] = (dwork[ij] / scale + dwork[ji] * scale) / TWO;
                    dwork[ji] = dwork[ji] - dwork[ij];
                    ji += n2;
                }
            }
        }

        char loup_str[2] = {loup, '\0'};
        f64 conv = SLC_DLANSY("F", loup_str, &n2, &dwork[isv], &n2, dwork);
        if (conv <= tol * hnorm) {
            goto L240;
        }
    }

    *info = 2;

L240:
    if (!lower) {
        SLC_DLACPY("U", &n2, &n2, &dwork[ini], &n2, dwork, &n2);
    }

    if (lower) {
        for (i32 i = 0; i < n2; i++) {
            i32 count = n2 - i;
            i32 int1 = 1;
            f64 factor = -HALF;
            SLC_DSCAL(&count, &factor, &dwork[i * n2 + i], &int1);
        }
    } else {
        for (i32 i = 0; i < n2; i++) {
            i32 count = i + 1;
            i32 int1 = 1;
            f64 factor = -HALF;
            SLC_DSCAL(&count, &factor, &dwork[i * n2], &int1);
        }
    }

    ma02ed(uplo_c, n2, dwork, n2);

    for (i32 j = 0; j < n2; j++) {
        for (i32 i = j * n2; i < j * n2 + n; i++) {
            f64 temp = dwork[i];
            dwork[i] = -dwork[i + n];
            dwork[i + n] = temp;
        }
    }

    for (i32 i = 0; i < n2; i++) {
        iwork[i] = 0;
        dwork[i * n2 + i] += HALF;
    }

    i32 info2;
    i32 lwork = ldwork - iwrk;
    SLC_DGEQP3(&n2, &n2, dwork, &n2, iwork, &dwork[itau], &dwork[iwrk], &lwork, &info2);

    i32 opt = iwrk + (i32)dwork[iwrk];
    if (opt > lwamax) lwamax = opt;

    i32 ib = n * n;
    i32 iaf = n2 * n;

    SLC_DLASET("F", &n2, &n, &ZERO, &ONE, &dwork[iaf], &n2);

    lwork = ldwork - iwrk;
    SLC_DORMQR("L", "N", &n2, &n, &n, dwork, &n2, &dwork[itau],
              &dwork[iaf], &n2, &dwork[iwrk], &lwork, &info2);

    opt = iwrk + (i32)dwork[iwrk];
    if (opt > lwamax) lwamax = opt;

    SLC_DLACPY("F", &n, &n, &dwork[iaf], &n2, dwork, &n);
    ma02ad("F", n, n, &dwork[iaf + n], n2, &dwork[ib], n);

    i32 ir = iaf + ib;
    i32 ic = ir + n;
    i32 ifr = ic + n;
    i32 ibr = ifr + n;
    iwrk = ibr + n;

    char equed;
    SLC_DGESVX("E", "T", &n, &n, dwork, &n, &dwork[iaf], &n,
              iwork, &equed, &dwork[ir], &dwork[ic],
              &dwork[ib], &n, x, &ldx, rcond, &dwork[ifr],
              &dwork[ibr], &dwork[iwrk], &iwork[n], &info2);

    if (info2 > 0) {
        *info = 3;
        return;
    }

    for (i32 i = 0; i < n - 1; i++) {
        for (i32 j = i + 1; j < n; j++) {
            f64 temp = (x[i + j * ldx] + x[j + i * ldx]) / TWO;
            x[i + j * ldx] = temp;
            x[j + i * ldx] = temp;
        }
    }

    if (iscl == 1) {
        SLC_DLASCL("G", &(i32){0}, &(i32){0}, &gnorm2, &qnorm2, &n, &n, x, &ldx, &info2);
    }

    if (all) {
        i32 it = 0;
        i32 iu = it + n * n;
        iwrk = iu + n * n;

        SLC_DLACPY("Full", &n, &n, a, &lda, &dwork[it], &n);

        if (notrna) {
            f64 neg1 = -ONE;
            SLC_DSYMM("L", uplo, &n, &n, &neg1, g, &ldg, x, &ldx, &ONE, &dwork[it], &n);
        } else {
            f64 neg1 = -ONE;
            SLC_DSYMM("R", uplo, &n, &n, &neg1, g, &ldg, x, &ldx, &ONE, &dwork[it], &n);
        }

        lwork = ldwork - iwrk;
        i32 sdim;
        i32 bwork_dummy[1] = {0};
        SLC_DGEES("V", "N", select_dummy, &n, &dwork[it], &n, &sdim, wr, wi,
                 &dwork[iu], &n, &dwork[iwrk], &lwork, bwork_dummy, &info2);

        if (info2 > 0) {
            *info = 4;
            return;
        }

        opt = iwrk + (i32)dwork[iwrk];
        if (opt > lwamax) lwamax = opt;

        f64 sep_tmp;
        lwork = ldwork - iwrk;
        sb02qd("B", "F", trana, uplo, "O", n, a, lda, &dwork[it], n, &dwork[iu], n,
               g, ldg, q, ldq, x, ldx, &sep_tmp, rcond, ferr, iwork, &dwork[iwrk], lwork, &info2);

        opt = iwrk + (i32)dwork[iwrk];
        if (opt > lwamax) lwamax = opt;
    }

    dwork[0] = (f64)lwamax;
}
