/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

void tg01od(
    const char* jobe,
    const i32 n,
    f64* dcba, const i32 lddcba,
    f64* e, const i32 lde,
    i32* nz,
    f64* g,
    const f64 tol,
    f64* dwork, const i32 ldwork,
    i32* info
)
{
    const f64 one = 1.0;
    const f64 three = 3.0;
    const f64 four = 4.0;
    const f64 zero = 0.0;

    bool descr, lquery;
    i32 i, imax, itau, iwrk, j, jf, maxwrk, minwrk, n1, nc;
    f64 absd, maxa, nrmb, nrmc, tau, toldef;
    char jobt[2] = "I";
    i32 int1 = 1;

    descr = (jobe[0] == 'G' || jobe[0] == 'g');
    *info = 0;
    n1 = n + 1;

    i32 max1n = (1 > n) ? 1 : n;

    if (!descr && !(jobe[0] == 'I' || jobe[0] == 'i')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lddcba < n1) {
        *info = -4;
    } else if (lde < 1 || (descr && lde < max1n)) {
        *info = -6;
    } else {
        if (descr) {
            minwrk = 2 * n + 1;
        } else if (n == 0) {
            minwrk = 1;
        } else {
            minwrk = n1;
        }
        maxwrk = minwrk;
        lquery = (ldwork == -1);

        if (lquery) {
            if (descr) {
                i32 qr_info;
                SLC_DGEQRF(&n, &n, e, &lde, dcba, dwork, &int1, &qr_info);
                i32 qr_opt = (i32)dwork[0];
                maxwrk = (maxwrk > qr_opt) ? maxwrk : qr_opt;

                SLC_DORMQR("L", "T", &n, &n1, &n, e, &lde, dcba,
                           dcba, &lddcba, dwork, &int1, &qr_info);
                i32 qr_opt2 = (i32)dwork[0];
                maxwrk = (maxwrk > qr_opt2) ? maxwrk : qr_opt2;
                dwork[0] = (f64)maxwrk;
            } else {
                dwork[0] = (f64)maxwrk;
            }
            return;
        } else if (ldwork < minwrk) {
            dwork[0] = (f64)minwrk;
            *info = -11;
        }
    }

    if (*info != 0) {
        return;
    }

    *nz = n;
    if (n == 0) {
        *g = dcba[0 + 0 * lddcba];
        dwork[0] = one;
        return;
    }

    toldef = tol;
    if (toldef <= zero) {
        f64 eps = SLC_DLAMCH("P");
        toldef = pow(eps, three / four);
    }

    *g = one;
    maxa = SLC_DLANGE("M", &n, &n, &dcba[1 + 1 * lddcba], &lddcba, dwork);
    nrmb = SLC_DNRM2(&n, &dcba[1 + 0 * lddcba], &int1);
    nrmc = SLC_DNRM2(&n, &dcba[0 + 1 * lddcba], &lddcba);

    if (fabs(dcba[0 + 0 * lddcba]) * (one + maxa) <= toldef * nrmb * nrmc) {
        if (descr) {
            itau = 0;
            iwrk = itau + n;
            i32 ldwork_qr = ldwork - n;
            i32 qr_info;

            SLC_DGEQRF(&n, &n, e, &lde, &dwork[itau], &dwork[iwrk], &ldwork_qr, &qr_info);
            i32 qr_opt = (i32)dwork[iwrk];
            maxwrk = (maxwrk > qr_opt) ? maxwrk : qr_opt;

            SLC_DORMQR("L", "T", &n, &n1, &n, e, &lde,
                       &dwork[itau], &dcba[1 + 0 * lddcba], &lddcba,
                       &dwork[iwrk], &ldwork_qr, &qr_info);
            qr_opt = (i32)dwork[iwrk];
            maxwrk = (maxwrk > qr_opt) ? maxwrk : qr_opt;

            if (n > 1) {
                i32 nm1 = n - 1;
                SLC_DLASET("L", &nm1, &nm1, &zero, &zero, &e[1 + 0 * lde], &lde);
            }
            jobt[0] = 'U';
        } else {
            jobt[0] = jobe[0];
        }

        for (i = 0; i < n; i++) {
            nc = *nz + 1;

            if (!descr) {
                if (dcba[(i + 1) + i * lddcba] == zero) {
                    imax = SLC_IDAMAX(nz, &dcba[(i + 1) + i * lddcba], &int1) - 1 + (i + 1);
                    SLC_DSWAP(&nc, &dcba[(i + 1) + i * lddcba], &lddcba,
                              &dcba[imax + i * lddcba], &lddcba);
                    SLC_DSWAP(&nc, &dcba[i + (i + 1) * lddcba], &int1,
                              &dcba[i + imax * lddcba], &int1);
                }

                i32 start_idx = (i + 2 < n1) ? i + 2 : n1 - 1;
                SLC_DLARFG(nz, &dcba[(i + 1) + i * lddcba],
                           &dcba[start_idx + i * lddcba], &int1, &tau);

                *g = *g * dcba[(i + 1) + i * lddcba];
                dcba[(i + 1) + i * lddcba] = one;

                SLC_DLARF("L", nz, nz, &dcba[(i + 1) + i * lddcba], &int1, &tau,
                          &dcba[(i + 1) + (i + 1) * lddcba], &lddcba, dwork);

                SLC_DLARF("R", &nc, nz, &dcba[(i + 1) + i * lddcba], &int1, &tau,
                          &dcba[i + (i + 1) * lddcba], &lddcba, dwork);
            } else {
                tg01oa(jobt, *nz, &dcba[i + i * lddcba], lddcba,
                       &e[i + i * lde], lde, info);

                *g = *g * dcba[(i + 1) + i * lddcba] / e[i + i * lde];
            }

            SLC_DCOPY(nz, &dcba[i + (i + 1) * lddcba], &lddcba,
                      &dcba[(i + 1) + (i + 1) * lddcba], &lddcba);

            (*nz)--;
            absd = fabs(dcba[(i + 1) + (i + 1) * lddcba]);
            nrmb = SLC_DNRM2(nz, &dcba[(i + 2) + (i + 1) * lddcba], &int1);
            nrmc = SLC_DNRM2(nz, &dcba[(i + 1) + (i + 2) * lddcba], &lddcba);

            if (absd == zero && (nrmb == zero || nrmc == zero)) {
                *nz = 0;
                goto move_results;
            }

            maxa = SLC_DLANGE("M", nz, nz, &dcba[(i + 2) + (i + 2) * lddcba], &lddcba, dwork);
            if (absd * (one + maxa) > toldef * nrmb * nrmc) {
                goto move_results;
            }
        }

        i = n;

move_results:
        jf = 0;

        for (j = i + 1; j <= n; j++) {
            i32 nz_plus_1 = *nz + 1;
            SLC_DCOPY(&nz_plus_1, &dcba[(i + 1) + j * lddcba], &int1,
                      &dcba[0 + jf * lddcba], &int1);
            jf++;
        }

        if (descr) {
            jf = 0;

            for (j = i + 1; j < n; j++) {
                SLC_DCOPY(nz, &e[(i + 1) + j * lde], &int1,
                          &e[0 + jf * lde], &int1);
                jf++;
            }
        }
    }

    *g = *g * dcba[0 + 0 * lddcba];
    dwork[0] = (f64)maxwrk;
}
