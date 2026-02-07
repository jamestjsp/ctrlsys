/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

static bool lsame(char ca, char cb) {
    if (ca >= 'a' && ca <= 'z') ca -= 32;
    if (cb >= 'a' && cb <= 'z') cb -= 32;
    return ca == cb;
}

void md03by(
    const char* cond,
    const i32 n,
    f64* r,
    const i32 ldr,
    const i32* ipvt,
    const f64* diag,
    const f64* qtb,
    const f64 delta,
    f64* par,
    i32* rank,
    f64* x,
    f64* rx,
    const f64 tol,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const i32 itmax = 10;
    const f64 p1 = 0.1;
    const f64 p001 = 0.001;
    const f64 zero = 0.0;
    const f64 svlmax = 0.0;

    i32 iter, j, l, n2;
    f64 dmino, dwarf, dxnorm, fp, gnorm, parc, parl, paru, temp, toldef;
    bool econd, ncond, sing, ucond;
    char condl;
    f64 dum[3];

    i32 int0 = 0;
    i32 int1 = 1;
    econd = lsame(*cond, 'E');
    ncond = lsame(*cond, 'N');
    ucond = lsame(*cond, 'U');
    *info = 0;

    if (!econd && !ncond && !ucond) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldr < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (delta <= zero) {
        *info = -8;
    } else if (*par < zero) {
        *info = -9;
    } else if (ucond && (*rank < 0 || *rank > n)) {
        *info = -10;
    } else if (ldwork < 2*n || (econd && ldwork < 4*n)) {
        *info = -15;
    } else if (n > 0) {
        dmino = diag[0];
        sing = false;

        for (j = 0; j < n; j++) {
            if (diag[j] < dmino) {
                dmino = diag[j];
            }
            sing = sing || (diag[j] == zero);
        }

        if (sing) {
            *info = -6;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        *par = zero;
        *rank = 0;
        return;
    }

    dwarf = SLC_DLAMCH("Underflow");
    n2 = n;

    if (econd) {
        n2 = 2*n;
        temp = tol;
        if (temp <= zero) {
            temp = (f64)n * SLC_DLAMCH("Epsilon");
        }

        i32* jpvt_tmp = (i32*)malloc(n * sizeof(i32));
        for (i32 i = 0; i < n; i++) {
            jpvt_tmp[i] = ipvt[i];
        }

        f64 dum_scalar[3];
        mb03od("No QR", n, n, r, ldr, jpvt_tmp, temp, svlmax, dwork,
               rank, dum_scalar, dwork, ldwork, info);

        free(jpvt_tmp);

    } else if (ncond) {
        j = 0;

        while (j < n && r[j + j*ldr] != zero) {
            j++;
        }

        *rank = j;
    }

    SLC_DCOPY(rank, qtb, &int1, rx, &int1);
    dum[0] = zero;
    if (*rank < n) {
        i32 nrank_diff = n - *rank;
        SLC_DCOPY(&nrank_diff, dum, &int0, &rx[*rank], &int1);
    }
    SLC_DTRSV("Upper", "No transpose", "Non unit", rank, r, &ldr, rx, &int1);

    for (j = 0; j < n; j++) {
        l = ipvt[j] - 1;
        if (l < 0 || l >= n) {
            break;
        }
        x[l] = rx[j];
    }

    iter = 0;

    for (j = 0; j < n; j++) {
        dwork[j] = diag[j] * x[j];
    }

    dxnorm = SLC_DNRM2(&n, dwork, &int1);
    fp = dxnorm - delta;

    if (fp > p1*delta) {

        if (ucond) {
            if (ldwork >= 4*n) {
                condl = 'E';
                toldef = (f64)n * SLC_DLAMCH("Epsilon");
            } else {
                condl = 'N';
                toldef = tol;
            }
        } else {
            condl = *cond;
            toldef = tol;
        }

        if (*rank == n) {

            for (j = 0; j < n; j++) {
                l = ipvt[j] - 1;
                if (l < 0 || l >= n) {
                    break;
                }
                rx[j] = diag[l] * (dwork[l] / dxnorm);
            }

            SLC_DTRSV("Upper", "Transpose", "Non unit", &n, r, &ldr, rx, &int1);
            temp = SLC_DNRM2(&n, rx, &int1);
            parl = ((fp/delta) / temp) / temp;

            if (condl != 'U' && dmino > zero) {
                condl = 'U';
            }
        } else {
            parl = zero;
        }

        for (j = 0; j < n; j++) {
            l = ipvt[j] - 1;
            if (l < 0 || l >= n) {
                break;
            }
            i32 jp1 = j + 1;
            rx[j] = SLC_DDOT(&jp1, &r[j*ldr], &int1, qtb, &int1) / diag[l];
        }

        gnorm = SLC_DNRM2(&n, rx, &int1);
        paru = gnorm / delta;
        if (paru == zero) {
            paru = dwarf / fmin(delta, p1) / p001;
        }

        *par = fmax(*par, parl);
        *par = fmin(*par, paru);
        if (*par == zero) {
            *par = gnorm / dxnorm;
        }

        while (true) {
            iter++;

            if (*par == zero) {
                *par = fmax(dwarf, p001*paru);
            }
            temp = sqrt(*par);

            for (j = 0; j < n; j++) {
                rx[j] = temp * diag[j];
            }

            mb02yd(&condl, n, r, ldr, ipvt, rx, qtb, rank, x,
                   toldef, dwork, ldwork, info);

            for (j = 0; j < n; j++) {
                dwork[n2 + j] = diag[j] * x[j];
            }

            dxnorm = SLC_DNRM2(&n, &dwork[n2], &int1);
            temp = fp;
            fp = dxnorm - delta;

            if (!(fabs(fp) > p1*delta &&
                  (parl != zero || fp > temp || temp >= zero) &&
                  iter < itmax)) {
                break;
            }

            for (j = 0; j < *rank; j++) {
                l = ipvt[j] - 1;
                if (l < 0 || l >= n) {
                    break;
                }
                rx[j] = diag[l] * (dwork[n2 + l] / dxnorm);
            }

            if (*rank < n) {
                i32 nrank_diff = n - *rank;
                SLC_DCOPY(&nrank_diff, dum, &int0, &rx[*rank], &int1);
            }

            i32 ldrp1 = ldr + 1;
            SLC_DSWAP(&n, r, &ldrp1, dwork, &int1);
            SLC_DTRSV("Lower", "No transpose", "Non Unit", rank, r, &ldr, rx, &int1);
            SLC_DSWAP(&n, r, &ldrp1, dwork, &int1);

            temp = SLC_DNRM2(rank, rx, &int1);
            parc = ((fp/delta) / temp) / temp;

            if (fp > zero) {
                parl = fmax(parl, *par);
            } else if (fp < zero) {
                paru = fmin(paru, *par);
            }

            *par = fmax(parl, *par + parc);
        }
    }

    if (econd && iter > 0) {

        for (j = 0; j < n; j++) {
            rx[j] = -dwork[n + j];
        }

        SLC_DTRMV("Upper", "NoTranspose", "NonUnit", &n, r, &ldr, rx, &int1);

    } else {

        for (j = 0; j < n; j++) {
            rx[j] = zero;
            l = ipvt[j] - 1;
            if (l < 0 || l >= n) {
                break;
            }
            i32 jp1 = j + 1;
            f64 neg_xl = -x[l];
            SLC_DAXPY(&jp1, &neg_xl, &r[j*ldr], &int1, rx, &int1);
        }
    }

    if (iter == 0) {
        *par = zero;

        for (j = 0; j < n - 1; j++) {
            dwork[j] = r[j + j*ldr];
            i32 nmj = n - j - 1;
            SLC_DCOPY(&nmj, &r[j + (j+1)*ldr], &ldr, &r[j+1 + j*ldr], &int1);
        }

        dwork[n-1] = r[(n-1) + (n-1)*ldr];
    }
}
