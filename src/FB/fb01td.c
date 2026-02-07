/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <ctype.h>
#include <string.h>


void fb01td(const char* jobx, const char* multrc,
            i32 n, i32 m, i32 p,
            f64* sinv, i32 ldsinv,
            const f64* ainv, i32 ldainv,
            const f64* ainvb, i32 ldainb,
            const f64* rinv, i32 ldrinv,
            const f64* c, i32 ldc,
            f64* qinv, i32 ldqinv,
            f64* x, const f64* rinvy, const f64* z, f64* e,
            f64 tol,
            i32* iwork,
            f64* dwork, i32 ldwork,
            i32* info)
{
    const f64 one = 1.0, two = 2.0;

    i32 np = n + p;
    i32 nm = n + m;
    i32 n1 = (n > 1) ? n : 1;
    i32 m1 = (m > 1) ? m : 1;
    i32 mp1 = m + 1;
    *info = 0;

    bool ljobx = (toupper((unsigned char)jobx[0]) == 'X');
    bool lmultr = (toupper((unsigned char)multrc[0]) == 'P');

    if (!ljobx && toupper((unsigned char)jobx[0]) != 'N') {
        *info = -1;
    } else if (!lmultr && toupper((unsigned char)multrc[0]) != 'N') {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (p < 0) {
        *info = -5;
    } else if (ldsinv < n1) {
        *info = -7;
    } else if (ldainv < n1) {
        *info = -9;
    } else if (ldainb < n1) {
        *info = -11;
    } else if (ldrinv < 1 || (!lmultr && ldrinv < p)) {
        *info = -13;
    } else if (ldc < ((p > 1) ? p : 1)) {
        *info = -15;
    } else if (ldqinv < m1) {
        *info = -17;
    } else {
        i32 min_ldwork;
        if (ljobx) {
            i32 val1 = n*(nm + m) + 3*m;
            i32 val2 = np*(n + 1) + n + ((n-1 > mp1) ? n-1 : mp1);
            min_ldwork = (val1 > val2) ? val1 : val2;
            min_ldwork = (min_ldwork > 3*n) ? min_ldwork : 3*n;
            min_ldwork = (min_ldwork > 2) ? min_ldwork : 2;
        } else {
            i32 val1 = n*(nm + m) + 3*m;
            i32 val2 = np*(n + 1) + n + ((n-1 > mp1) ? n-1 : mp1);
            min_ldwork = (val1 > val2) ? val1 : val2;
            min_ldwork = (min_ldwork > 1) ? min_ldwork : 1;
        }
        if (ldwork < min_ldwork) {
            *info = -25;
        }
    }

    if (*info != 0) {
        return;
    }

    if ((n > 0 ? n : 0) == 0 && (p > 0 ? p : 0) == 0) {
        if (ljobx) {
            dwork[0] = two;
            dwork[1] = one;
        } else {
            dwork[0] = one;
        }
        return;
    }

    i32 ldw = n1;
    i32 i32_off = n*m;

    SLC_DLACPY("Upper", &n, &m, (f64*)ainvb, &ldainb, dwork, &ldw);

    i32 min_mn = (m < n) ? m : n;
    SLC_DLACPY("Full", &min_mn, &n, (f64*)ainv, &ldainv, dwork + i32_off, &ldw);

    if (n > m) {
        i32 n_minus_m = n - m;
        SLC_DLACPY("Upper", &n_minus_m, &n, (f64*)ainv + m, &ldainv,
                   dwork + i32_off + m, &ldw);
    }

    i32 ii = 0;
    i32 i13 = n*nm;
    i32 wrkopt = (n*nm + n > 1) ? n*nm + n : 1;

    for (i32 i = 0; i < n; i++) {
        i32 i_plus_1 = i + 1;
        SLC_DCOPY(&i_plus_1, dwork + ii, &(i32){1}, dwork + i13, &(i32){1});
        SLC_DTRMV("Upper", "No transpose", "Non-unit", &i_plus_1, sinv, &ldsinv,
                  dwork + i13, &(i32){1});
        SLC_DCOPY(&i_plus_1, dwork + i13, &(i32){1}, dwork + ii, &(i32){1});
        ii += n;
    }

    SLC_DTRMM("Left", "Upper", "No transpose", "Non-unit", &n, &m,
              &one, sinv, &ldsinv, dwork + ii, &ldw);

    SLC_DCOPY(&m, (f64*)z, &(i32){1}, dwork + i13, &(i32){1});
    SLC_DTRMV("Upper", "No transpose", "Non-unit", &m, qinv, &ldqinv,
              dwork + i13, &(i32){1});

    SLC_DTRMV("Upper", "No transpose", "Non-unit", &n, sinv, &ldsinv, x, &(i32){1});

    i32 i12 = i13 + m;
    i32 itau = i12 + m*n;
    i32 jwork = itau + m;

    mb04kd('U', m, n, n, qinv, ldqinv, dwork, ldw,
           dwork + i32_off, ldw, dwork + i12, m1,
           dwork + itau, dwork + jwork);

    wrkopt = n*(nm + m) + 3*m;
    wrkopt = wrkopt > 1 ? wrkopt : 1;

    if (n == 0) {
        SLC_DCOPY(&p, (f64*)rinvy, &(i32){1}, e, &(i32){1});
        if (ljobx) {
            dwork[1] = one;
        }
        dwork[0] = (f64)wrkopt;
        return;
    }

    i32 ij = 0;
    for (i32 i = 0; i < m; i++) {
        i32 len = (i + 1 < n) ? i + 1 : n;
        f64 dot = SLC_DDOT(&len, dwork + ij, &(i32){1}, x, &(i32){1});
        f64 scale = -dwork[itau + i] * (dwork[i13 + i] + dot);
        SLC_DAXPY(&len, &scale, dwork + ij, &(i32){1}, x, &(i32){1});
        ij += n;
    }

    i32 min_mn2 = (m < n) ? m : n;
    SLC_DLACPY("Full", &min_mn2, &n, dwork + i32_off, &ldw, dwork, &ldw);
    if (n > m) {
        i32 n_minus_m = n - m;
        SLC_DLACPY("Upper", &n_minus_m, &n, dwork + i32_off + m, &ldw,
                   dwork + m, &ldw);
    }

    ldw = (np > 1) ? np : 1;

    for (i32 i = n - 1; i >= 0; i--) {
        i32 max_ij = ((i + 1 + m) < n) ? i + 1 + m : n;
        for (i32 ij_idx = max_ij - 1; ij_idx >= 0; ij_idx--) {
            dwork[np*i + p + ij_idx] = dwork[n1*i + ij_idx];
        }
    }

    SLC_DLACPY("Full", &p, &n, (f64*)c, &ldc, dwork, &ldw);
    if (!lmultr) {
        SLC_DTRMM("Left", "Upper", "No transpose", "Non-unit", &p, &n,
                  &one, (f64*)rinv, &ldrinv, dwork, &ldw);
    }

    i32 i23 = np*n;
    i32 i33 = i23 + p;
    SLC_DCOPY(&p, (f64*)rinvy, &(i32){1}, dwork + i23, &(i32){1});
    SLC_DCOPY(&n, x, &(i32){1}, dwork + i33, &(i32){1});

    wrkopt = wrkopt > np*(n + 1) ? wrkopt : np*(n + 1);

    i32 itau2 = i23 + np;
    i32 jwork2 = itau2 + n;
    i32 k_param = (n - mp1 > 0) ? n - mp1 : 0;
    i32 one_val = 1;
    i32 ldwork_remaining = ldwork - jwork2;

    mb04id(np, n, k_param, one_val, dwork, ldw,
           dwork + i23, ldw, dwork + itau2,
           dwork + jwork2, ldwork_remaining, info);

    i32 wrkopt2 = (i32)dwork[jwork2] + jwork2;
    wrkopt = wrkopt2 > wrkopt ? wrkopt2 : wrkopt;

    SLC_DLACPY("Upper", &n, &n, dwork, &ldw, sinv, &ldsinv);
    SLC_DCOPY(&n, dwork + i23, &(i32){1}, x, &(i32){1});
    if (p > 0) {
        SLC_DCOPY(&p, dwork + i23 + n, &(i32){1}, e, &(i32){1});
    }

    if (ljobx) {
        f64 rcond = 0.0;
        mb02od("Left", "Upper", "No transpose", "Non-unit", "1-norm",
               n, 1, one, sinv, ldsinv, x, n, &rcond, tol, iwork, dwork, info);
        if (*info == 0) {
            wrkopt = wrkopt > 3*n ? wrkopt : 3*n;
            dwork[1] = rcond;
        }
    }

    dwork[0] = (f64)wrkopt;
}
