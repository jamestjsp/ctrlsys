/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdbool.h>

void mb02fd(const char* typet, i32 k, i32 n, i32 p, i32 s,
            f64* t, i32 ldt, f64* r, i32 ldr,
            f64* dwork, i32 ldwork, i32* info)
{
    const f64 one = 1.0;

    char typet_u = (char)toupper((unsigned char)typet[0]);
    bool isrow = (typet_u == 'R');

    *info = 0;

    if (!isrow && typet_u != 'C') {
        *info = -1;
    } else if (k < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (p < 0 || p > n) {
        *info = -4;
    } else if (s < 0 || s > (n - p)) {
        *info = -5;
    } else if (ldt < 1 || (isrow && ldt < k) ||
               (!isrow && ldt < (n - p) * k)) {
        *info = -7;
    } else if (ldr < 1 ||
               (isrow && p == 0 && ldr < s * k) ||
               (isrow && p > 0 && ldr < (s + 1) * k) ||
               (!isrow && p == 0 && ldr < n * k) ||
               (!isrow && p > 0 && ldr < (n - p + 1) * k)) {
        *info = -9;
    } else {
        i32 countr;
        if (p == 0) {
            countr = (n + 1) * k;
        } else {
            countr = (n - p + 2) * k;
        }
        i32 fourk = 4 * k;
        if (countr < fourk) countr = fourk;

        if (ldwork < 1 || ldwork < countr) {
            dwork[0] = (f64)(countr > 1 ? countr : 1);
            *info = -11;
        }
    }

    if (*info != 0) {
        return;
    }

    i32 min_kns = k;
    if (n < min_kns) min_kns = n;
    if (s < min_kns) min_kns = s;

    if (min_kns == 0) {
        dwork[0] = one;
        return;
    }

    i32 maxwrk = 1;
    i32 ierr = 0;

    if (isrow) {
        i32 st, countr, startr;

        if (p == 0) {
            SLC_DPOTRF("U", &k, t, &ldt, &ierr);
            if (ierr != 0) {
                *info = 1;
                return;
            }

            if (n > 1) {
                i32 cols = (n - 1) * k;
                SLC_DTRSM("L", "U", "T", "N", &k, &cols, &one, t, &ldt, &t[k * ldt], &ldt);
            }

            i32 nk = n * k;
            SLC_DLACPY("U", &k, &nk, t, &ldt, r, &ldr);

            if (s == 1) {
                dwork[0] = one;
                return;
            }

            st = 2;
            countr = (n - 1) * k;
        } else {
            st = 1;
            countr = (n - p) * k;
        }

        startr = 0;

        for (i32 i = st; i <= s; i++) {
            SLC_DLACPY("U", &k, &countr, &r[startr + startr * ldr], &ldr,
                       &r[(startr + k) + (startr + k) * ldr], &ldr);
            startr += k;
            countr -= k;

            i32 threek = 3 * k;
            i32 work_remain = ldwork - threek;
            mb02cx("R", k, k, k, &r[startr + startr * ldr], ldr,
                   &t[startr * ldt], ldt, dwork, threek, &dwork[threek],
                   work_remain, &ierr);

            if (ierr != 0) {
                *info = 1;
                return;
            }

            i32 wrk = (i32)dwork[threek] + threek;
            if (wrk > maxwrk) maxwrk = wrk;

            mb02cy("R", "N", k, k, countr, k,
                   &r[startr + (startr + k) * ldr], ldr,
                   &t[(startr + k) * ldt], ldt,
                   &t[startr * ldt], ldt,
                   dwork, threek, &dwork[threek], work_remain, &ierr);

            wrk = (i32)dwork[threek] + threek;
            if (wrk > maxwrk) maxwrk = wrk;
        }
    } else {
        i32 st, countr, startr;

        if (p == 0) {
            SLC_DPOTRF("L", &k, t, &ldt, &ierr);
            if (ierr != 0) {
                *info = 1;
                return;
            }

            if (n > 1) {
                i32 rows = (n - 1) * k;
                SLC_DTRSM("R", "L", "T", "N", &rows, &k, &one, t, &ldt, &t[k], &ldt);
            }

            i32 nk = n * k;
            SLC_DLACPY("L", &nk, &k, t, &ldt, r, &ldr);

            if (s == 1) {
                dwork[0] = one;
                return;
            }

            st = 2;
            countr = (n - 1) * k;
        } else {
            st = 1;
            countr = (n - p) * k;
        }

        startr = 0;

        for (i32 i = st; i <= s; i++) {
            SLC_DLACPY("L", &countr, &k, &r[startr + startr * ldr], &ldr,
                       &r[(startr + k) + (startr + k) * ldr], &ldr);
            startr += k;
            countr -= k;

            i32 threek = 3 * k;
            i32 work_remain = ldwork - threek;
            mb02cx("C", k, k, k, &r[startr + startr * ldr], ldr,
                   &t[startr], ldt, dwork, threek, &dwork[threek],
                   work_remain, &ierr);

            if (ierr != 0) {
                *info = 1;
                return;
            }

            i32 wrk = (i32)dwork[threek] + threek;
            if (wrk > maxwrk) maxwrk = wrk;

            mb02cy("C", "N", k, k, countr, k,
                   &r[(startr + k) + startr * ldr], ldr,
                   &t[startr + k], ldt,
                   &t[startr], ldt,
                   dwork, threek, &dwork[threek], work_remain, &ierr);

            wrk = (i32)dwork[threek] + threek;
            if (wrk > maxwrk) maxwrk = wrk;
        }
    }

    dwork[0] = (f64)maxwrk;
}
