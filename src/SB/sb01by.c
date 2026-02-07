/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sb01by(
    const i32 n,
    const i32 m,
    const f64 s,
    const f64 p,
    f64* a,
    f64* b,
    f64* f,
    const f64 tol,
    f64* dwork,
    i32* info
)
{
    const f64 FOUR = 4.0;
    const f64 ONE = 1.0;
    const f64 THREE = 3.0;
    const f64 ZERO = 0.0;

    i32 ir, j;
    f64 absr, b1, b2, b21, c, c0, c1, c11, c12, c21, c22, c3, c4;
    f64 cs, cu, cv, dc0, dc2, dc3, diffr, r, rn, s12, s21, sig;
    f64 sn, su, sv, tau1, tau2, wi, wi1, wr, wr1, x, y, z;

    *info = 0;

    if (n == 1) {
        i32 one = 1;
        i32 m_val = m;

        if (m > 1) {
            SLC_DLARFG(&m_val, &b[0], &b[n], &n, &tau1);
        }

        b1 = b[0];
        if (fabs(b1) <= tol) {
            *info = 1;
            return;
        }

        f[0] = (s - a[0]) / b1;

        if (m > 1) {
            i32 mm1 = m - 1;
            SLC_DLASET("Full", &mm1, &one, &ZERO, &ZERO, &f[1], &m_val);
            SLC_DLATZM("Left", &m_val, &one, &b[n], &n, &tau1, &f[0], &f[1], &m_val, dwork);
        }
        return;
    }

    if (m == 1) {
        b1 = b[0];
        b21 = b[1];
        b2 = ZERO;
    } else {
        i32 nm1 = n - 1;
        i32 m_val = m;

        SLC_DLARFG(&m_val, &b[0], &b[n], &n, &tau1);
        SLC_DLATZM("Right", &nm1, &m_val, &b[n], &n, &tau1, &b[1], &b[1 + n], &n, dwork);

        b1 = b[0];
        b21 = b[1];

        if (m > 2) {
            i32 mm1 = m - 1;
            SLC_DLARFG(&mm1, &b[1 + n], &b[1 + 2*n], &n, &tau2);
        }
        b2 = b[1 + n];
    }

    SLC_DLASV2(&b1, &b21, &b2, &x, &y, &su, &cu, &sv, &cv);
    su = -su;
    b1 = y;
    b2 = x;

    {
        i32 two = 2;
        i32 one = 1;
        SLC_DROT(&two, &a[1], &two, &a[0], &two, &cu, &su);
        SLC_DROT(&two, &a[n], &one, &a[0], &one, &cu, &su);
    }

    ir = 0;
    if (fabs(b2) > tol) ir++;
    if (fabs(b1) > tol) ir++;

    if (ir == 0 || (ir == 1 && fabs(a[1]) <= tol)) {
        f[0] = cu;
        f[1] = -su;
        *info = 1;
        return;
    }

    x = SLC_DLAMC3(&b1, &b2);
    if (x == b1) {
        f[0] = (s - (a[0] + a[1 + n])) / b1;
        f[1] = -(a[1 + n] * (a[1 + n] - s) + a[1] * a[n] + p) / a[1] / b1;

        if (m > 1) {
            f[m] = ZERO;
            f[m + 1] = ZERO;
        }
    } else {
        z = (s - (a[0] + a[1 + n])) / (b1 * b1 + b2 * b2);
        f[0] = b1 * z;
        f[m + 1] = b2 * z;

        x = a[0] + b1 * f[0];
        c = x * (s - x) - p;

        if (c >= ZERO) {
            sig = ONE;
        } else {
            sig = -ONE;
        }

        s12 = b1 / b2;
        s21 = b2 / b1;
        c11 = ZERO;
        c12 = ONE;
        c21 = sig * s12 * c;
        c22 = a[n] - sig * s12 * a[1];

        SLC_DLANV2(&c11, &c12, &c21, &c22, &wr, &wi, &wr1, &wi1, &cs, &sn);

        if (fabs(wr - a[n]) > fabs(wr1 - a[n])) {
            r = wr1;
        } else {
            r = wr;
        }

        c0 = -c * c;
        c1 = c * a[1];
        c4 = s21 * s21;
        c3 = -c4 * a[n];
        dc0 = c1;
        dc2 = THREE * c3;
        dc3 = FOUR * c4;

        for (j = 0; j < 10; j++) {
            x = c0 + r * (c1 + r * r * (c3 + r * c4));
            y = dc0 + r * r * (dc2 + r * dc3);
            if (y == ZERO) break;
            rn = r - x / y;
            absr = fabs(r);
            diffr = fabs(r - rn);
            z = SLC_DLAMC3(&absr, &diffr);
            if (z == absr) break;
            r = rn;
        }

        if (r == ZERO) {
            r = SLC_DLAMCH("Epsilon");
        }

        f[m] = (r - a[n]) / b1;
        f[1] = (c / r - a[1]) / b2;
    }

    {
        i32 min_m_2 = (m < 2) ? m : 2;
        i32 one = 1;
        SLC_DROT(&min_m_2, &f[0], &one, &f[m], &one, &cu, &su);
    }

    if (m == 1) {
        return;
    }

    {
        i32 two = 2;
        SLC_DROT(&two, &f[1], &m, &f[0], &m, &cv, &sv);
    }

    if (m > n) {
        i32 mn = m - n;
        SLC_DLASET("Full", &mn, &n, &ZERO, &ZERO, &f[n], &m);
    }

    if (m > 2) {
        i32 mm1 = m - 1;
        i32 two = 2;
        SLC_DLATZM("Left", &mm1, &two, &b[1 + 2*n], &n, &tau2, &f[1], &f[2], &m, dwork);
    }

    {
        i32 two = 2;
        SLC_DLATZM("Left", &m, &two, &b[n], &n, &tau1, &f[0], &f[1], &m, dwork);
    }
}
