/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sb01fy(
    const bool discr,
    const i32 n,
    const i32 m,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    f64* f,
    const i32 ldf,
    f64* v,
    const i32 ldv,
    i32* info
)
{
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 ZERO = 0.0;

    *info = 0;

    f64 at[4], dummy[4], u[4];
    f64 cs, sn, r11, r12, r22, scale, temp;

    ma02ad("Full", n, m, b, ldb, f, ldf);

    if (n == 1) {
        if (m > 1) {
            SLC_DLARFG(&m, &f[0], &f[1], &ldf, &temp);
        }
        r11 = fabs(f[0]);

        if (discr) {
            temp = fabs(a[0]);
            if (temp <= ONE) {
                *info = 2;
                return;
            } else {
                temp = r11 / sqrt((temp - ONE) * (temp + ONE));
            }
        } else {
            if (a[0] <= ZERO) {
                *info = 2;
                return;
            } else {
                temp = r11 / sqrt(fabs(TWO * a[0]));
            }
        }
        u[0] = temp;
        scale = ONE;
    } else {
        if (m > 1) {
            i32 nm1 = n - 1;
            SLC_DLARFG(&m, &f[0], &f[1], &ldf, &temp);
            SLC_DLATZM("Left", &m, &nm1, &f[1], &ldf, &temp, &f[0 + 1*ldf],
                       &f[1 + 1*ldf], &ldf, v);
        }
        r11 = f[0];
        r12 = f[0 + 1*ldf];
        if (m > 2) {
            i32 mm1 = m - 1;
            SLC_DLARFG(&mm1, &f[1 + 1*ldf], &f[2 + 1*ldf], &ldf, &temp);
        }
        if (m == 1) {
            r22 = ZERO;
        } else {
            r22 = f[1 + 1*ldf];
        }

        at[0 + 0*2] = a[0 + 0*lda];
        at[0 + 1*2] = a[1 + 0*lda];
        at[1 + 0*2] = a[0 + 1*lda];
        at[1 + 1*2] = a[1 + 1*lda];

        u[0 + 0*2] = r11;
        u[0 + 1*2] = r12;
        u[1 + 1*2] = r22;

        i32 isgn = -1;
        sb03oy(discr, false, isgn, at, 2, u, 2, dummy, 2, &scale, info);

        if (*info != 0) {
            if (*info != 4) {
                *info = 2;
            } else {
                *info = 3;
            }
            return;
        }
    }

    for (i32 i = 0; i < n; i++) {
        if (u[i + i*2] == ZERO) {
            *info = 1;
            return;
        }
    }

    i32 mm = m;
    SLC_DLASET("Upper", &mm, &mm, &ZERO, &ONE, v, &ldv);

    if (discr) {
        for (i32 i = 0; i < m; i++) {
            f[i + 0*ldf] = b[0 + i*ldb] / u[0] * scale;
        }
        if (n == 2) {
            for (i32 i = 0; i < m; i++) {
                f[i + 1*ldf] = (b[1 + i*ldb] - f[i + 0*ldf] * u[0 + 1*2]) / u[1 + 1*2] * scale;
            }
            mb04ox(m, v, ldv, &f[0 + 1*ldf], 1);
        }
        mb04ox(m, v, ldv, &f[0 + 0*ldf], 1);

        i32 mm_val = m;
        i32 ldv_val = ldv;
        SLC_DTRTRI("Upper", "NonUnit", &mm_val, v, &ldv_val, info);
    }

    if (n == 1) {
        if (discr) {
            temp = -a[0];
            r11 = SLC_DLAPY2(&u[0], &r11);
            for (i32 i = 0; i < m; i++) {
                f[i + 0*ldf] = ((b[0 + i*ldb] / r11) / r11) * temp;
            }
        } else {
            r11 = u[0];
            for (i32 i = 0; i < m; i++) {
                f[i + 0*ldf] = -((b[0 + i*ldb] / r11) / r11);
            }
        }
    } else {
        if (discr) {
            temp = u[0];
            SLC_DROTG(&r11, &temp, &cs, &sn);
            temp = -sn * r12 + cs * u[0 + 1*2];
            r12 = cs * r12 + sn * u[0 + 1*2];
            r22 = SLC_DLAPY3(&r22, &temp, &u[1 + 1*2]);
        } else {
            r11 = u[0];
            r12 = u[0 + 1*2];
            r22 = u[1 + 1*2];
        }

        for (i32 i = 0; i < m; i++) {
            f[i + 0*ldf] = -b[0 + i*ldb] / r11;
            f[i + 1*ldf] = -(b[1 + i*ldb] + f[i + 0*ldf] * r12) / r22;
            f[i + 1*ldf] = f[i + 1*ldf] / r22;
            f[i + 0*ldf] = (f[i + 0*ldf] - f[i + 1*ldf] * r12) / r11;
        }

        if (discr) {
            for (i32 i = 0; i < m; i++) {
                temp = f[i + 0*ldf] * a[0 + 0*lda] + f[i + 1*ldf] * a[1 + 0*lda];
                f[i + 1*ldf] = f[i + 0*ldf] * a[0 + 1*lda] + f[i + 1*ldf] * a[1 + 1*lda];
                f[i + 0*ldf] = temp;
            }
        }
    }
}
