/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdio.h>

/* #define MB03AF_DEBUG 1 */

void mb03af(const char *shft, i32 k, i32 n, const i32 *amap, const i32 *s,
            i32 sinv, const f64 *a, i32 lda1, i32 lda2, f64 *c1, f64 *s1,
            f64 *c2, f64 *s2) {
    const f64 ONE = 1.0;

    char shft_upper = (char)toupper((unsigned char)shft[0]);
    bool sgle = (shft_upper == 'S');

    i32 m = n - 1;
    i32 ldas = lda1 * lda2;
    i32 ai = amap[k - 1] - 1;
    const f64 *a_slice = a + ai * ldas;

    f64 temp;
    f64 alpha, beta, gamma, delta, epsil, eta, theta, zeta;
    f64 c1_v, s1_v, c2_v, s2_v;
    f64 c2r, s2r, c3, s3, c3r, s3r, c4, s4, c4r, s4r;
    f64 c5, s5, c5r, s5r, c6, s6, c6r, s6r;
    f64 cx, sx, cs, ss, sss, ssss;
    f64 val1, val2, val3, val4, val5;

    f64 a11 = a_slice[0];
    f64 a21 = a_slice[1];
#ifdef MB03AF_DEBUG
    fprintf(stderr, "MB03AF: Initial: ai=%d, a11=%.6e, a21=%.6e\n", ai, a11, a21);
#endif
    SLC_DLARTG(&a11, &a21, &c1_v, &s1_v, &temp);
    SLC_DLARTG(&temp, &ONE, &c2_v, &s2_v, &temp);

    for (i32 i = k - 2; i >= 0; i--) {
        ai = amap[i] - 1;
        a_slice = a + ai * ldas;
        f64 a_11 = a_slice[0];
        f64 a_12 = a_slice[lda1];
        f64 a_22 = a_slice[1 + lda1];
        f64 a_nn = a_slice[(n - 1) + (n - 1) * lda1];
#ifdef MB03AF_DEBUG
        fprintf(stderr, "MB03AF: Loop i=%d, ai=%d, s[ai]=%d, a(1,1)=%.4e, a(1,2)=%.4e, a(2,2)=%.4e, a(n,n)=%.4e\n",
                i, ai, s[ai], a_11, a_12, a_22, a_nn);
#endif

        if (s[ai] == sinv) {
            alpha = c2_v * (c1_v * a_11 + s1_v * a_12);
            beta = s1_v * c2_v * a_22;
            gamma = s2_v * a_nn;
            SLC_DLARTG(&alpha, &beta, &c1_v, &s1_v, &temp);
            SLC_DLARTG(&temp, &gamma, &c2_v, &s2_v, &val1);
        } else {
            alpha = c1_v * s2_v * a_11;
            gamma = s1_v * a_11;
            beta = s2_v * (c1_v * a_12 + s1_v * a_22);
            delta = c1_v * a_22 - s1_v * a_12;
            SLC_DLARTG(&delta, &gamma, &c1_v, &s1_v, &temp);
            alpha = c1_v * alpha + s1_v * beta;
            beta = c2_v * a_nn;
            SLC_DLARTG(&beta, &alpha, &c2_v, &s2_v, &temp);
        }
    }

    ai = amap[k - 1] - 1;
    a_slice = a + ai * ldas;
    f64 a_nn = a_slice[(n - 1) + (n - 1) * lda1];
    alpha = s2_v * a_nn - c1_v * c2_v;
    beta = -s1_v * c2_v;

    if (sgle) {
        SLC_DLARTG(&alpha, &beta, &c1_v, &s1_v, &temp);
        *c1 = c1_v;
        *s1 = s1_v;
        *c2 = ONE;
        *s2 = 0.0;
        return;
    }

    f64 a_nm = a_slice[(n - 1) + (m - 1) * lda1];
    gamma = -s2_v * a_nm;
    SLC_DLARTG(&alpha, &gamma, &c2_v, &s2_v, &temp);
    SLC_DLARTG(&temp, &beta, &c1_v, &s1_v, &temp);
    cx = c1_v * c2_v;
    sx = c1_v * s2_v;

    f64 a_mm = a_slice[(m - 1) + (m - 1) * lda1];
    f64 a_mn = a_slice[(m - 1) + (n - 1) * lda1];
    f64 a_32 = a_slice[2 + 1 * lda1];
    f64 a_21 = a_slice[1];

    beta = s1_v * a_nm;
    alpha = cx * a_nm + sx * a_nn;
    gamma = s1_v * a_mm;
    delta = cx * a_mm + sx * a_mn;
    val1 = s1_v * a_32;
    val2 = cx * a_21 + s1_v * a_slice[1 + lda1];
    val3 = cx * a_slice[0] + s1_v * a_slice[lda1];

    SLC_DLARTG(&alpha, &beta, &c1_v, &s1_v, &temp);
    SLC_DLARTG(&gamma, &temp, &c2_v, &s2_v, &temp);
    SLC_DLARTG(&delta, &temp, &c3, &s3, &temp);
    SLC_DLARTG(&val1, &temp, &c4, &s4, &temp);
    SLC_DLARTG(&val2, &temp, &c5, &s5, &temp);
    SLC_DLARTG(&val3, &temp, &c6, &s6, &temp);

    for (i32 i = k - 2; i >= 0; i--) {
        ai = amap[i] - 1;
        a_slice = a + ai * ldas;
        f64 a_ii_11 = a_slice[0];
        f64 a_ii_12 = a_slice[lda1];
        f64 a_ii_13 = a_slice[2 * lda1];
        f64 a_ii_22 = a_slice[1 + lda1];
        f64 a_ii_23 = a_slice[1 + 2 * lda1];
        f64 a_ii_33 = a_slice[2 + 2 * lda1];
        f64 a_ii_mm = a_slice[(m - 1) + (m - 1) * lda1];
        f64 a_ii_mn = a_slice[(m - 1) + (n - 1) * lda1];
        f64 a_ii_nn = a_slice[(n - 1) + (n - 1) * lda1];

        if (s[ai] == sinv) {
            ss = s3 * s4;
            sss = s2_v * ss;
            ssss = s1_v * sss;
            val1 = c4 * a_ii_13;
            val2 = c4 * a_ii_23;
            val3 = c4 * a_ii_33;
            alpha = s4 * c3 * a_ii_mm + sss * c1_v * a_ii_mn;
            beta = ss * c2_v * a_ii_mm + ssss * a_ii_mn;
            gamma = sss * c1_v * a_ii_nn;
            delta = ssss * a_ii_nn;

            ss = s5 * s6;
            cs = c5 * s6;
            val1 = ss * val1 + cs * a_ii_12 + c6 * a_ii_11;
            val2 = ss * val2 + cs * a_ii_22;
            val3 = ss * val3;
            alpha = ss * alpha;
            beta = ss * beta;
            gamma = ss * gamma;
            delta = ss * delta;

            SLC_DLARTG(&gamma, &delta, &c1_v, &s1_v, &temp);
            SLC_DLARTG(&beta, &temp, &c2_v, &s2_v, &temp);
            SLC_DLARTG(&alpha, &temp, &c3, &s3, &temp);
            SLC_DLARTG(&val3, &temp, &c4, &s4, &temp);
            SLC_DLARTG(&val2, &temp, &c5, &s5, &temp);
            SLC_DLARTG(&val1, &temp, &c6, &s6, &temp);
        } else {
            delta = c1_v * a_ii_nn;
            epsil = s1_v * a_ii_nn;

            alpha = c2_v * a_ii_mm;
            beta = s2_v * delta;
            gamma = -s2_v * a_ii_mm;
            zeta = c2_v * a_ii_mn + s2_v * epsil;
            eta = -s2_v * a_ii_mn + c2_v * epsil;

            delta = c1_v * c2_v * delta + s1_v * eta;
            SLC_DLARTG(&delta, &(f64){-gamma}, &c2r, &s2r, &temp);

            delta = c3 * a_ii_mm;
            epsil = s3 * alpha;
            eta = c3 * a_ii_mn + s3 * beta;
            theta = s3 * zeta;
            gamma = -s3 * a_ii_mm;
            beta = -s3 * a_ii_mn + c3 * beta;

            alpha = c2r * c3 * alpha + s2r * (c1_v * beta + s1_v * c3 * zeta);
            SLC_DLARTG(&alpha, &(f64){-gamma}, &c3r, &s3r, &temp);

            val1 = c4 * a_ii_33;
            val2 = s4 * delta;
            val3 = s4 * epsil;
            val4 = s4 * eta;
            val5 = s4 * theta;
            beta = -s4 * a_ii_33;
            delta = c4 * delta;
            epsil = c4 * epsil;
            zeta = c4 * eta;
            eta = c4 * theta;

            alpha = c3r * delta + s3r * (c2r * epsil + s2r * (c1_v * zeta + s1_v * eta));
            SLC_DLARTG(&alpha, &(f64){-beta}, &c4r, &s4r, &temp);

            beta = c5 * a_ii_22;
            delta = c5 * a_ii_23 + s5 * val1;
            epsil = s5 * val2;
            zeta = s5 * val3;
            eta = s5 * val4;
            theta = s5 * val5;
            gamma = -s5 * a_ii_22;
            val1 = c5 * val1 - s5 * a_ii_23;
            val2 = c5 * val2;
            val3 = c5 * val3;
            val4 = c5 * val4;
            val5 = c5 * val5;

            alpha = c4r * val1 + s4r * (c3r * val2 + s3r * (c2r * val3 + s2r * (c1_v * val4 + s1_v * val5)));
            SLC_DLARTG(&alpha, &(f64){-gamma}, &c5r, &s5r, &temp);

            gamma = -s6 * a_ii_11;
            beta = c6 * beta - s6 * a_ii_12;
            delta = c6 * delta - s6 * a_ii_13;
            epsil = c6 * epsil;
            zeta = c6 * zeta;
            eta = c6 * eta;
            theta = c6 * theta;

            alpha = c5r * beta + s5r * (c4r * delta + s4r * (c3r * epsil + s3r * (c2r * zeta + s2r * (c1_v * eta + s1_v * theta))));
            SLC_DLARTG(&alpha, &(f64){-gamma}, &c6r, &s6r, &temp);

            c2_v = c2r;
            s2_v = s2r;
            c3 = c3r;
            s3 = s3r;
            c4 = c4r;
            s4 = s4r;
            c5 = c5r;
            s5 = s5r;
            c6 = c6r;
            s6 = s6r;
        }
    }

    val1 = s5 * s6;
    val2 = s4 * val1;
    val3 = s3 * val2;
    alpha = c3 * val2 - c6;
    beta = c2_v * val3 - c5 * s6;
    gamma = -c4 * val1;
    SLC_DLARTG(&beta, &gamma, &c2_v, &s2_v, &temp);
    SLC_DLARTG(&alpha, &temp, &c1_v, &s1_v, &val1);

    *c1 = c1_v;
    *s1 = s1_v;
    *c2 = c2_v;
    *s2 = s2_v;

#ifdef MB03AF_DEBUG
    fprintf(stderr, "MB03AF: shft=%c, k=%d, n=%d, sinv=%d\n", shft_upper, k, n, sinv);
    fprintf(stderr, "MB03AF: amap=[");
    for (i32 i = 0; i < k; i++) fprintf(stderr, "%d%s", amap[i], i < k-1 ? "," : "");
    fprintf(stderr, "], s=[");
    for (i32 i = 0; i < k; i++) fprintf(stderr, "%d%s", s[amap[i]-1], i < k-1 ? "," : "");
    fprintf(stderr, "]\n");
    fprintf(stderr, "MB03AF: c1=%.6e, s1=%.6e, c2=%.6e, s2=%.6e\n", *c1, *s1, *c2, *s2);
#endif
}
