// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb03ts(bool isham, bool wantu, i32 n, f64 *a, i32 lda, f64 *g, i32 ldg,
            f64 *u1, i32 ldu1, f64 *u2, i32 ldu2, i32 j1, i32 n1, i32 n2,
            f64 *dwork, i32 *info) {
    const f64 ZERO = 0.0;
    const f64 HALF = 0.5;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 THIRTY = 30.0;
    const f64 FORTY = 40.0;
    const i32 LDD = 4;
    const i32 LDX = 2;

    i32 int1 = 1;
    bool lblk;
    i32 ierr, j2, j3, j4, k, nd;
    f64 a11, a22, a33, cs, dnorm, eps, scale, smlnum;
    f64 sn, tau, tau1, tau2, temp, thresh, wi1, wi2, wr1, wr2, xnorm;
    f64 d[LDD * 4];
    f64 v[3], v1[3], v2[3], x[LDX * 2];

    *info = 0;

    if (n == 0 || n1 == 0 || n2 == 0) {
        return;
    }

    lblk = (j1 + n1 > n);

    j2 = j1 + 1;
    j3 = j1 + 2;
    j4 = j1 + 3;

    i32 j1c = j1 - 1;
    i32 j2c = j2 - 1;
    i32 j3c = j3 - 1;
    i32 j4c = j4 - 1;

    if (lblk && n1 == 1) {
        if (isham) {
            a11 = a[(n-1) + (n-1)*lda];
            f64 temp_g = g[(n-1) + (n-1)*ldg];
            f64 two_a11 = -TWO * a11;
            SLC_DLARTG(&temp_g, &two_a11, &cs, &sn, &temp);
            i32 nm1 = n - 1;
            SLC_DROT(&nm1, &a[0 + (n-1)*lda], &int1, &g[0 + (n-1)*ldg], &int1, &cs, &sn);
            a[(n-1) + (n-1)*lda] = -a11;
            if (wantu) {
                SLC_DROT(&n, &u1[0 + (n-1)*ldu1], &int1, &u2[0 + (n-1)*ldu2], &int1, &cs, &sn);
            }
        } else {
            i32 nm1 = n - 1;
            SLC_DSWAP(&nm1, &a[0 + (n-1)*lda], &int1, &g[0 + (n-1)*ldg], &int1);
            f64 neg_one = -ONE;
            SLC_DSCAL(&nm1, &neg_one, &a[0 + (n-1)*lda], &int1);
            if (wantu) {
                SLC_DSWAP(&n, &u1[0 + (n-1)*ldu1], &int1, &u2[0 + (n-1)*ldu2], &int1);
                SLC_DSCAL(&n, &neg_one, &u1[0 + (n-1)*ldu1], &int1);
            }
        }

    } else if (lblk && n1 == 2) {
        if (isham) {
            nd = 4;
            i32 two = 2;
            SLC_DLACPY("Full", &two, &two, &a[(n-2) + (n-2)*lda], &lda, d, &LDD);
            SLC_DLASET("All", &two, &two, &ZERO, &ZERO, &d[2 + 0*LDD], &LDD);
            SLC_DLACPY("Upper", &two, &two, &g[(n-2) + (n-2)*ldg], &ldg, &d[0 + 2*LDD], &LDD);
            d[1 + 2*LDD] = d[0 + 3*LDD];
            d[2 + 2*LDD] = -d[0 + 0*LDD];
            d[3 + 2*LDD] = -d[0 + 1*LDD];
            d[2 + 3*LDD] = -d[1 + 0*LDD];
            d[3 + 3*LDD] = -d[1 + 1*LDD];
            dnorm = SLC_DLANGE("Max", &nd, &nd, d, &LDD, dwork);

            eps = SLC_DLAMCH("P");
            smlnum = SLC_DLAMCH("S") / eps;
            thresh = fmax(FORTY * eps * dnorm, smlnum);

            i32 ltranl = 0, ltranr = 0, isgn = -1;
            SLC_DLASY2(&ltranl, &ltranr, &isgn, &two, &two, d, &LDD, &d[2 + 2*LDD],
                       &LDD, &d[0 + 2*LDD], &LDD, &scale, x, &LDX, &xnorm, &ierr);

            temp = -x[0 + 0*LDX];
            SLC_DLARTG(&temp, &scale, &v1[0], &v2[0], &x[0 + 0*LDX]);
            SLC_DLARTG(&x[0 + 0*LDX], &(f64){-x[1 + 0*LDX]}, &v1[1], &v2[1], &temp);
            x[0 + 1*LDX] = -x[0 + 1*LDX];
            x[1 + 1*LDX] = -x[1 + 1*LDX];
            x[0 + 0*LDX] = ZERO;
            x[1 + 0*LDX] = scale;
            SLC_DROT(&int1, &x[0 + 1*LDX], &int1, &x[0 + 0*LDX], &int1, &v1[0], &v2[0]);
            SLC_DROT(&int1, &x[0 + 1*LDX], &int1, &x[1 + 1*LDX], &int1, &v1[1], &v2[1]);
            SLC_DROT(&int1, &x[0 + 0*LDX], &int1, &x[1 + 0*LDX], &int1, &v1[1], &v2[1]);
            SLC_DLARTG(&x[1 + 1*LDX], &x[1 + 0*LDX], &v1[2], &v2[2], &temp);

            i32 four = 4;
            SLC_DROT(&four, &d[0 + 0*LDD], &LDD, &d[2 + 0*LDD], &LDD, &v1[0], &v2[0]);
            SLC_DROT(&four, &d[0 + 0*LDD], &LDD, &d[1 + 0*LDD], &LDD, &v1[1], &v2[1]);
            SLC_DROT(&four, &d[2 + 0*LDD], &LDD, &d[3 + 0*LDD], &LDD, &v1[1], &v2[1]);
            SLC_DROT(&four, &d[1 + 0*LDD], &LDD, &d[3 + 0*LDD], &LDD, &v1[2], &v2[2]);
            SLC_DROT(&four, &d[0 + 0*LDD], &int1, &d[0 + 2*LDD], &int1, &v1[0], &v2[0]);
            SLC_DROT(&four, &d[0 + 0*LDD], &int1, &d[0 + 1*LDD], &int1, &v1[1], &v2[1]);
            SLC_DROT(&four, &d[0 + 2*LDD], &int1, &d[0 + 3*LDD], &int1, &v1[1], &v2[1]);
            SLC_DROT(&four, &d[0 + 1*LDD], &int1, &d[0 + 3*LDD], &int1, &v1[2], &v2[2]);

            if (fmax(fmax(fabs(d[2 + 0*LDD]), fabs(d[2 + 1*LDD])),
                     fmax(fabs(d[3 + 0*LDD]), fabs(d[3 + 1*LDD]))) > thresh) {
                *info = 1;
                return;
            }

            SLC_DLACPY("All", &two, &two, d, &LDD, &a[(n-2) + (n-2)*lda], &lda);
            SLC_DLACPY("Upper", &two, &two, &d[0 + 2*LDD], &LDD, &g[(n-2) + (n-2)*ldg], &ldg);

            if (n > 2) {
                i32 nm2 = n - 2;
                SLC_DROT(&nm2, &a[0 + (n-2)*lda], &int1, &g[0 + (n-2)*ldg], &int1, &v1[0], &v2[0]);
                SLC_DROT(&nm2, &a[0 + (n-2)*lda], &int1, &a[0 + (n-1)*lda], &int1, &v1[1], &v2[1]);
                SLC_DROT(&nm2, &g[0 + (n-2)*ldg], &int1, &g[0 + (n-1)*ldg], &int1, &v1[1], &v2[1]);
                SLC_DROT(&nm2, &a[0 + (n-1)*lda], &int1, &g[0 + (n-1)*ldg], &int1, &v1[2], &v2[2]);
            }

            if (wantu) {
                SLC_DROT(&n, &u1[0 + (n-2)*ldu1], &int1, &u2[0 + (n-2)*ldu2], &int1, &v1[0], &v2[0]);
                SLC_DROT(&n, &u1[0 + (n-2)*ldu1], &int1, &u1[0 + (n-1)*ldu1], &int1, &v1[1], &v2[1]);
                SLC_DROT(&n, &u2[0 + (n-2)*ldu2], &int1, &u2[0 + (n-1)*ldu2], &int1, &v1[1], &v2[1]);
                SLC_DROT(&n, &u1[0 + (n-1)*ldu1], &int1, &u2[0 + (n-1)*ldu2], &int1, &v1[2], &v2[2]);
            }
        } else {
            if (fabs(a[(n-2) + (n-1)*lda]) > fabs(a[(n-1) + (n-2)*lda])) {
                temp = g[(n-2) + (n-1)*ldg];
                SLC_DLARTG(&temp, &a[(n-2) + (n-1)*lda], &cs, &sn, &g[(n-2) + (n-1)*ldg]);
                sn = -sn;
                i32 nm2 = n - 2;
                SLC_DROT(&nm2, &a[0 + (n-1)*lda], &int1, &g[0 + (n-1)*ldg], &int1, &cs, &sn);

                a[(n-2) + (n-1)*lda] = -sn * a[(n-1) + (n-2)*lda];
                temp = -cs * a[(n-1) + (n-2)*lda];
                a[(n-1) + (n-2)*lda] = g[(n-2) + (n-1)*ldg];
                g[(n-2) + (n-1)*ldg] = temp;
                if (wantu) {
                    SLC_DROT(&n, &u1[0 + (n-1)*ldu1], &int1, &u2[0 + (n-1)*ldu2], &int1, &cs, &sn);
                }
                SLC_DSWAP(&nm2, &a[0 + (n-2)*lda], &int1, &g[0 + (n-2)*ldg], &int1);
                f64 neg_one = -ONE;
                SLC_DSCAL(&nm2, &neg_one, &a[0 + (n-2)*lda], &int1);
                if (wantu) {
                    SLC_DSWAP(&n, &u1[0 + (n-2)*ldu1], &int1, &u2[0 + (n-2)*ldu2], &int1);
                    SLC_DSCAL(&n, &neg_one, &u1[0 + (n-2)*ldu1], &int1);
                }
            } else {
                temp = g[(n-2) + (n-1)*ldg];
                SLC_DLARTG(&temp, &a[(n-1) + (n-2)*lda], &cs, &sn, &g[(n-2) + (n-1)*ldg]);
                i32 nm2 = n - 2;
                SLC_DROT(&nm2, &a[0 + (n-2)*lda], &int1, &g[0 + (n-2)*ldg], &int1, &cs, &sn);
                a[(n-1) + (n-2)*lda] = -sn * a[(n-2) + (n-1)*lda];
                a[(n-2) + (n-1)*lda] = cs * a[(n-2) + (n-1)*lda];
                if (wantu) {
                    SLC_DROT(&n, &u1[0 + (n-2)*ldu1], &int1, &u2[0 + (n-2)*ldu2], &int1, &cs, &sn);
                }
                i32 nm1 = n - 1;
                SLC_DSWAP(&nm1, &a[0 + (n-1)*lda], &int1, &g[0 + (n-1)*ldg], &int1);
                f64 neg_one = -ONE;
                SLC_DSCAL(&nm1, &neg_one, &a[0 + (n-1)*lda], &int1);
                if (wantu) {
                    SLC_DSWAP(&n, &u1[0 + (n-1)*ldu1], &int1, &u2[0 + (n-1)*ldu2], &int1);
                    SLC_DSCAL(&n, &neg_one, &u1[0 + (n-1)*ldu1], &int1);
                }
            }
        }

        SLC_DLANV2(&a[(n-2) + (n-2)*lda], &a[(n-2) + (n-1)*lda], &a[(n-1) + (n-2)*lda],
                   &a[(n-1) + (n-1)*lda], &wr1, &wi1, &wr2, &wi2, &cs, &sn);
        i32 nm2 = n - 2;
        SLC_DROT(&nm2, &a[0 + (n-2)*lda], &int1, &a[0 + (n-1)*lda], &int1, &cs, &sn);
        if (isham) {
            temp = g[(n-2) + (n-1)*ldg];
            i32 nm1 = n - 1;
            SLC_DROT(&nm1, &g[0 + (n-2)*ldg], &int1, &g[0 + (n-1)*ldg], &int1, &cs, &sn);
            tau = cs * temp + sn * g[(n-1) + (n-1)*ldg];
            g[(n-1) + (n-1)*ldg] = cs * g[(n-1) + (n-1)*ldg] - sn * temp;
            g[(n-2) + (n-2)*ldg] = cs * g[(n-2) + (n-2)*ldg] + sn * tau;
            SLC_DROT(&int1, &g[(n-2) + (n-1)*ldg], &ldg, &g[(n-1) + (n-1)*ldg], &ldg, &cs, &sn);
        } else {
            SLC_DROT(&nm2, &g[0 + (n-2)*ldg], &int1, &g[0 + (n-1)*ldg], &int1, &cs, &sn);
        }
        if (wantu) {
            SLC_DROT(&n, &u1[0 + (n-2)*ldu1], &int1, &u1[0 + (n-1)*ldu1], &int1, &cs, &sn);
            SLC_DROT(&n, &u2[0 + (n-2)*ldu2], &int1, &u2[0 + (n-1)*ldu2], &int1, &cs, &sn);
        }

    } else if (n1 == 1 && n2 == 1) {
        a11 = a[j1c + j1c*lda];
        a22 = a[j2c + j2c*lda];

        f64 diff = a22 - a11;
        SLC_DLARTG(&a[j1c + j2c*lda], &diff, &cs, &sn, &temp);

        if (j3c < n) {
            i32 len = n - j1 - 1;
            SLC_DROT(&len, &a[j1c + j3c*lda], &lda, &a[j2c + j3c*lda], &lda, &cs, &sn);
        }
        i32 j1m1 = j1 - 1;
        SLC_DROT(&j1m1, &a[0 + j1c*lda], &int1, &a[0 + j2c*lda], &int1, &cs, &sn);

        a[j1c + j1c*lda] = a22;
        a[j2c + j2c*lda] = a11;

        if (isham) {
            temp = g[j1c + j2c*ldg];
            SLC_DROT(&j1, &g[0 + j1c*ldg], &int1, &g[0 + j2c*ldg], &int1, &cs, &sn);
            tau = cs * temp + sn * g[j2c + j2c*ldg];
            g[j2c + j2c*ldg] = cs * g[j2c + j2c*ldg] - sn * temp;
            g[j1c + j1c*ldg] = cs * g[j1c + j1c*ldg] + sn * tau;
            i32 len = n - j1;
            SLC_DROT(&len, &g[j1c + j2c*ldg], &ldg, &g[j2c + j2c*ldg], &ldg, &cs, &sn);
        } else {
            if (n > j1 + 1) {
                i32 len = n - j1 - 1;
                SLC_DROT(&len, &g[j1c + (j1c+2)*ldg], &ldg, &g[j2c + (j1c+2)*ldg], &ldg, &cs, &sn);
            }
            i32 j1m1 = j1 - 1;
            SLC_DROT(&j1m1, &g[0 + j1c*ldg], &int1, &g[0 + j2c*ldg], &int1, &cs, &sn);
        }
        if (wantu) {
            SLC_DROT(&n, &u1[0 + j1c*ldu1], &int1, &u1[0 + j2c*ldu1], &int1, &cs, &sn);
            SLC_DROT(&n, &u2[0 + j1c*ldu2], &int1, &u2[0 + j2c*ldu2], &int1, &cs, &sn);
        }

    } else {
        nd = n1 + n2;
        SLC_DLACPY("Full", &nd, &nd, &a[j1c + j1c*lda], &lda, d, &LDD);
        dnorm = SLC_DLANGE("Max", &nd, &nd, d, &LDD, dwork);

        eps = SLC_DLAMCH("P");
        smlnum = SLC_DLAMCH("S") / eps;
        thresh = fmax(THIRTY * eps * dnorm, smlnum);

        i32 ltranl = 0, ltranr = 0, isgn = -1;
        SLC_DLASY2(&ltranl, &ltranr, &isgn, &n1, &n2, d, &LDD, &d[n1 + n1*LDD],
                   &LDD, &d[0 + n1*LDD], &LDD, &scale, x, &LDX, &xnorm, &ierr);

        k = n1 + n1 + n2 - 3;

        if (k == 1) {
            // N1 = 1, N2 = 2: generate elementary reflector H so that:
            // ( scale, X11, X12 ) H = ( 0, 0, * ).
            v[0] = scale;
            v[1] = x[0 + 0*LDX];
            v[2] = x[0 + 1*LDX];
            i32 three = 3;
            SLC_DLARFG(&three, &v[2], v, &int1, &tau);
            v[2] = ONE;
            a11 = a[j1c + j1c*lda];

            SLC_DLARFX("Left", &three, &three, v, &tau, d, &LDD, dwork);
            SLC_DLARFX("Right", &three, &three, v, &tau, d, &LDD, dwork);

            if (fmax(fabs(d[2 + 0*LDD]), fmax(fabs(d[2 + 1*LDD]), fabs(d[2 + 2*LDD] - a11))) > thresh) {
                *info = 1;
                return;
            }

            i32 len = n - j1 + 1;
            SLC_DLARFX("Left", &three, &len, v, &tau, &a[j1c + j1c*lda], &lda, dwork);
            SLC_DLARFX("Right", &j2, &three, v, &tau, &a[0 + j1c*lda], &lda, dwork);

            a[j3c + j1c*lda] = ZERO;
            a[j3c + j2c*lda] = ZERO;
            a[j3c + j3c*lda] = a11;

            if (isham) {
                i32 j1m1 = j1 - 1;
                SLC_DLARFX("Right", &j1m1, &three, v, &tau, &g[0 + j1c*ldg], &ldg, dwork);
                SLC_DSYMV("Upper", &three, &tau, &g[j1c + j1c*ldg], &ldg, v, &int1, &ZERO, dwork, &int1);
                temp = -HALF * tau * SLC_DDOT(&three, dwork, &int1, v, &int1);
                SLC_DAXPY(&three, &temp, v, &int1, dwork, &int1);
                f64 neg_one = -ONE;
                SLC_DSYR2("Upper", &three, &neg_one, v, &int1, dwork, &int1, &g[j1c + j1c*ldg], &ldg);
                if (n > j1 + 2) {
                    i32 len = n - j1 - 2;
                    SLC_DLARFX("Left", &three, &len, v, &tau, &g[j1c + (j1c+3)*ldg], &ldg, dwork);
                }
            } else {
                i32 j1m1 = j1 - 1;
                SLC_DLARFX("Right", &j1m1, &three, v, &tau, &g[0 + j1c*ldg], &ldg, dwork);
                {i32 ign; mb01md('U', three, tau, &g[j1c + j1c*ldg], ldg, v, 1, ZERO, dwork, 1, &ign);}
                {i32 ign; mb01nd('U', three, ONE, v, 1, dwork, 1, &g[j1c + j1c*ldg], ldg, &ign);}
                if (n > j1 + 2) {
                    i32 len = n - j1 - 2;
                    SLC_DLARFX("Left", &three, &len, v, &tau, &g[j1c + (j1c+3)*ldg], &ldg, dwork);
                }
            }

            if (wantu) {
                SLC_DLARFX("R", &n, &three, v, &tau, &u1[0 + j1c*ldu1], &ldu1, dwork);
                SLC_DLARFX("R", &n, &three, v, &tau, &u2[0 + j1c*ldu2], &ldu2, dwork);
            }

        } else if (k == 2) {
            // N1 = 2, N2 = 1: generate elementary reflector H so that:
            // H ( -X11 )   ( * )
            //   ( -X21 ) = ( 0 )
            //   ( scale)   ( 0 )
            v[0] = -x[0 + 0*LDX];
            v[1] = -x[1 + 0*LDX];
            v[2] = scale;
            i32 three = 3;
            SLC_DLARFG(&three, &v[0], &v[1], &int1, &tau);
            v[0] = ONE;
            a33 = a[j3c + j3c*lda];

            SLC_DLARFX("L", &three, &three, v, &tau, d, &LDD, dwork);
            SLC_DLARFX("R", &three, &three, v, &tau, d, &LDD, dwork);

            if (fmax(fabs(d[1 + 0*LDD]), fmax(fabs(d[2 + 0*LDD]), fabs(d[0 + 0*LDD] - a33))) > thresh) {
                *info = 1;
                return;
            }

            SLC_DLARFX("Right", &j3, &three, v, &tau, &a[0 + j1c*lda], &lda, dwork);
            i32 len = n - j1;
            SLC_DLARFX("Left", &three, &len, v, &tau, &a[j1c + j2c*lda], &lda, dwork);

            a[j1c + j1c*lda] = a33;
            a[j2c + j1c*lda] = ZERO;
            a[j3c + j1c*lda] = ZERO;

            if (isham) {
                i32 j1m1 = j1 - 1;
                SLC_DLARFX("Right", &j1m1, &three, v, &tau, &g[0 + j1c*ldg], &ldg, dwork);
                SLC_DSYMV("Upper", &three, &tau, &g[j1c + j1c*ldg], &ldg, v, &int1, &ZERO, dwork, &int1);
                temp = -HALF * tau * SLC_DDOT(&three, dwork, &int1, v, &int1);
                SLC_DAXPY(&three, &temp, v, &int1, dwork, &int1);
                f64 neg_one = -ONE;
                SLC_DSYR2("Upper", &three, &neg_one, v, &int1, dwork, &int1, &g[j1c + j1c*ldg], &ldg);
                if (n > j1 + 2) {
                    i32 len = n - j1 - 2;
                    SLC_DLARFX("Left", &three, &len, v, &tau, &g[j1c + (j1c+3)*ldg], &ldg, dwork);
                }
            } else {
                i32 j1m1 = j1 - 1;
                SLC_DLARFX("Right", &j1m1, &three, v, &tau, &g[0 + j1c*ldg], &ldg, dwork);
                {i32 ign; mb01md('U', three, tau, &g[j1c + j1c*ldg], ldg, v, 1, ZERO, dwork, 1, &ign);}
                {i32 ign; mb01nd('U', three, ONE, v, 1, dwork, 1, &g[j1c + j1c*ldg], ldg, &ign);}
                if (n > j1 + 2) {
                    i32 len = n - j1 - 2;
                    SLC_DLARFX("Left", &three, &len, v, &tau, &g[j1c + (j1c+3)*ldg], &ldg, dwork);
                }
            }

            if (wantu) {
                SLC_DLARFX("R", &n, &three, v, &tau, &u1[0 + j1c*ldu1], &ldu1, dwork);
                SLC_DLARFX("R", &n, &three, v, &tau, &u2[0 + j1c*ldu2], &ldu2, dwork);
            }

        } else {
            v1[0] = -x[0 + 0*LDX];
            v1[1] = -x[1 + 0*LDX];
            v1[2] = scale;
            i32 three = 3;
            SLC_DLARFG(&three, &v1[0], &v1[1], &int1, &tau1);
            v1[0] = ONE;

            temp = -tau1 * (x[0 + 1*LDX] + v1[1] * x[1 + 1*LDX]);
            v2[0] = -temp * v1[1] - x[1 + 1*LDX];
            v2[1] = -temp * v1[2];
            v2[2] = scale;
            SLC_DLARFG(&three, &v2[0], &v2[1], &int1, &tau2);
            v2[0] = ONE;

            i32 four = 4;
            SLC_DLARFX("L", &three, &four, v1, &tau1, d, &LDD, dwork);
            SLC_DLARFX("R", &four, &three, v1, &tau1, d, &LDD, dwork);
            SLC_DLARFX("L", &three, &four, v2, &tau2, &d[1 + 0*LDD], &LDD, dwork);
            SLC_DLARFX("R", &four, &three, v2, &tau2, &d[0 + 1*LDD], &LDD, dwork);

            if (fmax(fmax(fabs(d[2 + 0*LDD]), fabs(d[2 + 1*LDD])),
                     fmax(fabs(d[3 + 0*LDD]), fabs(d[3 + 1*LDD]))) > thresh) {
                *info = 1;
                return;
            }

            i32 len = n - j1 + 1;
            SLC_DLARFX("L", &three, &len, v1, &tau1, &a[j1c + j1c*lda], &lda, dwork);
            SLC_DLARFX("R", &j4, &three, v1, &tau1, &a[0 + j1c*lda], &lda, dwork);
            SLC_DLARFX("L", &three, &len, v2, &tau2, &a[j2c + j1c*lda], &lda, dwork);
            SLC_DLARFX("R", &j4, &three, v2, &tau2, &a[0 + j2c*lda], &lda, dwork);

            a[j3c + j1c*lda] = ZERO;
            a[j3c + j2c*lda] = ZERO;
            a[j4c + j1c*lda] = ZERO;
            a[j4c + j2c*lda] = ZERO;

            if (isham) {
                i32 j1m1 = j1 - 1;
                SLC_DLARFX("Right", &j1m1, &three, v1, &tau1, &g[0 + j1c*ldg], &ldg, dwork);
                SLC_DSYMV("Upper", &three, &tau1, &g[j1c + j1c*ldg], &ldg, v1, &int1, &ZERO, dwork, &int1);
                temp = -HALF * tau1 * SLC_DDOT(&three, dwork, &int1, v1, &int1);
                SLC_DAXPY(&three, &temp, v1, &int1, dwork, &int1);
                f64 neg_one = -ONE;
                SLC_DSYR2("Upper", &three, &neg_one, v1, &int1, dwork, &int1, &g[j1c + j1c*ldg], &ldg);
                if (n > j1 + 2) {
                    i32 len = n - j1 - 2;
                    SLC_DLARFX("Left", &three, &len, v1, &tau1, &g[j1c + (j1c+3)*ldg], &ldg, dwork);
                }

                i32 j2m1 = j2 - 1;
                SLC_DLARFX("Right", &j2m1, &three, v2, &tau2, &g[0 + j2c*ldg], &ldg, dwork);
                SLC_DSYMV("Upper", &three, &tau2, &g[j2c + j2c*ldg], &ldg, v2, &int1, &ZERO, dwork, &int1);
                temp = -HALF * tau2 * SLC_DDOT(&three, dwork, &int1, v2, &int1);
                SLC_DAXPY(&three, &temp, v2, &int1, dwork, &int1);
                SLC_DSYR2("Upper", &three, &neg_one, v2, &int1, dwork, &int1, &g[j2c + j2c*ldg], &ldg);
                if (n > j2 + 2) {
                    i32 len = n - j2 - 2;
                    SLC_DLARFX("Left", &three, &len, v2, &tau2, &g[j2c + (j2c+3)*ldg], &ldg, dwork);
                }
            } else {
                i32 j1m1 = j1 - 1;
                SLC_DLARFX("Right", &j1m1, &three, v1, &tau1, &g[0 + j1c*ldg], &ldg, dwork);
                {i32 ign; mb01md('U', three, tau1, &g[j1c + j1c*ldg], ldg, v1, 1, ZERO, dwork, 1, &ign);}
                {i32 ign; mb01nd('U', three, ONE, v1, 1, dwork, 1, &g[j1c + j1c*ldg], ldg, &ign);}
                if (n > j1 + 2) {
                    i32 len = n - j1 - 2;
                    SLC_DLARFX("Left", &three, &len, v1, &tau1, &g[j1c + (j1c+3)*ldg], &ldg, dwork);
                }
                i32 j2m1 = j2 - 1;
                SLC_DLARFX("Right", &j2m1, &three, v2, &tau2, &g[0 + j2c*ldg], &ldg, dwork);
                {i32 ign; mb01md('U', three, tau2, &g[j2c + j2c*ldg], ldg, v2, 1, ZERO, dwork, 1, &ign);}
                {i32 ign; mb01nd('U', three, ONE, v2, 1, dwork, 1, &g[j2c + j2c*ldg], ldg, &ign);}
                if (n > j2 + 2) {
                    i32 len = n - j2 - 2;
                    SLC_DLARFX("Left", &three, &len, v2, &tau2, &g[j2c + (j2c+3)*ldg], &ldg, dwork);
                }
            }

            if (wantu) {
                SLC_DLARFX("R", &n, &three, v1, &tau1, &u1[0 + j1c*ldu1], &ldu1, dwork);
                SLC_DLARFX("R", &n, &three, v2, &tau2, &u1[0 + j2c*ldu1], &ldu1, dwork);
                SLC_DLARFX("R", &n, &three, v1, &tau1, &u2[0 + j1c*ldu2], &ldu2, dwork);
                SLC_DLARFX("R", &n, &three, v2, &tau2, &u2[0 + j2c*ldu2], &ldu2, dwork);
            }
        }

        if (n2 == 2) {
            SLC_DLANV2(&a[j1c + j1c*lda], &a[j1c + j2c*lda], &a[j2c + j1c*lda],
                       &a[j2c + j2c*lda], &wr1, &wi1, &wr2, &wi2, &cs, &sn);
            i32 len = n - j1 - 1;
            SLC_DROT(&len, &a[j1c + (j1c+2)*lda], &lda, &a[j2c + (j1c+2)*lda], &lda, &cs, &sn);
            i32 j1m1 = j1 - 1;
            SLC_DROT(&j1m1, &a[0 + j1c*lda], &int1, &a[0 + j2c*lda], &int1, &cs, &sn);
            if (isham) {
                temp = g[j1c + j2c*ldg];
                SLC_DROT(&j1, &g[0 + j1c*ldg], &int1, &g[0 + j2c*ldg], &int1, &cs, &sn);
                tau = cs * temp + sn * g[j2c + j2c*ldg];
                g[j2c + j2c*ldg] = cs * g[j2c + j2c*ldg] - sn * temp;
                g[j1c + j1c*ldg] = cs * g[j1c + j1c*ldg] + sn * tau;
                i32 len = n - j1;
                SLC_DROT(&len, &g[j1c + j2c*ldg], &ldg, &g[j2c + j2c*ldg], &ldg, &cs, &sn);
            } else {
                if (n > j1 + 1) {
                    i32 len = n - j1 - 1;
                    SLC_DROT(&len, &g[j1c + (j1c+2)*ldg], &ldg, &g[j2c + (j1c+2)*ldg], &ldg, &cs, &sn);
                }
                i32 j1m1 = j1 - 1;
                SLC_DROT(&j1m1, &g[0 + j1c*ldg], &int1, &g[0 + j2c*ldg], &int1, &cs, &sn);
            }
            if (wantu) {
                SLC_DROT(&n, &u1[0 + j1c*ldu1], &int1, &u1[0 + j2c*ldu1], &int1, &cs, &sn);
                SLC_DROT(&n, &u2[0 + j1c*ldu2], &int1, &u2[0 + j2c*ldu2], &int1, &cs, &sn);
            }
        }

        if (n1 == 2) {
            i32 j3l = j1 + n2;
            i32 j4l = j3l + 1;
            i32 j3lc = j3l - 1;
            i32 j4lc = j4l - 1;

            SLC_DLANV2(&a[j3lc + j3lc*lda], &a[j3lc + j4lc*lda], &a[j4lc + j3lc*lda],
                       &a[j4lc + j4lc*lda], &wr1, &wi1, &wr2, &wi2, &cs, &sn);
            if (j3l + 2 <= n) {
                i32 len = n - j3l - 1;
                SLC_DROT(&len, &a[j3lc + (j3lc+2)*lda], &lda, &a[j4lc + (j3lc+2)*lda], &lda, &cs, &sn);
            }
            i32 j3m1 = j3l - 1;
            SLC_DROT(&j3m1, &a[0 + j3lc*lda], &int1, &a[0 + j4lc*lda], &int1, &cs, &sn);
            if (isham) {
                temp = g[j3lc + j4lc*ldg];
                SLC_DROT(&j3l, &g[0 + j3lc*ldg], &int1, &g[0 + j4lc*ldg], &int1, &cs, &sn);
                tau = cs * temp + sn * g[j4lc + j4lc*ldg];
                g[j4lc + j4lc*ldg] = cs * g[j4lc + j4lc*ldg] - sn * temp;
                g[j3lc + j3lc*ldg] = cs * g[j3lc + j3lc*ldg] + sn * tau;
                i32 len = n - j3l;
                SLC_DROT(&len, &g[j3lc + j4lc*ldg], &ldg, &g[j4lc + j4lc*ldg], &ldg, &cs, &sn);
            } else {
                if (n > j3l + 1) {
                    i32 len = n - j3l - 1;
                    SLC_DROT(&len, &g[j3lc + (j3lc+2)*ldg], &ldg, &g[j4lc + (j3lc+2)*ldg], &ldg, &cs, &sn);
                }
                i32 j3m1 = j3l - 1;
                SLC_DROT(&j3m1, &g[0 + j3lc*ldg], &int1, &g[0 + j4lc*ldg], &int1, &cs, &sn);
            }
            if (wantu) {
                SLC_DROT(&n, &u1[0 + j3lc*ldu1], &int1, &u1[0 + j4lc*ldu1], &int1, &cs, &sn);
                SLC_DROT(&n, &u2[0 + j3lc*ldu2], &int1, &u2[0 + j4lc*ldu2], &int1, &cs, &sn);
            }
        }
    }
}
