/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * BB01AD - Benchmark examples for continuous-time algebraic Riccati equations
 *
 * Purpose:
 *   Generate benchmark examples for the numerical solution of continuous-time
 *   algebraic Riccati equations (CAREs) of the form:
 *       0 = Q + A'X + XA - XGX
 *   corresponding to the Hamiltonian matrix H = [A G; Q -A'].
 *
 *   G and Q are symmetric and may be given in factored form:
 *       (I) G = B * R^{-1} * B'
 *       (II) Q = C' * W * C
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <string.h>
#include <ctype.h>

#define ZERO 0.0
#define ONE 1.0
#define TWO 2.0
#define THREE 3.0
#define FOUR 4.0
#define PI 3.141592653589793

#define NEX1 6
#define NEX2 9
#define NEX3 2
#define NEX4 4
#define NMAX 9

static const i32 NEX[4] = {NEX1, NEX2, NEX3, NEX4};

static const i32 NDEF[4][NMAX] = {
    {2, 2, 4, 8, 9, 30, 0, 0, 0},
    {2, 2, 2, 2, 2, 3, 4, 4, 55},
    {20, 64, 0, 0, 0, 0, 0, 0, 0},
    {21, 100, 30, 211, 0, 0, 0, 0, 0}
};

static const i32 MDEF[2][NMAX] = {
    {1, 1, 2, 2, 3, 3, 0, 0, 0},
    {1, 2, 1, 2, 1, 3, 1, 1, 2}
};

static const i32 PDEF[2][NMAX] = {
    {2, 2, 4, 8, 9, 5, 0, 0, 0},
    {1, 1, 2, 2, 2, 3, 2, 1, 10}
};

static const f64 PARDEF[4][NMAX] = {
    {ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 0, 0, 0},
    {1e-5, 1e-7, 1e7, 1e-6, ZERO, 1e7, 1e-5, 1e-5, 1.0},
    {ZERO, ZERO, 0, 0, 0, 0, 0, 0, 0},
    {ONE, 0.01, FOUR, ZERO, 0, 0, 0, 0, 0}
};

static const char* NOTES[4][NMAX] = {
    {"Laub 1979, Ex.1", "Laub 1979, Ex.2: uncontrollable-unobservable data",
     "Beale/Shafai 1989: model of L-1011 aircraft",
     "Bhattacharyya et al. 1983: binary distillation column",
     "Patnaik et al. 1980: tubular ammonia reactor",
     "Davison/Gesing 1978: J-100 jet engine", NULL, NULL, NULL},
    {"Arnold/Laub 1984, Ex.1: (A,B) unstabilizable as EPS -> 0",
     "Arnold/Laub 1984, Ex.3: control weighting matrix singular as EPS -> 0",
     "Kenney/Laub/Wette 1989, Ex.2: ARE ill conditioned for EPS -> oo",
     "Bai/Qian 1994: ill-conditioned Hamiltonian for EPS -> 0",
     "Laub 1992: H-infinity problem, eigenvalues +/- EPS +/- i",
     "Petkov et al. 1987: increasingly badly scaled Hamiltonian as EPS -> oo",
     "Chow/Kokotovic 1976: magnetic tape control system",
     "Arnold/Laub 1984, Ex.2: poor sep. of closed-loop spectrum as EPS -> 0",
     "IFAC Benchmark Problem #90-06: LQG design for modified Boeing B-767 at flutter condition"},
    {"Laub 1979, Ex.4: string of high speed vehicles",
     "Laub 1979, Ex.5: circulant matrices", NULL, NULL, NULL, NULL, NULL, NULL, NULL},
    {"Laub 1979, Ex.6: ill-conditioned Riccati equation",
     "Rosen/Wang 1992: lq control of 1-dimensional heat flow",
     "Hench et al. 1995: coupled springs, dashpots and masses",
     "Lang/Penzl 1994: rotating axle", NULL, NULL, NULL, NULL, NULL}
};

static bool lsame(char a, char b) {
    return toupper((unsigned char)a) == toupper((unsigned char)b);
}

void bb01ad(const char* def, const i32* nr, f64* dpar, i32* ipar,
            const bool* bpar, char* chpar, bool* vec, i32* n, i32* m, i32* p,
            f64* a, const i32 lda, f64* b, const i32 ldb,
            f64* c, const i32 ldc, f64* g, const i32 ldg,
            f64* q, const i32 ldq, f64* x, const i32 ldx,
            f64* dwork, const i32 ldwork, i32* info)
{
    i32 i, j, l, isymm;
    i32 nsymm, msymm, psymm, gdimm, qdimm;
    f64 temp, ttemp, appind, b1, b2, c1, c2;
    char ident[5] = "0000";
    i32 int1 = 1;
    f64 dbl0 = ZERO, dbl1 = ONE, dblm1 = -ONE;

    *info = 0;
    for (i = 0; i < 9; i++) {
        vec[i] = false;
    }

    i32 nr1 = nr[0];
    i32 nr2 = nr[1];

    if ((nr1 != 1) && !(lsame(def[0], 'N') || lsame(def[0], 'D'))) {
        *info = -1;
    } else if ((nr1 < 1) || (nr2 < 1) || (nr1 > 4) || (nr2 > NEX[nr1 - 1])) {
        *info = -2;
    } else if (nr1 > 2) {
        if (!lsame(def[0], 'N')) ipar[0] = NDEF[nr1 - 1][nr2 - 1];
        if (nr1 == 3) {
            if (nr2 == 1) {
                ipar[1] = ipar[0];
                ipar[2] = ipar[0] - 1;
                ipar[0] = 2 * ipar[0] - 1;
            } else if (nr2 == 2) {
                ipar[1] = ipar[0];
                ipar[2] = ipar[0];
            } else {
                ipar[1] = 1;
                ipar[2] = 1;
            }
        } else if (nr1 == 4) {
            if (nr2 == 3) {
                l = ipar[0];
                ipar[1] = 2;
                ipar[2] = 2 * l;
                ipar[0] = 2 * l;
            } else if (nr2 == 4) {
                l = ipar[0];
                ipar[1] = l;
                ipar[2] = l;
                ipar[0] = 2 * l - 1;
            } else {
                ipar[1] = 1;
                ipar[2] = 1;
            }
        }
    } else if ((nr1 == 2) && (nr2 == 9) && (ipar[0] == 2)) {
        ipar[0] = NDEF[nr1 - 1][nr2 - 1];
        ipar[1] = MDEF[nr1 - 1][nr2 - 1];
        ipar[2] = 3;
    } else {
        ipar[0] = NDEF[nr1 - 1][nr2 - 1];
        ipar[1] = MDEF[nr1 - 1][nr2 - 1];
        ipar[2] = PDEF[nr1 - 1][nr2 - 1];
    }

    if (*info != 0) goto cleanup;

    if (ipar[0] < 1) {
        *info = -4;
    } else if (ipar[0] > lda) {
        *info = -12;
    } else if (ipar[0] > ldb) {
        *info = -14;
    } else if (ipar[2] > ldc) {
        *info = -16;
    } else if (bpar[1] && (ipar[0] > ldg)) {
        *info = -18;
    } else if (bpar[4] && (ipar[0] > ldq)) {
        *info = -20;
    } else if (ldx < 1) {
        *info = -22;
    } else if ((nr1 == 1) && ((nr2 == 1) || (nr2 == 2))) {
        if (ipar[0] > ldx) *info = -22;
    } else if ((nr1 == 2) && (nr2 == 1)) {
        if (ipar[0] > ldx) *info = -22;
    } else if ((nr1 == 2) && ((nr2 >= 3) && (nr2 <= 6))) {
        if (ipar[0] > ldx) *info = -22;
    } else if ((nr1 == 3) && (nr2 == 2)) {
        if (ipar[0] > ldx) *info = -22;
    } else if (ldwork < ipar[0] * (ipar[0] > 4 ? ipar[0] : 4)) {
        *info = -24;
    }

cleanup:
    if (*info != 0) {
        SLC_XERBLA("BB01AD", info);
        return;
    }

    nsymm = (ipar[0] * (ipar[0] + 1)) / 2;
    msymm = (ipar[1] * (ipar[1] + 1)) / 2;
    psymm = (ipar[2] * (ipar[2] + 1)) / 2;
    if (!lsame(def[0], 'N')) dpar[0] = PARDEF[nr1 - 1][nr2 - 1];

    SLC_DLASET("A", &ipar[0], &ipar[0], &dbl0, &dbl0, a, &lda);
    SLC_DLASET("A", &ipar[0], &ipar[1], &dbl0, &dbl0, b, &ldb);
    SLC_DLASET("A", &ipar[2], &ipar[0], &dbl0, &dbl0, c, &ldc);
    SLC_DLASET("L", &msymm, &int1, &dbl0, &dbl0, g, &int1);
    SLC_DLASET("L", &psymm, &int1, &dbl0, &dbl0, q, &int1);

    if (nr1 == 1) {
        if (nr2 == 1) {
            a[0 + 1 * lda] = ONE;
            b[1 + 0 * ldb] = ONE;
            q[0] = ONE;
            q[2] = TWO;
            strncpy(ident, "0101", 4);
            SLC_DLASET("A", &ipar[0], &ipar[0], &dbl1, &(f64){TWO}, x, &ldx);
        } else if (nr2 == 2) {
            a[0 + 0 * lda] = FOUR;
            a[1 + 0 * lda] = -4.5;
            a[0 + 1 * lda] = THREE;
            a[1 + 1 * lda] = -3.5;
            SLC_DLASET("A", &ipar[0], &ipar[1], &dblm1, &dbl1, b, &ldb);
            q[0] = 9.0;
            q[1] = 6.0;
            q[2] = FOUR;
            strncpy(ident, "0101", 4);
            temp = ONE + sqrt(TWO);
            SLC_DLASET("A", &ipar[0], &ipar[0], &(f64){6.0 * temp}, &(f64){FOUR * temp}, x, &ldx);
            x[0 + 0 * ldx] = 9.0 * temp;
        } else if ((nr2 >= 3) && (nr2 <= 6)) {
            *info = 1;
            goto end_examples;
        }
    } else if (nr1 == 2) {
        if (nr2 == 1) {
            a[0 + 0 * lda] = ONE;
            a[1 + 1 * lda] = -TWO;
            b[0 + 0 * ldb] = dpar[0];
            SLC_DLASET("U", &ipar[2], &ipar[0], &dbl1, &dbl1, c, &ldc);
            strncpy(ident, "0011", 4);
            if (dpar[0] != ZERO) {
                temp = SLC_DLAPY2(&dbl1, &dpar[0]);
                x[0 + 0 * ldx] = (ONE + temp) / dpar[0] / dpar[0];
                x[1 + 0 * ldx] = ONE / (TWO + temp);
                x[0 + 1 * ldx] = x[1 + 0 * ldx];
                ttemp = dpar[0] * x[0 + 1 * ldx];
                temp = (ONE - ttemp) * (ONE + ttemp);
                x[1 + 1 * ldx] = temp / FOUR;
            } else {
                *info = 2;
            }
        } else if (nr2 == 2) {
            a[0 + 0 * lda] = -0.1;
            a[1 + 1 * lda] = -0.02;
            b[0 + 0 * ldb] = 0.1;
            b[1 + 0 * ldb] = 0.001;
            b[1 + 1 * ldb] = 0.01;
            SLC_DLASET("L", &msymm, &int1, &dbl1, &dbl1, g, &msymm);
            g[0] = g[0] + dpar[0];
            c[0 + 0 * ldc] = 10.0;
            c[0 + 1 * ldc] = 100.0;
            strncpy(ident, "0010", 4);
        } else if (nr2 == 3) {
            a[0 + 1 * lda] = dpar[0];
            b[1 + 0 * ldb] = ONE;
            strncpy(ident, "0111", 4);
            if (dpar[0] != ZERO) {
                temp = sqrt(ONE + TWO * dpar[0]);
                SLC_DLASET("A", &ipar[0], &ipar[0], &dbl1, &temp, x, &ldx);
                x[0 + 0 * ldx] = x[0 + 0 * ldx] / dpar[0];
            } else {
                *info = 2;
            }
        } else if (nr2 == 4) {
            temp = dpar[0] + ONE;
            SLC_DLASET("A", &ipar[0], &ipar[0], &dbl1, &temp, a, &lda);
            q[0] = dpar[0] * dpar[0];
            q[2] = q[0];
            strncpy(ident, "1101", 4);
            x[0 + 0 * ldx] = TWO * temp + sqrt(TWO) * (sqrt(temp * temp + ONE) + dpar[0]);
            x[0 + 0 * ldx] = x[0 + 0 * ldx] / TWO;
            x[1 + 1 * ldx] = x[0 + 0 * ldx];
            ttemp = x[0 + 0 * ldx] - temp;
            if (ttemp != ZERO) {
                x[1 + 0 * ldx] = x[0 + 0 * ldx] / ttemp;
                x[0 + 1 * ldx] = x[1 + 0 * ldx];
            } else {
                *info = 2;
            }
        } else if (nr2 == 5) {
            a[0 + 0 * lda] = THREE - dpar[0];
            a[1 + 0 * lda] = FOUR;
            a[0 + 1 * lda] = ONE;
            a[1 + 1 * lda] = TWO - dpar[0];
            SLC_DLASET("L", &ipar[0], &ipar[1], &dbl1, &dbl1, b, &ldb);
            q[0] = FOUR * dpar[0] - 11.0;
            q[1] = TWO * dpar[0] - 5.0;
            q[2] = TWO * dpar[0] - TWO;
            strncpy(ident, "0101", 4);
            SLC_DLASET("A", &ipar[0], &ipar[0], &dbl1, &dbl1, x, &ldx);
            x[0 + 0 * ldx] = TWO;
        } else if (nr2 == 6) {
            if (dpar[0] != ZERO) {
                a[0 + 0 * lda] = dpar[0];
                a[1 + 1 * lda] = dpar[0] * TWO;
                a[2 + 2 * lda] = dpar[0] * THREE;
                temp = TWO / THREE;
                f64 val = ONE - temp;
                f64 neg_temp = -temp;
                SLC_DLASET("A", &ipar[2], &ipar[0], &neg_temp, &val, c, &ldc);
                SLC_DSYMM("L", "L", &ipar[0], &ipar[0], &dbl1, c, &ldc, a, &lda, &dbl0, dwork, &ipar[0]);
                SLC_DSYMM("R", "L", &ipar[0], &ipar[0], &dbl1, c, &ldc, dwork, &ipar[0], &dbl0, a, &lda);
                g[0] = dpar[0];
                g[3] = dpar[0];
                g[5] = dpar[0];
                q[0] = ONE / dpar[0];
                q[3] = ONE;
                q[5] = dpar[0];
                strncpy(ident, "1000", 4);
                SLC_DLASET("A", &ipar[0], &ipar[0], &dbl0, &dbl0, x, &ldx);
                temp = dpar[0] * dpar[0];
                x[0 + 0 * ldx] = temp + sqrt(temp * temp + ONE);
                x[1 + 1 * ldx] = temp * TWO + sqrt(FOUR * temp * temp + dpar[0]);
                x[2 + 2 * ldx] = temp * THREE + dpar[0] * sqrt(9.0 * temp + ONE);
                SLC_DSYMM("L", "L", &ipar[0], &ipar[0], &dbl1, c, &ldc, x, &ldx, &dbl0, dwork, &ipar[0]);
                SLC_DSYMM("R", "L", &ipar[0], &ipar[0], &dbl1, c, &ldc, dwork, &ipar[0], &dbl0, x, &ldx);
            } else {
                *info = 2;
            }
        } else if (nr2 == 7) {
            if (dpar[0] != ZERO) {
                a[0 + 1 * lda] = 0.400;
                a[1 + 2 * lda] = 0.345;
                a[2 + 1 * lda] = -0.524 / dpar[0];
                a[2 + 2 * lda] = -0.465 / dpar[0];
                a[2 + 3 * lda] = 0.262 / dpar[0];
                a[3 + 3 * lda] = -ONE / dpar[0];
                b[3 + 0 * ldb] = ONE / dpar[0];
                c[0 + 0 * ldc] = ONE;
                c[1 + 2 * ldc] = ONE;
                strncpy(ident, "0011", 4);
            } else {
                *info = 2;
            }
        } else if (nr2 == 8) {
            a[0 + 0 * lda] = -dpar[0];
            a[1 + 0 * lda] = -ONE;
            a[0 + 1 * lda] = ONE;
            a[1 + 1 * lda] = -dpar[0];
            a[2 + 2 * lda] = dpar[0];
            a[3 + 2 * lda] = -ONE;
            a[2 + 3 * lda] = ONE;
            a[3 + 3 * lda] = dpar[0];
            SLC_DLASET("L", &ipar[0], &ipar[1], &dbl1, &dbl1, b, &ldb);
            SLC_DLASET("U", &ipar[2], &ipar[0], &dbl1, &dbl1, c, &ldc);
            strncpy(ident, "0011", 4);
        } else if (nr2 == 9) {
            *info = 1;
            goto end_examples;
        }
    } else if (nr1 == 3) {
        if (nr2 == 1) {
            for (i = 0; i < ipar[0]; i++) {
                if ((i + 1) % 2 == 1) {
                    a[i + i * lda] = -ONE;
                    b[i + (i / 2) * ldb] = ONE;
                } else {
                    a[i + (i - 1) * lda] = ONE;
                    a[i + (i + 1) * lda] = -ONE;
                    c[(i / 2) + i * ldc] = ONE;
                }
            }
            isymm = 0;
            for (i = ipar[2]; i >= 1; i--) {
                q[isymm] = 10.0;
                isymm = isymm + i;
            }
            strncpy(ident, "0001", 4);
        } else if (nr2 == 2) {
            for (i = 0; i < ipar[0]; i++) {
                a[i + i * lda] = -TWO;
                if (i < ipar[0] - 1) {
                    a[i + (i + 1) * lda] = ONE;
                    a[(i + 1) + i * lda] = ONE;
                }
            }
            a[0 + (ipar[0] - 1) * lda] = ONE;
            a[(ipar[0] - 1) + 0 * lda] = ONE;
            strncpy(ident, "1111", 4);
            temp = TWO * PI / (f64)ipar[0];
            for (i = 0; i < ipar[0]; i++) {
                dwork[i] = cos(temp * (f64)i);
                dwork[ipar[0] + i] = -TWO + TWO * dwork[i] +
                    sqrt(5.0 + FOUR * dwork[i] * (dwork[i] - TWO));
            }
            for (j = 0; j < ipar[0]; j++) {
                for (i = 0; i < ipar[0]; i++) {
                    dwork[2 * ipar[0] + i] = cos(temp * (f64)i * (f64)j);
                }
                x[j + 0 * ldx] = SLC_DDOT(&ipar[0], &dwork[ipar[0]], &int1,
                                          &dwork[2 * ipar[0]], &int1) / (f64)ipar[0];
            }
            for (i = 1; i < ipar[0]; i++) {
                i32 len1 = ipar[0] - i;
                i32 len2 = i;
                SLC_DCOPY(&len1, &x[0 + 0 * ldx], &int1, &x[i + i * ldx], &int1);
                SLC_DCOPY(&len2, &x[(ipar[0] - i) + 0 * ldx], &int1, &x[0 + i * ldx], &int1);
            }
        }
    } else if (nr1 == 4) {
        if (nr2 == 1) {
            if (!lsame(def[0], 'N')) {
                dpar[0] = ONE;
                dpar[1] = ONE;
            }
            i32 nm1 = ipar[0] - 1;
            SLC_DLASET("A", &nm1, &nm1, &dbl0, &dbl1, &a[0 + 1 * lda], &lda);
            b[(ipar[0] - 1) + 0 * ldb] = ONE;
            c[0 + 0 * ldc] = ONE;
            q[0] = dpar[0];
            g[0] = dpar[1];
            strncpy(ident, "0000", 4);
        } else if (nr2 == 2) {
            appind = (f64)(ipar[0] + 1);
            if (!lsame(def[0], 'N')) {
                dpar[0] = PARDEF[nr1 - 1][nr2 - 1];
                dpar[1] = ONE;
                dpar[2] = ONE;
                dpar[3] = 0.2;
                dpar[4] = 0.3;
                dpar[5] = 0.2;
                dpar[6] = 0.3;
            }
            temp = -dpar[0] * appind;
            f64 diag_val = TWO * temp;
            SLC_DLASET("A", &ipar[0], &ipar[0], &dbl0, &diag_val, a, &lda);
            for (i = 0; i < ipar[0] - 1; i++) {
                a[(i + 1) + i * lda] = -temp;
                a[i + (i + 1) * lda] = -temp;
            }
            temp = ONE / (6.0 * appind);
            f64 diag_d = FOUR * temp;
            i32 nm1 = ipar[0] - 1;
            SLC_DLASET("L", &ipar[0], &int1, &diag_d, &diag_d, dwork, &ipar[0]);
            SLC_DLASET("L", &nm1, &int1, &temp, &temp, &dwork[ipar[0]], &ipar[0]);
            SLC_DPTTRF(&ipar[0], dwork, &dwork[ipar[0]], info);
            SLC_DPTTRS(&ipar[0], &ipar[0], dwork, &dwork[ipar[0]], a, &lda, info);
            for (i = 0; i < ipar[0]; i++) {
                b1 = fmax((f64)i / appind, dpar[3]);
                b2 = fmin((f64)(i + 2) / appind, dpar[4]);
                c1 = fmax((f64)i / appind, dpar[5]);
                c2 = fmin((f64)(i + 2) / appind, dpar[6]);
                if (b1 >= b2) {
                    b[i + 0 * ldb] = ZERO;
                } else {
                    b[i + 0 * ldb] = b2 - b1;
                    temp = fmin(b2, (f64)(i + 1) / appind);
                    if (b1 < temp) {
                        b[i + 0 * ldb] += appind * (temp * temp - b1 * b1) / TWO;
                        b[i + 0 * ldb] += (f64)(i + 1) * (b1 - temp);
                    }
                    temp = fmax(b1, (f64)(i + 1) / appind);
                    if (temp < b2) {
                        b[i + 0 * ldb] -= appind * (b2 * b2 - temp * temp) / TWO;
                        b[i + 0 * ldb] -= (f64)(i + 1) * (temp - b2);
                    }
                }
                if (c1 >= c2) {
                    c[0 + i * ldc] = ZERO;
                } else {
                    c[0 + i * ldc] = c2 - c1;
                    temp = fmin(c2, (f64)(i + 1) / appind);
                    if (c1 < temp) {
                        c[0 + i * ldc] += appind * (temp * temp - c1 * c1) / TWO;
                        c[0 + i * ldc] += (f64)(i + 1) * (c1 - temp);
                    }
                    temp = fmax(c1, (f64)(i + 1) / appind);
                    if (temp < c2) {
                        c[0 + i * ldc] -= appind * (c2 * c2 - temp * temp) / TWO;
                        c[0 + i * ldc] -= (f64)(i + 1) * (temp - c2);
                    }
                }
            }
            SLC_DSCAL(&ipar[0], &dpar[1], &b[0 + 0 * ldb], &int1);
            SLC_DSCAL(&ipar[0], &dpar[2], &c[0 + 0 * ldc], &ldc);
            SLC_DPTTRS(&ipar[0], &int1, dwork, &dwork[ipar[0]], b, &ldb, info);
            strncpy(ident, "0011", 4);
        } else if (nr2 == 3) {
            if (!lsame(def[0], 'N')) {
                dpar[0] = PARDEF[nr1 - 1][nr2 - 1];
                dpar[1] = FOUR;
                dpar[2] = ONE;
            }
            if (dpar[0] != ZERO) {
                l = ipar[0] / 2;
                SLC_DLASET("A", &l, &l, &dbl0, &dbl1, &a[0 + l * lda], &lda);
                temp = dpar[2] / dpar[0];
                a[l + 0 * lda] = -temp;
                a[l + 1 * lda] = temp;
                a[(ipar[0] - 1) + (l - 2) * lda] = temp;
                a[(ipar[0] - 1) + (l - 1) * lda] = -temp;
                ttemp = TWO * temp;
                for (i = 1; i < l - 1; i++) {
                    a[(l + i) + i * lda] = -ttemp;
                    a[(l + i) + (i + 1) * lda] = temp;
                    a[(l + i) + (i - 1) * lda] = temp;
                }
                f64 damping = -dpar[1] / dpar[0];
                SLC_DLASET("A", &l, &l, &dbl0, &damping, &a[l + l * lda], &lda);
                b[l + 0 * ldb] = ONE / dpar[0];
                b[(ipar[0] - 1) + 1 * ldb] = -ONE / dpar[0];
                strncpy(ident, "0111", 4);
            } else {
                *info = 2;
            }
        } else if (nr2 == 4) {
            *info = 1;
            goto end_examples;
        }
    }

end_examples:
    if (*info != 0) return;

    if (bpar[0]) {
        gdimm = ipar[0];
        if (ident[3] == '0') {
            SLC_DPPTRF("L", &ipar[1], g, info);
            if (*info == 0) {
                SLC_DPPTRI("L", &ipar[1], g, info);
                if (ident[0] == '0') {
                    for (i = 0; i < ipar[0]; i++) {
                        SLC_DSPMV("L", &ipar[1], &dbl1, g, &b[i + 0 * ldb], &ldb,
                                  &dbl0, &dwork[i * ipar[0]], &int1);
                    }
                    SLC_DGEMV("T", &ipar[1], &ipar[0], &dbl1, dwork, &ipar[0],
                              &b[0 + 0 * ldb], &ldb, &dbl0, g, &int1);
                    isymm = ipar[0];
                    for (i = 1; i < ipar[0]; i++) {
                        SLC_DGEMV("T", &ipar[1], &ipar[0], &dbl1, dwork, &ipar[0],
                                  &b[i + 0 * ldb], &ldb, &dbl0, &b[0 + 0 * ldb], &ldb);
                        i32 len = ipar[0] - i;
                        SLC_DCOPY(&len, &b[i + 0 * ldb], &ldb, &g[isymm], &int1);
                        isymm += (ipar[0] - i);
                    }
                }
            } else {
                if (*info > 0) {
                    *info = 3;
                    return;
                }
            }
        } else {
            if (ident[0] == '0') {
                if (ipar[1] == 1) {
                    SLC_DLASET("L", &nsymm, &int1, &dbl0, &dbl0, g, &int1);
                    SLC_DSPR("L", &ipar[0], &dbl1, b, &int1, g);
                } else {
                    SLC_DSYRK("L", "N", &ipar[0], &ipar[1], &dbl1,
                              b, &ldb, &dbl0, dwork, &ipar[0]);
                    ma02dd("P", "L", ipar[0], dwork, ipar[0], g);
                }
            } else {
                isymm = 0;
                for (i = ipar[0]; i >= 1; i--) {
                    g[isymm] = ONE;
                    isymm += i;
                }
            }
        }
    } else {
        gdimm = ipar[1];
        if (ident[0] == '1')
            SLC_DLASET("A", &ipar[0], &ipar[1], &dbl0, &dbl1, b, &ldb);
        if (ident[3] == '1') {
            isymm = 0;
            for (i = ipar[1]; i >= 1; i--) {
                g[isymm] = ONE;
                isymm += i;
            }
        }
    }

    if (bpar[3]) {
        qdimm = ipar[0];
        if (ident[2] == '0') {
            if (ident[1] == '0') {
                for (i = 0; i < ipar[0]; i++) {
                    SLC_DSPMV("L", &ipar[2], &dbl1, q, &c[0 + i * ldc], &int1,
                              &dbl0, &dwork[i * ipar[0]], &int1);
                }
                isymm = ipar[0];
                for (i = 1; i < ipar[0]; i++) {
                    SLC_DGEMV("T", &ipar[2], &ipar[0], &dbl1, dwork, &ipar[0],
                              &c[0 + i * ldc], &int1, &dbl0, q, &int1);
                    i32 len = ipar[0] - i;
                    SLC_DCOPY(&len, &q[i], &int1, &q[isymm], &int1);
                    isymm += (ipar[0] - i);
                }
                SLC_DGEMV("T", &ipar[2], &ipar[0], &dbl1, dwork, &ipar[0],
                          &c[0 + 0 * ldc], &int1, &dbl0, q, &int1);
            }
        } else {
            if (ident[1] == '0') {
                if (ipar[2] == 1) {
                    SLC_DLASET("L", &nsymm, &int1, &dbl0, &dbl0, q, &int1);
                    SLC_DSPR("L", &ipar[0], &dbl1, c, &ldc, q);
                } else {
                    SLC_DSYRK("L", "T", &ipar[0], &ipar[2], &dbl1, c, &ldc,
                              &dbl0, dwork, &ipar[0]);
                    ma02dd("P", "L", ipar[0], dwork, ipar[0], q);
                }
            } else {
                isymm = 0;
                for (i = ipar[0]; i >= 1; i--) {
                    q[isymm] = ONE;
                    isymm += i;
                }
            }
        }
    } else {
        qdimm = ipar[2];
        if (ident[1] == '1')
            SLC_DLASET("A", &ipar[2], &ipar[0], &dbl0, &dbl1, c, &ldc);
        if (ident[2] == '1') {
            isymm = 0;
            for (i = ipar[2]; i >= 1; i--) {
                q[isymm] = ONE;
                isymm += i;
            }
        }
    }

    if (bpar[1]) {
        isymm = (gdimm * (gdimm + 1)) / 2;
        SLC_DCOPY(&isymm, g, &int1, dwork, &int1);
        ma02dd("U", "L", gdimm, g, ldg, dwork);
        ma02ed('L', gdimm, g, ldg);
    } else if (bpar[2]) {
        ma02dd("U", "L", gdimm, dwork, gdimm, g);
        ma02ed('L', gdimm, dwork, gdimm);
        ma02dd("P", "U", gdimm, dwork, gdimm, g);
    }
    if (bpar[4]) {
        isymm = (qdimm * (qdimm + 1)) / 2;
        SLC_DCOPY(&isymm, q, &int1, dwork, &int1);
        ma02dd("U", "L", qdimm, q, ldq, dwork);
        ma02ed('L', qdimm, q, ldq);
    } else if (bpar[5]) {
        ma02dd("U", "L", qdimm, dwork, qdimm, q);
        ma02ed('L', qdimm, dwork, qdimm);
        ma02dd("P", "U", qdimm, dwork, qdimm, q);
    }

    vec[0] = true;
    vec[1] = true;
    vec[2] = true;
    vec[3] = true;
    vec[4] = !bpar[0];
    vec[5] = !bpar[3];
    vec[6] = true;
    vec[7] = true;
    if (nr1 == 1) {
        if ((nr2 == 1) || (nr2 == 2)) vec[8] = true;
    } else if (nr1 == 2) {
        if ((nr2 == 1) || ((nr2 >= 3) && (nr2 <= 6))) vec[8] = true;
    } else if (nr1 == 3) {
        if (nr2 == 2) vec[8] = true;
    }
    if (NOTES[nr1 - 1][nr2 - 1] != NULL) {
        strncpy(chpar, NOTES[nr1 - 1][nr2 - 1], 255);
    }
    *n = ipar[0];
    *m = ipar[1];
    *p = ipar[2];
}
