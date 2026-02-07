/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * BB02AD - Benchmark examples for discrete-time algebraic Riccati equations
 *
 * Purpose:
 *   Generate benchmark examples for the numerical solution of discrete-time
 *   algebraic Riccati equations (DAREs) of the form:
 *       0 = A^T X A - X - (A^T X B + S)(R + B^T X B)^{-1}(B^T X A + S^T) + Q
 *   as presented in the DAREX collection.
 *
 *   Q and R are symmetric. Q may be given in factored form:
 *       (I) Q = C^T Q0 C
 *   and if R is nonsingular and S = 0:
 *       (II) G = B R^{-1} B^T
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
#define FIVE 5.0

#define NEX1 13
#define NEX2 5
#define NEX3 0
#define NEX4 1
#define NMAX 13

static const i32 NEX[4] = {NEX1, NEX2, NEX3, NEX4};

static const i32 NDEF[4][NMAX] = {
    {2, 2, 2, 3, 4, 4, 4, 5, 6, 9, 11, 13, 26},
    {2, 2, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
};

static const i32 MDEF[2][NMAX] = {
    {1, 2, 1, 2, 2, 2, 4, 2, 2, 3, 2, 2, 6},
    {1, 2, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0}
};

static const i32 PDEF[2][NMAX] = {
    {1, 2, 2, 3, 4, 4, 4, 5, 2, 2, 4, 4, 12},
    {2, 2, 2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0}
};

static const char* NOTES[4][NMAX] = {
    {"Van Dooren 1981, Ex. II: singular R matrix",
     "Ionescu/Weiss 1992: singular R matrix, nonzero S matrix",
     "Jonckheere 1981: (A,B) controllable, no solution X <= 0",
     "Sun 1998: R singular, Q non-definite",
     "Ackerson/Fu 1970 : satellite control problem",
     "Litkouhi 1983 : system with slow and fast modes",
     "Lu/Lin 1993, Ex. 4.3",
     "Gajic/Shen 1993, Section 2.7.4: chemical plant",
     "Davison/Wang 1974: nonzero S matrix",
     "Patnaik et al. 1980: tubular ammonia reactor",
     "Sima 1996, Sec. 1.2.2: paper machine model error integrators",
     "Sima 1996, Ex. 2.6: paper machine model with with disturbances",
     "Power plant model, Katayama et al., 1985"},
    {"Laub 1979, Ex. 2: uncontrollable-unobservable data",
     "Laub 1979, Ex. 3: increasingly ill-conditioned R-matrix",
     "increasingly bad scaled system as eps -> oo",
     "Petkov et. al. 1989 : increasingly bad scaling as eps -> oo",
     "Pappas et al. 1980: process control of paper machine",
     NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
    {"Pappas et al. 1980, Ex. 3", NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}
};

static bool lsame_local(char a, char b) {
    return toupper((unsigned char)a) == toupper((unsigned char)b);
}

void bb02ad(const char* def, const i32* nr, f64* dpar, i32* ipar,
            const bool* bpar, char* chpar, bool* vec, i32* n, i32* m, i32* p,
            f64* a, const i32 lda, f64* b, const i32 ldb,
            f64* c, const i32 ldc, f64* q, const i32 ldq,
            f64* r, const i32 ldr, f64* s, const i32 lds,
            f64* x, const i32 ldx, f64* dwork, const i32 ldwork, i32* info)
{
    i32 i, isymm;
    i32 nsymm, msymm, psymm, qdimm, rdimm;
    f64 temp, alpha, beta;
    char ident[5] = "0000";
    i32 int1 = 1;
    f64 dbl0 = ZERO, dbl1 = ONE, dblm1 = -ONE;

    *info = 0;
    for (i = 0; i < 10; i++) {
        vec[i] = false;
    }

    i32 nr1 = nr[0];
    i32 nr2 = nr[1];

    if ((nr1 >= 2) && !(lsame_local(def[0], 'D') || lsame_local(def[0], 'N'))) {
        *info = -1;
    } else if ((nr1 < 1) || (nr1 > 4) || (nr2 < 0) || (nr2 > NEX[nr1 - 1])) {
        *info = -2;
    }

    if (*info != 0) {
        SLC_XERBLA("BB02AD", info);
        return;
    }

    if (nr1 >= 3) {
        if (lsame_local(def[0], 'D')) ipar[0] = NDEF[nr1 - 1][nr2 - 1];
        ipar[1] = 1;
        ipar[2] = ipar[0];
    } else {
        ipar[0] = NDEF[nr1 - 1][nr2 - 1];
        ipar[1] = MDEF[nr1 - 1][nr2 - 1];
        ipar[2] = PDEF[nr1 - 1][nr2 - 1];
    }

    if (ipar[0] < 1) {
        *info = -4;
    } else if (ipar[0] > lda) {
        *info = -12;
    } else if (ipar[0] > ldb) {
        *info = -14;
    } else if (ipar[2] > ldc) {
        *info = -16;
    } else if (bpar[1] && (((!bpar[0]) && (ipar[2] > ldq)) || (bpar[0] && (ipar[0] > ldq)))) {
        *info = -18;
    } else if (bpar[4] && ((bpar[3] && (ipar[0] > ldr)) || ((!bpar[3]) && (ipar[1] > ldr)))) {
        *info = -20;
    } else if (lds < 1 || (bpar[6] && (ipar[0] > lds))) {
        *info = -22;
    } else if (ldx < 1) {
        *info = -24;
    } else if (((nr1 == 1) && ((nr2 == 1) || (nr2 == 3) || (nr2 == 4))) ||
               ((nr1 == 2) && ((nr2 == 1) || (nr2 >= 3))) ||
               (nr1 == 4)) {
        if (ipar[0] > ldx) {
            *info = -24;
        } else {
            SLC_DLASET("A", &ipar[0], &ipar[0], &dbl0, &dbl0, x, &ldx);
        }
    } else if (ldwork < ipar[0] * ipar[0]) {
        *info = -26;
    }

    if (*info != 0) {
        SLC_XERBLA("BB02AD", info);
        return;
    }

    nsymm = (ipar[0] * (ipar[0] + 1)) / 2;
    msymm = (ipar[1] * (ipar[1] + 1)) / 2;
    psymm = (ipar[2] * (ipar[2] + 1)) / 2;

    SLC_DLASET("A", &ipar[0], &ipar[0], &dbl0, &dbl0, a, &lda);
    SLC_DLASET("A", &ipar[0], &ipar[1], &dbl0, &dbl0, b, &ldb);
    SLC_DLASET("A", &ipar[2], &ipar[0], &dbl0, &dbl0, c, &ldc);
    SLC_DLASET("L", &psymm, &int1, &dbl0, &dbl0, q, &int1);
    SLC_DLASET("L", &msymm, &int1, &dbl0, &dbl0, r, &int1);
    if (bpar[6]) {
        SLC_DLASET("A", &ipar[0], &ipar[1], &dbl0, &dbl0, s, &lds);
    }

    if (nr1 == 1) {
        if (nr2 == 1) {
            a[0 + 0 * lda] = TWO;
            a[1 + 0 * lda] = ONE;
            a[0 + 1 * lda] = -ONE;
            b[0 + 0 * ldb] = ONE;
            q[0] = ONE;
            c[0 + 1 * ldc] = ONE;
            r[0] = ZERO;
            SLC_DLASET("A", &ipar[0], &ipar[0], &dbl0, &dbl1, x, &ldx);
            strncpy(ident, "0000", 4);
        } else if (nr2 == 2) {
            a[0 + 1 * lda] = ONE;
            a[1 + 1 * lda] = -ONE;
            b[0 + 0 * ldb] = ONE;
            b[1 + 0 * ldb] = TWO;
            b[1 + 1 * ldb] = ONE;
            r[0] = 9.0;
            r[1] = THREE;
            r[2] = ONE;
            f64 val = -FOUR;
            SLC_DLASET("A", &psymm, &int1, &val, &val, q, &psymm);
            q[2] = 7.0;
            f64 scale = 11.0;
            SLC_DRSCL(&msymm, &scale, q, &int1);
            if (bpar[6]) {
                s[0 + 0 * lds] = THREE;
                s[1 + 0 * lds] = -ONE;
                s[0 + 1 * lds] = ONE;
                s[1 + 1 * lds] = 7.0;
            }
            strncpy(ident, "0100", 4);
        } else if (nr2 == 3) {
            a[0 + 1 * lda] = ONE;
            b[1 + 0 * ldb] = ONE;
            q[0] = ONE;
            q[1] = TWO;
            q[2] = FOUR;
            x[0 + 0 * ldx] = ONE;
            x[1 + 0 * ldx] = TWO;
            x[0 + 1 * ldx] = TWO;
            x[1 + 1 * ldx] = TWO + sqrt(FIVE);
            strncpy(ident, "0101", 4);
        } else if (nr2 == 4) {
            a[0 + 1 * lda] = 0.1;
            a[1 + 2 * lda] = 0.01;
            b[0 + 0 * ldb] = ONE;
            b[2 + 1 * ldb] = ONE;
            r[2] = ONE;
            q[0] = 1e5;
            q[3] = 1e3;
            q[5] = -10.0;
            x[0 + 0 * ldx] = 1e5;
            x[1 + 1 * ldx] = 1e3;
            strncpy(ident, "0100", 4);
        } else if (nr2 == 9) {
            a[0 + 1 * lda] = ONE;
            a[1 + 2 * lda] = ONE;
            a[3 + 4 * lda] = ONE;
            a[4 + 5 * lda] = ONE;
            b[2 + 0 * ldb] = ONE;
            b[5 + 1 * ldb] = ONE;
            c[0 + 0 * ldc] = ONE;
            c[0 + 1 * ldc] = ONE;
            c[1 + 3 * ldc] = ONE;
            c[1 + 4 * ldc] = -ONE;
            r[0] = THREE;
            r[2] = ONE;
            if (bpar[6]) {
                s[0 + 0 * lds] = ONE;
                s[1 + 0 * lds] = ONE;
                s[3 + 0 * lds] = ONE;
                s[4 + 0 * lds] = -ONE;
            }
            strncpy(ident, "0010", 4);
        } else if (((nr2 >= 5) && (nr2 <= 8)) || (nr2 == 10) || (nr2 == 11) || (nr2 == 12) || (nr2 == 13)) {
            *info = 1;
            return;
        }
    } else if (nr1 == 2) {
        if (nr2 == 1) {
            if (lsame_local(def[0], 'D')) dpar[0] = 1e7;
            a[0 + 0 * lda] = FOUR;
            a[1 + 0 * lda] = -4.5;
            a[0 + 1 * lda] = THREE;
            a[1 + 1 * lda] = -3.5;
            SLC_DLASET("A", &ipar[0], &ipar[1], &dblm1, &dbl1, b, &ldb);
            r[0] = dpar[0];
            q[0] = 9.0;
            q[1] = 6.0;
            q[2] = FOUR;
            temp = (ONE + sqrt(ONE + FOUR * dpar[0])) / TWO;
            x[0 + 0 * ldx] = temp * q[0];
            x[1 + 0 * ldx] = temp * q[1];
            x[0 + 1 * ldx] = x[1 + 0 * ldx];
            x[1 + 1 * ldx] = temp * q[2];
            strncpy(ident, "0100", 4);
        } else if (nr2 == 2) {
            if (lsame_local(def[0], 'D')) dpar[0] = 1e7;
            if (dpar[0] == ZERO) {
                *info = 2;
                return;
            }
            a[0 + 0 * lda] = 0.9512;
            a[1 + 1 * lda] = 0.9048;
            f64 val = 4.877;
            SLC_DLASET("A", &int1, &ipar[1], &val, &val, b, &ldb);
            b[1 + 0 * ldb] = -1.1895;
            b[1 + 1 * ldb] = 3.569;
            r[0] = ONE / (THREE * dpar[0]);
            r[2] = THREE * dpar[0];
            q[0] = 0.005;
            q[2] = 0.02;
            strncpy(ident, "0100", 4);
        } else if (nr2 == 3) {
            if (lsame_local(def[0], 'D')) dpar[0] = 1e7;
            a[0 + 1 * lda] = dpar[0];
            b[1 + 0 * ldb] = ONE;
            x[0 + 0 * ldx] = ONE;
            x[1 + 1 * ldx] = ONE + dpar[0] * dpar[0];
            strncpy(ident, "0111", 4);
        } else if (nr2 == 4) {
            if (lsame_local(def[0], 'D')) dpar[0] = 1e7;
            a[1 + 1 * lda] = ONE;
            a[2 + 2 * lda] = THREE;
            r[0] = dpar[0];
            r[3] = dpar[0];
            r[5] = dpar[0];
            temp = TWO / THREE;
            f64 val = ONE - temp;
            f64 neg_temp = -temp;
            SLC_DLASET("A", &ipar[2], &ipar[0], &neg_temp, &val, c, &ldc);
            SLC_DSYMM("L", "L", &ipar[0], &ipar[0], &dbl1, c, &ldc, a, &lda, &dbl0, dwork, &ipar[0]);
            SLC_DSYMM("R", "L", &ipar[0], &ipar[0], &dbl1, c, &ldc, dwork, &ipar[0], &dbl0, a, &lda);
            q[0] = dpar[0];
            q[3] = dpar[0];
            q[5] = dpar[0];
            x[0 + 0 * ldx] = dpar[0];
            x[1 + 1 * ldx] = dpar[0] * (ONE + sqrt(FIVE)) / TWO;
            x[2 + 2 * ldx] = dpar[0] * (9.0 + sqrt(85.0)) / TWO;
            SLC_DSYMM("L", "L", &ipar[0], &ipar[0], &dbl1, c, &ldc, x, &ldx, &dbl0, dwork, &ipar[0]);
            SLC_DSYMM("R", "L", &ipar[0], &ipar[0], &dbl1, c, &ldc, dwork, &ipar[0], &dbl0, x, &ldx);
            strncpy(ident, "1000", 4);
        } else if (nr2 == 5) {
            if (lsame_local(def[0], 'D')) {
                dpar[3] = 0.25;
                dpar[2] = ONE;
                dpar[1] = ONE;
                dpar[0] = 1e9;
            }
            if (dpar[0] == ZERO) {
                *info = 2;
                return;
            }
            temp = dpar[1] / dpar[0];
            beta = dpar[2] * temp;
            alpha = ONE - temp;
            a[0 + 0 * lda] = alpha;
            i32 nm1 = ipar[0] - 1;
            SLC_DLASET("A", &nm1, &nm1, &dbl0, &dbl1, &a[1 + 0 * lda], &lda);
            b[0 + 0 * ldb] = beta;
            c[0 + 3 * ldc] = ONE;
            r[0] = dpar[3];
            if (beta == ZERO) {
                *info = 2;
                return;
            }
            SLC_DLASET("A", &ipar[0], &ipar[0], &dbl0, &dbl1, x, &ldx);
            f64 beta2 = beta * beta;
            temp = dpar[3] * (alpha + ONE) * (alpha - ONE) + beta2;
            x[0 + 0 * ldx] = (temp + sqrt(temp * temp + FOUR * beta2 * dpar[3])) / TWO / beta2;
            strncpy(ident, "0010", 4);
        }
    } else if (nr1 == 4) {
        if (nr2 == 1) {
            if (lsame_local(def[0], 'D')) dpar[0] = ONE;
            i32 nm1 = ipar[0] - 1;
            SLC_DLASET("A", &nm1, &nm1, &dbl0, &dbl1, &a[0 + 1 * lda], &lda);
            b[(ipar[0] - 1) + 0 * ldb] = ONE;
            r[0] = dpar[0];
            for (i = 0; i < ipar[0]; i++) {
                x[i + i * ldx] = (f64)(i + 1);
            }
            strncpy(ident, "0110", 4);
        }
    }

    if (*info != 0) return;

    if (bpar[3]) {
        rdimm = ipar[0];
        if (ident[3] == '0') {
            SLC_DPPTRF("L", &ipar[1], r, info);
            if (*info == 0) {
                SLC_DPPTRI("L", &ipar[1], r, info);
                if (ident[0] == '0') {
                    for (i = 0; i < ipar[0]; i++) {
                        SLC_DSPMV("L", &ipar[1], &dbl1, r, &b[i + 0 * ldb], &ldb, &dbl0, &dwork[i * ipar[0]], &int1);
                    }
                    SLC_DGEMV("T", &ipar[1], &ipar[0], &dbl1, dwork, &ipar[0], &b[0 + 0 * ldb], &ldb, &dbl0, r, &int1);
                    isymm = ipar[0];
                    for (i = 1; i < ipar[0]; i++) {
                        SLC_DGEMV("T", &ipar[1], &ipar[0], &dbl1, dwork, &ipar[0], &b[i + 0 * ldb], &ldb, &dbl0, &b[0 + 0 * ldb], &ldb);
                        i32 len = ipar[0] - i;
                        SLC_DCOPY(&len, &b[i + 0 * ldb], &ldb, &r[isymm], &int1);
                        isymm += len;
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
                    SLC_DLASET("L", &nsymm, &int1, &dbl0, &dbl0, r, &int1);
                    SLC_DSPR("L", &ipar[0], &dbl1, b, &int1, r);
                } else {
                    SLC_DSYRK("L", "N", &ipar[0], &ipar[1], &dbl1, b, &ldb, &dbl0, dwork, &ipar[0]);
                    ma02dd("P", "L", ipar[0], dwork, ipar[0], r);
                }
            } else {
                isymm = 0;
                for (i = ipar[0]; i >= 1; i--) {
                    r[isymm] = ONE;
                    isymm += i;
                }
            }
        }
    } else {
        rdimm = ipar[1];
        if (ident[0] == '1') {
            SLC_DLASET("A", &ipar[0], &ipar[1], &dbl0, &dbl1, b, &ldb);
        }
        if (ident[3] == '1') {
            isymm = 0;
            for (i = ipar[1]; i >= 1; i--) {
                r[isymm] = ONE;
                isymm += i;
            }
        }
    }

    if (bpar[0]) {
        qdimm = ipar[0];
        if (ident[2] == '0') {
            if (ident[1] == '0') {
                for (i = 0; i < ipar[0]; i++) {
                    SLC_DSPMV("L", &ipar[2], &dbl1, q, &c[0 + i * ldc], &int1, &dbl0, &dwork[i * ipar[0]], &int1);
                }
                isymm = ipar[0];
                for (i = 1; i < ipar[0]; i++) {
                    SLC_DGEMV("T", &ipar[2], &ipar[0], &dbl1, dwork, &ipar[0], &c[0 + i * ldc], &int1, &dbl0, q, &int1);
                    i32 len = ipar[0] - i;
                    SLC_DCOPY(&len, &q[i], &int1, &q[isymm], &int1);
                    isymm += len;
                }
                SLC_DGEMV("T", &ipar[2], &ipar[0], &dbl1, dwork, &ipar[0], &c[0 + 0 * ldc], &int1, &dbl0, q, &int1);
            }
        } else {
            if (ident[1] == '0') {
                if (ipar[2] == 1) {
                    SLC_DLASET("L", &nsymm, &int1, &dbl0, &dbl0, q, &int1);
                    SLC_DSPR("L", &ipar[0], &dbl1, c, &ldc, q);
                } else {
                    SLC_DSYRK("L", "T", &ipar[0], &ipar[2], &dbl1, c, &ldc, &dbl0, dwork, &ipar[0]);
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
        if (ident[1] == '1') {
            SLC_DLASET("A", &ipar[2], &ipar[0], &dbl0, &dbl1, c, &ldc);
        }
        if (ident[2] == '1') {
            isymm = 0;
            for (i = ipar[2]; i >= 1; i--) {
                q[isymm] = ONE;
                isymm += i;
            }
        }
    }

    if (bpar[1]) {
        isymm = (qdimm * (qdimm + 1)) / 2;
        SLC_DCOPY(&isymm, q, &int1, dwork, &int1);
        ma02dd("U", "L", qdimm, q, ldq, dwork);
        ma02ed('L', qdimm, q, ldq);
    } else if (bpar[2]) {
        ma02dd("U", "L", qdimm, dwork, qdimm, q);
        ma02ed('L', qdimm, dwork, qdimm);
        ma02dd("P", "U", qdimm, dwork, qdimm, q);
    }

    if (bpar[4]) {
        isymm = (rdimm * (rdimm + 1)) / 2;
        SLC_DCOPY(&isymm, r, &int1, dwork, &int1);
        ma02dd("U", "L", rdimm, r, ldr, dwork);
        ma02ed('L', rdimm, r, ldr);
    } else if (bpar[5]) {
        ma02dd("U", "L", rdimm, dwork, rdimm, r);
        ma02ed('L', rdimm, dwork, rdimm);
        ma02dd("P", "U", rdimm, dwork, rdimm, r);
    }

    vec[0] = true;
    vec[1] = true;
    vec[2] = true;
    vec[3] = true;
    vec[4] = !bpar[3];
    vec[5] = !bpar[0];
    vec[6] = true;
    vec[7] = true;
    vec[8] = bpar[6];
    if (((nr1 == 1) && ((nr2 == 1) || (nr2 == 3) || (nr2 == 4))) ||
        ((nr1 == 2) && ((nr2 == 1) || (nr2 >= 3))) ||
        (nr1 == 4)) {
        vec[9] = true;
    }

    if (NOTES[nr1 - 1][nr2 - 1] != NULL) {
        strncpy(chpar, NOTES[nr1 - 1][nr2 - 1], 255);
    }
    *n = ipar[0];
    *m = ipar[1];
    *p = ipar[2];
}
