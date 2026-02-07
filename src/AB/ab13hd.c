// SPDX-License-Identifier: BSD-3-Clause
//
// AB13HD - L-infinity norm of proper continuous-time or causal discrete-time
//          descriptor state-space system
//
// Translated from SLICOT AB13HD.f (Fortran 77)
// Note: This is a simplified implementation that handles the basic cases.
// Complex descriptor system cases (JOBE='G' or 'C') require additional
// SLICOT routines not yet translated.

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <complex.h>

static inline i32 max_i32(i32 a, i32 b) { return a > b ? a : b; }
static inline i32 min_i32(i32 a, i32 b) { return a < b ? a : b; }
static inline f64 min_f64(f64 a, f64 b) { return a < b ? a : b; }

void ab13hd(const char *dico, const char *jobe, const char *equil,
            const char *jobd, const char *ckprop, const char *reduce,
            const char *poles, i32 n, i32 m, i32 p, i32 ranke, f64 *fpeak,
            f64 *a, i32 lda, f64 *e, i32 lde, f64 *b, i32 ldb,
            f64 *c, i32 ldc, f64 *d, i32 ldd, i32 *nr, f64 *gpeak, f64 *tol,
            i32 *iwork, f64 *dwork, i32 ldwork, c128 *zwork, i32 lzwork,
            bool *bwork, i32 *iwarn, i32 *info)
{
    bool discr = (*dico == 'D' || *dico == 'd');
    bool contd = (*dico == 'C' || *dico == 'c');
    bool unite = (*jobe == 'I' || *jobe == 'i');
    bool gene = (*jobe == 'G' || *jobe == 'g');
    bool cmpre = (*jobe == 'C' || *jobe == 'c');
    bool lequil = (*equil == 'S' || *equil == 's');
    bool nequil = (*equil == 'N' || *equil == 'n');
    bool withd = (*jobd == 'D' || *jobd == 'd');
    bool fullrd = (*jobd == 'F' || *jobd == 'f');
    bool zerod = (*jobd == 'Z' || *jobd == 'z');
    bool wckprp = (*ckprop == 'C' || *ckprop == 'c');
    bool nckprp = (*ckprop == 'N' || *ckprop == 'n');
    bool wreduc = (*reduce == 'R' || *reduce == 'r');
    bool nreduc = (*reduce == 'N' || *reduce == 'n');
    bool allpol = (*poles == 'A' || *poles == 'a');
    bool stabp = (*poles == 'P' || *poles == 'p');
    bool withe = gene || cmpre;
    bool lquery = (ldwork == -1) || (lzwork == -1);

    i32 minpm = min_i32(p, m);

    *iwarn = 0;
    *info = 0;

    if (!discr && !contd) {
        *info = -1;
    } else if (!withe && !unite) {
        *info = -2;
    } else if (!lequil && !nequil) {
        *info = -3;
    } else if (!withd && !fullrd && !zerod) {
        *info = -4;
    } else if (!wckprp && !nckprp) {
        if (!(discr || unite))
            *info = -5;
    } else if (!wreduc && !nreduc) {
        if (wckprp)
            *info = -6;
    } else if (!allpol && !stabp) {
        *info = -7;
    } else if (n < 0) {
        *info = -8;
    } else if (m < 0) {
        *info = -9;
    } else if (p < 0) {
        *info = -10;
    } else if (cmpre && (ranke < 0 || ranke > n)) {
        *info = -11;
    } else if (min_f64(fpeak[0], fpeak[1]) < 0.0) {
        *info = -12;
    } else if (lda < max_i32(1, n)) {
        *info = -14;
    } else if (lde < 1 || (gene && lde < n) || (cmpre && lde < ranke)) {
        *info = -16;
    } else if (ldb < max_i32(1, n)) {
        *info = -18;
    } else if (ldc < max_i32(1, p)) {
        *info = -20;
    } else if (ldd < 1 || (!zerod && ldd < p)) {
        *info = -22;
    } else if (tol[0] < 0.0 || tol[0] >= 1.0) {
        *info = -25;
    } else if (!lequil && tol[1] >= 1.0) {
        *info = -25;
    }

    if (*info != 0) {
        return;
    }

    *nr = n;

    if (n == 0 || minpm == 0) {
        gpeak[0] = 0.0;
        gpeak[1] = 1.0;
        if (dwork != NULL) dwork[0] = 1.0;
        if (zwork != NULL) zwork[0] = (c128)1;
        return;
    }

    if (gene || cmpre) {
        *info = -2;
        return;
    }

    if (unite) {
        const char *jbdd = zerod ? "Z" : "D";

        if (lquery) {
            f64 dwork_query[1];
            c128 cwork_query[1];
            i32 info_dd = 0;

            ab13dd(dico, "I", equil, jbdd, n, m, p, fpeak,
                   a, lda, e, lde, b, ldb, c, ldc, d, ldd,
                   gpeak, tol[0], iwork, dwork_query, -1, cwork_query, -1, &info_dd);

            if (dwork != NULL) dwork[0] = dwork_query[0];
            if (zwork != NULL) zwork[0] = cwork_query[0];
            return;
        }

        i32 info_dd = 0;
        ab13dd(dico, "I", equil, jbdd, n, m, p, fpeak,
               a, lda, e, lde, b, ldb, c, ldc, d, ldd,
               gpeak, tol[0], iwork, dwork, ldwork, zwork, lzwork, &info_dd);

        *info = info_dd;
        return;
    }
}
