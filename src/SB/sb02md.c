/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB02MD - Continuous/Discrete-time Algebraic Riccati Equation Solver
 *
 * Solves for X the continuous-time algebraic Riccati equation
 *   Q + A'*X + X*A - X*G*X = 0
 *
 * or the discrete-time algebraic Riccati equation
 *   X = A'*X*A - A'*X*B*(R + B'*X*B)^-1 B'*X*A + Q
 *
 * where G = B*R^-1*B' must be provided on input.
 *
 * Uses Laub's Schur vector approach.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <ctype.h>
#include <math.h>

static int select_stable_continuous(const f64* reig, const f64* ieig) {
    return *reig < 0.0;
}

static int select_unstable_continuous(const f64* reig, const f64* ieig) {
    return *reig >= 0.0;
}

static int select_stable_discrete(const f64* reig, const f64* ieig) {
    f64 modulus = sqrt((*reig) * (*reig) + (*ieig) * (*ieig));
    return modulus < 1.0;
}

static int select_unstable_discrete(const f64* reig, const f64* ieig) {
    f64 modulus = sqrt((*reig) * (*reig) + (*ieig) * (*ieig));
    return modulus >= 1.0;
}

void sb02md(
    const char* dico_str,
    const char* hinv_str,
    const char* uplo_str,
    const char* scal_str,
    const char* sort_str,
    const i32 n,
    f64* a,
    const i32 lda,
    const f64* g,
    const i32 ldg,
    f64* q,
    const i32 ldq,
    f64* rcond,
    f64* wr,
    f64* wi,
    f64* s,
    const i32 lds,
    f64* u,
    const i32 ldu,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 half = 0.5;
    const f64 one = 1.0;
    const i32 int1 = 1;

    char dico = toupper((unsigned char)dico_str[0]);
    char hinv = toupper((unsigned char)hinv_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);
    char scal = toupper((unsigned char)scal_str[0]);
    char sort = toupper((unsigned char)sort_str[0]);

    bool discr = (dico == 'D');
    bool lscal = (scal == 'G');
    bool lsort = (sort == 'S');
    bool lhinv = false;

    if (discr) {
        lhinv = (hinv == 'D');
    }

    i32 n2 = n + n;
    i32 np1 = n + 1;

    *info = 0;

    if (dico != 'C' && dico != 'D') {
        *info = -1;
    } else if (discr && hinv != 'D' && hinv != 'I') {
        *info = -2;
    } else if (uplo != 'U' && uplo != 'L') {
        *info = -3;
    } else if (scal != 'G' && scal != 'N') {
        *info = -4;
    } else if (sort != 'S' && sort != 'U') {
        *info = -5;
    } else if (n < 0) {
        *info = -6;
    } else if (lda < 1 || (n > 0 && lda < n)) {
        *info = -8;
    } else if (ldg < 1 || (n > 0 && ldg < n)) {
        *info = -10;
    } else if (ldq < 1 || (n > 0 && ldq < n)) {
        *info = -12;
    } else if (lds < 1 || (n > 0 && lds < n2)) {
        *info = -17;
    } else if (ldu < 1 || (n > 0 && ldu < n2)) {
        *info = -19;
    } else {
        i32 minwrk = discr ? (6 * n > 3 ? 6 * n : 3) : (6 * n > 2 ? 6 * n : 2);
        if (ldwork < minwrk) {
            *info = -22;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        *rcond = one;
        dwork[0] = one;
        dwork[1] = one;
        if (discr) dwork[2] = one;
        return;
    }

    f64 qnorm = 0.0;
    f64 gnorm = 0.0;

    if (lscal) {
        qnorm = SLC_DLANSY("1", uplo_str, &n, q, &ldq, dwork);
        gnorm = SLC_DLANSY("1", uplo_str, &n, g, &ldg, dwork);
    }

    sb02mu(dico_str, hinv_str, uplo_str, n, a, lda, g, ldg, q, ldq, s, lds,
           iwork, dwork, ldwork, info);

    if (*info != 0) {
        *info = 1;
        return;
    }

    f64 wrkopt = dwork[0];
    f64 rconda = 0.0;
    if (discr) rconda = dwork[1];

    i32 iscl = 0;
    if (lscal) {
        if (qnorm > gnorm && gnorm > zero) {
            i32 ierr;
            SLC_DLASCL("G", &(i32){0}, &(i32){0}, &qnorm, &gnorm, &n, &n,
                       &s[(np1 - 1)], &lds, &ierr);
            SLC_DLASCL("G", &(i32){0}, &(i32){0}, &gnorm, &qnorm, &n, &n,
                       &s[(np1 - 1) * lds], &lds, &ierr);
            iscl = 1;
        }
    }

    typedef int (*SelectFn)(const f64*, const f64*);
    SelectFn select_fn;
    i32 sdim;

    if (!discr) {
        if (lsort) {
            select_fn = select_stable_continuous;
        } else {
            select_fn = select_unstable_continuous;
        }
    } else {
        if (lsort) {
            select_fn = select_stable_discrete;
        } else {
            select_fn = select_unstable_discrete;
        }
    }

    SLC_DGEES("V", "S", select_fn, &n2, s, &lds, &sdim,
              wr, wi, u, &ldu, dwork, &ldwork, bwork, info);

    if (*info > n2) {
        *info = 3;
        return;
    } else if (*info > 0) {
        *info = 2;
        return;
    } else if (sdim != n) {
        *info = 4;
        return;
    }

    if (discr && lhinv) {
        SLC_DSWAP(&n, wr, &int1, &wr[np1 - 1], &int1);
        SLC_DSWAP(&n, wi, &int1, &wi[np1 - 1], &int1);
    }

    wrkopt = (wrkopt > dwork[0]) ? wrkopt : dwork[0];

    f64 unorm = SLC_DLANGE("1", &n, &n, u, &ldu, dwork);

    SLC_DLACPY("F", &n, &n, u, &ldu, &s[np1 - 1], &lds);

    SLC_DGETRF(&n, &n, &s[np1 - 1], &lds, iwork, info);

    if (*info > 0) {
        *info = 5;
        *rcond = zero;
        goto set_dwork;
    }

    SLC_DGECON("1", &n, &s[np1 - 1], &lds, &unorm, rcond,
               dwork, &iwork[np1 - 1], info);

    f64 eps = SLC_DLAMCH("E");
    if (*rcond < eps) {
        *info = 5;
        return;
    }

    for (i32 i = 0; i < n; i++) {
        SLC_DCOPY(&n, &u[(np1 - 1) + i * ldu], &int1, &q[i], &ldq);
    }

    SLC_DGETRS("T", &n, &n, &s[np1 - 1], &lds, iwork, q, &ldq, info);

    SLC_DLASET("F", &n, &n, &zero, &zero, &s[np1 - 1], &lds);

    for (i32 i = 0; i < n - 1; i++) {
        i32 len = n - i - 1;
        SLC_DAXPY(&len, &one, &q[i + (i + 1) * ldq], &ldq, &q[i + 1 + i * ldq], &int1);
        SLC_DSCAL(&len, &half, &q[i + 1 + i * ldq], &int1);
        SLC_DCOPY(&len, &q[i + 1 + i * ldq], &int1, &q[i + (i + 1) * ldq], &ldq);
    }

    if (lscal) {
        if (iscl == 1) {
            i32 ierr;
            SLC_DLASCL("G", &(i32){0}, &(i32){0}, &gnorm, &qnorm, &n, &n,
                       q, &ldq, &ierr);
        }
    }

    dwork[0] = wrkopt;

set_dwork:
    if (iscl == 1) {
        dwork[1] = qnorm / gnorm;
    } else {
        dwork[1] = one;
    }
    if (discr) dwork[2] = rconda;
}
