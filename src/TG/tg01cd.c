// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"

void tg01cd(const char *compq, i32 l, i32 n, i32 m,
            f64 *a, i32 lda, f64 *e, i32 lde,
            f64 *b, i32 ldb, f64 *q, i32 ldq,
            f64 *dwork, i32 ldwork, i32 *info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool ilq;
    i32 icompq;

    if (*compq == 'N' || *compq == 'n') {
        ilq = false;
        icompq = 1;
    } else if (*compq == 'U' || *compq == 'u') {
        ilq = true;
        icompq = 2;
    } else if (*compq == 'I' || *compq == 'i') {
        ilq = true;
        icompq = 3;
    } else {
        icompq = 0;
    }

    *info = 0;
    i32 ln = (l < n) ? l : n;
    i32 maxlnm = l;
    if (n > maxlnm) maxlnm = n;
    if (m > maxlnm) maxlnm = m;
    i32 wrkopt = (ln + maxlnm > 1) ? ln + maxlnm : 1;

    if (icompq == 0) {
        *info = -1;
    } else if (l < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (lda < (l > 1 ? l : 1)) {
        *info = -6;
    } else if (lde < (l > 1 ? l : 1)) {
        *info = -8;
    } else if (ldb < 1 || (m > 0 && ldb < l)) {
        *info = -10;
    } else if ((ilq && ldq < l) || ldq < 1) {
        *info = -12;
    } else if (ldwork < wrkopt) {
        *info = -14;
    }

    if (*info != 0) {
        return;
    }

    if (icompq == 3) {
        SLC_DLASET("F", &l, &l, &ZERO, &ONE, q, &ldq);
    }

    if (l == 0 || n == 0) {
        dwork[0] = ONE;
        return;
    }

    i32 ierr;
    i32 iwrk = ln;
    i32 lwork_avail = ldwork - ln;

    SLC_DGEQRF(&l, &n, e, &lde, dwork, &dwork[iwrk], &lwork_avail, &ierr);
    wrkopt = (i32)dwork[iwrk] + ln;

    SLC_DORMQR("L", "T", &l, &n, &ln, e, &lde, dwork,
               a, &lda, &dwork[iwrk], &lwork_avail, &ierr);
    i32 opt = (i32)dwork[iwrk] + ln;
    if (opt > wrkopt) wrkopt = opt;

    if (m > 0) {
        SLC_DORMQR("L", "T", &l, &m, &ln, e, &lde, dwork,
                   b, &ldb, &dwork[iwrk], &lwork_avail, &ierr);
        opt = (i32)dwork[iwrk] + ln;
        if (opt > wrkopt) wrkopt = opt;
    }

    if (ilq) {
        SLC_DORMQR("R", "N", &l, &l, &ln, e, &lde, dwork,
                   q, &ldq, &dwork[iwrk], &lwork_avail, &ierr);
        opt = (i32)dwork[iwrk] + ln;
        if (opt > wrkopt) wrkopt = opt;
    }

    if (l >= 2) {
        i32 lm1 = l - 1;
        SLC_DLASET("L", &lm1, &ln, &ZERO, &ZERO, &e[1], &lde);
    }

    dwork[0] = (f64)wrkopt;
}
