#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>
#include <stdlib.h>

void ib01px(const char *job, i32 nobr, i32 n, i32 m, i32 l,
            f64 *uf, i32 lduf, const f64 *un, i32 ldun,
            f64 *ul, i32 ldul, const f64 *pgal, i32 ldpgal,
            const f64 *k, i32 ldk, f64 *r, i32 ldr, f64 *x,
            f64 *b, i32 ldb, f64 *d, i32 ldd, f64 tol,
            i32 *iwork, f64 *dwork, i32 ldwork, i32 *iwarn, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool withd = (job[0] == 'D' || job[0] == 'd');
    bool withb = (job[0] == 'B' || job[0] == 'b') || withd;

    i32 mnobr = m * nobr;
    i32 lnobr = l * nobr;
    i32 ldun2 = lnobr - l;
    i32 npl = n + l;

    *iwarn = 0;
    *info = 0;

    if (!withb) {
        *info = -1;
    } else if (nobr <= 1) {
        *info = -2;
    } else if (n >= nobr || n <= 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (l <= 0) {
        *info = -5;
    } else if (lduf < 1 || (mnobr > 0 && lduf < mnobr)) {
        *info = -7;
    } else if (ldun < ldun2) {
        *info = -9;
    } else if (ldul < npl) {
        *info = -11;
    } else if (ldpgal < n) {
        *info = -13;
    } else if (ldk < npl) {
        *info = -15;
    } else if (ldr < 1 || (mnobr * npl > 0 && ldr < mnobr * npl)) {
        *info = -17;
    } else if (ldb < n) {
        *info = -20;
    } else if (ldd < 1 || (withd && ldd < l)) {
        *info = -22;
    } else {
        i32 minwrk = npl * npl;
        i32 alt = 4 * m * npl + 1;
        if (alt > minwrk) minwrk = alt;

        if (ldwork < minwrk) {
            *info = -26;
            dwork[0] = (f64)minwrk;
        }
    }

    if (*info != 0) {
        return;
    }

    if (m == 0) {
        dwork[0] = ONE;
        return;
    }

    for (i32 j = 0; j < l; j++) {
        for (i32 i = 0; i < npl; i++) {
            ul[i + j * ldul] = -ul[i + j * ldul];
        }
        ul[n + j + j * ldul] = ONE + ul[n + j + j * ldul];
    }

    for (i32 j = l; j < lnobr; j++) {
        for (i32 i = 0; i < n; i++) {
            ul[i + j * ldul] = pgal[i + (j - l) * ldpgal] - ul[i + j * ldul];
        }
        for (i32 i = n; i < npl; i++) {
            ul[i + j * ldul] = -ul[i + j * ldul];
        }
    }

    i32 mkron_npl = mnobr * npl;
    i32 nkron = m * npl;
    SLC_DLASET("Full", &mkron_npl, &nkron, &ZERO, &ZERO, r, &ldr);

    i32 mnobr_m1 = mnobr - 1;
    if (mnobr > 1) {
        SLC_DLASET("Lower", &mnobr_m1, &mnobr_m1, &ZERO, &ZERO, &uf[1], &lduf);
    }

    i32 jwork = npl * l;

    for (i32 i = 0; i < nobr; i++) {
        SLC_DLACPY("Full", &npl, &l, &ul[i * l * ldul], &ldul, dwork, &npl);

        if (i < nobr - 1) {
            i32 ncols = l * (nobr - i - 1);
            SLC_DGEMM("NoTranspose", "NoTranspose", &npl, &n, &ncols,
                      &ONE, &ul[(i + 1) * l * ldul], &ldul, un, &ldun,
                      &ZERO, &dwork[jwork], &npl);
        } else {
            SLC_DLASET("Full", &npl, &n, &ZERO, &ZERO, &dwork[jwork], &npl);
        }

        i32 mkron, nkron_out;
        mb01vd("NoTranspose", "NoTranspose", mnobr, m, npl, npl,
               ONE, ONE, &uf[i * m * lduf], lduf, dwork, npl,
               r, ldr, &mkron, &nkron_out, info);

        SLC_DLACPY("Full", &npl, &m, &k[i * m * ldk], &ldk,
                   &x[i * nkron_out], &npl);
    }

    f64 toll = tol;
    if (toll <= ZERO) {
        toll = (f64)(mkron_npl * nkron) * SLC_DLAMCH("Precision");
    }

    for (i32 i = 0; i < nkron; i++) {
        iwork[i] = 0;
    }

    i32 nrhs = 1;
    i32 rank;
    SLC_DGELSY(&mkron_npl, &nkron, &nrhs, r, &ldr, x, &mkron_npl, iwork,
               &toll, &rank, dwork, &ldwork, info);
    i32 maxwrk = (i32)dwork[0];

    f64 rcond;
    SLC_DTRCON("1-norm", "Upper", "NonUnit", &nkron, r, &ldr, &rcond,
               dwork, iwork, info);

    if (rank < nkron) {
        *iwarn = 4;
    }

    if (withd) {
        SLC_DLACPY("Full", &l, &m, x, &npl, d, &ldd);
    }

    SLC_DLACPY("Full", &n, &m, &x[l], &npl, b, &ldb);

    i32 minwrk = npl * npl;
    i32 alt = 4 * m * npl + 1;
    if (alt > minwrk) minwrk = alt;
    dwork[0] = (f64)(minwrk > maxwrk ? minwrk : maxwrk);
    dwork[1] = rcond;
}
