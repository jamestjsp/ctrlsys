#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>
#include <math.h>

void ib01py(const char *meth, const char *job, i32 nobr, i32 n, i32 m, i32 l,
            i32 rankr1, f64 *ul, i32 ldul, const f64 *r1, i32 ldr1,
            const f64 *tau1, const f64 *pgal, i32 ldpgal,
            f64 *k, i32 ldk, f64 *r, i32 ldr, f64 *h, i32 ldh,
            f64 *b, i32 ldb, f64 *d, i32 ldd, f64 tol,
            i32 *iwork, f64 *dwork, i32 ldwork, i32 *iwarn, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 THREE = 3.0;

    bool moesp = (meth[0] == 'M' || meth[0] == 'm');
    bool n4sid = (meth[0] == 'N' || meth[0] == 'n');
    bool withd = (job[0] == 'D' || job[0] == 'd');
    bool withb = (job[0] == 'B' || job[0] == 'b') || withd;

    i32 mnobr = m * nobr;
    i32 lnobr = l * nobr;
    i32 ldun2 = lnobr - l;
    i32 lp1 = l + 1;
    i32 nrow = moesp ? (lnobr - n) : (n + l);
    i32 nrowml = nrow - l;

    *iwarn = 0;
    *info = 0;

    if (!(moesp || n4sid)) {
        *info = -1;
    } else if (!(withb || job[0] == 'N' || job[0] == 'n')) {
        *info = -2;
    } else if (nobr <= 1) {
        *info = -3;
    } else if (n >= nobr || n <= 0) {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (l <= 0) {
        *info = -6;
    } else if ((moesp && withb && m > 0) && (rankr1 < 0 || rankr1 > n)) {
        *info = -7;
    } else if ((moesp && ldul < lnobr) || (n4sid && ldul < nrow)) {
        *info = -9;
    } else if (ldr1 < 1 || (m > 0 && withb && moesp && ldr1 < ldun2 && rankr1 == n)) {
        *info = -11;
    } else if (ldpgal < 1 || (ldpgal < n && (n4sid || (withb && m > 0 && moesp && rankr1 < n)))) {
        *info = -14;
    } else if (ldk < nrow) {
        *info = -16;
    } else if (ldr < lnobr) {
        *info = -18;
    } else if (ldh < lnobr) {
        *info = -20;
    } else if (ldb < 1 || (m > 0 && withb && ldb < n)) {
        *info = -22;
    } else if (ldd < 1 || (m > 0 && withd && ldd < l)) {
        *info = -24;
    } else {
        i32 minwrk = 2 * l;
        if (lnobr > minwrk) minwrk = lnobr;
        if (l + mnobr > minwrk) minwrk = l + mnobr;

        if (m > 0 && withb) {
            if (4 * lnobr + 1 > minwrk) minwrk = 4 * lnobr + 1;
            if (lnobr + m > minwrk) minwrk = lnobr + m;
        }

        if (ldwork < minwrk) {
            *info = -28;
            dwork[0] = (f64)minwrk;
        }
    }

    if (*info != 0) {
        return;
    }

    i32 itau = 0;
    i32 jwork = itau + l;

    if (moesp) {
        for (i32 i = 0; i < nobr; i++) {
            ma02ad("Full", l, nrow, &ul[(l * i) + (n) * ldul], ldul,
                   &r[(l * (nobr - i - 1)) * ldr], ldr);
        }
    } else {
        i32 jl = lnobr - 1;
        i32 jm = ldun2 - 1;

        for (i32 ji = 0; ji < ldun2; ji += l) {
            for (i32 j = ji + l - 1; j >= ji; j--) {
                for (i32 i = 0; i < n; i++) {
                    r[i + j * ldr] = pgal[i + jm * ldpgal] - ul[i + jl * ldul];
                }
                for (i32 i = n; i < nrow; i++) {
                    r[i + j * ldr] = -ul[i + jl * ldul];
                }
                jl--;
                jm--;
            }
        }

        for (i32 j = lnobr - 1; j >= ldun2; j--) {
            for (i32 i = 0; i < nrow; i++) {
                r[i + j * ldr] = -ul[i + jl * ldul];
            }
            jl--;
            r[(n + j - ldun2) + j * ldr] = ONE + r[(n + j - ldun2) + j * ldr];
        }
    }

    SLC_DGEQRF(&nrow, &l, r, &ldr, &dwork[itau], &dwork[jwork], &(i32){ldwork - jwork}, info);

    SLC_DORMQR("Left", "Transpose", &nrow, &ldun2, &l, r, &ldr,
               &dwork[itau], &r[lp1 * ldr], &ldr, &dwork[jwork], &(i32){ldwork - jwork}, info);

    SLC_DORMQR("Left", "Transpose", &nrow, &mnobr, &l, r, &ldr,
               &dwork[itau], k, &ldk, &dwork[jwork], &(i32){ldwork - jwork}, info);

    SLC_DLACPY("Full", &nrowml, &ldun2, &r[l + lp1 * ldr], &ldr, ul, &ldul);

    SLC_DLACPY("Full", &l, &m, k, &ldk, h, &ldh);

    for (i32 i = 0; i < nobr - 1; i++) {
        i32 lnobr_li1 = lnobr - l * (i + 1);
        SLC_DLACPY("Upper", &l, &lnobr_li1, &r[l * i + l * i * ldr], &ldr,
                   &r[l * (i + 1) + l * (i + 1) * ldr], &ldr);

        i32 lnobr_lip1 = lnobr - l * (i + 2);
        mb04od("Full", l, lnobr_lip1, nrowml, &r[l * (i + 1) + l * (i + 1) * ldr], ldr,
               &ul[l * i * ldul], ldul, &r[l * (i + 1) + l * (i + 2) * ldr], ldr,
               &ul[l * (i + 1) * ldul], ldul, &dwork[itau], &dwork[jwork]);

        for (i32 j = 0; j < l; j++) {
            i32 m_nobr_i = m * (nobr - i - 1);
            mb04oy(&nrowml, &m_nobr_i, &ul[l * i * ldul + j], &dwork[j],
                   &k[j + m * (i + 1) * ldk], &ldk, &k[l], &ldk, &dwork[jwork]);
        }

        SLC_DLACPY("Full", &l, &m, &k[m * (i + 1) * ldk], &ldk, &h[l * (i + 1)], &ldh);
    }

    if (m == 0 || !withb) {
        i32 maxwrk_val = 2 * l;
        if (lnobr > maxwrk_val) maxwrk_val = lnobr;
        if (l + mnobr > maxwrk_val) maxwrk_val = l + mnobr;
        dwork[0] = (f64)maxwrk_val;
        return;
    }

    f64 eps = SLC_DLAMCH("Precision");
    f64 thresh = pow(eps, TWO / THREE);
    f64 toll = tol;
    if (toll <= ZERO) toll = (f64)(lnobr * lnobr) * eps;
    f64 svlmax = ZERO;

    f64 rcond;
    SLC_DTRCON("1-norm", "Upper", "NonUnit", &lnobr, r, &ldr, &rcond,
               dwork, iwork, info);

    if (rcond > (toll > thresh ? toll : thresh)) {
        SLC_DTRSM("Left", "Upper", "NoTranspose", "Non-unit",
                  &lnobr, &m, &ONE, r, &ldr, h, &ldh);
    } else {
        for (i32 i = 0; i < lnobr; i++) {
            iwork[i] = 0;
        }

        jwork = itau + lnobr;
        i32 lnobr_m1 = lnobr - 1;
        SLC_DLASET("Lower", &lnobr_m1, &lnobr, &ZERO, &ZERO, &r[1], &ldr);

        i32 rank;
        f64 sval[3];
        mb03od("QR", lnobr, lnobr, r, ldr, iwork, toll, svlmax,
               &dwork[itau], &rank, sval, &dwork[jwork], ldwork - jwork, info);

        SLC_DORMQR("Left", "Transpose", &lnobr, &m, &lnobr, r, &ldr,
                   &dwork[itau], h, &ldh, &dwork[jwork], &(i32){ldwork - jwork}, info);

        if (rank < lnobr) {
            *iwarn = 4;
        }

        mb02qy(lnobr, lnobr, m, rank, r, ldr, iwork, h, ldh,
               &dwork[itau], &dwork[jwork], ldwork - jwork, info);
    }

    if (withd) {
        SLC_DLACPY("Full", &l, &m, &h[ldun2], &ldh, d, &ldd);
    }

    i32 nobrh = nobr / 2 + (nobr % 2) - 1;
    for (i32 j = 0; j < m; j++) {
        for (i32 i = 0; i < nobrh; i++) {
            SLC_DSWAP(&l, &h[l * i + j * ldh], &(i32){1},
                      &h[l * (nobr - i - 2) + j * ldh], &(i32){1});
        }
    }

    if (moesp && rankr1 == n) {
        SLC_DORMQR("Left", "Transpose", &ldun2, &m, &n, (f64*)r1, &ldr1,
                   (f64*)tau1, h, &ldh, dwork, &ldwork, info);

        SLC_DLACPY("Full", &n, &m, h, &ldh, b, &ldb);

        SLC_DTRTRS("Upper", "NoTranspose", "NonUnit", &n, &m, (f64*)r1, &ldr1,
                   b, &ldb, info);
        if (*info > 0) {
            *info = 3;
            return;
        }
    } else {
        SLC_DGEMM("NoTranspose", "NoTranspose", &n, &m, &ldun2, &ONE,
                  (f64*)pgal, &ldpgal, h, &ldh, &ZERO, b, &ldb);
    }

    i32 maxwrk = 2 * l;
    if (lnobr > maxwrk) maxwrk = lnobr;
    if (l + mnobr > maxwrk) maxwrk = l + mnobr;
    dwork[0] = (f64)maxwrk;
    dwork[1] = rcond;
}
