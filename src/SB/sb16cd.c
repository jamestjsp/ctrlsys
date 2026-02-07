// SPDX-License-Identifier: BSD-3-Clause
// Translated from SLICOT SB16CD.f

#include "slicot.h"
#include "slicot_blas.h"

void sb16cd(
    const char* dico,
    const char* jobd,
    const char* jobmr,
    const char* jobcf,
    const char* ordsel,
    const i32 n,
    const i32 m,
    const i32 p,
    i32* ncr,
    f64* a, const i32 lda,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    const f64* d, const i32 ldd,
    f64* f, const i32 ldf,
    f64* g, const i32 ldg,
    f64* hsv,
    const f64 tol,
    i32* iwork,
    f64* dwork, const i32 ldwork,
    i32* iwarn,
    i32* info)
{
    *info = 0;
    *iwarn = 0;

    bool discr = (dico[0] == 'D' || dico[0] == 'd');
    bool withd = (jobd[0] == 'D' || jobd[0] == 'd');
    bool bal = (jobmr[0] == 'B' || jobmr[0] == 'b');
    bool left = (jobcf[0] == 'L' || jobcf[0] == 'l');
    bool fixord = (ordsel[0] == 'F' || ordsel[0] == 'f');

    i32 mp = left ? m : p;

    i32 n_mp = n > mp ? n : mp;
    i32 n_m_p = m > p ? m : p;
    i32 min_n_mp = n < mp ? n : mp;
    i32 lw = 2 * n * n;
    i32 lw1 = 2 * n * n + 5 * n;
    i32 lw2 = n * n_m_p;
    i32 lw3 = n * (n + n_mp + min_n_mp + 6);
    i32 lw_max = lw1;
    if (lw2 > lw_max) lw_max = lw2;
    if (lw3 > lw_max) lw_max = lw3;
    if (1 > lw_max) lw_max = 1;
    lw += lw_max;

    if (!(dico[0] == 'C' || dico[0] == 'c' || discr)) {
        *info = -1;
    } else if (!(withd || jobd[0] == 'Z' || jobd[0] == 'z')) {
        *info = -2;
    } else if (!(bal || jobmr[0] == 'F' || jobmr[0] == 'f')) {
        *info = -3;
    } else if (!(left || jobcf[0] == 'R' || jobcf[0] == 'r')) {
        *info = -4;
    } else if (!(fixord || ordsel[0] == 'A' || ordsel[0] == 'a')) {
        *info = -5;
    } else if (n < 0) {
        *info = -6;
    } else if (m < 0) {
        *info = -7;
    } else if (p < 0) {
        *info = -8;
    } else if (fixord && (*ncr < 0 || *ncr > n)) {
        *info = -9;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -11;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -13;
    } else if (ldc < (p > 1 ? p : 1)) {
        *info = -15;
    } else if (ldd < 1 || (withd && ldd < p)) {
        *info = -17;
    } else if (ldf < (m > 1 ? m : 1)) {
        *info = -19;
    } else if (ldg < (n > 1 ? n : 1)) {
        *info = -21;
    } else if (ldwork < lw) {
        *info = -26;
    }

    if (*info != 0) {
        return;
    }

    i32 min_nmp = n;
    if (m < min_nmp) min_nmp = m;
    if (p < min_nmp) min_nmp = p;
    if (min_nmp == 0 || (fixord && *ncr == 0)) {
        *ncr = 0;
        dwork[0] = 1.0;
        return;
    }

    i32 kt = 0;
    i32 kti = kt + n * n;
    i32 kw = kti + n * n;

    f64 scalec, scaleo;
    i32 sb16cy_info = 0;
    sb16cy(dico, jobcf, n, m, p, a, lda, b, ldb, c, ldc,
           f, ldf, g, ldg,
           &scalec, &scaleo,
           &dwork[kti], n,
           &dwork[kt], n,
           &dwork[kw], ldwork - kw, &sb16cy_info);

    if (sb16cy_info != 0) {
        *info = sb16cy_info;
        return;
    }

    i32 wrkopt = (i32)dwork[kw] + kw;

    i32 nminr = 0;
    i32 ab09ix_iwarn = 0;
    i32 ab09ix_info = ab09ix(dico, jobmr, "N", ordsel, n, m, p, ncr,
                              scalec, scaleo, a, lda, b, ldb, c, ldc, (f64*)d, ldd,
                              &dwork[kti], n, &dwork[kt], n,
                              &nminr, hsv, tol, tol,
                              iwork, &dwork[kw], ldwork - kw, &ab09ix_iwarn);

    if (ab09ix_info != 0) {
        *info = 6;
        return;
    }
    *iwarn = ab09ix_iwarn;

    i32 kw_opt = (i32)dwork[kw] + kw;
    if (kw_opt > wrkopt) wrkopt = kw_opt;

    f64 dbl0 = 0.0;
    f64 dbl1 = 1.0;

    SLC_DLACPY("F", &n, &p, g, &ldg, &dwork[kw], &n);
    SLC_DGEMM("N", "N", ncr, &p, &n, &dbl1, &dwork[kti], &n, &dwork[kw], &n, &dbl0, g, &ldg);

    SLC_DLACPY("F", &m, &n, f, &ldf, &dwork[kw], &m);
    SLC_DGEMM("N", "N", &m, ncr, &n, &dbl1, &dwork[kw], &m, &dwork[kt], &n, &dbl0, f, &ldf);

    SLC_DLACPY("F", &p, ncr, c, &ldc, dwork, &p);
    if (withd) {
        SLC_DGEMM("N", "N", &p, ncr, &m, &dbl1, d, &ldd, f, &ldf, &dbl1, dwork, &p);
    }
    SLC_DGEMM("N", "N", ncr, ncr, &p, &dbl1, g, &ldg, dwork, &p, &dbl1, a, &lda);
    SLC_DGEMM("N", "N", ncr, ncr, &m, &dbl1, b, &ldb, f, &ldf, &dbl1, a, &lda);

    dwork[0] = (f64)wrkopt;
}
