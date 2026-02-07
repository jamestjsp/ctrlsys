/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

static inline bool lsame(char ca, char cb) {
    if (ca >= 'a' && ca <= 'z') ca -= 32;
    if (cb >= 'a' && cb <= 'z') cb -= 32;
    return ca == cb;
}

static inline i32 max_i32(i32 a, i32 b) {
    return a > b ? a : b;
}

static inline i32 min_i32(i32 a, i32 b) {
    return a < b ? a : b;
}

static inline i32 max3_i32(i32 a, i32 b, i32 c) {
    return max_i32(a, max_i32(b, c));
}

static inline i32 max4_i32(i32 a, i32 b, i32 c, i32 d) {
    return max_i32(max_i32(a, b), max_i32(c, d));
}

void ab08nd(const char* equil, i32 n, i32 m, i32 p, f64* a, i32 lda,
            f64* b, i32 ldb, f64* c, i32 ldc, f64* d, i32 ldd,
            i32* nu, i32* rank, i32* dinfz, i32* nkror, i32* nkrol,
            i32* infz, i32* kronr, i32* kronl, f64* af, i32 ldaf,
            f64* bf, i32 ldbf, f64 tol, i32* iwork, f64* dwork, i32 ldwork,
            i32* info) {

    const f64 ZERO = 0.0, ONE = 1.0;

    bool lequil = lsame(*equil, 'S');
    bool lquery = (ldwork == -1);

    *info = 0;
    i32 ii = min_i32(p, m);
    i32 np = n + p;
    i32 nm = n + m;

    if (!lequil && !lsame(*equil, 'N')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (p < 0) {
        *info = -4;
    } else if (lda < max_i32(1, n)) {
        *info = -6;
    } else if (ldb < max_i32(1, n)) {
        *info = -8;
    } else if (ldc < max_i32(1, p)) {
        *info = -10;
    } else if (ldd < max_i32(1, p)) {
        *info = -12;
    } else if (ldaf < max_i32(1, n + m)) {
        *info = -22;
    } else if (ldbf < max_i32(1, n + p)) {
        *info = -24;
    } else {
        i32 minwork = max4_i32(ii + max_i32(3*m - 1, n),
                               min_i32(p, n) + max3_i32(3*p - 1, np, nm),
                               min_i32(m, n) + max_i32(3*m - 1, nm),
                               1);
        if (lquery) {
            f64 svlmax = ZERO;
            i32 ninfz_q = 0;
            i32 mu, nu_q, nkrol_q;
            i32 ro_q = p, sigma_q = 0;
            ab08nx(n, m, p, &ro_q, &sigma_q, svlmax, bf, max_i32(1, ldbf),
                   &ninfz_q, infz, kronl, &mu, &nu_q, &nkrol_q, tol, iwork,
                   dwork, -1, info);
            i32 wrkopt = max_i32(minwork, (i32)dwork[0]);

            ro_q = m - ii;
            sigma_q = ii;
            ab08nx(n, ii, m, &ro_q, &sigma_q, svlmax, af, max_i32(1, ldaf),
                   &ninfz_q, infz, kronl, &mu, &nu_q, &nkrol_q, tol, iwork,
                   dwork, -1, info);
            wrkopt = max_i32(wrkopt, (i32)dwork[0]);

            sl_int nii = n + ii, ii_sl = ii, n_sl = n;
            SLC_DTZRZF(&ii_sl, &nii, af, &ldaf, dwork, dwork, &(sl_int){-1}, info);
            wrkopt = max_i32(wrkopt, ii + (i32)dwork[0]);

            sl_int ldaf_sl = ldaf;
            SLC_DORMRZ("Right", "Transpose", &n_sl, &nii, &ii_sl, &n_sl, af,
                       &ldaf_sl, dwork, af, &ldaf_sl, dwork, &(sl_int){-1}, info);
            wrkopt = max_i32(wrkopt, ii + (i32)dwork[0]);
            dwork[0] = (f64)wrkopt;
            return;
        } else if (ldwork < minwork) {
            *info = -28;
        }
    }

    if (*info != 0) {
        return;
    }

    *dinfz = 0;
    *nkrol = 0;
    *nkror = 0;

    if (n == 0) {
        if (min_i32(m, p) == 0) {
            *nu = 0;
            *rank = 0;
            dwork[0] = ONE;
            return;
        }
    }

    i32 mm = m, nn = n, pp = p;

    for (i32 i = 0; i < n; i++) {
        infz[i] = 0;
    }

    if (m > 0) {
        for (i32 i = 0; i <= n; i++) {
            kronr[i] = 0;
        }
    }

    if (p > 0) {
        for (i32 i = 0; i <= n; i++) {
            kronl[i] = 0;
        }
    }

    sl_int nn_sl = nn, mm_sl = mm, pp_sl = pp;
    sl_int lda_sl = lda, ldb_sl = ldb, ldc_sl = ldc, ldd_sl = ldd;
    sl_int ldbf_sl = ldbf, ldaf_sl = ldaf;
    sl_int np_sl = np, nm_sl = nm;

    i32 wrkopt = 1;

    SLC_DLACPY("Full", &nn_sl, &mm_sl, b, &ldb_sl, bf, &ldbf_sl);
    if (pp > 0) {
        SLC_DLACPY("Full", &pp_sl, &mm_sl, d, &ldd_sl, &bf[nn], &ldbf_sl);
    }
    if (nn > 0) {
        SLC_DLACPY("Full", &nn_sl, &nn_sl, a, &lda_sl, &bf[(mm)*ldbf], &ldbf_sl);
        if (pp > 0) {
            SLC_DLACPY("Full", &pp_sl, &nn_sl, c, &ldc_sl, &bf[(mm)*ldbf + nn], &ldbf_sl);
        }
    }

    if (lequil && nn > 0 && pp > 0) {
        f64 maxred = ZERO;
        tb01id("A", nn, mm, pp, &maxred, &bf[(mm)*ldbf], ldbf, bf, ldbf,
               &bf[(mm)*ldbf + nn], ldbf, dwork, info);
        wrkopt = n;
    }

    f64 thresh = sqrt((f64)(np * nm)) * SLC_DLAMCH("Precision");
    f64 toler = tol;
    if (toler < thresh) toler = thresh;

    f64 svlmax = SLC_DLANGE("Frobenius", &np_sl, &nm_sl, bf, &ldbf_sl, dwork);

    i32 ro = pp;
    i32 sigma = 0;
    i32 ninfz = 0;
    i32 mu;

    ab08nx(nn, mm, pp, &ro, &sigma, svlmax, bf, ldbf, &ninfz, infz,
           kronl, &mu, nu, nkrol, toler, iwork, dwork, ldwork, info);
    wrkopt = max_i32(wrkopt, (i32)dwork[0]);
    *rank = mu;

    i32 numu = *nu + mu;
    if (numu != 0) {
        i32 mnu = mm + *nu;
        sl_int mnu_sl = mnu, neg_one = -1;
        for (i32 i = 0; i < numu; i++) {
            SLC_DCOPY(&mnu_sl, &bf[i], &ldbf_sl, &af[(numu - 1 - i) * ldaf], &neg_one);
        }

        if (mu != mm) {
            pp = mm;
            nn = *nu;
            mm = mu;

            ro = pp - mm;
            sigma = mm;
            ab08nx(nn, mm, pp, &ro, &sigma, svlmax, af, ldaf, &ninfz, infz,
                   kronr, &mu, nu, nkror, toler, iwork, dwork, ldwork, info);
            wrkopt = max_i32(wrkopt, (i32)dwork[0]);
        }

        if (*nu != 0) {
            sl_int nu_sl = *nu, mu_sl = mu;
            sl_int i1 = *nu + mu;
            sl_int i1_sl = i1;

            SLC_DLASET("Full", &nu_sl, &mu_sl, &ZERO, &ZERO, bf, &ldbf_sl);
            SLC_DLASET("Full", &nu_sl, &nu_sl, &ZERO, &ONE, &bf[mu * ldbf], &ldbf_sl);

            if (*rank != 0) {
                i32 nu1 = *nu;

                SLC_DTZRZF(&mu_sl, &i1_sl, &af[nu1], &ldaf_sl, dwork, &dwork[mu], &(sl_int){ldwork - mu}, info);
                wrkopt = max_i32(wrkopt, (i32)dwork[mu] + mu);

                SLC_DORMRZ("Right", "Transpose", &nu_sl, &i1_sl, &mu_sl, &nu_sl,
                           &af[nu1], &ldaf_sl, dwork, af, &ldaf_sl,
                           &dwork[mu], &(sl_int){ldwork - mu}, info);
                wrkopt = max_i32(wrkopt, (i32)dwork[mu] + mu);

                SLC_DORMRZ("Right", "Transpose", &nu_sl, &i1_sl, &mu_sl, &nu_sl,
                           &af[nu1], &ldaf_sl, dwork, bf, &ldbf_sl,
                           &dwork[mu], &(sl_int){ldwork - mu}, info);
            }

            SLC_DLACPY("Full", &nu_sl, &nu_sl, &af[mu * ldaf], &ldaf_sl, af, &ldaf_sl);
            if (*rank != 0) {
                SLC_DLACPY("Full", &nu_sl, &nu_sl, &bf[mu * ldbf], &ldbf_sl, bf, &ldbf_sl);
            }
        }
    }

    if (*nkror > 0) {
        i32 j = 0;
        for (i32 i = 0; i <= n; i++) {
            for (i32 ii_idx = j; ii_idx < j + kronr[i]; ii_idx++) {
                if (ii_idx < max_i32(n, m) + 1) {
                    iwork[ii_idx] = i;
                }
            }
            j += kronr[i];
            kronr[i] = 0;
        }
        *nkror = j;
        for (i32 i = 0; i < *nkror; i++) {
            kronr[i] = iwork[i];
        }
    }

    if (*nkrol > 0) {
        i32 j = 0;
        for (i32 i = 0; i <= n; i++) {
            for (i32 ii_idx = j; ii_idx < j + kronl[i]; ii_idx++) {
                if (ii_idx < max_i32(n, p) + 1) {
                    iwork[ii_idx] = i;
                }
            }
            j += kronl[i];
            kronl[i] = 0;
        }
        *nkrol = j;
        for (i32 i = 0; i < *nkrol; i++) {
            kronl[i] = iwork[i];
        }
    }

    if (n > 0) {
        *dinfz = n;
        while (*dinfz > 0 && infz[*dinfz - 1] == 0) {
            (*dinfz)--;
        }
    }

    dwork[0] = (f64)wrkopt;
}
