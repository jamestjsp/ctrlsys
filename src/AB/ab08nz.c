/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>

i32 slicot_ab8nxz(i32 n, i32 m, i32 p, i32* ro, i32* sigma, f64 svlmax,
                  c128* abcd, i32 ldabcd, i32* ninfz, i32* infz, i32* kronl,
                  i32* mu, i32* nu, i32* nkrol, f64 tol, i32* iwork,
                  f64* dwork, c128* zwork, i32 lzwork);

i32 slicot_ab08nz(char equil, i32 n, i32 m, i32 p, c128* a, i32 lda,
                  c128* b, i32 ldb, c128* c, i32 ldc, c128* d, i32 ldd,
                  i32* nu, i32* rank, i32* dinfz, i32* nkror, i32* nkrol,
                  i32* infz, i32* kronr, i32* kronl, c128* af, i32 ldaf,
                  c128* bf, i32 ldbf, f64 tol, i32* iwork, f64* dwork,
                  c128* zwork, i32 lzwork) {
    const c128 ZERO = 0.0 + 0.0*I;
    const c128 ONE = 1.0 + 0.0*I;
    const f64 DZERO = 0.0;

    i32 info = 0;
    char equil_up = (char)toupper((unsigned char)equil);
    bool lequil = (equil_up == 'S');
    bool lquery = (lzwork == -1);

    if (!lequil && equil_up != 'N') {
        info = -1;
    } else if (n < 0) {
        info = -2;
    } else if (m < 0) {
        info = -3;
    } else if (p < 0) {
        info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        info = -6;
    } else if (ldb < (n > 1 ? n : 1)) {
        info = -8;
    } else if (ldc < (p > 1 ? p : 1)) {
        info = -10;
    } else if (ldd < (p > 1 ? p : 1)) {
        info = -12;
    } else if (ldaf < (n + m > 1 ? n + m : 1)) {
        info = -22;
    } else if (ldbf < (n + p > 1 ? n + p : 1)) {
        info = -24;
    } else {
        i32 ii = (p < m) ? p : m;
        i32 minpn = (p < n) ? p : n;
        i32 minmn = (m < n) ? m : n;
        i32 t1 = ii + ((3 * m - 1 > n) ? 3 * m - 1 : n);
        i32 t2 = minpn + (3 * p - 1 > (n + p > n + m ? n + p : n + m) ?
                         3 * p - 1 : (n + p > n + m ? n + p : n + m));
        i32 t3 = minmn + ((3 * m - 1 > n + m) ? 3 * m - 1 : n + m);
        i32 i_val = (t1 > t2 ? t1 : t2);
        i_val = (i_val > t3 ? i_val : t3);
        i_val = (i_val > 1 ? i_val : 1);

        if (lquery) {
            f64 svlmax_q = DZERO;
            i32 ninfz_q = 0;
            i32 ro_q = p;
            i32 sigma_q = 0;
            i32 mu_q, nu_q, nkrol_q;
            slicot_ab8nxz(n, m, p, &ro_q, &sigma_q, svlmax_q, bf, ldbf,
                         &ninfz_q, infz, kronl, &mu_q, &nu_q, &nkrol_q, tol,
                         iwork, dwork, zwork, -1);
            i32 wrkopt = (i_val > (i32)creal(zwork[0]) ? i_val : (i32)creal(zwork[0]));

            ro_q = (m - ii > 0 ? m - ii : 0);
            sigma_q = ii;
            slicot_ab8nxz(n, ii, m, &ro_q, &sigma_q, svlmax_q, af, ldaf,
                         &ninfz_q, infz, kronl, &mu_q, &nu_q, &nkrol_q, tol,
                         iwork, dwork, zwork, -1);
            wrkopt = (wrkopt > (i32)creal(zwork[0]) ? wrkopt : (i32)creal(zwork[0]));

            i32 nii = n + ii;
            SLC_ZTZRZF(&ii, &nii, af, &ldaf, zwork, zwork, &(i32){-1}, &info);
            wrkopt = (wrkopt > ii + (i32)creal(zwork[0]) ? wrkopt : ii + (i32)creal(zwork[0]));

            SLC_ZUNMRZ("Right", "Conjugate transpose", &n, &nii, &ii, &n,
                      af, &ldaf, zwork, af, &ldaf, zwork, &(i32){-1}, &info);
            wrkopt = (wrkopt > ii + (i32)creal(zwork[0]) ? wrkopt : ii + (i32)creal(zwork[0]));

            zwork[0] = (f64)wrkopt + 0.0*I;
            return 0;
        } else if (lzwork < i_val) {
            info = -29;
        }
    }

    if (info != 0) {
        i32 xinfo = -info;
        SLC_XERBLA("AB08NZ", &xinfo);
        return info;
    }

    *dinfz = 0;
    *nkrol = 0;
    *nkror = 0;

    if (n == 0) {
        i32 minmp = (m < p) ? m : p;
        if (minmp == 0) {
            *nu = 0;
            *rank = 0;
            zwork[0] = ONE;
            return 0;
        }
    }

    i32 mm = m;
    i32 nn = n;
    i32 pp = p;

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

    i32 wrkopt = 1;

    SLC_ZLACPY("Full", &nn, &mm, b, &ldb, bf, &ldbf);
    if (pp > 0) {
        SLC_ZLACPY("Full", &pp, &mm, d, &ldd, &bf[nn], &ldbf);
    }
    if (nn > 0) {
        SLC_ZLACPY("Full", &nn, &nn, a, &lda, &bf[mm * ldbf], &ldbf);
        if (pp > 0) {
            SLC_ZLACPY("Full", &pp, &nn, c, &ldc, &bf[nn + mm * ldbf], &ldbf);
        }
    }

    if (lequil && nn > 0 && pp > 0) {
        f64 maxred = DZERO;
        slicot_tb01iz('A', nn, mm, pp, &maxred, &bf[mm * ldbf], ldbf, bf,
                     ldbf, &bf[nn + mm * ldbf], ldbf, dwork);
    }

    f64 thresh = sqrt((f64)((n + p) * (n + m))) * SLC_DLAMCH("Precision");
    f64 toler = tol;
    if (toler < thresh) toler = thresh;

    i32 np_mm = nn + pp;
    i32 nm_mm = nn + mm;
    f64 svlmax = SLC_ZLANGE("Frobenius", &np_mm, &nm_mm, bf, &ldbf, dwork);

    i32 ro = pp;
    i32 sigma = 0;
    i32 ninfz = 0;
    i32 mu, local_nu;
    slicot_ab8nxz(nn, mm, pp, &ro, &sigma, svlmax, bf, ldbf, &ninfz, infz,
                 kronl, &mu, &local_nu, nkrol, toler, iwork, dwork, zwork, lzwork);
    wrkopt = (wrkopt > (i32)creal(zwork[0]) ? wrkopt : (i32)creal(zwork[0]));
    *rank = mu;

    i32 numu = local_nu + mu;
    if (numu != 0) {
        i32 mnu = mm + local_nu;
        i32 numu1 = numu;

        for (i32 i = 0; i < numu; i++) {
            for (i32 j = 0; j < mnu; j++) {
                af[mnu - 1 - j + (numu1 - 1 - i) * ldaf] = bf[i + j * ldbf];
            }
        }

        if (mu != mm) {
            pp = mm;
            nn = local_nu;
            mm = mu;

            ro = pp - mm;
            sigma = mm;
            slicot_ab8nxz(nn, mm, pp, &ro, &sigma, svlmax, af, ldaf, &ninfz,
                         infz, kronr, &mu, &local_nu, nkror, toler, iwork,
                         dwork, zwork, lzwork);
            wrkopt = (wrkopt > (i32)creal(zwork[0]) ? wrkopt : (i32)creal(zwork[0]));
        }

        if (local_nu != 0) {
            for (i32 j = 0; j < mu; j++) {
                for (i32 i = 0; i < local_nu; i++) {
                    bf[i + j * ldbf] = ZERO;
                }
            }
            for (i32 j = 0; j < local_nu; j++) {
                for (i32 i = 0; i < local_nu; i++) {
                    bf[i + (mu + j) * ldbf] = (i == j) ? ONE : ZERO;
                }
            }

            if (*rank != 0) {
                i32 nu1 = local_nu;
                i32 i1 = local_nu + mu;

                i32 lwork = lzwork - mu;
                SLC_ZTZRZF(&mu, &i1, &af[nu1], &ldaf, zwork, &zwork[mu],
                          &lwork, &info);
                wrkopt = (wrkopt > (i32)creal(zwork[mu]) + mu ? wrkopt : (i32)creal(zwork[mu]) + mu);

                SLC_ZUNMRZ("Right", "Conjugate transpose", &nu1, &i1, &mu, &nu1,
                          &af[nu1], &ldaf, zwork, af, &ldaf, &zwork[mu],
                          &lwork, &info);
                wrkopt = (wrkopt > (i32)creal(zwork[mu]) + mu ? wrkopt : (i32)creal(zwork[mu]) + mu);

                SLC_ZUNMRZ("Right", "Conjugate transpose", &nu1, &i1, &mu, &nu1,
                          &af[nu1], &ldaf, zwork, bf, &ldbf, &zwork[mu],
                          &lwork, &info);
            }

            SLC_ZLACPY("Full", &local_nu, &local_nu, &af[mu * ldaf], &ldaf, af, &ldaf);
            if (*rank != 0) {
                SLC_ZLACPY("Full", &local_nu, &local_nu, &bf[mu * ldbf], &ldbf, bf, &ldbf);
            }
        }
    }

    *nu = local_nu;

    if (*nkror > 0) {
        i32 j = 0;
        for (i32 i = 0; i <= n; i++) {
            for (i32 ii = j; ii < j + kronr[i]; ii++) {
                if (ii < n + m + 1) {
                    iwork[ii] = i;
                }
            }
            j = j + kronr[i];
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
            for (i32 ii = j; ii < j + kronl[i]; ii++) {
                if (ii < n + p + 1) {
                    iwork[ii] = i;
                }
            }
            j = j + kronl[i];
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

    zwork[0] = (f64)wrkopt + 0.0*I;
    return 0;
}
