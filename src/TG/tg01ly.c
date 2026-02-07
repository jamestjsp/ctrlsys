/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * TG01LY - Finite-infinite decomposition of structured descriptor system
 *
 * Reduces a regular pole pencil A-lambda*E of descriptor system (A-lambda*E,B,C)
 * with A and E in structured form to finite-infinite separated form.
 */

#include "slicot/tg.h"
#include "slicot/mb03.h"
#include "slicot_blas.h"

#include <math.h>
#include <stdlib.h>

void tg01ly(
    const bool compq, const bool compz,
    const i32 n, const i32 m, const i32 p,
    const i32 ranke, const i32 rnka22,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* nf, i32* niblck, i32* iblck,
    const f64 tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
)
{
    const f64 one = 1.0;
    const f64 zero = 0.0;
    i32 int1 = 1;

    bool first, lquery;
    i32 i, i0, i1, icol, ipiv, irow, itau, j, jwork1, k;
    i32 mm1, n1, nd, nr, rank, ro, ro1, sigma, wrkopt, minwrk;
    f64 co, rcond, si, svlmax, t, toldef;
    f64 dum[1], sval[3];

    *info = 0;

    if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (p < 0) {
        *info = -5;
    } else if (ranke < 0 || ranke > n) {
        *info = -6;
    } else if (rnka22 < 0 || rnka22 + ranke > n) {
        *info = -7;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (lde < (n > 1 ? n : 1)) {
        *info = -11;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -13;
    } else if (ldc < (p > 1 ? p : 1)) {
        *info = -15;
    } else if ((compq && ldq < n) || ldq < 1) {
        *info = -17;
    } else if ((compz && ldz < n) || ldz < 1) {
        *info = -19;
    } else if (tol >= one) {
        *info = -23;
    } else {
        lquery = (ldwork == -1);

        nd = n - ranke;
        ro1 = nd;
        if (ranke == n) {
            minwrk = 1;
        } else {
            i32 max_nm = n > m ? n : m;
            minwrk = 4 * nd - 1;
            i32 alt = ro1 + max_nm;
            if (alt > minwrk) minwrk = alt;
        }
        if (lquery) {
            i32 lwork_temp = -1;
            SLC_DORMQR("L", "T", &ro1, &ranke, &nd,
                       a, &lda, dwork, a, &lda, dwork, &lwork_temp, info);
            wrkopt = minwrk;
            i32 opt1 = (i32)dwork[0] + nd;
            if (opt1 > wrkopt) wrkopt = opt1;

            SLC_DORMQR("L", "T", &ro1, &m, &nd, a, &lda,
                       dwork, b, &ldb, dwork, &lwork_temp, info);
            opt1 = (i32)dwork[0] + nd;
            if (opt1 > wrkopt) wrkopt = opt1;

            if (compq) {
                SLC_DORMQR("R", "N", &n, &ro1, &nd, a,
                           &lda, dwork, q, &ldq, dwork, &lwork_temp, info);
                opt1 = (i32)dwork[0] + nd;
                if (opt1 > wrkopt) wrkopt = opt1;
            }
        } else if (ldwork < minwrk) {
            *info = -26;
        }
    }

    if (*info != 0) {
        i32 neg_info = -(*info);
        SLC_XERBLA("TG01LY", &neg_info);
        return;
    }

    if (lquery) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    *nf = ranke;
    *niblck = 0;

    if (ranke == n) {
        dwork[0] = one;
        return;
    }

    if (ranke == 0 && rnka22 == 0) {
        *info = 1;
        return;
    }

    wrkopt = minwrk;
    if (rnka22 == nd)
        goto label_110;

    toldef = tol;
    if (toldef <= zero) {
        toldef = (f64)(n * n) * SLC_DLAMCH("P");
    }

    mm1 = nd + 1;
    SLC_DLACPY("F", &n, &nd, &a[(ranke) * lda], &lda, &e[(ranke) * lde], &lde);

    if (ranke <= nd) {
        SLC_DLACPY("F", &n, &ranke, a, &lda, &a[(mm1 - 1) * lda], &lda);
    } else {
        k = ranke % nd;
        for (i = n - 2 * nd; i >= k; i -= nd) {
            if (i < 0) break;
            SLC_DLACPY("F", &n, &nd, &a[i * lda], &lda, &a[(i + nd) * lda], &lda);
        }
        if (k != 0)
            SLC_DLACPY("F", &n, &k, a, &lda, &a[(mm1 - 1) * lda], &lda);
    }
    SLC_DLACPY("F", &n, &nd, &e[(ranke) * lde], &lde, a, &lda);

    if (compz) {
        SLC_DLACPY("F", &n, &nd, &z[(ranke) * ldz], &ldz, &e[(ranke) * lde], &lde);
        if (ranke <= nd) {
            SLC_DLACPY("F", &n, &ranke, z, &ldz, &z[(mm1 - 1) * ldz], &ldz);
        } else {
            for (i = n - 2 * nd; i >= k; i -= nd) {
                if (i < 0) break;
                SLC_DLACPY("F", &n, &nd, &z[i * ldz], &ldz, &z[(i + nd) * ldz], &ldz);
            }
            if (k != 0)
                SLC_DLACPY("F", &n, &k, z, &ldz, &z[(mm1 - 1) * ldz], &ldz);
        }
        SLC_DLACPY("F", &n, &nd, &e[(ranke) * lde], &lde, z, &ldz);
    }

    if (p <= n) {
        SLC_DLACPY("F", &p, &nd, &c[(ranke) * ldc], &ldc, &e[(ranke) * lde], &lde);
        if (ranke <= nd) {
            SLC_DLACPY("F", &p, &ranke, c, &ldc, &c[(mm1 - 1) * ldc], &ldc);
        } else {
            for (i = n - 2 * nd; i >= k; i -= nd) {
                if (i < 0) break;
                SLC_DLACPY("F", &p, &nd, &c[i * ldc], &ldc, &c[(i + nd) * ldc], &ldc);
            }
            if (k != 0)
                SLC_DLACPY("F", &p, &k, c, &ldc, &c[(mm1 - 1) * ldc], &ldc);
        }
        SLC_DLACPY("F", &p, &nd, &e[(ranke) * lde], &lde, c, &ldc);
    } else {
        for (i = 0; i < p; i++) {
            SLC_DCOPY(&nd, &c[i + (ranke) * ldc], &ldc, dwork, &int1);
            SLC_DCOPY(&ranke, &c[i], &ldc, &dwork[nd], &int1);
            SLC_DCOPY(&n, dwork, &int1, &c[i], &ldc);
        }
    }

    if (ranke <= nd) {
        SLC_DLACPY("F", &n, &ranke, e, &lde, &e[(mm1 - 1) * lde], &lde);
    } else {
        for (i = n - 2 * nd; i >= k; i -= nd) {
            if (i < 0) break;
            SLC_DLACPY("F", &n, &nd, &e[i * lde], &lde, &e[(i + nd) * lde], &lde);
        }
        if (k != 0)
            SLC_DLACPY("F", &n, &k, e, &lde, &e[(mm1 - 1) * lde], &lde);
    }
    SLC_DLASET("F", &n, &nd, &zero, &zero, e, &lde);

    sval[0] = SLC_DLANGE("F", &ranke, &rnka22, a, &lda, dwork);
    sval[1] = SLC_DLANTR("F", "U", "N", &rnka22, &rnka22, &a[ranke], &lda, dwork);
    i32 temp = ranke + rnka22;
    i32 temp2 = nd - rnka22;
    sval[2] = SLC_DLANGE("F", &temp, &temp2, &a[(rnka22) * lda], &lda, dwork);

    f64 norm1 = SLC_DNRM2(&(i32){3}, sval, &int1);
    f64 norm2 = SLC_DLANGE("F", &n, &ranke, &a[(nd) * lda], &lda, dwork);
    svlmax = SLC_DLAPY2(&norm1, &norm2) / (f64)n;

    ro = nd;
    sigma = 0;
    first = true;
    itau = 0;
    jwork1 = itau + nd;
    dum[0] = zero;

label_60:
    if (first) {
        ro1 = nd - rnka22;
    } else {
        ro1 = ro;

        irow = *nf;
        for (icol = 0; icol < sigma; icol++) {
            i32 ro_plus1 = ro + 1;
            SLC_DLARFG(&ro_plus1, &a[irow + icol * lda], &a[irow + 1 + icol * lda], &int1, &t);

            i32 ncols = n - icol - 1;
            SLC_DLATZM("L", &ro_plus1, &ncols, &a[irow + 1 + icol * lda], &int1, &t,
                       &a[irow + (icol + 1) * lda], &a[irow + 1 + (icol + 1) * lda], &lda, dwork);

            SLC_DLATZM("L", &ro_plus1, &ranke, &a[irow + 1 + icol * lda], &int1, &t,
                       &e[irow + (nd) * lde], &e[irow + 1 + (nd) * lde], &lde, dwork);

            if (compq) {
                SLC_DLATZM("R", &n, &ro_plus1, &a[irow + 1 + icol * lda], &int1, &t,
                           &q[(irow) * ldq], &q[(irow + 1) * ldq], &ldq, dwork);
            }

            SLC_DLATZM("L", &ro_plus1, &m, &a[irow + 1 + icol * lda], &int1, &t,
                       &b[irow], &b[irow + 1], &ldb, dwork);

            i32 nd_minus_icol_minus1 = nd - icol - 1;
            for (i32 ii = 0; ii < nd_minus_icol_minus1; ii++) {
                a[irow + 1 + ii + icol * lda] = zero;
            }
            irow++;
        }

        i32 irow_c = (*nf + sigma < n) ? (*nf + sigma) : (n - 1);
        i32 icol_c = (sigma < nd) ? sigma : (nd - 1);
        i32 m_qr = ro1;
        i32 n_qr = nd - sigma;

        i32 mb03oy_info;
        mb03oy(m_qr, n_qr, &a[irow_c + icol_c * lda], lda, toldef,
               svlmax, &rank, sval, iwork, &dwork[itau], &dwork[jwork1], &mb03oy_info);

        i32 l_forwrd = 1;
        i32 nf_plus_sigma = *nf + sigma;
        SLC_DLAPMT(&l_forwrd, &nf_plus_sigma, &n_qr, &a[icol_c * lda], &lda, iwork);
        SLC_DLAPMT(&l_forwrd, &p, &n_qr, &c[icol_c * ldc], &ldc, iwork);
        if (compz) {
            SLC_DLAPMT(&l_forwrd, &n, &n_qr, &z[icol_c * ldz], &ldz, iwork);
        }

        if (rank > 0) {
            i32 lwork_avail = ldwork - jwork1;
            SLC_DORMQR("L", "T", &ro1, &ranke, &rank,
                       &a[irow_c + icol_c * lda], &lda, &dwork[itau],
                       &a[irow_c + (mm1 - 1) * lda], &lda,
                       &dwork[jwork1], &lwork_avail, info);
            i32 opt1 = (i32)dwork[jwork1] + jwork1;
            if (opt1 > wrkopt) wrkopt = opt1;

            SLC_DORMQR("L", "T", &ro1, &m, &rank,
                       &a[irow_c + icol_c * lda], &lda, &dwork[itau],
                       &b[irow_c], &ldb, &dwork[jwork1], &lwork_avail, info);
            opt1 = (i32)dwork[jwork1] + jwork1;
            if (opt1 > wrkopt) wrkopt = opt1;

            SLC_DORMQR("L", "T", &ro1, &ranke, &rank,
                       &a[irow_c + icol_c * lda], &lda, &dwork[itau],
                       &e[irow_c + (mm1 - 1) * lde], &lde,
                       &dwork[jwork1], &lwork_avail, info);
            opt1 = (i32)dwork[jwork1] + jwork1;
            if (opt1 > wrkopt) wrkopt = opt1;

            if (compq) {
                SLC_DORMQR("R", "N", &n, &ro1, &rank,
                           &a[irow_c + icol_c * lda], &lda, &dwork[itau],
                           &q[(irow_c) * ldq], &ldq, &dwork[jwork1], &lwork_avail, info);
                opt1 = (i32)dwork[jwork1] + jwork1;
                if (opt1 > wrkopt) wrkopt = opt1;
            }

            i32 ro1_m1 = ro1 - 1;
            i32 min_ro1_m1_rank = (ro1_m1 < rank) ? ro1_m1 : rank;
            i32 irow_lower = (irow_c + 1 < n) ? (irow_c + 1) : (n - 1);
            if (min_ro1_m1_rank > 0 && irow_lower < n) {
                SLC_DLASET("L", &min_ro1_m1_rank, &min_ro1_m1_rank, &zero, &zero,
                           &a[irow_lower + icol_c * lda], &lda);
            }
            ro1 = ro1 - rank;
        }
    }

    if (ro1 > 0) {
        sigma = nd - ro1;
        (*niblck)++;

        ipiv = *nf + nd;
        n1 = *nf;
        for (i = 0; i < ro1; i++) {
            ipiv--;
            n1--;

            for (k = 0; k < n1; k++) {
                j = nd + k;

                t = a[(ipiv) + (j + 1) * lda];
                SLC_DLARTG(&t, &a[(ipiv) + j * lda], &co, &si, &a[(ipiv) + (j + 1) * lda]);
                a[(ipiv) + j * lda] = zero;

                i32 ipiv_c = ipiv;
                SLC_DROT(&ipiv_c, &a[(j + 1) * lda], &int1, &a[j * lda], &int1, &co, &si);

                i32 k_plus1 = k + 1;
                SLC_DROT(&k_plus1, &e[(j + 1) * lde], &int1, &e[j * lde], &int1, &co, &si);

                SLC_DROT(&p, &c[(j + 1) * ldc], &int1, &c[j * ldc], &int1, &co, &si);

                if (compz) {
                    SLC_DROT(&n, &z[(j + 1) * ldz], &int1, &z[j * ldz], &int1, &co, &si);
                }

                t = e[k + j * lde];
                SLC_DLARTG(&t, &e[k + 1 + j * lde], &co, &si, &e[k + j * lde]);
                e[k + 1 + j * lde] = zero;

                i32 n_minus_j = n - j - 1;
                SLC_DROT(&n_minus_j, &e[k + (j + 1) * lde], &lde, &e[k + 1 + (j + 1) * lde], &lde, &co, &si);

                SLC_DROT(&n, &a[k], &lda, &a[k + 1], &lda, &co, &si);

                SLC_DROT(&m, &b[k], &ldb, &b[k + 1], &ldb, &co, &si);

                if (compq) {
                    SLC_DROT(&n, &q[k * ldq], &int1, &q[(k + 1) * ldq], &int1, &co, &si);
                }
            }
        }

        f64 norm_check = SLC_DLANTR("F", "U", "N", &ro1, &ro1, &a[(ipiv) + (ipiv) * lda], &lda, dwork);
        if (norm_check <= toldef * svlmax) {
            *info = 1;
        } else {
            SLC_DTRCON("1", "U", "N", &ro1, &a[(ipiv) + (ipiv) * lda], &lda, &rcond, dwork, iwork, info);
            if (rcond <= toldef)
                *info = 1;
        }

        if (*info != 0)
            return;

        *nf = *nf - ro1;

        iblck[*niblck - 1] = ro1;
        ro = ro1;
        first = false;

        goto label_60;
    }

    if (*nf > 0) {
        nr = *nf + nd;
        dum[0] = zero;
        SLC_DLACPY("F", &nr, &nd, a, &lda, e, &lde);
        SLC_DLACPY("F", &nr, nf, &a[(nd) * lda], &lda, a, &lda);
        SLC_DLACPY("F", &nr, &nd, e, &lde, &a[(*nf) * lda], &lda);

        if (compz) {
            SLC_DLACPY("F", &n, &nd, z, &ldz, e, &lde);
            SLC_DLACPY("F", &n, nf, &z[(nd) * ldz], &ldz, z, &ldz);
            SLC_DLACPY("F", &n, &nd, e, &lde, &z[(*nf) * ldz], &ldz);
        }

        if (p <= n) {
            SLC_DLACPY("F", &p, &nd, c, &ldc, e, &lde);
            SLC_DLACPY("F", &p, nf, &c[(nd) * ldc], &ldc, c, &ldc);
            SLC_DLACPY("F", &p, &nd, e, &lde, &c[(*nf) * ldc], &ldc);
        } else {
            for (i = 0; i < p; i++) {
                SLC_DCOPY(&nd, &c[i], &ldc, dwork, &int1);
                SLC_DCOPY(nf, &c[i + (nd) * ldc], &ldc, &c[i], &ldc);
                SLC_DCOPY(&nd, dwork, &int1, &c[i + (*nf) * ldc], &ldc);
            }
        }

        SLC_DLACPY("F", &n, nf, &e[(nd) * lde], &lde, e, &lde);
        SLC_DLASET("F", &n, &nd, &zero, &zero, &e[(*nf) * lde], &lde);

        i32 n_minus_nf_minus_nd = n - *nf - nd;
        i32 nf_plus_nd = *nf + nd;
        if (n_minus_nf_minus_nd > 0) {
            SLC_DLASET("F", &n_minus_nf_minus_nd, &nf_plus_nd, &zero, &zero,
                       &a[nf_plus_nd], &lda);
        }
    }

label_110:
    i1 = *nf + nd;
    for (i0 = nd - 1; i0 >= 0; i0--) {
        i1--;

        k = 0;
        for (j = 0; j < *nf; j++) {
            t = a[(i1) + (i1) * lda];
            SLC_DLARTG(&t, &a[(i1) + j * lda], &co, &si, &a[(i1) + (i1) * lda]);
            a[(i1) + j * lda] = zero;

            i32 i1_c = i1;
            SLC_DROT(&i1_c, &a[(i1) * lda], &int1, &a[j * lda], &int1, &co, &si);

            i32 k_plus1 = k + 1;
            SLC_DROT(&k_plus1, &e[(i1) * lde], &int1, &e[j * lde], &int1, &co, &si);

            SLC_DROT(&p, &c[(i1) * ldc], &int1, &c[j * ldc], &int1, &co, &si);

            if (compz) {
                SLC_DROT(&n, &z[(i1) * ldz], &int1, &z[j * ldz], &int1, &co, &si);
            }
            k++;
        }
    }

    dwork[0] = (f64)wrkopt;
}
