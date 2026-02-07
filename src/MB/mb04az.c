/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * MB04AZ - Eigenvalues of complex skew-Hamiltonian/Hamiltonian pencil
 *
 * Computes eigenvalues of aS - bH where:
 *   S = J*Z^H*J^T*Z
 *   H = [[B, F], [G, -B^H]]
 *   J = [[0, I], [-I, 0]]
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>
#include <ctype.h>
#include <math.h>
#include <stdlib.h>

void mb04az(const char *job, const char *compq, const char *compu,
            i32 n, c128 *z, i32 ldz, c128 *b, i32 ldb, c128 *fg, i32 ldfg,
            c128 *d, i32 ldd, c128 *c, i32 ldc, c128 *q, i32 ldq,
            c128 *u, i32 ldu, f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork,
            c128 *zwork, i32 lzwork, bool *bwork, i32 *info) {

    const f64 ZERO = 0.0;
    const f64 FOUR = 4.0;
    const f64 MONE = -1.0;
    const c128 CZERO = 0.0 + 0.0*I;
    const c128 CONE = 1.0 + 0.0*I;
    const c128 MCIMAG = 0.0 - 1.0*I;

    char job_upper = (char)toupper((unsigned char)job[0]);
    char compq_upper = (char)toupper((unsigned char)compq[0]);
    char compu_upper = (char)toupper((unsigned char)compu[0]);

    bool ltri = (job_upper == 'T');
    bool lcmpq = (compq_upper == 'C');
    bool lcmpu = (compu_upper == 'C');

    i32 m = n / 2;
    i32 nn = n * n;
    i32 n2 = 2 * n;

    i32 k_dim;
    if (ltri) {
        k_dim = (1 > n) ? 1 : n;
    } else {
        k_dim = (1 > m) ? 1 : m;
    }

    i32 mindw, minzw, mindb;
    if (n == 0) {
        mindw = 4;
        minzw = 1;
    } else {
        if (ltri) {
            if (lcmpq) {
                minzw = 4 * n2 + 28;
            } else {
                minzw = 3 * n2 + 28;
            }
        } else {
            minzw = 1;
        }
        i32 i_coef, j_coef;
        if (lcmpu) {
            i_coef = 12;
            j_coef = 18;
        } else {
            i_coef = 10;
            if (lcmpq) {
                j_coef = 16;
            } else {
                j_coef = 13;
            }
        }
        mindb = i_coef * nn + n;
        i32 max_3n2_27 = (3 * n2 > 27) ? 3 * n2 : 27;
        mindw = j_coef * nn + n + max_3n2_27;
    }

    bool lquery = (ldwork == -1 || lzwork == -1);

    *info = 0;
    if (!(job_upper == 'E' || ltri)) {
        *info = -1;
    } else if (!(compq_upper == 'N' || lcmpq)) {
        *info = -2;
    } else if (!(compu_upper == 'N' || lcmpu)) {
        *info = -3;
    } else if (n < 0 || (n % 2) != 0) {
        *info = -4;
    } else if (ldz < ((1 > n) ? 1 : n)) {
        *info = -6;
    } else if (ldb < k_dim) {
        *info = -8;
    } else if (ldfg < k_dim) {
        *info = -10;
    } else if (ldd < 1 || (ltri && ldd < n)) {
        *info = -12;
    } else if (ldc < 1 || (ltri && ldc < n)) {
        *info = -14;
    } else if (ldq < 1 || (lcmpq && ldq < n2)) {
        *info = -16;
    } else if (ldu < 1 || (lcmpu && ldu < n)) {
        *info = -18;
    } else if (liwork < n2 + 9) {
        *info = -23;
    } else if (!lquery) {
        if (ldwork < mindw) {
            dwork[0] = (f64)mindw;
            *info = -25;
        } else if (lzwork < minzw) {
            zwork[0] = (c128)minzw;
            *info = -27;
        }
    }

    if (*info != 0) {
        return;
    }

    i32 optdw, optzw;
    i32 nb = 2;

    if (n > 0) {
        if (ltri) {
            i32 lwork_query = -1;
            c128 work_query;
            i32 info_query;
            SLC_ZGEQRF(&n, &n, z, &ldz, zwork, &work_query, &lwork_query, &info_query);
            i32 qrf_work = (i32)creal(work_query);
            nb = (qrf_work / n > 2) ? qrf_work / n : 2;
        }

        const char *cmpq = lcmpq ? "Initialize" : "N";
        const char *cmpu = lcmpu ? "Initialize" : "N";

        if (lquery) {
            i32 lwork_ed = -1;
            f64 ed_opt;
            i32 info_ed;
            mb04ed(job, cmpq, cmpu, n2, dwork, n2, dwork, n,
                   dwork, n, dwork, n2, dwork, n, dwork, n,
                   alphai, alphar, beta, iwork, liwork, &ed_opt,
                   lwork_ed, &info_ed);
            optdw = (mindw > mindb + (i32)ed_opt) ? mindw : mindb + (i32)ed_opt;

            if (ltri) {
                i32 qrf_opt = (i32)creal(zwork[0]);
                optzw = (minzw > qrf_opt) ? minzw : qrf_opt;
            } else {
                optzw = minzw;
            }
            dwork[0] = (f64)optdw;
            zwork[0] = (c128)optzw;
            return;
        } else {
            optzw = minzw;
        }
    }

    if (n == 0) {
        iwork[0] = 0;
        dwork[0] = FOUR;
        dwork[1] = ZERO;
        dwork[2] = ZERO;
        dwork[3] = ZERO;
        zwork[0] = CONE;
        return;
    }

    f64 eps = SLC_DLAMCH("Precision");

    i32 iq_idx = 0;
    i32 iu_idx, iz11_idx;
    if (lcmpu) {
        iu_idx = iq_idx + n2 * n2;
        iz11_idx = iu_idx + n2 * n;
    } else {
        iu_idx = 0;
        iz11_idx = iq_idx + n2 * n2;
    }
    i32 ib_idx = iz11_idx + n2 * n2;
    i32 ifg_idx = ib_idx + nn;
    i32 iwrk_idx = ifg_idx + nn + n;

    i32 int1 = 1;
    i32 iw, is, iw1, i1;
    iw = iz11_idx;
    is = iw + n2 * m;

    for (i32 jj = 0; jj < n; jj++) {
        iw1 = iw;
        for (i32 ii = 0; ii < m; ii++) {
            dwork[iw] = creal(z[ii + jj * ldz]);
            iw++;
        }
        for (i32 ii = 0; ii < m; ii++) {
            dwork[iw] = cimag(z[ii + jj * ldz]);
            dwork[is] = -dwork[iw];
            iw++;
            is++;
        }
        SLC_DCOPY(&m, &dwork[iw1], &int1, &dwork[is], &int1);
        iw1 = iw;
        is += m;

        for (i32 ii = m; ii < n; ii++) {
            dwork[iw] = creal(z[ii + jj * ldz]);
            iw++;
        }
        for (i32 ii = m; ii < n; ii++) {
            dwork[iw] = cimag(z[ii + jj * ldz]);
            dwork[is] = -dwork[iw];
            iw++;
            is++;
        }
        SLC_DCOPY(&m, &dwork[iw1], &int1, &dwork[is], &int1);
        iw1 = iw;
        is += m;

        if (((jj + 1) % m) == 0) {
            iw += n2 * m;
            is += n2 * m;
        }
    }

    iw = ib_idx;
    is = iw + n * m;
    for (i32 jj = 0; jj < m; jj++) {
        iw1 = iw;
        for (i32 ii = 0; ii < m; ii++) {
            dwork[iw] = -cimag(b[ii + jj * ldb]);
            iw++;
        }
        for (i32 ii = 0; ii < m; ii++) {
            dwork[iw] = creal(b[ii + jj * ldb]);
            dwork[is] = -dwork[iw];
            iw++;
            is++;
        }
        SLC_DCOPY(&m, &dwork[iw1], &int1, &dwork[is], &int1);
        is += m;
    }

    iw = ifg_idx;
    for (i32 jj = 0; jj < m + 1; jj++) {
        for (i32 ii = 0; ii < m; ii++) {
            dwork[iw] = -cimag(fg[ii + jj * ldfg]);
            iw++;
        }
        iw += jj;
        is = iw;
        for (i32 ii = jj; ii < m; ii++) {
            dwork[iw] = creal(fg[ii + jj * ldfg]);
            dwork[is] = dwork[iw];
            iw++;
            is += n;
        }
    }

    iw1 = iw;
    i1 = iw;
    for (i32 jj = 1; jj < m + 1; jj++) {
        is = i1;
        i1++;
        for (i32 ii = 0; ii < jj; ii++) {
            dwork[iw] = -creal(fg[ii + (jj) * ldfg]);
            dwork[is] = dwork[iw];
            iw++;
            is += n;
        }
        iw += n - jj;
    }

    i32 mp1 = m + 1;
    SLC_DLACPY("Full", &m, &mp1, &dwork[ifg_idx], &n, &dwork[iw1 - m], &n);

    const char *cmpq = lcmpq ? "Initialize" : "N";
    const char *cmpu = lcmpu ? "Initialize" : "N";
    i32 ldwork_ed = ldwork - iwrk_idx;

    mb04ed(job, cmpq, cmpu, n2, &dwork[iz11_idx], n2, &dwork[ib_idx],
           n, &dwork[ifg_idx], n, &dwork[iq_idx], n2, &dwork[iu_idx], n,
           &dwork[iu_idx + nn], n, alphai, alphar, beta, iwork,
           liwork, &dwork[iwrk_idx], ldwork_ed, info);

    if (*info > 0 && *info < 3) {
        return;
    }
    optdw = (mindw > mindb + (i32)dwork[iwrk_idx]) ? mindw : mindb + (i32)dwork[iwrk_idx];

    SLC_DSCAL(&n, &MONE, alphai, &int1);

    if (lcmpq) {
        iw = iq_idx;
        for (i32 jj = 0; jj < n2; jj++) {
            for (i32 ii = 0; ii < n2; ii++) {
                q[ii + jj * ldq] = dwork[iw];
                iw++;
            }
        }
    }

    if (lcmpu) {
        iw = iu_idx;
        for (i32 jj = 0; jj < n2; jj++) {
            for (i32 ii = 0; ii < n; ii++) {
                u[ii + jj * ldu] = dwork[iw];
                iw++;
            }
        }
    }

    if (!ltri) {
        zwork[0] = (c128)optzw;
        dwork[0] = (f64)optdw;
        i32 iw1_val = iwork[0];
        i32 iw_val = iwork[2 * iw1_val + 3];
        i32 k_copy = 3 * (n - 2 * iw_val + 1) + 12 * iw_val;
        SLC_DCOPY(&k_copy, &dwork[iwrk_idx + 1], &int1, &dwork[1], &int1);
        return;
    }

    iw = iz11_idx;
    for (i32 jj = 0; jj < n; jj++) {
        for (i32 ii = 0; ii <= jj; ii++) {
            z[ii + jj * ldz] = dwork[iw];
            iw++;
        }
        iw += n2 - jj - 1;
    }

    iw = iz11_idx + n2 * n;
    for (i32 jj = 0; jj < n; jj++) {
        for (i32 ii = 0; ii < n; ii++) {
            d[ii + jj * ldd] = dwork[iw];
            iw++;
        }
        iw += jj;
        for (i32 ii = jj; ii < n; ii++) {
            c[ii + jj * ldc] = dwork[iw];
            iw++;
        }
    }

    iw = ib_idx;
    for (i32 jj = 0; jj < n; jj++) {
        i32 min_jp2_n = (jj + 2 < n) ? jj + 2 : n;
        for (i32 ii = 0; ii < min_jp2_n; ii++) {
            b[ii + jj * ldb] = dwork[iw];
            iw++;
        }
        iw += n - jj - 2;
        if (iw < ib_idx) iw = ib_idx + nn - 1;
    }

    iw = ifg_idx + n;
    for (i32 jj = 0; jj < n; jj++) {
        for (i32 ii = 0; ii < jj; ii++) {
            fg[ii + jj * ldfg] = dwork[iw];
            iw++;
        }
        fg[jj + jj * ldfg] = CZERO;
        iw += n - jj;
    }

    i32 is_save = iwrk_idx;
    i32 iq2_idx = 0;
    i32 iq_zw_idx = iq2_idx + 4;
    i32 iu_zw_idx = iq_zw_idx + 4;
    i32 ib_zw_idx = iu_zw_idx + 4;
    i32 iz11_zw_idx = ib_zw_idx + 4;
    i32 iz22_zw_idx = iz11_zw_idx + 4;
    i32 iev_idx = iz22_zw_idx + 4;
    i32 iqb_idx_start = iev_idx + 4;
    i32 iub_idx_start = iqb_idx_start + 4 * m;
    i32 iwrk_zw_idx = iub_idx_start + 4 * m;

    i32 iworkz[5];
    f64 dworkz[2];

    iworkz[0] = 1;
    iworkz[1] = -1;
    iworkz[2] = -1;

    i32 j = 0;
    i32 j1 = 0;
    i32 j2 = (n < nb) ? n : nb;
    i32 iqb_idx = iqb_idx_start;
    i32 iub_idx = iub_idx_start;

    while (j < n - 1) {
        f64 nrmb = cabs(b[j + j * ldb]) + cabs(b[(j + 1) + (j + 1) * ldb]);
        if (cabs(b[(j + 1) + j * ldb]) > nrmb * eps) {

            i32 nc = (j2 - j - 2 > 0) ? j2 - j - 2 : 0;
            i32 jp2 = (j + 2 < n) ? j + 2 : n;
            i32 nj1 = (n - j - 2 > 0) ? n - j - 2 : 1;

            i32 two = 2;
            SLC_ZLACPY("Full", &two, &two, &b[j + j * ldb], &ldb, &zwork[ib_zw_idx], &two);
            SLC_ZLACPY("Upper", &two, &two, &z[j + j * ldz], &ldz, &zwork[iz11_zw_idx], &two);
            zwork[iz11_zw_idx + 1] = CZERO;
            zwork[iz22_zw_idx] = conj(c[j + j * ldc]);
            zwork[iz22_zw_idx + 1] = CZERO;
            zwork[iz22_zw_idx + 2] = conj(c[(j + 1) + j * ldc]);
            zwork[iz22_zw_idx + 3] = conj(c[(j + 1) + (j + 1) * ldc]);

            i32 mb03bz_k = 3;
            i32 mb03bz_n = 2;
            i32 mb03bz_ilo = 1;
            i32 mb03bz_ihi = 2;
            i32 lzwork_mb03bz = lzwork - iwrk_zw_idx;
            i32 mb03bz_info;
            mb03bz("S", "I", mb03bz_k, mb03bz_n, mb03bz_ilo, mb03bz_ihi, iworkz,
                   &zwork[ib_zw_idx], two, two, &zwork[iq2_idx], two, two,
                   &zwork[iev_idx], &zwork[iev_idx + 2], &iworkz[3],
                   dworkz, 2, &zwork[iwrk_zw_idx], lzwork_mb03bz, &mb03bz_info);

            if (mb03bz_info > 0) {
                *info = 2;
                return;
            }

            i32 jm1_dim = (j > 0) ? j : 1;
            if (j > 0) {
                SLC_ZGEMM("N", "N", &j, &two, &two, &CONE, &z[0 + j * ldz], &ldz,
                          &zwork[iq_zw_idx], &two, &CZERO, &zwork[iwrk_zw_idx], &jm1_dim);
                SLC_ZLACPY("Full", &j, &two, &zwork[iwrk_zw_idx], &jm1_dim, &z[0 + j * ldz], &ldz);
            }
            SLC_ZLACPY("Upper", &two, &two, &zwork[iz11_zw_idx], &two, &z[j + j * ldz], &ldz);
            z[(j + 1) + j * ldz] = CZERO;

            if (nc > 0 && jp2 < n) {
                i32 nc_actual = n - jp2;
                SLC_ZGEMM("C", "N", &two, &nc_actual, &two, &CONE, &zwork[iu_zw_idx], &two,
                          &z[j + jp2 * ldz], &ldz, &CZERO, &zwork[iwrk_zw_idx], &two);
                SLC_ZLACPY("Full", &two, &nc_actual, &zwork[iwrk_zw_idx], &two, &z[j + jp2 * ldz], &ldz);
            }

            SLC_ZGEMM("N", "N", &n, &two, &two, &CONE, &d[0 + j * ldd], &ldd,
                      &zwork[iq2_idx], &two, &CZERO, &zwork[iwrk_zw_idx], &n);
            SLC_ZLACPY("Full", &n, &two, &zwork[iwrk_zw_idx], &n, &d[0 + j * ldd], &ldd);

            c[j + j * ldc] = conj(zwork[iz22_zw_idx]);
            c[(j + 1) + j * ldc] = conj(zwork[iz22_zw_idx + 2]);
            c[j + (j + 1) * ldc] = CZERO;
            c[(j + 1) + (j + 1) * ldc] = conj(zwork[iz22_zw_idx + 3]);

            if (n - j - 2 > 0) {
                i32 rows = n - j - 2;
                SLC_ZGEMM("N", "N", &rows, &two, &two, &CONE, &c[jp2 + j * ldc], &ldc,
                          &zwork[iq2_idx], &two, &CZERO, &zwork[iwrk_zw_idx], &nj1);
                SLC_ZLACPY("Full", &rows, &two, &zwork[iwrk_zw_idx], &nj1, &c[jp2 + j * ldc], &ldc);
            }

            if (j > 0) {
                SLC_ZGEMM("N", "N", &j, &two, &two, &CONE, &b[0 + j * ldb], &ldb,
                          &zwork[iq_zw_idx], &two, &CZERO, &zwork[iwrk_zw_idx], &jm1_dim);
                SLC_ZLACPY("Full", &j, &two, &zwork[iwrk_zw_idx], &jm1_dim, &b[0 + j * ldb], &ldb);
            }
            SLC_ZLACPY("Upper", &two, &two, &zwork[ib_zw_idx], &two, &b[j + j * ldb], &ldb);
            b[(j + 1) + j * ldb] = CZERO;

            if (nc > 0 && jp2 < n) {
                i32 nc_actual = n - jp2;
                SLC_ZGEMM("C", "N", &two, &nc_actual, &two, &CONE, &zwork[iq2_idx], &two,
                          &b[j + jp2 * ldb], &ldb, &CZERO, &zwork[iwrk_zw_idx], &two);
                SLC_ZLACPY("Full", &two, &nc_actual, &zwork[iwrk_zw_idx], &two, &b[j + jp2 * ldb], &ldb);
            }

            c128 tmp = fg[(j + 1) + j * ldfg];
            fg[(j + 1) + j * ldfg] = -fg[j + (j + 1) * ldfg];
            i32 jp2_dim = j + 2;
            SLC_ZGEMM("N", "N", &jp2_dim, &two, &two, &CONE, &fg[0 + j * ldfg], &ldfg,
                      &zwork[iq2_idx], &two, &CZERO, &zwork[iwrk_zw_idx], &jp2_dim);
            SLC_ZLACPY("Full", &jp2_dim, &two, &zwork[iwrk_zw_idx], &jp2_dim, &fg[0 + j * ldfg], &ldfg);

            i32 j2_j_p1 = j2 - j;
            if (j2_j_p1 > 0) {
                SLC_ZGEMM("C", "N", &two, &j2_j_p1, &two, &CONE, &zwork[iq2_idx], &two,
                          &fg[j + j * ldfg], &ldfg, &CZERO, &zwork[iwrk_zw_idx], &two);
                SLC_ZLACPY("Full", &two, &j2_j_p1, &zwork[iwrk_zw_idx], &two, &fg[j + j * ldfg], &ldfg);
            }
            fg[(j + 1) + j * ldfg] = tmp;

            if (lcmpq) {
                SLC_ZGEMM("N", "N", &n2, &two, &two, &CONE, &q[0 + j * ldq], &ldq,
                          &zwork[iq_zw_idx], &two, &CZERO, &zwork[iwrk_zw_idx], &n2);
                SLC_ZLACPY("Full", &n2, &two, &zwork[iwrk_zw_idx], &n2, &q[0 + j * ldq], &ldq);
                SLC_ZGEMM("N", "N", &n2, &two, &two, &CONE, &q[0 + (n + j) * ldq], &ldq,
                          &zwork[iq2_idx], &two, &CZERO, &zwork[iwrk_zw_idx], &n2);
                SLC_ZLACPY("Full", &n2, &two, &zwork[iwrk_zw_idx], &n2, &q[0 + (n + j) * ldq], &ldq);
            }

            if (lcmpu) {
                SLC_ZGEMM("N", "N", &n, &two, &two, &CONE, &u[0 + j * ldu], &ldu,
                          &zwork[iu_zw_idx], &two, &CZERO, &zwork[iwrk_zw_idx], &n);
                SLC_ZLACPY("Full", &n, &two, &zwork[iwrk_zw_idx], &n, &u[0 + j * ldu], &ldu);
                SLC_ZGEMM("N", "N", &n, &two, &two, &CONE, &u[0 + (n + j) * ldu], &ldu,
                          &zwork[iu_zw_idx], &two, &CZERO, &zwork[iwrk_zw_idx], &n);
                SLC_ZLACPY("Full", &n, &two, &zwork[iwrk_zw_idx], &n, &u[0 + (n + j) * ldu], &ldu);
            }

            bwork[j] = true;
            j += 2;
            SLC_ZLACPY("Full", &two, &two, &zwork[iq2_idx], &two, &zwork[iqb_idx], &two);
            SLC_ZLACPY("Full", &two, &two, &zwork[iu_zw_idx], &two, &zwork[iub_idx], &two);
            iqb_idx += 4;
            iub_idx += 4;
        } else {
            bwork[j] = false;
            b[(j + 1) + j * ldb] = CZERO;
            j += 1;
        }

        if (j >= j2 - 1 && j <= n - 1) {
            i32 iqb_tmp = iev_idx + 4;
            i32 iub_tmp = iqb_tmp + 4 * m;

            i32 i_loop = 0;
            j1 = j2;
            j2 = (n < j1 + nb) ? n : j1 + nb;
            i32 nc_loop = j2 - j1;

            while (i_loop < j) {
                if (bwork[i_loop]) {
                    i32 two = 2;
                    SLC_ZGEMM("C", "N", &two, &nc_loop, &two, &CONE, &zwork[iub_tmp], &two,
                              &z[i_loop + j1 * ldz], &ldz, &CZERO, &zwork[iwrk_zw_idx], &two);
                    SLC_ZLACPY("Full", &two, &nc_loop, &zwork[iwrk_zw_idx], &two,
                               &z[i_loop + j1 * ldz], &ldz);

                    SLC_ZGEMM("C", "N", &two, &nc_loop, &two, &CONE, &zwork[iqb_tmp], &two,
                              &b[i_loop + j1 * ldb], &ldb, &CZERO, &zwork[iwrk_zw_idx], &two);
                    SLC_ZLACPY("Full", &two, &nc_loop, &zwork[iwrk_zw_idx], &two,
                               &b[i_loop + j1 * ldb], &ldb);

                    SLC_ZGEMM("C", "N", &two, &nc_loop, &two, &CONE, &zwork[iqb_tmp], &two,
                              &fg[i_loop + j1 * ldfg], &ldfg, &CZERO, &zwork[iwrk_zw_idx], &two);
                    SLC_ZLACPY("Full", &two, &nc_loop, &zwork[iwrk_zw_idx], &two,
                               &fg[i_loop + j1 * ldfg], &ldfg);

                    iqb_tmp += 4;
                    iub_tmp += 4;
                    i_loop += 2;
                } else {
                    i_loop += 1;
                }
            }
        }
    }

    j1 = 0;
    j2 = (n < nb) ? n : nb;

    while ((j1 < n) && (j2 <= n)) {
        i32 iqb_tmp = iev_idx + 4;
        i32 iub_tmp = iqb_tmp + 4 * m;

        i32 i_loop = 0;
        i32 nc_loop = j2 - j1;

        while (i_loop < n - 1) {
            if (bwork[i_loop]) {
                i32 two = 2;
                SLC_ZGEMM("C", "N", &two, &nc_loop, &two, &CONE, &zwork[iub_tmp], &two,
                          &d[i_loop + j1 * ldd], &ldd, &CZERO, &zwork[iwrk_zw_idx], &two);
                SLC_ZLACPY("Full", &two, &nc_loop, &zwork[iwrk_zw_idx], &two,
                           &d[i_loop + j1 * ldd], &ldd);

                if (i_loop > j1) {
                    i32 j3_local = (nc_loop < i_loop - j1) ? nc_loop : i_loop - j1;
                    if (j3_local > 0) {
                        SLC_ZGEMM("C", "N", &two, &j3_local, &two, &CONE, &zwork[iub_tmp], &two,
                                  &c[i_loop + j1 * ldc], &ldc, &CZERO, &zwork[iwrk_zw_idx], &two);
                        SLC_ZLACPY("Full", &two, &j3_local, &zwork[iwrk_zw_idx], &two,
                                   &c[i_loop + j1 * ldc], &ldc);
                    }
                }

                iqb_tmp += 4;
                iub_tmp += 4;
                i_loop += 2;
            } else {
                i_loop += 1;
            }
        }
        j1 = j2;
        j2 = (n < j1 + nb) ? n : j1 + nb;
        if (j1 >= n) break;
    }

    for (i32 ii = 0; ii < n; ii++) {
        i32 len = ii + 1;
        SLC_ZSCAL(&len, &MCIMAG, &b[0 + ii * ldb], &int1);
    }

    for (i32 ii = 0; ii < n; ii++) {
        i32 len = ii + 1;
        SLC_ZSCAL(&len, &MCIMAG, &fg[0 + ii * ldfg], &int1);
    }

    zwork[0] = (c128)optzw;
    dwork[0] = (f64)optdw;
    i32 iw1_val = iwork[0];
    i32 iw_val = iwork[2 * iw1_val + 3];
    i32 k_copy = 3 * (n - 2 * iw_val + 1) + 12 * iw_val;
    SLC_DCOPY(&k_copy, &dwork[is_save + 1], &int1, &dwork[1], &int1);
}
