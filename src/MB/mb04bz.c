/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * MB04BZ - Eigenvalues of complex skew-Hamiltonian/Hamiltonian pencil
 *
 * Computes eigenvalues of aS - bH where:
 *   S = [[A, D], [E, A^H]] with D, E skew-Hermitian
 *   H = [[B, F], [G, -B^H]] with F Hermitian, G Hermitian
 *
 * Uses embedding to real skew-Hamiltonian/skew-Hamiltonian pencil.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>
#include <ctype.h>
#include <math.h>
#include <stdlib.h>

void mb04bz(const char *job, const char *compq, i32 n,
            c128 *a, i32 lda, c128 *de, i32 ldde, c128 *b, i32 ldb,
            c128 *fg, i32 ldfg, c128 *q, i32 ldq,
            f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, f64 *dwork, i32 ldwork,
            c128 *zwork, i32 lzwork, bool *bwork, i32 *info) {

    const f64 ZERO = 0.0;
    const f64 THREE = 3.0;
    const f64 MONE = -1.0;
    const c128 CZERO = 0.0 + 0.0*I;
    const c128 CONE = 1.0 + 0.0*I;
    const c128 MCIMAG = 0.0 - 1.0*I;

    char job_upper = (char)toupper((unsigned char)job[0]);
    char compq_upper = (char)toupper((unsigned char)compq[0]);

    bool ltri = (job_upper == 'T');
    bool lcmpq = (compq_upper == 'C');

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
        mindw = 3;
        minzw = 1;
    } else if (lcmpq) {
        mindb = 8 * nn + n2;
        mindw = 11 * nn + n2;
        minzw = 8 * n + 4;
    } else {
        mindb = 4 * nn + n2;
        if (ltri) {
            mindw = 5 * nn + 3 * n;
            minzw = 6 * n + 4;
        } else {
            mindw = 4 * nn + 3 * n;
            minzw = 1;
        }
    }

    bool lquery = (ldwork == -1 || lzwork == -1);

    *info = 0;
    if (!(job_upper == 'E' || ltri)) {
        *info = -1;
    } else if (!(compq_upper == 'N' || lcmpq)) {
        *info = -2;
    } else if (n < 0 || (n % 2) != 0) {
        *info = -3;
    } else if (lda < k_dim) {
        *info = -5;
    } else if (ldde < k_dim) {
        *info = -7;
    } else if (ldb < k_dim) {
        *info = -9;
    } else if (ldfg < k_dim) {
        *info = -11;
    } else if (ldq < 1 || (lcmpq && ldq < n2)) {
        *info = -13;
    } else if (!lquery) {
        if (ldwork < mindw) {
            dwork[0] = (f64)mindw;
            *info = -19;
        } else if (lzwork < minzw) {
            zwork[0] = (c128)minzw;
            *info = -21;
        }
    }

    if (*info != 0) {
        return;
    }

    i32 optdw = mindw;
    i32 optzw = minzw;
    i32 nb = 2;

    if (n > 0) {
        optzw = minzw;
        const char *cmpq = lcmpq ? "Initialize" : "No Computation";
        const char *jobf = ltri ? "Triangularize" : "Eigenvalues";

        if (ltri) {
            i32 lwork_query = -1;
            c128 work_query;
            i32 info_query;
            SLC_ZGEQRF(&n, &n, zwork, &n, zwork, &work_query, &lwork_query, &info_query);
            i32 qrf_work = (i32)creal(work_query);
            nb = (qrf_work / n > 2) ? qrf_work / n : 2;
        }

        if (lquery) {
            i32 lwork_fd = -1;
            f64 fd_opt;
            i32 info_fd;
            mb04fd(jobf, cmpq, n2, dwork, n, dwork, n,
                   dwork, n, dwork, n, dwork, n2,
                   alphai, alphar, beta, iwork, &fd_opt, lwork_fd, &info_fd);
            optdw = (mindw > mindb + (i32)fd_opt) ? mindw : mindb + (i32)fd_opt;
            dwork[0] = (f64)optdw;
            zwork[0] = (c128)optzw;
            return;
        }
    } else if (lquery) {
        dwork[0] = (f64)mindw;
        zwork[0] = (c128)minzw;
        return;
    }

    if (n == 0) {
        iwork[0] = 0;
        dwork[0] = THREE;
        dwork[1] = ZERO;
        dwork[2] = ZERO;
        zwork[0] = CONE;
        return;
    }

    i32 iq_idx = 0;
    i32 ia_idx;
    if (lcmpq) {
        ia_idx = iq_idx + n2 * n2;
    } else {
        ia_idx = 0;
    }
    i32 ide_idx = ia_idx + nn;
    i32 ib_idx = ide_idx + nn + n;
    i32 ifg_idx = ib_idx + nn;
    i32 iwrk_idx = ifg_idx + nn + n;

    i32 int1 = 1;
    i32 iw, is, iw1, i1;

    iw = ia_idx;
    is = iw + n * m;
    for (i32 jj = 0; jj < m; jj++) {
        iw1 = iw;
        for (i32 ii = 0; ii < m; ii++) {
            dwork[iw] = creal(a[ii + jj * lda]);
            iw++;
        }
        for (i32 ii = 0; ii < m; ii++) {
            dwork[iw] = cimag(a[ii + jj * lda]);
            dwork[is] = -dwork[iw];
            iw++;
            is++;
        }
        SLC_DCOPY(&m, &dwork[iw1], &int1, &dwork[is], &int1);
        is += m;
    }

    iw = ide_idx;
    for (i32 jj = 0; jj < m + 1; jj++) {
        for (i32 ii = 0; ii < m; ii++) {
            dwork[iw] = creal(de[ii + jj * ldde]);
            iw++;
        }
        iw += jj;
        is = iw;
        for (i32 ii = jj; ii < m; ii++) {
            dwork[iw] = cimag(de[ii + jj * ldde]);
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
            dwork[iw] = -cimag(de[ii + jj * ldde]);
            dwork[is] = dwork[iw];
            iw++;
            is += n;
        }
        iw += n - jj;
    }
    i32 mp1 = m + 1;
    SLC_DLACPY("Full", &m, &mp1, &dwork[ide_idx], &n, &dwork[iw1 - m], &n);

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
            dwork[iw] = -creal(fg[ii + jj * ldfg]);
            dwork[is] = dwork[iw];
            iw++;
            is += n;
        }
        iw += n - jj;
    }
    SLC_DLACPY("Full", &m, &mp1, &dwork[ifg_idx], &n, &dwork[iw1 - m], &n);

    const char *cmpq = lcmpq ? "Initialize" : "No Computation";
    const char *jobf = ltri ? "Triangularize" : "Eigenvalues";
    i32 ldwork_fd = ldwork - iwrk_idx;

    mb04fd(jobf, cmpq, n2, &dwork[ia_idx], n, &dwork[ide_idx], n,
           &dwork[ib_idx], n, &dwork[ifg_idx], n, &dwork[iq_idx], n2,
           alphai, alphar, beta, iwork, &dwork[iwrk_idx], ldwork_fd, info);

    if (*info == 1) {
        return;
    } else if (*info == 2) {
        *info = 3;
    }
    optdw = (mindw > mindb + (i32)dwork[iwrk_idx]) ? mindw : mindb + (i32)dwork[iwrk_idx];
    f64 nrmbs = dwork[iwrk_idx + 1];
    f64 nrmbt = dwork[iwrk_idx + 2];

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

    i32 i_val = iwork[0];
    i32 is_cnt = 0;
    i32 iw_cnt = 0;
    for (i32 jj = 0; jj < i_val; jj++) {
        if (iwork[jj + 1] > 0) {
            is_cnt++;
        } else if (iwork[jj + 1] < 0) {
            iw_cnt++;
        }
    }

    i32 i2 = 2 * i_val + 2;
    iwork[i2 - 1] = is_cnt;
    iwork[i2] = iw_cnt;

    if (ltri) {
        iw = ia_idx;
        for (i32 jj = 0; jj < n; jj++) {
            for (i32 ii = 0; ii <= jj; ii++) {
                a[ii + jj * lda] = dwork[iw];
                iw++;
            }
            iw += n - jj - 1;
        }

        iw = ide_idx + n;
        for (i32 jj = 0; jj < n; jj++) {
            for (i32 ii = 0; ii < jj; ii++) {
                de[ii + jj * ldde] = dwork[iw];
                iw++;
            }
            de[jj + jj * ldde] = CZERO;
            iw += n - jj;
        }

        iw = ib_idx;
        for (i32 jj = 0; jj < n; jj++) {
            i32 min_jp2_n = (jj + 2 < n) ? jj + 2 : n;
            for (i32 ii = 0; ii < min_jp2_n; ii++) {
                b[ii + jj * ldb] = dwork[iw];
                iw++;
            }
            iw += n - jj - 2;
            if (iw < ib_idx) iw = ib_idx;
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
    }

    i32 i1_cnt = 0;
    i32 ii = 0;
    while (ii < n - 1) {
        if (alphar[ii] != ZERO && beta[ii] != ZERO && alphai[ii] != ZERO) {
            i1_cnt++;
            ii += 2;
        } else {
            ii++;
        }
    }

    i32 i3 = 2 * iwork[0] + 4;
    iwork[i3 - 1] = i1_cnt;

    SLC_DCOPY(&n, &dwork[ia_idx], &(i32){n + 1}, &dwork[ifg_idx], &int1);
    dwork[0] = (f64)optdw;
    dwork[1] = nrmbs;
    dwork[2] = nrmbt;

    i32 k = 3;
    ii = 0;
    i32 iw_iwork = iwork[0];
    i32 jj = 0;
    i32 l = (n - 2 * i1_cnt) * 2 + k;
    bool unrel = false;

    while (ii < n) {
        if (jj < iw_iwork) {
            unrel = (ii == abs(iwork[jj + 1]) - 1);
        }
        if (alphar[ii] != ZERO && beta[ii] != ZERO && alphai[ii] != ZERO) {
            if (unrel) {
                jj++;
                iwork[jj + iw_iwork] = l + 1;
                unrel = false;
            }
            i32 two = 2;
            SLC_DLACPY("Full", &two, &two, &dwork[ib_idx + ii * (n + 1)], &n, &dwork[l], &two);
            l += 4;
            SLC_DCOPY(&two, &dwork[ifg_idx + ii], &int1, &dwork[l], &(i32){3});
            dwork[l + 1] = ZERO;
            dwork[l + 2] = ZERO;
            l += 4;
            ii += 2;
        } else {
            if (unrel) {
                jj++;
                iwork[jj + iw_iwork] = k + 1;
                unrel = false;
            }
            dwork[k] = dwork[ib_idx + ii * (n + 1)];
            dwork[k + 1] = dwork[ifg_idx + ii];
            k += 2;
            ii++;
        }
    }

    if (!ltri) {
        zwork[0] = (c128)optzw;
        return;
    }

    i32 iq2_idx = 0;
    i32 iev_idx = 4;
    i32 iq_zw_idx = 8;
    i32 iwrk_zw_idx = iq_zw_idx + 4 * (n - 1);

    jj = 0;
    i32 j1 = 0;
    i32 j2 = (n < nb) ? n : nb;

    while (jj < n - 1) {
        if (b[(jj + 1) + jj * ldb] != CZERO) {
            i32 nc = (j2 - jj - 2 > 0) ? j2 - jj - 2 : 0;
            i32 jm1 = (jj > 0) ? jj : 1;
            i32 jp2 = (jj + 2 < n) ? jj + 2 : n;
            c128 tmp = a[(jj + 1) + jj * lda];
            a[(jj + 1) + jj * lda] = CZERO;

            i32 two = 2;
            i32 ione = 1;
            i32 lzwork_qz = lzwork - iwrk_zw_idx;
            i32 info_qz;
            SLC_ZHGEQZ("Schur Form", "Initialize", "Initialize", &two, &ione,
                       &two, &b[jj + jj * ldb], &ldb, &a[jj + jj * lda], &lda,
                       &zwork[iev_idx], &zwork[iev_idx + 2], &zwork[iq_zw_idx], &two,
                       &zwork[iq2_idx], &two, &zwork[iwrk_zw_idx], &lzwork_qz,
                       &dwork[4 * n + 3], &info_qz);

            a[(jj + 1) + jj * lda] = tmp;
            if (info_qz > 0) {
                *info = 2;
                return;
            }

            if (jj > 0) {
                i32 jj_dim = jj;
                SLC_ZGEMM("N", "N", &jj_dim, &two, &two, &CONE, &a[0 + jj * lda], &lda,
                          &zwork[iq2_idx], &two, &CZERO, &zwork[iwrk_zw_idx], &jm1);
                SLC_ZLACPY("Full", &jj_dim, &two, &zwork[iwrk_zw_idx], &jm1, &a[0 + jj * lda], &lda);
            }
            if (nc > 0) {
                SLC_ZGEMM("C", "N", &two, &nc, &two, &CONE, &zwork[iq_zw_idx], &two,
                          &a[jj + jp2 * lda], &lda, &CZERO, &zwork[iwrk_zw_idx], &two);
                SLC_ZLACPY("Full", &two, &nc, &zwork[iwrk_zw_idx], &two, &a[jj + jp2 * lda], &lda);
            }

            tmp = de[(jj + 1) + jj * ldde];
            de[(jj + 1) + jj * ldde] = -de[jj + (jj + 1) * ldde];
            i32 jp2_dim = jj + 2;
            SLC_ZGEMM("N", "N", &jp2_dim, &two, &two, &CONE, &de[0 + jj * ldde], &ldde,
                      &zwork[iq_zw_idx], &two, &CZERO, &zwork[iwrk_zw_idx], &jp2_dim);
            SLC_ZLACPY("Full", &jp2_dim, &two, &zwork[iwrk_zw_idx], &jp2_dim, &de[0 + jj * ldde], &ldde);
            i32 j2_j_p1 = j2 - jj;
            SLC_ZGEMM("C", "N", &two, &j2_j_p1, &two, &CONE, &zwork[iq_zw_idx], &two,
                      &de[jj + jj * ldde], &ldde, &CZERO, &zwork[iwrk_zw_idx], &two);
            SLC_ZLACPY("Full", &two, &j2_j_p1, &zwork[iwrk_zw_idx], &two, &de[jj + jj * ldde], &ldde);
            de[(jj + 1) + jj * ldde] = tmp;

            if (jj > 0) {
                i32 jj_dim = jj;
                SLC_ZGEMM("N", "N", &jj_dim, &two, &two, &CONE, &b[0 + jj * ldb], &ldb,
                          &zwork[iq2_idx], &two, &CZERO, &zwork[iwrk_zw_idx], &jm1);
                SLC_ZLACPY("Full", &jj_dim, &two, &zwork[iwrk_zw_idx], &jm1, &b[0 + jj * ldb], &ldb);
            }
            if (nc > 0) {
                SLC_ZGEMM("C", "N", &two, &nc, &two, &CONE, &zwork[iq_zw_idx], &two,
                          &b[jj + jp2 * ldb], &ldb, &CZERO, &zwork[iwrk_zw_idx], &two);
                SLC_ZLACPY("Full", &two, &nc, &zwork[iwrk_zw_idx], &two, &b[jj + jp2 * ldb], &ldb);
            }

            tmp = fg[(jj + 1) + jj * ldfg];
            fg[(jj + 1) + jj * ldfg] = -fg[jj + (jj + 1) * ldfg];
            SLC_ZGEMM("N", "N", &jp2_dim, &two, &two, &CONE, &fg[0 + jj * ldfg], &ldfg,
                      &zwork[iq_zw_idx], &two, &CZERO, &zwork[iwrk_zw_idx], &jp2_dim);
            SLC_ZLACPY("Full", &jp2_dim, &two, &zwork[iwrk_zw_idx], &jp2_dim, &fg[0 + jj * ldfg], &ldfg);
            SLC_ZGEMM("C", "N", &two, &j2_j_p1, &two, &CONE, &zwork[iq_zw_idx], &two,
                      &fg[jj + jj * ldfg], &ldfg, &CZERO, &zwork[iwrk_zw_idx], &two);
            SLC_ZLACPY("Full", &two, &j2_j_p1, &zwork[iwrk_zw_idx], &two, &fg[jj + jj * ldfg], &ldfg);
            fg[(jj + 1) + jj * ldfg] = tmp;

            if (lcmpq) {
                SLC_ZGEMM("N", "N", &n2, &two, &two, &CONE, &q[0 + jj * ldq], &ldq,
                          &zwork[iq2_idx], &two, &CZERO, &zwork[iwrk_zw_idx], &n2);
                SLC_ZLACPY("Full", &n2, &two, &zwork[iwrk_zw_idx], &n2, &q[0 + jj * ldq], &ldq);
                SLC_ZGEMM("N", "N", &n2, &two, &two, &CONE, &q[0 + (n + jj) * ldq], &ldq,
                          &zwork[iq_zw_idx], &two, &CZERO, &zwork[iwrk_zw_idx], &n2);
                SLC_ZLACPY("Full", &n2, &two, &zwork[iwrk_zw_idx], &n2, &q[0 + (n + jj) * ldq], &ldq);
            }

            bwork[jj] = true;
            jj += 2;
            iq_zw_idx += 4;
        } else {
            bwork[jj] = false;
            b[(jj + 1) + jj * ldb] = CZERO;
            jj++;
        }

        if (jj >= j2) {
            j1 = j2;
            j2 = (n < j1 + nb) ? n : j1 + nb;
            i32 nc_loop = j2 - j1;

            i32 ii_loop = 0;
            i32 iqb_idx = 8;
            while (ii_loop < jj - 1) {
                if (bwork[ii_loop]) {
                    i32 two = 2;
                    SLC_ZGEMM("C", "N", &two, &nc_loop, &two, &CONE, &zwork[iqb_idx], &two,
                              &a[ii_loop + j1 * lda], &lda, &CZERO, &zwork[iwrk_zw_idx], &two);
                    SLC_ZLACPY("Full", &two, &nc_loop, &zwork[iwrk_zw_idx], &two,
                               &a[ii_loop + j1 * lda], &lda);

                    SLC_ZGEMM("C", "N", &two, &nc_loop, &two, &CONE, &zwork[iqb_idx], &two,
                              &de[ii_loop + j1 * ldde], &ldde, &CZERO, &zwork[iwrk_zw_idx], &two);
                    SLC_ZLACPY("Full", &two, &nc_loop, &zwork[iwrk_zw_idx], &two,
                               &de[ii_loop + j1 * ldde], &ldde);

                    SLC_ZGEMM("C", "N", &two, &nc_loop, &two, &CONE, &zwork[iqb_idx], &two,
                              &b[ii_loop + j1 * ldb], &ldb, &CZERO, &zwork[iwrk_zw_idx], &two);
                    SLC_ZLACPY("Full", &two, &nc_loop, &zwork[iwrk_zw_idx], &two,
                               &b[ii_loop + j1 * ldb], &ldb);

                    SLC_ZGEMM("C", "N", &two, &nc_loop, &two, &CONE, &zwork[iqb_idx], &two,
                              &fg[ii_loop + j1 * ldfg], &ldfg, &CZERO, &zwork[iwrk_zw_idx], &two);
                    SLC_ZLACPY("Full", &two, &nc_loop, &zwork[iwrk_zw_idx], &two,
                               &fg[ii_loop + j1 * ldfg], &ldfg);

                    iqb_idx += 4;
                    ii_loop += 2;
                } else {
                    ii_loop++;
                }
            }
        }
    }

    for (ii = 0; ii < n; ii++) {
        i32 len = ii + 1;
        SLC_ZSCAL(&len, &MCIMAG, &b[0 + ii * ldb], &int1);
    }

    for (ii = 0; ii < n; ii++) {
        i32 len = ii + 1;
        SLC_ZSCAL(&len, &MCIMAG, &fg[0 + ii * ldfg], &int1);
    }

    zwork[0] = (c128)optzw;
}
