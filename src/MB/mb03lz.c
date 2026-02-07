// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <ctype.h>

void mb03lz(const char *compq, const char *orth, i32 n,
            c128 *a, i32 lda, c128 *de, i32 ldde, c128 *b, i32 ldb,
            c128 *fg, i32 ldfg, i32 *neig, c128 *q, i32 ldq,
            f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, f64 *dwork, i32 ldwork,
            c128 *zwork, i32 lzwork, bool *bwork, i32 *info) {

    const f64 ONE = 1.0;
    const c128 CZERO = 0.0 + 0.0 * I;
    const c128 CONE = 1.0 + 0.0 * I;
    const c128 CIMAG = 0.0 + 1.0 * I;

    char compq_upper = (char)toupper((unsigned char)compq[0]);
    char orth_upper = (char)toupper((unsigned char)orth[0]);

    i32 m = n / 2;
    i32 nn = n * n;
    i32 n2 = 2 * n;
    *neig = 0;

    bool lcmpq = (compq_upper == 'C');
    bool qr = false, qrp = false, svd = false;
    if (lcmpq) {
        qr = (orth_upper == 'Q');
        qrp = (orth_upper == 'P');
        svd = (orth_upper == 'S');
    }

    i32 mindb, mindw, minzw;
    if (n == 0) {
        mindw = 1;
        minzw = 1;
    } else if (lcmpq) {
        mindb = 8 * nn + n2;
        mindw = 11 * nn + n2;
        minzw = 8 * n + 4;
    } else {
        mindb = 4 * nn + n2;
        mindw = 4 * nn + n2 + (3 > n ? 3 : n);
        minzw = 1;
    }

    bool lquery = (ldwork == -1 || lzwork == -1);

    *info = 0;
    if (compq_upper != 'N' && !lcmpq) {
        *info = -1;
    } else if (lcmpq && !(qr || qrp || svd)) {
        *info = -2;
    } else if (n < 0 || (n % 2) != 0) {
        *info = -3;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -5;
    } else if (ldde < (1 > m ? 1 : m)) {
        *info = -7;
    } else if (ldb < (1 > m ? 1 : m)) {
        *info = -9;
    } else if (ldfg < (1 > m ? 1 : m)) {
        *info = -11;
    } else if (ldq < 1 || (lcmpq && ldq < n2)) {
        *info = -14;
    } else if (!lquery) {
        if (ldwork < mindw) {
            dwork[0] = (f64)mindw;
            *info = -20;
        } else if (lzwork < minzw) {
            zwork[0] = (c128)minzw;
            *info = -22;
        }
    }

    if (*info != 0) {
        return;
    }

    i32 nb = 2;
    i32 optdw, optzw;

    if (n > 0) {
        if (lcmpq) {
            i32 neg1 = -1;
            SLC_ZGEQRF(&n, &n, q, &ldq, zwork, zwork, &neg1, info);
            i32 i_opt = (i32)creal(zwork[0]);
            nb = (i_opt / n > 2) ? i_opt / n : 2;
        }

        if (lquery) {
            i32 neg1 = -1;
            const char *job = lcmpq ? "T" : "E";
            const char *cmpq_fd = lcmpq ? "I" : "N";
            mb04fd(job, cmpq_fd, n2, dwork, n, dwork, n, dwork, n,
                   dwork, n, dwork, n2, alphai, alphar, beta,
                   iwork, dwork, neg1, info);
            optdw = (mindw > mindb + (i32)dwork[0]) ? mindw : mindb + (i32)dwork[0];

            if (lcmpq) {
                if (svd) {
                    i32 neg1_z = -1;
                    SLC_ZGESVD("O", "N", &n, &n, q, &ldq, dwork, zwork, &n,
                               zwork, &n, zwork, &neg1_z, dwork, info);
                    i32 j = (i32)creal(zwork[0]);
                    optzw = (minzw > j) ? minzw : j;
                } else {
                    i32 j, j1, j2;
                    i32 neg1_z = -1;
                    if (qr) {
                        j = m;
                        SLC_ZGEQRF(&n, &j, q, &ldq, zwork, zwork, &neg1_z, info);
                        j1 = (i32)creal(zwork[0]);
                    } else {
                        j = n;
                        SLC_ZGEQP3(&n, &j, q, &ldq, iwork, zwork, zwork, &neg1_z, dwork, info);
                        j1 = (i32)creal(zwork[0]);
                    }
                    SLC_ZUNGQR(&n, &j, &j, q, &ldq, zwork, &zwork[1], &neg1_z, info);
                    j2 = (i32)creal(zwork[1]);
                    j = j + ((j1 > j2) ? j1 : j2);
                    optzw = (minzw > j) ? minzw : j;
                }
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
        dwork[0] = ONE;
        zwork[0] = CONE;
        return;
    }

    f64 eps = SLC_DLAMCH("P");
    f64 tol = sqrt(eps);

    i32 iq, ia, ide, ib, ifg_idx, iwrk;
    if (lcmpq) {
        iq = 0;
        ia = iq + n2 * n2;
    } else {
        iq = 0;
        ia = 0;
    }
    ide = ia + nn;
    ib = ide + nn + n;
    ifg_idx = ib + nn;
    iwrk = ifg_idx + nn + n;

    i32 iw, is, iw1, i1;

    iw = ia;
    is = iw + n * m;
    for (i32 j = 0; j < m; j++) {
        iw1 = iw;
        for (i32 i = 0; i < m; i++) {
            dwork[iw] = creal(a[i + j * lda]);
            iw++;
        }
        for (i32 i = 0; i < m; i++) {
            dwork[iw] = cimag(a[i + j * lda]);
            dwork[is] = -dwork[iw];
            iw++;
            is++;
        }
        i32 i1_copy = m;
        SLC_DCOPY(&i1_copy, &dwork[iw1], &(i32){1}, &dwork[is], &(i32){1});
        is += m;
    }

    iw = ide;
    for (i32 j = 0; j < m + 1; j++) {
        for (i32 i = 0; i < m; i++) {
            dwork[iw] = creal(de[i + j * ldde]);
            iw++;
        }
        iw += j;
        is = iw;
        for (i32 i = j; i < m; i++) {
            dwork[iw] = cimag(de[i + j * ldde]);
            dwork[is] = dwork[iw];
            iw++;
            is += n;
        }
    }

    iw1 = iw;
    i1 = iw;
    for (i32 j = 1; j < m + 1; j++) {
        is = i1;
        i1++;
        for (i32 i = 0; i < j; i++) {
            dwork[iw] = -cimag(de[i + (j) * ldde]);
            dwork[is] = dwork[iw];
            iw++;
            is += n;
        }
        iw += n - j;
    }
    i32 m_copy = m, mp1 = m + 1;
    SLC_DLACPY("F", &m_copy, &mp1, &dwork[ide], &n, &dwork[iw1 - m], &n);

    iw = ib;
    is = iw + n * m;
    for (i32 j = 0; j < m; j++) {
        iw1 = iw;
        for (i32 i = 0; i < m; i++) {
            dwork[iw] = -cimag(b[i + j * ldb]);
            iw++;
        }
        for (i32 i = 0; i < m; i++) {
            dwork[iw] = creal(b[i + j * ldb]);
            dwork[is] = -dwork[iw];
            iw++;
            is++;
        }
        i32 m_len = m;
        SLC_DCOPY(&m_len, &dwork[iw1], &(i32){1}, &dwork[is], &(i32){1});
        is += m;
    }

    iw = ifg_idx;
    for (i32 j = 0; j < m + 1; j++) {
        for (i32 i = 0; i < m; i++) {
            dwork[iw] = -cimag(fg[i + j * ldfg]);
            iw++;
        }
        iw += j;
        is = iw;
        for (i32 i = j; i < m; i++) {
            dwork[iw] = creal(fg[i + j * ldfg]);
            dwork[is] = dwork[iw];
            iw++;
            is += n;
        }
    }

    iw1 = iw;
    i1 = iw;
    for (i32 j = 1; j < m + 1; j++) {
        is = i1;
        i1++;
        for (i32 i = 0; i < j; i++) {
            dwork[iw] = -creal(fg[i + (j) * ldfg]);
            dwork[is] = dwork[iw];
            iw++;
            is += n;
        }
        iw += n - j;
    }
    SLC_DLACPY("F", &m_copy, &mp1, &dwork[ifg_idx], &n, &dwork[iw1 - m], &n);

    const char *job_fd = lcmpq ? "T" : "E";
    const char *cmpq_fd = lcmpq ? "I" : "N";
    i32 ldwork_mb04fd = ldwork - iwrk;
    mb04fd(job_fd, cmpq_fd, n2, &dwork[ia], n, &dwork[ide], n,
           &dwork[ib], n, &dwork[ifg_idx], n, &dwork[iq], n2,
           alphai, alphar, beta, iwork, &dwork[iwrk], ldwork_mb04fd, info);

    i32 iwa = 0;
    if (*info == 2) {
        iwa = 4;
    } else if (*info > 0) {
        return;
    }

    optdw = (mindw > mindb + (i32)dwork[iwrk]) ? mindw : mindb + (i32)dwork[iwrk];

    f64 neg_one = -ONE;
    SLC_DSCAL(&n, &neg_one, alphai, &(i32){1});

    if (!lcmpq) {
        dwork[0] = (f64)optdw;
        zwork[0] = (c128)optzw;
        *info = iwa;
        return;
    }

    iw = ia;
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i <= j; i++) {
            a[i + j * lda] = dwork[iw] + 0.0 * I;
            iw++;
        }
        if (j >= m && j + 1 < n)
            a[j + 1 + j * lda] = CZERO;
        iw += n - j - 1;
    }

    iw = ide + n;
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < j; i++) {
            de[i + j * ldde] = dwork[iw] + 0.0 * I;
            iw++;
        }
        de[j + j * ldde] = CZERO;
        if (j >= m && j < n)
            de[j + 1 + j * ldde] = CZERO;
        iw += n - j;
    }

    iw = ib;
    for (i32 j = 0; j < n; j++) {
        i32 limit = (j + 1 < n) ? j + 1 : n - 1;
        for (i32 i = 0; i <= limit; i++) {
            b[i + j * ldb] = dwork[iw] + 0.0 * I;
            iw++;
        }
        iw += n - j - 2;
        if (iw < 0) iw = 0;
    }

    iw = ifg_idx + n;
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < j; i++) {
            fg[i + j * ldfg] = dwork[iw] + 0.0 * I;
            iw++;
        }
        fg[j + j * ldfg] = CZERO;
        if (j >= m && j < n)
            fg[j + 1 + j * ldfg] = CZERO;
        iw += n - j;
    }

    iw = iq;
    for (i32 j = 0; j < n2; j++) {
        for (i32 i = 0; i < n2; i++) {
            q[i + j * ldq] = dwork[iw] + 0.0 * I;
            iw++;
        }
    }

    i32 iq_z = 8;
    i32 iwrk_z = iq_z + 4 * (n - 1);

    i32 j = 0;
    i32 j1 = 0;
    i32 j2 = (n < j1 + nb) ? n : j1 + nb;
    i32 nc = 0;

    while (j < n - 1) {
        f64 nrmb = cabs(b[j + j * ldb]) + cabs(b[j + 1 + (j + 1) * ldb]);
        if (cabs(b[j + 1 + j * ldb]) > nrmb * eps) {
            nc = (j2 - j - 2 > 0) ? j2 - j - 2 : 0;
            i32 jm1 = (j > 0) ? j : 0;
            i32 jp2 = (j + 2 < n) ? j + 2 : n;

            c128 tmp = a[j + 1 + j * lda];
            a[j + 1 + j * lda] = CZERO;

            i32 two = 2, one_i = 1;
            c128 alpha_ev[2], beta_ev[2], zq[4], zq2[4];
            f64 rwork_ev[4];

            SLC_ZHGEQZ("S", "I", "I", &two, &one_i, &two,
                       &b[j + j * ldb], &ldb, &a[j + j * lda], &lda,
                       alpha_ev, beta_ev, zq, &two, zq2, &two,
                       &zwork[iwrk_z], &(i32){lzwork - iwrk_z}, rwork_ev, info);

            a[j + 1 + j * lda] = tmp;
            if (*info > 0) {
                *info = 2;
                return;
            }

            i32 len_j = j;
            if (len_j > 0) {
                SLC_ZGEMM("N", "N", &len_j, &two, &two, &CONE,
                          &a[j * lda], &lda, zq2, &two, &CZERO,
                          &zwork[iwrk_z], &jm1);
                SLC_ZLACPY("F", &len_j, &two, &zwork[iwrk_z], &jm1,
                           &a[j * lda], &lda);
            }

            if (nc > 0) {
                SLC_ZGEMM("C", "N", &two, &nc, &two, &CONE,
                          zq, &two, &a[j + jp2 * lda], &lda, &CZERO,
                          &zwork[iwrk_z], &two);
                SLC_ZLACPY("F", &two, &nc, &zwork[iwrk_z], &two,
                           &a[j + jp2 * lda], &lda);
            }

            tmp = de[j + 1 + j * ldde];
            de[j + 1 + j * ldde] = -de[j + (j + 1) * ldde];
            i32 jp1 = j + 1;
            SLC_ZGEMM("N", "N", &jp1, &two, &two, &CONE,
                      &de[j * ldde], &ldde, zq, &two, &CZERO,
                      &zwork[iwrk_z], &jp1);
            SLC_ZLACPY("F", &jp1, &two, &zwork[iwrk_z], &jp1,
                       &de[j * ldde], &ldde);
            i32 len_de = j2 - j;
            SLC_ZGEMM("C", "N", &two, &len_de, &two, &CONE,
                      zq, &two, &de[j + j * ldde], &ldde, &CZERO,
                      &zwork[iwrk_z], &two);
            SLC_ZLACPY("F", &two, &len_de, &zwork[iwrk_z], &two,
                       &de[j + j * ldde], &ldde);
            de[j + 1 + j * ldde] = tmp;

            if (len_j > 0) {
                SLC_ZGEMM("N", "N", &len_j, &two, &two, &CONE,
                          &b[j * ldb], &ldb, zq2, &two, &CZERO,
                          &zwork[iwrk_z], &jm1);
                SLC_ZLACPY("F", &len_j, &two, &zwork[iwrk_z], &jm1,
                           &b[j * ldb], &ldb);
            }
            if (nc > 0) {
                SLC_ZGEMM("C", "N", &two, &nc, &two, &CONE,
                          zq, &two, &b[j + jp2 * ldb], &ldb, &CZERO,
                          &zwork[iwrk_z], &two);
                SLC_ZLACPY("F", &two, &nc, &zwork[iwrk_z], &two,
                           &b[j + jp2 * ldb], &ldb);
            }

            tmp = fg[j + 1 + j * ldfg];
            fg[j + 1 + j * ldfg] = -fg[j + (j + 1) * ldfg];
            SLC_ZGEMM("N", "N", &jp1, &two, &two, &CONE,
                      &fg[j * ldfg], &ldfg, zq, &two, &CZERO,
                      &zwork[iwrk_z], &jp1);
            SLC_ZLACPY("F", &jp1, &two, &zwork[iwrk_z], &jp1,
                       &fg[j * ldfg], &ldfg);
            i32 len_fg = j2 - j;
            SLC_ZGEMM("C", "N", &two, &len_fg, &two, &CONE,
                      zq, &two, &fg[j + j * ldfg], &ldfg, &CZERO,
                      &zwork[iwrk_z], &two);
            SLC_ZLACPY("F", &two, &len_fg, &zwork[iwrk_z], &two,
                       &fg[j + j * ldfg], &ldfg);
            fg[j + 1 + j * ldfg] = tmp;

            SLC_ZGEMM("N", "N", &n2, &two, &two, &CONE,
                      &q[j * ldq], &ldq, zq2, &two, &CZERO,
                      &zwork[iwrk_z], &n2);
            SLC_ZLACPY("F", &n2, &two, &zwork[iwrk_z], &n2,
                       &q[j * ldq], &ldq);
            SLC_ZGEMM("N", "N", &n2, &two, &two, &CONE,
                      &q[(n + j) * ldq], &ldq, zq, &two, &CZERO,
                      &zwork[iwrk_z], &n2);
            SLC_ZLACPY("F", &n2, &two, &zwork[iwrk_z], &n2,
                       &q[(n + j) * ldq], &ldq);

            for (i32 ii = iq_z; ii < iq_z + 4; ii++) {
                zwork[ii] = zq[ii - iq_z];
            }

            bwork[j] = true;
            j += 2;
            iq_z += 4;
        } else {
            bwork[j] = false;
            b[j + 1 + j * ldb] = CZERO;
            j += 1;
        }

        if (j >= j2) {
            j1 = j2;
            j2 = (n < j1 + nb) ? n : j1 + nb;
            nc = j2 - j1;

            i32 i_loop = 0;
            i32 iqb = 8;
            while (i_loop < j) {
                if (bwork[i_loop]) {
                    i32 two = 2;
                    SLC_ZGEMM("C", "N", &two, &nc, &two, &CONE,
                              &zwork[iqb], &two, &a[i_loop + j1 * lda], &lda,
                              &CZERO, &zwork[iwrk_z], &two);
                    SLC_ZLACPY("F", &two, &nc, &zwork[iwrk_z], &two,
                               &a[i_loop + j1 * lda], &lda);

                    SLC_ZGEMM("C", "N", &two, &nc, &two, &CONE,
                              &zwork[iqb], &two, &de[i_loop + j1 * ldde], &ldde,
                              &CZERO, &zwork[iwrk_z], &two);
                    SLC_ZLACPY("F", &two, &nc, &zwork[iwrk_z], &two,
                               &de[i_loop + j1 * ldde], &ldde);

                    SLC_ZGEMM("C", "N", &two, &nc, &two, &CONE,
                              &zwork[iqb], &two, &b[i_loop + j1 * ldb], &ldb,
                              &CZERO, &zwork[iwrk_z], &two);
                    SLC_ZLACPY("F", &two, &nc, &zwork[iwrk_z], &two,
                               &b[i_loop + j1 * ldb], &ldb);

                    SLC_ZGEMM("C", "N", &two, &nc, &two, &CONE,
                              &zwork[iqb], &two, &fg[i_loop + j1 * ldfg], &ldfg,
                              &CZERO, &zwork[iwrk_z], &two);
                    SLC_ZLACPY("F", &two, &nc, &zwork[iwrk_z], &two,
                               &fg[i_loop + j1 * ldfg], &ldfg);

                    iqb += 4;
                    i_loop += 2;
                } else {
                    i_loop += 1;
                }
            }
        }
    }

    c128 neg_cimag = -CIMAG;
    for (i32 i = 0; i < n; i++) {
        i32 ip1 = i + 1;
        SLC_ZSCAL(&ip1, &neg_cimag, &b[i * ldb], &(i32){1});
    }
    for (i32 i = 0; i < n; i++) {
        i32 ip1 = i + 1;
        SLC_ZSCAL(&ip1, &neg_cimag, &fg[i * ldfg], &(i32){1});
    }

    mb03jz("U", n2, a, lda, de, ldde, b, ldb, fg, ldfg, q, ldq, neig, tol, info);

    if (qr)
        *neig = *neig / 2;

    i32 itau = 0;
    iwrk_z = *neig;

    if (*neig <= m) {
        for (i32 i = 0; i < *neig; i++) {
            SLC_ZAXPY(&m, &CIMAG, &q[m + i * ldq], &(i32){1}, &q[i * ldq], &(i32){1});
        }
        SLC_ZLACPY("F", &m, neig, &q[n + 0 * ldq], &ldq, &q[m + 0 * ldq], &ldq);
        for (i32 i = 0; i < *neig; i++) {
            SLC_ZAXPY(&m, &CIMAG, &q[m + n + i * ldq], &(i32){1}, &q[m + i * ldq], &(i32){1});
        }
    } else {
        for (i32 i = 0; i < m; i++) {
            SLC_ZAXPY(&m, &CIMAG, &q[m + i * ldq], &(i32){1}, &q[i * ldq], &(i32){1});
        }
        SLC_ZLACPY("F", &m, &m, &q[n + 0 * ldq], &ldq, &q[m + 0 * ldq], &ldq);
        for (i32 i = 0; i < m; i++) {
            SLC_ZAXPY(&m, &CIMAG, &q[m + n + i * ldq], &(i32){1}, &q[m + i * ldq], &(i32){1});
        }

        i32 neig_minus_m = *neig - m;
        for (i32 i = 0; i < neig_minus_m; i++) {
            SLC_ZAXPY(&m, &CIMAG, &q[m + (m + i) * ldq], &(i32){1}, &q[(m + i) * ldq], &(i32){1});
        }
        SLC_ZLACPY("F", &m, &neig_minus_m, &q[n + m * ldq], &ldq, &q[m + m * ldq], &ldq);
        for (i32 i = 0; i < neig_minus_m; i++) {
            SLC_ZAXPY(&m, &CIMAG, &q[m + n + (m + i) * ldq], &(i32){1}, &q[m + (m + i) * ldq], &(i32){1});
        }
    }

    if (svd) {
        i32 lzwork_svd = lzwork;
        SLC_ZGESVD("O", "N", &n, neig, q, &ldq, dwork,
                   zwork, &(i32){1}, zwork, &(i32){1}, zwork, &lzwork_svd,
                   &dwork[iwrk_z], info);
        if (*info > 0) {
            *info = 3;
            return;
        }
        optzw = (optzw > (i32)creal(zwork[0])) ? optzw : (i32)creal(zwork[0]);
        *neig = *neig / 2;
    } else {
        if (qr) {
            i32 lzwork_qr = lzwork - iwrk_z;
            SLC_ZGEQRF(&n, neig, q, &ldq, &zwork[itau], &zwork[iwrk_z],
                       &lzwork_qr, info);
            optzw = (optzw > (i32)creal(zwork[iwrk_z]) + iwrk_z) ? optzw : (i32)creal(zwork[iwrk_z]) + iwrk_z;
        } else {
            for (j = 0; j < *neig; j++) {
                iwork[j] = 0;
            }
            i32 lzwork_qp3 = lzwork - iwrk_z;
            SLC_ZGEQP3(&n, neig, q, &ldq, iwork, zwork, &zwork[iwrk_z],
                       &lzwork_qp3, dwork, info);
            optzw = (optzw > (i32)creal(zwork[iwrk_z]) + iwrk_z) ? optzw : (i32)creal(zwork[iwrk_z]) + iwrk_z;
        }

        i32 lzwork_ungqr = lzwork - iwrk_z;
        SLC_ZUNGQR(&n, neig, neig, q, &ldq, &zwork[itau],
                   &zwork[iwrk_z], &lzwork_ungqr, info);
        optzw = (optzw > (i32)creal(zwork[iwrk_z]) + iwrk_z) ? optzw : (i32)creal(zwork[iwrk_z]) + iwrk_z;

        if (qrp)
            *neig = *neig / 2;
    }

    dwork[0] = (f64)optdw;
    zwork[0] = (c128)optzw;
    *info = iwa;
}
