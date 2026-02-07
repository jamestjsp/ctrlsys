// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void mb03fz(const char *compq, const char *compu, const char *orth, i32 n,
            c128 *z, i32 ldz, c128 *b, i32 ldb, c128 *fg, i32 ldfg, i32 *neig,
            c128 *d, i32 ldd, c128 *c, i32 ldc, c128 *q, i32 ldq, c128 *u,
            i32 ldu, f64 *alphar, f64 *alphai, f64 *beta, i32 *iwork,
            i32 liwork, f64 *dwork, i32 ldwork, c128 *zwork, i32 lzwork,
            bool *bwork, i32 *info) {
    const f64 one = 1.0;
    const c128 czero = 0.0 + 0.0 * I;
    const c128 cone = 1.0 + 0.0 * I;
    const c128 cimone = 0.0 + 1.0 * I;

    i32 m = n / 2;
    i32 nn = n * n;
    i32 n2 = 2 * n;
    *neig = 0;

    bool lcmpq = (compq[0] == 'C' || compq[0] == 'c');
    bool lcmpu = (compu[0] == 'C' || compu[0] == 'c');
    bool lcmp = lcmpq || lcmpu;

    bool qr = false, qrp = false, svd = false;
    if (lcmp) {
        qr = (orth[0] == 'Q' || orth[0] == 'q');
        qrp = (orth[0] == 'P' || orth[0] == 'p');
        svd = (orth[0] == 'S' || orth[0] == 's');
    }

    i32 mindw, minzw;
    if (n == 0) {
        mindw = 1;
        minzw = 1;
    } else {
        i32 j_coef;
        if (!lcmpu) {
            if (!lcmpq) {
                j_coef = 13;
                minzw = 1;
            } else {
                j_coef = 16;
                minzw = 4 * n2 + 28;
            }
        } else {
            j_coef = 18;
            if (lcmpq) {
                minzw = 4 * n2 + 28;
            } else {
                minzw = 3 * n2 + 28;
            }
        }
        i32 max_6n_27 = (6 * n > 27) ? 6 * n : 27;
        mindw = j_coef * nn + n + max_6n_27;
    }

    bool lquery = (ldwork == -1 || lzwork == -1);

    *info = 0;
    bool compq_n = (compq[0] == 'N' || compq[0] == 'n');
    bool compu_n = (compu[0] == 'N' || compu[0] == 'n');

    if (!compq_n && !lcmpq) {
        *info = -1;
    } else if (!compu_n && !lcmpu) {
        *info = -2;
    } else if (lcmp && !(qr || qrp || svd)) {
        *info = -3;
    } else if (n < 0 || (n % 2) != 0) {
        *info = -4;
    } else if (ldz < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (ldfg < (n > 1 ? n : 1)) {
        *info = -10;
    } else if (ldd < 1 || (lcmp && ldd < n)) {
        *info = -13;
    } else if (ldc < 1 || (lcmp && ldc < n)) {
        *info = -15;
    } else if (ldq < 1 || (lcmpq && ldq < n2)) {
        *info = -17;
    } else if (ldu < 1 || (lcmpu && ldu < n)) {
        *info = -19;
    } else if (liwork < n2 + 9) {
        *info = -24;
    } else if (!lquery) {
        if (ldwork < mindw) {
            dwork[0] = (f64)mindw;
            *info = -26;
        } else if (lzwork < minzw) {
            zwork[0] = (c128)minzw;
            *info = -28;
        }
    }

    if (*info != 0) {
        return;
    }

    i32 nb = 2;
    i32 optdw, optzw;

    if (n > 0) {
        char cmpq_str[16], cmpu_str[16], job_str[16];
        if (lcmpq) {
            strcpy(cmpq_str, "Initialize");
        } else {
            strcpy(cmpq_str, "No Computation");
        }
        if (lcmpu) {
            strcpy(cmpu_str, "Initialize");
        } else {
            strcpy(cmpu_str, "No Computation");
        }
        if (lcmp) {
            strcpy(job_str, "Triangularize");
            i32 lwork_query = -1;
            c128 work_opt;
            i32 info_qry;
            SLC_ZGEQRF(&n, &n, z, &ldz, zwork, &work_opt, &lwork_query, &info_qry);
            i32 i_opt = (i32)creal(work_opt);
            nb = (i_opt / n > 2) ? i_opt / n : 2;
        } else {
            strcpy(job_str, "Eigenvalues");
        }

        if (lquery) {
            i32 mindb = (lcmpu ? 12 : (lcmpq ? 10 : 10)) * nn + n;
            i32 info_mb04;
            i32 ldwork_mb04 = -1;
            mb04ed(job_str, cmpq_str, cmpu_str, n2, dwork, n2, dwork, n, dwork,
                   n, dwork, n2, dwork, n, dwork, n, alphai, alphar, beta,
                   iwork, liwork, dwork, ldwork_mb04, &info_mb04);
            i32 opt_mb04 = (i32)dwork[0];
            optdw = (mindw > mindb + opt_mb04) ? mindw : mindb + opt_mb04;

            if (lcmp) {
                i32 j_opt = 0;
                if (svd) {
                    i32 lwork_svd = -1;
                    i32 info_svd;
                    c128 work_svd;
                    f64 rwork_svd[1];
                    SLC_ZGESVD("O", "N", &n, &n, q, &ldq, dwork, zwork, &(i32){1},
                               zwork, &(i32){1}, &work_svd, &lwork_svd, rwork_svd,
                               &info_svd);
                    j_opt = (i32)creal(work_svd);
                } else {
                    i32 j_dim;
                    c128 work1, work2;
                    i32 lwork_1 = -1, info_1;
                    if (qr) {
                        j_dim = m;
                        SLC_ZGEQRF(&n, &j_dim, q, &ldq, zwork, &work1, &lwork_1,
                                   &info_1);
                    } else {
                        j_dim = n;
                        f64 rwork_qp3[1];
                        SLC_ZGEQP3(&n, &j_dim, q, &ldq, iwork, zwork, &work1,
                                   &lwork_1, rwork_qp3, &info_1);
                    }
                    SLC_ZUNGQR(&n, &j_dim, &j_dim, q, &ldq, zwork, &work2,
                               &lwork_1, &info_1);
                    i32 w1 = (i32)creal(work1);
                    i32 w2 = (i32)creal(work2);
                    j_opt = j_dim + ((w1 > w2) ? w1 : w2);
                }
                i32 i_zgeqrf = (lcmp) ? (i32)creal(zwork[0]) : 0;
                optzw = minzw;
                if (i_zgeqrf > optzw) optzw = i_zgeqrf;
                if (j_opt > optzw) optzw = j_opt;
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
        dwork[0] = one;
        zwork[0] = cone;
        return;
    }

    f64 eps = SLC_DLAMCH("P");
    f64 tol = sqrt(eps);

    i32 iq_off = 0;
    i32 iu_off, iz11_off, ib_off, ifg_off, iwrk_off;
    if (lcmpu) {
        iu_off = iq_off + n2 * n2;
        iz11_off = iu_off + n2 * n;
    } else {
        iu_off = 0;
        iz11_off = iq_off + n2 * n2;
    }
    ib_off = iz11_off + n2 * n2;
    ifg_off = ib_off + nn;
    iwrk_off = ifg_off + nn + n;

    i32 iw = iz11_off;
    i32 is_off = iw + n2 * m;
    for (i32 j = 0; j < n; j++) {
        i32 iw1 = iw;
        for (i32 i = 0; i < m; i++) {
            dwork[iw] = creal(z[i + j * ldz]);
            iw++;
        }
        for (i32 i = 0; i < m; i++) {
            dwork[iw] = cimag(z[i + j * ldz]);
            dwork[is_off] = -dwork[iw];
            iw++;
            is_off++;
        }
        SLC_DCOPY(&m, &dwork[iw1], &(i32){1}, &dwork[is_off], &(i32){1});
        iw1 = iw;
        is_off += m;

        for (i32 i = m; i < n; i++) {
            dwork[iw] = creal(z[i + j * ldz]);
            iw++;
        }
        for (i32 i = m; i < n; i++) {
            dwork[iw] = cimag(z[i + j * ldz]);
            dwork[is_off] = -dwork[iw];
            iw++;
            is_off++;
        }
        SLC_DCOPY(&m, &dwork[iw1], &(i32){1}, &dwork[is_off], &(i32){1});
        iw1 = iw;
        is_off += m;
        if ((j + 1) % m == 0) {
            iw += n2 * m;
            is_off += n2 * m;
        }
    }

    iw = ib_off;
    is_off = iw + n * m;
    for (i32 j = 0; j < m; j++) {
        i32 iw1 = iw;
        for (i32 i = 0; i < m; i++) {
            dwork[iw] = -cimag(b[i + j * ldb]);
            iw++;
        }
        for (i32 i = 0; i < m; i++) {
            dwork[iw] = creal(b[i + j * ldb]);
            dwork[is_off] = -dwork[iw];
            iw++;
            is_off++;
        }
        SLC_DCOPY(&m, &dwork[iw1], &(i32){1}, &dwork[is_off], &(i32){1});
        is_off += m;
    }

    iw = ifg_off;
    for (i32 j = 0; j <= m; j++) {
        for (i32 i = 0; i < m; i++) {
            dwork[iw] = -cimag(fg[i + j * ldfg]);
            iw++;
        }
        iw += j;
        i32 is_tmp = iw;
        for (i32 i = j; i < m; i++) {
            dwork[iw] = creal(fg[i + j * ldfg]);
            dwork[is_tmp] = dwork[iw];
            iw++;
            is_tmp += n;
        }
    }

    i32 iw1 = iw;
    i32 i1 = iw;
    for (i32 j = 1; j <= m; j++) {
        i32 is_tmp = i1;
        i1++;
        for (i32 i = 0; i < j; i++) {
            dwork[iw] = -creal(fg[i + (j) * ldfg]);
            dwork[is_tmp] = dwork[iw];
            iw++;
            is_tmp += n;
        }
        iw += n - j;
    }
    i32 mp1 = m + 1;
    SLC_DLACPY("F", &m, &mp1, &dwork[ifg_off], &n, &dwork[iw1 - m], &n);

    char job_str[16], cmpq_str[16], cmpu_str[16];
    if (lcmp) {
        strcpy(job_str, "T");
    } else {
        strcpy(job_str, "E");
    }
    if (lcmpq) {
        strcpy(cmpq_str, "I");
    } else {
        strcpy(cmpq_str, "N");
    }
    if (lcmpu) {
        strcpy(cmpu_str, "I");
    } else {
        strcpy(cmpu_str, "N");
    }

    i32 info_mb04;

#ifdef MB03FZ_DEBUG
    fprintf(stderr, "MB03FZ: Before mb04ed, n=%d, m=%d, n2=%d\n", n, m, n2);
    fprintf(stderr, "MB03FZ: Z embedding (%d x %d):\n", n2, n2);
    for (i32 row = 0; row < n2; row++) {
        fprintf(stderr, "  ");
        for (i32 col = 0; col < n2; col++) {
            fprintf(stderr, "%+.4e ", dwork[iz11_off + row + col*n2]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "MB03FZ: B embedding (%d x %d):\n", n, n);
    for (i32 row = 0; row < n; row++) {
        fprintf(stderr, "  ");
        for (i32 col = 0; col < n; col++) {
            fprintf(stderr, "%+.4e ", dwork[ib_off + row + col*n]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "MB03FZ: FG embedding (%d x %d):\n", n, n+m);
    for (i32 row = 0; row < n; row++) {
        fprintf(stderr, "  ");
        for (i32 col = 0; col < n+m; col++) {
            fprintf(stderr, "%+.4e ", dwork[ifg_off + row + col*n]);
        }
        fprintf(stderr, "\n");
    }
#endif

    // Fortran MB03FZ intentionally swaps ALPHAI/ALPHAR in call to MB04ED.
    // Our C mb04ed has the same parameter order as Fortran MB04ED,
    // so we must also swap to match Fortran MB03FZ behavior.
    mb04ed(job_str, cmpq_str, cmpu_str, n2, &dwork[iz11_off], n2, &dwork[ib_off],
           n, &dwork[ifg_off], n, &dwork[iq_off], n2, &dwork[iu_off], n,
           &dwork[iu_off + nn], n, alphai, alphar, beta, iwork, liwork,
           &dwork[iwrk_off], ldwork - iwrk_off, &info_mb04);

#ifdef MB03FZ_DEBUG
    fprintf(stderr, "MB03FZ: After mb04ed, info_mb04=%d\n", info_mb04);
    fprintf(stderr, "MB03FZ: alphai (before DSCAL): ");
    for (i32 i = 0; i < n; i++) fprintf(stderr, "%.6e ", alphai[i]);
    fprintf(stderr, "\nMB03FZ: alphar: ");
    for (i32 i = 0; i < n; i++) fprintf(stderr, "%.6e ", alphar[i]);
    fprintf(stderr, "\nMB03FZ: beta: ");
    for (i32 i = 0; i < n; i++) fprintf(stderr, "%.6e ", beta[i]);
    fprintf(stderr, "\n");
#endif

    if (info_mb04 > 0 && info_mb04 < 3) {
        *info = info_mb04;
        return;
    }
    i32 mindb = (lcmpu ? 12 : (lcmpq ? 10 : 10)) * nn + n;
    optdw = (mindw > mindb + (i32)dwork[iwrk_off]) ? mindw : mindb + (i32)dwork[iwrk_off];

    f64 neg_one = -one;
    SLC_DSCAL(&n, &neg_one, alphai, &(i32){1});

    if (!lcmp) {
        dwork[0] = (f64)optdw;
        zwork[0] = (c128)optzw;
        return;
    }

    iw = iz11_off;
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i <= j; i++) {
            z[i + j * ldz] = dwork[iw];
            iw++;
        }
        iw += n2 - j - 1;
    }

    iw = iz11_off + n2 * n;
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < n; i++) {
            d[i + j * ldd] = dwork[iw];
            iw++;
        }
        iw += j;
        for (i32 i = j; i < n; i++) {
            c[i + j * ldc] = dwork[iw];
            iw++;
        }
    }

    iw = ib_off;
    for (i32 j = 0; j < n; j++) {
        i32 max_i = (j + 2 < n) ? j + 2 : n;
        for (i32 i = 0; i < max_i; i++) {
            b[i + j * ldb] = dwork[iw];
            iw++;
        }
        iw += n - j - 2;
        if (n - j - 2 < 0) iw += 0;
    }

    iw = ifg_off + n;
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < j; i++) {
            fg[i + j * ldfg] = dwork[iw];
            iw++;
        }
        fg[j + j * ldfg] = czero;
        iw += n - j;
    }

    if (lcmpq) {
        iw = iq_off;
        for (i32 j = 0; j < n2; j++) {
            for (i32 i = 0; i < n2; i++) {
                q[i + j * ldq] = dwork[iw];
                iw++;
            }
        }
    }

    if (lcmpu) {
        iw = iu_off;
        for (i32 j = 0; j < n2; j++) {
            for (i32 i = 0; i < n; i++) {
                u[i + j * ldu] = dwork[iw];
                iw++;
            }
        }
    }

    i32 iq2 = 0;
    i32 iq_bl = iq2 + 4;
    i32 iu_bl = iq_bl + 4;
    i32 ib_bl = iu_bl + 4;
    i32 iz11_bl = ib_bl + 4;
    i32 iz22_bl = iz11_bl + 4;
    i32 iev_bl = iz22_bl + 4;
    i32 iqb_bl = iev_bl + 4;
    i32 iub_bl = iqb_bl + 4 * m;
    i32 iwrk_bl = iub_bl + 4 * m;

    iwork[0] = 1;
    iwork[1] = -1;
    iwork[2] = -1;

    i32 j_idx = 0;
    i32 j1 = 0;
    i32 j2 = (n < nb) ? n : nb;
    i32 int1 = 1;
    i32 int2 = 2;

    while (j_idx + 1 < n) {
        f64 nrmb_local = cabs(b[j_idx + j_idx * ldb]) + cabs(b[(j_idx + 1) + (j_idx + 1) * ldb]);
        if (cabs(b[(j_idx + 1) + j_idx * ldb]) > nrmb_local * eps) {
            i32 nc = (j2 - j_idx - 2 > 0) ? j2 - j_idx - 2 : 0;
            i32 jm1 = (j_idx > 0) ? j_idx : 1;
            i32 jp2 = (j_idx + 2 < n) ? j_idx + 2 : n;
            i32 nj1 = (n - j_idx - 1 > 0) ? n - j_idx - 1 : 1;

            SLC_ZLACPY("F", &int2, &int2, &b[j_idx + j_idx * ldb], &ldb, &zwork[ib_bl], &int2);
            SLC_ZLACPY("U", &int2, &int2, &z[j_idx + j_idx * ldz], &ldz, &zwork[iz11_bl], &int2);
            zwork[iz11_bl + 1] = czero;
            zwork[iz22_bl] = conj(c[j_idx + j_idx * ldc]);
            zwork[iz22_bl + 1] = czero;
            zwork[iz22_bl + 2] = conj(c[(j_idx + 1) + j_idx * ldc]);
            zwork[iz22_bl + 3] = conj(c[(j_idx + 1) + (j_idx + 1) * ldc]);

            i32 info_bz;
            mb03bz("S", "I", 3, 2, 1, 2, iwork, &zwork[ib_bl], 2, 2,
                   &zwork[iq2], 2, 2, &zwork[iev_bl], &zwork[iev_bl + 2],
                   &iwork[3], dwork, ldwork, &zwork[iwrk_bl], lzwork - iwrk_bl,
                   &info_bz);
            if (info_bz > 0) {
                *info = 2;
                return;
            }

            i32 len_jm1 = j_idx;
            SLC_ZGEMM("N", "N", &len_jm1, &int2, &int2, &cone, &z[0 + j_idx * ldz],
                      &ldz, &zwork[iq_bl], &int2, &czero, &zwork[iwrk_bl], &jm1);
            SLC_ZLACPY("F", &len_jm1, &int2, &zwork[iwrk_bl], &jm1, &z[0 + j_idx * ldz], &ldz);
            SLC_ZLACPY("U", &int2, &int2, &zwork[iz11_bl], &int2, &z[j_idx + j_idx * ldz], &ldz);
            z[(j_idx + 1) + j_idx * ldz] = czero;
            if (nc > 0) {
                SLC_ZGEMM("C", "N", &int2, &nc, &int2, &cone, &zwork[iu_bl], &int2,
                          &z[j_idx + jp2 * ldz], &ldz, &czero, &zwork[iwrk_bl], &int2);
                SLC_ZLACPY("F", &int2, &nc, &zwork[iwrk_bl], &int2, &z[j_idx + jp2 * ldz], &ldz);
            }
            SLC_ZGEMM("N", "N", &n, &int2, &int2, &cone, &d[0 + j_idx * ldd],
                      &ldd, &zwork[iq2], &int2, &czero, &zwork[iwrk_bl], &n);
            SLC_ZLACPY("F", &n, &int2, &zwork[iwrk_bl], &n, &d[0 + j_idx * ldd], &ldd);
            c[j_idx + j_idx * ldc] = conj(zwork[iz22_bl]);
            c[(j_idx + 1) + j_idx * ldc] = conj(zwork[iz22_bl + 2]);
            c[j_idx + (j_idx + 1) * ldc] = czero;
            c[(j_idx + 1) + (j_idx + 1) * ldc] = conj(zwork[iz22_bl + 3]);
            i32 len_nj = n - j_idx - 2;
            if (len_nj > 0) {
                SLC_ZGEMM("N", "N", &len_nj, &int2, &int2, &cone, &c[jp2 + j_idx * ldc],
                          &ldc, &zwork[iq2], &int2, &czero, &zwork[iwrk_bl], &nj1);
                SLC_ZLACPY("F", &len_nj, &int2, &zwork[iwrk_bl], &nj1, &c[jp2 + j_idx * ldc], &ldc);
            }
            SLC_ZGEMM("N", "N", &len_jm1, &int2, &int2, &cone, &b[0 + j_idx * ldb],
                      &ldb, &zwork[iq_bl], &int2, &czero, &zwork[iwrk_bl], &jm1);
            SLC_ZLACPY("F", &len_jm1, &int2, &zwork[iwrk_bl], &jm1, &b[0 + j_idx * ldb], &ldb);
            SLC_ZLACPY("U", &int2, &int2, &zwork[ib_bl], &int2, &b[j_idx + j_idx * ldb], &ldb);
            b[(j_idx + 1) + j_idx * ldb] = czero;
            if (nc > 0) {
                SLC_ZGEMM("C", "N", &int2, &nc, &int2, &cone, &zwork[iq2], &int2,
                          &b[j_idx + jp2 * ldb], &ldb, &czero, &zwork[iwrk_bl], &int2);
                SLC_ZLACPY("F", &int2, &nc, &zwork[iwrk_bl], &int2, &b[j_idx + jp2 * ldb], &ldb);
            }

            c128 tmp = fg[(j_idx + 1) + j_idx * ldfg];
            fg[(j_idx + 1) + j_idx * ldfg] = -fg[j_idx + (j_idx + 1) * ldfg];
            i32 len_jp1 = j_idx + 1;
            i32 ldj1 = j_idx + 1;
            SLC_ZGEMM("N", "N", &len_jp1, &int2, &int2, &cone, &fg[0 + j_idx * ldfg],
                      &ldfg, &zwork[iq2], &int2, &czero, &zwork[iwrk_bl], &ldj1);
            SLC_ZLACPY("F", &len_jp1, &int2, &zwork[iwrk_bl], &ldj1, &fg[0 + j_idx * ldfg], &ldfg);
            i32 len_j2j = j2 - j_idx;
            SLC_ZGEMM("C", "N", &int2, &len_j2j, &int2, &cone, &zwork[iq2], &int2,
                      &fg[j_idx + j_idx * ldfg], &ldfg, &czero, &zwork[iwrk_bl], &int2);
            SLC_ZLACPY("F", &int2, &len_j2j, &zwork[iwrk_bl], &int2, &fg[j_idx + j_idx * ldfg], &ldfg);
            fg[(j_idx + 1) + j_idx * ldfg] = tmp;

            if (lcmpq) {
                SLC_ZGEMM("N", "N", &n2, &int2, &int2, &cone, &q[0 + j_idx * ldq],
                          &ldq, &zwork[iq_bl], &int2, &czero, &zwork[iwrk_bl], &n2);
                SLC_ZLACPY("F", &n2, &int2, &zwork[iwrk_bl], &n2, &q[0 + j_idx * ldq], &ldq);
                i32 nj_col = n + j_idx;
                SLC_ZGEMM("N", "N", &n2, &int2, &int2, &cone, &q[0 + nj_col * ldq],
                          &ldq, &zwork[iq2], &int2, &czero, &zwork[iwrk_bl], &n2);
                SLC_ZLACPY("F", &n2, &int2, &zwork[iwrk_bl], &n2, &q[0 + nj_col * ldq], &ldq);
            }

            if (lcmpu) {
                SLC_ZGEMM("N", "N", &n, &int2, &int2, &cone, &u[0 + j_idx * ldu],
                          &ldu, &zwork[iu_bl], &int2, &czero, &zwork[iwrk_bl], &n);
                SLC_ZLACPY("F", &n, &int2, &zwork[iwrk_bl], &n, &u[0 + j_idx * ldu], &ldu);
                i32 nj_col = n + j_idx;
                SLC_ZGEMM("N", "N", &n, &int2, &int2, &cone, &u[0 + nj_col * ldu],
                          &ldu, &zwork[iu_bl], &int2, &czero, &zwork[iwrk_bl], &n);
                SLC_ZLACPY("F", &n, &int2, &zwork[iwrk_bl], &n, &u[0 + nj_col * ldu], &ldu);
            }

            bwork[j_idx] = true;
            j_idx += 2;
            SLC_ZLACPY("F", &int2, &int2, &zwork[iq2], &int2, &zwork[iqb_bl], &int2);
            SLC_ZLACPY("F", &int2, &int2, &zwork[iu_bl], &int2, &zwork[iub_bl], &int2);
            iqb_bl += 4;
            iub_bl += 4;
        } else {
            bwork[j_idx] = false;
            b[(j_idx + 1) + j_idx * ldb] = czero;
            j_idx++;
        }

        if (j_idx >= j2 && j_idx <= n) {
            iqb_bl = iev_bl + 4;
            iub_bl = iqb_bl + 4 * m;

            i32 i_idx = 0;
            j1 = j2;
            j2 = (n < j1 + nb) ? n : j1 + nb;
            i32 nc = j2 - j1;

            while (i_idx < j_idx - 1) {
                if (bwork[i_idx]) {
                    SLC_ZGEMM("C", "N", &int2, &nc, &int2, &cone, &zwork[iub_bl], &int2,
                              &z[i_idx + j1 * ldz], &ldz, &czero, &zwork[iwrk_bl], &int2);
                    SLC_ZLACPY("F", &int2, &nc, &zwork[iwrk_bl], &int2, &z[i_idx + j1 * ldz], &ldz);

                    SLC_ZGEMM("C", "N", &int2, &nc, &int2, &cone, &zwork[iqb_bl], &int2,
                              &b[i_idx + j1 * ldb], &ldb, &czero, &zwork[iwrk_bl], &int2);
                    SLC_ZLACPY("F", &int2, &nc, &zwork[iwrk_bl], &int2, &b[i_idx + j1 * ldb], &ldb);

                    SLC_ZGEMM("C", "N", &int2, &nc, &int2, &cone, &zwork[iqb_bl], &int2,
                              &fg[i_idx + j1 * ldfg], &ldfg, &czero, &zwork[iwrk_bl], &int2);
                    SLC_ZLACPY("F", &int2, &nc, &zwork[iwrk_bl], &int2, &fg[i_idx + j1 * ldfg], &ldfg);
                    iqb_bl += 4;
                    iub_bl += 4;
                    i_idx += 2;
                } else {
                    i_idx++;
                }
            }
        }
    }

    j1 = 0;
    j2 = (n < nb) ? n : nb;

    while (j1 < n && j2 <= n) {
        i32 iqb_tmp = iev_bl + 4;
        i32 iub_tmp = iqb_tmp + 4 * m;

        i32 i_idx = 0;
        i32 nc = j2 - j1;

        while (i_idx < n) {
            if (bwork[i_idx]) {
                SLC_ZGEMM("C", "N", &int2, &nc, &int2, &cone, &zwork[iub_tmp], &int2,
                          &d[i_idx + j1 * ldd], &ldd, &czero, &zwork[iwrk_bl], &int2);
                SLC_ZLACPY("F", &int2, &nc, &zwork[iwrk_bl], &int2, &d[i_idx + j1 * ldd], &ldd);

                if (i_idx > j1) {
                    i32 j3_loc = (nc < i_idx - j1) ? nc : i_idx - j1;
                    SLC_ZGEMM("C", "N", &int2, &j3_loc, &int2, &cone, &zwork[iub_tmp], &int2,
                              &c[i_idx + j1 * ldc], &ldc, &czero, &zwork[iwrk_bl], &int2);
                    SLC_ZLACPY("F", &int2, &j3_loc, &zwork[iwrk_bl], &int2, &c[i_idx + j1 * ldc], &ldc);
                }
                iqb_tmp += 4;
                iub_tmp += 4;
                i_idx += 2;
            } else {
                i_idx++;
            }
        }
        j1 = j2;
        j2 = (n < j1 + nb) ? n : j1 + nb;
    }

    for (i32 i = 0; i < n; i++) {
        i32 len = i + 1;
        c128 neg_cimone = -cimone;
        SLC_ZSCAL(&len, &neg_cimone, &b[0 + i * ldb], &int1);
    }

    for (i32 i = 0; i < n; i++) {
        i32 len = i + 1;
        c128 neg_cimone = -cimone;
        SLC_ZSCAL(&len, &neg_cimone, &fg[0 + i * ldfg], &int1);
    }

    char cmpq_upd[16], cmpu_upd[16];
    if (lcmpq) {
        strcpy(cmpq_upd, "U");
    } else {
        strcpy(cmpq_upd, "N");
    }
    if (lcmpu) {
        strcpy(cmpu_upd, "U");
    } else {
        strcpy(cmpu_upd, "N");
    }

    mb03iz(cmpq_upd, cmpu_upd, n2, z, ldz, c, ldc, d, ldd, b, ldb, fg, ldfg, q,
           ldq, u, ldu, &u[0 + (n) * ldu], ldu, neig, tol, &info_mb04);

#ifdef MB03FZ_DEBUG
    fprintf(stderr, "MB03FZ: After mb03iz, neig=%d, info_mb04=%d\n", *neig, info_mb04);
    fprintf(stderr, "MB03FZ: qr=%d, qrp=%d, svd=%d\n", qr, qrp, svd);
    fprintf(stderr, "MB03FZ: ldq=%d, ldu=%d, n=%d, m=%d, n2=%d\n", ldq, ldu, n, m, n2);
    fprintf(stderr, "MB03FZ: Q array pointer: %p\n", (void*)q);
    fprintf(stderr, "MB03FZ: U array pointer: %p\n", (void*)u);
    fprintf(stderr, "MB03FZ: zwork pointer: %p, lzwork=%d\n", (void*)zwork, lzwork);
#endif

    if (qr) {
        *neig = *neig / 2;
    }
    i32 itau = 0;
    i32 iwrk = *neig;

    if (lcmpq) {
        if (*neig <= m) {
            for (i32 i = 0; i < *neig; i++) {
                SLC_ZAXPY(&m, &cimone, &q[(m) + i * ldq], &int1, &q[0 + i * ldq], &int1);
            }
            SLC_ZLACPY("F", &m, neig, &q[(n) + 0 * ldq], &ldq, &q[(m) + 0 * ldq], &ldq);
            for (i32 i = 0; i < *neig; i++) {
                SLC_ZAXPY(&m, &cimone, &q[(m + n) + i * ldq], &int1, &q[(m) + i * ldq], &int1);
            }
        } else {
            for (i32 i = 0; i < m; i++) {
                SLC_ZAXPY(&m, &cimone, &q[(m) + i * ldq], &int1, &q[0 + i * ldq], &int1);
            }
            SLC_ZLACPY("F", &m, &m, &q[(n) + 0 * ldq], &ldq, &q[(m) + 0 * ldq], &ldq);
            for (i32 i = 0; i < m; i++) {
                SLC_ZAXPY(&m, &cimone, &q[(m + n) + i * ldq], &int1, &q[(m) + i * ldq], &int1);
            }

            i32 neig_m = *neig - m;
            for (i32 i = 0; i < neig_m; i++) {
                SLC_ZAXPY(&m, &cimone, &q[(m) + (m + i) * ldq], &int1, &q[0 + (m + i) * ldq], &int1);
            }
            SLC_ZLACPY("F", &m, &neig_m, &q[(n) + (m) * ldq], &ldq, &q[(m) + (m) * ldq], &ldq);
            for (i32 i = 0; i < neig_m; i++) {
                SLC_ZAXPY(&m, &cimone, &q[(m + n) + (m + i) * ldq], &int1, &q[(m) + (m + i) * ldq], &int1);
            }
        }

        if (svd) {
            i32 info_svd;
            // Fortran passes full ZWORK array for workspace (U,VT not referenced for JOBU='O',JOBVT='N')
            SLC_ZGESVD("O", "N", &n, neig, q, &ldq, dwork, zwork, &(i32){1},
                       zwork, &(i32){1}, zwork, &lzwork,
                       &dwork[iwrk], &info_svd);
            if (info_svd > 0) {
                *info = 3;
                return;
            }
            i32 zopt = (i32)creal(zwork[0]);
            if (zopt > optzw) optzw = zopt;
            if (!lcmpu) {
                *neig = *neig / 2;
            }
        } else {
            if (qr) {
                i32 info_qr;
                SLC_ZGEQRF(&n, neig, q, &ldq, &zwork[itau], &zwork[iwrk],
                           &(i32){lzwork - iwrk}, &info_qr);
            } else {
                for (i32 jj = 0; jj < *neig; jj++) {
                    iwork[jj] = 0;
                }
                i32 info_qp;
                SLC_ZGEQP3(&n, neig, q, &ldq, iwork, zwork, &zwork[iwrk],
                           &(i32){lzwork - iwrk}, dwork, &info_qp);
            }
            i32 zopt = (i32)creal(zwork[iwrk]) + iwrk;
            if (zopt > optzw) optzw = zopt;

            i32 info_ungqr;
            SLC_ZUNGQR(&n, neig, neig, q, &ldq, &zwork[itau], &zwork[iwrk],
                       &(i32){lzwork - iwrk}, &info_ungqr);
            zopt = (i32)creal(zwork[iwrk]) + iwrk;
            if (zopt > optzw) optzw = zopt;
            if (qrp && !lcmpu) {
                *neig = *neig / 2;
            }
        }
    }

    if (lcmpu) {
        if (*neig <= m) {
            for (i32 i = 0; i < *neig; i++) {
                SLC_ZAXPY(&m, &cimone, &u[(m) + i * ldu], &int1, &u[0 + i * ldu], &int1);
            }
            SLC_ZLACPY("F", &m, neig, &u[0 + (n) * ldu], &ldu, &u[(m) + 0 * ldu], &ldu);
            for (i32 i = 0; i < *neig; i++) {
                SLC_ZAXPY(&m, &cimone, &u[(m) + (n + i) * ldu], &int1, &u[(m) + i * ldu], &int1);
            }
        } else {
            for (i32 i = 0; i < m; i++) {
                SLC_ZAXPY(&m, &cimone, &u[(m) + i * ldu], &int1, &u[0 + i * ldu], &int1);
            }
            SLC_ZLACPY("F", &m, neig, &u[0 + (n) * ldu], &ldu, &u[(m) + 0 * ldu], &ldu);
            for (i32 i = 0; i < m; i++) {
                SLC_ZAXPY(&m, &cimone, &u[(m) + (n + i) * ldu], &int1, &u[(m) + i * ldu], &int1);
            }

            i32 neig_m = *neig - m;
            for (i32 i = 0; i < neig_m; i++) {
                SLC_ZAXPY(&m, &cimone, &u[(m) + (m + i) * ldu], &int1, &u[0 + (m + i) * ldu], &int1);
            }
            SLC_ZLACPY("F", &m, &neig_m, &u[0 + (n + m) * ldu], &ldu, &u[(m) + (m) * ldu], &ldu);
            for (i32 i = 0; i < neig_m; i++) {
                SLC_ZAXPY(&m, &cimone, &u[(m) + (n + m + i) * ldu], &int1, &u[(m) + (m + i) * ldu], &int1);
            }
        }

        for (i32 jj = 0; jj < *neig; jj++) {
            for (i32 ii = m; ii < n; ii++) {
                u[ii + jj * ldu] = -u[ii + jj * ldu];
            }
        }

        if (svd) {
            i32 info_svd;
            // Fortran passes full ZWORK array for workspace (U,VT not referenced for JOBU='O',JOBVT='N')
            SLC_ZGESVD("O", "N", &n, neig, u, &ldu, dwork, zwork, &(i32){1},
                       zwork, &(i32){1}, zwork, &lzwork,
                       &dwork[iwrk], &info_svd);
            if (info_svd > 0) {
                *info = 3;
                return;
            }
            i32 zopt = (i32)creal(zwork[0]);
            if (zopt > optzw) optzw = zopt;
            *neig = *neig / 2;
        } else {
            if (qr) {
                i32 info_qr;
                SLC_ZGEQRF(&n, neig, u, &ldu, &zwork[itau], &zwork[iwrk],
                           &(i32){lzwork - iwrk}, &info_qr);
            } else {
                for (i32 jj = 0; jj < *neig; jj++) {
                    iwork[jj] = 0;
                }
                i32 info_qp;
                SLC_ZGEQP3(&n, neig, u, &ldu, iwork, zwork, &zwork[iwrk],
                           &(i32){lzwork - iwrk}, dwork, &info_qp);
            }
            i32 zopt = (i32)creal(zwork[iwrk]) + iwrk;
            if (zopt > optzw) optzw = zopt;

            i32 info_ungqr;
            SLC_ZUNGQR(&n, neig, neig, u, &ldu, &zwork[itau], &zwork[iwrk],
                       &(i32){lzwork - iwrk}, &info_ungqr);
            zopt = (i32)creal(zwork[iwrk]) + iwrk;
            if (zopt > optzw) optzw = zopt;
            if (qrp) {
                *neig = *neig / 2;
            }
        }
    }

    dwork[0] = (f64)optdw;
    zwork[0] = (c128)optzw;
}
