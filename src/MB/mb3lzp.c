// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

void mb3lzp(const char *compq, const char *orth, i32 n,
            c128 *a, i32 lda, c128 *de, i32 ldde, c128 *b, i32 ldb,
            c128 *fg, i32 ldfg, i32 *neig, c128 *q, i32 ldq,
            f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, f64 *dwork, i32 ldwork, c128 *zwork, i32 lzwork,
            bool *bwork, i32 *info) {
    const f64 one = 1.0;
    const c128 czero = 0.0 + 0.0 * I;
    const c128 cone = 1.0 + 0.0 * I;
    const c128 cimag_unit = 0.0 + 1.0 * I;

    i32 nbl = *info;
    i32 m = n / 2;
    i32 nn = n * n;
    i32 n2 = 2 * n;
    *neig = 0;

    bool lcmpq = (compq[0] == 'C' || compq[0] == 'c');
    bool qr = false, qrp = false, svd = false;

    if (lcmpq) {
        qr = (orth[0] == 'Q' || orth[0] == 'q');
        qrp = (orth[0] == 'P' || orth[0] == 'p');
        svd = (orth[0] == 'S' || orth[0] == 's');
    }

    i32 mindw, minzw;
    if (n == 0) {
        mindw = 1;
        minzw = 1;
    } else if (lcmpq) {
        mindw = 11 * nn + 2 * n;
        minzw = 8 * n + 4;
    } else {
        i32 maxn3 = (n > 3) ? n : 3;
        mindw = 4 * nn + 2 * n + maxn3;
        minzw = 1;
    }

    bool lquery = (ldwork == -1) || (lzwork == -1);

    *info = 0;
    bool compq_n = (compq[0] == 'N' || compq[0] == 'n');

    if (!compq_n && !lcmpq) {
        *info = -1;
    } else if (lcmpq && !(qr || qrp || svd)) {
        *info = -2;
    } else if (n < 0 || (n % 2) != 0) {
        *info = -3;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -5;
    } else if (ldde < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (ldfg < (n > 1 ? n : 1)) {
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

    if (lquery) {
        dwork[0] = (f64)mindw;
        zwork[0] = (c128)minzw;
        return;
    }

    if (n == 0) {
        dwork[0] = one;
        zwork[0] = cone;
        return;
    }

    f64 eps = SLC_DLAMCH("P");
    f64 tol = sqrt(eps);

    i32 iq, ia, ide, ib, ifg, iwrk;
    i32 nb = 2;

    if (lcmpq) {
        iq = 0;
        ia = iq + n2 * n2;
    } else {
        iq = 0;
        ia = 0;
    }
    ide = ia + nn;
    ib = ide + nn + n;
    ifg = ib + nn;
    iwrk = ifg + nn + n;

    // Build the embedding of A
    i32 iw = ia;
    i32 is_off = iw + n * m;
    for (i32 j = 0; j < m; j++) {
        i32 iw1 = iw;
        for (i32 i = 0; i < m; i++) {
            dwork[iw] = creal(a[i + j * lda]);
            iw++;
        }
        for (i32 i = 0; i < m; i++) {
            dwork[iw] = cimag(a[i + j * lda]);
            dwork[is_off] = -dwork[iw];
            iw++;
            is_off++;
        }
        SLC_DCOPY(&m, &dwork[iw1], &(i32){1}, &dwork[is_off], &(i32){1});
        is_off += m;
    }

    // Build the embedding of D and E
    iw = ide;
    for (i32 j = 0; j < m + 1; j++) {
        for (i32 i = 0; i < m; i++) {
            dwork[iw] = creal(de[i + j * ldde]);
            iw++;
        }
        iw += j;
        is_off = iw;
        for (i32 i = j; i < m; i++) {
            dwork[iw] = cimag(de[i + j * ldde]);
            dwork[is_off] = dwork[iw];
            iw++;
            is_off += n;
        }
    }

    i32 iw1 = iw;
    i32 i1 = iw;
    for (i32 j = 1; j < m + 1; j++) {
        is_off = i1;
        i1++;
        for (i32 i = 0; i < j; i++) {
            dwork[iw] = -cimag(de[i + j * ldde]);
            dwork[is_off] = dwork[iw];
            iw++;
            is_off += n;
        }
        iw += n - j;
    }
    i32 mp1 = m + 1;
    SLC_DLACPY("F", &m, &mp1, &dwork[ide], &n, &dwork[iw1 - m], &n);

    // Build the embedding of B
    iw = ib;
    is_off = iw + n * m;
    for (i32 j = 0; j < m; j++) {
        iw1 = iw;
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

    // Build the embedding of F and G
    iw = ifg;
    for (i32 j = 0; j < m + 1; j++) {
        for (i32 i = 0; i < m; i++) {
            dwork[iw] = -cimag(fg[i + j * ldfg]);
            iw++;
        }
        iw += j;
        is_off = iw;
        for (i32 i = j; i < m; i++) {
            dwork[iw] = creal(fg[i + j * ldfg]);
            dwork[is_off] = dwork[iw];
            iw++;
            is_off += n;
        }
    }

    iw1 = iw;
    i1 = iw;
    for (i32 j = 1; j < m + 1; j++) {
        is_off = i1;
        i1++;
        for (i32 i = 0; i < j; i++) {
            dwork[iw] = -creal(fg[i + j * ldfg]);
            dwork[is_off] = dwork[iw];
            iw++;
            is_off += n;
        }
        iw += n - j;
    }
    SLC_DLACPY("F", &m, &mp1, &dwork[ifg], &n, &dwork[iw1 - m], &n);

    // STEP 1: Apply MB04FP for structured Schur form
    const char *job = lcmpq ? "T" : "E";
    const char *cmpq_str = lcmpq ? "I" : "N";

    *info = nbl;
    i32 ldwork_call = ldwork - iwrk;
    mb04fp(job, cmpq_str, n2, &dwork[ia], n, &dwork[ide], n, &dwork[ib], n,
           &dwork[ifg], n, &dwork[iq], n2, alphai, alphar, beta, iwork,
           &dwork[iwrk], ldwork_call, info);

    i32 iwa = 0;
    if (*info == 2) {
        iwa = 4;
    } else if (*info > 0) {
        return;
    }

    // Scale eigenvalues
    SLC_DSCAL(&n, &(f64){-one}, alphai, &(i32){1});

    if (!lcmpq) {
        dwork[0] = (f64)mindw;
        zwork[0] = (c128)minzw;
        *info = iwa;
        return;
    }

    // Convert results to complex datatype
    iw = ia;
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i <= j; i++) {
            a[i + j * lda] = dwork[iw] + 0.0 * I;
            iw++;
        }
        if (j >= m - 1 && j < n - 1) {
            a[j + 1 + j * lda] = czero;
        }
        iw += n - j - 1;
    }

    iw = ide + n;
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < j; i++) {
            de[i + j * ldde] = dwork[iw] + 0.0 * I;
            iw++;
        }
        de[j + j * ldde] = czero;
        if (j >= m - 1 && j < n - 1) {
            de[j + 1 + j * ldde] = czero;
        }
        iw += n - j;
    }

    iw = ib;
    for (i32 j = 0; j < n; j++) {
        i32 max_i = (j + 2 < n) ? j + 2 : n;
        for (i32 i = 0; i < max_i; i++) {
            b[i + j * ldb] = dwork[iw] + 0.0 * I;
            iw++;
        }
        iw += n - j - 2;
    }

    iw = ifg + n;
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < j; i++) {
            fg[i + j * ldfg] = dwork[iw] + 0.0 * I;
            iw++;
        }
        fg[j + j * ldfg] = czero;
        if (j >= m - 1 && j < n - 1) {
            fg[j + 1 + j * ldfg] = czero;
        }
        iw += n - j;
    }

    iw = iq;
    for (i32 j = 0; j < n2; j++) {
        for (i32 i = 0; i < n2; i++) {
            q[i + j * ldq] = dwork[iw] + 0.0 * I;
            iw++;
        }
    }

    // Triangularize 2-by-2 diagonal blocks in B using ZHGEQZ
    i32 iq2 = 0;
    i32 iev = 4;
    i32 iqz = 8;
    i32 izwrk = iqz + 4 * (n - 1);

    i32 jj = 0;
    i32 j1 = 0;
    i32 j2 = (n < j1 + nb) ? n : j1 + nb;

    while (jj < n - 1) {
        f64 nrmb = cabs(b[jj + jj * ldb]) + cabs(b[jj + 1 + (jj + 1) * ldb]);
        if (cabs(b[jj + 1 + jj * ldb]) > nrmb * eps) {
            i32 nc = (j2 - jj - 2 > 0) ? j2 - jj - 2 : 0;
            i32 jm1 = (jj > 0) ? jj : 0;
            i32 jp2 = (jj + 2 < n) ? jj + 2 : n;

            c128 tmp = a[jj + 1 + jj * lda];
            a[jj + 1 + jj * lda] = czero;

            i32 two = 2, int1 = 1, info_qz;
            i32 lzwork_call = lzwork - izwrk;
            SLC_ZHGEQZ("S", "I", "I", &two, &int1, &two,
                       &b[jj + jj * ldb], &ldb, &a[jj + jj * lda], &lda,
                       &zwork[iev], &zwork[iev + 2], &zwork[iqz], &two,
                       &zwork[iq2], &two, &zwork[izwrk], &lzwork_call,
                       dwork, &info_qz);

            a[jj + 1 + jj * lda] = tmp;
            if (info_qz > 0) {
                *info = 2;
                return;
            }

            // Update A
            if (jm1 > 0) {
                SLC_ZGEMM("N", "N", &jm1, &two, &two, &cone, &a[jj * lda], &lda,
                          &zwork[iq2], &two, &czero, &zwork[izwrk], &jm1);
                SLC_ZLACPY("F", &jm1, &two, &zwork[izwrk], &jm1, &a[jj * lda], &lda);
            }

            if (nc > 0) {
                SLC_ZGEMM("C", "N", &two, &nc, &two, &cone, &zwork[iqz], &two,
                          &a[jj + jp2 * lda], &lda, &czero, &zwork[izwrk], &two);
                SLC_ZLACPY("F", &two, &nc, &zwork[izwrk], &two, &a[jj + jp2 * lda], &lda);
            }

            // Update DE
            tmp = de[jj + 1 + jj * ldde];
            de[jj + 1 + jj * ldde] = -de[jj + (jj + 1) * ldde];
            i32 jp1 = jj + 2;
            if (jp1 > 0) {
                SLC_ZGEMM("N", "N", &jp1, &two, &two, &cone, &de[jj * ldde], &ldde,
                          &zwork[iqz], &two, &czero, &zwork[izwrk], &jp1);
                SLC_ZLACPY("F", &jp1, &two, &zwork[izwrk], &jp1, &de[jj * ldde], &ldde);
            }

            i32 nc_de = j2 - jj;
            SLC_ZGEMM("C", "N", &two, &nc_de, &two, &cone, &zwork[iqz], &two,
                      &de[jj + jj * ldde], &ldde, &czero, &zwork[izwrk], &two);
            SLC_ZLACPY("F", &two, &nc_de, &zwork[izwrk], &two, &de[jj + jj * ldde], &ldde);
            de[jj + 1 + jj * ldde] = tmp;

            // Update B
            if (jm1 > 0) {
                SLC_ZGEMM("N", "N", &jm1, &two, &two, &cone, &b[jj * ldb], &ldb,
                          &zwork[iq2], &two, &czero, &zwork[izwrk], &jm1);
                SLC_ZLACPY("F", &jm1, &two, &zwork[izwrk], &jm1, &b[jj * ldb], &ldb);
            }

            if (nc > 0) {
                SLC_ZGEMM("C", "N", &two, &nc, &two, &cone, &zwork[iqz], &two,
                          &b[jj + jp2 * ldb], &ldb, &czero, &zwork[izwrk], &two);
                SLC_ZLACPY("F", &two, &nc, &zwork[izwrk], &two, &b[jj + jp2 * ldb], &ldb);
            }

            // Update FG
            tmp = fg[jj + 1 + jj * ldfg];
            fg[jj + 1 + jj * ldfg] = -fg[jj + (jj + 1) * ldfg];
            if (jp1 > 0) {
                SLC_ZGEMM("N", "N", &jp1, &two, &two, &cone, &fg[jj * ldfg], &ldfg,
                          &zwork[iqz], &two, &czero, &zwork[izwrk], &jp1);
                SLC_ZLACPY("F", &jp1, &two, &zwork[izwrk], &jp1, &fg[jj * ldfg], &ldfg);
            }

            SLC_ZGEMM("C", "N", &two, &nc_de, &two, &cone, &zwork[iqz], &two,
                      &fg[jj + jj * ldfg], &ldfg, &czero, &zwork[izwrk], &two);
            SLC_ZLACPY("F", &two, &nc_de, &zwork[izwrk], &two, &fg[jj + jj * ldfg], &ldfg);
            fg[jj + 1 + jj * ldfg] = tmp;

            // Update Q
            SLC_ZGEMM("N", "N", &n2, &two, &two, &cone, &q[jj * ldq], &ldq,
                      &zwork[iq2], &two, &czero, &zwork[izwrk], &n2);
            SLC_ZLACPY("F", &n2, &two, &zwork[izwrk], &n2, &q[jj * ldq], &ldq);

            SLC_ZGEMM("N", "N", &n2, &two, &two, &cone, &q[(n + jj) * ldq], &ldq,
                      &zwork[iqz], &two, &czero, &zwork[izwrk], &n2);
            SLC_ZLACPY("F", &n2, &two, &zwork[izwrk], &n2, &q[(n + jj) * ldq], &ldq);

            bwork[jj] = true;
            jj += 2;
            iqz += 4;
        } else {
            bwork[jj] = false;
            b[jj + 1 + jj * ldb] = czero;
            jj += 1;
        }

        if (jj >= j2 - 1) {
            j1 = j2;
            j2 = (n < j1 + nb) ? n : j1 + nb;
            i32 nc_upd = j2 - j1;

            // Update columns j1 to j2 of A, DE, B, FG for previous transforms
            i32 ii = 0;
            i32 iqb = 8;
            while (ii < jj - 1) {
                if (bwork[ii]) {
                    if (nc_upd > 0) {
                        SLC_ZGEMM("C", "N", &(i32){2}, &nc_upd, &(i32){2}, &cone,
                                  &zwork[iqb], &(i32){2}, &a[ii + j1 * lda], &lda,
                                  &czero, &zwork[izwrk], &(i32){2});
                        SLC_ZLACPY("F", &(i32){2}, &nc_upd, &zwork[izwrk], &(i32){2},
                                   &a[ii + j1 * lda], &lda);

                        SLC_ZGEMM("C", "N", &(i32){2}, &nc_upd, &(i32){2}, &cone,
                                  &zwork[iqb], &(i32){2}, &de[ii + j1 * ldde], &ldde,
                                  &czero, &zwork[izwrk], &(i32){2});
                        SLC_ZLACPY("F", &(i32){2}, &nc_upd, &zwork[izwrk], &(i32){2},
                                   &de[ii + j1 * ldde], &ldde);

                        SLC_ZGEMM("C", "N", &(i32){2}, &nc_upd, &(i32){2}, &cone,
                                  &zwork[iqb], &(i32){2}, &b[ii + j1 * ldb], &ldb,
                                  &czero, &zwork[izwrk], &(i32){2});
                        SLC_ZLACPY("F", &(i32){2}, &nc_upd, &zwork[izwrk], &(i32){2},
                                   &b[ii + j1 * ldb], &ldb);

                        SLC_ZGEMM("C", "N", &(i32){2}, &nc_upd, &(i32){2}, &cone,
                                  &zwork[iqb], &(i32){2}, &fg[ii + j1 * ldfg], &ldfg,
                                  &czero, &zwork[izwrk], &(i32){2});
                        SLC_ZLACPY("F", &(i32){2}, &nc_upd, &zwork[izwrk], &(i32){2},
                                   &fg[ii + j1 * ldfg], &ldfg);
                    }
                    iqb += 4;
                    ii += 2;
                } else {
                    ii += 1;
                }
            }
        }
    }

    // Scale B and FG by -i
    for (i32 ii = 0; ii < n; ii++) {
        i32 len = ii + 1;
        SLC_ZSCAL(&len, &(c128){-cimag_unit}, &b[ii * ldb], &(i32){1});
    }
    for (i32 ii = 0; ii < n; ii++) {
        i32 len = ii + 1;
        SLC_ZSCAL(&len, &(c128){-cimag_unit}, &fg[ii * ldfg], &(i32){1});
    }

    // STEP 2: Apply MB3JZP for eigenvalue reordering
    const char *cmpq_upd = "U";
    *info = nbl;
    mb3jzp(cmpq_upd, n2, a, lda, de, ldde, b, ldb, fg, ldfg, q, ldq,
           neig, tol, dwork, zwork, info);

    i32 neig_val = *neig;
    if (qr) neig_val = neig_val / 2;

    i32 itau = 0;
    izwrk = neig_val;

    // STEP 3: Compute deflating subspace
    if (neig_val <= m) {
        for (i32 ii = 0; ii < neig_val; ii++) {
            SLC_ZAXPY(&m, &cimag_unit, &q[m + ii * ldq], &(i32){1},
                      &q[ii * ldq], &(i32){1});
        }
        SLC_ZLACPY("F", &m, &neig_val, &q[n], &ldq, &q[m], &ldq);
        for (i32 ii = 0; ii < neig_val; ii++) {
            SLC_ZAXPY(&m, &cimag_unit, &q[m + n + ii * ldq], &(i32){1},
                      &q[m + ii * ldq], &(i32){1});
        }
    } else {
        for (i32 ii = 0; ii < m; ii++) {
            SLC_ZAXPY(&m, &cimag_unit, &q[m + ii * ldq], &(i32){1},
                      &q[ii * ldq], &(i32){1});
        }
        SLC_ZLACPY("F", &m, &m, &q[n], &ldq, &q[m], &ldq);
        for (i32 ii = 0; ii < m; ii++) {
            SLC_ZAXPY(&m, &cimag_unit, &q[m + n + ii * ldq], &(i32){1},
                      &q[m + ii * ldq], &(i32){1});
        }

        i32 nm = neig_val - m;
        for (i32 ii = 0; ii < nm; ii++) {
            SLC_ZAXPY(&m, &cimag_unit, &q[m + (m + ii) * ldq], &(i32){1},
                      &q[(m + ii) * ldq], &(i32){1});
        }
        SLC_ZLACPY("F", &m, &nm, &q[n + m * ldq], &ldq, &q[m + m * ldq], &ldq);
        for (i32 ii = 0; ii < nm; ii++) {
            SLC_ZAXPY(&m, &cimag_unit, &q[m + n + (m + ii) * ldq], &(i32){1},
                      &q[m + (m + ii) * ldq], &(i32){1});
        }
    }

    // Orthogonalize the basis
    i32 info_orth;
    if (svd) {
        i32 lzwork_call = lzwork;
        SLC_ZGESVD("O", "N", &n, &neig_val, q, &ldq, dwork, zwork, &(i32){1},
                   zwork, &(i32){1}, zwork, &lzwork_call, &dwork[neig_val], &info_orth);
        if (info_orth > 0) {
            *info = 3;
            return;
        }
        neig_val = neig_val / 2;
    } else {
        if (qr) {
            i32 lzwork_call = lzwork - izwrk;
            SLC_ZGEQRF(&n, &neig_val, q, &ldq, &zwork[itau], &zwork[izwrk],
                       &lzwork_call, &info_orth);
        } else {
            for (i32 jj = 0; jj < neig_val; jj++) {
                iwork[jj] = 0;
            }
            i32 lzwork_call = lzwork - izwrk;
            SLC_ZGEQP3(&n, &neig_val, q, &ldq, iwork, zwork, &zwork[izwrk],
                       &lzwork_call, dwork, &info_orth);
        }

        i32 lzwork_call = lzwork - izwrk;
        SLC_ZUNGQR(&n, &neig_val, &neig_val, q, &ldq, &zwork[itau],
                   &zwork[izwrk], &lzwork_call, &info_orth);

        if (qrp) neig_val = neig_val / 2;
    }

    *neig = neig_val;
    dwork[0] = (f64)mindw;
    zwork[0] = (c128)minzw;
    *info = iwa;
}
