/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB03LF - Eigenvalues and deflating subspace of skew-Hamiltonian/Hamiltonian pencil
 *
 * Computes the relevant eigenvalues of a real N-by-N skew-Hamiltonian/Hamiltonian
 * pencil aS - bH in factored form, with optional computation of the right
 * deflating subspace and companion subspace.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void mb03lf(const char *compq, const char *compu, const char *orth,
            i32 n, f64 *z, i32 ldz, const f64 *b, i32 ldb,
            const f64 *fg, i32 ldfg, i32 *neig, f64 *q, i32 ldq,
            f64 *u, i32 ldu, f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork,
            bool *bwork, i32 *iwarn, i32 *info)
{
    const f64 ZERO = 0.0, ONE = 1.0, TWO = 2.0;
    i32 int1 = 1;

    char compq_c = (char)toupper((unsigned char)compq[0]);
    char compu_c = (char)toupper((unsigned char)compu[0]);
    char orth_c = (char)toupper((unsigned char)orth[0]);

    i32 m = n / 2;
    i32 n2 = n * 2;
    i32 nn = n * n;

    *neig = 0;

    bool lcmpq = (compq_c == 'C');
    bool lcmpu = (compu_c == 'C');
    bool lcmp = lcmpq || lcmpu;
    bool qr_flag = false, qrp = false, svd = false;

    i32 miniw, mindw;
    i32 i_coef = 0, j_coef = 0, mindb = 0;

    if (lcmp) {
        qr_flag = (orth_c == 'Q');
        qrp = (orth_c == 'P');
        svd = (orth_c == 'S');
        miniw = (n2 + 1 > 48) ? n2 + 1 : 48;
    } else {
        miniw = n + 18;
    }

    if (n == 0) {
        miniw = 1;
        mindw = 1;
    } else {
        if (lcmpq) {
            i_coef = 4;
            j_coef = 10;
        } else {
            i_coef = 2;
            j_coef = 7;
        }
        if (lcmpu) {
            i_coef += 1;
            j_coef += 1;
        }
        mindb = i_coef * nn;
        if (lcmp) {
            i32 temp = (m + 252 > 432) ? m + 252 : 432;
            mindw = j_coef * nn + temp;
        } else {
            i32 temp = (6 * n > 54) ? 6 * n : 54;
            mindw = mindb + 3 * (nn / 2) + temp;
        }
    }

    bool lquery = (ldwork == -1);

    *info = 0;

    if (compq_c != 'N' && compq_c != 'C') {
        *info = -1;
    } else if (compu_c != 'N' && compu_c != 'C') {
        *info = -2;
    } else if (lcmp && !(qr_flag || qrp || svd)) {
        *info = -3;
    } else if (n < 0 || (n % 2) != 0) {
        *info = -4;
    } else if (ldz < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldb < (m > 1 ? m : 1)) {
        *info = -8;
    } else if (ldfg < (m > 1 ? m : 1)) {
        *info = -10;
    } else if (ldq < 1 || (lcmpq && ldq < n2)) {
        *info = -13;
    } else if (ldu < 1 || (lcmpu && ldu < n)) {
        *info = -15;
    } else if (liwork < miniw) {
        iwork[0] = miniw;
        *info = -20;
    } else if (!lquery && ldwork < mindw) {
        dwork[0] = (f64)mindw;
        *info = -22;
    }

    if (*info != 0) {
        return;
    }

    i32 optdw;

    if (n > 0) {
        const char *cmpi = "Initialize";
        const char *cmqi = lcmpq ? cmpi : "No Computation";
        const char *cmui = lcmpu ? cmpi : "No Computation";
        const char *job = lcmp ? "Triangularize" : "Eigenvalues";

        if (lquery) {
            f64 dum[7];
            mb04ad(job, cmqi, cmqi, cmui, cmui, n, dwork, n, dwork, n,
                   dwork, n, dwork, n, dwork, m, dwork, m, dwork, m,
                   dwork, m, dwork, n, dwork, dwork, dwork, iwork, liwork,
                   dum, -1, info);

            if (lcmp) {
                i32 iw = i_coef * nn;
                i32 mindb_q = iw + 2 * nn;
                if (lcmpq) {
                    iw += nn;
                    mindb_q += nn;
                } else {
                    iw = 0;
                }

                mb04cd(cmqi, cmpi, cmpi, n, dwork, n, dwork, n, dwork, n,
                       dwork, n, dwork, n, dwork, n, iwork, liwork, &dum[1],
                       -1, bwork, info);

                i32 j_temp;
                if (svd) {
                    SLC_DGESVD("O", "N", &n, &n, q, &ldq, dwork, dwork, &ldq,
                               dwork, &int1, &dum[2], &int1, info);
                    j_temp = n + (i32)dum[2];
                } else {
                    i32 max_ld = (ldq > ldu) ? ldq : ldu;
                    if (qr_flag) {
                        SLC_DGEQRF(&n, &m, q, &max_ld, dwork, &dum[2], &int1, info);
                        j_temp = m;
                    } else {
                        SLC_DGEQP3(&n, &n, q, &max_ld, iwork, dwork, &dum[2], &int1, info);
                        j_temp = n;
                    }
                    SLC_DORGQR(&n, &j_temp, &j_temp, q, &max_ld, dwork, &dum[3], &int1, info);
                    i32 max_dum = ((i32)dum[2] > (i32)dum[3]) ? (i32)dum[2] : (i32)dum[3];
                    j_temp = j_temp + iw + max_dum;
                }
                i32 temp1 = mindb_q + (i32)dum[1];
                optdw = (temp1 > j_temp) ? temp1 : j_temp;
            } else {
                optdw = 0;
            }
            i32 temp2 = mindb + (i32)dum[0];
            optdw = (mindw > optdw) ? mindw : optdw;
            optdw = (optdw > temp2) ? optdw : temp2;
            dwork[0] = (f64)optdw;
            return;
        }
    }

    *iwarn = 0;
    if (n == 0) {
        dwork[0] = ONE;
        return;
    }

    i32 mm = m * m;
    i32 nm = n * m;
    i32 nmm = nm + m;
    i32 np1 = n + 1;
    i32 mp1 = m + 1;

    i32 iq1, iq2, iu11, iu12, iu21, iu22, ih;

    iq1 = 0;
    if (lcmpq) {
        iq2 = iq1 + nn;
        iu11 = iq2 + nn;
        ih = iu11;
    } else {
        iq2 = 0;
        iu11 = 0;
        ih = 0;
    }

    if (lcmpu) {
        iu12 = iu11 + mm;
        iu21 = iu12 + mm;
        iu22 = iu21 + mm;
        ih = iu22 + mm;
    } else {
        iu12 = 0;
        iu21 = 0;
        iu22 = 0;
    }

    i32 iw = ih;
    i32 is_idx = ih + m + n;
    for (i32 j = 0; j < m; j++) {
        SLC_DCOPY(&m, &b[j * ldb], &int1, &dwork[iw], &int1);
        iw += m + j;
        i32 cnt = m - j;
        SLC_DCOPY(&cnt, &fg[j + j * ldfg], &int1, &dwork[iw], &int1);
        i32 cnt2 = m - j - 1;
        if (cnt2 > 0) {
            SLC_DCOPY(&cnt2, &dwork[iw + 1], &int1, &dwork[is_idx], &n);
        }
        iw += m - j;
        is_idx += np1;
    }

    iw = ih + nm;
    is_idx = iw;
    for (i32 j = 0; j < m; j++) {
        i32 cnt = j + 1;
        SLC_DCOPY(&cnt, &fg[(j + 1) * ldfg], &int1, &dwork[iw], &int1);
        if (j > 0) {
            SLC_DCOPY(&j, &dwork[iw], &int1, &dwork[is_idx], &n);
        }
        iw += m;
        is_idx += 1;
        for (i32 i = 0; i < m; i++) {
            dwork[iw] = -b[j + i * ldb];
            iw++;
        }
    }

    i32 it = ih + nn;
    i32 iwrk = it + nn;

    const char *job_str = lcmp ? "Triangularize" : "Eigenvalues";
    const char *cmpi_str = "Initialize";
    const char *cmqi_str = lcmpq ? cmpi_str : "No Computation";
    const char *cmui_str = lcmpu ? cmpi_str : "No Computation";

    mb04ad(job_str, cmqi_str, cmqi_str, cmui_str, cmui_str, n, z, ldz,
           &dwork[ih], n, &dwork[iq1], n, &dwork[iq2], n, &dwork[iu11], m,
           &dwork[iu12], m, &dwork[iu21], m, &dwork[iu22], m, &dwork[it], n,
           alphar, alphai, beta, iwork, liwork, &dwork[iwrk], ldwork - iwrk, info);

    if (*info == 3) {
        *iwarn = 1;
    } else if (*info > 0) {
        *info = 1;
        return;
    }
    optdw = (mindw > (i32)dwork[iwrk] + iwrk) ? mindw : (i32)dwork[iwrk] + iwrk;

    if (!lcmp) {
        dwork[0] = (f64)optdw;
        return;
    }

    iw = it + nm;
    is_idx = ih + nm;
    if (lcmpu) {
        for (i32 j = 0; j < m; j++) {
            SLC_DCOPY(&m, &z[(m + j) * ldz], &int1, &u[j * ldu], &int1);
            SLC_DCOPY(&m, &dwork[iw], &int1, &u[m + j * ldu], &int1);
            iw += n;
        }
        for (i32 j = 0; j < m; j++) {
            SLC_DCOPY(&m, &dwork[is_idx], &int1, &u[(m + j) * ldu], &int1);
            is_idx += n;
        }
    } else {
        for (i32 j = 0; j < m; j++) {
            SLC_DCOPY(&m, &z[(m + j) * ldz], &int1, &q[j * ldq], &int1);
            SLC_DCOPY(&m, &dwork[iw], &int1, &q[m + j * ldq], &int1);
            iw += n;
        }
        for (i32 j = 0; j < m; j++) {
            SLC_DCOPY(&m, &dwork[is_idx], &int1, &q[(m + j) * ldq], &int1);
            is_idx += n;
        }
    }

    i32 iu3 = iwrk;
    i32 iq3 = iu3 + nn;
    i32 iq4 = iq3 + nn;
    if (lcmpq) {
        iwrk = iq4 + nn;
    } else {
        iwrk = iq4;
    }

    iw = ih;
    for (i32 j = 0; j < m; j++) {
        is_idx = iw + nm;
        SLC_DCOPY(&m, &dwork[iw], &int1, &dwork[is_idx], &int1);
        iw += m;
        is_idx = ih + nmm + j;
        for (i32 i = 0; i < m; i++) {
            dwork[iw] = -dwork[is_idx];
            iw++;
            is_idx += n;
        }
    }

    is_idx = it + nmm;
    for (i32 j = 0; j < m; j++) {
        SLC_DSWAP(&m, &z[j * ldz], &int1, &z[mp1 - 1 + (m + j) * ldz], &int1);
        SLC_DSWAP(&m, &z[j * ldz], &int1, &dwork[is_idx], &n);
        is_idx++;
    }

    mb04cd(cmqi_str, cmpi_str, cmpi_str, n, &dwork[it], n, z, ldz, &dwork[ih], n,
           &dwork[iq4], n, &dwork[iu3], n, &dwork[iq3], n, iwork, liwork,
           &dwork[iwrk], ldwork - iwrk, bwork, info);

    if (getenv("MB03LF_DEBUG") != NULL) {
        fprintf(stderr, "MB03LF: After mb04cd, B matrix (dwork[ih]) subdiagonals:\n");
        for (i32 ii = 0; ii < n - 1; ii++) {
            i32 bidx = ih + (ii + 1) + ii * n;
            fprintf(stderr, "  B(%d,%d) = %.16e\n", ii + 2, ii + 1, dwork[bidx]);
        }
    }

    if (*info > 0) {
        if (*info > 2) {
            *info = 2;
        }
        return;
    }
    i32 temp_optdw = (i32)dwork[iwrk] + iwrk;
    optdw = (optdw > temp_optdw) ? optdw : temp_optdw;

    i32 iz12 = iwrk;
    i32 ih12 = iz12 + nn;
    iwrk = ih12 + nn;

    f64 neg_one = -ONE;
    if (lcmpu) {
        SLC_DGEMM("T", "N", &m, &n, &m, &neg_one, &u[mp1 - 1], &ldu,
                  &dwork[iq3], &n, &ZERO, &dwork[ih12], &n);
        SLC_DGEMM("N", "N", &m, &n, &m, &ONE, u, &ldu,
                  &dwork[iq3 + m], &n, &ZERO, &dwork[ih12 + m], &n);
    } else {
        SLC_DGEMM("T", "N", &m, &n, &m, &neg_one, &q[mp1 - 1], &ldq,
                  &dwork[iq3], &n, &ZERO, &dwork[ih12], &n);
        SLC_DGEMM("N", "N", &m, &n, &m, &ONE, q, &ldq,
                  &dwork[iq3 + m], &n, &ZERO, &dwork[ih12 + m], &n);
    }
    SLC_DGEMM("T", "N", &n, &n, &n, &ONE, &dwork[iu3], &n,
              &dwork[ih12], &n, &ZERO, &dwork[iz12], &n);

    if (lcmpu) {
        SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &u[mp1 * ldu], &ldu,
                  &dwork[iq3 + m], &n, &ZERO, &z[mp1 - 1], &ldz);
    } else {
        SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q[mp1 * ldq], &ldq,
                  &dwork[iq3 + m], &n, &ZERO, &z[mp1 - 1], &ldz);
    }
    SLC_DGEMM("T", "N", &m, &m, &m, &ONE, &dwork[iq3], &n,
              &z[mp1 - 1], &ldz, &ZERO, &dwork[ih12], &n);

    is_idx = 0;
    for (i32 j = 0; j < m; j++) {
        i32 cnt = j + 1;
        SLC_DAXPY(&cnt, &ONE, &dwork[ih12 + j], &n, &dwork[ih12 + is_idx], &int1);
        is_idx += n;
    }

    SLC_DGEMM("T", "N", &m, &m, &m, &ONE, &z[mp1 - 1], &ldz,
              &dwork[iq3 + nm], &n, &ZERO, &dwork[ih12 + nm], &n);

    if (lcmpu) {
        SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &u[mp1 * ldu], &ldu,
                  &dwork[iq3 + nmm], &n, &ZERO, &z[mp1 - 1], &ldz);
    } else {
        SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q[mp1 * ldq], &ldq,
                  &dwork[iq3 + nmm], &n, &ZERO, &z[mp1 - 1], &ldz);
    }
    SLC_DGEMM("T", "N", &m, &m, &m, &ONE, &dwork[iq3], &n,
              &z[mp1 - 1], &ldz, &ONE, &dwork[ih12 + nm], &n);

    SLC_DGEMM("T", "N", &m, &m, &m, &ONE, &z[mp1 - 1], &ldz,
              &dwork[iq3 + nm], &n, &ZERO, &dwork[ih12 + nmm], &n);

    is_idx = 0;
    for (i32 j = 0; j < m; j++) {
        i32 cnt = j + 1;
        SLC_DAXPY(&cnt, &ONE, &dwork[ih12 + nmm + j], &n, &dwork[ih12 + nmm + is_idx], &int1);
        is_idx += n;
    }

    ma02ed('U', n, &dwork[it], n);

    if (getenv("MB03LF_DEBUG") != NULL) {
        fprintf(stderr, "MB03LF: Before mb03id, n=%d, n2=%d, m=%d, ih=%d\n", n, n2, m, ih);
        fprintf(stderr, "MB03LF: B matrix is n2 x n2 (%d x %d) with leading dim n=%d\n", n2, n2, n);
        fprintf(stderr, "MB03LF: B matrix (dwork[ih]) subdiagonals:\n");
        for (i32 ii = 0; ii < n2 - 1; ii++) {
            i32 bidx = ih + (ii + 1) + ii * n;  /* B(ii+2, ii+1) in 1-based, ld=n */
            f64 bval = dwork[bidx];
            fprintf(stderr, "  B(%d,%d) at idx %d = %.16e\n", ii + 2, ii + 1, bidx, bval);
        }
        fprintf(stderr, "MB03LF: Done printing B subdiagonals\n");
    }

    mb03id(cmqi_str, cmui_str, n2, z, ldz, &dwork[it], n, &dwork[iz12], n,
           &dwork[ih], n, &dwork[ih12], n, q, ldq, u, ldu, &u[n * ldu], ldu,
           neig, iwork, liwork, &dwork[iwrk], ldwork - iwrk, info);

    if (*info > 0) {
        return;
    }

    if (qr_flag) {
        *neig = *neig / 2;
    }

    iwrk = iz12;

    if (lcmpq) {
        SLC_DLACPY("F", &m, &m, &dwork[iq1 + nmm], &n, &dwork[ih], &n);

        iw = ih + m;
        is_idx = iq1 + nm;
        for (i32 j = 0; j < m; j++) {
            for (i32 i = 0; i < m; i++) {
                dwork[iw] = -dwork[is_idx];
                iw++;
                is_idx++;
            }
            iw += m;
            is_idx += m;
        }

        iw = ih + nm;
        is_idx = iq1 + m;
        for (i32 j = 0; j < m; j++) {
            for (i32 i = 0; i < m; i++) {
                dwork[iw] = -dwork[is_idx];
                iw++;
                is_idx++;
            }
            iw += m;
            is_idx += m;
        }
        SLC_DLACPY("F", &m, &m, &dwork[iq1], &n, &dwork[ih + nmm], &n);

        SLC_DLACPY("F", &n, &n, &dwork[iq2], &n, &dwork[it], &n);

        SLC_DGEMM("N", "N", &m, neig, &n, &ONE, &dwork[iq4], &n,
                  q, &ldq, &ZERO, &dwork[iwrk], &n2);
        SLC_DGEMM("N", "N", &m, neig, &n, &ONE, &dwork[iq3], &n,
                  &q[n], &ldq, &ZERO, &dwork[iwrk + m], &n2);
        SLC_DGEMM("N", "N", &m, neig, &n, &ONE, &dwork[iq4 + m], &n,
                  q, &ldq, &ZERO, &dwork[iwrk + n], &n2);
        SLC_DGEMM("N", "N", &m, neig, &n, &ONE, &dwork[iq3 + m], &n,
                  &q[n], &ldq, &ZERO, &dwork[iwrk + n + m], &n2);

        f64 scale_val = sqrt(TWO) / TWO;
        SLC_DGEMM("N", "N", &n, neig, &n2, &scale_val, &dwork[ih], &n,
                  &dwork[iwrk], &n2, &ZERO, q, &ldq);

        if (lcmpu) {
            iwrk = iq3;
        } else {
            iwrk = *neig;
        }

        if (svd) {
            i32 lwork_svd = ldwork - iwrk;
            SLC_DGESVD("O", "N", &n, neig, q, &ldq, dwork, dwork, &int1,
                       dwork, &int1, &dwork[iwrk], &lwork_svd, info);
            if (*info > 0) {
                *info = 4;
                return;
            }
            temp_optdw = (i32)dwork[iwrk] + iwrk;
            optdw = (optdw > temp_optdw) ? optdw : temp_optdw;
            if (!lcmpu) {
                *neig = *neig / 2;
            }
        } else {
            if (qr_flag) {
                i32 lwork_qr = ldwork - iwrk;
                SLC_DGEQRF(&n, neig, q, &ldq, dwork, &dwork[iwrk], &lwork_qr, info);
            } else {
                for (i32 j = 0; j < *neig; j++) {
                    iwork[j] = 0;
                }
                i32 lwork_qr = ldwork - iwrk;
                SLC_DGEQP3(&n, neig, q, &ldq, iwork, dwork, &dwork[iwrk], &lwork_qr, info);
            }
            temp_optdw = (i32)dwork[iwrk] + iwrk;
            optdw = (optdw > temp_optdw) ? optdw : temp_optdw;

            i32 lwork_org = ldwork - iwrk;
            SLC_DORGQR(&n, neig, neig, q, &ldq, dwork, &dwork[iwrk], &lwork_org, info);
            temp_optdw = (i32)dwork[iwrk] + iwrk;
            optdw = (optdw > temp_optdw) ? optdw : temp_optdw;

            if (qrp && !lcmpu) {
                *neig = *neig / 2;
            }
        }
    }

    if (lcmpu) {
        SLC_DLACPY("F", &m, &n, &dwork[iu11], &m, &dwork[ih], &n);

        iw = ih + m;
        is_idx = iu12;
        for (i32 j = 0; j < m; j++) {
            for (i32 i = 0; i < m; i++) {
                dwork[iw] = -dwork[is_idx];
                iw++;
                is_idx++;
            }
            iw += m;
        }
        SLC_DLACPY("F", &m, &m, &dwork[iu11], &m, &dwork[ih + nmm], &n);

        SLC_DLACPY("F", &m, &n, &dwork[iu21], &m, &dwork[it], &n);

        iw = it + m;
        is_idx = iu22;
        for (i32 j = 0; j < m; j++) {
            for (i32 i = 0; i < m; i++) {
                dwork[iw] = -dwork[is_idx];
                iw++;
                is_idx++;
            }
            iw += m;
        }
        SLC_DLACPY("F", &m, &m, &dwork[iu21], &m, &dwork[it + nmm], &n);

        SLC_DGEMM("N", "N", &m, neig, &n, &ONE, &dwork[iu3], &n,
                  u, &ldu, &ZERO, &dwork[iwrk], &n2);
        SLC_DGEMM("N", "N", &m, neig, &n, &neg_one, &dwork[iu3], &n,
                  &u[n * ldu], &ldu, &ZERO, &dwork[iwrk + m], &n2);
        SLC_DGEMM("N", "N", &m, neig, &n, &ONE, &dwork[iu3 + m], &n,
                  u, &ldu, &ZERO, &dwork[iwrk + n], &n2);
        SLC_DGEMM("N", "N", &m, neig, &n, &neg_one, &dwork[iu3 + m], &n,
                  &u[n * ldu], &ldu, &ZERO, &dwork[iwrk + n + m], &n2);

        f64 scale_val = sqrt(TWO) / TWO;
        SLC_DGEMM("N", "N", &n, neig, &n2, &scale_val, &dwork[ih], &n,
                  &dwork[iwrk], &n2, &ZERO, u, &ldu);

        iwrk = *neig;

        if (svd) {
            i32 lwork_svd = ldwork - iwrk;
            SLC_DGESVD("O", "N", &n, neig, u, &ldu, dwork, dwork, &int1,
                       dwork, &int1, &dwork[iwrk], &lwork_svd, info);
            if (*info > 0) {
                *info = 4;
                return;
            }
            temp_optdw = (i32)dwork[iwrk] + iwrk;
            optdw = (optdw > temp_optdw) ? optdw : temp_optdw;
            *neig = *neig / 2;
        } else {
            if (qr_flag) {
                i32 lwork_qr = ldwork - iwrk;
                SLC_DGEQRF(&n, neig, u, &ldu, dwork, &dwork[iwrk], &lwork_qr, info);
            } else {
                for (i32 j = 0; j < *neig; j++) {
                    iwork[j] = 0;
                }
                i32 lwork_qr = ldwork - iwrk;
                SLC_DGEQP3(&n, neig, u, &ldu, iwork, dwork, &dwork[iwrk], &lwork_qr, info);
            }
            temp_optdw = (i32)dwork[iwrk] + iwrk;
            optdw = (optdw > temp_optdw) ? optdw : temp_optdw;

            i32 lwork_org = ldwork - iwrk;
            SLC_DORGQR(&n, neig, neig, u, &ldu, dwork, &dwork[iwrk], &lwork_org, info);
            temp_optdw = (i32)dwork[iwrk] + iwrk;
            optdw = (optdw > temp_optdw) ? optdw : temp_optdw;

            if (qrp) {
                *neig = *neig / 2;
            }
        }
    }

    dwork[0] = (f64)optdw;
}
