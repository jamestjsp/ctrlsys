/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB03LP - Eigenvalues and right deflating subspace of a real
 *          skew-Hamiltonian/Hamiltonian pencil (block variant)
 *
 * Computes the relevant eigenvalues of a real N-by-N skew-Hamiltonian/
 * Hamiltonian pencil aS - bH with:
 *   S = [[A, D], [E, A']]  where D, E are skew-symmetric
 *   H = [[B, F], [G, -B']] where F, G are symmetric
 *
 * Optionally computes orthogonal basis of right deflating subspace
 * corresponding to eigenvalues with strictly negative real part.
 *
 * This variant applies transformations on panels of columns for better
 * performance on large matrices.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void mb03lp(const char *compq, const char *orth, i32 n,
            f64 *a, i32 lda, f64 *de, i32 ldde, f64 *b, i32 ldb,
            f64 *fg, i32 ldfg, i32 *neig, f64 *q, i32 ldq,
            f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork,
            i32 *bwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    bool liniq, lquery, qr, qrp, svd;
    i32 m, n2, nn, mm, nb;
    i32 ib, ic2, ifo, ih11, ih12, iq1, iq2, iq3, iq4;
    i32 irt, is11, is12, iw, iwrk, j;
    i32 mindw, miniw, nm, nmm, optdw;
    f64 dum[3];

    i32 int1 = 1, neg_one = -1;
    f64 dbl0 = ZERO, dbl1 = ONE;

    nb = *info;
    m = n / 2;
    n2 = n * 2;
    nn = n * n;
    mm = m * m;
    *neig = 0;

    liniq = (compq[0] == 'C' || compq[0] == 'c');
    if (liniq) {
        qr = (orth[0] == 'Q' || orth[0] == 'q');
        qrp = (orth[0] == 'P' || orth[0] == 'p');
        svd = (orth[0] == 'S' || orth[0] == 's');
    } else {
        qr = false;
        qrp = false;
        svd = false;
    }

    if (n == 0) {
        miniw = 1;
        mindw = 1;
    } else if (liniq) {
        miniw = (32 > 3 * n2 - 3) ? 32 : (3 * n2 - 3);
        mindw = 8 * nn + ((13 * n + 32) > 272 ? (13 * n + 32) : 272);
    } else {
        i32 l;
        if ((m % 2) == 0) {
            l = ((4 * n) > 32 ? (4 * n) : 32) + 4;
        } else {
            l = (4 * n) > 36 ? (4 * n) : 36;
        }
        miniw = ((n + 12) > (n2 + 3)) ? (n + 12) : (n2 + 3);
        mindw = 3 * mm + nn + l;
    }
    lquery = (ldwork == -1);

    *info = 0;
    if (!(compq[0] == 'N' || compq[0] == 'n' || liniq)) {
        *info = -1;
    } else if (liniq) {
        if (!(qr || qrp || svd)) {
            *info = -2;
        }
    }
    if (*info == 0 && (n < 0 || (n % 2) != 0)) {
        *info = -3;
    }
    if (*info == 0) {
        i32 m1 = (1 > m) ? 1 : m;
        if (lda < m1) {
            *info = -5;
        } else if (ldde < m1) {
            *info = -7;
        } else if (ldb < m1) {
            *info = -9;
        } else if (ldfg < m1) {
            *info = -11;
        } else if (ldq < 1 || (liniq && ldq < n2)) {
            *info = -14;
        } else if (liwork < miniw) {
            iwork[0] = miniw;
            *info = -19;
        } else if (!lquery && ldwork < mindw) {
            dwork[0] = (f64)mindw;
            *info = -21;
        }
    }

    if (*info != 0) {
        i32 neg_info = -(*info);
        SLC_XERBLA("MB03LP", &neg_info);
        return;
    }

    if (n > 0) {
        if (lquery) {
            if (liniq) {
                mb04hd("I", "I", n, dwork, n, dwork, n, dwork, n,
                       dwork, n, iwork, liwork, dum, -1, bwork, info);
                if (svd) {
                    SLC_DGESVD("O", "N", &n, &n, q, &ldq, dwork, dwork,
                               &ldq, dwork, &int1, &dum[1], &neg_one, info);
                    j = n + (i32)dum[1];
                } else {
                    if (qr) {
                        SLC_DGEQRF(&n, &m, q, &ldq, dwork, &dum[1], &neg_one, info);
                        j = m;
                    } else {
                        SLC_DGEQP3(&n, &n, q, &ldq, iwork, dwork, &dum[1], &neg_one, info);
                        j = n;
                    }
                    SLC_DORGQR(&n, &j, &j, q, &ldq, dwork, &dum[2], &neg_one, info);
                    i32 d1 = (i32)dum[1];
                    i32 d2 = (i32)dum[2];
                    j = j + ((d1 > d2) ? d1 : d2);
                }
                i32 opt1 = 6 * nn + (i32)dum[0];
                optdw = (mindw > opt1) ? mindw : opt1;
                optdw = (optdw > j) ? optdw : j;
            } else {
                optdw = mindw;
            }
            dwork[0] = (f64)optdw;
            return;
        }
    }

    if (n == 0) {
        dwork[0] = ONE;
        return;
    }

    ifo = 0;

    *info = nb;
    if (liniq) {
        char cmpq[15];
        strcpy(cmpq, "Initialize");
        iq1 = 0;
        iq2 = iq1 + nn;
        iwrk = iq2 + nn;
        if ((m % 4) == 0) {
            ic2 = m / 4;
        } else {
            ic2 = m / 4 + 1;
        }
        ib = 2 * ic2;
        ic2 = ic2;

        mb04bp("Triangularize", cmpq, cmpq, n, a, lda, de, ldde,
               b, ldb, fg, ldfg, &dwork[iq1], n, &dwork[iq2], n,
               &q[ib * ldq], m, &q[ifo * ldq], m, &q[ic2 * ldq], m,
               alphar, alphai, beta, iwork, liwork,
               &dwork[iwrk], ldwork - iwrk, info);
    } else {
        ib = ifo + mm;
        ic2 = ib + mm;
        iwrk = ic2 + mm;
        mb04bp("Eigenvalues", "No Computation", "No Computation", n, a, lda, de, ldde,
               b, ldb, fg, ldfg, dwork, n, dwork, n, &dwork[ib], m,
               &dwork[ifo], m, &dwork[ic2], m, alphar, alphai, beta,
               iwork, liwork, &dwork[iwrk], ldwork - iwrk, info);
    }
    optdw = mindw > ((i32)dwork[iwrk] + iwrk) ? mindw : ((i32)dwork[iwrk] + iwrk);

    if (*info > 0 && *info < 3) {
        *info = 1;
        return;
    } else if (*info == 3) {
        iw = 5;
    } else {
        iw = 0;
    }

    if (!liniq) {
        ma02ad("Upper", m, m, &dwork[ic2], m, de, ldde);
        if (m > 1) {
            i32 mm1 = m - 1;
            i32 ldc2_p1 = m + 1;
            i32 ldde_p1 = ldde + 1;
            SLC_DCOPY(&mm1, &dwork[ic2 + 1], &ldc2_p1, &de[ldde], &ldde_p1);
        }
        dwork[0] = (f64)optdw;
        *info = iw;
        return;
    }

    nm = n * m;
    nmm = nm + m;
    iq3 = iwrk;
    iq4 = iq3 + nn;
    is11 = iq4 + nn;
    ih11 = is11 + nn;
    iwrk = ih11 + nn;

    SLC_DLACPY("Full", &m, &m, a, &lda, &dwork[is11], &n);
    SLC_DLACPY("Full", &m, &m, &q[ib * ldq], &m, &dwork[is11 + nmm], &n);
    SLC_DSCAL(&mm, &ONE, &q[ic2 * ldq], &int1);
    for (i32 i = 0; i < mm; i++) {
        q[ic2 * ldq + i] = -q[ic2 * ldq + i];
    }
    SLC_DLACPY("Full", &m, &m, &q[ic2 * ldq], &m, &dwork[ih11 + m], &n);
    SLC_DLACPY("Full", &m, &m, b, &ldb, &dwork[ih11 + nm], &n);

    mb04hd("Initialize", "Initialize", n, &dwork[is11], n, &dwork[ih11], n,
           &dwork[iq3], n, &dwork[iq4], n, iwork, liwork,
           &dwork[iwrk], ldwork - iwrk, bwork, info);
    if (*info > 0) {
        if (*info > 2) {
            *info = 2;
        }
        return;
    }
    optdw = optdw > ((i32)dwork[iwrk] + iwrk) ? optdw : ((i32)dwork[iwrk] + iwrk);

    is12 = iwrk;
    ih12 = is12 + nn;
    iwrk = ih12;

    if (m > 1) {
        i32 mm1 = m - 1;

        SLC_DLACPY("Full", &mm1, &m, &dwork[iq4 + nm + 1], &n, &dwork[is12], &m);
        SLC_DLACPY("Full", &mm1, &m, &dwork[iq4 + nm], &n, &q[(ib + 1) * ldq], &m);
        SLC_DTRMM("Left", "Upper", "No Transpose", "Non-Unit", &mm1,
                  &m, &dbl1, &de[2 * ldde], &ldde, &dwork[is12], &m);

        mb01kd("Upper", "Transpose", m, mm1, ONE,
               &dwork[iq4 + nm], n, &dwork[is12], m, ZERO,
               &dwork[is12 + nmm], n, info);

        f64 neg1 = -ONE;
        SLC_DTRMM("Left", "Upper", "Transpose", "Non-Unit", &mm1,
                  &m, &neg1, &de[2 * ldde], &ldde, &q[(ib + 1) * ldq], &m);
        dum[0] = ZERO;
        i32 mm1_inc = m - 1;
        for (i32 i = 0; i < m; i++) {
            dwork[is12 + mm1_inc + i * m] = ZERO;
        }
        for (i32 i = 0; i < m; i++) {
            q[ib * ldq + i * m] = ZERO;
        }
        SLC_DAXPY(&mm, &dbl1, &q[ib * ldq], &int1, &dwork[is12], &int1);

        SLC_DLACPY("Full", &mm1, &m, &dwork[iq4 + nmm + 1], &n, &dwork[iwrk], &m);
        SLC_DLACPY("Full", &mm1, &m, &dwork[iq4 + nmm], &n, &q[(ib + 1) * ldq], &m);
        SLC_DTRMM("Left", "Upper", "No Transpose", "Non-Unit", &mm1,
                  &m, &dbl1, &q[(m + 1) * ldq + ifo * ldq], &m, &dwork[iwrk], &m);

        mb01kd("Upper", "Transpose", m, mm1, ONE,
               &dwork[iq4 + nmm], n, &dwork[iwrk], m, ONE,
               &dwork[is12 + nmm], n, info);

        SLC_DTRMM("Left", "Upper", "Transpose", "Non-Unit", &mm1,
                  &m, &neg1, &q[(m + 1) * ldq + ifo * ldq], &m, &q[(ib + 1) * ldq], &m);
        for (i32 i = 0; i < m; i++) {
            dwork[iwrk + mm1_inc + i * m] = ZERO;
        }
        SLC_DAXPY(&mm, &dbl1, &q[ib * ldq], &int1, &dwork[iwrk], &int1);

        SLC_DGEMM("Transpose", "No Transpose", &m, &m, &m, &dbl1,
                  &dwork[iq4], &n, &dwork[is12], &m, &dbl0,
                  &dwork[is12 + nm], &n);
        SLC_DGEMM("Transpose", "No Transpose", &m, &m, &m, &dbl1,
                  &dwork[iq4 + m], &n, &dwork[iwrk], &m, &dbl1,
                  &dwork[is12 + nm], &n);

        i32 iwrk_size = ldwork - iwrk;
        mb01ld("Upper", "Transpose", m, m, ZERO, ONE,
               &dwork[is12], n, &dwork[iq4], n, &de[ldde], ldde,
               &dwork[iwrk], iwrk_size, info);
        mb01ld("Upper", "Transpose", m, m, ONE, ONE,
               &dwork[is12], n, &dwork[iq4 + m], n, &q[ifo * ldq],
               m, &dwork[iwrk], iwrk_size, info);
    }

    SLC_DGEMM("Transpose", "No Transpose", &m, &m, &m, &dbl1,
              &fg[ldfg], &ldfg, &dwork[iq4 + nm], &n, &dbl0,
              &q[ifo * ldq], &m);
    SLC_DGEMM("Transpose", "No Transpose", &m, &m, &m, &dbl1,
              &fg[ldfg], &ldfg, &dwork[iq4], &n, &dbl0,
              &dwork[ih12 + nmm], &n);
    SLC_DGEMM("Transpose", "No Transpose", &m, &m, &m, &dbl1,
              &dwork[iq4 + m], &n, &q[ifo * ldq], &m, &dbl0,
              &dwork[ih12 + nm], &n);
    SLC_DGEMM("Transpose", "No Transpose", &m, &m, &m, &dbl1,
              &dwork[ih12 + nmm], &n, &dwork[iq4 + nmm], &n, &dbl1,
              &dwork[ih12 + nm], &n);

    SLC_DSYR2K("Upper", "Transpose", &m, &m, &dbl1, &dwork[ih12 + nmm],
               &n, &dwork[iq4 + m], &n, &dbl0, &dwork[ih12], &n);

    SLC_DSYR2K("Upper", "Transpose", &m, &m, &dbl1, &q[ifo * ldq],
               &m, &dwork[iq4 + nmm], &n, &dbl0, &dwork[ih12 + nmm], &n);

    for (i32 i = 0; i < mm; i++) {
        q[ic2 * ldq + i] = -q[ic2 * ldq + i];
    }
    ma02ad("Upper", m, m, &q[ic2 * ldq], m, de, ldde);
    if (m > 1) {
        i32 mm1 = m - 1;
        i32 ldc2_p1 = m + 1;
        i32 ldde_p1 = ldde + 1;
        SLC_DCOPY(&mm1, &q[(ic2 + 1) * ldq], &ldc2_p1, &de[ldde], &ldde_p1);
    }

    iwrk = ih12 + nn;

    *info = nb;
    mb03jp("Initialize", n2, &dwork[is11], n, &dwork[is12], n,
           &dwork[ih11], n, &dwork[ih12], n, q, ldq, neig,
           iwork, liwork, &dwork[iwrk], ldwork - iwrk, info);
    if (*info > 0) {
        *info = *info + 1;
        return;
    }

    iwrk = is11;
    if (qr) {
        *neig = *neig / 2;
    }

    SLC_DLACPY("Full", &m, &m, &dwork[iq1 + nmm], &n, &dwork[iwrk], &n);
    SLC_DLACPY("Full", &m, &m, &dwork[iq1 + nm], &n, &dwork[iwrk + m], &n);
    for (j = 0; j < m; j++) {
        f64 neg1 = -ONE;
        SLC_DSCAL(&m, &neg1, &dwork[iwrk + m + j * n], &int1);
    }
    SLC_DLACPY("Full", &m, &m, &dwork[iq1 + m], &n, &dwork[iwrk + nm], &n);
    for (j = 0; j < m; j++) {
        f64 neg1 = -ONE;
        SLC_DSCAL(&m, &neg1, &dwork[iwrk + nm + j * n], &int1);
    }
    SLC_DLACPY("Full", &m, &m, &dwork[iq1], &n, &dwork[iwrk + nmm], &n);

    SLC_DLACPY("Full", &n, &n, &dwork[iq2], &n, &dwork[iwrk + nn], &n);

    irt = iwrk + n * n2;
    SLC_DGEMM("No Transpose", "No Transpose", &m, neig, &n, &dbl1,
              &dwork[iq3], &n, q, &ldq, &dbl0, &dwork[irt], &n2);
    SLC_DGEMM("No Transpose", "No Transpose", &m, neig, &n, &dbl1,
              &dwork[iq4], &n, &q[n], &ldq, &dbl0, &dwork[irt + m], &n2);
    SLC_DGEMM("No Transpose", "No Transpose", &m, neig, &n, &dbl1,
              &dwork[iq3 + m], &n, q, &ldq, &dbl0, &dwork[irt + n], &n2);
    SLC_DGEMM("No Transpose", "No Transpose", &m, neig, &n, &dbl1,
              &dwork[iq4 + m], &n, &q[n], &ldq, &dbl0, &dwork[irt + n + m], &n2);

    f64 scale = sqrt(TWO) / TWO;
    SLC_DGEMM("No Transpose", "No Transpose", &n, neig, &n2, &scale,
              &dwork[iwrk], &n, &dwork[irt], &n2, &dbl0, q, &ldq);

    iwrk = *neig;
    if (svd) {
        i32 neig_out = *neig;
        SLC_DGESVD("Overwrite", "No V", &n, &neig_out, q, &ldq, dwork,
                   dwork, &int1, dwork, &int1, &dwork[iwrk], &(i32){ldwork - iwrk}, info);
        if (*info > 0) {
            *info = 4;
            return;
        }
        optdw = optdw > ((i32)dwork[iwrk] + iwrk) ? optdw : ((i32)dwork[iwrk] + iwrk);
        *neig = *neig / 2;
    } else {
        if (qr) {
            SLC_DGEQRF(&n, neig, q, &ldq, dwork, &dwork[iwrk], &(i32){ldwork - iwrk}, info);
        } else {
            for (j = 0; j < *neig; j++) {
                iwork[j] = 0;
            }
            SLC_DGEQP3(&n, neig, q, &ldq, iwork, dwork, &dwork[iwrk], &(i32){ldwork - iwrk}, info);
        }
        optdw = optdw > ((i32)dwork[iwrk] + iwrk) ? optdw : ((i32)dwork[iwrk] + iwrk);

        SLC_DORGQR(&n, neig, neig, q, &ldq, dwork, &dwork[iwrk], &(i32){ldwork - iwrk}, info);
        optdw = optdw > ((i32)dwork[iwrk] + iwrk) ? optdw : ((i32)dwork[iwrk] + iwrk);
        if (qrp) {
            *neig = *neig / 2;
        }
    }

    dwork[0] = (f64)optdw;
    *info = iw;
}
