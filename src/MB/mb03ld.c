/*
 * SPDX-License-Identifier: BSD-3-Clause
 * MB03LD - Eigenvalues and right deflating subspace of a real
 *          skew-Hamiltonian/Hamiltonian pencil
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>
#include <stdlib.h>

void mb03ld(const char *compq, const char *orth, i32 n,
            f64 *a, i32 lda, f64 *de, i32 ldde,
            f64 *b, i32 ldb, f64 *fg, i32 ldfg,
            i32 *neig, f64 *q, i32 ldq,
            f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork,
            bool *bwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    char c_compq = (char)toupper((unsigned char)compq[0]);
    char c_orth = (char)toupper((unsigned char)orth[0]);

    bool liniq = (c_compq == 'C');
    bool qr = false, qrp = false, svd = false;
    if (liniq) {
        qr = (c_orth == 'Q');
        qrp = (c_orth == 'P');
        svd = (c_orth == 'S');
    }
    bool lquery = (ldwork == -1);

    i32 m = n / 2;
    i32 n2 = n * 2;
    i32 nn = n * n;
    i32 mm = m * m;
    *neig = 0;

    i32 miniw, mindw;
    if (n == 0) {
        miniw = 1;
        mindw = 1;
    } else if (liniq) {
        i32 tmp1 = n2 + 3;
        miniw = (32 > tmp1) ? 32 : tmp1;
        i32 tmp2 = 8 * n + 32;
        mindw = 8 * nn + ((tmp2 > 272) ? tmp2 : 272);
    } else {
        i32 j;
        if ((m % 2) == 0) {
            i32 tmp = 4 * n;
            j = ((tmp > 32) ? tmp : 32) + 4;
        } else {
            i32 tmp = 4 * n;
            j = (tmp > 36) ? tmp : 36;
        }
        i32 tmp1 = n + 12;
        i32 tmp2 = n2 + 3;
        miniw = (tmp1 > tmp2) ? tmp1 : tmp2;
        mindw = 3 * mm + nn + j;
    }

    *info = 0;
    if (c_compq != 'N' && !liniq) {
        *info = -1;
    } else if (liniq && !(qr || qrp || svd)) {
        *info = -2;
    } else if (n < 0 || (n % 2) != 0) {
        *info = -3;
    } else if (lda < ((1 > m) ? 1 : m)) {
        *info = -5;
    } else if (ldde < ((1 > m) ? 1 : m)) {
        *info = -7;
    } else if (ldb < ((1 > m) ? 1 : m)) {
        *info = -9;
    } else if (ldfg < ((1 > m) ? 1 : m)) {
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

    if (*info != 0) {
        return;
    }

    i32 int1 = 1;
    i32 int0 = 0;
    i32 optdw = mindw;

    if (n > 0 && lquery) {
        f64 dum[4];
        if (liniq) {
            i32 intm1 = -1;
            mb04hd("I", "I", n, dwork, n, dwork, n, dwork, n, dwork, n,
                   iwork, liwork, dum, intm1, bwork, info);
            if (svd) {
                SLC_DGESVD("O", "N", &n, &n, q, &ldq, dwork, dwork, &ldq,
                           dwork, &int1, &dum[1], &intm1, info);
                i32 j = n + (i32)dum[1];
                optdw = mindw > (6 * nn + (i32)dum[0]) ? mindw : (6 * nn + (i32)dum[0]);
                if (j > optdw) optdw = j;
            } else {
                if (qr) {
                    SLC_DGEQRF(&n, &m, q, &ldq, dwork, &dum[1], &intm1, info);
                    i32 j = m;
                    SLC_DORGQR(&n, &j, &j, q, &ldq, dwork, &dum[2], &intm1, info);
                    i32 tmp = (i32)dum[1] > (i32)dum[2] ? (i32)dum[1] : (i32)dum[2];
                    j = j + tmp;
                    optdw = mindw > (6 * nn + (i32)dum[0]) ? mindw : (6 * nn + (i32)dum[0]);
                    if (j > optdw) optdw = j;
                } else {
                    SLC_DGEQP3(&n, &n, q, &ldq, iwork, dwork, &dum[1], &intm1, info);
                    i32 j = n;
                    SLC_DORGQR(&n, &j, &j, q, &ldq, dwork, &dum[2], &intm1, info);
                    i32 tmp = (i32)dum[1] > (i32)dum[2] ? (i32)dum[1] : (i32)dum[2];
                    j = j + tmp;
                    optdw = mindw > (6 * nn + (i32)dum[0]) ? mindw : (6 * nn + (i32)dum[0]);
                    if (j > optdw) optdw = j;
                }
            }
        }
        dwork[0] = (f64)optdw;
        return;
    }

    if (n == 0) {
        dwork[0] = ONE;
        return;
    }

    i32 ifo = 0;

    i32 iq1, iq2, iwrk, ic2, ib;
    i32 nm = n * m;
    i32 nmm = nm + m;

    if (liniq) {
        iq1 = 0;
        iq2 = iq1 + nn;
        iwrk = iq2 + nn;
        i32 tmp;
        if ((m % 4) == 0) {
            tmp = m / 4;
        } else {
            tmp = m / 4 + 1;
        }
        ib = 2 * tmp;
        ic2 = tmp;

        mb04bd("T", "I", "I", n, a, lda, de, ldde, b, ldb, fg, ldfg,
               &dwork[iq1], n, &dwork[iq2], n,
               &q[ib * ldq], m, &q[ifo * ldq], m, &q[ic2 * ldq], m,
               alphar, alphai, beta, iwork, liwork,
               &dwork[iwrk], ldwork - iwrk, info);
    } else {
        ib = ifo + mm;
        ic2 = ib + mm;
        iwrk = ic2 + mm;

        mb04bd("E", "N", "N", n, a, lda, de, ldde, b, ldb, fg, ldfg,
               dwork, n, dwork, n,
               &dwork[ib], m, &dwork[ifo], m, &dwork[ic2], m,
               alphar, alphai, beta, iwork, liwork,
               &dwork[iwrk], ldwork - iwrk, info);
    }
    i32 tmp = iwrk - 1 + (i32)dwork[iwrk];
    if (tmp > optdw) optdw = tmp;

    i32 iw = 0;
    if (*info > 0 && *info < 3) {
        *info = 1;
        return;
    } else if (*info == 3) {
        iw = 5;
    }

    if (!liniq) {
        ma02ad("U", m, m, &dwork[ic2], m, de, ldde);
        if (m > 1) {
            i32 mp1 = m + 1;
            SLC_DCOPY(&m, &dwork[ic2 + 1], &mp1, &de[ldde], &mp1);
        }
        dwork[0] = (f64)optdw;
        *info = iw;
        return;
    }

    i32 iq3 = iwrk;
    i32 iq4 = iq3 + nn;
    i32 is11 = iq4 + nn;
    i32 ih11 = is11 + nn;
    iwrk = ih11 + nn;

    SLC_DLACPY("F", &m, &m, a, &lda, &dwork[is11], &n);
    SLC_DLACPY("F", &m, &m, &q[ib * ldq], &m, &dwork[is11 + nmm], &n);
    SLC_DSCAL(&mm, &(f64){-ONE}, &q[ic2 * ldq], &int1);
    SLC_DLACPY("F", &m, &m, &q[ic2 * ldq], &m, &dwork[ih11 + m], &n);
    SLC_DLACPY("F", &m, &m, b, &ldb, &dwork[ih11 + nm], &n);

    mb04hd("I", "I", n, &dwork[is11], n, &dwork[ih11], n,
           &dwork[iq3], n, &dwork[iq4], n, iwork, liwork,
           &dwork[iwrk], ldwork - iwrk, bwork, info);

    if (*info > 0) {
        if (*info > 2) *info = 2;
        return;
    }
    tmp = iwrk - 1 + (i32)dwork[iwrk];
    if (tmp > optdw) optdw = tmp;

    i32 is12 = iwrk;
    i32 ih12 = is12 + nn;
    iwrk = ih12;

    if (m > 1) {
        i32 mm1 = m - 1;

        SLC_DLACPY("F", &mm1, &m, &dwork[iq4 + nm + 1], &n, &dwork[is12], &m);
        SLC_DLACPY("F", &mm1, &m, &dwork[iq4 + nm], &n, &q[(ib) * ldq + 1], &m);
        i32 ldde3 = ldde;
        SLC_DTRMM("L", "U", "N", "N", &mm1, &m, &ONE, &de[2 * ldde], &ldde3, &dwork[is12], &m);

        mb01kd("U", "T", m, mm1, ONE, &dwork[iq4 + nm], n, &dwork[is12], m, ZERO, &dwork[is12 + nmm], n, info);

        f64 neg_one = -ONE;
        SLC_DTRMM("L", "U", "T", "N", &mm1, &m, &neg_one, &de[2 * ldde], &ldde, &q[(ib) * ldq + 1], &m);
        f64 dum0 = ZERO;
        SLC_DCOPY(&m, &dum0, &int0, &dwork[is12 + m - 1], &m);
        SLC_DCOPY(&m, &dum0, &int0, &q[ib * ldq], &m);
        SLC_DAXPY(&mm, &ONE, &q[ib * ldq], &int1, &dwork[is12], &int1);

        SLC_DLACPY("F", &mm1, &m, &dwork[iq4 + nmm + 1], &n, &dwork[iwrk], &m);
        SLC_DLACPY("F", &mm1, &m, &dwork[iq4 + nmm], &n, &q[(ib) * ldq + 1], &m);
        SLC_DTRMM("L", "U", "N", "N", &mm1, &m, &ONE, &q[(m + 1) * ldq + ifo], &m, &dwork[iwrk], &m);

        mb01kd("U", "T", m, mm1, ONE, &dwork[iq4 + nmm], n, &dwork[iwrk], m, ONE, &dwork[is12 + nmm], n, info);

        SLC_DTRMM("L", "U", "T", "N", &mm1, &m, &neg_one, &q[(m + 1) * ldq + ifo], &m, &q[(ib) * ldq + 1], &m);
        SLC_DCOPY(&m, &dum0, &int0, &dwork[iwrk + m - 1], &m);
        SLC_DAXPY(&mm, &ONE, &q[ib * ldq], &int1, &dwork[iwrk], &int1);

        SLC_DGEMM("T", "N", &m, &m, &m, &ONE, &dwork[iq4], &n, &dwork[is12], &m, &ZERO, &dwork[is12 + nm], &n);
        SLC_DGEMM("T", "N", &m, &m, &m, &ONE, &dwork[iq4 + m], &n, &dwork[iwrk], &m, &ONE, &dwork[is12 + nm], &n);

        mb01ld("U", "T", m, m, ZERO, ONE, &dwork[is12], n, &dwork[iq4], n, &de[ldde], ldde, &dwork[iwrk], ldwork - iwrk, info);
        mb01ld("U", "T", m, m, ONE, ONE, &dwork[is12], n, &dwork[iq4 + m], n, &q[ifo * ldq], m, &dwork[iwrk], ldwork - iwrk, info);
    }

    SLC_DGEMM("T", "N", &m, &m, &m, &ONE, &fg[ldfg], &ldfg, &dwork[iq4 + nm], &n, &ZERO, &q[ifo * ldq], &m);
    SLC_DGEMM("T", "N", &m, &m, &m, &ONE, &fg[ldfg], &ldfg, &dwork[iq4], &n, &ZERO, &dwork[ih12 + nmm], &n);
    SLC_DGEMM("T", "N", &m, &m, &m, &ONE, &dwork[iq4 + m], &n, &q[ifo * ldq], &m, &ZERO, &dwork[ih12 + nm], &n);
    SLC_DGEMM("T", "N", &m, &m, &m, &ONE, &dwork[ih12 + nmm], &n, &dwork[iq4 + nmm], &n, &ONE, &dwork[ih12 + nm], &n);

    SLC_DSYR2K("U", "T", &m, &m, &ONE, &dwork[ih12 + nmm], &n, &dwork[iq4 + m], &n, &ZERO, &dwork[ih12], &n);

    SLC_DSYR2K("U", "T", &m, &m, &ONE, &q[ifo * ldq], &m, &dwork[iq4 + nmm], &n, &ZERO, &dwork[ih12 + nmm], &n);

    SLC_DSCAL(&mm, &(f64){-ONE}, &q[ic2 * ldq], &int1);
    ma02ad("U", m, m, &q[ic2 * ldq], m, de, ldde);
    if (m > 1) {
        i32 mp1 = m + 1;
        SLC_DCOPY(&m, &q[ic2 * ldq + 1], &mp1, &de[ldde], &mp1);
    }

    iwrk = ih12 + nn;

    mb03jd("I", n2, &dwork[is11], n, &dwork[is12], n,
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

    SLC_DLACPY("F", &m, &m, &dwork[iq1 + nmm], &n, &dwork[iwrk], &n);
    SLC_DLACPY("F", &m, &m, &dwork[iq1 + nm], &n, &dwork[iwrk + m], &n);
    for (i32 j = 0; j < m; j++) {
        SLC_DSCAL(&m, &(f64){-ONE}, &dwork[iwrk + m + j * n], &int1);
    }
    SLC_DLACPY("F", &m, &m, &dwork[iq1 + m], &n, &dwork[iwrk + nm], &n);
    for (i32 j = 0; j < m; j++) {
        SLC_DSCAL(&m, &(f64){-ONE}, &dwork[iwrk + nm + j * n], &int1);
    }
    SLC_DLACPY("F", &m, &m, &dwork[iq1], &n, &dwork[iwrk + nmm], &n);

    SLC_DLACPY("F", &n, &n, &dwork[iq2], &n, &dwork[iwrk + nn], &n);

    i32 irt = iwrk + n * n2;
    SLC_DGEMM("N", "N", &m, neig, &n, &ONE, &dwork[iq3], &n, q, &ldq, &ZERO, &dwork[irt], &n2);
    SLC_DGEMM("N", "N", &m, neig, &n, &ONE, &dwork[iq4], &n, &q[n], &ldq, &ZERO, &dwork[irt + m], &n2);
    SLC_DGEMM("N", "N", &m, neig, &n, &ONE, &dwork[iq3 + m], &n, q, &ldq, &ZERO, &dwork[irt + n], &n2);
    SLC_DGEMM("N", "N", &m, neig, &n, &ONE, &dwork[iq4 + m], &n, &q[n], &ldq, &ZERO, &dwork[irt + n + m], &n2);

    f64 scale = sqrt(TWO) / TWO;

    SLC_DGEMM("N", "N", &n, neig, &n2, &scale, &dwork[iwrk], &n, &dwork[irt], &n2, &ZERO, q, &ldq);

    iwrk = *neig;

    if (svd) {
        SLC_DGESVD("O", "N", &n, neig, q, &ldq, dwork, dwork, &int1, dwork, &int1,
                   &dwork[iwrk], &(i32){ldwork - iwrk}, info);
        if (*info > 0) {
            *info = 4;
            return;
        }
        tmp = iwrk + (i32)dwork[iwrk];
        if (tmp > optdw) optdw = tmp;
        *neig = *neig / 2;
    } else {
        if (qr) {
            SLC_DGEQRF(&n, neig, q, &ldq, dwork, &dwork[iwrk], &(i32){ldwork - iwrk}, info);
        } else {
            for (i32 j = 0; j < *neig; j++) {
                iwork[j] = 0;
            }
            SLC_DGEQP3(&n, neig, q, &ldq, iwork, dwork, &dwork[iwrk], &(i32){ldwork - iwrk}, info);
        }
        tmp = iwrk + (i32)dwork[iwrk];
        if (tmp > optdw) optdw = tmp;

        SLC_DORGQR(&n, neig, neig, q, &ldq, dwork, &dwork[iwrk], &(i32){ldwork - iwrk}, info);
        tmp = iwrk + (i32)dwork[iwrk];
        if (tmp > optdw) optdw = tmp;
        if (qrp) {
            *neig = *neig / 2;
        }
    }

    dwork[0] = (f64)optdw;
    *info = iw;
}
