/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>

void mb04tt(bool updatq, bool updatz, i32 m, i32 n,
            i32 ifira, i32 ifica, i32 nca,
            f64 *a, i32 lda, f64 *e, i32 lde,
            f64 *q, i32 ldq, f64 *z, i32 ldz,
            i32 *istair, i32 *rank, f64 tol,
            i32 *iwork)
{
    const f64 ZERO = 0.0;

    i32 int1 = 1;

    *rank = 0;
    if (m <= 0 || n <= 0) {
        return;
    }

    i32 nj = nca;
    i32 mj = m + 1 - ifira;
    i32 ifira1 = ifira - 1;
    i32 ifica1 = ifica - 1;

    for (i32 i = 0; i < nj; i++) {
        iwork[i] = i + 1;
    }

    i32 k = 1;
    bool lzero = false;
    *rank = (nj < mj) ? nj : mj;
    i32 mxrank = *rank;

    while ((k <= mxrank) && (!lzero)) {
        f64 bmxnrm = ZERO;
        i32 lsav = k;
        i32 kk = ifira1 + k;

        for (i32 l = k; l <= nj; l++) {
            i32 ll = ifica1 + l;
            i32 len = mj - k + 1;
            i32 idx = SLC_IDAMAX(&len, &a[(kk - 1) + (ll - 1) * lda], &int1);
            f64 bmx = fabs(a[(kk - 1) + (idx - 1) + (ll - 1) * lda]);
            if (bmx > bmxnrm) {
                bmxnrm = bmx;
                lsav = l;
            }
        }

        i32 ll = ifica1 + k;
        if (bmxnrm <= tol) {
            i32 rows = mj - k + 1;
            i32 cols = nj - k + 1;
            f64 zero = ZERO;
            SLC_DLASET("Full", &rows, &cols, &zero, &zero, &a[(kk - 1) + (ll - 1) * lda], &lda);
            lzero = true;
            *rank = k - 1;
        } else {
            if (lsav != k) {
                SLC_DSWAP(&m, &a[(ll - 1) * lda], &int1, &a[(ifica1 + lsav - 1) * lda], &int1);
                i32 ip = iwork[lsav - 1];
                iwork[lsav - 1] = iwork[k - 1];
                iwork[k - 1] = ip;
            }

            k = k + 1;
            i32 mk1 = n - ll + 1;

            for (i32 i = mj; i >= k; i--) {
                i32 ii = ifira1 + i;

                f64 sc, ss;
                SLC_DROTG(&a[(ii - 2) + (ll - 1) * lda], &a[(ii - 1) + (ll - 1) * lda], &sc, &ss);
                i32 len = mk1 - 1;
                if (len > 0) {
                    SLC_DROT(&len, &a[(ii - 2) + ll * lda], &lda, &a[(ii - 1) + ll * lda], &lda, &sc, &ss);
                }
                a[(ii - 1) + (ll - 1) * lda] = ZERO;
                if (updatq) {
                    SLC_DROT(&m, &q[(ii - 2) * ldq], &int1, &q[(ii - 1) * ldq], &int1, &sc, &ss);
                }

                i32 ist1 = istair[ii - 2];
                i32 ist2 = istair[ii - 1];

                i32 itype;
                if ((ist1 > 0 && ist2 > 0) || (ist1 < 0 && ist2 < 0)) {
                    if (ist1 > 0) {
                        itype = 1;
                    } else {
                        itype = 3;
                    }
                } else {
                    if (ist1 < 0) {
                        itype = 2;
                    } else {
                        itype = 4;
                    }
                }

                i32 jc1 = abs(ist1);
                i32 jc2 = abs(ist2);
                i32 jpvt = (jc1 < jc2) ? jc1 : jc2;

                i32 rot_len = n - jpvt + 1;
                SLC_DROT(&rot_len, &e[(ii - 2) + (jpvt - 1) * lde], &lde, &e[(ii - 1) + (jpvt - 1) * lde], &lde, &sc, &ss);
                f64 eijpvt = e[(ii - 1) + (jpvt - 1) * lde];

                if (itype == 1) {
                    SLC_DROTG(&e[(ii - 1) + jpvt * lde], &e[(ii - 1) + (jpvt - 1) * lde], &sc, &ss);
                    i32 col_len = ii - 1;
                    if (col_len > 0) {
                        SLC_DROT(&col_len, &e[jpvt * lde], &int1, &e[(jpvt - 1) * lde], &int1, &sc, &ss);
                    }
                    e[(ii - 1) + (jpvt - 1) * lde] = ZERO;

                    SLC_DROT(&m, &a[jpvt * lda], &int1, &a[(jpvt - 1) * lda], &int1, &sc, &ss);
                    if (updatz) {
                        SLC_DROT(&n, &z[jpvt * ldz], &int1, &z[(jpvt - 1) * ldz], &int1, &sc, &ss);
                    }
                } else if (itype == 2) {
                    if (fabs(eijpvt) <= tol) {
                        i32 istpvt = istair[ii - 1];
                        istair[ii - 2] = istpvt;
                        istair[ii - 1] = -(istpvt + 1);
                        e[(ii - 1) + (jpvt - 1) * lde] = ZERO;
                    }
                } else if (itype == 4) {
                    if (fabs(eijpvt) > tol) {
                        i32 istpvt = istair[ii - 2];
                        istair[ii - 2] = -istpvt;
                        istair[ii - 1] = istpvt;
                    }
                }
            }
        }
    }

    i32 perm_rows = ifira1 + *rank;
    i32 forwrd = 0;
    SLC_DLAPMT(&forwrd, &perm_rows, &nj, &a[(ifica - 1) * lda], &lda, iwork);
}
