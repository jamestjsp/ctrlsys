/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB02PD - Solve linear equations with LU factorization and iterative refinement
 *
 * Solves op(A)*X = B using LU factorization with optional equilibration
 * and iterative refinement for improved accuracy.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>

void mb02pd(
    const char* fact_str,
    const char* trans_str,
    const i32 n,
    const i32 nrhs,
    f64* a,
    const i32 lda,
    f64* af,
    const i32 ldaf,
    i32* ipiv,
    char* equed,
    f64* r,
    f64* c,
    f64* b,
    const i32 ldb,
    f64* x,
    const i32 ldx,
    f64* rcond,
    f64* ferr,
    f64* berr,
    i32* iwork,
    f64* dwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    char fact = toupper((unsigned char)fact_str[0]);
    char trans = toupper((unsigned char)trans_str[0]);

    bool nofact = (fact == 'N');
    bool equil = (fact == 'E');
    bool notran = (trans == 'N');

    bool rowequ, colequ;
    f64 smlnum, bignum, rowcnd, colcnd;

    *info = 0;

    if (nofact || equil) {
        *equed = 'N';
        rowequ = false;
        colequ = false;
    } else {
        rowequ = (*equed == 'R' || *equed == 'B');
        colequ = (*equed == 'C' || *equed == 'B');
        smlnum = SLC_DLAMCH("Safe minimum");
        bignum = one / smlnum;
    }

    if (!nofact && !equil && fact != 'F') {
        *info = -1;
    } else if (!notran && trans != 'T' && trans != 'C') {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (nrhs < 0) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldaf < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (fact == 'F' && !rowequ && !colequ && *equed != 'N') {
        *info = -10;
    } else {
        if (rowequ) {
            f64 rcmin = bignum;
            f64 rcmax = zero;
            for (i32 j = 0; j < n; j++) {
                if (r[j] < rcmin) rcmin = r[j];
                if (r[j] > rcmax) rcmax = r[j];
            }
            if (rcmin <= zero) {
                *info = -11;
            } else if (n > 0) {
                rowcnd = fmax(rcmin, smlnum) / fmin(rcmax, bignum);
            } else {
                rowcnd = one;
            }
        }
        if (colequ && *info == 0) {
            f64 rcmin = bignum;
            f64 rcmax = zero;
            for (i32 j = 0; j < n; j++) {
                if (c[j] < rcmin) rcmin = c[j];
                if (c[j] > rcmax) rcmax = c[j];
            }
            if (rcmin <= zero) {
                *info = -12;
            } else if (n > 0) {
                colcnd = fmax(rcmin, smlnum) / fmin(rcmax, bignum);
            } else {
                colcnd = one;
            }
        }
        if (*info == 0) {
            if (ldb < (n > 1 ? n : 1)) {
                *info = -14;
            } else if (ldx < (n > 1 ? n : 1)) {
                *info = -16;
            }
        }
    }

    if (*info != 0) {
        return;
    }

    if (equil) {
        i32 infequ;
        f64 amax;
        SLC_DGEEQU(&n, &n, a, &lda, r, c, &rowcnd, &colcnd, &amax, &infequ);
        if (infequ == 0) {
            SLC_DLAQGE(&n, &n, a, &lda, r, c, &rowcnd, &colcnd, &amax, equed);
            rowequ = (*equed == 'R' || *equed == 'B');
            colequ = (*equed == 'C' || *equed == 'B');
        }
    }

    if (notran) {
        if (rowequ) {
            for (i32 j = 0; j < nrhs; j++) {
                for (i32 i = 0; i < n; i++) {
                    b[i + j * ldb] = r[i] * b[i + j * ldb];
                }
            }
        }
    } else if (colequ) {
        for (i32 j = 0; j < nrhs; j++) {
            for (i32 i = 0; i < n; i++) {
                b[i + j * ldb] = c[i] * b[i + j * ldb];
            }
        }
    }

    f64 rpvgrw, anorm;
    const char *norm;

    if (nofact || equil) {
        SLC_DLACPY("Full", &n, &n, a, &lda, af, &ldaf);
        SLC_DGETRF(&n, &n, af, &ldaf, ipiv, info);

        if (*info > 0) {
            i32 info_val = *info;
            rpvgrw = SLC_DLANTR("M", "U", "N", &info_val, &info_val, af, &ldaf, dwork);
            if (rpvgrw == zero) {
                rpvgrw = one;
            } else {
                rpvgrw = SLC_DLANGE("M", &n, &info_val, a, &lda, dwork) / rpvgrw;
            }
            dwork[0] = rpvgrw;
            *rcond = zero;
            return;
        }

        if (notran) {
            norm = "1";
        } else {
            norm = "I";
        }
        anorm = SLC_DLANGE(norm, &n, &n, a, &lda, dwork);
        rpvgrw = SLC_DLANTR("M", "U", "N", &n, &n, af, &ldaf, dwork);
        if (rpvgrw == zero) {
            rpvgrw = one;
        } else {
            rpvgrw = SLC_DLANGE("M", &n, &n, a, &lda, dwork) / rpvgrw;
        }

        SLC_DGECON(norm, &n, af, &ldaf, &anorm, rcond, dwork, iwork, info);

        if (*rcond < SLC_DLAMCH("Epsilon")) {
            *info = n + 1;
        }
    } else {
        rpvgrw = dwork[0];
    }

    SLC_DLACPY("Full", &n, &nrhs, b, &ldb, x, &ldx);
    SLC_DGETRS(&trans, &n, &nrhs, af, &ldaf, ipiv, x, &ldx, info);

    SLC_DGERFS(&trans, &n, &nrhs, a, &lda, af, &ldaf, ipiv, b, &ldb,
               x, &ldx, ferr, berr, dwork, iwork, info);

    if (notran) {
        if (colequ) {
            for (i32 j = 0; j < nrhs; j++) {
                for (i32 i = 0; i < n; i++) {
                    x[i + j * ldx] = c[i] * x[i + j * ldx];
                }
            }
            for (i32 j = 0; j < nrhs; j++) {
                ferr[j] = ferr[j] / colcnd;
            }
        }
    } else if (rowequ) {
        for (i32 j = 0; j < nrhs; j++) {
            for (i32 i = 0; i < n; i++) {
                x[i + j * ldx] = r[i] * x[i + j * ldx];
            }
        }
        for (i32 j = 0; j < nrhs; j++) {
            ferr[j] = ferr[j] / rowcnd;
        }
    }

    dwork[0] = rpvgrw;
}
