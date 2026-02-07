/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB03MX - Solve discrete-time Lyapunov equation for Schur form matrix
 *
 * Solves: op(A)' * X * op(A) - X = scale * C
 * where A is upper quasi-triangular (Schur form), C is symmetric.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sb03mx(
    const char* trana,
    const i32 n,
    const f64* a,
    const i32 lda,
    f64* c,
    const i32 ldc,
    f64* scale,
    f64* dwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool notrna = (*trana == 'N' || *trana == 'n');
    bool lupper = true;
    i32 ilupper = lupper;

    *info = 0;
    if (!notrna && *trana != 'T' && *trana != 't' && *trana != 'C' && *trana != 'c') {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (ldc < (n > 1 ? n : 1)) {
        *info = -6;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("SB03MX", &neginfo);
        return;
    }

    *scale = ONE;

    if (n == 0) {
        return;
    }

    f64 eps = SLC_DLAMCH("P");
    f64 smlnum = SLC_DLAMCH("S");
    f64 bignum = ONE / smlnum;
    SLC_DLABAD(&smlnum, &bignum);
    smlnum = smlnum * (f64)(n * n) / eps;
    bignum = ONE / smlnum;

    i32 nn = n;
    f64 smin = smlnum;
    {
        f64 anorm = SLC_DLANHS("Max", &nn, a, &lda, dwork);
        smin = fmax(smlnum, eps * anorm);
    }

    i32 np1 = n + 1;
    f64 vec[4], x[4];

    if (notrna) {
        i32 lnext = 0;
        for (i32 l = 0; l < n; ) {
            if (l < lnext) {
                l++;
                continue;
            }
            i32 l1 = l;
            i32 l2 = l;
            if (l < n - 1) {
                if (a[(l+1) + l*lda] != ZERO) {
                    l2 = l + 1;
                }
                lnext = l2 + 1;
            } else {
                lnext = n;
            }

            dwork[l1] = ZERO;
            dwork[n + l1] = ZERO;

            i32 len = l1;
            f64 alpha = ONE;
            f64 beta = ZERO;
            i32 one = 1;
            if (len > 0) {
                SLC_DSYMV("Lower", &len, &alpha, c, &ldc, &a[0 + l1*lda], &one, &beta, dwork, &one);
                SLC_DSYMV("Lower", &len, &alpha, c, &ldc, &a[0 + l2*lda], &one, &beta, &dwork[np1-1], &one);
            }

            i32 knext = l;
            for (i32 k = l; k < n; ) {
                if (k < knext) {
                    k++;
                    continue;
                }
                i32 k1 = k;
                i32 k2 = k;
                if (k < n - 1) {
                    if (a[(k+1) + k*lda] != ZERO) {
                        k2 = k + 1;
                    }
                    knext = k2 + 1;
                } else {
                    knext = n;
                }

                f64 scaloc = ONE;
                f64 xnorm;
                i32 ierr;

                if (l1 == l2 && k1 == k2) {
                    f64 dot1 = ZERO;
                    for (i32 i = 0; i < l1; i++) {
                        dot1 += c[k1 + i*ldc] * a[i + l1*lda];
                    }
                    dwork[k1] = dot1;

                    f64 sum1 = ZERO;
                    for (i32 i = 0; i <= k1; i++) {
                        sum1 += a[i + k1*lda] * dwork[i];
                    }
                    f64 sum2 = ZERO;
                    for (i32 i = 0; i < k1; i++) {
                        sum2 += a[i + k1*lda] * c[i + l1*ldc];
                    }

                    vec[0] = c[k1 + l1*ldc] - (sum1 + a[l1 + l1*lda] * sum2);

                    f64 a11 = a[k1 + k1*lda] * a[l1 + l1*lda] - ONE;
                    f64 da11 = fabs(a11);
                    if (da11 <= smin) {
                        a11 = smin;
                        da11 = smin;
                        *info = 1;
                    }
                    f64 db = fabs(vec[0]);
                    if (da11 < ONE && db > ONE) {
                        if (db > bignum * da11) {
                            scaloc = ONE / db;
                        }
                    }
                    x[0] = (vec[0] * scaloc) / a11;

                    if (scaloc != ONE) {
                        for (i32 j = 0; j < n; j++) {
                            i32 nn_loc = n;
                            SLC_DSCAL(&nn_loc, &scaloc, &c[0 + j*ldc], &one);
                        }
                        SLC_DSCAL(&nn, &scaloc, dwork, &one);
                        *scale = (*scale) * scaloc;
                    }
                    c[k1 + l1*ldc] = x[0];
                    if (k1 != l1) {
                        c[l1 + k1*ldc] = x[0];
                    }
                } else if (l1 == l2 && k1 != k2) {
                    f64 dot1 = ZERO, dot2 = ZERO;
                    for (i32 i = 0; i < l1; i++) {
                        dot1 += c[k1 + i*ldc] * a[i + l1*lda];
                        dot2 += c[k2 + i*ldc] * a[i + l1*lda];
                    }
                    dwork[k1] = dot1;
                    dwork[k2] = dot2;

                    f64 sum1 = ZERO, sum2 = ZERO;
                    for (i32 i = 0; i <= k2; i++) {
                        sum1 += a[i + k1*lda] * dwork[i];
                        sum2 += a[i + k2*lda] * dwork[i];
                    }
                    f64 t1 = ZERO, t2 = ZERO;
                    for (i32 i = 0; i < k1; i++) {
                        t1 += a[i + k1*lda] * c[i + l1*ldc];
                        t2 += a[i + k2*lda] * c[i + l1*ldc];
                    }

                    vec[0] = c[k1 + l1*ldc] - (sum1 + a[l1 + l1*lda] * t1);
                    vec[1] = c[k2 + l1*ldc] - (sum2 + a[l1 + l1*lda] * t2);

                    f64 al11 = a[l1 + l1*lda];
                    SLC_DLALN2(&ilupper, (i32[]){2}, (i32[]){1}, &smin, &al11,
                               &a[k1 + k1*lda], &lda, (f64[]){ONE}, (f64[]){ONE},
                               vec, (i32[]){2}, (f64[]){ONE}, (f64[]){ZERO},
                               x, (i32[]){2}, &scaloc, &xnorm, &ierr);
                    if (ierr != 0) *info = 1;

                    if (scaloc != ONE) {
                        for (i32 j = 0; j < n; j++) {
                            i32 nn_loc = n;
                            SLC_DSCAL(&nn_loc, &scaloc, &c[0 + j*ldc], &one);
                        }
                        SLC_DSCAL(&nn, &scaloc, dwork, &one);
                        *scale = (*scale) * scaloc;
                    }
                    c[k1 + l1*ldc] = x[0];
                    c[k2 + l1*ldc] = x[1];
                    c[l1 + k1*ldc] = x[0];
                    c[l1 + k2*ldc] = x[1];
                } else if (l1 != l2 && k1 == k2) {
                    f64 dot1 = ZERO, dot2 = ZERO;
                    for (i32 i = 0; i < l1; i++) {
                        dot1 += c[k1 + i*ldc] * a[i + l1*lda];
                        dot2 += c[k1 + i*ldc] * a[i + l2*lda];
                    }
                    dwork[k1] = dot1;
                    dwork[n + k1] = dot2;

                    f64 p11 = ZERO, p12 = ZERO;
                    for (i32 i = 0; i < k1; i++) {
                        p11 += a[i + k1*lda] * c[i + l1*ldc];
                        p12 += a[i + k1*lda] * c[i + l2*ldc];
                    }

                    f64 sum1 = ZERO, sum2 = ZERO;
                    for (i32 i = 0; i <= k1; i++) {
                        sum1 += a[i + k1*lda] * dwork[i];
                        sum2 += a[i + k1*lda] * dwork[n + i];
                    }

                    vec[0] = c[k1 + l1*ldc] - (sum1 + p11*a[l1 + l1*lda] + p12*a[l2 + l1*lda]);
                    vec[1] = c[k1 + l2*ldc] - (sum2 + p11*a[l1 + l2*lda] + p12*a[l2 + l2*lda]);

                    f64 akk = a[k1 + k1*lda];
                    SLC_DLALN2(&ilupper, (i32[]){2}, (i32[]){1}, &smin, &akk,
                               &a[l1 + l1*lda], &lda, (f64[]){ONE}, (f64[]){ONE},
                               vec, (i32[]){2}, (f64[]){ONE}, (f64[]){ZERO},
                               x, (i32[]){2}, &scaloc, &xnorm, &ierr);
                    if (ierr != 0) *info = 1;

                    if (scaloc != ONE) {
                        for (i32 j = 0; j < n; j++) {
                            i32 nn_loc = n;
                            SLC_DSCAL(&nn_loc, &scaloc, &c[0 + j*ldc], &one);
                        }
                        SLC_DSCAL(&nn, &scaloc, dwork, &one);
                        i32 nn_loc = n;
                        SLC_DSCAL(&nn_loc, &scaloc, &dwork[np1-1], &one);
                        *scale = (*scale) * scaloc;
                    }
                    c[k1 + l1*ldc] = x[0];
                    c[k1 + l2*ldc] = x[1];
                    c[l1 + k1*ldc] = x[0];
                    c[l2 + k1*ldc] = x[1];
                } else {
                    f64 dot1 = ZERO, dot2 = ZERO, dot3 = ZERO, dot4 = ZERO;
                    for (i32 i = 0; i < l1; i++) {
                        dot1 += c[k1 + i*ldc] * a[i + l1*lda];
                        dot2 += c[k2 + i*ldc] * a[i + l1*lda];
                        dot3 += c[k1 + i*ldc] * a[i + l2*lda];
                        dot4 += c[k2 + i*ldc] * a[i + l2*lda];
                    }
                    dwork[k1] = dot1;
                    dwork[k2] = dot2;
                    dwork[n + k1] = dot3;
                    dwork[n + k2] = dot4;

                    f64 p11 = ZERO, p12 = ZERO, p21 = ZERO, p22 = ZERO;
                    for (i32 i = 0; i < k1; i++) {
                        p11 += a[i + k1*lda] * c[i + l1*ldc];
                        p12 += a[i + k1*lda] * c[i + l2*ldc];
                        p21 += a[i + k2*lda] * c[i + l1*ldc];
                        p22 += a[i + k2*lda] * c[i + l2*ldc];
                    }

                    f64 sum1 = ZERO, sum2 = ZERO, sum3 = ZERO, sum4 = ZERO;
                    for (i32 i = 0; i <= k2; i++) {
                        sum1 += a[i + k1*lda] * dwork[i];
                        sum2 += a[i + k1*lda] * dwork[n + i];
                        sum3 += a[i + k2*lda] * dwork[i];
                        sum4 += a[i + k2*lda] * dwork[n + i];
                    }

                    vec[0] = c[k1 + l1*ldc] - (sum1 + p11*a[l1 + l1*lda] + p12*a[l2 + l1*lda]);
                    vec[2] = c[k1 + l2*ldc] - (sum2 + p11*a[l1 + l2*lda] + p12*a[l2 + l2*lda]);
                    vec[1] = c[k2 + l1*ldc] - (sum3 + p21*a[l1 + l1*lda] + p22*a[l2 + l1*lda]);
                    vec[3] = c[k2 + l2*ldc] - (sum4 + p21*a[l1 + l2*lda] + p22*a[l2 + l2*lda]);

                    if (k1 == l1) {
                        sb03mv(false, lupper, &a[k1 + k1*lda], lda, vec, 2, &scaloc, x, 2, &xnorm, &ierr);
                        if (lupper) {
                            x[1] = x[2];
                        } else {
                            x[2] = x[1];
                        }
                    } else {
                        sb04px(true, false, -1, 2, 2, &a[k1 + k1*lda], lda,
                               &a[l1 + l1*lda], lda, vec, 2, &scaloc, x, 2, &xnorm, &ierr);
                    }
                    if (ierr != 0) *info = 1;

                    if (scaloc != ONE) {
                        for (i32 j = 0; j < n; j++) {
                            i32 nn_loc = n;
                            SLC_DSCAL(&nn_loc, &scaloc, &c[0 + j*ldc], &one);
                        }
                        SLC_DSCAL(&nn, &scaloc, dwork, &one);
                        i32 nn_loc = n;
                        SLC_DSCAL(&nn_loc, &scaloc, &dwork[np1-1], &one);
                        *scale = (*scale) * scaloc;
                    }
                    c[k1 + l1*ldc] = x[0];
                    c[k1 + l2*ldc] = x[2];
                    c[k2 + l1*ldc] = x[1];
                    c[k2 + l2*ldc] = x[3];
                    if (k1 != l1) {
                        c[l1 + k1*ldc] = x[0];
                        c[l2 + k1*ldc] = x[2];
                        c[l1 + k2*ldc] = x[1];
                        c[l2 + k2*ldc] = x[3];
                    }
                }
                k = knext;
            }
            l = lnext;
        }
    } else {
        i32 lnext = n - 1;
        for (i32 l = n - 1; l >= 0; ) {
            if (l > lnext) {
                l--;
                continue;
            }
            i32 l1 = l;
            i32 l2 = l;
            if (l > 0) {
                if (a[l + (l-1)*lda] != ZERO) {
                    l1 = l - 1;
                    dwork[l1] = ZERO;
                    dwork[n + l1] = ZERO;
                }
                lnext = l1 - 1;
            } else {
                lnext = -1;
            }
            i32 minl1n = (l1 + 1 < n) ? l1 + 1 : n;
            i32 minl2n = (l2 + 1 < n) ? l2 + 1 : n;

            if (l2 < n - 1) {
                i32 len = n - l2 - 1;
                f64 alpha = ONE;
                f64 beta = ZERO;
                i32 one = 1;
                SLC_DSYMV("Upper", &len, &alpha, &c[(l2+1) + (l2+1)*ldc], &ldc,
                          &a[l1 + (l2+1)*lda], &lda, &beta, &dwork[l2+1], &one);
                SLC_DSYMV("Upper", &len, &alpha, &c[(l2+1) + (l2+1)*ldc], &ldc,
                          &a[l2 + (l2+1)*lda], &lda, &beta, &dwork[np1 + l2], &one);
            }

            i32 knext = l;
            for (i32 k = l; k >= 0; ) {
                if (k > knext) {
                    k--;
                    continue;
                }
                i32 k1 = k;
                i32 k2 = k;
                if (k > 0) {
                    if (a[k + (k-1)*lda] != ZERO) {
                        k1 = k - 1;
                    }
                    knext = k1 - 1;
                } else {
                    knext = -1;
                }
                i32 mink1n = (k1 + 1 < n) ? k1 + 1 : n;
                i32 mink2n = (k2 + 1 < n) ? k2 + 1 : n;

                f64 scaloc = ONE;
                f64 xnorm;
                i32 ierr;
                i32 one = 1;

                if (l1 == l2 && k1 == k2) {
                    f64 dot = ZERO;
                    for (i32 i = minl1n; i < n; i++) {
                        dot += c[k1 + i*ldc] * a[l1 + i*lda];
                    }
                    dwork[k1] = dot;

                    f64 sum1 = ZERO;
                    for (i32 i = k1; i < n; i++) {
                        sum1 += a[k1 + i*lda] * dwork[i];
                    }
                    f64 sum2 = ZERO;
                    for (i32 i = mink1n; i < n; i++) {
                        sum2 += a[k1 + i*lda] * c[i + l1*ldc];
                    }

                    vec[0] = c[k1 + l1*ldc] - (sum1 + sum2 * a[l1 + l1*lda]);

                    f64 a11 = a[k1 + k1*lda] * a[l1 + l1*lda] - ONE;
                    f64 da11 = fabs(a11);
                    if (da11 <= smin) {
                        a11 = smin;
                        da11 = smin;
                        *info = 1;
                    }
                    f64 db = fabs(vec[0]);
                    if (da11 < ONE && db > ONE) {
                        if (db > bignum * da11) {
                            scaloc = ONE / db;
                        }
                    }
                    x[0] = (vec[0] * scaloc) / a11;

                    if (scaloc != ONE) {
                        for (i32 j = 0; j < n; j++) {
                            i32 nn_loc = n;
                            SLC_DSCAL(&nn_loc, &scaloc, &c[0 + j*ldc], &one);
                        }
                        SLC_DSCAL(&nn, &scaloc, dwork, &one);
                        *scale = (*scale) * scaloc;
                    }
                    c[k1 + l1*ldc] = x[0];
                    if (k1 != l1) {
                        c[l1 + k1*ldc] = x[0];
                    }
                } else if (l1 == l2 && k1 != k2) {
                    f64 dot1 = ZERO, dot2 = ZERO;
                    for (i32 i = minl1n; i < n; i++) {
                        dot1 += c[k1 + i*ldc] * a[l1 + i*lda];
                        dot2 += c[k2 + i*ldc] * a[l1 + i*lda];
                    }
                    dwork[k1] = dot1;
                    dwork[k2] = dot2;

                    f64 sum1 = ZERO, sum2 = ZERO;
                    for (i32 i = k1; i <= k2; i++) {
                        sum1 += a[k1 + i*lda] * dwork[i];
                        sum2 += a[k2 + i*lda] * dwork[i];
                    }
                    for (i32 i = k1 + 1; i < n; i++) {
                        sum1 += a[k1 + i*lda] * dwork[i];
                    }
                    for (i32 i = k2 + 1; i < n; i++) {
                        sum2 += a[k2 + i*lda] * dwork[i];
                    }
                    f64 t1 = ZERO, t2 = ZERO;
                    for (i32 i = mink2n; i < n; i++) {
                        t1 += a[k1 + i*lda] * c[i + l1*ldc];
                        t2 += a[k2 + i*lda] * c[i + l1*ldc];
                    }

                    vec[0] = c[k1 + l1*ldc] - (sum1 + t1 * a[l1 + l1*lda]);
                    vec[1] = c[k2 + l1*ldc] - (sum2 + t2 * a[l1 + l1*lda]);

                    i32 ibfalse = 0;
                    f64 al11 = a[l1 + l1*lda];
                    SLC_DLALN2(&ibfalse, (i32[]){2}, (i32[]){1}, &smin, &al11,
                               &a[k1 + k1*lda], &lda, (f64[]){ONE}, (f64[]){ONE},
                               vec, (i32[]){2}, (f64[]){ONE}, (f64[]){ZERO},
                               x, (i32[]){2}, &scaloc, &xnorm, &ierr);
                    if (ierr != 0) *info = 1;

                    if (scaloc != ONE) {
                        for (i32 j = 0; j < n; j++) {
                            i32 nn_loc = n;
                            SLC_DSCAL(&nn_loc, &scaloc, &c[0 + j*ldc], &one);
                        }
                        SLC_DSCAL(&nn, &scaloc, dwork, &one);
                        *scale = (*scale) * scaloc;
                    }
                    c[k1 + l1*ldc] = x[0];
                    c[k2 + l1*ldc] = x[1];
                    c[l1 + k1*ldc] = x[0];
                    c[l1 + k2*ldc] = x[1];
                } else if (l1 != l2 && k1 == k2) {
                    f64 dot1 = ZERO, dot2 = ZERO;
                    for (i32 i = minl2n; i < n; i++) {
                        dot1 += c[k1 + i*ldc] * a[l1 + i*lda];
                        dot2 += c[k1 + i*ldc] * a[l2 + i*lda];
                    }
                    dwork[k1] = dot1;
                    dwork[n + k1] = dot2;

                    f64 p11 = ZERO, p12 = ZERO;
                    for (i32 i = mink1n; i < n; i++) {
                        p11 += a[k1 + i*lda] * c[i + l1*ldc];
                        p12 += a[k1 + i*lda] * c[i + l2*ldc];
                    }

                    f64 sum1 = ZERO, sum2 = ZERO;
                    for (i32 i = k1; i < n; i++) {
                        sum1 += a[k1 + i*lda] * dwork[i];
                        sum2 += a[k1 + i*lda] * dwork[n + i];
                    }

                    vec[0] = c[k1 + l1*ldc] - (sum1 + p11*a[l1 + l1*lda] + p12*a[l1 + l2*lda]);
                    vec[1] = c[k1 + l2*ldc] - (sum2 + p11*a[l2 + l1*lda] + p12*a[l2 + l2*lda]);

                    i32 ibfalse = 0;
                    f64 akk = a[k1 + k1*lda];
                    SLC_DLALN2(&ibfalse, (i32[]){2}, (i32[]){1}, &smin, &akk,
                               &a[l1 + l1*lda], &lda, (f64[]){ONE}, (f64[]){ONE},
                               vec, (i32[]){2}, (f64[]){ONE}, (f64[]){ZERO},
                               x, (i32[]){2}, &scaloc, &xnorm, &ierr);
                    if (ierr != 0) *info = 1;

                    if (scaloc != ONE) {
                        for (i32 j = 0; j < n; j++) {
                            i32 nn_loc = n;
                            SLC_DSCAL(&nn_loc, &scaloc, &c[0 + j*ldc], &one);
                        }
                        SLC_DSCAL(&nn, &scaloc, dwork, &one);
                        i32 nn_loc = n;
                        SLC_DSCAL(&nn_loc, &scaloc, &dwork[np1-1], &one);
                        *scale = (*scale) * scaloc;
                    }
                    c[k1 + l1*ldc] = x[0];
                    c[k1 + l2*ldc] = x[1];
                    c[l1 + k1*ldc] = x[0];
                    c[l2 + k1*ldc] = x[1];
                } else {
                    f64 dot1 = ZERO, dot2 = ZERO, dot3 = ZERO, dot4 = ZERO;
                    for (i32 i = minl2n; i < n; i++) {
                        dot1 += c[k1 + i*ldc] * a[l1 + i*lda];
                        dot2 += c[k2 + i*ldc] * a[l1 + i*lda];
                        dot3 += c[k1 + i*ldc] * a[l2 + i*lda];
                        dot4 += c[k2 + i*ldc] * a[l2 + i*lda];
                    }
                    dwork[k1] = dot1;
                    dwork[k2] = dot2;
                    dwork[n + k1] = dot3;
                    dwork[n + k2] = dot4;

                    f64 p11 = ZERO, p12 = ZERO, p21 = ZERO, p22 = ZERO;
                    for (i32 i = mink2n; i < n; i++) {
                        p11 += a[k1 + i*lda] * c[i + l1*ldc];
                        p12 += a[k1 + i*lda] * c[i + l2*ldc];
                        p21 += a[k2 + i*lda] * c[i + l1*ldc];
                        p22 += a[k2 + i*lda] * c[i + l2*ldc];
                    }

                    f64 sum1 = ZERO, sum2 = ZERO, sum3 = ZERO, sum4 = ZERO;
                    for (i32 i = k1; i <= k2; i++) {
                        sum1 += a[k1 + i*lda] * dwork[i];
                        sum2 += a[k1 + i*lda] * dwork[n + i];
                    }
                    for (i32 i = k1; i <= k2; i++) {
                        sum3 += a[k2 + i*lda] * dwork[i];
                        sum4 += a[k2 + i*lda] * dwork[n + i];
                    }

                    vec[0] = c[k1 + l1*ldc] - (sum1 + p11*a[l1 + l1*lda] + p12*a[l1 + l2*lda]);
                    vec[2] = c[k1 + l2*ldc] - (sum2 + p11*a[l2 + l1*lda] + p12*a[l2 + l2*lda]);
                    vec[1] = c[k2 + l1*ldc] - (sum3 + p21*a[l1 + l1*lda] + p22*a[l1 + l2*lda]);
                    vec[3] = c[k2 + l2*ldc] - (sum4 + p21*a[l2 + l1*lda] + p22*a[l2 + l2*lda]);

                    if (k1 == l1) {
                        sb03mv(true, lupper, &a[k1 + k1*lda], lda, vec, 2, &scaloc, x, 2, &xnorm, &ierr);
                        if (lupper) {
                            x[1] = x[2];
                        } else {
                            x[2] = x[1];
                        }
                    } else {
                        sb04px(false, true, -1, 2, 2, &a[k1 + k1*lda], lda,
                               &a[l1 + l1*lda], lda, vec, 2, &scaloc, x, 2, &xnorm, &ierr);
                    }
                    if (ierr != 0) *info = 1;

                    if (scaloc != ONE) {
                        for (i32 j = 0; j < n; j++) {
                            i32 nn_loc = n;
                            SLC_DSCAL(&nn_loc, &scaloc, &c[0 + j*ldc], &one);
                        }
                        SLC_DSCAL(&nn, &scaloc, dwork, &one);
                        i32 nn_loc = n;
                        SLC_DSCAL(&nn_loc, &scaloc, &dwork[np1-1], &one);
                        *scale = (*scale) * scaloc;
                    }
                    c[k1 + l1*ldc] = x[0];
                    c[k1 + l2*ldc] = x[2];
                    c[k2 + l1*ldc] = x[1];
                    c[k2 + l2*ldc] = x[3];
                    if (k1 != l1) {
                        c[l1 + k1*ldc] = x[0];
                        c[l2 + k1*ldc] = x[2];
                        c[l1 + k2*ldc] = x[1];
                        c[l2 + k2*ldc] = x[3];
                    }
                }
                k = knext;
            }
            l = lnext;
        }
    }
}
