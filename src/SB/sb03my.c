/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB03MY - Solve continuous-time Lyapunov equation for Schur form matrix
 *
 * Solves: op(A)' * X + X * op(A) = scale * C
 * where A is upper quasi-triangular (Schur form), C is symmetric.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sb03my(
    const char* trana,
    const i32 n,
    const f64* a,
    const i32 lda,
    f64* c,
    const i32 ldc,
    f64* scale,
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
        SLC_XERBLA("SB03MY", &neginfo);
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

    f64 dum[1];
    i32 nn = n;
    f64 anorm = SLC_DLANHS("Max", &nn, a, &lda, dum);
    f64 smin = fmax(smlnum, eps * anorm);

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
                i32 one = 1;

                if (l1 == l2 && k1 == k2) {
                    f64 dot1 = ZERO;
                    for (i32 i = 0; i < k1; i++) {
                        dot1 += a[i + k1*lda] * c[i + l1*ldc];
                    }
                    f64 dot2 = ZERO;
                    for (i32 i = 0; i < l1; i++) {
                        dot2 += c[k1 + i*ldc] * a[i + l1*lda];
                    }

                    vec[0] = c[k1 + l1*ldc] - (dot1 + dot2);

                    f64 a11 = a[k1 + k1*lda] + a[l1 + l1*lda];
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
                        *scale = (*scale) * scaloc;
                    }
                    c[k1 + l1*ldc] = x[0];
                    if (k1 != l1) {
                        c[l1 + k1*ldc] = x[0];
                    }
                } else if (l1 == l2 && k1 != k2) {
                    f64 dot1 = ZERO, dot2 = ZERO;
                    for (i32 i = 0; i < k1; i++) {
                        dot1 += a[i + k1*lda] * c[i + l1*ldc];
                        dot2 += a[i + k2*lda] * c[i + l1*ldc];
                    }
                    f64 dot3 = ZERO, dot4 = ZERO;
                    for (i32 i = 0; i < l1; i++) {
                        dot3 += c[k1 + i*ldc] * a[i + l1*lda];
                        dot4 += c[k2 + i*ldc] * a[i + l1*lda];
                    }

                    vec[0] = c[k1 + l1*ldc] - (dot1 + dot3);
                    vec[1] = c[k2 + l1*ldc] - (dot2 + dot4);

                    f64 negall = -a[l1 + l1*lda];
                    SLC_DLALN2(&ilupper, (i32[]){2}, (i32[]){1}, &smin, (f64[]){ONE},
                               &a[k1 + k1*lda], &lda, (f64[]){ONE}, (f64[]){ONE},
                               vec, (i32[]){2}, &negall, (f64[]){ZERO},
                               x, (i32[]){2}, &scaloc, &xnorm, &ierr);
                    if (ierr != 0) *info = 1;

                    if (scaloc != ONE) {
                        for (i32 j = 0; j < n; j++) {
                            i32 nn_loc = n;
                            SLC_DSCAL(&nn_loc, &scaloc, &c[0 + j*ldc], &one);
                        }
                        *scale = (*scale) * scaloc;
                    }
                    c[k1 + l1*ldc] = x[0];
                    c[k2 + l1*ldc] = x[1];
                    c[l1 + k1*ldc] = x[0];
                    c[l1 + k2*ldc] = x[1];
                } else if (l1 != l2 && k1 == k2) {
                    f64 dot1 = ZERO, dot2 = ZERO;
                    for (i32 i = 0; i < k1; i++) {
                        dot1 += a[i + k1*lda] * c[i + l1*ldc];
                        dot2 += a[i + k1*lda] * c[i + l2*ldc];
                    }
                    f64 dot3 = ZERO, dot4 = ZERO;
                    for (i32 i = 0; i < l1; i++) {
                        dot3 += c[k1 + i*ldc] * a[i + l1*lda];
                        dot4 += c[k1 + i*ldc] * a[i + l2*lda];
                    }

                    vec[0] = c[k1 + l1*ldc] - (dot1 + dot3);
                    vec[1] = c[k1 + l2*ldc] - (dot2 + dot4);

                    f64 negakk = -a[k1 + k1*lda];
                    SLC_DLALN2(&ilupper, (i32[]){2}, (i32[]){1}, &smin, (f64[]){ONE},
                               &a[l1 + l1*lda], &lda, (f64[]){ONE}, (f64[]){ONE},
                               vec, (i32[]){2}, &negakk, (f64[]){ZERO},
                               x, (i32[]){2}, &scaloc, &xnorm, &ierr);
                    if (ierr != 0) *info = 1;

                    if (scaloc != ONE) {
                        for (i32 j = 0; j < n; j++) {
                            i32 nn_loc = n;
                            SLC_DSCAL(&nn_loc, &scaloc, &c[0 + j*ldc], &one);
                        }
                        *scale = (*scale) * scaloc;
                    }
                    c[k1 + l1*ldc] = x[0];
                    c[k1 + l2*ldc] = x[1];
                    c[l1 + k1*ldc] = x[0];
                    c[l2 + k1*ldc] = x[1];
                } else {
                    f64 dot1 = ZERO, dot2 = ZERO, dot3 = ZERO, dot4 = ZERO;
                    for (i32 i = 0; i < k1; i++) {
                        dot1 += a[i + k1*lda] * c[i + l1*ldc];
                        dot2 += a[i + k1*lda] * c[i + l2*ldc];
                        dot3 += a[i + k2*lda] * c[i + l1*ldc];
                        dot4 += a[i + k2*lda] * c[i + l2*ldc];
                    }
                    f64 dot5 = ZERO, dot6 = ZERO, dot7 = ZERO, dot8 = ZERO;
                    for (i32 i = 0; i < l1; i++) {
                        dot5 += c[k1 + i*ldc] * a[i + l1*lda];
                        dot6 += c[k1 + i*ldc] * a[i + l2*lda];
                        dot7 += c[k2 + i*ldc] * a[i + l1*lda];
                        dot8 += c[k2 + i*ldc] * a[i + l2*lda];
                    }

                    vec[0] = c[k1 + l1*ldc] - (dot1 + dot5);
                    vec[2] = c[k1 + l2*ldc] - (dot2 + dot6);
                    vec[1] = c[k2 + l1*ldc] - (dot3 + dot7);
                    vec[3] = c[k2 + l2*ldc] - (dot4 + dot8);

                    if (k1 == l1) {
                        sb03mw(false, lupper, &a[k1 + k1*lda], lda, vec, 2, &scaloc, x, 2, &xnorm, &ierr);
                        if (lupper) {
                            x[1] = x[2];
                        } else {
                            x[2] = x[1];
                        }
                    } else {
                        SLC_DLASY2(&ilupper, (i32[]){0}, (i32[]){1}, (i32[]){2}, (i32[]){2},
                                   &a[k1 + k1*lda], &lda, &a[l1 + l1*lda], &lda,
                                   vec, (i32[]){2}, &scaloc, x, (i32[]){2}, &xnorm, &ierr);
                    }
                    if (ierr != 0) *info = 1;

                    if (scaloc != ONE) {
                        for (i32 j = 0; j < n; j++) {
                            i32 nn_loc = n;
                            SLC_DSCAL(&nn_loc, &scaloc, &c[0 + j*ldc], &one);
                        }
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
                }
                lnext = l1 - 1;
            } else {
                lnext = -1;
            }
            i32 minl1n = (l1 + 1 < n) ? l1 + 1 : n;
            i32 minl2n = (l2 + 1 < n) ? l2 + 1 : n;

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
                    f64 dot1 = ZERO;
                    for (i32 i = mink1n; i < n; i++) {
                        dot1 += a[k1 + i*lda] * c[i + l1*ldc];
                    }
                    f64 dot2 = ZERO;
                    for (i32 i = minl1n; i < n; i++) {
                        dot2 += c[k1 + i*ldc] * a[l1 + i*lda];
                    }

                    vec[0] = c[k1 + l1*ldc] - (dot1 + dot2);

                    f64 a11 = a[k1 + k1*lda] + a[l1 + l1*lda];
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
                        *scale = (*scale) * scaloc;
                    }
                    c[k1 + l1*ldc] = x[0];
                    if (k1 != l1) {
                        c[l1 + k1*ldc] = x[0];
                    }
                } else if (l1 == l2 && k1 != k2) {
                    f64 dot1 = ZERO, dot2 = ZERO;
                    for (i32 i = mink2n; i < n; i++) {
                        dot1 += a[k1 + i*lda] * c[i + l1*ldc];
                        dot2 += a[k2 + i*lda] * c[i + l1*ldc];
                    }
                    f64 dot3 = ZERO, dot4 = ZERO;
                    for (i32 i = minl2n; i < n; i++) {
                        dot3 += c[k1 + i*ldc] * a[l1 + i*lda];
                        dot4 += c[k2 + i*ldc] * a[l1 + i*lda];
                    }

                    vec[0] = c[k1 + l1*ldc] - (dot1 + dot3);
                    vec[1] = c[k2 + l1*ldc] - (dot2 + dot4);

                    i32 ibfalse = 0;
                    f64 negall = -a[l1 + l1*lda];
                    SLC_DLALN2(&ibfalse, (i32[]){2}, (i32[]){1}, &smin, (f64[]){ONE},
                               &a[k1 + k1*lda], &lda, (f64[]){ONE}, (f64[]){ONE},
                               vec, (i32[]){2}, &negall, (f64[]){ZERO},
                               x, (i32[]){2}, &scaloc, &xnorm, &ierr);
                    if (ierr != 0) *info = 1;

                    if (scaloc != ONE) {
                        for (i32 j = 0; j < n; j++) {
                            i32 nn_loc = n;
                            SLC_DSCAL(&nn_loc, &scaloc, &c[0 + j*ldc], &one);
                        }
                        *scale = (*scale) * scaloc;
                    }
                    c[k1 + l1*ldc] = x[0];
                    c[k2 + l1*ldc] = x[1];
                    c[l1 + k1*ldc] = x[0];
                    c[l1 + k2*ldc] = x[1];
                } else if (l1 != l2 && k1 == k2) {
                    f64 dot1 = ZERO, dot2 = ZERO;
                    for (i32 i = mink1n; i < n; i++) {
                        dot1 += a[k1 + i*lda] * c[i + l1*ldc];
                        dot2 += a[k1 + i*lda] * c[i + l2*ldc];
                    }
                    f64 dot3 = ZERO, dot4 = ZERO;
                    for (i32 i = minl2n; i < n; i++) {
                        dot3 += c[k1 + i*ldc] * a[l1 + i*lda];
                        dot4 += c[k1 + i*ldc] * a[l2 + i*lda];
                    }

                    vec[0] = c[k1 + l1*ldc] - (dot1 + dot3);
                    vec[1] = c[k1 + l2*ldc] - (dot2 + dot4);

                    i32 ibfalse = 0;
                    f64 negakk = -a[k1 + k1*lda];
                    SLC_DLALN2(&ibfalse, (i32[]){2}, (i32[]){1}, &smin, (f64[]){ONE},
                               &a[l1 + l1*lda], &lda, (f64[]){ONE}, (f64[]){ONE},
                               vec, (i32[]){2}, &negakk, (f64[]){ZERO},
                               x, (i32[]){2}, &scaloc, &xnorm, &ierr);
                    if (ierr != 0) *info = 1;

                    if (scaloc != ONE) {
                        for (i32 j = 0; j < n; j++) {
                            i32 nn_loc = n;
                            SLC_DSCAL(&nn_loc, &scaloc, &c[0 + j*ldc], &one);
                        }
                        *scale = (*scale) * scaloc;
                    }
                    c[k1 + l1*ldc] = x[0];
                    c[k1 + l2*ldc] = x[1];
                    c[l1 + k1*ldc] = x[0];
                    c[l2 + k1*ldc] = x[1];
                } else {
                    f64 dot1 = ZERO, dot2 = ZERO, dot3 = ZERO, dot4 = ZERO;
                    for (i32 i = mink2n; i < n; i++) {
                        dot1 += a[k1 + i*lda] * c[i + l1*ldc];
                        dot2 += a[k1 + i*lda] * c[i + l2*ldc];
                        dot3 += a[k2 + i*lda] * c[i + l1*ldc];
                        dot4 += a[k2 + i*lda] * c[i + l2*ldc];
                    }
                    f64 dot5 = ZERO, dot6 = ZERO, dot7 = ZERO, dot8 = ZERO;
                    for (i32 i = minl2n; i < n; i++) {
                        dot5 += c[k1 + i*ldc] * a[l1 + i*lda];
                        dot6 += c[k1 + i*ldc] * a[l2 + i*lda];
                        dot7 += c[k2 + i*ldc] * a[l1 + i*lda];
                        dot8 += c[k2 + i*ldc] * a[l2 + i*lda];
                    }

                    vec[0] = c[k1 + l1*ldc] - (dot1 + dot5);
                    vec[2] = c[k1 + l2*ldc] - (dot2 + dot6);
                    vec[1] = c[k2 + l1*ldc] - (dot3 + dot7);
                    vec[3] = c[k2 + l2*ldc] - (dot4 + dot8);

                    if (k1 == l1) {
                        sb03mw(true, lupper, &a[k1 + k1*lda], lda, vec, 2, &scaloc, x, 2, &xnorm, &ierr);
                        if (lupper) {
                            x[1] = x[2];
                        } else {
                            x[2] = x[1];
                        }
                    } else {
                        i32 ibfalse = 0;
                        i32 ibtrue = 1;
                        SLC_DLASY2(&ibfalse, &ibtrue, (i32[]){1}, (i32[]){2}, (i32[]){2},
                                   &a[k1 + k1*lda], &lda, &a[l1 + l1*lda], &lda,
                                   vec, (i32[]){2}, &scaloc, x, (i32[]){2}, &xnorm, &ierr);
                    }
                    if (ierr != 0) *info = 1;

                    if (scaloc != ONE) {
                        for (i32 j = 0; j < n; j++) {
                            i32 nn_loc = n;
                            SLC_DSCAL(&nn_loc, &scaloc, &c[0 + j*ldc], &one);
                        }
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
